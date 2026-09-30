# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side trims for the eager GDN core of mixed (prefill + MTP spec-decode)
steps of the Qwen GDN layer.

In a mixed step every GDN layer runs its core as an eager splitting op, and
that stretch is host-bound: per layer the GPU work is short while the Python
around each launch (predicates, slices, small tensor ops, launch binding) is
repeated for every GDN layer. All GDN layers of one KV-cache group share one
GDNAttentionMetadata object per step, and each layer only touches its own
conv / SSM state, which the features below exploit.

GGM_OG2=1:
  VLLM_GDN_GROUP_MATERIALIZE=1 (default; needs GDN_STATE_COMMIT=1): the
      deferred state commit's materialize_non_spec() commits every GDN layer of
      the KV-cache group in ONE materialize launch (grid z = layers) at the
      first call of the group and returns immediately for the other layers,
      instead of one launch (plus small torch ops) per layer.
  VLLM_GDN_MERGED_ROW_QUANT=1 (default; needs NQF=1): the MXFP8 row quant of
      the GDN out_proj input rows the fused gated RMSNorm did not write (the
      spec-decode rows and the padding rows) runs as ONE launch of a two-range
      variant of the row-quant kernel instead of two launches.

Numerics: bit-exact. A layer's state is only touched by that layer's kernels
and the order of operations on it is unchanged; the batched launches run the
same per-(item, head, layer) / per-row code on the same inputs.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# ----------------------------------------------------------------------------
# gates (read once per process)
# ----------------------------------------------------------------------------
HOST_TRIMS = os.environ.get("GGM_OG2", "0") == "1"
GROUP_MATERIALIZE = (
    HOST_TRIMS and os.environ.get("VLLM_GDN_GROUP_MATERIALIZE", "1") == "1"
)
MERGED_ROW_QUANT = (
    HOST_TRIMS and os.environ.get("VLLM_GDN_MERGED_ROW_QUANT", "1") == "1"
)

TRIM_STATS = {
    "gsc_group_launch": 0,
    "gsc_layers_skipped": 0,
    "gsc_fallback": 0,
    "qrows_merged": 0,
    "qrows_single": 0,
    "qrows_multi": 0,
}
_LOGGED: set = set()
_MODS: dict = {}


def _gsc():
    """The deferred GDN state commit module (imported lazily: it imports this
    module at its top).
    """
    m = _MODS.get("gsc")
    if m is None:
        from vllm.model_executor.layers.mamba.ops import gdn_state_commit as m

        _MODS["gsc"] = m
    return m


def check_config(state_commit: bool, norm_quant_fusion: bool) -> None:
    """Called once when the Qwen GDN layer module is imported (state_commit:
    GDN_STATE_COMMIT=1; norm_quant_fusion: NQF=1).
    """
    if GROUP_MATERIALIZE and state_commit:
        logger.info("GDN state materialize: one launch per KV group enabled")
    if MERGED_ROW_QUANT and norm_quant_fusion:
        logger.info("GDN uncovered-row quant merge enabled")


# ----------------------------------------------------------------------------
# deferred state commit: one materialize launch per KV-cache group
# ----------------------------------------------------------------------------
_GROUP_CACHE: dict = {}


def _group_layers(layer, md):
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    raw = fc.attn_metadata
    if not isinstance(raw, dict):
        return None
    names = tuple(n for n, m in raw.items() if m is md)
    if len(names) <= 1:
        return None
    ent = _GROUP_CACHE.get(names)
    if ent is None:
        mods = [fc.no_compile_layers.get(n) for n in names]
        if any(
            m is None or not hasattr(m, "kv_cache") or not hasattr(m, "A_log")
            for m in mods
        ):
            _GROUP_CACHE[names] = False
            return None
        ent = {"layers": mods, "key": None, "table": None}
        _GROUP_CACHE[names] = ent
    if ent is False:
        return None
    if all(m is not layer for m in ent["layers"]):
        return None
    return ent


def group_materialize_non_spec(layer, md, per_layer) -> None:
    """gdn_state_commit.materialize_non_spec for every GDN layer sharing `md`,
    in one launch at the first call of the group (VLLM_GDN_GROUP_MATERIALIZE).
    `per_layer` is the single-layer version, used when the group cannot be
    resolved. The per-slot inputs (slots, accepted counts, has_initial_state)
    are identical for all layers of the group.
    """
    done = md.__dict__.setdefault("_gsc_done", set())
    if id(layer) in done:
        TRIM_STATS["gsc_layers_skipped"] += 1
        return
    slots = md.non_spec_state_indices_tensor
    items = md.num_prefills + md.num_decodes
    if slots is None or items <= 0:
        done.add(id(layer))
        return
    try:
        ent = _group_layers(layer, md)
    except Exception as e:  # noqa: BLE001 - never break serving; per-layer path
        if "group-lookup" not in _LOGGED:
            _LOGGED.add("group-lookup")
            logger.warning(
                "GDN state materialize: group lookup failed (%r); per-layer "
                "materialize",
                e,
            )
        ent = None
    if ent is None:
        TRIM_STATS["gsc_fallback"] += 1
        return per_layer(layer, md)
    gsc = _gsc()
    layers = ent["layers"]
    for L in layers:
        gsc._layer_check(L)
    key = tuple(L.kv_cache[1].data_ptr() for L in layers)
    if ent["key"] != key:
        ent["table"] = gsc.LayerTable(
            [(L.kv_cache[1], L.A_log, L.dt_bias, 0) for L in layers], slots.device
        )
        ent["H"] = layers[0]._gsc_H
        assert all(L._gsc_H == ent["H"] for L in layers)
        ent["key"] = key
        logger.info_once(
            "GDN state materialize: one launch per KV group (%d GDN layers)",
            len(layers),
        )
    # identical to gdn_state_commit.materialize_non_spec, once for the group
    items = min(items, slots.size(0))
    n_src = getattr(md, "gsc_non_spec_num_accepted", None)
    n = torch.zeros(items, dtype=torch.int32, device=slots.device)
    if n_src is not None and n_src.numel() > 0:
        k = min(items, n_src.numel())
        n[:k] = n_src[:k]
    has_init = md.has_initial_state
    hi = None
    if has_init is not None:
        hi = torch.zeros(items, dtype=torch.bool, device=slots.device)
        k = min(items, has_init.numel())
        hi[:k] = has_init[:k]
    gsc.materialize(0, items, ent["table"], ent["H"], slots[:items], n, has_init=hi)
    done.update(id(L) for L in layers)
    TRIM_STATS["gsc_group_launch"] += 1
