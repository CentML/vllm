# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step plan and host-side trims for the eager GDN core of mixed
(prefill + MTP spec-decode) steps of the Qwen GDN layer.

In a mixed step every GDN layer runs its core as an eager splitting op, and
that stretch is host-bound: per layer the GPU work is short while the Python
around each launch (predicates, slices, small tensor ops, launch binding) is
repeated for every GDN layer. All GDN layers of one KV-cache group share one
GDNAttentionMetadata object per step, and each layer only touches its own
conv / SSM state, which the features below exploit.

GGM=1 (requires GDN_STATE_COMMIT=1): for metadata objects whose step takes
the zero-copy mixed fast path (spec rows on the deferred-commit decode kernel,
prefill rows on the fused conv + FlashInfer / V-split chunked kernel with
in-place state-pool I/O), a plan is built ONCE per metadata object (= once
per KV-cache group per step) and cached on it: every layer-invariant
predicate, slice, contiguous / int32 copy, the CP / V-split decision and the
CUDA conv launch shape. Per layer only the kernel launches remain: the same
kernels with the same arguments in the same order. Also:
  VLLM_GDN_STEP_PLAN_MAT=1 (default): one deferred-commit materialize launch
      for all GDN layers of the group instead of one per layer.
  VLLM_GDN_STEP_PLAN_ZERO=1 (default): one launch zeroing the fresh prefill
      state slots of all those layers (prefill slots are disjoint from spec
      slots and not read before that layer's chunk kernel).
  VLLM_GDN_STEP_PLAN_BUFS=1 (default): the conv outputs (q, k, v, g, beta)
      are allocated once per step and reused by every layer (stream-ordered).
  GGM_VSDIRECT=1: after the V-split adapter compiled its kernel for this
      step's key, the other layers call the cached compiled object directly
      with one workspace per step.
  GGM_LAZY=1: the GDN metadata builder defers the FLA chunk metadata and the
      Triton causal-conv1d metadata, which only the fallback paths read (pure
      functions of CPU query_start_loc tensors); fill_lazy() computes them
      before any fallback path runs.
  GGM_MDREUSE=1: the GDN builder runs once per GDN KV-cache group with
      identical inputs except the block table; the 2nd/3rd group's metadata of
      a step with prefills is derived from the first one (shallow copy, only
      the block-table-derived state indices recomputed as the builder does).
      Reuse is keyed on the identity of the step's input objects.
      GGM_MDCHECK=N: also run the full build for the first N derived builds and
      compare every field (mismatch -> warning, the full build is used).
      VLLM_GDN_STEP_PLAN_MDREUSE_PTR_KEY=1: key the reuse on the (pointer, shape,
      stride, device) of the accepted / draft-count tensors instead of object
      identity (per-group slices of inference tensors are distinct objects, so
      the identity key does not match on the V2 model runner).
  VLLM_GDN_STEP_PLAN_LAZY_IDX=1 (with GGM_LAZY): the spec / non-spec token
      permutation of mixed batches (repeat_interleave + argsort), read only by
      the fallback paths, is deferred as well and computed by fill_lazy().
      VLLM_GDN_STEP_PLAN_LAZY_IDX_CHECK=N: for the first N deferred builds also
      compute the permutation eagerly and compare (mismatch -> eager values).
  VLLM_GDN_STEP_PLAN_LOG=1 (default) / VLLM_GDN_STEP_PLAN_LOG_EVERY=2000:
      log the plan counters every N fast-path plans.
Whatever the plan cannot prove identical (verify mode, gather/scatter CP path,
V-split check mode, permuted spec batches, decode-only steps, non-FlashInfer
prefill backends, CUDA conv / V-split not loaded yet) falls through to the
unmodified path.

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
and the order of operations on it is unchanged (materialize -> spec decode ->
conv -> zero -> chunk); the batched launches run the same per-(item, head,
layer) / per-row code on the same inputs.
"""

import copy
import math
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# ----------------------------------------------------------------------------
# gates (read once per process)
# ----------------------------------------------------------------------------
ENABLED = os.environ.get("GGM", "0") == "1"
MAT = os.environ.get("VLLM_GDN_STEP_PLAN_MAT", "1") == "1"
ZERO = os.environ.get("VLLM_GDN_STEP_PLAN_ZERO", "1") == "1"
BUFS = os.environ.get("VLLM_GDN_STEP_PLAN_BUFS", "1") == "1"
VSDIRECT = os.environ.get("GGM_VSDIRECT", "0") == "1"
LAZY = ENABLED and os.environ.get("GGM_LAZY", "0") == "1"
MDREUSE = ENABLED and os.environ.get("GGM_MDREUSE", "0") == "1"
MDCHECK = [int(os.environ.get("GGM_MDCHECK", "0"))]
LAZY_IDX = LAZY and os.environ.get("VLLM_GDN_STEP_PLAN_LAZY_IDX", "0") == "1"
LAZY_IDX_CHECK = [int(os.environ.get("VLLM_GDN_STEP_PLAN_LAZY_IDX_CHECK", "0"))]
MDREUSE_PTR_KEY = os.environ.get("VLLM_GDN_STEP_PLAN_MDREUSE_PTR_KEY", "0") == "1"
LOG = os.environ.get("VLLM_GDN_STEP_PLAN_LOG", "1") == "1"
LOG_EVERY = int(os.environ.get("VLLM_GDN_STEP_PLAN_LOG_EVERY", "2000"))
# Host-side hoists of run_plan (exact: same kernels, same arguments; default off,
# one gate each so they can be attributed separately; never used during a CUDA
# graph capture, which keeps the stock path):
#   VLLM_GDN_PLAN_HOIST_STATIC=1  per-layer constants (conv-state / conv-weight
#       views, layout flag, scalar casts) built once per layer and reused while
#       the layer's KV-cache tensors and conv weight are the same objects;
#   VLLM_GDN_PLAN_HOIST_CONV=1    the spec-row causal_conv1d_update launch with
#       its scalar / constexpr arguments computed once per step plan and checked
#       per layer (dtype, strides, shapes, bias, activation), falling back to
#       the stock wrapper on any difference.
HOIST_STATIC = ENABLED and os.environ.get("VLLM_GDN_PLAN_HOIST_STATIC", "0") == "1"
HOIST_CONV = ENABLED and os.environ.get("VLLM_GDN_PLAN_HOIST_CONV", "0") == "1"
# Spec || prefill overlap of one planned mixed GDN layer (rubin-gdnpf; exact by
# construction, default off; host-side only: the GDN core is a splitting op, so
# this changes no traced / compiled code and is not a compile-hash factor):
#   VLLM_GDN_SPEC_OVERLAP=1  run the spec-row branch (causal_conv1d_update + the
#       deferred-commit decode, rows [0, S)) concurrently with the prefill chunk
#       kernel + gated RMSNorm (rows [S, N)). The two branches touch disjoint
#       rows of mixed_qkv / core_attn_out and disjoint conv / SSM state slots;
#       the CUDA conv + post-conv of the prefill rows runs first, the fork is
#       recorded after it, and the join is enqueued on the caller's stream
#       before run_plan returns (before any reader of either branch).
#       Same kernels, same arguments, same per-buffer order.
#   VLLM_GDN_SPEC_OVERLAP_MODE=side (default): spec branch on a side stream,
#       chunk + norm on the caller's stream, enqueued first (GB300 gdnp mode 2;
#       the side stream's event wake gives the chunk's full-SM CTAs a head start);
#       =hp: chunk + norm on a high-priority stream, spec branch on the caller's
#       stream (measured ~1.5-2.5 us/layer slower on VR200: the chunk then waits
#       for the cross-stream wake itself).
#   VLLM_GDN_SPEC_OVERLAP_MIN_S (192): only steps with >= this many spec tokens
#       (host cost ~4 stream ops per layer; low C is host-bound).
#   VLLM_GDN_SPEC_OVERLAP_MAX_NS (0 = no limit): only steps with <= this many
#       prefill sequences (the chunk grid fills the GPU from 4 sequences on).
#   Never during a CUDA graph capture (the GDN layer graphs keep their own path).
OVL = ENABLED and os.environ.get("VLLM_GDN_SPEC_OVERLAP", "0") == "1"
OVL_MODE = os.environ.get("VLLM_GDN_SPEC_OVERLAP_MODE", "side").strip().lower()
OVL_MIN_S = int(os.environ.get("VLLM_GDN_SPEC_OVERLAP_MIN_S", "192"))
OVL_MAX_NS = int(os.environ.get("VLLM_GDN_SPEC_OVERLAP_MAX_NS", "0"))
_OVL_RES: dict = {}

HOST_TRIMS = os.environ.get("GGM_OG2", "0") == "1"
GROUP_MATERIALIZE = (
    HOST_TRIMS and os.environ.get("VLLM_GDN_GROUP_MATERIALIZE", "1") == "1"
)
MERGED_ROW_QUANT = (
    HOST_TRIMS and os.environ.get("VLLM_GDN_MERGED_ROW_QUANT", "1") == "1"
)

STATS = {
    "plans": 0,
    "plans_fast": 0,
    "layers_fast": 0,
    "fallback_reasons": {},
    "mat_launches": 0,
    "zero_launches": 0,
    "conv_fallback": 0,
}
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


def _gdn():
    """The Qwen GDN layer module (imported lazily: it imports this module at its
    top). Its flags are read at call time.
    """
    m = _MODS.get("gdn")
    if m is None:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as m

        _MODS["gdn"] = m
    return m


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
    if ENABLED:
        if not state_commit:
            raise RuntimeError(
                "GGM=1 (GDN step plan) requires GDN_STATE_COMMIT=1 (deferred GDN "
                "state commit)"
            )
        logger.info(
            "GDN step plan enabled: MAT=%d ZERO=%d BUFS=%d LAZY=%d VSDIRECT=%d "
            "MDREUSE=%d",
            int(MAT),
            int(ZERO),
            int(BUFS),
            int(LAZY),
            int(VSDIRECT),
            int(MDREUSE),
        )
    if HOIST_STATIC or HOIST_CONV:
        logger.info(
            "GDN step plan host hoists enabled: static=%d conv=%d",
            int(HOIST_STATIC),
            int(HOIST_CONV),
        )
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
    from vllm.v1.attention.backends import gdn_fused_metadata

    views = gdn_fused_metadata.mat_inputs(md, items)
    if views is not None:
        # VLLM_GDN_FUSED_MD_MAT: views of the fused metadata, same values as the
        # zeros + slice copies below
        gsc.materialize(
            0, items, gt.gsc_table, gt.H, slots[:items], views[0], has_init=views[1]
        )
        STATS["mat_launches"] += 1
        return
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


# ----------------------------------------------------------------------------
# GGM=1: per-step GDN plan
# ----------------------------------------------------------------------------
_TABLES: dict = {}


# batched zero-state kernel: state[layer][slot[i], head] = 0 for sequences without
# an initial state (one launch for all layers of a KV-cache group)
# fmt: off
@triton.jit
def _gdn_zero_state_slots_layers_kernel(ptr_table, slot_ptr, has_init_ptr, stride_slot,
                                        HEAD_ELEMS: tl.constexpr, BLOCK: tl.constexpr, BF16: tl.constexpr):  # noqa: E501
    i_seq = tl.program_id(0)
    i_head = tl.program_id(1)
    i_layer = tl.program_id(2)
    has_init = tl.load(has_init_ptr + i_seq)
    slot = tl.load(slot_ptr + i_seq).to(tl.int64)
    if (has_init != 0) or (slot < 0):
        return
    base_i = tl.load(ptr_table + i_layer)
    if BF16:
        base = base_i.to(tl.pointer_type(tl.bfloat16))
    else:
        base = base_i.to(tl.pointer_type(tl.float32))
    base = base + slot * stride_slot + i_head * HEAD_ELEMS
    offs = tl.arange(0, BLOCK)
    zeros = tl.zeros([BLOCK], dtype=base.dtype.element_ty)
    for start in range(0, HEAD_ELEMS, BLOCK):
        tl.store(base + start + offs, zeros)
# fmt: on


class _GroupTable:
    """Per metadata-sharing layer group: deferred-commit LayerTable + SSM base
    pointer table (for the batched zero kernel).
    """

    def __init__(self, layers, gsc):
        entries = []
        for L in layers:
            gsc._layer_check(L)
            entries.append((L.kv_cache[1], L.A_log, L.dt_bias, 0))
        self.gsc_table = gsc.LayerTable(entries, entries[0][0].device)
        st = entries[0][0]
        self.ptrs = torch.tensor(
            [e[0].data_ptr() for e in entries], dtype=torch.int64, device=st.device
        )
        self.n = len(entries)
        self.dtype = st.dtype
        self.hv = st.size(1)
        self.head_elems = st.size(2) * st.size(3)
        self.slot_stride = st.stride(0)
        for s, _, _, _ in entries:
            assert (
                s.dtype == st.dtype
                and s.stride(0) == st.stride(0)
                and s.stride(1) == self.head_elems
            )
            assert s.stride(3) == 1 and s.size(1) == self.hv
        self.H = layers[0]._gsc_H


class _Plan:
    __slots__ = (
        "S",
        "P",
        "N",
        "nr",
        "conv_si",
        "nacc",
        "cu_s",
        "mql",
        "dec_si",
        "dec_direct",
        "ci",
        "hi",
        "cu_ns",
        "ns",
        "tph",
        "rv",
        "gext",
        "slots",
        "has_init",
        "cu_p",
        "want_cp",
        "vsf",
        "cu_p_i32",
        "maxlen_kw",
        "zero_done",
        "bufs",
        "vsmod",
        "scale",
        "names",
        "vs_direct",
        "conv_pre",
    )


def _vs_direct(p, q, v, st):
    """The V-split adapter's cached compiled kernel for exactly the key the
    adapter used for this step (None if not compiled), a workspace for this
    step and the current stream.
    """
    import cuda.bindings.driver as cuda

    ad = p.vsmod.adapter
    HQ, HV = q.size(1), v.size(1)
    devi = q.device.index if q.device.index is not None else torch.cuda.current_device()
    num_sm = ad._num_sm(devi)
    key = (
        devi,
        num_sm,
        str(q.dtype),
        str(st.dtype),
        HQ,
        HV,
        HQ >= HV,
        True,
        True,
        True,
        str(p.slots.dtype),
        tuple(st.stride()[1:]),
        tuple(st.stride()[1:]),
        int(p.vsf),
    )
    if hasattr(ad, "_cg0_split"):
        # The gdnchunk adapter (CG0 split / C1 reorder) appends its two layout
        # flags to the cache key (use_init is True here). Without them this
        # lookup never hits, and every layer falls back to the full adapter
        # call plus a fresh _vs_direct attempt (host time only; same kernel).
        key = key + (ad._cg0_split(int(p.vsf), True), bool(ad._C1_REORDER))
    # Guard: _vs_direct runs right after an adapter call with the same inputs, so
    # the adapter's last key must equal this rebuilt key (length, field order and
    # values). A future adapter-key change would otherwise silently disable the
    # direct launch (every layer falling back to the full adapter call).
    lk = getattr(ad, "_LAST_KEY", None)
    if lk is not None and lk[0] is not None and lk[0] != key:
        STATS["vs_direct_key_mismatch"] = STATS.get("vs_direct_key_mismatch", 0) + 1
        logger.warning_once(
            "GDN step plan: VSDIRECT key mismatch vs the V-split adapter "
            "(adapter key %d fields, plan key %d fields): direct launch disabled",
            len(lk[0]),
            len(key),
        )
        return None
    c = ad._cache(*key)
    if "compiled" not in c:
        STATS["vs_direct_miss"] = STATS.get("vs_direct_miss", 0) + 1
        return None
    B = p.cu_p_i32.size(0) - 1
    ws = torch.empty(
        ad.GatedDeltaNetChunkedKernel.get_workspace_size(num_sm, B, HQ, HV, True),
        dtype=torch.int8,
        device=q.device,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    STATS["vs_direct"] = STATS.get("vs_direct", 0) + 1
    logger.info_once(
        "GDN step plan: VSDIRECT direct launch engaged (adapter key %d fields)", len(key)
    )
    return (c["compiled"], ws, stream, tuple(st.stride()))


def _fallback(reason):
    d = STATS["fallback_reasons"]
    d[reason] = d.get(reason, 0) + 1
    return False


def _materialize_rows(md, layers, gt):
    """gdn_state_commit.materialize_non_spec for every layer of the group, in one
    launch (identical per-slot inputs).
    """
    gsc = _gsc()
    done = md.__dict__.setdefault("_gsc_done", set())
    todo = [L for L in layers if id(L) not in done]
    if not todo:
        return
    if len(todo) != len(layers):
        # partially done already (not expected): the missing layers one by one
        STATS["fallback_reasons"]["mat_partial"] = (
            STATS["fallback_reasons"].get("mat_partial", 0) + 1
        )
        for L in todo:
            gsc.materialize_non_spec(L, md)
        return
    slots = md.non_spec_state_indices_tensor
    items = md.num_prefills + md.num_decodes
    for L in layers:
        done.add(id(L))
    if slots is None or items <= 0:
        return
    items = min(items, slots.size(0))
    from vllm.v1.attention.backends import gdn_fused_metadata

    views = gdn_fused_metadata.mat_inputs(md, items)
    if views is not None:
        # VLLM_GDN_FUSED_MD_MAT: views of the fused metadata, same values as the
        # zeros + slice copies below
        gsc.materialize(
            0, items, gt.gsc_table, gt.H, slots[:items], views[0], has_init=views[1]
        )
        STATS["mat_launches"] += 1
        return
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
    gsc.materialize(0, items, gt.gsc_table, gt.H, slots[:items], n, has_init=hi)
    STATS["mat_launches"] += 1


def build_plan(layer, md, raw):
    """The step plan for `md` (built at the first GDN layer of the group that
    sees it), or False when the step does not take the planned path.
    """
    mod = _gdn()
    gsc = _gsc()
    STATS["plans"] += 1
    if not isinstance(md, mod.GDNAttentionMetadata):
        return _fallback("not_gdn_md")
    if layer._can_use_fused_gdn_mtp_decode(md) and md.num_prefills == 0:
        return _fallback("decode_only")
    if not layer._can_use_mixed_fastpath(md):
        return _fallback("not_mixed_fastpath")
    if mod._GDN_VERIFY_LEFT[0] > 0 or mod._GDN_MIXED_SPEC_TRITON:
        return _fallback("verify_or_spec_triton")
    if not (mod._GDN_FUSED_CONV and getattr(mod, "_GDN_CONV_CUDA", False)):
        return _fallback("no_fused_conv_cuda")
    if not mod._GDN_CONV_CUDA_MOD or mod._GDN_CONV_CUDA_MOD[0] is None:
        return _fallback("conv_cuda_not_loaded")
    if layer.gdn_prefill_backend != "flashinfer" or not mod._GDN_FI_STATE_POOL:
        return _fallback("fi_backend_or_pool")
    p = _Plan()
    S = md.num_spec_decode_tokens
    P = md.num_prefill_tokens
    p.S, p.P, p.N = S, P, S + P
    # ---- spec part ----
    p.nr = md.num_spec_decodes
    if S > 0:
        si = md.spec_state_indices_tensor
        p.conv_si = si[: p.nr, 0]
        p.nacc = md.num_accepted_tokens[: p.nr]
        p.cu_s = md.spec_query_start_loc[: p.nr + 1]
        p.mql = si.size(1)
        dec = si[: p.nr]
        # the MTP decode custom op dispatches to the deferred-commit decode kernel:
        # call its extension directly (same kernel, same arguments)
        p.dec_direct = gsc.decode_routed()
        if p.dec_direct and not dec.is_contiguous():
            dec = dec[:, :1].contiguous()
        p.dec_si = dec
    # ---- prefill conv (CUDA fused conv1d + post-conv) ----
    gcc = mod._GDN_CONV_CUDA_MOD[0]
    conv_w = layer.conv1d.weight
    H, K, V = layer.num_k_heads // layer.tp_size, layer.head_k_dim, layer.head_v_dim
    if K != 128 or V != 128 or conv_w.size(2) != 4:
        return _fallback("conv_contract")
    p.gext = gcc.load()
    ci = md.non_spec_state_indices_tensor.contiguous()
    p.ci = ci.contiguous()
    p.hi = md.has_initial_state.contiguous()
    cu = md.non_spec_query_start_loc
    if cu.dtype != torch.int32:
        cu = cu.to(torch.int32)
    p.cu_ns = cu.contiguous()
    p.ns = int(md.num_prefills)
    if mod._GDN_CONV_CUDA_TPH == "auto":
        p.tph = 4 if P < 1024 * max(p.ns, 1) else 8  # noqa: SIM300
    else:
        p.tph = int(mod._GDN_CONV_CUDA_TPH)
    p.rv = int(gcc.RV)
    # ---- FlashInfer / V-split chunk ----
    slots = md.prefill_state_indices
    has_init = md.prefill_has_initial_state
    if slots is None or has_init is None or md.prefill_query_start_loc is None:
        return _fallback("no_prefill_md")
    p.slots = slots.contiguous()
    p.has_init = has_init.contiguous()
    p.cu_p = md.prefill_query_start_loc
    n = p.cu_p.numel() - 1
    maxlen = getattr(md, "prefill_max_seqlen", 0)
    p.want_cp = mod._gdn_fi_want_cp(n, P, maxlen)
    if p.want_cp and not mod._GDN_FI_CP_POOL:
        return _fallback("cp_gather_path")
    p.maxlen_kw = mod._gdn_fi_maxlen_kw(p.want_cp, md)
    p.vsf = 1
    p.vsmod = None
    p.vs_direct = None
    p.conv_pre = None
    if mod._GDN_FI_VSPLIT and not p.want_cp:
        st = mod._GDN_VSPLIT_STATE
        if mod._GDN_FI_VSPLIT_CHECK:
            return _fallback("vsplit_check_mode")
        if not st.get("disabled"):
            if "mod" not in st:
                return _fallback("vsplit_not_loaded")
            vs = st["mod"]
            HV = layer.num_v_heads // layer.tp_size
            vsf = vs.choose_vsplit(n, P, int(maxlen), hv=HV)
            if vsf != 1:
                ssm = layer.kv_cache[1]
                if (
                    ssm.dtype not in (torch.float32, torch.bfloat16)
                    or ssm.stride(3) != 1
                ):
                    return _fallback("vsplit_ineligible")
                p.vsf = vsf
                p.vsmod = vs
                p.cu_p_i32 = p.cu_p.to(torch.int32)
    p.scale = 1.0 / math.sqrt(K)
    # ---- per-step buffers ----
    p.bufs = None
    if BUFS:
        HV = layer.num_v_heads // layer.tp_size
        dev, dt = conv_w.device, conv_w.dtype
        p.bufs = (
            torch.empty(P, H, K, dtype=dt, device=dev),
            torch.empty(P, H, K, dtype=dt, device=dev),
            torch.empty(P, HV, V, dtype=dt, device=dev),
            torch.empty(P, HV, dtype=torch.float32, device=dev),
            torch.empty(P, HV, dtype=torch.float32, device=dev),
        )
    # ---- group (all layers sharing this metadata object) ----
    p.zero_done = False
    if MAT or ZERO:
        from vllm.forward_context import get_forward_context

        fc = get_forward_context()
        names = tuple(k for k, v in raw.items() if v is md)
        layers = [fc.no_compile_layers.get(k) for k in names]
        C = mod.QwenGatedDeltaNetAttention
        if (
            any(L is None or not isinstance(L, C) for L in layers)
            or layer.prefix not in names
        ):
            return _fallback("group_layers")
        key = (names, tuple(L.kv_cache[1].data_ptr() for L in layers))
        gt = _TABLES.get(key)
        if gt is None:
            gt = _TABLES[key] = _GroupTable(layers, gsc)
        if MAT:
            _materialize_rows(md, layers, gt)
        if md.__dict__.get("_step_plan_zeroed"):
            p.zero_done = True  # done by the GDN layer-graph builder step
        elif ZERO and p.slots.numel() > 0:
            _gdn_zero_state_slots_layers_kernel[(p.slots.numel(), gt.hv, gt.n)](
                gt.ptrs,
                p.slots,
                p.has_init,
                gt.slot_stride,
                HEAD_ELEMS=gt.head_elems,
                BLOCK=4096,
                BF16=gt.dtype == torch.bfloat16,
                num_warps=4,
            )
            STATS["zero_launches"] += 1
            p.zero_done = True
    STATS["plans_fast"] += 1
    # shape histograms (number of prefill seqs, spec requests)
    hb = STATS.setdefault("ns_hist", {})
    kb = next(b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1 << 30) if p.ns <= b)
    hb[kb] = hb.get(kb, 0) + 1
    hr = STATS.setdefault("nr_hist", {})
    kr = next(b for b in (0, 32, 64, 128, 192, 256, 1 << 30) if p.nr <= b)
    hr[kr] = hr.get(kr, 0) + 1
    STATS["vsf2"] = STATS.get("vsf2", 0) + int(p.vsf == 2)
    STATS["cp"] = STATS.get("cp", 0) + int(bool(p.want_cp))
    if LOG and STATS["plans_fast"] % LOG_EVERY == 1:
        logger.info(
            "GDN step plan #%d: S=%d P=%d nr=%d ns=%d vsf=%d cp=%s stats=%s",
            STATS["plans_fast"],
            S,
            P,
            p.nr,
            p.ns,
            p.vsf,
            p.want_cp,
            {k: v for k, v in STATS.items()},
        )
    return p


_CC: list = []


def _cc():
    """The causal_conv1d ops module (imported lazily, like _gdn / _gsc)."""
    if not _CC:
        from vllm.model_executor.layers.mamba.ops import causal_conv1d as m

        _CC.append(m)
    return _CC[0]


_CONV_DIM_FIRST: list = []


def _layer_static(layer, mod, kv0, kv1, w):
    """VLLM_GDN_PLAN_HOIST_STATIC: the per-layer values run_plan rebuilds on
    every call, computed exactly as run_plan does. Valid while kv0 / kv1 / w are
    the same tensor objects (their metadata cannot change in place)."""
    if not _CONV_DIM_FIRST:
        _CONV_DIM_FIRST.append(bool(mod.is_conv_state_dim_first()))
    conv_state = kv0 if _CONV_DIM_FIRST[0] else kv0.transpose(-1, -2)
    conv_weights = w.view(w.size(0), w.size(2))
    STATS["hoist_static_build"] = STATS.get("hoist_static_build", 0) + 1
    return (
        kv0,
        kv1,
        w,
        conv_state,
        conv_weights,
        float(layer.head_k_dim**-0.5),
        float(layer.layer_norm_epsilon),
        layer.norm.activation == "sigmoid",
        int(layer.num_k_heads // layer.tp_size),
    )


class _ConvPre:
    """VLLM_GDN_PLAN_HOIST_CONV: the arguments causal_conv1d_update passes to
    _causal_conv1d_update_kernel for this step's spec rows (varlen + spec
    decoding, out = x, default null_block_id), computed once per plan from the
    first layer and valid for every layer with the same checked properties."""

    __slots__ = ("chk", "grid", "scal", "kw")


def _conv_pre(p, x, conv_state, weight, bias, activation):
    cc = _cc()
    if isinstance(activation, bool):
        act = "silu" if activation is True else None
    else:
        act = activation
    if act is not None and act not in ("silu", "swish"):
        return None
    if p.nacc is None or p.cu_s is None or p.conv_si is None:
        return None  # not the varlen spec-decoding form this path mirrors
    batch = p.conv_si.size(0)
    dim = x.size(1)
    seqlen = p.mql
    _, width = weight.shape
    num_cache_lines, _, _ = conv_state.size()
    stride_w_dim, stride_w_width = weight.stride()
    stride_x_token, stride_x_dim = x.stride()
    s_seq, s_dim, s_tok = conv_state.stride()
    state_len = width - 1 + (seqlen - 1)  # num_accepted_tokens is not None
    c = _ConvPre()
    c.chk = (
        x.dtype,
        x.stride(),
        dim,
        conv_state.dtype,
        tuple(conv_state.shape),
        conv_state.stride(),
        tuple(weight.shape),
        weight.stride(),
        bias is None,
        activation,
    )
    c.grid = (batch, triton.cdiv(dim, 256))
    c.scal = (
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        0,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        s_seq,
        s_dim,
        s_tok,
        p.conv_si.stride(0),
        0,
        stride_x_dim,
        stride_x_token,
        cc.NULL_BLOCK_ID,
    )
    c.kw = dict(
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=act in ["silu", "swish"],
        IS_VARLEN=True,
        IS_APC_ENABLED=False,
        IS_SPEC_DECODING=True,
        NP2_STATELEN=triton.next_power_of_2(state_len),
        HAS_NULL_BLOCK=cc.NULL_BLOCK_ID is not None,
        BLOCK_N=256,
        launch_pdl=cc.current_platform.is_arch_support_pdl(),
    )
    return c


def conv_update_spec(p, x, conv_state, weight, bias, activation):
    """causal_conv1d_update(x, conv_state, weight, bias, activation,
    conv_state_indices=p.conv_si, num_accepted_tokens=p.nacc,
    query_start_loc=p.cu_s, max_query_len=p.mql, validate_data=False) with the
    per-step arguments hoisted (VLLM_GDN_PLAN_HOIST_CONV). Same kernel object,
    same arguments; launched on the current stream. Returns x (the wrapper's
    result when x.dtype == conv_state.dtype)."""
    c = p.conv_pre
    if c is None:  # first spec-row layer of this plan; False = not applicable
        c = _conv_pre(p, x, conv_state, weight, bias, activation)
        p.conv_pre = c if c is not None else False
    if c and c.chk == (
        x.dtype,
        x.stride(),
        x.size(1),
        conv_state.dtype,
        tuple(conv_state.shape),
        conv_state.stride(),
        tuple(weight.shape),
        weight.stride(),
        bias is None,
        activation,
    ) and x.dtype == conv_state.dtype and x.dim() == 2:
        _cc()._causal_conv1d_update_kernel[c.grid](
            x,
            weight,
            bias,
            conv_state,
            p.conv_si,
            p.nacc,
            p.cu_s,
            None,
            None,
            x,
            *c.scal,
            **c.kw,
        )
        STATS["hoist_conv"] = STATS.get("hoist_conv", 0) + 1
        return x
    STATS["hoist_conv_fallback"] = STATS.get("hoist_conv_fallback", 0) + 1
    return _gdn().causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices=p.conv_si,
        num_accepted_tokens=p.nacc,
        query_start_loc=p.cu_s,
        max_query_len=p.mql,
        validate_data=False,
    )


def _run_spec(layer, p, mod, gsc, hoist, mixed_qkv, b, a, output_gate, core_attn_out,
              ssm_state, conv_state, conv_weights, k_scale, n_eps, n_sig) -> None:
    """Spec rows [0, S) of a planned mixed layer: causal_conv1d_update + the
    deferred-commit MTP decode (unchanged kernels and arguments)."""
    S = p.S
    if hoist and HOIST_CONV:
        mixed_qkv_spec = conv_update_spec(
            p, mixed_qkv[:S], conv_state, conv_weights, layer.conv1d.bias, layer.activation
        )
    else:
        mixed_qkv_spec = mod.causal_conv1d_update(
            mixed_qkv[:S],
            conv_state,
            conv_weights,
            layer.conv1d.bias,
            layer.activation,
            conv_state_indices=p.conv_si,
            num_accepted_tokens=p.nacc,
            query_start_loc=p.cu_s,
            max_query_len=p.mql,
            validate_data=False,
        )
    if p.dec_direct:
        gsc.STATS["decode_calls"] += 1
        gsc.load().decode(
            mixed_qkv_spec,
            a[:S],
            b[:S],
            layer.A_log,
            layer.dt_bias,
            p.dec_si,
            p.cu_s,
            p.nacc,
            ssm_state,
            output_gate[:S],
            layer.norm.weight,
            core_attn_out[:S],
            k_scale,
            n_eps,
            n_sig,
        )
    else:
        mod.ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed_qkv_spec,
            a=a[:S],
            b=b[:S],
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            state_indices=p.dec_si,
            cu_seqlens=p.cu_s,
            num_accepted_tokens=p.nacc,
            state=ssm_state,
            output_gate=output_gate[:S],
            norm_weight=layer.norm.weight,
            out=core_attn_out[:S],
            scale=layer.head_k_dim**-0.5,
            norm_eps=layer.layer_norm_epsilon,
            output_gate_activation=layer.norm.activation,
        )


def _run_chunk_norm(layer, p, mod, q, k, v, g, beta, out, output_gate, ssm_state,
                    cu_stream=None, main=None) -> None:
    """Prefill rows [S, N) of a planned mixed layer after the CUDA conv: fresh-slot
    zeroing, the chunked GDN kernel (state pool in place) and the gated RMSNorm
    (unchanged kernels and arguments). cu_stream: explicit CUstream for the
    V-split direct launch (overlap mode hp); main: the caller's stream (overlap).
    """
    S, N = p.S, p.N
    # ---- chunked GDN (state pool in place) ----
    if not p.zero_done:
        mod.gdn_zero_state_slots(ssm_state, p.slots, p.has_init)
    done = False
    if p.vsf != 1:  # noqa: SIM102 - same structure as the unplanned path
        if (
            q.is_contiguous()
            and k.is_contiguous()
            and v.is_contiguous()
            and out.is_contiguous()
            and g.is_contiguous()
            and beta.is_contiguous()
            and q.size(2) == 128
        ):
            st = mod._GDN_VSPLIT_STATE
            d = p.vs_direct
            if d is not None and d[3] == tuple(ssm_state.stride()):
                # same compiled object, same arguments as chunk_gated_delta_rule_vsplit
                d[0](
                    q,
                    k,
                    v,
                    g,
                    beta,
                    out,
                    p.cu_p_i32,
                    ssm_state,
                    ssm_state,
                    p.slots,
                    None,
                    None,
                    0,
                    p.scale,
                    d[1],
                    d[2] if cu_stream is None else cu_stream,
                )
            else:
                p.vsmod.chunk_gated_delta_rule_vsplit(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    out,
                    p.cu_p_i32,
                    ssm_state,
                    ssm_state,
                    p.scale,
                    state_indices=p.slots,
                    v_split=p.vsf,
                )
                if VSDIRECT:
                    if main is None:
                        p.vs_direct = _vs_direct(p, q, v, ssm_state)
                    else:
                        # overlap: cache the caller's stream + allocate the workspace on it
                        cur = torch.cuda.current_stream()
                        torch.cuda.set_stream(main)
                        try:
                            p.vs_direct = _vs_direct(p, q, v, ssm_state)
                        finally:
                            torch.cuda.set_stream(cur)
            st["calls"] = st.get("calls", 0) + 1
            done = True
    if not done:
        from flashinfer.gdn_prefill import (
            chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
        )

        chunk_gated_delta_rule_fi(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_state,
            output_final_state=True,
            cu_seqlens=p.cu_p,
            output=out,
            output_state=ssm_state,
            use_cp=p.want_cp,
            state_indices=p.slots,
            **p.maxlen_kw,
        )
    mod.gdn_gated_rmsnorm_(
        out,
        output_gate[S:N],
        layer.norm.weight,
        layer.layer_norm_epsilon,
        layer.norm.activation,
    )


def _ovl_res(dev):
    """(stream, CUstream, fork event, join event) of the overlap on this device."""
    r = _OVL_RES.get(dev)
    if r is None:
        import cuda.bindings.driver as cuda

        hp = OVL_MODE == "hp"
        st = torch.cuda.Stream(device=dev, priority=-1 if hp else 0)
        r = _OVL_RES[dev] = (
            st,
            cuda.CUstream(st.cuda_stream),
            torch.cuda.Event(),
            torch.cuda.Event(),
        )
        logger.info(
            "GDN step plan: spec||prefill overlap stream created (mode=%s, stream "
            "priority %d, caller stream priority %d, min S %d, max ns %d)",
            OVL_MODE,
            st.priority,
            torch.cuda.current_stream(dev).priority,
            OVL_MIN_S,
            OVL_MAX_NS,
        )
    return r


def _ovl_ok(p, mixed_qkv, conv_state) -> bool:
    if OVL_MAX_NS and p.ns > OVL_MAX_NS:
        STATS["ovl_off_ns"] = STATS.get("ovl_off_ns", 0) + 1
        return False
    if (
        p.N <= p.S
        or mixed_qkv.dtype != conv_state.dtype
        or torch.cuda.is_current_stream_capturing()
    ):
        STATS["ovl_off_other"] = STATS.get("ovl_off_other", 0) + 1
        return False
    return True


def run_plan(layer, p, md, mixed_qkv, b, a, output_gate, core_attn_out) -> None:
    """One layer of a planned mixed step: the kernels of the zero-copy mixed
    path, in its order, with its arguments.
    """
    mod = _gdn()
    gsc = _gsc()
    S, N = p.S, p.N
    hoist = (HOIST_STATIC or HOIST_CONV) and not torch.cuda.is_current_stream_capturing()
    if hoist and HOIST_STATIC:
        kvc = layer.kv_cache
        kv0, kv1, w = kvc[0], kvc[1], layer.conv1d.weight
        ls = layer.__dict__.get("_gdn_plan_static")
        if ls is None or ls[0] is not kv0 or ls[1] is not kv1 or ls[2] is not w:
            ls = layer.__dict__["_gdn_plan_static"] = _layer_static(layer, mod, kv0, kv1, w)
        ssm_state, conv_state, conv_weights = kv1, ls[3], ls[4]
        k_scale, n_eps, n_sig, n_kh = ls[5], ls[6], ls[7], ls[8]
    else:
        ssm_state = layer.kv_cache[1]
        conv_state = (
            layer.kv_cache[0]
            if mod.is_conv_state_dim_first()
            else layer.kv_cache[0].transpose(-1, -2)
        )
        conv_weights = layer.conv1d.weight.view(
            layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
        )
        k_scale = float(layer.head_k_dim**-0.5)
        n_eps = float(layer.layer_norm_epsilon)
        n_sig = layer.norm.activation == "sigmoid"
        n_kh = int(layer.num_k_heads // layer.tp_size)
    if not MAT:
        gsc._layer_check(layer)
        gsc.materialize_non_spec(layer, md)
    ovl = OVL and S >= OVL_MIN_S and _ovl_ok(p, mixed_qkv, conv_state)
    if S > 0 and not ovl:
        _run_spec(layer, p, mod, gsc, hoist, mixed_qkv, b, a, output_gate, core_attn_out,
                  ssm_state, conv_state, conv_weights, k_scale, n_eps, n_sig)
    # ---- prefill conv + post-conv (CUDA) ----
    if p.bufs is not None:
        q, k, v, g, beta = p.bufs
    else:
        P = p.P
        H, HV = layer.num_k_heads // layer.tp_size, layer.A_log.shape[0]
        x = mixed_qkv
        q = torch.empty(P, H, 128, dtype=x.dtype, device=x.device)
        k = torch.empty(P, H, 128, dtype=x.dtype, device=x.device)
        v = torch.empty(P, HV, 128, dtype=x.dtype, device=x.device)
        g = torch.empty(P, HV, dtype=torch.float32, device=x.device)
        beta = torch.empty(P, HV, dtype=torch.float32, device=x.device)
    ok = p.gext.run(
        mixed_qkv[S:N],
        conv_weights,
        conv_state,
        p.ci,
        p.hi,
        p.cu_ns,
        p.ns,
        a[S:N],
        b[S:N],
        layer.A_log,
        layer.dt_bias,
        q,
        k,
        v,
        g,
        beta,
        n_kh,
        int(p.tph),
        int(p.rv),
    )
    gcc = mod._GDN_CONV_CUDA_MOD[0]
    if ok:
        gcc.STATS["calls"] += 1
    else:
        # as in the unplanned path: the CUDA kernel refused (no launch) -> Triton kernel
        gcc.STATS["fallbacks"] += 1
        STATS["conv_fallback"] += 1
        q, k, v, g, beta = mod.gdn_fused_conv_post_conv(
            mixed_qkv[S:N],
            conv_weights,
            conv_state,
            p.ci,
            p.hi,
            p.cu_ns,
            p.ns,
            a[S:N],
            b[S:N],
            layer.A_log,
            layer.dt_bias,
            layer.num_k_heads // layer.tp_size,
            layer.head_k_dim,
            layer.head_v_dim,
            use_cuda=False,
        )
    out = core_attn_out[S:N]
    if not ovl:
        _run_chunk_norm(layer, p, mod, q, k, v, g, beta, out, output_gate, ssm_state)
        STATS["layers_fast"] += 1
        return
    # ---- spec || prefill overlap: fork after the prefill conv, join before return ----
    main = torch.cuda.current_stream()
    st, cu_st, ev_fork, ev_join = _ovl_res(main.device)
    ev_fork.record(main)
    st.wait_event(ev_fork)
    torch.cuda.set_stream(st)
    try:
        if OVL_MODE == "hp":
            # chunk + norm first, on the high-priority stream (explicit CUstream for
            # the cached direct launch, whose own stream is the caller's)
            _run_chunk_norm(layer, p, mod, q, k, v, g, beta, out, output_gate, ssm_state,
                            cu_stream=cu_st, main=main)
        else:
            torch.cuda.set_stream(main)
            # chunk + norm first (host order), on the caller's stream
            _run_chunk_norm(layer, p, mod, q, k, v, g, beta, out, output_gate, ssm_state)
            torch.cuda.set_stream(st)
            _run_spec(layer, p, mod, gsc, hoist, mixed_qkv, b, a, output_gate, core_attn_out,
                      ssm_state, conv_state, conv_weights, k_scale, n_eps, n_sig)
    finally:
        torch.cuda.set_stream(main)
    ev_join.record(st)
    if OVL_MODE == "hp":
        _run_spec(layer, p, mod, gsc, hoist, mixed_qkv, b, a, output_gate, core_attn_out,
                  ssm_state, conv_state, conv_weights, k_scale, n_eps, n_sig)
    main.wait_event(ev_join)
    STATS["layers_fast"] += 1
    STATS["ovl_layers"] = STATS.get("ovl_layers", 0) + 1
    if STATS["ovl_layers"] == 1:
        logger.info(
            "GDN step plan: spec||prefill overlap engaged (mode=%s, S=%d, P=%d, ns=%d, vsf=%d)",
            OVL_MODE, S, p.P, p.ns, p.vsf,
        )


def forward_core_fused_norm(
    layer, md, raw, mixed_qkv, b, a, output_gate, core_attn_out
) -> bool:
    """Start of QwenGatedDeltaNetAttention._forward_core_fused_norm with GGM=1
    (md = this layer's metadata, raw = the forward context's metadata dict).
    Returns True when the layer ran on the step plan (the caller returns).
    """
    d = md.__dict__
    p = d.get("_step_plan")
    if p is None:
        p = d["_step_plan"] = build_plan(layer, md, raw)
    if p is False:
        if LAZY:
            fill_lazy(md)
        return False
    run_plan(layer, p, md, mixed_qkv, b, a, output_gate, core_attn_out)
    return True


# ----------------------------------------------------------------------------
# GGM_LAZY: deferred fallback-only GDN metadata
# ----------------------------------------------------------------------------
def defer_metadata(md, pending) -> None:
    """Builder side: attach the deferred (fn, kind, args, kwargs) calls to md."""
    md.__dict__["_step_plan_lazy"] = pending
    STATS["lazy_deferred"] = STATS.get("lazy_deferred", 0) + 1


class LazyT:
    """Deferred tensor op (VLLM_GDN_STEP_PLAN_LAZY_IDX): fn(*args, **kwargs), or
    parent[key] for a slice of a deferred value.
    """

    __slots__ = ("fn", "args", "kwargs", "parent", "key")

    def __init__(self, fn=None, args=(), kwargs=None, parent=None, key=None):
        self.fn, self.args, self.kwargs, self.parent, self.key = (
            fn,
            args,
            kwargs or {},
            parent,
            key,
        )

    def __getitem__(self, key):
        return LazyT(parent=self, key=key)

    def materialize(self):
        if self.parent is not None:
            return self.parent.materialize()[self.key]
        args = tuple(a.materialize() if isinstance(a, LazyT) else a for a in self.args)
        return self.fn(*args, **self.kwargs)


_IDX_FIELDS = ("spec_token_indx", "non_spec_token_indx")


def _fill_idx(md) -> None:
    for f in _IDX_FIELDS:
        v = md.__dict__.get(f)
        if isinstance(v, LazyT):
            md.__dict__[f] = v.materialize()
            STATS["lazy_idx_filled"] = STATS.get("lazy_idx_filled", 0) + 1


def check_lazy_indices(md, eager_build) -> None:
    """VLLM_GDN_STEP_PLAN_LAZY_IDX_CHECK: compare the deferred token permutation
    of md with an eager full build (eager_build()); use the eager one on
    mismatch.
    """
    if LAZY_IDX_CHECK[0] > 0 and isinstance(md.__dict__.get("spec_token_indx"), LazyT):
        LAZY_IDX_CHECK[0] -= 1
        full = eager_build()
        ok = all(
            torch.equal(md.__dict__[f].materialize(), getattr(full, f))
            for f in _IDX_FIELDS
        )
        STATS["lazy_idx_checked"] = STATS.get("lazy_idx_checked", 0) + 1
        if not ok:
            STATS["lazy_idx_check_fail"] = STATS.get("lazy_idx_check_fail", 0) + 1
            logger.warning("GDN lazy token-index check mismatch; using eager indices")
            md.spec_token_indx, md.non_spec_token_indx = (
                full.spec_token_indx,
                full.non_spec_token_indx,
            )


def fill_lazy(md) -> None:
    """Compute the deferred metadata of md (no-op if none is pending)."""
    if md is None:
        return
    _fill_idx(md)
    lz = md.__dict__.pop("_step_plan_lazy", None)
    if not lz:
        return
    STATS["lazy_filled"] = STATS.get("lazy_filled", 0) + 1
    for fn, kind, a, k in lz:
        if kind == "chunk":
            md.chunk_indices, md.chunk_offsets = fn(*a, **k)
        else:
            md.nums_dict, md.batch_ptr, md.token_chunk_offset_ptr = fn(*a, **k)


# ----------------------------------------------------------------------------
# GGM_MDREUSE: GDN metadata of the other KV-cache groups derived from the first
# ----------------------------------------------------------------------------
_MDTPL: list = [None]
_FLAG_KEYS = ("_step_plan", "_gsc_done", "_step_plan_graph", "_step_plan_zeroed")


def _base(t):
    if t is None:
        return None
    if MDREUSE_PTR_KEY:
        # identity of the storage region a (possibly re-sliced) tensor views
        return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.device.type)
    return t._base if t._base is not None else t


def _same(a, b) -> bool:
    return a == b if MDREUSE_PTR_KEY else a is b


def _md_equal(a, b):
    bad = []
    for k in set(a.__dict__) | set(b.__dict__):
        if k.startswith("_step_plan") or k.startswith("_gsc"):
            continue
        x, y = a.__dict__.get(k), b.__dict__.get(k)
        if isinstance(x, LazyT) or isinstance(y, LazyT):
            x = x.materialize() if isinstance(x, LazyT) else x
            y = y.materialize() if isinstance(y, LazyT) else y
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            if (
                not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor))
                or x.shape != y.shape
                or x.dtype != y.dtype
                or x.device != y.device
                or not torch.equal(x, y)
            ):
                bad.append(k)
        elif k == "nums_dict":
            continue  # CPU-tensor dict (fallback-only), same CPU inputs
        elif x != y:
            bad.append(k)
    return bad


def _derive(builder, t, m, ndd):
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

    md = copy.copy(t["md"])
    d = md.__dict__
    if "_step_plan_lazy" in d:
        d["_step_plan_lazy"] = list(d["_step_plan_lazy"])
    bt = mamba_get_block_table_tensor(
        m.block_table_tensor,
        m.seq_lens,
        builder.kv_cache_spec,
        builder.vllm_config.cache_config.mamba_cache_mode,
    )
    if md.spec_sequence_masks is None:
        ns = bt[:, 0]
        md.non_spec_state_indices_tensor = ns
        md.prefill_state_indices = ns[md.num_decodes :] if md.num_decodes > 0 else ns
    else:
        mask = ndd >= 0
        si = bt[mask, : builder.num_spec + 1]
        w = builder.num_spec + 1
        if si.size(1) < w:  # deferred state commit: broadcast the single base slot
            si = si[:, :1].expand(si.size(0), w)
        ns = bt[~mask, 0]
        md.spec_state_indices_tensor = si
        md.non_spec_state_indices_tensor = ns
        md.prefill_state_indices = ns
    return md


def mdreuse_build(
    builder,
    full_build,
    common_prefix_len,
    common_attn_metadata,
    num_accepted_tokens=None,
    num_decode_draft_tokens_cpu=None,
    fast_build=False,
):
    """GDNAttentionMetadataBuilder.build with GGM_MDREUSE=1 (full_build = the
    builder's full build, incl. the deferred-commit post-step and GGM_LAZY).
    """
    m = common_attn_metadata
    t = _MDTPL[0]
    mode = builder.vllm_config.cache_config.mamba_cache_mode
    why = None
    if t is None:
        STATS["md_tpl_none"] = STATS.get("md_tpl_none", 0) + 1
    else:
        checks = (
            ("builder", t["builder"] is not builder),
            ("qsl_cpu", t["qsl_cpu"] is m.query_start_loc_cpu),
            ("qsl", t["qsl"] is m.query_start_loc),
            ("seq_lens", t["seq_lens"] is m.seq_lens),
            ("nacc", _same(t["nacc"], _base(num_accepted_tokens))),
            ("ndd", _same(t["ndd"], _base(num_decode_draft_tokens_cpu))),
            ("nreq", t["nreq"] == m.num_reqs),
            ("nat", t["nat"] == m.num_actual_tokens),
            ("num_spec", t["num_spec"] == builder.num_spec),
            ("bs", t["bs"] == builder.kv_cache_spec.block_size),
            ("nsb", t["nsb"] == builder.kv_cache_spec.num_speculative_blocks),
            ("mode", t["mode"] == mode),
            ("cpl", common_prefix_len == t["cpl"]),
            ("fast", fast_build == t["fast"]),
        )
        why = next((n for n, ok in checks if not ok), None)
        if why is not None:
            d = STATS.setdefault("md_reuse_miss", {})
            d[why] = d.get(why, 0) + 1
    if t is not None and why is None:
        md = _derive(builder, t, m, num_decode_draft_tokens_cpu)
        STATS["md_reused"] = STATS.get("md_reused", 0) + 1
        if MDCHECK[0] > 0:
            MDCHECK[0] -= 1
            full = full_build(
                common_prefix_len,
                common_attn_metadata,
                num_accepted_tokens,
                num_decode_draft_tokens_cpu,
                fast_build,
            )
            bad = _md_equal(md, full)
            STATS["md_checked"] = STATS.get("md_checked", 0) + 1
            if bad:
                STATS["md_check_fail"] = STATS.get("md_check_fail", 0) + 1
                logger.warning(
                    "GDN metadata reuse check mismatch fields=%s; using the full build",
                    bad,
                )
                return full
        return md
    md = full_build(
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens,
        num_decode_draft_tokens_cpu,
        fast_build,
    )
    if getattr(md, "num_prefills", 0) > 0:
        tpl = copy.copy(md)
        for k in _FLAG_KEYS:
            tpl.__dict__.pop(k, None)
        if "_step_plan_lazy" in tpl.__dict__:
            tpl.__dict__["_step_plan_lazy"] = list(tpl.__dict__["_step_plan_lazy"])
        _MDTPL[0] = dict(
            md=tpl,
            builder=builder,
            qsl_cpu=m.query_start_loc_cpu,
            qsl=m.query_start_loc,
            seq_lens=m.seq_lens,
            nacc=_base(num_accepted_tokens),
            ndd=_base(num_decode_draft_tokens_cpu),
            nreq=m.num_reqs,
            nat=m.num_actual_tokens,
            num_spec=builder.num_spec,
            bs=builder.kv_cache_spec.block_size,
            nsb=builder.kv_cache_spec.num_speculative_blocks,
            mode=mode,
            cpl=common_prefix_len,
            fast=fast_build,
        )
    else:
        _MDTPL[0] = None
    return md
