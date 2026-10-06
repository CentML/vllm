# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V-split non-CP GDN chunked prefill with a device-side valid-sequence count.

Copy of vllm.third_party.flashinfer_gdn_vsplit whose kernel additionally takes
``device_nb=True``: the number of valid sequences is read on device from
``cu_seqlens[B + 1]`` (one int32 past the logical end of cu_seqlens), and the
tile scheduler only enumerates the tiles of those sequences. Padded
(zero-length) sequences then cost nothing and nb == 0 is an empty launch, which
lets a CUDA graph capture the launch once for a padded batch. Tiles of valid
sequences keep their linear index (same CTA, same math: bitwise identical).
``chunk_gated_delta_rule_vsplit`` also accepts a caller-owned ``workspace``.
Used only by the GDN layer graphs (gdn_layer_graphs, VLLM_GDN_LAYER_GRAPHS_DEVICE_NB=1);
a separate copy keeps the regular V-split kernel source unchanged.

V-split non-CP GDN chunked prefill (patched copy of a FlashInfer kernel).

This package vendors a PATCHED COPY of FlashInfer's Blackwell (SM100-family) CuTe-DSL
chunked Gated DeltaNet prefill kernel and its persistent tile scheduler:

- ``gdn_chunked_vs.py``: ``flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py``
  (as shipped in FlashInfer 0.6.18.post1) plus a ``v_split`` option;
- ``gdn_sched_vs.py``: ``flashinfer/gdn_kernels/blackwell/gated_delta_net_tile_scheduler.py``
  plus the matching ``v_slice`` tile coordinate;
- ``adapter.py``: host launcher adapted from FlashInfer's
  ``gdn_kernels/blackwell/gdn_prefill.py`` (``chunk_gated_delta_rule_sm100``).

The proper home of this change is FlashInfer (Apache-2.0 project; the two kernel files keep
their upstream license headers). Once FlashInfer exposes ``v_split`` (or picks it itself),
this package should be deleted and the call site switched to FlashInfer.

The non-CP FlashInfer kernel runs one persistent CTA per (sequence, value head), so a step
with few prefill sequences under-fills the GPU. With ``v_split=2`` the work item becomes
(sequence, value head, 64-row V slice) and the state-side tcgen05 GEMMs run as M=64 tiles.
Output and final state are bitwise identical to the stock kernel: the V rows of the state
evolve independently and the K-reduction order is unchanged.

Dispatch (used by qwen_gdn_linear_attn.py on the non-CP state-pool path only):
  VLLM_GDN_FI_VSPLIT=1                  enable
  VLLM_GDN_FI_VSPLIT_RULE=r2|model|n<=K default "r2" = "model" + always split when the
                                        stock grid (num_seqs * HV) exceeds the SM count;
                                        "model" picks v_split with a persistent-schedule
                                        cost model from (num_seqs, total tokens, max seq
                                        len); "n<=3" = split iff num_seqs <= 3.
Anything else (and any failure) falls back to the stock FlashInfer call.
"""
import os

from .adapter import _num_sm, chunk_gated_delta_rule_vsplit  # noqa: F401

VSPLIT_RULE = os.environ.get("VLLM_GDN_FI_VSPLIT_RULE", "r2").strip().lower()

# Per-tile cost model (fixed + per-token term) fitted on sm_107 single-sequence launches.
_COEF = {1: (27.0, 0.0217), 2: (9.0, 0.0191)}


def _est(vs: int, lens, hv: int, num_sm: int) -> float:
    a, b = _COEF[vs]
    tiles = [a + b * L for L in lens for _ in range(hv * vs)]
    g = min(len(tiles), num_sm)
    per = [0.0] * g
    for i, t in enumerate(tiles):
        per[i % g] += t
    return max(per)


def _device_num_sm() -> int:
    import torch

    return _num_sm(torch.cuda.current_device())


def choose_vsplit(num_seqs: int, num_tokens: int, max_len: int, hv: int = 32,
                  num_sm: int | None = None) -> int:
    """v_split for a non-CP prefill step (1 = stock kernel). Only host-side ints, no sync.

    num_sm defaults to the SM count of the current CUDA device."""
    if num_seqs <= 0:
        return 1
    if VSPLIT_RULE.startswith("n<="):
        return 2 if num_seqs <= int(VSPLIT_RULE[3:]) else 1
    if num_sm is None:
        num_sm = _device_num_sm()
    if num_seqs * hv * 2 <= num_sm:
        return 2
    # r2: when the stock grid already needs more than one persistent wave (num_seqs*hv > SMs),
    # v_split=2 halves the per-tile fixed cost and was faster on every such batch measured,
    # except some near-balanced batches of 4-6 sequences; those still go through the cost
    # model below.
    if VSPLIT_RULE == "r2" and num_seqs * hv > num_sm:
        return 2
    if max_len <= 0:
        return 1
    rest = max(num_tokens - max_len, 0) / max(num_seqs - 1, 1)
    lens = [max_len] + [rest] * (num_seqs - 1)
    return 2 if _est(2, lens, hv, num_sm) < 0.97 * _est(1, lens, hv, num_sm) else 1
