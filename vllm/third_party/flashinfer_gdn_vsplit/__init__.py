# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V-split non-CP GDN chunked prefill (patched copy of a FlashInfer kernel).

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

Dispatch (fi_chunk_gated_delta_rule in qwen_gdn_linear_attn.py, non-CP state-pool path):
  VLLM_GDN_FI_VSPLIT=0                  disable (default on; the prefill warmup
                                        compiles the kernel, a failed compile keeps
                                        it off)
  VLLM_GDN_FI_VSPLIT_RULE=r2|model|n<=K default "r2" = "model" + always split when the
                                        stock grid (num_seqs * HV) exceeds the SM count;
                                        "model" picks v_split with a persistent-schedule
                                        cost model from (num_seqs, total tokens, max seq
                                        len); "n<=3" = split iff num_seqs <= 3.
Steps the rule leaves at v_split=1 run the stock FlashInfer call
(VLLM_GDN_FI_VSPLIT_V1=1: this package's kernel at v_split=1).

Optional memory/scheduling variants (all independently default OFF):
  VLLM_GDN_VSPLIT_STAGED_STORE=1       bf16 indexed v_split=2 final state via SMEM
  VLLM_GDN_VSPLIT_STAGED_LOAD=1        initial state via SMEM (requires staged store)
  VLLM_GDN_VSPLIT_VEC_STATE=1          aligned bf16 v_split=1 vector state traffic
  VLLM_GDN_VSPLIT_EARLY_RELINQUISH=1   release the TMEM allocation permit early
  VLLM_GDN_VSPLIT_HEAD_MAJOR=1         CUDA conv allocates head-major g/beta;
                                     explicit caller buffers retain their layout
  VLLM_GDN_VSPLIT_PDL=1                adjacent conv/chunk edge in layer graphs only

These variants retain the arithmetic and state conversions (EXACT design);
bitwise output and complete-state comparison against frozen qwen-v2 is required
on the target GPU before adoption. State transport falls back to stock for
unsupported tile strides, alignment or dtype. VEC_STATE does not change route
selection: VLLM_GDN_FI_VSPLIT_V1=1 is needed for serving's v_split=1 steps.
Head-major gates are made contiguous on the FlashInfer/CP fallback. PDL is not
inferred on the eager path: recurrent/zeroing kernels may intervene there.
No device-nb, step-plan or prebuilt-descriptor implementation is reintroduced.
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
