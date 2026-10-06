# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in sparse processed-logits sampling for the spec-decode verify step.

In the verify step the split-row sampler path (``fused_sampling_fast_path``)
writes the full fp32 processed-logits row for every logits row, and the
rejection sampler then reads full rows again. With top-k active, almost every
chunk of a row is entirely masked. When enabled and eligible, the verify step
instead uses kernels (``sparse_verify_kernels.py``) that never materialise the
fully masked chunks while keeping every value any consumer reads bit-identical,
plus a rejection-sampling variant that skips those chunks.

Eligible (otherwise the regular path runs): the split-row fast path applies,
every temperature is 1.0, min_p is unused, no draft logits (one-hot drafts),
no watermarking, standard (non-block) verification, no synthetic acceptance
rates, no adaptive verification, no processed-logprobs consumer and a single
verify chunk.

Environment:
    VLLM_SPARSE_VERIFY_SAMPLING=1          enable (default off)
    VLLM_SPARSE_VERIFY_SAMPLING_LOG=N      log counters every N verify steps
                                           (default 0 = off)
    VLLM_SPARSE_VERIFY_SLOTS / _GATHER_SLOTS / _RS_SPLIT
                                           kernel launch knobs (default 4/4/8)
"""

import contextlib
import os
import threading
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import triton

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_SPARSE_VERIFY_SAMPLING", "0") == "1"
_LOG_EVERY = int(os.environ.get("VLLM_SPARSE_VERIFY_SAMPLING_LOG", "0"))

STATS = {"verify": 0, "sps": 0, "sps_rs": 0, "ineligible": 0, "pres_only": 0}
_TLS = threading.local()

if ENABLED:
    # Requires the sub-chunk top-k/top-p kernels; fail at start-up if missing.
    from vllm.v1.worker.gpu.sample import sparse_verify_kernels as _kernels
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample as _rejection_sample,
    )

    logger.info_once("sparse verify sampling enabled")


@contextlib.contextmanager
def _verify_state(ok: bool, cu_num_logits: torch.Tensor | None):
    _TLS.ok = ok
    _TLS.stash = None
    _TLS.cu = cu_num_logits if ok else None
    try:
        yield
    finally:
        _TLS.ok = False
        _TLS.stash = None
        _TLS.cu = None
        if _LOG_EVERY and STATS["verify"] % _LOG_EVERY == 0:
            logger.info("sparse verify sampling stats %s", STATS)


def verify_scope(
    rejection_sampler: Any,
    logits: torch.Tensor,
    input_batch: Any,
    draft_logits: torch.Tensor | None,
    max_chunk_logits: int,
    max_num_logprobs: int,
) -> contextlib.AbstractContextManager:
    """Context around ``RejectionSampler._verify_in_chunks``."""
    if not ENABLED:
        return contextlib.nullcontext()
    from vllm.config.model import PROCESSED_LOGPROBS_MODES
    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS

    rs = rejection_sampler
    STATS["verify"] += 1
    ok = (
        draft_logits is None
        and rs.watermark_key is None
        and not rs.use_block_verification
        and rs.synthetic_conditional_rates is None
        and not rs.enable_adaptive_verification
        and logits.shape[0] <= max_chunk_logits
        and (
            max_num_logprobs == NO_LOGPROBS
            or rs.sampler.logprobs_mode not in PROCESSED_LOGPROBS_MODES
        )
    )
    return _verify_state(ok, input_batch.cu_num_logits)


def in_eligible_verify() -> bool:
    return getattr(_TLS, "ok", False)


def sparse_prep(
    states: Any,
    logits: torch.Tensor,
    sampler: Any,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    split_row_enabled: bool,
) -> torch.Tensor | None:
    """Sparse replacement of ``SamplingStates.fused_sampling_fast_path`` in an
    eligible verify step, or None to run the regular path.
    """
    # Identical applicability checks to the split-row fast path, plus: every
    # temperature == 1.0.
    ok = (
        split_row_enabled
        and logits.is_cuda
        and not np.any(sampler.logit_bias_state.use_logit_bias[idx_mapping_np])
        and int(sampler.bad_words_state.num_bad_words.np[idx_mapping_np].max()) == 0
    )
    if ok:
        tb = sampler.thinking_budget_state
        ok = not (tb.enabled and np.any(tb.use_thinking_budget[idx_mapping_np]))
    if ok:
        ok = bool(np.all(states.temperature.np[idx_mapping_np] == 1.0)) and bool(
            np.all(states.min_p.np[idx_mapping_np] == 0.0)
        )
    if ok:
        kmax = int(states.top_k.np[idx_mapping_np].max())
        ok = (
            kmax <= _kernels.FAST_TOPK_KMAX
            and triton.cdiv(logits.shape[1], _kernels._CHUNK) >= kmax
        )
    if ok:
        top_k, top_p = states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
        ok = top_k is not None
    if not ok:
        STATS["ineligible"] += 1
        return None
    pen = sampler.penalties_state
    use_pen = bool(np.any(pen.use_penalty[idx_mapping_np]))
    cu = getattr(_TLS, "cu", None)
    if cu is None or cu.shape[0] - 1 != len(idx_mapping_np):
        STATS["ineligible"] += 1
        return None
    pres_only = False
    if use_pen:
        rep = pen.repetition_penalty.np[idx_mapping_np]
        frq = pen.frequency_penalty.np[idx_mapping_np]
        pres_only = bool(np.all(rep == 1.0)) and bool(np.all(frq.view(np.int32) == 0))
    out, live = _kernels.sparse_prep_topk_topp5(
        logits,
        pen if use_pen else None,
        expanded_idx_mapping,
        input_ids,
        expanded_local_pos,
        top_k,
        top_p,
        kmax,
        cu,
        pres_only,
    )
    if pres_only:
        STATS["pres_only"] += 1
    _TLS.stash = (out, live)
    STATS["sps"] += 1
    return out


def rejection_sample(
    target_logits,
    draft_logits,
    draft_sampled,
    cu_num_logits,
    pos,
    idx_mapping,
    expanded_idx_mapping,
    expanded_local_pos,
    temperature,
    seed,
    num_speculative_steps,
    synthetic_conditional_rates=None,
    use_fp64=False,
    use_block_verification=False,
    contexts=None,
    watermarking=None,
    watermark_key=None,
):
    """``rejection_sample`` with the sparse variant for logits produced by
    ``sparse_prep`` in the same verify step; otherwise the regular function.
    """
    st = getattr(_TLS, "stash", None)
    if (
        st is not None
        and st[0] is target_logits
        and draft_logits is None
        and synthetic_conditional_rates is None
        and not use_block_verification
        and contexts is None
        and watermarking is None
        and watermark_key is None
    ):
        _TLS.stash = None
        STATS["sps_rs"] += 1
        if STATS["sps_rs"] == 1:
            logger.info("sparse verify sampling in use")
        return _kernels.rejection_sample_sparse4(
            target_logits,
            st[1],
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seed,
            num_speculative_steps,
            use_fp64=use_fp64,
        )
    return _rejection_sample(
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps,
        synthetic_conditional_rates,
        use_fp64=use_fp64,
        use_block_verification=use_block_verification,
        contexts=contexts,
        watermarking=watermarking,
        watermark_key=watermark_key,
    )
