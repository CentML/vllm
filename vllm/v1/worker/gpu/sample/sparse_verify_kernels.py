# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# fmt: off
"""Kernels for sparse processed-logits sampling in the spec-decode verify step.

Used by ``sparse_verify.py`` (opt-in). Per verify step with B logits rows the
split-row chain (fused prep with sub-chunk maxima, threshold, gather, select,
apply) writes the full fp32 processed-logits row, then the rejection sampler
reads full rows again for its per-block statistics and the resample step.

Here every value any consumer reads stays bit-identical, but chunks that are
entirely masked ("dead": chunk max below the top-k lower bound on a
non-overflow row) are never materialised:
  prep     same bf16 load and penalty math, stores ONLY the chunk and sub-chunk
           maxima (no fp32 row write)
  thresh / select   the sub-chunk threshold kernels, unchanged
  gather   dead chunks: no work; live chunks: the penalised values are
           recomputed with the same code
  emit     live chunks: writes where(key >= thr, x, -inf) exactly as the apply
           pass would; overflow rows: the full row exactly as the regular
           chain leaves it for the fallback; dead chunks: nothing, plus
           LIVE[row, chunk] = 0. If a row's draft token (draft_sampled[row + 1],
           the only element the rejection kernel reads) falls in a dead chunk,
           that single element is written as -inf.
  stats    rejection-sampler statistics kernel with the load masked by chunk
           liveness (dead lanes read as other=-inf, the value the regular chain
           stored there); fully dead blocks skip the load entirely.
  resample regular resample math; a fully dead block returns
           tl.max(full(-inf), return_indices=True), which is what the regular
           kernel computes on an all -inf block (-inf + finite Gumbel noise =
           -inf), without the hash.
Everything else (rejection, insert, fallback) is the regular kernel on the same
tensor.

Requires the sub-chunk top-k/top-p kernels module
(``vllm.v1.worker.gpu.sample.topk_topp_subchunk``).
"""
import os

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample import topk_topp_subchunk as _subchunk
from vllm.v1.worker.gpu.sample.topk_topp_subchunk import (  # jit helpers (Triton resolves module globals)
    _make_key,
    _store_maxima,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (  # jit helpers
    _compute_max_and_sumexp,
    _seeded_resample_argmax,
)

_CHUNK = _subchunk._CHUNK
_SUB = _subchunk._SUB
_CAP = _subchunk._CAP
FAST_TOPK_KMAX = _subchunk.FAST_TOPK_KMAX

_BUF: dict = {}


@triton.jit
def _penalize(logits, block, mask, token_idx, c,
              expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr,
              repetition_penalty_ptr, frequency_penalty_ptr, presence_penalty_ptr,
              prompt_bin_mask_ptr, prompt_bin_mask_stride,
              output_bin_counts_ptr, output_bin_counts_stride,
              VOCAB: tl.constexpr, CHUNK: tl.constexpr, USE_PENALTY: tl.constexpr):
    """Verbatim penalty body of the fused sampler prep kernels."""
    if USE_PENALTY:
        req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
        rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
        freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
        pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)
        use_rep_penalty = rep_penalty != 1.0
        use_freq_penalty = freq_penalty != 0.0
        use_pres_penalty = pres_penalty != 0.0
        if use_rep_penalty or use_freq_penalty or use_pres_penalty:
            output_bin_counts = tl.load(
                output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
                mask=mask, other=0)
            pos = tl.load(expanded_local_pos_ptr + token_idx)
            start_idx = token_idx - pos
            for prev_pos in tl.range(pos):
                prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
                output_bin_counts = output_bin_counts + (block == prev_token).to(tl.int32)
            output_bin_mask = output_bin_counts > 0
            if use_rep_penalty:
                packed_block = c * CHUNK // 32 + tl.arange(0, CHUNK // 32)
                packed_mask = tl.load(
                    prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
                    mask=packed_block < tl.cdiv(VOCAB, 32), other=0)
                prompt_bin_mask = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
                prompt_bin_mask = prompt_bin_mask.to(tl.int1).reshape(CHUNK)
                scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
                logits *= tl.where(logits > 0, 1.0 / scale, scale)
            logits -= freq_penalty * output_bin_counts
            logits -= pres_penalty * output_bin_mask
    return logits


@triton.jit
def _sps_prep_kernel(
    LOGITS_IN, in_stride, CMAX, SMAX, CNT,
    expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr,
    repetition_penalty_ptr, frequency_penalty_ptr, presence_penalty_ptr,
    prompt_bin_mask_ptr, prompt_bin_mask_stride,
    output_bin_counts_ptr, output_bin_counts_stride,
    VOCAB: tl.constexpr, CHUNK: tl.constexpr, NCH_PAD: tl.constexpr,
    NSUB_PAD: tl.constexpr, SUB: tl.constexpr, USE_PENALTY: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    block = c * CHUNK + tl.arange(0, CHUNK)
    mask = block < VOCAB
    logits = tl.load(LOGITS_IN + token_idx * in_stride + block, mask=mask,
                     other=-float("inf")).to(tl.float32)
    logits = _penalize(logits, block, mask, token_idx, c,
                       expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr,
                       repetition_penalty_ptr, frequency_penalty_ptr, presence_penalty_ptr,
                       prompt_bin_mask_ptr, prompt_bin_mask_stride,
                       output_bin_counts_ptr, output_bin_counts_stride, VOCAB, CHUNK, USE_PENALTY)
    _store_maxima(tl.where(mask, logits, -float("inf")), CMAX, SMAX, token_idx, c,
                       NCH_PAD, NSUB_PAD, CHUNK, SUB)
    if c == 0:
        tl.store(CNT + token_idx, 0)


def _geom(V):
    return _subchunk._geom(V)


def _buffers(device, batch, V):
    nch, nch_pad, nsub, nsub_pad = _geom(V)
    b = _BUF.get(device)
    if b is None or b["live"].shape[0] < batch or b["V"] != V:
        cap_b = max(batch, 256)
        b = dict(V=V, live=torch.empty((cap_b, nch_pad), dtype=torch.int8, device=device))
        _BUF[device] = b
    return b


@triton.jit
def _sps_stats2_kernel(
    target_local_max_ptr, target_local_max_stride,
    target_local_sumexp_ptr, target_local_sumexp_stride,
    target_logits_ptr, target_logits_stride,
    LIVE, live_stride,
    expanded_local_pos_ptr,
    vocab_size, num_speculative_steps,
    NB: tl.constexpr, NB_PAD: tl.constexpr, BLOCK_SIZE: tl.constexpr, CHUNK: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    if draft_step_idx >= num_speculative_steps:
        return
    CPB: tl.constexpr = BLOCK_SIZE // CHUNK
    b = tl.arange(0, NB_PAD)
    anyl = tl.zeros([NB_PAD], tl.int32)
    for q in tl.static_range(CPB):
        cq = b * CPB + q
        anyl = anyl | tl.load(LIVE + logit_idx * live_stride + cq, mask=(b < NB) & (cq * CHUNK < vocab_size),
                              other=0).to(tl.int32)
    bl = (anyl != 0) & (b < NB)
    dead = (b < NB) & (anyl == 0)
    tl.store(target_local_max_ptr + logit_idx * target_local_max_stride + b,
             tl.full([NB_PAD], float("-inf"), tl.float32), mask=dead)
    tl.store(target_local_sumexp_ptr + logit_idx * target_local_sumexp_stride + b,
             tl.zeros([NB_PAD], tl.float32), mask=dead)
    nl = tl.sum(bl.to(tl.int32), axis=0)
    cur = tl.sum(tl.zeros([2], tl.int32), axis=0) - 1
    for _it in tl.range(nl):
        block_idx = tl.min(tl.where(bl & (b > cur), b, NB_PAD), axis=0)
        cur = block_idx
        ci = block_idx * CPB + tl.arange(0, CPB)
        lv = tl.load(LIVE + logit_idx * live_stride + ci, mask=ci * CHUNK < vocab_size, other=0)
        block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < vocab_size
        lane_live = tl.reshape(tl.broadcast_to(lv[:, None], (CPB, CHUNK)), (BLOCK_SIZE,)) != 0
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask & lane_live,
            other=float("-inf"),
        ).to(tl.float32)
        target_max, target_sumexp = _compute_max_and_sumexp(target_logits)
        tl.store(target_local_max_ptr + logit_idx * target_local_max_stride + block_idx, target_max)
        tl.store(target_local_sumexp_ptr + logit_idx * target_local_sumexp_stride + block_idx, target_sumexp)


@triton.jit
def _sps_resample4_kernel(
    resampled_local_argmax_ptr, resampled_local_argmax_stride,
    resampled_local_max_ptr, resampled_local_max_stride,
    target_logits_ptr, target_logits_stride,
    LIVE, live_stride,
    rejected_step_ptr, cu_num_logits_ptr, expanded_idx_mapping_ptr, draft_sampled_ptr,
    temp_ptr, seed_ptr, pos_ptr,
    vocab_size,
    NBLK: tl.constexpr, NBLK_PAD: tl.constexpr, BLOCK_SIZE: tl.constexpr, CHUNK: tl.constexpr,
    SPLIT: tl.constexpr, USE_FP64: tl.constexpr,
):
    req_idx = tl.program_id(0)
    split = tl.program_id(1)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        return
    b = tl.arange(0, NBLK_PAD)
    lvb = tl.load(LIVE + resample_token_idx * live_stride + (b * BLOCK_SIZE) // CHUNK, mask=b < NBLK, other=0) != 0
    mine = (b % SPLIT) == split
    dead = (b < NBLK) & (~lvb) & mine
    if USE_FP64:
        neg = tl.full([BLOCK_SIZE], float("-inf"), tl.float64)
        zv = tl.zeros([NBLK_PAD], tl.float64)
    else:
        neg = tl.full([BLOCK_SIZE], float("-inf"), tl.float32)
        zv = tl.zeros([NBLK_PAD], tl.float32)
    v_dead, i_dead = tl.max(neg, axis=0, return_indices=True)
    tl.store(resampled_local_argmax_ptr + req_idx * resampled_local_argmax_stride + b, b * BLOCK_SIZE + i_dead,
             mask=dead)
    tl.store(resampled_local_max_ptr + req_idx * resampled_local_max_stride + b, zv + v_dead, mask=dead)
    live = (b < NBLK) & lvb & mine
    nl = tl.sum(live.to(tl.int32), axis=0)
    rejected_draft_token = tl.load(draft_sampled_ptr + resample_token_idx + 1, mask=not is_bonus, other=0)
    is_valid_rejected_draft = rejected_draft_token >= 0
    cur = tl.sum(tl.zeros([2], tl.int32), axis=0) - 1
    for _it in tl.range(nl):
        block_idx = tl.min(tl.where(live & (b > cur), b, NBLK_PAD), axis=0)
        cur = block_idx
        block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        target_logits = tl.load(
            target_logits_ptr + resample_token_idx * target_logits_stride + block,
            mask=mask, other=float("-inf"),
        ).to(tl.float32)
        if is_bonus or not is_valid_rejected_draft:
            residual_logits = target_logits
        else:
            residual_logits = tl.where(block != rejected_draft_token, target_logits, float("-inf")).to(tl.float32)
        value, idx = _seeded_resample_argmax(
            residual_logits, block, mask, resample_token_idx, expanded_idx_mapping_ptr,
            temp_ptr, seed_ptr, pos_ptr, vocab_size, USE_FP64=USE_FP64)
        token_id = block_idx * BLOCK_SIZE + idx
        tl.store(resampled_local_argmax_ptr + req_idx * resampled_local_argmax_stride + block_idx, token_id)
        tl.store(resampled_local_max_ptr + req_idx * resampled_local_max_stride + block_idx, value)


_BITS: dict = {}

_RS_SPLIT = int(os.environ.get("VLLM_SPARSE_VERIFY_RS_SPLIT", "8"))


def rejection_sample_sparse4(target_logits, live, draft_sampled, cu_num_logits, pos, idx_mapping,
                             expanded_idx_mapping, expanded_local_pos, temperature, seed, num_speculative_steps,
                             use_fp64=False):
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as R

    assert target_logits.ndim == 2 and target_logits.stride(-1) == 1
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    target_local_argmax = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.int64)
    target_local_max = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
    target_local_sumexp = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
    draft_local_max = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
    draft_local_sumexp = target_logits.new_empty(num_logits, vocab_num_blocks, dtype=torch.float32)
    _sps_stats2_kernel[(num_logits,)](
        target_local_max, target_local_max.stride(0),
        target_local_sumexp, target_local_sumexp.stride(0),
        target_logits, target_logits.stride(0),
        live, live.stride(0), expanded_local_pos,
        vocab_size, num_speculative_steps,
        NB=vocab_num_blocks, NB_PAD=padded_vocab_num_blocks, BLOCK_SIZE=VOCAB_BLOCK_SIZE, CHUNK=_CHUNK,
    )
    sampled = draft_sampled.new_empty(num_reqs, num_speculative_steps + 1, dtype=torch.int64)
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    R._rejection_kernel[(num_reqs,)](
        sampled, sampled.stride(0), num_sampled,
        target_rejected_logsumexp, draft_rejected_logsumexp,
        target_logits, target_logits.stride(0),
        target_local_argmax, target_local_argmax.stride(0),
        target_local_max, target_local_max.stride(0),
        target_local_sumexp, target_local_sumexp.stride(0),
        draft_sampled,
        None, 0, 0,
        draft_local_max, draft_local_max.stride(0),
        draft_local_sumexp, draft_local_sumexp.stride(0),
        cu_num_logits, idx_mapping, temperature, seed, pos,
        None, None, None, 0,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=False, SYNTHETIC_MODE=False, USE_BLOCK_VERIFICATION=False,
        num_warps=1,
    )
    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits.new_empty(num_reqs, resample_num_blocks, dtype=torch.int64)
    resampled_local_max = target_logits.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.float64 if use_fp64 else torch.float32)
    _sps_resample4_kernel[(num_reqs, _RS_SPLIT)](
        resampled_local_argmax, resampled_local_argmax.stride(0),
        resampled_local_max, resampled_local_max.stride(0),
        target_logits, target_logits.stride(0),
        live, live.stride(0),
        num_sampled, cu_num_logits, expanded_idx_mapping, draft_sampled,
        temperature, seed, pos,
        vocab_size,
        NBLK=resample_num_blocks, NBLK_PAD=padded_resample_num_blocks, BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        CHUNK=_CHUNK, SPLIT=_RS_SPLIT, USE_FP64=use_fp64,
    )
    R._insert_resampled_kernel[(num_reqs,)](
        sampled, sampled.stride(0), num_sampled,
        resampled_local_argmax, resampled_local_argmax.stride(0),
        resampled_local_max, resampled_local_max.stride(0),
        resample_num_blocks, cu_num_logits, expanded_idx_mapping, temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
    )
    return sampled, num_sampled


@triton.jit
def _pen_bits(x, offs, token_idx, c, OBITS, obits_stride,
              expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr, presence_penalty_ptr,
              CHUNK: tl.constexpr, MAXPOS: tl.constexpr):
    """== stock penalty math when repetition_penalty == 1.0 and frequency_penalty == +0.0 (host-checked):
    x - (+0.0 * counts) == x bitwise, so only `x -= pres * ((counts + in-flight) > 0)` remains; counts > 0 comes
    from the per-request-slot bitmask (CHUNK/32 words per chunk)."""
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)
    if pres_penalty != 0.0:
        W: tl.constexpr = CHUNK // 32
        words = tl.load(OBITS + req_state_idx.to(tl.int64) * obits_stride + c * W + tl.arange(0, W))
        ob = ((words[:, None] >> (tl.arange(0, 32)[None, :])) & 1).to(tl.int1).reshape(CHUNK)
        pos = tl.load(expanded_local_pos_ptr + token_idx)
        start_idx = token_idx - pos
        for prev_pos in tl.range(pos):
            prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
            ob = ob | (offs == prev_token)
        x -= pres_penalty * ob
    return x


@triton.jit
def _sps_obits5_kernel(OBITS, obits_stride, CU, expanded_idx_mapping_ptr,
                       output_bin_counts_ptr, output_bin_counts_stride,
                       VOCAB: tl.constexpr, CHUNK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1)
    start = tl.load(CU + r)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + start).to(tl.int64)
    block = c * CHUNK + tl.arange(0, CHUNK)
    cnt = tl.load(output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block, mask=block < VOCAB,
                  other=0)
    b = (cnt > 0).to(tl.int32)
    W: tl.constexpr = CHUNK // 32
    words = tl.sum(tl.reshape(b, (W, 32)) << tl.arange(0, 32)[None, :], axis=1)
    tl.store(OBITS + req_state_idx * obits_stride + c * W + tl.arange(0, W), words)


@triton.jit
def _sps_prep5_kernel(
    LOGITS_IN, in_stride, CMAX, SMAX, CNT, OBITS, obits_stride,
    expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr, presence_penalty_ptr,
    VOCAB: tl.constexpr, CHUNK: tl.constexpr, NCH_PAD: tl.constexpr,
    NSUB_PAD: tl.constexpr, SUB: tl.constexpr, MAXPOS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    block = c * CHUNK + tl.arange(0, CHUNK)
    mask = block < VOCAB
    logits = tl.load(LOGITS_IN + token_idx * in_stride + block, mask=mask,
                     other=-float("inf")).to(tl.float32)
    logits = _pen_bits(logits, block, token_idx, c, OBITS, obits_stride, expanded_idx_mapping_ptr, token_ids_ptr,
                       expanded_local_pos_ptr, presence_penalty_ptr, CHUNK, MAXPOS)
    _store_maxima(tl.where(mask, logits, -float("inf")), CMAX, SMAX, token_idx, c,
                  NCH_PAD, NSUB_PAD, CHUNK, SUB)
    if c == 0:
        tl.store(CNT + token_idx, 0)


@triton.jit
def _chunk_vals(LOGITS_IN, in_stride, token_idx, c, OBITS, obits_stride,
                expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr,
                repetition_penalty_ptr, frequency_penalty_ptr, presence_penalty_ptr,
                prompt_bin_mask_ptr, prompt_bin_mask_stride,
                output_bin_counts_ptr, output_bin_counts_stride,
                VOCAB: tl.constexpr, CHUNK: tl.constexpr, USE_PENALTY: tl.constexpr, BITS: tl.constexpr,
                MAXPOS: tl.constexpr):
    offs = c * CHUNK + tl.arange(0, CHUNK)
    inb = offs < VOCAB
    x = tl.load(LOGITS_IN + token_idx * in_stride + offs, mask=inb, other=-float("inf")).to(tl.float32)
    if BITS:
        x = _pen_bits(x, offs, token_idx, c, OBITS, obits_stride, expanded_idx_mapping_ptr, token_ids_ptr,
                      expanded_local_pos_ptr, presence_penalty_ptr, CHUNK, MAXPOS)
    else:
        x = _penalize(x, offs, inb, token_idx, c,
                      expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr,
                      repetition_penalty_ptr, frequency_penalty_ptr, presence_penalty_ptr,
                      prompt_bin_mask_ptr, prompt_bin_mask_stride,
                      output_bin_counts_ptr, output_bin_counts_stride, VOCAB, CHUNK, USE_PENALTY)
    return x, offs, inb


@triton.jit
def _sps_gather5_kernel(
    LOGITS_IN, in_stride, CMAX, CNT, CAND, LBUF, OBITS, obits_stride,
    expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr,
    repetition_penalty_ptr, frequency_penalty_ptr, presence_penalty_ptr,
    prompt_bin_mask_ptr, prompt_bin_mask_stride,
    output_bin_counts_ptr, output_bin_counts_stride,
    VOCAB: tl.constexpr, CHUNK: tl.constexpr, NCH: tl.constexpr, NCH_PAD: tl.constexpr, CAP: tl.constexpr,
    USE_PENALTY: tl.constexpr, BITS: tl.constexpr, MAXPOS: tl.constexpr, SLOTS: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.program_id(1)
    token_idx = row.to(tl.int64)
    L = tl.load(LBUF + row)
    ci = tl.arange(0, NCH_PAD)
    cm = tl.load(CMAX + row * NCH_PAD + ci, mask=ci < NCH, other=-float("inf"))
    livev = (cm >= L) & (ci < NCH)
    rank = tl.cumsum(livev.to(tl.int32), axis=0) - 1
    mine = livev & ((rank % SLOTS) == slot)
    nm = tl.sum(mine.to(tl.int32), axis=0)
    cur = tl.sum(tl.zeros([2], tl.int32), axis=0) - 1
    for _it in tl.range(nm):
        c = tl.min(tl.where(mine & (ci > cur), ci, NCH_PAD), axis=0)
        cur = c
        x, offs, inb = _chunk_vals(LOGITS_IN, in_stride, token_idx, c, OBITS, obits_stride,
                                   expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr,
                                   repetition_penalty_ptr, frequency_penalty_ptr, presence_penalty_ptr,
                                   prompt_bin_mask_ptr, prompt_bin_mask_stride,
                                   output_bin_counts_ptr, output_bin_counts_stride, VOCAB, CHUNK, USE_PENALTY,
                                   BITS, MAXPOS)
        x = tl.where(inb, x, -float("inf"))
        sel = (x >= L) & inb
        s = tl.sum(sel.to(tl.int32), axis=0)
        base = tl.atomic_add(CNT + row, s)
        pos = base + tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(CAND + row * CAP + pos, _make_key(x, offs), mask=sel & (pos < CAP))


@triton.jit
def _sps_emit5_kernel(
    LOGITS_IN, in_stride, OUT, out_stride, CMAX, LBUF, THR, CNT, LIVE, DRAFT, B, OBITS, obits_stride,
    expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr,
    repetition_penalty_ptr, frequency_penalty_ptr, presence_penalty_ptr,
    prompt_bin_mask_ptr, prompt_bin_mask_stride,
    output_bin_counts_ptr, output_bin_counts_stride,
    VOCAB: tl.constexpr, CHUNK: tl.constexpr, NCH: tl.constexpr, NCH_PAD: tl.constexpr, CAP: tl.constexpr,
    MASK_VALUE: tl.constexpr, USE_PENALTY: tl.constexpr, BITS: tl.constexpr, MAXPOS: tl.constexpr,
    SLOTS: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.program_id(1)
    token_idx = row.to(tl.int64)
    L = tl.load(LBUF + row)
    n = tl.load(CNT + row)
    ovf = n > CAP
    ci = tl.arange(0, NCH_PAD)
    cm = tl.load(CMAX + row * NCH_PAD + ci, mask=ci < NCH, other=-float("inf"))
    livev = (cm >= L) & (ci < NCH)
    todo = livev | (ovf & (ci < NCH))
    if slot == 0:
        tl.store(LIVE + row * NCH_PAD + ci, todo.to(tl.int8))
    rank = tl.cumsum(todo.to(tl.int32), axis=0) - 1
    mine = todo & ((rank % SLOTS) == slot)
    nm = tl.sum(mine.to(tl.int32), axis=0)
    thr = tl.load(THR + row)
    cur = tl.sum(tl.zeros([2], tl.int32), axis=0) - 1
    for _it in tl.range(nm):
        c = tl.min(tl.where(mine & (ci > cur), ci, NCH_PAD), axis=0)
        cur = c
        lv = tl.load(CMAX + row * NCH_PAD + c) >= L
        offs = c * CHUNK + tl.arange(0, CHUNK)
        inb = offs < VOCAB
        optr = OUT + token_idx * out_stride + offs
        if lv:
            x, o2, i2 = _chunk_vals(LOGITS_IN, in_stride, token_idx, c, OBITS, obits_stride,
                                    expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr,
                                    repetition_penalty_ptr, frequency_penalty_ptr, presence_penalty_ptr,
                                    prompt_bin_mask_ptr, prompt_bin_mask_stride,
                                    output_bin_counts_ptr, output_bin_counts_stride, VOCAB, CHUNK, USE_PENALTY,
                                    BITS, MAXPOS)
            if ovf:
                tl.store(optr, x, mask=inb)
            else:
                xk = tl.where(inb, x, -float("inf"))
                keep = _make_key(xk, offs) >= thr
                tl.store(optr, tl.where(keep, x, MASK_VALUE), mask=inb)
        else:
            tl.store(optr, tl.full([CHUNK], MASK_VALUE, tl.float32), mask=inb)
    if (slot == 0) and (not ovf) and (row + 1 < B):
        t = tl.maximum(tl.load(DRAFT + row + 1).to(tl.int64), 0)
        if tl.load(CMAX + row * NCH_PAD + t // CHUNK) < L:
            tl.store(OUT + token_idx * out_stride + t, MASK_VALUE)


_SLOTS = int(os.environ.get("VLLM_SPARSE_VERIFY_SLOTS", "4"))
_GSLOTS = int(os.environ.get("VLLM_SPARSE_VERIFY_GATHER_SLOTS", "4"))
_MAXPOS = 8


def sparse_prep_topk_topp5(logits_in, penalties, expanded_idx_mapping, input_ids, expanded_local_pos,
                           k, p, kmax, cu_num_logits, pres_only, mask_value=float("-inf")):
    """pres_only: host-verified (all rows) repetition_penalty == 1.0 and frequency_penalty is +0.0; every row's
    expanded_local_pos < 8.
    """
    assert mask_value == float("-inf")
    assert 1 <= kmax <= FAST_TOPK_KMAX
    B, V = logits_in.shape
    dev = logits_in.device
    out = torch.empty((B, V), dtype=torch.float32, device=dev)
    nch, nch_pad, nsub, nsub_pad = _geom(V)
    assert nch >= kmax
    buf = _subchunk._buffers(dev, B, V)
    live = _buffers(dev, B, V)["live"]
    use = penalties is not None
    bits = use and pres_only
    dummy = buf["cnt"]
    pen = (
        expanded_idx_mapping if use else dummy, input_ids if use else dummy,
        expanded_local_pos if use else dummy,
        penalties.repetition_penalty.gpu if use else dummy,
        penalties.frequency_penalty.gpu if use else dummy,
        penalties.presence_penalty.gpu if use else dummy,
        penalties.prompt_bin_mask if use else dummy,
        penalties.prompt_bin_mask.stride(0) if use else 0,
        penalties.output_bin_counts if use else dummy,
        penalties.output_bin_counts.stride(0) if use else 0,
    )
    if bits:
        W = nch * (_CHUNK // 32)
        nst = penalties.output_bin_counts.shape[0]
        ob = _BITS.get((dev, "v5"))
        if ob is None or ob.shape[0] < nst or ob.shape[1] != W:
            ob = torch.empty((nst, W), dtype=torch.int32, device=dev)
            _BITS[(dev, "v5")] = ob
        num_reqs = cu_num_logits.shape[0] - 1
        _sps_obits5_kernel[(num_reqs, nch)](ob, ob.stride(0), cu_num_logits, expanded_idx_mapping,
                                            penalties.output_bin_counts, penalties.output_bin_counts.stride(0),
                                            VOCAB=V, CHUNK=_CHUNK, num_warps=8)
        _sps_prep5_kernel[(B, nch)](
            logits_in, logits_in.stride(0), buf["cmax"], buf["smax"], buf["cnt"], ob, ob.stride(0),
            expanded_idx_mapping, input_ids, expanded_local_pos, penalties.presence_penalty.gpu,
            VOCAB=V, CHUNK=_CHUNK, NCH_PAD=nch_pad, NSUB_PAD=nsub_pad, SUB=_SUB, MAXPOS=_MAXPOS, num_warps=8)
    else:
        ob = dummy
        _sps_prep_kernel[(B, nch)](
            logits_in, logits_in.stride(0), buf["cmax"], buf["smax"], buf["cnt"], *pen,
            VOCAB=V, CHUNK=_CHUNK, NCH_PAD=nch_pad, NSUB_PAD=nsub_pad, SUB=_SUB, USE_PENALTY=use, num_warps=8)
    obs = ob.stride(0) if bits else 0
    KM = max(16, triton.next_power_of_2(kmax))
    k32 = k.to(torch.int32)
    p32 = p.to(torch.float32) if p is not None else k32
    _subchunk._thresh_kernel[(B,)](buf["smax"], k32, buf["lbuf"], NSUB=nsub, NSUB_PAD=nsub_pad, num_warps=8)
    _sps_gather5_kernel[(B, _GSLOTS)](
        logits_in, logits_in.stride(0), buf["cmax"], buf["cnt"], buf["cand"], buf["lbuf"], ob, obs, *pen,
        VOCAB=V, CHUNK=_CHUNK, NCH=nch, NCH_PAD=nch_pad, CAP=_CAP, USE_PENALTY=use, BITS=bits, MAXPOS=_MAXPOS,
        SLOTS=_GSLOTS, num_warps=8)
    _subchunk._select_kernel[(B,)](buf["cnt"], buf["cand"], k32, p32, buf["thr"], buf["keff"], buf["peff"],
                              buf["ovf"], VOCAB=V, CAP=_CAP, KMAX=KM, TOPP=p is not None, STATS=False,
                              num_warps=8)
    _sps_emit5_kernel[(B, _SLOTS)](
        logits_in, logits_in.stride(0), out, out.stride(0), buf["cmax"], buf["lbuf"], buf["thr"], buf["cnt"],
        live, input_ids, B, ob, obs, *pen,
        VOCAB=V, CHUNK=_CHUNK, NCH=nch, NCH_PAD=nch_pad, CAP=_CAP, MASK_VALUE=mask_value, USE_PENALTY=use,
        BITS=bits, MAXPOS=_MAXPOS, SLOTS=_SLOTS, num_warps=8)
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

    apply_top_k_top_p_triton(out, buf["keff"][:B], buf["peff"][:B] if p is not None else None, mask_value)
    return out, live[:B]
