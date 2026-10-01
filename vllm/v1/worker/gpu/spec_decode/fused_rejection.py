# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused rejection sampling for top-k-bounded speculative batches.

The unfused path copies the target logits into an fp32 canvas, applies
penalties and temperature in place, masks the canvas to the top-k/top-p
survivors (``spec_topk_topp``) and runs the rejection kernels over the full
canvas. Each of those steps streams [num_logits, vocab] fp32.

When every row has 1 <= top_k <= MAX_TOP_K and only penalties, temperature,
top-k and top-p apply, every consumer after the top-k mask only reads the
survivors. This module reads the raw logits once and keeps the survivors
compact:

1. ``_penalized_submax_kernel`` (num_reqs x V/BLOCK): applies penalties and
   temperature with the unfused kernels' arithmetic and writes the max of every
   SUB-wide sub-block. The logits rows of one request share one load of its
   penalty state.
2. ``_select_survivors_kernel`` (num_logits): picks the KP sub-blocks with the
   largest (max value, lowest index) keys. They hold the row's KP largest
   logits under the (value desc, token id asc) order: if a token outranks the
   KP-th largest logit, its sub-block's max key outranks it too, and there are
   only KP such keys. It recomputes the processed logits of those sub-blocks,
   keeps exactly top_k of them by that order, applies top-p and writes the
   survivors (value, token id) and their logsumexp.
3. ``_compact_rejection_kernel`` (num_reqs): the probability-ratio test
   against one-hot drafts, the resample with the same per-token Gumbel noise
   and tie rule as ``_resample_kernel``, the insertion and the
   num_sampled/num_rejected bookkeeping of ``_insert_resampled_kernel``.

Survivor sets, processed logit values, uniforms and Gumbel noise are bitwise
those of the unfused path. The logsumexp is summed over the compact survivors
instead of per-8192-token block partials, so the acceptance log-probability
can differ in the last fp32 bits.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_logits, tl_rand32
from vllm.v1.worker.gpu.sample.spec_topk_topp import (
    MAX_TOP_K,
    _pack_keys,
    _top_k_pow2,
    _unpack_keys,
)

# Vocab block per program of the first kernel and the sub-block granularity of
# the survivor search. The second kernel loads KP sub-blocks (KP * SUB logits).
_BLOCK_SIZE = 4096
_SUB_SIZE = 128


@triton.jit
def _penalize(
    logits,
    output_bin_counts,
    prompt_bin_mask,
    rep_penalty,
    freq_penalty,
    pres_penalty,
    use_rep_penalty,
):
    # Same arithmetic, in the same order, as _penalties_kernel.
    output_bin_mask = output_bin_counts > 0
    if use_rep_penalty:
        scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
        logits *= tl.where(logits > 0, 1.0 / scale, scale)
    logits -= freq_penalty * output_bin_counts
    logits -= pres_penalty * output_bin_mask
    return logits


@triton.jit
def _penalized_submax_kernel(
    # [num_logits, submax_stride] fp32
    submax_ptr,
    submax_stride,
    # [num_logits, V]
    logits_ptr,
    logits_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [num_tokens]
    input_ids_ptr,
    # [num_logits]
    logits_indices_ptr,
    # [max_num_reqs]
    temperature_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    # [max_num_reqs, cdiv(V, 32)]
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    # [max_num_reqs, V]
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    SUB_SIZE: tl.constexpr,
    HAS_PENALTIES: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    temp = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    NUM_SUB: tl.constexpr = BLOCK_SIZE // SUB_SIZE
    sub_offs = block_idx * NUM_SUB + tl.arange(0, NUM_SUB)

    use_penalty = False
    use_rep_penalty = False
    rep_penalty = 1.0
    freq_penalty = 0.0
    pres_penalty = 0.0
    output_bin_counts = tl.zeros((BLOCK_SIZE,), tl.int32)
    prompt_bin_mask = tl.zeros((BLOCK_SIZE,), tl.int1)
    if HAS_PENALTIES:
        rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
        freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
        pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)
        use_rep_penalty = rep_penalty != 1.0
        use_penalty = use_rep_penalty or freq_penalty != 0.0 or pres_penalty != 0.0
        if use_penalty:
            output_bin_counts = tl.load(
                output_bin_counts_ptr
                + req_state_idx * output_bin_counts_stride
                + block,
                mask=mask,
                other=0,
            )
        if use_rep_penalty:
            packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
            packed_mask = tl.load(
                prompt_bin_mask_ptr
                + req_state_idx * prompt_bin_mask_stride
                + packed_block,
                mask=packed_block < tl.cdiv(vocab_size, 32),
                other=0,
            )
            bits = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
            prompt_bin_mask = bits.to(tl.int1).reshape(BLOCK_SIZE)

    for row in range(start, end):
        # Row start + p also counts the draft tokens at positions 1..p of the
        # request, as _penalties_kernel does.
        if HAS_PENALTIES and row > start:
            token_idx = tl.load(logits_indices_ptr + row)
            prev_token = tl.load(input_ids_ptr + token_idx)
            output_bin_counts += (block == prev_token).to(tl.int32)
        logits = tl.load(
            logits_ptr + row * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        if use_penalty:
            logits = _penalize(
                logits,
                output_bin_counts,
                prompt_bin_mask,
                rep_penalty,
                freq_penalty,
                pres_penalty,
                use_rep_penalty,
            )
        if temp != 0.0 and temp != 1.0:
            logits = logits / temp
        logits = tl.where(mask, logits, float("-inf"))
        submax = tl.max(tl.reshape(logits, (NUM_SUB, SUB_SIZE)), axis=1)
        tl.store(submax_ptr + row * submax_stride + sub_offs, submax)


@triton.jit
def _select_survivors_kernel(
    # [num_logits, KP] fp32 / int32, [num_logits] int32 / fp32
    surv_val_ptr,
    surv_idx_ptr,
    num_surv_ptr,
    lse_ptr,
    # [num_logits, submax_stride]
    submax_ptr,
    submax_stride,
    num_sub,
    # [num_logits, V]
    logits_ptr,
    logits_stride,
    # [num_logits]
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    # [num_tokens]
    input_ids_ptr,
    # [num_logits]
    logits_indices_ptr,
    # [max_num_reqs]
    top_k_ptr,
    top_p_ptr,
    temperature_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    KP: tl.constexpr,
    SUB_SIZE: tl.constexpr,
    PADDED_NUM_SUB: tl.constexpr,
    HAS_PENALTIES: tl.constexpr,
    TOP_P: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    top_k = tl.load(top_k_ptr + req_state_idx)
    temp = tl.load(temperature_ptr + req_state_idx).to(tl.float32)

    # 1. The KP sub-blocks with the largest (max, -index) keys.
    sub = tl.arange(0, PADDED_NUM_SUB)
    submax = tl.load(
        submax_ptr + row * submax_stride + sub,
        mask=sub < num_sub,
        other=float("-inf"),
    )
    top_sub = tl.topk(_pack_keys(submax, sub, PADDED_NUM_SUB), KP)
    _, sub_idx = _unpack_keys(top_sub, PADDED_NUM_SUB)

    # 2. Their processed logits.
    offs = sub_idx[:, None] * SUB_SIZE + tl.arange(0, SUB_SIZE)[None, :]
    mask = offs < vocab_size
    logits = tl.load(
        logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")
    ).to(tl.float32)
    if HAS_PENALTIES:
        rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
        freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
        pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)
        use_rep_penalty = rep_penalty != 1.0
        use_penalty = use_rep_penalty or freq_penalty != 0.0 or pres_penalty != 0.0
        if use_penalty:
            output_bin_counts = tl.load(
                output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + offs,
                mask=mask,
                other=0,
            )
            pos = tl.load(expanded_local_pos_ptr + row)
            start = row - pos
            for prev_pos in range(pos):
                token_idx = tl.load(logits_indices_ptr + start + prev_pos + 1)
                prev_token = tl.load(input_ids_ptr + token_idx)
                output_bin_counts += (offs == prev_token).to(tl.int32)
            prompt_bin_mask = tl.zeros((KP, SUB_SIZE), tl.int1)
            if use_rep_penalty:
                word = tl.load(
                    prompt_bin_mask_ptr
                    + req_state_idx * prompt_bin_mask_stride
                    + offs // 32,
                    mask=mask,
                    other=0,
                )
                prompt_bin_mask = ((word >> (offs % 32).to(tl.int32)) & 1).to(tl.int1)
            logits = _penalize(
                logits,
                output_bin_counts,
                prompt_bin_mask,
                rep_penalty,
                freq_penalty,
                pres_penalty,
                use_rep_penalty,
            )
    if temp != 0.0 and temp != 1.0:
        logits = logits / temp

    # 3. Exact top-k by (value desc, token id asc), then top-p. Same selection
    # arithmetic as spec_topk_topp._select_threshold.
    sel = mask & (logits > float("-inf"))
    # INT64_MIN marks non-candidates (masked, -inf or NaN logits).
    pad = -9223372036854775808
    keys = tl.where(sel, _pack_keys(logits, offs, vocab_size), pad)
    top = tl.topk(tl.reshape(keys, (KP * SUB_SIZE,)), KP)
    pos_k = tl.arange(0, KP)
    num_valid = tl.sum(((pos_k < top_k) & (top != pad)).to(tl.int32))
    valid = pos_k < num_valid
    value, token = _unpack_keys(top, vocab_size)
    num_keep = num_valid
    if TOP_P:
        top_p = tl.load(top_p_ptr + req_state_idx)
        max_value = tl.max(tl.where(valid, value, float("-inf")), axis=0)
        e = tl.where(valid, tl.exp(value - max_value), 0.0)
        prob = e / tl.sum(e, axis=0)
        cum = tl.cumsum(prob, axis=0)
        below = valid & (pos_k < num_valid - 1) & (cum < top_p)
        num_below = tl.sum(below.to(tl.int32))
        num_keep = tl.where(
            top_p < 1.0, tl.minimum(num_below + 1, num_valid), num_valid
        )
    keep = pos_k < num_keep

    # 4. Compact survivors in (value desc, token id asc) order + logsumexp.
    max_value = tl.max(tl.where(keep, value, float("-inf")), axis=0)
    sumexp = tl.sum(tl.where(keep, tl.exp(value - max_value), 0.0), axis=0)
    lse = tl.where(num_keep > 0, max_value + tl.log(sumexp), float("-inf"))
    tl.store(surv_val_ptr + row * KP + pos_k, value)
    tl.store(surv_idx_ptr + row * KP + pos_k, token.to(tl.int32))
    tl.store(num_surv_ptr + row, num_keep)
    tl.store(lse_ptr + row, lse)


@triton.jit
def _compact_rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    num_rejected_ptr,
    # [num_logits, KP] / [num_logits]
    surv_val_ptr,
    surv_idx_ptr,
    num_surv_ptr,
    lse_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    seq_lens_ptr,
    # [max_num_reqs]
    prefill_len_ptr,
    temperature_ptr,
    seeds_ptr,
    # [num_tokens]
    input_ids_ptr,
    positions_ptr,
    # [num_logits]
    logits_indices_ptr,
    KP: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end - start - 1
    seed = tl.load(seeds_ptr + req_state_idx)
    temp = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0
    k = tl.arange(0, KP)

    # Probability-ratio test as in _rejection_kernel (one-hot draft: log q = 0).
    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start + i
        draft_sampled = tl.load(
            input_ids_ptr + tl.load(logits_indices_ptr + logit_idx + 1)
        ).to(tl.int64)
        is_valid_draft = draft_sampled >= 0
        draft_sampled = tl.maximum(0, draft_sampled)
        if not is_greedy:
            verifying &= is_valid_draft
        if verifying:
            pos = tl.load(positions_ptr + tl.load(logits_indices_ptr + logit_idx))
            u = tl_rand32(seed, pos, includes_zero=False)
            num_surv = tl.load(num_surv_ptr + logit_idx)
            if is_greedy:
                # The first survivor is the first-index argmax (0 if none).
                target_argmax = tl.where(
                    num_surv > 0, tl.load(surv_idx_ptr + logit_idx * KP), 0
                ).to(tl.int64)
                accepted = (target_argmax == draft_sampled) & is_valid_draft
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    draft_sampled if accepted else target_argmax,
                )
            else:
                surv = k < num_surv
                token = tl.load(surv_idx_ptr + logit_idx * KP + k, mask=surv, other=-1)
                value = tl.load(surv_val_ptr + logit_idx * KP + k)
                target_logit = tl.max(
                    tl.where(token == draft_sampled, value, float("-inf")), axis=0
                )
                target_logprob = target_logit - tl.load(lse_ptr + logit_idx)
                accepted = target_logprob > tl.log(u) + 0.0
                verifying = accepted
                accepted_length += accepted
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)

    # Bookkeeping as in the fused branch of _insert_resampled_kernel.
    is_chunked_prefilling = tl.load(seq_lens_ptr + req_idx) < tl.load(
        prefill_len_ptr + req_state_idx
    )
    final_num_sampled = tl.where(is_chunked_prefilling, 0, accepted_length + 1)
    tl.store(num_sampled_ptr + req_idx, final_num_sampled.to(tl.int32))
    num_rejected = end - start - final_num_sampled
    tl.store(
        num_rejected_ptr + req_idx,
        tl.where(is_chunked_prefilling, 0, num_rejected).to(tl.int32),
    )

    resample_token_idx = start + accepted_length
    is_bonus = resample_token_idx == end - 1
    if temp == 0.0 and not is_bonus:
        # Greedy rejection: the target argmax is already in place.
        return

    # Resample from the residual (survivors minus the rejected draft token)
    # with the per-token noise of _resample_kernel. Exact ties go to the
    # lowest token id, as the blocked argmax there does.
    rejected_draft_token = tl.load(
        input_ids_ptr
        + tl.load(
            logits_indices_ptr + resample_token_idx + 1, mask=not is_bonus, other=0
        ),
        mask=not is_bonus,
        other=0,
    )
    is_valid_rejected_draft = rejected_draft_token >= 0
    num_surv = tl.load(num_surv_ptr + resample_token_idx)
    residual = k < num_surv
    token = tl.load(surv_idx_ptr + resample_token_idx * KP + k, mask=residual, other=0)
    if not is_bonus and is_valid_rejected_draft:
        residual &= token != rejected_draft_token
    value = tl.load(surv_val_ptr + resample_token_idx * KP + k)
    value = tl.where(residual, value, float("-inf"))
    pos = tl.load(positions_ptr + tl.load(logits_indices_ptr + resample_token_idx))
    noised = gumbel_noised_logits(
        value,
        token,
        residual,
        seed,
        pos,
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    best = tl.max(noised, axis=0)
    resampled = tl.min(tl.where(residual & (noised == best), token, 2147483647))
    resampled = tl.where(best > float("-inf"), resampled, 0)
    tl.store(
        sampled_ptr + req_idx * sampled_stride + accepted_length,
        resampled.to(tl.int64),
    )


def select_survivors(
    # [num_logits, V] raw target logits (any float dtype, unit stride)
    logits: torch.Tensor,
    # [num_tokens] / [num_logits] input batch
    input_ids: torch.Tensor,
    logits_indices: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs] persistent request state
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    # None, or (repetition, frequency, presence, prompt_bin_mask,
    # output_bin_counts) when some request uses penalties
    penalties: tuple[torch.Tensor, ...] | None,
    max_top_k: int,
    use_top_p: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-k/top-p survivors of the penalized, temperature-scaled logits.

    Requires 1 <= top_k <= max_top_k <= MAX_TOP_K on every row. Returns
    (values [num_logits, KP] fp32, token ids [num_logits, KP] int32,
    num_survivors [num_logits] int32, logsumexp [num_logits] fp32); the first
    num_survivors entries of a row are its survivors in (value desc, token id
    asc) order, the values bitwise those of the unfused masked canvas.
    """
    assert logits.ndim == 2 and logits.stride(1) == 1
    assert 1 <= max_top_k <= MAX_TOP_K
    num_logits, vocab_size = logits.shape
    num_reqs = cu_num_logits.shape[0] - 1
    device = logits.device
    kp = _top_k_pow2(max_top_k)
    num_blocks = triton.cdiv(vocab_size, _BLOCK_SIZE)
    num_sub = num_blocks * (_BLOCK_SIZE // _SUB_SIZE)
    submax = torch.empty(num_logits, num_sub, dtype=torch.float32, device=device)
    surv_val = torch.empty(num_logits, kp, dtype=torch.float32, device=device)
    surv_idx = torch.empty(num_logits, kp, dtype=torch.int32, device=device)
    num_surv = torch.empty(num_logits, dtype=torch.int32, device=device)
    lse = torch.empty(num_logits, dtype=torch.float32, device=device)
    if num_logits == 0:
        return surv_val, surv_idx, num_surv, lse

    has_penalties = penalties is not None
    if penalties is None:
        # Never dereferenced: HAS_PENALTIES=False compiles the loads out.
        rep = freq = pres = prompt_bin_mask = output_bin_counts = temperature
        prompt_stride = counts_stride = 0
    else:
        rep, freq, pres, prompt_bin_mask, output_bin_counts = penalties
        prompt_stride = prompt_bin_mask.stride(0)
        counts_stride = output_bin_counts.stride(0)

    _penalized_submax_kernel[(num_reqs, num_blocks)](
        submax,
        submax.stride(0),
        logits,
        logits.stride(0),
        cu_num_logits,
        idx_mapping,
        input_ids,
        logits_indices,
        temperature,
        rep,
        freq,
        pres,
        prompt_bin_mask,
        prompt_stride,
        output_bin_counts,
        counts_stride,
        vocab_size,
        BLOCK_SIZE=_BLOCK_SIZE,
        SUB_SIZE=_SUB_SIZE,
        HAS_PENALTIES=has_penalties,
        num_warps=8,
    )
    _select_survivors_kernel[(num_logits,)](
        surv_val,
        surv_idx,
        num_surv,
        lse,
        submax,
        submax.stride(0),
        num_sub,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        expanded_local_pos,
        input_ids,
        logits_indices,
        top_k,
        top_p,
        temperature,
        rep,
        freq,
        pres,
        prompt_bin_mask,
        prompt_stride,
        output_bin_counts,
        counts_stride,
        vocab_size,
        KP=kp,
        SUB_SIZE=_SUB_SIZE,
        # At least KP wide, so topk returns KP distinct (possibly empty)
        # sub-blocks when the vocab has fewer sub-blocks than KP.
        PADDED_NUM_SUB=max(triton.next_power_of_2(num_sub), kp),
        HAS_PENALTIES=has_penalties,
        TOP_P=use_top_p,
        num_warps=8,
    )
    return surv_val, surv_idx, num_surv, lse


def fused_rejection_sample(
    # [num_logits, V] raw target logits (any float dtype, unit stride)
    logits: torch.Tensor,
    # [num_tokens] / [num_logits] input batch
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits_indices: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    seq_lens: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs] persistent request state
    prefill_len: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    penalties: tuple[torch.Tensor, ...] | None,
    max_top_k: int,
    use_top_p: bool,
    num_speculative_steps: int,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rejection-sample one-hot drafts against the penalized, temperature-scaled,
    top-k/top-p-masked target without materializing the masked canvas.

    Returns (sampled [num_reqs, steps + 1] int64, num_sampled [num_reqs] int32,
    num_rejected [num_reqs] int32), as rejection_sample with its fused
    num_rejected outputs.
    """
    num_reqs = cu_num_logits.shape[0] - 1
    device = logits.device
    sampled = torch.empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64, device=device
    )
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=device)
    if num_reqs == 0:
        return sampled, num_sampled, num_rejected
    surv_val, surv_idx, num_surv, lse = select_survivors(
        logits,
        input_ids,
        logits_indices,
        cu_num_logits,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        top_k,
        top_p,
        penalties,
        max_top_k,
        use_top_p,
    )
    _compact_rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        num_rejected,
        surv_val,
        surv_idx,
        num_surv,
        lse,
        cu_num_logits,
        idx_mapping,
        seq_lens,
        prefill_len,
        temperature,
        seeds,
        input_ids,
        positions,
        logits_indices,
        KP=surv_val.shape[1],
        USE_FP64=use_fp64,
        num_warps=1,
    )
    return sampled, num_sampled, num_rejected
