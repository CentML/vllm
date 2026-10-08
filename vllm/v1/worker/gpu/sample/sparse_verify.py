# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse preparation and verification for one-hot drafts at temperature 1.

Only live 1024-token chunks and the scalar draft lookup are initialized. This
private canvas must never reach a dense consumer (including processed logprobs).
Raw logprobs still read the original, unmodified logits. Survivor selection reuses
the newer fused top-k kernels; verification retains the ordinary 8192-token
logsumexp reductions and 1024-token Gumbel draws, not the compact sampler's
slightly different logsumexp reduction order.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rejection
from vllm.v1.worker.gpu.spec_decode.fused_rejection import select_survivors

_CHUNK = 1024
_STATS_BLOCK = 8192


@triton.jit
def _emit_kernel(
    out_ptr,
    out_stride,
    live_ptr,
    live_stride,
    values_ptr,
    tokens_ptr,
    num_survivors_ptr,
    draft_sampled_ptr,
    num_rows,
    vocab_size,
    KP: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    k = tl.arange(0, KP)
    count = tl.load(num_survivors_ptr + row)
    tokens = tl.load(tokens_ptr + row * KP + k)
    here = (k < count) & (tokens // CHUNK == chunk)
    live = tl.sum(here.to(tl.int32)) > 0
    tl.store(live_ptr + row * live_stride + chunk, live)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    if live:
        tl.store(
            out_ptr + row * out_stride + offs, float("-inf"), mask=offs < vocab_size
        )
        # Different warps clear and scatter the chunk. No consumer can observe
        # it until this launch completes, but the stores inside it must order.
        tl.debug_barrier()
        values = tl.load(values_ptr + row * KP + k)
        tl.store(out_ptr + row * out_stride + tokens, values, mask=here)
    else:
        # The rejection kernel reads this scalar even when its chunk is dead.
        # The next request's first token on a bonus row is harmless: that row
        # has no acceptance test and the sparse resampler ignores dead chunks.
        draft = tl.load(draft_sampled_ptr + row + 1, mask=row + 1 < num_rows, other=-1)
        tl.store(
            out_ptr + row * out_stride + offs,
            float("-inf"),
            mask=(offs < vocab_size) & (offs == draft),
        )


@triton.jit
def _stats_kernel(
    logits_ptr,
    logits_stride,
    live_ptr,
    live_stride,
    max_ptr,
    max_stride,
    sumexp_ptr,
    sumexp_stride,
    expanded_local_pos_ptr,
    vocab_size,
    num_speculative_steps,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    if tl.load(expanded_local_pos_ptr + row) >= num_speculative_steps:
        return
    CPB: tl.constexpr = BLOCK // CHUNK
    chunks = block * CPB + tl.arange(0, CPB)
    live = tl.load(
        live_ptr + row * live_stride + chunks,
        mask=chunks * CHUNK < vocab_size,
        other=0,
    )
    max_value = float("-inf")
    sumexp = 0.0
    if tl.sum(live.to(tl.int32)) > 0:
        offs = block * BLOCK + tl.arange(0, BLOCK)
        lane_live = tl.broadcast_to(live[:, None], (CPB, CHUNK)).reshape(BLOCK)
        logits = tl.load(
            logits_ptr + row * logits_stride + offs,
            mask=(offs < vocab_size) & lane_live,
            other=float("-inf"),
        )
        max_value, sumexp = rejection._compute_max_and_sumexp(logits)
    tl.store(max_ptr + row * max_stride + block, max_value)
    tl.store(sumexp_ptr + row * sumexp_stride + block, sumexp)


@triton.jit
def _resample_kernel(
    logits_ptr,
    logits_stride,
    live_ptr,
    live_stride,
    argmax_ptr,
    argmax_stride,
    max_ptr,
    max_stride,
    rejected_step_ptr,
    cu_num_logits_ptr,
    expanded_idx_mapping_ptr,
    draft_sampled_ptr,
    temperature_ptr,
    seed_ptr,
    pos_ptr,
    vocab_size,
    CHUNK: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    req = tl.program_id(0)
    chunk = tl.program_id(1)
    start = tl.load(cu_num_logits_ptr + req).to(tl.int64)
    end = tl.load(cu_num_logits_ptr + req + 1)
    row = start + tl.load(rejected_step_ptr + req)
    is_bonus = row == end - 1
    live = tl.load(live_ptr + row * live_stride + chunk)
    if USE_FP64:
        value = tl.full((), float("-inf"), tl.float64)
    else:
        value = tl.full((), float("-inf"), tl.float32)
    token_id = chunk * CHUNK
    if live:
        offs = chunk * CHUNK + tl.arange(0, CHUNK)
        mask = offs < vocab_size
        logits = tl.load(
            logits_ptr + row * logits_stride + offs,
            mask=mask,
            other=float("-inf"),
        )
        draft = tl.load(draft_sampled_ptr + row + 1, mask=not is_bonus, other=-1)
        logits = tl.where((not is_bonus) & (offs == draft), float("-inf"), logits)
        value, idx = rejection._seeded_resample_argmax(
            logits,
            offs,
            mask,
            row,
            expanded_idx_mapping_ptr,
            temperature_ptr,
            seed_ptr,
            pos_ptr,
            vocab_size,
            USE_FP64=USE_FP64,
        )
        token_id += idx
    tl.store(argmax_ptr + req * argmax_stride + chunk, token_id)
    tl.store(max_ptr + req * max_stride + chunk, value)


def sparse_rejection_sample(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    logits_indices: torch.Tensor,
    draft_sampled: torch.Tensor,
    pos: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    penalties: tuple[torch.Tensor, ...] | None,
    max_top_k: int,
    use_top_p: bool,
    num_speculative_steps: int,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sampled tokens and raw counts, before chunked-prefill bookkeeping.

    Caller guarantees one-hot drafts, all temperatures 1, ordinary verification
    and no dense processed-logits consumer. Raw logprobs are safe. RNG keys and
    reduction geometry match rejection_sample exactly for the same survivors.
    """
    values, tokens, counts, _ = select_survivors(
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
    num_rows, vocab_size = logits.shape
    num_reqs = idx_mapping.shape[0]
    num_chunks = triton.cdiv(vocab_size, _CHUNK)
    canvas = torch.empty_like(logits, dtype=torch.float32)
    live = torch.empty(num_rows, num_chunks, dtype=torch.bool, device=logits.device)
    _emit_kernel[(num_rows, num_chunks)](
        canvas,
        canvas.stride(0),
        live,
        live.stride(0),
        values,
        tokens,
        counts,
        draft_sampled,
        num_rows,
        vocab_size,
        KP=values.shape[1],
        CHUNK=_CHUNK,
    )
    num_blocks = triton.cdiv(vocab_size, _STATS_BLOCK)
    local_max = canvas.new_empty(num_rows, num_blocks)
    local_sumexp = canvas.new_empty(num_rows, num_blocks)
    _stats_kernel[(num_rows, num_blocks)](
        canvas,
        canvas.stride(0),
        live,
        live.stride(0),
        local_max,
        local_max.stride(0),
        local_sumexp,
        local_sumexp.stride(0),
        expanded_local_pos,
        vocab_size,
        num_speculative_steps,
        BLOCK=_STATS_BLOCK,
        CHUNK=_CHUNK,
    )
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_lse = canvas.new_empty(num_reqs)
    draft_lse = canvas.new_empty(num_reqs)
    # Greedy verification is excluded by eligibility, but its runtime branch
    # still needs a typed argmax pointer during Triton compilation. Reuse the
    # int64 output; that branch never reads it on the sparse path.
    rejection._rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_lse,
        draft_lse,
        canvas,
        canvas.stride(0),
        sampled,
        0,
        local_max,
        local_max.stride(0),
        local_sumexp,
        local_sumexp.stride(0),
        draft_sampled,
        None,
        0,
        0,
        None,
        0,
        None,
        0,
        cu_num_logits,
        idx_mapping,
        temperature,
        seeds,
        pos,
        None,
        None,
        None,
        0,
        num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
        HAS_DRAFT_LOGITS=False,
        SYNTHETIC_MODE=False,
        USE_BLOCK_VERIFICATION=False,
        num_warps=1,
    )
    resampled_argmax = sampled.new_empty(num_reqs, num_chunks)
    resampled_max = canvas.new_empty(
        num_reqs, num_chunks, dtype=torch.float64 if use_fp64 else torch.float32
    )
    _resample_kernel[(num_reqs, num_chunks)](
        canvas,
        canvas.stride(0),
        live,
        live.stride(0),
        resampled_argmax,
        resampled_argmax.stride(0),
        resampled_max,
        resampled_max.stride(0),
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seeds,
        pos,
        vocab_size,
        CHUNK=_CHUNK,
        USE_FP64=use_fp64,
    )
    rejection._insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_argmax,
        resampled_argmax.stride(0),
        resampled_max,
        resampled_max.stride(0),
        num_chunks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=triton.next_power_of_2(num_chunks),
    )
    return sampled, num_sampled
