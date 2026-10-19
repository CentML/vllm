# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Varlen causal short conv + SiLU for the KDA prefill path, planar output.

Same math and conv-state contract as ``causal_conv1d_fn`` (vLLM Triton, width
``W``; state positions ``0..W-2`` of ``conv_states[cache_indices[s]]`` hold the
``W-1`` tokens before sequence ``s`` when ``has_initial_state[s]``, and receive
its last ``W-1`` tokens), with two differences:

- The output is planar, ``[num_planes, T, dim // num_planes]``: for the merged
  q|k|v channels each of q, k and v is a dense ``[T, H * D]`` tensor, which is
  what FlashKDA consumes, so no ``.contiguous()`` copies are needed.
- Programs tile tokens independently of sequence boundaries (each token finds
  its sequence by binary search over ``query_start_loc``) and read 2D tiles
  with wide vector loads; the state update is a second, tiny launch, so the
  initial-state reads of the first launch never race with the writes.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID


# Batch-dependent scalars are not specialized (no recompiles per batch shape).
@triton.jit(do_not_specialize=["num_tokens", "num_seqs", "search_step"])
def _kda_conv_fwd_planar_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    state_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    query_start_loc_ptr,
    out_ptr,
    num_tokens,
    num_seqs,
    search_step,
    dim,
    plane_dim,
    stride_x_token,
    stride_w_dim,
    stride_w_width,
    stride_state_seq,
    stride_state_dim,
    stride_state_tok,
    stride_out_plane,
    stride_out_token,
    stride_cache_indices,
    stride_has_initial_state,
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_t = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_f = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    total = tl.load(query_start_loc_ptr + num_seqs)
    mask_t = offs_t < tl.minimum(total, num_tokens)

    # Sequence of each token: the last s with query_start_loc[s] <= t.
    seq = tl.zeros([BLOCK_M], dtype=tl.int32)
    step = search_step
    while step > 0:
        cand = seq + step
        start_cand = tl.load(
            query_start_loc_ptr + tl.minimum(cand, num_seqs),
            mask=mask_t,
            other=0,
        )
        seq = tl.where((cand <= num_seqs) & (start_cand <= offs_t), cand, seq)
        step = step // 2
    seq = tl.minimum(seq, num_seqs - 1)
    seq_start = tl.load(query_start_loc_ptr + seq, mask=mask_t, other=0)
    pos = offs_t - seq_start
    cache_idx = tl.load(
        cache_indices_ptr + seq * stride_cache_indices, mask=mask_t, other=pad_slot_id
    ).to(tl.int64)
    has_state = tl.load(
        has_initial_state_ptr + seq * stride_has_initial_state, mask=mask_t, other=0
    ).to(tl.int1)
    has_state = has_state & (cache_idx != pad_slot_id) & (cache_idx != null_block_id)
    cache_idx = tl.where(has_state, cache_idx, 0)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_f, mask=mask_f, other=0.0).to(tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + bias[None, :]
    else:
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    mask_tf = mask_t[:, None] & mask_f[None, :]
    x_rows = x_ptr + offs_t[:, None] * stride_x_token + offs_f[None, :]
    near_start = tl.min(tl.where(mask_t, pos, KERNEL_WIDTH)) < KERNEL_WIDTH - 1
    if near_start:
        # Some token of this block is among the first W-1 of its sequence:
        # taps before the sequence start read the initial conv state.
        for j in tl.static_range(KERNEL_WIDTH):
            back = KERNEL_WIDTH - 1 - j
            in_seq = pos >= back
            x_val = tl.load(
                x_rows - back * stride_x_token,
                mask=mask_tf & in_seq[:, None],
                other=0.0,
            )
            if back > 0:
                state_pos = tl.where(in_seq, 0, KERNEL_WIDTH - 1 + pos - back)
                s_val = tl.load(
                    state_ptr
                    + cache_idx[:, None] * stride_state_seq
                    + offs_f[None, :] * stride_state_dim
                    + state_pos[:, None] * stride_state_tok,
                    mask=mask_tf & ((~in_seq) & has_state)[:, None],
                    other=0.0,
                )
                x_val = tl.where(in_seq[:, None], x_val, s_val)
            w_j = tl.load(
                w_ptr + offs_f * stride_w_dim + j * stride_w_width, mask=mask_f
            )
            acc += x_val * w_j[None, :]
    else:
        for j in tl.static_range(KERNEL_WIDTH):
            back = KERNEL_WIDTH - 1 - j
            x_val = tl.load(x_rows - back * stride_x_token, mask=mask_tf, other=0.0)
            w_j = tl.load(
                w_ptr + offs_f * stride_w_dim + j * stride_w_width, mask=mask_f
            )
            acc += x_val * w_j[None, :]
    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))

    # BLOCK_N divides plane_dim, so a program writes inside one plane.
    plane = (tl.program_id(1) * BLOCK_N) // plane_dim
    col = offs_f - plane * plane_dim
    out_ptrs = (
        out_ptr
        + plane.to(tl.int64) * stride_out_plane
        + offs_t[:, None] * stride_out_token
        + col[None, :]
    )
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_tf)


@triton.jit
def _kda_conv_state_update_kernel(
    x_ptr,
    state_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    query_start_loc_ptr,
    dim,
    stride_x_token,
    stride_state_seq,
    stride_state_dim,
    stride_state_tok,
    stride_cache_indices,
    stride_has_initial_state,
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq = tl.program_id(0)
    offs_f = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    cache_idx = tl.load(cache_indices_ptr + seq * stride_cache_indices).to(tl.int64)
    seq_start = tl.load(query_start_loc_ptr + seq).to(tl.int64)
    seq_len = (tl.load(query_start_loc_ptr + seq + 1) - seq_start).to(tl.int32)
    if (cache_idx == pad_slot_id) | (cache_idx == null_block_id) | (seq_len <= 0):
        return
    has_state = tl.load(has_initial_state_ptr + seq * stride_has_initial_state).to(
        tl.int1
    )
    state_base = state_ptr + cache_idx * stride_state_seq + offs_f * stride_state_dim
    # New state position k holds sequence position seq_len - (W-1) + k: from x,
    # or (short sequences) the old state shifted left by seq_len. All loads
    # happen before the first store.
    v0 = _kda_conv_new_state(
        x_ptr,
        state_base,
        seq_start,
        seq_len,
        has_state,
        offs_f,
        mask_f,
        stride_x_token,
        stride_state_tok,
        0,
        KERNEL_WIDTH,
    )
    if KERNEL_WIDTH >= 3:
        v1 = _kda_conv_new_state(
            x_ptr,
            state_base,
            seq_start,
            seq_len,
            has_state,
            offs_f,
            mask_f,
            stride_x_token,
            stride_state_tok,
            1,
            KERNEL_WIDTH,
        )
    if KERNEL_WIDTH >= 4:
        v2 = _kda_conv_new_state(
            x_ptr,
            state_base,
            seq_start,
            seq_len,
            has_state,
            offs_f,
            mask_f,
            stride_x_token,
            stride_state_tok,
            2,
            KERNEL_WIDTH,
        )
    tl.debug_barrier()
    tl.store(state_base, v0, mask=mask_f)
    if KERNEL_WIDTH >= 3:
        tl.store(state_base + stride_state_tok, v1, mask=mask_f)
    if KERNEL_WIDTH >= 4:
        tl.store(state_base + 2 * stride_state_tok, v2, mask=mask_f)


@triton.jit
def _kda_conv_new_state(
    x_ptr,
    state_base,
    seq_start,
    seq_len,
    has_state,
    offs_f,
    mask_f,
    stride_x_token,
    stride_state_tok,
    k: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
):
    rel = seq_len - (KERNEL_WIDTH - 1) + k
    x_val = tl.load(
        x_ptr + (seq_start + tl.maximum(rel, 0)) * stride_x_token + offs_f,
        mask=mask_f & (rel >= 0),
        other=0.0,
    )
    s_val = tl.load(
        state_base + tl.minimum(k + seq_len, KERNEL_WIDTH - 2) * stride_state_tok,
        mask=mask_f & (rel < 0) & has_state,
        other=0.0,
    )
    return tl.where(rel >= 0, x_val, s_val)


def kda_causal_conv1d_planar(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    num_planes: int = 3,
    activation: str | None = "silu",
    out: torch.Tensor | None = None,
    block_m: int = 32,
    block_n: int = 256,
    num_warps: int = 8,
) -> torch.Tensor:
    """Causal conv1d (+ SiLU) over a varlen token batch, planar output.

    Args:
        x: ``[T, dim]`` token-major input; rows may be strided (e.g. the
            q|k|v slice of a wider projection), channels must be dense.
        weight: ``[dim, width]`` conv weight.
        bias: optional ``[dim]`` bias.
        conv_states: ``[num_slots, dim, >= width - 1]`` view of the conv
            state pool (any strides, so both SD and DS layouts work); updated
            in place for every sequence with a real cache slot.
        query_start_loc: ``[N + 1]`` int32 cumulative token offsets.
        cache_indices: ``[>= N]`` cache slot per sequence (pad/null skipped),
            may be strided (e.g. column 0 of the block table).
        has_initial_state: ``[>= N]`` bool, read the old state as history.
        num_planes: output planes; ``dim`` must be divisible by it.
        activation: ``"silu"``/``"swish"`` or ``None``.
        out: optional ``[num_planes, >= T, dim // num_planes]`` output with a
            dense last dim.
        block_m: tokens per program.
        block_n: channels per program; must divide ``dim // num_planes``.
        num_warps: warps per program.

    Returns:
        ``[num_planes, T, dim // num_planes]`` output in ``x.dtype``.

    """
    assert x.dim() == 2 and x.stride(1) == 1
    num_tokens, dim = x.shape
    width = weight.shape[1]
    assert 2 <= width <= 4
    assert weight.shape[0] == dim and dim % num_planes == 0
    assert conv_states.shape[1] == dim and conv_states.shape[2] >= width - 1
    plane_dim = dim // num_planes
    assert plane_dim % block_n == 0
    if out is None:
        out = torch.empty(
            (num_planes, num_tokens, plane_dim), dtype=x.dtype, device=x.device
        )
    assert out.shape[0] == num_planes and out.shape[2] == plane_dim
    assert out.stride(2) == 1
    num_seqs = query_start_loc.numel() - 1
    if num_tokens == 0 or num_seqs <= 0:
        return out[:, :num_tokens]
    silu = activation in ("silu", "swish")
    grid = (triton.cdiv(num_tokens, block_m), triton.cdiv(dim, block_n))
    _kda_conv_fwd_planar_kernel[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        num_tokens,
        num_seqs,
        1 << (num_seqs.bit_length() - 1),
        dim,
        plane_dim,
        x.stride(0),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        out.stride(0),
        out.stride(1),
        cache_indices.stride(0),
        has_initial_state.stride(0),
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=NULL_BLOCK_ID,
        KERNEL_WIDTH=width,
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=silu,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    update_block_n = 1024
    _kda_conv_state_update_kernel[(num_seqs, triton.cdiv(dim, update_block_n))](
        x,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        dim,
        x.stride(0),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        cache_indices.stride(0),
        has_initial_state.stride(0),
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=NULL_BLOCK_ID,
        KERNEL_WIDTH=width,
        BLOCK_N=update_block_n,
        num_warps=4,
    )
    return out[:, :num_tokens]
