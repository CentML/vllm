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
- Programs tile 16 tokens x BLOCK_N channels independently of sequence
  boundaries (each token finds its sequence by binary search over
  ``query_start_loc``). Away from sequence starts a program issues one 2D
  load of its 16 rows plus a 4-row halo (16-byte vectors, all in flight),
  unpacks the rows in registers and slides the width-4 window over them;
  causal_conv1d_fn instead loops token by token with 4-byte loads and ~200
  registers per thread, which kept VR200 at ~3 TB/s. Blocks that contain the
  first tokens of a sequence go token by token and read the initial state.
  The state update is a second, tiny launch, so the initial-state reads of
  the first launch never race with the writes.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID


@triton.jit
def _split_rows(v):
    """[BN, 2k] -> (rows 0, 2, .., rows 1, 3, ..), each [BN, k]; in registers."""
    return tl.split(tl.reshape(v, [v.shape[0], v.shape[1] // 2, 2]))


@triton.jit
def _join_rows(a, b):
    """Inverse of _split_rows."""
    return tl.reshape(tl.join(a, b), [a.shape[0], a.shape[1] * 2])


@triton.jit
def _rows4(v):
    """[BN, 4] -> its 4 rows as [BN] tensors."""
    e, o = _split_rows(v)
    r0, r2 = _split_rows(e)
    r1, r3 = _split_rows(o)
    n: tl.constexpr = v.shape[0]
    return (
        tl.reshape(r0, [n]),
        tl.reshape(r1, [n]),
        tl.reshape(r2, [n]),
        tl.reshape(r3, [n]),
    )


@triton.jit
def _unrows4(r0, r1, r2, r3):
    """Inverse of _rows4."""
    n: tl.constexpr = r0.shape[0]
    e = _join_rows(tl.reshape(r0, [n, 1]), tl.reshape(r2, [n, 1]))
    o = _join_rows(tl.reshape(r1, [n, 1]), tl.reshape(r3, [n, 1]))
    return _join_rows(e, o)


@triton.jit
def _conv_silu(a, b, c, d, w0, w1, w2, w3, acc0, SILU_ACTIVATION: tl.constexpr):
    # Same accumulation order as causal_conv1d_fn (oldest tap first).
    acc = acc0 + a * w0
    acc += b * w1
    acc += c * w2
    acc += d * w3
    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))
    return acc


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
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Width-4 conv, 16 tokens x BLOCK_N channels per program.
    BLOCK_M: tl.constexpr = 16
    t0 = tl.program_id(0).to(tl.int64) * BLOCK_M
    rows = tl.arange(0, BLOCK_M)
    offs_t = t0 + rows
    offs_f = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    total = tl.load(query_start_loc_ptr + num_seqs)
    limit = tl.minimum(total, num_tokens)
    mask_t = offs_t < limit

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

    w0 = tl.load(w_ptr + offs_f * stride_w_dim, mask=mask_f, other=0.0)
    w1 = tl.load(w_ptr + offs_f * stride_w_dim + stride_w_width, mask=mask_f, other=0.0)
    w2 = tl.load(
        w_ptr + offs_f * stride_w_dim + 2 * stride_w_width, mask=mask_f, other=0.0
    )
    w3 = tl.load(
        w_ptr + offs_f * stride_w_dim + 3 * stride_w_width, mask=mask_f, other=0.0
    )
    if HAS_BIAS:
        acc0 = tl.load(bias_ptr + offs_f, mask=mask_f, other=0.0).to(tl.float32)
    else:
        acc0 = tl.zeros([BLOCK_N], dtype=tl.float32)
    # BLOCK_N divides plane_dim, so a program writes inside one plane.
    plane = (tl.program_id(1) * BLOCK_N) // plane_dim
    col = offs_f - plane * plane_dim
    out_base = out_ptr + plane.to(tl.int64) * stride_out_plane + col
    x_base = x_ptr + offs_f

    near_start = tl.min(tl.where(mask_t, pos, 3)) < 3
    if near_start:
        # Some token is among the first 3 of its sequence: taps before the
        # sequence start read the initial conv state. Token by token (rare).
        cache_idx = tl.load(
            cache_indices_ptr + seq * stride_cache_indices,
            mask=mask_t,
            other=pad_slot_id,
        ).to(tl.int64)
        has_state = tl.load(
            has_initial_state_ptr + seq * stride_has_initial_state,
            mask=mask_t,
            other=0,
        ).to(tl.int32)
        has_state = tl.where(
            (cache_idx != pad_slot_id) & (cache_idx != null_block_id), has_state, 0
        )
        for i in range(BLOCK_M):
            sel = rows == i
            p = tl.sum(tl.where(sel, pos, 0))
            ci = tl.sum(tl.where(sel & (has_state != 0), cache_idx, 0))
            hs = tl.sum(tl.where(sel, has_state, 0)) != 0
            t = t0 + i
            m = mask_f & (t < limit)
            st_base = state_ptr + ci * stride_state_seq + offs_f * stride_state_dim
            a = tl.load(x_base + (t - 3) * stride_x_token, mask=m & (p >= 3), other=0.0)
            a = tl.where(
                p >= 3,
                a,
                tl.load(
                    st_base + tl.maximum(p, 0) * stride_state_tok,
                    mask=m & (p < 3) & hs,
                    other=0.0,
                ),
            )
            b = tl.load(x_base + (t - 2) * stride_x_token, mask=m & (p >= 2), other=0.0)
            b = tl.where(
                p >= 2,
                b,
                tl.load(
                    st_base + tl.minimum(tl.maximum(p + 1, 0), 2) * stride_state_tok,
                    mask=m & (p < 2) & hs,
                    other=0.0,
                ),
            )
            c = tl.load(x_base + (t - 1) * stride_x_token, mask=m & (p >= 1), other=0.0)
            c = tl.where(
                p >= 1,
                c,
                tl.load(
                    st_base + 2 * stride_state_tok,
                    mask=m & (p < 1) & hs,
                    other=0.0,
                ),
            )
            d = tl.load(x_base + t * stride_x_token, mask=m, other=0.0)
            y = _conv_silu(a, b, c, d, w0, w1, w2, w3, acc0, SILU_ACTIVATION)
            tl.store(
                out_base + t * stride_out_token,
                y.to(out_ptr.dtype.element_ty),
                mask=m,
            )
    else:
        # One 2D load of the 16 rows (+ a 4-row halo), all in flight, then
        # rows are unpacked in registers and slid through the window.
        mask_x = mask_f[:, None] & mask_t[None, :]
        xt = tl.load(
            x_base[:, None] + offs_t[None, :] * stride_x_token, mask=mask_x, other=0.0
        )
        halo_t = t0 - 4 + tl.arange(0, 4)
        hx = tl.load(
            x_base[:, None] + halo_t[None, :] * stride_x_token,
            mask=mask_f[:, None] & (halo_t >= 0)[None, :],
            other=0.0,
        )
        _, h1, h2, h3 = _rows4(hx)
        e, o = _split_rows(xt)
        s0, s2 = _split_rows(e)  # rows 0,4,8,12 / 2,6,10,14
        s1, s3 = _split_rows(o)  # rows 1,5,9,13 / 3,7,11,15
        x0, x4, x8, x12 = _rows4(s0)
        x1, x5, x9, x13 = _rows4(s1)
        x2, x6, x10, x14 = _rows4(s2)
        x3, x7, x11, x15 = _rows4(s3)
        dt = out_ptr.dtype.element_ty
        y0 = _conv_silu(h1, h2, h3, x0, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
        y1 = _conv_silu(h2, h3, x0, x1, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
        y2 = _conv_silu(h3, x0, x1, x2, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
        y3 = _conv_silu(x0, x1, x2, x3, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
        y4 = _conv_silu(x1, x2, x3, x4, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
        y5 = _conv_silu(x2, x3, x4, x5, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
        y6 = _conv_silu(x3, x4, x5, x6, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
        y7 = _conv_silu(x4, x5, x6, x7, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
        y8 = _conv_silu(x5, x6, x7, x8, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
        y9 = _conv_silu(x6, x7, x8, x9, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
        y10 = _conv_silu(x7, x8, x9, x10, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
        y11 = _conv_silu(x8, x9, x10, x11, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
        y12 = _conv_silu(x9, x10, x11, x12, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(
            dt
        )
        y13 = _conv_silu(x10, x11, x12, x13, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(
            dt
        )
        y14 = _conv_silu(x11, x12, x13, x14, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(
            dt
        )
        y15 = _conv_silu(x12, x13, x14, x15, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(
            dt
        )
        u0 = _unrows4(y0, y4, y8, y12)
        u1 = _unrows4(y1, y5, y9, y13)
        u2 = _unrows4(y2, y6, y10, y14)
        u3 = _unrows4(y3, y7, y11, y15)
        yt = _join_rows(_join_rows(u0, u2), _join_rows(u1, u3))
        tl.store(
            out_base[:, None] + offs_t[None, :] * stride_out_token, yt, mask=mask_x
        )


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
    block_n: int = 512,
    num_warps: int = 4,
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
        block_n: channels per program (16 tokens each); must divide
            ``dim // num_planes``.
        num_warps: warps per program.

    Returns:
        ``[num_planes, T, dim // num_planes]`` output in ``x.dtype``.

    """
    assert x.dim() == 2 and x.stride(1) == 1
    num_tokens, dim = x.shape
    width = weight.shape[1]
    assert width == 4
    assert weight.shape[0] == dim and dim % num_planes == 0
    assert conv_states.shape[1] == dim and conv_states.shape[2] >= width - 1
    plane_dim = dim // num_planes
    block_n = min(block_n, plane_dim)
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
    grid = (triton.cdiv(num_tokens, 16), triton.cdiv(dim, block_n))
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
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=silu,
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
