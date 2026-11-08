# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Varlen causal short conv + SiLU for the KDA prefill path, planar output.

Same math and conv-state contract as ``causal_conv1d_fn`` (vLLM Triton, width
``W``; state positions ``0..W-2`` of ``conv_states[cache_indices[s]]`` hold the
``W-1`` tokens before sequence ``s`` when ``has_initial_state[s]``, and receive
its last ``W-1`` tokens), with two differences:

- The output is planar, ``[num_planes, T, dim // num_planes]``: for the merged
  q|k|v channels each of q, k and v is a dense ``[T, H * D]`` tensor, so
  consumers that need contiguous q/k/v do not copy them.
- The main kernel is a sequence-agnostic width-4 stencil over tiles of 16
  tokens x BLOCK_N channels (no per-token metadata): each tile is one 2D
  load (registers, the next tile prefetched; or TMA through shared memory,
  pipelined over TILES tiles per program), rows are unpacked in registers
  and slid through the window, and a tile's last 3 rows are the next one's
  halo. A second launch recomputes the first 3 tokens of every sequence with
  the initial conv state (or zeros) as history; a third updates the state.
  causal_conv1d_fn instead loops token by token with 4-byte loads.

Outputs and conv states are bit-identical to ``causal_conv1d_fn`` (fp32
weights accumulate through explicit fused multiply-adds, the form that
kernel's multiply-adds compile to; see ``_conv_silu``), except that rows of
sequences on a null cache slot are written (as if the sequence had no
initial state) where ``causal_conv1d_fn`` leaves them unset.
"""

import math

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

logger = init_logger(__name__)


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
    # Same accumulation order as causal_conv1d_fn (oldest tap first). With fp32
    # weights the taps are explicit fused multiply-adds (one rounding per tap):
    # that is what causal_conv1d_fn's fp32 multiply-adds compile to, and
    # written as separate multiplies and adds the compiler does not always
    # contract them, which costs 1 bf16 ulp on a few outputs per million. With
    # bf16 weights the product is a bf16 multiply in both kernels, so
    # mul-then-add stays.
    if w0.dtype == tl.float32:
        acc = tl.fma(a.to(tl.float32), w0, acc0)
        acc = tl.fma(b.to(tl.float32), w1, acc)
        acc = tl.fma(c.to(tl.float32), w2, acc)
        acc = tl.fma(d.to(tl.float32), w3, acc)
    else:
        acc = acc0 + a * w0
        acc += b * w1
        acc += c * w2
        acc += d * w3
    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))
    return acc


@triton.jit
def _conv_tile(
    xt,
    h1,
    h2,
    h3,
    w0,
    w1,
    w2,
    w3,
    acc0,
    dt: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
):
    """Conv + SiLU of 16 rows ``xt`` [BN, 16] given the 3 rows before them;
    returns the outputs [BN, 16] and the tile's last 3 rows."""
    e, o = _split_rows(xt)
    s0, s2 = _split_rows(e)  # rows 0,4,8,12 / 2,6,10,14
    s1, s3 = _split_rows(o)  # rows 1,5,9,13 / 3,7,11,15
    x0, x4, x8, x12 = _rows4(s0)
    x1, x5, x9, x13 = _rows4(s1)
    x2, x6, x10, x14 = _rows4(s2)
    x3, x7, x11, x15 = _rows4(s3)
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
    y12 = _conv_silu(x9, x10, x11, x12, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y13 = _conv_silu(x10, x11, x12, x13, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y14 = _conv_silu(x11, x12, x13, x14, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y15 = _conv_silu(x12, x13, x14, x15, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    u0 = _unrows4(y0, y4, y8, y12)
    u1 = _unrows4(y1, y5, y9, y13)
    u2 = _unrows4(y2, y6, y10, y14)
    u3 = _unrows4(y3, y7, y11, y15)
    return _join_rows(_join_rows(u0, u2), _join_rows(u1, u3)), x13, x14, x15


@triton.jit
def _load_weights(
    w_ptr,
    bias_ptr,
    offs_f,
    mask_f,
    stride_w_dim,
    stride_w_width,
    HAS_BIAS: tl.constexpr,
):
    w_row = w_ptr + offs_f * stride_w_dim
    w0 = tl.load(w_row, mask=mask_f, other=0.0)
    w1 = tl.load(w_row + stride_w_width, mask=mask_f, other=0.0)
    w2 = tl.load(w_row + 2 * stride_w_width, mask=mask_f, other=0.0)
    w3 = tl.load(w_row + 3 * stride_w_width, mask=mask_f, other=0.0)
    if HAS_BIAS:
        acc0 = tl.load(bias_ptr + offs_f, mask=mask_f, other=0.0).to(tl.float32)
    else:
        acc0 = tl.zeros(offs_f.shape, dtype=tl.float32)
    return w0, w1, w2, w3, acc0


@triton.jit(do_not_specialize=["num_tokens"])
def _kda_conv_fwd_planar_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    num_tokens,
    dim,
    plane_dim,
    stride_x_token,
    stride_w_dim,
    stride_w_width,
    stride_out_plane,
    stride_out_token,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES: tl.constexpr,
):
    # Sequence-agnostic width-4 stencil over TILES consecutive 16-token
    # blocks x BLOCK_N channels: every token takes its 3 predecessors from x.
    # The first 3 tokens of each sequence are redone by the fixup kernel.
    # The next block's rows are loaded before the current block is computed
    # and stored; a block's last 3 rows are the next block's halo.
    rows = tl.arange(0, 16)
    t_first = tl.program_id(0).to(tl.int64) * (16 * TILES)
    offs_f = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    w0, w1, w2, w3, acc0 = _load_weights(
        w_ptr, bias_ptr, offs_f, mask_f, stride_w_dim, stride_w_width, HAS_BIAS
    )
    # BLOCK_N divides plane_dim, so a program writes inside one plane.
    plane = (tl.program_id(1) * BLOCK_N) // plane_dim
    out_base = (
        out_ptr + plane.to(tl.int64) * stride_out_plane + offs_f - plane * plane_dim
    )
    x_base = x_ptr + offs_f
    dt: tl.constexpr = out_ptr.dtype.element_ty

    halo_t = t_first - 4 + tl.arange(0, 4)
    hx = tl.load(
        x_base[:, None] + halo_t[None, :] * stride_x_token,
        mask=mask_f[:, None] & ((halo_t >= 0) & (halo_t < num_tokens))[None, :],
        other=0.0,
    )
    _, h1, h2, h3 = _rows4(hx)
    xt = tl.load(
        x_base[:, None] + (t_first + rows)[None, :] * stride_x_token,
        mask=mask_f[:, None] & ((t_first + rows) < num_tokens)[None, :],
        other=0.0,
    )
    for it in range(TILES):
        offs_t = t_first + it * 16 + rows
        nxt_t = offs_t + 16
        xn = tl.load(
            x_base[:, None] + nxt_t[None, :] * stride_x_token,
            mask=mask_f[:, None] & ((nxt_t < num_tokens) & (it + 1 < TILES))[None, :],
            other=0.0,
        )
        yt, h1, h2, h3 = _conv_tile(
            xt, h1, h2, h3, w0, w1, w2, w3, acc0, dt, SILU_ACTIVATION
        )
        tl.store(
            out_base[:, None] + offs_t[None, :] * stride_out_token,
            yt,
            mask=mask_f[:, None] & (offs_t < num_tokens)[None, :],
        )
        xt = xn


@triton.jit(do_not_specialize=["num_tokens"])
def _kda_conv_fwd_planar_tma_kernel(
    x_desc,
    x_halo_desc,
    out_desc,
    w_ptr,
    bias_ptr,
    num_tokens,
    dim,
    plane_dim,
    stride_w_dim,
    stride_w_width,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # As _kda_conv_fwd_planar_kernel, with TMA loads (pipelined over the
    # tiles; rows outside [0, num_tokens) read as zero) and TMA stores.
    t_first = tl.program_id(0) * (16 * TILES)
    f0 = tl.program_id(1) * BLOCK_N
    offs_f = f0 + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    w0, w1, w2, w3, acc0 = _load_weights(
        w_ptr, bias_ptr, offs_f, mask_f, stride_w_dim, stride_w_width, HAS_BIAS
    )
    plane = f0 // plane_dim
    col0 = f0 - plane * plane_dim
    dt: tl.constexpr = out_desc.dtype
    for it in tl.range(0, TILES, num_stages=NUM_STAGES):
        t0 = t_first + it * 16
        _, h1, h2, h3 = _rows4(tl.trans(x_halo_desc.load([t0 - 4, f0])))
        xt = tl.trans(x_desc.load([t0, f0]))  # [BLOCK_N, 16]
        yt, _, _, _ = _conv_tile(
            xt, h1, h2, h3, w0, w1, w2, w3, acc0, dt, SILU_ACTIVATION
        )
        out_desc.store([plane, t0, col0], tl.reshape(tl.trans(yt), [1, 16, BLOCK_N]))


@triton.jit
def _kda_conv_fixup_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    state_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    query_start_loc_ptr,
    out_ptr,
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
    # Recompute the first 3 tokens of sequence program_id(0): their taps
    # before the sequence start come from the initial conv state (or zero).
    seq = tl.program_id(0)
    offs_f = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    start = tl.load(query_start_loc_ptr + seq).to(tl.int64)
    seq_len = (tl.load(query_start_loc_ptr + seq + 1) - start).to(tl.int32)
    if seq_len <= 0:
        return
    cache_idx = tl.load(cache_indices_ptr + seq * stride_cache_indices).to(tl.int64)
    has_state = (
        tl.load(has_initial_state_ptr + seq * stride_has_initial_state).to(tl.int1)
        & (cache_idx != pad_slot_id)
        & (cache_idx != null_block_id)
    )
    cache_idx = tl.where(has_state, cache_idx, 0)
    w0, w1, w2, w3, acc0 = _load_weights(
        w_ptr, bias_ptr, offs_f, mask_f, stride_w_dim, stride_w_width, HAS_BIAS
    )
    plane = (tl.program_id(1) * BLOCK_N) // plane_dim
    out_base = (
        out_ptr + plane.to(tl.int64) * stride_out_plane + offs_f - plane * plane_dim
    )
    x_base = x_ptr + offs_f
    st_base = state_ptr + cache_idx * stride_state_seq + offs_f * stride_state_dim
    m_st = mask_f & has_state
    s0 = tl.load(st_base, mask=m_st, other=0.0)
    s1 = tl.load(st_base + stride_state_tok, mask=m_st, other=0.0)
    s2 = tl.load(st_base + 2 * stride_state_tok, mask=m_st, other=0.0)
    x0 = tl.load(x_base + start * stride_x_token, mask=mask_f, other=0.0)
    y = _conv_silu(s0, s1, s2, x0, w0, w1, w2, w3, acc0, SILU_ACTIVATION)
    tl.store(
        out_base + start * stride_out_token, y.to(out_ptr.dtype.element_ty), mask=mask_f
    )
    m1 = mask_f & (seq_len > 1)
    x1 = tl.load(x_base + (start + 1) * stride_x_token, mask=m1, other=0.0)
    y = _conv_silu(s1, s2, x0, x1, w0, w1, w2, w3, acc0, SILU_ACTIVATION)
    tl.store(
        out_base + (start + 1) * stride_out_token,
        y.to(out_ptr.dtype.element_ty),
        mask=m1,
    )
    m2 = mask_f & (seq_len > 2)
    x2 = tl.load(x_base + (start + 2) * stride_x_token, mask=m2, other=0.0)
    y = _conv_silu(s2, x0, x1, x2, w0, w1, w2, w3, acc0, SILU_ACTIVATION)
    tl.store(
        out_base + (start + 2) * stride_out_token,
        y.to(out_ptr.dtype.element_ty),
        mask=m2,
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


def kda_planar_conv_impl(
    impl: str,
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    num_planes: int = 3,
) -> str:
    """The kernel ``kda_causal_conv1d_planar(x, ..., impl=impl)`` runs: "tma"
    or "regs".

    TMA descriptors need SM90+ and 16-byte aligned base addresses and row
    strides for ``x`` and ``out`` (``out=None`` stands for the dense output
    the function allocates itself). ``impl="auto"`` takes the TMA kernel when
    they can address the tensors and the register kernel otherwise;
    ``impl="tma"`` raises instead of falling back, so a caller that expects
    the TMA kernel is told rather than served the slower one; ``impl="regs"``
    is the register kernel.
    """
    if impl not in ("auto", "tma", "regs"):
        raise ValueError(f"Unsupported impl {impl!r}: 'auto', 'tma' or 'regs'")
    if impl == "regs":
        return "regs"
    e = x.element_size()
    if out is None:
        # A fresh allocation: 16-byte aligned base, both strides multiples of
        # the plane width.
        plane_dim = x.shape[1] // num_planes
        out_ok = (plane_dim * e) % 16 == 0
    else:
        out_ok = (
            out.data_ptr() % 16 == 0
            and (out.stride(0) * e) % 16 == 0
            and (out.stride(1) * e) % 16 == 0
        )
    tma_ok = (
        current_platform.is_cuda()
        and current_platform.has_device_capability(90)
        and x.data_ptr() % 16 == 0
        and (x.stride(0) * e) % 16 == 0
        and out_ok
    )
    if tma_ok:
        return "tma"
    if impl == "tma":
        raise ValueError(
            "impl='tma' needs SM90+ and 16-byte aligned rows: x row stride "
            f"{x.stride(0) * e} bytes at {x.data_ptr() % 16} mod 16, output "
            f"{'aligned' if out_ok else 'misaligned'}; use impl='auto' to fall "
            "back to the register kernel"
        )
    return "regs"


# Dispatch guard. The planar path (three launches, no per-token metadata) beats
# causal_conv1d_fn once the prefill tokens pay for the launches and for the
# single-token decode rows that share the step: a decode row costs the planar
# path about twice what it costs the stock kernel, about 6 prefill tokens of
# saving. Break-even from a conv-only sweep on one SM10x GPU (bf16; local head
# counts of the TP1 to TP8 shards of GLM-5.3-Flash): about 0.5K prefill tokens
# at 64 heads, 1K at 32, 1.8K at 16 and 3.5K at 8 for a single sequence, plus
# 5 to 6 tokens per decode row at every head count. The intercepts below add a
# margin; below 8 heads nothing was measured and the trend puts break-even
# higher still. Re-measure on other GPUs before changing them.
PLANAR_MIN_PREFILL_TOKENS = {64: 1024, 32: 1536, 16: 2048, 8: 4096}
PLANAR_TOKENS_PER_DECODE_ROW = 6


def planar_conv_pays_off(
    num_prefill_tokens: int, num_decode_rows: int, local_num_heads: int
) -> bool:
    """Whether ``kda_causal_conv1d_planar`` is expected to beat
    ``causal_conv1d_fn`` on a step with ``num_prefill_tokens`` prefill tokens
    and ``num_decode_rows`` single-token rows at ``local_num_heads`` heads:
    ``prefill tokens > intercept(heads) + 6 x decode rows``, with the intercept
    of the largest tabulated head count not above ``local_num_heads`` (twice
    the largest intercept below the table).
    """
    intercept = 2 * max(PLANAR_MIN_PREFILL_TOKENS.values())
    for heads, tokens in sorted(PLANAR_MIN_PREFILL_TOKENS.items(), reverse=True):
        if local_num_heads >= heads:
            intercept = tokens
            break
    return (
        num_prefill_tokens > intercept + PLANAR_TOKENS_PER_DECODE_ROW * num_decode_rows
    )


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
    tiles: int = 8,
    impl: str = "auto",
    num_stages: int = 3,
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
        block_n: channels per program (16 tokens each), a power of 2;
            reduced to the largest power of 2 dividing ``dim // num_planes``.
        num_warps: warps per program.
        tiles: consecutive 16-token blocks per program.
        impl: "regs" (16-row tiles loaded into registers, the next tile
            prefetched), "tma" (TMA loads/stores through shared memory,
            pipelined over the tiles; needs SM90+ and 16-byte aligned rows and
            raises otherwise) or "auto" ("tma" where possible, else "regs";
            see ``kda_planar_conv_impl``). The kernel in use is logged once.
        num_stages: TMA pipeline depth (impl="tma").

    Returns:
        ``[num_planes, T, dim // num_planes]`` output in ``x.dtype``.

    """
    if x.dim() != 2 or x.stride(1) != 1:
        raise ValueError(
            "x must be a 2D [T, dim] tensor with dense channels, got shape "
            f"{tuple(x.shape)} with strides {x.stride()}"
        )
    num_tokens, dim = x.shape
    if weight.dim() != 2 or weight.shape[1] != 4:
        raise ValueError(
            "kda_causal_conv1d_planar implements conv width 4 only, got a weight "
            f"of shape {tuple(weight.shape)}"
        )
    width = weight.shape[1]
    if weight.shape[0] != dim:
        raise ValueError(f"weight has {weight.shape[0]} channels, x has {dim}")
    if num_planes < 1 or dim % num_planes != 0:
        raise ValueError(f"dim {dim} is not divisible by num_planes {num_planes}")
    if (
        conv_states.dim() != 3
        or conv_states.shape[1] != dim
        or conv_states.shape[2] < width - 1
    ):
        raise ValueError(
            f"conv_states must be [num_slots, {dim}, >= {width - 1}], got shape "
            f"{tuple(conv_states.shape)}"
        )
    if activation not in ("silu", "swish", None):
        raise ValueError(
            f"Unsupported activation {activation!r}: 'silu', 'swish' or None"
        )
    if block_n < 1 or block_n & (block_n - 1):
        raise ValueError(f"block_n must be a power of two, got {block_n}")
    num_seqs = query_start_loc.numel() - 1
    if cache_indices.numel() < num_seqs or has_initial_state.numel() < num_seqs:
        raise ValueError(
            f"cache_indices ({cache_indices.numel()}) and has_initial_state "
            f"({has_initial_state.numel()}) need an entry per sequence ({num_seqs})"
        )
    plane_dim = dim // num_planes
    # Programs must not straddle planes.
    block_n = math.gcd(block_n, plane_dim)
    if out is None:
        out = torch.empty(
            (num_planes, num_tokens, plane_dim), dtype=x.dtype, device=x.device
        )
    elif (
        out.dim() != 3
        or out.shape[0] != num_planes
        or out.shape[1] < num_tokens
        or out.shape[2] != plane_dim
        or out.stride(2) != 1
    ):
        raise ValueError(
            f"out must be [{num_planes}, >= {num_tokens}, {plane_dim}] with a "
            f"dense last dim, got shape {tuple(out.shape)} with strides "
            f"{out.stride()}"
        )
    elif out.dtype != x.dtype or out.device != x.device:
        raise TypeError(
            f"out must match x in dtype and device ({x.dtype}, {x.device}), got "
            f"{out.dtype}, {out.device}"
        )
    impl_used = kda_planar_conv_impl(impl, x, out, num_planes)
    logger.info_once(
        "kda_causal_conv1d_planar: %s kernel for %d channels in %d planes (x row "
        "stride %d bytes, impl=%r)",
        impl_used,
        dim,
        num_planes,
        x.stride(0) * x.element_size(),
        impl,
    )
    if num_tokens == 0 or num_seqs <= 0:
        return out[:, :num_tokens]
    silu = activation in ("silu", "swish")
    grid = (triton.cdiv(num_tokens, 16 * tiles), triton.cdiv(dim, block_n))
    meta = dict(
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=silu,
        BLOCK_N=block_n,
        TILES=tiles,
        num_warps=num_warps,
    )
    if impl_used == "tma":
        from triton.tools.tensor_descriptor import TensorDescriptor

        x_desc = TensorDescriptor(x, [num_tokens, dim], [x.stride(0), 1], [16, block_n])
        x_halo_desc = TensorDescriptor(
            x, [num_tokens, dim], [x.stride(0), 1], [4, block_n]
        )
        out_desc = TensorDescriptor(
            out,
            [num_planes, num_tokens, plane_dim],
            [out.stride(0), out.stride(1), 1],
            [1, 16, block_n],
        )
        _kda_conv_fwd_planar_tma_kernel[grid](
            x_desc,
            x_halo_desc,
            out_desc,
            weight,
            bias,
            num_tokens,
            dim,
            plane_dim,
            weight.stride(0),
            weight.stride(1),
            NUM_STAGES=num_stages,
            **meta,
        )
    else:
        _kda_conv_fwd_planar_kernel[grid](
            x,
            weight,
            bias,
            out,
            num_tokens,
            dim,
            plane_dim,
            x.stride(0),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            **meta,
        )
    fixup_block_n = math.gcd(1024, plane_dim)
    _kda_conv_fixup_kernel[(num_seqs, triton.cdiv(dim, fixup_block_n))](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
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
        BLOCK_N=fixup_block_n,
        num_warps=4,
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
