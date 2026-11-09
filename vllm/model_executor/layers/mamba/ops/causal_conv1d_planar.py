# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Varlen causal short conv + SiLU as a tile stencil, planar output.

Written for the GLM-5.3-Flash KDA prefill path (``kda_prefill_conv=planar``);
the kernels are width-4 ``causal_conv1d_fn`` replacements for any caller.

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
  halo in both variants. A second launch recomputes the first 3 tokens of
  every sequence with the initial conv state (or zeros) as history; a third
  updates the state. causal_conv1d_fn instead loops token by token with
  4-byte loads.

Outputs and conv states are bit-identical to ``causal_conv1d_fn`` (fp32
weights accumulate through explicit fused multiply-adds, the form that
kernel's multiply-adds compile to; see ``_conv_silu``), except that rows of
sequences on a null cache slot are written (as if the sequence had no
initial state) where ``causal_conv1d_fn`` leaves them unset.
"""

import math
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import WarmupChoices
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

logger = init_logger(__name__)

# Geometry the kernels are written for: the stencil kernels unpack tiles of
# TILE_TOKENS tokens in registers (_conv_tile spells out its 16 rows), the conv
# has CONV_WIDTH taps (so a sequence's history is CONV_WIDTH - 1 tokens) and the
# per-sequence kernels (fixup, state update) take _SEQ_BLOCK_N channels each.
# Globals read inside a kernel must be tl.constexpr instances; host code
# reads their .value.
TILE_TOKENS = tl.constexpr(16)
CONV_WIDTH = tl.constexpr(4)
_SEQ_BLOCK_N = 1024

# Launch defaults of the stencil kernels, chosen with the conv-only benchmark
# (benchmarks/kernels/benchmark_glm5next_kda_prefill_conv.py) on one SM10x GPU
# at 64 to 8 local heads: 512 channels x 8 tiles of 16 tokens per program, 4
# warps, a 3-stage TMA pipeline. Nothing is autotuned; the public keyword
# arguments exist to re-measure on other GPUs.
_DEFAULT_BLOCK_N = 512
_DEFAULT_NUM_WARPS = 4
_DEFAULT_TILES = 8
_DEFAULT_NUM_STAGES = 3


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
    """Conv + SiLU of the TILE_TOKENS (16) rows ``xt`` [BN, 16] given the 3
    rows before them; returns the outputs [BN, 16] and the tile's last 3 rows
    (the next tile's history)."""
    tl.static_assert(xt.shape[1] == TILE_TOKENS)
    tl.static_assert(CONV_WIDTH == 4)
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
    # Sequence-agnostic width-4 stencil over TILES consecutive TILE_TOKENS-token
    # blocks x BLOCK_N channels: every token takes its CONV_WIDTH - 1
    # predecessors from x. The first CONV_WIDTH - 1 tokens of each sequence are
    # redone by the fixup kernel. The next block's rows are loaded before the
    # current block is computed and stored; a block's last CONV_WIDTH - 1 rows
    # are the next block's halo.
    rows = tl.arange(0, TILE_TOKENS)
    t_first = tl.program_id(0).to(tl.int64) * (TILE_TOKENS * TILES)
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

    halo_t = t_first - CONV_WIDTH + tl.arange(0, CONV_WIDTH)
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
        offs_t = t_first + it * TILE_TOKENS + rows
        nxt_t = offs_t + TILE_TOKENS
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
    # tiles; rows outside [0, num_tokens) read as zero) and TMA stores. The
    # halo is loaded once per program; a tile's last rows are the next tile's
    # history, as in the register kernel.
    t_first = tl.program_id(0) * (TILE_TOKENS * TILES)
    f0 = tl.program_id(1) * BLOCK_N
    offs_f = f0 + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    w0, w1, w2, w3, acc0 = _load_weights(
        w_ptr, bias_ptr, offs_f, mask_f, stride_w_dim, stride_w_width, HAS_BIAS
    )
    plane = f0 // plane_dim
    col0 = f0 - plane * plane_dim
    dt: tl.constexpr = out_desc.dtype
    _, h1, h2, h3 = _rows4(tl.trans(x_halo_desc.load([t_first - CONV_WIDTH, f0])))
    for it in tl.range(0, TILES, num_stages=NUM_STAGES):
        t0 = t_first + it * TILE_TOKENS
        xt = tl.trans(x_desc.load([t0, f0]))  # [BLOCK_N, TILE_TOKENS]
        yt, h1, h2, h3 = _conv_tile(
            xt, h1, h2, h3, w0, w1, w2, w3, acc0, dt, SILU_ACTIVATION
        )
        out_desc.store(
            [plane, t0, col0], tl.reshape(tl.trans(yt), [1, TILE_TOKENS, BLOCK_N])
        )


@triton.jit
def _slot_flags(
    cache_indices_ptr,
    has_initial_state_ptr,
    seq,
    stride_cache_indices,
    stride_has_initial_state,
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
):
    """The conv-state contract of sequence ``seq``, shared by the fixup and
    the state update: ``(cache_idx, null_or_pad, has_state)``. A sequence on a
    pad or null slot owns no state: nothing is read from or written to the
    pool for it and ``has_state`` is false, so its first tokens are computed
    with zero history."""
    cache_idx = tl.load(cache_indices_ptr + seq * stride_cache_indices).to(tl.int64)
    null_or_pad = (cache_idx == pad_slot_id) | (cache_idx == null_block_id)
    has_state = (
        tl.load(has_initial_state_ptr + seq * stride_has_initial_state).to(tl.int1)
        & (cache_idx != pad_slot_id)
        & (cache_idx != null_block_id)
    )
    return cache_idx, null_or_pad, has_state


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
    # Recompute the first CONV_WIDTH - 1 (3) tokens of sequence program_id(0):
    # their taps before the sequence start come from the initial conv state
    # (or zero).
    tl.static_assert(CONV_WIDTH == 4)
    seq = tl.program_id(0)
    offs_f = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    start = tl.load(query_start_loc_ptr + seq).to(tl.int64)
    seq_len = (tl.load(query_start_loc_ptr + seq + 1) - start).to(tl.int32)
    if seq_len <= 0:
        return
    cache_idx, _, has_state = _slot_flags(
        cache_indices_ptr,
        has_initial_state_ptr,
        seq,
        stride_cache_indices,
        stride_has_initial_state,
        pad_slot_id,
        null_block_id,
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
):
    # New state position k holds sequence position seq_len - (CONV_WIDTH - 1) + k:
    # from x, or (short sequences) the old state shifted left by seq_len.
    rel = seq_len - (CONV_WIDTH - 1) + k
    x_val = tl.load(
        x_ptr + (seq_start + tl.maximum(rel, 0)) * stride_x_token + offs_f,
        mask=mask_f & (rel >= 0),
        other=0.0,
    )
    s_val = tl.load(
        state_base + tl.minimum(k + seq_len, CONV_WIDTH - 2) * stride_state_tok,
        mask=mask_f & (rel < 0) & has_state,
        other=0.0,
    )
    return tl.where(rel >= 0, x_val, s_val)


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
    BLOCK_N: tl.constexpr,
):
    # Write the CONV_WIDTH - 1 (3) state positions of sequence program_id(0).
    tl.static_assert(CONV_WIDTH == 4)
    seq = tl.program_id(0)
    offs_f = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    cache_idx, null_or_pad, has_state = _slot_flags(
        cache_indices_ptr,
        has_initial_state_ptr,
        seq,
        stride_cache_indices,
        stride_has_initial_state,
        pad_slot_id,
        null_block_id,
    )
    seq_start = tl.load(query_start_loc_ptr + seq).to(tl.int64)
    seq_len = (tl.load(query_start_loc_ptr + seq + 1) - seq_start).to(tl.int32)
    if null_or_pad | (seq_len <= 0):
        return
    state_base = state_ptr + cache_idx * stride_state_seq + offs_f * stride_state_dim
    # All loads happen before the first store.
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
    )
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
    )
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
    )
    tl.debug_barrier()
    tl.store(state_base, v0, mask=mask_f)
    tl.store(state_base + stride_state_tok, v1, mask=mask_f)
    tl.store(state_base + 2 * stride_state_tok, v2, mask=mask_f)


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
    return _resolve_impl(
        impl,
        x.data_ptr() % 16 == 0 and (x.stride(0) * e) % 16 == 0,
        out_ok,
        f"x row stride {x.stride(0) * e} bytes at {x.data_ptr() % 16} mod 16",
    )


def planar_conv_impl_for_rows(
    impl: str, dtype: torch.dtype, row_stride: int, plane_dim: int
) -> str:
    """``kda_planar_conv_impl`` from the shapes alone: ``x`` rows of
    ``row_stride`` elements at an aligned base and the function's own dense
    output (used to register the JIT warmup of the kernel that will run)."""
    e = dtype.itemsize
    return _resolve_impl(
        impl,
        (row_stride * e) % 16 == 0,
        (plane_dim * e) % 16 == 0,
        f"x row stride {row_stride * e} bytes",
    )


def _resolve_impl(impl: str, x_ok: bool, out_ok: bool, x_text: str) -> str:
    if impl not in ("auto", "tma", "regs"):
        raise ValueError(f"Unsupported impl {impl!r}: 'auto', 'tma' or 'regs'")
    if impl == "regs":
        return "regs"
    tma_ok = (
        current_platform.is_cuda()
        and current_platform.has_device_capability(90)
        and x_ok
        and out_ok
    )
    if tma_ok:
        return "tma"
    if impl == "tma":
        raise ValueError(
            f"impl='tma' needs SM90+ and 16-byte aligned rows: {x_text}, output "
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


# --- launches and JIT warmup -------------------------------------------------
# Every launch goes through a dispatcher that doubles as the kernel's warmup
# entry (docs/contributing/jit_kernel_warmup.md): the dispatch function maps a
# call's tensors to the grid and the kernel arguments, and a warmup-inputs
# function describes the tensors of one layer configuration (dtype, shape,
# strides, alignment) so that the same mapping yields the compile keys at
# startup without real tensors or launches. The TMA kernel builds its
# descriptors inside the dispatch; Triton accepts them on warmup tensors.


def _warmup_bias(dtype: torch.dtype | None, dim: int) -> TritonWarmupTensor | None:
    return None if dtype is None else TritonWarmupTensor(dtype, shape=(dim,))


def _warmup_conv_states(
    dtype: torch.dtype, dim: int, dim_first: bool, state_len: int
) -> TritonWarmupTensor:
    # One slot of the pool as the kernels see it, [slot, dim, state_len] with
    # state_len >= width - 1 (plus the speculative tokens, which changes the
    # stride classes): the DS layout directly, the SD layout as the transposed
    # view kda.py passes.
    strides = (
        (state_len * dim, state_len, 1) if dim_first else (state_len * dim, 1, dim)
    )
    return TritonWarmupTensor(dtype, shape=(1, dim, state_len), strides=strides)


def _stencil_case(
    dtype, weight_dtype, bias_dtype, dim, num_planes, row_stride, num_tokens, **launch
):
    plane_dim = dim // num_planes
    case = dict(
        x=TritonWarmupTensor(dtype, shape=(num_tokens, dim), strides=(row_stride, 1)),
        weight=TritonWarmupTensor(weight_dtype, shape=(dim, CONV_WIDTH.value)),
        bias=_warmup_bias(bias_dtype, dim),
        out=TritonWarmupTensor(dtype, shape=(num_planes, num_tokens, plane_dim)),
        activation="silu",
    )
    case.update(launch)
    return case


def _stencil_regs_warmup_inputs(
    *,
    dtype,
    weight_dtype,
    bias_dtype,
    dim,
    num_planes,
    row_stride,
    block_n,
    tiles,
    num_warps,
):
    # 16 and 17 tokens: both classes of the output's plane stride.
    num_tokens = WarmupChoices(16, 17)
    return _stencil_case(
        dtype,
        weight_dtype,
        bias_dtype,
        dim,
        num_planes,
        row_stride,
        num_tokens,
        block_n=block_n,
        tiles=tiles,
        num_warps=num_warps,
    )


def _stencil_tma_warmup_inputs(
    *,
    dtype,
    weight_dtype,
    bias_dtype,
    dim,
    num_planes,
    row_stride,
    block_n,
    tiles,
    num_warps,
    num_stages,
):
    num_tokens = WarmupChoices(16, 17)
    return _stencil_case(
        dtype,
        weight_dtype,
        bias_dtype,
        dim,
        num_planes,
        row_stride,
        num_tokens,
        block_n=block_n,
        tiles=tiles,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _state_update_case(
    dtype,
    conv_state_dtype,
    state_len,
    dim,
    row_stride,
    conv_state_dim_first,
    cache_indices_stride,
    num_tokens,
):
    # The warmup replays a case through the dispatch, so a case holds exactly
    # the dispatch's parameters.
    return dict(
        x=TritonWarmupTensor(dtype, shape=(num_tokens, dim), strides=(row_stride, 1)),
        conv_states=_warmup_conv_states(
            conv_state_dtype, dim, conv_state_dim_first, state_len
        ),
        cache_indices=TritonWarmupTensor(
            torch.int32, shape=(2,), strides=(cache_indices_stride,)
        ),
        has_initial_state=TritonWarmupTensor(torch.bool, shape=(2,)),
        query_start_loc=TritonWarmupTensor(torch.int32, shape=(3,)),
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=NULL_BLOCK_ID,
    )


def _fixup_case(
    dtype,
    conv_state_dtype,
    state_len,
    weight_dtype,
    bias_dtype,
    dim,
    num_planes,
    row_stride,
    conv_state_dim_first,
    cache_indices_stride,
    num_tokens,
):
    plane_dim = dim // num_planes
    case = _state_update_case(
        dtype,
        conv_state_dtype,
        state_len,
        dim,
        row_stride,
        conv_state_dim_first,
        cache_indices_stride,
        num_tokens,
    )
    case.update(
        weight=TritonWarmupTensor(weight_dtype, shape=(dim, CONV_WIDTH.value)),
        bias=_warmup_bias(bias_dtype, dim),
        out=TritonWarmupTensor(dtype, shape=(num_planes, num_tokens, plane_dim)),
        activation="silu",
    )
    return case


def _fixup_warmup_inputs(
    *,
    dtype,
    conv_state_dtype,
    state_len,
    weight_dtype,
    bias_dtype,
    dim,
    num_planes,
    row_stride,
    conv_state_dim_first,
    cache_indices_stride,
):
    num_tokens = WarmupChoices(16, 17)
    return _fixup_case(
        dtype,
        conv_state_dtype,
        state_len,
        weight_dtype,
        bias_dtype,
        dim,
        num_planes,
        row_stride,
        conv_state_dim_first,
        cache_indices_stride,
        num_tokens,
    )


def _state_update_warmup_inputs(
    *,
    dtype,
    conv_state_dtype,
    state_len,
    dim,
    row_stride,
    conv_state_dim_first,
    cache_indices_stride,
):
    num_tokens = WarmupChoices(16, 17)
    return _state_update_case(
        dtype,
        conv_state_dtype,
        state_len,
        dim,
        row_stride,
        conv_state_dim_first,
        cache_indices_stride,
        num_tokens,
    )


def _stencil_kwargs(x, weight, bias, out, block_n, tiles, num_warps, activation):
    num_tokens, dim = x.shape
    plane_dim = out.shape[2]
    grid = (
        triton.cdiv(num_tokens, TILE_TOKENS.value * tiles),
        triton.cdiv(dim, block_n),
    )
    return grid, dict(
        w_ptr=weight,
        bias_ptr=bias,
        num_tokens=num_tokens,
        dim=dim,
        plane_dim=plane_dim,
        stride_w_dim=weight.stride(0),
        stride_w_width=weight.stride(1),
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        BLOCK_N=block_n,
        TILES=tiles,
        num_warps=num_warps,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_kda_conv_fwd_planar_kernel, warmup_inputs=_stencil_regs_warmup_inputs
)
def _planar_stencil_regs(
    x, weight, bias, out, block_n, tiles, num_warps, activation
) -> DispatchSpec:
    grid, kwargs = _stencil_kwargs(
        x, weight, bias, out, block_n, tiles, num_warps, activation
    )
    kwargs.update(
        x_ptr=x,
        out_ptr=out,
        stride_x_token=x.stride(0),
        stride_out_plane=out.stride(0),
        stride_out_token=out.stride(1),
    )
    return grid, kwargs


@triton_kernel_dispatcher_with_warmup(
    kernel=_kda_conv_fwd_planar_tma_kernel, warmup_inputs=_stencil_tma_warmup_inputs
)
def _planar_stencil_tma(
    x, weight, bias, out, block_n, tiles, num_warps, num_stages, activation
) -> DispatchSpec:
    from triton.tools.tensor_descriptor import TensorDescriptor

    num_tokens, dim = x.shape
    num_planes, _, plane_dim = out.shape
    grid, kwargs = _stencil_kwargs(
        x, weight, bias, out, block_n, tiles, num_warps, activation
    )
    kwargs.update(
        x_desc=TensorDescriptor(
            x, [num_tokens, dim], [x.stride(0), 1], [TILE_TOKENS.value, block_n]
        ),
        x_halo_desc=TensorDescriptor(
            x, [num_tokens, dim], [x.stride(0), 1], [CONV_WIDTH.value, block_n]
        ),
        out_desc=TensorDescriptor(
            out,
            [num_planes, num_tokens, plane_dim],
            [out.stride(0), out.stride(1), 1],
            [1, TILE_TOKENS.value, block_n],
        ),
        NUM_STAGES=num_stages,
    )
    return grid, kwargs


@triton_kernel_dispatcher_with_warmup(
    kernel=_kda_conv_fixup_kernel, warmup_inputs=_fixup_warmup_inputs
)
def _planar_fixup(
    x,
    weight,
    bias,
    conv_states,
    cache_indices,
    has_initial_state,
    query_start_loc,
    out,
    activation,
    pad_slot_id,
    null_block_id,
) -> DispatchSpec:
    dim, plane_dim = x.shape[1], out.shape[2]
    num_seqs = query_start_loc.shape[0] - 1
    block_n = math.gcd(_SEQ_BLOCK_N, plane_dim)
    return (num_seqs, triton.cdiv(dim, block_n)), dict(
        x_ptr=x,
        w_ptr=weight,
        bias_ptr=bias,
        state_ptr=conv_states,
        cache_indices_ptr=cache_indices,
        has_initial_state_ptr=has_initial_state,
        query_start_loc_ptr=query_start_loc,
        out_ptr=out,
        dim=dim,
        plane_dim=plane_dim,
        stride_x_token=x.stride(0),
        stride_w_dim=weight.stride(0),
        stride_w_width=weight.stride(1),
        stride_state_seq=conv_states.stride(0),
        stride_state_dim=conv_states.stride(1),
        stride_state_tok=conv_states.stride(2),
        stride_out_plane=out.stride(0),
        stride_out_token=out.stride(1),
        stride_cache_indices=cache_indices.stride(0),
        stride_has_initial_state=has_initial_state.stride(0),
        pad_slot_id=pad_slot_id,
        null_block_id=null_block_id,
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        BLOCK_N=block_n,
        num_warps=4,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_kda_conv_state_update_kernel, warmup_inputs=_state_update_warmup_inputs
)
def _planar_state_update(
    x,
    conv_states,
    cache_indices,
    has_initial_state,
    query_start_loc,
    pad_slot_id,
    null_block_id,
) -> DispatchSpec:
    dim = x.shape[1]
    num_seqs = query_start_loc.shape[0] - 1
    return (num_seqs, triton.cdiv(dim, _SEQ_BLOCK_N)), dict(
        x_ptr=x,
        state_ptr=conv_states,
        cache_indices_ptr=cache_indices,
        has_initial_state_ptr=has_initial_state,
        query_start_loc_ptr=query_start_loc,
        dim=dim,
        stride_x_token=x.stride(0),
        stride_state_seq=conv_states.stride(0),
        stride_state_dim=conv_states.stride(1),
        stride_state_tok=conv_states.stride(2),
        stride_cache_indices=cache_indices.stride(0),
        stride_has_initial_state=has_initial_state.stride(0),
        pad_slot_id=pad_slot_id,
        null_block_id=null_block_id,
        BLOCK_N=_SEQ_BLOCK_N,
        num_warps=4,
    )


def planar_conv_warmup_plan(
    *,
    dtype: torch.dtype,
    weight_dtype: torch.dtype,
    bias_dtype: torch.dtype | None,
    dim: int,
    num_planes: int,
    row_strides: tuple[int, ...],
    conv_state_dim_first: bool,
    conv_state_dtype: torch.dtype | None = None,
    state_len: int = CONV_WIDTH.value - 1,
    impl: str = "auto",
    block_n: int = _DEFAULT_BLOCK_N,
    tiles: int = _DEFAULT_TILES,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
    cache_indices_strides: tuple[int, ...] = (1, 2, 16),
) -> list[tuple[Any, dict[str, Any]]]:
    """The (dispatcher, warmup kwargs) pairs that cover one layer's planar conv.

    For each row stride of ``x`` (the fused projection's row; the dense row of
    the gathered non-spec rows of a spec-decode step) the stencil kernel that
    ``kda_planar_conv_impl`` selects for it, plus the fixup and the state
    update for the conv-state pool (``conv_state_dtype``, default the
    activation dtype; ``state_len`` positions per channel, width - 1 plus the
    speculative tokens; the layout) and every specialization class of the
    cache-index stride (contiguous, generic, multiple of 16: the block table's
    width is not known at init).
    """
    plane_dim = dim // num_planes
    block_n = math.gcd(block_n, plane_dim)
    if conv_state_dtype is None:
        conv_state_dtype = dtype
    plan: list[tuple[Any, dict[str, Any]]] = []
    for row_stride in dict.fromkeys(row_strides):
        common = dict(
            dtype=dtype,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
            dim=dim,
            num_planes=num_planes,
            row_stride=row_stride,
        )
        launch = dict(common, block_n=block_n, tiles=tiles, num_warps=num_warps)
        if planar_conv_impl_for_rows(impl, dtype, row_stride, plane_dim) == "tma":
            plan.append((_planar_stencil_tma, dict(launch, num_stages=num_stages)))
        else:
            plan.append((_planar_stencil_regs, launch))
        for cache_indices_stride in cache_indices_strides:
            per_seq = dict(
                conv_state_dtype=conv_state_dtype,
                state_len=state_len,
                conv_state_dim_first=conv_state_dim_first,
                cache_indices_stride=cache_indices_stride,
            )
            plan.append((_planar_fixup, dict(common, **per_seq)))
            plan.append(
                (
                    _planar_state_update,
                    dict(per_seq, dtype=dtype, dim=dim, row_stride=row_stride),
                )
            )
    return plan


def register_planar_conv_warmup(**config: Any) -> None:
    """Register one layer configuration's planar conv kernels for startup JIT
    warmup (``planar_conv_warmup_plan`` lists what is registered)."""
    for dispatcher, kwargs in planar_conv_warmup_plan(**config):
        dispatcher.register_warmup(**kwargs)


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
    block_n: int = _DEFAULT_BLOCK_N,
    num_warps: int = _DEFAULT_NUM_WARPS,
    tiles: int = _DEFAULT_TILES,
    impl: str = "auto",
    num_stages: int = _DEFAULT_NUM_STAGES,
    pad_slot_id: int = PAD_SLOT_ID,
    null_block_id: int = NULL_BLOCK_ID,
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
        cache_indices: ``[>= N]`` cache slot per sequence, may be strided
            (e.g. column 0 of the block table). Sequences on a pad or null
            slot read and write no state; their output rows are computed with
            zero history (``causal_conv1d_fn`` leaves them unset).
        has_initial_state: ``[>= N]`` bool, read the old state as history.
        num_planes: output planes; ``dim`` must be divisible by it.
        activation: ``"silu"``/``"swish"`` or ``None``.
        out: optional ``[num_planes, >= T, dim // num_planes]`` output with a
            dense last dim.
        block_n: channels per program (16 tokens each), a power of 2;
            reduced to the largest power of 2 dividing ``dim // num_planes``.
        num_warps: warps per program.
        tiles: consecutive 16-token blocks per program. The defaults of these
            three and of ``num_stages`` are the module's ``_DEFAULT_*`` values
            (measured on one GPU, see there).
        impl: "regs" (16-row tiles loaded into registers, the next tile
            prefetched), "tma" (TMA loads/stores through shared memory,
            pipelined over the tiles; needs SM90+ and 16-byte aligned rows and
            raises otherwise) or "auto" ("tma" where possible, else "regs";
            see ``kda_planar_conv_impl``). The kernel in use is logged once.
        num_stages: TMA pipeline depth (impl="tma").
        pad_slot_id: the cache slot that marks a padded (empty) sequence
            (vLLM's constant by default).
        null_block_id: the cache slot that marks a sequence without a slot
            (vLLM's constant by default); its rows are computed with zero
            history and no state is read or written.

    Returns:
        ``[num_planes, T, dim // num_planes]`` output in ``x.dtype``.

    """
    if x.dim() != 2 or x.stride(1) != 1:
        raise ValueError(
            "x must be a 2D [T, dim] tensor with dense channels, got shape "
            f"{tuple(x.shape)} with strides {x.stride()}"
        )
    num_tokens, dim = x.shape
    if weight.dim() != 2 or weight.shape[1] != CONV_WIDTH.value:
        raise ValueError(
            f"kda_causal_conv1d_planar implements conv width {CONV_WIDTH.value} "
            f"only, got a weight of shape {tuple(weight.shape)}"
        )
    width = weight.shape[1]
    if weight.shape[0] != dim:
        raise ValueError(f"weight has {weight.shape[0]} channels, x has {dim}")
    if bias is not None and (
        bias.dim() != 1 or bias.numel() != dim or bias.device != x.device
    ):
        raise ValueError(
            f"bias must be a [{dim}] tensor on {x.device}, got shape "
            f"{tuple(bias.shape)} on {bias.device}"
        )
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
    if tiles < 1 or num_stages < 1 or num_warps < 1 or num_warps & (num_warps - 1):
        raise ValueError(
            f"tiles ({tiles}) and num_stages ({num_stages}) must be positive and "
            f"num_warps ({num_warps}) a positive power of two"
        )
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
    if impl_used == "tma":
        _planar_stencil_tma(
            x, weight, bias, out, block_n, tiles, num_warps, num_stages, activation
        )
    else:
        _planar_stencil_regs(
            x, weight, bias, out, block_n, tiles, num_warps, activation
        )
    _planar_fixup(
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        activation,
        pad_slot_id,
        null_block_id,
    )
    _planar_state_update(
        x,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        pad_slot_id,
        null_block_id,
    )
    return out[:, :num_tokens]
