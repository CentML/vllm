# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the sigmoid-gated shared expert of Qwen3-Next /
Qwen3.5 MoE blocks.

  seg_scale_(out, g)
      out[m, :] <- bf16(f32(bf16(sigmoid(f32 g[m]))) * f32(out[m, :])), in
      place. Bit-identical to torch ``F.sigmoid(g) * out`` (bf16); replaces
      the sigmoid kernel and the non-vectorized [M, 1]-broadcast mul kernel.
  seg_gemv_scale_(x, w, out)
      g[m] = bf16(sum_k x[m, k] * w[k]) (fp32 accumulation) computed in the
      kernel, then as above. Also removes the N=1 gate GEMM. g is not
      bit-equal to cuBLAS (different fp32 summation order): about one bf16
      ulp of g on a few rows.
  seg_route_fold(logits, E, K)
      Fused routing for the "shared expert = routed expert E" fold:
      softmax(logits[:, :E]) -> top-K -> renormalize, then append
      (E, sigmoid(logits[:, E])). Returns ids int32 [M, K + 1] and weights
      [M, K + 1].

All launches are CUDA-graph safe (no host sync, no allocation other than
torch.empty on the current stream).
"""

import os

import torch

from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as _ld
from vllm.lcd_pdl.triton_switch import lcd_pdl_triton_on as _lcd_pdl_on  # noqa: E402

# ruff: noqa: E501
# Kernel sources are kept verbatim (Triton cache keys hash the source).
# fmt: off


@triton.jit
def _sigmoid_bf16_exact(g):
    # torch CUDA sigmoid (opmath fp32): 1 / (1 + exp(-x)), IEEE expf + IEEE div, then round to bf16.
    e = _ld.exp(-g)
    s = _ld.div_rn(1.0, 1.0 + e)
    return s.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _seg_scale_kernel(out_ptr, g_ptr, M, stride_om, stride_g,
                      N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M
    g = tl.load(g_ptr + rows * stride_g, mask=rmask, other=0.0).to(tl.float32)
    s = _sigmoid_bf16_exact(g)[:, None]
    for n0 in tl.static_range(0, N, BN):
        cols = n0 + tl.arange(0, BN)
        offs = rows[:, None] * stride_om + cols[None, :]
        m2 = rmask[:, None]
        o = tl.load(out_ptr + offs, mask=m2, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, (s * o).to(tl.bfloat16), mask=m2)


@triton.jit
def _seg_gemv_scale_kernel(x_ptr, w_ptr, out_ptr, M, stride_xm, stride_om,
                           K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                           BN: tl.constexpr, launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M
    acc = tl.zeros([BM], dtype=tl.float32)
    for k0 in tl.static_range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :], mask=rmask[:, None], other=0.0)
        w = tl.load(w_ptr + ks)
        acc += tl.sum(x.to(tl.float32) * w.to(tl.float32)[None, :], axis=1)
    g = acc.to(tl.bfloat16).to(tl.float32)  # GEMM output dtype is bf16
    s = _sigmoid_bf16_exact(g)[:, None]
    for n0 in tl.static_range(0, N, BN):
        cols = n0 + tl.arange(0, BN)
        offs = rows[:, None] * stride_om + cols[None, :]
        m2 = rmask[:, None]
        o = tl.load(out_ptr + offs, mask=m2, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, (s * o).to(tl.bfloat16), mask=m2)


def _cfg(M, N):
    # 2 rows x 2048 cols per program at large M (enough CTAs to fill the GPU), 1 row at small M.
    BM = 1 if M <= 1024 else 2
    BN = 2048 if N % 2048 == 0 else (1024 if N % 1024 == 0 else 256)
    return BM, BN


def seg_scale_(out: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """In-place out *= sigmoid(g) with torch-identical bf16 rounding. out [M,N] bf16 (row-contig), g [M,1]."""
    M, N = out.shape
    if M == 0:
        return out
    assert out.stride(1) == 1 and out.dtype == torch.bfloat16
    BM, BN = _cfg(M, N)
    _seg_scale_kernel[(triton.cdiv(M, BM),)](out, g, M, out.stride(0), g.stride(0), N=N, BM=BM, BN=BN,
                                            num_warps=4, launch_pdl=_lcd_pdl_on())
    return out


def seg_gemv_scale_(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """In-place out *= sigmoid(bf16(x @ w^T)); x [M,K] bf16, w [1,K] bf16, out [M,N] bf16."""
    M, N = out.shape
    if M == 0:
        return out
    K = x.shape[1]
    assert x.stride(1) == 1 and out.stride(1) == 1 and w.is_contiguous()
    BM, BN = _cfg(M, N)
    _seg_gemv_scale_kernel[(triton.cdiv(M, BM),)](x, w, out, M, x.stride(0), out.stride(0), K=K, N=N, BM=BM,
                                                 BK=min(K, 2048), BN=BN, num_warps=4, launch_pdl=_lcd_pdl_on())
    return out


@triton.jit
def _route_fold_kernel(lg_ptr, ids_ptr, w_ptr, M, stride_l,
                       E: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, BT: tl.constexpr, launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)
    rmask = rows < M
    cols = tl.arange(0, E)
    v = tl.load(lg_ptr + rows[:, None] * stride_l + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    mx = tl.max(v, axis=1)
    kidx = tl.arange(0, KP)
    sel_v = tl.zeros([BT, KP], dtype=tl.float32)
    sel_i = tl.zeros([BT, KP], dtype=tl.int32)
    cur = v
    for j in tl.static_range(K):
        m = tl.max(cur, axis=1)
        # lowest index among ties
        cand = tl.where(cur == m[:, None], cols[None, :], E)
        i = tl.min(cand, axis=1)
        sel_v = tl.where(kidx[None, :] == j, m[:, None], sel_v)
        sel_i = tl.where(kidx[None, :] == j, i[:, None], sel_i)
        cur = tl.where(cols[None, :] == i[:, None], float("-inf"), cur)
    p = tl.where(kidx[None, :] < K, tl.exp(sel_v - mx[:, None]), 0.0)
    p = p / tl.sum(p, axis=1)[:, None]
    gs = tl.load(lg_ptr + rows * stride_l + E, mask=rmask, other=0.0).to(tl.float32)
    s = _ld.div_rn(1.0, 1.0 + _ld.exp(-gs))
    p = tl.where(kidx[None, :] == K, s[:, None], p)
    sel_i = tl.where(kidx[None, :] == K, E, sel_i)
    om = rmask[:, None] & (kidx[None, :] <= K)
    offs = rows[:, None] * (K + 1) + kidx[None, :]
    tl.store(ids_ptr + offs, sel_i, mask=om)
    tl.store(w_ptr + offs, p.to(w_ptr.dtype.element_ty), mask=om)


@triton.jit
def _route_fold_packed_kernel(lg_ptr, ids_ptr, w_ptr, M, stride_l,
                              E: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, BT: tl.constexpr, launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # bf16 logits only: fp32(bf16) has 16 zero low bits, so (order-preserving int32 key | (E-1-idx)) is a unique
    # sortable key -> one max-reduction per top-k round (ties -> lowest expert index), value recovered from the key.
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)
    rmask = rows < M
    cols = tl.arange(0, E)
    v = tl.load(lg_ptr + rows[:, None] * stride_l + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    b = v.to(tl.int32, bitcast=True)
    key = b ^ ((b >> 31) & 0x7FFFFFFF)
    key = (key & -65536) | (E - 1 - cols)[None, :]
    kidx = tl.arange(0, KP)
    sel_v = tl.zeros([BT, KP], dtype=tl.float32)
    sel_i = tl.zeros([BT, KP], dtype=tl.int32)
    for j in tl.static_range(K):
        kmax = tl.max(key, axis=1)
        i = (E - 1) - (kmax & 0xFFFF)
        hb = kmax & -65536
        vb = (hb ^ ((hb >> 31) & 0x7FFFFFFF)) & -65536
        val = vb.to(tl.float32, bitcast=True)
        sel_v = tl.where(kidx[None, :] == j, val[:, None], sel_v)
        sel_i = tl.where(kidx[None, :] == j, i[:, None], sel_i)
        key = tl.where(key == kmax[:, None], -2147483647 - 1, key)
    mx = tl.max(sel_v, axis=1)
    p = tl.where(kidx[None, :] < K, tl.exp(sel_v - mx[:, None]), 0.0)
    p = p / tl.sum(p, axis=1)[:, None]
    gs = tl.load(lg_ptr + rows * stride_l + E, mask=rmask, other=0.0).to(tl.float32)
    s = _ld.div_rn(1.0, 1.0 + _ld.exp(-gs))
    p = tl.where(kidx[None, :] == K, s[:, None], p)
    sel_i = tl.where(kidx[None, :] == K, E, sel_i)
    om = rmask[:, None] & (kidx[None, :] <= K)
    offs = rows[:, None] * (K + 1) + kidx[None, :]
    tl.store(ids_ptr + offs, sel_i, mask=om)
    tl.store(w_ptr + offs, p.to(w_ptr.dtype.element_ty), mask=om)


@triton.jit
def _route_fold_packed_pad_kernel(lg_ptr, pad_ptr, cnt_ptr, ids_ptr, w_ptr, M, stride_l,
                                  E: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, BT: tl.constexpr,
                                  PROBE: tl.constexpr, launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # Same instructions as _route_fold_packed_kernel for rows with pad[row] == False (ids / weights of real rows are
    # bit-identical). Rows with pad[row] == True (CUDA-graph padding rows, vLLM's forward_context.is_padding) get
    # expert id -1 in all K+1 slots and weight 0: the trtllm routing kernels (block / cluster / coop and the exact
    # single-CTA routing) treat -1 as a non-local expert, so the row is not permuted, reads no expert weights, and the
    # finalize (trtllm or the QGF gather, idx >= 0) gives it a zero MoE output.
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)
    rmask = rows < M
    cols = tl.arange(0, E)
    v = tl.load(lg_ptr + rows[:, None] * stride_l + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    b = v.to(tl.int32, bitcast=True)
    key = b ^ ((b >> 31) & 0x7FFFFFFF)
    key = (key & -65536) | (E - 1 - cols)[None, :]
    kidx = tl.arange(0, KP)
    sel_v = tl.zeros([BT, KP], dtype=tl.float32)
    sel_i = tl.zeros([BT, KP], dtype=tl.int32)
    for j in tl.static_range(K):
        kmax = tl.max(key, axis=1)
        i = (E - 1) - (kmax & 0xFFFF)
        hb = kmax & -65536
        vb = (hb ^ ((hb >> 31) & 0x7FFFFFFF)) & -65536
        val = vb.to(tl.float32, bitcast=True)
        sel_v = tl.where(kidx[None, :] == j, val[:, None], sel_v)
        sel_i = tl.where(kidx[None, :] == j, i[:, None], sel_i)
        key = tl.where(key == kmax[:, None], -2147483647 - 1, key)
    mx = tl.max(sel_v, axis=1)
    p = tl.where(kidx[None, :] < K, tl.exp(sel_v - mx[:, None]), 0.0)
    p = p / tl.sum(p, axis=1)[:, None]
    gs = tl.load(lg_ptr + rows * stride_l + E, mask=rmask, other=0.0).to(tl.float32)
    s = _ld.div_rn(1.0, 1.0 + _ld.exp(-gs))
    p = tl.where(kidx[None, :] == K, s[:, None], p)
    sel_i = tl.where(kidx[None, :] == K, E, sel_i)
    pad = tl.load(pad_ptr + rows, mask=rmask, other=0).to(tl.int32) != 0
    sel_i = tl.where(pad[:, None], -1, sel_i)
    p = tl.where(pad[:, None], 0.0, p)
    om = rmask[:, None] & (kidx[None, :] <= K)
    offs = rows[:, None] * (K + 1) + kidx[None, :]
    tl.store(ids_ptr + offs, sel_i, mask=om)
    tl.store(w_ptr + offs, p.to(w_ptr.dtype.element_ty), mask=om)
    if PROBE:
        npad = tl.sum((pad & rmask).to(tl.int32), axis=0)
        nreal = tl.sum(((pad == 0) & rmask).to(tl.int32), axis=0)
        tl.atomic_add(cnt_ptr, npad)
        tl.atomic_add(cnt_ptr + 1, nreal)


_ROUTE_PLAIN = False  # tests: force the unpacked reference kernel
# Opt-in faster launch of the packed routing kernels for M >= VLLM_SEG_ROUTE_FAST_MIN tokens (0 = off): 2 rows per
# program, 1 warp. Every (rows/program, warps) launch computes each row with identical instructions (bitwise-equal ids
# and weights, checked for BT 1..32 x warps 1..8 at 2.4K-16K tokens on GB300); this one is ~2x faster at mixed-step sizes.
_ROUTE_FAST_MIN = int(os.environ.get("VLLM_SEG_ROUTE_FAST_MIN", "0"))
if _ROUTE_FAST_MIN > 0:
    from vllm.logger import init_logger as _init_logger

    _init_logger(__name__).info(
        "SEG-fold routing: fast launch (2 rows/program, 1 warp) for M >= %d tokens", _ROUTE_FAST_MIN)


def _route_launch(M: int) -> tuple[int, int]:
    if _ROUTE_FAST_MIN > 0 and M >= _ROUTE_FAST_MIN:
        return 2, 1
    return (8 if M >= 4096 else 4), 4


def seg_route_fold(logits: torch.Tensor, E: int = 256, K: int = 8, w_dtype: torch.dtype = torch.bfloat16,
                   pad: torch.Tensor | None = None, cnt: torch.Tensor | None = None):
    """logits [M, >=E+1] (bf16/fp32, row stride arbitrary, unit col stride) -> ids int32 [M,K+1], w [M,K+1] (w_dtype:
    bf16 for the trtllm routed MoE, fp32 for consumers that take fp32 routing weights).
    pad: optional bool [>= M] device mask (True = CUDA-graph padding row -> ids -1, weights 0; bf16 logits only);
    cnt: optional int32 [2] device counters (padded rows, real rows) for an engage probe."""
    M = logits.shape[0]
    ids = torch.empty(M, K + 1, dtype=torch.int32, device=logits.device)
    w = torch.empty(M, K + 1, dtype=w_dtype, device=logits.device)
    if M and pad is not None and logits.dtype == torch.bfloat16 and E <= 65536 and not _ROUTE_PLAIN:
        BT, NW = _route_launch(M)
        _route_fold_packed_pad_kernel[(triton.cdiv(M, BT),)](logits, pad, cnt if cnt is not None else ids, ids, w, M,
                                                             logits.stride(0), E=E, K=K,
                                                             KP=triton.next_power_of_2(K + 1), BT=BT,
                                                             PROBE=cnt is not None, num_warps=NW,
                                                             launch_pdl=_lcd_pdl_on())
        return ids, w
    if M:
        if logits.dtype == torch.bfloat16 and E <= 65536 and not _ROUTE_PLAIN:
            BT, NW = _route_launch(M)
            _route_fold_packed_kernel[(triton.cdiv(M, BT),)](logits, ids, w, M, logits.stride(0), E=E, K=K,
                                                            KP=triton.next_power_of_2(K + 1), BT=BT, num_warps=NW, launch_pdl=_lcd_pdl_on())
        else:
            BT = 4
            _route_fold_kernel[(triton.cdiv(M, BT),)](logits, ids, w, M, logits.stride(0), E=E, K=K,
                                                     KP=triton.next_power_of_2(K + 1), BT=BT, num_warps=4, launch_pdl=_lcd_pdl_on())
    return ids, w
