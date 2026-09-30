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

All launches are CUDA-graph safe (no host sync, no allocation other than
torch.empty on the current stream).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as _ld

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
                      N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
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
                           BN: tl.constexpr):
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
                                            num_warps=4)
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
                                                 BK=min(K, 2048), BN=BN, num_warps=4)
    return out
