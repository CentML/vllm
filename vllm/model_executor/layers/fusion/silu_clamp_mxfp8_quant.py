# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SwiGLU-with-clamp + MXFP8 (F8_128x4 swizzled ue8m0) activation quant.

Producer of the QuantizedActivation contract (``kMxfp8Dynamic``) for a
``SiluAndMulWithClamp`` feeding an MXFP8 linear (e.g. the DeepSeek-V4.1 shared
expert's down_proj on the FlashInfer MegaMoE path). It replaces two launches,
the activation and the down_proj input quantization, with one, and never
materialises the BF16 activation:

    act = bf16( g * sigmoid(alpha * g) * (u + beta) ),
        g = min(x[:, :d], limit), u = clamp(x[:, d:], -limit, limit)   (fp32 math)
    (xq, sf) = mxfp8_quantize_swizzled(act)

The activation is the fp32 single-rounding form of
``SiluAndMulWithClamp.forward_native`` (what torch.compile generates for the
op); the quantization is bit-identical to
``mxfp8_utils.mxfp8_quantize_swizzled_triton`` (== FlashInfer's cute-dsl
MXFP8 quantizer) applied to that BF16 activation.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8


@triton.jit(do_not_specialize=["M", "padded_m"])
def _silu_clamp_mxfp8_kernel(
    x,
    xq,
    scales,
    M,
    padded_m,
    x_stride,
    limit,
    alpha,
    beta,
    D: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    NB: tl.constexpr = BK // 32
    PADDED_COLS: tl.constexpr = D // 32
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BK + tl.arange(0, BK)
    rmask = rows < M
    r64 = rows.to(tl.int64)
    g = tl.load(
        x + r64[:, None] * x_stride + cols[None, :], rmask[:, None], other=0.0
    ).to(tl.float32)
    u = tl.load(
        x + r64[:, None] * x_stride + D + cols[None, :], rmask[:, None], other=0.0
    ).to(tl.float32)
    if HAS_CLAMP:
        g = tl.minimum(g, limit)
        u = tl.maximum(tl.minimum(u, limit), -limit)
    act = g * tl.sigmoid(alpha * g) * (u + beta)
    # The unfused path hands a BF16 activation to the quantizer.
    v = act.to(tl.bfloat16).to(tl.float32)
    gr = tl.reshape(v, (BM, NB, 32))
    amax = tl.max(tl.abs(gr), 2)
    # FlashInfer float_to_ue8m0_fast / ue8m0_to_inv_scale_fast (as in
    # mxfp8_utils._mxfp8_quant_swizzled_triton_kernel).
    normalized = amax * (1.0 / 448.0)
    bits = normalized.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.minimum(exponent + bump, 254)
    sf = tl.where(normalized <= 0, 0, sf)
    inv_bits = tl.where(sf == 0, 0, (254 - sf) << 23)
    inv_scale = inv_bits.to(tl.float32, bitcast=True)
    q = gr * inv_scale[:, :, None]
    q = tl.maximum(tl.minimum(q, 448.0), -448.0)
    q = tl.reshape(q, (BM, BK)).to(tl.float8e4nv)
    tl.store(xq + r64[:, None] * D + cols[None, :], q, rmask[:, None])
    groups = tl.program_id(1) * NB + tl.arange(0, NB)
    sf = tl.where(rmask[:, None], sf, 0).to(tl.uint8)
    # F8_128x4: [row/128, group/4, row%32, row%128/32, group%4].
    offsets = (
        r64[:, None] // 128 * (128 * PADDED_COLS)
        + groups[None, :] // 4 * 512
        + r64[:, None] % 32 * 16
        + r64[:, None] % 128 // 32 * 4
        + groups[None, :] % 4
    )
    tl.store(scales + offsets, sf, rows[:, None] < padded_m)


def silu_and_mul_clamp_mxfp8_quant(
    x: torch.Tensor,
    limit: float | None,
    alpha: float = 1.0,
    beta: float = 0.0,
    block_m: int = 16,
    block_k: int = 0,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 ``[M, 2d]`` gate/up -> (e4m3 ``[M, d]``, flat F8_128x4 ue8m0 scales).

    ``d % 128 == 0``. Scale buffer: ``round_up(M, 128) * d / 32`` bytes (the same
    allocation as ``mxfp8_quantize_swizzled_triton``)."""
    assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.stride(1) == 1
    M, two_d = x.shape
    d = two_d // 2
    assert d % 128 == 0, d
    if not block_k:
        block_k = 512 if d % 512 == 0 else (256 if d % 256 == 0 else 128)
    padded_m = (M + 127) // 128 * 128
    xq = torch.empty((M, d), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    scales = torch.empty(padded_m * (d // 32), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    if padded_m:
        from vllm.platforms import current_platform

        launch_pdl = current_platform.is_arch_support_pdl()
        _silu_clamp_mxfp8_kernel[(triton.cdiv(padded_m, block_m), d // block_k)](
            x,
            xq,
            scales,
            M,
            padded_m,
            x.stride(0),
            float(limit) if limit is not None else 0.0,
            float(alpha),
            float(beta),
            D=d,
            BM=block_m,
            BK=block_k,
            HAS_CLAMP=limit is not None,
            LAUNCH_PDL=launch_pdl,
            num_warps=num_warps,
            launch_pdl=launch_pdl,
        )
    return xq, scales
