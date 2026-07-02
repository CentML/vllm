# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct fp8 -> NVFP4 activation quantization.

Produces packed e2m1 data + swizzled e4m3 block scales byte-identical to
``scaled_fp4_quant(x.to(bf16), backend="flashinfer-cutlass")``, without the
bf16 round-trip. Used with ``attention_config.fp8_attention_output``. The
kernel is launched via ``wrap_triton`` into ``create_fp4_output_tensors``
buffers, so it is safe under torch.compile piecewise CUDA graph capture.
"""

import torch
from torch.library import wrap_triton

from vllm._custom_ops import create_fp4_output_tensors
from vllm.triton_utils import tl, triton

_BLOCK_M = 8
_BLOCK_K = 2048


@triton.jit
def _fp8_to_nvfp4_kernel(
    x_ptr,
    global_scale_inv_ptr,
    packed_ptr,
    scale_ptr,
    M,
    K,
    NUM_BLOCKS,
    NUM_BLOCK_ATOMS,
    stride_m,
    stride_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    NBLK: tl.constexpr = BLOCK_K // 16

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < K
    x = tl.load(
        x_ptr + rows[:, None] * stride_m + cols[None, :] * stride_k,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Per-16-element block scales: e4m3(amax / 6 * global_scale_inv).
    xr = tl.reshape(x, (BLOCK_M, NBLK, 16))
    amax = tl.max(tl.abs(xr), axis=2)
    global_scale_inv = tl.load(global_scale_inv_ptr)
    block_scale = ((amax / 6.0) * global_scale_inv).to(tl.float8e4nv)
    scale_deq = block_scale.to(tl.float32) / global_scale_inv
    scale_deq = tl.where(scale_deq == 0.0, 1.0, scale_deq)
    q = xr / scale_deq[:, :, None]

    # Hardware e2m1 conversion; two fp32 values pack into one byte.
    lo, hi = tl.split(tl.reshape(q, (BLOCK_M, NBLK, 8, 2)))
    byte16 = tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u16.u8 $0, t; }",
        "=h,f,f",
        [lo, hi],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    )
    byte = tl.reshape(byte16.to(tl.uint8), (BLOCK_M, BLOCK_K // 2))
    pcols = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(
        packed_ptr + rows[:, None] * (K // 2) + pcols[None, :],
        byte,
        mask=row_mask[:, None] & (pcols < (K // 2))[None, :],
    )

    # Block scales in the flashinfer-cutlass 128x4 swizzled layout.
    m = rows[:, None]
    nb = (pid_k * NBLK + tl.arange(0, NBLK))[None, :]
    lm = m % 128
    j = (lm % 32) * 4 + (lm // 32)
    off = (m // 128 * NUM_BLOCK_ATOMS + nb // 4) * 512 + j * 4 + (nb % 4)
    tl.store(scale_ptr + off, block_scale, mask=row_mask[:, None] & (nb < NUM_BLOCKS))


def fp8_to_nvfp4_supported_shape(m: int, k: int, weights_padding_bytes: int) -> bool:
    """Whether the direct fp8 -> nvfp4 path supports this activation shape.

    Restricted to the validated regime; other shapes fall back to the
    bf16 upcast + ``scaled_fp4_quant`` path.
    """
    return weights_padding_bytes == 0 and m % 128 == 0 and k % 64 == 0


def quantize_fp8_to_nvfp4(
    x: torch.Tensor, global_scale_inv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D fp8 activation to NVFP4 (flashinfer-cutlass layout).

    Args:
        x: [M, K] fp8 (e4m3) activation; K % 64 == 0, M % 128 == 0.
        global_scale_inv: scalar tensor, the layer's ``input_global_scale_inv``.

    Returns:
        (packed [M, K/2] uint8 e2m1, swizzled e4m3 block scales).
    """
    m, k = x.shape
    num_blocks = k // 16
    packed, scale = create_fp4_output_tensors(
        m, k, x.device, is_sf_swizzled_layout=True, padded_n=None
    )
    scale = scale.view(torch.float8_e4m3fn)
    grid = (triton.cdiv(m, _BLOCK_M), triton.cdiv(k, _BLOCK_K))
    wrap_triton(_fp8_to_nvfp4_kernel)[grid](
        x,
        global_scale_inv,
        packed,
        scale,
        m,
        k,
        num_blocks,
        num_blocks // 4,
        x.stride(0),
        x.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_K=_BLOCK_K,
        num_warps=8,
        num_stages=3,
    )
    return packed, scale
