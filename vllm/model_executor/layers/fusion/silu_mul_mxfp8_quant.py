# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SiLU-and-mul producer that emits the MXFP8 input of the following linear.

One kernel replaces the SiluAndMul kernel and FlashInfer's
``mxfp8_quantize(act, is_sf_swizzled_layout=True)``: it computes
``silu(x[:, :d]) * x[:, d:]`` with the arithmetic of the Inductor-compiled
native SiluAndMul, rounds to bf16 and quantizes that bf16 row to e4m3 with
F8_128x4-swizzled UE8M0 scales.
"""

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
    MXFP8_BLOCK,
    mxfp8_quantize_row,
    mxfp8_store_swizzled_scales,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit(do_not_specialize=["num_tokens"])
def _silu_mul_mxfp8_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    num_tokens,
    x_stride,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    groups = tl.arange(0, BLOCK // 32)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
    if row < num_tokens:
        a = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0)
        b = tl.load(x_ptr + row * x_stride + D + cols, mask=mask, other=0.0)
        a = a.to(tl.float32)
        # Inductor's lowering of F.silu(a) * b: a / (exp(-a) + 1) * b.
        y = a / (tldevice.exp(-a) + 1.0) * b.to(tl.float32)
        y = y.to(tl.bfloat16)
        quantized, sf = mxfp8_quantize_row(y.to(tl.float32), BLOCK)
        tl.store(q_ptr + row * D + cols, quantized, mask=mask)
    else:
        sf = tl.full((BLOCK // 32,), 0, tl.uint32)
    mxfp8_store_swizzled_scales(scale_ptr, row, groups, sf, D // 32)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _output_shapes(x: torch.Tensor) -> tuple[tuple[int, int], int]:
    tokens, two_d = x.shape
    d = two_d // 2
    padded_tokens = triton.cdiv(tokens, 128) * 128
    padded_groups = triton.cdiv(d // MXFP8_BLOCK, 4) * 4
    return (tokens, d), padded_tokens * padded_groups


def silu_mul_mxfp8_quant(
    x: torch.Tensor, launch_pdl: bool | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``silu(x[:, :d]) * x[:, d:]`` as MXFP8: ``(e4m3 [T, d], swizzled scales)``.

    ``scales`` is FlashInfer's flat F8_128x4 UE8M0 buffer with the padding rows
    zero-filled, as ``mxfp8_quantize(..., is_sf_swizzled_layout=True)``.
    """
    assert x.ndim == 2 and x.stride(-1) == 1 and x.dtype == torch.bfloat16
    tokens = x.shape[0]
    q_shape, scale_numel = _output_shapes(x)
    d = q_shape[1]
    assert d % MXFP8_BLOCK == 0
    q = torch.empty(q_shape, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(scale_numel, dtype=torch.uint8, device=x.device)
    if tokens:
        if launch_pdl is None:
            launch_pdl = current_platform.is_arch_support_pdl()
        _silu_mul_mxfp8_kernel[(triton.cdiv(tokens, 128) * 128,)](
            x,
            q,
            scales,
            tokens,
            x.stride(0),
            D=d,
            BLOCK=triton.next_power_of_2(d),
            LAUNCH_PDL=launch_pdl,
            launch_pdl=launch_pdl,
            num_warps=4,
        )
    return q, scales


def _silu_mul_mxfp8_quant_impl(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return silu_mul_mxfp8_quant(x)


def _silu_mul_mxfp8_quant_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_shape, scale_numel = _output_shapes(x)
    return (
        x.new_empty(q_shape, dtype=torch.float8_e4m3fn),
        x.new_empty(scale_numel, dtype=torch.uint8),
    )


# Opaque to torch.compile, like vllm::mxfp8_quantize it replaces: Inductor must
# not trace the Triton kernel (it cannot resolve the libdevice exp it uses).
direct_register_custom_op(
    op_name="silu_mul_mxfp8_quant",
    op_func=_silu_mul_mxfp8_quant_impl,
    fake_impl=_silu_mul_mxfp8_quant_fake,
)


def silu_and_mul_mxfp8_dynamic(x: torch.Tensor, linear) -> QuantizedActivation:
    """Producer for ``(SiluAndMul, kMxfp8Dynamic)`` in the fused-act registry."""
    del linear
    q, scales = torch.ops.vllm.silu_mul_mxfp8_quant(x.view(-1, x.shape[-1]))
    out_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    return QuantizedActivation(
        data=q,
        scale=scales,
        orig_dtype=x.dtype,
        orig_shape=out_shape,
        quant_key=kMxfp8Dynamic,
    )
