# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SiLU-and-mul producer that emits the MXFP8 input of the following linear.

One kernel replaces the SiluAndMul kernel and FlashInfer's
``mxfp8_quantize(act, is_sf_swizzled_layout=True)``: it computes
``silu(x[:, :d]) * x[:, d:]`` with the arithmetic of the Inductor-compiled
native SiluAndMul, rounds to bf16 and quantizes that bf16 row to e4m3 with
F8_128x4-swizzled UE8M0 scales. Each program handles ``ROWS`` token rows; the
math is element-wise (the group max is order-free), so every launch config
gives the same bits.
"""

import torch

from vllm.model_executor.layers.fusion.mxfp8_pdl import mxfp8_producer_early_trigger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
    MXFP8_BLOCK,
    mxfp8_quantize_rows,
    mxfp8_store_swizzled_scales_rows,
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
    ROWS: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    groups = tl.arange(0, BLOCK // 32)
    valid = rows < num_tokens
    mask = valid[:, None] & (cols[None, :] < D)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        if EARLY_TRIGGER:
            tl.extra.cuda.gdc_launch_dependents()
    a_ptrs = x_ptr + rows[:, None] * x_stride + cols[None, :]
    a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(a_ptrs + D, mask=mask, other=0.0)
    # Inductor's lowering of F.silu(a) * b: a / (exp(-a) + 1) * b.
    y = a / (tldevice.exp(-a) + 1.0) * b.to(tl.float32)
    y = y.to(tl.bfloat16)
    quantized, sf = mxfp8_quantize_rows(y.to(tl.float32), ROWS, BLOCK)
    tl.store(q_ptr + rows[:, None] * D + cols[None, :], quantized, mask=mask)
    # Rows past num_tokens only zero-fill the 128-row scale padding.
    sf = tl.where(valid[:, None], sf, 0)
    mxfp8_store_swizzled_scales_rows(scale_ptr, rows, groups, sf, D // 32)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _output_shapes(x: torch.Tensor) -> tuple[tuple[int, int], int]:
    tokens, two_d = x.shape
    d = two_d // 2
    padded_tokens = triton.cdiv(tokens, 128) * 128
    padded_groups = triton.cdiv(d // MXFP8_BLOCK, 4) * 4
    return (tokens, d), padded_tokens * padded_groups


# (rows per program, num_warps), measured on VR at d = 512 (cold-L2 CUDA-graph
# microbench): one 4-warp row per program below 128 tokens; 4 rows with 8 warps
# up to 2048 tokens (1.36x at 2048) and 4 rows with 4 warps above (1.6-2.0x at
# 3840-8192 tokens), where 1-row programs are too small to keep DRAM busy.
def _launch_config(tokens: int) -> tuple[int, int]:
    if tokens < 128:
        return 1, 4
    return 4, 8 if tokens <= 2048 else 4


# One token count per launch config above, for kernel warmup.
LAUNCH_CONFIG_TOKEN_COUNTS = (1, 128, 2049)


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
        rows, num_warps = _launch_config(tokens)
        # The grid covers the 128-row scale padding; rows divides 128.
        _silu_mul_mxfp8_kernel[(triton.cdiv(tokens, 128) * 128 // rows,)](
            x,
            q,
            scales,
            tokens,
            x.stride(0),
            D=d,
            BLOCK=triton.next_power_of_2(d),
            ROWS=rows,
            LAUNCH_PDL=launch_pdl,
            EARLY_TRIGGER=launch_pdl and mxfp8_producer_early_trigger(),
            launch_pdl=launch_pdl,
            num_warps=num_warps,
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
