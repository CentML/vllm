# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention output gate ``attn * sigmoid(gate)`` fused with its MXFP8 quant.

One kernel replaces the Inductor gate kernel and the ``mxfp8_quantize`` of
o_proj's input. The gated value is rounded to bf16 exactly as the Inductor
kernel stores it, then quantized with the FlashInfer-exact MXFP8 helpers of
``rms_norm_mxfp8_quant``, so the e4m3 values and F8_128x4-swizzled UE8M0
scales equal ``mxfp8_quantize(attn * sigmoid(gate), is_sf_swizzled_layout=True)``.
``gate`` may be a strided ``[tokens, heads, head_dim]`` view (the gate half of
the QKV projection read in place).
"""

import math

import torch

from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
    MXFP8_BLOCK,
    mxfp8_quantize_row,
    mxfp8_store_swizzled_scales,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit(do_not_specialize=["num_tokens"])
def _attn_gate_mxfp8_kernel(
    attn_ptr,
    gate_ptr,
    q_ptr,
    scale_ptr,
    num_tokens,
    attn_stride_t,
    gate_stride_t,
    gate_stride_h,
    HIDDEN: tl.constexpr,
    GATE_HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    groups = chunk * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
    if row < num_tokens:
        a = tl.load(attn_ptr + row * attn_stride_t + cols).to(tl.float32)
        gate_offs = cols // GATE_HEAD_DIM * gate_stride_h + cols % GATE_HEAD_DIM
        g = tl.load(gate_ptr + row * gate_stride_t + gate_offs).to(tl.float32)
        # The Inductor gate kernel: fp32 sigmoid and product, stored as bf16.
        y = (a * tl.sigmoid(g)).to(tl.bfloat16)
        quantized, sf = mxfp8_quantize_row(y.to(tl.float32), BLOCK)
        tl.store(q_ptr + row * HIDDEN + cols, quantized)
    else:
        # Rows past num_tokens only zero-fill the 128-row scale padding.
        sf = tl.full((BLOCK // 32,), 0, tl.uint32)
    mxfp8_store_swizzled_scales(scale_ptr, row, groups, sf, HIDDEN // 32)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _shapes(attn: torch.Tensor) -> tuple[int, int, int]:
    tokens = attn.shape[0]
    hidden = math.prod(attn.shape[1:])
    padded_tokens = triton.cdiv(tokens, 128) * 128
    return tokens, hidden, padded_tokens * triton.cdiv(hidden // MXFP8_BLOCK, 4) * 4


def _launch_config(tokens: int, hidden: int) -> tuple[int, int, bool]:
    """(BLOCK, num_warps, launch_pdl)."""
    block = min(hidden, 1024) if tokens < 1024 else hidden
    num_warps = 4 if block <= 1024 else 8
    return block, num_warps, tokens < 4096 and current_platform.is_arch_support_pdl()


def attn_gate_mxfp8_quant(
    attn: torch.Tensor,
    gate: torch.Tensor,
    block: int | None = None,
    num_warps: int | None = None,
    launch_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 of ``attn * sigmoid(gate)`` for a swizzled-scale MXFP8 GEMM.

    ``attn`` is contiguous bf16 ``[tokens, hidden]`` or ``[tokens, heads,
    head_dim]``; ``gate`` has the same shape and a unit last-dim stride.
    Returns ``(e4m3 [tokens, hidden], flat F8_128x4 UE8M0 scales)`` like
    ``mxfp8_quantize(x.view(-1, hidden), is_sf_swizzled_layout=True)``.
    """
    assert attn.dtype == torch.bfloat16 and gate.dtype == torch.bfloat16
    assert attn.is_contiguous() and gate.shape == attn.shape
    assert gate.stride(-1) == 1 and attn.dim() in (2, 3)
    tokens, hidden, scale_numel = _shapes(attn)
    assert hidden % (4 * MXFP8_BLOCK) == 0
    if attn.dim() == 3:
        gate_head_dim, gate_stride_h = gate.shape[2], gate.stride(1)
    else:
        gate_head_dim, gate_stride_h = hidden, 0
    q = torch.empty((tokens, hidden), dtype=torch.float8_e4m3fn, device=attn.device)
    scales = torch.empty(scale_numel, dtype=torch.uint8, device=attn.device)
    if tokens:
        d_block, d_warps, d_pdl = _launch_config(tokens, hidden)
        block = d_block if block is None else block
        num_warps = d_warps if num_warps is None else num_warps
        launch_pdl = d_pdl if launch_pdl is None else launch_pdl
        assert hidden % block == 0 and block % (4 * MXFP8_BLOCK) == 0
        _attn_gate_mxfp8_kernel[(triton.cdiv(tokens, 128) * 128, hidden // block)](
            attn,
            gate,
            q,
            scales,
            tokens,
            hidden,
            gate.stride(0),
            gate_stride_h,
            HIDDEN=hidden,
            GATE_HEAD_DIM=gate_head_dim,
            BLOCK=block,
            LAUNCH_PDL=launch_pdl,
            launch_pdl=launch_pdl,
            num_warps=num_warps,
        )
    return q, scales


def _attn_gate_mxfp8_quant_impl(
    attn: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return attn_gate_mxfp8_quant(attn, gate)


def _attn_gate_mxfp8_quant_fake(
    attn: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, hidden, scale_numel = _shapes(attn)
    return (
        attn.new_empty((tokens, hidden), dtype=torch.float8_e4m3fn),
        attn.new_empty(scale_numel, dtype=torch.uint8),
    )


direct_register_custom_op(
    op_name="attn_gate_mxfp8_quant",
    op_func=_attn_gate_mxfp8_quant_impl,
    fake_impl=_attn_gate_mxfp8_quant_fake,
)
