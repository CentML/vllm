# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention output gate ``attn * sigmoid(gate)`` fused with its MXFP8 quant.

One kernel replaces the Inductor gate kernel and the ``mxfp8_quantize`` of
o_proj's input. The gated value is rounded to bf16 exactly as the Inductor
kernel stores it, then quantized with the FlashInfer-exact MXFP8 helpers of
``rms_norm_mxfp8_quant``, so the e4m3 values and F8_128x4-swizzled UE8M0
scales equal ``mxfp8_quantize(attn * sigmoid(gate), is_sf_swizzled_layout=True)``.
``gate`` may be a strided ``[tokens, heads, head_dim]`` view (the gate half of
the QKV projection read in place). Each program handles a ``[ROWS, BLOCK]``
tile; the math is element-wise, so every launch config gives the same bits.
"""

import math

import torch

from vllm.model_executor.layers.fusion.mxfp8_pdl import mxfp8_producer_early_trigger
from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
    MXFP8_BLOCK,
    mxfp8_quantize_rows,
    mxfp8_store_swizzled_scales_rows,
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
    gate_stride_t,
    gate_stride_h,
    HIDDEN: tl.constexpr,
    GATE_HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    STORE_8B: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    groups = chunk * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    valid = rows < num_tokens
    r2 = rows[:, None]
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        if EARLY_TRIGGER:
            tl.extra.cuda.gdc_launch_dependents()
    a = tl.load(attn_ptr + r2 * HIDDEN + cols[None, :], mask=valid[:, None], other=0.0)
    gate_offs = cols // GATE_HEAD_DIM * gate_stride_h + cols % GATE_HEAD_DIM
    g = tl.load(
        gate_ptr + r2 * gate_stride_t + gate_offs[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    # The Inductor gate kernel: fp32 sigmoid and product, stored as bf16.
    y = (a.to(tl.float32) * tl.sigmoid(g.to(tl.float32))).to(tl.bfloat16)
    quantized, sf = mxfp8_quantize_rows(y.to(tl.float32), ROWS, BLOCK)
    if STORE_8B:
        # Address the e4m3 row as 8-byte chunks: a 16 B/thread e4m3 store
        # would make Triton move the tile through shared memory.
        c = (
            chunk * BLOCK
            + tl.arange(0, BLOCK // 8)[None, :, None] * 8
            + tl.arange(0, 8)[None, None, :]
        )
        tl.store(
            q_ptr + rows[:, None, None] * HIDDEN + c,
            tl.reshape(quantized, (ROWS, BLOCK // 8, 8)),
            mask=valid[:, None, None],
        )
    else:
        tl.store(q_ptr + r2 * HIDDEN + cols[None, :], quantized, mask=valid[:, None])
    # Rows past num_tokens only zero-fill the 128-row scale padding.
    sf = tl.where(valid[:, None], sf, 0)
    mxfp8_store_swizzled_scales_rows(scale_ptr, rows, groups, sf, HIDDEN // 32)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _shapes(attn: torch.Tensor) -> tuple[int, int, int]:
    tokens = attn.shape[0]
    hidden = math.prod(attn.shape[1:])
    padded_tokens = triton.cdiv(tokens, 128) * 128
    return tokens, hidden, padded_tokens * triton.cdiv(hidden // MXFP8_BLOCK, 4) * 4


# (rows per program, columns per program, num_warps), measured on VR with
# 16 x 256 heads (cold-L2 CUDA-graph microbench): 4 x 256 tiles with 4 warps up
# to 256 tokens (1.2x at 8-64 tokens), 4 x 1024 with 8 warps below 1024 tokens
# (1.2x at 512) and 4 x 1024 with 4 warps above (1.1-1.16x at 1024-8192).
def _launch_config(tokens: int) -> tuple[int, int, int]:
    if tokens <= 256:
        return 4, 256, 4
    return 4, 1024, 8 if tokens < 1024 else 4


# One token count per launch config above (and per PDL setting), for warmup.
LAUNCH_CONFIG_TOKEN_COUNTS = (1, 257, 1024, 4096)


def attn_gate_mxfp8_quant(
    attn: torch.Tensor, gate: torch.Tensor, launch_pdl: bool | None = None
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
        rows, block, num_warps = _launch_config(tokens)
        # The largest power-of-two column block (>= 128) that divides hidden.
        block = min(block, hidden & -hidden)
        if launch_pdl is None:
            launch_pdl = tokens < 4096 and current_platform.is_arch_support_pdl()
        # The grid covers the 128-row scale padding; rows divides 128.
        grid = (triton.cdiv(tokens, 128) * 128 // rows, hidden // block)
        _attn_gate_mxfp8_kernel[grid](
            attn,
            gate,
            q,
            scales,
            tokens,
            gate.stride(0),
            gate_stride_h,
            HIDDEN=hidden,
            GATE_HEAD_DIM=gate_head_dim,
            BLOCK=block,
            ROWS=rows,
            STORE_8B=rows * block >= 16 * 32 * num_warps,
            LAUNCH_PDL=launch_pdl,
            EARLY_TRIGGER=launch_pdl and mxfp8_producer_early_trigger(),
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
