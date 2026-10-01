# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused (add +) RMSNorm producer that also emits the MXFP8 activation.

One kernel replaces the norm and the MXFP8 activation quant that follows it.
It writes the e4m3 values and F8_128x4-swizzled UE8M0 scales that FlashInfer's
``mxfp8_quantize(normed, is_sf_swizzled_layout=True)`` produces from the bf16
normalized output, so the MXFP8 GEMM consumes them unchanged. The norm follows
the native ``vllm.ir`` ``fused_add_rms_norm`` / ``rms_norm`` numerics.

Optionally it also writes the same scales in the linear ``[tokens, hidden/32]``
layout of ``mxfp8_quantize(normed, is_sf_swizzled_layout=False)`` (the TRT-LLM
MoE input), and gates the second summand like the Qwen shared expert:
``x + bf16(bf16(sigmoid(gate)) * x2)``.

``vllm::moe_finalize_add_rms_norm_mxfp8_quant`` takes ``x`` as a deferred
TRT-LLM MoE output instead (permuted GEMM2 rows, routing weights, permute
map) and forms each row with ``vllm::moe_finalize``'s arithmetic, so the routed
output is never written.
"""

import torch

from vllm.model_executor.layers.fusion.moe_finalize import (
    check_unfinalized,
    moe_finalize_row,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.torch_utils import direct_register_custom_op

MXFP8_BLOCK = 32


@triton.jit
def mxfp8_quantize_row(y, BLOCK: tl.constexpr):
    """FlashInfer-exact MXFP8 of one row of bf16-representable fp32 values.

    ``y`` has shape ``(BLOCK,)``, BLOCK a multiple of 32. Returns the fp32
    values to store through an e4m3 pointer (the store rounds them exactly as
    FlashInfer does) and the uint32 UE8M0 exponent of each 32-element group.
    """
    grouped = tl.reshape(y, (BLOCK // 32, 32))
    amax = tl.max(tl.abs(grouped), 1)
    normalized = amax * (1.0 / 448.0)
    bits = normalized.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.minimum(exponent + bump, 254)
    sf = tl.where(normalized <= 0, 0, sf)
    # Match FlashInfer's UE8M0 conversion, including zero/subnormal scales.
    inv_bits = tl.where(sf == 0, 0, (254 - sf) << 23)
    inv_scale = inv_bits.to(tl.float32, bitcast=True)
    quantized = tl.reshape(grouped * inv_scale[:, None], (BLOCK,))
    return quantized, sf


@triton.jit
def mxfp8_store_swizzled_scales(scale_ptr, row, groups, sf, NUM_GROUPS: tl.constexpr):
    """Store UE8M0 scales of ``row`` for group indices ``groups`` (F8_128x4).

    Groups in ``[NUM_GROUPS, PADDED_GROUPS)`` are zero-filled; rows past the
    token count should pass ``sf = 0`` to fill the 128-row padding.
    """
    PADDED_GROUPS: tl.constexpr = (NUM_GROUPS + 3) // 4 * 4
    sf = tl.where(groups < NUM_GROUPS, sf, 0)
    # F8_128x4: [row/128, group/4, row%32, row%128/32, group%4].
    offsets = (
        row // 128 * (128 * PADDED_GROUPS)
        + groups // 4 * 512
        + row % 32 * 16
        + row % 128 // 32 * 4
        + groups % 4
    )
    tl.store(scale_ptr + offsets, sf.to(tl.uint8), mask=groups < PADDED_GROUPS)


@triton.jit(do_not_specialize=["num_tokens"])
def _add_rms_norm_mxfp8_kernel(
    x_ptr,
    x2_ptr,
    residual_ptr,
    weight_ptr,
    residual_out_ptr,
    normed_ptr,
    q_ptr,
    scale_ptr,
    gate_ptr,
    linear_scale_ptr,
    moe_idx_ptr,
    moe_weights_ptr,
    num_tokens,
    x_stride,
    x2_stride,
    residual_stride,
    gate_stride,
    eps,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_X2: tl.constexpr,
    HAS_GATE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    WEIGHT_OFFSET: tl.constexpr,
    STORE_NORMED: tl.constexpr,
    STORE_LINEAR_SF: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
    MOE_TOP_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    groups = tl.arange(0, BLOCK // 32)
    NUM_GROUPS: tl.constexpr = HIDDEN // 32
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        # The consumer (MXFP8 GEMM) waits on this grid before reading, so let
        # it launch and run its prologue while this one works.
        tl.extra.cuda.gdc_launch_dependents()
    if row < num_tokens:
        if MOE_TOP_K > 0:
            # x_ptr holds permuted GEMM2 rows: reduce them as moe_finalize
            # does, including its bf16 rounding.
            x = moe_finalize_row(
                x_ptr,
                moe_weights_ptr,
                moe_idx_ptr,
                row,
                cols,
                mask,
                x_stride,
                MOE_TOP_K,
            )
            x = x.to(x_ptr.dtype.element_ty).to(tl.float32)
        else:
            x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0)
            x = x.to(tl.float32)
        if HAS_X2:
            # fp32 sum without a bf16 rounding, as Inductor fuses the combine.
            x2 = tl.load(x2_ptr + row * x2_stride + cols, mask=mask, other=0.0)
            if HAS_GATE:
                # Shared-expert gate, rounded like the eager bf16 sigmoid and
                # bf16 product it replaces.
                dtype = x2_ptr.dtype.element_ty
                g = tl.load(gate_ptr + row * gate_stride).to(tl.float32)
                g = tl.sigmoid(g).to(dtype).to(tl.float32)
                x2 = (g * x2.to(tl.float32)).to(dtype)
            x = x + x2.to(tl.float32)
        if HAS_RESIDUAL:
            res = tl.load(
                residual_ptr + row * residual_stride + cols, mask=mask, other=0.0
            )
            x = x + res.to(tl.float32)
            tl.store(
                residual_out_ptr + row * HIDDEN + cols,
                x.to(residual_out_ptr.dtype.element_ty),
                mask=mask,
            )
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = w + WEIGHT_OFFSET
        variance = tl.sum(x * x, 0) / HIDDEN
        y = x * tldevice.rsqrt(variance + eps)
        y = (y * w).to(normed_ptr.dtype.element_ty)
        if STORE_NORMED:
            tl.store(normed_ptr + row * HIDDEN + cols, y, mask=mask)
        # Quantize the materialized bf16 output, as the standalone quant does.
        quantized, sf = mxfp8_quantize_row(y.to(tl.float32), BLOCK)
        tl.store(q_ptr + row * HIDDEN + cols, quantized, mask=mask)
        if STORE_LINEAR_SF:
            tl.store(
                linear_scale_ptr + row * NUM_GROUPS + groups,
                sf.to(tl.uint8),
                mask=groups < NUM_GROUPS,
            )
    else:
        # Rows past num_tokens only zero-fill the 128-row scale padding.
        sf = tl.full((BLOCK // 32,), 0, tl.uint32)
    mxfp8_store_swizzled_scales(scale_ptr, row, groups, sf, NUM_GROUPS)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _output_shapes(
    tokens: int,
    hidden: int,
    has_residual: bool,
    store_normed: bool,
    store_linear_scales: bool = False,
) -> tuple[tuple[int, int], tuple[int, int], int, tuple[int, int]]:
    padded_tokens = triton.cdiv(tokens, 128) * 128
    padded_groups = triton.cdiv(hidden // MXFP8_BLOCK, 4) * 4
    return (
        (tokens if store_normed else 0, hidden),
        (tokens if has_residual else 0, hidden),
        padded_tokens * padded_groups,
        (tokens if store_linear_scales else 0, hidden // MXFP8_BLOCK),
    )


# Measured on VR (hidden 2048): 8 warps per row win below 1024 tokens, fewer
# warps above as the kernel turns bandwidth bound. At 4096+ tokens a PDL launch
# gains nothing and its waiting CTAs can steal SMs from the predecessor.
def _launch_config(tokens: int) -> tuple[int, bool]:
    num_warps = 8 if tokens < 1024 else 4 if tokens < 2048 else 2
    return num_warps, tokens < 4096 and current_platform.is_arch_support_pdl()


def add_rms_norm_mxfp8_quant(
    x: torch.Tensor,
    x2: torch.Tensor | None,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    weight_offset: float,
    store_normed: bool,
    x2_gate: torch.Tensor | None = None,
    store_linear_scales: bool = False,
    num_warps: int | None = None,
    launch_pdl: bool | None = None,
    moe_route: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm of ``x (+ x2) (+ residual)`` plus its MXFP8 activation.

    ``weight + weight_offset`` is the fp32 norm weight (1.0 for GemmaRMSNorm).
    With ``x2_gate`` (``[tokens, 1]`` logits) the second summand is
    ``bf16(bf16(sigmoid(x2_gate)) * x2)``.
    With ``moe_route = (expert_weights, permuted_idx)``, ``x`` holds the
    permuted GEMM2 rows of a deferred MoE and each row of the first summand is
    ``moe_finalize(x, expert_weights, permuted_idx)``.
    Returns ``(normed, residual_out, e4m3, scales, linear_scales)``. ``normed``
    has zero rows unless ``store_normed``; ``residual_out`` has zero rows without
    a residual. ``scales`` is FlashInfer's flat F8_128x4 swizzled UE8M0 buffer,
    with the padding rows zero-filled. ``linear_scales`` holds the same scales
    as ``[tokens, hidden / 32]`` when ``store_linear_scales``, else zero rows.
    """
    assert x.ndim == 2 and x.stride(-1) == 1
    if moe_route is None:
        tokens, hidden = x.shape
        moe_weights, moe_idx, top_k = x, x, 0
    else:
        moe_weights, moe_idx = moe_route
        tokens, hidden, top_k = check_unfinalized(x, moe_weights, moe_idx)
    shape = (tokens, hidden)
    assert hidden % MXFP8_BLOCK == 0
    assert weight.shape == (hidden,) and weight.is_contiguous()
    if x2 is not None:
        assert x2.shape == shape and x2.dtype == x.dtype and x2.stride(-1) == 1
    if x2_gate is not None:
        assert x2 is not None and x2_gate.shape == (tokens, 1)
    if residual is not None:
        assert residual.shape == shape and residual.dtype == x.dtype
        assert residual.stride(-1) == 1
    normed_shape, residual_shape, scale_numel, linear_shape = _output_shapes(
        tokens, hidden, residual is not None, store_normed, store_linear_scales
    )
    normed = torch.empty(normed_shape, dtype=x.dtype, device=x.device)
    residual_out = torch.empty(residual_shape, dtype=x.dtype, device=x.device)
    q = torch.empty(shape, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(scale_numel, dtype=torch.uint8, device=x.device)
    linear_scales = torch.empty(linear_shape, dtype=torch.uint8, device=x.device)
    if tokens:
        default_warps, default_pdl = _launch_config(tokens)
        num_warps = default_warps if num_warps is None else num_warps
        launch_pdl = default_pdl if launch_pdl is None else launch_pdl
        _add_rms_norm_mxfp8_kernel[(triton.cdiv(tokens, 128) * 128,)](
            x,
            x if x2 is None else x2,
            x if residual is None else residual,
            weight,
            residual_out,
            normed,
            q,
            scales,
            x if x2_gate is None else x2_gate,
            linear_scales,
            moe_idx,
            moe_weights,
            tokens,
            x.stride(0),
            x.stride(0) if x2 is None else x2.stride(0),
            x.stride(0) if residual is None else residual.stride(0),
            0 if x2_gate is None else x2_gate.stride(0),
            epsilon,
            HIDDEN=hidden,
            BLOCK=triton.next_power_of_2(hidden),
            HAS_X2=x2 is not None,
            HAS_GATE=x2_gate is not None,
            HAS_RESIDUAL=residual is not None,
            WEIGHT_OFFSET=weight_offset,
            STORE_NORMED=store_normed,
            STORE_LINEAR_SF=store_linear_scales,
            LAUNCH_PDL=launch_pdl,
            MOE_TOP_K=top_k,
            launch_pdl=launch_pdl,
            num_warps=num_warps,
        )
    return normed, residual_out, q, scales, linear_scales


def _add_rms_norm_mxfp8_quant_impl(
    x: torch.Tensor,
    x2: torch.Tensor | None,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    weight_offset: float,
    store_normed: bool,
    x2_gate: torch.Tensor | None = None,
    store_linear_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return add_rms_norm_mxfp8_quant(
        x,
        x2,
        residual,
        weight,
        epsilon,
        weight_offset,
        store_normed,
        x2_gate,
        store_linear_scales,
    )


def _add_rms_norm_mxfp8_quant_fake(
    x: torch.Tensor,
    x2: torch.Tensor | None,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    weight_offset: float,
    store_normed: bool,
    x2_gate: torch.Tensor | None = None,
    store_linear_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, hidden = x.shape
    normed_shape, residual_shape, scale_numel, linear_shape = _output_shapes(
        tokens, hidden, residual is not None, store_normed, store_linear_scales
    )
    return (
        x.new_empty(normed_shape),
        x.new_empty(residual_shape),
        x.new_empty(x.shape, dtype=torch.float8_e4m3fn),
        x.new_empty(scale_numel, dtype=torch.uint8),
        x.new_empty(linear_shape, dtype=torch.uint8),
    )


direct_register_custom_op(
    op_name="add_rms_norm_mxfp8_quant",
    op_func=_add_rms_norm_mxfp8_quant_impl,
    fake_impl=_add_rms_norm_mxfp8_quant_fake,
)


def _moe_finalize_add_rms_norm_mxfp8_quant_impl(
    permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    permuted_idx: torch.Tensor,
    x2: torch.Tensor | None,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    weight_offset: float,
    store_normed: bool,
    x2_gate: torch.Tensor | None = None,
    store_linear_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return add_rms_norm_mxfp8_quant(
        permuted,
        x2,
        residual,
        weight,
        epsilon,
        weight_offset,
        store_normed,
        x2_gate,
        store_linear_scales,
        moe_route=(expert_weights, permuted_idx),
    )


def _moe_finalize_add_rms_norm_mxfp8_quant_fake(
    permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    permuted_idx: torch.Tensor,
    x2: torch.Tensor | None,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    weight_offset: float,
    store_normed: bool,
    x2_gate: torch.Tensor | None = None,
    store_linear_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = permuted.new_empty((permuted_idx.shape[0], permuted.shape[1]))
    return _add_rms_norm_mxfp8_quant_fake(
        x,
        x2,
        residual,
        weight,
        epsilon,
        weight_offset,
        store_normed,
        x2_gate,
        store_linear_scales,
    )


# add_rms_norm_mxfp8_quant whose first summand is a deferred TRT-LLM MoE
# output, reduced in-kernel (MLPerf submission path; specialized for
# Qwen3.6-35B-A3B). Produced by RMSNormMxfp8QuantFusionPass from
# vllm::moe_finalize feeding the norm.
direct_register_custom_op(
    op_name="moe_finalize_add_rms_norm_mxfp8_quant",
    op_func=_moe_finalize_add_rms_norm_mxfp8_quant_impl,
    fake_impl=_moe_finalize_add_rms_norm_mxfp8_quant_fake,
)
