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

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.moe_finalize import (
    check_unfinalized,
    check_unfinalized2,
    fin2_one_load,
    log_first_call,
    moe_finalize2,
    moe_finalize_row,
    moe_finalize_row2,
)
from vllm.model_executor.layers.fusion.mxfp8_pdl import mxfp8_producer_early_trigger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

MXFP8_BLOCK = 32
# VLLM_MXFP8_NORM_LARGE_M=1: the deferred-MoE form (8 gathered GEMM2 rows per
# token) takes 4 warps per row instead of 2 from VLLM_MXFP8_NORM_LARGE_M_TOKENS
# tokens (meant for large token counts; the threshold depends on the GPU, no
# default: unset, every token count). A different warp count reduces the row's
# sum of squares in another order, so a few output bytes can differ (bf16/e4m3
# rounding-boundary flips): not bitwise.
NORM_LARGE_M = os.environ.get("VLLM_MXFP8_NORM_LARGE_M", "0") == "1"
NORM_LARGE_M_TOKENS = int(os.environ.get("VLLM_MXFP8_NORM_LARGE_M_TOKENS") or 0)
if NORM_LARGE_M and not os.environ.get("VLLM_MXFP8_NORM_LARGE_M_TOKENS"):
    logger.info(
        "VLLM_MXFP8_NORM_LARGE_M=1 without VLLM_MXFP8_NORM_LARGE_M_TOKENS: the "
        "large-M warp count applies at every token count"
    )


@triton.jit
def _ue8m0_scale(amax):
    """FlashInfer's UE8M0 exponent of each 32-group's ``amax`` (uint32) and the
    fp32 reciprocal scale, including zero/subnormal scales.
    """
    normalized = amax * (1.0 / 448.0)
    bits = normalized.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.minimum(exponent + bump, 254)
    sf = tl.where(normalized <= 0, 0, sf)
    inv_bits = tl.where(sf == 0, 0, (254 - sf) << 23)
    return sf, inv_bits.to(tl.float32, bitcast=True)


@triton.jit
def _swizzled_scale_offsets(row, groups, PADDED_GROUPS: tl.constexpr):
    # F8_128x4: [row/128, group/4, row%32, row%128/32, group%4].
    return (
        row // 128 * (128 * PADDED_GROUPS)
        + groups // 4 * 512
        + row % 32 * 16
        + row % 128 // 32 * 4
        + groups % 4
    )


@triton.jit
def mxfp8_quantize_row(y, BLOCK: tl.constexpr):
    """FlashInfer-exact MXFP8 of one row of bf16-representable fp32 values.

    ``y`` has shape ``(BLOCK,)``, BLOCK a multiple of 32. Returns the fp32
    values to store through an e4m3 pointer (the store rounds them exactly as
    FlashInfer does) and the uint32 UE8M0 exponent of each 32-element group.
    """
    grouped = tl.reshape(y, (BLOCK // 32, 32))
    sf, inv_scale = _ue8m0_scale(tl.max(tl.abs(grouped), 1))
    quantized = tl.reshape(grouped * inv_scale[:, None], (BLOCK,))
    return quantized, sf


@triton.jit
def mxfp8_quantize_rows(y, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    """``mxfp8_quantize_row`` of each row of ``y`` ``(ROWS, BLOCK)``; ``sf`` is
    ``(ROWS, BLOCK // 32)``.
    """
    grouped = tl.reshape(y, (ROWS, BLOCK // 32, 32))
    sf, inv_scale = _ue8m0_scale(tl.max(tl.abs(grouped), 2))
    quantized = tl.reshape(grouped * inv_scale[:, :, None], (ROWS, BLOCK))
    return quantized, sf


@triton.jit
def mxfp8_store_swizzled_scales(scale_ptr, row, groups, sf, NUM_GROUPS: tl.constexpr):
    """Store UE8M0 scales of ``row`` for group indices ``groups`` (F8_128x4).

    Groups in ``[NUM_GROUPS, PADDED_GROUPS)`` are zero-filled; rows past the
    token count should pass ``sf = 0`` to fill the 128-row padding.
    """
    PADDED_GROUPS: tl.constexpr = (NUM_GROUPS + 3) // 4 * 4
    sf = tl.where(groups < NUM_GROUPS, sf, 0)
    offsets = _swizzled_scale_offsets(row, groups, PADDED_GROUPS)
    tl.store(scale_ptr + offsets, sf.to(tl.uint8), mask=groups < PADDED_GROUPS)


@triton.jit
def mxfp8_store_swizzled_scales_rows(
    scale_ptr, rows, groups, sf, NUM_GROUPS: tl.constexpr
):
    """``mxfp8_store_swizzled_scales`` of ``sf`` ``(ROWS, G)`` for ``rows``
    ``(ROWS,)`` and ``groups`` ``(G,)``.
    """
    PADDED_GROUPS: tl.constexpr = (NUM_GROUPS + 3) // 4 * 4
    g = groups[None, :]
    sf = tl.where(g < NUM_GROUPS, sf, 0)
    offsets = _swizzled_scale_offsets(rows[:, None], g, PADDED_GROUPS)
    tl.store(scale_ptr + offsets, sf.to(tl.uint8), mask=g < PADDED_GROUPS)


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
    EARLY_TRIGGER: tl.constexpr,
    MOE_TOP_K: tl.constexpr,
    PREFETCH: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    groups = tl.arange(0, BLOCK // 32)
    NUM_GROUPS: tl.constexpr = HIDDEN // 32
    if PREFETCH:
        # Inputs complete before this kernel can launch (written ahead of the
        # PDL producer chain, or joined by a full edge) and the norm weight:
        # read before the wait. The permuted rows and routing come after it.
        pmask = mask & (row < num_tokens)
        w_pre = tl.load(weight_ptr + cols, mask=pmask, other=0.0)
        if HAS_X2:
            x2_pre = tl.load(x2_ptr + row * x2_stride + cols, mask=pmask, other=0.0)
            if HAS_GATE:
                g_pre = tl.load(gate_ptr + row * gate_stride, mask=row < num_tokens)
        if HAS_RESIDUAL:
            res_pre = tl.load(
                residual_ptr + row * residual_stride + cols, mask=pmask, other=0.0
            )
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        if EARLY_TRIGGER:
            # The consumer (MXFP8 GEMM) waits on this grid before reading, so
            # let it launch and run its prologue while this one works.
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
            if PREFETCH:
                x2 = x2_pre
            else:
                x2 = tl.load(x2_ptr + row * x2_stride + cols, mask=mask, other=0.0)
            if HAS_GATE:
                # Shared-expert gate, rounded like the eager bf16 sigmoid and
                # bf16 product it replaces.
                dtype = x2_ptr.dtype.element_ty
                if PREFETCH:
                    g = g_pre.to(tl.float32)
                else:
                    g = tl.load(gate_ptr + row * gate_stride).to(tl.float32)
                g = tl.sigmoid(g).to(dtype).to(tl.float32)
                x2 = (g * x2.to(tl.float32)).to(dtype)
            x = x + x2.to(tl.float32)
        if HAS_RESIDUAL:
            if PREFETCH:
                res = res_pre
            else:
                res = tl.load(
                    residual_ptr + row * residual_stride + cols, mask=mask, other=0.0
                )
            x = x + res.to(tl.float32)
            tl.store(
                residual_out_ptr + row * HIDDEN + cols,
                x.to(residual_out_ptr.dtype.element_ty),
                mask=mask,
            )
        if PREFETCH:
            w = w_pre.to(tl.float32)
        else:
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
        if NORM_LARGE_M and top_k > 0 and tokens >= NORM_LARGE_M_TOKENS:
            default_warps = 4
        num_warps = default_warps if num_warps is None else num_warps
        launch_pdl = default_pdl if launch_pdl is None else launch_pdl
        prefetch = top_k > 0 and launch_pdl and _finalize_prefetch()
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
            EARLY_TRIGGER=launch_pdl and mxfp8_producer_early_trigger(),
            MOE_TOP_K=top_k,
            PREFETCH=prefetch,
            launch_pdl=launch_pdl,
            num_warps=num_warps,
        )
    return normed, residual_out, q, scales, linear_scales


def _finalize_prefetch() -> bool:
    """VLLM_H82_FINALIZE_PREFETCH=1 (opt-in): the MoE finalize
    norm reads the residual, the shared-expert output and gate, and the norm
    weight before its PDL wait. Every one of them is complete before the
    kernel can launch: the shared branch joins by a full edge, and the residual
    is written ahead of the routing kernel, which waits on its producer before
    it triggers. Same loads and arithmetic, issued earlier.
    """
    if os.environ.get("VLLM_H82_FINALIZE_PREFETCH", "0") != "1":
        return False
    logger.info_once(
        "H82: MoE finalize norm prefetches residual, shared output, gate and "
        "norm weight before its PDL wait (VLLM_H82_FINALIZE_PREFETCH)"
    )
    return True


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
# output, reduced in-kernel (opt-in; specialized for
# Qwen3.6-35B-A3B). Produced by RMSNormMxfp8QuantFusionPass from
# vllm::moe_finalize feeding the norm.
direct_register_custom_op(
    op_name="moe_finalize_add_rms_norm_mxfp8_quant",
    op_func=_moe_finalize_add_rms_norm_mxfp8_quant_impl,
    fake_impl=_moe_finalize_add_rms_norm_mxfp8_quant_fake,
)


# Two-buffer form of the deferred-MoE consumer (die-split MoE, moe_die_split,
# VLLM_MOE_DIE_SPLIT=1): the first summand is vllm::moe_finalize2 of two GEMM2
# outputs. The kernel is _add_rms_norm_mxfp8_kernel's MoE form with the row
# reduced by moe_finalize_row2 (the same k-order fused multiply-adds); the
# one-buffer kernel is left as it is.
@triton.jit(do_not_specialize=["num_tokens"])
def _moe2_add_rms_norm_mxfp8_kernel(
    x_ptr,
    x1_ptr,
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
    moe_idx1_ptr,
    moe_weights_ptr,
    num_tokens,
    x_stride,
    x1_stride,
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
    EARLY_TRIGGER: tl.constexpr,
    MOE_TOP_K: tl.constexpr,
    ONE_LOAD: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    groups = tl.arange(0, BLOCK // 32)
    NUM_GROUPS: tl.constexpr = HIDDEN // 32
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        if EARLY_TRIGGER:
            tl.extra.cuda.gdc_launch_dependents()
    if row < num_tokens:
        x = moe_finalize_row2(
            x_ptr,
            x1_ptr,
            moe_weights_ptr,
            moe_idx_ptr,
            moe_idx1_ptr,
            row,
            cols,
            mask,
            x_stride,
            x1_stride,
            MOE_TOP_K,
            ONE_LOAD,
        )
        x = x.to(x_ptr.dtype.element_ty).to(tl.float32)
        if HAS_X2:
            x2 = tl.load(x2_ptr + row * x2_stride + cols, mask=mask, other=0.0)
            if HAS_GATE:
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
        quantized, sf = mxfp8_quantize_row(y.to(tl.float32), BLOCK)
        tl.store(q_ptr + row * HIDDEN + cols, quantized, mask=mask)
        if STORE_LINEAR_SF:
            tl.store(
                linear_scale_ptr + row * NUM_GROUPS + groups,
                sf.to(tl.uint8),
                mask=groups < NUM_GROUPS,
            )
    else:
        sf = tl.full((BLOCK // 32,), 0, tl.uint32)
    mxfp8_store_swizzled_scales(scale_ptr, row, groups, sf, NUM_GROUPS)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# VLLM_MOE_FIN2_UNFUSED_MIN_TOKENS=N (default 0: off; os.environ, vllm/envs.py
# cannot be overlaid): calls with at least N tokens run moe_finalize2 and then
# the one-buffer norm instead of the fused two-buffer kernel (prefill M).
_FIN2_UNFUSED_MIN_TOKENS = int(
    os.environ.get("VLLM_MOE_FIN2_UNFUSED_MIN_TOKENS", "0") or "0"
)
_fin2_unfused_logged: list[bool] = []


def _log_fin2_unfused(tokens: int) -> None:
    if not _fin2_unfused_logged:
        _fin2_unfused_logged.append(True)
        logger.info(
            "moe-die-split: finalize norm unfused (moe_finalize2 + one-buffer "
            "norm, bitwise) from %d tokens (VLLM_MOE_FIN2_UNFUSED_MIN_TOKENS), "
            "first at %d tokens",
            _FIN2_UNFUSED_MIN_TOKENS,
            tokens,
        )


def moe2_add_rms_norm_mxfp8_quant(
    permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    permuted_idx: torch.Tensor,
    permuted1: torch.Tensor,
    permuted_idx1: torch.Tensor,
    x2: torch.Tensor | None,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    weight_offset: float,
    store_normed: bool,
    x2_gate: torch.Tensor | None = None,
    store_linear_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``add_rms_norm_mxfp8_quant`` with ``moe_route`` whose first summand is
    ``moe_finalize2(permuted, expert_weights, permuted_idx, permuted1,
    permuted_idx1)``; same outputs, launch configuration and arithmetic.
    """
    tokens, hidden, top_k = check_unfinalized2(
        permuted, expert_weights, permuted_idx, permuted1, permuted_idx1
    )
    log_first_call("moe_finalize2_add_rms_norm_mxfp8_quant", tokens)
    shape = (tokens, hidden)
    assert hidden % MXFP8_BLOCK == 0
    assert weight.shape == (hidden,) and weight.is_contiguous()
    if x2 is not None:
        assert x2.shape == shape and x2.dtype == permuted.dtype
        assert x2.stride(-1) == 1
    if x2_gate is not None:
        assert x2 is not None and x2_gate.shape == (tokens, 1)
    if residual is not None:
        assert residual.shape == shape and residual.dtype == permuted.dtype
        assert residual.stride(-1) == 1
    normed_shape, residual_shape, scale_numel, linear_shape = _output_shapes(
        tokens, hidden, residual is not None, store_normed, store_linear_scales
    )
    dev = permuted.device
    normed = torch.empty(normed_shape, dtype=permuted.dtype, device=dev)
    residual_out = torch.empty(residual_shape, dtype=permuted.dtype, device=dev)
    q = torch.empty(shape, dtype=torch.float8_e4m3fn, device=dev)
    scales = torch.empty(scale_numel, dtype=torch.uint8, device=dev)
    linear_scales = torch.empty(linear_shape, dtype=torch.uint8, device=dev)
    if tokens:
        num_warps, launch_pdl = _launch_config(tokens)
        # The one-buffer MoE form's warp rule where this tree has
        # VLLM_MXFP8_NORM_LARGE_M (the reduction order follows the warp count).
        if globals().get("NORM_LARGE_M") and tokens >= globals().get(
            "NORM_LARGE_M_TOKENS", tokens + 1
        ):
            num_warps = 4
        if 0 < _FIN2_UNFUSED_MIN_TOKENS <= tokens:
            # The finalize unfused: moe_finalize2 rounds each row to bf16 as
            # this kernel does before the add, then the one-buffer kernel
            # (MOE_TOP_K 0, no prefetch) runs the same add, norm and quant
            # with the same warp count and PDL: bitwise.
            _log_fin2_unfused(tokens)
            return add_rms_norm_mxfp8_quant(
                moe_finalize2(
                    permuted, expert_weights, permuted_idx, permuted1, permuted_idx1
                ),
                x2,
                residual,
                weight,
                epsilon,
                weight_offset,
                store_normed,
                x2_gate,
                store_linear_scales,
                num_warps=num_warps,
                launch_pdl=launch_pdl,
            )
        x = permuted
        _moe2_add_rms_norm_mxfp8_kernel[(triton.cdiv(tokens, 128) * 128,)](
            x,
            permuted1,
            x if x2 is None else x2,
            x if residual is None else residual,
            weight,
            residual_out,
            normed,
            q,
            scales,
            x if x2_gate is None else x2_gate,
            linear_scales,
            permuted_idx,
            permuted_idx1,
            expert_weights,
            tokens,
            x.stride(0),
            permuted1.stride(0),
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
            EARLY_TRIGGER=launch_pdl and mxfp8_producer_early_trigger(),
            MOE_TOP_K=top_k,
            ONE_LOAD=fin2_one_load(),
            launch_pdl=launch_pdl,
            num_warps=num_warps,
        )
    return normed, residual_out, q, scales, linear_scales


def _moe_finalize2_add_rms_norm_mxfp8_quant_fake(
    permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    permuted_idx: torch.Tensor,
    permuted1: torch.Tensor,
    permuted_idx1: torch.Tensor,
    x2: torch.Tensor | None,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    weight_offset: float,
    store_normed: bool,
    x2_gate: torch.Tensor | None = None,
    store_linear_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _moe_finalize_add_rms_norm_mxfp8_quant_fake(
        permuted,
        expert_weights,
        permuted_idx,
        x2,
        residual,
        weight,
        epsilon,
        weight_offset,
        store_normed,
        x2_gate,
        store_linear_scales,
    )


# moe_finalize_add_rms_norm_mxfp8_quant of a die-split MoE output (two GEMM2
# buffers). Produced by RMSNormMxfp8QuantFusionPass from vllm::moe_finalize2
# feeding the norm.
direct_register_custom_op(
    op_name="moe_finalize2_add_rms_norm_mxfp8_quant",
    op_func=moe2_add_rms_norm_mxfp8_quant,
    fake_impl=_moe_finalize2_add_rms_norm_mxfp8_quant_fake,
)
