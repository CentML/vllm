# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k reduction of a deferred (``do_finalize=False``) TRT-LLM MoE output.

``vllm::moe_finalize`` computes, per token ``t``,
``bf16(sum_k w[t, k] * permuted[idx[t, k]])`` with the arithmetic of the
TRT-LLM-Gen finalize kernels (``finalizeKernel`` and ``finalizeKernelVecLoad``):
an fp32 accumulator starting at zero, one fused multiply-add per expert in
``k`` order, entries with ``idx == -1`` skipped, one round to bf16. The
reduction (``moe_finalize_row``) is also inlined into the consumer
``vllm::moe_finalize_add_rms_norm_mxfp8_quant``, so the routed output is never
written; the standalone op serves consumers that are not fused.

``permuted`` is the ``[tokens * top_k, hidden]`` leading view of the GEMM2
output, whose storage holds every permuted row (the expert-tile padding makes
it longer); rows are addressed through ``idx``.

``VLLM_MOE_FINALIZE_PDL=1`` opts the standalone launch into PDL on supported
devices. It waits before reading the deferred triple and releases dependents
early only under the same safe-consumer policy as the fused MXFP8 producers.
The default is off; reduction order, rounding and launch geometry do not change.
"""

import torch

from vllm import envs
from vllm.model_executor.layers.fusion.mxfp8_pdl import (
    mxfp8_producer_early_trigger,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def moe_finalize_row(
    permuted_ptr,
    weights_ptr,
    idx_ptr,
    row,
    cols,
    mask,
    permuted_stride,
    TOP_K: tl.constexpr,
):
    """fp32 top-k sum of token ``row`` over columns ``cols``, TRT-LLM order."""
    acc = tl.zeros(cols.shape, tl.float32)
    for k in tl.static_range(TOP_K):
        p = tl.load(idx_ptr + row * TOP_K + k)
        w = tl.load(weights_ptr + row * TOP_K + k).to(tl.float32)
        valid = p >= 0
        v = tl.load(
            permuted_ptr + p.to(tl.int64) * permuted_stride + cols,
            mask=mask & valid,
            other=0.0,
        ).to(tl.float32)
        acc = tl.where(valid, tl.fma(w, v, acc), acc)
    return acc


@triton.jit
def _moe_finalize_kernel(
    permuted_ptr,
    weights_ptr,
    idx_ptr,
    out_ptr,
    permuted_stride,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
):
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        if EARLY_TRIGGER:
            tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    acc = moe_finalize_row(
        permuted_ptr, weights_ptr, idx_ptr, row, cols, mask, permuted_stride, TOP_K
    )
    tl.store(out_ptr + row * HIDDEN + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)
    if LAUNCH_PDL and not EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()


def check_unfinalized(
    permuted: torch.Tensor, expert_weights: torch.Tensor, permuted_idx: torch.Tensor
) -> tuple[int, int, int]:
    """Validate the deferred MoE triple; return (tokens, hidden, top_k)."""
    assert permuted_idx.ndim == 2 and permuted_idx.dtype == torch.int32
    tokens, top_k = permuted_idx.shape
    assert expert_weights.shape == (tokens, top_k)
    assert expert_weights.is_contiguous() and permuted_idx.is_contiguous()
    assert permuted.ndim == 2 and permuted.stride(-1) == 1
    assert permuted.shape[0] == tokens * top_k
    return tokens, permuted.shape[1], top_k


def moe_finalize(
    permuted: torch.Tensor, expert_weights: torch.Tensor, permuted_idx: torch.Tensor
) -> torch.Tensor:
    tokens, hidden, top_k = check_unfinalized(permuted, expert_weights, permuted_idx)
    out = torch.empty((tokens, hidden), dtype=permuted.dtype, device=permuted.device)
    if tokens:
        block = triton.next_power_of_2(hidden)
        launch_pdl = (
            envs.VLLM_MOE_FINALIZE_PDL and current_platform.is_arch_support_pdl()
        )
        _moe_finalize_kernel[(tokens,)](
            permuted,
            expert_weights,
            permuted_idx,
            out,
            permuted.stride(0),
            HIDDEN=hidden,
            BLOCK=block,
            TOP_K=top_k,
            LAUNCH_PDL=launch_pdl,
            EARLY_TRIGGER=launch_pdl and mxfp8_producer_early_trigger(),
            launch_pdl=launch_pdl,
            # 16-byte loads per thread for bf16, as finalizeKernelVecLoad.
            num_warps=max(1, min(8, block // 256)),
        )
    return out


def _moe_finalize_fake(
    permuted: torch.Tensor, expert_weights: torch.Tensor, permuted_idx: torch.Tensor
) -> torch.Tensor:
    return permuted.new_empty((permuted_idx.shape[0], permuted.shape[1]))


direct_register_custom_op(
    op_name="moe_finalize",
    op_func=moe_finalize,
    fake_impl=_moe_finalize_fake,
)
