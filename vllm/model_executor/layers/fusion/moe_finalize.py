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
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# H105: launch the standalone finalize with PDL (it waits before its first
# load and triggers nothing, so any producer and consumer stay ordered).
_H105_PDL = os.environ.get("VLLM_H105_PDL", "0") == "1"
_pdl: list[bool] = []


def _launch_pdl() -> bool:
    if not _H105_PDL:
        return False
    if not _pdl:
        from vllm.platforms import current_platform

        _pdl.append(current_platform.is_arch_support_pdl())
        logger.info(
            "H105 PDL on (VLLM_H105_PDL=1): standalone MoE finalize launches with "
            "PDL: %s",
            _pdl[0],
        )
    return _pdl[0]


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
    LAUNCH_PDL: tl.constexpr = False,
):
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    acc = moe_finalize_row(
        permuted_ptr, weights_ptr, idx_ptr, row, cols, mask, permuted_stride, TOP_K
    )
    tl.store(out_ptr + row * HIDDEN + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)


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
        pdl = _launch_pdl()
        _moe_finalize_kernel[(tokens,)](
            permuted,
            expert_weights,
            permuted_idx,
            out,
            permuted.stride(0),
            HIDDEN=hidden,
            BLOCK=block,
            TOP_K=top_k,
            LAUNCH_PDL=pdl,
            launch_pdl=pdl,
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


# Two-buffer form (die-split MoE, moe_die_split): the routes of each token are
# split between two GEMM2 outputs, each (t, k) valid (>= 0) in exactly one map.
# The k-order fused multiply-adds and the single bf16 round are moe_finalize's.
_FIRST_CALLS: set[str] = set()


def log_first_call(op: str, tokens: int) -> None:
    """Log the first call of a two-buffer op in this process (engagement)."""
    if op not in _FIRST_CALLS:
        _FIRST_CALLS.add(op)
        logger.info("moe-die-split: first %s call, %d tokens", op, tokens)


# VLLM_MOE_FIN2_ONE_LOAD=1 (default off; os.environ: vllm/envs.py cannot be
# overlaid): the two-buffer reduction takes each route's row address from the
# map that holds it and issues one masked row load per k, instead of one masked
# load per buffer and a select. The row bits loaded, the k-order fp32 FMAs and
# the bf16 round are unchanged (also if both or neither map held the route),
# so the result is bit for bit the two-load form's.
_FIN2_ONE_LOAD = os.environ.get("VLLM_MOE_FIN2_ONE_LOAD", "0") == "1"
_fin2_one_load_logged: list[bool] = []


def fin2_one_load() -> bool:
    """Whether the two-buffer finalize kernels use the one-load form."""
    if _FIN2_ONE_LOAD and not _fin2_one_load_logged:
        _fin2_one_load_logged.append(True)
        logger.info(
            "moe-die-split: two-buffer finalize loads one row per route "
            "(VLLM_MOE_FIN2_ONE_LOAD=1; same k-order FMAs, bitwise)"
        )
    return _FIN2_ONE_LOAD


@triton.jit
def moe_finalize_row2(
    permuted_ptr,
    permuted1_ptr,
    weights_ptr,
    idx_ptr,
    idx1_ptr,
    row,
    cols,
    mask,
    permuted_stride,
    permuted1_stride,
    TOP_K: tl.constexpr,
    ONE_LOAD: tl.constexpr,
):
    """``moe_finalize_row`` over the routes of two permuted buffers."""
    acc = tl.zeros(cols.shape, tl.float32)
    for k in tl.static_range(TOP_K):
        p = tl.load(idx_ptr + row * TOP_K + k)
        p1 = tl.load(idx1_ptr + row * TOP_K + k)
        w = tl.load(weights_ptr + row * TOP_K + k).to(tl.float32)
        valid = p >= 0
        valid1 = p1 >= 0
        if ONE_LOAD:
            # Buffer 0 wins if both maps hold the route, as tl.where below.
            src = tl.where(
                valid,
                permuted_ptr + p.to(tl.int64) * permuted_stride,
                permuted1_ptr + p1.to(tl.int64) * permuted1_stride,
            )
            v = tl.load(src + cols, mask=mask & (valid | valid1), other=0.0).to(
                tl.float32
            )
        else:
            v = tl.load(
                permuted_ptr + p.to(tl.int64) * permuted_stride + cols,
                mask=mask & valid,
                other=0.0,
            ).to(tl.float32)
            v1 = tl.load(
                permuted1_ptr + p1.to(tl.int64) * permuted1_stride + cols,
                mask=mask & valid1,
                other=0.0,
            ).to(tl.float32)
            v = tl.where(valid, v, v1)
        acc = tl.where(valid | valid1, tl.fma(w, v, acc), acc)
    return acc


@triton.jit
def _moe_finalize2_kernel(
    permuted_ptr,
    permuted1_ptr,
    weights_ptr,
    idx_ptr,
    idx1_ptr,
    out_ptr,
    permuted_stride,
    permuted1_stride,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
    ONE_LOAD: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    acc = moe_finalize_row2(
        permuted_ptr,
        permuted1_ptr,
        weights_ptr,
        idx_ptr,
        idx1_ptr,
        row,
        cols,
        mask,
        permuted_stride,
        permuted1_stride,
        TOP_K,
        ONE_LOAD,
    )
    tl.store(out_ptr + row * HIDDEN + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)


def check_unfinalized2(
    permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    permuted_idx: torch.Tensor,
    permuted1: torch.Tensor,
    permuted_idx1: torch.Tensor,
) -> tuple[int, int, int]:
    """Validate the two-buffer deferred MoE output; return (tokens, hidden, top_k)."""
    tokens, hidden, top_k = check_unfinalized(permuted, expert_weights, permuted_idx)
    assert permuted_idx1.shape == permuted_idx.shape
    assert permuted_idx1.dtype == torch.int32 and permuted_idx1.is_contiguous()
    assert permuted1.shape == permuted.shape and permuted1.dtype == permuted.dtype
    assert permuted1.stride(-1) == 1
    return tokens, hidden, top_k


def moe_finalize2(
    permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    permuted_idx: torch.Tensor,
    permuted1: torch.Tensor,
    permuted_idx1: torch.Tensor,
) -> torch.Tensor:
    tokens, hidden, top_k = check_unfinalized2(
        permuted, expert_weights, permuted_idx, permuted1, permuted_idx1
    )
    log_first_call("moe_finalize2", tokens)
    out = torch.empty((tokens, hidden), dtype=permuted.dtype, device=permuted.device)
    if tokens:
        block = triton.next_power_of_2(hidden)
        _moe_finalize2_kernel[(tokens,)](
            permuted,
            permuted1,
            expert_weights,
            permuted_idx,
            permuted_idx1,
            out,
            permuted.stride(0),
            permuted1.stride(0),
            HIDDEN=hidden,
            BLOCK=block,
            TOP_K=top_k,
            ONE_LOAD=fin2_one_load(),
            num_warps=max(1, min(8, block // 256)),
        )
    return out


def _moe_finalize2_fake(
    permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    permuted_idx: torch.Tensor,
    permuted1: torch.Tensor,
    permuted_idx1: torch.Tensor,
) -> torch.Tensor:
    return permuted.new_empty((permuted_idx.shape[0], permuted.shape[1]))


direct_register_custom_op(
    op_name="moe_finalize2",
    op_func=moe_finalize2,
    fake_impl=_moe_finalize2_fake,
)
