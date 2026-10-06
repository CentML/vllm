# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-kernel BF16 GEMMs for the small decode projections on SM107.

At decode token counts cuBLASLt runs the unquantized Qwen3.5/3.6 MoE router
(``M x 2048 -> 256``), ``shared_expert_gate`` (``-> 1``) and GDN ``in_proj_ba``
(``-> 64``) as a split-K GEMM followed by a separate ``splitKreduce`` (or
``dot_kernel`` + ``reduce_1Block``) kernel. On VR (SM107, DRAM clock locked)
one call costs 3.5-4.3 us that way. This module replaces them, per layer, with
one kernel each:

* router / ``in_proj_ba`` (N % 16 == 0): FlashInfer ``tinygemm_bf16`` (bias
  path, which selects the SM100-family ``tinygemm2_sm100`` kernel; the bias is
  a zero vector), 2.1-3.2 us;
* ``shared_expert_gate`` (N == 1): a Triton row-dot kernel, 0.9-1.5 us.

Outside the measured M range of a shape the layer falls back to ``F.linear``.
The M dispatch happens inside a custom op, so one compiled graph serves every
CUDA-graph capture size. Results differ from cuBLAS in the last bit for some
elements (different fp32 summation order); accumulation stays fp32.
"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn.functional as F

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# VLLM_ROWDOT_PDL=1: launch the row-dot kernel with PDL. It waits on the
# predecessor (griddepcontrol.wait) before its first load, so its launch and
# CTA setup overlap the tail of the preceding kernel (the shared expert's
# down_proj GEMM); results are unchanged. Off by default.
ROWDOT_PDL = os.environ.get("VLLM_ROWDOT_PDL", "0") == "1"

# (N, K) -> (largest M served, backend). Measured on VR-288GB (SM107) with the
# DRAM clock locked at 4752 MHz; above the bound cuBLAS is as fast or faster.
_SM107_PLANS: dict[tuple[int, int], tuple[int, str]] = {
    (256, 2048): (208, "tinygemm"),  # MoE router (target and MTP layers)
    (64, 2048): (384, "tinygemm"),  # GDN in_proj_ba
    (1, 2048): (512, "rowdot"),  # shared_expert_gate
}


@triton.jit
def _rowdot_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    K,
    stride_xm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    """``out[m] = sum_k x[m, k] * w[k]`` for BLOCK_M rows per program."""
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        w = tl.load(w_ptr + k0 + offs_k).to(tl.float32)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :],
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += x * w[None, :]
    total = tl.sum(acc, axis=1)
    tl.store(out_ptr + offs_m, total.to(out_ptr.dtype.element_ty), mask=mask_m)


def _rowdot(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    m, k = x.shape
    if out is None:
        out = torch.empty((m, 1), dtype=x.dtype, device=x.device)
    block_k = min(2048, triton.next_power_of_2(k))
    launch_pdl = ROWDOT_PDL and current_platform.is_arch_support_pdl()
    _rowdot_kernel[(m,)](
        x,
        weight,
        out,
        m,
        k,
        x.stride(0),
        BLOCK_M=1,
        BLOCK_K=block_k,
        LAUNCH_PDL=launch_pdl,
        launch_pdl=launch_pdl,
        num_warps=4,
    )
    return out


def _tinygemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    zero_bias: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    from flashinfer.gemm.routergemm import tinygemm_bf16

    if out is None:
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    # Shapes and dtypes are validated when the layer opts in.
    tinygemm_bf16(x, weight, out, bias=zero_bias, skip_check=True)
    return out


@functools.cache
def _tinygemm_available(device: torch.device) -> bool:
    """Whether FlashInfer's ``tinygemm_bf16`` imports and runs on ``device``.

    One tiny call at layer construction also moves its JIT build out of
    profiling and graph capture.
    """
    try:
        x = torch.zeros((1, 64), dtype=torch.bfloat16, device=device)
        w = torch.zeros((16, 64), dtype=torch.bfloat16, device=device)
        zero_bias = torch.zeros(16, dtype=torch.bfloat16, device=device)
        _tinygemm(x, w, zero_bias)
        torch.cuda.synchronize(device)
    except Exception as e:
        logger.warning_once(
            "FlashInfer tinygemm_bf16 unavailable on %s (%s); low-M BF16 GEMM "
            "keeps the default GEMM for its shapes.",
            device,
            e,
        )
        return False
    return True


def lowm_bf16_gemm_impl(
    x: torch.Tensor, weight: torch.Tensor, zero_bias: torch.Tensor
) -> torch.Tensor:
    n, k = weight.shape
    x_2d = x.reshape(-1, k)
    m = x_2d.shape[0]
    max_m, backend = _SM107_PLANS[(n, k)]
    if m == 0 or m > max_m or not x_2d.is_contiguous():
        return F.linear(x, weight)
    if backend == "tinygemm":
        out = _tinygemm(x_2d, weight, zero_bias)
    else:
        out = _rowdot(x_2d, weight)
    return out.view(*x.shape[:-1], n)


def lowm_bf16_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, zero_bias: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="lowm_bf16_gemm",
    op_func=lowm_bf16_gemm_impl,
    fake_impl=lowm_bf16_gemm_fake,
)


def _lowm_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if bias is not None or x.dtype != torch.bfloat16:
        return F.linear(x, weight, bias)
    return torch.ops.vllm.lowm_bf16_gemm(x, weight, layer.lowm_zero_bias)


def lowm_bf16_gemm_out(
    layer: torch.nn.Module, x: torch.Tensor, out: torch.Tensor
) -> bool:
    """Write ``x @ layer.weight.T`` into ``out`` (2-D, contiguous) with the
    low-M kernel when ``layer`` opted in via ``maybe_use_lowm_bf16_gemm`` and
    M is within its plan. Returns False otherwise; the caller then runs its own
    GEMM. For callers that issue the GEMM themselves (e.g. on an aux stream)
    instead of through the layer's ``quant_method``.
    """
    zero_bias = getattr(layer, "lowm_zero_bias", None)
    if zero_bias is None or x.dtype != torch.bfloat16 or not x.is_contiguous():
        return False
    weight = layer.weight
    max_m, backend = _SM107_PLANS[tuple(weight.shape)]
    m = x.shape[0]
    if m == 0 or m > max_m:
        return False
    if backend == "tinygemm":
        _tinygemm(x, weight, zero_bias, out)
    else:
        _rowdot(x, weight, out)
    return True


def maybe_use_lowm_bf16_gemm(layer: torch.nn.Module) -> bool:
    """Route an unquantized bias-free BF16 linear through the low-M kernels.

    Only shapes with a measured plan on SM107 opt in; everything else keeps
    the default unquantized GEMM. Call after the layer is constructed.
    """
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if not (
        envs.VLLM_LOWM_BF16_GEMM
        and current_platform.is_cuda()
        and current_platform.is_device_capability((10, 7))
    ):
        return False
    quant_method = getattr(layer, "quant_method", None)
    weight = getattr(layer, "weight", None)
    if (
        type(quant_method) is not UnquantizedLinearMethod
        or getattr(layer, "bias", None) is not None
        or weight is None
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
    ):
        return False
    n, k = weight.shape
    plan = _SM107_PLANS.get((n, k))
    if plan is None:
        return False
    if plan[1] == "tinygemm" and not _tinygemm_available(weight.device):
        return False
    layer.register_buffer(
        "lowm_zero_bias",
        torch.zeros(n, dtype=torch.bfloat16, device=weight.device),
        persistent=False,
    )
    quant_method._gemm_impl = _lowm_gemm
    logger.info_once(
        "Using low-M BF16 GEMM (%s, M <= %d) for N=%d, K=%d.", plan[1], plan[0], n, k
    )
    return True
