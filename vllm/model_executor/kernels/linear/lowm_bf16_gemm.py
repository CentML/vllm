# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-kernel BF16 GEMMs for small decode projections.

FlashInfer's bias-path ``tinygemm_bf16`` (a zero bias selects TinyGEMM2)
serves contiguous BF16 weights with N % 16 == 0 and K % 64 == 0, at
1 <= M <= 64. This includes TP-sharded GDN ``in_proj_ba`` with N=32. SM107
uses it by default and additionally retains its measured router and BA plans
with larger M windows, and its shared-expert-gate row-dot plan. SM100 and
SM103 (GB300) use TinyGEMM2 only with ``VLLM_LOWM_BF16_GEMM_SM100=1``.
Other architectures, layouts, dtypes, and token counts keep ``F.linear``.

``VLLM_LOWM_BF16_GEMM`` enables layer opt-in (on by default).
``VLLM_LOWM_BF16_GEMM_SM100`` additionally opts SM100/SM103 in (off by
default). ``VLLM_LOWM_BF16_GEMM_PDL`` optionally launches TinyGEMM2 with
programmatic dependent launch (off by default). Construction probes the
selected launch variant before profiling or graph capture. SM107 row-dot
retains its separate ``VLLM_ROWDOT_PDL`` control.

M dispatch happens inside a custom op, so one compiled graph serves every
CUDA-graph capture size. Results can differ from cuBLAS in the last BF16 bit
due to fp32 summation order; accumulation stays fp32.
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


def _plan(n: int, k: int, device: torch.device) -> tuple[int, str] | None:
    """Select only the architectures supported by the SM100 TinyGEMM2 kernel.

    SM107 is eligible by default; SM100/SM103 require the
    ``VLLM_LOWM_BF16_GEMM_SM100`` opt-in.
    """
    if device.type != "cuda" or not current_platform.is_cuda():
        return None
    device_id = device.index
    if device_id is None:
        device_id = torch.accelerator.current_device_index()
    capability = current_platform.get_device_capability(device_id)
    if capability is None:
        return None
    cc = (capability.major, capability.minor)
    if cc == (10, 7):
        measured = _SM107_PLANS.get((n, k))
        if measured is not None:
            return measured
    elif cc in ((10, 0), (10, 3)):
        if not (envs.VLLM_LOWM_BF16_GEMM and envs.VLLM_LOWM_BF16_GEMM_SM100):
            return None
    else:
        return None
    if n > 0 and k > 0 and n % 16 == 0 and k % 64 == 0:
        return 64, "tinygemm"
    return None


def _runtime_plan(
    x: torch.Tensor, weight: torch.Tensor, zero_bias: torch.Tensor
) -> tuple[int, str] | None:
    # Check layouts before flattening: reshape must not silently copy a
    # strided input merely to make it eligible for the fast path.
    if (
        weight.ndim != 2
        or x.ndim < 1
        or x.shape[-1] != weight.shape[1]
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or zero_bias.dtype != torch.bfloat16
        or x.device != weight.device
        or zero_bias.device != weight.device
        or zero_bias.shape != (weight.shape[0],)
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or not zero_bias.is_contiguous()
        or x.storage_offset() % 8 != 0
        or weight.storage_offset() % 8 != 0
        or zero_bias.storage_offset() % 8 != 0
    ):
        return None
    plan = _plan(weight.shape[0], weight.shape[1], weight.device)
    if plan is None:
        return None
    m = x.numel() // weight.shape[1]
    return plan if 0 < m <= plan[0] else None


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
    *,
    use_pdl: bool | None = None,
) -> torch.Tensor:
    from flashinfer.gemm.routergemm import tinygemm_bf16

    if out is None:
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    # Shapes, dtypes, devices, and layouts are validated by the dispatcher.
    if use_pdl is None:
        use_pdl = envs.VLLM_LOWM_BF16_GEMM_PDL
    tinygemm_bf16(x, weight, out, bias=zero_bias, use_pdl=use_pdl, skip_check=True)
    return out


@functools.cache
def _tinygemm_available(device: torch.device, use_pdl: bool) -> bool:
    """Whether FlashInfer's ``tinygemm_bf16`` imports and runs on ``device``.

    One tiny call at layer construction also moves its JIT build out of
    profiling and graph capture.
    """
    try:
        x = torch.zeros((1, 64), dtype=torch.bfloat16, device=device)
        w = torch.zeros((16, 64), dtype=torch.bfloat16, device=device)
        zero_bias = torch.zeros(16, dtype=torch.bfloat16, device=device)
        _tinygemm(x, w, zero_bias, use_pdl=use_pdl)
        torch.accelerator.synchronize(device)
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
    plan = _runtime_plan(x, weight, zero_bias)
    if plan is None:
        return F.linear(x, weight)
    n, k = weight.shape
    _, backend = plan
    x_2d = x.view(-1, k)
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
    if zero_bias is None or x.ndim != 2:
        return False
    weight = layer.weight
    plan = _runtime_plan(x, weight, zero_bias)
    if plan is None:
        return False
    _, backend = plan
    m = x.shape[0]
    if (
        out.shape != (m, weight.shape[0])
        or out.dtype != x.dtype
        or out.device != x.device
        or not out.is_contiguous()
        or out.storage_offset() % 8 != 0
    ):
        return False
    if backend == "tinygemm":
        _tinygemm(x, weight, zero_bias, out)
    else:
        _rowdot(x, weight, out)
    return True


def maybe_use_lowm_bf16_gemm(layer: torch.nn.Module) -> bool:
    """Route an unquantized bias-free BF16 linear through the low-M kernels.

    TinyGEMM2 covers aligned shapes up to M=64 on SM107, and on SM100/SM103
    with ``VLLM_LOWM_BF16_GEMM_SM100=1``. SM107 keeps its measured larger-M
    plans, including shared-gate row-dot.
    Call after the layer is constructed.
    """
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if not envs.VLLM_LOWM_BF16_GEMM:
        return False
    quant_method = getattr(layer, "quant_method", None)
    weight = getattr(layer, "weight", None)
    if (
        type(quant_method) is not UnquantizedLinearMethod
        or getattr(layer, "bias", None) is not None
        or weight is None
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or not weight.is_contiguous()
        or weight.storage_offset() % 8 != 0
    ):
        return False
    n, k = weight.shape
    plan = _plan(n, k, weight.device)
    if plan is None:
        return False
    if plan[1] == "tinygemm" and not _tinygemm_available(
        weight.device, envs.VLLM_LOWM_BF16_GEMM_PDL
    ):
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
