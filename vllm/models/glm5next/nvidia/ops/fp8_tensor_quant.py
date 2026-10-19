# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic per-tensor FP8 E4M3 quantization in two Triton launches.

Used by the GLM-5.3 hybrid_tensor FP8 linear path
(``additional_config.glm53_fp8_linear_act_quant = "triton"``) in place of
``ops.scaled_fp8_quant(x)`` (absmax_reduction_flat +
scaled_fp8_quant_kernel_strided_dynamic). Same math: scale = absmax / 448,
q = clamp(x / scale). The absmax kernel is persistent (a few CTAs per SM,
fp32 atomic max); the quant kernel is a flat, vectorized elementwise pass.
No host synchronization, CUDA-graph safe.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_BLOCK = 4096


@triton.jit
def _absmax_kernel(x_ptr, amax_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(pid * BLOCK, n, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(v))
    tl.atomic_max(amax_ptr, tl.max(acc, 0))


@triton.jit
def _quant_kernel(
    x_ptr, amax_ptr, q_ptr, scale_ptr, n, FP8_MAX: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    scale = tl.maximum(tl.load(amax_ptr) / FP8_MAX, 1e-12)
    if pid == 0:
        tl.store(scale_ptr, scale)
    v = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    q = tl.clamp(v / scale, -FP8_MAX, FP8_MAX)
    tl.store(q_ptr + offs, q.to(q_ptr.dtype.element_ty), mask=offs < n)


_NUM_SMS: list[int] = []


def fp8_tensor_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous tensor to FP8 E4M3 with one dynamic scale.

    Returns (q with x's shape, scale as a 0-d fp32 tensor with x ~= q * scale).
    """
    assert x.is_contiguous()
    if not _NUM_SMS:
        _NUM_SMS.append(
            torch.cuda.get_device_properties(x.device).multi_processor_count
        )
    fp8_max = torch.finfo(current_platform.fp8_dtype()).max
    n = x.numel()
    amax = torch.zeros((), device=x.device, dtype=torch.float32)
    _absmax_kernel[(max(1, min(4 * _NUM_SMS[0], triton.cdiv(n, _BLOCK))),)](
        x, amax, n, BLOCK=_BLOCK
    )
    q = torch.empty_like(x, dtype=current_platform.fp8_dtype())
    scale = torch.empty((), device=x.device, dtype=torch.float32)
    _quant_kernel[(triton.cdiv(n, _BLOCK),)](
        x, amax, q, scale, n, FP8_MAX=fp8_max, BLOCK=_BLOCK
    )
    return q, scale
