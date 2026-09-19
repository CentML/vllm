# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental MXFP8 entry point for FlashInfer's native SM107 GEMM."""

from functools import cache
from threading import Lock
from typing import Any

import torch

_MMA_TILE = (512, 256, 128)
_MMA_INST = (256, 256, 64)
_CLUSTER = (2, 1)
_KERNELS: dict[tuple[int, int, int], tuple[Any, torch.Tensor]] = {}
_KERNEL_LOCK = Lock()


@cache
def _native_kernel_type():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cutlass_dsl import GPUArch
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm107 import (
        Sm107BlockScaledPersistentDenseGemmKernel,
    )

    # Check that this toolchain recognizes the per-compilation target.
    GPUArch("sm_107a")

    class Mxfp8TensorAdapter(Sm107BlockScaledPersistentDenseGemmKernel):
        @cute.jit
        def launch_mxfp8(
            self,
            a: cute.Tensor,
            w: cute.Tensor,
            c: cute.Tensor,
            a_scale: cute.Pointer,
            b_scale: cute.Pointer,
            alpha: cute.Tensor,
            max_clusters: cutlass.Constexpr,
            stream: cuda.CUstream,
        ):
            # The upstream tensor wrapper interprets its inputs as packed FP4.
            # The pointer interface also supports FP8 without doubling K.
            problem = (
                cutlass.Int32(cute.size(a, mode=[0])),
                cutlass.Int32(cute.size(w, mode=[0])),
                cutlass.Int32(cute.size(a, mode=[1])),
                1,
            )
            layouts = (
                OperandMajorMode.K,
                OperandMajorMode.K,
                utils.LayoutEnum.ROW_MAJOR,
            )
            self(
                a.iterator,
                w.iterator,
                a_scale,
                b_scale,
                c.iterator,
                alpha,
                layouts,
                problem,
                max_clusters,
                stream,
            )

    return Mxfp8TensorAdapter


@cache
def sm107_mxfp8_support_reason() -> str | None:
    """Return the missing optional dependency, if any, for explicit selection."""
    try:
        _native_kernel_type()
    except ImportError as exc:
        return f"Native SM107 MXFP8 requires compatible FlashInfer/CuTe DSL: {exc}"
    return None


def _get_kernel(device_index: int, n: int, k: int):
    key = (device_index, n, k)
    with torch.accelerator.device_index(device_index), _KERNEL_LOCK:
        if key in _KERNELS:
            return _KERNELS[key]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Warm up native SM107 MXFP8 for this device and N/K before "
                "CUDA graph capture; compilation during capture is unsupported."
            )

        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as utils
        from cutlass.cute.runtime import make_ptr

        kernel = _native_kernel_type()(
            32,
            _MMA_INST,
            _MMA_TILE,
            _CLUSTER,
            prefetch_dist=0,
            swizzle_size=1,
            raster_order="m",
        )
        max_clusters = int(utils.HardwareInfo().get_max_active_clusters(2))
        m = cute.sym_int(divisibility=32)
        a = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (m, k),
            stride_order=(1, 0),
            assumed_align=16,
        )
        w = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (n, k),
            stride_order=(1, 0),
            assumed_align=16,
        )
        out = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (m, n),
            stride_order=(1, 0),
            assumed_align=16,
        )
        sf = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
        alpha = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), assumed_align=4
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
        compiled = cute.compile(
            kernel.launch_mxfp8,
            a,
            w,
            out,
            sf,
            sf,
            alpha,
            max_clusters,
            stream,
            options="--gpu-arch sm_107a --opt-level 2 --enable-tvm-ffi",
        )
        alpha_one = torch.ones(
            1, device=torch.device("cuda", device_index), dtype=torch.float32
        )
        # Publish only after initialization: later calls may use another stream.
        torch.cuda.current_stream(device_index).synchronize()
        entry = (compiled, alpha_one)
        _KERNELS[key] = entry
        return entry


def _native_eligible(A, B, A_scale, B_scale, out_dtype) -> bool:
    m, k = A.shape
    n = B.shape[1]
    if (
        m == 0
        or n == 0
        or n % 32
        or k == 0
        or k % 128
        or out_dtype != torch.bfloat16
        or A.dtype != torch.float8_e4m3fn
        or B.dtype != A.dtype
        or not A.is_contiguous()
        or not B.t().is_contiguous()
    ):
        return False
    if torch.cuda.get_device_capability(A.device) != (10, 7):
        return False
    for tensor in (A, B, A_scale, B_scale):
        if tensor.device != A.device or tensor.data_ptr() % 16:
            return False
    for sf, rows in ((A_scale, m), (B_scale, n)):
        if (
            sf.dtype != torch.uint8
            or sf.ndim != 1
            or not sf.is_contiguous()
            or sf.numel() != ((rows + 127) // 128) * 128 * (k // 32)
        ):
            return False
    return True


@torch.library.custom_op("vllm::mm_mxfp8_sm107", mutates_args=[], device_types="cuda")
def mm_mxfp8_sm107(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute A[M,K] @ B[K,N] with F8_128x4 E8M0 block32 scales."""
    from vllm.utils.flashinfer import mm_mxfp8

    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("Expected A[M,K] and column-major B[K,N]")
    if not _native_eligible(A, B, A_scale, B_scale, out_dtype):
        return mm_mxfp8(A, B, A_scale, B_scale, out_dtype, backend="cute-dsl")

    import cuda.bindings.driver as cuda
    import cutlass
    from flashinfer.autotuner import AutoTuner

    m, k = A.shape
    n = B.shape[1]
    padded_m = (m + 31) // 32 * 32
    kernel_type = _native_kernel_type()
    if not kernel_type._can_implement_impl(
        (padded_m, n, k, 1),
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        cutlass.BFloat16,
        "k",
        "k",
        "n",
        32,
        _MMA_TILE,
        _MMA_INST,
        _CLUSTER,
    ):
        return mm_mxfp8(A, B, A_scale, B_scale, out_dtype, backend="cute-dsl")

    compiled, alpha = _get_kernel(A.device.index, n, k)
    # Warm both paths while the existing FlashInfer tuning context is active.
    if AutoTuner.get().is_tuning_mode:
        return mm_mxfp8(A, B, A_scale, B_scale, out_dtype, backend="cute-dsl")

    if padded_m != m:
        padded_a = torch.empty((padded_m, k), dtype=A.dtype, device=A.device)
        padded_a[:m].copy_(A)
        padded_a[m:].zero_()
        A = padded_a
        # Scales are [M128,K128,32,4,4]. Initialize only the extra executed
        # rows to E8M0 1.0; incoming QuantizedActivation scales remain untouched.
        A_scale = A_scale.clone()
        A_scale.view(-1, k // 128, 32, 4, 4)[
            m // 128, :, m % 32 :, (m % 128) // 32, :
        ].fill_(127)
    out = torch.empty((padded_m, n), dtype=out_dtype, device=A.device)
    stream = cuda.CUstream(torch.cuda.current_stream(A.device).cuda_stream)
    compiled(A, B.t(), out, A_scale.data_ptr(), B_scale.data_ptr(), alpha, stream)
    return out[:m]


@mm_mxfp8_sm107.register_fake
def _mm_mxfp8_sm107_fake(A, B, A_scale, B_scale, out_dtype):
    return A.new_empty((A.shape[0], B.shape[1]), dtype=out_dtype)
