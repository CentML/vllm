# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin (sm_107) K=64 MXFP8 tactics for FlashInfer's autotuned CuTe-DSL mm_mxfp8.

FlashInfer 0.6.18.post1's ``mm_mxfp8(backend="cute-dsl")`` only runs the Sm100
block-scaled kernel (K=32 tcgen05 encoding). The Sm107 kernel
(``flashinfer/gemm/kernels/dense_blockscaled_gemm_sm107.py``, installed by
``tools/install_flashinfer_sm107.py``) emits the doubled-K (instruction K=64)
FP8/MXFP8 MMA. This module ADDS Sm107 tactics to the CuTe-DSL MXFP8 runner's
tactic list, so FlashInfer's autotuner chooses per M bucket between the stock
tactics and these. The linear backend selection is not changed.

Tactic encoding (JSON-safe, so it round-trips through the autotune cache file):
``(107, tiler_m, tile_n, inst_m, cluster_m)``. K=64 tactics are offered only on
sm_107, for bf16/fp16 output, when ``M % 32 == 0``, ``M >= MIN_M`` and the
kernel ``can_implement`` the problem. At run time a K=64 tactic falls back to
the stock default tactic (-1) if the output is not contiguous, M is 0, or the
kernel is not compiled yet while a CUDA graph is being captured (it never
compiles during capture). An unaligned M (eager steps only) is padded to a
multiple of 32 rows.

Installed explicitly by :func:`maybe_install` from the FlashInfer CuTe-DSL MXFP8
linear kernels' ``process_weights_after_loading``; importing this module has no
side effects.

Env (``vllm/envs.py``):

- ``VLLM_FLASHINFER_MXFP8_K64`` (default on; any value except ``0`` / ``false``
  / ``off`` enables): only has an effect on sm_107.
- ``VLLM_FLASHINFER_MXFP8_K64_TACTICS`` (default
  ``512,256,256,2;256,256,256,2;256,128,256,2``): ``;``-separated
  ``tiler_m,tile_n,inst_m,cluster_m``.
- ``VLLM_FLASHINFER_MXFP8_K64_MIN_M`` (default 256).
"""

import threading
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

TACTIC_ARCH = 107
_LOCK = threading.Lock()
_COMPILED: dict = {}
STATS = {"k64_calls": 0, "k64_fallback": 0, "compiles": 0, "tactic_offers": 0}
_STATE: dict[str, Any] = {"installed": False, "tactics": [], "min_m": 256}


def parse_tactics(spec: str) -> list[tuple[int, ...]]:
    """``"a,b,c,d;e,f,g,h"`` -> ``[(a, b, c, d), (e, f, g, h)]``."""
    return [tuple(int(x) for x in t.split(",")) for t in spec.split(";") if t]


def encode_tactic(t: tuple[int, ...]) -> tuple[int, ...]:
    """``(tiler_m, tile_n, inst_m, cluster_m)`` -> autotune tactic."""
    return (TACTIC_ARCH,) + tuple(t)


def is_k64_tactic(tactic: Any) -> bool:
    return (
        isinstance(tactic, (tuple, list))
        and len(tactic) == 5
        and tactic[0] == TACTIC_ARCH
    )


def tactic_shapes(t):
    """Tactic -> (mma_tiler, mma_inst_shape, cluster_shape_mn)."""
    _, tm, tn, im, cm = t
    return (tm, tn, 128), (im, tn, 64), (cm, 1)


def offer_m(m: int, min_m: int) -> bool:
    return m % 32 == 0 and m >= min_m


def _adapter_cls():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.nvgpu import OperandMajorMode
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm107 import (
        Sm107BlockScaledPersistentDenseGemmKernel,
    )

    class _K64Adapter(Sm107BlockScaledPersistentDenseGemmKernel):
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

    return _K64Adapter


def _can_implement(t, m, n, k) -> bool:
    import cutlass
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm107 import (
        Sm107BlockScaledPersistentDenseGemmKernel as K,
    )

    tile, inst, cl = tactic_shapes(t)
    try:
        return bool(
            K._can_implement_impl(
                (m, n, k, 1),
                cutlass.Float8E4M3FN,
                cutlass.Float8E4M3FN,
                cutlass.Float8E8M0FNU,
                cutlass.BFloat16,
                "k",
                "k",
                "n",
                32,
                tile,
                inst,
                cl,
            )
        )
    except Exception:
        # Any failure in the kernel's own check means "not implementable".
        return False


def _get_compiled(dev, n, k, t, out_dtype):
    import torch

    key = (dev, n, k, t, out_dtype)
    c = _COMPILED.get(key)
    if c is not None:
        return c
    if torch.cuda.is_current_stream_capturing():
        return None
    with _LOCK:
        c = _COMPILED.get(key)
        if c is not None:
            return c
        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as utils
        from cutlass.cute.runtime import make_ptr

        tile, inst, cl = tactic_shapes(t)
        kern = _adapter_cls()(
            32, inst, tile, cl, prefetch_dist=0, swizzle_size=1, raster_order="m"
        )
        max_clusters = int(utils.HardwareInfo().get_max_active_clusters(cl[0] * cl[1]))
        m = cute.sym_int(divisibility=32)
        cdt = cutlass.BFloat16 if out_dtype == torch.bfloat16 else cutlass.Float16
        a = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (m, k), stride_order=(1, 0), assumed_align=16
        )
        w = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (n, k), stride_order=(1, 0), assumed_align=16
        )
        o = cute.runtime.make_fake_compact_tensor(
            cdt, (m, n), stride_order=(1, 0), assumed_align=16
        )
        sf = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
        alpha = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), assumed_align=4
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
        fn = cute.compile(
            kern.launch_mxfp8,
            a,
            w,
            o,
            sf,
            sf,
            alpha,
            max_clusters,
            stream,
            options="--gpu-arch sm_107a --opt-level 2 --enable-tvm-ffi",
        )
        one = torch.ones(1, device=torch.device("cuda", dev), dtype=torch.float32)
        torch.cuda.current_stream(dev).synchronize()
        c = (fn, one)
        _COMPILED[key] = c
        STATS["compiles"] += 1
        return c


def install(gb, tactics, min_m: int) -> None:
    """Wrap ``gb._cute_dsl_gemm_mxfp8_runner`` (``flashinfer.gemm.gemm_base``)."""
    import torch

    orig_factory = gb._cute_dsl_gemm_mxfp8_runner
    cache: dict = {}

    def factory(sm_major, sm_minor, enable_pdl, out_dtype):
        key = (sm_major, sm_minor, enable_pdl, out_dtype)
        if key in cache:
            # The stock factory builds a new class + instance per call; the
            # wrapped runner is built once per key.
            return cache[key]
        base = orig_factory(sm_major, sm_minor, enable_pdl, out_dtype)
        if (sm_major, sm_minor) != (10, 7) or out_dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            return base
        Base = type(base)

        class CuteDSLMxfp8K64GemmRunner(Base):
            def get_valid_tactics(self, inputs, profile):
                out = list(Base.get_valid_tactics(self, inputs, profile))
                a, b = inputs[0], inputs[1]
                m, k, n = a.shape[0], a.shape[1], b.shape[1]
                if offer_m(m, min_m) and inputs[5].is_contiguous():
                    for t in tactics:
                        tt = encode_tactic(t)
                        if _can_implement(tt, m, n, k):
                            out.append(tt)
                            STATS["tactic_offers"] += 1
                return out

            def forward(self, inputs, tactic=None, do_preparation=False, **kwargs):
                if is_k64_tactic(tactic):
                    a, b, a_sf, b_sf, _, out, _ = inputs
                    m, k, n = a.shape[0], a.shape[1], b.shape[1]
                    c = None
                    if out.is_contiguous() and m > 0:
                        c = _get_compiled(
                            a.device.index, n, k, tuple(tactic), out.dtype
                        )
                    if c is None:
                        STATS["k64_fallback"] += 1
                        return Base.forward(
                            self,
                            inputs,
                            tactic=-1,
                            do_preparation=do_preparation,
                            **kwargs,
                        )
                    import cuda.bindings.driver as cuda

                    fn, one = c
                    stream = cuda.CUstream(
                        torch.cuda.current_stream(a.device).cuda_stream
                    )
                    if m % 32 == 0:
                        fn(a, b.t(), out, a_sf.data_ptr(), b_sf.data_ptr(), one, stream)
                    else:
                        # Unaligned M (eager steps only; graph capture sizes are
                        # aligned): pad rows to 32. The 128x4-swizzled scales
                        # already cover ceil(M/128)*128 rows; padded rows are
                        # computed from zero data and discarded.
                        pm = (m + 31) // 32 * 32
                        a_p = torch.zeros((pm, k), dtype=a.dtype, device=a.device)
                        a_p[:m].copy_(a)
                        o_p = torch.empty((pm, n), dtype=out.dtype, device=out.device)
                        fn(
                            a_p,
                            b.t(),
                            o_p,
                            a_sf.data_ptr(),
                            b_sf.data_ptr(),
                            one,
                            stream,
                        )
                        out.copy_(o_p[:m])
                        STATS["k64_padded"] = STATS.get("k64_padded", 0) + 1
                    STATS["k64_calls"] += 1
                    return out
                return Base.forward(
                    self, inputs, tactic=tactic, do_preparation=do_preparation, **kwargs
                )

        r = CuteDSLMxfp8K64GemmRunner.__new__(CuteDSLMxfp8K64GemmRunner)
        r.__dict__.update(base.__dict__)
        cache[key] = r
        return r

    gb._cute_dsl_gemm_mxfp8_runner = factory


def maybe_install() -> bool:
    """Install the K=64 tactics once per process.

    Only on sm_107 and when ``VLLM_FLASHINFER_MXFP8_K64`` is on. Returns
    whether they are installed. Never raises: any failure is logged and the
    stock tactics stay in use.
    """
    if _STATE["installed"]:
        return True
    if not envs.VLLM_FLASHINFER_MXFP8_K64:
        return False
    from vllm.platforms import current_platform

    if not (current_platform.is_cuda() and current_platform.is_device_capability(107)):
        return False
    with _LOCK:
        if _STATE["installed"]:
            return True
        try:
            import flashinfer.gemm.gemm_base as gb

            tactics = parse_tactics(envs.VLLM_FLASHINFER_MXFP8_K64_TACTICS)
            min_m = int(envs.VLLM_FLASHINFER_MXFP8_K64_MIN_M)
            install(gb, tactics, min_m)
        except Exception as e:
            logger.warning("FlashInfer sm_107 K=64 MXFP8 tactics not installed: %r", e)
            return False
        _STATE.update(installed=True, tactics=tactics, min_m=min_m)
    logger.info(
        "FlashInfer sm_107 K=64 CuTe-DSL MXFP8 tactics added: %s (min M %d)",
        tactics,
        min_m,
    )
    return True
