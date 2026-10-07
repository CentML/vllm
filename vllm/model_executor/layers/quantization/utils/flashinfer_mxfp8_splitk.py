# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mid-M split-K tactics (and a cluster-size cap) for FlashInfer's CuTe-DSL mm_mxfp8.

FlashInfer 0.6.18.post1 offers its split-K MXFP8 kernel
(``Sm100BlockScaledSplitKGemmKernel``: swap-AB, a (1, 1, s) K-cluster per
output tile, fp32 partials reduced in the owner CTA over DSMEM) only for
M <= 32, with tile N tied to M (8/16/32). Above that every dense MXFP8 GEMM
runs the persistent kernel with one tile per CTA, so the K = 4096 GEMMs
(out_proj / o_proj, N = 2048) stream all of K serially on 64-176 CTAs.

This module ADDS split-K tactics with tile N 64/128 and 2 or 4 K-slices for
32 < M <= ``VLLM_MXFP8_SPLITK_MIDM_MAX_M`` to the CuTe-DSL MXFP8 runner's
tactic list, so FlashInfer's autotuner picks per M bucket between them and
the stock tactics. The kernel is ``vllm/lcd_pdl/
fi_dense_blockscaled_gemm_sm100_splitk_midm.py``: FlashInfer's split-K kernel
(with the lcd_pdl PFWS weight prefetch) extended to several token tiles (SFB
slicing for tile N 64) and with a cheaper reduction: each peer stages its
fp32 partial subtile in its own idle A stage buffers and sends it with one
bulk DSMEM copy (the stock kernel issues one ``st.async`` + remote mbarrier
update per 16 B, which costs ~2.5 us per call at tile N 64), all sends before
any owner waits.

Numerics: each K-slice accumulates in fp32; the owner CTA of an epilogue
subtile adds the peers' partials in fixed slot order, then applies alpha and
casts once. The summation order is fixed per tactic, so the output is
bitwise-reproducible run to run, but differs from the unsplit tactics in the
fp32 summation order (1-ulp class).

Tactic encoding (JSON-safe, round-trips through the autotune cache file):
``(1072, tile_n, split_k)``. A tactic is offered only when the output is
contiguous, ``K % (128 * split_k) == 0`` and the grid
(``ceil(N/128) * ceil(M/tile_n)`` K-clusters of ``split_k`` CTAs) fits in one
wave. The wrapped runner keeps the base runner's class name, so autotune file
keys do not change; a file that contains a ``1072`` tactic needs this module
enabled.

``VLLM_MXFP8_MAX_CLUSTER_CTAS`` (setting experiment for the mixed-batch grids
capped at 176/184 CTAs by 8/4-CTA clusters): when > 0, stock persistent
tactics whose cluster has more CTAs are dropped at
M >= ``VLLM_MXFP8_MAX_CLUSTER_CTAS_MIN_M``.

Installed explicitly by :func:`maybe_install` from the FlashInfer CuTe-DSL
MXFP8 linear kernels' ``process_weights_after_loading`` (after the K=64
tactics); importing this module has no side effects.
"""

import functools
import threading
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

TACTIC_TAG = 1072
TILE_M = 128
TILE_K = 128
SUPPORTED_TILE_N = (64, 128)
SUPPORTED_SPLIT_K = (2, 4)
_LOCK = threading.Lock()
_KERNEL_CACHE: dict = {}
STATS = {"calls": 0, "tactic_offers": 0, "cluster_dropped": 0}
_STATE: dict[str, Any] = {"installed": False}


def parse_tactics(spec: str) -> list[tuple[int, int]]:
    """``"64,2;128,2"`` -> ``[(64, 2), (128, 2)]``; validated."""
    out = []
    for t in spec.split(";"):
        if not t.strip():
            continue
        tile_n, s = (int(x) for x in t.split(","))
        if tile_n not in SUPPORTED_TILE_N or s not in SUPPORTED_SPLIT_K:
            raise ValueError(f"unsupported mid-M split-K tactic {t!r}")
        out.append((tile_n, s))
    return out


def encode_tactic(tile_n: int, split_k: int) -> tuple[int, int, int]:
    return (TACTIC_TAG, tile_n, split_k)


def is_splitk_tactic(tactic: Any) -> bool:
    return (
        isinstance(tactic, (tuple, list))
        and len(tactic) == 3
        and tactic[0] == TACTIC_TAG
    )


@functools.cache
def _max_active_clusters(cluster_size: int) -> int:
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    return int(get_max_active_clusters(cluster_size))


def offer(m: int, n: int, k: int, tile_n: int, split_k: int, max_m: int) -> bool:
    """Whether the mid-M split-K tactic is offered for an (m, n, k) problem."""
    if not (32 < m <= max_m):
        return False
    if k % (TILE_K * split_k) != 0 or n % 8 != 0:
        return False
    if tile_n not in SUPPORTED_TILE_N or split_k not in SUPPORTED_SPLIT_K:
        return False
    clusters = -(-n // TILE_M) * -(-m // tile_n)
    return clusters <= _max_active_clusters(split_k)


def cluster_ctas(tactic: Any) -> int:
    """CTAs per cluster of a stock persistent tactic (1 if not one)."""
    if (
        isinstance(tactic, (tuple, list))
        and len(tactic) == 5
        and isinstance(tactic[1], (tuple, list))
    ):
        return int(tactic[1][0]) * int(tactic[1][1])
    return 1


@functools.cache
def _kernel_cls():
    from vllm.lcd_pdl.fi_dense_blockscaled_gemm_sm100_splitk_midm import (
        Sm100BlockScaledSplitKMidMGemmKernel,
    )

    return Sm100BlockScaledSplitKMidMGemmKernel


def run(gb, inputs, tile_n: int, split_k: int, enable_pdl: bool):
    """Launch the mid-M split-K kernel on mm_mxfp8 runner inputs."""
    import cutlass
    import torch

    a, b, a_descale, b_descale, _, out, _ = inputs
    m, real_k = a.shape
    n = b.shape[1]
    if not out.is_contiguous() or real_k % (TILE_K * split_k) != 0:
        raise ValueError(
            f"invalid mid-M split-K tactic {(tile_n, split_k)} for "
            f"M={m} N={n} K={real_k} (out contiguous={out.is_contiguous()})"
        )
    out_dtype = out.dtype
    c_dtype = cutlass.BFloat16 if out_dtype == torch.bfloat16 else cutlass.Float16
    sf_vec_size = 32
    # swap-AB: kernel M = weight rows (N), kernel N = tokens (M).
    sf_m = (n + 127) // 128
    sf_n = (m + 127) // 128
    sf_k = (real_k // sf_vec_size + 3) // 4
    mma_tiler_mn = (TILE_M, tile_n)
    cache_key = (
        "midm",
        sf_vec_size,
        mma_tiler_mn,
        (1, 1),
        True,
        False,
        enable_pdl,
        out_dtype,
        split_k,
    )
    kcls = _kernel_cls()
    compiled, _ = gb._compile_block_scaled_gemm(
        _KERNEL_CACHE,
        cache_key,
        lambda: kcls(sf_vec_size, mma_tiler_mn, split_k, enable_pdl),
        ab_cutlass_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        c_cutlass_dtype=c_dtype,
        ab_assumed_align=16,
        cluster_shape_mn=(1, 1),
        swap_ab=True,
        sf_m=sf_m,
        sf_n=sf_n,
        sf_k=sf_k,
        batch_size=1,
        cluster_shape_k=split_k,
    )
    alpha = gb._prepare_alpha_for_launch(None, a.device)
    compiled(
        b.T,
        a,
        out.as_strided(out.shape, (1, out.shape[0])),
        sf_m,
        sf_n,
        sf_k,
        b_descale.data_ptr(),
        a_descale.data_ptr(),
        alpha,
    )
    STATS["calls"] += 1
    return out


def install(
    gb,
    tactics: list[tuple[int, int]],
    max_m: int,
    max_cluster_ctas: int = 0,
    cluster_min_m: int = 513,
) -> None:
    """Wrap ``gb._cute_dsl_gemm_mxfp8_runner`` (``flashinfer.gemm.gemm_base``)."""
    import torch

    orig_factory = gb._cute_dsl_gemm_mxfp8_runner
    cache: dict = {}

    def factory(sm_major, sm_minor, enable_pdl, out_dtype):
        key = (sm_major, sm_minor, enable_pdl, out_dtype)
        if key in cache:
            return cache[key]
        base = orig_factory(sm_major, sm_minor, enable_pdl, out_dtype)
        if out_dtype not in (torch.bfloat16, torch.float16):
            return base
        Base = type(base)

        class Runner(Base):
            def get_valid_tactics(self, inputs, profile):
                out = list(Base.get_valid_tactics(self, inputs, profile))
                a, b = inputs[0], inputs[1]
                m, k, n = a.shape[0], a.shape[1], b.shape[1]
                if max_cluster_ctas > 0 and m >= cluster_min_m:
                    kept = [t for t in out if cluster_ctas(t) <= max_cluster_ctas]
                    if kept:
                        STATS["cluster_dropped"] += len(out) - len(kept)
                        out = kept
                if inputs[5].is_contiguous():
                    for tile_n, s in tactics:
                        if offer(m, n, k, tile_n, s, max_m):
                            out.append(encode_tactic(tile_n, s))
                            STATS["tactic_offers"] += 1
                return out

            def forward(self, inputs, tactic=None, do_preparation=False, **kwargs):
                if is_splitk_tactic(tactic):
                    return run(gb, inputs, int(tactic[1]), int(tactic[2]), enable_pdl)
                return Base.forward(
                    self, inputs, tactic=tactic, do_preparation=do_preparation, **kwargs
                )

        # Keep the autotune cache/file key (runner class name) unchanged.
        Runner.__name__ = Base.__name__
        Runner.__qualname__ = Base.__qualname__
        r = Runner.__new__(Runner)
        r.__dict__.update(base.__dict__)
        cache[key] = r
        return r

    gb._cute_dsl_gemm_mxfp8_runner = factory


def maybe_install() -> bool:
    """Install the mid-M split-K tactics / cluster cap once per process.

    Only on CUDA SM10x when ``VLLM_MXFP8_SPLITK_MIDM`` is on or
    ``VLLM_MXFP8_MAX_CLUSTER_CTAS`` > 0. Returns whether installed. Never
    raises: any failure is logged and the stock tactics stay in use.
    """
    if _STATE["installed"]:
        return True
    midm = envs.VLLM_MXFP8_SPLITK_MIDM
    max_cluster_ctas = int(envs.VLLM_MXFP8_MAX_CLUSTER_CTAS)
    if not midm and max_cluster_ctas <= 0:
        return False
    from vllm.platforms import current_platform

    if not (current_platform.is_cuda() and current_platform.is_device_capability_family(100)):
        return False
    with _LOCK:
        if _STATE["installed"]:
            return True
        try:
            import flashinfer.gemm.gemm_base as gb

            tactics = parse_tactics(envs.VLLM_MXFP8_SPLITK_MIDM_TACTICS) if midm else []
            max_m = int(envs.VLLM_MXFP8_SPLITK_MIDM_MAX_M)
            cluster_min_m = int(envs.VLLM_MXFP8_MAX_CLUSTER_CTAS_MIN_M)
            install(gb, tactics, max_m, max_cluster_ctas, cluster_min_m)
        except Exception as e:
            logger.warning("FlashInfer mid-M split-K MXFP8 tactics not installed: %r", e)
            return False
        _STATE["installed"] = True
    logger.info(
        "FlashInfer CuTe-DSL MXFP8: mid-M split-K tactics %s (32 < M <= %d); "
        "max cluster CTAs %s at M >= %d",
        tactics,
        max_m,
        max_cluster_ctas or "unlimited",
        cluster_min_m,
    )
    return True
