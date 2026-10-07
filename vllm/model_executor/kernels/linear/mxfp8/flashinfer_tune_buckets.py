# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extra FlashInfer CuTe-DSL ``mm_mxfp8`` autotune buckets (VLLM_MXFP8_TUNE_BUCKETS).

FlashInfer tunes ``mm_mxfp8`` at hybrid M buckets (1, 2, 4, ..., 256, then 512,
768, ... in steps of 256 up to 2048, ...) and rounds a runtime M *up* to the next
bucket (``map_to_hybrid_bucket_uncapped``). C512 decode batches (M = 272..320
after CUDA-graph padding) therefore run the tactic profiled at M = 512, e.g. a
(256, 256) cluster-(2, 1) tile for qkvz that wastes most of its second wave.

With ``VLLM_MXFP8_TUNE_BUCKETS`` set, this module

1. replaces the dynamic-M spec of FlashInfer's cute-dsl ``mm_mxfp8`` tuning
   config (``gemm_base._MM_MXFP8_CUTE_DSL_TUNING_CONFIG``, read at call time) so
   that the extra buckets join the hybrid ones: a runtime M maps to the smallest
   bucket >= M of the union (``install``), and
2. records each ``FlashInferCutedslMxfp8LinearKernel`` weight shape
   (``maybe_install``, from ``process_weights_after_loading``) so the kernel
   warmup can profile those shapes at the extra buckets before CUDA-graph
   capture (``autotune_extra_buckets``, from ``flashinfer_autotune``). The
   warmup's dummy runs pass the hybrid list as an explicit ``tuning_buckets``
   override, so without this pass the extra buckets would never be profiled and
   every M that maps to one would fall back to tactic -1.

Only the cute-dsl-only config is replaced; the shared ``_MM_MXFP8_TUNING_CONFIG``
(CUTLASS / cuDNN / TRT-LLM / auto backends) is untouched.

Numerics: the dense persistent tactics (tile, cluster, swap-AB, prefetch) reduce
each output element over K in the same order, so a tactic change is bitwise for
M > 32: on VR all 5,753 (shape, M in 256..512, tactic) runs of the Qwen3.6 MXFP8
linears matched the bucket-512 tactic exactly. Split-K tactics (offered only for
M <= 32) change the order: buckets <= 32 can change numerics.

Grammar (comma-separated items, union of all):
  ``N``               one bucket, e.g. ``288``
  ``LO-HI:STEP``      ``LO, LO+STEP, ... <= HI``, e.g. ``272-512:16``
  ``capture[:LO-HI]`` the CUDA-graph capture sizes, optionally within
                      ``[LO, HI]``, e.g. ``capture:257-512``
Buckets above ``max_num_batched_tokens`` are dropped (never reached).
"""

from __future__ import annotations

import bisect
import dataclasses
import functools
import weakref
from collections.abc import Iterable, Sequence

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# (N, K, out_dtype) -> weak reference to one layer holding that weight.
_SHAPES: dict[tuple[int, int, torch.dtype], weakref.ReferenceType] = {}
# The FlashInfer config object before install(), and the installed buckets.
_ORIG_CONFIG = None
_INSTALLED: tuple[int, ...] | None = None


def parse_tune_buckets(
    spec: str | None, capture_sizes: Iterable[int] = (), max_m: int | None = None
) -> tuple[int, ...]:
    """Parse the VLLM_MXFP8_TUNE_BUCKETS grammar into sorted unique buckets."""
    if spec is None or not spec.strip():
        return ()
    out: set[int] = set()
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            if item == "capture" or item.startswith("capture:"):
                lo, hi = 1, None
                if item != "capture":
                    lo_s, hi_s = item.split(":", 1)[1].split("-")
                    lo, hi = int(lo_s), int(hi_s)
                out.update(
                    int(s) for s in capture_sizes if s >= lo and (hi is None or s <= hi)
                )
            elif "-" in item:
                rng, step_s = item.split(":")
                lo_s, hi_s = rng.split("-")
                lo, hi, step = int(lo_s), int(hi_s), int(step_s)
                if step <= 0 or lo > hi:
                    raise ValueError
                out.update(range(lo, hi + 1, step))
            else:
                out.add(int(item))
        except ValueError:
            raise ValueError(
                f"VLLM_MXFP8_TUNE_BUCKETS: cannot parse {item!r} in {spec!r}; "
                "expected N, LO-HI:STEP or capture[:LO-HI]"
            ) from None
    if any(b <= 0 for b in out):
        raise ValueError(f"VLLM_MXFP8_TUNE_BUCKETS: buckets must be > 0: {spec!r}")
    if max_m is not None:
        out = {b for b in out if b <= max_m}
    return tuple(sorted(out))


def map_to_bucket(x: int, extras: tuple[int, ...]) -> int:
    """Smallest bucket >= x in hybrid(x) U extras (round up, as FlashInfer)."""
    from flashinfer.fused_moe.utils import map_to_hybrid_bucket_uncapped

    hybrid = map_to_hybrid_bucket_uncapped(x)
    i = bisect.bisect_left(extras, x)
    return min(hybrid, extras[i]) if i < len(extras) else hybrid


def gen_buckets(max_m: int, extras: tuple[int, ...]) -> tuple[int, ...]:
    """Hybrid buckets up to max_m plus the extras <= max_m."""
    from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

    return tuple(
        sorted(
            set(get_hybrid_num_tokens_buckets(max_m))
            | {b for b in extras if b <= max_m}
        )
    )


@functools.cache
def _bucket_fns(extras: tuple[int, ...]):
    # One stable (gen, map) pair per bucket list: AutoTuner's nearest-profile
    # LRU cache is keyed on the spec, i.e. on these callables' identity.
    return (
        functools.partial(_gen, extras=extras),
        functools.partial(_map, extras=extras),
    )


def _gen(max_m: int, *, extras: tuple[int, ...]) -> tuple[int, ...]:
    return gen_buckets(max_m, extras)


def _map(x: int, *, extras: tuple[int, ...]) -> int:
    return map_to_bucket(x, extras)


def install(extras: Sequence[int]) -> bool:
    """Make FlashInfer's cute-dsl mm_mxfp8 bucketing include ``extras``.

    Idempotent for the same list. Returns False (and changes nothing) when the
    FlashInfer config does not have the expected hybrid spec.
    """
    global _ORIG_CONFIG, _INSTALLED
    extras = tuple(sorted(set(extras)))
    if not extras:
        return False
    if _INSTALLED is not None:
        if extras != _INSTALLED:
            raise RuntimeError(
                f"mm_mxfp8 tune buckets already installed as {_INSTALLED}, "
                f"refusing {extras}"
            )
        return True
    import flashinfer.gemm.gemm_base as gemm_base
    from flashinfer.autotuner import DynamicTensorSpec
    from flashinfer.fused_moe.utils import (
        get_hybrid_num_tokens_buckets,
        map_to_hybrid_bucket_uncapped,
    )

    cfg = getattr(gemm_base, "_MM_MXFP8_CUTE_DSL_TUNING_CONFIG", None)
    specs = getattr(cfg, "dynamic_tensor_specs", ())
    if (
        len(specs) != 1
        or specs[0].input_idx != (0,)
        or specs[0].dim_idx != (0,)
        or specs[0].gen_tuning_buckets is not get_hybrid_num_tokens_buckets
        or specs[0].map_to_tuning_buckets is not map_to_hybrid_bucket_uncapped
    ):
        logger.warning(
            "VLLM_MXFP8_TUNE_BUCKETS ignored: this FlashInfer's cute-dsl "
            "mm_mxfp8 tuning config is not the expected hybrid M spec (%s).",
            specs,
        )
        return False
    gen, map_fn = _bucket_fns(extras)
    new_spec = DynamicTensorSpec(
        input_idx=specs[0].input_idx,
        dim_idx=specs[0].dim_idx,
        gen_tuning_buckets=gen,
        map_to_tuning_buckets=map_fn,
    )
    _ORIG_CONFIG = cfg
    gemm_base._MM_MXFP8_CUTE_DSL_TUNING_CONFIG = dataclasses.replace(
        cfg, dynamic_tensor_specs=(new_spec,)
    )
    _INSTALLED = extras
    logger.info("FlashInfer cute-dsl mm_mxfp8 extra tune buckets: %s", extras)
    return True


def uninstall() -> None:
    """Restore FlashInfer's config (tests)."""
    global _ORIG_CONFIG, _INSTALLED
    if _ORIG_CONFIG is not None:
        import flashinfer.gemm.gemm_base as gemm_base

        gemm_base._MM_MXFP8_CUTE_DSL_TUNING_CONFIG = _ORIG_CONFIG
    _ORIG_CONFIG = None
    _INSTALLED = None
    _SHAPES.clear()


def installed_buckets() -> tuple[int, ...] | None:
    return _INSTALLED


@functools.cache
def _resolve_from_env() -> tuple[int, ...]:
    import vllm.envs as envs

    spec = envs.VLLM_MXFP8_TUNE_BUCKETS
    if not spec:
        return ()
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    capture: list[int] = []
    max_m = None
    if cfg is not None:
        capture = list(cfg.compilation_config.cudagraph_capture_sizes or ())
        max_m = cfg.scheduler_config.max_num_batched_tokens
    if "capture" in spec and not capture:
        logger.warning(
            "VLLM_MXFP8_TUNE_BUCKETS=%s: no CUDA-graph capture sizes known here.",
            spec,
        )
    return parse_tune_buckets(spec, capture, max_m)


def maybe_install(layer: torch.nn.Module, out_dtype: torch.dtype) -> None:
    """From process_weights_after_loading (weight already [K, N] col-major):
    install the extra buckets once and record the layer's GEMM shape.
    """
    extras = _resolve_from_env()
    if not extras or not install(extras):
        return
    k, n = layer.weight.shape
    key = (n, k, out_dtype)
    ref = _SHAPES.get(key)
    if ref is None or ref() is None:
        _SHAPES[key] = weakref.ref(layer)


@torch.inference_mode()
def autotune_extra_buckets(max_num_tokens: int) -> list[tuple[int, int]]:
    """Profile every recorded mm_mxfp8 shape at the extra buckets.

    Call inside the warmup's ``autotune(tune_mode=True)`` context, before
    CUDA-graph capture; every rank runs the same calls in the same order (the
    distributed tuner all-reduces per-tactic timings).
    """
    if _INSTALLED is None:
        return []
    buckets = tuple(b for b in _INSTALLED if b <= max_num_tokens)
    if not buckets:
        return []
    import vllm.utils.flashinfer as fi_utils
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    m = buckets[-1]
    tuned: list[tuple[int, int]] = []
    for (n, k, out_dtype), ref in sorted(_SHAPES.items(), key=lambda kv: kv[0][:2]):
        layer = ref()
        if layer is None:
            continue
        w, w_scale = layer.weight, layer.weight_scale
        x = torch.randn(m, k, device=w.device, dtype=out_dtype)
        x_q, x_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        # Explicit bucket list: profile exactly the extras (the hybrid ones
        # were tuned by the dummy runs).
        with fi_utils.autotune(tuning_buckets=buckets):
            fi_utils.mm_mxfp8(x_q, w, x_scale, w_scale, out_dtype, backend="cute-dsl")
        tuned.append((n, k))
    logger.info(
        "Autotuned FlashInfer cute-dsl mm_mxfp8 at extra buckets %s for (N, K) %s.",
        buckets,
        tuned,
    )
    return tuned
