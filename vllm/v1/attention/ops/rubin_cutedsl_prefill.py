# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in dispatch to FlashInfer's published Rubin paged FMHA kernels."""

import functools
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_LEGACY_KERNEL_OPTIONS: Mapping[str, Any] = MappingProxyType({})


@functools.cache
def _tuned_prefill_policy() -> (
    tuple[Callable[..., tuple[str, str]], dict[str, Any]] | None
):
    """FlashInfer's query-layout selector and tuned kernel policy, or None for
    the legacy export (the default, or FlashInfer without the tuned API)."""
    if envs.VLLM_RUBIN_CUTEDSL_PREFILL_POLICY != "tuned":
        logger.info("Rubin CuTeDSL paged prefill policy: legacy (packed query).")
        return None
    try:
        from flashinfer.attention.cute_dsl import (
            RUBIN_GQA_TUNED_POLICY,
            rubin_paged_prefill_query_layout,
        )
    except ImportError:
        logger.warning(
            "VLLM_RUBIN_CUTEDSL_PREFILL_POLICY=tuned needs FlashInfer with "
            "rubin_paged_prefill_query_layout and RUBIN_GQA_TUNED_POLICY; "
            "using the legacy Rubin CuTeDSL paged prefill export."
        )
        return None
    policy = dict(RUBIN_GQA_TUNED_POLICY)
    logger.info(
        "Rubin CuTeDSL paged prefill policy: tuned %s, per-batch query layout.",
        policy,
    )
    return rubin_paged_prefill_query_layout, policy


def _prefill_kernel_options(
    query_lens_cpu: tuple[int, ...] | None, fp4_output: bool
) -> Mapping[str, Any]:
    """Extra cute_dsl_fmha_paged_prefill kwargs; none for the legacy policy."""
    if _tuned_prefill_policy() is None:
        return _LEGACY_KERNEL_OPTIONS
    return _tuned_kernel_options(query_lens_cpu, fp4_output)


@functools.lru_cache(maxsize=64)
def _tuned_kernel_options(
    query_lens_cpu: tuple[int, ...] | None, fp4_output: bool
) -> Mapping[str, Any]:
    """Tuned policy plus the batch's query layout, cached because every layer
    of a step passes the same lengths.

    The layout changes the launch geometry per batch. That is safe because
    prefill attention never runs inside a captured CUDA graph: FlashInfer
    reports at most UNIFORM_BATCH support, so FULL graphs hold decode-only
    batches and batches with prefills run PIECEWISE (attention eager) or eager.
    """
    tuned = _tuned_prefill_policy()
    assert tuned is not None
    select_query_layout, policy = tuned
    options = dict(policy)
    if query_lens_cpu and min(query_lens_cpu) > 0:
        query_layout, output_store = select_query_layout(
            query_lens_cpu, fp4_output=fp4_output
        )
        options.update(query_layout=query_layout, output_store=output_store)
    return MappingProxyType(options)


def try_rubin_cutedsl_prefill(
    *,
    query: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    output: torch.Tensor,
    output_block_scale: torch.Tensor | None,
    o_sf_scale: float | None,
    o_sf_start_index: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: float,
    bmm2_scale: float,
    causal: bool,
    window_left: int,
    sinks: torch.Tensor | None,
    logits_soft_cap: float | None,
    query_lens_cpu: tuple[int, ...] | None = None,
) -> bool:
    """Run compatible HND FP8 prefill, otherwise retain the existing backend.

    ``query_lens_cpu`` holds the host query length of each request; with
    VLLM_RUBIN_CUTEDSL_PREFILL_POLICY=tuned it selects the query layout.
    Missing artifacts or ABI failures are intentionally not caught: an opted-in
    compatible request must not silently claim to use CuTeDSL while falling back.
    """
    if not isinstance(kv_cache, tuple) or len(kv_cache) != 2:
        logger.warning_once(
            "Rubin CuTeDSL prefill requires separate FP8 HND K/V views; "
            "keeping the existing backend for this KV representation."
        )
        return False
    k_cache, v_cache = kv_cache
    supported = (
        current_platform.is_cuda()
        and current_platform.is_device_capability(107)
        and query.ndim == 3
        and query.shape[-1] == 128
        and query.dtype == torch.float8_e4m3fn
        and k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn
        and k_cache.ndim == v_cache.ndim == 4
        and k_cache.shape[2] in (16, 64, 128)
        and output.dtype in (torch.bfloat16, torch.uint8)
        and (output.dtype != torch.uint8 or query.shape[1] == 64)
        and causal
        and window_left == -1
        and sinks is None
        and not logits_soft_cap
    )
    if not supported:
        logger.warning_once(
            "Rubin CuTeDSL prefill not selected for this configuration; "
            "using the existing FlashInfer prefill backend. Requires SM107, "
            "FP8 D128 HND KV, page16/64/128, causal full attention, BF16 "
            "or NVFP4 output (64 query heads for NVFP4)."
        )
        return False

    from flashinfer.attention.cute_dsl import cute_dsl_fmha_paged_prefill

    fp4_output = output.dtype == torch.uint8
    cute_dsl_fmha_paged_prefill(
        query,
        k_cache,
        v_cache,
        output,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        block_tables,
        seq_lens,
        max_q_len,
        max_kv_len,
        kv_layout="HND",
        sm_scale=bmm1_scale,
        scale_v=bmm2_scale,
        output_block_scale=output_block_scale,
        o_sf_scale=o_sf_scale,
        o_sf_start_index=o_sf_start_index if fp4_output else 0,
        use_fp16_softmax=True,
        **_prefill_kernel_options(query_lens_cpu, fp4_output),
    )
    logger.info_once(
        "Rubin CuTeDSL paged prefill active: FP8 HND KV, %s output, "
        "mixed-FP16 softmax.",
        "fused NVFP4/Layout128x4" if fp4_output else "BF16",
    )
    return True
