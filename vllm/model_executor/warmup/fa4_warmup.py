# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup FA4 (flash_attn.cute) kernels for ViT/MM encoder attention.

We specifically warm up the FlashAttention Cute-DSL (FA4) compile cache by
running a few representative varlen attention calls that differ only in
sequence length. This helps avoid JIT compilation in the hot path.

This warmup is:
- Blackwell-only (compute capability 10.x)
- Opt-in (only when mm_encoder_attn_backend == FLASH_ATTN_CUTE)
- Scoped to Qwen3-VL / Qwen3-VL-MoE vision transformer workloads
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _get_default_qwen3_vit_warmup_seqlens(
    max_positions: int | None = None,
) -> list[int]:
    candidates = [
        16**2,  # 256
        24**2,  # 576
        32**2,  # 1024
        48**2,  # 2304
        64**2,  # 4096
        96**2,  # 9216
        128**2,  # 16384
        192**2,  # 36864
        256**2,  # 65536
    ]
    if max_positions is None:
        return candidates
    return [s for s in candidates if s <= max_positions]


def should_fa4_vit_warmup(worker: Worker) -> bool:
    """Fast predicate used by `kernel_warmup` to gate FA4 warmup."""
    if not current_platform.is_cuda():
        return False
    cc = current_platform.get_device_capability()
    if cc is None or cc.major != 10:
        return False

    mm_cfg = getattr(worker.model_config, "multimodal_config", None)
    return (
        mm_cfg is not None
        and mm_cfg.mm_encoder_attn_backend == AttentionBackendEnum.FLASH_ATTN_CUTE
    )


def fa4_vit_warmup(worker: Worker) -> None:
    """Warm up FA4 kernels for Qwen3-VL(-MoE) ViT attention."""

    # Config gating: only warm up when explicitly selected for mm encoder.
    if not should_fa4_vit_warmup(worker):
        return

    # Dependency gating.
    from vllm.v1.attention.backends.fa4_utils import (
        is_flash_attn_cute_available,
        supports_dtype,
        warn_if_unoptimized_head_size,
    )

    if not is_flash_attn_cute_available():
        logger.warning(
            "Skipping FA4 warmup: `flash_attn.cute.interface` is not available."
        )
        return

    model = worker.get_model()
    visual = getattr(model, "visual", None)
    if visual is None:
        # Not a Qwen3-VL(-MoE) style model, or vision tower disabled.
        logger.warning("Skipping FA4 warmup: vision tower disabled or not found.")
        return

    # Derive head shape and dtype from the actual vision attention module.
    try:
        first_attn = visual.blocks[0].attn  # Qwen2_5_VisionAttention
        head_size = int(first_attn.hidden_size_per_attention_head)
        num_heads = int(first_attn.num_attention_heads_per_partition)
        scale = float(first_attn.hidden_size_per_attention_head**-0.5)
        dtype = visual.dtype
    except Exception:
        # If the model structure is unexpected, skip warmup.
        return

    if not supports_dtype(dtype):
        # If dtype is not supported, the FA4 backend should not have been selected.
        logger.warning_once(
            "Skipping FA4 warmup: dtype %s is not supported by flash_attn.cute.",
            dtype,
        )
        return

    warn_if_unoptimized_head_size(head_size)

    seqlens = tuple(_get_default_qwen3_vit_warmup_seqlens())

    logger.info_once(
        "Warming up FA4 (flash_attn.cute) ViT kernels for seqlens=%s "
        "(head_size=%d, num_heads=%d, dtype=%s).",
        seqlens,
        head_size,
        num_heads,
        dtype,
    )

    # Run a small number of representative calls that only vary seqlen.
    # Compilation key can be found under `flash_attn/cute/interface.py`.
    from vllm.v1.attention.backends.fa4_utils import flash_attn_varlen_func

    device = torch.device("cuda")
    with torch.inference_mode():
        for seqlen in seqlens:
            q = torch.empty((seqlen, num_heads, head_size), device=device, dtype=dtype)
            k = torch.empty_like(q)
            v = torch.empty_like(q)
            cu = torch.tensor([0, seqlen], device=device, dtype=torch.int32)

            # This call will populate FA4's internal compile cache (Cute-DSL).
            _ = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=seqlen,
                max_seqlen_k=seqlen,
                softmax_scale=scale,
                causal=False,
            )
