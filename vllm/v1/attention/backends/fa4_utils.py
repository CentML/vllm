# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from importlib.util import find_spec
from typing import Optional, Tuple

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# flash_attn.cute.interface (Cute-DSL / FA4).
#
# NOTE: vLLM currently only enables this path for **Blackwell** GPUs
# (compute capability 10.x) and only for ViT/MM encoder attention.
# It is NOT a KV-cache attention backend.
_OPTIMIZED_HEAD_SIZES: tuple[int, ...] = (64, 96, 128, 192)


def warn_if_unoptimized_head_size(head_size: int) -> None:
    """Warn if `head_size` is outside the known-optimized set.

    We intentionally don't hard-block on head_size here, since upstream support
    may evolve and some shapes may still work (albeit slower).
    """
    if head_size not in _OPTIMIZED_HEAD_SIZES:
        logger.warning_once(
            "FA4 (flash_attn.cute) selected for head_size=%d, which is not in the "
            "known-optimized set %s. The kernel may be slower or unsupported.",
            head_size,
            _OPTIMIZED_HEAD_SIZES,
        )


def supports_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16)


def supports_device() -> bool:
    if not current_platform.is_cuda():
        return False
    cc = current_platform.get_device_capability()
    return cc is not None and cc.major == 10


def is_flash_attn_cute_available() -> bool:
    """Best-effort availability check for FA4 (flash_attn.cute).

    This intentionally avoids importing `flash_attn.cute.interface` because
    that may pull in heavy deps (cutlass-dsl / cuda-python). The actual import
    happens in `flash_attn_varlen_func`.
    """
    if not supports_device():
        return False
    return find_spec("flash_attn.cute.interface") is not None


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    deterministic: bool = False,
) -> torch.Tensor:
    """FA4 (Cute-DSL) FlashAttention varlen forward.

    Wraps `flash_attn.cute.interface.flash_attn_varlen_func`, which returns
    `(out, lse)`. vLLM only needs `out` for inference.
    """
    if not current_platform.is_cuda():
        raise RuntimeError("FA4 (flash_attn.cute) is only supported on CUDA.")

    try:
        from flash_attn.cute.interface import flash_attn_varlen_func as _fa4_varlen
    except Exception as e:
        raise ImportError(
            "FA4 (flash_attn.cute) is not available. "
            "Please ensure the Cute-DSL FlashAttention build is installed "
            "(e.g. nvidia-cutlass-dsl) and cuda-python bindings are present."
        ) from e

    out, _lse = _fa4_varlen(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
    )
    return out

