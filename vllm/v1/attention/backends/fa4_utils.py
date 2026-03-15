# SPDX-License-Identifier: Apache-2.0
"""Utilities for Flash Attention 4 (flash_attn.cute) on Blackwell."""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_FA4_AVAILABLE: bool | None = None
_FA4_FUNC = None

# Head sizes optimized for FA4 on Blackwell
FA4_SUPPORTED_HEAD_SIZES = (64, 96, 128, 192)


def _import_fa4_fwd():
    """Try importing FA4. Prefer flash_attn_cute to avoid polluting the
    flash_attn namespace which would break vllm's flash_attn.ops imports."""
    try:
        from flash_attn_cute.interface import _flash_attn_fwd
        return _flash_attn_fwd
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
        return _flash_attn_fwd
    except (ImportError, ModuleNotFoundError):
        pass
    return None


def is_flash_attn_cute_available() -> bool:
    global _FA4_AVAILABLE
    if _FA4_AVAILABLE is not None:
        return _FA4_AVAILABLE
    _FA4_AVAILABLE = _import_fa4_fwd() is not None
    return _FA4_AVAILABLE


def _get_fa4_func():
    global _FA4_FUNC
    if _FA4_FUNC is None:
        _FA4_FUNC = _import_fa4_fwd()
        if _FA4_FUNC is None:
            raise ImportError(
                "flash_attn.cute is not available. "
                "Install flash-attn-4 for Blackwell FA4 support."
            )
    return _FA4_FUNC


def flash_attn_cute_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Wrapper around flash_attn.cute for varlen (variable-length) attention."""
    fa4_fwd = _get_fa4_func()

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    result = fa4_fwd(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=causal,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    # _flash_attn_fwd returns (output, softmax_lse); we only need output
    if isinstance(result, tuple):
        return result[0]
    return result
