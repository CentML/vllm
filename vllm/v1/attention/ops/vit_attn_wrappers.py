# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains ops for ViT attention to be compatible with torch.compile
as there are operations here not supported by torch.compile (for instance,
`.item()` in flash attention)

Using these ops and wrapping vision blocks with `torch.compile` can speed up
throughput in vision models by ~5% relative on H100, and improve token
latencies by ~7% (see qwen2_5_vl for example usage)

To use these ops, you must have a recent version of PyTorch installed (>= 2.4.0)
"""

import einops
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def flash_attn_maxseqlen_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs = {}
    if is_rocm_aiter:
        from aiter import flash_attn_varlen_func
    else:
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        if not current_platform.is_rocm() and fa_version is not None:
            kwargs["fa_version"] = fa_version

    q_len = q.size(1)
    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=q.device
        )
    max_seqlen = q_len if max_seqlen is None else max_seqlen.item()

    q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        causal=False,
        softmax_scale=scale,
        **kwargs,
    )
    context_layer = einops.rearrange(output, "(b s) h d -> b s h d", b=batch_size)
    return context_layer


def flash_attn_maxseqlen_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


direct_register_custom_op(
    op_name="flash_attn_maxseqlen_wrapper",
    op_func=flash_attn_maxseqlen_wrapper,
    fake_impl=flash_attn_maxseqlen_wrapper_fake,
)


def vit_flash_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.flash_attn_maxseqlen_wrapper(
        q,
        k,
        v,
        batch_size,
        is_rocm_aiter,
        fa_version,
        scale,
        cu_seqlens,
        max_seqlen,
    )


def apply_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Input shape:
    (batch_size x seq_len x num_heads x head_size)
    """
    q, k, v = (einops.rearrange(x, "b s h d -> b h s d") for x in [q, k, v])
    output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)
    output = einops.rearrange(output, "b h s d -> b s h d ")
    return output


# TODO: Once we have a torch 2.10, we can use tensor slices
# so we won't need to wrap this in custom ops
def torch_sdpa_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    # Never remove the contiguous logic for ROCm
    # Without it, hallucinations occur with the backend
    if current_platform.is_rocm():
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    if cu_seqlens is None:
        return apply_sdpa(q, k, v, scale=scale)

    outputs = []

    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    q_chunks = torch.split(q, lens, dim=1)
    k_chunks = torch.split(k, lens, dim=1)
    v_chunks = torch.split(v, lens, dim=1)
    for q_i, k_i, v_i in zip(q_chunks, k_chunks, v_chunks):
        output_i = apply_sdpa(q_i, k_i, v_i, scale=scale)
        outputs.append(output_i)
    context_layer = torch.cat(outputs, dim=1)
    return context_layer


def torch_sdpa_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    return torch.empty_like(q)


direct_register_custom_op(
    op_name="torch_sdpa_wrapper",
    op_func=torch_sdpa_wrapper,
    fake_impl=torch_sdpa_wrapper_fake,
)


def vit_torch_sdpa_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.torch_sdpa_wrapper(q, k, v, scale, cu_seqlens)


# Batch buckets for cuDNN graph caching - graphs are cached per bucket size
# This avoids creating a new graph for each unique batch size at runtime
BATCH_BUCKETS = [8, 16, 32, 64]


def _pad_to_batch_bucket(
    batch_size: int,
    actual_seq_lens: torch.Tensor,
    batch_offsets_q: torch.Tensor,
    batch_offsets_k: torch.Tensor,
    batch_offsets_v: torch.Tensor,
    batch_offsets_o: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad actual_seq_lens and batch_offsets to match the nearest batch bucket.

    This follows the same padding strategy as cuDNN frontend's SDPA caching:
    https://github.com/NVIDIA/cudnn-frontend/blob/main/test/python/test_sdpa_with_caching.py

    Args:
        batch_size: Actual batch size
        actual_seq_lens: Tensor of shape (batch_size, 1, 1, 1) with sequence lengths
        batch_offsets_q: Tensor of shape (batch_size + 1, 1, 1, 1) with offsets for q
        batch_offsets_k: Tensor of shape (batch_size + 1, 1, 1, 1) with offsets for k
        batch_offsets_v: Tensor of shape (batch_size + 1, 1, 1, 1) with offsets for v
        batch_offsets_o: Tensor of shape (batch_size + 1, 1, 1, 1) with offsets for output

    Returns:
        Tuple of (padded_batch_size, padded_actual_seq_lens,
                  padded_batch_offsets_q, padded_batch_offsets_k,
                  padded_batch_offsets_v, padded_batch_offsets_o)
    """
    # Find the nearest bucket size >= actual batch size
    batch_size_padded = next(
        (b for b in BATCH_BUCKETS if b >= batch_size), BATCH_BUCKETS[-1]
    )

    if batch_size_padded == batch_size:
        return (batch_size, actual_seq_lens,
                batch_offsets_q, batch_offsets_k, batch_offsets_v, batch_offsets_o)

    pad_size = batch_size_padded - batch_size
    device = actual_seq_lens.device

    # Pad actual_seq_lens with zeros (padded sequences have length 0)
    zeros_seq_lens = torch.zeros(
        (pad_size, 1, 1, 1),
        dtype=actual_seq_lens.dtype,
        device=device,
    )
    actual_seq_lens_padded = torch.cat([actual_seq_lens, zeros_seq_lens], dim=0)

    # Pad each batch_offsets by repeating the last offset value
    # The last offset is the end of the last real sequence, and padded
    # sequences (with length 0) should all start at that same position
    # Note: batch_offsets has shape (batch_size + 1, 1, 1, 1)
    def pad_offsets(offsets: torch.Tensor) -> torch.Tensor:
        last_offset = offsets[-1:].expand(pad_size, 1, 1, 1)
        return torch.cat([offsets, last_offset], dim=0)

    return (
        batch_size_padded,
        actual_seq_lens_padded,
        pad_offsets(batch_offsets_q),
        pad_offsets(batch_offsets_k),
        pad_offsets(batch_offsets_v),
        pad_offsets(batch_offsets_o),
    )


def flashinfer_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    act_seq_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm.v1.attention.backends.flashinfer import cudnn_batch_prefill_with_kv_cache

    is_reshaped = q.dim() == 4
    batch_size = q.shape[0] if is_reshaped else act_seq_lens.shape[0]

    if is_reshaped:
        q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

    q_len = q.size(0)
    max_seqlen = q_len if max_seqlen is None else max_seqlen.item()

    # Compute cumulative token offsets from act_seq_lens
    # act_seq_lens contains token counts per sequence (not scaled)
    token_offsets = torch.zeros(
        act_seq_lens.shape[0] + 1,
        dtype=torch.int32,
        device=act_seq_lens.device,
    )
    token_offsets[1:] = act_seq_lens.cumsum(0)

    # Compute batch_offsets for each tensor using their actual stride(0)
    # This correctly handles both contiguous and strided (non-contiguous) tensors
    # stride(0) = number of elements to skip in memory per token
    batch_offsets_q = (token_offsets * q.stride(0)).view(-1, 1, 1, 1)
    batch_offsets_k = (token_offsets * k.stride(0)).view(-1, 1, 1, 1)
    batch_offsets_v = (token_offsets * v.stride(0)).view(-1, 1, 1, 1)

    # Output is allocated as contiguous by the kernel, so use contiguous stride
    output_stride = q.shape[1] * q.shape[2]  # num_heads * head_dim
    batch_offsets_o = (token_offsets * output_stride).view(-1, 1, 1, 1)

    actual_seq_lens = act_seq_lens.view(-1, 1, 1, 1)

    # Pad batch_offsets and actual_seq_lens to nearest batch bucket
    # This enables cuDNN graph caching for better performance
    (padded_batch_size, actual_seq_lens_padded,
     batch_offsets_q_padded, batch_offsets_k_padded,
     batch_offsets_v_padded, batch_offsets_o_padded) = _pad_to_batch_bucket(
        batch_size, actual_seq_lens,
        batch_offsets_q, batch_offsets_k, batch_offsets_v, batch_offsets_o,
    )

    output = cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        scale,
        workspace_buffer,
        max_token_per_sequence=max_seqlen,
        max_sequence_kv=max_seqlen,
        actual_seq_lens_q=actual_seq_lens_padded,
        actual_seq_lens_kv=actual_seq_lens_padded,
        causal=False,
        return_lse=False,
        batch_offsets_q=batch_offsets_q_padded,
        batch_offsets_o=batch_offsets_o_padded,
        batch_offsets_k=batch_offsets_k_padded,
        batch_offsets_v=batch_offsets_v_padded,
    )
    if isinstance(output, tuple):
        for i in output:
            if isinstance(i, torch.Tensor):
                output = i
    if is_reshaped:
        output = einops.rearrange(output, "(b s) h d -> b s h d", b=batch_size)

    return output


def vit_flashinfer_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    act_seq_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


direct_register_custom_op(
    op_name="flashinfer_wrapper",
    op_func=flashinfer_wrapper,
    fake_impl=vit_flashinfer_wrapper_fake,
)


def vit_flashinfer_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    act_seq_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.flashinfer_wrapper(
        q, k, v, scale, workspace_buffer, cu_seqlens, max_seqlen, act_seq_lens
    )
