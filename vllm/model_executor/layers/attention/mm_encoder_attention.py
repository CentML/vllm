# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.config import MultiModalConfig
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.vit_attn_wrappers import (
    vit_fa4_flash_attn_wrapper,
    vit_flash_attn_wrapper,
    vit_flashinfer_wrapper,
    vit_torch_sdpa_wrapper,
)

logger = init_logger(__name__)

try:
    from transformer_engine.common.recipe import DelayedScaling
    from transformer_engine.pytorch import DotProductAttention, fp8_autocast
except ImportError:
    DotProductAttention = None
    fp8_autocast = None
    DelayedScaling = None
    logger.warning("TransformerEngine is not installed.")


# --8<-- [start:mm_encoder_attn]
@CustomOp.register("mm_encoder_attn")
class MMEncoderAttention(CustomOp):
    """Multi-headed attention without any cache, used for multimodal encoder."""

    # --8<-- [end:mm_encoder_attn]

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
        multimodal_config: MultiModalConfig | None = None,
        workspace_buffer: torch.Tensor | None = None,  # Only used for FlashInfer
    ) -> None:
        """
        Args:
            num_heads: number of attention heads per partition.
            head_size: hidden_size per attention head.
            scale: scale factor.
            num_kv_heads: number of kv heads.
            prefix: This has no effect, it is only here to make it easier to
                    swap between Attention and MultiHeadAttention
            multimodal_config: configs for multi-modal.
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = 1.0 / (head_size**0.5) if scale is None else scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.layer_name = prefix
        self.workspace_buffer = workspace_buffer
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) is not "
            f"divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()

        # Try to get vision attention backend from multimodal_config.
        attn_backend_override = None
        if multimodal_config is not None:
            attn_backend_override = multimodal_config.mm_encoder_attn_backend

        # Get device-specific vision attention backend.
        self.attn_backend = get_vit_attn_backend(
            head_size=head_size,
            dtype=dtype,
            attn_backend_override=attn_backend_override,
        )

        self.is_flash_attn_backend = self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }

        self.is_fa4_backend = self.attn_backend == AttentionBackendEnum.FLASH_ATTN_CUTE

        self._fa_version = (
            get_flash_attn_version() if self.is_flash_attn_backend else None
        )

        # Initialize Transformer Engine FP8 attention if backend is TE
        self.te_attn_op = None
        self.te_fp8_recipe = None
        self.is_te_fp8_backend = (
            self.attn_backend == AttentionBackendEnum.TE_FP8
            if hasattr(AttentionBackendEnum, 'TE_FP8')
            else False
        )
        
        if self.is_te_fp8_backend:
            if DotProductAttention is None:
                raise ImportError(
                    "TransformerEngine is not installed but TE_FP8 backend was selected"
                )
            self.te_fp8_recipe = DelayedScaling(fp8_dpa=True, fp8_mha=True)
            logger.info_once("Initialized FP8 Transformer Engine for MMEncoderAttention.")

        logger.info_once(f"Using {self.attn_backend} for MMEncoderAttention.")

    @classmethod
    def enabled(cls) -> bool:
        return True

    def maybe_reshape_qkv_to_4d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 4D tensors:
        (batch_size, seq_len, num_heads, head_size)
        """
        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=2)
            value = torch.repeat_interleave(value, num_repeat, dim=2)

        return query, key, value

    def _lazy_init_te_attn(
        self,
        num_attention_heads: int,
        kv_channels: int,
        num_gqa_groups: int | None,
        attn_mask_type: str,
        softmax_scale: float | None,
    ) -> None:
        """Lazily initialize Transformer Engine attention operator."""
        if self.te_attn_op is None:
            self.te_attn_op = DotProductAttention(
                num_attention_heads,
                kv_channels,
                num_gqa_groups=num_gqa_groups,
                attn_mask_type=attn_mask_type,
                softmax_scale=softmax_scale,
                qkv_format="bshd",  # batch, seq, heads, dim
            )

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.maybe_reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        output = vit_torch_sdpa_wrapper(
            q=query,
            k=key,
            v=value,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def _forward_fa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        assert (cu_seqlens is not None and max_seqlen is not None) or (
            cu_seqlens is None and max_seqlen is None
        ), "cu_seqlens and max_seqlen should be both set or both None."

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.maybe_reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        output = vit_flash_attn_wrapper(
            q=query,
            k=key,
            v=value,
            batch_size=bsz,
            is_rocm_aiter=(self.attn_backend == AttentionBackendEnum.ROCM_AITER_FA),
            fa_version=self._fa_version,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def _forward_te_fp8(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using Transformer Engine FP8 attention.
        
        Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        
        Note: TE natively supports GQA, so we don't expand KV heads like other backends.
        Note: Head dimension is padded to multiple of 16 for optimal performance.
        """
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4
        
        # Reshape to 4D if needed, but don't expand KV heads for GQA
        # TE handles GQA natively through num_gqa_groups parameter
        if is_reshaped:
            query = query.view(bsz, q_len, self.num_heads, self.head_size)
            key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
            value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        
        # Pad head dimension to multiple of 16 for optimal performance
        original_head_size = self.head_size
        padded_head_size = ((self.head_size + 15) // 16) * 16
        needs_padding = padded_head_size != original_head_size
        
        if needs_padding:
            pad_size = padded_head_size - original_head_size
            query = torch.nn.functional.pad(query, (0, pad_size))
            key = torch.nn.functional.pad(key, (0, pad_size))
            value = torch.nn.functional.pad(value, (0, pad_size))
        
        # For vision encoder attention, we typically don't need causal masking
        attn_mask_type = "no_mask"
        
        # Determine GQA groups - TE will handle the GQA logic internally
        num_gqa_groups = self.num_kv_heads if self.num_kv_heads != self.num_heads else None
        
        # Lazy initialization of TE attention operator
        self._lazy_init_te_attn(
            num_attention_heads=self.num_heads,
            kv_channels=padded_head_size,
            num_gqa_groups=num_gqa_groups,
            attn_mask_type=attn_mask_type,
            softmax_scale=self.scale,
        )
        
        # Run attention with FP8 autocasting
        # TE expects format: (batch, seq_len, num_heads, head_size)
        with fp8_autocast(enabled=True, fp8_recipe=self.te_fp8_recipe):
            output = self.te_attn_op(
                query, key, value, attention_mask=None
            ).unflatten(-1, (-1, padded_head_size))
        
        # Remove padding from output if needed
        if needs_padding:
            output = output[..., :original_head_size]
        
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        
        return output

    def _forward_flashinfer(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor
        | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        return vit_flashinfer_wrapper(
            q=query,
            k=key,
            v=value,
            scale=self.scale,
            workspace_buffer=self.workspace_buffer,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

    def _forward_fa4(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        """FA4 (flash_attn.cute) attention for multimodal encoder (no KV cache)."""
        assert (cu_seqlens is not None and max_seqlen is not None) or (
            cu_seqlens is None and max_seqlen is None
        ), "cu_seqlens and max_seqlen should be both set or both None."

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.maybe_reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        output = vit_fa4_flash_attn_wrapper(
            q=query,
            k=key,
            v=value,
            batch_size=bsz,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor
        | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        return self._forward_sdpa(query, key, value, cu_seqlens)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor
        | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        if self.is_fa4_backend:
            return self._forward_fa4(query, key, value, cu_seqlens, max_seqlen)
        elif self.is_flash_attn_backend:
            return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
        elif self.attn_backend == AttentionBackendEnum.FLASHINFER:
            return self._forward_flashinfer(
                query, key, value, cu_seqlens, max_seqlen, sequence_lengths
            )
        elif self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
            return self._forward_sdpa(query, key, value, cu_seqlens)
        elif self.is_te_fp8_backend:
            return self._forward_te_fp8(query, key, value, cu_seqlens)
        else:
            raise ValueError(
                f"Unsupported multi-modal encoder attention backend for CUDA: "
                f"{self.attn_backend}."
            )

    def forward_cpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor
        | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        return self._forward_sdpa(query, key, value, cu_seqlens)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor
        | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        assert self.is_flash_attn_backend, (
            "XPU only supports FLASH_ATTN for vision attention."
        )
        return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
