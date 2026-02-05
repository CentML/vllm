# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.config import MultiModalConfig
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
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


@functools.cache
def _load_fp8_scales_file(path: str | None) -> dict[str, dict[str, float]]:
    """Load FP8 scales from file. Results are cached."""
    if path is None:
        return {}
    
    path = str(Path(path).expanduser())
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle nested "layers" format
    if "layers" in data and isinstance(data["layers"], dict):
        data = data["layers"]
    
    scales: dict[str, dict[str, float]] = {}
    for layer_name, layer_scales in data.items():
        if not isinstance(layer_scales, dict):
            continue
        q = layer_scales.get("q", layer_scales.get("q_scale"))
        k = layer_scales.get("k", layer_scales.get("k_scale"))
        v = layer_scales.get("v", layer_scales.get("v_scale"))
        if q is not None and k is not None and v is not None:
            scales[layer_name] = {"q": float(q), "k": float(k), "v": float(v)}
    
    logger.info(f"Loaded FP8 attention scales from {path} ({len(scales)} layers)")
    return scales


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
            prefix: layer name prefix, used to look up FP8 scales.
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

        logger.info_once(f"Using {self.attn_backend} for MMEncoderAttention.")

        # FP8 attention support
        self.fp8_enabled = False
        self.fp8_scales: dict[str, float] | None = None
        self.fp8_quant: QuantFP8 | None = None
        
        if envs.VLLM_MM_ENCODER_FP8_ATTN:
            self._init_fp8_attention(prefix)

    def _init_fp8_attention(self, layer_name: str) -> None:
        """Initialize FP8 attention for this layer."""
        scale_path = envs.VLLM_MM_ENCODER_FP8_ATTN_SCALE_PATH
        all_scales = _load_fp8_scales_file(scale_path)
        
        if scale_path is None:
            # No scale path provided - use scale=1.0 and warn
            logger.warning_once(
                "VLLM_MM_ENCODER_FP8_ATTN enabled but "
                "VLLM_MM_ENCODER_FP8_ATTN_SCALE_PATH not set. "
                "Using scale=1.0 for all Q/K/V (may cause accuracy issues)."
            )
            self.fp8_scales = {"q": 1.0, "k": 1.0, "v": 1.0}
        else:
            # Scale path provided - layer must exist
            layer_scales = all_scales.get(layer_name)
            if layer_scales is None:
                raise ValueError(
                    "FP8 attention enabled but scales not found for layer "
                    f"'{layer_name}' in {scale_path}. "
                    f"Available layers: {list(all_scales.keys())}"
                )
            self.fp8_scales = layer_scales
        
        # Register scale tensors as buffers (auto-move to device with module)
        # Shape (1, 1, 1, 1) as required by cuDNN
        self.register_buffer(
            "_fp8_q_scale",
            torch.tensor([self.fp8_scales["q"]], dtype=torch.float32).view(1, 1, 1, 1)
        )
        self.register_buffer(
            "_fp8_k_scale",
            torch.tensor([self.fp8_scales["k"]], dtype=torch.float32).view(1, 1, 1, 1)
        )
        self.register_buffer(
            "_fp8_v_scale",
            torch.tensor([self.fp8_scales["v"]], dtype=torch.float32).view(1, 1, 1, 1)
        )
        
        # Create QuantFP8 for efficient quantization
        self.fp8_quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
        self.fp8_enabled = True
        
        logger.debug(
            f"FP8 attention enabled for {layer_name}: "
            f"q={self.fp8_scales['q']:.4f}, "
            f"k={self.fp8_scales['k']:.4f}, "
            f"v={self.fp8_scales['v']:.4f}"
        )

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

    def _quantize_to_fp8(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize a 3D tensor (total_tokens, num_heads, head_dim) to FP8.
        
        Uses QuantFP8 CustomOp for backend-aware quantization.
        """
        assert self.fp8_quant is not None
        orig_shape = tensor.shape
        # QuantFP8 expects 2D input: (total_tokens, num_heads * head_dim)
        tensor_2d = tensor.view(orig_shape[0], -1)
        fp8_tensor, _ = self.fp8_quant(tensor_2d, scale=scale)
        return fp8_tensor.view(orig_shape)

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
        if self.fp8_enabled:
            assert self.fp8_quant is not None and self.fp8_scales is not None
            query = self._quantize_to_fp8(query, self._fp8_q_scale)
            key = self._quantize_to_fp8(key, self._fp8_k_scale)
            value = self._quantize_to_fp8(value, self._fp8_v_scale)
            
        return vit_flashinfer_wrapper(
            q=query,
            k=key,
            v=value,
            scale=self.scale,
            workspace_buffer=self.workspace_buffer,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
            q_scale=self._fp8_q_scale if self.fp8_enabled else None,
            k_scale=self._fp8_k_scale if self.fp8_enabled else None,
            v_scale=self._fp8_v_scale if self.fp8_enabled else None,
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
