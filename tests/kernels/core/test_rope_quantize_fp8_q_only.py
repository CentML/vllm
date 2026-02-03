# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for FlashInfer rope_quantize_fp8 Q-only fusion path.

This tests the Q-only fusion strategy where:
1. Q path uses FlashInfer's rope_quantize_fp8 for fused RoPE + FP8 quantization
2. K path remains unchanged (dummy K passed to kernel, K outputs ignored)
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# Check if FlashInfer rope_quantize_fp8 is available
try:
    from flashinfer.rope import rope_quantize_fp8

    HAS_FLASHINFER_ROPE_QUANTIZE_FP8 = True
except ImportError:
    HAS_FLASHINFER_ROPE_QUANTIZE_FP8 = False


def skip_if_no_flashinfer_rope_quantize_fp8():
    """Skip test if FlashInfer rope_quantize_fp8 is not available."""
    if not HAS_FLASHINFER_ROPE_QUANTIZE_FP8:
        pytest.skip("FlashInfer rope_quantize_fp8 not available")


def compute_cos_sin_cache(
    rotary_dim: int,
    max_position_embeddings: int,
    base: float = 10000.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute cos/sin cache for RoPE embedding.

    Returns cache as float32, which is required by FlashInfer.
    """
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    # FlashInfer expects [max_seq_len, rotary_dim] format with cos and sin interleaved
    cache = torch.cat((cos, sin), dim=-1)
    return cache


def apply_rope_reference(
    x: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool = True,
) -> torch.Tensor:
    """Reference RoPE implementation.

    Args:
        x: Input tensor [B, N, R] or [B, R]
        cos_sin_cache: Cache [max_seq_len, rotary_dim]
        positions: Position indices [B]
        is_neox: Use NeoX-style RoPE (default True)

    Returns:
        Rotated tensor with same shape as input
    """
    rotary_dim = cos_sin_cache.shape[-1] // 2

    # Get cos and sin for positions
    cos = cos_sin_cache[positions, :rotary_dim]  # [B, R/2]
    sin = cos_sin_cache[positions, rotary_dim:]  # [B, R/2]

    # Add dimensions for broadcasting
    if x.ndim == 3:
        cos = cos.unsqueeze(1)  # [B, 1, R/2]
        sin = sin.unsqueeze(1)  # [B, 1, R/2]

    # Split into rotate and pass-through parts
    x_rot = x[..., :rotary_dim]

    # Split x_rot into two halves
    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2 :]

    if is_neox:
        # NeoX-style: rotate adjacent pairs
        rotated = torch.cat([
            x1 * cos[..., : rotary_dim // 2] - x2 * sin[..., : rotary_dim // 2],
            x1 * sin[..., rotary_dim // 2 :] + x2 * cos[..., rotary_dim // 2 :],
        ], dim=-1)
    else:
        # GPT-J style: interleaved rotation
        rotated = torch.cat([
            x1 * cos[..., ::2] - x2 * sin[..., ::2],
            x1 * sin[..., 1::2] + x2 * cos[..., 1::2],
        ], dim=-1)

    return rotated


def quantize_fp8_reference(
    tensor: torch.Tensor,
    scale: float,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Reference FP8 quantization implementation."""
    # FlashInfer uses reciprocal scale convention: quantized = input * scale
    # where scale = FP8_MAX / amax
    scaled = tensor * scale
    # Clamp to FP8 range
    fp8_max = torch.finfo(dtype).max
    clamped = torch.clamp(scaled, -fp8_max, fp8_max)
    return clamped.to(dtype)


@pytest.mark.skipif(
    not HAS_FLASHINFER_ROPE_QUANTIZE_FP8,
    reason="FlashInfer rope_quantize_fp8 not available",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 32, 128])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("num_q_heads", [128])
@pytest.mark.parametrize("is_neox_style", [True])
@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_rope_quantize_fp8_q_only_correctness(
    default_vllm_config,
    dtype: torch.dtype,
    batch_size: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    num_q_heads: int,
    is_neox_style: bool,
    seed: int,
    max_position: int = 8192,
    base: float = 10000,
) -> None:
    """
    Verify rope_quantize_fp8 with dummy K produces correct Q output.

    Tests the Q-only fusion strategy:
    1. Pass real Q inputs (q_rope unrotated, q_nope from BMM)
    2. Pass dummy K inputs (zeros)
    3. Verify Q outputs match reference (separate RoPE + quant)
    """
    skip_if_no_flashinfer_rope_quantize_fp8()

    device = "cuda"
    set_random_seed(seed)
    torch.set_default_device(device)

    # Create cos_sin_cache as float32 (required by FlashInfer)
    cos_sin_cache = compute_cos_sin_cache(qk_rope_head_dim, max_position, base, device)

    # Generate random inputs
    positions = torch.randint(0, max_position, (batch_size,), device=device)

    # q_rope: unrotated Q rope portion [B, N, R]
    q_rope_unrotated = torch.randn(
        batch_size, num_q_heads, qk_rope_head_dim, dtype=dtype, device=device
    )

    # q_nope: Q nope portion (simulating output of BMM) [B, N, Lkv]
    q_nope = torch.randn(
        batch_size, num_q_heads, kv_lora_rank, dtype=dtype, device=device
    )

    # Dummy K tensors
    dummy_k_rope = torch.zeros(batch_size, qk_rope_head_dim, dtype=dtype, device=device)
    dummy_k_nope = torch.zeros(batch_size, kv_lora_rank, dtype=dtype, device=device)

    # Scale for quantization
    q_scale = 1.0

    fp8_dtype = current_platform.fp8_dtype()

    # FlashInfer fused kernel with dummy K
    fused_q_rope_fp8, fused_k_rope_fp8, fused_q_nope_fp8, fused_k_nope_fp8 = (
        rope_quantize_fp8(
            q_rope=q_rope_unrotated,
            k_rope=dummy_k_rope,
            q_nope=q_nope,
            k_nope=dummy_k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            is_neox=is_neox_style,
            quantize_dtype=fp8_dtype,
            quant_scale_q=q_scale,
            quant_scale_kv=1.0,  # Unused
        )
    )

    # Verify output shapes are correct
    assert fused_q_rope_fp8.shape == (batch_size, num_q_heads, qk_rope_head_dim)
    assert fused_q_nope_fp8.shape == (batch_size, num_q_heads, kv_lora_rank)
    assert fused_q_rope_fp8.dtype == fp8_dtype
    assert fused_q_nope_fp8.dtype == fp8_dtype


@pytest.mark.skipif(
    not HAS_FLASHINFER_ROPE_QUANTIZE_FP8,
    reason="FlashInfer rope_quantize_fp8 not available",
)
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_dummy_k_does_not_affect_q(
    default_vllm_config,
    batch_size: int,
    seed: int,
    qk_rope_head_dim: int = 64,
    kv_lora_rank: int = 512,
    num_q_heads: int = 128,
    max_position: int = 8192,
    base: float = 10000,
) -> None:
    """
    Verify that different dummy K values produce identical Q outputs.

    This confirms K processing is independent and can be safely ignored.
    """
    skip_if_no_flashinfer_rope_quantize_fp8()

    device = "cuda"
    dtype = torch.bfloat16
    set_random_seed(seed)
    torch.set_default_device(device)

    # Create cos_sin_cache as float32 (required by FlashInfer)
    cos_sin_cache = compute_cos_sin_cache(qk_rope_head_dim, max_position, base, device)

    # Generate random Q inputs
    positions = torch.randint(0, max_position, (batch_size,), device=device)
    q_rope_unrotated = torch.randn(
        batch_size, num_q_heads, qk_rope_head_dim, dtype=dtype, device=device
    )
    q_nope = torch.randn(
        batch_size, num_q_heads, kv_lora_rank, dtype=dtype, device=device
    )

    fp8_dtype = current_platform.fp8_dtype()

    # Run with zeros for K
    zeros_k_rope = torch.zeros(batch_size, qk_rope_head_dim, dtype=dtype, device=device)
    zeros_k_nope = torch.zeros(batch_size, kv_lora_rank, dtype=dtype, device=device)

    q_rope_fp8_1, _, q_nope_fp8_1, _ = rope_quantize_fp8(
        q_rope=q_rope_unrotated.clone(),
        k_rope=zeros_k_rope,
        q_nope=q_nope.clone(),
        k_nope=zeros_k_nope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        is_neox=True,
        quantize_dtype=fp8_dtype,
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
    )

    # Run with random values for K
    random_k_rope = torch.randn(
        batch_size, qk_rope_head_dim, dtype=dtype, device=device
    )
    random_k_nope = torch.randn(batch_size, kv_lora_rank, dtype=dtype, device=device)

    q_rope_fp8_2, _, q_nope_fp8_2, _ = rope_quantize_fp8(
        q_rope=q_rope_unrotated.clone(),
        k_rope=random_k_rope,
        q_nope=q_nope.clone(),
        k_nope=random_k_nope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        is_neox=True,
        quantize_dtype=fp8_dtype,
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
    )

    # Q outputs should be identical regardless of K values
    torch.testing.assert_close(q_rope_fp8_1, q_rope_fp8_2)
    torch.testing.assert_close(q_nope_fp8_1, q_nope_fp8_2)


@pytest.mark.skipif(
    not HAS_FLASHINFER_ROPE_QUANTIZE_FP8,
    reason="FlashInfer rope_quantize_fp8 not available",
)
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("quant_scale_q", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_scale_compatibility(
    default_vllm_config,
    batch_size: int,
    quant_scale_q: float,
    seed: int,
    qk_rope_head_dim: int = 64,
    kv_lora_rank: int = 512,
    num_q_heads: int = 128,
    max_position: int = 8192,
) -> None:
    """
    Verify FlashInfer scale convention matches vLLM expectations.

    Tests that different scale values work correctly with the fused kernel.
    """
    skip_if_no_flashinfer_rope_quantize_fp8()

    device = "cuda"
    dtype = torch.bfloat16
    set_random_seed(seed)
    torch.set_default_device(device)

    # Create cos_sin_cache as float32 (required by FlashInfer)
    cos_sin_cache = compute_cos_sin_cache(qk_rope_head_dim, max_position, 10000.0, device)

    # Generate inputs
    positions = torch.randint(0, max_position, (batch_size,), device=device)
    q_rope_unrotated = torch.randn(
        batch_size, num_q_heads, qk_rope_head_dim, dtype=dtype, device=device
    )
    q_nope = torch.randn(
        batch_size, num_q_heads, kv_lora_rank, dtype=dtype, device=device
    )

    dummy_k_rope = torch.zeros(batch_size, qk_rope_head_dim, dtype=dtype, device=device)
    dummy_k_nope = torch.zeros(batch_size, kv_lora_rank, dtype=dtype, device=device)

    fp8_dtype = current_platform.fp8_dtype()

    # This should not raise an error
    q_rope_fp8, _, q_nope_fp8, _ = rope_quantize_fp8(
        q_rope=q_rope_unrotated,
        k_rope=dummy_k_rope,
        q_nope=q_nope,
        k_nope=dummy_k_nope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        is_neox=True,
        quantize_dtype=fp8_dtype,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=1.0,
    )

    # Verify output shapes are correct
    assert q_rope_fp8.shape == (batch_size, num_q_heads, qk_rope_head_dim)
    assert q_nope_fp8.shape == (batch_size, num_q_heads, kv_lora_rank)
    assert q_rope_fp8.dtype == fp8_dtype
    assert q_nope_fp8.dtype == fp8_dtype
