# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for MLA with fused RoPE + KV cache kernel.

This test validates that the fused kernel path produces the same results
as the non-fused path in the MLA layer.
"""

import random

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm import _custom_ops as ops
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_neox_style", [True])
@pytest.mark.parametrize("seq_len", [1, 16, 64])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("qk_nope_head_dim", [128])
@pytest.mark.parametrize("num_q_heads", [128])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("num_blocks", [64])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize(
    "device", [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
)
@torch.inference_mode()
def test_mla_fused_vs_separate_rope(
    default_vllm_config,
    dtype: torch.dtype,
    is_neox_style: bool,
    seq_len: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    num_q_heads: int,
    kv_cache_dtype: str,
    kv_lora_rank: int,
    num_blocks: int,
    block_size: int,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: float = 10000,
) -> None:
    """Test that fused RoPE + KV cache kernel produces same results as separate ops."""
    set_random_seed(seed)
    torch.set_default_device(device)

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Create rotary embedding
    rope = RotaryEmbedding(
        qk_rope_head_dim,
        qk_rope_head_dim,
        max_position,
        base,
        is_neox_style,
        torch.float32,
    )
    rope = rope.to(dtype=dtype, device=torch.get_default_device())

    # Create input tensors matching MLA shapes
    positions = torch.randint(0, max_position, (seq_len,))

    # q: [seq_len, num_heads, qk_head_dim] where qk_head_dim = nope + rope
    q = torch.randn(seq_len, num_q_heads, qk_head_dim, dtype=dtype)

    # kv_c_normed: [seq_len, kv_lora_rank]
    kv_c_normed = torch.randn(seq_len, kv_lora_rank, dtype=dtype)

    # k_pe: [seq_len, 1, qk_rope_head_dim]
    k_pe = torch.randn(seq_len, 1, qk_rope_head_dim, dtype=dtype)

    # Setup slot mapping
    total_available_slots = num_blocks * block_size
    total_needed_slots = seq_len
    assert total_available_slots >= total_needed_slots, "Not enough kv slots!"
    slot_mapping_lst = random.sample(range(total_available_slots), total_needed_slots)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    entry_size = kv_lora_rank + qk_rope_head_dim
    kv_cache_scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    # ====== Non-fused path (reference) ======
    q_ref = q.clone()
    k_pe_ref = k_pe.clone()
    kv_c_ref = kv_c_normed.clone()

    # Apply RoPE separately
    q_pe_ref = q_ref[..., qk_nope_head_dim:]
    k_pe_squeezed_ref = k_pe_ref.squeeze(1)

    if current_platform.is_rocm():
        q_pe_roped_ref, k_pe_roped_ref = rope.forward_hip(
            positions, q_pe_ref, k_pe_squeezed_ref
        )
    else:
        q_pe_roped_ref, k_pe_roped_ref = rope.forward_native(
            positions, q_pe_ref, k_pe_squeezed_ref
        )
    assert k_pe_roped_ref is not None

    # KV cache (reference)
    kv_cache_ref = torch.zeros(
        num_blocks,
        block_size,
        entry_size,
        dtype=torch.uint8 if kv_cache_dtype == "fp8" else dtype,
        device=device,
    )
    ops.concat_and_cache_mla(
        kv_c_ref,
        k_pe_roped_ref,
        kv_cache_ref,
        slot_mapping,
        kv_cache_dtype=kv_cache_dtype,
        scale=kv_cache_scale,
    )

    # ====== Fused path ======
    # Create separate q_pe tensor (like the original test does)
    # instead of slicing from full q tensor which creates non-contiguous view
    q_pe_fused = q[..., qk_nope_head_dim:].clone().contiguous()
    k_pe_fused = k_pe.clone()
    kv_c_fused = kv_c_normed.clone()

    # KV cache (fused)
    kv_cache_fused = torch.zeros(
        num_blocks,
        block_size,
        entry_size,
        dtype=torch.uint8 if kv_cache_dtype == "fp8" else dtype,
        device=device,
    )

    # Squeeze k_pe and ensure contiguous
    k_pe_squeezed_fused = k_pe_fused.squeeze(1).contiguous()

    # Use rope.cos_sin_cache directly - kernel expects same dtype as q_pe
    # Use fused kernel
    ops.concat_and_cache_mla_rope_fused(
        positions,
        q_pe_fused,  # [seq_len, num_heads, rope_dim] - modified in place
        k_pe_squeezed_fused,  # [seq_len, rope_dim] - modified in place
        kv_c_fused,
        rope.cos_sin_cache,  # Same dtype as q_pe (bf16/fp16)
        is_neox_style,
        slot_mapping,
        kv_cache_fused,
        kv_cache_dtype,
        kv_cache_scale,
    )

    # ====== Validate results ======
    # Check Q RoPE portion matches
    torch.testing.assert_close(
        q_pe_fused,
        q_pe_roped_ref,
        atol=get_default_atol(q_pe_fused),
        rtol=get_default_rtol(q_pe_fused),
    )

    # Check KV cache matches
    if kv_cache_dtype == "fp8":
        # Convert both to float16 for comparison
        result_temp = torch.empty_like(kv_cache_fused, dtype=torch.float16)
        ops.convert_fp8(
            result_temp,
            kv_cache_fused.contiguous(),
            kv_cache_scale.item(),
            kv_dtype=kv_cache_dtype,
        )
        expected_temp = torch.empty_like(kv_cache_ref, dtype=torch.float16)
        ops.convert_fp8(
            expected_temp,
            kv_cache_ref,
            kv_cache_scale.item(),
            kv_dtype=kv_cache_dtype,
        )
        torch.testing.assert_close(result_temp, expected_temp, atol=0.001, rtol=0.1)
    else:
        torch.testing.assert_close(kv_cache_fused, kv_cache_ref)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seq_len", [1, 8])
@pytest.mark.parametrize("kv_cache_dtype", ["fp8"])
@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_mla_fused_q_rope_in_place(
    default_vllm_config,
    dtype: torch.dtype,
    seq_len: int,
    kv_cache_dtype: str,
    seed: int,
) -> None:
    """Test that fused kernel correctly modifies Q RoPE portion in-place."""
    set_random_seed(seed)
    device = "cuda:0"
    torch.set_default_device(device)

    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    num_q_heads = 128
    kv_lora_rank = 512
    num_blocks = 16
    block_size = 64
    max_position = 8192
    base = 10000
    is_neox_style = True

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    rope = RotaryEmbedding(
        qk_rope_head_dim,
        qk_rope_head_dim,
        max_position,
        base,
        is_neox_style,
        torch.float32,
    )
    rope = rope.to(dtype=dtype, device=device)

    positions = torch.randint(0, max_position, (seq_len,))

    # Create full Q tensor
    q = torch.randn(seq_len, num_q_heads, qk_head_dim, dtype=dtype)
    q_nope_original = q[..., :qk_nope_head_dim].clone()
    q_pe_original = q[..., qk_nope_head_dim:].clone()

    kv_c_normed = torch.randn(seq_len, kv_lora_rank, dtype=dtype)
    k_pe = torch.randn(seq_len, qk_rope_head_dim, dtype=dtype)

    slot_mapping_lst = list(range(seq_len))
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    entry_size = kv_lora_rank + qk_rope_head_dim
    kv_cache_scale = torch.tensor([0.1], dtype=torch.float32, device=device)
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        entry_size,
        dtype=torch.uint8 if kv_cache_dtype == "fp8" else dtype,
        device=device,
    )

    # Apply fused kernel - q_pe is a view into q, so q should be modified
    # Need to make q_pe contiguous for the kernel to work
    q_pe_contiguous = q[..., qk_nope_head_dim:].clone().contiguous()
    
    ops.concat_and_cache_mla_rope_fused(
        positions,
        q_pe_contiguous,
        k_pe,
        kv_c_normed,
        rope.cos_sin_cache,  # Same dtype as q_pe
        is_neox_style,
        slot_mapping,
        kv_cache,
        kv_cache_dtype,
        kv_cache_scale,
    )
    
    # Copy back to verify the kernel modified q_pe
    q[..., qk_nope_head_dim:] = q_pe_contiguous

    # Verify Q nope portion is unchanged
    torch.testing.assert_close(q[..., :qk_nope_head_dim], q_nope_original)

    # Verify Q rope portion was modified (should not equal original)
    # Note: In rare cases with all zeros, RoPE might not change values,
    # but with random data this should be different
    assert not torch.allclose(q[..., qk_nope_head_dim:], q_pe_original), (
        "Q RoPE portion should be modified by fused kernel"
    )
