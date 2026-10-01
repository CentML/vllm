# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    RotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# Qwen3.6 TP=1 attention geometry.
HEAD_DIM = 256
ROTARY_DIM = 64
RMS_NORM_EPS = 1e-6
MAX_POSITION_EMBEDDINGS = 262144
ROPE_THETA = 10000000.0
DTYPE = torch.bfloat16
SEED = 13
MROPE_SECTION = (11, 11, 10)
ROPE_CASES = [
    pytest.param(24, 4, None, id="rope"),
    pytest.param(16, 2, MROPE_SECTION, id="interleaved-mrope"),
]


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_rmsnorm_rope_gate Triton kernel requires CUDA/ROCm",
)
@pytest.mark.parametrize("num_q_heads,num_kv_heads,mrope_section", ROPE_CASES)
@pytest.mark.parametrize("num_tokens", [1, 4, 37])
@torch.inference_mode()
def test_fused_qk_norm_rope_gate_matches_reference(
    default_vllm_config,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    mrope_section: tuple[int, int, int] | None,
) -> None:
    device = torch.device("cuda", torch.accelerator.current_device_index())
    torch.set_default_device(device)
    set_random_seed(SEED)

    q_gate = torch.randn(
        num_tokens, num_q_heads * 2 * HEAD_DIM, dtype=DTYPE, device=device
    )
    k = torch.randn(num_tokens, num_kv_heads * HEAD_DIM, dtype=DTYPE, device=device)

    q_norm = GemmaRMSNorm(HEAD_DIM, eps=RMS_NORM_EPS).to(device, dtype=DTYPE)
    k_norm = GemmaRMSNorm(HEAD_DIM, eps=RMS_NORM_EPS).to(device, dtype=DTYPE)
    q_norm.weight.normal_(std=0.1)
    k_norm.weight.normal_(std=0.1)

    q_gate_heads = q_gate.view(num_tokens, num_q_heads, 2 * HEAD_DIM)
    q = q_gate_heads[..., :HEAD_DIM]
    gate_ref = q_gate_heads[..., HEAD_DIM:].reshape(num_tokens, num_q_heads * HEAD_DIM)
    q_ref = q_norm.forward_native(q)
    k_ref = k_norm.forward_native(k.view(num_tokens, num_kv_heads, HEAD_DIM))
    assert isinstance(q_ref, torch.Tensor)
    assert isinstance(k_ref, torch.Tensor)
    q_ref = q_ref.reshape(num_tokens, num_q_heads * HEAD_DIM)
    k_ref = k_ref.reshape(num_tokens, num_kv_heads * HEAD_DIM)

    if mrope_section is None:
        rope = RotaryEmbedding(
            HEAD_DIM,
            ROTARY_DIM,
            MAX_POSITION_EMBEDDINGS,
            ROPE_THETA,
            True,
            DTYPE,
        ).to(device)
        positions = torch.arange(num_tokens, dtype=torch.long, device=device)
    else:
        rope = MRotaryEmbedding(
            HEAD_DIM,
            ROTARY_DIM,
            MAX_POSITION_EMBEDDINGS,
            ROPE_THETA,
            True,
            DTYPE,
            mrope_section=list(mrope_section),
            mrope_interleaved=True,
        ).to(device)
        positions = torch.arange(3 * num_tokens, dtype=torch.long, device=device).view(
            3, num_tokens
        )
        assert torch.unique(positions[:, 0]).numel() == 3

    q_ref, k_ref = rope.forward_native(positions, q_ref, k_ref)
    assert k_ref is not None

    q_out, k_out, gate_out = fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        q_norm.weight,
        k_norm.weight,
        rope.cos_sin_cache,
        positions,
        RMS_NORM_EPS,
        num_q_heads,
        num_kv_heads,
        HEAD_DIM,
        ROTARY_DIM,
        mrope_section=mrope_section,
        norm_beta=1.0,
    )

    # The built-in reference performs RoPE in BF16, while the fused kernel
    # promotes the BF16-normalized values to FP32 for RoPE before storing BF16.
    torch.testing.assert_close(q_out, q_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out, k_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(gate_out, gate_ref, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="fused_qk_rmsnorm_rope Triton kernel requires CUDA",
)
@pytest.mark.parametrize("num_q_heads,num_kv_heads,mrope_section", ROPE_CASES)
@pytest.mark.parametrize("num_tokens", [1, 37, 512])
@torch.inference_mode()
def test_fused_qk_rmsnorm_rope_matches_gate_kernel_bitwise(
    default_vllm_config,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    mrope_section: tuple[int, int, int] | None,
) -> None:
    """The token-tile op keeps the per-head kernel's exact reduction order."""
    from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    device = torch.device("cuda", torch.accelerator.current_device_index())
    set_random_seed(SEED)
    # Magnitudes over several decades, so sums of squares round differently
    # under any other reduction tree.
    scale = torch.logspace(-3, 2, num_tokens, device=device)[:, None]
    q_gate = (
        torch.randn(num_tokens, num_q_heads * 2 * HEAD_DIM, device=device) * scale
    ).to(DTYPE)
    k = (torch.randn(num_tokens, num_kv_heads * HEAD_DIM, device=device) * scale).to(
        DTYPE
    )
    q_weight = (torch.randn(HEAD_DIM, device=device) * 0.1).to(DTYPE)
    k_weight = (torch.randn(HEAD_DIM, device=device) * 0.1).to(DTYPE)
    cos_sin_cache = torch.randn(4096, ROTARY_DIM, device=device).to(DTYPE)
    if mrope_section is None:
        positions = torch.randint(0, 4096, (num_tokens,), device=device)
    else:
        positions = torch.randint(0, 4096, (3, num_tokens), device=device)
    args = (q_gate, k, q_weight, k_weight, cos_sin_cache, positions)
    geometry = (RMS_NORM_EPS, num_q_heads, num_kv_heads, HEAD_DIM, ROTARY_DIM)
    sections = list(mrope_section) if mrope_section else None

    q_ref, k_ref, gate_ref = fused_qk_rmsnorm_rope_gate(
        *args, *geometry, mrope_section=sections, norm_beta=1.0
    )
    q_out, k_out, gate_out = fused_qk_rmsnorm_rope(
        *args, None, *geometry, mrope_section=sections, norm_beta=1.0, store_gate=True
    )
    torch.testing.assert_close(gate_out, gate_ref, atol=0, rtol=0)
    torch.testing.assert_close(q_out, q_ref, atol=0, rtol=0)
    torch.testing.assert_close(k_out, k_ref, atol=0, rtol=0)
    gate = q_gate.view(num_tokens, num_q_heads, 2, HEAD_DIM)[:, :, 1]
    torch.testing.assert_close(gate.reshape(num_tokens, -1), gate_ref, atol=0, rtol=0)

    # FP8 q equals the attention layer's static per-tensor query quant as
    # Inductor compiles it.
    query_quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
    quant = torch.compile(query_quant.forward_native, fullgraph=True)
    for s in (1.0, 0.7, 0.01):
        q_scale = torch.tensor(s, dtype=torch.float32, device=device)
        q_fp8, k_fp8, _ = fused_qk_rmsnorm_rope(
            *args, q_scale, *geometry, mrope_section=sections, norm_beta=1.0
        )
        q_fp8_ref, _ = quant(q_ref, q_scale)
        assert torch.equal(q_fp8.view(torch.uint8), q_fp8_ref.view(torch.uint8))
        torch.testing.assert_close(k_fp8, k_ref, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="fused_qk_rmsnorm_rope Triton kernel requires CUDA",
)
@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "auto"])
@pytest.mark.parametrize("num_tokens,num_slots", [(1, 1), (13, 9), (300, 300)])
# General scales take the IEEE division; power-of-two scales (the default 1.0)
# the exact multiplication by the reciprocal.
@pytest.mark.parametrize("kv_scales", [(0.37, 7.0), (1.0, 1.0), (0.25, 0.5)])
@torch.inference_mode()
def test_fused_qk_rmsnorm_rope_kv_write_matches_reshape_and_cache_flash(
    default_vllm_config,
    num_tokens: int,
    num_slots: int,
    kv_cache_dtype: str,
    kv_scales: tuple[float, float],
) -> None:
    """The paged-cache write equals reshape_and_cache_flash of the op's k and v
    (padding slots -1 and rows past slot_mapping untouched), byte for byte.
    """
    from vllm import _custom_ops  # noqa: F401  (registers _C_cache_ops)
    from vllm.model_executor.layers.fused_qk_norm_rope import (
        PagedKVWrite,
        fused_qk_rmsnorm_rope,
    )

    num_q_heads, num_kv_heads, block_size, num_blocks = 16, 2, 16, 64
    device = torch.device("cuda", torch.accelerator.current_device_index())
    set_random_seed(SEED)
    # Per-element magnitudes over 1e-4..1e3, so FP8 saturates (> 448 after the
    # scale) and underflows in every case.
    width = (2 * num_q_heads + 2 * num_kv_heads) * HEAD_DIM
    magnitude = 10 ** (torch.rand(num_tokens, width, device=device) * 7 - 4)
    qkv = (torch.randn(num_tokens, width, device=device) * magnitude).to(DTYPE)
    q_size, kv_size = num_q_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    q_gate, k, v = qkv.split([2 * q_size, kv_size, kv_size], dim=-1)
    # 7 * 0.390625 and 7 * 0.78125: divided by 7.0 they are exact e4m3 ties
    # (0.375 / 0.75, round to even); multiplied by fp32(1 / 7) they round up.
    v[:, 0:4] = 2.734375
    v[:, 4:8] = -5.46875
    v[:, 8:12] = 4096.0  # saturates under every scale
    q_weight = (torch.randn(HEAD_DIM, device=device) * 0.1).to(DTYPE)
    k_weight = (torch.randn(HEAD_DIM, device=device) * 0.1).to(DTYPE)
    cos_sin_cache = torch.randn(4096, ROTARY_DIM, device=device).to(DTYPE)
    positions = torch.randint(0, 4096, (num_tokens,), device=device)
    slots = torch.randperm(num_blocks * block_size, device=device)[:num_slots]
    slots[1::5] = -1
    k_scale = torch.tensor(kv_scales[0], dtype=torch.float32, device=device)
    v_scale = torch.tensor(kv_scales[1], dtype=torch.float32, device=device)
    # The FlashInfer backend's cache (B, H, N, 2 * D) and its write views.
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else DTYPE
    caches = []
    for _ in range(2):
        cache = torch.zeros(
            num_blocks, num_kv_heads, block_size, 2 * HEAD_DIM, device=device
        ).to(cache_dtype)
        caches.append(cache)
    caches[1].copy_(caches[0])
    args = (q_gate, k, q_weight, k_weight, cos_sin_cache, positions, None)
    geometry = (RMS_NORM_EPS, num_q_heads, num_kv_heads, HEAD_DIM, ROTARY_DIM)

    q_ref, k_ref, _ = fused_qk_rmsnorm_rope(*args, *geometry, norm_beta=1.0)
    k_cache, v_cache = caches[0].transpose(1, 2).split(HEAD_DIM, dim=-1)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        k_ref.view(-1, num_kv_heads, HEAD_DIM),
        v.view(-1, num_kv_heads, HEAD_DIM),
        k_cache,
        v_cache,
        slots,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    k_cache, v_cache = caches[1].transpose(1, 2).split(HEAD_DIM, dim=-1)
    if kv_cache_dtype == "fp8":
        k_cache = k_cache.view(torch.float8_e4m3fn)
        v_cache = v_cache.view(torch.float8_e4m3fn)
    q_out, k_out, _ = fused_qk_rmsnorm_rope(
        *args,
        *geometry,
        norm_beta=1.0,
        kv_write=PagedKVWrite(v, k_cache, v_cache, slots, k_scale, v_scale),
    )
    assert torch.equal(q_out, q_ref) and torch.equal(k_out, k_ref)
    assert torch.equal(caches[1].view(torch.uint8), caches[0].view(torch.uint8))
    if kv_cache_dtype == "fp8":
        # The case exercises saturation, zeros and both scales.
        written = caches[0].view(torch.float8_e4m3fn).float()
        assert written.abs().max() == 448.0 and (written == 0).any()
