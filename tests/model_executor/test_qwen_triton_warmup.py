# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.warmup.qwen_triton_warmup import (
    _FLA_POST_CONV_WARMUP_LENGTHS,
    _QwenGDNWarmupConfig,
    _warm_causal_conv1d_fwd_kernel,
    _warm_fused_post_conv_kernel,
    _warm_gated_rms_norm_kernel,
    _warm_mxfp8_producers,
)
from vllm.platforms import current_platform


def _cuda_gdn_config() -> _QwenGDNWarmupConfig:
    h, hv, k, v = 2, 2, 16, 16
    conv_kernel_size = 4
    conv_dim = 2 * h * k + hv * v
    device = torch.device("cuda")
    conv_state = torch.empty(
        (8, conv_dim, conv_kernel_size - 1),
        dtype=torch.bfloat16,
        device=device,
    )
    return _QwenGDNWarmupConfig(
        h=h,
        hv=hv,
        k=k,
        v=v,
        conv_kernel_size=conv_kernel_size,
        conv_state=conv_state,
        conv_dtype=conv_state.dtype,
        norm_weight_dtype=torch.bfloat16,
        norm_before_gate=True,
        norm_activation="silu",
        a_log=torch.zeros(hv, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(hv, dtype=torch.float32, device=device),
        state_stride_token=hv * v * k,
        state_dtype=torch.float32,
    )


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_qwen_gdn_prefill_warmup_kernels_compile_on_gpu() -> None:
    config = _cuda_gdn_config()
    device = torch.device("cuda")
    _warm_gated_rms_norm_kernel(
        device, config, max_num_tokens=16, x_dtype=config.conv_dtype
    )
    _warm_causal_conv1d_fwd_kernel(device, config)
    _warm_fused_post_conv_kernel(device, config)
    assert _FLA_POST_CONV_WARMUP_LENGTHS == (1, 2, 16)
    torch.accelerator.synchronize(device)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA is required")
@torch.inference_mode()
def test_mxfp8_producer_warmup_covers_every_launch_config(default_vllm_config) -> None:
    """After the warmup, no token count compiles another producer variant."""
    from triton import knobs

    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.fusion.attn_gate_mxfp8_quant import (
        attn_gate_mxfp8_quant,
    )
    from vllm.model_executor.layers.fusion.silu_mul_mxfp8_quant import (
        silu_mul_mxfp8_quant,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    d, heads, head_dim = 384, 8, 128
    mlp = torch.nn.Module()
    mlp.act_fn = SiluAndMul()
    mlp.down_proj = torch.nn.Module()
    mlp.down_proj._input_quant_key = kMxfp8Dynamic
    mlp.down_proj.input_size_per_partition = d
    attn = torch.nn.Module()
    attn.attn_gate_mxfp8 = True
    attn.num_heads, attn.head_dim = heads, head_dim
    model = torch.nn.ModuleDict({"mlp": mlp, "attn": attn})

    class _ModelConfig:
        dtype = torch.bfloat16

    class _Runner:
        model_config = _ModelConfig()

        def get_model(self) -> torch.nn.Module:
            return model

    device = torch.device("cuda")
    _warm_mxfp8_producers(_Runner(), device)

    compiled: list[str] = []
    previous_hook = knobs.runtime.jit_post_compile_hook
    knobs.runtime.jit_post_compile_hook = lambda **kw: compiled.append(
        getattr(kw.get("fn"), "name", "?")
    )
    try:
        for num_tokens in (2, 100, 129, 300, 1000, 2048, 3000, 5000):
            x = torch.randn(num_tokens, 2 * d, device=device).bfloat16()
            silu_mul_mxfp8_quant(x)
            a = torch.randn(num_tokens, heads, head_dim, device=device).bfloat16()
            q_gate = torch.randn(num_tokens, heads, 2, head_dim, device=device)
            attn_gate_mxfp8_quant(a, q_gate.bfloat16()[:, :, 1])
    finally:
        knobs.runtime.jit_post_compile_hook = previous_hook
    assert compiled == []


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA is required")
@torch.inference_mode()
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_gdn_state_kernel_warmup_covers_serving_calls(state_dtype) -> None:
    """After the warmup, serving-shaped calls compile no other variant of the
    fresh-row zeroing kernel (index/flag slices at any offset, padded pool) or
    of the Triton spec-decode recurrence (every batch of <= the Triton limit).
    """
    import dataclasses
    import types

    from triton import knobs

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        GDN_MTP_TRITON_MAX_REQUESTS,
    )
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        zero_fresh_state_rows,
    )
    from vllm.model_executor.layers.mamba.ops.gdn_mtp_decode import (
        gdn_mtp_recurrence,
    )
    from vllm.model_executor.warmup.qwen_triton_warmup import (
        _warm_gdn_mtp_recurrence_kernel,
        _warm_zero_fresh_state_rows_kernel,
    )

    config = dataclasses.replace(
        _cuda_gdn_config(), state_dtype=state_dtype, state_in_place=True
    )
    device = torch.device("cuda")
    num_spec = 4
    runner = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(num_speculative_tokens=num_spec)
    )
    _warm_zero_fresh_state_rows_kernel(device, config)
    _warm_gdn_mtp_recurrence_kernel(runner, device, config, torch.bfloat16)

    hv, k, v = config.hv, config.k, config.v
    slots = 8
    padded = hv * v * k + 64
    pool = torch.zeros(slots * padded, dtype=state_dtype, device=device).as_strided(
        (slots, hv, v, k), (padded, v * k, k, 1)
    )
    width = num_spec + 1
    compiled: list[str] = []
    previous_hook = knobs.runtime.jit_post_compile_hook
    knobs.runtime.jit_post_compile_hook = lambda **kw: compiled.append(
        getattr(kw.get("fn"), "name", "?")
    )
    try:
        indices = torch.arange(32, dtype=torch.int32, device=device) % slots
        flags = torch.arange(32, device=device) % 2 == 0
        for start in range(17):
            zero_fresh_state_rows(
                pool, indices[start : start + 3], flags[start : start + 3]
            )
        for num_requests in range(1, GDN_MTP_TRITON_MAX_REQUESTS + 1):
            num_tokens = num_requests * width
            mixed_qkv = torch.randn(
                num_tokens, config.conv_dim, device=device, dtype=torch.bfloat16
            )
            a = torch.randn(num_tokens, hv, device=device, dtype=torch.bfloat16)
            gdn_mtp_recurrence(
                mixed_qkv,
                a,
                torch.randn_like(a),
                config.a_log,
                config.dt_bias,
                torch.randint(
                    1, slots, (num_requests, width), dtype=torch.int32, device=device
                ),
                torch.arange(
                    0, num_tokens + 1, width, dtype=torch.int32, device=device
                ),
                torch.ones(num_requests, dtype=torch.int32, device=device),
                pool,
                torch.empty(num_tokens, hv, v, dtype=torch.bfloat16, device=device),
                scale=k**-0.5,
            )
        torch.accelerator.synchronize(device)
    finally:
        knobs.runtime.jit_post_compile_hook = previous_hook
    assert compiled == []
