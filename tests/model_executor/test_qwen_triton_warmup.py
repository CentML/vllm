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
