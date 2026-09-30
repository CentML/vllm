# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused (add +) RMSNorm -> MXFP8 producer against the ops it replaces."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

EPS = 1e-6

requires_mxfp8_quantize = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.has_device_capability(100)
        and has_flashinfer()
    ),
    reason="FlashInfer MXFP8 quantize needs SM100+",
)


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@requires_mxfp8_quantize
@pytest.mark.parametrize("num_tokens", [1, 300, 512])
@pytest.mark.parametrize("hidden", [1024, 2048])
@pytest.mark.parametrize("inputs", ["x", "x+res", "x+x2+res"])
@pytest.mark.parametrize("weight_offset", [0.0, 1.0])
@torch.inference_mode()
def test_add_rms_norm_mxfp8_quant(
    num_tokens: int, hidden: int, inputs: str, weight_offset: float
) -> None:
    from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
        add_rms_norm_mxfp8_quant,
    )

    torch.manual_seed(0)
    # Per-token magnitudes over many decades and an all-zero row (zero scales).
    scale = torch.logspace(-6, 3, num_tokens, device="cuda")[:, None]
    x = (torch.randn(num_tokens, hidden, device="cuda") * scale).bfloat16()
    x2 = (torch.randn_like(x, dtype=torch.float) * scale).bfloat16()
    res = (torch.randn_like(x, dtype=torch.float) * scale).bfloat16()
    for t in (x, x2, res):
        t[0] = 0
    x2 = x2 if inputs == "x+x2+res" else None
    res = res if inputs != "x" else None
    w = (torch.randn(hidden, device="cuda") * 0.2).float()

    normed, res_out, q, s, linear_s = add_rms_norm_mxfp8_quant(
        x, x2, res, w, EPS, weight_offset, store_normed=True
    )
    _, _, q_nostore, s_nostore, _ = add_rms_norm_mxfp8_quant(
        x, x2, res, w, EPS, weight_offset, store_normed=False
    )
    assert linear_s.shape == (0, hidden // 32)

    # The quant stage is the standalone swizzled quant of the bf16 output,
    # including the zero-filled 128-row scale padding.
    q_ref, s_ref = torch.ops.vllm.mxfp8_quantize(normed, True, 0)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s, s_ref)
    assert torch.equal(q_nostore.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(s_nostore, s)

    # The sum is taken in fp32 and rounded to bf16 once, as the fused IR norm.
    total = x.float()
    for t in (x2, res):
        if t is not None:
            total = total + t.float()
    if res is None:
        assert res_out.shape == (0, hidden)
    else:
        assert torch.equal(res_out, total.bfloat16())

    variance = total.pow(2).mean(dim=-1, keepdim=True)
    ref = (total * torch.rsqrt(variance + EPS) * (w + weight_offset)).bfloat16()
    # Only the fp32 reduction order differs: at most 1 bf16 ulp.
    torch.testing.assert_close(normed, ref, rtol=2**-7, atol=0)


@requires_mxfp8_quantize
@pytest.mark.parametrize("num_tokens", [1, 300, 512, 5000])
@torch.inference_mode()
def test_linear_scales(num_tokens: int) -> None:
    """The dual-layout output of the post-attention norm (MoE input)."""
    from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
        add_rms_norm_mxfp8_quant,
    )

    hidden = 2048
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden, device="cuda").bfloat16()
    res = (torch.randn(num_tokens, hidden, device="cuda") * 4).bfloat16()
    w = (torch.randn(hidden, device="cuda") * 0.2).bfloat16()

    out = add_rms_norm_mxfp8_quant(
        x, None, res, w, EPS, 1.0, True, store_linear_scales=True
    )
    normed, _, q, s, linear_s = out
    # Same e4m3 bytes; scales in both of FlashInfer's layouts.
    q_ref, s_ref = torch.ops.vllm.mxfp8_quantize(normed, True, 0)
    q_lin, s_lin = torch.ops.vllm.mxfp8_quantize(normed, False, 0)
    assert _bitwise(q, q_ref) and _bitwise(q, q_lin)
    assert torch.equal(s, s_ref)
    assert torch.equal(linear_s, s_lin.view(num_tokens, hidden // 32))
    # The swizzled outputs do not depend on the extra store.
    default = add_rms_norm_mxfp8_quant(x, None, res, w, EPS, 1.0, True)
    assert all(_bitwise(a, b) for a, b in zip(default[:4], out[:4]))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("num_tokens", [1, 300, 5000])
@pytest.mark.parametrize("store_normed", [False, True])
@torch.inference_mode()
def test_shared_expert_gate(num_tokens: int, store_normed: bool) -> None:
    """The in-kernel gate equals the eager ``sigmoid(g) * x2`` it replaces."""
    from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
        add_rms_norm_mxfp8_quant,
    )

    hidden = 2048
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden, device="cuda").bfloat16()
    x2 = (torch.randn(num_tokens, hidden, device="cuda") * 0.5).bfloat16()
    res = (torch.randn(num_tokens, hidden, device="cuda") * 4).bfloat16()
    w = (torch.randn(hidden, device="cuda") * 0.2).bfloat16()
    # A column of a wider tensor, as the gate GEMM output may be strided.
    logits = (torch.randn(num_tokens, 16, device="cuda") * 3).bfloat16()
    gate = logits[:, 5:6]

    gated = add_rms_norm_mxfp8_quant(
        x, x2, res, w, EPS, 1.0, store_normed, x2_gate=gate
    )
    pre_gated = torch.sigmoid(gate) * x2  # bf16 sigmoid, bf16 product
    ref = add_rms_norm_mxfp8_quant(x, pre_gated, res, w, EPS, 1.0, store_normed)
    assert all(_bitwise(a, b) for a, b in zip(gated, ref))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@torch.inference_mode()
def test_shared_expert_gate_sigmoid_rounding() -> None:
    """bf16(sigmoid(g)) matches ATen for every finite bf16 ``g``."""
    from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
        add_rms_norm_mxfp8_quant,
    )

    bits = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16)
    gate = bits.view(torch.bfloat16).cuda()
    gate = gate[torch.isfinite(gate.float())].reshape(-1, 1)
    n, hidden = gate.shape[0], 32
    ones = torch.ones(n, hidden, device="cuda", dtype=torch.bfloat16)
    zeros = torch.zeros_like(ones)
    # With x = residual = 0 and x2 = 1 the residual output is bf16(sigmoid(g)).
    _, res_out, _, _, _ = add_rms_norm_mxfp8_quant(
        zeros, ones, zeros, zeros[0], EPS, 1.0, False, x2_gate=gate
    )
    assert torch.equal(res_out, torch.sigmoid(gate).expand(n, hidden))
