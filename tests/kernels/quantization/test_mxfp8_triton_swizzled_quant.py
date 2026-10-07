# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MXFP8_TRITON_QUANT: the Triton BF16 -> MXFP8 quant with F8_128x4 swizzled
scales must be bit-identical to FlashInfer's cute-dsl kernel (values and the
whole scale buffer, padding rows included)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import mxfp8_utils as mu
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.has_device_capability(100)
        and has_flashinfer()
    ),
    reason="needs SM100+ and FlashInfer",
)


@pytest.mark.parametrize("k", [1280, 5120, 8192])
@pytest.mark.parametrize("m", [1, 130, 1800])
@pytest.mark.parametrize("kind", ["randn", "scaled", "special"])
def test_triton_swizzled_quant_matches_flashinfer(k, m, kind):
    from flashinfer import mxfp8_quantize

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(m, k, device="cuda", generator=g)
    if kind == "scaled":
        e = torch.randint(-40, 40, (m, k // 32, 1), device="cuda", generator=g).float()
        x = (x.view(m, k // 32, 32) * torch.exp2(e)).view(m, k)
    elif kind == "special":
        x = x.view(m, k // 32, 32)
        sel = torch.rand(m, k // 32, 1, device="cuda", generator=g)
        x = torch.where(sel < 0.1, torch.zeros_like(x), x)
        x = torch.where(sel > 0.9, x * 1e-38, x)
        x = torch.where((sel > 0.5) & (sel < 0.55), x * 3e38, x)
        x = x.view(m, k)
        x[0, :4] = torch.tensor([float("inf"), -float("inf"), float("nan"), -0.0])
    x = x.bfloat16()
    q_ref, s_ref = mxfp8_quantize(
        x, is_sf_swizzled_layout=True, alignment=32, backend="cute-dsl"
    )
    q, s = mu.mxfp8_quantize_swizzled_triton(x)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert s.shape == s_ref.shape and torch.equal(s, s_ref)


@pytest.mark.parametrize("d", [256, 2304])
@pytest.mark.parametrize("m", [1, 130, 2304])
@pytest.mark.parametrize("limit", [10.0, None])
def test_fused_silu_clamp_mxfp8_quant_matches_unfused(d, m, limit):
    """VLLM_MXFP8_FUSED_SWIGLU_QUANT: the one-kernel SwiGLU(+clamp) + MXFP8 quant
    equals the fp32 single-rounding activation (forward_native as torch.compile
    emits it) followed by mxfp8_quantize_swizzled_triton, bitwise."""
    from vllm.model_executor.layers.fusion import fused_act_quant
    from vllm.model_executor.layers.fusion.silu_clamp_mxfp8_quant import (
        silu_and_mul_clamp_mxfp8_quant,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    # ~N(0, 4): a few percent of the gate / up values exceed the clamp.
    x = (torch.randn(m, 2 * d, device="cuda", generator=g) * 4.0).bfloat16()
    gate, up = x[:, :d].float(), x[:, d:].float()
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    act = (gate * torch.sigmoid(gate) * up).bfloat16()
    q_ref, s_ref = mu.mxfp8_quantize_swizzled_triton(act)

    q, s = silu_and_mul_clamp_mxfp8_quant(x, limit)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert s.shape == s_ref.shape and torch.equal(s, s_ref)

    if limit is not None:
        # The registered producer reads the SiluAndMulWithClamp parameters.
        act_fn = SimpleNamespace(swiglu_limit=limit, alpha=1.0, beta=0.0)
        out = fused_act_quant._silu_and_mul_clamp_mxfp8_dynamic(act_fn, x, None)
        assert out.quant_key == kMxfp8Dynamic and out.orig_shape == (m, d)
        assert torch.equal(out.data.view(torch.uint8), q_ref.view(torch.uint8))
        assert torch.equal(out.scale, s_ref)
