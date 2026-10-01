# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused SiLU-and-mul -> MXFP8 producer against the ops it replaces."""

import pytest
import torch

import vllm.model_executor.layers.fusion.silu_mul_mxfp8_quant  # noqa: F401
import vllm.model_executor.layers.quantization.utils.mxfp8_utils  # noqa: F401
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.has_device_capability(100)
        and has_flashinfer()
    ),
    reason="FlashInfer MXFP8 quantize needs SM100+",
)


# Both sides of the launch-config boundaries (128, 2048) and partial 128-row
# scale padding.
@pytest.mark.parametrize("num_tokens", [1, 32, 127, 128, 300, 2048, 2049, 5000])
@pytest.mark.parametrize("scale", [1.0, 30.0])
@torch.inference_mode()
def test_silu_mul_mxfp8_quant(num_tokens: int, scale: float) -> None:
    from vllm.model_executor.layers.activation import SiluAndMul

    # What the shared expert runs without the producer: the compiled native
    # SiluAndMul, then the swizzled MXFP8 quant of its bf16 output.
    reference_act = torch.compile(SiluAndMul.forward_native, dynamic=True)
    torch.manual_seed(0)
    # scale=30 drives exp(-a) into overflow and underflow.
    x = (torch.randn(num_tokens, 1024, device="cuda") * scale).bfloat16()

    q, s = torch.ops.vllm.silu_mul_mxfp8_quant(x)
    q_ref, s_ref = torch.ops.vllm.mxfp8_quantize(reference_act(x), True, 0)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s, s_ref)


@torch.inference_mode()
def test_maybe_fused_act_quant_mxfp8(default_vllm_config) -> None:
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.fusion.fused_act_quant import (
        maybe_fused_act_quant,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    linear = torch.nn.Module()
    linear._input_quant_key = kMxfp8Dynamic
    x = torch.randn(3, 5, 1024, device="cuda").bfloat16()

    qa = maybe_fused_act_quant(SiluAndMul(), x, linear)
    assert qa.quant_key == kMxfp8Dynamic
    assert qa.orig_shape == (3, 5, 512) and qa.orig_dtype == torch.bfloat16
    q, s = torch.ops.vllm.silu_mul_mxfp8_quant(x.view(15, 1024))
    assert torch.equal(qa.data.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(qa.scale, s)
