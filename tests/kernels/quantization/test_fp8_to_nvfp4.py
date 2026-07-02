# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct fp8 -> NVFP4 quantization must be byte-identical to the production
``scaled_fp4_quant(flashinfer-cutlass)`` path on the bf16 upcast of the same
fp8 input (an fp8 -> bf16 upcast is lossless, so block amax and e2m1 codes
match exactly)."""

import pytest
import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.quantization.utils.fp8_to_nvfp4 import (
    fp8_to_nvfp4_supported_shape,
    quantize_fp8_to_nvfp4,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="NVFP4 requires compute capability of 10 or above.",
        allow_module_level=True,
    )
if not has_flashinfer():
    pytest.skip(reason="flashinfer required.", allow_module_level=True)

SHAPES = [(128, 2048), (256, 4096), (1024, 8192)]
SEEDS = [42]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_fp8_to_nvfp4_matches_scaled_fp4_quant(
    shape: tuple[int, int], seed: int, strided: bool
) -> None:
    m, k = shape
    assert fp8_to_nvfp4_supported_shape(m, k, 0)
    set_random_seed(seed)
    fp8 = current_platform.fp8_dtype()

    if strided:
        # The consumer may hand a strided view (e.g. under torch.compile).
        buf = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16).to(fp8)
        x_fp8 = buf[:, :k]
        assert not x_fp8.is_contiguous()
    else:
        x_fp8 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(fp8)
    x_bf16 = x_fp8.to(torch.bfloat16)

    global_scale = (x_bf16.abs().amax().float() / (6.0 * 448.0)).clamp(min=1e-8)
    global_scale_inv = (1.0 / global_scale).to(torch.float32)

    ref_packed, ref_scale = scaled_fp4_quant(
        x_bf16.contiguous(),
        global_scale_inv,
        is_sf_swizzled_layout=True,
        backend="flashinfer-cutlass",
    )
    packed, scale = quantize_fp8_to_nvfp4(x_fp8, global_scale_inv)

    assert torch.equal(packed.view(torch.uint8), ref_packed.view(torch.uint8))
    assert torch.equal(scale.view(torch.uint8), ref_scale.view(torch.uint8))


def test_fp8_to_nvfp4_supported_shape() -> None:
    assert fp8_to_nvfp4_supported_shape(128, 64, 0)
    assert not fp8_to_nvfp4_supported_shape(127, 64, 0)  # m % 128
    assert not fp8_to_nvfp4_supported_shape(128, 48, 0)  # k % 64
    assert not fp8_to_nvfp4_supported_shape(128, 64, 8)  # padded weights
