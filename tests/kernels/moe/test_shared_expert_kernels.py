# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared-expert gate kernels (fused_moe/shared_expert_kernels.py).

Run `pytest tests/kernels/moe/test_shared_expert_kernels.py`.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.shared_expert_kernels import (
    seg_gemv_scale_,
    seg_scale_,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton CUDA kernels"
)


@pytest.mark.parametrize("M", [1, 7, 804, 2048])
@pytest.mark.parametrize("N", [256, 2048])
def test_seg_scale_matches_torch_bitwise(M: int, N: int):
    torch.manual_seed(0)
    out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    g = (torch.randn(M, 1, device="cuda") * 4).to(torch.bfloat16)
    ref = F.sigmoid(g) * out
    seg_scale_(out, g)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("M", [1, 804, 2048])
def test_seg_gemv_scale_close_to_torch(M: int):
    torch.manual_seed(0)
    K, N = 2048, 2048
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(1, K, device="cuda") * 0.02).to(torch.bfloat16)
    out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    ref = F.sigmoid(F.linear(x, w)) * out
    seg_gemv_scale_(x, w, out)
    # g is accumulated in a different fp32 order than cuBLAS.
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
