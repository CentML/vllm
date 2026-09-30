# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared-expert gate kernels (fused_moe/shared_expert_kernels.py).

Run `pytest tests/kernels/moe/test_shared_expert_kernels.py`.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe import shared_expert_kernels
from vllm.model_executor.layers.fused_moe.shared_expert_kernels import (
    seg_gemv_scale_,
    seg_route_fold,
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


def _torch_route_fold(logits: torch.Tensor, E: int, K: int):
    probs = torch.softmax(logits[:, :E].float(), dim=-1)
    w, ids = probs.topk(K, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    shared = torch.sigmoid(logits[:, E : E + 1].float())
    ids = torch.cat([ids.int(), torch.full_like(ids[:, :1], E, dtype=torch.int32)], 1)
    return ids, torch.cat([w, shared], 1)


@pytest.mark.parametrize("M", [1, 64, 804, 4096])
def test_seg_route_fold(M: int, monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(0)
    E, K = 256, 8
    x = torch.randn(M, 2048, device="cuda", dtype=torch.bfloat16)
    w264 = (torch.randn(E + 8, 2048, device="cuda") * 0.02).to(torch.bfloat16)
    w264[E + 1 :] = 0
    logits = F.linear(x, w264)

    ids, wts = seg_route_fold(logits, E, K)
    assert ids.shape == (M, K + 1) and ids.dtype == torch.int32
    assert wts.shape == (M, K + 1) and wts.dtype == torch.bfloat16

    # The packed-key kernel (bf16 logits) matches the plain reference kernel.
    monkeypatch.setattr(shared_expert_kernels, "_ROUTE_PLAIN", True)
    ids_ref, wts_ref = seg_route_fold(logits, E, K)
    torch.testing.assert_close(ids, ids_ref, atol=0, rtol=0)
    torch.testing.assert_close(wts, wts_ref, atol=0, rtol=0)

    # Same expert set and weights as a torch softmax -> top-k -> renorm.
    ids_t, wts_t = _torch_route_fold(logits, E, K)
    assert torch.equal(ids[:, K], ids_t[:, K])
    order = ids[:, :K].argsort(1)
    order_t = ids_t[:, :K].argsort(1)
    assert torch.equal(ids[:, :K].gather(1, order), ids_t[:, :K].gather(1, order_t))
    torch.testing.assert_close(
        wts[:, :K].float().gather(1, order),
        wts_t[:, :K].gather(1, order_t).to(torch.bfloat16).float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        wts[:, K].float(), wts_t[:, K].to(torch.bfloat16).float(), atol=1e-2, rtol=0
    )
