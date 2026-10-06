# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exactness checks for the split-row sampling kernels of the V2 GPU runner:
top-k masking, the fused fp32 copy + chunk-max prep pass, and the greedy
draft argmax.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

VOCAB = 248320
DEVICE = "cuda"


def _topk_reference(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(logits, float("-inf"))
    for row in range(logits.shape[0]):
        idx = torch.topk(logits[row], int(k[row])).indices
        out[row, idx] = logits[row, idx]
    return out


@pytest.mark.parametrize("batch", [1, 5, 64])
def test_fast_top_k_matches_reference(batch: int):
    from vllm.v1.worker.gpu.sample.states import fast_top_k_top_p

    gen = torch.Generator(device=DEVICE).manual_seed(batch)
    # fp32 randn: ties at the k-th value are practically impossible.
    logits = torch.randn(batch, VOCAB, device=DEVICE, generator=gen) * 3
    k = torch.randint(1, 49, (batch,), device=DEVICE, generator=gen)
    kmax = int(k.max())

    ref = _topk_reference(logits, k)
    out = fast_top_k_top_p(logits.clone(), k, None, kmax)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 5, 64])
def test_fused_prep_without_penalties(batch: int):
    from vllm.v1.worker.gpu.sample.states import fast_top_k_top_p, fused_prep

    gen = torch.Generator(device=DEVICE).manual_seed(100 + batch)
    logits = (torch.randn(batch, VOCAB, device=DEVICE, generator=gen) * 3).to(
        torch.bfloat16
    )
    prepped = fused_prep(logits, None, None, None, None)
    torch.testing.assert_close(prepped, logits.float(), rtol=0, atol=0)

    # Ties are frequent in bf16, so compare against the unfused split-row
    # path (same tie-breaking) instead of torch.topk.
    k = torch.randint(1, 49, (batch,), device=DEVICE, generator=gen)
    kmax = int(k.max())
    fused = fast_top_k_top_p(prepped, k, None, kmax, cmax_ready=True)
    unfused = fast_top_k_top_p(logits.float(), k, None, kmax)
    torch.testing.assert_close(fused, unfused, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 7, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fast_argmax_matches_torch(batch: int, dtype: torch.dtype):
    from vllm.v1.worker.gpu.spec_decode.speculator import fast_argmax

    gen = torch.Generator(device=DEVICE).manual_seed(200 + batch)
    logits = torch.randn(batch, VOCAB, device=DEVICE, generator=gen).to(dtype)
    # Ties: a flat row and a row whose maximum appears in two chunks.
    logits[0] = 0.0
    if batch > 1:
        logits[1, 70000] = 100.0
        logits[1, 9000] = 100.0
    torch.testing.assert_close(
        fast_argmax(logits), logits.argmax(dim=-1), rtol=0, atol=0
    )
