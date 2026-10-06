# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exactness checks for the sub-chunk candidate threshold of the split-row
top-k path (vllm/v1/worker/gpu/sample/topk_topp_subchunk.py)."""

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
@pytest.mark.parametrize("clustered", [False, True])
def test_subchunk_top_k_matches_reference(batch: int, clustered: bool):
    from vllm.v1.worker.gpu.sample import topk_topp_subchunk as sc

    gen = torch.Generator(device=DEVICE).manual_seed(batch + 1000 * clustered)
    # fp32 randn: ties at the k-th value are practically impossible.
    logits = torch.randn(batch, VOCAB, device=DEVICE, generator=gen) * 3
    if clustered:
        # large logits concentrated in the low-id chunks (BPE-like), the case
        # where the chunk-max threshold lets many candidates through
        logits[:, :8192] += 12.0
    k = torch.randint(1, 49, (batch,), device=DEVICE, generator=gen)
    kmax = int(k.max())

    ref = _topk_reference(logits, k)
    out = sc.fast_top_k_top_p(logits.clone(), k, None, kmax)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 5, 64])
def test_subchunk_fused_prep_without_penalties(batch: int):
    from vllm.v1.worker.gpu.sample import topk_topp_subchunk as sc

    gen = torch.Generator(device=DEVICE).manual_seed(100 + batch)
    logits = (torch.randn(batch, VOCAB, device=DEVICE, generator=gen) * 3).to(
        torch.bfloat16
    )
    prepped = sc.fused_prep(logits, None, None, None, None)
    torch.testing.assert_close(prepped, logits.float(), rtol=0, atol=0)

    k = torch.randint(1, 49, (batch,), device=DEVICE, generator=gen)
    kmax = int(k.max())
    fused = sc.fast_top_k_top_p(prepped, k, None, kmax, cmax_ready=True)
    unfused = sc.fast_top_k_top_p(logits.float(), k, None, kmax)
    torch.testing.assert_close(fused, unfused, rtol=0, atol=0)
