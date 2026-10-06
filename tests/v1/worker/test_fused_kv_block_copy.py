# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""copy_kv_cache_blocks_inplace: direct block copy kernel vs stock indexing."""

import pytest
import torch

from vllm.v1.worker import utils as worker_utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


@pytest.fixture(params=[True, False], ids=["fused", "stock"])
def fused(request, monkeypatch):
    monkeypatch.setattr(worker_utils, "_FUSED_BLOCK_COPY", request.param)
    return request.param


@pytest.mark.parametrize(
    "num_blocks,page_bytes,num_pairs", [(64, 1 << 20, 4), (96, 1 << 18, 30)]
)
def test_whole_storage_page_layout(fused, num_blocks, page_bytes, num_pairs):
    # unified hybrid cache: one storage of num_blocks x page_bytes, typed layer view
    store = torch.randint(
        0, 255, (num_blocks, page_bytes), dtype=torch.uint8, device="cuda"
    )
    perm = torch.randperm(num_blocks)[: 2 * num_pairs].view(num_pairs, 2).tolist()
    copies = [tuple(p) for p in perm]
    ref = store.clone()
    src = torch.tensor([c[0] for c in copies])
    dst = torch.tensor([c[1] for c in copies])
    ref[dst] = ref[src]
    view = store.view(torch.float32)[:, : page_bytes // 8].view(num_blocks, -1)
    worker_utils.copy_kv_cache_blocks_inplace([view], num_blocks, copies)
    torch.accelerator.synchronize()
    assert torch.equal(store, ref)


def test_per_layer_view(fused):
    num_blocks = 32
    kv = torch.randn(num_blocks, 2, 16, 8, 128, device="cuda").to(torch.bfloat16)
    big = torch.empty(
        num_blocks + 7, *kv.shape[1:], device="cuda", dtype=torch.bfloat16
    )[:num_blocks]
    big.copy_(kv)
    copies = [(1, 5), (9, 2), (20, 30)]
    ref = big.clone()
    ref[torch.tensor([5, 2, 30])] = ref[torch.tensor([1, 9, 20])]
    worker_utils.copy_kv_cache_blocks_inplace([big], num_blocks, copies)
    torch.accelerator.synchronize()
    assert torch.equal(big, ref)


def test_overlapping_src_dst_falls_back(fused):
    # a block that is both a source and a destination must keep the
    # gather-then-scatter semantics
    x = torch.arange(10 * 64, device="cuda", dtype=torch.float32).view(10, 64)
    y = x.clone()
    copies = [(1, 2), (2, 3)]
    worker_utils.copy_kv_cache_blocks_inplace([x], 10, copies)
    y[torch.tensor([2, 3])] = y[torch.tensor([1, 2])]
    torch.accelerator.synchronize()
    assert torch.equal(x, y)
