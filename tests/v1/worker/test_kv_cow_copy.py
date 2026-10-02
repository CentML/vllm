# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch copy-on-write block copy vs the per-storage gather/scatter path."""

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout, MambaSpec
from vllm.v1.worker import utils as worker_utils

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the one-launch copy is a GPU kernel"
)

NUM_BLOCKS = 24
NUM_LAYERS = 3


def _hybrid_views(raw, layout, attn_dtype, block_size, kernel_block_size):
    attn = FullAttentionSpec(
        block_size=block_size, num_kv_heads=2, head_size=64, dtype=attn_dtype
    )
    # GDN-like conv (bf16) + SSM (fp32) state padded to the attention page, as
    # the hybrid allocator unifies page sizes; overlays the same bytes.
    mamba = MambaSpec(
        block_size=block_size,
        shapes=((3, 160), (2, 32, 48)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=attn.page_size_bytes,
    )
    views = dense_kv_cache_views(
        raw, attn, NUM_BLOCKS, NUM_LAYERS, layout, kernel_block_size
    )
    views += dense_kv_cache_views(raw, mamba, NUM_BLOCKS, NUM_LAYERS, layout)
    return views


def _run(make_caches, copies, expect_fused, attn_dtype=torch.bfloat16):
    raw = torch.randint(
        -128, 127, (_raw_bytes(attn_dtype),), dtype=torch.int8, device="cuda"
    )
    ref_raw = raw.clone()
    worker_utils._cow_copy_plans.clear()
    worker_utils.copy_kv_cache_blocks_inplace(make_caches(raw), NUM_BLOCKS, copies)
    assert any(p is not None for p in worker_utils._cow_copy_plans.values()) == (
        expect_fused
    )
    worker_utils._copy_kv_cache_blocks_inplace_per_storage(
        make_caches(ref_raw), NUM_BLOCKS, np.array(copies, dtype=np.int64)
    )
    torch.accelerator.synchronize()
    assert torch.equal(raw, ref_raw)
    return raw


def _raw_bytes(attn_dtype):
    page = FullAttentionSpec(
        block_size=128, num_kv_heads=2, head_size=64, dtype=attn_dtype
    ).page_size_bytes
    return NUM_BLOCKS * NUM_LAYERS * page


def _copies(num_pairs, seed=0):
    perm = np.random.default_rng(seed).permutation(NUM_BLOCKS)[: 2 * num_pairs]
    return [KVCacheBlockCopy(int(s), int(d)) for s, d in perm.reshape(-1, 2)]


@pytest.mark.parametrize("layout", list(KVCacheLayout))
@pytest.mark.parametrize("attn_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("num_pairs", [1, 7, 12])
def test_hybrid_attention_and_state_views(layout, attn_dtype, num_pairs):
    if layout == KVCacheLayout.LHBNC:
        pytest.skip("head-split rows are not one contiguous block row")

    def make(raw):
        return _hybrid_views(raw, layout, attn_dtype, 128, None)

    _run(make, _copies(num_pairs, seed=num_pairs), True, attn_dtype)


@pytest.mark.parametrize("layout", [KVCacheLayout.LBHNC, KVCacheLayout.LBNHC])
def test_block_size_128_split_into_kernel_blocks(layout):
    def make(raw):
        return _hybrid_views(raw, layout, torch.float8_e4m3fn, 128, 64)

    _run(make, _copies(9), True, torch.float8_e4m3fn)


def test_raw_storage_entry_with_views():
    # Specs without layer views hand the raw backing tensor to the runner; it
    # is copied as whole-storage rows next to the per-layer views.
    def make(raw):
        return [raw] + _hybrid_views(
            raw, KVCacheLayout.BLHNC, torch.float8_e4m3fn, 128, None
        )

    _run(make, _copies(5), True, torch.float8_e4m3fn)


def test_head_split_layout_falls_back():
    def make(raw):
        return _hybrid_views(raw, KVCacheLayout.LHBNC, torch.bfloat16, 128, None)

    _run(make, _copies(4), expect_fused=False)


@pytest.mark.parametrize(
    "copies",
    [
        # chain: block 2 is a destination and a source
        [KVCacheBlockCopy(1, 2), KVCacheBlockCopy(2, 3)],
        # duplicate destination (same source: index_put order is unspecified)
        [KVCacheBlockCopy(1, 5), KVCacheBlockCopy(4, 6), KVCacheBlockCopy(1, 5)],
    ],
)
def test_overlapping_pairs_keep_gather_then_scatter(copies):
    def make(raw):
        return _hybrid_views(raw, KVCacheLayout.LBHNC, torch.bfloat16, 128, None)

    _run(make, copies, expect_fused=False)
