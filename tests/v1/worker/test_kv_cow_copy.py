# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise copy-on-write parity across GPU block-copy layouts and boundaries."""

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout, MambaSpec
from vllm.v1.worker import utils as worker_utils

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="direct block copies are GPU kernels"
)

NUM_BLOCKS = 24
NUM_LAYERS = 3


@pytest.fixture(autouse=True)
def _enable_copy_paths(monkeypatch):
    monkeypatch.setenv("VLLM_KV_COW_ONE_LAUNCH", "1")
    monkeypatch.setenv("VLLM_FUSED_KV_BLOCK_COPY", "1")


@pytest.fixture(autouse=True, params=[False, True], ids=["walk", "layout_cache"])
def layout_cache(request, monkeypatch):
    """Every test with and without VLLM_KV_COW_LAYOUT_CACHE."""
    monkeypatch.setattr(worker_utils, "KV_COW_LAYOUT_CACHE", request.param)
    worker_utils._cow_layout_plans.clear()
    return request.param


def _hybrid_views(
    raw,
    layout,
    attn_dtype,
    block_size,
    kernel_block_size,
    ssm_dtype=torch.float32,
    gdn_first=True,
):
    attn = FullAttentionSpec(
        block_size=block_size, num_kv_heads=2, head_size=64, dtype=attn_dtype
    )
    # GDN-like conv (bf16) + SSM (fp32 or bf16) state padded to the attention
    # page, as the hybrid allocator unifies page sizes; overlays the same bytes.
    mamba = MambaSpec(
        block_size=block_size,
        shapes=((3, 160), (2, 32, 48)),
        dtypes=(torch.bfloat16, ssm_dtype),
        page_size_padded=attn.page_size_bytes,
    )
    attn_views = dense_kv_cache_views(
        raw, attn, NUM_BLOCKS, NUM_LAYERS, layout, kernel_block_size
    )
    mamba_views = dense_kv_cache_views(raw, mamba, NUM_BLOCKS, NUM_LAYERS, layout)
    # The V2 runner hands GDN/Mamba views BEFORE the attention views that alias
    # the same bytes: exercise that order by default.
    return mamba_views + attn_views if gdn_first else attn_views + mamba_views


def _block_byte_span(cache):
    one = cache[: max(1, cache.shape[0] // NUM_BLOCKS)]
    return (
        sum((d - 1) * st for d, st in zip(one.shape, one.stride())) + 1
    ) * cache.element_size()


def _gather_copy(caches, copies):
    """Independent gather/scatter oracle, including whole-storage ownership.

    Order-independent: of the views aliasing one address, copy the one that
    covers the most bytes of a block (an attention page, not the GDN payload
    prefix that aliases it), so the reference is the whole page whatever the
    view order.
    """
    src, dst = torch.tensor(copies, dtype=torch.int64, device="cuda").T
    widest = {}
    for cache in caches:
        prev = widest.get(cache.data_ptr())
        if prev is None or _block_byte_span(cache) > _block_byte_span(prev):
            widest[cache.data_ptr()] = cache
    seen_storages = set()
    for cache in widest.values():
        kernel_blocks_per_block = cache.shape[0] // NUM_BLOCKS
        storage = cache.untyped_storage()
        block_stride = cache.stride(0) * cache.element_size() * kernel_blocks_per_block
        if storage.nbytes() == NUM_BLOCKS * block_stride:
            if storage.data_ptr() in seen_storages:
                continue
            seen_storages.add(storage.data_ptr())
            blocks = torch.empty(0, dtype=torch.uint8, device=cache.device)
            blocks.set_(storage)
            blocks = blocks.view(NUM_BLOCKS, -1)
        else:
            blocks = cache.unflatten(0, (NUM_BLOCKS, kernel_blocks_per_block))
        blocks[dst] = blocks[src]


def _run(
    make_caches,
    copies,
    attn_dtype=torch.bfloat16,
    *,
    raw_bytes=None,
):
    raw = torch.randint(
        -128,
        127,
        (raw_bytes if raw_bytes is not None else _raw_bytes(attn_dtype),),
        dtype=torch.int8,
        device="cuda",
    )
    worker_utils._cow_copy_plans.clear()
    ref_raw = raw.clone()
    worker_utils._cow_layout_plans.clear()
    worker_utils.copy_kv_cache_blocks_inplace(make_caches(raw), NUM_BLOCKS, copies)
    _gather_copy(make_caches(ref_raw), copies)
    torch.accelerator.synchronize()
    # Comparing the entire allocation also checks every untouched block, page
    # gap, and sentinel byte, not just the destination views.
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
@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gdn_first", [True, False], ids=["gdn_first", "attn_first"])
def test_hybrid_attention_and_state_views(
    layout, attn_dtype, num_pairs, ssm_dtype, gdn_first
):
    if layout == KVCacheLayout.LHBNC:
        pytest.skip("head-split rows are not one contiguous block row")

    def make(raw):
        return _hybrid_views(
            raw, layout, attn_dtype, 128, None, ssm_dtype, gdn_first=gdn_first
        )

    _run(make, _copies(num_pairs, seed=num_pairs), attn_dtype)


@pytest.mark.parametrize("layout", [KVCacheLayout.LBHNC, KVCacheLayout.LBNHC])
def test_block_size_128_split_into_kernel_blocks(layout):
    def make(raw):
        return _hybrid_views(raw, layout, torch.float8_e4m3fn, 128, 64)

    _run(make, _copies(9), torch.float8_e4m3fn)


def test_raw_storage_entry_with_views():
    # Specs without layer views hand the raw backing tensor to the runner; it
    # is copied as whole-storage rows next to the per-layer views.
    def make(raw):
        return [raw] + _hybrid_views(
            raw, KVCacheLayout.BLHNC, torch.float8_e4m3fn, 128, None
        )

    _run(make, _copies(5), torch.float8_e4m3fn)


def test_head_split_layout_falls_back():
    def make(raw):
        return _hybrid_views(raw, KVCacheLayout.LHBNC, torch.bfloat16, 128, None)

    _run(make, _copies(4))


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

    _run(make, copies)


@pytest.mark.parametrize("fused", [False, True])
def test_disabled_one_launch_keeps_safe_per_storage_copy(monkeypatch, fused):
    monkeypatch.setenv("VLLM_KV_COW_ONE_LAUNCH", "0")
    monkeypatch.setenv("VLLM_FUSED_KV_BLOCK_COPY", str(int(fused)))
    row_bytes = 32

    def make(raw):
        return _layer_outer_regions(raw, [row_bytes] * NUM_LAYERS)

    _run(
        make,
        _copies(5),
        raw_bytes=NUM_BLOCKS * NUM_LAYERS * row_bytes + 16,
    )


def test_cached_plan_does_not_keep_kv_cache_alive():
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_allocated()
    raw = torch.zeros(_raw_bytes(torch.bfloat16), dtype=torch.int8, device="cuda")
    worker_utils._cow_copy_plans.clear()
    worker_utils.copy_kv_cache_blocks_inplace(
        _hybrid_views(raw, KVCacheLayout.LBHNC, torch.bfloat16, 128, None),
        NUM_BLOCKS,
        _copies(3),
    )
    torch.accelerator.synchronize()
    del raw
    # Only the small per-layout tables stay allocated.
    assert torch.accelerator.memory_allocated() - before < _raw_bytes(torch.bfloat16)


def _layer_outer_regions(raw, row_bytes, offset=0):
    views = []
    for size in row_bytes:
        end = offset + NUM_BLOCKS * size
        views.append(raw[offset:end].view(NUM_BLOCKS, size))
        offset = end
    return views


@pytest.mark.parametrize("row_bytes", [4, 12, 20, 28])
@pytest.mark.parametrize("offset", [0, 4])
def test_four_byte_aligned_rows_copy_exactly(row_bytes, offset):
    def make(raw):
        return _layer_outer_regions(raw, [row_bytes] * NUM_LAYERS, offset)

    _run(
        make,
        _copies(7),
        raw_bytes=offset + NUM_BLOCKS * NUM_LAYERS * row_bytes + 12,
    )


@pytest.mark.parametrize("row_bytes", [[16, 32, 64], [64, 16, 32]])
def test_mixed_page_layer_outer_regions_copy_exactly(row_bytes):
    # Each region is disjoint, but the block strides differ. They need not
    # satisfy the stricter slot proof used by the one-launch plan.
    def make(raw):
        return _layer_outer_regions(raw, row_bytes)

    _run(
        make,
        _copies(6),
        raw_bytes=NUM_BLOCKS * sum(row_bytes) + 16,
    )


@pytest.mark.parametrize(
    "copies",
    [
        [KVCacheBlockCopy(1, 2), KVCacheBlockCopy(2, 3)],
        [KVCacheBlockCopy(1, 5), KVCacheBlockCopy(1, 5)],
        [KVCacheBlockCopy(1, 1), KVCacheBlockCopy(2, 3)],
    ],
)
def test_pair_conflicts_preserve_gather_scatter_semantics(copies):
    def make(raw):
        return [raw.view(NUM_BLOCKS, 32)]

    _run(
        make,
        copies,
        raw_bytes=NUM_BLOCKS * 32,
    )


def test_overlapping_physical_rows_keep_gather_then_scatter():
    def make(raw):
        return [raw.as_strided((NUM_BLOCKS, 8), (4, 1))]

    raw = torch.randint(
        -128,
        127,
        ((NUM_BLOCKS - 1) * 4 + 8,),
        dtype=torch.int8,
        device="cuda",
    )
    before = raw.clone()
    indices = torch.tensor([[1], [2]], dtype=torch.int64, device="cuda")
    assert not worker_utils._fused_copy_kv_cache_block_rows(make(raw)[0], indices)
    torch.accelerator.synchronize()
    assert torch.equal(raw, before)

    _run(
        make,
        [KVCacheBlockCopy(1, 2)],
        raw_bytes=(NUM_BLOCKS - 1) * 4 + 8,
    )


@pytest.mark.parametrize("row_bytes", [3, 5, 7])
def test_subword_rows_keep_gather_then_scatter(row_bytes):
    def make(raw):
        return [raw.view(NUM_BLOCKS, row_bytes)]

    _run(
        make,
        _copies(3),
        raw_bytes=NUM_BLOCKS * row_bytes,
    )


@pytest.mark.parametrize(
    "layout", [KVCacheLayout.LBHNC, KVCacheLayout.LBNHC, KVCacheLayout.BLHNC]
)
def test_supported_hybrid_layouts_copy_exactly(layout):
    def make(raw):
        return _hybrid_views(raw, layout, torch.bfloat16, 128, None)

    _run(make, _copies(5))


def test_layout_cache_walks_once_per_layout(layout_cache, monkeypatch):
    """With the cache, the caches are walked once per layout; a new
    allocation or another view layout is walked again and copied correctly,
    and an unsupported layout's per-storage fallback is remembered too.
    """
    if not layout_cache:
        pytest.skip("cache only")
    monkeypatch.setenv("VLLM_KV_COW_ONE_LAUNCH", "1")
    walks: list[int] = []
    real = worker_utils._cow_copy_rows

    def _counted_walk(*args):
        walks.append(1)
        return real(*args)

    monkeypatch.setattr(worker_utils, "_cow_copy_rows", _counted_walk)

    def lbhnc(raw):
        return _hybrid_views(raw, KVCacheLayout.LBHNC, torch.bfloat16, 128, None)

    def split(raw):
        return _hybrid_views(raw, KVCacheLayout.LBHNC, torch.bfloat16, 128, 64)

    def head_split(raw):
        return _hybrid_views(raw, KVCacheLayout.LHBNC, torch.bfloat16, 128, None)

    raw = torch.randint(
        -128, 127, (_raw_bytes(torch.bfloat16),), dtype=torch.int8, device="cuda"
    )

    def copy_and_check(make, raw, copies):
        ref_raw = raw.clone()
        worker_utils.copy_kv_cache_blocks_inplace(make(raw), NUM_BLOCKS, copies)
        _gather_copy(make(ref_raw), copies)
        torch.accelerator.synchronize()
        assert torch.equal(raw, ref_raw)

    # Fresh view objects of the same tensors each call (as a runner may hand
    # over): one walk.
    for seed in range(3):
        copy_and_check(lbhnc, raw, _copies(4, seed=seed))
    assert len(walks) == 1
    # Same bytes, other view layout (kernel blocks of 64): a new key.
    copy_and_check(split, raw, _copies(4, seed=5))
    assert len(walks) == 2
    # A new allocation (the cache tensors changed): walked again.
    raw2 = raw.clone()
    copy_and_check(lbhnc, raw2, _copies(6, seed=6))
    assert len(walks) == 3
    copy_and_check(lbhnc, raw2, _copies(2, seed=7))
    assert len(walks) == 3
    # Unsupported layout: the per-storage path, also from the cache.
    copy_and_check(head_split, raw, _copies(3, seed=8))
    copy_and_check(head_split, raw, _copies(3, seed=9))
    assert len(walks) == 4
    assert any(plan is None for plan in worker_utils._cow_layout_plans.values())


@pytest.mark.parametrize(
    "layout", [KVCacheLayout.LBHNC, KVCacheLayout.LBNHC, KVCacheLayout.BLHNC]
)
@pytest.mark.parametrize("one_launch", ["1", "0"], ids=["one_launch", "per_storage"])
def test_gdn_first_partial_hit_copies_whole_attention_page(
    layout, one_launch, monkeypatch
):
    """Regression for first-seen dedup: with the GDN views
    first, a copy must still move every byte of the aliased attention page,
    not only the GDN payload prefix, on both copy paths.
    """
    monkeypatch.setenv("VLLM_KV_COW_ONE_LAUNCH", one_launch)

    def make(raw):
        return _hybrid_views(raw, layout, torch.bfloat16, 128, None, gdn_first=True)

    _run(make, _copies(5, seed=11))
