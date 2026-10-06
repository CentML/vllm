# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy-on-write KV / state block copies for partial prefix hits in ONE launch.

copy_kv_cache_blocks_inplace() visits every distinct cache storage (attention
KV, conv / SSM state pages of hybrid models) and launches, per storage, the
contiguous src/dst index copies and one row-copy kernel. With many storages
this is a long chain of small launches on the step's critical path.

VLLM_FUSED_KV_BLOCK_COPY_MULTI=1 computes exactly the same (storage, src row,
dst row) byte copies, but ships the src/dst indices and a per-storage table
(int32 base address, row elements, row stride) in ONE pinned host-to-device
copy and launches ONE Triton kernel over (pair, chunk, storage). Copies of
different storages run in parallel only when that provably equals the
sequential order: every storage comes from a whole-storage view (distinct,
disjoint memory), or the per-layer views' byte intervals commute (no
destination range overlaps a source range; destination ranges are disjoint
unless identical). Anything else falls back to the per-storage path, as do its
own preconditions (non-unique destinations, blocks that are both source and
destination, unaligned layouts).

Numerics: bitwise identical (int32 word copies of the same bytes).
"""

import os

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_FUSED_KV_BLOCK_COPY_MULTI", "0") == "1"
STATS = {"calls": 0, "fused_calls": 0, "fallback_calls": 0, "fallback_reasons": {}}


# fmt: off
@triton.jit
def _copy_rows_multi_kernel(tab_ptr, idx_ptr, n_pairs, BLOCK: tl.constexpr, ITERS: tl.constexpr):  # noqa: E501
    pair = tl.program_id(0)
    chunk = tl.program_id(1)
    s = tl.program_id(2)
    base_i = tl.load(tab_ptr + s * 3)
    row_elems = tl.load(tab_ptr + s * 3 + 1)
    row_stride = tl.load(tab_ptr + s * 3 + 2)
    start = chunk.to(tl.int64) * (BLOCK * ITERS)
    if start < row_elems:
        base = base_i.to(tl.pointer_type(tl.int32))
        src = tl.load(idx_ptr + pair)
        dst = tl.load(idx_ptr + n_pairs + pair)
        s_base = base + src * row_stride
        d_base = base + dst * row_stride
        for i in tl.static_range(ITERS):
            offs = start + i * BLOCK + tl.arange(0, BLOCK)
            m = offs < row_elems
            v = tl.load(s_base + offs, mask=m)
            tl.store(d_base + offs, v, mask=m)
# fmt: on


def _commute(tab, idx):
    """True iff running every (entry, pair) row copy in parallel equals running
    them in the original order: no destination byte range overlaps any source
    byte range, and destination ranges are pairwise disjoint unless they are
    exact duplicates (same bytes from the same source).
    """
    t = np.asarray(tab, dtype=np.int64)
    base, n, st = t[:, 0:1], t[:, 1:2] * 4, t[:, 2:3] * 4
    src0 = (base + idx[None, :, 0] * st).reshape(-1)
    dst0 = (base + idx[None, :, 1] * st).reshape(-1)
    ln = np.broadcast_to(n, (t.shape[0], idx.shape[0])).reshape(-1)
    trip = np.unique(np.stack([dst0, ln, src0], 1), axis=0)
    d0, dl = trip[:, 0], trip[:, 1]
    o = np.argsort(d0, kind="stable")
    d0s, d1s = d0[o], d0[o] + dl[o]
    if d0s.size > 1 and np.any(d0s[1:] < d1s[:-1]):
        return False
    so = np.argsort(src0, kind="stable")
    s0s, s1s = src0[so], (src0 + ln)[so]
    pm = np.maximum.accumulate(s1s)
    j = np.searchsorted(s0s, d1s, side="left")
    hit = (j > 0) & (pm[np.maximum(j - 1, 0)] > d0s)
    return not bool(np.any(hit))


def _fallback(fallback, reason, *args):
    STATS["fallback_calls"] += 1
    STATS["fallback_reasons"][reason] = STATS["fallback_reasons"].get(reason, 0) + 1
    return fallback(*args)


def copy_kv_cache_blocks_inplace(
    kv_caches, num_blocks, kv_cache_block_copies, fallback
) -> None:
    """Same contract as vllm.v1.worker.utils.copy_kv_cache_blocks_inplace;
    `fallback` is the per-storage implementation.
    """
    STATS["calls"] += 1
    if not kv_cache_block_copies:
        return
    kv_caches = list(kv_caches)
    args = (kv_caches, num_blocks, kv_cache_block_copies)
    indices_np = np.array(kv_cache_block_copies, dtype=np.int64)
    if not (
        len(np.unique(indices_np[:, 1])) == len(indices_np)
        and not np.intersect1d(indices_np[:, 0], indices_np[:, 1]).size
    ):
        return _fallback(fallback, "overlap_or_dup", *args)
    seen, storages, tab = set(), set(), []
    dev = None
    for cache in kv_caches:
        key = (cache.device, cache.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        dev = cache.device
        kbpb, rem = divmod(cache.shape[0], num_blocks)
        if rem:
            return _fallback(fallback, "remainder", *args)
        storage = cache.untyped_storage()
        skey = (cache.device, storage.data_ptr())
        sbs = cache.stride(0) * cache.element_size() * kbpb
        if storage.nbytes() != num_blocks * sbs:
            # per-layer view of a shared allocation (hybrid attention/mamba
            # layout): the per-storage path copies
            # `cache.unflatten(0, (num_blocks, kbpb))` row by row; its rows go
            # into the table as well, and the copies only run in parallel after
            # the byte-interval check below proves they commute.
            try:
                rows = cache.unflatten(0, (num_blocks, kbpb)).view(num_blocks, -1)
            except RuntimeError:
                return _fallback(fallback, "view_not_viewable", *args)
            nbytes = rows.shape[1] * rows.element_size()
            S = rows.stride(0) * rows.element_size()
            if rows.stride(1) != 1 or nbytes % 4 or S % 4 or rows.data_ptr() % 16:
                return _fallback(fallback, "view_align", *args)
            tab.append((rows.data_ptr(), nbytes // 4, S // 4))
            continue
        if skey in storages:
            continue
        storages.add(skey)
        row_bytes = (
            storage.nbytes() // num_blocks
        )  # blocks = uint8 storage viewed (num_blocks, -1)
        if row_bytes % 4 or storage.data_ptr() % 16:
            return _fallback(fallback, "align", *args)
        tab.append((storage.data_ptr(), row_bytes // 4, row_bytes // 4))
    if not tab:
        return
    if not _commute(tab, indices_np):
        return _fallback(fallback, "intervals", *args)
    n = len(indices_np)
    host = np.concatenate(
        [
            np.asarray(tab, dtype=np.int64).reshape(-1),
            indices_np[:, 0],
            indices_np[:, 1],
        ]
    )
    # same pinned-host path as async_tensor_h2d (the caching host allocator
    # keeps the buffer alive until the async copy has run; no host sync)
    d = torch.from_numpy(host).pin_memory().to(dev, non_blocking=True)
    S = len(tab)
    tab_d, idx_d = d[: 3 * S], d[3 * S :]
    BLOCK, ITERS = 2048, 8
    max_row = max(t[1] for t in tab)
    grid = (n, triton.cdiv(max_row, BLOCK * ITERS), S)
    _copy_rows_multi_kernel[grid](
        tab_d, idx_d, n, BLOCK=BLOCK, ITERS=ITERS, num_warps=8
    )
    STATS["fused_calls"] += 1
    if STATS["fused_calls"] == 1:
        logger.info(
            "fused multi-storage KV block copy active (%d storages, %d pairs)", S, n
        )
    logger.debug("fused multi-storage KV block copy stats %s", STATS)


def warmup() -> None:
    """JIT-compile the copy kernel at model load (best effort; otherwise it
    compiles on first use).
    """
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return
        buf = torch.zeros(4 * 64, dtype=torch.int32, device="cuda")
        host = np.array([buf.data_ptr(), 64, 64, 0, 1], dtype=np.int64)
        d = torch.from_numpy(host).to("cuda")
        _copy_rows_multi_kernel[(1, 1, 1)](
            d[:3], d[3:], 1, BLOCK=2048, ITERS=8, num_warps=8
        )
        torch.cuda.synchronize()
        STATS["warm"] = 1
    except Exception as e:  # noqa: BLE001 - best effort; the kernel then compiles on first use
        STATS["warm_failed"] = repr(e)[:120]
    logger.info(
        "fused multi-storage KV block copy enabled (kernel warm=%s)",
        STATS.get("warm", STATS.get("warm_failed", 0)),
    )
