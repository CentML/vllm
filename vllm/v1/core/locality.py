# SPDX-License-Identifier: Apache-2.0
"""Opt-in locality-domain KV/state placement for SM107.

VLLM_LOCALITY_SPLIT=1 splits the KV cache backing allocation so that, for every layer view, blocks
[0, num_blocks // 2) live in locality domain 0 and [num_blocks // 2, num_blocks) in domain 1
(vllm/v1/worker/locality_kv.py). The scheduler side keeps one free queue per domain and assigns each
request a sticky domain, so all its new blocks (attention KV pages and, in mamba align mode, GDN
conv/ssm state slots, which are KV blocks of the same buffer) come from that domain's half.

Correctness never depends on placement: a request may hold blocks of both domains (prefix-cache hits,
or fallback when its domain is full). Only locality-aware kernels read the domain (block_id >= half).
Exact: no kernel math changes.
"""

import os
from collections.abc import Iterator

from vllm import envs
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


def enabled() -> bool:
    return envs.VLLM_LOCALITY_SPLIT


def num_domains() -> int:
    return 2


def boundary(kv_cache_config) -> int:
    """Block-granular domain boundary shared with the worker (locality_kv.plan_split; no GPU)."""
    from vllm.v1.worker.locality_kv import ppd_block_of

    return ppd_block_of(kv_cache_config)


def domain_of(block_id: int, half: int) -> int:
    return 1 if block_id >= half else 0


class DomainFreeKVCacheBlockQueue:
    """Drop-in for FreeKVCacheBlockQueue with one LRU queue per locality domain.

    ``preferred`` (set per request by the KV cache manager) selects the queue that
    popleft/popleft_n drain first; the other queue is the fallback. Frees route by
    block id. ``num_free_blocks`` is the total (admission stays global).
    """

    def __init__(self, blocks: list[KVCacheBlock], half: int) -> None:
        self.half = half
        self.queues = (
            FreeKVCacheBlockQueue([b for b in blocks if b.block_id < half]),
            FreeKVCacheBlockQueue([b for b in blocks if b.block_id >= half]),
        )
        self.preferred = 0
        # stats: blocks handed out from the preferred vs the other domain
        self.num_popped_local = 0
        self.num_popped_fallback = 0

    # -- helpers -----------------------------------------------------------------
    def _q(self, block: KVCacheBlock) -> FreeKVCacheBlockQueue:
        return self.queues[1 if block.block_id >= self.half else 0]

    @property
    def num_free_blocks(self) -> int:
        return self.queues[0].num_free_blocks + self.queues[1].num_free_blocks

    def num_free_in(self, domain: int) -> int:
        return self.queues[domain].num_free_blocks

    # -- FreeKVCacheBlockQueue API -----------------------------------------------
    def popleft(self) -> KVCacheBlock:
        q = self.queues[self.preferred]
        if q.num_free_blocks == 0:
            q = self.queues[1 - self.preferred]
            self.num_popped_fallback += 1
        else:
            self.num_popped_local += 1
        return q.popleft()

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        p = self.queues[self.preferred]
        k = min(n, p.num_free_blocks)
        ret = p.popleft_n(k)
        if k < n:
            ret += self.queues[1 - self.preferred].popleft_n(n - k)
        self.num_popped_local += k
        self.num_popped_fallback += n - k
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        self._q(block).remove(block)

    def append(self, block: KVCacheBlock) -> None:
        self._q(block).append(block)

    def _split(self, blocks: list[KVCacheBlock]) -> tuple[list[KVCacheBlock], list[KVCacheBlock]]:
        lo: list[KVCacheBlock] = []
        hi: list[KVCacheBlock] = []
        for b in blocks:
            (hi if b.block_id >= self.half else lo).append(b)
        return lo, hi

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        lo, hi = self._split(blocks)
        self.queues[0].prepend_n(lo)
        self.queues[1].prepend_n(hi)

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        lo, hi = self._split(blocks)
        self.queues[0].append_n(lo)
        self.queues[1].append_n(hi)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        return self.queues[0].get_all_free_blocks() + self.queues[1].get_all_free_blocks()

    def iter_blocks_after(self, cursor) -> Iterator[KVCacheBlock]:  # simple_kv_offload: unsupported
        raise NotImplementedError("VLLM_LOCALITY_SPLIT is not supported with simple KV offload")


def domain_weights() -> tuple[float, float]:
    """Relative domain weights for capacity balancing; equal by default."""
    v = os.environ.get("VLLM_LOCALITY_DOMAIN_WEIGHTS", "1,1").split(",")
    w0, w1 = float(v[0]), float(v[1])
    assert w0 > 0 and w1 > 0, "VLLM_LOCALITY_DOMAIN_WEIGHTS must be two positive numbers"
    return w0, w1


class DomainAssigner:
    """Sticky per-request domain. New requests go to the domain holding most of their prefix-cache hit
    (a hit saves more than locality gains); otherwise to the domain with the lower used-blocks per unit of
    compute (used / physical SMs), which balances the KV + state bytes each die's SMs stream per step;
    ties alternate."""

    def __init__(self, queue: DomainFreeKVCacheBlockQueue) -> None:
        self.queue = queue
        self.domain: dict[str, int] = {}
        self._rr = 0
        self.assigned = [0, 0]
        self.weights = domain_weights()
        # created before the null block (id 0) is popped: domain sizes are the full halves
        self.size = [queue.half, queue.queues[1].num_free_blocks]

    def assign(self, request_id: str, hit_block_ids: list[int] | None) -> int:
        d = self.domain.get(request_id)
        if d is not None:
            return d
        half = self.queue.half
        hits = [b for b in (hit_block_ids or []) if b > 0]
        if hits:
            n1 = sum(1 for b in hits if b >= half)
            d = 1 if 2 * n1 > len(hits) else 0
        else:
            u0 = (self.size[0] - self.queue.num_free_in(0)) / self.weights[0]
            u1 = (self.size[1] - self.queue.num_free_in(1)) / self.weights[1]
            if u0 != u1:
                d = 0 if u0 < u1 else 1
            else:
                d = self._rr
                self._rr ^= 1
        self.domain[request_id] = d
        self.assigned[d] += 1
        return d

    def release(self, request_id: str) -> None:
        self.domain.pop(request_id, None)
