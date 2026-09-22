# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical cache occupancy and bounded, observational eviction attribution."""

from collections import OrderedDict
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig


def cache_block_bytes(config: "KVCacheConfig") -> int:
    """Use the worker's shared backing allocation, not the sum of aliased views."""
    sizes = {tensor.size for tensor in config.kv_cache_tensors}
    if len(sizes) != 1 or config.num_blocks <= 0:
        raise ValueError("KV cache usage metrics require one shared backing allocation")
    (size,) = sizes
    block_bytes, remainder = divmod(size, config.num_blocks)
    if remainder or not block_bytes:
        raise ValueError("KV cache usage metrics require uniform physical blocks")
    return block_bytes


@dataclass
class KVCacheUsageStats:
    capacity_blocks: int = 0
    block_bytes: int = 0
    active_blocks: int = 0
    inactive_cached_blocks: int = 0
    free_uncached_blocks: int = 0
    history_entries: int = 0
    attribution_supported: bool = False
    evicted_blocks: int = 0
    invalidated_blocks: int = 0
    evicted_prefixes: int = 0
    history_dropped: int = 0
    recompute_requests: int = 0
    recompute_tokens: int = 0
    prefill_tokens: int = 0


class KVCacheUsageTracker:
    """Counts physical blocks once, regardless of references or hash aliases.

    Eviction history stores keys only, never KV tensors. Its size limit makes
    recomputation attribution a lower bound once old keys have been dropped.
    """

    def __init__(self, block_bytes: int, history_size: int):
        if block_bytes <= 0 or history_size < 0:
            raise ValueError("block_bytes must be positive; history_size nonnegative")
        self.block_bytes = block_bytes
        self.history_size = history_size
        self.inactive_cached: set[int] = set()
        self.evicted: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()
        self.intervals: dict[str, tuple[int, int]] = {}
        self.affected_requests: set[str] = set()
        self.pending: dict[str, int] = {}
        self.finished: set[str] = set()
        self.attribution_supported = False
        self._delta = KVCacheUsageStats()

    def on_cached(self, block: "KVCacheBlock", key: "BlockHashWithGroupId") -> None:
        self.evicted.pop(key, None)
        if block.ref_cnt == 0 and not block.is_null:
            self.inactive_cached.add(block.block_id)

    def on_released(self, block: "KVCacheBlock") -> None:
        if not block.is_null and block.ref_cnt == 0 and block.block_hash is not None:
            self.inactive_cached.add(block.block_id)

    def on_acquired(self, block: "KVCacheBlock") -> None:
        self.inactive_cached.discard(block.block_id)

    def on_removed(
        self, block: "KVCacheBlock", keys: list["BlockHashWithGroupId"]
    ) -> None:
        self.inactive_cached.discard(block.block_id)
        for key in keys:
            self.evicted.pop(key, None)

    def on_evicted(
        self, keys_without_copies: list["BlockHashWithGroupId"], capacity: bool
    ) -> None:
        if not capacity:
            self._delta.invalidated_blocks += 1
            return
        self._delta.evicted_blocks += 1
        self._delta.evicted_prefixes += len(keys_without_copies)
        for key in keys_without_copies:
            self.evicted[key] = None
            self.evicted.move_to_end(key)
            if len(self.evicted) > self.history_size:
                self.evicted.popitem(last=False)
                self._delta.history_dropped += 1

    def record_lookup(self, request_id: str, actual: int, retained: int) -> None:
        # Repeated, unsuccessful admission attempts replace the observation.
        self.intervals[request_id] = (actual, max(actual, retained))

    def scheduled_work(
        self, request_id: str, start: int, count: int, prompt_tokens: int
    ) -> tuple[int, int]:
        end = min(start + count, prompt_tokens)
        prefill = max(0, end - start)
        if prefill:
            self.pending[request_id] = self.pending.get(request_id, 0) + 1
        lo, hi = self.intervals.get(request_id, (0, 0))
        return prefill, max(0, min(end, hi) - max(start, lo))

    def completed_work(
        self, request_id: str, work: tuple[int, int], *, succeeded: bool = True
    ) -> None:
        prefill, recomputed = work
        if succeeded:
            self._delta.prefill_tokens += prefill
            self._delta.recompute_tokens += recomputed
            if recomputed and request_id not in self.affected_requests:
                self.affected_requests.add(request_id)
                self._delta.recompute_requests += 1
        self.pending[request_id] -= 1
        if self.pending[request_id] == 0:
            del self.pending[request_id]
            if request_id in self.finished:
                self.finish_request(request_id)

    def finish_request(self, request_id: str) -> None:
        self.intervals.pop(request_id, None)
        if request_id in self.pending:
            self.finished.add(request_id)
        else:
            self.finished.discard(request_id)
            self.affected_requests.discard(request_id)

    def reset(self) -> None:
        self.inactive_cached.clear()
        self.evicted.clear()
        self.intervals.clear()
        # Cumulative counter deltas must survive an explicit cache reset.

    def snapshot(self, capacity: int, free: int) -> KVCacheUsageStats:
        stats, self._delta = self._delta, KVCacheUsageStats()
        stats.capacity_blocks = capacity
        stats.block_bytes = self.block_bytes
        stats.active_blocks = capacity - free
        stats.inactive_cached_blocks = len(self.inactive_cached)
        stats.free_uncached_blocks = free - stats.inactive_cached_blocks
        stats.history_entries = len(self.evicted)
        stats.attribution_supported = self.attribution_supported
        assert stats.free_uncached_blocks >= 0
        return stats


class RetainedPrefixPool:
    """Read-only lookup overlay: live blocks plus remembered capacity evictions.

    Placeholder blocks are used only by cache-hit length calculation. They
    must never be allocated, touched, or handed to a model runner.
    """

    def __init__(self, pool: "BlockPool", tracker: KVCacheUsageTracker):
        from vllm.v1.core.kv_cache_utils import KVCacheBlock

        self.pool = pool
        self.tracker = tracker
        self.hash_block_size = pool.hash_block_size
        self.null_block = pool.null_block
        self._placeholder = KVCacheBlock(block_id=-1)

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

        blocks = []
        for group_id in kv_cache_group_ids:
            key = make_block_hash_with_group_id(block_hash, group_id)
            block = self.pool.cached_block_hash_to_block.get_one_block(key)
            if block is None:
                if key not in self.tracker.evicted:
                    return None
                block = self._placeholder
            blocks.append(block)
        return blocks


def retained_prefix_coordinator(
    coordinator: "KVCacheCoordinator", tracker: KVCacheUsageTracker
) -> "KVCacheCoordinator":
    """Reuse the engine's alignment, hybrid reconciliation and MTP-drop rules."""
    result = copy(coordinator)
    pool = RetainedPrefixPool(coordinator.block_pool, tracker)
    # This coordinator is lookup-only; lookup uses just the proxy's three
    # attributes/methods. Never pass it to allocation or model execution.
    result.block_pool = cast("BlockPool", pool)
    result.single_type_managers = tuple(
        copy(m) for m in coordinator.single_type_managers
    )
    for manager in result.single_type_managers:
        manager.block_pool = cast("BlockPool", pool)
    return result
