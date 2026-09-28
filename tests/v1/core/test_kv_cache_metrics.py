# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import prometheus_client
import pytest

from vllm.config import ObservabilityConfig
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import (
    BlockMetricsState,
    KVCacheMetricsCollector,
)
from vllm.v1.core.kv_cache_usage_metrics import KVCacheUsageTracker, cache_block_bytes
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.stats import SchedulerStats


class TestBlockMetricsState:
    def test_init(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()
            assert state.birth_time_ns == 1000000000
            assert state.last_access_ns == 1000000000
            assert len(state.access_history) == 0

    def test_access_tracking(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        with patch("time.monotonic_ns", return_value=2000000000):
            state.record_access()

        assert state.last_access_ns == 2000000000
        assert list(state.access_history) == [2000000000]

    def test_ring_buffer_wraps_at_4(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        for i in range(5):
            t = 1000000000 + (i + 1) * 1000000000
            with patch("time.monotonic_ns", return_value=t):
                state.record_access()

        assert len(state.access_history) == 4
        assert list(state.access_history) == [
            3000000000,
            4000000000,
            5000000000,
            6000000000,
        ]

    def test_lifetime(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()
        with patch("time.monotonic_ns", return_value=6500000000):
            assert abs(state.get_lifetime_seconds() - 5.5) < 0.001

    def test_idle_time(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()
        state.last_access_ns = 2000000000
        with patch("time.monotonic_ns", return_value=5200000000):
            assert abs(state.get_idle_time_seconds() - 3.2) < 0.001

    def test_reuse_gaps(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        base = 1000000000
        for offset in [0, 1.5, 3.0, 5.5]:
            state.access_history.append(base + int(offset * 1e9))

        gaps = state.get_reuse_gaps_seconds()
        assert len(gaps) == 3
        assert gaps[0] == 1.5 and gaps[1] == 1.5 and gaps[2] == 2.5

    def test_ring_wrap_only_gives_3_gaps(self):
        # 5 accesses in size-4 buffer = 3 gaps
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        for i in range(5):
            state.access_history.append(1000000000 + i * 1000000000)

        assert len(state.get_reuse_gaps_seconds()) == 3


class TestKVCacheMetricsCollector:
    def test_sample_rate_validation(self):
        with pytest.raises(AssertionError):
            KVCacheMetricsCollector(sample_rate=-0.1)
        with pytest.raises(AssertionError):
            KVCacheMetricsCollector(sample_rate=1.5)
        with pytest.raises(AssertionError):
            KVCacheMetricsCollector(sample_rate=0.0)

    def test_sampling(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)
        assert sum(1 for _ in range(100) if c.should_sample_block()) == 100

        c = KVCacheMetricsCollector(sample_rate=0.5)
        samples = sum(1 for _ in range(1000) if c.should_sample_block())
        assert 400 < samples < 600

    def test_alloc(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        with patch("time.monotonic_ns", return_value=1000000000):
            for block in blocks:
                c.on_block_allocated(block)

        assert len(c.block_metrics) == 5

    def test_access(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)
        block = KVCacheBlock(block_id=0)

        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        for i in range(3):
            t = 1000000000 + (i + 1) * 1000000000
            with patch("time.monotonic_ns", return_value=t):
                c.on_block_accessed(block)

        assert len(c.block_metrics[0].access_history) == 3

    def test_evict_no_accesses(self):
        # lifetime should equal idle if never accessed
        c = KVCacheMetricsCollector(sample_rate=1.0)

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=6000000000):
            c.on_block_evicted(block)

        events = c.drain_events()
        assert len(events) == 1
        assert abs(events[0].lifetime_seconds - 5.0) < 0.001
        assert abs(events[0].idle_seconds - 5.0) < 0.001

    def test_evict(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=2000000000):
            c.on_block_accessed(block)
        with patch("time.monotonic_ns", return_value=3000000000):
            c.on_block_accessed(block)

        with patch("time.monotonic_ns", return_value=4000000000):
            c.on_block_evicted(block)

        events = c.drain_events()
        assert len(events) == 1
        sample = events[0]
        assert abs(sample.lifetime_seconds - 3.0) < 0.001
        assert abs(sample.idle_seconds - 1.0) < 0.001
        assert sample.reuse_gaps_seconds == (1.0,)
        assert 0 not in c.block_metrics

    def test_reset(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        with patch("time.monotonic_ns", return_value=1000000000):
            for i in range(5):
                c.on_block_allocated(KVCacheBlock(block_id=i))

        assert len(c.block_metrics) == 5
        c.reset()
        assert len(c.block_metrics) == 0

        with patch("time.monotonic_ns", return_value=2000000000):
            c.on_block_allocated(KVCacheBlock(block_id=10))
        assert 10 in c.block_metrics

    def test_huge_time_jump(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=9999999999999999):
            c.on_block_evicted(block)

        events = c.drain_events()
        assert len(events) == 1
        assert events[0].lifetime_seconds > 0


def test_kv_cache_metrics_collector_smoke() -> None:
    """Simple smoke test for KVCacheMetricsCollector on CPU."""
    collector = KVCacheMetricsCollector(sample_rate=1.0)
    block = KVCacheBlock(block_id=123)

    # Allocate at t = 1.0s.
    with patch("time.monotonic_ns", return_value=1_000_000_000):
        collector.on_block_allocated(block)

    # Access at t = 2.0s and t = 3.0s.
    with patch("time.monotonic_ns", return_value=2_000_000_000):
        collector.on_block_accessed(block)
    with patch("time.monotonic_ns", return_value=3_000_000_000):
        collector.on_block_accessed(block)

    # Evict at t = 4.0s.
    with patch("time.monotonic_ns", return_value=4_000_000_000):
        collector.on_block_evicted(block)

    events = collector.drain_events()
    assert len(events) == 1

    event = events[0]
    # Lifetime: 1.0s → 4.0s.
    assert abs(event.lifetime_seconds - 3.0) < 1e-6
    # Idle: last access at 3.0s, evicted at 4.0s.
    assert abs(event.idle_seconds - 1.0) < 1e-6
    # One reuse gap between the two accesses.
    assert event.reuse_gaps_seconds == (1.0,)


def test_bytes_use_one_backing_allocation_not_aliased_layer_views():
    config = SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[SimpleNamespace(size=4096), SimpleNamespace(size=4096)],
    )
    assert cache_block_bytes(config) == 1024


def test_occupancy_counts_physical_blocks_not_hash_aliases_or_references():
    tracker = KVCacheUsageTracker(block_bytes=1024, history_size=4)
    block = SimpleNamespace(block_id=1, ref_cnt=2, block_hash=b"a", is_null=False)
    tracker.on_cached(block, b"a")
    tracker.on_cached(block, b"alias")
    assert tracker.snapshot(3, 2).active_blocks == 1
    block.ref_cnt = 1
    tracker.on_released(block)
    assert tracker.snapshot(3, 2).inactive_cached_blocks == 0
    block.ref_cnt = 0
    tracker.on_released(block)
    stats = tracker.snapshot(3, 3)
    assert (stats.active_blocks, stats.inactive_cached_blocks) == (0, 1)
    assert stats.free_uncached_blocks == 2
    tracker.on_acquired(block)
    assert tracker.snapshot(3, 2).inactive_cached_blocks == 0


def test_eviction_is_not_recomputation_and_invalidation_is_not_capacity():
    tracker = KVCacheUsageTracker(1024, 4)
    tracker.on_evicted([b"old"], capacity=True)
    tracker.on_evicted([b"invalid"], capacity=False)
    stats = tracker.snapshot(3, 3)
    assert stats.evicted_blocks == 1
    assert stats.invalidated_blocks == 1
    assert stats.recompute_tokens == 0
    assert list(tracker.evicted) == [b"old"]


def test_history_bound_and_reset_expose_loss_without_resetting_counters():
    tracker = KVCacheUsageTracker(1024, 2)
    for key in (b"a", b"b", b"c"):
        tracker.on_evicted([key], capacity=True)
    assert list(tracker.evicted) == [b"b", b"c"]
    tracker.reset()
    stats = tracker.snapshot(3, 3)
    assert stats.history_entries == 0
    assert stats.history_dropped == 1
    assert stats.evicted_blocks == 3
    assert tracker.snapshot(3, 3).evicted_blocks == 0


def test_reprefill_counts_completed_chunks_once_and_excludes_new_input_decode():
    tracker = KVCacheUsageTracker(1024, 4)
    tracker.record_lookup("r", actual=32, retained=96)
    # Failed admission is superseded by a lookup that now hits more tokens.
    tracker.record_lookup("r", actual=48, retained=96)
    first = tracker.scheduled_work("r", 48, 32, prompt_tokens=110)
    second = tracker.scheduled_work("r", 80, 40, prompt_tokens=110)
    assert first == (32, 32)
    assert second == (30, 16)
    assert tracker.snapshot(3, 3).recompute_tokens == 0
    tracker.completed_work("r", first)
    tracker.completed_work("r", second)
    assert tracker.scheduled_work("r", 120, 4, 110) == (0, 0)
    stats = tracker.snapshot(3, 3)
    assert (stats.prefill_tokens, stats.recompute_tokens) == (62, 48)
    assert stats.recompute_requests == 1
    tracker.finish_request("r")
    assert not tracker.intervals and not tracker.affected_requests


def test_aborted_request_keeps_inflight_accounting_until_model_returns():
    tracker = KVCacheUsageTracker(1024, 4)
    tracker.record_lookup("r", 0, 64)
    chunks = [tracker.scheduled_work("r", start, 32, 64) for start in (0, 32)]
    tracker.finish_request("r")
    for chunk in chunks:
        tracker.completed_work("r", chunk)
    stats = tracker.snapshot(3, 3)
    assert stats.recompute_requests == 1
    assert stats.recompute_tokens == 64
    assert not tracker.pending and not tracker.finished
    assert not tracker.affected_requests


def test_failed_external_load_does_not_count_as_completed_prefill():
    tracker = KVCacheUsageTracker(1024, 4)
    tracker.record_lookup("r", 0, 64)
    work = tracker.scheduled_work("r", 0, 32, 64)
    tracker.completed_work("r", work, succeeded=False)
    stats = tracker.snapshot(3, 3)
    assert stats.prefill_tokens == stats.recompute_tokens == 0
    assert not tracker.pending


def test_pool_hooks_count_aliases_and_duplicate_copies_once():
    pool = BlockPool(4, enable_caching=True, hash_block_size=16)
    tracker = pool.usage_tracker = KVCacheUsageTracker(1024, 10)
    a, b, scratch = pool.get_new_blocks(3)
    key, alias = (BlockHashWithGroupId(k) for k in (b"prefix", b"alias"))
    pool._insert_block_hash(key, a, num_tokens=16)
    pool._insert_block_hash(alias, a, num_tokens=8)
    pool._insert_block_hash(key, b, num_tokens=16)
    pool.free_blocks([a, b])
    stats = tracker.snapshot(3, pool.get_num_free_blocks())
    assert (stats.active_blocks, stats.inactive_cached_blocks) == (1, 2)
    pool.touch([a])
    assert tracker.snapshot(3, pool.get_num_free_blocks()).inactive_cached_blocks == 1
    pool.free_blocks([a])
    # Queue order is now b, a; evicting b leaves the duplicate prefix in a.
    first = pool.get_new_blocks(1)
    assert first == [b]
    stats = tracker.snapshot(3, pool.get_num_free_blocks())
    assert stats.evicted_blocks == 1 and stats.evicted_prefixes == 0
    second = pool.get_new_blocks(1)
    assert second == [a]
    stats = tracker.snapshot(3, pool.get_num_free_blocks())
    assert stats.evicted_blocks == 1 and stats.evicted_prefixes == 2
    assert set(tracker.evicted) == {key, alias}
    pool.free_blocks(first + second + [scratch])
    assert pool.reset_prefix_cache()
    stats = tracker.snapshot(3, pool.get_num_free_blocks())
    assert stats.free_uncached_blocks == 3 and stats.history_entries == 0


def test_usage_stats_survive_engine_msgpack_transport():
    tracker = KVCacheUsageTracker(1024, 10)
    tracker.on_evicted([BlockHashWithGroupId(b"prefix")], capacity=True)
    stats = SchedulerStats(kv_cache_usage_stats=tracker.snapshot(3, 2))
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(stats), type=SchedulerStats)
    assert decoded.kv_cache_usage_stats == stats.kv_cache_usage_stats


def test_prometheus_exports_bytes_and_counter_deltas_per_engine():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            served_model_name="cache-test", max_model_len=128, is_diffusion=False
        ),
        observability_config=ObservabilityConfig(kv_cache_usage_metrics=True),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
    )
    logger = PrometheusStatLogger(config, [0, 1])
    tracker = KVCacheUsageTracker(1024, 4)
    tracker.on_evicted([BlockHashWithGroupId(b"old")], capacity=True)
    logger.record(
        SchedulerStats(kv_cache_usage_stats=tracker.snapshot(4, 3)),
        None,
        engine_idx=1,
    )
    # A second stats packet must not repeat the eviction delta.
    logger.record(
        SchedulerStats(kv_cache_usage_stats=tracker.snapshot(4, 4)),
        None,
        engine_idx=1,
    )
    labels = {"model_name": "cache-test", "engine": "1"}
    registry = prometheus_client.REGISTRY
    assert registry.get_sample_value("vllm:kv_cache_capacity_bytes", labels) == 4096
    assert registry.get_sample_value("vllm:kv_cache_active_bytes", labels) == 0
    assert (
        registry.get_sample_value("vllm:kv_cache_evicted_bytes_total", labels) == 1024
    )
    labels["engine"] = "0"
    assert registry.get_sample_value("vllm:kv_cache_evicted_bytes_total", labels) == 0


def test_checkpoint_invalidation_is_not_capacity_eviction():
    pool = BlockPool(3, enable_caching=True, hash_block_size=16)
    tracker = pool.usage_tracker = KVCacheUsageTracker(1024, 4)
    block = pool.get_new_blocks(1)[0]
    pool._insert_block_hash(BlockHashWithGroupId(b"old"), block, num_tokens=16)
    assert pool._maybe_evict_cached_block(block)
    stats = tracker.snapshot(2, pool.get_num_free_blocks())
    assert stats.invalidated_blocks == 1 and stats.evicted_blocks == 0
    assert not tracker.evicted


def test_pool_reset_counts_inactive_physical_blocks_not_hash_aliases():
    pool = BlockPool(4, enable_caching=True, hash_block_size=16)
    tracker = pool.usage_tracker = KVCacheUsageTracker(1024, 4)
    first, second = pool.get_new_blocks(2)
    for key, block in ((b"a", first), (b"alias", first), (b"b", second)):
        pool._insert_block_hash(BlockHashWithGroupId(key), block, num_tokens=16)
    # A rejected reset must not invalidate blocks still held by a request.
    assert not pool.reset_prefix_cache()
    assert tracker.snapshot(3, pool.get_num_free_blocks()).invalidated_blocks == 0
    pool.free_blocks([first, second])
    assert pool.reset_prefix_cache()
    stats = tracker.snapshot(3, pool.get_num_free_blocks())
    assert stats.invalidated_blocks == 2
    assert stats.evicted_blocks == stats.inactive_cached_blocks == 0
    assert stats.free_uncached_blocks == 3
    assert pool.reset_prefix_cache()
    assert tracker.snapshot(3, pool.get_num_free_blocks()).invalidated_blocks == 0
