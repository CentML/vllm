# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_ATTN_DROP_SUPERSEDED_TAIL: drop a previous turn's partial prompt-tail
attention block once the turn that copied it (CoW) registered a deeper entry.

Each follow-up turn hits the previous prompt's tail mid-block and copies that
block. Without the flag the source keeps its partial entry and returns to the
MRU end of the free queue although nothing can resume there any more. With the
flag it loses its hashes and goes to the front of the free queue. Hits are
unchanged.
"""

import pytest

from tests.v1.core.test_mamba_pin_own_checkpoint import (
    DROP_SUPERSEDED,
    HASH,
    PIN,
    _all_free,
    _decode,
    _manager,
    _prefill,
)
from tests.v1.core.test_prefix_caching import make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash

FLAG = "VLLM_ATTN_DROP_SUPERSEDED_TAIL"
STATS = "VLLM_KV_POOL_STATS"

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def _conversation(manager, turns):
    history = list(range(1, 1000))
    nid = 70_000_000
    hits = []
    attn = manager.coordinator.single_type_managers[0]
    for turn in range(turns):
        req = make_request(f"c{turn}", list(history), HASH, sha256)
        hits.append(_prefill(manager, req))
        nid = _decode(manager, req, 4, nid)
        assert (
            not attn._cow_sources
            or turn == 0
            or req.request_id not in (attn._cow_sources)
        )
        manager.free(req)
        history += list(range(nid, nid + 157))
        nid += 157
    return hits, history


@pytest.mark.parametrize("drop", [False, True])
def test_superseded_tail(monkeypatch, drop):
    monkeypatch.setenv(PIN, "1")
    monkeypatch.setenv(DROP_SUPERSEDED, "1")
    monkeypatch.setenv(FLAG, "1" if drop else "0")
    monkeypatch.setenv(STATS, "60")
    manager = _manager(num_blocks=200, num_spec=3)
    attn = manager.coordinator.single_type_managers[0]
    assert attn.track_superseded_tail
    hits, history = _conversation(manager, 6)
    # Follow-up turns that resumed mid-block (CoW) superseded their source.
    assert all(h > 0 for h in hits[1:])
    num_cow = sum(1 for h in hits[1:] if h % 64)
    assert num_cow >= 3
    assert attn.num_superseded_tails == num_cow
    if drop:
        assert attn.num_superseded_tails_dropped == num_cow
        assert attn.count_resident_superseded_tails() == 0
    else:
        assert attn.num_superseded_tails_dropped == 0
        assert attn.count_resident_superseded_tails() == num_cow
    _all_free(manager)
    manager.log_pool_stats()
    # The next turn still resumes past the latest hit.
    req = make_request("next", history + [3] * 50, HASH, sha256)
    assert manager.get_computed_blocks(req)[1] > hits[-1]


def test_same_hits_with_and_without_flag(monkeypatch):
    monkeypatch.setenv(PIN, "1")
    monkeypatch.setenv(DROP_SUPERSEDED, "1")
    results = []
    for flag in ("0", "1"):
        monkeypatch.setenv(FLAG, flag)
        results.append(_conversation(_manager(num_blocks=200, num_spec=3), 5)[0])
    assert results[0] == results[1]


def _spy_cow(attn, on_cow=None):
    """Record (and optionally act on) every CoW source of the attention group."""
    sources = []
    orig = attn._apply_cow

    def spy(request_id, block_idx, source_block, cow_block):
        sources.append((source_block, source_block.block_hash))
        orig(request_id, block_idx, source_block, cow_block)
        if on_cow is not None:
            on_cow(source_block)

    attn._apply_cow = spy
    return sources


def test_dropped_tail_goes_to_queue_front(monkeypatch):
    monkeypatch.setenv(PIN, "1")
    monkeypatch.setenv(DROP_SUPERSEDED, "1")
    monkeypatch.setenv(FLAG, "1")
    manager = _manager(num_blocks=200, num_spec=0)
    attn = manager.coordinator.single_type_managers[0]
    sources = _spy_cow(attn)
    queue = manager.block_pool.free_block_queue
    prepended = []
    orig_prepend = queue.prepend_n

    def prepend_spy(blocks):
        prepended.extend(blocks)
        orig_prepend(blocks)

    queue.prepend_n = prepend_spy
    turn1 = list(range(1, 1000))
    req1 = make_request("t1", turn1, HASH, sha256)
    _prefill(manager, req1)
    nid = _decode(manager, req1, 4, 80_000_000)
    manager.free(req1)
    req2 = make_request("t2", turn1 + list(range(7000, 7200)), HASH, sha256)
    _prefill(manager, req2)
    # The CoW retention is released after the step; a later step drops it.
    _decode(manager, req2, 1, nid)
    assert len(sources) == 1
    source, source_hash = sources[0]
    assert req2.request_id not in attn._cow_sources
    assert attn.num_superseded_tails_dropped == 1
    assert source.block_hash != source_hash
    assert source in prepended
    manager.free(req2)
    _all_free(manager)


def test_sibling_hold_keeps_source(monkeypatch):
    """A source another request still references is not dropped."""
    monkeypatch.setenv(PIN, "1")
    monkeypatch.setenv(DROP_SUPERSEDED, "1")
    monkeypatch.setenv(FLAG, "1")
    manager = _manager(num_blocks=300, num_spec=0)
    attn = manager.coordinator.single_type_managers[0]
    held = []

    def sibling_hit(source_block):
        # A sibling hits the same prefix entry while the CoW is in flight.
        manager.block_pool.touch([source_block])
        held.append(source_block)

    sources = _spy_cow(attn, sibling_hit)
    turn1 = list(range(1, 1000))
    req1 = make_request("t1", turn1, HASH, sha256)
    _prefill(manager, req1)
    manager.free(req1)
    req2 = make_request("t2", turn1 + list(range(7000, 7200)), HASH, sha256)
    _prefill(manager, req2)
    _decode(manager, req2, 2, 90_000_000)
    manager.free(req2)
    source, source_hash = sources[0]
    assert source.block_hash == source_hash
    assert attn.num_superseded_tails_dropped == 0
    assert not attn._cow_sources
    manager.block_pool.free_blocks(held)
    _all_free(manager)
