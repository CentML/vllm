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
from vllm.v1.core.kv_cache_utils import get_group_id, init_none_hash

MAMBA_COW = "VLLM_MAMBA_DROP_SUPERSEDED_COW"

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
        monkeypatch.setenv(MAMBA_COW, flag)
        results.append(_conversation(_manager(num_blocks=200, num_spec=3), 5)[0])
    assert results[0] == results[1]


def _free_cached(manager, gid):
    return sum(
        1
        for b in manager.block_pool.blocks
        if not b.is_null
        and b.ref_cnt == 0
        and b.block_hash is not None
        and get_group_id(b.block_hash) == gid
    )


def test_mamba_superseded_cow_source(monkeypatch):
    """VLLM_MAMBA_DROP_SUPERSEDED_COW: a "prev" checkpoint consumed through a
    partial hit (CoW) is dropped once free; without it, it stays cached.
    """
    monkeypatch.setenv(PIN, "1")
    monkeypatch.setenv(DROP_SUPERSEDED, "1")
    cached = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(MAMBA_COW, flag)
        manager = _manager(num_blocks=200, num_spec=3)
        hits, history = _conversation(manager, 6)
        _all_free(manager)
        mambas = manager.coordinator.single_type_managers[1:]
        for m in mambas:
            assert m.num_superseded_cow >= 1
            if flag == "1":
                assert m.num_superseded_cow_dropped == m.num_superseded_cow
                assert not m._pending_superseded
            else:
                assert m.num_superseded_cow_dropped == 0
        cached[flag] = [_free_cached(manager, m.kv_cache_group_id) for m in mambas]
        num_cow = mambas[0].num_superseded_cow
        # The next turn still resumes past the latest hit.
        req = make_request("next", history + [3] * 50, HASH, sha256)
        assert manager.get_computed_blocks(req)[1] > hits[-1]
    assert all(off - on == num_cow for off, on in zip(cached["0"], cached["1"]))


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


def test_checkpoint_export_turns_drop_cow_sources(monkeypatch):
    """Recipe configuration (GDN tail export, block 2176, MTP k=4, one-chunk
    prefills): every follow-up turn supersedes its prev checkpoint in the CoW
    step, so only the CoW branch can drop it.
    """
    import random

    from tests.v1.core import test_mamba_checkpointed_chunk_cache as chunk

    hits = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(PIN, "1")
        monkeypatch.setenv(DROP_SUPERSEDED, "1")
        monkeypatch.setenv(MAMBA_COW, flag)
        monkeypatch.setenv(FLAG, flag)
        manager = chunk._make_manager(num_spec=4)
        rng = random.Random(1)
        history = chunk._tokens(rng, 20000)
        prompts, arrival = {}, {}
        for turn in range(4):
            prompts[f"c{turn}"] = list(history)
            arrival[f"c{turn}"] = 3 * turn
            history += chunk._tokens(rng, 700)
        trace = chunk._run(
            manager, prompts, free_done=set(prompts), arrival_step=arrival
        )
        hits[flag] = [trace.hits[rid][0] for rid in prompts]
        for mgr in chunk._mamba_managers(manager):
            assert mgr.num_superseded_cow == 3
            assert mgr.num_superseded_cow_dropped == (3 if flag == "1" else 0)
        attn = manager.coordinator.single_type_managers[-1]
        assert attn.num_superseded_tails_dropped == (
            attn.num_superseded_tails if flag == "1" else 0
        )
    assert hits["0"] == hits["1"] and all(h > 0 for h in hits["0"][1:])
