# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_PREFIX_EXACT_JUNCTION: exact (sub-block) shared-prefix junctions.

Requests sharing a long prefix that ends inside a cache block (e.g. a common
system prompt followed by a per-request salt) only reused it up to a block
boundary (one EAGLE block below the last shared full block). With the flag, the
second sibling that shares the prefix stops its prefill where the GDN prefill
export lands on the exact junction and registers the attention entry one hash
unit above it, so later siblings hit the whole shared prefix minus one unit.

Recipe configuration (Qwen3.6-35B-A3B at C1024): block 1152, prefix-match unit
32, MTP k=4, retention 0, GDN prefill checkpoint, pin/drop flags. Every run
checks that each consumer restores exactly the state at its hit position and
that every registered Mamba hash holds the state it claims, then that freeing
all requests returns every block.
"""

import random

import pytest

from tests.v1.core import test_mamba_checkpointed_chunk_cache as chunk
from tests.v1.core.test_prefix_caching import make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

FLAG = "VLLM_PREFIX_EXACT_JUNCTION"
DROPS = (
    "VLLM_MAMBA_PIN_OWN_CKPT",
    "VLLM_MAMBA_DROP_SUPERSEDED_STATE",
    "VLLM_MAMBA_DROP_SUPERSEDED_COW",
    "VLLM_ATTN_DROP_SUPERSEDED_TAIL",
)
BLOCK = 1152
HASH = 32
# Workato-like first turns: shared tools + system prompt, then a per-request
# salt and user message.
SHARED = 32131
JUNCTION = SHARED // HASH * HASH - HASH  # 32096: one EAGLE unit below
BLOCK_HIT = (SHARED // BLOCK - 1) * BLOCK  # 29952: today's block-floored hit


@pytest.fixture(autouse=True)
def _setup(monkeypatch: pytest.MonkeyPatch):
    init_none_hash(sha256)
    monkeypatch.setattr(chunk, "BLOCK", BLOCK)


def _env(monkeypatch: pytest.MonkeyPatch, exact: bool, drops: bool = True) -> None:
    monkeypatch.setenv(FLAG, "1" if exact else "0")
    for name in DROPS:
        monkeypatch.setenv(name, "1" if drops else "0")


def _siblings(
    n: int,
    shared: int = SHARED,
    tail: int = 80,
    gap: int = 1,
    turn2_at: int | None = 8,
    seed: int = 0,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    rng = random.Random(seed)
    prefix = chunk._tokens(rng, shared)
    prompts: dict[str, list[int]] = {}
    arrival: dict[str, int] = {}
    for i in range(n):
        prompt = prefix + chunk._tokens(rng, tail + 7 * i)
        prompts[f"r{i}"] = prompt
        arrival[f"r{i}"] = gap * i
        if turn2_at is not None:
            prompts[f"t2_r{i}"] = prompt + chunk._tokens(rng, 300)
            arrival[f"t2_r{i}"] = turn2_at + gap * i
    return prompts, arrival


def _run(prompts, arrival):
    manager = chunk._make_manager(num_spec=4)
    trace = chunk._run(manager, prompts, free_done=set(prompts), arrival_step=arrival)
    hits = {rid: trace.hits[rid][0] for rid in prompts}
    chunk._check_cached_states_and_free(manager, trace)
    return manager, trace, hits


@pytest.mark.parametrize("drops", [False, True], ids=["no_drops", "recipe_drops"])
def test_siblings_hit_exact_junction(monkeypatch, drops):
    prompts, arrival = _siblings(6)
    _env(monkeypatch, exact=False, drops=drops)
    _, _, base = _run(prompts, arrival)
    _env(monkeypatch, exact=True, drops=drops)
    manager, trace, hits = _run(prompts, arrival)
    # Cold first request; the second recomputes everything (no GDN state yet)
    # and materializes the block-floored Marconi junction, as without the flag.
    assert base["r0"] == hits["r0"] == 0
    assert base["r1"] == hits["r1"] == 0
    assert [base[f"r{i}"] for i in range(2, 6)] == [BLOCK_HIT] * 4
    # r1 and r2 record the shared block; the fourth request sees two sharers
    # and produces the exact junction.
    assert hits["r2"] == hits["r3"] == BLOCK_HIT
    assert JUNCTION + 2 * HASH not in trace.chunk_ends["r2"]
    assert JUNCTION + 2 * HASH in trace.chunk_ends["r3"]
    # Everyone after it hits the whole shared prefix minus one unit.
    assert [hits[f"r{i}"] for i in (4, 5)] == [JUNCTION] * 2
    # Follow-up turns are unaffected.
    assert {k: v for k, v in hits.items() if k.startswith("t2_")} == {
        k: v for k, v in base.items() if k.startswith("t2_")
    }
    attn = manager.coordinator.single_type_managers[-1]
    assert attn.num_exact_junctions == 1
    assert manager.exact_junction.num_produced == 1
    assert manager.exact_junction.num_junction_hits >= 2


def test_flag_off_has_no_detector(monkeypatch):
    _env(monkeypatch, exact=False)
    manager = chunk._make_manager(num_spec=4)
    assert manager.exact_junction is None
    for mgr in manager.coordinator.single_type_managers:
        assert not mgr.exact_junction


def test_junction_survives_sequential_consumers(monkeypatch):
    """Each consumer finishes (and its CoW source becomes droppable) before the
    next arrives; the junction entries are exempt from the superseded drops.
    """
    _env(monkeypatch, exact=True)
    prompts, arrival = _siblings(7, gap=2, turn2_at=None)
    _, _, hits = _run(prompts, arrival)
    assert [hits[f"r{i}"] for i in range(4, 7)] == [JUNCTION] * 3


def test_burst_arrival(monkeypatch):
    """Start-up burst: all siblings arrive in the same step."""
    _env(monkeypatch, exact=True)
    prompts, arrival = _siblings(7, gap=0, turn2_at=None)
    _, _, hits = _run(prompts, arrival)
    assert hits["r0"] == hits["r1"] == 0
    assert hits["r2"] == hits["r3"] == BLOCK_HIT
    assert [hits[f"r{i}"] for i in range(4, 7)] == [JUNCTION] * 3


@pytest.mark.parametrize("min_tokens", [None, "0"], ids=["default", "no_floor"])
def test_cross_replay_tools_junction(monkeypatch, min_tokens):
    """A shallow shared prefix (the tools block rendered before the per-pass
    salt, shared across dataset passes): below the default 8192-token floor
    no exact junction is produced, so nothing beyond today's block-floored
    reuse happens; without the floor it would be made exact.
    """
    _env(monkeypatch, exact=True)
    if min_tokens is not None:
        monkeypatch.setenv("VLLM_PREFIX_EXACT_JUNCTION_MIN_TOKENS", min_tokens)
    shared = 4755
    prompts, arrival = _siblings(5, shared=shared, tail=27450, turn2_at=None)
    _env(monkeypatch, exact=False)
    _, base_trace, base = _run(prompts, arrival)
    _env(monkeypatch, exact=True)
    manager, trace, hits = _run(prompts, arrival)
    if min_tokens is None:
        assert hits == base and trace.chunk_ends == base_trace.chunk_ends
        assert manager.exact_junction.num_productions == 0
    else:
        assert hits["r4"] == shared // HASH * HASH - HASH


@pytest.mark.parametrize("shared", [290, 1200], ids=["sub_block", "one_block"])
def test_unexportable_junction_is_not_produced(monkeypatch, shared):
    """A junction the GDN export cannot land on (checkpoint column below the
    first block) must not split any prefill.
    """
    prompts, arrival = _siblings(4, shared=shared, tail=500, turn2_at=None)
    _env(monkeypatch, exact=False)
    _, base_trace, base = _run(prompts, arrival)
    _env(monkeypatch, exact=True)
    manager, trace, hits = _run(prompts, arrival)
    assert hits == base
    assert trace.chunk_ends == base_trace.chunk_ends
    assert manager.exact_junction.num_produced == 0


def test_repeated_lookup_never_matches_itself(monkeypatch):
    """A waiting request is looked up again every step; its own recorded
    hashes must not turn its prompt tail into a junction.
    """
    _env(monkeypatch, exact=True)
    manager = chunk._make_manager(num_spec=4)
    rng = random.Random(3)
    req = make_request("solo", chunk._tokens(rng, 5000), HASH, sha256)
    for _ in range(3):
        _, hit, junction = manager.get_computed_blocks(req)
        assert (hit, junction) == (0, 0)
        assert req.exact_junction is None


@pytest.mark.parametrize("gap", [0, 1], ids=["burst", "staggered"])
def test_no_same_step_read_of_unfilled_junction_copy(monkeypatch, gap):
    """The producer's next allocation moves the junction entry to a copy that
    only this step's worker copy fills. A sibling admitted in that same step
    must not CoW from it: the worker gathers every copy source before it
    scatters, so a chained copy (block -> copy -> cow) would read stale KV.
    Models the worker's gather-then-scatter: within one drain, no source may
    be another pair's destination.
    """
    _env(monkeypatch, exact=True)
    prompts, arrival = _siblings(8, gap=gap, turn2_at=None)
    manager = chunk._make_manager(num_spec=4)
    # The harness drains copies per request; the scheduler drains them once
    # per step, so collect each step's destinations across requests.
    take = manager.take_kv_cache_block_copies
    new_step = manager.new_step_starts
    step_dsts: set[int] = set()
    chained: list = []

    def checked_take():
        copies, retained = take()
        chained.extend(c for c in copies if c.src_block_id in step_dsts)
        step_dsts.update(c.dst_block_id for c in copies)
        return copies, retained

    def checked_new_step():
        step_dsts.clear()
        new_step()

    monkeypatch.setattr(manager, "take_kv_cache_block_copies", checked_take)
    monkeypatch.setattr(manager, "new_step_starts", checked_new_step)
    trace = chunk._run(manager, prompts, free_done=set(prompts), arrival_step=arrival)
    assert not chained
    hits = [trace.hits[f"r{i}"][0] for i in range(8)]
    assert hits[-1] == JUNCTION
    chunk._check_cached_states_and_free(manager, trace)


def _detector(monkeypatch, min_sharers=None):
    from vllm.v1.core.exact_junction import ExactJunctionDetector

    monkeypatch.setenv("VLLM_PREFIX_EXACT_JUNCTION_MIN_TOKENS", "0")
    if min_sharers is not None:
        monkeypatch.setenv("VLLM_PREFIX_EXACT_JUNCTION_MIN_SHARERS", str(min_sharers))
    return ExactJunctionDetector(
        block_size=BLOCK,
        hash_block_size=HASH,
        eagle_unit=HASH,
        export_valid=lambda start, stop, junction: True,
    )


def _req(rid, tokens):
    return make_request(rid, tokens, HASH, sha256)


def test_previous_turn_alone_is_not_a_family(monkeypatch):
    """A follow-up turn whose own cached tail was lost (e.g. evicted) matches
    only its previous turn: one sharer produces nothing by default.
    """
    rng = random.Random(5)
    history = chunk._tokens(rng, 3 * BLOCK + 500)
    longest = 3 * BLOCK - HASH  # attention covers 3 blocks (after the drop)
    for min_sharers, produced in ((None, False), (1, True)):
        det = _detector(monkeypatch, min_sharers)
        prev = _req("turn1", list(history))
        det.lookup(prev, hit=longest, longest=longest, junction=0)
        nxt = _req("turn2", history + chunk._tokens(rng, 900))
        det.lookup(nxt, hit=longest, longest=longest, junction=0)
        assert (nxt.exact_junction is not None) == produced
        if produced:
            entry = (3 * BLOCK + 500) // HASH * HASH
            assert nxt.exact_junction == (entry - HASH, entry, entry + HASH)


def test_full_block_match_is_not_a_junction(monkeypatch):
    """Siblings that share the whole uncovered block (e.g. queued before the
    first one is cached) diverge later; no junction inside this block.
    """
    det = _detector(monkeypatch)
    rng = random.Random(6)
    shared = chunk._tokens(rng, 2 * BLOCK + 300)
    longest = BLOCK - HASH
    reqs = [_req(f"s{i}", shared + chunk._tokens(rng, 50)) for i in range(3)]
    for req in reqs:
        det.lookup(req, hit=longest, longest=longest, junction=0)
        assert req.exact_junction is None


def test_short_producer_prompt_produces_nothing(monkeypatch):
    """A would-be producer whose prompt ends within two units after the
    junction stop: its prompt-tail checkpoint stop would cut the chunk before
    the junction export, so it must not produce (and must split nothing).
    """
    # r3 gets tail 9 + 21 = 30: prompt 32,161 < S + 2 * 32 + 1.
    prompts, arrival = _siblings(5, tail=9, gap=2, turn2_at=None)
    _env(monkeypatch, exact=False)
    _, base_trace, base = _run(prompts, arrival)
    _env(monkeypatch, exact=True)
    manager, trace, hits = _run(prompts, arrival)
    assert trace.chunk_ends["r3"] == base_trace.chunk_ends["r3"]
    assert hits["r3"] == base["r3"]
    attn = manager.coordinator.single_type_managers[-1]
    assert attn.num_exact_junctions == manager.exact_junction.num_productions
