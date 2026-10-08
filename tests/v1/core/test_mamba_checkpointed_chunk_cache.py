# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A prefill chunk that ends with the GDN internal checkpoint runs through every
block boundary below its checkpoint column without stopping, so it materializes
none of those states. Align-mode Mamba must not register a prefix-cache hash
on any of those columns, even when they hold a physical block:

- the previous chunk's speculative scratch blocks (never written in prefill);
- the private copy of a sub-block prefix hit (it holds the chunk-start state).

Otherwise every request sharing the prefix restores that block as its initial
recurrent state. Mirrors the deployment where this was observed
(Qwen3.6-35B-A3B): effective block 2176, prefix-match unit 32, MTP k=4,
retention 0, GDN prefill checkpoint.
"""

import random
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.distributed.kv_events import BlockRemoved
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BLOCK = 2176
HASH = 32
NUM_SPEC = 4
NUM_MAMBA_GROUPS = 3
TOKEN_BUDGET = 32768
PIN = "VLLM_MAMBA_PIN_OWN_CKPT"
DROP_SUPERSEDED = "VLLM_MAMBA_DROP_SUPERSEDED_STATE"

P_LEN = 32202
# P's replay boundary: block-floored prompt end, minus the eagle-dropped block.
REPLAY_BOUNDARY = (P_LEN // BLOCK - 1) * BLOCK  # 28288


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def _make_manager(
    num_spec: int = NUM_SPEC,
    eagle_drop: bool = True,
    retention_interval: int | None = 0,
    prefill_lookahead: int | None = None,
    fine_grained: bool = False,
    events: bool = False,
    eagle_attention_only: bool = False,
) -> KVCacheManager:
    if prefill_lookahead is None:
        prefill_lookahead = int(eagle_drop)
    groups = [
        KVCacheGroupSpec(
            [f"gdn{g}"],
            MambaSpec(
                block_size=BLOCK,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
                num_speculative_blocks=num_spec,
                num_prefill_checkpoint_blocks=1,
                prefill_checkpoint_alignment=1,
                prefill_checkpoint_reuses_initial_block=True,
                prefill_checkpoint_copies_initial_block=prefill_lookahead > 1,
            ),
        )
        for g in range(NUM_MAMBA_GROUPS)
    ]
    groups.append(
        KVCacheGroupSpec(
            ["attn"],
            FullAttentionSpec(
                block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float16
            ),
            is_eagle_group=eagle_drop and eagle_attention_only,
        )
    )
    cfg = KVCacheConfig(num_blocks=400, kv_cache_tensors=[], kv_cache_groups=groups)
    return make_kv_cache_manager(
        kv_cache_config=cfg,
        max_model_len=1 << 17,
        enable_caching=True,
        hash_block_size=HASH,
        retention_interval=retention_interval,
        use_eagle=eagle_drop,
        num_prefill_lookahead=prefill_lookahead,
        enable_mamba_fine_grained_prefix_cache=fine_grained,
        enable_kv_cache_events=events,
    )


@pytest.fixture(params=[False, True], ids=["no_pin_drop", "pin_drop"])
def manager(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    flag = "1" if request.param else "0"
    monkeypatch.setenv(PIN, flag)
    monkeypatch.setenv(DROP_SUPERSEDED, flag)
    return _make_manager()


def _split(
    manager: KVCacheManager,
    request: Request,
    num_new: int,
    hit: int,
    *,
    max_prefill_tokens: int = TOKEN_BUDGET,
) -> int:
    """The real `Scheduler._mamba_block_aligned_split` with the incident flags."""
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK),
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        max_num_scheduled_tokens=max_prefill_tokens,
        use_eagle_block_drop=_mamba_managers(manager)[0].drop_eagle_checkpoint_block,
        hash_block_size=HASH,
        mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=1,
        mamba_prefill_checkpoint_reuses_initial_block=True,
        mamba_prefill_checkpoint_copies_initial_block=(
            _mamba_managers(manager)[
                0
            ].kv_cache_spec.prefill_checkpoint_copies_initial_block
        ),
        num_prefill_lookahead=manager.coordinator.num_reprefillable_tokens + 1,
        mamba_partial_cache_hit=manager.coordinator.enable_partial_hash_hits,
        mamba_fine_grained_prefix_cache=manager.mamba_fine_grained_prefix_cache,
    )
    scheduled = Scheduler._mamba_block_aligned_split(stub, request, num_new, hit)
    return Scheduler._reserve_prefill_lookahead(
        stub, request, request.num_computed_tokens + hit, scheduled
    )


def _mamba_gids(manager: KVCacheManager) -> list[int]:
    return [
        gid
        for gid, m in enumerate(manager.coordinator.single_type_managers)
        if isinstance(m, MambaManager)
    ]


def _mamba_managers(manager: KVCacheManager) -> list[MambaManager]:
    managers = manager.coordinator.single_type_managers
    return [managers[gid] for gid in _mamba_gids(manager)]


@dataclass
class _Trace:
    requests: dict[str, Request]
    # Physical mamba block id -> (writer request, token position of its state).
    state_at: dict[int, tuple[str, int]] = field(default_factory=dict)
    chunk_ends: dict[str, list[int]] = field(default_factory=dict)
    # Admitted prefix hit: (tokens, [(block id, its state then)] per mamba group).
    hits: dict[str, tuple[int, list[tuple[int, tuple[str, int] | None]]]] = field(
        default_factory=dict
    )


def _gdn_forward(
    manager: KVCacheManager, trace: _Trace, request: Request, start: int, end: int
) -> None:
    """Apply this allocation's CoW copies, then the GDN prefill writes.

    The running column ends up at ``end``; with a valid internal checkpoint,
    column ``cdiv(end, BLOCK) - 2`` holds the state at the checkpoint position
    (gdn_attn.py checkpoint plan).
    """
    copies, retained = manager.take_kv_cache_block_copies()
    for copy in copies:
        if copy.src_block_id in trace.state_at:
            trace.state_at[copy.dst_block_id] = trace.state_at[copy.src_block_id]
    # The copies run before this step's forward; release their endpoints.
    if retained:
        manager.block_pool.free_blocks(retained)
    ckpt = get_mamba_prefill_checkpoint_position(
        end,
        HASH,
        drop_eagle_block=_mamba_managers(manager)[0].drop_eagle_checkpoint_block,
        query_start=start,
        mamba_block_size=BLOCK,
        copy_initial_block=(
            _mamba_managers(manager)[
                0
            ].kv_cache_spec.prefill_checkpoint_copies_initial_block
        ),
    )
    has_ckpt = is_mamba_prefill_checkpoint_valid(
        query_start=start,
        query_end=end,
        checkpoint_position=ckpt,
        hash_block_size=HASH,
        mamba_block_size=BLOCK,
        checkpoint_alignment=1,
        reuse_initial_block=True,
        copy_initial_block=(
            _mamba_managers(manager)[
                0
            ].kv_cache_spec.prefill_checkpoint_copies_initial_block
        ),
    )
    rid = request.request_id
    for mgr in _mamba_managers(manager):
        blocks = mgr.req_to_blocks[rid]
        trace.state_at[blocks[cdiv(end, BLOCK) - 1].block_id] = (rid, end)
        if has_ckpt:
            ckpt_block = blocks[cdiv(end, BLOCK) - 2]
            assert not ckpt_block.is_null
            trace.state_at[ckpt_block.block_id] = (rid, ckpt)
    trace.chunk_ends.setdefault(rid, []).append(end)


def _run(
    manager: KVCacheManager,
    prompts: dict[str, list[int]],
    free_done: set[str],
    arrival_step: dict[str, int] | None = None,
    finalize_draft_kv: bool = False,
) -> _Trace:
    """FCFS steps under the incident token budget.

    Requests arrive at step 0 unless ``arrival_step`` says otherwise. Prefill
    only: a request leaves once its prompt is computed (and is freed if listed
    in ``free_done``).
    """
    trace = _Trace(
        {rid: make_request(rid, ids, HASH, sha256) for rid, ids in prompts.items()}
    )
    num_spec = _mamba_managers(manager)[0].num_speculative_blocks
    arrival_step = arrival_step or {}
    pending = list(trace.requests.values())
    waiting: list[Request] = []
    running: list[Request] = []
    for step in range(16):
        waiting += [r for r in pending if arrival_step.get(r.request_id, 0) <= step]
        pending = [r for r in pending if r not in waiting]
        if not pending and not waiting and not running:
            break
        manager.new_step_starts()
        budget = TOKEN_BUDGET
        for req in running:
            start = req.num_computed_tokens
            num_new = _split(manager, req, min(req.num_tokens - start, budget), 0)
            if num_new == 0:
                continue
            assert manager.allocate_slots(req, num_new, num_lookahead_tokens=num_spec)
            _gdn_forward(manager, trace, req, start, start + num_new)
            req.num_computed_tokens = start + num_new
            budget -= num_new
        for req in list(waiting):
            blocks, hit, junction = manager.get_computed_blocks(req)
            req.shared_prefix_boundary = junction
            num_new = _split(manager, req, min(req.num_tokens - hit, budget), hit)
            if num_new == 0:
                break
            if (
                manager.allocate_slots(
                    req,
                    num_new,
                    num_new_computed_tokens=hit,
                    new_computed_blocks=blocks,
                    num_lookahead_tokens=num_spec,
                )
                is None
            ):
                # Same-step mamba hits are deferred to the next step.
                break
            hit_blocks = (
                [blocks.blocks[gid][-1] for gid in _mamba_gids(manager)]
                if hit > 0
                else []
            )
            trace.hits[req.request_id] = (
                hit,
                [(b.block_id, trace.state_at.get(b.block_id)) for b in hit_blocks],
            )
            waiting.remove(req)
            running.append(req)
            _gdn_forward(manager, trace, req, hit, hit + num_new)
            req.num_computed_tokens = hit + num_new
            budget -= num_new
        for req in [r for r in running if r.num_computed_tokens >= r.num_prompt_tokens]:
            running.remove(req)
            if finalize_draft_kv:
                # Multi-module draft KV trails target prefill by K-1 tokens.
                # An accepted decode step supplies that missing attention
                # evidence before a later sibling probes the retained tail.
                start = req.num_computed_tokens
                for _ in range(num_spec):
                    req.append_output_token_ids(0)
                assert manager.allocate_slots(
                    req, num_spec, num_lookahead_tokens=num_spec
                )
                _gdn_forward(manager, trace, req, start, start + num_spec)
                req.num_computed_tokens = start + num_spec
            if req.request_id in free_done:
                manager.free(req)
    assert not pending and not waiting and not running, "prefill did not finish"
    return trace


def _assert_holds_state(
    trace: _Trace,
    block_id: int,
    written: tuple[str, int] | None,
    tokens: int,
    reader: Request,
) -> None:
    where = f"{reader.request_id} reads block {block_id} as state@{tokens}"
    assert written is not None, f"{where}, but it was never written"
    writer, pos = written
    prefix = reader.prompt_token_ids[:tokens]
    assert (
        pos == tokens and trace.requests[writer].prompt_token_ids[:tokens] == prefix
    ), f"{where}, but it holds {writer}@{pos}"


def _check_cached_states_and_free(manager: KVCacheManager, trace: _Trace) -> None:
    # Consumers restore their initial recurrent state from their mamba hit
    # blocks, which must hold a sharer's state at exactly the hit position.
    for rid, (hit, hit_blocks) in trace.hits.items():
        for block_id, written in hit_blocks:
            _assert_holds_state(trace, block_id, written, hit, trace.requests[rid])
    # Producers: every hash-registered mamba block holds the state its hash
    # claims.
    for mgr in _mamba_managers(manager):
        for rid, blocks in mgr.req_to_blocks.items():
            for block in blocks:
                if not block.is_null and block.block_hash is not None:
                    _assert_holds_state(
                        trace,
                        block.block_id,
                        trace.state_at.get(block.block_id),
                        block.block_hash_num_tokens,
                        trace.requests[rid],
                    )
    # Ref counts: freeing every request returns every block to the pool.
    for req in trace.requests.values():
        manager.free(req)
    pool = manager.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    assert all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null)


def _tokens(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(150_000) for _ in range(n)]


@pytest.mark.parametrize(
    "consumer_after_materialization", [False, True], ids=["co_batched", "post_export"]
)
@pytest.mark.parametrize(
    "x_len",
    # Includes scratch windows straddling the replay column and ending below it.
    [4000, 5000, 7000, 9000, 13000, 14000],
)
def test_checkpointed_chunk_skips_stale_speculative_block(
    manager: KVCacheManager, x_len: int, consumer_after_materialization: bool
) -> None:
    """A co-batched request fragments P's budget; S shares 32,131 tokens."""
    rng = random.Random(0)
    p_ids = _tokens(rng, P_LEN)
    prompts = {
        "X": _tokens(rng, x_len),
        "P": p_ids,
        "S": p_ids[:32131] + _tokens(rng, 57),
    }
    # Keep original same-step admissions as stale-state safety coverage.
    # A second consumer timing probes reuse after Eagle's attention evidence
    # above the replay state has actually been exported.
    trace = _run(
        manager,
        prompts,
        free_done={"X"},
        arrival_step={"S": 5} if consumer_after_materialization else {},
    )

    # Budget fragmentation must not lose P's reusable replay state, whether
    # earlier chunks stopped before it or speculative scratch occupied its slot.
    assert REPLAY_BOUNDARY in trace.chunk_ends["P"]
    if consumer_after_materialization:
        assert trace.hits["S"][0] == REPLAY_BOUNDARY
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize(
    "consumer_after_materialization", [False, True], ids=["co_batched", "post_export"]
)
def test_checkpointed_chunk_skips_sub_block_hit_copy(
    manager: KVCacheManager, consumer_after_materialization: bool
) -> None:
    """A partial-hit CoW block must become a real boundary state before reuse.

    S2 restores P@32160, then materializes its replay state at 32640 instead
    of publishing the untouched private initial block under that boundary.
    """
    rng = random.Random(1)
    p_ids = _tokens(rng, P_LEN)
    s2_ids = p_ids + _tokens(rng, 2700)
    prompts = {"P": p_ids, "S2": s2_ids, "S3": s2_ids[:34840] + _tokens(rng, 1000)}
    trace = _run(
        manager,
        prompts,
        free_done=set(),
        arrival_step=(
            {"S2": 3, "S3": 6} if consumer_after_materialization else {"S3": 5}
        ),
    )

    p_checkpoint = (P_LEN // HASH - 1) * HASH  # 32160
    if consumer_after_materialization:
        assert trace.hits["S2"][0] == p_checkpoint
    assert 15 * BLOCK in trace.chunk_ends["S2"]
    assert trace.hits["S3"][0] == 15 * BLOCK
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("retention", [None, 0, BLOCK, 2 * BLOCK])
@pytest.mark.parametrize("pin", [False, True])
@pytest.mark.parametrize(
    "seed_live", [None, False, True], ids=["cold", "warm", "private"]
)
def test_replay_boundary_and_tail_survive_sibling_reuse(
    monkeypatch: pytest.MonkeyPatch,
    retention: int | None,
    pin: bool,
    seed_live: bool | None,
) -> None:
    """Both state@4352 and state@4992 remain reusable, including warm CoW.

    The final checkpoint column aliases the block-aligned state@4352 here.
    Keeping that shared state intentionally costs a separate tail-boundary
    step rather than overwriting it with the internal tail export.
    """
    monkeypatch.setenv(PIN, str(int(pin)))
    monkeypatch.setenv(DROP_SUPERSEDED, str(int(pin)))
    manager = _make_manager(num_spec=0, eagle_drop=False, retention_interval=retention)
    rng = random.Random(9)
    p_ids = _tokens(rng, 5000)
    prompts = {"P": p_ids}
    arrival = {"S": 6, "T": 10}
    if seed_live is not None:
        prompts = {"seed": p_ids[:4002], **prompts}
        arrival["P"] = 3
    prompts["S"] = p_ids[:4400] + _tokens(rng, 700)
    prompts["T"] = p_ids + _tokens(rng, 500)
    trace = _run(
        manager,
        prompts,
        free_done={"seed"} if seed_live is False else set(),
        arrival_step=arrival,
    )
    if seed_live is not None:
        assert trace.hits["P"][0] == 4000
        # Preserving P's initial state must not mutate seed's shared checkpoint.
        for block_id, _ in trace.hits["P"][1]:
            _assert_holds_state(
                trace,
                block_id,
                trace.state_at[block_id],
                4000,
                trace.requests["P"],
            )
    assert trace.hits["S"][0] == 2 * BLOCK
    assert trace.hits["T"][0] == 4992
    if pin:
        for mgr in _mamba_managers(manager):
            checkpoint = mgr._own_pin["P"]
            assert checkpoint.block_hash_num_tokens == 4992
            assert checkpoint.ref_cnt > 0
    _check_cached_states_and_free(manager, trace)


def test_eagle_keeps_replay_boundary_and_internal_tail_export(
    manager: KVCacheManager,
) -> None:
    """Eagle's dropped full block and hash unit remain distinct checkpoints."""
    rng = random.Random(10)
    p_ids = _tokens(rng, 8000)
    trace = _run(
        manager,
        {
            "P": p_ids,
            "S": p_ids[:6600] + _tokens(rng, 500),
            "T": p_ids + _tokens(rng, 500),
        },
        free_done=set(),
        arrival_step={"S": 4, "T": 8},
    )
    assert trace.hits["S"][0] == 2 * BLOCK
    # 8000 is hash-aligned: recompute its last token, then drop Eagle's unit.
    assert trace.hits["T"][0] == 7936
    assert trace.chunk_ends["P"] == [2 * BLOCK, 8000]
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("eagle_drop", [False, True])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_block_aligned_prompt_keeps_resend_and_extension_states(
    monkeypatch: pytest.MonkeyPatch, eagle_drop: bool, delta: int
) -> None:
    monkeypatch.setenv(PIN, "1")
    monkeypatch.setenv(DROP_SUPERSEDED, "1")
    manager = _make_manager(
        num_spec=NUM_SPEC if eagle_drop else 0,
        eagle_drop=eagle_drop,
    )
    prompt_len = (3 if eagle_drop else 2) * BLOCK + delta
    trace = _run(
        manager,
        {"P": _tokens(random.Random(11), prompt_len)},
        free_done=set(),
    )
    request = trace.requests["P"]
    replay = ((prompt_len - 1) // BLOCK - int(eagle_drop)) * BLOCK
    tail = get_mamba_prefill_checkpoint_position(prompt_len, HASH, eagle_drop)
    positions = {replay, tail}
    if eagle_drop and delta == 0:
        # The extension drops from the exact prompt end, unlike the resend.
        positions.add(prompt_len - BLOCK)
    for position in positions:
        blocks = manager.block_pool.get_cached_block(
            request.block_hashes[position // HASH - 1], _mamba_gids(manager)
        )
        assert blocks is not None, f"missing reusable checkpoint at {position}"
        for block in blocks:
            _assert_holds_state(
                trace, block.block_id, trace.state_at[block.block_id], position, request
            )
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("prompt_len", [4353, 4360, 4384, 4385, 4416])
def test_warm_eagle_preserves_private_initial_checkpoint_export(
    manager: KVCacheManager, prompt_len: int
) -> None:
    """Re-aligning past an already satisfied replay stop must not lose the tail."""
    p_ids = _tokens(random.Random(12), prompt_len)
    trace = _run(
        manager,
        {
            "A": p_ids[:4040],
            "P": p_ids,
            "T": p_ids + _tokens(random.Random(13), 200),
        },
        free_done={"A"},
        arrival_step={"P": 4, "T": 8},
    )
    assert trace.hits["P"][0] == 4000
    expected_tail = get_mamba_prefill_checkpoint_position(prompt_len, HASH, True)
    assert trace.hits["T"][0] == expected_tail
    # This valid internal export uses a private initial column; splitting at
    # 4352 would turn the following step's column into a forbidden shared alias.
    assert trace.chunk_ends["P"] == [prompt_len]
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("retention", [None, 0, BLOCK, 2 * BLOCK])
def test_internal_export_at_replay_boundary_needs_no_extra_step(
    monkeypatch: pytest.MonkeyPatch, retention: int | None
) -> None:
    monkeypatch.setenv(PIN, "1")
    monkeypatch.setenv(DROP_SUPERSEDED, "1")
    manager = _make_manager(num_spec=0, eagle_drop=False, retention_interval=retention)
    p_ids = _tokens(random.Random(14), 2 * BLOCK + 1)
    trace = _run(
        manager,
        {"P": p_ids, "T": p_ids + _tokens(random.Random(15), 200)},
        free_done=set(),
        arrival_step={"T": 4},
    )
    assert trace.hits["T"][0] == 2 * BLOCK
    assert trace.chunk_ends["P"] == [len(p_ids)]
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("eagle_drop", [False, True])
@pytest.mark.parametrize("residue", [1, 2, 3])
@pytest.mark.parametrize("pin", [False, True])
@pytest.mark.parametrize(
    "seed_live", [None, False, True], ids=["cold", "warm", "private"]
)
def test_multimodule_lookahead_keeps_progress_replay_and_tail(
    monkeypatch: pytest.MonkeyPatch,
    eagle_drop: bool,
    residue: int,
    pin: bool,
    seed_live: bool | None,
) -> None:
    monkeypatch.setenv(PIN, str(int(pin)))
    monkeypatch.setenv(DROP_SUPERSEDED, str(int(pin)))
    manager = _make_manager(num_spec=4, eagle_drop=eagle_drop, prefill_lookahead=4)
    prompt_len = 4992 + residue
    p_ids = _tokens(random.Random(16), prompt_len)
    prompts = {"P": p_ids}
    arrivals = {"T": 8}
    if seed_live is not None:
        seed_len = 4385 if eagle_drop else 4353
        prompts = {"A": p_ids[:seed_len], **prompts}
        arrivals["P"] = 4
    prompts["T"] = p_ids + _tokens(random.Random(17), 200)
    # _run uses the real split followed by reserve, exactly as schedule().
    trace = _run(
        manager,
        prompts,
        free_done={"A"} if seed_live is False else set(),
        arrival_step=arrivals,
        finalize_draft_kv=eagle_drop,
    )
    tail = get_mamba_prefill_checkpoint_position(prompt_len, HASH, eagle_drop)
    assert trace.hits["T"][0] == tail
    if seed_live is not None:
        assert trace.hits["P"][0] == 2 * BLOCK
        for block_id, _ in trace.hits["P"][1]:
            # A cached full hit was privatized, not overwritten in place.
            assert trace.state_at[block_id] == ("A", 2 * BLOCK)
            assert (
                manager.block_pool.blocks[block_id].block_hash_num_tokens == 2 * BLOCK
            )
    replay = ((prompt_len - 1) // BLOCK - int(eagle_drop)) * BLOCK
    for position in {replay, tail}:
        blocks = manager.block_pool.get_cached_block(
            trace.requests["P"].block_hashes[position // HASH - 1], _mamba_gids(manager)
        )
        assert blocks is not None
        for block in blocks:
            _assert_holds_state(
                trace,
                block.block_id,
                trace.state_at[block.block_id],
                position,
                trace.requests["P"],
            )
    if pin:
        for mgr in _mamba_managers(manager):
            assert mgr._own_pin["P"].block_hash_num_tokens == tail
    _check_cached_states_and_free(manager, trace)


def test_checkpoint_copy_capability_never_exports_a_negative_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PIN, "0")
    monkeypatch.setenv(DROP_SUPERSEDED, "0")
    manager = _make_manager(num_spec=4, eagle_drop=False, prefill_lookahead=4)
    assert not is_mamba_prefill_checkpoint_valid(
        query_start=0,
        query_end=1000,
        checkpoint_position=992,
        hash_block_size=HASH,
        mamba_block_size=BLOCK,
        checkpoint_alignment=1,
        copy_initial_block=True,
    )
    trace = _run(
        manager,
        {
            "A": _tokens(random.Random(18), 1000),
            "B": _tokens(random.Random(19), 900),
        },
        free_done=set(),
    )
    _check_cached_states_and_free(manager, trace)


def test_warm_junction_clip_materializes_boundary_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PIN, "0")
    monkeypatch.setenv(DROP_SUPERSEDED, "0")
    manager = _make_manager(fine_grained=True)
    p_ids = _tokens(random.Random(20), 4400)
    trace = _run(manager, {"A": p_ids[:4370]}, free_done=set())
    request = make_request("P", p_ids, HASH, sha256)
    trace.requests["P"] = request
    manager.new_step_starts()
    blocks, hit, _ = manager.get_computed_blocks(request)
    assert hit == 4320
    request.shared_prefix_boundary = 4384
    num_new = _split(manager, request, request.num_tokens - hit, hit)
    assert manager.allocate_slots(
        request,
        num_new,
        num_new_computed_tokens=hit,
        new_computed_blocks=blocks,
        num_lookahead_tokens=NUM_SPEC,
    )
    _gdn_forward(manager, trace, request, hit, hit + num_new)
    request.num_computed_tokens = hit + num_new
    while request.num_computed_tokens < request.num_tokens:
        manager.new_step_starts()
        start = request.num_computed_tokens
        num_new = _split(manager, request, request.num_tokens - start, 0)
        assert num_new > 0
        assert manager.allocate_slots(request, num_new, num_lookahead_tokens=NUM_SPEC)
        _gdn_forward(manager, trace, request, start, start + num_new)
        request.num_computed_tokens = start + num_new
    assert 2 * BLOCK in trace.chunk_ends["P"]
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("pin", [False, True])
def test_copy_alias_offloads_immutable_replay_and_keeps_cache_events_live(
    monkeypatch: pytest.MonkeyPatch, pin: bool
) -> None:
    monkeypatch.setenv(PIN, str(int(pin)))
    monkeypatch.setenv(DROP_SUPERSEDED, str(int(pin)))
    manager = _make_manager(
        num_spec=4, eagle_drop=False, prefill_lookahead=4, events=True
    )
    request = make_request("P", _tokens(random.Random(21), 4994), HASH, sha256)
    trace = _Trace({"P": request})
    manager.new_step_starts()
    num_new = _split(manager, request, request.num_tokens, 0)
    assert num_new == 2 * BLOCK
    assert manager.allocate_slots(request, num_new, num_lookahead_tokens=4)
    _gdn_forward(manager, trace, request, 0, num_new)
    request.num_computed_tokens = num_new
    initial = [m.req_to_blocks["P"][1] for m in _mamba_managers(manager)]
    first_offers = manager.take_boundary_state_offloads()
    assert all(
        position != 2 * BLOCK
        for entries in first_offers.values()
        for _, _, position in entries
    )
    manager.take_events()

    manager.new_step_starts()
    start = request.num_computed_tokens
    num_new = _split(manager, request, request.num_tokens - start, 0)
    assert start + num_new == request.num_tokens
    assert manager.allocate_slots(request, num_new, num_lookahead_tokens=4)
    offers = manager.take_boundary_state_offloads()["P"]
    replay_offers = [(gid, bid) for gid, bid, pos in offers if pos == 2 * BLOCK]
    assert len(replay_offers) == NUM_MAMBA_GROUPS
    retained_offloads = []
    for (gid, block_id), source in zip(replay_offers, initial):
        block = manager.block_pool.blocks[block_id]
        assert block is not source
        # Before execution, both copy endpoints are retained independently
        # of request ownership and an asynchronous connector's hold.
        assert source.ref_cnt >= 2
        assert block.ref_cnt >= 2
        block.ref_cnt += 1
        retained_offloads.append(block)
    events = manager.take_events()
    assert not any(isinstance(event, BlockRemoved) for event in events)
    _gdn_forward(manager, trace, request, start, start + num_new)
    request.num_computed_tokens = start + num_new
    for (gid, block_id), source in zip(replay_offers, initial):
        assert trace.state_at[block_id] == ("P", 2 * BLOCK)
        assert trace.state_at[source.block_id] == ("P", 4992)
    manager.free(request)
    for block in retained_offloads:
        assert block.ref_cnt == 1
        assert block.block_hash_num_tokens == 2 * BLOCK
        assert trace.state_at[block.block_id] == ("P", 2 * BLOCK)
    manager.block_pool.free_blocks(retained_offloads)
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("prompt_extra", [1, 2, 3, 31, 32])
@pytest.mark.parametrize("junction", [False, True])
@pytest.mark.parametrize("pin", [False, True])
@pytest.mark.parametrize("start_mode", ["cached_initial", "running_initial"])
def test_eagle_initial_tail_exports_next_boundary_without_runway_stall(
    monkeypatch: pytest.MonkeyPatch,
    prompt_extra: int,
    junction: bool,
    pin: bool,
    start_mode: str,
) -> None:
    monkeypatch.setenv(PIN, str(int(pin)))
    monkeypatch.setenv(DROP_SUPERSEDED, str(int(pin)))
    manager = _make_manager(prefill_lookahead=4, fine_grained=junction)
    all_ids = _tokens(random.Random(22), 4500)
    prompt_len = 2 * BLOCK + prompt_extra
    request = make_request("P", all_ids[:prompt_len], HASH, sha256)
    if start_mode == "cached_initial":
        trace = _run(
            manager, {"A": all_ids[:4370]}, free_done=set(), finalize_draft_kv=True
        )
        manager.new_step_starts()
        blocks, hit, _ = manager.get_computed_blocks(request)
        assert hit == 4320
    else:
        trace = _Trace({})
        blocks, hit = None, 0
    trace.requests["P"] = request
    request.shared_prefix_boundary = 2 * BLOCK if junction else 0
    if start_mode == "running_initial":
        # A real sub-block-budget prefill leaves private running state@4320.
        while request.num_computed_tokens < 4320:
            manager.new_step_starts()
            start = request.num_computed_tokens
            scheduled = _split(
                manager, request, min(144, 4320 - start), 0, max_prefill_tokens=144
            )
            assert scheduled > 0
            assert manager.allocate_slots(
                request, scheduled, num_lookahead_tokens=NUM_SPEC
            )
            _gdn_forward(manager, trace, request, start, start + scheduled)
            request.num_computed_tokens = start + scheduled
        hit = 0
    manager.new_step_starts()
    start = request.num_computed_tokens + hit
    assert start == 4320
    scheduled = _split(manager, request, request.num_tokens - start, hit)
    assert start + scheduled == prompt_len
    assert manager.allocate_slots(
        request,
        scheduled,
        num_new_computed_tokens=hit,
        new_computed_blocks=blocks,
        num_lookahead_tokens=NUM_SPEC,
    )
    _gdn_forward(manager, trace, request, start, start + scheduled)
    request.num_computed_tokens = start + scheduled
    for position in (4320, 2 * BLOCK):
        cached = manager.block_pool.get_cached_block(
            request.block_hashes[position // HASH - 1], _mamba_gids(manager)
        )
        assert cached is not None, f"lost tail/boundary@{position}"
        for block in cached:
            _assert_holds_state(
                trace, block.block_id, trace.state_at[block.block_id], position, request
            )
    manager.new_step_starts()
    start = request.num_computed_tokens
    for _ in range(NUM_SPEC):
        request.append_output_token_ids(0)
    assert manager.allocate_slots(request, NUM_SPEC, num_lookahead_tokens=NUM_SPEC)
    _gdn_forward(manager, trace, request, start, start + NUM_SPEC)
    request.num_computed_tokens += NUM_SPEC
    manager.new_step_starts()
    sibling = make_request("T", all_ids[:4450], HASH, sha256)
    trace.requests["T"] = sibling
    sibling_blocks, sibling_hit, _ = manager.get_computed_blocks(sibling)
    assert sibling_hit == (2 * BLOCK if prompt_extra == 32 else 4320)
    trace.hits["T"] = (
        sibling_hit,
        [
            (
                sibling_blocks.blocks[gid][-1].block_id,
                trace.state_at[sibling_blocks.blocks[gid][-1].block_id],
            )
            for gid in _mamba_gids(manager)
        ],
    )
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("pin", [False, True])
def test_deferred_full_boundary_offload_survives_budget_limited_progress(
    monkeypatch: pytest.MonkeyPatch, pin: bool
) -> None:
    monkeypatch.setenv(PIN, str(int(pin)))
    monkeypatch.setenv(DROP_SUPERSEDED, str(int(pin)))
    manager = _make_manager(num_spec=4, eagle_drop=False, prefill_lookahead=4)
    request = make_request("P", _tokens(random.Random(23), 4994), HASH, sha256)
    trace = _Trace({"P": request})
    manager.new_step_starts()
    scheduled = _split(manager, request, request.num_tokens, 0)
    assert scheduled == 2 * BLOCK
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=4)
    _gdn_forward(manager, trace, request, 0, scheduled)
    request.num_computed_tokens = scheduled
    original = [m.req_to_blocks["P"][1] for m in _mamba_managers(manager)]
    assert not manager.take_boundary_state_offloads()
    manager.new_step_starts()
    start = request.num_computed_tokens
    scheduled = _split(manager, request, 8, 0, max_prefill_tokens=16)
    assert scheduled == 8
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=4)
    _gdn_forward(manager, trace, request, start, start + scheduled)
    request.num_computed_tokens += scheduled
    offers = manager.take_boundary_state_offloads()["P"]
    assert {bid for _, bid, pos in offers if pos == 2 * BLOCK} == {
        b.block_id for b in original
    }
    for block in original:
        assert block.ref_cnt > 0
        block.ref_cnt += 1  # The asynchronous store keeps its own hold.
    manager.new_step_starts()
    start = request.num_computed_tokens
    scheduled = _split(manager, request, 16, 0, max_prefill_tokens=16)
    assert scheduled == 16
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=4)
    assert not manager.take_boundary_state_offloads()  # No duplicate offer.
    _gdn_forward(manager, trace, request, start, start + scheduled)
    request.num_computed_tokens += scheduled
    manager.free(request)
    for block in original:
        assert block.ref_cnt == 1
        assert trace.state_at[block.block_id] == ("P", 2 * BLOCK)
    manager.block_pool.free_blocks(original)
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("pin", [False, True])
def test_waiting_full_hit_retries_cold_after_failed_admission_and_eviction(
    monkeypatch: pytest.MonkeyPatch, pin: bool
) -> None:
    monkeypatch.setenv(PIN, str(int(pin)))
    monkeypatch.setenv(DROP_SUPERSEDED, str(int(pin)))
    manager = _make_manager(num_spec=4, eagle_drop=False, prefill_lookahead=4)
    ids = _tokens(random.Random(24), 4994)
    trace = _run(manager, {"A": ids}, free_done=set())
    # Diverge before A's partial tail so admission targets its full-block state.
    waiting_ids = ids[:4500] + _tokens(random.Random(25), len(ids) - 4500)
    request = make_request("P", waiting_ids, HASH, sha256)
    trace.requests["P"] = request
    manager.new_step_starts()
    blocks, hit, junction = manager.get_computed_blocks(request)
    assert hit == 2 * BLOCK
    request.shared_prefix_boundary = junction
    scheduled = _split(manager, request, request.num_tokens - hit, hit)

    # Hold the unused pool without evicting A's live source. Admission must
    # fail before the waiting request commits a private copy of its hit.
    pressure = manager.block_pool.get_new_blocks(
        manager.block_pool.get_num_free_blocks()
    )
    assert (
        manager.allocate_slots(
            request,
            scheduled,
            num_new_computed_tokens=hit,
            new_computed_blocks=blocks,
            num_lookahead_tokens=4,
        )
        is None
    )

    # Release and evict the producer while the request is still waiting. A
    # retry must use the new cold-prefix result, not its abandoned CoW source.
    manager.free(trace.requests["A"])
    evicted = manager.block_pool.get_new_blocks(
        manager.block_pool.get_num_free_blocks()
    )
    manager.block_pool.free_blocks(pressure + evicted)
    manager.new_step_starts()
    blocks, hit, junction = manager.get_computed_blocks(request)
    assert hit == 0
    request.shared_prefix_boundary = junction
    scheduled = _split(manager, request, request.num_tokens, hit)
    assert manager.allocate_slots(
        request,
        scheduled,
        num_new_computed_tokens=hit,
        new_computed_blocks=blocks,
        num_lookahead_tokens=4,
    )
    trace.hits["P"] = (0, [])
    _gdn_forward(manager, trace, request, 0, scheduled)
    request.num_computed_tokens = scheduled
    while request.num_computed_tokens < request.num_tokens:
        manager.new_step_starts()
        start = request.num_computed_tokens
        scheduled = _split(manager, request, request.num_tokens - start, 0)
        assert scheduled > 0
        assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=4)
        _gdn_forward(manager, trace, request, start, start + scheduled)
        request.num_computed_tokens += scheduled

    manager.new_step_starts()
    sibling = make_request("T", ids[:4400], HASH, sha256)
    trace.requests["T"] = sibling
    blocks, hit, _ = manager.get_computed_blocks(sibling)
    assert hit == 2 * BLOCK
    trace.hits["T"] = (
        hit,
        [
            (b.block_id, trace.state_at[b.block_id])
            for gid in _mamba_gids(manager)
            for b in blocks.blocks[gid][-1:]
        ],
    )
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("pin", [False, True])
@pytest.mark.parametrize(
    ("copy_initial", "retention"),
    [(False, 0), (False, None), (False, BLOCK), (False, 2 * BLOCK), (True, 0)],
)
@pytest.mark.parametrize("junction", [False, True])
@pytest.mark.parametrize("async_lag", [False, True])
def test_memory_estimate_bounds_retained_snapshots_and_later_decode(
    monkeypatch: pytest.MonkeyPatch,
    pin: bool,
    copy_initial: bool,
    retention: int | None,
    junction: bool,
    async_lag: bool,
) -> None:
    monkeypatch.setenv(PIN, str(int(pin)))
    monkeypatch.setenv(DROP_SUPERSEDED, str(int(pin)))
    manager = _make_manager(
        eagle_drop=True,
        prefill_lookahead=4 if copy_initial else 1,
        retention_interval=retention,
    )
    prompt_len = ((4 if junction else 3) if retention == 0 else 8) * BLOCK
    request = make_request("P", _tokens(random.Random(25), prompt_len), HASH, sha256)
    request.shared_prefix_boundary = BLOCK if junction else 0
    trace = _Trace({"P": request})
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_cache_mode="align", prefix_cache_retention_interval=retention
        ),
        model_config=SimpleNamespace(max_model_len=prompt_len + NUM_SPEC),
    )

    def check_owned_pages() -> None:
        for mamba in _mamba_managers(manager):
            blocks = [
                *mamba.req_to_blocks["P"],
                *mamba._preserved_initial_blocks.get("P", ()),
            ]
            pinned = mamba._own_pin.get("P")
            if pinned is not None:
                blocks.append(pinned)
            owned = {b.block_id for b in blocks if not b.is_null}
            estimated = mamba.kv_cache_spec.max_memory_usage_bytes(config)
            assert estimated >= len(owned) * mamba.kv_cache_spec.page_size_bytes

    previous_scheduled = 0
    while request.num_computed_tokens < request.num_prompt_tokens:
        manager.new_step_starts()
        start = request.num_computed_tokens
        request.num_in_flight_tokens = previous_scheduled if async_lag else 0
        scheduled = _split(
            manager,
            request,
            min(request.num_prompt_tokens - start, BLOCK),
            0,
            max_prefill_tokens=BLOCK,
        )
        assert scheduled > 0
        assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
        check_owned_pages()
        _gdn_forward(manager, trace, request, start, start + scheduled)
        request.num_computed_tokens += scheduled
        check_owned_pages()
        previous_scheduled = scheduled

    # Cross the exact prompt block boundary. The old internal tail can now
    # be pinned outside the running/speculative table, alongside side copies.
    manager.new_step_starts()
    start = request.num_computed_tokens
    request.num_in_flight_tokens = previous_scheduled if async_lag else 0
    for _ in range(NUM_SPEC):
        request.append_output_token_ids(0)
    assert manager.allocate_slots(request, NUM_SPEC, num_lookahead_tokens=NUM_SPEC)
    check_owned_pages()
    _gdn_forward(manager, trace, request, start, start + NUM_SPEC)
    request.num_computed_tokens += NUM_SPEC
    check_owned_pages()
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("pin", [False, True])
def test_intermediate_checkpoint_offloads_preserved_replay_before_alias_write(
    monkeypatch: pytest.MonkeyPatch, pin: bool
) -> None:
    monkeypatch.setenv(PIN, str(int(pin)))
    monkeypatch.setenv(DROP_SUPERSEDED, str(int(pin)))
    manager = _make_manager(
        eagle_drop=True, prefill_lookahead=4, eagle_attention_only=True
    )
    request = make_request("P", _tokens(random.Random(26), 3 * BLOCK), HASH, sha256)
    trace = _Trace({"P": request})
    manager.new_step_starts()
    scheduled = _split(manager, request, request.num_tokens, 0)
    assert scheduled == BLOCK
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
    _gdn_forward(manager, trace, request, 0, scheduled)
    request.num_computed_tokens = scheduled
    offers = manager.take_boundary_state_offloads()
    assert all(
        position != BLOCK for entries in offers.values() for _, _, position in entries
    )

    manager.new_step_starts()
    start = request.num_computed_tokens
    scheduled = _split(manager, request, request.num_tokens - start, 0)
    assert start + scheduled == 2 * BLOCK
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
    offers = manager.take_boundary_state_offloads()["P"]
    held = [
        manager.block_pool.blocks[bid]
        for _, bid, position in offers
        if position == BLOCK
    ]
    assert len(held) == NUM_MAMBA_GROUPS
    for block in held:
        block.ref_cnt += 1  # Independent asynchronous store ownership.
    _gdn_forward(manager, trace, request, start, start + scheduled)
    request.num_computed_tokens += scheduled

    manager.new_step_starts()
    start = request.num_computed_tokens
    scheduled = _split(manager, request, request.num_tokens - start, 0)
    assert start + scheduled == request.num_prompt_tokens
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
    _gdn_forward(manager, trace, request, start, start + scheduled)
    request.num_computed_tokens += scheduled
    manager.free(request)
    for block in held:
        assert block.ref_cnt == 1
        assert trace.state_at[block.block_id] == ("P", BLOCK)
    manager.block_pool.free_blocks(held)
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("pin", [False, True])
def test_full_offload_survives_later_partial_producer_marker(
    monkeypatch: pytest.MonkeyPatch, pin: bool
) -> None:
    monkeypatch.setenv(PIN, str(int(pin)))
    monkeypatch.setenv(DROP_SUPERSEDED, str(int(pin)))
    manager = _make_manager(
        eagle_drop=True, prefill_lookahead=4, eagle_attention_only=True
    )
    ids = _tokens(random.Random(27), 4500)
    request = make_request("P", ids[:4448], HASH, sha256)
    request.shared_prefix_boundary = 2 * BLOCK
    trace = _Trace({"P": request})
    for end in (BLOCK, 2 * BLOCK):
        manager.new_step_starts()
        start = request.num_computed_tokens
        scheduled = _split(manager, request, request.num_tokens - start, 0)
        assert start + scheduled == end
        assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
        _gdn_forward(manager, trace, request, start, end)
        request.num_computed_tokens = end
        offers = manager.take_boundary_state_offloads()
        assert all(
            position != 2 * BLOCK
            for entries in offers.values()
            for _, _, position in entries
        )

    # This query writes the regular desired tail4384, with no internal
    # checkpoint. Its producer marker must not discard the deferred full4352.
    manager.new_step_starts()
    start = request.num_computed_tokens
    scheduled = _split(manager, request, 32, 0, max_prefill_tokens=32)
    assert start + scheduled == 4384
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
    held = [
        manager.block_pool.blocks[bid]
        for _, bid, position in manager.take_boundary_state_offloads()["P"]
        if position == 2 * BLOCK
    ]
    assert len(held) == NUM_MAMBA_GROUPS
    for block in held:
        block.ref_cnt += 1
    _gdn_forward(manager, trace, request, start, start + scheduled)
    request.num_computed_tokens += scheduled

    manager.new_step_starts()
    start = request.num_computed_tokens
    scheduled = _split(manager, request, request.num_tokens - start, 0)
    assert start + scheduled == request.num_prompt_tokens
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
    _gdn_forward(manager, trace, request, start, start + scheduled)
    request.num_computed_tokens += scheduled
    # Eagle attention still prunes a whole block on this short sibling.
    # Probe the SSM cache directly so that pruning cannot hide a lost tail.
    cached = manager.block_pool.get_cached_block(
        request.block_hashes[4384 // HASH - 1], _mamba_gids(manager)
    )
    assert cached is not None
    for block in cached:
        assert trace.state_at[block.block_id] == ("P", 4384)
    manager.free(request)
    for block in held:
        assert block.ref_cnt == 1
        assert trace.state_at[block.block_id] == ("P", 2 * BLOCK)
    manager.block_pool.free_blocks(held)
    _check_cached_states_and_free(manager, trace)


def test_finalized_kv_prefix_never_hashes_unexported_running_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PIN, "0")
    monkeypatch.setenv(DROP_SUPERSEDED, "0")
    manager = _make_manager(
        eagle_drop=True, prefill_lookahead=4, eagle_attention_only=True
    )
    ids = _tokens(random.Random(28), 4500)
    request = make_request("P", ids[:4448], HASH, sha256)
    request.shared_prefix_boundary = 2 * BLOCK
    trace = _Trace({"P": request})
    for end in (BLOCK, 2 * BLOCK):
        manager.new_step_starts()
        start = request.num_computed_tokens
        scheduled = _split(manager, request, request.num_tokens - start, 0)
        assert start + scheduled == end
        assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
        _gdn_forward(manager, trace, request, start, end)
        request.num_computed_tokens = end

    manager.new_step_starts()
    start = request.num_computed_tokens
    # Exercise the allocator's arbitrary-query contract directly: the
    # scheduler may round this budget, but callers must not mislabel state.
    scheduled = 35
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
    _gdn_forward(manager, trace, request, start, start + scheduled)
    request.num_computed_tokens += scheduled
    # A consumer can only cache finalized KV through4384, but the model
    # wrote state4387, not4384: no internal export was valid for this query.
    manager.cache_blocks(request, 4384)

    manager.new_step_starts()
    start = request.num_computed_tokens
    scheduled = _split(manager, request, request.num_tokens - start, 0)
    assert start + scheduled == request.num_prompt_tokens
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
    _gdn_forward(manager, trace, request, start, start + scheduled)
    request.num_computed_tokens += scheduled
    # Check the recurrent-state cache itself: an attention miss must not
    # conceal an incorrectly registered state4387 under the4384 hash.
    assert (
        manager.block_pool.get_cached_block(
            request.block_hashes[4384 // HASH - 1], _mamba_gids(manager)
        )
        is None
    )
    cached = manager.block_pool.get_cached_block(
        request.block_hashes[2 * BLOCK // HASH - 1], _mamba_gids(manager)
    )
    assert cached is not None
    for block in cached:
        assert trace.state_at[block.block_id] == ("P", 2 * BLOCK)
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("pin", [False, True])
def test_terminal_prefill_emits_nonalias_full_snapshot_to_generic_offload(
    monkeypatch: pytest.MonkeyPatch, pin: bool
) -> None:
    monkeypatch.setenv(PIN, str(int(pin)))
    monkeypatch.setenv(DROP_SUPERSEDED, str(int(pin)))
    manager = _make_manager(
        eagle_drop=True, prefill_lookahead=4, eagle_attention_only=True
    )
    request = make_request("P", _tokens(random.Random(29), 6000), HASH, sha256)
    trace = _Trace({"P": request})
    manager.new_step_starts()
    scheduled = _split(manager, request, request.num_tokens, 0)
    assert scheduled == BLOCK
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
    _gdn_forward(manager, trace, request, 0, scheduled)
    request.num_computed_tokens = scheduled
    assert not manager.take_boundary_state_offloads()

    manager.new_step_starts()
    start = request.num_computed_tokens
    scheduled = _split(manager, request, request.num_tokens - start, 0)
    assert start + scheduled == request.num_prompt_tokens
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
    # Generic offload metadata is built in this terminal pass, before the
    # scheduler processes EOS/max_tokens=1 and frees the request.
    held = [
        manager.block_pool.blocks[bid]
        for _, bid, position in manager.take_boundary_state_offloads()["P"]
        if position == BLOCK
    ]
    assert len(held) == NUM_MAMBA_GROUPS
    for block in held:
        block.ref_cnt += 1
    _gdn_forward(manager, trace, request, start, start + scheduled)
    request.num_computed_tokens += scheduled
    # The producer-only finalizer cannot duplicate the generic handoff.
    assert not manager.finalize_partial_tail_offloads(request)
    manager.free(request)
    for block in held:
        assert block.ref_cnt == 1
        assert trace.state_at[block.block_id] == ("P", BLOCK)
    manager.block_pool.free_blocks(held)
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("pin", [False, True])
def test_waiting_full_hit_keeps_newer_partial_offload_producer(
    monkeypatch: pytest.MonkeyPatch, pin: bool
) -> None:
    monkeypatch.setenv(PIN, str(int(pin)))
    monkeypatch.setenv(DROP_SUPERSEDED, str(int(pin)))
    # Non-Eagle attention permits a full4352 hit on this short prompt.
    # Eagle attention prunes that hit to2176, missing the transition below.
    manager = _make_manager(eagle_drop=False, prefill_lookahead=4)
    ids = _tokens(random.Random(30), 3 * BLOCK)
    trace = _run(manager, {"A": ids}, free_done=set())
    manager.take_boundary_state_offloads()
    request = make_request("P", ids[:4416], HASH, sha256)
    trace.requests["P"] = request
    manager.new_step_starts()
    blocks, hit, _ = manager.get_computed_blocks(request)
    assert hit == 2 * BLOCK
    request.shared_prefix_boundary = 4400
    trace.hits["P"] = (
        hit,
        [
            (b.block_id, trace.state_at[b.block_id])
            for gid in _mamba_gids(manager)
            for b in blocks.blocks[gid][-1:]
        ],
    )
    scheduled = _split(manager, request, 32, hit, max_prefill_tokens=32)
    assert hit + scheduled == 4384
    assert manager.allocate_slots(
        request,
        scheduled,
        num_new_computed_tokens=hit,
        new_computed_blocks=blocks,
        num_lookahead_tokens=NUM_SPEC,
    )
    full_offers = manager.take_boundary_state_offloads()["P"]
    assert sum(pos == 2 * BLOCK for _, _, pos in full_offers) == NUM_MAMBA_GROUPS
    assert all(pos != 4384 for _, _, pos in full_offers)
    _gdn_forward(manager, trace, request, hit, hit + scheduled)
    request.num_computed_tokens = hit + scheduled

    manager.new_step_starts()
    start = request.num_computed_tokens
    scheduled = _split(manager, request, request.num_tokens - start, 0)
    assert start + scheduled == request.num_prompt_tokens
    assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
    tail_offers = manager.take_boundary_state_offloads()["P"]
    assert all(pos != 2 * BLOCK for _, _, pos in tail_offers)
    held = [
        manager.block_pool.blocks[bid] for _, bid, pos in tail_offers if pos == 4384
    ]
    assert len(held) == NUM_MAMBA_GROUPS
    for block in held:
        block.ref_cnt += 1
    _gdn_forward(manager, trace, request, start, start + scheduled)
    request.num_computed_tokens += scheduled
    manager.free(request)
    for block in held:
        assert block.ref_cnt == 1
        assert trace.state_at[block.block_id] == ("P", 4384)
    manager.block_pool.free_blocks(held)
    _check_cached_states_and_free(manager, trace)


@pytest.mark.parametrize("pin", [False, True])
@pytest.mark.parametrize("async_lag", [False, True])
def test_resumed_output_replay_memory_includes_original_tail_snapshot(
    monkeypatch: pytest.MonkeyPatch, pin: bool, async_lag: bool
) -> None:
    monkeypatch.setenv(PIN, str(int(pin)))
    monkeypatch.setenv(DROP_SUPERSEDED, str(int(pin)))
    manager = _make_manager(eagle_drop=True, prefill_lookahead=4)
    trace = _run(manager, {"P": _tokens(random.Random(31), 4 * BLOCK)}, free_done=set())
    request = trace.requests["P"]
    assert request.sampling_params is not None
    request.sampling_params.max_tokens = 2 * BLOCK
    request.max_tokens = 2 * BLOCK
    for _ in range(200 // NUM_SPEC):
        manager.new_step_starts()
        start = request.num_computed_tokens
        for _ in range(NUM_SPEC):
            request.append_output_token_ids(0)
        assert manager.allocate_slots(request, NUM_SPEC, num_lookahead_tokens=NUM_SPEC)
        _gdn_forward(manager, trace, request, start, start + NUM_SPEC)
        request.num_computed_tokens += NUM_SPEC

    # Preemption releases all request-owned copies and evictable cache states.
    # Replaying the generated outputs must preserve the original prompt tail
    # as well as the two replay boundaries and the newly discovered junction.
    manager.free(request)
    evicted = manager.block_pool.get_new_blocks(
        manager.block_pool.get_num_free_blocks()
    )
    manager.block_pool.free_blocks(evicted)
    request.num_computed_tokens = 0
    request.status = RequestStatus.PREEMPTED
    request.shared_prefix_boundary = BLOCK
    config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        model_config=SimpleNamespace(max_model_len=manager.max_model_len),
    )

    def check_owned_pages() -> None:
        for mamba in _mamba_managers(manager):
            blocks = [
                *mamba.req_to_blocks["P"],
                *mamba._preserved_initial_blocks.get("P", ()),
            ]
            pinned = mamba._own_pin.get("P")
            if pinned is not None:
                blocks.append(pinned)
            owned = {b.block_id for b in blocks if not b.is_null}
            assert mamba.kv_cache_spec.max_memory_usage_bytes(config) >= (
                len(owned) * mamba.kv_cache_spec.page_size_bytes
            )

    previous_scheduled = 0
    while request.num_computed_tokens < request.num_tokens:
        manager.new_step_starts()
        start = request.num_computed_tokens
        request.num_in_flight_tokens = previous_scheduled if async_lag else 0
        scheduled = _split(
            manager,
            request,
            min(request.num_tokens - start, BLOCK),
            0,
            max_prefill_tokens=BLOCK,
        )
        assert scheduled > 0
        assert manager.allocate_slots(request, scheduled, num_lookahead_tokens=NUM_SPEC)
        check_owned_pages()
        _gdn_forward(manager, trace, request, start, start + scheduled)
        request.num_computed_tokens += scheduled
        check_owned_pages()
        previous_scheduled = scheduled

    request.num_in_flight_tokens = 0
    while request.num_computed_tokens <= 5 * BLOCK:
        manager.new_step_starts()
        start = request.num_computed_tokens
        for _ in range(NUM_SPEC):
            request.append_output_token_ids(0)
        assert manager.allocate_slots(request, NUM_SPEC, num_lookahead_tokens=NUM_SPEC)
        check_owned_pages()
        _gdn_forward(manager, trace, request, start, start + NUM_SPEC)
        request.num_computed_tokens += NUM_SPEC
        check_owned_pages()
    _check_cached_states_and_free(manager, trace)
