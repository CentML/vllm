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
from vllm.v1.request import Request

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


@pytest.fixture(params=[False, True], ids=["no_pin_drop", "pin_drop"])
def manager(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    flag = "1" if request.param else "0"
    monkeypatch.setenv(PIN, flag)
    monkeypatch.setenv(DROP_SUPERSEDED, flag)
    groups = [
        KVCacheGroupSpec(
            [f"gdn{g}"],
            MambaSpec(
                block_size=BLOCK,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
                num_speculative_blocks=NUM_SPEC,
                num_prefill_checkpoint_blocks=1,
                prefill_checkpoint_alignment=1,
                prefill_checkpoint_reuses_initial_block=True,
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
        )
    )
    cfg = KVCacheConfig(num_blocks=400, kv_cache_tensors=[], kv_cache_groups=groups)
    return make_kv_cache_manager(
        kv_cache_config=cfg,
        max_model_len=1 << 17,
        enable_caching=True,
        hash_block_size=HASH,
        retention_interval=0,
        use_eagle=True,
        num_prefill_lookahead=1,
    )


def _split(manager: KVCacheManager, request: Request, num_new: int, hit: int) -> int:
    """The real `Scheduler._mamba_block_aligned_split` with the incident flags."""
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK),
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        max_num_scheduled_tokens=TOKEN_BUDGET,
        use_eagle_block_drop=True,
        hash_block_size=HASH,
        mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=1,
        mamba_prefill_checkpoint_reuses_initial_block=True,
        mamba_partial_cache_hit=manager.coordinator.enable_partial_hash_hits,
        mamba_fine_grained_prefix_cache=False,
    )
    return Scheduler._mamba_block_aligned_split(stub, request, num_new, hit)


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
    ckpt = get_mamba_prefill_checkpoint_position(end, HASH, drop_eagle_block=True)
    has_ckpt = is_mamba_prefill_checkpoint_valid(
        query_start=start,
        query_end=end,
        checkpoint_position=ckpt,
        hash_block_size=HASH,
        mamba_block_size=BLOCK,
        checkpoint_alignment=1,
        reuse_initial_block=True,
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
) -> _Trace:
    """FCFS steps under the incident token budget.

    Requests arrive at step 0 unless ``arrival_step`` says otherwise. Prefill
    only: a request leaves once its prompt is computed (and is freed if listed
    in ``free_done``).
    """
    trace = _Trace(
        {rid: make_request(rid, ids, HASH, sha256) for rid, ids in prompts.items()}
    )
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
            assert manager.allocate_slots(req, num_new, num_lookahead_tokens=NUM_SPEC)
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
                    num_lookahead_tokens=NUM_SPEC,
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
    ("x_len", "p_first_chunk"),
    [
        # Healthy: P's first chunk reaches the replay boundary itself.
        (4000, REPLAY_BOUNDARY),
        # Co-batched prefill in (4480, 13184]: P's first chunk stops short and
        # its speculative window covers the replay-boundary column.
        (5000, REPLAY_BOUNDARY - BLOCK),
        (7000, REPLAY_BOUNDARY - 2 * BLOCK),
        (9000, REPLAY_BOUNDARY - 3 * BLOCK),
        (13000, REPLAY_BOUNDARY - 4 * BLOCK),
        # The speculative window ends below the replay-boundary column.
        (14000, REPLAY_BOUNDARY - 5 * BLOCK),
    ],
)
def test_checkpointed_chunk_skips_stale_speculative_block(
    manager: KVCacheManager, x_len: int, p_first_chunk: int
) -> None:
    """X (unrelated) is scheduled ahead of P in the same step, so P's first
    chunk stops short and the second runs to the prompt end with the internal
    checkpoint. S shares P's first 32,131 tokens.
    """
    rng = random.Random(0)
    p_ids = _tokens(rng, P_LEN)
    prompts = {
        "X": _tokens(rng, x_len),
        "P": p_ids,
        "S": p_ids[:32131] + _tokens(rng, 57),
    }
    trace = _run(manager, prompts, free_done={"X"})

    assert trace.chunk_ends["P"] == [p_first_chunk, P_LEN]
    if p_first_chunk == REPLAY_BOUNDARY:
        # A materialized replay boundary is still reused.
        assert trace.hits["S"][0] == REPLAY_BOUNDARY
    _check_cached_states_and_free(manager, trace)


def test_checkpointed_chunk_skips_sub_block_hit_copy(manager: KVCacheManager) -> None:
    """S2 resumes from P's checkpoint at 32,160 (inside block 14) and runs to
    its end with the internal checkpoint, through boundary 32,640. Its private
    copy of the hit block still holds state@32,160 and sits on S2's replay
    boundary column. S3 arrives after S2 ran and shares it past S2's attention
    block at 34,816 but not up to its checkpoint at 34,848, so its deepest
    mamba candidate is that replay-boundary column.
    """
    rng = random.Random(1)
    p_ids = _tokens(rng, P_LEN)
    s2_ids = p_ids + _tokens(rng, 2700)
    prompts = {"P": p_ids, "S2": s2_ids, "S3": s2_ids[:34840] + _tokens(rng, 1000)}
    trace = _run(manager, prompts, free_done=set(), arrival_step={"S3": 2})

    p_checkpoint = (P_LEN // HASH - 1) * HASH  # 32160
    assert trace.hits["S2"][0] == p_checkpoint
    assert trace.chunk_ends["S2"] == [len(s2_ids)]
    _check_cached_states_and_free(manager, trace)
