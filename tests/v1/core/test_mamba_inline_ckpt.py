# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[F122] In-step Mamba prefill checkpoints (VLLM_MAMBA_TAIL_CKPT=1).

A prompt chunk runs through its block-boundary stop b and its partial-tail stop T
instead of splitting there. The KV cache manager keys a reserved block D at T
(pre-filled by a page copy with the state at the chunk start) and tells the
worker to write the state at b into the block-table column that held the chunk's
initial state, and the state at T into D.

These tests run the real scheduler split and KV cache manager with a small
"state position" emulator of the worker: every block carries the number of
prompt tokens its recurrent state represents. The emulator applies the CoW /
pre-copy page copies, the align-mode pre-copy across block columns, the step's
forward (running block -> chunk end) and the in-step checkpoints, and checks
that every state is read where it is expected and that every prefix-cache hit
lands on a block holding exactly the hit position.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched import mamba_inline_ckpt
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

pytestmark = pytest.mark.skip_global_cleanup

B, HB, POOL, NG = 2208, 32, 6000, 3


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def _set_flags(monkeypatch, on: bool, block: bool = True, tail: bool = True,
               per_step: int = 4) -> None:
    monkeypatch.setattr(mamba_inline_ckpt, "ENABLED", on)
    monkeypatch.setattr(mamba_inline_ckpt, "MERGE_BLOCK", on and block)
    monkeypatch.setattr(mamba_inline_ckpt, "MERGE_TAIL", on and tail)
    monkeypatch.setattr(mamba_inline_ckpt, "MAX_PER_STEP", per_step)
    monkeypatch.setattr(mamba_inline_ckpt, "MAX_RUNNING", 0)


def _manager():
    groups = [
        KVCacheGroupSpec(
            ["attn"],
            FullAttentionSpec(block_size=B, num_kv_heads=1, head_size=1, dtype=torch.float16),
        )
    ]
    for g in range(NG):
        groups.append(
            KVCacheGroupSpec(
                [f"gdn{g}"],
                MambaSpec(
                    block_size=B,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                    mamba_cache_mode="align",
                    num_speculative_blocks=0,
                ),
            )
        )
    cfg = KVCacheConfig(num_blocks=POOL, kv_cache_tensors=[], kv_cache_groups=groups)
    return make_kv_cache_manager(
        kv_cache_config=cfg,
        max_model_len=1 << 20,
        enable_caching=True,
        hash_block_size=HB,
        retention_interval=0,
        use_eagle=True,
    )


def _stub(mgr, max_tokens=1 << 20):
    on = mgr.mamba_inline_ckpt
    st = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=B),
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        max_num_scheduled_tokens=max_tokens,
        use_eagle=True,
        use_eagle_block_drop=True,
        hash_block_size=HB,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=None,
        mamba_partial_cache_hit=True,
        mamba_fine_grained_prefix_cache=False,
        mamba_inline_ckpt=on,
        _inline_ckpt_used=0,
        inline_ckpt_stats={
            "merged": 0,
            "merged_block": 0,
            "merged_tail": 0,
            "chunks_saved": 0,
            "refused_budget": 0,
        },
        kv_cache_manager=mgr,
        running=[],
    )
    st._inline_ckpt_split = Scheduler._inline_ckpt_split.__get__(st)
    return st


class Emu:
    """Worker emulator: block id -> prompt position of the state it holds."""

    def __init__(self, mgr):
        self.mgr = mgr
        self.content: dict[int, int] = {}
        self.state_col: dict[str, int] = {}
        self.mamba = [
            (i, m)
            for i, m in enumerate(mgr.coordinator.single_type_managers)
            if isinstance(m, MambaManager)
        ]
        self.ckpts_seen: list = []

    def step(self, req, start, end):
        mgr = self.mgr
        copies, retained = mgr.take_kv_cache_block_copies()
        mgr.mamba_inline_tail_pending.clear()  # as Scheduler.schedule
        inline = mgr.take_mamba_inline_ckpts() if mgr.mamba_inline_ckpt else {}
        # 1. page copies (CoW + checkpoint pre-copies) before the forward
        for c in copies:
            self.content[c.dst_block_id] = self.content.get(c.src_block_id, -1)
        rid = req.request_id
        prev = self.state_col.get(rid, (start - 1) // B if start > 0 else -1)
        curr = cdiv(end, B) - 1
        plan = inline.get(rid)
        for gidx, m in self.mamba:
            blocks = m.req_to_blocks[rid]
            run = blocks[curr]
            assert not run.is_null
            # 2. align pre-copy across block columns
            if prev >= 0 and prev != curr:
                assert not blocks[prev].is_null
                self.content[run.block_id] = self.content.get(blocks[prev].block_id, -1)
            # 3. the chunk's initial state is the state at `start`
            if start > 0:
                assert self.content.get(run.block_id) == start, (rid, gidx, start, end)
            # 4. in-step checkpoints (replays read the state at `start`)
            if plan is not None:
                p_start, p_end, cks = plan
                assert (p_start, p_end) == (start, end)
                for pos, kind, zero_init, by_gid in cks:
                    blk = by_gid[gidx]
                    if kind == mamba_inline_ckpt.KIND_RUN:
                        assert blk == run.block_id
                        continue
                    if kind == mamba_inline_ckpt.KIND_BLOCK:
                        assert pos == (start // B + 1) * B
                        assert blk == blocks[start // B].block_id
                    if zero_init:
                        assert start == 0
                    else:
                        assert self.content.get(blk) == start, (rid, kind, pos)
                    self.content[blk] = pos
            # 5. forward: the running block holds the state at `end`
            self.content[run.block_id] = end
        if plan is not None:
            self.ckpts_seen.append((rid, plan))
        self.state_col[rid] = curr
        if retained:
            mgr.block_pool.free_blocks(retained)

    def check_hit(self, blocks, hit):
        if hit <= 0:
            return
        for gidx, _ in self.mamba:
            blk = blocks.blocks[gidx][-1]
            assert not blk.is_null
            assert self.content.get(blk.block_id) == hit, (gidx, hit, blk.block_id)


def _prefill(mgr, stub, emu, req):
    mgr.new_step_starts()
    blocks, hit, junction = mgr.get_computed_blocks(req)
    req.shared_prefix_boundary = junction
    emu.check_hit(blocks, hit)
    chunks, first = [], True
    while req.num_computed_tokens < req.num_tokens:
        local = hit if first else 0
        start = req.num_computed_tokens + local
        if start >= req.num_tokens:
            break
        stub._inline_ckpt_used = 0
        n = Scheduler._mamba_block_aligned_split(
            stub, req, min(req.num_tokens - start, stub.max_num_scheduled_tokens), local, 0
        )
        assert n > 0
        if not first:
            mgr.new_step_starts()
        out = mgr.allocate_slots(
            req,
            n,
            num_new_computed_tokens=local,
            new_computed_blocks=blocks if first else None,
            num_lookahead_tokens=3,
        )
        assert out is not None
        req.num_computed_tokens = start + n
        emu.step(req, start, start + n)
        chunks.append((start, start + n))
        first = False
    return hit, chunks


# turn prompt lengths: warm turns ending in the same block, crossing one block
# boundary, crossing two (an interior last-cacheable stop), a turn whose tail
# boundary sits just below a block boundary, and cold-ish long first turns
TURNS = (1500, 2300, 2349, 3100, 4416 + 17, 5000, 7340, 9000, 9001, 13300)


def _run(monkeypatch, on, pin, turns=TURNS, max_tokens=1 << 20, block=True, tail=True):
    monkeypatch.setenv("VLLM_MAMBA_PIN_OWN_CKPT", pin)
    monkeypatch.setenv("VLLM_MAMBA_DROP_SUPERSEDED_STATE", pin)
    _set_flags(monkeypatch, on, block, tail)
    mgr = _manager()
    assert mgr.mamba_inline_ckpt == on
    stub = _stub(mgr, max_tokens)
    emu = Emu(mgr)
    base = list(range(1000, 1000 + max(turns) + 10))
    hits, all_chunks = [], []
    for t, plen in enumerate(turns):
        req = make_request(f"c#{t}", base[:plen], HB, sha256)
        hit, chunks = _prefill(mgr, stub, emu, req)
        hits.append(hit)
        all_chunks.append(chunks)
        mgr.free(req)
    # every block is back (only the null block is not free)
    assert mgr.block_pool.get_num_free_blocks() == POOL - 1
    assert not mgr.mamba_inline_tail_pending
    for _, m in emu.mamba:
        assert not m._inline_plan and not m._inline_step and not m._inline_alloc
    # a fresh conversation over the same prompts hits where each turn ended
    return hits, all_chunks, emu, stub


@pytest.mark.parametrize("pin", ["0", "1"])
def test_inline_ckpt_matches_split_flow(monkeypatch, pin):
    hits0, chunks0, _, _ = _run(monkeypatch, False, pin)
    hits1, chunks1, emu1, stub1 = _run(monkeypatch, True, pin)
    # same prefix-cache hits on every turn
    assert hits0 == hits1, (hits0, hits1)
    n0 = sum(len(c) for c in chunks0)
    n1 = sum(len(c) for c in chunks1)
    assert n1 < n0
    st = stub1.inline_ckpt_stats
    assert st["merged_block"] > 0 and st["merged_tail"] > 0
    assert n0 - n1 == st["chunks_saved"]
    # split flow: every last chunk is the <= 63-token tail
    for c0, c1, plen in zip(chunks0, chunks1, TURNS):
        assert c0[-1][1] - c0[-1][0] <= 63
        assert c1[-1][1] == plen
    # merged flow: no chunk starting mid-block stops at its next block boundary
    # unless it is the last cacheable position of a longer prompt
    for chunks in chunks1:
        for s, e in chunks[:-1]:
            if s % B:
                assert e != (s // B + 1) * B or e % B == 0


@pytest.mark.parametrize("block,tail", [(True, False), (False, True)])
def test_inline_ckpt_partial_modes(monkeypatch, block, tail):
    hits0, _, _, _ = _run(monkeypatch, False, "1")
    hits1, _, _, st = _run(monkeypatch, True, "1", block=block, tail=tail)
    assert hits0 == hits1
    s = st.inline_ckpt_stats
    assert (s["merged_block"] > 0) == block and (s["merged_tail"] > 0) == tail


def test_inline_ckpt_token_budget(monkeypatch):
    # a per-step token budget below the prompt: chunks end block-aligned, the
    # tail is merged only into the chunk that reaches the prompt end
    hits0, _, _, _ = _run(monkeypatch, False, "1", max_tokens=4096)
    hits1, chunks1, _, _ = _run(monkeypatch, True, "1", max_tokens=4096)
    assert hits0 == hits1
    for chunks in chunks1:
        for s, e in chunks:
            assert e - s <= 4096


def test_inline_ckpt_per_step_budget(monkeypatch):
    _set_flags(monkeypatch, True, per_step=1)
    mgr = _manager()
    stub = _stub(mgr)
    stub._inline_ckpt_used = 1  # this step's budget is already used
    req = make_request("x#0", list(range(7000, 7000 + 800)), HB, sha256)
    req.shared_prefix_boundary = 0
    n = Scheduler._mamba_block_aligned_split(stub, req, 800, 0, 0)
    assert n == 800 // HB * HB - HB  # split at the tail boundary
    assert stub.inline_ckpt_stats["refused_budget"] == 1
    assert "x#0" not in mgr.mamba_inline_tail_pending


def test_inline_ckpt_off_is_stock(monkeypatch):
    """Flag off: the manager and splitter take the stock paths."""
    _set_flags(monkeypatch, False)
    mgr = _manager()
    assert not mgr.mamba_inline_ckpt
    for _, m in Emu(mgr).mamba:
        assert not m.inline_ckpt


@pytest.mark.parametrize("pin", ["0", "1"])
@pytest.mark.parametrize("mutate", [False, True])
def test_inline_ckpt_block_state_is_hit(monkeypatch, pin, mutate):
    """A branch that shares the prompt only up to the merged block boundary
    resumes from the in-step block checkpoint X: X must hold the state at b.
    mutate=True drops the worker's X write and must be caught."""
    monkeypatch.setenv("VLLM_MAMBA_PIN_OWN_CKPT", pin)
    monkeypatch.setenv("VLLM_MAMBA_DROP_SUPERSEDED_STATE", pin)
    _set_flags(monkeypatch, True)
    mgr = _manager()
    stub = _stub(mgr)
    emu = Emu(mgr)
    if mutate:
        orig = emu.step

        def step(req, start, end):
            # emulate a worker that forgets the block checkpoint
            plan = None
            saved = mgr.take_mamba_inline_ckpts

            def take():
                out = saved()
                return {
                    r: (s, e, tuple(c for c in cks if c[1] != mamba_inline_ckpt.KIND_BLOCK))
                    for r, (s, e, cks) in out.items()
                }

            mgr.take_mamba_inline_ckpts = take
            try:
                orig(req, start, end)
            finally:
                mgr.take_mamba_inline_ckpts = saved
            return plan

        emu.step = step
    base = list(range(1000, 1000 + 6000))
    # turn 2 runs [1440, 4916) through b = 2208, which is also its last
    # cacheable boundary (one block below the last full one): X is cached
    for t, plen in enumerate((1500, 4916)):
        req = make_request(f"c#{t}", base[:plen], HB, sha256)
        _prefill(mgr, stub, emu, req)
        mgr.free(req)
    assert stub.inline_ckpt_stats["merged_block"] == 1
    assert all(m.num_inline_block_ckpts == 1 for _, m in emu.mamba)
    # branch: same tokens up to 4500 (past the full attention block at 4416,
    # which EAGLE matching needs one unit past the hit, before T = 4864), then
    # different ones: the deepest cached GDN state on its path is X at b
    branch = base[:4500] + list(range(90000, 90000 + 300))
    req = make_request("b#0", branch, HB, sha256)
    mgr.new_step_starts()
    blocks, hit, _ = mgr.get_computed_blocks(req)
    assert hit == B
    if mutate:
        with pytest.raises(AssertionError):
            emu.check_hit(blocks, hit)
    else:
        emu.check_hit(blocks, hit)


def test_inline_ckpt_missing_tail_write_is_caught(monkeypatch):
    """Mutation check of the emulator: a worker that skips the tail checkpoint
    leaves D at the chunk start and the next turn's hit check fails."""
    orig = Emu.step

    def step(self, req, start, end):
        saved = self.mgr.take_mamba_inline_ckpts

        def take():
            return {
                r: (s, e, tuple(c for c in cks if c[1] != mamba_inline_ckpt.KIND_TAIL))
                for r, (s, e, cks) in saved().items()
            }

        self.mgr.take_mamba_inline_ckpts = take
        try:
            orig(self, req, start, end)
        finally:
            self.mgr.take_mamba_inline_ckpts = saved

    monkeypatch.setattr(Emu, "step", step)
    with pytest.raises(AssertionError):
        _run(monkeypatch, True, "1")
