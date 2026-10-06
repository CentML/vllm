# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MAMBA_PIN_OWN_CKPT: a running request pins its own latest Mamba checkpoint.

Align-mode Mamba registers the request's resume checkpoint (the prefix-match-unit
partial tail, or the block-aligned replay boundary) during prefill and releases
it into the LRU free queue right away. A long decode can then age it out while
the conversation is still active. With the flag on, the request keeps a
reference on its deepest own checkpoint until it is freed, and releases it last
(MRU end of the free queue).
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

PIN = "VLLM_MAMBA_PIN_OWN_CKPT"
DROP_SUPERSEDED = "VLLM_MAMBA_DROP_SUPERSEDED_STATE"
BLOCK = 64
HASH = 16
NUM_MAMBA_GROUPS = 3

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def _manager(num_blocks, num_spec):
    groups = [
        KVCacheGroupSpec(
            ["attn"],
            FullAttentionSpec(
                block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float16
            ),
        )
    ]
    for g in range(NUM_MAMBA_GROUPS):
        groups.append(
            KVCacheGroupSpec(
                [f"gdn{g}"],
                MambaSpec(
                    block_size=BLOCK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=num_spec,
                ),
            )
        )
    cfg = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )
    return make_kv_cache_manager(
        kv_cache_config=cfg,
        max_model_len=1 << 16,
        enable_caching=True,
        hash_block_size=HASH,
        retention_interval=0,
        use_eagle=True,
    )


def _stub(manager):
    partial = HASH < BLOCK and manager.coordinator.enable_partial_hash_hits
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK),
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        max_num_scheduled_tokens=1 << 20,
        use_eagle=True,
        use_eagle_block_drop=True,
        hash_block_size=HASH,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_partial_cache_hit=partial,
        mamba_fine_grained_prefix_cache=False,
    )


def _prefill(manager, request):
    stub = _stub(manager)
    manager.new_step_starts()
    blocks, hit, junction = manager.get_computed_blocks(request)
    request.shared_prefix_boundary = junction
    first = True
    while request.num_computed_tokens < request.num_tokens:
        local = hit if first else 0
        start = request.num_computed_tokens + local
        if start >= request.num_tokens:
            break
        n = Scheduler._mamba_block_aligned_split(
            stub, request, request.num_tokens - start, local, 0
        )
        if not first:
            manager.new_step_starts()
        out = manager.allocate_slots(
            request,
            n,
            num_new_computed_tokens=local,
            new_computed_blocks=blocks if first else None,
            num_lookahead_tokens=3,
        )
        assert out is not None
        request.num_computed_tokens = start + n
        _, retained = manager.take_kv_cache_block_copies()
        if retained:
            manager.block_pool.free_blocks(retained)
        first = False
    return hit


def _decode(manager, request, num_tokens, next_id):
    for _ in range(num_tokens):
        manager.new_step_starts()
        assert manager.allocate_slots(request, 1, num_lookahead_tokens=3) is not None
        request.append_output_token_ids([next_id])
        next_id += 1
        request.num_computed_tokens += 1
        _, retained = manager.take_kv_cache_block_copies()
        if retained:
            manager.block_pool.free_blocks(retained)
    return next_id


def _churn(manager, base, count, length):
    """Unrelated one-shot requests that fill and cycle the free queue."""
    for i in range(count):
        req = make_request(
            f"churn{base + i}",
            list(
                range(base * 100_000 + i * 10_000, base * 100_000 + i * 10_000 + length)
            ),
            HASH,
            sha256,
        )
        _prefill(manager, req)
        manager.free(req)


def _all_free(manager):
    pool = manager.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    assert all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null)


@pytest.mark.parametrize("num_spec", [0, 3])
@pytest.mark.parametrize("pin", [False, True])
def test_long_decode_keeps_resume_checkpoint(monkeypatch, pin, num_spec):
    monkeypatch.setenv(PIN, "1" if pin else "0")
    manager = _manager(num_blocks=100, num_spec=num_spec)
    turn1 = list(range(1, 1000))  # 999 tokens: partial tail at 976 = 992 - 16 (EAGLE)
    req1 = make_request("t1", turn1, HASH, sha256)
    _prefill(manager, req1)
    # While t1 decodes, unrelated traffic cycles the whole free queue.
    nid = _decode(manager, req1, 8, 50_000_000)
    _churn(manager, 1, 10, 1500)
    nid = _decode(manager, req1, 8, nid)
    manager.free(req1)
    # Turn 2 strictly extends the prompt of turn 1 (recorded history + tool output).
    req2 = make_request("t2", turn1 + list(range(7000, 7200)), HASH, sha256)
    hit = _prefill(manager, req2)
    if pin:
        assert hit == 976  # resumes at its own partial-tail checkpoint
    else:
        assert hit < 976  # the checkpoint aged out during the decode
    manager.free(req2)
    _all_free(manager)
    mgr = manager.coordinator.single_type_managers[1]
    assert (mgr.num_own_checkpoints_pinned > 0) is pin
    assert not mgr._own_pin


@pytest.mark.parametrize("drop_superseded", [False, True])
def test_pin_refcounts_and_mru_release(monkeypatch, drop_superseded):
    monkeypatch.setenv(PIN, "1")
    monkeypatch.setenv(DROP_SUPERSEDED, "1" if drop_superseded else "0")
    manager = _manager(num_blocks=200, num_spec=0)
    history = list(range(1, 700))
    nid = 60_000_000
    for turn in range(6):
        req = make_request(f"c{turn}", list(history), HASH, sha256)
        _prefill(manager, req)
        mgrs = manager.coordinator.single_type_managers[1:]
        for m in mgrs:
            pinned = m._own_pin.get(req.request_id)
            assert pinned is not None and pinned.block_hash is not None
            assert pinned.ref_cnt >= 1
            # The pinned block is the deepest own checkpoint of this prompt.
            assert pinned.block_hash_num_tokens == (
                len(history) // HASH
            ) * HASH - HASH or (pinned.block_hash_num_tokens % BLOCK == 0)
        nid = _decode(manager, req, 5, nid)
        manager.free(req)
        # After the free, the pinned checkpoint sits at the MRU end of the queue.
        tail = manager.block_pool.free_block_queue.get_all_free_blocks()[-1]
        assert tail.block_hash is not None
        history += list(range(nid, nid + 157))
        nid += 157
    _all_free(manager)


def test_free_before_cow_does_not_leak(monkeypatch):
    """A request aborted right after the step that registered its partial tail
    (pin still on the running state block, no CoW yet) frees cleanly.
    """
    monkeypatch.setenv(PIN, "1")
    manager = _manager(num_blocks=64, num_spec=0)
    req = make_request("a", list(range(1, 1000)), HASH, sha256)
    stub = _stub(manager)
    blocks, hit, _ = manager.get_computed_blocks(req)
    # Run chunks only until the partial-tail stop (976).
    while req.num_computed_tokens < 976:
        local = hit if req.num_computed_tokens == 0 else 0
        start = req.num_computed_tokens + local
        n = Scheduler._mamba_block_aligned_split(
            stub, req, req.num_tokens - start, local, 0
        )
        n = min(n, 976 - start)
        manager.new_step_starts()
        assert manager.allocate_slots(req, n, num_lookahead_tokens=3) is not None
        req.num_computed_tokens = start + n
    mgr = manager.coordinator.single_type_managers[1]
    assert req.request_id in mgr._own_pin
    pinned_pos = mgr._own_pin[req.request_id].block_hash_num_tokens
    manager.free(req)
    _, retained = manager.take_kv_cache_block_copies()
    assert not retained
    _all_free(manager)
    # The pinned GDN checkpoint survives the abort as a normal cache entry (the
    # attention partial tail at 992 was never computed, so the hybrid hit
    # floors to the block-aligned checkpoint).
    assert pinned_pos == 976
    assert any(
        b.block_hash is not None and b.block_hash_num_tokens == 976
        for b in manager.block_pool.blocks
        if not b.is_null
    )
    req2 = make_request("b", list(range(1, 1000)) + [5] * 40, HASH, sha256)
    assert manager.get_computed_blocks(req2)[1] > 0
