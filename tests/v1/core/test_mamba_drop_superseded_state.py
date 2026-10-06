# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MAMBA_DROP_SUPERSEDED_STATE: prev+curr Mamba checkpoint retention.

A multi-turn conversation resumes each turn from the previous turn's Mamba
checkpoint ("prev") and registers a deeper one ("curr"). With the flag on,
"prev" leaves the prefix cache once "curr" exists; with it off, "prev" stays
as an evictable LRU entry. Junction checkpoints (shared prefixes) are kept.
"""

import pytest
import torch

from tests.v1.core.test_prefix_caching import (
    _make_hybrid_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

ENV = "VLLM_MAMBA_DROP_SUPERSEDED_STATE"
BLOCK = 32
NUM_SPEC = 3
MAMBA_GROUP = 1

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def _mtp_manager():
    kv_cache_config = KVCacheConfig(
        num_blocks=200,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float16
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_mtp"],
                MambaSpec(
                    block_size=BLOCK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                ),
            ),
        ],
    )
    return make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=BLOCK,
        retention_interval=0,
        use_eagle=True,
    )


def _run_turn(manager, req, chunk_ends):
    """Prefill ``req`` in the given chunk ends (align-mode scheduler style)."""
    manager.new_step_starts()
    blocks, num_computed, _ = manager.get_computed_blocks(req)
    first = True
    for end in chunk_ends:
        if not first:
            manager.new_step_starts()
        start = req.num_computed_tokens if not first else num_computed
        out = manager.allocate_slots(
            req,
            end - start,
            num_computed if first else 0,
            blocks if first else None,
            num_lookahead_tokens=NUM_SPEC,
        )
        assert out is not None
        req.num_computed_tokens = end
        first = False
    return num_computed


def _mamba_cached(manager, req, block_idx):
    return (
        manager.block_pool.get_cached_block(
            req.block_hashes[block_idx], kv_cache_group_ids=[MAMBA_GROUP]
        )
        is not None
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_multi_turn_drops_superseded_checkpoint(monkeypatch, enabled):
    monkeypatch.setenv(ENV, "1" if enabled else "0")
    manager = _mtp_manager()

    # Turn 1: 144 tokens. MTP retains the state at 96 (block 2): the
    # extension boundary 128 minus the EAGLE drop.
    turn1 = list(range(144))
    req1 = make_request("t1", turn1, BLOCK, sha256)
    assert _run_turn(manager, req1, (96, 144)) == 0
    manager.free(req1)
    assert _mamba_cached(manager, req1, 2)

    # Turn 2 strictly extends turn 1 (history + new tool output).
    turn2 = turn1 + list(range(1000, 1100))  # 244 tokens
    req2 = make_request("t2", turn2, BLOCK, sha256)
    # Resumes from turn 1's checkpoint, registers its own at 192 (block 5),
    # then finishes the prompt.
    assert _run_turn(manager, req2, (192, 244)) == 96
    assert _mamba_cached(manager, req2, 5)
    # "prev" (turn 1's checkpoint) is superseded by "curr".
    assert _mamba_cached(manager, req1, 2) is (not enabled)
    manager.free(req2)
    assert _mamba_cached(manager, req2, 5)

    # Turn 3 extends turn 2: identical hit either way.
    req3 = make_request("t3", turn2 + list(range(2000, 2040)), BLOCK, sha256)
    _, num_computed, _ = manager.get_computed_blocks(req3)
    assert num_computed == 192

    # A branch that shares only turn 1 (e.g. a retry of turn 2 with a different
    # tool result) can no longer resume at 96 when the flag is on. This is the
    # trade-off of prev+curr retention.
    branch = make_request("b", turn1 + list(range(3000, 3100)), BLOCK, sha256)
    _, num_computed, _ = manager.get_computed_blocks(branch)
    assert num_computed == (0 if enabled else 96)

    # Pool accounting stays consistent.
    mgr = manager.coordinator.single_type_managers[MAMBA_GROUP]
    if enabled:
        assert mgr.num_superseded_states_dropped == 1
        assert not mgr._drop_on_release
        assert not mgr._consumed_state


def test_no_deeper_checkpoint_keeps_prev(monkeypatch):
    """If a turn adds too few tokens to register a deeper checkpoint, the one it
    resumed from is still the latest and must be kept.
    """
    monkeypatch.setenv(ENV, "1")
    manager = _mtp_manager()
    turn1 = list(range(144))
    req1 = make_request("t1", turn1, BLOCK, sha256)
    _run_turn(manager, req1, (96, 144))
    manager.free(req1)

    # 150 tokens: extension boundary 128 - 32 = 96, same as the hit.
    req2 = make_request("t2", turn1 + [5000] * 6, BLOCK, sha256)
    assert _run_turn(manager, req2, (150,)) == 96
    manager.free(req2)
    assert _mamba_cached(manager, req1, 2)
    mgr = manager.coordinator.single_type_managers[MAMBA_GROUP]
    assert mgr.num_superseded_states_dropped == 0


def test_shared_prefix_junction_is_kept(monkeypatch):
    """Marconi junction checkpoints are cross-request shared prefixes; a
    request that resumes from one and goes deeper must not drop it.
    """
    monkeypatch.setenv(ENV, "1")
    block_size = 16
    manager = make_kv_cache_manager(
        _make_hybrid_kv_cache_config(block_size, 200, ["full", "mamba_align"]),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        retention_interval=0,
    )
    shared = [7] * (2 * block_size)

    def distinct(v):
        # 4.5 blocks: the prompt tail checkpoint lands at shared + 4 blocks.
        return [v] * (4 * block_size + block_size // 2)

    tail = len(shared) + 4 * block_size  # block-aligned retained checkpoint
    total = len(shared) + len(distinct(0))

    def prefill(req, nc, cb, stops):
        start = nc
        for i, end in enumerate(stops):
            manager.new_step_starts()
            if i == 0:
                assert manager.allocate_slots(req, end - start, nc, cb) is not None
            else:
                assert manager.allocate_slots(req, end - start) is not None
            req.num_computed_tokens = end
            start = end

    req0 = make_request("0", shared + distinct(50), block_size, sha256)
    cb, nc, _ = manager.get_computed_blocks(req0)
    prefill(req0, nc, cb, (tail, total))
    manager.free(req0)

    # req1 detects the junction and caches its state (Marconi chunk).
    req1 = make_request("1", shared + distinct(60), block_size, sha256)
    cb, nc, boundary = manager.get_computed_blocks(req1)
    assert boundary == 2 * block_size
    req1.shared_prefix_boundary = boundary
    prefill(req1, nc, cb, (boundary, tail, total))
    manager.free(req1)

    mgr = manager.coordinator.single_type_managers[1]
    # req2 resumes from the junction and registers a deeper checkpoint.
    req2 = make_request("2", shared + distinct(70), block_size, sha256)
    cb, nc, _ = manager.get_computed_blocks(req2)
    assert nc == 2 * block_size
    prefill(req2, nc, cb, (tail, total))
    manager.free(req2)
    # It did register a deeper checkpoint (so the junction was a candidate)...
    req2_ext = make_request("2x", shared + distinct(70) + [1] * 40, block_size, sha256)
    assert manager.get_computed_blocks(req2_ext)[1] == tail
    # ...but the junction was protected.
    assert mgr.num_superseded_states_dropped == 0

    # req3 still reuses the junction.
    req3 = make_request("3", shared + distinct(80), block_size, sha256)
    _, nc3, _ = manager.get_computed_blocks(req3)
    assert nc3 == 2 * block_size
