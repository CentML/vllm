# SPDX-License-Identifier: Apache-2.0
"""CPU tests for locality-domain KV placement.

Hybrid KV config built like qwen-v2's builder: one KVCacheTensor per group (layers = the group's layers,
offset 0, strides from compute_layout_strides), Mamba/GDN groups aliasing the attention slots.

- boundary oracle: a byte-level reference (owner block of every byte -> chunk domain) gives ppd_block and the
  straddle count independently of chunk_domains/plan_split, on small geometries with a small chunk size;
- fail-closed: wrong placement / closed gate / wrong granularity -> None, SPLIT_ACTIVE False, ppd 0;
- the scheduler boundary equals the worker ppd; pages_per_domain in kernel pages.
"""
import numpy as np
import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker import locality_kv as LK


def hybrid_config(block_size: int, num_blocks: int, n_layers: int, head_size: int = 256, n_gdn_groups: int = 3,
                  layout: KVCacheLayout = KVCacheLayout.LBHNC) -> tuple[KVCacheConfig, int]:
    attn = FullAttentionSpec(block_size=block_size, num_kv_heads=2, head_size=head_size, dtype=torch.uint8)
    page = attn.page_size_bytes
    mamba = MambaSpec(block_size=block_size, shapes=((4,), (4,)), dtypes=(torch.uint8, torch.uint8),
                      mamba_cache_mode="align", page_size_padded=page)
    groups = [KVCacheGroupSpec([f"model.layers.{4 * i + g}.linear_attn" for i in range(n_layers)], mamba)
              for g in range(n_gdn_groups)]
    groups.append(KVCacheGroupSpec([f"model.layers.{4 * i + 3}.self_attn.attn" for i in range(n_layers)], attn))
    size = n_layers * page * num_blocks
    tensors = []
    for g in groups:
        ls, bs, *_ = compute_layout_strides(g.kv_cache_spec, num_blocks, n_layers, layout)
        tensors.append(KVCacheTensor(size=size, layers=list(g.layer_names), layer_stride=ls, block_stride=bs,
                                     offset=0))
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups), page


def byte_oracle(cfg: KVCacheConfig, gran: int) -> tuple[int, int]:
    """(ppd_block, straddle) from a per-byte owner map (independent of the shim's chunk walk)."""
    t = cfg.kv_cache_tensors[-1]
    nb, nl, ls, bs = cfg.num_blocks, len(t.layers), t.layer_stride, t.block_stride
    size = t.size
    owner = np.full(size, -1, dtype=np.int64)
    ranges = []  # (slot, block, start, end)
    for lay in range(nl):
        for b in range(nb):
            s = t.offset + (lay * ls + b * bs if ls >= bs * nb else b * bs + lay * (bs // nl))
            e = s + (bs if ls >= bs * nb else bs // nl)
            owner[s:e] = b
            ranges.append((lay, b, s, e))
    half = nb // 2
    nch = -(-size // gran)
    cdom = np.array([1 if owner[c * gran] >= half else 0 for c in range(nch)])
    byte_dom = np.repeat(cdom, gran)[:size]
    full1 = {b: True for b in range(nb)}
    straddle = 0
    for _, b, s, e in ranges:
        want = 1 if b >= half else 0
        doms = byte_dom[s:e]
        if (doms != want).any():
            straddle += 1
        if (doms != 1).any():
            full1[b] = False
    ppd = next((b for b in range(half, nb) if full1[b]), nb)
    return ppd, straddle


@pytest.mark.parametrize("block_size,num_blocks,n_layers,head_size,gran", [
    (16, 101, 3, 256, 4096),  # aligned pages
    (16, 101, 3, 20, 4096),   # chunk boundaries cut pages
    (24, 64, 2, 52, 4096),    # even block count
    (40, 77, 4, 36, 8192),    # odd block count
    (8, 50, 5, 100, 2048),    # multiple layer slots
])
def test_boundary_and_straddle_match_byte_oracle(block_size, num_blocks, n_layers, head_size, gran):
    cfg, page = hybrid_config(block_size, num_blocks, n_layers, head_size=head_size)
    size = cfg.kv_cache_tensors[0].size
    plan = LK.plan_split(size, cfg, gran=gran)
    ppd, straddle = byte_oracle(cfg, gran)
    assert plan.half == num_blocks // 2
    assert plan.ppd_block == ppd
    assert plan.straddle == straddle
    assert plan.ppd_block >= plan.half
    assert set(plan.dom) <= {0, 1} and plan.runs == 2 * n_layers  # one cut per layer slot


def test_oracle_cases_exercise_straddling_and_shifted_boundary():
    # guard against a vacuous oracle: some small cases must straddle and push ppd past N/2
    cases = [(16, 101, 3, 20, 4096), (24, 64, 2, 52, 4096), (40, 77, 4, 36, 8192), (8, 50, 5, 100, 2048)]
    plans = []
    for bs, nb, nl, hd, gran in cases:
        cfg, _ = hybrid_config(bs, nb, nl, head_size=hd)
        plans.append(LK.plan_split(cfg.kv_cache_tensors[0].size, cfg, gran=gran))
    assert sum(p.straddle for p in plans) > 0
    assert any(p.ppd_block > p.half for p in plans)


@pytest.mark.parametrize("block_size,num_blocks,ppd_extra,straddle", [(1184, 14670, 2, 28), (2368, 7335, 1, 19)])
def test_point32_geometry_regression(block_size, num_blocks, ppd_extra, straddle):
    # Representative hybrid-cache geometries using the production granularity.
    cfg, _ = hybrid_config(block_size, num_blocks, 10)
    plan = LK.plan_split(cfg.kv_cache_tensors[0].size, cfg)
    assert (plan.ppd_block - plan.half, plan.straddle, plan.runs) == (ppd_extra, straddle, 20)


def test_hybrid_aliasing_dedups_geometry():
    cfg, _ = hybrid_config(16, 40, 3)
    geo = LK._geometry(cfg.kv_cache_tensors, cfg.num_blocks_of)
    assert len(geo) == 1  # 3 GDN groups + attention alias the same slots


def test_scheduler_boundary_equals_worker_plan():
    from vllm.v1.core import locality as core_loc

    cfg, _ = hybrid_config(1184, 300, 4)
    size = cfg.kv_cache_tensors[0].size
    assert core_loc.boundary(cfg) == LK.plan_split(size, cfg).ppd_block == LK.ppd_block_of(cfg)


@pytest.fixture
def fake_gpu(monkeypatch):
    """CPU allocation with controlled topology and placement responses."""
    from vllm.model_executor.layers.locality import memory

    state = {"ords": None, "gate": True, "gran": LK.GRAN}
    monkeypatch.setattr(LK, "_gate_open", lambda dev: state["gate"])
    monkeypatch.setattr(memory, "granularity", lambda dev: state["gran"])
    monkeypatch.setattr(LK, "_alloc", lambda n, dom, dev: torch.full((len(dom) * LK.GRAN,), 7, dtype=torch.uint8))
    monkeypatch.setattr(memory, "chunk_ordinals", lambda t, n, c: state["ords"](n // c))
    monkeypatch.setattr(LK, "SPLIT_ACTIVE", False)
    return state


def _cfg_small():
    cfg, page = hybrid_config(1184, 40, 2)
    return cfg, cfg.kv_cache_tensors[0].size


def test_verified_split_activates_and_ppd_in_kernel_pages(fake_gpu):
    cfg, size = _cfg_small()
    plan = LK.plan_split(size, cfg)
    fake_gpu["ords"] = lambda n: list(plan.dom)
    buf = LK.allocate_split(size, torch.device("cpu", 0), cfg)
    assert buf is not None and buf.numel() == size and buf.dtype == torch.int8 and int(buf.abs().sum()) == 0
    assert LK.SPLIT_ACTIVE
    kpb = 37  # 1184 / 32: FP8 kernel page 32 tokens
    assert LK.pages_per_domain(cfg.num_blocks * kpb) == plan.ppd_block * kpb
    assert LK.pages_per_domain(cfg.num_blocks * kpb, kpb) == plan.ppd_block * kpb
    assert LK.pages_per_domain(cfg.num_blocks * kpb + 1) == 0  # not a whole number of blocks -> off


def test_wrong_placement_fails_closed(fake_gpu):
    cfg, size = _cfg_small()
    plan = LK.plan_split(size, cfg)
    bad = list(plan.dom)
    bad[len(bad) // 3] ^= 1
    fake_gpu["ords"] = lambda n: bad
    assert LK.allocate_split(size, torch.device("cpu", 0), cfg) is None
    assert not LK.SPLIT_ACTIVE
    assert LK.pages_per_domain(cfg.num_blocks * 37) == 0


def test_closed_gate_and_bad_granularity_fail_closed(fake_gpu):
    cfg, size = _cfg_small()
    fake_gpu["ords"] = lambda n: list(LK.plan_split(size, cfg).dom)
    fake_gpu["gate"] = False
    assert LK.allocate_split(size, torch.device("cpu", 0), cfg) is None and not LK.SPLIT_ACTIVE
    fake_gpu["gate"], fake_gpu["gran"] = True, 4096
    assert LK.allocate_split(size, torch.device("cpu", 0), cfg) is None and not LK.SPLIT_ACTIVE


def test_previous_success_is_reset_by_a_failed_allocation(fake_gpu):
    cfg, size = _cfg_small()
    plan = LK.plan_split(size, cfg)
    fake_gpu["ords"] = lambda n: list(plan.dom)
    assert LK.allocate_split(size, torch.device("cpu", 0), cfg) is not None and LK.SPLIT_ACTIVE
    fake_gpu["gate"] = False
    assert LK.allocate_split(size, torch.device("cpu", 0), cfg) is None and not LK.SPLIT_ACTIVE
