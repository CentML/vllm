# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CoW block copy at an oversized hybrid attention-page geometry.

Qwen3.6-35B-A3B: fp8 KV (1 B), head_dim 256, block 2368, Hkv 2 (one view) and
Hkv 1 (head-split, two views); GDN conv+SSM bf16 states (16 k-heads, 32
v-heads, 128x128, conv kernel 4, num_spec 0 and 3) padded to the attention
page. Views come from the real allocate_kv_cache in the V2-runner order (GDN
views before the attention views that alias their bytes).

Drives both copy paths directly:
  * one-launch: _cow_copy_rows -> _cow_copy_plan -> copy_kv_cache_blocks_inplace
    (VLLM_KV_COW_ONE_LAUNCH=1; the plan must be accepted, no silent fallback);
  * per-storage: _copy_kv_cache_blocks_inplace_per_storage (fused 0 and 1).
Measures the fraction of every destination attention page that equals its
source page, prints one COW_COVERAGE line per case, and requires 100% with no
write outside the destination block. The first-seen data_ptr dedup copies only
the GDN payload prefix (partial coverage); widest_alias_views copies it all.
"""
import os

import numpy as np
import pytest
import torch

from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker import utils as WU

DEV = "cuda" if torch.cuda.is_available() else (
    "cpu" if os.environ.get("COW_TEST_CPU") == "1" else None
)
pytestmark = pytest.mark.skipif(DEV is None, reason="needs CUDA")

NB, L, BS = 4, 2, 2368
SRC, DST = 0, 2


def _build(hkv_split, num_spec):
    hkv = 1 if hkv_split else 2
    attn = FullAttentionSpec(
        block_size=BS, num_kv_heads=hkv, head_size=256, dtype=torch.uint8
    )
    page = attn.page_size_bytes
    conv = (3 + num_spec, 16 * 128 * 2 + 32 * 128)
    ssm = (32, 128, 128)
    mamba = MambaSpec(
        block_size=BS,
        shapes=(conv, ssm),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_cache_mode="align",
        page_size_padded=page,
    )
    payload = sum(torch.Size(s).numel() * 2 for s in (conv, ssm))
    assert payload < page, (payload, page)
    groups = [
        KVCacheGroupSpec(
            [f"model.layers.{4 * i + g}.linear_attn" for i in range(L)], mamba
        )
        for g in range(3)
    ]
    names = ["attn_h0", "attn_h1"] if hkv_split else ["attn"]
    groups += [
        KVCacheGroupSpec(
            [f"model.layers.{4 * i + 3}.self_attn.{n}" for i in range(L)], attn
        )
        for n in names
    ]
    tensors = []
    for g in groups:
        ls, bstride, *_ = compute_layout_strides(
            g.kv_cache_spec, NB, L, KVCacheLayout.LBHNC
        )
        tensors.append(
            KVCacheTensor(
                size=L * page * NB,
                layers=list(g.layer_names),
                layer_stride=ls,
                block_stride=bstride,
                offset=0,
            )
        )
    cfg = KVCacheConfig(num_blocks=NB, kv_cache_tensors=tensors, kv_cache_groups=groups)
    caches = WU.allocate_kv_cache(cfg, torch.device(DEV), KVCacheLayout.LBHNC)
    attn_names = [n for n in caches if "self_attn" in n]
    shared = {caches[n].data_ptr() for n in attn_names} & {
        c.data_ptr() for n, c in caches.items() if "linear_attn" in n
    }
    # The bug needs a GDN view first on an address an attention view also has.
    assert shared, "no GDN/attention aliasing at this geometry"
    first = list(caches)
    assert any(
        "linear_attn" in n
        and first.index(n)
        < min(
            first.index(a) for a in attn_names if caches[a].data_ptr() == caches[n].data_ptr()
        )
        for n in first
        if caches[n].data_ptr() in shared
    ), "V2-runner order (GDN before attention) not reproduced"
    return caches, attn_names, page, payload


def _fill(caches, attn_names):
    for n in attn_names:
        v = caches[n].view(NB, -1)
        idx = torch.arange(v.shape[1], device=v.device) % 251
        for b in range(NB):
            v[b] = (idx ^ (b * 7 + 1)).to(torch.uint8)
    return {n: caches[n].view(NB, -1).clone() for n in attn_names}


def _coverage(caches, attn_names, before, page):
    if DEV == "cuda":
        torch.cuda.synchronize()
    worst, outside = 1.0, 0
    for n in attn_names:
        v = caches[n].view(NB, -1)
        ok = int((v[DST] == before[n][SRC]).sum())
        worst = min(worst, ok / page)
        for b in range(NB):
            if b != DST:
                outside += int((v[b] != before[n][b]).sum())
    return worst, outside


def _report(path, hkv_split, num_spec, extra, cov, outside, page, payload):
    tag = "H(Hkv1x2)" if hkv_split else "S(Hkv2)"
    fixed = "present" if hasattr(WU, "widest_alias_views") else "absent"
    print(
        f"\nCOW_COVERAGE path={path}{extra} bs={BS} {tag} spec={num_spec} "
        f"page={page} gdn_payload={payload} coverage={100 * cov:.1f}% "
        f"({int(round(cov * BS))}/{BS} tok) outside_writes={outside} "
        f"widest_alias_views={fixed}"
    )


def _reset(monkeypatch):
    monkeypatch.setattr(WU, "KV_COW_LAYOUT_CACHE", False, raising=False)
    WU._cow_copy_plans.clear()
    WU._cow_layout_plans.clear()


@pytest.mark.skipif(DEV != "cuda", reason="one-launch path is CUDA-only")
@pytest.mark.parametrize("num_spec", [0, 3])
@pytest.mark.parametrize("hkv_split", [False, True], ids=["hkv2", "hkv1"])
def test_one_launch_cow_copy_rows_real_geometry(hkv_split, num_spec, monkeypatch):
    monkeypatch.setenv("VLLM_KV_COW_ONE_LAUNCH", "1")
    _reset(monkeypatch)
    caches, attn_names, page, payload = _build(hkv_split, num_spec)
    kv = list(caches.values())
    layout = WU._cow_copy_rows(kv, NB)
    assert layout is not None, "_cow_copy_rows rejected the layout"
    # Row-level view of _cow_copy_rows: bytes per block it will copy at each
    # attention view's address.
    rows = layout[2]
    row_len = {addr: length for _, addr, length, _ in rows}
    row_cov = min(row_len.get(caches[n].data_ptr(), 0) for n in attn_names) / page
    plan = WU._cow_copy_plan(kv, NB)
    assert plan is not None, "one-launch plan refused (would fall back)"
    before = _fill(caches, attn_names)
    WU.copy_kv_cache_blocks_inplace(kv, NB, [KVCacheBlockCopy(SRC, DST)])
    cov, outside = _coverage(caches, attn_names, before, page)
    _report("one_launch", hkv_split, num_spec, f" rows={100 * row_cov:.1f}%",
            cov, outside, page, payload)
    assert outside == 0
    assert row_cov == 1.0 and cov == 1.0, f"partial copy {100 * cov:.1f}%"


@pytest.mark.parametrize("fused", ["0", "1"], ids=["plain", "fused"])
@pytest.mark.parametrize("num_spec", [0, 3])
@pytest.mark.parametrize("hkv_split", [False, True], ids=["hkv2", "hkv1"])
def test_per_storage_real_geometry(hkv_split, num_spec, fused, monkeypatch):
    monkeypatch.setenv("VLLM_KV_COW_ONE_LAUNCH", "0")
    monkeypatch.setenv("VLLM_FUSED_KV_BLOCK_COPY", fused)
    _reset(monkeypatch)
    caches, attn_names, page, payload = _build(hkv_split, num_spec)
    kv = list(caches.values())
    before = _fill(caches, attn_names)
    WU._copy_kv_cache_blocks_inplace_per_storage(
        kv, NB, np.array([[SRC, DST]], dtype=np.int64)
    )
    cov, outside = _coverage(caches, attn_names, before, page)
    _report("per_storage", hkv_split, num_spec, f" fused={fused}",
            cov, outside, page, payload)
    assert outside == 0
    assert cov == 1.0, f"partial copy {100 * cov:.1f}%"
