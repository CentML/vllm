# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel Factory paged-FP8 prefill attention (vllm/v1/attention/ops/kf_prefill_attn).

1. The C++ planner is bit-identical to the solution's Python setup() (full and reduced
   searches), and every kernel form setup() picks is in the warmup set.
2. SM107: the kernel through the runtime (plan -> launch) is finite, run-to-run
   bitwise, leaves its merge counters at zero, and its error vs an fp32 reference is
   no worse than the FlashInfer trtllm-gen path's on the same FP8 inputs.
"""

import random

import pytest
import torch

pytest.importorskip("cutlass")
from vllm.platforms import current_platform  # noqa: E402
from vllm.third_party.kf_prefill_attn import kernel as K  # noqa: E402
from vllm.v1.attention.ops.kf_prefill_attn import planner  # noqa: E402
from vllm.v1.attention.ops.kf_prefill_attn.runtime import (  # noqa: E402
    MAXP,
    KfPrefillAttn,
    reachable_variants,
    search_grids,
)

NSM = 212
EDGE = [
    [[1, 10]],
    [[1, 1]],
    [[3, 200003]],
    [[1, 150001], [7, 99999], [9, 4097], [129, 129], [255, 70000], [8, 8]],
    [[128, 128]],
    [[4096, 4096]],
    [[129, 2306], [1, 4352], [127, 132]],
    [[8, 8]] * 256,
    [[44, 78252], [224, 85088], [224, 53088], [512, 33184], [30464, 152320]],
    [[2176, 262144]],
]


def _random_launches(n: int, seed: int) -> list[list[list[int]]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        b = rng.choice([1, 1, 2, 3, 4, 6, 8, 16, 64])
        reqs = []
        for _ in range(b):
            q = rng.choice([1, 7, 16, 37, 128, 600, 2048, 4000])
            reqs.append([q, q + rng.choice([0, 1, 127, 5000, 60000, 140000])])
        out.append(reqs)
    return out


@pytest.fixture
def python_setup(monkeypatch):
    """The solution's own setup() on CPU (no compile, no device sync)."""
    monkeypatch.setattr(K, "_NSM", [NSM])
    monkeypatch.setattr(K, "_COMPILED", [object()] * len(K._COMPILED))
    monkeypatch.setattr(K, "from_dlpack", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)

    def run(reqs, preset):
        kgrid, base, dense, filler = search_grids(K, preset)
        monkeypatch.setattr(K, "_KGRID", kgrid)
        monkeypatch.setattr(K, "_BASE_WAVE_GRID", base)
        monkeypatch.setattr(K, "_DENSE_WAVE_GRID", dense)
        if not filler:
            monkeypatch.setattr(K, "_filler_splits", lambda *a, **k: None)
        ql = torch.tensor([r[0] for r in reqs], dtype=torch.int32)
        sl = torch.tensor([r[1] for r in reqs], dtype=torch.int32)
        return K.setup(
            {
                "device": "cpu",
                "setup_inputs": {"query_lens_cpu": ql, "seq_lens_cpu": sl},
            }
        )

    return run


@pytest.mark.parametrize("preset", ["r3", "full"])
def test_cpp_planner_matches_solution_setup(python_setup, preset):
    grids = search_grids(K, preset)
    launches = EDGE + _random_launches(40 if preset == "full" else 120, seed=7)
    allowed = set(reachable_variants(K.NOPS, precise=True))
    hp = torch.empty((32768, 12), dtype=torch.int32)
    hb = torch.empty(NSM + 2, dtype=torch.int32)
    hm = torch.empty((32768, 8), dtype=torch.int32)
    hmb = torch.empty(NSM + 2, dtype=torch.int32)
    for reqs in launches:
        st = python_setup(reqs, preset)
        nwork, nb, nm, nmb, nslot, ngrp, variant, precise_variant, _, _ = (
            planner.plan_into(
                [r[0] for r in reqs], [r[1] for r in reqs], NSM, hp, hb, hm, hmb, *grids
            )
        )
        assert nwork == st["nwork"], reqs
        assert torch.equal(hp[:nwork], st["plan"]), reqs
        assert torch.equal(hb[:nb], st["bins"]), reqs
        assert torch.equal(hm[:nm], st["mrg"]), reqs
        assert torch.equal(hmb[:nmb], st["mbin"]), reqs
        assert (nslot, max(4 * ngrp, 4) + 4) == (st["nslot"], st["cnt"].numel())
        assert (variant, precise_variant) == (st["variant"], st["precise_variant"])
        assert variant in allowed and precise_variant in allowed


def _fp32_reference(q, kv, bt, reqs, b1, b2):
    from vllm.v1.attention.backends.flashinfer import _fp32_paged_prefill_reference

    cu = [0]
    for ql, _ in reqs:
        cu.append(cu[-1] + ql)
    return _fp32_paged_prefill_reference(q, kv, bt, cu, [L for _, L in reqs], b1, b2)


def _flashinfer_prefill(q, kv, bt, reqs, b1, b2):
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache

    dev = q.device
    npg = [(L + 127) // 128 for _, L in reqs]
    cu_q = torch.tensor(
        [0] + list(torch.tensor([r[0] for r in reqs]).cumsum(0)),
        dtype=torch.int32,
        device=dev,
    )
    cu_kv = torch.tensor(
        [0] + list(torch.tensor(npg).cumsum(0)), dtype=torch.int32, device=dev
    )
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=dev)
    ws = torch.zeros(128 << 20, dtype=torch.uint8, device=dev)
    trtllm_batch_context_with_kv_cache(
        query=q,
        kv_cache=(kv[..., :256], kv[..., 256:]),
        workspace_buffer=ws,
        block_tables=bt,
        seq_lens=torch.tensor([L for _, L in reqs], dtype=torch.int32, device=dev),
        max_q_len=max(r[0] for r in reqs),
        max_kv_len=max(r[1] for r in reqs),
        bmm1_scale=b1,
        bmm2_scale=b2,
        batch_size=len(reqs),
        cum_seq_lens_q=cu_q,
        cum_seq_lens_kv=cu_kv,
        window_left=-1,
        out=out,
        kv_layout="HND",
    )
    return out


@pytest.mark.skipif(
    not current_platform.is_device_capability(107), reason="SM107 kernel"
)
@pytest.mark.parametrize(
    "reqs",
    [
        [[2048, 60000]],
        [[1, 100000]],
        [[37, 70001], [600, 9000], [128, 140128], [4000, 4000]],
        [[16, 50000 + 777 * i] for i in range(16)],
    ],
)
@pytest.mark.parametrize("precise", [False, True])
def test_kernel_through_runtime(reqs, precise):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    npg = [(L + 127) // 128 for _, L in reqs]
    pages = sum(npg) + 64
    kv = (torch.randn(pages, 2, 128, 512, device=dev, generator=gen) * 0.7).to(
        torch.float8_e4m3fn
    )
    perm = torch.randperm(pages, device=dev, generator=gen).to(torch.int32)
    bt = torch.zeros(len(reqs), MAXP, dtype=torch.int32, device=dev)
    o = 0
    for b, n in enumerate(npg):
        bt[b, :n] = perm[o : o + n]
        o += n
    T = sum(r[0] for r in reqs)
    q = (torch.randn(T, 16, 256, device=dev, generator=gen) * 0.7).to(
        torch.float8_e4m3fn
    )
    ws = torch.zeros(394 << 20, dtype=torch.uint8, device=dev)
    rt = KfPrefillAttn(dev, ws)
    rt.planner.build()
    rt.min_kv = 0
    rt.force_precise = precise  # VLLM_KF_PREFILL_ATTN_PRECISE
    hp, hb, hm, hmb = rt.host[0]
    variant, precise_variant = rt.planner.plan_into(
        [r[0] for r in reqs], [r[1] for r in reqs], rt.nsm, hp, hb, hm, hmb, *rt.grids
    )[6:8]
    form = precise_variant if precise else variant
    rt.warmup(precise=False, variants=[form])
    plan = rt.plan([r[0] for r in reqs], [r[1] for r in reqs])
    assert plan is not None
    assert (plan.variant, plan.precise_variant) == (variant, precise_variant)
    outs = []
    for _ in range(2):
        out = torch.full((T, 16, 256), float("nan"), dtype=torch.bfloat16, device=dev)
        assert rt.launch(plan, q, kv, bt, out, 0.0625, 1.0)
        outs.append(out)
    torch.accelerator.synchronize()
    assert bool((rt.cnt == 0).all())
    assert torch.equal(outs[0], outs[1])
    kf_out = outs[0].float()
    assert bool(torch.isfinite(kf_out).all())
    ref = _fp32_reference(q, kv, bt, reqs, 0.0625, 1.0)
    fi = _flashinfer_prefill(q, kv, bt, reqs, 0.0625, 1.0).float()
    rel = lambda x: float((x - ref).norm() / ref.norm())  # noqa: E731
    assert rel(kf_out) <= 1.05 * rel(fi), (rel(kf_out), rel(fi))
    # Below the KV threshold the step stays on FlashInfer.
    rt.min_kv = max(r[1] for r in reqs) + 1
    assert rt.plan([r[0] for r in reqs], [r[1] for r in reqs]) is None
