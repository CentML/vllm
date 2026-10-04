# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel Factory paged-FP8 prefill attention (vllm/v1/attention/ops/kf_prefill_attn).

1. The C++ planner is bit-identical to the solution's Python setup() (full and reduced
   searches), and every kernel form setup() picks is in the warmup set.
2. SM107: the kernel through the runtime (plan -> launch) is finite, run-to-run
   bitwise, leaves its merge counters at zero, and its error vs an fp32 reference is
   no worse than the FlashInfer trtllm-gen path's on the same FP8 inputs.
3. SM107, per-row gate: a q=1 / KV=16 row planned with an optimistic length (an
   async-spec decode row's upper bound) is wrong while the exact length is fine, and
   plan() keeps any step with an inexact-length row, or (MIN_KV > 0) a row below
   MIN_KV next to a long row, on the production path.
"""

import random
import types

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
    stub = types.SimpleNamespace(mark_layout_dynamic=lambda **k: None)
    monkeypatch.setattr(K, "from_dlpack", lambda *a, **k: stub)
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


def _inputs(reqs, width, seed=0):
    """FP8 Q and paged KV (every page slot random, like stale KV past a row's end)."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    npg = [(L + 127) // 128 for _, L in reqs]
    pages = sum(npg) + 64
    kv = (torch.randn(pages, 2, 128, 512, device=dev, generator=gen) * 0.7).to(
        torch.float8_e4m3fn
    )
    perm = torch.randperm(pages, device=dev, generator=gen).to(torch.int32)
    bt = torch.zeros(len(reqs), width, dtype=torch.int32, device=dev)
    o = 0
    for b, n in enumerate(npg):
        bt[b, :n] = perm[o : o + n]
        o += n
    T = sum(r[0] for r in reqs)
    q = (torch.randn(T, 16, 256, device=dev, generator=gen) * 0.7).to(
        torch.float8_e4m3fn
    )
    return q, kv, bt


def _runtime(planned, precise=False):
    """A runtime with MIN_KV 0 whose compiled forms cover each planned launch."""
    dev = torch.device("cuda")
    rt = KfPrefillAttn(dev, torch.zeros(394 << 20, dtype=torch.uint8, device=dev))
    rt.planner.build()
    rt.min_kv = 0
    rt.force_precise = precise  # VLLM_KF_PREFILL_ATTN_PRECISE
    hp, hb, hm, hmb = rt.host[0]
    forms = set()
    for reqs in planned:
        ql, kvs = [r[0] for r in reqs], [r[1] for r in reqs]
        res = rt.planner.plan_into(ql, kvs, rt.nsm, hp, hb, hm, hmb, *rt.grids)
        forms.add(res[7] if precise else res[6])  # precise_variant / variant
    rt.warmup(precise=False, variants=sorted(forms))
    return rt


def _launch(rt, plan, q, kv):
    out = torch.full(
        (q.shape[0], 16, 256), float("nan"), dtype=torch.bfloat16, device=q.device
    )
    assert rt.launch(plan, q, kv, out, 0.0625, 1.0)
    torch.accelerator.synchronize()
    return out


def _row_rel(x, ref, reqs):
    """Per-row relL2 of x vs ref."""
    out, o = [], 0
    for ql, _ in reqs:
        r = ref[o : o + ql]
        out.append(float((x[o : o + ql].float() - r).norm() / r.norm()))
        o += ql
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
# 2052 = vLLM's block-table width at max_model_len 262144 with the bf16-state
# 1152-token hybrid block: plan() pads it into the kernel's 2057-stride buffer.
@pytest.mark.parametrize("width", [MAXP, 2052])
def test_kernel_through_runtime(reqs, precise, width):
    q, kv, bt = _inputs(reqs, width)
    T = q.shape[0]
    rt = _runtime([reqs], precise)
    ql, kvs, exact = [r[0] for r in reqs], [r[1] for r in reqs], [True] * len(reqs)
    plan = rt.plan(ql, kvs, bt, exact)
    assert plan is not None
    assert rt.compiled.keys() == {plan.precise_variant if precise else plan.variant}
    assert plan.block_tables.stride(0) == MAXP
    outs = []
    for _ in range(2):
        out = torch.full(
            (T, 16, 256), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        assert rt.launch(plan, q, kv, out, 0.0625, 1.0)
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
    # Every row below the KV threshold: the step stays on FlashInfer.
    rt.min_kv = max(kvs) + 1
    assert rt.plan(ql, kvs, bt, exact) is None
    assert rt.stats["below_min_kv"] == 1


@pytest.mark.skipif(
    not current_platform.is_device_capability(107), reason="SM107 kernel"
)
def test_q1_kv16_row_optimistic_length_is_wrong_and_gated():
    """The in-situ q=1 / KV=16 failure (VR S4, MIN_KV=0: relL2 0.05-0.43 vs FlashInfer
    0.006-0.012): the kernel attends past the row's end when plan() gets the row's
    optimistic async-spec length; with the exact length the row is fine. The
    builder flags such rows inexact and plan() keeps the step on FlashInfer.
    """
    reqs = [[1, 16]]
    q, kv, bt = _inputs(reqs, 2052)
    ref = _fp32_reference(q, kv, bt, reqs, 0.0625, 1.0)
    fi = _flashinfer_prefill(q, kv, bt, reqs, 0.0625, 1.0)
    (rel_fi,) = _row_rel(fi, ref, reqs)
    rt = _runtime([[[1, 16]], [[1, 18]]])
    # Exact length: KF is as good as FlashInfer.
    plan = rt.plan([1], [16], bt, [True])
    assert plan is not None
    (rel_kf,) = _row_rel(_launch(rt, plan, q, kv), ref, reqs)
    assert rel_kf <= 1.05 * rel_fi, (rel_kf, rel_fi)
    # Optimistic length (2 rejected drafts): the raw kernel output is wrong.
    plan = rt.plan([1], [18], bt, [True])
    assert plan is not None
    (rel_kf,) = _row_rel(_launch(rt, plan, q, kv), ref, reqs)
    assert rel_kf > 3 * rel_fi and rel_kf > 0.02, (rel_kf, rel_fi)
    # The builder marks the row inexact (not prefilling): production path.
    assert rt.plan([1], [18], bt, [False]) is None
    assert rt.plan([1], [18], bt, None) is None
    assert rt.stats["fallback_inexact_kv"] == 2


@pytest.mark.skipif(
    not current_platform.is_device_capability(107), reason="SM107 kernel"
)
@pytest.mark.parametrize("min_kv", [4096, 0])
def test_mixed_long_and_q1_kv16_rows(min_kv):
    """One long row co-scheduled with a q=1 / KV=16 row. MIN_KV 4096 gates per row:
    the short row keeps the whole step on FlashInfer (not just the step's longest KV).
    MIN_KV 0: both rows run on KF and each is as good as FlashInfer's; an inexact
    (async-spec decode) short row still keeps the step on FlashInfer.
    """
    reqs = [[64, 30000], [1, 16]]
    ql, kvs = [r[0] for r in reqs], [r[1] for r in reqs]
    q, kv, bt = _inputs(reqs, 2052)
    rt = _runtime([reqs, reqs[:1]])
    rt.min_kv = min_kv
    assert rt.plan(ql, [30000, 18], bt, [True, False]) is None
    assert rt.stats["fallback_inexact_kv"] == 1
    plan = rt.plan(ql, kvs, bt, [True, True])
    if min_kv:
        assert plan is None
        assert rt.stats["fallback_short_row"] == 1
        assert rt.plan(ql[:1], kvs[:1], bt[:1], [True]) is not None
        return
    assert plan is not None
    ref = _fp32_reference(q, kv, bt, reqs, 0.0625, 1.0)
    rel_kf = _row_rel(_launch(rt, plan, q, kv), ref, reqs)
    rel_fi = _row_rel(_flashinfer_prefill(q, kv, bt, reqs, 0.0625, 1.0), ref, reqs)
    assert all(k <= 1.05 * f for k, f in zip(rel_kf, rel_fi)), (rel_kf, rel_fi)
