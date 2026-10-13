# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_ATTN_MIXED_FORK / VLLM_ATTN_MIXED_FORK_KF_SMS (KF prefill on a side stream,
planned for fewer SMs than its compiled grid).

1. pad_bins extends a start-offset table with empty bins.
2. C++ planner at a reduced SM count: the work bins cover every plan row exactly once
   within the reduced grid, merge bins cover every merge row, and the padded tables
   give every CTA past the planned grid an empty work and merge range.
3. KfPrefillAttn.plan(nsm=N) hands the kernel tables padded to its compiled extent
   (N = 0 keeps the planner's own tables), on a stub kernel module (no CuTe needed).
4. FlashInferImpl._kf_prefill(fork=True) launches on the high-priority side stream
   and _kf_fork_join orders the current stream after it; without the flag (or env)
   it launches on the current stream.
5. SM107: a reduced-SM plan is a valid plan for the full compiled grid (finite, merge
   counters back at zero, error vs an fp32 reference on par with the full-grid plan),
   and the forked launch is bitwise equal to the serial one.
6. The per-step SM budget model (VLLM_ATTN_MIXED_FORK_KF_SMS=-1) keeps every SM when
   the decode cannot stream faster on fewer SMs, picks a multiple of 8 when it can,
   and passes fixed budgets through.
"""

import sys
import types

import pytest
import torch

from vllm.v1.attention.ops.kf_prefill_attn import planner
from vllm.v1.attention.ops.kf_prefill_attn.runtime import MAXP, pad_bins

NSM = 212
MIXED = [
    # C1024 mixed steps: a few new-turn suffixes over ~100k cached tokens each.
    [[780, 106_000], [512, 98_000], [1200, 121_000], [640, 88_000], [700, 110_000]],
    [[2600, 130_000], [96, 77_000]],
    [[8800, 140_000], [300, 60_000], [33, 45_000], [1500, 101_000]],
    [[51, 46_880]],
    [[14_204, 63_584], [3000, 95_000]],
]
REDUCED = [192, 160, 128, 106, 64, 7]


def test_pad_bins():
    t = torch.tensor([0, 3, 5, 9, -1, -1, -1], dtype=torch.int32)
    assert pad_bins(t, 4, 7) == 7
    assert t.tolist() == [0, 3, 5, 9, 9, 9, 9]
    # Already long enough: untouched.
    assert pad_bins(t, 7, 5) == 7
    assert t.tolist() == [0, 3, 5, 9, 9, 9, 9]


@pytest.mark.parametrize(
    "sms,gbps,expect",
    [
        ("0", "100", "zero"),
        ("160", "100", "160"),
        ("-1", "64", "zero"),  # today's average per-SM decode rate: no gain
        ("-1", "130", "split"),
        ("-1", "1000", "split"),
    ],
)
def test_kf_sms_model(monkeypatch, sms, gbps, expect):
    from vllm.v1.attention.backends.flashinfer import _attn_mixed_fork_kf_sms

    monkeypatch.setenv("VLLM_ATTN_MIXED_FORK_KF_SMS", sms)
    monkeypatch.setenv("VLLM_ATTN_MIXED_FORK_DEC_GBPS_PER_SM", gbps)
    reqs = MIXED[0]
    ql, kv = [r[0] for r in reqs], [r[1] for r in reqs]
    n = _attn_mixed_fork_kf_sms(ql, kv, 130 * 100_000, NSM)
    if expect == "zero":
        assert n == 0
    elif expect == "split":
        assert 64 <= n < NSM and n % 8 == 0
    else:
        assert n == int(expect)
    if expect == "split":
        # A heavier decode leaves the prefill fewer SMs (or the same).
        assert _attn_mixed_fork_kf_sms(ql, kv, 250 * 100_000, NSM) <= n
        # No decode: nothing to overlap.
        assert _attn_mixed_fork_kf_sms(ql, kv, 0, NSM) == 0


def _host(nsm: int):
    nb = nsm + 2
    return (
        torch.empty((32768, 12), dtype=torch.int32),
        torch.empty(nb, dtype=torch.int32),
        torch.empty((32768, 8), dtype=torch.int32),
        torch.empty(nb, dtype=torch.int32),
    )


@pytest.fixture(scope="module")
def built_planner():
    planner.build()
    return planner


@pytest.mark.parametrize("reqs", MIXED)
@pytest.mark.parametrize("nsm", REDUCED)
def test_planner_reduced_grid_padding(built_planner, reqs, nsm):
    ql, kv = [r[0] for r in reqs], [r[1] for r in reqs]
    plan_nsm = max(2, nsm - nsm % 2)
    hp, hb, hm, hmb = _host(NSM)
    res = built_planner.plan_into(
        ql, kv, plan_nsm, hp, hb, hm, hmb, (), (0.25, 0.5, 1, 2, 4), (), False
    )
    nwork, nbins, nmrg, nmbin, _, _, _, _, grid, use_mcast = res
    assert nwork > 0
    # The planner's own extents follow the reduced SM count.
    assert grid == ((plan_nsm // 2) * 2 if use_mcast else plan_nsm)
    assert nbins == (plan_nsm // 2 if use_mcast else plan_nsm) + 1
    assert nmbin == grid + 1
    bins = hb[:nbins].tolist()
    mbin = hmb[:nmbin].tolist()
    # Bins are monotone and cover every plan row once; merge bins every merge row.
    assert bins[0] == 0 and bins[-1] == nwork
    assert all(a <= b for a, b in zip(bins, bins[1:]))
    assert mbin[0] == 0 and mbin[-1] in (nmrg, 0)
    assert all(a <= b for a, b in zip(mbin, mbin[1:]))

    # Pad to the compiled extent of a full-SM kernel (runtime.plan does this).
    total = NSM + 1
    assert pad_bins(hb, nbins, total) == total
    assert pad_bins(hmb, nmbin, total) == total
    kbins = (NSM // 2 if use_mcast else NSM) + 1  # kernel's mBin extent
    kgrid = (NSM // 2) * 2 if use_mcast else NSM  # kernel's grid (mMBin extent - 1)
    pb, pm = hb[:kbins].tolist(), hmb[: kgrid + 1].tolist()
    assert pb[:nbins] == bins and pm[:nmbin] == mbin
    for c in range(kbins - 1):  # CTA bin c: [pb[c], pb[c+1])
        if c >= nbins - 1:
            assert pb[c + 1] - pb[c] == 0
    for c in range(kgrid):
        if c >= nmbin - 1:
            assert pm[c + 1] - pm[c] == 0


def _stub_kernel_module(monkeypatch):
    """runtime.KfPrefillAttn only reads constants from the CuTe kernel module."""
    stub = types.ModuleType("vllm.third_party.kf_prefill_attn.kernel")
    for name, value in {
        "NSLOT_CAP": 1520,
        "MROW": 128,
        "LOG2E": 1.4426950408889634,
        "NOPS": 60,
        "_BASE_WAVE_GRID": (0.25, 0.5, 1, 2, 4),
    }.items():
        setattr(stub, name, value)
    monkeypatch.setitem(sys.modules, "vllm.third_party.kf_prefill_attn.kernel", stub)
    pkg = sys.modules.get("vllm.third_party.kf_prefill_attn")
    if pkg is not None:
        monkeypatch.setattr(pkg, "kernel", stub, raising=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device ring buffers")
@pytest.mark.parametrize("nsm", [0, 128, 106])
def test_runtime_plan_pads_device_tables(monkeypatch, nsm):
    from vllm.v1.attention.ops.kf_prefill_attn.runtime import KfPrefillAttn

    _stub_kernel_module(monkeypatch)
    dev = torch.device("cuda")
    rt = KfPrefillAttn(dev, torch.zeros(394 << 20, dtype=torch.uint8, device=dev))
    rt.nsm = NSM  # the kernel's compiled grid on VR (the local GPU may differ)
    nb = NSM + 2
    rt.host = [tuple(torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                     for t in _host(NSM)) for _ in rt.host]  # fmt: skip
    rt.dev = [tuple(torch.empty_like(t, device=dev) for t in h) for h in rt.host]
    assert rt.host[0][1].numel() == nb
    rt.planner.build()
    rt.min_kv = 0
    rt.compiled = {v: object() for v in range(400)}
    for reqs in MIXED:
        ql, kv = [r[0] for r in reqs], [r[1] for r in reqs]
        bt = torch.zeros((len(reqs), MAXP), dtype=torch.int32, device=dev)
        plan = rt.plan(ql, kv, bt, [True] * len(reqs), nsm=nsm)
        assert plan is not None
        torch.accelerator.synchronize()
        hp, hb, hm, hmb = _host(NSM)
        ref = rt.planner.plan_into(
            ql, kv, NSM if nsm == 0 else nsm, hp, hb, hm, hmb, *rt.grids
        )
        nwork, nbins, nmrg, nmbin = ref[:4]
        assert plan.nwork == nwork
        assert torch.equal(plan.plan.cpu(), hp[:nwork])
        assert torch.equal(plan.mrg.cpu(), hm[:nmrg])
        if nsm == 0:
            # Unchanged production tables.
            assert torch.equal(plan.bins.cpu(), hb[:nbins])
            assert torch.equal(plan.mbin.cpu(), hmb[:nmbin])
        else:
            b, m = plan.bins.cpu(), plan.mbin.cpu()
            assert b.numel() == NSM + 1 and m.numel() == NSM + 1
            assert torch.equal(b[:nbins], hb[:nbins])
            assert torch.equal(m[:nmbin], hmb[:nmbin])
            assert bool((b[nbins:] == hb[nbins - 1]).all())
            assert bool((m[nmbin:] == hmb[nmbin - 1]).all())


class _FakeRuntime:
    """Records the stream of each launch; the 'kernel' spins, then fills out."""

    def __init__(self):
        self.streams: list[int] = []

    def launch(self, plan, q, kv, out, bmm1, bmm2):
        s = torch.cuda.current_stream()
        self.streams.append(s.cuda_stream)
        torch.cuda._sleep(20_000_000)
        out.fill_(1.0)
        return True


def _impl_stub():
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    stub = types.SimpleNamespace(
        window_left=-1,
        sinks=None,
        logits_soft_cap=None,
        num_heads=16,
        head_size=256,
        bmm1_scale=0.0625,
        bmm2_scale=1.0,
        _kf_join=None,
    )
    stub._kf_fork_join = types.MethodType(FlashInferImpl._kf_fork_join, stub)
    return stub, FlashInferImpl


@pytest.mark.skipif(not torch.cuda.is_available(), reason="streams")
@pytest.mark.parametrize("env_on,fork", [(True, True), (True, False), (False, True)])
def test_kf_prefill_fork_streams(monkeypatch, env_on, fork):
    import vllm.v1.attention.backends.flashinfer as fi

    monkeypatch.setenv("VLLM_ATTN_MIXED_FORK", "1" if env_on else "0")
    fake = _FakeRuntime()
    monkeypatch.setattr(fi, "get_runtime", lambda dev: fake)
    monkeypatch.setattr(fi, "kf_supported_kv", lambda kv: True)
    stub, impl = _impl_stub()
    dev = torch.device("cuda")
    meta = types.SimpleNamespace(prefill=fi.TRTLLMPrefill.__new__(fi.TRTLLMPrefill))
    meta.prefill.kf = object()
    q = torch.zeros((8, 16, 256), dtype=fi.FP8_DTYPE, device=dev)
    out = torch.zeros((8, 16 * 256), dtype=torch.bfloat16, device=dev)
    main = torch.cuda.current_stream()
    kv = torch.empty(0, device=dev)
    assert impl._kf_prefill(stub, None, meta, q, kv, out, fork=fork)
    forked = env_on and fork
    side = fi._attn_mixed_fork_streams()[0]
    assert fake.streams == [side.cuda_stream if forked else main.cuda_stream]
    assert (stub._kf_join is not None) == forked
    assert side.priority == torch.cuda.Stream.priority_range()[1]
    stub._kf_fork_join()
    assert stub._kf_join is None
    # The current stream is ordered after the (spinning) launch.
    got = out.float().sum()
    assert got.item() == out.numel()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not __import__(
        "vllm.platforms", fromlist=["current_platform"]
    ).current_platform.is_device_capability(107),
    reason="SM107 kernel",
)
@pytest.mark.parametrize("nsm", [160, 106])
def test_reduced_grid_kernel_and_fork_bitwise(nsm):
    pytest.importorskip("cutlass")
    from vllm.v1.attention.ops.kf_prefill_attn.runtime import KfPrefillAttn

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    rt = KfPrefillAttn(dev, torch.zeros(394 << 20, dtype=torch.uint8, device=dev))
    rt.planner.build()
    rt.min_kv = 0
    reqs = [[700, 40_000], [300, 22_000], [1200, 51_000]]
    ql, kv = [r[0] for r in reqs], [r[1] for r in reqs]
    hp, hb, hm, hmb = rt.host[0]
    forms = set()
    for n in (rt.nsm, nsm):
        forms.add(rt.planner.plan_into(ql, kv, n, hp, hb, hm, hmb, *rt.grids)[6])
    rt.warmup(precise=False, variants=sorted(forms))
    pages = [(L + 127) // 128 for L in kv]
    npage = sum(pages) + 1
    kvc = (torch.randn((npage, 2, 128, 512), generator=gen, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    bt = torch.zeros((len(reqs), MAXP), dtype=torch.int32, device=dev)
    o = 1
    for i, p in enumerate(pages):
        bt[i, :p] = torch.arange(o, o + p, dtype=torch.int32, device=dev)
        o += p
    q = (torch.randn((sum(ql), 16, 256), generator=gen, device=dev)).to(
        torch.float8_e4m3fn
    )

    def run(n, stream=None):
        plan = rt.plan(ql, kv, bt, [True] * len(reqs), nsm=n)
        assert plan is not None
        out = torch.full((sum(ql), 16, 256), float("nan"), dtype=torch.bfloat16,
                         device=dev)  # fmt: skip
        if stream is None:
            assert rt.launch(plan, q, kvc, out, 0.0625, 1.0)
        else:
            ev = torch.cuda.Event()
            ev.record()
            with torch.cuda.stream(stream):
                ev.wait(stream)
                assert rt.launch(plan, q, kvc, out, 0.0625, 1.0)
            torch.cuda.current_stream().wait_stream(stream)
        torch.accelerator.synchronize()
        assert bool((rt.cnt == 0).all())
        return out

    full = run(0)
    red = run(nsm)
    assert torch.isfinite(red.float()).all()
    rel = float((red.float() - full.float()).norm() / full.float().norm())
    assert rel < 2e-2, rel
    _, high = torch.cuda.Stream.priority_range()
    side = torch.cuda.Stream(priority=high)
    assert torch.equal(run(nsm, side), red)
    assert torch.equal(run(0, side), full)
