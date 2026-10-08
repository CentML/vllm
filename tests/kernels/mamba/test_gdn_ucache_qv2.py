# SPDX-License-Identifier: Apache-2.0
"""U-cache GDN decode on qwen-v2 (ops/gdn_ucache) vs its deferred-state reference
gdn_state_commit and vs qwen-v2's gdn_mtp_cuda. GPU (sm_100+). Prints "UCQ <tag> <json>" lines.

  1. test_bitwise_vs_gsc       16 chained steps (random acceptance, a NULL row, flushes): outputs, checkpoint
                               blocks, rings and cursors BITWISE equal to gsc's u-cache on its page layout; the
                               band-exit prepass (PRE_STOCK, column acc - 1) bitwise equal to gsc materialize mode 0.
                               Run for the default kernel and, when the gsc build has it, VLLM_GDN_UCACHE_KQFIX.
  2. test_r1_r2_folds          R2 (in place incl. bias 0, and to another block) and R1 (to the new block) folds
                               bitwise equal to gsc materialize mode 0 with n = bias + 1 / acc; inactive slots
                               untouched (the stock kernels copy them); tag / active bookkeeping.
  3. test_band_transitions     u-cache <-> gdn_mtp_cuda switching through the prepass on one pool (float-order
                               class): every step's outputs and the final state within a rel-L2 bound of a pure
                               gdn_mtp_cuda chain; a wrong transition (wrong column / stale ring) is O(1).
  4. test_short_rows_staged    T = 1 rows (zero-draft rows, gdnfix) in a mixed-T call: staged path, NaN-free,
                               within the float-order bound of gdn_mtp_cuda; the 4-token rows unchanged vs a
                               uniform call's numerics class.
  5. test_slot_reuse           a new request in a reused slot and block (reset_slot) starts fresh: bitwise equal
                               to a never-used slot.
"""
import importlib.util
import json
import os
import types

import pytest
import torch

H, HV, K, V, T = 16, 32, 128, 128, 4
SLOTS, BS, C0 = 256, 1120, 2  # ring slots, mamba block size, spec start column
dev = "cuda"


def _gpu_ok():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _log(tag, d):
    print("UCQ", tag, json.dumps(d), flush=True)


def _cfg():
    return types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=SLOTS),
        compilation_config=types.SimpleNamespace(cudagraph_capture_sizes=[4 * SLOTS]),
        cache_config=types.SimpleNamespace(mamba_cache_mode="align", mamba_ssm_cache_dtype="auto"),
        use_v2_model_runner=True, num_speculative_tokens=3)


def _rel(x, r):
    return float((x.float() - r.float()).norm() / r.float().norm())


class _World:
    """One GDN layer: qwen-v2 page pool (conv + ssm, no log) for gdn_ucache / gdn_mtp_cuda, a gsc page pool (+ log),
    block table [N, 8] (spec columns C0..C0+3), request slots, initial states."""

    def __init__(self, G, U, N, seed, A_range=(1, 16), kq=False):
        self.G, self.U, self.N = G, U, N
        g = torch.Generator(device=dev).manual_seed(seed)
        gc = torch.Generator().manual_seed(seed)
        self.g, self.gc = g, gc
        self.PB = 5 * N + 8
        self.conv = (2 * H * K + HV * V) * 6 * 2
        self.page_q = self.conv + HV * V * K * 2
        self.page_g = ((self.conv + HV * V * K * 2 + G.log_bytes(H, HV) + 16383) // 16384) * 16384
        self.buf_q = torch.zeros(self.PB, self.page_q, dtype=torch.uint8, device=dev)
        self.ssm_q = torch.as_strided(self.buf_q.view(torch.bfloat16), (self.PB, HV, V, K),
                                      (self.page_q // 2, V * K, K, 1), self.conv // 2)
        self.buf_g = torch.zeros(self.PB, self.page_g, dtype=torch.uint8, device=dev)
        self.ssm_g = torch.as_strided(self.buf_g.view(torch.bfloat16), (self.PB, HV, V, K),
                                      (self.page_g // 2, V * K, K, 1), self.conv // 2)
        self.A_log = torch.log(torch.empty(HV, device=dev).uniform_(*A_range, generator=g))
        self.dt_bias = (torch.randn(HV, device=dev, generator=g) * 0.5 + 1).to(torch.bfloat16)
        self.w = (1 + 0.1 * torch.randn(V, device=dev, generator=g)).to(torch.bfloat16)
        self.null = N // 2
        perm = (torch.randperm(self.PB - 1, generator=gc) + 1).to(torch.int32)
        self.bt = torch.zeros(N, 8, dtype=torch.int32)
        self.bt[:, C0:C0 + 5] = perm[:5 * N].view(N, 5)
        self.bt[self.null] = 0
        self.bt = self.bt.to(dev)
        self.slots = torch.randperm(SLOTS, generator=gc)[:N].to(torch.int32).to(dev)
        self.seq_lens = torch.full((N,), C0 * BS + 7, dtype=torch.int32, device=dev)
        live = torch.ones(N, dtype=torch.bool)
        live[self.null] = False
        self.live = live.to(dev)
        init = (torch.randn(N, HV, V, K, device=dev, generator=g) * 0.05).to(torch.bfloat16)
        self.ssm_q.copy_(torch.randn(self.PB, HV, V, K, device=dev, generator=g).to(torch.bfloat16))  # garbage
        c0 = self.bt[self.live, C0].long()
        self.ssm_q[c0] = init[self.live]
        self.ssm_g[c0] = init[self.live]  # gsc: one block per request = the same id as column C0
        self.si_q = self.bt[:, C0:C0 + 4].contiguous()
        self.si_g = self.bt[:, C0:C0 + 1].contiguous()
        self.lay_q = self._layer_q()
        self.lay_g = self._layer_g()

    def _layer_q(self):
        lay = types.SimpleNamespace()
        lay.A_log = torch.nn.Parameter(self.A_log.clone(), requires_grad=False)
        lay.dt_bias = self.dt_bias
        self.U.register_layer(lay, H, HV, _cfg())
        lay.kv_cache = [torch.zeros(1, device=dev), self.ssm_q]
        return lay

    def _layer_g(self):
        class _L:
            pass
        lay = _L()
        lay.A_log = torch.nn.Parameter(self.A_log.clone(), requires_grad=False)
        self.G._UC.register_layer(lay, H, HV, _cfg())
        return lay

    def bind(self):
        """gdn_ucache's worker fold table for this world's block table (one mamba group)."""
        ctx = types.SimpleNamespace(mamba_group_ids=[0], block_size=BS, block_table_stride_req=self.bt.stride(0),
                                    block_table_ptrs=torch.tensor([self.bt.data_ptr()], dtype=torch.int64,
                                                                  device=dev))
        kvc = types.SimpleNamespace(kv_cache_groups=[types.SimpleNamespace(layer_names=["L0"])])
        self.U.UC.layers = [self.lay_q]
        self.U.bind_worker(ctx, kvc, {"L0": self.lay_q})

    def inputs(self, Ts=None):
        N = self.N
        Ts = Ts or [T] * N
        TT = sum(Ts)
        # exactly TT token rows: gdn_mtp_cuda's contract needs qkv / a / b / gate / out with the same row count
        qkvz = torch.randn(TT, 2 * H * K + 2 * HV * V, device=dev, dtype=torch.bfloat16, generator=self.g)
        ba = torch.randn(TT, 2 * HV, device=dev, dtype=torch.bfloat16, generator=self.g)
        qkv, gate = qkvz[:, :2 * H * K + HV * V], qkvz[:, 2 * H * K + HV * V:].view(-1, HV, V)
        b, a = ba[:, :HV], ba[:, HV:]
        cu = torch.zeros(N + 1, dtype=torch.int32)
        cu[1:] = torch.tensor(Ts).cumsum(0)
        return qkv, gate, a, b, cu.to(dev)

    def accept(self, Ts=None):
        acc = torch.multinomial(torch.tensor([0.215, 0.151, 0.106, 0.528]), self.N, replacement=True,
                                generator=self.gc) + 1
        if Ts is not None:
            acc = torch.minimum(acc, torch.tensor(Ts))
        return acc.to(torch.int32).to(dev)


# Ring dtype and kq-fix are PROCESS-WIDE in the kernel module / gsc (read at import): the job runs this file once per
# ring dtype (UCQ_RING=bf16|fp16). KQ (UCQ_KQ, default "0,1" where gsc has it): kq-fix modes; every test resets the
# port's and gsc's mode explicitly (no state leaks between tests).
RING = os.environ.get("UCQ_RING", "bf16")
os.environ["GDN_UCACHE_RING_DTYPE"] = RING


def _kq_modes(G):
    """kq-fix levels to test (UCQ_KQ, default 0,1,2): level > 0 needs a gsc that has UCACHE_KQFIX; level 2 a gsc /
    kernel that parse it as an int (KQFIX=2 builds)."""
    want = [int(x) for x in os.environ.get("UCQ_KQ", "0,1,2").split(",") if x]
    return [k for k in want if not k or hasattr(G, "UCACHE_KQFIX")]


def _set_kq(G, U, kq):
    kq = int(kq)
    U.KQFIX = kq
    U._KQ = {"kq_fix": kq} if kq else {}
    if hasattr(G, "UCACHE_KQFIX"):
        G.UCACHE_KQFIX = kq


def _load(kq=0):
    os.environ.setdefault("GDN_STATE_COMMIT", "1")
    for k, v in dict(CK="3", MINB="4", CK3_MAP="1", CK3_ORDER="1", CK3_REMAP="1").items():
        os.environ.setdefault("GDN_STATE_COMMIT_" + k, v)
    os.environ["VLLM_GDN_DECODE_UCACHE"] = "1"
    from vllm.model_executor.layers.mamba.ops import gdn_state_commit as G
    from vllm.model_executor.layers.mamba.ops import gdn_ucache as U
    G._UC = G._Ucache()
    G._ext, G._exts = None, {}
    uc = G.load(4)
    G._ext = uc
    U.ENABLED = True
    U.UC = U._State()
    _set_kq(G, U, kq)
    # The gsc reference keeps an address-keyed fp32 norm-weight cache: drop it per test, so a new world's weight
    # at a reused address cannot hit a freed one (harmless in serving: persistent parameters)
    getattr(G, "_NW32", {}).clear()
    return G, U, uc


def _mtp():
    src = os.environ.get("UCQ_MTP_SRC")
    if not src:
        return None
    spec = importlib.util.spec_from_file_location("qv2_gdn_mtp_cuda", src)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    return M.build(M.tuned_source("", False))


def _uc_q(U, W, qkv, gate, a, b, cu, nacc, out):
    U.decode(W.lay_q, qkv, a, b, W.si_q, cu, nacc, W.ssm_q, gate, W.w, out, K ** -0.5, 1e-6, False)


def _uc_g(G, uc, W, qkv, gate, a, b, cu, nacc, out, kq):
    G.UCACHE_FUSED_NORM = True
    try:
        uc.decode(qkv, a, b, W.lay_g.A_log, W.dt_bias, W.si_g, cu, nacc, W.ssm_g, gate, W.w, out, K ** -0.5, 1e-6,
                  False)
    finally:
        G.UCACHE_FUSED_NORM = False


def _canary(n):
    return torch.full((n, HV, V), -1, dtype=torch.int16, device=dev).view(torch.bfloat16)


def _eq(x, y):
    return int((x.contiguous().view(torch.int16) != y.contiguous().view(torch.int16)).sum()) if x.dtype != torch.float32 \
        else int((x.view(torch.int32) != y.view(torch.int32)).sum())


def _chain(G, U, uc, W, steps, kq):
    """Both u-caches over `steps` steps; returns (mismatch counters, last acc)."""
    U.UC.slot[:W.N].copy_(W.slots)
    G._UC.slot[:W.N].copy_(W.slots)
    U.set_step(True, True)
    acc = torch.ones(W.N, dtype=torch.int32, device=dev)
    mm = dict(out=0, page=0, ring=0, cursor=0, null_nonzero=0, canary=0)
    rq = W.lay_q._gdn_ucache
    for _ in range(steps):
        qkv, gate, a, b, cu = W.inputs()
        oq, og = _canary(4 * W.N), _canary(4 * W.N)
        _uc_q(U, W, qkv, gate, a, b, cu, acc, oq)
        _uc_g(G, uc, W, qkv, gate, a, b, cu, acc, og, kq)
        torch.cuda.synchronize()
        mm["out"] += _eq(oq, og)
        mm["canary"] += int((oq.view(torch.int16) == -1).sum())
        mm["null_nonzero"] += int((oq.view(W.N, 4, HV, V)[W.null].view(torch.int16) != 0).sum())
        c0 = W.bt[W.live, C0].long()
        mm["page"] += _eq(W.ssm_q[c0], W.ssm_g[c0])
        kg, ug, gg, cg = G._UC.buffers(W.ssm_g, W.lay_g.A_log)[:4]
        sl = W.slots[W.live].long()
        mm["ring"] += _eq(rq["kr"][sl], kg[sl]) + _eq(rq["ur"][sl], ug[sl]) + _eq(rq["gr"][sl], gg[sl])
        mm["cursor"] += int((rq["cur"][sl] != cg[sl]).sum())
        acc = W.accept()
    return mm, acc


@pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_100+ GPU")
def test_bitwise_vs_gsc():
    G, U, uc = _load()
    for kq in _kq_modes(G):
        _set_kq(G, U, kq)
        getattr(G, "_NW32", {}).clear()  # see _load: a new world per mode
        W = _World(G, U, 40, seed=11 + int(kq))
        W.bind()
        uc.ucache_errors(True)
        U.errors(True)
        mm, acc = _chain(G, U, uc, W, 16, kq)
        # band exit: gsc mode 0 in place (n = acc) vs gdn_ucache PRE_STOCK into column acc - 1
        live = W.live
        bl = W.bt[live, C0]
        al = acc[live]
        n = int(live.sum())
        G.materialize(0, n, G.LayerTable([(W.ssm_g, W.lay_g.A_log, W.dt_bias, 0)], dev), H, bl, al)
        U.set_step(False, True)
        U.prepass(W.N, W.slots, W.seq_lens, _by_slot(acc, W.slots))
        torch.cuda.synchronize()
        dst = W.bt[live].gather(1, (C0 + al.long() - 1)[:, None])[:, 0].long()
        mm["prepass_state"] = _eq(W.ssm_q[dst], W.ssm_g[bl.long()])
        mm["active_left"] = int(U.UC.active[W.slots[live].long()].sum())
        e = U.errors(True)
        eg = uc.ucache_errors(True).tolist()
        r = dict(kq=kq, **mm, err=e[:8], err_gsc=eg[:4])
        r["ok"] = (all(mm[k] == 0 for k in ("out", "page", "ring", "cursor", "null_nonzero", "canary",
                                             "prepass_state", "active_left"))
                   and e[0] == 0 and e[1] == 0 and e[2] == 0 and e[3] == eg[3])
        _log("bitwise", r)
        assert r["ok"], r


def _by_slot(acc, slots):
    """num_accepted indexed by request slot (the prepass reads it by slot, as num_accepted_tokens_gpu)."""
    t = torch.ones(SLOTS, dtype=torch.int32, device=dev)
    t[slots.long()] = acc
    return t


@pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_100+ GPU")
@pytest.mark.parametrize("kq", [0, 1, 2])
def test_r1_r2_folds(kq):
    G, U, uc = _load(kq)
    if kq not in _kq_modes(G):
        return
    W = _World(G, U, 40, seed=23)
    W.bind()
    mm, acc = _chain(G, U, uc, W, 7, kq)
    assert mm["out"] == 0 and mm["page"] == 0, mm
    live = W.live
    N = W.N
    gc = torch.Generator().manual_seed(5)
    # ---- R2: per row bias in [0, acc - 1]; even rows in place (dst col C0), odd rows to column C0 + 1
    bias = (torch.rand(N, generator=gc).to(dev) * acc.float()).floor().to(torch.int32)
    to_next = (torch.arange(N, device=dev) % 2 == 1)
    aligned = torch.where(to_next, (C0 + 2) * BS, (C0 + 1) * BS).to(torch.int32)
    running = aligned - bias
    new_computed = running + acc - 1
    state_idx = torch.full((SLOTS,), C0, dtype=torch.int32, device=dev)
    g2 = W.buf_g.clone()  # expected: gsc materialize mode 0 (in place) on a copy, n = bias + 1
    ssm_g2 = torch.as_strided(g2.view(torch.bfloat16), W.ssm_g.shape, W.ssm_g.stride(), W.conv // 2)
    bl = W.bt[live, C0]
    G.materialize(0, int(live.sum()), G.LayerTable([(ssm_g2, W.lay_g.A_log, W.dt_bias, 0)], dev), H, bl,
                  (bias + 1)[live])
    U.fold_r2(N, _by_slot(acc, W.slots), state_idx, _by_slot(new_computed, W.slots), W.slots)
    torch.cuda.synchronize()
    dcol = torch.where(to_next, C0 + 1, C0)
    dst = W.bt.gather(1, dcol[:, None].long())[:, 0].long()
    tags = W.lay_q._gdn_ucache["tags"]
    src_odd = W.bt[live & to_next, C0].long()
    r = dict(r2_mismatch=_eq(W.ssm_q[dst[live]], ssm_g2[bl.long()]),
             r2_dst_tag_left=int((tags[dst[live]] != 0).sum()),
             r2_src_tag_kept=int((tags[src_odd] != 0).all(dim=1).sum()) == int(src_odd.numel()),
             r2_in_place_rows=int((live & ~to_next).sum()), r2_zero_bias_rows=int((live & (bias == 0)).sum()))
    r["kq"] = kq
    _log("r2", r)
    assert r["r2_mismatch"] == 0 and r["r2_dst_tag_left"] == 0 and r["r2_src_tag_kept"], r

    G, U, uc = _load(kq)
    if kq not in _kq_modes(G):
        return
    W = _World(G, U, 40, seed=29)
    W.bind()
    mm, acc = _chain(G, U, uc, W, 9, kq)
    assert mm["out"] == 0, mm
    inactive_slot = W.slots[0].long()
    U.UC.active[inactive_slot] = 0
    g2 = W.buf_g.clone()
    ssm_g2 = torch.as_strided(g2.view(torch.bfloat16), W.ssm_g.shape, W.ssm_g.stride(), W.conv // 2)
    bl = W.bt[live, C0]
    G.materialize(0, int(live.sum()), G.LayerTable([(ssm_g2, W.lay_g.A_log, W.dt_bias, 0)], dev), H, bl, acc[live])
    dst_col = torch.full((SLOTS,), C0 + 2, dtype=torch.int32, device=dev)
    src_col = torch.full((SLOTS,), C0, dtype=torch.int32, device=dev)
    dst = W.bt[:, C0 + 2].long()
    before_inactive = W.ssm_q[dst[0]].clone()
    U.fold_r1(N, dst_col, src_col, _by_slot(acc - 1, W.slots), W.slots)
    torch.cuda.synchronize()
    act_rows = live.clone()
    act_rows[0] = False
    r = dict(r1_mismatch=_eq(W.ssm_q[dst[act_rows]], ssm_g2[W.bt[act_rows, C0].long()]),
             r1_inactive_untouched=_eq(W.ssm_q[dst[0]], before_inactive),
             r1_dst_tag_left=int((W.lay_q._gdn_ucache["tags"][dst[act_rows]] != 0).sum()),
             err=U.errors(True)[:8])
    r["ok"] = r["r1_mismatch"] == 0 and r["r1_inactive_untouched"] == 0 and r["r1_dst_tag_left"] == 0
    r["kq"] = kq
    _log("r1", r)
    assert r["ok"], r


@pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_100+ GPU")
@pytest.mark.parametrize("kq", [0, 1, 2])
def test_band_transitions(kq):
    mtp = _mtp()
    if mtp is None:
        pytest.skip("UCQ_MTP_SRC (qwen-v2 gdn_mtp_cuda.py) not set")
    G, U, uc = _load(kq)
    if kq not in _kq_modes(G):
        return
    W = _World(G, U, 40, seed=31, A_range=(0.05, 2.0))
    W.bind()
    # reference: a pure gdn_mtp_cuda chain on a copy of the pool (stock layout)
    ref_buf = W.buf_q.clone()
    ref_ssm = torch.as_strided(ref_buf.view(torch.bfloat16), W.ssm_q.shape, W.ssm_q.stride(), W.conv // 2)
    U.UC.slot[:W.N].copy_(W.slots)
    sched = [1] * 5 + [0] * 3 + [1] * 4 + [0] * 4 + [1] * 3  # 1: u-cache step, 0: gdn_mtp_cuda step
    acc = torch.ones(W.N, dtype=torch.int32, device=dev)
    worst, final = 0.0, None
    for s, band in enumerate(sched):
        qkv, gate, a, b, cu = W.inputs()
        U.set_step(bool(band), True)
        U.prepass(W.N, W.slots, W.seq_lens, _by_slot(acc, W.slots))
        o, o_ref = _canary(4 * W.N), _canary(4 * W.N)
        if band:
            _uc_q(U, W, qkv, gate, a, b, cu, acc, o)
        else:
            assert mtp.run(qkv, a, b, W.A_log, W.dt_bias, W.si_q, cu, acc, W.ssm_q, gate, W.w, o, K ** -0.5, 1e-6,
                           False)
        assert mtp.run(qkv, a, b, W.A_log, W.dt_bias, W.si_q, cu, acc, ref_ssm, gate, W.w, o_ref, K ** -0.5, 1e-6,
                       False)
        torch.cuda.synchronize()
        lv = W.live.repeat_interleave(4)
        assert bool(torch.isfinite(o[lv].float()).all())
        worst = max(worst, _rel(o[lv], o_ref[lv]))
        acc = W.accept()
    # final committed state: leave the band (last step was u-cache) and compare column acc - 1
    U.set_step(False, True)
    U.prepass(W.N, W.slots, W.seq_lens, _by_slot(acc, W.slots))
    torch.cuda.synchronize()
    col = W.bt[W.live].gather(1, (C0 + acc[W.live].long() - 1)[:, None])[:, 0].long()
    final = _rel(W.ssm_q[col], ref_ssm[col])
    r = dict(out_rel_max=round(worst, 6), state_rel=round(final, 6), err=U.errors(True)[:8])
    r["ok"] = worst < 2e-2 and final < 2e-2 and r["err"][0] == 0 and r["err"][1] == 0 and r["err"][2] == 0
    r["kq"] = kq
    _log("transitions", r)
    assert r["ok"], r


@pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_100+ GPU")
@pytest.mark.parametrize("kq", [0, 1, 2])
def test_short_rows_staged(kq):
    mtp = _mtp()
    if mtp is None:
        pytest.skip("UCQ_MTP_SRC (qwen-v2 gdn_mtp_cuda.py) not set")
    G, U, uc = _load(kq)
    if kq not in _kq_modes(G):
        return
    W = _World(G, U, 40, seed=37, A_range=(0.05, 2.0))
    W.bind()
    ref_buf = W.buf_q.clone()
    ref_ssm = torch.as_strided(ref_buf.view(torch.bfloat16), W.ssm_q.shape, W.ssm_q.stride(), W.conv // 2)
    U.UC.slot[:W.N].copy_(W.slots)
    acc = torch.ones(W.N, dtype=torch.int32, device=dev)
    worst, short_worst, staged0 = 0.0, 0.0, U.STATS.get("staged", 0)
    for s in range(10):
        Ts = [1 if (i + s) % 5 == 0 else T for i in range(W.N)] if s % 2 else [T] * W.N
        qkv, gate, a, b, cu = W.inputs(Ts)
        U.set_step(True, all(t == T for t in Ts))
        TT = int(cu[-1])
        o, o_ref = _canary(TT), _canary(TT)
        _uc_q(U, W, qkv, gate, a, b, cu, acc, o)
        assert mtp.run(qkv, a, b, W.A_log, W.dt_bias, W.si_q, cu, acc, ref_ssm, gate, W.w, o_ref, K ** -0.5, 1e-6,
                       False)
        torch.cuda.synchronize()
        rows = torch.repeat_interleave(torch.arange(W.N, device=dev), (cu[1:] - cu[:-1]).long())
        lv = W.live[rows]
        assert bool(torch.isfinite(o[lv].float()).all())
        worst = max(worst, _rel(o[lv], o_ref[lv]))
        short = lv & torch.tensor([Ts[i] == 1 for i in rows.tolist()], device=dev)
        if bool(short.any()):
            short_worst = max(short_worst, _rel(o[short], o_ref[short]))
        acc = W.accept(Ts)
    r = dict(out_rel_max=round(worst, 6), short_rows_rel_max=round(short_worst, 6),
             staged_calls=U.STATS.get("staged", 0) - staged0, err=U.errors(True)[:8])
    r["ok"] = (worst < 2e-2 and short_worst < 2e-2 and r["staged_calls"] == 5 and r["err"][0] == 0
               and r["err"][1] == 0)
    r["kq"] = kq
    _log("short_rows", r)
    assert r["ok"], r


@pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_100+ GPU")
@pytest.mark.parametrize("kq", [0, 1, 2])
def test_slot_reuse(kq):
    G, U, uc = _load(kq)
    if kq not in _kq_modes(G):
        return
    W = _World(G, U, 40, seed=41)
    W.bind()
    _chain(G, U, uc, W, 6, kq)  # leaves valid tags (count c, slot r) on the C0 blocks
    # a new request takes the same slot and the same block with a new (prefilled) state
    fresh = (torch.randn(W.N, HV, V, K, device=dev, generator=W.g) * 0.05).to(torch.bfloat16)
    c0 = W.bt[W.live, C0].long()
    W.ssm_q[c0] = fresh[W.live]
    for s in W.slots.tolist():
        U.reset_slot(s)
    qkv, gate, a, b, cu = W.inputs()
    ones = torch.ones(W.N, dtype=torch.int32, device=dev)
    o = _canary(4 * W.N)
    _uc_q(U, W, qkv, gate, a, b, cu, ones, o)
    # never-used slots: a second world-free layer on a copy of the same blocks
    W2buf = W.buf_q.clone()
    W2ssm = torch.as_strided(W2buf.view(torch.bfloat16), W.ssm_q.shape, W.ssm_q.stride(), W.conv // 2)
    W2ssm[c0] = fresh[W.live]
    lay2 = types.SimpleNamespace(A_log=W.lay_q.A_log, dt_bias=W.dt_bias)
    U.register_layer(lay2, H, HV, _cfg())
    lay2.kv_cache = [None, W2ssm]
    o2 = _canary(4 * W.N)
    U.decode(lay2, qkv, a, b, W.si_q, cu, ones, W2ssm, gate, W.w, o2, K ** -0.5, 1e-6, False)
    torch.cuda.synchronize()
    r = dict(out_mismatch=_eq(o, o2), err=U.errors(True)[:8])
    r["ok"] = r["out_mismatch"] == 0 and r["err"][1] == 0
    r["kq"] = kq
    _log("slot_reuse", r)
    assert r["ok"], r


@pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_100+ GPU")
@pytest.mark.parametrize("kq", [0, 1, 2])
def test_paths_equivalent(kq):
    """Uniform T = 4 steps through the three u-cache launch paths on identical pools / rings: fused-norm strided
    (production), unfused strided, staged (compact copies; the T < 4 path). Expected BITWISE equal (outputs,
    checkpoints, rings), localizing path-specific kernel differences."""
    G, U, uc = _load(kq)
    if kq not in _kq_modes(G):
        return
    W = _World(G, U, 40, seed=47, A_range=(0.05, 2.0))
    W.bind()
    paths = {"fused": dict(F=True, S=True), "unfused": dict(F=False, S=True), "staged": dict(F=False, S=False)}
    pools, lays = {}, {}
    for p in paths:
        buf = W.buf_q.clone()
        ssm = torch.as_strided(buf.view(torch.bfloat16), W.ssm_q.shape, W.ssm_q.stride(), W.conv // 2)
        lay = types.SimpleNamespace(A_log=W.lay_q.A_log, dt_bias=W.dt_bias)
        U.register_layer(lay, H, HV, _cfg())
        lay.kv_cache = [None, ssm]
        pools[p], lays[p] = (buf, ssm), lay
    U.UC.slot[:W.N].copy_(W.slots)
    U.set_step(True, True)
    acc = torch.ones(W.N, dtype=torch.int32, device=dev)
    mm = {f"{a}_vs_fused_{k}": 0 for a in ("unfused", "staged") for k in ("out", "page", "ring")}
    rel_staged = 0.0
    for _ in range(12):
        qkv, gate, a, b, cu = W.inputs()
        outs = {}
        for p, flags in paths.items():
            U.FUSED_NORM, U.STRIDED = flags["F"], flags["S"]
            # every path keeps its own slot activity / cursors: swap the active bits per path
            act = U.UC.__dict__.setdefault(f"_act_{p}", torch.zeros_like(U.UC.active))
            saved = U.UC.active.clone()
            U.UC.active.copy_(act)
            o = _canary(4 * W.N)
            U.decode(lays[p], qkv, a, b, W.si_q, cu, acc, pools[p][1], gate, W.w, o, K ** -0.5, 1e-6, False)
            act.copy_(U.UC.active)
            U.UC.active.copy_(saved)
            outs[p] = o
        torch.cuda.synchronize()
        for p in ("unfused", "staged"):
            mm[f"{p}_vs_fused_out"] += _eq(outs[p], outs["fused"])
            mm[f"{p}_vs_fused_page"] += int((pools[p][0] != pools["fused"][0]).sum())
            bp, bf_ = lays[p]._gdn_ucache, lays["fused"]._gdn_ucache
            mm[f"{p}_vs_fused_ring"] += sum(int((bp[k].view(torch.uint8) != bf_[k].view(torch.uint8)).sum())
                                            for k in ("kr", "ur", "gr", "cur"))
        lv = W.live.repeat_interleave(4)
        rel_staged = max(rel_staged, _rel(outs["staged"][lv], outs["fused"][lv]))
        acc = W.accept()
    U.FUSED_NORM, U.STRIDED = True, True
    r = dict(kq=kq, ring=RING, **mm, staged_rel_max=round(rel_staged, 6), err=U.errors(True)[:8])
    r["ok"] = all(v == 0 for v in mm.values()) and r["err"][0] == 0 and r["err"][1] == 0
    _log("paths", r)
    assert r["ok"], r


def test_norm_weight_cache():
    """The fused epilogue's fp32 norm-weight cache must not reuse a freed tensor's
    cached values for a new tensor at the same storage address."""
    from vllm.model_executor.layers.mamba.ops import gdn_ucache as U
    stale = 0
    keep = (1 + 0.1 * torch.randn(128)).to(torch.bfloat16)
    same = U._norm_w32(keep) is U._norm_w32(keep)
    for _ in range(32):
        w = (1 + 0.1 * torch.randn(128)).to(torch.bfloat16)
        stale += int(not torch.equal(U._norm_w32(w), w.float()))
        del w
    r = dict(stale=stale, same_tensor_reused=same)
    _log("norm_w_cache", r)
    assert stale == 0 and same, r
