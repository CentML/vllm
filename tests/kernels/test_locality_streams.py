# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locality streams (BACKFILL green contexts), the HBM-clock gate and fork/join
on supported locality-domain GPUs. GPU checks are skipped elsewhere.

Runs under pytest, or standalone (``python test_locality_streams.py``; prints
``QV2L {json}`` lines and ``QV2L_DONE``, exit 1 on any failure). Standalone
without vLLM installed: set ``OVL_DIR`` to the tree holding
``vllm/model_executor/layers/locality`` and the ``vllm`` packages are stubbed.

Checks:
- backfill: the contexts' SM counts sum to every SM (no remainder) and
  equal the topology's physical-die counts; without backfill they equal the
  green-context domain counts and the remainder is left over.
- binding: a kernel on stream k reaches every SM of domain k and no SM of the
  other domain; the remainder SMs it gets sit on die k.
- gate: below VLLM_LOCALITY_MIN_MEMCLK locality is inactive (logged),
  get_domain_streams returns None and VLLM_LOCALITY_LM_HEAD does nothing; at
  or above it, active; unreadable clock -> active; gate 0 -> active.
- fork/join: eager result, exception path still joins, CUDA-graph capture and
  replay across the green streams.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import types

if os.environ.get("OVL_DIR"):
    _ovl = os.environ["OVL_DIR"]
    for _n in ("vllm", "vllm.model_executor", "vllm.model_executor.layers"):
        _m = types.ModuleType(_n)
        _m.__path__ = [os.path.join(_ovl, *_n.split("."))]
        sys.modules[_n] = _m
    _lg = types.ModuleType("vllm.logger")
    _lg.init_logger = logging.getLogger
    sys.modules["vllm.logger"] = _lg

import torch  # noqa: E402

try:
    import pytest
except ImportError:  # standalone
    pytest = None

from vllm.model_executor.layers.locality import gate, streams  # noqa: E402
from vllm.model_executor.layers.locality.topology import get_topology  # noqa: E402

DEV = 0


def _two_domains() -> bool:
    return torch.cuda.is_available() and get_topology(DEV) is not None


if pytest is not None:
    pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

RESULTS: list[dict] = []


def emit(kind: str, **rec) -> None:
    rec = dict(kind=kind, **rec)
    RESULTS.append(rec)
    print("QV2L", json.dumps(rec, default=str), flush=True)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs: list[str] = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def _streams(backfill: bool):
    streams._cache.pop((DEV, backfill), None)
    return streams.get_domain_streams(DEV, backfill=backfill)


def _with_gate_on():
    os.environ["VLLM_LOCALITY_MIN_MEMCLK"] = "0"  # this test is about topology, not the clock
    gate.locality_active(DEV, refresh=True)


def test_topology_and_backfill_counts():
    if not _two_domains():
        return _skip("fewer than 2 locality domains")
    _with_gate_on()
    topo = get_topology(DEV)
    emit("topology", **topo.summary())
    bf, nb = _streams(True), _streams(False)
    assert bf is not None and nb is not None
    green = (topo.green_map.count(0), topo.green_map.count(1))
    rem = topo.green_map.count(-1)
    emit("counts", backfill=bf.summary(), domain_only=nb.summary(), green=list(green), remainder=rem,
         die_sms=list(topo.domain_sms))
    assert sum(bf.sm_counts) == topo.num_sms, (bf.sm_counts, topo.num_sms)
    assert bf.remainder_sms == 0
    assert tuple(bf.sm_counts) == tuple(topo.domain_sms), (bf.sm_counts, topo.domain_sms)
    assert tuple(nb.sm_counts) == green and nb.remainder_sms == rem


def test_binding():
    if not _two_domains():
        return _skip("fewer than 2 locality domains")
    _with_gate_on()
    for backfill in (True, False):
        ds = _streams(backfill)
        for k, b in enumerate(ds.binding):
            emit("binding", backfill=backfill, **b)
            assert b["domain_sms_seen"] == b["domain_sms_expected"], b
            assert b["foreign_domain_sms"] == 0, b
            assert b["seen"] == ds.sm_counts[k], (b, ds.sm_counts)
            assert b["orphans_wrong_die"] == 0, b
            if not backfill:
                assert b["orphans_right_die"] == 0, b


def test_gate():
    h = _Capture()
    lg = logging.getLogger(gate.__name__)
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    real = gate.memclk_mhz
    saved = os.environ.get("VLLM_LOCALITY_MIN_MEMCLK")
    try:
        emit("memclk", mhz=real(DEV), gate=gate.DEFAULT_MIN_MEMCLK)
        os.environ.pop("VLLM_LOCALITY_MIN_MEMCLK", None)
        thr = gate.DEFAULT_MIN_MEMCLK
        cases = [
            (thr - 2, None, False),
            (thr - 1, None, False), (thr, None, True),
            (thr + 1, None, True),
            (None, None, True),
            (thr - 1, "0", True),
            (thr + 1, str(thr + 2), False)]
        for clk, env, want in cases:
            if env is None:
                os.environ.pop("VLLM_LOCALITY_MIN_MEMCLK", None)
            else:
                os.environ["VLLM_LOCALITY_MIN_MEMCLK"] = env
            gate.memclk_mhz = lambda dev, _c=clk: _c
            h.msgs.clear()
            got = gate.locality_active(DEV, refresh=True)
            logged = any(("INACTIVE" in m) for m in h.msgs) if not want else any("ctive" in m or "gate" in m for m in h.msgs)
            emit("gate", clk=clk, env=env, want=want, got=got, logged=logged)
            assert got == want and logged, (clk, env, got, h.msgs)
            if not want and torch.cuda.is_available():
                assert _streams(True) is None
                _check_lm_head_gated()
    finally:
        gate.memclk_mhz = real
        if saved is None:
            os.environ.pop("VLLM_LOCALITY_MIN_MEMCLK", None)
        else:
            os.environ["VLLM_LOCALITY_MIN_MEMCLK"] = saved
        gate._decided.clear()
        streams._cache.clear()
        lg.removeHandler(h)


def _check_lm_head_gated():
    from vllm.model_executor.layers.locality import lm_head, topology

    def boom(*a, **k):
        raise AssertionError("topology probed although the gate is closed")

    saved = (lm_head.TARGET_ON, topology.get_topology)
    lm_head.TARGET_ON = True
    topology.get_topology = boom
    try:
        lm_head.maybe_enable(None, None)  # must return at the gate
    finally:
        lm_head.TARGET_ON, topology.get_topology = saved
    emit("lm_head_gated", ok=True)


def test_fork_join_and_graph():
    if not _two_domains():
        return _skip("fewer than 2 locality domains")
    _with_gate_on()
    ds = _streams(True)
    x = torch.arange(1 << 20, device=f"cuda:{DEV}", dtype=torch.float32)
    out = [torch.empty_like(x) for _ in range(2)]

    def body(i):
        torch.mul(x, float(i + 1), out=out[i])

    for o in out:
        o.fill_(-1)
    ds.run(body)
    eager = all(torch.equal(out[i], x * (i + 1)) for i in range(2))

    # exception path: stream 0 ran, fn(1) raises; the caller must still be joined with stream 0
    for o in out:
        o.fill_(-1)

    def bad(i):
        if i == 1:
            raise RuntimeError("boom")
        body(i)

    raised = False
    try:
        ds.run(bad)
    except RuntimeError:
        raised = True
    joined = torch.equal(out[0].clone(), x)  # clone on the caller stream: ordered after the join

    # graph capture across the green streams
    for o in out:
        o.fill_(-1)
    g = torch.cuda.CUDAGraph()
    cap = torch.cuda.Stream(device=f"cuda:{DEV}")
    cap.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap):
        with torch.cuda.graph(g, stream=cap):
            ds.run(body)
    torch.cuda.current_stream().wait_stream(cap)
    for o in out:
        o.fill_(-1)
    g.replay()
    torch.cuda.synchronize()
    graph = all(torch.equal(out[i], x * (i + 1)) for i in range(2))
    emit("fork_join", eager=eager, exception_raised=raised, exception_joined=joined, graph=graph)
    assert eager and raised and joined and graph


def _skip(why: str):
    emit("skip", why=why)
    if pytest is not None and "PYTEST_CURRENT_TEST" in os.environ:
        pytest.skip(why)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fails = []
    for name in ("test_gate", "test_topology_and_backfill_counts", "test_binding", "test_fork_join_and_graph"):
        try:
            globals()[name]()
            emit("result", test=name, ok=True)
        except Exception as e:  # noqa: BLE001
            fails.append(name)
            emit("result", test=name, ok=False, err=repr(e)[:400])
    emit("summary", fails=fails)
    print("QV2L_DONE", flush=True)
    sys.exit(1 if fails else 0)
