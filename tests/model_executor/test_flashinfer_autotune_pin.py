# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned FlashInfer autotune tables (flashinfer_autotune_pin.py).

The GPU tests drive FlashInfer's real AutoTuner with a stand-in runner whose
inputs mimic mm_mxfp8's (a, b column-major, a_sf, b_sf, out dtype, out,
workspace), registered under the op name ``mxfp8_gemm`` so the mxfp8 fingerprint
rules (PFW / VLLM_MXFP8_* envs, weights-cold record batches) apply.
"""

import json
import multiprocessing
import os
from pathlib import Path

import pytest

import vllm.model_executor.warmup.flashinfer_autotune_pin as pin

HOST = {k: "x" for k in pin.HOST_KEYS}


def _rec_key(samples, tactics=None, *, g="mxfp8_gemm|R|fp0", tl=None, ra="ra"):
    tactics = tactics if tactics is not None else list(range(len(samples)))
    return {
        "op": "mxfp8_gemm",
        "runner": "R",
        "g": g,
        "ra": ra,
        "l2": "weights-cold",
        "tactics": tactics,
        "samples": samples,
        "tl": tl if tl is not None else pin.tactic_list_hash(tactics),
    }


def _record(keys, *, host=HOST, worker="w0", groups=None):
    groups = groups or {
        "mxfp8_gemm|R|fp0": {
            "op": "mxfp8_gemm",
            "runner": "R",
            "fp": "fp0",
            "parts": {"env": {}},
        }
    }
    return {
        "kind": pin.RECORD_KIND,
        "schema": pin.SCHEMA,
        "host": dict(host),
        "fi_meta": {k: host[k] for k in pin.FI_META_KEYS},
        "groups": groups,
        "worker": {"hostname": worker, "gpu_uuid": worker},
        "keys": keys,
    }


# --------------------------------------------------------------------------
# Merge / file format (CPU)
# --------------------------------------------------------------------------


def test_merge_picks_lowest_median_over_workers():
    fk = "('mxfp8_gemm', 'R', ((16, 128),), ())"
    # Tactic 1 is fastest on worker 0 only (one lucky worker = the lottery);
    # tactic 0 has the lowest median over workers. Worker 1's 50 us outlier
    # round does not move its median.
    recs = [
        _record({fk: _rec_key([[10, 10, 10], [9, 9, 30]])}, worker="w0"),
        _record({fk: _rec_key([[10, 10, 50], [11, 11, 11]])}, worker="w1"),
        _record({fk: _rec_key([[10, 10, 10], [12, 12, 12]])}, worker="w2"),
    ]
    table, report = pin.merge_records([("c512", recs)])
    runner, tactic, meta = table[fk]
    assert (runner, tactic) == ("R", 0)
    assert meta["us"] == 10 and meta["us2"] == 11
    assert meta["w"] == 3 and meta["agree"] == 2
    assert meta["tl"] == pin.tactic_list_hash([0, 1])
    t = pin.parse_table(table)
    assert not t.legacy and t.host == HOST
    assert [c.tactic for c in t.entries[fk]] == [0]
    assert report["keys"][0]["margin"] == pytest.approx(0.1)


def test_merge_rejects_inconsistent_workers():
    fk = "('mxfp8_gemm', 'R', ((16, 128),), ())"
    a = _record({fk: _rec_key([[1.0], [2.0]], [0, 1])})
    b = _record({fk: _rec_key([[1.0], [2.0], [3.0]], [0, 1, 2])})
    with pytest.raises(ValueError, match="different tactic lists"):
        pin.merge_records([("s", [a, b])])
    other = dict(HOST, sm_count="148")
    with pytest.raises(ValueError, match="different hosts"):
        pin.merge_records([("s", [a, _record({}, host=other)])])
    # A tactic that failed on one worker is never picked.
    c = _record({fk: _rec_key([[1.0], [2.0]], [0, 1])})
    d = _record({fk: _rec_key([None, [2.0]], [0, 1])})
    table, _ = pin.merge_records([("s", [c, d])])
    assert table[fk][1] == 1


def test_merge_variants_primary_top_level_and_alt():
    fk = "('mxfp8_gemm', 'R', ((16, 128),), ())"
    g_a, g_b = "mxfp8_gemm|R|fpA", "mxfp8_gemm|R|fpB"
    groups = {
        g: {"op": "mxfp8_gemm", "runner": "R", "fp": g[-3:], "parts": {}}
        for g in (g_a, g_b)
    }
    rec_a = _record({fk: _rec_key([[1.0], [2.0]], g=g_a)}, groups=groups)
    rec_b = _record({fk: _rec_key([[2.0], [1.0]], g=g_b)}, groups=groups)
    table, _ = pin.merge_records([("a", [rec_a]), ("b", [rec_b])], primary="b")
    # FlashInfer (and the #148 seeder) see only the primary variant.
    assert table[fk][1] == 1 and table[fk][2]["g"] == g_b
    t = pin.parse_table(table)
    assert [(c.gid, c.tactic) for c in t.entries[fk]] == [(g_b, 1), (g_a, 0)]


def test_legacy_flashinfer_file_and_host_mismatch():
    legacy = {
        "_metadata": {k: "x" for k in pin.FI_META_KEYS},
        "('mxfp8_gemm', 'R', ((1, 8),), ())": [
            "R",
            [[128, 16], [1, 1], True, False, 1],
        ],
        "_generation": "g",
    }
    t = pin.parse_table(legacy, path="legacy.json")
    assert t.legacy and t.num_entries() == 1
    (c,) = t.entries["('mxfp8_gemm', 'R', ((1, 8),), ())"]
    assert c.gid is None and c.tactic == [[128, 16], [1, 1], True, False, 1]
    assert pin.host_mismatches(t, HOST) == {}
    assert set(pin.host_mismatches(t, dict(HOST, gpu="other"))) == {"gpu"}
    # A schema-2 table also compares SM count / FlashInfer commit.
    t2 = pin.parse_table(
        pin.build_table_json(
            fi_meta={}, host=HOST, groups={}, entries={}, provenance={}
        )
    )
    assert set(pin.host_mismatches(t2, dict(HOST, sm_count="148"))) == {"sm_count"}
    session = pin.PinSession(
        [pin.parse_table(legacy, path="legacy.json")], host=dict(HOST, gpu="other")
    )
    assert session.index == {} and session.stats["stale_host"] == 1
    with pytest.raises(ValueError, match="schema"):
        pin.parse_table({"_records": {pin.NAMESPACE: {"schema": 1}}})


MOE_OP = "flashinfer::trtllm_fp8_block_scale_moe"


class _MoERunner:
    pass


def test_moe_fingerprint_keys_on_prebuilt_content_not_path(tmp_path, monkeypatch):
    a, b, c = tmp_path / "a.so", tmp_path / "copy" / "a.so", tmp_path / "c.so"
    b.parent.mkdir()
    a.write_bytes(b"kernels v1")
    b.write_bytes(b"kernels v1")  # the same .so at a recipe-specific path
    c.write_bytes(b"kernels v2")

    def parts(path):
        monkeypatch.setenv("GS2_ROUTE_PREBUILT", str(path))
        return pin.op_fingerprint_parts(MOE_OP, _MoERunner())

    pa, pb, pc = parts(a), parts(b), parts(c)
    assert "GS2_ROUTE_PREBUILT" not in pa["env"]
    assert pin.stable_hash(pa) == pin.stable_hash(pb) != pin.stable_hash(pc)

    # A record written while the path was still in "env" merges into the
    # group the loader computes now, so its entries are not stale.
    old = {**pa, "env": {**pa["env"], "GS2_ROUTE_PREBUILT": str(a)}}
    old_fp = pin.stable_hash(old)
    old_g = pin.make_gid(MOE_OP, "_MoERunner", old_fp)
    groups = {old_g: {"op": MOE_OP, "runner": "_MoERunner", "fp": old_fp, "parts": old}}
    fk = f"('{MOE_OP}', '_MoERunner', ((16, 256),), ())"
    rec = _record({fk: _rec_key([[2.0], [1.0]], g=old_g)}, groups=groups)
    table, _ = pin.merge_records([("s", [rec])])
    new_g = pin.make_gid(MOE_OP, "_MoERunner", pin.stable_hash(pb))
    assert table[fk][1] == 1 and table[fk][2]["g"] == new_g
    assert pin.parse_table(table).groups[new_g]["parts"] == pb


def test_write_json_atomic_failure_keeps_old_file(tmp_path, monkeypatch):
    path = tmp_path / "t.json"
    pin.write_json_atomic(path, {"v": 1})

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pin.os, "replace", boom)
    with pytest.raises(OSError):
        pin.write_json_atomic(path, {"v": 2})
    assert json.loads(path.read_text()) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["t.json"]


def _hammer(path: str, n: int, tag: int) -> None:
    for i in range(n):
        pin.write_json_atomic(path, {"tag": tag, "i": i, "pad": "x" * 20000})


def test_write_json_atomic_concurrent_writers_never_tear(tmp_path):
    path = tmp_path / "t.json"
    pin.write_json_atomic(path, {"tag": -1, "i": 0, "pad": ""})
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_hammer, args=(str(path), 150, t)) for t in range(4)]
    for p in procs:
        p.start()
    reads = 0
    while any(p.is_alive() for p in procs):
        json.loads(path.read_text())  # never a partial file
        reads += 1
    for p in procs:
        p.join()
        assert p.exitcode == 0
    assert reads > 0
    assert not [p for p in tmp_path.iterdir() if p.name != "t.json"]


def test_interprocess_lock_excludes(tmp_path):
    lock = tmp_path / "x.lock"
    with pin.interprocess_lock(lock) as (_, locked):
        assert locked
        import fcntl

        fd = os.open(lock, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


# --------------------------------------------------------------------------
# FlashInfer AutoTuner integration (GPU)
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch")
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

OP = "mxfp8_gemm"
BUCKETS = (1, 2, 4, 8, 16)


def _fi():
    return pytest.importorskip("flashinfer.autotuner")


def _make_runner_cls():
    fa = _fi()

    class StandInRunner(fa.TunableRunner):  # type: ignore[name-defined]
        def __init__(self, tactics, broken=()):
            self._tactics = tuple(tactics)
            self._broken = tuple(broken)
            self._ran = set()  # (M, tactic) executed

        def get_valid_tactics(self, inputs, profile):
            return list(self._tactics)

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            a, b, _, _, _, out, _ = inputs
            if tactic in self._broken:
                raise RuntimeError("kernel unavailable on this build")
            self._ran.add((a.shape[0], tactic))
            for _ in range(1 if tactic == -1 else int(tactic) % 3 + 1):
                torch.matmul(a, b, out=out)
            return out

    return StandInRunner


_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        fa = _fi()
        _CFG = fa.TuningConfig(
            dynamic_tensor_specs=(
                fa.DynamicTensorSpec(
                    (0,), (0,), BUCKETS, fa.make_bucket_mapper(BUCKETS, True)
                ),
            ),
            constraint_specs=(
                fa.ConstraintSpec(2, 0, lambda s: s[0][0]),
                fa.ConstraintSpec(5, 0, lambda s: s[0][0]),
            ),
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )
    return _CFG


def _call(runner, m, n=256, k=512):
    fa = _fi()
    dev = "cuda"
    inputs = [
        torch.randn(m, k, device=dev, dtype=torch.bfloat16),
        torch.randn(n, k, device=dev, dtype=torch.bfloat16).t(),
        torch.ones(m, 4, device=dev),
        torch.ones(n, 4, device=dev),
        torch.bfloat16,
        torch.empty(m, n, device=dev, dtype=torch.bfloat16),
        torch.empty(64, device=dev, dtype=torch.uint8),
    ]
    r, tactic = fa.AutoTuner.get().choose_one(OP, [runner], _cfg(), inputs)
    r(inputs, tactic=tactic)
    return tactic


def _tune(runner):
    fa = _fi()
    with torch.inference_mode(), fa.autotune(True):
        _call(runner, BUCKETS[-1])


class _World:
    rank_in_group = 0
    world_size = 1
    cpu_group = None

    def broadcast_object(self, obj, src=0):
        return obj

    def barrier(self):
        pass


@pytest.fixture
def tuner():
    fa = _fi()
    t = fa.AutoTuner.get()
    t.clear_cache()
    yield t
    # run_pinned_autotune restores FlashInfer's methods itself.
    assert "search_cache" not in vars(t) and "choose_one" not in vars(t)
    t.clear_cache()


def _generate(tmp_path, runner, name="rec") -> dict:
    """Record mode on the stand-in runner -> one worker record."""
    out = tmp_path / name
    fa = _fi()
    fa.AutoTuner.get().clear_cache()
    path = pin.run_record_autotune(
        None,
        tuner=fa.AutoTuner.get(),
        out_dir=out,
        rounds=3,
        run_passes=lambda: _tune(runner),
    )
    fa.AutoTuner.get().clear_cache()
    return json.loads(path.read_text())


def _pinned(tmp_path, sets, primary=None, name="pin.json") -> Path:
    table, _ = pin.merge_records(sets, primary=primary)
    path = tmp_path / name
    pin.write_json_atomic(path, table)
    return path


def _run(tmp_path, runner, pinned, strict=False, cache="cache"):
    fa = _fi()
    cache_path = tmp_path / cache / "autotune_configs.json"
    return pin.run_pinned_autotune(
        None,
        world=_World(),
        tuner=fa.AutoTuner.get(),
        cache_path=cache_path,
        pinned_path=pinned,
        strict=strict,
        run_passes=lambda: _tune(runner),
    ), cache_path


@gpu
def test_record_then_pinned_roundtrip(tmp_path, tuner, monkeypatch):
    monkeypatch.delenv("PFW", raising=False)
    runner = _make_runner_cls()((0, 1, 2, 3))
    rec = _generate(tmp_path, runner)
    assert len(rec["keys"]) == len(BUCKETS)
    for k in rec["keys"].values():
        assert k["op"] == OP and k["tactics"] == [0, 1, 2, 3]
        assert all(len(s) == 3 for s in k["samples"])
        # Small M: the weight rotates, activation / output stay resident.
        assert k["l2"] == "weights-cold"
    pinned = _pinned(tmp_path, [("a", [rec])])
    summary, cache_path = _run(tmp_path, runner, pinned)
    assert summary["hits"] == len(BUCKETS)
    assert summary["misses"] == summary["stale"] == 0
    assert not cache_path.exists()  # nothing tuned: nothing written
    assert not tuner.profiling_cache
    # Run time (no tuning context): the pinned tactics are served.
    table = pin.read_table(pinned)
    for m in BUCKETS:
        fk = next(k for k in table.entries if f"(({m}, 512)" in k)
        assert _call(runner, m) == table.entries[fk][0].tactic


@gpu
def test_changed_tactic_list_is_stale_and_retuned(tmp_path, tuner):
    cls = _make_runner_cls()
    pinned = _pinned(tmp_path, [("a", [_generate(tmp_path, cls((0, 1, 2, 3)))])])
    summary, cache_path = _run(tmp_path, cls((0, 1, 2, 3, 4)), pinned)
    assert summary["hits"] == 0 and summary["stale"] == len(BUCKETS)
    assert summary["stale_reasons"] == {f"{OP}: tactic list": len(BUCKETS)}
    assert summary["misses"] == len(BUCKETS)
    # The one writer stores the re-tuned entries with this process's identity.
    t = pin.read_table(cache_path)
    for cands in t.entries.values():
        assert cands[0].meta["src"] == "tuned" and cands[0].meta["nt"] == 5


@gpu
def test_fingerprint_change_is_stale(tmp_path, tuner, monkeypatch):
    monkeypatch.delenv("PFW", raising=False)
    runner = _make_runner_cls()((0, 1, 2))
    pinned = _pinned(tmp_path, [("a", [_generate(tmp_path, runner)])])
    monkeypatch.setenv("PFW", "1")  # e.g. the lcd_pdl prefetch knob
    summary, _ = _run(tmp_path, runner, pinned)
    assert summary["hits"] == 0 and summary["stale"] == len(BUCKETS)
    assert summary["stale_reasons"] == {f"{OP}: fingerprint": len(BUCKETS)}


@gpu
def test_variant_chosen_by_fingerprint(tmp_path, tuner, monkeypatch):
    runner = _make_runner_cls()((0, 1, 2))
    monkeypatch.delenv("PFW", raising=False)
    rec_a = _generate(tmp_path, runner, "a")
    monkeypatch.setenv("PFW", "1")
    rec_b = _generate(tmp_path, runner, "b")
    pinned = _pinned(tmp_path, [("a", [rec_a]), ("b", [rec_b])], primary="a")
    summary, _ = _run(tmp_path, runner, pinned)
    assert summary["hits"] == len(BUCKETS) and summary["stale"] == 0
    # Every key was served by the PFW=1 variant (an alt entry), none by the
    # top-level PFW-unset one.
    table = pin.read_table(pinned)
    want = {
        fk: c.tactic
        for fk, cands in table.entries.items()
        for c in cands
        if c.gid in rec_b["groups"]
    }
    assert len(want) == len(BUCKETS)
    accepted = pin.LAST_SESSION.accepted
    assert {fk: v[1] for fk, v in accepted.items()} == want
    assert {v[2]["g"] for v in accepted.values()} == set(rec_b["groups"])
    assert pin.LAST_SESSION.stats["variant_skipped"] == len(BUCKETS)


@gpu
def test_accepted_tactics_run_before_capture_and_broken_ones_are_rejected(
    tmp_path, tuner
):
    cls = _make_runner_cls()
    rec = _generate(tmp_path, cls((0, 1, 2)))
    t = json.loads(_pinned(tmp_path, [("a", [rec])]).read_text())
    keys = sorted(k for k in t if not k.startswith("_"))
    for k in keys:  # pin every key to tactic 1, one key to the broken tactic 2
        t[k][1] = 1
    t[keys[0]][1] = 2
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(t))
    runner = cls((0, 1, 2), broken=(2,))
    summary, _ = _run(tmp_path, runner, path)
    assert summary["hits"] == len(BUCKETS) - 1
    assert summary["stale_reasons"] == {f"{OP}: warm-up failed": 1}
    # Every accepted tactic ran at its bucket inside the warmup (JIT compile
    # happens here, not during CUDA-graph capture or serving).
    assert {(m, 1) for m in BUCKETS if f"(({m}, 512)" not in keys[0]} <= runner._ran


@gpu
def test_legacy_file_tactic_membership(tmp_path, tuner, monkeypatch):
    runner = _make_runner_cls()((0, 1, 2))
    pinned = _pinned(tmp_path, [("a", [_generate(tmp_path, runner)])])
    t = json.loads(pinned.read_text())
    legacy = {"_metadata": t["_metadata"]}
    keys = sorted(k for k in t if not k.startswith("_"))
    legacy[keys[0]] = ["StandInRunner", 1]
    legacy[keys[1]] = ["StandInRunner", 7]  # not offered by the runner
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    summary, _ = _run(tmp_path, runner, path)
    assert summary["hit_detail"] == {"pinned_legacy": 1}
    assert summary["stale_reasons"] == {f"{OP}: tactic not offered": 1}
    assert summary["misses"] == len(BUCKETS) - 1


@gpu
def test_strict_mode_fails_without_tuning(tmp_path, tuner):
    runner = _make_runner_cls()((0, 1, 2))
    rec = _generate(tmp_path, runner)
    keep = sorted(rec["keys"])[:2]
    rec["keys"] = {k: rec["keys"][k] for k in keep}
    pinned = _pinned(tmp_path, [("a", [rec])])
    with pytest.raises(RuntimeError, match="STRICT=1: 3 FlashInfer autotune keys"):
        _run(tmp_path, runner, pinned, strict=True)
    assert not tuner.profiling_cache  # misses ran the fallback, no profiling


def _dp_worker(tmp: str, pinned: str, cuda: int, q) -> None:
    torch.accelerator.set_device_index(cuda)
    import flashinfer.autotuner as fa

    fa.AutoTuner.get().clear_cache()
    runner = _make_runner_cls()((0, 1, 2, 3))
    summary, cache_path = _run(Path(tmp), runner, Path(pinned), cache="shared")
    q.put(summary)


@gpu
def test_dp_workers_tune_misses_once_and_share_one_table(tmp_path, tuner):
    runner = _make_runner_cls()((0, 1, 2, 3))
    rec = _generate(tmp_path, runner)
    rec["keys"] = {k: rec["keys"][k] for k in sorted(rec["keys"])[:3]}
    pinned = _pinned(tmp_path, [("a", [rec])])
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    n = torch.accelerator.device_count()
    procs = [
        ctx.Process(target=_dp_worker, args=(str(tmp_path), str(pinned), i % n, q))
        for i in range(4)
    ]
    for p in procs:
        p.start()
    results = [q.get(timeout=600) for _ in procs]
    for p in procs:
        p.join()
        assert p.exitcode == 0
    # One worker tuned the 2 uncovered keys; the others loaded its entries.
    assert sorted(r["misses"] for r in results) == [0, 0, 0, 2]
    for r in results:
        assert r["hits"] + r["misses"] == len(BUCKETS)
        if r["misses"] == 0:
            assert r["hit_detail"] == {"pinned": 3, "cache": 2}
    assert len({r["digest"] for r in results}) == 1
