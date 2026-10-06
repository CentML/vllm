"""[gdnh] F108 in-tree port: unit test of vllm.v1.attention.backends.gdn_fused_metadata vs the regular
GDNAttentionMetadataBuilder._build_full (same tree, same env: GDN_STATE_COMMIT / GGM / GGM_LAZY as set by the caller),
on randomized spec-decode batches. Checks every metadata field: values, dtype, shape, stride, aliasing / shared storage,
the deferred-call kinds, and for SPEC_ONLY the full contents of the builder's persistent FULL-cudagraph buffers.
Also times regular vs fused per group (GPU events + host) and counts GPU ops (torch profiler).
Run inside the server image with the gdnh vllm tree first on PYTHONPATH."""
import json
import os
import random
import time
import types

os.environ.setdefault("VLLM_GDN_FUSED_MD", "1")
import torch  # noqa: E402

import vllm.v1.attention.backends.gdn_attn as ga  # noqa: E402
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402
from vllm.v1.attention.backends import gdn_fused_metadata as fm  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import gdn_step_plan  # noqa: E402

dev = torch.device("cuda")
B = ga.GDNAttentionMetadataBuilder
print(json.dumps({"gsc_deferred": ga._GDN_STATE_COMMIT_DEFERRED, "lazy": gdn_step_plan.LAZY,
                  "fused_enabled": fm.ENABLED, "file": fm.__file__}), flush=True)


def mk_builder(mode, nsb, num_spec=3, max_bs=1024, full=True, bs_tokens=2144):
    b = object.__new__(B)
    spec = ga.MambaSpec(block_size=bs_tokens, shapes=((4, 4),), dtypes=(torch.bfloat16,), mamba_cache_mode=mode,
                        num_speculative_blocks=nsb)
    b.kv_cache_spec = spec
    b.vllm_config = types.SimpleNamespace(cache_config=types.SimpleNamespace(mamba_cache_mode=mode))
    b.num_spec = num_spec
    b.use_spec_decode = True
    b.use_full_cuda_graph = full
    b.decode_cudagraph_max_bs = max_bs
    b.gdn_prefill_backend = "flashinfer"
    b.spec_state_indices_tensor = torch.full((max_bs, num_spec + 1), -7, dtype=torch.int32, device=dev)
    b.non_spec_state_indices_tensor = torch.full((max_bs,), -7, dtype=torch.int32, device=dev)
    b.spec_sequence_masks = torch.zeros((max_bs,), dtype=torch.bool, device=dev)
    b.spec_token_indx = torch.full((max_bs * (num_spec + 1),), -7, dtype=torch.int32, device=dev)
    b.non_spec_token_indx = torch.full((max_bs * (num_spec + 1),), -7, dtype=torch.int32, device=dev)
    b.spec_query_start_loc = torch.full((max_bs + 1,), -7, dtype=torch.int32, device=dev)
    b.non_spec_query_start_loc = torch.full((max_bs + 1,), -7, dtype=torch.int32, device=dev)
    b.num_accepted_tokens = torch.full((max_bs,), -7, dtype=torch.int32, device=dev)
    return b


def mk_batch(rng, kind, R_spec, n_pre, n_dec1, n_pad, num_spec=3, bs_tokens=2144, cols=128, maxpre=3000):
    rows = []
    for _ in range(R_spec):
        d = rng.randint(1, num_spec)
        rows.append(("s", d + 1, d))
    for _ in range(n_pre):
        rows.append(("p", rng.randint(2, maxpre), -1))
    for _ in range(n_dec1):
        rows.append(("p", 1, -1))
    if kind == "mixed_shuffled_nonspec":
        tail = rows[R_spec:]
        rng.shuffle(tail)
        rows = rows[:R_spec] + tail
    if kind == "spec_not_prefix" and len(rows) > 1:
        rows = rows[1:] + rows[:1]
    for _ in range(n_pad):
        rows.append(("z", 0, -1))
    R = len(rows)
    qlen = [r[1] for r in rows]
    qsl = [0]
    for q in qlen:
        qsl.append(qsl[-1] + q)
    ndd = [r[2] for r in rows]
    seq = []
    for r in rows:
        if r[0] == "z":
            seq.append(0)
        else:
            ctx = 0 if (r[0] == "p" and rng.random() < 0.3) else rng.randint(
                1, (max(cols, 128) - 8) * bs_tokens - r[1] - 1)
            seq.append(ctx + r[1])
    bt = torch.randint(1, 50000, (R + rng.randint(0, 3), cols), dtype=torch.int32)
    nacc = torch.tensor([rng.randint(1, num_spec + 1) for _ in range(R)], dtype=torch.int32)
    qsl_cpu = torch.tensor(qsl, dtype=torch.int32)
    m = CommonAttentionMetadata(
        query_start_loc=qsl_cpu.to(dev), query_start_loc_cpu=qsl_cpu,
        seq_lens=torch.tensor(seq, dtype=torch.int32, device=dev),
        num_reqs=R, num_actual_tokens=qsl[-1], max_query_len=max(qlen) if qlen else 0, max_seq_len=max(seq),
        block_table_tensor=bt.to(dev), slot_mapping=torch.empty(0, dtype=torch.int64, device=dev))
    return m, nacc.to(dev), torch.tensor(ndd, dtype=torch.int32)


def regular(b, m, nacc, ndd):
    return b._build_full(0, m, nacc, ndd, False)


def check_case(b, m, nacc, ndd):
    bufs = {k: getattr(b, k) for k in fm._BUF_KEYS}
    snap = {k: v.clone() for k, v in bufs.items()}
    before = dict(fm.STATS)
    mine = fm._fast(b, m, nacc, ndd)
    if mine is None:
        return "fallback", []
    mode = "mixed" if fm.STATS["mixed"] != before["mixed"] else "spec"
    after = {k: v.clone() for k, v in bufs.items()}
    views = {}
    if mode == "spec":
        for k, v in bufs.items():
            v.copy_(snap[k])
        for f in bufs:
            t = mine.__dict__.get(f)
            if isinstance(t, torch.Tensor):
                off = t.storage_offset() - bufs[f].storage_offset()
                views[f] = after[f].as_strided(t.shape, t.stride(), after[f].storage_offset() + off)
    ref = regular(b, m, nacc, ndd)
    torch.cuda.synchronize()
    bad = fm._compare(ref, mine, views)
    full = [k for k in bufs if not torch.equal(bufs[k], after[k])]
    if full:
        bad.append(f"persistent:{full}")
    # deferred calls (GGM_LAZY): run both and compare their results too
    la, lb = ref.__dict__.get("_step_plan_lazy"), mine.__dict__.get("_step_plan_lazy")
    if la and lb:
        for (fa, ka, aa, kwa), (fb, kb, ab, kwb) in zip(la, lb):
            ra, rb = fa(*aa, **kwa), fb(*ab, **kwb)
            ra = ra if isinstance(ra, tuple) else (ra,)
            rb = rb if isinstance(rb, tuple) else (rb,)
            for x, y in zip(ra, rb):
                if isinstance(x, torch.Tensor) and not torch.equal(x, y):
                    bad.append(f"lazy:{ka}")
    return mode, bad


def timeit(fn, n=200):
    torch.cuda.synchronize()
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    h0 = time.perf_counter()
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    h1 = time.perf_counter()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / n, (h1 - h0) * 1e6 / n


def main():
    rng = random.Random(1234)
    res = {"cases": 0, "by_mode": {}, "fallback": 0, "fail": []}
    configs = [("align", 0), ("align", 3), ("none", 0), ("all", 0)]
    kinds = []
    for _ in range(60):
        kinds.append(("mixed", rng.randint(1, 200), rng.randint(1, 6), rng.randint(0, 3), 0))
    for _ in range(20):
        kinds.append(("mixed", rng.randint(1, 8), rng.randint(1, 2), rng.randint(0, 40), 0))
    for _ in range(20):
        kinds.append(("mixed_shuffled_nonspec", rng.randint(1, 100), rng.randint(1, 4), rng.randint(1, 5), 0))
    kinds += [("mixed", 1, 1, 0, 0), ("mixed", 255, 1, 0, 0), ("mixed", 1, 16, 0, 0), ("mixed", 64, 3, 0, 0),
              ("mixed", 3, 1, 0, 0), ("mixed", 96, 2, 0, 0), ("spec_not_prefix", 5, 2, 0, 0)]
    for _ in range(40):
        kinds.append(("spec", rng.randint(1, 256), 0, 0, rng.randint(0, 20)))
    kinds += [("spec", 1, 0, 0, 0), ("spec", 256, 0, 0, 0), ("spec", 16, 0, 0, 16), ("spec", 4, 0, 0, 0)]
    for mode, nsb in configs:
        for kind, rs, npre, nd1, npad in kinds:
            b = mk_builder(mode, nsb)
            cols = 128 if mode == "align" else (1 + nsb if mode == "none" else 64)
            m, nacc, ndd = mk_batch(rng, kind, rs, npre, nd1, npad, cols=cols)
            if mode in ("none", "all"):
                m.block_table_tensor = m.block_table_tensor[: m.num_reqs].contiguous()
            try:
                got, bad = check_case(b, m, nacc, ndd)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                got, bad = "error", [repr(e)]
            res["cases"] += 1
            if got == "fallback":
                res["fallback"] += 1
                continue
            res["by_mode"][got] = res["by_mode"].get(got, 0) + 1
            if bad:
                res["fail"].append({"cfg": [mode, nsb], "kind": [kind, rs, npre, nd1, npad], "bad": bad[:6]})
    res["fallback_reasons"] = fm.STATS["fallback"]
    tim = {}
    for name, (kind, rs, npre, nd1, npad) in {"mixed_4s_1p": ("mixed", 4, 1, 0, 0),
                                              "mixed_80s_2p": ("mixed", 80, 2, 0, 0),
                                              "mixed_160s_2p": ("mixed", 160, 2, 0, 0),
                                              "spec_4s": ("spec", 4, 0, 0, 0),
                                              "spec_96s": ("spec", 96, 0, 0, 0),
                                              "spec_180s_12pad": ("spec", 180, 0, 0, 12)}.items():
        b = mk_builder("align", 0)
        m, nacc, ndd = mk_batch(rng, kind, rs, npre, nd1, npad)
        tf = timeit(lambda: fm._fast(b, m, nacc, ndd))
        ts = timeit(lambda: regular(b, m, nacc, ndd))
        tim[name] = {"fused_gpu_us": round(tf[0], 2), "fused_host_us": round(tf[1], 2),
                     "regular_gpu_us": round(ts[0], 2), "regular_host_us": round(ts[1], 2)}
    res["timing_per_group"] = tim
    from torch.profiler import ProfilerActivity, profile
    cnt = {}
    for name, (kind, rs, npre, nd1, npad) in {"mixed": ("mixed", 80, 2, 0, 0), "spec": ("spec", 96, 0, 0, 0)}.items():
        b = mk_builder("align", 0)
        m, nacc, ndd = mk_batch(rng, kind, rs, npre, nd1, npad)
        for label, fn in (("fused", lambda: fm._fast(b, m, nacc, ndd)), ("regular", lambda: regular(b, m, nacc, ndd))):
            fn()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                fn()
                torch.cuda.synchronize()
            ev = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            cnt[f"{name}_{label}"] = {"gpu_ops": len(ev), "names": sorted({e.name[:50] for e in ev})[:40]}
    res["launches_per_group"] = cnt
    ok = not res["fail"] and res["by_mode"].get("mixed", 0) > 0 and res["by_mode"].get("spec", 0) > 0
    print(json.dumps(res, indent=1, default=str))
    print("FUSED_MD_UT", "ALL_PASS" if ok else f"FAIL {len(res['fail'])}", flush=True)


if __name__ == "__main__":
    main()
