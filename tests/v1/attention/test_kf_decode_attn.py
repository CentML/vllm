# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel Factory paged-FP8 decode attention (vllm/v1/attention/ops/kf_decode_attn).

1. Builder routing: a uniform batch and a CUDA-graph-padded one (real rows of Q
   tokens, then zero-length requests) run on the kernel with their block table
   copied into the MAXP-stride buffer; a ragged batch stays on trtllm-gen.
2. SM107: the kernel through the runtime (warmed forms, shared scratch, a vLLM
   2052-wide table in the 2057-stride buffer) on C32 launch shapes for Q = 6 (MTP
   k=5 verify), 5 and 1, eager and replayed from a CUDA graph captured at other
   lengths: finite, run-to-run bitwise, merge counters reset, and per row no worse
   than trtllm-gen vs an fp32 reference (relL2, and max|d| within 4 bf16 ulp).
"""

import math

import numpy as np
import pytest
import torch

pytest.importorskip("cutlass")
from vllm.platforms import current_platform  # noqa: E402
from vllm.v1.attention.backends import flashinfer as fi_backend  # noqa: E402
from vllm.v1.attention.ops.kf_decode_attn.runtime import (  # noqa: E402
    MAXP,
    KfDecodeAttn,
    parse_qlens,
)

WIDTH = 2052  # vLLM's block-table width at max_model_len 262144, bf16 GDN state


def test_parse_qlens():
    assert parse_qlens("") is None
    assert parse_qlens("1, 6") == {1, 6}
    with pytest.raises(ValueError):
        parse_qlens("1,9")


class _Route:
    """Stand-in runtime: routes every uniform batch of up to 8 requests."""

    def __init__(self):
        self.calls = []

    def route(self, q, bl, uniform):
        self.calls.append((q, bl, uniform))
        return uniform and bl <= 8


def _builder(monkeypatch):
    b = object.__new__(fi_backend.FlashInferMetadataBuilder)
    b.device = torch.device("cpu")
    b._kf_bt = torch.zeros(8, MAXP, dtype=torch.int32)
    b._kf_bt_src = (0, 0)
    b._kf_draft_index = 0
    rt = _Route()
    monkeypatch.setattr(fi_backend, "get_kf_decode_runtime", lambda dev: rt)
    return b, rt


@pytest.mark.parametrize(
    "q_lens, routed",
    [
        ([6, 6, 6], True),  # uniform verify batch
        ([6, 6, 6, 6, 6, 0], True),  # B=5 padded to the BL=6 graph
        ([1] * 7 + [0], True),  # draft step, padded
        ([6, 0, 6, 0], False),  # a zero-length row before a real one
        ([6, 3, 6], False),  # ragged
    ],
)
def test_builder_routing(monkeypatch, q_lens, routed):
    b, rt = _builder(monkeypatch)
    bl = len(q_lens)
    q = max(q_lens)
    qo = torch.tensor(np.concatenate([[0], np.cumsum(q_lens)]), dtype=torch.int32)
    bt = torch.randint(1, 1 << 20, (bl + 3, WIDTH), dtype=torch.int32)
    uniform = all(x == q_lens[0] for x in q_lens)
    res = b._kf_decode_route(bt, qo, bl, q * bl, uniform)
    assert rt.calls[-1][:2] == (q, bl)
    if not routed:
        assert res is None
        return
    assert res is not None
    kf_bt, kf_q = res
    assert kf_q == q and kf_bt.shape == (bl, MAXP) and kf_bt.stride(0) == MAXP
    assert torch.equal(kf_bt[:, :WIDTH], bt[:bl])
    # A later draft step on a different table (e.g. after the target step) copies.
    b._kf_draft_index = 2
    bt2 = bt.flip(0).contiguous()
    kf_bt, _ = b._kf_decode_route(bt2, qo, bl, q * bl, uniform)
    assert torch.equal(kf_bt[:, :WIDTH], bt2[:bl])


def _inputs(lens, q_len, seed):
    """FP8 Q and paged KV in 9-page (1152-token) hybrid blocks; padded rows have
    seq_len 0 and the null block.
    """
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    nblk = sum(-(-((L + 127) // 128) // 9) for L in lens) + 2
    pages = (nblk + 1) * 9
    kv = (torch.randn(pages, 2, 128, 512, device=dev, generator=g) * 0.7).to(
        torch.float8_e4m3fn
    )
    blocks = torch.randperm(nblk, generator=torch.Generator().manual_seed(seed)) + 1
    bt = torch.zeros(len(lens), WIDTH, dtype=torch.int32)
    k = 0
    for b, L in enumerate(lens):
        n = (L + 127) // 128
        for j in range(0, n, 9):
            w = min(9, n - j)
            bt[b, j : j + w] = torch.arange(blocks[k] * 9, blocks[k] * 9 + w)
            k += 1
    q = (torch.randn(len(lens) * q_len, 16, 256, device=dev, generator=g) * 0.7).to(
        torch.float8_e4m3fn
    )
    sl = torch.tensor(lens, dtype=torch.int32, device=dev)
    return q, kv, bt.to(dev), sl


def _trtllm(q, kv, bt, sl, q_len, b1, b2):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=kv.split(256, dim=-1),
        workspace_buffer=torch.zeros(394 << 20, dtype=torch.uint8, device=q.device),
        block_tables=bt,
        seq_lens=sl,
        max_seq_len=262144,
        bmm1_scale=b1,
        bmm2_scale=b2,
        out=out,
        kv_layout="HND",
        backend="trtllm-gen",
        q_len_per_req=q_len,
    )
    return out


def _row_errors(x, ref, lens, q_len):
    """Per real row: (relL2, max|d|, |x| at the worst element)."""
    out = []
    for b, L in enumerate(lens):
        if L == 0:
            continue
        r = ref[b * q_len : (b + 1) * q_len]
        d = (x[b * q_len : (b + 1) * q_len].float() - r).abs()
        i = int(d.argmax())
        xb = x[b * q_len : (b + 1) * q_len].float().flatten()
        out.append((float(d.norm() / r.norm()), float(d.max()), float(xb[i].abs())))
    return out


def _bf16_ulp(x):
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x > 0 else 2.0**-133


@pytest.mark.skipif(
    not current_platform.is_device_capability(107), reason="SM107 kernel"
)
@pytest.mark.parametrize("q_len", [6, 5, 1])
@pytest.mark.parametrize(
    "lens",
    [
        [138252],
        [33150, 132599],
        [5, 129, 16385, 262144 - 8],
        [22743, 39800, 56857, 73914, 90972, 0],
        [67067, 80000, 94500, 105000, 119000, 0, 0, 0],
    ],
)
def test_kernel_through_runtime(q_len, lens):
    lens = [max(L, q_len) if L else 0 for L in lens]
    bl = len(lens)
    rt = KfDecodeAttn(torch.device("cuda"), {q_len}, 8)
    rt.warmup()
    assert set(rt.forms) == {(q_len, b) for b in range(1, 9)}
    q, kv, bt, sl = _inputs(lens, q_len, seed=7)
    kf_bt = torch.zeros(8, MAXP, dtype=torch.int32, device="cuda")
    kf_bt[:bl, :WIDTH] = bt
    outs = []
    for _ in range(2):
        out = torch.full(q.shape, float("nan"), dtype=torch.bfloat16, device="cuda")
        assert rt.launch(q, kv, kf_bt[:bl], sl, out, q_len, 0.0625, 1.0)
        outs.append(out)
    torch.accelerator.synchronize()
    assert bool((rt.cnt == 0).all())
    real = torch.tensor([L > 0 for L in lens], device="cuda").repeat_interleave(q_len)
    assert torch.equal(outs[0][real], outs[1][real])

    # Captured at other (shorter) lengths, replayed on these: the split comes from
    # the device seq_lens.
    graph_out = torch.full_like(outs[0], float("nan"))
    sl_graph = torch.tensor(
        [max(q_len, L // 7) if L else 0 for L in lens], dtype=torch.int32, device="cuda"
    )
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        rt.launch(q, kv, kf_bt[:bl], sl_graph, graph_out, q_len, 0.0625, 1.0)
    sl_graph.copy_(sl)
    g.replay()
    torch.accelerator.synchronize()
    assert torch.equal(graph_out[real], outs[0][real])
    assert bool((rt.cnt == 0).all())

    kf_out = outs[0]
    assert bool(torch.isfinite(kf_out[real].float()).all())
    ref = torch.zeros(q.shape, dtype=torch.float32, device="cuda")
    rows = [b for b, L in enumerate(lens) if L]
    idx = torch.cat([torch.arange(b * q_len, (b + 1) * q_len) for b in rows]).cuda()
    ref[idx] = fi_backend._fp32_paged_prefill_reference(
        q[idx],
        kv,
        bt[rows],
        [i * q_len for i in range(len(rows) + 1)],
        [lens[b] for b in rows],
        0.0625,
        1.0,
    )
    tr = _trtllm(q, kv, bt, sl, q_len, 0.0625, 1.0)
    for (kr, km, kat), (fr, fm, _) in zip(
        _row_errors(kf_out, ref, lens, q_len), _row_errors(tr, ref, lens, q_len)
    ):
        assert kr <= fr, (kr, fr)
        assert km <= fm + 4 * _bf16_ulp(max(kat, fm)), (km, fm, kat)
