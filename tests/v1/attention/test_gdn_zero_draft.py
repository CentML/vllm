# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""R4: a GDN decode row without drafts after an in-block MTP verify with num_accepted > 1.

1. Classifier (CPU): with VLLM_GDN_ZERO_DRAFT_AS_SPEC the row is a 1-token spec row, also in an all-zero-draft batch
   and with a stale draft count; the stock rule (num_decode_draft_tokens >= 0, all-zero collapse) drops it.
2. Kernel semantics (GPU, the spec kernel gdn_mtp_cuda): a spec step of 4 tokens with acc = 3, then one more token
   - fixed routing (spec row, T = 1, reads column acc - 1) == the reference (tokens 0, 1, 2, then the new token as
     single-token steps on one state slot), within float-order tolerance;
   - stock routing (non-spec: reads column 0 = the state after token 0) is far from the reference (2 tokens lost).
"""
import pytest
import torch

from vllm.v1.attention.backends.gdn_zero_draft import spec_row_mask


def _stock_mask(ndt):
    m = ndt >= 0
    if m.sum() == 0 or ndt[m].sum() == 0:
        return None
    return m


def test_classifier():
    # rows: 0 spec 3 drafts | 1 decode no drafts (ndt -1) | 2 decode 0 drafts | 3 prefill chunk | 4 first token
    # (prompt of 1) | 5 padding
    qlens = torch.tensor([4, 1, 1, 37, 1, 0])
    seqs = torch.tensor([90, 51, 60, 400, 1, 0])
    ndt = torch.tensor([3, -1, 0, -1, -1, -1])
    m = spec_row_mask(qlens, seqs, ndt)
    assert m.tolist() == [True, True, True, False, False, False]
    assert _stock_mask(ndt).tolist() == [True, False, True, False, False, False]  # row 1 lost by stock
    # all-zero-draft batch: stock collapses everything to non-spec, the fix keeps the decode rows on the spec path
    ndt0 = torch.tensor([0, 0, -1])
    q0, s0 = torch.tensor([1, 1, 1]), torch.tensor([20, 30, 40])
    assert _stock_mask(ndt0) is None
    assert spec_row_mask(q0, s0, ndt0).tolist() == [True, True, True]
    # stale draft count (query length disagrees): re-derived from the query length
    assert spec_row_mask(torch.tensor([1, 6]), torch.tensor([9, 50]), torch.tensor([3, 3])).tolist() == [True, False]
    # unknown context: draft rows only
    assert spec_row_mask(qlens, None, ndt).tolist() == [True, False, True, False, False, False]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
                    reason="needs an sm_100+ GPU")
def test_kernel_semantics_acc3_then_no_draft():
    M = pytest.importorskip("vllm.model_executor.layers.mamba.ops.gdn_mtp_cuda")
    ext = M.build(M.tuned_source("", False))
    dev = "cuda"
    H, HV, K, V = 16, 32, 128, 128
    g = torch.Generator(device=dev).manual_seed(5)
    PB = 16
    state = torch.zeros(PB, HV, V, K, device=dev, dtype=torch.bfloat16)
    s0 = (torch.randn(HV, V, K, device=dev, generator=g) * 0.05).to(torch.bfloat16)
    A_log = torch.log(torch.empty(HV, device=dev).uniform_(1, 16, generator=g))
    dt_bias = (torch.randn(HV, device=dev, generator=g) * 0.5 + 1).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(V, device=dev, generator=g)).to(torch.bfloat16)
    toks = 5  # tokens 0..3 (spec step), token 4 (the no-draft step)
    qkv = torch.randn(toks, 2 * H * K + HV * V, device=dev, dtype=torch.bfloat16, generator=g)
    a = torch.randn(toks, HV, device=dev, dtype=torch.bfloat16, generator=g)
    b = torch.randn(toks, HV, device=dev, dtype=torch.bfloat16, generator=g)
    gate = torch.randn(toks, HV, V, device=dev, dtype=torch.bfloat16, generator=g)
    i32 = dict(dtype=torch.int32, device=dev)

    def call(rows, si, acc):
        n = rows.stop - rows.start
        out = torch.empty(n, HV, V, device=dev, dtype=torch.bfloat16)
        ok = ext.run(qkv[rows], a[rows], b[rows], A_log, dt_bias, torch.tensor([si], **i32),
                     torch.tensor([0, n], **i32), torch.tensor([acc], **i32), state, gate[rows], w, out,
                     K ** -0.5, 1e-6, False)
        assert ok
        return out

    # reference: one state slot (1), tokens 0, 1, 2, 4 as single-token steps (token 3 was rejected)
    state[1] = s0
    for t in (0, 1, 2):
        call(slice(t, t + 1), [1], 1)
    ref_state = state[1].clone()
    ref_out = call(slice(4, 5), [1], 1).float()
    ref_state_after = state[1].float().clone()
    # spec step: columns 2..5, 4 tokens, initial state at column 0 (acc 1); then verify accepts 3 tokens
    state[2] = s0
    call(slice(0, 4), [2, 3, 4, 5], 1)
    torch.cuda.synchronize()
    rel = lambda x, y: float((x.float() - y.float()).norm() / y.float().norm())  # noqa: E731
    assert rel(state[4], ref_state) < 2e-2  # column acc - 1 = 2 holds the state after token 2
    # fixed routing: spec row, T = 1, acc = 3 -> reads column 2, writes column 0
    saved = state.clone()
    fix_out = call(slice(4, 5), [2, 3, 4, 5], 3).float()
    fix_state = state[2].float().clone()
    # stock routing: non-spec decode reads column 0 (= the state after token 0)
    state.copy_(saved)
    stock_out = call(slice(4, 5), [2, 3, 4, 5], 1).float()
    stock_state = state[2].float().clone()
    e_fix, e_stock = rel(fix_state, ref_state_after), rel(stock_state, ref_state_after)
    o_fix, o_stock = rel(fix_out, ref_out), rel(stock_out, ref_out)
    print(f"state rel-L2 fix {e_fix:.3e} stock {e_stock:.3e}; out rel-L2 fix {o_fix:.3e} stock {o_stock:.3e}")
    assert e_fix < 2e-2 and o_fix < 2e-2, (e_fix, o_fix)
    assert e_stock > 10 * max(e_fix, 1e-3), (e_stock, e_fix)


def test_r4_counter(monkeypatch):
    """VLLM_GDN_R4_COUNT: stock-exposed decode rows and their acc > 1 subset, once per step (dedup across groups)."""
    import types

    import vllm.v1.attention.backends.gdn_zero_draft as Z

    monkeypatch.setattr(Z, "_C", {"steps": 0, "decode_rows": 0, "exposed_rows": 0, "dev": None, "last": None})
    monkeypatch.setattr(Z, "R4_COUNT_EVERY", 0)
    # rows: spec(3 drafts) | decode no drafts acc 3 | decode no drafts acc 1 | prefill | padding
    qsl = torch.tensor([0, 4, 5, 6, 43, 43])
    m = types.SimpleNamespace(query_start_loc_cpu=qsl, num_actual_tokens=43,
                              seq_lens_cpu_upper_bound=torch.tensor([90, 51, 60, 400, 0]))
    ndt = torch.tensor([3, -1, -1, -1, -1])
    acc = torch.tensor([2, 3, 1, 1, 1])
    Z.count(m, ndt, acc)
    Z.count(m, ndt, acc)  # second KV-cache group of the same step: ignored
    assert Z._C["steps"] == 1 and Z._C["decode_rows"] == 2 and Z._C["exposed_rows"] == 2
    assert int(Z._C["dev"].item()) == 1
    # all-zero-draft step: every decode row is stock-non-spec
    m2 = types.SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 1, 2]), num_actual_tokens=2,
                               seq_lens_cpu_upper_bound=torch.tensor([30, 40]))
    Z.count(m2, torch.tensor([0, 0]), torch.tensor([4, 2]))
    assert Z._C["steps"] == 2 and Z._C["decode_rows"] == 4 and Z._C["exposed_rows"] == 4
    assert int(Z._C["dev"].item()) == 3
