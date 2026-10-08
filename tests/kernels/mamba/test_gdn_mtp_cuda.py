# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register-resident CUDA GDN spec-decode recurrence + gated norm (JIT) vs a
float64 reference.
"""

import functools

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA required", allow_module_level=True)

from vllm.model_executor.layers.mamba.ops import gdn_mtp_cuda  # noqa: E402

K = V = 128
EPS = 1e-6


@pytest.fixture(scope="module", autouse=True)
def _built():
    if not gdn_mtp_cuda.enable():
        pytest.skip("GDN MTP CUDA kernel did not build")


def _reference(qkv, a, b, A_log, dt_bias, si, cu, acc, state, gate, w, scale, act, H):
    """Per request: state of the last accepted token -> gated delta rule over
    its tokens, state after token t -> slot si[r, t] (if > 0); output = gated
    RMSNorm of the bf16-rounded recurrence output. Invalid source -> zeros.
    """
    HV = state.shape[1]
    state = state.double().clone()
    out = torch.zeros(qkv.shape[0], HV, V, dtype=torch.float64, device=qkv.device)
    for r in range(si.shape[0]):
        bos, eos = int(cu[r]), int(cu[r + 1])
        n_acc = int(acc[r])
        src = int(si[r, n_acc - 1]) if 0 < n_acc <= si.shape[1] else 0
        if eos <= bos or src <= 0:
            continue
        h = state[src].clone()  # [HV, V, K]
        for t in range(eos - bos):
            row = qkv[bos + t].double()
            q = row[: H * K].view(H, K).repeat_interleave(HV // H, 0)
            k = row[H * K : 2 * H * K].view(H, K).repeat_interleave(HV // H, 0)
            v = row[2 * H * K :].view(HV, V)
            q = F.normalize(q, dim=-1, eps=1e-3) * scale
            k = F.normalize(k, dim=-1, eps=1e-3)
            g = -torch.exp(A_log.double()) * F.softplus(
                a[bos + t].double() + dt_bias.double()
            )
            beta = torch.sigmoid(b[bos + t].double())
            h = h * torch.exp(g)[:, None, None]
            delta = (v - torch.einsum("hvk,hk->hv", h, k)) * beta[:, None]
            h = h + delta[:, :, None] * k[:, None, :]
            o = torch.einsum("hvk,hk->hv", h, q).to(torch.bfloat16).double()
            rstd = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
            z = gate[bos + t].double()
            z = torch.sigmoid(z) if act == "sigmoid" else F.silu(z)
            out[bos + t] = o * rstd * w.double() * z
            dst = int(si[r, t])
            if dst > 0:
                state[dst] = h
    return out, state


def _inputs(state_dtype, H, HV, width):
    device = torch.device("cuda")
    n = 9
    # Varlen requests (1..width tokens); request 2 has an invalid source
    # (num_accepted 0), request 0 skips the state of token 1 (slot 0).
    lens = [width, 1, 3, 2, width, width - 1, 1, 2, 3]
    acc = torch.tensor([width, 1, 0, 2, 1, width - 1, 1, 2, 3], dtype=torch.int32)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    slots = 1 + n * width + 3
    si = (torch.randperm(slots - 1) + 1)[: n * width].view(n, width).to(torch.int32)
    si[0, 1] = 0
    L = int(cu[-1])
    qkv_w = 2 * H * K + HV * V
    # Row stride wider than q/k/v (the packed in_proj output).
    qkvz = torch.randn(L, qkv_w + 64, device=device).to(torch.bfloat16)
    qkv = qkvz[:, :qkv_w]
    ba = torch.randn(L, 2 * HV, device=device).to(torch.bfloat16)
    b, a = ba[:, :HV], ba[:, HV:]
    A_log = torch.log(torch.empty(HV, device=device).uniform_(1, 16))
    dt_bias = torch.randn(HV, device=device)
    state = (0.05 * torch.randn(slots, HV, V, K, device=device)).to(state_dtype)
    gate = torch.randn(L, HV, V, device=device).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(V, device=device)).to(torch.bfloat16)
    si, cu, acc = si.to(device), cu.to(device), acc.to(device)
    return n, lens, slots, L, (qkv, a, b, A_log, dt_bias, si, cu, acc, state, gate, w)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("H,HV", [(2, 4), (1, 4), (4, 4)])
@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("act", ["silu", "sigmoid"])
def test_gdn_mtp_cuda_matches_reference(state_dtype, H, HV, width, act):
    torch.manual_seed(width + HV // H)
    n, lens, slots, L, args = _inputs(state_dtype, H, HV, width)
    qkv, a, b, A_log, dt_bias, si, cu, acc, state, gate, w = args
    out = torch.full((L, HV, V), 7.0, dtype=torch.bfloat16, device=qkv.device)
    scale = K**-0.5

    ref_out, ref_state = _reference(
        qkv, a, b, A_log, dt_bias, si, cu, acc, state, gate, w, scale, act, H
    )
    new_state = state.clone()
    assert gdn_mtp_cuda.gdn_mtp_cuda(
        qkv, a, b, A_log, dt_bias, si, cu, acc, new_state, gate, w, out, scale, EPS, act
    )

    # The output is bf16-rounded twice (before and after the norm).
    torch.testing.assert_close(out.double(), ref_out, atol=3e-2, rtol=3e-2)
    written = torch.zeros(slots, dtype=torch.bool, device=qkv.device)
    for r in range(n):
        n_acc = int(acc[r])
        if 0 < n_acc <= width and int(si[r, n_acc - 1]) > 0:
            for t in range(lens[r]):
                if int(si[r, t]) > 0:
                    written[int(si[r, t])] = True
    tol = 2e-2 if state_dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(
        new_state[written].double(), ref_state[written], atol=tol, rtol=tol
    )
    # Slots that no destination names keep their bytes.
    assert torch.equal(new_state[~written], state[~written])


@pytest.mark.parametrize(
    "tune", ["RS=1+LASTN=1+FADD2=1+PF=212", "RS=1,PF=1", "RPT=4+MINB=1+RS=1"]
)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [4, 6, 8])
def test_gdn_mtp_cuda_tuned_bitwise(tune, state_dtype, width):
    """VLLM_GDN_MTP_CUDA_TUNE builds: same state and output bits as the stock
    kernel (same per-element operations and reduction trees).
    """
    torch.manual_seed(width)
    _, _, _, L, args = _inputs(state_dtype, 2, 4, width)
    state = args[8]
    res = []
    for ext in (
        gdn_mtp_cuda.build(gdn_mtp_cuda.tuned_source("", pdl=False)),
        gdn_mtp_cuda.build(gdn_mtp_cuda.tuned_source(tune, pdl=True)),
    ):
        st = state.clone()
        out = torch.full((L, 4, V), 7.0, dtype=torch.bfloat16, device=state.device)
        run_args = list(args)
        run_args[8] = st
        assert ext.run(*run_args, out, K**-0.5, EPS, False)
        res.append((st, out))
    assert torch.equal(res[0][0], res[1][0])
    assert torch.equal(res[0][1], res[1][1])


def test_gdn_mtp_cuda_rejects_unsupported_layout():
    """Contract violations launch nothing and return False (csrc fallback)."""
    device = torch.device("cuda")
    H, HV, L = 1, 2, 4
    qkv = torch.zeros(L, 2 * H * K + HV * V, device=device, dtype=torch.bfloat16)
    ab = torch.zeros(L, HV, device=device, dtype=torch.bfloat16)
    state = torch.zeros(3, HV, V, 64, device=device)  # head_k_dim 64
    gate = torch.zeros(L, HV, V, device=device, dtype=torch.bfloat16)
    out = torch.full((L, HV, V), 7.0, device=device, dtype=torch.bfloat16)
    si = torch.ones(1, 4, device=device, dtype=torch.int32)
    cu = torch.tensor([0, L], device=device, dtype=torch.int32)
    acc = torch.ones(1, device=device, dtype=torch.int32)
    vec = torch.zeros(HV, device=device)
    w = torch.ones(V, device=device)
    assert not gdn_mtp_cuda.gdn_mtp_cuda(
        qkv, ab, ab, vec, vec, si, cu, acc, state, gate, w, out, 1.0, EPS, "silu"
    )
    assert torch.all(out == 7.0)


@functools.cache
def _ext(tune: str, quant: bool):
    # Tuned builds with PDL, as deployed (VLLM_GDN_MTP_CUDA_PDL=1).
    return gdn_mtp_cuda.build(
        gdn_mtp_cuda.tuned_source(tune, pdl=tune != "", quant=quant)
    )


def _decode_batch(n, width, pad, state_dtype, H=16, HV=32):
    """A FULL-graph-shaped decode-only spec batch (Qwen3.6-35B-A3B heads): n
    requests of ``width`` tokens, then ``pad`` graph-padding requests without
    tokens; the activation has the graph's (n + pad) * width rows. Request 1
    has an invalid source (num_accepted 0), request 2 skips the state of token
    1, the last request runs width - 1 tokens. The norm weight puts three of
    every head's four MXFP8 blocks in scale corners: values 32..63 around the
    UE8M0 subnormal rounding edge (amax / 448 just below 2^-126), 64..95 zero,
    96..127 bf16 subnormals.
    """
    device = torch.device("cuda")
    N = n + pad
    lens = [width] * n + [0] * pad
    if n > 2:
        lens[n - 1] = width - 1
    acc = torch.randint(1, width + 1, (N,), dtype=torch.int32)
    if n > 1:
        acc[1] = 0
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    slots = 1 + N * width
    si = (torch.randperm(slots - 1) + 1)[: N * width].view(N, width).to(torch.int32)
    if n > 2:
        si[2, 1] = 0
    rows = N * width
    qkv_w = 2 * H * K + HV * V
    qkv = torch.randn(rows, qkv_w + 64, device=device).to(torch.bfloat16)[:, :qkv_w]
    ba = torch.randn(rows, 2 * HV, device=device).to(torch.bfloat16)
    b, a = ba[:, :HV], ba[:, HV:]
    A_log = torch.log(torch.empty(HV, device=device).uniform_(1, 16))
    dt_bias = torch.randn(HV, device=device)
    state = (0.05 * torch.randn(slots, HV, V, K, device=device)).to(state_dtype)
    gate = torch.randn(rows, HV, V, device=device).to(torch.bfloat16)
    w = 1 + 0.1 * torch.randn(V, device=device)
    w[32:64] *= 2.0**-120
    w[64:96] = 0
    w[96:] *= 2.0**-130
    si, cu, acc = si.to(device), cu.to(device), acc.to(device)
    return (qkv, a, b, A_log, dt_bias, si, cu, acc, state, gate, w.to(torch.bfloat16))


def _poisoned(rows, HV, device):
    """bf16 output, e4m3 values and UE8M0 scales filled with NaN encodings."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        gdn_mxfp8_scale_numel,
    )

    out = torch.full((rows, HV, V), float("nan"), dtype=torch.bfloat16, device=device)
    q = torch.full((rows, HV * V), 0x7F, dtype=torch.uint8, device=device)
    numel = gdn_mxfp8_scale_numel(rows, HV * V)
    scale = torch.full((numel,), 0xFF, dtype=torch.uint8, device=device)
    return out, q.view(torch.float8_e4m3fn), scale


@pytest.mark.parametrize(
    "tune", ["", "RS=1+PF=212", "RS=1+LASTN=1+FADD2=1+PF=212", "RPT=4+MINB=1+RS=1"]
)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [5, 6])
@pytest.mark.parametrize("n", [1, 8, 56, 72, 96])
def test_gdn_mtp_cuda_fused_quant_bitwise(tune, state_dtype, width, n):
    """VLLM_GDN_MTP_FUSED_QUANT: run_quant writes the e4m3 values and swizzled
    scales of the two-kernel path it replaces (run, then gdn_gated_norm_mxfp8
    with norm_rows=(0, 0) and the device row count), bit for bit, into
    NaN-poisoned buffers, graph-padding rows and 128-row scale padding
    included. The state writes are unchanged; the bf16 output is not written.
    """
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        gdn_gated_norm_mxfp8,
    )

    torch.manual_seed(10 * n + width)
    pad = n % 3
    args = _decode_batch(n, width, pad, state_dtype)
    cu, state, gate, w = args[6], args[8], args[9], args[10]
    rows, HV = gate.shape[0], gate.shape[1]
    N = n + pad

    ref_state = state.clone()
    out, ref_q, ref_scale = _poisoned(rows, HV, gate.device)
    run_args = list(args)
    run_args[8] = ref_state
    assert _ext(tune, False).run(*run_args, out, K**-0.5, EPS, False)
    gdn_gated_norm_mxfp8(
        out, gate, w, EPS, "silu", ref_q, ref_scale, (0, 0), cu[N : N + 1]
    )

    new_state = state.clone()
    new_out, q, scale = _poisoned(rows, HV, gate.device)
    run_args[8] = new_state
    assert _ext(tune, True).run_quant(*run_args, new_out, q, scale, K**-0.5, EPS, False)

    assert torch.equal(new_state, ref_state)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(scale, ref_scale)
    assert not (q.view(torch.uint8) & 0x7F == 0x7F).any()
    assert not (scale == 0xFF).any()
    assert torch.isnan(new_out).all()
    if n >= 8:
        # Blocks whose amax / 448 is an fp32 subnormal that rounds up to
        # UE8M0 1 (0 with flush-to-zero multiplies) are covered.
        blocks = out[: int(cu[N])].float().abs().view(-1, 32).amax(-1)
        edge = (blocks > 448 * 2.0**-127) & (blocks < 448 * 2.0**-126)
        assert int(edge.sum()) > 0


def test_gdn_mtp_cuda_fused_quant_rejects_bad_outputs():
    """run_quant launches nothing and returns False (the caller keeps the
    separate quant) for outputs it cannot write, or without requests (no grid
    to zero the rows).
    """
    torch.manual_seed(0)
    run_quant = _ext("", True).run_quant
    args = list(_decode_batch(3, 4, 0, torch.bfloat16, H=1, HV=2))
    rows, HV, device = args[0].shape[0], 2, args[0].device
    out, q, scale = _poisoned(rows, HV, device)
    _, short_q, _ = _poisoned(rows - 1, HV, device)
    no_requests = list(args)
    no_requests[5] = args[5][:0]
    no_requests[6] = args[6][:1]
    no_requests[7] = args[7][:0]
    for run_args, q_, scale_ in (
        (args, short_q, scale),
        (args, q.view(torch.uint8), scale),
        (args, q, scale[:-4]),
        (no_requests, q, scale),
    ):
        assert not run_quant(*run_args, out, q_, scale_, K**-0.5, EPS, False)
    assert (q.view(torch.uint8) == 0x7F).all() and (scale == 0xFF).all()


@pytest.mark.parametrize("H,HV", [(2, 4), (1, 4)])
@pytest.mark.parametrize("width", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("act", ["silu", "sigmoid"])
@pytest.mark.parametrize("tune", ["1", "QS=3+PF=3"])
def test_gdn_mtp_cuda_wy(H, HV, width, act, tune):
    """VLLM_GDN_MTP_CUDA_WY build (bf16 state): matches the float64 reference
    like the stock kernel, agrees with the stock kernel to bf16 rounding, and
    leaves every slot that no destination names untouched.
    """
    torch.manual_seed(width + HV // H)
    n, lens, slots, L, args = _inputs(torch.bfloat16, H, HV, width)
    qkv, a, b, A_log, dt_bias, si, cu, acc, state, gate, w = args
    scale = K**-0.5
    ref_out, ref_state = _reference(
        qkv, a, b, A_log, dt_bias, si, cu, acc, state, gate, w, scale, act, H
    )
    res = []
    for ext in (
        gdn_mtp_cuda.build(gdn_mtp_cuda.tuned_source("", pdl=False)),
        gdn_mtp_cuda.build(gdn_mtp_cuda.wy_source(tune, pdl=True)),
    ):
        st = state.clone()
        out = torch.full((L, HV, V), 7.0, dtype=torch.bfloat16, device=state.device)
        run_args = list(args)
        run_args[8] = st
        assert ext.run(*run_args, out, scale, EPS, act == "sigmoid")
        res.append((st, out))
    (stock_state, stock_out), (wy_state, wy_out) = res

    torch.testing.assert_close(wy_out.double(), ref_out, atol=3e-2, rtol=3e-2)
    written = torch.zeros(slots, dtype=torch.bool, device=state.device)
    for r in range(n):
        n_acc = int(acc[r])
        if 0 < n_acc <= width and int(si[r, n_acc - 1]) > 0:
            for t in range(lens[r]):
                if int(si[r, t]) > 0:
                    written[int(si[r, t])] = True
    torch.testing.assert_close(
        wy_state[written].double(), ref_state[written], atol=2e-2, rtol=2e-2
    )
    assert torch.equal(wy_state[~written], state[~written])
    # Same values as the stock kernel up to bf16 rounding of the stored state
    # and of the output before the norm (fp32 re-association only).
    ref = stock_state.double()
    ds = (wy_state.double() - ref).norm() / ref.norm()
    do = (wy_out.double() - stock_out.double()).norm() / stock_out.double().norm()
    assert ds < 1e-4, ds
    assert do < 1e-3, do
    # Requests with an invalid source: zero output (as stock).
    assert torch.equal(
        wy_out[int(cu[2]) : int(cu[3])], stock_out[int(cu[2]) : int(cu[3])]
    )


def test_gdn_mtp_cuda_wy_rejects_fp32_state():
    """The WY build supports the bf16 state only: fp32 launches nothing."""
    _, _, _, L, args = _inputs(torch.float32, 2, 4, 4)
    ext = gdn_mtp_cuda.build(gdn_mtp_cuda.wy_source("1", pdl=False))
    out = torch.full((L, 4, V), 7.0, dtype=torch.bfloat16, device=args[0].device)
    assert not ext.run(*args, out, K**-0.5, EPS, False)
    assert torch.all(out == 7.0)
