# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register-resident CUDA GDN spec-decode recurrence + gated norm (JIT) vs a
float64 reference.
"""

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


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("H,HV", [(2, 4), (1, 4), (4, 4)])
@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("act", ["silu", "sigmoid"])
def test_gdn_mtp_cuda_matches_reference(state_dtype, H, HV, width, act):
    torch.manual_seed(width + HV // H)
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
    out = torch.full((L, HV, V), 7.0, dtype=torch.bfloat16, device=device)
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
    written = torch.zeros(slots, dtype=torch.bool, device=device)
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
