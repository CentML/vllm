# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton value-split GDN spec-decode recurrence vs a float64 reference."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA required", allow_module_level=True)

from vllm.model_executor.layers.mamba.ops.gdn_mtp_decode import (  # noqa: E402
    gdn_mtp_recurrence,
)

H, HV, K, V = 2, 4, 128, 128
QKV = 2 * H * K + HV * V


def _reference(qkv, a, b, A_log, dt_bias, si, cu, acc, state, scale):
    """Per request: state of the last accepted token -> gated delta rule over
    the request's tokens; state after token t -> slot si[r, t] (if > 0).
    """
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
            out[bos + t] = torch.einsum("hvk,hk->hv", h, q)
            dst = int(si[r, t])
            if dst > 0:
                state[dst] = h
    return out, state


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_requests", [1, 3, 9])
@pytest.mark.parametrize("block_v", [None, 8, 64])
def test_gdn_mtp_recurrence_matches_reference(state_dtype, num_requests, block_v):
    torch.manual_seed(num_requests)
    device = torch.device("cuda")
    S = 4
    # Varlen requests (1..S tokens), one with an invalid source (num_accepted 0)
    # and one with a skipped destination slot (0).
    lens = [S, 1, 3, 2, S, S, 1, 2, 3][:num_requests]
    acc = torch.tensor([4, 1, 2, 3, 1, 2, 4, 3, 1][:num_requests], dtype=torch.int32)
    if num_requests > 2:
        acc[2] = 0
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    slots = 1 + num_requests * S + 3
    perm = torch.randperm(slots - 1) + 1
    si = perm[: num_requests * S].view(num_requests, S).to(torch.int32)
    si[0, 1] = 0  # request 0 (source si[0, 3]) skips the state of token 1
    L = int(cu[-1])
    qkvz = torch.randn(L, QKV + 64, device=device).to(torch.bfloat16)
    qkv = qkvz[:, :QKV]
    ba = torch.randn(L, 2 * HV, device=device).to(torch.bfloat16)
    b, a = ba[:, :HV], ba[:, HV:]
    A_log = torch.log(torch.empty(HV, device=device).uniform_(1, 16))
    dt_bias = torch.randn(HV, device=device)
    state = (0.05 * torch.randn(slots, HV, V, K, device=device)).to(state_dtype)
    si, cu, acc = si.to(device), cu.to(device), acc.to(device)
    out = torch.full((L, HV, V), 7.0, dtype=torch.bfloat16, device=device)
    scale = K**-0.5

    ref_out, ref_state = _reference(
        qkv, a, b, A_log, dt_bias, si, cu, acc, state, scale
    )
    new_state = state.clone()
    gdn_mtp_recurrence(
        qkv, a, b, A_log, dt_bias, si, cu, acc, new_state, out, scale, block_v=block_v
    )

    tol = 2e-2 if state_dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(out.double(), ref_out, atol=tol, rtol=tol)
    written = torch.zeros(slots, dtype=torch.bool, device=device)
    for r in range(num_requests):
        n_acc = int(acc[r])
        if 0 < n_acc <= S and int(si[r, n_acc - 1]) > 0:
            for t in range(lens[r]):
                if int(si[r, t]) > 0:
                    written[int(si[r, t])] = True
    torch.testing.assert_close(
        new_state[written].double(), ref_state[written], atol=tol, rtol=tol
    )
    # Slots that no destination names keep their bytes.
    assert torch.equal(new_state[~written], state[~written])
