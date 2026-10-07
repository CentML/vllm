# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch spec-decode GDN core vs the three kernels it replaces:
``causal_conv1d_update`` -> ``gdn_mtp_recurrence`` -> ``gdn_gated_norm_mxfp8``.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA required", allow_module_level=True)

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (  # noqa: E402
    gdn_gated_norm_mxfp8,
    gdn_mxfp8_scale_numel,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gdn_mtp_decode import (  # noqa: E402
    gdn_mtp_recurrence,
)
from vllm.model_executor.layers.mamba.ops.gdn_mtp_fused_decode import (  # noqa: E402
    gdn_mtp_fused_decode,
)

H, HV, K, V, W = 16, 32, 128, 128, 4
CONV_DIM = 2 * H * K + HV * V
HIDDEN = HV * V
EPS = 1e-6


def _case(num_requests, num_padded, seed, device, S=4):
    """Requests of 1..S tokens (one without a valid source state, one with a
    skipped destination slot), plus ``num_padded`` FULL-graph padding
    requests (no tokens) whose rows are padding.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    lens = [min(x, S) for x in [S, 3, S, 1, 2, S][:num_requests]]
    acc = [min(x, S) for x in [1, 2, 4, 3, 1, 2][:num_requests]]
    if num_requests > 2:
        acc[2] = 0  # no valid source state
    n = num_requests + num_padded
    lens += [0] * num_padded
    acc += [1] * num_padded
    rows = n * S  # padded token count of the FULL graph
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    slots = 1 + n * S + 5
    si = (torch.randperm(slots - 1, generator=g)[: n * S] + 1).view(n, S)
    si = si.to(torch.int32)
    if S > 1:
        si[0, 1] = 0  # request 0 skips the SSM state of token 1
    si[num_requests:] = 0
    qkvz = torch.randn(rows, CONV_DIM + HIDDEN + 64, generator=g).to(torch.bfloat16)
    ba = torch.randn(rows, 2 * HV, generator=g).to(torch.bfloat16)
    conv_w = (0.5 * torch.randn(CONV_DIM, W, generator=g)).to(torch.bfloat16)
    # SD layout: [slots, state_len, dim] storage, [slots, dim, state_len] view.
    conv_state = torch.randn(slots, W - 1 + S - 1, CONV_DIM, generator=g)
    conv_state = conv_state.to(torch.bfloat16)
    state = 0.05 * torch.randn(slots, HV, V, K, generator=g)
    A_log = torch.log(torch.empty(HV).uniform_(1, 16, generator=g))
    dt_bias = torch.randn(HV, generator=g)
    norm_w = (1 + 0.1 * torch.randn(V, generator=g)).to(torch.bfloat16)
    t = dict(
        qkvz=qkvz, ba=ba, conv_w=conv_w, conv_state=conv_state, state=state,
        A_log=A_log, dt_bias=dt_bias, norm_w=norm_w, si=si, cu=cu,
        acc=torch.tensor(acc, dtype=torch.int32),
    )  # fmt: skip
    return {k: v.to(device) for k, v in t.items()}, n, rows


def _views(t):
    qkvz = t["qkvz"]
    mixed_qkv = qkvz[:, :CONV_DIM]
    z = qkvz[:, CONV_DIM : CONV_DIM + HIDDEN].view(-1, HV, V)
    b, a = t["ba"][:, :HV], t["ba"][:, HV:]
    return mixed_qkv, z, a, b, t["conv_state"].transpose(-1, -2)


def _outputs(rows, device):
    out_q = torch.full((rows, HIDDEN), 3.0, device=device).to(torch.float8_e4m3fn)
    out_scale = torch.full(
        (gdn_mxfp8_scale_numel(rows, HIDDEN),), 9, dtype=torch.uint8, device=device
    )
    return out_q, out_scale


def _run_reference(t, n, rows):
    mixed_qkv, z, a, b, conv_state = _views(t)
    num_actual = int(t["cu"][-1])
    conv_out = causal_conv1d_update(
        mixed_qkv[:num_actual],
        conv_state,
        t["conv_w"],
        None,
        "silu",
        conv_state_indices=t["si"][:, 0],
        num_accepted_tokens=t["acc"],
        query_start_loc=t["cu"],
        max_query_len=t["si"].shape[1],
        validate_data=False,
    )
    core = torch.full((rows, HV, V), 7.0, dtype=torch.bfloat16, device=a.device)
    gdn_mtp_recurrence(
        conv_out, a[:num_actual], b[:num_actual], t["A_log"], t["dt_bias"],
        t["si"], t["cu"], t["acc"], t["state"], core[:num_actual], scale=K**-0.5,
    )  # fmt: skip
    out_q, out_scale = _outputs(rows, a.device)
    gdn_gated_norm_mxfp8(
        core, z, t["norm_w"], EPS, "silu", out_q, out_scale,
        norm_rows=(0, rows), num_valid=t["cu"][n : n + 1],
    )  # fmt: skip
    return out_q, out_scale


def _run_fused(t, n, rows, counters=None, **kw):
    mixed_qkv, z, a, b, conv_state = _views(t)
    core = torch.full((rows, HV, V), 7.0, dtype=torch.bfloat16, device=a.device)
    out_q, out_scale = _outputs(rows, a.device)
    if counters is None:
        counters = torch.zeros(n * H, dtype=torch.int32, device=a.device)
    gdn_mtp_fused_decode(
        mixed_qkv, a, b, conv_state, t["conv_w"], t["A_log"], t["dt_bias"],
        t["si"], t["cu"], t["acc"], t["state"], core, z, t["norm_w"], EPS, "silu",
        out_q, out_scale, counters, scale=K**-0.5, **kw,
    )  # fmt: skip
    # Every arrival counter is back at zero for the next launch.
    assert int(counters.abs().sum()) == 0
    return out_q, out_scale


def _dequant(out_q, out_scale):
    """fp32 values [rows, HIDDEN // 32, 32] of the swizzled MXFP8 activation."""
    rows = out_q.shape[0]
    r = torch.arange(rows, device=out_q.device)[:, None]
    g = torch.arange(HIDDEN // 32, device=out_q.device)[None, :]
    off = (
        r // 128 * (128 * (HIDDEN // 32))
        + g // 4 * 512
        + r % 32 * 16
        + r % 128 // 32 * 4
        + g % 4
    )
    sf = out_scale[off].to(torch.float32)
    scale = torch.where(sf > 0, torch.exp2(sf - 127), torch.zeros_like(sf))
    return out_q.to(torch.float32).view(rows, -1, 32) * scale[..., None]


def _padding_scales(out_scale, rows, n_valid):
    """Scale bytes of rows n_valid .. the 128-row padding."""
    padded = (rows + 127) // 128 * 128
    r = torch.arange(n_valid, padded, device=out_scale.device)[:, None]
    g = torch.arange(HIDDEN // 32, device=out_scale.device)[None, :]
    off = (
        r // 128 * (128 * (HIDDEN // 32))
        + g // 4 * 512
        + r % 32 * 16
        + r % 128 // 32 * 4
        + g % 4
    )
    return out_scale[off]


@pytest.mark.parametrize(
    "num_requests,num_padded", [(1, 0), (2, 0), (3, 1), (4, 0), (6, 2)]
)
@pytest.mark.parametrize("block_v,num_warps", [(None, None), (8, 2), (32, 4)])
@pytest.mark.parametrize("spec_tokens", [1, 4, 5, 6, 7])
def test_fused_decode_matches_three_kernels(
    num_requests, num_padded, block_v, num_warps, spec_tokens
):
    """``spec_tokens`` = MTP draft tokens + 1 per request (k = 3 -> 4)."""
    device = torch.device("cuda")
    ref_t, n, rows = _case(
        num_requests, num_padded, 7 * num_requests, device, S=spec_tokens
    )
    fused_t = {k: v.clone() for k, v in ref_t.items()}
    n_valid = int(ref_t["cu"][-1])

    ref_q, ref_scale = _run_reference(ref_t, n, rows)
    out_q, out_scale = _run_fused(
        fused_t, n, rows, block_v=block_v, num_warps=num_warps
    )

    # Conv window roll (stored raw values, every slot): bitwise.
    assert torch.equal(fused_t["conv_state"], ref_t["conv_state"])
    # SSM states: written slots agree to fp32 rounding (the q/k L2-norm sums
    # reduce in a different order); untouched slots keep their bytes.
    torch.testing.assert_close(fused_t["state"], ref_t["state"], rtol=1e-5, atol=1e-6)
    # Padding rows: zero values, and zero scales up to the 128-row padding.
    assert not out_q[n_valid:].view(torch.uint8).any()
    assert not _padding_scales(out_scale, rows, n_valid).any()
    # Valid rows: the same MXFP8 activation up to rare one-step rounding
    # flips of a value or a block scale.
    deq = _dequant(out_q, out_scale)[:n_valid]
    ref = _dequant(ref_q, ref_scale)[:n_valid]
    torch.testing.assert_close(deq, ref, rtol=0.07, atol=1e-3)
    flips = out_q[:n_valid].view(torch.uint8) != ref_q[:n_valid].view(torch.uint8)
    assert flips.float().mean() < 1e-3


def test_fused_decode_replays_identically():
    """The projection rows are not modified (the conv is not in place) and the
    arrival counters return to zero, so back-to-back launches from the same
    inputs (graph replays) give the same bytes.
    """
    device = torch.device("cuda")
    t0, n, rows = _case(2, 0, 3, device)
    counters = torch.zeros(n * H, dtype=torch.int32, device=device)
    results = []
    for _ in range(3):
        t = {k: v.clone() for k, v in t0.items()}
        out_q, out_scale = _run_fused(t, n, rows, counters=counters)
        assert torch.equal(t["qkvz"], t0["qkvz"])
        results.append(
            (out_q.view(torch.uint8), out_scale, t["state"], t["conv_state"])
        )
    for r in results[1:]:
        for x, y in zip(r, results[0]):
            assert torch.equal(x, y)
