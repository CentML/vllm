# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V-split FlashInfer GDN prefill: bitwise equal to FlashInfer's non-CP kernel
on the in-place state pool, through the Qwen GDN layer's dispatch.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip(
        reason="The V-split GDN prefill kernel is SM10x-only.",
        allow_module_level=True,
    )

from flashinfer.gdn_prefill import chunk_gated_delta_rule  # noqa: E402

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as qgdn  # noqa: E402

H, HV, D = 16, 32, 128


def _case(seqlens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(seqlens)
    q = torch.nn.functional.normalize(
        torch.randn(T, H, D, device="cuda", generator=g), dim=-1
    ).to(torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(T, H, D, device="cuda", generator=g), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(T, HV, D, device="cuda", generator=g).to(torch.bfloat16)
    g_exp = torch.exp(-torch.rand(T, HV, device="cuda", generator=g) * 0.5)
    beta = torch.sigmoid(torch.randn(T, HV, device="cuda", generator=g))
    slots = 2 * len(seqlens) + 3
    pool = 0.05 * torch.randn(slots, HV, D, D, device="cuda", generator=g)
    perm = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(seed))
    state_indices = (perm[: len(seqlens)] + 1).to(torch.int32).cuda()
    cu32 = torch.tensor(
        [0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.int32
    ).cuda()
    return q, k, v, g_exp, beta, pool, state_indices, cu32


@pytest.fixture(scope="module", autouse=True)
def _vsplit_enabled():
    qgdn._gdn_vsplit_warmup(H, HV, D, torch.bfloat16, torch.device("cuda"))
    assert qgdn._gdn_vsplit_ready, "V-split kernel failed to compile"


@pytest.mark.parametrize(
    "seqlens,expect_vsplit",
    [
        ([64], True),
        ([1000], True),
        ([3584], True),
        ([6144], True),
        ([8192], True),  # largest single sequence on the non-CP path
        ([2048, 64], True),
        ([3000, 3119, 17], True),
        ([1500] * 6, False),  # cost model keeps the stock kernel (VR, 212 SMs)
        ([1] * 8 + [3000], True),  # stock grid exceeds the SM count
        ([300] * 12, True),
    ],
)
def test_vsplit_dispatch_bitwise_vs_flashinfer_non_cp(
    seqlens, expect_vsplit, monkeypatch
):
    gdn_vsplit = qgdn._gdn_vsplit_ready[0]
    calls = []
    vsplit_fn = gdn_vsplit.chunk_gated_delta_rule_vsplit

    def counting_vsplit(*args, **kwargs):
        calls.append(kwargs["v_split"])
        return vsplit_fn(*args, **kwargs)

    monkeypatch.setattr(gdn_vsplit, "chunk_gated_delta_rule_vsplit", counting_vsplit)
    q, k, v, g_exp, beta, pool, state_indices, cu32 = _case(seqlens, len(seqlens))
    pool_ref = pool.clone()
    out_ref = torch.empty_like(v)
    chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g_exp,
        beta=beta,
        initial_state=pool_ref,
        output_final_state=True,
        cu_seqlens=cu32.to(torch.int64),
        output=out_ref,
        output_state=pool_ref,
        state_indices=state_indices,
        use_cp=False,
    )
    out = torch.empty_like(v)
    qgdn.fi_chunk_gated_delta_rule(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g_exp.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu32.to(torch.int64),
        use_qk_l2norm_in_kernel=False,
        g_is_exp=True,
        output=out,
        state_indices=state_indices,
        max_seqlen=max(seqlens),
        cu_seqlens_i32=cu32,
    )
    torch.accelerator.synchronize()
    if torch.cuda.get_device_properties(0).multi_processor_count == 212:
        assert calls == ([2] if expect_vsplit else [])
    assert torch.equal(out, out_ref)
    # Every pool row, including the ones outside state_indices.
    assert torch.equal(pool, pool_ref)
