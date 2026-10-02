# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    pytest.skip(
        reason="FlashInfer GDN prefill is not supported on ROCm.",
        allow_module_level=True,
    )

import flashinfer.gdn_prefill  # noqa: E402

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as qgdn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    fi_chunk_gated_delta_rule,
)

SM10X = current_platform.is_cuda() and current_platform.is_device_capability_family(100)


def test_flashinfer_gdn_prefill_uses_int64_cu_seqlens(monkeypatch):
    captured_cu_seqlens = None

    def fake_chunk_gated_delta_rule(**kwargs):
        nonlocal captured_cu_seqlens
        captured_cu_seqlens = kwargs["cu_seqlens"]
        return kwargs["q"]

    monkeypatch.setattr(
        flashinfer.gdn_prefill,
        "chunk_gated_delta_rule",
        fake_chunk_gated_delta_rule,
    )
    q = torch.zeros(1, 2, 1, 2)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

    output, final_state = fi_chunk_gated_delta_rule(
        q=q,
        k=q,
        v=q,
        g=torch.zeros(1, 2, 1),
        beta=torch.zeros(1, 2, 1),
        initial_state=torch.zeros(1, 1, 2, 2),
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    assert captured_cu_seqlens is not None
    assert captured_cu_seqlens.dtype == torch.int64
    assert output.shape == q.shape
    assert final_state is None


H, HV, D = 16, 32, 128


def _pool_case(seqlens, seed):
    """Prefill inputs as fused_post_conv_prep hands them over (L2-normed q/k,
    exp(g), beta) and an fp32 pool with padded rows (a hybrid page holds the
    conv state next to the SSM state).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(seqlens)

    def unit(n):
        x = torch.randn(T, n, D, device="cuda", generator=g)
        return torch.nn.functional.normalize(x, dim=-1).to(torch.bfloat16)

    q, k = unit(H), unit(H)
    v = torch.randn(T, HV, D, device="cuda", generator=g).to(torch.bfloat16)
    g_exp = torch.exp(-torch.rand(T, HV, device="cuda", generator=g) * 0.5)
    beta = torch.sigmoid(torch.randn(T, HV, device="cuda", generator=g))
    slots = 2 * len(seqlens) + 3
    row = HV * D * D
    raw = torch.randn(slots, row + 64, device="cuda", generator=g)
    pool = (0.05 * raw)[:, :row].view(slots, HV, D, D)
    perm = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(seed))
    state_indices = (perm[: len(seqlens)] + 1).to(torch.int32).cuda()
    cu32 = torch.tensor(
        [0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.int32
    ).cuda()
    return q, k, v, g_exp, beta, pool, state_indices, cu32


def _with_dtype(pool, dtype):
    """``pool`` converted to ``dtype`` with the same padded row layout."""
    out = torch.empty_strided(
        pool.shape, pool.stride(), dtype=dtype, device=pool.device
    )
    out.copy_(pool)
    return out


def _in_place(inputs, pool, seqlens):
    q, k, v, g_exp, beta, _, state_indices, cu32 = inputs
    out = torch.empty_like(v)
    fi_chunk_gated_delta_rule(
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
    return out


@pytest.mark.skipif(not SM10X, reason="in-place state pool needs SM10x")
@pytest.mark.parametrize(
    "seqlens",
    [
        [64],
        [1000],
        [2048, 64],
        [3000, 3119, 17],
        [300] * 12,
        [6000],  # over the non-CP limit (4608 without V-split): CP
    ],
)
def test_bf16_pool_in_place_matches_gather_path(seqlens, monkeypatch):
    """A bf16 pool updated in place (FlashInfer ``state_indices``, non-CP or
    CP) gives the outputs and stored states of the gather path: rows copied to
    an fp32 state, the same FlashInfer kernel, the final state rounded to
    nearest even into the pool. Rows outside ``state_indices`` keep their
    bytes. An fp32 pool from the same states gives the same outputs and final
    states that round to the bf16 pool's.
    """
    # Stock FlashInfer kernels only (V-split: test_gdn_prefill_vsplit.py).
    monkeypatch.setattr(qgdn, "_gdn_vsplit_ready", [])
    cp_calls = []
    cp = flashinfer.gdn_prefill.cp_delta_rule_dsl_sm100

    def counting_cp(*args, **kwargs):
        cp_calls.append(kwargs.get("state_indices") is not None)
        return cp(*args, **kwargs)

    monkeypatch.setattr(flashinfer.gdn_prefill, "cp_delta_rule_dsl_sm100", counting_cp)
    inputs = _pool_case(seqlens, len(seqlens) + seqlens[0])
    q, k, v, g_exp, beta, pool32, state_indices, cu32 = inputs
    pool = _with_dtype(pool32, torch.bfloat16)
    pool_init = pool.clone()
    pool_ref = pool.clone()

    out = _in_place(inputs, pool, seqlens)
    in_place_cp = list(cp_calls)

    rows = state_indices.long()
    out_ref, final = fi_chunk_gated_delta_rule(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g_exp.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=pool_ref[rows],
        output_final_state=True,
        cu_seqlens=cu32.to(torch.int64),
        use_qk_l2norm_in_kernel=False,
        g_is_exp=True,
    )
    assert final.dtype == torch.float32
    pool_ref[rows] = final.to(torch.bfloat16)
    torch.accelerator.synchronize()
    # Same kernel family on both sides (CP for one long sequence).
    assert in_place_cp == [True] * len(in_place_cp)
    assert len(cp_calls) == 2 * len(in_place_cp)
    if seqlens == [6000]:
        assert in_place_cp == [True]
    assert torch.equal(out, out_ref.squeeze(0))
    # Every pool row, including the ones outside state_indices.
    assert torch.equal(pool, pool_ref)
    # fp32 pool in place from the same (bf16-representable) states: the same
    # outputs, and final states that round to the bf16 pool's.
    pool_f = _with_dtype(pool_init, torch.float32)
    out_f = _in_place(inputs, pool_f, seqlens)
    torch.accelerator.synchronize()
    assert torch.equal(out, out_f)
    assert torch.equal(pool[rows], pool_f[rows].to(torch.bfloat16))
