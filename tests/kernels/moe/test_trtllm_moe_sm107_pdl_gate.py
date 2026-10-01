# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which trtllm-gen MoE calls the SM107 PDL opt-in may cover."""

import pytest

from vllm.model_executor.layers.fused_moe.config import RoutingMethodType as R
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
    _sm107_moe_pdl_allowed,
)


@pytest.fixture(autouse=True)
def _max_tokens_16(monkeypatch):
    monkeypatch.setenv("VLLM_FI_SM107_MOE_PDL_MAX_TOKENS", "16")


def allowed(routing, n_group=None, top_k=8, tokens=8, experts=256, hidden=2048):
    return _sm107_moe_pdl_allowed(tokens, experts, hidden, routing, n_group, top_k)


@pytest.mark.parametrize(
    "routing",
    [R.Default, R.Renormalize, R.RenormalizeNaive, R.TopK, R.SigmoidRenorm, R.Sigmoid],
)
def test_routing_custom_family_allowed(routing):
    assert allowed(routing)


def test_qwen36_shape_allowed_up_to_16_tokens():
    # Qwen3.6-35B: RenormalizeNaive, 256 experts, top_k 8, hidden 2048.
    assert allowed(R.RenormalizeNaive, tokens=16)
    assert not allowed(R.RenormalizeNaive, tokens=17)
    assert not allowed(R.RenormalizeNaive, tokens=0)


@pytest.mark.parametrize("n_group", [2, 8])
def test_grouped_deepseek_v3_rejected(n_group):
    # routingDeepSeek takes the Cluster kernel at every T <= 16.
    assert not allowed(R.DeepSeekV3, n_group=n_group)


@pytest.mark.parametrize("n_group", [None, 0, 1])
def test_ungrouped_deepseek_v3_allowed(n_group):
    assert allowed(R.DeepSeekV3, n_group=n_group)


def test_llama4_rejected():
    assert not allowed(R.Llama4, top_k=1)


@pytest.mark.parametrize("routing", [R.DeepSeekV3, R.MiniMax2])
def test_sigmoid_bias_top_k_bound(routing):
    # top_k 23-32 dispatches to Tier<1024, 32> (Cluster path at T = 5..16).
    assert allowed(routing, top_k=22)
    assert not allowed(routing, top_k=23)
    assert not allowed(routing, top_k=32)


def test_non_sigmoid_bias_top_k_not_bounded():
    assert allowed(R.Renormalize, top_k=32)


@pytest.mark.parametrize(
    "routing", [R.Unspecified, R.DeepseekV4, R.Custom, R.Simulated]
)
def test_other_routing_rejected(routing):
    assert not allowed(routing)


def test_env_zero_disables(monkeypatch):
    monkeypatch.setenv("VLLM_FI_SM107_MOE_PDL_MAX_TOKENS", "0")
    assert not allowed(R.RenormalizeNaive)


def test_expert_and_finalize_grid_bounds():
    assert not allowed(R.RenormalizeNaive, experts=513)
    # (hidden / 256 rounded up) * tokens must stay below 1184 finalize CTAs.
    assert not allowed(R.RenormalizeNaive, tokens=16, hidden=256 * 74)
    assert allowed(R.RenormalizeNaive, tokens=16, hidden=256 * 73)


# Above 16 tokens: only where the single-CTA routing permutation
# (flashinfer_exact_routing, GS2_ROUTE) replaces the Cluster/Coop kernels.
@pytest.fixture
def single_cta(monkeypatch):
    import vllm.model_executor.layers.fused_moe.flashinfer_exact_routing as fer

    monkeypatch.setenv("VLLM_FI_SM107_MOE_PDL_MAX_TOKENS", "4096")
    for name in (
        "GS2_ROUTE_MAXN",
        "GS2_ROUTE_MINTOK",
        "GS2_ROUTE_LARGE",
        "GS2_ROUTE_PACKED",
        "FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GS2_ROUTE", "1")
    monkeypatch.setattr(fer, "_installed", True)
    return monkeypatch


def big(tokens, deferred=True, routing=R.RenormalizeNaive, experts=256, top_k=8):
    return _sm107_moe_pdl_allowed(
        tokens, experts, 2048, routing, None, top_k, deferred=deferred
    )


def test_above_16_needs_the_loaded_patched_module(single_cta):
    import vllm.model_executor.layers.fused_moe.flashinfer_exact_routing as fer

    assert big(17)
    single_cta.setattr(fer, "_installed", False)  # stock module: Cluster/Coop
    assert not big(17) and big(16)
    single_cta.setattr(fer, "_installed", True)
    single_cta.setenv("GS2_ROUTE", "0")  # runtime switch selects the stock kernels
    assert not big(64) and big(16)


def test_single_cta_range(single_cta):
    # GS2_ROUTE_MAXN default 8192 entries -> 1024 tokens at top-8.
    assert big(17) and big(256) and big(257) and big(1024)
    assert not big(1025)
    single_cta.setenv("GS2_ROUTE_MAXN", "1000000")  # capped at 32768 entries
    assert big(4096) and not big(4097)
    single_cta.setenv("GS2_ROUTE_LARGE", "0")  # > 256 tokens keep Coop
    assert big(256) and not big(257)


def test_split_top_k_condition_for_17_to_256(single_cta):
    # E < 160 runs the fused score + permutation Cluster kernel at 17-256 tokens.
    assert not big(64, experts=128)
    assert big(512, experts=128)  # large-batch hook covers any E <= 1024
    single_cta.setenv("FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN", "1")
    assert big(64, experts=128)
    single_cta.setenv("FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN", "off")
    assert not big(64) and big(512)


def test_above_16_other_routing_rejected(single_cta):
    assert not big(64, routing=R.Default)
    assert not big(64, routing=R.TopK)
    assert big(16, routing=R.Default)


def test_finalize_vecload_excluded_unless_deferred(single_cta):
    # finalizeKernelVecLoad runs from 148 tokens at hidden 2048.
    assert big(147, deferred=False)
    assert not big(148, deferred=False)
    assert big(148, deferred=True)
