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
