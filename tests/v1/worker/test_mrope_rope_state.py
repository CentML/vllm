# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for text-only M-RoPE position preparation."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.v1.worker.gpu.mm import rope


class _MRoPEModel(nn.Module, SupportsMRoPE):
    supports_mrope = True

    def __init__(self, supports_linear_text_mrope: bool):
        super().__init__()
        self.supports_linear_text_mrope = supports_linear_text_mrope
        self.get_positions = Mock(side_effect=self._get_positions)

    @staticmethod
    def _get_positions(input_tokens, mm_features):
        del mm_features
        positions = torch.arange(len(input_tokens)).unsqueeze(0).expand(3, -1)
        return positions, 0

    def get_mrope_input_positions(self, input_tokens, mm_features):
        return self.get_positions(input_tokens, mm_features)


@pytest.mark.parametrize(
    ("multimodal_config", "model_opt_in", "expected_linear"),
    [
        (SimpleNamespace(language_model_only=True, enable_mm_embeds=False), True, True),
        (
            SimpleNamespace(language_model_only=False, enable_mm_embeds=False),
            True,
            False,
        ),
        (SimpleNamespace(language_model_only=True, enable_mm_embeds=True), True, False),
        (
            SimpleNamespace(language_model_only=True, enable_mm_embeds=False),
            False,
            False,
        ),
        (None, True, False),
    ],
)
def test_get_rope_state_requires_explicit_text_only_mrope_config(
    monkeypatch,
    multimodal_config,
    model_opt_in,
    expected_linear,
):
    """Only opted-in deployments without multimodal ingress skip staging."""
    monkeypatch.setattr(rope, "StagedWriteTensor", Mock())
    monkeypatch.setattr(rope, "UvaBackedTensor", Mock())
    model_config = SimpleNamespace(
        uses_mrope=True,
        mrope_num_dims=3,
        multimodal_config=multimodal_config,
    )

    state = rope.get_rope_state(
        model_config,
        _MRoPEModel(model_opt_in),
        max_num_reqs=2,
        max_num_tokens=8,
        max_model_len=16,
        device=torch.device("cpu"),
    )

    assert state.linear_prefill_positions is expected_linear


def test_linear_rope_state_contract(monkeypatch):
    """Text-only state skips staging and rejects media before model setup."""

    def fail_staging_allocation(*args, **kwargs):
        raise AssertionError("text-only path allocated a staged prefill buffer")

    monkeypatch.setattr(rope, "StagedWriteTensor", fail_staging_allocation)
    monkeypatch.setattr(rope, "UvaBackedTensor", fail_staging_allocation)

    state = rope.RopeState(
        num_dims=3,
        max_num_reqs=2,
        max_num_tokens=8,
        max_model_len=16,
        device=torch.device("cpu"),
        linear_prefill_positions=True,
    )

    assert state.prefill_positions is None
    assert state.prefill_delta is None
    model = _MRoPEModel(supports_linear_text_mrope=True)

    state.init_prefill_positions(
        req_idx=0,
        model=model,
        prefill_token_ids=[1, 2, 3],
        mm_features=[],
    )

    model.get_positions.assert_not_called()
    mm_feature = object()

    with pytest.raises(RuntimeError, match="multimodal"):
        state.init_prefill_positions(
            req_idx=0,
            model=model,
            prefill_token_ids=[1, 2, 3],
            mm_features=[mm_feature],
        )

    model.get_positions.assert_not_called()


def test_generic_rope_state_preserves_multimodal_position_constructor():
    """Multimodal-capable deployments retain the model-specific M-RoPE path."""
    state = object.__new__(rope.RopeState)
    state.linear_prefill_positions = False
    state.num_dims = 3
    state.prefill_delta = SimpleNamespace(np=np.zeros(2, dtype=np.int32))
    state.prefill_positions = Mock()
    model = _MRoPEModel(supports_linear_text_mrope=True)
    mm_feature = object()

    state.init_prefill_positions(
        req_idx=1,
        model=model,
        prefill_token_ids=[4, 5, 6],
        mm_features=[mm_feature],
    )

    model.get_positions.assert_called_once_with([4, 5, 6], [mm_feature])
    assert state.prefill_delta.np[1] == 0
    assert state.prefill_positions.stage_write.call_count == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs CUDA")
def test_linear_rope_positions_match_staged_path_for_mixed_request_shapes():
    """Direct text positions match staged M-RoPE for prefill and multi-token decode."""
    device = torch.device("cuda")
    max_num_tokens = 1152
    max_model_len = 1600
    generic = rope.RopeState(
        num_dims=3,
        max_num_reqs=3,
        max_num_tokens=max_num_tokens,
        max_model_len=max_model_len,
        device=device,
    )
    linear = rope.RopeState(
        num_dims=3,
        max_num_reqs=3,
        max_num_tokens=max_num_tokens,
        max_model_len=max_model_len,
        device=device,
        linear_prefill_positions=True,
    )
    model = _MRoPEModel(supports_linear_text_mrope=True)
    for req_idx, prefill_len in enumerate((64, 9, 1500)):
        generic.init_prefill_positions(
            req_idx, model, list(range(prefill_len)), mm_features=[]
        )
    generic.apply_staged_writes()

    # Batch order differs from persistent request-state order. It contains a
    # >1024-token chunk, a prefix-cached prefill tail, and four MTP decode tokens.
    idx_mapping = torch.tensor([2, 0, 1], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor(
        [0, 1107, 1111, 1115], dtype=torch.int32, device=device
    )
    prefill_lens = torch.tensor([64, 9, 1500], dtype=torch.int32, device=device)
    num_computed_tokens = torch.tensor([60, 9, 120], dtype=torch.int32, device=device)

    for state in (generic, linear):
        state.prepare_positions(
            idx_mapping, query_start_loc, prefill_lens, num_computed_tokens
        )
    torch.accelerator.synchronize()

    total_num_tokens = query_start_loc[-1].item()
    generic_positions = generic.get_positions(total_num_tokens)
    linear_positions = linear.get_positions(total_num_tokens)
    expected_1d = torch.cat(
        (
            torch.arange(120, 1227, device=device),
            torch.arange(60, 64, device=device),
            torch.arange(9, 13, device=device),
        )
    )
    expected = expected_1d.unsqueeze(0).expand(3, -1)

    torch.testing.assert_close(linear_positions, generic_positions, rtol=0, atol=0)
    torch.testing.assert_close(linear_positions, expected, rtol=0, atol=0)
    assert linear_positions.stride(0) == max_num_tokens + 1
    assert linear.get_positions(max_num_tokens).stride(0) == max_num_tokens + 1
