# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for text-only M-RoPE position preparation."""

from types import SimpleNamespace
from unittest.mock import Mock

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
    ("supports_multimodal_inputs", "model_opt_in", "expected_linear"),
    [
        (False, True, True),
        (True, True, False),
        (False, False, False),
        (True, False, False),
    ],
)
def test_get_rope_state_requires_text_only_deployment_and_model_opt_in(
    monkeypatch,
    supports_multimodal_inputs,
    model_opt_in,
    expected_linear,
):
    """Only opted-in models without multimodal ingress skip staging."""
    monkeypatch.setattr(rope, "StagedWriteTensor", Mock())
    monkeypatch.setattr(rope, "UvaBackedTensor", Mock())
    monkeypatch.setattr(
        rope,
        "MULTIMODAL_REGISTRY",
        SimpleNamespace(
            supports_multimodal_inputs=lambda _: supports_multimodal_inputs
        ),
    )
    model_config = SimpleNamespace(uses_mrope=True, mrope_num_dims=3)

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
    assert state.text_only is None
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
    # No per-request text_only flag is staged or uploaded in linear mode.
    state.apply_staged_writes()


class _MixedMRoPEModel:
    """Text-only positions are arange(L) in every dim with delta 0; requests
    with multimodal features get distinct per-dim positions and a nonzero
    delta, so they must be read from the staged buffer."""

    MM_DELTA = -7

    def __init__(self):
        self.get_positions = Mock(side_effect=self._get_positions)

    def _get_positions(self, input_tokens, mm_features):
        positions = torch.arange(len(input_tokens))
        if not mm_features:
            return positions.unsqueeze(0).expand(3, -1), 0
        return torch.stack([positions // (d + 1) for d in range(3)]), self.MM_DELTA

    def get_mrope_input_positions(self, input_tokens, mm_features):
        return self.get_positions(input_tokens, mm_features)


def _staged_state(
    device: torch.device, max_num_reqs=4, max_num_tokens=8, max_model_len=16
):
    return rope.RopeState(
        num_dims=3,
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
        max_model_len=max_model_len,
        device=device,
    )


@pytest.mark.parametrize("fast_path", [True, False])
def test_text_only_fast_path_host_staging(monkeypatch, fast_path):
    """With multimodal ingress, text-only requests skip the host position
    table (fast path on) while multimodal requests are still staged."""
    monkeypatch.setenv("VLLM_MROPE_TEXT_ONLY_FAST_PATH", "1" if fast_path else "0")
    state = _staged_state(torch.device("cpu"))
    assert state.text_only_fast_path is fast_path
    model = _MixedMRoPEModel()
    staged = state.prefill_positions._staged_write_indices

    # A reused slot: a multimodal request first, then a text-only one.
    state.init_prefill_positions(0, model, [1, 2, 3], mm_features=[object()])
    assert state.text_only.np[0] == 0
    assert state.prefill_delta.np[0] == _MixedMRoPEModel.MM_DELTA
    assert staged == [0, 1, 2]
    state.init_prefill_positions(0, model, [1, 2, 3, 4], mm_features=[])

    if fast_path:
        assert model.get_positions.call_count == 1
        assert staged == [0, 1, 2]
        assert state.text_only.np[0] == 1
    else:
        assert model.get_positions.call_count == 2
        assert staged == [0, 1, 2, 0, 1, 2]
        assert state.text_only.np[0] == 0
    assert state.prefill_delta.np[0] == 0

    # The staged position writes need the Triton write kernel; only the
    # per-request flag upload is checked here.
    state.prefill_positions.clear_staged_writes()
    state.apply_staged_writes()
    torch.testing.assert_close(
        state.text_only.gpu.cpu(), torch.from_numpy(state.text_only.np)
    )


def _mixed_batch_positions(device, fast_path, monkeypatch) -> list[torch.Tensor]:
    monkeypatch.setenv("VLLM_MROPE_TEXT_ONLY_FAST_PATH", "1" if fast_path else "0")
    prompt_lens = [100, 1500, 3000, 777]
    has_mm = [False, True, False, True]
    max_reqs = 8
    state = _staged_state(
        device, max_num_reqs=max_reqs, max_num_tokens=4096, max_model_len=8192
    )
    model = _MixedMRoPEModel()
    for req_idx, (length, mm) in enumerate(zip(prompt_lens, has_mm)):
        mm_features = [object()] if mm else []
        state.init_prefill_positions(req_idx, model, [7] * length, mm_features)
    state.apply_staged_writes()

    num_reqs = len(prompt_lens)
    prefill_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
    prefill_lens[:num_reqs] = torch.tensor(prompt_lens, dtype=torch.int32)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    outs = []
    # Chunked prefill, the prefill tail, then decode steps.
    for step, (frac, max_query_len) in enumerate(
        [(0.5, 512), (0.99, 200), (1.0, 4), (1.0, 4)]
    ):
        computed = [min(int(n * frac), n) + step for n in prompt_lens]
        query_lens = [
            max_query_len if c >= n else min(max_query_len, n - c)
            for c, n in zip(computed, prompt_lens)
        ]
        num_computed = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        num_computed[:num_reqs] = torch.tensor(computed, dtype=torch.int32)
        query_start_loc = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(0, dtype=torch.int32)
        state.prepare_positions(idx_mapping, query_start_loc, prefill_lens, num_computed)
        outs.append(state.get_positions(int(query_start_loc[-1])).clone())
    return outs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs CUDA")
def test_text_only_fast_path_matches_staged_positions(monkeypatch):
    """Kernel-computed text-only positions are bit-identical to the staged
    ones, also with multimodal requests in the same batch."""
    device = torch.device("cuda")
    staged = _mixed_batch_positions(device, False, monkeypatch)
    fast = _mixed_batch_positions(device, True, monkeypatch)
    for ref, out in zip(staged, fast):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


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
