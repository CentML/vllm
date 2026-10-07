# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_SAMPLER_STATE_DIRTY: Sampler.apply_staged_writes skips re-staging the
per-request sampler states on steps without a new request. That is exact only
because every host-side writer of those states runs from Sampler.add_request;
these tests pin that contract. The sampler states are UVA-backed, so the
tests need a CUDA device even though the sampler runs on CPU here."""

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample import sampler as sampler_mod
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="UVA-backed sampler states need CUDA"
)

DEVICE = torch.device("cpu")
VOCAB_SIZE = 128
MAX_REQS = 4


class _ReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def _make_sampler() -> Sampler:
    req_states = RequestState(
        max_num_reqs=MAX_REQS,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        max_num_reqs=MAX_REQS,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        reasoning_config=_ReasoningConfig(),
        enable_trace_replay=True,
    )


_SUB_STATES = (
    "sampling_states",
    "penalties_state",
    "logit_bias_state",
    "bad_words_state",
    "logprob_token_ids_state",
    "thinking_budget_state",
    "trace_replay_state",
)


def _count_staging(smp: Sampler) -> dict[str, int]:
    calls = dict.fromkeys(_SUB_STATES, 0)
    for name in _SUB_STATES:
        state = getattr(smp, name)
        orig = state.apply_staged_writes

        def wrapped(_orig=orig, _name=name):
            calls[_name] += 1
            _orig()

        state.apply_staged_writes = wrapped
    return calls


def _assert_device_matches_host(smp: Sampler) -> None:
    st = smp.sampling_states
    for p in (st.temperature, st.top_k, st.top_p, st.min_p, st.seeds):
        np.testing.assert_array_equal(p.gpu.cpu().numpy(), p.np)
    tb = smp.thinking_budget_state
    np.testing.assert_array_equal(
        tb.thinking_token_budget.gpu.cpu().numpy(), tb.thinking_token_budget.np
    )
    for uva in (
        smp.logprob_token_ids_state.num_token_ids,
        smp.trace_replay_state.trace_len,
        smp.bad_words_state.num_bad_words,
        smp.logit_bias_state.min_lens,
    ):
        np.testing.assert_array_equal(uva.gpu.cpu().numpy(), uva.np)


def _params(i: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.5 + 0.25 * i, top_k=5 + i, seed=7 + i, thinking_token_budget=3 + i
    )


def test_gate_on_skips_only_steps_without_new_requests(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_STATE_DIRTY_GATE", True)
    smp = _make_sampler()
    calls = _count_staging(smp)

    smp.add_request(0, 1, _params(0))
    smp.apply_staged_writes()
    assert all(n == 1 for n in calls.values()), calls
    tb = smp.thinking_budget_state
    assert not tb._reset_reqs and not tb._budget_dirty
    _assert_device_matches_host(smp)

    # Decode steps without admissions stage nothing and the device copies stay
    # equal to the host source of truth.
    for _ in range(3):
        smp.apply_staged_writes()
    assert all(n == 1 for n in calls.values()), calls
    _assert_device_matches_host(smp)

    smp.add_request(1, 1, _params(1))
    smp.apply_staged_writes()
    assert all(n == 2 for n in calls.values()), calls
    _assert_device_matches_host(smp)
    assert int(tb.cached_scan_pos[1]) == 0


@pytest.mark.parametrize("gate", [False, True])
def test_gate_matches_ungated_device_state(monkeypatch, gate):
    monkeypatch.setattr(sampler_mod, "_STATE_DIRTY_GATE", gate)
    smp = _make_sampler()
    calls = _count_staging(smp)
    for step in range(MAX_REQS):
        smp.add_request(step, 1, _params(step))
        smp.apply_staged_writes()
        smp.apply_staged_writes()
        _assert_device_matches_host(smp)
    expected = MAX_REQS if gate else 2 * MAX_REQS
    assert all(n == expected for n in calls.values()), calls
