# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time guard of the deferred GDN state commit.

1 + num_speculative_tokens must not exceed the decode kernel's MAX_T (host-only,
no GPU needed).
"""

import pytest

from vllm.model_executor.layers.mamba.ops import gdn_state_commit as gsc


@pytest.fixture(autouse=True)
def _state_commit_on(monkeypatch):
    monkeypatch.setenv("GDN_STATE_COMMIT", "1")
    monkeypatch.delenv("GDN_STATE_COMMIT_LAYOUT_ONLY", raising=False)
    monkeypatch.delenv("GDN_STATE_COMMIT_GUARD", raising=False)


@pytest.mark.parametrize("k", range(gsc.MAX_T))
def test_width_within_max_t_passes(k):
    assert gsc.check_num_speculative_tokens(k, "test") == k + 1


@pytest.mark.parametrize("k", [gsc.MAX_T, gsc.MAX_T + 1, gsc.MAX_T + 3])
def test_width_above_max_t_refuses(k):
    with pytest.raises(RuntimeError, match="refusing to start"):
        gsc.check_num_speculative_tokens(k, "test")


def test_layout_only_is_not_limited(monkeypatch):
    monkeypatch.setenv("GDN_STATE_COMMIT_LAYOUT_ONLY", "1")
    assert gsc.check_num_speculative_tokens(gsc.MAX_T + 3, "test") == gsc.MAX_T + 4


def test_guard_switch(monkeypatch):
    assert gsc.guard_enabled()
    monkeypatch.setenv("GDN_STATE_COMMIT_GUARD", "0")
    assert not gsc.guard_enabled()
    monkeypatch.setenv("GDN_STATE_COMMIT_GUARD", "1")
    monkeypatch.setenv("GDN_STATE_COMMIT", "0")
    assert not gsc.guard_enabled()


def test_none_means_no_speculation():
    assert gsc.check_num_speculative_tokens(None, "test") == 1
