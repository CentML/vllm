# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MTP_DRAFT_WINDOW: the MTP draft attention window (no GPU needed)."""

import types

import pytest
import torch.nn as nn

from vllm.v1.attention.backends import draft_prefill_pruning as pruning


class _FakeAttention(nn.Module):
    def __init__(self, window_left=-1):
        super().__init__()
        self.impl = types.SimpleNamespace(window_left=window_left)
        self.layer_name = f"mtp.attn.{window_left}"


@pytest.fixture
def fake_attention(monkeypatch):
    import vllm.model_executor.layers.attention.attention as attn_mod

    monkeypatch.setattr(attn_mod, "Attention", _FakeAttention)
    return _FakeAttention


def _model(*windows):
    m = nn.Module()
    m.layers = nn.ModuleList([_FakeAttention(w) for w in windows])
    m.other = nn.Linear(2, 2)
    return m


def test_off_by_default(fake_attention, monkeypatch):
    monkeypatch.setattr(pruning, "DRAFT_WINDOW", 0)
    m = _model(-1)
    pruning.install_window(m)
    assert not hasattr(m.layers[0].impl, "draft_window_left")


def test_sets_window_on_draft_attention(fake_attention, monkeypatch):
    monkeypatch.setattr(pruning, "DRAFT_WINDOW", 8192)
    m = _model(-1, -1)
    pruning.install_window(m)
    assert [layer.impl.draft_window_left for layer in m.layers] == [8191, 8191]


def test_keeps_a_narrower_model_window(fake_attention, monkeypatch):
    monkeypatch.setattr(pruning, "DRAFT_WINDOW", 8192)
    m = _model(1023, 16383)
    pruning.install_window(m)
    assert not hasattr(m.layers[0].impl, "draft_window_left")  # 1024-token model window is narrower: kept
    assert m.layers[1].impl.draft_window_left == 8191


def test_impl_without_draft_window_uses_model_window():
    impl = types.SimpleNamespace(window_left=-1)
    assert getattr(impl, "draft_window_left", impl.window_left) == -1
