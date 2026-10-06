# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gating of the opt-in MTP draft prefill pruning (no GPU needed)."""

import pytest

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends import draft_prefill_pruning as pruning


@pytest.fixture
def ctx(monkeypatch):
    monkeypatch.setattr(pruning, "ENABLED", True)
    monkeypatch.setattr(pruning, "CTX", pruning._Ctx())
    return pruning.CTX


def _scope(mode=CUDAGraphMode.PIECEWISE, attn_metadata=None):
    spec = object()
    md = {} if attn_metadata is None else attn_metadata
    return spec, pruning.prefill_scope(spec, 7, md, mode)


def test_inactive_before_capture(ctx):
    _, scope = _scope()
    with scope:
        assert not ctx.active


def test_active_only_inside_eager_or_piecewise_prefill(ctx):
    pruning.mark_ready()
    for mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
        spec, scope = _scope(mode)
        with scope:
            assert ctx.active and ctx.spec is spec and ctx.num_reqs == 7
        assert not ctx.active
    _, scope = _scope(CUDAGraphMode.FULL)
    with scope:
        assert not ctx.active


def test_no_attn_metadata_or_disabled(ctx, monkeypatch):
    pruning.mark_ready()
    with pruning.prefill_scope(object(), 3, None, CUDAGraphMode.NONE):
        assert not ctx.active
    monkeypatch.setattr(pruning, "ENABLED", False)
    _, scope = _scope()
    with scope:
        assert not ctx.active


def test_scope_resets_on_error(ctx):
    pruning.mark_ready()
    _, scope = _scope()
    with pytest.raises(RuntimeError), scope:
        assert ctx.active
        raise RuntimeError
    assert not ctx.active
