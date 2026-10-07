# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampler CUDA graphs key on the addresses of the sampling kernels' cached
scratch buffers: reallocating any of them must change the key."""

import sys
import types

import pytest
import torch

from vllm.v1.worker.gpu.sample_graph import SamplerGraphs

_SPEC = "vllm.v1.worker.gpu.sample.spec_topk_topp"
_QRITA = "vllm.v1.sample.ops.topk_topp_triton"


@pytest.fixture
def fake_modules(monkeypatch):
    spec = types.ModuleType(_SPEC)
    qrita = types.ModuleType(_QRITA)
    spec._SCRATCH = {("cpu", 8): {"vals": torch.zeros(4)}}
    qrita._TRITON_BUFFER_CACHE = {("cpu", torch.float32, 8): torch.zeros(4)}
    qrita._TRITON_SPLIT_CACHE = {"cpu": {"a": torch.zeros(4)}}
    qrita._TRITON_TABLE_CACHE = {("cpu",): (torch.zeros(4), torch.zeros(4))}
    monkeypatch.setitem(sys.modules, _SPEC, spec)
    monkeypatch.setitem(sys.modules, _QRITA, qrita)
    return spec, qrita


def test_missing_modules_and_caches_contribute_nothing(monkeypatch):
    monkeypatch.delitem(sys.modules, _SPEC, raising=False)
    monkeypatch.setitem(sys.modules, _QRITA, types.ModuleType(_QRITA))
    assert SamplerGraphs._scratch_ptrs() == ()


@pytest.mark.parametrize(
    "realloc",
    [
        lambda s, q: s._SCRATCH[("cpu", 8)].__setitem__("vals", torch.zeros(8)),
        lambda s, q: q._TRITON_BUFFER_CACHE.__setitem__(
            ("cpu", torch.float32, 8), torch.zeros(8)
        ),
        lambda s, q: q._TRITON_SPLIT_CACHE["cpu"].__setitem__("a", torch.zeros(8)),
        lambda s, q: q._TRITON_TABLE_CACHE.__setitem__(
            ("cpu",), (torch.zeros(4), torch.zeros(8))
        ),
    ],
    ids=["spec_scratch", "qrita_buffer", "qrita_split", "qrita_table"],
)
def test_scratch_realloc_changes_key(fake_modules, realloc):
    before = SamplerGraphs._scratch_ptrs()
    assert len(before) == 5
    realloc(*fake_modules)
    assert SamplerGraphs._scratch_ptrs() != before
