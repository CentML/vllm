# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The R4 counter hook in the real gdn_attn.py: GDNAttentionMetadataBuilder.build() calls
gdn_zero_draft.count() before splitting the batch, returns early (nothing counted, nothing on device) under CUDA graph
capture, and report() reads the device counter. gdn_attn.py's other vllm imports are auto-stubbed (this runs without
the qwen-v2 runtime); build() is stopped right after the hook by a sentinel _split_batch."""
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import sys
import types

import pytest
import torch


class _AnyMeta(type):
    """Class-level attribute access / subscripting on stub classes (e.g. AttentionCGSupport.UNIFORM_BATCH)."""

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()

    def __getitem__(cls, item):
        return cls


class _Any(metaclass=_AnyMeta):
    """Stand-in for any symbol of an absent vllm module: a class usable as a base, generic, decorator or value."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return a[0] if (len(a) == 1 and callable(a[0]) and not k) else self

    def __getattr__(self, name):
        return _Any()


def _stub_module(name):
    m = types.ModuleType(name)

    def ga(attr, _m=m):
        if attr.startswith("__"):
            raise AttributeError(attr)
        if attr == "triton":
            t = types.SimpleNamespace(jit=lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f)),
                                      cdiv=lambda x, y: -(-x // y), next_power_of_2=lambda x: x)
            return t
        if attr == "tl":
            return types.SimpleNamespace(constexpr=int)
        c = _AnyMeta(attr, (_Any,), {})
        setattr(_m, attr, c)
        return c
    m.__getattr__ = ga
    m.__path__ = []
    return m


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    REAL = {"vllm.v1.attention.backends.gdn_attn", "vllm.v1.attention.backends.gdn_zero_draft"}

    def find_spec(self, name, path, target=None):
        if not name.startswith("vllm.") or name in self.REAL:
            return None
        if importlib.machinery.PathFinder.find_spec(name, path) is not None:
            return None
        return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        return _stub_module(spec.name)

    def exec_module(self, module):
        pass


def _import_gdn_attn():
    finder = _StubFinder()
    sys.meta_path.insert(0, finder)
    try:
        sys.modules.pop("vllm.v1.attention.backends.gdn_attn", None)
        # modules that exist in the test's minimal vllm namespace but lack gdn_attn's names: stub the missing ones
        import os
        import re
        spec = importlib.util.find_spec("vllm.v1.attention.backends.gdn_attn")
        src = open(spec.origin).read()
        for mod in sorted(set(re.findall(r"^from (vllm[\w.]*) import", src, flags=re.M))):
            if mod in _StubFinder.REAL:
                continue
            m = importlib.import_module(mod)
            if not hasattr(m, "__getattr__"):
                m.__getattr__ = _stub_module(mod).__getattr__
        import vllm.v1.attention.backends.gdn_attn as ga
        return ga
    finally:
        sys.meta_path.remove(finder)


class _Stop(Exception):
    pass


def test_counter_hook_in_build(monkeypatch):
    ga = _import_gdn_attn()
    Z = ga.gdn_zero_draft
    assert Z.ZERO_DRAFT_AS_SPEC and Z.R4_COUNT
    monkeypatch.setattr(Z, "_C", {"steps": 0, "decode_rows": 0, "exposed_rows": 0, "dev": None, "last": None})
    monkeypatch.setattr(Z, "R4_COUNT_EVERY", 1)  # exercise the periodic report() from inside count()
    seen = []
    monkeypatch.setattr(Z, "report", lambda tag, _r=Z.report: (seen.append(tag), _r(tag)))
    b = object.__new__(ga.GDNAttentionMetadataBuilder)
    b.use_spec_decode = True

    def stop(*a, **k):
        raise _Stop()
    b._split_batch = stop
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = types.SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 4, 5, 6]), num_actual_tokens=6,
                              seq_lens_cpu_upper_bound=torch.tensor([90, 51, 60]))
    ndt = torch.tensor([3, -1, -1])
    acc = torch.tensor([2, 3, 1], dtype=torch.int32, device=dev)
    # 1. under capture: the hook returns before counting (no device work recorded)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(_Stop):
        ga.GDNAttentionMetadataBuilder.build(b, 0, m, num_accepted_tokens=acc, num_decode_draft_tokens_cpu=ndt)
    assert Z._C["steps"] == 0 and Z._C["dev"] is None
    # 2. normal build: counted before the split, periodic report reads the device counter
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(_Stop):
        ga.GDNAttentionMetadataBuilder.build(b, 0, m, num_accepted_tokens=acc, num_decode_draft_tokens_cpu=ndt)
    assert Z._C["steps"] == 1 and Z._C["decode_rows"] == 2 and Z._C["exposed_rows"] == 2
    assert int(Z._C["dev"].item()) == 1 and Z._C["dev"].device.type == acc.device.type
    assert seen == ["periodic"]
    # 3. spec decode off: no hook
    b.use_spec_decode = False
    with pytest.raises(_Stop):
        ga.GDNAttentionMetadataBuilder.build(b, 0, m, num_accepted_tokens=acc, num_decode_draft_tokens_cpu=ndt)
    assert Z._C["steps"] == 1
    # 4. exit report path
    Z.report("exit")
    assert seen[-1] == "exit"
