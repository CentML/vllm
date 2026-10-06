# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks of the sm_107 K=64 MXFP8 tactic encoding and gating."""

import sys
import types

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils import flashinfer_mxfp8_k64 as k64

DEFAULT_TACTICS = "512,256,256,2;256,256,256,2;256,128,256,2"


def test_tactic_encoding_matches_pinned_autotune_entries():
    tactics = k64.parse_tactics(DEFAULT_TACTICS)
    assert tactics == [(512, 256, 256, 2), (256, 256, 256, 2), (256, 128, 256, 2)]
    encoded = [k64.encode_tactic(t) for t in tactics]
    assert encoded[0] == (107, 512, 256, 256, 2)
    assert all(k64.is_k64_tactic(t) for t in encoded)
    # JSON round trip through the autotune cache turns tuples into lists.
    assert k64.is_k64_tactic([107, 512, 256, 256, 2])
    assert not k64.is_k64_tactic(-1)
    assert not k64.is_k64_tactic((100, 512, 256, 256, 2))
    assert not k64.is_k64_tactic((107, 512, 256, 256))
    assert k64.tactic_shapes(encoded[2]) == ((256, 128, 128), (256, 128, 64), (2, 1))
    assert k64.parse_tactics("") == []


def test_offer_m():
    assert k64.offer_m(256, 256)
    assert not k64.offer_m(224, 256)
    assert not k64.offer_m(288 + 1, 256)
    assert k64.offer_m(4096, 256)


def test_env_defaults_and_disable(monkeypatch):
    for name in (
        "VLLM_FLASHINFER_MXFP8_K64",
        "VLLM_FLASHINFER_MXFP8_K64_TACTICS",
        "VLLM_FLASHINFER_MXFP8_K64_MIN_M",
    ):
        monkeypatch.delenv(name, raising=False)
    ev = envs.environment_variables
    assert ev["VLLM_FLASHINFER_MXFP8_K64"]() is True
    assert ev["VLLM_FLASHINFER_MXFP8_K64_TACTICS"]() == DEFAULT_TACTICS
    assert ev["VLLM_FLASHINFER_MXFP8_K64_MIN_M"]() == 256
    for off in ("0", "false", "off"):
        monkeypatch.setenv("VLLM_FLASHINFER_MXFP8_K64", off)
        assert ev["VLLM_FLASHINFER_MXFP8_K64"]() is False
    monkeypatch.setenv("VLLM_FLASHINFER_MXFP8_K64", "1")
    assert ev["VLLM_FLASHINFER_MXFP8_K64"]() is True


class _Tensor:
    def __init__(self, *shape, contiguous=True):
        self.shape = shape
        self._c = contiguous
        self.device = types.SimpleNamespace(index=0)
        self.dtype = torch.bfloat16

    def is_contiguous(self):
        return self._c


class _BaseRunner:
    def get_valid_tactics(self, inputs, profile):
        return [-1, 0]

    def forward(self, inputs, tactic=None, do_preparation=False, **kwargs):
        return ("base", tactic)


def _fake_gb():
    gb = types.SimpleNamespace(calls=0)

    def factory(sm_major, sm_minor, enable_pdl, out_dtype):
        gb.calls += 1
        return _BaseRunner()

    gb._cute_dsl_gemm_mxfp8_runner = factory
    return gb


def _inputs(m, k=2048, n=4096, out_contiguous=True):
    a, b = _Tensor(m, k), _Tensor(k, n)
    out = _Tensor(m, n, contiguous=out_contiguous)
    return [a, b, None, None, None, out, None]


def test_factory_gating_and_tactics(monkeypatch):
    monkeypatch.setattr(k64, "_can_implement", lambda t, m, n, k: t[1] != 512)
    monkeypatch.setattr(k64, "_get_compiled", lambda *a: None)
    gb = _fake_gb()
    k64.install(gb, k64.parse_tactics(DEFAULT_TACTICS), 256)
    f = gb._cute_dsl_gemm_mxfp8_runner

    # Other architectures / output dtypes keep the stock runner.
    assert type(f(10, 3, True, torch.bfloat16)) is _BaseRunner
    assert type(f(10, 0, True, torch.bfloat16)) is _BaseRunner
    assert type(f(10, 7, True, torch.float32)) is _BaseRunner

    r = f(10, 7, True, torch.bfloat16)
    assert type(r) is not _BaseRunner and isinstance(r, _BaseRunner)
    assert f(10, 7, True, torch.bfloat16) is r  # built once per key
    assert r.get_valid_tactics(_inputs(512), None) == [
        -1,
        0,
        (107, 256, 256, 256, 2),
        (107, 256, 128, 256, 2),
    ]
    assert r.get_valid_tactics(_inputs(224), None) == [-1, 0]  # M < MIN_M
    assert r.get_valid_tactics(_inputs(272), None) == [-1, 0]  # M % 32 != 0
    assert r.get_valid_tactics(_inputs(512, out_contiguous=False), None) == [-1, 0]

    # Not compiled (e.g. during capture): stock default tactic.
    assert r.forward(_inputs(512), tactic=[107, 256, 256, 256, 2]) == ("base", -1)
    # Stock tactics pass through unchanged.
    assert r.forward(_inputs(512), tactic=0) == ("base", 0)


def test_maybe_install_gating(monkeypatch):
    from vllm.platforms import current_platform

    gb = _fake_gb()
    pkg = types.ModuleType("flashinfer.gemm")
    pkg.gemm_base = gb
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", pkg)
    monkeypatch.setitem(sys.modules, "flashinfer.gemm.gemm_base", gb)
    monkeypatch.setattr(
        k64, "_STATE", {"installed": False, "tactics": [], "min_m": 256}
    )
    stock = gb._cute_dsl_gemm_mxfp8_runner

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True, raising=False)
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda cap, device_id=0: cap == 103
    )
    monkeypatch.setenv("VLLM_FLASHINFER_MXFP8_K64", "1")
    assert k64.maybe_install() is False  # not sm_107
    assert gb._cute_dsl_gemm_mxfp8_runner is stock

    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda cap, device_id=0: cap == 107
    )
    monkeypatch.setenv("VLLM_FLASHINFER_MXFP8_K64", "0")
    assert k64.maybe_install() is False  # disabled
    assert gb._cute_dsl_gemm_mxfp8_runner is stock

    monkeypatch.setenv("VLLM_FLASHINFER_MXFP8_K64", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MXFP8_K64_MIN_M", "512")
    assert k64.maybe_install() is True
    assert gb._cute_dsl_gemm_mxfp8_runner is not stock
    assert k64._STATE["min_m"] == 512
    wrapped = gb._cute_dsl_gemm_mxfp8_runner
    assert k64.maybe_install() is True  # idempotent
    assert gb._cute_dsl_gemm_mxfp8_runner is wrapped


@pytest.mark.parametrize("cap", [100, 103])
def test_no_effect_off_sm107(monkeypatch, cap):
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        k64, "_STATE", {"installed": False, "tactics": [], "min_m": 256}
    )
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True, raising=False)
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda c, device_id=0: c == cap
    )
    monkeypatch.delenv("VLLM_FLASHINFER_MXFP8_K64", raising=False)
    assert k64.maybe_install() is False
