# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which layers maybe_use_lowm_bf16_gemm opts in (no GPU needed)."""

import sys

import pytest
import torch

from vllm.model_executor.kernels.linear import lowm_bf16_gemm as lowm
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


@pytest.fixture(autouse=True)
def _sm107(monkeypatch):
    monkeypatch.setattr(lowm.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        lowm.current_platform, "is_device_capability", lambda cap: cap == (10, 7)
    )
    lowm._tinygemm_available.cache_clear()


def _layer(n: int) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.quant_method = UnquantizedLinearMethod()
    layer.weight = torch.nn.Parameter(
        torch.zeros(n, 2048, dtype=torch.bfloat16), requires_grad=False
    )
    layer.bias = None
    return layer


def _engaged(layer: torch.nn.Module) -> bool:
    return layer.quant_method._gemm_impl is lowm._lowm_gemm


@pytest.mark.parametrize("n", [1, 64, 256])
def test_env_off_engages_nothing(monkeypatch, n):
    monkeypatch.setenv("VLLM_LOWM_BF16_GEMM", "0")
    monkeypatch.setattr(
        lowm, "_tinygemm_available", lambda device: pytest.fail("probed")
    )
    layer = _layer(n)
    assert not lowm.maybe_use_lowm_bf16_gemm(layer)
    assert not _engaged(layer)
    assert not hasattr(layer, "lowm_zero_bias")


def test_tinygemm_import_failure(monkeypatch):
    # A None entry makes `from flashinfer.gemm.routergemm import ...` raise.
    monkeypatch.setitem(sys.modules, "flashinfer.gemm.routergemm", None)
    for n in (64, 256):
        layer = _layer(n)
        assert not lowm.maybe_use_lowm_bf16_gemm(layer)
        assert not _engaged(layer)
    # The row-dot plan does not depend on FlashInfer.
    layer = _layer(1)
    assert lowm.maybe_use_lowm_bf16_gemm(layer)
    assert _engaged(layer)
