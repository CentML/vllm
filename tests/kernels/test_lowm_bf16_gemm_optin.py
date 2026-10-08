# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Architecture, shape, and layout eligibility boundaries (no GPU needed)."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from vllm.model_executor.kernels.linear import lowm_bf16_gemm as lowm
from vllm.platforms.interface import DeviceCapability


@pytest.fixture
def architecture(monkeypatch):
    monkeypatch.setattr(lowm.current_platform, "is_cuda", lambda: True)
    # SM100/SM103 eligibility cases opt in explicitly; SM107 ignores this flag.
    monkeypatch.setenv("VLLM_LOWM_BF16_GEMM", "1")
    monkeypatch.setenv("VLLM_LOWM_BF16_GEMM_SM100", "1")

    def select(major, minor):
        monkeypatch.setattr(
            lowm.current_platform,
            "get_device_capability",
            lambda device_id=0: DeviceCapability(major, minor),
        )

    return select


@pytest.fixture
def fake_mode():
    return FakeTensorMode()


def _tensor(
    shape,
    mode,
    *,
    dtype=torch.bfloat16,
    device="cuda:0",
    layout="contiguous",
):
    # Real meta storage/strides, with CUDA device metadata only. Constructing
    # these tensors neither allocates on CUDA nor launches a replacement GEMM.
    meta = torch.empty(shape, dtype=dtype, device="meta")
    if layout == "strided":
        meta = meta.t().contiguous().t()
    elif layout == "offset":
        meta = torch.empty(meta.numel() + 1, dtype=dtype, device="meta")[1:]
        meta = meta.view(shape)
    return FakeTensor(mode, meta, torch.device(device))


@pytest.mark.parametrize("minor", [0, 3, 7])
@pytest.mark.parametrize("n,k", [(16, 64), (32, 2048), (64, 128), (272, 2048)])
def test_aligned_shape_plan(architecture, minor, n, k):
    architecture(10, minor)
    assert lowm._plan(n, k, torch.device("cuda:0")) == (64, "tinygemm")


@pytest.mark.parametrize("capability", [(9, 0), (10, 1), (10, 2), (12, 0)])
def test_unsupported_architecture_has_no_plan(architecture, capability):
    architecture(*capability)
    assert lowm._plan(32, 2048, torch.device("cuda:0")) is None
    assert lowm._plan(256, 2048, torch.device("cuda:0")) is None
    assert lowm._plan(1, 2048, torch.device("cuda:0")) is None


@pytest.mark.parametrize("minor", [0, 3, 7])
@pytest.mark.parametrize("n,k", [(0, 64), (16, 0), (17, 64), (32, 65)])
def test_unaligned_shape_has_no_plan(architecture, minor, n, k):
    architecture(10, minor)
    assert lowm._plan(n, k, torch.device("cuda:0")) is None


def test_cpu_has_no_plan(architecture):
    architecture(10, 7)
    assert lowm._plan(32, 2048, torch.device("cpu")) is None


@pytest.mark.parametrize("minor", [0, 3])
def test_sm100_family_does_not_inherit_sm107_windows(architecture, minor):
    architecture(10, minor)
    assert lowm._plan(256, 2048, torch.device("cuda:0")) == (64, "tinygemm")
    assert lowm._plan(64, 2048, torch.device("cuda:0")) == (64, "tinygemm")
    assert lowm._plan(1, 2048, torch.device("cuda:0")) is None


@pytest.mark.parametrize("minor", [0, 3])
@pytest.mark.parametrize(
    "env",
    [
        {},
        {"VLLM_LOWM_BF16_GEMM_SM100": "0"},
        {"VLLM_LOWM_BF16_GEMM": "0", "VLLM_LOWM_BF16_GEMM_SM100": "1"},
    ],
)
def test_sm100_family_is_ineligible_without_opt_in(
    architecture, monkeypatch, minor, env
):
    architecture(10, minor)
    monkeypatch.delenv("VLLM_LOWM_BF16_GEMM", raising=False)
    monkeypatch.delenv("VLLM_LOWM_BF16_GEMM_SM100", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    for n in (256, 64, 32, 16):
        assert lowm._plan(n, 2048, torch.device("cuda:0")) is None


def test_sm107_does_not_require_sm100_opt_in(architecture, monkeypatch):
    architecture(10, 7)
    monkeypatch.delenv("VLLM_LOWM_BF16_GEMM_SM100", raising=False)
    assert lowm._plan(256, 2048, torch.device("cuda:0")) == (208, "tinygemm")
    assert lowm._plan(32, 2048, torch.device("cuda:0")) == (64, "tinygemm")


@pytest.mark.parametrize(
    "minor,n,k,bound,backend",
    [
        (0, 32, 2048, 64, "tinygemm"),
        (3, 32, 2048, 64, "tinygemm"),
        (7, 32, 2048, 64, "tinygemm"),
        (7, 256, 2048, 208, "tinygemm"),
        (7, 64, 2048, 384, "tinygemm"),
        (7, 1, 2048, 512, "rowdot"),
    ],
)
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_runtime_eligibility_at_m_boundary(
    architecture, fake_mode, minor, n, k, bound, backend, offset
):
    architecture(10, minor)
    m = bound + offset
    x = _tensor((m, k), fake_mode)
    w = _tensor((n, k), fake_mode)
    zb = _tensor((n,), fake_mode)
    expected = (bound, backend) if m <= bound else None
    assert lowm._runtime_plan(x, w, zb) == expected


@pytest.mark.parametrize(
    "invalid",
    [
        "empty",
        "input_stride",
        "weight_stride",
        "input_offset",
        "weight_offset",
        "bias_offset",
        "dtype",
        "device",
    ],
)
def test_invalid_input_metadata_is_ineligible(architecture, fake_mode, invalid):
    architecture(10, 3)
    x = _tensor((4, 2048), fake_mode)
    w = _tensor((32, 2048), fake_mode)
    zb = _tensor((32,), fake_mode)
    if invalid == "empty":
        x = _tensor((0, 2048), fake_mode)
    elif invalid == "input_stride":
        x = _tensor((4, 2048), fake_mode, layout="strided")
    elif invalid == "weight_stride":
        w = _tensor((32, 2048), fake_mode, layout="strided")
    elif invalid == "input_offset":
        x = _tensor((4, 2048), fake_mode, layout="offset")
    elif invalid == "weight_offset":
        w = _tensor((32, 2048), fake_mode, layout="offset")
    elif invalid == "bias_offset":
        zb = _tensor((32,), fake_mode, layout="offset")
    elif invalid == "dtype":
        x = _tensor((4, 2048), fake_mode, dtype=torch.float16)
    else:
        x = _tensor((4, 2048), fake_mode, device="cuda:1")
    assert lowm._runtime_plan(x, w, zb) is None


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "stride", "offset"])
def test_invalid_output_metadata_is_declined(architecture, fake_mode, invalid):
    architecture(10, 3)
    x = _tensor((4, 2048), fake_mode)
    w = _tensor((32, 2048), fake_mode)
    zb = _tensor((32,), fake_mode)
    out = _tensor((4, 32), fake_mode)
    if invalid == "shape":
        out = _tensor((4, 64), fake_mode)
    elif invalid == "dtype":
        out = _tensor((4, 32), fake_mode, dtype=torch.float16)
    elif invalid == "device":
        out = _tensor((4, 32), fake_mode, device="cuda:1")
    elif invalid == "stride":
        out = _tensor((4, 32), fake_mode, layout="strided")
    else:
        out = _tensor((4, 32), fake_mode, layout="offset")
    assert not lowm.lowm_bf16_gemm_out(
        SimpleNamespace(weight=w, lowm_zero_bias=zb), x, out
    )


@pytest.mark.parametrize("use_pdl", [False, True])
def test_tinygemm_import_failure_keeps_baseline(monkeypatch, use_pdl):
    monkeypatch.setitem(sys.modules, "flashinfer.gemm.routergemm", None)
    lowm._tinygemm_available.cache_clear()
    try:
        assert not lowm._tinygemm_available(torch.device("cpu"), use_pdl)
    finally:
        lowm._tinygemm_available.cache_clear()
