# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.parameter as parameter
from vllm.model_executor.kernels.linear import lowm_bf16_gemm as lowm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.platforms.interface import DeviceCapability

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)


@pytest.fixture(autouse=True)
def _sm100_opt_in(monkeypatch):
    # TinyGEMM2 on SM100/SM103 is opt-in; exercise it there as well as on SM107.
    monkeypatch.setenv("VLLM_LOWM_BF16_GEMM_SM100", "1")


def _ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x.float() @ w.float().t()).to(torch.bfloat16)


def _requires_plan(n: int, k: int = 2048):
    plan = lowm._plan(
        n, k, torch.device("cuda", torch.accelerator.current_device_index())
    )
    if plan is None:
        pytest.skip("No low-M backend for this architecture and shape")
    if plan[1] == "tinygemm":
        pytest.importorskip("flashinfer.gemm.routergemm")
    return plan


def _inputs(m: int, n: int, k: int):
    torch.manual_seed(m * 7 + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    return x, w, torch.zeros(n, device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("m", [1, 3, 8, 16, 200, 512])
@pytest.mark.parametrize("provided_out", [False, True])
def test_rowdot_matches_fp32_reference(m: int, provided_out: bool):
    # Test the portable Triton kernel on every CUDA architecture, independently
    # of the deliberately narrower production SM107 dispatch plan.
    x, w, _ = _inputs(m, 1, 2048)
    destination = (
        torch.full((m, 1), float("nan"), device=x.device, dtype=x.dtype)
        if provided_out
        else None
    )
    out = lowm._rowdot(x, w, destination)
    assert out.shape == (m, 1)
    torch.testing.assert_close(out, _ref(x, w), rtol=1e-2, atol=1e-2)
    if destination is not None:
        torch.testing.assert_close(destination, _ref(x, w), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize("n,k", [(16, 64), (32, 2048), (64, 128), (272, 2048)])
@pytest.mark.parametrize("m", [1, 64])
def test_generic_tinygemm_matches_fp32_reference(monkeypatch, use_pdl, n, k, m):
    _requires_plan(n, k)
    monkeypatch.setenv("VLLM_LOWM_BF16_GEMM_PDL", str(int(use_pdl)))
    x, w, zb = _inputs(m, n, k)
    out = lowm.lowm_bf16_gemm_impl(x, w, zb)
    torch.testing.assert_close(out, _ref(x, w), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("n", [256, 64, 1])
def test_measured_plan_boundary_matches_fp32_reference(n):
    max_m, _ = _requires_plan(n)
    x, w, zb = _inputs(max_m, n, 2048)
    out = lowm.lowm_bf16_gemm_impl(x, w, zb)
    torch.testing.assert_close(out, _ref(x, w), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "n,k", [(32, 2048), (272, 128), (256, 2048), (64, 2048), (1, 2048)]
)
def test_above_plan_bound_is_cublas(n, k):
    max_m, _ = _requires_plan(n, k)
    x, w, zb = _inputs(max_m + 1, n, k)
    assert torch.equal(lowm.lowm_bf16_gemm_impl(x, w, zb), F.linear(x, w))


@pytest.mark.parametrize("n,k", [(17, 2048), (32, 65), (32, 2048)])
def test_unsupported_shapes_or_architectures_keep_baseline(n, k):
    x, w, zb = _inputs(4, n, k)
    if lowm._plan(n, k, w.device) is not None:
        pytest.skip("Shape supported on this architecture")
    assert torch.equal(lowm.lowm_bf16_gemm_impl(x, w, zb), F.linear(x, w))


@pytest.mark.parametrize(
    "layout", ["input_strided", "weight_strided", "input_offset", "weight_offset"]
)
def test_unsupported_layout_keeps_baseline(layout):
    x, w, zb = _inputs(4, 32, 2048)
    if layout == "input_strided":
        x = x.t().contiguous().t()
    elif layout == "weight_strided":
        w = w.t().contiguous().t()
    elif layout == "input_offset":
        backing = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)
        shifted = backing[1:].view_as(x)
        shifted.copy_(x)
        x = shifted
    else:
        backing = torch.empty(w.numel() + 1, device=w.device, dtype=w.dtype)
        shifted = backing[1:].view_as(w)
        shifted.copy_(w)
        w = shifted
    assert torch.equal(lowm.lowm_bf16_gemm_impl(x, w, zb), F.linear(x, w))


def test_multidimensional_input_matches_reference():
    _requires_plan(32)
    x, w, zb = _inputs(6, 32, 2048)
    x = x.view(2, 3, 2048)
    out = lowm.lowm_bf16_gemm_impl(x, w, zb)
    assert out.shape == (2, 3, 32)
    torch.testing.assert_close(out, _ref(x, w), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("use_pdl", [False, True])
def test_cuda_graph_replay_uses_new_inputs(monkeypatch, use_pdl):
    _requires_plan(32)
    monkeypatch.setenv("VLLM_LOWM_BF16_GEMM_PDL", str(int(use_pdl)))
    x, w, zb = _inputs(8, 32, 2048)
    static_x = x.clone()
    torch.ops.vllm.lowm_bf16_gemm(static_x, w, zb)
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = torch.ops.vllm.lowm_bf16_gemm(static_x, w, zb)
    for seed in range(3):
        torch.manual_seed(100 + seed)
        new_x = torch.randn_like(x)
        static_x.copy_(new_x)
        g.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(static_out, _ref(new_x, w), rtol=1e-2, atol=1e-2)


def test_rowdot_cuda_graph_replay_uses_new_inputs():
    x, w, _ = _inputs(8, 1, 2048)
    static_x = x.clone()
    lowm._rowdot(static_x, w)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = lowm._rowdot(static_x, w)
    for seed in range(3):
        torch.manual_seed(200 + seed)
        new_x = torch.randn_like(x)
        static_x.copy_(new_x)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(static_out, _ref(new_x, w), rtol=1e-2, atol=1e-2)


def _opted_in_layer(w: torch.Tensor, zb: torch.Tensor) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.register_buffer("lowm_zero_bias", zb, persistent=False)
    return layer


@pytest.mark.parametrize("n,k", [(1, 2048), (32, 2048), (64, 2048), (272, 128)])
def test_out_variant_matches_custom_op(n, k):
    max_m, _ = _requires_plan(n, k)
    for m in (1, max_m):
        x, w, zb = _inputs(m, n, k)
        out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        assert lowm.lowm_bf16_gemm_out(_opted_in_layer(w, zb), x, out)
        assert torch.equal(out, lowm.lowm_bf16_gemm_impl(x, w, zb))


@pytest.mark.parametrize("n,k", [(1, 2048), (32, 2048), (64, 2048)])
def test_out_variant_declines_above_plan_bound(n, k):
    max_m, _ = _requires_plan(n, k)
    x, w, zb = _inputs(max_m + 1, n, k)
    out = torch.zeros(max_m + 1, n, device="cuda", dtype=torch.bfloat16)
    assert not lowm.lowm_bf16_gemm_out(_opted_in_layer(w, zb), x, out)
    assert not out.any()


@pytest.mark.parametrize("m", [0, 4])
def test_out_variant_declines_layer_without_plan(m):
    x, w, _ = _inputs(m, 64, 2048)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    out = torch.zeros(m, 64, device="cuda", dtype=torch.bfloat16)
    assert not lowm.lowm_bf16_gemm_out(layer, x, out)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "stride", "offset"])
def test_declined_destination_is_untouched(invalid):
    _requires_plan(32)
    x, w, zb = _inputs(4, 32, 2048)
    if invalid == "shape":
        out = torch.full((4, 64), 7, device="cuda", dtype=torch.bfloat16)
    elif invalid == "dtype":
        out = torch.full((4, 32), 7, device="cuda", dtype=torch.float16)
    elif invalid == "device":
        out = torch.full((4, 32), 7, device="cpu", dtype=torch.bfloat16)
    elif invalid == "stride":
        out = torch.full((32, 4), 7, device="cuda", dtype=torch.bfloat16).t()
    else:
        out = torch.full((129,), 7, device="cuda", dtype=torch.bfloat16)
        out = out[1:].view(4, 32)
    before = out.clone()
    assert not lowm.lowm_bf16_gemm_out(_opted_in_layer(w, zb), x, out)
    assert torch.equal(out, before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_non_bf16_inputs_keep_baseline(dtype):
    x, w, zb = _inputs(4, 32, 2048)
    x, w, zb = x.to(dtype), w.to(dtype), zb.to(dtype)
    assert torch.equal(lowm.lowm_bf16_gemm_impl(x, w, zb), F.linear(x, w))


@pytest.mark.parametrize("n", [256, 32, 1])
@pytest.mark.parametrize("enabled", [False, True])
def test_replicated_linear_routes_eligible_projections(monkeypatch, n, enabled):
    monkeypatch.setenv("VLLM_LOWM_BF16_GEMM", str(int(enabled)))
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    with torch.device("cuda"):
        layer = ReplicatedLinear(
            2048, n, bias=False, params_dtype=torch.bfloat16, disable_tp=True
        )
    x, w, _ = _inputs(4, n, 2048)
    layer.weight.data.copy_(w)
    plan = lowm._plan(n, 2048, w.device)
    if enabled and plan is not None and plan[1] == "tinygemm":
        pytest.importorskip("flashinfer.gemm.routergemm")
    assert lowm.maybe_use_lowm_bf16_gemm(layer) == (enabled and plan is not None)
    tiny = Mock(wraps=lowm._tinygemm)
    rowdot = Mock(wraps=lowm._rowdot)
    monkeypatch.setattr(lowm, "_tinygemm", tiny)
    monkeypatch.setattr(lowm, "_rowdot", rowdot)
    out, bias = layer(x)
    assert bias is None
    torch.testing.assert_close(out, _ref(x, w), rtol=1e-2, atol=1e-2)
    if enabled and plan is not None:
        selected, unused = (tiny, rowdot) if plan[1] == "tinygemm" else (rowdot, tiny)
        selected.assert_called_once()
        unused.assert_not_called()
    else:
        tiny.assert_not_called()
        rowdot.assert_not_called()
        assert torch.equal(out, F.linear(x, w))


@pytest.mark.parametrize("n", [256, 1])
def test_replicated_linear_missing_flashinfer_preserves_consumers(monkeypatch, n):
    monkeypatch.setenv("VLLM_LOWM_BF16_GEMM", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setitem(sys.modules, "flashinfer.gemm.routergemm", None)
    # TinyGEMM must never launch on the test GPU under this capability fixture:
    # its genuine import failure leaves the router on F.linear. The shared gate
    # uses the architecture-independent Triton row-dot kernel.
    monkeypatch.setattr(
        lowm.current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(10, 7),
    )
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    with torch.device("cuda"):
        layer = ReplicatedLinear(
            2048, n, bias=False, params_dtype=torch.bfloat16, disable_tp=True
        )
    x, w, _ = _inputs(4, n, 2048)
    layer.weight.data.copy_(w)
    lowm._tinygemm_available.cache_clear()
    try:
        assert lowm.maybe_use_lowm_bf16_gemm(layer) == (n == 1)
        out, bias = layer(x)
        assert bias is None
        expected = lowm._rowdot(x, w) if n == 1 else F.linear(x, w)
        assert torch.equal(out, expected)
        torch.testing.assert_close(out, _ref(x, w), rtol=1e-2, atol=1e-2)
    finally:
        lowm._tinygemm_available.cache_clear()
