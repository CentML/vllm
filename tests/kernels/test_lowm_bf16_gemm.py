# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear import lowm_bf16_gemm as lowm

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)


def _ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x.float() @ w.float().t()).to(torch.bfloat16)


def _requires_tinygemm():
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("tinygemm2_sm100 plan is only used on SM10x")
    pytest.importorskip("flashinfer.gemm.routergemm")


def _inputs(m: int, n: int, k: int):
    torch.manual_seed(m * 7 + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    return x, w, torch.zeros(n, device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("m", [1, 3, 8, 16, 200, 512])
def test_rowdot_matches_fp32_reference(m: int):
    x, w, zb = _inputs(m, 1, 2048)
    out = lowm.lowm_bf16_gemm_impl(x, w, zb)
    assert out.shape == (m, 1)
    # fp32 accumulation: at most one bf16 rounding step from the exact result.
    torch.testing.assert_close(out, _ref(x, w), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("n,m", [(256, 1), (256, 16), (256, 208), (64, 4), (64, 384)])
def test_tinygemm_plan_matches_fp32_reference(n: int, m: int):
    _requires_tinygemm()
    x, w, zb = _inputs(m, n, 2048)
    out = lowm.lowm_bf16_gemm_impl(x, w, zb)
    torch.testing.assert_close(out, _ref(x, w), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("n,m", [(256, 209), (64, 385), (1, 513)])
def test_above_plan_bound_is_cublas(n: int, m: int):
    # Larger batches (prefill / mixed) keep the cuBLAS result bit for bit.
    x, w, zb = _inputs(m, n, 2048)
    assert torch.equal(lowm.lowm_bf16_gemm_impl(x, w, zb), F.linear(x, w))


def test_cuda_graph_replay_uses_new_inputs():
    n = 1
    if torch.cuda.get_device_capability()[0] == 10:
        n = 256
    m = 8
    x, w, zb = _inputs(m, n, 2048)
    static_x = x.clone()
    torch.ops.vllm.lowm_bf16_gemm(static_x, w, zb)  # compile/JIT outside capture
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


def _opted_in_layer(w: torch.Tensor, zb: torch.Tensor) -> torch.nn.Module:
    # What maybe_use_lowm_bf16_gemm leaves on the layer (it opts in on SM107 only).
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.register_buffer("lowm_zero_bias", zb, persistent=False)
    return layer


@pytest.mark.parametrize("n,m", [(1, 1), (1, 512), (64, 4), (64, 384)])
def test_out_variant_matches_custom_op(n: int, m: int):
    # GDN's aux-stream in_proj_ba fork writes through lowm_bf16_gemm_out; it must
    # produce exactly what the layer's own (custom-op) path produces.
    if n != 1:
        _requires_tinygemm()
    x, w, zb = _inputs(m, n, 2048)
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    assert lowm.lowm_bf16_gemm_out(_opted_in_layer(w, zb), x, out)
    assert torch.equal(out, lowm.lowm_bf16_gemm_impl(x, w, zb))


@pytest.mark.parametrize("n,m", [(1, 513), (64, 385)])
def test_out_variant_declines_above_plan_bound(n: int, m: int):
    x, w, zb = _inputs(m, n, 2048)
    out = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    assert not lowm.lowm_bf16_gemm_out(_opted_in_layer(w, zb), x, out)
    assert not out.any()


def test_out_variant_declines_layer_without_plan():
    x, w, _ = _inputs(4, 64, 2048)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    out = torch.zeros(4, 64, device="cuda", dtype=torch.bfloat16)
    assert not lowm.lowm_bf16_gemm_out(layer, x, out)
