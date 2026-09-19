# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arithmetic and lifetime contracts for the opt-in native SM107 linear path."""

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8 import flashinfer_sm107
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslSm107Mxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutedsl

if not (
    current_platform.is_cuda()
    and current_platform.is_device_capability(107)
    and has_flashinfer_cutedsl()
):
    pytest.skip("Requires SM107 and FlashInfer CuTe DSL", allow_module_level=True)


@pytest.fixture(autouse=True)
def _fp32_reference_math():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def _scale_offsets(rows, groups):
    """F8_128x4: [row//128, group//4, row%32, row%128//32, group%4]."""
    r, c = rows[:, None], torch.arange(groups, device=rows.device)[None, :]
    padded_groups = (groups + 3) // 4 * 4
    return (
        (r // 128) * (128 * padded_groups)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + c % 4
    )


def _dequantize(q, sf, *, swizzled):
    """Index scales independently of the production swizzle and GEMM helpers."""
    m, k = q.shape
    if swizzled:
        sf = sf.reshape(-1)[_scale_offsets(torch.arange(m, device=q.device), k // 32)]
    else:
        sf = sf.reshape(m, k // 32)
    scales = torch.exp2(sf.float() - 127)
    return q.float() * scales.repeat_interleave(32, dim=1)


def _make_case(m, n, k, seed=1309):
    torch.manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    # Different K-group magnitudes expose transposed or misplaced scale groups.
    gain = torch.exp2(torch.arange(k // 32, device="cuda") % 7 - 3)
    w = (w.float() * gain.repeat_interleave(32)).to(torch.bfloat16)
    wq, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    w_ref = _dequantize(wq, ws, swizzled=False)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(wq, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(ws, requires_grad=False)
    kernel = FlashInferCutedslSm107Mxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    return x, layer, kernel, w_ref


def _activation(x, kernel):
    q, sf = mxfp8_e4m3_quantize(x.reshape(-1, x.shape[-1]), True)
    return QuantizedActivation(q, sf, x.dtype, x.shape, kernel.input_quant_key())


def _reference(x, w_ref):
    q, sf = mxfp8_e4m3_quantize(x.reshape(-1, x.shape[-1]), True)
    return (_dequantize(q, sf, swizzled=True) @ w_ref.t()).to(x.dtype)


def _assert_arithmetic(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    diff = actual.float() - expected.float()
    relative_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(
        expected.float()
    ).clamp_min(1e-12)
    assert relative_l2.item() < 5e-4
    # Global L2 alone could overlook corruption of the final logical row.
    torch.testing.assert_close(
        actual.reshape(-1, actual.shape[-1])[-1],
        expected.reshape(-1, expected.shape[-1])[-1],
        rtol=0.01,
        atol=0.01,
    )


@pytest.mark.parametrize("shape", [(35, 320, 256), (160, 256, 256)])
@pytest.mark.parametrize("prequantized", [False, True])
@torch.inference_mode()
def test_sm107_mxfp8_bias_and_3d_shape_use_native(shape, prequantized, monkeypatch):
    """Catch silent compatibility routing and changes to the linear I/O contract."""
    m, n, k = shape
    x, layer, kernel, w_ref = _make_case(m, n, k)
    x = x.reshape(5, m // 5, k)
    bias = torch.linspace(-1, 1, n, device="cuda", dtype=torch.bfloat16)
    expected = (_reference(x, w_ref) + bias).reshape(5, m // 5, n)
    get_kernel = flashinfer_sm107._get_kernel
    native_calls = []

    def track_native(*args, **kwargs):
        native_calls.append(True)
        return get_kernel(*args, **kwargs)

    monkeypatch.setattr(flashinfer_sm107, "_get_kernel", track_native)
    activation = _activation(x, kernel) if prequantized else x
    actual = kernel.apply_weights(layer, activation, bias)
    _assert_arithmetic(actual, expected)
    assert native_calls, "The opt-in path did not obtain a native SM107 kernel"


@torch.inference_mode()
def test_sm107_mxfp8_ignores_poisoned_tail_scales_without_mutating_inputs():
    """Padded E8M0 NaNs must not leak into output or alter caller-owned scales."""
    x, layer, kernel, w_ref = _make_case(35, 320, 256)
    qa = _activation(x, kernel)
    tail = torch.arange(35, 128, device="cuda")
    qa.scale[_scale_offsets(tail, 8)] = 255
    inputs = [qa.data, qa.scale, layer.weight, layer.weight_scale]
    before = [t.contiguous().view(torch.uint8).clone() for t in inputs]
    expected = _reference(x, w_ref)
    _assert_arithmetic(kernel.apply_weights(layer, qa), expected)
    for actual, original in zip(inputs, before):
        torch.testing.assert_close(
            actual.contiguous().view(torch.uint8), original, rtol=0, atol=0
        )


@torch.inference_mode()
def test_sm107_mxfp8_cached_kernel_uses_fresh_operand_pointers():
    """Compiled kernels must not retain the first layer's input/weight buffers."""
    x, layer, kernel, w_ref = _make_case(35, 256, 256)
    first = kernel.apply_weights(layer, x)
    _assert_arithmetic(first, _reference(x, w_ref))
    fresh_x, fresh_layer, fresh_kernel, fresh_ref = _make_case(35, 256, 256, seed=1310)
    _assert_arithmetic(
        fresh_kernel.apply_weights(fresh_layer, fresh_x), _reference(fresh_x, fresh_ref)
    )
    _assert_arithmetic(first, _reference(x, w_ref))


@pytest.mark.parametrize("prequantized", [False, True])
@torch.inference_mode()
def test_sm107_mxfp8_graph_replay_reads_changed_input(prequantized):
    """An M=160 warmup must support capture at M=35 and changed-input replay."""
    flashinfer_sm107._KERNELS.clear()
    x, layer, kernel, w_ref = _make_case(160, 320, 256)
    kernel.apply_weights(layer, x)
    x = x[:35].clone()
    qa = _activation(x, kernel)  # Warm quantization, but not native GEMM, at M=35.
    activation = qa if prequantized else x
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = kernel.apply_weights(layer, activation)

    # Change both values and quantization scales while keeping capture pointers.
    x.mul_(-2)
    if prequantized:
        changed = _activation(x, kernel)
        activation.data.copy_(changed.data)
        activation.scale.copy_(changed.scale)
    output.fill_(float("nan"))
    graph.replay()
    torch.accelerator.synchronize()
    _assert_arithmetic(output, _reference(x, w_ref))
    torch.testing.assert_close(
        output, kernel.apply_weights(layer, activation), rtol=0, atol=0
    )


@torch.inference_mode()
def test_sm107_mxfp8_k_not_multiple_of_128_uses_compatibility(monkeypatch):
    """A valid MXFP8 K=160 must fall back instead of entering native K=128 tiles."""
    x, layer, kernel, w_ref = _make_case(35, 256, 160)

    def unexpected_native(*args, **kwargs):
        pytest.fail("K=160 must use the compatibility implementation")

    monkeypatch.setattr(flashinfer_sm107, "_get_kernel", unexpected_native)
    _assert_arithmetic(kernel.apply_weights(layer, x), _reference(x, w_ref))
