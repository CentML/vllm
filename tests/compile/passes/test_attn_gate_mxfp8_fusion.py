# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AttnGateMxfp8QuantFusionPass on Inductor graphs of the Qwen3.5/3.6 gate."""

import pytest
import torch

import vllm.config
import vllm.model_executor.layers.quantization.utils.mxfp8_utils  # noqa: F401
from tests.compile.backend import TestBackend
from vllm.compilation.passes.fusion.attn_gate_mxfp8_fusion import (
    AttnGateMxfp8QuantFusionPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import CompilationConfig, CompilationMode, VllmConfig
from vllm.model_executor.layers.fusion.attn_gate_mxfp8_quant import (
    attn_gate_mxfp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.has_device_capability(100)
        and has_flashinfer()
    ),
    reason="FlashInfer MXFP8 quantize needs SM100+",
)

HEADS, HEAD_DIM, M = 16, 256, 300
HIDDEN = HEADS * HEAD_DIM
FUSED = torch.ops.vllm.attn_gate_mxfp8_quant.default
QUANT = torch.ops.vllm.mxfp8_quantize.default


def _quant(y: torch.Tensor):
    return torch.ops.vllm.mxfp8_quantize(y.view(-1, HIDDEN), True, 0)


def gate_copy(attn, gate):
    """Contiguous [T, K] gate (the gate copied by the qk-norm-RoPE kernel)."""
    return _quant(attn * torch.sigmoid(gate))


def gate_view_in_graph(attn, q_gate):
    """Strided [T, H, D] gate read in place from the QKV output."""
    gate = q_gate.view(-1, HEADS, 2, HEAD_DIM)[:, :, 1]
    return _quant((attn.view(gate.shape) * torch.sigmoid(gate)).flatten(1))


def product_reused(attn, gate):
    """The bf16 product has another reader: keep the chain."""
    y = attn * torch.sigmoid(gate)
    return (*_quant(y), y)


def gate_from_pointwise(attn, gate):
    """Inductor would fuse the gate's producer into the gate kernel: keep."""
    return _quant(attn * torch.sigmoid(gate * 2.0))


@pytest.fixture
def vllm_config():
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    with vllm.config.set_current_vllm_config(config):
        torch.manual_seed(0)
        yield config


def _compile(config, fn, *inputs):
    fusion = AttnGateMxfp8QuantFusionPass(config)
    backend = TestBackend(NoOpEliminationPass(config), fusion, PostCleanupPass(config))
    for t in inputs:
        torch._dynamo.mark_dynamic(t, 0)
    out = torch.compile(fn, backend=backend, fullgraph=True)(*inputs)
    return fusion, backend, out


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attn = torch.randn(M, HIDDEN, device="cuda").bfloat16()
    # The q/gate half of a QKV output with k and v columns after it.
    qkv = (torch.randn(M, 2 * HIDDEN + 1024, device="cuda") * 3).bfloat16()
    q_gate = qkv[:, : 2 * HIDDEN]
    gate = q_gate.view(M, HEADS, 2, HEAD_DIM)[:, :, 1]
    return attn, q_gate, gate


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_attn_gate_fused(vllm_config, strided: bool) -> None:
    attn, q_gate, gate = _inputs()
    if strided:
        fusion, backend, out = _compile(vllm_config, gate_view_in_graph, attn, q_gate)
        attn_ref = attn.view(gate.shape)
    else:
        gate = gate.reshape(M, HIDDEN).contiguous()
        fusion, backend, out = _compile(vllm_config, gate_copy, attn, gate)
        attn_ref = attn

    assert fusion.matched_count == 1
    assert backend.op_count(QUANT) == 0
    assert backend.op_count(FUSED) == 1
    q_k, s_k = attn_gate_mxfp8_quant(attn_ref, gate)
    assert _bitwise(out[0], q_k) and _bitwise(out[1], s_k)


@pytest.mark.parametrize("fn", [product_reused, gate_from_pointwise])
@torch.inference_mode()
def test_attn_gate_not_fused(vllm_config, fn) -> None:
    attn, _, gate = _inputs()
    fusion, backend, _ = _compile(
        vllm_config, fn, attn, gate.reshape(M, HIDDEN).contiguous()
    )

    assert fusion.matched_count == 0
    assert backend.op_count(QUANT) == 1
    assert backend.op_count(FUSED) == 0
