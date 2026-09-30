# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RMSNormMxfp8QuantFusionPass on Inductor graphs of Qwen3.5/3.6 input norms."""

import pytest
import torch

import vllm.config
import vllm.ir.ops
import vllm.model_executor.layers.quantization.utils.mxfp8_utils  # noqa: F401
from tests.compile.backend import TestBackend
from vllm.compilation.passes.fusion.rms_norm_mxfp8_fusion import (
    RMSNormMxfp8QuantFusionPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import CompilationConfig, CompilationMode, VllmConfig
from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
    add_rms_norm_mxfp8_quant,
)
from vllm.model_executor.layers.fusion.shared_expert_gate import (
    apply_shared_expert_gate,
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

HIDDEN, EPS, M = 2048, 1e-6, 300
FUSED = torch.ops.vllm.add_rms_norm_mxfp8_quant.default
QUANT = torch.ops.vllm.mxfp8_quantize.default


def _quant(y: torch.Tensor):
    return torch.ops.vllm.mxfp8_quantize(y.view(-1, HIDDEN), True, 0)


class GemmaAddNorm(torch.nn.Module):
    """MoE combine -> input norm -> MXFP8 quant, optionally a bf16 consumer."""

    def __init__(self, bf16_consumer: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(HIDDEN, device="cuda").bfloat16())
        self.wba = torch.nn.Parameter(
            (torch.randn(64, HIDDEN, device="cuda") * 0.05).bfloat16()
        )
        self.bf16_consumer = bf16_consumer

    def forward(self, a, b, residual):
        x = (a + b).view(-1, HIDDEN)
        weight = self.weight.float() + 1.0
        y, residual = vllm.ir.ops.fused_add_rms_norm(x, residual, weight, EPS)
        q, s = _quant(y)
        if self.bf16_consumer:  # GDN in_proj_ba reads the normed output
            return q, s, residual, torch.mm(y, self.wba.t())
        return q, s, residual


class PlainNorm(torch.nn.Module):
    """rms_norm of an input with a plain weight (fp32 or bf16)."""

    def __init__(self, dtype: torch.dtype, embed: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(HIDDEN, device="cuda").to(dtype))
        self.embed = embed

    def forward(self, x, emb):
        if self.embed:  # Inductor fuses the gather into the norm
            x = torch.nn.functional.embedding(x, emb)
        return _quant(vllm.ir.ops.rms_norm(x, self.weight, EPS))


class GatedCombine(torch.nn.Module):
    """Deferred shared-expert gate + routed combine -> input norm -> quant."""

    def __init__(self, gate_reused: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(HIDDEN, device="cuda").bfloat16())
        self.gate_reused = gate_reused

    def forward(self, g, shared, routed, residual):
        gated = apply_shared_expert_gate(g, shared)
        x = (gated + routed).view(-1, HIDDEN)
        y, residual = vllm.ir.ops.fused_add_rms_norm(
            x, residual, self.weight.float() + 1.0, EPS
        )
        q, s = _quant(y)
        if self.gate_reused:  # the gated tensor is needed elsewhere
            return q, s, residual, gated
        return q, s, residual


@pytest.fixture
def vllm_config():
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    with vllm.config.set_current_vllm_config(config):
        torch.manual_seed(0)
        yield config


def _compile(config, model, *inputs, dynamic: bool):
    fusion = RMSNormMxfp8QuantFusionPass(config)
    backend = TestBackend(NoOpEliminationPass(config), fusion, PostCleanupPass(config))
    if dynamic:
        for t in inputs:
            torch._dynamo.mark_dynamic(t, 0)
    out = torch.compile(model, backend=backend, fullgraph=True)(*inputs)
    return fusion, backend, out


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("bf16_consumer", [False, True])
@torch.inference_mode()
def test_gemma_add_norm_fused(vllm_config, bf16_consumer: bool) -> None:
    model = GemmaAddNorm(bf16_consumer)
    a, b = (torch.randn(M, HIDDEN, device="cuda").bfloat16() for _ in range(2))
    residual = (torch.randn(M, HIDDEN, device="cuda") * 4).bfloat16()
    fusion, backend, out = _compile(vllm_config, model, a, b, residual, dynamic=True)

    assert fusion.matched_count == 1
    assert backend.op_count(QUANT) == 0
    assert backend.op_count(torch.ops.vllm_ir.fused_add_rms_norm) == 0
    (node,) = backend.graph_post_pass.find_nodes(op="call_function", target=FUSED)
    assert node.args[5] == 1.0  # Gemma w.float() + 1 absorbed
    assert node.args[6] is bf16_consumer  # bf16 output only when read

    normed, res_k, q_k, s_k, _ = add_rms_norm_mxfp8_quant(
        a, b, residual, model.weight, EPS, 1.0, True
    )
    assert _bitwise(out[0], q_k) and _bitwise(out[1], s_k)
    assert torch.equal(out[2], res_k)
    if bf16_consumer:
        assert torch.equal(out[3], torch.mm(normed, model.wba.t()))


@torch.inference_mode()
def test_fp32_weight_norm_fused(vllm_config) -> None:
    model = PlainNorm(torch.float32, embed=False)
    x = torch.randn(M, HIDDEN, device="cuda").bfloat16()
    fusion, backend, out = _compile(vllm_config, model, x, None, dynamic=False)

    assert fusion.matched_count == 1
    assert backend.op_count(QUANT) == 0
    _, _, q_k, s_k, _ = add_rms_norm_mxfp8_quant(
        x, None, None, model.weight, EPS, 0.0, False
    )
    assert _bitwise(out[0], q_k) and _bitwise(out[1], s_k)


@pytest.mark.parametrize(
    "dtype, embed",
    [
        (torch.float32, True),  # keeps Inductor's gather + norm fusion
        (torch.bfloat16, False),  # bf16 weight rounds before the multiply
    ],
)
@torch.inference_mode()
def test_norm_not_fused(vllm_config, dtype: torch.dtype, embed: bool) -> None:
    model = PlainNorm(dtype, embed)
    if embed:
        inputs = (
            torch.randint(0, 1000, (M,), device="cuda"),
            torch.randn(1000, HIDDEN, device="cuda").bfloat16(),
        )
    else:
        inputs = (torch.randn(M, HIDDEN, device="cuda").bfloat16(), None)
    fusion, backend, _ = _compile(vllm_config, model, *inputs, dynamic=False)

    assert fusion.matched_count == 0
    assert backend.op_count(QUANT) == 1
    assert backend.op_count(FUSED) == 0


@pytest.mark.parametrize("gate_reused", [False, True])
@torch.inference_mode()
def test_shared_expert_gate_absorbed(vllm_config, gate_reused: bool) -> None:
    model = GatedCombine(gate_reused)
    g = (torch.randn(M, 1, device="cuda") * 3).bfloat16()
    shared, routed = (torch.randn(M, HIDDEN, device="cuda").bfloat16() for _ in "ab")
    residual = (torch.randn(M, HIDDEN, device="cuda") * 4).bfloat16()
    fusion, backend, out = _compile(
        vllm_config, model, g, shared, routed, residual, dynamic=True
    )

    sigmoid = torch.ops.aten.sigmoid.default
    # The eager gate: bf16 sigmoid, bf16 broadcast product.
    pre_gated = torch.sigmoid(g) * shared
    if gate_reused:
        # The gated tensor is not materialized for the norm alone, so the
        # chain is left to Inductor and the gate keeps its eager rounding.
        assert fusion.matched_count == 0
        assert backend.op_count(sigmoid) == 1
        assert _bitwise(out[3], pre_gated)
        return

    assert fusion.matched_count == 1
    assert backend.op_count(sigmoid) == 0
    (node,) = backend.graph_post_pass.find_nodes(op="call_function", target=FUSED)
    assert node.args[7] is not None  # the gate logits
    _, res_k, q_k, s_k, _ = add_rms_norm_mxfp8_quant(
        routed, pre_gated, residual, model.weight, EPS, 1.0, False
    )
    assert _bitwise(out[0], q_k) and _bitwise(out[1], s_k)
    assert torch.equal(out[2], res_k)
