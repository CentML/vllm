# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm::moe_finalize and its fusion into the MXFP8 norm producer.

On SM100 the TRT-LLM test also checks bit equality with FlashInfer's own
finalize (``do_finalize=True``) on both of its kernels.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

HIDDEN, TOP_K, EPS = 2048, 8, 1e-6

requires_cuda = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA")
requires_mxfp8_quantize = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.has_device_capability(100)
        and has_flashinfer()
    ),
    reason="FlashInfer MXFP8 quantize needs SM100+",
)


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _deferred(tokens: int, pad_rows: int, drop: bool):
    """A deferred MoE triple whose GEMM2 buffer has NaN padding rows.

    Routes land in distinct random rows of a [tokens*TOP_K + pad_rows, HIDDEN]
    buffer, like expert-tile padding; with ``drop`` some routes are -1.
    """
    rows = tokens * TOP_K + pad_rows
    buf = torch.full((rows, HIDDEN), float("nan"), device="cuda").bfloat16()
    idx = torch.randperm(rows, device="cuda")[: tokens * TOP_K].int()
    # Per-row magnitudes over decades so the fp32 sum order matters.
    scale = torch.logspace(-3, 3, tokens * TOP_K, device="cuda")[:, None]
    buf[idx.long()] = (torch.randn(tokens * TOP_K, HIDDEN, device="cuda") * scale).to(
        torch.bfloat16
    )
    idx = idx.view(tokens, TOP_K)
    if drop:
        idx[::3, 1] = -1
    weights = torch.rand(tokens, TOP_K, device="cuda").bfloat16()
    return buf[: tokens * TOP_K], weights, idx.contiguous()


def _reference(permuted, weights, idx) -> torch.Tensor:
    """fp32 sequential FMA in k order (exact via fp64), -1 skipped, bf16 out."""
    storage = permuted.as_strided((idx.max().item() + 1, HIDDEN), (HIDDEN, 1))
    acc = torch.zeros(idx.shape[0], HIDDEN, device="cuda")
    for k in range(TOP_K):
        p = idx[:, k]
        valid = (p >= 0)[:, None]
        rows = storage[p.clamp(min=0).long()].double()
        fma = (weights[:, k : k + 1].double() * rows + acc.double()).float()
        acc = torch.where(valid, fma, acc)
    return acc.bfloat16()


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 37, 300])
@pytest.mark.parametrize("drop", [False, True])
@torch.inference_mode()
def test_moe_finalize(tokens: int, drop: bool) -> None:
    from vllm.model_executor.layers.fusion.moe_finalize import moe_finalize

    torch.manual_seed(0)
    permuted, weights, idx = _deferred(tokens, pad_rows=1000, drop=drop)
    out = moe_finalize(permuted, weights, idx)
    assert not out.isnan().any()  # padding rows are never read
    assert _bitwise(out, _reference(permuted, weights, idx))


@requires_mxfp8_quantize
@pytest.mark.parametrize("tokens", [1, 37, 300])
@pytest.mark.parametrize("gated", [False, True])
@torch.inference_mode()
def test_moe_finalize_add_rms_norm_mxfp8_quant(tokens: int, gated: bool) -> None:
    """The in-kernel reduction equals moe_finalize feeding the plain op."""
    from vllm.model_executor.layers.fusion.moe_finalize import moe_finalize
    from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
        add_rms_norm_mxfp8_quant,
    )

    torch.manual_seed(0)
    permuted, weights, idx = _deferred(tokens, pad_rows=500, drop=False)
    shared = torch.randn(tokens, HIDDEN, device="cuda").bfloat16()
    gate = (torch.randn(tokens, 1, device="cuda") * 3).bfloat16() if gated else None
    residual = (torch.randn(tokens, HIDDEN, device="cuda") * 4).bfloat16()
    w = torch.randn(HIDDEN, device="cuda").bfloat16()

    routed = moe_finalize(permuted, weights, idx)
    ref = add_rms_norm_mxfp8_quant(
        routed, shared, residual, w, EPS, 1.0, True, gate, True
    )
    out = torch.ops.vllm.moe_finalize_add_rms_norm_mxfp8_quant(
        permuted, weights, idx, shared, residual, w, EPS, 1.0, True, gate, True
    )
    for a, b in zip(out, ref):
        assert _bitwise(a, b)


@requires_mxfp8_quantize
@pytest.mark.parametrize("tokens", [1, 32, 147, 148, 512, 2048])
@torch.inference_mode()
def test_moe_finalize_matches_trtllm(tokens: int) -> None:
    """Bit-equal to FlashInfer's TRT-LLM finalize on the MXFP8 MoE.

    148 tokens is where FlashInfer switches finalizeKernel -> VecLoad (H=2048).
    """
    if not current_platform.is_device_capability_family(100):
        pytest.skip("TRT-LLM-Gen MoE needs SM10x")
    from flashinfer import mxfp8_quantize
    from flashinfer.fused_moe import (
        Fp8QuantizationType,
        WeightLayout,
        trtllm_fp8_block_scale_moe,
    )

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fusion.moe_finalize import moe_finalize
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )

    def quant(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The GEMM layout does not matter here: both finalizes read the same
        # GEMM2 output.
        q, s = mxfp8_quantize(t.reshape(-1, t.shape[-1]), False)
        return q.view(t.shape), s.view(torch.uint8).view(*t.shape[:-1], -1)

    torch.manual_seed(0)
    experts, inter = 256, 512
    x = torch.randn(tokens, HIDDEN, device="cuda").bfloat16()
    logits = torch.randn(tokens, experts, device="cuda").bfloat16()
    w1_q, w1_s = quant(
        (torch.randn(experts, 2 * inter, HIDDEN, device="cuda") * 0.05).bfloat16()
    )
    w2_q, w2_s = quant(
        (torch.randn(experts, HIDDEN, inter, device="cuda") * 0.05).bfloat16()
    )
    x_q, x_s = quant(x)
    kwargs = dict(
        routing_logits=logits,
        routing_bias=None,
        hidden_states=x_q,
        hidden_states_scale=x_s,
        gemm1_weights=w1_q,
        gemm1_weights_scale=w1_s,
        gemm2_weights=w2_q,
        gemm2_weights_scale=w2_s,
        num_experts=experts,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=inter,
        local_expert_offset=0,
        local_num_experts=experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.MajorK,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=activation_to_flashinfer_int(MoEActivation.SILU),
    )
    finalized = trtllm_fp8_block_scale_moe(**kwargs)
    if isinstance(finalized, (list, tuple)):
        finalized = finalized[0]
    permuted, weights, idx = trtllm_fp8_block_scale_moe(**kwargs, do_finalize=False)[:3]
    idx = idx.view(tokens, TOP_K)
    out = moe_finalize(permuted[: tokens * TOP_K], weights.view(tokens, TOP_K), idx)
    assert _bitwise(out, finalized)
