# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 copy of a BF16 lm_head for speculative-draft logits.

The Qwen3.5/3.6 MTP drafter proposes tokens by greedy argmax over the full
vocabulary through the target's BF16 lm_head (``248320 x 2048``, 1 GB): about
77 us per draft step on VR, three steps per decode iteration. The proposals
only decide which tokens the target verifies, so this head runs the draft
logits as an MXFP8 GEMM (FlashInfer CuTe-DSL, the backend of the model's other
MXFP8 linears) on a once-quantized copy of the weight: 41 us for M <= 32 rows
in the served model. Larger M keeps the BF16 GEMM: in serving, the GEMM tactic
picked for 40-176 draft rows took 83-114 us, slower than BF16 (78-80 us).

The FlashInfer warmup autotune never runs the draft head (serving logged
"No tuned config covers mxfp8_gemm ... 248320; falling back"), and the
fallback tactic is the slow one at M 33-63 and >= 176 (VR, M=40: 111 us vs
47 us tuned). ``Mxfp8DraftLmHead.autotune`` profiles the head at every
power-of-two M up to the cap during the FlashInfer autotune warmup; with it,
``VLLM_MTP_DRAFT_LM_HEAD_MXFP8_MAX_M`` (default 32, the original cap) can
extend the MXFP8 head to C512 draft batches (M 64-128: 47-51 us vs 85 us BF16).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.torch_utils import direct_register_custom_op

# Largest draft row count served by the MXFP8 GEMM (Nsight A/B on SM107);
# VLLM_MTP_DRAFT_LM_HEAD_MXFP8_MAX_M raises it (use with the warmup autotune).
_MXFP8_MAX_M = int(os.environ.get("VLLM_MTP_DRAFT_LM_HEAD_MXFP8_MAX_M", "32"))


def mtp_draft_logits_mxfp8_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_q_t: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    k = weight.shape[1]
    x_2d = x.reshape(-1, k)
    if x_2d.shape[0] == 0 or x_2d.shape[0] > _MXFP8_MAX_M:
        return F.linear(x, weight)
    x_q, x_scale = mxfp8_e4m3_quantize(x_2d.contiguous(), is_sf_swizzled_layout=True)
    out = vllm_flashinfer.mm_mxfp8(
        x_q, weight_q_t, x_scale, weight_scale, out_dtype=x.dtype, backend="cute-dsl"
    )
    return out.view(*x.shape[:-1], weight.shape[0])


def mtp_draft_logits_mxfp8_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_q_t: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="mtp_draft_logits_mxfp8",
    op_func=mtp_draft_logits_mxfp8_impl,
    fake_impl=mtp_draft_logits_mxfp8_fake,
)


class Mxfp8DraftLmHead(torch.nn.Module):
    """Holds the MXFP8 weight copy; the BF16 head stays the source of truth."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        n, k = weight.shape
        assert weight.dtype == torch.bfloat16 and k % MXFP8_BLOCK_SIZE == 0
        weight_q, weight_scale = mxfp8_e4m3_quantize(weight.contiguous())
        weight_scale = swizzle_mxfp8_scale(
            weight_scale.view(n, k // MXFP8_BLOCK_SIZE), M=n, K=k
        )
        # mm_mxfp8 takes operand B column-major: [K, N] view of [N, K].
        self.register_buffer("weight_q_t", weight_q.t(), persistent=False)
        self.register_buffer(
            "weight_scale", weight_scale.contiguous(), persistent=False
        )

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.mtp_draft_logits_mxfp8(
            x, weight, self.weight_q_t, self.weight_scale
        )

    @torch.inference_mode()
    def autotune(self, weight: torch.Tensor) -> list[int]:
        """Profile the head's GEMM at M = 1, 2, 4, ... up to the MXFP8 cap.
        Call inside FlashInfer's ``autotune(tune_mode=True)`` context (the
        kernel warmup's), whose buckets are powers of two.
        """
        ms = []
        m = 1
        while m <= _MXFP8_MAX_M:
            ms.append(m)
            m *= 2
        if ms and ms[-1] < _MXFP8_MAX_M:
            ms.append(_MXFP8_MAX_M)
        k = weight.shape[1]
        for m in ms:
            x = torch.randn(m, k, device=weight.device, dtype=weight.dtype)
            mtp_draft_logits_mxfp8_impl(x, weight, self.weight_q_t, self.weight_scale)
        return ms
