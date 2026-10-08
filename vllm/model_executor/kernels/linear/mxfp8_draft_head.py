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
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.model_executor.layers.locality.mxgemm import DomainMxGemm

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

    weight_q_t: torch.Tensor
    weight_scale: torch.Tensor

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
        # VLLM_LOCALITY_LM_HEAD (draft): domain-local tcgen05 kernel for M <= loc_max_m
        self.loc_gemm: DomainMxGemm | None = None
        self.loc_max_m = 0
        self.loc_pdl = False

    def enable_locality(
        self, topo, weight: torch.Tensor, max_m: int, pdl: bool
    ) -> str | None:
        """Move the e4m3 weight to a 2 MiB-interleaved localized range and
        serve M <= max_m rows with ``locality.mxgemm.DomainMxGemm``. The
        FlashInfer fallback (larger M) reads the localized weight and the
        domain-0 copy of the scales; the cudaMalloc originals are released.
        Returns None, or why the head was left alone.
        """
        from vllm.model_executor.layers.locality import mxgemm
        from vllm.model_executor.layers.locality.memory import localize

        if not mxgemm.supported():
            return "the MXFP8 domain kernel needs SM107"
        n, k = weight.shape
        if k != mxgemm.K_DIM or n % 128:
            return f"needs a [N % 128 == 0, {mxgemm.K_DIM}] head, got {(n, k)}"
        weight_q, _ = mxfp8_e4m3_quantize(weight.contiguous())
        if not torch.equal(weight_q.t(), self.weight_q_t):
            return (
                "the MXFP8 copy differs from the shared lm_head (MTP checkpoint head)"
            )
        loc = localize(weight_q.view(torch.uint8), "interleave")
        del weight_q
        self.loc_gemm = mxgemm.DomainMxGemm(
            topo, loc, self.weight_scale, max_m=min(max_m, mxgemm.MAX_M)
        )
        self.weight_q_t = loc.tensor.view(self.weight_q_t.dtype).t()
        self.weight_scale = self.loc_gemm.sfa[0]
        self.loc_max_m = self.loc_gemm.max_m
        self.loc_pdl = pdl
        return None

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.loc_gemm is not None:
            x_2d = x.reshape(-1, x.shape[-1])
            m = x_2d.shape[0]
            if (
                0 < m <= self.loc_max_m
                and x_2d.stride(-1) == 1
                and x_2d.stride(0) % 8 == 0
                and x_2d.data_ptr() % 16 == 0
            ):
                out = self.loc_gemm(x_2d, pdl=self.loc_pdl)
                return out.view(*x.shape[:-1], self.loc_gemm.n)
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
