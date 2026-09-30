# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused (add +) RMSNorm -> MXFP8 producer against the ops it replaces."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

EPS = 1e-6


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.has_device_capability(100)
        and has_flashinfer()
    ),
    reason="FlashInfer MXFP8 quantize needs SM100+",
)
@pytest.mark.parametrize("num_tokens", [1, 300, 512])
@pytest.mark.parametrize("hidden", [1024, 2048])
@pytest.mark.parametrize("inputs", ["x", "x+res", "x+x2+res"])
@pytest.mark.parametrize("weight_offset", [0.0, 1.0])
@torch.inference_mode()
def test_add_rms_norm_mxfp8_quant(
    num_tokens: int, hidden: int, inputs: str, weight_offset: float
) -> None:
    from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
        add_rms_norm_mxfp8_quant,
    )

    torch.manual_seed(0)
    # Per-token magnitudes over many decades and an all-zero row (zero scales).
    scale = torch.logspace(-6, 3, num_tokens, device="cuda")[:, None]
    x = (torch.randn(num_tokens, hidden, device="cuda") * scale).bfloat16()
    x2 = (torch.randn_like(x, dtype=torch.float) * scale).bfloat16()
    res = (torch.randn_like(x, dtype=torch.float) * scale).bfloat16()
    for t in (x, x2, res):
        t[0] = 0
    x2 = x2 if inputs == "x+x2+res" else None
    res = res if inputs != "x" else None
    w = (torch.randn(hidden, device="cuda") * 0.2).float()

    normed, res_out, q, s = add_rms_norm_mxfp8_quant(
        x, x2, res, w, EPS, weight_offset, store_normed=True
    )
    _, _, q_nostore, s_nostore = add_rms_norm_mxfp8_quant(
        x, x2, res, w, EPS, weight_offset, store_normed=False
    )

    # The quant stage is the standalone swizzled quant of the bf16 output,
    # including the zero-filled 128-row scale padding.
    q_ref, s_ref = torch.ops.vllm.mxfp8_quantize(normed, True, 0)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s, s_ref)
    assert torch.equal(q_nostore.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(s_nostore, s)

    # The sum is taken in fp32 and rounded to bf16 once, as the fused IR norm.
    total = x.float()
    for t in (x2, res):
        if t is not None:
            total = total + t.float()
    if res is None:
        assert res_out.shape == (0, hidden)
    else:
        assert torch.equal(res_out, total.bfloat16())

    variance = total.pow(2).mean(dim=-1, keepdim=True)
    ref = (total * torch.rsqrt(variance + EPS) * (w + weight_offset)).bfloat16()
    # Only the fp32 reduction order differs: at most 1 bf16 ulp.
    torch.testing.assert_close(normed, ref, rtol=2**-7, atol=0)
