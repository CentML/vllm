# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused attention gate -> MXFP8 producer against the chain it replaces."""

import pytest
import torch

from vllm.platforms import current_platform

HEADS, HEAD_DIM = 16, 256
HIDDEN = HEADS * HEAD_DIM


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="FlashInfer MXFP8 quantize needs SM100+",
)
# Both sides of the launch-config boundaries (256, 1024) and partial 128-row
# scale padding.
@pytest.mark.parametrize("num_tokens", [1, 129, 256, 257, 1023, 1024, 4800])
@pytest.mark.parametrize("strided_gate", [False, True])
@torch.inference_mode()
def test_attn_gate_mxfp8_quant_matches_inductor_chain(
    num_tokens: int, strided_gate: bool
) -> None:
    import vllm.model_executor.layers.quantization.utils.mxfp8_utils  # noqa: F401
    from vllm.model_executor.layers.fusion.attn_gate_mxfp8_quant import (
        attn_gate_mxfp8_quant,
    )

    torch.manual_seed(0)
    # Per-token magnitudes over many decades, an all-zero row (zero scales).
    scale = torch.logspace(-6, 3, num_tokens, device="cuda")[:, None]
    attn = (torch.randn(num_tokens, HIDDEN, device="cuda") * scale).bfloat16()
    attn[0] = 0
    q_gate = (torch.randn(num_tokens, 2 * HIDDEN + 1024, device="cuda") * 4).bfloat16()
    gate = q_gate[:, : 2 * HIDDEN].view(num_tokens, HEADS, 2, HEAD_DIM)[:, :, 1]
    if not strided_gate:
        gate = gate.contiguous()
    attn_in = attn.view(num_tokens, HEADS, HEAD_DIM)

    # What serving runs: the Inductor gate kernel, then o_proj's input quant.
    def chain(a, g):
        y = (a * torch.sigmoid(g)).flatten(1)
        return torch.ops.vllm.mxfp8_quantize(y, True, 0)

    q_ref, s_ref = torch.compile(chain, fullgraph=True)(attn_in, gate)
    q, s = attn_gate_mxfp8_quant(attn_in, gate)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s, s_ref)
