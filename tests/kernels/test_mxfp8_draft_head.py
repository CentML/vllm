# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("CuTe-DSL mm_mxfp8 needs an SM10x GPU", allow_module_level=True)

from vllm.model_executor.kernels.linear.mxfp8_draft_head import (  # noqa: E402
    Mxfp8DraftLmHead,
)

VOCAB, HIDDEN = 8192, 2048


@pytest.fixture(scope="module")
def head():
    torch.manual_seed(0)
    w = torch.randn(VOCAB, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    return w, Mxfp8DraftLmHead(w)


@pytest.mark.parametrize("m", [1, 4, 16, 32])
def test_mxfp8_logits_track_bf16(head, m: int):
    w, mx = head
    torch.manual_seed(m)
    x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(x, w).float()
    out = mx(x, w)
    assert out.shape == (m, VOCAB) and out.dtype == torch.bfloat16
    # MXFP8 (e4m3, 32-element blocks) on both operands: a few % of the logit
    # range, far below the gap that decides a confident draft argmax.
    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < 0.05, err
    # A clear winner per row survives quantization.
    x_peaked = x.clone()
    x_peaked[:, :] = w[torch.arange(m, device="cuda") * 7] * 50
    assert torch.equal(mx(x_peaked, w).argmax(-1), F.linear(x_peaked, w).argmax(-1))


def test_large_m_keeps_bf16(head):
    w, mx = head
    x = torch.randn(33, HIDDEN, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(mx(x, w), F.linear(x, w))
