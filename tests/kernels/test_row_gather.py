# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm::gather_rows3 (MTP draft prefill row pruning) vs index_select."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA required", allow_module_level=True)

from vllm.model_executor.layers.row_gather import gather_rows3  # noqa: E402


@pytest.mark.parametrize("num_tokens,rows", [(5, [4]), (290, None), (12, [0, 0, 7])])
@torch.inference_mode()
def test_gather_rows3_matches_index_select(num_tokens: int, rows: list[int] | None):
    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, head_dim, hidden = 16, 256, 2048
    attn = torch.randn(num_tokens, heads, head_dim, dtype=torch.bfloat16, device=device)
    # The gate as Qwen3NextAttention reads it in place: [:, :, 1] of a
    # [tokens, heads, 2, head_dim] view of the QKV projection (row stride
    # includes k and v).
    qkv = torch.randn(
        num_tokens, heads * 2 * head_dim + 1024, dtype=torch.bfloat16, device=device
    )
    gate = qkv[:, : heads * 2 * head_dim].view(-1, heads, 2, head_dim)[:, :, 1]
    residual = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    if rows is None:
        # Last accepted row of 58 requests of 5 rows each, in request order.
        idx = torch.arange(58, device=device) * 5 + torch.randint(
            0, 5, (58,), device=device
        )
    else:
        idx = torch.tensor(rows, dtype=torch.int64, device=device)

    outs = gather_rows3(attn, gate, residual, idx)

    for out, x in zip(outs, (attn, gate, residual)):
        assert out.is_contiguous() and out.shape == (idx.numel(), *x.shape[1:])
        torch.testing.assert_close(out, x.index_select(0, idx), atol=0, rtol=0)
