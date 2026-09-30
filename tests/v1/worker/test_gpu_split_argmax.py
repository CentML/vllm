# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.split_argmax import split_argmax

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _logits(kind: str, rows: int, vocab: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(rows)
    if kind == "ties":
        # Many exact ties at the maximum; the lowest index must win.
        return torch.randint(
            -3, 4, (rows, vocab), device="cuda", generator=g
        ).bfloat16()
    x = (torch.randn(rows, vocab, device="cuda", generator=g) * 3).bfloat16()
    if kind == "nan":
        x[:, 200000] = float("inf")
        x[:, 12345] = float("nan")
        x[:, 99000] = float("nan")
    elif kind == "signed_zero":
        x.fill_(-1.0)
        x[:, 7000] = 0.0
        x[:, 5000] = -0.0
    elif kind == "all_neg_inf":
        x.fill_(float("-inf"))
    return x


@pytest.mark.parametrize("rows", [1, 8, 128])
@pytest.mark.parametrize("kind", ["randn", "ties", "nan", "signed_zero", "all_neg_inf"])
def test_matches_torch_argmax(rows: int, kind: str):
    logits = _logits(kind, rows, 248320)
    out = split_argmax(logits)
    assert out.dtype == torch.int64
    assert torch.equal(out, logits.argmax(dim=-1))


def test_row_stride_and_short_vocab():
    base = (torch.randn(5, 248320 + 64, device="cuda") * 3).bfloat16()
    logits = base[:, :248320]
    assert torch.equal(split_argmax(logits), logits.argmax(dim=-1))
    short = (torch.randn(3, 1000, device="cuda")).float()
    assert torch.equal(split_argmax(short), short.argmax(dim=-1))
