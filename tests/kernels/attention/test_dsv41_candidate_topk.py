# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused candidate-block top-k (VLLM_DSV41_FAST_CANDIDATE_TOPK) must be
bit-identical to ``scores.topk(k)`` + ``_store_candidates_kernel``, including
tie order, +-0.0, NaN and -inf handling."""

import pytest
import torch

from vllm.model_executor.kernels.attention.dsa import candidate_blocks as cb
from vllm.triton_utils import triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(scores: torch.Tensor, out_k: int) -> torch.Tensor:
    rows, nblocks = scores.shape
    out = torch.full((rows, out_k), -7, dtype=torch.int32, device=scores.device)
    top = scores.topk(min(out_k, nblocks), dim=-1)
    cb._store_candidates_kernel[(rows, triton.cdiv(out_k, 256))](
        top.values, top.indices, out, *out.stride(), top.values.shape[1], out_k, 256
    )
    return out


def _scores(case: str, rows: int, n: int, g: torch.Generator) -> torch.Tensor:
    dev = "cuda"
    s = torch.randn(rows, n, device=dev, generator=g)
    if case == "ties":
        s = torch.randint(0, 7, (rows, n), device=dev, generator=g).float()
    elif case == "signed_zero":
        choices = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5], device=dev)
        s = choices[torch.randint(0, 5, (rows, n), device=dev, generator=g)]
    elif case == "nan":
        z = torch.rand(rows, n, device=dev, generator=g)
        s = torch.where(z < 0.01, torch.full_like(s, float("nan")), s)
        neg_nan = torch.tensor([-1], dtype=torch.int32, device=dev).view(torch.float32)
        s = torch.where(z > 0.99, neg_nan.expand_as(s), s)
    if case in ("tail", "ties", "nan"):
        lens = torch.randint(1, n + 1, (rows,), device=dev, generator=g)
        cols = torch.arange(n, device=dev)[None, :]
        s = torch.where(cols < lens[:, None], s, torch.full_like(s, -float("inf")))
        s[torch.arange(rows, device=dev), lens - 1] = float("inf")
    return s.contiguous()


@pytest.mark.parametrize("case", ["random", "ties", "tail", "signed_zero", "nan"])
@pytest.mark.parametrize("rows,n,out_k", [(48, 9000, 2048), (16, 1000, 2048), (8, 300, 64)])
def test_fast_select_topk_matches_torch(case, rows, n, out_k):
    g = torch.Generator(device="cuda").manual_seed(0)
    scores = _scores(case, rows, n, g)
    out = torch.full((rows, out_k), -7, dtype=torch.int32, device="cuda")
    cb.fast_select_topk(scores, min(out_k, n), out)
    torch.testing.assert_close(out, _reference(scores, out_k), atol=0, rtol=0)


@pytest.mark.parametrize("has_starts,row_repeat,max_row_len", [(False, 2, 6000), (True, 1, None)])
def test_select_candidate_blocks_fast_path_bitwise(
    monkeypatch, has_starts, row_repeat, max_row_len
):
    g = torch.Generator(device="cuda").manual_seed(1)
    rows, width, out_k = 64, 9000, 512
    logits = torch.randn(rows, width, device="cuda", generator=g).bfloat16().float()
    nb = rows // row_repeat
    ke = torch.randint(0, width + 1, (nb,), device="cuda", generator=g, dtype=torch.int32)
    if max_row_len is not None:
        ke = ke.clamp(max=max_row_len)
    ks = (ke.float() * 0.3).int() if has_starts else None
    outs = []
    for fast in (False, True):
        monkeypatch.setattr(cb, "FAST_CANDIDATE_TOPK", fast)
        out = torch.full((rows, out_k), -7, dtype=torch.int32, device="cuda")
        cb.select_candidate_blocks(
            logits, ks, ke, out_k, 8, out, row_repeat=row_repeat, max_row_len=max_row_len
        )
        outs.append(out)
    torch.testing.assert_close(outs[1], outs[0], atol=0, rtol=0)
