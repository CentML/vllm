# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduced-vocabulary greedy draft head (VLLM_DRAFT_LMH_VOCAB)."""

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.worker.gpu.spec_decode.draft_vocab_head import (
    DraftVocabHead,
    parse_ranges,
    subset_argmax,
)


def test_parse_ranges():
    assert parse_ranges("0-10, 20-30", 100) == [(0, 10), (20, 30)]
    assert parse_ranges("20-30,0-10", 100) == [(0, 10), (20, 30)]
    with pytest.raises(ValueError):
        parse_ranges("0-10,5-20", 100)
    with pytest.raises(ValueError):
        parse_ranges("0-101", 100)
    with pytest.raises(ValueError):
        parse_ranges("7", 100)


def _ref(h, w, ranges):
    ids = torch.cat([torch.arange(a, b) for a, b in ranges])
    logits = F.linear(h.float(), w.float())
    sub = logits[:, ids]
    return logits.argmax(-1), ids[sub.argmax(-1)], ids


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_subset_argmax_matches_reference(device):
    torch.manual_seed(0)
    vocab, hidden = 1000, 64
    w = torch.randn(vocab, hidden, dtype=torch.bfloat16, device=device)
    ranges = [(0, 600), (990, 1000)]
    head = DraftVocabHead(w, ranges, "bf16")
    h = torch.randn(33, hidden, dtype=torch.bfloat16, device=device)
    got = head(h)
    # Same GEMM rows as the full head -> identical logits per kept id.
    full_logits = F.linear(h, w)
    ids = head.ids
    want = ids[full_logits[:, ids].argmax(-1)]
    assert torch.equal(got, want)
    # Rows whose full argmax is inside the subset get the full argmax.
    full = full_logits.argmax(-1)
    inside = torch.isin(full, ids)
    assert inside.any()
    assert torch.equal(got[inside], full[inside])
    assert got.dtype == torch.int64


def test_ties_pick_lowest_id():
    w = torch.zeros(8, 4, dtype=torch.bfloat16)
    w[3] = 1.0
    w[6] = 1.0
    head = DraftVocabHead(w, [(2, 8)], "bf16")
    h = torch.ones(2, 4, dtype=torch.bfloat16)
    assert head(h).tolist() == [3, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("n,v", [(1, 98432), (4, 98432), (7, 5000), (64, 131072)])
def test_subset_argmax_kernel_matches_torch(n, v):
    torch.manual_seed(1)
    x = torch.randn(n, v, device="cuda", dtype=torch.bfloat16)
    x[0, 17] = x[0].max() + 1
    x[0, v - 3] = x[0, 17]  # tie: lowest index wins
    ids = torch.arange(v, device="cuda", dtype=torch.int64) * 3 + 5
    assert torch.equal(subset_argmax(x, ids), ids[x.argmax(-1)])
