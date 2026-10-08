# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row selection of MTP draft prefill attention pruning (``draft_prefill_rows``)."""

import pytest
import torch

pytest.importorskip("flashinfer")

from vllm.v1.attention.backends.flashinfer import (  # noqa: E402
    _bounded_dequant_block_tables,
    draft_prefill_rows,
)


def _batch(reqs, max_num_reqs=None, lti_dtype=torch.int64, meta_dtype=torch.int32):
    """Build a draft prefill batch the way the speculator and the FlashInfer
    metadata builder lay it out.

    ``reqs``: (query_len, seq_len, num_rejected, is_decode) in batch order,
    decodes first. Returns the attention inputs and the reference sampled rows
    and KV lengths, computed independently: the sampled row is the last
    accepted token (``query_start + query_len - num_rejected - 1``), and its KV
    length is its position + 1.
    """
    num_reqs = len(reqs)
    max_num_reqs = max_num_reqs or num_reqs
    lti = torch.zeros(max_num_reqs, dtype=lti_dtype)
    positions = []
    start = 0
    for i, (q_len, seq_len, num_rejected, _) in enumerate(reqs):
        lti[i] = start + q_len - num_rejected - 1
        positions += list(range(seq_len - q_len, seq_len))
        start += q_len
    positions = torch.tensor(positions)

    decodes = [r for r in reqs if r[3]]
    prefills = [r for r in reqs if not r[3]]
    assert reqs[: len(decodes)] == decodes, "decodes must come first"
    num_decodes = len(decodes)
    num_decode_tokens = sum(r[0] for r in decodes)
    cum_q = [0]
    for q_len, *_ in prefills:
        cum_q.append(cum_q[-1] + q_len)
    cum_q = torch.tensor(cum_q, dtype=meta_dtype)
    seq_lens = torch.tensor([r[1] for r in prefills], dtype=meta_dtype)

    prefill_lti = lti[num_decodes:num_reqs].long()
    ref_rows = prefill_lti - num_decode_tokens
    ref_kv_lens = (positions[prefill_lti] + 1).to(torch.int32)
    return (
        dict(
            cum_q=cum_q,
            seq_lens=seq_lens,
            last_token_indices=lti,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=len(prefills),
        ),
        ref_rows,
        ref_kv_lens,
    )


# (query_len, seq_len, num_rejected, is_decode)
_MIXED = [
    # Spec decodes with 3 query tokens each: num_decode_tokens > num_decodes.
    (3, 50, 1, True),
    (3, 61, 0, True),
    # Prefill-bucket requests carrying draft tokens, with and without rejections.
    (5, 100, 2, False),
    (5, 80, 0, False),
    # Chunked prefill over a cached prefix.
    (64, 1000, 0, False),
    # Fresh prefill.
    (7, 7, 0, False),
]


@pytest.mark.parametrize("lti_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("meta_dtype", [torch.int32, torch.int64])
def test_rows_and_kv_lens_match_sampled_rows(lti_dtype, meta_dtype):
    kwargs, ref_rows, ref_kv_lens = _batch(
        _MIXED, lti_dtype=lti_dtype, meta_dtype=meta_dtype
    )
    assert kwargs["num_decode_tokens"] > kwargs["num_decodes"]

    rows, kv_lens = draft_prefill_rows(**kwargs)

    assert rows.dtype == torch.int64
    assert kv_lens.dtype == torch.int32
    torch.testing.assert_close(rows, ref_rows, rtol=0, atol=0)
    torch.testing.assert_close(kv_lens, ref_kv_lens, rtol=0, atol=0)
    # The rejected-draft request samples an earlier row than its chunk's last,
    # with a shorter causal KV than its sequence length.
    assert rows[0].item() == 2 and kv_lens[0].item() == 98
    assert rows[1].item() == 9 and kv_lens[1].item() == 80
    assert rows[2].item() == 73 and kv_lens[2].item() == 1000


def test_padded_buffers_are_sliced():
    # The speculator's buffer is max_num_reqs long and zero-padded; metadata
    # tensors may also be longer than the prefill bucket.
    kwargs, ref_rows, ref_kv_lens = _batch(_MIXED, max_num_reqs=16)
    kwargs["cum_q"] = torch.cat([kwargs["cum_q"], kwargs["cum_q"][-1:].repeat(3)])
    kwargs["seq_lens"] = torch.cat(
        [kwargs["seq_lens"], torch.zeros(3, dtype=torch.int32)]
    )

    rows, kv_lens = draft_prefill_rows(**kwargs)

    assert rows.shape == kv_lens.shape == (kwargs["num_prefills"],)
    torch.testing.assert_close(rows, ref_rows, rtol=0, atol=0)
    torch.testing.assert_close(kv_lens, ref_kv_lens, rtol=0, atol=0)


def test_prefill_only_batch():
    kwargs, ref_rows, ref_kv_lens = _batch([r for r in _MIXED if not r[3]])
    assert kwargs["num_decodes"] == kwargs["num_decode_tokens"] == 0

    rows, kv_lens = draft_prefill_rows(**kwargs)

    torch.testing.assert_close(rows, ref_rows, rtol=0, atol=0)
    torch.testing.assert_close(kv_lens, ref_kv_lens, rtol=0, atol=0)


def test_zero_indices_are_clamped_into_each_span():
    # CUDA-graph capture and dummy runs see a zero-filled index buffer; an
    # unclamped row would be negative or in another request's span.
    kwargs, _, _ = _batch(_MIXED)
    kwargs["last_token_indices"] = torch.zeros_like(kwargs["last_token_indices"])

    rows, kv_lens = draft_prefill_rows(**kwargs)

    cum_q = kwargs["cum_q"].long()
    seq_lens = kwargs["seq_lens"]
    assert torch.all(rows >= cum_q[:-1]) and torch.all(rows <= cum_q[1:] - 1)
    # Clamped to the span's first row: KV up to and including that row.
    torch.testing.assert_close(rows, cum_q[:-1], rtol=0, atol=0)
    q_lens = cum_q[1:] - cum_q[:-1]
    torch.testing.assert_close(
        kv_lens, (seq_lens - q_lens + 1).to(torch.int32), rtol=0, atol=0
    )
    assert torch.all(kv_lens >= 1) and torch.all(kv_lens <= seq_lens)


def test_out_of_span_high_index_is_clamped_to_span_end():
    kwargs, _, _ = _batch(_MIXED)
    lti = kwargs["last_token_indices"].clone()
    lti[kwargs["num_decodes"] :] = 10_000
    kwargs["last_token_indices"] = lti

    rows, kv_lens = draft_prefill_rows(**kwargs)

    cum_q = kwargs["cum_q"].long()
    torch.testing.assert_close(rows, cum_q[1:] - 1, rtol=0, atol=0)
    torch.testing.assert_close(kv_lens, kwargs["seq_lens"], rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [16, 128])
def test_dequant_tables_bound_live_pages_not_model_capacity(block_size):
    # A 262k-token model's persistent table contains stale, out-of-cache IDs.
    # Only two live pages are needed in this batch; rejected tails also must
    # not cause the shorter request's stale second page to be read.
    capacity = 262144 // block_size
    block_tables = torch.full((32, capacity), 1_000_000, dtype=torch.int32)
    block_tables[:, 0] = torch.arange(1, 33, dtype=torch.int32)
    block_tables[0, 1] = 33
    seq_lens = torch.full((32,), block_size - 1, dtype=torch.int32)
    seq_lens[0] = block_size + 1
    actual = _bounded_dequant_block_tables(
        block_tables, seq_lens, block_size + 1, block_size
    )
    assert actual.shape == (32, 2)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual[:, 0], block_tables[:, 0], atol=0, rtol=0)
    assert actual[0, 1] == 33
    assert torch.count_nonzero(actual[1:, 1]) == 0
    # Dequantization's allocation is one null page + requests * table width.
    assert 1 + actual.numel() == 65
