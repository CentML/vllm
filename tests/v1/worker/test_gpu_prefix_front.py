# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-prefix decode ordering (VLLM_PREFIX_SPREAD) must be an exact permutation
of the leading uniform decode segment that makes every shared-prefix set contiguous
and leaves the extend/prefill tail alone.
"""

import random

import numpy as np
import pytest

from vllm.v1.worker.gpu import prefix_front

DQL = 5  # decode query length (k=4)


def _batch(rows: dict[str, list[int]], tail: dict[str, int], max_blocks: int = 64):
    """rows: decode req_id -> block-id row; tail: non-decode req_id -> num tokens."""
    req_ids = list(rows) + list(tail)
    num_tokens = {r: DQL for r in rows} | tail
    draft = {r: [0] * (DQL - 1) for r in rows}
    index = {r: i for i, r in enumerate(req_ids)}
    block_rows = np.full((len(req_ids) + 3, max_blocks), -1, dtype=np.int32)
    nblk = np.zeros(len(req_ids) + 3, dtype=np.int64)
    for r, row in rows.items():
        block_rows[index[r], : len(row)] = row
        nblk[index[r]] = len(row)
    return req_ids, num_tokens, draft, index, block_rows, nblk


def _lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _random_forest(rng: random.Random, n: int) -> dict[str, list[int]]:
    """Rows built like prefix-cache chains: each request extends a random earlier
    request's prefix.
    """
    next_id = [1000]

    def fresh(k: int) -> list[int]:
        out = list(range(next_id[0], next_id[0] + k))
        next_id[0] += k
        return out

    rows: dict[str, list[int]] = {}
    for i in range(n):
        if rows and rng.random() < 0.8:
            parent = rows[rng.choice(list(rows))]
            row = parent[: rng.randint(1, len(parent))] + fresh(rng.randint(1, 6))
        else:
            row = fresh(rng.randint(1, 8))
        rows[f"r{i}"] = row
    keys = list(rows)
    rng.shuffle(keys)
    return {k: rows[k] for k in keys}


@pytest.mark.parametrize("seed", range(20))
def test_exact_permutation_and_contiguous_prefix_sets(seed):
    rng = random.Random(seed)
    rows = _random_forest(rng, rng.randint(2, 40))
    tail = {"p0": 3, "p1": 700, "p2": 2}
    args = _batch(rows, tail)
    req_ids = args[0]
    out = prefix_front.reorder(*(args[:3]), DQL, *(args[3:]))
    n = len(rows)
    assert sorted(out) == sorted(req_ids)
    assert len(out) == len(req_ids)
    assert out[n:] == req_ids[n:]
    seg = out[:n]
    assert set(seg) == set(rows)
    # Every set of rows sharing a leading prefix of any depth is one contiguous run.
    for depth in range(1, 9):
        seen_closed: set[tuple[int, ...]] = set()
        prev = None
        for r in seg:
            row = rows[r]
            key = tuple(row[:depth]) if len(row) >= depth else None
            if key != prev and prev is not None:
                seen_closed.add(prev)
            assert key is None or key not in seen_closed, (depth, key)
            prev = key
    # Adjacent shared prefix is maximal: each row's deepest share with any other
    # row is with a neighbour.
    for i, r in enumerate(seg):
        best = max((_lcp(rows[r], rows[o]) for o in seg if o != r), default=0)
        nb = [_lcp(rows[r], rows[seg[j]]) for j in (i - 1, i + 1) if 0 <= j < n]
        assert max(nb) == best


def test_no_shared_first_block_keeps_stock_order(monkeypatch):
    monkeypatch.setattr(prefix_front, "FORCE", False)
    rows = {"a": [5, 6], "b": [7], "c": [9, 10, 11]}
    args = _batch(rows, {"p": 40})
    out = prefix_front.reorder(*(args[:3]), DQL, *(args[3:]))
    assert out is args[0]


def test_only_leading_decode_segment_is_permuted():
    # A request with draft tokens but a different query length ends the segment.
    rows = {"a": [1, 2, 9], "b": [3], "c": [1, 2, 8]}
    req_ids, num_tokens, draft, index, block_rows, nblk = _batch(rows, {})
    num_tokens["b"] = DQL - 1
    out = prefix_front.reorder(req_ids, num_tokens, draft, DQL, index, block_rows, nblk)
    assert out == req_ids  # segment is ["a"] only: nothing to permute
    req_ids2 = ["a", "c", "b"]
    out2 = prefix_front.reorder(
        req_ids2, num_tokens, draft, DQL, index, block_rows, nblk
    )
    assert sorted(out2[:2]) == ["a", "c"] and out2[2] == "b"


def test_rows_padding_does_not_merge_distinct_prefixes():
    rows = {"x": [4, 1], "y": [4], "z": [4, 2], "w": [3]}
    args = _batch(rows, {})
    out = prefix_front.reorder(*(args[:3]), DQL, *(args[3:]))
    pos = {r: i for i, r in enumerate(out)}
    assert max(pos["x"], pos["y"], pos["z"]) - min(pos["x"], pos["y"], pos["z"]) == 2
