# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-prefix decode-batch order (rx-cascade "front" grouping, CentML/vllm #142).
Default OFF.

VLLM_PREFIX_SPREAD=1 enables it. Inside the uniform decode/verify segment that
sort_batch_req_ids puts first, decode requests are ordered by their full-attention KV
block-id row (prefix-cache block ids), i.e. in prefix-trie order. Prefix caching gives
requests that share a prefix the same block ids (chain hashing: an equal block implies
equal ancestors), so after the sort every set of decode requests sharing a prefix of
any depth is contiguous, and adjacent requests share the longest prefix that any pair
in the batch shares. The key is the whole block row: no depth constant, group size or
other parameter derived from a dataset's prompt layout. (This replaces #142's key on
the block at a fixed depth, PREFIX_SPREAD_DEPTH=8, and its spread-clusters mode.)

Why: the split-KV decode/verify attention kernels launch CTAs request-major, so adjacent
requests read the shared prefix pages close in time and hit L2 instead of DRAM.

Exactness: a pure permutation of the independent rows of the uniform decode segment. The
permutation is applied to req_ids before the batch is built, so every per-request tensor
(idx_mapping, input ids, positions, block tables, seq lens, spec-decode metadata, GDN
state indices, sampling state, output order) follows it. Batch size, padding, CUDA-graph
keys, the verification-first invariant and the extend/prefill tail are unchanged;
with no shared first block the stock order is returned as is. Host-only: not a
compile-hash factor.

Env:
  VLLM_PREFIX_SPREAD=1               enable (default 0)
  VLLM_PREFIX_SPREAD_LOG_EVERY=2000  log statistics every N steps (0 = off)
  VLLM_PREFIX_SPREAD_FORCE=1         gate/debug only: apply the trie order even when no
                                     two decode requests share a block (exercises a
                                     non-trivial permutation in correctness gates)
"""

import os
import time

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_PREFIX_SPREAD", "0").strip() == "1"
LOG_EVERY = int(os.environ.get("VLLM_PREFIX_SPREAD_LOG_EVERY", "2000") or 0)
FORCE = os.environ.get("VLLM_PREFIX_SPREAD_FORCE", "0").strip() == "1"
# Adjacent-share statistics are sampled every this many steps (host cost).
_STAT_SAMPLE = 8

STATS = {
    "steps": 0,
    "reordered": 0,
    "moved": 0,
    "decodes": 0,
    "host_ns": 0,
    "sampled": 0,
    "blocks": 0,
    "shared_blocks": 0,
    "adj_before": 0,
    "adj_after": 0,
}
_ENGAGED = [False]


def log_engage(attn_gid: int | None, group_names: list[str]) -> None:
    logger.info(
        "prefix-front: shared-prefix decode ordering ON (VLLM_PREFIX_SPREAD=1, "
        "key = full-attention block row); KV group %s of %s",
        attn_gid,
        group_names,
    )


def _adjacent_lcp(rows: np.ndarray) -> np.ndarray:
    """Common-prefix length (blocks) of each adjacent row pair; rows padded with -1."""
    eq = (rows[1:] == rows[:-1]) & (rows[1:] >= 0)
    # argmin finds the first mismatch; an all-equal pair has lcp = width.
    lcp = np.argmin(eq, axis=1)
    lcp[eq.all(axis=1)] = rows.shape[1]
    return lcp


def trie_order(rows: np.ndarray) -> np.ndarray:
    """rows: [n, D] int32 block-id rows (-1 padded). Returns a stable permutation
    that sorts the rows lexicographically by bytes, which makes every set of rows
    sharing a leading prefix contiguous.
    """
    rows = np.ascontiguousarray(rows)
    keys = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()
    return np.argsort(keys, kind="stable")


def reorder(
    req_ids: list[str],
    num_tokens_per_req: dict[str, int],
    draft_tokens: dict[str, list[int]],
    decode_query_len: int,
    req_id_to_index: dict[str, int],
    block_rows: np.ndarray,
    num_blocks: np.ndarray,
) -> list[str]:
    """req_ids: sort_batch_req_ids output. Only the leading uniform decode/verify
    run is permuted. block_rows: [max_num_reqs, max_blocks] host mirror of the
    full-attention KV block ids (-1 = none); num_blocks: [max_num_reqs] valid
    length of each row.
    """
    t0 = time.perf_counter_ns()
    n = 0
    for r in req_ids:
        if draft_tokens.get(r) and num_tokens_per_req[r] == decode_query_len:
            n += 1
        else:
            break
    out = req_ids
    sample = False
    if n > 1:
        idx = np.fromiter(
            map(req_id_to_index.__getitem__, req_ids[:n]), dtype=np.intp, count=n
        )
        first = block_rows[idx, 0]
        # Any decode request sharing its first block with another one?
        u, cnt = np.unique(first[first >= 0], return_counts=True)
        if FORCE or (u.size and cnt.max() >= 2):
            width = max(int(num_blocks[idx].max()), 1)
            rows = block_rows[idx, :width]
            perm = trie_order(rows)
            out = [req_ids[i] for i in perm.tolist()] + req_ids[n:]
            STATS["reordered"] += 1
            if (perm != np.arange(n)).any():
                STATS["moved"] += 1
            sample = LOG_EVERY > 0 and STATS["steps"] % _STAT_SAMPLE == 0
            if sample:
                before = _adjacent_lcp(rows)
                srows = rows[perm]
                after = _adjacent_lcp(srows)
                # Deepest share of each row = max(lcp with predecessor, successor).
                deep = np.zeros(n, dtype=np.int64)
                deep[1:] = after
                deep[:-1] = np.maximum(deep[:-1], after)
                STATS["sampled"] += 1
                STATS["blocks"] += int(num_blocks[idx].sum())
                STATS["shared_blocks"] += int(deep.sum())
                STATS["adj_before"] += int(before.sum())
                STATS["adj_after"] += int(after.sum())
            if not _ENGAGED[0]:
                _ENGAGED[0] = True
                logger.info(
                    "prefix-front: first reordered decode batch "
                    "(%d decodes, %d distinct first blocks)",
                    n,
                    u.size,
                )
    STATS["steps"] += 1
    STATS["decodes"] += n
    STATS["host_ns"] += time.perf_counter_ns() - t0
    if LOG_EVERY and STATS["steps"] % LOG_EVERY == 0:
        s = STATS["steps"]
        m = max(STATS["sampled"], 1)
        blocks = max(STATS["blocks"], 1)
        logger.info(
            "prefix-front stats: %d steps, %d reordered (%d moved), mean decodes %.1f, "
            "shared KV blocks %.1f%% (blocks/step %.0f), adjacent common prefix "
            "blocks/step before %.0f after %.0f, host %.1f us/step",
            s,
            STATS["reordered"],
            STATS["moved"],
            STATS["decodes"] / s,
            100.0 * STATS["shared_blocks"] / blocks,
            STATS["blocks"] / m,
            STATS["adj_before"] / m,
            STATS["adj_after"] / m,
            STATS["host_ns"] / s / 1e3,
        )
    return out
