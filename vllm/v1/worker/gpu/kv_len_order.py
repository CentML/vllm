# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Longest-KV-first decode-batch order (``VLLM_DIE_LOCAL_KV_LPT=1``). Default OFF.

Inside the uniform decode/verify segment that ``sort_batch_req_ids`` puts first,
decode requests are ordered by their KV length, longest first (stable). With
``prefix_front`` (``VLLM_PREFIX_SPREAD=1``) the two orders compose: the requests
sharing a first prefix block stay one contiguous run in prefix-trie order (the
L2 reuse comes from that adjacency), and the runs and the other requests are
ordered by their longest KV, longest first.

Why: trtllm-gen's multi-CTA-KV decode kernels split each request's own KV length
over its ``k`` CTAs per KV head and dispatch the grid request-major, so with uneven
lengths the batch order decides how the CTAs pack onto the SMs. Longest-first is
LPT list scheduling: the long CTAs start in the first wave and the short ones fill
the tail. With die-local KV each die's launch runs on about half the SMs, with the
per-die split-k tiers of ``VLLM_DIE_LOCAL_KV_SPLIT_K``
(``vllm/v1/worker/die_local_kv.py``).

Exactness: a pure permutation of the independent rows of the uniform decode
segment, applied to req_ids before the batch is built (as ``prefix_front``), so
every per-request tensor follows it and sampling keeps its per-request seeds.
Batch size, padding, CUDA-graph keys, the verification-first invariant and the
extend/prefill tail are unchanged. Host-only.

Env:
  VLLM_DIE_LOCAL_KV_LPT=1               enable (default 0)
  VLLM_DIE_LOCAL_KV_LPT_LOG_EVERY=2000  log statistics every N steps (0 = off)
"""

import os
import time

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_DIE_LOCAL_KV_LPT"
ENABLED = os.environ.get(ENV, "0").strip() == "1"
LOG_EVERY = int(os.environ.get("VLLM_DIE_LOCAL_KV_LPT_LOG_EVERY", "2000") or 0)

STATS = {"steps": 0, "batches": 0, "moved": 0, "decodes": 0, "units": 0}
STATS_NS = [0]
_ACTIVE = [False]
_LOGGED = [False, False]


def engage(prefix_front_on: bool) -> bool:
    """Called by the model runner that applies ``reorder``; returns whether it
    should. Until then ``active()`` is False."""
    if not ENABLED:
        return False
    _ACTIVE[0] = True
    if not _LOGGED[0]:
        _LOGGED[0] = True
        logger.info(
            "die-local KV: longest-first decode order ON (%s=1%s)",
            ENV,
            "; shared-prefix runs kept contiguous" if prefix_front_on else "",
        )
    return True


def active() -> bool:
    return _ACTIVE[0]


def _runs(first: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of consecutive rows sharing a first block id (>= 0);
    every other row is a run of its own."""
    runs, s = [], 0
    for i in range(1, len(first) + 1):
        if i == len(first) or first[s] < 0 or first[i] != first[s]:
            runs.append((s, i))
            s = i
    return runs


def reorder(
    req_ids: list[str],
    num_tokens_per_req: dict[str, int],
    draft_tokens: dict[str, list[int]],
    decode_query_len: int,
    req_id_to_index: dict[str, int],
    num_computed_tokens: np.ndarray,
    block_rows: np.ndarray | None = None,
) -> list[str]:
    """req_ids: sort_batch_req_ids output (after ``prefix_front.reorder`` when
    ``block_rows``, its host mirror of the full-attention block ids, is given).
    Only the leading uniform decode/verify run is permuted: its runs of requests
    sharing a first block (single requests without ``block_rows``) by descending
    longest ``num_computed_tokens`` (host mirror, by request state index), stable,
    each run's internal order kept."""
    t0 = time.perf_counter_ns()
    n = 0
    for r in req_ids:
        if draft_tokens.get(r) and num_tokens_per_req[r] == decode_query_len:
            n += 1
        else:
            break
    out = req_ids
    if n > 1:
        idx = np.fromiter(
            map(req_id_to_index.__getitem__, req_ids[:n]), dtype=np.intp, count=n
        )
        lens = num_computed_tokens[idx].astype(np.int64)
        if block_rows is None:
            perm = np.argsort(-lens, kind="stable")
            nruns = n
        else:
            runs = _runs(block_rows[idx, 0])
            keys = np.array([lens[a:b].max() for a, b in runs], dtype=np.int64)
            order = np.argsort(-keys, kind="stable")
            perm = np.concatenate([np.arange(*runs[u]) for u in order.tolist()])
            nruns = len(runs)
        STATS["batches"] += 1
        STATS["units"] += nruns
        if (perm != np.arange(n)).any():
            out = [req_ids[i] for i in perm.tolist()] + req_ids[n:]
            STATS["moved"] += 1
        if not _LOGGED[1]:
            _LOGGED[1] = True
            logger.info(
                "die-local KV: first longest-first decode batch (%d decodes in %d "
                "runs, KV %d..%d tokens)",
                n,
                nruns,
                int(lens.min()),
                int(lens.max()),
            )
    STATS["steps"] += 1
    STATS["decodes"] += n
    STATS_NS[0] += time.perf_counter_ns() - t0
    if LOG_EVERY and STATS["steps"] % LOG_EVERY == 0:
        s, b = STATS["steps"], max(STATS["batches"], 1)
        logger.info(
            "die-local KV longest-first: %d steps, %d ordered, %d moved, mean "
            "decodes %.1f in %.1f runs, host %.1f us/step",
            s,
            STATS["batches"],
            STATS["moved"],
            STATS["decodes"] / s,
            STATS["units"] / b,
            STATS_NS[0] / s / 1e3,
        )
    return out
