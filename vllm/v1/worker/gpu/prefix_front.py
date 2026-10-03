# SPDX-License-Identifier: Apache-2.0
"""Exact decode-batch order for shared-prefix decode attention (rx-cascade "front" grouping). Default OFF.

PREFIX_SPREAD=1 enables it. Inside the uniform decode/verify segment that sort_batch_req_ids puts first, decode requests
whose FIRST full-attention KV block is shared with at least one other decode request of the step (prefix-cache groups,
e.g. a common system prompt) are placed contiguously at the head of the segment: largest group first, stock order inside
a group, then the other requests in stock order. PREFIX_SPREAD_K>0 instead cuts the members into same-group clusters of K
and spreads the clusters evenly between the other requests (measured slower than front on Rubin; kept for study).

Why: the split-KV decode attention kernel dispatches CTAs request-major, so adjacent members read the shared prefix pages
from L2 instead of HBM (rx-cascade rxc-mb1, Rubin r1024 window: -0.31 ms/step of decode attention, bitwise equal).

Exactness: a pure permutation of independent rows inside the uniform decode segment. Batch size, padding, CUDA-graph keys,
the verification-first invariant (split_decodes_and_prefills, adaptive verification) and the extend/prefill tail are
unchanged; with no sharing the stock order is returned as is. Host-only: not a compile-hash factor.

Env:
  PREFIX_SPREAD=1              enable (default 0)
  PREFIX_SPREAD_K=0            0 = front (default); K > 0 = spread clusters of K
  PREFIX_SPREAD_LOG_EVERY=2000 log group / host-time statistics every N decode steps (0 = off)
"""
import os
import time

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("PREFIX_SPREAD", "0").strip() == "1"
K = int(os.environ.get("PREFIX_SPREAD_K", "0") or 0)
LOG_EVERY = int(os.environ.get("PREFIX_SPREAD_LOG_EVERY", "2000") or 0)
STATS = {"steps": 0, "reordered_steps": 0, "groups": 0, "grouped_reqs": 0, "decodes": 0, "host_ns": 0}
_ENGAGED = [False]


def log_engage(attn_gid, group_names) -> None:
    logger.info(
        "prefix-front: shared-prefix decode grouping ON (PREFIX_SPREAD=1, K=%d, %s); full-attention KV group %s of %s",
        K, "front" if K <= 0 else "spread", attn_gid, group_names)


def spread_segment(seg, first_block, k):
    """seg: decode request ids in stock order; first_block: req_id -> first block id (or None).
    Returns (permuted seg, number of groups, number of grouped requests)."""
    counts = {}
    for r in seg:
        fb = first_block(r)
        if fb is not None:
            counts[fb] = counts.get(fb, 0) + 1
    shared = {fb for fb, c in counts.items() if c >= 2}
    if not shared:
        return seg, 0, 0
    groups, gorder, non = {}, [], []
    for r in seg:
        fb = first_block(r)
        if fb in shared:
            if fb not in groups:
                groups[fb] = []
                gorder.append(fb)
            groups[fb].append(r)
        else:
            non.append(r)
    first_seen = {fb: i for i, fb in enumerate(gorder)}
    gl = [r for fb in sorted(gorder, key=lambda f: (-len(groups[f]), first_seen[f])) for r in groups[fb]]
    if k <= 0 or not non:
        return gl + non, len(groups), len(gl)
    clusters = [gl[j:j + k] for j in range(0, len(gl), k)]
    slots = {}
    for c, cl in enumerate(clusters):
        slots.setdefault(min(len(non), int((c + 0.5) * len(non) / len(clusters))), []).append(cl)
    out = []
    for j in range(len(non) + 1):
        for cl in slots.get(j, []):
            out.extend(cl)
        if j < len(non):
            out.append(non[j])
    return out, len(groups), len(gl)


def _front_numpy(seg, fb):
    """Front policy, vectorised: fb[i] = first block of seg[i] (-1 = unknown). Returns (perm or None, groups, members)."""
    n = len(seg)
    uniq, inv, cnt = np.unique(fb, return_inverse=True, return_counts=True)
    member = (fb >= 0) & (cnt[inv] >= 2)
    if not member.any():
        return None, 0, 0
    first_pos = np.full(len(uniq), n, dtype=np.int64)
    np.minimum.at(first_pos, inv, np.arange(n))
    gorder = np.lexsort((first_pos, -cnt))          # largest group first, ties by first appearance
    grank = np.empty(len(uniq), dtype=np.int64)
    grank[gorder] = np.arange(len(uniq))
    key = np.where(member, grank[inv], len(uniq))   # non-members after every group
    perm = np.argsort(key, kind="stable")           # stable: stock order inside a group and among non-members
    ng = int(np.count_nonzero(np.bincount(inv[member], minlength=len(uniq))))
    return perm, ng, int(member.sum())


def reorder(req_ids, num_tokens_per_req, draft_tokens, decode_query_len, req_id_to_index, first_block_np):
    """req_ids: sort_batch_req_ids output. Only the leading uniform decode/verify run is permuted."""
    t0 = time.perf_counter_ns()
    n = 0
    for r in req_ids:
        if draft_tokens.get(r) and num_tokens_per_req[r] == decode_query_len:
            n += 1
        else:
            break
    out = req_ids
    ng = nr = 0
    if n > 1:
        seg = req_ids[:n]
        fb = first_block_np[np.fromiter(map(req_id_to_index.__getitem__, seg), dtype=np.intp, count=n)]
        if K <= 0:
            perm, ng, nr = _front_numpy(seg, fb)
            if perm is not None:
                out = [seg[i] for i in perm] + list(req_ids[n:])
        else:
            fbd = {r: (int(v) if v >= 0 else None) for r, v in zip(seg, fb)}
            pseg, ng, nr = spread_segment(seg, fbd.get, K)
            if ng:
                out = list(pseg) + list(req_ids[n:])
        if ng and not _ENGAGED[0]:
            _ENGAGED[0] = True
            logger.info("prefix-front: first reordered decode batch (%d decodes, %d groups, %d grouped requests)",
                        n, ng, nr)
    STATS["steps"] += 1
    STATS["decodes"] += n
    if ng:
        STATS["reordered_steps"] += 1
        STATS["groups"] += ng
        STATS["grouped_reqs"] += nr
    STATS["host_ns"] += time.perf_counter_ns() - t0
    if LOG_EVERY and STATS["steps"] % LOG_EVERY == 0:
        s = STATS["steps"]
        logger.info(
            "prefix-front stats: %d steps, %d reordered, mean decodes %.1f, groups/step %.2f, grouped reqs/step %.1f, "
            "host %.1f us/step", s, STATS["reordered_steps"], STATS["decodes"] / s, STATS["groups"] / s,
            STATS["grouped_reqs"] / s, STATS["host_ns"] / s / 1e3)
    return out
