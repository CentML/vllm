# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact shared-prefix junctions for hybrid (attention + Mamba "align") models.

Opt-in with ``VLLM_PREFIX_EXACT_JUNCTION=1`` (default off; outputs unchanged).

Problem. Requests that share a long prefix (e.g. a common system prompt) but
diverge inside a cache block only reuse the prefix up to a cache-block boundary:
the first request's prompt tail block is registered under its own tail hash, so
a sibling's attention lookup stops at the last fully shared block, and the
Marconi junction it then materializes is block-floored. With 1152-token blocks
and a 32-token prefix-match unit this recomputes up to a block plus one unit of
shared prefix for every sibling.

Mechanism (content-agnostic; only prefix-chain hashes are compared):

1. Detection. Every request records the ``hash_block_size``-granular prefix
   hashes of the first block its attention hit does not cover (at most one
   block of hashes). A later request probes the same region of its own hashes
   against them; the longest prefix recorded by at least ``MIN_SHARERS`` other
   requests is the exact shared prefix ``S`` (rounded down to the hash unit).
   Equal chained hashes imply equal token prefixes. Two sharers keep a
   conversation's own previous turn (one sharer) from looking like a family,
   and a match reaching the end of the block is skipped (the divergence lies
   further and the full block will be cached anyway).
2. Production. If the junction ``J = S - eagle_unit`` beats the request's
   current hit, the request becomes the junction producer: its prefill chunk
   stops at ``J + eagle_unit + hash_block_size`` so the GDN prefill-checkpoint
   export lands exactly on ``J`` (registered and kept because
   ``shared_prefix_boundary == J``), and the attention group registers a
   partial entry at ``S`` that is moved to an immutable copy before the
   producer writes past it.
3. Reuse. Later siblings hit ``S`` in attention (EAGLE drops one unit, landing
   on ``J``) and ``J`` in every Mamba group: the hit is the full shared prefix
   minus one hash unit.

Junction entries are ordinary prefix-cache entries (LRU-evictable); they are
only exempted from the superseded-tail/state drops, like Marconi junctions.
"""

import os
from collections import OrderedDict
from collections.abc import Callable

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request

logger = init_logger(__name__)

EXACT_JUNCTION_ENV = "VLLM_PREFIX_EXACT_JUNCTION"
# Minimum tokens a junction must add over the current hit to be produced.
MIN_GAIN_ENV = "VLLM_PREFIX_EXACT_JUNCTION_MIN_GAIN"
# Distinct earlier requests that must share the prefix (default 2).
MIN_SHARERS_ENV = "VLLM_PREFIX_EXACT_JUNCTION_MIN_SHARERS"
# Remembered probe hashes (one per hash unit of one block per request).
_MAX_SEEN = 1 << 17
_MAX_KEYS = 4096
_LOG_EVERY = 500
# Log the first productions individually (diagnostics; a handful per pass).
_LOG_PRODUCTIONS = 200


def exact_junction_enabled() -> bool:
    return os.getenv(EXACT_JUNCTION_ENV, "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class ExactJunctionDetector:
    """Hash-granular shared-prefix detector (scheduler side, single thread)."""

    def __init__(
        self,
        block_size: int,
        hash_block_size: int,
        eagle_unit: int,
        export_valid: Callable[[int, int, int], bool],
    ):
        """``export_valid(start, stop, junction)``: whether every Mamba group
        exports its prefill checkpoint exactly at ``junction`` for a chunk
        ``[start, stop)``; a junction it cannot export is not produced.
        """
        assert block_size % hash_block_size == 0 and block_size > hash_block_size
        self.block_size = block_size
        self.unit = hash_block_size
        self.eagle_unit = eagle_unit
        self.export_valid = export_valid
        self.min_gain = int(os.getenv(MIN_GAIN_ENV, str(hash_block_size)) or 0)
        self.min_sharers = max(1, int(os.getenv(MIN_SHARERS_ENV, "2") or 1))
        # prefix hash -> distinct ids of the first requests that recorded it
        # (at most min_sharers + 1, so one can be the probing request itself)
        self._seen: OrderedDict[BlockHash, list[str]] = OrderedDict()
        # prefix hash at a produced junction's attention entry
        self._keys: OrderedDict[BlockHash, None] = OrderedDict()
        self.num_produced = 0  # distinct junctions
        self.num_productions = 0  # producer lookups (re-productions included)
        self.num_junction_hits = 0
        self.junction_hit_tokens = 0
        self.num_lookups = 0
        logger.info(
            "%s=1: exact shared-prefix junctions at %d-token granularity "
            "(block %d, eagle unit %d, min gain %d, min sharers %d).",
            EXACT_JUNCTION_ENV,
            hash_block_size,
            block_size,
            eagle_unit,
            self.min_gain,
            self.min_sharers,
        )

    def lookup(self, request: Request, hit: int, longest: int, junction: int) -> int:
        """Return the shared-prefix boundary to use for ``request``.

        ``hit`` is the reconciled prefix-cache hit, ``longest`` the longest
        single-group (attention) hit after the EAGLE drop and ``junction`` the
        Marconi junction already detected (0 if none). Sets
        ``request.exact_junction`` when the request should produce one.
        """
        request.exact_junction = None
        self.num_lookups += 1
        hashes = request.block_hashes
        unit = self.unit
        num_prompt = request.num_prompt_tokens
        # Siblings hitting an exact junction: statistics only.
        if hit > 0 and (hit + self.eagle_unit) % unit == 0:
            idx = (hit + self.eagle_unit) // unit - 1
            if idx < len(hashes) and hashes[idx] in self._keys:
                self.num_junction_hits += 1
                self.junction_hit_tokens += hit
                if self.num_junction_hits % _LOG_EVERY == 0:
                    self.log_stats()
        # First block the attention hit does not cover; a sibling's divergence
        # point can only lie inside it (a later full block would have matched).
        raw = longest + self.eagle_unit if longest > 0 else 0
        block_end = (raw // self.block_size + 1) * self.block_size
        region_end = min(block_end, num_prompt, len(hashes) * unit)
        first = raw // unit
        last = region_end // unit  # hashes[i] covers (i + 1) * unit tokens
        if last <= first:
            return junction
        rid = request.request_id
        seen = self._seen
        best = 0
        best_owner = None
        for i in range(last - 1, first - 1, -1):
            owners = seen.get(hashes[i])
            if owners is None:
                continue
            others = [o for o in owners if o != rid]
            if len(others) >= self.min_sharers:
                best = (i + 1) * unit
                best_owner = others[0]
                break
        cap = self.min_sharers + 1
        for i in range(first, last):
            owners = seen.get(hashes[i])
            if owners is None:
                seen[hashes[i]] = [rid]
                if len(seen) > _MAX_SEEN:
                    seen.popitem(last=False)
            elif len(owners) < cap and rid not in owners:
                owners.append(rid)
        if not best or best >= block_end:
            return junction
        new_junction = best - self.eagle_unit
        # The GDN prefill export of a chunk ending at ``stop`` lands at
        # ``(stop - 1) // unit * unit - eagle_unit``, i.e. on ``new_junction``.
        stop = best + unit
        if (
            new_junction <= max(hit, junction) + self.min_gain - 1
            or new_junction % self.block_size == 0
            or stop > num_prompt
            or not self.export_valid(hit, stop, new_junction)
        ):
            return junction
        request.exact_junction = (new_junction, best, stop)
        self.num_productions += 1
        if self.num_productions <= _LOG_PRODUCTIONS:
            logger.info(
                "exact junction #%d: req=%s prompt=%d hit=%d attn=%d "
                "marconi=%d -> junction=%d (match with %s)",
                self.num_productions,
                rid,
                num_prompt,
                hit,
                raw,
                junction,
                new_junction,
                best_owner,
            )
        key = hashes[best // unit - 1]
        if key not in self._keys:
            self._keys[key] = None
            self.num_produced += 1
            if len(self._keys) > _MAX_KEYS:
                self._keys.popitem(last=False)
        return new_junction

    def log_stats(self) -> None:
        logger.info(
            "exact junction stats: produced=%d productions=%d junction_hits=%d "
            "mean_junction_hit=%.0f lookups=%d",
            self.num_produced,
            self.num_productions,
            self.num_junction_hits,
            self.junction_hit_tokens / max(self.num_junction_hits, 1),
            self.num_lookups,
        )
