# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local KV cache, scheduler side (``VLLM_DIE_LOCAL_KV=1``).

A GPU with two memory locality domains interleaves ordinary
allocations over both dies' HBM, so half of every KV read crosses the
die-to-die link. With this switch the worker backs KV block ids ``[0, s)`` with
die 0's memory and ``[s, N)`` with die 1's (``die_split``), and every decode
attention launch runs once per die on that die's SMs
(``vllm/v1/worker/die_local_kv.py``). Here the block pool keeps one free queue
per die and gives each request the blocks of one die: the die holding most of
its prefix-cache hit (when that covers at least half of its tokens and the die
is not overloaded), else the die with fewer blocks in use.

Placement only: block contents and every kernel's math are unchanged.

``VLLM_DIE_LOCAL_KV_ADDR_ORDER=1`` also hands out the blocks of each allocation
in ascending block id (address order on their die). They are the blocks the
die's LRU queue pops, so evictions and prefix-cache hits stay the same; only
which of them backs which position of the request changes. ``=0`` keeps the
stock order and logs the same page-order counts, for comparison.

``VLLM_DIE_LOCAL_KV_STRIPE=1`` (with ``VLLM_DIE_LOCAL_KV=1``) stripes each
request over both dies instead (``DieStripePolicy``): block i of its attention
block table is on die (i + s) % 2, s its start die, alternating per new request,
so every step's KV is split about evenly between the dies whatever the batch.
The worker then runs no per-die split launch; a die-aware attention kernel
(``VLLM_H211_DL_VERIFY``, ``vllm/v1/attention/ops/die_verify_attn.py``) reads
each die's blocks from that die's SMs.
"""

import os
from collections.abc import Iterable, Iterator, Sequence
from operator import attrgetter
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

ENV = "VLLM_DIE_LOCAL_KV"
# Keep a request on its prefix's die unless that die holds this many more
# in-use blocks than the other (fraction of one die's blocks).
ENV_SLACK = "VLLM_DIE_LOCAL_KV_SLACK"
# Stay on the prefix's die only if the hit covers this fraction of the tokens.
ENV_STICKY = "VLLM_DIE_LOCAL_KV_STICKY"
# Log assignment statistics every this many request-to-die assignments.
ENV_LOG_EVERY = "VLLM_DIE_LOCAL_KV_LOG_EVERY"
# Log the per-die split KV every this many steps (0: off); the worker's
# die_local_stats reads the same variable.
ENV_STATS = "VLLM_DIE_LOCAL_KV_STATS"
# The worker splits decode launches with at least this many rows (unset: 1).
ENV_MIN_ROWS = "VLLM_DIE_LOCAL_KV_MIN_ROWS"
# 1: each allocation's blocks in ascending block id per die, plus page-order
# counts (log only); 0: the counts only; unset: neither.
ENV_ADDR_ORDER = "VLLM_DIE_LOCAL_KV_ADDR_ORDER"
# 1 (with VLLM_DIE_LOCAL_KV=1): stripe each request's attention blocks over the
# dies (DieStripePolicy) instead of homing it on one die.
ENV_STRIPE = "VLLM_DIE_LOCAL_KV_STRIPE"

_block_id = attrgetter("block_id")


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def stripe_enabled() -> bool:
    """Striped placement (VLLM_DIE_LOCAL_KV_STRIPE=1 with VLLM_DIE_LOCAL_KV=1)."""
    return enabled() and os.environ.get(ENV_STRIPE, "0") == "1"


def stripe_split(tokens: int, block_size: int, start: int) -> tuple[int, int]:
    """KV tokens on die 0 and on die 1 of a request with ``tokens`` tokens whose
    block i is on die (i + start) % 2."""
    full, rem = divmod(max(tokens, 0), block_size)
    even = (full + 1) // 2 * block_size + (rem if full % 2 == 0 else 0)
    odd = max(tokens, 0) - even
    return (even, odd) if start == 0 else (odd, even)


def _stats_every() -> int:
    try:
        return max(int(os.environ.get(ENV_STATS, "0") or "0"), 0)
    except ValueError:
        return 0


def _addr_order_mode() -> int | None:
    raw = os.environ.get(ENV_ADDR_ORDER, "").strip()
    if raw in ("0", "1"):
        return int(raw)
    if raw:
        logger.warning(
            "die-local KV: %s=%r is not 0 or 1; ignored", ENV_ADDR_ORDER, raw
        )
    return None


def die_split(num_blocks: int) -> int:
    """First block id on die 1: ids ``[0, s)`` are die 0's, ``[s, N)`` die 1's."""
    return num_blocks // 2


class DieFreeBlockQueue:
    """``FreeKVCacheBlockQueue`` with one LRU queue per die.

    Removals, appends and prepends go to the block's die; pops take ``pref[0]``
    first and fall back to the other die only when it has no free block. With
    ``addr_order`` (VLLM_DIE_LOCAL_KV_ADDR_ORDER=1) the blocks a ``popleft_n``
    takes from each die are returned in ascending block id: the same blocks,
    and each die's part still in its place.
    """

    def __init__(self, blocks: list[KVCacheBlock], split: int) -> None:
        self.split = split
        self.queues = (
            FreeKVCacheBlockQueue([b for b in blocks if b.block_id < split]),
            FreeKVCacheBlockQueue([b for b in blocks if b.block_id >= split]),
        )
        self.pref = (0, 1)
        self.num_spilled = 0
        self.addr_order = False

    @property
    def num_free_blocks(self) -> int:
        return self.queues[0].num_free_blocks + self.queues[1].num_free_blocks

    def die_of(self, block: KVCacheBlock) -> int:
        return int(block.block_id >= self.split)

    def popleft(self) -> KVCacheBlock:
        for die in self.pref:
            if self.queues[die].num_free_blocks:
                self.num_spilled += die != self.pref[0]
                return self.queues[die].popleft()
        raise ValueError("No free blocks available")

    def popleft_dies(self, dies: Sequence[int]) -> list[KVCacheBlock]:
        """One block per entry of ``dies`` from that die's queue, in that order
        (VLLM_DIE_LOCAL_KV_STRIPE). A die without enough free blocks takes the
        rest from the other die, counted as spilled. With ``addr_order`` each
        die's blocks come in ascending block id."""
        assert self.num_free_blocks >= len(dies)
        want = [0, 0]
        for die in dies:
            want[die] += 1
        own = [min(want[d], self.queues[d].num_free_blocks) for d in (0, 1)]
        # lent[d]: die d's blocks that stand in for the other die's
        lent = [want[1 - d] - own[1 - d] for d in (0, 1)]
        got = [self.queues[d].popleft_n(own[d] + lent[d]) for d in (0, 1)]
        if self.addr_order:
            for part in got:
                part.sort(key=_block_id)
        self.num_spilled += lent[0] + lent[1]
        taken = [0, 0]
        spare = [own[0], own[1]]
        ret = []
        for die in dies:
            if taken[die] < own[die]:
                ret.append(got[die][taken[die]])
                taken[die] += 1
            else:
                other = 1 - die
                ret.append(got[other][spare[other]])
                spare[other] += 1
        return ret

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        assert self.num_free_blocks >= n
        first, second = self.pref
        k = min(n, self.queues[first].num_free_blocks)
        ret = self.queues[first].popleft_n(k)
        if self.addr_order:
            ret.sort(key=_block_id)
        if k < n:
            self.num_spilled += n - k
            rest = self.queues[second].popleft_n(n - k)
            if self.addr_order:
                rest.sort(key=_block_id)
            ret += rest
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        self.queues[self.die_of(block)].remove(block)

    def append(self, block: KVCacheBlock) -> None:
        self.queues[self.die_of(block)].append(block)

    def _by_die(
        self, blocks: Iterable[KVCacheBlock]
    ) -> tuple[list[KVCacheBlock], list[KVCacheBlock]]:
        split: tuple[list[KVCacheBlock], list[KVCacheBlock]] = ([], [])
        for block in blocks:
            split[self.die_of(block)].append(block)
        return split

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        for queue, part in zip(self.queues, self._by_die(blocks)):
            queue.prepend_n(part)

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        for queue, part in zip(self.queues, self._by_die(blocks)):
            queue.append_n(part)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        return self.queues[0].get_all_free_blocks() + (
            self.queues[1].get_all_free_blocks()
        )

    def iter_blocks_after(self, cursor: KVCacheBlock | None) -> Iterator[KVCacheBlock]:
        raise NotImplementedError(
            f"{ENV}=1 does not support the simple KV offload connector"
        )


class DieKVStats:
    """Per-die split KV, from the scheduler's view (log only).

    Each step's decode rows (requests whose prompt is computed) and their KV
    tokens are summed per die; a row's die is its request's (the die of its
    newest block unless an allocation spilled; striped: its start die, and its
    KV is split between the dies block by block). Over the steps with at least
    ``min_rows`` decode rows (those the worker splits per die), a summary of
    die-max / die-mean is logged every ``every`` steps with decode rows.
    """

    def __init__(self, every: int, min_rows: int) -> None:
        self.every = every
        self.min_rows = min_rows
        # request id -> (die, KV tokens on die 0, KV tokens on die 1)
        self.rows: dict[str, tuple[int, int, int]] = {}
        self.done = 0
        self._reset()

    def _reset(self) -> None:
        self.steps = 0
        self.kv_ratio: list[float] = []
        self.rows_ratio: list[float] = []
        self.kv_max = 0
        self.kv_mean = 0.0
        self.die_rows = [0, 0]
        self.die_kv = [0, 0]

    def drop(self, request_id: str) -> None:
        self.rows.pop(request_id, None)

    def step(self) -> None:
        """Close the step that ends (called when the next one starts)."""
        if not self.rows:
            return
        rows, kv = [0, 0], [0, 0]
        for die, kv0, kv1 in self.rows.values():
            rows[die] += 1
            kv[0] += kv0
            kv[1] += kv1
        self.rows.clear()
        self.steps += 1
        if rows[0] + rows[1] >= self.min_rows:
            mean = (kv[0] + kv[1]) / 2
            self.kv_ratio.append(max(kv) / mean if mean else 1.0)
            self.rows_ratio.append(2 * max(rows) / (rows[0] + rows[1]))
            self.kv_max += max(kv)
            self.kv_mean += mean
            for die in (0, 1):
                self.die_rows[die] += rows[die]
                self.die_kv[die] += kv[die]
        if self.steps >= self.every:
            self._log()

    def _log(self) -> None:
        first, last = self.done + 1, self.done + self.steps
        self.done = last
        n = len(self.kv_ratio)
        if not n:
            logger.info(
                "die-local KV stats [die-kv] %d-%d: no step with >= %d decode rows",
                first,
                last,
                self.min_rows,
            )
            self._reset()
            return
        kv, rows = sorted(self.kv_ratio), sorted(self.rows_ratio)
        logger.info(
            "die-local KV stats [die-kv] %d-%d: %d of %d steps with >= %d decode "
            "rows; split KV die-max/die-mean mean %.3f p50 %.3f p90 %.3f max %.3f, "
            "KV-weighted %.3f; rows die-max/die-mean mean %.3f p90 %.3f; per step "
            "die 0 / die 1: rows %.1f / %.1f, KV tokens %.0f / %.0f",
            first,
            last,
            n,
            self.steps,
            self.min_rows,
            sum(kv) / n,
            kv[n // 2],
            kv[min(n - 1, n * 9 // 10)],
            kv[-1],
            self.kv_max / self.kv_mean if self.kv_mean else 1.0,
            sum(rows) / n,
            rows[min(n - 1, n * 9 // 10)],
            self.die_rows[0] / n,
            self.die_rows[1] / n,
            self.die_kv[0] / n,
            self.die_kv[1] / n,
        )
        self._reset()


class DieAddrStats:
    """Page order of the attention block tables (VLLM_DIE_LOCAL_KV_ADDR_ORDER,
    log only).

    For every allocation that appends blocks to a request's attention block
    table, each new adjacent pair of block ids is counted: the pair from the
    request's previous block to the first new one (its edge), and the pairs
    inside the new blocks. A pair is ascending when the second id is larger,
    contiguous when it is the next id. Cumulative counts are logged every
    ``every`` allocations.
    """

    def __init__(self, mode: int, every: int) -> None:
        self.mode = mode
        self.every = max(every, 1)
        self.allocs = 0
        # [pairs, ascending, contiguous] inside allocations and at their edge
        self.inside = [0, 0, 0]
        self.edge = [0, 0, 0]

    @staticmethod
    def _count(acc: list[int], a: KVCacheBlock, b: KVCacheBlock) -> None:
        acc[0] += 1
        acc[1] += b.block_id > a.block_id
        acc[2] += b.block_id == a.block_id + 1

    def note(self, blocks: Sequence[KVCacheBlock], start: int) -> None:
        """``blocks[start:]`` were just appended to a request's block table."""
        if start >= len(blocks):
            return
        prev = blocks[start - 1] if start > 0 else None
        if prev is not None and not prev.is_null and not blocks[start].is_null:
            self._count(self.edge, prev, blocks[start])
        for a, b in zip(blocks[start:], blocks[start + 1 :]):
            if not a.is_null and not b.is_null:
                self._count(self.inside, a, b)
        self.allocs += 1
        if self.allocs % self.every == 0:
            self.log()

    def log(self) -> None:
        pairs = self.inside[0] + self.edge[0]
        asc = self.inside[1] + self.edge[1]
        contig = self.inside[2] + self.edge[2]
        logger.info(
            "die-local KV page order (%s=%d): %d allocations, %d new block pairs: "
            "ascending %.4f, contiguous %.4f; inside allocations %d pairs "
            "ascending %.4f; at the request's previous block %d pairs "
            "ascending %.4f",
            ENV_ADDR_ORDER,
            self.mode,
            self.allocs,
            pairs,
            asc / pairs if pairs else 0.0,
            contig / pairs if pairs else 0.0,
            self.inside[0],
            self.inside[1] / self.inside[0] if self.inside[0] else 0.0,
            self.edge[0],
            self.edge[1] / self.edge[0] if self.edge[0] else 0.0,
        )


class DieLocalPolicy:
    """Request-to-die assignment for a block pool with a ``DieFreeBlockQueue``."""

    # Striped placement (DieStripePolicy): attention managers pass dies.
    striped = False

    def __init__(self, queue: DieFreeBlockQueue, num_blocks: int) -> None:
        self.queue = queue
        self.sizes = (queue.split, num_blocks - queue.split)
        self.slack = int(float(os.environ.get(ENV_SLACK, "0.05")) * self.sizes[0])
        self.sticky = float(os.environ.get(ENV_STICKY, "0.5"))
        self.req_die: dict[str, int] = {}
        self.stats = {"new": [0, 0], "sticky": 0, "moved": 0}
        self.log_every = int(os.environ.get(ENV_LOG_EVERY, "20000"))
        self._assigned = 0
        every = _stats_every()
        self.kv_stats = (
            DieKVStats(every, int(os.environ.get(ENV_MIN_ROWS, "") or 1))
            if every
            else None
        )
        mode = _addr_order_mode()
        queue.addr_order = mode == 1
        self.addr_stats = (
            None if mode is None else DieAddrStats(mode, self.log_every or 20000)
        )
        if self.addr_stats is not None:
            logger.info(
                "die-local KV: %s=%d: %s; page-order counts every %d allocations",
                ENV_ADDR_ORDER,
                self.addr_stats.mode,
                "each allocation's blocks in ascending block id per die (the "
                "blocks the LRU queues pop)"
                if queue.addr_order
                else "stock allocation order",
                self.addr_stats.every,
            )

    def used(self, die: int) -> int:
        return self.sizes[die] - self.queue.queues[die].num_free_blocks

    def _least_used(self) -> int:
        return int(self.used(1) < self.used(0))

    def assign(
        self,
        request: "Request",
        hit_blocks: Sequence[Sequence[KVCacheBlock]],
        block_size: int,
    ) -> int:
        """The die of ``request`` (assigned on its first allocation)."""
        die = self.req_die.get(request.request_id)
        if die is not None:
            return die
        # The group with the most hit blocks (full attention in a hybrid model)
        # decides; the other groups hit a few state blocks.
        counts = [0, 0]
        for group in hit_blocks:
            group_counts = [0, 0]
            for block in group:
                if not block.is_null:
                    group_counts[self.queue.die_of(block)] += 1
            if sum(group_counts) > sum(counts):
                counts = group_counts
        hit_die = int(counts[1] > counts[0])
        hit_tokens = max(counts) * block_size
        if counts[0] != counts[1] and hit_tokens >= self.sticky * request.num_tokens:
            if self.used(hit_die) - self.used(1 - hit_die) <= self.slack:
                die = hit_die
                self.stats["sticky"] += 1
            else:
                die = 1 - hit_die
                self.stats["moved"] += 1
        else:
            die = self._least_used()
            self.stats["new"][die] += 1
        self.req_die[request.request_id] = die
        self._assigned += 1
        if self.log_every > 0 and self._assigned % self.log_every == 0:
            logger.info(
                "die-local KV: %d assignments %s, blocks in use per die %d / %d, "
                "spilled block allocations %d",
                self._assigned,
                self.stats,
                self.used(0),
                self.used(1),
                self.queue.num_spilled,
            )
        return die

    def prefer(self, die: int | None) -> None:
        """Pop order for the following allocations (None: least-used die first)."""
        if die is None:
            die = self._least_used()
        self.queue.pref = (die, 1 - die)

    def note(self, request: "Request", die: int, kv_tokens: int) -> None:
        """A request scheduled this step with ``kv_tokens`` of KV after it
        (VLLM_DIE_LOCAL_KV_STATS)."""
        stats = self.kv_stats
        if (
            stats is not None
            and request.num_computed_tokens >= request.num_prompt_tokens
        ):
            stats.rows[request.request_id] = (
                die,
                kv_tokens if die == 0 else 0,
                kv_tokens if die == 1 else 0,
            )

    def step(self) -> None:
        """A scheduler step starts."""
        if self.kv_stats is not None:
            self.kv_stats.step()

    def forget(self, request_id: str) -> None:
        self.req_die.pop(request_id, None)
        if self.kv_stats is not None:
            self.kv_stats.drop(request_id)


class DieStripePolicy(DieLocalPolicy):
    """Striped placement for a block pool with a ``DieFreeBlockQueue``
    (VLLM_DIE_LOCAL_KV_STRIPE=1).

    Block i of a request's attention block table is on die (i + s) % 2, where
    s, the request's start die, is the die of its first block: a request whose
    first block is a prefix-cache hit keeps that block's die, and any other
    request takes the next die of an alternation. A request then holds at most
    one block more on one die than on the other, and the start dies alternate,
    so the dies' KV stays about even whatever the lengths. The attention
    managers pass the dies of their new blocks to the pool (``dies_for``);
    other groups' blocks (Mamba state) come from the die with fewer blocks in
    use. A die without a free block lends one of the other's (counted as
    spilled); the kernel still reads it correctly, from the far die.
    """

    striped = True

    def __init__(self, queue: DieFreeBlockQueue, num_blocks: int) -> None:
        super().__init__(queue, num_blocks)
        self._next_start = 0
        self.block_size = 1
        # start dies of new requests and of prefix-cache hits; blocks per die
        self.stripe_stats = {"new": [0, 0], "hit": [0, 0], "blocks": [0, 0]}
        logger.info(
            "die-local KV: %s=1: block i of a request's attention block table on "
            "die (i + start die) %% 2, the start die alternating per new request "
            "(a prefix-cache hit keeps its first block's die); other groups' "
            "blocks from the die with fewer blocks in use; no per-die split "
            "launches",
            ENV_STRIPE,
        )

    def _alternate(self) -> int:
        die = self._next_start
        self._next_start ^= 1
        return die

    def assign(
        self,
        request: "Request",
        hit_blocks: Sequence[Sequence[KVCacheBlock]],
        block_size: int,
    ) -> int:
        """The start die of ``request`` (fixed on its first allocation)."""
        die = self.req_die.get(request.request_id)
        if die is not None:
            return die
        self.block_size = max(block_size, 1)
        # The group with the most hit blocks (full attention in a hybrid model)
        # holds the request's first block.
        first, best = None, 0
        for group in hit_blocks:
            n = sum(1 for block in group if not block.is_null)
            if n > best and not group[0].is_null:
                first, best = group[0], n
        if first is not None:
            die = self.queue.die_of(first)
            self.stripe_stats["hit"][die] += 1
        else:
            die = self._alternate()
            self.stripe_stats["new"][die] += 1
        self.req_die[request.request_id] = die
        self._assigned += 1
        if self.log_every > 0 and self._assigned % self.log_every == 0:
            self.log_stripe()
        return die

    def dies_for(
        self,
        request_id: str,
        req_blocks: Sequence[KVCacheBlock],
        start: int,
        n: int,
    ) -> list[int]:
        """The dies of ``n`` new blocks at positions ``start``.. of a request's
        attention block table ``req_blocks``: block i on die (i + s) % 2, s the
        die of its first block (before it has one, its assigned start die)."""
        if req_blocks and not req_blocks[0].is_null:
            s = self.queue.die_of(req_blocks[0])
        else:
            s = self.req_die.get(request_id)
            if s is None:  # no assign() for this request: alternate
                s = self.req_die[request_id] = self._alternate()
                self.stripe_stats["new"][s] += 1
        dies = [(start + i + s) & 1 for i in range(n)]
        placed = self.stripe_stats["blocks"]
        for die in dies:
            placed[die] += 1
        return dies

    def prefer(self, die: int | None) -> None:
        """Blocks allocated without dies (other groups): least-used die first."""
        super().prefer(None)

    def note(self, request: "Request", die: int, kv_tokens: int) -> None:
        stats = self.kv_stats
        if (
            stats is not None
            and request.num_computed_tokens >= request.num_prompt_tokens
        ):
            kv0, kv1 = stripe_split(kv_tokens, self.block_size, die)
            stats.rows[request.request_id] = (die, kv0, kv1)

    def log_stripe(self) -> None:
        st = self.stripe_stats
        logger.info(
            "die-local KV stripe: %d requests, start die 0 / 1: new %d / %d, from "
            "a prefix-cache hit %d / %d; attention blocks placed on die 0 / 1: "
            "%d / %d; spilled block allocations %d; blocks in use per die %d / %d",
            self._assigned,
            st["new"][0],
            st["new"][1],
            st["hit"][0],
            st["hit"][1],
            st["blocks"][0],
            st["blocks"][1],
            self.queue.num_spilled,
            self.used(0),
            self.used(1),
        )
