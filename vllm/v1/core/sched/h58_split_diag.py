# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H58 diagnostic (log only): why a prefill turn runs in two scheduler steps.

``VLLM_H58_SPLIT_DIAG=1`` records, for every prefill chunk the scheduler
admits from the waiting queue (a turn's first chunk) or continues from the
running queue, how far the chunk was meant to go (``want``: to the end of the
request's tokens) and what shortened it:

- ``budget``: the token budget / long-prefill threshold (before the Mamba split);
- the Mamba split (``Scheduler._mamba_block_aligned_split``): the stop that
  ended the chunk (``tail_boundary``, ``next_block_boundary``,
  ``extension_stop``, ``replay_stop``, ``junction``) or ``block_align`` (the
  chunk end floored to a Mamba block), with the internal-checkpoint export
  terms (``is_mamba_prefill_checkpoint_valid``) that decided whether the
  tail could be exported inside one chunk;
- ``lookahead``: ``_reserve_prefill_lookahead`` (multi-module MTP runway).

It prints the scheduler's split configuration once, one line per split turn
(first ``VLLM_H58_SPLIT_DIAG_MAX_LINES`` per process, default 400) and a
cumulative summary every ``VLLM_H58_SPLIT_DIAG_LOG_S`` seconds (default 60).
Nothing it computes feeds back into scheduling.
"""

import os
import time
from collections import Counter
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_H58_SPLIT_DIAG", "0") == "1"
MAX_LINES = int(os.environ.get("VLLM_H58_SPLIT_DIAG_MAX_LINES", "400") or 400)
LOG_S = float(os.environ.get("VLLM_H58_SPLIT_DIAG_LOG_S", "60") or 60)

STOP_NAMES = (
    "next_block_boundary",
    "extension_stop",
    "replay_stop",
    "tail_boundary",
    "junction",
)

_counts: Counter = Counter()
_tok: Counter = Counter()
_lines = [0]
_last_log = [time.monotonic()]


def note_guard() -> None:
    """VLLM_H58_TAIL_CKPT guard fired (Scheduler._h58_tail_guard); counted in
    the summary as ``h58_guard_shortened`` (expected 0)."""
    _counts["h58_guard_shortened"] += 1


def log_config(sched: Any) -> None:
    logger.info(
        "H58 split diag on (VLLM_H58_SPLIT_DIAG=1): need_mamba_split=%s "
        "block_size=%s hash_block_size=%s use_eagle_block_drop=%s "
        "has_ckpt_blocks=%s ckpt_alignment=%s reuses_initial=%s "
        "copies_initial=%s reuse_stops=%s partial_cache_hit=%s "
        "fine_grained=%s num_prefill_lookahead=%s max_scheduled=%s "
        "long_prefill_threshold=%s; log-only",
        sched.need_mamba_block_aligned_split,
        sched.cache_config.block_size,
        sched.hash_block_size,
        sched.use_eagle_block_drop,
        sched.mamba_has_prefill_checkpoint_blocks,
        sched.mamba_prefill_checkpoint_alignment,
        sched.mamba_prefill_checkpoint_reuses_initial_block,
        sched.mamba_prefill_checkpoint_copies_initial_block,
        sched.mamba_prefill_checkpoint_reuse_stops,
        sched.mamba_partial_cache_hit,
        sched.mamba_fine_grained_prefix_cache,
        sched.num_prefill_lookahead,
        sched.max_num_scheduled_tokens,
        sched.scheduler_config.long_prefill_token_threshold,
    )


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def note_split(
    sched: Any,
    request: Any,
    start: int,
    num_new_tokens: int,
    prefill_end: int,
    checkpoint_position: int,
    use_internal_checkpoint: bool,
    stops: tuple,
    end_before_stops: int,
    end: int,
) -> None:
    """Called at the end of _mamba_block_aligned_split (past its early return)."""
    bs = sched.cache_config.block_size
    hbs = sched.hash_block_size
    align = sched.mamba_prefill_checkpoint_alignment
    init_col = (start - 1) // bs
    ckpt_col = _cdiv(start + num_new_tokens, bs) - 2
    column_ok = ckpt_col > init_col or (
        ckpt_col == init_col >= 0
        and (
            sched.mamba_prefill_checkpoint_copies_initial_block
            or (
                sched.mamba_prefill_checkpoint_reuses_initial_block
                and start % bs != 0
            )
        )
    )
    failed = []
    if not use_internal_checkpoint:
        if not sched.mamba_has_prefill_checkpoint_blocks:
            failed.append("no_ckpt_blocks")
        if start + num_new_tokens < prefill_end:
            failed.append("chunk_short_of_end")
        if align is None:
            failed.append("no_alignment")
        if start % hbs:
            failed.append("start_not_hash_aligned")
        if not column_ok:
            failed.append("column")
        if start + hbs > checkpoint_position:
            failed.append("ckpt_within_one_hash_block")
        if not start < checkpoint_position < start + num_new_tokens:
            failed.append("ckpt_outside_chunk")
        if align and (checkpoint_position - start) % align:
            failed.append("ckpt_misaligned")
    chosen = [
        name
        for name, s in zip(STOP_NAMES, stops)
        if s == end and start < s < end_before_stops
    ]
    if end < end_before_stops:
        why = "+".join(chosen) or "stop?"
    elif end_before_stops < start + num_new_tokens:
        why = "block_align"
    else:
        why = ""
    tail_ckpt = use_internal_checkpoint and not column_ok
    sched._h58_rec = (
        request,
        why,
        "use_internal=%s tail_ckpt=%s ckpt_pos=%d init_col=%d ckpt_col=%d "
        "failed=%s stops=%s" % (
            use_internal_checkpoint,
            tail_ckpt,
            checkpoint_position,
            init_col,
            ckpt_col,
            ",".join(failed) or "-",
            dict(zip(STOP_NAMES, stops)),
        ),
        start + num_new_tokens,
        end,
        failed,
        tail_ckpt,
    )


def after_schedule(
    sched: Any,
    request: Any,
    path: str,
    num_computed_tokens: int,
    num_after_budget: int,
    num_final: int,
) -> None:
    """After _reserve_prefill_lookahead, for a request still in prefill."""
    rec = getattr(sched, "_h58_rec", None)
    sched._h58_rec = None
    if rec is not None and rec[0] is not request:
        rec = None
    if num_computed_tokens >= max(request.num_prompt_tokens, request.num_tokens - 1):
        return
    want = request.num_tokens - num_computed_tokens
    reasons = []
    if num_after_budget < want:
        reasons.append("budget")
    split_end = num_computed_tokens + num_after_budget
    if rec is not None and rec[1]:
        reasons.append(rec[1])
        split_end = rec[4]
    if num_computed_tokens + num_final < split_end:
        reasons.append("lookahead")
    kind = "first" if path == "waiting" else "cont"
    _counts[kind] += 1
    _tok[kind] += num_final
    if num_final < want and num_final > 0:
        reason = "+".join(reasons) or "other"
        _counts[f"{kind}_split:{reason}"] += 1
        _counts[f"{kind}_split"] += 1
        _tok[f"{kind}_split_rest"] += want - num_final
        if rec is not None:
            for f in rec[5]:
                _counts[f"{kind}_split_ckpt_fail:{f}"] += 1
        if _lines[0] < MAX_LINES:
            _lines[0] += 1
            logger.info(
                "H58 split %s req=%s N=%d num_tokens=%d start=%d want=%d "
                "after_budget=%d scheduled=%d rest=%d reason=%s | %s",
                kind,
                request.request_id,
                request.num_prompt_tokens,
                request.num_tokens,
                num_computed_tokens,
                want,
                num_after_budget,
                num_final,
                want - num_final,
                reason,
                rec[2] if rec is not None else "no mamba split record",
            )
    elif num_final == want:
        _counts[f"{kind}_one_step"] += 1
        if rec is not None and rec[6]:
            _counts[f"{kind}_one_step_h58_tail_ckpt"] += 1
    now = time.monotonic()
    if now - _last_log[0] >= LOG_S:
        _last_log[0] = now
        log = logger.warning if _counts["h58_guard_shortened"] else logger.info
        log(
            "H58 split diag summary: h58_guard_shortened=%d %s | tokens %s",
            _counts["h58_guard_shortened"],
            dict(sorted(_counts.items())),
            dict(sorted(_tok.items())),
        )
