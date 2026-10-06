# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in suffix-only staging of ``all_token_ids`` for new requests.

On admission the V2 model runner stages the whole token history of a request
(``prefill_token_ids``) into the UVA-backed ``RequestState.all_token_ids``
buffer. The list -> pinned conversion and the staged write run on the engine
main thread and scale with the prompt length. With a prefix-cache hit, the
tokens below ``num_computed_tokens`` are never read on the GPU again: prefill
input gathering, post-update and the speculative draft read only positions
``>= num_computed_tokens``, and frequency / presence penalties count output
tokens (``>= prompt_len``) only.

When enabled, a new request stages only
``all_token_ids[start:]`` with
``start = min(num_computed_tokens, prompt_len) - margin``. Positions
``[0, start)`` of the request's row keep stale data from the previous
occupant of the slot, so every request whose features read older history
falls back to full staging: requests without sampling params, pooling,
multimodal features, prompt logprobs, repetition penalty (prompt bin mask),
bad words and thinking budget. Requests with a short hit
(``start < min start``) are staged fully as well.

Environment:
    VLLM_SUFFIX_TOKEN_STAGING=<margin>
        Enable (any non-empty value). ``margin`` is the number of tokens
        below ``min(num_computed_tokens, prompt_len)`` that are still staged.
    VLLM_SUFFIX_TOKEN_STAGING_MIN=<tokens>
        Only skip the prefix when ``start`` is at least this large
        (default 4096).
"""

import os
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData

logger = init_logger(__name__)

_margin_env = os.environ.get("VLLM_SUFFIX_TOKEN_STAGING")
SUFFIX_STAGING_ENABLED = bool(_margin_env)
_MARGIN = int(_margin_env) if _margin_env else 0
_MIN_START = int(os.environ.get("VLLM_SUFFIX_TOKEN_STAGING_MIN", "4096") or 4096)

# Process-wide counters, logged periodically (grep "suffix staging stats").
_stats: dict[str, int] = {
    "reqs": 0,
    "suffix_reqs": 0,
    "tokens_total": 0,
    "tokens_skipped": 0,
}

if SUFFIX_STAGING_ENABLED:
    logger.info_once(
        "suffix-only all_token_ids staging on (margin %d, min start %d)",
        _MARGIN,
        _MIN_START,
    )


def _suffix_start(new_req_data: "NewRequestData") -> int:
    """Start of the ``all_token_ids`` range that must be staged, or 0 (full).

    Readers of ``all_token_ids`` below ``num_computed_tokens`` exist only for
    prompt logprobs, repetition penalty (prompt bin mask; presence/frequency
    use output tokens >= prompt_len), bad words / thinking budget (scan the
    history), multimodal pruning and pooling. Any of those -> full staging.
    """
    sp = new_req_data.sampling_params
    why = None
    if (
        sp is None
        or getattr(new_req_data, "pooling_params", None) is not None
        or getattr(new_req_data, "mm_features", None)
    ):
        why = "no_sp_or_mm_or_pooling"
    elif getattr(sp, "prompt_logprobs", None) is not None:
        why = "prompt_logprobs"
    elif getattr(sp, "repetition_penalty", 1.0) not in (None, 1.0):
        why = "repetition_penalty"
    elif getattr(sp, "bad_words_token_ids", None) or getattr(sp, "bad_words", None):
        why = "bad_words"
    elif getattr(sp, "thinking_token_budget", None) is not None:
        why = "thinking_budget"
    start = min(new_req_data.num_computed_tokens, new_req_data.prompt_len) - _MARGIN
    if why is None and start < _MIN_START:
        why = "short_hit"
    if why is not None:
        _stats[why] = _stats.get(why, 0) + 1
        return 0
    return start


def get_suffix_staging_start(new_req_data: "NewRequestData") -> int:
    """Return the first ``all_token_ids`` position to stage for a new request.

    0 means "stage everything" (the default behaviour). Must only be called
    when ``SUFFIX_STAGING_ENABLED``; updates the staging counters.
    """
    prefill_token_ids = new_req_data.prefill_token_ids
    start = _suffix_start(new_req_data) if prefill_token_ids is not None else 0
    _stats["reqs"] += 1
    if _stats["reqs"] % 50 == 0:
        logger.info("suffix staging stats %s", _stats)
    if prefill_token_ids is not None:
        _stats["tokens_total"] += len(prefill_token_ids)
    if start > 0:
        _stats["suffix_reqs"] += 1
        _stats["tokens_skipped"] += start
        if _stats["suffix_reqs"] % 25 == 1:
            logger.info("suffix staging stats %s", _stats)
    return start
