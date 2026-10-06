# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""F122: in-step Mamba (GDN) prefill checkpoints for mamba_cache_mode="align".

With a prefix-match unit finer than the Mamba block (e.g. prefix-match-unit 32
and a 2208-token GDN block), the align-mode chunk splitter ends a prompt's
prefill chunk at every position whose recurrent state must be cached:

  * the next block boundary b = (start // B + 1) * B when the chunk starts
    mid-block (the state at b is the full-block-k prefix-cache entry and lives
    in block column k, the column that holds the chunk's initial state);
  * the prompt's partial-tail hash boundary T (the partial-tail prefix-cache
    entry the next turn of a conversation resumes from).

So a typical warm turn runs 2-3 mixed steps before its first token:
[start, b) [b, T) [T, P). With VLLM_MAMBA_TAIL_CKPT=1 the scheduler runs
through b and T in one step and the states at b and T are written inside that
step (worker side: gdn_inline_ckpt.py):

  * b: the GDN layer re-runs the conv + chunked delta rule over [start, b) in
    place on block column k (which still holds the state at start: the
    align-mode pre-copy moved it to the running column) -> identical to the
    split flow's chunk ending at b;
  * T: the KV cache manager reserves one extra block D per GDN KV-cache group,
    queues a page copy (block holding the state at start -> D) with the
    step's CoW copies, and keys D at T (the same partial-tail entry the split
    flow registers); the GDN layer re-runs [start, T) in place on D.

The checkpoint states and conv windows are therefore the split flow's
(same kernels, same chunk grid from start, same initial state). Only the
step's batch composition changes (float-order class, like any scheduling
change). Default off; with the flag off nothing here is reached.

Env:
  VLLM_MAMBA_TAIL_CKPT=1                 enable (scheduler + manager + worker)
  VLLM_MAMBA_TAIL_CKPT_BLOCK=1           also merge the block-boundary split (default 1)
  VLLM_MAMBA_TAIL_CKPT_TAIL=1            merge the partial-tail split (default 1)
  VLLM_MAMBA_TAIL_CKPT_MAX_PER_STEP=N    merged requests per step (default 4)
  VLLM_MAMBA_TAIL_CKPT_MAX_RUNNING=N     no merges while more than N requests run (0 = off)
  VLLM_MAMBA_TAIL_CKPT_LOG_EVERY=N       scheduler stats log period in merges (default 2000)
  VLLM_MAMBA_TAIL_CKPT_VERIFY=N          worker: re-run the split flow for the first N
                                         merged rows and compare bitwise (gdn_inline_ckpt)
"""

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


ENABLED = os.environ.get("VLLM_MAMBA_TAIL_CKPT", "0") == "1"
MERGE_BLOCK = ENABLED and os.environ.get("VLLM_MAMBA_TAIL_CKPT_BLOCK", "1") == "1"
MERGE_TAIL = ENABLED and os.environ.get("VLLM_MAMBA_TAIL_CKPT_TAIL", "1") == "1"
MAX_PER_STEP = _env_int("VLLM_MAMBA_TAIL_CKPT_MAX_PER_STEP", 4)
MAX_RUNNING = _env_int("VLLM_MAMBA_TAIL_CKPT_MAX_RUNNING", 0)
LOG_EVERY = max(1, _env_int("VLLM_MAMBA_TAIL_CKPT_LOG_EVERY", 2000))

# kinds of an in-step checkpoint
KIND_BLOCK = 0  # state at a block boundary -> the block-table column that held the
#                 chunk's initial state (written in place)
KIND_TAIL = 1  # state at the partial-tail boundary -> a reserved side block
KIND_RUN = 2  # the running block (state at the chunk end; worker VERIFY only)
KIND_SPLIT = 3  # a block boundary whose state is not cached: no write, but the
#                 worker splits a later checkpoint's replay there (split-flow grid)


def tail_boundary(num_prompt_tokens: int, hash_block_size: int, eagle_drop: bool) -> int:
    """The prompt's partial-tail hash boundary, exactly as the align-mode
    splitter (Scheduler._mamba_block_aligned_split) and the Mamba manager
    (_cache_partial_tail_block) compute it."""
    t = num_prompt_tokens // hash_block_size * hash_block_size
    if eagle_drop:
        t = max(t - hash_block_size, 0)
    return t


def block_ckpt_position(start: int, end: int, block_size: int) -> int:
    """The block boundary a chunk [start, end) runs through without stopping
    (the state there belongs in column start // block_size), or 0."""
    if start % block_size == 0:
        return 0
    b = (start // block_size + 1) * block_size
    return b if start < b < end else 0
