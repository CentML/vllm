# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-row two-stage argmax for greedy draft tokens.

Bitwise equal to ``torch.argmax(logits, dim=-1)``: the first index of the
maximum wins, NaN counts as larger than any number (first NaN wins), and
-0.0 equals +0.0. Grid shapes depend only on the (padded) logits shape, so the
kernels are CUDA-graph safe.
"""

import torch

from vllm.triton_utils import tl, triton

_BLOCK_SIZE = 8192


@triton.jit
def _argmax_key(x):
    # Order-preserving int32 key of x (as fp32); NaN maps to the top.
    x = x.to(tl.float32) + 0.0
    bits = x.to(tl.int32, bitcast=True)
    key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return tl.where(x != x, 2147483647, key)


@triton.jit
def _block_argmax_kernel(
    logits_ptr,
    logits_stride,
    block_key_ptr,
    block_idx_ptr,
    vocab_size,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(
        logits_ptr + row * logits_stride + offs,
        mask=offs < vocab_size,
        other=float("-inf"),
    )
    key = tl.where(offs < vocab_size, _argmax_key(x), -2147483648)
    best, pos = tl.max(
        key, axis=0, return_indices=True, return_indices_tie_break_left=True
    )
    tl.store(block_key_ptr + row * num_blocks + block, best)
    tl.store(block_idx_ptr + row * num_blocks + block, block * BLOCK_SIZE + pos)


@triton.jit
def _reduce_argmax_kernel(
    block_key_ptr,
    block_idx_ptr,
    out_ptr,
    num_blocks,
    PADDED_NUM_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, PADDED_NUM_BLOCKS)
    keys = tl.load(
        block_key_ptr + row * num_blocks + offs,
        mask=offs < num_blocks,
        other=-2147483648,
    )
    _, best = tl.max(
        keys, axis=0, return_indices=True, return_indices_tie_break_left=True
    )
    idx = tl.load(block_idx_ptr + row * num_blocks + best)
    tl.store(out_ptr + row, idx.to(tl.int64))


def split_argmax(logits: torch.Tensor) -> torch.Tensor:
    """``logits.argmax(dim=-1)`` for a 2D [rows, vocab] tensor (int64 output)."""
    assert logits.ndim == 2 and logits.stride(1) == 1
    num_rows, vocab_size = logits.shape
    out = torch.empty(num_rows, dtype=torch.int64, device=logits.device)
    if num_rows == 0:
        return out
    num_blocks = triton.cdiv(vocab_size, _BLOCK_SIZE)
    block_key = torch.empty(
        num_rows, num_blocks, dtype=torch.int32, device=logits.device
    )
    block_idx = torch.empty_like(block_key)
    _block_argmax_kernel[(num_rows, num_blocks)](
        logits,
        logits.stride(0),
        block_key,
        block_idx,
        vocab_size,
        num_blocks,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    _reduce_argmax_kernel[(num_rows,)](
        block_key,
        block_idx,
        out,
        num_blocks,
        PADDED_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
        num_warps=1,
    )
    return out
