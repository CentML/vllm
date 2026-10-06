# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather the same rows of three activations in one launch.

``vllm::gather_rows3(x0, x1, x2, rows)`` returns ``(x0[rows], x1[rows],
x2[rows])`` as contiguous tensors. Each input is ``[N, D]`` or ``[N, H, D]``
with a contiguous last dimension; the row and head strides are free, so a
strided view (e.g. the attention gate read in place from the QKV projection)
is gathered without a copy first. Pure copies: the outputs are bit-identical
to ``index_select``.

Used by the MTP draft prefill (VLLM_MTP_DRAFT_PREFILL_ROWS) to keep only the
rows the speculator samples after the draft attention.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_BLOCK = 1024


@triton.jit
def _gather_rows3_kernel(
    rows_ptr,
    x0_ptr,
    x1_ptr,
    x2_ptr,
    y0_ptr,
    y1_ptr,
    y2_ptr,
    x0_row,
    x0_head,
    x1_row,
    x1_head,
    x2_row,
    x2_head,
    H0: tl.constexpr,
    D0: tl.constexpr,
    H1: tl.constexpr,
    D1: tl.constexpr,
    H2: tl.constexpr,
    D2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_row = tl.program_id(0).to(tl.int64)
    which = tl.program_id(1)
    src_row = tl.load(rows_ptr + out_row).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    if which == 0:
        for e0 in range(0, H0 * D0, BLOCK):
            e = e0 + offs
            mask = e < H0 * D0
            src = src_row * x0_row + (e // D0) * x0_head + e % D0
            v = tl.load(x0_ptr + src, mask=mask)
            tl.store(y0_ptr + out_row * (H0 * D0) + e, v, mask=mask)
    elif which == 1:
        for e0 in range(0, H1 * D1, BLOCK):
            e = e0 + offs
            mask = e < H1 * D1
            src = src_row * x1_row + (e // D1) * x1_head + e % D1
            v = tl.load(x1_ptr + src, mask=mask)
            tl.store(y1_ptr + out_row * (H1 * D1) + e, v, mask=mask)
    else:
        for e0 in range(0, H2 * D2, BLOCK):
            e = e0 + offs
            mask = e < H2 * D2
            src = src_row * x2_row + (e // D2) * x2_head + e % D2
            v = tl.load(x2_ptr + src, mask=mask)
            tl.store(y2_ptr + out_row * (H2 * D2) + e, v, mask=mask)


def _layout(x: torch.Tensor) -> tuple[int, int, int, int]:
    """(row stride, head stride, heads, head dim) of a [N, D] / [N, H, D]."""
    assert x.dim() in (2, 3) and x.stride(-1) == 1, (x.shape, x.stride())
    if x.dim() == 2:
        return x.stride(0), 0, 1, x.shape[1]
    return x.stride(0), x.stride(1), x.shape[1], x.shape[2]


def _gather_rows3_impl(
    x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outs = tuple(x.new_empty((rows.shape[0], *x.shape[1:])) for x in (x0, x1, x2))
    if rows.shape[0] == 0:
        return outs
    (r0, h0, n0, d0), (r1, h1, n1, d1), (r2, h2, n2, d2) = (
        _layout(x) for x in (x0, x1, x2)
    )
    _gather_rows3_kernel[(rows.shape[0], 3)](
        rows,
        x0,
        x1,
        x2,
        *outs,
        r0,
        h0,
        r1,
        h1,
        r2,
        h2,
        H0=n0,
        D0=d0,
        H1=n1,
        D1=d1,
        H2=n2,
        D2=d2,
        BLOCK=_BLOCK,
        num_warps=4,
    )
    return outs


def _gather_rows3_fake(
    x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(x.new_empty((rows.shape[0], *x.shape[1:])) for x in (x0, x1, x2))


direct_register_custom_op(
    op_name="gather_rows3",
    op_func=_gather_rows3_impl,
    fake_impl=_gather_rows3_fake,
)


def gather_rows3(
    x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.gather_rows3(x0, x1, x2, rows)
