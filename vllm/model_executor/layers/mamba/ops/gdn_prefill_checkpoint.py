# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy kernels of the GDN internal prefill checkpoint, one launch each.

``QwenGatedDeltaNetAttention._chunk_prefill_with_checkpoint`` and
``_store_conv_checkpoint`` run in every GDN layer of a step that exports a
checkpoint. Built from ATen indexing they cost ~14 small kernels per layer
(arange, add, gathers, index_put, 5 ``index_select``); these kernels do the
same copies with one launch per copy step and no temporaries. All of them only
move values, so the results are bitwise identical to the ATen indexing.
"""

import torch

from vllm.triton_utils import tl, triton


# The index arguments are views at arbitrary offsets into packed per-step
# uploads: no alignment specialization, so one compiled variant serves all.
@triton.jit(do_not_specialize_on_alignment=["token_ptr", "slot_ptr"])
def _store_conv_checkpoint_kernel(
    x_ptr,
    state_ptr,
    token_ptr,
    slot_ptr,
    dim,
    stride_x_tok,
    stride_s_slot,
    stride_s_dim,
    stride_s_col,
    STATE_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    ckpt = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < dim
    token = tl.load(token_ptr + ckpt).to(tl.int64)
    slot = tl.load(slot_ptr + ckpt).to(tl.int64)
    dst = state_ptr + slot * stride_s_slot + offs * stride_s_dim
    for j in tl.static_range(STATE_LEN):
        row = token - STATE_LEN + j
        val = tl.load(x_ptr + row * stride_x_tok + offs, mask=mask)
        tl.store(
            dst + j * stride_s_col,
            val.to(state_ptr.dtype.element_ty),
            mask=mask,
        )


def store_conv_checkpoint(
    conv_input: torch.Tensor,
    conv_state: torch.Tensor,
    state_len: int,
    token_indices: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """``conv_state[slots[c], :, j] = conv_input[token_indices[c] - state_len + j]``
    for ``j < state_len``: the ``state_len`` conv inputs before each checkpoint
    token, newest last.

    ``conv_input``: ``[tokens, dim]`` with unit stride along ``dim``;
    ``conv_state``: the ``[slots, dim, cols]`` view of the conv pool (any
    strides, ``cols >= state_len``). Every ``token_indices[c] >= state_len``.
    """
    num_ckpt = token_indices.numel()
    if num_ckpt == 0:
        return
    assert conv_input.dim() == 2 and conv_input.stride(1) == 1
    assert conv_state.dim() == 3 and conv_state.size(1) == conv_input.size(1)
    assert conv_state.size(2) >= state_len
    assert slots.numel() == num_ckpt
    dim = conv_input.size(1)
    block_d = 1024
    _store_conv_checkpoint_kernel[(num_ckpt, triton.cdiv(dim, block_d))](
        conv_input,
        conv_state,
        token_indices,
        slots,
        dim,
        conv_input.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        STATE_LEN=state_len,
        BLOCK_D=block_d,
        num_warps=4,
    )


@triton.jit(do_not_specialize_on_alignment=["dst_ptr", "src_ptr"])
def _copy_state_rows_kernel(
    pool_ptr,
    dst_ptr,
    src_ptr,
    stride_slot,
    ROW_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    dst = tl.load(dst_ptr + i).to(tl.int64)
    src = tl.load(src_ptr + i).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < ROW_NUMEL
    val = tl.load(pool_ptr + src * stride_slot + offs, mask=mask)
    tl.store(pool_ptr + dst * stride_slot + offs, val, mask=mask)


def copy_state_rows(
    pool: torch.Tensor, dst_slots: torch.Tensor, src_slots: torch.Tensor
) -> None:
    """``pool[dst_slots] = pool[src_slots]`` without a gathered temporary. The
    source and destination slot sets must not overlap. Pool rows may be padded
    (``stride(0)`` larger than a row) but each row must be contiguous.
    """
    num_rows = dst_slots.numel()
    if num_rows == 0:
        return
    assert src_slots.numel() == num_rows
    row_numel = pool[0].numel()
    assert pool[0].is_contiguous()
    block = 4096
    _copy_state_rows_kernel[(num_rows, triton.cdiv(row_numel, block))](
        pool,
        dst_slots,
        src_slots,
        pool.stride(0),
        ROW_NUMEL=row_numel,
        BLOCK=block,
        num_warps=4,
    )


@triton.jit
def _copy_row(
    src_ptr, dst_ptr, src_row, dst_row, ROW: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    mask = offs < ROW
    val = tl.load(src_ptr + src_row * ROW + offs, mask=mask)
    tl.store(dst_ptr + dst_row * ROW + offs, val, mask=mask)


@triton.jit(do_not_specialize_on_alignment=["idx_ptr"])
def _gather_prefill_rows_kernel(
    idx_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    g_out_ptr,
    beta_out_ptr,
    QK_ROW: tl.constexpr,
    V_ROW: tl.constexpr,
    G_ROW: tl.constexpr,
    QK_BLOCK: tl.constexpr,
    V_BLOCK: tl.constexpr,
    G_BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    src = tl.load(idx_ptr + i).to(tl.int64)
    _copy_row(q_ptr, q_out_ptr, src, i, QK_ROW, QK_BLOCK)
    _copy_row(k_ptr, k_out_ptr, src, i, QK_ROW, QK_BLOCK)
    _copy_row(v_ptr, v_out_ptr, src, i, V_ROW, V_BLOCK)
    _copy_row(g_ptr, g_out_ptr, src, i, G_ROW, G_BLOCK)
    _copy_row(beta_ptr, beta_out_ptr, src, i, G_ROW, G_BLOCK)


def gather_prefill_rows(
    token_indices: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``t.index_select(1, token_indices)`` for the five ``[1, T, ...]``
    chunk-kernel inputs, in one launch. Inputs must be contiguous.
    """
    for t in (q, k, v, g, beta):
        assert t.is_contiguous() and t.size(0) == 1
    assert k.shape == q.shape and beta.shape == g.shape
    n = token_indices.numel()
    outs = tuple(
        torch.empty((1, n, *t.shape[2:]), dtype=t.dtype, device=t.device)
        for t in (q, k, v, g, beta)
    )
    if n == 0:
        return outs  # type: ignore[return-value]
    q_out, k_out, v_out, g_out, beta_out = outs
    qk_row, v_row, g_row = q[0, 0].numel(), v[0, 0].numel(), g[0, 0].numel()
    _gather_prefill_rows_kernel[(n,)](
        token_indices,
        q,
        k,
        v,
        g,
        beta,
        q_out,
        k_out,
        v_out,
        g_out,
        beta_out,
        QK_ROW=qk_row,
        V_ROW=v_row,
        G_ROW=g_row,
        QK_BLOCK=triton.next_power_of_2(qk_row),
        V_BLOCK=triton.next_power_of_2(v_row),
        G_BLOCK=triton.next_power_of_2(g_row),
        num_warps=4,
    )
    return outs  # type: ignore[return-value]
