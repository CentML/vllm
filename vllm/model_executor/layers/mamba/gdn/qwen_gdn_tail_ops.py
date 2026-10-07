# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels and launch configs for the tail of the Qwen GDN core op.

- ``gdn_gated_norm_mxfp8``: the gated RMSNorm of the GDN output fused with the
  MXFP8 activation quant of ``out_proj``. It writes e4m3 values and the
  F8_128x4-swizzled UE8M0 scales that FlashInfer's
  ``mxfp8_quantize(normed, is_sf_swizzled_layout=True)`` produces from the bf16
  normalized output, so the MXFP8 GEMM consumes them unchanged. Rows the core
  kernels never wrote (FULL-graph and piecewise padding) are zero-filled
  without being read.
- ``gdn_norm_launch_config``: launch config of the FLA gated RMSNorm for the
  ``[T, HV * V]`` mixed-batch launch.
- ``zero_fresh_state_rows``: zeroes the SSM pool rows of prefill sequences
  without an initial state, on the device, before FlashInfer updates the pool
  in place through ``state_indices``.
"""

import math
import weakref

import torch

from vllm.model_executor.layers.fusion.mxfp8_pdl import mxfp8_producer_early_trigger
from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
    MXFP8_BLOCK,
    mxfp8_quantize_row,
    mxfp8_store_swizzled_scales,
)
from vllm.model_executor.layers.mamba.ops.gdn_host_trim import (
    GDN_HOST_TRIM,
    arch_support_pdl,
    launcher,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
    calc_rows_per_block,
)
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_rows", "norm_lo", "norm_hi", "num_valid"])
def _gdn_gated_norm_mxfp8_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    q_ptr,
    scale_ptr,
    valid_ptr,
    num_rows,
    norm_lo,
    norm_hi,
    num_valid,
    stride_x,
    stride_z,
    eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VALID_FROM_PTR: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
    STORE_8B: tl.constexpr,
):
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        if EARLY_TRIGGER:
            tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    K: tl.constexpr = HEADS * HEAD_DIM
    BLOCK: tl.constexpr = BLOCK_H * HEAD_DIM
    heads = tl.arange(0, BLOCK_H)
    cols = tl.arange(0, HEAD_DIM)
    offs = heads[:, None] * HEAD_DIM + cols[None, :]
    n_valid = tl.load(valid_ptr).to(tl.int64) if VALID_FROM_PTR else num_valid
    flat = tl.arange(0, BLOCK)
    sf = tl.zeros((BLOCK // 32,), dtype=tl.uint32)
    if (row < num_rows) & (row < n_valid):
        # x and z are loaded up front and the norm-or-not choice is a select,
        # so both loads are in flight together.
        normed = (row >= norm_lo) & (row < norm_hi)
        mask = heads[:, None] < HEADS
        x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        z = tl.load(z_ptr + row * stride_z + offs, mask=mask & normed, other=0.0)
        z = z.to(tl.float32)
        # Same per-(token, head) math and op order as the FLA
        # layer_norm_fwd_kernel (RMS, norm before gate).
        var = tl.sum(x * x, axis=1) / HEAD_DIM
        rstd = tl.rsqrt(var + eps)
        w = tl.load(w_ptr + cols).to(tl.float32)
        y = x * rstd[:, None]
        y = y * w[None, :]
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)
        # Quantize the bf16 value the standalone norm would store; rows
        # normalized upstream (the fused CUDA MTP kernel) are quantized as is.
        y = tl.where(normed, y.to(tl.bfloat16).to(tl.float32), x)
        quantized, sf = mxfp8_quantize_row(tl.reshape(y, (BLOCK,)), BLOCK)
        if STORE_8B:
            # Address the e4m3 row as 8-byte chunks: a 16 B/thread e4m3 store
            # would make Triton move the row through shared memory.
            c = tl.arange(0, BLOCK // 8)[:, None] * 8 + tl.arange(0, 8)[None, :]
            tl.store(
                q_ptr + row * K + c,
                tl.reshape(quantized, (BLOCK // 8, 8)),
                mask=c < K,
            )
        else:
            tl.store(q_ptr + row * K + flat, quantized, mask=flat < K)
    elif row < num_rows:
        # Rows >= num_valid are not read: zero values (and zero scales).
        tl.store(q_ptr + row * K + flat, tl.zeros((BLOCK,), tl.float32), mask=flat < K)
    # Rows past num_rows only fill the 128-row scale padding (with zeros).
    mxfp8_store_swizzled_scales(scale_ptr, row, tl.arange(0, BLOCK // 32), sf, K // 32)


_norm_launch = launcher(_gdn_gated_norm_mxfp8_kernel)


def gdn_mxfp8_scale_numel(num_tokens: int, hidden: int) -> int:
    """Bytes of the flat F8_128x4 UE8M0 scale buffer for [num_tokens, hidden]."""
    return (
        triton.cdiv(num_tokens, 128) * 128 * triton.cdiv(hidden // MXFP8_BLOCK, 4) * 4
    )


def _gdn_norm_mxfp8_num_warps(
    num_norm_rows: int, block_h: int, device: torch.device
) -> int:
    """One program per token row ([BLOCK_H, V] tile). The per-head fp32 sum of
    squares must reduce in the same order as the layer_norm_fwd launch it
    replaces: when that launch uses 1-row tiles (T <= 2 * SMs), its warp holds
    one row with lanes replicated, which only one head per warp reproduces;
    with >= 2-row tiles any >= 2 heads per warp matches. 8 warps: fastest on
    VR at T = 2144..8192 (sweep 2/4/8/16).
    """
    if num_norm_rows > 0 and calc_rows_per_block(num_norm_rows, device) == 1:
        return min(block_h, 32)
    return min(8, max(block_h // 2, 1))


def gdn_gated_norm_mxfp8(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    activation: str,
    out_q: torch.Tensor,
    out_scale: torch.Tensor,
    norm_rows: tuple[int, int],
    num_valid: int | torch.Tensor,
) -> None:
    """Gated RMSNorm (per head, norm before gate) + swizzled MXFP8 of the output.

    ``x`` is ``[T, HV, V]`` (the pre-norm core output), ``z`` the output gate of
    the same shape (row-strided view of the projection is fine) and ``weight``
    the ``[V]`` norm weight shared by all heads. Rows ``< num_valid`` are
    quantized into ``out_q`` ``[T, HV * V]`` e4m3 and ``out_scale`` (flat
    F8_128x4); of those, rows in ``[norm_rows[0], norm_rows[1])`` are
    normalized first and the others are taken as already normalized. Rows
    ``>= num_valid`` get zero values and scales and are not read. ``num_valid``
    may be a 1-element int tensor on the device (FULL-graph replay, where the
    real row count is only known there).
    """
    num_rows, heads, head_dim = x.shape
    hidden = heads * head_dim
    if not GDN_HOST_TRIM:
        assert z.shape == x.shape
        assert x.stride(-1) == 1 and z.stride(-1) == 1
        assert x.stride(1) == head_dim and z.stride(1) == head_dim
        assert weight.shape == (head_dim,) and weight.is_contiguous()
        assert (
            head_dim == triton.next_power_of_2(head_dim) and hidden % MXFP8_BLOCK == 0
        )
        assert out_q.shape == (num_rows, hidden) and out_q.is_contiguous()
        assert out_q.dtype == torch.float8_e4m3fn
        scale_numel = gdn_mxfp8_scale_numel(num_rows, hidden)
        assert out_scale.numel() == scale_numel and out_scale.dtype == torch.uint8
    valid_from_ptr = isinstance(num_valid, torch.Tensor)
    if num_rows == 0:
        return
    padded_rows = triton.cdiv(num_rows, 128) * 128
    block_h = triton.next_power_of_2(heads)
    num_warps = _gdn_norm_mxfp8_num_warps(
        norm_rows[1] - norm_rows[0], block_h, x.device
    )
    # PDL below 4096 rows, as the fused RMSNorm -> MXFP8 producer.
    launch_pdl = num_rows < 4096 and (
        arch_support_pdl() if GDN_HOST_TRIM else current_platform.is_arch_support_pdl()
    )
    _norm_launch[(padded_rows,)](
        x,
        z,
        weight,
        out_q,
        out_scale,
        num_valid if valid_from_ptr else out_scale,
        num_rows,
        norm_rows[0],
        norm_rows[1],
        0 if valid_from_ptr else num_valid,
        x.stride(0),
        z.stride(0),
        eps,
        HEADS=heads,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        VALID_FROM_PTR=valid_from_ptr,
        ACTIVATION=activation,
        LAUNCH_PDL=launch_pdl,
        EARLY_TRIGGER=launch_pdl and mxfp8_producer_early_trigger(),
        launch_pdl=launch_pdl,
        STORE_8B=block_h * head_dim >= 16 * 32 * num_warps,
        num_warps=num_warps,
    )


def gdn_norm_launch_config(num_rows: int, device: torch.device) -> tuple[int, int]:
    """(rows_per_block, num_warps) of the FLA gated RMSNorm for the mixed-batch
    ``[T, HV * V]`` launch with one group per head (grid ``cdiv(T, rows) x HV``).

    Warps split rows (each warp keeps whole 128-wide rows), so the per-row
    reduction, and the output, match the default launch whenever that uses
    at least 2 rows per block (T > 2 * SMs). Below that the default config is
    kept, as its 1-row tile reduces in a different order.
    """
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        calc_rows_per_block,
    )

    rows_per_block = calc_rows_per_block(num_rows, device)
    if rows_per_block < 2 or num_rows < 512:
        return rows_per_block, 1
    if num_rows < 2048:
        return 8, 2
    # VR sweep, T = 2144..8192: (64, 8) is fastest or tied (-35..-44% vs default).
    return 64, 8


@triton.jit
def _zero_fresh_state_rows_kernel(
    pool_ptr,
    indices_ptr,
    has_initial_state_ptr,
    stride_slot,
    ROW_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    if tl.load(has_initial_state_ptr + seq) == 0:
        slot = tl.load(indices_ptr + seq).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        tl.store(
            pool_ptr + slot * stride_slot + offs,
            tl.zeros((BLOCK,), dtype=pool_ptr.dtype.element_ty),
            # Padded rows carry a negative slot id and own no pool row.
            mask=(offs < ROW_NUMEL) & (slot >= 0),
        )


_zero_launch = launcher(_zero_fresh_state_rows_kernel)


def zero_fresh_state_rows(
    pool: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> None:
    """``pool[state_indices[i]] = 0`` where ``not has_initial_state[i]``, with no
    host sync. Negative (padding) slot ids are skipped. Pool rows may be padded
    (``stride(0)`` larger than a row) but each row must be contiguous.
    """
    num_seqs = state_indices.numel()
    if num_seqs == 0:
        return
    assert state_indices.is_contiguous() and has_initial_state.is_contiguous()
    if GDN_HOST_TRIM:
        # pool[0].numel() without indexing the pool (the caller's pool rows
        # are contiguous).
        row_numel = math.prod(pool.shape[1:])
    else:
        row_numel = pool[0].numel()
        assert pool[0].is_contiguous()
    assert has_initial_state.numel() == num_seqs
    block = 4096
    _zero_launch[(num_seqs, triton.cdiv(row_numel, block))](
        pool,
        state_indices,
        has_initial_state,
        pool.stride(0),
        ROW_NUMEL=row_numel,
        BLOCK=block,
        num_warps=4,
    )


@triton.jit
def _zero_fresh_state_rows_layers_kernel(
    pool_ptr,
    offsets_ptr,
    indices_ptr,
    has_initial_state_ptr,
    stride_slot,
    ROW_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
    ELEMS_16B: tl.constexpr,
):
    # _zero_fresh_state_rows_kernel on the pool of layer program_id(2), at
    # pool_ptr + offsets[layer] 16-byte units (the scaling keeps the 16-byte
    # alignment of pool_ptr provable, so the stores stay vectorized).
    seq = tl.program_id(0)
    if tl.load(has_initial_state_ptr + seq) == 0:
        base = pool_ptr + tl.load(offsets_ptr + tl.program_id(2)) * ELEMS_16B
        slot = tl.load(indices_ptr + seq).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        tl.store(
            base + slot * stride_slot + offs,
            tl.zeros((BLOCK,), dtype=pool_ptr.dtype.element_ty),
            mask=offs < ROW_NUMEL,
        )


_zero_layers_launch = launcher(_zero_fresh_state_rows_layers_kernel)


class StatePoolGroup:
    """The SSM pools of several GDN layers as one zeroing target: pool 0
    plus the other pools' 16-byte offsets from it (device int64). The pools
    are held weakly (a cached group must not keep a released KV cache alive);
    the caller launches only while they are the live pools of its layers.
    """

    __slots__ = ("refs", "offsets", "row_numel", "stride_slot", "elems_16b")

    def __init__(self, pools: tuple[torch.Tensor, ...]) -> None:
        self.refs = tuple(weakref.ref(p) for p in pools)
        ref = pools[0]
        es = ref.element_size()
        base = ref.data_ptr()
        offsets = [(p.data_ptr() - base) // 16 for p in pools]
        self.offsets = torch.tensor(offsets, dtype=torch.int64, device=ref.device)
        self.row_numel = math.prod(ref.shape[1:])
        self.stride_slot = ref.stride(0)
        self.elems_16b = 16 // es

    @staticmethod
    def compatible(pools: tuple[torch.Tensor, ...]) -> bool:
        """Whether one launch can address every pool: same dtype, row shape and
        slot stride, contiguous rows, 16-byte aligned starts.
        """
        ref = pools[0]
        if 16 % ref.element_size():
            return False
        for p in pools:
            if (
                p.dtype != ref.dtype
                or p.device != ref.device
                or p.shape[1:] != ref.shape[1:]
                or p.stride(0) != ref.stride(0)
                or not p[0].is_contiguous()
                or p.data_ptr() % 16
            ):
                return False
        return True


def zero_fresh_state_rows_layers(
    group: StatePoolGroup,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> None:
    """``zero_fresh_state_rows(pool, state_indices, has_initial_state)`` for
    every pool of ``group`` in one launch (the same elements are zeroed).
    """
    num_seqs = state_indices.numel()
    if num_seqs == 0:
        return
    assert state_indices.is_contiguous() and has_initial_state.is_contiguous()
    assert has_initial_state.numel() == num_seqs
    block = 4096
    pool = group.refs[0]()
    assert pool is not None
    _zero_layers_launch[
        (num_seqs, triton.cdiv(group.row_numel, block), len(group.refs))
    ](
        pool,
        group.offsets,
        state_indices,
        has_initial_state,
        group.stride_slot,
        ROW_NUMEL=group.row_numel,
        BLOCK=block,
        ELEMS_16B=group.elems_16b,
        num_warps=4,
    )
