# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN spec-decode (MTP) recurrence with the value dim split across programs.

Same contract as the recurrence part of ``fused_gdn_decode_post_conv_mtp``
(csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu): per request, read the
state of the last accepted token (``state_indices[r, num_accepted[r] - 1]``),
run the gated delta rule over the request's tokens (q/k L2 norm, decay
``exp(-exp(A_log) * softplus(a + dt_bias))``, beta ``sigmoid(b)``) and write the
state after token t to ``state_indices[r, t]`` (slots <= 0 are skipped). It
writes the raw core output (bf16, before the gated RMSNorm), so the caller runs
the norm, e.g. inside ``gdn_gated_norm_mxfp8``.

The CUDA kernel runs one 256-thread CTA per (request, value head) and walks the
128 value rows in 4 sequential chunks; at decode batch sizes of a few requests
it is latency-bound. Here each program owns BLOCK_V value rows, so a
(request, head) pair spreads over 128 / BLOCK_V programs.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton


@triton.jit(do_not_specialize=["num_requests"])
def _gdn_mtp_recurrence_kernel(
    mixed_qkv_ptr,  # [L, 2 * H * K + HV * V] bf16 (row stride stride_mixed)
    a_ptr,  # [L, HV]
    b_ptr,  # [L, HV]
    a_log_ptr,  # [HV] fp32
    dt_bias_ptr,  # [HV]
    state_indices_ptr,  # [N, S] int32
    cu_seqlens_ptr,  # [N + 1] int32
    num_accepted_ptr,  # [N] int32
    state_ptr,  # [slots, HV, V, K]
    out_ptr,  # [L, HV, V] bf16, contiguous
    num_requests,
    stride_mixed,
    stride_a,
    stride_b,
    stride_state_slot,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    S_PAD: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    request = tl.program_id(0)
    value_head = tl.program_id(1)
    v_block = tl.program_id(2)
    bos = tl.load(cu_seqlens_ptr + request)
    eos = tl.load(cu_seqlens_ptr + request + 1)
    num_tokens = eos - bos
    if num_tokens <= 0:
        return

    offs_k = tl.arange(0, K)
    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    out_cols = value_head * V + offs_v

    accepted = tl.load(num_accepted_ptr + request)
    source_slot = tl.load(
        state_indices_ptr + request * S + accepted - 1,
        mask=(accepted > 0) & (accepted <= S),
        other=0,
    )
    if source_slot <= 0 or num_tokens > S:
        # Same as the CUDA kernel: no valid source state -> zero output.
        for t in range(num_tokens):
            tl.store(
                out_ptr + (bos + t).to(tl.int64) * (HV * V) + out_cols,
                tl.zeros((BLOCK_V,), dtype=out_ptr.dtype.element_ty),
            )
        return

    key_head = value_head // (HV // H)
    head_offset = value_head * (V * K)
    tile = offs_v[:, None] * K + offs_k[None, :]
    h = tl.load(
        state_ptr + source_slot.to(tl.int64) * stride_state_slot + head_offset + tile
    ).to(tl.float32)

    # Everything the recurrence reads besides the state is loaded up front (all
    # S token rows at once), so the serial token loop only computes and stores.
    rows = tl.arange(0, S_PAD)
    row_valid = rows < num_tokens
    tokens = (bos + rows).to(tl.int64)
    qkv_rows = mixed_qkv_ptr + tokens[:, None] * stride_mixed
    q_all = tl.load(
        qkv_rows + key_head * K + offs_k[None, :], mask=row_valid[:, None], other=0.0
    ).to(tl.float32)
    k_all = tl.load(
        qkv_rows + (H + key_head) * K + offs_k[None, :],
        mask=row_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    v_all = tl.load(
        qkv_rows + 2 * H * K + value_head * V + offs_v[None, :],
        mask=row_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    q_all = (
        q_all * (tldevice.rsqrt(tl.sum(q_all * q_all, axis=1) + 1e-6) * scale)[:, None]
    )
    k_all = k_all * tldevice.rsqrt(tl.sum(k_all * k_all, axis=1) + 1e-6)[:, None]
    x = tl.load(a_ptr + tokens * stride_a + value_head, mask=row_valid, other=0.0).to(
        tl.float32
    ) + tl.load(dt_bias_ptr + value_head).to(tl.float32)
    softplus = tl.where(x > 20.0, x, tldevice.log1p(tl.exp(x)))
    decay_all = tl.exp(
        -tl.exp(tl.load(a_log_ptr + value_head).to(tl.float32)) * softplus
    )
    beta_all = tl.sigmoid(
        tl.load(b_ptr + tokens * stride_b + value_head, mask=row_valid, other=0.0).to(
            tl.float32
        )
    )
    dst_all = tl.load(state_indices_ptr + request * S + rows, mask=rows < S, other=0)

    for t in tl.static_range(S):
        if t < num_tokens:
            pick = rows == t
            q = tl.sum(tl.where(pick[:, None], q_all, 0.0), axis=0)
            k = tl.sum(tl.where(pick[:, None], k_all, 0.0), axis=0)
            v = tl.sum(tl.where(pick[:, None], v_all, 0.0), axis=0)
            decay = tl.sum(tl.where(pick, decay_all, 0.0))
            beta = tl.sum(tl.where(pick, beta_all, 0.0))
            dst_slot = tl.sum(tl.where(pick, dst_all, 0))

            h = h * decay
            delta = (v - tl.sum(h * k[None, :], axis=1)) * beta
            h = h + k[None, :] * delta[:, None]
            o = tl.sum(h * q[None, :], axis=1)
            tl.store(
                out_ptr + (bos + t).to(tl.int64) * (HV * V) + out_cols,
                o.to(out_ptr.dtype.element_ty),
            )
            if dst_slot > 0:
                tl.store(
                    state_ptr
                    + dst_slot.to(tl.int64) * stride_state_slot
                    + head_offset
                    + tile,
                    h.to(state_ptr.dtype.element_ty),
                )


def gdn_mtp_launch_config(num_requests: int) -> tuple[int, int]:
    """(BLOCK_V, num_warps): split each (request, head) pair into more programs
    the smaller the batch (latency-bound). Measured on VR (SM107, 212 SMs) at
    H=16, HV=32, K=V=128, 4 tokens per request, fp32 state.
    """
    if num_requests <= 2:
        return 8, 2
    if num_requests <= 4:
        return 16, 2
    return 32, 4


def gdn_mtp_recurrence(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    block_v: int | None = None,
    num_warps: int | None = None,
) -> None:
    """Spec-decode GDN recurrence; ``out`` [L, HV, V] gets the raw core output.

    Rows of requests with no tokens are left untouched; requests without a
    valid source state get zeros (as in the CUDA MTP kernel).
    """
    num_requests, width = state_indices.shape
    num_value_heads, dim_v, dim_k = state.shape[1:]
    num_key_heads = (mixed_qkv.shape[1] - num_value_heads * dim_v) // (2 * dim_k)
    assert mixed_qkv.stride(1) == 1 and a.stride(1) == 1 and b.stride(1) == 1
    assert state.stride(1) == dim_v * dim_k and state.stride(3) == 1
    assert state_indices.is_contiguous() and out.is_contiguous()
    assert num_value_heads % num_key_heads == 0
    default_block_v, default_num_warps = gdn_mtp_launch_config(num_requests)
    block_v = default_block_v if block_v is None else block_v
    num_warps = default_num_warps if num_warps is None else num_warps
    grid = (num_requests, num_value_heads, dim_v // block_v)
    _gdn_mtp_recurrence_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        state,
        out,
        num_requests,
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        state.stride(0),
        scale,
        H=num_key_heads,
        HV=num_value_heads,
        K=dim_k,
        V=dim_v,
        S=width,
        S_PAD=triton.next_power_of_2(width),
        BLOCK_V=block_v,
        num_warps=num_warps,
    )
