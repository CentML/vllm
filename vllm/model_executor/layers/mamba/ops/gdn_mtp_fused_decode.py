# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec-decode (MTP) GDN core in one launch for small decode batches.

At small batches (C32: 1-4 requests per rank) every GDN layer runs its core as
three latency-bound launches: ``causal_conv1d_update`` (spec-decode window
roll), the value-split Triton recurrence (``gdn_mtp_decode``) and
``gdn_gated_norm_mxfp8``. This kernel does all three in one launch:

- every program owns ``(request, value head, BLOCK_V value rows)``. It computes
  the causal conv + SiLU of the q/k channels of its key head and of its own v
  channels for each of the request's (at most 7) tokens as separate vectors,
  straight from the pre-conv projection rows and the previous step's conv
  window, then runs the gated delta rule token by token and writes the raw
  core output (bf16 scratch) and the per-token SSM states. Keeping each
  token's q/k/v as its own vector avoids the per-token row extraction
  (cross-thread reductions) of ``gdn_mtp_decode``;
- each program rolls the conv window of its own v channels;
- the last program of each ``(request, key head)`` group (an atomic arrival
  counter that this program resets) rolls the group's q/k conv window, which
  all its siblings have read by then, and applies the gated RMSNorm + the
  MXFP8 quantization of out_proj's activation to the group's value heads,
  reading the core output its siblings wrote;
- ``NUM_FILL`` extra programs zero the activation rows past the valid token
  count and the scale rows up to the 128-row padding (FULL-graph padding).

The SSM state, the conv window and the weights are loaded before
``gdc_wait``. Conv, gating, recurrence, norm and quantization follow the op
order of the kernels it replaces; the q/k L2-norm sums reduce in a different
order than ``gdn_mtp_decode`` (the results agree to about one fp32 ulp), so
the output is not bitwise identical to the three-kernel path.
"""

import torch

from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import (
    MXFP8_BLOCK,
    mxfp8_quantize_rows,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton

# Tokens per request the kernel unrolls for (MTP with up to 6 draft tokens).
# Token 6 (k = 6) is only compiled in when S = 7.
MAX_FUSED_DECODE_TOKENS = 7


@triton.jit
def _conv_token(u0, u1, u2, u3, w0, w1, w2, w3, x, apply):
    """Width-4 causal conv + SiLU of one token, rounded to the input dtype as
    ``_causal_conv1d_update_kernel`` stores it; the raw token ``x`` when the
    conv is skipped (null conv slot).
    """
    acc = tl.zeros(u0.shape, dtype=tl.float32)
    acc += u0 * w0
    acc += u1 * w1
    acc += u2 * w2
    acc += u3 * w3
    acc = acc / (1 + tl.exp(-acc))
    return tl.where(apply, acc.to(x.dtype), x).to(tl.float32)


@triton.jit
def _token_row(x_ptr, tok0, t: tl.constexpr, stride_x, ch, n, S: tl.constexpr):
    """Pre-conv projection row of the request's token ``t`` for ``ch``; zeros
    past the token count (and constant zeros past ``S``).
    """
    if t < S:
        return tl.load(x_ptr + (tok0 + t) * stride_x + ch, mask=n > t, other=0.0)
    return tl.zeros(ch.shape, dtype=x_ptr.dtype.element_ty)


@triton.jit
def _token_rows(x_ptr, tok0, stride_x, ch, n, S: tl.constexpr):
    """Rows of tokens 0..6 (``_token_row``)."""
    return (
        _token_row(x_ptr, tok0, 0, stride_x, ch, n, S),
        _token_row(x_ptr, tok0, 1, stride_x, ch, n, S),
        _token_row(x_ptr, tok0, 2, stride_x, ch, n, S),
        _token_row(x_ptr, tok0, 3, stride_x, ch, n, S),
        _token_row(x_ptr, tok0, 4, stride_x, ch, n, S),
        _token_row(x_ptr, tok0, 5, stride_x, ch, n, S),
        _token_row(x_ptr, tok0, 6, stride_x, ch, n, S),
    )


@triton.jit
def _conv_tokens(c0, c1, c2, x0, x1, x2, x3, x4, x5, x6, w0, w1, w2, w3, apply):
    """Conv + SiLU of tokens 0..6: token t's window is
    ``[c0, c1, c2, x0, .., x6][t : t + 4]``.
    """
    return (
        _conv_token(c0, c1, c2, x0, w0, w1, w2, w3, x0, apply),
        _conv_token(c1, c2, x0, x1, w0, w1, w2, w3, x1, apply),
        _conv_token(c2, x0, x1, x2, w0, w1, w2, w3, x2, apply),
        _conv_token(x0, x1, x2, x3, w0, w1, w2, w3, x3, apply),
        _conv_token(x1, x2, x3, x4, w0, w1, w2, w3, x4, apply),
        _conv_token(x2, x3, x4, x5, w0, w1, w2, w3, x5, apply),
        _conv_token(x3, x4, x5, x6, w0, w1, w2, w3, x6, apply),
    )


@triton.jit
def _window(conv_state_ptr, base, stride_cs_dim, stride_cs_tok, ch):
    """The 3 conv-window columns the first token reads (previous step)."""
    p = conv_state_ptr + base + ch * stride_cs_dim
    return (
        tl.load(p),
        tl.load(p + stride_cs_tok),
        tl.load(p + 2 * stride_cs_tok),
    )


@triton.jit
def _conv_weights(w_ptr, ch, stride_w_dim, stride_w_width):
    p = w_ptr + ch * stride_w_dim
    return (
        tl.load(p),
        tl.load(p + stride_w_width),
        tl.load(p + 2 * stride_w_width),
        tl.load(p + 3 * stride_w_width),
    )


@triton.jit
def _store_window(
    conv_state_ptr, slot_base, stride_cs_dim, stride_cs_tok, ch, n, c1, c2,
    x0, x1, x2, x3, x4, x5, x6, apply, S: tl.constexpr,
):  # fmt: skip
    """``_causal_conv1d_update_kernel``'s spec-decode roll: the slot's new
    window is ``[c1, c2, x0, .., x_{n-1}]`` (positions ``< 2 + n``).
    """
    p = conv_state_ptr + slot_base + ch * stride_cs_dim
    tl.store(p, c1, mask=apply)
    tl.store(p + stride_cs_tok, c2, mask=apply)
    tl.store(p + 2 * stride_cs_tok, x0, mask=apply & (n > 0))
    if S > 1:
        tl.store(p + 3 * stride_cs_tok, x1, mask=apply & (n > 1))
    if S > 2:
        tl.store(p + 4 * stride_cs_tok, x2, mask=apply & (n > 2))
    if S > 3:
        tl.store(p + 5 * stride_cs_tok, x3, mask=apply & (n > 3))
    if S > 4:
        tl.store(p + 6 * stride_cs_tok, x4, mask=apply & (n > 4))
    if S > 5:
        tl.store(p + 7 * stride_cs_tok, x5, mask=apply & (n > 5))
    if S > 6:
        tl.store(p + 8 * stride_cs_tok, x6, mask=apply & (n > 6))


@triton.jit
def _gate(a_ptr, b_ptr, tok, stride_a, stride_b, head, valid, dt_bias, a_log):
    """Decay ``exp(-exp(A_log) * softplus(a + dt_bias))`` and ``sigmoid(b)``."""
    a = tl.load(a_ptr + tok * stride_a + head, mask=valid, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + tok * stride_b + head, mask=valid, other=0.0).to(tl.float32)
    x = a + dt_bias
    softplus = tl.where(x > 20.0, x, tldevice.log1p(tl.exp(x)))
    return tl.exp(-tl.exp(a_log) * softplus), tl.sigmoid(b)


@triton.jit
def _l2norm(q, k, scale):
    q = q * (tldevice.rsqrt(tl.sum(q * q) + 1e-6) * scale)
    k = k * tldevice.rsqrt(tl.sum(k * k) + 1e-6)
    return q, k


@triton.jit
def _delta_step(
    h, q, k, v, decay, beta, o_ptr, row, out_cols, state_ptr, dst, valid
):  # fmt: skip
    """One token of the gated delta rule; stores the output row and, when
    ``dst`` (the destination slot's int64 element offset) is > 0, the state
    after the token.
    """
    h = h * decay
    delta = (v - tl.sum(h * k[None, :], axis=1)) * beta
    h = h + k[None, :] * delta[:, None]
    o = tl.sum(h * q[None, :], axis=1)
    # No valid source state: zero output and no state (as gdn_mtp_decode).
    tl.store(o_ptr + row + out_cols, tl.where(valid, o, 0.0).to(o_ptr.dtype.element_ty))
    if valid & (dst > 0):
        tl.store(state_ptr + dst, h.to(state_ptr.dtype.element_ty))
    return h


@triton.jit
def _token_prep(
    q, k, t: tl.constexpr, n, tok0, a_ptr, b_ptr, stride_a, stride_b, head,
    dt_bias, a_log, scale, si_row_ptr, stride_state_slot, S: tl.constexpr,
):  # fmt: skip
    """Everything token ``t``'s step needs besides the state, computed before
    the serial steps: L2-normed q (scaled) and k, decay, beta and the
    destination slot's element offset (0: skip).
    """
    g = 0.0
    beta = 0.0
    dst = 0
    if t < S:
        g, beta = _gate(a_ptr, b_ptr, tok0 + t, stride_a, stride_b, head, n > t,
                        dt_bias, a_log)  # fmt: skip
        q, k = _l2norm(q, k, scale)
        dst = tl.load(si_row_ptr + t).to(tl.int64) * stride_state_slot
    return q, k, g, beta, dst


@triton.jit
def _token_step(
    h, q, k, v, g, beta, dst, t: tl.constexpr, n, tok0, o_ptr, QH: tl.constexpr,
    out_cols, st, valid, S: tl.constexpr,
):  # fmt: skip
    # Compile-time token bound, then the request's runtime token count.
    if t < S:  # noqa: SIM102
        if n > t:
            h = _delta_step(h, q, k, v, g, beta, o_ptr, (tok0 + t) * QH, out_cols,
                            st, dst, valid)  # fmt: skip
    return h


@triton.jit(do_not_specialize=["num_requests", "num_rows", "padded_rows"])
def _gdn_mtp_fused_decode_kernel(
    x_ptr,  # [L, 2 * H * K + HV * V] pre-conv projection (row stride stride_x)
    a_ptr,  # [L, HV]
    b_ptr,  # [L, HV]
    conv_state_ptr,  # [slots, conv_dim, state_len] (strided)
    conv_w_ptr,  # [conv_dim, 4]
    a_log_ptr,  # [HV] fp32
    dt_bias_ptr,  # [HV]
    state_indices_ptr,  # [N, S] int32
    cu_seqlens_ptr,  # [N + 1] int32
    num_accepted_ptr,  # [N] int32
    state_ptr,  # [slots, HV, V, K]
    o_ptr,  # [L, HV, V] bf16 scratch (raw core output)
    z_ptr,  # [L, HV, V] output gate (row stride stride_z)
    norm_w_ptr,  # [V]
    q_out_ptr,  # [L, HV * V] e4m3
    scale_ptr,  # flat F8_128x4 UE8M0 scales
    counter_ptr,  # [N * H] int32, zero between launches
    num_requests,
    num_rows,
    padded_rows,
    stride_x,
    stride_a,
    stride_b,
    stride_cs_slot,
    stride_cs_dim,
    stride_cs_tok,
    stride_w_dim,
    stride_w_width,
    stride_state_slot,
    stride_z,
    scale,
    eps,
    null_block_id,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    S_PAD: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_FILL: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    NVB: tl.constexpr = V // BLOCK_V
    G: tl.constexpr = HV // H
    QH: tl.constexpr = HV * V
    PADDED_GROUPS: tl.constexpr = (QH // 32 + 3) // 4 * 4
    pid = tl.program_id(0)
    num_main = num_requests * (HV * NVB)

    if pid >= num_main:
        # Rows >= the valid count: zero values and scales; scale rows up to
        # the 128-row padding too.
        if LAUNCH_PDL:
            tl.extra.cuda.gdc_wait()
        n_valid = tl.load(cu_seqlens_ptr + num_requests)
        cols = tl.arange(0, QH)
        groups = tl.arange(0, PADDED_GROUPS)
        for row in range(n_valid + (pid - num_main), padded_rows, NUM_FILL):
            if row < num_rows:
                tl.store(
                    q_out_ptr + tl.cast(row, tl.int64) * QH + cols,
                    tl.zeros((QH,), dtype=tl.float32),
                )
            off = (
                row // 128 * (128 * PADDED_GROUPS)
                + groups // 4 * 512
                + row % 32 * 16
                + row % 128 // 32 * 4
                + groups % 4
            )
            tl.store(scale_ptr + off, tl.zeros((PADDED_GROUPS,), dtype=tl.uint8))
        return

    request = pid // (HV * NVB)
    value_head = pid % (HV * NVB) // NVB
    v_block = pid % NVB
    key_head = value_head // G
    bos = tl.load(cu_seqlens_ptr + request)
    n = tl.load(cu_seqlens_ptr + request + 1) - bos
    if n <= 0:
        return

    offs_k = tl.arange(0, K)
    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    q_ch = key_head * K + offs_k
    k_ch = (H + key_head) * K + offs_k
    v_ch = 2 * H * K + value_head * V + offs_v

    # --- independent of this step's projections: before gdc_wait ---
    accepted = tl.load(num_accepted_ptr + request)
    source_slot = tl.load(
        state_indices_ptr + request * S + accepted - 1,
        mask=(accepted > 0) & (accepted <= S),
        other=0,
    )
    valid = (source_slot > 0) & (n <= S)
    head_offset = value_head * (V * K)
    tile = offs_v[:, None] * K + offs_k[None, :]
    h = tl.load(
        state_ptr + source_slot.to(tl.int64) * stride_state_slot + head_offset + tile,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    dst0 = tl.load(state_indices_ptr + request * S)
    a_log = tl.load(a_log_ptr + value_head).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + value_head).to(tl.float32)
    # Conv window of the previous step, offset by the accepted tokens - 1, in
    # the slot of token 0 (causal_conv1d_update's conv_state_indices).
    conv_slot = dst0.to(tl.int64)
    apply_conv = conv_slot != null_block_id
    win = conv_slot * stride_cs_slot + (accepted - 1).to(tl.int64) * stride_cs_tok
    qc0, qc1, qc2 = _window(conv_state_ptr, win, stride_cs_dim, stride_cs_tok, q_ch)
    kc0, kc1, kc2 = _window(conv_state_ptr, win, stride_cs_dim, stride_cs_tok, k_ch)
    vc0, vc1, vc2 = _window(conv_state_ptr, win, stride_cs_dim, stride_cs_tok, v_ch)
    qw0, qw1, qw2, qw3 = _conv_weights(conv_w_ptr, q_ch, stride_w_dim, stride_w_width)
    kw0, kw1, kw2, kw3 = _conv_weights(conv_w_ptr, k_ch, stride_w_dim, stride_w_width)
    vw0, vw1, vw2, vw3 = _conv_weights(conv_w_ptr, v_ch, stride_w_dim, stride_w_width)

    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()

    tok0 = bos.to(tl.int64)
    qx0, qx1, qx2, qx3, qx4, qx5, qx6 = _token_rows(x_ptr, tok0, stride_x, q_ch, n, S)
    kx0, kx1, kx2, kx3, kx4, kx5, kx6 = _token_rows(x_ptr, tok0, stride_x, k_ch, n, S)
    vx0, vx1, vx2, vx3, vx4, vx5, vx6 = _token_rows(x_ptr, tok0, stride_x, v_ch, n, S)
    q0, q1, q2, q3, q4, q5, q6 = _conv_tokens(
        qc0, qc1, qc2, qx0, qx1, qx2, qx3, qx4, qx5, qx6, qw0, qw1, qw2, qw3,
        apply_conv,
    )  # fmt: skip
    k0, k1, k2, k3, k4, k5, k6 = _conv_tokens(
        kc0, kc1, kc2, kx0, kx1, kx2, kx3, kx4, kx5, kx6, kw0, kw1, kw2, kw3,
        apply_conv,
    )  # fmt: skip
    v0, v1, v2, v3, v4, v5, v6 = _conv_tokens(
        vc0, vc1, vc2, vx0, vx1, vx2, vx3, vx4, vx5, vx6, vw0, vw1, vw2, vw3,
        apply_conv,
    )  # fmt: skip

    si_row = state_indices_ptr + request * S
    q0, k0, g0, b0, d0 = _token_prep(q0, k0, 0, n, tok0, a_ptr, b_ptr, stride_a,
                                     stride_b, value_head, dt_bias, a_log, scale,
                                     si_row, stride_state_slot, S)  # fmt: skip
    q1, k1, g1, b1, d1 = _token_prep(q1, k1, 1, n, tok0, a_ptr, b_ptr, stride_a,
                                     stride_b, value_head, dt_bias, a_log, scale,
                                     si_row, stride_state_slot, S)  # fmt: skip
    q2, k2, g2, b2, d2 = _token_prep(q2, k2, 2, n, tok0, a_ptr, b_ptr, stride_a,
                                     stride_b, value_head, dt_bias, a_log, scale,
                                     si_row, stride_state_slot, S)  # fmt: skip
    q3, k3, g3, b3, d3 = _token_prep(q3, k3, 3, n, tok0, a_ptr, b_ptr, stride_a,
                                     stride_b, value_head, dt_bias, a_log, scale,
                                     si_row, stride_state_slot, S)  # fmt: skip
    q4, k4, g4, b4, d4 = _token_prep(q4, k4, 4, n, tok0, a_ptr, b_ptr, stride_a,
                                     stride_b, value_head, dt_bias, a_log, scale,
                                     si_row, stride_state_slot, S)  # fmt: skip
    q5, k5, g5, b5, d5 = _token_prep(q5, k5, 5, n, tok0, a_ptr, b_ptr, stride_a,
                                     stride_b, value_head, dt_bias, a_log, scale,
                                     si_row, stride_state_slot, S)  # fmt: skip
    q6, k6, g6, b6, d6 = _token_prep(q6, k6, 6, n, tok0, a_ptr, b_ptr, stride_a,
                                     stride_b, value_head, dt_bias, a_log, scale,
                                     si_row, stride_state_slot, S)  # fmt: skip

    out_cols = value_head * V + offs_v
    st = state_ptr + head_offset + tile
    h = _token_step(h, q0, k0, v0, g0, b0, d0, 0, n, tok0, o_ptr, QH, out_cols, st,
                    valid, S)  # fmt: skip
    h = _token_step(h, q1, k1, v1, g1, b1, d1, 1, n, tok0, o_ptr, QH, out_cols, st,
                    valid, S)  # fmt: skip
    h = _token_step(h, q2, k2, v2, g2, b2, d2, 2, n, tok0, o_ptr, QH, out_cols, st,
                    valid, S)  # fmt: skip
    h = _token_step(h, q3, k3, v3, g3, b3, d3, 3, n, tok0, o_ptr, QH, out_cols, st,
                    valid, S)  # fmt: skip
    h = _token_step(h, q4, k4, v4, g4, b4, d4, 4, n, tok0, o_ptr, QH, out_cols, st,
                    valid, S)  # fmt: skip
    h = _token_step(h, q5, k5, v5, g5, b5, d5, 5, n, tok0, o_ptr, QH, out_cols, st,
                    valid, S)  # fmt: skip
    h = _token_step(h, q6, k6, v6, g6, b6, d6, 6, n, tok0, o_ptr, QH, out_cols, st,
                    valid, S)  # fmt: skip
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Only this program reads its v channels' window: roll it.
    slot_base = conv_slot * stride_cs_slot
    _store_window(conv_state_ptr, slot_base, stride_cs_dim, stride_cs_tok, v_ch, n,
                  vc1, vc2, vx0, vx1, vx2, vx3, vx4, vx5, vx6, apply_conv,
                  S)  # fmt: skip

    # --- group tail: the last of the G * NVB programs of (request, key head) ---
    tl.debug_barrier()
    counter = counter_ptr + request * H + key_head
    arrival = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
    if arrival != G * NVB - 1:
        return
    tl.atomic_xchg(counter, 0, sem="relaxed", scope="gpu")

    # Every sibling has read the q/k window: roll it.
    _store_window(conv_state_ptr, slot_base, stride_cs_dim, stride_cs_tok, q_ch, n,
                  qc1, qc2, qx0, qx1, qx2, qx3, qx4, qx5, qx6, apply_conv,
                  S)  # fmt: skip
    _store_window(conv_state_ptr, slot_base, stride_cs_dim, stride_cs_tok, k_ch, n,
                  kc1, kc2, kx0, kx1, kx2, kx3, kx4, kx5, kx6, apply_conv,
                  S)  # fmt: skip

    # Gated RMSNorm (per head, norm before gate) + MXFP8 of the group's
    # heads, as _gdn_gated_norm_mxfp8_kernel.
    rows = tl.arange(0, S_PAD)
    row_valid = rows < n
    tokens = tok0 + rows
    offs_d = tl.arange(0, V)
    w = tl.load(norm_w_ptr + offs_d).to(tl.float32)
    groups = tl.arange(0, V // 32)
    for g in tl.static_range(G):
        head = key_head * G + g
        cols = head * V + offs_d
        x = tl.load(
            o_ptr + tokens[:, None] * QH + cols[None, :],
            mask=row_valid[:, None],
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        z = tl.load(
            z_ptr + tokens[:, None] * stride_z + cols[None, :],
            mask=row_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        var = tl.sum(x * x, axis=1) / V
        rstd = tl.rsqrt(var + eps)
        y = x * rstd[:, None]
        y = y * w[None, :]
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)
        y = y.to(tl.bfloat16).to(tl.float32)
        quantized, sf = mxfp8_quantize_rows(y, S_PAD, V)
        tl.store(
            q_out_ptr + tokens[:, None] * QH + cols[None, :],
            quantized,
            mask=row_valid[:, None],
        )
        gidx = head * (V // 32) + groups
        off = (
            tokens[:, None] // 128 * (128 * PADDED_GROUPS)
            + gidx[None, :] // 4 * 512
            + tokens[:, None] % 32 * 16
            + tokens[:, None] % 128 // 32 * 4
            + gidx[None, :] % 4
        )
        tl.store(scale_ptr + off, sf.to(tl.uint8), mask=row_valid[:, None])


def gdn_mtp_fused_decode_launch_config(
    num_requests: int, tokens_per_request: int = 4
) -> tuple[int, int]:
    """(BLOCK_V, num_warps). Measured on VR (SM107, DRAM 4752 MHz) between
    the in_proj/out_proj MXFP8 GEMMs at H=16, HV=32, K=V=128 (job 675792):
    4 tokens/request: 16/4 at 1-3 requests, 32/4 at 4, 64/4 at 5-6;
    5 tokens/request: 16/4 at 1-2 and 4 requests, 32/4 at 3 (also used at
    6 tokens/request).
    7 tokens/request (core only, bf16 state, 30 layers per graph, job 708057;
    us per layer vs the three-kernel chain): 8/4 at 1 request (10.3 vs 12.3),
    16/2 at 2-3 (12.8 vs 13.7, 13.0 vs 14.3), 64/4 at 4 (16.1, chain 15.3).
    """
    if tokens_per_request <= 4:
        if num_requests <= 3:
            return 16, 4
        return (32, 4) if num_requests == 4 else (64, 4)
    if tokens_per_request >= 7:
        if num_requests == 1:
            return 8, 4
        return (16, 2) if num_requests <= 3 else (64, 4)
    return (32, 4) if num_requests == 3 else (16, 4)


def gdn_mtp_fused_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state: torch.Tensor,
    core_out: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    activation: str,
    out_q: torch.Tensor,
    out_scale: torch.Tensor,
    counters: torch.Tensor,
    scale: float,
    null_block_id: int = 0,
    block_v: int | None = None,
    num_warps: int | None = None,
    num_fill: int = 4,
) -> None:
    """Conv + SiLU, gated delta rule, gated RMSNorm and MXFP8 of the
    spec-decode rows in one launch.

    ``mixed_qkv`` [T, conv_dim] is the pre-conv projection (row-strided view
    is fine; it is not modified), ``a``/``b`` [T, HV], ``conv_state``
    [slots, conv_dim, state_len] (any strides), ``conv_weight`` [conv_dim, 4]
    (no bias), ``state`` [slots, HV, V, K] (updated in place at
    ``state_indices``), ``core_out`` [T, HV, V] bf16 scratch, ``output_gate``
    [T, HV, V] (row strided), ``out_q`` [T, HV * V] e4m3 and ``out_scale`` the
    flat F8_128x4 scales of out_proj's activation. ``counters`` [>= N * H]
    int32 must be zero and is left zero. Rows past ``cu_seqlens[N]`` get zero
    values and scales; requests with no tokens (FULL-graph padding) are
    skipped.

    Contracts: ``causal_conv1d_update`` with ``conv_state_indices =
    state_indices[:, 0]``, SiLU, and ``num_accepted_tokens`` (spec-decode
    window roll); ``gdn_mtp_recurrence``; ``gdn_gated_norm_mxfp8`` with every
    valid row normalized.
    """
    num_requests, width = state_indices.shape
    num_rows, conv_dim = mixed_qkv.shape
    num_value_heads, dim_v, dim_k = state.shape[1:]
    num_key_heads = (conv_dim - num_value_heads * dim_v) // (2 * dim_k)
    hidden = num_value_heads * dim_v
    assert width <= MAX_FUSED_DECODE_TOKENS and conv_weight.shape == (conv_dim, 4)
    assert mixed_qkv.stride(1) == 1 and a.stride(1) == 1 and b.stride(1) == 1
    assert state.stride(1) == dim_v * dim_k and state.stride(3) == 1
    assert state_indices.is_contiguous() and core_out.is_contiguous()
    assert core_out.shape == (num_rows, num_value_heads, dim_v)
    assert output_gate.shape == core_out.shape and output_gate.stride(2) == 1
    assert output_gate.stride(1) == dim_v
    assert num_value_heads % num_key_heads == 0
    assert conv_state.shape[1] == conv_dim and conv_state.dtype == mixed_qkv.dtype
    assert conv_state.shape[2] >= 3 + width - 1
    assert out_q.shape == (num_rows, hidden) and out_q.is_contiguous()
    assert out_q.dtype == torch.float8_e4m3fn and out_scale.dtype == torch.uint8
    assert dim_v % MXFP8_BLOCK == 0
    assert counters.dtype == torch.int32
    assert counters.numel() >= num_requests * num_key_heads
    padded_rows = triton.cdiv(num_rows, 128) * 128
    assert out_scale.numel() == padded_rows * triton.cdiv(hidden // MXFP8_BLOCK, 4) * 4
    if num_rows == 0:
        return
    default_block_v, default_num_warps = gdn_mtp_fused_decode_launch_config(
        num_requests, width
    )
    block_v = default_block_v if block_v is None else block_v
    num_warps = default_num_warps if num_warps is None else num_warps
    launch_pdl = current_platform.is_arch_support_pdl()
    grid = (num_requests * num_value_heads * (dim_v // block_v) + num_fill,)
    _gdn_mtp_fused_decode_kernel[grid](
        mixed_qkv,
        a,
        b,
        conv_state,
        conv_weight,
        A_log,
        dt_bias,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        state,
        core_out,
        output_gate,
        norm_weight,
        out_q,
        out_scale,
        counters,
        num_requests,
        num_rows,
        padded_rows,
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weight.stride(0),
        conv_weight.stride(1),
        state.stride(0),
        output_gate.stride(0),
        scale,
        norm_eps,
        null_block_id,
        H=num_key_heads,
        HV=num_value_heads,
        K=dim_k,
        V=dim_v,
        S=width,
        S_PAD=triton.next_power_of_2(width),
        BLOCK_V=block_v,
        NUM_FILL=num_fill,
        ACTIVATION=activation,
        LAUNCH_PDL=launch_pdl,
        launch_pdl=launch_pdl,
        num_warps=num_warps,
    )
