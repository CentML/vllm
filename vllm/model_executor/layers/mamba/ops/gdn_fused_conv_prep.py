# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN prefill: causal conv1d (+ conv-state update) and the post-conv prep in
one kernel.

Replaces the ``causal_conv1d_fn`` + ``fused_post_conv_prep(output_g_exp=True)``
pair of the prefill rows: conv1d + SiLU of the q/k/v channels, q/k L2 norm,
``exp(g) = exp(-exp(A_log) * softplus(a + dt_bias))`` and ``beta = sigmoid(b)``,
written straight into the contiguous q/k/v/g/beta that FlashInfer reads. This
saves the bf16 [T, conv_dim] write and re-read between the two kernels and one
launch.

Same per-element arithmetic as the pair, so the outputs and the conv state are
bitwise identical: bf16 tap products accumulated in fp32 in tap order, the conv
output rounded to bf16 before the prep reads it, ``1 / sqrt(sum + 1e-6)`` L2
norm, and libdevice ``expf`` for exp(g).

Only the final conv state of each sequence is written (no APC intermediate
states) and the conv has no bias; callers fall back to the pair otherwise.

On SM10x, once ``enable_cuda_kernel()`` built it (the GDN prefill warmup does),
the CUDA version in gdn_conv_cuda.py runs instead: same outputs bit for bit,
faster (VLLM_GDN_CONV_CUDA=0 keeps the Triton kernel).
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, tldevice, triton

logger = init_logger(__name__)

# Tile configs (VR sweep), (BT, ST): BT tokens of one sequence per program,
# processed in ST-token sub-tiles. Long rows: 32-token tiles. Short batches
# (< 2048 rows, or decode rows folded in, < 128 rows per sequence): 16-token
# tiles waste fewer lanes on the partial last tile of each sequence.
FUSED_CONV_TILE_LONG = (32, 32)
FUSED_CONV_TILE_SHORT = (16, 16)
FUSED_CONV_WARPS = 4
FUSED_CONV_STAGES = 1
# Metadata scan tile. Batches up to this size keep the single-reduction path;
# larger batches scan masked tiles without specializing on the sequence count.
FUSED_CONV_SEQ_TILE = 64

GDN_CONV_CUDA = os.environ.get("VLLM_GDN_CONV_CUDA", "1") == "1"
_cuda_kernel_ready: list = []
_cuda_kernel_tried: list = []


def enable_cuda_kernel() -> None:
    """Build (JIT, nvcc) and enable the CUDA kernel; once per process. A
    failed build logs a warning and keeps the Triton kernel.
    """
    if _cuda_kernel_tried:
        return
    _cuda_kernel_tried.append(True)
    try:
        from vllm.model_executor.layers.mamba.ops import gdn_conv_cuda

        gdn_conv_cuda.load()
    except Exception:
        logger.warning(
            "GDN prefill: CUDA conv kernel build failed; using the Triton kernel.",
            exc_info=True,
        )
        return
    _cuda_kernel_ready.append(gdn_conv_cuda.gdn_conv_cuda_prep)
    logger.info("GDN prefill: CUDA fused conv kernel built and enabled.")


@triton.jit
def _conv_tile(
    x_ptr,
    stride_x_tok,
    w_ptr,
    stride_w_dim,
    stride_w_width,
    cs_ptr,
    s_cs_seq,
    s_cs_dim,
    s_cs_tok,
    slot,
    use_state,
    seq_start,
    offs_t,
    mask_t,
    ch_base,
    D: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """Causal depthwise conv + SiLU of one [ST, D] channel tile of one
    sequence; taps before the sequence start read the conv state.
    """
    offs_c = ch_base + tl.arange(0, D)
    acc = tl.zeros([offs_t.shape[0], D], dtype=tl.float32)
    for j in tl.static_range(WIDTH):
        w = tl.load(w_ptr + offs_c * stride_w_dim + j * stride_w_width)
        tau = offs_t - (WIDTH - 1) + j  # absolute token index
        in_seq = tau >= seq_start
        xv = tl.load(
            x_ptr + tau.to(tl.int64)[:, None] * stride_x_tok + offs_c[None, :],
            mask=(mask_t & in_seq)[:, None],
            other=0.0,
        )
        d = seq_start - tau  # 1..WIDTH-1 -> conv-state row WIDTH-1-d
        row = (WIDTH - 1) - d
        sv = tl.load(
            cs_ptr
            + slot * s_cs_seq
            + offs_c[None, :] * s_cs_dim
            + row[:, None] * s_cs_tok,
            mask=(mask_t & (d > 0) & use_state)[:, None],
            other=0.0,
        )
        acc += (xv + sv) * w[None, :]
    acc = acc / (1 + tl.exp(-acc))
    return acc.to(x_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _conv_tile_interior(
    x_ptr,
    stride_x_tok,
    w_ptr,
    stride_w_dim,
    stride_w_width,
    offs_t,
    mask_t,
    ch_base,
    D: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """``_conv_tile`` for tiles whose taps all lie inside the sequence."""
    offs_c = ch_base + tl.arange(0, D)
    acc = tl.zeros([offs_t.shape[0], D], dtype=tl.float32)
    for j in tl.static_range(WIDTH):
        w = tl.load(w_ptr + offs_c * stride_w_dim + j * stride_w_width)
        tau = offs_t - (WIDTH - 1) + j
        xv = tl.load(
            x_ptr + tau.to(tl.int64)[:, None] * stride_x_tok + offs_c[None, :],
            mask=mask_t[:, None],
            other=0.0,
        )
        acc += xv * w[None, :]
    acc = acc / (1 + tl.exp(-acc))
    return acc.to(x_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _write_conv_state(
    x_ptr,
    stride_x_tok,
    cs_ptr,
    s_cs_seq,
    s_cs_dim,
    s_cs_tok,
    slot,
    use_state,
    seq_start,
    seq_end,
    ch_base,
    D: tl.constexpr,
    WIDTH: tl.constexpr,
    NP2W: tl.constexpr,
):
    """conv_state[slot, :, 0..WIDTH-2] = the last WIDTH-1 inputs of the
    sequence (older rows shifted from the previous state for short chunks).
    """
    offs_c = ch_base + tl.arange(0, D)
    r = tl.arange(0, NP2W)
    rmask = r < (WIDTH - 1)
    tau = seq_end - (WIDTH - 1) + r
    in_seq = tau >= seq_start
    xv = tl.load(
        x_ptr + tau.to(tl.int64)[:, None] * stride_x_tok + offs_c[None, :],
        mask=(rmask & in_seq)[:, None],
        other=0.0,
    )
    d = seq_start - tau
    srow = (WIDTH - 1) - d
    sv = tl.load(
        cs_ptr
        + slot * s_cs_seq
        + offs_c[None, :] * s_cs_dim
        + srow[:, None] * s_cs_tok,
        mask=(rmask & (d > 0) & use_state)[:, None],
        other=0.0,
    )
    tl.debug_barrier()
    tl.store(
        cs_ptr + slot * s_cs_seq + offs_c[None, :] * s_cs_dim + r[:, None] * s_cs_tok,
        xv + sv,
        mask=rmask[:, None],
    )


# The per-step index tensors are slices at varying offsets (and stride 1 or
# the block-table width): no specialization, so one compile serves every step.
@triton.jit(
    do_not_specialize=["num_seqs", "stride_cidx", "stride_hinit"],
    do_not_specialize_on_alignment=["cidx_ptr", "hinit_ptr", "cu_ptr"],
)
def _gdn_fused_conv_prep_kernel(
    x_ptr,
    stride_x_tok,
    w_ptr,
    stride_w_dim,
    stride_w_width,
    cs_ptr,
    s_cs_seq,
    s_cs_dim,
    s_cs_tok,
    cidx_ptr,
    stride_cidx,
    hinit_ptr,
    stride_hinit,
    cu_ptr,
    num_seqs,
    a_ptr,
    b_ptr,
    stride_a,
    stride_b,
    a_log_ptr,
    dtb_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    WIDTH: tl.constexpr,
    NP2W: tl.constexpr,
    BT: tl.constexpr,
    ST: tl.constexpr,
    MAXS: tl.constexpr,
    SCAN_SEQS: tl.constexpr,
    NSTAGES: tl.constexpr,
):
    """Grid (sum of per-sequence BT chunks (upper bound), 2 * H + HV).

    Axis 1: one q head, one k head, or one v head (+ its g/beta). Each program
    walks the BT tokens of one chunk of one sequence in ST-token sub-tiles. The
    chunk-0 program of a sequence also writes that sequence's final conv state
    for its channels; it is the only reader of the old state of those channels.
    """
    pid = tl.program_id(0)
    i_head = tl.program_id(1)
    si = tl.arange(0, MAXS)
    if SCAN_SEQS:
        # Find the tile containing pid before reducing its sequence metadata.
        # Keep registers/scan width bounded even for very large request batches.
        seq_base = 0
        chunk_base = 0
        seq = num_seqs
        seq_start = 0
        seq_end = 0
        chunk = 0
        while (seq_base < num_seqs) & (seq == num_seqs):
            seq_ids = seq_base + si
            c0 = tl.load(cu_ptr + seq_ids, mask=seq_ids < num_seqs, other=0)
            c1 = tl.load(cu_ptr + seq_ids + 1, mask=seq_ids < num_seqs, other=0)
            nb = tl.where(seq_ids < num_seqs, tl.cdiv(c1 - c0, BT), 0)
            ends = tl.cumsum(nb, axis=0)
            tile_chunks = tl.sum(nb, axis=0)
            if pid < chunk_base + tile_chunks:
                local_pid = pid - chunk_base
                local_seq = tl.sum((ends <= local_pid).to(tl.int32), axis=0)
                sel = si == local_seq
                seq = seq_base + local_seq
                seq_start = tl.sum(tl.where(sel, c0, 0), axis=0)
                seq_end = tl.sum(tl.where(sel, c1, 0), axis=0)
                chunk = local_pid - tl.sum(tl.where(sel, ends - nb, 0), axis=0)
            chunk_base += tile_chunks
            seq_base += MAXS
    else:
        c0 = tl.load(cu_ptr + si, mask=si < num_seqs, other=0)
        c1 = tl.load(cu_ptr + si + 1, mask=si < num_seqs, other=0)
        nb = tl.where(si < num_seqs, tl.cdiv(c1 - c0, BT), 0)
        ends = tl.cumsum(nb, axis=0)
        seq = tl.sum((ends <= pid).to(tl.int32), axis=0)
        sel = si == seq
        seq_start = tl.sum(tl.where(sel, c0, 0), axis=0)
        seq_end = tl.sum(tl.where(sel, c1, 0), axis=0)
        chunk = pid - (
            tl.sum(tl.where(sel, ends, 0), axis=0)
            - tl.sum(tl.where(sel, nb, 0), axis=0)
        )
    if seq >= num_seqs:
        return
    slot = tl.load(cidx_ptr + seq * stride_cidx).to(tl.int64)
    use_state = tl.load(hinit_ptr + seq * stride_hinit) != 0
    chunk_start = seq_start + chunk * BT
    if i_head < 2 * H:
        # q head (i_head < H) or k head: conv, SiLU, L2 norm
        is_k = i_head >= H
        hh = i_head - H * is_k.to(tl.int32)
        ch = i_head * K  # q channels [0, H*K), k channels [H*K, 2*H*K)
        out_ptr = tl.where(is_k, k_ptr, q_ptr)
        offs_k = tl.arange(0, K)
        for sub in tl.range(0, BT // ST, num_stages=NSTAGES):
            offs_t = chunk_start + sub * ST + tl.arange(0, ST)
            mask_t = offs_t < seq_end
            interior = (chunk_start + sub * ST) >= (seq_start + WIDTH - 1)
            row64 = offs_t.to(tl.int64)
            if interior:
                y = _conv_tile_interior(
                    x_ptr,
                    stride_x_tok,
                    w_ptr,
                    stride_w_dim,
                    stride_w_width,
                    offs_t,
                    mask_t,
                    ch,
                    K,
                    WIDTH,
                )
            else:
                y = _conv_tile(
                    x_ptr,
                    stride_x_tok,
                    w_ptr,
                    stride_w_dim,
                    stride_w_width,
                    cs_ptr,
                    s_cs_seq,
                    s_cs_dim,
                    s_cs_tok,
                    slot,
                    use_state,
                    seq_start,
                    offs_t,
                    mask_t,
                    ch,
                    K,
                    WIDTH,
                )
            y = y * (1.0 / tl.sqrt(tl.sum(y * y, axis=1) + 1e-6))[:, None]
            o = row64[:, None] * (H * K) + hh * K + offs_k[None, :]
            tl.store(out_ptr + o, y.to(q_ptr.dtype.element_ty), mask=mask_t[:, None])
        if chunk == 0:
            _write_conv_state(
                x_ptr,
                stride_x_tok,
                cs_ptr,
                s_cs_seq,
                s_cs_dim,
                s_cs_tok,
                slot,
                use_state,
                seq_start,
                seq_end,
                ch,
                K,
                WIDTH,
                NP2W,
            )
    else:
        # v head: conv, SiLU, plus the gating of that head
        i_hv = i_head - 2 * H
        vch = 2 * H * K + i_hv * V
        offs_v = tl.arange(0, V)
        A_log = tl.load(a_log_ptr + i_hv).to(tl.float32)
        dtb = tl.load(dtb_ptr + i_hv).to(tl.float32)
        for sub in tl.range(0, BT // ST, num_stages=NSTAGES):
            offs_t = chunk_start + sub * ST + tl.arange(0, ST)
            mask_t = offs_t < seq_end
            interior = (chunk_start + sub * ST) >= (seq_start + WIDTH - 1)
            row64 = offs_t.to(tl.int64)
            if interior:
                v = _conv_tile_interior(
                    x_ptr,
                    stride_x_tok,
                    w_ptr,
                    stride_w_dim,
                    stride_w_width,
                    offs_t,
                    mask_t,
                    vch,
                    V,
                    WIDTH,
                )
            else:
                v = _conv_tile(
                    x_ptr,
                    stride_x_tok,
                    w_ptr,
                    stride_w_dim,
                    stride_w_width,
                    cs_ptr,
                    s_cs_seq,
                    s_cs_dim,
                    s_cs_tok,
                    slot,
                    use_state,
                    seq_start,
                    offs_t,
                    mask_t,
                    vch,
                    V,
                    WIDTH,
                )
            tl.store(
                v_ptr + row64[:, None] * (HV * V) + i_hv * V + offs_v[None, :],
                v.to(v_ptr.dtype.element_ty),
                mask=mask_t[:, None],
            )
            av = tl.load(a_ptr + row64 * stride_a + i_hv, mask=mask_t, other=0).to(
                tl.float32
            )
            bv = tl.load(b_ptr + row64 * stride_b + i_hv, mask=mask_t, other=0).to(
                tl.float32
            )
            xx = av + dtb
            sp = tl.where(
                xx > 0, xx + tl.log(1.0 + tl.exp(-xx)), tl.log(1.0 + tl.exp(xx))
            )
            sp = tl.where(xx <= 20.0, sp, xx)
            # libdevice expf, as fused_post_conv_prep(output_g_exp=True)
            g = tldevice.exp(-tl.exp(A_log) * sp)
            tl.store(g_ptr + row64 * HV + i_hv, g, mask=mask_t)
            tl.store(beta_ptr + row64 * HV + i_hv, tl.sigmoid(bv), mask=mask_t)
        if chunk == 0:
            _write_conv_state(
                x_ptr,
                stride_x_tok,
                cs_ptr,
                s_cs_seq,
                s_cs_dim,
                s_cs_tok,
                slot,
                use_state,
                seq_start,
                seq_end,
                vch,
                V,
                WIDTH,
                NP2W,
            )


def gdn_fused_conv_prep(
    x: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    tile: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conv1d + SiLU + conv-state update + post-conv prep of prefill rows.

    x: [P, conv_dim] pre-conv rows (channel stride 1, any row stride).
    conv_weights: [conv_dim, width]. conv_state: [slots, conv_dim, >= width - 1]
    (any strides). cache_indices / has_initial_state: [num_seqs].
    cu_seqlens: int32 [num_seqs + 1], relative to x row 0, with no sequence-count
    limit. a, b: [P, HV] (row-strided).

    Returns q, k: [P, H, K]; v: [P, HV, V] (x dtype); exp(g), beta: [P, HV] fp32.
    ``tile`` forces the Triton kernel with that (BT, ST) (the warmup compiles
    both).
    """
    P = x.shape[0]
    H, K, V = num_k_heads, head_k_dim, head_v_dim
    HV = A_log.shape[0]
    width = conv_weights.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    assert x.stride(1) == 1 and x.shape[1] == 2 * H * K + HV * V
    assert cu_seqlens.dtype == torch.int32
    if _cuda_kernel_ready and tile is None:
        out = _cuda_kernel_ready[0](
            x,
            conv_weights,
            conv_state,
            cache_indices,
            has_initial_state,
            cu_seqlens,
            a,
            b,
            A_log,
            dt_bias,
            num_k_heads,
            head_k_dim,
            head_v_dim,
        )
        if out is not None:
            return out
    q = torch.empty(P, H, K, dtype=x.dtype, device=x.device)
    k = torch.empty(P, H, K, dtype=x.dtype, device=x.device)
    v = torch.empty(P, HV, V, dtype=x.dtype, device=x.device)
    g = torch.empty(P, HV, dtype=torch.float32, device=x.device)
    beta = torch.empty(P, HV, dtype=torch.float32, device=x.device)
    if P == 0:
        return q, k, v, g, beta
    if tile is None:
        short = P < 2048 or 128 * num_seqs > P
        tile = FUSED_CONV_TILE_SHORT if short else FUSED_CONV_TILE_LONG
    BT, ST = tile
    grid = (triton.cdiv(P, BT) + num_seqs, 2 * H + HV)
    _gdn_fused_conv_prep_kernel[grid](
        x,
        x.stride(0),
        conv_weights,
        conv_weights.stride(0),
        conv_weights.stride(1),
        conv_state,
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        cache_indices,
        cache_indices.stride(0),
        has_initial_state,
        has_initial_state.stride(0),
        cu_seqlens,
        num_seqs,
        a,
        b,
        a.stride(0),
        b.stride(0),
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        H=H,
        HV=HV,
        K=K,
        V=V,
        WIDTH=width,
        NP2W=triton.next_power_of_2(width - 1),
        BT=BT,
        ST=ST,
        MAXS=FUSED_CONV_SEQ_TILE,
        SCAN_SEQS=num_seqs > FUSED_CONV_SEQ_TILE,
        NSTAGES=FUSED_CONV_STAGES,
        num_warps=FUSED_CONV_WARPS,
    )
    return q, k, v, g, beta
