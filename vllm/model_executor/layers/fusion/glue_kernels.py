# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GB300 glue-kernel variants (gb300 study, agent "glue"): same math as the port Triton kernels in
norm_quant_kernels.py, restructured for latency (all loads of a row issued before the first store,
no second pass over global memory). Bit-identical outputs by construction:

  * norm_quant_rr: (residual adds) + Gemma RMSNorm + MXFP8 epilogue. Same [XBLOCK, RB] fp32 sum-of-squares
    accumulator, the same two `acc + x * x` updates (chunk 0, then chunk 1) and the same tl.sum as
    norm_quant_kernels._nqf_norm_quant_kernel's 2-iteration loop (H == 2 * RB), the same launch config
    (XBLOCK / RB / num_warps), so the reduction tree is unchanged; the row is kept in registers (as in
    _qgf_fin_norm_quant_kernel) instead of being re-read for the output pass.
  * gate_mul_quant_rr: bf16(f32(attn) * sigmoid(f32(gate))) + MXFP8 epilogue; all N/BN chunks loaded first.
    Optionally reads the gate straight from the interleaved [q | gate] QKV projection (GATE_D > 0: column c of
    the gate lives at G + row * stride_g + (c // GATE_D) * 2 * GATE_D + GATE_D + c % GATE_D), so the separate
    contiguous gate copy is not needed (pure data movement, bit-identical).

Env gates (read by the callers, default off): GLUE_NQRR=1 (norm_quant_rr), GLUE_GMRR=1 (gate_mul_quant_rr).
"""
# ruff: noqa: E501
# fmt: off
import torch

from vllm.model_executor.layers.fusion.norm_quant_kernels import INV_E4M3_MAX_C, _mx_epilogue
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice


@triton.jit
def _glue_mx_store(y, row, cols, xmask, Q, SF_SWZ, SF_LIN, PADDED_M, stride_q, c0,
                   H: tl.constexpr, XBLOCK: tl.constexpr, RB: tl.constexpr,
                   EMIT_SWZ: tl.constexpr, EMIT_LIN: tl.constexpr, PADDED_SF_COLS: tl.constexpr):
    """== the EMIT_Q block of _nqf_norm_quant_kernel for one [XBLOCK, RB] chunk starting at column c0."""
    NSF_RB: tl.constexpr = RB // 32
    yf = tl.reshape(y.to(tl.float32), [XBLOCK, NSF_RB, 32])
    amax = tl.max(tl.abs(yf), 2)
    nm = amax * INV_E4M3_MAX_C
    bits = nm.to(tl.int32, bitcast=True)
    e = (bits >> 23) & 255
    mant = bits & 0x7FFFFF
    bump = tl.where((mant != 0) & ~((e == 0) & (mant <= 0x400000)), 1, 0)
    e2 = tl.minimum(e + bump, 254)
    e2 = tl.where(nm <= 0.0, 0, e2)
    inv = tl.where(e2 == 0, 0.0, ((254 - e2) << 23).to(tl.float32, bitcast=True))
    qv = tl.clamp(yf * inv[:, :, None], -448.0, 448.0).to(tl.float8e4nv)
    qoff = tl.reshape(row * stride_q + cols, [XBLOCK, NSF_RB, 32])
    tl.store(Q + qoff, qv, tl.reshape(xmask & (cols >= 0), [XBLOCK, NSF_RB, 32]))
    j = c0 // 32 + tl.arange(0, NSF_RB)[None, :]
    sf = e2.to(tl.uint8)
    if EMIT_LIN:
        tl.store(SF_LIN + row * (H // 32) + j, sf, xmask)
    if EMIT_SWZ:
        off = (j % 4) + (j // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 \
            + (row // 128) * (128 * PADDED_SF_COLS)
        tl.store(SF_SWZ + off, sf, row < PADDED_M)


@triton.jit
def _glue_norm_quant_rr_kernel(
    S, F, A, R, W, OUT, RES_OUT, Q, SF_SWZ, SF_LIN,
    M, PADDED_M, eps,
    stride_s, stride_f, stride_a, stride_r, stride_out, stride_res, stride_q,
    H: tl.constexpr, XBLOCK: tl.constexpr, RB: tl.constexpr, NUM_IN: tl.constexpr,
    EMIT_Q: tl.constexpr, EMIT_SWZ: tl.constexpr, EMIT_LIN: tl.constexpr,
    PADDED_SF_COLS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * XBLOCK + tl.arange(0, XBLOCK)[:, None]   # [XBLOCK, 1]
    xmask = row < M
    rbase = tl.arange(0, RB)[None, :]
    c0 = rbase
    c1 = RB + rbase
    a0 = tl.load(A + row * stride_a + c0, xmask, other=0.0).to(tl.float32)
    r0 = tl.load(R + row * stride_r + c0, xmask, other=0.0).to(tl.float32)
    a1 = tl.load(A + row * stride_a + c1, xmask, other=0.0).to(tl.float32)
    r1 = tl.load(R + row * stride_r + c1, xmask, other=0.0).to(tl.float32)
    if NUM_IN == 4:
        s0 = tl.load(S + row * stride_s + c0, xmask, other=0.0).to(tl.float32)
        f0 = tl.load(F + row * stride_f + c0, xmask, other=0.0).to(tl.float32)
        s1 = tl.load(S + row * stride_s + c1, xmask, other=0.0).to(tl.float32)
        f1 = tl.load(F + row * stride_f + c1, xmask, other=0.0).to(tl.float32)
        x0 = (s0 + f0) + (a0 + r0)
        x1 = (s1 + f1) + (a1 + r1)
    else:
        x0 = a0 + r0
        x1 = a1 + r1
    w0 = tl.load(W + c0).to(tl.float32)
    w1 = tl.load(W + c1).to(tl.float32)
    acc = tl.full([XBLOCK, RB], 0, tl.float32)
    acc = tl.where(xmask, acc + x0 * x0, acc)
    if NUM_IN == 4:
        tl.store(RES_OUT + row * stride_res + c0, x0.to(tl.bfloat16), xmask)
    acc = tl.where(xmask, acc + x1 * x1, acc)
    if NUM_IN == 4:
        tl.store(RES_OUT + row * stride_res + c1, x1.to(tl.bfloat16), xmask)
    ssum = tl.sum(acc, 1)[:, None]
    rs = libdevice.rsqrt(ssum / tl.full([1, 1], H, tl.float32) + eps)
    y0 = ((x0 * rs) * (w0 + 1.0)).to(tl.bfloat16)
    tl.store(OUT + row * stride_out + c0, y0, xmask)
    if EMIT_Q:
        _glue_mx_store(y0, row, c0, xmask, Q, SF_SWZ, SF_LIN, PADDED_M, stride_q, 0, H, XBLOCK, RB,
                       EMIT_SWZ, EMIT_LIN, PADDED_SF_COLS)
    y1 = ((x1 * rs) * (w1 + 1.0)).to(tl.bfloat16)
    tl.store(OUT + row * stride_out + c1, y1, xmask)
    if EMIT_Q:
        _glue_mx_store(y1, row, c1, xmask, Q, SF_SWZ, SF_LIN, PADDED_M, stride_q, RB, H, XBLOCK, RB,
                       EMIT_SWZ, EMIT_LIN, PADDED_SF_COLS)


def norm_quant_rr(a, r, w, eps, s=None, f=None, emit_q=True, emit_swz=True, emit_lin=True, config=None):
    """Drop-in for norm_quant_kernels.norm_quant (H == 2 * RB). Same returns."""
    from vllm.model_executor.layers.fusion import norm_quant_kernels as K
    cfg = config or K.CONFIG
    M, H = a.shape
    assert H == 2 * cfg["RB"] and a.dtype == torch.bfloat16
    num_in = 4 if s is not None else 2
    dev = a.device
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    res = torch.empty((M, H), dtype=torch.bfloat16, device=dev) if num_in == 4 else None
    nsf = H // 32
    padded_sf_cols = (nsf + 3) // 4 * 4
    padded_m = (M + 127) // 128 * 128
    q = sf_swz = sf_lin = None
    if emit_q:
        q = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dev)
        if emit_swz:
            sf_swz = torch.empty((padded_m * padded_sf_cols,), dtype=torch.uint8, device=dev)
        if emit_lin:
            sf_lin = torch.empty((M * nsf,), dtype=torch.uint8, device=dev)
    emit_swz = emit_q and emit_swz
    emit_lin = emit_q and emit_lin
    grid_rows = padded_m if emit_swz else M
    xb = cfg["XBLOCK"]
    if grid_rows == 0:
        return out, res, q, sf_swz, sf_lin
    dummy = out
    _glue_norm_quant_rr_kernel[(triton.cdiv(grid_rows, xb),)](
        s if s is not None else dummy, f if f is not None else dummy, a, r, w,
        out, res if res is not None else dummy,
        q if q is not None else dummy,
        sf_swz if sf_swz is not None else dummy, sf_lin if sf_lin is not None else dummy,
        M, padded_m, eps,
        s.stride(0) if s is not None else 0, f.stride(0) if f is not None else 0,
        a.stride(0), r.stride(0), out.stride(0), res.stride(0) if res is not None else 0,
        q.stride(0) if q is not None else 0,
        H=H, XBLOCK=xb, RB=cfg["RB"], NUM_IN=num_in, EMIT_Q=emit_q, EMIT_SWZ=emit_swz, EMIT_LIN=emit_lin,
        PADDED_SF_COLS=padded_sf_cols, num_warps=cfg["num_warps"], num_stages=1,
    )
    return out, res, q, sf_swz, sf_lin


@triton.jit
def _glue_gate_mul_quant_kernel(A, G, OUT, Q, SF_SWZ, M, PADDED_M, stride_a, stride_g, stride_o, stride_q,
                                N: tl.constexpr, BN: tl.constexpr, XB: tl.constexpr, GATE_D: tl.constexpr,
                                PADDED_SF_COLS: tl.constexpr):
    """== _nqf_gate_mul_quant_kernel (out = bf16(f32(attn) * tl.sigmoid(f32(gate))) + MXFP8), XB rows per program,
    every chunk's loads issued before the first store. GATE_D > 0: gate read from the interleaved QKV rows."""
    row = tl.program_id(0).to(tl.int64) * XB + tl.arange(0, XB)[:, None]
    rmask = row < M
    cols = tl.arange(0, N)[None, :]
    a = tl.load(A + row * stride_a + cols, rmask, other=0.0).to(tl.float32)
    if GATE_D > 0:
        gcols = (cols // GATE_D) * (2 * GATE_D) + GATE_D + cols % GATE_D
    else:
        gcols = cols
    g = tl.load(G + row * stride_g + gcols, rmask, other=0.0).to(tl.float32)
    yb = (a * tl.sigmoid(g)).to(tl.bfloat16)
    tl.store(OUT + row * stride_o + cols, yb, rmask)
    _mx_epilogue(yb, row, cols, rmask, row < PADDED_M, Q, SF_SWZ, stride_q, 0, XB, N, PADDED_SF_COLS)


def gate_mul_quant_rr(a, g, gate_d=0, xb=1, num_warps=8):
    """Drop-in for norm_quant_kernels.gate_mul_quant. g: [M, N] contiguous-last-dim gate (gate_d=0), or the
    interleaved [M, >= 2N] QKV projection rows with per-head [q (gate_d) | gate (gate_d)] (gate_d = head dim)."""
    M, N = a.shape
    assert a.stride(-1) == 1 and g.stride(-1) == 1 and (N & (N - 1)) == 0
    nsf = N // 32
    psc = (nsf + 3) // 4 * 4
    pm = (M + 127) // 128 * 128
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=a.device)
    sf = torch.empty((pm * psc,), dtype=torch.uint8, device=a.device)
    if pm:
        _glue_gate_mul_quant_kernel[(triton.cdiv(pm, xb),)](
            a, g, out, q, sf, M, pm, a.stride(0), g.stride(0), out.stride(0), q.stride(0),
            N=N, BN=N, XB=xb, GATE_D=gate_d, PADDED_SF_COLS=psc, num_warps=num_warps)
    return out, q, sf


# ------------------------------------------------------------------------------------------------------------------
# Full-attention QKV prologue (port fused_qkv_prologue._ews_qkv_kernel) with HG heads per program (2-D [HG, HEAD_BLOCK]
# tiles, num_warps = 4 * HG so every head row keeps the stock per-row layout: 2 elements / thread, 32 lanes, 4 warps ->
# same sum-of-squares reduction) and an optional gate copy (GATE_COPY=0: the o_proj gate-mul reads the gate straight
# from the QKV rows, see gate_mul_quant_rr).
# ------------------------------------------------------------------------------------------------------------------
@triton.jit
def _glue_ews_qkv_kernel(
    qkv_ptr, qkv_stride_t,
    q8_ptr, k_out_ptr, gate_out_ptr,
    q_weight_ptr, k_weight_ptr, cos_sin_cache_ptr, positions_ptr,
    q8_stride_t, k_out_stride_t, gate_out_stride_t, cache_stride_p,
    positions_stride_m, positions_stride_t,
    slot_ptr, k_cache_ptr, v_cache_ptr, block_size,
    kc_stride_b, kc_stride_p, kc_stride_h, vc_stride_b, vc_stride_p, vc_stride_h,
    q_scale_ptr, k_scale_ptr, v_scale_ptr,
    n_tokens, n_slots,
    num_q_heads: tl.constexpr, num_kv_heads: tl.constexpr, head_dim: tl.constexpr,
    rotary_dim: tl.constexpr, half_rotary: tl.constexpr, eps: tl.constexpr, norm_beta: tl.constexpr,
    INPUT_DTYPE: tl.constexpr, HEAD_BLOCK: tl.constexpr, ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr, HAS_MROPE: tl.constexpr, MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr, WRITE_KV: tl.constexpr, TPP: tl.constexpr, HG: tl.constexpr,
    GATE_COPY: tl.constexpr,
):
    NQG: tl.constexpr = num_q_heads // HG
    grp = tl.program_id(1)
    is_k = grp >= NQG
    hloc = tl.arange(0, HG)
    local_head = tl.where(is_k, hloc, grp * HG + hloc)                     # [HG]
    hmask = tl.where(is_k, hloc < num_kv_heads, hloc < HG)                  # [HG]
    k_off: tl.constexpr = num_q_heads * 2 * head_dim
    v_off: tl.constexpr = num_q_heads * 2 * head_dim + num_kv_heads * head_dim
    head_offs = tl.arange(0, HEAD_BLOCK)[None, :]
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)[None, :]
    head_mask = (head_offs < head_dim) & hmask[:, None]
    rot_mask = (rot_offs < half_rotary) & hmask[:, None]
    hoff = tl.where(is_k, k_off + local_head * head_dim, local_head * 2 * head_dim)[:, None]   # [HG, 1]
    wsel = tl.where(is_k, k_weight_ptr, q_weight_ptr)
    for tt in tl.static_range(TPP):
        token = tl.program_id(0) * TPP + tt
        if token < n_tokens:
            row = qkv_ptr + token.to(tl.int64) * qkv_stride_t
            in_base = row + hoff
            x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
            var = tl.sum(x * x, axis=1)[:, None] / head_dim
            inv_rms = tl.rsqrt(var + eps)
            w = tl.load(wsel + head_offs, mask=head_offs < head_dim, other=0.0).to(tl.float32) + norm_beta
            x_norm = (x * inv_rms * w).to(INPUT_DTYPE).to(tl.float32)
            x_rot1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
            x_rot2 = tl.load(in_base + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
            w_rot1 = tl.load(wsel + rot_offs, mask=rot_offs < half_rotary, other=0.0).to(tl.float32) + norm_beta
            w_rot2 = tl.load(wsel + half_rotary + rot_offs, mask=rot_offs < half_rotary, other=0.0).to(tl.float32) + norm_beta
            x_rot1 = (x_rot1 * inv_rms * w_rot1).to(INPUT_DTYPE).to(tl.float32)
            x_rot2 = (x_rot2 * inv_rms * w_rot2).to(INPUT_DTYPE).to(tl.float32)
            pos_t = tl.load(positions_ptr + token * positions_stride_t).to(tl.int64)
            if HAS_MROPE:
                pos_h = tl.load(positions_ptr + positions_stride_m + token * positions_stride_t).to(tl.int64)
                pos_w = tl.load(positions_ptr + 2 * positions_stride_m + token * positions_stride_t).to(tl.int64)
                is_h = (rot_offs % 3 == 1) & (rot_offs < 3 * MROPE_SECTION_H)
                is_w = (rot_offs % 3 == 2) & (rot_offs < 3 * MROPE_SECTION_W)
                pos = tl.where(is_h, pos_h, tl.where(is_w, pos_w, pos_t))
            else:
                pos = pos_t + tl.zeros([1, ROT_HALF_BLOCK], tl.int64)
            cache_offset = pos * cache_stride_p
            cos = tl.load(cos_sin_cache_ptr + cache_offset + rot_offs, mask=rot_offs < half_rotary, other=0.0).to(tl.float32)
            sin = tl.load(cos_sin_cache_ptr + cache_offset + half_rotary + rot_offs, mask=rot_offs < half_rotary,
                          other=0.0).to(tl.float32)
            o1 = (x_rot1 * cos - x_rot2 * sin).to(INPUT_DTYPE).to(tl.float32)
            o2 = (x_rot2 * cos + x_rot1 * sin).to(INPUT_DTYPE).to(tl.float32)
            if is_k:
                ko = k_out_ptr + token * k_out_stride_t + local_head[:, None] * head_dim
                if HAS_PASS:
                    tl.store(ko + head_offs, x_norm, mask=head_mask & (head_offs >= rotary_dim))
                tl.store(ko + rot_offs, o1, mask=rot_mask)
                tl.store(ko + half_rotary + rot_offs, o2, mask=rot_mask)
                if WRITE_KV:
                    if token < n_slots:
                        slot = tl.load(slot_ptr + token).to(tl.int64)
                        if slot >= 0:
                            blk = slot // block_size
                            off = slot - blk * block_size
                            k_scale = tl.load(k_scale_ptr)
                            v_scale = tl.load(v_scale_ptr)
                            kd = k_cache_ptr + blk * kc_stride_b + off * kc_stride_p + local_head[:, None] * kc_stride_h
                            if HAS_PASS:
                                tl.store(kd + head_offs, tl.math.div_rn(x_norm, k_scale).to(tl.float8e4nv),
                                         mask=head_mask & (head_offs >= rotary_dim))
                            tl.store(kd + rot_offs, tl.math.div_rn(o1, k_scale).to(tl.float8e4nv), mask=rot_mask)
                            tl.store(kd + half_rotary + rot_offs, tl.math.div_rn(o2, k_scale).to(tl.float8e4nv),
                                     mask=rot_mask)
                            vv = tl.load(row + v_off + local_head[:, None] * head_dim + head_offs, mask=head_mask,
                                         other=0.0).to(tl.float32)
                            vd = v_cache_ptr + blk * vc_stride_b + off * vc_stride_p + local_head[:, None] * vc_stride_h
                            tl.store(vd + head_offs, tl.math.div_rn(vv, v_scale).to(tl.float8e4nv), mask=head_mask)
            else:
                q_scale = tl.load(q_scale_ptr)
                r = 1.0 / q_scale
                qo = q8_ptr + token * q8_stride_t + local_head[:, None] * head_dim
                if HAS_PASS:
                    qp = tl.minimum(tl.maximum(x_norm * r, -448.0, tl.PropagateNan.ALL), 448.0, tl.PropagateNan.ALL)
                    tl.store(qo + head_offs, qp.to(tl.float8e4nv), mask=head_mask & (head_offs >= rotary_dim))
                q1 = tl.minimum(tl.maximum(o1 * r, -448.0, tl.PropagateNan.ALL), 448.0, tl.PropagateNan.ALL)
                q2 = tl.minimum(tl.maximum(o2 * r, -448.0, tl.PropagateNan.ALL), 448.0, tl.PropagateNan.ALL)
                tl.store(qo + rot_offs, q1.to(tl.float8e4nv), mask=rot_mask)
                tl.store(qo + half_rotary + rot_offs, q2.to(tl.float8e4nv), mask=rot_mask)
                if GATE_COPY:
                    g = tl.load(in_base + head_dim + head_offs, mask=head_mask, other=0.0)
                    tl.store(gate_out_ptr + token * gate_out_stride_t + local_head[:, None] * head_dim + head_offs, g,
                             mask=head_mask)


EWS_VARIANTS = [dict(hg=1, tpp=2, gate_copy=True), dict(hg=2, tpp=1, gate_copy=True), dict(hg=2, tpp=2, gate_copy=True),
                dict(hg=4, tpp=1, gate_copy=True), dict(hg=8, tpp=1, gate_copy=True), dict(hg=2, tpp=1, gate_copy=False),
                dict(hg=4, tpp=1, gate_copy=False)]


def ews_launch(qkv, positions, q_weight, k_weight, cos_sin_cache, eps, num_q_heads, num_kv_heads, head_dim,
               rotary_dim, mrope_section, norm_beta, q_scale, k_scale, v_scale, slot_mapping, k_cache, v_cache,
               tpp=1, hg=2, gate_copy=True, num_warps=None):
    """Same contract as fused_qkv_prologue.launch; gate_copy=False returns gate=None (read it from qkv)."""
    T = qkv.shape[0]
    dev = qkv.device
    assert num_q_heads % hg == 0 and num_kv_heads <= hg
    q8 = torch.empty((T, num_q_heads * head_dim), dtype=torch.float8_e4m3fn, device=dev)
    k_out = torch.empty((T, num_kv_heads * head_dim), dtype=qkv.dtype, device=dev)
    gate = torch.empty((T, num_q_heads * head_dim), dtype=qkv.dtype, device=dev) if gate_copy else None
    if T == 0:
        return q8, k_out, gate
    has_mrope = positions.ndim == 2
    if has_mrope:
        pm, pt = positions.stride()
        mh, mw = mrope_section[1], mrope_section[2]
    else:
        pm, pt = 0, positions.stride(0)
        mh = mw = 0
    write_kv = slot_mapping is not None
    if write_kv:
        kc = k_cache.view(torch.float8_e4m3fn) if k_cache.dtype != torch.float8_e4m3fn else k_cache
        vc = v_cache.view(torch.float8_e4m3fn) if v_cache.dtype != torch.float8_e4m3fn else v_cache
        block_size = kc.shape[1]
        kcs = kc.stride()[:3]
        vcs = vc.stride()[:3]
        sm = slot_mapping
    else:
        kc = vc = q8
        block_size = 1
        kcs = vcs = (0, 0, 0)
        sm = q8
    head_block = triton.next_power_of_2(head_dim)
    nw = num_warps or max(1, head_block // 64) * hg
    g_out = gate if gate is not None else k_out
    grid = (triton.cdiv(T, tpp), num_q_heads // hg + 1)
    _glue_ews_qkv_kernel[grid](
        qkv, qkv.stride(0), q8, k_out, g_out, q_weight, k_weight, cos_sin_cache, positions,
        q8.stride(0), k_out.stride(0), g_out.stride(0), cos_sin_cache.stride(0), pm, pt,
        sm, kc, vc, block_size, kcs[0], kcs[1], kcs[2], vcs[0], vcs[1], vcs[2],
        q_scale, k_scale, v_scale, T, sm.shape[0] if write_kv else 0,
        num_q_heads, num_kv_heads, head_dim, rotary_dim, rotary_dim // 2, eps, norm_beta=norm_beta,
        INPUT_DTYPE=tl.bfloat16 if qkv.dtype == torch.bfloat16 else tl.float16,
        HEAD_BLOCK=head_block, ROT_HALF_BLOCK=triton.next_power_of_2(rotary_dim // 2),
        HAS_PASS=rotary_dim < head_dim, HAS_MROPE=has_mrope, MROPE_SECTION_H=mh, MROPE_SECTION_W=mw,
        WRITE_KV=write_kv, TPP=tpp, HG=hg, GATE_COPY=gate_copy, num_warps=nw, num_stages=2,
    )
    return q8, k_out, gate
