# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels: (residual adds) + Gemma RMSNorm + MXFP8 (E4M3, 32-elem E8M0 block) quant.

The reduction is written exactly like the Inductor kernels vLLM generates for the Qwen3.5
`GemmaRMSNorm` (triton_red_fused__to_copy_add_fused_add_rms_norm_*):
  sumsq accumulated in an fp32 [1, RB] tile over H/RB chunks (acc + x*x), tl.sum(acc, 1), mean = sumsq / H
  (Triton '/' like Inductor), rsqrt via libdevice, y = x * rsqrt * (w.f32 + 1), bf16 round-to-nearest.
Inputs (all bf16, row-major, last dim contiguous):
  NUM_IN == 2 ("post" norm):  x = f32(A) + f32(R)                                 (no residual stored,
                                                                                    like Inductor)
  NUM_IN == 4 ("pre"  norm):  x = (f32(S) + f32(F)) + (f32(A) + f32(R)); RES_OUT = bf16(x)
The MXFP8 epilogue reproduces FlashInfer 0.6.18 `mxfp8_quantize(backend="cute-dsl")` bit-for-bit:
  amax over the 32 bf16 outputs, e8m0 = ceil-exponent(amax * f32(1/448)) with FlashInfer's subnormal rule,
  clamp 254, inv = 2^(127-e8m0) (0 if e8m0 == 0), q = cvt.rn.satfinite.e4m3(clamp(y*inv, +-448)).
  Scales are written in the linear [M, H/32] layout and/or the 128x4-swizzled layout (padded rows zeroed).
Further producers with the same MXFP8 epilogue: SiLU*mul (shared expert down_proj input), sigmoid gate-mul
(full-attention o_proj input), the GDN gated RMSNorm (GDN out_proj input) and a plain row quant.
The trtllm MoE finalize (unpermute + top-k weighting, finalizeKernelVecLoad order) is reproduced bit-exactly,
standalone or fused into the pre-norm, for the MoE finalize fold.
Kernel sources are kept verbatim (Triton cache keys hash the source).
"""
# ruff: noqa: E501
# fmt: off
import os

import torch

from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice

SF_VEC = 32
INV_E4M3_MAX = 1.0 / 448.0
INV_E4M3_MAX_C = tl.constexpr(INV_E4M3_MAX)


@triton.jit
def _nqf_norm_quant_kernel(
    S, F, A, R, W, OUT, RES_OUT, Q, SF_SWZ, SF_LIN,
    M, PADDED_M, eps,
    stride_s, stride_f, stride_a, stride_r, stride_out, stride_res, stride_q,
    H: tl.constexpr, XBLOCK: tl.constexpr, RB: tl.constexpr, NUM_IN: tl.constexpr,
    EMIT_Q: tl.constexpr, EMIT_SWZ: tl.constexpr, EMIT_LIN: tl.constexpr,
    PADDED_SF_COLS: tl.constexpr,
):
    # Same tile shape / loop structure as Inductor's triton_red_fused__to_copy_add_fused_add_rms_norm_*
    # so the fp32 sum-of-squares has the identical summation order (bit-identical bf16 outputs).
    NSF_RB: tl.constexpr = RB // 32
    row = tl.program_id(0).to(tl.int64) * XBLOCK + tl.arange(0, XBLOCK)[:, None]   # [XBLOCK, 1]
    xmask = row < M
    rbase = tl.arange(0, RB)[None, :]
    acc = tl.full([XBLOCK, RB], 0, tl.float32)
    for r0 in tl.range(0, H, RB):
        cols = r0 + rbase
        a = tl.load(A + row * stride_a + cols, xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        r = tl.load(R + row * stride_r + cols, xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        if NUM_IN == 4:
            s = tl.load(S + row * stride_s + cols, xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            f = tl.load(F + row * stride_f + cols, xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            x = (s + f) + (a + r)
        else:
            x = a + r
        acc = tl.where(xmask, acc + x * x, acc)
        if NUM_IN == 4:
            tl.store(RES_OUT + row * stride_res + cols, x.to(tl.bfloat16), xmask)
    ssum = tl.sum(acc, 1)[:, None]
    rs = libdevice.rsqrt(ssum / tl.full([1, 1], H, tl.float32) + eps)
    for r0 in tl.range(0, H, RB):
        cols = r0 + rbase
        a = tl.load(A + row * stride_a + cols, xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        r = tl.load(R + row * stride_r + cols, xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        w = tl.load(W + cols, eviction_policy='evict_last').to(tl.float32)
        if NUM_IN == 4:
            s = tl.load(S + row * stride_s + cols, xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            f = tl.load(F + row * stride_f + cols, xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            x = (s + f) + (a + r)
        else:
            x = a + r
        y = (x * rs) * (w + 1.0)
        yb = y.to(tl.bfloat16)
        tl.store(OUT + row * stride_out + cols, yb, xmask)
        if EMIT_Q:
            # rows >= M load zeros -> amax 0 -> e8m0 0: exactly FlashInfer's zeroed padding scales
            yf = tl.reshape(yb.to(tl.float32), [XBLOCK, NSF_RB, 32])
            amax = tl.max(tl.abs(yf), 2)                                   # [XBLOCK, NSF_RB]
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
            j = r0 // 32 + tl.arange(0, NSF_RB)[None, :]                   # [1, NSF_RB]
            sf = e2.to(tl.uint8)
            if EMIT_LIN:
                tl.store(SF_LIN + row * (H // 32) + j, sf, xmask)
            if EMIT_SWZ:
                off = (j % 4) + (j // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 \
                    + (row // 128) * (128 * PADDED_SF_COLS)
                tl.store(SF_SWZ + off, sf, row < PADDED_M)


# Launch config; overridable for tests (must mirror the Inductor reduction config of the unfused graph
# for bit-identical bf16 outputs).
CONFIG = {"XBLOCK": 2, "RB": 1024, "num_warps": 8}  # = the Inductor config for H=2048 (sm_107)


def norm_quant(a, r, w, eps, s=None, f=None, emit_q=True, emit_swz=True, emit_lin=True, config=None):
    """Returns (out_bf16, res_out_bf16|None, q_fp8|None, sf_swz_u8_1d|None, sf_lin_u8_1d|None)."""
    cfg = config or CONFIG
    M, H = a.shape
    assert H % 32 == 0 and H % cfg["RB"] == 0 and a.dtype == torch.bfloat16
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
    _nqf_norm_quant_kernel[(triton.cdiv(grid_rows, xb),)](
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
def _nqf_silu_mul_quant_kernel(X, Q, SF_SWZ, M, PADDED_M, stride_x, stride_q,
                               I: tl.constexpr, XBLOCK: tl.constexpr, PADDED_SF_COLS: tl.constexpr):
    """y = bf16(silu(gate) * up) exactly like Inductor's triton_poi_fused_mul_silu_slice_0
    (gate / (1 + exp(-gate)) * up in fp32), then the same MXFP8 epilogue; y itself is not stored."""
    NSF: tl.constexpr = I // 32
    row = tl.program_id(0).to(tl.int64) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    xmask = row < M
    cols = tl.arange(0, I)[None, :]
    g = tl.load(X + row * stride_x + cols, xmask, other=0.0).to(tl.float32)
    u = tl.load(X + row * stride_x + I + cols, xmask, other=0.0).to(tl.float32)
    y = (g / (libdevice.exp(-g) + 1.0)) * u
    yf = tl.reshape(y.to(tl.bfloat16).to(tl.float32), [XBLOCK, NSF, 32])
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
    qoff = tl.reshape(row * stride_q + cols, [XBLOCK, NSF, 32])
    tl.store(Q + qoff, qv, tl.reshape(xmask & (cols >= 0), [XBLOCK, NSF, 32]))
    j = tl.arange(0, NSF)[None, :]
    off = (j % 4) + (j // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 \
        + (row // 128) * (128 * PADDED_SF_COLS)
    tl.store(SF_SWZ + off, e2.to(tl.uint8), row < PADDED_M)


SILU_CONFIG = {"XBLOCK": 4, "num_warps": 4}


def silu_mul_quant(x, config=None):
    """x: [M, 2*I] bf16 (gate | up). Returns (q_fp8 [M, I], sf_swizzled_u8_1d) == FlashInfer
    mxfp8_quantize(silu_and_mul(x), is_sf_swizzled_layout=True) bit-for-bit."""
    cfg = config or SILU_CONFIG
    M, I2 = x.shape
    I = I2 // 2
    assert I % 32 == 0 and (I & (I - 1)) == 0 and (I // 32) % 4 == 0 and x.stride(-1) == 1
    nsf = I // 32
    padded_sf_cols = (nsf + 3) // 4 * 4
    padded_m = (M + 127) // 128 * 128
    q = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((padded_m * padded_sf_cols,), dtype=torch.uint8, device=x.device)
    if padded_m:
        _nqf_silu_mul_quant_kernel[(triton.cdiv(padded_m, cfg["XBLOCK"]),)](
            x, q, sf, M, padded_m, x.stride(0), q.stride(0), I=I, XBLOCK=cfg["XBLOCK"],
            PADDED_SF_COLS=padded_sf_cols, num_warps=cfg["num_warps"])
    return q, sf


# ------------------------------------------------------------------------------------------------ 4096-wide producers
@triton.jit
def _mx_epilogue(yb, row, cols, dmask, smask, Q, SF_SWZ, stride_q, j0,
                 XB: tl.constexpr, NC: tl.constexpr, PADDED_SF_COLS: tl.constexpr):
    """MXFP8 epilogue for a [XB, NC] bf16 tile at rows `row` ([XB,1]) / cols `cols` ([1,NC]) (NC % 32 == 0,
    tile starts on a 32-col boundary; j0 = first scale column). Bit-identical to FlashInfer cute-dsl."""
    NSF: tl.constexpr = NC // 32
    yf = tl.reshape(yb.to(tl.float32), [XB, NSF, 32])
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
    qoff = tl.reshape(row * stride_q + cols, [XB, NSF, 32])
    tl.store(Q + qoff, qv, tl.reshape(dmask & (cols >= 0), [XB, NSF, 32]))
    j = j0 + tl.arange(0, NSF)[None, :]
    off = (j % 4) + (j // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 \
        + (row // 128) * (128 * PADDED_SF_COLS)
    tl.store(SF_SWZ + off, e2.to(tl.uint8), smask)


@triton.jit
def _nqf_gate_mul_quant_kernel(A, G, OUT, Q, SF_SWZ, M, PADDED_M, stride_a, stride_g, stride_o, stride_q,
                               N: tl.constexpr, BN: tl.constexpr, PADDED_SF_COLS: tl.constexpr):
    """out = bf16(f32(attn) * tl.sigmoid(f32(gate)))  == Inductor triton_poi_fused_mul_sigmoid_view_0; + MXFP8."""
    row = tl.program_id(0).to(tl.int64) + tl.zeros([1, 1], tl.int64)
    rmask = row < M
    for c0 in tl.static_range(0, N, BN):
        cols = c0 + tl.arange(0, BN)[None, :]
        a = tl.load(A + row * stride_a + cols, rmask, other=0.0).to(tl.float32)
        g = tl.load(G + row * stride_g + cols, rmask, other=0.0).to(tl.float32)
        yb = (a * tl.sigmoid(g)).to(tl.bfloat16)
        tl.store(OUT + row * stride_o + cols, yb, rmask)
        _mx_epilogue(yb, row, cols, rmask, row < PADDED_M, Q, SF_SWZ, stride_q, c0 // 32, 1, BN, PADDED_SF_COLS)


def gate_mul_quant(a, g, bn=1024):
    M, N = a.shape
    assert N % bn == 0 and a.stride(-1) == 1 and g.stride(-1) == 1 and g.shape == a.shape
    nsf = N // 32
    psc = (nsf + 3) // 4 * 4
    pm = (M + 127) // 128 * 128
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=a.device)
    sf = torch.empty((pm * psc,), dtype=torch.uint8, device=a.device)
    if pm:
        _nqf_gate_mul_quant_kernel[(pm,)](a, g, out, q, sf, M, pm, a.stride(0), g.stride(0), out.stride(0),
                                          q.stride(0), N=N, BN=bn, PADDED_SF_COLS=psc, num_warps=4)
    return out, q, sf


@triton.jit(do_not_specialize=["T", "ROW0"])
def _nqf_gdn_gated_rmsnorm_quant_kernel(x_ptr, z_ptr, w_ptr, y_ptr, Q, SF_SWZ, T, ROW0, stride_z_tok, stride_q, eps,
                                        HV: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
                                        SIGMOID_GATE: tl.constexpr, PADDED_SF_COLS: tl.constexpr):
    """== _gdn_gated_rmsnorm_kernel of qwen_gdn_linear_attn (same tile, same math; y may alias x) + MXFP8 epilogue writing
    rows ROW0 + t of the [*, HV*D] fp8 / swizzled-scale buffers."""
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    mask = (offs_t < T)[:, None]
    row = offs_t.to(tl.int64)[:, None]
    xo = row * (HV * D) + i_h * D + offs_d[None, :]
    x = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + row * stride_z_tok + i_h * D + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs_d).to(tl.float32)
    var = tl.sum(x * x, axis=1) / D
    rstd = tl.rsqrt(var + eps)
    y = x * rstd[:, None] * w[None, :]
    if SIGMOID_GATE:
        y = y * tl.sigmoid(z)
    else:
        y = y * (z * tl.sigmoid(z))
    yb = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + xo, yb, mask=mask)
    grow = row + ROW0
    _mx_epilogue(yb, grow, i_h * D + offs_d[None, :], mask, mask, Q, SF_SWZ, stride_q, i_h * (D // 32), BT, D, PADDED_SF_COLS)


def gdn_gated_rmsnorm_quant_(x, z, weight, eps, activation, q, sf, row0, padded_sf_cols):
    T, HV, D = x.shape
    if T == 0:
        return
    assert x.is_contiguous() and z.stride(2) == 1 and z.stride(1) == D
    BT = 16
    _nqf_gdn_gated_rmsnorm_quant_kernel[(triton.cdiv(T, BT), HV)](
        x, z, weight, x, q, sf, T, row0, z.stride(0), q.stride(0), eps,
        HV=HV, D=D, BT=BT, SIGMOID_GATE=(activation == "sigmoid"), PADDED_SF_COLS=padded_sf_cols, num_warps=4)


@triton.jit(do_not_specialize=["LO", "HI", "M"])
def _nqf_quant_rows_kernel(X, Q, SF_SWZ, LO, HI, M, stride_x, stride_q,
                           N: tl.constexpr, BN: tl.constexpr, XB: tl.constexpr, PADDED_SF_COLS: tl.constexpr):
    """Plain MXFP8 quant of rows [LO, HI) (rows >= M are padding: zero scales, no data)."""
    row = LO + tl.program_id(0).to(tl.int64) * XB + tl.arange(0, XB)[:, None]
    in_rng = row < HI
    dmask = in_rng & (row < M)
    for c0 in tl.static_range(0, N, BN):
        cols = c0 + tl.arange(0, BN)[None, :]
        yb = tl.load(X + row * stride_x + cols, dmask, other=0.0)
        _mx_epilogue(yb, row, cols, dmask, in_rng, Q, SF_SWZ, stride_q, c0 // 32, XB, BN, PADDED_SF_COLS)


def _qr_config(nrows):
    # tuned on sm_107 for N=4096: small row counts 1x2048 / 4 warps, large 4x512 / 2 warps
    return {"XB": 1, "BN": 2048, "num_warps": 4} if nrows <= 1024 else {"XB": 4, "BN": 512, "num_warps": 2}


def fi_quant_into(x, q, sf):
    """FlashInfer's own compiled cute-dsl swizzled MXFP8 kernel (exactly what vLLM runs), writing into the given
    q [M,K] / sf [padded_M*padded_cols] buffers (all rows incl. padding) -- used when no row was fused.
    Adapter: uses FlashInfer-internal helpers of flashinfer.quantization.kernels.mxfp8_quantize; its proper home
    is a FlashInfer "quantize into caller buffers" API."""
    from flashinfer.quantization.kernels import mxfp8_quantize as fm
    from flashinfer.utils import device_support_pdl
    m, k = x.shape
    pdl = device_support_pdl(x.device)
    nsf = k // 32
    use_2t = m * nsf >= fm.MXFP8_2T_SF_THRESHOLD
    padded_m = (m + fm.ROW_TILE_SIZE - 1) // fm.ROW_TILE_SIZE * fm.ROW_TILE_SIZE
    kernel_fn, rpb = fm._get_compiled_kernel_mxfp8_swizzled(x.dtype == torch.bfloat16, k, pdl, use_2t, fm.SF_LAYOUT_128x4)
    nb = min((padded_m + rpb - 1) // rpb, fm.get_num_sm(x.device) * fm._BLOCKS_PER_SM)
    kernel_fn(x, q.view(torch.uint8), sf, m, padded_m, nb)


def quant_rows(x, q, sf, lo, hi, m, padded_sf_cols, config=None):
    if hi <= lo:
        return
    c = config or _qr_config(hi - lo)
    N = x.shape[-1]
    _nqf_quant_rows_kernel[(triton.cdiv(hi - lo, c["XB"]),)](
        x, q, sf, lo, hi, m, x.stride(0), q.stride(0), N=N, BN=c["BN"], XB=c["XB"],
        PADDED_SF_COLS=padded_sf_cols, num_warps=c["num_warps"])


@triton.jit(do_not_specialize=["LO", "HI", "LO2", "HI2", "NB1", "M"])
def _quant_rows_two_range_kernel(X, Q, SF_SWZ, LO, HI, LO2, HI2, NB1, M, stride_x, stride_q,
                                 N: tl.constexpr, BN: tl.constexpr, XB: tl.constexpr,
                                 PADDED_SF_COLS: tl.constexpr):
    """_nqf_quant_rows_kernel over rows [LO, HI) U [LO2, HI2): programs < NB1 take the first range."""
    pid = tl.program_id(0).to(tl.int64)
    first = pid < NB1
    base = tl.where(first, LO + pid * XB, LO2 + (pid - NB1) * XB)
    hi = tl.where(first, HI, HI2)
    row = base + tl.arange(0, XB)[:, None]
    in_rng = row < hi
    dmask = in_rng & (row < M)
    for c0 in tl.static_range(0, N, BN):
        cols = c0 + tl.arange(0, BN)[None, :]
        yb = tl.load(X + row * stride_x + cols, dmask, other=0.0)
        _mx_epilogue(yb, row, cols, dmask, in_rng, Q, SF_SWZ, stride_q, c0 // 32, XB, BN, PADDED_SF_COLS)


def quant_rows2(x, q, sf, r1, r2, m, padded_sf_cols, config):
    """quant_rows over two row ranges r1 = (lo, hi), r2 = (lo2, hi2) (r1 before r2) in one launch; `config`
    (= _qr_config of either range) must be the same for both ranges. Same per-row math as quant_rows."""
    (lo1, hi1), (lo2, hi2) = r1, r2
    xb = config["XB"]
    nb1 = triton.cdiv(hi1 - lo1, xb)
    nb2 = triton.cdiv(hi2 - lo2, xb)
    _quant_rows_two_range_kernel[(nb1 + nb2,)](x, q, sf, lo1, hi1, lo2, hi2, nb1, m, x.stride(0), q.stride(0),
                                               N=x.shape[-1], BN=config["BN"], XB=xb,
                                               PADDED_SF_COLS=padded_sf_cols, num_warps=config["num_warps"])


# ================================================================================================================
# MoE finalize folded into the next pre-norm, and a single-launch GDN row fixup
# (all bit-identical to the unfused chains they replace)
# ================================================================================================================
@triton.jit(do_not_specialize=["T", "L0", "H0", "L1", "H1", "L2", "H2", "L3", "H3"])
def _qgf_gdn_fixup_kernel(X, Q, SF_SWZ, SLOT, T, L0, H0, L1, H1, L2, H2, L3, H3, stride_x, stride_q,
                          N: tl.constexpr, BN: tl.constexpr, HAS_SLOT: tl.constexpr, ZERO_ONLY: tl.constexpr,
                          PADDED_SF_COLS: tl.constexpr):
    """One program per row of the GDN out_proj input (core_attn_out viewed [T, N] bf16) after the GDN core op:
      pad row (slot < 0):   bf16 row := 0 (replaces a separate pad-row zeroing kernel), fp8 row := 0, scales := 0
      row >= T (< 128-pad): swizzled scales := 0 (FlashInfer zeroes the scale padding rows)
      real row not covered by a fused producer (ranges [Li, Hi)): plain MXFP8 quant of the bf16 row
      covered row: untouched (written by a fused MXFP8 producer, e.g. the fused gated-RMSNorm+quant kernel)."""
    row = tl.program_id(0).to(tl.int64) + tl.zeros([1, 1], tl.int64)
    real = row < T
    if HAS_SLOT:
        s = tl.load(SLOT + row, mask=real, other=0)
        pad = real & (s < 0)
    else:
        pad = real & (row < 0)
    unc = real & (pad == 0) & (((row >= L0) & (row < H0)) | ((row >= L1) & (row < H1))
                               | ((row >= L2) & (row < H2)) | ((row >= L3) & (row < H3)))
    for c0 in tl.static_range(0, N, BN):
        cols = c0 + tl.arange(0, BN)[None, :]
        tl.store(X + row * stride_x + cols, tl.zeros([1, BN], tl.bfloat16), pad & (cols >= 0))
        if not ZERO_ONLY:
            yb = tl.load(X + row * stride_x + cols, unc & (cols >= 0), other=0.0)
            _mx_epilogue(yb, row, cols, unc | pad, unc | pad | (real == 0), Q, SF_SWZ, stride_q, c0 // 32, 1, BN,
                         PADDED_SF_COLS)


def gdn_fixup(x2, q, sf, slot, T, pm, psc, uncovered, zero_only=False):
    """uncovered: list of (lo, hi) host ranges (<= 4) of real rows that no fused producer wrote."""
    rng = list(uncovered) + [(0, 0)] * (4 - len(uncovered))
    assert len(rng) == 4
    grid = T if zero_only else pm
    if grid == 0:
        return
    N = x2.shape[1]
    _qgf_gdn_fixup_kernel[(grid,)](
        x2, q if q is not None else x2, sf if sf is not None else x2, slot if slot is not None else x2, T,
        rng[0][0], rng[0][1], rng[1][0], rng[1][1], rng[2][0], rng[2][1], rng[3][0], rng[3][1],
        x2.stride(0), q.stride(0) if q is not None else 0,
        N=N, BN=1024, HAS_SLOT=slot is not None, ZERO_ONLY=zero_only, PADDED_SF_COLS=psc, num_warps=4)


@triton.jit
def _qgf_fin_gather(G, WT, IDX, row, xmask, cols, stride_g, TOPK: tl.constexpr, USE_FMA: tl.constexpr,
                    XB: tl.constexpr, BH: tl.constexpr):
    """trtllm-gen moe::dev::finalize::finalizeKernelVecLoad for one tile: acc = 0; for k in order:
    acc = acc + w_k * f32(gemm2_out[perm(t, k)]) (nvcc contracts to FFMA), skipping perm == -1; bf16 RN."""
    acc = tl.zeros([XB, BH], tl.float32)
    for k in tl.static_range(TOPK):
        idx = tl.load(IDX + row * TOPK + k, xmask, other=-1)
        w = tl.load(WT + row * TOPK + k, xmask, other=0.0).to(tl.float32)
        ok = xmask & (idx >= 0)
        v = tl.load(G + idx.to(tl.int64) * stride_g + cols, ok, other=0.0).to(tl.float32)
        if USE_FMA:
            nxt = tl.fma(w, v, acc)
        else:
            nxt = acc + w * v
        acc = tl.where(ok, nxt, acc)
    return acc.to(tl.bfloat16)


@triton.jit(do_not_specialize=["M"])
def _qgf_finalize_kernel(G, WT, IDX, OUT, M, stride_g, stride_o, H: tl.constexpr, BH: tl.constexpr,
                         TOPK: tl.constexpr, XB: tl.constexpr, USE_FMA: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * XB + tl.arange(0, XB)[:, None]
    xmask = row < M
    for c0 in tl.static_range(0, H, BH):
        cols = c0 + tl.arange(0, BH)[None, :]
        s = _qgf_fin_gather(G, WT, IDX, row, xmask, cols, stride_g, TOPK, USE_FMA, XB, BH)
        tl.store(OUT + row * stride_o + cols, s, xmask & (cols >= 0))


FIN_FMA = os.environ.get("VLLM_MOE_FINALIZE_FOLD_FMA", "1") != "0"


def finalize(g2, wts, idx, M, H, out=None):
    """Standalone bit-exact finalize (used where the deferred MoE output must be materialized)."""
    topk = idx.numel() // M if M else 1
    assert g2.dim() == 2 and wts.numel() == idx.numel() and idx.numel() == M * topk
    if out is None:
        out = torch.empty((M, H), dtype=torch.bfloat16, device=g2.device)
    if M:
        _qgf_finalize_kernel[(triton.cdiv(M, 2),)](g2, wts, idx, out, M, g2.stride(0), out.stride(0), H=H, BH=1024,
                                                   TOPK=topk, XB=2, USE_FMA=FIN_FMA, num_warps=4)
    return out


@triton.jit
def _qgf_fin_x(G, WT, IDX, F, A, R, row, xmask, cols, stride_g, stride_f, stride_a, stride_r,
               TOPK: tl.constexpr, USE_FMA: tl.constexpr, XB: tl.constexpr, BH: tl.constexpr):
    s = _qgf_fin_gather(G, WT, IDX, row, xmask, cols, stride_g, TOPK, USE_FMA, XB, BH).to(tl.float32)
    f = tl.load(F + row * stride_f + cols, xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    a = tl.load(A + row * stride_a + cols, xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    r = tl.load(R + row * stride_r + cols, xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    return (s + f) + (a + r)


@triton.jit
def _qgf_norm_out(x, rs, row, xmask, cols, W, OUT, Q, SF_SWZ, PADDED_M, stride_out, stride_q, c0,
                  XBLOCK: tl.constexpr, RB: tl.constexpr, PADDED_SF_COLS: tl.constexpr):
    NSF_RB: tl.constexpr = RB // 32
    w = tl.load(W + cols, eviction_policy='evict_last').to(tl.float32)
    y = (x * rs) * (w + 1.0)
    yb = y.to(tl.bfloat16)
    tl.store(OUT + row * stride_out + cols, yb, xmask)
    yf = tl.reshape(yb.to(tl.float32), [XBLOCK, NSF_RB, 32])
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
    off = (j % 4) + (j // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 \
        + (row // 128) * (128 * PADDED_SF_COLS)
    tl.store(SF_SWZ + off, e2.to(tl.uint8), row < PADDED_M)


@triton.jit(do_not_specialize=["M", "PADDED_M"])
def _qgf_fin_norm_quant_kernel(G, WT, IDX, F, A, R, W, OUT, RES_OUT, Q, SF_SWZ, M, PADDED_M, eps,
                               stride_g, stride_f, stride_a, stride_r, stride_out, stride_res, stride_q,
                               H: tl.constexpr, XBLOCK: tl.constexpr, RB: tl.constexpr, TOPK: tl.constexpr,
                               PADDED_SF_COLS: tl.constexpr, USE_FMA: tl.constexpr):
    """== finalizeKernelVecLoad (MoE unpermute + top-k weighting, bf16 out) followed by nqf pre_norm
    (x = (s + f) + (a + r); Inductor-order RMSNorm; MXFP8 swizzled epilogue). H == 2 * RB: the row is kept in
    registers so the 9 gathered rows are read once; the sum of squares uses the same [XBLOCK, RB] accumulator
    and update order as Inductor's 2-iteration loop (bit-identical)."""
    row = tl.program_id(0).to(tl.int64) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    xmask = row < M
    rbase = tl.arange(0, RB)[None, :]
    cols0 = rbase
    cols1 = RB + rbase
    x0 = _qgf_fin_x(G, WT, IDX, F, A, R, row, xmask, cols0, stride_g, stride_f, stride_a, stride_r, TOPK, USE_FMA,
                    XBLOCK, RB)
    x1 = _qgf_fin_x(G, WT, IDX, F, A, R, row, xmask, cols1, stride_g, stride_f, stride_a, stride_r, TOPK, USE_FMA,
                    XBLOCK, RB)
    acc = tl.full([XBLOCK, RB], 0, tl.float32)
    acc = tl.where(xmask, acc + x0 * x0, acc)
    tl.store(RES_OUT + row * stride_res + cols0, x0.to(tl.bfloat16), xmask)
    acc = tl.where(xmask, acc + x1 * x1, acc)
    tl.store(RES_OUT + row * stride_res + cols1, x1.to(tl.bfloat16), xmask)
    ssum = tl.sum(acc, 1)[:, None]
    rs = libdevice.rsqrt(ssum / tl.full([1, 1], H, tl.float32) + eps)
    _qgf_norm_out(x0, rs, row, xmask, cols0, W, OUT, Q, SF_SWZ, PADDED_M, stride_out, stride_q, 0,
                  XBLOCK, RB, PADDED_SF_COLS)
    _qgf_norm_out(x1, rs, row, xmask, cols1, W, OUT, Q, SF_SWZ, PADDED_M, stride_out, stride_q, RB,
                  XBLOCK, RB, PADDED_SF_COLS)


def fin_norm_quant(g2, wts, idx, f, a, r, w, eps, config=None):
    """Returns (out_bf16, res_bf16, q_fp8, sf_swz_u8_1d) == nqf pre_norm(finalize(g2, wts, idx), f, a, r)."""
    cfg = config or CONFIG
    M, H = a.shape
    assert g2.dim() == 2 and M and idx.numel() % M == 0 and wts.numel() == idx.numel()
    assert H == 2 * cfg["RB"] and a.dtype == torch.bfloat16 and cfg["XBLOCK"] == 2
    dev = a.device
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    res = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    nsf = H // 32
    psc = (nsf + 3) // 4 * 4
    pm = (M + 127) // 128 * 128
    q = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dev)
    sf = torch.empty((pm * psc,), dtype=torch.uint8, device=dev)
    if pm:
        _qgf_fin_norm_quant_kernel[(triton.cdiv(pm, cfg["XBLOCK"]),)](
            g2, wts, idx, f, a, r, w, out, res, q, sf, M, pm, eps,
            g2.stride(0), f.stride(0), a.stride(0), r.stride(0), out.stride(0), res.stride(0), q.stride(0),
            H=H, XBLOCK=cfg["XBLOCK"], RB=cfg["RB"], TOPK=idx.numel() // M, PADDED_SF_COLS=psc, USE_FMA=FIN_FMA,
            num_warps=cfg["num_warps"], num_stages=1)
    return out, res, q, sf
