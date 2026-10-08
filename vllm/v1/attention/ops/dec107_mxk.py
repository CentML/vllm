# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dec107 MXFP4-K / FP8-V KV cache ("mxk"): writer, Hadamard rotation, and FP8 dequant of prefix pages for prefill.

Page layout (one layer, HND, page = 32 tokens, head_dim 256), a uint8 tensor ``[pages, page_bytes]`` view::

    [K data  Hkv x 32 x 128 B  (E2M1 packed: element 2j low nibble, 2j+1 high nibble)
     K scale Hkv x 32 x 8 B    (UE8M0 per 32 head-dim elements, token-major)
     V       Hkv x 32 x 256 B  (E4M3, per-layer v_scale)]

K is stored rotated by the normalized 256-point Walsh-Hadamard matrix H (H = H^T = H^-1) when ``hadamard`` is set;
the decode kernel then needs Q H (``rotate_q``), since (Q H)(K H)^T = Q K^T. The dec107 MXFP4-K decode kernel
(``dec107_kernel/dec107_mx.cu``) reads this layout with tcgen05 kind::mxf8f6f4 block32 MMAs (no dequant).

Scale choice per 32-block: k0 = ceil(log2(amax / 6)) (never clips), and k0 - 1 if its block MSE is lower ("best of 2").
"""

import torch

try:
    from vllm.triton_utils import tl, triton
except ImportError:  # Allow standalone numerical reference use.
    import triton
    import triton.language as tl

import os

D = 256
PAGE = 32
# Hadamard rotation of K (writer) and Q (attention layer, before FP8 quantization): DEC107_MXK_HADAMARD (default 1)
HADAMARD = os.environ.get("DEC107_MXK_HADAMARD", "1").strip() == "1"
KROW, KSROW, VROW = 128, 8, 256


def page_bytes(hkv: int) -> int:
    return hkv * PAGE * (KROW + KSROW + VROW)


_H = {}


def hadamard(device, dtype=torch.float32) -> torch.Tensor:
    key = (device, dtype)
    if key not in _H:
        h = torch.ones(1, 1)
        while h.shape[0] < D:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H[key] = (h / 16.0).to(device=device, dtype=dtype)
    return _H[key]


def hadamard_cpu() -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < D:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / 16.0


def rotate_q(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q [..., 256] (any float, incl. FP8 E4M3 decoded) -> (E4M3 q H, fp32 per-tensor scale) for the decode kernel;
    the caller multiplies bmm1 by the returned scale. Graph-capturable (no host sync)."""
    qf = fwht256(q)
    s = qf.abs().amax().clamp_min(1e-12) / 448.0
    return (qf / s).to(torch.float8_e4m3fn), s


@triton.jit
def _e2m1_code(a):
    c = (a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
    c += (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) + (a > 5.0).to(tl.int32)
    return c


@triton.jit
def _e2m1_value(code):
    mag = code & 7
    v = tl.where(mag < 4, mag.to(tl.float32) * 0.5, tl.where(mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))))
    return tl.where((code & 8) != 0, -v, v)


def fwht256(x: torch.Tensor) -> torch.Tensor:
    """Normalized 256-point Walsh-Hadamard transform in fp32 over the last dim.
    Uses fixed-order butterflies followed by exact power-of-two normalization."""
    s = x.shape
    y = x.float().reshape(-1, D)
    h = 1
    while h < D:
        y = y.view(-1, D // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return (y.reshape(-1, D) * 0.0625).reshape(s)


@triton.jit
def _pow2(k):
    # 2^k for integer k in [-126, 127], exact (exponent bits)
    return ((k + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _blk_err(xs, inv, sc, s):
    # one element tensor [8, 4]: returns (code, s + fma(d, d)) with d = |x| - grid * 2^k
    t = tl.abs(xs) * inv
    c = _e2m1_code(t)
    g = tl.where(c < 4, c.to(tl.float32) * 0.5, tl.where(c == 4, 2.0, tl.where(c == 5, 3.0, tl.where(c == 6, 4.0, 6.0))))
    d = tl.abs(xs) - g * sc
    return c | tl.where(xs < 0, 8, 0), tl.fma(d, d, s)


@triton.jit
def _sum4(s):
    # [8, 4] lane sums -> (l0 + l1) + (l2 + l3) per block
    a, b = tl.split(tl.reshape(s, (8, 2, 2)))       # a = (l0, l2), b = (l1, l3)
    u, w = tl.split(a + b)
    return u + w


@triton.jit
def _write_kernel(k_ptr, v_ptr, cache_ptr, slot_ptr, v_scale, stride_kt, stride_kh, stride_vt, stride_vh, stride_cp,
                  HKV: tl.constexpr, PG: tl.constexpr):
    # Program per (token, KV head). K arrives already rotated, fp32 [T, Hkv, 256].
    # x_i[j, l] = k[32 j + 8 l + i]; each lane accumulates a fixed FMA chain,
    # then the reduction combines (l0 + l1) + (l2 + l3).
    pid = tl.program_id(0)
    h = pid % HKV
    t = pid // HKV
    slot = tl.load(slot_ptr + t)
    if slot < 0:
        return
    page = (slot // PG).to(tl.int64)
    tok = slot % PG
    base = cache_ptr + page * stride_cp
    j = tl.arange(0, 8)[:, None]
    l = tl.arange(0, 4)[None, :]
    kb = k_ptr + t * stride_kt + h * stride_kh
    x0 = tl.load(kb + j * 32 + l * 8 + 0)
    x1 = tl.load(kb + j * 32 + l * 8 + 1)
    x2 = tl.load(kb + j * 32 + l * 8 + 2)
    x3 = tl.load(kb + j * 32 + l * 8 + 3)
    x4 = tl.load(kb + j * 32 + l * 8 + 4)
    x5 = tl.load(kb + j * 32 + l * 8 + 5)
    x6 = tl.load(kb + j * 32 + l * 8 + 6)
    x7 = tl.load(kb + j * 32 + l * 8 + 7)
    am = tl.maximum(tl.maximum(tl.maximum(tl.abs(x0), tl.abs(x1)), tl.maximum(tl.abs(x2), tl.abs(x3))),
                    tl.maximum(tl.maximum(tl.abs(x4), tl.abs(x5)), tl.maximum(tl.abs(x6), tl.abs(x7))))
    am = tl.max(am, axis=1)                                                     # [8] (max: order-free)
    e = ((am.to(tl.int32, bitcast=True) >> 23) & 255) - 127
    p2e = _pow2(tl.maximum(e, -126))
    k0 = tl.where(am <= p2e * 1.5, e - 2, e - 1)
    k0 = tl.where(am == 0.0, -126, tl.minimum(tl.maximum(k0, -126), 127))
    kka = k0[:, None]
    inva = _pow2(-kka)
    sca = _pow2(kka)
    sa = tl.zeros((8, 4), dtype=tl.float32)
    a0, sa = _blk_err(x0, inva, sca, sa)
    a1, sa = _blk_err(x1, inva, sca, sa)
    a2, sa = _blk_err(x2, inva, sca, sa)
    a3, sa = _blk_err(x3, inva, sca, sa)
    a4, sa = _blk_err(x4, inva, sca, sa)
    a5, sa = _blk_err(x5, inva, sca, sa)
    a6, sa = _blk_err(x6, inva, sca, sa)
    a7, sa = _blk_err(x7, inva, sca, sa)
    e0 = _sum4(sa)
    kkb = tl.maximum(k0 - 1, -126)[:, None]
    invb = _pow2(-kkb)
    scb = _pow2(kkb)
    sb = tl.zeros((8, 4), dtype=tl.float32)
    b0, sb = _blk_err(x0, invb, scb, sb)
    b1, sb = _blk_err(x1, invb, scb, sb)
    b2, sb = _blk_err(x2, invb, scb, sb)
    b3, sb = _blk_err(x3, invb, scb, sb)
    b4, sb = _blk_err(x4, invb, scb, sb)
    b5, sb = _blk_err(x5, invb, scb, sb)
    b6, sb = _blk_err(x6, invb, scb, sb)
    b7, sb = _blk_err(x7, invb, scb, sb)
    e1 = _sum4(sb)
    use1 = (e1 < e0)[:, None]
    kexp = tl.maximum(k0 - (e1 < e0).to(tl.int32), -126)
    dbase = base + (h * PG + tok) * 128 + j * 16 + l * 4
    tl.store(dbase + 0, (tl.where(use1, b0, a0) | (tl.where(use1, b1, a1) << 4)).to(tl.uint8))
    tl.store(dbase + 1, (tl.where(use1, b2, a2) | (tl.where(use1, b3, a3) << 4)).to(tl.uint8))
    tl.store(dbase + 2, (tl.where(use1, b4, a4) | (tl.where(use1, b5, a5) << 4)).to(tl.uint8))
    tl.store(dbase + 3, (tl.where(use1, b6, a6) | (tl.where(use1, b7, a7) << 4)).to(tl.uint8))
    tl.store(base + HKV * PG * 128 + (h * PG + tok) * 8 + tl.arange(0, 8), (kexp + 127).to(tl.uint8))
    dv = tl.arange(0, 256)
    v = tl.load(v_ptr + t * stride_vt + h * stride_vh + dv).to(tl.float32)
    v = tl.math.div_rn(v, v_scale)
    tl.store(base + HKV * PG * 136 + (h * PG + tok) * 256 + dv, v.to(tl.float8e4nv).to(tl.uint8, bitcast=True))


def mxk_cache_write(key: torch.Tensor, value: torch.Tensor, cache: torch.Tensor, slot_mapping: torch.Tensor,
                    v_scale: float, hadamard_k: bool = True) -> None:
    """Write key/value [T, Hkv, 256] to uint8 pages at slot_mapping; negative slots are skipped."""
    T = slot_mapping.numel()       # key/value may be padded (CUDA graphs); slot_mapping is not
    hkv, d = key.shape[1], key.shape[2]
    key, value = key[:T], value[:T]
    assert d == D and cache.dtype == torch.uint8 and cache.dim() == 2 and cache.size(1) >= page_bytes(hkv)
    assert cache.stride(1) == 1 and value.stride(2) == 1
    if T == 0:
        return
    key = (fwht256(key) if hadamard_k else key.float()).contiguous()
    _write_kernel[(T * hkv,)](key, value, cache, slot_mapping, float(v_scale), key.stride(0), key.stride(1),
                              value.stride(0), value.stride(1), cache.stride(0), HKV=hkv, PG=PAGE)


@triton.jit
def _deq_kernel(src_ptr, pages_ptr, dst_ptr, inv_k, s_sp, d_sp, HKV: tl.constexpr, PG: tl.constexpr):
    # program per (listed page, head, token) -> FP8 [K 256 | V 256] row (production FP8 layout), K in units of k_scale
    pid = tl.program_id(0)
    tok = pid % PG
    h = (pid // PG) % HKV
    i = (pid // (PG * HKV)).to(tl.int64)
    page = tl.load(pages_ptr + i).to(tl.int64)
    base = src_ptr + page * s_sp
    jj = tl.arange(0, 128)
    b = tl.load(base + (h * PG + tok) * 128 + jj).to(tl.int32)
    sc = tl.load(base + HKV * PG * 128 + (h * PG + tok) * 8 + jj // 16).to(tl.float32)
    m = tl.exp2(sc - 127.0) * inv_k
    lo = _e2m1_value(b & 15) * m
    hi = _e2m1_value(b >> 4) * m
    out = tl.interleave(lo, hi).to(tl.float8e4nv)
    drow = dst_ptr + i * d_sp + (h * PG + tok) * 512   # i is int64
    tl.store(drow + tl.arange(0, 256), out.to(tl.uint8, bitcast=True))
    v = tl.load(base + HKV * PG * 136 + (h * PG + tok) * 256 + tl.arange(0, 256))
    tl.store(drow + 256 + tl.arange(0, 256), v)


def mxk_pages_to_fp8(cache: torch.Tensor, pages: torch.Tensor, hkv: int, k_scale: float,
                     out: torch.Tensor | None = None) -> torch.Tensor:
    """Dequantize listed pages to the production FP8 layout [n, Hkv, 32, 512] (K | V). K comes out ROTATED (if it was
    written with Hadamard); prefill must then use Q H. ``k_scale`` = FP8 scale for the dequantized K."""
    n = pages.numel()
    if out is None:
        out = torch.empty((n, hkv, PAGE, 2 * D), dtype=torch.uint8, device=cache.device)
    if n:
        _deq_kernel[(n * hkv * PAGE,)](cache, pages, out, 1.0 / k_scale, cache.stride(0), out.stride(0), HKV=hkv, PG=PAGE)
    return out.view(torch.float8_e4m3fn)


# ---------------- torch reference ----------------
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def ref_quant_k(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [..., 256] fp32 (already rotated) -> (packed uint8 [..., 128], ue8m0 uint8 [..., 8]); same rule as the writer
    (exact exponent; best of {k0, k0 - 1} by block squared error; error sums may differ from the kernels' fixed order
    in the last bit, so ties can flip: use the Triton writer as the bitwise reference)."""
    xb = x.float().reshape(*x.shape[:-1], 8, 32)
    am = xb.abs().amax(-1, keepdim=True)
    e = ((am.view(torch.int32) >> 23) & 255) - 127
    p2e = torch.ldexp(torch.ones_like(am), e.clamp_min(-126).float())
    k0 = torch.where(am <= p2e * 1.5, e - 2, e - 1).clamp(-126, 127)
    k0 = torch.where(am == 0, torch.full_like(k0, -126), k0).float()

    def q(k):
        a = xb.abs() * torch.exp2(-k)
        c = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int() + (a > 2.5).int()
             + (a >= 3.5).int() + (a > 5.0).int())
        val = _E2M1.to(x.device)[c] * torch.exp2(k)
        return c, ((xb.abs() - val) ** 2).sum(-1, keepdim=True)

    c0, e0 = q(k0)
    c1, e1 = q((k0 - 1).clamp_min(-126))
    use1 = e1 < e0
    code = torch.where(use1, c1, c0) | torch.where(xb < 0, 8, 0)
    kexp = torch.where(use1, (k0 - 1).clamp_min(-126), k0)
    code = code.reshape(*x.shape[:-1], 128, 2)
    return (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8), (kexp.squeeze(-1) + 127).to(torch.uint8)


def ref_dequant_k(data: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    lut = _E2M1.to(data.device)
    d = data.long()
    vals = torch.stack([lut[d & 15], lut[d >> 4]], -1).reshape(*data.shape[:-1], 256)
    return vals * torch.exp2(sc.float() - 127).repeat_interleave(32, -1)
