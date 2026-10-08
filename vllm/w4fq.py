# SPDX-License-Identifier: Apache-2.0
"""W4A8 fake quantization (experimental, env-gated, default OFF).

Rounds MoE expert BF16 weights onto an FP4 (E2M1) grid right before the online MXFP8
quantization, so the UNCHANGED production MXFP8 trtllm-gen kernels run with expert weights
restricted to W4 values (activations stay MXFP8 exactly as in production).

W4FQ=<variant>   off (default) | mxfp4 | mxfp4m | nvfp4r | nvfp4x8 | mxfp4h8
  mxfp4   OCP MXFP4: E2M1 x UE8M0/32, scale 2^(floor(log2 amax)-2), saturating RTN.     EXACT in MXFP8
  mxfp4m  MXFP4, per-block scale exponent in {e-2, e-1} picked by min block MSE.        EXACT in MXFP8
  nvfp4r  NVFP4 E2M1 x E4M3/16 x global, with the global rounded up to a power of 2 and
          block-scale mantissa restricted to {1, 1.25, 1.5} (x2^k). EXACT in MXFP8 (ceil rule)
  nvfp4x8 true NVFP4 dequant, then the stock MXFP8 quantizer re-rounds it (APPROXIMATE).
  mxfp4h8 MXFP4 of W.H32 (block Hadamard on K), rotated back, then MXFP8 re-rounded:
          weight-side-only emulation of rotation (activations NOT rotated) (APPROXIMATE).
W4FQ_SHARED=1    also fake-quant the folded shared expert (#256)                    (default 1)
W4FQ_MTP=1       also fake-quant MTP-drafter MoE layers                              (default 1)
W4FQ_STRICT=1    exact variants: verify dequant(MXFP8) == FP4 grid bitwise; on mismatch requantize
                 with a ceil-rule torch MXFP8 quantizer; still mismatched -> raise     (default 1)
W4FQ_GAIN=ub     data-free per-expert output gain correction (default off): c_e = c13^2 * c2 with
                 c = ||W||^2 / <Q(W), W> computed from the ORIGINAL BF16 weights and the values actually stored
                 (dequantized MXFP8 incl. any FP4 grid), applied by multiplying expert e's routing weight by c_e
                 in the folded route (shared expert #256 included). Works with any W4FQ (also W4FQ=off = MXFP8).
Logs one 'W4FQ' line per tensor (rel Frobenius error, mismatches) and a 'W4FQ_SUMMARY'.
"""
import hashlib
import math

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

VARIANT = envs.W4FQ
SHARED = envs.W4FQ_SHARED
MTP = envs.W4FQ_MTP
STRICT = envs.W4FQ_STRICT
EXACT = {"mxfp4", "mxfp4m", "nvfp4r"}
VALID = EXACT | {"nvfp4x8", "mxfp4h8"}
ENABLED = VARIANT != "off"
GAIN = envs.W4FQ_GAIN
if GAIN not in ("off", "ub"):
    raise ValueError(f"W4FQ_GAIN={GAIN!r} not in ('off', 'ub')")
GAIN_ON = GAIN == "ub"
GAIN_STATS = {"tensors": 0, "blocks_applied": 0, "c": []}
if ENABLED and VARIANT not in VALID:
    raise ValueError(f"W4FQ={VARIANT!r} not in {sorted(VALID)} or 'off'")
try:
    with open(__file__, "rb") as _source:
        SHA = hashlib.sha256(_source.read()).hexdigest()[:12]
except OSError:
    SHA = "unknown"
TOT = {"tensors": 0, "experts": 0, "se": 0.0, "sw": 0.0, "mismatch": 0, "requant": 0, "skipped": 0}
_ANNOUNCED = False

_E2M1 = {}
def _grid(dev):
    if dev not in _E2M1:
        g = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=dev)
        _E2M1[dev] = (g, (g[1:] + g[:-1]) / 2)
    return _E2M1[dev]

def _rnd(x):
    g, mid = _grid(x.device)
    return torch.sign(x) * g[torch.bucketize(x.abs().clamp(max=6.0), mid)]

def _fl2(a):
    return torch.floor(torch.log2(a.clamp(min=2.0 ** -126)))

def _blk(w, b):
    return w.reshape(*w.shape[:-1], w.shape[-1] // b, b)

def _mxfp4(w, mse=False):
    x = _blk(w, 32); e = _fl2(x.abs().amax(-1, keepdim=True)) - 2
    q0 = _rnd(x / torch.exp2(e)) * torch.exp2(e)
    if mse:
        q1 = _rnd(x / torch.exp2(e + 1)) * torch.exp2(e + 1)
        m0 = ((q0 - x) ** 2).sum(-1, keepdim=True); m1 = ((q1 - x) ** 2).sum(-1, keepdim=True)
        q0 = torch.where(m1 < m0, q1, q0)
    return q0.reshape(w.shape)

def _nvfp4(w, restricted):
    g = w.abs().amax() / (6.0 * 448.0)
    if g == 0:
        return torch.zeros_like(w)
    if restricted:
        g = torch.exp2(torch.ceil(torch.log2(g)))
    x = _blk(w, 16)
    s = (x.abs().amax(-1, keepdim=True) / 6.0 / g).clamp(2.0 ** -9, 448.0)
    s = s.to(torch.float8_e4m3fn).float()
    if restricted:
        e = _fl2(s); m = s / torch.exp2(e)
        sg = torch.tensor([1.0, 1.25, 1.5, 2.0], device=w.device)
        s = sg[(m.unsqueeze(-1) - sg).abs().argmin(-1)] * torch.exp2(e)
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (_rnd(x / (s * g)) * s * g).reshape(w.shape)

_H = {}
def _h32(dev):
    if dev not in _H:
        H = torch.ones(1, 1)
        while H.shape[0] < 32:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _H[dev] = (H / math.sqrt(32)).to(dev)
    return _H[dev]

TINY = 2.0 ** -100  # 32-blocks with amax below this are flushed to exact 0 (see flush_tiny)


def flush_tiny(w):
    """Flush tiny blocks whose scales cannot round-trip through GPU quantization.

    Returns the weight tensor and the count of flushed blocks.
    """
    b = _blk(w, 32)
    tiny = (b.abs().amax(-1, keepdim=True) < TINY) & (b.abs().amax(-1, keepdim=True) > 0)
    n = int(tiny.sum())
    if n:
        b = torch.where(tiny, torch.zeros_like(b), b)
    return b.reshape(w.shape), n


def fq_one(w):
    """w: [N, K] float32, blocks along K (last dim). Returns float32 fake-quant."""
    if VARIANT == "mxfp4": return _mxfp4(w)
    if VARIANT == "mxfp4m": return _mxfp4(w, mse=True)
    if VARIANT == "nvfp4r": return _nvfp4(w, True)
    if VARIANT == "nvfp4x8": return _nvfp4(w, False)
    if VARIANT == "mxfp4h8":
        H = _h32(w.device)
        r = (_blk(w, 32) @ H).reshape(w.shape)
        return (_blk(_mxfp4(r), 32) @ H.T).reshape(w.shape)
    raise AssertionError(VARIANT)

def want(name: str, shared: bool = False) -> bool:
    global _ANNOUNCED
    if not ENABLED:
        return False
    if not _ANNOUNCED:
        _ANNOUNCED = True
        logger.warning("W4FQ ENABLED variant=%s shared=%s mtp=%s strict=%s sha=%s (TEST ONLY: expert weights "
                       "restricted to an FP4 grid)", VARIANT, SHARED, MTP, STRICT, SHA)
    if shared and not SHARED:
        TOT["skipped"] += 1; return False
    if "mtp" in (name or "") and not MTP:
        TOT["skipped"] += 1; return False
    return True

@torch.no_grad()
def apply_(w: torch.Tensor, name: str) -> None:
    """In place on a BF16/FP16 [E, N, K] or [N, K] weight."""
    v = w if w.dim() == 3 else w.unsqueeze(0)
    se = sw = 0.0
    nfl = 0
    for e in range(v.shape[0]):
        x = v[e].float(); q, n = flush_tiny(fq_one(x)); nfl += n
        qb = q.to(v.dtype)
        if VARIANT in EXACT:
            assert torch.equal(qb.float(), q), f"W4FQ {name}: FP4 grid not exact in {v.dtype}"
        se += float(((q - x) ** 2).sum()); sw += float((x ** 2).sum())
        v[e].copy_(qb)
    TOT["tensors"] += 1; TOT["experts"] += v.shape[0]; TOT["se"] += se; TOT["sw"] += sw
    TOT["flushed_blocks"] = TOT.get("flushed_blocks", 0) + nfl
    logger.info("W4FQ %s variant=%s E=%d shape=%s rel=%.5f flushed_tiny_blocks=%d", name, VARIANT, v.shape[0],
                tuple(v.shape[1:]), math.sqrt(se / max(sw, 1e-30)), nfl)

def _ceil_quant(x):
    """Torch MXFP8 (E4M3 x UE8M0/32), ceil scale rule: [N, K] -> (fp8 [N,K], uint8 [N,K/32])."""
    b = _blk(x.double(), 32); amax = b.abs().amax(-1, keepdim=True)   # float64: no denormal flush on GPU
    e = torch.ceil(torch.log2((amax / 448.0).clamp(min=2.0 ** -127))).clamp(-127, 127)
    q = (b / torch.exp2(e)).clamp(-448, 448).float().to(torch.float8_e4m3fn)
    return q.reshape(x.shape), (e.squeeze(-1) + 127).to(torch.uint8)

def _deq(q, s, shape):
    s = s.reshape(*shape[:-1], shape[-1] // 32).view(torch.uint8).float() if s.dtype != torch.uint8 else \
        s.reshape(*shape[:-1], shape[-1] // 32).float()
    return (_blk(q.double().reshape(shape), 32) * torch.exp2(s.double() - 127).unsqueeze(-1)).reshape(shape).float()

@torch.no_grad()
def verify_(wbf, q, s, name):
    """Exact variants: dequant(q, s) must equal the BF16 grid bitwise. Fixes (q, s) in place if not."""
    if VARIANT not in EXACT or not STRICT:
        return
    v = wbf if wbf.dim() == 3 else wbf.unsqueeze(0)
    qq = q if q.dim() == 3 else q.unsqueeze(0)
    ss = s if s.dim() == 3 else s.unsqueeze(0)
    t_mism = t_exp = 0
    for e in range(v.shape[0]):
        ref = v[e].float()
        mism = int((_deq(qq[e], ss[e], ref.shape) != ref).sum())
        if mism:
            t_mism += mism; t_exp += 1
            nq, ns = _ceil_quant(ref)
            if int((_deq(nq, ns, ref.shape) != ref).sum()):
                raise RuntimeError(f"W4FQ {name} expert {e}: grid not representable in MXFP8 even with ceil rule")
            qq[e].copy_(nq.view(qq.dtype)); ss[e].copy_(ns.reshape(ss[e].shape).view(ss.dtype))
    TOT["mismatch"] += t_mism; TOT["requant"] += t_exp; TOT["verified"] = TOT.get("verified", 0) + 1
    logger.info("W4FQ_VERIFY %s stock_quant_mismatch_elems=%d repaired_experts=%d/%d", name, t_mism, t_exp,
                v.shape[0])

def summary(tag=""):
    if ENABLED:
        logger.warning("W4FQ_SUMMARY %s variant=%s sha=%s tensors=%d experts=%d rel=%.5f stock_quant_mismatch_elems=%d "
                       "requant_experts=%d skipped=%d verified_tensors=%d flushed_tiny_blocks=%d", tag, VARIANT, SHA, TOT["tensors"], TOT["experts"],
                       math.sqrt(TOT["se"] / max(TOT["sw"], 1e-30)), TOT["mismatch"], TOT["requant"], TOT["skipped"], TOT.get("verified", 0), TOT.get("flushed_blocks", 0))


# ----------------------------------------------------------------------------- real W4A8 kernel (W4REAL=1)
# Test only. Needs an exact MXFP4 variant (W4FQ=mxfp4|mxfp4m). The folded MoE blocks then run
# FlashInfer trtllm_fp4_block_scale_routed_moe (trtllm-gen bmm_*MxE2m1MxE4m3*: MXFP4 weights, MXFP8 activations)
# on the SAME FP4-grid weights the fake-quant arm uses, so W4REAL=1 vs 0 differs only in kernel (fp32 order).
REAL = envs.W4REAL
if REAL and VARIANT not in ("mxfp4", "mxfp4m"):
    raise ValueError("W4REAL=1 needs W4FQ=mxfp4 or mxfp4m")
if REAL and not SHARED:
    raise ValueError("W4REAL=1 needs W4FQ_SHARED=1 (the folded shared expert shares the W4A8 GEMM)")
REAL_STATS = {"encoded": 0, "blocks_real": 0, "calls": 0}


@torch.no_grad()
def encode_mxfp4(w: torch.Tensor):
    """Exact MXFP4 encoding of an FP4-grid tensor [..., K] (blocks of 32 along K).
    Returns (packed uint8 [..., K/2] low nibble = even element, ue8m0 uint8 [..., K/32]). Raises if not exact."""
    g, _ = _grid(w.device)
    x = _blk(w.float(), 32)
    amax = x.abs().amax(-1, keepdim=True)
    best = None
    for d in (2, 1, 3):  # candidate scale exponents: floor(log2 amax) - d
        e = _fl2(amax) - d
        y = x / torch.exp2(e)
        idx = (y.abs().unsqueeze(-1) == g).to(torch.uint8).argmax(-1)
        ok = (g[idx.long()] == y.abs()).all(-1, keepdim=True) | (amax == 0)
        if best is None:
            best = (e, idx, y, ok)
        else:
            be, bi, by, bok = best
            take = ~bok & ok
            best = (torch.where(take, e, be), torch.where(take, idx, bi), torch.where(take, y, by), bok | ok)
    e, idx, y, ok = best
    if not bool(ok.all()):
        raise RuntimeError(f"W4REAL: {int((~ok).sum())} blocks not on the MXFP4 grid")
    code = (idx.to(torch.uint8) | (((y < 0) & (idx > 0)).to(torch.uint8) << 3)).reshape(w.shape)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8).contiguous()
    assert packed.dtype == torch.uint8
    sc = (e.squeeze(-1) + 127).clamp(0, 254).to(torch.uint8)
    REAL_STATS["encoded"] += 1
    return packed, sc


@torch.no_grad()
def to_trtllm_mxfp4(p13, s13, p2, s2, cache):
    """vLLM convert_weight_to_mxfp4_moe_kernel_format (TRTLLM) on packed codes. p13 [E, 2I, H/2] in vLLM
    [gate; up] order. Returns (w13 u8, s13 f8, w2 u8, s2 f8) in trtllm-gen layout."""
    from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache
    E, N13, Kh = p13.shape
    I = N13 // 2
    H = Kh * 2
    p13 = torch.stack([p13[:, I:], p13[:, :I]], dim=2).reshape(E, N13, Kh)
    s13 = torch.stack([s13[:, I:], s13[:, :I]], dim=2).reshape(E, N13, -1)
    perm = get_w2_permute_indices_with_cache(cache, p13[0], 128).to(p13.device)
    w13 = p13[:, perm].contiguous()
    sp = get_w2_permute_indices_with_cache(cache, s13[0], 128, num_elts_per_sf=16).to(p13.device)
    t = s13[:, sp].contiguous()
    s13o = nvfp4_block_scale_interleave(t.reshape(E * N13, -1)).reshape(E, N13, H // 32).view(torch.float8_e4m3fn)
    perm2 = get_w2_permute_indices_with_cache(cache, p2[0], 128).to(p2.device)
    w2 = p2[:, perm2].contiguous()
    sp2 = get_w2_permute_indices_with_cache(cache, s2[0], 128, num_elts_per_sf=16).to(p2.device)
    t2 = s2[:, sp2].contiguous()
    s2o = nvfp4_block_scale_interleave(t2.reshape(E * H, -1)).reshape(E, H, I // 32).view(torch.float8_e4m3fn)
    return w13, s13o, w2, s2o


# ----------------------------------------------------------------------------- data-free gain correction (W4FQ_GAIN=ub)
_GAIN_ANNOUNCED = False


def gain_on() -> bool:
    global _GAIN_ANNOUNCED
    if GAIN_ON and not _GAIN_ANNOUNCED:
        _GAIN_ANNOUNCED = True
        logger.warning("W4FQ_GAIN ENABLED mode=ub variant=%s sha=%s (TEST ONLY: per-expert routing-weight gain c_e = "
                       "c13^2*c2, c=||W||^2/<Q,W>, data-free)", VARIANT, SHA)
    return GAIN_ON


@torch.no_grad()
def gain_from(orig: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Per-expert unbiased gain factor ||W||^2/<Q,W> (float64 sums). orig: BF16 [E,N,K] or [N,K] original weights;
    q/s: the stored MXFP8 values / UE8M0 scales (linear layout, [E,N,K] / [E,N,K/32]). Returns float32 [E]."""
    o = orig if orig.dim() == 3 else orig.unsqueeze(0)
    qq = q if q.dim() == 3 else q.unsqueeze(0)
    ss = s if s.dim() == 3 else s.unsqueeze(0)
    out = torch.empty(o.shape[0], dtype=torch.float32, device=o.device)
    for e in range(o.shape[0]):
        w = o[e].double(); d = _deq(qq[e], ss[e], w.shape).double()
        qw = float((d * w).sum()); ww = float((w * w).sum())
        out[e] = (ww / qw) if qw > 0 else 1.0
    GAIN_STATS["tensors"] += 1
    return out


def gain_summary(tag=""):
    if GAIN_ON and GAIN_STATS["c"]:
        c = torch.cat(GAIN_STATS["c"])
        logger.warning("W4FQ_GAIN_SUMMARY %s variant=%s sha=%s blocks_applied=%d experts=%d c_mean=%.5f c_min=%.5f "
                       "c_max=%.5f", tag, VARIANT, SHA, GAIN_STATS["blocks_applied"], c.numel(), float(c.mean()),
                       float(c.min()), float(c.max()))
