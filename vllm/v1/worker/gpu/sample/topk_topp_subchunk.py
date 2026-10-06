# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tighter candidate threshold for the split-row top-k/top-p of
vllm.v1.worker.gpu.sample.states (fast_top_k_top_p / fused_prep).

The split-row path gathers candidates >= L into a per-row list of CAP
entries; rows whose list overflows fall back to the generic
apply_top_k_top_p_triton kernel. With L = the k-th largest *chunk* maximum
(chunks of 4096 logits), BPE vocabularies, which put frequent tokens at low
ids, cluster the top logits in a few chunks, so L lies far below the row's
true k-th value and many rows overflow.

Fix (exact): also record *sub-chunk* maxima (SUB-element sub-chunks) in the
same prep / chunk-max pass and use L' = the k-th largest sub-chunk maximum
(with multiplicity), computed once per row by a bisection on
order-preserving int32 keys.
  * L <= L' because the sub-chunk maxima are a superset of the chunk maxima;
  * L' <= v_k (the row's true k-th largest value) because every sub-chunk
    maximum is a row element.
So {x >= L'} is a subset of the old candidate set and still contains every
top-k element including ties at v_k. The exact selection (value desc, index
asc; top-p on the renormalized top-k) and the threshold re-mask are
unchanged, so every row that did not overflow before gives a bit-identical
result, and far fewer rows need the fallback (whose semantics are unchanged
for rows that still overflow). Rows that used to overflow now get the fast
path's semantics (identical kept values; only exact top-p boundary
duplicates can differ from the fallback's pivot).

Bound: every candidate lies in a sub-chunk whose max is >= L', i.e. in one of
the <= k-1 sub-chunks ranked strictly above L' or in one of the t sub-chunks
whose max ties at L', so n <= (k - 1 + t) * SUB. With CAP = 2048 a k=20 row
overflows only if t >= 13 sub-chunk maxima tie exactly at L'.

Env (read at import): EWS=1 and VLLM_SAMPLER_SUBCHUNK_THRESHOLD (default 1)
enable it; VLLM_SAMPLER_SUBCHUNK_SIZE (default 64), VLLM_SAMPLER_SUBCHUNK_CAP
(default 2048).
Kernel sources are kept verbatim (Triton cache keys hash the source).
"""

# ruff: noqa: E501
# fmt: off
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

ENABLED = (os.environ.get("EWS", "0") == "1"
           and os.environ.get("VLLM_SAMPLER_SUBCHUNK_THRESHOLD", "1") == "1")
_CHUNK = 4096
_SUB = int(os.environ.get("VLLM_SAMPLER_SUBCHUNK_SIZE", "64"))
_SPC = _CHUNK // _SUB  # sub-chunks per chunk
_CAP = int(os.environ.get("VLLM_SAMPLER_SUBCHUNK_CAP", "2048"))
FAST_TOPK_KMAX = 64
_I64_MIN = tl.constexpr(-(2**63))
_I64_MAX = tl.constexpr(2**63 - 1)
_BUF: dict = {}


@triton.jit
def _ordered_i32(x):
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _ordered_to_f32(o):
    bits = tl.where(o < 0, o ^ 0x7FFFFFFF, o)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _make_key(x, idx):
    hi = _ordered_i32(x).to(tl.int64) << 32
    lo = (4294967295 - idx.to(tl.int64))
    return hi | lo


@triton.jit
def _key_value(key):
    o = (key >> 32).to(tl.int32)
    return _ordered_to_f32(o)


@triton.jit
def _store_maxima(x, CMAX, SMAX, row, c, NCH_PAD: tl.constexpr, NSUB_PAD: tl.constexpr,
                  CHUNK: tl.constexpr, SUB: tl.constexpr):
    # x: [CHUNK] fp32 with out-of-range lanes = -inf
    tl.store(CMAX + row * NCH_PAD + c, tl.max(x, axis=0))
    SPC: tl.constexpr = CHUNK // SUB
    sm = tl.max(tl.reshape(x, (SPC, SUB)), axis=1)
    tl.store(SMAX + row * NSUB_PAD + c * SPC + tl.arange(0, SPC), sm)


@triton.jit
def _chunk_max_kernel(LOGITS, stride, CMAX, SMAX, CNT, VOCAB: tl.constexpr, CHUNK: tl.constexpr,
                      NCH_PAD: tl.constexpr, NSUB_PAD: tl.constexpr, SUB: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1)
    offs = c * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(LOGITS + row.to(tl.int64) * stride + offs, mask=offs < VOCAB, other=-float("inf"))
    _store_maxima(x, CMAX, SMAX, row, c, NCH_PAD, NSUB_PAD, CHUNK, SUB)
    if c == 0:
        tl.store(CNT + row, 0)


@triton.jit
def _thresh_kernel(SMAX, K, LBUF, NSUB: tl.constexpr, NSUB_PAD: tl.constexpr):
    """L = k-th largest (with multiplicity) of the row's sub-chunk maxima: the largest key t with
    count(keys >= t) >= k, by bisection over the int32 order-preserving key space (exact)."""
    row = tl.program_id(0)
    k = tl.load(K + row)
    i = tl.arange(0, NSUB_PAD)
    m = tl.load(SMAX + row * NSUB_PAD + i, mask=i < NSUB, other=-float("inf"))
    keys = _ordered_i32(m).to(tl.int64)
    lo = tl.min(keys, axis=0)  # count(keys >= lo) = NSUB_PAD >= k always
    hi = tl.max(keys, axis=0)
    for _ in tl.static_range(33):
        mid = lo + (hi - lo + 1) // 2
        cnt = tl.sum((keys >= mid).to(tl.int32), axis=0)
        ok = cnt >= k
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid - 1)
    tl.store(LBUF + row, _ordered_to_f32(lo.to(tl.int32)))


@triton.jit
def _gather_kernel(LOGITS, stride, CMAX, CNT, CAND, LBUF,
                   VOCAB: tl.constexpr, CHUNK: tl.constexpr, NCH_PAD: tl.constexpr,
                   CAP: tl.constexpr, MASK_VALUE: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1)
    L = tl.load(LBUF + row)
    cmax = tl.load(CMAX + row * NCH_PAD + c)
    offs = c * CHUNK + tl.arange(0, CHUNK)
    ptr = LOGITS + row.to(tl.int64) * stride + offs
    inb = offs < VOCAB
    if cmax < L:
        tl.store(ptr, tl.full([CHUNK], MASK_VALUE, tl.float32), mask=inb)
    else:
        x = tl.load(ptr, mask=inb, other=-float("inf"))
        sel = (x >= L) & inb
        n = tl.sum(sel.to(tl.int32), axis=0)
        base = tl.atomic_add(CNT + row, n)
        pos = base + tl.cumsum(sel.to(tl.int32), axis=0) - 1
        keys = _make_key(x, offs)
        tl.store(CAND + row * CAP + pos, keys, mask=sel & (pos < CAP))


@triton.jit
def _select_kernel(CNT, CAND, K, P, THR, KEFF, PEFF, OVF,
                   VOCAB: tl.constexpr, CAP: tl.constexpr, KMAX: tl.constexpr,
                   TOPP: tl.constexpr, STATS: tl.constexpr):
    # identical to the split-row _select_kernel of states.py (CAP is a parameter here) + optional overflow counter
    row = tl.program_id(0)
    n = tl.load(CNT + row)
    k = tl.load(K + row)
    if TOPP:
        p = tl.load(P + row)
    else:
        p = 1.0
    if n > CAP:
        tl.store(THR + row, _I64_MIN)
        tl.store(KEFF + row, k)
        tl.store(PEFF + row, p)
        if STATS:
            tl.atomic_add(OVF, 1)
    else:
        ci = tl.arange(0, CAP)
        keys = tl.load(CAND + row * CAP + ci, mask=ci < n, other=_I64_MIN)
        j = tl.arange(0, KMAX)
        top = tl.full([KMAX], _I64_MIN, tl.int64)
        for t in tl.static_range(KMAX):
            mk = tl.max(keys, axis=0)
            top = tl.where(j == t, mk, top)
            keys = tl.where(keys == mk, _I64_MIN, keys)
        v = _key_value(top)
        valid = (j < k) & (j < n) & (v > -float("inf"))
        v0 = tl.max(tl.where(valid, v, -float("inf")), axis=0)
        e = tl.where(valid, tl.exp(v - v0), 0.0)
        prob = e / tl.sum(e, axis=0)
        excl = tl.cumsum(prob, axis=0) - prob
        keep = valid & ((excl < p) | (j == 0) | (p >= 1.0))
        thr = tl.min(tl.where(keep, top, _I64_MAX), axis=0)
        tl.store(THR + row, thr)
        tl.store(KEFF + row, VOCAB)
        tl.store(PEFF + row, 1.0)


@triton.jit
def _apply_kernel(LOGITS, stride, CMAX, LBUF, THR, CNT,
                  VOCAB: tl.constexpr, CHUNK: tl.constexpr, NCH_PAD: tl.constexpr,
                  CAP: tl.constexpr, MASK_VALUE: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.load(CNT + row)
    cmax = tl.load(CMAX + row * NCH_PAD + c)
    L = tl.load(LBUF + row)
    if (n <= CAP) & (cmax >= L):
        thr = tl.load(THR + row)
        offs = c * CHUNK + tl.arange(0, CHUNK)
        ptr = LOGITS + row.to(tl.int64) * stride + offs
        inb = offs < VOCAB
        x = tl.load(ptr, mask=inb, other=-float("inf"))
        keep = _make_key(x, offs) >= thr
        tl.store(ptr, tl.where(keep, x, MASK_VALUE), mask=inb)


@triton.jit
def _prep_kernel(
    LOGITS_IN, in_stride, OUT, out_stride, CMAX, SMAX, CNT,
    expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr,
    repetition_penalty_ptr, frequency_penalty_ptr, presence_penalty_ptr,
    prompt_bin_mask_ptr, prompt_bin_mask_stride,
    output_bin_counts_ptr, output_bin_counts_stride,
    VOCAB: tl.constexpr, CHUNK: tl.constexpr, NCH_PAD: tl.constexpr,
    NSUB_PAD: tl.constexpr, SUB: tl.constexpr, USE_PENALTY: tl.constexpr,
):
    """Byte-for-byte the math of states.py's _prep_kernel (fp32 copy + penalties) + sub-chunk maxima."""
    token_idx = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    block = c * CHUNK + tl.arange(0, CHUNK)
    mask = block < VOCAB
    logits = tl.load(LOGITS_IN + token_idx * in_stride + block, mask=mask,
                     other=-float("inf")).to(tl.float32)
    if USE_PENALTY:
        req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
        rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
        freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
        pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)
        use_rep_penalty = rep_penalty != 1.0
        use_freq_penalty = freq_penalty != 0.0
        use_pres_penalty = pres_penalty != 0.0
        if use_rep_penalty or use_freq_penalty or use_pres_penalty:
            output_bin_counts = tl.load(
                output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
                mask=mask, other=0)
            pos = tl.load(expanded_local_pos_ptr + token_idx)
            start_idx = token_idx - pos
            for prev_pos in tl.range(pos):
                prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
                output_bin_counts = output_bin_counts + (block == prev_token).to(tl.int32)
            output_bin_mask = output_bin_counts > 0
            if use_rep_penalty:
                packed_block = c * CHUNK // 32 + tl.arange(0, CHUNK // 32)
                packed_mask = tl.load(
                    prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
                    mask=packed_block < tl.cdiv(VOCAB, 32), other=0)
                prompt_bin_mask = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
                prompt_bin_mask = prompt_bin_mask.to(tl.int1).reshape(CHUNK)
                scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
                logits *= tl.where(logits > 0, 1.0 / scale, scale)
            logits -= freq_penalty * output_bin_counts
            logits -= pres_penalty * output_bin_mask
    tl.store(OUT + token_idx * out_stride + block, logits, mask=mask)
    _store_maxima(tl.where(mask, logits, -float("inf")), CMAX, SMAX, token_idx, c,
                  NCH_PAD, NSUB_PAD, CHUNK, SUB)
    if c == 0:
        tl.store(CNT + token_idx, 0)


def _geom(V):
    nch = triton.cdiv(V, _CHUNK)
    nch_pad = triton.next_power_of_2(nch)
    nsub = nch * _SPC
    nsub_pad = triton.next_power_of_2(nsub)
    return nch, nch_pad, nsub, nsub_pad


def _buffers(device, batch, V):
    nch, nch_pad, nsub, nsub_pad = _geom(V)
    b = _BUF.get(device)
    if b is None or b["cmax"].shape[0] < batch or b["V"] != V:
        cap_b = max(batch, 256)
        b = dict(
            V=V,
            cmax=torch.empty((cap_b, nch_pad), dtype=torch.float32, device=device),
            smax=torch.empty((cap_b, nsub_pad), dtype=torch.float32, device=device),
            cnt=torch.empty(cap_b, dtype=torch.int32, device=device),
            cand=torch.empty((cap_b, _CAP), dtype=torch.int64, device=device),
            lbuf=torch.empty(cap_b, dtype=torch.float32, device=device),
            thr=torch.empty(cap_b, dtype=torch.int64, device=device),
            keff=torch.empty(cap_b, dtype=torch.int32, device=device),
            peff=torch.empty(cap_b, dtype=torch.float32, device=device),
            ovf=torch.zeros(1, dtype=torch.int32, device=device),
        )
        _BUF[device] = b
    return b


def fused_prep(logits_in, penalties, expanded_idx_mapping, input_ids, expanded_local_pos):
    B, V = logits_in.shape
    out = torch.empty((B, V), dtype=torch.float32, device=logits_in.device)
    nch, nch_pad, nsub, nsub_pad = _geom(V)
    buf = _buffers(logits_in.device, B, V)
    use = penalties is not None
    dummy = buf["cnt"]
    _prep_kernel[(B, nch)](
        logits_in, logits_in.stride(0), out, out.stride(0), buf["cmax"], buf["smax"], buf["cnt"],
        expanded_idx_mapping if use else dummy, input_ids if use else dummy,
        expanded_local_pos if use else dummy,
        penalties.repetition_penalty.gpu if use else dummy,
        penalties.frequency_penalty.gpu if use else dummy,
        penalties.presence_penalty.gpu if use else dummy,
        penalties.prompt_bin_mask if use else dummy,
        penalties.prompt_bin_mask.stride(0) if use else 0,
        penalties.output_bin_counts if use else dummy,
        penalties.output_bin_counts.stride(0) if use else 0,
        VOCAB=V, CHUNK=_CHUNK, NCH_PAD=nch_pad, NSUB_PAD=nsub_pad, SUB=_SUB, USE_PENALTY=use,
        num_warps=8,
    )
    return out


def fast_top_k_top_p(logits, k, p, kmax, mask_value=float("-inf"), fallback=True, cmax_ready=False):
    """Same contract as states.fast_top_k_top_p (in-place; every k <= kmax <= 64)."""
    assert logits.ndim == 2 and logits.dtype == torch.float32
    assert 1 <= kmax <= FAST_TOPK_KMAX
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    B, V = logits.shape
    if B == 0:
        return logits
    nch, nch_pad, nsub, nsub_pad = _geom(V)
    assert nch >= kmax, "vocab too small for the chunked path"
    KM = max(16, triton.next_power_of_2(kmax))
    buf = _buffers(logits.device, B, V)
    k32 = k.to(torch.int32)
    p32 = p.to(torch.float32) if p is not None else k32
    stride = logits.stride(0)
    if not cmax_ready:
        _chunk_max_kernel[(B, nch)](logits, stride, buf["cmax"], buf["smax"], buf["cnt"], VOCAB=V,
                                    CHUNK=_CHUNK, NCH_PAD=nch_pad, NSUB_PAD=nsub_pad, SUB=_SUB,
                                    num_warps=8)
    _thresh_kernel[(B,)](buf["smax"], k32, buf["lbuf"], NSUB=nsub, NSUB_PAD=nsub_pad, num_warps=8)
    _gather_kernel[(B, nch)](logits, stride, buf["cmax"], buf["cnt"], buf["cand"], buf["lbuf"],
                             VOCAB=V, CHUNK=_CHUNK, NCH_PAD=nch_pad, CAP=_CAP, MASK_VALUE=mask_value,
                             num_warps=8)
    stats = False  # the overflow counter (host-synced diagnostics) is not wired up
    _select_kernel[(B,)](buf["cnt"], buf["cand"], k32, p32, buf["thr"], buf["keff"], buf["peff"],
                         buf["ovf"], VOCAB=V, CAP=_CAP, KMAX=KM, TOPP=p is not None, STATS=stats,
                         num_warps=8)
    _apply_kernel[(B, nch)](logits, stride, buf["cmax"], buf["lbuf"], buf["thr"], buf["cnt"],
                            VOCAB=V, CHUNK=_CHUNK, NCH_PAD=nch_pad, CAP=_CAP, MASK_VALUE=mask_value,
                            num_warps=8)
    if fallback:
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        apply_top_k_top_p_triton(logits, buf["keff"][:B], buf["peff"][:B] if p is not None else None,
                                 mask_value)
    return logits


if ENABLED:
    logger.info("sampler: sub-chunk top-k threshold enabled (sub=%d, cap=%d)", _SUB, _CAP)
