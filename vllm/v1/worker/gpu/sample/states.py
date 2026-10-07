# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
import os
from vllm.triton_utils import tl, triton

NO_LOGPROBS = -1
_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max


class SamplingStates:
    def __init__(self, max_num_reqs: int, vocab_size: int):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size

        self.temperature = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.top_k = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.top_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.min_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.seeds = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        # Tracks whether `seed` was set explicitly by the user, so callers
        # can fall back from RNG paths that don't honor per-request seeds.
        self.seeds_set = np.zeros(max_num_reqs, dtype=bool)

        # Initialize top_k and top_p manually because 0 is an invalid value for them.
        self.top_k.np.fill(self.vocab_size)
        self.top_k.copy_to_uva()
        self.top_p.np.fill(1.0)
        self.top_p.copy_to_uva()

        self.num_logprobs = np.empty(self.max_num_reqs, dtype=np.int32)
        # -1 means no logprobs are requested.
        self.num_logprobs.fill(NO_LOGPROBS)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        self.temperature.np[req_idx] = sampling_params.temperature
        self.top_p.np[req_idx] = sampling_params.top_p
        top_k = sampling_params.top_k
        if top_k <= 0 or top_k > self.vocab_size:
            top_k = self.vocab_size
        self.top_k.np[req_idx] = top_k
        self.min_p.np[req_idx] = sampling_params.min_p

        seed = sampling_params.seed
        self.seeds_set[req_idx] = seed is not None
        if seed is None:
            seed = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
        self.seeds.np[req_idx] = seed

        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = NO_LOGPROBS
        elif num_logprobs == -1:
            num_logprobs = self.vocab_size
        self.num_logprobs[req_idx] = num_logprobs

    def apply_staged_writes(self) -> None:
        self.temperature.copy_to_uva()
        self.top_p.copy_to_uva()
        self.top_k.copy_to_uva()
        self.min_p.copy_to_uva()
        self.seeds.copy_to_uva()

    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        temp_np = self.temperature.np[idx_mapping_np]
        if np.all((temp_np == 0.0) | (temp_np == 1.0)):
            # No request requires temperature. Skip the kernel launch.
            return

        apply_temperature(logits, expanded_idx_mapping, self.temperature.gpu)

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if np.all(self.min_p.np[idx_mapping_np] == 0.0):
            # No request uses min_p. Skip the kernel launch.
            return
        apply_min_p(logits, expanded_idx_mapping, self.min_p.gpu)

    def get_top_k_top_p(
        self, expanded_idx_mapping: torch.Tensor, idx_mapping_np: np.ndarray
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        do_top_k = np.any(self.top_k.np[idx_mapping_np] != self.vocab_size)
        do_top_p = np.any(self.top_p.np[idx_mapping_np] != 1.0)
        top_k = self.top_k.gpu[expanded_idx_mapping] if do_top_k else None
        top_p = self.top_p.gpu[expanded_idx_mapping] if do_top_p else None
        return top_k, top_p

    def apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor:
        top_k, top_p = self.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
        if top_k is None and top_p is None:
            return logits
        # Split-row small-k top-k/top-p (host-side check, no sync).
        if (
            _SPLIT_ROW_TOPK
            and top_k is not None
            and logits.is_cuda
            and logits.dtype == torch.float32
        ):
            kmax = int(self.top_k.np[idx_mapping_np].max())
            if kmax <= FAST_TOPK_KMAX and triton.cdiv(logits.shape[1], _CHUNK) >= kmax:
                return fast_top_k_top_p(logits, top_k, top_p, kmax)
        return apply_top_k_top_p(logits, top_k, top_p)

    def fused_sampling_fast_path(
        self,
        logits: torch.Tensor,
        sampler,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ):
        """Fused copy+penalties+top-k/top-p, or None if not applicable.

        Applicable when the only active logits processors are penalties and top-k(<=KMAX)
        (+ optional top-p): no logit bias / bad words / thinking budget / min_p and all
        temperatures in {0, 1} (so no temperature scaling is applied before top-k/top-p).
        """
        if not (_SPLIT_ROW_TOPK and _FUSED_PREP and logits.is_cuda):
            return None
        if np.any(sampler.logit_bias_state.use_logit_bias[idx_mapping_np]):
            return None
        if int(sampler.bad_words_state.num_bad_words.np[idx_mapping_np].max()) != 0:
            return None
        tb = sampler.thinking_budget_state
        if tb.enabled and np.any(tb.use_thinking_budget[idx_mapping_np]):
            return None
        temp = self.temperature.np[idx_mapping_np]
        if not np.all((temp == 0.0) | (temp == 1.0)):
            return None
        if not np.all(self.min_p.np[idx_mapping_np] == 0.0):
            return None
        topk_np = self.top_k.np[idx_mapping_np]
        kmax = int(topk_np.max())
        if kmax > FAST_TOPK_KMAX or triton.cdiv(logits.shape[1], _CHUNK) < kmax:
            return None
        top_k, top_p = self.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
        if top_k is None:
            return None
        pen = sampler.penalties_state
        use_pen = bool(np.any(pen.use_penalty[idx_mapping_np]))
        out = fused_prep(logits, pen if use_pen else None, expanded_idx_mapping,
                         input_ids, expanded_local_pos)
        return fast_top_k_top_p(out, top_k, top_p, kmax, cmax_ready=True)

    def any_greedy(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.temperature.np[idx_mapping_np] == 0.0))

    def any_explicit_seed(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.seeds_set[idx_mapping_np]))

    def max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int:
        return int(np.max(self.num_logprobs[idx_mapping_np]))


# ============================================================================
# Split-row top-k/top-p for small k and the fused sampler prep pass.
_SPLIT_ROW_TOPK = os.environ.get('VLLM_SAMPLER_SPLIT_ROW_TOPK', '1') == '1'
_FUSED_PREP = os.environ.get('VLLM_SAMPLER_FUSED_PREP', '1') == '1'



FAST_TOPK_KMAX = 64  # largest top_k served by this path (constexpr bucket <= 64)
_CHUNK = 4096
_CAP = 256
_I64_MIN = tl.constexpr(-(2**63))
_I64_MAX = tl.constexpr(2**63 - 1)

_BUF_CACHE: dict = {}


@triton.jit
def _ordered_i32(x):
    # float32 -> int32 whose signed order matches the float order.
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _make_key(x, idx):
    # (value desc, index asc) as one signed int64 key: larger key == ranked higher.
    hi = _ordered_i32(x).to(tl.int64) << 32
    lo = (4294967295 - idx.to(tl.int64))
    return hi | lo


@triton.jit
def _key_value(key):
    o = (key >> 32).to(tl.int32)
    bits = tl.where(o < 0, o ^ 0x7FFFFFFF, o)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _chunk_max_kernel(
    LOGITS, stride, CMAX, CNT, VOCAB: tl.constexpr, CHUNK: tl.constexpr,
    NCH_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.program_id(1)
    offs = c * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(LOGITS + row.to(tl.int64) * stride + offs, mask=offs < VOCAB,
                other=-float("inf"))
    tl.store(CMAX + row * NCH_PAD + c, tl.max(x, axis=0))
    if c == 0:
        tl.store(CNT + row, 0)


@triton.jit
def _kth_largest_chunk_max(CMAX, row, k, NCH: tl.constexpr, NCH_PAD: tl.constexpr):
    i = tl.arange(0, NCH_PAD)
    m = tl.load(CMAX + row * NCH_PAD + i, mask=i < NCH, other=-float("inf"))
    # number of chunk maxima strictly greater than each entry
    gt = tl.sum((m[None, :] > m[:, None]).to(tl.int32), axis=1)
    # k-th largest (with multiplicity) = smallest m_i having < k strictly greater
    cand = tl.where((gt < k) & (i < NCH), m, float("inf"))
    return tl.min(cand, axis=0)


@triton.jit
def _gather_kernel(
    LOGITS, stride, CMAX, CNT, CAND, LBUF, K,
    VOCAB: tl.constexpr, CHUNK: tl.constexpr, NCH: tl.constexpr,
    NCH_PAD: tl.constexpr, CAP: tl.constexpr, MASK_VALUE: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.program_id(1)
    k = tl.load(K + row)
    L = _kth_largest_chunk_max(CMAX, row, k, NCH, NCH_PAD)
    if c == 0:
        tl.store(LBUF + row, L)
    cmax = tl.load(CMAX + row * NCH_PAD + c)
    offs = c * CHUNK + tl.arange(0, CHUNK)
    ptr = LOGITS + row.to(tl.int64) * stride + offs
    inb = offs < VOCAB
    if cmax < L:
        # nothing in this chunk can be in the top-k: mask without reading
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
def _select_kernel(
    CNT, CAND, K, P, THR, KEFF, PEFF,
    VOCAB: tl.constexpr, CAP: tl.constexpr, KMAX: tl.constexpr,
    TOPP: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.load(CNT + row)
    k = tl.load(K + row)
    if TOPP:
        p = tl.load(P + row)
    else:
        p = 1.0
    if n > CAP:
        # overflow: let the upstream kernel handle this row exactly
        tl.store(THR + row, _I64_MIN)
        tl.store(KEFF + row, k)
        tl.store(PEFF + row, p)
    else:
        ci = tl.arange(0, CAP)
        keys = tl.load(CAND + row * CAP + ci, mask=ci < n,
                       other=_I64_MIN)
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
        # keep is a prefix of the ranking -> threshold = smallest kept key
        thr = tl.min(tl.where(keep, top, _I64_MAX), axis=0)
        tl.store(THR + row, thr)
        tl.store(KEFF + row, VOCAB)
        tl.store(PEFF + row, 1.0)


@triton.jit
def _apply_kernel(
    LOGITS, stride, CMAX, LBUF, THR, CNT,
    VOCAB: tl.constexpr, CHUNK: tl.constexpr, NCH_PAD: tl.constexpr,
    CAP: tl.constexpr, MASK_VALUE: tl.constexpr,
):
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
    LOGITS_IN, in_stride, OUT, out_stride, CMAX, CNT,
    expanded_idx_mapping_ptr, token_ids_ptr, expanded_local_pos_ptr,
    repetition_penalty_ptr, frequency_penalty_ptr, presence_penalty_ptr,
    prompt_bin_mask_ptr, prompt_bin_mask_stride,
    output_bin_counts_ptr, output_bin_counts_stride,
    VOCAB: tl.constexpr, CHUNK: tl.constexpr, NCH_PAD: tl.constexpr,
    USE_PENALTY: tl.constexpr,
):
    """fp32 copy + (repetition/frequency/presence) penalties + per-chunk max.

    Replaces `torch.empty_like(logits, fp32).copy_(logits)` + `_penalties_kernel`
    + `_chunk_max_kernel`.  Penalty math is identical to vLLM's _penalties_kernel."""
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
    tl.store(CMAX + token_idx * NCH_PAD + c, tl.max(tl.where(mask, logits, -float("inf")), axis=0))
    if c == 0:
        tl.store(CNT + token_idx, 0)


def fused_prep(logits_in, penalties, expanded_idx_mapping, input_ids, expanded_local_pos):
    """Returns (fp32 logits with penalties applied, row count). Also fills the chunk-max
    buffer so a following fast_top_k_top_p(..., cmax_ready=True) can skip its first pass.
    `penalties` is a vLLM PenaltiesState or None."""
    B, V = logits_in.shape
    out = torch.empty((B, V), dtype=torch.float32, device=logits_in.device)
    nch = triton.cdiv(V, _CHUNK)
    nch_pad = triton.next_power_of_2(nch)
    buf = _buffers(logits_in.device, B, nch_pad)
    use = penalties is not None
    dummy = buf["cnt"]
    _prep_kernel[(B, nch)](
        logits_in, logits_in.stride(0), out, out.stride(0), buf["cmax"], buf["cnt"],
        expanded_idx_mapping if use else dummy, input_ids if use else dummy,
        expanded_local_pos if use else dummy,
        penalties.repetition_penalty.gpu if use else dummy,
        penalties.frequency_penalty.gpu if use else dummy,
        penalties.presence_penalty.gpu if use else dummy,
        penalties.prompt_bin_mask if use else dummy,
        penalties.prompt_bin_mask.stride(0) if use else 0,
        penalties.output_bin_counts if use else dummy,
        penalties.output_bin_counts.stride(0) if use else 0,
        VOCAB=V, CHUNK=_CHUNK, NCH_PAD=nch_pad, USE_PENALTY=use, num_warps=8,
    )
    return out


def _buffers(device, batch, nch_pad):
    key = device
    b = _BUF_CACHE.get(key)
    if b is None or b["cmax"].shape[0] < batch:
        cap_b = max(batch, 256)
        b = dict(
            cmax=torch.empty((cap_b, nch_pad), dtype=torch.float32, device=device),
            cnt=torch.empty(cap_b, dtype=torch.int32, device=device),
            cand=torch.empty((cap_b, _CAP), dtype=torch.int64, device=device),
            lbuf=torch.empty(cap_b, dtype=torch.float32, device=device),
            thr=torch.empty(cap_b, dtype=torch.int64, device=device),
            keff=torch.empty(cap_b, dtype=torch.int32, device=device),
            peff=torch.empty(cap_b, dtype=torch.float32, device=device),
        )
        _BUF_CACHE[key] = b
    return b


def fast_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor | None,
    kmax: int,
    mask_value: float = float("-inf"),
    fallback: bool = True,
    cmax_ready: bool = False,
) -> torch.Tensor:
    """In-place top-k(+top-p) masking; requires every k <= kmax <= FAST_TOPK_KMAX."""
    assert logits.ndim == 2 and logits.dtype == torch.float32
    assert 1 <= kmax <= FAST_TOPK_KMAX
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    B, V = logits.shape
    if B == 0:
        return logits
    nch = triton.cdiv(V, _CHUNK)
    nch_pad = triton.next_power_of_2(nch)
    assert nch >= kmax, "vocab too small for the chunked path"
    KM = max(16, triton.next_power_of_2(kmax))
    buf = _buffers(logits.device, B, nch_pad)
    k32 = k.to(torch.int32)
    p32 = p.to(torch.float32) if p is not None else k32  # dummy ptr
    stride = logits.stride(0)
    if not cmax_ready:  # else fused_prep() already wrote cmax/cnt for these rows
        _chunk_max_kernel[(B, nch)](logits, stride, buf["cmax"], buf["cnt"],
                                    VOCAB=V, CHUNK=_CHUNK, NCH_PAD=nch_pad, num_warps=8)
    _gather_kernel[(B, nch)](logits, stride, buf["cmax"], buf["cnt"], buf["cand"],
                             buf["lbuf"], k32, VOCAB=V, CHUNK=_CHUNK, NCH=nch,
                             NCH_PAD=nch_pad, CAP=_CAP, MASK_VALUE=mask_value,
                             num_warps=8)
    _select_kernel[(B,)](buf["cnt"], buf["cand"], k32, p32, buf["thr"], buf["keff"],
                         buf["peff"], VOCAB=V, CAP=_CAP, KMAX=KM,
                         TOPP=p is not None, num_warps=4)
    _apply_kernel[(B, nch)](logits, stride, buf["cmax"], buf["lbuf"], buf["thr"],
                            buf["cnt"], VOCAB=V, CHUNK=_CHUNK, NCH_PAD=nch_pad,
                            CAP=_CAP, MASK_VALUE=mask_value, num_warps=8)
    if fallback:
        # Exact fallback for rows whose candidate list overflowed (normally none):
        # upstream kernel with (k, p) for flagged rows and (V, 1.0) no-ops elsewhere.
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        apply_top_k_top_p_triton(
            logits, buf["keff"][:B], buf["peff"][:B] if p is not None else None,
            mask_value,
        )
    return logits
