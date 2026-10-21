# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k/top-p masking for the speculative (rejection-sampling) path.

Replaces ``apply_top_k_top_p_triton`` (Qrita, one program per logits row) when
every row has an active top-k no larger than ``MAX_TOP_K``; other batches keep
Qrita. The rejection kernels are unchanged: the fp32 canvas is masked in place
with -inf, exactly as before.

Pipeline (4 launches, all full-GPU):

1. ``_submax_kernel`` (rows x V/1024): max of every 256-wide sub-block.
2. ``_lower_bound_kernel`` (rows): L = the K-th largest sub-block max, with
   K = the batch's largest top-k. The K largest sub-block maxima are K distinct
   logits, so L <= the row's K-th largest logit and {x >= L} contains the
   row's top-K. It also zeroes the candidate counters.
3. ``_gather_kernel`` (rows x V/2048): appends every finite x >= L to a per-row
   candidate buffer as a packed (value, index) key and writes -inf over the
   rest of the row.
4. ``_finalize_kernel`` (rows): orders the candidates by (value desc, index asc),
   keeps the first k, applies top-p on the renormalized survivors, and writes
   -inf over the candidates that do not survive. A row with more than
   ``_CAP`` candidates (heavily clustered or tied logits) takes an exact
   full-row fallback inside the same kernel.

Tie rule: exactly min(k, #finite) tokens survive top-k, ordered by (value desc,
token id asc), the rule Qrita documents. Top-p keeps the shortest prefix of
that order whose renormalized mass reaches p, so ties at the top-p boundary are
also resolved by token id.
"""

import torch

from vllm.triton_utils import tl, triton

# Largest per-row top-k handled here; larger values fall back to Qrita.
MAX_TOP_K = 64
_SUBMAX_BLOCK = 1024
_SUBMAX_SUB = 256
_GATHER_BLOCK = 2048
_CAP = 4096
_FALLBACK_BLOCK = 4096


@triton.jit
def _pack_keys(x, idx, vocab_size):
    # Order-preserving int64 key: high 32 bits sort like the fp32 value, low
    # 32 bits sort by descending token id, so larger key = larger value, then
    # smaller id. -0.0 is folded into +0.0 so that equal values tie.
    x = x + 0.0
    bits = x.to(tl.int32, bitcast=True)
    ikey = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return (ikey.to(tl.int64) << 32) | (vocab_size - 1 - idx).to(tl.int64)


@triton.jit
def _unpack_keys(key, vocab_size):
    ikey = (key >> 32).to(tl.int32)
    bits = tl.where(ikey < 0, ikey ^ 0x7FFFFFFF, ikey)
    value = bits.to(tl.float32, bitcast=True)
    idx = vocab_size - 1 - (key & 0xFFFFFFFF)
    return value, idx


@triton.jit
def _submax_kernel(
    logits_ptr,
    logits_stride,
    submax_ptr,
    submax_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    SUB_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(
        logits_ptr + row * logits_stride + offs,
        mask=offs < vocab_size,
        other=float("-inf"),
    )
    NUM_SUB: tl.constexpr = BLOCK_SIZE // SUB_SIZE
    submax = tl.max(tl.reshape(x, (NUM_SUB, SUB_SIZE)), axis=1)
    tl.store(
        submax_ptr + row * submax_stride + block_idx * NUM_SUB + tl.arange(0, NUM_SUB),
        submax,
    )


@triton.jit
def _prepare_kernel(
    raw_logits_ptr,
    raw_logits_stride,
    logits_ptr,
    logits_stride,
    submax_ptr,
    submax_stride,
    expanded_idx_mapping_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    temperature_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    SUB_SIZE: tl.constexpr,
    HAS_PENALTIES: tl.constexpr,
):
    """Copy, penalties, temperature and top-k submax in a single row pass."""
    row = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size
    req = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    x = tl.load(
        raw_logits_ptr + row * raw_logits_stride + offs,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    if HAS_PENALTIES:
        rep = tl.load(repetition_penalty_ptr + req)
        freq = tl.load(frequency_penalty_ptr + req)
        pres = tl.load(presence_penalty_ptr + req)
        if rep != 1.0 or freq != 0.0 or pres != 0.0:
            counts = tl.load(
                output_bin_counts_ptr + req * output_bin_counts_stride + offs,
                mask=mask,
                other=0,
            )
            pos = tl.load(expanded_local_pos_ptr + row)
            start = row - pos
            for prev in range(pos):
                token = tl.load(input_ids_ptr + start + prev + 1)
                counts += (offs == token).to(tl.int32)
            output_mask = counts > 0
            if rep != 1.0:
                packed_offs = block_idx * BLOCK_SIZE // 32 + tl.arange(
                    0, BLOCK_SIZE // 32
                )
                packed = tl.load(
                    prompt_bin_mask_ptr + req * prompt_bin_mask_stride + packed_offs,
                    mask=packed_offs < tl.cdiv(vocab_size, 32),
                    other=0,
                )
                prompt_mask = (
                    ((packed[:, None] >> tl.arange(0, 32)[None, :]) & 1)
                    .to(tl.int1)
                    .reshape(BLOCK_SIZE)
                )
                scale = tl.where(prompt_mask | output_mask, rep, 1.0)
                x *= tl.where(x > 0, 1.0 / scale, scale)
            x -= freq * counts
            x -= pres * output_mask
    temp = tl.load(temperature_ptr + req)
    if temp != 0.0 and temp != 1.0:
        x = x / temp
    x = tl.where(mask, x, float("-inf"))
    tl.store(logits_ptr + row * logits_stride + offs, x, mask=mask)
    NUM_SUB: tl.constexpr = BLOCK_SIZE // SUB_SIZE
    tl.store(
        submax_ptr + row * submax_stride + block_idx * NUM_SUB + tl.arange(0, NUM_SUB),
        tl.max(tl.reshape(x, (NUM_SUB, SUB_SIZE)), axis=1),
    )


@triton.jit
def _lower_bound_kernel(
    submax_ptr,
    submax_stride,
    bound_ptr,
    count_ptr,
    num_submax,
    max_top_k,
    KP: tl.constexpr,
    PADDED_NUM_SUBMAX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, PADDED_NUM_SUBMAX)
    submax = tl.load(
        submax_ptr + row * submax_stride + offs,
        mask=offs < num_submax,
        other=float("-inf"),
    )
    top = tl.topk(submax, KP)
    pos = tl.arange(0, KP)
    bound = tl.max(tl.where(pos == max_top_k - 1, top, float("-inf")), axis=0)
    tl.store(bound_ptr + row, bound)
    tl.store(count_ptr + row, 0)


@triton.jit
def _gather_kernel(
    logits_ptr,
    logits_stride,
    bound_ptr,
    cand_ptr,
    cand_stride,
    count_ptr,
    vocab_size,
    CAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf"))
    sel = mask & (x > float("-inf")) & (x >= tl.load(bound_ptr + row))
    num_sel = tl.sum(sel.to(tl.int32), axis=0)
    if num_sel > 0:
        base = tl.atomic_add(count_ptr + row, num_sel)
        pos = base + tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(
            cand_ptr + row * cand_stride + pos,
            _pack_keys(x, offs, vocab_size),
            mask=sel & (pos < CAP),
        )
    # Unconditional (boundary-masked) store keeps the write vectorized.
    tl.store(
        logits_ptr + row * logits_stride + offs,
        tl.where(sel, x, float("-inf")),
        mask=mask,
    )


@triton.jit
def _select_threshold(keys, top_k, top_p, KP: tl.constexpr, TOP_P: tl.constexpr):
    """Return the smallest surviving key among `keys` (top-KP picked here)."""
    if keys.shape[0] > KP:  # noqa: SIM108 (constexpr branch in Triton)
        top = tl.topk(keys, KP)
    else:
        top = tl.sort(keys, descending=True)
    pos = tl.arange(0, KP)
    # `keys` holds finite values only; padding is INT64_MIN.
    pad = -9223372036854775808
    num_valid = tl.sum(((pos < top_k) & (top != pad)).to(tl.int32))
    valid = pos < num_valid
    num_keep = num_valid
    if TOP_P:
        value, _ = _unpack_keys(top, 0)
        max_value = tl.max(tl.where(valid, value, float("-inf")), axis=0)
        e = tl.where(valid, tl.exp(value - max_value), 0.0)
        prob = e / tl.sum(e, axis=0)
        cum = tl.cumsum(prob, axis=0)
        # Keep the shortest prefix whose mass reaches p: position j survives
        # iff the mass strictly before it is < p.
        below = valid & (pos < num_valid - 1) & (cum < top_p)
        num_below = tl.sum(below.to(tl.int32))
        num_keep = tl.where(
            top_p < 1.0, tl.minimum(num_below + 1, num_valid), num_valid
        )
    return tl.min(tl.where(pos < num_keep, top, 9223372036854775807), axis=0)


@triton.jit
def _finalize_tier(
    logits_row_ptr,
    cand_row_ptr,
    count,
    top_k,
    top_p,
    vocab_size,
    CP: tl.constexpr,
    KP: tl.constexpr,
    TOP_P: tl.constexpr,
):
    offs = tl.arange(0, CP)
    cmask = offs < count
    keys = tl.load(cand_row_ptr + offs, mask=cmask, other=-9223372036854775808)
    threshold = _select_threshold(keys, top_k, top_p, KP, TOP_P)
    _, idx = _unpack_keys(keys, vocab_size)
    tl.store(logits_row_ptr + idx, float("-inf"), mask=cmask & (keys < threshold))


@triton.jit
def _ukey(x):
    # Order-preserving key of an fp32 value in [0, 2**32), as int64
    # (-0.0 folded into +0.0).
    x = x + 0.0
    bits = x.to(tl.int32, bitcast=True)
    ikey = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return ikey.to(tl.int64) + 2147483648


@triton.jit
def _finalize_fallback(
    logits_row_ptr,
    cand_row_ptr,
    top_k,
    top_p,
    vocab_size,
    KP: tl.constexpr,
    TOP_P: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Exact top-k for rows whose candidate set overflowed the buffer.

    Bit-by-bit search for the k-th largest value over the full row (32 row
    sweeps), then one sweep that gathers exactly k survivors in token-id order
    (first ids win ties), then the usual top-p selection and a masking sweep.
    Slow, but only reached for rows with >_CAP logits at or above the bound
    (e.g. constant or heavily tied logits).
    """
    NEG_INF_KEY: tl.constexpr = 0x007FFFFF  # _ukey(-inf)
    num_finite = tl.zeros((), tl.int32)
    for start in range(0, vocab_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(logits_row_ptr + offs, mask=offs < vocab_size, other=float("-inf"))
        num_finite += tl.sum((x > float("-inf")).to(tl.int32))
    target = tl.minimum(top_k, num_finite)
    prefix = tl.zeros((), tl.int64)
    for i in tl.static_range(32):
        trial = prefix + (1 << (31 - i))
        num_ge = tl.zeros((), tl.int32)
        for start in range(0, vocab_size, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            x = tl.load(
                logits_row_ptr + offs, mask=offs < vocab_size, other=float("-inf")
            )
            u = _ukey(x)
            num_ge += tl.sum(((u >= trial) & (u > NEG_INF_KEY)).to(tl.int32))
        prefix = tl.where(num_ge >= target, trial, prefix)
    num_gt = tl.zeros((), tl.int32)
    for start in range(0, vocab_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(logits_row_ptr + offs, mask=offs < vocab_size, other=float("-inf"))
        u = _ukey(x)
        num_gt += tl.sum(((u > prefix) & (u > NEG_INF_KEY)).to(tl.int32))
    num_ties = target - num_gt
    tie_seen = tl.zeros((), tl.int32)
    written = tl.zeros((), tl.int32)
    for start in range(0, vocab_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(logits_row_ptr + offs, mask=offs < vocab_size, other=float("-inf"))
        u = _ukey(x)
        finite = u > NEG_INF_KEY
        eq = (u == prefix) & finite
        tie_rank = tie_seen + tl.cumsum(eq.to(tl.int32), axis=0)
        sel = ((u > prefix) & finite) | (eq & (tie_rank <= num_ties))
        pos = written + tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(cand_row_ptr + pos, _pack_keys(x, offs, vocab_size), mask=sel)
        tie_seen += tl.sum(eq.to(tl.int32))
        written += tl.sum(sel.to(tl.int32))
    coffs = tl.arange(0, KP)
    keys = tl.load(
        cand_row_ptr + coffs, mask=coffs < written, other=-9223372036854775808
    )
    threshold = _select_threshold(keys, top_k, top_p, KP, TOP_P)
    for start in range(0, vocab_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        x = tl.load(logits_row_ptr + offs, mask=mask, other=float("-inf"))
        keep = _pack_keys(x, offs, vocab_size) >= threshold
        tl.store(logits_row_ptr + offs, tl.where(keep, x, float("-inf")), mask=mask)


@triton.jit
def _finalize_kernel(
    logits_ptr,
    logits_stride,
    cand_ptr,
    cand_stride,
    count_ptr,
    # [num_logits] logits row -> request state index
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    top_k_ptr,
    # [max_num_reqs]
    top_p_ptr,
    vocab_size,
    CAP: tl.constexpr,
    KP: tl.constexpr,
    TOP_P: tl.constexpr,
    FALLBACK_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row)
    top_k = tl.load(top_k_ptr + req_state_idx)
    top_p = 1.0
    if TOP_P:
        top_p = tl.load(top_p_ptr + req_state_idx)
    count = tl.load(count_ptr + row)
    logits_row_ptr = logits_ptr + row * logits_stride
    cand_row_ptr = cand_ptr + row * cand_stride
    # The first KP candidates in key order are enough; the tier only bounds
    # the candidate load.
    SMALL: tl.constexpr = 64 if KP <= 64 else KP
    if count <= SMALL:
        _finalize_tier(
            logits_row_ptr, cand_row_ptr, count, top_k, top_p, vocab_size,
            SMALL, KP, TOP_P,
        )  # fmt: skip
    elif count <= 512:
        _finalize_tier(
            logits_row_ptr, cand_row_ptr, count, top_k, top_p, vocab_size,
            512, KP, TOP_P,
        )  # fmt: skip
    elif count <= CAP:
        _finalize_tier(
            logits_row_ptr, cand_row_ptr, count, top_k, top_p, vocab_size,
            CAP, KP, TOP_P,
        )  # fmt: skip
    else:
        _finalize_fallback(
            logits_row_ptr, cand_row_ptr, top_k, top_p, vocab_size,
            KP, TOP_P, FALLBACK_BLOCK,
        )  # fmt: skip


def _top_k_pow2(max_top_k: int) -> int:
    return max(16, triton.next_power_of_2(max_top_k))


# Per-(device, vocab) scratch reused across calls (stream-ordered; saves the
# host cost of four allocations per step).
_SCRATCH: dict[tuple[torch.device, int], dict[str, torch.Tensor]] = {}


def _get_scratch(
    device: torch.device, num_rows: int, vocab_size: int
) -> dict[str, torch.Tensor]:
    key = (device, vocab_size)
    scratch = _SCRATCH.get(key)
    if scratch is None or scratch["count"].shape[0] < num_rows:
        rows = triton.next_power_of_2(num_rows)
        num_submax = triton.cdiv(vocab_size, _SUBMAX_BLOCK) * (
            _SUBMAX_BLOCK // _SUBMAX_SUB
        )
        scratch = {
            "count": torch.empty(rows, dtype=torch.int32, device=device),
            "bound": torch.empty(rows, dtype=torch.float32, device=device),
            "cand": torch.empty(rows, _CAP, dtype=torch.int64, device=device),
            "submax": torch.empty(rows, num_submax, dtype=torch.float32, device=device),
        }
        _SCRATCH[key] = scratch
    return scratch


def apply_spec_top_k_top_p(
    # [num_logits, vocab_size] fp32, masked in place
    logits: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [max_num_reqs] int32 per-request top-k
    top_k: torch.Tensor,
    # [max_num_reqs] fp32 per-request top-p
    top_p: torch.Tensor,
    max_top_k: int,
    use_top_p: bool,
    prepared_submax: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask `logits` to each row's top-k/top-p survivors (in place).

    Every row must have 1 <= top_k <= max_top_k <= MAX_TOP_K. `top_k`/`top_p`
    are the persistent per-request state arrays, indexed through
    `expanded_idx_mapping` inside the kernels.
    A fused preparation pass can supply `prepared_submax` while producing the
    fp32 canvas, avoiding a second full-row read in _submax_kernel.
    """
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert 1 <= max_top_k <= MAX_TOP_K
    num_rows, vocab_size = logits.shape
    if num_rows == 0:
        return logits
    device = logits.device
    kp = _top_k_pow2(max_top_k)
    scratch = _get_scratch(device, num_rows, vocab_size)
    count = scratch["count"]
    cand = scratch["cand"]
    bound = scratch["bound"]
    num_blocks = triton.cdiv(vocab_size, _SUBMAX_BLOCK)
    submax = scratch["submax"]
    num_submax = submax.shape[1]
    if prepared_submax is None:
        _submax_kernel[(num_rows, num_blocks)](
            logits,
            logits.stride(0),
            submax,
            submax.stride(0),
            vocab_size,
            BLOCK_SIZE=_SUBMAX_BLOCK,
            SUB_SIZE=_SUBMAX_SUB,
            num_warps=4,
        )
    else:
        submax = prepared_submax
        num_submax = submax.shape[1]
    _lower_bound_kernel[(num_rows,)](
        submax,
        submax.stride(0),
        bound,
        count,
        num_submax,
        max_top_k,
        KP=kp,
        # At least KP wide: with fewer sub-blocks than top-k the bound is
        # -inf and every finite logit is a candidate.
        PADDED_NUM_SUBMAX=max(triton.next_power_of_2(num_submax), kp),
        num_warps=8,
    )
    _gather_kernel[(num_rows, triton.cdiv(vocab_size, _GATHER_BLOCK))](
        logits,
        logits.stride(0),
        bound,
        cand,
        cand.stride(0),
        count,
        vocab_size,
        CAP=_CAP,
        BLOCK_SIZE=_GATHER_BLOCK,
        num_warps=4,
    )
    _finalize_kernel[(num_rows,)](
        logits,
        logits.stride(0),
        cand,
        cand.stride(0),
        count,
        expanded_idx_mapping,
        top_k,
        top_p,
        vocab_size,
        CAP=_CAP,
        KP=kp,
        TOP_P=use_top_p,
        FALLBACK_BLOCK=_FALLBACK_BLOCK,
        num_warps=4,
    )
    return logits


def prepare_spec_top_k_top_p(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    penalties: tuple[torch.Tensor, ...] | None,
    max_top_k: int,
    use_top_p: bool,
) -> torch.Tensor:
    """Fused preparation with a full, initialized fp32 top-k/top-p canvas.

    Unlike compact rejection, this canvas is safe for probabilistic drafts,
    block/synthetic/adaptive verification, watermarking and processed logprobs.
    Raw logits are never modified. Only penalties, temperature and bounded
    top-k/top-p may be active; callers must exclude other logits processors.
    """
    assert logits.stride(1) == 1 and 1 <= max_top_k <= MAX_TOP_K
    num_rows, vocab_size = logits.shape
    out = torch.empty_like(logits, dtype=torch.float32)
    if num_rows == 0:
        return out
    submax = _get_scratch(logits.device, num_rows, vocab_size)["submax"]
    if penalties is None:
        rep = freq = pres = prompt = counts = temperature
        prompt_stride = counts_stride = 0
    else:
        rep, freq, pres, prompt, counts = penalties
        prompt_stride = prompt.stride(0)
        counts_stride = counts.stride(0)
    _prepare_kernel[(num_rows, triton.cdiv(vocab_size, _SUBMAX_BLOCK))](
        logits,
        logits.stride(0),
        out,
        out.stride(0),
        submax,
        submax.stride(0),
        expanded_idx_mapping,
        input_ids,
        expanded_local_pos,
        temperature,
        rep,
        freq,
        pres,
        prompt,
        prompt_stride,
        counts,
        counts_stride,
        vocab_size,
        BLOCK_SIZE=_SUBMAX_BLOCK,
        SUB_SIZE=_SUBMAX_SUB,
        HAS_PENALTIES=penalties is not None,
        num_warps=4,
    )
    return apply_spec_top_k_top_p(
        out,
        expanded_idx_mapping,
        top_k,
        top_p,
        max_top_k,
        use_top_p,
        prepared_submax=submax,
    )
