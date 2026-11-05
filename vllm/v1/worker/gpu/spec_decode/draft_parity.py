# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H53: draft / target sampling-parameter parity for probabilistic drafts.

With ``draft_sample_method="probabilistic"`` the drafter samples q from the
temperature-scaled raw draft softmax and ignores the requests' penalties,
top-k and top-p (``speculator.py`` ``_copy_request_inputs``). The target p is
penalized, temperature-scaled and top-k / top-p masked. Draft mass outside the
target's support is always rejected.

``VLLM_H53_DIAG=1`` (log-only, no change to any output): on every
``VLLM_H53_DIAG_EVERY``-th unfused verify step, for the first
``VLLM_H53_DIAG_MAX_REQS`` requests, it rebuilds q_proc, the cached raw draft
logits put through the target's own processing (``apply_sampling_params``:
the same penalties, with the same draft-token history, temperature, top-k and
top-p), and measures per draft position:

* ``alpha_raw`` = sum_x min(p, q_raw): the acceptance probability of today's
  draft; ``alpha_proc`` = sum_x min(p, q_proc); ``alpha_t0.9`` / ``alpha_t0.8``
  the same with q_proc sharpened by 1/0.9 and 1/0.8 on its own support;
* ``out_tok``: share of drafted tokens with p = 0 (outside the target's
  top-k / top-p set); ``out_mass``: q_raw mass outside the target's support.

and per request the chain estimate E[accepted] = sum_j prod_{i<=j} alpha_i for
each q, next to the measured accepted count of the same steps. Every
``VLLM_H53_DIAG_LOG_S`` seconds it logs one cumulative ``H53 diag`` line.
Only batches the fused preparation covers (penalties, temperature and top-k
<= 64 / top-p, no other logits processor) are observed, so the extra
``apply_sampling_params`` call has no side effects.

``VLLM_H53_DRAFT_PARITY=1`` (needs ``draft_sample_method="probabilistic"``):

* Draft: each draft row is processed as the target processes its verify row:
  the request's repetition / frequency / presence penalties over
  ``output_bin_counts`` plus drafts 1..j of this proposal (the same history the
  target's verify row j counts), rounded to the draft-logits cache dtype, then
  top-k (by value desc, token id asc, at most ``VLLM_H53_DRAFT_KP`` = 32
  survivors), temperature, and top-p on the renormalized survivors. The draft is
  a Gumbel-max over the survivors with the stock draft noise stream. The
  survivors (pre-temperature values, token ids), their count and the
  logsumexp are cached per (request, step): q is sparse. Two kernels per draft
  step, both CUDA-graph safe (device-side parameters only).
* Verify: when ``fused_rejection_sample`` covers the batch, the same compact
  target survivors are tested against that sparse q: accept iff
  log p(x) > log u + log q(x); on rejection, resample from max(0, p - q) over
  the target survivors; the bonus token from p. Otherwise the sparse q is
  written into the dense draft-logits cache (-inf off its support) and the
  stock unfused path runs: the same q, value for value.

The verifier always uses the q the draft was drawn from, so the output
distribution is the target's for any processing (Leviathan et al. 2023); the
processing only moves acceptance. Same draft head, same weights.
"""

import os
import time

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton
from vllm.v1.worker.gpu.glue_pdl import glue_pdl
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_logits, tl_rand32
from vllm.v1.worker.gpu.sample.spec_topk_topp import (
    MAX_TOP_K,
    _pack_keys,
    _top_k_pow2,
    _unpack_keys,
)
from vllm.v1.worker.gpu.spec_decode.fused_rejection import (
    _BLOCK_SIZE,
    _GROUP,
    _SUB_SIZE,
    _penalize,
    select_survivors,
)

logger = init_logger(__name__)

# On by default (off: VLLM_H53_DRAFT_PARITY=0). It acts only on drafters with
# draft_sample_method="probabilistic"; greedy drafts keep the stock path.
PARITY = os.environ.get("VLLM_H53_DRAFT_PARITY", "1") == "1"
_PARITY_SET = "VLLM_H53_DRAFT_PARITY" in os.environ
PARITY_KP = int(os.environ.get("VLLM_H53_DRAFT_KP", "32"))
# H97: the same draft draw and q cache from restructured kernels (see
# _h97_select_kernel).
FAST_SELECT = os.environ.get("VLLM_H97_FAST_SELECT", "0") == "1"
# H105: the parity kernels launch with PDL under the glue convention (wait,
# then trigger) while VLLM_GLUE_PDL is in force.
H105_PDL = os.environ.get("VLLM_H105_PDL", "0") == "1"
# H159 (sampler diet), see the H159 section below.
# VLLM_H159_TAIL: the parity verify tail in two launches; bit-identical outputs.
H159_TAIL = os.environ.get("VLLM_H159_TAIL", "0") == "1"


def _pdl() -> bool:
    if not H105_PDL:
        return False
    from vllm.v1.worker.gpu.glue_pdl import glue_pdl

    return glue_pdl()

DIAG = os.environ.get("VLLM_H53_DIAG", "0") == "1"
DIAG_EVERY = max(1, int(os.environ.get("VLLM_H53_DIAG_EVERY", "8")))
DIAG_MAX_REQS = max(1, int(os.environ.get("VLLM_H53_DIAG_MAX_REQS", "32")))
DIAG_LOG_S = float(os.environ.get("VLLM_H53_DIAG_LOG_S", "60"))
# Draft temperature multipliers tried on q_proc's support (exact for any q).
_SHARPEN = (0.9, 0.8)
# Per-row stats: valid, alpha_raw, alpha_proc, alpha_t0.9, alpha_t0.8,
# q_raw mass outside p's support, q_proc mass outside it, draft token outside.
_NUM_STATS = 8
_NUM_Q = 4  # raw, proc, t0.9, t0.8
_REQUIRE_CUDA = True  # the fused preparation kernels; CPU tests clear it


@triton.jit
def _online_lse(m, s, x):
    new_m = tl.maximum(m, tl.max(x, axis=0))
    safe = tl.where(new_m == float("-inf"), 0.0, new_m)
    s = s * tl.exp(m - safe) + tl.sum(tl.exp(x - safe), axis=0)
    return new_m, s


@triton.jit
def _finish_lse(m, s):
    return tl.where(s > 0.0, m + tl.log(s), float("-inf"))


@triton.jit
def _diag_row_kernel(
    # [n, NUM_STATS] fp32
    out_ptr,
    # [n, V] processed target logits (post-temperature, -inf outside support)
    p_ptr,
    p_stride,
    # [n, V] raw draft logits (pre-temperature)
    qr_ptr,
    qr_stride,
    # [n, V] processed draft logits (post-temperature)
    qp_ptr,
    qp_stride,
    # [n] the draft token verified at the row, -1 if none
    draft_tok_ptr,
    # [n] request state index
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    inv_t1,
    inv_t2,
    NUM_STATS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, NUM_STATS)
    tok = tl.load(draft_tok_ptr + row).to(tl.int64)
    req = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    if (tok < 0) | (temp == 0.0):
        tl.store(out_ptr + row * NUM_STATS + cols, tl.zeros((NUM_STATS,), tl.float32))
        return

    neg = float("-inf")
    m_p = tl.full((), neg, tl.float32)
    s_p = tl.zeros((), tl.float32)
    m_r = m_p
    s_r = s_p
    m_q = m_p
    s_q = s_p
    m_1 = m_p
    s_1 = s_p
    m_2 = m_p
    s_2 = s_p
    for start in range(0, vocab_size, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < vocab_size
        p = tl.load(p_ptr + row * p_stride + offs, mask=mask, other=neg).to(tl.float32)
        qr = (
            tl.load(qr_ptr + row * qr_stride + offs, mask=mask, other=neg).to(
                tl.float32
            )
            / temp
        )
        qp = tl.load(qp_ptr + row * qp_stride + offs, mask=mask, other=neg).to(
            tl.float32
        )
        m_p, s_p = _online_lse(m_p, s_p, p)
        m_r, s_r = _online_lse(m_r, s_r, qr)
        m_q, s_q = _online_lse(m_q, s_q, qp)
        m_1, s_1 = _online_lse(m_1, s_1, qp * inv_t1)
        m_2, s_2 = _online_lse(m_2, s_2, qp * inv_t2)
    lse_p = _finish_lse(m_p, s_p)
    lse_r = _finish_lse(m_r, s_r)
    lse_q = _finish_lse(m_q, s_q)
    lse_1 = _finish_lse(m_1, s_1)
    lse_2 = _finish_lse(m_2, s_2)

    a_r = tl.zeros((), tl.float32)
    a_q = a_r
    a_1 = a_r
    a_2 = a_r
    out_r = a_r
    out_q = a_r
    for start in range(0, vocab_size, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < vocab_size
        p = tl.load(p_ptr + row * p_stride + offs, mask=mask, other=neg).to(tl.float32)
        qr = (
            tl.load(qr_ptr + row * qr_stride + offs, mask=mask, other=neg).to(
                tl.float32
            )
            / temp
        )
        qp = tl.load(qp_ptr + row * qp_stride + offs, mask=mask, other=neg).to(
            tl.float32
        )
        pp = tl.exp(p - lse_p)
        pr = tl.exp(qr - lse_r)
        pq = tl.exp(qp - lse_q)
        p1 = tl.exp(qp * inv_t1 - lse_1)
        p2 = tl.exp(qp * inv_t2 - lse_2)
        a_r += tl.sum(tl.minimum(pp, pr), axis=0)
        a_q += tl.sum(tl.minimum(pp, pq), axis=0)
        a_1 += tl.sum(tl.minimum(pp, p1), axis=0)
        a_2 += tl.sum(tl.minimum(pp, p2), axis=0)
        outside = p == neg
        out_r += tl.sum(tl.where(outside, pr, 0.0), axis=0)
        out_q += tl.sum(tl.where(outside, pq, 0.0), axis=0)
    p_tok = tl.load(p_ptr + row * p_stride + tok).to(tl.float32)
    tok_out = tl.where(p_tok == neg, 1.0, 0.0)
    stats = tl.where(cols == 0, 1.0, 0.0)
    stats = tl.where(cols == 1, a_r, stats)
    stats = tl.where(cols == 2, a_q, stats)
    stats = tl.where(cols == 3, a_1, stats)
    stats = tl.where(cols == 4, a_2, stats)
    stats = tl.where(cols == 5, out_r, stats)
    stats = tl.where(cols == 6, out_q, stats)
    stats = tl.where(cols == 7, tok_out, stats)
    tl.store(out_ptr + row * NUM_STATS + cols, stats)


class _DiagState:
    def __init__(self, num_steps: int, device: torch.device):
        self.num_steps = num_steps
        self.calls = 0
        self.observed = 0
        self.last_log = time.monotonic()
        # [k, NUM_STATS] per draft position.
        self.pos_acc = torch.zeros(
            num_steps, _NUM_STATS, dtype=torch.float64, device=device
        )
        # [requests, drafts, measured accepted, chain E[accepted] x NUM_Q]
        self.chain_acc = torch.zeros(3 + _NUM_Q, dtype=torch.float64, device=device)


_state: _DiagState | None = None


def _fmt(values: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.4f}" for v in values) + "]"


def _log(st: _DiagState) -> None:
    pos = st.pos_acc.cpu().numpy()
    chain = st.chain_acc.cpu().numpy()
    n = np.maximum(pos[:, 0], 1.0)
    reqs, drafts, measured = chain[0], chain[1], chain[2]
    est = chain[3:]
    denom = max(reqs, 1.0)
    al = 1.0 + est / denom  # tokens per verify step, chain estimate
    al_meas = 1.0 + measured / denom
    # Calibrated prediction: measured AL scaled by the chain ratio to q_raw.
    scale = est / max(est[0], 1e-9)
    pred = 1.0 + (measured / denom) * scale
    gain = (pred / max(al_meas, 1e-9) - 1.0) * 100.0
    logger.info(
        "H53 diag: observed_steps=%d request_steps=%d drafts=%d rows/pos=%s | "
        "alpha_raw=%s alpha_proc=%s alpha_t0.9=%s alpha_t0.8=%s | "
        "out_tok=%s out_mass_raw=%s out_mass_proc=%s | "
        "AL measured=%.4f chain_raw=%.4f chain_proc=%.4f chain_t0.9=%.4f "
        "chain_t0.8=%.4f | pred_AL proc=%.4f t0.9=%.4f t0.8=%.4f | "
        "pred_gain%% proc=%+.2f t0.9=%+.2f t0.8=%+.2f",
        st.observed,
        int(reqs),
        int(drafts),
        "[" + ",".join(str(int(v)) for v in pos[:, 0]) + "]",
        _fmt(pos[:, 1] / n),
        _fmt(pos[:, 2] / n),
        _fmt(pos[:, 3] / n),
        _fmt(pos[:, 4] / n),
        _fmt(pos[:, 7] / n),
        _fmt(pos[:, 5] / n),
        _fmt(pos[:, 6] / n),
        al_meas,
        al[0],
        al[1],
        al[2],
        al[3],
        pred[1],
        pred[2],
        pred[3],
        gain[1],
        gain[2],
        gain[3],
    )


def diag_observe(
    sampler,
    # [n_chunk, V] processed target logits of the chunk
    processed_logits: torch.Tensor,
    # [max_num_reqs, k, V] raw cached draft logits (pre-temperature)
    draft_logits: torch.Tensor,
    # [n_chunk] input token of each logits row (row r + 1 holds draft r)
    draft_sampled: torch.Tensor,
    pos: torch.Tensor,
    # [chunk_reqs + 1] host, starts at 0
    cu_num_logits_np: np.ndarray,
    idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    # [chunk_reqs] accepted + 1 (0 for chunked prefills)
    num_sampled: torch.Tensor,
) -> None:
    """Log-only: compare acceptance under q_raw and q_proc (module docstring)."""
    global _state
    num_steps = draft_logits.shape[1]
    if _state is None:
        _state = _DiagState(num_steps, processed_logits.device)
        logger.info(
            "H53 diag on (VLLM_H53_DIAG=1): every %d verify steps, first %d "
            "requests, log every %.0f s; outputs unchanged",
            DIAG_EVERY,
            DIAG_MAX_REQS,
            DIAG_LOG_S,
        )
    st = _state
    st.calls += 1
    if st.calls % DIAG_EVERY != 0:
        return

    num_reqs = min(idx_mapping_np.shape[0], DIAG_MAX_REQS)
    states = sampler.sampling_states
    if (
        num_reqs == 0
        or not (states.use_split_row_top_k and states.use_fused_prep)
        or (_REQUIRE_CUDA and not processed_logits.is_cuda)
        or sampler.fused_spec_sampling_params(idx_mapping_np[:num_reqs]) is None
    ):
        return
    cu = cu_num_logits_np[: num_reqs + 1].astype(np.int64)
    counts = np.diff(cu)
    num_drafts = counts - 1
    num_rows = int(cu[-1])
    if num_rows == 0 or int(num_drafts.max()) <= 0:
        return
    vocab_size = min(processed_logits.shape[1], draft_logits.shape[2])

    # Host layout of the observed rows (one small upload).
    req_of_row = np.repeat(np.arange(num_reqs), counts)
    local = np.arange(num_rows) - cu[:-1][req_of_row]
    is_draft_row = local < num_drafts[req_of_row]
    has_draft = num_drafts > 0
    host = np.concatenate(
        [req_of_row, np.minimum(local, num_steps - 1), is_draft_row, has_draft]
    ).astype(np.int64)
    dev = torch.from_numpy(host).to(processed_logits.device, non_blocking=True)
    req_dev = dev[:num_rows]
    local_dev = dev[num_rows : 2 * num_rows]
    is_draft_dev = dev[2 * num_rows : 3 * num_rows].bool()
    has_draft_dev = dev[3 * num_rows :].to(torch.float64)

    exp_idx = expanded_idx_mapping[:num_rows]
    steps = expanded_local_pos[:num_rows].long().clamp(max=num_steps - 1)
    raw = draft_logits[exp_idx.long(), steps][:, :vocab_size]
    q_proc = sampler.apply_sampling_params(
        raw,
        exp_idx,
        idx_mapping[:num_reqs],
        idx_mapping_np[:num_reqs],
        pos[:num_rows],
        draft_sampled[:num_rows],
        expanded_local_pos[:num_rows],
    )
    draft_tok = torch.full_like(draft_sampled[:num_rows], -1)
    draft_tok[: num_rows - 1] = draft_sampled[1:num_rows]
    draft_tok = torch.where(is_draft_dev, draft_tok, -1)

    stats = torch.empty(
        num_rows, _NUM_STATS, dtype=torch.float32, device=processed_logits.device
    )
    p = processed_logits[:num_rows, :vocab_size]
    _diag_row_kernel[(num_rows,)](
        stats,
        p,
        p.stride(0),
        raw,
        raw.stride(0),
        q_proc,
        q_proc.stride(0),
        draft_tok,
        exp_idx,
        sampler.sampling_states.temperature.gpu,
        vocab_size,
        1.0 / _SHARPEN[0],
        1.0 / _SHARPEN[1],
        NUM_STATS=_NUM_STATS,
        BLOCK=4096,
        num_warps=8,
    )
    stats64 = stats.to(torch.float64)
    st.pos_acc.index_add_(0, local_dev, stats64)

    # Chain estimate of E[accepted] per request: sum_j prod_{i<=j} alpha_i.
    alpha = torch.zeros(
        num_reqs, num_steps, _NUM_Q, dtype=torch.float64, device=stats.device
    )
    alpha.index_put_((req_dev, local_dev), stats64[:, 1 : 1 + _NUM_Q], accumulate=True)
    expected = torch.cumprod(alpha, dim=1).sum(dim=1)  # [num_reqs, NUM_Q]
    accepted = (num_sampled[:num_reqs].to(torch.float64) - 1.0).clamp(min=0.0)
    chain = torch.cat(
        [
            has_draft_dev.sum().view(1),
            torch.zeros(1, dtype=torch.float64, device=stats.device),
            (accepted * has_draft_dev).sum().view(1),
            (expected * has_draft_dev[:, None]).sum(dim=0),
        ]
    )
    chain[1] = float(num_drafts[has_draft].sum())
    st.chain_acc += chain
    st.observed += 1

    now = time.monotonic()
    if now - st.last_log >= DIAG_LOG_S:
        st.last_log = now
        _log(st)


# ---------------------------------------------------------------------------
# VLLM_H53_DRAFT_PARITY: processed, sparse draft distribution + sparse verify.
# ---------------------------------------------------------------------------

# Globals read inside @triton.jit must be constexpr.
_INT64_MIN = tl.constexpr(-9223372036854775808) if HAS_TRITON else 0
_INT32_MAX = tl.constexpr(2147483647) if HAS_TRITON else 0
_INT32_MIN = tl.constexpr(-2147483648) if HAS_TRITON else 0


@triton.jit
def _parity_process(
    x,
    offs,
    mask,
    row,
    req,
    # [rows, num_steps] int64 drafts 1..k of this proposal (batch-indexed)
    draft_tokens_ptr,
    draft_tokens_stride,
    # [] int64 current draft step j: drafts 1..j are history
    draft_step_ptr,
    # only its element type: the draft-logits cache dtype
    round_ptr,
    rep_ptr,
    freq_ptr,
    pres_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
):
    """Penalties as the target's verify row j applies them, then rounding to the
    cache dtype (so the dense fallback holds the sampled values exactly)."""
    rep = tl.load(rep_ptr + req)
    freq = tl.load(freq_ptr + req)
    pres = tl.load(pres_ptr + req)
    use_rep = rep != 1.0
    if use_rep or freq != 0.0 or pres != 0.0:
        counts = tl.load(
            output_bin_counts_ptr + req * output_bin_counts_stride + offs,
            mask=mask,
            other=0,
        )
        step = tl.load(draft_step_ptr)
        for s in range(step):
            tok = tl.load(draft_tokens_ptr + row * draft_tokens_stride + s)
            counts += (offs == tok).to(tl.int32)
        prompt_mask = offs < 0
        if use_rep:
            word = tl.load(
                prompt_bin_mask_ptr + req * prompt_bin_mask_stride + offs // 32,
                mask=mask,
                other=0,
            )
            prompt_mask = ((word >> (offs % 32).to(tl.int32)) & 1).to(tl.int1)
        x = _penalize(x, counts, prompt_mask, rep, freq, pres, use_rep)
    return x.to(round_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _parity_submax_kernel(
    # [max_num_reqs, num_sub] fp32
    submax_ptr,
    submax_stride,
    # [rows, V] raw draft logits
    logits_ptr,
    logits_stride,
    # [rows] request state index, -1 for CUDA-graph padding
    idx_mapping_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    draft_step_ptr,
    round_ptr,
    rep_ptr,
    freq_ptr,
    pres_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    SUB_SIZE: tl.constexpr,
    GLUE_PDL: tl.constexpr = False,
):
    if GLUE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    req = tl.load(idx_mapping_ptr + row).to(tl.int64)
    if req < 0:
        return
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    x = tl.load(
        logits_ptr + row * logits_stride + block, mask=mask, other=float("-inf")
    ).to(tl.float32)
    x = _parity_process(
        x,
        block,
        mask,
        row,
        req,
        draft_tokens_ptr,
        draft_tokens_stride,
        draft_step_ptr,
        round_ptr,
        rep_ptr,
        freq_ptr,
        pres_ptr,
        prompt_bin_mask_ptr,
        prompt_bin_mask_stride,
        output_bin_counts_ptr,
        output_bin_counts_stride,
    )
    x = tl.where(mask, x, float("-inf"))
    NUM_SUB: tl.constexpr = BLOCK_SIZE // SUB_SIZE
    sub_offs = block_idx * NUM_SUB + tl.arange(0, NUM_SUB)
    tl.store(
        submax_ptr + row * submax_stride + sub_offs,
        tl.max(tl.reshape(x, (NUM_SUB, SUB_SIZE)), axis=1),
    )


@triton.jit
def _parity_select_kernel(
    # [rows] int64
    sampled_ptr,
    # [max_num_reqs, num_steps, KP] fp32 / int32, [max_num_reqs, num_steps]
    q_val_ptr,
    q_idx_ptr,
    q_num_ptr,
    q_lse_ptr,
    num_steps,
    submax_ptr,
    submax_stride,
    num_sub,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    # [rows] sampling position key of each row
    pos_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    draft_step_ptr,
    round_ptr,
    # [max_num_reqs] persistent request state
    temperature_ptr,
    seeds_ptr,
    top_k_ptr,
    top_p_ptr,
    rep_ptr,
    freq_ptr,
    pres_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    KP: tl.constexpr,
    SUB_SIZE: tl.constexpr,
    PADDED_NUM_SUB: tl.constexpr,
    GROUP: tl.constexpr,
    USE_FP64: tl.constexpr,
    GLUE_PDL: tl.constexpr = False,
):
    if GLUE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(idx_mapping_ptr + row).to(tl.int64)
    if req < 0:
        tl.store(sampled_ptr + row, 0)
        return
    top_k = tl.load(top_k_ptr + req)
    top_p = tl.load(top_p_ptr + req)
    temp = tl.load(temperature_ptr + req).to(tl.float32)

    # 1. The KP sub-blocks with the largest (max, -index) keys (as in
    # fused_rejection._select_survivors_kernel): they hold the row's KP largest
    # processed logits.
    if GROUP > 0:
        NUM_GROUPS: tl.constexpr = PADDED_NUM_SUB // GROUP
        grp = tl.arange(0, NUM_GROUPS)
        sub = grp[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
        submax = tl.load(
            submax_ptr + row * submax_stride + sub,
            mask=sub < num_sub,
            other=float("-inf"),
        )
        top_grp = tl.topk(_pack_keys(tl.max(submax, axis=1), grp, NUM_GROUPS), KP)
        _, grp_idx = _unpack_keys(top_grp, NUM_GROUPS)
        sub = grp_idx[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
        submax = tl.load(
            submax_ptr + row * submax_stride + sub,
            mask=sub < num_sub,
            other=float("-inf"),
        )
        sub_keys = tl.reshape(_pack_keys(submax, sub, PADDED_NUM_SUB), (KP * GROUP,))
    else:
        sub = tl.arange(0, PADDED_NUM_SUB)
        submax = tl.load(
            submax_ptr + row * submax_stride + sub,
            mask=sub < num_sub,
            other=float("-inf"),
        )
        sub_keys = _pack_keys(submax, sub, PADDED_NUM_SUB)
    top_sub = tl.topk(sub_keys, KP)
    _, sub_idx = _unpack_keys(top_sub, PADDED_NUM_SUB)

    # 2. Their processed (penalized, rounded) pre-temperature logits.
    offs = sub_idx[:, None] * SUB_SIZE + tl.arange(0, SUB_SIZE)[None, :]
    mask = offs < vocab_size
    v = tl.load(
        logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")
    ).to(tl.float32)
    v = _parity_process(
        v,
        offs,
        mask,
        row,
        req,
        draft_tokens_ptr,
        draft_tokens_stride,
        draft_step_ptr,
        round_ptr,
        rep_ptr,
        freq_ptr,
        pres_ptr,
        prompt_bin_mask_ptr,
        prompt_bin_mask_stride,
        output_bin_counts_ptr,
        output_bin_counts_stride,
    )

    # 3. Top-k by (value desc, token id asc), at most KP; temperature; top-p on
    # the renormalized survivors (spec_topk_topp._select_threshold's rule).
    sel = mask & (v > float("-inf"))
    keys = tl.where(sel, _pack_keys(v, offs, vocab_size), _INT64_MIN)
    top = tl.topk(tl.reshape(keys, (KP * SUB_SIZE,)), KP)
    pos_k = tl.arange(0, KP)
    num_valid = tl.sum(((pos_k < top_k) & (top != _INT64_MIN)).to(tl.int32))
    valid = pos_k < num_valid
    value, token = _unpack_keys(top, vocab_size)
    x = value
    if temp != 0.0 and temp != 1.0:
        x = value / temp
    max_x = tl.max(tl.where(valid, x, float("-inf")), axis=0)
    e = tl.where(valid, tl.exp(x - max_x), 0.0)
    cum = tl.cumsum(e / tl.sum(e, axis=0), axis=0)
    below = valid & (pos_k < num_valid - 1) & (cum < top_p)
    num_keep = tl.where(
        top_p < 1.0, tl.minimum(tl.sum(below.to(tl.int32)) + 1, num_valid), num_valid
    )
    keep = pos_k < num_keep
    max_x = tl.max(tl.where(keep, x, float("-inf")), axis=0)
    sumexp = tl.sum(tl.where(keep, tl.exp(x - max_x), 0.0), axis=0)
    lse = tl.where(num_keep > 0, max_x + tl.log(sumexp), float("-inf"))

    # 4. Draw: Gumbel-max over the survivors with the stock draft noise (keyed
    # by token id at the draft's position), or the first survivor at temp 0.
    if temp == 0.0:
        sampled = tl.sum(tl.where(pos_k == 0, token, 0))
    else:
        seed = tl.load(seeds_ptr + req)
        pos = tl.load(pos_ptr + row)
        noised = gumbel_noised_logits(
            x,
            token,
            keep,
            seed,
            pos,
            temp,
            IS_DRAFTING=True,
            USE_FP64=USE_FP64,
            APPLY_TEMPERATURE=False,
        )
        best = tl.max(noised, axis=0)
        sampled = tl.min(tl.where(keep & (noised == best), token, _INT32_MAX))
        sampled = tl.where(best > float("-inf"), sampled, 0)
    tl.store(sampled_ptr + row, sampled.to(tl.int64))

    # 5. Cache q for the verifier.
    cell = req * num_steps + tl.load(draft_step_ptr)
    tl.store(q_val_ptr + cell * KP + pos_k, tl.where(keep, value, float("-inf")))
    tl.store(q_idx_ptr + cell * KP + pos_k, tl.where(keep, token, -1).to(tl.int32))
    tl.store(q_num_ptr + cell, num_keep)
    tl.store(q_lse_ptr + cell, lse)


@triton.jit
def _h97_pack32(x, idx, n):
    """_pack_keys's order in 32 bits, for bf16-exact values (low 16 fp32 bits
    zero, or -inf) and n <= 65536: high half the value's order key, low half
    n - 1 - idx. Same order as _pack_keys(x, idx, n) on such values."""
    x = x + 0.0
    bits = x.to(tl.int32, bitcast=True)
    ikey = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return ((ikey >> 16) << 16) | (n - 1 - idx).to(tl.int32)


@triton.jit
def _h97_process(
    x,
    offs,
    mask,
    req,
    counts,
    hist,
    rep,
    freq,
    pres,
    round_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    MAX_HIST: tl.constexpr,
):
    """_parity_process with its operands loaded by the caller (no load after a
    branch or inside the history loop). Same integer counts, same arithmetic."""
    use_rep = rep != 1.0
    if use_rep or freq != 0.0 or pres != 0.0:
        har = tl.arange(0, MAX_HIST)
        for s in tl.static_range(MAX_HIST):
            # hist holds -1 past the current step: no token matches it.
            tok = tl.max(tl.where(har == s, hist, -1), axis=0)
            counts += (offs == tok).to(tl.int32)
        prompt_mask = offs < 0
        if use_rep:
            word = tl.load(
                prompt_bin_mask_ptr + req * prompt_bin_mask_stride + offs // 32,
                mask=mask,
                other=0,
            )
            prompt_mask = ((word >> (offs % 32).to(tl.int32)) & 1).to(tl.int1)
        x = _penalize(x, counts, prompt_mask, rep, freq, pres, use_rep)
    return x.to(round_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _h97_submax_kernel(
    submax_ptr,
    submax_stride,
    # [max_num_reqs, padded groups] fp32 maxima of GROUP sub-blocks
    gmax_ptr,
    gmax_stride,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    draft_step_ptr,
    round_ptr,
    rep_ptr,
    freq_ptr,
    pres_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    SUB_SIZE: tl.constexpr,
    GROUP: tl.constexpr,
    GLUE_PDL: tl.constexpr = False,
):
    """_parity_submax_kernel, plus the maxima of each GROUP sub-blocks for the
    select's first stage."""
    if GLUE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    req = tl.load(idx_mapping_ptr + row).to(tl.int64)
    if req < 0:
        return
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    x = tl.load(
        logits_ptr + row * logits_stride + block, mask=mask, other=float("-inf")
    ).to(tl.float32)
    x = _parity_process(
        x,
        block,
        mask,
        row,
        req,
        draft_tokens_ptr,
        draft_tokens_stride,
        draft_step_ptr,
        round_ptr,
        rep_ptr,
        freq_ptr,
        pres_ptr,
        prompt_bin_mask_ptr,
        prompt_bin_mask_stride,
        output_bin_counts_ptr,
        output_bin_counts_stride,
    )
    x = tl.where(mask, x, float("-inf"))
    NUM_SUB: tl.constexpr = BLOCK_SIZE // SUB_SIZE
    sm = tl.max(tl.reshape(x, (NUM_SUB, SUB_SIZE)), axis=1)
    tl.store(
        submax_ptr + row * submax_stride + block_idx * NUM_SUB + tl.arange(0, NUM_SUB),
        sm,
    )
    if GROUP > 0:
        NUM_G: tl.constexpr = NUM_SUB // GROUP
        tl.store(
            gmax_ptr + row * gmax_stride + block_idx * NUM_G + tl.arange(0, NUM_G),
            tl.max(tl.reshape(sm, (NUM_G, GROUP)), axis=1),
        )


@triton.jit
def _h97_select_kernel(
    sampled_ptr,
    q_val_ptr,
    q_idx_ptr,
    q_num_ptr,
    q_lse_ptr,
    num_steps,
    submax_ptr,
    submax_stride,
    num_sub,
    gmax_ptr,
    gmax_stride,
    num_groups,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    pos_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    draft_step_ptr,
    round_ptr,
    temperature_ptr,
    seeds_ptr,
    top_k_ptr,
    top_p_ptr,
    rep_ptr,
    freq_ptr,
    pres_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    KP: tl.constexpr,
    SUB_SIZE: tl.constexpr,
    PADDED_NUM_SUB: tl.constexpr,
    GROUP: tl.constexpr,
    USE_FP64: tl.constexpr,
    MAX_HIST: tl.constexpr,
    KEY32A: tl.constexpr,
    KEY32B: tl.constexpr,
    GLUE_PDL: tl.constexpr = False,
):
    """_parity_select_kernel's draw and q cache, bit for bit, restructured:

    * every scalar and the draft history are loaded up front;
    * the group maxima come from _h97_submax_kernel (no 8192-wide reload);
    * KEY32A: both sub-block top-k's sort 32-bit keys (same order: the maxima
      are bf16-exact);
    * KEY32B: so does the final top-k over the 1024 candidates, with the rows in
      sub-block order so that a 10-bit local index orders like the token id;
      the token is looked up after the sort, and `top` rebuilt as the stock key.
    The survivors, their order and everything after the top-k are unchanged.
    """
    if GLUE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(idx_mapping_ptr + row).to(tl.int64)
    if req < 0:
        tl.store(sampled_ptr + row, 0)
        return
    top_k = tl.load(top_k_ptr + req)
    top_p = tl.load(top_p_ptr + req)
    temp = tl.load(temperature_ptr + req).to(tl.float32)
    seed = tl.load(seeds_ptr + req)
    pos = tl.load(pos_ptr + row)
    step = tl.load(draft_step_ptr)
    rep = tl.load(rep_ptr + req)
    freq = tl.load(freq_ptr + req)
    pres = tl.load(pres_ptr + req)
    har = tl.arange(0, MAX_HIST)
    hist = tl.load(
        draft_tokens_ptr + row * draft_tokens_stride + har, mask=har < step, other=-1
    )

    # 1. The KP sub-blocks with the largest (max, -index) keys.
    if GROUP > 0:
        NUM_GROUPS: tl.constexpr = PADDED_NUM_SUB // GROUP
        grp = tl.arange(0, NUM_GROUPS)
        gm = tl.load(
            gmax_ptr + row * gmax_stride + grp,
            mask=grp < num_groups,
            other=float("-inf"),
        )
        if KEY32A:
            top_grp = tl.topk(_h97_pack32(gm, grp, NUM_GROUPS), KP)
            grp_idx = NUM_GROUPS - 1 - (top_grp & 0xFFFF)
        else:
            top_grp = tl.topk(_pack_keys(gm, grp, NUM_GROUPS), KP)
            _, grp_idx = _unpack_keys(top_grp, NUM_GROUPS)
        sub = grp_idx[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
        submax = tl.load(
            submax_ptr + row * submax_stride + sub,
            mask=sub < num_sub,
            other=float("-inf"),
        )
        if KEY32A:
            sub_keys = tl.reshape(
                _h97_pack32(submax, sub, PADDED_NUM_SUB), (KP * GROUP,)
            )
        else:
            sub_keys = tl.reshape(
                _pack_keys(submax, sub, PADDED_NUM_SUB), (KP * GROUP,)
            )
    else:
        sub = tl.arange(0, PADDED_NUM_SUB)
        submax = tl.load(
            submax_ptr + row * submax_stride + sub,
            mask=sub < num_sub,
            other=float("-inf"),
        )
        if KEY32A:
            sub_keys = _h97_pack32(submax, sub, PADDED_NUM_SUB)
        else:
            sub_keys = _pack_keys(submax, sub, PADDED_NUM_SUB)
    top_sub = tl.topk(sub_keys, KP)
    if KEY32A:
        sub_idx = (PADDED_NUM_SUB - 1 - (top_sub & 0xFFFF)).to(tl.int64)
    else:
        _, sub_idx = _unpack_keys(top_sub, PADDED_NUM_SUB)
    if KEY32B:
        # Rows in sub-block order: local index r * SUB_SIZE + c orders like
        # the token id.
        sub_idx = tl.sort(sub_idx)

    # 2. Their processed (penalized, rounded) pre-temperature logits.
    offs = sub_idx[:, None] * SUB_SIZE + tl.arange(0, SUB_SIZE)[None, :]
    mask = offs < vocab_size
    counts = tl.load(
        output_bin_counts_ptr + req * output_bin_counts_stride + offs,
        mask=mask,
        other=0,
    )
    v = tl.load(
        logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")
    ).to(tl.float32)
    v = _h97_process(
        v,
        offs,
        mask,
        req,
        counts,
        hist,
        rep,
        freq,
        pres,
        round_ptr,
        prompt_bin_mask_ptr,
        prompt_bin_mask_stride,
        MAX_HIST,
    )

    # 3. Top-k by (value desc, token id asc), at most KP; temperature; top-p on
    # the renormalized survivors (from here on as _parity_select_kernel).
    sel = mask & (v > float("-inf"))
    if KEY32B:
        NC: tl.constexpr = KP * SUB_SIZE
        local = (
            tl.arange(0, KP)[:, None] * SUB_SIZE + tl.arange(0, SUB_SIZE)[None, :]
        )
        keys32 = tl.where(sel, _h97_pack32(v, local, NC), _INT32_MIN)
        top32 = tl.topk(tl.reshape(keys32, (NC,)), KP)
        lsel = NC - 1 - (top32 & 0xFFFF)
        token32 = tl.gather(sub_idx, lsel // SUB_SIZE, 0) * SUB_SIZE + (
            lsel % SUB_SIZE
        )
        hv = top32 >> 16
        ikey = (hv << 16) | tl.where(hv < 0, 0xFFFF, 0)
        vbits = tl.where(ikey < 0, ikey ^ 0x7FFFFFFF, ikey)
        value32 = vbits.to(tl.float32, bitcast=True)
        top = tl.where(
            top32 != _INT32_MIN,
            _pack_keys(value32, token32, vocab_size),
            _INT64_MIN,
        )
    else:
        keys = tl.where(sel, _pack_keys(v, offs, vocab_size), _INT64_MIN)
        top = tl.topk(tl.reshape(keys, (KP * SUB_SIZE,)), KP)
    pos_k = tl.arange(0, KP)
    num_valid = tl.sum(((pos_k < top_k) & (top != _INT64_MIN)).to(tl.int32))
    valid = pos_k < num_valid
    value, token = _unpack_keys(top, vocab_size)
    x = value
    if temp != 0.0 and temp != 1.0:
        x = value / temp
    max_x = tl.max(tl.where(valid, x, float("-inf")), axis=0)
    e = tl.where(valid, tl.exp(x - max_x), 0.0)
    cum = tl.cumsum(e / tl.sum(e, axis=0), axis=0)
    below = valid & (pos_k < num_valid - 1) & (cum < top_p)
    num_keep = tl.where(
        top_p < 1.0, tl.minimum(tl.sum(below.to(tl.int32)) + 1, num_valid), num_valid
    )
    keep = pos_k < num_keep
    max_x = tl.max(tl.where(keep, x, float("-inf")), axis=0)
    sumexp = tl.sum(tl.where(keep, tl.exp(x - max_x), 0.0), axis=0)
    lse = tl.where(num_keep > 0, max_x + tl.log(sumexp), float("-inf"))

    # 4. Draw: Gumbel-max over the survivors with the stock draft noise.
    if temp == 0.0:
        sampled = tl.sum(tl.where(pos_k == 0, token, 0))
    else:
        noised = gumbel_noised_logits(
            x,
            token,
            keep,
            seed,
            pos,
            temp,
            IS_DRAFTING=True,
            USE_FP64=USE_FP64,
            APPLY_TEMPERATURE=False,
        )
        best = tl.max(noised, axis=0)
        sampled = tl.min(tl.where(keep & (noised == best), token, _INT32_MAX))
        sampled = tl.where(best > float("-inf"), sampled, 0)
    tl.store(sampled_ptr + row, sampled.to(tl.int64))

    # 5. Cache q for the verifier.
    cell = req * num_steps + step
    tl.store(q_val_ptr + cell * KP + pos_k, tl.where(keep, value, float("-inf")))
    tl.store(q_idx_ptr + cell * KP + pos_k, tl.where(keep, token, -1).to(tl.int32))
    tl.store(q_num_ptr + cell, num_keep)
    tl.store(q_lse_ptr + cell, lse)


@triton.jit
def _load_q_row(
    q_val_ptr,
    q_idx_ptr,
    q_num_ptr,
    q_lse_ptr,
    cell,
    temp,
    KPQ: tl.constexpr,
):
    """Survivor ids, post-temperature logits (as the drafter sampled them) and
    logsumexp of q at one (request, step) cell."""
    kq = tl.arange(0, KPQ)
    live = kq < tl.load(q_num_ptr + cell)
    q_idx = tl.load(q_idx_ptr + cell * KPQ + kq, mask=live, other=-1)
    q_x = tl.load(q_val_ptr + cell * KPQ + kq, mask=live, other=float("-inf"))
    if temp != 0.0 and temp != 1.0:
        q_x = q_x / temp
    return q_idx, q_x, tl.load(q_lse_ptr + cell)


@triton.jit
def _parity_rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    num_rejected_ptr,
    # [num_logits, KP] / [num_logits] target survivors
    surv_val_ptr,
    surv_idx_ptr,
    num_surv_ptr,
    lse_ptr,
    # [max_num_reqs, num_steps, KPQ] / [max_num_reqs, num_steps] draft q
    q_val_ptr,
    q_idx_ptr,
    q_num_ptr,
    q_lse_ptr,
    num_steps,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    seq_lens_ptr,
    # [max_num_reqs]
    prefill_len_ptr,
    temperature_ptr,
    seeds_ptr,
    # [num_tokens]
    input_ids_ptr,
    positions_ptr,
    # [num_logits]
    logits_indices_ptr,
    KP: tl.constexpr,
    KPQ: tl.constexpr,
    USE_FP64: tl.constexpr,
    GLUE_PDL: tl.constexpr = False,
):
    """fused_rejection._compact_rejection_kernel with a sparse draft q."""
    if GLUE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end - start - 1
    seed = tl.load(seeds_ptr + req_state_idx)
    temp = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0
    k = tl.arange(0, KP)

    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start + i
        draft_sampled = tl.load(
            input_ids_ptr + tl.load(logits_indices_ptr + logit_idx + 1)
        ).to(tl.int64)
        is_valid_draft = draft_sampled >= 0
        draft_sampled = tl.maximum(0, draft_sampled)
        if not is_greedy:
            verifying &= is_valid_draft
        if verifying:
            pos = tl.load(positions_ptr + tl.load(logits_indices_ptr + logit_idx))
            u = tl_rand32(seed, pos, includes_zero=False)
            num_surv = tl.load(num_surv_ptr + logit_idx)
            if is_greedy:
                target_argmax = tl.where(
                    num_surv > 0, tl.load(surv_idx_ptr + logit_idx * KP), 0
                ).to(tl.int64)
                accepted = (target_argmax == draft_sampled) & is_valid_draft
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    draft_sampled if accepted else target_argmax,
                )
            else:
                surv = k < num_surv
                token = tl.load(surv_idx_ptr + logit_idx * KP + k, mask=surv, other=-1)
                value = tl.load(surv_val_ptr + logit_idx * KP + k)
                target_logit = tl.max(
                    tl.where(token == draft_sampled, value, float("-inf")), axis=0
                )
                target_logprob = target_logit - tl.load(lse_ptr + logit_idx)
                q_idx, q_x, q_lse = _load_q_row(
                    q_val_ptr,
                    q_idx_ptr,
                    q_num_ptr,
                    q_lse_ptr,
                    req_state_idx * num_steps + i,
                    temp,
                    KPQ,
                )
                q_logit = tl.max(
                    tl.where(q_idx == draft_sampled, q_x, float("-inf")), axis=0
                )
                # A draft not drawn from this q (never, for drafter output) is
                # tested as a one-hot proposal.
                draft_logprob = tl.where(
                    q_logit > float("-inf"), q_logit - q_lse, 0.0
                )
                # Probability-ratio test: p(x) > u * q(x).
                accepted = target_logprob > tl.log(u) + draft_logprob
                verifying = accepted
                accepted_length += accepted
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)

    is_chunked_prefilling = tl.load(seq_lens_ptr + req_idx) < tl.load(
        prefill_len_ptr + req_state_idx
    )
    final_num_sampled = tl.where(is_chunked_prefilling, 0, accepted_length + 1)
    tl.store(num_sampled_ptr + req_idx, final_num_sampled.to(tl.int32))
    num_rejected = end - start - final_num_sampled
    tl.store(
        num_rejected_ptr + req_idx,
        tl.where(is_chunked_prefilling, 0, num_rejected).to(tl.int32),
    )

    resample_token_idx = start + accepted_length
    is_bonus = resample_token_idx == end - 1
    if temp == 0.0 and not is_bonus:
        return

    # Resample from norm(max(0, p - q)) over the target survivors (p = 0
    # elsewhere), or from p for the bonus token, with _resample_kernel's
    # per-token noise and lowest-id tie rule.
    rejected_draft_token = tl.load(
        input_ids_ptr
        + tl.load(
            logits_indices_ptr + resample_token_idx + 1, mask=not is_bonus, other=0
        ),
        mask=not is_bonus,
        other=0,
    )
    is_valid_rejected_draft = rejected_draft_token >= 0
    num_surv = tl.load(num_surv_ptr + resample_token_idx)
    residual = k < num_surv
    token = tl.load(surv_idx_ptr + resample_token_idx * KP + k, mask=residual, other=0)
    value = tl.load(surv_val_ptr + resample_token_idx * KP + k)
    value = tl.where(residual, value, float("-inf"))
    if not is_bonus and is_valid_rejected_draft:
        q_idx, q_x, q_lse = _load_q_row(
            q_val_ptr,
            q_idx_ptr,
            q_num_ptr,
            q_lse_ptr,
            req_state_idx * num_steps + accepted_length,
            temp,
            KPQ,
        )
        q_rejected = tl.max(
            tl.where(q_idx == rejected_draft_token, q_x, float("-inf")), axis=0
        )
        if q_rejected > float("-inf"):
            log_p = value - tl.load(lse_ptr + resample_token_idx)
            q_at = tl.max(
                tl.where(token[:, None] == q_idx[None, :], q_x[None, :], float("-inf")),
                axis=1,
            )
            ratio = tl.exp((q_at - q_lse) - log_p)
            res = tl.where(
                residual & (ratio < 1.0),
                log_p + tldevice.log1p(-ratio),
                float("-inf"),
            )
            # p == q to rounding leaves no residual mass: keep sampling from p.
            if tl.max(res, axis=0) > float("-inf"):
                value = res
                residual = res > float("-inf")
        else:
            residual &= token != rejected_draft_token
            value = tl.where(residual, value, float("-inf"))
    pos = tl.load(positions_ptr + tl.load(logits_indices_ptr + resample_token_idx))
    noised = gumbel_noised_logits(
        value,
        token,
        residual,
        seed,
        pos,
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    best = tl.max(noised, axis=0)
    resampled = tl.min(tl.where(residual & (noised == best), token, _INT32_MAX))
    resampled = tl.where(best > float("-inf"), resampled, 0)
    tl.store(
        sampled_ptr + req_idx * sampled_stride + accepted_length,
        resampled.to(tl.int64),
    )


@triton.jit
def _densify_kernel(
    # [max_num_reqs, num_steps, V] dense draft-logits cache (pre-temperature)
    dense_ptr,
    dense_stride_0,
    dense_stride_1,
    q_val_ptr,
    q_idx_ptr,
    q_num_ptr,
    num_steps,
    # [num_logits]
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    vocab_size,
    KPQ: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    step = tl.load(expanded_local_pos_ptr + row).to(tl.int64)
    if step >= num_steps:
        return
    req = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    cell = req * num_steps + step
    num = tl.load(q_num_ptr + cell)
    offs = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # A cell the drafter never wrote reads as the stock cache's zero fill.
    fill = tl.where(num > 0, float("-inf"), 0.0)
    x = tl.zeros((BLOCK_SIZE,), tl.float32) + fill
    for s in tl.static_range(KPQ):
        tok = tl.load(q_idx_ptr + cell * KPQ + s)
        val = tl.load(q_val_ptr + cell * KPQ + s)
        x = tl.where((offs == tok) & (s < num), val, x)
    tl.store(
        dense_ptr + req * dense_stride_0 + step * dense_stride_1 + offs,
        x.to(dense_ptr.dtype.element_ty),
        mask=offs < vocab_size,
    )


# ---------------------------------------------------------------------------
# H159: sampler diet (the verify-tail fold).
#
# VLLM_H159_TAIL=1, bit-identical to the stock parity verify:
#   _h159_tsubmax_kernel replaces fused_rejection._penalized_submax_kernel. One
#   CTA per (row, vocab block) instead of one per (request, block) that walks
#   the request's rows one after another, each behind two dependent loads of
#   the draft history; it also writes the maxima of GROUP sub-blocks.
#   _h159_tselect_kernel replaces _select_survivors_kernel and
#   _parity_rejection_kernel. One program per (request, row) selects the row's
#   survivors (stage A from the group maxima), then does the row's part of the
#   rejection sampler: its accept bit and the token the row emits if it is the
#   first rejected position (or the bonus one). The request's last program to
#   finish picks the accepted length and writes the outputs.
#   The integer counts, the per-element arithmetic and the op chain from the
#   top-k on (8 warps) are the stock ones, so survivors, logsumexp, uniforms
#   and Gumbel noise are too, and so are sampled / num_sampled / num_rejected.
#   PREWAIT (with glue PDL): both read what earlier kernels wrote (request
#   state, output counts, token ids, the draft q cache) before the PDL wait,
#   and only the logits and their own predecessor's output after it. Under the
#   glue convention a kernel triggers right after its wait, so when a kernel
#   starts, every kernel two or more launches back has completed.
# ---------------------------------------------------------------------------
@triton.jit
def _h159_tsubmax_kernel(
    # [num_logits, num_sub] / [num_logits, num_sub // GROUP] fp32
    submax_ptr,
    submax_stride,
    gmax_ptr,
    gmax_stride,
    # [num_logits, V]
    logits_ptr,
    logits_stride,
    # [num_logits]
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    # [num_tokens] / [num_logits]
    input_ids_ptr,
    logits_indices_ptr,
    # [max_num_reqs]
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
    GROUP: tl.constexpr,
    # Power of two above the most logits rows of one request (1 + drafts).
    MAX_HIST: tl.constexpr,
    HAS_PENALTIES: tl.constexpr,
    PREWAIT: tl.constexpr,
    GLUE_PDL: tl.constexpr = False,
):
    """fused_rejection._penalized_submax_kernel's sub-block maxima, bit for bit,
    from one CTA per (row, vocab block) instead of one per (request, block)
    walking the request's rows, plus the maxima of each GROUP sub-blocks. Row
    p's draft history (the input tokens of rows 1..p) is one vector load,
    added only where it falls in the CTA's block. PREWAIT reads everything but
    the logits (request state, output counts, token ids: all written before
    the target forward) before the PDL wait."""
    if GLUE_PDL and not PREWAIT:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    temp = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    if HAS_PENALTIES:
        rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
        freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
        pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)
        use_rep_penalty = rep_penalty != 1.0
        use_penalty = use_rep_penalty or freq_penalty != 0.0 or pres_penalty != 0.0
        output_bin_counts = tl.load(
            output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
            mask=mask & use_penalty,
            other=0,
        )
        local_pos = tl.load(expanded_local_pos_ptr + row)
        h = tl.arange(0, MAX_HIST)
        in_hist = (h >= 1) & (h <= local_pos) & use_penalty
        token_idx = tl.load(logits_indices_ptr + row - local_pos + h, mask=in_hist, other=0)
        prev_tokens = tl.load(input_ids_ptr + token_idx, mask=in_hist, other=-1)
        in_block = (
            in_hist
            & (prev_tokens >= block_idx * BLOCK_SIZE)
            & (prev_tokens < (block_idx + 1) * BLOCK_SIZE)
        )
        if tl.max(in_block.to(tl.int32), axis=0) > 0:
            for s in tl.static_range(MAX_HIST):
                prev_token = tl.max(tl.where(h == s, prev_tokens, -1), axis=0)
                output_bin_counts += (block == prev_token).to(tl.int32)
        prompt_bin_mask = tl.zeros((BLOCK_SIZE,), tl.int1)
        if use_rep_penalty:
            packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
            packed_mask = tl.load(
                prompt_bin_mask_ptr
                + req_state_idx * prompt_bin_mask_stride
                + packed_block,
                mask=packed_block < tl.cdiv(vocab_size, 32),
                other=0,
            )
            bits = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
            prompt_bin_mask = bits.to(tl.int1).reshape(BLOCK_SIZE)
    if GLUE_PDL and PREWAIT:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    logits = tl.load(
        logits_ptr + row * logits_stride + block, mask=mask, other=float("-inf")
    ).to(tl.float32)
    if HAS_PENALTIES:
        if use_penalty:
            logits = _penalize(
                logits,
                output_bin_counts,
                prompt_bin_mask,
                rep_penalty,
                freq_penalty,
                pres_penalty,
                use_rep_penalty,
            )
    if temp != 0.0 and temp != 1.0:
        logits = logits / temp
    logits = tl.where(mask, logits, float("-inf"))
    NUM_SUB: tl.constexpr = BLOCK_SIZE // SUB_SIZE
    submax = tl.max(tl.reshape(logits, (NUM_SUB, SUB_SIZE)), axis=1)
    tl.store(
        submax_ptr + row * submax_stride + block_idx * NUM_SUB + tl.arange(0, NUM_SUB),
        submax,
    )
    if GROUP > 0:
        NUM_G: tl.constexpr = NUM_SUB // GROUP
        tl.store(
            gmax_ptr + row * gmax_stride + block_idx * NUM_G + tl.arange(0, NUM_G),
            tl.max(tl.reshape(submax, (NUM_G, GROUP)), axis=1),
        )


@triton.jit
def _h159_draw(value, token, residual, seed, pos, temp, USE_FP64: tl.constexpr):
    """_parity_rejection_kernel's final draw (per-token Gumbel noise, lowest
    id among ties, 0 if nothing is left)."""
    noised = gumbel_noised_logits(
        value,
        token,
        residual,
        seed,
        pos,
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    best = tl.max(noised, axis=0)
    drawn = tl.min(tl.where(residual & (noised == best), token, _INT32_MAX))
    return tl.where(best > float("-inf"), drawn, 0).to(tl.int64)


@triton.jit
def _h159_tselect_kernel(
    # [num_reqs, num_speculative_steps + 1] int64, [num_reqs] int32
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    # [num_logits] int32 / int64 per-row results; [max_num_reqs] int32, zero
    # between launches (the last arrival resets its entry)
    accept_ptr,
    cand_ptr,
    arrive_ptr,
    # from _h159_tsubmax_kernel
    submax_ptr,
    submax_stride,
    num_sub,
    gmax_ptr,
    gmax_stride,
    num_groups,
    # [num_logits, V]
    logits_ptr,
    logits_stride,
    # [num_reqs + 1] / [num_reqs]
    cu_num_logits_ptr,
    idx_mapping_ptr,
    seq_lens_ptr,
    # [num_tokens] / [num_logits]
    input_ids_ptr,
    positions_ptr,
    logits_indices_ptr,
    # [max_num_reqs]
    prefill_len_ptr,
    top_k_ptr,
    top_p_ptr,
    temperature_ptr,
    seeds_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    # [max_num_reqs, num_steps, KPQ] / [max_num_reqs, num_steps] draft q
    q_val_ptr,
    q_idx_ptr,
    q_num_ptr,
    q_lse_ptr,
    num_steps,
    vocab_size,
    KP: tl.constexpr,
    KPQ: tl.constexpr,
    SUB_SIZE: tl.constexpr,
    PADDED_NUM_SUB: tl.constexpr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_PENALTIES: tl.constexpr,
    TOP_P: tl.constexpr,
    USE_FP64: tl.constexpr,
    PREWAIT: tl.constexpr,
    GLUE_PDL: tl.constexpr = False,
):
    if GLUE_PDL and not PREWAIT:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    req_idx = tl.program_id(0)
    i = tl.program_id(1)
    start = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    num_rows = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64) - start
    if i >= num_rows:
        return
    row = start + i
    num_draft_tokens = num_rows - 1
    is_bonus = i == num_draft_tokens
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    top_k = tl.load(top_k_ptr + req_state_idx)
    temp = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    seed = tl.load(seeds_ptr + req_state_idx)
    pos = tl.load(positions_ptr + tl.load(logits_indices_ptr + row))
    # The draft this row verifies: the next row's input token.
    next_token = tl.load(
        input_ids_ptr
        + tl.load(logits_indices_ptr + row + 1, mask=not is_bonus, other=0),
        mask=not is_bonus,
        other=0,
    )
    if HAS_PENALTIES:
        rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
        freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
        pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)
        use_rep_penalty = rep_penalty != 1.0
        use_penalty = use_rep_penalty or freq_penalty != 0.0 or pres_penalty != 0.0
        # The row's draft history: the input tokens of rows 1..i.
        h = tl.arange(0, ROWS)
        in_hist = (h >= 1) & (h <= i) & use_penalty
        token_idx = tl.load(logits_indices_ptr + start + h, mask=in_hist, other=0)
        prev_tokens = tl.load(input_ids_ptr + token_idx, mask=in_hist, other=-1)
    # The q of the draft this row verifies (unused for the bonus row).
    q_idx, q_x, q_lse = _load_q_row(
        q_val_ptr,
        q_idx_ptr,
        q_num_ptr,
        q_lse_ptr,
        req_state_idx * num_steps + tl.minimum(i, num_steps - 1),
        temp,
        KPQ,
    )
    if GLUE_PDL and PREWAIT:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    # 1. The KP sub-blocks with the largest (max, -index) keys, two-level from
    # the group maxima (the stock kernel reloads every sub-block maximum).
    if GROUP > 0:
        NUM_GROUPS: tl.constexpr = PADDED_NUM_SUB // GROUP
        grp = tl.arange(0, NUM_GROUPS)
        gm = tl.load(
            gmax_ptr + row * gmax_stride + grp,
            mask=grp < num_groups,
            other=float("-inf"),
        )
        top_grp = tl.topk(_pack_keys(gm, grp, NUM_GROUPS), KP)
        _, grp_idx = _unpack_keys(top_grp, NUM_GROUPS)
        sub = grp_idx[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
        submax = tl.load(
            submax_ptr + row * submax_stride + sub,
            mask=sub < num_sub,
            other=float("-inf"),
        )
        sub_keys = tl.reshape(_pack_keys(submax, sub, PADDED_NUM_SUB), (KP * GROUP,))
    else:
        sub = tl.arange(0, PADDED_NUM_SUB)
        submax = tl.load(
            submax_ptr + row * submax_stride + sub,
            mask=sub < num_sub,
            other=float("-inf"),
        )
        sub_keys = _pack_keys(submax, sub, PADDED_NUM_SUB)
    top_sub = tl.topk(sub_keys, KP)
    _, sub_idx = _unpack_keys(top_sub, PADDED_NUM_SUB)

    # 2. Their processed logits.
    offs = sub_idx[:, None] * SUB_SIZE + tl.arange(0, SUB_SIZE)[None, :]
    mask = offs < vocab_size
    logits = tl.load(
        logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")
    ).to(tl.float32)
    if HAS_PENALTIES:
        if use_penalty:
            output_bin_counts = tl.load(
                output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + offs,
                mask=mask,
                other=0,
            )
            for s in tl.static_range(1, ROWS):
                # prev_tokens holds -1 outside rows 1..i: no token matches it.
                prev_token = tl.max(tl.where(h == s, prev_tokens, -1), axis=0)
                output_bin_counts += (offs == prev_token).to(tl.int32)
            prompt_bin_mask = tl.zeros((KP, SUB_SIZE), tl.int1)
            if use_rep_penalty:
                word = tl.load(
                    prompt_bin_mask_ptr
                    + req_state_idx * prompt_bin_mask_stride
                    + offs // 32,
                    mask=mask,
                    other=0,
                )
                prompt_bin_mask = ((word >> (offs % 32).to(tl.int32)) & 1).to(tl.int1)
            logits = _penalize(
                logits,
                output_bin_counts,
                prompt_bin_mask,
                rep_penalty,
                freq_penalty,
                pres_penalty,
                use_rep_penalty,
            )
    if temp != 0.0 and temp != 1.0:
        logits = logits / temp

    # 3. Exact top-k by (value desc, token id asc), then top-p, and 4. the
    # survivors' logsumexp: _select_survivors_kernel's op chain.
    sel = mask & (logits > float("-inf"))
    keys = tl.where(sel, _pack_keys(logits, offs, vocab_size), _INT64_MIN)
    top = tl.topk(tl.reshape(keys, (KP * SUB_SIZE,)), KP)
    pos_k = tl.arange(0, KP)
    num_valid = tl.sum(((pos_k < top_k) & (top != _INT64_MIN)).to(tl.int32))
    valid = pos_k < num_valid
    value, token = _unpack_keys(top, vocab_size)
    num_keep = num_valid
    if TOP_P:
        top_p = tl.load(top_p_ptr + req_state_idx)
        max_value = tl.max(tl.where(valid, value, float("-inf")), axis=0)
        e = tl.where(valid, tl.exp(value - max_value), 0.0)
        prob = e / tl.sum(e, axis=0)
        cum = tl.cumsum(prob, axis=0)
        below = valid & (pos_k < num_valid - 1) & (cum < top_p)
        num_below = tl.sum(below.to(tl.int32))
        num_keep = tl.where(
            top_p < 1.0, tl.minimum(num_below + 1, num_valid), num_valid
        )
    keep = pos_k < num_keep
    max_value = tl.max(tl.where(keep, value, float("-inf")), axis=0)
    sumexp = tl.sum(tl.where(keep, tl.exp(value - max_value), 0.0), axis=0)
    lse = tl.where(num_keep > 0, max_value + tl.log(sumexp), float("-inf"))

    # 5. This row's part of _parity_rejection_kernel: the accept test of the
    # draft it verifies, and what the stock kernel writes at this row's slot if
    # the row is the first rejected one (or the bonus position).
    accepted = tl.zeros((), tl.int32)
    if is_bonus:
        cand = _h159_draw(
            tl.where(keep, value, float("-inf")),
            tl.where(keep, token, 0),
            keep,
            seed,
            pos,
            temp,
            USE_FP64,
        )
    else:
        draft = next_token.to(tl.int64)
        is_valid_draft = draft >= 0
        draft = tl.maximum(0, draft)
        if temp == 0.0:
            target_argmax = tl.where(
                num_keep > 0, tl.sum(tl.where(pos_k == 0, token, 0)), 0
            ).to(tl.int64)
            accepted = ((target_argmax == draft) & is_valid_draft).to(tl.int32)
            cand = target_argmax
        else:
            u = tl_rand32(seed, pos, includes_zero=False)
            target_logit = tl.max(
                tl.where(tl.where(keep, token, -1) == draft, value, float("-inf")),
                axis=0,
            )
            target_logprob = target_logit - lse
            q_logit = tl.max(tl.where(q_idx == draft, q_x, float("-inf")), axis=0)
            draft_logprob = tl.where(q_logit > float("-inf"), q_logit - q_lse, 0.0)
            accepted = (
                is_valid_draft & (target_logprob > tl.log(u) + draft_logprob)
            ).to(tl.int32)
            # Resample from norm(max(0, p - q)) over the target survivors.
            residual = keep
            rtok = tl.where(residual, token, 0)
            rval = tl.where(residual, value, float("-inf"))
            if next_token >= 0:
                q_rejected = tl.max(
                    tl.where(q_idx == next_token, q_x, float("-inf")), axis=0
                )
                if q_rejected > float("-inf"):
                    log_p = rval - lse
                    q_at = tl.max(
                        tl.where(
                            rtok[:, None] == q_idx[None, :],
                            q_x[None, :],
                            float("-inf"),
                        ),
                        axis=1,
                    )
                    ratio = tl.exp((q_at - q_lse) - log_p)
                    res = tl.where(
                        residual & (ratio < 1.0),
                        log_p + tldevice.log1p(-ratio),
                        float("-inf"),
                    )
                    # p == q to rounding leaves no residual mass: keep p.
                    if tl.max(res, axis=0) > float("-inf"):
                        rval = res
                        residual = res > float("-inf")
                else:
                    residual &= rtok != next_token
                    rval = tl.where(residual, rval, float("-inf"))
            cand = _h159_draw(rval, rtok, residual, seed, pos, temp, USE_FP64)
    tl.store(accept_ptr + row, accepted)
    tl.store(cand_ptr + row, cand)

    # 6. Publish this row; the request's last row to arrive writes its outputs
    # as _parity_rejection_kernel does.
    tl.debug_barrier()
    arrived = tl.atomic_add(arrive_ptr + req_idx, 1, sem="acq_rel", scope="gpu")
    if arrived == num_rows - 1:
        tl.store(arrive_ptr + req_idx, 0)
        j = tl.arange(0, ROWS)
        is_draft = j < num_draft_tokens
        acc = tl.load(
            accept_ptr + start + j, mask=is_draft, other=0, cache_modifier=".cg"
        )
        accepted_length = tl.min(tl.where(is_draft & (acc == 0), j, num_draft_tokens))
        is_chunked_prefilling = tl.load(seq_lens_ptr + req_idx) < tl.load(
            prefill_len_ptr + req_state_idx
        )
        final_num_sampled = tl.where(is_chunked_prefilling, 0, accepted_length + 1)
        tl.store(num_sampled_ptr + req_idx, final_num_sampled.to(tl.int32))
        tl.store(
            num_rejected_ptr + req_idx,
            tl.where(is_chunked_prefilling, 0, num_rows - final_num_sampled).to(
                tl.int32
            ),
        )
        kept = j < accepted_length
        drafts = tl.load(
            input_ids_ptr
            + tl.load(logits_indices_ptr + start + 1 + j, mask=kept, other=0),
            mask=kept,
            other=0,
        ).to(tl.int64)
        tl.store(
            sampled_ptr + req_idx * sampled_stride + j,
            tl.maximum(0, drafts),
            mask=kept,
        )
        tl.store(
            sampled_ptr + req_idx * sampled_stride + accepted_length,
            tl.load(cand_ptr + start + accepted_length, cache_modifier=".cg"),
        )


class ParityState:
    """The drafter's sparse q cache and the sampler state it processes with."""

    def __init__(self, speculator, sampler):
        assert PARITY_KP in (16, 32, 64), "VLLM_H53_DRAFT_KP must be 16, 32 or 64"
        self.kp = PARITY_KP
        self.num_steps = speculator.num_speculative_steps
        self.dense = speculator.draft_logits
        max_reqs, _, vocab_size = self.dense.shape
        device = self.dense.device
        shape = (max_reqs, self.num_steps)
        self.q_val = torch.full(
            (*shape, self.kp), float("-inf"), dtype=torch.float32, device=device
        )
        self.q_idx = torch.full((*shape, self.kp), -1, dtype=torch.int32, device=device)
        self.q_num = torch.zeros(shape, dtype=torch.int32, device=device)
        self.q_lse = torch.zeros(shape, dtype=torch.float32, device=device)
        self.num_sub = triton.cdiv(vocab_size, _BLOCK_SIZE) * (_BLOCK_SIZE // _SUB_SIZE)
        self.padded_num_sub = max(triton.next_power_of_2(self.num_sub), self.kp)
        self.group = (
            _GROUP if _GROUP and self.padded_num_sub // _GROUP >= self.kp else 0
        )
        # Persistent: the drafter's CUDA graphs bake its address in.
        self.submax = torch.empty(
            max_reqs, self.num_sub, dtype=torch.float32, device=device
        )
        states = sampler.sampling_states
        pen = sampler.penalties_state
        self.top_k = states.top_k.gpu
        self.top_p = states.top_p.gpu
        self.penalties = (
            pen.repetition_penalty.gpu,
            pen.frequency_penalty.gpu,
            pen.presence_penalty.gpu,
            pen.prompt_bin_mask,
            pen.output_bin_counts,
        )
        self.use_fp64 = speculator.use_fp64_gumbel
        self.num_sparse = 0
        self.num_dense = 0
        # VLLM_H97_FAST_SELECT: the same draw and q cache from _h97_*_kernel.
        self.fast = FAST_SELECT
        self.max_hist = triton.next_power_of_2(max(self.num_steps, 1))
        # 32-bit sort keys need bf16-exact processed values (the cache dtype).
        bf16 = self.dense.dtype == torch.bfloat16
        self.key32a = bf16 and self.padded_num_sub <= 65536
        self.key32b = bf16 and self.kp * _SUB_SIZE <= 65536
        self.select_warps = 8
        self.num_groups = self.num_sub // self.group if self.group else 0
        self.gmax = torch.empty(
            max_reqs,
            max(self.padded_num_sub // self.group, 1) if self.group else 1,
            dtype=torch.float32,
            device=device,
        )
        # H159 verify-tail fold; unused with VLLM_H159_TAIL off.
        self.h159_tail = H159_TAIL
        # Read earlier kernels' outputs before the PDL wait.
        self.h159_prewait = True
        self.h159_rows = triton.next_power_of_2(self.num_steps + 1)
        self.h159_arrive = (
            torch.zeros(max_reqs, dtype=torch.int32, device=device)
            if self.h159_tail
            else None
        )


def bind(speculator, sampler, rejection_sampler) -> None:
    """Attach a ParityState to the drafter and the rejection sampler (before
    CUDA-graph capture). No-op with VLLM_H53_DRAFT_PARITY=0, and silently so
    (stock path) when the flag is left at its default and the drafts are not
    sampled probabilistically."""
    if not PARITY:
        return
    if not _PARITY_SET and getattr(speculator, "draft_logits", None) is None:
        return
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        AutoRegressiveSpeculator,
    )
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    reason = None
    if getattr(speculator, "draft_logits", None) is None:
        reason = "needs speculative-config draft_sample_method=probabilistic"
    elif not isinstance(speculator, AutoRegressiveSpeculator):
        reason = f"{type(speculator).__name__} is not an autoregressive drafter"
    elif (
        speculator.draft_watermarker is not None
        or speculator.acceptance_estimator is not None
        or speculator.draft_vocab_head is not None
        or speculator.use_local_argmax_reduction
        or speculator.pcp_manager is not None
    ):
        reason = (
            "draft watermarking, adaptive verification, a reduced-vocab head, "
            "local argmax reduction and PCP are not supported"
        )
    elif type(sampler) is not Sampler or type(rejection_sampler) is not (
        RejectionSampler
    ):
        reason = "needs the stock Sampler and RejectionSampler"
    elif speculator.draft_logits.shape[0] != sampler.sampling_states.max_num_reqs:
        reason = "draft-logits cache and sampler disagree on max_num_reqs"
    if reason is not None:
        logger.warning("H53 draft parity not applied: %s", reason)
        return
    state = ParityState(speculator, sampler)
    speculator.h53 = state
    rejection_sampler.h53 = state
    logger.info(
        "H53 draft parity on (VLLM_H53_DRAFT_PARITY=1): drafts sampled from the "
        "penalized, temperature-scaled, top-k/top-p processed MTP distribution "
        "(sparse, KP=%d); the verifier tests against that q",
        state.kp,
    )
    if state.fast:
        logger.info(
            "H97 fast parity select on (VLLM_H97_FAST_SELECT=1): same draws and q "
            "cache from _h97_submax_kernel + _h97_select_kernel (32-bit keys: "
            "stage A %s, final top-k %s; %d warps)",
            state.key32a,
            state.key32b,
            state.select_warps,
        )
    if H105_PDL:
        logger.info(
            "H105 PDL on (VLLM_H105_PDL=1): parity submax/select/rejection kernels "
            "launch with PDL: %s",
            _pdl(),
        )
    if state.h159_tail:
        logger.info(
            "H159 tail on (VLLM_H159_TAIL=1): parity verify as _h159_tsubmax_kernel "
            "+ _h159_tselect_kernel (rejection folded into the survivor select; "
            "bit-identical outputs), vectorized post_update"
        )


def sample(
    state: ParityState,
    # [rows, V] raw draft logits
    logits: torch.Tensor,
    # [rows] request state index (-1: CUDA-graph padding)
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    # [rows] sampling position key
    pos: torch.Tensor,
    # [] int64 current draft step
    draft_step: torch.Tensor,
    # [max_num_reqs, num_steps] drafts of this proposal (batch-indexed)
    draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """Draw one draft token per row from the processed q and cache q."""
    num_rows, vocab_size = logits.shape
    sampled = torch.empty(num_rows, dtype=torch.int64, device=logits.device)
    if num_rows == 0:
        return sampled
    assert num_rows <= state.submax.shape[0] and logits.stride(1) == 1
    assert draft_step.dim() == 0 and draft_tokens.stride(1) == 1
    rep, freq, pres, prompt_bin_mask, output_bin_counts = state.penalties
    pos = pos.contiguous()
    pdl = _pdl()
    if state.fast:
        _h97_sample(
            state, sampled, logits, idx_mapping, temperature, seeds, pos, draft_step,
            draft_tokens, pdl,
        )  # fmt: skip
        return sampled
    _parity_submax_kernel[(num_rows, triton.cdiv(vocab_size, _BLOCK_SIZE))](
        state.submax,
        state.submax.stride(0),
        logits,
        logits.stride(0),
        idx_mapping,
        draft_tokens,
        draft_tokens.stride(0),
        draft_step,
        state.dense,
        rep,
        freq,
        pres,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=_BLOCK_SIZE,
        SUB_SIZE=_SUB_SIZE,
        GLUE_PDL=pdl,
        launch_pdl=pdl,
        num_warps=4,
    )
    _parity_select_kernel[(num_rows,)](
        sampled,
        state.q_val,
        state.q_idx,
        state.q_num,
        state.q_lse,
        state.num_steps,
        state.submax,
        state.submax.stride(0),
        state.num_sub,
        logits,
        logits.stride(0),
        idx_mapping,
        pos,
        draft_tokens,
        draft_tokens.stride(0),
        draft_step,
        state.dense,
        temperature,
        seeds,
        state.top_k,
        state.top_p,
        rep,
        freq,
        pres,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        KP=state.kp,
        SUB_SIZE=_SUB_SIZE,
        PADDED_NUM_SUB=state.padded_num_sub,
        GROUP=state.group,
        USE_FP64=state.use_fp64,
        GLUE_PDL=pdl,
        launch_pdl=pdl,
        num_warps=8,
    )
    return sampled


def _h97_sample(
    state: ParityState,
    sampled: torch.Tensor,
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    draft_step: torch.Tensor,
    draft_tokens: torch.Tensor,
    pdl: bool = False,
) -> None:
    """VLLM_H97_FAST_SELECT: sample()'s two launches with the H97 kernels."""
    num_rows, vocab_size = logits.shape
    rep, freq, pres, prompt_bin_mask, output_bin_counts = state.penalties
    _h97_submax_kernel[(num_rows, triton.cdiv(vocab_size, _BLOCK_SIZE))](
        state.submax,
        state.submax.stride(0),
        state.gmax,
        state.gmax.stride(0),
        logits,
        logits.stride(0),
        idx_mapping,
        draft_tokens,
        draft_tokens.stride(0),
        draft_step,
        state.dense,
        rep,
        freq,
        pres,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=_BLOCK_SIZE,
        SUB_SIZE=_SUB_SIZE,
        GROUP=state.group,
        GLUE_PDL=pdl,
        launch_pdl=pdl,
        num_warps=4,
    )
    _h97_select_kernel[(num_rows,)](
        sampled,
        state.q_val,
        state.q_idx,
        state.q_num,
        state.q_lse,
        state.num_steps,
        state.submax,
        state.submax.stride(0),
        state.num_sub,
        state.gmax,
        state.gmax.stride(0),
        state.num_groups,
        logits,
        logits.stride(0),
        idx_mapping,
        pos,
        draft_tokens,
        draft_tokens.stride(0),
        draft_step,
        state.dense,
        temperature,
        seeds,
        state.top_k,
        state.top_p,
        rep,
        freq,
        pres,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        KP=state.kp,
        SUB_SIZE=_SUB_SIZE,
        PADDED_NUM_SUB=state.padded_num_sub,
        GROUP=state.group,
        USE_FP64=state.use_fp64,
        MAX_HIST=state.max_hist,
        KEY32A=state.key32a,
        KEY32B=state.key32b,
        GLUE_PDL=pdl,
        launch_pdl=pdl,
        num_warps=state.select_warps,
    )


def _h159_rejection_sample(
    state: ParityState,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits_indices: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    seq_lens: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    prefill_len: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    penalties: tuple[torch.Tensor, ...] | None,
    max_top_k: int,
    use_top_p: bool,
    use_fp64: bool,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> None:
    """VLLM_H159_TAIL: select_survivors + _parity_rejection_kernel's outputs
    from _h159_tsubmax_kernel + _h159_tselect_kernel."""
    assert logits.ndim == 2 and logits.stride(1) == 1
    assert 1 <= max_top_k <= MAX_TOP_K
    num_logits, vocab_size = logits.shape
    num_reqs = cu_num_logits.shape[0] - 1
    assert num_reqs <= state.h159_arrive.shape[0]
    assert num_logits <= num_reqs * (state.num_steps + 1)
    device = logits.device
    kp = _top_k_pow2(max_top_k)
    num_blocks = triton.cdiv(vocab_size, _BLOCK_SIZE)
    num_sub = num_blocks * (_BLOCK_SIZE // _SUB_SIZE)
    padded_num_sub = max(triton.next_power_of_2(num_sub), kp)
    group = _GROUP if _GROUP and padded_num_sub // _GROUP >= kp else 0
    num_groups = num_sub // group if group else 0
    submax = torch.empty(num_logits, num_sub, dtype=torch.float32, device=device)
    gmax = torch.empty(
        num_logits, max(num_groups, 1), dtype=torch.float32, device=device
    )
    accept = torch.empty(num_logits, dtype=torch.int32, device=device)
    cand = torch.empty(num_logits, dtype=torch.int64, device=device)
    has_penalties = penalties is not None
    if penalties is None:
        # Never dereferenced: HAS_PENALTIES=False compiles the loads out.
        rep = freq = pres = prompt_bin_mask = output_bin_counts = temperature
        prompt_stride = counts_stride = 0
    else:
        rep, freq, pres, prompt_bin_mask, output_bin_counts = penalties
        prompt_stride = prompt_bin_mask.stride(0)
        counts_stride = output_bin_counts.stride(0)
    pdl = glue_pdl()
    prewait = state.h159_prewait and pdl
    _h159_tsubmax_kernel[(num_logits, num_blocks)](
        submax,
        submax.stride(0),
        gmax,
        gmax.stride(0),
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        expanded_local_pos,
        input_ids,
        logits_indices,
        temperature,
        rep,
        freq,
        pres,
        prompt_bin_mask,
        prompt_stride,
        output_bin_counts,
        counts_stride,
        vocab_size,
        BLOCK_SIZE=_BLOCK_SIZE,
        SUB_SIZE=_SUB_SIZE,
        GROUP=group,
        MAX_HIST=state.h159_rows,
        HAS_PENALTIES=has_penalties,
        PREWAIT=prewait,
        GLUE_PDL=pdl,
        launch_pdl=pdl,
        num_warps=4,
    )
    _h159_tselect_kernel[(num_reqs, state.h159_rows)](
        sampled,
        sampled.stride(0),
        num_sampled,
        num_rejected,
        accept,
        cand,
        state.h159_arrive,
        submax,
        submax.stride(0),
        num_sub,
        gmax,
        gmax.stride(0),
        num_groups,
        logits,
        logits.stride(0),
        cu_num_logits,
        idx_mapping,
        seq_lens,
        input_ids,
        positions,
        logits_indices,
        prefill_len,
        top_k,
        top_p,
        temperature,
        seeds,
        rep,
        freq,
        pres,
        prompt_bin_mask,
        prompt_stride,
        output_bin_counts,
        counts_stride,
        state.q_val,
        state.q_idx,
        state.q_num,
        state.q_lse,
        state.num_steps,
        vocab_size,
        KP=kp,
        KPQ=state.kp,
        SUB_SIZE=_SUB_SIZE,
        PADDED_NUM_SUB=padded_num_sub,
        GROUP=group,
        ROWS=state.h159_rows,
        HAS_PENALTIES=has_penalties,
        TOP_P=use_top_p,
        USE_FP64=use_fp64,
        PREWAIT=prewait,
        GLUE_PDL=pdl,
        launch_pdl=pdl,
        num_warps=8,
    )


def rejection_sample(
    state: ParityState,
    # [num_logits, V] raw target logits
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits_indices: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    seq_lens: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    prefill_len: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    penalties: tuple[torch.Tensor, ...] | None,
    max_top_k: int,
    use_top_p: bool,
    num_speculative_steps: int,
    use_fp64: bool = False,
    count: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fused_rejection_sample against the cached sparse draft q."""
    num_reqs = cu_num_logits.shape[0] - 1
    device = logits.device
    sampled = torch.empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64, device=device
    )
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=device)
    if num_reqs == 0:
        return sampled, num_sampled, num_rejected
    if state.h159_tail:
        _h159_rejection_sample(
            state, logits, input_ids, positions, logits_indices, cu_num_logits,
            idx_mapping, seq_lens, expanded_idx_mapping, expanded_local_pos,
            prefill_len, temperature, seeds, top_k, top_p,
            penalties, max_top_k, use_top_p, use_fp64, sampled, num_sampled,
            num_rejected,
        )  # fmt: skip
        if count:
            if state.num_sparse == 0:
                logger.info(
                    "H53 draft parity: first sparse verify (%d requests); H159 "
                    "tail: folded",
                    num_reqs,
                )
            state.num_sparse += 1
            _maybe_log_counts(state)
        return sampled, num_sampled, num_rejected
    surv_val, surv_idx, num_surv, lse = select_survivors(
        logits,
        input_ids,
        logits_indices,
        cu_num_logits,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        top_k,
        top_p,
        penalties,
        max_top_k,
        use_top_p,
    )
    _parity_rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        num_rejected,
        surv_val,
        surv_idx,
        num_surv,
        lse,
        state.q_val,
        state.q_idx,
        state.q_num,
        state.q_lse,
        state.num_steps,
        cu_num_logits,
        idx_mapping,
        seq_lens,
        prefill_len,
        temperature,
        seeds,
        input_ids,
        positions,
        logits_indices,
        KP=surv_val.shape[1],
        KPQ=state.kp,
        USE_FP64=use_fp64,
        GLUE_PDL=_pdl(),
        launch_pdl=_pdl(),
        num_warps=1,
    )
    if count:
        if state.num_sparse == 0:
            logger.info(
                "H53 draft parity: first sparse verify (%d requests)", num_reqs
            )
        state.num_sparse += 1
        _maybe_log_counts(state)
    return sampled, num_sampled, num_rejected


def densify(
    state: ParityState,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
) -> None:
    """Write the batch's sparse q into the dense draft-logits cache, for the
    stock unfused verifier (logprobs, other logits processors, ...)."""
    num_logits = expanded_idx_mapping.shape[0]
    if num_logits == 0:
        return
    dense = state.dense
    block = 4096
    _densify_kernel[(num_logits, triton.cdiv(dense.shape[2], block))](
        dense,
        dense.stride(0),
        dense.stride(1),
        state.q_val,
        state.q_idx,
        state.q_num,
        state.num_steps,
        expanded_idx_mapping,
        expanded_local_pos,
        dense.shape[2],
        KPQ=state.kp,
        BLOCK_SIZE=block,
        num_warps=4,
    )
    if state.num_dense == 0:
        logger.info(
            "H53 draft parity: first dense-fallback verify (%d rows)", num_logits
        )
    state.num_dense += 1
    _maybe_log_counts(state)


_COUNT_LOG_EVERY = 50000


def _maybe_log_counts(state: ParityState) -> None:
    if (state.num_sparse + state.num_dense) % _COUNT_LOG_EVERY == 0:
        logger.info(
            "H53 draft parity: verify steps sparse=%d dense-fallback=%d",
            state.num_sparse,
            state.num_dense,
        )
