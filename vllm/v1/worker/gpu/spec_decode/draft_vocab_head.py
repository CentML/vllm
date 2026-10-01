# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in reduced-vocabulary lm_head for greedy draft tokens.

A greedy draft step only needs ``argmax(lm_head(h))``. The stock path computes
logits for the whole vocabulary (Qwen3.6: 248,320 x 2,048 BF16, a 1.02 GB
weight read per draft step, three steps per iteration with k=3). With this
option the draft argmax is taken over a fixed subset of token ids, from a
compact copy of their lm_head rows, so a draft step reads only those rows.

Only the draft proposals can change: when the full-vocabulary argmax lies
outside the subset, the draft proposes the best in-subset token instead. The
target model, its verification and its sampling are untouched, and greedy
drafts are verified by rejection sampling with one-hot draft probabilities,
so the output distribution is the target's. The cost of a too-small subset is
acceptance length, nothing else.

The subset is given as half-open token-id ranges. It is chosen without any
benchmark data, e.g. ``0-98304,248044-248070`` for Qwen3.6 = the first 98,304
byte-level BPE ids (merge order) plus the tokenizer's added/special tokens.

Environment:
    VLLM_DRAFT_LMH_VOCAB=a-b[,c-d...]   enable; token-id ranges [a, b) kept for
                                        the draft argmax (default off)
    VLLM_DRAFT_LMH_DTYPE=bf16|mxfp8     subset head precision (default bf16;
                                        mxfp8 = block-32 E4M3 weights and
                                        activations, cuBLAS block-scaled GEMM)
    VLLM_DRAFT_LMH_DIAG=1               also compute the full-vocabulary argmax
                                        and count draft tokens that differ
                                        (device counters, logged every
                                        VLLM_DRAFT_LMH_DIAG_S seconds; costs a
                                        full lm_head per step: diagnostics only)

When ``VLLM_DRAFT_LMH_VOCAB`` is set, this head replaces the draft model's
lm_head for the draft argmax, so it takes precedence over the MXFP8 draft
lm_head (``VLLM_MTP_DRAFT_LM_HEAD_MXFP8``); do not combine the two.
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_SPEC = os.environ.get("VLLM_DRAFT_LMH_VOCAB", "").strip()
_DTYPE = os.environ.get("VLLM_DRAFT_LMH_DTYPE", "bf16").strip().lower()
_DIAG = os.environ.get("VLLM_DRAFT_LMH_DIAG", "0") == "1"
_DIAG_S = float(os.environ.get("VLLM_DRAFT_LMH_DIAG_S", "60") or 60)

ENABLED = bool(_SPEC) and _SPEC not in ("0", "off")


def parse_ranges(spec: str, vocab_size: int) -> list[tuple[int, int]]:
    """Parse ``a-b,c-d`` into sorted, non-overlapping [a, b) ranges."""
    ranges = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, sep, b = part.partition("-")
        if not sep:
            raise ValueError(f"VLLM_DRAFT_LMH_VOCAB: bad range {part!r} (want a-b)")
        lo, hi = int(a), int(b)
        if not 0 <= lo < hi <= vocab_size:
            raise ValueError(
                f"VLLM_DRAFT_LMH_VOCAB: range {part!r} outside [0, {vocab_size})"
            )
        ranges.append((lo, hi))
    if not ranges:
        raise ValueError("VLLM_DRAFT_LMH_VOCAB: no ranges")
    ranges.sort()
    for (a0, b0), (a1, _) in zip(ranges, ranges[1:]):
        if a1 < b0:
            raise ValueError(f"VLLM_DRAFT_LMH_VOCAB: overlapping ranges {ranges}")
    return ranges


def _mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 values + E8M0 block-32 scales in the 128x4 swizzled layout.

    The F8_128x4 swizzle (vLLM ``swizzle_mxfp8_scale``) equals cuBLAS's
    block-scaling-factor layout for MXFP8.
    """
    from flashinfer import mxfp8_quantize

    q, s = mxfp8_quantize(
        x, is_sf_swizzled_layout=True, alignment=32, backend="cute-dsl"
    )
    return q, s.view(torch.float8_e8m0fnu)


_CHUNK = 4096


@triton.jit
def _subset_argmax_stage1(
    X, stride, VAL, IDX, V: tl.constexpr, CHUNK: tl.constexpr, NCH_PAD: tl.constexpr
):
    # Per (row, chunk): max and the lowest column index attaining it.
    row = tl.program_id(0)
    c = tl.program_id(1)
    offs = c * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(
        X + row.to(tl.int64) * stride + offs, mask=offs < V, other=-float("inf")
    ).to(tl.float32)
    m = tl.max(x, axis=0)
    i = tl.min(tl.where(x == m, offs, 2147483647), axis=0)
    tl.store(VAL + row * NCH_PAD + c, m)
    tl.store(IDX + row * NCH_PAD + c, i)


@triton.jit
def _subset_argmax_stage2(
    VAL, IDX, IDS, OUT, NCH: tl.constexpr, NCH_PAD: tl.constexpr
):
    # Per row: lowest column index of the row max, mapped to its token id.
    row = tl.program_id(0)
    j = tl.arange(0, NCH_PAD)
    v = tl.load(VAL + row * NCH_PAD + j, mask=j < NCH, other=-float("inf"))
    i = tl.load(IDX + row * NCH_PAD + j, mask=j < NCH, other=2147483647)
    m = tl.max(v, axis=0)
    best = tl.min(tl.where(v == m, i, 2147483647), axis=0)
    tl.store(OUT + row, tl.load(IDS + best))


def subset_argmax(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """ids[argmax(logits, -1)] for 2D CUDA logits (lowest index wins ties)."""
    if not logits.is_cuda or logits.ndim != 2 or logits.stride(1) != 1:
        return ids[logits.argmax(dim=-1)]
    n, v = logits.shape
    out = torch.empty(n, dtype=torch.int64, device=logits.device)
    if n == 0:
        return out
    nch = triton.cdiv(v, _CHUNK)
    nch_pad = triton.next_power_of_2(nch)
    val = torch.empty((n, nch_pad), dtype=torch.float32, device=logits.device)
    idx = torch.empty((n, nch_pad), dtype=torch.int32, device=logits.device)
    _subset_argmax_stage1[(n, nch)](
        logits, logits.stride(0), val, idx, V=v, CHUNK=_CHUNK, NCH_PAD=nch_pad,
        num_warps=4,
    )
    _subset_argmax_stage2[(n,)](
        val, idx, ids, out, NCH=nch, NCH_PAD=nch_pad, num_warps=1
    )
    return out


class DraftVocabHead(nn.Module):
    """Greedy draft token = ids[argmax(h @ W[ids].T)]."""

    def __init__(
        self,
        lm_head_weight: torch.Tensor,
        ranges: list[tuple[int, int]],
        dtype: str,
    ) -> None:
        super().__init__()
        device = lm_head_weight.device
        vocab, hidden = lm_head_weight.shape
        ids = torch.cat([torch.arange(a, b, dtype=torch.int64) for a, b in ranges])
        self.ranges = ranges
        self.num_ids = int(ids.numel())
        # Pad the row count to a multiple of 128 (cuBLAS block-scaled GEMMs
        # need N % 16 == 0; 128 keeps whole scale tiles) by repeating the last
        # id: a duplicated row has the same logit, and argmax keeps the first.
        pad = -self.num_ids % 128
        if pad:
            ids = torch.cat([ids, ids[-1:].expand(pad)])
        self.vocab_size = int(vocab)
        self.dtype_name = dtype
        self.register_buffer("ids", ids.to(device), persistent=False)
        w = lm_head_weight.index_select(0, self.ids).contiguous()
        if dtype == "bf16":
            self.register_buffer("weight", w, persistent=False)
            self.weight_q = None
            self.weight_sf = None
        elif dtype == "mxfp8":
            if hidden % 128:
                raise ValueError(f"mxfp8 draft head needs hidden % 128 == 0 ({hidden})")
            wq, wsf = _mxfp8_quantize(w)
            del w
            self.weight = None
            self.register_buffer("weight_q", wq, persistent=False)
            self.register_buffer("weight_sf", wsf, persistent=False)
        else:
            raise ValueError(f"VLLM_DRAFT_LMH_DTYPE={dtype!r} (want bf16 or mxfp8)")
        self.eager_calls = 0
        self.captured_calls = 0

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            return F.linear(hidden_states, self.weight)
        q, s = _mxfp8_quantize(hidden_states.contiguous())
        return torch._scaled_mm(
            q,
            self.weight_q.t(),
            scale_a=s,
            scale_b=self.weight_sf,
            out_dtype=hidden_states.dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
            self.captured_calls += 1
        else:
            self.eager_calls += 1
        # The lowest index among equal maxima wins and ids are ascending, so
        # ties resolve as in the full-vocabulary argmax.
        return subset_argmax(self.logits(hidden_states), self.ids)


class _Diag:
    """Device-side diagnostics (VLLM_DRAFT_LMH_DIAG=1; not for scored runs).

    counts = [draft rows, rows whose draft differs from the full argmax,
    rows whose full argmax is outside the subset, target-sampled tokens
    observed, target-sampled tokens outside the subset]. Draft rows include
    CUDA-graph padding rows and rows of still-prefilling requests (discarded
    drafts); the target-token counts are free of both. Histograms of the
    outside ids (draft full argmax / target samples) name what the subset
    misses.
    """

    def __init__(self, device: torch.device, vocab_size: int) -> None:
        self.counts = torch.zeros(5, dtype=torch.int64, device=device)
        self.hist_draft = torch.zeros(vocab_size, dtype=torch.int32, device=device)
        self.hist_target = torch.zeros(vocab_size, dtype=torch.int32, device=device)
        self.t_last = time.monotonic()
        self.last = [0] * 5

    @staticmethod
    def _outside(head: DraftVocabHead, tok: torch.Tensor) -> torch.Tensor:
        outside = torch.ones_like(tok, dtype=torch.bool)
        for a, b in head.ranges:
            outside &= (tok < a) | (tok >= b)
        return outside

    def observe(
        self, head: DraftVocabHead, full_logits: torch.Tensor, sub_tokens: torch.Tensor
    ) -> None:
        full = full_logits.argmax(dim=-1)
        outside = self._outside(head, full)
        self.counts[0] += full.numel()
        self.counts[1] += (full != sub_tokens).sum()
        self.counts[2] += outside.sum()
        self.hist_draft.index_add_(0, full, outside.to(torch.int32))

    def observe_target(
        self,
        head: DraftVocabHead,
        last_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
    ) -> None:
        # last_sampled is [max_num_reqs] or [max_num_reqs, 1] (one token per
        # request state); index it flat like the prefill-input kernel does.
        tok = last_sampled.reshape(-1)[idx_mapping.long()].long()
        valid = num_sampled > 0
        outside = self._outside(head, tok) & valid
        self.counts[3] += valid.sum()
        self.counts[4] += outside.sum()
        self.hist_target.index_add_(0, tok, outside.to(torch.int32))

    def maybe_log(self, head: DraftVocabHead, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.t_last < _DIAG_S:
            return
        self.t_last = now
        c = self.counts.tolist()
        d = [x - y for x, y in zip(c, self.last)]
        self.last = c

        def top(h):
            v, i = h.topk(12)
            return [(int(x), int(y)) for x, y in zip(i.tolist(), v.tolist()) if y > 0]

        logger.info(
            "[draft-lmh] diag: window draft rows=%d differ=%d (%.4f%%) | target "
            "tokens=%d outside=%d (%.4f%%) | total draft rows=%d differ=%d "
            "(%.4f%%) target tokens=%d outside=%d (%.4f%%) | eager calls=%d "
            "captured=%d | top outside draft ids %s | top outside target ids %s",
            d[0],
            d[1],
            100.0 * d[1] / max(1, d[0]),
            d[3],
            d[4],
            100.0 * d[4] / max(1, d[3]),
            c[0],
            c[1],
            100.0 * c[1] / max(1, c[0]),
            c[3],
            c[4],
            100.0 * c[4] / max(1, c[3]),
            head.eager_calls,
            head.captured_calls,
            top(self.hist_draft),
            top(self.hist_target),
        )


def maybe_build(draft_model: nn.Module) -> DraftVocabHead | None:
    """Build the subset head from the draft's (BF16, unsharded) lm_head."""
    if not ENABLED:
        return None
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    head = getattr(draft_model, "lm_head", None)
    lp = getattr(draft_model, "logits_processor", None)
    if head is None or not hasattr(head, "weight"):
        raise RuntimeError("[draft-lmh] draft model has no lm_head weight")
    if get_tensor_model_parallel_world_size() != 1:
        raise RuntimeError("[draft-lmh] only tensor-parallel size 1 is supported")
    if type(getattr(head, "quant_method", None)) is not UnquantizedEmbeddingMethod:
        raise RuntimeError(
            f"[draft-lmh] lm_head method {type(head.quant_method).__name__} "
            "is not the unquantized embedding method"
        )
    if lp is not None and (
        getattr(lp, "logits_as_input", False) or not getattr(lp, "scale", 1.0) > 0
    ):
        raise RuntimeError("[draft-lmh] unsupported logits processor settings")
    weight = head.weight.data
    vocab = getattr(lp, "org_vocab_size", None) or weight.shape[0]
    ranges = parse_ranges(_SPEC, min(int(vocab), weight.shape[0]))
    free0 = torch.cuda.mem_get_info()[0]
    m = DraftVocabHead(weight, ranges, _DTYPE)
    torch.cuda.synchronize()
    free1 = torch.cuda.mem_get_info()[0]
    logger.info(
        "[draft-lmh] installed: %d of %d token ids (%.1f%%) in ranges %s, %s; "
        "draft lm_head bytes/step %.3f GB -> %.3f GB; GPU memory %+.3f GiB; diag=%s",
        m.num_ids,
        weight.shape[0],
        100.0 * m.num_ids / weight.shape[0],
        ranges,
        _DTYPE,
        weight.numel() * weight.element_size() / 1e9,
        (m.num_ids * weight.shape[1] * (2 if _DTYPE == "bf16" else 1 + 1 / 32)) / 1e9,
        (free1 - free0) / 2**30,
        _DIAG,
    )
    if _DIAG:
        m.diag = _Diag(weight.device, weight.shape[0])
    else:
        m.diag = None
    return m


def sample(
    head: DraftVocabHead, draft_model: nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    tokens = head(hidden_states)
    if head.diag is not None:
        head.diag.observe(head, draft_model.compute_logits(hidden_states), tokens)
    return tokens


def observe_target(
    head: DraftVocabHead | None,
    last_sampled: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
) -> None:
    """Diagnostics only: count target-sampled tokens outside the subset."""
    if head is not None and head.diag is not None:
        head.diag.observe_target(head, last_sampled, idx_mapping, num_sampled)


def maybe_log(head: DraftVocabHead | None) -> None:
    if head is None:
        return
    if head.diag is not None:
        if not torch.cuda.is_current_stream_capturing():
            head.diag.maybe_log(head)
        return
    now = time.monotonic()
    t_last = getattr(head, "_t_last", None)
    if t_last is None:
        head._t_last = now
        return
    if now - t_last >= 600:
        head._t_last = now
        logger.info(
            "[draft-lmh] calls: eager=%d captured=%d (%s, %d ids)",
            head.eager_calls,
            head.captured_calls,
            head.dtype_name,
            head.num_ids,
        )
