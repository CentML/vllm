# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

from vllm.v1.worker.gpu.sample.spec_topk_topp import apply_spec_top_k_top_p
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEVICE = "cuda"


def _reference_mask(
    logits: torch.Tensor, top_k: list[int], top_p: list[float]
) -> tuple[torch.Tensor, list[bool]]:
    """Exact top-k by (value desc, id asc), then the shortest prefix whose
    renormalized mass reaches p. Also flags rows whose cumulative mass lands
    within 1e-6 of p, where fp32 and fp64 may legitimately disagree.
    """
    out = torch.full_like(logits, float("-inf"))
    ambiguous = []
    x = logits.double()
    for row in range(logits.shape[0]):
        order = torch.argsort(-x[row], stable=True)
        order = order[torch.isfinite(x[row, order])][: top_k[row]]
        amb = False
        if top_p[row] < 1.0 and order.numel() > 0:
            cum = torch.cumsum(torch.softmax(x[row, order], 0), 0)
            amb = bool(((cum - top_p[row]).abs() < 1e-6).any())
            order = order[: 1 + int((cum[:-1] < top_p[row]).sum())]
        out[row, order] = logits[row, order]
        ambiguous.append(amb)
    return out, ambiguous


def _run(logits, top_k, top_p, rows_per_req=1):
    num_reqs = logits.shape[0] // rows_per_req
    expanded = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    expanded = expanded.repeat_interleave(rows_per_req)
    k_state = torch.tensor(top_k, dtype=torch.int32, device=DEVICE)
    p_state = torch.tensor(top_p, dtype=torch.float32, device=DEVICE)
    out = apply_spec_top_k_top_p(
        logits.clone(),
        expanded,
        k_state,
        p_state,
        max(top_k),
        any(p != 1.0 for p in top_p),
    )
    per_row_k = [top_k[r // rows_per_req] for r in range(logits.shape[0])]
    per_row_p = [top_p[r // rows_per_req] for r in range(logits.shape[0])]
    ref, ambiguous = _reference_mask(logits, per_row_k, per_row_p)
    mismatch = (out != ref).any(dim=1).tolist()
    bad = [r for r, (m, a) in enumerate(zip(mismatch, ambiguous)) if m and not a]
    assert not bad, f"rows differing from the exact reference: {bad}"


@pytest.mark.parametrize("vocab_size", [248320, 50000, 3000])
def test_matches_exact_reference_bf16_logits(vocab_size: int):
    torch.manual_seed(0)
    num_rows = 24
    logits = (torch.randn(num_rows, vocab_size, device=DEVICE) * 3).bfloat16().float()
    # Presence-penalty-like shifts create extra exact ties on the bf16 grid.
    logits[:, ::7] -= 1.5
    top_k = torch.randint(1, 65, (num_rows,)).tolist()
    top_p = [0.95, 1.0, 0.5, 0.99, 0.1, 0.8] * (num_rows // 6)
    _run(logits, top_k, top_p)


def test_ties_keep_lowest_token_ids():
    torch.manual_seed(1)
    # Coarse integer logits: many tokens tie at the k-th value and at the
    # top-p boundary; exactly k survive and ties go to the lowest ids.
    logits = torch.randint(-40, 20, (8, 30000), device=DEVICE).float() * 0.5
    _run(logits, [20] * 8, [0.95, 1.0, 0.7, 0.3, 0.95, 1.0, 0.99, 0.5])


def test_rows_with_few_finite_logits():
    torch.manual_seed(2)
    logits = (torch.randn(6, 40000, device=DEVICE) * 2).bfloat16().float()
    keep = torch.rand(6, 40000, device=DEVICE) < torch.tensor(
        [[0.0], [5e-5], [2e-4], [1e-3], [0.5], [1.0]], device=DEVICE
    )
    logits[~keep] = float("-inf")
    _run(logits, [20, 20, 20, 5, 64, 20], [0.95, 0.95, 0.9, 1.0, 0.95, 0.9])


def test_candidate_overflow_fallback():
    # Constant, heavily tied or clustered rows put far more than the candidate
    # buffer above the lower bound and take the full-row fallback.
    vocab_size = 60000
    logits = torch.zeros(5, vocab_size, device=DEVICE)
    logits[1] = 1.0
    logits[2] = torch.randn(vocab_size, device=DEVICE).bfloat16().float() - 20
    logits[2, 10000:20000] = 5.0 + torch.randn(10000, device=DEVICE) * 0.01
    logits[3, 5000:13000] = 3.0
    logits[4] = torch.randint(0, 2, (vocab_size,), device=DEVICE).float()
    _run(logits, [20, 5, 20, 64, 33], [0.95, 0.5, 0.9, 1.0, 0.99])


def test_expanded_rows_use_request_params():
    torch.manual_seed(3)
    logits = (torch.randn(40, 30000, device=DEVICE) * 3).bfloat16().float()
    top_k = [20, 1, 50, 20, 7, 20, 20, 64, 3, 20]
    top_p = [0.95, 0.9, 0.8, 1.0, 0.95, 0.1, 0.99, 0.95, 0.95, 0.95]
    _run(logits, top_k, top_p, rows_per_req=4)


@pytest.mark.parametrize("draft_is_top", [True, False])
def test_rejection_output_follows_masked_target(draft_is_top: bool):
    """With a greedy (one-hot) draft, the first emitted token of the rejection
    sampler must follow the top-k/top-p-masked target distribution.
    """
    torch.manual_seed(4)
    vocab_size, top_k, top_p, num_trials, steps = 2000, 20, 0.9, 30000, 3
    target_1d = (torch.randn(vocab_size, device=DEVICE) * 2).bfloat16().float()
    num_logits = num_trials * (steps + 1)
    logits = target_1d.expand(num_logits, -1).contiguous()
    expanded = torch.arange(num_trials, dtype=torch.int32, device=DEVICE)
    expanded = expanded.repeat_interleave(steps + 1)
    k_state = torch.full((num_trials,), top_k, dtype=torch.int32, device=DEVICE)
    p_state = torch.full((num_trials,), top_p, dtype=torch.float32, device=DEVICE)
    masked = apply_spec_top_k_top_p(logits, expanded, k_state, p_state, top_k, True)

    ref, _ = _reference_mask(target_1d[None], [top_k], [top_p])
    target_probs = torch.softmax(ref[0].double(), 0).float()
    survivors = torch.nonzero(target_probs).flatten()
    draft = survivors[0] if draft_is_top else survivors[-1]
    draft_sampled = torch.zeros(num_logits, dtype=torch.int64, device=DEVICE)
    draft_sampled.view(num_trials, steps + 1)[:, 1:] = draft
    sampled, _ = rejection_sample(
        masked,
        None,
        draft_sampled,
        torch.arange(num_trials + 1, dtype=torch.int32, device=DEVICE) * (steps + 1),
        torch.arange(num_logits, dtype=torch.int64, device=DEVICE),
        torch.arange(num_trials, dtype=torch.int32, device=DEVICE),
        expanded,
        torch.arange(steps + 1, dtype=torch.int32, device=DEVICE).repeat(num_trials),
        torch.ones(num_trials, dtype=torch.float32, device=DEVICE),
        torch.arange(num_trials, dtype=torch.int64, device=DEVICE),
        steps,
    )
    first = sampled[:, 0]
    assert bool(torch.isin(first, survivors).all()), "emitted a masked token"
    observed = torch.bincount(first, minlength=vocab_size).float()
    expected = target_probs * num_trials
    keep = expected >= 5
    chi2 = (((observed - expected) ** 2) / expected.clamp_min(1e-9))[keep].sum()
    dof = int(keep.sum()) - 1
    assert float(chi2) < dof + 10 * math.sqrt(2 * dof)
