# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact rejection, fused preparation and sparse verification semantics."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.penalties import apply_penalties
from vllm.v1.worker.gpu.sample.spec_topk_topp import (
    apply_spec_top_k_top_p,
    prepare_spec_top_k_top_p,
)
from vllm.v1.worker.gpu.spec_decode.fused_rejection import (
    fused_rejection_sample,
    select_survivors,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEVICE = "cuda"
STEPS = 3


@dataclass
class Batch:
    logits: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    logits_indices: torch.Tensor
    cu_num_logits: torch.Tensor
    idx_mapping: torch.Tensor
    seq_lens: torch.Tensor
    expanded_idx_mapping: torch.Tensor
    expanded_local_pos: torch.Tensor
    prefill_len: torch.Tensor
    temperature: torch.Tensor
    seeds: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    penalties: tuple[torch.Tensor, ...] | None
    max_top_k: int
    use_top_p: bool


def _make_batch(
    logits: torch.Tensor,
    rows_per_req: list[int],
    *,
    top_k: list[int],
    top_p: list[float],
    temperature: list[float],
    penalties: bool,
    seed: int,
    chunked: list[bool] | None = None,
) -> Batch:
    """Request r owns rows_per_req[r] consecutive logits rows. Each request also
    has a few non-sampled tokens in front, so logits_indices is not the
    identity, and its persistent state lives at a permuted slot.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_reqs = len(rows_per_req)
    num_logits, vocab_size = logits.shape
    assert sum(rows_per_req) == num_logits
    max_num_reqs = num_reqs + 3
    slots = torch.randperm(max_num_reqs, generator=g)[:num_reqs]

    input_ids: list[int] = []
    positions: list[int] = []
    logits_indices: list[int] = []
    cu = [0]
    for r, n in enumerate(rows_per_req):
        pre = int(torch.randint(0, 3, (1,), generator=g))
        base = len(input_ids)
        start_pos = int(torch.randint(10, 5000, (1,), generator=g))
        row0 = cu[-1]
        for t in range(pre + n):
            if t < pre:
                tok = int(torch.randint(0, vocab_size, (1,), generator=g))
            else:
                j = t - pre
                kind = int(torch.randint(0, 5, (1,), generator=g))
                if j == 0 or kind == 0:
                    tok = int(torch.randint(0, vocab_size, (1,), generator=g))
                elif kind == 1:
                    tok = -1  # placeholder draft
                else:
                    # A likely survivor of the previous row, so that drafts are
                    # accepted often enough to reach every step and the bonus.
                    prev_row = logits[row0 + j - 1].float()
                    order = torch.argsort(prev_row, descending=True, stable=True)
                    tok = int(order[min(kind - 2, 2)])
                logits_indices.append(base + t)
            input_ids.append(tok)
            positions.append(start_pos + t)
        cu.append(cu[-1] + n)

    def state(vals, dtype, fill):
        t = torch.full((max_num_reqs,), fill, dtype=dtype)
        t[slots] = torch.tensor(vals, dtype=dtype)
        return t.to(DEVICE)

    expanded = torch.repeat_interleave(slots, torch.tensor(rows_per_req))
    local_pos = torch.cat([torch.arange(n) for n in rows_per_req])
    chunked = chunked or [False] * num_reqs
    prefill = state([1000 if c else 0 for c in chunked], torch.int32, 0)
    seq_lens = torch.tensor([10] * num_reqs, dtype=torch.int32, device=DEVICE)

    pen = None
    if penalties:
        rep = state(
            [[1.0, 1.3, 0.8][r % 3] for r in range(num_reqs)], torch.float32, 1.0
        )
        freq = state(
            [[0.0, 0.0, 0.4, -0.2][r % 4] for r in range(num_reqs)],
            torch.float32,
            0.0,
        )
        pres = state(
            [[1.5, 0.0, 1.5, 2.0, -1.0][r % 5] for r in range(num_reqs)],
            torch.float32,
            0.0,
        )
        prompt_bin_mask = torch.randint(
            -(2**31), 2**31 - 1, (max_num_reqs, (vocab_size + 31) // 32),
            dtype=torch.int32, generator=g,
        ).to(DEVICE)  # fmt: skip
        counts = torch.randint(0, 4, (max_num_reqs, vocab_size), generator=g)
        counts *= torch.rand(max_num_reqs, vocab_size, generator=g) < 0.3
        pen = (rep, freq, pres, prompt_bin_mask, counts.int().to(DEVICE))

    return Batch(
        logits=logits,
        input_ids=torch.tensor(input_ids, dtype=torch.int32, device=DEVICE),
        positions=torch.tensor(positions, dtype=torch.int64, device=DEVICE),
        logits_indices=torch.tensor(logits_indices, dtype=torch.int32, device=DEVICE),
        cu_num_logits=torch.tensor(cu, dtype=torch.int32, device=DEVICE),
        idx_mapping=slots.int().to(DEVICE),
        seq_lens=seq_lens,
        expanded_idx_mapping=expanded.int().to(DEVICE),
        expanded_local_pos=local_pos.int().to(DEVICE),
        prefill_len=prefill,
        temperature=state(temperature, torch.float32, 1.0),
        seeds=state(
            torch.randint(-(2**62), 2**62, (num_reqs,), generator=g).tolist(),
            torch.int64,
            0,
        ),
        top_k=state(top_k, torch.int32, vocab_size),
        top_p=state(top_p, torch.float32, 1.0),
        penalties=pen,
        max_top_k=max(top_k),
        use_top_p=any(p != 1.0 for p in top_p),
    )


def _unfused_canvas(b: Batch) -> torch.Tensor:
    """The masked fp32 canvas of RejectionSampler's unfused path."""
    canvas = b.logits.to(torch.float32, copy=True)
    draft_sampled = b.input_ids[b.logits_indices]
    if b.penalties is not None:
        apply_penalties(
            canvas, b.expanded_idx_mapping, draft_sampled, b.expanded_local_pos,
            *b.penalties,
        )  # fmt: skip
    apply_temperature(canvas, b.expanded_idx_mapping, b.temperature)
    return apply_spec_top_k_top_p(
        canvas, b.expanded_idx_mapping, b.top_k, b.top_p, b.max_top_k, b.use_top_p
    )


def _survivors(b: Batch):
    return select_survivors(
        b.logits, b.input_ids, b.logits_indices, b.cu_num_logits, b.idx_mapping,
        b.expanded_idx_mapping, b.expanded_local_pos, b.temperature, b.top_k,
        b.top_p, b.penalties, b.max_top_k, b.use_top_p,
    )  # fmt: skip


def _check_survivors(b: Batch) -> None:
    ref = _unfused_canvas(b)
    val, idx, num, lse = _survivors(b)
    canvas = torch.full_like(ref, float("-inf"))
    for row in range(ref.shape[0]):
        n = int(num[row])
        ids = idx[row, :n].long()
        assert ids.unique().numel() == n
        canvas[row, ids] = val[row, :n]
        # Survivor order: value desc, token id asc.
        v = val[row, :n].double()
        assert bool(
            ((v[:-1] > v[1:]) | ((v[:-1] == v[1:]) & (ids[:-1] < ids[1:]))).all()
        )
    bad = (canvas.view(torch.int32) != ref.view(torch.int32)).any(dim=1)
    assert not bool(bad.any()), f"rows differ from the unfused canvas: {bad.nonzero()}"
    ref_lse = torch.logsumexp(ref.double(), dim=1)
    finite = num > 0
    torch.testing.assert_close(
        lse[finite].double(), ref_lse[finite], rtol=1e-6, atol=1e-5
    )


def _rows(num_reqs: int, rows: int = STEPS + 1) -> list[int]:
    return [rows] * num_reqs


def _request_logits(rows_per_req: list[int], vocab_size: int, scale: float):
    """bf16 logits whose rows within a request are close, as for consecutive
    positions: the drafts (top tokens of the previous row) are then also top
    tokens of the later rows, where the draft-prefix penalties hit them.
    """
    base = torch.randn(len(rows_per_req), vocab_size, device=DEVICE) * scale
    base = base.repeat_interleave(torch.tensor(rows_per_req, device=DEVICE), dim=0)
    return (base + 0.3 * torch.randn_like(base)).bfloat16()


@pytest.mark.parametrize("vocab_size", [248320, 50000, 3000])
@pytest.mark.parametrize("penalties", [False, True])
def test_survivors_bitwise_unfused(vocab_size: int, penalties: bool):
    torch.manual_seed(0)
    num_reqs = 12
    logits = _request_logits(_rows(num_reqs), vocab_size, 3.0)
    b = _make_batch(
        logits,
        _rows(num_reqs),
        top_k=torch.randint(1, 65, (num_reqs,)).tolist(),
        top_p=[0.95, 1.0, 0.5, 0.99, 0.1, 0.8] * 2,
        temperature=[1.0, 0.7, 0.0, 1.3] * 3,
        penalties=penalties,
        seed=vocab_size,
    )
    _check_survivors(b)


def test_survivors_adversarial_ties():
    """Ties at the k-th value, across sub-block maxima and at the top-p
    boundary; constant rows; rows with fewer finite logits than top_k.
    """
    torch.manual_seed(1)
    vocab_size = 60000
    rows = []
    rows.append(torch.zeros(vocab_size))  # constant
    rows.append(torch.full((vocab_size,), 1.0))
    coarse = torch.randint(-40, 20, (vocab_size,)).float() * 0.5
    rows.append(coarse)  # dense ties everywhere
    blocky = torch.randn(vocab_size).bfloat16().float() - 20
    blocky[::128] = 3.0  # every sub-block max ties
    rows.append(blocky)
    cluster = torch.randn(vocab_size) * 0.01
    cluster[1000:1300] = 5.0  # many ties inside a few sub-blocks
    rows.append(cluster)
    few = torch.full((vocab_size,), float("-inf"))
    few[torch.randperm(vocab_size)[:7]] = torch.randn(7)
    rows.append(few)  # 7 finite < top_k
    none = torch.full((vocab_size,), float("-inf"))
    rows.append(none)  # nothing finite
    binary = torch.randint(0, 2, (vocab_size,)).float()
    rows.append(binary)
    logits = torch.stack(rows).to(DEVICE)
    n = logits.shape[0]
    b = _make_batch(
        logits,
        [1] * n,
        top_k=[20, 5, 20, 64, 33, 20, 20, 1],
        top_p=[0.95, 0.5, 0.9, 1.0, 0.99, 0.95, 0.9, 0.3],
        temperature=[1.0] * n,
        penalties=False,
        seed=1,
    )
    _check_survivors(b)


def _unfused_rejection(b: Batch):
    canvas = _unfused_canvas(b)
    num_reqs = b.cu_num_logits.shape[0] - 1
    num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=DEVICE)
    sampled, num_sampled = rejection_sample(
        canvas,
        None,
        b.input_ids[b.logits_indices].long(),
        b.cu_num_logits,
        b.positions[b.logits_indices],
        b.idx_mapping,
        b.expanded_idx_mapping,
        b.expanded_local_pos,
        b.temperature,
        b.seeds,
        STEPS,
        num_rejected=num_rejected,
        seq_lens=b.seq_lens,
        prefill_len=b.prefill_len,
    )
    return sampled, num_sampled, num_rejected


def _fused_rejection(b: Batch):
    return fused_rejection_sample(
        b.logits, b.input_ids, b.positions, b.logits_indices, b.cu_num_logits,
        b.idx_mapping, b.seq_lens, b.expanded_idx_mapping, b.expanded_local_pos,
        b.prefill_len, b.temperature, b.seeds, b.top_k, b.top_p, b.penalties,
        b.max_top_k, b.use_top_p, STEPS,
    )  # fmt: skip


@pytest.mark.parametrize("penalties", [False, True])
def test_rejection_matches_unfused(penalties: bool):
    torch.manual_seed(2)
    vocab_size = 248320
    # Decode requests (4 rows), prefill/bonus-only requests (1 row), partial
    # drafts (2 rows), one chunked-prefill request; greedy and sampled rows.
    rows_per_req = [4, 4, 1, 4, 2, 4, 1, 4] * 6
    num_reqs = len(rows_per_req)
    logits = _request_logits(rows_per_req, vocab_size, 2.5)
    chunked = [r == 2 for r in range(num_reqs)]
    totals = {"accepted": 0, "steps": 0, "bonus": 0}
    for trial in range(4):
        b = _make_batch(
            logits,
            rows_per_req,
            top_k=[20, 20, 1, 50, 20, 7, 64, 20] * 6,
            top_p=[0.95, 0.95, 0.9, 0.8, 1.0, 0.95, 0.5, 0.95] * 6,
            temperature=[1.0, 1.0, 1.0, 0.0, 0.6, 1.0, 1.0, 1.0] * 6,
            penalties=penalties,
            seed=100 + trial,
            chunked=chunked,
        )
        ref_sampled, ref_num_sampled, ref_num_rejected = _unfused_rejection(b)
        sampled, num_sampled, num_rejected = _fused_rejection(b)
        assert torch.equal(num_sampled, ref_num_sampled)
        assert torch.equal(num_rejected, ref_num_rejected)
        for r in range(num_reqs):
            n = max(int(num_sampled[r]), 1)
            assert torch.equal(sampled[r, :n], ref_sampled[r, :n]), r
            totals["accepted"] += n - 1
            totals["steps"] += rows_per_req[r] - 1
            totals["bonus"] += int(n == rows_per_req[r] and not chunked[r])
    # The batch must exercise acceptance, rejection and the bonus token.
    assert 0 < totals["accepted"] < totals["steps"]
    assert totals["bonus"] > 0


def _sampler_stub(num_reqs: int, vocab_size: int = 1000):
    from types import SimpleNamespace

    import numpy as np

    from vllm.v1.worker.gpu.sample.sampler import Sampler

    def arr(fill, dtype):
        return SimpleNamespace(np=np.full(num_reqs, fill, dtype=dtype))

    s = object.__new__(Sampler)
    s.sampling_states = SimpleNamespace(
        top_k=arr(20, np.int32), top_p=arr(0.95, np.float32), min_p=arr(0, np.float32)
    )
    s.logit_bias_state = SimpleNamespace(use_logit_bias=np.zeros(num_reqs, bool))
    s.bad_words_state = SimpleNamespace(num_bad_words=arr(0, np.int32))
    s.thinking_budget_state = SimpleNamespace(
        enabled=False, use_thinking_budget=np.zeros(num_reqs, bool)
    )
    s.penalties_state = SimpleNamespace(use_penalty=np.zeros(num_reqs, bool))
    return s, vocab_size


def test_fused_dispatch_predicate():
    import numpy as np

    batch = np.array([0, 2])
    s, vocab_size = _sampler_stub(4)
    s.sampling_states.top_p.np[:] = 1.0
    assert s.fused_spec_sampling_params(batch) == (20, False, False)
    s.penalties_state.use_penalty[2] = True
    s.sampling_states.top_p.np[0] = 0.9
    s.sampling_states.top_k.np[2] = 64
    assert s.fused_spec_sampling_params(batch) == (64, True, True)
    # Requests outside the batch do not matter.
    s.sampling_states.top_k.np[1] = vocab_size
    s.sampling_states.min_p.np[3] = 0.1
    assert s.fused_spec_sampling_params(batch) is not None
    # Anything the fused kernels do not implement keeps the unfused path.
    for case in ("no_top_k", "top_k_65", "min_p", "logit_bias", "bad_words", "think"):
        s, _ = _sampler_stub(4)
        if case == "no_top_k":
            s.sampling_states.top_k.np[2] = vocab_size
        elif case == "top_k_65":
            s.sampling_states.top_k.np[0] = 65
        elif case == "min_p":
            s.sampling_states.min_p.np[0] = 0.05
        elif case == "logit_bias":
            s.logit_bias_state.use_logit_bias[2] = True
        elif case == "bad_words":
            s.bad_words_state.num_bad_words.np[0] = 1
        else:
            s.thinking_budget_state.enabled = True
            s.thinking_budget_state.use_thinking_budget[2] = True
        assert s.fused_spec_sampling_params(batch) is None, case


@pytest.mark.parametrize("penalties", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_preparation_canvas_matches_separate_passes(penalties: bool, tied: bool):
    torch.manual_seed(17)
    rows = [10, 4, 1, 2]
    logits = _request_logits(rows, 50003, 2.5)
    if tied:
        logits = (logits.float() * 2).round().bfloat16()
    b = _make_batch(
        logits,
        rows,
        top_k=[64, 7, 20, 1],
        top_p=[0.95, 0.7, 1.0, 0.8],
        temperature=[1.0, 0.0, 0.7, 1.3],
        penalties=penalties,
        seed=19,
    )
    original = logits.clone()
    prepared = prepare_spec_top_k_top_p(
        logits,
        b.expanded_idx_mapping,
        b.input_ids[b.logits_indices],
        b.expanded_local_pos,
        b.temperature,
        b.top_k,
        b.top_p,
        b.penalties,
        b.max_top_k,
        b.use_top_p,
    )
    ref = _unfused_canvas(b)
    assert torch.equal(prepared.view(torch.int32), ref.view(torch.int32))
    assert torch.equal(logits, original)


def _runtime_sampler(b: Batch, logprobs_mode="raw_logprobs", watermark=False):
    """Real sampler state with the same permuted persistent slots as Batch."""
    from types import SimpleNamespace

    from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    req = RequestState(
        b.temperature.numel(),
        64,
        b.input_ids.numel(),
        STEPS,
        b.logits.shape[1],
        torch.device(DEVICE),
    )
    kwargs = dict(
        max_num_reqs=b.temperature.numel(),
        vocab_size=b.logits.shape[1],
        device=torch.device(DEVICE),
        req_states=req,
        logprobs_mode=logprobs_mode,
        num_speculative_tokens=STEPS,
    )
    if watermark:
        sampler = GPUWatermarkSampler(
            SimpleNamespace(context_width=2), deduplicate_contexts="none", **kwargs
        )
    else:
        sampler = Sampler(**kwargs)
    states = sampler.sampling_states
    for name in ("temperature", "top_k", "top_p", "seeds"):
        getattr(states, name).np[:] = getattr(b, name).cpu().numpy()
    states.min_p.np.fill(0)
    states.params.sync()
    req.prompt_len.np.fill(0)
    req.prefill_len.np[:] = b.prefill_len.cpu().numpy()
    req.lens.sync()
    req.total_len.gpu.zero_()
    req.all_token_ids.gpu.zero_()
    sampler.needs_logits_processing.fill(True)
    if b.penalties is not None:
        penalties = sampler.penalties_state
        for name, value in zip(
            ("repetition_penalty", "frequency_penalty", "presence_penalty"),
            b.penalties[:3],
        ):
            getattr(penalties, name).np[:] = value.cpu().numpy()
        penalties.params.sync()
        penalties.use_penalty[:] = (
            ((b.penalties[0] != 1) | (b.penalties[1] != 0) | (b.penalties[2] != 0))
            .cpu()
            .numpy()
        )
        penalties.prompt_bin_mask.copy_(b.penalties[3])
        penalties.output_bin_counts.copy_(b.penalties[4])
    return sampler


def _input_batch(b: Batch):
    from types import SimpleNamespace

    return SimpleNamespace(
        input_ids=b.input_ids,
        positions=b.positions,
        logits_indices=b.logits_indices,
        num_reqs=b.idx_mapping.numel(),
        idx_mapping=b.idx_mapping,
        idx_mapping_np=b.idx_mapping.cpu().numpy(),
        cu_num_logits=b.cu_num_logits,
        cu_num_logits_np=b.cu_num_logits.cpu().numpy(),
        seq_lens=b.seq_lens,
        expanded_idx_mapping=b.expanded_idx_mapping,
        expanded_local_pos=b.expanded_local_pos,
    )


def _assert_outputs_equal(actual, expected):
    assert torch.equal(actual.num_sampled, expected.num_sampled)
    assert torch.equal(actual.num_rejected, expected.num_rejected)
    for row, count in enumerate(actual.num_sampled.tolist()):
        assert torch.equal(
            actual.sampled_token_ids[row, :count],
            expected.sampled_token_ids[row, :count],
        )
    actual_lp, expected_lp = actual.logprobs_tensors, expected.logprobs_tensors
    assert (actual_lp is None) == (expected_lp is None)
    if actual_lp is not None:
        assert torch.equal(actual_lp.logprob_token_ids, expected_lp.logprob_token_ids)
        assert torch.equal(actual_lp.logprobs, expected_lp.logprobs)
        assert torch.equal(
            actual_lp.selected_token_ranks, expected_lp.selected_token_ranks
        )
        assert actual_lp.cu_num_generated_tokens == expected_lp.cu_num_generated_tokens
        if actual_lp.cu_num_generated_tokens_tensor is not None:
            assert torch.equal(
                actual_lp.cu_num_generated_tokens_tensor,
                expected_lp.cu_num_generated_tokens_tensor,
            )


@pytest.mark.parametrize(
    "case",
    [
        "raw_logprobs",
        "processed_logprobs",
        "draft",
        "block",
        "synthetic",
        "watermark",
        "adaptive",
    ],
)
def test_refused_compact_batches_use_fused_preparation(monkeypatch, case):
    from types import SimpleNamespace

    import vllm.envs as envs
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    monkeypatch.setattr(envs, "VLLM_SAMPLER_SPLIT_ROW_TOPK", True)
    monkeypatch.setattr(envs, "VLLM_SAMPLER_FUSED_PREP", True)
    monkeypatch.setattr(envs, "VLLM_FUSED_REJECTION_SAMPLER", True)
    monkeypatch.setattr(envs, "VLLM_SPARSE_VERIFY_SAMPLING", True)
    torch.manual_seed(23)
    rows = [4, 4, 2, 1, 4, 4]
    b = _make_batch(
        _request_logits(rows, 50003, 2.5),
        rows,
        top_k=[7, 12, 3, 1, 7, 12],
        top_p=[0.9, 1.0, 0.7, 1.0, 0.8, 0.95],
        temperature=(
            [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
            if case == "raw_logprobs"
            else [1.0] * len(rows)
        ),
        penalties=True,
        seed=29,
    )
    mode = case if case.endswith("logprobs") else "raw_logprobs"
    sampler = _runtime_sampler(b, mode, watermark=case == "watermark")
    logprobs = 3 if case.endswith("logprobs") or case == "adaptive" else -1
    sampler.sampling_states.num_logprobs.fill(logprobs)
    spec_config = SimpleNamespace(
        num_speculative_tokens=STEPS,
        enable_adaptive_verification=case == "adaptive",
        rejection_sample_method="block" if case == "block" else "standard",
    )
    rs = RejectionSampler(
        sampler,
        spec_config,
        torch.device(DEVICE),
        watermark_key=1234567 if case == "watermark" else None,
    )
    if case == "synthetic":
        rs.synthetic_conditional_rates = torch.tensor([0.8, 0.5, 0.2], device=DEVICE)
    draft = None
    if case == "draft":
        draft = torch.randn(
            b.temperature.numel(), STEPS, b.logits.shape[1], device=DEVICE
        )
    batch = _input_batch(b)
    if case == "adaptive":
        # Simulate an already compacted batch: the device boundaries, not this
        # stale host layout, govern rejection and returned logprob boundaries.
        batch.cu_num_logits_np = batch.cu_num_logits_np + np.arange(len(rows) + 1)
    original = b.logits.clone()
    actual = rs(b.logits, batch, draft)
    sampler.sampling_states.use_fused_prep = False
    expected = rs(b.logits, batch, draft)
    _assert_outputs_equal(actual, expected)
    assert torch.equal(b.logits, original)


@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "raw_logits"])
@pytest.mark.parametrize("use_fp64", [False, True])
def test_sparse_raw_scores_and_rng_match_dense(monkeypatch, logprobs_mode, use_fp64):
    from types import SimpleNamespace

    import vllm.envs as envs
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs_module

    monkeypatch.setattr(envs, "VLLM_SAMPLER_SPLIT_ROW_TOPK", True)
    monkeypatch.setattr(envs, "VLLM_SAMPLER_FUSED_PREP", True)
    monkeypatch.setattr(envs, "VLLM_FUSED_REJECTION_SAMPLER", True)
    monkeypatch.setattr(envs, "VLLM_SPARSE_VERIFY_SAMPLING", True)
    torch.manual_seed(31)
    rows = [4, 4, 1, 2, 4, 4]
    b = _make_batch(
        _request_logits(rows, 50003, 2.5),
        rows,
        top_k=[12, 7, 1, 3, 12, 7],
        top_p=[0.95, 0.8, 1.0, 0.9, 1.0, 0.95],
        temperature=[1.0] * len(rows),
        penalties=True,
        seed=37,
        chunked=[False, False, True, False, False, False],
    )
    sampler = _runtime_sampler(b, logprobs_mode)
    sampler.use_fp64_gumbel = use_fp64
    sampler.sampling_states.num_logprobs.fill(3)
    rs = rs_module.RejectionSampler(
        sampler,
        SimpleNamespace(
            num_speculative_tokens=STEPS,
            enable_adaptive_verification=False,
            rejection_sample_method="standard",
        ),
        torch.device(DEVICE),
    )
    original = b.logits.clone()
    actual = rs(b.logits, _input_batch(b))
    rs.use_sparse = False
    expected = rs(b.logits, _input_batch(b))
    _assert_outputs_equal(actual, expected)
    assert torch.equal(b.logits, original)
    # Repeat to catch uninitialized dead-chunk/scalar reads and stateful RNG.
    rs.use_sparse = True
    repeated = rs(b.logits, _input_batch(b))
    _assert_outputs_equal(repeated, expected)


@pytest.mark.parametrize(
    "split,fused,sparse",
    [
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (False, False, True),
        (True, True, True),
    ],
)
def test_split_disable_overrides_compact_toggle(monkeypatch, split, fused, sparse):
    from types import SimpleNamespace

    import vllm.envs as envs
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs_module

    monkeypatch.setattr(envs, "VLLM_SAMPLER_SPLIT_ROW_TOPK", split)
    monkeypatch.setattr(envs, "VLLM_SAMPLER_FUSED_PREP", True)
    monkeypatch.setattr(envs, "VLLM_FUSED_REJECTION_SAMPLER", fused)
    monkeypatch.setattr(envs, "VLLM_SPARSE_VERIFY_SAMPLING", sparse)
    torch.manual_seed(41)
    rows = [4, 1, 2]
    b = _make_batch(
        _request_logits(rows, 3000, 2.5),
        rows,
        top_k=[1] * 3,
        top_p=[1.0] * 3,
        temperature=[1.0] * 3,
        penalties=False,
        seed=43,
    )
    sampler = _runtime_sampler(b)
    rs = rs_module.RejectionSampler(
        sampler,
        SimpleNamespace(
            num_speculative_tokens=STEPS,
            enable_adaptive_verification=False,
            rejection_sample_method="standard",
        ),
        torch.device(DEVICE),
    )
    batch = _input_batch(b)
    params = (b.max_top_k, b.use_top_p, False)
    assert rs._fused_spec_params(batch, None, -1) == (
        params if split and fused else None
    )
    assert rs._sparse_spec_params(b.logits, batch, None, b.logits.shape[0], -1) == (
        params if split and sparse else None
    )
    actual = rs(b.logits, _input_batch(b))
    rs.use_fused = False
    rs.use_sparse = False
    sampler.sampling_states.use_fused_prep = False
    sampler.sampling_states.use_split_row_top_k = False
    expected = rs(b.logits, _input_batch(b))
    _assert_outputs_equal(actual, expected)


def test_probabilistic_draft_distribution_after_fused_preparation():
    """The output marginal is p, not the draft q or a one-hot approximation."""
    torch.manual_seed(47)
    n, vocab = 4096, 4
    p = torch.tensor([0.1, 0.2, 0.3, 0.4], device=DEVICE)
    q = torch.tensor([0.4, 0.3, 0.2, 0.1], device=DEVICE)
    logits = p.log().expand(n * 2, vocab).contiguous()
    idx = torch.arange(n, device=DEVICE, dtype=torch.int32)
    expanded = idx.repeat_interleave(2)
    local_pos = torch.tensor([0, 1], device=DEVICE, dtype=torch.int32).repeat(n)
    draft_sampled = torch.zeros(n * 2, device=DEVICE, dtype=torch.int64)
    draft_sampled[1::2] = torch.multinomial(q.expand(n, -1), 1).squeeze(1)
    pos = torch.arange(n * 2, device=DEVICE, dtype=torch.int64)
    cu = torch.arange(0, n * 2 + 1, 2, device=DEVICE, dtype=torch.int32)
    temp = torch.ones(n, device=DEVICE)
    seeds = torch.arange(n, device=DEVICE, dtype=torch.int64) * 100003 + 53
    top_k = torch.full((n,), vocab, device=DEVICE, dtype=torch.int32)
    top_p = torch.ones(n, device=DEVICE)
    prepared = prepare_spec_top_k_top_p(
        logits,
        expanded,
        draft_sampled,
        local_pos,
        temp,
        top_k,
        top_p,
        None,
        vocab,
        False,
    )
    draft_logits = q.log().expand(n, 1, vocab).contiguous()
    sampled, counts = rejection_sample(
        prepared,
        draft_logits,
        draft_sampled,
        cu,
        pos,
        idx,
        expanded,
        local_pos,
        temp,
        seeds,
        1,
    )
    ref_sampled, ref_counts = rejection_sample(
        logits,
        draft_logits,
        draft_sampled,
        cu,
        pos,
        idx,
        expanded,
        local_pos,
        temp,
        seeds,
        1,
    )
    assert torch.equal(counts, ref_counts)
    assert torch.equal(sampled[:, 0], ref_sampled[:, 0])
    empirical = torch.bincount(sampled[:, 0], minlength=vocab).float() / n
    torch.testing.assert_close(empirical, p, rtol=0, atol=0.03)


@pytest.mark.parametrize("use_fp64", [False, True])
def test_sparse_ties_and_dead_chunks_match_dense(use_fp64):
    from vllm.v1.worker.gpu.sample.sparse_verify import sparse_rejection_sample

    torch.manual_seed(59)
    vocab = 50003
    base = torch.zeros(4, vocab, device=DEVICE)
    base[1] = (torch.randn(vocab, device=DEVICE) * 2).round()
    base[2:] = float("-inf")
    base[2, [1, 1023, 8192, 49000]] = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
    rows = [4] * 4
    b = _make_batch(
        base.repeat_interleave(4, dim=0),
        rows,
        top_k=[64, 20, 20, 7],
        top_p=[0.9, 0.95, 0.9, 1.0],
        temperature=[1.0] * 4,
        penalties=False,
        seed=61,
    )
    draft = b.input_ids[b.logits_indices]
    pos = b.positions[b.logits_indices]
    actual, actual_counts = sparse_rejection_sample(
        b.logits,
        b.input_ids,
        b.logits_indices,
        draft,
        pos,
        b.cu_num_logits,
        b.idx_mapping,
        b.expanded_idx_mapping,
        b.expanded_local_pos,
        b.temperature,
        b.seeds,
        b.top_k,
        b.top_p,
        None,
        b.max_top_k,
        b.use_top_p,
        STEPS,
        use_fp64=use_fp64,
    )
    expected, expected_counts = rejection_sample(
        _unfused_canvas(b),
        None,
        draft,
        b.cu_num_logits,
        pos,
        b.idx_mapping,
        b.expanded_idx_mapping,
        b.expanded_local_pos,
        b.temperature,
        b.seeds,
        STEPS,
        use_fp64=use_fp64,
    )
    assert torch.equal(actual_counts, expected_counts)
    for row, count in enumerate(actual_counts.tolist()):
        assert torch.equal(actual[row, :count], expected[row, :count])
