# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that the M-RoPE text-only fast path in the V2 GPU runner produces
positions identical to the staged-positions path.

Text-only requests skip staging their [num_dims, prompt_len] prefill positions
and the position kernel computes them as plain 1D positions instead. The
result must be bit-identical to staging the model's text-only positions, also
when text-only and multimodal requests share a batch.
"""

import pytest
import torch

from vllm.utils.platform_utils import is_uva_available

FAST_PATH_ENV = "VLLM_MROPE_TEXT_ONLY_FAST_PATH"
NUM_DIMS = 3
MM_DELTA = -7


class StubMRoPEModel:
    """Text-only positions are arange(L) in every dim with delta 0 (as for
    Qwen-VL style models); requests with multimodal features get distinct
    per-dim positions and a nonzero delta so that they must be read from the
    staged buffer.
    """

    def get_mrope_input_positions(self, input_tokens, mm_features):
        n = len(input_tokens)
        positions = torch.arange(n, dtype=torch.long)
        if not mm_features:
            return positions.unsqueeze(0).expand(NUM_DIMS, -1), 0
        rows = [positions // (d + 1) for d in range(NUM_DIMS)]
        return torch.stack(rows), MM_DELTA


def _run(monkeypatch, fast_path: bool) -> list[torch.Tensor]:
    monkeypatch.setenv(FAST_PATH_ENV, "1" if fast_path else "0")
    from vllm.v1.worker.gpu.mm.rope import RopeState

    device = torch.device("cuda")
    prompt_lens = [100, 1500, 3000, 777]
    has_mm = [False, True, False, True]
    max_reqs = 8
    state = RopeState(
        num_dims=NUM_DIMS,
        max_num_reqs=max_reqs,
        max_num_tokens=4096,
        max_model_len=8192,
        device=device,
    )
    assert state.text_only_fast_path == fast_path

    model = StubMRoPEModel()
    for req_idx, (length, mm) in enumerate(zip(prompt_lens, has_mm)):
        tokens = list(range(1000, 1000 + length))
        mm_features = [object()] if mm else []
        state.init_prefill_positions(req_idx, model, tokens, mm_features)
    state.apply_staged_writes()

    num_reqs = len(prompt_lens)
    prefill_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
    prefill_lens[:num_reqs] = torch.tensor(prompt_lens, dtype=torch.int32)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)

    outs = []
    # Chunked prefill, the prefill tail, then decode steps.
    steps = [(0.5, 512), (0.99, 200), (1.0, 4), (1.0, 4)]
    for step, (computed_frac, max_query_len) in enumerate(steps):
        computed = [min(int(n * computed_frac), n) + step for n in prompt_lens]
        # A prefilling request never gets tokens past its prefill length.
        query_lens = [
            max_query_len if c >= n else min(max_query_len, n - c)
            for c, n in zip(computed, prompt_lens)
        ]
        num_computed = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        num_computed[:num_reqs] = torch.tensor(computed, dtype=torch.int32)
        query_start_loc = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(0, dtype=torch.int32)
        state.prepare_positions(
            idx_mapping, query_start_loc, prefill_lens, num_computed
        )
        outs.append(state.get_positions(int(query_start_loc[-1])).clone())
    torch.cuda.synchronize()
    return outs


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_uva_available(),
    reason="requires CUDA with UVA",
)
def test_text_only_fast_path_matches_staged_positions(monkeypatch):
    staged = _run(monkeypatch, fast_path=False)
    fast = _run(monkeypatch, fast_path=True)
    assert len(staged) == len(fast)
    for ref, out in zip(staged, fast):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
