# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""causal_conv1d_fn's tile-stencil fast path (``VLLM_CAUSAL_CONV1D_STENCIL=1``).

With the variable set, width-4 bf16 varlen calls with per-sequence slots run
through the kernels of causal_conv1d_planar.py when the step has enough
tokens per sequence. Outputs and conv states must equal the per-token
kernel's bit for bit (the rows of null-slot sequences excepted: the stencil
writes them, the per-token kernel leaves them unset), the output layout must
be the same, a short step must stay on the per-token kernel, and the path
must stay off without the variable.
"""

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.mamba.ops import causal_conv1d as stock
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

# (lengths, has_initial_state, slots). Four decode rows and two prompts clear
# the guard at every width below; the short step does not.
LONG_STEP = (
    [1] * 4 + [6000, 4000],
    [True] * 4 + [False, True],
    [1, 2, 3, 4, 5, NULL_BLOCK_ID],
)
SHORT_STEP = ([1, 1, 300, 200], [True, True, False, True], [1, 2, 3, 4])
# (channels, weight dtype, bias, activation): the Mamba2, GDN and KDA shapes
# of the conv, scaled down, and a bias-only conv without activation.
CONFIGS = {
    "mamba2": (2048, torch.bfloat16, True, "silu"),
    "gdn": (1536, torch.bfloat16, False, "silu"),
    "kda_plane": (1024, torch.float32, False, "silu"),
    "no_activation": (512, torch.float32, True, None),
}


def _inputs(step, dim, weight_dtype, bias, state_len, dim_first):
    lens, has_init, slots = step
    set_random_seed(0)
    num_tokens = sum(lens)
    # The layers pass the conv input as the [dim, T] view of a column slice of
    # a wider projection: channel stride 1, token stride the row width.
    proj = torch.randn(num_tokens, dim + 64, device="cuda", dtype=torch.bfloat16)
    x = proj[:, :dim].transpose(0, 1)
    weight = (torch.randn(dim, 4, device="cuda") * 0.5).to(weight_dtype)
    bias_t = (torch.randn(dim, device="cuda") * 0.1).to(weight_dtype) if bias else None
    num_slots = max(slots) + 2
    shape = (num_slots, dim, state_len) if dim_first else (num_slots, state_len, dim)
    pool = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    states = pool if dim_first else pool.transpose(-1, -2)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    cache_indices = torch.tensor(slots, dtype=torch.int32, device="cuda")
    has_initial_state = torch.tensor(has_init, device="cuda")
    return (
        x,
        weight,
        bias_t,
        pool,
        states,
        query_start_loc,
        cache_indices,
        has_initial_state,
    )


def _run(
    monkeypatch,
    enabled,
    step,
    dim,
    weight_dtype,
    bias,
    activation,
    state_len,
    dim_first,
):
    monkeypatch.setattr(envs, "VLLM_CAUSAL_CONV1D_STENCIL", enabled)
    x, weight, bias_t, pool, states, qsl, ci, his = _inputs(
        step, dim, weight_dtype, bias, state_len, dim_first
    )
    out = stock.causal_conv1d_fn(
        x,
        weight,
        bias_t,
        conv_states=states,
        query_start_loc=qsl,
        cache_indices=ci,
        has_initial_state=his,
        activation=activation,
    )
    return out, pool


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("dim_first", [True, False], ids=["DS", "SD"])
@pytest.mark.parametrize("state_len", [3, 6])
@torch.inference_mode()
def test_stencil_fast_path_matches_per_token_kernel(
    monkeypatch, config, dim_first, state_len
):
    dim, weight_dtype, bias, activation = CONFIGS[config]
    calls = []
    planar = stock.kda_causal_conv1d_planar

    def counting_planar(*args, **kwargs):
        calls.append(1)
        return planar(*args, **kwargs)

    monkeypatch.setattr(stock, "kda_causal_conv1d_planar", counting_planar)
    args = (dim, weight_dtype, bias, activation, state_len, dim_first)

    ref, ref_pool = _run(monkeypatch, False, LONG_STEP, *args)
    assert not calls
    out, pool = _run(monkeypatch, True, LONG_STEP, *args)
    assert len(calls) == 1
    assert out.shape == ref.shape and out.stride() == ref.stride()
    # The per-token kernel leaves the rows of null-slot sequences unset.
    lens, _, slots = LONG_STEP
    starts = [0, *torch.tensor(lens).cumsum(0).tolist()]
    valid = torch.ones(sum(lens), dtype=torch.bool, device="cuda")
    for s, slot in enumerate(slots):
        if slot == NULL_BLOCK_ID:
            valid[starts[s] : starts[s + 1]] = False
    assert torch.equal(out[:, valid], ref[:, valid])
    assert torch.equal(pool, ref_pool)

    # A short step stays on the per-token kernel.
    out, pool = _run(monkeypatch, True, SHORT_STEP, *args)
    ref, ref_pool = _run(monkeypatch, False, SHORT_STEP, *args)
    assert len(calls) == 1
    assert torch.equal(out, ref) and torch.equal(pool, ref_pool)
