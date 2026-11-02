# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash planar KDA prefill conv (``kda_prefill_conv=planar``).

``kda_causal_conv1d_planar`` replaces ``causal_conv1d_fn`` on the chunked
prefill path, so its outputs and conv-state updates must be bit-identical to
it for every batch shape the scheduler produces: decode rows mixed with
prefills, sequences shorter than the conv window, sequences starting inside a
program's tiles, fresh and continued sequences, null cache slots and both
conv-state layouts.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.models.glm5next.nvidia.ops.kda_prefill_conv import (
    kda_causal_conv1d_planar,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

D = 128
WIDTH = 4

# (sequence lengths, has_initial_state, cache slots); slot NULL_BLOCK_ID marks
# a padded sequence, which causal_conv1d_fn skips; PAD_SLOT_ID only appears on
# empty sequences.
CASES = {
    "prefill_with_decode_rows": (
        [1] * 5 + [700, 1000, 33],
        [True] * 5 + [True, False, True],
        list(range(1, 9)),
    ),
    "short_and_null": (
        [2, 1, 3, 5, 0, 31, 64, 129],
        [True, True, True, False, True, True, False, True],
        [1, 2, NULL_BLOCK_ID, 3, PAD_SLOT_ID, 5, 6, 7],
    ),
    "starts_inside_tiles": (
        [1] * 9 + [301, 129, 2, 700, 1, 5, 300, 15, 17, 4],
        [True] * 9 + [True, False, True, True, False, True, False, True, True, True],
        list(range(1, 20)),
    ),
}


def _run(case: str, num_heads: int, dim_first: bool, row_pad: int, impl: str):
    lens, has_init, slots = CASES[case]
    set_random_seed(0)
    device = "cuda"
    dim = 3 * num_heads * D
    num_tokens = sum(lens)
    # q|k|v is a row-strided column slice of the merged projection.
    x = torch.randn(num_tokens, dim + row_pad, device=device, dtype=torch.bfloat16)
    x = x[:, :dim]
    weight = torch.randn(dim, WIDTH, device=device) * 0.5
    num_slots = max(slots) + 2
    if dim_first:
        pool = torch.randn(
            num_slots, dim, WIDTH - 1, device=device, dtype=torch.bfloat16
        )
    else:
        pool = torch.randn(
            num_slots, WIDTH - 1, dim, device=device, dtype=torch.bfloat16
        )
    query_start_loc = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    # Strided like column 0 of a block table.
    cache_indices = torch.tensor(slots, dtype=torch.int32, device=device)
    cache_indices = torch.stack([cache_indices, cache_indices + 1000], 1)[:, 0]
    has_initial_state = torch.tensor(has_init, dtype=torch.bool, device=device)

    def states(p):
        return p if dim_first else p.transpose(-1, -2)

    ref_pool, pool = pool.clone(), pool.clone()
    ref = causal_conv1d_fn(
        x.transpose(0, 1),
        weight,
        None,
        conv_states=states(ref_pool),
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    ).transpose(0, 1)
    out = kda_causal_conv1d_planar(
        x,
        weight,
        None,
        states(pool),
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        impl=impl,
    )
    assert out.shape == (3, num_tokens, dim // 3)
    assert all(plane.is_contiguous() for plane in out.unbind(0))

    # causal_conv1d_fn leaves rows of null-slot sequences unwritten.
    valid = torch.ones(num_tokens, dtype=torch.bool, device=device)
    starts = query_start_loc.tolist()
    for s, slot in enumerate(slots):
        if slot == NULL_BLOCK_ID:
            valid[starts[s] : starts[s + 1]] = False
    torch.testing.assert_close(
        torch.cat(out.unbind(0), dim=1)[valid], ref[valid], atol=0, rtol=0
    )
    torch.testing.assert_close(pool, ref_pool, atol=0, rtol=0)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("num_heads", [2, 16])
@pytest.mark.parametrize("dim_first", [True, False], ids=["DS", "SD"])
@pytest.mark.parametrize("impl", ["regs", "tma"])
@torch.inference_mode()
def test_planar_conv_matches_causal_conv1d(
    case: str, num_heads: int, dim_first: bool, impl: str
):
    if impl == "tma" and not current_platform.has_device_capability(90):
        pytest.skip("TMA needs SM90+")
    _run(case, num_heads, dim_first, row_pad=256, impl=impl)


@torch.inference_mode()
def test_planar_conv_unaligned_rows_fall_back():
    # Rows that are not 16-byte aligned cannot use TMA descriptors; the
    # register kernel must take over with the same results.
    _run("starts_inside_tiles", 2, True, row_pad=3, impl="tma")
