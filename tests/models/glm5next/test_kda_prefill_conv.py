# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash planar KDA prefill conv (``kda_prefill_conv=planar``).

``kda_causal_conv1d_planar`` replaces ``causal_conv1d_fn`` on the chunked
prefill path, so its outputs and conv-state updates must be bit-identical to
it for every batch shape the scheduler produces: decode rows mixed with
prefills, sequences shorter than the conv window, sequences starting inside a
program's tiles, fresh and continued sequences, null cache slots and both
conv-state layouts.

The identity holds because both kernels accumulate the fp32 weights with fused
multiply-adds in tap order (the planar kernel writes them out explicitly). If
a compiler stops contracting the stock kernel's multiply-adds, the outputs
can differ by 1 bf16 ulp on a few elements per million while the conv states
stay exact; relax the output tolerance to 1 ulp here in that case rather than
dropping the comparison.

Inputs are built the model's way: q|k|v is the leading column slice of the
merged projection, whose row holds ``3 * H * D + H + 2 * D`` elements, so the
row is 16-byte aligned (TMA) iff ``H % 8 == 0``. The head counts below cover
both kernels at that stride (2 and 12 fall back to registers, 16 and 64 take
TMA on SM90+), next to a synthetic misaligned row.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.models.glm5next.nvidia.ops.kda_prefill_conv import (
    kda_causal_conv1d_planar,
    kda_planar_conv_impl,
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
# A full chunked-prefill step: 194 decode rows, then eight 8K prompts, half of
# them continued (has_initial_state) and half fresh; about 65K tokens.
LARGE_CASE = (
    [1] * 194 + [8192] * 8,
    [True] * 194 + [True, False] * 4,
    list(range(1, 203)),
)


def model_row_pad(num_heads: int) -> int:
    """Columns after q|k|v in the merged projection (beta and the two gates)."""
    return num_heads + 2 * D


def _run(
    case,
    num_heads: int,
    dim_first: bool,
    row_pad: int,
    impl: str,
    *,
    bias: bool = False,
    weight_dtype: torch.dtype = torch.float32,
    num_planes: int = 3,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Compare the planar kernel with causal_conv1d_fn on one case; returns x."""
    lens, has_init, slots = CASES[case] if isinstance(case, str) else case
    set_random_seed(0)
    device = "cuda"
    dim = 3 * num_heads * D
    num_tokens = sum(lens)
    # q|k|v is a row-strided column slice of the merged projection.
    x = torch.randn(num_tokens, dim + row_pad, device=device, dtype=torch.bfloat16)
    x = x[:, :dim]
    weight = (torch.randn(dim, WIDTH, device=device) * 0.5).to(weight_dtype)
    bias_t = (torch.randn(dim, device=device) * 0.1).to(weight_dtype) if bias else None
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

    def stock(p, cache_indices, has_initial_state):
        return causal_conv1d_fn(
            x.transpose(0, 1),
            weight,
            bias_t,
            conv_states=states(p),
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
        ).transpose(0, 1)

    ref_pool, pool = pool.clone(), pool.clone()
    ref = stock(ref_pool, cache_indices, has_initial_state)
    out = kda_causal_conv1d_planar(
        x,
        weight,
        bias_t,
        states(pool),
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        num_planes=num_planes,
        activation=activation,
        impl=impl,
    )
    assert out.shape == (num_planes, num_tokens, dim // num_planes)
    assert all(plane.is_contiguous() for plane in out.unbind(0))
    out = torch.cat(out.unbind(0), dim=1)

    # causal_conv1d_fn leaves rows of null-slot sequences unwritten.
    valid = torch.ones(num_tokens, dtype=torch.bool, device=device)
    starts = query_start_loc.tolist()
    for s, slot in enumerate(slots):
        if slot == NULL_BLOCK_ID:
            valid[starts[s] : starts[s + 1]] = False
    torch.testing.assert_close(out[valid], ref[valid], atol=0, rtol=0)
    torch.testing.assert_close(pool, ref_pool, atol=0, rtol=0)

    if not bool(valid.all()):
        # The planar kernel writes those rows as if the sequence had no
        # initial state: the stock kernel's output for the same sequences on
        # scratch slots without state is the reference for them.
        scratch_slots = [
            num_slots + s if slot == NULL_BLOCK_ID else slot
            for s, slot in enumerate(slots)
        ]
        scratch_pool = torch.cat(
            [ref_pool, torch.zeros_like(ref_pool[:1]).expand(len(slots), -1, -1)]
        )
        ref_null = stock(
            scratch_pool,
            torch.tensor(scratch_slots, dtype=torch.int32, device=device),
            torch.tensor(
                [h and slot != NULL_BLOCK_ID for h, slot in zip(has_init, slots)],
                device=device,
            ),
        )
        torch.testing.assert_close(out[~valid], ref_null[~valid], atol=0, rtol=0)
    return x


def _expected_impl(impl: str, num_heads: int) -> str:
    tma = current_platform.has_device_capability(90) and num_heads % 8 == 0
    return "tma" if impl == "auto" and tma else "regs"


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("num_heads", [2, 12, 16, 64])
@pytest.mark.parametrize("dim_first", [True, False], ids=["DS", "SD"])
@pytest.mark.parametrize("impl", ["regs", "auto"])
@torch.inference_mode()
def test_planar_conv_matches_causal_conv1d(
    case: str, num_heads: int, dim_first: bool, impl: str
):
    x = _run(case, num_heads, dim_first, model_row_pad(num_heads), impl)
    assert kda_planar_conv_impl(impl, x) == _expected_impl(impl, num_heads)


@pytest.mark.parametrize(
    ("bias", "weight_dtype"),
    [(True, torch.float32), (False, torch.bfloat16), (True, torch.bfloat16)],
    ids=["bias", "bf16", "bias-bf16"],
)
@pytest.mark.parametrize("impl", ["regs", "auto"])
@torch.inference_mode()
def test_planar_conv_bias_and_bf16_weights(bias, weight_dtype, impl):
    # Dead for GLM-5.3-Flash (fp32 weights, no bias), alive for a generic op.
    _run(
        "starts_inside_tiles",
        16,
        True,
        model_row_pad(16),
        impl,
        bias=bias,
        weight_dtype=weight_dtype,
    )


@pytest.mark.parametrize("impl", ["regs", "auto"])
@torch.inference_mode()
def test_planar_conv_single_plane_without_activation(impl):
    _run(
        "short_and_null",
        16,
        False,
        model_row_pad(16),
        impl,
        num_planes=1,
        activation=None,
    )


@torch.inference_mode()
def test_planar_conv_empty_batch():
    # A step without tokens (the early return) leaves the states alone.
    dim = 3 * 16 * D
    x = torch.empty(0, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, WIDTH, device="cuda")
    pool = torch.randn(3, dim, WIDTH - 1, device="cuda", dtype=torch.bfloat16)
    before = pool.clone()
    out = kda_causal_conv1d_planar(
        x,
        weight,
        None,
        pool,
        query_start_loc=torch.zeros(2, dtype=torch.int32, device="cuda"),
        cache_indices=torch.tensor([PAD_SLOT_ID], dtype=torch.int32, device="cuda"),
        has_initial_state=torch.tensor([False], device="cuda"),
    )
    assert out.shape == (3, 0, dim // 3)
    assert torch.equal(pool, before)


@pytest.mark.slow_test
@pytest.mark.parametrize("impl", ["regs", "auto"])
@torch.inference_mode()
def test_planar_conv_full_chunked_prefill_step(impl):
    # About 65K tokens at the TP8 shard: the full grid and the TMA coordinates
    # at the serving size.
    _run(LARGE_CASE, 8, True, model_row_pad(8), impl)


@torch.inference_mode()
def test_planar_conv_unaligned_rows_fall_back():
    # Rows that are not 16-byte aligned cannot use TMA descriptors: impl="auto"
    # must take the register kernel with the same results, and impl="tma" must
    # say so instead of silently doing the same.
    x = _run("starts_inside_tiles", 2, True, 3, "auto")
    assert kda_planar_conv_impl("auto", x) == "regs"
    with pytest.raises(ValueError, match="aligned"):
        kda_planar_conv_impl("tma", x)


@torch.inference_mode()
def test_planar_conv_rejects_bad_inputs():
    # Contract violations raise with a message instead of failing inside a
    # kernel (or, with a short out=, writing past its end).
    device = "cuda"
    dim, num_tokens = 3 * 2 * D, 8
    x = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(dim, WIDTH, device=device)
    pool = torch.zeros(3, dim, WIDTH - 1, device=device, dtype=torch.bfloat16)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([1], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([False], device=device)

    def call(**overrides):
        kwargs = dict(
            x=x,
            weight=weight,
            bias=None,
            conv_states=pool,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
        )
        kwargs.update(overrides)
        return kda_causal_conv1d_planar(**kwargs)

    call()  # the valid baseline
    with pytest.raises(ValueError, match="width 4"):
        call(weight=weight[:, :3])
    with pytest.raises(ValueError, match="dense channels"):
        call(x=x.transpose(0, 1))
    with pytest.raises(ValueError, match="num_planes"):
        call(num_planes=5)
    with pytest.raises(ValueError, match="conv_states"):
        call(conv_states=pool[:, :, :1])
    with pytest.raises(ValueError, match="activation"):
        call(activation="gelu")
    with pytest.raises(ValueError, match="impl"):
        call(impl="TMA")
    with pytest.raises(ValueError, match="power of two"):
        call(block_n=48)
    with pytest.raises(ValueError, match="entry per sequence"):
        call(cache_indices=cache_indices[:0])
    short = torch.empty(3, num_tokens - 1, dim // 3, device=device, dtype=x.dtype)
    with pytest.raises(ValueError, match="out must be"):
        call(out=short)
    wrong_dtype = torch.empty(3, num_tokens, dim // 3, device=device)
    with pytest.raises(TypeError, match="dtype"):
        call(out=wrong_dtype)
