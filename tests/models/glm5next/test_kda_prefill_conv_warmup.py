# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup JIT warmup of the planar KDA prefill conv (``kda_prefill_conv=planar``).

``register_planar_conv_warmup`` describes one layer configuration; compiling
its keys must leave nothing for the first prefill step to compile: a real call
with that configuration's shapes (the model's row stride, the conv-state
layout, every stride class of the cache indices) must hit Triton's kernel
cache for all three launches.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops import causal_conv1d_planar as planar
from vllm.platforms import current_platform
from vllm.triton_utils import triton

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

D = 128


def _compiled(dispatchers) -> list[int]:
    """Compiled variants per kernel in Triton's cache for the current device."""
    device = triton.runtime.driver.active.get_current_device()
    return [len(d.kernel.device_caches[device][0]) for d in dispatchers]


def _step(dim: int, row: int, dim_first: bool, cache_indices: torch.Tensor, state_len):
    lens = [1, 1, 300, 40]
    x = torch.randn(sum(lens), row, device="cuda", dtype=torch.bfloat16)[:, :dim]
    weight = torch.randn(dim, 4, device="cuda")
    shape = (8, dim, state_len) if dim_first else (8, state_len, dim)
    pool = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    return planar.kda_causal_conv1d_planar(
        x,
        weight,
        None,
        pool if dim_first else pool.transpose(-1, -2),
        query_start_loc=torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        ),
        cache_indices=cache_indices,
        has_initial_state=torch.tensor([True, True, False, True], device="cuda"),
    )


# Head counts no other glm5next test compiles (4: register kernel at the
# model's row stride; 24: TMA), so the cache probe below is live when the
# directory runs in one process.
@pytest.mark.parametrize("num_heads", [4, 24])
@pytest.mark.parametrize("dim_first", [True, False], ids=["DS", "SD"])
# 3 state positions (no speculative tokens) and 16 (13 speculative tokens):
# the two Triton stride classes of the conv-state pool.
@pytest.mark.parametrize("state_len", [3, 16])
@torch.inference_mode()
def test_planar_conv_warmup_covers_the_runtime_variants(
    num_heads, dim_first, state_len
):
    dim = 3 * num_heads * D
    row = dim + num_heads + 2 * D  # the fused projection's row
    plan = planar.planar_conv_warmup_plan(
        dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        bias_dtype=None,
        dim=dim,
        num_planes=3,
        row_strides=(row,),
        conv_state_dim_first=dim_first,
        state_len=state_len,
    )
    expected = "tma" if current_platform.has_device_capability(90) else "regs"
    if num_heads % 8:
        expected = "regs"
    stencil = (
        planar._planar_stencil_tma if expected == "tma" else planar._planar_stencil_regs
    )
    assert [d for d, _ in plan] == [stencil] + [
        planar._planar_fixup,
        planar._planar_state_update,
    ] * 3
    for dispatcher, kwargs in plan:
        keys = dispatcher.get_warmup_keys(**kwargs)
        assert keys, (dispatcher.__name__, kwargs)
        dispatcher.compile_many(keys)

    dispatchers = [stencil, planar._planar_fixup, planar._planar_state_update]
    before = _compiled(dispatchers)
    # Cache indices as column 0 of block tables of three widths: the three
    # specialization classes of their stride (16-multiple, generic, 1).
    for width in (16, 7, 1):
        table = torch.arange(1, 1 + 4 * width, dtype=torch.int32, device="cuda")
        out = _step(dim, row, dim_first, table.view(4, width)[:, 0], state_len)
        torch.cuda.synchronize()
        assert out.shape == (3, 342, dim // 3)
        assert _compiled(dispatchers) == before, (
            f"the call with cache-index stride {width} compiled a variant the "
            "warmup did not cover"
        )
    # Positive control for the probe: int64 cache indices are a new variant.
    table = torch.arange(1, 5, dtype=torch.int64, device="cuda")
    _step(dim, row, dim_first, table, state_len)
    torch.cuda.synchronize()
    assert _compiled(dispatchers)[1:] == [n + 1 for n in before[1:]]
