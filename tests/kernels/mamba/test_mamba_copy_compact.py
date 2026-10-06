# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MAMBA_COPY_COMPACT: the planned / compact mamba state copies must leave
every state pool and accepted-token count byte-identical to the fused kernels
(``precopy_mamba_align_fused_kernel`` / ``postprocess_mamba_fused_kernel``)
for the same decisions, through the real ``MambaSpecDecodeGPUContext`` entry
points (V2 precopy and align postprocess, V1 postprocess).
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.mamba_utils import (
    _COMPACT_ENTRY,
    MambaSpecDecodeGPUContext,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="mamba copy kernels need CUDA/Triton"
)

NUM_LAYERS = 3
CONV_WIDTH = 7  # conv_kernel - 1 + num_spec at k=4
CONV_DIM = 96
SSM_SHAPE = (4, 16, 16)
MAX_COLS = 8
BLOCK = 16  # mamba block size in tokens


def _pools(num_blocks, dim_first, ssm_dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    convs, ssms = [], []
    for _ in range(NUM_LAYERS):
        shape = (
            (num_blocks, CONV_DIM, CONV_WIDTH)
            if dim_first
            else (num_blocks, CONV_WIDTH, CONV_DIM)
        )
        convs.append(torch.randn(*shape, generator=g, device="cuda").bfloat16())
        ssms.append(
            torch.randn(num_blocks, *SSM_SHAPE, generator=g, device="cuda").to(
                ssm_dtype
            )
        )
    return convs, ssms


def _ctx(convs, ssms, bt, dim_first, max_reqs, compact):
    n = NUM_LAYERS * 2
    meta = {
        k: torch.zeros(n, dtype=t, device="cuda")
        for k, t in (
            ("state_base_addrs", torch.int64),
            ("state_block_strides", torch.int64),
            ("state_elem_sizes", torch.int32),
            ("state_inner_sizes", torch.int64),
            ("state_conv_widths", torch.int32),
            ("state_group_indices", torch.int32),
            ("state_dim_row_count", torch.int32),
            ("state_dim_row_stride", torch.int64),
        )
    }
    i = 0
    for conv, ssm in zip(convs, ssms):
        meta["state_base_addrs"][i] = conv.data_ptr()
        meta["state_block_strides"][i] = conv.stride(0) * conv.element_size()
        meta["state_elem_sizes"][i] = conv.element_size()
        if dim_first:
            meta["state_conv_widths"][i] = conv.size(2)
            meta["state_inner_sizes"][i] = 1
            meta["state_dim_row_count"][i] = conv.size(1)
            meta["state_dim_row_stride"][i] = conv.stride(1) * conv.element_size()
        else:
            meta["state_conv_widths"][i] = conv.size(1)
            meta["state_inner_sizes"][i] = conv.stride(1)
        i += 1
        meta["state_base_addrs"][i] = ssm.data_ptr()
        meta["state_block_strides"][i] = ssm.stride(0) * ssm.element_size()
        meta["state_elem_sizes"][i] = ssm.element_size()
        meta["state_inner_sizes"][i] = ssm[0].numel()
        i += 1
    return MambaSpecDecodeGPUContext(
        **meta,
        block_size=BLOCK,
        num_states=n,
        mamba_group_ids=[0],
        num_groups=1,
        num_accepted_tokens_out=torch.zeros(max_reqs, dtype=torch.int32, device="cuda"),
        block_table_ptrs=torch.tensor(
            [bt.data_ptr()], dtype=torch.int64, device="cuda"
        ),
        block_table_stride_req=bt.stride(0),
        compact_work=torch.full(
            (max_reqs, _COMPACT_ENTRY), -7, dtype=torch.int32, device="cuda"
        )
        if compact
        else None,
        compact_count=torch.full((1,), 99, dtype=torch.int32, device="cuda")
        if compact
        else None,
        is_initialized=True,
    )


def _case(num_reqs, max_reqs, seed):
    """Random decisions with every branch represented: fresh / same-block /
    crossing precopies, and post-step counts that land before, on and past a
    block boundary with 1..5 accepted tokens. idx_mapping is a permutation of
    request slots with some -1 (skipped) rows.
    """
    g = torch.Generator().manual_seed(seed)
    bt = torch.empty(num_reqs, MAX_COLS, dtype=torch.int32)
    perm = torch.randperm(num_reqs * MAX_COLS, generator=g).int() + 1
    bt[:] = perm.view(num_reqs, MAX_COLS)  # distinct blocks, no aliasing
    slots = torch.randperm(max_reqs, generator=g)[:num_reqs].int()
    idx_mapping = slots.clone()
    idx_mapping[torch.rand(num_reqs, generator=g) < 0.1] = -1
    src_col = torch.randint(-1, 3, (max_reqs,), generator=g).int()
    state_idx = torch.randint(0, 3, (max_reqs,), generator=g).int()
    bias = torch.randint(0, 5, (max_reqs,), generator=g).int()
    accepted = torch.randint(1, 6, (max_reqs,), generator=g).int()
    # post-step computed counts around the 2nd/3rd block boundary
    new_computed = torch.randint(
        2 * BLOCK - 4, 3 * BLOCK + 4, (max_reqs,), generator=g
    ).int()
    sched = torch.randint(1, 6, (max_reqs,), generator=g).int()
    draft = (sched - 1).clamp(min=0)
    computed = torch.randint(2 * BLOCK - 6, 3 * BLOCK, (max_reqs,), generator=g).int()
    # post: state_idx must point at the running block so copies stay in range
    post_state_idx = ((new_computed - accepted) // BLOCK).clamp(0, MAX_COLS - 6).int()
    t = lambda x: x.cuda()  # noqa: E731
    return {
        k: t(v)
        for k, v in dict(
            bt=bt,
            idx_mapping=idx_mapping,
            src_col=src_col,
            state_idx=state_idx,
            bias=bias,
            accepted=accepted,
            new_computed=new_computed,
            sched=sched,
            draft=draft,
            computed=computed,
            post_state_idx=post_state_idx,
        ).items()
    }


@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_reqs", [1, 5, 71, 300])
@pytest.mark.parametrize("step", ["precopy", "post_align", "post_v1"])
def test_compact_copies_match_fused(dim_first, ssm_dtype, num_reqs, step):
    max_reqs = num_reqs + 8
    for seed in range(3):
        c = _case(num_reqs, max_reqs, seed)
        num_blocks = num_reqs * MAX_COLS + 1
        results = []
        for compact in (False, True):
            convs, ssms = _pools(num_blocks, dim_first, ssm_dtype, seed)
            ctx = _ctx(convs, ssms, c["bt"], dim_first, max_reqs, compact)
            accepted = c["accepted"].clone()
            if step == "precopy":
                ctx.run_fused_precopy(
                    num_reqs, c["state_idx"], c["src_col"], c["bias"], c["idx_mapping"]
                )
            elif step == "post_align":
                ctx.run_fused_postprocess_align(
                    num_reqs,
                    accepted,
                    c["post_state_idx"],
                    c["new_computed"],
                    c["idx_mapping"],
                )
            else:
                ctx.run_fused_postprocess(
                    num_reqs,
                    accepted,
                    c["post_state_idx"],
                    c["sched"],
                    c["computed"],
                    c["draft"],
                )
            torch.accelerator.synchronize()
            results.append((convs, ssms, accepted, ctx.num_accepted_tokens_out.clone()))
        (c0, s0, a0, o0), (c1, s1, a1, o1) = results
        for layer in range(NUM_LAYERS):
            assert torch.equal(c0[layer].view(torch.int16), c1[layer].view(torch.int16))
            assert torch.equal(s0[layer].view(torch.uint8), s1[layer].view(torch.uint8))
        assert torch.equal(a0, a1)
        if step == "post_v1":
            assert torch.equal(o0[:num_reqs], o1[:num_reqs])


def test_compact_plan_counts_only_crossings():
    """The plan writes exactly the crossing requests, in batch order."""
    num_reqs, max_reqs = 6, 8
    bt = (
        torch.arange(1, 1 + num_reqs * MAX_COLS, dtype=torch.int32)
        .view(num_reqs, MAX_COLS)
        .cuda()
    )
    convs, ssms = _pools(num_reqs * MAX_COLS + 1, False, torch.float32, 0)
    ctx = _ctx(convs, ssms, bt, False, max_reqs, True)
    src = torch.tensor([-1, 1, 2, 0, 1, 3, 0, 0], dtype=torch.int32).cuda()
    dst = torch.tensor([0, 1, 1, 1, 0, 2, 0, 0], dtype=torch.int32).cuda()
    bias = torch.tensor([0, 0, 2, 1, 0, 3, 0, 0], dtype=torch.int32).cuda()
    ctx.run_fused_precopy(num_reqs, dst, src, bias, None)
    torch.accelerator.synchronize()
    assert int(ctx.compact_count[0]) == 4
    expect = [[2, 2, 1, 2], [3, 0, 1, 1], [4, 1, 0, 0], [5, 3, 2, 3]]
    assert ctx.compact_work[:4].tolist() == expect
