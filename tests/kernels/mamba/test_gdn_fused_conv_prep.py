# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gdn_fused_conv_prep (Triton kernel, and the CUDA kernel on SM10x) is bitwise
equal to causal_conv1d_fn followed by fused_post_conv_prep(output_g_exp=True),
conv state included.
"""

import types

import pytest
import torch

import vllm.model_executor.layers.mamba.ops.gdn_fused_conv_prep as fcp
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.model_executor.layers.mamba.ops.gdn_fused_conv_prep import (
    gdn_fused_conv_prep,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
)
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

if not current_platform.is_cuda():
    pytest.skip(reason="Triton CUDA kernels.", allow_module_level=True)

# Qwen3.6-35B-A3B GDN at TP1
H, K, HV, V = 16, 128, 32, 128
DIM = 2 * H * K + HV * V
WIDTH = 4
STATE_LEN = WIDTH - 1 + 3  # conv state rows incl. 3 spec tokens


@pytest.fixture(params=["triton", "cuda"])
def kernel(request, monkeypatch):
    if request.param == "triton":
        monkeypatch.setattr(fcp, "_cuda_kernel_ready", [])
    else:
        if not current_platform.is_device_capability_family(100):
            pytest.skip("The CUDA kernel is SM10x-only.")
        fcp.enable_cuda_kernel()
        assert fcp._cuda_kernel_ready, "CUDA kernel build failed"
    return request.param


@pytest.mark.parametrize(
    "seqlens",
    [
        [3000],
        [5, 2, 1, 700, 64, 3],  # sequences shorter than the conv window
        [1] * 40 + [3000],  # decode rows folded into the prefill block
        [33] * 64,  # the most sequences one launch takes
    ],
)
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("index_stride", [1, 7])  # block_table[:, 0] is strided
def test_fused_conv_prep_bitwise(seqlens, dim_first, index_stride, kernel):
    torch.manual_seed(len(seqlens))
    n, total = len(seqlens), sum(seqlens)
    # x: q/k/v columns of the [T, DIM + HV * V] in_proj output; a/b: columns
    # of ba.
    x = torch.randn(total, DIM + HV * V, device="cuda").to(torch.bfloat16)[:, :DIM]
    ba = torch.randn(total, 2 * HV, device="cuda").to(torch.bfloat16)
    b, a = ba[:, :HV], ba[:, HV:]
    w = (0.5 * torch.randn(DIM, WIDTH, device="cuda")).to(torch.bfloat16)
    slots = 2 * n + 1
    if dim_first:
        pool = torch.randn(slots, DIM, STATE_LEN, device="cuda").to(torch.bfloat16)
    else:
        pool = torch.randn(slots, STATE_LEN, DIM, device="cuda").to(torch.bfloat16)

    def conv_state(p):
        return p if dim_first else p.transpose(-1, -2)

    cu = torch.tensor(
        [0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.int32
    ).cuda()
    table = torch.zeros(n, index_stride, dtype=torch.int32)
    table[:, 0] = torch.randperm(slots - 1)[:n] + 1
    cache_indices = table.cuda()[:, 0]
    has_init = (torch.rand(n) < 0.5).cuda()
    A_log = torch.randn(HV, device="cuda")
    dt_bias = torch.randn(HV, device="cuda")
    nums_dict, batch_ptr, offsets = compute_causal_conv1d_metadata(
        cu.cpu(), device=cu.device
    )

    pool_ref = pool.clone()
    y = causal_conv1d_fn(
        x.transpose(0, 1),
        w,
        None,
        conv_states=conv_state(pool_ref),
        query_start_loc=cu,
        cache_indices=cache_indices,
        has_initial_state=has_init,
        activation="silu",
        metadata=types.SimpleNamespace(
            nums_dict=nums_dict, batch_ptr=batch_ptr, token_chunk_offset_ptr=offsets
        ),
    ).transpose(0, 1)
    ref = fused_post_conv_prep(
        conv_output=y,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        num_k_heads=H,
        head_k_dim=K,
        head_v_dim=V,
        apply_l2norm=True,
        output_g_exp=True,
    )
    out = gdn_fused_conv_prep(
        x=x,
        conv_weights=w,
        conv_state=conv_state(pool),
        cache_indices=cache_indices,
        has_initial_state=has_init,
        cu_seqlens=cu,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        num_k_heads=H,
        head_k_dim=K,
        head_v_dim=V,
    )
    for name, r, o in zip(("q", "k", "v", "exp_g", "beta"), ref, out):
        assert torch.equal(r, o), name
    assert torch.equal(pool, pool_ref)
