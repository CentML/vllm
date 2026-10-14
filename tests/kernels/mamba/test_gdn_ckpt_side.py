# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_GDN_CKPT_SIDE: the prefill checkpoint's conv-window store and tail gather
run on a side stream. One forward with internal checkpoints must give bitwise the
outputs, conv pool and ssm pool of the serial forward, also when the side-stream
copies are delayed (a missing wait would let the tail call read the gathered
rows, or the layer end, before they are written). The side stream must be used.
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="GDN kernels need CUDA", allow_module_level=True)

from tests.kernels.mamba import test_gdn_prefill_checkpoint as T  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q  # noqa: E402
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.utils.math_utils import cdiv  # noqa: E402


@pytest.mark.parametrize("layout", ["three", "one"])
@pytest.mark.parametrize("num_decodes", [0, 2])
@pytest.mark.parametrize("delay", [False, True])
def test_ckpt_side_bitwise(monkeypatch, layout: str, num_decodes: int, delay: bool):
    torch.manual_seed(0)
    device = torch.device("cuda")
    vllm_config = T._make_vllm_config("auto")
    prefills, checkpointed = T.LAYOUTS[layout]
    requests = [(63, 64)] * num_decodes + prefills
    max_blocks = cdiv(max(end for _, end in requests), T.BLOCK)
    num_rows = len(requests)
    block_table = (
        torch.arange(num_rows * max_blocks, dtype=torch.int32, device=device).view(
            num_rows, max_blocks
        )
        + 1
    )
    pool_size = num_rows * max_blocks + 1
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, T.H, T.HV, T.K, T.V, T.CONV_KERNEL, num_spec=3
    )
    conv0 = torch.randn(pool_size, *conv_shape, dtype=torch.bfloat16, device=device)
    ssm0 = torch.randn(pool_size, *ssm_shape, device=device).to(torch.bfloat16)
    conv0 *= 0.05
    ssm0 *= 0.05
    weights = (
        torch.randn(T.HV, dtype=torch.float32, device=device) * 0.1,
        torch.randn(T.HV, dtype=torch.float32, device=device) * 0.1,
        torch.randn(T.CONV_DIM, 1, T.CONV_KERNEL, dtype=torch.bfloat16, device=device)
        * 0.1,
        torch.randn(T.CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1,
    )
    longest = max(end for _, end in requests)
    x = (
        torch.randn(num_rows, longest, T.CONV_DIM, dtype=torch.bfloat16, device=device)
        * 0.1,
        torch.randn(num_rows, longest, T.HV, dtype=torch.bfloat16, device=device) * 0.1,
        torch.randn(num_rows, longest, T.HV, dtype=torch.bfloat16, device=device) * 0.1,
    )
    segments = [(r, s, e) for r, (s, e) in enumerate(requests)]

    def run(side: bool):
        monkeypatch.setattr(Q, "GDN_CKPT_SIDE", side)
        builder = T._make_builder(vllm_config, True, True, device)
        conv, ssm = conv0.clone(), ssm0.clone()
        meta, out = T._forward(
            vllm_config, builder, block_table, segments, (conv, ssm), weights, x
        )
        torch.accelerator.synchronize()
        assert meta.prefill_checkpoint is not None
        return out, conv, ssm

    out_a, conv_a, ssm_a = run(False)

    streams: list[str] = []
    side_stream = Q._gdn_ckpt_side_streams()[0]
    real_gather, real_store = Q.gather_prefill_rows, Q.store_conv_checkpoint

    def spy(name, fn):
        def wrapped(*args, **kwargs):
            on_side = torch.cuda.current_stream() == side_stream
            streams.append(f"{name}:{'side' if on_side else 'main'}")
            if delay and on_side:
                torch.cuda._sleep(
                    50_000_000
                )  # ~20 ms: lands after any unordered reader
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(Q, "gather_prefill_rows", spy("gather", real_gather))
    monkeypatch.setattr(Q, "store_conv_checkpoint", spy("store", real_store))
    out_b, conv_b, ssm_b = run(True)

    # The gather moves to the side stream on the multi-checkpoint layout; the
    # window store does whenever the fused conv prep runs.
    if layout == "three":
        assert "gather:side" in streams
    assert all(s.endswith(":side") for s in streams if s.startswith("gather")), streams
    for r in out_a:
        assert torch.equal(out_a[r], out_b[r]), r
    assert torch.equal(conv_a, conv_b)
    assert torch.equal(ssm_a, ssm_b)
