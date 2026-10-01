# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN output-tail kernels against the ops they replace, bit for bit."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

V = 128
EPS = 1e-6

requires_fi_mxfp8 = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.has_device_capability(100)
        and has_flashinfer()
    ),
    reason="FlashInfer MXFP8 quantize needs SM100+",
)


# Token counts on both sides of the layer_norm_fwd 1-row tile boundary
# (T <= 2 * SMs), where the fused kernel switches its reduction layout.
TOKEN_CASES = ["1", "20", "2sm", "2sm+1", "8sm+3"]


def _num_tokens(case: str) -> int:
    from vllm.utils.platform_utils import num_compute_units

    two_sms = 2 * num_compute_units(torch.accelerator.current_device_index())
    by_sms = {"2sm": two_sms, "2sm+1": two_sms + 1, "8sm+3": 4 * two_sms + 3}
    return by_sms[case] if case in by_sms else int(case)


def _inputs(num_tokens: int, heads: int, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """Pre-norm output [T, HV, V], the output gate as a row-strided view of a
    wider projection (as in the model) and the shared [V] norm weight.
    """
    g = torch.Generator(device="cuda").manual_seed(num_tokens * 97 + heads + seed)
    x = torch.randn(num_tokens, heads, V, generator=g, device="cuda").bfloat16()
    qkvz = torch.randn(num_tokens, 3 * heads * V, generator=g, device="cuda").bfloat16()
    z = qkvz[:, 2 * heads * V :].view(num_tokens, heads, V)
    w = (1.0 + 0.1 * torch.randn(V, generator=g, device="cuda")).bfloat16()
    return x, z, w


def _layer_norm_strided(x, z, w, activation, launch_config=True) -> torch.Tensor:
    """The mixed-batch gated norm of the layer (one group per head)."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        gdn_norm_launch_config,
    )
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        layer_norm_fwd,
    )

    num_tokens, heads, _ = x.shape
    rows_per_block = num_warps = None
    if launch_config:
        rows_per_block, num_warps = gdn_norm_launch_config(num_tokens, x.device)
    y = x.clone()
    y2 = y.view(num_tokens, heads * V)
    layer_norm_fwd(
        y2,
        w.repeat(heads),
        None,
        EPS,
        z=z.view(num_tokens, heads * V),
        out=y2,
        group_size=V,
        norm_before_gate=True,
        is_rms_norm=True,
        activation=activation,
        rows_per_block=rows_per_block,
        num_warps=num_warps,
    )
    return y


def _fused(x, z, w, activation, norm_rows, num_valid):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        gdn_gated_norm_mxfp8,
        gdn_mxfp8_scale_numel,
    )

    num_tokens, heads, _ = x.shape
    # Garbage-filled outputs: every byte must be written.
    q = torch.full(
        (num_tokens, heads * V), 0x7F, dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e4m3fn)
    s = torch.full(
        (gdn_mxfp8_scale_numel(num_tokens, heads * V),),
        0xAB,
        dtype=torch.uint8,
        device="cuda",
    )
    gdn_gated_norm_mxfp8(x, z, w, EPS, activation, q, s, norm_rows, num_valid)
    return q.view(torch.uint8), s


def _quant(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """out_proj's own activation quant (swizzled F8_128x4 scales)."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    q, s = mxfp8_e4m3_quantize(
        y.reshape(y.shape[0], -1).contiguous(), is_sf_swizzled_layout=True
    )
    return q.view(torch.uint8), s.reshape(-1)


@requires_fi_mxfp8
@pytest.mark.parametrize("tokens", TOKEN_CASES)
@pytest.mark.parametrize("heads", [16, 24, 32])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_gdn_gated_norm_mxfp8_matches_norm_then_quant(
    tokens: str, heads: int, activation: str
) -> None:
    """e4m3 values and swizzled scales equal FlashInfer's quant of the layer's
    bf16 gated-norm output, including the 128-row scale padding.
    """
    num_tokens = _num_tokens(tokens)
    x, z, w = _inputs(num_tokens, heads)
    normed = _layer_norm_strided(x, z, w, activation)
    ref_q, ref_s = _quant(normed)

    q, s = _fused(x, z, w, activation, (0, num_tokens), num_tokens)
    assert torch.equal(q, ref_q)
    assert torch.equal(s, ref_s)

    # Valid row count read on the device (FULL-graph decode replay).
    count = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
    q, s = _fused(x, z, w, activation, (0, num_tokens), count)
    assert torch.equal(q, ref_q)
    assert torch.equal(s, ref_s)

    # Rows normalized upstream (the MTP kernel's spec rows) are quantized as is.
    q, s = _fused(normed, z, w, activation, (0, 0), num_tokens)
    assert torch.equal(q, ref_q)
    assert torch.equal(s, ref_s)


@requires_fi_mxfp8
@pytest.mark.parametrize("heads", [16, 32])
@torch.inference_mode()
def test_gdn_gated_norm_mxfp8_mixed_rows_and_invalid_tail(heads: int) -> None:
    """A spec-like prefix quantized as is, the norm on the rest, and an
    unwritten tail that must come out as zero values and zero scales.
    """
    num_tokens = 300
    spec_rows, num_valid = 12, 290
    x, z, w = _inputs(num_tokens, heads)
    x[num_valid:] = float("nan")
    normed = _layer_norm_strided(x, z, w, "silu")
    x_in = x.clone()
    x_in[:spec_rows] = normed[:spec_rows]
    q, s = _fused(x_in, z, w, "silu", (spec_rows, num_valid), num_valid)

    expected = normed.clone()
    expected[num_valid:] = 0
    ref_q, ref_s = _quant(expected)
    assert torch.equal(q, ref_q)
    assert torch.equal(s, ref_s)


@requires_fi_mxfp8
@pytest.mark.parametrize("heads", [16, 32])
@torch.inference_mode()
def test_gdn_gated_norm_mxfp8_one_row_tile_reduction_order(heads: int) -> None:
    """At T = 2 * SMs the replaced launch uses 1-row tiles. A different fp32
    sum-of-squares order flips an e4m3 value only about once per 1.5 M
    elements, so this runs enough rows for a mismatch to show.
    """
    num_tokens = _num_tokens("2sm")
    for seed in range(32):
        x, z, w = _inputs(num_tokens, heads, seed=1000 * seed)
        ref_q, ref_s = _quant(_layer_norm_strided(x, z, w, "silu"))
        q, s = _fused(x, z, w, "silu", (0, num_tokens), num_tokens)
        assert torch.equal(q, ref_q), f"seed {seed}"
        assert torch.equal(s, ref_s), f"seed {seed}"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("tokens", TOKEN_CASES + ["600", "2144", "5000"])
@torch.inference_mode()
def test_gdn_norm_launch_config_is_bitwise(tokens: str) -> None:
    """The tuned mixed-batch launch reduces each row as the default does."""
    num_tokens = _num_tokens(tokens)
    x, z, w = _inputs(num_tokens, 32)
    tuned = _layer_norm_strided(x, z, w, "silu", launch_config=True)
    default = _layer_norm_strided(x, z, w, "silu", launch_config=False)
    assert torch.equal(tuned, default)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("num_seqs", [1, 7, 64])
@torch.inference_mode()
def test_zero_fresh_state_rows(num_seqs: int) -> None:
    """Only the slots of sequences without initial state are zeroed, and the
    pool's row padding (vLLM pads the slot stride) is left untouched.
    """
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        zero_fresh_state_rows,
    )

    heads, k, pad, slots = 4, 128, 4096, 96
    row = heads * V * k
    g = torch.Generator(device="cuda").manual_seed(num_seqs)
    pool_raw = torch.randn(slots, row + pad, generator=g, device="cuda")
    pool = pool_raw[:, :row].view(slots, heads, V, k)
    cpu = torch.Generator().manual_seed(num_seqs)
    indices = (torch.randperm(slots - 1, generator=cpu)[:num_seqs] + 1).int().cuda()
    has_initial_state = torch.rand(num_seqs, generator=g, device="cuda") < 0.5
    has_initial_state[0] = False
    expected = pool_raw.clone()
    expected[indices.long()[~has_initial_state], :row] = 0

    zero_fresh_state_rows(pool, indices, has_initial_state)
    assert torch.equal(pool_raw, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@torch.inference_mode()
def test_zero_fresh_state_rows_skips_pad_slots() -> None:
    """Padded rows (negative slot id, no initial state) write nothing: the
    pool is a view one row into a larger buffer, so a store at slot -1 would
    land in the buffer's first row.
    """
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        zero_fresh_state_rows,
    )

    heads, k, slots = 4, 128, 8
    row = heads * V * k
    g = torch.Generator(device="cuda").manual_seed(0)
    buf = torch.randn(slots + 1, row, generator=g, device="cuda")
    pool = buf[1:].view(slots, heads, V, k)
    indices = torch.tensor([3, -1, 5, -1], dtype=torch.int32, device="cuda")
    has_initial_state = torch.tensor([False, False, True, False], device="cuda")
    expected = buf.clone()
    expected[1 + 3] = 0

    zero_fresh_state_rows(pool, indices, has_initial_state)
    assert torch.equal(buf, expected)
