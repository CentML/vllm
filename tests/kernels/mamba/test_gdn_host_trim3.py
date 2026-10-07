# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_GDN_HOST_TRIM3: the third round of exact GDN host trims.

- ZERO: one launch zeroes the fresh SSM pool rows of every GDN layer of a
  metadata group; it must zero exactly what the per-layer launches zero.
- HOIST_CONV: the spec rows' causal_conv1d_update launch is bound once per
  step and relaunched per layer; it must equal the wrapper's launch.
- BUFS: the fused conv prep writes into buffers shared per (step, stream).

The forward-level test runs several GDN layers that share one metadata
object through the real metadata builder and ``_forward_core`` (prefill-only
batches, with and without internal checkpoints), with flags off and on, and
compares everything each layer's chunk op sees plus all outputs and pools
bitwise. The chunk op is a stand-in for FlashInfer's in-place SM10x kernel
(the only backend that zeroes fresh rows), so the test runs on any CUDA GPU.
"""

from __future__ import annotations

import dataclasses
import types
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="GDN kernels need CUDA", allow_module_level=True)

from tests.kernels.mamba.test_gdn_prefill_checkpoint import (  # noqa: E402
    BLOCK,
    CONV_DIM,
    CONV_KERNEL,
    HV,
    LAYOUTS,
    H,
    K,
    V,
    _build_layer,
    _make_builder,
    _make_vllm_config,
)
from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
)
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (  # noqa: E402
    StatePoolGroup,
    zero_fresh_state_rows,
    zero_fresh_state_rows_layers,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops import causal_conv1d  # noqa: E402
from vllm.model_executor.layers.mamba.ops.gdn_fused_conv_prep import (  # noqa: E402
    gdn_fused_conv_prep,
)
from vllm.model_executor.layers.mamba.ops.gdn_host_trim import (  # noqa: E402
    step_conv_bufs,
)
from vllm.utils.math_utils import cdiv  # noqa: E402


def _counted(fn, calls: list):
    """``fn``, appending to ``calls`` on every call."""

    def wrapped(*args, **kwargs):
        calls.append(1)
        return fn(*args, **kwargs)

    return wrapped


# ---------------------------------------------------------------- ZERO kernel


def _layer_pools(dtype, num_layers, slots, shared):
    """SSM pools [slots, HV, K, V] with padded slot rows: separate tensors, or
    views at increasing and decreasing offsets of one allocation (as the
    hybrid allocator lays layers out).
    """
    row = HV * K * V
    stride = row + 512
    g = torch.Generator(device="cuda").manual_seed(0)
    if shared:
        raw = torch.randn(num_layers, slots, stride, generator=g, device="cuda")
        raw = raw.to(dtype)
        order = list(range(num_layers))[::-1]  # pool 0 last: negative offsets
        pools = [raw[i, :, :row].view(slots, HV, K, V) for i in order]
        return [raw], pools
    raws = [
        torch.randn(slots, stride, generator=g, device="cuda").to(dtype)
        for _ in range(num_layers)
    ]
    return raws, [r[:, :row].view(slots, HV, K, V) for r in raws]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("num_seqs", [1, 9])
@torch.inference_mode()
def test_zero_fresh_state_rows_layers_matches_per_layer(dtype, shared, num_seqs):
    """Same elements zeroed in every pool (row padding untouched)."""
    slots = 40
    raws, pools = _layer_pools(dtype, 3, slots, shared)
    ref_raws = [r.clone() for r in raws]
    ref_pools = [  # the same views on the clones
        torch.as_strided(
            ref_raws[0] if shared else ref_raws[i],
            p.shape,
            p.stride(),
            p.storage_offset(),
        )
        for i, p in enumerate(pools)
    ]
    cpu = torch.Generator().manual_seed(num_seqs)
    indices = (torch.randperm(slots - 1, generator=cpu)[:num_seqs] + 1).int().cuda()
    flags = torch.rand(num_seqs, generator=cpu) < 0.5
    flags[0] = False
    flags = flags.cuda()
    for p in ref_pools:
        zero_fresh_state_rows(p, indices, flags)
    assert StatePoolGroup.compatible(tuple(pools))
    zero_fresh_state_rows_layers(StatePoolGroup(tuple(pools)), indices, flags)
    for r, ref in zip(raws, ref_raws):
        assert torch.equal(r, ref)


def test_state_pool_group_rejects_mixed_layouts():
    a = torch.zeros(4, HV, K, V, device="cuda")
    assert not StatePoolGroup.compatible((a, a.to(torch.bfloat16)))
    assert not StatePoolGroup.compatible((a, torch.zeros(4, HV, K, V + 4).cuda()))
    unaligned = torch.zeros(4 * HV * K * V + 1, device="cuda")[1:].view(4, HV, K, V)
    assert not StatePoolGroup.compatible((a, unaligned))


# ---------------------------------------------------------- HOIST_CONV launch


def _spec_inputs(num_layers, num_reqs, width, seed):
    """Per-layer (x, conv_state, weight) and the step's index tensors of the
    spec rows of a mixed batch (x: row-strided view, as the layer passes).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = 4  # tokens per spec request (k = 3)
    state_len = width - 1 + q - 1
    slots = 3 * num_reqs + 1
    layers = []
    for _ in range(num_layers):
        x_full = torch.randn(
            num_reqs * q + 5, CONV_DIM + 64, generator=g, device="cuda"
        ).bfloat16()
        conv_state = torch.randn(
            slots, CONV_DIM, state_len, generator=g, device="cuda"
        ).bfloat16()
        weight = torch.randn(CONV_DIM, width, generator=g, device="cuda").bfloat16()
        layers.append((x_full[: num_reqs * q, :CONV_DIM], conv_state, weight))
    cpu = torch.Generator().manual_seed(seed)
    slot_ids = (torch.randperm(slots - 1, generator=cpu)[:num_reqs] + 1).int().cuda()
    accepted = torch.randint(1, q + 1, (num_reqs,), generator=cpu).int().cuda()
    cu = torch.arange(0, (num_reqs + 1) * q, q, dtype=torch.int32, device="cuda")
    return layers, (slot_ids, accepted, cu, q)


def _wrapper(x, conv_state, weight, idx):
    slot_ids, accepted, cu, q = idx
    return causal_conv1d.causal_conv1d_update(
        x,
        conv_state,
        weight,
        None,
        "silu",
        conv_state_indices=slot_ids,
        num_accepted_tokens=accepted,
        query_start_loc=cu,
        max_query_len=q,
        validate_data=False,
    )


@pytest.mark.parametrize("num_reqs", [1, 5])
@torch.inference_mode()
def test_spec_conv_step_matches_wrapper(num_reqs, monkeypatch):
    """Every layer of the step gets the wrapper's results (x updated in place
    and the conv state); after the first (binding) layer, the wrapper is not
    called.
    """
    layers, idx = _spec_inputs(3, num_reqs, CONV_KERNEL, seed=num_reqs)
    ref = [tuple(t.clone() for t in layer) for layer in layers]
    for x, conv_state, weight in ref:
        _wrapper(x, conv_state, weight, idx)  # also compiles the kernel

    calls: list[int] = []
    real = causal_conv1d.causal_conv1d_update
    monkeypatch.setattr(
        causal_conv1d,
        "causal_conv1d_update",
        _counted(real, calls),
    )
    md = types.SimpleNamespace()
    slot_ids, accepted, cu, q = idx
    for x, conv_state, weight in layers:
        out = causal_conv1d.causal_conv1d_update_spec_step(
            md, x, conv_state, weight, None, "silu", slot_ids, accepted, cu, q
        )
        assert out is x
    assert md._htrim3_spec_conv
    assert not calls
    for (x, conv_state, _), (rx, rconv, _) in zip(layers, ref):
        assert torch.equal(x, rx)
        assert torch.equal(conv_state, rconv)


@torch.inference_mode()
def test_spec_conv_step_falls_back_on_other_specialization(monkeypatch):
    """A layer whose x has another alignment or stride (another kernel
    specialization or other arguments) takes the wrapper.
    """
    layers, idx = _spec_inputs(2, 3, CONV_KERNEL, seed=7)
    x1, conv1, w1 = layers[1]
    rows = x1.shape[0]
    misaligned = torch.empty(rows * CONV_DIM + 1, device="cuda").bfloat16()
    misaligned = misaligned[1:].view(rows, CONV_DIM)
    misaligned.copy_(x1)
    layers[1] = (misaligned, conv1, w1)
    ref = [tuple(t.clone() for t in layer) for layer in layers]
    for x, conv_state, weight in ref:
        _wrapper(x, conv_state, weight, idx)
    calls: list[int] = []
    real = causal_conv1d.causal_conv1d_update
    monkeypatch.setattr(
        causal_conv1d,
        "causal_conv1d_update",
        _counted(real, calls),
    )
    md = types.SimpleNamespace()
    slot_ids, accepted, cu, q = idx
    for x, conv_state, weight in layers:
        causal_conv1d.causal_conv1d_update_spec_step(
            md, x, conv_state, weight, None, "silu", slot_ids, accepted, cu, q
        )
    assert len(calls) == 1
    for (x, conv_state, _), (rx, rconv, _) in zip(layers, ref):
        assert torch.equal(x, rx)
        assert torch.equal(conv_state, rconv)


# ------------------------------------------------------------------- BUFS


@torch.inference_mode()
def test_conv_prep_into_step_buffers():
    """The conv prep writes the shared buffers with the values it would
    allocate; the buffers are per (metadata object, stream).
    """
    g = torch.Generator(device="cuda").manual_seed(3)
    lens = [37, 300, 5]
    P = sum(lens)
    x = torch.randn(P, CONV_DIM, generator=g, device="cuda").bfloat16() * 0.1
    w = torch.randn(CONV_DIM, CONV_KERNEL, generator=g, device="cuda").bfloat16()
    slots = 8
    state0 = torch.randn(
        slots, CONV_DIM, CONV_KERNEL - 1, generator=g, device="cuda"
    ).bfloat16()
    cache_indices = torch.tensor([3, 5, 1], dtype=torch.int32, device="cuda")
    has_init = torch.tensor([True, False, True], device="cuda")
    cu = torch.tensor([0, 37, 337, 342], dtype=torch.int32, device="cuda")
    ab = torch.randn(P, 2 * HV, generator=g, device="cuda").bfloat16()
    A_log = torch.randn(HV, generator=g, device="cuda")
    dt_bias = torch.randn(HV, generator=g, device="cuda")

    def run(out):
        state = state0.clone()
        result = gdn_fused_conv_prep(
            x, w, state, cache_indices, has_init, cu, ab[:, :HV], ab[:, HV:],
            A_log, dt_bias, H, K, V, out=out,
        )  # fmt: skip
        return result, state

    ref, ref_state = run(None)
    md = types.SimpleNamespace()
    bufs = step_conv_bufs(md, P, H, K, HV, V, x.dtype, x.device)
    assert bufs is not None
    assert step_conv_bufs(md, P, H, K, HV, V, x.dtype, x.device) is bufs
    got, state = run(bufs)
    assert all(a is b for a, b in zip(got, bufs))
    for a, b in zip(got, ref):
        assert torch.equal(a, b)
    assert torch.equal(state, ref_state)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        other = step_conv_bufs(md, P, H, K, HV, V, x.dtype, x.device)
    assert other is not None and other is not bufs
    # Another shape on the same stream: new buffers.
    assert step_conv_bufs(md, P - 1, H, K, HV, V, x.dtype, x.device) is not bufs


# ------------------------------------------------- forward over a layer group


class _InPlaceChunk:
    """Stand-in for FlashInfer's in-place chunk op (``state_indices``): logs
    what each call reads, and updates the pool rows from their current values
    (so a missing or extra zeroing changes the result).
    """

    expects_exp_g = True

    def __init__(self, log: list) -> None:
        self.log = log

    def updates_state_in_place(self, dtype: torch.dtype) -> bool:
        return True

    def __call__(
        self, q, k, v, g, beta, initial_state, output_final_state,
        cu_seqlens=None, use_qk_l2norm_in_kernel=False, core_attn_out=None,
        state_indices=None, max_seqlen=0, cu_seqlens_i32=None,
    ):  # fmt: skip
        assert state_indices is not None and state_indices.is_contiguous()
        self.log.append(
            [t.clone() for t in (q, k, v, g, beta, initial_state, state_indices)]
        )
        rows = state_indices.long()
        initial_state[rows] = initial_state[rows] * 0.5 + 1.0
        vv = v.squeeze(0)
        out = torch.empty_like(vv) if core_attn_out is None else core_attn_out
        out = out.view(vv.shape)
        out.copy_(vv)
        return out.unsqueeze(0), initial_state


NUM_LAYERS = 3


def _layer_names():
    return [f"model.layers.{i}.linear_attn" for i in range(NUM_LAYERS)]


def _forward_group(vllm_config, builder, block_table, segments, pools, x, log):
    """One step over ``segments`` through NUM_LAYERS layers sharing the step's
    metadata object; returns the per-layer outputs.
    """
    mixed_qkv, a, b = x
    seq_lens = [end for _, _, end in segments]
    query_lens = [end - start for _, start, end in segments]
    device = block_table.device
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens), BLOCK, device
    )
    common = dataclasses.replace(
        common, block_table_tensor=block_table[[r for r, _, _ in segments]]
    )
    with set_current_vllm_config(vllm_config):
        meta = builder.build(common_prefix_len=0, common_attn_metadata=common)
    tokens = [torch.arange(s, e, device=device) for _, s, e in segments]
    inputs = [
        torch.cat([t[r][tok] for (r, _, _), tok in zip(segments, tokens)])
        for t in (mixed_qkv, a, b)
    ]
    num_tokens = sum(query_lens)
    names = _layer_names()
    layers = {}
    for name, (conv_state, ssm_state, weights) in zip(names, pools):
        layer = _build_layer(vllm_config, conv_state, ssm_state, weights)
        layer.prefix = name
        layer.chunk_gated_delta_rule = _InPlaceChunk(log)
        layers[name] = layer
    ctx = types.SimpleNamespace(
        attn_metadata=dict.fromkeys(names, meta), no_compile_layers=layers
    )
    outs = []
    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        for name in names:
            out = torch.zeros(num_tokens, HV, V, dtype=torch.bfloat16, device=device)
            layers[name]._forward_core(
                mixed_qkv=inputs[0], b=inputs[2], a=inputs[1], core_attn_out=out
            )
            outs.append(out)
    return meta, outs


@pytest.mark.parametrize("layout", ["none", "three", "one"])
@pytest.mark.parametrize("host_trim", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_group_forward_bitwise_and_launch_counts(
    layout, host_trim, state_dtype, monkeypatch
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    vllm_config = _make_vllm_config("auto")
    checkpoints = layout != "none"
    builder = _make_builder(vllm_config, checkpoints, False, device)
    prefills, _ = LAYOUTS["three" if layout == "none" else layout]
    num_rows = len(prefills)
    max_blocks = cdiv(max(end for _, end in prefills), BLOCK)
    block_table = (
        torch.arange(num_rows * max_blocks, dtype=torch.int32, device=device).view(
            num_rows, max_blocks
        )
        + 1
    )
    pool_size = num_rows * max_blocks + 1
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, CONV_KERNEL, num_spec=3
    )
    longest = max(end for _, end in prefills)
    x = tuple(
        torch.randn(num_rows, longest, d, dtype=torch.bfloat16, device=device) * 0.1
        for d in (CONV_DIM, HV, HV)
    )
    layer_init = []
    for _ in range(NUM_LAYERS):
        conv0 = torch.randn(pool_size, *conv_shape, device=device).bfloat16() * 0.05
        ssm0 = (torch.randn(pool_size, *ssm_shape, device=device) * 0.05).to(
            state_dtype
        )
        weights = (
            torch.randn(HV, device=device) * 0.1,
            torch.randn(HV, device=device) * 0.1,
            torch.randn(CONV_DIM, 1, CONV_KERNEL, device=device).bfloat16() * 0.1,
            None,  # no conv bias: the fused conv prep path
        )
        layer_init.append((conv0, ssm0, weights))
    segments = [(r, s, e) for r, (s, e) in enumerate(prefills)]

    counts = {"layer": 0, "group": 0}
    real_zero, real_zero_layers = (
        qwen_gdn_linear_attn.zero_fresh_state_rows,
        qwen_gdn_linear_attn.zero_fresh_state_rows_layers,
    )

    def count(key, fn):
        def wrapped(*args, **kwargs):
            counts[key] += 1
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        qwen_gdn_linear_attn, "zero_fresh_state_rows", count("layer", real_zero)
    )
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "zero_fresh_state_rows_layers",
        count("group", real_zero_layers),
    )
    monkeypatch.setattr(qwen_gdn_linear_attn, "GDN_HOST_TRIM", host_trim)

    results = {}
    for trim3 in (False, True):
        monkeypatch.setattr(qwen_gdn_linear_attn, "GDN_HOST_TRIM3", trim3)
        counts.update(layer=0, group=0)
        pools = [(c.clone(), s.clone(), w) for c, s, w in layer_init]
        log: list = []
        meta, outs = _forward_group(
            vllm_config, builder, block_table, segments, pools, x, log
        )
        assert (meta.prefill_checkpoint is not None) == checkpoints
        results[trim3] = (log, outs, pools, dict(counts))

    (log0, outs0, pools0, counts0), (log1, outs1, pools1, counts1) = (
        results[False],
        results[True],
    )
    # Everything each chunk call read (inputs, the whole pool, the indices).
    assert len(log0) == len(log1) == NUM_LAYERS * (2 if checkpoints else 1)
    for call0, call1 in zip(log0, log1):
        for t0, t1 in zip(call0, call1):
            assert torch.equal(t0, t1)
    for o0, o1 in zip(outs0, outs1):
        assert torch.equal(o0, o1)
    for (c0, s0, _), (c1, s1, _) in zip(pools0, pools1):
        assert torch.equal(c0, c1)
        assert torch.equal(s0, s1)
    # Fresh rows: one launch per group instead of one per layer (none when
    # the builder saw on the host that no prefill row is fresh).
    assert counts0["group"] == 0 and counts0["layer"] in (0, NUM_LAYERS)
    assert counts1 == {"layer": 0, "group": int(counts0["layer"] > 0)}


# ------------------------------------- mixed batch (spec block + prefills)


@pytest.mark.parametrize("fork", ["serial", "spec_side", "prefill_first"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_mixed_group_bitwise_with_fork(fork, state_dtype, monkeypatch):
    """A mixed batch (spec block, then two prefills, one fresh) through the
    fused-norm core op of NUM_LAYERS layers in two metadata groups (KV-cache
    groups), with the real kernels of this device: flags off and on give
    bitwise equal outputs and pools, with VLLM_GDN_MIXED_FORK off or on
    (either half on the side stream). With the flags on, every layer's spec
    conv update is the bound launch (no wrapper call), the conv-prep outputs
    are one set for the step and, where the prefill kernel updates the pool
    in place (SM10x FlashInfer), one zeroing launch covers each group.
    """
    from tests.kernels.mamba import test_gdn_fused_mtp as mtp
    from vllm.config import SpeculativeConfig
    from vllm.utils.torch_utils import _encode_layer_name
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    torch.manual_seed(5)
    device = torch.device("cuda")
    vllm_config = mtp._make_vllm_config()
    # The serving backend selection (FlashInfer where available: fused conv
    # prep, and the in-place pool on SM10x).
    vllm_config.additional_config = {"gdn_prefill_backend": "auto"}
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=mtp.NUM_SPEC
    )
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=mtp.BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=mtp.NUM_SPEC,
        ),
        layer_names=[mtp.PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    spec = mtp.SPEC_TOKENS
    batch = BatchSpec(seq_lens=[128, 112, 96, 160], query_lens=[spec, spec, 64, 160])
    common = create_common_attn_metadata(
        batch, mtp.BLOCK_SIZE, device, arange_block_indices=True
    )
    common.block_table_tensor.add_(1)

    def build():
        with set_current_vllm_config(vllm_config):
            return builder.build(
                common_prefix_len=0,
                common_attn_metadata=common,
                num_accepted_tokens=torch.tensor(
                    [2, 1, 1, 1], dtype=torch.int32, device=device
                ),
                num_decode_draft_tokens_cpu=torch.tensor(
                    [mtp.NUM_SPEC, mtp.NUM_SPEC, -1, -1], dtype=torch.int32
                ),
            )

    # Two KV-cache groups (two metadata objects of the step), interleaved as
    # the hybrid allocator assigns GDN layers to groups.
    metadata, metadata_b = build(), build()
    assert metadata.num_prefills == 2 and metadata.spec_token_start is not None
    pool_size = int(common.block_table_tensor.max().item()) + 1
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, mtp.H, mtp.HV, mtp.K, mtp.V, mtp.CONV_KERNEL, mtp.NUM_SPEC
    )
    num_tokens = batch.compute_num_tokens()
    names = _layer_names()
    init = []
    for _ in names:
        init.append(
            (
                0.05 * torch.randn(pool_size, *conv_shape, device=device).bfloat16(),
                (0.01 * torch.randn(pool_size, *ssm_shape, device=device)).to(
                    state_dtype
                ),
                0.1 * torch.randn(mtp.HV, device=device),
                0.1 * torch.randn(mtp.HV, device=device),
                0.1
                * torch.randn(
                    mtp.CONV_DIM, 1, mtp.CONV_KERNEL, device=device
                ).bfloat16(),
                torch.randn(mtp.V, device=device),
                0.1
                * torch.randn(
                    num_tokens, mtp.CONV_DIM + mtp.HV * mtp.V, device=device
                ).bfloat16(),
                0.1 * torch.randn(num_tokens, 2 * mtp.HV, device=device).bfloat16(),
            )
        )

    spec_conv_calls: list[int] = []
    real_update = qwen_gdn_linear_attn.causal_conv1d_update
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "causal_conv1d_update",
        _counted(real_update, spec_conv_calls),
    )
    group_zero: list[int] = []
    real_group = qwen_gdn_linear_attn.zero_fresh_state_rows_layers
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "zero_fresh_state_rows_layers",
        _counted(real_group, group_zero),
    )
    monkeypatch.setattr(qwen_gdn_linear_attn, "GDN_HOST_TRIM", True)
    monkeypatch.setattr(qwen_gdn_linear_attn, "GDN_MIXED_FORK", fork != "serial")
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "GDN_MIXED_FORK_PREFILL_FIRST",
        0 if fork == "prefill_first" else 1 << 30,
    )

    def run(trim3: bool):
        monkeypatch.setattr(qwen_gdn_linear_attn, "GDN_HOST_TRIM3", trim3)
        for md in (metadata, metadata_b):
            for key in ("_htrim3_zeroed", "_htrim3_spec_conv"):
                md.__dict__.pop(key, None)
        layers = {}
        for name, (conv, ssm, a_log, dt_bias, w, nw, _, _) in zip(names, init):
            layer = mtp._build_layer(
                vllm_config, conv.clone(), ssm.clone(), a_log, dt_bias, w, nw, "silu"
            )
            layer.prefix = name
            layers[name] = layer
        context = types.SimpleNamespace(
            attn_metadata={
                name: metadata_b if i % 2 else metadata for i, name in enumerate(names)
            },
            no_compile_layers=layers,
        )
        outs = []
        spec_conv_calls.clear()
        group_zero.clear()
        with patch.object(
            qwen_gdn_linear_attn, "get_forward_context", return_value=context
        ):
            for name, (*_, qkvz, ba) in zip(names, init):
                out = torch.zeros(
                    num_tokens, mtp.HV, mtp.V, dtype=torch.bfloat16, device=device
                )
                torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                    qkvz.clone(), ba, out, layer_name=_encode_layer_name(name)
                )
                outs.append(out)
        torch.accelerator.synchronize()
        return context, layers, outs, len(spec_conv_calls), len(group_zero)

    _, ref_layers, ref_outs, ref_conv_calls, ref_zero = run(False)
    context, layers, outs, conv_calls, zero = run(True)
    for o, r in zip(outs, ref_outs):
        assert torch.equal(o, r)
    for name in names:
        assert torch.equal(layers[name].kv_cache[0], ref_layers[name].kv_cache[0])
        assert torch.equal(layers[name].kv_cache[1], ref_layers[name].kv_cache[1])
    assert ref_conv_calls == NUM_LAYERS and ref_zero == 0
    assert conv_calls == 0
    in_place = ref_layers[names[0]].chunk_gated_delta_rule.updates_state_in_place(
        state_dtype
    )
    # One launch per metadata group.
    assert zero == 2 * int(in_place)
    # One set of conv-prep outputs for the step (per stream; both groups) when
    # the fused conv prep runs (FlashInfer backends).
    bufs = context.__dict__.get("_htrim3_conv_bufs")
    if qwen_gdn_linear_attn._fused_conv_prep_applies(ref_layers[names[0]], metadata):
        assert bufs is not None and len(bufs) == 1
    else:
        assert bufs is None
