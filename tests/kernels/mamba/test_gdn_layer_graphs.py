# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501 - scenario tables
"""VLLM_GDN_LAYER_GRAPHS: the CUDA-graph replay of the mixed / prefill-only
GDN core is bitwise equal to the eager path.

Each scenario runs a sequence of steps of one variant (same padded token
count, spec-kernel family and bucket, prefill-sequence count, launch configs)
whose per-step data differ (spec requests, prefill lengths and row offsets,
state slots, initial-state flags). The eager path and the layer-graph path run
on identical inputs and state pools; every step compares out_proj's MXFP8
activation and scales, the core output rows, the in-place conv rows of
mixed_qkvz and both state pools bitwise. Two layers share the step metadata
(one KV-cache group), so the second layer's capture runs after the first
layer's replay of the same step.

The chunk kernel is the vendored V-split kernel on SM10x
(``test_layer_graphs_bitwise_vsplit``) and, elsewhere, a deterministic Triton
stand-in for it that reads its cu_seqlens / state slots on the device the way
the V-split kernel does (``test_layer_graphs_bitwise``); the rest (spec conv
update, Triton and CUDA MTP kernels, CUDA conv prep, fresh-state zeroing,
gated norm + MXFP8) is the production code in both.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.has_device_capability(90)):
    pytest.skip("GDN layer graphs need CUDA SM90+", allow_module_level=True)

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import (  # noqa: E402
    gdn_layer_graphs as lg,
)
from vllm.model_executor.layers.mamba.gdn import (  # noqa: E402
    qwen_gdn_linear_attn as M,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (  # noqa: E402
    gdn_mxfp8_scale_numel,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops import (  # noqa: E402
    gdn_fused_conv_prep,
    gdn_mtp_cuda,
)
from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata  # noqa: E402

# Qwen3.6-35B-A3B GDN at TP1; MTP k=5 (6 tokens per spec request).
H, HV, K, V = 16, 32, 128, 128
WIDTH = 4
NUM_SPEC = 5
W = NUM_SPEC + 1
CONV_DIM = 2 * H * K + HV * V
NUM_SLOTS = 160
LAYERS = ("model.layers.0.linear_attn", "model.layers.4.linear_attn")


@triton.jit
def _ref_chunk_kernel(
    q,
    k,
    v,
    g,
    beta,
    out,
    cu,
    state,
    si,
    s_slot,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """Stand-in for the V-split chunk kernel: per (sequence, head, V block),
    the sequential gated delta rule over the rows [cu[i], cu[i+1]) read on the
    device, state read from and written back to pool slot si[i] (also for
    zero-length sequences, like the V-split kernel).
    """
    seq = tl.program_id(0)
    hv = tl.program_id(1)
    vb = tl.program_id(2)
    bos = tl.load(cu + seq)
    eos = tl.load(cu + seq + 1)
    slot = tl.load(si + seq).to(tl.int64)
    h = hv // (HV // H)
    offs_k = tl.arange(0, K)
    offs_v = vb * BV + tl.arange(0, BV)
    sp = state + slot * s_slot + hv * V * K + offs_v[:, None] * K + offs_k[None, :]
    S = tl.load(sp).to(tl.float32)
    for t in range(bos, eos):
        t64 = t.to(tl.int64)  # type: ignore[attr-defined]
        kk = tl.load(k + t64 * H * K + h * K + offs_k).to(tl.float32)
        qq = tl.load(q + t64 * H * K + h * K + offs_k).to(tl.float32)
        vv = tl.load(v + t64 * HV * V + hv * V + offs_v).to(tl.float32)
        gg = tl.load(g + t64 * HV + hv)
        bb = tl.load(beta + t64 * HV + hv)
        S = S * gg
        pred = tl.sum(S * kk[None, :], axis=1)
        S = S + (bb * (vv - pred))[:, None] * kk[None, :]
        o = tl.sum(S * qq[None, :], axis=1) * scale
        tl.store(out + t64 * HV * V + hv * V + offs_v, o.to(tl.bfloat16))
    tl.store(sp, S.to(state.dtype.element_ty))


def _ref_vsplit(
    q,
    k,
    v,
    gate,
    beta,
    output,
    cu_seqlens,
    initial_state,
    output_state,
    scale,
    state_indices=None,
    v_split=2,
    workspace=None,
):
    assert initial_state is output_state and state_indices is not None
    B = cu_seqlens.numel() - 1
    BV = 32
    _ref_chunk_kernel[(B, HV, V // BV)](
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu_seqlens,
        initial_state,
        state_indices,
        initial_state.stride(0),
        scale,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=BV,
        num_warps=4,
    )


class _Norm:
    def __init__(self, weight, activation):
        self.weight = weight
        self.bias = None
        self.eps = 1e-6
        self.group_size = None
        self.norm_before_gate = True
        self.activation = activation


def _can_use_fused_mtp(self, md) -> bool:
    # _can_use_fused_gdn_mtp_decode without the csrc-op check (the mixed path
    # runs the JIT CUDA MTP kernel or the Triton recurrence here).
    si = md.spec_state_indices_tensor
    return (
        md.spec_sequence_masks is not None
        and md.num_decodes == 0
        and md.num_spec_decodes > 0
        and si is not None
        and si.size(1) <= 8
    )


def _build_layer(prefix, gen, state_dtype, activation):
    dev = torch.device("cuda")
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, WIDTH, NUM_SPEC
    )
    conv_state = (
        torch.randn((NUM_SLOTS, *conv_shape), generator=gen, device=dev) * 0.5
    ).to(torch.bfloat16)
    ssm = (torch.randn((NUM_SLOTS, *ssm_shape), generator=gen, device=dev) * 0.05).to(
        state_dtype
    )
    layer = types.SimpleNamespace(
        prefix=prefix,
        enable_packed_recurrent_decode=False,
        disable_tp_for_ba_proj=False,
        tp_size=1,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=H * K,
        value_dim=HV * V,
        activation="silu",
        A_log=torch.rand(HV, generator=gen, device=dev) * 2 - 1,
        dt_bias=torch.rand(HV, generator=gen, device=dev) - 0.5,
        conv1d=types.SimpleNamespace(
            weight=(
                torch.randn(CONV_DIM, 1, WIDTH, generator=gen, device=dev) * 0.5
            ).to(torch.bfloat16),
            bias=None,
        ),
        kv_cache=(conv_state, ssm),
        norm=_Norm(
            (1 + 0.1 * torch.randn(V, generator=gen, device=dev)).to(torch.bfloat16),
            activation,
        ),
        layer_norm_epsilon=1e-6,
        gdn_decode_kernel="cuda",
        _fused_decode_counters=None,
        _ba_pending=False,
    )

    class _Chunk:
        expects_exp_g = True

        def updates_state_in_place(self, dtype):
            return True

        def __call__(self, **kw):
            return ChunkGatedDeltaRule.forward_cuda(self, **kw)  # type: ignore[arg-type]

    layer.chunk_gated_delta_rule = _Chunk()
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
        "_forward_core_decode_spec_post_conv_fused_norm",
        "_forward_core_decode_spec_fused_norm",
        "_rms_norm_gated_cuda",
        "_rms_norm_gated_strided_gate_cuda",
        "_forward_core_fused_norm",
        "_forward_core_fused_norm_packed",
        "_gated_norm_mxfp8",
        "split_ba",
        "_in_proj_ba_join",
    ):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    layer._can_use_fused_gdn_mtp_decode = types.MethodType(_can_use_fused_mtp, layer)
    return layer


def _metadata(gen, nr, prefill_lens, fresh):
    """GDN metadata of a step: nr spec requests (W tokens each, leading rows)
    then len(prefill_lens) prefill sequences; distinct live slots >= 1.
    """
    dev = torch.device("cuda")
    i32 = dict(dtype=torch.int32, device=dev)
    npf, P = len(prefill_lens), sum(prefill_lens)
    S = nr * W
    perm = torch.randperm(NUM_SLOTS - 1, generator=gen, device="cpu") + 1
    spec_slots = perm[: nr * W].view(nr, W).to(**i32) if nr else None
    pre_slots = perm[nr * W : nr * W + npf].to(**i32)
    pcu = torch.tensor([0] + list(torch.tensor(prefill_lens).cumsum(0)), **i32)
    hi = torch.tensor(fresh, dtype=torch.bool, device=dev).logical_not()
    md = GDNAttentionMetadata(
        num_prefills=npf,
        num_prefill_tokens=P,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=nr,
        num_spec_decode_tokens=S,
        num_actual_tokens=S + P,
        has_initial_state=hi,
        non_spec_query_start_loc=pcu,
        non_spec_state_indices_tensor=pre_slots,
        prefill_query_start_loc=pcu,
        prefill_query_start_loc_i64=pcu.to(torch.int64),
        prefill_state_indices=pre_slots,
        prefill_has_initial_state=hi,
        prefill_all_initial_state=not any(fresh),
        prefill_max_seqlen=max(prefill_lens),
    )
    if nr:
        md.spec_sequence_masks = torch.ones(nr, dtype=torch.bool, device=dev)
        md.spec_state_indices_tensor = spec_slots
        md.spec_query_start_loc = torch.arange(0, S + 1, W, **i32)
        md.num_accepted_tokens = torch.randint(
            1, W + 1, (nr,), generator=gen, device="cpu"
        ).to(**i32)
        md.spec_token_start = 0
        md.non_spec_token_start = S
    return md


def _run(layers, md, bufs, use_graphs):
    fc = types.SimpleNamespace(
        attn_metadata={L.prefix: md for L in layers},
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
    )
    with (
        patch.object(M, "get_forward_context", return_value=fc),
        patch("vllm.forward_context.get_forward_context", return_value=fc),
    ):
        lg._ARMED[0] = use_graphs
        try:
            for L, (qkvz, ba, out, out_q, out_scale) in zip(layers, bufs):
                L._forward_core_fused_norm_packed(
                    qkvz, ba, out, out_q=out_q, out_scale=out_scale
                )
        finally:
            lg._ARMED[0] = False


def _scenarios():
    # (name, T, steps: [(spec requests, prefill lengths, fresh flags)])
    return [
        (
            "prefill_1seq",
            1024,
            [(0, [37], [False]), (0, [300], [True]), (0, [700], [False])],
        ),
        (
            "prefill_2seq",
            2048,
            [(0, [900, 600], [False, True]), (0, [1200, 777], [False, False])],
        ),
        (
            "prefill_3seq_tph8",
            4096,
            [
                (0, [3000, 100, 40], [False, False, True]),
                (0, [1100, 2200, 77], [True, False, False]),
            ],
        ),
        (
            "mixed_triton_n12",
            1024,
            [(1, [900], [False]), (2, [500], [False]), (2, [1000], [True])],
        ),
        (
            "mixed_triton_n34",
            2048,
            [(3, [1500], [False]), (4, [1200], [False]), (3, [1800], [True])],
        ),
        (
            "mixed_cuda_n5_8",
            2048,
            [
                (5, [800, 600], [False, True]),
                (8, [1000, 900], [False, False]),
                (6, [100, 1700], [False, False]),
            ],
        ),
    ]


def _check_scenario(name, T, steps, state_dtype, activation):
    gen = torch.Generator(device="cuda").manual_seed(7)
    gen_cpu = torch.Generator().manual_seed(11)
    lg.drop_graphs()
    lg.STATS.clear()
    eager = [_build_layer(p, gen, state_dtype, activation) for p in LAYERS]
    graph = [_build_layer(p, gen, state_dtype, activation) for p in LAYERS]
    for e, g in zip(eager, graph):
        for name_ in ("A_log", "dt_bias"):
            setattr(g, name_, getattr(e, name_))
        g.conv1d.weight = e.conv1d.weight
        g.norm.weight = e.norm.weight
        g.kv_cache = tuple(t.clone() for t in e.kv_cache)
    hidden = HV * V

    def buffers():
        # Fixed-address per-layer buffers of the padded size, as the
        # piecewise graphs' outputs.
        out = []
        for _ in LAYERS:
            out.append(
                (
                    torch.empty(
                        T, CONV_DIM + hidden, dtype=torch.bfloat16, device="cuda"
                    ),
                    torch.empty(T, 2 * HV, dtype=torch.bfloat16, device="cuda"),
                    torch.empty(T, HV, V, dtype=torch.bfloat16, device="cuda"),
                    torch.empty(T, hidden, dtype=torch.float8_e4m3fn, device="cuda"),
                    torch.empty(
                        gdn_mxfp8_scale_numel(T, hidden),
                        dtype=torch.uint8,
                        device="cuda",
                    ),
                )
            )
        return out

    eb, gb = buffers(), buffers()
    replays = 0
    for nr, lens, fresh in steps:
        md_e = _metadata(gen_cpu, nr, lens, fresh)
        # identical metadata contents for both paths (fresh objects: the
        # layer-graph plan is cached on the metadata object)
        md_g = GDNAttentionMetadata(
            **{f: getattr(md_e, f) for f in md_e.__dataclass_fields__}
        )
        for (qe, be, oe, qqe, se), (qg, bg, og, qqg, sg) in zip(eb, gb):
            src = torch.randn(T, CONV_DIM + hidden, device="cuda", generator=gen)
            qe.copy_(src)
            qg.copy_(src)
            src = torch.randn(T, 2 * HV, device="cuda", generator=gen)
            be.copy_(src)
            bg.copy_(src)
            for t in (oe, og):
                t.fill_(float("nan"))
        _run(eager, md_e, eb, use_graphs=False)
        _run(graph, md_g, gb, use_graphs=True)
        torch.accelerator.synchronize()
        assert not isinstance(md_g.__dict__.get(lg._KEY), str), md_g.__dict__.get(
            lg._KEY
        )
        N = md_e.num_actual_tokens
        for li, (e, g) in enumerate(zip(eb, gb)):
            for what, x, y in (
                ("qkvz", e[0], g[0]),
                ("core_out", e[2][:N], g[2][:N]),
                ("out_q", e[3], g[3]),
                ("out_scale", e[4], g[4]),
            ):
                assert torch.equal(x.view(torch.uint8), y.view(torch.uint8)), (
                    f"{name} step nr={nr} lens={lens} layer {li}: {what} differs"
                )
        for li, (e, g) in enumerate(zip(eager, graph)):
            for what, x, y in zip(("conv", "ssm"), e.kv_cache, g.kv_cache):
                assert torch.equal(x.view(torch.uint8), y.view(torch.uint8)), (
                    f"{name} step nr={nr} lens={lens} layer {li}: {what} state differs"
                )
        replays += len(LAYERS)
    assert lg.STATS.get("replays", 0) == replays, lg.STATS
    return lg.STATS.get("captured", 0)


@pytest.fixture
def _kernels(monkeypatch):
    gdn_fused_conv_prep.enable_cuda_kernel()
    if not gdn_fused_conv_prep._cuda_kernel_ready:
        pytest.skip("CUDA conv kernel did not build")
    gdn_mtp_cuda.enable()
    if not gdn_mtp_cuda.ready():
        pytest.skip("GDN MTP CUDA kernel did not build")
    monkeypatch.setattr(lg, "ENABLED", True)


@torch.inference_mode()
def test_warm_triton_kernels_cover_graph_launches(_kernels, monkeypatch):
    """After ``warm_triton_kernels`` (start-up), capturing and replaying every
    scenario's layer graphs compiles no further variant of the kernels only
    the graphs launch with their own arguments (no JIT while serving).
    """
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_tail_ops
    from vllm.model_executor.layers.mamba.ops import causal_conv1d

    fake = types.SimpleNamespace(
        choose_vsplit=lambda *a, **k: 2, chunk_gated_delta_rule_vsplit=_ref_vsplit
    )
    monkeypatch.setattr(M, "_gdn_vsplit_ready", [fake])
    monkeypatch.setattr(lg, "_workspace", lambda *a: None)
    kernels = (
        lg._pack_kernel,
        causal_conv1d._causal_conv1d_update_kernel,
        qwen_gdn_tail_ops._gdn_gated_norm_mxfp8_kernel,
    )
    names = {k.fn.__qualname__ for k in kernels}
    # Drop the in-memory variants so every compile below reaches the hook.
    for k in kernels:
        k.device_caches.clear()
    compiled: list[str] = []
    prev = triton.knobs.runtime.jit_cache_hook

    def hook(*, fn, repr, **kw):
        if fn.name in names:
            compiled.append(repr)
        return prev(fn=fn, repr=repr, **kw) if prev else False

    monkeypatch.setattr(triton.knobs.runtime, "jit_cache_hook", hook)
    gen = torch.Generator(device="cuda").manual_seed(7)
    gen_cpu = torch.Generator().manual_seed(11)
    lg.warm_triton_kernels(
        _build_layer(LAYERS[0], gen, torch.bfloat16, "silu"), NUM_SPEC, torch.bfloat16
    )
    torch.accelerator.synchronize()
    assert {r.split("[", 1)[0] for r in compiled} == names
    compiled.clear()
    hidden = HV * V
    lg.drop_graphs()
    lg.STATS.clear()
    for _, T, steps in _scenarios():
        layers = [_build_layer(p, gen, torch.bfloat16, "silu") for p in LAYERS]
        bufs = [
            (
                torch.randn(T, CONV_DIM + hidden, device="cuda").to(torch.bfloat16),
                torch.randn(T, 2 * HV, device="cuda").to(torch.bfloat16),
                torch.empty(T, HV, V, dtype=torch.bfloat16, device="cuda"),
                torch.empty(T, hidden, dtype=torch.float8_e4m3fn, device="cuda"),
                torch.empty(
                    gdn_mxfp8_scale_numel(T, hidden), dtype=torch.uint8, device="cuda"
                ),
            )
            for _ in LAYERS
        ]
        for nr, lens, fresh in steps:
            _run(layers, _metadata(gen_cpu, nr, lens, fresh), bufs, use_graphs=True)
    torch.accelerator.synchronize()
    assert lg.STATS.get("replays", 0) == len(LAYERS) * sum(
        len(s[2]) for s in _scenarios()
    ), lg.STATS
    assert not compiled, compiled


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize(
    "state_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
@pytest.mark.parametrize("scenario", _scenarios(), ids=lambda s: s[0])
@torch.inference_mode()
def test_layer_graphs_bitwise(scenario, state_dtype, activation, _kernels, monkeypatch):
    fake = types.SimpleNamespace(
        choose_vsplit=lambda *a, **k: 2, chunk_gated_delta_rule_vsplit=_ref_vsplit
    )
    monkeypatch.setattr(M, "_gdn_vsplit_ready", [fake])
    monkeypatch.setattr(lg, "_workspace", lambda *a: None)
    name, T, steps = scenario
    captured = _check_scenario(name, T, steps, state_dtype, activation)
    # Graphs are reused across steps of a variant: at most one capture per
    # (layer, variant), fewer than the replays.
    assert 0 < captured <= len(LAYERS) * len(steps)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the V-split chunk kernel is SM10x only",
)
@pytest.mark.parametrize("scenario", _scenarios(), ids=lambda s: s[0])
@torch.inference_mode()
def test_layer_graphs_bitwise_vsplit(scenario, _kernels):
    M._gdn_vsplit_warmup(H, HV, K, torch.bfloat16, torch.bfloat16, torch.device("cuda"))
    if not M._gdn_vsplit_ready:
        pytest.skip("V-split kernel did not compile")
    name, T, steps = scenario
    _check_scenario(name, T, steps, torch.bfloat16, "silu")
