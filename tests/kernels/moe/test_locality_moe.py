# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MOE_LOCALITY_KERNEL (layers/fused_moe/locality_moe.py) on a GPU with two
locality domains (VR200): the one-launch MoE block (router GEMM + routing + FC1 +
FC2 + shared expert) against the production chain (router GEMM through
vllm::lowm_bf16_gemm, FlashInfer's trtllm-gen MXFP8 MoE with do_finalize=False;
shared expert = mm_mxfp8 gate_up -> silu_mul_mxfp8_quant -> mm_mxfp8 down, gate
logits by the lowm row-dot) and an fp32 reference of the same MXFP8 operands.

- Routing: on logits that every accumulation order computes exactly (identity
  router; integer and tie-gate rows, kfmoeA/tie), the selected experts equal
  production's on every row (top-8 by bf16 logit, lower expert id first) and
  the bf16 expert weights equal trtllm-gen's bit for bit. With a dense router
  the kernel's own logits may differ from production's in the last bit; its
  selection then equals production's rule applied to its own logits.
- Output: relL2 of the finalized routed output and of the ungated shared-expert
  output vs the fp32 reference within 1.10x of production's; shared-expert gate
  logits within one bf16 step of production's; bitwise run to run, on cudaMalloc
  weights, with or without the shared expert and under CUDA-graph replay.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)
if torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the locality MoE kernel is SM107-only", allow_module_level=True)

import flashinfer  # noqa: E402
from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout  # noqa: E402

import vllm.model_executor.kernels.linear.lowm_bf16_gemm  # noqa: E402, F401
from vllm.model_executor.kernels.linear.lowm_bf16_gemm import _rowdot  # noqa: E402
from vllm.model_executor.layers.fused_moe import locality_moe as lm  # noqa: E402
from vllm.model_executor.layers.fusion.silu_mul_mxfp8_quant import (  # noqa: E402
    silu_mul_mxfp8_quant,
)
from vllm.model_executor.layers.locality import get_topology  # noqa: E402
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (  # noqa: E402
    _shuffle_mxfp8_moe_weights,
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    swizzle_mxfp8_scale,
)
from vllm.utils.flashinfer import mm_mxfp8  # noqa: E402

E, K, H, INTER = lm.E, lm.TOPK, lm.HID, lm.INTER
BF = torch.bfloat16


def _mx_quant(t: torch.Tensor):
    """MXFP8 E4M3, UE8M0 block-32 scale = ceil(log2(amax / 448))."""
    f = t.float()
    b = f.reshape(*f.shape[:-1], -1, 32)
    amax = (b.abs().amax(-1) / 448.0).clamp_min(2.0**-126)
    bits = amax.view(torch.int32)
    exp = (((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()).clamp(
        -126, 127
    )
    scale = ((exp + 127) << 23).view(torch.float32)
    q = (b / scale.unsqueeze(-1)).reshape(f.shape).to(torch.float8_e4m3fn)
    return q, (exp + 127).to(torch.uint8)


def _mx_requant_ocp(t: torch.Tensor):
    """FC1 output requant (trtllm-gen's OCP floor rule)."""
    b = t.float().reshape(*t.shape[:-1], -1, 32)
    amax = b.abs().amax(-1)
    ex = ((amax.view(torch.int32) >> 23) & 0xFF) - 127 - 8
    e8 = torch.where(amax > 0, (ex + 127).clamp(0, 254), torch.zeros_like(ex))
    inv = ((127 - e8 + 127).clamp(1, 254) << 23).view(torch.float32)
    q = (b * inv.unsqueeze(-1)).clamp(-448, 448).reshape(t.shape)
    return q.to(torch.float8_e4m3fn), e8.to(torch.uint8)


def _dq(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    s = (sf.int() << 23).view(torch.float32)
    return q.float() * s.repeat_interleave(32, dim=-1)


def _mxfp8_linear(w: torch.Tensor, s: torch.Tensor) -> SimpleNamespace:
    """An MXFP8 linear as FlashInferCutedslMxfp8LinearKernel leaves it: weight
    = [K, N] view of [N, K], F8_128x4-swizzled scales.
    """
    n, k = w.shape
    return SimpleNamespace(
        weight=w.t(),
        weight_scale=swizzle_mxfp8_scale(s, M=n, K=k),
        output_size_per_partition=n,
        input_size_per_partition=k,
    )


@pytest.fixture(scope="module")
def layer():
    dev = torch.accelerator.current_device_index()
    if get_topology(dev) is None:
        pytest.skip("the GPU has no two locality domains")
    g = torch.Generator(device="cuda").manual_seed(777)
    bank = []
    for shape, fan in (((E, 2 * INTER, H), H), ((E, H, INTER), INTER)):
        q = torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn)
        s = torch.empty(*shape[:-1], shape[-1] // 32, device="cuda", dtype=torch.uint8)
        for e in range(E):
            w = torch.randn(shape[1:], device="cuda", generator=g) * fan**-0.5 * 4
            q[e], s[e] = _mx_quant(w)
        bank += [q, s]
    w13, s13, w2, s2 = bank
    p13, p2, ps13, ps2 = _shuffle_mxfp8_moe_weights(
        swap_w13_to_w31(w13), w2, swap_w13_to_w31(s13), s2, True
    )
    p13, ps13, p2, ps2 = (t.contiguous() for t in (p13, ps13, p2, ps2))
    # shared expert (gate rows first) + its bf16 gate, shaped like Qwen2MoeMLP
    gu, gus = _mx_quant(torch.randn(2 * INTER, H, device="cuda", generator=g) * 0.09)
    dn, dns = _mx_quant(torch.randn(H, INTER, device="cuda", generator=g) * 0.18)
    wg = (torch.randn(1, H, device="cuda", generator=g) * 0.05).to(BF)
    mlp = SimpleNamespace(
        gate_up_proj=_mxfp8_linear(gu, gus),
        down_proj=_mxfp8_linear(dn, dns),
        expert_gate=SimpleNamespace(weight=wg),
    )
    rt = lm.runtime(dev)
    l13, l2 = lm.place_expert_pairs(p13, p2)
    lw = rt.layer_weights(l13, l2, ps13, ps2)
    key = (l13.data_ptr(), l2.data_ptr())
    lm._LAYERS[key] = lw  # as maybe_place registers it
    assert lm.is_placed(l13, l2) and not lm.is_placed(p13, p2)
    assert lm.attach_shared_expert(l13, l2, mlp)
    lw_plain = rt.layer_weights(p13, p2, ps13, ps2)
    lw_plain.shared = lw.shared
    yield dict(
        bank=bank,
        prod=(p13, ps13, p2, ps2),
        se=(gu, gus, dn, dns, wg),
        mlp=mlp,
        rt=rt,
        lw=lw,
        lw_plain=lw_plain,
        zero_bias=torch.zeros(E, dtype=BF, device="cuda"),
    )
    del lm._LAYERS[key]


def _trt(prod, logits, xq, xsf):
    p13, ps13, p2, ps2 = prod
    g2, w, idx = flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        routing_logits=logits,
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xsf,
        gemm1_weights=p13,
        gemm1_weights_scale=ps13,
        gemm2_weights=p2,
        gemm2_weights_scale=ps2,
        num_experts=E,
        top_k=K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTER,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=4,  # RenormalizeNaive
        use_shuffled_weight=True,
        weight_layout=WeightLayout.MajorK,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=3,  # SwiGLU
        do_finalize=False,
    )
    T = xq.shape[0]
    return g2, w.view(T, K), idx.view(T, K)


def _select(logits: torch.Tensor) -> torch.Tensor:
    """Production selection: top-8 by bf16 logit, lower expert id first."""
    return torch.sort(-logits.double(), dim=-1, stable=True).indices[:, :K].int()


def _finalize(g2, w, idx):
    T = w.shape[0]
    rows = g2[idx.reshape(-1).long()].float().view(T, K, H)
    return (rows * w.float().view(T, K, 1)).sum(1)


def _reference(bank, xq, xsf, ids, w):
    w13, s13, w2, s2 = bank
    x = _dq(xq, xsf)
    out = torch.zeros(x.shape, device="cuda")
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for e in torch.unique(ids).tolist():
            t, k = (ids == e).nonzero(as_tuple=True)
            h = x[t] @ _dq(w13[e], s13[e]).t()
            act = F.silu(h[:, :INTER]) * h[:, INTER:]
            y = _dq(*_mx_requant_ocp(act)) @ _dq(w2[e], s2[e]).t()
            out.index_add_(0, t, y.to(BF).float() * w[t, k].float().unsqueeze(-1))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return out


def _step(v: float, k: int) -> float:
    """V moved by k bf16 steps."""
    b = torch.tensor([v]).to(BF).view(torch.int16).item()
    key = (b if b >= 0 else -(b & 0x7FFF)) + k
    nb = key if key >= 0 else (0x8000 | -key)
    return (
        torch.tensor([nb - 65536 if nb >= 32768 else nb], dtype=torch.int16)
        .view(BF)
        .item()
    )


def _tie_logits(T: int, case: str, seed: int) -> torch.Tensor:
    """KfmoeA tie-gate rows: bf16(N(3.5, 1)) with exact boundary ties (`tie`),
    one-step near ties in both id orders (`near`) or ties inside the top 8.
    """
    gc = torch.Generator().manual_seed(7919 * T + seed)
    rows = torch.randn(T, E, generator=gc).add_(3.5).to(BF).float()
    for t in range(T):
        r = rows[t]
        s = torch.sort(-r.double(), stable=True).indices.tolist()
        if case == "tie":
            k, m = [(2, 1), (3, 1), (3, 2), (4, 2), (9, 8)][t % 5]
            u = r[s[K - m]].item()
            r[s[K - m : K - m + k]] = u
            r[s[K - m + k :]] = torch.minimum(
                r[s[K - m + k :]], torch.tensor(_step(u, -1))
            )
        elif case == "near":
            u = r[s[7]].item()
            i8, i9 = (s[7], s[8]) if (s[7] > s[8]) == (t % 2 == 1) else (s[8], s[7])
            r[s[:6]] = torch.maximum(r[s[:6]], torch.tensor(_step(u, 2)))
            r[s[6]], r[i8], r[i9] = _step(u, 1), u, _step(u, -1)
            r[s[9:]] = torch.minimum(r[s[9:]], torch.tensor(_step(u, -2)))
        else:  # inside
            r[s[0:3]] = r[s[0]].item()
            r[s[4:8]] = r[s[4]].item()
            r[s[8:]] = torch.minimum(r[s[8:]], torch.tensor(_step(r[s[7]].item(), -1)))
    return rows.cuda()


def _inputs(T: int, case: str, seed: int = 0):
    """(x_bf16, w_router): a dense router for `dense`; otherwise the identity
    router with x[:, :256] = the case's logits (exact for any summation order).
    """
    g = torch.Generator(device="cuda").manual_seed(31 * T + seed)
    x = torch.randn(T, H, device="cuda", generator=g)
    if case == "dense":
        gw = torch.Generator(device="cuda").manual_seed(99)
        return x.to(BF), (torch.randn(E, H, device="cuda", generator=gw) * 0.05).to(BF)
    if case == "integer":
        x[:, :E] = torch.randint(-3, 4, (T, E), device="cuda", generator=g).float()
    else:
        x[:, :E] = _tie_logits(T, case, seed)
    w = torch.zeros(E, H, device="cuda")
    w[torch.arange(E), torch.arange(E)] = 1.0
    return x.to(BF), w.to(BF)


def _run(layer, T, case, seed=0):
    xb, wr = _inputs(T, case, seed)
    xq, xsf = _mx_quant(xb)
    logits_prod = torch.ops.vllm.lowm_bf16_gemm(xb, wr, layer["zero_bias"])
    own = torch.empty(T, E, dtype=BF, device="cuda")
    out = layer["rt"].forward(xb, wr, xq, xsf, layer["lw"], logits=own)
    return xb, wr, xq, xsf, logits_prod, own, out


@pytest.mark.parametrize("T", [17, 150, 384, 385, 512])
@pytest.mark.parametrize("case", ["integer", "tie", "near", "inside"])
def test_routing_equals_production(layer, T: int, case: str):
    xb, wr, xq, xsf, lp, own, (g2, ew, idx, ids, _, _) = _run(layer, T, case)
    assert torch.equal(own, lp)  # exact logits for any accumulation order
    assert torch.equal(ids, _select(lp))
    _, tw, _ = _trt(layer["prod"], lp, xq, xsf)
    assert torch.equal(ew.view(torch.int16), tw.view(torch.int16))


@pytest.mark.parametrize("T", [17, 64, 208, 209, 370, 512])
def test_dense_router(layer, T: int):
    xb, wr, xq, xsf, lp, own, (g2, ew, idx, ids, _, _) = _run(layer, T, "dense")
    # bf16 logits within one bf16 step of production's (different fp32 order)
    mag = torch.maximum(own.float().abs(), lp.float().abs())
    assert bool(((own.float() - lp.float()).abs() <= mag * 2.0**-7 + 1e-6).all())
    assert torch.equal(ids, _select(own))
    _, tw, _ = _trt(layer["prod"], own, xq, xsf)
    assert torch.equal(ew.view(torch.int16), tw.view(torch.int16))
    tg2, tw_p, tidx = _trt(layer["prod"], lp, xq, xsf)
    out = _finalize(g2, ew, idx)
    ref = _reference(layer["bank"], xq, xsf, ids.long(), ew)
    ref_t = _reference(layer["bank"], xq, xsf, _select(lp).long(), tw_p)
    rel = ((out - ref).norm() / ref.norm()).item()
    rel_t = ((_finalize(tg2, tw_p, tidx) - ref_t).norm() / ref_t.norm()).item()
    assert math.isfinite(rel) and rel <= 1.10 * rel_t + 1e-6


@pytest.mark.parametrize("T", [17, 150, 370, 512])
def test_shared_expert_vs_production(layer, T: int):
    xb, wr, xq, xsf, lp, own, (_, _, _, _, so, sg) = _run(layer, T, "dense")
    gu, gus, dn, dns, wg = layer["se"]
    mlp = layer["mlp"]
    # production: the MXFP8 linears on the norm's input (swizzled scales)
    pgu = mm_mxfp8(
        xq,
        mlp.gate_up_proj.weight,
        swizzle_mxfp8_scale(xsf, M=T, K=H),
        mlp.gate_up_proj.weight_scale,
        BF,
        backend="cute-dsl",
    )
    aq, asf = silu_mul_mxfp8_quant(pgu)
    po = mm_mxfp8(
        aq, mlp.down_proj.weight, asf, mlp.down_proj.weight_scale, BF, "cute-dsl"
    )
    pg = _rowdot(xb, wg)
    # fp32 reference: bf16 gate_up, silu * up -> bf16 -> MXFP8 (ceil rule), down
    h = (_dq(xq, xsf) @ _dq(gu, gus).t()).to(BF).float()
    act = (h[:, :INTER] / (torch.exp(-h[:, :INTER]) + 1) * h[:, INTER:]).to(BF)
    ref = _dq(*_mx_quant(act)) @ _dq(dn, dns).t()
    rel = ((so.float() - ref).norm() / ref.norm()).item()
    rel_p = ((po.float() - ref).norm() / ref.norm()).item()
    assert math.isfinite(rel) and rel <= 1.10 * rel_p + 1e-6
    assert sg.shape == (T, 1) and so.shape == (T, H)
    # gate logits: one bf16 step, plus fp32 summation-order slack on cancellation
    mag = torch.maximum(sg.float().abs(), pg.float().abs())
    slack = (xb.float().abs() @ wg.float().abs().t()) * 2.0**-20
    assert bool(((sg.float() - pg.float()).abs() <= mag * 2.0**-7 + slack).all())


@pytest.mark.parametrize("T", [17, 290, 512])
def test_bitwise_rerun_placement_and_graph(layer, T: int):
    rt = layer["rt"]
    xb, wr, xq, xsf, lp, own, first = _run(layer, T, "dense")
    again = rt.forward(xb, wr, xq, xsf, layer["lw"])
    plain = rt.forward(xb, wr, xq, xsf, layer["lw_plain"])
    for a in (again, plain):
        assert all(torch.equal(u, v) for u, v in zip(a, first))
    # the routed output does not depend on the shared expert running too
    nose = rt.forward(xb, wr, xq, xsf, layer["lw"], shared=False)
    assert nose[4] is None and nose[5] is None
    assert all(torch.equal(u, v) for u, v in zip(nose[:4], first[:4]))
    # CUDA graph: capture once, replay with fresh inputs, compare with eager
    sx, sq, ss = xb.clone(), xq.clone(), xsf.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        rt.forward(sx, wr, sq, ss, layer["lw"])
        with torch.cuda.graph(graph):
            captured = rt.forward(sx, wr, sq, ss, layer["lw"])
    torch.cuda.current_stream().wait_stream(s)
    for seed in (1, 2):
        xb2, _ = _inputs(T, "dense", seed)
        xq2, xsf2 = _mx_quant(xb2)
        sx.copy_(xb2)
        sq.copy_(xq2)
        ss.copy_(xsf2)
        graph.replay()
        eager = rt.forward(xb2, wr, xq2, xsf2, layer["lw"])
        assert all(torch.equal(u, v) for u, v in zip(captured, eager))


def test_try_apply_gating(layer):
    lw = layer["lw"]
    l13, l2 = lw.w13, lw.w2
    for T, served in ((16, False), (17, True), (512, True), (513, False)):
        xb, wr = _inputs(T, "dense")
        xq, xsf = _mx_quant(xb)
        out = lm.try_apply(xb, wr, xq, xsf, l13, l2, True)
        assert (out is not None) == (
            served and lm.GATE_MIN_TOKENS <= T <= lm.GATE_MAX_TOKENS
        )
        if out is not None:
            assert len(out) == 4 and out[3] is not None
    xb, wr = _inputs(64, "dense")
    xq, xsf = _mx_quant(xb)
    p13, _, p2, _ = layer["prod"]
    assert lm.try_apply(xb, wr, xq, xsf, p13, p2, True) is None  # not placed
    assert lm.try_apply(xb.float(), wr, xq, xsf, l13, l2, True) is None
    assert lm.try_apply(xb, wr.t(), xq, xsf, l13, l2, True) is None
    x_strided = xb.t().contiguous().t()  # [64, H], not contiguous
    assert lm.try_apply(x_strided, wr, xq, xsf, l13, l2, True) is None
    # a shared expert requested but not attached: not served; without it: served
    sw, lw.shared = lw.shared, None
    try:
        assert lm.try_apply(xb, wr, xq, xsf, l13, l2, True) is None
        out = lm.try_apply(xb, wr, xq, xsf, l13, l2, False)
        assert out is not None and out[3] is None
    finally:
        lw.shared = sw
