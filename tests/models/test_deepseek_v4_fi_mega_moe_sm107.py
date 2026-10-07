# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU test: DeepSeek-V4 flashinfer moe_ep MegaMoE on SM107 (Rubin), world 1.

Builds ``DeepseekV4MegaMoEExpertsFI`` (moe_backend
``flashinfer_moe_ep_mega_cutedsl``) for the NVFP4 expert checkpoint layout
and checks it against

* the same flashinfer kernel called directly (``MoEEpLayer.forward`` with the
  same capacity profile, scalars and knobs): the vLLM plumbing must not change
  the numerics (NVFP4 alphas, per-expert ``fc1_norm_const``, capacity-profile
  selection);
* a dequantized fp32 MoE with DeepSeek's SwiGLU clamp (sanity bound only: the
  kernel quantizes activations to NVFP4);
* itself under CUDA-graph replay with new routing.

Needs an SM107 GPU and a flashinfer build with the sm107 moe_ep kernels
(flashinfer claude/wp6-megamoe or later); skipped otherwise.
"""

import contextlib
import os
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

H, I, E, TOPK, CLAMP = 512, 256, 8, 4, 10.0
MAX_TOKENS = 512


def _has_sm107_moe_ep() -> bool:
    if not current_platform.is_cuda() or not current_platform.is_device_capability(
        107
    ):
        return False
    try:
        from flashinfer.moe_ep import (  # noqa: F401
            Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        )
    except ImportError:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        not _has_sm107_moe_ep(),
        reason="needs SM107 and flashinfer with the sm107 moe_ep MegaMoE kernels",
    ),
    # The world-1 distributed environment is module-scoped (sm107_dist): the
    # conftest's per-test cleanup_dist_env_and_memory() would tear it down
    # after the first test.
    pytest.mark.skip_global_cleanup,
]

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    lut = _E2M1.to(v.device)
    mag = (v.abs().unsqueeze(-1) - lut).abs().argmin(-1)
    return (mag | ((v < 0).to(torch.int64) << 3)).to(torch.uint8)


def _pack(codes: torch.Tensor) -> torch.Tensor:
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.cat([_E2M1, -_E2M1]).to(packed.device)
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def _quant_nvfp4(w: torch.Tensor):
    """[E, N, K] fp32 -> (packed u8 [E,N,K/2], e4m3 sf [E,N,K/16], s2 [E], deq)."""
    e = w.shape[0]
    s2 = w.abs().amax(dim=(1, 2)) / (6.0 * 448.0)
    blocks = w.view(e, w.shape[1], -1, 16)
    sf = (blocks.abs().amax(-1) / 6.0 / s2[:, None, None]).to(torch.float8_e4m3fn)
    scale = sf.float() * s2[:, None, None]
    codes = _e2m1_codes(blocks / scale.clamp_min(1e-30)[..., None]).flatten(-2)
    packed = _pack(codes)
    deq = (_unpack(packed).view_as(blocks) * scale[..., None]).flatten(-2)
    return packed, sf, s2, deq


def _ref_moe(x, w13, w2, ids, wts):
    """Dequantized fp32 MoE, DeepSeek clamp (gate <= lim, |up| <= lim)."""
    x = x.float()
    out = torch.zeros_like(x)
    for e in range(w13.shape[0]):
        r, k = (ids == e).nonzero(as_tuple=True)
        if r.numel() == 0:
            continue
        h = x[r] @ w13[e].t()
        g, u = h[:, :I].clamp(max=CLAMP), h[:, I:].clamp(-CLAMP, CLAMP)
        a = torch.nn.functional.silu(g) * u
        out.index_add_(0, r, (a @ w2[e].t()) * wts[r, k].unsqueeze(1))
    return out


def _calib_w2_input_scale(w13: torch.Tensor) -> torch.Tensor:
    """Per-expert amax(router weight * SwiGLU(x W13^T)) / 2688 on a calibration batch."""
    x = torch.randn(1024, H, device="cuda")
    ids, wts = _routing(1024, seed=12345)
    amax = torch.zeros(E)
    for e in range(E):
        r, k = (ids == e).nonzero(as_tuple=True)
        h = x[r] @ w13[e].t()
        g, u = h[:, :I].clamp(max=CLAMP), h[:, I:].clamp(-CLAMP, CLAMP)
        a = torch.nn.functional.silu(g) * u * wts[r, k].unsqueeze(1)
        amax[e] = a.abs().max().cpu()
    return amax / 2688.0


def _routing(n: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.argsort(torch.rand(n, E, device="cuda", generator=g), dim=1)[:, :TOPK]
    wts = torch.softmax(torch.randn(n, TOPK, device="cuda", generator=g), dim=-1)
    return ids.to(torch.int64), wts.float()


def _rel(a, b) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


@pytest.fixture(scope="module")
def sm107_dist():
    import tempfile

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
    fd, path = tempfile.mkstemp()
    os.close(fd)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{path}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        yield
    cleanup_dist_env_and_memory()
    with contextlib.suppress(OSError):
        os.unlink(path)


def _vllm_config(tp: int = 1, pp: int = 1):
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        quant_config=SimpleNamespace(moe_quant_algo="NVFP4"),
        kernel_config=SimpleNamespace(moe_backend="flashinfer_moe_ep_mega_cutedsl"),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
            data_parallel_size=1,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=MAX_TOKENS),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def _make_module(prefix: str, tp: int = 1, pp: int = 1):
    from vllm.models.deepseek_v4.nvidia.fi_moe import DeepseekV4MegaMoEExpertsFI

    with torch.device("cuda"):
        return DeepseekV4MegaMoEExpertsFI(
            _vllm_config(tp, pp),
            num_experts=E,
            num_local_experts=E,
            experts_start_idx=0,
            top_k=TOPK,
            hidden_size=H,
            intermediate_size=I,
            num_shared_experts=0,
            prefix=prefix,
            activation_clamp=CLAMP,
        )


def _load_scalar(mod, name: str, shard: str, e: int, value: float):
    param = getattr(mod, name)
    param.weight_loader(
        param,
        torch.tensor(value, dtype=torch.float32),
        f"experts.{name}",
        shard_id=shard,
        expert_id=e,
    )


def _build(seed: int, prefix: str):
    torch.manual_seed(seed)
    w13 = torch.randn(E, 2 * I, H, device="cuda") * H**-0.5 * 3.0
    w2 = torch.randn(E, H, I, device="cuda") * I**-0.5
    mod = _make_module(prefix)
    q13, sf13, s13, d13 = _quant_nvfp4(w13)
    q2, sf2, s2, d2 = _quant_nvfp4(w2)
    mod.w13_weight.data.copy_(q13)
    mod.w13_weight_scale.data.copy_(sf13)
    mod.w2_weight.data.copy_(q2)
    mod.w2_weight_scale.data.copy_(sf2)
    # Calibrated static scales like modelopt's (amax / (6 * 448)): FC1
    # input from x ~ N(0, 1); per-expert FC2 input from the routed,
    # router-weighted SwiGLU output (the kernel applies the router
    # weight before the FC1-output requant), spread 1-4x per expert so
    # the per-expert fc1_norm_const values differ.
    w2_in = _calib_w2_input_scale(d13) * torch.linspace(1.0, 4.0, E)
    for e in range(E):
        for shard in ("w1", "w3"):
            _load_scalar(mod, "w13_weight_scale_2", shard, e, float(s13[e]))
            _load_scalar(mod, "w13_input_scale", shard, e, 6.0 / 2688.0 * (1 + e % 3))
        _load_scalar(mod, "w2_weight_scale_2", "w2", e, float(s2[e]))
        _load_scalar(mod, "w2_input_scale", "w2", e, float(w2_in[e]))
    raw = dict(q13=q13, sf13=sf13, s13=s13, q2=q2, sf2=sf2, s2=s2, w2_in=w2_in)
    mod.finalize_weights()
    return mod, d13, d2, raw


def _direct_layer(raw: dict, mod):
    """The same flashinfer kernel, built without vLLM."""
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    mk = Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=I,
        top_k=TOPK,
        gate_up_clamp=CLAMP,
        knobs="cache",
        input_norm_const=mod._mega_layer._megakernel_config.input_norm_const,
    )
    pack = MoEWeightPack(raw["q13"], raw["q2"], raw["sf13"], raw["sf2"])
    return MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        fleet_params=FleetParams(
            num_experts=E, max_tokens_per_rank=MAX_TOKENS, token_hidden_size=H
        ),
        weights=pack,
        backend=MegaConfig(megakernel=mk),
    )


@pytest.mark.parametrize(
    "fc2_scale", ["layer_max", "per_expert"], ids=["nvfp4", "nvfp4_per_expert"]
)
def test_fi_mega_moe_sm107_world1(sm107_dist, monkeypatch, fc2_scale):
    from flashinfer.moe_ep import MoEEpTensors

    from vllm.models.deepseek_v4.nvidia.fi_moe import sm107_capacity_profiles

    monkeypatch.setenv("VLLM_FI_MEGA_MOE_NVFP4_FC2_INPUT_SCALE", fc2_scale)
    prefix = f"model.layers.1{fc2_scale}.ffn.experts"
    mod, d13, d2, raw = _build(seed=2, prefix=prefix)
    assert mod._megakernel == "sm107_nvfp4_nvfp4_bf16_cutedsl"
    assert mod._profile_caps == sm107_capacity_profiles(MAX_TOKENS, "auto")
    assert mod.w13_weight is None  # loader params released after preprocess

    direct = _direct_layer(raw, mod)
    handles = {cap: direct.create_workspace(cap) for cap in mod._profile_caps}
    fc1_alpha, fc2_alpha, nc = mod._epilogue_alphas
    w2_in = raw["w2_in"].cuda()
    if fc2_scale == "layer_max":
        w2_in = w2_in.max().expand_as(w2_in)
    torch.testing.assert_close(nc, 1.0 / w2_in)
    torch.testing.assert_close(fc2_alpha, raw["s2"].cuda() * w2_in)
    scal = dict(fc1_alpha=fc1_alpha, fc2_alpha=fc2_alpha, fc1_norm_const=nc)

    for n in (1, 37, 128, 200, MAX_TOKENS):
        x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16)
        ids, wts = _routing(n, seed=n)
        y = mod(x, wts, ids, activation_clamp=CLAMP).clone()
        cap = next(c for c in mod._profile_caps if c >= n)
        y_direct = direct.forward(
            MoEEpTensors(x, ids.to(torch.int32), wts, **scal), workspace=handles[cap]
        )
        assert y.shape == (n, H) and y.dtype == torch.bfloat16
        assert torch.isfinite(y).all()
        assert _rel(y, y_direct) < 2e-3, (n, _rel(y, y_direct))
        ref = _ref_moe(x, d13, d2, ids, wts)
        cos = torch.nn.functional.cosine_similarity(
            y.float().flatten(), ref.flatten(), dim=0
        )
        assert cos > 0.98 and _rel(y, ref) < 0.25, (n, float(cos), _rel(y, ref))

    # Unit fc1_norm_const (the legacy vLLM behaviour) must give a
    # different result: proves the per-expert norm const reaches the kernel.
    x = torch.randn(64, H, device="cuda", dtype=torch.bfloat16) * 0.05
    ids, wts = _routing(64, seed=7)
    y = mod(x, wts, ids, activation_clamp=CLAMP).clone()
    unit = dict(scal, fc1_norm_const=torch.ones_like(scal["fc1_norm_const"]))
    unit["fc2_alpha"] = raw["s2"].cuda().float()
    y_unit = direct.forward(
        MoEEpTensors(x, ids.to(torch.int32), wts, **unit),
        workspace=handles[128],
    )
    ref = _ref_moe(x, d13, d2, ids, wts)
    assert _rel(y, ref) < _rel(y_unit, ref), (_rel(y, ref), _rel(y_unit, ref))

    # CUDA graph: capture once, replay with new routing / activations.
    n = 37
    x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16)
    ids, wts = _routing(n, seed=11)
    mod(x, wts, ids, activation_clamp=CLAMP)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_graph = mod(x, wts, ids, activation_clamp=CLAMP)
    for seed in (12, 13):
        x.copy_(torch.randn_like(x))
        new_ids, new_wts = _routing(n, seed=seed)
        ids.copy_(new_ids)
        wts.copy_(new_wts)
        graph.replay()
        torch.accelerator.synchronize()
        y_graph_out = y_graph.clone()
        y_eager = mod(x, wts, ids, activation_clamp=CLAMP).clone()
        assert _rel(y_graph_out, y_eager) < 1e-3
    del graph
    torch.accelerator.synchronize()
    for h in handles.values():
        h.destroy()
    direct.destroy()


# ----------------------------------------------------------------------------
# Shared expert next to the megakernel (VLLM_FI_MEGA_MOE_SHARED).
class _FakeFp8Linear:
    def __init__(self, w: torch.Tensor):
        blocks = w.view(w.shape[0] // 32, 32, w.shape[1] // 32, 32)
        exp = torch.ceil(torch.log2(blocks.abs().amax(dim=(1, 3)).clamp_min(1e-30) / 448.0))
        scale = torch.exp2(exp).repeat_interleave(32, 0).repeat_interleave(32, 1)
        self.weight = SimpleNamespace(data=(w / scale).to(torch.float8_e4m3fn))
        self.weight_scale = SimpleNamespace(
            data=(exp + 127).to(torch.uint8), dtype=torch.uint8
        )
        self.weight_block_size = [32, 32]
        self.deq = self.weight.data.float() * scale


class _FakeSharedMLP(torch.nn.Module):
    """FP8 32x32-block shared MLP (DeepSeek-V4 layout); fp32 math for overlap."""

    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.gate_up_proj = _FakeFp8Linear(
            torch.randn(2 * I, H, device="cuda", generator=g) * H**-0.5 * 3.0
        )
        self.down_proj = _FakeFp8Linear(
            torch.randn(H, I, device="cuda", generator=g) * I**-0.5
        )

    def forward(self, x):
        h = x.float() @ self.gate_up_proj.deq.t()
        g, u = h[:, :I].clamp(max=CLAMP), h[:, I:].clamp(-CLAMP, CLAMP)
        return ((torch.nn.functional.silu(g) * u) @ self.down_proj.deq.t()).to(x.dtype)


@pytest.mark.parametrize("max_sm", ["0", "auto"])
def test_fi_mega_moe_sm107_overlapped_shared(sm107_dist, monkeypatch, max_sm):
    """overlap: shared MLP on the aux stream next to the (SM-budgeted)
    megakernel == serial add, eager and under CUDA-graph replay."""
    monkeypatch.setenv("VLLM_FI_MEGA_MOE_SHARED", "overlap")
    monkeypatch.setenv("VLLM_FI_MEGA_MOE_MAX_SM_COUNT", max_sm)
    shared = _FakeSharedMLP(seed=31)
    mod, d13, d2, raw = _build(seed=6, prefix=f"model.layers.shov{max_sm}.ffn.experts")
    assert mod.overlaps_shared_experts and not mod.has_fused_shared_experts
    n = 77
    x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16)
    ids, wts = _routing(n, seed=3)
    y_ov = mod(x, wts, ids, activation_clamp=CLAMP, shared_experts=shared).clone()
    y_ser = mod(x, wts, ids, activation_clamp=CLAMP).clone() + shared(x)
    assert torch.equal(y_ov, y_ser), _rel(y_ov, y_ser)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_graph = mod(x, wts, ids, activation_clamp=CLAMP, shared_experts=shared)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.accelerator.synchronize()
    # y_graph is a view of the profile's workspace output: copy it before the
    # eager reference call reuses that workspace.
    y_graph_out = y_graph.clone()
    y_ref = mod(x, wts, ids, activation_clamp=CLAMP).clone() + shared(x)
    assert torch.equal(y_graph_out, y_ref), _rel(y_graph_out, y_ref)
    del graph


class _StreamRecordingSharedMLP(_FakeSharedMLP):
    def __init__(self, seed: int):
        super().__init__(seed)
        self.streams: list[int] = []

    def forward(self, x):
        self.streams.append(torch.cuda.current_stream().cuda_stream)
        return super().forward(x)


def test_fi_mega_moe_sm107_overlapped_shared_breakable(sm107_dist, monkeypatch):
    """overlap inside a breakable CUDA-graph segment (DeepSeek-V4's PIECEWISE
    mode for mixed batches): the shared MLP is captured on the side stream
    between attention-like eager breaks and replay == serial, bitwise."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.models.deepseek_v4.nvidia import fi_moe
    from vllm.utils.torch_utils import _current_stream_tls

    monkeypatch.setenv("VLLM_FI_MEGA_MOE_SHARED", "overlap")
    shared = _StreamRecordingSharedMLP(seed=41)
    mod, *_ = _build(seed=7, prefix="model.layers.shovbrk.ffn.experts")
    assert mod.overlaps_shared_experts
    side = mod._shared_stream.cuda_stream
    n = 77
    x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16)
    x_in, out = torch.empty_like(x), torch.empty_like(x)
    ids, wts = _routing(n, seed=4)
    prev = getattr(_current_stream_tls, "value", None)
    cap_stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(cap_stream):
            mod(x, wts, ids, activation_clamp=CLAMP, shared_experts=shared)
            cap_stream.synchronize()
            before = fi_moe.SM107_OVERLAP_CONTEXTS["breakable_graph"]
            shared.streams.clear()
            cap = BreakableCUDAGraphCapture()
            with cap:
                x_pre = x.mul(1.0)
                cap.add_eager(lambda: x_in.copy_(x_pre))
                y = mod(x_in, wts, ids, activation_clamp=CLAMP, shared_experts=shared)
                cap.add_eager(lambda: out.copy_(y))
                out.mul_(1.0)
            assert (cap.num_graphs, cap.num_eager_breaks) == (3, 2)
            assert fi_moe.SM107_OVERLAP_CONTEXTS["breakable_graph"] == before + 1
            assert shared.streams == [side] and side != cap_stream.cuda_stream
            for seed in (5, 6):
                x.copy_(torch.randn_like(x))
                new_ids, new_wts = _routing(n, seed=seed)
                ids.copy_(new_ids)
                wts.copy_(new_wts)
                cap.replay()
                got = out.clone()
                ref = mod(x, wts, ids, activation_clamp=CLAMP).clone() + shared(x)
                cap_stream.synchronize()
                assert torch.equal(got, ref), (seed, _rel(got, ref))
    finally:
        torch.cuda.current_stream().wait_stream(cap_stream)
        _current_stream_tls.value = prev


def test_fi_mega_moe_sm107_overlap_tp_sharded_falls_back(sm107_dist, monkeypatch):
    """PP + TP (no SP) keeps the shared MLP TP-sharded; its all-reduce would run
    on the side stream next to the NVSHMEM megakernel: overlap runs as separate.
    (TP > 1 at PP 1 is sequence parallel for the mega backends: overlap kept.)"""
    monkeypatch.setenv("VLLM_FI_MEGA_MOE_SHARED", "overlap")
    sp = _make_module("model.layers.shovsp.ffn.experts", tp=2)
    assert sp.overlaps_shared_experts
    mod = _make_module("model.layers.shovtp2.ffn.experts", tp=2, pp=2)
    assert not mod.overlaps_shared_experts and mod._shared_mode == "separate"
    assert mod._shared_stream is None


# ----------------------------------------------------------------------------
# Shared-expert add inside flashinfer's top-k reduce (separate mode:
# forward(shared_output=); overlap mode: deferred reduce after the join, covered
# by the overlap == serial tests above).
def _fi_fast_reduce(mod) -> bool:
    from vllm.models.deepseek_v4.nvidia.fi_moe import sm107_supports_fused_addend

    kernel, workspace, _ = mod._profiles[mod._profile_caps[0]]
    return sm107_supports_fused_addend(kernel, workspace)


def test_fi_mega_moe_sm107_shared_output_fused_add(sm107_dist, monkeypatch):
    """separate mode: forward(shared_output=s) == forward() + s, bitwise (the
    addend is added inside flashinfer's reduce when supported), eager and under
    CUDA-graph replay."""
    monkeypatch.setenv("VLLM_FI_MEGA_MOE_SHARED", "separate")
    shared = _FakeSharedMLP(seed=51)
    mod, *_ = _build(seed=8, prefix="model.layers.shfu1.ffn.experts")
    assert mod.accepts_shared_output and not mod.overlaps_shared_experts
    if not _fi_fast_reduce(mod):
        pytest.skip("flashinfer without the fused-addend reduce")
    for n in (1, 77, MAX_TOKENS):
        x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16)
        ids, wts = _routing(n, seed=20 + n)
        s = shared(x)
        y = mod(x, wts, ids, activation_clamp=CLAMP, shared_output=s).clone()
        ref = mod(x, wts, ids, activation_clamp=CLAMP).clone() + s
        assert torch.equal(y, ref), (n, _rel(y, ref))
    n = 77
    x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16)
    ids, wts = _routing(n, seed=9)
    s = shared(x)
    mod(x, wts, ids, activation_clamp=CLAMP, shared_output=s)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_graph = mod(x, wts, ids, activation_clamp=CLAMP, shared_output=s)
    x.copy_(torch.randn_like(x))
    s.copy_(shared(x))
    graph.replay()
    torch.accelerator.synchronize()
    got = y_graph.clone()
    ref = mod(x, wts, ids, activation_clamp=CLAMP).clone() + s
    assert torch.equal(got, ref), _rel(got, ref)
    del graph
