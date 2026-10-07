# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the flashinfer moe_ep backend plumbing.

Everything here runs without a GPU or a flashinfer install: the flashinfer
modules the helpers import lazily are replaced with capture fakes.
"""

import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config.kernel import (
    FLASHINFER_MOE_EP_BACKENDS,
    MEGA_MOE_BACKENDS,
    validate_flashinfer_moe_ep_model,
)
from vllm.utils.flashinfer_moe_ep import (
    _E2M1_LUT,
    FI_MOE_EP_BACKEND_SPECS,
    _dequant_fp4_ue8m0_gran32,
    build_fi_mega_config,
    fi_moe_ep_backend_spec,
    make_fi_moe_ep_bootstrap,
    megakernel_runtime_requirements,
)


@dataclass
class _FakeBootstrapConfig:
    world_size: int
    rank: int
    process_group: Any = None
    auto_bootstrap: bool = True
    device: int | None = field(default=None, kw_only=True)


@dataclass
class _FakeDeepGemmMegaMoeConfig:
    intermediate_size: int
    top_k: int
    activation_clamp: float | None
    fast_math: bool


@dataclass
class _FakeNvfp4CutedslMegaMoeConfig:
    intermediate_size: int
    top_k: int
    activation_clamp: float | None
    fast_math: bool


@dataclass
class _FakeMegaConfig:
    megakernel: Any
    preprocess_weights: bool
    quantize_input: bool


@pytest.fixture
def fake_flashinfer(monkeypatch):
    """Install a minimal fake flashinfer.moe_ep for the lazy imports."""
    moe_ep = ModuleType("flashinfer.moe_ep")
    core = ModuleType("flashinfer.moe_ep.core")
    runtime = ModuleType("flashinfer.moe_ep.core.runtime")
    flashinfer = ModuleType("flashinfer")
    fake_attrs: dict[ModuleType, dict[str, Any]] = {
        moe_ep: {
            "BootstrapConfig": _FakeBootstrapConfig,
            "DeepGemmMegaMoeConfig": _FakeDeepGemmMegaMoeConfig,
            "Nvfp4CutedslMegaMoeConfig": _FakeNvfp4CutedslMegaMoeConfig,
            "MegaConfig": _FakeMegaConfig,
            "core": core,
        },
        runtime: {"TORCH_DIST": "torch_dist", "NVSHMEM": "nvshmem"},
        flashinfer: {"moe_ep": moe_ep},
        core: {"runtime": runtime},
    }
    for mod, attrs in fake_attrs.items():
        for attr, value in attrs.items():
            setattr(mod, attr, value)

    for name, mod in {
        "flashinfer": flashinfer,
        "flashinfer.moe_ep": moe_ep,
        "flashinfer.moe_ep.core": core,
        "flashinfer.moe_ep.core.runtime": runtime,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return moe_ep


def test_fi_backend_strings_are_registered_mega_moe_backends():
    assert set(FI_MOE_EP_BACKEND_SPECS) == FLASHINFER_MOE_EP_BACKENDS
    assert FLASHINFER_MOE_EP_BACKENDS < MEGA_MOE_BACKENDS


@pytest.mark.parametrize("moe_backend", sorted(FLASHINFER_MOE_EP_BACKENDS))
def test_fi_moe_ep_backend_rejected_for_non_dsv4(moe_backend):
    """An FI moe_ep backend with a non-DSv4 model must fail at config time
    instead of silently falling through to the generic FusedMoE path."""
    with pytest.raises(ValueError, match="only supported for DeepSeek-V4"):
        validate_flashinfer_moe_ep_model(moe_backend, ["MixtralForCausalLM"])


@pytest.mark.parametrize("moe_backend", sorted(FLASHINFER_MOE_EP_BACKENDS))
def test_fi_moe_ep_backend_accepted_for_dsv4(moe_backend):
    validate_flashinfer_moe_ep_model(moe_backend, ["DeepseekV4ForCausalLM"])


@pytest.mark.parametrize("moe_backend", sorted(FLASHINFER_MOE_EP_BACKENDS))
@pytest.mark.parametrize("arch", ["DSparkV41DraftModel", "DSparkDraftModel"])
def test_fi_moe_ep_backend_accepted_for_dspark_draft(moe_backend, arch):
    """load_dspark_model builds the draft VllmConfig with the target's
    kernel_config before applying speculative moe_backend, so the DSpark
    draft architectures must pass the FI moe_ep architecture check."""
    validate_flashinfer_moe_ep_model(moe_backend, [arch])


@pytest.mark.parametrize(
    "architectures",
    [["KimiK3ForConditionalGeneration"], ["MixtralForCausalLM"]],
)
def test_native_deep_gemm_mega_moe_not_arch_gated(architectures):
    """VLLM's own deep_gemm mega path is not DSv4-only (Kimi K3 uses it);
    models validate their own constraints at construction time."""
    validate_flashinfer_moe_ep_model("deep_gemm_mega_moe", architectures)


def test_non_fi_backend_ignores_architectures():
    validate_flashinfer_moe_ep_model("auto", ["MixtralForCausalLM"])


@pytest.mark.parametrize("moe_backend", sorted(MEGA_MOE_BACKENDS))
def test_all_mega_backends_get_sequence_parallel_moe(moe_backend):
    """Every mega backend must qualify for sequence-parallel MoE at
    TP>1/EP: the predicate once matched only the native backend string,
    which silently ran the fi backends full-batch with an all-reduce on
    every rank — 0.42-0.65x native e2e at TP8."""
    from vllm.models.deepseek_v4.nvidia.model import _use_sequence_parallel

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            tensor_parallel_size=8,
            data_parallel_size=1,
        ),
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
    )
    assert _use_sequence_parallel(vllm_config)


def test_fi_moe_ep_backend_spec_kernel_and_nvshmem_contract():
    dg = fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_deep_gemm")
    assert dg.megakernel == "deep_gemm_mega"
    assert not dg.needs_nvshmem

    cd = fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_cutedsl")
    assert cd.megakernel == "nvfp4_cutedsl"
    assert cd.needs_nvshmem

    with pytest.raises(ValueError, match="not a flashinfer moe_ep backend"):
        fi_moe_ep_backend_spec("deep_gemm_mega_moe")


def test_megakernel_runtime_requirements(fake_flashinfer):
    dg = megakernel_runtime_requirements(
        fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_deep_gemm")
    )
    assert dg == frozenset({"torch_dist"})

    cd = megakernel_runtime_requirements(
        fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_cutedsl")
    )
    assert cd == frozenset({"torch_dist", "nvshmem"})


def test_bootstrap_pins_the_device_vllm_bound(fake_flashinfer, monkeypatch):
    """The runtime must not rederive the device from LOCAL_RANK/rank: under a
    remapped CUDA_VISIBLE_DEVICES that ordinal points at the wrong GPU
    (CUDA_ERROR_ILLEGAL_ADDRESS in the weight transforms). vLLM passes the
    device it already bound via BootstrapConfig.device."""
    import vllm.utils.flashinfer_moe_ep as mod

    pg = object()
    monkeypatch.setattr(
        mod,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=2, device_group=pg),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)

    bootstrap = make_fi_moe_ep_bootstrap()

    assert bootstrap.world_size == 4
    assert bootstrap.rank == 2
    assert bootstrap.process_group is pg
    assert bootstrap.auto_bootstrap is False
    assert bootstrap.device == 3


def test_build_fi_mega_config_selects_kernel_config(fake_flashinfer):
    dg = build_fi_mega_config(
        intermediate_size=2048,
        top_k=8,
        activation_clamp=7.0,
        megakernel="deep_gemm_mega",
    )
    assert isinstance(dg.megakernel, _FakeDeepGemmMegaMoeConfig)
    assert dg.megakernel.intermediate_size == 2048
    assert dg.megakernel.top_k == 8
    assert dg.megakernel.activation_clamp == 7.0
    assert dg.preprocess_weights and dg.quantize_input

    cd = build_fi_mega_config(
        intermediate_size=2048,
        top_k=8,
        activation_clamp=None,
        megakernel="nvfp4_cutedsl",
    )
    assert isinstance(cd.megakernel, _FakeNvfp4CutedslMegaMoeConfig)

    with pytest.raises(ValueError, match="Unsupported fi_moe_ep megakernel"):
        build_fi_mega_config(
            intermediate_size=2048,
            top_k=8,
            activation_clamp=None,
            megakernel="deep_gemm",
        )


def test_ckpt_uses_nvfp4_experts_reads_moe_quant_algo():
    from vllm.models.deepseek_v4.nvidia.fi_moe import ckpt_uses_nvfp4_experts

    nvfp4 = SimpleNamespace(quant_config=SimpleNamespace(moe_quant_algo="NVFP4"))
    assert ckpt_uses_nvfp4_experts(nvfp4)

    mxfp4 = SimpleNamespace(quant_config=SimpleNamespace(moe_quant_algo=None))
    assert not ckpt_uses_nvfp4_experts(mxfp4)

    no_algo = SimpleNamespace(quant_config=SimpleNamespace())
    assert not ckpt_uses_nvfp4_experts(no_algo)


def test_dequant_fp4_ue8m0_gran32_decodes_lut_and_scales():
    """One 32-element scale group per row: low nibble is the even element,
    high nibble the odd one, ue8m0 scale applies to the whole group."""
    packed = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    sf = torch.tensor([[127], [128]], dtype=torch.uint8)  # 2**0, 2**1

    out = _dequant_fp4_ue8m0_gran32(packed, sf)

    assert out.shape == (2, 32)
    assert out.dtype == torch.bfloat16
    expected = torch.empty(2, 32)
    for row in range(2):
        for col in range(16):
            byte = int(packed[row, col])
            expected[row, 2 * col] = _E2M1_LUT[byte & 0x0F]
            expected[row, 2 * col + 1] = _E2M1_LUT[byte >> 4]
        expected[row] *= 2.0**row
    assert torch.equal(out, expected.to(torch.bfloat16))


# ---------------------------------------------------------------------------
# SM107 (Rubin) MegaMoE plumbing (CPU only; flashinfer replaced by fakes)
# ---------------------------------------------------------------------------


@dataclass
class _FakeSm107Nvfp4Config:
    intermediate_size: int
    top_k: int
    gate_up_clamp: float | None = None
    knobs: Any = None
    kernel_variant: str = "inference"
    input_norm_const: float = 1.0
    max_sm_count: Any = None


@dataclass
class _FakeWeightPack:
    w13: Any
    w2: Any
    w13_scale: Any = None
    w2_scale: Any = None


@pytest.fixture
def fake_flashinfer_sm107(fake_flashinfer):
    fake_flashinfer.Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig = (
        _FakeSm107Nvfp4Config
    )
    fake_flashinfer.MoEWeightPack = _FakeWeightPack
    return fake_flashinfer


def test_sm107_capability_is_per_backend():
    from vllm.utils.flashinfer_moe_ep import SM107_CAPABILITY

    assert SM107_CAPABILITY in fi_moe_ep_backend_spec(
        "flashinfer_moe_ep_mega_cutedsl"
    ).capabilities
    # The flashinfer DeepGEMM wrapper is not validated on Rubin.
    assert SM107_CAPABILITY not in fi_moe_ep_backend_spec(
        "flashinfer_moe_ep_mega_deep_gemm"
    ).capabilities


@pytest.mark.parametrize(
    "moe_backend,cc,nvfp4,expected",
    [
        ("flashinfer_moe_ep_mega_cutedsl", (10, 7), True, "sm107_nvfp4_nvfp4_bf16_cutedsl"),
        ("flashinfer_moe_ep_mega_cutedsl", (10, 0), True, "nvfp4_cutedsl"),
        ("flashinfer_moe_ep_mega_cutedsl", (10, 0), False, "nvfp4_cutedsl"),
        ("flashinfer_moe_ep_mega_deep_gemm", (10, 0), False, "deep_gemm_mega"),
    ],
)
def test_resolve_fi_megakernel_is_arch_and_ckpt_aware(moe_backend, cc, nvfp4, expected):
    from vllm.utils.flashinfer_moe_ep import resolve_fi_megakernel

    got = resolve_fi_megakernel(moe_backend, nvfp4_checkpoint=nvfp4, capability=cc)
    assert got == expected


def test_resolve_fi_megakernel_sm107_needs_nvfp4_checkpoint():
    """SM107 runs only the NVFP4 megakernel: an MXFP4 expert checkpoint (e.g. a
    DSpark draft without a speculative moe_backend override) fails at layer
    construction with a message naming the DeepGEMM backend."""
    from vllm.utils.flashinfer_moe_ep import resolve_fi_megakernel

    with pytest.raises(ValueError, match="deep_gemm_mega_moe"):
        resolve_fi_megakernel(
            "flashinfer_moe_ep_mega_cutedsl", nvfp4_checkpoint=False, capability=(10, 7)
        )


def test_validate_rejects_deep_gemm_wrapper_on_sm107(monkeypatch):
    import vllm.utils.flashinfer_moe_ep as mod

    monkeypatch.setattr(mod, "_device_capability", lambda: (10, 7))
    cfg = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="flashinfer_moe_ep_mega_deep_gemm"),
        parallel_config=SimpleNamespace(enable_eplb=False),
    )
    with pytest.raises(ValueError, match="only supported on compute capability"):
        mod.validate_fi_moe_ep_config(cfg)
    cfg.kernel_config.moe_backend = "flashinfer_moe_ep_mega_cutedsl"
    mod.validate_fi_moe_ep_config(cfg)  # accepted on SM107


@pytest.mark.parametrize("knobs_env,expected", [(None, "cache"), ("config", None)])
def test_build_fi_mega_config_sm107(fake_flashinfer_sm107, monkeypatch, knobs_env, expected):
    if knobs_env is None:
        monkeypatch.delenv("VLLM_FI_MEGA_MOE_KNOBS", raising=False)
    else:
        monkeypatch.setenv("VLLM_FI_MEGA_MOE_KNOBS", knobs_env)
    nv = build_fi_mega_config(
        intermediate_size=2304,
        top_k=6,
        activation_clamp=10,
        megakernel="sm107_nvfp4_nvfp4_bf16_cutedsl",
        input_norm_const=123.5,
    )
    assert isinstance(nv.megakernel, _FakeSm107Nvfp4Config)
    # The knob cache keys on the exact float clamp.
    assert nv.megakernel.gate_up_clamp == 10.0
    assert isinstance(nv.megakernel.gate_up_clamp, float)
    assert nv.megakernel.input_norm_const == 123.5
    assert nv.megakernel.knobs == expected
    assert nv.preprocess_weights and nv.quantize_input

    # The generic RubinInferenceMegaMoE kernel (the knob cache keys on it).
    assert nv.megakernel.kernel_variant == "inference"

    no_clamp = build_fi_mega_config(
        intermediate_size=2304,
        top_k=6,
        activation_clamp=None,
        megakernel="sm107_nvfp4_nvfp4_bf16_cutedsl",
        sm107_extra={"max_sm_count": 196},
    )
    assert no_clamp.megakernel.gate_up_clamp is None
    assert no_clamp.megakernel.max_sm_count == 196


def test_sm107_variant_env_is_inference_only(monkeypatch):
    """VLLM_FI_MEGA_MOE_SM107_VARIANT: the recipes set "inference"; GenPhase is
    not wired in this FlashInfer, so any other value is rejected up front."""
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_FI_MEGA_MOE_SM107_VARIANT", raising=False)
    assert envs.VLLM_FI_MEGA_MOE_SM107_VARIANT == "inference"
    monkeypatch.setenv("VLLM_FI_MEGA_MOE_SM107_VARIANT", "inference")
    assert envs.VLLM_FI_MEGA_MOE_SM107_VARIANT == "inference"
    monkeypatch.setenv("VLLM_FI_MEGA_MOE_SM107_VARIANT", "genphase")
    with pytest.raises(ValueError):
        _ = envs.VLLM_FI_MEGA_MOE_SM107_VARIANT


def test_megakernel_runtime_requirements_single_rank(fake_flashinfer, monkeypatch):
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    cd = megakernel_runtime_requirements(
        fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_cutedsl")
    )
    assert cd == frozenset()


@pytest.mark.parametrize(
    "max_tokens,spec,expected",
    [
        (16384, "auto", [128, 256, 512, 1024, 2048, 4096, 8192, 16384]),
        (10000, "auto", [128, 256, 512, 1024, 2048, 4096, 8192, 10000]),
        (100, "auto", [100]),
        (16384, "max", [16384]),
        (16384, "512, 4096,99999", [512, 4096, 16384]),
    ],
)
def test_sm107_capacity_profiles(max_tokens, spec, expected):
    from vllm.models.deepseek_v4.nvidia.fi_moe import sm107_capacity_profiles

    assert sm107_capacity_profiles(max_tokens, spec) == expected


@pytest.mark.parametrize(
    "mnbt,sp,spec,expected",
    [
        # TEP2 (EP2 + SP2), mnbt 32768: per-rank tokens <= 16384, so no 32768
        # profile (review_wp7 #1).
        (32768, 2, "auto", [128, 256, 512, 1024, 2048, 4096, 8192, 16384]),
        (32767, 2, "auto", [128, 256, 512, 1024, 2048, 4096, 8192, 16384]),
        (32768, 2, "max", [16384]),
        (32768, 2, "512,16384,32768", [512, 16384]),
        (16384, 1, "auto", [128, 256, 512, 1024, 2048, 4096, 8192, 16384]),
        (300, 4, "auto", [75]),
    ],
)
def test_sm107_capacity_profiles_capped_by_sp(mnbt, sp, spec, expected):
    from vllm.models.deepseek_v4.nvidia.fi_moe import (
        sm107_capacity_profiles,
        sm107_max_tokens_per_rank,
    )

    assert sm107_capacity_profiles(mnbt, spec, sp_size=sp) == expected
    assert sm107_max_tokens_per_rank(mnbt, sp) == -(-mnbt // sp)
    try:  # same ladder as flashinfer's helper (w2f) when it is installed
        from flashinfer.moe_ep import sm107_capacity_ladder
    except ImportError:
        return
    assert list(sm107_capacity_ladder(mnbt, sp_size=sp, spec=spec)) == expected


def test_sm107_rank_fallback_capped_by_sp(monkeypatch):
    """No DP metadata (dp > 1): every rank falls back to the LARGEST profile,
    which under SP is ceil(mnbt / sp), not mnbt (review_w2f #7)."""
    import vllm.forward_context as fc
    from vllm.models.deepseek_v4.nvidia.fi_moe import (
        DeepseekV4MegaMoEExpertsFI,
        sm107_capacity_profiles,
    )

    caps = sm107_capacity_profiles(32768, "auto", sp_size=2)
    fake = SimpleNamespace(
        _dp_size=2,
        _sp_size=2,
        max_num_tokens=32768,
        _profile_caps=caps,
        _profiles={c: ("k", c, "w") for c in caps},
    )
    fake._rank_consistent_num_tokens = (
        lambda n: DeepseekV4MegaMoEExpertsFI._rank_consistent_num_tokens(fake, n)
    )
    monkeypatch.setattr(fc, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        fc, "get_forward_context", lambda: SimpleNamespace(dp_metadata=None)
    )
    assert DeepseekV4MegaMoEExpertsFI._select_profile(fake, 5)[1] == 16384


class _FakeEP:
    def __init__(self, world_size, peer_bad):
        self.world_size = world_size
        self.device_group = object()
        self.peer_bad = peer_bad
        self.calls = []


def test_sm107_check_scales_reduces_validity_before_raising(monkeypatch):
    """review_wp7 #3: the validity flag is MAX-reduced over the EP group
    before any rank raises, so a bad scale on one rank raises on every rank
    instead of leaving the peers blocked in the next all-reduce."""
    import torch.distributed as dist

    import vllm.distributed as vdist
    from vllm.models.deepseek_v4.nvidia.fi_moe import DeepseekV4MegaMoEExpertsFI

    for peer_bad in (False, True):
        ep = _FakeEP(2, peer_bad)

        def fake_all_reduce(t, op=None, group=None, _ep=ep):
            _ep.calls.append(("all_reduce", op, group))
            if _ep.peer_bad:
                t.fill_(1)

        monkeypatch.setattr(vdist, "get_ep_group", lambda _ep=ep: _ep)
        monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
        good = torch.tensor([0.5, 0.25])
        if peer_bad:
            with pytest.raises(ValueError, match="another EP rank"):
                DeepseekV4MegaMoEExpertsFI._check_scales(good, "w13.input_scale")
        else:
            DeepseekV4MegaMoEExpertsFI._check_scales(good, "w13.input_scale")
        assert ep.calls == [("all_reduce", dist.ReduceOp.MAX, ep.device_group)]
    # a locally bad scale: the collective still runs first, then this rank raises
    ep = _FakeEP(2, False)
    monkeypatch.setattr(vdist, "get_ep_group", lambda: ep)
    monkeypatch.setattr(
        dist, "all_reduce", lambda t, op=None, group=None: ep.calls.append("ar")
    )
    for bad in (torch.tensor([0.5, 0.0]), torch.tensor([float("nan"), 1.0])):
        ep.calls.clear()
        with pytest.raises(ValueError, match="this rank"):
            DeepseekV4MegaMoEExpertsFI._check_scales(bad, "w2.input_scale")
        assert ep.calls == ["ar"]
    # world 1: no collective
    ep1 = _FakeEP(1, False)
    monkeypatch.setattr(vdist, "get_ep_group", lambda: ep1)
    DeepseekV4MegaMoEExpertsFI._check_scales(torch.tensor([1.0]), "x")
    assert ep1.calls == []


def test_sm107_nvfp4_epilogue_scalars_trtllm_convention():
    from vllm.models.deepseek_v4.nvidia.fi_moe import sm107_nvfp4_epilogue_scalars

    gate_s2 = torch.tensor([2.0**-13, 2.0**-12])
    w2_s2 = torch.tensor([2.0**-11, 2.0**-10])
    w2_in = torch.tensor([8.1e-4, 3.9e-3])
    fc1_alpha, fc2_alpha, nc = sm107_nvfp4_epilogue_scalars(
        gate_s2, w2_s2, w2_in, input_norm_const=250.0
    )
    torch.testing.assert_close(fc1_alpha, gate_s2 / 250.0)
    torch.testing.assert_close(nc, 1.0 / w2_in)
    # FC2 alpha absorbs the per-expert norm const: w2_s2 / nc.
    torch.testing.assert_close(fc2_alpha, w2_s2 * w2_in)
    torch.testing.assert_close(fc2_alpha * nc, w2_s2)
    with pytest.raises(ValueError, match="w2.input_scale"):
        sm107_nvfp4_epilogue_scalars(
            gate_s2, w2_s2, torch.tensor([0.0, 1.0]), input_norm_const=1.0
        )


def test_sm107_rank_consistent_capacity(monkeypatch):
    """Every EP rank must launch the same capacity profile: DP ranks use the
    step's max token count (SP-sharded), never their local count alone."""
    import vllm.forward_context as fc
    from vllm.models.deepseek_v4.nvidia.fi_moe import DeepseekV4MegaMoEExpertsFI

    caps = [128, 256, 512, 1024, 16384]
    fake = SimpleNamespace(
        _dp_size=4,
        _sp_size=1,
        max_num_tokens=16384,
        _profile_caps=caps,
        _profiles={c: ("k", c, "w") for c in caps},
    )
    fake._rank_consistent_num_tokens = (
        lambda n: DeepseekV4MegaMoEExpertsFI._rank_consistent_num_tokens(fake, n)
    )
    dp = SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([5, 300, 7, 1]))
    monkeypatch.setattr(fc, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        fc, "get_forward_context", lambda: SimpleNamespace(dp_metadata=dp)
    )
    select = DeepseekV4MegaMoEExpertsFI._select_profile
    assert select(fake, 5)[1] == 512  # max over DP ranks = 300
    fake._sp_size = 2
    assert select(fake, 3)[1] == 256  # ceil(300 / 2) = 150
    # No DP metadata: every rank falls back to the largest profile.
    monkeypatch.setattr(
        fc, "get_forward_context", lambda: SimpleNamespace(dp_metadata=None)
    )
    assert select(fake, 5)[1] == 16384
    # DP=1 (EP = TP group under SP): local shards are equal across ranks.
    fake._dp_size = 1
    assert select(fake, 5)[1] == 128
    assert select(fake, 129)[1] == 256
    with pytest.raises(ValueError, match="largest workspace profile"):
        select(fake, 20000)


def test_sm107_rank_local_batch_uses_largest_profile(monkeypatch):
    """DSv4.1 bounded-replay seam graphs are captured per replay size and each
    rank replays the size of its own replay rows, so the capacity profile baked
    into such a graph must be the same for every size: the largest one. A rank
    trimming a prefill to 128 rows (graph 144) and an idle rank replaying the
    whole 3072-token dummy batch must launch the same workspace."""
    import vllm.forward_context as fc
    from vllm.models.deepseek_v4.nvidia.fi_moe import DeepseekV4MegaMoEExpertsFI

    caps = [128, 256, 512, 1024, 2048, 3072]
    fake = SimpleNamespace(
        _dp_size=4,
        _sp_size=1,
        max_num_tokens=3072,
        _profile_caps=caps,
        _profiles={c: ("k", c, "w") for c in caps},
    )
    fake._rank_consistent_num_tokens = (
        lambda n: DeepseekV4MegaMoEExpertsFI._rank_consistent_num_tokens(fake, n)
    )
    select = DeepseekV4MegaMoEExpertsFI._select_profile
    # Seam capture of the 144-row replay graph: all ranks capture size 144 and
    # the batch's DP metadata says 144 everywhere, yet the graph must hold 3072.
    dp = SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([144, 144, 144, 144]))
    ctx = SimpleNamespace(dp_metadata=dp, additional_kwargs={fc.RANK_LOCAL_BATCH: True})
    monkeypatch.setattr(fc, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(fc, "get_forward_context", lambda: ctx)
    assert select(fake, 144)[1] == 3072
    assert select(fake, 6)[1] == 3072
    # The batch graphs (no flag) keep the DP-synced ladder.
    ctx.additional_kwargs = {}
    assert select(fake, 144)[1] == 256
    # A context without additional_kwargs (older producers) behaves as before.
    monkeypatch.setattr(
        fc, "get_forward_context", lambda: SimpleNamespace(dp_metadata=dp)
    )
    assert select(fake, 144)[1] == 256
    # DP=1: every rank holds the same batch; the flag does not matter.
    fake._dp_size = 1
    monkeypatch.setattr(fc, "get_forward_context", lambda: ctx)
    ctx.additional_kwargs = {fc.RANK_LOCAL_BATCH: True}
    assert select(fake, 144)[1] == 256


def test_sm107_fused_shared_add_needs_flashinfer_support_and_env(monkeypatch):
    """The shared-expert add moves into flashinfer's top-k reduce only when the
    backend advertises it (older flashinfer: the elementwise add stays) and
    VLLM_FI_MEGA_MOE_FUSED_SHARED_ADD is not 0."""
    from vllm.models.deepseek_v4.nvidia.fi_moe import sm107_supports_fused_addend

    supported = SimpleNamespace(supports_fused_addend=lambda workspace: True)
    monkeypatch.delenv("VLLM_FI_MEGA_MOE_FUSED_SHARED_ADD", raising=False)
    assert sm107_supports_fused_addend(supported, object())
    assert not sm107_supports_fused_addend(SimpleNamespace(), object())
    monkeypatch.setenv("VLLM_FI_MEGA_MOE_FUSED_SHARED_ADD", "0")
    assert not sm107_supports_fused_addend(supported, object())


def test_sm107_shared_mode_resolution():
    from vllm.models.deepseek_v4.nvidia.fi_moe import (
        resolve_sm107_shared_mode,
        sm107_max_sm_count,
    )

    ok = dict(replicated_shared=True, has_side_stream=True)
    assert resolve_sm107_shared_mode("overlap", **ok) == ("overlap", None)
    mode, why = resolve_sm107_shared_mode(
        "overlap", replicated_shared=False, has_side_stream=True
    )
    assert mode == "separate" and "tensor-parallel" in why
    assert sm107_max_sm_count("auto", mode) is None
    mode, _ = resolve_sm107_shared_mode(
        "overlap", replicated_shared=True, has_side_stream=False
    )
    assert mode == "separate"
    assert resolve_sm107_shared_mode(
        "separate", replicated_shared=False, has_side_stream=False
    ) == ("separate", None)


def test_mega_moe_padding_keeps_one_route_per_rank():
    """SM107 megakernel deadlock (p3mega): an EP rank whose whole batch is
    padding must still stage one routed row; partially padded batches mask
    exactly the padding rows, as before."""
    from vllm.models.deepseek_v4.nvidia.fi_moe import apply_mega_moe_routing_preprocess

    ids = torch.arange(24, dtype=torch.int32).view(4, 6)
    all_pad = torch.ones(4, dtype=torch.bool)
    out = apply_mega_moe_routing_preprocess(ids, is_padding=all_pad)
    assert (out == -1).all()  # legacy: zero routes on this rank
    out = apply_mega_moe_routing_preprocess(ids, is_padding=all_pad, keep_first_row=True)
    assert torch.equal(out[0], ids[0]) and (out[1:] == -1).all()
    tail_pad = torch.tensor([False, False, True, True])
    for keep in (False, True):
        out = apply_mega_moe_routing_preprocess(
            ids, is_padding=tail_pad, keep_first_row=keep
        )
        assert torch.equal(out[:2], ids[:2]) and (out[2:] == -1).all()
    assert apply_mega_moe_routing_preprocess(ids, keep_first_row=True) is ids


def test_sm107_max_sm_count_spec():
    from vllm.models.deepseek_v4.nvidia.fi_moe import sm107_max_sm_count

    assert sm107_max_sm_count("auto", "separate") is None
    auto = sm107_max_sm_count("auto", "overlap")
    assert auto is not None
    assert sm107_max_sm_count("0", "overlap") is None
    assert sm107_max_sm_count("180", "separate") == 180
    assert sm107_max_sm_count("16384:148, 2048:180", "overlap") == (
        (2048, 180),
        (16384, 148),
    )
