# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the FlashInfer MoE-EP adapter on real Blackwell GPUs.

Each test spawns one process per GPU, bootstraps vLLM's expert-parallel
groups, builds a ``FlashInferMoeEp`` adapter from bf16 expert weights at the
MoE geometry of a real model, warms it up and runs a forward whose inputs are
chosen so that every quantization step in the megakernel is exact, for the
CuTeDSL kernel (NVFP4 activations and weights, E4M3 scales per 16) as well as
the DeepGEMM kernel (FP8 activations and FP4 weights, power-of-two scales per
32). The expected output is then known in closed form and compared bit for
bit.

A test that needs no SM100 also checks the adapter's kwargs against the
installed FlashInfer's config types, so config drift fails on any CI runner
with FlashInfer installed.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.distributed.eplb_utils import distributed_run, set_env_vars_and_device
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import FLASHINFER_MOE_EP_CUTEDSL, FLASHINFER_MOE_EP_DEEP_GEMM
from vllm.distributed.parallel_state import ensure_model_parallel_initialized
from vllm.model_executor.layers.fused_moe import flashinfer_moe_ep as fi_ep
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_moe_ep import (
    FlashInferMoeEpExperts,
)
from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (
    FlashInferMoeEp,
    FlashInferMoeEpWeights,
    apply_topk_in_fc1,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm


@dataclass(frozen=True)
class Geometry:
    name: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int


# Routed-expert geometry of the models the branch was evaluated on. DeepGEMM
# keeps one UE8M0 scale byte per 32 elements and requires the per-token scale
# rows to be TMA aligned, so hidden and intermediate sizes must be multiples
# of 512 (FlashInfer's own validation only checks 128); all four qualify.
GEOMETRIES = (
    Geometry("dsv4-pro", 7168, 3072, 384, 6),
    Geometry("dsv4-flash", 4096, 2048, 256, 6),
    Geometry("mistral-small-4", 4096, 2048, 128, 4),
    Geometry("mistral-large-3", 7168, 4096, 128, 4),
)
BACKENDS = (FLASHINFER_MOE_EP_CUTEDSL, FLASHINFER_MOE_EP_DEEP_GEMM)

NUM_TOKENS_PER_RANK = 4
MAX_TOKENS_PER_RANK = 128

# Every block scale below is either 1 or a power of two, so the FP4 codes
# and the FP8 values survive quantization exactly: NVFP4 scales a 16-block by
# absmax / 6 in E4M3, DeepGEMM scales a 32-block by absmax / 6 (FP4) or
# absmax / 448 (FP8) rounded up to a power of two. A 6.0 in a block pins the
# FP4 scale to 1; 2.0, 6.0 and 144 are exact FP8 values at any such scale.
#
# Token t carries X_SIGNAL at hidden column 16 * t and a 6.0 sentinel next to
# it; every weight column the sentinel meets is zero.
X_SIGNAL = 2.0
X_SENTINEL = 6.0
# Gate and up rows are identical, so h = silu(12) * 12 = 144 = 6 * 24
# whichever half the kernel treats as the gate, and 144 requantizes exactly.
W13_DIAG = 6.0
H_SIGNAL = 144.0
# Expert e writes E2M1_CODES[e % 7] * H_SIGNAL into output channel e. The 6.0
# sentinel sits next to a zero intermediate channel and pins the block scale.
W2_SENTINEL = 6.0
E2M1_CODES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _expert_out(expert: int) -> float:
    return E2M1_CODES[expert % len(E2M1_CODES)]


def _topk_weights(top_k: int) -> tuple[float, ...]:
    # The DeepSeek-V4 routing policy applies the router weight before fc2, so
    # weight * 144 must requantize exactly too: powers of two do.
    return tuple(2.0 ** -(k + 1) for k in range(top_k))


def _routing_method(backend: str) -> RoutingMethodType:
    if backend == FLASHINFER_MOE_EP_DEEP_GEMM:
        return RoutingMethodType.DeepseekV4
    return RoutingMethodType.Default


def _routing(
    geometry: Geometry, rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """top_k distinct experts per token, strided across ranks so dispatch goes
    remote."""
    base = (
        torch.arange(NUM_TOKENS_PER_RANK, device=device) + rank * NUM_TOKENS_PER_RANK
    ) % geometry.num_experts
    stride = geometry.num_experts // geometry.top_k
    offsets = torch.arange(geometry.top_k, device=device) * stride
    topk_ids = ((base[:, None] + offsets[None, :]) % geometry.num_experts).to(
        torch.int32
    )
    topk_weights = torch.tensor(
        [_topk_weights(geometry.top_k)] * NUM_TOKENS_PER_RANK,
        dtype=torch.float32,
        device=device,
    )
    return topk_ids, topk_weights


def _hidden_states(geometry: Geometry, device: torch.device) -> torch.Tensor:
    x = torch.zeros(
        NUM_TOKENS_PER_RANK, geometry.hidden_size, dtype=torch.bfloat16, device=device
    )
    for t in range(NUM_TOKENS_PER_RANK):
        x[t, 16 * t] = X_SIGNAL
        x[t, 16 * t + 1] = X_SENTINEL
    return x


def _local_expert_weights(
    geometry: Geometry, rank: int, num_local_experts: int, device: torch.device
) -> FlashInferMoeEpWeights:
    w13 = torch.zeros(
        num_local_experts,
        2 * geometry.intermediate_size,
        geometry.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.zeros(
        num_local_experts,
        geometry.hidden_size,
        geometry.intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    token_columns = [16 * t for t in range(NUM_TOKENS_PER_RANK)]
    for local in range(num_local_experts):
        expert = rank * num_local_experts + local
        w13[local, 0, token_columns] = W13_DIAG
        w13[local, geometry.intermediate_size, token_columns] = W13_DIAG
        w2[local, expert, 0] = _expert_out(expert)
        w2[local, expert, 2] = W2_SENTINEL
    return FlashInferMoeEpWeights(w13=w13, w2=w2)


def _expected_output(
    geometry: Geometry, topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    expected = torch.zeros(
        NUM_TOKENS_PER_RANK,
        geometry.hidden_size,
        dtype=torch.float32,
        device=topk_ids.device,
    )
    for t in range(NUM_TOKENS_PER_RANK):
        for k in range(geometry.top_k):
            expert = int(topk_ids[t, k])
            weight = float(topk_weights[t, k])
            expected[t, expert] += weight * _expert_out(expert) * H_SIGNAL
    return expected.to(torch.bfloat16)


def _moe_config(
    vllm_config: VllmConfig,
    geometry: Geometry,
    world_size: int,
    device: torch.device,
    backend: str,
) -> FusedMoEConfig:
    parallel = FusedMoEParallelConfig.make(
        tp_size_=1,
        pcp_size_=1,
        dp_size_=world_size,
        sp_size_=1,
        vllm_parallel_config=vllm_config.parallel_config,
    )
    return FusedMoEConfig(
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
        hidden_dim=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        num_local_experts=geometry.num_experts // world_size,
        num_logical_experts=geometry.num_experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=_routing_method(backend),
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        moe_backend=backend,
        max_num_tokens=MAX_TOKENS_PER_RANK,
    )


def _check_expert_views_alias_kernel_memory(
    adapter: FlashInferMoeEp,
    geometry: Geometry,
    rank: int,
    num_local_experts: int,
    device: torch.device,
) -> None:
    """EPLB moves experts through ``expert_weight_views``; the kernel must see it.

    Swap local experts 0 and 1 in place on every rank, then route through the
    swapped physical slots: the output must be the one the unswapped logical
    experts would have produced.
    """
    for view in adapter.expert_weight_views():
        first = view[0].clone()
        view[0].copy_(view[1])
        view[1].copy_(first)
    torch.distributed.barrier()

    topk_ids, topk_weights = _routing(geometry, rank, device)
    slot = topk_ids % num_local_experts
    physical_ids = torch.where(slot == 0, topk_ids + 1, topk_ids)
    physical_ids = torch.where(slot == 1, topk_ids - 1, physical_ids)
    output = adapter(_hidden_states(geometry, device), physical_ids, topk_weights)
    torch.accelerator.synchronize()
    assert torch.equal(output, _expected_output(geometry, topk_ids, topk_weights))


def _init_warmup_forward(
    env: dict[str, str], world_size: int, geometry: Geometry, backend: str
) -> None:
    set_env_vars_and_device(env)
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.kernel_config.moe_backend = backend

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        moe = _moe_config(vllm_config, geometry, world_size, device, backend)
        supported, reason = FlashInferMoeEpExperts.is_supported_config(
            FlashInferMoeEpExperts,
            moe,
            kMxfp4Static,
            None,
            mk.FusedMoEActivationFormat.Standard,
        )
        assert supported, reason

        adapter = FlashInferMoeEp(
            moe,
            _local_expert_weights(geometry, rank, moe.num_local_experts, device),
            apply_topk_in_fc1=apply_topk_in_fc1(moe),
        )
        try:
            adapter.warmup()

            topk_ids, topk_weights = _routing(geometry, rank, device)
            output = adapter(_hidden_states(geometry, device), topk_ids, topk_weights)
            torch.accelerator.synchronize()

            expected = _expected_output(geometry, topk_ids, topk_weights)
            assert output.dtype == expected.dtype and output.shape == expected.shape
            assert torch.equal(output, expected), (
                f"rank {rank}: max abs diff "
                f"{(output.float() - expected.float()).abs().max().item()}, "
                f"nonzero output channels per token "
                f"{[torch.nonzero(row).flatten().tolist() for row in output]}"
            )

            if backend == FLASHINFER_MOE_EP_CUTEDSL:
                _check_expert_views_alias_kernel_memory(
                    adapter, geometry, rank, moe.num_local_experts, device
                )
        finally:
            adapter.destroy()


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100) or fi_ep.is_sm107_device(),
    reason="These MXFP4 cases need SM100/SM103 (SM107 runs only NVFP4 experts)",
)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=[g.name for g in GEOMETRIES])
@pytest.mark.parametrize("backend", BACKENDS, ids=("cutedsl", "deep_gemm"))
@pytest.mark.parametrize("world_size", [2, 4])
def test_init_warmup_forward_is_exact_on_grid_inputs(
    world_size: int, backend: str, geometry: Geometry
):
    pytest.importorskip("flashinfer.moe_ep")
    if backend == FLASHINFER_MOE_EP_DEEP_GEMM and not has_deep_gemm():
        pytest.skip("DeepGEMM is not available")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    distributed_run(_init_warmup_forward, world_size, geometry, backend)


@pytest.mark.parametrize(
    ("backend", "sm107"),
    [
        (FLASHINFER_MOE_EP_CUTEDSL, False),
        (FLASHINFER_MOE_EP_DEEP_GEMM, False),
        (FLASHINFER_MOE_EP_CUTEDSL, True),
    ],
    ids=("cutedsl", "deep_gemm", "cutedsl-sm107"),
)
def test_adapter_kwargs_match_installed_flashinfer(
    monkeypatch: pytest.MonkeyPatch, backend: str, sm107: bool
):
    """The adapter's kwargs construct the installed FlashInfer's config types.

    Needs no SM100, so config kwarg drift fails in CI rather than at model load.
    Fakes only the megakernel layer (symmetric memory), the EP group, the device
    index, the DeepGEMM import and the SM107 device probe, so the SM100 and the
    SM107 (Rubin) branches both run on any device.
    """
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    if sm107 and not hasattr(moe_ep, "Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig"):
        pytest.skip("The installed FlashInfer has no SM107 NVFP4 MegaMoE")
    layer_args: list[tuple] = []
    workspaces: list[int] = []

    class FakeMegaLayer:
        def __init__(self, *args) -> None:
            layer_args.append(args)

        def create_workspace(self, capacity: int) -> int:
            workspaces.append(capacity)
            return capacity

        def destroy(self) -> None:
            pass

    monkeypatch.setattr(moe_ep, "MoEEpMegaLayer", FakeMegaLayer)
    monkeypatch.setattr(fi_ep, "is_sm107_device", lambda: sm107)
    monkeypatch.setattr(fi_ep, "_expose_deep_gemm_to_flashinfer", lambda: None)
    monkeypatch.setattr(
        fi_ep,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0, device_group=None),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

    geometry = GEOMETRIES[0]
    moe = SimpleNamespace(
        moe_backend=backend,
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
        hidden_dim=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        max_num_tokens=MAX_TOKENS_PER_RANK,
        swiglu_limit=10.0,
        routing_method=_routing_method(backend),
    )
    weight = torch.zeros(1, dtype=torch.bfloat16)
    FlashInferMoeEp(
        moe,
        FlashInferMoeEpWeights(w13=weight, w2=weight),
        apply_topk_in_fc1=apply_topk_in_fc1(moe),
    )

    ((_, _, _, mega_config),) = layer_args
    if sm107:
        expected = moe_ep.Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
    elif backend == FLASHINFER_MOE_EP_CUTEDSL:
        expected = moe_ep.Nvfp4CutedslMegaMoeConfig
    else:
        expected = moe_ep.DeepGemmMegaMoeConfig
    assert isinstance(mega_config.megakernel, expected)
    # SM107 creates one workspace per capacity profile; SM100 uses the layer's.
    assert workspaces == (
        fi_ep.sm107_capacity_profiles(MAX_TOKENS_PER_RANK) if sm107 else []
    )


# GLM-5.3-Flash routed experts: the SM107 MegaMoE path with a non-DSv4 router
# (grouped sigmoid top-k with correction bias), clamped SwiGLU (limit 10) and
# routing weights applied after FC2. The tests below fake FlashInfer's
# megakernel layer, so they run on any device.
GLM53 = Geometry("glm-5.3-flash", 4096, 2048, 288, 8)


class _FakeSm107MegaLayer:
    """Records workspaces and launches; the "kernel" doubles its input rows."""

    def __init__(self, *args) -> None:
        self.args = args
        self.workspaces: list[int] = []
        self.calls: list[tuple] = []
        self._staged = None

    def create_workspace(self, capacity: int) -> int:
        self.workspaces.append(capacity)
        return capacity

    def forward(self, tensors, *, workspace, return_workspace_view):
        self.calls.append(("forward", workspace, tensors.hidden_states.shape[0]))
        return tensors.hidden_states * 2

    def stage_inputs(self, tensors, *, workspace) -> None:
        self.calls.append(("stage", workspace, tensors.hidden_states.shape[0]))
        self._staged = tensors.hidden_states

    def compute_staged(self, *, output):
        self.calls.append(("compute",))
        return self._staged * 2

    def destroy(self) -> None:
        pass


def _sm107_glm53_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_num_tokens: int = MAX_TOKENS_PER_RANK,
    dp_size: int = 1,
    shared_mode: str = "separate",
) -> tuple[FlashInferMoeEp, _FakeSm107MegaLayer]:
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    if not hasattr(moe_ep, "Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig"):
        pytest.skip("The installed FlashInfer has no SM107 NVFP4 MegaMoE")
    layers: list[_FakeSm107MegaLayer] = []

    def make_layer(*args):
        layers.append(_FakeSm107MegaLayer(*args))
        return layers[-1]

    monkeypatch.setattr(moe_ep, "MoEEpMegaLayer", make_layer)
    monkeypatch.setattr(moe_ep, "MoEEpTensors", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(fi_ep, "is_sm107_device", lambda: True)
    monkeypatch.setattr(
        fi_ep,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=0, device_group=None),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    moe = SimpleNamespace(
        moe_backend=FLASHINFER_MOE_EP_CUTEDSL,
        num_experts=GLM53.num_experts,
        experts_per_token=GLM53.top_k,
        hidden_dim=GLM53.hidden_size,
        intermediate_size=GLM53.intermediate_size,
        max_num_tokens=max_num_tokens,
        swiglu_limit=10.0,
        routing_method=RoutingMethodType.DeepSeekV3,
        dp_size=dp_size,
    )
    weight = torch.zeros(1, dtype=torch.bfloat16)
    adapter = FlashInferMoeEp(
        moe,
        FlashInferMoeEpWeights(w13=weight, w2=weight),
        apply_topk_in_fc1=apply_topk_in_fc1(moe),
        shared_mode=shared_mode,
    )
    (layer,) = layers
    return adapter, layer


@pytest.mark.parametrize(
    ("max_num_tokens", "capacity_cap"),
    [(MAX_TOKENS_PER_RANK, "0"), (65536, "0"), (65536, "8192")],
    ids=("decode", "agg-uncapped", "agg-capped"),
)
def test_sm107_glm53_config_and_capacity_ladder(
    monkeypatch: pytest.MonkeyPatch, max_num_tokens: int, capacity_cap: str
):
    """GLM-5.3's clamp, top-k and late routing weights reach the SM107 config,
    and VLLM_FI_MEGA_MOE_MAX_CAPACITY caps the workspace ladder."""
    monkeypatch.setenv("VLLM_FI_MEGA_MOE_MAX_CAPACITY", capacity_cap)
    adapter, layer = _sm107_glm53_adapter(monkeypatch, max_num_tokens=max_num_tokens)
    megakernel = layer.args[3].megakernel
    assert megakernel.intermediate_size == GLM53.intermediate_size
    assert megakernel.top_k == GLM53.top_k
    assert megakernel.gate_up_clamp == 10.0
    assert megakernel.apply_topk_in_fc1 is False
    assert megakernel.knobs == "cache"
    cap = int(capacity_cap) or max_num_tokens
    assert layer.workspaces == fi_ep.sm107_capacity_profiles(min(max_num_tokens, cap))
    # apply() copies the result right away, so SM107 returns the workspace view.
    assert adapter._return_workspace_view


def _set_dp_tokens(monkeypatch: pytest.MonkeyPatch, tokens: list[int]) -> None:
    import vllm.forward_context as forward_context

    context = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(tokens))
    )
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)


@pytest.mark.parametrize(
    ("local_tokens", "expected_rows"),
    [(600, [256, 256, 88]), (300, [256, 44, 0])],
    ids=("largest-rank", "smaller-rank"),
)
def test_sm107_steps_above_the_cap_run_rank_consistent_chunks(
    monkeypatch: pytest.MonkeyPatch, local_tokens: int, expected_rows: list[int]
):
    """A step above the largest profile runs ceil(max over DP / cap) launches of
    that profile on every rank (empty ones where a rank runs out of tokens),
    and every row is computed exactly once."""
    monkeypatch.setenv("VLLM_FI_MEGA_MOE_MAX_CAPACITY", "256")
    adapter, layer = _sm107_glm53_adapter(monkeypatch, max_num_tokens=1024, dp_size=2)
    _set_dp_tokens(monkeypatch, [600, 300])
    hidden = torch.randn(local_tokens, GLM53.hidden_size, dtype=torch.bfloat16)
    ids = torch.zeros(local_tokens, GLM53.top_k, dtype=torch.int32)
    weights = torch.ones(local_tokens, GLM53.top_k)

    output = adapter(hidden, ids, weights)

    assert [(c[1], c[2]) for c in layer.calls] == [(256, n) for n in expected_rows]
    torch.testing.assert_close(output, hidden * 2, rtol=0, atol=0)


def test_sm107_pre_shared_mode_waits_for_the_shared_mlp_before_compute(
    monkeypatch: pytest.MonkeyPatch,
):
    """VLLM_FI_MEGA_MOE_SHARED=pre: stage on the main stream, wait for the
    shared MLP's event, then launch the megakernel; the event is consumed."""
    adapter, layer = _sm107_glm53_adapter(monkeypatch, shared_mode="pre")
    waited: list[object] = []
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(wait_event=lambda event: waited.append(event)),
    )
    event = object()
    fi_ep.set_pending_shared_experts_event(event)
    hidden = torch.randn(4, GLM53.hidden_size, dtype=torch.bfloat16)
    ids = torch.zeros(4, GLM53.top_k, dtype=torch.int32)
    weights = torch.ones(4, GLM53.top_k)

    adapter(hidden, ids, weights)
    assert [c[0] for c in layer.calls] == ["stage", "compute"]
    assert waited == [event]

    adapter(hidden, ids, weights)
    assert waited == [event], "the event must be consumed by one launch"


def test_sm107_resolve_pre_shared_mode():
    assert fi_ep.resolve_sm107_shared_mode(
        "pre", replicated_shared=False, has_side_stream=True
    ) == ("pre", None)
    mode, reason = fi_ep.resolve_sm107_shared_mode(
        "pre", replicated_shared=True, has_side_stream=False
    )
    assert mode == "separate" and reason


@pytest.mark.parametrize("mask_padding", [True, False])
@pytest.mark.parametrize("skip_padding_env", ["1", "0"])
def test_kernel_routes_mask_padding_rows(
    monkeypatch: pytest.MonkeyPatch, mask_padding: bool, skip_padding_env: str
):
    """Padding rows (CUDA-graph / DP padding) become masked slots (-1, 0), so
    FlashInfer's route check never sees what a non-DSv4 router computed from
    their garbage hidden states (here NaN weights and duplicate ids)."""
    from vllm.model_executor.layers.fused_moe.experts import (
        flashinfer_moe_ep as experts_mod,
    )

    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", skip_padding_env)
    is_padding = torch.tensor([False, False, True, True, True])  # padded buffer
    monkeypatch.setattr(experts_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        experts_mod,
        "get_forward_context",
        lambda: SimpleNamespace(is_padding=is_padding),
    )
    ids = torch.arange(4 * GLM53.top_k, dtype=torch.int64).view(4, GLM53.top_k)
    ids[2:] = 7  # duplicate ids, as a router may produce on garbage rows
    weights = torch.rand(4, GLM53.top_k)
    weights[2:] = float("nan")

    out_ids, out_weights = experts_mod._kernel_routes(
        ids, weights, mask_padding=mask_padding
    )

    assert out_ids.dtype == torch.int32
    torch.testing.assert_close(out_ids[:2], ids[:2].to(torch.int32))
    torch.testing.assert_close(out_weights[:2], weights[:2])
    if mask_padding and skip_padding_env == "1":
        assert (out_ids[2:] == -1).all()
        assert (out_weights[2:] == 0).all()
    else:
        torch.testing.assert_close(out_ids[2:], ids[2:].to(torch.int32))
        assert out_weights[2:].isnan().all()
