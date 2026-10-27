# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    KernelConfig,
    LoadConfig,
    ParallelConfig,
    SpeculativeConfig,
)
from vllm.config.kernel import (
    FLASHINFER_MOE_EP_CUTEDSL,
    FLASHINFER_MOE_EP_DEEP_GEMM,
    PASSTHROUGH_ALL2ALL_BACKEND,
    bind_passthrough_all2all_backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    _get_dspark_parallel_config,
    load_dspark_model,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


@dataclass
class _FakeHFConfig:
    model_type: str


@dataclass
class _FakeModelConfig:
    hf_config: _FakeHFConfig


@dataclass
class _FakeEPLBConfig:
    num_redundant_experts: int = 0


@dataclass
class _FakeParallelConfig:
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 8
    enable_eplb: bool = True
    eplb_config: _FakeEPLBConfig = field(
        default_factory=lambda: _FakeEPLBConfig(num_redundant_experts=32)
    )
    enable_elastic_ep: bool = True

    def __post_init__(self) -> None:
        if not self.enable_eplb and self.eplb_config.num_redundant_experts:
            raise ValueError("redundant experts require EPLB")
        if self.enable_elastic_ep and not self.enable_eplb:
            raise ValueError("elastic EP requires EPLB")


@pytest.mark.parametrize(
    ("model_type", "enable_eplb", "num_redundant_experts", "shares_eplb_config"),
    [
        ("deepseek_v4", True, 32, True),
        ("deepseek_v41", False, 0, False),
    ],
)
def test_dspark_parallel_config_eplb_by_model_type(
    model_type: str,
    enable_eplb: bool,
    num_redundant_experts: int,
    shares_eplb_config: bool,
):
    target_config = _FakeParallelConfig()
    draft_model_config = _FakeModelConfig(
        hf_config=_FakeHFConfig(model_type=model_type)
    )

    draft_config = _get_dspark_parallel_config(
        target_config,
        tensor_parallel_size=4,
        draft_model_config=draft_model_config,
    )

    assert target_config.pipeline_parallel_size == 2
    assert target_config.tensor_parallel_size == 8
    assert target_config.enable_eplb
    assert target_config.eplb_config.num_redundant_experts == 32
    assert target_config.enable_elastic_ep

    assert draft_config is not target_config
    assert draft_config.pipeline_parallel_size == 1
    assert draft_config.tensor_parallel_size == 4
    assert draft_config.enable_eplb is enable_eplb
    assert draft_config.eplb_config.num_redundant_experts == num_redundant_experts
    assert not draft_config.enable_elastic_ep
    if shares_eplb_config:
        assert draft_config.eplb_config is target_config.eplb_config
    else:
        assert draft_config.eplb_config is not target_config.eplb_config


@dataclass
class _FakeSpeculativeConfig:
    moe_backend: str | None
    attention_backend: AttentionBackendEnum | None = None
    kv_cache_dtype: str | None = None
    draft_model_config: _FakeModelConfig = field(
        default_factory=lambda: _FakeModelConfig(_FakeHFConfig("deepseek_v41"))
    )
    draft_parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    draft_load_config: LoadConfig | None = None

    apply_draft_overrides = SpeculativeConfig.apply_draft_overrides


@dataclass
class _FakeVllmConfig:
    kernel_config: KernelConfig
    speculative_config: _FakeSpeculativeConfig
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    load_config: LoadConfig = field(default_factory=LoadConfig)
    quant_config: object = None

    def __post_init__(self) -> None:
        # The all2all binding VllmConfig.__post_init__ runs on every replace().
        bind_passthrough_all2all_backend(
            self.kernel_config.moe_backend, self.parallel_config
        )


def _load_dspark_draft_config(
    monkeypatch, target: _FakeVllmConfig, non_causal: bool = False
):
    class CapturedConfig(Exception):
        pass

    def capture_get_model(*, vllm_config, model_config):
        raise CapturedConfig(vllm_config)

    monkeypatch.setattr("vllm.model_executor.model_loader.get_model", capture_get_model)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.utils.get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
        lambda hf_config: non_causal,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.utils.get_draft_quant_config",
        lambda vllm_config: None,
    )
    with pytest.raises(CapturedConfig) as captured:
        load_dspark_model(object(), target)
    return captured.value.args[0]


AG_RS = "allgather_reducescatter"
PASSTHROUGH = PASSTHROUGH_ALL2ALL_BACKEND
MOE_EP = FLASHINFER_MOE_EP_CUTEDSL


@pytest.mark.parametrize(
    ("target_moe", "draft_moe", "dp_size", "expected_moe", "expected_all2all"),
    [
        ("flashinfer_b12x", None, 2, "flashinfer_b12x", AG_RS),
        ("flashinfer_b12x", "flashinfer_cutlass", 2, "flashinfer_cutlass", AG_RS),
        # A FlashInfer MoE-EP target binds all2all_backend="passthrough", which
        # only a draft on such a backend may keep.
        (MOE_EP, None, 2, MOE_EP, PASSTHROUGH),
        (
            MOE_EP,
            FLASHINFER_MOE_EP_DEEP_GEMM,
            2,
            FLASHINFER_MOE_EP_DEEP_GEMM,
            PASSTHROUGH,
        ),
        # MegaMoE communicates itself; without DP a fused-MoE draft needs no
        # all2all.
        (MOE_EP, "deep_gemm_mega_moe", 2, "deep_gemm_mega_moe", AG_RS),
        (MOE_EP, "flashinfer_cutlass", 1, "flashinfer_cutlass", AG_RS),
        # The draft binds passthrough to its own copy of the parallel config.
        ("flashinfer_b12x", MOE_EP, 2, MOE_EP, PASSTHROUGH),
    ],
    ids=[
        "unset",
        "fused-to-fused",
        "moe-ep-unset",
        "moe-ep-to-moe-ep",
        "moe-ep-to-mega-moe",
        "moe-ep-to-fused-without-dp",
        "fused-to-moe-ep",
    ],
)
def test_dspark_draft_uses_speculative_moe_backend(
    monkeypatch, target_moe, draft_moe, dp_size, expected_moe, expected_all2all
):
    """speculative_config.moe_backend selects the DSpark draft's MoE backend and
    the draft's all2all backend follows it; the target config is unchanged."""
    target = _FakeVllmConfig(
        kernel_config=KernelConfig(moe_backend=target_moe),
        speculative_config=_FakeSpeculativeConfig(moe_backend=draft_moe),
        parallel_config=ParallelConfig(
            data_parallel_size=dp_size, enable_expert_parallel=True
        ),
    )
    target_all2all = target.parallel_config.all2all_backend

    draft = _load_dspark_draft_config(monkeypatch, target)

    assert draft.kernel_config.moe_backend == expected_moe
    assert draft.parallel_config.all2all_backend == expected_all2all
    assert target.kernel_config.moe_backend == target_moe
    assert target.parallel_config.all2all_backend == target_all2all


@pytest.mark.parametrize(
    ("target_moe", "target_all2all", "draft_moe", "error"),
    [
        # The target binds passthrough: no all2all for a fused-MoE draft.
        (MOE_EP, AG_RS, "flashinfer_cutlass", "needs the EP all-to-all"),
        # A self-dispatching draft next to the target's all2all.
        ("flashinfer_b12x", "deepep_low_latency", MOE_EP, "while the target uses"),
    ],
    ids=["moe-ep-to-fused", "deepep-to-moe-ep"],
)
def test_dspark_draft_rejects_moe_backend_the_ep_group_cannot_serve(
    monkeypatch, target_moe, target_all2all, draft_moe, error
):
    """The draft shares the target's EP group and its all2all manager: a draft
    backend that needs another all2all fails at load, not at its first
    forward."""
    target = _FakeVllmConfig(
        kernel_config=KernelConfig(moe_backend=target_moe),
        speculative_config=_FakeSpeculativeConfig(moe_backend=draft_moe),
        parallel_config=ParallelConfig(
            data_parallel_size=2,
            enable_expert_parallel=True,
            all2all_backend=target_all2all,
        ),
    )
    with pytest.raises(ValueError, match=error):
        _load_dspark_draft_config(monkeypatch, target)


def test_dspark_draft_moe_backend_keeps_attention_and_cache_overrides(monkeypatch):
    """The overrides re-apply the attention backend and KV cache dtype DSpark
    resolved, and keep its non-causal attention."""
    target = _FakeVllmConfig(
        kernel_config=KernelConfig(moe_backend="flashinfer_b12x"),
        speculative_config=_FakeSpeculativeConfig(
            moe_backend="flashinfer_cutlass",
            attention_backend=AttentionBackendEnum.TRITON_ATTN,
            kv_cache_dtype="fp8",
        ),
    )

    draft = _load_dspark_draft_config(monkeypatch, target, non_causal=True)

    assert draft.kernel_config.moe_backend == "flashinfer_cutlass"
    assert draft.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    assert draft.attention_config.use_non_causal
    assert draft.cache_config.cache_dtype == "fp8"
    assert target.attention_config.backend is None
    assert not target.attention_config.use_non_causal
    assert target.cache_config.cache_dtype == "auto"


@pytest.mark.parametrize("pcp_size", [1, 4])
@pytest.mark.parametrize("dcp_size", [1, 4])
@pytest.mark.parametrize("use_mla", [False, True])
def test_draft_context_parallelism_without_changing_target(
    monkeypatch, pcp_size, dcp_size, use_mla
):
    target_parallel = ParallelConfig(
        tensor_parallel_size=4,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=16,
        distributed_executor_backend="mp",
    )
    target_config = SimpleNamespace(
        parallel_config=target_parallel,
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(use_mla=use_mla)
        ),
    )

    class CapturedConfig(Exception):
        pass

    def capture_init(self, config, device):
        raise CapturedConfig(config)

    monkeypatch.setattr(DraftModelSpeculator, "__init__", capture_init)
    with pytest.raises(CapturedConfig) as captured:
        DFlashSpeculator(target_config, device=None)
    draft_parallel = captured.value.args[0].parallel_config
    assert draft_parallel.tensor_parallel_size == 4
    assert draft_parallel.prefill_context_parallel_size == 1
    assert draft_parallel.decode_context_parallel_size == (dcp_size if use_mla else 1)
    assert draft_parallel.cp_kv_cache_interleave_size == 16
    assert target_parallel.prefill_context_parallel_size == pcp_size
    assert target_parallel.decode_context_parallel_size == dcp_size
    assert target_parallel.cp_kv_cache_interleave_size == 16


@pytest.mark.parametrize("configured_dcp", [1, 4])
def test_attention_uses_draft_dcp_setting_inside_target_process_group(
    monkeypatch, configured_dcp
):
    import vllm.config as config_module
    from vllm.distributed import parallel_state
    from vllm.v1.attention.backend import AttentionImplBase, AttentionMetadata

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=configured_dcp)
    )
    monkeypatch.setattr(
        config_module, "get_current_vllm_config_or_none", lambda: config
    )

    monkeypatch.setattr(
        parallel_state, "_DCP", SimpleNamespace(world_size=4, rank_in_group=2)
    )
    impl: AttentionImplBase[AttentionMetadata] = AttentionImplBase()
    assert impl.dcp_world_size == configured_dcp
    assert impl.dcp_rank == (0 if configured_dcp == 1 else 2)


@pytest.mark.parametrize("draft_dcp_size", [1, 4])
def test_runner_marks_only_replicated_draft_caches(monkeypatch, draft_dcp_size):
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    monkeypatch.setattr(
        model_runner, "get_kv_cache_spec", lambda _: {"target": spec, "draft": spec}
    )
    runner = object.__new__(model_runner.GPUModelRunner)
    runner.vllm_config = None
    runner.dcp_size = 4
    runner.speculator = object.__new__(DFlashSpeculator)
    runner.speculator.dcp_size = draft_dcp_size
    runner.speculator.draft_attn_layer_names = {"draft"}

    specs = runner.get_kv_cache_spec()

    assert specs["target"].dcp_sharded
    assert specs["draft"].dcp_sharded == (draft_dcp_size == 4)
