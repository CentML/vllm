# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import functools
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from vllm.config import get_current_vllm_config
from vllm.config.kernel import (
    FLASHINFER_MOE_EP_BACKENDS,
    FLASHINFER_MOE_EP_CUTEDSL,
    FLASHINFER_MOE_EP_DEEP_GEMM,
)
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kNvfp4Static,
    require_finite_positive_scales,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _import_deep_gemm

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)

FlashInferMoeEpKernel = Literal["cutedsl", "deep_gemm"]
FlashInferMoeEpWeightFormat = Literal["nvfp4", "mxfp4"]
_WEIGHT_FORMATS: dict[QuantKey | None, FlashInferMoeEpWeightFormat] = {
    kNvfp4Static: "nvfp4",
    kMxfp4Static: "mxfp4",
}

_SM100_FAMILY_CAPABILITIES = frozenset({(10, 0), (10, 3)})
# Rubin (SM107) runs only the CuTeDSL backend: FlashInfer's
# sm107_nvfp4_nvfp4_bf16_cutedsl megakernel on the NVFP4 expert checkpoint
# (MXFP4 experts run on SM107 with moe_backend=deep_gemm_mega_moe).
SM107_CAPABILITY = (10, 7)
SM107_NVFP4_MEGAKERNEL = "sm107_nvfp4_nvfp4_bf16_cutedsl"
_SM107_NVFP4_CONFIG = "Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig"
# Smallest workspace capacity profile (tokens per rank) of the SM107 ladder.
_SM107_MIN_CAPACITY = 128


def _device_capability() -> tuple[int, int] | None:
    capability = current_platform.get_device_capability()
    if capability is None:
        return None
    return (capability.major, capability.minor)


@functools.cache
def flashinfer_has_sm107_nvfp4_megakernel() -> bool:
    """Whether the installed FlashInfer ships the SM107 NVFP4 MegaMoE.

    vLLM pins a FlashInfer release without it, so it is probed, not assumed.
    """
    try:
        import flashinfer.moe_ep as moe_ep
    except Exception:
        return False
    return hasattr(moe_ep, _SM107_NVFP4_CONFIG)


def is_sm107_device() -> bool:
    return _device_capability() == SM107_CAPABILITY


def supports_current_device() -> bool:
    capability = _device_capability()
    if capability in _SM100_FAMILY_CAPABILITIES:
        return True
    return capability == SM107_CAPABILITY and flashinfer_has_sm107_nvfp4_megakernel()


# SM budget of the SM107 megakernel when the shared MLP overlaps it: the
# persistent kernel is enqueued first and the shared MLP runs on the remaining
# SMs. 196 of VR200's SMs was the best single budget for DSv4.1 at 1260-2520
# tokens/rank (true EP4), measured on the DSv4.1 branch.
SM107_OVERLAP_SM_BUDGET = 196


def sm107_max_sm_count(spec: str | None, shared_mode: str) -> int | None:
    """``VLLM_FI_MEGA_MOE_MAX_SM_COUNT`` -> FlashInfer ``max_sm_count``.

    ``auto`` (also an empty value): ``SM107_OVERLAP_SM_BUDGET`` in overlap
    mode, else all SMs (None). ``0`` / ``none``: all SMs. ``<n>``: n SMs.
    """
    spec = (spec or "").strip().lower() or "auto"
    if spec == "auto":
        return SM107_OVERLAP_SM_BUDGET if shared_mode == "overlap" else None
    if spec in ("0", "none"):
        return None
    try:
        value = int(spec)
    except ValueError:
        raise ValueError(
            "VLLM_FI_MEGA_MOE_MAX_SM_COUNT must be 'auto', 'none' or an SM "
            f"count, got {spec!r}"
        ) from None
    if value < 0:
        raise ValueError(f"max SM count must be >= 0, got {value}")
    return value or None


def resolve_sm107_shared_mode(
    requested: str, *, replicated_shared: bool, has_side_stream: bool
) -> tuple[str, str | None]:
    """Effective ``VLLM_FI_MEGA_MOE_SHARED`` mode and the reason for a fallback.

    ``overlap`` needs a replicated shared MLP (sequence parallel or TP 1): a
    tensor-parallel one would be reduced across ranks while the NVSHMEM
    megakernel runs. A fallback runs ``separate``, with no megakernel SM budget
    unless ``VLLM_FI_MEGA_MOE_MAX_SM_COUNT`` is an explicit count.
    """
    if requested != "overlap":
        return requested, None
    if not replicated_shared:
        return "separate", "the shared MLP is tensor-parallel (no SP, TP > 1)"
    if not has_side_stream:
        return "separate", "no CUDA side stream"
    return "overlap", None


# Side stream of VLLM_FI_MEGA_MOE_SHARED=overlap, private to it (not the global
# aux_stream()). The shared MLP's output is allocated on it and read on the main
# stream after the join without record_stream (which would hold the block until
# the end of each CUDA-graph capture). Reusing that block is safe because every
# later use of this stream first waits on a fork event recorded on the main
# stream after the consumer (the runner's add).
_SHARED_EXPERTS_STREAM: torch.cuda.Stream | None = None


def _shared_experts_stream() -> torch.cuda.Stream:
    global _SHARED_EXPERTS_STREAM
    if _SHARED_EXPERTS_STREAM is None:
        _SHARED_EXPERTS_STREAM = torch.cuda.Stream()
    return _SHARED_EXPERTS_STREAM


def sm107_capacity_profiles(max_num_tokens: int, sp_size: int = 1) -> list[int]:
    """Workspace capacity ladder (tokens per rank) of the SM107 megakernel.

    Powers of two from 128 below the per-rank cap ``ceil(max_num_tokens /
    sp_size)``, plus the cap (sequence-parallel shards are equal, so a rank
    never holds more than the cap). Each profile runs the knobs tuned for its
    size.
    """
    if max_num_tokens <= 0:
        raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}")
    cap = -(-int(max_num_tokens) // max(int(sp_size), 1))
    caps = {cap}
    profile = _SM107_MIN_CAPACITY
    while profile < cap:
        caps.add(profile)
        profile *= 2
    return sorted(caps)


_E2M1_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@dataclass(frozen=True)
class FlashInferMoeEpBackendSpec:
    kernel: FlashInferMoeEpKernel
    weight_formats: frozenset[FlashInferMoeEpWeightFormat]
    capabilities: frozenset[tuple[int, int]] = _SM100_FAMILY_CAPABILITIES


@dataclass(frozen=True)
class FlashInferMoeEpWeights:
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class FlashInferMoeEpEpilogue:
    input_norm_const: float = 1.0
    fc1_alpha: torch.Tensor | None = None
    fc2_alpha: torch.Tensor | None = None
    fc1_norm_const: torch.Tensor | None = None


_BACKEND_SPECS = {
    FLASHINFER_MOE_EP_CUTEDSL: FlashInferMoeEpBackendSpec(
        kernel="cutedsl",
        weight_formats=frozenset({"nvfp4", "mxfp4"}),
        capabilities=_SM100_FAMILY_CAPABILITIES | {SM107_CAPABILITY},
    ),
    FLASHINFER_MOE_EP_DEEP_GEMM: FlashInferMoeEpBackendSpec(
        kernel="deep_gemm",
        weight_formats=frozenset({"mxfp4"}),
    ),
}

assert frozenset(_BACKEND_SPECS) == FLASHINFER_MOE_EP_BACKENDS


def is_flashinfer_moe_ep_backend(moe_backend: str) -> bool:
    return moe_backend in _BACKEND_SPECS


def flashinfer_moe_ep_backend_spec(
    moe_backend: str,
) -> FlashInferMoeEpBackendSpec:
    try:
        return _BACKEND_SPECS[moe_backend]
    except KeyError:
        raise ValueError(
            f"{moe_backend!r} is not a FlashInfer MoE-EP backend; expected "
            f"one of {sorted(_BACKEND_SPECS)}"
        ) from None


def uses_sm107_megakernel(moe_backend: str) -> bool:
    """Whether ``moe_backend`` runs FlashInfer's SM107 NVFP4 megakernel here."""
    spec = _BACKEND_SPECS.get(moe_backend)
    return spec is not None and spec.kernel == "cutedsl" and is_sm107_device()


def flashinfer_moe_ep_unsupported_reasons(
    moe: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> tuple[str, ...]:
    """Deployment constraints of the megakernel beyond the framework's checks.

    ``activation_key`` is ``None`` for MXFP4 checkpoints (the kernel quantizes
    bf16 activations itself) but marks a W4A16 recipe for NVFP4 weights.
    """
    spec = flashinfer_moe_ep_backend_spec(moe.moe_backend)
    unsupported: list[str] = []
    weight_format = _WEIGHT_FORMATS.get(weight_key)
    if weight_format is None:
        unsupported.append(f"weight scheme {weight_key}")
    elif weight_format not in spec.weight_formats:
        unsupported.append(f"{weight_format.upper()} weights")
    if weight_key == kNvfp4Static and activation_key is None:
        unsupported.append("A16 activations with NVFP4 weights")
    if moe.skip_final_all_reduce:
        unsupported.append("skip_final_all_reduce")
    if moe.in_dtype != torch.bfloat16:
        unsupported.append(f"activation dtype {moe.in_dtype}")
    if moe.has_bias:
        unsupported.append("expert bias")
    if moe.swiglu_alpha is not None or moe.swiglu_beta is not None:
        unsupported.append("custom SwiGLU alpha or beta")
    if (
        spec.kernel == "deep_gemm"
        and moe.routing_method is not RoutingMethodType.DeepseekV4
    ):
        unsupported.append(f"routing method {moe.routing_method.name}")
    on_sm107 = is_sm107_device()
    if on_sm107:
        if SM107_CAPABILITY not in spec.capabilities:
            unsupported.append(f"compute capability 10.7 ({spec.kernel} kernel)")
        if weight_format == "mxfp4":
            unsupported.append(
                "MXFP4 weights on SM107 (the SM107 megakernel consumes the NVFP4 "
                "expert checkpoint; run MXFP4 experts with "
                "moe_backend=deep_gemm_mega_moe)"
            )

    vllm_config = get_current_vllm_config()
    if vllm_config.weight_transfer_config is not None:
        unsupported.append("runtime weight transfer")
    if vllm_config.parallel_config.enable_dbo:
        unsupported.append("dual batch overlap")
    # SM107: the kernel-resident weight aliases have not been validated
    # against expert rebalancing, and the epilogue scalars are derived ones.
    if (spec.kernel == "deep_gemm" or on_sm107) and (
        vllm_config.parallel_config.enable_eplb
    ):
        unsupported.append("EPLB")
    return tuple(unsupported)


def validate_flashinfer_moe_ep_layer(layer: RoutedExperts) -> None:
    if layer.expert_map_manager.placement_strategy != "linear":
        raise ValueError(
            f"{layer.moe_config.moe_backend} requires contiguous linear "
            "expert placement"
        )


def apply_topk_in_fc1(
    moe: FusedMoEConfig,
    *,
    apply_router_weight_on_input: bool = False,
) -> bool:
    return (
        apply_router_weight_on_input
        or moe.routing_method is RoutingMethodType.DeepseekV4
    )


def _dequant_mxfp4_ue8m0_gran32(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    raw = packed.view(torch.uint8)
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=raw.device)
    values = torch.empty(
        raw.shape[0],
        raw.shape[1] * 2,
        dtype=torch.float32,
        device=raw.device,
    )
    values[:, ::2] = lut[(raw & 0x0F).to(torch.int64)]
    values[:, 1::2] = lut[(raw >> 4).to(torch.int64)]
    encoded_scale = (
        scale
        if scale.dtype == torch.float8_e8m0fnu
        else scale.view(torch.float8_e8m0fnu)
    )
    decoded_scale = encoded_scale.float()
    return (values * decoded_scale.repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def _dequant_mxfp4_experts(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    num_experts, output_size, packed_input_size = weight.shape
    result = torch.empty(
        num_experts,
        output_size,
        packed_input_size * 2,
        dtype=torch.bfloat16,
        device=weight.device,
    )
    for expert_id in range(num_experts):
        result[expert_id] = _dequant_mxfp4_ue8m0_gran32(
            weight[expert_id], scale[expert_id]
        )
    return result


def mxfp4_moe_ep_weights(
    moe_backend: str,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> FlashInferMoeEpWeights:
    spec = flashinfer_moe_ep_backend_spec(moe_backend)
    if spec.kernel == "deep_gemm":
        return FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale)
    return FlashInferMoeEpWeights(
        w13=_dequant_mxfp4_experts(w13, w13_scale),
        w2=_dequant_mxfp4_experts(w2, w2_scale),
    )


def fold_nvfp4_gate_up_weight_scale_2(
    w13_weight_scale: torch.Tensor,
    w13_weight_scale_2: torch.Tensor,
    *,
    strict: bool,
) -> bool:
    """Give gate (w1) and up (w3) one per-expert NVFP4 global scale, in place.

    The megakernel takes one fc1 alpha per expert, but the checkpoint carries a
    ``weight_scale_2`` for each of gate and up. Where they differ, the up/gate
    ratio is folded into the up half's E4M3 block scales and the up global
    scale is set to the gate's. The fold is applied only when it round-trips
    exactly for every such expert; otherwise nothing changes. Then ``strict``
    (the SM107 megakernel, which has no other way to honour both scales)
    raises, since a lossy rescale would silently change the model; without it
    the caller keeps the gate value, as vLLM's other NVFP4 MoE backends do.

    ``w13_weight_scale``: ``[E, 2 * I, K / 16]`` E4M3 (gate rows first);
    ``w13_weight_scale_2``: ``[E, 2]`` fp32 (gate, up). Returns whether any
    expert was folded.
    """
    scale = w13_weight_scale.data
    scale_2 = w13_weight_scale_2.data
    if scale_2.dim() != 2 or scale_2.shape[1] != 2:
        raise ValueError(
            f"expected [E, 2] gate/up weight_scale_2, got {tuple(scale_2.shape)}"
        )
    gate_s2 = scale_2[:, 0].float()
    up_s2 = scale_2[:, 1].float()
    if strict:
        require_finite_positive_scales("w1 weight_scale_2", gate_s2)
        require_finite_positive_scales("w3 weight_scale_2", up_s2)
    differ = (gate_s2 != up_s2).nonzero().flatten().tolist()
    if not differ:
        return False
    inter = scale.shape[1] // 2
    folded: dict[int, torch.Tensor] = {}
    for expert in differ:
        gate, up = float(gate_s2[expert]), float(up_s2[expert])
        if math.isfinite(gate) and math.isfinite(up) and gate > 0 and up > 0:
            block = scale[expert, inter:].float() * (up_s2[expert] / gate_s2[expert])
            block_e4m3 = block.to(torch.float8_e4m3fn)
            if torch.equal(block_e4m3.float(), block):
                folded[expert] = block_e4m3
                continue
        if strict:
            raise ValueError(
                "NVFP4 MoE cannot merge the gate and up weight_scale_2 values "
                f"of expert {expert}: their ratio is not exactly representable "
                "in the float8_e4m3fn block scales. Use a checkpoint quantized "
                "with a shared gate/up weight_scale_2."
            )
        return False
    for expert, block_e4m3 in folded.items():
        scale[expert, inter:] = block_e4m3.to(scale.dtype)
        scale_2[expert, 1] = scale_2[expert, 0]
    logger.info_once(
        "FlashInfer MoE-EP: folded the up/gate NVFP4 weight_scale_2 ratio into "
        "the up block scales (exact)."
    )
    return True


def _ep_global_max(local: torch.Tensor) -> torch.Tensor:
    """``max(local)`` MAX-reduced over the EP group, as an fp32 ``[1]`` tensor.

    A non-finite local entry makes it +inf on every rank, so callers validate
    the reduced value and all ranks agree.
    """
    local = local.float()
    amax = local.amax().reshape(1).clone()
    if not torch.isfinite(local).all():
        amax.fill_(math.inf)
    ep_group = get_ep_group()
    if ep_group.world_size > 1:
        import torch.distributed as dist

        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=ep_group.device_group)
    return amax


def sm107_nvfp4_epilogue(
    epilogue: FlashInferMoeEpEpilogue,
    *,
    w13_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
) -> FlashInferMoeEpEpilogue:
    """SM107 NVFP4 epilogue scalars (TensorRT-LLM convention).

    The SM107 megakernel quantizes the bf16 tokens with one global scale and
    requantizes the FC1 output with a per-expert norm constant. With the unit
    constants of the SM100 path, the FC1-output block scales of DSv4.1's
    small-activation layers underflow E4M3 (``w2.input_scale`` is about 1e-4
    in some layers), so the checkpoint's calibrated input scales are used:

    - ``input_norm_const = 1 / max(w13.input_scale)``;
    - ``fc1_norm_const = 1 / max(w2.input_scale)``;
    - ``fc2_alpha = w2.weight_scale_2 * max(w2.input_scale)``;
    - ``fc1_alpha = gate weight_scale_2 / input_norm_const``.

    Both maxima run over every expert of the layer (EP MAX all-reduce of the
    local maxima): tokens are quantized on their source rank and consumed on
    the expert's rank, and one grid per layer keeps the numerics independent
    of the EP size. Only those maxima enter the numerics: every input-scale
    entry must be finite and each maximum positive, checked on the reduced
    values so every rank agrees. Collective: call on every EP rank in the same
    layer order.
    """
    if epilogue.fc1_alpha is None or epilogue.fc2_alpha is None:
        raise ValueError("the SM107 megakernel needs the NVFP4 weight global scales")
    gate_s2 = epilogue.fc1_alpha.float().reshape(-1)
    w2_s2 = epilogue.fc2_alpha.float().reshape(-1)
    if gate_s2.shape != w2_s2.shape:
        raise ValueError(
            "per-expert NVFP4 scalars must all have shape [E_local]: got "
            f"{tuple(gate_s2.shape)} and {tuple(w2_s2.shape)}"
        )
    w13_in = _ep_global_max(w13_input_scale)
    w2_in = _ep_global_max(w2_input_scale)
    require_finite_positive_scales("w13_input_scale (layer max)", w13_in)
    require_finite_positive_scales("w2_input_scale (layer max)", w2_in)
    # fp32 reciprocal, as the FusedMoE NVFP4 backends form a1_gscale.
    input_norm_const = float((1.0 / w13_in).item())
    w2_in = w2_in.to(w2_s2.device)
    return FlashInferMoeEpEpilogue(
        input_norm_const=input_norm_const,
        fc1_alpha=(gate_s2 / input_norm_const).contiguous(),
        fc2_alpha=(w2_s2 * w2_in).contiguous(),
        fc1_norm_const=(1.0 / w2_in).expand_as(w2_s2).contiguous(),
    )


class _FlashInferMoeEpApi:
    def __init__(self) -> None:
        try:
            from flashinfer.moe_ep import (
                BootstrapConfig,
                DeepGemmMegaMoeConfig,
                FleetParams,
                MegaConfig,
                MoEEpMegaLayer,
                MoEEpTensors,
                MoEWeightPack,
                Nvfp4CutedslMegaMoeConfig,
            )
        except ImportError as err:
            raise RuntimeError(
                "FlashInfer MoE-EP requires the public API shipped with "
                "FlashInfer 0.7.0 or newer"
            ) from err

        self.BootstrapConfig = BootstrapConfig
        self.DeepGemmMegaMoeConfig = DeepGemmMegaMoeConfig
        self.FleetParams = FleetParams
        self.MegaConfig = MegaConfig
        self.MoEEpMegaLayer = MoEEpMegaLayer
        self.MoEEpTensors = MoEEpTensors
        self.MoEWeightPack = MoEWeightPack
        self.Nvfp4CutedslMegaMoeConfig = Nvfp4CutedslMegaMoeConfig
        # Not in every FlashInfer release (see flashinfer_has_sm107_...).
        import flashinfer.moe_ep as moe_ep

        self.Sm107Nvfp4MegaMoeConfig: Any = getattr(moe_ep, _SM107_NVFP4_CONFIG, None)


def _load_flashinfer_moe_ep_api() -> _FlashInferMoeEpApi:
    return _FlashInferMoeEpApi()


def _expose_deep_gemm_to_flashinfer() -> None:
    deep_gemm = _import_deep_gemm()
    if deep_gemm is None:
        raise RuntimeError(
            "FlashInfer MoE-EP DeepGEMM requires a usable DeepGEMM installation"
        )
    sys.modules.setdefault("deep_gemm", deep_gemm)


class FlashInferMoeEp:
    def __init__(
        self,
        moe: FusedMoEConfig,
        weights: FlashInferMoeEpWeights,
        epilogue: FlashInferMoeEpEpilogue | None = None,
        *,
        apply_topk_in_fc1: bool,
        shared_mode: str = "separate",
        max_sm_count: int | None = None,
    ) -> None:
        spec = flashinfer_moe_ep_backend_spec(moe.moe_backend)
        if spec.kernel == "deep_gemm":
            _expose_deep_gemm_to_flashinfer()
        api = _load_flashinfer_moe_ep_api()
        if epilogue is None:
            epilogue = FlashInferMoeEpEpilogue()
        # SM107 (Rubin): FlashInfer's sm107_nvfp4_nvfp4_bf16_cutedsl, with one
        # workspace per capacity profile (see _select_workspace).
        self._sm107 = spec.kernel == "cutedsl" and is_sm107_device()
        max_tokens_per_rank = moe.max_num_tokens
        capacities: list[int] = []
        sp_size = 1
        if self._sm107:
            if api.Sm107Nvfp4MegaMoeConfig is None:
                raise RuntimeError(
                    f"moe_backend={moe.moe_backend!r} on SM107 needs a FlashInfer "
                    f"build that exports flashinfer.moe_ep.{_SM107_NVFP4_CONFIG}"
                )
            sp_size = max(int(getattr(moe, "sp_size", 1) or 1), 1)
            capacities = sm107_capacity_profiles(moe.max_num_tokens, sp_size)
            max_tokens_per_rank = capacities[-1]
        ep_group = get_ep_group()
        bootstrap = api.BootstrapConfig(
            world_size=ep_group.world_size,
            rank=ep_group.rank_in_group,
            process_group=ep_group.device_group,
            device=torch.accelerator.current_device_index(),
        )
        fleet_params = api.FleetParams(
            num_experts=moe.num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            token_hidden_size=moe.hidden_dim,
        )
        weight_pack = api.MoEWeightPack(
            w13=weights.w13,
            w2=weights.w2,
            w13_scale=weights.w13_scale,
            w2_scale=weights.w2_scale,
        )
        if self._sm107:
            # The knob cache keys on the exact clamp float. knobs="cache": the
            # per-capacity winners of FlashInfer's offline knob cache
            # (FLASHINFER_MOE_EP_KNOB_CACHE), else its per-capacity heuristic,
            # instead of the config's fixed grouped-schedule defaults. The
            # per-expert alphas and fc1_norm_const are staged per call through
            # MoEEpTensors.
            megakernel = api.Sm107Nvfp4MegaMoeConfig(
                intermediate_size=moe.intermediate_size,
                top_k=moe.experts_per_token,
                gate_up_clamp=(
                    float(moe.swiglu_limit) if moe.swiglu_limit is not None else None
                ),
                fast_math=True,
                apply_topk_in_fc1=apply_topk_in_fc1,
                combine_dtype="bf16",
                input_norm_const=float(epilogue.input_norm_const),
                knobs="cache",
                max_sm_count=max_sm_count,
            )
        elif spec.kernel == "cutedsl":
            # In-kernel fc2 reduce keeps its default (off); FlashInfer 0.7.1rc1
            # renamed in_kernel_fc2_reduce to enable_in_kernel_fc2_reduce.
            megakernel = api.Nvfp4CutedslMegaMoeConfig(
                intermediate_size=moe.intermediate_size,
                top_k=moe.experts_per_token,
                gate_up_clamp=moe.swiglu_limit,
                fast_math=True,
                apply_topk_in_fc1=apply_topk_in_fc1,
                combine_dtype="bf16",
                input_norm_const=epilogue.input_norm_const,
                fc1_alpha=None,
                fc2_alpha=None,
                fc1_norm_const=None,
                knobs=None,
            )
        else:
            megakernel = api.DeepGemmMegaMoeConfig(
                intermediate_size=moe.intermediate_size,
                top_k=moe.experts_per_token,
                activation_clamp=moe.swiglu_limit,
                fast_math=True,
            )
        backend = api.MegaConfig(
            megakernel=megakernel,
            quantize_input=True,
            preprocess_weights=True,
        )

        self._moe_ep_tensors_cls = api.MoEEpTensors
        self._mega_layer: Any | None = api.MoEEpMegaLayer(
            bootstrap,
            fleet_params,
            weight_pack,
            backend,
        )
        self._device = weights.w13.device
        self._hidden_size = moe.hidden_dim
        self._max_num_tokens = max_tokens_per_rank
        self._num_experts = moe.num_experts
        self._top_k = moe.experts_per_token
        self._epilogue: FlashInferMoeEpEpilogue | None = epilogue
        self._return_workspace_view = moe.routing_method is RoutingMethodType.DeepseekV4

        # SM107 capacity profiles: capacity -> FlashInfer workspace handle.
        # Workspace creation is an EP collective on the symmetric heap: every
        # rank creates the same profiles in the same order (layers process
        # their weights in model order). FlashInfer pools the workspaces of a
        # capacity across layers.
        self._capacities: tuple[int, ...] = tuple(capacities)
        self._workspaces: dict[int, Any] = {}
        self._dp_size = int(getattr(moe, "dp_size", 1) or 1)
        self._sp_size = sp_size
        for capacity in self._capacities:
            self._workspaces[capacity] = self._mega_layer.create_workspace(capacity)
        # VLLM_FI_MEGA_MOE_SHARED=overlap: forward_overlapping_shared_experts
        # forks the shared MLP onto a private side stream (created here, not
        # inside a CUDA-graph capture).
        self._shared_stream: torch.cuda.Stream | None = None
        self._shared_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        if self._sm107 and shared_mode == "overlap":
            self._shared_stream = _shared_experts_stream()
            self._shared_events = (torch.cuda.Event(), torch.cuda.Event())
        if self._sm107:
            _log_sm107_megakernel(
                moe,
                self._capacities,
                epilogue.input_norm_const,
                shared_expert=shared_mode,
                max_sm_count=max_sm_count,
            )

    @property
    def can_overlap_shared_experts(self) -> bool:
        # The experts own the overlap (FlashInferMoeEpExperts.
        # overlaps_shared_experts); prepare/finalize has nothing to overlap.
        return False

    @property
    def output_is_reduced(self) -> bool:
        return True

    @property
    def topk_indices_dtype(self) -> torch.dtype:
        return torch.int32

    @property
    def is_monolithic(self) -> bool:
        return False

    def _rank_consistent_num_tokens(self, num_tokens: int) -> int:
        """Token count every EP rank agrees on for this step.

        Dispatch and combine address peer workspaces at the same symmetric
        offsets, so every EP rank must launch the same capacity profile.
        Sequence-parallel shards are equal; DP ranks may differ, so use the
        step's maximum over DP ranks (DSv4.1's decoder-replay graphs agree
        their replay size across DP, and their forward context carries the
        DP metadata). Without DP metadata under DP > 1, use the largest
        profile, which every rank selects identically.
        """
        if self._dp_size <= 1:
            return num_tokens
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        context = get_forward_context() if is_forward_context_available() else None
        dp_metadata = context.dp_metadata if context is not None else None
        if dp_metadata is None:
            logger.info_once(
                "FlashInfer SM107 MegaMoE: no DP metadata in the forward "
                "context; using the largest capacity profile."
            )
            return self._capacities[-1]
        n_max = int(dp_metadata.num_tokens_across_dp_cpu.max())
        if self._sp_size > 1:
            n_max = -(-n_max // self._sp_size)
        return max(num_tokens, n_max)

    def _select_workspace(self, num_tokens: int) -> Any | None:
        """Workspace handle of the smallest capacity profile that holds the
        step (SM107), or None for the layer's single workspace (SM100)."""
        if not self._capacities:
            return None
        n = self._rank_consistent_num_tokens(num_tokens)
        for capacity in self._capacities:
            if capacity >= n:
                return self._workspaces[capacity]
        raise ValueError(
            f"FlashInfer MoE-EP got {n} tokens per rank, but the largest "
            f"workspace profile holds {self._capacities[-1]}"
        )

    def _tensors(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Any:
        epilogue = cast(FlashInferMoeEpEpilogue, self._epilogue)
        return self._moe_ep_tensors_cls(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            fc1_alpha=epilogue.fc1_alpha,
            fc2_alpha=epilogue.fc2_alpha,
            fc1_norm_const=epilogue.fc1_norm_const,
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._mega_layer is None:
            raise RuntimeError("FlashInfer MoE-EP layer was destroyed")
        if hidden_states.shape[0] > self._max_num_tokens:
            raise ValueError(
                f"FlashInfer MoE-EP got {hidden_states.shape[0]} tokens, but "
                f"the workspace supports at most {self._max_num_tokens}"
            )
        workspace = self._select_workspace(hidden_states.shape[0])
        tensors = self._tensors(hidden_states, topk_ids, topk_weights)
        return_view = self._return_workspace_view and getattr(
            self._mega_layer, "supports_output_view", False
        )
        if workspace is not None:
            # SM107: the step's capacity profile.
            return self._mega_layer.forward(
                tensors, workspace=workspace, return_workspace_view=return_view
            )
        # SM100/SM103: the layer's single workspace. vLLM's pinned FlashInfer
        # has no workspace argument in MoEEpMegaLayer.forward().
        if return_view:
            return self._mega_layer.forward(tensors, return_workspace_view=True)
        return self._mega_layer(tensors)

    def forward_overlapping_shared_experts(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        run_shared_experts: Callable[[], None],
        *,
        output: torch.Tensor,
    ) -> None:
        """Megakernel on the current stream, shared experts on a side stream.

        The fork is recorded after the input staging and the persistent
        megakernel is enqueued before the shared MLP, so the megakernel takes
        its SMs (``max_sm_count``) first and the shared MLP runs on the rest;
        forking before the gate, TensorRT-LLM's order, lost the SM-budget gain
        at true EP4 on the DSv4.1 branch. The routed output is copied into
        ``output`` on the current stream, then the side stream is joined, so
        the shared experts' output is ready for the caller. The fork and the
        join complete inside this call, so eager mode and CUDA-graph capture
        (FULL or piecewise) behave alike.
        """
        if self._mega_layer is None:
            raise RuntimeError("FlashInfer MoE-EP layer was destroyed")
        if hidden_states.shape[0] > self._max_num_tokens:
            raise ValueError(
                f"FlashInfer MoE-EP got {hidden_states.shape[0]} tokens, but "
                f"the workspace supports at most {self._max_num_tokens}"
            )
        if self._shared_stream is None or self._shared_events is None:
            raise RuntimeError(
                "FlashInfer MoE-EP was not built to overlap the shared experts "
                "(VLLM_FI_MEGA_MOE_SHARED=overlap on SM107)"
            )
        logger.info_once(
            "FlashInfer SM107 MegaMoE: shared expert MLP overlapped on a side stream."
        )
        workspace = self._select_workspace(hidden_states.shape[0])
        self._mega_layer.stage_inputs(
            self._tensors(hidden_states, topk_ids, topk_weights), workspace=workspace
        )
        fork, join = self._shared_events
        fork.record()
        # SM107 megakernels return a workspace view (supports_output_view).
        routed = self._mega_layer.compute_staged(output=None)
        with torch.cuda.stream(self._shared_stream):
            fork.wait()
            run_shared_experts()
            join.record()
        output.copy_(routed)
        join.wait()

    @torch.inference_mode()
    def warmup(self) -> None:
        if self._mega_layer is None:
            return
        hidden_states = torch.zeros(
            1,
            self._hidden_size,
            dtype=torch.bfloat16,
            device=self._device,
        )
        topk_ids = (
            torch.arange(self._top_k, dtype=torch.int32, device=self._device)
            .mul_(self._num_experts // self._top_k)
            .view(1, self._top_k)
        )
        topk_weights = torch.full(
            (1, self._top_k),
            1.0 / self._top_k,
            dtype=torch.float32,
            device=self._device,
        )
        tensors = self._tensors(hidden_states, topk_ids, topk_weights)
        if not self._workspaces:
            self._mega_layer.warmup(tensors)
            return
        # Compile and launch every capacity profile once (collective), so no
        # lazy compile can fire inside a CUDA-graph capture.
        for capacity in self._capacities:
            self._mega_layer.warmup(tensors, workspace=self._workspaces[capacity])

    def kernel_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Kernel-resident fc1 weight, fc1 scales, fc2 weight, fc2 scales.

        Contiguous per-expert byte tensors that alias the memory the megakernel
        reads, so a caller may permute experts in place (EPLB).
        """
        if self._mega_layer is None:
            raise RuntimeError("FlashInfer MoE-EP layer was destroyed")
        (fc1_weight, fc1_sf), (fc2_weight, fc2_sf) = self._mega_layer._transformed
        return (
            _expert_major_bytes(fc1_weight),
            _expert_major_bytes(fc1_sf),
            _expert_major_bytes(fc2_weight),
            _expert_major_bytes(fc2_sf),
        )

    def expert_weight_views(self) -> tuple[torch.Tensor, ...]:
        """Every per-expert tensor the kernel reads: weights plus epilogue."""
        epilogue = cast(FlashInferMoeEpEpilogue, self._epilogue)
        vectors = (epilogue.fc1_alpha, epilogue.fc2_alpha, epilogue.fc1_norm_const)
        return (*self.kernel_weights(), *(v for v in vectors if v is not None))

    def destroy(self) -> None:
        mega_layer = self._mega_layer
        if mega_layer is None:
            return
        mega_layer.destroy()
        self._mega_layer = None
        self._epilogue = None
        self._workspaces = {}

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()


_SM107_LOGGED: set[tuple] = set()


def _log_sm107_megakernel(
    moe: FusedMoEConfig,
    capacities: tuple[int, ...],
    input_norm_const: float,
    *,
    shared_expert: str,
    max_sm_count: int | None,
) -> None:
    key = (moe.num_experts, moe.experts_per_token, capacities, shared_expert)
    if key in _SM107_LOGGED:  # once per distinct layer config
        return
    _SM107_LOGGED.add(key)
    logger.info(
        "FlashInfer SM107 MegaMoE: megakernel %s, %d of %d experts local, "
        "I %d, top-%d, clamp %s, input_norm_const %.6g, capacity profiles %s, "
        "shared expert %s, megakernel SM budget %s.",
        SM107_NVFP4_MEGAKERNEL,
        getattr(moe, "num_local_experts", moe.num_experts),
        moe.num_experts,
        moe.intermediate_size,
        moe.experts_per_token,
        moe.swiglu_limit,
        input_norm_const,
        list(capacities),
        shared_expert,
        "all" if max_sm_count is None else max_sm_count,
    )


def _expert_major_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Contiguous storage of a per-expert kernel tensor, 1-byte dtypes as uint8.

    The CuTeDSL kernel keeps (E, K, N) transpose views over (E, N, K) storage.
    """
    if not tensor.is_contiguous():
        tensor = tensor.transpose(1, 2)
    if not tensor.is_contiguous():
        raise ValueError(f"expert tensor is not expert-major: {tensor.stride()}")
    return tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor
