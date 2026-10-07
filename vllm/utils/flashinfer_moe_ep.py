# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer ``moe_ep`` helpers for DeepSeek V4 vLLM integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config.kernel import FLASHINFER_MOE_EP_BACKENDS
from vllm.distributed import get_ep_group
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from flashinfer.moe_ep import BootstrapConfig, MoEEpMegaLayer

    from vllm.config import VllmConfig


# Every mega kernel is Blackwell-family-only, so the arch is validated
# against the live device instead of being encoded in the backend name. An
# explicit per-backend allowlist so new archs fail loudly until flashinfer
# supports them. Rubin (SM107) runs only the CuteDSL family: flashinfer's
# sm107 RubinInferenceMegaMoE port (``sm107_*_cutedsl`` megakernels). The
# flashinfer DeepGEMM wrapper stays SM100/SM103 until it is validated on
# Rubin (vLLM's native deep_gemm_mega_moe path covers SM107 today).
_SM100_FAMILY_CAPABILITIES = frozenset({(10, 0), (10, 3)})
SM107_CAPABILITY = (10, 7)
FI_MOE_EP_SUPPORTED_CAPABILITIES = _SM100_FAMILY_CAPABILITIES | {SM107_CAPABILITY}

# SM107 megakernel (flashinfer moe_ep backends/mega/kernel/sm107/*): NVFP4
# experts x NVFP4 activations with per-expert alphas + fc1_norm_const. It needs
# the NVFP4 expert checkpoint; the original MXFP4 checkpoint runs on SM107 with
# moe_backend=deep_gemm_mega_moe.
SM107_NVFP4_MEGAKERNEL = "sm107_nvfp4_nvfp4_bf16_cutedsl"
SM107_MEGAKERNELS = frozenset({SM107_NVFP4_MEGAKERNEL})
# Megakernels that consume an NVFP4 checkpoint (prequantized + alphas).
NVFP4_CKPT_MEGAKERNELS = frozenset({"nvfp4_cutedsl", SM107_NVFP4_MEGAKERNEL})


@dataclass(frozen=True)
class FiMoeEpBackendSpec:
    """Static properties of one ``flashinfer_moe_ep_*`` backend string.

    The backend names the kernel *family*; the arch comes from the device and
    the weight handling from the checkpoint, so this only has to carry which
    megakernel to build and whether it needs NVSHMEM in the runtime set.
    """

    megakernel: str
    needs_nvshmem: bool
    capabilities: frozenset[tuple[int, int]] = _SM100_FAMILY_CAPABILITIES


FI_MOE_EP_BACKEND_SPECS: dict[str, FiMoeEpBackendSpec] = {
    # Consumes an MXFP4 checkpoint verbatim (e2m1 weights + E8M0 per-32
    # scales) -- the same recipe the native deep_gemm mega path uses.
    "flashinfer_moe_ep_mega_deep_gemm": FiMoeEpBackendSpec(
        megakernel="deep_gemm_mega",
        needs_nvshmem=False,
    ),
    # The checkpoint picks the weight path, not the kernel: on SM100 an NVFP4
    # checkpoint is consumed prequantized (no round trip), while MXFP4 weights
    # are dequantized to bf16 and requantized. On SM107 it runs flashinfer's
    # sm107 nvfp4 x nvfp4 megakernel on the prequantized NVFP4 checkpoint only
    # (resolve_fi_megakernel). See
    # models/deepseek_v4/nvidia/fi_moe.py:ckpt_uses_nvfp4_experts.
    "flashinfer_moe_ep_mega_cutedsl": FiMoeEpBackendSpec(
        megakernel="nvfp4_cutedsl",
        needs_nvshmem=True,
        capabilities=FI_MOE_EP_SUPPORTED_CAPABILITIES,
    ),
}

assert set(FI_MOE_EP_BACKEND_SPECS) == FLASHINFER_MOE_EP_BACKENDS

_FI_RUNTIME_HANDLE: Any = None


def is_fi_moe_ep_backend(moe_backend: str) -> bool:
    return moe_backend in FI_MOE_EP_BACKEND_SPECS


def fi_moe_ep_backend_spec(moe_backend: str) -> FiMoeEpBackendSpec:
    try:
        return FI_MOE_EP_BACKEND_SPECS[moe_backend]
    except KeyError:
        raise ValueError(
            f"{moe_backend!r} is not a flashinfer moe_ep backend; expected "
            f"one of {sorted(FI_MOE_EP_BACKEND_SPECS)}"
        ) from None


def _device_capability() -> tuple[int, int] | None:
    capability = current_platform.get_device_capability()
    if capability is None:
        return None
    return (capability.major, capability.minor)


def resolve_fi_megakernel(
    moe_backend: str,
    *,
    nvfp4_checkpoint: bool,
    capability: tuple[int, int] | None = None,
) -> str:
    """Megakernel name for ``moe_backend`` on this device and checkpoint.

    SM100/SM103 keep the backend's single megakernel. On SM107 (Rubin) the
    CuteDSL backend maps to flashinfer's sm107 nvfp4 x nvfp4 kernel, which
    consumes the NVFP4 expert checkpoint as is (no weight re-quantization).
    """
    spec = fi_moe_ep_backend_spec(moe_backend)
    if capability is None:
        capability = _device_capability()
    if capability == SM107_CAPABILITY and spec.megakernel == "nvfp4_cutedsl":
        if not nvfp4_checkpoint:
            raise ValueError(
                f"moe_backend={moe_backend!r} on SM107 needs an NVFP4 expert "
                "checkpoint (quantization_config moe_quant_algo NVFP4). Run "
                "MXFP4 experts with moe_backend=deep_gemm_mega_moe (for a "
                "DSpark draft: speculative_config moe_backend)."
            )
        return SM107_NVFP4_MEGAKERNEL
    return spec.megakernel


def validate_fi_moe_ep_config(vllm_config: VllmConfig) -> None:
    """Config-time checks for the mega-MoE backends, native and flashinfer."""
    moe_backend = vllm_config.kernel_config.moe_backend
    if not is_fi_moe_ep_backend(moe_backend):
        return

    # flashinfer validates the arch too, but not until the layer constructor
    # runs during weight load; check here so the error names the flag the user
    # actually typed.
    cc = _device_capability()
    if cc is not None:
        supported_ccs = fi_moe_ep_backend_spec(moe_backend).capabilities
        if cc not in supported_ccs:
            supported = ", ".join(f"{m}.{n}" for m, n in sorted(supported_ccs))
            raise ValueError(
                f"moe_backend={moe_backend!r} is only supported on compute "
                f"capability {supported}, but this device is {cc[0]}.{cc[1]}."
            )

    if vllm_config.parallel_config.enable_eplb:
        raise NotImplementedError(
            f"EPLB is not supported with moe_backend={moe_backend!r}: the "
            "flashinfer moe_ep experts neither apply the logical-to-physical "
            "expert map nor report per-expert load, so rebalancing would move "
            "weights without moving routing. Use "
            "moe_backend=deep_gemm_mega_moe to run the mega path with EPLB."
        )


def make_fi_moe_ep_bootstrap() -> BootstrapConfig:
    from flashinfer.moe_ep import BootstrapConfig

    ep = get_ep_group()
    return BootstrapConfig(
        world_size=ep.world_size,
        rank=ep.rank_in_group,
        process_group=ep.device_group,
        auto_bootstrap=False,
        # vLLM already bound this worker's (possibly remapped) device;
        # without this the runtime would rebind to cuda:LOCAL_RANK|rank and
        # launch weight transforms against another device's pointers.
        device=torch.accelerator.current_device_index(),
    )


def _mega_no_dist() -> bool:
    return bool(int(os.environ.get("MEGA_NO_DIST", "0")))


def megakernel_runtime_requirements(spec: FiMoeEpBackendSpec) -> frozenset[str]:
    from flashinfer.moe_ep.core.runtime import NVSHMEM, TORCH_DIST

    if spec.needs_nvshmem:
        # flashinfer's single-rank mode: symmetric buffers are plain CUDA
        # tensors and nothing is bootstrapped (no NVSHMEM PE, no extra group).
        if _mega_no_dist():
            return frozenset()
        return frozenset({TORCH_DIST, NVSHMEM})
    return frozenset({TORCH_DIST})


def ensure_fi_moe_ep_runtime(vllm_config: VllmConfig) -> None:
    """Acquire the process-wide flashinfer moe_ep runtime once per worker."""
    global _FI_RUNTIME_HANDLE
    if _FI_RUNTIME_HANDLE is not None:
        return

    from flashinfer.moe_ep import bootstrap_moe_ep_runtime

    bootstrap = make_fi_moe_ep_bootstrap()
    spec = fi_moe_ep_backend_spec(vllm_config.kernel_config.moe_backend)
    if spec.needs_nvshmem and _device_capability() == SM107_CAPABILITY:
        if bootstrap.world_size == 1:
            # A one-rank EP group has no peers: run flashinfer's SM107
            # kernels without NVSHMEM (read at allocation time).
            os.environ.setdefault("MEGA_NO_DIST", "1")
        # flashinfer's SM107 staging validates every route on device (ids in
        # [-1, E), unique per row, finite weights) and raises a sticky
        # device-side assert otherwise. vLLM's routers produce valid routes
        # for real tokens, but the dummy/profiling and CUDA-graph warmup
        # batches run on uninitialized activations whose router weights can
        # be non-finite, which would kill the engine at startup; the native
        # deep_gemm mega path does not validate either. Opt back in with
        # FLASHINFER_SM107_MEGA_CHECK_ROUTES=1.
        os.environ.setdefault("FLASHINFER_SM107_MEGA_CHECK_ROUTES", "0")
    _FI_RUNTIME_HANDLE = bootstrap_moe_ep_runtime(
        bootstrap,
        megakernel_runtime_requirements(spec),
    )


def finalize_fi_moe_ep_runtime() -> None:
    """Release the process-wide flashinfer moe_ep runtime."""
    global _FI_RUNTIME_HANDLE
    if _FI_RUNTIME_HANDLE is None:
        return

    from flashinfer.moe_ep import finalize_moe_ep_runtime

    finalize_moe_ep_runtime(_FI_RUNTIME_HANDLE)
    _FI_RUNTIME_HANDLE = None


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


def _dequant_fp4_ue8m0_gran32(
    packed: torch.Tensor, sf_ue8m0: torch.Tensor
) -> torch.Tensor:
    """[rows, K//2] packed e2m1 + [rows, K//32] ue8m0-uint8 scales -> bf16 [rows, K]."""
    raw = packed.view(torch.uint8)
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=raw.device)
    vals = torch.empty(
        raw.shape[0], raw.shape[1] * 2, dtype=torch.float32, device=raw.device
    )
    vals[:, ::2] = lut[(raw & 0x0F).to(torch.int64)]
    vals[:, 1::2] = lut[(raw >> 4).to(torch.int64)]
    sf = (sf_ue8m0.to(torch.int32) << 23).view(torch.float32)
    return (vals * sf.repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def _dequant_expert_weights_to_bf16(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """[E, N, K//2] fp4 + [E, N, K//32] ue8m0 -> [E, N, K] bf16 (expert loop)."""
    num_experts, n, k_half = weight.shape
    out = torch.empty(
        num_experts, n, k_half * 2, dtype=torch.bfloat16, device=weight.device
    )
    for e in range(num_experts):
        out[e] = _dequant_fp4_ue8m0_gran32(weight[e], scale[e])
    return out


def mega_moe_weight_pack_from_params(
    w13_weight: nn.Parameter,
    w13_weight_scale: nn.Parameter,
    w2_weight: nn.Parameter,
    w2_weight_scale: nn.Parameter,
    *,
    megakernel: str = "deep_gemm_mega",
):
    from flashinfer.moe_ep import MoEWeightPack

    if megakernel == "deep_gemm_mega":
        # Same fp4-e2m1 + ue8m0-per-32 recipe as the native path: pass verbatim,
        # flashinfer runs the identical deep_gemm transform.
        return MoEWeightPack(
            w13=w13_weight.data,
            w2=w2_weight.data,
            w13_scale=w13_weight_scale.data,
            w2_scale=w2_weight_scale.data,
        )
    # The cutedsl kernel quantizes with its own recipe (nvfp4
    # e2m1+e4m3-per-16): dequantize the checkpoint fp4 to bf16 and let the
    # backend preprocess requantize. Double quantization: outputs are close to
    # but not bit-identical with the native path.
    return MoEWeightPack(
        w13=_dequant_expert_weights_to_bf16(w13_weight.data, w13_weight_scale.data),
        w2=_dequant_expert_weights_to_bf16(w2_weight.data, w2_weight_scale.data),
    )


def sm107_mega_knobs() -> dict | str | None:
    """Knob policy for the SM107 megakernels (``VLLM_FI_MEGA_MOE_KNOBS``).

    ``cache`` (default): flashinfer's offline knob cache
    (``FLASHINFER_MOE_EP_KNOB_CACHE``, written by ``python -m
    flashinfer.moe_ep.tune``), falling back to flashinfer's per-capacity
    heuristic on a miss. ``config``: the config dataclass defaults (fixed
    tile/cluster regardless of capacity; debugging only).
    """
    import vllm.envs as envs

    policy = envs.VLLM_FI_MEGA_MOE_KNOBS
    return None if policy == "config" else policy


def build_fi_mega_config(
    *,
    intermediate_size: int,
    top_k: int,
    activation_clamp: float | None,
    megakernel: str,
    input_norm_const: float = 1.0,
    kernel_variant: str = "inference",
    sm107_extra: dict[str, Any] | None = None,
):
    """``kernel_variant``: SM107 kernel variant (``"inference"``, the generic
    RubinInferenceMegaMoE; also a knob-cache key). ``sm107_extra``: extra SM107
    config fields, e.g. ``max_sm_count`` (SM budget of the megakernel next to
    a concurrent shared MLP)."""
    from flashinfer.moe_ep import MegaConfig

    if megakernel == SM107_NVFP4_MEGAKERNEL:
        from flashinfer.moe_ep import Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig

        # The SM107 config names the SwiGLU clamp gate_up_clamp (the knob
        # cache keys on it, so pass the exact float). Per-expert NVFP4
        # alphas / fc1_norm_const are staged per forward via MoEEpTensors.
        common: dict[str, Any] = dict(
            intermediate_size=intermediate_size,
            top_k=top_k,
            gate_up_clamp=(
                float(activation_clamp) if activation_clamp is not None else None
            ),
            knobs=sm107_mega_knobs(),
            kernel_variant=kernel_variant,
        )
        common.update(sm107_extra or {})
        mk = Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            input_norm_const=float(input_norm_const), **common
        )
        return MegaConfig(
            megakernel=mk,
            preprocess_weights=True,
            quantize_input=True,
        )

    from flashinfer.moe_ep import DeepGemmMegaMoeConfig, Nvfp4CutedslMegaMoeConfig

    # fast_math selects approximate exp/rcp in DeepGEMM's fused SwiGLU
    # epilogue; the cutedsl kernels accept it for API parity only.
    if megakernel == "deep_gemm_mega":
        mk = DeepGemmMegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            activation_clamp=activation_clamp,
            fast_math=True,
        )
    elif megakernel == "nvfp4_cutedsl":
        mk = Nvfp4CutedslMegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            activation_clamp=activation_clamp,
            fast_math=True,
        )
    else:
        raise ValueError(f"Unsupported fi_moe_ep megakernel {megakernel!r}")

    return MegaConfig(
        megakernel=mk,
        preprocess_weights=True,
        quantize_input=True,
    )


# All MoE layers share one symmetric workspace, like the native path's
# class-level DeepseekV4MegaMoEExperts._symm_buffer_cache. Without this the
# fi path allocates one symm buffer PER LAYER (43x memory + cold working
# sets); the workspace is stateless across forwards (kernel tail-cleans) and
# layers execute sequentially on one stream, so sharing is safe.
def build_fi_mega_layer(
    bootstrap: BootstrapConfig,
    *,
    vllm_config: VllmConfig,
    num_experts: int,
    max_tokens_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    activation_clamp: float | None,
    weights,
    megakernel: str | None = None,
    input_norm_const: float = 1.0,
    kernel_variant: str = "inference",
    sm107_extra: dict[str, Any] | None = None,
) -> MoEEpMegaLayer:
    from flashinfer.moe_ep import FleetParams, MoEEpLayer

    if megakernel is None:
        megakernel = fi_moe_ep_backend_spec(
            vllm_config.kernel_config.moe_backend
        ).megakernel
    mega_config = build_fi_mega_config(
        intermediate_size=intermediate_size,
        top_k=top_k,
        activation_clamp=activation_clamp,
        megakernel=megakernel,
        input_norm_const=input_norm_const,
        kernel_variant=kernel_variant,
        sm107_extra=sm107_extra,
    )
    layer = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            token_hidden_size=hidden_size,
        ),
        weights=weights,
        backend=mega_config,
    )
    from flashinfer.moe_ep import MoEEpMegaLayer

    if not isinstance(layer, MoEEpMegaLayer):
        raise TypeError(
            f"fi_moe_ep expected MoEEpMegaLayer, got {type(layer).__name__}"
        )
    return layer


__all__ = [
    "FI_MOE_EP_BACKEND_SPECS",
    "FI_MOE_EP_SUPPORTED_CAPABILITIES",
    "NVFP4_CKPT_MEGAKERNELS",
    "SM107_CAPABILITY",
    "SM107_MEGAKERNELS",
    "SM107_NVFP4_MEGAKERNEL",
    "FiMoeEpBackendSpec",
    "build_fi_mega_config",
    "build_fi_mega_layer",
    "ensure_fi_moe_ep_runtime",
    "fi_moe_ep_backend_spec",
    "finalize_fi_moe_ep_runtime",
    "is_fi_moe_ep_backend",
    "make_fi_moe_ep_bootstrap",
    "mega_moe_weight_pack_from_params",
    "megakernel_runtime_requirements",
    "resolve_fi_megakernel",
    "sm107_mega_knobs",
    "validate_fi_moe_ep_config",
]
