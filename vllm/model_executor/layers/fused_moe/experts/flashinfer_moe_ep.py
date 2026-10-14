# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer MoE-EP megakernels as a modular-kernel experts implementation.

The megakernel routes on vLLM's top-k output and then dispatches, computes and
combines in one launch, so it is paired with a pass-through prepare/finalize.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (
    FlashInferMoeEp,
    FlashInferMoeEpEpilogue,
    FlashInferMoeEpWeights,
    apply_topk_in_fc1,
    flashinfer_moe_ep_unsupported_reasons,
    resolve_sm107_shared_mode,
    sm107_max_sm_count,
    sm107_nvfp4_epilogue,
    supports_current_device,
    uses_sm107_megakernel,
    validate_flashinfer_moe_ep_layer,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kNvfp4Static,
    require_finite_positive_scales,
)
from vllm.model_executor.utils import replace_parameter

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

# Layer parameters the megakernel reads. After ``process_weights_after_loading``
# they alias the megakernel's own tensors, so EPLB permutes exactly what it uses.
KERNEL_WEIGHT_NAMES = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")


_supported_weight_quant_schemes: frozenset[QuantKey] = frozenset(
    (kNvfp4Static, kMxfp4Static)
)

_supported_activations: frozenset[MoEActivation] = frozenset((MoEActivation.SILU,))


class FlashInferMoeEpExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self._adapter: FlashInferMoeEp | None = None
        # Shared experts next to the SM107 megakernel (VLLM_FI_MEGA_MOE_SHARED):
        # "separate" runs them serially before the routed experts (the runner),
        # "overlap" on a side stream inside apply_overlapping_shared_experts,
        # with the megakernel capped at max_sm_count SMs.
        self._shared_mode = "separate"
        self._max_sm_count: int | None = None
        if uses_sm107_megakernel(moe_config.moe_backend):
            replicated_shared = (
                moe_config.is_sequence_parallel
                or get_tensor_model_parallel_world_size() == 1
            )
            self._shared_mode, fallback = resolve_sm107_shared_mode(
                envs.VLLM_FI_MEGA_MOE_SHARED,
                replicated_shared=replicated_shared,
                has_side_stream=torch.cuda.is_available(),
            )
            if fallback is not None:
                logger.warning_once(
                    "VLLM_FI_MEGA_MOE_SHARED=overlap runs as separate, with no "
                    "megakernel SM budget unless VLLM_FI_MEGA_MOE_MAX_SM_COUNT is "
                    "an explicit count. Reason: %s.",
                    fallback,
                )
            self._max_sm_count = sm107_max_sm_count(
                envs.VLLM_FI_MEGA_MOE_MAX_SM_COUNT, self._shared_mode
            )

    @property
    def overlaps_shared_experts(self) -> bool:
        return self._shared_mode == "overlap"

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return False, reason
        reasons = flashinfer_moe_ep_unsupported_reasons(
            moe_config, weight_key, activation_key
        )
        if reasons:
            return False, f"kernel does not support {', '.join(reasons)}"
        return True, None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return supports_current_device()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key in _supported_weight_quant_schemes

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in _supported_activations

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return (
            moe_parallel_config.use_ep
            and not moe_parallel_config.use_batched_activation_format
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (0,), (0,), (M, K)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Build the megakernel from the layer's canonical weights.

        The oracle has already put the weights in kernel format; the per-expert
        weight global scales arrive through the quant config and become the
        epilogue alphas. On SM107 the NVFP4 checkpoint's input scales also set
        the activation and FC1-output quantization grids (sm107_nvfp4_epilogue).
        """
        validate_flashinfer_moe_ep_layer(layer)  # type: ignore[arg-type]
        epilogue = epilogue_from_quant_config(self.quant_config)
        if epilogue.fc1_alpha is not None and uses_sm107_megakernel(
            self.moe_config.moe_backend
        ):
            epilogue = sm107_nvfp4_epilogue(
                epilogue,
                w13_input_scale=layer.w13_input_scale,
                w2_input_scale=layer.w2_input_scale,
            )
        adapter = FlashInferMoeEp(
            self.moe_config,
            FlashInferMoeEpWeights(
                w13=layer.w13_weight,
                w2=layer.w2_weight,
                w13_scale=getattr(layer, "w13_weight_scale", None),
                w2_scale=getattr(layer, "w2_weight_scale", None),
            ),
            epilogue,
            apply_topk_in_fc1=apply_topk_in_fc1(
                self.moe_config,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
            ),
            shared_mode=self._shared_mode,
            max_sm_count=self._max_sm_count,
        )
        adapter.warmup()
        for name, tensor in zip(KERNEL_WEIGHT_NAMES, adapter.kernel_weights()):
            replace_parameter(layer, name, tensor)
        self._adapter = adapter

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if self._adapter is None:
            raise RuntimeError("FlashInfer MoE-EP weights have not been processed")
        output.copy_(
            self._adapter(hidden_states, topk_ids.to(torch.int32), topk_weights)
        )

    def apply_overlapping_shared_experts(
        self,
        run_shared_experts: Callable[[], None],
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if self._adapter is None:
            raise RuntimeError("FlashInfer MoE-EP weights have not been processed")
        self._adapter.forward_overlapping_shared_experts(
            hidden_states,
            topk_ids.to(torch.int32),
            topk_weights,
            run_shared_experts,
            output=output,
        )


def epilogue_from_quant_config(
    quant_config: FusedMoEQuantConfig,
) -> FlashInferMoeEpEpilogue:
    """Per-expert epilogue constants from the canonical quant config.

    The weight global scales (``g1_alphas``/``g2_alphas``) become the fc1/fc2
    alphas. Activations are quantized dynamically inside the megakernel, so the
    checkpoint's input scales drop out. MXFP4 checkpoints carry no global scales
    and keep the default epilogue. The alphas alias the layer's parameters, so
    EPLB permutations reach the kernel.
    """
    fc1_alpha, fc2_alpha = quant_config.g1_alphas, quant_config.g2_alphas
    if fc1_alpha is None and fc2_alpha is None:
        return FlashInferMoeEpEpilogue()
    assert fc1_alpha is not None and fc2_alpha is not None
    require_finite_positive_scales("g1_alphas", fc1_alpha)
    require_finite_positive_scales("g2_alphas", fc2_alpha)
    if fc1_alpha.dim() != 1 or fc2_alpha.dim() != 1:
        raise ValueError("the megakernel needs one weight global scale per expert")
    return FlashInferMoeEpEpilogue(
        fc1_alpha=fc1_alpha.float().contiguous(),
        fc2_alpha=fc2_alpha.float().contiguous(),
    )
