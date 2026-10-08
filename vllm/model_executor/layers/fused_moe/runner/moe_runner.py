# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

import vllm.model_executor.layers.fusion.moe_finalize  # noqa: F401
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPMonolithic,
)
from vllm.model_executor.layers.fused_moe.routed_experts import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.fusion.shared_expert_gate import (
    apply_shared_expert_gate,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerName,
    direct_register_custom_op,
)

logger = init_logger(__name__)


def register_layer_for_moe_forward_op(
    vllm_config: VllmConfig,
    layer: "MoERunner",
):
    # For smuggling this layer into the fused moe custom op
    prefix = layer.layer_name
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError("Duplicate layer name: {}".format(prefix))
    compilation_config.static_forward_context[prefix] = layer
    compilation_config.static_all_moe_layers.append(prefix)


def get_layer_from_name(layer_name: str) -> MoERunnerInterface:
    forward_context: ForwardContext = get_forward_context()
    if not _USE_LAYERNAME and layer_name == "from_forward_context":
        all_moe_layers = forward_context.all_moe_layers
        assert all_moe_layers is not None
        moe_layer_index = forward_context.moe_layer_index
        if moe_layer_index >= len(all_moe_layers):
            raise AssertionError(
                "We expected the number of MOE layers in `all_moe_layers` "
                "to be equal to the number of "
                "{vllm.moe_forward, vllm.moe_forward_shared} calls."
            )
        layer_name = all_moe_layers[moe_layer_index]
        forward_context.moe_layer_index += 1
    layer = forward_context.no_compile_layers[layer_name]
    assert isinstance(layer, MoERunnerInterface)
    return layer


# On torch >= 2.11, layer_name is a hoisted LayerName opaque object;
# on older versions it remains a plain str.
if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | LayerName
else:
    _layer_name_type = LayerName if _USE_LAYERNAME else str


@torch.compiler.assume_constant_result
def _resolve_layer_name(layer_name: str | LayerName) -> str:
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(layer_name, LayerName):
        return layer_name.value
    elif isinstance(layer_name, FakeScriptObject):
        return layer_name.real_obj.value
    return layer_name


# Note: _moe_forward and _moe_forward_shared should not contain any
# implementation details, They should merely pass along control to
# the runner's '_forward_impl' method.
# These functions should never be called directly since they do not
# include all the functionality of the MoE layer.
def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return cast(
        torch.Tensor,
        layer._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        ),
    )


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    # `hidden_dim_unpadded > 0` only on the TRT-LLM MXFP4 path, where the
    # real kernel writes narrower than `hidden_states.shape[-1]`. Plumbed
    # as an op arg (not peeked from the layer registry) to keep the fake
    # a pure shape function of its inputs and preserve subgraph dedup.
    if hidden_dim_unpadded > 0:
        return hidden_states.new_empty((*hidden_states.shape[:-1], hidden_dim_unpadded))
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        layer._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        ),
    )


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # `fused_out`: see `_moe_forward_fake` for hidden_dim_unpadded semantics.
    # `shared_out`: matches `shared_experts_input` if provided (latent MoE),
    # else `hidden_states`.
    if hidden_dim_unpadded > 0:
        fused_out = hidden_states.new_empty(
            (*hidden_states.shape[:-1], hidden_dim_unpadded)
        )
    else:
        fused_out = torch.empty_like(hidden_states)
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


# NOTE: `moe_forward` and `moe_forward_shared` being opaque custom ops is a
# load-bearing assumption for the MoE-LoRA dual-stream path.
direct_register_custom_op(
    op_name="moe_forward",
    op_func=_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=_moe_forward_shared,
    fake_impl=_moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _moe_forward_shared_ext(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    act_q: torch.Tensor | None,
    act_scale_swizzled: torch.Tensor | None,
    act_scale_linear: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
    shared_gate_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    quantized_input = (
        None if act_q is None else (act_q, act_scale_swizzled, act_scale_linear)
    )
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        layer._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            quantized_input=quantized_input,
        ),
    )


def _moe_forward_shared_ext_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    act_q: torch.Tensor | None,
    act_scale_swizzled: torch.Tensor | None,
    act_scale_linear: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
    shared_gate_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shared_out, fused_out = _moe_forward_shared_fake(
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
        layer_name,
        hidden_dim_unpadded,
    )
    gate = shared_out.new_empty((*shared_out.shape[:-1], shared_gate_dim))
    return shared_out, gate, fused_out


# Variant of moe_forward_shared for shared experts that can hand work to the
# ops around the MoE (MLPerf submission path; specialized for
# Qwen3.6-35B-A3B). It optionally takes the MoE input pre-quantized to MXFP8
# (e4m3 plus swizzled scales for the shared expert and linear scales for the
# routed experts) and, with shared_gate_dim > 0, returns the shared-expert
# output before its sigmoid gate plus the [tokens, shared_gate_dim] gate logits,
# so the gate multiply can fuse with the ops after the MoE. With
# shared_gate_dim == 0 the shared output is gated and the gate output is empty.
direct_register_custom_op(
    op_name="moe_forward_shared_ext",
    op_func=_moe_forward_shared_ext,
    fake_impl=_moe_forward_shared_ext_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _moe_forward_shared_ext_unfinalized(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    act_q: torch.Tensor | None,
    act_scale_swizzled: torch.Tensor | None,
    act_scale_linear: torch.Tensor | None,
    layer_name: _layer_name_type,
    shared_gate_dim: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    layer._enable_deferred_finalize()
    quantized_input = (
        None if act_q is None else (act_q, act_scale_swizzled, act_scale_linear)
    )
    shared, gate, routed = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | UnfinalizedMoEOutput],
        layer._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            quantized_input=quantized_input,
        ),
    )
    num_tokens = hidden_states.shape[0]
    if not isinstance(routed, UnfinalizedMoEOutput):
        # Only the 0-token call keeps the finalized (empty) form.
        assert num_tokens == 0, "the routed experts did not defer the finalize"
        return (
            shared,
            gate,
            routed.new_empty((0, routed.shape[-1])),
            routed.new_empty((0, top_k)),
            routed.new_empty((0, top_k), dtype=torch.int32),
        )
    permuted = routed.gemm2_permuted
    # Every route lands in a distinct permuted row, so the GEMM2 output has at
    # least tokens * top_k rows; return that leading view (a shape the fake can
    # state) and leave the expert-tile padding rows to be reached via the map.
    assert permuted.shape[0] >= num_tokens * top_k and permuted.is_contiguous()
    return (
        shared,
        gate,
        permuted[: num_tokens * top_k],
        routed.expert_weights,
        routed.expanded_idx_to_permuted_idx,
    )


def _moe_forward_shared_ext_unfinalized_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    act_q: torch.Tensor | None,
    act_scale_swizzled: torch.Tensor | None,
    act_scale_linear: torch.Tensor | None,
    layer_name: _layer_name_type,
    shared_gate_dim: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shared_out, gate, _ = _moe_forward_shared_ext_fake(
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
        act_q,
        act_scale_swizzled,
        act_scale_linear,
        layer_name,
        0,
        shared_gate_dim,
    )
    num_tokens, hidden = hidden_states.shape
    return (
        shared_out,
        gate,
        hidden_states.new_empty((num_tokens * top_k, hidden)),
        hidden_states.new_empty((num_tokens, top_k)),
        hidden_states.new_empty((num_tokens, top_k), dtype=torch.int32),
    )


# moe_forward_shared_ext whose routed output stays unfinalized (MLPerf
# submission path; specialized for Qwen3.6-35B-A3B): it returns the TRT-LLM
# GEMM2 rows ([tokens * top_k, hidden] leading view of the permuted buffer),
# the [tokens, top_k] routing weights and the int32 permute map, for
# vllm::moe_finalize -- which the compiled graph folds into the next norm.
direct_register_custom_op(
    op_name="moe_forward_shared_ext_unfinalized",
    op_func=_moe_forward_shared_ext_unfinalized,
    fake_impl=_moe_forward_shared_ext_unfinalized_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _unpack(
    result: torch.Tensor
    | UnfinalizedMoEOutput
    | tuple[torch.Tensor, torch.Tensor | UnfinalizedMoEOutput],
) -> tuple[torch.Tensor | None, torch.Tensor | UnfinalizedMoEOutput]:
    if isinstance(result, tuple):
        return result
    else:
        return (None, result)


def _shared_ext_path_allowed(shared_experts: torch.nn.Module | None) -> bool:
    # moe_forward_shared_ext only runs through the CUDA custom-op entry and
    # leaves the MoE-LoRA wrapping untouched.
    return (
        shared_experts is not None
        and current_platform.is_cuda()
        and get_current_vllm_config().lora_config is None
    )


def _can_defer_shared_gate(shared_experts: torch.nn.Module | None) -> bool:
    """Whether the shared expert can return its output before a [T, 1] gate."""
    if not _shared_ext_path_allowed(shared_experts):
        return False
    gate = getattr(shared_experts, "expert_gate", None)
    return (
        gate is not None
        and getattr(gate, "output_size", None) == 1
        and callable(getattr(shared_experts, "forward_ungated", None))
    )


def _can_accept_quantized_input(shared_experts: torch.nn.Module | None) -> bool:
    """Whether the shared expert's input linear consumes swizzled MXFP8."""
    if not _shared_ext_path_allowed(shared_experts):
        return False
    linear = getattr(shared_experts, "gate_up_proj", None)
    kernel = getattr(getattr(linear, "quant_method", None), "kernel", None)
    input_quant_key = getattr(kernel, "input_quant_key", None)
    return callable(input_quant_key) and input_quant_key() == kMxfp8Dynamic


class MoERunner(MoERunnerInterface):
    """Standard MoE runner implementation for executing Mixture of Experts layers.

    This is the primary concrete implementation of MoE execution logic, providing
    comprehensive support for standard MoE operations. It handles:
    - Expert routing and token dispatching using various routing strategies
    - Shared experts computation with optional parallel execution using CUDA streams
    - Tensor model parallel and expert parallel operations
    - Multiple quantization methods and optimized kernel selection
    - Both monolithic and decomposed expert execution paths
    - Integration with various parallel execution modes (TP, EP, DP)

    The runner orchestrates the complete MoE forward pass including routing tokens
    to experts, executing expert computations in parallel, and combining results.
    It supports advanced features like overlapped execution of shared experts,
    optimized kernels for different parallel configurations, and seamless
    integration with vLLM's distributed execution framework.

    Eventually, this class may be split into more specialized implementations
    for different configurations (e.g., with/without shared experts, gates, etc.).
    """

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
        enable_dbo: bool = False,
        gate: torch.nn.Module | None = None,
        shared_experts: torch.nn.Module | None = None,
        shared_expert_gate: torch.nn.Module | None = None,
        routed_input_transform: torch.nn.Module | None = None,
        routed_output_transform: torch.nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.routed_output_transform = routed_output_transform
        self.routed_scaling_factor = routed_scaling_factor
        self.gate = gate
        self.shared_expert_gate = shared_expert_gate
        self.routed_experts = routed_experts
        self.enable_dbo = enable_dbo

        # When both gates are present and FSE is enabled, fuse their
        # weight matrices into [num_experts + num_shared, hidden] so one
        # F.linear produces combined logits. The topk kernel can then
        # apply routing softmax and shared expert activation (sigmoid)
        # in a single launch.
        self._fse_fuse_gate = gate is not None and shared_expert_gate is not None
        self._combined_gate_weight: torch.Tensor | None = None

        # The shared expert's sigmoid gate is applied after the MoE op, where
        # the compiled graph fuses it into the next layer's add + RMSNorm.
        self._defer_shared_gate = _can_defer_shared_gate(shared_experts)
        # The op may receive the MoE input pre-quantized to MXFP8 by the
        # preceding norm.
        self.accepts_quantized_input = _can_accept_quantized_input(shared_experts)

        self._shared_experts: SharedExperts | None = None
        if shared_experts is not None:
            can_overlap = lambda: self._quant_method.mk_can_overlap_shared_experts
            self._shared_experts = SharedExperts(
                shared_experts,
                moe_config=moe_config,
                enable_dbo=enable_dbo,
                mk_can_overlap_shared_experts=can_overlap,
                defer_gate=self._defer_shared_gate,
            )

        # Needed for string -> MoERunner layer lookup in custom ops.
        self.layer_name = layer_name

        self._forward_entry = self._select_forward()

        # For smuggling this layer into the fused moe custom op
        register_layer_for_moe_forward_op(get_current_vllm_config(), self)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        return self.routed_experts.load_weights(weights)

    @property
    def _uses_shared_ext_op(self) -> bool:
        return self._defer_shared_gate or self.accepts_quantized_input

    def _routed_finalize_deferred(self) -> bool:
        """Whether the routed experts hand back their output unfinalized.

        Then forward() runs moe_forward_shared_ext_unfinalized plus
        vllm::moe_finalize, which the compiled graph folds into the next
        layer's norm (MLPerf submission path; specialized for Qwen3.6-35B-A3B).
        Requires the shared-ext op with a deferred gate, a single-rank
        monolithic kernel whose experts defer (TRT-LLM MXFP8), and no step
        between the experts and the shared + routed add. A read-only
        predicate, so Dynamo traces it; the op enables the experts' side
        (_enable_deferred_finalize) when it runs, also for a cached graph.
        """
        if not (
            self._defer_shared_gate
            and self._shared_experts is not None
            and current_platform.is_cuda()
            and self.moe_config.in_dtype == torch.bfloat16
            and self.moe_config.tp_size == 1
            and self.moe_config.dp_size == 1
            and self.moe_config.ep_size == 1
            and self.moe_config.pcp_size == 1
            and not self.moe_config.is_sequence_parallel
            and not self.do_naive_dispatch_combine
            and self.routed_input_transform is None
            and self.routed_output_transform is None
            and self.routed_scaling_factor == 1.0
            and not isinstance(self.router, ZeroExpertRouter)
            and self.moe_config.hidden_dim == self.moe_config.hidden_dim_unpadded
        ):
            return False
        quant_method = self._quant_method
        kernel = getattr(quant_method, "moe_kernel", None)
        return (
            quant_method.is_monolithic
            and not quant_method.has_unpadded_output
            and kernel is not None
            and kernel.supports_deferred_moe_finalize()
            and kernel.fused_experts.supports_deferred_finalize()
        )

    def _enable_deferred_finalize(self) -> None:
        """Make the routed experts stop after GEMM2 (idempotent, eager)."""
        kernel = self._quant_method.moe_kernel
        if kernel is None or not self._routed_finalize_deferred():
            raise RuntimeError(f"{self.layer_name} cannot defer the MoE finalize.")
        experts = kernel.fused_experts
        if not experts.moe_config.defer_moe_finalize_local:
            experts.moe_config.defer_moe_finalize_local = True
            assert experts.moe_config.use_deferred_moe_finalize
            logger.info_once(
                "Deferring the routed MoE top-k finalize into its consumer norm."
            )

    def _select_forward(self) -> Callable:
        if current_platform.is_tpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            return _moe_forward if self._shared_experts is None else _moe_forward_shared

        if current_platform.is_cpu():
            # CPU never touches the workspace manager (Monolithic experts
            # skip it entirely; Modular experts' _allocate_buffers bypasses
            # it too, see modular_kernel.py) -- the ContextVar-based lane
            # lookup was the only part of this call graph Dynamo can't
            # trace, so CPU can always call the fused-MoE op directly.
            return _moe_forward if self._shared_experts is None else _moe_forward_shared

        if self._uses_shared_ext_op:
            return torch.ops.vllm.moe_forward_shared_ext

        return (
            torch.ops.vllm.moe_forward
            if self._shared_experts is None
            else torch.ops.vllm.moe_forward_shared
        )

    @property
    def shared_experts(self) -> SharedExperts | None:
        return self._shared_experts

    # TODO(bnell): Temporary hack. Get rid of this.
    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        self.routed_experts._replace_quant_method(quant_method)

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config
        self.routed_experts._set_moe_config(new_moe_config)
        if self._shared_experts is not None:
            self._shared_experts._set_moe_config(new_moe_config)

    def _maybe_fuse_gate_weights(self):
        """Fuse router and shared expert gate weights on first call.

        Cannot be done at __init__ because gate weights are loaded after
        module construction (via weight_loader). Called once from
        _forward_impl before the first forward pass.
        """
        if self._combined_gate_weight is None:
            assert self.gate is not None and self.shared_expert_gate is not None
            self._combined_gate_weight = torch.cat(
                [self.gate.weight, self.shared_expert_gate.weight],
                dim=0,
            )

    @property
    def _quant_method(self) -> FusedMoEMethodBase:
        return self.routed_experts.quant_method

    def apply_routed_input_transform(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply transform for routed experts (e.g., latent projection).

        This is called by MoERunner.forward_native. The original hidden_states
        is saved separately so shared experts get [S, hidden_size] while
        routed experts get the transformed [S, moe_latent_size].

        Returns (possibly transformed) hidden states and the input for shared
        experts (or None if there are no shared experts).
        """
        if self.routed_input_transform is not None:
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear returns (output, extra_bias) tuple.
            # We only need the output tensor; extra_bias is not used here.
            if isinstance(result, tuple):
                return result[0], hidden_states
            return result, hidden_states

        return (
            hidden_states,
            hidden_states if self._shared_experts is not None else None,
        )

    def apply_routed_output_transform(
        self,
        fused_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transform to routed expert output (e.g., latent to full dim).

        Used by latent MoE models (e.g., NemotronH) where routed experts
        operate in a compressed latent space and need projection back to
        the full hidden dimension before combining with shared expert output.
        """
        if self.routed_output_transform is not None:
            r = self.routed_output_transform(fused_output)
            fused_output = r[0] if isinstance(r, tuple) else r
        return fused_output

    def _maybe_apply_routed_scale_to_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Apply routed_scaling_factor to the output with FP16 overflow
        protection.

        Scale the fused expert output by routed_scaling_factor. For FP16,
        avoid overflow by dividing shared_output by the scale instead
        (the decoder layer compensates with matching divisions).
        """
        if self.routed_scaling_factor != 1.0:
            if fused_output.dtype != torch.float16 or shared_output is None:
                fused_output *= self.routed_scaling_factor
            elif shared_output is not None:
                shared_output *= 1.0 / self.routed_scaling_factor
        return shared_output, fused_output

    @property
    def _fused_output_is_reduced(self) -> bool:
        return (
            self._quant_method.moe_kernel is not None
            and self._quant_method.moe_kernel.output_is_reduced()
        )

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool | None = None,
    ) -> torch.Tensor | None:
        """All-reduce shared expert output when the combine kernel already
        reduced fused output.

        * If the combine kernel does the reduction for fused_output, reduce
          shared_output separately. O.w, reduce fused_output+shared_output later.
        * If we have SP (TP=N, DP=M, EP), there is a separate AG step handled
          in the model.
        """
        if fused_output_is_reduced is None:
            fused_output_is_reduced = self._fused_output_is_reduced

        if (
            shared_output is not None
            and not self.moe_config.is_sequence_parallel
            and fused_output_is_reduced
        ):
            shared_output = tensor_model_parallel_all_reduce(shared_output)
        return shared_output

    def _maybe_reduce_routed_output_before_transform(
        self,
        fused_output: torch.Tensor,
        fused_output_is_reduced: bool,
    ) -> tuple[torch.Tensor, bool]:
        """All-reduce latent routed output before its output transform.

        Latent MoE output transforms may contain non-linear ops, e.g. RMSNorm.
        TP partial routed outputs must be summed in latent space before such
        transforms are applied.

        A transform that commutes with the TP sum is exempt: if
        ``sum_r T(x_r) == T(sum_r x_r)``, applying the transform to the local
        partial output and letting the existing late all-reduce sum the
        combined result is equivalent, and costs one collective instead of two.
        Such a transform opts out by setting ``reduce_commutative = True``.
        The default is False, so transforms that do not declare themselves
        keep being reduced early.
        """
        if (
            self.routed_output_transform is not None
            and not getattr(self.routed_output_transform, "reduce_commutative", False)
            and not self.moe_config.is_sequence_parallel
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not fused_output_is_reduced
        ):
            fused_output = tensor_model_parallel_all_reduce(fused_output)
            fused_output_is_reduced = True
        return fused_output, fused_output_is_reduced

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int | None,
        output_is_reduced: bool | None = None,
    ) -> torch.Tensor:
        """All-reduce the combined output if needed.

        This is the "late" all-reduce path. When neither fused nor shared
        output was individually reduced, the combined sum is all-reduced
        here. Skipped when sequence-parallel is active (SP handles its
        own reduction) or when the early path already reduced both outputs.
        """
        # skip_final_all_reduce must not coexist with a pre-reduced fused
        # output. This should be enforced by MoE config initialization.
        if self.moe_config.skip_final_all_reduce:
            assert not self._fused_output_is_reduced, (
                "skip_final_all_reduce requires an un-reduced fused output"
            )

        # We don't need to reduce the final output if:
        # - We are not running with TP or DP
        # - The MK already reduced the fused output itself.
        if output_is_reduced is None:
            output_is_reduced = self._fused_output_is_reduced

        if (
            not self.moe_config.is_sequence_parallel
            and not self.moe_config.skip_final_all_reduce
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not output_is_reduced
        ):
            states = tensor_model_parallel_all_reduce(states)

        return states[..., :trunc_size] if trunc_size is not None else states

    def _encode_layer_name(self) -> str | LayerName:
        if _USE_LAYERNAME:
            return LayerName(self.layer_name)
        # Can be unavailable or None in unittests
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            return "from_forward_context"
        return self.layer_name

    def _maybe_pad_hidden_states(
        self,
        shared_experts_input: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int | None, int | None]:
        """Pad hidden_states to moe_config.hidden_dim and compute the
        original dimension for later truncation.

        For latent MoE, the routed hidden_states may be smaller than
        hidden_dim. Padding ensures uniform tensor sizes through the
        fused MoE kernel. The returned trunc_size is used by
        _maybe_reduce_final_output to strip the padding from the result.
        """
        shared_experts_hidden_dim = (
            shared_experts_input.shape[-1] if shared_experts_input is not None else 0
        )
        transformed_hidden_dim: int | None = hidden_states.shape[-1]
        if (
            not self._quant_method.skip_forward_padding
            and self.moe_config.hidden_dim != transformed_hidden_dim
        ):
            assert transformed_hidden_dim is not None
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        # Truncation sizes for stripping kernel padding from the output.
        # None means no truncation needed (no padding was applied).
        #
        # Two truncation points exist in forward():
        #   pre_xform:  applied to fused_output BEFORE routed_output_transform
        #   post_xform: applied to the final result AFTER all-reduce
        #
        # MoE with routed output transform or shared experts:
        #   - pre_xform applies if the transform needs unpadded routed output
        #     or shared+routed add needs matching hidden dims. For Nemotron-3
        #     Nano, TRTLLM NVFP4 pads routed MoE hidden dim 2688->2816, while
        #     shared output stays 2688.
        #   - post_xform uses shared_experts_hidden_dim when transform and shared
        #     experts make the final output full hidden dim.
        #
        # Standard MoE / MoE without transforms (GPT-OSS, Mixtral):
        #   - pre_xform is None (no early truncation)
        #   - post_xform strips padding after all-reduce (or None if unpadded)
        if transformed_hidden_dim == hidden_states.shape[-1]:
            transformed_hidden_dim = None

        pre_xform_trunc_size = None
        if self.routed_output_transform is not None or shared_experts_hidden_dim > 0:
            pre_xform_trunc_size = transformed_hidden_dim
        post_xform_trunc_size = transformed_hidden_dim
        if self.routed_output_transform is not None and shared_experts_hidden_dim > 0:
            post_xform_trunc_size = shared_experts_hidden_dim

        return hidden_states, pre_xform_trunc_size, post_xform_trunc_size

    def _maybe_apply_shared_experts(
        self,
        shared_experts_input: torch.Tensor | None,
        order: SharedExpertsOrder,
        shared_quantized_input: QuantizedActivation | None = None,
    ):
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts(shared_experts_input, order, shared_quantized_input)

    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        shared_experts_overlapping: bool = False,
        shared_quantized_input: QuantizedActivation | None = None,
        routed_quantized_input: QuantizedActivation | None = None,
        routed_out: UnfinalizedMoEOutput | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | UnfinalizedMoEOutput]:
        """Run expert routing and the fused MoE kernel via the quant method.

        Orchestrates shared expert execution (before/after), expert selection
        via the router, and the actual fused MoE computation. Returns
        (shared_expert_output, fused_expert_output).

        `shared_experts_overlapping` should be True only if using multi-stream
        overlap. Then the shared expert was already launched in a separate
        stream, so the results only have to be awaited here.

        The optional quantized inputs are pre-quantized copies of the MoE input
        (written by the post-attention norm) that replace the shared expert's
        and the monolithic routed experts' own input quantization.
        ``routed_out``, when given, is the routed output already computed by
        the locality MoE kernel.
        """
        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP, shared_quantized_input
        )

        if routed_out is not None:
            fused_out = routed_out
        elif self.routed_experts.quant_method.is_monolithic:
            # Monolithic kernels: pass router_logits to routed_experts
            fused_out = self.routed_experts.forward_monolithic(
                x=(
                    hidden_states
                    if routed_quantized_input is None
                    else routed_quantized_input
                ),
                router_logits=router_logits,
                input_ids=input_ids,
            )
        else:
            # Modular kernels: select experts first, then call routed_experts
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )

            fused_out = self.routed_experts.forward_modular(
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts=self._shared_experts,
                shared_experts_input=shared_experts_input,
            )

        if shared_experts_overlapping:
            assert self._shared_experts is not None
            self._shared_experts.wait()

        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
        )

    def _sequence_parallel_context(self):
        """Return a context manager for sequence-parallel token
        redistribution.

        When sequence parallelism is active, returns a context that handles
        local size tracking for proper token scatter/gather. Otherwise
        returns a no-op context.
        """
        ctx = get_forward_context()
        return (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

    def _maybe_add_zero_expert_output(
        self,
        result: torch.Tensor,
    ) -> torch.Tensor:
        """Add the zero expert's contribution to the final result.

        When a ZeroExpertRouter is used, it computes a bias-like output
        from the "zero expert" that is added to the combined routed+shared
        expert output.
        """
        if isinstance(self.router, ZeroExpertRouter):
            zero_expert_output = self.router.zero_expert_output
            assert zero_expert_output is not None
            result = result + zero_expert_output
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
        quantized_input: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Invoke the fused moe layer.

        Input:
        - hidden_states
        - router_logits

        Output:
        - The new hidden_states.

        Calling sequence
        - forward
          - self._forward_entry (_moe_forward or _moe_forward_shared custom op)
            - _forward_impl

        Note: The existence of _moe_forward and _moe_forward_shared custom ops are due
        to the following reason:
        1. pytorch cannot handle union types in custom op signatures so
           _moe_forward and _moe_forward_shared must be split.

        ``quantized_input`` (only when ``accepts_quantized_input``) is
        ``hidden_states`` quantized to MXFP8: ``(e4m3, F8_128x4 swizzled
        scales, [tokens, hidden/32] linear scales)``.
        """
        # Apply transform for routed experts (e.g., latent projection for
        # latent MoE). When the caller pre-applies the routed input transform
        # outside the runner (e.g. to overlap it on a separate stream), it
        # passes the already-transformed routed input as ``hidden_states`` and
        # the original hidden states as ``shared_experts_input``; skip the
        # transform in that case so shared experts still see the original input.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )

        # Record before `_maybe_pad_hidden_states` pads activations to match
        # `moe_config.hidden_dim`, e.g. after `align_trtllm_fp4_moe_hidden_dim_for_fi`
        # so routed output can be trimmed before
        # shared+routed add / latent up proj if needed.

        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        hidden_dim_unpadded = (
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0
        )
        if self._uses_shared_ext_op:
            act_q, act_sf_swizzled, act_sf_linear = (
                quantized_input
                if quantized_input is not None and self.accepts_quantized_input
                else (None, None, None)
            )
            if self._routed_finalize_deferred():
                shared_output, shared_gate, permuted, weights, permuted_idx = (
                    torch.ops.vllm.moe_forward_shared_ext_unfinalized(
                        hidden_states,
                        router_logits,
                        shared_experts_input,
                        input_ids,
                        act_q,
                        act_sf_swizzled,
                        act_sf_linear,
                        self._encode_layer_name(),
                        1,
                        self.moe_config.experts_per_token,
                    )
                )
                # The compiled graph folds the reduction into the consumer
                # norm together with the shared + routed add.
                fused_output = torch.ops.vllm.moe_finalize(
                    permuted, weights, permuted_idx
                )
            else:
                shared_output, shared_gate, fused_output = self._forward_entry(
                    hidden_states,
                    router_logits,
                    shared_experts_input,
                    input_ids,
                    act_q,
                    act_sf_swizzled,
                    act_sf_linear,
                    self._encode_layer_name(),
                    hidden_dim_unpadded,
                    1 if self._defer_shared_gate else 0,
                )
            if self._defer_shared_gate:
                # Applied before any reduction or scaling, exactly where the
                # shared expert applies it; the compiled graph fuses it into
                # the consumer (the next layer's add + RMSNorm).
                shared_output = apply_shared_expert_gate(shared_gate, shared_output)
        else:
            result = self._forward_entry(
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
                self._encode_layer_name(),
                hidden_dim_unpadded,
            )
            # Extract outputs from result
            shared_output, fused_output = _unpack(result)

        #
        # Note: there are two all-reduce points below. They are mutually
        # exclusive, controlled by _fused_output_is_reduced
        #  - When True: the combine kernel already reduced fused_output,
        #    so we reduce shared_output here to match, then skip the
        #    all-reduce in _maybe_reduce_final_output.
        #  - When False: neither output is reduced yet, so we combine
        #    them first and all-reduce the sum in _maybe_reduce_final_output.
        fused_output = cast(torch.Tensor, fused_output)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        fused_output_is_reduced = self._fused_output_is_reduced

        # Latent routed output has to be reduced before output transform,
        # because the transform may include non-linear normalization.
        fused_output, fused_output_is_reduced = (
            self._maybe_reduce_routed_output_before_transform(
                fused_output,
                fused_output_is_reduced,
            )
        )

        # If routed output is already reduced, reduce shared to match.
        # See note above re: the two all-reduce points.
        shared_output = self._maybe_reduce_shared_expert_output(
            shared_output, fused_output_is_reduced
        )

        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
            shared_output, fused_output
        )

        # Apply output transform (e.g. latent -> full dim)
        fused_output = self.apply_routed_output_transform(fused_output)

        if shared_output is not None:
            result = shared_output + fused_output
        else:
            result = fused_output

        result = self._maybe_reduce_final_output(
            result, og_hidden_dim_post_xform, fused_output_is_reduced
        )

        return self._maybe_add_zero_expert_output(result)

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return (
            self.moe_config.dp_size > 1 or self.moe_config.is_sequence_parallel
        ) and not self._quant_method.supports_internal_mk

    def _maybe_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # For naive dispatch/combine Dp/Ep, dispatch the hidden states and
        # router logits to all experts.
        # NOTE: this will be removed once all kernels are migrated into the
        # MoEKernel framework.
        if self.do_naive_dispatch_combine:
            result = get_ep_group().dispatch_router_logits(
                hidden_states,
                router_logits,
                self.moe_config.is_sequence_parallel,
            )
            assert len(result) == 2
            hidden_states, router_logits = result

        if (
            self.moe_config.pcp_size > 1
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
        ):
            hidden_states = get_pcp_group().all_gather(hidden_states, dim=0)
            router_logits = get_pcp_group().all_gather(router_logits, dim=0)

        return hidden_states, router_logits

    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor | UnfinalizedMoEOutput,
    ) -> (
        torch.Tensor
        | UnfinalizedMoEOutput
        | tuple[torch.Tensor | None, torch.Tensor | UnfinalizedMoEOutput]
    ):
        if self.do_naive_dispatch_combine:
            if isinstance(hidden_states, UnfinalizedMoEOutput):
                raise RuntimeError(
                    "Naive expert-parallel combine cannot consume a deferred "
                    "MoE output."
                )
            hidden_states = get_ep_group().combine(
                hidden_states,
                self.moe_config.is_sequence_parallel,
            )

        if (
            self.moe_config.pcp_size > 1
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
        ):
            if isinstance(hidden_states, UnfinalizedMoEOutput):
                raise RuntimeError(
                    "PCP reduce-scatter cannot consume a deferred MoE output."
                )
            hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)

        if self.shared_experts is not None:
            assert shared_output is not None
            return shared_output, hidden_states
        else:
            return hidden_states

    def _routed_takes_linear_mxfp8(self) -> bool:
        """Whether the routed experts quantize their input to linear-layout
        MXFP8 in a no-DP/EP monolithic prepare step that can be skipped.
        """
        if (
            self.do_naive_dispatch_combine
            or self.moe_config.pcp_size > 1
            or not self._quant_method.is_monolithic
        ):
            return False
        kernel = getattr(self._quant_method, "moe_kernel", None)
        if kernel is None or not isinstance(
            kernel.prepare_finalize, MoEPrepareAndFinalizeNoDPEPMonolithic
        ):
            return False
        experts = kernel.fused_experts
        qc = experts.quant_config
        return (
            not experts.expects_unquantized_inputs
            and qc.quant_dtype == "mxfp8"
            and not qc.is_scale_swizzled
            and qc.block_shape == [1, MXFP8_BLOCK_SIZE]
            and qc.mx_alignment in (0, MXFP8_BLOCK_SIZE)
        )

    def _wrap_quantized_input(
        self,
        hidden_states: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        quantized_input: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[QuantizedActivation | None, QuantizedActivation | None]:
        """Split a pre-quantized MoE input into the shared expert's (swizzled
        scales) and the routed experts' (linear scales) activations. Either is
        None where that consumer quantizes the bf16 input itself.
        """
        if quantized_input is None:
            return None, None
        act_q, act_sf_swizzled, act_sf_linear = quantized_input
        shared_q = None
        if shared_experts_input is not None and (
            shared_experts_input.shape == act_q.shape
        ):
            shared_q = QuantizedActivation(
                data=act_q,
                scale=act_sf_swizzled,
                orig_dtype=shared_experts_input.dtype,
                orig_shape=shared_experts_input.shape,
                quant_key=kMxfp8Dynamic,
            )
        routed_q = None
        if hidden_states.shape == act_q.shape and self._routed_takes_linear_mxfp8():
            routed_q = QuantizedActivation(
                data=act_q,
                scale=act_sf_linear,
                orig_dtype=hidden_states.dtype,
                orig_shape=hidden_states.shape,
                quant_key=kMxfp8Dynamic,
            )
        return shared_q, routed_q

    def _maybe_locality_moe(
        self,
        hidden_states: torch.Tensor,
        routed_q: QuantizedActivation | None,
    ) -> UnfinalizedMoEOutput | None:
        """The routed MoE output of this call from the locality-domain kernel
        (VLLM_MOE_LOCALITY_KERNEL, layers/fused_moe/locality_moe.py), which
        computes the router GEMM itself from the runner-held gate weight and
        reads the pre-quantized linear MXFP8 input; None when it does not serve
        the call (the gate and the routed experts then run as usual).
        """
        if (
            routed_q is None
            or self.gate is None
            or self._fse_fuse_gate
            or self.do_naive_dispatch_combine
            or self.moe_config.pcp_size > 1
        ):
            return None
        kernel = getattr(self._quant_method, "moe_kernel", None)
        experts = getattr(kernel, "fused_experts", None)
        try_locality_moe = getattr(experts, "try_locality_moe", None)
        if try_locality_moe is None:
            return None
        return try_locality_moe(
            self.routed_experts,
            hidden_states,
            self.gate.weight,
            routed_q.data,
            routed_q.scale,
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        quantized_input: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> (
        torch.Tensor
        | UnfinalizedMoEOutput
        | tuple[torch.Tensor, torch.Tensor | UnfinalizedMoEOutput]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Entry point called by the custom op to run the MoE computation.

        Handles pre-dispatch setup (gate application, external shared expert
        triggering, quant config init) then performs the following steps
        within the sequence-parallel context.

        - Performs expert routing
        - fused MoE kernel execution
        - shared expert computation.

        Returns routed output, optionally paired with shared-expert output. A
        fused consumer may request the routed output in deferred-finalize form.
        For moe_forward_shared_ext it returns (shared output, shared-expert
        gate logits or an empty [tokens, 0] tensor, routed output).
        """
        # TODO(bnell): this can be removed after MK migration is complete.
        self.routed_experts._ensure_moe_quant_config_init()

        shared_q, routed_q = self._wrap_quantized_input(
            hidden_states, shared_experts_input, quantized_input
        )

        # If using multi-stream overlap for shared experts, we must launch it
        # before routed expert dispatch.
        shared_experts_overlapping = False
        if self._shared_experts is not None:
            shared_experts_overlapping = self._shared_experts.maybe_forward_async(
                shared_experts_input, shared_q
            )

        # VLLM_MOE_LOCALITY_KERNEL: one kernel runs the router GEMM and the
        # routed experts. Otherwise, if the Runner holds the gate, apply it
        # after the stream sync, so it can run overlapped with the shared
        # experts.
        # NOTE: in future PR, MoE runner will always hold the gate.
        routed_out = self._maybe_locality_moe(hidden_states, routed_q)
        if self.gate is not None and routed_out is None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = F.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)

        with self._sequence_parallel_context():
            # TODO(bnell): parts of the dispatch/combine steps will go away once
            # #32567 lands and the remaining kernels are made MKs.  The PCP
            # code will probably remain
            hidden_states, router_logits = self._maybe_dispatch(
                hidden_states,
                router_logits,
            )

            shared_output, hidden_states = self._apply_quant_method(
                hidden_states=hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input,
                input_ids=input_ids,
                shared_experts_overlapping=shared_experts_overlapping,
                shared_quantized_input=shared_q,
                routed_quantized_input=routed_q,
                routed_out=routed_out,
            )

            result = self._maybe_combine(
                shared_output,
                hidden_states,
            )
            if not self._uses_shared_ext_op:
                return result
            shared, fused = cast(tuple[torch.Tensor, torch.Tensor], result)
            if isinstance(shared, tuple):
                shared, gate = shared
            else:
                gate = shared.new_empty((*shared.shape[:-1], 0))
            return shared, gate, fused

    #########################################################
    #
    # Old methods from FusedMoE layer. Remove when possible.
    #
    #########################################################

    #
    # Properties
    #

    @property
    def layer_id(self):
        # Delayed import to avoid circular dependency
        from vllm.model_executor.models.utils import extract_layer_index

        return extract_layer_index(self.layer_name)

    #
    # Attributes still needed by models
    #

    @property
    def is_monolithic(self) -> bool:
        return self.routed_experts.quant_method.is_monolithic

    @property
    def activation(self) -> MoEActivation:
        return self.routed_experts.activation

    #
    # Expert maps
    #

    @property
    def expert_map_manager(self):
        """Forward to routed_experts.expert_map_manager for backward compatibility."""
        return self.routed_experts.expert_map_manager

    @property
    def expert_placement_strategy(self) -> ExpertPlacementStrategy:
        return self.expert_map_manager.placement_strategy

    @property
    def expert_global_to_physical(self) -> torch.Tensor | None:
        tables = self.expert_map_manager.routing_tables
        return tables[0] if tables else None

    @property
    def expert_physical_to_global(self) -> torch.Tensor | None:
        """Routing table: physical expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[1] if tables else None

    @property
    def expert_local_to_global(self) -> torch.Tensor | None:
        """Routing table: local expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[2] if tables else None

    @property
    def expert_map(self) -> torch.Tensor | None:
        return self.routed_experts.expert_map

    def _expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return self.routed_experts._expert_routing_tables()

    def update_expert_map(self):
        self.routed_experts.update_expert_map()

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        """Map global expert ID to local expert ID."""
        return self.routed_experts._map_global_expert_id_to_local_expert_id(expert_id)

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        return self.routed_experts.get_expert_weights()

    #
    # EPLB
    #

    @property
    def eplb_state(self) -> EplbLayerState | None:
        return self.router.eplb_state

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        if self.router.eplb_state is not None:
            self.router.eplb_state.set_layer_state(
                moe_layer_idx,
                expert_load_view,
                logical_to_physical_map,
                logical_replica_count,
            )
