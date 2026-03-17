# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CuTe DSL NvFP4 MoE experts using FlashInfer's CuteDslMoEWrapper.

This expert class wraps the purpose-built MoE pipeline from
flashinfer.fused_moe.cute_dsl (moe_sort → gather-GEMM1+SwiGLU → async memset
→ scatter-GEMM2+finalize) that handles token routing internally via moe_sort.
It uses Standard activation format (flat [M, K] input).

This is distinct from the CUTEDSL (masked GEMM) backend which uses the generic
grouped_gemm_nt_masked primitive from flashinfer.gemm on BatchedExperts data.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform


class CuteDslMoENvFp4Experts(mk.FusedMoEExpertsModular):
    """
    CuTe DSL NvFP4 MoE experts (gather/scatter pipeline).

    Uses FlashInfer's ``CuteDslMoEWrapper`` from
    ``flashinfer.fused_moe.cute_dsl``, the purpose-built MoE pipeline with
    gather-GEMM1+SwiGLU → scatter-GEMM2+finalize (with atomic scatter-reduce
    and router weight application).

    The wrapper pre-allocates internal buffers (moe_sort outputs, GEMM1
    intermediate, CUDA streams/events) for CUDA graph compatibility.

    The kernel handles token-to-expert routing internally via moe_sort, so
    this uses Standard activation format (not BatchedExperts).
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        assert quant_config.quant_dtype == "nvfp4", (
            "Only nvfp4 quantization is supported."
        )
        self.topk = moe_config.experts_per_token
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

        from flashinfer.fused_moe.cute_dsl.fused_moe import (
            CuteDslMoEWrapper,
        )

        self._wrapper = CuteDslMoEWrapper(
            num_experts=moe_config.num_experts,
            top_k=moe_config.experts_per_token,
            hidden_size=moe_config.hidden_dim,
            intermediate_size=(moe_config.intermediate_size_per_partition),
            use_cuda_graph=True,
            max_num_tokens=moe_config.max_num_tokens,
            num_local_experts=moe_config.num_local_experts,
            local_expert_offset=(self.ep_rank * self.local_num_experts),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fuse input scales into weight scale_2
        # (same as TrtLlm / CuteDSL masked GEMM).
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.is_device_capability_family(100)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        # SwiGLU is fused into the pipeline, so gated activation
        # is required.
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kNvfp4Static, kNvfp4Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        # CuTe DSL GEMM tiles require alignment to 128.
        return hidden_dim % 128 == 0

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(
        self,
    ) -> mk.TopKWeightAndReduce:
        # The kernel applies router weights and does the
        # scatter-reduce internally.
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
        # Internal workspaces (moe_sort buffers, GEMM1 intermediate,
        # etc.) are pre-allocated inside the CuteDslMoEWrapper.
        # We only need the output tensor from vLLM.
        workspace1 = (0,)
        workspace2 = (0,)
        # hidden_states are NvFP4 packed into uint8, so K is half
        # the real hidden dim.
        assert self.hidden_dim == K * 2
        output = (M, self.hidden_dim)
        return (workspace1, workspace2, output)

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
    ):
        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None
        assert self.quant_config.g1_alphas is not None
        assert self.quant_config.g2_alphas is not None
        assert self.quant_config.a2_gscale is not None
        assert a1q_scale is not None

        import vllm.utils.flashinfer as fi_utils

        # Skip during FlashInfer autotuning dummy runs.
        if fi_utils._is_fi_autotuning:
            return hidden_states

        # Reshape activation scales to [M, K//16] as block scales.
        num_tokens = hidden_states.shape[0]
        k_packed = hidden_states.shape[1]  # hidden_size // 2
        x_sf = a1q_scale.view(torch.uint8).reshape(num_tokens, -1)[:, : k_packed // 8]

        # fc2_input_scale: quantization global scale for
        # intermediate activations between GEMM1 and GEMM2.
        # In vLLM, a2_gscale = 1/a2_scale, so invert it back.
        fc2_input_scale = (1.0 / self.quant_config.a2_gscale[0:1]).to(torch.float32)

        # Point the wrapper's output buffer at vLLM's output
        # tensor so the kernel writes directly into it. The
        # wrapper's other pre-allocated buffers (moe_sort,
        # GEMM1 intermediate, CUDA streams) are still reused.
        self._wrapper._moe_output = output

        self._wrapper.run(
            x=hidden_states,
            x_sf=x_sf,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
            w1_weight=w1,
            w1_weight_sf=self.quant_config.w1_scale.view(torch.uint8),
            w1_alpha=self.quant_config.g1_alphas,
            fc2_input_scale=fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.quant_config.w2_scale.view(torch.uint8),
            w2_alpha=self.quant_config.g2_alphas,
        )
