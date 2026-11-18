# SPDX-License-Identifier: Apache-2.0
"""FlashInfer MXFP4 W4A4 DSpark experts for the native DeepSeek-V4.1 path."""

from bisect import bisect_left

import torch

import vllm.envs as envs
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
    MegaMoEWeightSpec,
)

logger = init_logger(__name__)


class FlashInferSm107W4A4DraftExperts(DeepGemmMegaMoEExperts):
    """Load the checkpoint's packed MXFP4 unchanged and quantize only at runtime."""

    uses_flashinfer_w4a4 = True

    def _kernel_fuses_shared_experts(self, spec: MegaMoEWeightSpec) -> bool:
        return spec.dtype == torch.float8_e4m3fn and spec.scale_group_k == 32

    @staticmethod
    def _kernel_layout_shard(dst, src, shard_id, is_scale):
        # Keep gate and up contiguous, with unswizzled packed E8M0 scales.
        # FlashInfer packs its own layouts once after checkpoint loading.
        if shard_id == "w2":
            return dst, src
        rows = src.shape[0]
        offset = 0 if shard_id == "w1" else rows
        return dst[offset : offset + rows], src

    @staticmethod
    def _as_prequantized(weight, scale, data_type):
        return weight.view(data_type), scale.contiguous().view(torch.float8_e8m0fnu)

    def _initialize_flashinfer(self):
        from flashinfer.moe_ep import (
            BootstrapConfig,
            FleetParams,
            MegaConfig,
            MoEEpLayer,
            MoEWeightPack,
            Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
            Sm107SharedExpertConfig,
        )
        from flashinfer.moe_ep.sm107_nvfp4 import decode_kernel

        if self.num_experts != self.num_logical_experts:
            raise RuntimeError(
                "FlashInfer W4A4 DSpark currently requires EPLB off for the drafter"
            )
        if self.num_shared_experts != 1:
            raise RuntimeError(
                "FlashInfer W4A4 DSpark requires one fused MXFP8 shared expert"
            )
        if self.activation.alpha != 1.0 or self.activation.beta != 0.0:
            raise RuntimeError(
                "FlashInfer W4A4 DSpark requires standard SwiGLU constants"
            )

        ep = get_ep_group()
        w13, s13 = self._as_prequantized(
            self.w13_weight, self.w13_weight_scale, torch.uint8
        )
        w2, s2 = self._as_prequantized(
            self.w2_weight, self.w2_weight_scale, torch.uint8
        )
        sw13, ss13 = self._as_prequantized(
            self.shared_l1_weight, self.shared_l1_weight_scale, torch.float8_e4m3fn
        )
        sw2, ss2 = self._as_prequantized(
            self.shared_l2_weight, self.shared_l2_weight_scale, torch.float8_e4m3fn
        )
        variant = decode_kernel(self.max_num_tokens)
        options = (
            dict(kernel_variant="genphase", cluster_shape_mn=(4, 1), fc2_use_bulk=True)
            if variant == "genphase"
            else {}
        )
        config = Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
            self.intermediate_size,
            self.top_k,
            gate_up_clamp=self.activation.clamp,
            shared_expert=Sm107SharedExpertConfig(self.intermediate_size),
            knobs="cache",
            **options,
        )
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                rank=ep.rank_in_group,
                world_size=ep.world_size,
                process_group=ep.device_group,
                device=torch.cuda.current_device(),
            ),
            fleet_params=FleetParams(
                num_experts=self.num_experts,
                max_tokens_per_rank=self.max_num_tokens,
                token_hidden_size=self.hidden_size,
            ),
            weights=MoEWeightPack(
                w13,
                w2,
                s13,
                s2,
                shared_expert=MoEWeightPack(
                    sw13[None], sw2[None], ss13[None], ss2[None]
                ),
            ),
            backend=MegaConfig(megakernel=config),
        )
        buckets = (512, 1024, 2048, 4096, 6144, 8192)
        self._flashinfer_buckets = tuple(
            m for m in buckets if m < self.max_num_tokens
        ) + (self.max_num_tokens,)
        self._flashinfer_workspaces = []
        for capacity in self._flashinfer_buckets:
            workspace = layer.create_workspace(capacity)
            layer.warmup(workspace=workspace)
            self._flashinfer_workspaces.append(workspace)
        self._flashinfer_layer = layer
        logger.info(
            "%s: FlashInfer SM107 W4A4 DSpark (%s, EP%d, fused MXFP8 shared expert), buckets %s",
            self.prefix,
            variant,
            ep.world_size,
            self._flashinfer_buckets,
        )

    def forward(self, hidden_states, topk_weights, topk_ids, shared_experts=None):
        from flashinfer.moe_ep import MoEEpTensors

        if shared_experts is not None:
            raise RuntimeError("Expected DSpark shared expert to be fused")
        if not hasattr(self, "_flashinfer_layer"):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "FlashInfer W4A4 DSpark must warm up before CUDA graph capture"
                )
            self._synchronize_first_launch()
            self._initialize_flashinfer()

        n = hidden_states.shape[0]
        maximum = self.max_num_tokens
        if is_forward_context_available():
            context = get_forward_context()
            dp = context.dp_metadata
            across = getattr(dp, "num_tokens_across_dp_cpu", None)
            if across is not None and across.numel():
                maximum = max(n, int(across.max()))
            if self.capture_fn is not None:
                self.capture_fn(topk_ids)
            padding = getattr(context, "is_padding", None)
            if envs.VLLM_MOE_SKIP_PADDING and padding is not None:
                mask = padding[:n, None]
                topk_ids = topk_ids.masked_fill(mask, -1)
                topk_weights = topk_weights.masked_fill(mask, 0.0)

        workspace = self._flashinfer_workspaces[
            bisect_left(self._flashinfer_buckets, maximum)
        ]
        return self._flashinfer_layer.forward(
            MoEEpTensors(
                hidden_states.contiguous(),
                topk_ids.contiguous(),
                topk_weights.contiguous(),
            ),
            workspace=workspace,
        )


FlashInferSm107W4A4DraftExperts.weight_loader.supports_moe_loading = True
