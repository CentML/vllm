# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

if TYPE_CHECKING:
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

logger = init_logger(__name__)

# VLLM_MTP_DRAFT_PREFILL_ROWS=1: after its attention (which writes the draft KV
# of every row), the MTP layer of the draft prefill runs only on each request's
# last accepted row, the one the speculator samples (o_proj, MoE and norms at
# M = requests instead of M = verified tokens). Off by default.
MTP_DRAFT_PREFILL_ROWS = os.environ.get("VLLM_MTP_DRAFT_PREFILL_ROWS", "0") == "1"


class MTPSpeculator(AutoRegressiveSpeculator):
    share_mtp_topk_indices: bool = False

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.vllm_config)
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        # Detect index_share_for_mtp_iteration. When True, the proposer
        # toggles skip_topk so step 0 computes MTP's own indices and
        # steps 1+ reuse them.
        self.share_mtp_topk_indices = (
            self.vllm_config.parallel_config.prefill_context_parallel_size == 1
            and getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
            and hasattr(draft_model.model, "set_skip_topk")
            and hasattr(draft_model.model, "compact_topk_indices")
        )
        self._draft_prefill_prune_layer: Attention | None = None
        self._draft_prefill_prune_group_id: int | None = None
        if envs.VLLM_MTP_DRAFT_PREFILL_PRUNE:
            # Under prefill context parallelism the draft prefill is re-laid
            # out, so last_token_indices does not index the attention's rows.
            # The PCP manager is attached after model load; read the config.
            pcp = self.vllm_config.parallel_config.prefill_context_parallel_size > 1
            self._draft_prefill_prune_layer = _enable_draft_prefill_prune(
                draft_model, None if pcp else self.last_token_indices
            )
        if MTP_DRAFT_PREFILL_ROWS:
            if getattr(draft_model, "supports_draft_out_rows", False) and (
                self.dp_size == 1
            ):
                self.enable_draft_out_rows()
                logger.info("MTP draft prefill row pruning on")
            else:
                logger.warning(
                    "MTP draft prefill row pruning off: needs a draft model "
                    "with out_rows support (%s) and no data parallelism",
                    type(draft_model).__name__,
                )
        return draft_model

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        layer = self._draft_prefill_prune_layer
        impl = cast("FlashInferImpl", layer.impl) if layer is not None else None
        saved_indices = (
            impl.draft_prefill_last_token_indices if impl is not None else None
        )
        if impl is not None and self.pcp_manager is not None:
            impl.draft_prefill_last_token_indices = None
        if (
            layer is not None
            and impl is not None
            and num_reqs > 0
            and attn_metadata is not None
            and self.pcp_manager is None
            and cudagraph_runtime_mode != CUDAGraphMode.FULL
            and not torch.cuda.is_current_stream_capturing()
        ):
            from vllm.v1.attention.backends.flashinfer import DraftPrefillPruning

            # Resolve the actual draft layer's cache group, not the target's
            # first group (Qwen's target has both linear and full attention).
            group_id = self._draft_prefill_prune_group_id
            if group_id is None:
                group_id = next(
                    (
                        i
                        for i, group in enumerate(self.kv_cache_config.kv_cache_groups)
                        if layer.layer_name in group.layer_names
                    ),
                    -1,
                )
                self._draft_prefill_prune_group_id = group_id
            if group_id >= 0:
                rows = self.last_token_indices[:num_reqs]
                impl.draft_prefill_pruning = DraftPrefillPruning(
                    rows=rows,
                    seq_lens=(self.input_buffers.positions[rows] + 1).to(torch.int32),
                    block_tables=self.block_tables.input_block_tables[group_id][
                        :num_reqs
                    ],
                    max_seq_len=self.draft_max_seq_len,
                )
            else:
                # Do not use the bucket-local fallback with an unknown group.
                impl.draft_prefill_last_token_indices = None
        try:
            super()._prefill(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                mm_inputs,
            )
        finally:
            if impl is not None:
                impl.draft_prefill_pruning = None
                impl.draft_prefill_last_token_indices = saved_indices

    def on_prefill_begin(self, num_reqs: int) -> None:
        # Step 0 computes its own top-k. Unconditional, so a step that died
        # midway cannot leave reuse mode on.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)

    def on_prefill_end(self, num_reqs: int) -> None:
        # Step 0 (prefill) wrote topk indices for every query token in the
        # multi-token batch. Compact them down to each request's last token so
        # steps 1+ can reuse them from the shared buffer.
        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:
            self.model.model.compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        # Switch to reuse mode so draft steps 1+ skip the indexer op and read
        # the indices that step 0 wrote into the shared buffer.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)


def _enable_draft_prefill_prune(
    draft_model: nn.Module, last_token_indices: torch.Tensor | None
) -> Attention | None:
    """Enable sampled-row attention for a single safe MTP draft layer.

    ``last_token_indices`` is the speculator's persistent buffer of those rows,
    written before every draft prefill; None keeps pruning off. The draft
    prefill samples only those rows, and with a single attention layer the
    draft KV comes from the layer input, not from attention outputs. A deeper
    draft would feed every row's attention output into the next layer's KV, so
    pruning stays off there (and for non-FlashInfer layers).
    """
    if last_token_indices is None:
        logger.warning(
            "MTP draft prefill attention pruning off: not supported with prefill "
            "context parallelism"
        )
        return None
    layers = [m for m in draft_model.modules() if isinstance(m, Attention)]
    if len(layers) != 1 or not hasattr(
        layers[0].impl, "draft_prefill_last_token_indices"
    ):
        logger.warning(
            "MTP draft prefill attention pruning off: needs exactly one FlashInfer "
            "draft attention layer, got %s",
            [(m.layer_name, type(m.impl).__name__) for m in layers],
        )
        return None
    layers[0].impl.draft_prefill_last_token_indices = last_token_indices
    logger.info("MTP draft prefill attention pruning on for %s", layers[0].layer_name)
    return layers[0]
