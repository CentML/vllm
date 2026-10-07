# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

logger = init_logger(__name__)


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
        if envs.VLLM_MTP_DRAFT_PREFILL_PRUNE:
            # Under prefill context parallelism the draft prefill is re-laid
            # out, so last_token_indices does not index the attention's rows.
            # The PCP manager is attached after model load; read the config.
            pcp = self.vllm_config.parallel_config.prefill_context_parallel_size > 1
            _enable_draft_prefill_prune(
                draft_model, None if pcp else self.last_token_indices
            )
        return draft_model

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
) -> None:
    """Let the draft attention layer compute prefill attention only for each
    prefill request's sampled row (FlashInferImpl._draft_prefill_last_rows).

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
        return
    layers = [m for m in draft_model.modules() if isinstance(m, Attention)]
    if len(layers) != 1 or not hasattr(
        layers[0].impl, "draft_prefill_last_token_indices"
    ):
        logger.warning(
            "MTP draft prefill attention pruning off: needs exactly one FlashInfer "
            "draft attention layer, got %s",
            [(m.layer_name, type(m.impl).__name__) for m in layers],
        )
        return
    layers[0].impl.draft_prefill_last_token_indices = last_token_indices
    logger.info("MTP draft prefill attention pruning on for %s", layers[0].layer_name)
