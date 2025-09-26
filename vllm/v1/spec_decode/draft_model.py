# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from typing import Any, Optional

import torch

from vllm.attention.layer import Attention
from vllm.config import ModelConfig, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer


class DraftModelProposer(SpecDecodeBaseProposer):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            pass_cudagraph_args_to_forward_ctx=True,
            # The draft model runs one forward pass to prefill
            # the target_token_ids, and another forward pass for decoding
            # based on the next_token_ids. I.e. it needs 1 more forward pass.
            one_extra_forward_pass=False,
            # the first draft_token_ids are replaced by next_token_ids, so
            # they don't need to be returned as proposed tokens
            drop_first_drafted_tokens=False,
            runner=runner)
        self._raise_if_multimodal()
        self._raise_if_mrope()

    def _raise_if_multimodal(self):
        if self.is_multimodal_model:
            raise NotImplementedError("Speculative Decoding with draft models "
                                      "does not support multimodal models yet")

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError("Speculative Decoding with draft models "
                                      "does not support M-RoPE yet")

    def _model_kwargs(self, num_tokens: int) -> dict[str, Any]:
        self._raise_if_multimodal()
        self._raise_if_mrope()
        return {
            "input_ids": self.input_ids[:num_tokens],
            "positions": self.positions[:num_tokens],
        }

    def dummy_run(self, num_tokens: int, forward_ctx_kwargs: dict):
        model_kwargs = self._model_kwargs(num_tokens)
        assert isinstance(self.model, torch.nn.Module)
        with set_forward_context(
                vllm_config=self.vllm_config,
                num_tokens=num_tokens,
                **forward_ctx_kwargs,
        ):
            self.model(**model_kwargs)

    def set_input_ids_first_pass(self, target_token_ids: torch.Tensor,
                                 next_token_ids: torch.Tensor, num_tokens: int,
                                 last_token_indices: torch.Tensor) -> None:
        pass
        # self.input_ids[:num_tokens] = target_token_ids

    def update_inputs_for_prefill_tokens(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        last_token_indices: Optional[torch.Tensor],
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               CommonAttentionMetadata]:
        """
        Update the input_ids, positions, and attention metadata to account
        for the additional prefill token that must be left in.

        For EAGLE, we slice the first prefill token from the target_token_ids
        and replace the last token with `next_token_ids`. For draft-model,
        we need to include all target_token_ids and also insert the 
        next_token_ids at the end. This means the sequence length increases
        by 1 for each prefill request, and we need to update the metadata
        accordingly.

        This function returns updated input_ids (with zeros inserted),
        positions (with last position + 1 inserted), updated last_token_indices,
        as well as updated common_attn_metadata.
        """
        # Step 1: Determine which requests need a token inserted.
        query_lens = (common_attn_metadata.query_start_loc_cpu[1:] -
                      common_attn_metadata.query_start_loc_cpu[:-1])
        req_needs_insert = (common_attn_metadata.seq_lens_cpu == query_lens)
        num_inserts = int(req_needs_insert.sum())

        if num_inserts == 0:
            # Per the logic, non-insert requests still modify tokens.
            # So we only return early if *all* requests are non-insert and seq_len==1.
            # This implementation handles the general case regardless.
            pass

        # Step 2: Calculate new tensor sizes and create them.
        num_tokens_input = target_token_ids.size(0)
        new_input_size = num_tokens_input + num_inserts
        new_input_ids = torch.empty(new_input_size,
                                    dtype=target_token_ids.dtype,
                                    device=target_token_ids.device)
        new_positions = torch.empty(new_input_size,
                                    dtype=target_positions.dtype,
                                    device=target_positions.device)

        # Step 3: Calculate destination indices for all tokens using a cumsum trick.
        # This is the core of the vectorized logic.
        prefix_inserts_per_req = torch.zeros_like(req_needs_insert,
                                                  dtype=torch.long)
        prefix_inserts_per_req[1:] = req_needs_insert.cumsum(dim=0)[:-1]

        # Create a mask to identify which original tokens to *keep*.
        # We keep all tokens except the very first token of non-insert ("slice") requests.
        tokens_to_keep_mask = torch.ones(num_tokens_input,
                                         dtype=torch.bool,
                                         device=target_token_ids.device)
        slice_req_indices = (~req_needs_insert).nonzero().squeeze(-1)

        if slice_req_indices.numel() > 0:
            slice_start_indices = common_attn_metadata.query_start_loc_cpu[
                slice_req_indices]
            tokens_to_keep_mask[slice_start_indices] = False

        # Get the source indices of the tokens we are keeping.
        source_indices_kept_tokens = tokens_to_keep_mask.nonzero().squeeze(-1)
        source_indices_kept_tokens = source_indices_kept_tokens.to(
            new_input_ids.device)

        # Calculate the destination indices for these kept tokens.
        # The new index is the original index minus the number of tokens already
        # dropped, plus the number of new tokens already inserted.
        num_sliced_tokens_prefix_sum = (~tokens_to_keep_mask).cumsum(dim=0)

        # Broadcast the per-request insert count to a per-token level.
        token_level_prefix_inserts = prefix_inserts_per_req.repeat_interleave(
            query_lens).to(new_input_ids.device)

        offset_temp = num_sliced_tokens_prefix_sum + token_level_prefix_inserts

        dest_indices_kept_tokens = (
            torch.arange(num_tokens_input, device=new_input_ids.device) -
            offset_temp)[source_indices_kept_tokens]

        # Step 4: Populate the new tensors with the original data.
        new_input_ids[dest_indices_kept_tokens] = target_token_ids[
            source_indices_kept_tokens]
        new_positions[dest_indices_kept_tokens] = target_positions[
            source_indices_kept_tokens]

        # Step 5: Add the `next_token_ids` to the end of each request's new location.
        # The new end location is the original end, plus cumulative inserts, minus 1 for slice requests.
        # `req_needs_insert` (as 0 or 1) handles the "-1" for us.
        new_end_locs = (common_attn_metadata.query_start_loc_cpu[1:] +
                        prefix_inserts_per_req + req_needs_insert)
        dest_indices_new_tokens = new_end_locs - 1

        dest_indices_new_tokens = dest_indices_new_tokens.to(
            new_input_ids.device)
        new_input_ids[dest_indices_new_tokens] = next_token_ids

        # For positions, insert requests get the last position + 1.
        # Slice requests get the position of the token they replaced (now second to last).
        insert_req_indices = req_needs_insert.nonzero().squeeze(-1)
        if insert_req_indices.numel() > 0:
            last_pos_indices = common_attn_metadata.query_start_loc_cpu[
                insert_req_indices + 1] - 1
            new_positions[dest_indices_new_tokens[insert_req_indices]] = \
                target_positions[last_pos_indices] + 1

        if slice_req_indices.numel() > 0:
            # The new token takes the position of the token it replaced (the old last token).
            # We find the new position of the second-to-last old token.
            second_to_last_pos_indices = new_end_locs[slice_req_indices] - 2
            new_positions[dest_indices_new_tokens[slice_req_indices]] = \
                new_positions[second_to_last_pos_indices]

        # Step 6: Update last_token_indices.
        # This is simply the destination indices we just calculated for the new tokens.
        last_token_indices = dest_indices_new_tokens

        # Step 7: Update common_attn_metadata

        common_attn_metadata.num_actual_tokens = (
            common_attn_metadata.num_actual_tokens + num_inserts)
        common_attn_metadata.seq_lens_cpu += req_needs_insert
        common_attn_metadata.seq_lens = common_attn_metadata.seq_lens_cpu.to(
            common_attn_metadata.seq_lens.device)
        common_attn_metadata.query_start_loc_cpu = torch.zeros(
            common_attn_metadata.query_start_loc_cpu.size(0), dtype=torch.long)
        common_attn_metadata.query_start_loc_cpu[1:] = new_end_locs
        common_attn_metadata.query_start_loc = (
            common_attn_metadata.query_start_loc_cpu.to(
                common_attn_metadata.query_start_loc.device))

        common_attn_metadata.max_query_len = (
            common_attn_metadata.query_start_loc_cpu[1:] -
            common_attn_metadata.query_start_loc_cpu[:-1]).max().item()
        common_attn_metadata.max_seq_len = common_attn_metadata.seq_lens_cpu.max(
        ).item()

        common_attn_metadata.num_computed_tokens_cpu = common_attn_metadata.seq_lens_cpu - 1

        block_numbers = new_positions // self.block_size
        block_ids = common_attn_metadata.block_table_tensor.gather(
            dim=1, index=block_numbers.view(1, -1))
        block_ids = block_ids.view(-1)
        common_attn_metadata.slot_mapping = (block_ids * self.block_size +
                                             new_positions % self.block_size)

        # The user will update common_attn_metadata themselves.
        return (new_input_ids, new_positions, last_token_indices,
                common_attn_metadata)

    def load_model(self, target_model: Any) -> None:
        """Takes target_model to satisfy the type checker."""
        draft_model_config: ModelConfig = (
            self.vllm_config.speculative_config.draft_model_config)
        vllm_config_draft: VllmConfig = replace(
            self.vllm_config, model_config=draft_model_config)

        # This must be computed before loading the draft model
        # because that mutates the forward_context of the vllm_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())

        from vllm.compilation.backends import set_model_tag

        with set_model_tag("draft_model"):
            self.model = get_model(
                vllm_config=vllm_config_draft,
                model_config=draft_model_config,
                prefix="draft_model",
            )

        # This must be computed after loading the draft model
        # because that mutates the forward_context of the vllm_config
        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
            target_attn_layer_names)
        self.attn_layer_names = list(draft_attn_layer_names)
