# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import torch

from vllm.triton_utils import triton, tl

from vllm.attention.layer import Attention
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.speculative import SpeculativeConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    extend_all_queries,
    extend_flat_seqs,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID, SpecDecodeBaseProposer

logger = init_logger(__name__)


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
            runner=runner,
        )
        self._raise_if_multimodal()
        self._raise_if_mrope()
        self._raise_if_padded_drafter_batch()
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

        try:
            self.pard_token_id = int(self.draft_model_config.hf_config.pard_token)
        except Exception:
            self.pard_token_id = None
        if self.pard_token_id is None:
            logger.warning("Could not find pard token ID in draft model hf config: %s", 
                           self.draft_model_config.hf_config)
            raise ValueError("PARD token ID must be specified in the draft model hf config.")

    @torch.cuda.nvtx.range("DraftModelProposer.propose")
    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        This function processes the inputs first before calling the .propose()
        method of the parent class.
        """
        inputs = DraftModelInputs(
            cad=common_attn_metadata,
            token_ids=target_token_ids,
            positions=target_positions,
        )

        all_next_token_ids = [next_token_ids]
        for i in range(1, self.num_speculative_tokens):
            all_next_token_ids.append(
                torch.full_like(next_token_ids, fill_value=self.pard_token_id)
            )
        next_token_ids = torch.stack(all_next_token_ids, dim=0)

        with torch.cuda.nvtx.range("merge_next_token_ids_into_token_ids"):
            inputs = merge_next_token_ids_into_token_ids(
                inputs=inputs,
                next_token_ids=next_token_ids,
                block_size=self.block_size,
                max_model_len=self.max_model_len,
                arange=self.arange,
            )

        common_attn_metadata = inputs.cad
        target_token_ids = inputs.token_ids
        target_positions = inputs.positions

        num_tokens = target_token_ids.shape[0]
        self.input_ids[:num_tokens] = target_token_ids

        assert last_token_indices is None
        assert self.runner is not None

        # Build ATTN Metadata
        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder
        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )
        assert not self.draft_indexer_metadata_builder
        assert not self.indexer_layer_names

        # At this moment, we assume all layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        cudagraph_runtime_mode = CUDAGraphMode.NONE
        if self.use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
            cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        else:
            num_input_tokens = num_tokens

        # copy inputs to buffer for cudagraph
        self._set_positions(num_tokens, target_positions)

        assert not self.pass_hidden_states_to_model
        assert not self.supports_mm_inputs
        input_ids = self.input_ids[:num_input_tokens]
        inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }

        with torch.cuda.nvtx.range("draft_model_forward"):
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
            ):
                last_hidden_states = self.model(**model_kwargs)
                assert isinstance(last_hidden_states, torch.Tensor)
        
        # with torch.cuda.nvtx.range("draft_model_sample"):
        #     draft_sampled_tokens = []
        #     for i in range(self.num_speculative_tokens):
        #         last_token_indices = common_attn_metadata.query_start_loc[1:] - self.num_speculative_tokens + i
        #         sample_hidden_states = last_hidden_states[last_token_indices]
        #         logits = self.model.compute_logits(sample_hidden_states)
        #         draft_token_ids = logits.argmax(dim=-1)
        #         draft_sampled_tokens.append(draft_token_ids)
        # return torch.stack(draft_sampled_tokens, dim=1)

        with torch.cuda.nvtx.range("draft_model_sample"):
            seq_ends = common_attn_metadata.query_start_loc[1:]
            step_offsets = torch.arange(
                self.num_speculative_tokens, 
                device=last_hidden_states.device, 
                dtype=torch.long
            )
            
            base_starts = seq_ends - self.num_speculative_tokens
            all_indices = base_starts.unsqueeze(1) + step_offsets.unsqueeze(0)
            flat_indices = all_indices.view(-1)
            sample_hidden_states = last_hidden_states[flat_indices]
            logits = self.model.compute_logits(sample_hidden_states)
            draft_token_ids_flat = logits.argmax(dim=-1)
            return draft_token_ids_flat.view(-1, self.num_speculative_tokens)

    def _raise_if_multimodal(self):
        if self.supports_mm_inputs:
            raise NotImplementedError(
                "Speculative Decoding with draft models "
                "does not support multimodal models yet"
            )

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative Decoding with draft models does not support M-RoPE yet"
            )

    def _raise_if_padded_drafter_batch(self):
        if not self.vllm_config.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Speculative Decoding with draft models does not support "
                "padded drafter batch yet. Please pass --disable-padded-drafter-batch "
                "in the speculative_config."
            )

    def _raise_if_vocab_size_mismatch(self):
        self.vllm_config.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self):
        # Note(Tomas Ruiz) If we run the target model with TP > 1 and
        # the draft model with TP = 1, then the different TP ranks collide.
        # Specifically when all ranks compile the draft model on rank 0
        # (because TP=1), then the torch compile cache is overwritten and corrupted.
        # We need a mechanism like this: https://github.com/vllm-project/vllm/pull/5414
        # To prevent this error, we assert that both TP sizes must be the same.
        spec_cfg: SpeculativeConfig = self.vllm_config.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    def set_input_ids_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        num_tokens: int,
        last_token_indices: torch.Tensor,
    ) -> None:
        self.input_ids[:num_tokens] = target_token_ids

    def load_model(self, target_model: Any) -> None:
        """Takes target_model to satisfy the type checker."""

        # This must be computed before loading the draft model
        # because that mutates the forward_context of the vllm_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys()
        )

        from vllm.compilation.backends import set_model_tag

        draft_vllm_config: VllmConfig = create_vllm_config_for_draft_model(
            target_model_vllm_config=self.vllm_config
        )
        logger.info(
            "Starting to load draft model %s. TP=%d, rank=%d",
            draft_vllm_config.model_config.model,
            draft_vllm_config.parallel_config.tensor_parallel_size,
            draft_vllm_config.parallel_config.rank,
        )
        with set_model_tag("draft_model"):
            self.model = get_model(vllm_config=draft_vllm_config, prefix="draft_model")

        # This must be computed after loading the draft model
        # because that mutates the forward_context of the vllm_config
        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys()
            - target_attn_layer_names
        )
        self.attn_layer_names = list(draft_attn_layer_names)


def create_vllm_config_for_draft_model(
    target_model_vllm_config: VllmConfig,
) -> VllmConfig:
    """The vllm_config is configured for the target model, e.g.
    its quant_config and parallel_config. But the draft model is potentially
    quantized differently, and has potentially different tensor_parallel_size.
    This function creates a new vllm_config configured for the draft model.
    The vllm_config is useful when loading the draft model with get_model().
    """
    old = target_model_vllm_config
    new_parallel_config = old.speculative_config.draft_parallel_config.replace(
        rank=old.parallel_config.rank
    )
    new: VllmConfig = old.replace(
        quant_config=None,  # quant_config is recomputed in __init__()
        model_config=old.speculative_config.draft_model_config,
        parallel_config=new_parallel_config,
    )
    return new


@dataclass
class DraftModelInputs:
    token_ids: torch.Tensor
    positions: torch.Tensor
    cad: CommonAttentionMetadata

@triton.jit
def _fused_extend_kernel(
    # Pointers
    old_tokens_ptr, old_pos_ptr,       # Input sequences
    new_tokens_ptr, new_pos_ptr,       # Output sequences
    next_tokens_ptr,                   # New tokens to append [K, B]
    base_end_pos_ptr,                  # Last position of existing seqs [B]
    query_start_loc_ptr,               # Start indices of existing seqs [B+1]
    
    # Strides
    stride_next_k, stride_next_b,      # Strides for next_token_ids
    
    # Shapes / Constants
    K,                                 # Number of tokens to extend per sequence
    BLOCK_SIZE: tl.constexpr           # Triton block size
):
    # Map the program ID to a specific sequence (batch index)
    seq_idx = tl.program_id(0)
    
    # Get the start and end of the *original* sequence
    # query_start_loc usually has shape [B+1], so we load i and i+1
    curr_start = tl.load(query_start_loc_ptr + seq_idx)
    next_start = tl.load(query_start_loc_ptr + seq_idx + 1)
    old_len = next_start - curr_start
    
    # Calculate the start index in the *new* buffer.
    # Logic: The previous sequences (0 to seq_idx-1) have all grown by K.
    # So the shift is exactly (seq_idx * K).
    new_start_offset = curr_start + (seq_idx * K)
    
    # The total length of the sequence after extension
    total_new_len = old_len + K

    # Handle the block offset for the sequence
    block_offset = tl.program_id(1) * BLOCK_SIZE
    offs = block_offset + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent writing out of bounds for this specific sequence
    mask = offs < total_new_len

    # --- Logic: Am I processing an old token or a new token? ---
    is_old_data = offs < old_len
    
    # 1. Handle Token IDs
    # -------------------
    
    # Path A: Read from old tokens (simple copy)
    # We use 'other=0' to prevent OOB reads, though mask handles the store.
    old_val_ptr = old_tokens_ptr + curr_start + offs
    old_vals = tl.load(old_val_ptr, mask=is_old_data, other=0)
    
    # Path B: Read from next_token_ids
    # We need to map the flat offset 'offs' to the K index.
    # k_idx = offs - old_len. 
    # Example: if old_len is 100, and offs is 101, this is the 2nd new token (index 1).
    k_indices = offs - old_len
    
    # Calculate pointer for next_tokens. 
    # Assumes next_token_ids is [K, B] or similar, handled by strides.
    # Address = base + (k_idx * stride_k) + (seq_idx * stride_b)
    next_val_ptr = next_tokens_ptr + (k_indices * stride_next_k) + (seq_idx * stride_next_b)
    
    # Only load if we are in the "new" section and within total bounds
    is_new_data = ~is_old_data & mask
    new_vals = tl.load(next_val_ptr, mask=is_new_data, other=0)
    
    # Merge and Store
    res_tokens = tl.where(is_old_data, old_vals, new_vals)
    tl.store(new_tokens_ptr + new_start_offset + offs, res_tokens, mask=mask)

    # 2. Handle Positions
    # -------------------
    
    # Path A: Read old positions
    old_pos_ptr_loc = old_pos_ptr + curr_start + offs
    old_pos_vals = tl.load(old_pos_ptr_loc, mask=is_old_data, other=0)
    
    # Path B: Calculate new positions
    # Logic: base_end_positions[seq_idx] + 1 + k_index
    # We load the base_end_pos just once per program instance (it's uniform for the seq)
    base_pos = tl.load(base_end_pos_ptr + seq_idx)
    calc_pos_vals = base_pos + 1 + k_indices
    
    # Merge and Store
    res_pos = tl.where(is_old_data, old_pos_vals, calc_pos_vals)
    tl.store(new_pos_ptr + new_start_offset + offs, res_pos, mask=mask)


def extend_seqs_fused(
    inputs: DraftModelInputs,
    cad: CommonAttentionMetadata,
    next_token_ids: torch.Tensor,
):
    """
    Drop-in replacement for the loop-based extension.
    """
    # 1. Prepare Metadata
    batch_size = cad.query_start_loc.shape[0] - 1
    num_extend_tokens = next_token_ids.shape[0] # K
    
    if num_extend_tokens == 0:
        return inputs.token_ids, inputs.positions

    # Get max sequence length to determine grid size
    # (Computation on CPU is negligible compared to repeated kernel launches)
    seqlens = cad.query_start_loc_cpu[1:] - cad.query_start_loc_cpu[:-1]
    max_new_len = seqlens.max().item() + num_extend_tokens
    
    # 2. Allocate Output Tensors
    total_new_tokens = inputs.token_ids.shape[0] + (batch_size * num_extend_tokens)
    
    new_token_ids = torch.empty(
        total_new_tokens, dtype=inputs.token_ids.dtype, device=inputs.token_ids.device
    )
    new_positions = torch.empty(
        total_new_tokens, dtype=inputs.positions.dtype, device=inputs.positions.device
    )

    # 3. Prepare Helper Inputs
    # We need base_end_positions. In your original code:
    # query_end_locs = cad.query_start_loc[1:] - 1
    # base_end_positions = inputs.positions[query_end_locs]
    # We compute this once here.
    query_end_locs = cad.query_start_loc[1:] - 1
    base_end_positions = inputs.positions[query_end_locs]

    # 4. Launch Config
    BLOCK_SIZE = 1024
    # Grid: (Batch_Size, ceil(Max_Len / Block))
    # This handles variable length sequences efficiently.
    grid = (batch_size, triton.cdiv(max_new_len, BLOCK_SIZE))

    # 5. Launch
    _fused_extend_kernel[grid](
        # Input Pointers
        inputs.token_ids,
        inputs.positions,
        new_token_ids,
        new_positions,
        next_token_ids,
        base_end_positions,
        cad.query_start_loc,
        # Strides for next_token_ids (K, B)
        next_token_ids.stride(0),
        next_token_ids.stride(1),
        # Constants
        num_extend_tokens,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return new_token_ids, new_positions

def merge_next_token_ids_into_token_ids(
    inputs: DraftModelInputs,
    next_token_ids: torch.Tensor,
    block_size: int,
    max_model_len: int,
    arange: torch.Tensor,
) -> DraftModelInputs:
    """
    Merges the next token ids with the existing token ids into a flat sequence.
    Does the same for the positions, computes new slot mapping,
    and updates the common_attn_metadata. The inputs are not modified in-place.
    """
    cad: CommonAttentionMetadata = inputs.cad
    
    num_extend_tokens = next_token_ids.shape[0]
    new_token_ids, new_positions = extend_seqs_fused(
        inputs=inputs, 
        cad=inputs.cad, 
        next_token_ids=next_token_ids
    )
    # query_end_locs = cad.query_start_loc[1:] - 1
    # base_end_positions = inputs.positions[query_end_locs]
    # new_positions = inputs.positions
    # new_token_ids = inputs.token_ids
    # for i in range(num_extend_tokens):
    #     # merge token_ids and next_token_ids
    #     new_tok_vals = next_token_ids[i]
    #     new_token_ids = extend_flat_seqs(
    #         seqs=new_token_ids, end_locs=query_end_locs, new_vals=new_tok_vals
    #     )
    #     # append new positions
    #     positions_to_append = base_end_positions + 1 + i
    #     new_positions = extend_flat_seqs(
    #         seqs=new_positions, end_locs=query_end_locs, new_vals=positions_to_append
    #     )
    #     query_end_locs = query_end_locs + arange[:query_end_locs.shape[0]] + 1

    # recompute slot mapping
    batch_size, n_blocks_per_req = cad.block_table_tensor.shape
    req_indices = torch.arange(batch_size, device=cad.query_start_loc.device)
    req_indices = torch.repeat_interleave(req_indices, cad.query_lens() + num_extend_tokens)
    block_table_indices = req_indices * n_blocks_per_req + new_positions // block_size
    block_nums = cad.block_table_tensor.view(-1)[block_table_indices]
    block_offsets = new_positions % block_size
    new_slot_mapping = block_nums * block_size + block_offsets

    if cad.max_seq_len + 2 * num_extend_tokens >= max_model_len:
        # Mask out the position ids that exceed the max model length.
        exceeds_max_model_len = new_positions >= max_model_len
        new_slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

    # update common_attn_metadata
    new_cad: CommonAttentionMetadata = extend_all_queries(
        cad, arange=arange, new_slot_mapping=new_slot_mapping, num_extend_tokens=num_extend_tokens
    )
    return DraftModelInputs(
        token_ids=new_token_ids, positions=new_positions, cad=new_cad
    )
