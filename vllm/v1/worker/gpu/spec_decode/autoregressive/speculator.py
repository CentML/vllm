# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker import die_local_stats
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.glue_pdl import glue_pdl
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode import draft_vocab_head
from vllm.v1.worker.gpu.spec_decode.autoregressive import one_graph
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup, get_uniform_decode_token_count

logger = init_logger(__name__)

# Draft glue riders. Both default off; copies only, so
# every buffer they write holds the same bits as with the flags off.
# VLLM_DRAFT_TAIL_FUSE=1: the draft prefill's four tail writes (draft_tokens[:, 0],
#   the next pass's hidden states, positions, sample_src_positions) run as one
#   kernel, as _update_draft_inputs_kernel does for the later draft passes,
#   instead of four copies (memcpy nodes in the draft graphs).
# VLLM_DRAFT_PRELUDE_FUSE=1: the copy of the target hidden states into the
#   drafter's input buffer runs in the _prepare_prefill_inputs_kernel launch
#   (one extra program per row) instead of as an eager memcpy, for steps of at
#   most VLLM_DRAFT_PRELUDE_FUSE_MAX_ROWS (default 512) padded tokens.
DRAFT_TAIL_FUSE = os.environ.get("VLLM_DRAFT_TAIL_FUSE", "0") == "1"
DRAFT_PRELUDE_FUSE = os.environ.get("VLLM_DRAFT_PRELUDE_FUSE", "0") == "1"
DRAFT_PRELUDE_FUSE_MAX_ROWS = int(
    os.environ.get("VLLM_DRAFT_PRELUDE_FUSE_MAX_ROWS", "512")
)
_draft_glue_logged: set[str] = set()


def _draft_glue_log_once(key: str, warn: bool, msg: str, *args: Any) -> None:
    if key in _draft_glue_logged:
        return
    _draft_glue_logged.add(key)
    (logger.warning if warn else logger.info)(msg, *args)


def _shares_storage(
    srcs: tuple[torch.Tensor | None, ...], dsts: tuple[torch.Tensor, ...]
) -> bool:
    dst_ptrs = {t.untyped_storage().data_ptr() for t in dsts}
    return any(
        t is not None and t.untyped_storage().data_ptr() in dst_ptrs for t in srcs
    )


class AutoRegressiveSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.sample_src_positions = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )

        self.inputs_embeds: torch.Tensor | None = None

        self.prefill_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.use_fused_multi_step_decode = False

        # Draft prefill row pruning (enable_draft_out_rows): every draft model
        # call passes ``out_rows``, the rows its layer keeps after attention.
        self.draft_out_rows = False
        self._draft_row_ids: torch.Tensor | None = None

        # VLLM_DRAFT_ONE_GRAPH (one_graph.py); set by capture_one_graphs().
        self._one_graphs: one_graph.DraftOneGraphs | None = None

    def enable_draft_out_rows(self) -> None:
        """Draft prefill: the draft model runs everything after its attention
        only on the rows the speculator samples (each request's last accepted
        row), instead of on every verified row. Its attention still writes the
        draft KV of every row. Draft decode steps keep all rows (identity).
        Requires a model whose forward takes ``out_rows``.
        """
        self.draft_out_rows = True
        self._draft_row_ids = torch.arange(
            max(self.max_num_tokens, self.max_num_reqs),
            dtype=torch.int64,
            device=self.device,
        )

    def _uses_draft_out_rows(self) -> bool:
        return self.draft_out_rows and self.pcp_manager is None

    def _prefill_out_rows(
        self, num_reqs: int, num_tokens: int, cudagraph_runtime_mode: CUDAGraphMode
    ) -> int:
        """Rows the draft prefill keeps (a prefix of last_token_indices, whose
        tail past the real requests is zero-padded).
        """
        if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            # Piecewise graphs are captured per token count, so the row count
            # must follow the (padded) token count, not the request count.
            return min(num_tokens, self.max_num_reqs)
        if cudagraph_runtime_mode == CUDAGraphMode.FULL:
            # The graph's (padded) request count.
            return num_reqs
        # Eager. At least 2 rows: torch.compile specializes a dynamic dim seen
        # as 1 on the compiling call.
        return min(max(num_reqs, 2), self.max_num_reqs)

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if not self.supports_mm_inputs:
            return

        self.inputs_embeds = torch.zeros(
            self.max_num_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

    # Lifecycle hooks for model-specific optimizations. Subclasses override
    # the ones they need. These fire in both `capture` and `propose` so that
    # any state they toggle (e.g. attention flags baked into a CUDA graph) is
    # identical at capture time and replay time.
    def on_prefill_begin(self, num_reqs: int) -> None: ...

    def on_prefill_end(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_begin(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_end(self, num_reqs: int) -> None: ...

    @property
    def advance_draft_positions(self) -> bool:
        """Whether to increment positions and seq_lens between draft steps.

        True for Eagle/standard MTP (each step produces new KV).
        False for Gemma4 MTP (Q-only, shares target KV, constant positions).
        """
        return True

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._configure_fused_multi_step_decode()

    def _configure_fused_multi_step_decode(self) -> None:
        if self.num_speculative_steps == 1:
            self.use_fused_multi_step_decode = False
            return

        if not self.advance_draft_positions:
            self.use_fused_multi_step_decode = True
            return

        unsupported_backends = sorted(
            {
                attn_group.backend.get_name()
                for attn_groups in self.attn_groups
                for attn_group in attn_groups
                if not attn_group.supports_draft_decode_metadata_update
            }
        )
        self.use_fused_multi_step_decode = not unsupported_backends
        if unsupported_backends:
            logger.info_once(
                "Fused multi-step draft decode is not supported by attention "
                "backend(s) %s; falling back to rebuilding attention metadata "
                "between draft steps.",
                ", ".join(unsupported_backends),
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Initialize cudagraph manager for draft prefill (draft position 0).
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
        )

        # PIECEWISE cudagraphs are not supported for draft decodes.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        # Initialize cudagraph manager for draft decodes (draft positions > 0).
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=1,
        )

    def capture(self) -> None:
        logger.info("Capturing model for speculator...")
        # Reset indices to zeros to prevent stale values from prior
        # dummy runs to cause out-of-bounds indexing during capture.
        self.last_token_indices.zero_()
        self.idx_mapping.zero_()

        # Capture the prefill routine (model forward + compute_logits +
        # sample).
        # For FULL graphs, the entire routine is recorded as one graph.
        # For PIECEWISE, only the model's compiled regions are captured
        # and the rest (compute_logits, gumbel_sample) runs eagerly.
        # Draft prefill reuses the target model's attention metadata at
        # runtime, so capture builds its dummy metadata through the target
        # model runner's builders and buffers.
        assert self.prefill_cudagraph_manager is not None
        if self.prefill_cudagraph_manager.use_breakable_cg:
            self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)

        self.on_prefill_begin(self.max_num_reqs)
        self.prefill_cudagraph_manager.capture(
            self._prefill,
            self.model_state,
            self.target_input_buffers,
            self.block_tables,
            self.target_attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing prefill CUDA graphs",
        )
        self.on_prefill_end(self.max_num_reqs)

        if self.num_speculative_steps == 1:
            return

        self.on_multi_step_decode_begin(self.max_num_reqs)
        # Capture either the fused decode loop or one decode step per graph.
        assert self.decode_cudagraph_manager is not None
        decode_fn = (
            self._generate_fused_drafts
            if self.use_fused_multi_step_decode
            else self._generate_draft
        )
        self.decode_cudagraph_manager.capture(
            decode_fn,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing decode CUDA graphs",
        )
        self.on_multi_step_decode_end(self.max_num_reqs)

    def capture_one_graphs(
        self, target_tokens_padded: Callable[[int], int | None]
    ) -> None:
        """VLLM_DRAFT_ONE_GRAPH: capture the single-graph draft (after capture())."""
        if not one_graph.ENABLED:
            return
        reason = one_graph.unsupported_reason(self)
        if reason is not None:
            logger.warning("VLLM_DRAFT_ONE_GRAPH not applied: %s", reason)
            return
        graphs = one_graph.DraftOneGraphs(self)
        graphs.capture(target_tokens_padded)
        if not graphs.graphs:
            return
        self._one_graphs = graphs
        # Compile the COPY_REJECTED variant of _copy_idx_mapping_kernel now
        # (same argument dtypes as InputBatch: int64 idx_mapping, int32
        # num_rejected); every propose() launches it from here on, so it must
        # not JIT during inference.
        self._copy_request_inputs(
            1,
            torch.zeros(1, dtype=torch.int64, device=self.device),
            self.temperature,
            self.seeds,
            num_rejected=graphs.num_rejected[:1],
            num_rejected_out=graphs.num_rejected,
        )
        torch.accelerator.synchronize()

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_tokens = input_batch.num_tokens
        num_tokens_padded = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        max_query_len = input_batch.num_scheduled_tokens.max()
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_speculative_steps, self.max_model_len
        )

        # NOTE(woosuk): To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions. By doing so,
        # we can also reuse the attention metadata (e.g., query_start_loc,
        # seq_lens) of the target model.
        if aux_hidden_states:
            assert self.method == "eagle3"
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        one = self._one_graphs
        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
            # VLLM_DRAFT_ONE_GRAPH: static num_rejected, same launch.
            num_rejected=num_rejected if one is not None else None,
            num_rejected_out=one.num_rejected if one is not None else None,
        )

        # VLLM_DRAFT_PRELUDE_FUSE: the hidden-state copy below, in this launch.
        hidden_copy = (
            _prelude_hidden_copy(self.hidden_states, hidden_states, num_tokens_padded)
            if DRAFT_PRELUDE_FUSE and self.pcp_manager is None
            else None
        )
        # Get the input ids and last token indices for the speculator.
        prepare_prefill_inputs(
            self.last_token_indices,
            self.current_draft_step,
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
            hidden_copy=hidden_copy,
        )

        if self.pcp_manager is not None:
            self.pcp_manager.prepare_draft_prefill(
                input_batch,
                self.input_buffers.input_ids[:num_tokens_padded],
            )
            prefill = self.pcp_manager.draft_prefill_batch
            if prefill is not None:
                num_tokens = prefill.num_tokens
                num_tokens_padded = prefill.num_tokens_after_padding
        if hidden_copy is None:
            self.hidden_states[:num_tokens_padded].copy_(hidden_states)

        # When all requests are decoding (no true prefills), each has
        # num_speculative_steps + 1 tokens, enabling FULL graph replay.
        uniform_token_count = get_uniform_decode_token_count(
            num_reqs,
            # Use the actual number of tokens without padding added by
            # the target model during FULL cudagraph.
            num_tokens,
            max_query_len,
            input_batch.has_prefill,
        )
        prefill_batch_desc, prefill_batch_sync = dispatch_cg_and_sync_dp(
            self.prefill_cudagraph_manager,
            num_reqs,
            num_tokens_padded,
            uniform_token_count,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
            dp_sync=dp_sync,
        )
        num_tokens_across_dp = (
            prefill_batch_sync.num_tokens_across_dp
            if prefill_batch_sync is not None
            else None
        )

        self._prepare_eplb_forward(num_tokens)

        if (
            one is not None
            and not (dummy_run or is_profile)
            and dp_sync is None
            and self.pcp_manager is None
            and input_batch.seq_lens.data_ptr()
            == self.target_input_buffers.seq_lens.data_ptr()
            and (hit := one.lookup(num_reqs, prefill_batch_desc)) is not None
        ):
            graph, decode_batch_desc = hit
            self.on_prefill_begin(num_reqs)
            # The draft-decode attention metadata build, hoisted before the
            # draft prefill (one_graph.py: valid for its accepted backends).
            self._build_uniform_attn_metadata(
                num_reqs=num_reqs,
                batch_desc=decode_batch_desc,
                num_query_per_req=1,
                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                step=1,
            )
            get_offloader().sync_prev_onload()
            graph.replay()
            if die_local_stats.ON:
                die_local_stats.replayed(graph)
            one.replays += 1
            if one.replays == 1:
                logger.info("Draft one-graph: first replay (%d requests)", num_reqs)
            self.on_prefill_end(num_reqs)
            self.on_multi_step_decode_begin(num_reqs)
            self.on_multi_step_decode_end(num_reqs)
            if self.draft_vocab_head is not None:
                draft_vocab_head.observe_target(
                    self.draft_vocab_head,
                    last_sampled,
                    input_batch.idx_mapping,
                    num_sampled,
                )
                draft_vocab_head.maybe_log(self.draft_vocab_head)
            return self.draft_tokens[:num_reqs]

        self.on_prefill_begin(num_reqs)
        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Replay the full graph for draft prefill.
            assert self.prefill_cudagraph_manager is not None
            self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
        else:
            # The target model's attention metadata and slot mappings
            # can directly be used for draft prefill, because of the
            # identical batch shape and KV cache layout.
            self._prefill(
                num_reqs,
                prefill_batch_desc.num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=prefill_batch_desc.cg_mode,
                mm_inputs=mm_inputs,
            )
        self.on_prefill_end(num_reqs)

        if self.num_speculative_steps == 1:
            # Early exit.
            return self.draft_tokens[:num_reqs, :1]

        if self.pcp_manager is not None and not dummy_run:
            self.block_tables.gather_block_tables(
                input_batch.idx_mapping, num_reqs_padded=num_reqs
            )

        # Prepare the inputs for the decode steps.
        prepare_decode_inputs(
            self.draft_tokens[:num_reqs, 0],
            input_batch.seq_lens,
            num_rejected,
            self.input_buffers,
            self.sample_src_positions,
            self.max_model_len,
            self.max_num_reqs,
            advance_draft_positions=self.advance_draft_positions,
        )

        decode_batch_sync, num_batch_tokens = (
            self._build_uniform_batch_dp_sync(dp_sync, num_reqs, num_query_per_req=1)
            if dp_sync is not None
            else (None, num_reqs)
        )
        # Each request produces exactly 1 token per draft generation step,
        # enabling FULL graph replay.
        decode_batch_desc, decode_batch_sync = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
            num_reqs,
            num_batch_tokens,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
            dp_sync=decode_batch_sync,
        )
        num_tokens_across_dp = (
            decode_batch_sync.num_tokens_across_dp
            if decode_batch_sync is not None
            else None
        )

        self.on_multi_step_decode_begin(num_reqs)
        # Generate the remaining num_speculative_steps - 1 draft tokens.
        decode_fn = (
            self._fused_multi_step_decode
            if self.use_fused_multi_step_decode
            else self._multi_step_decode
        )
        decode_fn(
            num_reqs,
            dummy_run and skip_attn_for_dummy_run,
            decode_batch_desc,
            num_tokens_across_dp,
            input_batch.seq_lens_cpu_upper_bound,
        )
        self.on_multi_step_decode_end(num_reqs)
        if self.draft_vocab_head is not None and not (dummy_run or is_profile):
            draft_vocab_head.observe_target(
                self.draft_vocab_head,
                last_sampled,
                input_batch.idx_mapping,
                num_sampled,
            )
            draft_vocab_head.maybe_log(self.draft_vocab_head)

        return self.draft_tokens[:num_reqs]

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        out_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_buffers: InputBatch | InputBuffers = self.input_buffers
        is_padding = None
        if self.pcp_manager is not None:
            input_buffers = self.pcp_manager.get_draft_input_buffers(self.input_buffers)
            is_padding = input_buffers.is_padding[:num_tokens]
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
            is_padding=is_padding,
        ):
            inputs_embeds = None
            if self.supports_mm_inputs:
                assert self.inputs_embeds is not None
                # Merge multimodal embeddings with input ids.
                mm_embeds, is_mm_embed = mm_inputs or (None, None)
                num_input_tokens = (
                    is_mm_embed.shape[0] if is_mm_embed is not None else num_tokens
                )
                self.inputs_embeds[:num_input_tokens] = self.model.embed_input_ids(
                    input_buffers.input_ids[:num_input_tokens],
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
                inputs_embeds = self.inputs_embeds[:num_tokens]

            model_inputs = dict(
                input_ids=input_buffers.input_ids[:num_tokens],
                positions=input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
            if self._uses_draft_out_rows():
                # Every call passes out_rows (one compiled graph): draft decode
                # steps keep all rows.
                assert self._draft_row_ids is not None
                if out_rows is None:
                    out_rows = self._draft_row_ids[:num_tokens]
                model_inputs["out_rows"] = out_rows
            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                # Draft prefill with PIECEWISE cudagraph (compiled PW or breakable),
                # chosen inside run_pw_graph.
                assert self.prefill_cudagraph_manager is not None
                ret_hidden_states = self.prefill_cudagraph_manager.run_pw_graph(
                    self.model, model_inputs
                )
            else:
                # Eager (NONE): call the raw model directly.
                ret_hidden_states = self.model(**model_inputs)
        # Some MTP models declare a single-tensor contract but return
        # (logits_hidden, feedback_hidden) for final-norm correctness.
        if isinstance(ret_hidden_states, tuple):
            last_hidden_states, hidden_states = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

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
        last_token_indices = self.last_token_indices[:num_reqs]
        positions = self.input_buffers.positions[last_token_indices]
        # The output hidden state at position P (= positions) and the token id
        # at P+1 are used to draft the token at P+2. Sampling keys a draw by the
        # position before the sampled token, so the net adjustment is +1.
        sample_src_positions = positions + 1
        idx_mapping = self.idx_mapping[:num_reqs]

        out_rows = None
        if self._uses_draft_out_rows():
            out_rows = self.last_token_indices[
                : self._prefill_out_rows(num_reqs, num_tokens, cudagraph_runtime_mode)
            ]
        last_hidden_states, hidden_states = self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            mm_inputs=mm_inputs,
            out_rows=out_rows,
        )
        if self.pcp_manager is not None:
            last_hidden_states, hidden_states = self.pcp_manager.restore_draft_prefill(
                last_hidden_states, hidden_states
            )

        if out_rows is not None:
            # Row i of the output is already row last_token_indices[i].
            sample_hidden_states = last_hidden_states[:num_reqs]
            sampled_feedback = hidden_states[:num_reqs]
        else:
            sample_hidden_states = last_hidden_states[last_token_indices]
            sampled_feedback = None
        sampled = self.sample_draft(
            sample_hidden_states,
            sample_src_positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )
        if DRAFT_TAIL_FUSE:
            # The writes below as one kernel (same source rows).
            if sampled_feedback is not None:
                src, src_rows = sampled_feedback, None
            elif last_hidden_states is hidden_states:
                src, src_rows = sample_hidden_states, None
            else:
                src, src_rows = hidden_states, last_token_indices
            if self._fused_prefill_tail(
                num_reqs, sampled, src, src_rows, positions, sample_src_positions
            ):
                return
        self.draft_tokens[:num_reqs, 0] = sampled
        if sampled_feedback is not None:
            self.hidden_states[:num_reqs] = sampled_feedback
        elif last_hidden_states is hidden_states:
            self.hidden_states[:num_reqs] = sample_hidden_states
        else:
            self.hidden_states[:num_reqs] = hidden_states[last_token_indices]
        self.input_buffers.positions[:num_reqs] = positions
        self.sample_src_positions[:num_reqs] = sample_src_positions

    def _fused_prefill_tail(
        self,
        num_reqs: int,
        sampled: torch.Tensor,
        src: torch.Tensor,
        src_rows: torch.Tensor | None,
        positions: torch.Tensor,
        sample_src_positions: torch.Tensor,
    ) -> bool:
        """VLLM_DRAFT_TAIL_FUSE: _prefill's tail writes as _draft_tail_kernel.

        draft_tokens[i, 0] = sampled[i], hidden_states[i] = src[src_rows[i]]
        (src[i] without src_rows), positions[i] and sample_src_positions[i]
        for i < num_reqs. Returns False, so the caller runs the stock copies,
        unless every operand is one the kernel copies bit for bit (same dtypes,
        exact shapes, unit inner strides, no source sharing a destination's
        storage).
        """
        hidden = self.hidden_states
        out_positions = self.input_buffers.positions
        out_sample_src = self.sample_src_positions
        vectors = (sampled, positions, sample_src_positions)
        reason = None
        if num_reqs < 1:
            reason = "no requests"
        elif any(
            t.dim() != 1 or t.shape[0] != num_reqs or not t.is_contiguous()
            for t in vectors
        ):
            reason = "token or position operand shape"
        elif (
            sampled.dtype != self.draft_tokens.dtype
            or positions.dtype != out_positions.dtype
            or sample_src_positions.dtype != out_sample_src.dtype
        ):
            reason = "token or position dtype"
        elif (
            src.dim() != 2
            or src.shape[1] != hidden.shape[1]
            or src.dtype != hidden.dtype
            or src.stride(1) != 1
            or (src_rows is None and src.shape[0] != num_reqs)
        ):
            reason = "hidden-state operand"
        elif src_rows is not None and (
            src_rows.dim() != 1
            or src_rows.shape[0] != num_reqs
            or not src_rows.is_contiguous()
        ):
            reason = "row indices"
        elif _shares_storage(
            (sampled, src, src_rows, positions, sample_src_positions),
            (self.draft_tokens, hidden, out_positions, out_sample_src),
        ):
            reason = "a source shares a destination's storage"
        if reason is not None:
            _draft_glue_log_once(
                "tail-off:" + reason,
                True,
                "VLLM_DRAFT_TAIL_FUSE: stock draft-prefill tail copies (%s)",
                reason,
            )
            return False
        hidden_size = hidden.shape[1]
        _draft_tail_kernel[(num_reqs,)](
            sampled,
            self.draft_tokens,
            self.draft_tokens.stride(0),
            src,
            src.stride(0),
            src_rows if src_rows is not None else positions,
            hidden,
            hidden.stride(0),
            hidden_size,
            positions,
            out_positions,
            sample_src_positions,
            out_sample_src,
            BLOCK_SIZE=min(triton.next_power_of_2(hidden_size), 4096),
            GATHER=src_rows is not None,
            GLUE_PDL=glue_pdl(),
            launch_pdl=glue_pdl(),
        )
        _draft_glue_log_once(
            "tail-on",
            False,
            "Draft tail fuse on (VLLM_DRAFT_TAIL_FUSE=1): draft prefill tail as "
            "_draft_tail_kernel (draft_tokens[:, 0], hidden states%s, positions, "
            "sample_src_positions; bit-identical copies), first %d requests",
            " gathered" if src_rows is not None else "",
            num_reqs,
        )
        return True

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
    ) -> None:
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
            # Rebuild every step when positions advance, or just once
            # on the first step when positions are constant (Gemma4 MTP).
            if not skip_attn and (self.advance_draft_positions or step == 1):
                slot_mappings = self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    batch_desc.num_tokens,
                )
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
                attn_metadata = self._build_uniform_attn_metadata(
                    num_reqs=num_reqs,
                    batch_desc=batch_desc,
                    num_query_per_req=1,
                    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                    step=step,
                )

            self.current_draft_step.fill_(step)

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode,
                )

    def _fused_multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
    ) -> None:
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings_by_layer = None
        if not skip_attn:
            slot_mappings = self.block_tables.compute_slot_mappings(
                idx_mapping,
                query_start_loc,
                positions,
                batch_desc.num_tokens,
            )
            if batch_desc.cg_mode != CUDAGraphMode.FULL:
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
            attn_metadata = self._build_uniform_attn_metadata(
                num_reqs=num_reqs,
                batch_desc=batch_desc,
                num_query_per_req=1,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=1,
            )

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            return

        self._generate_fused_drafts(
            num_reqs,
            batch_desc.num_tokens,
            attn_metadata,
            slot_mappings_by_layer,
            num_tokens_across_dp,
            batch_desc.cg_mode,
        )

    def _generate_fused_drafts(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        attn_groups = (
            [group for groups in self.attn_groups for group in groups]
            if attn_metadata is not None
            else []
        )

        for step in range(1, self.num_speculative_steps):
            self.current_draft_step.fill_(step)
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            if (
                step < self.num_speculative_steps - 1
                and attn_metadata is not None
                and self.advance_draft_positions
            ):
                self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    num_tokens_padded,
                )
                for attn_group in attn_groups:
                    attn_group.update_draft_decode_metadata(attn_metadata)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self._prepare_eplb_forward(num_reqs)

        idx_mapping = self.idx_mapping[:num_reqs]
        # Run the draft model forward pass.
        last_hidden_states, hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        # Sample the draft tokens.
        sample_hidden_states = last_hidden_states[:num_reqs]
        sample_src_positions = self.sample_src_positions[:num_reqs]
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            sample_src_positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )

        # Update the inputs for the next step.
        update_draft_inputs(
            draft_tokens,
            self.current_draft_step,
            hidden_states,
            self.draft_tokens,
            self.hidden_states,
            self.input_buffers,
            self.sample_src_positions,
            num_reqs,
            self.max_model_len,
            self.num_speculative_steps,
            advance_draft_positions=self.advance_draft_positions,
        )


@triton.jit
def _prepare_prefill_inputs_kernel(
    last_token_indices_ptr,
    draft_current_step_ptr,
    draft_input_ids_ptr,
    draft_positions_ptr,
    draft_query_start_loc_ptr,
    draft_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
    GLUE_PDL: tl.constexpr = False,
):
    if GLUE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)

    # Get the true query length and next token after accounting for rejected tokens.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    query_len -= num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling.
        # Get the next prefill token.
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # Shift target_input_ids by one.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(draft_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(draft_input_ids_ptr + last_token_index, next_token)

    # Copy positions.
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(draft_positions_ptr + query_start + block, target_pos, mask=mask)

    # Copy query start locations.
    tl.store(draft_query_start_loc_ptr + req_idx, query_start)
    # Copy sequence lengths.
    tl.store(draft_seq_lens_ptr + req_idx, seq_len)
    if req_idx == (num_reqs - 1):
        # Reset the current draft step to 0.
        tl.store(draft_current_step_ptr, 0)
        # Pad query_start_loc for CUDA graphs.
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(draft_query_start_loc_ptr + block, query_end, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(draft_seq_lens_ptr + block, 0, mask=mask)
        # Pad last_token_indices for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


@triton.jit(do_not_specialize=["num_rows"])
def _prepare_prefill_inputs_hidden_kernel(
    last_token_indices_ptr,
    draft_current_step_ptr,
    draft_input_ids_ptr,
    draft_positions_ptr,
    draft_query_start_loc_ptr,
    draft_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_num_reqs,
    hidden_src_ptr,
    hidden_src_stride,
    hidden_dst_ptr,
    hidden_dst_stride,
    hidden_size,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    GLUE_PDL: tl.constexpr = False,
):
    """VLLM_DRAFT_PRELUDE_FUSE: programs (r, 0) are _prepare_prefill_inputs_kernel
    (grid axis 0 is still the request count); programs (r, c >= 1) copy row
    (c - 1) * num_reqs + r of hidden_dst[:num_rows] = hidden_src.
    """
    copy_idx = tl.program_id(1)
    if copy_idx == 0:
        _prepare_prefill_inputs_kernel(
            last_token_indices_ptr,
            draft_current_step_ptr,
            draft_input_ids_ptr,
            draft_positions_ptr,
            draft_query_start_loc_ptr,
            draft_seq_lens_ptr,
            target_input_ids_ptr,
            target_positions_ptr,
            idx_mapping_ptr,
            last_sampled_ptr,
            next_prefill_tokens_ptr,
            num_sampled_ptr,
            num_rejected_ptr,
            query_start_loc_ptr,
            seq_lens_ptr,
            max_num_reqs,
            BLOCK_SIZE=BLOCK_SIZE,
            GLUE_PDL=GLUE_PDL,
        )
    else:
        if GLUE_PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        row = (copy_idx - 1) * tl.num_programs(0) + tl.program_id(0)
        if row < num_rows:
            src_row = hidden_src_ptr + row.to(tl.int64) * hidden_src_stride
            dst_row = hidden_dst_ptr + row.to(tl.int64) * hidden_dst_stride
            for i in range(0, hidden_size, HIDDEN_BLOCK):
                block = i + tl.arange(0, HIDDEN_BLOCK)
                mask = block < hidden_size
                h = tl.load(src_row + block, mask=mask)
                tl.store(dst_row + block, h, mask=mask)


def _prelude_hidden_copy(
    buffer: torch.Tensor, hidden_states: torch.Tensor, num_rows: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """VLLM_DRAFT_PRELUDE_FUSE: (src, dst) of buffer[:num_rows].copy_(hidden_states)
    when the prepare launch can do it bit for bit (same shape and dtype, unit
    inner strides, distinct storage) and num_rows is within
    VLLM_DRAFT_PRELUDE_FUSE_MAX_ROWS; else None (the caller copies).
    """
    if not 0 < num_rows <= DRAFT_PRELUDE_FUSE_MAX_ROWS:
        return None
    dst = buffer[:num_rows]
    reason = None
    if hidden_states.shape != dst.shape:
        reason = "shape"
    elif hidden_states.dtype != dst.dtype:
        reason = "dtype"
    elif hidden_states.stride(-1) != 1 or dst.stride(-1) != 1:
        reason = "inner stride"
    elif _shares_storage((hidden_states,), (dst,)):
        reason = "shared storage"
    if reason is not None:
        _draft_glue_log_once(
            "prelude-off:" + reason,
            True,
            "VLLM_DRAFT_PRELUDE_FUSE: eager hidden-state copy (%s)",
            reason,
        )
        return None
    _draft_glue_log_once(
        "prelude-on",
        False,
        "Draft prelude fuse on (VLLM_DRAFT_PRELUDE_FUSE=1): target hidden-state "
        "copy in the _prepare_prefill_inputs_hidden_kernel launch (bit-identical; "
        "steps of up to %d padded tokens), first %d rows",
        DRAFT_PRELUDE_FUSE_MAX_ROWS,
        num_rows,
    )
    return hidden_states, dst


def prepare_prefill_inputs(
    # [num_reqs]
    last_token_indices: torch.Tensor,
    current_draft_step: torch.Tensor,
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    max_num_reqs,
    # VLLM_DRAFT_PRELUDE_FUSE: (src, dst) of a dst.copy_(src) done in this launch.
    hidden_copy: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    num_reqs = input_batch.num_reqs
    if hidden_copy is not None:
        src, dst = hidden_copy
        num_rows, hidden_size = dst.shape
        _prepare_prefill_inputs_hidden_kernel[
            (num_reqs, 1 + triton.cdiv(num_rows, num_reqs))
        ](
            last_token_indices,
            current_draft_step,
            input_buffers.input_ids,
            input_buffers.positions,
            input_buffers.query_start_loc,
            input_buffers.seq_lens,
            input_batch.input_ids,
            input_batch.positions,
            input_batch.idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
            input_batch.seq_lens,
            max_num_reqs,
            src,
            src.stride(0),
            dst,
            dst.stride(0),
            hidden_size,
            num_rows,
            BLOCK_SIZE=1024,
            HIDDEN_BLOCK=min(triton.next_power_of_2(hidden_size), 4096),
            GLUE_PDL=glue_pdl(),
            launch_pdl=glue_pdl(),
        )
        return last_token_indices
    _prepare_prefill_inputs_kernel[(num_reqs,)](
        last_token_indices,
        current_draft_step,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        input_batch.input_ids,
        input_batch.positions,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        max_num_reqs,
        BLOCK_SIZE=1024,
        GLUE_PDL=glue_pdl(),
        launch_pdl=glue_pdl(),
    )
    return last_token_indices


@triton.jit
def _prepare_decode_inputs_kernel(
    draft_tokens_ptr,
    draft_tokens_stride,
    target_seq_lens_ptr,
    num_rejected_ptr,
    input_ids_ptr,
    positions_ptr,
    sample_src_positions_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_model_len,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
    GLUE_PDL: tl.constexpr = False,
):
    if GLUE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        # Compute query_start_loc. Pad it with the last query_start_loc
        # for CUDA graphs.
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block < num_reqs, block, num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    # draft token -> input id.
    draft_token = tl.load(draft_tokens_ptr + req_idx * draft_tokens_stride)
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Advance the draft sampling key.
    sample_position = tl.load(sample_src_positions_ptr + req_idx)
    tl.store(sample_src_positions_ptr + req_idx, sample_position + 1)

    target_seq_len = tl.load(target_seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    seq_len = target_seq_len - num_rejected
    if ADVANCE_DRAFT_POSITIONS:
        # Compute position and seq_lens.
        # NOTE(woosuk): To prevent out-of-range access, we clamp these values
        # if they reach the max model length.
        position = tl.load(positions_ptr + req_idx)
        position = tl.minimum(position + 1, max_model_len - 1)
        tl.store(positions_ptr + req_idx, position)
        seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def prepare_decode_inputs(
    draft_tokens: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    sample_src_positions: torch.Tensor,
    max_model_len: int,
    max_num_reqs: int,
    advance_draft_positions: bool = True,
):
    num_reqs = draft_tokens.shape[0]
    _prepare_decode_inputs_kernel[(num_reqs + 1,)](
        draft_tokens,
        draft_tokens.stride(0),
        target_seq_lens,
        num_rejected,
        input_buffers.input_ids,
        input_buffers.positions,
        sample_src_positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
        GLUE_PDL=glue_pdl(),
        launch_pdl=glue_pdl(),
    )


@triton.jit
def _update_draft_inputs_kernel(
    output_draft_tokens_ptr,
    output_draft_tokens_stride,
    next_input_hidden_states_ptr,
    next_input_hidden_states_stride,
    input_ids_ptr,
    positions_ptr,
    sample_src_positions_ptr,
    seq_lens_ptr,
    draft_tokens_ptr,
    current_draft_step_ptr,
    hidden_states_ptr,
    hidden_states_stride,
    hidden_size,
    max_model_len,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
    GLUE_PDL: tl.constexpr = False,
):
    if GLUE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    req_idx = tl.program_id(0)

    # Write the sampled draft token into self.draft_tokens[req_idx, step].
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    step = tl.load(current_draft_step_ptr)
    tl.store(
        output_draft_tokens_ptr + req_idx * output_draft_tokens_stride + step,
        draft_token,
    )

    if step >= num_speculative_steps - 1:
        # This is the final step. Skip updating draft forward inputs.
        return

    # Advance the draft sampling key.
    sample_position = tl.load(sample_src_positions_ptr + req_idx)
    tl.store(sample_src_positions_ptr + req_idx, sample_position + 1)

    # Write the sampled draft token into the input ids tensor for the next
    # forward pass.
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Copy hidden states into the input hidden states tensor for the next
    # forward pass.
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        hidden_states = tl.load(
            hidden_states_ptr + req_idx * hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            next_input_hidden_states_ptr
            + req_idx * next_input_hidden_states_stride
            + block,
            hidden_states,
            mask=mask,
        )

    if ADVANCE_DRAFT_POSITIONS:
        # Increment position and seq_lens.
        # NOTE(woosuk): To prevent out-of-range access, we clamp these values
        # if they reach the max model length.
        position = tl.load(positions_ptr + req_idx)
        position = tl.minimum(position + 1, max_model_len - 1)
        tl.store(positions_ptr + req_idx, position)

        seq_len = tl.load(seq_lens_ptr + req_idx)
        seq_len = tl.minimum(seq_len + 1, max_model_len)
        tl.store(seq_lens_ptr + req_idx, seq_len)


def update_draft_inputs(
    draft_tokens: torch.Tensor,
    current_draft_step: torch.Tensor,
    hidden_states: torch.Tensor,
    output_draft_tokens: torch.Tensor,
    next_input_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    sample_src_positions: torch.Tensor,
    num_reqs: int,
    max_model_len: int,
    num_speculative_steps: int,
    advance_draft_positions: bool = True,
):
    _, hidden_size = hidden_states.shape
    _update_draft_inputs_kernel[(num_reqs,)](
        output_draft_tokens,
        output_draft_tokens.stride(0),
        next_input_hidden_states,
        next_input_hidden_states.stride(0),
        input_buffers.input_ids,
        input_buffers.positions,
        sample_src_positions,
        input_buffers.seq_lens,
        draft_tokens,
        current_draft_step,
        hidden_states,
        hidden_states.stride(0),
        hidden_size,
        max_model_len,
        num_speculative_steps,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
        GLUE_PDL=glue_pdl(),
        launch_pdl=glue_pdl(),
    )


@triton.jit
def _draft_tail_kernel(
    sampled_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    src_hidden_ptr,
    src_hidden_stride,
    src_rows_ptr,
    hidden_states_ptr,
    hidden_states_stride,
    hidden_size,
    positions_ptr,
    out_positions_ptr,
    sample_src_positions_ptr,
    out_sample_src_positions_ptr,
    BLOCK_SIZE: tl.constexpr,
    GATHER: tl.constexpr,
    GLUE_PDL: tl.constexpr = False,
):
    """VLLM_DRAFT_TAIL_FUSE: the draft prefill's tail writes for one request."""
    if GLUE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    req_idx = tl.program_id(0)

    draft_token = tl.load(sampled_ptr + req_idx)
    tl.store(draft_tokens_ptr + req_idx * draft_tokens_stride, draft_token)

    if GATHER:
        row = tl.load(src_rows_ptr + req_idx).to(tl.int64)
    else:
        row = req_idx.to(tl.int64)
    dst_row = req_idx.to(tl.int64)
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        h = tl.load(src_hidden_ptr + row * src_hidden_stride + block, mask=mask)
        tl.store(
            hidden_states_ptr + dst_row * hidden_states_stride + block, h, mask=mask
        )

    position = tl.load(positions_ptr + req_idx)
    tl.store(out_positions_ptr + req_idx, position)
    sample_position = tl.load(sample_src_positions_ptr + req_idx)
    tl.store(out_sample_src_positions_ptr + req_idx, sample_position)
