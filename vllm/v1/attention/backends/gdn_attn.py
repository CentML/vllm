# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import MambaSpec


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None

    # Per-step copies shared by every GDN layer so no layer re-derives them
    # (set when num_prefills > 0): int64 state indices (ATen indexing converts
    # int32 per call), the inverted initial-state mask, and (FlashInfer
    # backend) int64 cu_seqlens as fi_chunk_gated_delta_rule wants.
    prefill_state_indices_i64: torch.Tensor | None = None
    prefill_no_initial_state: torch.Tensor | None = None
    prefill_query_start_loc_i64: torch.Tensor | None = None
    # When the spec and non-spec tokens of a mixed batch form two contiguous
    # blocks (checked on the CPU), their start rows; the forward then slices
    # instead of gathering by spec/non_spec_token_indx.
    spec_token_start: int | None = None
    non_spec_token_start: int | None = None


@dataclass(frozen=True)
class GDNBatchSplit:
    """How GDNAttentionMetadataBuilder.build() classifies a batch's rows."""

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    # Set only when the batch has spec-decode rows.
    spec_sequence_masks_cpu: torch.Tensor | None = None
    non_spec_sequence_masks_cpu: torch.Tensor | None = None
    query_lens_cpu: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    kv_cache_spec: MambaSpec
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl"]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            assert prefill_query_start_loc is not None
            assert prefill_query_start_loc_cpu is not None
            total_tokens = int(prefill_query_start_loc_cpu[-1].item())
            return prepare_metadata_cutedsl(
                prefill_query_start_loc,
                total_tokens,
                FLA_CHUNK_SIZE,
            )

        # Only prefill batches use FLA chunk ops.
        # Pre-compute on CPU and async-copy to GPU to avoid
        # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        assert prefill_query_start_loc_cpu is not None
        return (
            async_tensor_h2d(
                prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
            async_tensor_h2d(
                prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
        )

    def _split_batch(
        self,
        m: CommonAttentionMetadata,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> "GDNBatchSplit":
        """Classify the batch rows. CPU only: no device work."""
        query_start_loc_cpu = m.query_start_loc_cpu
        spec_sequence_masks_cpu: torch.Tensor | None = None
        num_spec_decodes = 0
        if self.use_spec_decode and num_decode_draft_tokens_cpu is not None:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if (
                num_spec_decodes == 0
                or num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item()
                == 0
            ):
                num_spec_decodes = 0
                spec_sequence_masks_cpu = None

        if spec_sequence_masks_cpu is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            return GDNBatchSplit(
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
                num_prefills=num_prefills,
                num_prefill_tokens=num_prefill_tokens,
                num_spec_decodes=0,
                num_spec_decode_tokens=0,
            )

        non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        non_spec_query_lens_cpu = query_lens_cpu[non_spec_sequence_masks_cpu]
        num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
        # Exclude zero-length padded sequences from prefill count.
        num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
        num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
        num_decode_tokens = num_decodes
        num_prefill_tokens = non_spec_query_lens_cpu.sum().item() - num_decode_tokens
        num_spec_decode_tokens = (
            query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
        )

        # num_decodes and num_spec_decodes are mutually exclusive.
        # Reclassify non-spec decodes as prefills when spec decodes
        # exist — the prefill kernel handles 1-token sequences with
        # initial state correctly, producing identical results.
        if num_decodes > 0 and num_spec_decodes > 0:
            num_prefills += num_decodes
            num_prefill_tokens += num_decode_tokens
            num_decodes = 0
            num_decode_tokens = 0

        return GDNBatchSplit(
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            spec_sequence_masks_cpu=spec_sequence_masks_cpu,
            non_spec_sequence_masks_cpu=non_spec_sequence_masks_cpu,
            query_lens_cpu=query_lens_cpu,
        )

    def _uses_spec_decode_buffers(self, split: "GDNBatchSplit") -> bool:
        """Whether build() writes a spec-decode batch into the persistent
        FULL cudagraph buffers.
        """
        return (
            self.use_full_cuda_graph
            and split.num_prefills == 0
            and split.num_decodes == 0
            and split.num_spec_decodes <= self.decode_cudagraph_max_bs
            and split.num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _uses_decode_buffers(self, split: "GDNBatchSplit") -> bool:
        """Whether build() writes a non-spec decode batch into the persistent
        FULL cudagraph buffers.
        """
        return (
            self.use_full_cuda_graph
            and split.num_prefills == 0
            and split.num_spec_decodes == 0
            and split.num_decodes <= self.decode_cudagraph_max_bs
        )

    def _spec_token_size(
        self, split: "GDNBatchSplit", m: CommonAttentionMetadata
    ) -> int:
        """Length of spec_token_indx for a batch without prefills or decodes."""
        return min(
            split.num_spec_decodes * (self.num_spec + 1),
            m.query_start_loc_cpu[-1].item(),
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        fused_decode: "GDNFusedDecodeStep | None" = None,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        if fused_decode is None:
            split = self._split_batch(m, num_decode_draft_tokens_cpu)
        else:
            split = fused_decode.split_batch(self, m, num_decode_draft_tokens_cpu)
            if fused_decode.write_decode_buffers(self, split, m):
                return self._decode_buffers_metadata(split, m)
            num_accepted_tokens = fused_decode.num_accepted_tokens()

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        spec_token_start: int | None = None
        non_spec_token_start: int | None = None
        num_decodes = split.num_decodes
        num_decode_tokens = split.num_decode_tokens
        num_prefills = split.num_prefills
        num_prefill_tokens = split.num_prefill_tokens
        num_spec_decodes = split.num_spec_decodes
        num_spec_decode_tokens = split.num_spec_decode_tokens
        spec_sequence_masks_cpu = split.spec_sequence_masks_cpu
        if spec_sequence_masks_cpu is None:
            spec_sequence_masks = None
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            spec_sequence_masks = async_tensor_h2d(
                spec_sequence_masks_cpu, device=query_start_loc.device
            )
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            non_spec_sequence_masks_cpu = split.non_spec_sequence_masks_cpu
            query_lens_cpu = split.query_lens_cpu
            assert non_spec_sequence_masks_cpu is not None
            assert query_lens_cpu is not None

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = self._spec_token_size(split, m)
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                # Spec and non-spec tokens are two contiguous blocks iff the
                # spec mask of the non-empty requests flips exactly once. The
                # V2 runner orders draft carriers first, so this normally
                # holds. CPU-only; otherwise the forward keeps the index path.
                active_spec_mask = spec_sequence_masks_cpu[query_lens_cpu > 0]
                num_flips = (active_spec_mask[1:] != active_spec_mask[:-1]).sum()
                if num_flips.item() == 1:
                    spec_first = bool(active_spec_mask[0].item())
                    spec_token_start = 0 if spec_first else num_non_spec_tokens
                    non_spec_token_start = num_spec_decode_tokens if spec_first else 0

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    non_spec_sequence_masks_cpu, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
            # In a mixed non-spec batch, decodes are peeled off to the recurrent
            # kernel (decode-first front slice), so build chunk metadata from the
            # rebased prefill-only cu_seqlens; otherwise use the full non-spec one.
            # _forward_core keys off the same condition, so they agree.
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                prefill_query_start_loc_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            chunk_indices, chunk_offsets = self._build_chunk_metadata(
                prefill_query_start_loc,
                prefill_query_start_loc_cpu,
                query_start_loc.device,
            )

        if num_prefills > 0:
            context_lens_tensor = m.compute_num_computed_tokens()
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        prefill_state_indices_i64: torch.Tensor | None = None
        prefill_no_initial_state: torch.Tensor | None = None
        prefill_query_start_loc_i64: torch.Tensor | None = None
        if num_prefills > 0:
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            assert prefill_query_start_loc is not None
            prefill_state_indices_i64 = prefill_state_indices.to(torch.int64)
            prefill_no_initial_state = ~prefill_has_initial_state
            if self.gdn_prefill_backend == "flashinfer":
                prefill_query_start_loc_i64 = prefill_query_start_loc.to(torch.int64)

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if self._uses_spec_decode_buffers(split):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if self._uses_decode_buffers(split):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            prefill_state_indices_i64=prefill_state_indices_i64,
            prefill_no_initial_state=prefill_no_initial_state,
            prefill_query_start_loc_i64=prefill_query_start_loc_i64,
            spec_token_start=spec_token_start,
            non_spec_token_start=non_spec_token_start,
        )
        return attn_metadata

    def _decode_buffers_metadata(
        self, split: "GDNBatchSplit", m: CommonAttentionMetadata
    ) -> GDNAttentionMetadata:
        """Metadata of a decode-only batch whose persistent FULL cudagraph
        buffers were already written by GDNFusedDecodeStep. Matches what
        build() returns for the same batch.
        """
        batch_size = m.num_reqs
        metadata = GDNAttentionMetadata(
            num_prefills=split.num_prefills,
            num_prefill_tokens=split.num_prefill_tokens,
            num_decodes=split.num_decodes,
            num_decode_tokens=split.num_decode_tokens,
            num_spec_decodes=split.num_spec_decodes,
            num_spec_decode_tokens=split.num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
        )
        if split.spec_sequence_masks_cpu is None:
            metadata.non_spec_query_start_loc = self.non_spec_query_start_loc[
                : batch_size + 1
            ]
            metadata.non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            return metadata
        metadata.spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
        metadata.spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
        metadata.spec_sequence_masks = self.spec_sequence_masks[:batch_size]
        metadata.spec_token_indx = self.spec_token_indx[
            : self._spec_token_size(split, m)
        ]
        metadata.non_spec_token_indx = self.non_spec_token_indx[:0]
        metadata.num_accepted_tokens = self.num_accepted_tokens[:batch_size]
        return metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = torch.diff(m.query_start_loc_cpu).sub_(1)
        assert num_decode_draft_tokens_cpu.shape == num_accepted_tokens.shape

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)


class GDNDecodeMetadataFusion:
    """Writes the decode-only persistent metadata of several GDN builders at once.

    A hybrid model splits its GDN layers into several KV-cache groups, each with
    its own builder, persistent buffers and block table. When a batch takes one
    of build()'s FULL cudagraph decode paths, the buffers of these builders differ
    only in the state indices gathered from each group's block table. One kernel
    launch then writes the buffers of every builder, instead of about 19 small
    ops and H2D copies per builder, and build() only slices them.
    """

    def __init__(
        self,
        builders: Sequence[tuple[int, GDNAttentionMetadataBuilder]],
        device: torch.device,
    ):
        first = builders[0][1]
        self.device = device
        self.num_builders = len(builders)
        self.kv_cache_group_ids = [group_id for group_id, _ in builders]
        # Holding the builders keeps their persistent buffers, whose addresses
        # out_ptrs bakes in, alive, and their ids unique.
        self._builders = [builder for _, builder in builders]
        self._builder_ids = {id(builder) for builder in self._builders}
        self.num_state_cols = first.num_spec + 1
        self.align = first.vllm_config.cache_config.mamba_cache_mode == "align"
        self.mamba_block_size = first.kv_cache_spec.block_size
        # [num_builders, 7] addresses of the persistent buffers, in the order
        # _gdn_decode_metadata_kernel reads them.
        self.out_ptrs = torch.tensor(
            [
                [
                    t.data_ptr()
                    for t in (
                        builder.spec_state_indices_tensor,
                        builder.spec_sequence_masks,
                        builder.spec_token_indx,
                        builder.spec_query_start_loc,
                        builder.num_accepted_tokens,
                        builder.non_spec_state_indices_tensor,
                        builder.non_spec_query_start_loc,
                    )
                ]
                for _, builder in builders
            ],
            dtype=torch.uint64,
            device=device,
        )
        self._block_table_layouts: dict[
            tuple[int, ...], tuple[torch.Tensor, torch.Tensor] | None
        ] = {}

    @classmethod
    def create(
        cls,
        builders: Sequence[tuple[int, AttentionMetadataBuilder]],
        device: torch.device,
    ) -> "GDNDecodeMetadataFusion | None":
        """Fusion over (kv_cache_group_id, builder) pairs, or None when some
        builder's decode metadata is not what the fused kernel writes.
        """
        gdn_builders: list[tuple[int, GDNAttentionMetadataBuilder]] = []
        for group_id, builder in builders:
            # Subclasses may override build(); only the plain builder is covered.
            if type(builder) is not GDNAttentionMetadataBuilder:
                return None
            gdn_builders.append((group_id, builder))
        if not gdn_builders:
            return None

        def config(builder: GDNAttentionMetadataBuilder) -> tuple:
            return (
                builder.use_full_cuda_graph,
                builder.use_spec_decode,
                builder.num_spec,
                builder.decode_cudagraph_max_bs,
                builder.vllm_config.cache_config.mamba_cache_mode,
                builder.kv_cache_spec.block_size,
                builder.kv_cache_spec.num_speculative_blocks,
            )

        first = gdn_builders[0][1]
        if any(config(builder) != config(first) for _, builder in gdn_builders):
            return None
        # Without FULL cudagraphs build() never writes the persistent buffers.
        if not first.use_full_cuda_graph:
            return None
        mamba_cache_mode = first.vllm_config.cache_config.mamba_cache_mode
        if mamba_cache_mode not in ("align", "all", "none"):
            return None
        # In align mode build() takes the first num_spec + 1 of the
        # 1 + num_speculative_blocks gathered columns.
        if (
            mamba_cache_mode == "align"
            and first.kv_cache_spec.num_speculative_blocks < first.num_spec
        ):
            return None
        return cls(gdn_builders, device)

    def fuses(self, builder: AttentionMetadataBuilder) -> bool:
        return id(builder) in self._builder_ids

    def block_table_layout(
        self, block_tables: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Device (addresses, row strides) of the builders' block tables, or
        None when the kernel cannot read them as build() does.
        """
        tables = [block_tables[group_id] for group_id in self.kv_cache_group_ids]
        key = tuple(v for t in tables for v in (t.data_ptr(), t.stride(0), t.shape[1]))
        if key in self._block_table_layouts:
            return self._block_table_layouts[key]
        if torch.cuda.is_current_stream_capturing():
            # Creating the layout copies to the device; not while capturing.
            return None
        layout: tuple[torch.Tensor, torch.Tensor] | None = None
        if all(
            t.stride(1) == 1 and (self.align or t.shape[1] >= self.num_state_cols)
            for t in tables
        ):
            layout = (
                torch.tensor(
                    [t.data_ptr() for t in tables],
                    dtype=torch.uint64,
                    device=self.device,
                ),
                torch.tensor(
                    [t.stride(0) for t in tables],
                    dtype=torch.int64,
                    device=self.device,
                ),
            )
        self._block_table_layouts[key] = layout
        return layout

    def begin_step(
        self,
        block_tables: Sequence[torch.Tensor],
        idx_mapping: torch.Tensor,
        num_accepted_tokens_by_req: torch.Tensor,
        num_reqs: int,
        gather_num_accepted_tokens: Callable[[], torch.Tensor] | None,
    ) -> "GDNFusedDecodeStep":
        return GDNFusedDecodeStep(
            self,
            block_tables,
            idx_mapping,
            num_accepted_tokens_by_req,
            num_reqs,
            gather_num_accepted_tokens,
        )


class GDNFusedDecodeStep:
    """One build_attn_metadata call's state, shared by the fused GDN builders.

    The first builder to build classifies the batch and, if the batch takes one
    of build()'s FULL cudagraph decode paths, launches the kernel that writes the
    persistent buffers of every fused builder. The other builders reuse both.
    Other batches fall back to build()'s own path.
    """

    def __init__(
        self,
        fusion: GDNDecodeMetadataFusion,
        block_tables: Sequence[torch.Tensor],
        idx_mapping: torch.Tensor,
        num_accepted_tokens_by_req: torch.Tensor,
        num_reqs: int,
        gather_num_accepted_tokens: Callable[[], torch.Tensor] | None,
    ):
        self._fusion = fusion
        self._block_tables = block_tables
        # [num_reqs] batch row -> request state index.
        self._idx_mapping = idx_mapping
        # [max_num_reqs] accepted tokens per request state index.
        self._num_accepted_tokens_by_req = num_accepted_tokens_by_req
        # Real (unpadded) requests in the batch.
        self._num_reqs = num_reqs
        self._gather_num_accepted_tokens = gather_num_accepted_tokens
        self._num_accepted_tokens: torch.Tensor | None = None
        self._num_accepted_tokens_gathered = False
        self._split: GDNBatchSplit | None = None
        self._split_key: tuple[int, int, int] | None = None
        self._decode_buffers_written: bool | None = None

    def fuses(self, builder: AttentionMetadataBuilder) -> bool:
        return self._fusion.fuses(builder)

    def split_batch(
        self,
        builder: GDNAttentionMetadataBuilder,
        m: CommonAttentionMetadata,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> GDNBatchSplit:
        key = (m.num_reqs, m.num_actual_tokens, m.max_query_len)
        if self._split is None:
            self._split = builder._split_batch(m, num_decode_draft_tokens_cpu)
            self._split_key = key
        else:
            # All groups of one build_attn_metadata call share the CPU inputs.
            assert key == self._split_key
        return self._split

    def num_accepted_tokens(self) -> torch.Tensor | None:
        """[batch_size] accepted tokens per row, for build()'s own path."""
        if not self._num_accepted_tokens_gathered:
            if self._gather_num_accepted_tokens is not None:
                self._num_accepted_tokens = self._gather_num_accepted_tokens()
            self._num_accepted_tokens_gathered = True
        return self._num_accepted_tokens

    def write_decode_buffers(
        self,
        builder: GDNAttentionMetadataBuilder,
        split: GDNBatchSplit,
        m: CommonAttentionMetadata,
    ) -> bool:
        """Write every fused builder's persistent buffers for a decode-only
        batch, once per step. False if build()'s own path must run instead.
        """
        if self._decode_buffers_written is None:
            self._decode_buffers_written = self._write_decode_buffers(builder, split, m)
        return self._decode_buffers_written

    def _write_decode_buffers(
        self,
        builder: GDNAttentionMetadataBuilder,
        split: GDNBatchSplit,
        m: CommonAttentionMetadata,
    ) -> bool:
        fusion = self._fusion
        batch_size = m.num_reqs
        if batch_size > builder.decode_cudagraph_max_bs:
            return False
        spec_sequence_masks_cpu = split.spec_sequence_masks_cpu
        if spec_sequence_masks_cpu is not None:
            if not builder._uses_spec_decode_buffers(split):
                return False
            num_spec_decodes = split.num_spec_decodes
            # The kernel takes the spec rows to be the leading real requests,
            # which holds when every other row is zero-length padding.
            if num_spec_decodes > self._num_reqs or not bool(
                spec_sequence_masks_cpu[:num_spec_decodes].all()
            ):
                return False
            num_spec_tokens = builder._spec_token_size(split, m)
        else:
            if not builder._uses_decode_buffers(split):
                return False
            # build() copies all batch_size rows into num_decodes slots.
            if split.num_decodes != batch_size:
                return False
            num_spec_decodes = 0
            num_spec_tokens = 0
        layout = fusion.block_table_layout(self._block_tables)
        if layout is None:
            return False
        block_table_ptrs, block_table_strides = layout
        _gdn_decode_metadata_kernel[(fusion.num_builders,)](
            fusion.out_ptrs,
            block_table_ptrs,
            block_table_strides,
            m.seq_lens,
            m.query_start_loc,
            self._idx_mapping,
            self._num_accepted_tokens_by_req,
            batch_size,
            num_spec_decodes,
            num_spec_tokens,
            SPEC=spec_sequence_masks_cpu is not None,
            NUM_STATE_COLS=fusion.num_state_cols,
            ALIGN=fusion.align,
            MAMBA_BLOCK_SIZE=fusion.mamba_block_size,
            NULL_ID=NULL_BLOCK_ID,
            BLOCK_SIZE=1024,
        )
        return True


@triton.jit
def _load_out_ptr(out_ptrs, index, elem_dtype):
    ptr = tl.load(out_ptrs + index)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)


@triton.jit
def _state_block_start(
    seq_lens_ptr,
    row,
    mask,
    ALIGN: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
):
    # mamba_get_block_table_tensor: align mode starts at the block holding the
    # last computed token, max((seq_len - 1) // block_size, 0); others at 0.
    if ALIGN:
        seq_len = tl.load(seq_lens_ptr + row, mask=mask, other=0)
        start = tl.where(seq_len > 0, (seq_len - 1) // MAMBA_BLOCK_SIZE, 0)
    else:
        start = tl.zeros_like(row)
    return start


@triton.jit(do_not_specialize=["batch_size", "num_spec_decodes", "num_spec_tokens"])
def _gdn_decode_metadata_kernel(
    out_ptrs,  # [num_builders, 7] persistent buffer addresses
    block_table_ptrs,  # [num_builders]
    block_table_strides,  # [num_builders]
    seq_lens_ptr,  # [batch_size]
    query_start_loc_ptr,  # [batch_size + 1]
    idx_mapping_ptr,  # [num_spec_decodes]
    num_accepted_tokens_by_req_ptr,  # [max_num_reqs]
    batch_size,
    num_spec_decodes,
    num_spec_tokens,
    SPEC: tl.constexpr,
    NUM_STATE_COLS: tl.constexpr,
    ALIGN: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    NULL_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Writes what GDNAttentionMetadataBuilder.build() copies and fills into the
    # persistent buffers of one builder per program, for a batch whose first
    # num_spec_decodes rows are spec decodes (SPEC) or whose rows are all
    # decodes (not SPEC).
    builder = tl.program_id(0)
    outs = out_ptrs + builder * 7
    block_table = _load_out_ptr(block_table_ptrs, builder, tl.int32)
    block_table_stride = tl.load(block_table_strides + builder)
    offs = tl.arange(0, BLOCK_SIZE)
    if SPEC:
        # spec_state_indices_tensor[:batch_size]: gathered state blocks of the
        # spec rows, NULL_ID for the padded rows.
        state_indices = _load_out_ptr(outs, 0, tl.int32)
        num_states = batch_size * NUM_STATE_COLS
        for i in range(0, num_states, BLOCK_SIZE):
            idx = i + offs
            row = idx // NUM_STATE_COLS
            col = idx % NUM_STATE_COLS
            is_spec = (idx < num_states) & (row < num_spec_decodes)
            start = _state_block_start(
                seq_lens_ptr, row, is_spec, ALIGN, MAMBA_BLOCK_SIZE
            )
            block_ids = tl.load(
                block_table + row * block_table_stride + start + col,
                mask=is_spec,
                other=NULL_ID,
            )
            tl.store(state_indices + idx, block_ids, mask=idx < num_states)

        # spec_sequence_masks[:batch_size] and num_accepted_tokens[:batch_size].
        masks = _load_out_ptr(outs, 1, tl.int8)
        num_accepted = _load_out_ptr(outs, 4, tl.int32)
        for i in range(0, batch_size, BLOCK_SIZE):
            row = i + offs
            valid = row < batch_size
            is_spec = row < num_spec_decodes
            tl.store(masks + row, is_spec.to(tl.int8), mask=valid)
            req_idx = tl.load(idx_mapping_ptr + row, mask=is_spec, other=0)
            accepted = tl.load(
                num_accepted_tokens_by_req_ptr + req_idx, mask=is_spec, other=1
            )
            tl.store(num_accepted + row, accepted, mask=valid)

        # spec_token_indx[:num_spec_tokens] = arange(num_spec_tokens).
        token_indx = _load_out_ptr(outs, 2, tl.int32)
        for i in range(0, num_spec_tokens, BLOCK_SIZE):
            idx = i + offs
            tl.store(token_indx + idx, idx, mask=idx < num_spec_tokens)

        # spec_query_start_loc[:batch_size + 1]: query_start_loc up to the last
        # spec row, then its end repeated.
        spec_query_start_loc = _load_out_ptr(outs, 3, tl.int32)
        for i in range(0, batch_size + 1, BLOCK_SIZE):
            row = i + offs
            valid = row <= batch_size
            loc = tl.load(
                query_start_loc_ptr + tl.minimum(row, num_spec_decodes), mask=valid
            )
            tl.store(spec_query_start_loc + row, loc, mask=valid)
    else:
        # non_spec_state_indices_tensor[:batch_size]: first state block per row.
        state_indices = _load_out_ptr(outs, 5, tl.int32)
        for i in range(0, batch_size, BLOCK_SIZE):
            row = i + offs
            valid = row < batch_size
            start = _state_block_start(
                seq_lens_ptr, row, valid, ALIGN, MAMBA_BLOCK_SIZE
            )
            block_ids = tl.load(
                block_table + row * block_table_stride + start, mask=valid
            )
            tl.store(state_indices + row, block_ids, mask=valid)

        # non_spec_query_start_loc[:batch_size + 1] = query_start_loc.
        non_spec_query_start_loc = _load_out_ptr(outs, 6, tl.int32)
        for i in range(0, batch_size + 1, BLOCK_SIZE):
            row = i + offs
            valid = row <= batch_size
            loc = tl.load(query_start_loc_ptr + row, mask=valid)
            tl.store(non_spec_query_start_loc + row, loc, mask=valid)
