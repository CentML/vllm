# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CapturedAttentionState,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def _prepare_dflash_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    skip_attn: bool,
) -> CapturedAttentionState:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    attn_metadata = None
    if not skip_attn:
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=num_tokens // num_reqs,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_model_len,
            block_tables=input_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            for_cudagraph_capture=True,
            causal=False,
        )
    return CapturedAttentionState(attn_metadata, slot_mappings_by_layer)


class DFlashCudaGraphManager(CudaGraphManager):
    """DFlash CudaGraphManager for the parallel-drafting query forward,
    building its own non-causal attention metadata from scratch."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
    ):
        super().__init__(vllm_config, device, cudagraph_mode, decode_query_len)

        # Use a dedicated pool for DFlash to avoid memory overlap with the main
        # model's cudagraph. The base class uses a shared global pool, but
        # DFlash's internal allocations (e.g., gumbel_sample temporaries) can
        # conflict with the main model's allocations when sharing the same pool.
        if cudagraph_mode:
            self.pool = torch.cuda.graph_pool_handle()

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
        ) -> tuple[Callable[[CUDAGraphMode], None], CapturedAttentionState]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_state = _prepare_dflash_inputs_to_capture(
                num_reqs,
                num_tokens,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                max_model_len,
                skip_attn=(desc.cg_mode == CUDAGraphMode.PIECEWISE),
            )
            attn_metadata, slot_mappings = attn_state

            fwd = lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )
            return fwd, attn_state

        super().capture(create_forward_fn, progress_bar_desc)


class DFlashContextKVCudaGraphs:
    """Standalone CUDA graphs for ``precompute_and_store_context_kv``, keyed by
    padded context-token count.

    Unlike the query forward, the context-KV precompute has no attention, no
    forward context, and no DP sync: it is fixed-shape projection + RoPE + a
    scatter write into the KV cache. So it needs only fixed-address buffers and
    a bucketed token count, captured with a bare ``torch.cuda.CUDAGraph`` rather
    than the full ``CudaGraphManager`` machinery.

    The token count varies per step, so one graph is captured per capture size.
    At replay the smallest bucket that covers the actual token count is used; the
    padded tail of ``context_slot_mapping`` (PAD slots) and ``context_positions``
    (zeros) is filled by ``prepare_dflash_inputs``, while the ``hidden_states``
    tail is zeroed here so padded rows never produce NaNs.
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor,
        capture_sizes: list[int],
        max_num_tokens: int,
        device: torch.device,
    ):
        self._model = model
        self._hidden_states = hidden_states
        self._context_positions = context_positions
        self._context_slot_mapping = context_slot_mapping
        self._sizes = sorted({s for s in capture_sizes if 0 < s <= max_num_tokens})
        self._device = device
        self._pool = torch.cuda.graph_pool_handle() if self._sizes else None
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

    @property
    def enabled(self) -> bool:
        return len(self._sizes) > 0

    def _run(self, num_tokens: int) -> None:
        self._model.precompute_and_store_context_kv(
            self._hidden_states[:num_tokens],
            self._context_positions[:num_tokens],
            self._context_slot_mapping[:num_tokens],
        )

    @torch.inference_mode()
    def capture(
        self, progress_bar_desc: str = "Capturing dflash context-KV CUDA graphs"
    ) -> None:
        if not self._sizes:
            return
        # Capture from a safe state: PAD slots (so capture/warmup writes nothing
        # into the real cache) and zeroed inputs/positions (so RoPE indices stay
        # in range and no NaNs are produced).
        self._context_slot_mapping.fill_(PAD_SLOT_ID)
        self._context_positions.zero_()
        self._hidden_states.zero_()

        sizes = self._sizes
        if is_global_first_rank():
            sizes = tqdm(sizes, desc=progress_bar_desc)
        with graph_capture(device=self._device):
            for num_tokens in sizes:
                self._run(num_tokens)  # warmup
                logger.debug("Context-KV CG capture: num_tokens=%d", num_tokens)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self._pool):
                    self._run(num_tokens)
                self._graphs[num_tokens] = graph

    def maybe_run(self, num_tokens: int) -> bool:
        """Replay the smallest captured graph that covers ``num_tokens``.

        Returns False if no captured bucket fits, so the caller can fall back to
        running the precompute eagerly.
        """
        bucket = next((s for s in self._sizes if s >= num_tokens), None)
        if bucket is None:
            return False
        # The replayed graph always processes `bucket` tokens. context_positions
        # and context_slot_mapping tails are already padded by
        # prepare_dflash_inputs, but hidden_states is filled by a plain copy_
        # that leaves stale data (possibly NaN) in [num_tokens:bucket]. Zero it
        # so padded rows produce finite, discarded K/V.
        if bucket > num_tokens:
            self._hidden_states[num_tokens:bucket].zero_()
        self._graphs[bucket].replay()
        return True
