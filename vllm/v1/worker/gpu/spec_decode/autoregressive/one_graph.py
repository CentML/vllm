# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in: the whole MTP draft of a uniform decode step as one CUDA graph.

Stock decode step, after the target graph and the sampler, the draft runs as
draft prefill FULL graph -> ``_prepare_decode_inputs_kernel`` ->
``_compute_slot_mappings_kernel`` -> draft-decode attention metadata build (on
the C32 config its only device work is the KF block-table copy) -> fused
draft-decode FULL graph. Every graph boundary costs about 2.5-3 us of GPU idle
and every eager kernel boundary about 0.8-1.4 us.

With ``VLLM_DRAFT_ONE_GRAPH=1`` the draft prefill body, those two kernels and the
fused draft-decode body are captured into one graph per (real request count,
prefill descriptor, decode descriptor):

* The two graph bodies are captured from the same functions, with the same
  dummy attention metadata (``prepare_inputs_to_capture``) and the same padded
  shapes as the stock prefill / decode graphs, so they record the same kernels
  with the same arguments.
* ``prepare_decode_inputs`` and ``compute_slot_mappings`` are recorded with the
  stock runtime arguments. Their only per-step input with a fresh address,
  ``num_rejected``, is read from a static copy that ``_copy_idx_mapping_kernel``
  writes in the launch it already makes before the draft (no extra launch).
* The draft-decode attention metadata build stays eager and keeps its stock
  host behaviour, but runs before the replay instead of after the draft
  prefill. That is only valid when the build's device work neither reads nor
  writes anything the draft prefill or the two kernels touch. It holds for the
  FlashInfer trtllm-gen / KF and Triton builders: the FlashInfer pure-decode
  build only copies the gathered block tables into its own buffer, and the
  Triton build has no device work. Other backends are refused.

Same kernels, arguments and order of dependent operations as the stock path,
so the drafts are bitwise identical. Graphs are captured at start-up after the
stock speculator graphs, for 1..``VLLM_DRAFT_ONE_GRAPH_MAX_REQS`` requests;
any other batch takes the stock path.

Environment:
    VLLM_DRAFT_ONE_GRAPH=1              enable (default off)
    VLLM_DRAFT_ONE_GRAPH_MAX_REQS=16    largest request count captured
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.distributed.parallel_state import graph_capture
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.utils.torch_utils import current_stream
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        AutoRegressiveSpeculator,
    )

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_DRAFT_ONE_GRAPH", "0") == "1"
MAX_REQS = int(os.environ.get("VLLM_DRAFT_ONE_GRAPH_MAX_REQS", "16"))
# Draft-decode attention backends whose metadata build may run before the draft
# prefill (see the module docstring).
_HOISTABLE_BACKENDS = ("FLASHINFER", "TRITON_ATTN")

Key = tuple[int, BatchExecutionDescriptor, BatchExecutionDescriptor]


def unsupported_reason(spec: "AutoRegressiveSpeculator") -> str | None:
    if spec.num_speculative_steps < 2:
        return "needs at least 2 speculative steps"
    if not spec.use_fused_multi_step_decode:
        return "needs the fused multi-step draft decode"
    if spec.prefill_cudagraph_manager is None or spec.decode_cudagraph_manager is None:
        return "no speculator cudagraph managers"
    if spec.pcp_manager is not None or spec.dp_size != 1:
        return "prefill context / data parallelism"
    if spec.supports_mm_inputs:
        return "multimodal drafter"
    if getattr(spec, "share_mtp_topk_indices", False):
        return "shared MTP top-k indices"
    names = sorted(
        {g.backend.get_name() for groups in spec.attn_groups for g in groups}
    )
    if any(n not in _HOISTABLE_BACKENDS for n in names):
        return f"draft attention backend(s) {names}"
    return None


class DraftOneGraphs:
    def __init__(self, spec: "AutoRegressiveSpeculator"):
        self.spec = spec
        self.graphs: dict[Key, torch.cuda.CUDAGraph] = {}
        # Static copy of the step's num_rejected (written by
        # _copy_idx_mapping_kernel before the draft).
        self.num_rejected = torch.zeros(
            spec.max_num_reqs, dtype=torch.int32, device=spec.device
        )
        self.replays = 0

    def decode_desc(self, num_reqs: int) -> BatchExecutionDescriptor:
        # The stock decode dispatch of propose() without data parallelism.
        spec = self.spec
        desc, _ = dispatch_cg_and_sync_dp(
            spec.decode_cudagraph_manager,
            num_reqs,
            num_reqs,
            uniform_token_count=1,
            dp_size=spec.dp_size,
            dp_rank=spec.dp_rank,
        )
        return desc

    def _mid(self, num_reqs: int, decode_desc: BatchExecutionDescriptor) -> None:
        """The stock eager ops between the two draft graphs (minus the hoisted
        attention metadata build), with the stock runtime arguments.
        """
        from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
            prepare_decode_inputs,
        )

        spec = self.spec
        prepare_decode_inputs(
            spec.draft_tokens[:num_reqs, 0],
            spec.target_input_buffers.seq_lens[:num_reqs],
            self.num_rejected[:num_reqs],
            spec.input_buffers,
            spec.sample_src_positions,
            spec.max_model_len,
            spec.max_num_reqs,
            advance_draft_positions=spec.advance_draft_positions,
        )
        spec.block_tables.compute_slot_mappings(
            spec.idx_mapping[:num_reqs],
            spec.input_buffers.query_start_loc[: num_reqs + 1],
            spec.input_buffers.positions[:num_reqs],
            decode_desc.num_tokens,
        )

    def _make_body(
        self,
        num_reqs: int,
        prefill_desc: BatchExecutionDescriptor,
        decode_desc: BatchExecutionDescriptor,
    ) -> Callable[[], None]:
        # Fresh dummy inputs per pass, as SpeculatorCudaGraphManager.capture.
        spec = self.spec
        assert prefill_desc.num_reqs is not None and decode_desc.num_reqs is not None
        attn_p, slot_p = prepare_inputs_to_capture(
            prefill_desc.num_reqs,
            prefill_desc.num_tokens,
            spec.model_state,
            spec.target_input_buffers,
            spec.block_tables,
            spec.target_attn_groups,
            spec.kv_cache_config,
            full_cudagraph=True,
        )
        attn_d, slot_d = prepare_inputs_to_capture(
            decode_desc.num_reqs,
            decode_desc.num_tokens,
            spec.model_state,
            spec.input_buffers,
            spec.block_tables,
            spec.attn_groups,
            spec.kv_cache_config,
            full_cudagraph=True,
        )

        def body() -> None:
            spec._prefill(
                prefill_desc.num_reqs,
                prefill_desc.num_tokens,
                attn_p,
                slot_p,
                None,
                CUDAGraphMode.NONE,
            )
            self._mid(num_reqs, decode_desc)
            spec._generate_fused_drafts(
                decode_desc.num_reqs,
                decode_desc.num_tokens,
                attn_d,
                slot_d,
                None,
                CUDAGraphMode.NONE,
            )

        return body

    @torch.inference_mode()
    def capture(self, target_tokens_padded: Callable[[int], int | None]) -> None:
        """Capture one graph per request count. ``target_tokens_padded(r)`` is
        the target's padded token count of a uniform decode batch of r
        requests, or None when the target does not run it as a FULL graph.
        """
        spec = self.spec
        assert spec.prefill_cudagraph_manager is not None
        q = spec.num_speculative_steps + 1
        pool = spec.prefill_cudagraph_manager.pool
        with graph_capture(device=spec.device):
            for r in range(1, min(MAX_REQS, spec.max_num_reqs) + 1):
                tokens = target_tokens_padded(r)
                if tokens is None:
                    continue
                prefill_desc, _ = dispatch_cg_and_sync_dp(
                    spec.prefill_cudagraph_manager,
                    r,
                    tokens,
                    q,
                    dp_size=spec.dp_size,
                    dp_rank=spec.dp_rank,
                )
                decode_desc = self.decode_desc(r)
                if (
                    prefill_desc.cg_mode != CUDAGraphMode.FULL
                    or decode_desc.cg_mode != CUDAGraphMode.FULL
                ):
                    continue
                key = (r, prefill_desc, decode_desc)
                if key in self.graphs:
                    continue
                # Warm-up pass, then capture with fresh dummy inputs.
                self._make_body(r, prefill_desc, decode_desc)()
                body = self._make_body(r, prefill_desc, decode_desc)
                graph = torch.cuda.CUDAGraph()
                get_offloader().sync_prev_onload()
                set_graph_pool_id(pool)
                with torch.cuda.graph(graph, pool, stream=current_stream()):
                    body()
                    get_offloader().join_after_forward()
                self.graphs[key] = graph
        torch.accelerator.synchronize()
        logger.info(
            "Draft one-graph: captured %d graphs (request counts %s)",
            len(self.graphs),
            sorted({k[0] for k in self.graphs}),
        )

    def lookup(
        self, num_reqs: int, prefill_desc: BatchExecutionDescriptor
    ) -> tuple[torch.cuda.CUDAGraph, BatchExecutionDescriptor] | None:
        if prefill_desc.cg_mode != CUDAGraphMode.FULL or num_reqs > MAX_REQS:
            return None
        decode_desc = self.decode_desc(num_reqs)
        graph = self.graphs.get((num_reqs, prefill_desc, decode_desc))
        return None if graph is None else (graph, decode_desc)
