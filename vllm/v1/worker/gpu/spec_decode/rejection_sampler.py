# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import SpeculativeConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.fused_rejection import fused_rejection_sample
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

# Cap on the FP32 target-logits buffer materialized by apply_sampling_params.
# TODO(mgoin): Chunking is a workaround. The rejection kernels already upcast
# per vocab block on load and apply ops like temperature and gumbel, so folding
# sampling-param application into those kernels would remove this buffer and
# its traffic entirely.
MAX_CHUNK_BYTES = 2**30  # 1GB
_FP32_BYTES = 4


def get_max_chunk_logits(vocab_size: int) -> int:
    """Largest number of logits rows one verification chunk may hold."""
    return max(1, MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES))


def _iter_request_chunks(
    cu_num_logits: np.ndarray, max_chunk_logits: int
) -> Iterator[tuple[int, int]]:
    """Yield maximally packed request ranges without splitting requests."""
    assert max_chunk_logits > 0
    num_reqs = cu_num_logits.size - 1
    start = 0
    while start < num_reqs:
        max_logit = int(cu_num_logits[start]) + max_chunk_logits
        end = int(np.searchsorted(cu_num_logits, max_logit, side="right") - 1)
        end = min(num_reqs, max(start + 1, end))
        yield start, end
        start = end


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


@triton.jit
def _gather_draft_and_pos_kernel(
    input_ids_ptr,
    positions_ptr,
    logits_indices_ptr,
    draft_sampled_ptr,
    pos_ptr,
    num_logits,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_logits
    idx = tl.load(logits_indices_ptr + offs, mask=mask, other=0)
    tl.store(draft_sampled_ptr + offs, tl.load(input_ids_ptr + idx), mask=mask)
    tl.store(pos_ptr + offs, tl.load(positions_ptr + idx), mask=mask)


def _gather_draft_and_pos(
    input_ids: torch.Tensor, positions: torch.Tensor, logits_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """input_ids[logits_indices], positions[logits_indices] in one launch."""
    num_logits = logits_indices.shape[0]
    draft_sampled = input_ids.new_empty(num_logits)
    pos = positions.new_empty(num_logits)
    if num_logits > 0:
        block_size = 1024
        _gather_draft_and_pos_kernel[(triton.cdiv(num_logits, block_size),)](
            input_ids,
            positions,
            logits_indices,
            draft_sampled,
            pos,
            num_logits,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
    return draft_sampled, pos


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
        *,
        watermark_key: int | None = None,
    ):
        self.sampler = sampler
        self.watermark_key = watermark_key
        self.use_fused = envs.VLLM_FUSED_REJECTION_SAMPLER
        if watermark_key is not None:
            assert isinstance(sampler, GPUWatermarkSampler)
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.enable_adaptive_verification = spec_config.enable_adaptive_verification
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True

    def _get_logprobs_tensors(
        self,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
        max_num_logprobs: int,
    ) -> LogprobsTensors | None:
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != num_reqs
        cu_num_generated_tokens: list[int] | torch.Tensor | None = None
        if expanded_logits:
            if self.enable_adaptive_verification:
                # Adaptive verification keeps the true per-request boundaries
                # on device only; cu_num_logits_np holds the pre-compacted
                # layout.
                cu_num_generated_tokens = cu_num_logits.clone()
            else:
                cu_num_generated_tokens = cu_num_logits_np.tolist()
        return compute_topk_scores(
            logits,
            max_num_logprobs,
            flat_sampled,
            cu_num_generated_tokens,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def _watermarking_kwargs(
        self,
        draft_sampled: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> dict[str, Any]:
        if self.watermark_key is None:
            return {}
        assert isinstance(self.sampler, GPUWatermarkSampler)
        return {
            "contexts": self.sampler._get_contexts(
                expanded_idx_mapping, expanded_local_pos, draft_sampled
            ),
            "watermarking": self.sampler.watermarking.gpu,
            "watermark_key": self.watermark_key,
        }

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        num_rejected: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
        )
        fused_kwargs: dict[str, Any] = {}
        if num_rejected is not None:
            fused_kwargs = dict(
                num_rejected=num_rejected,
                seq_lens=seq_lens,
                prefill_len=self.sampler.req_states.prefill_len.gpu,
            )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
            **fused_kwargs,
            **self._watermarking_kwargs(
                draft_sampled, expanded_idx_mapping, expanded_local_pos
            ),
        )
        return processed_logits, sampled, num_sampled

    def _verify_in_chunks(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
        max_num_logprobs: int,
        num_rejected: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:
        cu_num_logits_np = input_batch.cu_num_logits_np
        use_processed_logits = self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        num_reqs = input_batch.num_reqs

        if logits.shape[0] <= max_chunk_logits:
            # One chunk covers the batch. Adaptive verification compacts the logits
            # without updating cu_num_logits_np (it keeps the pre-compacted layout),
            # so the stale sums must not pick chunk boundaries; its budget cap
            # guarantees the compacted batch always lands here.
            request_chunks: Iterable[tuple[int, int]] = ((0, num_reqs),)
        else:
            assert not self.enable_adaptive_verification
            assert num_rejected is None
            request_chunks = _iter_request_chunks(cu_num_logits_np, max_chunk_logits)

        sampled_chunks: list[torch.Tensor] = []
        num_sampled_chunks: list[torch.Tensor] = []
        logprobs_chunks: list[LogprobsTensors] = []

        for start, end in request_chunks:
            lo = int(cu_num_logits_np[start])
            hi = int(cu_num_logits_np[end])
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            if lo == 0:
                # Offsetting by zero is a no-op; skip the launch.
                chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1]
            else:
                chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1] - lo
            fused_kwargs: dict[str, Any] = {}
            if num_rejected is not None:
                fused_kwargs = dict(
                    num_rejected=num_rejected, seq_lens=input_batch.seq_lens
                )
            # draft_logits uses persistent request-state indices and stays global.
            processed_logits, sampled, num_sampled = self._verify(
                logits[lo:hi],
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                chunk_cu_num_logits,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.expanded_local_pos[lo:hi],
                **fused_kwargs,
            )
            chunk_logprobs = self._get_logprobs_tensors(
                sampled,
                num_sampled,
                processed_logits if use_processed_logits else logits[lo:hi],
                chunk_cu_num_logits,
                chunk_cu_num_logits_np,
                max_num_logprobs,
            )
            if chunk_logprobs is not None:
                logprobs_chunks.append(chunk_logprobs)
            del processed_logits
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)

        if len(sampled_chunks) == 1:
            logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
            return sampled_chunks[0], num_sampled_chunks[0], logprobs_tensors

        logprobs_tensors = None
        if logprobs_chunks:
            expanded_logits = logits.shape[0] != input_batch.num_reqs
            logprobs_tensors = LogprobsTensors.cat(
                logprobs_chunks,
                cu_num_generated_tokens=(
                    cu_num_logits_np.tolist() if expanded_logits else None
                ),
            )

        sampled = torch.cat(sampled_chunks)
        num_sampled = torch.cat(num_sampled_chunks)
        return sampled, num_sampled, logprobs_tensors

    def _fused_spec_params(
        self,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        max_num_logprobs: int,
    ) -> tuple[int, bool, bool] | None:
        """Params of fused_rejection_sample when it covers this batch: one-hot
        drafts, standard verification, no logprobs, and only penalties,
        temperature and bounded top-k/top-p as logits processing.
        """
        if (
            not self.use_fused
            or draft_logits is not None
            or self.synthetic_conditional_rates is not None
            or self.use_block_verification
            or self.watermark_key is not None
            or self.enable_adaptive_verification
            or max_num_logprobs != NO_LOGPROBS
        ):
            return None
        return self.sampler.fused_spec_sampling_params(input_batch.idx_mapping_np)

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        fused = self._fused_spec_params(input_batch, draft_logits, max_num_logprobs)
        if fused is not None:
            max_top_k, use_top_p, use_penalties = fused
            states = self.sampler.sampling_states
            penalties = self.sampler.penalties_state
            fused_sampled, fused_num_sampled, fused_num_rejected = (
                fused_rejection_sample(
                    logits,
                    input_batch.input_ids,
                    input_batch.positions,
                    input_batch.logits_indices,
                    input_batch.cu_num_logits,
                    input_batch.idx_mapping,
                    input_batch.seq_lens,
                    input_batch.expanded_idx_mapping,
                    input_batch.expanded_local_pos,
                    self.sampler.req_states.prefill_len.gpu,
                    states.temperature.gpu,
                    states.seeds.gpu,
                    states.top_k.gpu,
                    states.top_p.gpu,
                    (
                        penalties.repetition_penalty.gpu,
                        penalties.frequency_penalty.gpu,
                        penalties.presence_penalty.gpu,
                        penalties.prompt_bin_mask,
                        penalties.output_bin_counts,
                    )
                    if use_penalties
                    else None,
                    max_top_k,
                    use_top_p,
                    self.num_speculative_steps,
                    use_fp64=self.sampler.use_fp64_gumbel,
                )
            )
            return SamplerOutput(
                sampled_token_ids=fused_sampled,
                logprobs_tensors=None,
                num_nans=num_nans,
                num_sampled=fused_num_sampled,
                num_rejected=fused_num_rejected,
            )

        draft_sampled, pos = _gather_draft_and_pos(
            input_batch.input_ids, input_batch.positions, input_batch.logits_indices
        )

        chunk_logit_limit = get_max_chunk_logits(logits.shape[1])
        # Fold get_num_sampled_and_rejected into the last rejection kernel when
        # one chunk covers the batch and no logprobs read the raw num_sampled.
        num_rejected: torch.Tensor | None = None
        if max_num_logprobs == NO_LOGPROBS and logits.shape[0] <= chunk_logit_limit:
            num_rejected = torch.empty(
                input_batch.num_reqs, dtype=torch.int32, device=logits.device
            )
        sampled, num_sampled, logprobs_tensors = self._verify_in_chunks(
            logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            chunk_logit_limit,
            max_num_logprobs,
            num_rejected=num_rejected,
        )

        if num_rejected is None:
            num_sampled, num_rejected = get_num_sampled_and_rejected(
                num_sampled,
                input_batch.seq_lens,
                input_batch.cu_num_logits,
                input_batch.idx_mapping,
                self.sampler.req_states.prefill_len.gpu,
            )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
