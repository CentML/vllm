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
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.sparse_verify import sparse_rejection_sample
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.fused_rejection import fused_rejection_sample
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

# Bound the dense fp32 canvas used by preparation/verification fallbacks.
# Compact rejection avoids it; sparse verification only initializes live
# chunks, but still reserves its address range within the same memory limit.
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
def _gather_draft_sampled_kernel(
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits]
    pos_ptr,
    # [num_tokens]
    input_ids_ptr,
    # [num_tokens]
    positions_ptr,
    # [num_logits]
    logits_indices_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    prefill_len_ptr,
    num_logits,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < num_logits
    token_idx = tl.load(logits_indices_ptr + block, mask=mask)
    token_id = tl.load(input_ids_ptr + token_idx, mask=mask)
    pos = tl.load(positions_ptr + token_idx, mask=mask)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + block, mask=mask)
    local_pos = tl.load(expanded_local_pos_ptr + block, mask=mask)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx, mask=mask)
    # Drafts are only proposed after sampling past the prefill, so any draft
    # slots in a step starting within the prefill are placeholders (e.g. the
    # padded first step after a P/D remote KV load).
    is_placeholder = (local_pos > 0) & (pos - local_pos < prefill_len)
    token_id = tl.where(is_placeholder, -1, token_id)
    tl.store(draft_sampled_ptr + block, token_id, mask=mask)
    tl.store(pos_ptr + block, pos, mask=mask)


def gather_draft_sampled(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits_indices: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the input token and position of each logits row.

    Draft rows of requests that have not yet sampled past their prefill are
    set to -1 so that the rejection kernels reject them.
    """
    num_logits = logits_indices.shape[0]
    draft_sampled = input_ids.new_empty(num_logits)
    pos = positions.new_empty(num_logits)
    BLOCK_SIZE = 1024
    _gather_draft_sampled_kernel[(triton.cdiv(num_logits, BLOCK_SIZE),)](
        draft_sampled,
        pos,
        input_ids,
        positions,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        prefill_len,
        num_logits,
        BLOCK_SIZE=BLOCK_SIZE,
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
        self.use_sparse = envs.VLLM_SPARSE_VERIFY_SAMPLING
        if watermark_key is not None:
            assert isinstance(sampler, GPUWatermarkSampler)
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.enable_adaptive_verification = spec_config.enable_adaptive_verification
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.fly_window_size = 0
        self.fly_entropy_threshold = spec_config.fly_entropy_threshold
        self.fly_entropy_top_k = spec_config.fly_entropy_top_k
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
        elif rejection_sample_method == "fly":
            assert spec_config.fly_window_size is not None
            self.fly_window_size = spec_config.fly_window_size

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
                # on device only; cu_num_logits_np holds the pre-compacted layout.
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
        contexts = self.sampler._get_contexts(
            expanded_idx_mapping, expanded_local_pos, draft_sampled
        )
        watermarking_skip_mask = None
        if self.sampler.deduplicate_contexts != "none":
            watermarking_skip_mask = self.sampler._get_repeated_contexts(
                expanded_idx_mapping, contexts, expanded_local_pos
            )
        return {
            "contexts": contexts,
            "watermarking": self.sampler.watermarking.gpu,
            "watermarking_skip_mask": watermarking_skip_mask,
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
        seq_lens_upper_bound_np: np.ndarray,
        verify_draft_sampled: torch.Tensor,
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
            seq_lens_upper_bound_np,
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
            verify_draft_sampled,
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
            fly_window_size=self.fly_window_size,
            fly_entropy_threshold=self.fly_entropy_threshold,
            fly_entropy_top_k=self.fly_entropy_top_k,
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
        verify_draft_sampled: torch.Tensor | None = None,
        num_rejected: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        LogprobsTensors | None,
        SamplingMaskTensors | None,
    ]:
        if verify_draft_sampled is None:
            verify_draft_sampled = draft_sampled
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
        sampling_mask_chunks: list[SamplingMaskTensors] = []
        max_num_kept = None
        if self.sampler.return_sampling_mask:
            max_num_kept = int(
                np.max(
                    self.sampler.sampling_states.top_k.np[input_batch.idx_mapping_np]
                )
            )

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
                input_batch.seq_lens_cpu_upper_bound.numpy()[start:end],
                verify_draft_sampled[lo:hi],
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
            if max_num_kept is not None:
                sampling_mask_chunks.append(
                    SamplingMaskTensors.from_logits(
                        processed_logits,
                        chunk_cu_num_logits,
                        num_sampled,
                        max_num_kept,
                        self.num_speculative_steps + 1,
                    )
                )
            del processed_logits
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)

        if len(sampled_chunks) == 1:
            logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
            sampling_mask_tensors = (
                sampling_mask_chunks[0] if sampling_mask_chunks else None
            )
            return (
                sampled_chunks[0],
                num_sampled_chunks[0],
                logprobs_tensors,
                sampling_mask_tensors,
            )

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
        sampling_mask_tensors = (
            SamplingMaskTensors.cat(sampling_mask_chunks)
            if sampling_mask_chunks
            else None
        )
        return sampled, num_sampled, logprobs_tensors, sampling_mask_tensors

    def _compact_paths_excluded(
        self, input_batch: InputBatch, draft_logits: torch.Tensor | None
    ) -> bool:
        """Conditions under which neither compact path (fused, sparse) covers
        the batch: they assume one-hot drafts, standard verification and no
        dense processed-logits consumer.
        """
        return (
            draft_logits is not None
            or self.synthetic_conditional_rates is not None
            or self.use_block_verification
            or self.fly_window_size > 0
            or self.watermark_key is not None
            or self.enable_adaptive_verification
            # SamplingMaskTensors are built from the dense processed logits.
            or self.sampler.return_sampling_mask
            or _has_placeholder_drafts(input_batch)
        )

    def _fused_spec_params(
        self,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        max_num_logprobs: int,
        invalid_drafts: torch.Tensor | None,
    ) -> tuple[int, bool, bool] | None:
        """Params of fused_rejection_sample when it covers this batch: one-hot
        drafts, standard verification, no logprobs, no grammar-invalidated
        drafts, and only penalties, temperature and bounded top-k/top-p as
        logits processing.
        """
        if (
            not self.use_fused
            or not self.sampler.sampling_states.use_split_row_top_k
            or invalid_drafts is not None
            or max_num_logprobs != NO_LOGPROBS
            or self._compact_paths_excluded(input_batch, draft_logits)
        ):
            return None
        return self.sampler.fused_spec_sampling_params(input_batch.idx_mapping_np)

    def _sparse_spec_params(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        max_chunk_logits: int,
        max_num_logprobs: int,
    ) -> tuple[int, bool, bool] | None:
        """Sparse canvas consumers must never read uninitialized dead chunks."""
        states = self.sampler.sampling_states
        if (
            not self.use_sparse
            or not states.use_split_row_top_k
            or not states.use_fused_prep
            or not logits.is_cuda
            or logits.stride(1) != 1
            or self._compact_paths_excluded(input_batch, draft_logits)
            or logits.shape[0] > max_chunk_logits
            or (
                max_num_logprobs != NO_LOGPROBS
                and self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
            )
            or np.any(states.temperature.np[input_batch.idx_mapping_np] != 1.0)
        ):
            return None
        return self.sampler.fused_spec_sampling_params(input_batch.idx_mapping_np)

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
        invalid_drafts: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        fused = self._fused_spec_params(
            input_batch, draft_logits, max_num_logprobs, invalid_drafts
        )
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

        draft_sampled, pos = gather_draft_sampled(
            input_batch.input_ids,
            input_batch.positions,
            input_batch.logits_indices,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.req_states.prefill_len.gpu,
        )

        # Marking a draft invalid (`is_valid_draft = draft_sampled >= 0`) pins the
        # accepted length before it reaches a permissive bitmask row. Only
        # verification sees the copy: `apply_sampling_params` and watermarking
        # keep the real draft ids.
        verify_draft_sampled = draft_sampled
        if invalid_drafts is not None:
            verify_draft_sampled = draft_sampled.masked_fill(invalid_drafts, -1)

        chunk_logit_limit = get_max_chunk_logits(logits.shape[1])
        sparse = self._sparse_spec_params(
            logits, input_batch, draft_logits, chunk_logit_limit, max_num_logprobs
        )
        num_rejected: torch.Tensor | None
        if sparse is not None:
            max_top_k, use_top_p, use_penalties = sparse
            states = self.sampler.sampling_states
            penalties = self.sampler.penalties_state
            sampled, num_sampled = sparse_rejection_sample(
                logits,
                input_batch.input_ids,
                input_batch.logits_indices,
                verify_draft_sampled,
                pos,
                input_batch.cu_num_logits,
                input_batch.idx_mapping,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
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
            logprobs_tensors = self._get_logprobs_tensors(
                sampled,
                num_sampled,
                logits,
                input_batch.cu_num_logits,
                input_batch.cu_num_logits_np,
                max_num_logprobs,
            )
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

        # Fold get_num_sampled_and_rejected into the last rejection kernel when
        # one chunk covers the batch and nothing reads the raw num_sampled
        # (logprobs, sampling masks).
        num_rejected = None
        if (
            max_num_logprobs == NO_LOGPROBS
            and not self.sampler.return_sampling_mask
            and logits.shape[0] <= chunk_logit_limit
        ):
            num_rejected = torch.empty(
                input_batch.num_reqs, dtype=torch.int32, device=logits.device
            )
        (
            sampled,
            num_sampled,
            logprobs_tensors,
            sampling_mask_tensors,
        ) = self._verify_in_chunks(
            logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            chunk_logit_limit,
            max_num_logprobs,
            verify_draft_sampled,
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
            sampling_mask_tensors=sampling_mask_tensors,
        )


def _has_placeholder_drafts(input_batch: InputBatch) -> bool:
    """Whether a drafted request starts this step within its prefill (e.g. the
    padded first step after a P/D remote KV load). Its draft slots are
    placeholders that gather_draft_sampled sets to -1; the compact paths read
    the raw input ids (draft counts of the penalties, fused verification), so
    such batches take the dense path.
    """
    num_drafts = input_batch.num_draft_tokens_per_req
    return (
        input_batch.has_prefill
        and num_drafts is not None
        and bool(np.any(input_batch.is_prefilling_np & (num_drafts > 0)))
    )
