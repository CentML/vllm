# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm the fused top-k rejection sampler (``fused_rejection.py``) at start-up.

``fused_rejection_sample`` replaces the verify step's logits processing and
rejection sampling for speculative batches with three Triton kernels. Their
compile keys depend on the batch's sampling parameters (``KP`` = power-of-two
bound of the largest top-k, ``TOP_P``, ``HAS_PENALTIES``) and on
``use_fp64_gumbel``, so no profile or capture run reaches them, and every
worker JIT-compiles them (seconds) on its first speculative step under load.

This runs ``fused_rejection_sample`` once per parameter combination with the
runner's own persistent state tensors and input buffers (the tensors, dtypes
and strides of the serving call); only freshly allocated outputs are written.

``VLLM_FUSED_SPEC_SAMPLER_WARMUP``:

- ``default``: the ``KP`` of the model's default top-k (generation config;
  every ``KP`` without a usable default), with and without top-p and
  penalties (clients commonly override both).
- ``all``: every ``KP`` x ``TOP_P`` x penalties combination.
- ``0``: off.
"""

import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

_MODE = os.environ.get("VLLM_FUSED_SPEC_SAMPLER_WARMUP", "default")


def _warmup_combos(runner: "GPUModelRunner", mode: str) -> list[tuple[int, bool, bool]]:
    from vllm.v1.worker.gpu.sample.spec_topk_topp import MAX_TOP_K, _top_k_pow2

    kp_to_top_k = {_top_k_pow2(k): k for k in range(1, MAX_TOP_K + 1)}
    all_kp_top_ks = sorted(kp_to_top_k.values())
    top_ks = all_kp_top_ks
    if mode != "all":
        top_k = int(runner.model_config.get_diff_sampling_param().get("top_k") or 0)
        if 1 <= top_k <= MAX_TOP_K:
            top_ks = [top_k]
    return [(k, p, pen) for k in top_ks for p in (False, True) for pen in (False, True)]


@torch.inference_mode()
def fused_spec_sampler_warmup(runner: "GPUModelRunner") -> None:
    if _MODE == "0":
        return
    rs = getattr(runner, "rejection_sampler", None)
    if (
        rs is None
        or rs.synthetic_conditional_rates is not None
        or rs.use_block_verification
        or rs.watermark_key is not None
        or rs.enable_adaptive_verification
        or not rs.use_fused
    ):
        # fused_rejection_sample never runs for this configuration.
        return
    from vllm.v1.worker.gpu.spec_decode.fused_rejection import (
        fused_rejection_sample,
    )

    sampler = rs.sampler
    states = sampler.sampling_states
    pen = sampler.penalties_state
    buffers = runner.input_buffers
    device = runner.device
    num_logits = rs.num_speculative_steps + 1
    # One request with its target + draft rows, as in a decode step. Slot 0 of
    # the persistent state is only read.
    hidden = torch.zeros(
        num_logits,
        runner.model_config.get_hidden_size(),
        dtype=runner.model_config.dtype,
        device=device,
    )
    logits = runner.model.compute_logits(hidden)
    idx_mapping = torch.zeros(1, dtype=torch.int64, device=device)
    cu_num_logits = torch.tensor([0, num_logits], dtype=torch.int32, device=device)
    logits_indices = torch.arange(num_logits, dtype=torch.int64, device=device)
    expanded_idx_mapping = idx_mapping.new_zeros(num_logits)
    expanded_local_pos = torch.arange(num_logits, dtype=torch.int32, device=device)
    penalties = (
        pen.repetition_penalty.gpu,
        pen.frequency_penalty.gpu,
        pen.presence_penalty.gpu,
        pen.prompt_bin_mask,
        pen.output_bin_counts,
    )
    combos = _warmup_combos(runner, _MODE)
    for max_top_k, use_top_p, use_penalties in combos:
        fused_rejection_sample(
            logits,
            buffers.input_ids[:num_logits],
            buffers.positions[:num_logits],
            logits_indices,
            cu_num_logits,
            idx_mapping,
            buffers.seq_lens[:1],
            expanded_idx_mapping,
            expanded_local_pos,
            sampler.req_states.prefill_len.gpu,
            states.temperature.gpu,
            states.seeds.gpu,
            states.top_k.gpu,
            states.top_p.gpu,
            penalties if use_penalties else None,
            max_top_k,
            use_top_p,
            rs.num_speculative_steps,
            use_fp64=sampler.use_fp64_gumbel,
        )
    torch.accelerator.synchronize(device)
    logger.info(
        "Warmed the fused spec-decode sampler (top_k, top_p, penalties): %s.",
        combos,
    )
