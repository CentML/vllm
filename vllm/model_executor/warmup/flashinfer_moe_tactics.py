# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer trtllm-gen MoE autotuning under data/expert parallelism.

Opt-in via ``VLLM_FLASHINFER_MOE_EP_AWARE_TUNING=1``. Measured on VR200
(GLM-5.3-Flash NVFP4, DEP2 rank shapes, 144 local experts, job 800152): for
post-dispatch rows >= 8192 the default heuristic (tactic -1) runs tile 128
while tile 256 is 1.36-1.38x faster for the whole op (bitwise-identical
output), and the autotuner finds it. But vLLM tunes buckets only up to
``max_num_batched_tokens`` while the MoE of a DP+EP rank sees up to
``dp_size * max_num_batched_tokens`` rows (all2all receive buffer or
all-gather), and FlashInfer maps larger row counts to an untuned bucket, so
the biggest prefill steps always fall back to tactic -1.

With the flag:

- ``trtllm_moe_only_tuning`` + an extra dummy run (see kernel_warmup.py) tune
  the trtllm-gen MoE ops for buckets up to ``dp_size * max_num_batched_tokens``
  rows; every other op only looks up its cache during that run.
- ``trtllm_moe_tactic_candidates`` races the default heuristic (tactic -1)
  against FlashInfer's candidates, so a tuned entry is never slower than the
  untuned path at the profiled shape (FlashInfer only does this in
  autotune_v2, which vLLM does not use with world_size > 1), and keeps the
  default for profiles with fewer rows than
  ``VLLM_FLASHINFER_MOE_TUNE_MIN_TOKENS`` (0 = tune all), to tune only
  prefill-sized MoE calls.

Kernels, routing and the serving-time cache lookup are FlashInfer's.
"""

import contextlib
from collections.abc import Iterator
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def is_trtllm_moe_op(custom_op: str) -> bool:
    """FlashInfer trtllm-gen MoE ops: flashinfer::trtllm_<dtype>_moe."""
    return custom_op.startswith("flashinfer::trtllm_") and custom_op.endswith("_moe")


def _tactic_key(tactic: Any) -> tuple[int, ...]:
    if isinstance(tactic, int):
        return (tactic,)
    return tuple(int(x) for x in tactic)


def race_default_tactic(tactics: list[Any], num_tokens: int, min_tokens: int) -> list:
    """Tactic -1 first, then FlashInfer's candidates (deduplicated)."""
    if not tactics:
        return tactics
    if num_tokens < min_tokens:
        return [-1]
    result: list[Any] = [-1]
    seen = {(-1,)}
    for tactic in tactics:
        key = _tactic_key(tactic)
        if key not in seen:
            seen.add(key)
            result.append(tactic)
    return result


@contextlib.contextmanager
def trtllm_moe_tactic_candidates(enabled: bool, min_tokens: int = 0) -> Iterator[None]:
    """Race tactic -1 (and skip small profiles) in trtllm-gen MoE tuning."""
    if not enabled:
        yield
        return
    try:
        from flashinfer.fused_moe.backends.trtllm import MoERunner
        from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
    except ImportError:
        logger.warning_once(
            "VLLM_FLASHINFER_MOE_EP_AWARE_TUNING is set but this FlashInfer "
            "has no flashinfer.fused_moe.backends.trtllm.MoERunner; ignoring."
        )
        yield
        return

    original = MoERunner.__dict__["get_valid_tactics"]

    def get_valid_tactics(self, inputs, profile):
        num_tokens = int(MoeRunnerInputs.from_list(inputs).hidden_states.shape[0])
        tactics = list(original(self, inputs, profile))
        return race_default_tactic(tactics, num_tokens, min_tokens)

    logger.info_once(
        "FlashInfer trtllm-gen MoE autotune: racing the default tactic, "
        "tuning only profiles with >= %d rows.",
        min_tokens,
    )
    MoERunner.get_valid_tactics = get_valid_tactics
    try:
        yield
    finally:
        MoERunner.get_valid_tactics = original


@contextlib.contextmanager
def trtllm_moe_only_tuning() -> Iterator[None]:
    """Inside an autotune context, profile only the trtllm-gen MoE ops.

    Other ops are served from the cache (or their fallback tactic) as if
    tuning were off, so a pass with wider buckets does not re-tune them.
    """
    from flashinfer.autotuner import AutoTuner

    original = AutoTuner.__dict__["choose_one"]

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
        if not self.is_tuning_mode or is_trtllm_moe_op(custom_op):
            return original(self, custom_op, runners, tuning_config, inputs, **kwargs)
        self.is_tuning_mode = False
        try:
            return original(self, custom_op, runners, tuning_config, inputs, **kwargs)
        finally:
            self.is_tuning_mode = True

    AutoTuner.choose_one = choose_one
    try:
        yield
    finally:
        AutoTuner.choose_one = original
