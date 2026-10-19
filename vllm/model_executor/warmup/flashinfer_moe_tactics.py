# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EP-aware tactic candidates for FlashInfer trtllm-gen MoE autotuning.

FlashInfer's trtllm-gen MoE runner (``flashinfer.fused_moe.backends.trtllm.
MoERunner``) asks ``trtllm_get_valid_moe_configs`` for its autotune candidates.
That C++ helper (``computeSelectedTileN``) keeps only the tile sizes around
``num_tokens * top_k / num_local_experts``: one tile below the covering tile,
the covering tile and two above. The estimate assumes every top-k slot of
every row is local. Under expert parallelism the rows carry global expert ids
(the all2all receive buffer has ``ep_size * max_tokens_per_rank`` rows; TP+EP
without all2all sees all tokens), so only about
``top_k * num_local_experts / num_experts`` slots per row are local and the
estimate is ``ep_size`` times too large. Example, DEP2 decode at 257 requests
per rank: 544 rows, tuning bucket 768, 144 local experts, about 14 tokens per
local expert; the candidates are tiles 32-256 while the untuned default (the
smallest tile of the window around the runtime row count) runs tile 16. The
tuned tactic therefore cannot match the default, and autotuning this op
regressed DEP2 throughput while it helped TP4.

``ep_aware_trtllm_moe_tactics`` widens the candidates during the tuning pass:

- EP: also list the candidates for the EP-effective row count
  ``num_tokens * num_local_experts // num_experts`` (no change without EP).
- Always: race the default heuristic (tactic -1), so a tuned entry is never
  slower than the untuned path at the profiled shape.

Only candidate enumeration changes. Every candidate still goes through
FlashInfer's validity checks and profiling (failing tactics are skipped), and
kernels, routing and the serving-time cache lookup are unchanged.
"""

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _tactic_key(tactic: Any) -> tuple[int, ...]:
    if isinstance(tactic, int):
        return (tactic,)
    return tuple(int(x) for x in tactic)


def _valid_configs(runner: Any, inputs: list[Any], num_tokens: int) -> list[Any]:
    """``MoERunner.get_valid_tactics`` for another row count (same cache)."""
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs

    moe_inputs = MoeRunnerInputs.from_list(inputs)
    nfse = runner.num_fused_shared_experts
    # Same key layout as MoERunner.get_valid_tactics (FlashInfer 0.7.2).
    query_key = (
        runner.dtype_act,
        runner.dtype_weights,
        runner.fp8_quantization_type,
        runner.top_k + nfse,
        runner.hidden_size,
        runner.hidden_size_output,
        runner.intermediate_size,
        runner.num_local_experts + nfse,
        runner.activation_type,
        runner.use_shuffled_weight,
        runner.weight_layout,
        runner.use_per_token_scaling,
        num_tokens,
        moe_inputs.gemm1_lora_delta is not None,
    )
    cache = type(runner).valid_tactics_dict
    instance_key = (id(runner.moe_op), query_key)
    if instance_key not in cache:
        try:
            cache[instance_key] = runner.moe_op.trtllm_get_valid_moe_configs(*query_key)
        except Exception as e:
            logger.debug("trtllm_get_valid_moe_configs%s failed: %s", query_key, e)
            return []
    return list(cache[instance_key])


def ep_aware_valid_tactics(
    runner: Any,
    get_valid_tactics: Callable[..., list[Any]],
    inputs: list[Any],
    profile: Any,
) -> list[Any]:
    """Candidates of ``get_valid_tactics`` + EP-effective window + default."""
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs

    tactics = list(get_valid_tactics(runner, inputs, profile))
    if not tactics:
        return tactics
    num_local = int(runner.num_local_experts)
    num_experts = int(getattr(runner, "num_experts", num_local) or num_local)
    num_tokens = int(MoeRunnerInputs.from_list(inputs).hidden_states.shape[0])
    if 0 < num_local < num_experts and num_tokens > 0:
        eff_tokens = max(1, num_tokens * num_local // num_experts)
        extra = _valid_configs(runner, inputs, eff_tokens)
        logger.debug(
            "EP-aware trtllm-gen MoE candidates: %d rows (%d effective, "
            "%d/%d local experts): tiles %s + %s",
            num_tokens,
            eff_tokens,
            num_local,
            num_experts,
            sorted({_tactic_key(t)[0] for t in tactics}),
            sorted({_tactic_key(t)[0] for t in extra}),
        )
        tactics += extra
    result: list[Any] = [-1]
    seen = {(-1,)}
    for tactic in tactics:
        key = _tactic_key(tactic)
        if key not in seen:
            seen.add(key)
            result.append(tactic)
    return result


@contextlib.contextmanager
def ep_aware_trtllm_moe_tactics(enabled: bool) -> Iterator[None]:
    """Widen FlashInfer trtllm-gen MoE autotune candidates inside the block."""
    if not enabled:
        yield
        return
    try:
        from flashinfer.fused_moe.backends.trtllm import MoERunner
    except ImportError:
        logger.warning_once(
            "VLLM_FLASHINFER_MOE_EP_AWARE_TUNING is set but this FlashInfer "
            "has no flashinfer.fused_moe.backends.trtllm.MoERunner; ignoring."
        )
        yield
        return

    original = MoERunner.__dict__["get_valid_tactics"]

    def get_valid_tactics(self, inputs, profile):
        return ep_aware_valid_tactics(self, original, inputs, profile)

    logger.info_once(
        "FlashInfer trtllm-gen MoE autotune: EP-aware candidates and default "
        "tactic racing enabled (VLLM_FLASHINFER_MOE_EP_AWARE_TUNING=1)."
    )
    MoERunner.get_valid_tactics = get_valid_tactics
    try:
        yield
    finally:
        MoERunner.get_valid_tactics = original
