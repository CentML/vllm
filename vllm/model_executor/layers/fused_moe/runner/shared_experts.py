# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from enum import IntEnum

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
)

logger = init_logger(__name__)

# H158 (VLLM_H158_SHARED_PRIO=1): overlapped shared experts of
# VLLM_H158_SHARED_PRIO_MIN..MAX tokens (a window that depends on the GPU and the
# model, no defaults: an unset bound is open) run on a stream of the greatest
# priority instead of the aux stream, so their kernels are dispatched ahead of
# the routed experts' pending CTAs: a shared-expert kernel launched after a
# routed FC1 that fills every SM otherwise waits for FC1's last wave. Scheduling
# only: same kernels, same results. Graph kernel nodes keep the capture stream's
# priority (PyTorch instantiates with cudaGraphInstantiateFlagUseNodePriority).
_H158_PRIO = os.environ.get("VLLM_H158_SHARED_PRIO", "0") == "1"
_H158_PRIO_MIN = int(os.environ.get("VLLM_H158_SHARED_PRIO_MIN") or 0)
_H158_PRIO_MAX = int(os.environ.get("VLLM_H158_SHARED_PRIO_MAX") or -1)  # -1: open
_h158_prio_streams: dict[int, torch.cuda.Stream] = {}


def _h158_prio_stream(num_tokens: int) -> torch.cuda.Stream | None:
    """The high-priority shared-experts stream of the current device when
    VLLM_H158_SHARED_PRIO applies to ``num_tokens``, else None."""
    if not (
        _H158_PRIO
        and _H158_PRIO_MIN <= num_tokens
        and (_H158_PRIO_MAX < 0 or num_tokens <= _H158_PRIO_MAX)
    ):
        return None
    device = torch.cuda.current_device()
    stream = _h158_prio_streams.get(device)
    if stream is None:
        _, greatest = torch.cuda.Stream.priority_range()
        stream = torch.cuda.Stream(device=device, priority=greatest)
        _h158_prio_streams[device] = stream
        logger.info(
            "H158: overlapped shared experts of %d-%s tokens on a stream of "
            "priority %d (VLLM_H158_SHARED_PRIO)",
            _H158_PRIO_MIN,
            _H158_PRIO_MAX if _H158_PRIO_MAX >= 0 else "any",
            stream.priority,
        )
    return stream


def h155_token_set(name: str) -> frozenset[int]:
    """H155: the token counts listed in env ``name`` ("N" or "N,M,...";
    unset or empty: none).
    """
    raw = os.environ.get(name, "")
    return frozenset(int(t) for t in raw.replace(" ", "").split(",") if t)


# H155 (default off): token counts at which router-first (VLLM_H105_ROUTER_FIRST)
# runs the shared experts on a highest-priority side stream. See MoERunner.
H155_RF_SHARED_HIPRI_TOKENS = h155_token_set("VLLM_H155_RF_SHARED_HIPRI_TOKENS")
_h155_hipri_streams: dict[int, torch.cuda.Stream] = {}


def h155_layer_token_sets(layer_name: str) -> tuple[frozenset[int], frozenset[int]]:
    """H155: (VLLM_H155_RF_SKIP_TOKENS, VLLM_H155_RF_SHARED_HIPRI_TOKENS) for
    the MoE layer ``layer_name``. Both are token counts of the target model's
    decode graphs. The MTP drafter's MoE ("mtp" in its layer name) runs on one
    row per request, so the same token counts are other batch sizes there: it
    gets both only with VLLM_H155_RF_DRAFT=1.
    """
    skip = h155_token_set("VLLM_H155_RF_SKIP_TOKENS")
    hipri = H155_RF_SHARED_HIPRI_TOKENS
    if (
        (skip or hipri)
        and "mtp" in layer_name.split(".")
        and os.environ.get("VLLM_H155_RF_DRAFT", "0") != "1"
    ):
        logger.info_once(
            "H155: drafter MoE %s keeps the parent order (VLLM_H155_RF_DRAFT=0)",
            layer_name,
        )
        return frozenset(), frozenset()
    return skip, hipri


def _h155_hipri_stream() -> torch.cuda.Stream:
    """One highest-priority stream per device, shared by every MoE layer.
    Captured kernels take the stream's priority as their node priority, and
    torch instantiates graphs with cudaGraphInstantiateFlagUseNodePriority.
    """
    dev = torch.cuda.current_device()
    stream = _h155_hipri_streams.get(dev)
    if stream is None:
        _, high = torch.cuda.Stream.priority_range()
        stream = torch.cuda.Stream(priority=high)
        _h155_hipri_streams[dev] = stream
    return stream


class SharedExpertsOrder(IntEnum):
    # No shared experts.
    NONE = (0,)

    # No overlap - defensively called before MK.
    NO_OVERLAP = (1,)

    # Overlapped with dispatch/combine in DP/EP - called by the MK.
    MK_INTERNAL_OVERLAPPED = (2,)

    # Overlapped with the gate, router, experts in aux stream.
    MULTI_STREAM_OVERLAPPED = (3,)


class SharedExperts(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        enable_dbo: bool,
        mk_can_overlap_shared_experts: Callable[[], bool],
        defer_gate: bool = False,
    ):
        super().__init__()

        # With defer_gate, the layer returns (ungated output, gate logits) from
        # forward_ungated and the runner applies the gate after the MoE op.
        self.defer_gate = defer_gate

        # The SharedExperts need to handle DBO since they can be called from
        # an MK's finalize method.  We keep a list of outputs indexed by current
        # DBO ubatch id to handle this case.  If DBO is not enabled, the
        # index is always 0 and the second output list element is ignored.
        self.enable_dbo = enable_dbo
        self._output: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None] = [
            None,
            None,
        ]
        self._layer = layer
        self._moe_config = moe_config

        self._mk_can_overlap_shared_experts = mk_can_overlap_shared_experts

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream")
            self._stream = None
        else:
            self._stream = aux_stream()
            if self._stream is not None:
                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")
                # One pair per DBO ubatch id to sync aux and main stream.
                self._input_ready_event = [torch.cuda.Event(), torch.cuda.Event()]
                self._output_ready_event = [torch.cuda.Event(), torch.cuda.Event()]
        # H155: created here, never during a capture; None when the env is off.
        self._hipri_stream: torch.cuda.Stream | None = None
        if H155_RF_SHARED_HIPRI_TOKENS and self._stream is not None:
            self._hipri_stream = _h155_hipri_stream()

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config

    @property
    def _disable_shared_experts_overlap(self) -> bool:
        # Disable shared expert overlap if:
        #   - we are using eplb with non-safe backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain

        # Both these comm backends have been shown to be safe for shared expert overlap.
        _EPLB_OVERLAP_SAFE_BACKENDS = (
            "allgather_reducescatter",
            "flashinfer_nvlink_one_sided",
        )

        parallel_config = self._moe_config.moe_parallel_config
        if getattr(self._layer, "shard_sequence_parallel", False):
            # TODO: we may enable this to optimize further
            return True
        return (
            parallel_config.enable_eplb
            and parallel_config.all2all_backend not in _EPLB_OVERLAP_SAFE_BACKENDS
        ) or parallel_config.use_fi_nvl_two_sided_kernels

    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
        if self._disable_shared_experts_overlap:
            return SharedExpertsOrder.NO_OVERLAP

        if self._mk_can_overlap_shared_experts():
            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED

        should_run_shared_in_aux_stream = (
            current_platform.is_cuda_alike()
            and self._stream is not None
            and hidden_states.shape[0]
            <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        )

        if should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        else:
            return SharedExpertsOrder.NO_OVERLAP

    def _run_layer(
        self, shared_experts_input: torch.Tensor, quantized_input: object | None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.defer_gate:
            return self._layer.forward_ungated(shared_experts_input, quantized_input)
        if quantized_input is not None:
            return self._layer(shared_experts_input, quantized_input)
        return self._layer(shared_experts_input)

    def record_input_ready(self, shared_experts_input: torch.Tensor) -> bool:
        """H105: record the fork point now (the input's producer), so that
        kernels the main stream launches before ``maybe_forward_async(...,
        input_recorded=True)`` are not dependencies of the shared experts."""
        if (
            self._determine_shared_experts_order(shared_experts_input)
            != SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        ):
            return False
        self._input_ready_event[self._output_idx].record(current_stream())
        return True

    def maybe_forward_async(
        self,
        shared_experts_input: torch.Tensor,
        quantized_input: object | None = None,
        input_recorded: bool = False,
        high_priority: bool = False,
    ) -> bool:
        """Enqueue shared experts on the aux stream without waiting for them.

        Returns true if the shared experts were enqueued, false otherwise. Call
        `wait` to wait for the shared experts to finish if this returns true.
        ``quantized_input`` is an optional pre-quantized copy of the input for
        layers whose input linear can consume it. ``input_recorded``: the fork
        point was recorded by ``record_input_ready``. ``high_priority`` (H155):
        enqueue them on the highest-priority side stream instead (same kernels,
        inputs and events; only their dispatch priority changes).
        """
        if (
            self._determine_shared_experts_order(shared_experts_input)
            != SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        ):
            return False
        assert self._stream is not None
        stream = self._stream
        if high_priority and self._hipri_stream is not None:
            stream = self._hipri_stream
            logger.info_once(
                "H155: router-first shared experts on a highest-priority stream "
                "at %s tokens (VLLM_H155_RF_SHARED_HIPRI_TOKENS)",
                str(sorted(H155_RF_SHARED_HIPRI_TOKENS)),
            )
        idx = self._output_idx
        assert self._output[idx] is None
        if not input_recorded:
            self._input_ready_event[idx].record(current_stream())
        # H158: the rest of its window, on its own stream of the same (greatest)
        # priority. H155's cell above (router-first, target, its token set)
        # keeps H155's stream; either flag alone behaves as it did on its own.
        if _H158_PRIO and stream is self._stream:
            stream = _h158_prio_stream(shared_experts_input.shape[0]) or stream
        with torch.cuda.stream(stream):
            self._input_ready_event[idx].wait(stream)
            self._output[idx] = self._run_layer(shared_experts_input, quantized_input)
            self._output_ready_event[idx].record(stream)
        return True

    def wait(self) -> None:
        """Block the main stream until `maybe_forward_async` output is ready."""
        assert self._stream is not None
        self._output_ready_event[self._output_idx].wait(current_stream())

    @property
    def _output_idx(self) -> int:
        return dbo_current_ubatch_id() if self.enable_dbo else 0

    @property
    def output(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """The layer output; ``(ungated output, gate logits)`` with defer_gate."""
        assert self._output[self._output_idx] is not None
        output = self._output[self._output_idx]
        self._output[self._output_idx] = None
        return output

    def forward(
        self,
        shared_experts_input: torch.Tensor,
        order: SharedExpertsOrder,
        quantized_input: object | None = None,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if order != experts_order:
            return None

        assert self._output[self._output_idx] is None

        self._output[self._output_idx] = self._run_layer(
            shared_experts_input, quantized_input
        )

        assert self._output[self._output_idx] is not None
