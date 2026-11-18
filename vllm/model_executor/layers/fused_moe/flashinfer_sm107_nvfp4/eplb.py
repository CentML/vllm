# SPDX-License-Identifier: Apache-2.0
"""Host side of EPLB for NVFP4 MegaMoE layers.

vLLM maps logical expert ids to physical replicas, and records the expert load,
only inside its own MegaMoE input staging (``prepare_megamoe*_inputs(...,
eplb_state=...)``). The NVFP4 plugin replaces that staging with its own Triton
launch (``staging.py``), so the staging kernel calls vLLM's
``eplb_map_and_record`` itself with the arguments built here, exactly as vLLM's
staging does. Everything here is cheap host code; no CUDA is touched.
"""

from typing import Any

PLUGIN = "NVFP4 MegaMoE plugin"

# The arguments of an inert staging launch: no map, no load recording. Same layout as vLLM's
# ``prepare_megamoe._eplb_kernel_args`` (5 tensors, then num_logical, map_stride, num_physical).
INERT_EPLB_ARGS: tuple = (None,) * 5 + (0,) * 3

# EPLB communicators: the SM107 plugin's policy (vllm_sm107_megamoe.experts: SUPPORTED_EPLB_COMMUNICATORS and
# _check_eplb; a test keeps the sets equal), kept here so this package runs next to the SM107 package already
# installed in a serving image. NVFP4 weights are ordinary PyTorch allocations, so nixl is refused for lack of
# validation rather than for die-local memory.
#   symm_mem    copies rows into its own symmetric receive buffer, host barriers only; use it with async EPLB.
#   torch_gloo  stages rows through host memory.
#   torch_nccl, pynccl  NCCL sends and receives; synchronous EPLB only.
SUPPORTED_EPLB_COMMUNICATORS = frozenset({"symm_mem", "torch_gloo", "torch_nccl", "pynccl"})
# NCCL kernels on the async EPLB stream would run next to the persistent MegaMoE kernel, which keeps every SM
# resident and synchronizes them in software.
SYNC_ONLY_EPLB_COMMUNICATORS = frozenset({"torch_nccl", "pynccl"})
UNSUPPORTED_EPLB_COMMUNICATORS = {
    "nixl": (
        "it has not been validated with NVFP4 MegaMoE layers (it registers the live expert weights and alpha "
        "buffers for RDMA reads by peers while the persistent kernel runs)"
    ),
    "torch_xccl": (
        "it is the XPU backend; on CUDA it would run NCCL transfers on the async EPLB stream next to the "
        "MegaMoE kernel"
    ),
}


def check_eplb_policy(parallel_config: Any) -> None:
    """Refuse elastic EP and the EPLB communicators NVFP4 MegaMoE layers cannot use (before any allocation)."""
    if getattr(parallel_config, "enable_elastic_ep", False):
        raise NotImplementedError(
            f"{PLUGIN} does not support elastic EP: its kernel variants and symmetric buffer are built for a fixed "
            "EP size and physical expert count"
        )
    eplb_config = parallel_config.eplb_config
    communicator = eplb_config.communicator
    if communicator in SUPPORTED_EPLB_COMMUNICATORS:
        if communicator in SYNC_ONLY_EPLB_COMMUNICATORS and getattr(eplb_config, "use_async", False):
            raise NotImplementedError(
                f"{PLUGIN} cannot use EPLB communicator {communicator!r} with use_async=true: its NCCL kernels "
                "would run on the async EPLB stream next to the MegaMoE kernel, which needs every SM. Set "
                "--eplb-config.communicator=symm_mem."
            )
        return
    reason = UNSUPPORTED_EPLB_COMMUNICATORS.get(communicator, "it has not been checked with NVFP4 MegaMoE layers")
    raise NotImplementedError(
        f"{PLUGIN} cannot use EPLB communicator {communicator!r}: {reason}. Set "
        "--eplb-config.communicator=symm_mem, or torch_gloo; torch_nccl and pynccl need "
        "--eplb-config.use_async=false. vLLM selects nixl when NIXL is installed and no communicator is set."
    )


def _ubatch_id() -> int:
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id

    return dbo_current_ubatch_id()


def eplb_kernel_args(eplb_state: Any) -> tuple:
    """Replica-map and load-recording staging args of one layer; inert without EPLB.

    A local copy of vLLM's ``prepare_megamoe._eplb_kernel_args`` (vllm/model_executor/
    layers/fused_moe/prepare_megamoe.py, rubin_disagg_1111 @fe62a5c75), so a rename of
    that private helper cannot break the plugin's import. Keep it in sync:

    ``(logical_to_physical_map, logical_replica_count, expert_load_view,
    should_record_tensor, num_unpadded_tokens[ubatch], num_logical, map_stride,
    num_physical)``.

    Every tensor is allocated once by ``EplbState.add_model`` and updated in place
    (maps on commit, load on reset, the record flag and unpadded count per step), so a
    CUDA graph that captured them follows EPLB without re-capture.
    """
    if eplb_state is None or eplb_state.logical_to_physical_map is None:
        return INERT_EPLB_ARGS
    mapping = eplb_state.logical_to_physical_map
    replicas = eplb_state.logical_replica_count
    load = eplb_state.expert_load_view
    record = eplb_state.should_record_tensor
    unpadded = eplb_state.num_unpadded_tokens_tensors
    assert replicas is not None and load is not None and record is not None
    assert unpadded is not None
    # vLLM indexes by dbo_current_ubatch_id(), which is 0 without dual-batch overlap.
    index = 0 if len(unpadded) == 1 else _ubatch_id()
    return (
        mapping,
        replicas,
        load,
        record,
        unpadded[index],
        mapping.shape[0],
        mapping.stride(0),
        load.numel(),
    )


def check_eplb_args(args: tuple, num_logical: int, num_physical: int, prefix: str) -> None:
    """The layer's EPLB state matches its expert counts (staging would silently misroute otherwise)."""
    if args[0] is None:
        return
    if args[5] != num_logical or args[7] != num_physical:
        raise RuntimeError(
            f"{prefix}: EPLB state covers {args[5]} logical / {args[7]} physical experts, but the layer has "
            f"{num_logical} logical / {num_physical} physical"
        )


def view_signature(views: list) -> tuple:
    """What EPLB's communicators take from layer 0 for every layer: count, shapes and dtypes of the views."""
    return tuple((tuple(v.shape), v.dtype) for v in views)
