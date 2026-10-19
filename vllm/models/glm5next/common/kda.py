# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer with separate convolutions and a bounded safe gate."""

import importlib
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch
from torch import nn
from transformers import Glm5NextTextConfig

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.kda_checkpoint import (
    FlashKDAPrefillCheckpointExporter,
    kda_prefill_checkpoint_alignment,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import (
    maybe_disable_graph_partition,
    set_weight_attrs,
)
from vllm.models.glm5next.common.fp8_linear import glm53_fp8_linear_quant_config
from vllm.models.glm5next.nvidia.ops.kda_prefill_conv import kda_causal_conv1d_planar
from vllm.models.glm5next.nvidia.ops.kda_prefill_flashinfer import (
    FlashInferKdaPrefill,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_rocm():
    from vllm.models.glm5next.amd.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
        fused_recurrent_kda,
    )
else:
    from vllm.models.glm5next.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
        fused_recurrent_kda,
    )

logger = init_logger(__name__)


class _Glm5NextMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged projection with multiple replicated output shards.

    Extends K3's ``_KimiGDNMergedColumnParallelLinear`` to support two
    replicated shards (f_a, g_a) instead of one. Pre-multiplies each
    replicated entry's output_size by tp_size so the per-rank shard
    divides back to the full size, and forces tp_rank=0 during weight
    loading for replicated shards.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_ids: tuple[int, ...],
        tp_size: int,
        **kwargs,
    ) -> None:
        self.replicated_shard_ids = set(replicated_shard_ids)
        output_sizes = output_sizes.copy()
        for sid in self.replicated_shard_ids:
            output_sizes[sid] *= tp_size
        super().__init__(input_size, output_sizes, **kwargs)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader_v2(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank


@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def _cast_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Fuse the fp32 cast + sigmoid into one Inductor kernel."""
    return x.float().sigmoid()


def _resolve_kda_prefill_backend(
    backend: str, head_dim: int, dtype: torch.dtype, lower_bound: float | None
) -> str:
    """Pick the chunked-prefill kernel: FlashKDA (fused CUDA, ~2-4x faster on
    SM90/SM10x/SM12x for bf16, head_dim 128 and a bounded gate) or the Triton
    ``chunk_kda_with_fused_gate`` path. ``backend`` comes from
    ``additional_config.kda_prefill_backend`` (auto / triton / flashkda;
    flashkda_planar is resolved as flashkda by the caller)."""
    if backend not in ("auto", "triton", "flashkda"):
        raise ValueError(f"Unsupported KDA prefill backend: {backend}")
    capability = current_platform.get_device_capability()
    supported = (
        current_platform.is_cuda()
        and capability is not None
        and capability.major in (9, 10, 12)
        and head_dim == 128
        and dtype == torch.bfloat16
        and lower_bound is not None
    )
    if backend == "flashkda" and not supported:
        raise RuntimeError(
            "FlashKDA requires CUDA SM90/SM10x/SM12x, bfloat16, head_dim=128 "
            "and a bounded KDA gate."
        )
    return "flashkda" if supported and backend != "triton" else "triton"


def _resolve_kda_decode_backend(
    backend: str,
    head_dim: int,
    conv_size: int,
    num_spec: int,
    lower_bound: float | None,
    dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
    recurrent_state_dtype: torch.dtype,
) -> str:
    """Pick the plain single-token decode path: the three Triton kernels (conv
    update, recurrence, gated RMSNorm), FlashInfer's fused_kda_decode, or the
    pipelined CUDA kernel (kda_decode_pipelined), the latter two doing all
    three in one launch. ``backend`` comes from
    ``additional_config.kda_decode_backend``; "flashinfer" and "pipelined" are
    opt-in, "triton" (default) and "auto" keep the Triton kernels.

    With speculative decoding every decode row is a draft-verify row of
    1 + num_spec tokens (and the conv state is 3 + num_spec wide), so the
    single-token kernels never run; "flashinfer" and "pipelined" then select
    FlashInfer's packed multi-token fused decode for the verify rows
    (see _FlashInferPackedKdaSpecDecode)."""
    if backend in ("auto", "triton"):
        return "triton"
    if backend not in ("flashinfer", "pipelined"):
        raise ValueError(f"Unsupported GLM-5.3 KDA decode backend: {backend}")
    if num_spec > 0:
        capability = current_platform.get_device_capability()
        if not (
            current_platform.is_cuda()
            and capability is not None
            and capability.major == 10
            and head_dim == 128
            and conv_size == 4
            and 1 <= num_spec <= 7
            and lower_bound is not None
            and lower_bound < 0
            and dtype == torch.bfloat16
            and conv_state_dtype == torch.bfloat16
            and recurrent_state_dtype in (torch.float32, torch.bfloat16)
            and not is_conv_state_dim_first()
        ):
            raise RuntimeError(
                f"kda_decode_backend={backend} with speculative decoding uses "
                "FlashInfer's packed multi-token KDA decode, which requires "
                "CUDA SM10x, bfloat16 activations and conv state, a float32 or "
                "bfloat16 recurrent state, head_dim 128, conv kernel 4, a "
                "bounded gate, 1-7 speculative tokens and the SD conv-state "
                "layout (unset VLLM_SSM_CONV_STATE_LAYOUT=DS). Use "
                "kda_decode_backend=triton otherwise."
            )
        return backend
    capability = current_platform.get_device_capability()
    if not (
        current_platform.is_cuda()
        and (
            backend == "flashinfer"
            or (capability is not None and capability.major >= 9)
        )
        and head_dim == 128
        and conv_size == 4
        and num_spec == 0
        and lower_bound is not None
        and dtype == torch.bfloat16
        and conv_state_dtype == torch.bfloat16
        and recurrent_state_dtype in (torch.float32, torch.bfloat16)
    ):
        raise RuntimeError(
            f"kda_decode_backend={backend} requires CUDA (SM90+ for pipelined), "
            "bfloat16 activations and conv state, a float32 or bfloat16 "
            "recurrent state, head_dim 128, conv kernel 4, a bounded gate and no "
            "speculative decoding."
        )
    return backend


class _FlashInferFusedKdaDecode:
    """FlashInfer fused KDA decode: depthwise conv + SiLU, one recurrent KDA
    step and the gated RMSNorm in one launch (CuTe DSL, one token per row).

    SD conv-state layout: FlashInfer's public ``fused_kda_decode``. DS layout
    (``VLLM_SSM_CONV_STATE_LAYOUT=DS``, needed for NIXL Mamba state transfer):
    the ``fused_kda_decode_ds`` variant next to the GLM-5.3 ops, generated from
    the same FlashInfer source with only the conv-state addressing changed.
    FlashInfer gates the head count with a Python allow-list (8, 12, 24, 32,
    48, 96 in 0.7.2) that omits GLM-5.3's 16 (TP4) and 64 (TP1/DEP) heads per
    rank; the kernel is head-count generic, so the local count is added.

    The first call goes through FlashInfer's validating wrapper; later calls
    launch the cached compiled kernel directly, whose TVM-FFI entry still
    checks dtypes, ranks and static strides. KDA ``_forward`` runs eagerly
    between breakable CUDA graph segments, so the wrapper's Python checks
    would otherwise cost host time in every layer of every step.
    """

    def __init__(
        self,
        conv_state_dim_first: bool,
        num_heads: int,
        lower_bound: float,
        norm_eps: float,
        state_dtype: torch.dtype | None = None,
    ) -> None:
        module: Any  # the FlashInfer kernel module (or its DS variant)
        run: Callable[..., torch.Tensor]
        if conv_state_dim_first:
            from vllm.models.glm5next.nvidia.ops import fused_kda_decode_ds

            module = fused_kda_decode_ds
            run = fused_kda_decode_ds.run_fused_kda_decode
        else:
            from vllm.utils.flashinfer import (
                flashinfer_fused_kda_decode,
                has_flashinfer_fused_kda_decode,
            )

            if not has_flashinfer_fused_kda_decode():
                raise RuntimeError(
                    "kda_decode_backend=flashinfer needs FlashInfer's "
                    "fused_kda_decode (CuTe DSL)."
                )
            # flashinfer.kda_kernels rebinds the name "fused_kda_decode" to the
            # function, so import the submodule by its path.
            module = importlib.import_module("flashinfer.kda_kernels.fused_kda_decode")
            run = flashinfer_fused_kda_decode
        if num_heads not in module._SUPPORTED_HEADS:
            module._SUPPORTED_HEADS = (*module._SUPPORTED_HEADS, num_heads)
        self._module = module
        self._run = run
        self._lower_bound = lower_bound
        self._norm_eps = norm_eps
        self._kernel: Callable[..., None] | None = None
        get_kernel = getattr(module, "_get_compiled_kernel", None)
        if state_dtype is not None and get_kernel is not None:
            # CuTe DSL compile (disk-cached under FLASHINFER_WORKSPACE_BASE)
            # at model construction, not on the first decode step: profile
            # and PIECEWISE-capture runs carry no KDA metadata.
            get_kernel(state_dtype, lower_bound, float(norm_eps))
        logger.info_once(
            "Using FlashInfer fused KDA decode (%s conv-state layout).",
            "DS" if conv_state_dim_first else "SD",
        )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_indices: torch.Tensor,
        state: torch.Tensor,
        output_gate: torch.Tensor,
        norm_weight: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        kernel = self._kernel
        if kernel is not None:
            # Launch ABI of run_fused_kda_decode's direct (non-packed) path.
            kernel(
                x,
                weight,
                conv_state,
                raw_gate,
                raw_beta,
                A_log,
                dt_bias,
                state_indices,
                state,
                output_gate,
                norm_weight,
                output,
            )
            return
        self._run(
            x=x,
            weight=weight,
            conv_state=conv_state,
            raw_gate=raw_gate,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            state_indices=state_indices,
            state=state,
            output_gate=output_gate,
            norm_weight=norm_weight,
            lower_bound=self._lower_bound,
            norm_eps=self._norm_eps,
            output=output,
        )
        get_kernel = getattr(self._module, "_get_compiled_kernel", None)
        if get_kernel is not None:
            self._kernel = get_kernel(
                state.dtype, self._lower_bound, float(self._norm_eps)
            )


# Below this many (row, head) items a step is latency-bound and FlashInfer's
# one-CTA-per-item kernel is faster; at and above it the pipelined kernel wins
# with a bf16 state (VR200 jobs 800580/802514: 1.02-1.04x at 512 items,
# 1.26-1.32x at 4096-32768; 0.85-0.89x at 8-256 items).
_PIPELINED_MIN_ITEMS = 512


class _PipelinedKdaDecode:
    """Pipelined single-token KDA decode (kda_decode_pipelined.cu): one launch
    of a persistent, warp-specialised kernel that streams the recurrent state
    through a shared-memory ring with bulk async copies (conv + SiLU, q/k
    norms, gates, recurrence and gated RMSNorm fused), taking vLLM's merged
    [3*H*128, 4] conv weight and either conv-state layout as is.

    Used for bf16 recurrent state with at least _PIPELINED_MIN_ITEMS
    (row, head) items per step; smaller steps and fp32 state (where it only
    matches FlashInfer) go to FlashInfer's fused decode, which shares the
    same state contract. Both are built at construction (FlashInfer nvcc JIT
    for the CUDA module, CuTe DSL for FlashInfer), so a missing toolchain fails
    at startup and no compile lands in the first decode steps."""

    def __init__(
        self,
        conv_state_dim_first: bool,
        num_heads: int,
        lower_bound: float,
        norm_eps: float,
        state_dtype: torch.dtype,
    ) -> None:
        self._fallback = _FlashInferFusedKdaDecode(
            conv_state_dim_first, num_heads, lower_bound, norm_eps, state_dtype
        )
        self._kernel: Callable[..., None] | None = None
        self._min_rows = -(-_PIPELINED_MIN_ITEMS // num_heads)
        if state_dtype == torch.bfloat16:
            from vllm.models.glm5next.nvidia.ops import kda_decode_pipelined

            self._kernel = kda_decode_pipelined.get_kernel()
            logger.info_once(
                "Using the pipelined KDA decode kernel (bf16 state, >= %d rows "
                "per step; FlashInfer below).",
                self._min_rows,
            )
        else:
            logger.info_once(
                "kda_decode_backend=pipelined: %s recurrent state uses "
                "FlashInfer's fused decode (pipelined kernel is bf16-only).",
                state_dtype,
            )
        self._lower_bound = float(lower_bound)
        self._norm_eps = float(norm_eps)

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        fi_weight: torch.Tensor,
        conv_state: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_indices: torch.Tensor,
        state: torch.Tensor,
        output_gate: torch.Tensor,
        norm_weight: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """``weight``: merged [3*H*128, 4] conv weight (pipelined kernel);
        ``fi_weight``: the same in FlashInfer's [3, 4, H*128] layout."""
        if self._kernel is None or x.shape[0] < self._min_rows:
            self._fallback(
                x=x,
                weight=fi_weight,
                conv_state=conv_state,
                raw_gate=raw_gate,
                raw_beta=raw_beta,
                A_log=A_log,
                dt_bias=dt_bias,
                state_indices=state_indices,
                state=state,
                output_gate=output_gate,
                norm_weight=norm_weight,
                output=output,
            )
            return
        self._kernel(
            x,
            weight,
            conv_state,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_indices,
            state,
            output_gate,
            norm_weight,
            output,
            self._lower_bound,
            self._norm_eps,
            0,
            0,
            -1,
        )


class _FlashInferPackedKdaSpecDecode:
    """FlashInfer packed multi-token fused KDA decode for draft-verify rows
    (FlashInfer 0.7.2 ``fused_kda_decode_multitoken``, CuTe DSL, SM10x): conv +
    SiLU, T = 1 + num_spec recurrent KDA steps and the gated RMSNorm in one
    launch per layer, replacing causal_conv1d_update + fused_recurrent_kda +
    o_norm on the spec path.

    Same cache contract as vLLM's Triton spec path: ``state_indices`` [N, T],
    column t receives the recurrent checkpoint after token t; the source
    checkpoint is column ``num_accepted_tokens - 1``; the conv window (width
    T + 2, SD layout) lives in column 0's slot and is read at offset
    ``num_accepted_tokens - 1``. Non-positive slots and zero-length rows (CUDA
    graph padding) are null rows that touch no cache.

    FlashInfer's validating wrapper allows 12/24/32/48/96 heads; the kernel is
    head-count generic (grid H x N, H a compile-time constant), so like the
    single-token path we launch it for GLM-5.3's 16 (TP4) / 64 (TP1, DEP)
    local heads. That bypasses the wrapper, so this class repeats its tile
    pick and does the structural checks once, on the first call. All tile
    specialisations reachable for max_num_seqs are compiled at construction
    (disk-cached under FLASHINFER_WORKSPACE_BASE), so no CuTe DSL compile lands
    in a served step.
    """

    def __init__(
        self,
        num_heads: int,
        num_spec: int,
        lower_bound: float,
        norm_eps: float,
        state_dtype: torch.dtype,
        max_num_seqs: int,
    ) -> None:
        # flashinfer.kda_kernels rebinds the module name to the function, so
        # import the submodule by its path.
        module = importlib.import_module(
            "flashinfer.kda_kernels.fused_kda_decode_multitoken"
        )
        for attr in ("_pick_tile", "_get_compiled_kernel", "_required_smem_bytes"):
            if not callable(getattr(module, attr, None)):
                raise RuntimeError(
                    "kda_decode_backend with speculative decoding needs "
                    f"FlashInfer 0.7.2's fused_kda_decode_multitoken.{attr}."
                )
        self._module = module
        self._num_heads = num_heads
        self._T = num_spec + 1
        self._lower_bound = float(lower_bound)
        self._norm_eps = float(norm_eps)
        self._state_is_bf16 = state_dtype == torch.bfloat16
        device = torch.accelerator.current_device_index()
        props = torch.cuda.get_device_properties(device)
        self._sm_count = props.multi_processor_count
        self._capability = (props.major, props.minor)
        self._validated = False
        # Reachable tiles: _pick_tile depends on N * H only (and T, SMs).
        tiles = {self._tile(n) for n in range(1, max(1, max_num_seqs) + 1)}
        for tile, norm_order in sorted(tiles):
            required = module._required_smem_bytes(
                self._T, tile[0], tile[1], tile[3], tile[4], norm_order
            )
            if required > props.shared_memory_per_block_optin:
                raise RuntimeError(
                    f"FlashInfer packed KDA decode T={self._T} tile {tile} needs "
                    f"{required} B of shared memory; the device allows "
                    f"{props.shared_memory_per_block_optin} B."
                )
            self._kernel(tile, norm_order)
        logger.info_once(
            "Using FlashInfer packed fused KDA decode for spec-verify rows "
            "(T=%d, %d local heads, %s state, %d tile variants).",
            self._T,
            num_heads,
            "bf16" if self._state_is_bf16 else "fp32",
            len(tiles),
        )

    def _tile(self, num_seqs: int) -> tuple[tuple[int, ...], int]:
        tile = tuple(
            self._module._pick_tile(
                self._T, num_seqs * self._num_heads, self._sm_count, self._capability
            )
        )
        R, KQ, _, NCH, SPLIT = tile
        # Same rule as FlashInfer's _run_fused_kda_decode_multitoken.
        norm_order = 3 if (R, KQ, NCH, SPLIT) == (2, 16, 2, 1) else 1
        return tile, norm_order

    def _kernel(self, tile: tuple[int, ...], norm_order: int) -> Callable[..., None]:
        R, KQ, bpm, NCH, SPLIT = tile
        # functools.cache'd in FlashInfer: a dict lookup after the first call.
        return self._module._get_compiled_kernel(
            self._T,
            self._num_heads,
            self._lower_bound,
            self._norm_eps,
            R,
            KQ,
            bpm,
            NCH,
            SPLIT,
            self._state_is_bf16,
            norm_order,
        )

    def _validate(self, x, weight, conv_state, raw_gate, raw_beta, A_log,
                  dt_bias, state_indices, query_start_loc, num_accepted_tokens,
                  state, output_gate, norm_weight, output) -> None:  # fmt: skip
        H, T, D = self._num_heads, self._T, 128
        P = H * D
        n, rows = state_indices.shape[0], x.shape[0]
        checks = {
            "x [rows, 3P] bf16, unit channel stride": x.dtype == torch.bfloat16
            and x.shape[1] == 3 * P
            and x.stride(1) == 1,
            "weight [3, 4, P] fp32 contiguous": weight.dtype == torch.float32
            and tuple(weight.shape) == (3, 4, P)
            and weight.is_contiguous(),
            "conv_state [slots, 3P, T+2] bf16 SD": conv_state.dtype == torch.bfloat16
            and tuple(conv_state.shape[1:]) == (3 * P, T + 2)
            and conv_state.stride(1) == 1
            and conv_state.stride(2) == 3 * P,
            "raw_gate [1, rows, H, 128] contiguous": tuple(raw_gate.shape)
            == (1, rows, H, D)
            and raw_gate.is_contiguous(),
            "raw_beta [1, rows, H]": tuple(raw_beta.shape) == (1, rows, H)
            and raw_beta.stride(2) == 1,
            "A_log [H] fp32": A_log.dtype == torch.float32
            and tuple(A_log.shape) == (H,),
            "dt_bias [P] fp32": dt_bias.dtype == torch.float32
            and tuple(dt_bias.shape) == (P,),
            "state_indices [N, T] int32 contiguous": state_indices.dtype == torch.int32
            and state_indices.shape[1] == T
            and state_indices.is_contiguous(),
            "query_start_loc [N+1] int32": query_start_loc.dtype == torch.int32
            and tuple(query_start_loc.shape) == (n + 1,)
            and query_start_loc.is_contiguous(),
            "num_accepted_tokens [N] int32": num_accepted_tokens.dtype == torch.int32
            and tuple(num_accepted_tokens.shape) == (n,)
            and num_accepted_tokens.is_contiguous(),
            "state [slots, H, 128, 128]": state.dtype
            == (torch.bfloat16 if self._state_is_bf16 else torch.float32)
            and tuple(state.shape[1:]) == (H, D, D)
            and tuple(state.stride()[1:]) == (D * D, D, 1)
            and state.shape[0] == conv_state.shape[0],
            "output_gate [rows, H, 128]": tuple(output_gate.shape) == (rows, H, D)
            and output_gate.stride(2) == 1
            and output_gate.stride(1) == D,
            "norm_weight [128] fp32": norm_weight.dtype == torch.float32
            and tuple(norm_weight.shape) == (D,),
            "output [1, rows, H, 128] contiguous": tuple(output.shape)
            == (1, rows, H, D)
            and output.is_contiguous(),
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"FlashInfer packed KDA decode: bad inputs: {bad}")
        self._validated = True

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        state: torch.Tensor,
        output_gate: torch.Tensor,
        norm_weight: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if not state_indices.is_contiguous():
            # Unstaged (eager) metadata slices the block table.
            state_indices = state_indices.contiguous()
        if not self._validated:
            self._validate(x, weight, conv_state, raw_gate, raw_beta, A_log,
                           dt_bias, state_indices, query_start_loc,
                           num_accepted_tokens, state, output_gate,
                           norm_weight, output)  # fmt: skip
        tile, norm_order = self._tile(state_indices.shape[0])
        self._kernel(tile, norm_order)(
            x,
            weight,
            conv_state,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_indices,
            query_start_loc,
            num_accepted_tokens,
            state,
            output_gate,
            norm_weight,
            output,
            x.stride(0),
            raw_beta.stride(1),
            output_gate.stride(0),
        )


class Glm5NextLinearAttention(GatedDeltaNetAttention):
    head_dim: int
    num_heads: int
    conv_size: int

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> MambaSpec:
        spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MambaSpec)
        alignment = kda_prefill_checkpoint_alignment(self.kda_prefill_backend)
        return replace(
            spec,
            num_prefill_checkpoint_blocks=int(alignment is not None),
            prefill_checkpoint_alignment=alignment,
        )

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # conv_state width must include num_spec so the spec-decode conv update
        # (causal_conv1d_update with num_accepted_tokens + max_query_len) can
        # slide the window across the draft-verify tokens without reading past
        # the allocated width. Matches qwen_gdn_linear_attn.get_state_shape.
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config: Glm5NextTextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        # KDA projections remain BF16 because fp8 checkpoints omit their scales.
        saved_quant_config = vllm_config.quant_config
        try:
            vllm_config.quant_config = None
            super().__init__(config, vllm_config, prefix)
        finally:
            vllm_config.quant_config = saved_quant_config

        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.conv_size = config.linear_conv_kernel_dim
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(projection_size, self.tp_size)

        # Opt-in online FP8 (additional_config.glm53_fp8_linear); None keeps
        # the projection in BF16 (self.quant_config is None here).
        def _proj_quant_config(group: str):
            return (
                glm53_fp8_linear_quant_config(group, prefix, vllm_config)
                or self.quant_config
            )

        _, recurrent_state_shape = self.get_state_shape()
        _, recurrent_state_dtype = self.get_state_dtype()
        scatter_states.register_warmup(
            state_shape=recurrent_state_shape,
            state_dtype=recurrent_state_dtype,
            indices_dtype=torch.int32,
            max_num_tokens=vllm_config.scheduler_config.max_num_seqs,
        )

        # Merge q, k, v, b, f_a, g_a projections into one GEMM (6→1 launches).
        # Order matches checkpoint's fused_qkvbfg_a_proj convention.
        # Shards 4 (f_a) and 5 (g_a) are replicated across TP ranks.
        self.in_proj_qkvbfg_a = _Glm5NextMergedColumnParallelLinear(
            self.hidden_size,
            [
                projection_size,  # q (shard 0)
                projection_size,  # k (shard 1)
                projection_size,  # v (shard 2)
                self.num_heads,  # b (shard 3)
                self.head_dim,  # f_a (shard 4, replicated)
                self.head_dim,  # g_a (shard 5, replicated)
            ],
            replicated_shard_ids=(4, 5),
            tp_size=self.tp_size,
            bias=False,
            quant_config=_proj_quant_config("kda_in"),
            prefix=f"{prefix}.in_proj_qkvbfg_a",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=_proj_quant_config("kda_gate"),
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)
        # Lazily-built merged q|k|v conv weight (built on first forward, after
        # weights are loaded). See _forward.
        self._merged_conv_weight: torch.Tensor | None = None

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})

        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=_proj_quant_config("kda_gate"),
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=_proj_quant_config("kda_out"),
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Checkpoints store A_log as 1-D; the model parameter is 4-D.
        def _a_log_weight_loader(param, loaded_weight):
            if loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view([1, 1, -1, 1])
            return sharded_weight_loader(2)(param, loaded_weight)

        self.A_log.weight_loader = _a_log_weight_loader

        # GLM-5.3-Flash uses a bounded sigmoid gate instead of the default
        # unbounded softplus gate.
        self.kda_safe_gate = True
        self.kda_lower_bound = config.linear_lower_bound
        # Process-global conv-state layout, resolved once here instead of on
        # every _forward call (it reads an env-derived flag each time).
        self._conv_state_dim_first = is_conv_state_dim_first()

        additional_config = vllm_config.additional_config
        requested_prefill_backend = (
            additional_config.get("kda_prefill_backend", "auto")
            if isinstance(additional_config, dict)
            else "auto"
        )
        # "flashkda_planar": FlashKDA whose inputs come from a short conv with
        # planar q|k|v output (no .contiguous() copies), with the decode rows
        # of mixed steps peeled off to the recurrent kernel. "flashinfer":
        # the same, with FlashInfer's CuTe DSL prefill in place on the state
        # pool instead of FlashKDA (FlashKDA stays for spec-decode steps and
        # prefix-cache checkpoints).
        self._planar_prefill = requested_prefill_backend in (
            "flashkda_planar",
            "flashinfer",
        )
        self.kda_prefill_backend = _resolve_kda_prefill_backend(
            "flashkda" if self._planar_prefill else requested_prefill_backend,
            self.head_dim,
            vllm_config.model_config.dtype,
            self.kda_lower_bound,
        )
        self._fi_prefill: FlashInferKdaPrefill | None = None
        if requested_prefill_backend == "flashinfer":
            assert self.kda_lower_bound is not None
            self._fi_prefill = FlashInferKdaPrefill(
                self.local_num_heads, self.head_dim, self.kda_lower_bound
            )
        conv_state_dtype, recurrent_state_dtype = self.get_state_dtype()
        self.kda_decode_backend = _resolve_kda_decode_backend(
            additional_config.get("kda_decode_backend", "triton")
            if isinstance(additional_config, dict)
            else "triton",
            self.head_dim,
            self.conv_size,
            self.num_spec,
            self.kda_lower_bound,
            vllm_config.model_config.dtype,
            conv_state_dtype,
            recurrent_state_dtype,
        )
        self._fi_fused_kda_decode: (
            _FlashInferFusedKdaDecode | _PipelinedKdaDecode | None
        ) = None
        # Built on first use (after weight loading): FlashInfer's conv weight
        # layout [3, 4, H * 128] and the fp32 norm weight.
        self._fi_conv_weight: torch.Tensor | None = None
        self._fi_norm_weight: torch.Tensor | None = None
        # Spec decode: FlashInfer packed multi-token decode for the verify rows
        # (the single-token kernels below are never reached then).
        self._fi_spec_kda_decode: _FlashInferPackedKdaSpecDecode | None = None
        # Fused decode backends leave _forward's output gated-RMS-normed in
        # every step (fused kernels, or o_norm inside _forward on the Triton
        # paths), so forward() must not apply o_norm again.
        self._output_normed = self.kda_decode_backend != "triton"
        if self.kda_decode_backend != "triton" and self.num_spec > 0:
            assert self.kda_lower_bound is not None
            self._fi_spec_kda_decode = _FlashInferPackedKdaSpecDecode(
                self.local_num_heads,
                self.num_spec,
                self.kda_lower_bound,
                self.o_norm.eps,
                recurrent_state_dtype,
                vllm_config.scheduler_config.max_num_seqs,
            )
        elif self.kda_decode_backend == "flashinfer":
            assert self.kda_lower_bound is not None
            self._fi_fused_kda_decode = _FlashInferFusedKdaDecode(
                self._conv_state_dim_first,
                self.local_num_heads,
                self.kda_lower_bound,
                self.o_norm.eps,
                recurrent_state_dtype,
            )
        elif self.kda_decode_backend == "pipelined":
            assert self.kda_lower_bound is not None
            self._fi_fused_kda_decode = _PipelinedKdaDecode(
                self._conv_state_dim_first,
                self.local_num_heads,
                self.kda_lower_bound,
                self.o_norm.eps,
                recurrent_state_dtype,
            )
        self._flashkda_buffer_specs: (
            tuple[tuple[tuple[int, ...], torch.dtype], ...] | None
        ) = None
        self._checkpoint_exporter: FlashKDAPrefillCheckpointExporter | None = None
        if self.kda_prefill_backend == "flashkda":
            import vllm._flashkda_C  # noqa: F401

            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            max_seqs = vllm_config.scheduler_config.max_num_seqs
            workspace_size = torch.ops._flashkda_C.get_workspace_size(
                max_tokens, self.local_num_heads, max_seqs
            )
            self._flashkda_buffer_specs = (
                (
                    (max_seqs, self.local_num_heads, self.head_dim, self.head_dim),
                    self.get_state_dtype()[1],
                ),
                (
                    (max_seqs, self.local_num_heads, self.head_dim, self.head_dim),
                    self.get_state_dtype()[1],
                ),
                ((workspace_size,), torch.uint8),
                # Output buffer for steps that also carry spec-decode tokens:
                # the non-spec tokens are then scattered by non_spec_token_indx.
                (
                    (1, max_tokens, self.local_num_heads, self.head_dim),
                    vllm_config.model_config.dtype,
                ),
            )
            self._checkpoint_exporter = FlashKDAPrefillCheckpointExporter(
                state_len=self.conv_size - 1
            )

    def _flashkda_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor | None,
        checkpoint: MambaPrefillCheckpointMetadata | None = None,
        raw_qkv: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused KDA chunked prefill (FlashKDA). Takes the raw gate logits ``g``
        and raw ``beta`` logits, l2-normalizes q/k in-kernel and applies the
        bounded gate ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``,
        matching ``chunk_kda_with_fused_gate(..., safe_gate=True)``. Writes the
        attention output into ``out`` (a workspace buffer when ``None``) and
        returns ``(out, final_state)``."""
        assert self._flashkda_buffer_specs is not None
        final_state, checkpoint_state, workspace, workspace_out = (
            current_workspace_manager().get_simultaneous(*self._flashkda_buffer_specs)
        )
        final_state = final_state[: initial_state.shape[0]]
        checkpoint_state = (
            checkpoint_state[: initial_state.shape[0]]
            if checkpoint is not None
            else None
        )
        if out is None:
            out = workspace_out[:, : q.shape[1]]
        # FlashKDA hardcodes dense q/k/v/g strides; beta may be row-strided.
        torch.ops._flashkda_C.fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta,
            self.head_dim**-0.5,
            out,
            workspace,
            self.A_log.view(-1),
            self.dt_bias.view(-1, self.head_dim),
            self.kda_lower_bound,
            initial_state.contiguous(),
            final_state,
            cu_seqlens.contiguous(),
            checkpoint_state,
            checkpoint.checkpoint_offsets.contiguous()
            if checkpoint is not None
            else None,
        )
        if checkpoint is not None:
            assert self._checkpoint_exporter is not None
            assert raw_qkv is not None and conv_state is not None
            assert recurrent_state is not None
            self._checkpoint_exporter.export(
                checkpoint,
                raw_qkv=raw_qkv,
                conv_state=conv_state,
                recurrent_checkpoint=checkpoint_state,
                recurrent_state=recurrent_state,
                cu_seqlens=cu_seqlens,
            )
        return out, final_state

    def _fused_decode(
        self,
        x: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_state: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        state_indices: torch.Tensor,
        state: torch.Tensor,
        output_gate: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Single-token rows through the fused decode backend (conv + SiLU
        with conv-state update, recurrence, gated RMSNorm; normed output)."""
        assert self._fi_fused_kda_decode is not None
        if self._fi_norm_weight is None:
            self._fi_norm_weight = self.o_norm.weight.float().contiguous()
        if self._fi_conv_weight is None:
            self._fi_conv_weight = (
                conv_weights.view(3, self.local_projection_size, self.conv_size)
                .transpose(1, 2)
                .contiguous()
            )
        if isinstance(self._fi_fused_kda_decode, _PipelinedKdaDecode):
            self._fi_fused_kda_decode(
                x=x,
                weight=conv_weights,
                fi_weight=self._fi_conv_weight,
                conv_state=conv_state,
                raw_gate=raw_gate,
                raw_beta=raw_beta,
                A_log=self.A_log.view(-1),
                dt_bias=self.dt_bias,
                state_indices=state_indices,
                state=state,
                output_gate=output_gate,
                norm_weight=self._fi_norm_weight,
                output=output,
            )
        else:
            self._fi_fused_kda_decode(
                x=x,
                weight=self._fi_conv_weight,
                conv_state=conv_state,
                raw_gate=raw_gate,
                raw_beta=raw_beta,
                A_log=self.A_log.view(-1),
                dt_bias=self.dt_bias,
                state_indices=state_indices,
                state=state,
                output_gate=output_gate,
                norm_weight=self._fi_norm_weight,
                output=output,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        # One merged GEMM for q, k, v, b, f_a, g_a (replaces 6 separate GEMMs).
        projected = self.in_proj_qkvbfg_a(hidden_states)[0]
        qkv, beta_raw, f_a, g_a = projected.split(
            [
                3 * self.local_projection_size,
                self.local_num_heads,
                self.head_dim,
                self.head_dim,
            ],
            dim=-1,
        )

        # Beta stays raw (bf16) here: the recurrent kernel sigmoids it in fp32
        # at load (SIGMOID_BETA), and only the chunked prefill path needs the
        # pre-computed fp32 sigmoid — computed lazily in _forward. Pure decode
        # / spec-verify steps then skip the _cast_sigmoid kernel and its fp32
        # intermediate entirely.
        beta = beta_raw.unsqueeze(0)
        g1 = self.f_b_proj(f_a)[0]
        g1 = g1.reshape(1, -1, self.local_num_heads, self.head_dim)

        g_proj_states = self.g_b_proj(g_a)[0]
        # Must stay 3D: rms_norm_gated reads H from g.shape[-2].
        g2 = g_proj_states.reshape(-1, self.local_num_heads, self.head_dim)

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Call the decorated eager break directly so host-side prefill branches
        # are not captured by PIECEWISE CUDA graphs.
        self._forward(
            qkv_proj_states=qkv,
            g1=g1,
            beta=beta,
            core_attn_out=core_attn_out,
            g2=g2,
        )
        # With the FlashInfer decode backend _forward always leaves the output
        # gated-RMS-normed (fused kernel, or o_norm inside _forward on the
        # Triton paths). This must not depend on the step type: a breakable
        # CUDA graph replay re-runs _forward eagerly for the live batch but
        # replays whatever followed it at capture time.
        if not self._output_normed:
            core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = core_attn_out.reshape(core_attn_out.size(1), -1)
        return self.o_proj(core_attn_out)[0]

    @eager_break_during_capture
    def _forward(
        self,
        qkv_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        g2: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
        if attn_metadata_narrowed is None:
            # Profile/warmup dummy runs may omit mamba-family metadata.
            return
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        # Spec-decode metadata (all None when speculative decoding is disabled).
        spec_sequence_masks = attn_metadata_narrowed.spec_sequence_masks
        spec_query_start_loc = attn_metadata_narrowed.spec_query_start_loc
        spec_state_indices_tensor = attn_metadata_narrowed.spec_state_indices_tensor
        spec_token_indx = attn_metadata_narrowed.spec_token_indx
        non_spec_token_indx = attn_metadata_narrowed.non_spec_token_indx
        num_accepted_tokens = attn_metadata_narrowed.num_accepted_tokens
        num_spec_decodes = attn_metadata_narrowed.num_spec_decodes
        use_spec = spec_sequence_masks is not None and num_spec_decodes > 0
        # Safe-gate checkpoints use the bounded sigmoid variant.
        safe_gate = self.kda_safe_gate
        lower_bound = self.kda_lower_bound
        constant_caches = self.kv_cache

        qkv_proj_states = qkv_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        (conv_state, recurrent_state) = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        # Layout is process-global and resolved once at init (see __init__).
        if not self._conv_state_dim_first:
            conv_state = conv_state.transpose(-1, -2)

        # One merged short-conv over q|k|v instead of three separate calls. The
        # 1D conv is independent per channel, so concatenating q/k/v along the
        # channel dim and running a single causal_conv1d is bit-identical to
        # three calls. The merged weight is q|k|v conv weights concatenated;
        # built once and cached (params are fixed after load). conv_state is
        # already stored as the merged q|k|v state, so it is used directly.
        if self._merged_conv_weight is None:

            def _w(m):
                return m.weight.view(m.weight.size(0), m.weight.size(2))

            self._merged_conv_weight = torch.cat(
                [_w(self.q_conv1d), _w(self.k_conv1d), _w(self.v_conv1d)],
                dim=0,
            ).contiguous()
        conv_weights = self._merged_conv_weight
        conv_bias = self.q_conv1d.bias

        # Plain single-token decode step (no prefill, no spec tokens): fused
        # conv + recurrence + gated RMSNorm in one launch (FlashInfer or the
        # pipelined kernel), written straight into the layer output buffer.
        # Same state contract as the Triton path (recurrent state
        # [slot, H, V, K], conv state in the configured layout, slot 0 = null
        # row, no cache write).
        if (
            self._fi_fused_kda_decode is not None
            and not use_spec
            and attn_metadata_narrowed.num_prefills == 0
            and attn_metadata_narrowed.num_decodes > 0
        ):
            assert non_spec_state_indices_tensor is not None
            num_decodes = attn_metadata_narrowed.num_decodes
            assert num_decodes == num_actual_tokens
            self._fused_decode(
                x=qkv_proj_states,
                conv_weights=conv_weights,
                conv_state=conv_state,
                raw_gate=g1,
                raw_beta=beta,
                state_indices=non_spec_state_indices_tensor[:num_decodes],
                state=recurrent_state,
                output_gate=g2[:num_decodes],
                output=core_attn_out[:, :num_decodes],
            )
            return

        # Split projections / gating into spec (draft-verify) and non-spec token
        # groups when speculative decoding is active. Spec tokens carry
        # num_spec+1 recurrent-state columns each and are advanced with
        # num_accepted_tokens for rejection-sampling rollback; non-spec tokens
        # are one-per-request. Mirrors olmo_gdn_linear_attn.py. Projections are
        # [n, *] (token dim 0); g1/beta are [1, n, h, d] (token dim 1).
        if use_spec:
            # In a pure spec-verify step (no non-spec tokens) the metadata
            # builder sets spec_token_indx = arange(num_actual_tokens), making
            # the index_select calls below identity copies. Skip them on this
            # steady-state decode hot path. The outputs alias the inputs here;
            # the downstream conv/recurrent kernels read them without mutating
            # in place, so the aliasing is safe.
            if non_spec_token_indx is None or non_spec_token_indx.numel() == 0:
                qkv_spec = qkv_proj_states
                g1_spec = g1
                beta_spec = beta
            else:
                qkv_spec = qkv_proj_states.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
            if non_spec_token_indx is not None and non_spec_token_indx.numel() > 0:
                qkv_ns = qkv_proj_states.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
            else:
                qkv_ns = g1_ns = beta_ns = None
        else:
            qkv_spec = g1_spec = beta_spec = None
            qkv_ns, g1_ns, beta_ns = qkv_proj_states, g1, beta

        # Spec-verify rows through FlashInfer's packed fused decode (conv +
        # T recurrent steps + gated RMSNorm, one launch, normed output). A
        # pure verify step writes the layer output in place and is done; a
        # mixed step (prefill + verify rows) merges below, with the non-spec
        # rows normed before the merge.
        fused_spec = use_spec and self._fi_spec_kda_decode is not None
        if fused_spec:
            assert self._fi_spec_kda_decode is not None
            assert spec_state_indices_tensor is not None
            assert spec_query_start_loc is not None
            assert num_accepted_tokens is not None
            assert qkv_spec is not None and g1_spec is not None
            assert beta_spec is not None
            assert conv_bias is None
            if self._fi_norm_weight is None:
                self._fi_norm_weight = self.o_norm.weight.float().contiguous()
            if self._fi_conv_weight is None:
                self._fi_conv_weight = (
                    conv_weights.view(3, self.local_projection_size, self.conv_size)
                    .transpose(1, 2)
                    .contiguous()
                )
            pure_spec = non_spec_token_indx is None or non_spec_token_indx.numel() == 0
            if pure_spec:
                fused_spec_out = core_attn_out[:, :num_actual_tokens]
                fused_spec_gate = g2[:num_actual_tokens]
            else:
                assert spec_token_indx is not None
                fused_spec_out = torch.empty(
                    (1, qkv_spec.size(0), self.local_num_heads, self.head_dim),
                    dtype=core_attn_out.dtype,
                    device=core_attn_out.device,
                )
                fused_spec_gate = g2.index_select(0, spec_token_indx)
            self._fi_spec_kda_decode(
                x=qkv_spec,
                weight=self._fi_conv_weight,
                conv_state=conv_state,
                raw_gate=g1_spec,
                raw_beta=beta_spec,
                A_log=self.A_log.view(-1),
                dt_bias=self.dt_bias,
                state_indices=spec_state_indices_tensor,
                query_start_loc=spec_query_start_loc,
                num_accepted_tokens=num_accepted_tokens,
                state=recurrent_state,
                output_gate=fused_spec_gate,
                norm_weight=self._fi_norm_weight,
                output=fused_spec_out,
            )
            if pure_spec:
                return

        # --- causal conv1d: spec (draft-verify) path ---
        if use_spec and not fused_spec:
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            conv_idx = spec_state_indices_tensor[:, 0][:num_spec_decodes]
            conv_mql = spec_state_indices_tensor.size(-1)
            qkv_spec = causal_conv1d_update(
                qkv_spec,
                conv_state,
                conv_weights,
                conv_bias,
                activation="silu",
                conv_state_indices=conv_idx,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=conv_mql,
            )
            q_spec, k_spec, v_spec = qkv_spec.split(self.local_projection_size, dim=-1)

        # --- causal conv1d: non-spec path (prefill or plain decode) ---
        raw_qkv_ns = qkv_ns
        q_ns = k_ns = v_ns = None
        # Mixed non-spec step without prefix-cache checkpoints: the leading
        # decode rows are peeled off the chunked prefill (vLLM orders decodes
        # first; the builder precomputes the prefill tail).
        peel_decodes = (
            attn_metadata_narrowed.num_prefills > 0
            and self._planar_prefill
            and not use_spec
            and attn_metadata_narrowed.num_decodes > 0
            and attn_metadata_narrowed.checkpoint is None
            and attn_metadata_narrowed.prefill_query_start_loc is not None
        )
        # With a fused decode backend they go through the same single-token
        # kernel as pure decode steps (conv + recurrence + gated RMSNorm,
        # normed output), before the prefill rows' conv.
        num_fused_peeled = 0
        if peel_decodes and self._fi_fused_kda_decode is not None and conv_bias is None:
            assert qkv_ns is not None and g1_ns is not None and beta_ns is not None
            assert non_spec_state_indices_tensor is not None
            num_fused_peeled = attn_metadata_narrowed.num_decode_tokens
            self._fused_decode(
                x=qkv_ns[:num_fused_peeled],
                conv_weights=conv_weights,
                conv_state=conv_state,
                raw_gate=g1_ns[:, :num_fused_peeled],
                raw_beta=beta_ns[:, :num_fused_peeled],
                state_indices=non_spec_state_indices_tensor[:num_fused_peeled],
                state=recurrent_state,
                output_gate=g2[:num_fused_peeled],
                output=core_attn_out[:, :num_fused_peeled],
            )
        if attn_metadata_narrowed.num_prefills > 0 and self._planar_prefill:
            assert qkv_ns is not None
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            conv_query_start_loc = non_spec_query_start_loc
            conv_state_indices = non_spec_state_indices_tensor
            conv_has_initial_state = has_initial_state
            if num_fused_peeled:
                assert attn_metadata_narrowed.prefill_query_start_loc is not None
                assert attn_metadata_narrowed.prefill_state_indices is not None
                assert attn_metadata_narrowed.prefill_has_initial_state is not None
                conv_query_start_loc = attn_metadata_narrowed.prefill_query_start_loc
                conv_state_indices = attn_metadata_narrowed.prefill_state_indices
                conv_has_initial_state = (
                    attn_metadata_narrowed.prefill_has_initial_state
                )
            # Planar [3, T, P] output: q, k and v are each dense for FlashKDA.
            q_ns, k_ns, v_ns = kda_causal_conv1d_planar(
                qkv_ns[num_fused_peeled:],
                conv_weights,
                conv_bias,
                conv_state,
                query_start_loc=conv_query_start_loc,
                cache_indices=conv_state_indices,
                has_initial_state=conv_has_initial_state,
            ).unbind(0)
        elif attn_metadata_narrowed.num_prefills > 0:
            assert qkv_ns is not None
            qkv_ns = causal_conv1d_fn(
                qkv_ns.transpose(0, 1),
                conv_weights,
                conv_bias,
                activation="silu",
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)
        elif attn_metadata_narrowed.num_decodes > 0:
            assert non_spec_state_indices_tensor is not None
            decode_conv_indices = non_spec_state_indices_tensor[
                : attn_metadata_narrowed.num_decodes
            ]
            qkv_ns = causal_conv1d_update(
                qkv_ns,
                conv_state,
                conv_weights,
                conv_bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
            )
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)

        def _rearr(x):
            return x.reshape(1, -1, self.local_num_heads, self.head_dim)

        # --- core attention: spec (draft-verify) path ---
        core_attn_out_spec = None
        # In a pure spec-verify step (no non-spec tokens) the recurrent kernel
        # can write straight into the layer output buffer, skipping the
        # fresh allocation + copy below. Mixed steps must scatter via
        # spec_token_indx, so they keep the kernel-managed output.
        spec_out = (
            core_attn_out[0, :num_actual_tokens].unsqueeze(0)
            if non_spec_token_indx is None or non_spec_token_indx.numel() == 0
            else None
        )
        if fused_spec:
            core_attn_out_spec = fused_spec_out
        elif use_spec:
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            assert spec_query_start_loc is not None
            # Gate computed inside the recurrent kernel (COMPUTE_GATE) from
            # raw g1 — replicates fused_kda_gate's arithmetic bit-for-bit and
            # skips its launch + fp32 [n, H, D] intermediate per layer.
            core_attn_out_spec, _ = fused_recurrent_kda(
                q=_rearr(q_spec),
                k=_rearr(k_spec),
                v=_rearr(v_spec),
                g=g1_spec,
                beta=beta_spec,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                out=spec_out,
                sigmoid_beta=True,
                a_log=self.A_log,
                g_bias=self.dt_bias,
                compute_gate=True,
                lower_bound=lower_bound,
            )

        # --- core attention: non-spec path (prefill or plain decode) ---
        core_attn_out_non_spec = None
        # Only the plain-decode recurrent kernel can write straight into the
        # layer output buffer; the chunked prefill kernel cannot, so this
        # stays None there and the merge copy below runs as before.
        ns_out = None
        if attn_metadata_narrowed.num_prefills > 0:
            assert q_ns is not None and k_ns is not None and v_ns is not None
            assert g1_ns is not None and beta_ns is not None
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            prefill_query_start_loc = non_spec_query_start_loc
            prefill_state_indices = non_spec_state_indices_tensor
            prefill_has_initial_state = has_initial_state
            num_peeled = 0
            if num_fused_peeled:
                # Decode rows already done by the fused decode kernel; the
                # conv above only produced the prefill rows.
                num_peeled = num_fused_peeled
                prefill_query_start_loc = attn_metadata_narrowed.prefill_query_start_loc
                prefill_state_indices = attn_metadata_narrowed.prefill_state_indices
                prefill_has_initial_state = (
                    attn_metadata_narrowed.prefill_has_initial_state
                )
                assert prefill_query_start_loc is not None
                assert prefill_state_indices is not None
                assert prefill_has_initial_state is not None
                g1_ns, beta_ns = g1_ns[:, num_peeled:], beta_ns[:, num_peeled:]
            elif peel_decodes:
                # Mixed step: the leading decode rows go through the
                # recurrent kernel in place on the state pool, so FlashKDA,
                # gather and scatter only see the prefill rows.
                num_peeled = attn_metadata_narrowed.num_decode_tokens
                fused_recurrent_kda(
                    q=_rearr(q_ns[:num_peeled]),
                    k=_rearr(k_ns[:num_peeled]),
                    v=_rearr(v_ns[:num_peeled]),
                    g=g1_ns[:, :num_peeled],
                    beta=beta_ns[:, :num_peeled],
                    initial_state=recurrent_state,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata_narrowed.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    out=core_attn_out[:, :num_peeled],
                    sigmoid_beta=True,
                    a_log=self.A_log,
                    g_bias=self.dt_bias,
                    compute_gate=True,
                    lower_bound=lower_bound,
                )
                prefill_query_start_loc = attn_metadata_narrowed.prefill_query_start_loc
                prefill_state_indices = attn_metadata_narrowed.prefill_state_indices
                prefill_has_initial_state = (
                    attn_metadata_narrowed.prefill_has_initial_state
                )
                assert prefill_state_indices is not None
                assert prefill_has_initial_state is not None
                q_ns, k_ns, v_ns = (
                    q_ns[num_peeled:],
                    k_ns[num_peeled:],
                    v_ns[num_peeled:],
                )
                g1_ns, beta_ns = g1_ns[:, num_peeled:], beta_ns[:, num_peeled:]
            if (
                self._fi_prefill is not None
                and attn_metadata_narrowed.checkpoint is None
            ):
                # FlashInfer updates the state in place at the slot indices
                # (no gather / scatter); trailing padding rows are excluded.
                # With spec-verify rows in the step the prefill rows are not
                # contiguous in the output, so write a buffer that the merge
                # below scatters. (Falling back to FlashKDA here would ask the
                # locked workspace for buffers the warmup never reserved:
                # job 803405, "Workspace is locked ... 6960 MB".)
                if use_spec:
                    assert q_ns is not None
                    ns_out = torch.empty(
                        (1, q_ns.size(0), self.local_num_heads, self.head_dim),
                        dtype=core_attn_out.dtype,
                        device=core_attn_out.device,
                    )
                else:
                    ns_out = core_attn_out[:, num_peeled:num_actual_tokens]
                self._fi_prefill(
                    q=_rearr(q_ns),
                    k=_rearr(k_ns),
                    v=_rearr(v_ns),
                    g=g1_ns,
                    beta=beta_ns,
                    A_log=self.A_log.view(-1),
                    dt_bias=self.dt_bias.view(-1, self.head_dim),
                    state=recurrent_state,
                    state_indices=prefill_state_indices,
                    has_initial_state=prefill_has_initial_state,
                    cu_seqlens=prefill_query_start_loc,
                    num_seqs=attn_metadata_narrowed.num_prefills
                    + (0 if num_peeled else attn_metadata_narrowed.num_decodes),
                    out=ns_out,
                )
                core_attn_out_non_spec = ns_out
            else:
                initial_state = gather_initial_states(
                    recurrent_state, prefill_state_indices, prefill_has_initial_state
                )
                if self.kda_prefill_backend == "flashkda":
                    # Non-spec step: write straight into the layer output buffer
                    # (dense token order, no merge copy). Step with spec-decode
                    # tokens: write to the workspace buffer and scatter below.
                    ns_out = (
                        None
                        if use_spec
                        else core_attn_out[:, num_peeled:num_actual_tokens]
                    )
                    core_attn_out_non_spec, last_recurrent_state = (
                        self._flashkda_prefill(
                            q=_rearr(q_ns),
                            k=_rearr(k_ns),
                            v=_rearr(v_ns),
                            g=g1_ns,
                            beta=beta_ns,
                            initial_state=initial_state,
                            cu_seqlens=prefill_query_start_loc,
                            out=ns_out,
                            checkpoint=attn_metadata_narrowed.checkpoint,
                            raw_qkv=raw_qkv_ns,
                            conv_state=conv_state,
                            recurrent_state=recurrent_state,
                        )
                    )
                else:
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = chunk_kda_with_fused_gate(
                        q=_rearr(q_ns),
                        k=_rearr(k_ns),
                        v=_rearr(v_ns),
                        raw_g=g1_ns,
                        # Chunk path wants the pre-sigmoided fp32 beta (its
                        # kernels don't sigmoid); beta_ns is raw bf16 from forward.
                        beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),
                        A_log=self.A_log,
                        g_bias=self.dt_bias,
                        initial_state=initial_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=non_spec_query_start_loc,
                        safe_gate=safe_gate,
                        lower_bound=lower_bound,
                    )
                # Init cache
                scatter_states(
                    recurrent_state,
                    last_recurrent_state,
                    prefill_state_indices,
                )
        elif attn_metadata_narrowed.num_decodes > 0:
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
            # Plain decode step (no spec tokens): token order is dense, so the
            # kernel can write straight into the layer output buffer. A mixed
            # step scatters non-spec output via non_spec_token_indx instead.
            # Gate computed in-kernel (COMPUTE_GATE), beta sigmoided in-kernel.
            if not use_spec:
                ns_out = spec_out
            core_attn_out_non_spec, _ = fused_recurrent_kda(
                q=_rearr(q_ns),
                k=_rearr(k_ns),
                v=_rearr(v_ns),
                g=g1_ns,
                beta=beta_ns,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata_narrowed.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                out=ns_out,
                sigmoid_beta=True,
                a_log=self.A_log,
                g_bias=self.dt_bias,
                compute_gate=True,
                lower_bound=lower_bound,
            )

        # --- merge spec / non-spec outputs back into token order ---
        if use_spec and core_attn_out_non_spec is not None:
            assert core_attn_out_spec is not None
            if fused_spec:
                # Spec rows come out of the fused kernel normed; norm the
                # non-spec (prefill) rows here so the whole output is normed.
                assert non_spec_token_indx is not None
                core_attn_out_non_spec = self.o_norm(
                    core_attn_out_non_spec,
                    g2.index_select(0, non_spec_token_indx),
                )
            core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            core_attn_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        elif use_spec:
            assert core_attn_out_spec is not None
            if spec_out is None:
                core_attn_out[0, :num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            assert core_attn_out_non_spec is not None
            if ns_out is None:
                core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                    0, :num_actual_tokens
                ]

        if self._output_normed and not fused_spec:
            # forward() skips o_norm for the fused decode backends, so the
            # Triton paths (prefill / mixed steps) apply it here, in place.
            # (With fused_spec the spec rows are normed by the kernel and the
            # non-spec rows were normed before the merge above.)
            # Rows from the fused decode kernel are already normed.
            live_out = core_attn_out[:, num_fused_peeled:num_actual_tokens]
            normed = self.o_norm(live_out, g2[num_fused_peeled:num_actual_tokens])
            if normed.data_ptr() != live_out.data_ptr():
                live_out.copy_(normed)
