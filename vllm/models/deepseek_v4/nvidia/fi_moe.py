# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer ``moe_ep`` expert module for DeepSeek V4."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MegaMoEExperts,
    _use_sequence_parallel,
)
from vllm.utils.flashinfer_moe_ep import (
    NVFP4_CKPT_MEGAKERNELS,
    SM107_MEGAKERNELS,
    build_fi_mega_layer,
    ensure_fi_moe_ep_runtime,
    make_fi_moe_ep_bootstrap,
    mega_moe_weight_pack_from_params,
    resolve_fi_megakernel,
)

if TYPE_CHECKING:
    from flashinfer.moe_ep import MoEEpMegaLayer

    from vllm.config import VllmConfig
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MLP

logger = init_logger(__name__)

_MOE_SKIP_PADDING: bool | None = None
_SM107_LOGGED: set[tuple] = set()

# Smallest capacity profile of the SM107 workspace ladder.
_SM107_MIN_CAPACITY = 128


def sm107_max_tokens_per_rank(max_num_tokens: int, sp_size: int = 1) -> int:
    """Largest token count one rank can hand the SM107 megakernel: the step's
    ``max_num_batched_tokens`` split over the sequence-parallel group
    (``ceil(mnbt / sp)``; SP shards are equal, padded to a multiple of sp)."""
    if max_num_tokens <= 0:
        raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}")
    sp = max(int(sp_size), 1)
    return -(-int(max_num_tokens) // sp)


def sm107_capacity_profiles(max_num_tokens: int, sp_size: int = 1) -> list[int]:
    """Workspace capacity ladder (tokens per rank) for the SM107 megakernel.

    Powers of two from 128 below the per-rank maximum ``cap = ceil(
    max_num_tokens / sp)``, plus ``cap``. The ladder stops at ``cap``: under
    sequence parallelism (e.g. TEP2 with mnbt 32768) a ``max_num_tokens``
    profile could never be selected and would only waste symmetric memory.
    """
    cap = sm107_max_tokens_per_rank(max_num_tokens, sp_size)
    caps: set[int] = {cap}
    profile = _SM107_MIN_CAPACITY
    while profile < cap:
        caps.add(profile)
        profile *= 2
    return sorted(caps)


# SM budget of the SM107 megakernel in overlap mode: the persistent kernel is
# enqueued first and the shared MLP runs concurrently on the remaining SMs.
# 196 of VR200's 212 SMs was the best single budget for DSv4.1 at 1260-2520
# tokens/rank (true EP4).
_SM107_OVERLAP_SM_BUDGET = 196


def sm107_max_sm_count(spec: str, shared_mode: str) -> int | None:
    """``VLLM_FI_MEGA_MOE_MAX_SM_COUNT`` -> flashinfer ``max_sm_count``.

    ``auto`` (also an empty value): ``_SM107_OVERLAP_SM_BUDGET`` in overlap
    mode, else all SMs (None). ``0`` / ``none``: all SMs. ``<n>``: n SMs."""
    spec = (spec or "").strip().lower() or "auto"
    if spec == "auto":
        return _SM107_OVERLAP_SM_BUDGET if shared_mode == "overlap" else None
    if spec in ("0", "none"):
        return None
    try:
        value = int(spec)
    except ValueError:
        raise ValueError(
            "VLLM_FI_MEGA_MOE_MAX_SM_COUNT must be 'auto', 'none' or an SM "
            f"count, got {spec!r}"
        ) from None
    if value < 0:
        raise ValueError(f"max SM count must be >= 0, got {value}")
    return value or None


def resolve_sm107_shared_mode(
    requested: str, *, replicated_shared: bool, has_side_stream: bool
) -> tuple[str, str | None]:
    """Effective ``VLLM_FI_MEGA_MOE_SHARED`` mode and the reason for a fallback.

    ``overlap`` needs a replicated shared MLP (sequence parallel or TP 1, as
    the DeepGEMM fusion): a TP-sharded one would all-reduce on the side stream
    concurrently with the NVSHMEM megakernel. A fallback runs ``separate``,
    with no megakernel SM budget unless ``VLLM_FI_MEGA_MOE_MAX_SM_COUNT`` is
    an explicit count."""
    if requested != "overlap":
        return requested, None
    if not replicated_shared:
        return "separate", "the shared MLP is tensor-parallel (no SP, TP > 1)"
    if not has_side_stream:
        return "separate", "no CUDA side stream"
    return "overlap", None


# Side stream of VLLM_FI_MEGA_MOE_SHARED=overlap, private to it (not the global
# aux_stream()): the shared MLP output is allocated here and read by the main
# stream after the join without record_stream (which would hold the block until
# the end of each CUDA-graph capture). Reusing that block is safe because every
# later use of this stream first waits on a fork event recorded on the main
# stream after the consumer.
_SM107_SHARED_STREAM: torch.cuda.Stream | None = None


def sm107_shared_stream() -> torch.cuda.Stream | None:
    global _SM107_SHARED_STREAM
    if _SM107_SHARED_STREAM is None and torch.cuda.is_available():
        _SM107_SHARED_STREAM = torch.cuda.Stream()
    return _SM107_SHARED_STREAM


def sm107_compute_with_shared(
    kernel: Any,
    workspace: Any,
    transformed: Any,
    hidden_states: torch.Tensor,
    shared_experts: nn.Module,
    events: tuple[torch.cuda.Event, torch.cuda.Event],
    stream: torch.cuda.Stream,
) -> torch.Tensor:
    """Staged megakernel on the current stream + shared MLP on ``stream``.

    The fork is recorded after the input staging and the megakernel is
    enqueued before the shared MLP, so the persistent kernel takes its
    (VLLM_FI_MEGA_MOE_MAX_SM_COUNT-capped) SMs first and the MLP runs on the
    rest (forking before the gate, TensorRT-LLM's order, loses the SM-budget
    gain at true EP4). Same schedule in every graph mode: the fork and the join
    complete inside one breakable-capture segment (neither the megakernel nor
    the shared MLP has an eager break; a break in between would fail the
    capture with an unjoined-stream error), like DSv4 attention's
    execute_in_parallel."""
    logger.info_once(
        "FlashInfer SM107 MegaMoE: shared expert MLP overlapped on a side stream."
    )
    fork, join = events
    fork.record()
    y = kernel.compute(workspace, transformed, output=None)
    with torch.cuda.stream(stream):
        fork.wait()
        shared = shared_experts(hidden_states)
        join.record()
    join.wait()
    y += shared
    return y


def ckpt_uses_nvfp4_experts(vllm_config: VllmConfig) -> bool:
    """True when the loaded checkpoint quantizes experts with modelopt NVFP4
    (e2m1 + fp8-e4m3 per-16 block scales + per-tensor weight_scale_2), i.e.
    the recipe the nvfp4_cutedsl prequantized-weights path consumes verbatim.

    DeepSeek-V4 NVFP4-expert checkpoints declare ``moe_quant_algo: NVFP4`` in
    their ``quantization_config``, which DeepseekV4FP8Config surfaces as
    ``moe_quant_algo``.
    """
    return getattr(vllm_config.quant_config, "moe_quant_algo", "") == "NVFP4"


def resolve_mega_moe_is_padding(num_tokens: int) -> torch.Tensor | None:
    from vllm.forward_context import get_forward_context, is_forward_context_available

    global _MOE_SKIP_PADDING
    if _MOE_SKIP_PADDING is None:
        import vllm.envs as envs

        _MOE_SKIP_PADDING = bool(envs.VLLM_MOE_SKIP_PADDING)
    if not _MOE_SKIP_PADDING or not is_forward_context_available():
        return None
    is_padding = get_forward_context().is_padding
    if is_padding is None:
        return None
    return is_padding[:num_tokens]


_FI_PAD_MASK: bool | None = None


def fi_stages_route_padding_mask() -> bool:
    """flashinfer's MoEEpTensors takes ``route_padding_mask`` and masks those
    rows inside its one-launch SM107 route staging, so no separate masking
    kernel runs before the MegaMoE call."""
    global _FI_PAD_MASK
    if _FI_PAD_MASK is None:
        try:
            import dataclasses

            from flashinfer.moe_ep import MoEEpTensors

            names = {f.name for f in dataclasses.fields(MoEEpTensors)}
            _FI_PAD_MASK = "route_padding_mask" in names
        except Exception:
            _FI_PAD_MASK = False
    return _FI_PAD_MASK


def apply_mega_moe_routing_preprocess(
    topk_ids: torch.Tensor,
    *,
    is_padding: torch.Tensor | None = None,
) -> torch.Tensor:
    """Padding-only routing preprocess (EPLB hooks go here later)."""
    if is_padding is not None:
        topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
    return topk_ids


def nvfp4_prequant_pack_and_alphas(
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_scale_2: torch.Tensor,  # (E_local, 2) fp32: [:,0]=gate(w1), [:,1]=up(w3)
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,  # (E_local,) fp32
    *,
    intermediate_size: int,
):
    """NVFP4 checkpoint params -> (MoEWeightPack, fc1_alpha, fc2_alpha).

    Activation quantization is fully dynamic, so the checkpoint's static
    ``input_scale`` drops out and each GEMM's epilogue alpha reduces to the
    weight's per-tensor ``weight_scale_2``.

    fc1_alpha is one scalar per expert, but gate (w1) and up (w3) carry their
    own ``weight_scale_2``. When they differ, the ratio is folded into the up
    half's e4m3 block scales — only if the fold round-trips exactly, since a
    lossy rescale would silently change the model.
    """
    from flashinfer.moe_ep import MoEWeightPack

    inter = intermediate_size
    gate_s2 = w13_weight_scale_2[:, 0].float()
    up_s2 = w13_weight_scale_2[:, 1].float()
    if (gate_s2 <= 0).any() or (up_s2 <= 0).any() or (w2_weight_scale_2 <= 0).any():
        raise ValueError(
            "nvfp4 prequant: non-positive weight_scale_2 loaded — checkpoint "
            "scale tensors missing or loader routed them wrong."
        )

    w13_scale = w13_weight_scale
    if not torch.equal(gate_s2, up_s2):
        ratio = up_s2 / gate_s2  # (E_local,)
        up_sf = w13_scale[:, inter:, :].float()
        folded = up_sf * ratio[:, None, None]
        folded_e4m3 = folded.to(torch.float8_e4m3fn)
        if not torch.equal(folded_e4m3.float(), folded):
            raise ValueError(
                "NVFP4 MoE cannot merge the gate and up weight_scale_2 values "
                "because their ratio is not exactly representable in "
                "float8_e4m3fn. Use a checkpoint quantized with a shared "
                "gate/up weight_scale_2."
            )
        w13_scale = w13_scale.clone()
        w13_scale[:, inter:, :] = folded_e4m3

    fc1_alpha = gate_s2.clone().contiguous()
    fc2_alpha = w2_weight_scale_2.float().clone().contiguous()
    pack = MoEWeightPack(
        w13=w13_weight,
        w2=w2_weight,
        w13_scale=w13_scale,
        w2_scale=w2_weight_scale,
    )
    return pack, fc1_alpha, fc2_alpha


def sm107_nvfp4_epilogue_scalars(
    gate_s2: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,
    w2_input_scale: torch.Tensor,
    *,
    input_norm_const: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-local-expert ``(fc1_alpha, fc2_alpha, fc1_norm_const)`` for SM107.

    TRT-LLM convention: the in-kernel FC1-output -> FC2-input NVFP4 requant is
    normalised by the checkpoint's calibrated ``fc1_norm_const = 1 /
    w2.input_scale`` and FC2's accumulator scale absorbs it, ``fc2_alpha =
    w2.weight_scale_2 * w2.input_scale``. With the default unit norm const the
    FC1-output block scales (amax/6) of small-activation layers underflow E4M3
    (layer-6 ``w2.input_scale`` is ~1e-4). ``fc1_alpha`` divides out the
    global scale the bf16 activations were quantized with
    (``input_norm_const``). ``gate_s2`` is the gate ``weight_scale_2`` after
    the up/gate ratio fold.
    """
    s13 = gate_s2.float().reshape(-1)
    s2 = w2_weight_scale_2.float().reshape(-1)
    in2 = w2_input_scale.float().reshape(-1)
    if s13.shape != s2.shape or s2.shape != in2.shape:
        raise ValueError(
            "per-expert NVFP4 scalars must all have shape [E_local]: got "
            f"{tuple(s13.shape)}, {tuple(s2.shape)}, {tuple(in2.shape)}"
        )
    if not bool(torch.isfinite(in2).all()) or bool((in2 <= 0).any()):
        raise ValueError(
            "NVFP4 MegaMoE needs positive, finite w2.input_scale for every "
            "local expert (fc1_norm_const = 1 / w2.input_scale)."
        )
    if input_norm_const <= 0 or input_norm_const != input_norm_const:
        raise ValueError(f"input_norm_const must be positive, got {input_norm_const}")
    fc1_alpha = (s13 / float(input_norm_const)).contiguous()
    fc1_norm_const = (1.0 / in2).contiguous()
    fc2_alpha = (s2 * in2).contiguous()
    return fc1_alpha, fc2_alpha, fc1_norm_const


class DeepseekV4MegaMoEExpertsFI(DeepseekV4MegaMoEExperts):
    """Same weight layout/loader as the native mega experts, FI compute path.

    SM100/SM103: one ``max_num_tokens`` workspace and the backend's single
    megakernel (``nvfp4_cutedsl`` / ``deep_gemm_mega``).

    SM107 (Rubin): flashinfer's ``sm107_nvfp4_nvfp4_bf16_cutedsl``
    RubinInferenceMegaMoE on the NVFP4 expert checkpoint (per-expert
    ``fc1_alpha`` / ``fc2_alpha`` / ``fc1_norm_const``), with one workspace per
    capacity profile so every step runs the knobs tuned for its size.
    Workspaces are pooled across layers by flashinfer.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        activation_clamp: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(vllm_config, **kwargs)
        self._vllm_config = vllm_config
        self._activation_clamp = activation_clamp
        self._mega_layer: MoEEpMegaLayer | None = None
        self._fast_ctx: tuple[Any, Any, Any, int, bool] | None = None
        self._epilogue_alphas: tuple[torch.Tensor, ...] | None = None
        self._nvfp4_prequant = ckpt_uses_nvfp4_experts(vllm_config)
        self._megakernel = resolve_fi_megakernel(
            vllm_config.kernel_config.moe_backend,
            nvfp4_checkpoint=self._nvfp4_prequant,
        )
        self._is_sm107 = self._megakernel in SM107_MEGAKERNELS
        import vllm.envs as envs

        # SM107 capacity profiles: capacity -> (kernel, backend workspace,
        # transformed weights). Built in finalize_weights.
        self._profiles: dict[int, tuple[Any, Any, Any]] = {}
        self._profile_caps: list[int] = []
        self._ws_handles: list[Any] = []
        # Shared expert next to the SM107 megakernel (VLLM_FI_MEGA_MOE_SHARED):
        # separate | overlap (side stream).
        self._shared_mode = (
            envs.VLLM_FI_MEGA_MOE_SHARED if self._is_sm107 else "separate"
        )
        self._shared_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        parallel_config = vllm_config.parallel_config
        self._dp_size = parallel_config.data_parallel_size
        use_sp = _use_sequence_parallel(vllm_config)
        tp_size = parallel_config.tensor_parallel_size
        self._sp_size = tp_size if use_sp else 1
        replicated_shared = use_sp or tp_size == 1
        self._shared_stream: torch.cuda.Stream | None = (
            sm107_shared_stream()
            if self._shared_mode == "overlap" and replicated_shared
            else None
        )
        self._shared_mode, fallback = resolve_sm107_shared_mode(
            self._shared_mode,
            replicated_shared=replicated_shared,
            has_side_stream=self._shared_stream is not None,
        )
        if fallback is not None:
            logger.warning_once(
                "VLLM_FI_MEGA_MOE_SHARED=overlap runs as separate, with no "
                "megakernel SM budget unless VLLM_FI_MEGA_MOE_MAX_SM_COUNT is "
                "an explicit count. Reason: %s.",
                fallback,
            )
        if self._nvfp4_prequant:
            if self._megakernel not in NVFP4_CKPT_MEGAKERNELS:
                raise ValueError(
                    "NVFP4-quantized expert checkpoint requires "
                    "moe_backend=flashinfer_moe_ep_mega_cutedsl, got a "
                    f"backend using megakernel {self._megakernel!r} "
                    "(deep_gemm consumes the MXFP4 checkpoint instead)."
                )
            self._realloc_nvfp4_params()

    # ----------------------------------------------------------- shared expert
    @property
    def overlaps_shared_experts(self) -> bool:
        """VLLM_FI_MEGA_MOE_SHARED=overlap on SM107: the MoE block passes
        ``shared_experts`` to forward(), which runs it on a side stream
        concurrently with the (SM-budgeted) megakernel, in every graph mode."""
        return self._is_sm107 and self._shared_mode == "overlap"

    def _realloc_nvfp4_params(self) -> None:
        """Swap the mx-recipe scale params for the NVFP4 checkpoint's:
        fp8-e4m3 per-16 block scales plus the per-tensor second-level
        scales (weight_scale_2, input_scale) modelopt exports."""
        n_e = self.num_local_experts
        inter = self.intermediate_size
        hidden = self.hidden_size
        attrs = {"weight_loader": self.weight_loader}

        def _param(shape: tuple, dtype: torch.dtype) -> nn.Parameter:
            p = nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(p, attrs)
            return p

        self.w13_weight_scale = _param(
            (n_e, 2 * inter, hidden // 16), torch.float8_e4m3fn
        )
        self.w13_weight_scale.quant_method = "block"
        self.w2_weight_scale = _param((n_e, hidden, inter // 16), torch.float8_e4m3fn)
        self.w2_weight_scale.quant_method = "block"
        # (E, 2): column 0 = gate (w1), column 1 = up (w3).
        self.w13_weight_scale_2 = _param((n_e, 2), torch.float32)
        self.w2_weight_scale_2 = _param((n_e,), torch.float32)
        # SM100 ignores the input scales (dynamic activation quant). SM107
        # uses them: max(w13.input_scale) sets the activation quant grid and
        # w2.input_scale the FC1-output requant norm (fc1_norm_const).
        self.w13_input_scale = _param((n_e, 2), torch.float32)
        self.w2_input_scale = _param((n_e,), torch.float32)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        # NVFP4 checkpoint second-level scalars route here; everything
        # else (packed weights, block scales) matches the base layout.
        if "weight_scale_2" in weight_name or "input_scale" in weight_name:
            local_expert_ids = self._map_global_expert_id(expert_id)
            if not local_expert_ids:
                return False if return_success else None
            value = loaded_weight.reshape(()).to(torch.float32)
            for local_expert_id in local_expert_ids:
                if shard_id in ("w1", "w3"):
                    if "w13_" not in weight_name:
                        return False if return_success else None
                    param.data[local_expert_id, 0 if shard_id == "w1" else 1] = value
                elif shard_id == "w2":
                    if "w2_" not in weight_name:
                        return False if return_success else None
                    param.data[local_expert_id] = value
                else:
                    raise ValueError(f"Unsupported expert shard id: {shard_id}")
            return True if return_success else None
        return super().weight_loader(
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success,
        )

    # ------------------------------------------------------------------ SM107
    @staticmethod
    def _ep_global_max(local: torch.Tensor) -> torch.Tensor:
        """max(local) MAX-reduced over the EP group (fp32 [1] tensor)."""
        amax = local.float().max().reshape(1).clone()
        from vllm.distributed import get_ep_group

        ep = get_ep_group()
        if ep.world_size > 1:
            import torch.distributed as dist

            dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=ep.device_group)
        return amax

    def _sm107_input_norm_const(self) -> float:
        """Global scale for the in-kernel bf16 -> NVFP4 activation quant.

        Tokens are quantized on their source rank and consumed on the
        expert's rank, so the value must be identical on every EP rank:
        1 / max(w13.input_scale) over all experts (MAX all-reduce over the
        EP group of the local maxima), as the FusedMoE NVFP4 backends use.
        """
        amax = self._ep_global_max(self.w13_input_scale.data)
        # fp32 reciprocal, exactly as the FusedMoE NVFP4 backends form
        # a1_gscale = 1.0 / a13_scale on an fp32 tensor.
        return float((1.0 / amax).item())

    def _sm107_fc2_input_scale(self) -> torch.Tensor:
        """Per-local-expert FC2-input (FC1-output requant) scale.

        One value, max(w2.input_scale) over all experts of the layer (EP
        all-reduce), i.e. the grid every FusedMoE NVFP4 backend in vLLM uses
        (amax_for_moe_activation_quant), so the numerics do not depend on the
        EP size.
        """
        w2_in = self.w2_input_scale.data.float()
        return self._ep_global_max(w2_in).expand_as(w2_in).contiguous()

    def _sm107_build(self) -> None:
        import vllm.envs as envs

        # resolve_fi_megakernel admits only the NVFP4 checkpoint on SM107.
        assert self._nvfp4_prequant
        weights, gate_s2, _ = nvfp4_prequant_pack_and_alphas(
            self.w13_weight.data,
            self.w13_weight_scale.data,
            self.w13_weight_scale_2.data,
            self.w2_weight.data,
            self.w2_weight_scale.data,
            self.w2_weight_scale_2.data,
            intermediate_size=self.intermediate_size,
        )
        input_norm_const = self._sm107_input_norm_const()
        self._epilogue_alphas = sm107_nvfp4_epilogue_scalars(
            gate_s2,
            self.w2_weight_scale_2.data,
            self._sm107_fc2_input_scale(),
            input_norm_const=input_norm_const,
        )
        sm107_extra: dict[str, Any] = {}
        max_sm = sm107_max_sm_count(
            envs.VLLM_FI_MEGA_MOE_MAX_SM_COUNT, self._shared_mode
        )
        if max_sm is not None:
            sm107_extra["max_sm_count"] = max_sm
        self._mega_layer = build_fi_mega_layer(
            make_fi_moe_ep_bootstrap(),
            vllm_config=self._vllm_config,
            num_experts=self.num_experts,
            max_tokens_per_rank=self.max_num_tokens,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            top_k=self.top_k,
            activation_clamp=self._activation_clamp,
            weights=weights,
            megakernel=self._megakernel,
            input_norm_const=input_norm_const,
            sm107_extra=sm107_extra,
        )
        del weights
        layer = self._mega_layer
        transformed = layer._get_transformed_weights()
        caps = sm107_capacity_profiles(self.max_num_tokens, sp_size=self._sp_size)
        # Collective (symmetric heap) allocations: every EP rank builds the
        # same profiles in the same order (layers finalize in model order).
        for cap in caps:
            handle = layer.create_workspace(cap)
            self._ws_handles.append(handle)
            self._profiles[cap] = (
                layer._kernel,
                handle._backend_workspace,
                transformed,
            )
        self._profile_caps = caps
        log_key = (self._megakernel, self.num_experts, self.top_k, tuple(caps))
        if log_key not in _SM107_LOGGED:  # once per distinct layer config
            _SM107_LOGGED.add(log_key)
            logger.info(
                "FlashInfer SM107 MegaMoE: megakernel %s, %d of %d experts "
                "local, I %d, top-%d, clamp %s, input_norm_const %.6g, "
                "capacity profiles %s, shared expert %s, megakernel SM "
                "budget %s.",
                self._megakernel,
                self.num_local_experts,
                self.num_experts,
                self.intermediate_size,
                self.top_k,
                self._activation_clamp,
                input_norm_const,
                caps,
                self._shared_mode,
                sm107_extra.get("max_sm_count", "all"),
            )

    def _rank_consistent_num_tokens(self, num_tokens: int) -> int:
        """Token count every EP rank agrees on for this step.

        The megakernel's dispatch/combine address peer workspaces at the
        same symmetric offsets, so all ranks must launch the same capacity
        profile. SP ranks hold equal shards; DP ranks may differ (no DP
        padding in eager steps), so use the step's max over DP ranks.
        """
        if self._dp_size <= 1:
            return num_tokens
        from vllm.forward_context import (
            RANK_LOCAL_BATCH,
            get_forward_context,
            is_forward_context_available,
        )

        context = get_forward_context() if is_forward_context_available() else None
        if (getattr(context, "additional_kwargs", None) or {}).get(RANK_LOCAL_BATCH):
            # A rank-local batch (DSv4.1 bounded-replay seam graph, captured per
            # replay size and replayed at a size each rank picks for itself):
            # the profile is baked into the graph, so only the largest one is
            # the same on every rank whatever sizes the ranks replay.
            logger.info_once(
                "FlashInfer SM107 MegaMoE: rank-local batches (bounded-replay "
                "seam graphs) use the largest capacity profile."
            )
            return sm107_max_tokens_per_rank(self.max_num_tokens, self._sp_size)
        dp_metadata = context.dp_metadata if context is not None else None
        if dp_metadata is None:
            # No cross-rank token counts: fall back to the largest profile,
            # which every rank selects identically. That is the per-rank cap
            # ceil(mnbt / sp), not mnbt: the ladder stops there.
            return sm107_max_tokens_per_rank(self.max_num_tokens, self._sp_size)
        n_max = int(dp_metadata.num_tokens_across_dp_cpu.max())
        if self._sp_size > 1:
            n_max = -(-n_max // self._sp_size)
        return max(num_tokens, n_max)

    def _select_profile(self, num_tokens: int) -> tuple[Any, Any, Any]:
        n = self._rank_consistent_num_tokens(num_tokens)
        for cap in self._profile_caps:
            if cap >= n:
                return self._profiles[cap]
        raise ValueError(
            f"DeepSeek V4 MegaMoE got {n} tokens per rank, but the largest "
            f"workspace profile holds {self._profile_caps[-1]}."
        )

    # ---------------------------------------------------------------- finalize
    def finalize_weights(self, shared_experts: DeepseekV4MLP | None = None) -> None:
        # The FlashInfer megakernel has no shared-expert fusion; the caller
        # runs the shared MLP (serial, or overlapped on a side stream with
        # VLLM_FI_MEGA_MOE_SHARED=overlap on SM107).
        if self._mega_layer is not None:
            return
        if self.w13_weight is None:
            return

        self._check_runtime_supported()
        ensure_fi_moe_ep_runtime(self._vllm_config)

        if self._is_sm107:
            self._sm107_build()
        else:
            if self._nvfp4_prequant:
                # NVFP4 checkpoint: hand the packed weights + both scale
                # planes straight to the backend (no dequant->requant);
                # per-expert globals become fc1/fc2 epilogue alphas staged
                # at every forward via MoEEpTensors.
                weights, fc1_alpha, fc2_alpha = nvfp4_prequant_pack_and_alphas(
                    self.w13_weight.data,
                    self.w13_weight_scale.data,
                    self.w13_weight_scale_2.data,
                    self.w2_weight.data,
                    self.w2_weight_scale.data,
                    self.w2_weight_scale_2.data,
                    intermediate_size=self.intermediate_size,
                )
                self._epilogue_alphas = (fc1_alpha, fc2_alpha)
            else:
                weights = mega_moe_weight_pack_from_params(
                    self.w13_weight,
                    self.w13_weight_scale,
                    self.w2_weight,
                    self.w2_weight_scale,
                    megakernel=self._megakernel,
                )
            self._mega_layer = build_fi_mega_layer(
                make_fi_moe_ep_bootstrap(),
                vllm_config=self._vllm_config,
                num_experts=self.num_experts,
                max_tokens_per_rank=self.max_num_tokens,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.top_k,
                activation_clamp=self._activation_clamp,
                weights=weights,
            )
            del weights
            # Allocate (or attach to) the pooled workspace before first
            # forward so warmup/capture never hits the lazy path.
            self._mega_layer._ensure_workspace()
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None
        if self._nvfp4_prequant:
            self.w13_weight_scale_2 = None
            self.w2_weight_scale_2 = None
            self.w13_input_scale = None
            self.w2_input_scale = None

    # EPLB is rejected for these backends in validate_fi_moe_ep_config, so
    # nothing constructs an EplbState and none of these run. They are kept
    # as explicit errors rather than deleted because inheriting the native
    # implementations would be worse: set_eplb_state would record a map
    # this forward path never consults, silently splitting routing from
    # weights, and get_expert_weights reads transformed-weight attributes
    # the flashinfer path releases after preprocess.
    _EPLB_UNSUPPORTED = (
        "EPLB is not supported with the flashinfer moe_ep backends; "
        "validate_fi_moe_ep_config should have rejected this "
        "configuration at startup."
    )

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        raise NotImplementedError(self._EPLB_UNSUPPORTED)

    def get_expert_weights(self) -> list[torch.Tensor]:
        raise NotImplementedError(self._EPLB_UNSUPPORTED)

    def update_expert_map(self) -> None:
        raise NotImplementedError(self._EPLB_UNSUPPORTED)

    def _scalar_kwargs(self) -> dict[str, torch.Tensor | None]:
        alphas = self._epilogue_alphas
        if alphas is None:
            return {}
        out: dict[str, torch.Tensor | None] = {
            "fc1_alpha": alphas[0],
            "fc2_alpha": alphas[1],
        }
        if len(alphas) > 2:
            out["fc1_norm_const"] = alphas[2]
        return out

    def _compute_with_overlapped_shared(
        self,
        kernel: Any,
        workspace: Any,
        transformed: Any,
        hidden_states: torch.Tensor,
        shared_experts: nn.Module,
    ) -> torch.Tensor:
        assert self._shared_stream is not None
        if self._shared_events is None:
            self._shared_events = (torch.cuda.Event(), torch.cuda.Event())
        return sm107_compute_with_shared(
            kernel,
            workspace,
            transformed,
            hidden_states,
            shared_experts,
            self._shared_events,
            self._shared_stream,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        # fast_math is a native deep_gemm knob; the FI kernels have no
        # equivalent toggle, so it is accepted for signature parity only.
        # shared_experts (SM107, VLLM_FI_MEGA_MOE_SHARED=overlap): run it on the
        # side stream concurrently with the megakernel and return routed + shared.
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"DeepSeek V4 MegaMoE got {hidden_states.shape[0]} tokens, "
                f"but the symmetric buffer was sized for {self.max_num_tokens}."
            )

        from flashinfer.moe_ep import MoEEpTensors

        num_tokens = hidden_states.shape[0]
        is_padding = resolve_mega_moe_is_padding(num_tokens)
        # SM107: flashinfer masks the padding rows inside its route staging.
        pad_kwargs: dict[str, Any] = {}
        if (
            self._is_sm107
            and is_padding is not None
            and is_padding.is_contiguous()
            and fi_stages_route_padding_mask()
        ):
            pad_kwargs = {"route_padding_mask": is_padding}
        else:
            topk_ids = apply_mega_moe_routing_preprocess(
                topk_ids,
                is_padding=is_padding,
            )
        scalars = self._scalar_kwargs()

        if self._is_sm107:
            if self._mega_layer is None:
                ensure_fi_moe_ep_runtime(self._vllm_config)
                self.finalize_weights()
                scalars = self._scalar_kwargs()
            kernel, workspace, transformed = self._select_profile(num_tokens)
            t = MoEEpTensors(
                hidden_states=hidden_states.contiguous(),
                topk_ids=topk_ids.contiguous(),
                topk_weights=topk_weights.contiguous(),
                **scalars,
                **pad_kwargs,
            )
            kernel.stage_inputs(t, workspace, quantize_input=True)
            # Zero-copy workspace [:n] view: valid under stream ordering
            # until the next MoE layer's launch on the same profile.
            if shared_experts is None:
                return kernel.compute(workspace, transformed, output=None)
            return self._compute_with_overlapped_shared(
                kernel, workspace, transformed, hidden_states, shared_experts
            )

        # Fast path: after the first successful full forward the layer is
        # immutable, so skip MoEEpMegaLayer.forward()'s per-call validation
        # and go straight to the kernel backend's stage_inputs + compute.
        fast = self._fast_ctx
        if fast is not None:
            kernel, workspace, transformed, hidden_size, zero_copy = fast
            t = MoEEpTensors(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                **scalars,
            )
            kernel.stage_inputs(t, workspace, quantize_input=True)
            if zero_copy:
                # Zero-copy (cutedsl backends): consume the workspace [:n]
                # view directly — valid under stream ordering until the next
                # MoE layer's launch on the shared workspace.
                return kernel.compute(workspace, transformed, output=None)
            # deep_gemm_mega's compute() requires a real output tensor.
            out = torch.empty(
                num_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            return kernel.compute(workspace, transformed, output=out)

        ensure_fi_moe_ep_runtime(self._vllm_config)
        self.finalize_weights()
        assert self._mega_layer is not None

        y = self._mega_layer.forward(
            MoEEpTensors(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                **self._scalar_kwargs(),
            )
        )
        layer = self._mega_layer
        if hidden_states.dtype == torch.bfloat16:
            self._fast_ctx = (
                layer._kernel,
                layer._ensure_workspace(),
                layer._transformed,
                layer._fleet_params.token_hidden_size,
                # zero-copy output views are a cutedsl-backend contract
                layer._kernel.kernel_name() != "deep_gemm_mega",
            )
        return y


DeepseekV4MegaMoEExpertsFI.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
