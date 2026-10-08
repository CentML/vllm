# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import dataclasses
import os
from typing import Literal

import torch
from einops import rearrange
from torch import nn

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.kernels.linear.lowm_bf16_gemm import (
    lowm_bf16_gemm_out,
    maybe_use_lowm_bf16_gemm,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
    gdn_gated_norm_mxfp8,
    gdn_mxfp8_scale_numel,
    gdn_norm_launch_config,
    zero_fresh_state_rows,
)
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops import gdn_mtp_cuda
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gdn_fused_conv_prep import (
    FUSED_CONV_TILE_LONG,
    FUSED_CONV_TILE_SHORT,
    GDN_CONV_CUDA,
    enable_cuda_kernel,
    gdn_fused_conv_prep,
)
from vllm.model_executor.layers.mamba.ops.gdn_host_trim import (
    GDN_HOST_TRIM,
    GDN_HOST_TRIM2,
    drop_empty_triton_launch_hooks,
)
from vllm.model_executor.layers.mamba.ops.gdn_mtp_decode import gdn_mtp_recurrence
from vllm.model_executor.layers.mamba.ops.gdn_mtp_fused_decode import (
    MAX_FUSED_DECODE_TOKENS,
    gdn_mtp_fused_decode,
)
from vllm.model_executor.layers.mamba.ops.gdn_prefill_checkpoint import (
    copy_state_rows,
    gather_prefill_rows,
    store_conv_checkpoint,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.model_executor.layers.utils import default_unquantized_gemm
from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops import (
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    aux_stream,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNPrefillCheckpointMetadata,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

# Optional ROCm AITER Triton kernels for the GDN decode path.
# Availability is checked centrally via rocm_aiter_ops; the actual function
# references are imported here so that they can be called without per-call
# import overhead.
GDN_AITER_TRITON_AVAILABLE = (
    rocm_aiter_ops.are_gdn_triton_kernels_available()
    or rocm_aiter_ops.is_rdna_gdn_triton_kernels_available()
)

if GDN_AITER_TRITON_AVAILABLE:
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token as gdn_aiter_fused_reshape_causal_conv1d_update_single_token,  # noqa: E501
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule,  # noqa: E501
    )

logger = init_logger(__name__)

MAX_FUSED_GDN_MTP_TOKENS = 8
FUSED_GDN_STATE_DTYPES = (torch.float32, torch.bfloat16)
# Spec-decode batches of at most this many requests run the GDN recurrence in
# the value-split Triton kernel (gdn_mtp_recurrence; latency-bound regime) and
# the gated norm in the MXFP8/norm kernel after it. Larger batches use the CUDA
# MTP kernel (bandwidth-bound regime). VR crossover, norm included: ~4-6.
GDN_MTP_TRITON_MAX_REQUESTS = int(os.environ.get("VLLM_GDN_MTP_TRITON_MAX_REQS", "4"))
# Decode-only spec batches of at most this many requests run the whole GDN core
# (conv window roll + recurrence + gated norm + out_proj's MXFP8 quant) as one
# launch (gdn_mtp_fused_decode) instead of three. Off at 0 (default). VR, 4
# tokens per request, per GDN layer: -2.9 us at 1-2 requests, -3.4 at 3, -2.2 at
# 4, none at 6.
GDN_FUSED_DECODE_MAX_REQUESTS = int(
    os.environ.get("VLLM_GDN_FUSED_DECODE_MAX_REQS", "0")
)
# ... and at most this many tokens per request (MTP k + 1). The kernel unrolls
# up to MAX_FUSED_DECODE_TOKENS = 7 (k = 6); the default 6 keeps k = 6 on the
# three-kernel path until the 7-token kernel is adopted.
GDN_FUSED_DECODE_MAX_TOKENS = min(
    int(os.environ.get("VLLM_GDN_FUSED_DECODE_MAX_TOKENS", "6")),
    MAX_FUSED_DECODE_TOKENS,
)
# Larger spec-decode batches run the register-resident CUDA kernel of
# ops/gdn_mtp_cuda.py (JIT-built by the GDN warmup; same contract as the csrc
# MTP kernel, VR 77 requests: 77.0 vs 81.6 us, 48: 48.3 vs 55.3 us) instead of
# the csrc kernel; VLLM_GDN_MTP_JIT=0 keeps the csrc kernel.
GDN_MTP_CUDA_JIT = os.environ.get("VLLM_GDN_MTP_JIT", "1") == "1"
# VLLM_GDN_BA_LATE_JOIN=1: in FULL-graph decode-only spec batches, the main
# stream waits for the aux-stream in_proj_ba GEMM right before the recurrence
# (its first reader of b/a) instead of before the conv1d update, so the conv
# overlaps the GEMM's tail and the cross-stream join latency. The core op then
# receives the GEMM's input to keep it alive until the join. Every other batch
# (PIECEWISE pieces must join inside the captured piece) joins as before.
GDN_BA_LATE_JOIN = os.environ.get("VLLM_GDN_BA_LATE_JOIN", "0") == "1"
# FlashInfer GDN prefill: a single sequence of at most this many tokens runs
# the non-CP chunked kernel. FlashInfer's auto heuristic picks CP for every
# single sequence on SM10x, but on VR CP is slower up to ~4.6k tokens
# (2144 tokens: 96 vs 76 us).
GDN_FI_NON_CP_MAX_TOKENS = int(os.environ.get("VLLM_GDN_FI_NON_CP_MAX_TOKENS", "4608"))
# Prefill rows: causal conv1d and the post-conv prep (L2 norm, exp(g), beta) in
# one kernel (gdn_fused_conv_prep), bitwise equal to the two-kernel pair.
GDN_FUSED_CONV_PREP = os.environ.get("VLLM_GDN_FUSED_CONV_PREP", "1") == "1"
# Non-CP FlashInfer GDN prefill on the in-place state pool: steps whose
# (sequence x value head) grid under-fills the GPU run the V-split copy of the
# kernel (vllm/third_party/flashinfer_gdn_vsplit: one CTA per (sequence, head,
# 64-row V slice)), bitwise identical to FlashInfer's non-CP kernel. The rule
# (choose_vsplit) uses host ints only. The kernel is compiled by the prefill
# warmup; it stays off in a process whose warmup did not compile it.
GDN_FI_VSPLIT = os.environ.get("VLLM_GDN_FI_VSPLIT", "1") == "1"
# VLLM_GDN_FI_VSPLIT_V1=1: also run the steps the rule leaves at v_split=1 on
# the vendored kernel (same math, bitwise identical to FlashInfer's non-CP
# kernel) instead of FlashInfer. FlashInfer's indexed state-pool mode drops the
# state stride divisibility hint; the vendored adapter keeps it.
_GDN_FI_VSPLIT_V1 = envs.VLLM_GDN_FI_VSPLIT_V1
_gdn_vsplit_ready: list = []  # [module] once the warmup compiled the kernel
_gdn_vsplit_tried: list = []
# With V-split, the non-CP kernel beats CP for single sequences up to ~8.5k
# tokens on VR (us, CP -> V-split: 5120 130 -> 107, 6144 138 -> 127, 8192
# 167 -> 166; 10240 196 -> 205), so the CP threshold moves up while V-split is
# enabled.
GDN_FI_VSPLIT_NON_CP_MAX_TOKENS = int(
    os.environ.get("VLLM_GDN_FI_VSPLIT_NON_CP_MAX_TOKENS", "8192")
)
# Mixed batches (spec-decode rows + prefill rows): the spec rows' MTP recurrence
# and the prefill rows' conv + chunked prefill touch disjoint state slots and
# output rows, so with VLLM_GDN_MIXED_FORK=1 the two halves run on two streams
# (same kernels and inputs: bitwise). Steps with at least
# VLLM_GDN_MIXED_FORK_PREFILL_FIRST prefill tokens run the prefill half on a
# high-priority side stream (its one-CTA-per-SM chunk kernel is dispatched first
# and the MTP kernel fills the SMs it leaves); smaller ones run the spec half on
# a side stream. VR microbench (r4 mixed track), per GDN layer, 64 spec
# requests: -3 us at <= 1000 prefill tokens, -15 us at 2000, -29 us at 3000+.
GDN_MIXED_FORK = os.environ.get("VLLM_GDN_MIXED_FORK", "0") == "1"
GDN_MIXED_FORK_PREFILL_FIRST = int(
    os.environ.get("VLLM_GDN_MIXED_FORK_PREFILL_FIRST", "1500")
)
_gdn_mixed_fork_state: list = []  # [(spec_stream, prefill_stream, fork, join)]


def _gdn_mixed_fork_streams() -> tuple:
    """Per-process side streams and events of the mixed-batch fork."""
    if not _gdn_mixed_fork_state:
        _, high = torch.cuda.Stream.priority_range()
        _gdn_mixed_fork_state.append(
            (
                torch.cuda.Stream(),
                torch.cuda.Stream(priority=high),
                torch.cuda.Event(),
                torch.cuda.Event(),
            )
        )
    return _gdn_mixed_fork_state[0]


def _gdn_fi_non_cp_max_tokens() -> int:
    """Longest single sequence that runs FlashInfer's non-CP prefill."""
    if _gdn_vsplit_ready:
        return GDN_FI_VSPLIT_NON_CP_MAX_TOKENS
    return GDN_FI_NON_CP_MAX_TOKENS


def _gdn_vsplit_warmup(
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    device,
) -> None:
    """Compile the V-split kernel for the one variant serving uses (pool of
    ``state_dtype``, fp32 or bf16, with int32 state_indices, initial and final
    state) on a dummy pool and enable it; once per process. cute.compile keeps
    no on-disk cache: without this, the first eligible prefill of each worker
    would stall for seconds on the compile. A failed compile leaves
    FlashInfer's kernel in place.

    The stock FlashInfer non-CP kernel still runs the steps choose_vsplit
    leaves at v_split=1 (some 4-6 sequence batches). Without V-split, the
    warmup's dummy batches compile it; with V-split they all take the V-split
    kernel, so the same pooled variant is compiled here as well. With
    VLLM_GDN_FI_VSPLIT_V1=1 those steps run the vendored kernel at v_split=1,
    which is compiled here too.
    """
    if _gdn_vsplit_tried:
        return
    _gdn_vsplit_tried.append(True)
    try:
        import vllm.third_party.flashinfer_gdn_vsplit as gdn_vsplit

        T = 2 * FLA_CHUNK_SIZE
        q = torch.zeros(T, num_k_heads, head_dim, device=device, dtype=dtype)
        v = torch.zeros(T, num_v_heads, head_dim, device=device, dtype=dtype)
        gate = torch.ones(T, num_v_heads, device=device, dtype=torch.float32)
        pool = torch.zeros(
            2, num_v_heads, head_dim, head_dim, device=device, dtype=state_dtype
        )
        gdn_vsplit.chunk_gated_delta_rule_vsplit(
            q,
            q,
            v,
            gate,
            gate,
            torch.empty_like(v),
            torch.tensor([0, T], device=device, dtype=torch.int32),
            pool,
            pool,
            head_dim**-0.5,
            state_indices=torch.ones(1, device=device, dtype=torch.int32),
            v_split=2,
        )
        if _GDN_FI_VSPLIT_V1:
            gdn_vsplit.chunk_gated_delta_rule_vsplit(
                q,
                q,
                v,
                gate,
                gate,
                torch.empty_like(v),
                torch.tensor([0, T], device=device, dtype=torch.int32),
                pool,
                pool,
                head_dim**-0.5,
                state_indices=torch.ones(1, device=device, dtype=torch.int32),
                v_split=1,
            )
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        chunk_gated_delta_rule(
            q=q,
            k=q,
            v=v,
            g=gate,
            beta=gate,
            initial_state=pool,
            output_final_state=True,
            cu_seqlens=torch.tensor([0, T], device=device, dtype=torch.int64),
            output=torch.empty_like(v),
            output_state=pool,
            state_indices=torch.ones(1, device=device, dtype=torch.int32),
            use_cp=False,
        )
    except Exception:
        logger.warning(
            "GDN V-split prefill kernel failed to compile; FlashInfer's non-CP "
            "kernel stays in use.",
            exc_info=True,
        )
        return
    _gdn_vsplit_ready.append(gdn_vsplit)
    logger.info(
        "GDN prefill: V-split FlashInfer kernel compiled and enabled (%s state).",
        str(state_dtype).removeprefix("torch."),
    )


def _consumes_swizzled_mxfp8(linear: nn.Module) -> bool:
    """Whether ``linear`` takes a pre-quantized MXFP8 activation with FlashInfer's
    F8_128x4 swizzled scales, the layout ``gdn_gated_norm_mxfp8`` writes.
    """
    if not current_platform.is_cuda():
        return False
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )

    kernel = getattr(getattr(linear, "quant_method", None), "kernel", None)
    return (
        isinstance(
            kernel,
            (FlashInferCutedslMxfp8LinearKernel, FlashInferCutlassMxfp8LinearKernel),
        )
        and kernel.input_quant_key() == kMxfp8Dynamic
    )


def _resolve_gdn_prefill_backend(
    vllm_config: VllmConfig,
) -> tuple[str, Literal["triton", "flashinfer", "cutedsl"]]:
    """Resolve GDN prefill backend.

    FlashInfer's GDN prefill kernel is chosen when:
    * ``requested in ["flashinfer", "auto"]``;
    * ``platform == cuda``;
    * one of the following:
      - Hopper (SM90) - no further constraints;
      - Blackwell (SM10.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``;
      - Blackwell (SM12.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``.

    In-tree CuteDSL GDN prefill kernel is chosen when:
    * "cutedsl" is requested; (opt-in only)
    * Blackwell (SM10.x) with ``head_k_dim == 128``;
    """
    additional_config = vllm_config.additional_config
    backend_cfg = (
        additional_config.get("gdn_prefill_backend", "auto")
        if isinstance(additional_config, dict)
        else "auto"
    )
    backend = str(backend_cfg).strip().lower()

    if not current_platform.is_cuda():
        return backend, "triton"

    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )

    supports_flashinfer = False
    supports_cutedsl = False

    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True
    elif (
        current_platform.is_device_capability_family(120)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        # The in-tree CuteDSL kernel targets SM100 only, so it stays off here.
        supports_flashinfer = True

    if backend in ["flashinfer", "auto"] and supports_flashinfer:
        return backend, "flashinfer"
    if backend == "cutedsl" and supports_cutedsl:
        return backend, "cutedsl"
    return backend, "triton"


def _log_gdn_backend_decision(
    vllm_config: VllmConfig,
    requested_backend: str,
    active_backend: str,
) -> None:
    """Log the GDN prefill backend choice in the attention-selector style."""
    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )

    if current_platform.is_cpu():
        logger.info_once(
            "Using %s GDN prefill kernel (head_k_dim=%s).",
            "CPU",
            head_k_dim,
        )
        return

    chosen = {
        "flashinfer": "FlashInfer",
        "cutedsl": "CuteDSL",
        "triton": "Triton/FLA",
    }[active_backend]
    logger.info_once(
        "Using %s GDN prefill kernel (requested=%s, head_k_dim=%s).",
        chosen,
        requested_backend,
        head_k_dim,
    )
    if active_backend == "flashinfer" and current_platform.is_device_capability(90):
        logger.warning_once(
            "FlashInfer GDN prefill is JIT-compiled; first run may take a "
            "while. Set --gdn-prefill-backend triton to skip JIT.",
        )


def _contig(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def _to(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t if t.dtype == dtype else t.to(dtype)


def _rows(t: torch.Tensor, n: int) -> torch.Tensor:
    """``t[:n]``; ``t`` itself if it has at most ``n`` rows (same view)."""
    return t if t.size(0) <= n else t[:n]


def _host_spec_args(m: GDNAttentionMetadata) -> tuple:
    """(spec conv slots, spec state slots, accepted counts, cu_seqlens, max
    query len) of the spec-decode requests: views of the step's metadata,
    built once per metadata object (VLLM_GDN_HOST_TRIM) instead of once per
    GDN layer.
    """
    n = m.num_spec_decodes
    c = m.__dict__.get("_host_spec_args")
    if c is None or c[0] != n:
        si = m.spec_state_indices_tensor
        cu = m.spec_query_start_loc
        acc = m.num_accepted_tokens
        assert si is not None and cu is not None and acc is not None
        c = (n, si[:n, 0], si[:n], acc[:n], cu[: n + 1], si.size(1))
        m.__dict__["_host_spec_args"] = c
    return c[1:]


def _host_layer_views(layer) -> tuple[torch.Tensor, torch.Tensor]:
    """(conv state view for the conv kernels, 2-D conv weight) of ``layer``,
    built once per (conv pool, weight) object (VLLM_GDN_HOST_TRIM).
    """
    src = layer.kv_cache[0]
    w = layer.conv1d.weight
    c = layer.__dict__.get("_host_layer_views")
    if c is None or c[0] is not src or c[1] is not w:
        conv_state = src if is_conv_state_dim_first() else src.transpose(-1, -2)
        c = (src, w, conv_state, w.view(w.size(0), w.size(2)))
        layer.__dict__["_host_layer_views"] = c
    return c[2], c[3]


def _fused_conv_prep_applies(layer, attn_metadata: GDNAttentionMetadata) -> bool:
    """Whether ``layer._forward_core`` runs the prefill rows' conv1d and
    post-conv prep in one gdn_fused_conv_prep launch (the same condition as
    its ``fused_conv_prep``: prefill rows without peeled-off non-spec decodes,
    FlashInfer exp(g), no conv bias, any sequence count).
    """
    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
    return (
        GDN_FUSED_CONV_PREP
        and attn_metadata.num_prefills > 0
        and not (
            attn_metadata.spec_sequence_masks is None and attn_metadata.num_decodes > 0
        )
        and layer.chunk_gated_delta_rule.expects_exp_g
        and layer.conv1d.bias is None
        and non_spec_query_start_loc is not None
    )


def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    g_is_exp: bool = False,
    output: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    max_seqlen: int = 0,
    cu_seqlens_i32: torch.Tensor | None = None,
):
    """FlashInfer chunked GDN prefill.

    ``g_is_exp``: ``g`` already holds exp(g) (``fused_post_conv_prep`` with
    ``output_g_exp=True``). ``output``: contiguous buffer with ``v.numel()``
    elements that FlashInfer writes into instead of allocating.
    ``state_indices``: contiguous int32 slot ids; ``initial_state`` is then the
    SSM state pool (fp32 or bf16), read from and updated in place at those
    rows (SM10x).
    ``max_seqlen`` (longest sequence, host int) and ``cu_seqlens_i32`` (int32
    copy of ``cu_seqlens``) feed the V-split path; both are optional.
    """
    if GDN_HOST_TRIM:
        # Same tensors as below without the no-op dispatches (contiguous()
        # of a contiguous tensor and .to() to its own dtype return it).
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
        q = _contig(q.squeeze(0))
        k = _contig(k.squeeze(0))
        v = _contig(v.squeeze(0))
        g = _contig(g.squeeze(0))
        beta = _contig(beta.squeeze(0))
        fi_state = (
            initial_state
            if state_indices is not None
            else _to(initial_state, torch.float32)
        )
        fi_g = _to(g, torch.float32)
        fi_beta = _to(beta, torch.float32)
        if cu_seqlens is not None:
            cu_seqlens = _to(cu_seqlens, torch.int64)
    else:
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        # use flashinfer implementation
        q = q.squeeze(0).contiguous()
        k = k.squeeze(0).contiguous()
        v = v.squeeze(0).contiguous()

        g = g.squeeze(0).contiguous()
        beta = beta.squeeze(0).contiguous()
        # The in-place pool is passed as is (FlashInfer indexes its rows).
        fi_state = (
            initial_state
            if state_indices is not None
            else initial_state.to(torch.float32)
        )
        fi_g = g.to(torch.float32)
        fi_beta = beta.to(torch.float32)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(torch.int64)
    num_seqs = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
    non_cp_max_tokens = _gdn_fi_non_cp_max_tokens()
    use_non_cp = num_seqs > 1 or q.shape[0] <= non_cp_max_tokens
    if (
        _gdn_vsplit_ready
        and use_non_cp
        and state_indices is not None
        and g_is_exp
        and output_final_state
        and fi_state.dtype in (torch.float32, torch.bfloat16)
    ):
        gdn_vsplit = _gdn_vsplit_ready[0]
        v_split = gdn_vsplit.choose_vsplit(
            num_seqs,
            q.shape[0],
            q.shape[0] if num_seqs == 1 else max_seqlen,
            hv=v.shape[1],
        )
        if v_split > 1 or _GDN_FI_VSPLIT_V1:
            out = torch.empty_like(v) if output is None else output.view(v.shape)
            if cu_seqlens_i32 is None:
                assert cu_seqlens is not None
                cu_seqlens_i32 = cu_seqlens.to(torch.int32)
            gdn_vsplit.chunk_gated_delta_rule_vsplit(
                q,
                k,
                v,
                fi_g,
                fi_beta,
                out,
                cu_seqlens_i32,
                fi_state,
                fi_state,
                q.shape[-1] ** -0.5,
                state_indices=state_indices,
                v_split=v_split,
            )
            return out.unsqueeze(0), fi_state
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=fi_g if g_is_exp else torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        output=None if output is None else output.view(v.shape),
        output_state=fi_state if state_indices is not None else None,
        state_indices=state_indices,
        use_cp=(False if num_seqs == 1 and q.shape[0] <= non_cp_max_tokens else "auto"),
    )
    # FlashInfer returns (output, state) when output_final_state=True,
    # or just output when output_final_state=False.
    # Unsqueeze back to 4D (1, L, H, D) to match fla output format
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        vllm_config = get_current_vllm_config()
        backend, active_backend = _resolve_gdn_prefill_backend(vllm_config)
        self.gdn_prefill_backend = active_backend
        # Callers of the FlashInfer path pass exp(g), computed in
        # fused_post_conv_prep, instead of g.
        self.expects_exp_g = active_backend == "flashinfer"

        if backend in ("flashinfer", "cutedsl") and active_backend != backend:
            logger.warning_once(
                "GDN prefill backend '%s' is selected but cannot use this "
                "kernel on the current platform. Falling back to Triton/FLA.",
                backend,
            )
        _log_gdn_backend_decision(vllm_config, backend, active_backend)

        if active_backend == "flashinfer":
            self._forward_method = self.forward_cuda
        elif active_backend == "cutedsl":
            self._forward_method = self.forward_cutedsl
        else:
            self._forward_method = self.forward_native

    def updates_state_in_place(self, state_dtype: torch.dtype) -> bool:
        """Whether prefill reads and writes the SSM state pool in place
        (FlashInfer ``state_indices``, SM10x, fp32 or bf16 pool). FlashInfer's
        SM100 kernels (non-CP and CP) load a bf16 pool row into fp32, compute
        in fp32 and store the final state rounded to nearest even, as the
        gather path's ``last_state.to(ssm_state.dtype)`` does.
        """
        return (
            self.gdn_prefill_backend == "flashinfer"
            and current_platform.is_device_capability_family(100)
            and state_dtype in (torch.float32, torch.bfloat16)
        )

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        max_seqlen: int = 0,
        cu_seqlens_i32: torch.Tensor | None = None,
    ):
        o, final_state = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            g_is_exp=self.expects_exp_g,
            output=core_attn_out,
            state_indices=state_indices,
            max_seqlen=max_seqlen,
            cu_seqlens_i32=cu_seqlens_i32,
        )
        return o, final_state

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
        )

    def forward_cutedsl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
            chunk_gated_delta_rule_cutedsl,
        )

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        assert cu_seqlens is not None
        assert chunk_indices is not None
        assert chunk_offsets is not None

        o, final_state = chunk_gated_delta_rule_cutedsl(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )
        if not output_final_state:
            final_state = None
        return o, final_state


@PluggableLayer.register("qwen_gated_delta_net_attention")
class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if not (
            envs.VLLM_GDN_PREFILL_CHECKPOINT
            and isinstance(spec, MambaSpec)
            and spec.mamba_cache_mode == "align"
            and current_platform.is_cuda()
        ):
            return spec
        if vllm_config.cache_config.prefix_cache_retention_interval != 0:
            # With retained interior states, a single forward that skips a
            # block boundary would leave that boundary's slot unwritten.
            logger.warning_once(
                "VLLM_GDN_PREFILL_CHECKPOINT needs "
                "prefix_cache_retention_interval=0; keeping split prefills."
            )
            return spec
        # The prefill forward splits each checkpointed sequence at the
        # checkpoint (_chunk_prefill_with_checkpoint), so any offset works.
        return dataclasses.replace(
            spec,
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=1,
            prefill_checkpoint_reuses_initial_block=True,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        gqa_interleaved_layout=False,
        reduce_results: bool = True,
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.gqa_interleaved_layout = gqa_interleaved_layout
        if current_platform.is_xpu():
            self._forward_method = self.forward_xpu
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.mamba.ops.cpu.gdn_attention import (
                register_cpu_gdn_attention_ops,
            )

            register_cpu_gdn_attention_ops()
            self._forward_method = self.forward_cpu
        elif current_platform.is_rocm():
            self._forward_method = self.forward_hip
        else:
            self._forward_method = self.forward_cuda

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,
        # we need to create qkvz_proj adaptively here.
        # When create_in_proj_qkvz is False (e.g. LoRA enabled in Qwen3.5),
        # in_proj_qkv and in_proj_z are created separately instead.
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )

        # ba_proj doesn't support blockwise fp8 quantization.
        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint
        # layouts, so we use a factory method to create the projection.
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )
        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)
        # Decode-size BA GEMM: one kernel instead of cuBLAS split-K + reduce
        # (SM107; SM100/SM103 with VLLM_LOWM_BF16_GEMM_SM100=1; no-op elsewhere).
        ba_lowm = maybe_use_lowm_bf16_gemm(self.in_proj_ba)
        # in_proj_ba only reads the layer input, so it runs on the aux stream
        # concurrently with the in_proj_qkvz GEMM (which leaves SMs idle at
        # decode sizes). Only the plain bf16 GEMM qualifies: the aux stream
        # then issues the low-M kernel (when opted in above and M is in its
        # range) or the same torch.mm(out=) call Inductor emits for it.
        self._ba_stream: torch.cuda.Stream | None = None
        self._ba_stream_max_tokens = envs.VLLM_GDN_BA_STREAM_TOKEN_THRESHOLD
        self._ba_pending = False
        if (
            current_platform.is_cuda()
            and self._ba_stream_max_tokens > 0
            and not envs.VLLM_BATCH_INVARIANT
            and type(self.in_proj_ba.quant_method) is UnquantizedLinearMethod
            and (
                ba_lowm
                or self.in_proj_ba.quant_method._gemm_impl is default_unquantized_gemm
            )
            and self.in_proj_ba.bias is None
        ):
            self._ba_stream = aux_stream()
            self._ba_events = (torch.cuda.Event(), torch.cuda.Event())

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [
                query_key_settings,
                query_key_settings,
                value_settings,
            ],
            self.tp_size,
            self.tp_rank,
        )

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        assert output_gate_type in ["silu", "swish", "sigmoid"], (
            f"unsupported {output_gate_type=}"
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            activation=output_gate_type,
            device=current_platform.current_device(),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=reduce_results,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        self.gdn_prefill_backend = self.chunk_gated_delta_rule.gdn_prefill_backend
        self._prefill_kernels_warmed_up = False
        drop_empty_triton_launch_hooks()
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        )
        self.gdn_decode_kernel = envs.VLLM_GDN_DECODE_KERNEL.strip().lower()
        if self.gdn_decode_kernel == "cuda" and current_platform.is_cuda_alike():
            reason = self._fused_gdn_decode_unsupported_reason(vllm_config)
            if reason is not None:
                if "VLLM_GDN_DECODE_KERNEL" in os.environ:
                    raise ValueError(
                        f"VLLM_GDN_DECODE_KERNEL=cuda is not supported: {reason}"
                    )
                logger.info_once(
                    "Falling back to the Triton GDN decode path: %s", reason
                )
                self.gdn_decode_kernel = "triton"
        elif current_platform.is_cpu():
            self.gdn_decode_kernel = "CPU"

        self.enable_fused_gdn_decode = self.gdn_decode_kernel == "cuda"
        logger.info_once("GDN decode kernel: %s", self.gdn_decode_kernel)
        # The fused core op also emits out_proj's swizzled MXFP8 activation
        # when out_proj is a FlashInfer MXFP8 linear that consumes it.
        self.gdn_out_mxfp8 = self.enable_fused_gdn_decode and _consumes_swizzled_mxfp8(
            self.out_proj
        )
        # One-launch spec-decode core for small decode batches: its per
        # (request, key head) arrival counters (zero between launches); None
        # when off or ineligible.
        fused_decode = (
            GDN_FUSED_DECODE_MAX_REQUESTS > 0
            and self.gdn_out_mxfp8
            and self.conv_kernel_size == 4
            and self.conv1d.bias is None
            and self.activation in ("silu", "swish")
            and self.norm.bias is None
            and self.norm.norm_before_gate
        )
        self.register_buffer(
            "_fused_decode_counters",
            torch.zeros(
                GDN_FUSED_DECODE_MAX_REQUESTS * (self.num_k_heads // self.tp_size),
                dtype=torch.int32,
                device=current_platform.current_device(),
            )
            if fused_decode
            else None,
            persistent=False,
        )
        if fused_decode:
            logger.info_once(
                "GDN spec-decode core in one launch for decode batches of <= %d "
                "requests",
                GDN_FUSED_DECODE_MAX_REQUESTS,
            )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _fused_gdn_decode_unsupported_reason(
        self, vllm_config: VllmConfig
    ) -> str | None:
        conv_state_dtype, recurrent_state_dtype = self.get_state_dtype()
        if (
            self.gqa_interleaved_layout
            or self.head_k_dim != 128
            or self.head_v_dim != 128
            or self.norm.activation not in ("silu", "sigmoid")
            or vllm_config.model_config.dtype != torch.bfloat16
            or conv_state_dtype != torch.bfloat16
            or recurrent_state_dtype not in FUSED_GDN_STATE_DTYPES
            or not current_platform.has_device_capability(80)
        ):
            return (
                "the fused CUDA kernel requires a BF16 GDN model with "
                "K=V=128, SiLU or sigmoid gating, non-interleaved GQA "
                "layout, BF16 convolution cache, BF16 or FP32 recurrent "
                "state, and a GPU with compute capability 8.0+"
            )
        if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"):
            return "torch.ops._C.fused_gdn_decode_post_conv_mtp is not built"
        return None

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), qkvz weights are
        # stored as a single fused tensor with interleaved GQA layout, so we
        # use one output shard to preserve the interleaving across TP ranks.
        # When gqa_interleaved_layout=False (Qwen3.5), the checkpoint has
        # separate q, k, v, z weights, so we use 4 independent output sizes.
        output_sizes = (
            [sum((key_dim, key_dim, value_dim, value_dim))]
            if self.gqa_interleaved_layout
            else [key_dim, key_dim, value_dim, value_dim]
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), in_proj_ba is stored
        # as a single fused weight [b_g0, a_g0, b_g1, a_g1, ...] interleaved
        # by key-head group; a single output shard preserves this across TP.
        # When gqa_interleaved_layout=False (Qwen3.5), in_proj_b and in_proj_a
        # are separate checkpoint weights, so we use 2 independent output sizes.
        output_sizes = (
            [num_v_heads * 2] if self.gqa_interleaved_layout else [num_v_heads] * 2
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=self.maybe_disable_tp(quant_config),
        )

    def maybe_disable_tp(self, quant_config: QuantizationConfig | None) -> bool:
        """Whether to replicate ba_proj instead of TP-sharding it.

        Marlin requires output_size_per_partition >= MIN_THREAD_N=64, which
        the Qwen3.5 non-interleaved [num_v_heads]*2 layout violates at TP>=2
        (e.g. num_v_heads=64, TP=4 -> 16). Replicating the projection keeps
        each rank above the Marlin threshold; forward() then slices b/a to
        the local TP partition. Qwen3-Next's interleaved [num_v_heads*2]
        layout is unaffected and stays TP-sharded.

        See https://github.com/vllm-project/vllm/issues/35924
        """
        return (
            current_platform.is_cuda()
            and not self.gqa_interleaved_layout
            and isinstance(quant_config, (AutoAWQConfig, AutoGPTQConfig, INCConfig))
        )

    def split_ba(self, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, a = ba.chunk(2, dim=-1)
        if self.disable_tp_for_ba_proj and self.tp_size > 1:
            # ba_proj is replicated for Marlin; slice b/a to local TP rank.
            ba_chunk = self.num_v_heads // self.tp_size
            ba_start = self.tp_rank * ba_chunk
            b = b[:, ba_start : ba_start + ba_chunk]
            a = a[:, ba_start : ba_start + ba_chunk]
        return b, a

    def _in_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """in_proj_qkvz and in_proj_ba; in_proj_ba overlaps on the aux stream
        when enabled.
        """
        if self._ba_stream is None:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            return mixed_qkvz, ba
        layer_name = _encode_layer_name(self.prefix)
        ba = torch.empty(
            (hidden_states.size(0), self.in_proj_ba.weight.size(0)),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.gdn_in_proj_ba_fork(hidden_states, ba, layer_name)
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        # Passing hidden_states and mixed_qkvz keeps the aux-stream input alive
        # until the join and keeps in_proj_qkvz scheduled between the two ops.
        torch.ops.vllm.gdn_in_proj_ba_join(ba, hidden_states, mixed_qkvz, layer_name)
        return mixed_qkvz, ba

    def _in_proj_ba_gemm(self, hidden_states: torch.Tensor, ba: torch.Tensor) -> None:
        if not lowm_bf16_gemm_out(self.in_proj_ba, hidden_states, ba):
            torch.mm(hidden_states, self.in_proj_ba.weight.t(), out=ba)

    def _in_proj_ba_fork(self, hidden_states: torch.Tensor, ba: torch.Tensor) -> None:
        stream = self._ba_stream
        if (
            hidden_states.size(0) > self._ba_stream_max_tokens
            or BreakableCUDAGraphCapture.is_active()
        ):
            stream = None
        if stream is None:
            self._in_proj_ba_gemm(hidden_states, ba)
            return
        main_stream = torch.cuda.current_stream()
        fork_event, join_event = self._ba_events
        fork_event.record(main_stream)
        with torch.cuda.stream(stream):
            fork_event.wait(stream)
            self._in_proj_ba_gemm(hidden_states, ba)
            join_event.record(stream)
        self._ba_pending = True

    def _in_proj_ba_join(self) -> None:
        if self._ba_pending:
            self._ba_events[1].wait(torch.cuda.current_stream())
            self._ba_pending = False

    def _uses_fused_gdn_decode(self, dtype: torch.dtype) -> bool:
        """Whether forward_cuda takes the fused-norm (packed) core op."""
        return (
            self.enable_fused_gdn_decode
            and dtype == torch.bfloat16
            and self.norm.weight.dtype in (torch.bfloat16, torch.float32)
        )

    def _ba_join_deferred(self, ba: torch.Tensor) -> bool:
        """VLLM_GDN_BA_LATE_JOIN: leave the join to the fused core op, which
        waits right before its first read of b/a. Only inside FULL graphs: a
        PIECEWISE piece must join the aux stream before the core op (a
        splitting op) ends the captured piece.
        """
        return (
            GDN_BA_LATE_JOIN
            and self._ba_pending
            and self._uses_fused_gdn_decode(ba.dtype)
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """Derives `query`, `key` and `value` tensors from `mixed_qkvzba`."""
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    @torch.compile(fullgraph=True)
    def prepare_gdn_attention_core_inputs(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
        num_tokens: int,
    ):
        """Derives mixed_qkv, z, b, a from projected qkvz/ba for the GDN custom op.

        For gqa_interleaved_layout (Qwen3-Next): unpack the interleaved
        [ng, (hk + hk + np/ng*hv + np/ng*hv)] layout into contiguous qkv.
        For non-interleaved layout (Qwen3.5): simple split along last dim.
        """
        if not self.gqa_interleaved_layout:
            # Qwen3.5: weights are in [q, k, v, z] order
            assert num_tokens == mixed_qkvz.shape[0]
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z_flat = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            n = mixed_qkvz.shape[0]
            z_out = z_flat.reshape(n, -1, self.head_v_dim)
            b, a = mixed_ba.chunk(2, dim=-1)
            return mixed_qkv, z_out, b, a

        # Qwen3-Next: interleaved GQA layout
        base_shape_qkvz = mixed_qkvz.size()[:-1]
        base_shape_ba = mixed_ba.size()[:-1]
        ng = self.num_k_heads // self.tp_size

        new_tensor_shape_qkvz = base_shape_qkvz + (
            ng,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = base_shape_ba + (
            ng,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=-1)

        mixed_qkv_logical = torch.cat(
            [
                query.reshape(num_tokens, -1),
                key.reshape(num_tokens, -1),
                value.reshape(num_tokens, -1),
            ],
            dim=-1,
        )

        # The split above produces non-contiguous views into the interleaved
        # buffer.  Concatenating everything into a single flat tensor forces a
        # contiguous copy, then slicing back out gives contiguous q/k/v/z/b/a
        # tensors that downstream kernels require.  Doing this in one cat+slice
        # keeps torch.compile in a single Triton graph instead of emitting
        # separate copy kernels per tensor.  The original code used
        # rearrange(...).contiguous() on each tensor individually.
        fused = torch.cat(
            [
                mixed_qkv_logical.reshape(-1),
                z.reshape(-1),
                b.reshape(-1),
                a.reshape(-1),
            ],
            dim=0,
        )

        curr = 0
        qkv_numel = mixed_qkv_logical.numel()
        z_numel = z.numel()
        b_numel = b.numel()
        a_numel = a.numel()

        mixed_qkv_out = fused[curr : curr + qkv_numel].view(num_tokens, -1)
        curr += qkv_numel

        z_out = fused[curr : curr + z_numel].view(
            num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim
        )
        curr += z_numel

        b_out = fused[curr : curr + b_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )
        curr += b_numel

        a_out = fused[curr : curr + a_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )

        return mixed_qkv_out, z_out, b_out, a_out

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Split packed qkv into contiguous (1, seq, heads, dim) tensors.

        The original code used ``rearrange(x, "l (h d) -> 1 l h d", d=...)``
        followed by ``.contiguous()`` on each tensor.  This version flattens
        all three splits into a single buffer via ``torch.cat`` so that
        torch.compile emits one Triton copy kernel instead of three separate
        contiguous() calls.
        """
        if mixed_qkv is None:
            return None, None, None

        seq_len = mixed_qkv.shape[0]
        q_dim = self.key_dim // self.tp_size
        k_dim = self.key_dim // self.tp_size
        v_dim = self.value_dim // self.tp_size

        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

        fused = torch.cat(
            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
        )

        q_size = seq_len * q_dim
        k_size = seq_len * k_dim

        q_contig = fused[0:q_size]
        k_contig = fused[q_size : q_size + k_size]
        v_contig = fused[q_size + k_size :]

        query = q_contig.view(1, seq_len, -1, self.head_k_dim)
        key = k_contig.view(1, seq_len, -1, self.head_k_dim)
        value = v_contig.view(1, seq_len, -1, self.head_v_dim)

        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_method(hidden_states)

    def _output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Part 3: RMSNormGated + output linear projection.

        The RMSNormGated + quant sequence is eligible for fusion
        by the compilation pass when fuse_norm_quant is enabled.
        """
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        output, _ = self.out_proj(core_attn_out)
        return output

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """ROCm forward using AITER Triton fused projection+attention when
        available, otherwise falling back to the generic CUDA path.
        """
        if GDN_AITER_TRITON_AVAILABLE:
            num_tokens = hidden_states.size(0)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            projected_states_qkvz = projected_states_qkvz.view(num_tokens, -1)
            projected_states_ba = projected_states_ba.view(num_tokens, -1)
            core_attn_out = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            z = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=projected_states_qkvz.dtype,
                device=projected_states_qkvz.device,
            )

            torch.ops.vllm.qwen_gdn_attention_core(
                projected_states_qkvz,
                projected_states_ba,
                z,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
                use_aiter=True,
            )

            return self._output_projection(core_attn_out, z)
        else:
            return self.forward_cuda(hidden_states)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        mixed_qkvz, ba = self._in_projections(hidden_states)

        use_fused_gdn_decode = (
            self.enable_fused_gdn_decode
            and hidden_states.dtype == torch.bfloat16
            and self.norm.weight.dtype in (torch.bfloat16, torch.float32)
        )
        if use_fused_gdn_decode:
            layer_name = _encode_layer_name(self.prefix)
            # VLLM_GDN_BA_LATE_JOIN: the core op may join the aux-stream BA
            # GEMM itself; its input must stay alive until then.
            keep_alive = (
                {"ba_src": hidden_states}
                if GDN_BA_LATE_JOIN and self._ba_stream is not None
                else {}
            )
            # The core op writes or zeroes every row itself.
            core_attn_out = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            if self.gdn_out_mxfp8:
                # The core op writes out_proj's e4m3 activation and swizzled
                # scales; core_attn_out is only its bf16 scratch.
                hidden = core_attn_out.shape[1] * core_attn_out.shape[2]
                out_q = torch.empty(
                    (num_tokens, hidden),
                    dtype=torch.float8_e4m3fn,
                    device=hidden_states.device,
                )
                out_scale = torch.empty(
                    gdn_mxfp8_scale_numel(num_tokens, hidden),
                    dtype=torch.uint8,
                    device=hidden_states.device,
                )
                torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                    mixed_qkvz,
                    ba,
                    core_attn_out,
                    layer_name=layer_name,
                    out_q=out_q,
                    out_scale=out_scale,
                    **keep_alive,
                )
                output, _ = self.out_proj(
                    QuantizedActivation(
                        out_q,
                        out_scale,
                        hidden_states.dtype,
                        out_q.shape,
                        kMxfp8Dynamic,
                    )
                )
                return output
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                mixed_qkvz,
                ba,
                core_attn_out,
                layer_name=layer_name,
                **keep_alive,
            )
            output, _ = self.out_proj(core_attn_out.flatten(-2))
            return output

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self.split_ba(ba)

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b.contiguous(),
            a.contiguous(),
            core_attn_out,
            layer_name=_encode_layer_name(self.prefix),
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        return self._output_projection(core_attn_out, z)

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        # ============================================================
        # Part 2: Core Attention
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        z = torch.empty_like(core_attn_out)

        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        out, _ = self.out_proj(core_attn_out)
        return out

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert not hasattr(self, "in_proj_qkv"), "lora isn't supported on CPU."

        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)

        num_tokens = hidden_states.size(0)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.cpu_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.prefix),
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        out, _ = self.out_proj(core_attn_out)
        return out

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
        """Warm up GDN prefill kernels during V1 profiling.

        During V1 profile runs, ``_forward_core`` returns early because
        ``attn_metadata`` is ``None``, so the autotuned kernels used by
        ``chunk_gated_delta_rule`` (e.g. ``solve_tril``,
        ``chunk_scaled_dot_kkt``) are never invoked.  After profiling,
        vLLM allocates KV cache using most of the remaining GPU memory.
        When the first real inference triggers the autotuner it OOMs
        because there is not enough memory left for benchmarking.

        This method runs minimal forward passes through
        ``chunk_gated_delta_rule`` with small dummy tensors to force
        autotuning while GPU memory is still plentiful.  The autotuner
        results are cached globally, so only the first layer incurs
        actual benchmarking cost.

        All kernels including ``chunk_fwd_kernel_o`` now use a fixed
        ``BT = chunk_size`` (64).  A single warmup pass with T = 64
        is sufficient to populate the autotuner cache.

        The decode path uses ``gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule``
        which has fixed kernel parameters (no autotuning), so only the
        prefill (chunked) path needs warming up.
        """
        if self._prefill_kernels_warmed_up:
            return
        self._prefill_kernels_warmed_up = True

        device = qkv_or_qkvz.device
        dtype = qkv_or_qkvz.dtype
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        _, state_dtype = self.get_state_dtype()

        if (
            GDN_FI_VSPLIT
            and self.chunk_gated_delta_rule.updates_state_in_place(state_dtype)
            and self.head_k_dim == 128
            and self.head_v_dim == 128
        ):
            _gdn_vsplit_warmup(
                num_k_heads, num_v_heads, self.head_k_dim, dtype, state_dtype, device
            )
        if (
            GDN_MTP_CUDA_JIT
            and self.gdn_decode_kernel == "cuda"
            and current_platform.is_device_capability_family(100)
        ):
            gdn_mtp_cuda.enable()

        # All kernels use BT = chunk_size, so a single pass with T = chunk_size
        # is sufficient to populate every autotuner cache. Mirror the real
        # prefill path here: build q/k/v/g/beta via fused_post_conv_prep and
        # then run chunk_gated_delta_rule with in-kernel L2 norm disabled.
        # One chunk compiles the kernels of short prefills. FlashInfer runs
        # single sequences above _gdn_fi_non_cp_max_tokens() through its CP
        # kernels instead; warm them too, or they JIT-compile (seconds) at the
        # first long prefill under load. (The V-split warmup above sets that
        # threshold.)
        warmup_lengths = [FLA_CHUNK_SIZE]
        non_cp_max_tokens = _gdn_fi_non_cp_max_tokens()
        if self.gdn_prefill_backend == "flashinfer" and non_cp_max_tokens > 0:
            warmup_lengths.append(non_cp_max_tokens + FLA_CHUNK_SIZE)
        for T in warmup_lengths:
            dummy_mixed_qkv = torch.randn(
                T, qkv_or_qkvz.shape[-1] - v_dim, device=device, dtype=dtype
            )
            dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
            dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
            q, k, v, g, beta = fused_post_conv_prep(
                conv_output=dummy_mixed_qkv,
                a=dummy_a,
                b=dummy_b,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_k_heads=num_k_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                output_g_exp=self.chunk_gated_delta_rule.expects_exp_g,
            )
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            g = g.unsqueeze(0)
            beta = beta.unsqueeze(0)
            state = torch.zeros(
                1,
                num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
                device=device,
                dtype=state_dtype,
            )
            cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

            # CuteDSL kernels require metadata
            chunk_indices = None
            chunk_offsets = None
            if self.gdn_prefill_backend == "cutedsl":
                from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                    prepare_metadata_cutedsl,
                )

                chunk_indices, chunk_offsets = prepare_metadata_cutedsl(cu_seqlens, T)

            try:
                self.chunk_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    chunk_offsets=chunk_offsets,
                    use_qk_l2norm_in_kernel=False,
                )
            except Exception:
                logger.warning(
                    "GDN prefill kernel warmup (T=%d) failed for "
                    "layer %s. First inference may OOM due to "
                    "autotuner.",
                    T,
                    self.prefix,
                    exc_info=True,
                )
            else:
                logger.debug(
                    "GDN prefill kernel warmup (T=%d) completed for layer %s",
                    T,
                    self.prefix,
                )
            finally:
                del (
                    dummy_mixed_qkv,
                    q,
                    k,
                    v,
                    dummy_a,
                    dummy_b,
                    g,
                    beta,
                    state,
                    cu_seqlens,
                    chunk_indices,
                    chunk_offsets,
                )

        if (
            GDN_FUSED_CONV_PREP
            and self.chunk_gated_delta_rule.expects_exp_g
            and self.conv1d.bias is None
            and qkv_or_qkvz.shape[0] > 0
        ):
            # Compile both tile variants of gdn_fused_conv_prep with serving's
            # layout: the row-strided in_proj view, a/b as strided columns of
            # ba, conv state oriented like the KV cache.
            T = min(FLA_CHUNK_SIZE, qkv_or_qkvz.shape[0])
            conv_dim = qkv_or_qkvz.shape[-1] - v_dim
            conv_state = torch.zeros(
                66, *self.get_state_shape()[0], device=device, dtype=dtype
            )
            if not is_conv_state_dim_first():
                conv_state = conv_state.transpose(-1, -2)
            ba = torch.zeros(T, 2 * num_v_heads, device=device, dtype=dtype)
            if GDN_CONV_CUDA and current_platform.is_device_capability_family(100):
                enable_cuda_kernel()
            # Warm both metadata paths with one real sequence and empty tails.
            # tile=None also warms the CUDA kernel once enabled.
            for num_seqs in (1, 65):
                cache_indices = torch.arange(
                    1, num_seqs + 1, device=device, dtype=torch.int32
                )
                initial_state = torch.ones(num_seqs, device=device, dtype=torch.bool)
                cu_seqlens = torch.tensor(
                    [0] + [T] * num_seqs, device=device, dtype=torch.int32
                )
                for tile in (FUSED_CONV_TILE_SHORT, FUSED_CONV_TILE_LONG, None):
                    gdn_fused_conv_prep(
                        x=qkv_or_qkvz[:T, :conv_dim],
                        conv_weights=self.conv1d.weight.view(
                            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
                        ),
                        conv_state=conv_state,
                        cache_indices=cache_indices,
                        has_initial_state=initial_state,
                        cu_seqlens=cu_seqlens,
                        a=ba[:, num_v_heads:],
                        b=ba[:, :num_v_heads],
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        num_k_heads=num_k_heads,
                        head_k_dim=self.head_k_dim,
                        head_v_dim=self.head_v_dim,
                        tile=tile,
                    )
            del conv_state, ba

        torch.accelerator.empty_cache()

    def _forward_core_rocm(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """ROCm AITER fast path: conv1d + recurrent attention from packed
        qkvz/ba layout.

        For decode-only (no spec, no prefill) interleaved-GQA layouts,
        dispatches directly to ``_forward_core_decode_aiter``. Otherwise unpacks
        the packed layout and falls through to ``_forward_core``.

        Args:
            qkvz: packed [q, k, v, z] projection (num_tokens, qkvz_dim)
            ba:   packed [b, a] gating vectors    (num_tokens, 2*num_heads)
            z_out: **output** buffer for z        (num_tokens, num_heads,
                   head_dim); mutated in-place.
            core_attn_out: Pre-allocated output buffer for attention results.

        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            v_dim = core_attn_out.shape[-1] * core_attn_out.shape[-2]
            self._warmup_prefill_kernels(qkvz, v_dim)
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # The AITER fused reshape/conv kernel expects Qwen3-Next's interleaved
        # GQA layout. Qwen3.5 uses a non-interleaved q/k/v/z layout and must use
        # the generic path below to split/rearrange inputs correctly.
        if (
            self.gqa_interleaved_layout
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_aiter(
                qkvz=qkvz,
                ba=ba,
                z_out=z_out,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        core_attn_out.zero_()
        num_tokens_all = qkvz.shape[0]
        mixed_qkv, z, b, a = self.prepare_gdn_attention_core_inputs(
            qkvz, ba, num_tokens_all
        )
        z_out[:] = z
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
        )

    @staticmethod
    def _store_conv_checkpoint(
        conv_input: torch.Tensor,
        conv_state: torch.Tensor,
        width: int,
        checkpoint: GDNPrefillCheckpointMetadata,
    ) -> None:
        """Write each checkpoint's conv state, the ``width - 1`` conv inputs
        before the checkpoint token, into its checkpoint slot.

        ``conv_input``: the non-spec conv input rows ``[tokens, dim]``;
        ``conv_state``: the ``[..., dim, state_len]`` view of the conv pool.
        With MTP, ``state_len = width - 1 + num_spec``, but the prefill and
        decode conv kernels read the initial state from columns
        ``[0, width - 1)`` (newest last), so the checkpoint goes there.
        """
        store_conv_checkpoint(
            conv_input,
            conv_state,
            width - 1,
            checkpoint.conv_token_indices,
            checkpoint.checkpoint_state_indices,
        )

    def _run_prefill_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_start_loc_i32: torch.Tensor | None,
        max_seqlen: int,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor | None,
        no_initial_state: torch.Tensor | None,
        chunk_indices: torch.Tensor | None,
        chunk_offsets: torch.Tensor | None,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """One chunk-kernel call that reads and writes the ssm pool rows at
        ``state_indices`` (int32). ``has_initial_state``/``no_initial_state``
        None: every sequence starts from its pool row.
        """
        if self.chunk_gated_delta_rule.updates_state_in_place(ssm_state.dtype):
            if has_initial_state is not None:
                zero_fresh_state_rows(ssm_state, state_indices, has_initial_state)
            o, _ = self.chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=ssm_state,
                output_final_state=True,
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=False,
                core_attn_out=out,
                state_indices=state_indices,
                max_seqlen=max_seqlen,
                cu_seqlens_i32=query_start_loc_i32,
            )
            return o
        state_indices_i64 = state_indices.to(torch.int64)
        initial_state = ssm_state[state_indices_i64]
        if no_initial_state is not None:
            initial_state[no_initial_state, ...] = 0
        o, last_state = self.chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=query_start_loc,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=False,
            core_attn_out=out,
        )
        ssm_state[state_indices_i64] = last_state.to(ssm_state.dtype)
        return o

    def _chunk_prefill_with_checkpoint(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_state: torch.Tensor,
        checkpoint: GDNPrefillCheckpointMetadata,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """Prefill chunk whose checkpointed sequences also leave their state at
        the checkpoint in the checkpoint slots (see
        ``GDNPrefillCheckpointMetadata``). Same results as running each
        checkpointed sequence as two consecutive chunks split there.
        """
        # 1. Heads (and unsplit sequences) from their initial state; the tails
        # here are throwaway sequences on the checkpoint slots.
        o = self._run_prefill_chunk(
            q,
            k,
            v,
            g,
            beta,
            ssm_state,
            checkpoint.split_query_start_loc,
            checkpoint.split_query_start_loc_i32,
            checkpoint.split_max_seqlen,
            checkpoint.split_state_indices,
            checkpoint.split_has_initial_state,
            checkpoint.split_no_initial_state,
            checkpoint.split_chunk_indices,
            checkpoint.split_chunk_offsets,
            out,
        )
        # 2. The running slots now hold the checkpoint states: snapshot them.
        copy_state_rows(
            ssm_state,
            checkpoint.checkpoint_state_indices,
            checkpoint.tail_state_indices,
        )
        # 3. Rerun the tails from the checkpoint states. One tail is a token
        # range: run it on slices and write its outputs in place.
        idx = checkpoint.tail_token_indices
        if checkpoint.tail_token_range is not None:
            start, end = checkpoint.tail_token_range
            q_t, k_t, v_t, g_t, beta_t = (t[:, start:end] for t in (q, k, v, g, beta))
            out_t: torch.Tensor | None = o[0, start:end]
        else:
            assert idx is not None
            q_t, k_t, v_t, g_t, beta_t = gather_prefill_rows(idx, q, k, v, g, beta)
            out_t = None
        o_tail = self._run_prefill_chunk(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            ssm_state,
            checkpoint.tail_query_start_loc,
            checkpoint.tail_query_start_loc_i32,
            checkpoint.tail_max_seqlen,
            checkpoint.tail_state_indices,
            None,
            None,
            checkpoint.tail_chunk_indices,
            checkpoint.tail_chunk_offsets,
            out_t,
        )
        if out_t is None:
            assert idx is not None
            o.index_copy_(1, idx, o_tail.to(o.dtype))
        return o

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        spec_done: bool = False,
    ):
        """Core conv1d + recurrent attention (standard path).

        Args:
            mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)
            b: beta gating vector                   (num_tokens, num_heads)
            a: alpha gating vector                  (num_tokens, num_heads)
            core_attn_out: Pre-allocated output buffer for attention results.
            spec_done: the caller already ran the spec rows of this mixed
                batch (contiguous spec block only); process the rest.

        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if (
            self.enable_packed_recurrent_decode
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        # When the builder found the spec and non-spec tokens in two contiguous
        # blocks, slice instead of gathering and let the recurrent kernels write
        # their outputs into core_attn_out in place.
        spec_slice: slice | None = None
        non_spec_slice: slice | None = None
        if attn_metadata.spec_token_start is not None:
            assert attn_metadata.non_spec_token_start is not None
            spec_start = attn_metadata.spec_token_start
            non_spec_start = attn_metadata.non_spec_token_start
            spec_slice = slice(
                spec_start, spec_start + attn_metadata.num_spec_decode_tokens
            )
            non_spec_slice = slice(
                non_spec_start,
                non_spec_start
                + attn_metadata.num_prefill_tokens
                + attn_metadata.num_decode_tokens,
            )
        assert not spec_done or (
            spec_slice is not None and attn_metadata.num_prefills > 0
        )
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        if GDN_HOST_TRIM:
            conv_state, conv_weights = _host_layer_views(self)
            mixed_qkv = _rows(mixed_qkv, num_actual_tokens)
            b = _rows(b, num_actual_tokens)
            a = _rows(a, num_actual_tokens)
        else:
            # conv_state must be (..., dim, width-1) for the conv kernels.
            # DS layout stores it that way directly; SD layout needs a
            # transpose.
            conv_state = (
                self_kv_cache[0]
                if is_conv_state_dim_first()
                else self_kv_cache[0].transpose(-1, -2)
            )

            mixed_qkv = mixed_qkv[:num_actual_tokens]
            b = b[:num_actual_tokens]
            a = a[:num_actual_tokens]

            # 1. Convolution sequence transformation
            conv_weights = self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                a_spec = a
                b_spec = b
                mixed_qkv_non_spec = None
            elif spec_slice is not None and spec_done and GDN_HOST_TRIM:
                # The caller ran the spec rows: their views are unused.
                mixed_qkv_spec = a_spec = b_spec = None
                mixed_qkv_non_spec = mixed_qkv[non_spec_slice]
            elif spec_slice is not None:
                mixed_qkv_spec = mixed_qkv[spec_slice]
                a_spec = a[spec_slice]
                b_spec = b[spec_slice]
                mixed_qkv_non_spec = mixed_qkv[non_spec_slice]
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                a_spec = a.index_select(0, spec_token_indx)
                b_spec = b.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None and not spec_done:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        # Prefill rows without peeled-off decodes (FlashInfer backend, which
        # takes exp(g)) run conv1d and the post-conv prep in one kernel below.
        fused_conv_prep = (
            GDN_FUSED_CONV_PREP
            and attn_metadata.num_prefills > 0
            and not (spec_sequence_masks is None and attn_metadata.num_decodes > 0)
            and self.chunk_gated_delta_rule.expects_exp_g
            and self.conv1d.bias is None
            and non_spec_query_start_loc is not None
        )
        if (
            attn_metadata.num_prefills > 0
            and attn_metadata.prefill_checkpoint is not None
        ):
            # Both prefill conv paths read the pre-conv inputs; the checkpoint's
            # conv window is taken from them before either conv runs.
            assert mixed_qkv_non_spec is not None
            self._store_conv_checkpoint(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights.size(-1),
                attn_metadata.prefill_checkpoint,
            )
        if fused_conv_prep:
            pass  # mixed_qkv_non_spec stays pre-conv for gdn_fused_conv_prep
        elif attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(
            None if spec_done else mixed_qkv_spec
        )

        # Split mixed non-spec-decode+prefill to process independently
        split_non_spec = (
            spec_sequence_masks is None
            and attn_metadata.num_prefills > 0
            and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens

        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if non_spec_slice is not None:
                a_non_spec = a[non_spec_slice]
                b_non_spec = b[non_spec_slice]
            elif spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            if split_non_spec:
                conv_output_prefill = mixed_qkv_non_spec[num_decode_tokens:]
                a_prefill = a_non_spec[num_decode_tokens:]
                b_prefill = b_non_spec[num_decode_tokens:]
            else:
                conv_output_prefill = mixed_qkv_non_spec
                a_prefill = a_non_spec
                b_prefill = b_non_spec

            if fused_conv_prep:
                (
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g_non_spec,
                    beta_non_spec,
                ) = gdn_fused_conv_prep(
                    x=mixed_qkv_non_spec,
                    conv_weights=conv_weights,
                    conv_state=conv_state,
                    cache_indices=non_spec_state_indices_tensor,
                    has_initial_state=has_initial_state,
                    cu_seqlens=non_spec_query_start_loc,
                    a=a_prefill,
                    b=b_prefill,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    num_k_heads=self.num_k_heads // self.tp_size,
                    head_k_dim=self.head_k_dim,
                    head_v_dim=self.head_v_dim,
                )
            else:
                (
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g_non_spec,
                    beta_non_spec,
                ) = fused_post_conv_prep(
                    conv_output=conv_output_prefill,
                    a=a_prefill,
                    b=b_prefill,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    num_k_heads=self.num_k_heads // self.tp_size,
                    head_k_dim=self.head_k_dim,
                    head_v_dim=self.head_v_dim,
                    apply_l2norm=True,
                    output_g_exp=self.chunk_gated_delta_rule.expects_exp_g,
                )
            query_non_spec = query_non_spec.unsqueeze(0)
            key_non_spec = key_non_spec.unsqueeze(0)
            value_non_spec = value_non_spec.unsqueeze(0)
            g_non_spec = g_non_spec.unsqueeze(0)
            beta_non_spec = beta_non_spec.unsqueeze(0)
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            g_non_spec = None
            beta_non_spec = None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None and not spec_done:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a_spec,
                    b=b_spec,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_spec_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                    out=None if spec_slice is None else core_attn_out[spec_slice],
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process non-spec-decode part
        if split_non_spec:
            query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec[:num_decode_tokens]  # type: ignore[index]
            )
            core_attn_out_decode, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a[:num_decode_tokens],
                b=b[:num_decode_tokens],
                dt_bias=self.dt_bias,
                q=query_decode,
                k=key_decode,
                v=value_decode,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                    : attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_decode = None

        # 2.3: Process the remaining part (prefill chunk, or non-spec decode-only)
        if attn_metadata.num_prefills > 0:
            # State indices, initial-state mask and cu_seqlens for the chunk
            # kernel are precomputed by the metadata builder (the prefill tail
            # when decodes are peeled off, else the full non-spec batch), so they
            # don't need to be re-derived per layer.
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            prefill_query_start_loc = attn_metadata.prefill_query_start_loc
            if attn_metadata.prefill_query_start_loc_i64 is not None:
                prefill_query_start_loc = attn_metadata.prefill_query_start_loc_i64
            non_spec_out = (
                None if non_spec_slice is None else core_attn_out[non_spec_slice]
            )
            if attn_metadata.prefill_checkpoint is not None:
                core_attn_out_non_spec = self._chunk_prefill_with_checkpoint(
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g_non_spec,
                    beta_non_spec,
                    ssm_state,
                    attn_metadata.prefill_checkpoint,
                    non_spec_out,
                )
            elif self.chunk_gated_delta_rule.updates_state_in_place(ssm_state.dtype):
                # FlashInfer reads and updates the pool rows in place through
                # state_indices; only the rows of sequences without initial
                # state are zeroed first. Batches without spec rows index
                # block_table[:, 0], a strided view; FlashInfer and the zeroing
                # kernel need unit stride.
                if not GDN_HOST_TRIM:
                    prefill_state_indices = prefill_state_indices.contiguous()
                    zero_fresh_state_rows(
                        ssm_state, prefill_state_indices, prefill_has_initial_state
                    )
                else:
                    prefill_state_indices = _contig(prefill_state_indices)
                    # The builder saw on the host that every prefill row
                    # has an initial state: zeroing would be a no-op.
                    if not attn_metadata.prefill_all_initial_state:
                        zero_fresh_state_rows(
                            ssm_state, prefill_state_indices, prefill_has_initial_state
                        )
                core_attn_out_non_spec, _ = self.chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    output_final_state=True,
                    cu_seqlens=prefill_query_start_loc,
                    use_qk_l2norm_in_kernel=False,
                    core_attn_out=non_spec_out,
                    state_indices=prefill_state_indices,
                    max_seqlen=attn_metadata.prefill_max_seqlen,
                    cu_seqlens_i32=attn_metadata.prefill_query_start_loc,
                )
            else:
                if attn_metadata.prefill_state_indices_i64 is not None:
                    # int64 indices and the inverted mask were converted once
                    # per step by the metadata builder.
                    prefill_state_indices = attn_metadata.prefill_state_indices_i64
                    initial_state = ssm_state[prefill_state_indices]
                    initial_state[attn_metadata.prefill_no_initial_state, ...] = 0
                else:
                    initial_state = ssm_state[prefill_state_indices]
                    initial_state[~prefill_has_initial_state, ...] = 0
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = self.chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=prefill_query_start_loc,
                    chunk_indices=attn_metadata.chunk_indices,
                    chunk_offsets=attn_metadata.chunk_offsets,
                    use_qk_l2norm_in_kernel=False,
                    core_attn_out=non_spec_out,
                )
                # Init cache
                ssm_state[prefill_state_indices] = last_recurrent_state.to(
                    ssm_state.dtype
                )

            if split_non_spec:
                # Stitch the peeled decode outputs in front of the prefill
                # outputs (decode-first order).
                core_attn_out_non_spec = torch.cat(
                    [core_attn_out_decode, core_attn_out_non_spec], dim=1
                )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            # With spec_slice set, both blocks were written in place above.
            if spec_slice is None:
                core_attn_out.index_copy_(
                    0, spec_token_indx, core_attn_out_spec.squeeze(0)
                )
                core_attn_out.index_copy_(
                    0, non_spec_token_indx, core_attn_out_non_spec.squeeze(0)
                )
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    def _forward_core_decode_aiter(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        mixed_qkv_non_spec, b, a = (
            gdn_aiter_fused_reshape_causal_conv1d_update_single_token(
                qkvz,
                attn_metadata.num_actual_tokens,
                self.num_k_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
                ba,
                z_out,
                core_attn_out,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        )

        # 2. Recurrent attention
        gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            qkv=mixed_qkv_non_spec,
            key_dim=self.key_dim // self.tp_size,
            value_dim=self.value_dim // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],  # type: ignore[index]
            ssm_state_indices=non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
            core_attn_out=core_attn_out.reshape(-1),
        )

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """Core attention computation with a packed non-spec decode fast path."""
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            validate_data=False,
        )
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            use_qk_l2norm_in_kernel=True,
        )
        return

    def _forward_core_decode_spec_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> bool:
        """Conv + recurrence of the spec-decode rows into ``core_attn_out``.
        Returns whether the gated norm was applied too (CUDA MTP kernel); the
        Triton recurrence leaves it to the caller's norm launch.
        """
        num_requests = attn_metadata.num_spec_decodes
        num_actual_tokens = attn_metadata.num_actual_tokens
        if GDN_HOST_TRIM:
            conv_state, conv_weights = _host_layer_views(self)
            conv_slots, _, accepted, cu_seqlens, max_query_len = _host_spec_args(
                attn_metadata
            )
            # A mixed batch passes its spec rows only (fewer than
            # num_actual_tokens): [:num_actual_tokens] would be the same view.
            n = num_actual_tokens
            mixed_qkv = causal_conv1d_update(
                _rows(mixed_qkv, n),
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=conv_slots,
                num_accepted_tokens=accepted,
                query_start_loc=cu_seqlens,
                max_query_len=max_query_len,
                validate_data=False,
            )
            return self._forward_core_decode_spec_post_conv_fused_norm(
                mixed_qkv=mixed_qkv,
                b=_rows(b, n),
                a=_rows(a, n),
                output_gate=_rows(output_gate, n),
                core_attn_out=_rows(core_attn_out, n),
                attn_metadata=attn_metadata,
            )
        state_indices = attn_metadata.spec_state_indices_tensor
        cu_seqlens = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert state_indices is not None
        assert cu_seqlens is not None
        assert num_accepted_tokens is not None

        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv = causal_conv1d_update(
            mixed_qkv[:num_actual_tokens],
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=state_indices[:num_requests, 0],
            num_accepted_tokens=num_accepted_tokens[:num_requests],
            query_start_loc=cu_seqlens[: num_requests + 1],
            max_query_len=state_indices.size(1),
            validate_data=False,
        )
        return self._forward_core_decode_spec_post_conv_fused_norm(
            mixed_qkv=mixed_qkv,
            b=b[:num_actual_tokens],
            a=a[:num_actual_tokens],
            output_gate=output_gate[:num_actual_tokens],
            core_attn_out=core_attn_out[:num_actual_tokens],
            attn_metadata=attn_metadata,
        )

    def _forward_core_decode_spec_post_conv_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> bool:
        """Spec-decode recurrence; returns whether the gated norm was applied
        (CUDA MTP kernel) or is left to the caller (Triton recurrence).
        """
        # VLLM_GDN_BA_LATE_JOIN: the deferred aux-stream join (no-op otherwise);
        # the recurrence is the first reader of b/a.
        self._in_proj_ba_join()
        num_requests = attn_metadata.num_spec_decodes
        if GDN_HOST_TRIM:
            _, state_slots, accepted, cu_seqlens, _ = _host_spec_args(attn_metadata)
        else:
            state_indices = attn_metadata.spec_state_indices_tensor
            cu_seqlens = attn_metadata.spec_query_start_loc
            num_accepted_tokens = attn_metadata.num_accepted_tokens
            assert state_indices is not None
            assert cu_seqlens is not None
            assert num_accepted_tokens is not None
            state_slots = state_indices[:num_requests]
            cu_seqlens = cu_seqlens[: num_requests + 1]
            accepted = num_accepted_tokens[:num_requests]
        if num_requests <= GDN_MTP_TRITON_MAX_REQUESTS:
            gdn_mtp_recurrence(
                mixed_qkv,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state_slots,
                cu_seqlens,
                accepted,
                self.kv_cache[1],
                core_attn_out,
                scale=self.head_k_dim**-0.5,
            )
            return False
        args = (
            mixed_qkv,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state_slots,
            cu_seqlens,
            accepted,
            self.kv_cache[1],
            output_gate,
            self.norm.weight,
            core_attn_out,
            self.head_k_dim**-0.5,
            self.layer_norm_epsilon,
            self.norm.activation,
        )
        if gdn_mtp_cuda.ready() and gdn_mtp_cuda.gdn_mtp_cuda(*args):
            return True
        ops.fused_gdn_decode_post_conv_mtp(*args)
        return True

    def _forward_core_decode_spec_one_launch(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
        out_q: torch.Tensor,
        out_scale: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        """``_forward_core_decode_spec_fused_norm`` + ``_gated_norm_mxfp8`` of
        a decode-only spec batch in one launch (``gdn_mtp_fused_decode``):
        conv window roll, recurrence, gated norm and out_proj's MXFP8
        activation. Rows past the valid token count get zeros.
        """
        state_indices = attn_metadata.spec_state_indices_tensor
        cu_seqlens = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert state_indices is not None
        assert cu_seqlens is not None
        assert num_accepted_tokens is not None
        assert state_indices.size(1) <= GDN_FUSED_DECODE_MAX_TOKENS
        assert self._fused_decode_counters is not None
        num_requests = attn_metadata.num_spec_decodes
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        gdn_mtp_fused_decode(
            mixed_qkv,
            a,
            b,
            conv_state,
            self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            ),
            self.A_log,
            self.dt_bias,
            state_indices[:num_requests],
            cu_seqlens[: num_requests + 1],
            num_accepted_tokens[:num_requests],
            self.kv_cache[1],
            core_attn_out,
            output_gate,
            self.norm.weight,
            self.norm.eps,
            self.norm.activation,
            out_q,
            out_scale,
            self._fused_decode_counters,
            scale=self.head_k_dim**-0.5,
            null_block_id=NULL_BLOCK_ID,
        )

    def _forward_core_fused_norm_packed(
        self,
        mixed_qkvz: torch.Tensor,
        ba: torch.Tensor,
        core_attn_out: torch.Tensor,
        out_q: torch.Tensor | None = None,
        out_scale: torch.Tensor | None = None,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            self._in_proj_ba_join()
            self._warmup_prefill_kernels(mixed_qkvz[:, :qkv_size], 0)
            # The outputs are uninitialized; hand out_proj the zeros they
            # would have been allocated with.
            if out_q is not None:
                assert out_scale is not None
                out_q.zero_()
                out_scale.zero_()
            else:
                core_attn_out.zero_()
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)
        mixed_qkv, output_gate_flat = mixed_qkvz.split(
            [qkv_size, self.value_dim // self.tp_size], dim=-1
        )
        output_gate = output_gate_flat.reshape(
            output_gate_flat.size(0), -1, self.head_v_dim
        )
        b, a = self.split_ba(ba)
        self._forward_core_fused_norm(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            output_gate=output_gate,
            core_attn_out=core_attn_out,
            out_q=out_q,
            out_scale=out_scale,
        )
        # A deferred join is always taken before the first read of b/a; this
        # only guarantees that no path returns with the aux stream unjoined.
        self._in_proj_ba_join()

    def _can_use_fused_gdn_mtp_decode(
        self, attn_metadata: GDNAttentionMetadata
    ) -> bool:
        state_indices = attn_metadata.spec_state_indices_tensor
        return (
            attn_metadata.spec_sequence_masks is not None
            and attn_metadata.num_decodes == 0
            and attn_metadata.num_spec_decodes > 0
            and self.kv_cache[1].dtype in FUSED_GDN_STATE_DTYPES
            and self.gdn_decode_kernel == "cuda"
            and self.num_v_heads % self.num_k_heads == 0
            and self.num_v_heads // self.num_k_heads in (1, 2, 3, 4, 8)
            and state_indices is not None
            and state_indices.size(1) <= MAX_FUSED_GDN_MTP_TOKENS
            and hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")
        )

    def _rms_norm_gated_cuda(
        self,
        x: torch.Tensor,
        output_gate: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
            layer_norm_fwd,
        )

        x_shape = x.shape
        assert output_gate.shape == x_shape
        assert out.shape == x_shape
        x_2d = x.reshape(-1, x_shape[-1])
        output_gate_2d = output_gate.reshape(-1, x_shape[-1])
        out_2d = out.reshape(-1, x_shape[-1])
        assert x_2d.stride(-1) == 1
        assert output_gate_2d.stride(-1) == 1
        assert out_2d.stride(-1) == 1
        layer_norm_fwd(
            x_2d,
            self.norm.weight.contiguous(),
            self.norm.bias,
            self.norm.eps,
            z=output_gate_2d,
            out=out_2d,
            group_size=(
                x_shape[-1] if self.norm.group_size is None else self.norm.group_size
            ),
            norm_before_gate=self.norm.norm_before_gate,
            is_rms_norm=True,
            activation=self.norm.activation,
        )

    def _rms_norm_gated_strided_gate_cuda(
        self,
        x: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> None:
        """In-place ``_rms_norm_gated_cuda(x, output_gate, x)`` for [T, HV, V].

        Normalizes T rows of HV * V columns with one group per head (group size
        V), so every (token, head) sees the same math as the [T * HV, V] launch.
        The output gate, a row-strided view of mixed_qkvz, is read in place
        instead of first being copied into a compact [T * HV, V] tensor. The
        kernel indexes the weight per group, so it gets norm.weight tiled across
        heads (8 KiB, rebuilt per call so weight reloads are always honored).
        The launch uses more rows and warps per CTA than the heuristic for
        [T * HV, V] rows once T is large (``gdn_norm_launch_config``).
        """
        from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
            layer_norm_fwd,
        )

        num_tokens, num_heads, head_dim = x.shape
        assert output_gate.shape == x.shape
        x_2d = x.view(num_tokens, num_heads * head_dim)
        # view, not reshape: merging (HV, V) of the gate must not copy.
        output_gate_2d = output_gate.view(num_tokens, num_heads * head_dim)
        assert output_gate_2d.stride(-1) == 1
        rows_per_block, num_warps = gdn_norm_launch_config(num_tokens, x.device)
        layer_norm_fwd(
            x_2d,
            self.norm.weight.repeat(num_heads),
            self.norm.bias,
            self.norm.eps,
            z=output_gate_2d,
            out=x_2d,
            group_size=(
                head_dim if self.norm.group_size is None else self.norm.group_size
            ),
            norm_before_gate=self.norm.norm_before_gate,
            is_rms_norm=True,
            activation=self.norm.activation,
            rows_per_block=rows_per_block,
            num_warps=num_warps,
        )

    def _gated_norm_mxfp8(
        self,
        core_attn_out: torch.Tensor,
        output_gate: torch.Tensor,
        out_q: torch.Tensor,
        out_scale: torch.Tensor,
        norm_rows: tuple[int, int],
        num_valid: int | torch.Tensor,
    ) -> None:
        """Gated RMSNorm of ``norm_rows`` plus out_proj's MXFP8 activation of
        every valid row; rows ``>= num_valid`` are zero-filled unread.
        """
        assert self.norm.bias is None and self.norm.norm_before_gate
        assert self.norm.group_size in (None, self.head_v_dim)
        gdn_gated_norm_mxfp8(
            core_attn_out,
            output_gate,
            self.norm.weight,
            self.norm.eps,
            self.norm.activation,
            out_q,
            out_scale,
            norm_rows,
            num_valid,
        )

    def _forward_core_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
        out_q: torch.Tensor | None = None,
        out_scale: torch.Tensor | None = None,
    ) -> None:
        """Core + gated RMSNorm into ``core_attn_out``, or, when ``out_q`` and
        ``out_scale`` are given, into out_proj's swizzled MXFP8 activation, with
        ``core_attn_out`` as bf16 scratch. The outputs are uninitialized: every
        row is either written by the kernels or zeroed here.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)
        quantize = out_q is not None
        assert quantize == (out_scale is not None)
        if (
            self._can_use_fused_gdn_mtp_decode(attn_metadata)
            and attn_metadata.num_prefills == 0
        ):
            if (
                quantize
                and self._fused_decode_counters is not None
                and attn_metadata.num_spec_decodes <= GDN_FUSED_DECODE_MAX_REQUESTS
                and attn_metadata.spec_state_indices_tensor is not None
                and attn_metadata.spec_state_indices_tensor.size(1)
                <= GDN_FUSED_DECODE_MAX_TOKENS
            ):
                self._in_proj_ba_join()
                self._forward_core_decode_spec_one_launch(
                    mixed_qkv=mixed_qkv,
                    b=b,
                    a=a,
                    output_gate=output_gate,
                    core_attn_out=core_attn_out,
                    out_q=out_q,  # type: ignore[arg-type]
                    out_scale=out_scale,  # type: ignore[arg-type]
                    attn_metadata=attn_metadata,
                )
                return
            if not quantize:
                # The MTP kernels skip FULL-graph padding requests.
                core_attn_out.zero_()
            normalized = self._forward_core_decode_spec_fused_norm(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                output_gate=output_gate,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )
            if quantize:
                # Only the MTP rows of real requests are valid; their count is
                # on the device (FULL-graph replay pads the requests).
                num_spec = attn_metadata.num_spec_decodes
                assert attn_metadata.spec_query_start_loc is not None
                self._gated_norm_mxfp8(
                    core_attn_out,
                    output_gate,
                    out_q,  # type: ignore[arg-type]
                    out_scale,  # type: ignore[arg-type]
                    norm_rows=(0, 0 if normalized else core_attn_out.shape[0]),
                    num_valid=attn_metadata.spec_query_start_loc[
                        num_spec : num_spec + 1
                    ],
                )
            elif not normalized:
                self._rms_norm_gated_strided_gate_cuda(core_attn_out, output_gate)
            return

        self._in_proj_ba_join()
        num_actual_tokens = attn_metadata.num_actual_tokens
        if attn_metadata.num_prefills > 0:
            # Mixed batches run eagerly and the core writes every token row.
            num_written = (
                attn_metadata.num_spec_decode_tokens
                + attn_metadata.num_prefill_tokens
                + attn_metadata.num_decode_tokens
            )
        else:
            # Decode-only batches may replay FULL graphs, whose padding rows
            # these kernels are not known to write: zero first.
            core_attn_out.zero_()
            num_written = num_actual_tokens
        norm_rows = (0, num_written)
        if (
            attn_metadata.num_prefills > 0
            and attn_metadata.spec_token_start is not None
            and self._can_use_fused_gdn_mtp_decode(attn_metadata)
        ):
            # The contiguous spec block goes through the MTP decode kernels
            # (gating and recurrence, plus the gated norm in the CUDA kernel,
            # as in decode-only batches); the rest through the prefill path,
            # with b/a read strided.
            assert attn_metadata.non_spec_token_start is not None
            spec_start = attn_metadata.spec_token_start
            spec_rows = slice(
                spec_start, spec_start + attn_metadata.num_spec_decode_tokens
            )

            def spec_half() -> bool:
                return self._forward_core_decode_spec_fused_norm(
                    mixed_qkv=mixed_qkv[spec_rows],
                    b=b[spec_rows],
                    a=a[spec_rows],
                    output_gate=output_gate[spec_rows],
                    core_attn_out=core_attn_out[spec_rows],
                    attn_metadata=attn_metadata,
                )

            def prefill_half() -> None:
                self._forward_core(
                    mixed_qkv=mixed_qkv,
                    b=b,
                    a=a,
                    core_attn_out=core_attn_out,
                    spec_done=True,
                )

            if GDN_MIXED_FORK and not torch.cuda.is_current_stream_capturing():
                # Both halves only read mixed_qkv/b/a/output_gate, and write
                # disjoint rows of core_attn_out and disjoint conv/SSM slots.
                spec_stream, prefill_stream, fork, join = _gdn_mixed_fork_streams()
                main = torch.cuda.current_stream()
                fork.record(main)
                if attn_metadata.num_prefill_tokens >= GDN_MIXED_FORK_PREFILL_FIRST:
                    with torch.cuda.stream(prefill_stream):
                        fork.wait(prefill_stream)
                        prefill_half()
                        join.record(prefill_stream)
                    spec_normalized = spec_half()
                else:
                    with torch.cuda.stream(spec_stream):
                        fork.wait(spec_stream)
                        spec_normalized = spec_half()
                        join.record(spec_stream)
                    prefill_half()
                join.wait(main)
            else:
                spec_normalized = spec_half()
                prefill_half()
            if spec_normalized:
                non_spec_start = attn_metadata.non_spec_token_start
                norm_rows = (
                    non_spec_start,
                    non_spec_start
                    + attn_metadata.num_prefill_tokens
                    + attn_metadata.num_decode_tokens,
                )
        else:
            # VLLM_GDN_HOST_TRIM2: when the fused conv prep is the only reader
            # of b/a (prefill-only batch), it takes the row-strided views as
            # they are; otherwise they are made compact as before.
            strided_ba = (
                GDN_HOST_TRIM2
                and attn_metadata.spec_sequence_masks is None
                and attn_metadata.num_decodes == 0
                and _fused_conv_prep_applies(self, attn_metadata)
            )
            self._forward_core(
                mixed_qkv=mixed_qkv,
                b=b if strided_ba else b.contiguous(),
                a=a if strided_ba else a.contiguous(),
                core_attn_out=core_attn_out,
            )
        if quantize:
            self._gated_norm_mxfp8(
                core_attn_out,
                output_gate,
                out_q,  # type: ignore[arg-type]
                out_scale,  # type: ignore[arg-type]
                norm_rows=norm_rows,
                num_valid=num_written,
            )
            return
        norm_lo, norm_hi = norm_rows
        if attn_metadata.num_prefills > 0:
            # Read the strided output gate in place instead of copying it
            # compact. Decode-only batches (FULL graphs) keep the [T*HV, V]
            # launch.
            self._rms_norm_gated_strided_gate_cuda(
                core_attn_out[norm_lo:norm_hi],
                output_gate[norm_lo:norm_hi],
            )
            # Only the piecewise-graph padding rows are left to zero.
            core_attn_out[num_written:].zero_()
        else:
            self._rms_norm_gated_cuda(
                core_attn_out[:num_actual_tokens],
                output_gate[:num_actual_tokens],
                core_attn_out[:num_actual_tokens],
            )


@eager_break_during_capture
def qwen_gdn_attention_core(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    use_aiter: bool = False,
) -> None:
    """Custom op dispatching to _forward_core or _forward_core_rocm.

    Handles conv1d + recurrent attention only; input/output projections
    are performed by the caller.

    When ``use_aiter=False`` (standard path):
        qkv_or_qkvz is [q, k, v], b_or_ba is b, a_or_z_out is a (read-only).
    When ``use_aiter=True`` (AITER Triton path, ROCm only):
        qkv_or_qkvz is [q, k, v, z], b_or_ba is [b, a], a_or_z_out is the
        z output buffer (mutated in-place).

    ``core_attn_out`` is always mutated in-place.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if use_aiter:
        self._forward_core_rocm(
            qkvz=qkv_or_qkvz,
            ba=b_or_ba,
            z_out=a_or_z_out,
            core_attn_out=core_attn_out,
        )
    else:
        self._forward_core(
            mixed_qkv=qkv_or_qkvz,
            b=b_or_ba,
            a=a_or_z_out,
            core_attn_out=core_attn_out,
        )


direct_register_custom_op(
    op_name="qwen_gdn_attention_core",
    op_func=qwen_gdn_attention_core,
    mutates_args=["a_or_z_out", "core_attn_out"],
)


@eager_break_during_capture
def qwen_gdn_attention_core_fused_norm_packed(
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    out_q: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    ba_src: torch.Tensor | None = None,
) -> None:
    """``out_q``/``out_scale``: out_proj's e4m3 activation and flat swizzled
    UE8M0 scales, written instead of the normed bf16 output (``core_attn_out``
    is then scratch). ``ba_src`` (VLLM_GDN_BA_LATE_JOIN): the aux-stream BA
    GEMM's input, passed only to keep it alive until this op joins that stream.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core_fused_norm_packed(
        mixed_qkvz=mixed_qkvz,
        ba=ba,
        core_attn_out=core_attn_out,
        out_q=out_q,
        out_scale=out_scale,
    )


direct_register_custom_op(
    op_name="qwen_gdn_attention_core_fused_norm_packed",
    op_func=qwen_gdn_attention_core_fused_norm_packed,
    mutates_args=["core_attn_out", "out_q", "out_scale"],
)


def gdn_in_proj_ba_fork(
    hidden_states: torch.Tensor,
    ba: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Writes ``ba = hidden_states @ in_proj_ba.weight.T``, on the aux stream
    when enabled; ``gdn_in_proj_ba_join`` makes the current stream wait for it.
    """
    layer_name = _resolve_layer_name(layer_name)
    self = get_forward_context().no_compile_layers[layer_name]
    self._in_proj_ba_fork(hidden_states, ba)


direct_register_custom_op(
    op_name="gdn_in_proj_ba_fork",
    op_func=gdn_in_proj_ba_fork,
    mutates_args=["ba"],
)


def gdn_in_proj_ba_join(
    ba: torch.Tensor,
    hidden_states: torch.Tensor,
    mixed_qkvz: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Current stream waits for ``gdn_in_proj_ba_fork``'s aux-stream GEMM.
    ``ba`` is declared mutated to order its readers after the join;
    ``hidden_states`` and ``mixed_qkvz`` are only dependencies. With
    VLLM_GDN_BA_LATE_JOIN, FULL graphs leave the join to the fused core op.
    """
    layer_name = _resolve_layer_name(layer_name)
    self = get_forward_context().no_compile_layers[layer_name]
    if self._ba_join_deferred(ba):
        return
    self._in_proj_ba_join()


direct_register_custom_op(
    op_name="gdn_in_proj_ba_join",
    op_func=gdn_in_proj_ba_join,
    mutates_args=["ba"],
)


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
