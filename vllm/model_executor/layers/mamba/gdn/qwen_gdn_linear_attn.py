# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import math
import os
from typing import Literal

import torch
from einops import rearrange
from torch import nn

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.fusion import norm_quant
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn import (
    gdn_layer_graphs,
    gdn_out_alloc,
    gdn_step_plan,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops import gdn_state_commit
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.inc import INCConfig
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
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

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
# LCD2_BF16=tiny (+ LCD2_BA=1, default): decode-size in_proj_ba through FlashInfer's
# TinyGEMM2 (one kernel instead of cuBLAS GEMM + splitKreduce); see vllm/model_executor/layers/lcd2_bf16.py
from vllm.model_executor.layers import lcd2_bf16 as _lcd2  # noqa: E402


# ---------------------------------------------------------------------------
# Zero-copy mixed (prefill + MTP spec-decode) GDN path.
#   VLLM_GDN_MIXED_FASTPATH=1 (default) enables it; 0 restores the stock path.
#   VLLM_GDN_FI_USE_CP=auto|0|1 controls FlashInfer's context-parallel prefill
#   routing ("auto" = the length-aware rule in _gdn_fi_want_cp; the indexed
#   state-pool I/O used by the fast path is only available on the non-CP
#   kernel, so CP batches fall back to gather/scatter of the prefill states,
#   still without index_copy).
# ---------------------------------------------------------------------------
_GDN_MIXED_FASTPATH = os.environ.get("VLLM_GDN_MIXED_FASTPATH", "1") == "1"
_GDN_FI_USE_CP = os.environ.get("VLLM_GDN_FI_USE_CP", "auto").strip().lower()
# FlashInfer's own heuristic picks the 4-kernel context-parallel (CP) path for every
# single-sequence chunk, but CP only beats the non-CP kernel for one long sequence, so
# "auto" routes to CP only for a single sequence of >= VLLM_GDN_FI_CP_MIN_TOKENS tokens.
_GDN_FI_CP_MIN_TOKENS = int(os.environ.get("VLLM_GDN_FI_CP_MIN_TOKENS", "6144"))
# 1 = FlashInfer indexed state-pool I/O (state_indices=, in place, 1 launch);
# 0 = gather initial states + packed FI + scatter (3 launches, faster FI kernel).
_GDN_FI_STATE_POOL = os.environ.get("VLLM_GDN_FI_STATE_POOL", "1") == "1"


# CP routing v2 + CP with in-place state pool (both env-gated, default off = the routing
# and gather/scatter CP path above). CP also wins for a multi-sequence batch dominated by
# one long sequence and loses for balanced or short multi-sequence batches.
#   VLLM_GDN_FI_CP_RULE=v2: CP if 1 seq >= CP_MIN_TOKENS, or n>1 with max_len >= CP_MULTI_MIN_TOKENS [8192]
#                           and max_len >= CP_MULTI_FRAC [0.8] x total prefill tokens.
#   VLLM_GDN_FI_CP_POOL=1:  run CP with state_indices= (in-place pool I/O) instead of gather/where/scatter.
_GDN_FI_CP_RULE = os.environ.get("VLLM_GDN_FI_CP_RULE", "v1").strip().lower()
_GDN_FI_CP_MULTI_MIN = int(os.environ.get("VLLM_GDN_FI_CP_MULTI_MIN_TOKENS", "8192"))
_GDN_FI_CP_MULTI_FRAC = float(os.environ.get("VLLM_GDN_FI_CP_MULTI_FRAC", "0.8"))
_GDN_FI_CP_POOL = os.environ.get("VLLM_GDN_FI_CP_POOL", "0") == "1"


_GDN_FI_HAS_MAXLEN = []

# VLLM_GDN_FI_VSPLIT=1: non-CP FlashInfer GDN prefill steps whose (seq x value-head) grid under-fills
# the GPU run the V-split kernel (vllm/third_party/flashinfer_gdn_vsplit: one CTA per (seq, head,
# 64-row V slice), M=64 tcgen05 state GEMMs). Output and final state are bitwise identical to
# FlashInfer's non-CP kernel.
_GDN_FI_VSPLIT = os.environ.get("VLLM_GDN_FI_VSPLIT", "0") == "1"
_GDN_VSPLIT_STATE: dict = {}
_GDN_FI_VSPLIT_CHECK = os.environ.get("VLLM_GDN_FI_VSPLIT_CHECK", "0") == "1"


def _gdn_vsplit_call(q, k, v, g_exp, beta, out, ssm_state, slots, cu_seqlens, attn_metadata) -> bool:
    st = _GDN_VSPLIT_STATE
    if st.get("disabled"):
        return False
    try:
        if "mod" not in st:
            import vllm.third_party.flashinfer_gdn_vsplit as gdn_vsplit

            st["mod"] = gdn_vsplit
            st["calls"] = 0
        mod = st["mod"]
        n = cu_seqlens.numel() - 1
        vsf = mod.choose_vsplit(n, q.size(0), int(getattr(attn_metadata, "prefill_max_seqlen", 0)),
                                hv=v.size(1))
        if vsf == 1:
            return False
        if ssm_state.dtype not in (torch.float32, torch.bfloat16):
            if not st.get("warned_dtype"):
                st["warned_dtype"] = True
                logger.warning("gdn_vsplit: V-split ineligible: ssm state dtype %s (fp32/bf16 only); using FlashInfer",
                               ssm_state.dtype)
            return False
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
                and g_exp.is_contiguous() and beta.is_contiguous() and q.size(2) == 128
                and ssm_state.stride(3) == 1):
            if not st.get("warned_layout"):
                st["warned_layout"] = True
                logger.warning("gdn_vsplit: V-split ineligible: non-contiguous inputs; using FlashInfer")
            return False
        if _GDN_FI_VSPLIT_CHECK:
            # debug: run stock FlashInfer on copies and compare bitwise (slow; diagnostics only)
            from flashinfer.gdn_prefill import chunk_gated_delta_rule as _fi

            ref_pool = ssm_state.clone()  # debug only: clones the whole pool (can OOM on a full server)
            ref_out = out.clone()
            _fi(q=q, k=k, v=v, g=g_exp, beta=beta, initial_state=ref_pool, output_final_state=True,
                cu_seqlens=cu_seqlens, output=ref_out, output_state=ref_pool, use_cp=False, state_indices=slots)
        mod.chunk_gated_delta_rule_vsplit(
            q, k, v, g_exp, beta, out, cu_seqlens.to(torch.int32), ssm_state, ssm_state,
            1.0 / math.sqrt(q.size(2)), state_indices=slots, v_split=vsf,
        )
        if _GDN_FI_VSPLIT_CHECK:
            sl = slots.long()
            no = int((out.view(torch.int16) != ref_out.view(torch.int16)).sum())
            ns = int((ssm_state[sl].contiguous().view(torch.int32) != ref_pool[sl].contiguous().view(torch.int32)).sum())
            st["checked"] = st.get("checked", 0) + 1
            if no or ns or st["checked"] <= 3:
                logger.warning("gdn_vsplit: CHECK call=%d n=%d T=%d cu=%s out_mis=%d state_mis=%d q%s/%s k%s v%s o%s "
                               "g%s b%s pool%s%s slots=%s", st["checked"], n, q.size(0),
                               cu_seqlens.tolist()[:9], no, ns, tuple(q.shape), q.stride(), k.stride(), v.stride(),
                               out.stride(), g_exp.stride(), beta.stride(), tuple(ssm_state.shape), ssm_state.stride(),
                               slots.tolist()[:8])
        st["calls"] += 1
        if st["calls"] == 1:
            logger.info("gdn_vsplit: V-split GDN prefill active: first call took the V-split path (rule=%s, state=%s, "
                        "v_split=%d, n=%d)", mod.VSPLIT_RULE, str(ssm_state.dtype).replace("torch.", ""), vsf, n)
        return True
    except Exception as e:  # never break serving: fall back to FlashInfer
        st["disabled"] = True
        logger.warning("gdn_vsplit: V-split kernel unavailable (%s: %s); using FlashInfer", type(e).__name__, e)
        return False


def _gdn_fi_maxlen_kw(want_cp: bool, attn_metadata) -> dict:
    """Newer FlashInfer (upstream main GDN files) sizes CP grids from max_seqlen and
    otherwise assumes a balanced batch (under-launch -> wrong output for imbalanced multi-sequence CP).
    Pass the exact host-side maximum whenever the API has it."""
    if not want_cp:
        return {}
    if not _GDN_FI_HAS_MAXLEN:
        import inspect

        from flashinfer.gdn_prefill import chunk_gated_delta_rule as _f

        _GDN_FI_HAS_MAXLEN.append("max_seqlen" in inspect.signature(_f).parameters)
    m = getattr(attn_metadata, "prefill_max_seqlen", 0)
    if _GDN_FI_HAS_MAXLEN[0] and m > 0:
        return {"max_seqlen": int(m)}
    return {}


def _gdn_fi_want_cp(num_seqs: int, num_tokens: int, max_seqlen: int = 0) -> bool:
    if _GDN_FI_USE_CP in ("1", "true"):
        return True
    if _GDN_FI_USE_CP in ("0", "false"):
        return False
    if num_seqs == 1:
        return num_tokens >= _GDN_FI_CP_MIN_TOKENS
    if _GDN_FI_CP_RULE == "v2" and max_seqlen > 0:
        return (max_seqlen >= _GDN_FI_CP_MULTI_MIN
                and max_seqlen >= _GDN_FI_CP_MULTI_FRAC * num_tokens)
    return False


@triton.jit
def _gdn_zero_state_slots_kernel(
    state_ptr,
    slot_ptr,
    has_init_ptr,
    stride_slot,
    HEAD_ELEMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Zero state[slot[i], head] for sequences without an initial state."""
    i_seq = tl.program_id(0)
    i_head = tl.program_id(1)
    has_init = tl.load(has_init_ptr + i_seq)
    slot = tl.load(slot_ptr + i_seq).to(tl.int64)
    if (has_init != 0) or (slot < 0):
        return
    base = state_ptr + slot * stride_slot + i_head * HEAD_ELEMS
    offs = tl.arange(0, BLOCK)
    zeros = tl.zeros([BLOCK], dtype=state_ptr.dtype.element_ty)
    for start in range(0, HEAD_ELEMS, BLOCK):
        tl.store(base + start + offs, zeros)


def gdn_zero_state_slots(
    state: torch.Tensor, slots: torch.Tensor, has_initial_state: torch.Tensor
) -> None:
    n = slots.numel()
    if n == 0:
        return
    hv = state.size(1)
    head_elems = state.size(2) * state.size(3)
    assert state.stride(1) == head_elems and state.stride(3) == 1
    _gdn_zero_state_slots_kernel[(n, hv)](
        state,
        slots,
        has_initial_state,
        state.stride(0),
        HEAD_ELEMS=head_elems,
        BLOCK=4096,
        num_warps=4,
    )


@triton.jit(do_not_specialize=["T"])
def _gdn_gated_rmsnorm_kernel(
    x_ptr,  # [T, HV, D] contiguous (in/out)
    z_ptr,  # [T, HV*D] rows with stride stride_z_tok, D contiguous
    w_ptr,  # [D]
    y_ptr,  # [T, HV, D] contiguous (may alias x)
    T,
    stride_z_tok,
    eps,
    HV: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    SIGMOID_GATE: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    mask = (offs_t < T)[:, None]
    row = offs_t.to(tl.int64)[:, None]
    xo = row * (HV * D) + i_h * D + offs_d[None, :]
    x = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(
        z_ptr + row * stride_z_tok + i_h * D + offs_d[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    w = tl.load(w_ptr + offs_d).to(tl.float32)
    var = tl.sum(x * x, axis=1) / D
    rstd = tl.rsqrt(var + eps)
    y = x * rstd[:, None] * w[None, :]
    if SIGMOID_GATE:
        y = y * tl.sigmoid(z)
    else:
        y = y * (z * tl.sigmoid(z))
    tl.store(y_ptr + xo, y.to(y_ptr.dtype.element_ty), mask=mask)


from triton.language.extra import libdevice  # noqa: E402

_GDN_FUSED_CONV = os.environ.get("VLLM_GDN_FUSED_CONV", "1") == "1"
_GDN_FUSION_WARMED = False
# 1 = run the spec-decode part of mixed batches with the stock FLA Triton kernel
# (bit-exact with stock); 0 (default) = the CUDA MTP kernel used by decode-only steps
_GDN_MIXED_SPEC_TRITON = os.environ.get("VLLM_GDN_MIXED_SPEC_TRITON", "0") == "1"
_GDN_VERIFY_LEFT = [int(os.environ.get("VLLM_GDN_FASTPATH_VERIFY", "0"))]
_GDN_FUSED_CONV_BT = int(os.environ.get("VLLM_GDN_FUSED_CONV_BT", "16"))
_GDN_FUSED_CONV_WARPS = int(os.environ.get("VLLM_GDN_FUSED_CONV_WARPS", "4"))
_GDN_FUSED_CONV_V2 = os.environ.get("VLLM_GDN_FUSED_CONV_V2", "1") == "1"
_GDN_FUSED_CONV_V2_BT = int(os.environ.get("VLLM_GDN_FUSED_CONV_V2_BT", "32"))
_GDN_FUSED_CONV_V2_ST = int(os.environ.get("VLLM_GDN_FUSED_CONV_V2_ST", "32"))
_GDN_FUSED_CONV_V2_WARPS = int(os.environ.get("VLLM_GDN_FUSED_CONV_V2_WARPS", "4"))
_GDN_FUSED_CONV_V2_STAGES = int(os.environ.get("VLLM_GDN_FUSED_CONV_V2_STAGES", "1"))

# Deferred GDN state commit (GDN_STATE_COMMIT=1), see
# vllm/model_executor/layers/mamba/ops/gdn_state_commit. _DEFERRED is off in the
# layout-only control mode (same page layout, stock kernels).
_GDN_STATE_COMMIT = gdn_state_commit.enabled()
_GDN_STATE_COMMIT_DEFERRED = _GDN_STATE_COMMIT and not gdn_state_commit.layout_only()
if _GDN_STATE_COMMIT and _GDN_MIXED_SPEC_TRITON:
    raise RuntimeError(
        "gdn_state_commit requires VLLM_GDN_MIXED_SPEC_TRITON=0 (the FLA spec "
        "kernel writes per-token state slots)"
    )
# Per-step plan / host trims of the mixed-step GDN core (GGM, GGM_OG2), see
# gdn_step_plan.
gdn_step_plan.check_config(_GDN_STATE_COMMIT, norm_quant.NQF)
gdn_layer_graphs.check_config()


@triton.jit
def _gdn_conv_tile(
    x_ptr, stride_x_tok, w_ptr, stride_w_dim, stride_w_width,
    cs_ptr, s_cs_seq, s_cs_dim, s_cs_tok,
    slot, use_state, seq_start, offs_t, mask_t, ch_base,
    D: tl.constexpr, WIDTH: tl.constexpr,
):
    """Causal depthwise conv (+SiLU) for one [BT, D] channel tile of one sequence,
    bit-matching _causal_conv1d_fwd_kernel (bf16 x*w products, fp32 accumulate,
    bf16 output)."""
    offs_c = ch_base + tl.arange(0, D)
    acc = tl.zeros([offs_t.shape[0], D], dtype=tl.float32)
    for j in tl.static_range(WIDTH):
        w = tl.load(w_ptr + offs_c * stride_w_dim + j * stride_w_width)
        tau = offs_t - (WIDTH - 1) + j  # absolute token index
        in_seq = tau >= seq_start
        xv = tl.load(
            x_ptr + tau.to(tl.int64)[:, None] * stride_x_tok + offs_c[None, :],
            mask=(mask_t & in_seq)[:, None], other=0.0,
        )
        d = seq_start - tau  # 1..WIDTH-1 -> conv-state row WIDTH-1-d
        row = (WIDTH - 1) - d
        sv = tl.load(
            cs_ptr + slot * s_cs_seq + offs_c[None, :] * s_cs_dim
            + row[:, None] * s_cs_tok,
            mask=(mask_t & (d > 0) & use_state)[:, None], other=0.0,
        )
        acc += (xv + sv) * w[None, :]
    acc = acc / (1 + tl.exp(-acc))
    return acc.to(x_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _gdn_conv_tile_interior(
    x_ptr, stride_x_tok, w_ptr, stride_w_dim, stride_w_width,
    offs_t, mask_t, ch_base, D: tl.constexpr, WIDTH: tl.constexpr,
):
    """Same math as _gdn_conv_tile for tiles whose taps never reach before the
    sequence start (no conv-state loads, no per-row start masks)."""
    offs_c = ch_base + tl.arange(0, D)
    acc = tl.zeros([offs_t.shape[0], D], dtype=tl.float32)
    for j in tl.static_range(WIDTH):
        w = tl.load(w_ptr + offs_c * stride_w_dim + j * stride_w_width)
        tau = offs_t - (WIDTH - 1) + j
        xv = tl.load(
            x_ptr + tau.to(tl.int64)[:, None] * stride_x_tok + offs_c[None, :],
            mask=mask_t[:, None], other=0.0,
        )
        acc += xv * w[None, :]
    acc = acc / (1 + tl.exp(-acc))
    return acc.to(x_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _gdn_write_conv_state(
    x_ptr, stride_x_tok, cs_ptr, s_cs_seq, s_cs_dim, s_cs_tok,
    slot, use_state, seq_start, seq_end, ch_base,
    D: tl.constexpr, WIDTH: tl.constexpr, NP2W: tl.constexpr,
):
    """conv_state[slot, rows 0..W-2] = last W-1 inputs of the sequence (older rows
    shifted from the previous state when the chunk is shorter than W-1)."""
    offs_c = ch_base + tl.arange(0, D)
    r = tl.arange(0, NP2W)
    rmask = r < (WIDTH - 1)
    tau = seq_end - (WIDTH - 1) + r
    in_seq = tau >= seq_start
    xv = tl.load(
        x_ptr + tau.to(tl.int64)[:, None] * stride_x_tok + offs_c[None, :],
        mask=(rmask & in_seq)[:, None], other=0.0,
    )
    d = seq_start - tau
    srow = (WIDTH - 1) - d
    sv = tl.load(
        cs_ptr + slot * s_cs_seq + offs_c[None, :] * s_cs_dim + srow[:, None] * s_cs_tok,
        mask=(rmask & (d > 0) & use_state)[:, None], other=0.0,
    )
    tl.debug_barrier()
    tl.store(
        cs_ptr + slot * s_cs_seq + offs_c[None, :] * s_cs_dim + r[:, None] * s_cs_tok,
        xv + sv, mask=rmask[:, None],
    )


@triton.jit
def _gdn_fused_conv_post_conv_kernel(
    x_ptr, stride_x_tok, w_ptr, stride_w_dim, stride_w_width,
    cs_ptr, s_cs_seq, s_cs_dim, s_cs_tok,
    cidx_ptr, hinit_ptr, cu_ptr, num_seqs,
    a_ptr, b_ptr, stride_a, stride_b, alog_ptr, dtb_ptr,
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    WIDTH: tl.constexpr, NP2W: tl.constexpr, BT: tl.constexpr,
):
    """conv1d(+state) -> SiLU -> split q/k/v -> l2norm(q,k) -> g=exp(-exp(A)*softplus),
    beta=sigmoid(b), for prefill chunks. One program = BT tokens of ONE sequence x one
    head (q+k channels for i_head < H, v channels + gating otherwise). The chunk-0
    program of each sequence also writes that sequence's final conv state (it is the
    only reader of the old state, so there is no cross-program race)."""
    pid = tl.program_id(0)
    i_head = tl.program_id(1)
    # map program -> (sequence, chunk)
    seq = -1
    chunk = 0
    seq_start = 0
    seq_end = 0
    blocks = 0
    for i in range(num_seqs):
        s0 = tl.load(cu_ptr + i)
        s1 = tl.load(cu_ptr + i + 1)
        nb = tl.cdiv(s1 - s0, BT)
        hit = (pid >= blocks) & (pid < blocks + nb)
        seq = tl.where(hit, i, seq)
        chunk = tl.where(hit, pid - blocks, chunk)
        seq_start = tl.where(hit, s0, seq_start)
        seq_end = tl.where(hit, s1, seq_end)
        blocks += nb
    if seq < 0:
        return
    slot = tl.load(cidx_ptr + seq).to(tl.int64)
    use_state = tl.load(hinit_ptr + seq) != 0
    offs_t = seq_start + chunk * BT + tl.arange(0, BT)
    mask_t = offs_t < seq_end
    row64 = offs_t.to(tl.int64)
    if i_head < H:
        q = _gdn_conv_tile(x_ptr, stride_x_tok, w_ptr, stride_w_dim, stride_w_width,
                           cs_ptr, s_cs_seq, s_cs_dim, s_cs_tok, slot, use_state,
                           seq_start, offs_t, mask_t, i_head * K, K, WIDTH)
        k = _gdn_conv_tile(x_ptr, stride_x_tok, w_ptr, stride_w_dim, stride_w_width,
                           cs_ptr, s_cs_seq, s_cs_dim, s_cs_tok, slot, use_state,
                           seq_start, offs_t, mask_t, H * K + i_head * K, K, WIDTH)
        q = q * (1.0 / tl.sqrt(tl.sum(q * q, axis=1) + 1e-6))[:, None]
        k = k * (1.0 / tl.sqrt(tl.sum(k * k, axis=1) + 1e-6))[:, None]
        offs_k = tl.arange(0, K)
        o = row64[:, None] * (H * K) + i_head * K + offs_k[None, :]
        tl.store(q_ptr + o, q.to(q_ptr.dtype.element_ty), mask=mask_t[:, None])
        tl.store(k_ptr + o, k.to(k_ptr.dtype.element_ty), mask=mask_t[:, None])
        if chunk == 0:
            _gdn_write_conv_state(x_ptr, stride_x_tok, cs_ptr, s_cs_seq, s_cs_dim,
                                  s_cs_tok, slot, use_state, seq_start, seq_end,
                                  i_head * K, K, WIDTH, NP2W)
            _gdn_write_conv_state(x_ptr, stride_x_tok, cs_ptr, s_cs_seq, s_cs_dim,
                                  s_cs_tok, slot, use_state, seq_start, seq_end,
                                  H * K + i_head * K, K, WIDTH, NP2W)
    else:
        i_hv = i_head - H
        vch = 2 * H * K + i_hv * V
        v = _gdn_conv_tile(x_ptr, stride_x_tok, w_ptr, stride_w_dim, stride_w_width,
                           cs_ptr, s_cs_seq, s_cs_dim, s_cs_tok, slot, use_state,
                           seq_start, offs_t, mask_t, vch, V, WIDTH)
        offs_v = tl.arange(0, V)
        tl.store(v_ptr + row64[:, None] * (HV * V) + i_hv * V + offs_v[None, :],
                 v.to(v_ptr.dtype.element_ty), mask=mask_t[:, None])
        A_log = tl.load(alog_ptr + i_hv).to(tl.float32)
        dtb = tl.load(dtb_ptr + i_hv).to(tl.float32)
        av = tl.load(a_ptr + row64 * stride_a + i_hv, mask=mask_t, other=0).to(tl.float32)
        bv = tl.load(b_ptr + row64 * stride_b + i_hv, mask=mask_t, other=0).to(tl.float32)
        xx = av + dtb
        sp = tl.where(xx > 0, xx + tl.log(1.0 + tl.exp(-xx)), tl.log(1.0 + tl.exp(xx)))
        sp = tl.where(xx <= 20.0, sp, xx)
        # accurate expf (libdevice) so exp(g) is bit-identical to the stock
        # torch.exp(g) in fi_chunk_gated_delta_rule (tl.exp is the fast ex2 path)
        g = libdevice.exp(-tl.exp(A_log) * sp)
        tl.store(g_ptr + row64 * HV + i_hv, g, mask=mask_t)
        tl.store(beta_ptr + row64 * HV + i_hv, tl.sigmoid(bv), mask=mask_t)
        if chunk == 0:
            _gdn_write_conv_state(x_ptr, stride_x_tok, cs_ptr, s_cs_seq, s_cs_dim,
                                  s_cs_tok, slot, use_state, seq_start, seq_end,
                                  vch, V, WIDTH, NP2W)


@triton.jit(do_not_specialize=["num_seqs"])
def _gdn_fused_conv_post_conv_kernel_v2(
    x_ptr, stride_x_tok, w_ptr, stride_w_dim, stride_w_width,
    cs_ptr, s_cs_seq, s_cs_dim, s_cs_tok,
    cidx_ptr, hinit_ptr, cu_ptr, num_seqs,
    a_ptr, b_ptr, stride_a, stride_b, alog_ptr, dtb_ptr,
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    WIDTH: tl.constexpr, NP2W: tl.constexpr, BT: tl.constexpr, ST: tl.constexpr,
    MAXS: tl.constexpr, NSTAGES: tl.constexpr,
):
    """v2 of _gdn_fused_conv_post_conv_kernel (bitwise-identical math):
    - program -> (sequence, chunk) from ONE vectorized load of cu_seqlens
      (no per-sequence scalar scan);
    - BT-token chunks processed as a software-pipelined loop over ST-token
      sub-tiles (more bytes in flight, prologue amortized)."""
    pid = tl.program_id(0)
    i_head = tl.program_id(1)
    si = tl.arange(0, MAXS)
    c0 = tl.load(cu_ptr + si, mask=si < num_seqs, other=0)
    c1 = tl.load(cu_ptr + si + 1, mask=si < num_seqs, other=0)
    nb = tl.where(si < num_seqs, tl.cdiv(c1 - c0, BT), 0)
    ends = tl.cumsum(nb, axis=0)
    seq = tl.sum((ends <= pid).to(tl.int32), axis=0)
    if seq >= num_seqs:
        return
    sel = si == seq
    seq_start = tl.sum(tl.where(sel, c0, 0), axis=0)
    seq_end = tl.sum(tl.where(sel, c1, 0), axis=0)
    chunk = pid - (tl.sum(tl.where(sel, ends, 0), axis=0) - tl.sum(tl.where(sel, nb, 0), axis=0))
    slot = tl.load(cidx_ptr + seq).to(tl.int64)
    use_state = tl.load(hinit_ptr + seq) != 0
    chunk_start = seq_start + chunk * BT
    if i_head < 2 * H:
        # one 128-channel group per program: q head (i_head < H) or k head
        is_k = i_head >= H
        hh = i_head - H * is_k.to(tl.int32)
        ch = i_head * K  # q channels [0, H*K), k channels [H*K, 2*H*K)
        out_ptr = tl.where(is_k, k_ptr, q_ptr)
        offs_k = tl.arange(0, K)
        for sub in tl.range(0, BT // ST, num_stages=NSTAGES):
            offs_t = chunk_start + sub * ST + tl.arange(0, ST)
            mask_t = offs_t < seq_end
            interior = (chunk_start + sub * ST) >= (seq_start + WIDTH - 1)
            row64 = offs_t.to(tl.int64)
            if interior:
                y = _gdn_conv_tile_interior(x_ptr, stride_x_tok, w_ptr, stride_w_dim,
                                            stride_w_width, offs_t, mask_t, ch, K, WIDTH)
            else:
                y = _gdn_conv_tile(x_ptr, stride_x_tok, w_ptr, stride_w_dim, stride_w_width,
                                   cs_ptr, s_cs_seq, s_cs_dim, s_cs_tok, slot, use_state,
                                   seq_start, offs_t, mask_t, ch, K, WIDTH)
            y = y * (1.0 / tl.sqrt(tl.sum(y * y, axis=1) + 1e-6))[:, None]
            o = row64[:, None] * (H * K) + hh * K + offs_k[None, :]
            tl.store(out_ptr + o, y.to(q_ptr.dtype.element_ty), mask=mask_t[:, None])
        if chunk == 0:
            _gdn_write_conv_state(x_ptr, stride_x_tok, cs_ptr, s_cs_seq, s_cs_dim,
                                  s_cs_tok, slot, use_state, seq_start, seq_end,
                                  ch, K, WIDTH, NP2W)
    else:
        i_hv = i_head - 2 * H
        vch = 2 * H * K + i_hv * V
        offs_v = tl.arange(0, V)
        A_log = tl.load(alog_ptr + i_hv).to(tl.float32)
        dtb = tl.load(dtb_ptr + i_hv).to(tl.float32)
        for sub in tl.range(0, BT // ST, num_stages=NSTAGES):
            offs_t = chunk_start + sub * ST + tl.arange(0, ST)
            mask_t = offs_t < seq_end
            interior = (chunk_start + sub * ST) >= (seq_start + WIDTH - 1)
            row64 = offs_t.to(tl.int64)
            if interior:
                v = _gdn_conv_tile_interior(x_ptr, stride_x_tok, w_ptr, stride_w_dim,
                                                stride_w_width, offs_t, mask_t, vch, V, WIDTH)
            else:
                v = _gdn_conv_tile(x_ptr, stride_x_tok, w_ptr, stride_w_dim, stride_w_width,
                                       cs_ptr, s_cs_seq, s_cs_dim, s_cs_tok, slot, use_state,
                                       seq_start, offs_t, mask_t, vch, V, WIDTH)
            tl.store(v_ptr + row64[:, None] * (HV * V) + i_hv * V + offs_v[None, :],
                     v.to(v_ptr.dtype.element_ty), mask=mask_t[:, None])
            av = tl.load(a_ptr + row64 * stride_a + i_hv, mask=mask_t, other=0).to(tl.float32)
            bv = tl.load(b_ptr + row64 * stride_b + i_hv, mask=mask_t, other=0).to(tl.float32)
            xx = av + dtb
            sp = tl.where(xx > 0, xx + tl.log(1.0 + tl.exp(-xx)), tl.log(1.0 + tl.exp(xx)))
            sp = tl.where(xx <= 20.0, sp, xx)
            g = libdevice.exp(-tl.exp(A_log) * sp)
            tl.store(g_ptr + row64 * HV + i_hv, g, mask=mask_t)
            tl.store(beta_ptr + row64 * HV + i_hv, tl.sigmoid(bv), mask=mask_t)
        if chunk == 0:
            _gdn_write_conv_state(x_ptr, stride_x_tok, cs_ptr, s_cs_seq, s_cs_dim,
                                  s_cs_tok, slot, use_state, seq_start, seq_end,
                                  vch, V, WIDTH, NP2W)


# VLLM_GDN_CONV_CUDA=1 routes gdn_fused_conv_post_conv to the CUDA kernel in
# vllm/model_executor/layers/mamba/ops/gdn_conv_cuda (JIT-built into $GDN_CONV_CUDA_BUILD_DIR). Bit-exact
# with the Triton v2 kernel (q/k/v/g/beta and conv state); falls back to Triton if the extension cannot be
# built or loaded, or the layout contract is not met. VLLM_GDN_CONV_CUDA_TPH=auto|4|8|16 (auto: 4 if the
# mean prefill length is < 1024 tokens else 8).
_GDN_CONV_CUDA = os.environ.get("VLLM_GDN_CONV_CUDA", "0") == "1"
_GDN_CONV_CUDA_TPH = os.environ.get("VLLM_GDN_CONV_CUDA_TPH", "auto")
_GDN_CONV_CUDA_MOD = []


def _gdn_conv_cuda_call(x, conv_weights, conv_state, cache_indices, has_initial_state,
                        cu_seqlens, num_seqs, a, b, A_log, dt_bias, H, K, V):
    if not _GDN_CONV_CUDA_MOD:
        try:
            from vllm.model_executor.layers.mamba.ops import gdn_conv_cuda as _gcc

            _gcc.load()
            _GDN_CONV_CUDA_MOD.append(_gcc)
            logger.info("gdn_conv_cuda: CUDA fused conv1d+post-conv enabled")
        except Exception as e:  # noqa: BLE001
            _GDN_CONV_CUDA_MOD.append(None)
            logger.warning("gdn_conv_cuda: CUDA fused conv unavailable (%s); using Triton", e)
    m = _GDN_CONV_CUDA_MOD[0]
    if m is None:
        return None
    if _GDN_CONV_CUDA_TPH == "auto":
        tph = 4 if x.shape[0] < 1024 * max(int(num_seqs), 1) else 8
    else:
        tph = int(_GDN_CONV_CUDA_TPH)
    return m.fused_conv_post_conv(x, conv_weights, conv_state, cache_indices, has_initial_state,
                                  cu_seqlens, num_seqs, a, b, A_log, dt_bias, H, K, V, tph=tph)


def gdn_fused_conv_post_conv(
    x: torch.Tensor,  # [P, conv_dim] prefill rows of mixed_qkv (channels contiguous)
    conv_weights: torch.Tensor,  # [conv_dim, width]
    conv_state: torch.Tensor,  # [slots, conv_dim, >= width-1] (any strides)
    cache_indices: torch.Tensor,  # [num_seqs]
    has_initial_state: torch.Tensor,  # [num_seqs] bool
    cu_seqlens: torch.Tensor,  # [num_seqs+1] int32 (relative to x row 0)
    num_seqs: int,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    use_cuda: bool | None = None,  # None: VLLM_GDN_CONV_CUDA; False: Triton only
):
    P = x.shape[0]
    if _GDN_CONV_CUDA if use_cuda is None else use_cuda:
        # bit-exact CUDA (sm_107a) version of the v2 Triton kernel below
        res = _gdn_conv_cuda_call(x, conv_weights, conv_state, cache_indices, has_initial_state,
                                  cu_seqlens, num_seqs, a, b, A_log, dt_bias, num_k_heads,
                                  head_k_dim, head_v_dim)
        if res is not None:
            return res
    cache_indices = cache_indices.contiguous()
    has_initial_state = has_initial_state.contiguous()
    H, K, V = num_k_heads, head_k_dim, head_v_dim
    HV = A_log.shape[0]
    width = conv_weights.shape[1]
    assert x.stride(1) == 1 and x.shape[1] == 2 * H * K + HV * V
    q = torch.empty(P, H, K, dtype=x.dtype, device=x.device)
    k = torch.empty(P, H, K, dtype=x.dtype, device=x.device)
    v = torch.empty(P, HV, V, dtype=x.dtype, device=x.device)
    g = torch.empty(P, HV, dtype=torch.float32, device=x.device)
    beta = torch.empty(P, HV, dtype=torch.float32, device=x.device)
    if _GDN_FUSED_CONV_V2 and num_seqs <= 64:
        BT = _GDN_FUSED_CONV_V2_BT
        grid = (triton.cdiv(P, BT) + num_seqs, 2 * H + HV)
        _gdn_fused_conv_post_conv_kernel_v2[grid](
            x, x.stride(0), conv_weights, conv_weights.stride(0),
            conv_weights.stride(1),
            conv_state, conv_state.stride(0), conv_state.stride(1),
            conv_state.stride(2),
            cache_indices, has_initial_state, cu_seqlens, num_seqs,
            a, b, a.stride(0), b.stride(0), A_log, dt_bias,
            q, k, v, g, beta,
            H=H, HV=HV, K=K, V=V, WIDTH=width,
            NP2W=triton.next_power_of_2(width - 1), BT=BT,
            ST=_GDN_FUSED_CONV_V2_ST,
            MAXS=64,  # fixed: one compiled variant for num_seqs <= 64 (v1 otherwise)
            NSTAGES=_GDN_FUSED_CONV_V2_STAGES,
            num_warps=_GDN_FUSED_CONV_V2_WARPS,
        )
        return q, k, v, g, beta
    BT = _GDN_FUSED_CONV_BT
    grid = (triton.cdiv(P, BT) + num_seqs, H + HV)
    _gdn_fused_conv_post_conv_kernel[grid](
        x, x.stride(0), conv_weights, conv_weights.stride(0), conv_weights.stride(1),
        conv_state, conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        cache_indices, has_initial_state, cu_seqlens, num_seqs,
        a, b, a.stride(0), b.stride(0), A_log, dt_bias,
        q, k, v, g, beta,
        H=H, HV=HV, K=K, V=V, WIDTH=width,
        NP2W=triton.next_power_of_2(width - 1), BT=BT,
        num_warps=_GDN_FUSED_CONV_WARPS,
    )
    return q, k, v, g, beta


def gdn_gated_rmsnorm_(
    x: torch.Tensor,  # [T, HV, D] contiguous, normalized in place
    z: torch.Tensor,  # [T, HV, D] view with z.stride(1) == D, z.stride(2) == 1
    weight: torch.Tensor,
    eps: float,
    activation: str,
) -> None:
    if norm_quant.NQF and norm_quant.gdn_gated_rmsnorm_into_target(
        x, z, weight, eps, activation
    ):
        # NQF=1: rows of the layer inside the fused packed core were normed
        # with the out_proj MXFP8 quant epilogue (see norm_quant).
        return
    T, HV, D = x.shape
    if T == 0:
        return
    assert x.is_contiguous() and z.stride(2) == 1 and z.stride(1) == D
    BT = 16
    _gdn_gated_rmsnorm_kernel[(triton.cdiv(T, BT), HV)](
        x,
        z,
        weight,
        x,
        T,
        z.stride(0),
        eps,
        HV=HV,
        D=D,
        BT=BT,
        SIGMOID_GATE=(activation == "sigmoid"),
        num_warps=4,
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
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(torch.int64)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        # length-aware CP routing (see _gdn_fi_want_cp)
        use_cp=_gdn_fi_want_cp(
            cu_seqlens.numel() - 1 if cu_seqlens is not None else 1, q.size(0)
        ),
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
        )
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
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
    # Set per layer by norm_quant.configure_decoder_layer (NQF=1) when out_proj
    # takes the MXFP8 input produced by _forward_core_fused_norm_packed.
    _nqf_gdn: bool = False

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

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        dtypes = super().get_state_dtype()
        if _GDN_STATE_COMMIT:
            # Layer-level consumers unpack get_state_dtype() into (conv, ssm); the
            # token-log dtype is added by MambaBase.get_kv_cache_spec and the page
            # size comes from the model-class calculator, which includes the log.
            return tuple(dtypes)[:2]
        return dtypes

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
        self._lcd2_ba = False
        _w_ba = getattr(self.in_proj_ba, "weight", None)
        if _w_ba is not None and _lcd2.ba_eligible(_w_ba) and _lcd2.available():
            # zero bias: TinyGEMM2's sm100 variants run with a bias tensor
            self.register_buffer(
                "_lcd2_ba_b",
                torch.zeros(_w_ba.shape[0], dtype=_w_ba.dtype, device=_w_ba.device),
                persistent=False,
            )
            self._lcd2_ba = True

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
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        if self._lcd2_ba:
            # LCD2_BA: TinyGEMM2 for decode-size M, F.linear above LCD2_MAXM
            ba = torch.ops.vllm.lcd2_bf16_linear(
                hidden_states, self.in_proj_ba.weight, self._lcd2_ba_b
            )
        else:
            ba, _ = self.in_proj_ba(hidden_states)

        use_fused_gdn_decode = (
            self.enable_fused_gdn_decode
            and hidden_states.dtype == torch.bfloat16
            and self.norm.weight.dtype in (torch.bfloat16, torch.float32)
        )
        if use_fused_gdn_decode:
            if gdn_out_alloc.ENABLED:
                # empty + zero only the padding rows (all real rows are written
                # by the core op); see gdn_out_alloc.py
                core_attn_out = torch.ops.vllm.ews_gdn_out_alloc(
                    hidden_states, self.num_v_heads // self.tp_size, self.head_v_dim
                )
            else:
                core_attn_out = torch.zeros(
                    (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                mixed_qkvz,
                ba,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
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

        # All kernels use BT = chunk_size, so a single pass with T = chunk_size
        # is sufficient to populate every autotuner cache. Mirror the real
        # prefill path here: build q/k/v/g/beta via fused_post_conv_prep and
        # then run chunk_gated_delta_rule with in-kernel L2 norm disabled.
        T = FLA_CHUNK_SIZE
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
            output_g_exp=False,
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

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core conv1d + recurrent attention (standard path).

        Args:
            mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)
            b: beta gating vector                   (num_tokens, num_heads)
            a: alpha gating vector                  (num_tokens, num_heads)
            core_attn_out: Pre-allocated output buffer for attention results.

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
        if gdn_step_plan.LAZY:
            # GGM_LAZY=1: this path reads the deferred FLA / Triton-conv metadata
            gdn_step_plan.fill_lazy(attn_metadata)
        if _GDN_STATE_COMMIT_DEFERRED:
            # commit pending token logs of the non-spec slots before they are read
            gdn_state_commit.forward_core_prologue(self, attn_metadata)

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
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
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
        num_accepted_tokens = attn_metadata.num_accepted_tokens

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
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                a_spec = a.index_select(0, spec_token_indx)
                b_spec = b.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
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
        if attn_metadata.num_prefills > 0:
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

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

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
            if spec_sequence_masks is not None:
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
                output_g_exp=False,
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
        if spec_sequence_masks is not None:
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
                cu_seqlens=attn_metadata.prefill_query_start_loc,
                chunk_indices=attn_metadata.chunk_indices,
                chunk_offsets=attn_metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
            )
            # Init cache
            ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)

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
            core_attn_out.index_copy_(0, spec_token_indx, core_attn_out_spec.squeeze(0))
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
    ) -> None:
        state_indices = attn_metadata.spec_state_indices_tensor
        cu_seqlens = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert state_indices is not None
        assert cu_seqlens is not None
        assert num_accepted_tokens is not None

        num_requests = attn_metadata.num_spec_decodes
        num_actual_tokens = attn_metadata.num_actual_tokens
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
        self._forward_core_decode_spec_post_conv_fused_norm(
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
    ) -> None:
        state_indices = attn_metadata.spec_state_indices_tensor
        cu_seqlens = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert state_indices is not None
        assert cu_seqlens is not None
        assert num_accepted_tokens is not None

        num_requests = attn_metadata.num_spec_decodes
        ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state_indices=state_indices[:num_requests],
            cu_seqlens=cu_seqlens[: num_requests + 1],
            num_accepted_tokens=num_accepted_tokens[:num_requests],
            state=self.kv_cache[1],
            output_gate=output_gate,
            norm_weight=self.norm.weight,
            out=core_attn_out,
            scale=self.head_k_dim**-0.5,
            norm_eps=self.layer_norm_epsilon,
            output_gate_activation=self.norm.activation,
        )

    def _forward_core_fused_norm_packed(
        self,
        mixed_qkvz: torch.Tensor,
        ba: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        if gdn_layer_graphs.ENABLED and gdn_layer_graphs.forward_packed(
            self, mixed_qkvz, ba, core_attn_out
        ):
            # VLLM_GDN_LAYER_GRAPHS=1: served by this layer's CUDA graph
            return
        if norm_quant.NQF and norm_quant.gdn_packed_enabled(self, core_attn_out):
            # NQF=1: run the core with the gated RMSNorm writing out_proj's
            # MXFP8 input into static buffers, quantize the remaining rows and
            # stash the result for out_proj.
            norm_quant.gdn_forward_core_fused_norm_packed(
                self,
                self._forward_core_fused_norm_packed_impl,
                mixed_qkvz,
                ba,
                core_attn_out,
            )
            return
        self._forward_core_fused_norm_packed_impl(mixed_qkvz, ba, core_attn_out)

    def _forward_core_fused_norm_packed_impl(
        self,
        mixed_qkvz: torch.Tensor,
        ba: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkvz[:, :qkv_size], 0)
            self._gdn_fusion_warmup(mixed_qkvz)
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
        )

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

    # ------------------------------------------------------------------
    # zero-copy mixed path
    # ------------------------------------------------------------------
    def _verify_mixed_fastpath(
        self, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata
    ) -> None:
        """Debug (VLLM_GDN_FASTPATH_VERIFY=N): run fast and stock paths on the same
        inputs/state slots for the first N eligible calls and log the differences.
        The stock result is kept."""
        md = attn_metadata
        slots = [md.prefill_state_indices.long()]
        if md.spec_state_indices_tensor is not None:
            slots.append(md.spec_state_indices_tensor.flatten().long())
        slots = torch.cat(slots)
        if slots.unique().numel() != slots.numel() or bool((slots <= 0).any()):
            # dummy/profile runs share slots; not a meaningful comparison
            self._forward_core_mixed_fastpath(
                mixed_qkv=mixed_qkv, b=b, a=a, output_gate=output_gate,
                core_attn_out=core_attn_out, attn_metadata=md,
            )
            return
        _GDN_VERIFY_LEFT[0] -= 1
        conv, ssm = self.kv_cache[0], self.kv_cache[1]
        conv0, ssm0 = conv[slots].clone(), ssm[slots].clone()
        qkvz_rows = mixed_qkv.clone()
        out_f = torch.zeros_like(core_attn_out)
        self._forward_core_mixed_fastpath(
            mixed_qkv=mixed_qkv, b=b, a=a, output_gate=output_gate,
            core_attn_out=out_f, attn_metadata=md,
        )
        conv_f, ssm_f = conv[slots].clone(), ssm[slots].clone()
        conv[slots] = conv0
        ssm[slots] = ssm0
        mixed_qkv.copy_(qkvz_rows)
        self._forward_core(
            mixed_qkv=mixed_qkv, b=b.contiguous(), a=a.contiguous(),
            core_attn_out=core_attn_out,
        )
        n = md.num_actual_tokens
        self._rms_norm_gated_cuda(
            core_attn_out[:n], output_gate[:n], core_attn_out[:n]
        )
        S = md.num_spec_decode_tokens

        def rel(x, y):
            x, y = x.float(), y.float()
            return ((x - y).norm() / (y.norm() + 1e-12)).item()

        logger.warning(
            "gdn_mixed_fastpath verify: %s S=%d P=%d seqs=%d spec_rel=%.2e prefill_rel=%.2e "
            "ssm_rel=%.2e conv_rel=%.2e",
            self.prefix, S, md.num_prefill_tokens, md.num_prefills,
            rel(out_f[:S], core_attn_out[:S]) if S else 0.0,
            rel(out_f[S:n], core_attn_out[S:n]),
            rel(ssm_f, ssm[slots]), rel(conv_f, conv[slots]),
        )

    def _gdn_fusion_warmup(self, mixed_qkvz: torch.Tensor) -> None:
        """Compile the mixed fast-path Triton kernels during the profile run so no
        JIT happens while serving (one variant each; num_seqs/T unspecialized)."""
        global _GDN_FUSION_WARMED
        if _GDN_FUSION_WARMED or not _GDN_MIXED_FASTPATH or mixed_qkvz.size(0) < 8:
            return
        _GDN_FUSION_WARMED = True
        dev = mixed_qkvz.device
        H = self.num_k_heads // self.tp_size
        HV = self.num_v_heads // self.tp_size
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        T = 8
        x = torch.zeros(T, mixed_qkvz.size(1), device=dev, dtype=mixed_qkvz.dtype)
        conv_w = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        conv_dim = conv_w.size(0)
        rows = self.conv_kernel_size - 1 + 3
        # same layout class as the real cache (DS: [slots, dim, rows]; SD transposed)
        if is_conv_state_dim_first():
            cs = torch.zeros(2, conv_dim, rows, device=dev, dtype=mixed_qkvz.dtype)
        else:
            cs = torch.zeros(2, rows, conv_dim, device=dev,
                             dtype=mixed_qkvz.dtype).transpose(-1, -2)
        ssm_dtype = self.get_state_dtype()[1]
        idx = torch.tensor([0, 1], device=dev, dtype=torch.int32)
        hinit = torch.tensor([True, False], device=dev)
        cu = torch.tensor([0, 3, T], device=dev, dtype=torch.int32)
        ba = torch.zeros(T, 2 * HV, device=dev, dtype=mixed_qkvz.dtype)
        b, a = ba[:, :HV], ba[:, HV:]
        if _GDN_FUSED_CONV:
            gdn_fused_conv_post_conv(
                x[:, :qkv_size], conv_w, cs, idx, hinit, cu, 2, a, b,
                self.A_log, self.dt_bias, H, self.head_k_dim, self.head_v_dim,
            )
        o = torch.zeros(T, HV, self.head_v_dim, device=dev, dtype=mixed_qkvz.dtype)
        z = x[:, qkv_size:].reshape(T, HV, self.head_v_dim)
        gdn_gated_rmsnorm_(o, z, self.norm.weight, self.layer_norm_epsilon,
                           self.norm.activation)
        st = torch.zeros(2, HV, self.head_v_dim, self.head_k_dim, device=dev,
                         dtype=ssm_dtype)
        gdn_zero_state_slots(st, idx, hinit)

    def _can_use_mixed_fastpath(self, attn_metadata: GDNAttentionMetadata) -> bool:
        if not _GDN_MIXED_FASTPATH or attn_metadata.num_prefills <= 0:
            return False
        if attn_metadata.num_decodes != 0 or self.gdn_prefill_backend != "flashinfer":
            return False
        if self.kv_cache[1].dtype not in FUSED_GDN_STATE_DTYPES:
            return False
        if self.norm.activation not in ("silu", "sigmoid"):
            return False
        num_spec_tokens = attn_metadata.num_spec_decode_tokens
        if (
            num_spec_tokens + attn_metadata.num_prefill_tokens
            != attn_metadata.num_actual_tokens
        ):
            return False
        if attn_metadata.spec_sequence_masks is None:
            return num_spec_tokens == 0
        state_indices = attn_metadata.spec_state_indices_tensor
        return (
            attn_metadata.spec_tokens_are_prefix
            and attn_metadata.num_spec_decodes > 0
            and self.gdn_decode_kernel == "cuda"
            and self.num_v_heads % self.num_k_heads == 0
            and self.num_v_heads // self.num_k_heads in (1, 2, 3, 4, 8)
            and state_indices is not None
            and state_indices.size(1) <= MAX_FUSED_GDN_MTP_TOKENS
            and hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")
        )

    def _forward_core_mixed_fastpath(
        self,
        mixed_qkv: torch.Tensor,  # [T, qkv] strided view of in_proj output
        b: torch.Tensor,  # [T, HV] strided view
        a: torch.Tensor,  # [T, HV] strided view
        output_gate: torch.Tensor,  # [T, HV, V] strided view
        core_attn_out: torch.Tensor,  # [T_padded, HV, V] contiguous
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        """Mixed prefill(+MTP) GDN core without gathers/scatters.

        Spec tokens are [0, S) and prefill tokens [S, S+P). Spec decodes run the
        fused CUDA MTP kernel (conv-update in place + recurrence + gated RMSNorm)
        exactly as in decode-only steps. Prefill runs conv1d -> fused post-conv
        (l2norm, exp(g), beta) -> FlashInfer chunked GDN writing straight into
        core_attn_out[S:] and reading/writing the state pool by slot index,
        then a strided gated-RMSNorm in place. Replaces 6 index_select, 2
        index_copy, 2 .contiguous(), the rearrange/cat copies, the z reshape
        copy, the initial-state gather/mask/index_put and the dtype casts.
        """
        logger.info_once("gdn_mixed_fastpath: zero-copy mixed GDN fast path active")
        S = attn_metadata.num_spec_decode_tokens
        P = attn_metadata.num_prefill_tokens
        N = S + P
        ssm_state = self.kv_cache[1]
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        # ---- spec-decode part (tokens [0, S)) ----
        if S > 0:
            state_indices = attn_metadata.spec_state_indices_tensor
            cu_seqlens = attn_metadata.spec_query_start_loc
            num_accepted_tokens = attn_metadata.num_accepted_tokens
            num_requests = attn_metadata.num_spec_decodes
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv[:S],
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
            if _GDN_MIXED_SPEC_TRITON:
                # bit-exact with the stock mixed path (FLA Triton recurrence)
                q_s, k_s, v_s = self.rearrange_mixed_qkv(mixed_qkv_spec)
                o_s, _ = fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a[:S].contiguous(),
                    b=b[:S].contiguous(),
                    dt_bias=self.dt_bias,
                    q=q_s,
                    k=k_s,
                    v=v_s,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=cu_seqlens[: num_requests + 1],
                    ssm_state_indices=state_indices,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
                core_attn_out[:S].copy_(o_s.squeeze(0))
                gdn_gated_rmsnorm_(
                    core_attn_out[:S],
                    output_gate[:S],
                    self.norm.weight,
                    self.layer_norm_epsilon,
                    self.norm.activation,
                )
            else:
                ops.fused_gdn_decode_post_conv_mtp(
                    mixed_qkv=mixed_qkv_spec,
                    a=a[:S],
                    b=b[:S],
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    state_indices=state_indices[:num_requests],
                    cu_seqlens=cu_seqlens[: num_requests + 1],
                    num_accepted_tokens=num_accepted_tokens[:num_requests],
                    state=ssm_state,
                    output_gate=output_gate[:S],
                    norm_weight=self.norm.weight,
                    out=core_attn_out[:S],
                    scale=self.head_k_dim**-0.5,
                    norm_eps=self.layer_norm_epsilon,
                    output_gate_activation=self.norm.activation,
                )

        # ---- prefill part (tokens [S, N)) ----
        if _GDN_FUSED_CONV:
            q, k, v, g_exp, beta = gdn_fused_conv_post_conv(
                mixed_qkv[S:N],
                conv_weights,
                conv_state,
                # block_table[:, 0] is a strided column when there are no spec
                # decodes; the Triton/FlashInfer kernels index it densely.
                attn_metadata.non_spec_state_indices_tensor.contiguous(),
                attn_metadata.has_initial_state.contiguous(),
                attn_metadata.non_spec_query_start_loc,
                attn_metadata.num_prefills,
                a[S:N],
                b[S:N],
                self.A_log,
                self.dt_bias,
                self.num_k_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            q, k, v, g_exp, beta = self._prefill_conv_post_conv(
                mixed_qkv, a, b, S, N, conv_weights, conv_state, attn_metadata
            )
        prefill_out = core_attn_out[S:N]
        self._fi_prefill_state_pool(
            q, k, v, g_exp, beta, prefill_out, ssm_state, attn_metadata
        )
        gdn_gated_rmsnorm_(
            prefill_out,
            output_gate[S:N],
            self.norm.weight,
            self.layer_norm_epsilon,
            self.norm.activation,
        )

    def _prefill_conv_post_conv(
        self, mixed_qkv, a, b, S, N, conv_weights, conv_state, attn_metadata
    ):
        conv_out = causal_conv1d_fn(
            mixed_qkv[S:N].transpose(0, 1),
            conv_weights,
            self.conv1d.bias,
            activation=self.activation,
            conv_states=conv_state,
            has_initial_state=attn_metadata.has_initial_state,
            cache_indices=attn_metadata.non_spec_state_indices_tensor,
            query_start_loc=attn_metadata.non_spec_query_start_loc,
            metadata=attn_metadata,
        ).transpose(0, 1)
        assert conv_out.stride(1) == 1, "conv1d output must be channel-last"
        q, k, v, g_exp, beta = fused_post_conv_prep(
            conv_output=conv_out,
            a=a[S:N],
            b=b[S:N],
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=self.num_k_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        # torch.exp (accurate expf) as in the stock fi_chunk_gated_delta_rule
        return q, k, v, torch.exp(g_exp), beta

    def _fi_prefill_state_pool(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_exp: torch.Tensor,
        beta: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        from flashinfer.gdn_prefill import (
            chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
        )

        slots = attn_metadata.prefill_state_indices
        has_init = attn_metadata.prefill_has_initial_state
        assert slots is not None and has_init is not None
        slots = slots.contiguous()  # may be a strided block_table column
        has_init = has_init.contiguous()
        cu_seqlens = attn_metadata.prefill_query_start_loc
        assert slots is not None and has_init is not None and cu_seqlens is not None
        want_cp = _gdn_fi_want_cp(cu_seqlens.numel() - 1, q.size(0),
                                  getattr(attn_metadata, "prefill_max_seqlen", 0))

        if _GDN_FI_STATE_POOL and (not want_cp or _GDN_FI_CP_POOL):
            # Indexed state-pool I/O: read initial state from ssm_state[slot],
            # write final state back in place. Slots of fresh sequences must
            # start from zero.
            gdn_zero_state_slots(ssm_state, slots, has_init)
            # non-CP steps with few / imbalanced sequences: V-split kernel (bitwise identical)
            if _GDN_FI_VSPLIT and not want_cp and _gdn_vsplit_call(
                q, k, v, g_exp, beta, out, ssm_state, slots, cu_seqlens, attn_metadata
            ):
                return
            chunk_gated_delta_rule_fi(
                q=q,
                k=k,
                v=v,
                g=g_exp,
                beta=beta,
                initial_state=ssm_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                output=out,
                output_state=ssm_state,
                use_cp=want_cp,  # the CP path also supports the indexed pool
                state_indices=slots,
                **_gdn_fi_maxlen_kw(want_cp, attn_metadata),
            )
            return
        # CP path (and VLLM_GDN_FI_STATE_POOL=0): gather/scatter the states, but
        # still write the output in place.
        initial_state = torch.where(
            has_init.view(-1, 1, 1, 1),
            ssm_state[slots].to(torch.float32),
            0.0,
        )
        _, final_state = chunk_gated_delta_rule_fi(
            q=q,
            k=k,
            v=v,
            g=g_exp,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            output=out,
            use_cp=want_cp,
            **_gdn_fi_maxlen_kw(want_cp, attn_metadata),
        )
        ssm_state[slots] = final_state.to(ssm_state.dtype)

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

    def _forward_core_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)
        if gdn_step_plan.ENABLED and gdn_step_plan.forward_core_fused_norm(
            self, attn_metadata, attn_metadata_raw, mixed_qkv, b, a, output_gate,
            core_attn_out
        ):
            # GGM=1: mixed step ran on the per-step plan (same kernels and order)
            return
        if _GDN_STATE_COMMIT_DEFERRED and gdn_state_commit.fused_norm_prologue(
            self, attn_metadata, mixed_qkv, b, a, output_gate, core_attn_out
        ):
            # spec rows were not a token prefix: handled on a spec-first permutation
            return
        if (
            self._can_use_fused_gdn_mtp_decode(attn_metadata)
            and attn_metadata.num_prefills == 0
        ):
            self._forward_core_decode_spec_fused_norm(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                output_gate=output_gate,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )
            return
        if self._can_use_mixed_fastpath(attn_metadata):
            if _GDN_VERIFY_LEFT[0] > 0:
                self._verify_mixed_fastpath(
                    mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata
                )
                return
            self._forward_core_mixed_fastpath(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                output_gate=output_gate,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )
            return
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b.contiguous(),
            a=a.contiguous(),
            core_attn_out=core_attn_out,
        )
        num_actual_tokens = attn_metadata.num_actual_tokens
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
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core_fused_norm_packed(
        mixed_qkvz=mixed_qkvz,
        ba=ba,
        core_attn_out=core_attn_out,
    )


direct_register_custom_op(
    op_name="qwen_gdn_attention_core_fused_norm_packed",
    op_func=qwen_gdn_attention_core_fused_norm_packed,
    mutates_args=["core_attn_out"],
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
