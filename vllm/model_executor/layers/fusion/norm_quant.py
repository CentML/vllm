# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused (residual adds) + Gemma RMSNorm + MXFP8 activation quant for the
Qwen3.5 MoE decoder layer (online MXFP8, FlashInfer CuTe-DSL dense GEMMs +
trtllm MoE).

Dataflow per decoder layer when a layer is eligible (configure_decoder_layer):
  post_attention_layernorm(attn_out, res) -> torch.ops.nqf.post_norm
      (opaque custom op) writes the bf16 normed output (router gate /
      shared_expert_gate still read it), fp8 data, 128x4-swizzled scales
      (shared-expert gate_up) and linear scales (routed-MoE input); the
      residual is returned lazily as the pair (attn_out, res), exactly like
      Inductor, which never materializes it.
  MoE (MoERunner.forward) returns the pair (shared_out, routed_out) instead
      of their sum.
  next input_layernorm((shared, routed), (attn_out, res)) ->
      torch.ops.nqf.pre_norm: x = (shared + routed) + (attn_out + res) in
      fp32 (Inductor's order), stores the bf16 residual, the bf16 normed
      output (in_proj_ba reads it), fp8 data + swizzled scales for
      in_proj_qkvz / qkv_proj.
Consumer: vllm.mxfp8_quantize (the op every MXFP8 activation quant goes
through) first looks its input up in a one-entry stash keyed by
(data_ptr, shape, stride); on a hit it returns the pre-computed
(fp8, scales), which are bit-identical to what FlashInfer computes; on a
miss it runs the FlashInfer quant. The stash holds strong references, so the
looked-up memory cannot be recycled for another tensor. Producers and the
consumer are opaque custom ops, so Inductor cannot split or reorder them, and
the lookup runs at CUDA-graph capture time (Python), so replays use fixed
addresses.
Also fused into the same MXFP8 epilogue: the shared expert SiLU*mul (its
down_proj input), the full-attention sigmoid gate multiply (o_proj input)
and the GDN gated RMSNorm (GDN out_proj input, static buffers, see
gdn_forward_core_fused_norm_packed).

Env:
  NQF=1 enables (default off; any value other than "0" enables).
  VLLM_NORM_QUANT_FUSION_EMIT=0 keeps the fused norm ops but disables the
      quant epilogue / stash (default 1).
  VLLM_NORM_QUANT_FUSION_SILU=0 disables the shared-expert SiLU*mul fusion
      (default 1).
  VLLM_NORM_QUANT_FUSION_OUT_PROJ=0 disables the o_proj / GDN out_proj input
      fusions (default 1).
  QGF_FIN=1 (default off): MoE finalize fold. A deferred MoE runner whose
      shared expert is folded into the routed experts (SEG_FOLD=1) runs the
      trtllm MoE without its finalize kernel and returns a placeholder; the
      next pre_norm gathers and weights the expert outputs inside its fused
      kernel (bit-identical to finalize + pre_norm).
      VLLM_MOE_FINALIZE_FOLD_MAX_M (default 1024): above this many tokens the
      standalone finalize + pre_norm is used instead of the fused kernel.
      VLLM_MOE_FINALIZE_FOLD_FMA (default 1): finalize accumulates with FMA
      like the trtllm kernel.
  QGF_GDNM=1 (default off): in mixed steps, quantize all GDN out_proj input
      rows not written by the fused gated norm in one launch (bit-identical).
Numerics: bit-exact vs Inductor norm + FlashInfer mxfp8_quantize, as long
as the Inductor reduction config matches norm_quant_kernels.CONFIG.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion import norm_quant_kernels as K
from vllm.model_executor.layers.mamba.gdn import gdn_out_alloc, gdn_step_plan

logger = init_logger(__name__)

NQF = os.environ.get("NQF", "0") != "0"
EMIT = os.environ.get("VLLM_NORM_QUANT_FUSION_EMIT", "1") != "0"
_SILU_ENV = os.environ.get("VLLM_NORM_QUANT_FUSION_SILU", "1")
_OUT_PROJ_ENV = os.environ.get("VLLM_NORM_QUANT_FUSION_OUT_PROJ", "1")
_H_OK = (2048,)
STATS = {"hit_swz": 0, "hit_lin": 0, "miss": 0, "pre": 0, "post": 0}

# gb300 glue: latency-restructured (register-resident, same reduction) norm/quant and gate-mul kernels
# (glue_kernels.py; bit-identical). GLUE_NQRR=1 norm_quant_rr, GLUE_GMRR=1 gate_mul_quant_rr.
GLUE_NQRR = os.environ.get("GLUE_NQRR", "0") == "1"
GLUE_GMRR = os.environ.get("GLUE_GMRR", "0") == "1"
GLUE_GMRR_CFG = tuple(int(x) for x in os.environ.get("GLUE_GMRR_CFG", "1,8").split(","))  # rows/program, warps
# rows/program, warps of norm_quant_rr: 1 row x 4 warps keeps the [1, 1024] per-row reduction layout of the port
# config (2 rows x 8 warps) -> bit-identical (bench: -0.4..-1 us at decode M, -45 % at mixed M on GB300)
GLUE_NQRR_CFG = tuple(int(x) for x in os.environ.get("GLUE_NQRR_CFG", "1,4").split(","))
# GLUE_FNQ_XB=1: fused finalize + pre-norm + MXFP8 with 1 row x 4 warps per program (same kernel, same per-row layout)
GLUE_FNQ_XB = int(os.environ.get("GLUE_FNQ_XB", "0"))
QGF_GDNM = EMIT and os.environ.get("QGF_GDNM", "0") == "1"
QGF_FIN_MAXM = int(os.environ.get("VLLM_MOE_FINALIZE_FOLD_MAX_M", "1024"))
QGF_FIN = EMIT and os.environ.get("QGF_FIN", "0") == "1"
# handle data_ptr -> (handle, gemm2_out, weights, expanded_idx_to_permuted_idx)
_FIN: dict = {}


def fin_produce(handle, g2, wts, idx):
    """Register the unfinalized MoE output behind the placeholder `handle`
    (called by the folded MoE op when QGF_FIN=1)."""
    _FIN[handle.data_ptr()] = (handle, g2, wts, idx)
    STATS["fin_prod"] = STATS.get("fin_prod", 0) + 1
    if len(_FIN) > 8:  # never expected: a produced handle was not consumed
        STATS["fin_orphan"] = STATS.get("fin_orphan", 0) + 1
        _FIN.pop(next(iter(_FIN)))


def fin_take(t):
    e = _FIN.get(t.data_ptr()) if isinstance(t, torch.Tensor) else None
    if e is None or e[0].shape != t.shape:
        return None
    del _FIN[t.data_ptr()]
    return e

# --------------------------------------------------------------------- stash
_STASH: dict = {"key": None, "src": None, "q": None, "swz": None, "lin": None}


def _key(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()))


def _stash_set(src, q, swz, lin):
    _STASH.update(key=_key(src), src=src, q=q, swz=swz, lin=lin)


def lookup(x, is_sf_swizzled_layout, alignment):
    k = _STASH["key"]
    if (
        k is None
        or alignment not in (0, 32)
        or not isinstance(x, torch.Tensor)
        or x.dim() != 2
    ):
        return None
    if _key(x) != k:
        return None
    if is_sf_swizzled_layout:
        sf = _STASH["swz"]
        if sf is None:
            return None
        STATS["hit_swz"] += 1
        return _STASH["q"], sf
    sf = _STASH["lin"]
    if sf is None:
        return None
    STATS["hit_lin"] += 1
    return _STASH["q"], sf.view(x.size(0), -1)


def consume_stash(x, is_sf_swizzled_layout=False, alignment=0):
    """Called first by the vllm.mxfp8_quantize implementation when NQF=1:
    the pre-computed (fp8, scales) of x, or None (then the stock quant runs)."""
    hit = lookup(x, is_sf_swizzled_layout, alignment)
    if hit is None:
        STATS["miss"] += 1
    return hit


# ----------------------------------------------------------------------- ops
def _norm_quant(*args, **kw):
    if GLUE_NQRR and args[0].shape[-1] == 2 * K.CONFIG["RB"]:
        from vllm.model_executor.layers.fusion import glue_kernels as G

        STATS["glue_nqrr"] = STATS.get("glue_nqrr", 0) + 1
        cfg = {"XBLOCK": GLUE_NQRR_CFG[0], "RB": K.CONFIG["RB"], "num_warps": GLUE_NQRR_CFG[1]}
        return G.norm_quant_rr(*args, config=cfg, **kw)
    return K.norm_quant(*args, **kw)


def _post_norm(
    a: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float, emit: bool
) -> torch.Tensor:
    out, _, q, swz, lin = _norm_quant(
        a, r, w, eps, emit_q=emit, emit_swz=True, emit_lin=True
    )
    if emit:
        _stash_set(out, q, swz, lin)
    STATS["post"] += 1
    return out


def _post_norm_fake(
    a: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float, emit: bool
) -> torch.Tensor:
    return torch.empty_like(a)


def _pre_norm(
    s: torch.Tensor,
    f: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    emit: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    e = fin_take(s) if QGF_FIN else None
    if e is not None:
        # s is a deferred-finalize placeholder of the folded MoE: gather and
        # weight the expert outputs here (== trtllm finalize, then pre_norm).
        g2, wts, idx = e[1], e[2], e[3].view(-1)
        if s.shape[0] <= QGF_FIN_MAXM:
            if GLUE_FNQ_XB > 0:
                from vllm.model_executor.layers.fusion import glue_kernels as G

                out, res, q, swz = G.fin_norm_quant_rr(
                    g2, wts, idx, f, a, r, w, eps, GLUE_FNQ_XB, 4 * GLUE_FNQ_XB
                )
            else:
                out, res, q, swz = K.fin_norm_quant(g2, wts, idx, f, a, r, w, eps)
            STATS["fin_fused"] = STATS.get("fin_fused", 0) + 1
        else:  # large M: finalize + pre_norm is faster than the fused kernel
            s_m = K.finalize(g2, wts, idx, s.shape[0], s.shape[1])
            out, res, q, swz, _ = _norm_quant(
                a, r, w, eps, s=s_m, f=f, emit_q=True, emit_swz=True, emit_lin=False
            )
            STATS["fin_split"] = STATS.get("fin_split", 0) + 1
        _stash_set(out, q, swz, None)
        STATS["pre"] += 1
        return out, res
    out, res, q, swz, _ = _norm_quant(
        a, r, w, eps, s=s, f=f, emit_q=emit, emit_swz=True, emit_lin=False
    )
    if emit:
        _stash_set(out, q, swz, None)
    STATS["pre"] += 1
    return out, res


def _pre_norm_fake(
    s: torch.Tensor,
    f: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    emit: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(a), torch.empty_like(a)


def _silu_mul_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    STATS["silu"] = STATS.get("silu", 0) + 1
    return K.silu_mul_quant(x)


def _silu_mul_mxfp8_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, I = x.shape[0], x.shape[1] // 2
    nsf = I // 32
    padded_m = (M + 127) // 128 * 128
    return (
        x.new_empty((M, I), dtype=torch.float8_e4m3fn),
        x.new_empty((padded_m * ((nsf + 3) // 4 * 4),), dtype=torch.uint8),
    )


def _gate_mul_mxfp8(a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    if GLUE_GMRR and (a.shape[-1] & (a.shape[-1] - 1)) == 0:
        from vllm.model_executor.layers.fusion import glue_kernels as G

        STATS["glue_gmrr"] = STATS.get("glue_gmrr", 0) + 1
        out, q, sf = G.gate_mul_quant_rr(a, g, 0, GLUE_GMRR_CFG[0], GLUE_GMRR_CFG[1])
    else:
        out, q, sf = K.gate_mul_quant(a, g)
    _stash_set(out, q, sf, None)
    STATS["gate"] = STATS.get("gate", 0) + 1
    return out


def _gate_mul_mxfp8_fake(a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a)


def _gate_mul_mxfp8_qkv(a: torch.Tensor, g: torch.Tensor, head_dim: int) -> torch.Tensor:
    """gb300 glue: _gate_mul_mxfp8 with the gate read from the [q | gate]-interleaved QKV rows g ([T, 2N],
    row stride of the QKV projection output): gate column c = g[:, (c // D) * 2D + D + c % D]."""
    from vllm.model_executor.layers.fusion import glue_kernels as G

    out, q, sf = G.gate_mul_quant_rr(a, g, head_dim, GLUE_GMRR_CFG[0], GLUE_GMRR_CFG[1])
    _stash_set(out, q, sf, None)
    STATS["gate_qkv"] = STATS.get("gate_qkv", 0) + 1
    return out


def _gate_mul_mxfp8_qkv_fake(a: torch.Tensor, g: torch.Tensor, head_dim: int) -> torch.Tensor:
    return torch.empty_like(a)


def _fin_mat(t: torch.Tensor) -> torch.Tensor:
    """Deferred (unfinalized) MoE output -> the exact bf16 tensor the trtllm
    finalize would have produced (a copy of t if t is not a placeholder)."""
    e = fin_take(t)
    if e is None:
        STATS["fin_mat_passthru"] = STATS.get("fin_mat_passthru", 0) + 1
        return t.clone()
    STATS["fin_mat"] = STATS.get("fin_mat", 0) + 1
    return K.finalize(e[1], e[2], e[3].view(-1), t.shape[0], t.shape[1])


def _fin_mat_fake(t: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(t)


_LIB = None


def register_ops() -> None:
    """Register the nqf:: custom ops (once)."""
    global _LIB
    if _LIB is not None:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    _LIB = torch.library.Library("nqf", "FRAGMENT")
    direct_register_custom_op(
        "post_norm",
        _post_norm,
        mutates_args=[],
        fake_impl=_post_norm_fake,
        target_lib=_LIB,
        dispatch_key="CUDA",
    )
    direct_register_custom_op(
        "pre_norm",
        _pre_norm,
        mutates_args=[],
        fake_impl=_pre_norm_fake,
        target_lib=_LIB,
        dispatch_key="CUDA",
    )
    direct_register_custom_op(
        "gate_mul_mxfp8",
        _gate_mul_mxfp8,
        mutates_args=[],
        fake_impl=_gate_mul_mxfp8_fake,
        target_lib=_LIB,
        dispatch_key="CUDA",
    )
    direct_register_custom_op(
        "gate_mul_mxfp8_qkv",
        _gate_mul_mxfp8_qkv,
        mutates_args=[],
        fake_impl=_gate_mul_mxfp8_qkv_fake,
        target_lib=_LIB,
        dispatch_key="CUDA",
    )
    direct_register_custom_op(
        "silu_mul_mxfp8",
        _silu_mul_mxfp8,
        mutates_args=[],
        fake_impl=_silu_mul_mxfp8_fake,
        target_lib=_LIB,
        dispatch_key="CUDA",
    )
    direct_register_custom_op(
        "fin_mat",
        _fin_mat,
        mutates_args=[],
        fake_impl=_fin_mat_fake,
        target_lib=_LIB,
        dispatch_key="CUDA",
    )


# -------------------------------------------------------- shared-expert SiLU
class _SiluMulMxfp8(torch.nn.Module):
    """Shared expert act_fn: SiLU*mul + MXFP8 quant in one kernel. Returns a
    QuantizedActivation that down_proj's FlashInfer MXFP8 kernel consumes, so
    Qwen2MoeMLP.forward itself is unchanged."""

    def __init__(self, orig_act):
        super().__init__()
        self.orig_act = orig_act

    def forward(self, gate_up):
        if (
            EMIT
            and isinstance(gate_up, torch.Tensor)
            and gate_up.dim() == 2
            and gate_up.dtype == torch.bfloat16
            and gate_up.stride(-1) == 1
            and gate_up.shape[-1] == 1024
        ):
            from vllm.model_executor.layers.fusion.quant_activation import (
                QuantizedActivation,
            )
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                kMxfp8Dynamic,
            )

            q, sf = torch.ops.nqf.silu_mul_mxfp8(gate_up)
            return QuantizedActivation(
                data=q,
                scale=sf,
                orig_dtype=gate_up.dtype,
                orig_shape=q.shape,
                quant_key=kMxfp8Dynamic,
            )
        return self.orig_act(gate_up)


# ------------------------------------------------------------ norm producers
def is_fusable(t) -> bool:
    """Input accepted by the fused pre/post norm ops."""
    return (
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and t.dtype == torch.bfloat16
        and t.dim() == 2
        and t.shape[-1] in _H_OK
        and t.stride(-1) == 1
    )


# ---------------------------------------------------------- layer selection
def _dense_is_cutedsl_mxfp8(linear) -> bool:
    k = getattr(getattr(linear, "quant_method", None), "kernel", None)
    # MRO check: subclasses of the CuTe-DSL MXFP8 linear kernel consume the same
    # 128x4-swizzled activations through vllm.mxfp8_quantize / QuantizedActivation.
    return any(
        c.__name__ == "FlashInferCutedslMxfp8LinearKernel" for c in type(k).__mro__
    )


_MAX_TOKENS = [0]


def _set_max_tokens(vllm_config) -> None:
    if _MAX_TOKENS[0]:
        return
    try:
        mnbt = vllm_config.scheduler_config.max_num_batched_tokens
        caps = vllm_config.compilation_config.cudagraph_capture_sizes or [0]
        _MAX_TOKENS[0] = (max(mnbt, max(caps)) + 127) // 128 * 128
    except Exception:  # leave 0: the GDN out_proj fusion then stays off
        pass


def configure_decoder_layer(layer: torch.nn.Module, vllm_config) -> None:
    """Called at the end of Qwen3_5DecoderLayer.__init__ when NQF=1: flag the
    norms / MoE runner / attention of an eligible layer."""
    try:
        from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

        if not isinstance(getattr(layer, "mlp", None), Qwen3NextSparseMoeBlock):
            return
        if getattr(layer, "layer_scale", False) or getattr(
            layer, "use_attn_reduce_scatter_for_moe", False
        ):
            return
        dense = (
            layer.linear_attn.in_proj_qkvz
            if layer.layer_type == "linear_attention"
            else layer.self_attn.qkv_proj
        )
        se = layer.mlp.shared_expert
        if (
            se is None
            or not _dense_is_cutedsl_mxfp8(dense)
            or not _dense_is_cutedsl_mxfp8(se.gate_up_proj)
        ):
            logger.debug(
                "norm_quant_fusion: layer not eligible (%s)",
                type(getattr(getattr(dense, "quant_method", None), "kernel", None)),
            )
            return
        layer.input_layernorm._nqf_role = "pre"
        layer.post_attention_layernorm._nqf_role = "post"
        layer.mlp.experts._nqf_defer = True
        if (
            _dense_is_cutedsl_mxfp8(se.down_proj)
            and _SILU_ENV != "0"
            and type(se.act_fn).__name__ == "SiluAndMul"
        ):
            se.act_fn = _SiluMulMxfp8(se.act_fn)
        _set_max_tokens(vllm_config)
        if _OUT_PROJ_ENV != "0":
            if layer.layer_type == "full_attention" and _dense_is_cutedsl_mxfp8(
                layer.self_attn.o_proj
            ):
                layer.self_attn._nqf_gate = True
            if layer.layer_type == "linear_attention" and _dense_is_cutedsl_mxfp8(
                layer.linear_attn.out_proj
            ):
                layer.linear_attn._nqf_gdn = True
    except Exception as e:  # never break model construction; stock path stays
        logger.warning("norm_quant_fusion: decoder layer not configured: %r", e)


# ---------------------------------------------------- attention o_proj input
def gate_mul_fusable(attn: torch.nn.Module, attn_output, gate) -> bool:
    """Full-attention sigmoid(gate) * attn_output fused with the o_proj
    MXFP8 input quant (torch.ops.nqf.gate_mul_mxfp8)."""
    return (
        EMIT
        and getattr(attn, "_nqf_gate", False)
        and attn_output.dim() == 2
        and attn_output.dtype == torch.bfloat16
        and attn_output.shape[-1] % 1024 == 0
        and gate.numel() == attn_output.numel()
        and gate.is_contiguous()
        and attn_output.is_contiguous()
    )


def gate_mul_qkv_fusable(attn: torch.nn.Module, attn_output, g) -> bool:
    """gb300 glue (GLUE_EWS_NOGATE): the gate-mul + MXFP8 op on the interleaved [q | gate] QKV columns."""
    return (
        EMIT
        and getattr(attn, "_nqf_gate", False)
        and attn_output.dim() == 2
        and attn_output.dtype == torch.bfloat16
        and attn_output.is_contiguous()
        and g.dim() == 2
        and g.shape[0] == attn_output.shape[0]
        and g.shape[1] == 2 * attn_output.shape[1]
        and g.stride(-1) == 1
        and (attn_output.shape[-1] & (attn_output.shape[-1] - 1)) == 0
    )


# ---------------------------------------------------------- GDN out_proj input
# The GDN core op (qwen_gdn_attention_core_fused_norm_packed) runs eagerly
# between CUDA-graph pieces, so its fp8/scale outputs go to STATIC buffers
# (fixed addresses; one set shared by all GDN layers since out_proj consumes
# them before the next GDN layer runs). Every core-op call fills all rows
# (prefill rows fused into the gated RMSNorm epilogue, the rest - MTP-decode
# rows written by the CUDA kernel, padding - by a row quant kernel), so a
# consumer captured as a stash hit is always valid at replay.
_GDN_TGT: list = [None]
_GDN_BUF: dict = {}


def _gdn_bufs(K_, dev):
    key = (K_, dev)
    if key not in _GDN_BUF:
        n = _MAX_TOKENS[0]
        psc = (K_ // 32 + 3) // 4 * 4
        _GDN_BUF[key] = (
            torch.empty((n, K_), dtype=torch.float8_e4m3fn, device=dev),
            torch.empty((n * psc,), dtype=torch.uint8, device=dev),
            psc,
        )
    return _GDN_BUF[key]


def gdn_gated_rmsnorm_into_target(x, z, weight, eps, activation) -> bool:
    """Called first by qwen_gdn_linear_attn.gdn_gated_rmsnorm_ when NQF=1.
    If x is a row range of the core_attn_out of the GDN layer currently
    inside gdn_forward_core_fused_norm_packed, run the gated RMSNorm with the
    MXFP8 epilogue into the static buffers, record the rows as covered and
    return True; otherwise return False (the stock kernel runs)."""
    t = _GDN_TGT[0]
    if (
        t is not None
        and x.dim() == 3
        and x.is_contiguous()
        and x.shape[1] * x.shape[2] == t["K"]
        and x.dtype == torch.bfloat16
        and x.untyped_storage().data_ptr() == t["storage"]
    ):
        off = x.data_ptr() - t["base"]
        rb = t["K"] * x.element_size()
        if off % rb == 0 and z.stride(2) == 1 and z.stride(1) == x.shape[2]:
            r0 = off // rb
            K.gdn_gated_rmsnorm_quant_(
                x, z, weight, eps, activation, t["q"], t["sf"], r0, t["psc"]
            )
            t["covered"].append((r0, r0 + x.shape[0]))
            return True
    return False


def gdn_packed_enabled(layer: torch.nn.Module, core_attn_out: torch.Tensor) -> bool:
    """Enable predicate of the GDN out_proj input fusion for one core-op call."""
    T = core_attn_out.shape[0]
    K_ = (
        core_attn_out.shape[1] * core_attn_out.shape[2]
        if core_attn_out.dim() == 3
        else 0
    )
    return not (
        not (EMIT and getattr(layer, "_nqf_gdn", False))
        or K_ % 128
        or not _MAX_TOKENS[0]
        or T > _MAX_TOKENS[0]
        or core_attn_out.dtype != torch.bfloat16
        or not core_attn_out.is_contiguous()
    )


def gdn_uncovered_row_ranges(covered, pm: int) -> list[tuple[int, int]]:
    """Row ranges [lo, hi) of [0, pm) not written by the fused gated RMSNorm,
    in launch order (possibly empty ranges, which quant_rows skips)."""
    ranges = []
    lo = 0
    for a, b in sorted(covered):
        ranges.append((lo, a))
        lo = max(lo, b)
    ranges.append((lo, pm))
    return ranges


def gdn_quant_uncovered_rows(x2, q, sf, ranges, T: int, psc: int) -> None:
    """Plain MXFP8 row quant of the uncovered ranges, one launch per range."""
    if gdn_step_plan.MERGED_ROW_QUANT:
        _gdn_quant_uncovered_rows_merged(x2, q, sf, ranges, T, psc)
        return
    for lo, hi in ranges:
        K.quant_rows(x2, q, sf, lo, hi, T, psc)


def _gdn_quant_uncovered_rows_merged(x2, q, sf, ranges, T: int, psc: int) -> None:
    """VLLM_GDN_MERGED_ROW_QUANT (GGM_OG2=1): exactly two non-empty ranges
    (the spec-decode rows and the padding rows of a mixed step) with the same
    launch config run as ONE launch of the two-range row-quant kernel (same
    per-row math); otherwise one launch per non-empty range, in order.
    """
    pend = [(lo, hi) for lo, hi in ranges if hi > lo]
    if not pend:
        return
    stats = gdn_step_plan.TRIM_STATS
    if len(pend) == 2:
        (lo1, hi1), (lo2, hi2) = pend
        c1, c2 = K._qr_config(hi1 - lo1), K._qr_config(hi2 - lo2)
        if c1 == c2 and hi1 <= lo2:
            K.quant_rows2(x2, q, sf, (lo1, hi1), (lo2, hi2), T, psc, c1)
            stats["qrows_merged"] += 1
            logger.info_once("GDN uncovered-row quant: 2 launches merged into 1")
            return
    stats["qrows_single" if len(pend) == 1 else "qrows_multi"] += 1
    for lo, hi in pend:
        K.quant_rows(x2, q, sf, lo, hi, T, psc)


def gdn_forward_core_fused_norm_packed(
    layer: torch.nn.Module, core_fn, mixed_qkvz, ba, core_attn_out
) -> None:
    """The GDN core op with its out_proj MXFP8 input produced into static
    buffers: core_fn (the stock packed core) runs with this layer as the
    gated-RMSNorm target, then the rows the fused norm did not cover are
    quantized and the result is stashed for out_proj's vllm.mxfp8_quantize.
    Only called when gdn_packed_enabled(layer, core_attn_out)."""
    T = core_attn_out.shape[0]
    K_ = (
        core_attn_out.shape[1] * core_attn_out.shape[2]
        if core_attn_out.dim() == 3
        else 0
    )
    q, sf, psc = _gdn_bufs(K_, core_attn_out.device)
    t = {
        "K": K_,
        "base": core_attn_out.data_ptr(),
        "storage": core_attn_out.untyped_storage().data_ptr(),
        "q": q,
        "sf": sf,
        "psc": psc,
        "covered": [],
        "T": T,  # gb300 glue: rows of core_attn_out (GLUE_GSC_QO target check)
        "qo": False,  # set by the gdn_state_commit decode when it wrote q / sf of every row
    }
    _GDN_TGT[0] = t
    try:
        core_fn(mixed_qkvz, ba, core_attn_out)
    finally:
        _GDN_TGT[0] = None
    gdn_out_alloc.zero_pad_rows_late(core_attn_out, t["qo"])
    x2 = core_attn_out.view(T, K_)
    pm = (T + 127) // 128 * 128
    if QGF_GDNM and t["covered"]:
        # one launch for every row the fused gated norm did not write (real
        # uncovered rows, padding, scale padding)
        unc, lo = [], 0
        for a, b in sorted(t["covered"]):
            if a > lo:
                unc.append((lo, a))
            lo = max(lo, b)
        if lo < T:
            unc.append((lo, T))
        while len(unc) > 4:  # never expected; extra ranges via the row quant
            a, b = unc.pop()
            K.quant_rows(x2, q, sf, a, b, T, psc)
        K.gdn_fixup(x2, q, sf, None, T, pm, psc, unc)
        STATS["gdn_unc_rows"] = STATS.get("gdn_unc_rows", 0) + sum(
            b - a for a, b in unc
        )
    elif t["qo"] and not t["covered"]:
        # gb300 glue (GLUE_GSC_QO=1): the decode kernel already wrote the
        # MXFP8 data + swizzled scales of every row (padding rows zeroed),
        # bit-identical to the FlashInfer kernel below
        STATS["gdn_qo"] = STATS.get("gdn_qo", 0) + 1
    elif not t["covered"]:
        # decode-only / warmup: exactly the stock FlashInfer kernel, into the
        # static buffers
        K.fi_quant_into(x2, q[:T], sf[: pm * psc])
    else:
        gdn_quant_uncovered_rows(
            x2, q, sf, gdn_uncovered_row_ranges(t["covered"], pm), T, psc
        )
    _stash_set(x2, q[:T], sf[: pm * psc], None)
    STATS["gdn"] = STATS.get("gdn", 0) + 1
    STATS["gdn_fused_rows"] = STATS.get("gdn_fused_rows", 0) + sum(
        b - a for a, b in t["covered"]
    )


# ------------------------------------------------------------ compile cache
def compile_hash_factors() -> list[str]:
    """AOT compile-cache salts: the key does not include the traced sources,
    so a cache from the unfused graph would silently bypass these rewrites."""
    return [
        f"nqf-v1-emit{int(EMIT)}-silu{_SILU_ENV}",
        f"qgf-v3-m{int(QGF_GDNM)}-f{int(QGF_FIN)}-{QGF_FIN_MAXM}-c0",
        f"glue-v2-nqrr{int(GLUE_NQRR)}{GLUE_NQRR_CFG}-gmrr{int(GLUE_GMRR)}{GLUE_GMRR_CFG}-fnq{GLUE_FNQ_XB}",
    ]


if NQF:
    register_ops()
    logger.info("norm_quant_fusion enabled (emit=%s)", EMIT)
    if GLUE_NQRR or GLUE_GMRR or GLUE_FNQ_XB:
        logger.info(
            "[glue] norm_quant_rr=%s cfg=%s gate_mul_rr=%s cfg=%s fin_norm_quant rows/program=%s",
            GLUE_NQRR, GLUE_NQRR_CFG, GLUE_GMRR, GLUE_GMRR_CFG, GLUE_FNQ_XB,
        )
    if QGF_FIN:
        logger.info("norm_quant_fusion: MoE finalize fold enabled (QGF_FIN)")
    if QGF_GDNM:
        logger.info(
            "norm_quant_fusion: GDN single-launch row quant in mixed steps (QGF_GDNM)"
        )
