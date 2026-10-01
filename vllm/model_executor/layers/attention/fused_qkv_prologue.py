# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-attention "QKV prologue" for Qwen3-Next / Qwen3.5: QK-RMSNorm +
partial NeoX RoPE + gate copy + FP8 query quant + FP8 paged KV-cache write in
ONE Triton kernel (custom op vllm::ews_qkv_prologue).

Replaces, per full-attention layer (and the MTP layer), every step: the
fused qk-norm/rope/gate kernel, Inductor's Q -> FP8 kernel and
reshape_and_cache_flash (which runs eagerly in mixed steps because
vllm::unified_kv_cache_update is a piecewise split op); this also removes
one piecewise-graph split point per attention layer.

Bit-exactness (by construction):
  * the per-(token, head) norm/RoPE code is the stock kernel's code with the
    same 1-D [HEAD_BLOCK] tile, the same num_warps (HEAD_BLOCK // 64) and the
    same op order, so the fp32 reduction is identical;
  * q:  stock stores fp32 -> bf16 (RTNE) and Inductor then computes
        fp8(clamp(f32(q_bf16) * (1.0 / q_scale), -448, 448)); done exactly
        that way in registers;
  * k:  the bf16 k_out is still written (identical); the cache gets
        fp8e4m3(f32(k_bf16) / k_scale) with IEEE division + satfinite RN
        conversion, which is what reshape_and_cache_flash does
        (__nv_cvt_float_to_fp8(x / scale, __NV_SATFINITE, __NV_E4M3));
  * v:  fp8e4m3(f32(v_bf16) / v_scale) likewise, read straight from the qkv
        GEMM output;
  * slot < 0 (padding) -> no cache write, like the CUDA kernel;
  * gate: still copied to a contiguous buffer (the fused gate-mul + MXFP8
    quant of the o_proj input requires a contiguous gate).

Only active for layers with: FlashInfer backend with a separate KV-update op,
fp8 (e4m3) KV cache, query quant enabled (trtllm-gen FP8-Q path), no KV
sharing, the fused qk-norm-rope-gate path eligible, power-of-2 head_dim.
Everything else keeps the stock path.

Env (read at import): EWS=1 and VLLM_FUSED_QKV_PROLOGUE (default 1) enable
it; EWS_QKV_TPP = tokens per program (default 1 == the stock grid).
Kernel source kept verbatim (Triton cache keys hash the source).
"""

# ruff: noqa: E501
# fmt: off
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import _encode_layer_name
from vllm.lcd_pdl.triton_switch import lcd_pdl_triton_on as _lcd_pdl_on  # noqa: E402

logger = init_logger(__name__)

ENABLED = (os.environ.get("EWS", "0") == "1"
           and os.environ.get("VLLM_FUSED_QKV_PROLOGUE", "1") == "1")
TPP = int(os.environ.get("EWS_QKV_TPP", "1"))  # tokens per program (static unroll; 1 == stock grid)
# gb300 glue: GLUE_EWS_HG=<n> (> 0) runs glue_kernels._glue_ews_qkv_kernel with n heads per program (2-D tiles,
# 4*n warps, same per-head reduction layout -> bit-identical); GLUE_EWS_TPP tokens per program (default 1).
GLUE_EWS_HG = int(os.environ.get("GLUE_EWS_HG", "0"))
GLUE_EWS_TPP = int(os.environ.get("GLUE_EWS_TPP", "1"))
# GLUE_EWS_NOGATE=1 (with GLUE_EWS_HG > 0, NQF=1): no contiguous gate copy; the o_proj gate-mul + MXFP8 op reads the
# gate straight from the [q | gate]-interleaved QKV rows (nqf::gate_mul_mxfp8_qkv).
GLUE_EWS_NOGATE = GLUE_EWS_HG > 0 and os.environ.get("GLUE_EWS_NOGATE", "0") == "1"
STATS = {"calls": 0, "layers": 0}


@triton.jit
def _ews_qkv_kernel(
    qkv_ptr, qkv_stride_t,
    q8_ptr, k_out_ptr, gate_out_ptr,
    q_weight_ptr, k_weight_ptr, cos_sin_cache_ptr, positions_ptr,
    q8_stride_t, k_out_stride_t, gate_out_stride_t, cache_stride_p,
    positions_stride_m, positions_stride_t,
    slot_ptr, k_cache_ptr, v_cache_ptr, block_size,
    kc_stride_b, kc_stride_p, kc_stride_h, vc_stride_b, vc_stride_p, vc_stride_h,
    q_scale_ptr, k_scale_ptr, v_scale_ptr,
    n_tokens, n_slots,
    num_q_heads: tl.constexpr, num_kv_heads: tl.constexpr, head_dim: tl.constexpr,
    rotary_dim: tl.constexpr, half_rotary: tl.constexpr, eps: tl.constexpr, norm_beta: tl.constexpr,
    INPUT_DTYPE: tl.constexpr, HEAD_BLOCK: tl.constexpr, ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr, HAS_MROPE: tl.constexpr, MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr, WRITE_KV: tl.constexpr, TPP: tl.constexpr, launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)
    k_off: tl.constexpr = num_q_heads * 2 * head_dim
    v_off: tl.constexpr = num_q_heads * 2 * head_dim + num_kv_heads * head_dim
    for tt in tl.static_range(TPP):
        token = tl.program_id(0) * TPP + tt
        if token < n_tokens:
            row = qkv_ptr + token.to(tl.int64) * qkv_stride_t
            if is_k:
                in_base = row + k_off + local_head * head_dim
                w_ptr = k_weight_ptr
            else:
                in_base = row + local_head * 2 * head_dim
                w_ptr = q_weight_ptr

            # --- RMSNorm (stock code) ---
            head_offs = tl.arange(0, HEAD_BLOCK)
            head_mask = head_offs < head_dim
            x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
            var = tl.sum(x * x, axis=0) / head_dim
            inv_rms = tl.rsqrt(var + eps)
            w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32) + norm_beta
            x_norm = (x * inv_rms * w).to(INPUT_DTYPE).to(tl.float32)

            # --- partial RoPE (stock code) ---
            rot_offs = tl.arange(0, ROT_HALF_BLOCK)
            rot_mask = rot_offs < half_rotary
            x_rot1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
            x_rot2 = tl.load(in_base + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
            w_rot1 = tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32) + norm_beta
            w_rot2 = tl.load(w_ptr + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(tl.float32) + norm_beta
            x_rot1 = (x_rot1 * inv_rms * w_rot1).to(INPUT_DTYPE).to(tl.float32)
            x_rot2 = (x_rot2 * inv_rms * w_rot2).to(INPUT_DTYPE).to(tl.float32)
            pos_t = tl.load(positions_ptr + token * positions_stride_t).to(tl.int64)
            if HAS_MROPE:
                pos_h = tl.load(positions_ptr + positions_stride_m + token * positions_stride_t).to(tl.int64)
                pos_w = tl.load(positions_ptr + 2 * positions_stride_m + token * positions_stride_t).to(tl.int64)
                is_h = (rot_offs % 3 == 1) & (rot_offs < 3 * MROPE_SECTION_H)
                is_w = (rot_offs % 3 == 2) & (rot_offs < 3 * MROPE_SECTION_W)
                pos = tl.where(is_h, pos_h, tl.where(is_w, pos_w, pos_t))
            else:
                pos = pos_t
            cache_offset = pos * cache_stride_p
            cos = tl.load(cos_sin_cache_ptr + cache_offset + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
            sin = tl.load(cos_sin_cache_ptr + cache_offset + half_rotary + rot_offs, mask=rot_mask,
                          other=0.0).to(tl.float32)
            # stock stores o1/o2 (fp32) into a bf16 tensor -> RTNE rounding; do it explicitly
            o1 = (x_rot1 * cos - x_rot2 * sin).to(INPUT_DTYPE).to(tl.float32)
            o2 = (x_rot2 * cos + x_rot1 * sin).to(INPUT_DTYPE).to(tl.float32)

            if is_k:
                # bf16 k_out (unchanged semantics) + fp8 paged-cache write of K and V
                ko = k_out_ptr + token * k_out_stride_t + local_head * head_dim
                if HAS_PASS:
                    tl.store(ko + head_offs, x_norm, mask=head_mask & (head_offs >= rotary_dim))
                tl.store(ko + rot_offs, o1, mask=rot_mask)
                tl.store(ko + half_rotary + rot_offs, o2, mask=rot_mask)
                if WRITE_KV:
                  if token < n_slots:
                      slot = tl.load(slot_ptr + token).to(tl.int64)
                      if slot >= 0:
                          blk = slot // block_size
                          off = slot - blk * block_size
                          k_scale = tl.load(k_scale_ptr)
                          v_scale = tl.load(v_scale_ptr)
                          kd = k_cache_ptr + blk * kc_stride_b + off * kc_stride_p + local_head * kc_stride_h
                          if HAS_PASS:
                              tl.store(kd + head_offs, tl.math.div_rn(x_norm, k_scale).to(tl.float8e4nv),
                                       mask=head_mask & (head_offs >= rotary_dim))
                          tl.store(kd + rot_offs, tl.math.div_rn(o1, k_scale).to(tl.float8e4nv), mask=rot_mask)
                          tl.store(kd + half_rotary + rot_offs, tl.math.div_rn(o2, k_scale).to(tl.float8e4nv),
                                   mask=rot_mask)
                          vv = tl.load(row + v_off + local_head * head_dim + head_offs, mask=head_mask,
                                       other=0.0).to(tl.float32)
                          vd = v_cache_ptr + blk * vc_stride_b + off * vc_stride_p + local_head * vc_stride_h
                          tl.store(vd + head_offs, tl.math.div_rn(vv, v_scale).to(tl.float8e4nv), mask=head_mask)
            else:
                # q -> fp8 exactly as Inductor's triton_poi_fused__to_copy_clamp_mul_reciprocal
                q_scale = tl.load(q_scale_ptr)
                r = 1.0 / q_scale
                qo = q8_ptr + token * q8_stride_t + local_head * head_dim
                if HAS_PASS:
                    qp = tl.minimum(tl.maximum(x_norm * r, -448.0, tl.PropagateNan.ALL), 448.0, tl.PropagateNan.ALL)
                    tl.store(qo + head_offs, qp.to(tl.float8e4nv), mask=head_mask & (head_offs >= rotary_dim))
                q1 = tl.minimum(tl.maximum(o1 * r, -448.0, tl.PropagateNan.ALL), 448.0, tl.PropagateNan.ALL)
                q2 = tl.minimum(tl.maximum(o2 * r, -448.0, tl.PropagateNan.ALL), 448.0, tl.PropagateNan.ALL)
                tl.store(qo + rot_offs, q1.to(tl.float8e4nv), mask=rot_mask)
                tl.store(qo + half_rotary + rot_offs, q2.to(tl.float8e4nv), mask=rot_mask)
                # gate copy (verbatim, stock)
                g = tl.load(in_base + head_dim + head_offs, mask=head_mask, other=0.0)
                tl.store(gate_out_ptr + token * gate_out_stride_t + local_head * head_dim + head_offs, g,
                         mask=head_mask)


def launch(qkv, positions, q_weight, k_weight, cos_sin_cache, eps, num_q_heads, num_kv_heads, head_dim,
           rotary_dim, mrope_section, norm_beta, q_scale, k_scale, v_scale, slot_mapping, k_cache, v_cache,
           tpp=None):
    """Returns (q_fp8 [T, H*D], k_out bf16 [T, KVH*D], gate bf16 [T, H*D]); writes K/V to the cache when
    slot_mapping is not None."""
    T = qkv.shape[0]
    dev = qkv.device
    q8 = torch.empty((T, num_q_heads * head_dim), dtype=torch.float8_e4m3fn, device=dev)
    k_out = torch.empty((T, num_kv_heads * head_dim), dtype=qkv.dtype, device=dev)
    gate = torch.empty((T, num_q_heads * head_dim), dtype=qkv.dtype, device=dev)
    if T == 0:
        return q8, k_out, gate
    has_mrope = positions.ndim == 2
    if has_mrope:
        pm, pt = positions.stride()
        mh, mw = mrope_section[1], mrope_section[2]
    else:
        pm, pt = 0, positions.stride(0)
        mh = mw = 0
    write_kv = slot_mapping is not None
    if write_kv:
        kc = k_cache.view(torch.float8_e4m3fn) if k_cache.dtype != torch.float8_e4m3fn else k_cache
        vc = v_cache.view(torch.float8_e4m3fn) if v_cache.dtype != torch.float8_e4m3fn else v_cache
        block_size = kc.shape[1]
        kcs = kc.stride()[:3]
        vcs = vc.stride()[:3]
        sm = slot_mapping
    else:
        kc = vc = q8
        block_size = 1
        kcs = vcs = (0, 0, 0)
        sm = q8
    tpp = tpp or TPP
    head_block = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(T, tpp), num_q_heads + num_kv_heads)
    _ews_qkv_kernel[grid](
        qkv, qkv.stride(0), q8, k_out, gate, q_weight, k_weight, cos_sin_cache, positions,
        q8.stride(0), k_out.stride(0), gate.stride(0), cos_sin_cache.stride(0), pm, pt,
        sm, kc, vc, block_size, kcs[0], kcs[1], kcs[2], vcs[0], vcs[1], vcs[2],
        q_scale, k_scale, v_scale, T, sm.shape[0] if write_kv else 0,
        num_q_heads, num_kv_heads, head_dim, rotary_dim, rotary_dim // 2, eps, norm_beta=norm_beta,
        INPUT_DTYPE=tl.bfloat16 if qkv.dtype == torch.bfloat16 else tl.float16,
        HEAD_BLOCK=head_block, ROT_HALF_BLOCK=triton.next_power_of_2(rotary_dim // 2),
        HAS_PASS=rotary_dim < head_dim, HAS_MROPE=has_mrope, MROPE_SECTION_H=mh, MROPE_SECTION_W=mw,
        WRITE_KV=write_kv, TPP=tpp, num_warps=max(1, head_block // 64), num_stages=2, launch_pdl=_lcd_pdl_on())
    return q8, k_out, gate


# ----------------------------------------------------------------------------------------------------------
# vLLM integration
# ----------------------------------------------------------------------------------------------------------
_CFG: dict = {}  # layer_name -> {"mod": Qwen3NextAttention}


def _op_impl(qkv: torch.Tensor, positions: torch.Tensor, layer_name: str) -> tuple[torch.Tensor, torch.Tensor,
                                                                                  torch.Tensor]:
    from vllm.model_executor.layers.attention.attention import get_attention_context
    from vllm.utils.torch_utils import _resolve_layer_name

    name = _resolve_layer_name(layer_name)
    m = _CFG[name]["mod"]
    _, attn_layer, kv_cache, slot_mapping = get_attention_context(name)
    k_cache = v_cache = None
    if slot_mapping is not None:
        # identical views to FlashInferImpl.do_kv_cache_update: (B, H, N, 2*hs) -> (B, N, H, hs) x2
        k_cache, v_cache = kv_cache.transpose(1, 2).split(m.head_dim, dim=-1)
    STATS["calls"] += 1
    if GLUE_EWS_HG > 0:
        from vllm.model_executor.layers.fusion import glue_kernels as G

        STATS["glue"] = STATS.get("glue", 0) + 1
        q8, k_out, gate = G.ews_launch(
            qkv, positions, m.q_norm.weight, m.k_norm.weight, m.rotary_emb.cos_sin_cache,
            m.q_norm.variance_epsilon, m.num_heads, m.num_kv_heads, m.head_dim, m.rotary_emb.rotary_dim,
            getattr(m.rotary_emb, "mrope_section", None) if positions.ndim == 2 else None, 1.0,
            attn_layer._q_scale, attn_layer._k_scale, attn_layer._v_scale, slot_mapping, k_cache, v_cache,
            tpp=GLUE_EWS_TPP, hg=GLUE_EWS_HG, gate_copy=not GLUE_EWS_NOGATE)
        if gate is None:  # GLUE_EWS_NOGATE: zero-width placeholder (the gate is read from qkv)
            gate = qkv.new_empty((qkv.shape[0], 0))
        return q8, k_out, gate
    return launch(qkv, positions, m.q_norm.weight, m.k_norm.weight, m.rotary_emb.cos_sin_cache,
                  m.q_norm.variance_epsilon, m.num_heads, m.num_kv_heads, m.head_dim, m.rotary_emb.rotary_dim,
                  getattr(m.rotary_emb, "mrope_section", None) if positions.ndim == 2 else None, 1.0,
                  attn_layer._q_scale, attn_layer._k_scale, attn_layer._v_scale, slot_mapping, k_cache, v_cache)


def _op_fake(qkv: torch.Tensor, positions: torch.Tensor, layer_name: str, q_dim: int, kv_dim: int):
    # must not touch layer_name (an opaque LayerName under torch.compile)
    T = qkv.shape[0]
    return (qkv.new_empty((T, q_dim), dtype=torch.float8_e4m3fn), qkv.new_empty((T, kv_dim)),
            qkv.new_empty((T, 0 if GLUE_EWS_NOGATE else q_dim)))


_REGISTERED = [False]


def register_op():
    if _REGISTERED[0]:
        return
    from vllm.utils.torch_utils import LayerNameType, direct_register_custom_op

    def ews_qkv_prologue(qkv: torch.Tensor, positions: torch.Tensor, layer_name: LayerNameType, q_dim: int,
                         kv_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _op_impl(qkv, positions, layer_name)

    def ews_qkv_prologue_fake(qkv: torch.Tensor, positions: torch.Tensor, layer_name: LayerNameType, q_dim: int,
                              kv_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _op_fake(qkv, positions, layer_name, q_dim, kv_dim)

    direct_register_custom_op(op_name="ews_qkv_prologue", op_func=ews_qkv_prologue, mutates_args=[],
                              fake_impl=ews_qkv_prologue_fake)
    _REGISTERED[0] = True


def _eligible(mod) -> str | None:
    """Returns None if eligible, else the reason."""
    try:
        attn = mod.attn
        if not getattr(mod, "use_fused_qk_norm_rope_gate", False):
            return "fused qk-norm-rope-gate path not eligible"
        if attn.kv_cache_dtype not in ("fp8", "fp8_e4m3"):
            return f"kv_cache_dtype={attn.kv_cache_dtype}"
        if attn.query_quant is None or not attn.impl.supports_quant_query_input:
            return "query quant not active"
        if attn.attn_backend.forward_includes_kv_cache_update:
            return "backend writes KV inside forward"
        if attn.kv_sharing_target_layer_name is not None:
            return "kv sharing"
        if type(attn.impl).__name__ != "FlashInferImpl":
            return f"impl={type(attn.impl).__name__}"
        if getattr(attn.impl, "is_kvcache_nvfp4", False):
            return "nvfp4 kv"
        if attn.head_size != attn.head_size_v or attn.head_size & (attn.head_size - 1):
            return "head size"
        if getattr(attn, "dcp_world_size", 1) not in (1, None):
            return "dcp"
        return None
    except Exception as e:  # pragma: no cover
        return f"error {e!r}"


def configure_attention(mod) -> None:
    """Called at the end of Qwen3NextAttention.__init__ when ENABLED: decide
    per layer and mark the Attention layer as pre-fused (its forward then
    skips the query quant and the KV-cache update)."""
    why = _eligible(mod)
    mod._ews_qkv_on = why is None
    if mod._ews_qkv_on:
        _CFG[mod.attn.layer_name] = {"mod": mod}
        mod.attn._ews_prefused = True
        STATS["layers"] += 1
        logger.debug("fused qkv prologue enabled for %s", mod.attn.layer_name)
        logger.info_once("fused qkv prologue enabled (tokens per program %d)", TPP)
    else:
        logger.info("fused qkv prologue not used for %s: %s",
                    getattr(mod.attn, "layer_name", "?"), why)


def project_qkv_gate(mod, qkv, positions):
    """Qwen3NextAttention._project_qkv_gate for a pre-fused layer: returns
    (q_fp8, k, v, gate); K/V are already in the FP8 paged cache."""
    if positions.ndim == 2 and not getattr(mod.rotary_emb, "mrope_section", None):
        positions = positions[0]
    q8, k, gate = torch.ops.vllm.ews_qkv_prologue(qkv, positions, _encode_layer_name(mod.attn.layer_name),
                                                  mod.q_size, mod.kv_size)
    v = qkv[:, mod.q_size * 2 + mod.kv_size:]
    if GLUE_EWS_NOGATE:
        # [q | gate] per head, row stride of qkv: the gate-mul op reads the gate columns in place
        gate = qkv[:, : mod.q_size * 2]
    return q8, k, v, gate


if ENABLED:
    register_op()
