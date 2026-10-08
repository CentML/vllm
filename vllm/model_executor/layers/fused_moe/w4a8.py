# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in W4A8 fold for Qwen3-Next/3.5, including the MTP block.

PRECISION versus stock qwen-v2. An opaque, finalized folded op deliberately
bypasses the stock shared-gate/dual-input/deferred-finalize fusion contracts.
The load-time quantizer preserves packed layout and routing-weight gain.
"""
import torch
import torch.nn.functional as F

from vllm import w4fq
from vllm.logger import init_logger
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)
_PERM = {}
_LIB = None
_STATS = {"folded": 0, "stash": 0}


def enabled_for(prefix: str) -> bool:
    # Gain also applies to experts excluded from FP4 by coverage switches.
    return w4fq.GAIN_ON or w4fq.want(prefix)


def configure(block, prefix: str) -> None:
    """Reject unsupported geometries before quantization; mark only this block."""
    r = block.experts
    mc = r.moe_config
    re = r.routed_experts
    qm = re.quant_method
    if ((mc.tp_size, mc.dp_size, mc.ep_size, mc.pcp_size) != (1, 1, 1, 1)
            or mc.is_sequence_parallel or r.enable_dbo
            or block.enable_eplb or block.replicate_shared_expert
            or r.routed_input_transform is not None
            or r.routed_output_transform is not None
            or r.routed_scaling_factor != 1.0):
        raise ValueError("W4A8 requires TP/DP/EP/PCP=1, no DBO/EPLB/transforms")
    if (mc.num_experts != 256 or mc.experts_per_token != 8
            or block.shared_expert is None
            or type(qm).__name__ != "Mxfp8OnlineMoEMethod"
            or "trtllm" not in str(qm.fp8_backend).lower()
            or str(re.activation).lower().split('.')[-1] != "silu"):
        raise ValueError("W4A8 requires online TRTLLM MXFP8, SiLU, 256+1 experts/top-8")
    if (mc.in_dtype != torch.bfloat16 or mc.has_bias or mc.is_lora_enabled
            or any(value is not None for value in
                   (mc.swiglu_limit, mc.swiglu_alpha, mc.swiglu_beta))):
        raise ValueError("W4A8 requires BF16 input, no bias/LoRA/SwiGLU overrides")
    if (not getattr(r.router, "renormalize", False)
            or getattr(r.router, "scoring_func", None) != "softmax"):
        raise ValueError("W4A8 requires renormalized softmax routing")
    shared = block.shared_expert
    for layer in (shared.gate_up_proj, shared.down_proj):
        if type(layer.quant_method).__name__ != "Mxfp8OnlineLinearMethod":
            raise ValueError("W4A8 requires online MXFP8 shared-expert linears")
        layer._w4a8_stash = True
    re._w4a8_name = prefix
    r.accepts_quantized_input = False
    r._defer_shared_gate = False
    r._fse_fuse_gate = False
    mc.defer_moe_finalize_local = False


def stash_shared(layer) -> None:
    if getattr(layer, "_w4a8_stash", False):
        layer._w4a8_bf16 = layer.weight.detach().clone()
        _STATS["stash"] += 1


@torch.no_grad()
def quantize_routed(method, layer):
    """Called instead of stock quantization only for marked Qwen MoE blocks."""
    name = layer._w4a8_name
    fq = w4fq.want(name)
    gain = w4fq.gain_on()
    original13 = layer.w13_weight.detach().clone() if gain else None
    original2 = layer.w2_weight.detach().clone() if gain else None
    if fq:
        w4fq.apply_(layer.w13_weight.data, name + ".w13")
        w4fq.apply_(layer.w2_weight.data, name + ".w2")
    w13, s13 = method._quantize_mxfp8_moe_weight(layer.w13_weight)
    w2, s2 = method._quantize_mxfp8_moe_weight(layer.w2_weight)
    if fq:
        w4fq.verify_(layer.w13_weight.data, w13, s13, name + ".w13")
        w4fq.verify_(layer.w2_weight.data, w2, s2, name + ".w2")
        if w4fq.REAL:
            layer._w4_codes = (*w4fq.encode_mxfp4(layer.w13_weight.data),
                               *w4fq.encode_mxfp4(layer.w2_weight.data))
    if gain:
        layer._w4_gain = (w4fq.gain_from(original13, w13, s13) ** 2
                          * w4fq.gain_from(original2, w2, s2))
    return w13, w2, s13, s2


@torch.no_grad()
def fold_block(block):
    from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        prepare_fp8_moe_layer_for_fi,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    r = block.experts
    if hasattr(r, "_w4a8"):
        return
    re = r.routed_experts
    name = re._w4a8_name
    su = block.shared_expert
    gu, dn = su.gate_up_proj._w4a8_bf16, su.down_proj._w4a8_bf16
    h = block.gate.weight.shape[1]
    i = r.moe_config.intermediate_size_per_partition
    if (gu.shape != (2 * i, h) or dn.shape != (h, i)
            or block.gate.weight.dtype != torch.bfloat16
            or block.shared_expert_gate.weight.dtype != torch.bfloat16):
        raise ValueError("W4A8 requires matching shared/routed shapes and BF16 gates")
    fq = w4fq.want(name + ".shared", shared=True)
    gain = w4fq.gain_on()
    if gain:
        original_gu, original_dn = gu.clone(), dn.clone()
    if fq:
        w4fq.apply_(gu, name + ".shared.w13")
        w4fq.apply_(dn, name + ".shared.w2")
    q13, s13 = mxfp8_e4m3_quantize(gu.contiguous(), is_sf_swizzled_layout=False)
    q2, s2 = mxfp8_e4m3_quantize(dn.contiguous(), is_sf_swizzled_layout=False)
    if fq:
        w4fq.verify_(gu, q13, s13, name + ".shared.w13")
        w4fq.verify_(dn, q2, s2, name + ".shared.w2")
    combined_gain = None
    if gain:
        shared_gain = (w4fq.gain_from(original_gu, q13, s13) ** 2
                       * w4fq.gain_from(original_dn, q2, s2))
        combined_gain = torch.cat([
            re._w4_gain.float(), shared_gain.float(),
            torch.ones(3, dtype=torch.float32, device=gu.device),
        ]).contiguous()
        del re._w4_gain
        w4fq.GAIN_STATS["c"].append(combined_gain[:257].detach().cpu())
        w4fq.GAIN_STATS["blocks_applied"] += 1

    def append(big, one):
        pads = torch.zeros((3, *one.shape[1:]), dtype=one.dtype, device=one.device)
        # CPU and CUDA torch.cat need byte views for float8 on some builds.
        dtype = one.dtype
        return torch.cat([big.view(torch.uint8), one.view(torch.uint8),
                          pads.view(torch.uint8)], 0).view(dtype).contiguous()

    real = fq and w4fq.REAL
    if real:
        codes = re._w4_codes
        p13, ps13 = w4fq.encode_mxfp4(gu)
        p2, ps2 = w4fq.encode_mxfp4(dn)
        w13, ws13, w2, ws2 = w4fq.to_trtllm_mxfp4(
            append(codes[0], p13[None]), append(codes[1], ps13[None]),
            append(codes[2], p2[None]), append(codes[3], ps2[None]), _PERM,
        )
        del re._w4_codes
        w4fq.REAL_STATS["blocks_real"] += 1
    else:
        a13, a2, as13, as2 = prepare_fp8_moe_layer_for_fi(
            re, q13[None], q2[None], s13[None], None, s2[None], None, is_trtllm=True,
        )
        w13, w2 = append(re.w13_weight, a13), append(re.w2_weight, a2)
        ws13 = append(re.w13_weight_scale, as13)
        ws2 = append(re.w2_weight_scale, as2)
    qc = re.quant_method.moe_quant_config
    r._w4a8 = dict(
        w13=w13, w2=w2, s13=ws13, s2=ws2, real=real, gain=combined_gain,
        w264=torch.cat([block.gate.weight.data, block.shared_expert_gate.weight.data,
                       torch.zeros(7, h, dtype=torch.bfloat16, device=gu.device)], 0),
        intermediate=i, tune_max=fi_moe_largest_bucket(r.moe_config),
        input_quant=dict(quant_dtype=qc.quant_dtype,
                         per_act_token_quant=qc.per_act_token_quant,
                         block_shape=qc.block_shape,
                         is_scale_swizzled=qc.is_scale_swizzled,
                         mx_alignment=qc.mx_alignment),
    )
    for key, value in (("w13_weight", w13), ("w2_weight", w2),
                       ("w13_weight_scale", ws13), ("w2_weight_scale", ws2)):
        replace_parameter(re, key, value)
    # This block never calls the stock runner/kernel after cutover. Release
    # their FP8 scale/weight references, including the separate dense expert.
    re.quant_method.moe_kernel = None
    re.quant_method.moe_quant_config = None
    r._shared_experts = None
    del su.gate_up_proj._w4a8_bf16, su.down_proj._w4a8_bf16
    block.shared_expert = None
    _STATS["folded"] += 1
    logger.debug("shared_expert_fold: folded %s: w13 %s s13 %s w2 %s",
                 name, tuple(w13.shape), tuple(ws13.shape), tuple(w2.shape))


def _fold_moe(x: torch.Tensor, layer_name) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
    from vllm.model_executor.layers.fused_moe.w4a8_routing import route
    import flashinfer
    from flashinfer import RoutingMethodType

    st = mr.get_layer_from_name(mr._resolve_layer_name(layer_name))._w4a8
    if x.shape[0] == 0:
        return torch.empty_like(x)
    ids, weights = route(F.linear(x, st["w264"]))
    if st["gain"] is not None:
        # Round BF16 routing first, then multiply by the float32 gain.
        weights = weights * st["gain"][ids.long()]
    xq, xs = moe_kernel_quantize_input(x, None, **st["input_quant"])
    common = dict(
        num_experts=260, top_k=9, n_group=None, topk_group=None,
        intermediate_size=st["intermediate"], local_expert_offset=0,
        local_num_experts=260, routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.TopK.value,
        do_finalize=True, activation_type=flashinfer.ActivationType.Swiglu.value,
        tune_max_num_tokens=st["tune_max"],
    )
    if st["real"]:
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        out = trtllm_fp4_block_scale_routed_moe(
            (ids, weights), None, xq,
            xs.view(torch.float8_e4m3fn) if xs.dtype != torch.float8_e4m3fn else xs,
            st["w13"], st["s13"], None, None, None, None,
            st["w2"], st["s2"], None, None, None, None, **common,
        )
    else:
        from flashinfer.fused_moe import (
            Fp8QuantizationType, WeightLayout, trtllm_fp8_block_scale_routed_moe,
        )
        out = trtllm_fp8_block_scale_routed_moe(
            (ids, weights), None, xq, xs, st["w13"], st["s13"], st["w2"], st["s2"],
            use_shuffled_weight=True, weight_layout=WeightLayout.MajorK.value,
            fp8_quantization_type=Fp8QuantizationType.MxFp8, **common,
        )
    return out[0] if isinstance(out, (tuple, list)) else out


def _fold_moe_fake(x: torch.Tensor, layer_name) -> torch.Tensor:
    return torch.empty_like(x)


def register_op():
    global _LIB
    if _LIB is not None:
        return
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
    from vllm.utils.torch_utils import direct_register_custom_op

    _fold_moe.__annotations__["layer_name"] = mr._layer_name_type
    _fold_moe_fake.__annotations__["layer_name"] = mr._layer_name_type
    _LIB = torch.library.Library("w4a8", "FRAGMENT")
    direct_register_custom_op(
        "fold_moe", _fold_moe, mutates_args=[], fake_impl=_fold_moe_fake,
        target_lib=_LIB, dispatch_key="CUDA",
    )


def fold_model(model):
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

    blocks = [b for b in model.modules() if isinstance(b, Qwen3NextSparseMoeBlock)
              and b._w4a8_enabled]
    if not blocks:
        return
    register_op()
    for block in blocks:
        fold_block(block)
    tag = type(model).__name__
    logger.info(
        "shared_expert_fold enabled: folded %d MoE blocks (skipped %d, stashed %d)",
        _STATS["folded"], 0, _STATS["stash"],
    )
    w4fq.summary(tag)
    w4fq.gain_summary(tag)
    if w4fq.REAL:
        logger.warning("W4REAL_SUMMARY %s blocks_real=%d encoded=%d", tag,
                       w4fq.REAL_STATS["blocks_real"], w4fq.REAL_STATS["encoded"])
