# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fold the sigmoid-gated shared expert of Qwen3-Next / Qwen3.5 MoE blocks
into the trtllm-gen MXFP8 routed MoE as routed expert #256.

The shared expert has the routed-expert shape and its output is scaled by
sigmoid(shared_expert_gate(x)). It becomes expert 256 of 260 (plus 3
never-routed zero pad experts: the trtllm routing requires
num_experts % 4 == 0) and top-k goes from 8 to 9.

Per MoE block, after all weights are loaded and online-quantized
(``fold_model``, called by the model loader after
``process_weights_after_loading``):
  * router weight W264 = [gate (256 rows); shared_expert_gate (1 row);
    0 (7 rows)] (bf16). N=264 keeps the aligned cuBLAS path, so the 256
    router logits are bit-identical to the stock N=256 GEMM (N=257 falls on
    a much slower cuBLAS kernel).
  * the shared expert's BF16 gate_up/down weights (stashed right before
    their own online MXFP8 quantization) are quantized exactly like the
    routed experts (mxfp8_e4m3_quantize, linear scales), pushed through the
    same trtllm weight transform (prepare_fp8_moe_layer_for_fi: w13->w31
    swap, gated row interleave, shuffle, scale interleave) and appended.
Runtime (opaque custom op ``torch.ops.seg.fold_moe``, CUDA-graph safe):
  logits = x @ W264^T -> seg_route_fold (Triton: top-8 of 256, renormalized
  softmax, plus (256, sigmoid(logit 256))) -> the stock MoE input quant
  (moe_kernel_quantize_input) -> FlashInfer
  trtllm_fp8_block_scale_routed_moe(top_k=9, num_experts=260), whose
  finalize sums the routed and the gated shared expert outputs.
When the runner is marked ``_nqf_defer`` (a fused residual-add + norm
consumer takes the (shared, routed) pair instead of their sum), the folded
forward returns (moe_out, zeros[:M]); adding 0 is exact. With QGF_FIN=1 it
uses ``torch.ops.seg.fold_moe_nf`` instead: the trtllm MoE skips its finalize
kernel (do_finalize=False) and returns a placeholder [M, H] tensor; the
unfinalized outputs are registered under its address and the next fused
pre-norm does the finalize gather-reduce inside its kernel (bit-identical).

Eligible blocks: TP = DP = EP = 1, no sequence parallelism, no routed
input/output transforms, routed scaling factor 1.0, no EPLB,
Mxfp8OnlineMoEMethod with the trtllm backend, 256 experts, top-8, bf16 gate
weights. Every other block keeps the stock path.

Env: SEG_FOLD=1 enables the fold (default off).

Numerics: not bit-exact vs the unfolded path. The shared expert runs in the
routed MoE GEMMs and its output is summed with the routed output in fp32
inside the trtllm finalize (stock: separate dense GEMMs, sigmoid * out and a
bf16 add), so the summation order changes.
"""

import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.shared_expert_kernels import (
    seg_route_fold,
)
from vllm.model_executor.layers.fusion import norm_quant
from vllm.model_executor.layers import lcd2_bf16 as _lcd2

logger = init_logger(__name__)

SEG_FOLD = os.environ.get("SEG_FOLD", "0") == "1"

E_ROUTED, TOPK, E_PAD = 256, 8, 260
STATS = {"stash": 0, "folded": 0, "skipped": 0, "calls": 0}
_ZEROS: dict = {}


# ------------------------------------------ 1. stash shared-expert BF16 weights
def stash_shared_expert_weight(layer: torch.nn.Module) -> None:
    """Keep a BF16 copy of a shared-expert gate_up/down weight right before
    its own online MXFP8 quantization (called from
    Mxfp8OnlineLinearMethod.process_weights_after_loading when SEG_FOLD=1).
    fold_block quantizes the copy like the routed experts and drops it."""
    p = getattr(layer, "prefix", "") or ""
    if (
        ".shared_expert." in p
        and (p.endswith("gate_up_proj") or p.endswith("down_proj"))
        and not getattr(layer, "_already_called_process_weights_after_loading", False)
    ):
        w = layer.weight
        if w.dtype in (torch.bfloat16, torch.float16) and w.device.type == "cuda":
            layer._seg_bf16 = w.detach().clone()
            STATS["stash"] += 1


# ------------------------------------------------------------- 2. custom op
def _fold_moe(x: torch.Tensor, layer_name) -> torch.Tensor:
    return _fold_moe_impl(x, layer_name, False)


def _fold_moe_nf(x: torch.Tensor, layer_name) -> torch.Tensor:
    """Same MoE, but trtllm skips its finalize kernel (do_finalize=False).
    Returns a placeholder [M, H] tensor; (gemm2_out, weights,
    expanded_idx_to_permuted_idx) are registered under its address and the
    next fused pre-norm does the finalize gather-reduce (bit-identical)."""
    return _fold_moe_impl(x, layer_name, True)


def _fold_moe_impl(x: torch.Tensor, layer_name, no_finalize: bool) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    L = mr.get_layer_from_name(mr._resolve_layer_name(layer_name))
    st = L._seg_fold
    M = x.shape[0]
    if M == 0:
        return torch.empty_like(x)
    STATS["calls"] += 1
    if _lcd2.RTR:
        # LCD2_BF16=tiny: TinyGEMM2 router GEMM for decode-size M ([M, 272]
        # logits, read through their row stride); F.linear above LCD2_MAXM
        logits = _lcd2.router_logits(x, st)
    else:
        logits = F.linear(x, st["w264"])
    ids, wts = seg_route_fold(logits, E_ROUTED, TOPK)
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    qc = st["qc"]
    xq, xs = moe_kernel_quantize_input(
        x,
        None,
        quant_dtype=qc.quant_dtype,
        per_act_token_quant=qc.per_act_token_quant,
        block_shape=qc.block_shape,
        is_scale_swizzled=qc.is_scale_swizzled,
        mx_alignment=qc.mx_alignment,
    )
    import flashinfer
    from flashinfer import RoutingMethodType
    from flashinfer.fused_moe import (
        Fp8QuantizationType,
        WeightLayout,
        trtllm_fp8_block_scale_routed_moe,
    )

    o = trtllm_fp8_block_scale_routed_moe(
        (ids, wts),
        None,
        xq,
        xs,
        st["w13"],
        st["s13"],
        st["w2"],
        st["s2"],
        num_experts=E_PAD,
        top_k=TOPK + 1,
        n_group=None,
        topk_group=None,
        intermediate_size=st["I"],
        local_expert_offset=0,
        local_num_experts=E_PAD,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.TopK.value,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.MajorK.value,
        tune_max_num_tokens=st["tune_max"],
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=flashinfer.ActivationType.Swiglu.value,
        do_finalize=not no_finalize,
    )
    if not no_finalize:
        return o[0] if isinstance(o, (list, tuple)) else o
    g2, w_used, idx = o[0], o[1], o[2]
    handle = torch.empty_like(x)
    norm_quant.fin_produce(handle, g2, w_used, idx)
    return handle


def _fold_moe_fake(x: torch.Tensor, layer_name) -> torch.Tensor:
    return torch.empty_like(x)


_LIB = None


def register_op() -> None:
    """Register torch.ops.seg.fold_moe (once, on first fold)."""
    global _LIB
    if _LIB is not None:
        return
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
    from vllm.utils.torch_utils import direct_register_custom_op

    # Same layer-name argument type as vllm.moe_forward (str or LayerName).
    _fold_moe.__annotations__["layer_name"] = mr._layer_name_type
    _fold_moe_fake.__annotations__["layer_name"] = mr._layer_name_type
    _LIB = torch.library.Library("seg", "FRAGMENT")
    direct_register_custom_op(
        "fold_moe",
        _fold_moe,
        mutates_args=[],
        fake_impl=_fold_moe_fake,
        target_lib=_LIB,
        dispatch_key="CUDA",
    )
    _fold_moe_nf.__annotations__["layer_name"] = mr._layer_name_type
    direct_register_custom_op(
        "fold_moe_nf",
        _fold_moe_nf,
        mutates_args=[],
        fake_impl=_fold_moe_fake,
        target_lib=_LIB,
        dispatch_key="CUDA",
    )


# ---------------------------------------------------------- 3. fold at load
def _eligible(block, r) -> str | None:
    """None if the block can be folded, else the reason it cannot."""
    try:
        mc = r.moe_config
        if block.shared_expert is None or getattr(
            block, "replicate_shared_expert", False
        ):
            return "no shared expert"
        if (mc.tp_size, mc.dp_size, mc.ep_size) != (1, 1, 1) or mc.is_sequence_parallel:
            return f"parallel tp/dp/ep={mc.tp_size}/{mc.dp_size}/{mc.ep_size}"
        if (
            r.routed_input_transform is not None
            or r.routed_output_transform is not None
        ):
            return "routed transforms"
        if r.routed_scaling_factor != 1.0 or getattr(mc, "enable_eplb", False):
            return "scaling/eplb"
        qm = r.routed_experts.quant_method
        if type(qm).__name__ != "Mxfp8OnlineMoEMethod":
            return f"quant method {type(qm).__name__}"
        experts_cls = getattr(qm, "experts_cls", None)
        if (
            "trtllm" not in str(getattr(qm, "fp8_backend", "")).lower()
            and "TrtLlm" not in type(experts_cls or object).__name__
            and "TrtLlm" not in str(getattr(qm, "experts_cls", ""))
        ):
            return f"backend {getattr(qm, 'fp8_backend', None)}"
        if (
            r.moe_config.num_experts != E_ROUTED
            or r.moe_config.experts_per_token != TOPK
        ):
            return "shape"
        su = block.shared_expert
        if (
            getattr(su.gate_up_proj, "_seg_bf16", None) is None
            or getattr(su.down_proj, "_seg_bf16", None) is None
        ):
            return "no bf16 stash"
        if (
            block.gate.weight.dtype != torch.bfloat16
            or block.shared_expert_gate.weight.dtype != torch.bfloat16
        ):
            return "gate dtype"
    except Exception as e:  # report as not eligible; the stock path stays
        return f"check failed {e!r}"
    return None


@torch.no_grad()
def fold_block(block, name: str, zeros: torch.Tensor) -> None:
    from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        prepare_fp8_moe_layer_for_fi,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    r = block.experts
    re = r.routed_experts
    su = block.shared_expert
    gu = su.gate_up_proj._seg_bf16  # [2I, H] = [gate; up] (vLLM w13 convention)
    dn = su.down_proj._seg_bf16  # [H, I]
    q13, s13 = mxfp8_e4m3_quantize(gu.contiguous(), is_sf_swizzled_layout=False)
    q2, s2 = mxfp8_e4m3_quantize(dn.contiguous(), is_sf_swizzled_layout=False)
    w13n, w2n, s13n, s2n = prepare_fp8_moe_layer_for_fi(
        re, q13[None], q2[None], s13[None], None, s2[None], None, is_trtllm=True
    )
    W13, W2 = re.w13_weight, re.w2_weight
    S13, S2 = re.w13_weight_scale, re.w2_weight_scale
    assert W13.shape[1:] == w13n.shape[1:] and W2.shape[1:] == w2n.shape[1:], (
        W13.shape,
        w13n.shape,
    )
    assert S13.shape[1:] == s13n.shape[1:] and S2.shape[1:] == s2n.shape[1:], (
        S13.shape,
        s13n.shape,
    )
    pad = E_PAD - E_ROUTED - 1

    def cat(big, one):
        z = torch.zeros((pad, *one.shape[1:]), dtype=one.dtype, device=one.device)
        return torch.cat(
            [big.data.view(one.dtype) if big.dtype != one.dtype else big.data, one, z],
            0,
        ).contiguous()

    st = dict(w13=cat(W13, w13n), w2=cat(W2, w2n), s13=cat(S13, s13n), s2=cat(S2, s2n))
    # Free the 256-expert copies (the quant config keeps its own scale refs;
    # the folded forward reads weights from st only).
    from vllm.model_executor.utils import replace_parameter

    replace_parameter(re, "w13_weight", st["w13"])
    replace_parameter(re, "w2_weight", st["w2"])
    wg = block.gate.weight.data
    st["w264"] = torch.cat(
        [
            wg,
            block.shared_expert_gate.weight.data,
            torch.zeros(7, wg.shape[1], dtype=wg.dtype, device=wg.device),
        ],
        0,
    ).contiguous()
    if _lcd2.RTR:
        _lcd2.prep_router(st)
    st["I"] = r.moe_config.intermediate_size_per_partition
    st["qc"] = re.quant_method.moe_quant_config
    st["tune_max"] = fi_moe_largest_bucket(r.moe_config)
    r._seg_fold = st
    r.register_buffer("_seg_zeros", zeros, persistent=False)
    # The folded forward produces the (routed + shared) sum itself; the
    # shared-expert module is no longer run by the runner.
    r._shared_experts = None
    r._seg_fold_on = True
    del su.gate_up_proj._seg_bf16, su.down_proj._seg_bf16
    STATS["folded"] += 1
    logger.debug(
        "shared_expert_fold: folded %s: w13 %s s13 %s w2 %s",
        name,
        tuple(st["w13"].shape),
        tuple(st["s13"].shape),
        tuple(st["w2"].shape),
    )


def fold_model(model: torch.nn.Module) -> None:
    """Fold the shared expert of every eligible Qwen3-Next / Qwen3.5 sparse
    MoE block. Called once after process_weights_after_loading when
    SEG_FOLD=1."""
    try:
        from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
    except Exception:
        return
    blocks = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, Qwen3NextSparseMoeBlock)
    ]
    if not blocks:
        return
    register_op()
    from vllm.config import get_current_vllm_config

    try:
        cfg = get_current_vllm_config()
        mx = max(
            int(cfg.scheduler_config.max_num_batched_tokens),
            max(cfg.compilation_config.cudagraph_capture_sizes or [0]),
        )
    except Exception:
        mx = 16384
    mx = max(mx, 16384) + 1024
    for n, b in blocks:
        why = _eligible(b, b.experts)
        if why:
            STATS["skipped"] += 1
            logger.info("shared_expert_fold: not folding %s: %s", n, why)
            continue
        H = b.gate.weight.shape[1]
        key = (b.gate.weight.device, H)
        if key not in _ZEROS:
            _ZEROS[key] = torch.zeros(
                mx, H, dtype=torch.bfloat16, device=b.gate.weight.device
            )
        fold_block(b, n, _ZEROS[key])
    torch.cuda.empty_cache()
    logger.info(
        "shared_expert_fold enabled: folded %d MoE blocks (skipped %d, stashed %d)",
        STATS["folded"],
        STATS["skipped"],
        STATS["stash"],
    )
