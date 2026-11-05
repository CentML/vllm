# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local routed-expert halves (``VLLM_MOE_DIE_SPLIT=1``).

On a GPU with two memory locality domains an ordinary allocation
interleaves both dies' HBM, so about half of every routed-expert weight byte
crosses the die-to-die link. With this switch the expert weights and block
scales of each TRT-LLM-Gen MXFP8 MoE layer (``TrtLlmFp8ExpertsMonolithic``, the
target model's and the MTP draft's) are re-homed after loading: experts
``[0, E/2)`` on locality domain 0 and ``[E/2, E)`` on domain 1, one VA range per
tensor, so the full tensor is still one contiguous view. Every call of such a
layer with tokens (decode, prefill / mixed and draft; the single-launch decode
kernel of ``moe_decode_cuda`` is bypassed) then runs FlashInfer's stock
``trtllm_fp8_block_scale_moe`` once per die, on that die's green-context stream
(the SM groups of ``vllm/v1/worker/die_local_kv.py``), over its local half
(``local_expert_offset = d * E/2``, ``local_num_experts = E/2``), stopping after
GEMM2. Each (token, k) route lands in exactly one die's permuted rows (the other
die's map holds -1) and both dies compute the same routing weights, so the
two-buffer top-k reduction (``vllm::moe_finalize2``, also folded into the next
norm) adds the same rows with the same weights in the same k order: the output
equals the whole-GPU call's bit for bit wherever each die's GEMMs compute a row
as the whole-GPU tactic does.

Per-die tactics: a per-die call has the whole-GPU call's FlashInfer autotuner
key but another runner (half the experts), so the pinned table's validation
rejects its entries (runner attributes) and FlashInfer would tune in process.
Each per-die call therefore takes its tactic here: from
``VLLM_MOE_DIE_SPLIT_TACTICS``, else the
pinned whole-GPU tactic of the call's bucket (``VLLM_FLASHINFER_AUTOTUNE_FILE``),
else FlashInfer's own choice. A tactic the per-die runner does not offer at
that token count is never used.

FlashInfer sizes its cooperative routing grid (stock routing above 1024 tokens
with ``GS2_ROUTE``) as device SMs minus
``FLASHINFER_TRTLLM_MOE_OVERLAP_RESERVED_SMS``, read once per process; unset, it
is set here so that the grid fits the smaller die group.

Placement scheme as die-local KV (FlashInfer #6791); vLLM #61685 localizes the
CuTe DSL NVFP4 MoE per die along N, this splits TRT-LLM-Gen experts per die.
"""

import ast
import ctypes
import json
import os
import threading
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_MOE_DIE_SPLIT"
ENV_TACTICS = "VLLM_MOE_DIE_SPLIT_TACTICS"
# Check this many eager calls: routing weights equal on both dies, every route
# on exactly one die (synchronizes; diagnostics only).
ENV_CHECK = "VLLM_MOE_DIE_SPLIT_CHECK"
ENV_RESERVED = "FLASHINFER_TRTLLM_MOE_OVERLAP_RESERVED_SMS"
_OP = "flashinfer::trtllm_fp8_block_scale_moe"
# CUpointer_attribute of a mapping's locality domain (CUDA 13.4).
_ATTR_POINTER_LOCALITY_DOMAIN = 22
# Per-die tactics, "lo-hi:tile.config,..." over the call's token count; outside
# them (and by default) the pinned whole-GPU tactic, else FlashInfer chooses.
_TACTICS_DEFAULT = ""


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


@dataclass
class DieSplit:
    """A layer's die-local halves and the dies' green-context streams."""

    # Per die: (w13, w13 scales, w2, w2 scales) of its num_local experts.
    halves: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]
    num_local: int
    streams: tuple[Any, Any]


@dataclass
class _Ctx:
    cu: ctypes.CDLL
    cudev: int
    gran: int
    green: Any


class _Owner:
    """A mapped VA range for ``torch.as_tensor``; kept for the process."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


_OWNERS: list[tuple[_Owner, list]] = []
_TLS = threading.local()
_TABLE: list[tuple[int, int, list[int]]] = []
_PINNED: dict[int, list[int]] = {}
_OFFERED: dict[tuple[int, tuple], bool] = {}
_LOGGED: set[tuple] = set()
_CHECKS = [0]


def _context(device: torch.device) -> _Ctx | None:
    from vllm.v1.worker import die_local_kv as dlk

    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        cu = dlk._libcuda()
        cudev, count = dlk._int(), dlk._int()
        dlk._check(cu.cuDeviceGet(ctypes.byref(cudev), index), "cuDeviceGet")
        dlk._check(
            cu.cuDeviceGetAttribute(
                ctypes.byref(count), dlk._ATTR_LOCALITY_DOMAIN_COUNT, cudev
            ),
            "locality domain count",
        )
    except (OSError, AttributeError, RuntimeError) as e:
        logger.warning("moe-die-split off: no locality-domain API (%s)", e)
        return None
    if count.value != 2:
        logger.warning("moe-die-split off: %d locality domain(s)", count.value)
        return None
    try:
        gran = dlk._granularity(cu, cudev.value, 0)
        if dlk._GREEN is None:
            dlk._GREEN = dlk._select_green(
                cu, cudev.value, torch.device("cuda", index)
            )
    except RuntimeError as e:
        logger.warning("moe-die-split off: %s", e)
        return None
    fn = cu.cuPointerGetAttribute
    fn.restype, fn.argtypes = dlk._int, [dlk._vp, dlk._int, dlk._u64]
    return _Ctx(cu, cudev.value, gran, dlk._GREEN)


def _domain(ctx: _Ctx, ptr: int) -> int:
    v = ctypes.c_int(-1)
    ret = ctx.cu.cuPointerGetAttribute(
        ctypes.byref(v), _ATTR_POINTER_LOCALITY_DOMAIN, ptr
    )
    return v.value if ret == 0 else -100 - ret


def _home_halves(ctx: _Ctx, t: torch.Tensor) -> torch.Tensor:
    """A copy of ``t`` whose first half (bytes) is on domain 0, second on 1."""
    from vllm.v1.worker import die_local_kv as dlk

    nbytes = t.numel() * t.element_size()
    half = nbytes // 2
    n = half // ctx.gran
    va, total, chunks = dlk._map(ctx.cu, ctx.cudev, [0] * n + [1] * n, ctx.gran)
    homes = [
        _domain(ctx, p) for p in (va, va + half - 1, va + half, va + total - 1)
    ]
    if total != nbytes or homes != [0, 0, 1, 1]:
        dlk._free(ctx.cu, va, total, chunks)
        raise RuntimeError(f"halves on domains {homes} ({total} of {nbytes} B)")
    owner = _Owner(va, total)
    _OWNERS.append((owner, chunks))
    flat = torch.as_tensor(owner, device=t.device)
    new = flat.view(t.dtype).view(t.shape)
    new.copy_(t)
    return new


def _unsupported(module: torch.nn.Module, experts: Any, gran: int) -> str | None:
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

    qc, cfg = experts.quant_config, experts.moe_config
    if qc.block_shape != [1, 32]:
        return "not MXFP8"
    if experts.routing_method_type not in (
        RoutingMethodType.Renormalize,
        RoutingMethodType.RenormalizeNaive,
    ):
        return f"routing {experts.routing_method_type}"
    num_experts = cfg.num_experts
    if experts.local_num_experts != num_experts or num_experts % 2:
        return "expert parallel or an odd expert count"
    if (cfg.tp_size, cfg.dp_size, cfg.ep_size, cfg.pcp_size) != (1, 1, 1, 1):
        return "TP/DP/EP/PCP"
    if getattr(module, "expert_map", None) is not None:
        return "expert map"
    if any(
        p is not None
        for p in (experts.gemm1_alpha, experts.gemm1_beta, experts.gemm1_clamp_limit)
    ):
        return "SwiGLU parameters"
    own = {id(p) for p in module.parameters(recurse=False)}
    for t in (module.w13_weight, qc.w1_scale, module.w2_weight, qc.w2_scale):
        if not isinstance(t, torch.nn.Parameter) or id(t) not in own:
            return "weights or scales are not parameters of the layer"
        if not t.is_cuda or not t.is_contiguous() or t.shape[0] != num_experts:
            return "weight layout"
        if (t.numel() * t.element_size()) % (2 * gran):
            return "halves are not whole granules"
    return None


def localize_model(model: torch.nn.Module, device: torch.device) -> None:
    """Re-home the expert halves of every supported MoE layer of ``model``."""
    if not enabled():
        return
    from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
        TrtLlmFp8ExpertsMonolithic,
    )

    layers, seen = [], set()
    for name, module in model.named_modules():
        kernel = getattr(getattr(module, "quant_method", None), "moe_kernel", None)
        experts = getattr(kernel, "fused_experts", None)
        if (
            isinstance(experts, TrtLlmFp8ExpertsMonolithic)
            and id(experts) not in seen
            and hasattr(module, "w13_weight")
        ):
            seen.add(id(experts))
            layers.append((name, module, experts))
    if not layers:
        logger.info("moe-die-split: no TRT-LLM-Gen FP8 MoE layer in %s",
                    type(model).__name__)
        return
    device = torch.device(device)
    ctx = _context(device)
    if ctx is None:
        return
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free0 = torch.cuda.mem_get_info(device)[0]
    done, moved, skipped = 0, 0, {}
    for name, module, experts in layers:
        if getattr(experts, "die_split", None) is not None:
            continue
        why = _unsupported(module, experts, ctx.gran)
        if why is None:
            qc = experts.quant_config
            params = (module.w13_weight, qc.w1_scale, module.w2_weight, qc.w2_scale)
            try:
                homed = [_home_halves(ctx, p.data) for p in params]
            except RuntimeError as e:
                why = str(e)
            else:
                for p, new in zip(params, homed):
                    p.data = new
                    moved += new.numel() * new.element_size()
                h = experts.moe_config.num_experts // 2
                experts.die_split_active = True  # read by the runner's trace
                experts.die_split = DieSplit(
                    halves=tuple(  # type: ignore[arg-type]
                        tuple(p.data[d * h : (d + 1) * h] for p in params)
                        for d in (0, 1)
                    ),
                    num_local=h,
                    streams=ctx.green.streams,
                )
                done += 1
        if why is not None:
            skipped[why] = skipped.get(why, 0) + 1
            logger.warning("moe-die-split: %s stays whole-GPU: %s", name, why)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    free1 = torch.cuda.mem_get_info(device)[0]
    if done:
        _install_tactics()
        _set_reserved(device, ctx.green.sms)
    logger.info(
        "moe-die-split: %d of %d MoE layers of %s on die-local expert halves "
        "(%.2f GiB of expert weights and scales; experts [0, E/2) on locality "
        "domain 0, [E/2, E) on domain 1, placement read back); device free "
        "memory %+.0f MiB; per-die calls on green-context SMs %d+%d (%s)%s",
        done,
        len(layers),
        type(model).__name__,
        moved / 2**30,
        (free1 - free0) / 2**20,
        *ctx.green.sms,
        ctx.green.name,
        f"; skipped: {skipped}" if skipped else "",
    )


# ---------------------------------------------------------------- tactics
def _parse_table(raw: str) -> list[tuple[int, int, list[int]]]:
    raw = raw.strip()
    if raw in ("", "stock"):
        return []
    out = []
    try:
        for item in raw.split(","):
            span, tac = item.split(":")
            lo, hi = (int(x) for x in span.split("-"))
            tile, config = (int(x) for x in tac.split("."))
            if lo > hi:
                raise ValueError(item)
            out.append((lo, hi, [tile, config]))
    except ValueError:
        logger.warning(
            "moe-die-split: %s=%r is not lo-hi:tile.config,...; per-die calls "
            "take the pinned whole-GPU tactics",
            ENV_TACTICS,
            raw,
        )
        return []
    return out


def _load_pinned() -> dict[int, list[int]]:
    """Bucket -> tactic of the pinned file's trtllm_fp8_block_scale_moe keys."""
    path = os.environ.get("VLLM_FLASHINFER_AUTOTUNE_FILE")
    if not path:
        return {}
    try:
        with open(os.path.expanduser(path)) as f:
            table = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("moe-die-split: cannot read %s (%s)", path, e)
        return {}
    out: dict[int, list[int]] = {}
    for key, entry in table.items():
        if not key.startswith("(") or _OP not in key:
            continue
        try:
            bucket = int(ast.literal_eval(key)[2][0][0])
            tactic = [int(x) for x in entry[1]]
        except (ValueError, SyntaxError, TypeError, IndexError):
            continue
        out.setdefault(bucket, tactic)
    return out


def _install_tactics() -> None:
    """Route per-die calls' tactic choice through ``_per_die_tactic``.

    Patched on the class: the pinned-autotune warmup wraps the bound class
    method on the instance and removes its wrapper afterwards.
    """
    global _TABLE, _PINNED
    try:
        from flashinfer.autotuner import AutoTuner
    except ImportError:
        from flashinfer.autotuner.autotuner import AutoTuner

    if getattr(AutoTuner, "_moe_die_split", False):
        return
    _TABLE = _parse_table(os.environ.get(ENV_TACTICS, _TACTICS_DEFAULT))
    _PINNED = _load_pinned()
    orig = AutoTuner.choose_one

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
        if custom_op == _OP and getattr(_TLS, "die", None) is not None:
            tactic = _per_die_tactic(runners[0], tuning_config, inputs)
            if tactic is not None:
                return runners[0], tactic
        return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)

    AutoTuner.choose_one = choose_one
    AutoTuner._moe_die_split = True
    logger.info(
        "moe-die-split: per-die tactics: %d table ranges (%s), %d pinned "
        "whole-GPU buckets",
        len(_TABLE),
        os.environ.get(ENV_TACTICS, "default") or "none",
        len(_PINNED),
    )


def _per_die_tactic(runner: Any, tuning_config: Any, inputs: list) -> Any:
    num_tokens = _TLS.num_tokens
    tactic, src = None, "table"
    for lo, hi, tac in _TABLE:
        if lo <= num_tokens <= hi:
            tactic, span = tac, f"{lo}-{hi}"
            break
    if tactic is None:
        bucket = tuning_config.dynamic_tensor_specs[0].map_to_tuning_buckets(
            num_tokens
        )
        tactic, src, span = _PINNED.get(bucket), "pinned", f"bucket {bucket}"
    if tactic is None:
        if ("none", num_tokens) not in _LOGGED and len(_LOGGED) < 256:
            _LOGGED.add(("none", num_tokens))
            logger.info(
                "moe-die-split: no per-die tactic for %d tokens; FlashInfer "
                "chooses",
                num_tokens,
            )
        return None
    key = (num_tokens, tuple(tactic))
    offered = _OFFERED.get(key)
    if offered is None:
        valid = runner.get_valid_tactics(inputs, None)
        offered = _OFFERED[key] = tactic in [
            list(t) if isinstance(t, (list, tuple)) else t for t in valid
        ]
    if not offered:
        if ("bad", key) not in _LOGGED and len(_LOGGED) < 256:
            _LOGGED.add(("bad", key))
            logger.warning(
                "moe-die-split: %s tactic %s (%s) not offered per die at %d "
                "tokens; FlashInfer chooses",
                src,
                tactic,
                span,
                num_tokens,
            )
        return None
    if (src, span, tuple(tactic)) not in _LOGGED and len(_LOGGED) < 256:
        _LOGGED.add((src, span, tuple(tactic)))
        logger.info(
            "moe-die-split: per-die tactic %s (%s %s), first at %d tokens",
            tactic,
            src,
            span,
            num_tokens,
        )
    return tactic


def _set_reserved(device: torch.device, sms: tuple[int, int]) -> None:
    total = torch.cuda.get_device_properties(device).multi_processor_count
    need = total - min(sms)
    cur = os.environ.get(ENV_RESERVED)
    if cur is None:
        os.environ[ENV_RESERVED] = str(need)
        logger.info(
            "moe-die-split: %s=%d (cooperative routing grid %d of %d SMs fits "
            "the smaller die group)",
            ENV_RESERVED,
            need,
            total - need,
            total,
        )
    elif not cur.strip().isdigit() or int(cur) < need:
        logger.warning(
            "moe-die-split: %s=%s leaves a cooperative routing grid larger than "
            "a die group of %d SMs (needs >= %d)",
            ENV_RESERVED,
            cur,
            min(sms),
            need,
        )


# ---------------------------------------------------------------- forward
def apply(
    experts: Any,
    split: DieSplit,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    activation: Any,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    a1q_scale: torch.Tensor,
    num_expert_group: int | None,
    e_score_correction_bias: torch.Tensor | None,
    routed_scaling_factor: float | None,
    topk_group: int | None,
) -> Any:
    """``TrtLlmFp8ExpertsMonolithic._apply_block_scale`` (MXFP8) once per die.

    Returns the two-buffer ``SplitUnfinalizedMoEOutput`` when this call may
    defer the finalize, else the finalized [tokens, hidden] output.
    """
    import flashinfer
    from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout

    from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
        _sm107_moe_pdl,
    )
    from vllm.model_executor.layers.fused_moe.moe_output import (
        SplitUnfinalizedMoEOutput,
        convert_flashinfer_moe_output,
    )
    from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )

    assert expert_map is None
    num_tokens = hidden_states.shape[0]
    top_k = experts.topk
    n_group = num_expert_group or None
    replay = experts._maybe_make_routing_replay_buffer(
        num_tokens=num_tokens, device=hidden_states.device
    )
    kwargs = dict(
        routing_logits=router_logits,
        routing_bias=e_score_correction_bias,
        hidden_states=hidden_states,
        hidden_states_scale=a1q_scale,
        gemm1_alpha=experts.gemm1_alpha,
        gemm1_beta=experts.gemm1_beta,
        gemm1_clamp_limit=experts.gemm1_clamp_limit,
        num_experts=global_num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group or None,
        intermediate_size=experts.intermediate_size_per_partition,
        local_num_experts=split.num_local,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=experts.routing_method_type,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.MajorK,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        tune_max_num_tokens=fi_moe_largest_bucket(experts.moe_config),
        activation_type=activation_to_flashinfer_int(activation),
        do_finalize=False,
    )
    main = torch.cuda.current_stream()
    fork = torch.cuda.Event()
    fork.record(main)
    outs, joins = [], []
    for die, stream in enumerate(split.streams):
        w13, w13_scale, w2, w2_scale = split.halves[die]
        stream.wait_event(fork)
        with torch.cuda.stream(stream):
            _TLS.die, _TLS.num_tokens = die, num_tokens
            try:
                with _sm107_moe_pdl(
                    num_tokens,
                    global_num_experts,
                    hidden_states.shape[-1],
                    experts.routing_method_type,
                    n_group,
                    top_k,
                    deferred=True,
                ):
                    result = flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
                        gemm1_weights=w13,
                        gemm1_weights_scale=w13_scale,
                        gemm2_weights=w2,
                        gemm2_weights_scale=w2_scale,
                        local_expert_offset=die * split.num_local,
                        routing_replay_out=replay if die == 0 else None,
                        **kwargs,
                    )
            finally:
                _TLS.die = None
            outs.append(
                convert_flashinfer_moe_output(
                    result, do_finalize=False, num_tokens=num_tokens, top_k=top_k
                )
            )
            join = torch.cuda.Event()
            join.record(stream)
            joins.append(join)
    for join in joins:
        main.wait_event(join)
    experts._maybe_dispatch_routing_replay(replay, num_tokens=num_tokens)
    out0, out1 = outs
    if ("first",) not in _LOGGED:
        _LOGGED.add(("first",))
        logger.info(
            "moe-die-split: routed experts run one call per die (%d experts "
            "each), first call %d tokens",
            split.num_local,
            num_tokens,
        )
    _maybe_check(out0, out1)
    if experts.moe_config.should_defer_moe_finalize(num_tokens):
        return SplitUnfinalizedMoEOutput(
            gemm2_permuted=out0.gemm2_permuted,
            expert_weights=out0.expert_weights,
            expanded_idx_to_permuted_idx=out0.expanded_idx_to_permuted_idx,
            gemm2_permuted1=out1.gemm2_permuted,
            expanded_idx_to_permuted_idx1=out1.expanded_idx_to_permuted_idx,
        )
    rows = num_tokens * top_k
    return torch.ops.vllm.moe_finalize2(
        out0.gemm2_permuted[:rows],
        out0.expert_weights,
        out0.expanded_idx_to_permuted_idx,
        out1.gemm2_permuted[:rows],
        out1.expanded_idx_to_permuted_idx,
    )


def _maybe_check(out0: Any, out1: Any) -> None:
    limit = int(os.environ.get(ENV_CHECK, "0") or "0")
    if _CHECKS[0] >= limit or torch.cuda.is_current_stream_capturing():
        return
    _CHECKS[0] += 1
    i0 = out0.expanded_idx_to_permuted_idx
    i1 = out1.expanded_idx_to_permuted_idx
    once = bool((((i0 >= 0).int() + (i1 >= 0).int()) == 1).all())
    same = bool(torch.equal(out0.expert_weights, out1.expert_weights))
    logger.info(
        "moe-die-split check %d/%d: %d tokens, routes on die 0/1 %d/%d, every "
        "route on exactly one die %s, routing weights equal %s",
        _CHECKS[0],
        limit,
        i0.shape[0],
        int((i0 >= 0).sum()),
        int((i1 >= 0).sum()),
        once,
        same,
    )
    if not (once and same):
        raise RuntimeError("moe-die-split: per-die routing disagrees")
