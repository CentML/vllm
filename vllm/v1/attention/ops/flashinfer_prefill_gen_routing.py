# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route FP8 trtllm-gen context (chunked-prefill) attention to a faster kernel.

FlashInfer runs its trtllm-gen context kernel persistent with multi-CTA KV
disabled, so a short prefill chunk over a long cached prefix launches few CTAs
and gets no KV parallelism. Eligible launches of
``flashinfer.prefill.trtllm_batch_context_with_kv_cache`` (FP8 e4m3 Q/KV,
BF16 O, head_dim 256, page size 32, causal) can instead go to one of:

``gen``
    The trtllm-gen GENERATION kernels through
    ``flashinfer.decode.trtllm_batch_decode_with_kv_cache`` with varlen q
    (``max_q_len`` + ``cum_seq_lens_q``): grouped Q tile (16 tokens x 8 query
    heads of one KV head), multi-CTA KV split with in-kernel reduction
    (persistent zeroed counter buffer), one launch. The causal mask is
    bottom-right per request (the q tokens are the last q_len positions of the
    KV), which is exactly chunked prefill. The split count comes from
    FlashInfer's host heuristic::

        numCtasPerSeqKv = min(
            ceil(maxKv / 512), floor(sm_count / (ceil(max_q / 16) * H_kv * B))
        )

    so the route hands it a virtual SM count (see :func:`gen_vsm`).
``f107``
    A dedicated SM107 CuTe-DSL prefill kernel. It is not part of this module;
    when it is not available in the build, an ``f107`` route fails and falls
    back to the stock kernel (logged for the first failures).
``stock``
    The trtllm-gen context kernel (FlashInfer default).

Routing uses host-known values only (T = total q tokens, B, max_q_len; never
a device sync). With ``FMHA_GEN=1`` a launch goes to ``gen`` when
``T <= FMHA_GEN_MAX_T`` or ``B > FMHA107_MAX_B``; otherwise to ``f107`` when
``FMHA107_F107=1``, ``T >= FMHA107_MIN_TOKENS``,
``max_kv >= FMHA107_MIN_KV`` and ``FMHA107_MIN_B <= B <= FMHA107_MAX_B``
(and ``T <= FMHA107_MAX_TOKENS``, ``max_kv <= FMHA107_MAX_KV``); else stock.
Unsupported launches, CUDA-graph capture, a too small workspace and any
exception fall back to the stock kernel.

Environment (read at import):

* ``FMHA107=1`` enables the routing (default ``0``: stock path only).
* ``FMHA107_LOG=1`` logs the route of each new (T, B) class.
* ``FMHA107_F107`` (1), ``FMHA107_MIN_TOKENS`` (2048), ``FMHA107_MIN_KV``
  (4096), ``FMHA107_MIN_B`` (1), ``FMHA107_MAX_B`` (1000000),
  ``FMHA107_MAX_TOKENS`` (1e9), ``FMHA107_MAX_KV`` (1e9).
* ``FMHA_GEN=1`` enables the ``gen`` route (default 0), ``FMHA_GEN_MAX_T``
  (default: all), ``FMHA_GEN_TARGET`` (768), ``FMHA_GEN_MAX_S`` (8),
  ``FMHA_GEN_MAX_VSM`` (default: 8x the device SM count; bounds the
  multi-CTA scratch in the trtllm workspace), ``FMHA_GEN_REAL_B1_MAXQ`` (512),
  ``FMHA_GEN_PDL`` (0).
* ``FMHA_GEN_RULE`` selects which eligible launches take the ``gen`` route:
  ``rubin`` (default; every eligible launch, the behaviour measured on SM107)
  or ``gb300`` (selective; measured on SM103 where the arch-specific context
  kernel wins on large chunks). With ``gb300`` the default
  ``FMHA_GEN_TARGET`` is 450 and a launch goes to ``gen`` only if

  - one request with ``max_q < FMHA_GEN_B1_STOCK_Q`` (800), or
  - several requests with ``T <= FMHA_GEN_MULTI_MAX_T`` (1280), or
  - several requests whose mean query length ``T / B`` is at most
    ``FMHA_GEN_MULTI_MAX_MEANQ`` (96), i.e. many tiny chunks;

  every other launch keeps the stock context kernel.

Scratch bound: the multi-CTA partial O / stats of the generation kernels live
in the trtllm workspace and are sized by the passed ``sm_count``; FlashInfer
does not bound check them. ``FMHA_GEN_MAX_VSM`` caps the virtual SM count and
every ``gen`` launch checks the workspace (and counter buffer) size on the
host, falling back to the stock kernel if it is too small.

FlashInfer adapter: ``trtllm_batch_decode_with_kv_cache`` has no split /
``sm_count`` argument, so for the duration of a ``gen`` launch
``flashinfer.decode.get_device_sm_count`` is replaced by a function returning
the virtual SM count and restored afterwards. The proper home of this knob is
an explicit argument in FlashInfer.
"""

import os
from typing import Any

import flashinfer.decode as fid
import torch
from flashinfer.prefill import (
    trtllm_batch_context_with_kv_cache as _stock_trtllm_batch_context_with_kv_cache,
)

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("FMHA107", "0") == "1"
_F107 = os.environ.get("FMHA107_F107", "1") == "1"
_MIN_TOKENS = int(os.environ.get("FMHA107_MIN_TOKENS", "2048"))
_MIN_KV = int(os.environ.get("FMHA107_MIN_KV", "4096"))
_LOG = os.environ.get("FMHA107_LOG", "0") == "1"
_MIN_B = int(os.environ.get("FMHA107_MIN_B", "1"))
_MAX_B = int(os.environ.get("FMHA107_MAX_B", "1000000"))
_MAX_T = int(os.environ.get("FMHA107_MAX_TOKENS", "1000000000"))
_MAX_KV = int(os.environ.get("FMHA107_MAX_KV", "1000000000"))
_GEN = os.environ.get("FMHA_GEN", "0") == "1"
_GEN_MAX_T = int(os.environ.get("FMHA_GEN_MAX_T", "1000000000"))
_GEN_RULE = os.environ.get("FMHA_GEN_RULE", "rubin").strip().lower()
if _GEN_RULE not in ("rubin", "gb300"):
    raise ValueError(f"FMHA_GEN_RULE must be 'rubin' or 'gb300', got {_GEN_RULE!r}")
_GEN_TARGET = float(
    os.environ.get("FMHA_GEN_TARGET", "450" if _GEN_RULE == "gb300" else "768")
)
_GEN_B1_STOCK_Q = int(os.environ.get("FMHA_GEN_B1_STOCK_Q", "800"))
_GEN_MULTI_MAX_T = int(os.environ.get("FMHA_GEN_MULTI_MAX_T", "1280"))
_GEN_MULTI_MAX_MEANQ = int(os.environ.get("FMHA_GEN_MULTI_MAX_MEANQ", "96"))
_GEN_MAX_S = int(os.environ.get("FMHA_GEN_MAX_S", "8"))
# None: 8x the device SM count, resolved on the first ``gen`` launch.
_GEN_MAX_VSM: int | None = (
    int(os.environ["FMHA_GEN_MAX_VSM"]) if "FMHA_GEN_MAX_VSM" in os.environ else None
)
_GEN_REAL_B1_MAXQ = int(os.environ.get("FMHA_GEN_REAL_B1_MAXQ", "512"))
_GEN_PDL = os.environ.get("FMHA_GEN_PDL", "0") == "1"

_state: dict[str, Any] = {
    "fwd": None,
    "logged": set(),
    "fail": 0,
    "calls": 0,
    "routed": 0,
    "gen": 0,
    "gen_fn": None,
    "counter": None,
    "sms": None,
    "max_vsm": None,
}

if ENABLED:
    logger.info(
        "trtllm-gen prefill routing enabled (gen=%s rule=%s gen_max_t=%s "
        "gen_target=%s max_vsm=%s, f107=%s min_tokens=%s max_b=%s)",
        _GEN,
        _GEN_RULE,
        _GEN_MAX_T,
        _GEN_TARGET,
        _GEN_MAX_VSM if _GEN_MAX_VSM is not None else "8x SMs",
        _F107,
        _MIN_TOKENS,
        _MAX_B,
    )


def _get_fwd():
    """Forward entry point of the dedicated SM107 prefill kernel."""
    if _state["fwd"] is None:
        raise ImportError("the SM107 prefill kernel is not available in this build")
    return _state["fwd"]


def _supported(
    query: torch.Tensor,
    kv_cache: Any,
    block_tables: torch.Tensor,
    bmm1_scale: Any,
    bmm2_scale: Any,
    out: Any,
    kw: dict[str, Any],
) -> bool:
    if not ENABLED:
        return False
    if kw.get("window_left", -1) not in (-1, None) or kw.get("sinks") is not None:
        return False
    if (
        kw.get("o_sf_scale") is not None
        or kw.get("kv_cache_sf") is not None
        or kw.get("causal", True) is not True
    ):
        return False
    if kw.get("lse") is not None or kw.get("return_lse", False):
        return False
    if out is None or not isinstance(out, torch.Tensor) or out.dtype != torch.bfloat16:
        return False
    if not isinstance(kv_cache, (tuple, list)) or len(kv_cache) != 2:
        return False
    k, v = kv_cache
    if (
        query.dtype != torch.float8_e4m3fn
        or k.dtype != torch.float8_e4m3fn
        or v.dtype != torch.float8_e4m3fn
    ):
        return False
    if (
        query.dim() != 3
        or query.shape[-1] != 256
        or k.dim() != 4
        or k.shape[2] != 32
        or k.shape[3] != 256
    ):
        return False
    if (
        out.shape != query.shape
        or k.shape != v.shape
        or query.shape[1] % k.shape[1] != 0
    ):
        return False
    if k.stride(-1) != 1 or v.stride(-1) != 1 or block_tables.dtype != torch.int32:
        return False
    if (
        not isinstance(bmm1_scale, float)
        or not isinstance(bmm2_scale, (float, int))
        or float(bmm2_scale) != 1.0
    ):
        return False
    return not torch.cuda.is_current_stream_capturing()


def _max_vsm(real_sms: int) -> int:
    """Cap on the virtual SM count (``FMHA_GEN_MAX_VSM``, default: 8x the
    device SM count); bounds the multi-CTA scratch in the trtllm workspace.
    """
    return _GEN_MAX_VSM if _GEN_MAX_VSM is not None else 8 * real_sms


def gen_vsm(
    real_sms: int, T: int, B: int, max_q: int, hkv: int, max_vsm: int | None = None
) -> int:
    """Virtual SM count for the FlashInfer generation-kernel split heuristic
    numCtasPerSeqKv = min(ceil(maxKv/512),
                          floor(sm_count / (ceil(max_q/16) * H_kv * B))).

    Targets about ``FMHA_GEN_TARGET`` useful CTAs: ``S = round(TARGET /
    (2 * (ceil(T/16) + B/2)))`` KV splits, at most ``FMHA_GEN_MAX_S`` and with
    the virtual SM count capped at ``max_vsm`` (``FMHA_GEN_MAX_VSM``, default:
    8x the device SM count; bounds the multi-CTA scratch in the trtllm
    workspace). A lone request with
    ``max_q <= FMHA_GEN_REAL_B1_MAXQ``, or ``S <= 1``, keeps the real count.
    """
    if B == 1 and max_q <= _GEN_REAL_B1_MAXQ:
        return real_sms
    useful = 2 * (-(-T // 16) + B // 2)
    S = max(1, int(_GEN_TARGET / useful + 0.5))
    base = (-(-max_q // 16)) * hkv * B
    if max_vsm is None:
        max_vsm = _max_vsm(real_sms)
    S = min(S, _GEN_MAX_S, max(1, max_vsm // base))
    return real_sms if S <= 1 else max(real_sms, S * base)


def _gen(
    query: torch.Tensor,
    kv_cache: Any,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: float,
    batch_size: int,
    cum_seq_lens_q: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    if _state["gen_fn"] is None:
        # The stock FlashInfer entry point; the virtual SM count is set here,
        # not by the decode split-KV policy.
        _state["gen_fn"] = fid.trtllm_batch_decode_with_kv_cache
        _state["sms"] = torch.cuda.get_device_properties(
            query.device
        ).multi_processor_count
        _state["max_vsm"] = _max_vsm(_state["sms"])
        # Zero-initialised once; the kernel resets its semaphores at the end
        # of every launch.
        _state["counter"] = torch.zeros(1 << 20, dtype=torch.uint8, device=query.device)
    B = int(batch_size)
    mq = int(max_q_len)
    hkv = kv_cache[0].shape[1]
    vsm = gen_vsm(_state["sms"], int(query.shape[0]), B, mq, hkv, _state["max_vsm"])
    # Multi-CTA partial O + stats scratch (FlashInfer has no check).
    need = max(vsm, _state["sms"]) * 128 * (8 + 256 * 4) + (2 << 20)
    if workspace_buffer.numel() * workspace_buffer.element_size() < need or max(
        B * query.shape[1], vsm
    ) * 4 > (1 << 20):
        _state["ws_skip"] = _state.get("ws_skip", 0) + 1
        return False
    # FlashInfer adapter: override the SM count its host heuristic reads for
    # the duration of this launch only.
    real = fid.get_device_sm_count
    fid.get_device_sm_count = lambda d: vsm
    try:
        _state["gen_fn"](
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=int(max_kv_len),
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
            out=out,
            kv_layout="HND",
            backend="trtllm-gen",
            enable_pdl=_GEN_PDL,
            q_len_per_req=None,
            max_q_len=mq,
            cum_seq_lens_q=cum_seq_lens_q,
            multi_ctas_kv_counter_buffer=_state["counter"],
        )
    finally:
        fid.get_device_sm_count = real
    return True


def gen_rule_accepts(T: int, B: int, max_q: int) -> bool:
    """Whether ``FMHA_GEN_RULE`` sends an eligible launch to ``gen``.

    ``rubin``: always. ``gb300`` (SM103 microbenchmarks, FP8 hd256 P32): the
    generation kernels win on under-filled launches (one short chunk over a
    long prefix: 1.5-12x; a few short requests: 2-5x; many tiny chunks behind
    one long chunk: 1.5x), while the SM103 context kernel wins on large chunks
    (single request >= ~800 new tokens: 2-30%, balanced multi-request batches
    with longer chunks: up to 32%).
    """
    if _GEN_RULE != "gb300":
        return True
    if B <= 1:
        return max_q < _GEN_B1_STOCK_Q
    return T <= _GEN_MULTI_MAX_T or T <= _GEN_MULTI_MAX_MEANQ * B


def trtllm_batch_context_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Any,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Any,
    bmm2_scale: Any,
    batch_size: int,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    *args: Any,
    **kw: Any,
) -> Any:
    """``flashinfer.prefill.trtllm_batch_context_with_kv_cache`` with the
    routing described in the module docstring. Same signature.
    """
    out = kw.get("out")
    _state["calls"] += 1
    sup = (not args) and _supported(
        query, kv_cache, block_tables, bmm1_scale, bmm2_scale, out, kw
    )
    T, B = int(query.shape[0]), int(batch_size)
    route = "stock"
    if sup:
        if _GEN and (T <= _GEN_MAX_T or B > _MAX_B):
            if gen_rule_accepts(T, B, int(max_q_len)):
                route = "gen"
        elif (
            _F107
            and T >= _MIN_TOKENS
            and int(max_kv_len) >= _MIN_KV
            and _MIN_B <= B <= _MAX_B
            and T <= _MAX_T
            and int(max_kv_len) <= _MAX_KV
        ):
            route = "f107"
    if _LOG:
        key = (T // 128 if T < 1024 else 8 + T // 1024, min(B, 16), route)
        if key not in _state["logged"] and len(_state["logged"]) < 200:
            _state["logged"].add(key)
            logger.info(
                "trtllm-gen prefill routing: T=%s B=%s max_q=%s max_kv=%s -> %s",
                T,
                B,
                max_q_len,
                max_kv_len,
                route,
            )
    try:
        if route == "gen":
            if _gen(
                query,
                kv_cache,
                workspace_buffer,
                block_tables,
                seq_lens,
                max_q_len,
                max_kv_len,
                bmm1_scale,
                batch_size,
                cum_seq_lens_q,
                out,
            ):
                _state["gen"] += 1
                return out
        elif route == "f107":
            k, v = kv_cache
            _get_fwd()(
                query,
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                cu_seqlens_q=cum_seq_lens_q,
                seqused_k=seq_lens,
                max_seqlen_q=int(max_q_len),
                max_seqlen_k=int(block_tables.shape[1]) * 32,
                page_table=block_tables,
                softmax_scale=float(bmm1_scale),
                causal=True,
                out=out,
            )
            _state["routed"] += 1
            return out
    except Exception as e:  # never break serving: fall back to the stock kernel
        _state["fail"] += 1
        if _state["fail"] <= 3:
            logger.warning(
                "trtllm-gen prefill routing: %s failed, falling back to stock: %r",
                route,
                e,
            )
    return _stock_trtllm_batch_context_with_kv_cache(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        batch_size,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        *args,
        **kw,
    )
