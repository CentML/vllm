# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route FP8 trtllm-gen context (chunked-prefill) attention to a faster kernel.

FlashInfer runs its trtllm-gen context kernel persistent with multi-CTA KV
disabled, so a short prefill chunk over a long cached prefix launches few CTAs
and gets no KV parallelism. Eligible launches of
``flashinfer.prefill.trtllm_batch_context_with_kv_cache`` (FP8 e4m3 Q/KV,
BF16 O, head_dim 256, page size 32, causal) can instead go to:

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
``stock``
    The trtllm-gen context kernel (FlashInfer default).

Routing uses host-known values only (T = total q tokens, B, max_q_len; never
a device sync). With ``FMHA_GEN=1`` a launch goes to ``gen`` when
``T <= FMHA_GEN_MAX_T`` or ``B > FMHA107_MAX_B``, and ``FMHA_GEN_RULE``
accepts it; otherwise to stock. Unsupported launches, CUDA-graph capture, a
too small workspace and any exception fall back to the stock kernel.

Environment (read at import):

* ``FMHA107=1`` enables the routing (default ``0``: stock path only).
* ``FMHA107_LOG=1`` logs the route of each new (T, B) class.
* ``FMHA107_MAX_B`` (1000000).
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
_LOG = os.environ.get("FMHA107_LOG", "0") == "1"
_MAX_B = int(os.environ.get("FMHA107_MAX_B", "1000000"))
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
    "logged": set(),
    "fail": 0,
    "calls": 0,
    "gen": 0,
    "gen_fn": None,
    "counter": None,
    "sms": None,
    "max_vsm": None,
}

if ENABLED:
    logger.info(
        "trtllm-gen prefill routing enabled (gen=%s rule=%s gen_max_t=%s "
        "gen_target=%s max_vsm=%s max_b=%s)",
        _GEN,
        _GEN_RULE,
        _GEN_MAX_T,
        _GEN_TARGET,
        _GEN_MAX_VSM if _GEN_MAX_VSM is not None else "8x SMs",
        _MAX_B,
    )


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
    """Whether the selected rule sends an eligible launch to generation.

    The default rule accepts every eligible launch; the alternative rule
    retains context kernels for larger chunks.
    """
    if _GEN_RULE != "gb300":
        return True
    if B <= 1:
        return max_q < _GEN_B1_STOCK_Q
    return T <= _GEN_MULTI_MAX_T or T <= _GEN_MULTI_MAX_MEANQ * B


# ---------------------------------------------------------------------------------------------------------------------------
# PRE107: sm_107a-native prefill FMHA (patches/pre107, 1-CTA kernel, lazy rescale). Opt-in, default off.
#   PRE107=1                 enable
#   PRE107_LIB=<path>        libpre1.so (sm_107a build of patches/pre107/kernel/pre1.cu)
#   PRE107_VARIANT=60        kernel variant (60 = 1-CTA, Q in SMEM, separate P, rounded l, HW row max)
#   PRE107_TAU=0.8           lazy rescale threshold, log2 units (0 = exact class)
#   PRE107_MINQ=513          launches with max_q < MINQ stay on gen / stock
# Needs host per-request q / kv lengths for the step (set by the FlashInfer backend via pre107_set_step); a launch without
# them, or with an unexpected layout, falls back to the existing route. Numerics: float-order vs trtllm-gen; tau > 0 is a
# numerics-class change (det + accuracy x3 + SWE before adoption).
_PRE107 = os.environ.get("PRE107", "0") == "1"
_PRE107_LIB = os.environ.get("PRE107_LIB", "")
_PRE107_VARIANT = int(os.environ.get("PRE107_VARIANT", "60"))
_PRE107_TAU = float(os.environ.get("PRE107_TAU", "0.8"))
_PRE107_MINQ = int(os.environ.get("PRE107_MINQ", "513"))
_pre107 = {"lib": None, "key": None, "host": None, "plan": None, "ws": None, "n": 0, "fallback": 0}


def pre107_set_step(key: int, q_lens_cpu: Any, kv_lens_cpu: Any) -> None:
    """Called once per step by the FlashInfer backend with host prefill lengths (kv may be an upper bound)."""
    if _PRE107:
        _pre107["host"] = (key, [int(x) for x in q_lens_cpu], [int(x) for x in kv_lens_cpu])


def _pre107_lib():
    if _pre107["lib"] is None:
        import ctypes

        lib = ctypes.CDLL(_PRE107_LIB)
        lib.pre_plan.restype = ctypes.c_int
        lib.pre_plan.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.pre_run.restype = ctypes.c_int
        lib.pre_run.argtypes = ([ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 3
                                + [ctypes.c_float, ctypes.c_float, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
                                + [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_int])
        _pre107["lib"] = lib
    return _pre107["lib"]


def _pre107_run(query, kv_cache, block_tables, seq_lens, max_q_len, bmm1_scale, bmm2_scale, batch_size, cum_seq_lens_q,
                out) -> bool:
    import ctypes

    host = _pre107["host"]
    B = int(batch_size)
    if host is None or len(host[1]) != B or int(max_q_len) < _PRE107_MINQ:
        return False
    k, v = kv_cache
    # layout: per-layer [pages, Hkv=2, 32, 2*256] fp8 with K|V packed in the last dim (k / v are split views)
    if not (k.dim() == 4 and k.shape[1] == 2 and k.shape[2] == 32 and k.shape[3] == 256 and k.stride(3) == 1
            and k.stride(2) == 512 and k.stride(1) == 32 * 512 and k.stride(0) == 2 * 32 * 512
            and v.data_ptr() == k.data_ptr() + 256 and query.is_contiguous() and out.is_contiguous()
            and query.shape[1] == 16 and query.shape[2] == 256 and block_tables.dtype == torch.int32):
        _pre107["fallback"] += 1
        return False
    lib = _pre107_lib()
    key = host[0]
    if _pre107["key"] != key:  # plan once per step, reused by every full-attention layer
        q_l, kv_l = host[1], host[2]
        n_max = 1 << 15
        dev = query.device
        bufs = _pre107.get("bufs")
        if bufs is None:
            # persistent planner buffers: 2 pinned host slots (alternating, each guarded by an event on its last H2D copy)
            # + one device buffer per table. Avoids per-step 2 MB ctypes allocs and 262K-element list() conversions.
            pin = [(torch.empty(8 * n_max, dtype=torch.int32, pin_memory=True),
                    torch.empty(8 * n_max, dtype=torch.int32, pin_memory=True), torch.cuda.Event()) for _ in range(2)]
            bufs = dict(pin=pin, slot=0, cnt=(ctypes.c_int * 3)(),
                        d_it=torch.empty(8 * n_max, dtype=torch.int32, device=dev),
                        d_mg=torch.empty(8 * n_max, dtype=torch.int32, device=dev),
                        sms=torch.cuda.get_device_properties(dev).multi_processor_count)
            _pre107["bufs"] = bufs
        h_it, h_mg, ev = bufs["pin"][bufs["slot"]]
        ev.synchronize()  # the H2D copy issued from this slot two plans ago has completed
        cnt = bufs["cnt"]
        r = lib.pre_plan((ctypes.c_int * B)(*q_l), (ctypes.c_int * B)(*kv_l), B, bufs["sms"], 0.0, 8, 1,
                         h_it.data_ptr(), n_max, h_mg.data_ptr(), n_max, cnt)
        if r:
            _pre107["fallback"] += 1
            return False
        ni, nm, npart = cnt[0], cnt[1], cnt[2]
        it = bufs["d_it"][: 8 * ni]; it.copy_(h_it[: 8 * ni], non_blocking=True)
        mg = bufs["d_mg"][: 8 * max(nm, 1)]; mg.copy_(h_mg[: 8 * max(nm, 1)], non_blocking=True)
        ev.record(); bufs["slot"] ^= 1
        _npc = int(os.environ.get("PRE107_PLANCHK", "0"))
        if _npc and _pre107.setdefault("pc", [0, 0])[0] < _npc:
            # PRE107_PLANCHK (diagnostic): rebuild this step's plan on the old synchronous path (fresh ctypes tables, list
            # conversion, blocking copy) and require byte-equal device tables; re-verified at every later call of the step.
            items = (ctypes.c_int * (8 * n_max))(); merges = (ctypes.c_int * (8 * n_max))(); cnt2 = (ctypes.c_int * 3)()
            r2 = lib.pre_plan((ctypes.c_int * B)(*q_l), (ctypes.c_int * B)(*kv_l), B, bufs["sms"], 0.0, 8, 1, items, n_max,
                              merges, n_max, cnt2)
            ref_it = torch.tensor(list(items)[: 8 * cnt2[0]], dtype=torch.int32).to(dev)
            ref_mg = torch.tensor(list(merges)[: 8 * max(cnt2[1], 1)], dtype=torch.int32).to(dev)
            ok = (r2 == 0 and tuple(cnt2) == (ni, nm, npart) and torch.equal(it, ref_it) and torch.equal(mg, ref_mg))
            _pre107["pc_ref"] = (ref_it, ref_mg)
            pc = _pre107["pc"]; pc[0] += 1; pc[1] += (not ok)
            if not ok or pc[0] % 50 == 0 or pc[0] == _npc:
                logger.warning("PRE107_PLANCHK n=%d mismatch=%d calls_checked=%d", pc[0], pc[1], _pre107.get("pc_calls", 0))
        else:
            _pre107["pc_ref"] = None
        ws = _pre107["ws"]
        if ws is None or ws[0].numel() < max(npart, 1) * 128 * 256:
            cap = max(npart, 4096)
            ws = (torch.empty(cap * 128 * 256, dtype=torch.float32, device=dev),
                  torch.empty(cap * 128 * 2, dtype=torch.float32, device=dev),
                  torch.zeros(4, dtype=torch.int32, device=dev))
            _pre107["ws"] = ws
        _pre107["plan"] = (it, ni, mg, nm)
        _pre107["key"] = key
    it, ni, mg, nm = _pre107["plan"]
    if _pre107.get("pc_ref") is not None:  # PRE107_PLANCHK: tables still intact at this call of the step
        _pre107["pc_calls"] = _pre107.get("pc_calls", 0) + 1
        if not (torch.equal(it, _pre107["pc_ref"][0]) and torch.equal(mg, _pre107["pc_ref"][1])):
            _pre107["pc"][1] += 1
            logger.warning("PRE107_PLANCHK n=%d mismatch=%d calls_checked=%d (stale at call)", _pre107["pc"][0],
                           _pre107["pc"][1], _pre107["pc_calls"])
    ws_o, ws_ml, flags = _pre107["ws"]
    rc = lib.pre_run(query.data_ptr(), k.data_ptr(), int(k.shape[0]), block_tables.data_ptr(), int(block_tables.stride(0)),
                     seq_lens.data_ptr(), cum_seq_lens_q.data_ptr(), out.data_ptr(), float(bmm1_scale), float(bmm2_scale),
                     it.data_ptr(), ni, mg.data_ptr(), nm, ws_o.data_ptr(), ws_ml.data_ptr(), flags.data_ptr(),
                     torch.cuda.current_stream().cuda_stream, _PRE107_VARIANT, _PRE107_TAU, None, -1)
    if rc:
        _pre107["fallback"] += 1
        return False
    _pre107["n"] += 1
    if os.environ.get("PRE107_LOG", "0") == "1" and _pre107["n"] % 1000 == 1:
        logger.info("PRE107: routed=%d fallback=%d no_host=%d", _pre107["n"], _pre107["fallback"], _pre107.get("no_host", 0))
    return True


def _pre107_check(query, kv_cache, block_tables, seq_lens, cum_seq_lens_q, bmm1_scale, bmm2_scale, out, ship_out) -> None:
    """PRE107_CHECK: per-row rel-L2 of PRE107 and of ship vs an fp32 recompute on <= 64 sampled rows (diagnostic only)."""
    import random

    k, v = kv_cache
    cq = cum_seq_lens_q.tolist(); sl = seq_lens.tolist(); B = len(sl)
    rng = random.Random(len(_pre107.setdefault("chk", [])))
    rows = []
    for _ in range(64):
        b = rng.randrange(B); q = cq[b + 1] - cq[b]
        if q > 0: rows.append((b, rng.randrange(q)))
    e_pre, e_ship = [], []
    scale = float(bmm1_scale)
    for b, t in rows:
        q0 = cq[b]; kvl = sl[b]; P = kvl - (cq[b + 1] - cq[b])
        npg = (kvl + 31) // 32
        pages = block_tables[b, :npg].long()
        K = k[pages].float().permute(1, 0, 2, 3).reshape(2, -1, 256)[:, : P + t + 1]
        V = v[pages].float().permute(1, 0, 2, 3).reshape(2, -1, 256)[:, : P + t + 1]
        Q = query[q0 + t].float()
        ref = torch.empty(16, 256, device=query.device)
        for h in range(16):
            sc = (K[h // 8] @ Q[h]) * scale
            ref[h] = (torch.softmax(sc, -1) @ V[h // 8]) * float(bmm2_scale)
        nr = ref.norm().clamp_min(1e-12)
        e_pre.append(((out[q0 + t].float() - ref).norm() / nr).item())
        e_ship.append(((ship_out[q0 + t].float() - ref).norm() / nr).item())
    def p99(x): x = sorted(x); return x[min(len(x) - 1, int(0.99 * len(x)))]
    rec = dict(B=B, T=int(query.shape[0]), pre_max=max(e_pre), pre_p99=p99(e_pre), ship_max=max(e_ship), ship_p99=p99(e_ship),
               nan=int(torch.isnan(out).any().item()))
    _pre107["chk"].append(rec)
    logger.warning("PRE107_CHECK %s", rec)


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
    if (
        sup
        and _GEN
        and (T <= _GEN_MAX_T or B > _MAX_B)
        and gen_rule_accepts(T, B, int(max_q_len))
    ):
        route = "gen"
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
        if _PRE107 and sup and _pre107_run(query, kv_cache, block_tables, seq_lens, max_q_len, bmm1_scale, bmm2_scale,
                                           batch_size, cum_seq_lens_q, out):
            _ncheck = int(os.environ.get("PRE107_CHECK", "0"))
            if _ncheck and len(_pre107.get("chk", [])) < _ncheck:
                ship_out = torch.empty_like(out)
                if not (route == "gen" and _gen(query, kv_cache, workspace_buffer, block_tables, seq_lens, max_q_len, max_kv_len,
                                                bmm1_scale, batch_size, cum_seq_lens_q, ship_out)):
                    _stock_trtllm_batch_context_with_kv_cache(query, kv_cache, workspace_buffer, block_tables, seq_lens, max_q_len,
                                                              max_kv_len, bmm1_scale, bmm2_scale, batch_size, cum_seq_lens_q,
                                                              cum_seq_lens_kv, *args, **dict(kw, out=ship_out))
                _pre107_check(query, kv_cache, block_tables, seq_lens, cum_seq_lens_q, bmm1_scale, bmm2_scale, out, ship_out)
            _ndet = int(os.environ.get("PRE107_DET", "0"))
            if _ndet and _pre107.setdefault("det", [0, 0])[0] < _ndet:
                # PRE107_DET: rerun on the same inputs (same step plan) and require bitwise-equal output (diagnostic only)
                out2 = torch.empty_like(out)
                ok2 = _pre107_run(query, kv_cache, block_tables, seq_lens, max_q_len, bmm1_scale, bmm2_scale, batch_size,
                                  cum_seq_lens_q, out2)
                d = _pre107["det"]; d[0] += 1
                if not (ok2 and torch.equal(out.view(torch.uint8), out2.view(torch.uint8))):
                    d[1] += 1
                if d[0] % 50 == 0 or d[0] == _ndet or d[1] == 1:
                    logger.warning("PRE107_DET n=%d mismatch=%d T=%d B=%d", d[0], d[1], int(query.shape[0]), int(batch_size))
            return out
        if route == "gen" and _gen(
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
