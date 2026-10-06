# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State hygiene around the two-phase CUDA graph capture of the model runner.

GPUModelRunner.profile_cudagraph_memory() captures every CUDA graph once
against a throwaway minimal KV cache and graph pool, tears that state down,
and the real capture runs later against the real KV cache. Two kinds of state
can outlive the profiling phase:

1. Lazily allocated persistent device buffers. The trtllm-gen decode split-KV
   widening (vllm.v1.attention.ops.flashinfer_decode_splitkv) keeps one zeroed
   multi-CTA counter buffer per device, allocated on the first decode call.
   If that call happens inside a capture, the memory comes from the capture
   pool and its zero-fill is only captured, never executed. CFIX=1 allocates
   the buffer eagerly, outside any capture, right before the profiling capture
   and before the real capture.
2. Host-side tables holding views of the throwaway profiling KV cache, which
   keep it allocated for the whole run: the per-layer and per-worker tables of
   the deferred GDN state commit and the per-KV-group tables of the GDN step
   plan. CFIX_RESET=1 drops them after profiling (they are rebuilt from the
   live KV cache on first use, keyed by its pointers) and releases the memory.

When either is enabled, the following is also logged: every counter buffer
(re)allocation (capturing or not, phase, pointer), the counter buffer after
each capture (its non-zero byte count must be 0), and the bytes these tables
pin after profiling and after the real capture.

Env: CFIX=1, CFIX_RESET=1 (default 0 each).
Numerics: unchanged (buffer allocation time and host-table lifetime only).
"""

import gc
import os
import sys

import torch

from vllm.logger import init_logger
from vllm.v1.attention.ops import flashinfer_decode_splitkv as splitkv

logger = init_logger(__name__)

CFIX = os.environ.get("CFIX", "0") == "1"
CFIX_RESET = os.environ.get("CFIX_RESET", "0") == "1"
ENABLED = CFIX or CFIX_RESET

_TAG = "CUDA graph profiling cleanup: "
_GDN_STEP_PLAN = "vllm.model_executor.layers.mamba.gdn.gdn_step_plan"
_GDN_STATE_COMMIT = "vllm.model_executor.layers.mamba.ops.gdn_state_commit"
_PH = ["init"]


def _log(msg: str, *args) -> None:
    logger.info("%s%s", _TAG, msg % args if args else msg)


def _on_counter_alloc(key, buf: torch.Tensor, replaced: bool) -> None:
    _log(
        "decode split-KV counter alloc dev=%s bytes=%d ptr=%#x capturing=%s "
        "phase=%s replaced=%s",
        key,
        buf.numel(),
        buf.data_ptr(),
        torch.cuda.is_current_stream_capturing(),
        _PH[0],
        replaced,
    )


if ENABLED:
    splitkv.register_counter_alloc_callback(_on_counter_alloc)
    _log(
        "enabled: CFIX=%d CFIX_RESET=%d decode_splitkv=%s",
        int(CFIX),
        int(CFIX_RESET),
        splitkv.ENABLED,
    )


def _prealloc(device: torch.device, where: str) -> None:
    if not splitkv.ENABLED:
        _log("decode split-KV widening off (%s); nothing to do", where)
        return
    if (
        CFIX
        and splitkv.get_counter_buffer(device) is None
        and not torch.cuda.is_current_stream_capturing()
    ):
        # default first-allocation size (batch 1024 x 64 heads)
        buf = splitkv.preallocate_counter_buffer(device)
        torch.cuda.synchronize(device)
        _log("eager prealloc %s: ptr=%#x bytes=%d", where, buf.data_ptr(), buf.numel())


def _report(device: torch.device, where: str) -> None:
    if not splitkv.ENABLED:
        return
    buf = splitkv.get_counter_buffer(device)
    if buf is None:
        _log("%s: no counter buffer yet", where)
        return
    torch.cuda.synchronize(device)
    _log(
        "%s: counter buffer ptr=%#x bytes=%d nonzero=%d",
        where,
        buf.data_ptr(),
        buf.numel(),
        int(buf.count_nonzero()),
    )


def _tensors(obj, depth=0, out=None):
    out = [] if out is None else out
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif depth < 6 and isinstance(obj, (list, tuple)):
        for x in obj:
            _tensors(x, depth + 1, out)
    elif depth < 6 and isinstance(obj, dict):
        for x in obj.values():
            _tensors(x, depth + 1, out)
    elif depth < 6 and hasattr(obj, "__dict__") and not isinstance(obj, type):
        for k, x in vars(obj).items():
            if (
                k in ("_keep", "entries", "table", "lt", "t")
                or isinstance(x, (list, tuple, dict))
                or hasattr(x, "_keep")
            ):
                _tensors(x, depth + 1, out)
    return out


def _pinned_tables(runner):
    """(name, object) for every table that may hold profiling-phase KV views.
    Only modules the process already imported are consulted.
    """
    found = []
    plan = sys.modules.get(_GDN_STEP_PLAN)
    if plan is not None and getattr(plan, "_TABLES", None):
        found.append(("gdn step plan tables", plan._TABLES))
    gsc = sys.modules.get(_GDN_STATE_COMMIT)
    if (
        gsc is not None
        and getattr(getattr(gsc, "_WorkerTables", None), "table", None) is not None
    ):
        found.append(("gdn state commit worker table", gsc._WorkerTables.table))
    try:
        sfc = runner.vllm_config.compilation_config.static_forward_context
    except Exception:
        sfc = {}
    lt = [
        (n, L._gsc_table)
        for n, L in sfc.items()
        if getattr(L, "_gsc_table", None) is not None
    ]
    if lt:
        found.append((f"gdn state commit layer tables x{len(lt)}", [t for _, t in lt]))
    if plan is not None and getattr(plan, "_GROUP_CACHE", None):
        found.append(("gdn group materialize tables", plan._GROUP_CACHE))
    return found, sfc


def _pinned_bytes(runner, found):
    params = set()
    try:
        for p in runner.model.parameters():
            params.add(p.untyped_storage().data_ptr())
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    seen, per = {}, {}
    for name, obj in found:
        for t in _tensors(obj):
            st = t.untyped_storage()
            k = st.data_ptr()
            if k in params or k == 0:
                continue
            if k not in seen:
                seen[k] = st.nbytes()
                per[name] = per.get(name, 0) + st.nbytes()
    return sum(seen.values()), per


def _reset_tables(runner, where: str) -> None:
    found, sfc = _pinned_tables(runner)
    total, per = _pinned_bytes(runner, found)
    _log(
        "%s: pinned profiling-KV storages %.1f MiB by %s (allocated %.2f GiB)",
        where,
        total / 2**20,
        per,
        torch.cuda.memory_allocated() / 2**30,
    )
    if not CFIX_RESET:
        return
    torch.cuda.synchronize()
    a0 = torch.cuda.memory_allocated()
    plan = sys.modules.get(_GDN_STEP_PLAN)
    if plan is not None and getattr(plan, "_TABLES", None) is not None:
        plan._TABLES.clear()
    gsc = sys.modules.get(_GDN_STATE_COMMIT)
    if gsc is not None and hasattr(gsc, "_WorkerTables"):
        gsc._WorkerTables.table = None
    for L in sfc.values():
        if getattr(L, "_gsc_table", None) is not None:
            L._gsc_table = None
            L._gsc_checked = None
    if plan is not None and getattr(plan, "_GROUP_CACHE", None) is not None:
        plan._GROUP_CACHE.clear()
    gc.collect()
    torch.cuda.synchronize()
    a1 = torch.cuda.memory_allocated()
    torch.cuda.empty_cache()
    _log(
        "%s: reset done, freed %.1f MiB (allocated %.2f GiB)",
        where,
        (a0 - a1) / 2**20,
        a1 / 2**30,
    )


def _post_capture_check(runner) -> None:
    found, _ = _pinned_tables(runner)
    live = set()
    for L in runner.vllm_config.compilation_config.static_forward_context.values():
        kv = getattr(L, "kv_cache", None)
        kvs = (
            [kv]
            if torch.is_tensor(kv)
            else (list(kv) if isinstance(kv, (list, tuple)) else [])
        )
        for t in kvs:
            if torch.is_tensor(t):
                live.add(t.untyped_storage().data_ptr())
    stale = 0
    seen = set()
    for _, obj in found:
        for t in _tensors(obj):
            k = t.untyped_storage().data_ptr()
            if (
                k not in live
                and k not in seen
                and t.untyped_storage().nbytes() > (1 << 20)
            ):
                seen.add(k)
                stale += t.untyped_storage().nbytes()
    _log(
        "after real capture: tables still pin %.1f MiB of non-live KV storage; "
        "allocated %.2f GiB",
        stale / 2**20,
        torch.cuda.memory_allocated() / 2**30,
    )


def profile_cudagraph_memory(runner, impl):
    """GPUModelRunner.profile_cudagraph_memory with CFIX / CFIX_RESET."""
    _prealloc(runner.device, "before profiling capture")
    _PH[0] = "prof"
    try:
        return impl()
    finally:
        _PH[0] = "after-prof"
        _report(runner.device, "after profiling capture")
        try:
            _reset_tables(runner, "after profiling capture")
        except Exception as e:  # noqa: BLE001 - never break startup
            logger.warning("%stable reset failed: %r", _TAG, e)


def capture_model(runner, impl, profile_only: bool):
    """GPUModelRunner.capture_model with CFIX / CFIX_RESET (profile_only
    captures, issued from profile_cudagraph_memory, are passed through).
    """
    if not profile_only:
        _prealloc(runner.device, "before real capture")
        _PH[0] = "capture"
    try:
        return impl()
    finally:
        if not profile_only:
            _PH[0] = "run"
            _report(runner.device, "after real capture")
            try:
                _post_capture_check(runner)
            except Exception as e:  # noqa: BLE001 - diagnostics only
                logger.warning("%spost-capture pin check failed: %r", _TAG, e)
