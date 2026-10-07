# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hang forensics for the FlashInfer SM107 MegaMoE (debug only).

Enabled by ``VLLM_FI_MEGA_MOE_DEBUG_DIR=<dir>``; nothing here runs otherwise.

* ``mark()`` enqueues a one-thread Triton kernel that appends a record
  ``(seq, layer, capacity, tokens, phase, stream)`` to a per-process device ring.
  It is captured into CUDA graphs like any other kernel, so graph replays are
  traced too. Phases: 0 before staging, 1 after the megakernel (main stream),
  2 / 3 before / after the overlapped shared MLP (side stream).
* A daemon thread polls ``<dir>/dump.req``. Each time its mtime changes it
  copies the ring and the megakernel's synchronization state (every non-data
  region of the local / symmetric workspaces: NVLink barrier phase counter and
  signals, grid-sync counter, per-launch flags) to pinned host memory on a
  private stream and writes ``<dir>/fimega_<tag>_r<rank>_<pid>.json``. The copy
  does not wait for the (possibly spinning) compute stream, so it works while
  the GPU is deadlocked.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_RING = 4096
_I64 = 1 << 62
_FIELDS = 4
_MAX_REGION_BYTES = 1 << 20

_STATE: dict[str, Any] = {}
_LOCK = threading.Lock()


def debug_dir() -> str | None:
    return os.environ.get("VLLM_FI_MEGA_MOE_DEBUG_DIR") or None


def enabled() -> bool:
    return debug_dir() is not None


def _mark_kernel():
    kernel = _STATE.get("kernel")
    if kernel is not None:
        return kernel
    import triton
    import triton.language as tl

    # No value specialization and always-int64 arguments (bit 62 set): one
    # compiled variant, so no JIT ever happens inside a CUDA-graph capture.
    @triton.jit(do_not_specialize=["a", "b", "c"])
    def _fi_mega_mark(ring_ptr, counter_ptr, a, b, c, RING: tl.constexpr):
        seq = tl.atomic_add(counter_ptr, 1)
        base = ring_ptr + (seq % RING) * 4
        tl.store(base + 1, a)
        tl.store(base + 2, b)
        tl.store(base + 3, c)
        tl.store(base + 0, seq + 1)

    _STATE["kernel"] = _fi_mega_mark
    return _fi_mega_mark


def _ensure_ring(device: torch.device) -> None:
    if "ring" in _STATE:
        return
    with _LOCK:
        if "ring" in _STATE:
            return
        _STATE["ring"] = torch.zeros(_RING * _FIELDS, dtype=torch.int64, device=device)
        _STATE["counter"] = torch.zeros(1, dtype=torch.int64, device=device)
        _STATE["ring_host"] = torch.zeros(_RING * _FIELDS + 1, dtype=torch.int64).pin_memory()
        _STATE["workspaces"] = []
        _STATE["layers"] = {}
        _STATE["device"] = device
        _STATE["copy_stream"] = torch.cuda.Stream(device=device)
        _mark_kernel()
        t = threading.Thread(target=_watch, name="fi_mega_debug", daemon=True)
        t.start()


def layer_id(prefix: str) -> int:
    """Stable small id: target layer index, draft layers offset by 1000."""
    layers = _STATE.setdefault("layers", {})
    if prefix in layers:
        return layers[prefix]
    idx = len(layers)
    digits = [int(p) for p in prefix.split(".") if p.isdigit()]
    if digits:
        idx = digits[0]
    if any(k in prefix for k in ("mtp", "draft", "dspark", "spec")):
        idx += 1000
    while idx in layers.values():
        idx += 10000
    layers[prefix] = idx
    return idx


def register_workspace(tag: str, cap: int, sym_buffer: Any) -> None:
    """Remember a pooled SM107 workspace (deduplicated by address)."""
    if not enabled():
        return
    _ensure_ring(torch.device("cuda", torch.cuda.current_device()))
    key = int(sym_buffer.local_workspace.data_ptr())
    for ws in _STATE["workspaces"]:
        if ws["key"] == key:
            ws["tags"].add(f"{tag}@{cap}")
            return
    kernel = getattr(sym_buffer, "kernel", None)
    dw = getattr(kernel, "_device_workspace", None)
    regions = []
    if dw is not None:
        for space in ("local", "shared"):
            for region in dw.regions(space):
                if region.reset == "data":
                    continue
                nbytes = int(dw.nbytes(region.name))
                if nbytes > _MAX_REGION_BYTES:
                    nbytes = _MAX_REGION_BYTES
                regions.append(
                    {
                        "name": region.name,
                        "space": space,
                        "reset": region.reset,
                        "offset": int(dw.offset(region.name)),
                        "nbytes": nbytes,
                        "width": int(region.dtype.width),
                    }
                )
    total = sum(r["nbytes"] for r in regions)
    host = torch.zeros(max(total, 16), dtype=torch.uint8).pin_memory()
    cfg = getattr(sym_buffer, "config", None)
    _STATE["workspaces"].append(
        {
            "key": key,
            "tags": {f"{tag}@{cap}"},
            "buffer": sym_buffer,
            "regions": regions,
            "host": host,
            "config": {
                "max_tokens_per_rank": getattr(cfg, "max_tokens_per_rank", None),
                "max_sm_count": repr(getattr(cfg, "max_sm_count", None)),
                "cluster_shape_mn": repr(getattr(cfg, "cluster_shape_mn", None)),
                "fallback_cluster_shape_mn": repr(
                    getattr(cfg, "fallback_cluster_shape_mn", None)
                ),
                "rank": getattr(cfg, "rank", None),
                "world_size": getattr(cfg, "world_size", None),
                "quant_kind": getattr(cfg, "quant_kind", None),
            },
        }
    )


def mark(layer: int, cap: int, tokens: int, phase: int, side: bool = False) -> None:
    """Append a trace record on the current stream (graph-capturable)."""
    ring = _STATE.get("ring")
    if ring is None:
        return
    a = _I64 | ((int(layer) & 0x3FFFFFFF) << 32) | (int(cap) & 0xFFFFFFFF)
    b = _I64 | ((int(tokens) & 0xFFFFFFFF) << 8) | (int(phase) & 0xFF)
    stamp = 0 if torch.cuda.is_current_stream_capturing() else int(time.time())
    c = _I64 | (stamp << 1) | (1 if side else 0)
    _mark_kernel()[(1,)](ring, _STATE["counter"], a, b, c, RING=_RING)


def _decode_ring(host: torch.Tensor) -> dict[str, Any]:
    ring = host[: _RING * _FIELDS].view(_RING, _FIELDS).tolist()
    counter = int(host[_RING * _FIELDS])
    recs = []
    for seq1, a, b, c in ring:
        if seq1 <= 0:
            continue
        a, b, c = a & ~_I64, b & ~_I64, c & ~_I64
        recs.append(
            {
                "seq": seq1 - 1,
                "layer": a >> 32,
                "cap": a & 0xFFFFFFFF,
                "tokens": b >> 8,
                "phase": b & 0xFF,
                "side": c & 1,
                "host_time": c >> 1,
            }
        )
    recs.sort(key=lambda r: r["seq"])
    return {"counter": counter, "last": recs[-64:]}


def _region_summary(raw: torch.Tensor, region: dict[str, Any]) -> dict[str, Any]:
    width = region["width"]
    nbytes = region["nbytes"] - region["nbytes"] % 4
    words = raw[:nbytes].view(torch.int32) if nbytes >= 4 else raw[:0].view(torch.int32)
    out: dict[str, Any] = {k: region[k] for k in ("name", "space", "offset", "nbytes")}
    out["width"] = width
    nz = torch.nonzero(words).flatten()
    out["nonzero_words"] = int(nz.numel())
    if words.numel() <= 16 or "progress" in region["name"]:
        out["words"] = words.tolist()
    else:
        out["first_nonzero"] = [
            [int(i), int(words[i])] for i in nz[:32].tolist()
        ]
    return out


def dump(tag: str) -> str | None:
    """Copy ring + workspace sync state without waiting on compute streams."""
    d = debug_dir()
    if d is None or "ring" not in _STATE:
        return None
    dev = _STATE["device"]
    stream = _STATE["copy_stream"]
    host = _STATE["ring_host"]
    with torch.cuda.device(dev), torch.cuda.stream(stream):
        host[: _RING * _FIELDS].copy_(_STATE["ring"], non_blocking=True)
        host[_RING * _FIELDS :].copy_(_STATE["counter"], non_blocking=True)
        for ws in _STATE["workspaces"]:
            buf = ws["buffer"]
            pos = 0
            for region in ws["regions"]:
                src = buf.local_workspace if region["space"] == "local" else buf.shared_workspace
                n = region["nbytes"]
                ws["host"][pos : pos + n].copy_(
                    src[region["offset"] : region["offset"] + n], non_blocking=True
                )
                pos += n
    t0 = time.time()
    while not stream.query():
        if time.time() - t0 > 30:
            logger.warning("fi_mega_debug: device copy did not finish in 30 s")
            break
        time.sleep(0.01)
    rank = None
    try:
        from vllm.distributed import get_ep_group

        rank = get_ep_group().rank_in_group
    except Exception:  # noqa: BLE001
        pass
    payload: dict[str, Any] = {
        "tag": tag,
        "time": time.time(),
        "pid": os.getpid(),
        "ep_rank": rank,
        "device": str(dev),
        "ring": _decode_ring(host),
        "layers": _STATE.get("layers", {}),
        "workspaces": [],
    }
    for ws in _STATE["workspaces"]:
        pos = 0
        regions = []
        for region in ws["regions"]:
            n = region["nbytes"]
            regions.append(_region_summary(ws["host"][pos : pos + n], region))
            pos += n
        payload["workspaces"].append(
            {"tags": sorted(ws["tags"]), "config": ws["config"], "regions": regions}
        )
    path = os.path.join(d, f"fimega_{tag}_r{rank}_{os.getpid()}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return path


def _watch() -> None:
    d = debug_dir()
    assert d is not None
    req = os.path.join(d, "dump.req")
    last = None
    try:
        last = os.stat(req).st_mtime_ns
    except FileNotFoundError:
        pass
    while True:
        time.sleep(1.0)
        try:
            mtime = os.stat(req).st_mtime_ns
        except FileNotFoundError:
            continue
        if mtime == last:
            continue
        last = mtime
        try:
            path = dump(str(mtime))
            logger.warning("fi_mega_debug: wrote %s", path)
        except Exception as e:  # noqa: BLE001
            logger.warning("fi_mega_debug: dump failed: %r", e)
