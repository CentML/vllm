# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic only (not for upstream): dump MoE routing rows to files.

Set ``VLLM_DIAG_MOE_ROUTE_DUMP=<dir>``. Two kinds of record, per process:

- ``tune_*``: the post-dispatch ``topk_ids`` handed to FlashInfer's trtllm-gen
  FP4 MoE while its AutoTuner is in tuning mode, for the first MoE layer of
  each tuning dummy run. FlashInfer replays a prefix of exactly these rows
  when it profiles buckets no larger than the call (``make_repeating_tensor_
  initializer``), so they are the routing the tactics were chosen on.
- ``serve_*``: the same tensor in eager serving calls (no CUDA graph capture)
  with at most ``VLLM_DIAG_MOE_ROUTE_MAX_M`` rows, for MoE layers 0, 21 and
  41, every ``VLLM_DIAG_MOE_ROUTE_EVERY``-th call of that layer, up to
  ``VLLM_DIAG_MOE_ROUTE_MAX_FILES`` files per layer. Run the server with
  enforce-eager so decode steps reach Python.

Router statistics for the same layers (non-finite router logits, rows) go to
``router_*``. Every save syncs the GPU; never use this in a performance run.
"""

import os
from typing import Any

import torch

_DIR = os.environ.get("VLLM_DIAG_MOE_ROUTE_DUMP") or None
_MAX_M = int(os.environ.get("VLLM_DIAG_MOE_ROUTE_MAX_M", "8192"))
_EVERY = int(os.environ.get("VLLM_DIAG_MOE_ROUTE_EVERY", "16"))
_MAX_FILES = int(os.environ.get("VLLM_DIAG_MOE_ROUTE_MAX_FILES", "160"))
_SERVE_LAYERS = (0, 21, 41)

_state: dict[str, Any] = {"experts": {}, "routers": {}, "calls": {}, "files": {}}


def enabled() -> bool:
    return _DIR is not None


def _tuning() -> bool:
    try:
        from flashinfer.autotuner import AutoTuner
    except ImportError:
        return False
    return bool(AutoTuner.get().is_tuning_mode)


def _out_dir() -> str:
    assert _DIR is not None
    rank = os.environ.get("VLLM_DP_RANK") or os.environ.get("RANK") or "x"
    path = os.path.join(_DIR, f"dev{torch.cuda.current_device()}-dp{rank}-pid{os.getpid()}")
    os.makedirs(path, exist_ok=True)
    return path


def _layer_index(kind: str, obj: object) -> int:
    table = _state[kind]
    if id(obj) not in table:
        table[id(obj)] = len(table)
    return table[id(obj)]


def _save(name: str, payload: dict) -> None:
    torch.save(payload, os.path.join(_out_dir(), name + ".pt"))


def dump_expert_rows(experts: object, topk_ids: torch.Tensor, extra: dict) -> None:
    """Record the topk_ids a trtllm-gen MoE experts call receives."""
    if _DIR is None or torch.cuda.is_current_stream_capturing():
        return
    layer = _layer_index("experts", experts)
    m = int(topk_ids.shape[0])
    if _tuning():
        if layer != 0:
            return
        n = _state["files"].get("tune", 0)
        _state["files"]["tune"] = n + 1
        _save(
            f"tune_{n:02d}_M{m}",
            dict(kind="tune", layer=layer, M=m, topk_ids=topk_ids.cpu(), **extra),
        )
        return
    if layer not in _SERVE_LAYERS or m > _MAX_M:
        return
    calls = _state["calls"].get(layer, 0)
    _state["calls"][layer] = calls + 1
    nfiles = _state["files"].get(layer, 0)
    if calls % _EVERY or nfiles >= _MAX_FILES:
        return
    _state["files"][layer] = nfiles + 1
    _save(
        f"serve_L{layer:02d}_{nfiles:04d}_M{m}",
        dict(kind="serve", layer=layer, M=m, call=calls, topk_ids=topk_ids.cpu(), **extra),
    )


def dump_router_stats(router: object, router_logits: torch.Tensor, topk_ids: torch.Tensor) -> None:
    """Record per-call router statistics (non-finite logits, routed ids)."""
    if _DIR is None or torch.cuda.is_current_stream_capturing():
        return
    layer = _layer_index("routers", router)
    tuning = _tuning()
    if layer not in _SERVE_LAYERS or (not tuning and topk_ids.shape[0] > _MAX_M):
        return
    key = ("router", tuning, layer)
    calls = _state["calls"].get(key, 0)
    _state["calls"][key] = calls + 1
    nfiles = _state["files"].get(key, 0)
    if (not tuning and calls % _EVERY) or nfiles >= (8 if tuning else _MAX_FILES):
        return
    _state["files"][key] = nfiles + 1
    finite = torch.isfinite(router_logits.float()).all(dim=-1)
    _save(
        f"router_{'tune' if tuning else 'serve'}_L{layer:02d}_{nfiles:04d}_M{topk_ids.shape[0]}",
        dict(
            kind="router",
            tuning=tuning,
            layer=layer,
            rows=int(router_logits.shape[0]),
            nonfinite_rows=int((~finite).sum()),
            topk_ids=topk_ids.cpu(),
        ),
    )
