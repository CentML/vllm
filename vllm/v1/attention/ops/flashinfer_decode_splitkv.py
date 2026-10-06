# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-aware split-KV widening for FlashInfer trtllm-gen decode.

FlashInfer picks the KV split of its trtllm-gen generation kernels on the host
(``computeCtaAndClusterConfig`` in
``include/flashinfer/trtllm/fmha/fmhaKernels.cuh``)::

    numCtasPerSeqKv = min(
        ceil(maxSeqLenKv / 512), floor(sm_count / (numCtasQ * numHeadsKv * batch))
    )

vLLM captures decode CUDA graphs with ``max_seq_len = max_model_len``, so only
the SM term binds. For GQA models with few KV heads that term reaches 1 at
moderate batch sizes: split-KV is disabled and the kernel runs one persistent
CTA per (request, KV head), which cannot balance long, uneven context lengths.
``sm_count`` is looked up in Python (``flashinfer.decode.get_device_sm_count``),
so the split count can be raised host-side with no kernel change (same cubin
family, multi-CTA KV mode with in-kernel reduction).

Policy (environment, read at import):

``FI_DECODE_SPLITKV``
    ``off`` (default) disables the feature; any other value (e.g. ``auto``)
    enables it (``off``, ``0``, ``false`` and ``none`` all disable)::

        virtual_sm = max(SMs, min(CTAS_PER_SM * SMs, MAX_SPLITS * batch * num_kv_heads))

    with ``CTAS_PER_SM = FI_DECODE_SPLITKV_CTAS_PER_SM`` (default 16) and
    ``MAX_SPLITS = FI_DECODE_SPLITKV_MAX_SPLITS`` (default 16). Small batches
    keep the stock heuristic (it already splits them); large batches get up to
    ``MAX_SPLITS`` KV splits per sequence spread over several waves.
``FI_DECODE_VSM_SCALE=<f>``
    Fixed mode (overrides the batch-aware policy):
    ``virtual_sm = int(f * SMs)`` for every call.
``FI_DECODE_VSM_LOG=1``
    Log, once per (batch, q_len, num_kv_heads), the virtual SM count chosen.

Hardening that comes with the wider split:

* Persistent multi-CTA counter buffer: when the caller passes none (vLLM does
  not), one zeroed buffer per device is reused for every call; the kernel
  resets its semaphores to 0 at the end of each launch. This removes a
  per-layer ``torch.zeros`` in eager steps and CUDA graphs. The buffer grows
  (never shrinks) if a call needs more; the first allocation is sized for a
  batch of 1024 x 64 query heads. Use :func:`preallocate_counter_buffer` to
  allocate it outside CUDA-graph capture.
* Workspace guard: every call checks on the host (no device sync) that the
  trtllm workspace is at least ``virtual_sm * 32 * (8 + head_dim_v * 4) B +
  1 MiB`` (multi-CTA partial O / stats scratch, which FlashInfer does not bound
  check) and raises ``RuntimeError`` otherwise, i.e. at startup / warmup rather
  than as a silent out-of-bounds write at runtime.

The decision is taken on the host, so during CUDA-graph capture (vLLM captures
with ``max_seq_len = max_model_len`` and the padded batch) it is frozen into
the graph and applies to every replay.

FlashInfer adapter: ``trtllm_batch_decode_with_kv_cache`` has no argument for
the split count or the SM count, so when the feature is enabled vLLM replaces
``flashinfer.decode.get_device_sm_count`` by a function that returns a
thread-local override while a widened call is in flight and the real value
otherwise. The proper home of this knob is an explicit ``sm_count`` / split
argument in FlashInfer.
"""

import os
import threading
from collections.abc import Callable
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MODE = os.environ.get("FI_DECODE_SPLITKV", "off").strip().lower()
ENABLED = _MODE not in ("off", "0", "false", "none")
_SCALE = float(os.environ.get("FI_DECODE_VSM_SCALE", "0") or 0)
_CTAS_PER_SM = float(os.environ.get("FI_DECODE_SPLITKV_CTAS_PER_SM", "16"))
_MAX_SPLITS = int(os.environ.get("FI_DECODE_SPLITKV_MAX_SPLITS", "16"))
_LOG = os.environ.get("FI_DECODE_VSM_LOG", "0") == "1"

# Virtual SM count of the call in flight on this thread (None: real value).
_TLS = threading.local()
# Persistent multi-CTA counter buffers, keyed by (device.type, device.index).
_COUNTERS: dict[tuple[str, int | None], torch.Tensor] = {}
# Called as fn(key, buffer, replaced) after every counter buffer (re)allocation.
_COUNTER_ALLOC_CALLBACKS: list[
    Callable[[tuple[str, int | None], torch.Tensor, bool], None]
] = []
_LOGGED: set[tuple[Any, Any, Any]] = set()

# Adapter state: None = not installed yet, True = installed, False = failed.
_installed: bool | None = None
_real_sm_count: Callable[[torch.device], int] | None = None
_orig_decode: Callable[..., Any] | None = None


def _workspace_needed(vsm: int, step_q: int = 32, head_dim_v: int = 256) -> int:
    return vsm * step_q * (8 + head_dim_v * 4) + (1 << 20)


def _counter_buffer(
    device: torch.device, batch: int, num_qo_heads: int, vsm: int
) -> torch.Tensor:
    need = ((max(batch * num_qo_heads, vsm) + 7) // 8) * 8 * 4
    key = (device.type, device.index)
    buf = _COUNTERS.get(key)
    if buf is None or buf.numel() < need:
        replaced = buf is not None
        size = max(need, ((max(1024 * 64, vsm) + 7) // 8) * 8 * 4)
        buf = torch.zeros(size, dtype=torch.uint8, device=device)
        _COUNTERS[key] = buf
        for callback in _COUNTER_ALLOC_CALLBACKS:
            callback(key, buf, replaced)
    return buf


def get_counter_buffer(device: torch.device) -> torch.Tensor | None:
    """Return the persistent counter buffer of ``device`` if allocated."""
    return _COUNTERS.get((device.type, device.index))


def preallocate_counter_buffer(device: torch.device) -> torch.Tensor:
    """Allocate the persistent multi-CTA counter buffer of ``device`` now.

    Uses the default first-allocation size (batch 1024 x 64 query heads) and
    is a no-op if a buffer already exists. Call it outside CUDA-graph capture:
    a buffer first allocated during capture would come from the capture pool
    and its zero-fill would only be captured, not executed.
    """
    return _counter_buffer(device, 1, 1, 0)


def register_counter_alloc_callback(
    callback: Callable[[tuple[str, int | None], torch.Tensor, bool], None],
) -> None:
    """Register ``callback(key, buffer, replaced)``, run after every counter
    buffer (re)allocation (e.g. for allocation diagnostics).
    """
    _COUNTER_ALLOC_CALLBACKS.append(callback)


def policy_sm_count(
    real_sms: int,
    batch: int,
    num_kv_heads: int,
    scale: float | None = None,
    ctas_per_sm: float | None = None,
    max_splits: int | None = None,
) -> int:
    """Virtual SM count handed to the trtllm-gen host heuristic."""
    scale = _SCALE if scale is None else scale
    ctas_per_sm = _CTAS_PER_SM if ctas_per_sm is None else ctas_per_sm
    max_splits = _MAX_SPLITS if max_splits is None else max_splits
    if scale and scale > 0:
        return int(real_sms * scale)
    want = min(
        int(ctas_per_sm * real_sms),
        max_splits * max(1, batch) * max(1, num_kv_heads),
    )
    return max(real_sms, want)


def _install() -> bool:
    """Install the thread-local ``get_device_sm_count`` adapter (once).

    FlashInfer adapter: ``flashinfer.decode`` looks ``get_device_sm_count`` up
    as a module global inside ``trtllm_batch_decode_with_kv_cache``; replacing
    it is the only way to pass a split target today. Outside a widened call the
    replacement returns the real SM count, so every other FlashInfer caller is
    unaffected.
    """
    global _installed, _real_sm_count, _orig_decode
    if _installed is not None:
        return _installed
    import flashinfer.decode as fid

    _orig_decode = fid.trtllm_batch_decode_with_kv_cache
    try:
        real = fid.get_device_sm_count

        def sm_count(device):
            v = getattr(_TLS, "sm", None)
            return v if v is not None else real(device)

        fid.get_device_sm_count = sm_count
        _real_sm_count = real
    except Exception as e:  # never break serving: keep the stock kernel path
        _installed = False
        logger.warning("trtllm-gen decode split-KV widening not installed: %r", e)
        return False
    _installed = True
    desc = (
        f"fixed x{_SCALE:g}"
        if _SCALE > 0
        else f"auto (ctas/SM {_CTAS_PER_SM:g}, max splits {_MAX_SPLITS})"
    )
    logger.info_once("trtllm-gen decode split-KV widening enabled: %s", desc)
    return True


def trtllm_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Any,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """``flashinfer.decode.trtllm_batch_decode_with_kv_cache`` with a
    batch-aware virtual SM count (see module docstring). Same signature.
    """
    if not _install():
        assert _orig_decode is not None
        return _orig_decode(
            query, kv_cache, workspace_buffer, block_tables, seq_lens, *args, **kwargs
        )
    assert _real_sm_count is not None and _orig_decode is not None
    real = _real_sm_count
    vsm: int | None
    batch: Any
    qlen: Any
    hkv: Any
    try:
        qlen = kwargs.get("q_len_per_req", 1)
        cu = kwargs.get("cum_seq_lens_q")
        if qlen is None and cu is not None:
            batch = cu.size(0) - 1
        else:
            batch = query.size(0) // max(1, qlen or 1)
        kc = kv_cache[0] if isinstance(kv_cache, (tuple, list)) else kv_cache
        hkv = kc.size(-3)  # HND: [pages, (2,) H, P, D]
        vsm = policy_sm_count(real(query.device), batch, hkv)
    except Exception:
        vsm, batch, qlen, hkv = None, None, None, None
    if vsm is not None:
        ws_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
        need = _workspace_needed(vsm, head_dim_v=query.size(-1))
        if ws_bytes < need:
            raise RuntimeError(
                f"trtllm-gen decode split-KV: trtllm workspace {ws_bytes} B < "
                f"{need} B needed for virtual sm_count {vsm}; raise "
                "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE or lower "
                "FI_DECODE_SPLITKV_CTAS_PER_SM"
            )
        if kwargs.get("multi_ctas_kv_counter_buffer") is None and query.is_cuda:
            kwargs["multi_ctas_kv_counter_buffer"] = _counter_buffer(
                query.device, batch, query.size(1), vsm
            )
        if _LOG and (batch, qlen, hkv) not in _LOGGED:
            _LOGGED.add((batch, qlen, hkv))
            logger.info(
                "trtllm-gen decode split-KV: batch=%s q_len=%s hkv=%s hq=%s "
                "sm_count %s -> %s (~%s KV splits)",
                batch,
                qlen,
                hkv,
                query.size(1),
                real(query.device),
                vsm,
                max(1, vsm // max(1, batch * hkv)),
            )
    _TLS.sm = vsm
    try:
        return _orig_decode(
            query, kv_cache, workspace_buffer, block_tables, seq_lens, *args, **kwargs
        )
    finally:
        _TLS.sm = None
