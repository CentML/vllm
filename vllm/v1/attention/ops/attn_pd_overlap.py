# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in decode || prefill overlap of the trtllm-gen attention kernels in
mixed steps (``FlashInferImpl.forward``).

In a mixed step the prefill (context) FMHA and the decode (generation) FMHA
of one attention layer are independent: both only read the KV cache and they
write disjoint rows of the output. By default they run back to back on the
current stream. The prefill kernel is compute-bound and its last, partial wave
leaves SMs idle; the decode kernel is HBM-bound. With this feature the two run
on two streams so the decode CTAs backfill the SMs the prefill tail leaves
idle; the current stream waits for both before the attention op returns.

The CUDA work distributor dispatches the grid that becomes runnable first. If
both become runnable together (the usual case when the host runs ahead of the
GPU) the decode grid is dispatched first and the result is serial again, so
the order is forced:

* ``pp`` (default): the prefill runs on a high-priority side stream, the decode
  stays on the current stream and is launched without PDL (so it cannot become
  resident before the prefill is runnable).
* ``dly``: the prefill stays on the current stream; the decode runs on a side
  stream behind a short spin kernel (``torch.cuda._sleep``).

Numerics are unchanged (same kernels, same inputs, disjoint outputs); the
side-stream kernel gets a private trtllm workspace and the side decode a
private multi-CTA counter buffer, so no scratch is shared between concurrent
kernels.

Environment (read at import; default off):

``VLLM_ATTN_PD_OVERLAP``      1 enables the fork.
``VLLM_ATTN_PD_ORDER``        ``pp`` (default) or ``dly``.
``VLLM_ATTN_PD_MIN_DEC_ROWS`` fork only with at least this many decode rows (64).
``VLLM_ATTN_PD_SPIN_CYC``     spin cycles before the decode in ``dly`` (40000).
``VLLM_ATTN_PD_CHECK``        the first N forks per process are re-run serially
                              and compared bitwise; any mismatch keeps the serial
                              result and disables the fork.
``VLLM_ATTN_PD_LOG_EVERY``    log counts every N mixed attention calls (5000; 0 = off):
                              forks, calls below the decode-row minimum, and the
                              mean host cost of plan() when it does not fork.

Only eager calls fork (never during CUDA-graph capture); with PIECEWISE graphs
the attention op of mixed steps is eager, and FULL decode graphs have no
prefill rows.
"""

import os
import threading
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_ATTN_PD_OVERLAP", "0") == "1"
ORDER = os.environ.get("VLLM_ATTN_PD_ORDER", "pp").strip().lower()
MIN_DEC_ROWS = int(os.environ.get("VLLM_ATTN_PD_MIN_DEC_ROWS", "64"))
SPIN_CYC = int(os.environ.get("VLLM_ATTN_PD_SPIN_CYC", "40000"))
_CHECK = [int(os.environ.get("VLLM_ATTN_PD_CHECK", "0"))]
LOG_EVERY = int(os.environ.get("VLLM_ATTN_PD_LOG_EVERY", "5000"))
if ORDER not in ("pp", "dly"):
    raise ValueError(f"VLLM_ATTN_PD_ORDER must be pp or dly, got {ORDER!r}")

_TLS = threading.local()
_DEV: dict = {}  # device index -> per-device state
_STATS = {"calls": 0, "forks": 0, "below_min": 0, "capture": 0, "off_ns": 0, "fork_ns": 0,
          "checks": 0, "mismatch": 0, "disabled": False}
if ENABLED:
    logger.info(
        "attention decode||prefill overlap (attn-pdo) enabled: order=%s "
        "min_decode_rows=%d spin_cycles=%d check=%d",
        ORDER,
        MIN_DEC_ROWS,
        SPIN_CYC,
        _CHECK[0],
    )


class _DevState:
    def __init__(self, device: torch.device, ws_bytes: int):
        lo, hi = torch.cuda.Stream.priority_range()
        prio = hi if ORDER == "pp" else 0
        self.side = torch.cuda.Stream(device=device, priority=prio)
        self.ws = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
        self.counter = torch.zeros(1 << 20, dtype=torch.uint8, device=device)
        self.ev_in = torch.cuda.Event()
        self.ev_side = torch.cuda.Event()
        logger.info(
            "attn-pdo: side stream priority %d (range %s), private workspace "
            "%d B + counter buffer on %s",
            prio,
            (lo, hi),
            ws_bytes,
            device,
        )


class Fork:
    """One fork/join of one attention call. ``side_kind`` is the part that runs
    on the side stream: ``"p"`` (order pp) or ``"d"`` (order dly)."""

    __slots__ = ("st", "main", "side_kind", "_on_side", "_keep")

    def __init__(self, st: _DevState, main: torch.cuda.Stream):
        self.st = st
        self.main = main
        self.side_kind = "p" if ORDER == "pp" else "d"
        self._on_side = False
        self._keep: list = []

    def enter(self, part: str) -> None:
        if part != self.side_kind:
            return
        side = self.st.side
        side.wait_event(self.st.ev_in)
        torch.cuda.set_stream(side)
        self._on_side = True
        if part == "d" and SPIN_CYC > 0:
            torch.cuda._sleep(SPIN_CYC)

    def leave(self, part: str) -> None:
        if part != self.side_kind or not self._on_side:
            return
        self.st.ev_side.record(self.st.side)
        torch.cuda.set_stream(self.main)
        self._on_side = False

    def keep(self, *tensors) -> None:
        """Tensors allocated on the current stream and read on the side stream."""
        for t in tensors:
            if isinstance(t, torch.Tensor):
                t.record_stream(self.st.side)

    def workspace(self, part: str, ws: torch.Tensor) -> torch.Tensor:
        if part != self.side_kind:
            return ws
        if self.st.ws.numel() < ws.numel():
            raise RuntimeError("attn-pdo: private workspace smaller than the trtllm one")
        return self.st.ws

    def decode_kwargs(self) -> dict:
        if self.side_kind == "d":
            # side-stream decode: private multi-CTA counter buffer
            return {"multi_ctas_kv_counter_buffer": self.st.counter}
        # pp: the decode stays on the current stream; no PDL so it cannot be
        # resident before the (side-stream) prefill is runnable
        return {"enable_pdl": False}

    def join(self) -> None:
        if self._on_side:  # defensive: never leave the side stream current
            self.leave(self.side_kind)
        self.main.wait_event(self.st.ev_side)


def preallocate(device: torch.device) -> None:
    """Create the per-device state (side stream, private workspace + counter
    buffer) at model construction, i.e. before vLLM profiles memory for the KV
    cache, so the extra workspace is accounted for and cannot OOM later."""
    if not ENABLED:
        return
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _DEV:
        from vllm import envs

        _DEV[idx] = _DevState(device, envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE)


def _log_counts() -> None:
    c = _STATS
    off_n = max(1, c["calls"] - c["forks"])
    logger.info(
        "attn-pdo: mixed attention calls %d: forks %d (%.1f%%), below %d decode rows "
        "%d, under capture %d; plan() host cost: not forked %.0f ns/call, forked "
        "%.0f ns/call; checks ok %d, mismatches %d",
        c["calls"],
        c["forks"],
        100.0 * c["forks"] / max(1, c["calls"]),
        MIN_DEC_ROWS,
        c["below_min"],
        c["capture"],
        c["off_ns"] / off_n,
        c["fork_ns"] / max(1, c["forks"]),
        c["checks"],
        c["mismatch"],
    )


def plan(
    num_prefill_tokens: int, num_decode_tokens: int, device: torch.device
) -> "Fork | None":
    """Return a Fork (and record the fork point on the current stream) when this
    attention call should overlap its prefill and decode kernels."""
    if not ENABLED or num_prefill_tokens <= 0:
        return None
    t0 = time.perf_counter_ns()
    c = _STATS
    c["calls"] += 1
    if LOG_EVERY and c["calls"] % LOG_EVERY == 0:
        _log_counts()
    if (
        c["disabled"]
        or getattr(_TLS, "serial", False)
        or num_decode_tokens < MIN_DEC_ROWS
        or torch.cuda.is_current_stream_capturing()
    ):
        if num_decode_tokens < MIN_DEC_ROWS:
            c["below_min"] += 1
        elif not c["disabled"] and not getattr(_TLS, "serial", False):
            c["capture"] += 1
        if getattr(_TLS, "serial", False):
            c["calls"] -= 1  # the CHECK re-run is not a new call
        else:
            c["off_ns"] += time.perf_counter_ns() - t0
        return None
    idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _DEV.get(idx)
    if st is None:
        from vllm import envs

        st = _DEV[idx] = _DevState(device, envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE)
    main = torch.cuda.current_stream(device)
    st.ev_in.record(main)
    c["forks"] += 1
    c["fork_ns"] += time.perf_counter_ns() - t0
    return Fork(st, main)


def check_due() -> bool:
    return _CHECK[0] > 0 and not _STATS["disabled"]


class serial_scope:
    """Run the enclosed attention call(s) without forking (CHECK reference)."""

    def __enter__(self):
        _TLS.serial = True

    def __exit__(self, *exc):
        _TLS.serial = False


def check_result(fork_out: torch.Tensor, serial_out: torch.Tensor, layer_name: str) -> None:
    _CHECK[0] -= 1
    a = fork_out.contiguous().view(torch.uint8)
    b = serial_out.contiguous().view(torch.uint8)
    ok = bool(torch.equal(a, b))
    if ok:
        _STATS["checks"] += 1
        n = _STATS["checks"]
        if n == 1 or n % 100 == 0 or _CHECK[0] == 0:
            logger.info("attn-pdo: check ok #%d (%s)", n, layer_name)
        return
    _STATS["mismatch"] += 1
    _STATS["disabled"] = True
    bad = int((a != b).sum().item())
    logger.error(
        "attn-pdo: CHECK MISMATCH in %s (%d bytes differ); keeping the serial "
        "result and disabling the overlap",
        layer_name,
        bad,
    )
