# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ingest new requests only while the busy loop waits (``VLLM_INGEST_GATE``).

The EngineCore input thread decodes each ADD request and builds its
``Request`` (``EngineCore.preprocess_add_request``) while it holds the GIL. When
that happens while the main thread launches a step's eager forward, the main
thread waits for the GIL while the input thread builds the prompt list, the
FEH hash list and the Request.

With the gate on, the input thread still takes messages off its socket in
arrival order. It decodes and preprocesses an ADD only while the main thread is
blocked without the GIL:

- in ``AsyncOutput.get_output`` (``copy_event.synchronize()``), waiting for the
  previous step's GPU work;
- while the engine is idle (``_process_input_queue`` blocking on the queue);
- at the input-queue drain, where the main thread first waits until every ADD
  it has received is queued (``catch_up``).

Closing the gate waits for an ingest in progress. The two threads then take
turns instead of trading the GIL every switch interval.

Order and admission:

- The input thread still handles socket messages strictly in order, and queues
  them FIFO, as before.
- Every ADD the input thread has taken off its socket before a drain begins is
  queued before that drain, so it is admitted not later than the first drain
  after the input thread takes it. Without the gate, an ADD is admitted at the
  first drain after its ingest finishes, so one whose ingest would have
  straddled a drain is admitted one drain earlier with the gate.
- While the input thread waits at the gate with an ADD, it reads nothing else
  from its socket. An ADD that arrives meanwhile is taken only after that one
  is queued, so it can be admitted one or more drains later than without the
  gate.
- Messages behind a gated ADD (aborts, utility calls) wait with it, at most until
  the next drain, so an abort can never overtake its own ADD.

Environment (read once at import):

- ``VLLM_INGEST_GATE``: ``1`` enables it. The default ``0`` creates no gate and
  changes nothing.
- ``VLLM_INGEST_GATE_LOG_EVERY``: logs a stats line every N ADD requests
  (default 2000).
- ``VLLM_INGEST_GATE_TIMEOUT_S``: the safety bound on any one main-thread wait
  (default 30). If it expires, the gate turns itself off (stock behaviour) and
  logs a warning.
"""

import os
import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_INGEST_GATE", "0") == "1"
LOG_EVERY = int(os.environ.get("VLLM_INGEST_GATE_LOG_EVERY", "2000") or 2000)
TIMEOUT_S = float(os.environ.get("VLLM_INGEST_GATE_TIMEOUT_S", "30") or 30)

# The process' gate (EngineCoreProc with the switch on), else None. Read by
# AsyncOutput.get_output on the main thread.
GATE: "IngestGate | None" = None


class IngestGate:
    """Lets the input thread ingest an ADD only while the gate is open."""

    def __init__(self) -> None:
        self._cv = threading.Condition(threading.Lock())
        self._open = True  # open until the busy loop starts stepping
        self._where = "init"
        self._busy = False
        self._received = 0
        self._done = 0
        self._off = False
        self.stats = {
            "adds": 0,
            "gated": 0,
            "gated_ms": 0.0,
            "init": 0,
            "wait": 0,
            "idle": 0,
            "drain": 0,
            "catch_ups": 0,
            "catch_up_ms": 0.0,
            "spills": 0,
            "spill_ms": 0.0,
        }

    # ------------------------------------------------------- input thread
    def enter(self) -> None:
        """Before decoding an ADD: block until the main thread waits."""
        with self._cv:
            self._received += 1
            if not self._open and not self._off:
                t0 = time.perf_counter()
                self.stats["gated"] += 1
                while not self._open and not self._off:
                    self._cv.wait()
                self.stats["gated_ms"] += (time.perf_counter() - t0) * 1e3
            self._busy = True
            self.stats[self._where] += 1

    def leave(self) -> None:
        """After the ADD is queued (or rejected)."""
        with self._cv:
            self._busy = False
            self._done += 1
            self.stats["adds"] += 1
            n = self.stats["adds"]
            self._cv.notify_all()
        if n == 1:
            logger.info(
                "Ingest gate on (VLLM_INGEST_GATE=1): EngineCore decodes and "
                "preprocesses ADD requests only while the main thread waits "
                "(get_output, idle, input drain)"
            )
        if n % LOG_EVERY == 0:
            self.log_stats()

    # -------------------------------------------------------- main thread
    def open(self, where: str) -> None:
        """The main thread is about to block without the GIL."""
        with self._cv:
            self._open = True
            self._where = where
            self._cv.notify_all()

    def close(self) -> None:
        """The main thread resumes: no new ingest starts; one in progress is
        waited for (serially, instead of trading the GIL)."""
        with self._cv:
            self._open = False
            if self._busy and not self._off:
                t0 = time.perf_counter()
                self.stats["spills"] += 1
                self._wait_idle()
                self.stats["spill_ms"] += (time.perf_counter() - t0) * 1e3

    def catch_up(self) -> None:
        """Before the main thread drains the input queue: every ADD received
        so far is queued first; then the gate is closed."""
        with self._cv:
            target = self._received
            if self._done < target and not self._off:
                t0 = time.perf_counter()
                self.stats["catch_ups"] += 1
                self._open = True
                self._where = "drain"
                self._cv.notify_all()
                while self._done < target and not self._off:
                    if not self._cv.wait(timeout=TIMEOUT_S):
                        self._turn_off("catch-up")
                self.stats["catch_up_ms"] += (time.perf_counter() - t0) * 1e3
            self._open = False
            if self._busy and not self._off:
                # An ADD received after the snapshot started meanwhile.
                self._wait_idle()

    def _wait_idle(self) -> None:
        while self._busy and not self._off:
            if not self._cv.wait(timeout=TIMEOUT_S):
                self._turn_off("ingest in progress")

    def _turn_off(self, what: str) -> None:
        self._off = True
        self._open = True
        self._cv.notify_all()
        logger.warning(
            "Ingest gate: main thread waited %.0f s (%s); gate off, ADD "
            "ingest runs unconditionally from now on",
            TIMEOUT_S,
            what,
        )

    def log_stats(self) -> None:
        s = dict(self.stats)
        n = max(1, s["adds"])
        logger.info(
            "Ingest gate stats: adds=%d started in wait=%d idle=%d drain=%d "
            "init=%d | gated=%d (mean %.3f ms) | main-thread waits: catch-up "
            "%d (%.3f ms per ADD) spill %d (%.3f ms per ADD)%s",
            s["adds"],
            s["wait"],
            s["idle"],
            s["drain"],
            s["init"],
            s["gated"],
            s["gated_ms"] / max(1, s["gated"]),
            s["catch_ups"],
            s["catch_up_ms"] / n,
            s["spills"],
            s["spill_ms"] / n,
            " | OFF (timeout)" if self._off else "",
        )


def create() -> "IngestGate | None":
    """EngineCoreProc: the process' gate when the switch is on, else None."""
    global GATE
    if ENABLED and GATE is None:
        GATE = IngestGate()
    return GATE
