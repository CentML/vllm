# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy helpers for NVIDIA Confidential Computing (CC).

Under bounce-buffer CC a host<->device ``cudaMemcpyAsync`` is forced
host-SYNCHRONOUS: the issuing thread blocks until the copy -- and everything
already queued on its stream -- completes. Two consequences for the engine:

* An H2D issued on the compute stream (which has the in-flight forward
  queued) blocks the scheduler for ~one forward per copy, starving the GPU.
* The per-step D2H token readback blocks the thread that issues it, delaying
  the next step's CUDA graph launch by ~one decode step.

Two mitigations, both no-ops off CC:

* Staged H2D (``staged_h2d_enabled`` / ``staged_h2d_stream``): issue the H2D
  into a device staging buffer on an idle prep stream, so it only pays its own
  transfer (~tens of us) instead of the forward drain; then a D2D
  staging->dst on the compute stream. The D2D is genuinely async under CC and
  is ordered after the forward's read of the reused graph-input buffer, so
  there is no host block and no write-after-read race. Correctness relies on
  the pinned H2D being host-synchronous under CC: staging is fully populated
  by the time the copy call returns, so the D2D enqueued on another stream
  reads valid data without a cross-stream event. Callers double-buffer the
  staging tensor so the next step's H2D cannot overwrite a staging buffer
  whose D2D has not yet drained. A pool of prep streams is used round-robin
  so consecutive copies in one prepare_inputs are not queued behind each
  other either.

* ``AsyncD2HCopyWorker``: run the (still-blocking) result readback on a
  dedicated daemon thread so the scheduler keeps issuing work (mirrors
  TensorRT-LLM PR #8463). The copy stays synchronous under CC; it is merely
  non-blocking *to the scheduler thread*, restoring overlap.
"""

import queue
import threading
from collections.abc import Callable

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_STAGED_H2D_STREAM_POOL: dict[int, list[torch.cuda.Stream]] = {}
_STAGED_H2D_POOL_RR: dict[int, int] = {}
_STAGED_H2D_POOL_SIZE = 16
_STAGED_H2D_ENABLED: bool | None = None


def staged_h2d_enabled() -> bool:
    """Whether H2D input copies should take the CC staged path."""
    global _STAGED_H2D_ENABLED
    if _STAGED_H2D_ENABLED is None:
        import os

        if os.getenv("VLLM_STAGED_H2D", "1") == "0":
            _STAGED_H2D_ENABLED = False
        else:
            try:
                from vllm.platforms import current_platform

                _STAGED_H2D_ENABLED = bool(current_platform.is_confidential_compute())
                if _STAGED_H2D_ENABLED:
                    logger.info(
                        "Staged H2D input copies ENABLED under Confidential "
                        "Computing (H2D on a dedicated prep stream + D2D on the "
                        "compute stream) to avoid blocking the scheduler on the "
                        "in-flight forward."
                    )
            except Exception:
                _STAGED_H2D_ENABLED = False
    return _STAGED_H2D_ENABLED


def staged_h2d_stream(device: torch.device) -> torch.cuda.Stream:
    """Return the next idle prep stream (round-robin) for this device."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    pool = _STAGED_H2D_STREAM_POOL.get(idx)
    if pool is None:
        pool = [
            torch.cuda.Stream(device=device) for _ in range(_STAGED_H2D_POOL_SIZE)
        ]
        _STAGED_H2D_STREAM_POOL[idx] = pool
        _STAGED_H2D_POOL_RR[idx] = 0
    i = _STAGED_H2D_POOL_RR[idx]
    _STAGED_H2D_POOL_RR[idx] = (i + 1) % len(pool)
    return pool[i]


class StagedH2DCopier:
    """Double-buffered staged H2D into a persistent GPU tensor.

    One instance per destination tensor. Callers must only use this when
    ``staged_h2d_enabled()`` is true; off CC the cross-stream handoff would
    race (see module docstring).
    """

    def __init__(self, gpu_base: torch.Tensor):
        self._gpu = gpu_base
        # Staging is mutable runtime state, not inference data.
        with torch.inference_mode(False):
            self._stage = [torch.empty_like(gpu_base) for _ in range(2)]
        self._idx = 0

    def copy_(self, cpu_base: torch.Tensor, n: int | None = None) -> torch.Tensor:
        """Copy ``cpu_base[:n]`` into the GPU tensor via the staged path."""
        gpu_dst = self._gpu if n is None else self._gpu[:n]
        cpu_src = cpu_base if n is None else cpu_base[:n]
        stage = self._stage[self._idx]
        self._idx ^= 1
        stage_dst = stage if n is None else stage[:n]
        # H2D on the prep stream is host-synchronous under CC, so stage_dst is
        # populated on return; the D2D on the current (compute) stream is async
        # and ordered after the forward's read of the reused buffer.
        with torch.cuda.stream(staged_h2d_stream(self._gpu.device)):
            stage_dst.copy_(cpu_src, non_blocking=True)
        return gpu_dst.copy_(stage_dst, non_blocking=True)


class AsyncD2HCopyWorker:
    """Runs the per-step result D2H readback on a dedicated daemon thread.

    The scheduler thread records a CUDA event after the producing work and
    hands ``(event, copy_fn, done)`` here, then returns immediately. This
    worker ``cudaEventSynchronize``-s on that event (event-sync, NOT
    stream-wait, so the blocking copy does not stall the scheduler's CUDA API
    calls), runs the copy on the dedicated copy stream, blocks until it
    completes, and sets ``done``. The scheduler later waits on ``done``
    (worker-done => copy-done). See the module docstring for the CC rationale.
    """

    def __init__(self, device_module, copy_stream, device=None):
        self.device_module = device_module
        self.copy_stream = copy_stream
        self._device = device
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name="vllm-d2h-copy-worker", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        src_ready: torch.cuda.Event,
        copy_fn: Callable[[], None],
        done: threading.Event,
    ):
        """Enqueue a readback. ``src_ready`` must already be recorded on the
        stream that produces the copy sources; ``copy_fn`` performs the actual
        ``.to("cpu", ...)`` copies; ``done`` is set when they have completed."""
        self._queue.put((src_ready, copy_fn, done))

    def _loop(self):
        # A new thread does not inherit the main thread's CUDA context; set the
        # device so CUDA runtime calls do not implicitly create one on device 0.
        if self._device is not None:
            from vllm.platforms import current_platform

            current_platform.set_device(self._device)
        while True:
            item = self._queue.get()
            if item is None:
                return
            src_ready, copy_fn, done = item
            try:
                # Wait until the producing forward+sample has materialized the
                # source tensors, then issue the copies on a dedicated stream
                # owned by this thread (the current stream is thread-local, so
                # this does not affect the scheduler thread's stream).
                src_ready.synchronize()
                with self.device_module.stream(self.copy_stream):
                    copy_fn()
                self.copy_stream.synchronize()
            except Exception:
                logger.exception("AsyncD2HCopyWorker readback failed")
            finally:
                done.set()

    def shutdown(self, timeout: float = 2.0):
        """Signal the worker to stop and join it (best-effort, bounded wait)."""
        if not self._thread.is_alive():
            return
        self._queue.put(None)
        self._thread.join(timeout=timeout)
