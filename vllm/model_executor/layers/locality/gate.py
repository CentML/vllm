# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HBM-clock gate for locality-domain features.

``VLLM_LOCALITY_MIN_MEMCLK`` sets the minimum current memory clock required
for locality consumers. Below the configured threshold, locality stays off
and the decision is logged once per device. A nonpositive threshold disables
the gate. If the clock cannot be read, locality remains allowed and a warning
is logged. This gate does not itself enable a locality consumer.

The clock is read at the call (engine start-up). It is the current memory
clock, so a board whose clocks are pinned / locked reads its pinned value; a
board left on the default policy may read its idle clock and keep locality off.

Imports only torch (microbenchmarks load the package by file path).
"""

from __future__ import annotations

import ctypes
import logging
import os

import torch

logger = logging.getLogger(__name__)

DEFAULT_MIN_MEMCLK = 3600
_decided: dict[int, bool] = {}


def min_memclk() -> int:
    return int(os.environ.get("VLLM_LOCALITY_MIN_MEMCLK", str(DEFAULT_MIN_MEMCLK)))


def memclk_mhz(dev: int) -> int | None:
    """Current HBM clock of torch device ``dev`` (MHz) via NVML, or None."""
    try:
        nv = ctypes.CDLL("libnvidia-ml.so.1")
        if nv.nvmlInit_v2() != 0:
            return None
        h = ctypes.c_void_p()
        p = torch.cuda.get_device_properties(dev)
        rc = 1
        if isinstance(getattr(p, "pci_bus_id", None), int):
            # match by PCI address: correct under any CUDA_VISIBLE_DEVICES
            busid = f"{p.pci_domain_id:08x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
            rc = nv.nvmlDeviceGetHandleByPciBusId_v2(busid.encode(), ctypes.byref(h))
        if rc != 0:  # index fallback (correct when CUDA_VISIBLE_DEVICES is unset)
            rc = nv.nvmlDeviceGetHandleByIndex_v2(int(dev), ctypes.byref(h))
        if rc != 0:
            return None
        clk = ctypes.c_uint()
        if nv.nvmlDeviceGetClockInfo(h, 2, ctypes.byref(clk)) != 0:  # NVML_CLOCK_MEM
            return None
        return int(clk.value)
    except Exception:  # noqa: BLE001  no NVML in this environment: clock unknown
        return None


def locality_active(dev: int | None = None, refresh: bool = False) -> bool:
    """True if locality features may run on ``dev`` (HBM clock >= the gate).
    Decided once per device (logged); ``refresh`` re-reads the clock."""
    if dev is None:
        dev = torch.cuda.current_device()
    if dev in _decided and not refresh:
        return _decided[dev]
    gate = min_memclk()
    clk = memclk_mhz(dev)
    if gate <= 0:
        ok = True
        logger.info("Locality HBM-clock gate disabled (VLLM_LOCALITY_MIN_MEMCLK=%d); memclk %s MHz.", gate, clk)
    elif clk is None:
        ok = True
        logger.warning("Locality HBM-clock gate: clock unreadable (no NVML); locality left ON.")
    elif clk < gate:
        ok = False
        logger.warning(
            "Locality INACTIVE on device %d: HBM clock %d MHz < %d MHz (VLLM_LOCALITY_MIN_MEMCLK); "
            "domain-local reads are limited at low HBM clock.", dev, clk, gate)
    else:
        ok = True
        logger.info("Locality active on device %d: HBM clock %d MHz >= %d MHz.", dev, clk, gate)
    _decided[dev] = ok
    return ok
