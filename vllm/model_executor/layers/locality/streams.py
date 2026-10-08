# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-die green streams and fork/join for locality-domain consumers.

``get_domain_streams(device)`` returns a non-blocking stream per supported
locality domain, each on a persistent green context, or None when the topology
is unsupported or the HBM-clock gate in ``gate.py`` rejects the device.
With ``backfill=True`` (default), each context requests its domain's SMs plus
the remainder ("unassigned") SMs that the topology probe places on that die,
via ``CU_DEV_SM_RESOURCE_GROUP_BACKFILL``. The driver chooses the actual
binding; it is checked and reported rather than guaranteed by this request.
Without backfill, remainder SMs are not included in the domain streams.

BACKFILL picks remainder SMs itself (cuda.h has no per-SM choice); the
returned ``binding`` reports, per stream, whether every domain SM was reached,
whether any SM of the other domain was, and how many of the remainder SMs it
got sit on its own die (``orphans_right_die``) or the other die
(``orphans_wrong_die``; those still run, but read their "local" buffer across
the die-to-die link).

``execute_on_streams(streams, fn)`` runs ``fn(i)`` with ``streams[i]`` current:
each stream first waits on the caller stream, and the caller then waits on
every stream that was entered, also when ``fn`` raises. Events only, no host
sync, so it is CUDA-graph capturable. Size each domain's grid from
``sm_counts[i]``, not from the device SM count.

Call ``get_domain_streams`` once at start-up, outside graph capture (the
binding check launches a kernel and synchronizes). Inert unless a consumer
calls it.

Imports only torch (microbenchmarks load the package by file path).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence

import torch

from . import _ext
from .gate import locality_active
from .topology import Topology, get_topology

logger = logging.getLogger(__name__)


def execute_on_streams(streams: Sequence[torch.cuda.Stream], fn: Callable[[int], None]) -> None:
    """Fork/join: run fn(i) on streams[i]; the caller stream waits for all of them."""
    if not streams:
        raise ValueError("execute_on_streams needs at least one stream")
    caller = torch.cuda.current_stream()
    start = torch.cuda.Event()
    start.record(caller)
    entered: list[torch.cuda.Stream] = []
    try:
        for i, s in enumerate(streams):
            s.wait_event(start)
            entered.append(s)
            with torch.cuda.stream(s):
                fn(i)
    finally:
        for s in entered:
            done = torch.cuda.Event()
            done.record(s)
            caller.wait_event(done)


@dataclasses.dataclass(frozen=True)
class DomainStreams:
    device: int
    streams: tuple[torch.cuda.Stream, torch.cuda.Stream]
    sm_counts: tuple[int, int]  # SMs in each stream's green context
    remainder_sms: int  # SMs left outside the contexts, as reported by the driver
    backfill: bool
    extra: tuple[int, int]  # remainder SMs requested per die
    coscheduled: int  # coscheduledSmCount accepted by the driver
    binding: tuple[dict, dict]

    def run(self, fn: Callable[[int], None]) -> None:
        execute_on_streams(self.streams, fn)

    def summary(self) -> dict:
        return dict(device=self.device, sm_counts=list(self.sm_counts), remainder_sms=self.remainder_sms,
                    backfill=self.backfill, extra=list(self.extra), coscheduled=self.coscheduled,
                    binding=list(self.binding))


def remainder_per_die(topo: Topology) -> tuple[int, int]:
    """Count remainder SMs by the die assigned by the topology probe."""
    dom = topo.sm_domain.cpu().tolist()
    n = [0, 0]
    for s, g in enumerate(topo.green_map):
        if g < 0:
            n[int(dom[s])] += 1
    return n[0], n[1]


def check_binding(topo: Topology, stream: torch.cuda.Stream, k: int) -> dict:
    """Which SMs a kernel on ``stream`` reaches, against the topology of domain ``k``."""
    seen = _ext.load().stream_smids(topo.device, int(stream.cuda_stream)).tolist()
    dom = topo.sm_domain.cpu().tolist()
    g = topo.green_map
    exp = [s for s in range(topo.num_sms) if g[s] == k]
    return dict(
        domain=k,
        seen=sum(seen),
        domain_sms_expected=len(exp),
        domain_sms_seen=sum(seen[s] for s in exp),
        foreign_domain_sms=sum(seen[s] for s in range(topo.num_sms) if g[s] == 1 - k),
        orphans_right_die=sum(seen[s] for s in range(topo.num_sms) if g[s] < 0 and dom[s] == k),
        orphans_wrong_die=sum(seen[s] for s in range(topo.num_sms) if g[s] < 0 and dom[s] != k),
    )


_cache: dict[tuple[int, bool], DomainStreams | None] = {}


def get_domain_streams(device: int | None = None, backfill: bool = True) -> DomainStreams | None:
    if device is None:
        device = torch.cuda.current_device()
    key = (device, backfill)
    if key in _cache:
        return _cache[key]
    ds = None
    if locality_active(device):
        topo = get_topology(device)
        if topo is None:
            logger.info("Locality streams: device %d has no 2 locality domains.", device)
        else:
            extra = remainder_per_die(topo) if backfill else (0, 0)
            v = _ext.load().green_streams(device, extra[0], extra[1])
            streams = tuple(torch.cuda.ExternalStream(int(v[k]), device=f"cuda:{device}") for k in range(2))
            binding = tuple(check_binding(topo, streams[k], k) for k in range(2))
            ds = DomainStreams(device=device, streams=streams, sm_counts=(int(v[2]), int(v[3])),
                               remainder_sms=int(v[4]), backfill=backfill, extra=extra, coscheduled=int(v[5]),
                               binding=binding)
            bad = [b for b in binding if b["domain_sms_seen"] != b["domain_sms_expected"] or b["foreign_domain_sms"]]
            wrong = sum(b["orphans_wrong_die"] for b in binding)
            (logger.warning if bad or wrong else logger.info)(
                "Locality streams: %s%s", ds.summary(),
                " (BINDING MISMATCH)" if bad else (" (remainder SMs on the wrong die)" if wrong else ""))
    _cache[key] = ds
    return ds
