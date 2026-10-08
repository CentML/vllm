# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locality-domain topology: domain count and the SM -> domain table.

``get_topology(device)`` (cached per device) returns ``None`` on GPUs without
two locality domains. Otherwise the table is built from two independent
sources and cross-checked:

1. green contexts: ``cuDevSmResourceSplit`` with
   ``CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID`` gives each domain's SMs
   (100 + 100 on VR200); a ``%smid`` kernel run in each green context lists
   them. The SMs left in the remainder (12 on VR200, the "unassigned TPCs")
   get -1 there.
2. a latency probe: every SM chases a pointer chain (L2-resident) in a buffer
   placed on domain 0 and in one placed on domain 1; the near buffer is
   ~1.3-1.7x faster. This places the remainder SMs on their physical die.

``sm_domain`` (int8, on the device) is the green-context domain where there is
one and the probe's domain otherwise, so every SM has a domain. Kernels index
it with ``%smid``.
"""

from __future__ import annotations

import dataclasses
import functools

import torch

from . import _ext


@dataclasses.dataclass(frozen=True)
class Topology:
    device: int
    num_sms: int
    num_domains: int
    sms_per_domain: int  # driver attribute 157
    sm_domain: torch.Tensor  # int8 [num_sms] on `device`; every SM assigned
    green_map: tuple[int, ...]  # -1: SM not in a domain's green context
    probe_map: tuple[int, ...]  # -1: SM not reached by the probe
    probe_cycles: tuple[tuple[float, float], ...]  # (dom0, dom1) cycles/load
    domain_sms: tuple[int, int]  # SM count per domain in sm_domain
    probe_agree: int  # SMs where green map and probe agree
    probe_disagree: int

    def summary(self) -> dict:
        rem = [s for s, d in enumerate(self.green_map) if d < 0]
        return dict(
            device=self.device,
            num_sms=self.num_sms,
            num_domains=self.num_domains,
            sms_per_domain=self.sms_per_domain,
            green_counts=[self.green_map.count(0), self.green_map.count(1)],
            remainder_sms=rem,
            remainder_domains=[self.probe_map[s] for s in rem],
            domain_sms=list(self.domain_sms),
            probe_agree=self.probe_agree,
            probe_disagree=self.probe_disagree,
        )


def _probe(device: int) -> tuple[list[int], list[tuple[float, float]]]:
    from .memory import alloc_chunks

    gran = _ext.load().granularity(device)
    b0 = alloc_chunks(gran, [0], gran, device)
    b1 = alloc_chunks(gran, [1], gran, device)
    with torch.cuda.device(device):
        cyc = _ext.load().probe_sm_latency(b0, b1)
        torch.cuda.synchronize(device)
    cyc = cyc.cpu().tolist()
    del b0, b1
    pmap = []
    for c0, c1 in cyc:
        if c0 <= 0 or c1 <= 0 or max(c0, c1) / min(c0, c1) < 1.1:
            pmap.append(-1)  # SM not reached, or no clear near/far split
        else:
            pmap.append(0 if c0 < c1 else 1)
    return pmap, [tuple(c) for c in cyc]


@functools.cache
def get_topology(device: int | None = None) -> Topology | None:
    if device is None:
        device = torch.cuda.current_device()
    ext = _ext.load()
    nsm, nld, ldsm = ext.device_info(device)
    if nld != 2:
        return None
    green = [int(v) for v in ext.green_sm_map(device).tolist()]
    pmap, cyc = _probe(device)
    agree = sum(1 for g, p in zip(green, pmap) if g >= 0 and g == p)
    disagree = sum(1 for g, p in zip(green, pmap) if g >= 0 and p >= 0 and g != p)
    final = []
    for s in range(nsm):
        d = green[s] if green[s] >= 0 else pmap[s]
        final.append(d if d >= 0 else s & 1)  # neither source: any domain is correct, only slower
    t = torch.tensor(final, dtype=torch.int8, device=f"cuda:{device}")
    return Topology(
        device=device,
        num_sms=nsm,
        num_domains=nld,
        sms_per_domain=ldsm,
        sm_domain=t,
        green_map=tuple(green),
        probe_map=tuple(pmap),
        probe_cycles=tuple(cyc),
        domain_sms=(final.count(0), final.count(1)),
        probe_agree=agree,
        probe_disagree=disagree,
    )
