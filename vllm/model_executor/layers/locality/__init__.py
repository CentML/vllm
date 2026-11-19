# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locality-domain infrastructure for SM107 (sm_107).

- ``topology.get_topology(device)``: domain count and the SM -> domain table
  (green contexts + latency probe for unassigned TPCs); None if the topology
  is unsupported.
- ``memory.localize(t, layout)`` / ``memory.alloc_chunks``: tensors whose
  allocation chunks live on chosen domains (``cuMemCreate`` with
  ``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN``), placement verified with
  ``CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL``.
- ``skinny.DomainGemm``: one-launch, graph-safe, %smid-routed skinny GEMM
  that reads domain-local BF16 weight bytes.
- ``mxgemm.DomainMxGemm``: the MXFP8 (tcgen05 block-scaled, SM107) domain-local
  skinny GEMM of the draft lm_head.
- ``streams.get_domain_streams(dev)``: persistent green contexts and streams,
  with optional BACKFILL requests for remainder SMs and binding diagnostics,
  plus ``execute_on_streams`` fork/join. Inert until a consumer calls it.
- ``gate.locality_active(dev)``: the HBM-clock gate, which rejects clocks below
  ``VLLM_LOCALITY_MIN_MEMCLK`` without enabling consumers on its own.
- ``lm_head``: the ``VLLM_LOCALITY_LM_HEAD`` integration (vLLM-specific).

The modules above ``lm_head`` import only torch, so microbenchmarks can load
the package by file path.
"""

from .gate import locality_active
from .memory import Localized, alloc_chunks, localize, row_group_tiles
from .skinny import DomainGemm
from .streams import DomainStreams, execute_on_streams, get_domain_streams
from .topology import Topology, get_topology

__all__ = [
    "DomainGemm",
    "DomainStreams",
    "Localized",
    "Topology",
    "alloc_chunks",
    "execute_on_streams",
    "get_domain_streams",
    "get_topology",
    "localize",
    "locality_active",
    "row_group_tiles",
]
