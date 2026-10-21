# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locality-domain ("micro-GPU") infrastructure for Rubin-class GPUs.

- ``topology.get_topology(device)``: domain count and the SM -> domain table
  (green contexts + latency probe for the unassigned TPCs); None if the GPU
  has fewer than two locality domains.
- ``memory.localize(t, layout)`` / ``memory.alloc_chunks``: tensors whose
  2 MiB chunks live on chosen domains (``cuMemCreate`` with
  ``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN``), placement verified with
  ``CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL``.
- ``skinny.DomainGemm``: one-launch, graph-safe, %smid-routed skinny GEMM
  (BF16, M <= 32) that reads only domain-local weight bytes.
- ``mxgemm.DomainMxGemm``: the MXFP8 (tcgen05 block-scaled, SM107) domain-local
  skinny GEMM of the draft lm_head.
- ``_ext.load().green_streams(dev)``: a persistent green context + stream per
  domain (100 + 100 SMs on VR200) for fork/join dispatch.
- ``lm_head``: the ``VLLM_LOCALITY_LM_HEAD`` integration (vLLM-specific).

The modules above ``lm_head`` import only torch, so microbenchmarks can load
the package by file path.
"""

from .memory import Localized, alloc_chunks, localize, row_group_tiles
from .skinny import DomainGemm
from .topology import Topology, get_topology

__all__ = [
    "DomainGemm",
    "Localized",
    "Topology",
    "alloc_chunks",
    "get_topology",
    "localize",
    "row_group_tiles",
]
