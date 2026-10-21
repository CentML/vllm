# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensors whose bytes live on chosen locality domains.

``alloc_chunks(nbytes, chunk_domains, chunk_bytes)`` reserves one virtual range
and backs chunk i with a ``cuMemCreate`` allocation on locality domain
``chunk_domains[i]`` (``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN``). The
result is a flat uint8 torch tensor (``torch.from_blob``); it can be viewed as
any dtype/shape, sliced and passed to any kernel (cuBLAS, FlashInfer, ...).
The physical memory is freed when the last tensor view dies (the deleter
synchronizes the device first).

Granularity: chunks are multiples of the 2 MiB allocation granularity; the
range is padded up to whole chunks. Layouts (``plan``):

- ``"interleave"``: chunk i on domain i % 2 (2 MiB stripes). Domain-aware
  kernels read their own stripes; ordinary kernels see a coarse 50/50
  interleave, close to what ``cudaMalloc``'s 4 KB hashing gives them.
- ``"halves"``: the first half of the chunks on domain 0, the rest on domain 1.
  Domain-aware kernels are equally happy, but an ordinary kernel whose CTAs
  walk the rows in order streams from one die at a time (slow).

Memory accounting: vLLM sizes the KV cache from ``cudaMemGetInfo`` deltas
(``memory_profiling``: total_consumed = free before init - free after
profiling), so these allocations are counted like any other non-torch memory
as long as they are made before the profiling run (at weight loading). The
source tensor they replace must be dropped so the caching allocator can return
it (``torch.cuda.empty_cache`` runs inside ``memory_profiling``). The padding
to whole 2 MiB chunks is the only overhead.

``localize(t, layout)`` copies a tensor into such a range and returns a view
with the same shape and dtype plus the per-chunk domain ordinals read back
with ``CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL`` (the placement check).
``row_group_tiles`` turns the ordinals into the per-domain work lists of the
domain-aware kernels.
"""

from __future__ import annotations

import dataclasses

import torch

from . import _ext

MIB = 1 << 20


def alloc_chunks(
    nbytes: int, chunk_domains: list[int], chunk_bytes: int, device: int
) -> torch.Tensor:
    return _ext.load().alloc_localized(int(nbytes), list(chunk_domains), int(chunk_bytes), int(device))


def granularity(device: int) -> int:
    return int(_ext.load().granularity(device))


def plan(nbytes: int, layout: str, chunk_bytes: int) -> list[int]:
    n = -(-nbytes // chunk_bytes)
    if layout == "interleave":
        return [i % 2 for i in range(n)]
    if layout == "halves":
        h = -(-n // 2)
        return [0 if i < h else 1 for i in range(n)]
    if layout in ("dom0", "dom1"):
        return [int(layout[-1])] * n
    raise ValueError(f"unknown layout {layout!r}")


def chunk_ordinals(t: torch.Tensor, nbytes: int, chunk_bytes: int) -> list[int]:
    return [int(v) for v in _ext.load().chunk_ordinals(t.data_ptr(), int(nbytes), int(chunk_bytes))]


@dataclasses.dataclass
class Localized:
    tensor: torch.Tensor  # typed view of the source shape
    storage: torch.Tensor  # flat uint8 backing (keeps the mapping alive)
    chunk_bytes: int
    ordinals: list[int]  # verified domain of every chunk

    @property
    def padded_bytes(self) -> int:
        return self.storage.numel()


def localize(
    t: torch.Tensor, layout: str = "interleave", chunk_bytes: int | None = None
) -> Localized:
    """Copy a contiguous CUDA tensor into domain-placed memory (``layout``)."""
    assert t.is_cuda and t.is_contiguous()
    dev = t.get_device()
    gran = granularity(dev)
    chunk_bytes = chunk_bytes or gran
    assert chunk_bytes % gran == 0
    nbytes = t.numel() * t.element_size()
    doms = plan(nbytes, layout, chunk_bytes)
    with torch.cuda.device(dev):
        flat = alloc_chunks(nbytes, doms, chunk_bytes, dev)
        view = flat[:nbytes].view(t.dtype).view(t.shape)
        view.copy_(t)
        torch.cuda.synchronize(dev)
    ords = chunk_ordinals(flat, flat.numel(), chunk_bytes)
    if ords != doms:
        raise RuntimeError(
            f"locality placement check failed: wanted {doms[:8]}..., driver reports {ords[:8]}..."
        )
    return Localized(tensor=view, storage=flat, chunk_bytes=chunk_bytes, ordinals=ords)


def row_group_tiles(
    base: torch.Tensor, rows: int, row_bytes: int, ordinals: list[int] | None,
    chunk_bytes: int, group_rows: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-domain int32 lists of ``group_rows``-row groups of a row-major matrix
    starting at ``base``. ``ordinals`` (per chunk) None: memory is interleaved
    (cudaMalloc); the groups are then split alternately (control arm)."""
    assert rows % group_rows == 0
    ng = rows // group_rows
    gb = group_rows * row_bytes
    lists: list[list[int]] = [[], []]
    for gi in range(ng):
        if ordinals is None:
            d = gi & 1
        else:
            off = gi * gb
            c0, c1 = off // chunk_bytes, (off + gb - 1) // chunk_bytes
            assert ordinals[c0] == ordinals[c1], "a row group must not straddle domains"
            d = ordinals[c0]
        lists[d].append(gi)
    dev = base.device
    return (
        torch.tensor(lists[0], dtype=torch.int32, device=dev),
        torch.tensor(lists[1], dtype=torch.int32, device=dev),
    )
