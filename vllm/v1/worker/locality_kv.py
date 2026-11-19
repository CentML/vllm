# SPDX-License-Identifier: Apache-2.0
"""Opt-in locality-domain placement of shared KV/state backing allocations.

Each mapping chunk follows the block owning its first byte. The plan accounts
for both layer-outermost and block-outermost layouts, including aliased hybrid
cache groups. A block crossing a mapping boundary remains valid but may access
remote memory. Mappings live for the process and do not support sleep mode.
Unavailable topology or unverified placement falls back to ordinary allocation.
"""

import dataclasses
import os

import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_EXT = None
GRAN = 2 << 20  # CUDA VMM granularity, verified before allocation.
# Kernels use the boundary only after topology, clock and placement checks.
SPLIT_ACTIVE = False
_PPD_BLOCK = 0    # first manager block >= N/2 lying fully on domain 1 in every layer slot (block granular)
_NUM_BLOCKS = 0
_WARNED: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(msg, *args)


def pages_per_domain(num_kernel_pages: int, kernel_pages_per_block: int | None = None) -> int:
    """Kernel-page index boundary between domain 0 and 1 (compare with page_table[b][0] >= ppd); 0 when the split
    is not active. `num_kernel_pages` is the kernel view's size(0) (FP8 K view or MXK page-flat view); the shim
    derives kernel pages per manager block from it. Host int, no device work: safe on every launch and inside
    CUDA-graph capture."""
    if not SPLIT_ACTIVE:
        return 0
    kpb = kernel_pages_per_block or (num_kernel_pages // _NUM_BLOCKS if _NUM_BLOCKS else 0)
    if kpb <= 0 or kpb * _NUM_BLOCKS != num_kernel_pages:
        _warn_once(f"ppd{num_kernel_pages}", "[dp2g] pages_per_domain(%d): not a multiple of %d manager blocks; "
                   "locality off for this view", num_kernel_pages, _NUM_BLOCKS)
        return 0
    return _PPD_BLOCK * kpb


_CPP = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <string>
static void ck(CUresult r, const char* w) {
  if (r != CUDA_SUCCESS) { const char* s = nullptr; cuGetErrorString(r, &s); throw std::runtime_error(std::string(w) + ": " + (s ? s : "?")); }
}
int64_t domain_count(int64_t dev) {
  int n = 1;
  if (cudaDeviceGetAttribute(&n, cudaDevAttrLocalityDomainCount, (int)dev) != cudaSuccess) { cudaGetLastError(); return 1; }
  return n;
}
int64_t granularity(int64_t dev) {
  CUmemAllocationProp p; memset(&p, 0, sizeof p); p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN; p.location.localized.deviceId = (int)dev;
  size_t g = 0; ck(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), "granularity");
  return (int64_t)g;
}
// runs: sizes (bytes, multiples of the granularity) and domains; returns the base VA (never freed)
int64_t split_map(std::vector<int64_t> sizes, std::vector<int64_t> domains, int64_t dev) {
  ck(cuInit(0), "cuInit");
  size_t g = (size_t)granularity(dev), total = 0;
  for (auto s : sizes) { if ((size_t)s % g) throw std::runtime_error("run not granularity-aligned"); total += (size_t)s; }
  CUdeviceptr va = 0; ck(cuMemAddressReserve(&va, total, g, 0, 0), "cuMemAddressReserve");
  size_t off = 0;
  for (size_t i = 0; i < sizes.size(); i++) {
    CUmemAllocationProp p; memset(&p, 0, sizeof p); p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN; p.location.localized.deviceId = (int)dev;
    p.location.localized.localityDomainId = (unsigned char)domains[i];
    CUmemGenericAllocationHandle h; ck(cuMemCreate(&h, (size_t)sizes[i], &p, 0), "cuMemCreate");
    ck(cuMemMap(va + off, (size_t)sizes[i], 0, h, 0), "cuMemMap");
    ck(cuMemRelease(h), "cuMemRelease");
    off += (size_t)sizes[i];
  }
  CUmemAccessDesc a; memset(&a, 0, sizeof a); a.location.type = CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = (int)dev;
  a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ck(cuMemSetAccess(va, total, &a, 1), "cuMemSetAccess");
  return (int64_t)va;
}
int64_t ptr_domain(int64_t p) {
  cudaPointerAttributes at; memset(&at, 0, sizeof at);
  if (cudaPointerGetAttributes(&at, (const void*)p) != cudaSuccess) { cudaGetLastError(); return -100; }
  return at.localityDomainOrdinal;
}
"""


def _cudart_link_flags() -> list[str]:
    """Find a runtime library even when the unversioned development symlink
    is absent. Search CUDA_HOME, then the installed runtime wheels."""
    import glob
    import site

    home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    pats = [f"{home}/lib64/libcudart.so*", f"{home}/targets/*/lib/libcudart.so*", "/usr/lib/*/libcudart.so*"]
    for sp in site.getsitepackages() + [os.path.dirname(os.path.dirname(torch.__file__))]:
        pats += [f"{sp}/nvidia/*/lib/libcudart.so*", f"{sp}/nvidia/cu*/lib/libcudart.so*", f"{sp}/torch/lib/libcudart*.so*"]
    for pat in pats:
        hits = sorted(h for h in glob.glob(pat) if ".so" in h and not h.endswith(".a"))
        if hits:
            lib = hits[0]
            return [lib, f"-Wl,-rpath,{os.path.dirname(lib)}"]
    logger.warning("[dp2g] libcudart not found; relying on the process-loaded CUDA runtime symbols")
    return []


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        _EXT = load_inline(
            name="vllm_locality_kv",
            cpp_sources=_CPP,
            functions=["domain_count", "granularity", "split_map", "ptr_domain"],
            extra_ldflags=["-lcuda"] + _cudart_link_flags(),
            extra_include_paths=[os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "include")],
            build_directory=os.environ.get("VLLM_LOCALITY_BUILD_DIR") or None,
            verbose=False,
        )
    return _EXT


class _CAI:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = dict(shape=(nbytes,), typestr="|i1", data=(ptr, False), version=3)


def chunk_domains(nbytes: int, gran: int, tensors, num_blocks_of) -> list[int]:
    """Domain of each `gran`-sized chunk of the backing allocation."""
    nchunks = (nbytes + gran - 1) // gran
    dom = [0] * nchunks
    geo = []
    for t in tensors:
        if getattr(t, "host_resident", False):
            continue
        nb = num_blocks_of(t)
        geo.append((t.offset, t.layer_stride, t.block_stride, len(t.layers), nb, nb // 2))
    for c in range(nchunks):
        a = c * gran
        for off, ls, bs, nl, nb, half in geo:
            r = a - off
            if r < 0:
                continue
            if ls >= bs * nb:  # layer-outermost: layer regions of nb * bs bytes
                lay, rem = divmod(r, ls)
                if lay >= nl or rem >= nb * bs:
                    continue
                b = rem // bs
            else:  # block-outermost (layers interleaved inside a block)
                b = r // bs
                if b >= nb:
                    continue
            dom[c] = 1 if b >= half else 0
            break
    return dom


def _geometry(tensors, num_blocks_of) -> list[tuple[int, int, int, int, int]]:
    geo = []
    for t in tensors:
        if getattr(t, "host_resident", False):
            continue
        g = (t.offset, t.layer_stride, t.block_stride, len(t.layers), num_blocks_of(t))
        if g not in geo:  # hybrid groups alias the same slots: one entry per distinct geometry
            geo.append(g)
    return geo


def _block_ranges(g, b: int):
    off, ls, bs, nl, nb = g
    if ls >= bs * nb:  # layer-outermost: block b of every layer slot
        for lay in range(nl):
            s = off + lay * ls + b * bs
            yield s, s + bs
    else:  # block-outermost: one range spans all layers of block b
        s = off + b * bs
        yield s, s + bs


@dataclasses.dataclass(frozen=True)
class SplitPlan:
    dom: tuple[int, ...]  # domain of every GRAN chunk of the backing
    half: int             # N // 2: the block-index split the chunk placement follows
    ppd_block: int        # first block >= half whose bytes lie fully on domain 1 in every layer slot
    straddle: int         # (slot, block) ranges touching a chunk of the other domain (locality cost only)
    runs: int             # same-domain runs (one VA range)


def plan_split(nbytes: int, kv_cache_config, gran: int = GRAN) -> SplitPlan:
    """Pure function of the KV config (no GPU): per-chunk placement, block-granular boundary and straddle count.
    The scheduler (DomainFreeKVCacheBlockQueue) and the worker (pages_per_domain) share ppd_block through it."""
    tensors = kv_cache_config.kv_cache_tensors
    dom = chunk_domains(nbytes, gran, tensors, kv_cache_config.num_blocks_of)
    geo = _geometry(tensors, kv_cache_config.num_blocks_of)
    nb = kv_cache_config.num_blocks
    half = nb // 2

    def on(b: int, want: int) -> bool:
        return all(all(dom[c] == want for c in range(s // gran, (e - 1) // gran + 1))
                   for g in geo for s, e in _block_ranges(g, b))

    ppd = half
    while ppd < nb and not on(ppd, 1):
        ppd += 1
    straddle = sum(1 for g in geo for b in range(nb) for s, e in _block_ranges(g, b)
                   if any(dom[c] != (1 if b >= half else 0) for c in range(s // gran, (e - 1) // gran + 1)))
    runs = 1 + sum(1 for a, c in zip(dom, dom[1:]) if a != c) if dom else 0
    return SplitPlan(tuple(dom), half, ppd, straddle, runs)


def ppd_block_of(kv_cache_config) -> int:
    """Scheduler-side boundary (same plan as the worker); N // 2 when the plan cannot be built."""
    sizes = {t.size for t in kv_cache_config.kv_cache_tensors}
    if len(sizes) != 1:
        return kv_cache_config.num_blocks // 2
    return plan_split(sizes.pop(), kv_cache_config).ppd_block


def _gate_open(dev: int) -> bool:
    """Require supported topology and an open locality clock gate."""
    try:
        from vllm.model_executor.layers.locality import gate, topology
    except ImportError as e:
        _warn_once("gate-import", "[dp2g] VLLM_LOCALITY_SPLIT=1 needs overlay-qv2-locality (gate/topology): %s; "
                   "no split", e)
        return False
    if topology.get_topology(dev) is None:
        _warn_once("topo", "[dp2g] VLLM_LOCALITY_SPLIT=1 but < 2 locality domains; no split")
        return False
    if not gate.locality_active(dev):
        _warn_once("gate", "[dp2g] VLLM_LOCALITY_SPLIT=1 but the locality HBM-clock gate is closed: KV pool NOT "
                   "split (SPLIT_ACTIVE=False); the scheduler still partitions block ids at the domain boundary "
                   "(harmless for capacity; gate handshake deferred)")
        return False
    return True


def _alloc(buf_size: int, dom: list[int], dev: int) -> torch.Tensor:
    """Place each chunk on its planned domain. The optional runs allocator
    coalesces adjacent chunks with the same domain into one mapping."""
    if os.environ.get("VLLM_LOCALITY_SPLIT_ALLOC", "chunks") == "runs":
        runs_s, runs_d = [], []
        for d in dom:
            if runs_d and runs_d[-1] == d:
                runs_s[-1] += GRAN
            else:
                runs_s.append(GRAN)
                runs_d.append(d)
        ptr = int(_ext().split_map(runs_s, runs_d, dev))
        return torch.as_tensor(_CAI(ptr, sum(runs_s)), device=torch.device("cuda", dev))
    from vllm.model_executor.layers.locality import memory

    return memory.alloc_chunks(buf_size, dom, GRAN, dev)


def allocate_split(buf_size: int, device: torch.device, kv_cache_config) -> torch.Tensor | None:
    """Zero-filled int8 tensor of buf_size bytes whose blocks are placed per domain, or None (fall back to
    torch.zeros). Fail closed: any unverified chunk -> None and SPLIT_ACTIVE stays False."""
    global SPLIT_ACTIVE, _PPD_BLOCK, _NUM_BLOCKS
    SPLIT_ACTIVE = False
    dev = device.index if device.index is not None else torch.cuda.current_device()
    if not _gate_open(dev):
        return None
    from vllm.model_executor.layers.locality import memory

    gran = memory.granularity(dev)
    if gran != GRAN:
        logger.warning("[dp2g] allocation granularity %d != %d; no split", gran, GRAN)
        return None
    plan = plan_split(buf_size, kv_cache_config)
    dom = list(plan.dom)
    t0 = time.perf_counter()
    flat = _alloc(buf_size, dom, dev)
    t_map = time.perf_counter() - t0
    ords = memory.chunk_ordinals(flat, len(dom) * GRAN, GRAN)
    if list(ords) != dom:
        bad = sum(1 for a, b in zip(ords, dom) if a != b)
        logger.warning("[dp2g] KV split placement check FAILED (%d/%d chunks off-domain); no split", bad, len(dom))
        del flat
        return None
    buf = flat[:buf_size].view(torch.int8)
    buf.zero_()
    _PPD_BLOCK, _NUM_BLOCKS = plan.ppd_block, kv_cache_config.num_blocks
    SPLIT_ACTIVE = True
    logger.info(
        "[dp2g] KV backing split across locality domains: %.2f GiB, %d runs, domain1 %.1f%%, verified %d chunks, "
        "ppd_block %d (half %d), straddling %d, map %.1f s (%s)",
        len(dom)* GRAN / 2**30, plan.runs, 100.0 * sum(dom) / max(1, len(dom)), len(dom), plan.ppd_block, plan.half, plan.straddle, t_map,
        os.environ.get("VLLM_LOCALITY_SPLIT_ALLOC", "chunks"))
    return buf
