# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local KV cache, worker side (``VLLM_DIE_LOCAL_KV=1``).

``allocate_backing`` maps the KV cache backing allocation so that every page of
block ids ``[0, s)`` lives in locality domain 0's HBM and ``[s, N)`` in domain
1's (``vllm/v1/core/die_local_kv.py`` assigns each request the blocks of one
die), and creates one green-context stream per domain. ``split_launch`` then runs
a paged-KV attention launch twice, once per die on that die's SMs, each over the
rows whose newest page is on its die (the other rows get KV length 0), and keeps
each row's output from its own die's launch.

The green contexts' SM groups (``VLLM_DIE_LOCAL_KV_SM_GROUPS``, default
``probe``): at start-up one CTA per SM times dependent cold loads from each
domain's memory (the near domain answers faster), which gives every SM's die,
including the SMs in neither locality domain. Those need not split evenly
between the dies, in which case backfilled groups would hold SMs of the other
die. The largest split whose groups hold no SM of the other die is used; a probe
that disagrees with the domains' own SMs keeps backfill.

CUDA 13.4 locality-domain API (CUDA Programming Guide 4.7): ``cuMemCreate`` with
``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`` and ``cuDevSmResourceSplit`` with
``CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID``, called through ctypes on
libcuda (struct layouts of the 13.4 ``cuda.h``). Same placement scheme as
FlashInfer #6791 (``empty_per_locality_domain``,
``split_device_green_ctx_by_locality_domain``).

With ``VLLM_DIE_LOCAL_KV_STRIPE=1`` the scheduler stripes every request over
both dies instead (block i of a request on die (i + s) % 2;
``vllm/v1/core/die_local_kv.py``): no launch is split per die
(``split_rows_ok`` is false), die 1's trtllm workspace is not made, and
``reserve`` builds the die-aware decode kernel of ``VLLM_H211_DL_VERIFY``
(``vllm/v1/attention/ops/die_verify_attn.py``) instead, which reads each die's
blocks from that die's SMs.
"""

import ctypes
import gc
import os
import types
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.core.die_local_kv import ENV_STRIPE, die_split, enabled, stripe_enabled
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Both dies' launches write ``out`` when they run the multi-CTA-KV kernels with
# the KV split the vLLM policy raised (they leave rows with KV length 0
# untouched: checked bitwise); otherwise, or with
# =0, die 1 writes its own buffer and rows are selected after both launches.
ENV_SHARED_OUT = "VLLM_DIE_LOCAL_KV_SHARED_OUT"
# Split launches with fewer rows run on the whole GPU (unset: every launch with
# rows is split).
ENV_MIN_ROWS = "VLLM_DIE_LOCAL_KV_MIN_ROWS"
# "probe" (default), "backfill" (each domain's SMs plus half of the others,
# unchecked) or "strict" (each domain's SMs only).
ENV_SM_GROUPS = "VLLM_DIE_LOCAL_KV_SM_GROUPS"
# Split-k of the per-die launches, "min_rows:k,...": where max_seq_len is at
# least VLLM_DIE_LOCAL_KV_SPLIT_K_MIN_SEQ (CUDA graphs capture max_model_len),
# a launch over at least min_rows rows (the last tier that applies) runs that k
# (at least 2, at most the single launch's). Otherwise, or with "0" or unset, the
# single launch's k. The tiers depend on the GPU and the serving configuration:
# no defaults (VLLM_DIE_LOCAL_KV_SPLIT_K_MIN_SEQ unset: 0).
ENV_SPLIT_K = "VLLM_DIE_LOCAL_KV_SPLIT_K"
ENV_SPLIT_K_MIN_SEQ = "VLLM_DIE_LOCAL_KV_SPLIT_K_MIN_SEQ"
# Release the memory of an earlier backing still referenced by stale tensors
# (the CUDA-graph memory profiling's) before mapping the next one.
ENV_EVICT_STALE = "VLLM_DIE_LOCAL_KV_EVICT_STALE"
# =1: in eager execution (never while a CUDA graph is captured), split_launch
# reuses three preallocated fork/join events instead of creating three per call,
# and a caller that passes cache_masks=True (the target model's decode attention)
# gets the row masks computed once per forward: its 10 layers share the same
# block tables and KV lengths. Same kernels, same inputs: bitwise.
HOST_TRIM = os.environ.get("VLLM_DIE_LOCAL_KV_HOST_TRIM", "0") == "1"

_ATTR_LOCALITY_DOMAIN_COUNT = 149
_LOC_DEVICE, _LOC_DOMAIN = 1, 6
_GRP_BACKFILL, _GRP_LOCALITY_DOMAIN_ID = 1, 2
_LD, _LD_BF = _GRP_LOCALITY_DOMAIN_ID, _GRP_LOCALITY_DOMAIN_ID | _GRP_BACKFILL
# SM probe: dependent loads per SM and domain, the bytes per SM and domain they
# hop over, and the L2 flush before it. The far/near latency ratio every SM must
# show depends on the GPU: VLLM_DIE_LOCAL_KV_PROBE_MIN_RATIO, no default (unset:
# no margin check).
ENV_PROBE_MIN_RATIO = "VLLM_DIE_LOCAL_KV_PROBE_MIN_RATIO"
_PROBE_HOPS = 48
_PROBE_SLOT = 128 << 10
_PROBE_FLUSH = 512 << 20
_u64, _sz, _vp = ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p
_int, _uint = ctypes.c_int, ctypes.c_uint


class _MemLoc(ctypes.Structure):
    _fields_ = [("type", _int), ("id", _int)]


class _AllocProp(ctypes.Structure):
    _fields_ = [
        ("type", _int),
        ("requestedHandleTypes", _int),
        ("location", _MemLoc),
        ("win32HandleMetaData", _vp),
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _MemLoc), ("flags", _int)]


class _SmResource(ctypes.Structure):
    _fields_ = [
        ("smCount", _uint),
        ("minSmPartitionSize", _uint),
        ("smCoscheduledAlignment", _uint),
        ("flags", _uint),
        ("localityDomainId", _uint),
        ("pad", ctypes.c_ubyte * 20),
    ]


class _DevResource(ctypes.Structure):
    _fields_ = [
        ("type", _int),
        ("pad", ctypes.c_ubyte * 92),
        ("sm", _SmResource),
        ("nextResource", _vp),
    ]


class _GroupParams(ctypes.Structure):
    _fields_ = [
        ("smCount", _uint),
        ("coscheduledSmCount", _uint),
        ("preferredCoscheduledSmCount", _uint),
        ("flags", _uint),
        ("localityDomainId", _uint),
        ("reserved", _uint * 11),
    ]


def _libcuda() -> ctypes.CDLL:
    cu = ctypes.CDLL("libcuda.so.1")
    for name, args in (
        ("cuDeviceGet", [ctypes.POINTER(_int), _int]),
        ("cuDeviceGetAttribute", [ctypes.POINTER(_int), _int, _int]),
        ("cuMemGetAllocationGranularity",
         [ctypes.POINTER(_sz), ctypes.POINTER(_AllocProp), _int]),
        ("cuMemAddressReserve", [ctypes.POINTER(_u64), _sz, _sz, _u64, _u64]),
        ("cuMemCreate", [ctypes.POINTER(_u64), _sz, ctypes.POINTER(_AllocProp), _u64]),
        ("cuMemMap", [_u64, _sz, _sz, _u64, _u64]),
        ("cuMemSetAccess", [_u64, _sz, ctypes.POINTER(_AccessDesc), _sz]),
        ("cuMemUnmap", [_u64, _sz]),
        ("cuMemRelease", [_u64]),
        ("cuMemAddressFree", [_u64, _sz]),
        ("cuDeviceGetDevResource", [_int, ctypes.POINTER(_DevResource), _int]),
        ("cuDevSmResourceSplit",
         [ctypes.POINTER(_DevResource), _uint, ctypes.POINTER(_DevResource),
          ctypes.POINTER(_DevResource), _uint, ctypes.POINTER(_GroupParams)]),
        ("cuDevResourceGenerateDesc",
         [ctypes.POINTER(_vp), ctypes.POINTER(_DevResource), _uint]),
        ("cuGreenCtxCreate", [ctypes.POINTER(_vp), _vp, _int, _uint]),
        ("cuGreenCtxStreamCreate", [ctypes.POINTER(_vp), _vp, _uint, _int]),
        ("cuGreenCtxDestroy", [_vp]),
        ("cuStreamDestroy_v2", [_vp]),
    ):  # fmt: skip
        fn = getattr(cu, name)
        fn.restype, fn.argtypes = _int, args
    return cu


def _check(result: int, what: str) -> None:
    if result != 0:
        raise RuntimeError(f"die-local KV: {what} failed (CUresult {result})")


def _prop(cudev: int, home: int) -> _AllocProp:
    """Pinned memory on locality domain ``home`` (-1: the device, interleaved)."""
    p = _AllocProp()
    p.type = 1  # pinned
    if home < 0:
        p.location.type, p.location.id = _LOC_DEVICE, cudev
    else:
        p.location.type = _LOC_DOMAIN
        p.location.id = (cudev & 0xFF) | ((home & 0xFF) << 8)
    return p


def _granularity(cu: ctypes.CDLL, cudev: int, home: int) -> int:
    gran = _sz()
    _check(
        cu.cuMemGetAllocationGranularity(
            ctypes.byref(gran), ctypes.byref(_prop(cudev, home)), 1
        ),
        "granularity",
    )
    return gran.value


Chunks = list[tuple[int, int, int]]  # (offset, size, handle)


def _free(cu: ctypes.CDLL, ptr: int, nbytes: int, chunks: Chunks) -> None:
    for offset, run, handle in chunks:
        cu.cuMemUnmap(ptr + offset, run)
        cu.cuMemRelease(handle)
    cu.cuMemAddressFree(ptr, nbytes)


def _map(
    cu: ctypes.CDLL, cudev: int, homes: list[int], gran: int
) -> tuple[int, int, Chunks]:
    """Reserve and map ``len(homes)`` granules, granule i on domain ``homes[i]``.

    Returns (address, bytes, chunks); raises RuntimeError with nothing mapped.
    """
    nbytes = len(homes) * gran
    va = _u64()
    _check(cu.cuMemAddressReserve(ctypes.byref(va), nbytes, gran, 0, 0), "reserve")
    mapped: Chunks = []
    try:
        start = 0
        for i in range(1, len(homes) + 1):
            if i < len(homes) and homes[i] == homes[start]:
                continue
            run = (i - start) * gran
            handle = _u64()
            _check(
                cu.cuMemCreate(
                    ctypes.byref(handle),
                    run,
                    ctypes.byref(_prop(cudev, homes[start])),
                    0,
                ),
                f"cuMemCreate({run} B on domain {homes[start]})",
            )
            ret = cu.cuMemMap(va.value + start * gran, run, 0, handle.value, 0)
            if ret != 0:
                cu.cuMemRelease(handle.value)
                _check(ret, "cuMemMap")
            mapped.append((start * gran, run, handle.value))
            start = i
        access = _AccessDesc()
        access.location.type = _LOC_DEVICE
        access.location.id, access.flags = cudev, 3
        _check(cu.cuMemSetAccess(va.value, nbytes, ctypes.byref(access), 1), "access")
    except RuntimeError:
        _free(cu, va.value, nbytes, mapped)
        raise
    return va.value, nbytes, mapped


class _Raw:
    """A mapped range for ``torch.as_tensor``, freed by its owner."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|i1",
            "data": (ptr, False),
            "version": 3,
        }


class _Mapping(_Raw):
    """Owner of a mapped range; torch keeps it alive with the tensor and its
    views. When they are all gone (e.g. the CUDA-graph memory profiling KV
    cache), the range is unmapped and its memory released."""

    def __init__(self, cu: ctypes.CDLL, ptr: int, nbytes: int, chunks: Chunks):
        super().__init__(ptr, nbytes)
        _LIVE[ptr] = nbytes
        _CHUNKS[ptr] = chunks
        weakref.finalize(self, _unmap, cu, ptr, nbytes, chunks).atexit = False


# Backings whose tensors are alive: address -> bytes.
_LIVE: dict[int, int] = {}
# Their mapped chunks (the finalizer's list: emptied when evicted).
_CHUNKS: dict[int, Chunks] = {}
# Ranges whose tensors died; released (after a device synchronize) outside CUDA
# graph capture.
_DEFERRED: list[tuple[ctypes.CDLL, int, int, Chunks]] = []


def _unmap(cu: ctypes.CDLL, ptr: int, nbytes: int, chunks: Chunks) -> None:
    _LIVE.pop(ptr, None)
    _CHUNKS.pop(ptr, None)
    _DEFERRED.append((cu, ptr, nbytes, chunks))
    _release_deferred()


def _stale_holders(limit: int = 8) -> list[str]:
    """Who still references tensors on a live earlier backing (for the log):
    ``Owner.attribute`` for a tensor held in an object's ``__dict__``, else the
    referrer's type."""
    ranges = [(p, p + n) for p, n in _LIVE.items()]
    found: list[str] = []
    objs = gc.get_objects()
    for obj in objs:
        if len(found) >= limit:
            break
        if not issubclass(type(obj), torch.Tensor):
            continue
        try:
            ptr = obj.data_ptr()
        except RuntimeError:
            continue
        if not any(lo <= ptr < hi for lo, hi in ranges):
            continue
        for ref in gc.get_referrers(obj):
            if ref is objs or isinstance(ref, types.FrameType):
                continue
            if isinstance(ref, dict):
                owners = [
                    o
                    for o in gc.get_referrers(ref)
                    if getattr(o, "__dict__", None) is ref
                ]
                keys = [k for k, v in ref.items() if v is obj]
                name = type(owners[0]).__name__ if owners else "dict"
                found.append(f"{name}.{keys[0] if keys else '?'}")
            else:
                d = getattr(ref, "__dict__", None)
                keys = [k for k, v in d.items() if v is obj] if type(d) is dict else []
                found.append(type(ref).__name__ + (f".{keys[0]}" if keys else ""))
    del objs
    return found


def _evict_live() -> None:
    """Release the physical memory of earlier backings whose tensors are still
    referenced (``VLLM_DIE_LOCAL_KV_EVICT_STALE=1``), so it is free before the
    new backing is mapped. The address range stays reserved until those tensors
    are gone; an access through one faults instead of reading freed memory."""
    torch.cuda.synchronize()
    for ptr, chunks in _CHUNKS.items():
        if ptr not in _LIVE or not chunks:
            continue
        cu = _libcuda()
        for offset, run, handle in chunks:
            cu.cuMemUnmap(ptr + offset, run)
            cu.cuMemRelease(handle)
        chunks.clear()
        logger.info(
            "die-local KV: released the memory of an earlier %.1f GiB KV backing "
            "before mapping the new one (its address range stays reserved)",
            _LIVE[ptr] / 2**30,
        )


def _release_deferred() -> None:
    if not _DEFERRED or torch.cuda.is_current_stream_capturing():
        return
    torch.cuda.synchronize()
    while _DEFERRED:
        cu, ptr, nbytes, chunks = _DEFERRED.pop()
        _free(cu, ptr, nbytes, chunks)
        logger.info("die-local KV: released a %.1f GiB KV backing", nbytes / 2**30)


@triton.jit
def _smid():
    return tl.inline_asm_elementwise(
        "mov.u32 $0, %smid;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _sm_ids_kernel(out_ptr):
    tl.store(out_ptr + tl.program_id(0), _smid())


@triton.jit
def _probe_hops(ptr, lat_ptr, x, lines, zero, HOPS: tl.constexpr):
    # Each hop: clock, an L1-bypassing load, a (never taken) trap on its value
    # so that the second clock read waits for it; the next address depends on
    # the value through ``zero`` (a runtime 0).
    dep = zero.to(tl.int64)
    for h in range(HOPS):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        cycles, v = tl.inline_asm_elementwise(
            """
            {
            .reg .u64 t0, t1;
            .reg .pred p;
            mov.u64 t0, %clock64;
            ld.global.cg.u32 $1, [$2];
            setp.eq.u32 p, $3, 1;
            setp.eq.and.u32 p, $1, 0xFFFFFFFF, p;
            @p trap;
            mov.u64 t1, %clock64;
            sub.u64 t1, t1, t0;
            cvt.u32.u64 $0, t1;
            }
            """,
            "=r,=r,l,r",
            [ptr + ((x >> 8) % lines) * 32 + dep, zero],
            dtype=(tl.int32, tl.int32),
            is_pure=False,
            pack=1,
        )
        dep = (v & zero).to(tl.int64)
        tl.store(lat_ptr + h, cycles)


@triton.jit(do_not_specialize=["zero"])
def _sm_probe_kernel(
    buf0,
    buf1,
    lat_ptr,
    seen_ptr,
    nslots,
    slot_words,
    lines,
    zero,
    HOPS: tl.constexpr,
):
    # The first CTA on each SM times HOPS loads from each domain's slot of
    # that SM (128-byte lines, random order).
    sm = _smid()
    if sm < nslots:
        if tl.atomic_cas(seen_ptr + sm, 0, 1) == 0:
            offset = sm.to(tl.int64) * slot_words
            lat = lat_ptr + sm.to(tl.int64) * (2 * HOPS)
            x = sm.to(tl.int64) * 7919 + 12345
            _probe_hops(buf0 + offset, lat, x, lines, zero, HOPS)
            _probe_hops(buf1 + offset, lat + HOPS, x + 104729, lines, zero, HOPS)


@dataclass
class _Probe:
    dies: dict[int, int]  # smid -> die (lower median latency)
    near: float  # median over SMs of the near domain's median cycles
    far: float
    min_ratio: float  # the smallest far / near median over SMs
    low: list[int]  # SMs with far / near < VLLM_DIE_LOCAL_KV_PROBE_MIN_RATIO


def _probe_min_ratio() -> float | None:
    raw = os.environ.get(ENV_PROBE_MIN_RATIO, "")
    return float(raw) if raw else None


def _probe_dies(cu: ctypes.CDLL, cudev: int, device: torch.device, nsm: int) -> _Probe:
    nslots = 2 * nsm
    gran = _granularity(cu, cudev, 0)
    per_domain = -(-nslots * _PROBE_SLOT // gran)
    flush_gran = _granularity(cu, cudev, -1)
    maps = [_map(cu, cudev, [0] * per_domain + [1] * per_domain, gran)]
    try:
        maps.append(_map(cu, cudev, [-1] * -(-_PROBE_FLUSH // flush_gran), flush_gran))
        words = torch.as_tensor(_Raw(*maps[0][:2]), device=device).view(torch.int32)
        flush = torch.as_tensor(_Raw(*maps[1][:2]), device=device)
        lat = torch.zeros(nslots * 2 * _PROBE_HOPS, dtype=torch.int32, device=device)
        seen = torch.zeros(nslots, dtype=torch.int32, device=device)
        for ctas in (8 * nsm, 32 * nsm):
            flush.zero_()
            seen.zero_()
            _sm_probe_kernel[(ctas,)](
                words,
                words[per_domain * gran // 4 :],
                lat,
                seen,
                nslots,
                _PROBE_SLOT // 4,
                _PROBE_SLOT // 128,
                0,
                HOPS=_PROBE_HOPS,
                num_warps=1,
            )
            if int((seen > 0).sum()) >= nsm:
                break
        lat_host = lat.view(nslots, 2, _PROBE_HOPS).cpu()
        seen_host = seen.cpu().tolist()
    finally:
        torch.cuda.synchronize()
        for va, nbytes, chunks in maps:
            _free(cu, va, nbytes, chunks)
    dies: dict[int, int] = {}
    near, far, ratios, low = [], [], [], []
    margin = _probe_min_ratio()
    for sm, hit in enumerate(seen_host):
        if not hit:
            continue
        med = [float(lat_host[sm, d].float().median()) for d in (0, 1)]
        dies[sm] = int(med[1] < med[0])
        near.append(min(med))
        far.append(max(med))
        ratios.append(max(med) / max(min(med), 1.0))
        if margin is not None and ratios[-1] < margin:
            low.append(sm)
    return _Probe(
        dies=dies,
        near=sorted(near)[len(near) // 2] if near else 0.0,
        far=sorted(far)[len(far) // 2] if far else 0.0,
        min_ratio=min(ratios, default=0.0),
        low=low,
    )


@dataclass
class _Green:
    """One green context and stream per die."""

    name: str
    sms: tuple[int, int]
    streams: tuple[torch.cuda.ExternalStream, torch.cuda.ExternalStream]
    handles: list[tuple[int, int]]  # (green context, stream) per die


GroupSpec = tuple[tuple[int, int], tuple[int, int]]  # (smCount, flags) per die


def _make_green(
    cu: ctypes.CDLL,
    cudev: int,
    device: torch.device,
    res: _DevResource,
    name: str,
    spec: GroupSpec,
) -> _Green:
    params = (_GroupParams * 2)()
    for i, (count, flags) in enumerate(spec):
        params[i].smCount = count
        params[i].coscheduledSmCount = 2
        params[i].flags = flags
        params[i].localityDomainId = i
    groups, rem = (_DevResource * 2)(), _DevResource()
    _check(
        cu.cuDevSmResourceSplit(
            groups, 2, ctypes.byref(res), ctypes.byref(rem), 0, params
        ),
        "cuDevSmResourceSplit",
    )
    handles: list[tuple[int, int]] = []
    streams = []
    try:
        for i in range(2):
            desc, ctx, stream = _vp(), _vp(), _vp()
            _check(
                cu.cuDevResourceGenerateDesc(
                    ctypes.byref(desc), ctypes.byref(groups[i]), 1
                ),
                "cuDevResourceGenerateDesc",
            )
            _check(
                cu.cuGreenCtxCreate(ctypes.byref(ctx), desc, cudev, 1),
                "cuGreenCtxCreate",
            )
            ret = cu.cuGreenCtxStreamCreate(ctypes.byref(stream), ctx, 1, 0)
            if ret != 0:
                cu.cuGreenCtxDestroy(ctx)
                _check(ret, "cuGreenCtxStreamCreate")
            handles.append((ctx.value, stream.value))
            streams.append(torch.cuda.ExternalStream(stream.value, device=device))
    except RuntimeError:
        _destroy(cu, handles)
        raise
    sms = (groups[0].sm.smCount, groups[1].sm.smCount)
    return _Green(name, sms, (streams[0], streams[1]), handles)


def _destroy(cu: ctypes.CDLL, handles: list[tuple[int, int]]) -> None:
    torch.cuda.synchronize()
    for ctx, stream in handles:
        cu.cuStreamDestroy_v2(stream)
        cu.cuGreenCtxDestroy(ctx)


def _members(green: _Green, device: torch.device) -> list[set[int]]:
    """The SM ids each of ``green``'s streams runs CTAs on."""
    out = []
    for stream, count in zip(green.streams, green.sms):
        found: set[int] = set()
        for ctas in (8 * count, 32 * count):
            ids = torch.full((max(ctas, 1),), -1, dtype=torch.int32, device=device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _sm_ids_kernel[(ids.numel(),)](ids, num_warps=1)
            torch.cuda.synchronize()
            found |= set(ids.tolist()) - {-1}
            if len(found) >= count:
                break
        out.append(found)
    return out


def _far(members: list[set[int]], dies: dict[int, int]) -> int:
    """SMs in a die's group that the probe puts on the other die (or not at all)."""
    return sum(1 for d in (0, 1) for sm in members[d] if dies.get(sm) != d)


def _probe_problem(
    probe: _Probe, domain: list[set[int]], domain_sms: tuple[int, int], nsm: int
) -> str | None:
    """Why the probe cannot be trusted: it must see every SM with a clear
    near/far difference and agree with the locality domains' own SMs."""
    if len(probe.dies) != nsm:
        return f"probed {len(probe.dies)} of {nsm} SMs"
    if probe.low:
        return f"far/near < {_probe_min_ratio()} on SMs {probe.low}"
    if any(len(domain[d]) != domain_sms[d] for d in (0, 1)):
        return f"domain groups ran on {len(domain[0])}+{len(domain[1])} SMs"
    wrong = _far(domain, probe.dies)
    if wrong:
        return f"{wrong} SMs of a locality domain probe on the other die"
    return None


def _candidates(counts: tuple[int, int]) -> list[tuple[str, GroupSpec]]:
    """Splits that give each die its probe count of SMs, most SMs first: each
    die backfilled to its count ("matched"); the die with more SMs backfilled,
    the other its domain only ("asym"). "strict" is the fallback."""
    big = int(counts[1] > counts[0])
    asym = tuple((counts[i], _LD_BF) if i == big else (0, _LD) for i in (0, 1))
    return [
        ("matched", ((counts[0], _LD_BF), (counts[1], _LD_BF))),
        ("asym", asym),  # type: ignore[list-item]
    ]


def _log_probe(probe: _Probe, domain: list[set[int]]) -> None:
    top = max(probe.dies, default=-1)
    table = "".join(
        str(probe.dies[s]) if s in probe.dies else "?" for s in range(top + 1)
    )
    outside = sorted(set(probe.dies) - domain[0] - domain[1])
    logger.info(
        "die-local KV: SM probe, die per smid 0-%d: %s; near/far median %.0f/%.0f "
        "cycles, far/near min %.2f; per die %d/%d; SMs in neither locality "
        "domain: %s",
        top,
        table,
        probe.near,
        probe.far,
        probe.min_ratio,
        sum(1 for d in probe.dies.values() if d == 0),
        sum(1 for d in probe.dies.values() if d == 1),
        {s: probe.dies[s] for s in outside},
    )


def _select_green(cu: ctypes.CDLL, cudev: int, device: torch.device) -> _Green:
    res = _DevResource()
    _check(cu.cuDeviceGetDevResource(cudev, ctypes.byref(res), 1), "SM resource")
    nsm = res.sm.smCount
    backfill: GroupSpec = ((nsm // 2, _LD_BF), (nsm // 2, _LD_BF))
    mode = os.environ.get(ENV_SM_GROUPS, "probe")
    if mode == "backfill":
        return _make_green(cu, cudev, device, res, mode, backfill)
    strict = _make_green(cu, cudev, device, res, "strict", ((0, _LD), (0, _LD)))
    if mode == "strict":
        return strict
    if mode != "probe":
        logger.warning("die-local KV: %s=%r unknown, using probe", ENV_SM_GROUPS, mode)
    if _probe_min_ratio() is None:
        logger.info(
            "die-local KV: %s unset: the SM probe checks no far/near margin",
            ENV_PROBE_MIN_RATIO,
        )
    try:
        probe = _probe_dies(cu, cudev, device, nsm)
        domain = _members(strict, device)
    except Exception as e:  # noqa: BLE001 - any probe failure keeps backfill
        problem: str | None = f"probe failed ({e})"
    else:
        _log_probe(probe, domain)
        problem = _probe_problem(probe, domain, strict.sms, nsm)
    if problem:
        logger.warning("die-local KV: %s; backfill SM groups", problem)
        _destroy(cu, strict.handles)
        return _make_green(cu, cudev, device, res, "backfill", backfill)
    counts = (
        sum(1 for d in probe.dies.values() if d == 0),
        sum(1 for d in probe.dies.values() if d == 1),
    )
    chosen, tried = strict, []
    for name, spec in _candidates(counts):
        try:
            green = _make_green(cu, cudev, device, res, name, spec)
        except RuntimeError as e:
            tried.append(f"{name}: {e}")
            continue
        members = _members(green, device)
        far = _far(members, probe.dies)
        complete = all(len(members[d]) == green.sms[d] for d in (0, 1))
        tried.append(f"{name} {green.sms[0]}+{green.sms[1]} far {far}")
        if far == 0 and complete:
            chosen = green
            break
        _destroy(cu, green.handles)
    if chosen is not strict:
        _destroy(cu, strict.handles)
    logger.info(
        "die-local KV: SM groups %s %d+%d, no SM of the other die (tried: %s)",
        chosen.name,
        *chosen.sms,
        "; ".join(tried),
    )
    return chosen


@dataclass
class _State:
    num_blocks: int
    split: int
    streams: tuple[torch.cuda.ExternalStream, torch.cuda.ExternalStream]
    sms: tuple[int, int]
    workspace1: torch.Tensor
    shared_out: bool
    min_rows: int
    split_k: list[tuple[int, int]]  # (min rows, per-die k), ascending
    split_k_min_seq: int
    twins: dict[int, torch.Tensor] = field(default_factory=dict)
    logged: bool = False
    k_logged: set[tuple[str, int, int, int]] = field(default_factory=set)
    # VLLM_DIE_LOCAL_KV_HOST_TRIM: fork, join die 0, join die 1 (eager only).
    events: tuple[torch.cuda.Event, ...] | None = None
    trim_stats: dict[str, int] = field(
        default_factory=lambda: {"masks": 0, "reused": 0, "eager": 0}
    )
    # VLLM_DIE_LOCAL_KV_STRIPE: requests striped over the dies; no split launch.
    striped: bool = False


_STATE: _State | None = None
# The green contexts and die 1's trtllm workspace, made by reserve() inside the
# KV memory profile (else by the first allocate_backing that gets this far).
_GREEN: _Green | None = None
_WORKSPACE1: torch.Tensor | None = None
# VLLM_DIE_LOCAL_KV_STRIPE: reserve() ran (it makes no die 1 workspace).
_RESERVED = False


def _workspace1_bytes() -> int:
    """Die 1's trtllm workspace: half of the shared one (rounded up).

    It holds only die 1's split-KV partials. A launch runs once per die only
    where the split policy raised its split (``_trtllm_gen_decode_sm_count`` in
    ``attention/backends/flashinfer.py``), which keeps the launch's partials in
    half of the shared workspace (the forked KF prefill's are in the other
    half), and each die's launch runs at most that k (``split_sm_count``).
    """
    return -(-envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE // 2)


def _open(device: torch.device) -> tuple[ctypes.CDLL, int] | str:
    """The libcuda and CUdevice of ``device`` if it has two locality domains,
    else why it cannot hold die-local KV."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        cu = _libcuda()
        cudev, count = _int(), _int()
        _check(cu.cuDeviceGet(ctypes.byref(cudev), index), "cuDeviceGet")
        _check(
            cu.cuDeviceGetAttribute(
                ctypes.byref(count), _ATTR_LOCALITY_DOMAIN_COUNT, cudev
            ),
            "locality domain count",
        )
    except (OSError, AttributeError, RuntimeError) as e:
        return f"no locality-domain API ({e})"
    if count.value != 2:
        return f"{count.value} locality domain(s)"
    return cu, cudev.value


def _reserve_h211(device: torch.device, max_num_reqs: int | None) -> str:
    """VLLM_H211_DL_VERIFY=1: build the die-aware verify kernel, probe its SM ->
    die table and make its workspace (``max_num_reqs`` sequences) now, outside
    CUDA-graph capture; the FlashInfer backend's dispatch only looks it up."""
    from vllm.v1.attention.ops import die_verify_attn

    if not die_verify_attn.enabled():
        return f"{die_verify_attn.ENV_FLAG} off"
    try:
        rt = die_verify_attn.get_runtime(
            device, max_seqs=max_num_reqs or die_verify_attn.MAX_SEQS
        )
    except Exception as e:  # noqa: BLE001 - decode stays on trtllm-gen
        logger.warning(
            "H211 die-verify: kernel unavailable (%s); decode stays on trtllm-gen", e
        )
        return "H211 kernel unavailable"
    if rt is None:
        return "H211 kernel unavailable on this device"
    return (
        f"H211 die-aware verify workspace {rt.workspace_bytes / 2**20:.0f} MiB "
        f"for {rt.max_seqs} sequences"
    )


def reserve(device: torch.device, max_num_reqs: int | None = None) -> None:
    """Make the green contexts and die 1's trtllm workspace now, inside the KV
    memory profile (the model runner's ``profile_run``), so that the KV cache
    budget counts them. Once per process; with the switch off, or a device
    without two locality domains, nothing (``allocate_backing`` decides).

    VLLM_DIE_LOCAL_KV_STRIPE=1: no die 1 workspace (no split launches); the
    H211 kernel and its workspace instead (``_reserve_h211``).
    """
    global _GREEN, _WORKSPACE1, _RESERVED
    if not enabled() or _WORKSPACE1 is not None or _RESERVED:
        return
    device = torch.device(device)
    opened = _open(device)
    if isinstance(opened, str):
        return
    cu, cudev = opened
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info(device)[0]
    try:
        if _GREEN is None:
            _GREEN = _select_green(cu, cudev, device)
    except RuntimeError:
        return  # allocate_backing tries again and switches die-local KV off
    if stripe_enabled():
        _RESERVED = True
        what = _reserve_h211(device, max_num_reqs)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        free -= torch.cuda.mem_get_info(device)[0]
        logger.info(
            "die-local KV (%s=1): %.0f MiB made before the KV memory profile ends, "
            "so the KV budget counts them: green contexts %d+%d SMs (%s); %s; no die "
            "1 attention workspace (no per-die split launches)",
            ENV_STRIPE,
            free / 2**20,
            *_GREEN.sms,
            _GREEN.name,
            what,
        )
        return
    _WORKSPACE1 = torch.zeros(
        _workspace1_bytes(), dtype=torch.uint8, device=device
    )
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free -= torch.cuda.mem_get_info(device)[0]
    logger.info(
        "die-local KV: %.0f MiB made before the KV memory profile ends, so the KV "
        "budget counts them: die 1's attention workspace %.0f MiB (half of the "
        "shared %.0f MiB) and green contexts %d+%d SMs (%s)",
        free / 2**20,
        _WORKSPACE1.nbytes / 2**20,
        envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE / 2**20,
        *_GREEN.sms,
        _GREEN.name,
    )


def _workspace1(device: torch.device) -> torch.Tensor:
    global _WORKSPACE1
    if _WORKSPACE1 is None:
        _WORKSPACE1 = torch.zeros(
            _workspace1_bytes(), dtype=torch.uint8, device=device
        )
        logger.warning(
            "die-local KV: no reserve() before the KV memory profile; the KV budget "
            "does not count die 1's attention workspace (%.0f MiB)",
            _WORKSPACE1.nbytes / 2**20,
        )
    return _WORKSPACE1


def active() -> bool:
    return _STATE is not None


def split_rows_ok(num_rows: int) -> bool:
    """Whether a launch over ``num_rows`` requests runs once per die (never with
    striped KV: every request has blocks on both dies)."""
    return (
        _STATE is not None and not _STATE.striped and num_rows >= _STATE.min_rows
    )


def striped() -> bool:
    """Striped die-local KV is mapped in this worker (VLLM_DIE_LOCAL_KV_STRIPE)."""
    return _STATE is not None and _STATE.striped


def stripe_geometry() -> tuple[int, int]:
    """(blocks, first block id on die 1) of the current die-local backing."""
    assert _STATE is not None
    return _STATE.num_blocks, _STATE.split


def _page_block_ids(
    kv_cache_config: KVCacheConfig, size: int, gran: int
) -> list[int] | None:
    """Block id at the midpoint of every ``gran``-byte chunk of the backing
    allocation, or None if the tensors' layout is not recognized."""
    num_blocks = kv_cache_config.num_blocks
    tensors = [t for t in kv_cache_config.kv_cache_tensors if not t.host_resident]
    if not tensors:
        return None
    t = tensors[0]
    nlayers = len(t.layers)
    if t.layer_stride >= t.block_stride * num_blocks:
        layer_outer = True
    elif t.block_stride >= t.layer_stride * nlayers:
        layer_outer = False
    else:
        return None
    ids = []
    for i in range(-(-size // gran)):
        rel = min(i * gran + gran // 2, size - 1) - t.offset
        if rel < 0:
            ids.append(0)
            continue
        if layer_outer:
            block = (rel % t.layer_stride) // t.block_stride
        else:
            block = rel // t.block_stride
        ids.append(min(block, num_blocks - 1))
    return ids


def _placement_check(
    kv_cache_config: KVCacheConfig, homes: list[int], gran: int, split: int
) -> float:
    """Fraction of all layers' pages whose first and last byte are on their die."""
    good = total = 0
    for t in kv_cache_config.kv_cache_tensors:
        if t.host_resident:
            continue
        page = min(t.block_stride, t.layer_stride)
        for layer in range(len(t.layers)):
            for block in range(kv_cache_config.num_blocks):
                start = t.offset + layer * t.layer_stride + block * t.block_stride
                die = int(block >= split)
                total += 1
                good += (
                    homes[start // gran] == die
                    and homes[(start + page - 1) // gran] == die
                )
    return good / max(total, 1)


def _off(reason: str) -> None:
    """No die-local backing for this call; every launch runs whole-GPU."""
    global _STATE
    _STATE = None
    logger.warning("die-local KV off: %s", reason)


def allocate_backing(
    kv_cache_config: KVCacheConfig, device: torch.device, size: int
) -> torch.Tensor | None:
    """The zeroed int8 KV backing allocation with die-homed blocks, or None (the
    caller then allocates it as usual, and every launch runs whole-GPU) when the
    switch is off, the device has no two locality domains or the mapping fails.

    Every call (the CUDA-graph memory profiling's minimal KV cache, then the real
    one) maps a new backing and makes its block geometry the current one; a
    previous backing is released once its tensors are gone.
    """
    global _STATE, _GREEN
    if not enabled():
        return None
    if _LIVE:
        gc.collect()
    _release_deferred()
    if _LIVE and os.environ.get(ENV_EVICT_STALE, "0") == "1":
        logger.info(
            "die-local KV: %d earlier KV backing(s) (%.1f GiB) still referenced "
            "by %s",
            len(_LIVE),
            sum(_LIVE.values()) / 2**30,
            ", ".join(sorted(set(_stale_holders()))) or "?",
        )
        _evict_live()
    elif _LIVE:
        logger.info(
            "die-local KV: %d earlier KV backing(s) (%.1f GiB) still referenced; "
            "released once their tensors are gone",
            len(_LIVE),
            sum(_LIVE.values()) / 2**30,
        )
    device = torch.device(device)
    opened = _open(device)
    if isinstance(opened, str):
        return _off(opened)
    cu, cudev = opened
    num_blocks = kv_cache_config.num_blocks
    split = die_split(num_blocks)
    try:
        g = _granularity(cu, cudev, 0)
        block_ids = _page_block_ids(kv_cache_config, size, g)
        if block_ids is None:
            return _off("unrecognized KV cache tensor layout")
        if _GREEN is None:
            _GREEN = _select_green(cu, cudev, device)
        green = _GREEN
        homes = [int(b >= split) for b in block_ids]
        va, nbytes, mapped = _map(cu, cudev, homes, g)
    except RuntimeError as e:
        return _off(str(e))
    buf = torch.as_tensor(_Mapping(cu, va, nbytes, mapped), device=device)
    buf = buf[:size]
    buf.zero_()
    if _STATE is None:
        # Striped: no per-die split launch, so no die 1 workspace or split-k.
        is_striped = stripe_enabled()
        _STATE = _State(
            num_blocks=num_blocks,
            split=split,
            streams=green.streams,
            sms=green.sms,
            workspace1=torch.empty(0, dtype=torch.uint8, device=device)
            if is_striped
            else _workspace1(device),
            shared_out=os.environ.get(ENV_SHARED_OUT, "1") == "1",
            min_rows=0 if is_striped else _min_rows(),
            split_k=[] if is_striped else _split_k_tiers(),
            split_k_min_seq=_split_k_min_seq(),
            striped=is_striped,
        )
    else:
        # Streams, die 1's workspace and counter twins carry over.
        _STATE.num_blocks, _STATE.split, _STATE.logged = num_blocks, split, False
    placed = _placement_check(kv_cache_config, homes, g, split)
    logger.info(
        "die-local KV: %.1f GiB KV backing, blocks [0, %d) on domain 0 (%.1f GiB), "
        "[%d, %d) on domain 1; %.4f of pages on their die; green-context SMs %d+%d "
        "(%s)",
        nbytes / 2**30,
        split,
        homes.count(0) * g / 2**30,
        split,
        num_blocks,
        placed,
        *_STATE.sms,
        green.name,
    )
    if placed < 0.999:
        logger.warning(
            "die-local KV: only %.4f of KV pages lie wholly on their die", placed
        )
    if _STATE.striped:
        logger.info(
            "die-local KV striped (%s=1): each request's attention blocks alternate "
            "dies (die 1 from block %d of %d); no decode launch is split per die",
            ENV_STRIPE,
            split,
            num_blocks,
        )
    return buf


def twin(buffer: torch.Tensor | None) -> torch.Tensor | None:
    """A zeroed buffer like ``buffer`` for die 1's launch (multi-CTA semaphores),
    made outside CUDA graph capture; None (FlashInfer allocates one) otherwise."""
    assert _STATE is not None
    if buffer is None:
        return None
    key = buffer.data_ptr()
    t = _STATE.twins.get(key)
    if t is None or t.numel() < buffer.numel():
        if torch.cuda.is_current_stream_capturing():
            return None
        t = _STATE.twins[key] = torch.zeros_like(buffer)
    return t


def _min_rows() -> int:
    raw = os.environ.get(ENV_MIN_ROWS, "")
    if raw:
        return int(raw)
    logger.info(
        "die-local KV: %s unset: every launch with rows runs once per die",
        ENV_MIN_ROWS,
    )
    return 1


def _split_k_min_seq() -> int:
    return int(os.environ.get(ENV_SPLIT_K_MIN_SEQ, "0"))


def _split_k_tiers() -> list[tuple[int, int]]:
    raw = os.environ.get(ENV_SPLIT_K, "").strip()
    if not raw:
        logger.info(
            "die-local KV: %s unset: per-die launches keep the single launch's "
            "split-k",
            ENV_SPLIT_K,
        )
    if raw in ("", "0"):
        return []
    try:
        tiers = [tuple(map(int, item.split(":"))) for item in raw.split(",")]
        if any(len(t) != 2 or t[1] < 2 for t in tiers):
            raise ValueError
    except ValueError:
        logger.warning(
            "die-local KV: %s=%r is not min_rows:k,... (k >= 2); per-die launches "
            "keep the single launch's split-k",
            ENV_SPLIT_K,
            raw,
        )
        return []
    return sorted(tiers)  # type: ignore[arg-type]


def split_sm_count(
    site: str,
    num_reqs: int,
    q_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    sm_count: int,
) -> int:
    """The SM count to report to FlashInfer for a split site's per-die launches.

    ``sm_count`` is the single-launch policy's, ``k * num_kv_heads * num_reqs``
    with the split-k ``k`` sized for one launch on every SM. Each die's launch
    runs on about half of them; at the CUDA-graph max_seq_len, where most of a
    launch's CTAs start past their request's KV, ``VLLM_DIE_LOCAL_KV_SPLIT_K``
    picks a smaller k for the launch's rows. A launch is bitwise equal to a
    single launch at the same k.
    """
    assert _STATE is not None
    ctas = num_kv_heads * num_reqs
    k = sm_count // ctas
    k_tier = 0
    if max_seq_len >= _STATE.split_k_min_seq:
        for min_rows, tier_k in _STATE.split_k:
            if num_reqs >= min_rows:
                k_tier = tier_k
    k_die = max(2, min(k, k_tier)) if k_tier else k
    key = (site, q_len, k, k_die)
    if key not in _STATE.k_logged and len(_STATE.k_logged) < 64:
        _STATE.k_logged.add(key)
        logger.info(
            "die-local KV: split-k at %s q%d (%d rows, max_seq_len %d): %d per die "
            "(single launch %d)",
            site,
            q_len,
            num_reqs,
            max_seq_len,
            k_die,
            k,
        )
    return k_die * ctas


def workspace(die: int, workspace0: torch.Tensor) -> torch.Tensor:
    """The trtllm workspace of ``die``'s launch (die 0: the shared one)."""
    if die == 0:
        return workspace0
    assert _STATE is not None
    # The split policy kept the launch's partials in half of workspace0.
    assert 2 * _STATE.workspace1.nbytes >= workspace0.nbytes
    return _STATE.workspace1


@triton.jit
def _die_rows_kernel(
    bt_ptr,
    bt_stride,
    seq_ptr,
    s0_ptr,
    s1_ptr,
    tok_ptr,
    cu_ptr,
    q_len,
    page,
    lo,
    hi,
    HAS_TOK: tl.constexpr,
    HAS_CU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    seq = tl.load(seq_ptr + r)
    idx = tl.maximum(seq - 1, 0) // page
    bid = tl.load(bt_ptr + r.to(tl.int64) * bt_stride + idx)
    on1 = (bid >= lo) & (bid < hi) & (seq > 0)
    tl.store(s0_ptr + r, tl.where(on1, 0, seq))
    tl.store(s1_ptr + r, tl.where(on1, seq, 0))
    if HAS_TOK:
        if HAS_CU:
            start = tl.load(cu_ptr + r)
            end = tl.load(cu_ptr + r + 1)
        else:
            start = r * q_len
            end = start + q_len
        val = on1.to(tl.int8)
        for off in range(start, end, BLOCK):
            offs = off + tl.arange(0, BLOCK)
            tl.store(tok_ptr + offs, tl.zeros([BLOCK], tl.int8) + val, mask=offs < end)


def row_masks(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kernel_pages: int,
    page_tokens: int,
    num_tokens: int | None,
    q_len_per_req: int | None,
    cu_seqlens_q: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Per row: KV length for die 0's and die 1's launch (the other die's rows
    get 0); per token (if ``num_tokens``): 1 where the row is die 1's. A row's
    die is the die of its newest page."""
    assert _STATE is not None
    ratio, rem = divmod(num_kernel_pages, _STATE.num_blocks)
    if rem or not ratio:
        raise RuntimeError(
            f"die-local KV: {num_kernel_pages} kernel pages do not match the "
            f"{_STATE.num_blocks}-block die-local backing"
        )
    rows = seq_lens.shape[0]
    s0 = torch.empty_like(seq_lens)
    s1 = torch.empty_like(seq_lens)
    tok = None
    if num_tokens is not None:
        tok = torch.zeros(num_tokens, dtype=torch.int8, device=seq_lens.device)
    if rows:
        _die_rows_kernel[(rows,)](
            block_tables,
            block_tables.stride(0),
            seq_lens,
            s0,
            s1,
            tok if tok is not None else s0,
            cu_seqlens_q if cu_seqlens_q is not None else seq_lens,
            q_len_per_req or 1,
            page_tokens,
            _STATE.split * ratio,
            _STATE.num_blocks * ratio,
            HAS_TOK=tok is not None,
            HAS_CU=cu_seqlens_q is not None,
            BLOCK=16,
        )
    return s0, s1, tok


def _forward_row_masks(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kernel_pages: int,
    page_tokens: int,
    num_tokens: int | None,
    q_len_per_req: int | None,
    cu_seqlens_q: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """VLLM_DIE_LOCAL_KV_HOST_TRIM: ``row_masks`` once per forward for the same
    inputs. The cache lives on the forward context (a new one per forward), so
    masks never outlive the attention metadata they were made from; the key is
    every input of ``row_masks`` (tensor storage, shape and stride, the page
    geometry and the die split)."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    assert _STATE is not None
    if not is_forward_context_available():
        return row_masks(
            block_tables,
            seq_lens,
            num_kernel_pages,
            page_tokens,
            num_tokens,
            q_len_per_req,
            cu_seqlens_q,
        )
    ctx = get_forward_context()
    key = (
        block_tables.data_ptr(),
        tuple(block_tables.shape),
        tuple(block_tables.stride()),
        block_tables.dtype,
        seq_lens.data_ptr(),
        tuple(seq_lens.shape),
        seq_lens.dtype,
        None
        if cu_seqlens_q is None
        else (cu_seqlens_q.data_ptr(), tuple(cu_seqlens_q.shape), cu_seqlens_q.dtype),
        num_kernel_pages,
        page_tokens,
        num_tokens,
        q_len_per_req,
        _STATE.split,
        _STATE.num_blocks,
    )
    cached = getattr(ctx, "_die_local_row_masks", None)
    if cached is not None and cached[0] == key:
        _note_trim("reused")
        return cached[1]
    masks = row_masks(
        block_tables,
        seq_lens,
        num_kernel_pages,
        page_tokens,
        num_tokens,
        q_len_per_req,
        cu_seqlens_q,
    )
    ctx._die_local_row_masks = (key, masks)  # type: ignore[attr-defined]
    _note_trim("masks")
    return masks


def _note_trim(what: str) -> None:
    """VLLM_DIE_LOCAL_KV_HOST_TRIM engagement: count, log the first reuse and
    every 20000 eager split launches."""
    assert _STATE is not None
    st = _STATE.trim_stats
    st[what] += 1
    if (what == "reused" and st["reused"] == 1) or (
        what == "eager" and st["eager"] % 20000 == 0
    ):
        logger.info(
            "die-local KV host trim (VLLM_DIE_LOCAL_KV_HOST_TRIM=1): eager split "
            "launches %d on 3 preallocated events; target decode row masks "
            "computed %d, reused %d",
            st["eager"],
            st["masks"],
            st["reused"],
        )


def split_launch(
    launch: Callable[[int, torch.Tensor, torch.Tensor], None],
    *,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    num_kernel_pages: int,
    page_tokens: int,
    q_len_per_req: int | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    multi_cta: bool = False,
    cache_masks: bool = False,
) -> None:
    """Run ``launch(die, seq_lens, out)`` once per die on that die's green-context
    stream, between a fork from and a join to the current stream, then keep each
    token row of ``out`` from its own die's launch.

    ``cache_masks`` (VLLM_DIE_LOCAL_KV_HOST_TRIM): the caller's inputs are the
    same for every layer of this forward; the row masks are then computed once
    per forward and reused (eager only)."""
    assert _STATE is not None
    shared = _STATE.shared_out and multi_cta
    trim = HOST_TRIM and not torch.cuda.is_current_stream_capturing()
    args = (
        block_tables,
        seq_lens,
        num_kernel_pages,
        page_tokens,
        None if shared else out.shape[0],
        q_len_per_req,
        cu_seqlens_q,
    )
    if trim and cache_masks:
        s0, s1, tok = _forward_row_masks(*args)
    else:
        s0, s1, tok = row_masks(*args)
    out1 = out if shared else torch.empty_like(out)
    main = torch.cuda.current_stream()
    if trim:
        if _STATE.events is None:
            _STATE.events = (torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event())
        fork, *die_joins = _STATE.events
        _note_trim("eager")
    else:
        fork, die_joins = torch.cuda.Event(), None
    fork.record(main)
    joins = []
    for die, (stream, lens, o) in enumerate(zip(_STATE.streams, (s0, s1), (out, out1))):
        stream.wait_event(fork)
        with torch.cuda.stream(stream):
            launch(die, lens, o)
        join = torch.cuda.Event() if die_joins is None else die_joins[die]
        join.record(stream)
        joins.append(join)
    for join in joins:
        main.wait_event(join)
    if tok is not None:
        cond = tok.view(-1, *([1] * (out.dim() - 1))).bool()
        torch.where(cond, out1, out, out=out)
    if not _STATE.logged:
        _STATE.logged = True
        logger.info(
            "die-local KV: decode attention runs one launch per die (SMs %d+%d)",
            *_STATE.sms,
        )
