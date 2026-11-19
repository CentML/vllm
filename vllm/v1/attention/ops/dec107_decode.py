# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in SM107 FP8 paged decode attention (``DEC107_DECODE=1``).

The persistent kernel uses a graph-safe split-KV schedule, FP8 probabilities,
and a fixed-order merge. Supported shapes use FP8 E4M3 Q/K/V, head dimension
256, GQA ratio 8, kernel page size 32, uniform query length 1..4, and BF16 output.

``DEC107_LOCAL`` selects domain-aware queues with work stealing. The domain
map is probed at initialization; ``DEC107_LOCAL_PPD=auto`` obtains page
boundaries from the active split KV arena. Without an active arena it uses
a single queue.

``DEC107_MXK`` also loads the MXFP4-K / FP8-V kernel for the
``mxfp4k_fp8v`` cache dtype. This layout cannot fall back to ordinary FP8
decode. The backend must reject incompatible configurations before dispatch.

``DEC107_DECODE_QLENS``, ``DEC107_DECODE_MIN_B``, and ``DEC107_DECODE_MAX_B``
restrict FP8 routing. ``DEC107_CUBIN`` supplies a prebuilt FP8 cubin;
otherwise kernels are compiled into the cache at initialization.

Numerics: FLOAT-ORDER for decode and PRECISION for MXFP4-K. Determinism,
inline accuracy, and SWE validation are required before adoption.
"""

import ctypes
import hashlib
import os
import subprocess
import threading

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = envs.DEC107_DECODE
_QLENS = {int(x) for x in envs.DEC107_DECODE_QLENS.split(",") if x.strip()}
_MAX_B = envs.DEC107_DECODE_MAX_B
_MIN_B = envs.DEC107_DECODE_MIN_B
LOCAL = envs.DEC107_LOCAL
_PPD = envs.DEC107_LOCAL_PPD
MX = envs.DEC107_MXK

_HQ_PER_KV, _HD, _PAGE = 8, 256, 32
_K_BAL, _RED_SLICES, _MAX_REQ = 8, 64, 1024
_KDIR = os.path.join(os.path.dirname(__file__), "dec107_kernel")
if LOCAL:
    _THREADS, _SMEM = 384, 222528
    _SRC, _DEFS, _KNAME = os.path.join(_KDIR, "dec107_loc.cu"), ["-DDEC107_LOCAL"], b"kernel_dec107_decode_fp8x_hd256_p32"
else:
    _THREADS, _SMEM = 384, 222208
    _SRC, _DEFS, _KNAME = os.path.join(_KDIR, "dec107_decode_fp8.cu"), [], b"kernel_dec107_decode_fp8_hd256_p32"
# MXFP4-K uses a separate shared-memory layout.
_MX_SRC, _MX_DEFS, _MX_KNAME = os.path.join(_KDIR, "dec107_mx.cu"), (["-DDEC107_LOCAL"] if LOCAL else []), \
    b"kernel_dec107_decode_mxk_hd256_p32"
_MX_THREADS, _MX_SMEM = 512, 41984 + 5 * 17408 + 98304 + 320


def _cache_dir() -> str:
    d = os.path.join(os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm")), "dec107")
    os.makedirs(d, exist_ok=True)
    return d


class _TensorMap(ctypes.Structure):
    _fields_ = [("w", ctypes.c_uint64 * 16)]


class _Dev:
    """Per-device modules and buffers; side-stream decode has private workspace."""

    def __init__(self, device: torch.device):
        from cuda.bindings import driver as cu

        self.cu = cu
        self.device = device
        self.num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
        self.fn = self._load()
        self.max_split_items = 2 * _K_BAL * self.num_ctas
        self.max_split_tiles = _K_BAL * self.num_ctas
        self.qctr_off = ((self.max_split_tiles * 4 * 4 + 15) // 16) * 16
        po_bytes = self.max_split_items * 64 * _HD * 4
        self.ps_off = ((po_bytes + 255) // 256) * 256
        self.ws_bytes = self.ps_off + (self.max_split_items + 1) * 128 * 4
        self.bufs: dict[bool, tuple[torch.Tensor, torch.Tensor]] = {}
        self.sm_domain = None   # int8[256] device tensor (LOCAL); None = locality off
        if LOCAL:
            self.sm_domain = self._sm_domain_table()
        self.fn_mx = self._load(_MX_SRC, _MX_DEFS, _MX_KNAME, _MX_SMEM) if MX else None

    def _sm_domain_table(self):
        src = os.path.join(_KDIR, "dec107_smdom.cu")
        tag = hashlib.sha256(open(src, "rb").read()).hexdigest()[:16]
        lib = os.path.join(_cache_dir(), f"dec107_smdom_{tag}.so")
        if not os.path.exists(lib):
            tmp = f"{lib}.{os.getpid()}.tmp"
            subprocess.run(["nvcc", "-shared", "-Xcompiler", "-fPIC", "-O2", "-std=c++17", "-o", tmp, src, "-lcuda"],
                           check=True, capture_output=True)
            os.replace(tmp, lib)
        so = ctypes.CDLL(lib)
        tab = (ctypes.c_int8 * 256)()
        nsm = (ctypes.c_int * 3)()
        torch.cuda.current_device()
        rc = so.dec107_sm_domain_table(ctypes.c_int(self.device.index or 0), tab, nsm)
        if rc != 0:
            logger.warning("dec107 LOCAL: smid->domain table failed (rc=%d); running with one queue", rc)
            return None
        logger.info_once("dec107 LOCAL: SM->die map %d + %d SMs (%d no-domain SMs attributed by DRAM latency, "
                         "%d left unattributed)", nsm[0], nsm[1], nsm[2] >> 16, nsm[2] & 0xFFFF)
        return torch.tensor(list(tab), dtype=torch.int8, device=self.device)

    def _ck(self, r):
        err = r[0] if isinstance(r, tuple) else r
        if err != self.cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"dec107: CUDA driver error {err}")
        if isinstance(r, tuple):
            return r[1] if len(r) == 2 else r[1:]
        return None

    def _load(self, src_path=None, defs=None, kname=None, smem=None):
        src_path = src_path or _SRC
        defs = _DEFS if defs is None else defs
        kname = kname or _KNAME
        smem = smem or _SMEM
        cubin = os.environ.get("DEC107_CUBIN", "") if src_path == _SRC else ""
        if not cubin:
            maj, mnr = torch.cuda.get_device_capability(self.device)
            arch = f"sm_{maj}{mnr}a"
            src = open(src_path, "rb").read()
            tag = hashlib.sha256(src + arch.encode() + " ".join(defs).encode()).hexdigest()[:16]
            cubin = os.path.join(_cache_dir(), f"dec107_{arch}_{tag}.cubin")
            if not os.path.exists(cubin):
                tmp = f"{cubin}.{os.getpid()}.tmp"
                subprocess.run(["nvcc", "-cubin", f"-arch={arch}", "-O3", "-std=c++17", *defs, "-o", tmp, src_path],
                               check=True, capture_output=True)
                os.replace(tmp, cubin)
        torch.cuda.current_device()  # make sure the primary context is current
        mod = self._ck(self.cu.cuModuleLoad(cubin.encode()))
        fn = self._ck(self.cu.cuModuleGetFunction(mod, kname))
        self._ck(self.cu.cuFuncSetAttribute(
            fn, self.cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
        logger.info_once("dec107 decode kernel loaded: %s", cubin)
        return fn

    def buffers(self, side: bool) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.bufs.get(side)
        if b is None:
            ctr = torch.zeros((self.qctr_off + 64) // 4 + 64, dtype=torch.int32, device=self.device)
            ws = torch.empty(self.ws_bytes, dtype=torch.uint8, device=self.device)
            b = self.bufs[side] = (ctr, ws)
        return b

    def tmap_plain(self, ptr, gdim, gstride, box) -> _TensorMap:
        return self.tmap(ptr, gdim, gstride, box, swizzle=False)

    def tmap(self, ptr, gdim, gstride, box, swizzle=True) -> _TensorMap:
        cu = self.cu
        u64, u32 = cu.cuuint64_t, cu.cuuint32_t
        tm = self._ck(cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8, u32(len(gdim)), ptr, [u64(x) for x in gdim],
            [u64(x) for x in gstride], [u32(x) for x in box], [u32(1)] * len(gdim),
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B if swizzle else cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))
        t = _TensorMap()
        for i, x in enumerate(tm.opaque):
            t.w[i] = int(x)
        return t


_DEVS: dict[int, _Dev] = {}
_LOCK = threading.Lock()
_FAILED = False


def _dev(device: torch.device) -> _Dev:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    d = _DEVS.get(idx)
    if d is None:
        with _LOCK:
            d = _DEVS.get(idx)
            if d is None:
                d = _DEVS[idx] = _Dev(torch.device("cuda", idx))
    return d


def _auto_ppd(num_blocks: int) -> int:
    """Use the active split KV arena's page boundary, otherwise one queue."""
    try:
        from vllm.v1.worker import locality_kv
    except ImportError:
        return 0
    return int(locality_kv.pages_per_domain(num_blocks)) if getattr(locality_kv, "SPLIT_ACTIVE", False) else 0


def preallocate(device: torch.device, side: bool = False) -> None:
    """Build/load the kernel and allocate counters + partial workspace before KV-cache memory profiling."""
    global _FAILED
    if not ENABLED or _FAILED:
        return
    try:
        d = _dev(device)
        d.buffers(False)
        if side:
            d.buffers(True)
    except Exception as e:  # never break serving: fall back to trtllm-gen
        _FAILED = True
        logger.warning("dec107 decode disabled (load failed): %r", e)


def supports(query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, out: torch.Tensor,
             block_tables: torch.Tensor, q_len_per_req, window_left: int, sinks, return_lse: bool,
             bmm1_scale, bmm2_scale) -> bool:
    if not ENABLED or _FAILED:
        return False
    if q_len_per_req is None or q_len_per_req not in _QLENS:
        return False
    if window_left is not None and window_left >= 0:
        return False
    if sinks is not None or return_lse:
        return False
    if not isinstance(bmm1_scale, float) or not isinstance(bmm2_scale, float):
        return False
    if query.dtype != torch.float8_e4m3fn or k_cache.dtype != torch.float8_e4m3fn:
        return False
    if out.dtype != torch.bfloat16 or not out.is_contiguous() or not query.is_contiguous():
        return False
    if query.dim() != 3 or query.size(2) != _HD or query.size(1) % _HQ_PER_KV:
        return False
    if k_cache.dim() != 4 or k_cache.size(2) != _PAGE or k_cache.size(3) != _HD:
        return False
    if query.size(1) != _HQ_PER_KV * k_cache.size(1):
        return False
    B = block_tables.size(0)
    if B > min(_MAX_B, _MAX_REQ) or B < _MIN_B or query.size(0) != B * q_len_per_req:
        return False
    for t in (k_cache, v_cache):
        if t.stride(3) != 1 or any(s % 16 for s in t.stride()[:3]):
            return False
    return True


def run(query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_tables: torch.Tensor,
        seq_lens: torch.Tensor, bmm1_scale: float, bmm2_scale: float, out: torch.Tensor, q_len_per_req: int,
        side: bool = False, enable_pdl: bool = True) -> None:
    d = _dev(query.device)
    cu = d.cu
    ctr, ws = d.buffers(side)
    B, hq, hkv = block_tables.size(0), query.size(1), k_cache.size(1)
    T = query.size(0)
    tq = d.tmap(query.data_ptr(), [128, hq, T, 2], [_HD, hq * _HD, 128], [128, _HQ_PER_KV, 8, 2])
    tk = d.tmap(k_cache.data_ptr(), [128, _PAGE, 2, hkv, k_cache.size(0)],
                [k_cache.stride(2), 128, k_cache.stride(1), k_cache.stride(0)], [128, _PAGE, 1, 1, 1])
    tv = d.tmap(v_cache.data_ptr(), [128, _PAGE, 2, hkv, v_cache.size(0)],
                [v_cache.stride(2), 128, v_cache.stride(1), v_cache.stride(0)], [128, _PAGE, 1, 1, 1])
    max_items = B * hkv + d.max_split_items + d.max_split_tiles * _RED_SLICES
    args = [tq, tk, tv, ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(block_tables.data_ptr()),
            ctypes.c_void_p(seq_lens.data_ptr()), ctypes.c_void_p(ws.data_ptr()),
            ctypes.c_void_p(ws.data_ptr() + d.ps_off), ctypes.c_void_p(ctr.data_ptr()),
            ctypes.c_void_p(ctr.data_ptr() + d.qctr_off), ctypes.c_int(block_tables.size(1)),
            ctypes.c_float(bmm1_scale * 1.4426950408889634), ctypes.c_int(hq), ctypes.c_int(hkv), ctypes.c_int(B),
            ctypes.c_int(q_len_per_req), ctypes.c_uint(max_items), ctypes.c_float(bmm2_scale)]
    if LOCAL:
        ppd = 0
        if d.sm_domain is not None:
            ppd = _auto_ppd(k_cache.size(0)) if _PPD == "auto" else int(_PPD)
        args += [ctypes.c_void_p(d.sm_domain.data_ptr() if d.sm_domain is not None else 0), ctypes.c_int(ppd),
                 ctypes.c_void_p(0), ctypes.c_longlong(0), ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)]
    argv = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    stream = cu.CUstream(torch.cuda.current_stream(query.device).cuda_stream)
    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = d.num_ctas, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = _THREADS, 1, 1
    cfg.sharedMemBytes, cfg.hStream = _SMEM, stream
    if enable_pdl:
        at = cu.CUlaunchAttribute()
        at.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        at.value.programmaticStreamSerializationAllowed = 1
        cfg.attrs, cfg.numAttrs = [at], 1
    else:
        cfg.numAttrs = 0
    d._ck(cu.cuLaunchKernelEx(cfg, d.fn, ctypes.addressof(argv), 0))


COUNTS = {"run_mx": 0, "run_mx_pruning": 0, "mxk_dequant": 0, "mxk_dequant_pages": 0}


def count(name: str, n: int = 1) -> None:
    """dec107 MXFP4-K path counters; logged at 1, 10, 100, ... per counter (host-side, eager calls only; calls
    replayed inside CUDA graphs are counted once at capture)."""
    c = COUNTS[name] = COUNTS.get(name, 0) + n
    if name != "mxk_dequant_pages" and c > 0 and (c & (c - 1) == 0 or str(c).strip("0") == "1"):
        logger.info("DEC107 counters %s", " ".join(f"{k}={v}" for k, v in COUNTS.items()))


def run_mx(query: torch.Tensor, pages: torch.Tensor, hkv: int, block_tables: torch.Tensor, seq_lens: torch.Tensor,
           bmm1_scale: float, bmm2_scale: float, out: torch.Tensor, q_len_per_req: int, side: bool = False,
           enable_pdl: bool = True, pruning: bool = False) -> None:
    """dec107 MXFP4-K / FP8-V decode on ``pages`` [P, Hkv * 32 * 392] uint8 (dec107_mxk layout). ``query`` FP8 E4M3
    [B * q_len, Hq, 256] (already Hadamard-rotated if the cache is); bmm1 = sm_scale * q_scale (K needs no scale)."""
    count("run_mx_pruning" if pruning else "run_mx")
    d = _dev(query.device)
    cu = d.cu
    assert d.fn_mx is not None, "DEC107_MXK=1 required"
    assert query.dtype == torch.float8_e4m3fn and query.is_contiguous() and out.dtype == torch.bfloat16
    ctr, ws = d.buffers(side)
    B, hq, T = block_tables.size(0), query.size(1), query.size(0)
    pb = pages.stride(0)
    base = pages.data_ptr()
    tq = d.tmap(query.data_ptr(), [128, hq, T, 2], [_HD, hq * _HD, 128], [128, _HQ_PER_KV, 2, 1])
    tk = d.tmap(base, [128, _PAGE, hkv, pages.size(0)], [128, _PAGE * 128, pb], [128, _PAGE, 1, 1])
    tv = d.tmap(base + hkv * _PAGE * 136, [128, _PAGE, 2, hkv, pages.size(0)], [256, 128, _PAGE * 256, pb],
                [128, _PAGE, 1, 1, 1])
    tks = d.tmap_plain(base + hkv * _PAGE * 128, [256, hkv, pages.size(0)], [256, pb], [256, 1, 1])
    max_items = B * hkv + d.max_split_items + d.max_split_tiles * _RED_SLICES
    ppd = 0
    if d.sm_domain is not None:
        ppd = _auto_ppd(pages.size(0)) if _PPD == "auto" else int(_PPD)
    args = [tq, tk, tv, tks, ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(block_tables.data_ptr()),
            ctypes.c_void_p(seq_lens.data_ptr()), ctypes.c_void_p(ws.data_ptr()),
            ctypes.c_void_p(ws.data_ptr() + d.ps_off), ctypes.c_void_p(ctr.data_ptr()),
            ctypes.c_void_p(ctr.data_ptr() + d.qctr_off), ctypes.c_int(block_tables.size(1)),
            ctypes.c_float(bmm1_scale * 1.4426950408889634), ctypes.c_int(hq), ctypes.c_int(hkv), ctypes.c_int(B),
            ctypes.c_int(q_len_per_req), ctypes.c_uint(max_items), ctypes.c_float(bmm2_scale),
            ctypes.c_void_p(d.sm_domain.data_ptr() if d.sm_domain is not None else 0), ctypes.c_int(ppd),
            ctypes.c_void_p(0), ctypes.c_longlong(0), ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0),
            ctypes.c_void_p(0)]
    argv = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    stream = cu.CUstream(torch.cuda.current_stream(query.device).cuda_stream)
    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = d.num_ctas, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = _MX_THREADS, 1, 1
    cfg.sharedMemBytes, cfg.hStream = _MX_SMEM, stream
    if enable_pdl:
        at = cu.CUlaunchAttribute()
        at.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        at.value.programmaticStreamSerializationAllowed = 1
        cfg.attrs, cfg.numAttrs = [at], 1
    else:
        cfg.numAttrs = 0
    d._ck(cu.cuLaunchKernelEx(cfg, d.fn_mx, ctypes.addressof(argv), 0))
