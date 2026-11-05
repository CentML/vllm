# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H211: die-aware MTP-verify attention on die-striped KV (``VLLM_H211_DL_VERIFY``, default off).

Kernel: ``q36k_attn_decode_dl`` by Yibai Meng, in ``vllm/third_party/q36k_attn_dl/sources.py``. One persistent
launch per call, one CTA per SM (tcgen05 FP8 MMA, TMA page ring). The pages of every (sequence, KV head) stream are
split on the device from ``kv_lens`` into per-die chunk queues; a CTA takes chunks only from its own SM's die queue
(an SM -> die table from a latency probe), so on die-localized KV each die's SMs read only their die's HBM. Split
streams are merged at the end of the launch, in a fixed order, by the CTAs whose queues ran out.

Placement the kernel expects (``die_split_page=0``, "striped"): the pages of every sequence alternate dies of a KV
pool whose kernel pages [0, P/2) are in locality domain 0's HBM and [P/2, P) in domain 1's
(``die_local_kv.allocate_backing`` maps a backing that way). The source stripes single pages, page p on die p % 2.
vLLM allocates this model's KV in blocks of several 128-token kernel pages (9 at block size 1152), so the port makes
one change to the kernel (``STRIPE_PATCHES``, applied to the vendored text at build time): runs of ``stripe_pages``
pages alternate, page p on die (p // stripe_pages + phase) % 2, where a sequence's phase is 1 when its first page id
is >= ``stripe_phase_page`` > 0 (its first block on die 1), else 0. Starting every sequence on die 0 would leave die
0 about half a block per sequence ahead; an allocator that alternates the first block's die keeps the dies within
a fraction of a block. ``stripe_pages=1, stripe_phase_page=0`` is the source's own striping (same work split, same
page order).

Flag: ``VLLM_H211_DL_VERIFY=1`` lets ``get_runtime()`` build the kernel (the image's nvcc, sm_107a, cached under
``VLLM_H211_DL_CACHE_DIR`` or vLLM's cache root) and return it; unset or 0 returns None and builds nothing.

Serving: with ``VLLM_DIE_LOCAL_KV=1`` and ``VLLM_DIE_LOCAL_KV_STRIPE=1`` the scheduler alternates each request's
blocks between the dies (``vllm/v1/core/die_local_kv.py`` DieStripePolicy) and ``die_local_kv.reserve`` builds the
kernel before the KV memory profile ends. The FlashInfer backend then runs it in place of a trtllm-gen decode launch
of at least ``VLLM_H211_DL_VERIFY_MIN_ROWS`` rows (unset: every launch) with q_len <= 8, on the layers of
``VLLM_H211_DL_VERIFY_PARTS`` ("t": the target model, "d": the MTP draft; unset: t), within the contract below
(``dispatch_config``, ``decide``; one log line per decision kind). Smaller launches stay on trtllm-gen, which reads
the striped pool as it reads the interleaved one. The launch: grid = the device's SMs, one persistent wave, one chunk
per SM (chunks_per_sm 1, no stealing).

Contract (``DieVerifyAttn.check``): q FP8 e4m3 [T, 16, 256] contiguous, unit scale; the per-layer KV cache
[P, 2, 128, 512] FP8 with K|V packed per token (the FlashInfer backend's HND layout), or its (K, V) views; out BF16
[T, 16, 256] contiguous; page table int32 rows of kernel pages; kv_lens int32 [B] (cached + this step's tokens);
cu_seqlens_q int32 [B + 1]; q_len <= 8; B <= 1024; softmax scale 1/16 with unit q/k/v scales; no window, sinks,
soft cap or LSE output.

Numerics: the same attention as trtllm-gen, not bitwise: P keeps about 8 significant bits (hi/lo e4m3 split) and
split streams merge in fp32 in a fixed order (bitwise reproducible for the same lengths, die table, placement and
chunks_per_sm). Runs that use it change outputs and need --accuracy.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_FLAG = "VLLM_H211_DL_VERIFY"
ENV_CACHE = "VLLM_H211_DL_CACHE_DIR"
ENV_NVCC = "VLLM_H211_DL_NVCC"
# Serving dispatch (vllm/v1/attention/backends/flashinfer.py): decode launches of at least this many rows (unset:
# every launch), on these parts ("t" target, "d" MTP draft; "t,d" both; unset: t).
ENV_MIN_ROWS = "VLLM_H211_DL_VERIFY_MIN_ROWS"
ENV_PARTS = "VLLM_H211_DL_VERIFY_PARTS"
PARTS = ("t", "d")
# Debug: the first N eager die-aware launches of each layer also run trtllm-gen and log the difference (syncs).
ENV_CHECK = "VLLM_H211_DL_VERIFY_CHECK"
# The gate's numerics tolerance against the served kernel (rel L2): a check above it logs a warning.
CHECK_REL_L2_WARN = 0.05

# Geometry the kernel is compiled for (include/q36k/attn.h): the full-attention layers of Qwen3.6-35B-A3B at TP1.
HQ, HKV, HDIM, PAGE = 16, 2, 256, 128
PAGE_BYTES = HKV * PAGE * 2 * HDIM
MAX_Q = 8  # Q36K_ATTN_DL_MAX_Q_LEN
MAX_SEQS = 1024  # Q36K_ATTN_MAX_SEQS
MAX_CHUNKS_PER_SM = 4  # Q36K_ATTN_DL_MAX_CHUNKS_PER_SM
MAX_SMS = 256  # Q36K_DIE_MAX_SMS
SM_SCALE = 1.0 / 16.0  # 1/sqrt(head_dim), compiled into the kernel
COUNTER_OFFSET = 3584 * 1024  # Q36K_ATTN_DL_COUNTER_OFFSET
COUNTER_BYTES = (32 + 4 * MAX_SEQS) * 4  # Q36K_ATTN_DL_COUNTER_BYTES
CAPABILITY = (10, 7)  # tcgen05 kind::f8f6f4 with the sm_107 instruction descriptor (attn_dl.cu kIdesc*)
STATUS = {0: "ok", 1: "invalid argument", 2: "unsupported shape or configuration", 3: "CUDA error",
          4: "workspace too small"}

KERNEL = "attn/attn_dl.cu"
# H211's change to the kernel sources, (file, old, new): each `old` occurs exactly once in the source file.
# Striped placement in runs of die_stripe pages with a per-sequence start die: page p of a sequence on die
# (p / die_stripe + phase) % 2, phase 1 when die_stripe_phase > 0 and the sequence's first page id is >= it (its
# first run on die 1), else 0. The source has page p on die p % 2: die_stripe 1 with die_stripe_phase 0 gives back
# every one of its expressions. A new exported entry point takes both; q36k_attn_decode_dl keeps its ABI and runs
# with (1, 0).
STRIPE_PATCHES: tuple[tuple[str, str, str], ...] = (
    (KERNEL,
     "  int die_split;  // 0: striped placement (page p on die p % 2); > 0: per sequence, pages >= die_split on die 1\n"
     "};\n",
     "  int die_split;  // 0: striped placement (page p on die p % 2); > 0: per sequence, pages >= die_split on die 1\n"
     "  // H211 (vLLM port), striped placement: runs of die_stripe pages alternate dies, page p of a sequence on die\n"
     "  // (p / die_stripe + phase) % 2, phase 1 when die_stripe_phase > 0 and the sequence's first page id is\n"
     "  // >= die_stripe_phase (its first run on die 1), else 0. die_stripe 1, die_stripe_phase 0: as above.\n"
     "  int die_stripe, die_stripe_phase;\n"
     "};\n"),
    (KERNEL,
     "// Stream (b, h) in die d's page space: start s0 and n pages; its pieces = chunks of die d it falls into; the "
     "partial\n",
     "// H211 (vLLM port): striped placement in runs of `stripe` pages (vLLM allocates KV in blocks of several 128-token\n"
     "// pages; Params::die_stripe). Pages in even runs among a sequence's first np pages, and the sequence page of\n"
     "// the j-th page in runs of parity poff (pstride 2; pstride 1: page j). stripe 1: 2 j + poff, as the source.\n"
     "__device__ __forceinline__ int dl_stripe_die0_pages(int np, int stripe) {\n"
     "  return (np / (2 * stripe)) * stripe + min(np % (2 * stripe), stripe);\n"
     "}\n"
     "__device__ __forceinline__ int dl_page_pos(int j, int pstride, int poff, int stripe) {\n"
     "  if (pstride == 1) return j;\n"
     "  if (stripe == 1) return 2 * j + poff;\n"
     "  return (j / stripe) * 2 * stripe + poff * stripe + j % stripe;\n"
     "}\n"
     "\n"
     "// Stream (b, h) in die d's page space: start s0 and n pages; its pieces = chunks of die d it falls into; the "
     "partial\n"),
    (KERNEL,
     "        c[0] = !prm.sm_die ? 2 * np : one ? (d1 ? 0 : 2 * np) : 2 * ((np + 1) >> 1);\n"
     "        c[1] = !prm.sm_die ? 0 : one ? (d1 ? 2 * np : 0) : 2 * (np >> 1);\n",
     "        // H211: pages on die 0 under striping (die_stripe 1, phase 0: (np + 1) >> 1); see Params::die_stripe\n"
     "        const int ne = dl_stripe_die0_pages(np, prm.die_stripe);\n"
     "        const bool ph1 = prm.sm_die && prm.die_split == 0 && prm.die_stripe_phase > 0 && np > 0 &&\n"
     "                         __ldg(prm.page_table + static_cast<int64_t>(b) * prm.pt_stride) >= prm.die_stripe_phase;\n"
     "        const int n0 = ph1 ? np - ne : ne;\n"
     "        c[0] = !prm.sm_die ? 2 * np : one ? (d1 ? 0 : 2 * np) : 2 * n0;\n"
     "        c[1] = !prm.sm_die ? 0 : one ? (d1 ? 2 * np : 0) : 2 * (np - n0);\n"),
    (KERNEL,
     "        sg.poff = striped ? d : 0;\n",
     "        // H211: a sequence whose first run is on die 1 has die d's pages in the runs of the other parity\n"
     "        const int ph1 = striped && prm.die_stripe_phase > 0 &&\n"
     "                        __ldg(prm.page_table + static_cast<int64_t>(b) * prm.pt_stride) >= prm.die_stripe_phase;\n"
     "        sg.poff = striped ? (d ^ ph1) : 0;\n"),
    (KERNEL,
     "          const int id = lane < n ? __ldg(pt + sg.pstride * (pb + lane) + sg.poff) : 0;\n",
     "          const int id = lane < n ? __ldg(pt + dl_page_pos(pb + lane, sg.pstride, sg.poff, prm.die_stripe)) : 0;\n"),
    (KERNEL,
     "        const int p = cur.pstride * (cur.p0 + jj) + cur.poff, buf = g & 1;\n",
     "        const int p = dl_page_pos(cur.p0 + jj, cur.pstride, cur.poff, prm.die_stripe), buf = g & 1;\n"),
    (KERNEL,
     "Q36K_API int q36k_attn_decode_dl(void* out, const void* q_fp8, const void* kv_cache, int64_t num_pages,\n",
     "static int attn_decode_dl_impl(void* out, const void* q_fp8, const void* kv_cache, int64_t num_pages,\n"),
    (KERNEL,
     "                                 void* workspace, size_t workspace_bytes, cudaStream_t stream) {\n"
     "  if (num_seqs == 0) return Q36K_OK;\n",
     "                                 void* workspace, size_t workspace_bytes, cudaStream_t stream,\n"
     "                                 int64_t die_stripe_pages, int64_t die_stripe_phase_page) {\n"
     "  if (num_seqs == 0) return Q36K_OK;\n"
     "  if (die_stripe_pages < 1 || die_stripe_pages > (1 << 20) || die_stripe_phase_page < 0 ||  // H211\n"
     "      die_stripe_phase_page >= num_pages)\n"
     "    return Q36K_ERR_INVALID_ARG;\n"),
    (KERNEL,
     "  prm.die_split = static_cast<int>(die_split_page);\n",
     "  prm.die_split = static_cast<int>(die_split_page);\n"
     "  prm.die_stripe = static_cast<int>(die_stripe_pages);  // H211\n"
     "  prm.die_stripe_phase = static_cast<int>(die_stripe_phase_page);\n"),
    (KERNEL,
     "  const cudaError_t e = cudaLaunchKernelEx(&c.cfg, attn_dl_kernel, map, prm);\n"
     "  return q36k::cuda_status(e);\n"
     "}\n",
     "  const cudaError_t e = cudaLaunchKernelEx(&c.cfg, attn_dl_kernel, map, prm);\n"
     "  return q36k::cuda_status(e);\n"
     "}\n"
     "\n"
     "// ---- H211 (vLLM port): exported entry points -------------------------------------------------------------------\n"
     "// q36k_attn_decode_dl keeps the source's ABI and page striping (die_stripe 1, phase 0). h211_attn_decode_dl_striped\n"
     "// takes die_stripe_pages and die_stripe_phase_page: striped placement in runs of that many pages, page p of a\n"
     "// sequence on die (p / die_stripe_pages + phase) % 2, phase 1 for a sequence whose first page id is\n"
     "// >= die_stripe_phase_page > 0 (Params::die_stripe).\n"
     "Q36K_API int q36k_attn_decode_dl(void* out, const void* q_fp8, const void* kv_cache, int64_t num_pages,\n"
     "                                 const int32_t* page_table, int64_t page_table_stride, const int32_t* kv_lens,\n"
     "                                 const int32_t* cu_seqlens_q, int64_t num_seqs, int64_t num_tokens, "
     "int64_t max_q_len,\n"
     "                                 const uint8_t* sm_die, int64_t die_split_page, int64_t chunks_per_sm, "
     "int64_t steal,\n"
     "                                 void* workspace, size_t workspace_bytes, cudaStream_t stream) {\n"
     "  return attn_decode_dl_impl(out, q_fp8, kv_cache, num_pages, page_table, page_table_stride, kv_lens, "
     "cu_seqlens_q,\n"
     "                             num_seqs, num_tokens, max_q_len, sm_die, die_split_page, chunks_per_sm, steal, "
     "workspace,\n"
     "                             workspace_bytes, stream, 1, 0);\n"
     "}\n"
     "\n"
     "Q36K_API int h211_attn_decode_dl_striped(void* out, const void* q_fp8, const void* kv_cache, int64_t num_pages,\n"
     "                                         const int32_t* page_table, int64_t page_table_stride,\n"
     "                                         const int32_t* kv_lens, const int32_t* cu_seqlens_q, int64_t num_seqs,\n"
     "                                         int64_t num_tokens, int64_t max_q_len, const uint8_t* sm_die,\n"
     "                                         int64_t die_split_page, int64_t die_stripe_pages,\n"
     "                                         int64_t die_stripe_phase_page, int64_t chunks_per_sm, int64_t steal,\n"
     "                                         void* workspace, size_t workspace_bytes, cudaStream_t stream) {\n"
     "  return attn_decode_dl_impl(out, q_fp8, kv_cache, num_pages, page_table, page_table_stride, kv_lens, "
     "cu_seqlens_q,\n"
     "                             num_seqs, num_tokens, max_q_len, sm_die, die_split_page, chunks_per_sm, steal, "
     "workspace,\n"
     "                             workspace_bytes, stream, die_stripe_pages, die_stripe_phase_page);\n"
     "}\n"),
)
_TAIL_PATCH = len(STRIPE_PATCHES) - 1  # appends: its `old` must end the file

SOURCES_COMPILED = ("common/common.cu", "common/die.cu", KERNEL)
NVCC_FLAGS = ("-O3", "-std=c++20", "-lineinfo", "--expt-relaxed-constexpr", "-shared", "-Xptxas", "-v",
              "-Xcompiler", "-fPIC,-fvisibility=hidden")
LIB_NAME = "libh211_attn_dl.so"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


def verbatim_sources() -> dict[str, str]:
    """The kernel source files, each checked against its recorded sha256."""
    from vllm.third_party.q36k_attn_dl import sources

    out = {}
    for path, text in sources.FILES.items():
        got = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if got != sources.SHA256[path]:
            raise RuntimeError(f"H211: kernel source {path} sha256 {got[:16]} != recorded {sources.SHA256[path][:16]}")
        out[path] = text
    return out


def apply_patches(files: dict[str, str]) -> dict[str, str]:
    """``files`` with STRIPE_PATCHES applied (each `old` exactly once; the last one at the end of its file)."""
    out = dict(files)
    for i, (path, old, new) in enumerate(STRIPE_PATCHES):
        n = out[path].count(old)
        if n != 1:
            raise RuntimeError(f"H211 stripe patch {i}: {n} matches in {path} for {old[:70]!r}")
        if i == _TAIL_PATCH and not out[path].endswith(old):
            raise RuntimeError(f"H211 stripe patch {i}: not at the end of {path}")
        out[path] = out[path].replace(old, new)
    return out


def patched_sources() -> dict[str, str]:
    return apply_patches(verbatim_sources())


def _nvcc() -> str:
    cands = [os.environ.get(ENV_NVCC), shutil.which("nvcc")]
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME:
            cands.append(os.path.join(CUDA_HOME, "bin", "nvcc"))
    except Exception:  # noqa: BLE001 - only a candidate
        pass
    cands.append("/usr/local/cuda/bin/nvcc")
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise RuntimeError(f"H211: no nvcc (tried {[c for c in cands if c]}; set {ENV_NVCC})")


def _arch(device: torch.device | None) -> str:
    maj, mnr = torch.cuda.get_device_capability(device)
    return f"{maj}{mnr}a"


def _link_libcuda(cuda_root: str) -> list[str]:
    """Linker inputs for the driver API the kernel's host code calls (cuTensorMapEncodeTiled): the toolkit's stub
    library, else the libcuda.so.1 this process has mapped (torch initialized CUDA)."""
    for sub in ("lib64/stubs", "lib/stubs", "targets/sbsa-linux/lib/stubs", "targets/x86_64-linux/lib/stubs"):
        d = os.path.join(cuda_root, sub)
        if os.path.isfile(os.path.join(d, "libcuda.so")):
            return [f"-L{d}", "-lcuda"]
    try:
        with open("/proc/self/maps") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) >= 6 and os.path.basename(parts[-1]).startswith("libcuda.so"):
                    return ["-Xlinker", parts[-1]]
    except OSError:
        pass
    return ["-lcuda"]


def build_command(nvcc: str, arch: str, src_root: str, out: str, link: list[str]) -> list[str]:
    return [nvcc, *NVCC_FLAGS, f"-gencode=arch=compute_{arch},code=sm_{arch}", "-I",
            os.path.join(src_root, "include"), *[os.path.join(src_root, s) for s in SOURCES_COMPILED], "-o", out,
            *link]


def _cache_root() -> str:
    root = os.environ.get(ENV_CACHE)
    if not root:
        base = os.environ.get("VLLM_CACHE_ROOT") or os.path.join(os.path.expanduser("~"), ".cache", "vllm")
        root = os.path.join(base, "h211_attn_dl")
    return root


def build(device: torch.device | None = None) -> tuple[str, dict[str, Any]]:
    """Compile the patched sources into a shared library (once per key, process- and rank-safe); returns its path
    and what was built."""
    files = patched_sources()
    nvcc = _nvcc()
    arch = _arch(device)
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=120).stdout.strip()
    cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    link = _link_libcuda(cuda_root)
    key = hashlib.sha256(json.dumps(
        {"files": {p: hashlib.sha256(t.encode()).hexdigest() for p, t in sorted(files.items())},
         "flags": NVCC_FLAGS, "arch": arch, "nvcc": ver, "link": link}, sort_keys=True).encode()).hexdigest()
    root = _cache_root()
    final = os.path.join(root, f"h211-attn-dl-{key[:16]}")
    lib = os.path.join(final, LIB_NAME)
    info: dict[str, Any] = {"key": key[:16], "arch": f"sm_{arch}", "nvcc": nvcc, "nvcc_version": ver.splitlines()[-1:],
                            "link": link, "dir": final,
                            "kernel_sha256_patched": hashlib.sha256(files[KERNEL].encode()).hexdigest()[:16]}
    if os.path.isfile(lib):
        return lib, dict(info, built=False)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, ".h211-attn-dl.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.isfile(lib):
            return lib, dict(info, built=False)
        tmp = tempfile.mkdtemp(prefix=".h211-build-", dir=root)
        src_root = os.path.join(tmp, "csrc")
        for p, t in files.items():
            os.makedirs(os.path.dirname(os.path.join(src_root, p)), exist_ok=True)
            with open(os.path.join(src_root, p), "w", encoding="utf-8", newline="\n") as f:
                f.write(t)
        cmd = build_command(nvcc, arch, src_root, os.path.join(tmp, LIB_NAME), link)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        with open(os.path.join(tmp, "build.json"), "w") as f:
            json.dump({"cmd": cmd, "rc": r.returncode, "stderr": r.stderr[-20000:], **info}, f, indent=1)
        if r.returncode:
            raise RuntimeError(f"H211: nvcc failed (rc {r.returncode}): {r.stderr[-3000:]}")
        os.replace(tmp, final)
    return lib, dict(info, built=True)


class _Lib:
    def __init__(self, path: str):
        vp, i64, sz, cint = ctypes.c_void_p, ctypes.c_int64, ctypes.c_size_t, ctypes.c_int
        lib = ctypes.CDLL(path)
        lib.q36k_attn_decode_dl_workspace_size.argtypes, lib.q36k_attn_decode_dl_workspace_size.restype = [i64], sz
        lib.h211_attn_decode_dl_striped.argtypes = [vp, vp, vp, i64, vp, i64, vp, vp, i64, i64, i64, vp, i64, i64,
                                                    i64, i64, i64, vp, sz, vp]
        lib.h211_attn_decode_dl_striped.restype = cint
        lib.q36k_attn_decode_dl.argtypes = [vp, vp, vp, i64, vp, i64, vp, vp, i64, i64, i64, vp, i64, i64, i64, vp,
                                            sz, vp]
        lib.q36k_attn_decode_dl.restype = cint
        lib.q36k_die_probe.argtypes, lib.q36k_die_probe.restype = [vp, sz, vp, vp, vp], cint
        lib.q36k_pdl_enabled.argtypes, lib.q36k_pdl_enabled.restype = [], cint
        lib.q36k_set_pdl.argtypes, lib.q36k_set_pdl.restype = [cint], None
        self.lib, self.path = lib, path

    def workspace_size(self, max_seqs: int) -> int:
        return int(self.lib.q36k_attn_decode_dl_workspace_size(max_seqs))


def sm_die_from_dies(dies: dict[int, int]) -> list[int]:
    """SM -> die table entries (0 / 1; 0xFF for SM ids the probe did not see)."""
    table = [0xFF] * MAX_SMS
    for sm, d in dies.items():
        if 0 <= sm < MAX_SMS:
            table[sm] = 1 if d else 0
    return table


def probe_sm_die_vllm(device: torch.device) -> tuple[list[int], dict[str, Any]]:
    """SM -> die from die_local_kv's latency probe (the one the served die-local KV path uses)."""
    from vllm.v1.worker import die_local_kv as dl

    opened = dl._open(device)
    if isinstance(opened, str):
        raise RuntimeError(f"H211: {opened}")
    cu, cudev = opened
    res = dl._DevResource()
    dl._check(cu.cuDeviceGetDevResource(cudev, ctypes.byref(res), 1), "SM resource")
    probe = dl._probe_dies(cu, cudev, device, res.sm.smCount)
    table = sm_die_from_dies(probe.dies)
    return table, {"source": "vllm die_local_kv probe", "sms": len(probe.dies),
                   "per_die": [table.count(0), table.count(1)], "near_cycles": probe.near,
                   "far_cycles": probe.far, "min_ratio": round(probe.min_ratio, 3)}


def probe_sm_die_q36k(lib: _Lib, device: torch.device) -> tuple[list[int], dict[str, Any]]:
    """SM -> die from the vendored q36k_die_probe over a die-localized pool (64 MiB per die); idle device only."""
    from vllm.v1.worker import die_local_kv as dl

    opened = dl._open(device)
    if isinstance(opened, str):
        raise RuntimeError(f"H211: {opened}")
    cu, cudev = opened
    gran = dl._granularity(cu, cudev, 0)
    per = -(-(64 << 20) // gran)
    va, nbytes, chunks = dl._map(cu, cudev, [0] * per + [1] * per, gran)
    try:
        table = torch.full((MAX_SMS,), 0xFF, dtype=torch.uint8, device=device)
        lat = torch.zeros((MAX_SMS, 2), dtype=torch.float32, device=device)
        stream = torch.cuda.current_stream(device).cuda_stream
        rc = lib.lib.q36k_die_probe(ctypes.c_void_p(va), per * gran, ctypes.c_void_p(table.data_ptr()),
                                    ctypes.c_void_p(lat.data_ptr()), ctypes.c_void_p(stream))
        torch.cuda.synchronize(device)
        if rc:
            raise RuntimeError(f"H211: q36k_die_probe: {STATUS.get(rc, rc)}")
        t = [int(x) for x in table.cpu().tolist()]
        lat_h = lat.cpu().tolist()
    finally:
        torch.cuda.synchronize(device)
        dl._free(cu, va, nbytes, chunks)
    seen = [i for i, d in enumerate(t) if d in (0, 1)]
    near = sorted(min(lat_h[i]) for i in seen)
    far = sorted(max(lat_h[i]) for i in seen)
    return t, {"source": "q36k_die_probe", "sms": len(seen), "per_die": [t.count(0), t.count(1)],
               "near_cycles": near[len(near) // 2] if near else None, "far_cycles": far[len(far) // 2] if far else None}


def kv_view(kv_cache: Any) -> tuple[int, int] | str:
    """(base pointer, kernel pages) of the per-layer KV cache [P, 2, 128, 512] (K|V per token), given as the tensor
    or as its (K, V) views; else why the kernel cannot read it."""
    want = (PAGE_BYTES, PAGE * 2 * HDIM, 2 * HDIM, 1)
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) != 2:
            return "kv_cache tuple is not (K, V)"
        k, v = kv_cache
        if k.element_size() != 1 or v.element_size() != 1:
            return f"KV element size {k.element_size()} (FP8 only)"
        if tuple(k.shape[1:]) != (HKV, PAGE, HDIM) or tuple(k.stride()) != want:
            return f"K view shape {tuple(k.shape)} strides {tuple(k.stride())}"
        if v.shape != k.shape or v.stride() != k.stride() or v.data_ptr() != k.data_ptr() + HDIM:
            return "V view is not K's packed partner"
        return k.data_ptr(), int(k.shape[0])
    t = kv_cache
    if t.element_size() != 1:
        return f"KV element size {t.element_size()} (FP8 only)"
    if tuple(t.shape[1:]) != (HKV, PAGE, 2 * HDIM) or tuple(t.stride()) != want:
        return f"KV shape {tuple(t.shape)} strides {tuple(t.stride())}"
    return t.data_ptr(), int(t.shape[0])


class DieVerifyAttn:
    """The built kernel with its workspace (one per device; launches on a stream never overlap) and SM -> die
    table (None: one work queue, interleaved KV)."""

    def __init__(self, lib: _Lib, device: torch.device, max_seqs: int, sm_die: list[int] | None,
                 info: dict[str, Any] | None = None):
        self.lib, self.device = lib, torch.device(device)
        self.max_seqs = max(1, min(int(max_seqs), MAX_SEQS))
        self.workspace_bytes = lib.workspace_size(self.max_seqs)
        # The counters (COUNTER_OFFSET) must be zero before the first launch; every launch resets them.
        self.workspace = torch.zeros(self.workspace_bytes, dtype=torch.uint8, device=self.device)
        self.sm_die = (None if sm_die is None else
                       torch.tensor(sm_die, dtype=torch.uint8, device=self.device))
        self.info = dict(info or {})
        self._seen: set[tuple[int, int, int, int, bool]] = set()
        # cu_seqlens_q of uniform batches: row q - 1 is [0, q, 2 q, ...] (made here, outside CUDA-graph capture, so
        # a launch only slices it).
        ar = torch.arange(self.max_seqs + 1, dtype=torch.int32, device=self.device)
        self.cu_q = torch.stack([ar * q for q in range(1, MAX_Q + 1)])

    def cu_uniform(self, q_len: int, num_seqs: int) -> torch.Tensor:
        """cu_seqlens_q [num_seqs + 1] of a batch with ``q_len`` query rows per sequence (a slice, no launch)."""
        return self.cu_q[q_len - 1, :num_seqs + 1]

    def set_sm_die(self, sm_die: list[int] | None) -> None:
        self.sm_die = (None if sm_die is None else
                       torch.tensor(sm_die, dtype=torch.uint8, device=self.device))

    def counters_zero(self) -> bool:
        """The self-resetting counters are all zero (between launches; syncs)."""
        c = self.workspace[COUNTER_OFFSET:COUNTER_OFFSET + COUNTER_BYTES]
        return bool((c == 0).all().item())

    def check(self, out: torch.Tensor, q: torch.Tensor, kv_cache: Any, page_table: torch.Tensor,
              kv_lens: torch.Tensor, cu_seqlens_q: torch.Tensor, num_seqs: int, max_q_len: int,
              bmm1_scale: float = SM_SCALE, bmm2_scale: float = 1.0) -> str | None:
        """Why this call is outside the kernel's contract, or None."""
        if torch.cuda.get_device_capability(self.device) != CAPABILITY:
            return f"device capability {torch.cuda.get_device_capability(self.device)} (needs {CAPABILITY})"
        if abs(bmm1_scale - SM_SCALE) > 1e-6 * SM_SCALE or bmm2_scale != 1.0:
            return f"scales bmm1 {bmm1_scale} bmm2 {bmm2_scale} (needs 1/16 and 1: unit q/k/v scales)"
        if not 0 <= num_seqs <= self.max_seqs:
            return f"{num_seqs} sequences (workspace for {self.max_seqs})"
        if not 1 <= max_q_len <= MAX_Q:
            return f"max_q_len {max_q_len} (1..{MAX_Q})"
        if q.dtype != torch.float8_e4m3fn or not q.is_contiguous() or tuple(q.shape[1:]) != (HQ, HDIM):
            return f"q {q.dtype} {tuple(q.shape)} contiguous={q.is_contiguous()}"
        if (out.dtype != torch.bfloat16 or not out.is_contiguous() or out.numel() < q.numel()
                or out.shape[-1] not in (HDIM, HQ * HDIM)):
            return f"out {out.dtype} {tuple(out.shape)} contiguous={out.is_contiguous()}"
        kv = kv_view(kv_cache)
        if isinstance(kv, str):
            return kv
        if page_table.dtype != torch.int32 or page_table.dim() != 2 or page_table.stride(1) != 1:
            return f"page table {page_table.dtype} {tuple(page_table.shape)} strides {tuple(page_table.stride())}"
        for name, t, n in (("kv_lens", kv_lens, num_seqs), ("cu_seqlens_q", cu_seqlens_q, num_seqs + 1)):
            if t.dtype != torch.int32 or not t.is_contiguous() or t.numel() < n:
                return f"{name} {t.dtype} {tuple(t.shape)}"
        return None

    def run(self, out: torch.Tensor, q: torch.Tensor, kv_cache: Any, page_table: torch.Tensor,
            kv_lens: torch.Tensor, cu_seqlens_q: torch.Tensor, num_seqs: int, max_q_len: int, *,
            die_split_page: int = 0, stripe_pages: int = 1, stripe_phase_page: int = 0, chunks_per_sm: int = 1,
            steal: int = 0, die_aware: bool = True, stream: int | None = None) -> None:
        """One launch (capturable). Striped placement (die_split_page 0): runs of ``stripe_pages`` pages alternate
        dies, a sequence's first run on die 1 when its first page id is >= ``stripe_phase_page`` > 0, else on die 0.
        ``die_aware=False`` passes no SM -> die table: one queue, interleaved KV."""
        kv = kv_view(kv_cache)
        if isinstance(kv, str):
            raise ValueError(f"H211: {kv}")
        base, pages = kv
        sm_die = self.sm_die if die_aware else None
        if stream is None:
            stream = torch.cuda.current_stream(self.device).cuda_stream
        rc = self.lib.lib.h211_attn_decode_dl_striped(
            ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(base), pages,
            ctypes.c_void_p(page_table.data_ptr()), page_table.stride(0), ctypes.c_void_p(kv_lens.data_ptr()),
            ctypes.c_void_p(cu_seqlens_q.data_ptr()), num_seqs, q.shape[0], max_q_len,
            ctypes.c_void_p(None if sm_die is None else sm_die.data_ptr()), die_split_page, stripe_pages,
            stripe_phase_page, chunks_per_sm, steal, ctypes.c_void_p(self.workspace.data_ptr()),
            self.workspace_bytes, ctypes.c_void_p(stream))
        if rc:
            raise RuntimeError(f"H211: q36k_attn_decode_dl: {STATUS.get(rc, rc)} (B {num_seqs}, q {max_q_len}, "
                               f"pages {pages}, split {die_split_page}, stripe {stripe_pages} phase "
                               f"{stripe_phase_page}, chunks/SM {chunks_per_sm}, steal {steal})")
        key = (die_split_page, stripe_pages, stripe_phase_page > 0, chunks_per_sm, steal, sm_die is not None)
        if key not in self._seen:
            self._seen.add(key)
            logger.info("H211 die-verify: first launch B=%d q<=%d pages=%d die_aware=%s die_split_page=%d "
                        "stripe_pages=%d stripe_phase_page=%d chunks_per_sm=%d steal=%d (PDL %s)", num_seqs,
                        max_q_len, pages, sm_die is not None, die_split_page, stripe_pages, stripe_phase_page,
                        chunks_per_sm, steal, bool(self.lib.lib.q36k_pdl_enabled()))


_RUNTIMES: dict[int, DieVerifyAttn] = {}
_LOCK = threading.Lock()


def supported(device: torch.device) -> str | None:
    if not torch.cuda.is_available():
        return "no CUDA"
    cap = torch.cuda.get_device_capability(device)
    if cap != CAPABILITY:
        return f"capability {cap} (the kernel is sm_107a only)"
    if torch.cuda.get_device_properties(device).multi_processor_count > MAX_SMS:
        return "more SMs than the kernel's die table"
    return None


def runtime_if_ready(device: torch.device) -> DieVerifyAttn | None:
    """The kernel ``get_runtime`` made for ``device``, or None; never builds (safe during CUDA-graph capture)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _RUNTIMES.get(index)


@dataclass(frozen=True)
class DispatchConfig:
    """The serving dispatch's settings: launches of at least ``min_rows`` rows on ``parts``."""

    min_rows: int
    parts: frozenset[str]


def parse_dispatch(flag: str | None, min_rows: str | None, parts: str | None) -> tuple[DispatchConfig | None, str]:
    """(settings or None when off, the line to log) from the environment's raw values."""
    if flag != "1":
        return None, ""
    raw = (min_rows or "").strip()
    try:
        rows = int(raw) if raw else 1
    except ValueError:
        return None, f"{ENV_MIN_ROWS}={raw!r} is not an integer: dispatch off"
    if rows < 1:
        return None, f"{ENV_MIN_ROWS}={raw!r} is below 1: dispatch off"
    p = "t" if parts is None or not parts.strip() else parts.replace(",", "").replace(" ", "")
    if set(p) - set(PARTS):
        return None, f"{ENV_PARTS}={parts!r} is not t, d or t,d: dispatch off"
    cfg = DispatchConfig(rows, frozenset(p))
    return cfg, (f"decode launches of >= {rows} rows" + ("" if raw else f" ({ENV_MIN_ROWS} unset: every launch)") +
                 f" with q_len <= {MAX_Q} on parts {','.join(x for x in PARTS if x in cfg.parts)} run "
                 "q36k_attn_decode_dl on striped die-local KV (grid = SMs, chunks_per_sm 1); the others trtllm-gen")


_DISPATCH: list[DispatchConfig | None] = []


def dispatch_config() -> DispatchConfig | None:
    """The serving dispatch's settings (parsed and logged once per process), or None when VLLM_H211_DL_VERIFY is
    not 1 or a setting is invalid."""
    if not _DISPATCH:
        cfg, msg = parse_dispatch(os.environ.get(ENV_FLAG), os.environ.get(ENV_MIN_ROWS), os.environ.get(ENV_PARTS))
        _DISPATCH.append(cfg)
        if msg:
            (logger.info if cfg is not None else logger.warning)("H211 die-verify dispatch: %s", msg)
    return _DISPATCH[0]


def check_count(raw: str | None) -> int:
    """VLLM_H211_DL_VERIFY_CHECK: eager launches per layer to compare with trtllm-gen (unset or invalid: 0)."""
    try:
        return max(int((raw or "0").strip() or "0"), 0)
    except ValueError:
        return 0


def decide(cfg: DispatchConfig, part: str, num_reqs: int, max_q_len: int, max_seqs: int) -> str | None:
    """Why a decode launch of ``num_reqs`` rows (``max_q_len`` query rows at most) on ``part`` stays on trtllm-gen;
    None: it runs the die-aware kernel."""
    if part not in cfg.parts:
        return f"part {part}"
    if num_reqs < cfg.min_rows:
        return f"rows < {cfg.min_rows}"
    if not 1 <= max_q_len <= MAX_Q:
        return f"q_len {max_q_len}"
    if num_reqs > max_seqs:
        return f"rows > {max_seqs}"
    return None


_CHOICES: set[tuple[str, int, int, str | None]] = set()
_MAX_CHOICES = 128


def note_choice(part: str, num_reqs: int, max_q_len: int, why: str | None) -> None:
    """Log each (part, rows, q_len, decision) once (the first 128; CUDA-graph capture logs every captured size)."""
    key = (part, num_reqs, max_q_len, why)
    if key in _CHOICES or len(_CHOICES) > _MAX_CHOICES:
        return
    _CHOICES.add(key)
    if len(_CHOICES) > _MAX_CHOICES:
        logger.info("H211 die-verify: %d decisions logged; no more", _MAX_CHOICES)
        return
    logger.info("H211 die-verify: decode launch %s B=%d q<=%d -> %s", part, num_reqs, max_q_len,
                "die-aware kernel" if why is None else f"trtllm-gen ({why})")


def get_runtime(device: torch.device, max_seqs: int = MAX_SEQS, sm_die_source: str = "vllm",
                force: bool = False) -> DieVerifyAttn | None:
    """The kernel for ``device`` (built and probed once per process), or None when ``VLLM_H211_DL_VERIFY`` is not 1
    (``force``: the microbenchmark, which asks for it explicitly) or the device cannot run it (logged once)."""
    if not (force or enabled()):
        return None
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    with _LOCK:
        rt = _RUNTIMES.get(index)
        if rt is not None:
            return rt
        why = supported(device)
        if why:
            logger.warning("H211 die-verify: off on this device: %s", why)
            return None
        path, info = build(device)
        lib = _Lib(path)
        if sm_die_source == "q36k":
            table, pinfo = probe_sm_die_q36k(lib, device)
        elif sm_die_source == "none":
            table, pinfo = None, {"source": "none (interleaved KV)"}
        else:
            table, pinfo = probe_sm_die_vllm(device)
        info = dict(info, sm_die=pinfo)
        rt = DieVerifyAttn(lib, device, max_seqs, table, info)
        _RUNTIMES[index] = rt
        logger.info("H211 die-verify: q36k_attn_decode_dl %s (%s, key %s, kernel %s after %d stripe edits); "
                    "workspace %.1f MiB for %d sequences; SM -> die %s",
                    "built" if info.get("built") else "loaded", info["arch"], info["key"],
                    info["kernel_sha256_patched"], len(STRIPE_PATCHES), rt.workspace_bytes / 2**20, rt.max_seqs,
                    json.dumps(pinfo))
        return rt
