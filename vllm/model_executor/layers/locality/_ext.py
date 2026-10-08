# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""CUDA extension of the locality package (JIT-built with nvcc on first use).

SM107 (sm_107) supports locality domains with domain-local memory placement.
Domain-aware kernels can read locally placed buffers without crossing the
inter-domain fabric.

The source is kept in this module (``_SOURCE``) so the package ships only
Python; ``load()`` builds it with torch.utils.cpp_extension for the device's
arch-specific target under VLLM_CACHE_ROOT (concurrent processes share the
build directory via cpp_extension's file lock). It links libcuda (driver API).

Exported functions (all take/return torch tensors or ints):

- ``device_info(dev) -> [num_sms, num_domains, sms_per_domain]`` (device
  attributes 149 / 157; -1 when the driver does not report them).
- ``green_sm_map(dev) -> int8[num_sms]``: domain of every SM that a per-domain
  green context (``cuDevSmResourceSplit`` with
  ``CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID``) contains, -1 for the SMs left
  in the remainder (the unassigned TPCs). The green contexts are destroyed again.
- ``probe_sm_latency(buf0, buf1, flush) -> float32[num_sms, 2]``: cycles per
  dependent DRAM-miss load into domain-local buffers, using per-SM pointer
  chains after flushing L2. The relative latency identifies the domain of
  unassigned SMs.
- ``alloc_localized(nbytes, chunk_domains, chunk_bytes, dev) -> uint8 tensor``:
  one VA range; each chunk is aligned to the required allocation granularity
  and backed by a ``cuMemCreate`` allocation on locality domain
  ``chunk_domains[i]`` (``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN``),
  mapped with read/write access for ``dev``. The tensor's deleter synchronizes
  the device, unmaps and frees the range.
- ``granularity(dev)``, ``pointer_domain(ptr)``, ``chunk_ordinals(ptr, nbytes,
  chunk_bytes)`` (``CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL``; -1 for
  interleaved memory).
- ``green_streams(dev, extra0=0, extra1=0) -> [stream0, stream1, sms0, sms1,
  remainder, coscheduled]``: a persistent green context + non-blocking stream
  per domain (process lifetime, one set per argument tuple). ``extra*`` > 0
  makes group k a BACKFILL group of (domain-k SMs + extra_k) SMs. The driver
  chooses remainder SMs; ``streams.get_domain_streams`` requests extras from
  the topology probe and reports the actual binding.
- ``stream_smids(dev, stream) -> int32[num_sms]``: the SMs a kernel launched on
  ``stream`` reaches (binding check).
- ``lm_bf16(x, w, out, sm_dom, tiles0, tiles1, ctrs, warps, cfg)``: the
  domain-aware skinny GEMM ``out[M, N] = x[M, K] @ w[N, K]^T`` for a BF16
  weight (see the kernel comment); bf16 output, fp32 accumulation. (The MXFP8
  draft head uses the tcgen05 kernel in ``mxgemm.py``.)
"""

from __future__ import annotations

import hashlib
import os

import torch

_ext: list = []


def _cache_root() -> str:
    try:
        from vllm import envs

        return envs.VLLM_CACHE_ROOT
    except Exception:  # standalone use (microbenchmarks)
        return os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))


def build(name: str, source: str):
    """JIT-build ``source`` (a torch extension) for the current device's
    arch-specific target, cached under VLLM_CACHE_ROOT (concurrent processes
    share the build directory via cpp_extension's file lock).
    """
    major, minor = torch.cuda.get_device_capability()
    import torch.utils.cpp_extension as cpp

    arch = f"{major}{minor}{'a' if major >= 9 else ''}"
    bdir = os.path.join(_cache_root(), "locality_ext", f"sm{arch}")
    os.makedirs(bdir, exist_ok=True)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    src = os.path.join(bdir, f"{name}_{digest}.cu")
    if not os.path.exists(src):
        tmp = f"{src}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(source)
        os.replace(tmp, src)
    orig = cpp._get_cuda_arch_flags
    cpp._get_cuda_arch_flags = lambda cflags=None: [
        f"-gencode=arch=compute_{arch},code=sm_{arch}"
    ]
    try:
        return cpp.load(
            name=f"_{name}_{digest}",
            sources=[src],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"],
            extra_cflags=["-O3", "-std=c++20"],
            extra_ldflags=["-lcuda"],
            build_directory=bdir,
            verbose=False,
        )
    finally:
        cpp._get_cuda_arch_flags = orig


def load():
    """Build (or load the cached build of) the extension for the current device."""
    if not _ext:
        _ext.append(build("locality", _SOURCE))
    return _ext[0]


_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include <tuple>
#include <unordered_map>
#include <type_traits>

namespace loc {

#define DRV(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorName(r_, &s_); TORCH_CHECK(false, #x " failed: ", s_); } } while (0)
#define RTC(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { cudaGetLastError(); TORCH_CHECK(false, #x " failed: ", cudaGetErrorString(e_)); } } while (0)

static CUdevice cu_dev(int64_t dev) {
  RTC(cudaSetDevice((int)dev));
  RTC(cudaFree(0));  // primary context current for the driver API
  CUdevice d; DRV(cuDeviceGet(&d, (int)dev));
  return d;
}

__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }

// ------------------------------------------------------------------ topology
std::vector<int64_t> device_info(int64_t dev) {
  CUdevice d = cu_dev(dev);
  int nsm = 0, nld = -1, ldsm = -1;
  DRV(cuDeviceGetAttribute(&nsm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d));
  if (cuDeviceGetAttribute(&nld, (CUdevice_attribute)149, d) != CUDA_SUCCESS) nld = -1;   // LOCALITY_DOMAIN_COUNT
  if (cuDeviceGetAttribute(&ldsm, (CUdevice_attribute)157, d) != CUDA_SUCCESS) ldsm = -1; // LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT
  return {nsm, nld, ldsm};
}

__global__ void k_record_smid(int* seen) { if (threadIdx.x == 0) seen[smid()] = 1; }

// Per-domain SM split. extra == nullptr (or both <= 0): the domain SMs only (the map probe below needs exactly this).
// extra[k] > 0: group k requests its domain SMs plus extra[k] SMs via BACKFILL.
// BACKFILL takes the domain's own SMs first, then unassigned SMs, then another domain's.
// The caller requests extras based on the topology probe; actual binding must be checked.
// Try TPC-granular coscheduling first, then the driver default if rejected.
static int split_domains(CUdevice d, CUdevResource part[2], CUdevResource* rem, const int* extra = nullptr) {
  CUdevResource all; memset(&all, 0, sizeof(all));
  DRV(cuDeviceGetDevResource(d, &all, CU_DEV_RESOURCE_TYPE_SM));
  const bool bf = extra != nullptr && (extra[0] > 0 || extra[1] > 0);
  for (int cs : {2, 0}) {
    CU_DEV_SM_RESOURCE_GROUP_PARAMS gp[2]; memset(gp, 0, sizeof(gp));
    for (int k = 0; k < 2; k++) { gp[k].smCount = 0; gp[k].coscheduledSmCount = cs; gp[k].flags = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID; gp[k].localityDomainId = k; }
    memset(part, 0, 2 * sizeof(CUdevResource)); memset(rem, 0, sizeof(CUdevResource));
    if (cuDevSmResourceSplit(part, 2, &all, rem, 0, gp) != CUDA_SUCCESS) continue;
    if (!bf) return cs;
    const unsigned dom[2] = {part[0].sm.smCount, part[1].sm.smCount};
    for (int k = 0; k < 2; k++) if (extra[k] > 0) { gp[k].flags |= CU_DEV_SM_RESOURCE_GROUP_BACKFILL; gp[k].smCount = dom[k] + (unsigned)extra[k]; }
    memset(part, 0, 2 * sizeof(CUdevResource)); memset(rem, 0, sizeof(CUdevResource));
    if (cuDevSmResourceSplit(part, 2, &all, rem, 0, gp) == CUDA_SUCCESS) return cs;
  }
  TORCH_CHECK(false, "cuDevSmResourceSplit failed (backfill extra ", bf ? extra[0] : 0, "/", bf ? extra[1] : 0, ")");
  return -1;
}

torch::Tensor green_sm_map(int64_t dev) {
  CUdevice d = cu_dev(dev);
  int nsm = 0; DRV(cuDeviceGetAttribute(&nsm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d));
  CUdevResource part[2], rem; split_domains(d, part, &rem);
  auto out = torch::full({nsm}, -1, torch::dtype(torch::kInt8));
  int8_t* o = out.data_ptr<int8_t>();
  int* d_seen; RTC(cudaMalloc(&d_seen, 4 * nsm));
  std::vector<int> seen(nsm);
  for (int k = 0; k < 2; k++) {
    CUdevResourceDesc desc; DRV(cuDevResourceGenerateDesc(&desc, &part[k], 1));
    CUgreenCtx g; DRV(cuGreenCtxCreate(&g, desc, d, CU_GREEN_CTX_DEFAULT_STREAM));
    CUstream st; DRV(cuGreenCtxStreamCreate(&st, g, CU_STREAM_NON_BLOCKING, 0));
    RTC(cudaMemset(d_seen, 0, 4 * nsm)); RTC(cudaDeviceSynchronize());
    k_record_smid<<<64 * nsm, 32, 0, (cudaStream_t)st>>>(d_seen);
    RTC(cudaGetLastError()); RTC(cudaStreamSynchronize((cudaStream_t)st));
    RTC(cudaMemcpy(seen.data(), d_seen, 4 * nsm, cudaMemcpyDeviceToHost));
    for (int s = 0; s < nsm; s++) if (seen[s]) { TORCH_CHECK(o[s] < 0, "SM ", s, " in both green contexts"); o[s] = (int8_t)k; }
    DRV(cuStreamDestroy(st)); DRV(cuGreenCtxDestroy(g));
  }
  RTC(cudaFree(d_seen));
  return out;
}

// persistent per-domain green contexts + streams (process lifetime), one set per (device, extra0, extra1)
struct GreenSet { CUgreenCtx g[2]; CUstream s[2]; int64_t n[3]; int64_t cs; };
static std::map<std::tuple<int64_t, int64_t, int64_t>, GreenSet> g_green;
std::vector<int64_t> green_streams(int64_t dev, int64_t extra0, int64_t extra1) {
  auto key = std::make_tuple(dev, extra0, extra1);
  auto it = g_green.find(key);
  if (it == g_green.end()) {
    CUdevice d = cu_dev(dev);
    const int ex[2] = {(int)extra0, (int)extra1};
    CUdevResource part[2], rem;
    GreenSet gs; memset(&gs, 0, sizeof(gs));
    gs.cs = split_domains(d, part, &rem, ex);
    for (int k = 0; k < 2; k++) {
      CUdevResourceDesc desc; DRV(cuDevResourceGenerateDesc(&desc, &part[k], 1));
      DRV(cuGreenCtxCreate(&gs.g[k], desc, d, CU_GREEN_CTX_DEFAULT_STREAM));
      DRV(cuGreenCtxStreamCreate(&gs.s[k], gs.g[k], CU_STREAM_NON_BLOCKING, 0));
      gs.n[k] = (int64_t)part[k].sm.smCount;
    }
    gs.n[2] = rem.type == CU_DEV_RESOURCE_TYPE_SM ? (int64_t)rem.sm.smCount : 0;
    RTC(cudaSetDevice((int)dev));
    it = g_green.emplace(key, gs).first;
  }
  const GreenSet& gs = it->second;
  return {(int64_t)(uintptr_t)gs.s[0], (int64_t)(uintptr_t)gs.s[1], gs.n[0], gs.n[1], gs.n[2], gs.cs};
}

// SMs a kernel launched on `stream` actually reaches (binding check for green streams): int32[num_sms] of 0/1.
torch::Tensor stream_smids(int64_t dev, int64_t stream) {
  cu_dev(dev);
  int nsm = 0; RTC(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, (int)dev));
  int* d_seen; RTC(cudaMalloc(&d_seen, 4 * nsm));
  RTC(cudaMemset(d_seen, 0, 4 * nsm)); RTC(cudaDeviceSynchronize());
  k_record_smid<<<64 * nsm, 32, 0, (cudaStream_t)(uintptr_t)stream>>>(d_seen);
  RTC(cudaGetLastError()); RTC(cudaStreamSynchronize((cudaStream_t)(uintptr_t)stream));
  auto out = torch::zeros({nsm}, torch::dtype(torch::kInt32));
  RTC(cudaMemcpy(out.data_ptr<int>(), d_seen, 4 * nsm, cudaMemcpyDeviceToHost));
  RTC(cudaFree(d_seen));
  return out;
}

// Pointer chase: per-SM chain of N_PROBE dependent ld.global.cg loads into buf0, then buf1.
constexpr int N_PROBE = 32, PROBE_STRIDE_WORDS = 512;
__global__ void k_chain_init(uint64_t* b) {
  if (threadIdx.x != 0) return;
  const size_t base = (size_t)blockIdx.x * N_PROBE;
  for (int i = 0; i < N_PROBE; i++) b[(base + i) * PROBE_STRIDE_WORDS] = (uint64_t)((base + (i + 1) % N_PROBE) * PROBE_STRIDE_WORDS);
}
__global__ void k_flush(const uint4* p, size_t n, unsigned* sink) {
  unsigned a = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { uint4 v = __ldcs(p + i); a ^= v.x ^ v.w; }
  if (a == 0x7f4a7c15u) *sink = a;
}
__global__ void k_probe(const uint64_t* b0, const uint64_t* b1, float* out) {
  extern __shared__ unsigned char pad[];  // dynamic smem forces one CTA per SM
  if (threadIdx.x != 0) return;
  pad[0] = 0;
  const unsigned s = smid();
  for (int which = 0; which < 2; which++) {
    const uint64_t* b = which ? b1 : b0;
    uint64_t off = (uint64_t)s * N_PROBE * PROBE_STRIDE_WORDS, v;
    long long t0 = clock64();
    for (int i = 0; i < N_PROBE; i++) { asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(b + off) : "memory"); off = v; }
    long long t1 = clock64();
    out[2 * s + which] = (float)(t1 - t0) / N_PROBE + (off == 0xffffffffull ? 1e-9f : 0.f);
  }
}
torch::Tensor probe_sm_latency(torch::Tensor buf0, torch::Tensor buf1, torch::Tensor flush) {
  c10::cuda::CUDAGuard guard(buf0.device());
  int dev = buf0.get_device(), nsm = 0, smem_optin = 0;
  RTC(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  RTC(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  const size_t need = (size_t)nsm * N_PROBE * PROBE_STRIDE_WORDS * 8;
  TORCH_CHECK(buf0.is_cuda() && buf1.is_cuda() && (size_t)buf0.nbytes() >= need && (size_t)buf1.nbytes() >= need, "probe buffers need ", need, " bytes");
  auto st = at::cuda::getCurrentCUDAStream();
  k_chain_init<<<nsm, 32, 0, st>>>((uint64_t*)buf0.data_ptr());
  k_chain_init<<<nsm, 32, 0, st>>>((uint64_t*)buf1.data_ptr());
  auto sink = torch::zeros({16}, buf0.options().dtype(torch::kInt32));
  k_flush<<<4 * nsm, 512, 0, st>>>((const uint4*)flush.data_ptr(), (size_t)flush.nbytes() / 16, (unsigned*)sink.data_ptr());
  auto out = torch::zeros({nsm, 2}, buf0.options().dtype(torch::kFloat32));
  int smem = smem_optin > 0 ? smem_optin : 100 * 1024;
  RTC(cudaFuncSetAttribute(k_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  k_probe<<<nsm, 32, smem, st>>>((const uint64_t*)buf0.data_ptr(), (const uint64_t*)buf1.data_ptr(), out.data_ptr<float>());
  RTC(cudaGetLastError());
  return out;
}

// ------------------------------------------------------------------ localized memory
static CUmemAllocationProp loc_prop(int dev, int dom) {
  CUmemAllocationProp prop; memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  prop.location.localized.deviceId = dev;
  prop.location.localized.localityDomainId = (unsigned char)dom;
  return prop;
}
int64_t granularity(int64_t dev) {
  cu_dev(dev);
  CUmemAllocationProp prop = loc_prop((int)dev, 0);
  size_t g = 0;
  DRV(cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  return (int64_t)g;
}
torch::Tensor alloc_localized(int64_t nbytes, std::vector<int64_t> chunk_domains, int64_t chunk_bytes, int64_t dev) {
  cu_dev(dev);
  const size_t g = (size_t)granularity(dev);
  TORCH_CHECK(chunk_bytes > 0 && chunk_bytes % g == 0, "chunk_bytes must be a multiple of ", g);
  const size_t nchunks = (size_t)((nbytes + chunk_bytes - 1) / chunk_bytes);
  TORCH_CHECK(chunk_domains.size() == nchunks, "need one domain per chunk: ", nchunks);
  const size_t total = nchunks * (size_t)chunk_bytes;
  CUdeviceptr va; DRV(cuMemAddressReserve(&va, total, g, 0, 0));
  for (size_t i = 0; i < nchunks; i++) {
    CUmemAllocationProp prop = loc_prop((int)dev, (int)chunk_domains[i]);
    CUmemGenericAllocationHandle h; DRV(cuMemCreate(&h, (size_t)chunk_bytes, &prop, 0));
    DRV(cuMemMap(va + i * (size_t)chunk_bytes, (size_t)chunk_bytes, 0, h, 0));
    DRV(cuMemRelease(h));  // the mapping keeps the physical memory alive
  }
  CUmemAccessDesc ad; memset(&ad, 0, sizeof(ad));
  ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id = (int)dev; ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  DRV(cuMemSetAccess(va, total, &ad, 1));
  int idev = (int)dev;
  auto deleter = [va, total, idev](void*) {
    cudaSetDevice(idev);
    cudaDeviceSynchronize();  // no kernel may still read the range
    cuMemUnmap(va, total);
    cuMemAddressFree(va, total);
  };
  return torch::from_blob((void*)va, {(int64_t)total}, deleter,
                          torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, idev));
}
int64_t pointer_domain(int64_t ptr) {
  int ord = -99;
  if (cuPointerGetAttribute(&ord, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, (CUdeviceptr)ptr) != CUDA_SUCCESS) return -98;
  return ord;
}
std::vector<int64_t> chunk_ordinals(int64_t ptr, int64_t nbytes, int64_t chunk_bytes) {
  std::vector<int64_t> r;
  for (int64_t off = 0; off < nbytes; off += chunk_bytes) r.push_back(pointer_domain(ptr + off));
  return r;
}

// ------------------------------------------------------------------ domain-aware skinny GEMM
// out[M, N] (bf16, row stride ldo) = x[M, K] @ w[N, K]^T, fp32 accumulation, M <= 8 * NT.
// Work unit = one 16-row group of w (all K), owned by one warp. The groups are split into two queues by the locality
// domain of their memory (tiles0 / tiles1: group indices, built on the host from the verified chunk ordinals). Every
// CTA reads %smid -> sm_dom[] -> its domain and pulls groups from that domain's queue (atomicAdd on ctrs[d]); when the
// queue is empty it steals from the other one, so any CTA placement completes all groups. The next group is grabbed
// one group ahead and its first stages are prefetched while the current one finishes, so the load pipeline never
// drains between groups. The last warp to finish resets the counters, so the kernel is CUDA-graph safe.
// Loads go straight to registers with ld.global.nc; the MMA uses w as its A operand and
// x (held in shared memory with padded rows) as its B operand ("swap AB"). The K order
// inside an MMA is permuted identically for both operands for contiguous lane loads.
struct Q { const int* list0; const int* list1; int n0, n1; int* ctrs; };  // ctrs: [q0, q1, done, pad]

__device__ __forceinline__ int grab(const Q& q, int d) {
  int i = atomicAdd(&q.ctrs[d], 1);
  int n = d ? q.n1 : q.n0;
  if (i < n) return __ldg((d ? q.list1 : q.list0) + i);
  int o = d ^ 1;
  i = atomicAdd(&q.ctrs[o], 1);
  n = o ? q.n1 : q.n0;
  if (i < n) return __ldg((o ? q.list1 : q.list0) + i);
  return -1;
}
__device__ __forceinline__ void finish(const Q& q, int lane) {
  if (lane == 0) {
    __threadfence();
    const int total = (int)(gridDim.x * (blockDim.x >> 5));
    if (atomicAdd(&q.ctrs[2], 1) == total - 1) { q.ctrs[0] = 0; q.ctrs[1] = 0; q.ctrs[2] = 0; __threadfence(); }
  }
}
__device__ __forceinline__ uint4 ldg16(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

struct Args {
  const void* x; int ldx; int M;
  const void* w; int N;
  __nv_bfloat16* out; int ldo;
  const int8_t* sm_dom;
  Q q;
};

// BF16 staging uses LPS vector loads per row per lane; KIT counts stages per group.
// A slot is reused every PD stages across group boundaries, so PD must divide KIT.
template <int NT, int PD, int LPS>
__global__ void __launch_bounds__(256, 1) k_lm_bf16(Args a) {
  extern __shared__ __align__(16) unsigned char xs[];
  constexpr int K = 2048, XS = K * 2 + 64, KIT = 64 / LPS, SB = LPS * 64;  // SB: bytes per row per stage
  static_assert(KIT % PD == 0, "PD must divide the stages per group");
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  __shared__ int s_dom;
  if (threadIdx.x == 0) { int dd = a.sm_dom[smid()]; s_dom = dd < 0 ? 0 : dd; }
  __syncthreads();
  const int d = s_dom;
  int tile = -1, next = -1;
  if (lane == 0) { tile = grab(a.q, d); if (tile >= 0) next = grab(a.q, d); }
  tile = __shfl_sync(~0u, tile, 0);
  const char* W = (const char*)a.w;
  auto rowp = [&](int grp, int r) { return W + ((size_t)grp * 16 + r) * (size_t)(K * 2); };
  uint4 wr[PD][2][LPS];
#define LDSTAGE(slot, grp, it) do { const char* p0_ = rowp(grp, g) + (it) * SB + t * 16; const char* p1_ = rowp(grp, g + 8) + (it) * SB + t * 16; \
    _Pragma("unroll") for (int j = 0; j < LPS; j++) { wr[slot][0][j] = ldg16(p0_ + j * 64); wr[slot][1][j] = ldg16(p1_ + j * 64); } } while (0)
  if (tile >= 0) {
#pragma unroll
    for (int s = 0; s < PD; s++) LDSTAGE(s, tile, s);
  }
  // x -> shared (rows >= M zero)
  for (int i = threadIdx.x; i < 8 * NT * (K / 8); i += blockDim.x) {
    const int r = i / (K / 8), c = i % (K / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < a.M) v = *reinterpret_cast<const uint4*>((const __nv_bfloat16*)a.x + (size_t)r * a.ldx + c * 8);
    *reinterpret_cast<uint4*>(xs + r * XS + c * 16) = v;
  }
  __syncthreads();
  next = __shfl_sync(~0u, next, 0);
  while (tile >= 0) {
    int next2 = -1;
    if (lane == 0 && next >= 0) next2 = grab(a.q, d);
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
    for (int i = 0; i < KIT; i++) {
      const int sl = i % PD;
#pragma unroll
      for (int j = 0; j < LPS; j++) {
        const uint4 r0 = wr[sl][0][j], r1 = wr[sl][1][j];
#pragma unroll
        for (int n = 0; n < NT; n++) {
          const uint4 xv = *reinterpret_cast<const uint4*>(xs + (n * 8 + g) * XS + i * SB + j * 64 + t * 16);
          mma_bf16(acc[n], r0.x, r1.x, r0.y, r1.y, xv.x, xv.y);
          mma_bf16(acc[n], r0.z, r1.z, r0.w, r1.w, xv.z, xv.w);
        }
      }
      if (i + PD < KIT) LDSTAGE(sl, tile, i + PD);
      else if (next >= 0) LDSTAGE(sl, next, i + PD - KIT);
    }
    const int row = tile * 16 + g;
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int m = n * 8 + 2 * t + e;
        if (m < a.M) {
          a.out[(size_t)m * a.ldo + row] = __float2bfloat16(acc[n][e]);
          a.out[(size_t)m * a.ldo + row + 8] = __float2bfloat16(acc[n][2 + e]);
        }
      }
    tile = next;
    next = __shfl_sync(~0u, next2, 0);
  }
#undef LDSTAGE
  finish(a.q, lane);
}

// Cooperative variant (k_lm2): each CTA owns a row group, with K split across warps.
// Partial sums are reduced through double-buffered shared memory with a barrier per group.
// Group IDs come from the same per-domain queues. Thread 0 grabs them ahead through
// a software pipeline (atomic -> list lookup -> shared ids[]) to overlap lookup with work.
// Every warp prefetches the group selected for its register pipeline.
__device__ __forceinline__ int lookup(const Q& q, int i, int own) {
  const int n = own ? q.n1 : q.n0;
  if (i < n) return __ldg((own ? q.list1 : q.list0) + i);
  const int o = own ^ 1;  // own queue empty: steal (synchronous; only in the tail)
  const int j = atomicAdd(&q.ctrs[o], 1);
  if (j < (o ? q.n1 : q.n0)) return __ldg((o ? q.list1 : q.list0) + j);
  return -1;
}
template <int NT>
__device__ __forceinline__ void stage_x(const Args& a, unsigned char* xs) {
  constexpr int K = 2048, XS = K * 2 + 64;
  for (int i = threadIdx.x; i < 8 * NT * (K / 8); i += blockDim.x) {
    const int r = i / (K / 8), c = i % (K / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < a.M) v = *reinterpret_cast<const uint4*>((const __nv_bfloat16*)a.x + (size_t)r * a.ldx + c * 8);
    *reinterpret_cast<uint4*>(xs + r * XS + c * 16) = v;
  }
}

template <int NT, int LB, int PD>  // LB: 16-B loads per row per lane-quad per stage
__global__ void __launch_bounds__(256, 1) k_lm2(Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  constexpr int K = 2048, KW = K / 8, SE = LB * 32;  // K per warp; K per stage
  constexpr int SPU = KW / SE;                       // stages per group per warp
  static_assert(SPU >= 1 && KW % SE == 0 && PD % SPU == 0, "PD must be a multiple of the stages per group");
  constexpr int UF = PD / SPU, L = UF, R = 16;       // groups in flight ahead; id lookahead; id ring
  constexpr int XS = K * 2 + 64, MT = 8 * NT;
  unsigned char* xs = smem;
  float* red = reinterpret_cast<float*>(smem + MT * XS) + MT;  // [2][8 warps][16 rows][MT] (after MT floats of pad)
  __shared__ int ids[R];
  __shared__ int s_dom;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  if (threadIdx.x == 0) { int dd = a.sm_dom[smid()]; s_dom = dd < 0 ? 0 : dd; }
  __syncthreads();
  const int d = s_dom;
  int pt = -1, pi = 0;  // thread 0: tile for group u+L+1 (looked up), index for u+L+2 (atomic in flight)
  if (threadIdx.x == 0) {
    for (int k = 0; k <= L; k++) ids[k] = grab(a.q, d);
    pt = grab(a.q, d);
    pi = atomicAdd(&a.q.ctrs[d], 1);
  }
  __syncthreads();
  const uint8_t* W = (const uint8_t*)a.w;
  uint4 wr[PD][2][LB];
  auto ldstage = [&](int grp, int s, uint4 (&dst)[2][LB]) {
    const int kb = warp * KW + s * SE + 8 * t;  // element offset of this lane's first value
    const size_t r0 = (size_t)grp * 16 + g;
    const uint8_t* p0 = W + (r0 * K + kb) * 2;
#pragma unroll
    for (int j = 0; j < LB; j++) { dst[0][j] = ldg16(p0 + j * 64); dst[1][j] = ldg16(p0 + (size_t)8 * K * 2 + j * 64); }
  };
#pragma unroll
  for (int v = 0; v < UF; v++) {
    const int tl = ids[v];
    if (tl >= 0) {
#pragma unroll
      for (int s = 0; s < SPU; s++) ldstage(tl, s, wr[v * SPU + s]);
    }
  }
  stage_x<NT>(a, xs);
  __syncthreads();
  int u = 0;
  for (;;) {
#pragma unroll
    for (int sub = 0; sub < UF; sub++) {
      const int tile = ids[u % R];
      if (tile < 0) goto done;
      float acc[NT][4];
#pragma unroll
      for (int n = 0; n < NT; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
      for (int s = 0; s < SPU; s++) {
        const int sl = sub * SPU + s;
        const int kb = warp * KW + s * SE + 8 * t;
#pragma unroll
        for (int j = 0; j < LB; j++) {
          const uint4 r0 = wr[sl][0][j], r1 = wr[sl][1][j];
#pragma unroll
          for (int n = 0; n < NT; n++) {
            const uint4 xv = *reinterpret_cast<const uint4*>(xs + (n * 8 + g) * XS + (kb + 32 * j) * 2);
            mma_bf16(acc[n], r0.x, r1.x, r0.y, r1.y, xv.x, xv.y);
            mma_bf16(acc[n], r0.z, r1.z, r0.w, r1.w, xv.z, xv.w);
          }
        }
        const int tn = ids[(u + UF) % R];
        if (tn >= 0) ldstage(tn, s, wr[sl]);
      }
      float* rb = red + ((u & 1) * 8 + warp) * 16 * MT;
#pragma unroll
      for (int n = 0; n < NT; n++) {
        *reinterpret_cast<float2*>(rb + g * MT + n * 8 + 2 * t) = make_float2(acc[n][0], acc[n][1]);
        *reinterpret_cast<float2*>(rb + (g + 8) * MT + n * 8 + 2 * t) = make_float2(acc[n][2], acc[n][3]);
      }
      if (threadIdx.x == 0) {
        ids[(u + L + 1) % R] = pt;
        pt = lookup(a.q, pi, d);
        pi = atomicAdd(&a.q.ctrs[d], 1);
      }
      __syncthreads();
      for (int o = threadIdx.x; o < 16 * a.M; o += blockDim.x) {
        const int tok = o >> 4, r = o & 15;
        const float* rr = red + (u & 1) * 8 * 16 * MT + r * MT + tok;
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < 8; w++) sum += rr[w * 16 * MT];
        a.out[(size_t)tok * a.ldo + tile * 16 + r] = __float2bfloat16(sum);
      }
      u++;
    }
  }
done:
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&a.q.ctrs[2], 1) == (int)gridDim.x - 1) { a.q.ctrs[0] = 0; a.q.ctrs[1] = 0; a.q.ctrs[2] = 0; __threadfence(); }
  }
}

// ---- k_lm3: cooperative structure of k_lm2 with the weight stream staged through shared memory by cp.async
// (LDGSTS): coalesced warp copies keep PD groups ahead without staging weights in registers.
// Fragments are read back from padded rows to avoid bank conflicts.
// bf16 weights, mma.sync m16n8k16 (M <= 8 * NT); x is held in registers.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

template <int NT, int PD, int NW, int G>  // NW warps per CTA, each owning K / NW; G 16-row groups per unit
__global__ void __launch_bounds__(NW * 32, 1) k_lm3(Args a) {
  extern __shared__ __align__(128) unsigned char smem[];
  constexpr int K = 2048, KW = K / NW, R = 16, L = PD;
  constexpr int RB = KW * 2;                     // bytes per row per warp slice
  constexpr int J = KW / 32;                     // 32-K steps per warp slice
  constexpr int RS = RB;                         // smem row stride; 16-B chunks XOR-swizzled by row (see swz)
  constexpr int SUB = 16 * RS;                   // one 16-row group's slice
  constexpr int STG = G * SUB;                   // bytes per stage (one unit of G groups)
  // chunk swizzle: the 8 lanes of an LDS.128 phase read 2 rows x 4 chunks
  auto swz = [](int row) { return (row & 1) << 2; };
  constexpr int MT = 8 * NT;                     // output rows held
  unsigned char* ring = smem;                                  // [NW warps][PD][STG]
  float* red = reinterpret_cast<float*>(smem + NW * PD * STG) + MT;  // [2][NW warps][16 G rows][MT] (after MT floats of pad)
  __shared__ int ids[R];
  __shared__ int s_dom;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) { int dd = a.sm_dom[smid()]; s_dom = dd < 0 ? 0 : dd; }
  __syncthreads();
  const int d = s_dom;
  int pt = -1, pi = 0;
  if (threadIdx.x == 0) {
    for (int k = 0; k <= L; k++) ids[k] = grab(a.q, d);
    pt = grab(a.q, d);
    pi = atomicAdd(&a.q.ctrs[d], 1);
  }
  __syncthreads();
  const uint8_t* W = (const uint8_t*)a.w;
  const uint32_t ring0 = (uint32_t)__cvta_generic_to_shared(ring) + warp * PD * STG;
  // Copy one group's warp slice with contiguous vector loads.
  auto issue = [&](int grp, int slot) {
    if (grp >= 0) {
      constexpr int CPR = RB / 16, RPI = 32 / CPR;  // 16-B chunks per row; rows per instruction
      static_assert(CPR <= 32 && 32 % CPR == 0, "slice row must be <= 512 B");
#pragma unroll
      for (int sg = 0; sg < G; sg++) {
        const size_t row0 = ((size_t)grp * G + sg) * 16;
        const uint32_t dst = ring0 + slot * STG + sg * SUB;
#pragma unroll
        for (int i = 0; i < 16 / RPI; i++) {
          const int row = i * RPI + lane / CPR, ch = lane % CPR;
          cp_async16(dst + row * RS + (ch ^ swz(row)) * 16, W + (row0 + row) * (2 * K) + warp * RB + ch * 16);
        }
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int v = 0; v < PD - 1; v++) issue(ids[v], v);
  // lane (g, t) holds x[n * 8 + g][warp * 256 + 32 j + 8 t .. + 8] for all j (the same for every group)
  uint4 xb16[NT][J];
#pragma unroll
  for (int n = 0; n < NT; n++) {
    const int m = n * 8 + (lane >> 2);
#pragma unroll
    for (int j = 0; j < J; j++)
      xb16[n][j] = m < a.M ? *reinterpret_cast<const uint4*>((const __nv_bfloat16*)a.x + (size_t)m * a.ldx + warp * KW + 32 * j + 8 * (lane & 3))
                           : make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  int u = 0;
  for (;;) {
    const int tile = ids[u % R];
    if (tile < 0) break;
    __syncwarp();                                // every lane is done with the slot about to be refilled
    issue(ids[(u + PD - 1) % R], (u + PD - 1) % PD);
    cp_wait<PD - 1>();
    __syncwarp();                                // all lanes' copies of group u are visible
    float* rb = red + ((u & 1) * NW + warp) * 16 * G * MT;
#pragma unroll
    for (int sg = 0; sg < G; sg++) {
    const unsigned char* st = ring + (warp * PD + u % PD) * STG + sg * SUB;
    float acc[NT * 4];
#pragma unroll
    for (int q = 0; q < NT * 4; q++) acc[q] = 0.f;
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int j = 0; j < J; j++) {
      const int cpos = ((4 * j + t) ^ swz(g)) * 16;  // rows g and g + 8 share the swizzle
      const uint4 r0 = *reinterpret_cast<const uint4*>(st + g * RS + cpos);
      const uint4 r1 = *reinterpret_cast<const uint4*>(st + (g + 8) * RS + cpos);
#pragma unroll
      for (int n = 0; n < NT; n++) {
        const uint4 xv = xb16[n][j];
        float (&c)[4] = *reinterpret_cast<float (*)[4]>(&acc[n * 4]);
        mma_bf16(c, r0.x, r1.x, r0.y, r1.y, xv.x, xv.y);
        mma_bf16(c, r0.z, r1.z, r0.w, r1.w, xv.z, xv.w);
      }
    }
#pragma unroll
    for (int n = 0; n < NT; n++) {
      *reinterpret_cast<float2*>(rb + (sg * 16 + g) * MT + n * 8 + 2 * t) = make_float2(acc[n * 4 + 0], acc[n * 4 + 1]);
      *reinterpret_cast<float2*>(rb + (sg * 16 + g + 8) * MT + n * 8 + 2 * t) = make_float2(acc[n * 4 + 2], acc[n * 4 + 3]);
    }
    }  // sg
    if (threadIdx.x == 0) {
      ids[(u + L + 1) % R] = pt;
      pt = lookup(a.q, pi, d);
      pi = atomicAdd(&a.q.ctrs[d], 1);
    }
    __syncthreads();
    for (int o = threadIdx.x; o < 16 * G * a.M; o += blockDim.x) {
      const int m = o / (16 * G), rr = o % (16 * G);
      const float* p = red + (u & 1) * NW * 16 * G * MT + rr * MT + m;
      float sum = 0.f;
#pragma unroll
      for (int w = 0; w < NW; w++) sum += p[w * 16 * G * MT];
      a.out[(size_t)m * a.ldo + (size_t)tile * 16 * G + rr] = __float2bfloat16(sum);
    }
    u++;
  }
  cp_wait<0>();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&a.q.ctrs[2], 1) == (int)gridDim.x - 1) { a.q.ctrs[0] = 0; a.q.ctrs[1] = 0; a.q.ctrs[2] = 0; __threadfence(); }
  }
}

static int g_nsm[64];
static int num_sms(int dev) {
  if (!g_nsm[dev]) RTC(cudaDeviceGetAttribute(&g_nsm[dev], cudaDevAttrMultiProcessorCount, dev));
  return g_nsm[dev];
}

template <typename KF>
static void launch(KF kern, const Args& a, int warps, size_t smem, cudaStream_t st, int dev) {
  // the max-dynamic-smem attribute is set once per kernel (largest request so far); not a stream op (graph safe)
  static std::unordered_map<const void*, size_t> s_smem;
  size_t& cur = s_smem[(const void*)kern];
  if (smem > cur) { RTC(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); cur = smem; }
  kern<<<num_sms(dev), warps * 32, smem, st>>>(a);
  RTC(cudaGetLastError());
}

static Args make_args(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor sm_dom, torch::Tensor tiles0,
                      torch::Tensor tiles1, torch::Tensor ctrs) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(1) == 1 && x.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(out.dim() == 2 && out.stride(1) == 1 && out.scalar_type() == torch::kBFloat16 && out.size(0) == x.size(0));
  TORCH_CHECK(w.is_contiguous() && w.dim() == 2 && w.size(1) == x.size(1) && out.size(1) >= w.size(0) && w.size(0) % 16 == 0);
  TORCH_CHECK(sm_dom.scalar_type() == torch::kInt8 && tiles0.scalar_type() == torch::kInt32 && tiles1.scalar_type() == torch::kInt32 && ctrs.scalar_type() == torch::kInt32 && ctrs.numel() >= 4);
  TORCH_CHECK(w.size(0) % (16 * (tiles0.numel() + tiles1.numel())) == 0 && w.size(0) / (16 * (tiles0.numel() + tiles1.numel())) <= 4,
              "tile lists must cover every unit of 16 * G rows");
  Args a;
  a.x = x.data_ptr(); a.ldx = (int)x.stride(0); a.M = (int)x.size(0);
  a.w = w.data_ptr(); a.N = (int)w.size(0);
  a.out = (__nv_bfloat16*)out.data_ptr(); a.ldo = (int)out.stride(0);
  a.sm_dom = sm_dom.data_ptr<int8_t>();
  a.q.list0 = tiles0.data_ptr<int>(); a.q.list1 = tiles1.data_ptr<int>();
  a.q.n0 = (int)tiles0.numel(); a.q.n1 = (int)tiles1.numel(); a.q.ctrs = ctrs.data_ptr<int>();
  return a;
}

// warps: 8 (256 threads; launch bounds). cfg = 100 * LPS + PD (16-B loads per row per stage, stages in flight);
// PD must divide 64 / LPS.
void lm_bf16(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor sm_dom, torch::Tensor tiles0,
             torch::Tensor tiles1, torch::Tensor ctrs, int64_t warps, int64_t cfg) {
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && x.size(1) == 2048, "bf16 path: K = 2048");
  Args a = make_args(x, w, out, sm_dom, tiles0, tiles1, ctrs);
  const int M = a.M; TORCH_CHECK(M >= 1 && M <= 32, "bf16 path: 1 <= M <= 32");
  TORCH_CHECK(warps == 8, "warps must be 8");
  const int NT = (M + 7) / 8;
  const size_t smem = (size_t)8 * (NT == 3 ? 4 : NT) * (2048 * 2 + 64);
  c10::cuda::CUDAGuard guard(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const int dev = x.get_device();
#define B16(nt, l, p) if ((NT == nt || (nt == 4 && NT == 3)) && cfg == 100 * l + p) { launch(k_lm_bf16<nt, p, l>, a, 8, smem, st, dev); return; }
#define B16C(nt) B16(nt, 4, 2) B16(nt, 4, 4) B16(nt, 2, 4) B16(nt, 2, 8) B16(nt, 1, 8) B16(nt, 1, 16)
  B16C(1) B16C(2) B16C(4)
#undef B16C
#undef B16
  if (NT <= 2) {  // cp.async-staged kernel: cfg = 3000 + PD
    const int MT = 8 * NT;
    // cfg = 3000 + 100 * (NW / 8 - 1) + PD: NW = 8 or 16 warps
    const int nw = cfg >= 3100 ? 16 : 8, pd = cfg % 10;
    const size_t smem3 = (size_t)nw * pd * 16 * (4096 / nw) + MT * 4 + 2 * nw * 16 * MT * 4;
#define S16(nt, p, w) if (NT == nt && nw == w && pd == p && cfg % 100 < 10) { launch(k_lm3<nt, p, w, 1>, a, w, smem3, st, dev); return; }
    S16(1, 2, 8) S16(1, 3, 8) S16(2, 2, 8) S16(2, 3, 8) S16(1, 3, 16) S16(2, 3, 16)
#undef S16
  }
  {  // cooperative kernel: cfg = 1000 + 100 * LB + PD
    const int NTc = NT == 3 ? 4 : NT, MT = 8 * NTc;
    const size_t smem2 = (size_t)MT * (2048 * 2 + 64) + MT * 4 + 2 * 8 * 16 * MT * 4;
#define C16(nt, l, p) if (NTc == nt && cfg == 1000 + 100 * l + p) { launch(k_lm2<nt, l, p>, a, 8, smem2, st, dev); return; }
#define C16C(nt) C16(nt, 4, 2) C16(nt, 4, 4) C16(nt, 2, 4) C16(nt, 2, 8)
    C16C(1) C16C(2) C16C(4)
#undef C16C
#undef C16
  }
  TORCH_CHECK(false, "bf16 path: unsupported (M, cfg)");
}

}  // namespace loc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("device_info", &loc::device_info);
  m.def("green_sm_map", &loc::green_sm_map);
  m.def("green_streams", &loc::green_streams, pybind11::arg("dev"), pybind11::arg("extra0") = 0, pybind11::arg("extra1") = 0);
  m.def("stream_smids", &loc::stream_smids);
  m.def("probe_sm_latency", &loc::probe_sm_latency);
  m.def("granularity", &loc::granularity);
  m.def("alloc_localized", &loc::alloc_localized);
  m.def("pointer_domain", &loc::pointer_domain);
  m.def("chunk_ordinals", &loc::chunk_ordinals);
  m.def("lm_bf16", &loc::lm_bf16);
}
"""
