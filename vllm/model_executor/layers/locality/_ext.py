# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""CUDA extension of the locality package (JIT-built with nvcc on first use).

Rubin-class GPUs (VR200, sm_107) consist of two locality domains ("micro-GPUs",
one per die), each with its own HBM stacks and about half of the SMs. Reads by
an SM from its own domain's HBM never cross the die-to-die fabric; interleaved
(``cudaMalloc``) memory sends half of every stream across it, which caps a
streaming read at ~13.5-14 TB/s on Hecate (HBM 4752 MHz) against ~17 TB/s for
domain-local reads.

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
- ``probe_sm_latency(buf0, buf1) -> float32[num_sms, 2]``: cycles per dependent
  L2-hit load from each SM into two small buffers placed on domain 0 / 1. The
  home L2 slice of a localized line sits on its die, so a hop over the fabric
  shows up as +40-60% latency; this places the unassigned SMs.
- ``alloc_localized(nbytes, chunk_domains, chunk_bytes, dev) -> uint8 tensor``:
  one VA range; chunk i (``chunk_bytes`` each, a multiple of the 2 MiB
  granularity) is a ``cuMemCreate`` allocation on locality domain
  ``chunk_domains[i]`` (``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN``),
  mapped with read/write access for ``dev``. The tensor's deleter synchronizes
  the device, unmaps and frees the range.
- ``granularity(dev)``, ``pointer_domain(ptr)``, ``chunk_ordinals(ptr, nbytes,
  chunk_bytes)`` (``CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL``; -1 for
  interleaved memory).
- ``green_streams(dev) -> [stream0, stream1, sms0, sms1, remainder]``: a
  persistent green context + non-blocking stream per domain (process lifetime),
  for fork/join dispatch on 100+100 SMs.
- ``lm_bf16(x, w, out, sm_dom, tiles0, tiles1, ctrs, warps, pd)`` and
  ``lm_fp8(x, wq, wscale0, wscale1, out, sm_dom, tiles0, tiles1, ctrs, warps, pd)``: the
  domain-aware skinny GEMM ``out[M, N] = x[M, K] @ w[N, K]^T`` (see the kernel
  comment); bf16 output, fp32 accumulation.
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
        return os.environ.get(
            "VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm")
        )


def load():
    """Build (or load the cached build of) the extension for the current device."""
    if _ext:
        return _ext[0]
    major, minor = torch.cuda.get_device_capability()
    import torch.utils.cpp_extension as cpp

    arch = f"{major}{minor}{'a' if major >= 9 else ''}"
    bdir = os.path.join(_cache_root(), "locality_ext", f"sm{arch}")
    os.makedirs(bdir, exist_ok=True)
    digest = hashlib.sha256(_SOURCE.encode()).hexdigest()[:16]
    src = os.path.join(bdir, f"locality_{digest}.cu")
    if not os.path.exists(src):
        tmp = f"{src}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(_SOURCE)
        os.replace(tmp, src)
    orig = cpp._get_cuda_arch_flags
    cpp._get_cuda_arch_flags = lambda cflags=None: [
        f"-gencode=arch=compute_{arch},code=sm_{arch}"
    ]
    try:
        ext = cpp.load(
            name=f"_locality_{digest}",
            sources=[src],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"],
            extra_cflags=["-O3", "-std=c++20"],
            extra_ldflags=["-lcuda"],
            build_directory=bdir,
            verbose=False,
        )
    finally:
        cpp._get_cuda_arch_flags = orig
    _ext.append(ext)
    return ext


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
#include <unordered_map>

namespace loc {

#define DRV(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorName(r_, &s_); TORCH_CHECK(false, #x " failed: ", s_); } } while (0)
#define RTC(x) do { cudaError_t e_ = (x); TORCH_CHECK(e_ == cudaSuccess, #x " failed: ", cudaGetErrorString(e_)); } while (0)

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

static void split_domains(CUdevice d, CUdevResource part[2], CUdevResource* rem) {
  CUdevResource all; memset(&all, 0, sizeof(all));
  DRV(cuDeviceGetDevResource(d, &all, CU_DEV_RESOURCE_TYPE_SM));
  CU_DEV_SM_RESOURCE_GROUP_PARAMS gp[2]; memset(gp, 0, sizeof(gp));
  // coscheduledSmCount 2 (TPC granularity): the default (8) rounds each domain down below its SM count
  for (int k = 0; k < 2; k++) { gp[k].smCount = 0; gp[k].coscheduledSmCount = 2; gp[k].flags = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID; gp[k].localityDomainId = k; }
  memset(part, 0, 2 * sizeof(CUdevResource)); memset(rem, 0, sizeof(CUdevResource));
  DRV(cuDevSmResourceSplit(part, 2, &all, rem, 0, gp));
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

// persistent per-domain green contexts + streams (process lifetime)
static CUgreenCtx g_gctx[2]; static CUstream g_gst[2]; static int g_gsm[3] = {0, 0, 0}; static bool g_have_gc = false;
std::vector<int64_t> green_streams(int64_t dev) {
  if (!g_have_gc) {
    CUdevice d = cu_dev(dev);
    CUdevResource part[2], rem; split_domains(d, part, &rem);
    for (int k = 0; k < 2; k++) {
      CUdevResourceDesc desc; DRV(cuDevResourceGenerateDesc(&desc, &part[k], 1));
      DRV(cuGreenCtxCreate(&g_gctx[k], desc, d, CU_GREEN_CTX_DEFAULT_STREAM));
      DRV(cuGreenCtxStreamCreate(&g_gst[k], g_gctx[k], CU_STREAM_NON_BLOCKING, 0));
      g_gsm[k] = (int)part[k].sm.smCount;
    }
    g_gsm[2] = rem.type == CU_DEV_RESOURCE_TYPE_SM ? (int)rem.sm.smCount : 0;
    g_have_gc = true;
    RTC(cudaSetDevice((int)dev));
  }
  return {(int64_t)(uintptr_t)g_gst[0], (int64_t)(uintptr_t)g_gst[1], g_gsm[0], g_gsm[1], g_gsm[2]};
}

// pointer chase (L2-resident chain, 64 dependent ld.global.cg) from every SM into buf0 and buf1
__global__ void k_chain_init(uint64_t* b, int n, int stride_words) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    for (int i = 0; i < n; i++) b[(size_t)i * stride_words] = (uint64_t)(((i + 1) % n) * stride_words);
}
__global__ void k_probe(const uint64_t* b0, const uint64_t* b1, int n, float* out) {
  extern __shared__ unsigned char pad[];  // dynamic smem forces one CTA per SM
  if (threadIdx.x != 0) return;
  pad[0] = 0;
  unsigned s = smid();
  for (int which = 0; which < 2; which++) {
    const uint64_t* b = which ? b1 : b0;
    uint64_t off = 0, v;
    for (int i = 0; i < n; i++) { asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(b + off) : "memory"); off = v; }  // warm
    long long t0 = clock64();
    for (int r = 0; r < 4; r++)
      for (int i = 0; i < n; i++) { asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(b + off) : "memory"); off = v; }
    long long t1 = clock64();
    out[2 * s + which] = (float)(t1 - t0) / (4 * n) + (off == 0xffffffffull ? 1e-9f : 0.f);
  }
}
torch::Tensor probe_sm_latency(torch::Tensor buf0, torch::Tensor buf1) {
  TORCH_CHECK(buf0.is_cuda() && buf1.is_cuda() && buf0.nbytes() >= (1 << 16) && buf1.nbytes() >= (1 << 16));
  c10::cuda::CUDAGuard guard(buf0.device());
  int dev = buf0.get_device(), nsm = 0, smem_optin = 0;
  RTC(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  RTC(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  const int n = 64, stride_words = 128;  // 64 lines, 1 KB apart, in each buffer (64 KB)
  auto st = at::cuda::getCurrentCUDAStream();
  k_chain_init<<<1, 1, 0, st>>>((uint64_t*)buf0.data_ptr(), n, stride_words);
  k_chain_init<<<1, 1, 0, st>>>((uint64_t*)buf1.data_ptr(), n, stride_words);
  auto out = torch::zeros({nsm, 2}, buf0.options().dtype(torch::kFloat32));
  int smem = smem_optin > 0 ? smem_optin : 100 * 1024;
  RTC(cudaFuncSetAttribute(k_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  k_probe<<<nsm, 32, smem, st>>>((const uint64_t*)buf0.data_ptr(), (const uint64_t*)buf1.data_ptr(), n, out.data_ptr<float>());
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
// Loads go straight to registers (ld.global.nc, PD stages of 4 KB per warp); the MMA is mma.sync with w as the
// 16-row A operand and x (held in shared memory, padded rows) as the 8-column B operand ("swap AB"). The K order
// inside an MMA is permuted identically for both operands so that every lane loads contiguous 16 B (bf16) / 8 B (fp8).
// fp8 (MXFP8 weights, E8M0 scale per 32 K of every row, scales [N, K/32] uint8): x is quantized in the kernel to e4m3
// with one fp32 scale per row (448 / amax); each m16n8k32 covers one 32-K block, and its result is added to the
// accumulator times the row's block scale; the per-row x scale is applied at the end.
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
__device__ __forceinline__ uint2 ldg8(const void* p) {
  uint2 v; asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p)); return v;
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_e4m3(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}

struct Args {
  const void* x; int ldx; int M;
  const void* w; const uint8_t* wscale[2]; int N;  // fp8: one copy of the [N, K/32] scales per domain
  __nv_bfloat16* out; int ldo;
  const int8_t* sm_dom;
  Q q;
};

// ---- bf16, K = 2048: stage = LPS x 16 B per row per lane (rows g, g+8) = LPS KB per warp; KIT = 64 / LPS stages
// per group. A slot is reused every PD stages across group boundaries, so PD must divide KIT.
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

// ---- fp8, K = 2048: stage = BPS blocks x 8 B per row per lane (rows g, g+8) = BPS/2 KB per warp + the rows' BPS
// scale bytes; KIT = 64 / BPS stages per group (PD must divide KIT, see bf16)
__device__ __forceinline__ float e8m0(uint32_t e) { return __uint_as_float(e << 23); }
__device__ __forceinline__ uint32_t ldg4(const void* p) {
  uint32_t v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
template <int NT, int PD, int BPS>
__global__ void __launch_bounds__(256, 1) k_lm_fp8(Args a) {
  extern __shared__ __align__(16) unsigned char xs[];
  constexpr int K = 2048, XS = K + 32, KIT = 64 / BPS, SB = BPS * 32;  // XS: e4m3 row + pad (conflict-free 8-B reads)
  static_assert(KIT % PD == 0 && (BPS == 4 || BPS == 8), "bad fp8 config");
  float* sx = reinterpret_cast<float*>(xs + 8 * NT * XS);  // per-row x scale (amax / 448)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int nwarps = blockDim.x >> 5;
  __shared__ int s_dom;
  if (threadIdx.x == 0) { int dd = a.sm_dom[smid()]; s_dom = dd < 0 ? 0 : dd; }
  __syncthreads();
  const int d = s_dom;
  int tile = -1, next = -1;
  if (lane == 0) { tile = grab(a.q, d); if (tile >= 0) next = grab(a.q, d); }
  tile = __shfl_sync(~0u, tile, 0);
  const uint8_t* W = (const uint8_t*)a.w;
  const uint8_t* WS = a.wscale[d];  // this domain's copy of the scales
  uint2 wr[PD][2][BPS];
  uint2 sr[PD][2];
#define LDSTAGE(slot, grp, it) do { const size_t r0_ = (size_t)(grp) * 16 + g; \
    const uint8_t* p0_ = W + r0_ * K + (it) * SB + t * 8; const uint8_t* p1_ = p0_ + (size_t)8 * K; \
    _Pragma("unroll") for (int j = 0; j < BPS; j++) { wr[slot][0][j] = ldg8(p0_ + j * 32); wr[slot][1][j] = ldg8(p1_ + j * 32); } \
    if constexpr (BPS == 8) { sr[slot][0] = ldg8(WS + r0_ * (K / 32) + (it) * 8); sr[slot][1] = ldg8(WS + (r0_ + 8) * (K / 32) + (it) * 8); } \
    else { sr[slot][0].x = ldg4(WS + r0_ * (K / 32) + (it) * 4); sr[slot][1].x = ldg4(WS + (r0_ + 8) * (K / 32) + (it) * 4); } } while (0)
  if (tile >= 0) {
#pragma unroll
    for (int s = 0; s < PD; s++) LDSTAGE(s, tile, s);
  }
  // x -> e4m3 in shared, one scale per row (each warp quantizes rows warp, warp + nwarps, ...)
  for (int r = warp; r < 8 * NT; r += nwarps) {
    if (r >= a.M) {
      for (int c = lane; c < K / 16; c += 32) *reinterpret_cast<uint4*>(xs + r * XS + c * 16) = make_uint4(0, 0, 0, 0);
      if (lane == 0) sx[r] = 0.f;
      continue;
    }
    const __nv_bfloat16* xr = (const __nv_bfloat16*)a.x + (size_t)r * a.ldx;
    constexpr int PER = K / 32 / 8;  // 16-B vectors per lane
    uint4 v[PER];
#pragma unroll
    for (int u = 0; u < PER; u++) v[u] = *reinterpret_cast<const uint4*>(xr + (u * 32 + lane) * 8);
    float amax = 0.f;
#pragma unroll
    for (int u = 0; u < PER; u++) {
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v[u]);
#pragma unroll
      for (int e = 0; e < 4; e++) { float2 f = __bfloat1622float2(h[e]); amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y))); }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, o));
    const float inv = amax > 0.f ? 448.f / amax : 1.f;
    if (lane == 0) sx[r] = amax > 0.f ? amax / 448.f : 0.f;
#pragma unroll
    for (int u = 0; u < PER; u++) {
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v[u]);
      uint32_t q[2];
#pragma unroll
      for (int e = 0; e < 2; e++) {
        float2 f0 = __bfloat1622float2(h[2 * e]), f1 = __bfloat1622float2(h[2 * e + 1]);
        uint16_t lo, hi;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(f0.y * inv), "f"(f0.x * inv));
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(f1.y * inv), "f"(f1.x * inv));
        q[e] = (uint32_t)lo | ((uint32_t)hi << 16);
      }
      *reinterpret_cast<uint2*>(xs + r * XS + (u * 32 + lane) * 8) = make_uint2(q[0], q[1]);
    }
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
      for (int j = 0; j < BPS; j++) {
        const uint2 r0 = wr[sl][0][j], r1 = wr[sl][1][j];
        const uint32_t sw0w = j < 4 ? sr[sl][0].x : sr[sl][0].y, sw1w = j < 4 ? sr[sl][1].x : sr[sl][1].y;
        const float s0 = e8m0((sw0w >> (8 * (j & 3))) & 0xff), s1 = e8m0((sw1w >> (8 * (j & 3))) & 0xff);
#pragma unroll
        for (int n = 0; n < NT; n++) {
          const uint2 xv = *reinterpret_cast<const uint2*>(xs + (n * 8 + g) * XS + i * SB + j * 32 + t * 8);
          float c[4];
          mma_e4m3(c, r0.x, r1.x, r0.y, r1.y, xv.x, xv.y);
          acc[n][0] = fmaf(c[0], s0, acc[n][0]); acc[n][1] = fmaf(c[1], s0, acc[n][1]);
          acc[n][2] = fmaf(c[2], s1, acc[n][2]); acc[n][3] = fmaf(c[3], s1, acc[n][3]);
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
          const float s = sx[m];
          a.out[(size_t)m * a.ldo + row] = __float2bfloat16(acc[n][e] * s);
          a.out[(size_t)m * a.ldo + row + 8] = __float2bfloat16(acc[n][2 + e] * s);
        }
      }
    tile = next;
    next = __shfl_sync(~0u, next2, 0);
  }
#undef LDSTAGE
  finish(a.q, lane);
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
  TORCH_CHECK(tiles0.numel() + tiles1.numel() == w.size(0) / 16, "tile lists must cover every 16-row group");
  Args a;
  a.x = x.data_ptr(); a.ldx = (int)x.stride(0); a.M = (int)x.size(0);
  a.w = w.data_ptr(); a.wscale[0] = a.wscale[1] = nullptr; a.N = (int)w.size(0);
  a.out = (__nv_bfloat16*)out.data_ptr(); a.ldo = (int)out.stride(0);
  a.sm_dom = sm_dom.data_ptr<int8_t>();
  a.q.list0 = tiles0.data_ptr<int>(); a.q.list1 = tiles1.data_ptr<int>();
  a.q.n0 = (int)tiles0.numel(); a.q.n1 = (int)tiles1.numel(); a.q.ctrs = ctrs.data_ptr<int>();
  return a;
}

// warps: 8 (256 threads; launch bounds). cfg = 100 * LPS + PD (bf16: 16-B loads per row per stage, stages in flight)
// or 100 * BPS + PD (fp8: 32-K blocks per row per stage); PD must divide 64 / LPS (64 / BPS).
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
  TORCH_CHECK(false, "bf16 path: unsupported (M, cfg)");
}

void lm_fp8(torch::Tensor x, torch::Tensor wq, torch::Tensor wscale0, torch::Tensor wscale1, torch::Tensor out,
            torch::Tensor sm_dom, torch::Tensor tiles0, torch::Tensor tiles1, torch::Tensor ctrs, int64_t warps, int64_t cfg) {
  TORCH_CHECK(wq.element_size() == 1 && x.size(1) == 2048, "fp8 path: K = 2048");
  for (auto* s : {&wscale0, &wscale1})
    TORCH_CHECK(s->element_size() == 1 && s->is_contiguous() && s->numel() == wq.size(0) * (x.size(1) / 32), "wscale must be [N, K/32]");
  Args a = make_args(x, wq, out, sm_dom, tiles0, tiles1, ctrs);
  a.wscale[0] = (const uint8_t*)wscale0.data_ptr(); a.wscale[1] = (const uint8_t*)wscale1.data_ptr();
  const int M = a.M; TORCH_CHECK(M >= 1 && M <= 64, "fp8 path: 1 <= M <= 64");
  TORCH_CHECK(warps == 8, "warps must be 8");
  int NT = (M + 7) / 8; NT = NT <= 1 ? 1 : NT <= 2 ? 2 : NT <= 4 ? 4 : 8;
  const size_t smem = (size_t)8 * NT * (2048 + 32) + 8 * NT * 4;
  c10::cuda::CUDAGuard guard(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const int dev = x.get_device();
#define F8(nt, b, p) if (NT == nt && cfg == 100 * b + p) { launch(k_lm_fp8<nt, p, b>, a, 8, smem, st, dev); return; }
#define F8C(nt) F8(nt, 8, 2) F8(nt, 8, 4) F8(nt, 4, 4) F8(nt, 4, 8)
  F8C(1) F8C(2) F8C(4) F8C(8)
#undef F8C
#undef F8
  TORCH_CHECK(false, "fp8 path: unsupported (M, cfg)");
}

}  // namespace loc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("device_info", &loc::device_info);
  m.def("green_sm_map", &loc::green_sm_map);
  m.def("green_streams", &loc::green_streams);
  m.def("probe_sm_latency", &loc::probe_sm_latency);
  m.def("granularity", &loc::granularity);
  m.def("alloc_localized", &loc::alloc_localized);
  m.def("pointer_domain", &loc::pointer_domain);
  m.def("chunk_ordinals", &loc::chunk_ordinals);
  m.def("lm_bf16", &loc::lm_bf16);
  m.def("lm_fp8", &loc::lm_fp8);
}
"""
