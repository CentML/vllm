// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3-Flash mHC sublayer boundary at decode token counts (<= 32 rows) with
// the tensor-parallel all-reduce of the sublayer output fused in.
// hc_mult 4, hidden 4096, 24 mix outputs.
//
// Under TP every o_proj / MoE output is all-reduced (FlashInfer MNNVL one-shot
// Lamport kernel, ~5 us on VR200 at decode sizes) and then fed straight into
// TileLang's mhc_fused (hc_post + fp32 prenorm GEMM partials). This kernel
// replaces both launches: it takes the local TP partial sum and
//   1. pushes its slice into the current Lamport buffer of FlashInfer's MNNVL
//      all-reduce workspace (multicast store, one write reaches every rank),
//   2. polls the local copy of that buffer for the other ranks' slices and
//      sums them in rank order in fp32 (FlashInfer's one-shot / two-shot
//      order), rounding once to bf16,
//   3. runs the hc_post FMA chain and the fp32 prenorm GEMM partials with
//      TileLang's split structure and operation order.
// It follows FlashInfer's Lamport protocol on the same workspace and flags
// (three rotating buffers, the dirty one cleared by the next call, a per-call
// CTA arrival counter, flags rotated by CTA 0 once every CTA has arrived), so
// it interleaves freely with FlashInfer all-reduce calls on the same group.
//
// Numerics: the reduced x is bit-identical to FlashInfer's all-reduce output
// (fp32 0 + r0 + r1 + ..., one bf16 rounding; -0 -> +0 sanitizing of the
// payload cannot change that sum). residual_out, the GEMM partials yp and the
// sum-of-squares partials rp are bit-identical to mhc_fused_tilelang for the
// same (split, threads, iterations) config: thread t of split s accumulates h =
// s * HPS + it * NTHR + t over it in order with fma(w, r', acc) per stream,
// then the xor-16..1 butterfly, then 0 + warp 0 + warp 1 + ... . Which CTA
// computes which outputs (tile_n) and how many tokens share a CTA (fn reuse)
// does not change any sum, so those are free tuning knobs.
//
// Deadlock freedom needs every CTA of the grid to make progress while others
// poll; the grids here are small (<= 512 CTAs of 128-256 threads).
//
// Built without -use_fast_math (see mhc_ar_post.py).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace glm5next_mhcar {

constexpr int kHC = 4;
constexpr int kMix = kHC * (kHC + 2);  // 24
constexpr int kHidden = 4096;
constexpr int kK = kHC * kHidden;
constexpr uint32_t kNegZeroF32 = 0x80000000u;  // FlashInfer Lamport sentinel word

// Contraction order of TileLang's hc_post chain (see the compute loop); 1
// selects fma(cm0, r0, pm * x) for a toolchain that contracts the other way.
#ifndef GLM_MHCAR_X_FIRST
#define GLM_MHCAR_X_FIRST 0
#endif
#ifndef GLM_MHCAR_WORLD
#define GLM_MHCAR_WORLD 4
#endif

struct Params {
  const __nv_bfloat16* x;         // [M, 4096] local TP partial sum
  const __nv_bfloat16* residual;  // [M, 4, 4096]
  const float* post;              // [M, 4]
  const float* comb;              // [M, 4, 4]
  const float* fn;                // [24, 4, 4096]
  float* yp;                      // [S, M, 24]
  float* rp;                      // [S, M]
  __nv_bfloat16* residual_out;    // [M, 4, 4096]
  __nv_bfloat16* x_out;           // [M, 4096] reduced x (optional, tests)
  // FlashInfer MNNVL workspace (unused for world size 1).
  void* mcast;          // multicast base (nullptr: unicast stores to every rank)
  void* const* uc_ptrs; // [world] unicast bases (device array)
  uint32_t* flags;      // [9] buffer flags
  int rank;
  int M;
};

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ uint32_t sanitize(uint32_t w) {
  // bf16 -0 (0x8000) -> +0, so no payload word can equal the sentinel.
  if ((w & 0xffffu) == 0x8000u) w &= 0xffff0000u;
  if ((w >> 16) == 0x8000u) w &= 0x0000ffffu;
  return w;
}

__device__ __forceinline__ uint4 ld_volatile(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ bool dirty(const uint4& v) {
  return v.x == kNegZeroF32 || v.y == kNegZeroF32 || v.z == kNegZeroF32 ||
         v.w == kNegZeroF32;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem)
               : "memory");
}

// FlashInfer LamportFlags (trtllm_mnnvl_allreduce.cuh), non-CGA variant:
// flags = {cur idx, dirty idx, bytes per buffer, dirty stages,
//          bytes to clear per dirty stage [4], arrival counter}.
struct Lamport {
  uint32_t cur, dirty_idx, bytes_per_buffer, dirty_stages;
  uint32_t clear_bytes[4];

  __device__ explicit Lamport(const uint32_t* f) {
    const uint4 a = reinterpret_cast<const uint4*>(f)[0];
    const uint4 b = reinterpret_cast<const uint4*>(f)[1];
    cur = a.x;
    dirty_idx = a.y;
    bytes_per_buffer = a.z;
    dirty_stages = a.w;
    clear_bytes[0] = b.x;
    clear_bytes[1] = b.y;
    clear_bytes[2] = b.z;
    clear_bytes[3] = b.w;
  }

  __device__ char* cur_buf(void* base) const {
    return static_cast<char*>(base) + static_cast<size_t>(cur) * bytes_per_buffer;
  }

  // Every thread of the grid takes part (1-D rasterized).
  __device__ void clear_dirty(void* base, uint32_t gtid, uint32_t nthreads) const {
    if (dirty_stages == 0) return;
    const uint32_t stage_bytes = bytes_per_buffer / dirty_stages;
    const uint4 init = make_uint4(kNegZeroF32, kNegZeroF32, kNegZeroF32, kNegZeroF32);
    for (uint32_t st = 0; st < dirty_stages && st < 4; ++st) {
      uint4* p = reinterpret_cast<uint4*>(
          static_cast<char*>(base) +
          static_cast<size_t>(dirty_idx * dirty_stages + st) * stage_bytes);
      const uint32_t n = (clear_bytes[st] + 15) / 16;
      for (uint32_t i = gtid; i < n; i += nthreads) p[i] = init;
    }
  }
};

__device__ __forceinline__ void arrive(uint32_t* counter) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(counter), "r"(1)
               : "memory");
#else
  asm volatile("red.release.global.gpu.add.u32 [%0], %1;" ::"l"(counter), "r"(1)
               : "memory");
#endif
}

// NTHR threads, HIT iterations per thread, HPS = NTHR * HIT columns per split
// (TileLang: <= 4 tokens 256 x 1 over 16 splits, else 128 x 4 over 8 splits).
// A CTA covers TB tokens x TILE_N outputs of one split.
template <int WORLD, int NTHR, int HIT, int TB, int TILE_N>
__global__ void __launch_bounds__(NTHR) mhc_ar_post_kernel(Params p) {
  constexpr int HPS = NTHR * HIT;
  constexpr int NWARPS = NTHR / 32;
  constexpr int VEC_PER_TOK = HPS / 8;       // 16 B vectors per token slice
  constexpr int NVEC = TB * VEC_PER_TOK;
  constexpr bool kFnReg = TILE_N * kHC * HIT <= 96;
  static_assert(kMix % TILE_N == 0, "TILE_N must divide 24");

  __shared__ __align__(16) __nv_bfloat16 xs[TB][HPS];
  __shared__ __align__(16) __nv_bfloat16 rs[TB][kHC][HPS];
  __shared__ float s_post[TB][kHC];
  __shared__ float s_comb[TB][kHC * kHC];
  __shared__ float s_warp[TB][NWARPS][TILE_N + 1];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int mb = blockIdx.x, nt = blockIdx.y, s = blockIdx.z;
  const int NT = gridDim.y;
  const int m0 = mb * TB;
  const int h_split = s * HPS;
  const int o0 = nt * TILE_N;

  // fn is a weight: load it before waiting on the producer grid.
  float f[kFnReg ? TILE_N : 1][kHC][kFnReg ? HIT : 1];
  if constexpr (kFnReg) {
#pragma unroll
    for (int n = 0; n < TILE_N; ++n)
#pragma unroll
      for (int j = 0; j < kHC; ++j)
#pragma unroll
        for (int it = 0; it < HIT; ++it)
          f[n][j][it] = __ldg(p.fn + (static_cast<int64_t>(o0 + n) * kHC + j) * kHidden +
                              h_split + it * NTHR + tid);
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif

  // ---- residual / post / comb of the CTA's tokens -> smem (async, overlaps
  // the all-reduce).
  for (int v = tid; v < TB * kHC * VEC_PER_TOK; v += NTHR) {
    const int tb = v / (kHC * VEC_PER_TOK), r = v % (kHC * VEC_PER_TOK);
    const int k = r / VEC_PER_TOK, e = r % VEC_PER_TOK;
    const int m = m0 + tb;
    if (m < p.M)
      cp_async16(&rs[tb][k][e * 8],
                 p.residual + (static_cast<int64_t>(m) * kHC + k) * kHidden + h_split + e * 8);
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  if (tid < TB * (kHC + kHC * kHC)) {
    const int tb = tid / (kHC + kHC * kHC), r = tid % (kHC + kHC * kHC);
    const int m = m0 + tb;
    if (m < p.M) {
      if (r < kHC)
        s_post[tb][r] = p.post[m * kHC + r];
      else
        s_comb[tb][r - kHC] = p.comb[m * kHC * kHC + (r - kHC)];
    }
  }

  // ---- all-reduce of the CTA's x slice into xs.
  const int64_t tok_stride = static_cast<int64_t>(kHidden) * WORLD;  // Lamport [token][rank][h]
  if constexpr (WORLD == 1) {
    for (int v = tid; v < NVEC; v += NTHR) {
      const int tb = v / VEC_PER_TOK, e = v % VEC_PER_TOK;
      const int m = m0 + tb;
      if (m >= p.M) continue;
      const uint4 a = *reinterpret_cast<const uint4*>(p.x + static_cast<int64_t>(m) * kHidden +
                                                      h_split + e * 8);
      const uint32_t w[4] = {a.x, a.y, a.z, a.w};
      uint32_t o[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float lo = 0.f + bf_lo(w[q]), hi = 0.f + bf_hi(w[q]);
        const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
        o[q] = *reinterpret_cast<const uint32_t*>(&b);
      }
      *reinterpret_cast<uint4*>(&xs[tb][e * 8]) = make_uint4(o[0], o[1], o[2], o[3]);
    }
  } else {
    const Lamport lam(p.flags);
    const int nblocks = gridDim.x * gridDim.y * gridDim.z;
    const int cta = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    // Push: vector v of the slice goes out from tile (v % NT).
    for (int v = tid; v < NVEC; v += NTHR) {
      if (v % NT != nt) continue;
      const int tb = v / VEC_PER_TOK, e = v % VEC_PER_TOK;
      const int m = m0 + tb;
      if (m >= p.M) continue;
      const int h = h_split + e * 8;
      uint4 a = *reinterpret_cast<const uint4*>(p.x + static_cast<int64_t>(m) * kHidden + h);
      a = make_uint4(sanitize(a.x), sanitize(a.y), sanitize(a.z), sanitize(a.w));
      const int64_t off = (m * tok_stride + static_cast<int64_t>(p.rank) * kHidden + h) * 2;
      if (p.mcast != nullptr) {
        *reinterpret_cast<uint4*>(lam.cur_buf(p.mcast) + off) = a;
      } else {
#pragma unroll
        for (int r = 0; r < WORLD; ++r)
          *reinterpret_cast<uint4*>(lam.cur_buf(p.uc_ptrs[r]) + off) = a;
      }
    }
    // CTA arrival (after reading the flags), as FlashInfer's ctaArrive.
    __syncthreads();
    if (tid == 0) arrive(p.flags + 8);
    void* local = p.uc_ptrs[p.rank];
    lam.clear_dirty(local, static_cast<uint32_t>(cta) * NTHR + tid,
                    static_cast<uint32_t>(nblocks) * NTHR);

    // Poll the remote ranks' slices, then reduce in rank order.
    const char* cur_local = lam.cur_buf(local);
    for (int v = tid; v < NVEC; v += NTHR) {
      const int tb = v / VEC_PER_TOK, e = v % VEC_PER_TOK;
      const int m = m0 + tb;
      if (m >= p.M) continue;
      const int h = h_split + e * 8;
      const uint4 own = *reinterpret_cast<const uint4*>(p.x + static_cast<int64_t>(m) * kHidden + h);
      uint4 vals[WORLD];
      const char* base = cur_local + (m * tok_stride + h) * 2;
      bool ready;
      do {
        ready = true;
#pragma unroll
        for (int r = 0; r < WORLD; ++r) {
          if (r == p.rank) continue;
          vals[r] = ld_volatile(base + static_cast<int64_t>(r) * kHidden * 2);
          ready &= !dirty(vals[r]);
        }
      } while (!ready);
      float acc[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) acc[q] = 0.f;
#pragma unroll
      for (int r = 0; r < WORLD; ++r) {
        const uint4 a = r == p.rank ? own : vals[r];
        const uint32_t w[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          acc[2 * q] = acc[2 * q] + bf_lo(w[q]);
          acc[2 * q + 1] = acc[2 * q + 1] + bf_hi(w[q]);
        }
      }
      uint32_t o[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const __nv_bfloat162 b = __floats2bfloat162_rn(acc[2 * q], acc[2 * q + 1]);
        o[q] = *reinterpret_cast<const uint32_t*>(&b);
      }
      *reinterpret_cast<uint4*>(&xs[tb][e * 8]) = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // Nothing below waits on other CTAs or ranks.
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif

  if (p.x_out != nullptr && nt == 0) {
    for (int v = tid; v < NVEC; v += NTHR) {
      const int tb = v / VEC_PER_TOK, e = v % VEC_PER_TOK;
      const int m = m0 + tb;
      if (m < p.M)
        *reinterpret_cast<uint4*>(p.x_out + static_cast<int64_t>(m) * kHidden + h_split + e * 8) =
            *reinterpret_cast<const uint4*>(&xs[tb][e * 8]);
    }
  }

  // ---- hc_post + prenorm GEMM partials, TileLang's operation order.
#pragma unroll
  for (int tb = 0; tb < TB; ++tb) {
    const int m = m0 + tb;
    if (m >= p.M) break;  // CTA-uniform
    float pm[kHC], cm[kHC][kHC];
#pragma unroll
    for (int j = 0; j < kHC; ++j) pm[j] = s_post[tb][j];
#pragma unroll
    for (int k = 0; k < kHC; ++k)
#pragma unroll
      for (int j = 0; j < kHC; ++j) cm[k][j] = s_comb[tb][k * kHC + j];
    float acc[TILE_N];
#pragma unroll
    for (int n = 0; n < TILE_N; ++n) acc[n] = 0.f;
    float sqr = 0.f;
#pragma unroll
    for (int it = 0; it < HIT; ++it) {
      const int hl = it * NTHR + tid;
      const float xv = __bfloat162float(xs[tb][hl]);
      float r[kHC];
#pragma unroll
      for (int k = 0; k < kHC; ++k) r[k] = __bfloat162float(rs[tb][k][hl]);
      float nr[kHC];
#pragma unroll
      for (int j = 0; j < kHC; ++j) {
        // TileLang's `new_r = pm * x; new_r += cm[k] * r[k]` as nvcc contracts
        // it (SASS of mhc_fused_tilelang): fma(pm, x, cm0 * r0), then
        // fma(cm_k, r_k, .) for k = 1..3.
#if GLM_MHCAR_X_FIRST
        float t = __fmul_rn(pm[j], xv);
#pragma unroll
        for (int k = 0; k < kHC; ++k) t = __fmaf_rn(cm[k][j], r[k], t);
#else
        float t = __fmaf_rn(pm[j], xv, __fmul_rn(cm[0][j], r[0]));
#pragma unroll
        for (int k = 1; k < kHC; ++k) t = __fmaf_rn(cm[k][j], r[k], t);
#endif
        nr[j] = t;
      }
      if (nt == 0) {
#pragma unroll
        for (int j = 0; j < kHC; ++j) {
          p.residual_out[(static_cast<int64_t>(m) * kHC + j) * kHidden + h_split + hl] =
              __float2bfloat16_rn(nr[j]);
          sqr = __fmaf_rn(nr[j], nr[j], sqr);
        }
      }
#pragma unroll
      for (int n = 0; n < TILE_N; ++n) {
#pragma unroll
        for (int j = 0; j < kHC; ++j) {
          float w;
          if constexpr (kFnReg)
            w = f[n][j][it];
          else
            w = __ldg(p.fn + (static_cast<int64_t>(o0 + n) * kHC + j) * kHidden + h_split + hl);
          acc[n] = __fmaf_rn(w, nr[j], acc[n]);
        }
      }
    }
#pragma unroll
    for (int n = 0; n < TILE_N; ++n) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        acc[n] = acc[n] + __shfl_xor_sync(0xffffffffu, acc[n], off);
    }
    if (nt == 0) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) sqr = sqr + __shfl_xor_sync(0xffffffffu, sqr, off);
    }
    if (lane == 0) {
#pragma unroll
      for (int n = 0; n < TILE_N; ++n) s_warp[tb][warp][n] = acc[n];
      if (nt == 0) s_warp[tb][warp][TILE_N] = sqr;
    }
  }
  __syncthreads();
  if (tid < TB * (TILE_N + 1)) {
    const int tb = tid / (TILE_N + 1), n = tid % (TILE_N + 1);
    const int m = m0 + tb;
    if (m < p.M && (n < TILE_N || nt == 0)) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NWARPS; ++w) v = v + s_warp[tb][w][n];
      if (n < TILE_N)
        p.yp[(static_cast<int64_t>(s) * p.M + m) * kMix + o0 + n] = v;
      else
        p.rp[static_cast<int64_t>(s) * p.M + m] = v;
    }
  }

  if constexpr (WORLD > 1) {
    // FlashInfer waitAndUpdate: CTA 0 rotates the buffers once every CTA has
    // arrived (i.e. read the flags of this call).
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid == 0) {
      const uint32_t target = gridDim.x * gridDim.y * gridDim.z;
      volatile uint32_t* cnt = p.flags + 8;
      while (*cnt < target) {
      }
      const Lamport lam(p.flags);
      reinterpret_cast<uint4*>(p.flags)[0] =
          make_uint4((lam.cur + 1) % 3, lam.cur, lam.bytes_per_buffer, 1u);
      reinterpret_cast<uint4*>(p.flags)[1] =
          make_uint4(static_cast<uint32_t>(p.M) * kHidden * WORLD * 2u, 0u, 0u, 0u);
      *cnt = 0;
    }
  }
}

// Resident CTAs per SM of an instantiation, and the SM count (cached).
template <int WORLD, int NTHR, int HIT, int TB, int TILE_N>
int resident_ctas(int device) {
  static int cap = 0;
  if (cap == 0) {
    int per_sm = 0, sms = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, mhc_ar_post_kernel<WORLD, NTHR, HIT, TB, TILE_N>, NTHR, 0);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cap = per_sm * sms;
  }
  return cap;
}

template <int WORLD, int NTHR, int HIT, int TB, int TILE_N>
void launch(const Params& p, int64_t splits, bool pdl, int64_t resident_ranks, int device,
            cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((p.M + TB - 1) / TB, kMix / TILE_N, static_cast<unsigned>(splits));
  // Polling CTAs wait for pushes from other CTAs of the grid. Other kernels
  // that hold SMs meanwhile (the PDL-overlapped producer, side streams) finish
  // on their own, so the grid only must not be able to fill the GPU by
  // itself: keep it within half of its resident capacity. Tests that run
  // several ranks' grids on one GPU (resident_ranks > 1) need all of them
  // resident at once.
  const int64_t ctas = static_cast<int64_t>(cfg.gridDim.x) * cfg.gridDim.y * cfg.gridDim.z;
  if (WORLD > 1) {
    const int cap = resident_ctas<WORLD, NTHR, HIT, TB, TILE_N>(device);
    const int64_t need = resident_ranks > 1 ? ctas * resident_ranks : 2 * ctas;
    TVM_FFI_ICHECK(need <= cap)
        << "mhc_ar_post grid of " << ctas << " CTAs (x" << resident_ranks
        << " ranks per GPU) does not fit the resident limit (" << cap
        << " CTAs, half of it for one rank); use more tokens or outputs per CTA";
  }
  cfg.blockDim = dim3(NTHR);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, mhc_ar_post_kernel<WORLD, NTHR, HIT, TB, TILE_N>, p);
  TVM_FFI_ICHECK(e == cudaSuccess) << "mhc_ar_post launch failed: " << cudaGetErrorString(e);
}

template <int WORLD>
bool dispatch_cfg(const Params& p, int64_t nthr, int64_t splits, int64_t tb, int64_t tile_n,
                  bool pdl, int64_t resident_ranks, int device, cudaStream_t stream) {
#define GLM_MHCAR_CASE(NTHR, HIT, TB, TN)                                                  \
  if (nthr == NTHR && tb == TB && tile_n == TN) {                                         \
    launch<WORLD, NTHR, HIT, TB, TN>(p, splits, pdl, resident_ranks, device, stream);     \
    return true;                                                                          \
  }
  // <= 4 tokens: TileLang's 256 threads x 1 iteration over 16 splits.
  GLM_MHCAR_CASE(256, 1, 1, 2)
  GLM_MHCAR_CASE(256, 1, 1, 6)
  GLM_MHCAR_CASE(256, 1, 1, 24)
  GLM_MHCAR_CASE(256, 1, 2, 6)
  GLM_MHCAR_CASE(256, 1, 4, 6)
  // 5-32 tokens: 128 threads x 4 iterations over 8 splits.
  GLM_MHCAR_CASE(128, 4, 1, 2)
  GLM_MHCAR_CASE(128, 4, 1, 6)
  GLM_MHCAR_CASE(128, 4, 2, 6)
  GLM_MHCAR_CASE(128, 4, 4, 6)
  GLM_MHCAR_CASE(128, 4, 2, 3)
  GLM_MHCAR_CASE(128, 4, 4, 3)
#undef GLM_MHCAR_CASE
  return false;
}

#define GLM_MHCAR_CHECK(t, code)                               \
  TVM_FFI_ICHECK(encode_dlpack_dtype((t).dtype()) == (code) && \
                 (t).IsContiguous())                           \
      << #t " must be a contiguous " #code " tensor"

// x [M, 4096] local partial; residual [M, 4, 4096]; post [M, 4]; comb [M, 16];
// fn [24, 16384]; yp [S, M, 24]; rp [S, M]; residual_out [M, 4, 4096];
// x_out: [M, 4096] or empty. Workspace pointers are ints (0 = absent).
void mhc_ar_post(TensorView x, TensorView residual, TensorView post, TensorView comb,
                 TensorView fn, TensorView yp, TensorView rp, TensorView residual_out,
                 TensorView x_out, int64_t mcast_ptr, int64_t uc_ptrs_dev, TensorView flags,
                 int64_t world, int64_t rank, int64_t nthr, int64_t tb, int64_t tile_n,
                 int64_t pdl, int64_t resident_ranks) {
  GLM_MHCAR_CHECK(x, bfloat16_code);
  GLM_MHCAR_CHECK(residual, bfloat16_code);
  GLM_MHCAR_CHECK(post, float32_code);
  GLM_MHCAR_CHECK(comb, float32_code);
  GLM_MHCAR_CHECK(fn, float32_code);
  GLM_MHCAR_CHECK(yp, float32_code);
  GLM_MHCAR_CHECK(rp, float32_code);
  GLM_MHCAR_CHECK(residual_out, bfloat16_code);
  const int64_t M = x.size(0);
  TVM_FFI_ICHECK(x.numel() == M * kHidden) << "x must be [M, 4096]";
  TVM_FFI_ICHECK(residual.numel() == M * kK && residual_out.numel() == M * kK);
  TVM_FFI_ICHECK(post.numel() == M * kHC && comb.numel() == M * kHC * kHC);
  TVM_FFI_ICHECK(fn.numel() == static_cast<int64_t>(kMix) * kK);
  TVM_FFI_ICHECK(yp.ndim() == 3 && yp.size(1) == M && yp.size(2) == kMix);
  const int64_t S = yp.size(0);
  TVM_FFI_ICHECK(rp.numel() == S * M);
  TVM_FFI_ICHECK(S * nthr * (nthr == 256 ? 1 : 4) == kHidden)
      << "splits x threads x iterations must cover 4096 columns";
  TVM_FFI_ICHECK(M >= 1 && M <= 65535);
  const bool has_x_out = x_out.numel() > 0;
  if (has_x_out) {
    GLM_MHCAR_CHECK(x_out, bfloat16_code);
    TVM_FFI_ICHECK(x_out.numel() == M * kHidden);
  }
  if (world > 1) {
    TVM_FFI_ICHECK(uc_ptrs_dev != 0) << "world > 1 needs the workspace";
    TVM_FFI_ICHECK(flags.numel() >= 9 && flags.dtype().bits == 32 && flags.IsContiguous())
        << "flags must be the workspace's 32-bit buffer flags";
    TVM_FFI_ICHECK(rank >= 0 && rank < world);
  }

  Params p;
  p.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
  p.residual = static_cast<const __nv_bfloat16*>(residual.data_ptr());
  p.post = static_cast<const float*>(post.data_ptr());
  p.comb = static_cast<const float*>(comb.data_ptr());
  p.fn = static_cast<const float*>(fn.data_ptr());
  p.yp = static_cast<float*>(yp.data_ptr());
  p.rp = static_cast<float*>(rp.data_ptr());
  p.residual_out = static_cast<__nv_bfloat16*>(residual_out.data_ptr());
  p.x_out = has_x_out ? static_cast<__nv_bfloat16*>(x_out.data_ptr()) : nullptr;
  p.mcast = reinterpret_cast<void*>(mcast_ptr);
  p.uc_ptrs = reinterpret_cast<void* const*>(uc_ptrs_dev);
  p.flags = world > 1 ? static_cast<uint32_t*>(flags.data_ptr()) : nullptr;
  p.rank = static_cast<int>(rank);
  p.M = static_cast<int>(M);

  const int dev = x.device().device_id;
  ffi::CUDADeviceGuard device_guard(dev);
  const cudaStream_t stream = get_stream(x.device());
  bool ok = false;
  // GLM_MHCAR_WORLD (set by mhc_ar_post.py) builds one world size per module.
  TVM_FFI_ICHECK(world == GLM_MHCAR_WORLD)
      << "module built for world size " << GLM_MHCAR_WORLD << ", called with " << world;
  ok = dispatch_cfg<GLM_MHCAR_WORLD>(p, nthr, S, tb, tile_n, pdl, resident_ranks, dev, stream);
  TVM_FFI_ICHECK(ok) << "unsupported mhc_ar_post config: world " << world << ", threads "
                     << nthr << ", tokens/CTA " << tb << ", tile_n " << tile_n;
}

}  // namespace glm5next_mhcar

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_ar_post, glm5next_mhcar::mhc_ar_post);
