# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Decode MXFP8 routed MoE in one launch (routing + FC1 + SwiGLU + FC2), SM100+.

Replaces the trtllm-gen chain (routing kernel -> FC1 bmm with fused SwiGLU and
MXFP8 requant -> FC2 bmm) for small token counts (decode with MTP: T <= ~24)
with one persistent kernel. It reads the weights and scales exactly as
prepared for trtllm-gen (``_shuffle_mxfp8_moe_weights``: gate/up interleave and
32-row shuffle of K-major rows, 128x4-interleaved E8M0 scales) and produces the
deferred (``do_finalize=False``) output contract: unweighted BF16 GEMM2 rows per
(token, k), BF16 top-k weights and the permute map ``idx[t, k] = t * top_k + k``,
which the fused finalize consumer (``moe_finalize``) reduces in k order.

Per CTA (grid = #SMs, one CTA per SM, 8 consumer warps + 1 producer warp):

1. Before ``griddepcontrol.wait``: barrier init only (weights depend on routing).
2. Routing, redundantly in every CTA: top-k of the router logits (ties to the
   lower expert; trtllm-gen's packed-key order), softmax over the top-k in fp32
   (Renormalize / RenormalizeNaive), distinct experts sorted ascending =
   "slots", per slot the routes in token order. CTA 0 writes the weights.
3. Work units of 33,792 bytes (weights + scales): FC1 unit = 16 rows x 2048 of
   one slot's w13; FC2 unit = 64 rows x 512 of its w2. FC1 units of all slots
   come first, split evenly over the CTAs, then FC2 units (reversed CTA order).
   The producer warp streams each unit into a ring of SMEM slots with 1-D TMA
   bulk copies (one per row, rows padded by 32 B so the MMA fragment loads are
   conflict-free) and an mbarrier per slot; FC2 weights therefore stream while
   FC1 still computes.
4. MMA: mma.sync m16n8k16 f16 (fp8 -> f16 is exact), swap-AB: 16 weight rows x
   8 routes per tile; each 32-wide K block is one MXFP8 scale block, applied in
   fp32 to the block's partial (weight scale x activation scale).
5. FC1 writes fp32 gate/up values to an L2 scratch; the last FC1 unit of a slot
   (atomic counter) computes SwiGLU + MXFP8 requant for the slot's routes and
   publishes a ready flag; FC2 units of that slot wait on it (acquire) and read
   the quantized intermediate from L2.
6. The last CTA to finish resets the counters; ``griddepcontrol.launch_dependents``
   after the last store.

Numerics: not bitwise to trtllm-gen (different K accumulation); same requant
(block-32 E8M0 + E4M3 RN, see ``QMODE``) and the same routing arithmetic.

Env: ``VLLM_MOE_DECODE_MAX_TOKENS`` (default 0 = off): use this kernel for
MXFP8 trtllm MoE calls with 1 <= T <= this (max 32) that would defer finalize.
"""

import hashlib
import os

import torch

_ext: list = []
_instances: dict = {}  # device index -> MoeDecode | None (build failed)

MAX_TOKENS = int(os.environ.get("VLLM_MOE_DECODE_MAX_TOKENS", "0"))
# SwiGLU-output requant: 1 (default) = trtllm-gen's E8M0 = exponent(amax) - 8 with a saturating
# E4M3 cast (VR probe: 1e-6 relL2 to the chain's GEMM2 rows); 0 = OCP ceil(log2(amax / 448)).
QMODE = int(os.environ.get("VLLM_MOE_DECODE_QMODE", "1"))
# Slots in the SMEM ring (0 = as many as fit).
NSLOTS = int(os.environ.get("VLLM_MOE_DECODE_SLOTS", "0"))
# Persistent grid (0 = #SMs).
GRID = int(os.environ.get("VLLM_MOE_DECODE_GRID", "0"))
# Build macros "K=V+K=V" (MDC_FAST_EXP, ...).
TUNE = os.environ.get("VLLM_MOE_DECODE_TUNE", "")

_SOURCE = r"""
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef MDC_DEBUG
#define MDC_DEBUG 0
#endif
#ifndef MDC_FAST_EXP
#define MDC_FAST_EXP 0
#endif

namespace mdc {
constexpr int E = 256, TOPK = 8, H = 2048, I = 512;
constexpr int KB1 = H / 32, KB2 = I / 32;
constexpr int TW = 4;                      // warps per team = m-tiles (16 rows) per unit
constexpr int NTEAM = 3;                   // consumer teams
constexpr int NCW = TW * NTEAM;            // consumer warps
constexpr int NTHR = (NCW + 1) * 32;       // + producer warp
constexpr int BOX = 64 * 128;              // TMA box: 64 rows x 128 B (128B swizzle)
constexpr int SLOT = 4 * BOX + 2048;       // unit = 64 rows x 512 B (4 boxes) + 4 groups x 512 B of E8M0 scales
constexpr int XROW = 2 * H + 64;           // f16 activation row stride
constexpr int MAXT = 32, MAXR = MAXT * TOPK;
constexpr int KC = H / I;                  // FC1 K chunks (4)
constexpr int U1 = 16 * KC, U2 = 32;       // units per expert: FC1 16 row groups x 4 K chunks, FC2 32 row groups
constexpr int SMALL = 5120;                // fixed SMEM header

struct Params {
  CUtensorMap tm13, tm2;                     // 2-D maps of w13 [E*2I, H] and w2 [E*H, I] (u8), box 128 B x 64 rows
  const void* logits;
  const uint8_t* x; const uint8_t* xs;
  const uint8_t* w13; const uint8_t* s13; const uint8_t* w2; const uint8_t* s2;
  const int16_t* perm13; const int16_t* perm2;
  __nv_bfloat16* out; __nv_bfloat16* topk_w; int32_t* topk_ids;
  float* hpart; uint8_t* actq; uint8_t* acts;   // hpart: [KC][MAXR][2I] FC1 partial sums per K chunk
  int* cnt; int* ready; int* done;
  long long* prof;                           // optional [G][64] %globaltimer stamps / sums (microbench only)
  int T; int ns; int qmode;
};

struct Hdr {                                 // fixed SMEM header (< SMALL bytes)
  unsigned long long full[16], empty[16];
  uint32_t tokmask[E];                       // per expert: tokens that selected it
  int16_t route_e[MAXR];
  int16_t slot_e[MAXR];
  int16_t slot_n[MAXR];
  int16_t slot_off[MAXR];
  int16_t exp_off[E];
  uint8_t slot_routes[MAXR];
  uint32_t selw[8];
  int warp_tot[8];
  int D;
};

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(unsigned long long* b, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(su32(b)), "r"(c));
}
__device__ __forceinline__ void mbar_expect_tx(unsigned long long* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(su32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned long long* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(su32(b)) : "memory");
}
__device__ __forceinline__ bool mbar_try(unsigned long long* b, int parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n }"
               : "=r"(ok) : "r"(su32(b)), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(unsigned long long* b, int parity, int tag) {
#if MDC_DEBUG
  long long n = 0;
  while (!mbar_try(b, parity)) {
    if (++n == 20000000) printf("mdc hang: blk %d tid %d tag %d parity %d\n", blockIdx.x, threadIdx.x, tag, parity);
  }
#else
  while (!mbar_try(b, parity)) {}
#endif
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, unsigned long long* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(bar)) : "memory");
}
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* map, int x, int y, unsigned long long* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(su32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(su32(bar)) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t cvt_lo(uint32_t v) {   // bytes 0,1 (e4m3x2) -> f16x2
  uint32_t r; asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; cvt.rn.f16x2.e4m3x2 %0, l; }" : "=r"(r) : "r"(v)); return r;
}
__device__ __forceinline__ uint32_t cvt_hi(uint32_t v) {   // bytes 2,3
  uint32_t r; asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; cvt.rn.f16x2.e4m3x2 %0, h; }" : "=r"(r) : "r"(v)); return r;
}
__device__ __forceinline__ void hmma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                     uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// One 32-wide K block: rows g / g+8 (8 bytes each), route g (8 values), all at the same physical K positions.
// Bytes 0..3 feed the first k16 MMA (logical k 2t,2t+1 <- bytes 0,1; 2t+8,2t+9 <- bytes 2,3), 4..7 the second;
// b0lo/b0hi/b1lo/b1hi = the route's f16x2 of bytes (0,1), (2,3), (4,5), (6,7).
__device__ __forceinline__ void mma_k32(float (&d)[4], uint2 ag, uint2 ag8, uint32_t b0lo, uint32_t b0hi,
                                        uint32_t b1lo, uint32_t b1hi) {
  d[0] = d[1] = d[2] = d[3] = 0.f;
  hmma(d, cvt_lo(ag.x), cvt_lo(ag8.x), cvt_hi(ag.x), cvt_hi(ag8.x), b0lo, b0hi);
  hmma(d, cvt_lo(ag.y), cvt_lo(ag8.y), cvt_hi(ag.y), cvt_hi(ag8.y), b1lo, b1hi);
}
// Shuffled row -> logical row of the trtllm-gen MXFP8 weights (_shuffle_mxfp8_moe_weights: 32-row block shuffle
// srcToDstBlk32RowMap; for w13 after the gate/up interleave of swap_w13_to_w31). Checked against the traced
// permutation at init (moe_decode_cuda.permutations).
__device__ __forceinline__ int unshuffle32(int n) { return (n & ~31) | ((n & 7) << 2) | ((n & 31) >> 3); }
__device__ __forceinline__ int perm13(int n) {   // -> index into [gate(I) | up(I)]
  const int m = unshuffle32(n);
  return (m & 1) ? (m >> 1) : I + (m >> 1);
}
__device__ __forceinline__ int perm2(int n) { return unshuffle32(n); }
__device__ __forceinline__ float e8(uint32_t e) { return __uint_as_float(e << 23); }

__device__ __forceinline__ uint32_t twiddle16(uint32_t b) { return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u); }
__device__ __forceinline__ float untwiddle16(uint32_t v) {
  uint32_t b = (v & 0x8000u) ? (v & 0x7FFFu) : (~v & 0xFFFFu);
  return __uint_as_float(b << 16);
}
__device__ __forceinline__ uint32_t twiddle32(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
__device__ __forceinline__ float untwiddle32(uint32_t v) {
  return __uint_as_float((v & 0x80000000u) ? (v & 0x7FFFFFFFu) : ~v);
}
__device__ __forceinline__ long long gtime() {
  long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

// SwiGLU + MXFP8 requant of one slot's routes (by the warp that completed the slot's FC1):
// h = ((p0 + p1) + p2) + p3 over the K-chunk partials, act = silu(gate) * up, E8M0 per 32 + E4M3 RN satfinite.
__device__ void slot_act(const Params& p, const Hdr& h, int s, int lane) {
  const int n = h.slot_n[s], off = h.slot_off[s];
  for (int ri = 0; ri < n; ++ri) {
    const int r = h.slot_routes[off + ri];
    float gg[16], uu[16];
#pragma unroll
    for (int c = 0; c < KC; ++c) {
      const float* hp = p.hpart + ((size_t)c * MAXR + r) * (2 * I) + lane * 16;
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const float4 g4 = __ldcg(reinterpret_cast<const float4*>(hp) + v);
        const float4 u4 = __ldcg(reinterpret_cast<const float4*>(hp + I) + v);
        if (c == 0) {
          gg[4 * v] = g4.x; gg[4 * v + 1] = g4.y; gg[4 * v + 2] = g4.z; gg[4 * v + 3] = g4.w;
          uu[4 * v] = u4.x; uu[4 * v + 1] = u4.y; uu[4 * v + 2] = u4.z; uu[4 * v + 3] = u4.w;
        } else {
          gg[4 * v] += g4.x; gg[4 * v + 1] += g4.y; gg[4 * v + 2] += g4.z; gg[4 * v + 3] += g4.w;
          uu[4 * v] += u4.x; uu[4 * v + 1] += u4.y; uu[4 * v + 2] += u4.z; uu[4 * v + 3] += u4.w;
        }
      }
    }
    float a[16];
    float amax = 0.f;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
#if MDC_FAST_EXP
      const float sg = gg[c] / (1.f + __expf(-gg[c]));
#else
      const float sg = gg[c] / (1.f + expf(-gg[c]));
#endif
      a[c] = sg * uu[c];
      amax = fmaxf(amax, fabsf(a[c]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));   // lanes 2m, 2m+1 = block m
    int e;
    if (p.qmode == 1) {
      e = (int)((__float_as_uint(amax) >> 23) & 0xFF) - 8;
    } else {
      uint32_t bits = __float_as_uint(fmaxf(amax / 448.f, 1.17549435e-38f));
      e = (int)((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0);
    }
    e = min(max(e, 1), 253);
    const float inv = __uint_as_float((uint32_t)(254 - e) << 23);
    uint32_t q[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      uint16_t lo, hi;
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(a[4 * w + 1] * inv), "f"(a[4 * w] * inv));
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(a[4 * w + 3] * inv), "f"(a[4 * w + 2] * inv));
      q[w] = (uint32_t)lo | ((uint32_t)hi << 16);
    }
    __stcg(reinterpret_cast<uint4*>(p.actq + (size_t)r * I + lane * 16), make_uint4(q[0], q[1], q[2], q[3]));
    if ((lane & 1) == 0) p.acts[(size_t)r * KB2 + (lane >> 1)] = (uint8_t)e;
  }
  __threadfence();
  __syncwarp();
  if (lane == 0) st_release(p.ready + s, 1);
}

// One warp's FC1 unit of slot s is stored: publish it (fence, count); the warp completing the slot's last
// FC1 unit computes the slot's SwiGLU + requant and releases its ready flag.
__device__ __forceinline__ void fc1_done(const Params& p, const Hdr& h, int s, int lane) {
  __threadfence();
  __syncwarp();
  int old = 0;
  if (lane == 0) old = atomicAdd(p.cnt + s, 1);
  old = __shfl_sync(0xffffffffu, old, 0);
  if (old == U1 * TW - 1) {
    __threadfence();
    slot_act(p, h, s, lane);
  }
}

template <bool kF32Logits>
__global__ void __launch_bounds__(NTHR, 1) moe_decode_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  Hdr& h = *reinterpret_cast<Hdr*>(smem);
  const int T = p.T, R = T * TOPK, ns = p.ns;
  uint8_t* xssm = smem + SMALL;                               // [T][64] activation scales
  uint8_t* x16 = xssm + ((T * 64 + 127) & ~127);              // [T][XROW] activations as f16
  uint8_t* slots = smem + ((SMALL + ((T * 64 + 127) & ~127) + T * XROW + 1023) & ~1023);   // [ns][SLOT], 1 KB aligned
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  long long* prof = p.prof ? p.prof + blockIdx.x * 64 : nullptr;
  if (prof && tid == 0) prof[0] = gtime();

  if (tid == 0) {
    for (int i = 0; i < ns; ++i) { mbar_init(&h.full[i], 1); mbar_init(&h.empty[i], TW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < E; i += NTHR) h.tokmask[i] = 0;
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (prof && tid == 0) prof[1] = gtime();

  // ---- activations -> SMEM: scales as is, E4M3 values as f16 (exact), issued before the routing ----
  {
    const uint4* xg = reinterpret_cast<const uint4*>(p.x);
    for (int i = tid; i < T * (H / 16); i += NTHR) {
      const uint4 v = __ldcg(xg + i);
      const int t = i / (H / 16), k = (i % (H / 16)) * 16;
      uint4 lo = make_uint4(cvt_lo(v.x), cvt_hi(v.x), cvt_lo(v.y), cvt_hi(v.y));
      uint4 hi = make_uint4(cvt_lo(v.z), cvt_hi(v.z), cvt_lo(v.w), cvt_hi(v.w));
      uint4* dst = reinterpret_cast<uint4*>(x16 + t * XROW + 2 * k);
      dst[0] = lo;
      dst[1] = hi;
    }
    for (int i = tid; i < T * 16; i += NTHR)
      reinterpret_cast<uint32_t*>(xssm)[i] = __ldcg(reinterpret_cast<const unsigned int*>(p.xs) + i);
  }

  // ---- routing: top-k + softmax over the top-k, one warp per token ----
  for (int t = warp; t < T; t += NCW + 1) {
    uint32_t key[8];
    if constexpr (kF32Logits) {
      const float4* lp = reinterpret_cast<const float4*>(static_cast<const float*>(p.logits) + (size_t)t * E + lane * 8);
      float4 v0 = __ldcg(lp), v1 = __ldcg(lp + 1);
      float f[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) key[j] = twiddle32(__float_as_uint(f[j]));
    } else {
      uint4 v = __ldcg(reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.logits) + (size_t)t * E + lane * 8));
      uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        key[2 * j] = (twiddle16(w[j] & 0xFFFFu) << 16) | (uint32_t)(65535 - (lane * 8 + 2 * j));
        key[2 * j + 1] = (twiddle16(w[j] >> 16) << 16) | (uint32_t)(65535 - (lane * 8 + 2 * j + 1));
      }
    }
    float myv = -INFINITY;
    int myexp = 0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      int ex;
      float val;
      if constexpr (kF32Logits) {
        uint32_t m = 0; int mj = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) if (key[j] > m) { m = key[j]; mj = j; }
        const uint32_t best = __reduce_max_sync(0xffffffffu, m);
        // ties: lowest expert index = highest (65535 - e) among lanes holding the best value
        const uint32_t cand = (m == best) ? (uint32_t)(65535 - (lane * 8 + mj)) : 0u;
        const uint32_t wi = __reduce_max_sync(0xffffffffu, cand);
        ex = 65535 - (int)wi;
        val = untwiddle32(best);
        if (lane == (ex >> 3)) {
#pragma unroll
          for (int j = 0; j < 8; ++j) if (j == (ex & 7)) key[j] = 0;
        }
      } else {
        uint32_t m = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) m = max(m, key[j]);
        const uint32_t best = __reduce_max_sync(0xffffffffu, m);
        ex = 65535 - (int)(best & 0xFFFFu);
        val = untwiddle16(best >> 16);
        if (lane == (ex >> 3)) {
#pragma unroll
          for (int j = 0; j < 8; ++j) if (j == (ex & 7)) key[j] = 0;
        }
      }
      if (lane == k) { myv = val; myexp = ex; }
    }
    // softmax over the top-k (lanes 0..7 hold rank 0..7), trtllm-gen calcSoftmax order
    float mx = (lane < TOPK) ? myv : -INFINITY;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float ns_ = 0.f;
    if (lane < TOPK) {
#if MDC_FAST_EXP
      ns_ = __expf(myv - mx);
#else
      ns_ = expf(myv - mx);
#endif
    }
    float sum = ns_;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane < TOPK) {
      h.route_e[t * TOPK + lane] = (int16_t)myexp;
      atomicOr(&h.tokmask[myexp], 1u << t);
      if (blockIdx.x == 0) {
        p.topk_w[t * TOPK + lane] = __float2bfloat16_rn(ns_ / sum);
        if (p.topk_ids) p.topk_ids[t * TOPK + lane] = myexp;
      }
    }
  }
  __syncthreads();
  // ---- slots: distinct experts ascending; per expert the routes in token order ----
  if (warp < 8) {
    const int e = warp * 32 + lane;
    const uint32_t tm = h.tokmask[e];
    const uint32_t sel = __ballot_sync(0xffffffffu, tm != 0);
    int c = __popc(tm);
    int inc = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffffu, inc, o); if (lane >= o) inc += v; }
    h.exp_off[e] = (int16_t)(inc - c);                      // within-warp exclusive prefix for now
    if (lane == 31) h.warp_tot[warp] = inc;
    if (lane == 0) h.selw[warp] = sel;
  }
  __syncthreads();
  if (warp < 8) {
    const int e = warp * 32 + lane;
    int base = 0, sbase = 0;
    for (int w = 0; w < warp; ++w) { base += h.warp_tot[w]; sbase += __popc(h.selw[w]); }
    const uint32_t tm = h.tokmask[e];
    const int off = base + h.exp_off[e];
    h.exp_off[e] = (int16_t)off;
    if (tm) {
      const int s = sbase + __popc(h.selw[warp] & ((1u << lane) - 1u));
      h.slot_e[s] = (int16_t)e;
      h.slot_n[s] = (int16_t)__popc(tm);
      h.slot_off[s] = (int16_t)off;
    }
    if (warp == 7 && lane == 31) {
      int d = 0;
      for (int w = 0; w < 8; ++w) d += __popc(h.selw[w]);
      h.D = d;
    }
  }
  __syncthreads();
  for (int r = tid; r < R; r += NTHR) {
    const int e = h.route_e[r], t = r / TOPK;
    h.slot_routes[h.exp_off[e] + __popc(h.tokmask[e] & ((1u << t) - 1u))] = (uint8_t)r;
  }
  __syncthreads();
  if (prof && tid == 0) prof[2] = gtime();

  // ---- work split: FC1 units of all slots in contiguous per-CTA ranges, then FC2 units (reversed CTA order) ----
  const int D = h.D, G = gridDim.x, b = blockIdx.x, b2 = G - 1 - b;
  const int lo1 = (int)((long long)b * (D * U1) / G), n1 = (int)((long long)(b + 1) * (D * U1) / G) - lo1;
  const int lo2 = (int)((long long)b2 * (D * U2) / G), n2 = (int)((long long)(b2 + 1) * (D * U2) / G) - lo2;
  const int nu = n1 + n2;
  // unit i -> (fc2, slot, tile); FC1 tile = row group (64 rows) * KC + K chunk, FC2 tile = row group
  auto unit_at = [&](int i, int& fc2, int& s, int& tile) {
    if (i < n1) { fc2 = 0; s = (lo1 + i) / U1; tile = (lo1 + i) % U1; }
    else { fc2 = 1; s = (lo2 + i - n1) / U2; tile = (lo2 + i - n1) % U2; }
  };

  if (warp == NCW) {
    // ---- producer: one unit = 4 TMA boxes (64 rows x 128 B, 128B swizzle) + its 2 KB of E8M0 scales ----
    if (lane == 0) {
      for (int i = 0; i < nu; ++i) {
        const int st = i % ns;
        const long long tw0 = prof ? gtime() : 0;
        if (i >= ns) mbar_wait(&h.empty[st], ((i / ns) & 1) ^ 1, 1);
        if (prof) prof[40] += gtime() - tw0;
        uint8_t* sl = slots + st * SLOT;
        mbar_expect_tx(&h.full[st], SLOT);
        int fc2, s, tile;
        unit_at(i, fc2, s, tile);
        const int e = h.slot_e[s];
        if (!fc2) {
          const int rg = tile / KC, c = tile % KC;
#pragma unroll
          for (int kc = 0; kc < 4; ++kc) tma2d(sl + kc * BOX, &p.tm13, c * I + kc * 128, e * (2 * I) + rg * 64, &h.full[st]);
          // scale groups 4c..4c+3 of the 128-row tile rg/2 (128x4 interleave: 512 B per group, contiguous)
          bulk_g2s(sl + 4 * BOX, p.s13 + (size_t)e * (2 * I * KB1) + ((rg >> 1) * (KB1 / 4) + 4 * c) * 512, 2048, &h.full[st]);
        } else {
#pragma unroll
          for (int kc = 0; kc < 4; ++kc) tma2d(sl + kc * BOX, &p.tm2, kc * 128, e * H + tile * 64, &h.full[st]);
          bulk_g2s(sl + 4 * BOX, p.s2 + (size_t)e * (H * KB2) + ((tile >> 1) * (KB2 / 4)) * 512, 2048, &h.full[st]);
        }
      }
    }
  } else {
    // ---- consumers: team = warp / TW handles units team, team + C, ...; warp mt = m-tile of the unit ----
    // C = min(NTEAM, ns) teams; ns is a multiple of C, so the units of a slot always go to the same team,
    // which then waits on its slot's barrier one phase at a time (parity waits cannot tell a phase from the
    // one two later).
    const int C = min(NTEAM, ns);
    const int team = warp / TW, mt = warp % TW;
    const int g = lane >> 2, t4 = lane & 3;
    long long acc_t[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // full wait, fc1, -, act, ready wait, fc2, n1|n2<<16, -
    const bool rec = prof && lane == 0 && mt == 0;
    long long tq = rec ? gtime() : 0;
    auto lap = [&](int k) { if (rec) { const long long t = gtime(); acc_t[k] += t - tq; tq = t; } };
    // After its FC1 units every consumer warp joins one CTA barrier; then warp 0 publishes the CTA's FC1 work
    // (one fence covers all the CTA's stores) and computes SwiGLU + requant for the slots it completes.
    bool closed = false;
    auto close_fc1 = [&]() {
      closed = true;
      asm volatile("bar.sync 1, %0;" :: "r"(C * TW * 32) : "memory");
      if (warp == 0 && n1 > 0) {
        const int s0 = lo1 / U1, s1 = (lo1 + n1 - 1) / U1;
        for (int sb = s0; sb <= s1; sb += 32) {
          const int s = sb + lane;
          int old = -1, mine = 0;
          if (s <= s1) {
            mine = min(lo1 + n1, (s + 1) * U1) - max(lo1, s * U1);
            __threadfence();
            old = atomicAdd(p.cnt + s, mine);
          }
          unsigned last = __ballot_sync(0xffffffffu, s <= s1 && old + mine == U1);
          while (last) {
            const int k = __ffs(last) - 1;
            last &= last - 1;
            __threadfence();
            slot_act(p, h, sb + k, lane);
          }
        }
      }
      lap(3);
    };
    for (int i = team; team < C && i < nu; i += C) {
      const int st = i % ns;
      int fc2, s, tile;
      unit_at(i, fc2, s, tile);
      if (fc2 && !closed) close_fc1();
      const int n = h.slot_n[s], off = h.slot_off[s];
      const int rgrp = fc2 ? tile : tile / KC;                 // 64-row group within the expert
      mbar_wait(&h.full[st], (i / ns) & 1, 2);
      if (prof && lane == 0 && i == team && mt == 0) prof[3 + team] = gtime();
      lap(0);
      const uint8_t* sl = slots + st * SLOT;
      // A fragment bytes of row rr, K block kb, thread t4 (128B-swizzled boxes): box kb/4, row rr * 128,
      // 16-byte unit ((kb%4)*2 + t4/2) ^ (rr%8) (= g), + 8 * (t4%2)
      const int rr = mt * 16 + g;
      const uint8_t* A0 = sl + rr * 128 + (t4 & 1) * 8;
      const uint8_t* A1 = A0 + 8 * 128;
      int xo[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) xo[m] = (((m * 2 + (t4 >> 1)) ^ g) << 4);
      // weight scales of rows R0 / R1 (128x4 interleave): group k4 at + k4 * 512, (R%32)*16 + ((R%128)/32)*4
      const int R0 = rgrp * 64 + rr, R1 = R0 + 8;
      const uint8_t* SW = sl + 4 * BOX;
      uint32_t sw0[4], sw1[4];
#pragma unroll
      for (int k4 = 0; k4 < 4; ++k4) {
        sw0[k4] = *reinterpret_cast<const uint32_t*>(SW + k4 * 512 + (R0 & 31) * 16 + ((R0 & 127) >> 5) * 4);
        sw1[k4] = *reinterpret_cast<const uint32_t*>(SW + k4 * 512 + (R1 & 31) * 16 + ((R1 & 127) >> 5) * 4);
      }
      if (!fc2) {
        const int c = tile % KC;
        const int hid0 = perm13(R0), hid1 = perm13(R1);
        float* hp = p.hpart + (size_t)c * MAXR * (2 * I);
        for (int nt = 0; nt * 8 < n; ++nt) {
          const int rb = h.slot_routes[off + min(nt * 8 + g, n - 1)];
          const int c0 = nt * 8 + 2 * t4, c1 = c0 + 1;
          const int r0 = h.slot_routes[off + min(c0, n - 1)], r1 = h.slot_routes[off + min(c1, n - 1)];
          const uint8_t* B = x16 + (rb / TOPK) * XROW + c * (2 * I) + t4 * 16;
          const uint8_t* XS0 = xssm + (r0 / TOPK) * 64 + c * 16;
          const uint8_t* XS1 = xssm + (r1 / TOPK) * 64 + c * 16;
          float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int k4 = 0; k4 < 4; ++k4) {
            const uint32_t sx0 = *reinterpret_cast<const uint32_t*>(XS0 + k4 * 4);
            const uint32_t sx1 = *reinterpret_cast<const uint32_t*>(XS1 + k4 * 4);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
              const int kb = k4 * 4 + kk;
              const uint2 ag = *reinterpret_cast<const uint2*>(A0 + k4 * BOX + xo[kk]);
              const uint2 ag8 = *reinterpret_cast<const uint2*>(A1 + k4 * BOX + xo[kk]);
              const uint4 bh = *reinterpret_cast<const uint4*>(B + kb * 64);
              float d[4];
              mma_k32(d, ag, ag8, bh.x, bh.y, bh.z, bh.w);
              const float fa0 = e8((sw0[k4] >> (8 * kk)) & 0xFF), fa1 = e8((sw1[k4] >> (8 * kk)) & 0xFF);
              const float fb0 = e8((sx0 >> (8 * kk)) & 0xFF), fb1 = e8((sx1 >> (8 * kk)) & 0xFF);
              acc[0] = fmaf(d[0], fa0 * fb0, acc[0]);
              acc[1] = fmaf(d[1], fa0 * fb1, acc[1]);
              acc[2] = fmaf(d[2], fa1 * fb0, acc[2]);
              acc[3] = fmaf(d[3], fa1 * fb1, acc[3]);
            }
          }
          if (c0 < n) {
            hp[(size_t)r0 * (2 * I) + hid0] = acc[0];
            hp[(size_t)r0 * (2 * I) + hid1] = acc[2];
          }
          if (c1 < n) {
            hp[(size_t)r1 * (2 * I) + hid0] = acc[1];
            hp[(size_t)r1 * (2 * I) + hid1] = acc[3];
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&h.empty[st]);
        lap(1);
        if (rec) acc_t[6] += 1;
        continue;
      }
      // ---- fc2 unit ----
      if (lane == 0) {
#if MDC_DEBUG
        long long nn = 0;
        while (ld_acquire(p.ready + s) == 0) {
          __nanosleep(32);
          if (++nn == 2000000) printf("mdc hang: blk %d warp %d ready slot %d cnt %d\n", blockIdx.x, warp, s, p.cnt[s]);
        }
#else
        while (ld_acquire(p.ready + s) == 0) __nanosleep(32);
#endif
      }
      __syncwarp();
      lap(4);
      if (rec) acc_t[6] += 1 << 16;
      if (prof && lane == 0) atomicCAS(reinterpret_cast<unsigned long long*>(prof + 11), 0ull, (unsigned long long)gtime());
      const int o0 = perm2(R0), o1 = perm2(R1);
      for (int nt = 0; nt * 8 < n; ++nt) {
        const int rb = h.slot_routes[off + min(nt * 8 + g, n - 1)];
        const int c0 = nt * 8 + 2 * t4, c1 = c0 + 1;
        const int r0 = h.slot_routes[off + min(c0, n - 1)], r1 = h.slot_routes[off + min(c1, n - 1)];
        uint2 bq[KB2];
#pragma unroll
        for (int kb = 0; kb < KB2; ++kb) bq[kb] = __ldcg(reinterpret_cast<const uint2*>(p.actq + (size_t)rb * I + kb * 32 + t4 * 8));
        const uint4 s0 = __ldcg(reinterpret_cast<const uint4*>(p.acts + (size_t)r0 * KB2));
        const uint4 s1 = __ldcg(reinterpret_cast<const uint4*>(p.acts + (size_t)r1 * KB2));
        const uint32_t sx0[4] = {s0.x, s0.y, s0.z, s0.w}, sx1[4] = {s1.x, s1.y, s1.z, s1.w};
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k4 = 0; k4 < 4; ++k4) {
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const int kb = k4 * 4 + kk;
            const uint2 ag = *reinterpret_cast<const uint2*>(A0 + k4 * BOX + xo[kk]);
            const uint2 ag8 = *reinterpret_cast<const uint2*>(A1 + k4 * BOX + xo[kk]);
            float d[4];
            mma_k32(d, ag, ag8, cvt_lo(bq[kb].x), cvt_hi(bq[kb].x), cvt_lo(bq[kb].y), cvt_hi(bq[kb].y));
            const float fa0 = e8((sw0[k4] >> (8 * kk)) & 0xFF), fa1 = e8((sw1[k4] >> (8 * kk)) & 0xFF);
            const float fb0 = e8((sx0[k4] >> (8 * kk)) & 0xFF), fb1 = e8((sx1[k4] >> (8 * kk)) & 0xFF);
            acc[0] = fmaf(d[0], fa0 * fb0, acc[0]);
            acc[1] = fmaf(d[1], fa0 * fb1, acc[1]);
            acc[2] = fmaf(d[2], fa1 * fb0, acc[2]);
            acc[3] = fmaf(d[3], fa1 * fb1, acc[3]);
          }
        }
        if (c0 < n) {
          p.out[(size_t)r0 * H + o0] = __float2bfloat16_rn(acc[0]);
          p.out[(size_t)r0 * H + o1] = __float2bfloat16_rn(acc[2]);
        }
        if (c1 < n) {
          p.out[(size_t)r1 * H + o0] = __float2bfloat16_rn(acc[1]);
          p.out[(size_t)r1 * H + o1] = __float2bfloat16_rn(acc[3]);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&h.empty[st]);
      lap(5);
    }
    if (team < C && !closed) close_fc1();
    if (rec && team < C) {
#pragma unroll
      for (int k = 0; k < 8; ++k) prof[16 + team * 8 + k] = acc_t[k];
    }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(p.done, 1) == (int)gridDim.x - 1) {
      for (int s = 0; s < D; ++s) { p.cnt[s] = 0; p.ready[s] = 0; }
      *p.done = 0;
      __threadfence();
    }
  }
  __syncthreads();
  if (prof && tid == 0) prof[15] = gtime();
  asm volatile("griddepcontrol.launch_dependents;");
}

int smem_bytes(int T, int ns) {
  return ((SMALL + ((T * 64 + 127) & ~127) + T * XROW + 1023) & ~1023) + ns * SLOT;
}

static CUtensorMap make_map(const void* base, uint64_t rows, uint64_t rowbytes) {
  static PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
  if (!encode) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q));
    TORCH_CHECK(fn != nullptr && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }
  CUtensorMap m;
  cuuint64_t dims[2] = {rowbytes, rows};
  cuuint64_t strides[1] = {rowbytes};
  cuuint32_t box[2] = {128, 64};
  cuuint32_t estr[2] = {1, 1};
  const CUresult r = encode(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

static const CUtensorMap& cached_map(const void* base, uint64_t rows, uint64_t rowbytes) {
  static std::unordered_map<const void*, CUtensorMap> maps;   // weights never move after loading
  auto it = maps.find(base);
  if (it == maps.end()) it = maps.emplace(base, make_map(base, rows, rowbytes)).first;
  return it->second;
}

void run(at::Tensor logits, at::Tensor x, at::Tensor xs, at::Tensor w13, at::Tensor s13, at::Tensor w2,
         at::Tensor s2, at::Tensor perm13, at::Tensor perm2, at::Tensor out, at::Tensor topk_w,
         c10::optional<at::Tensor> topk_ids, at::Tensor hpart, at::Tensor actq, at::Tensor acts, at::Tensor sync,
         int64_t grid, int64_t ns, int64_t qmode, bool pdl,
         c10::optional<at::Tensor> prof) {
  const int T = (int)x.size(0);
  TORCH_CHECK(T >= 1 && T <= MAXT && ns >= 1 && ns <= 16 && ns % std::min<int64_t>(NTEAM, ns) == 0);
  TORCH_CHECK(x.size(1) == H && xs.numel() >= T * KB1 && logits.size(1) == E);
  TORCH_CHECK(x.is_contiguous() && xs.is_contiguous() && logits.is_contiguous());
  TORCH_CHECK(w13.numel() == (int64_t)E * 2 * I * H && w2.numel() == (int64_t)E * H * I);
  TORCH_CHECK(s13.numel() == (int64_t)E * 2 * I * KB1 && s2.numel() == (int64_t)E * H * KB2);
  TORCH_CHECK(hpart.numel() >= (int64_t)KC * MAXR * 2 * I);
  static_assert(sizeof(Hdr) <= SMALL, "hdr");
  Params p;
  p.tm13 = cached_map(w13.data_ptr(), (uint64_t)E * 2 * I, H);
  p.tm2 = cached_map(w2.data_ptr(), (uint64_t)E * H, I);
  p.logits = logits.data_ptr();
  p.x = (const uint8_t*)x.data_ptr(); p.xs = (const uint8_t*)xs.data_ptr();
  p.w13 = (const uint8_t*)w13.data_ptr(); p.s13 = (const uint8_t*)s13.data_ptr();
  p.w2 = (const uint8_t*)w2.data_ptr(); p.s2 = (const uint8_t*)s2.data_ptr();
  p.perm13 = (const int16_t*)perm13.data_ptr(); p.perm2 = (const int16_t*)perm2.data_ptr();
  p.out = (__nv_bfloat16*)out.data_ptr(); p.topk_w = (__nv_bfloat16*)topk_w.data_ptr();
  p.topk_ids = topk_ids.has_value() ? (int32_t*)topk_ids->data_ptr() : nullptr;
  p.hpart = (float*)hpart.data_ptr(); p.actq = (uint8_t*)actq.data_ptr(); p.acts = (uint8_t*)acts.data_ptr();
  int* sy = (int*)sync.data_ptr();
  p.cnt = sy; p.ready = sy + MAXR; p.done = sy + 2 * MAXR;
  p.prof = prof.has_value() ? (long long*)prof->data_ptr() : nullptr;
  p.T = T; p.ns = (int)ns; p.qmode = (int)qmode;
  const int smem = smem_bytes(T, (int)ns);
  const bool f32 = logits.scalar_type() == at::kFloat;
  auto kern = f32 ? moe_decode_kernel<true> : moe_decode_kernel<false>;
  static bool attr_set[2] = {false, false};
  if (!attr_set[f32]) {
    int dev, mx;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, mx));
    attr_set[f32] = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid);
  cfg.blockDim = dim3(NTHR);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, p));
}
}  // namespace mdc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &mdc::run);
  m.def("smem_bytes", &mdc::smem_bytes);
}
"""

E, TOPK, H, INTER = 256, 8, 2048, 512
MAXT = 32
_NTEAM = 3
_SMALL, _XROW, _SLOT = 5120, 2 * H + 64, 64 * INTER + 2048


def tuned_source(tune: str) -> str:
    defines = ""
    for item in filter(None, (x.strip() for x in tune.replace(",", "+").split("+"))):
        key, val = item.split("=")
        assert key in ("FAST_EXP", "DEBUG"), key
        defines += f"#define MDC_{key} {int(val)}\n"
    return defines + _SOURCE


def build_dir(arch: str) -> str:
    root = os.environ.get("VLLM_CACHE_ROOT") or os.path.expanduser("~/.cache/vllm")
    return os.path.join(root, "moe_decode_cuda", f"sm{arch}")


def load():
    if _ext:
        return _ext[0]
    major, minor = torch.cuda.get_device_capability()
    import torch.utils.cpp_extension as cpp

    arch = f"{major}{minor}{'a' if major >= 9 else ''}"
    bdir = build_dir(arch)
    os.makedirs(bdir, exist_ok=True)
    source = tuned_source(TUNE)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    src = os.path.join(bdir, f"moe_decode_cuda_{digest}.cu")
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
        ext = cpp.load(
            name=f"_moe_decode_cuda_{digest}",
            sources=[src],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"],
            extra_cflags=["-O3", "-std=c++20"],
            build_directory=bdir,
            verbose=False,
        )
    finally:
        cpp._get_cuda_arch_flags = orig
    _ext.append(ext)
    return ext


def permutations(device) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffled row -> logical row of the trtllm-gen MXFP8 MoE weights (int16).

    perm13[n]: index into the fp32 [gate | up] (2 x 512) FC1 output of shuffled w13 row n;
    perm2[n]: hidden index of shuffled w2 row n. Traced through vLLM's own
    preparation (``swap_w13_to_w31`` + ``_shuffle_mxfp8_moe_weights``) on index rows.
    """
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        _shuffle_mxfp8_moe_weights,
        swap_w13_to_w31,
    )

    # Encode the row index in the first two bytes of each row (K = 128: 4 scale blocks, the
    # minimum the 128x4 scale interleave keeps unpadded).
    def rows(m):
        r = torch.arange(m, dtype=torch.int32)
        t = torch.zeros(1, m, 128, dtype=torch.uint8)
        t[0, :, 0] = (r & 0xFF).to(torch.uint8)
        t[0, :, 1] = (r >> 8).to(torch.uint8)
        return t.to(device)

    w13 = rows(2 * INTER)
    w2 = rows(H)
    s13 = torch.zeros(1, 2 * INTER, 4, dtype=torch.uint8, device=device)
    s2 = torch.zeros(1, H, 4, dtype=torch.uint8, device=device)
    sw13, sw2, _, _ = _shuffle_mxfp8_moe_weights(
        swap_w13_to_w31(w13).view(torch.float8_e4m3fn),
        w2.view(torch.float8_e4m3fn),
        swap_w13_to_w31(s13),
        s2,
        True,
    )

    def decode(t):
        u = t.view(torch.uint8)[0].to(torch.int32)
        return (u[:, 0] | (u[:, 1] << 8)).to(torch.int16)

    return decode(sw13).contiguous(), decode(sw2).contiguous()


class MoeDecode:
    """Workspace + launcher for one device (shared by all MoE layers; calls are stream-ordered)."""

    def __init__(self, device, num_sms: int | None = None):
        self.ext = load()
        self.device = torch.device(device)
        self.perm13, self.perm2 = permutations(self.device)
        # The kernel uses the closed form of these permutations; refuse a layout it does not match.
        n = torch.arange(H, device=self.device)
        old = (n // 32) * 32 + (n % 8) * 4 + (n % 32) // 8
        cf13 = torch.where(
            old[: 2 * INTER] % 2 == 0,
            INTER + old[: 2 * INTER] // 2,
            old[: 2 * INTER] // 2,
        )
        if not (
            torch.equal(cf13, self.perm13.long())
            and torch.equal(old, self.perm2.long())
        ):
            raise RuntimeError(
                "trtllm-gen MXFP8 weight shuffle changed; moe_decode_cuda needs updating"
            )
        props = torch.cuda.get_device_properties(self.device)
        self.grid = GRID or (num_sms or props.multi_processor_count)
        self.prof: torch.Tensor | None = (
            None  # [grid, 16] int64 phase stamps (microbench)
        )
        self.smem_max = props.shared_memory_per_block_optin
        maxr = MAXT * TOPK
        # FC1 partial sums per 512-wide K chunk (reduced in fixed order by the SwiGLU step).
        self.hpart = torch.empty(
            H // INTER, maxr, 2 * INTER, dtype=torch.float32, device=self.device
        )
        self.actq = torch.empty(maxr, INTER, dtype=torch.uint8, device=self.device)
        self.acts = torch.empty(
            maxr, INTER // 32, dtype=torch.uint8, device=self.device
        )
        self.sync = torch.zeros(2 * maxr + 1, dtype=torch.int32, device=self.device)
        self._idx: dict[int, torch.Tensor] = {}

    def slots(self, T: int) -> int:
        fixed = (_SMALL + ((T * 64 + 127) & ~127) + T * _XROW + 1023) & ~1023
        ns = min(16, (self.smem_max - fixed) // _SLOT)
        if NSLOTS:
            ns = min(ns, NSLOTS)
        ns -= ns % min(_NTEAM, ns)  # whole slots per consumer team
        assert ns >= 1, (T, self.smem_max)
        return ns

    def idx(self, T: int) -> torch.Tensor:
        t = self._idx.get(T)
        if t is None:
            t = torch.arange(T * TOPK, dtype=torch.int32, device=self.device).view(
                T, TOPK
            )
            self._idx[T] = t
        return t

    def __call__(
        self, logits, x, xs, w13, s13, w2, s2, *, pdl: bool = True, topk_ids=None
    ):
        """Returns (gemm2_rows [T*TOPK, H] bf16, topk_w [T, TOPK] bf16, idx [T, TOPK] int32)."""
        T = x.shape[0]
        out = torch.empty(T * TOPK, H, dtype=torch.bfloat16, device=x.device)
        topk_w = torch.empty(T, TOPK, dtype=torch.bfloat16, device=x.device)
        self.ext.run(
            logits,
            x.view(torch.uint8),
            xs.view(torch.uint8),
            w13.view(torch.uint8),
            s13.view(torch.uint8),
            w2.view(torch.uint8),
            s2.view(torch.uint8),
            self.perm13,
            self.perm2,
            out,
            topk_w,
            topk_ids,
            self.hpart,
            self.actq,
            self.acts,
            self.sync,
            self.grid,
            self.slots(T),
            QMODE,
            pdl,
            self.prof,
        )
        return out, topk_w, self.idx(T)


def get(device_index: int) -> "MoeDecode | None":
    """The per-device instance (built once per process); None if the build failed."""
    if device_index not in _instances:
        from vllm.logger import init_logger

        logger = init_logger(__name__)
        try:
            _instances[device_index] = MoeDecode(torch.device("cuda", device_index))
            inst = _instances[device_index]
            logger.info(
                "MoE decode: single-launch kernel for T <= %d (grid %d, qmode %d, tune '%s').",
                min(MAX_TOKENS, MAXT),
                inst.grid,
                QMODE,
                TUNE,
            )
        except Exception:
            _instances[device_index] = None
            logger.warning(
                "MoE decode: kernel build failed; using the trtllm-gen chain.",
                exc_info=True,
            )
    return _instances[device_index]
