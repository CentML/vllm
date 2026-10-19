// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3-Flash mHC sublayer boundary for large token counts (prefill), one
// launch: previous hc_post + next hc_pre (prenorm GEMM, Sinkhorn, collapse)
// + the fused input RMSNorm. hc_mult 4, hidden 4096, 24 mix outputs.
//
// vLLM's TileLang path runs three kernels above 32 tokens (post, DeepGEMM
// TF32 prenorm GEMM, pre_big_fuse), which move 144 KB per token: post reads
// the residual streams r (32 KB) and the layer output x (8 KB) and writes the
// new streams r' (32 KB); the GEMM and the pre kernel each read r' again.
// This kernel reads r and x once and writes r' once (72 KB per token); the
// collapse re-reads r' from L2 right after it was written.
//
// Grid (splits, token tiles of BM). CTA (s, t):
//   1. streams r / x for its hidden slice [s * 4096 / S, +4096 / S) in
//      32-column sub-chunks through a cp.async ring; fn either through the
//      same ring or (fn_reg) straight from L1/L2 into the B fragments, which
//      leaves room for a deeper ring (2 CTAs per SM either way);
//   2. post: r'_o = post_o * x + sum_i comb[i][o] * r_i (fp32, same order as
//      the TileLang post kernel), rounded to bf16, stored to global and into
//      a shared-memory A tile;
//   3. prenorm GEMM partials over its k slice with mma.sync m16n8k16 BF16:
//      A = bf16 r' (exact), B = fn split into bf16 hi + lo. With fn_tf32 the
//      fp32 weights are first truncated to TF32, and hi + lo is then exactly
//      the TF32 value, so every product equals the one DeepGEMM's TF32 UMMA
//      forms; only the fp32 accumulation order differs. Without it, hi + lo
//      carries 16 mantissa bits (closer to fp32 than TF32);
//   4. writes fp32 partials [S, M, 24] and sqrsum [S, M] of bf16 r';
//   5. (fused mode) the last CTA of the tile to arrive (atomic counter,
//      zeroed by the caller per launch, so concurrent streams never share
//      one) sums
//      the S partials in split order, runs sigmoid / softmax / Sinkhorn with
//      the TileLang pre kernel's exact operation order (expf, IEEE division,
//      rsqrtf, same butterfly shapes; bit-identical post / comb on the same
//      partials), then collapses r' (from L2) and applies the RMSNorm, like
//      mhc_pre_big_fuse_with_norm (layer_input within 1 bf16 ulp: the sum of
//      squares is reduced in a different order).
// r' is bit-identical to the TileLang post kernel (same FMA chain).
// Mode 1 stops after 4 (partials for a separate pre kernel), mode 2 runs only
// step 5 as a standalone kernel from partials, mode 3 = mode 1 + mode 2.
//
// Built without -use_fast_math (see mhc_prefill.py) so expf / division match
// the TileLang kernels, which are compiled without fast math.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace glm5next_mhcp {

constexpr int kHC = 4;
constexpr int kMix = kHC * (kHC + 2);  // 24
constexpr int kHidden = 4096;
constexpr int kK = kHC * kHidden;  // 16384
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kHc = 32;          // hidden columns per sub-chunk
constexpr int kKc = kHC * kHc;   // GEMM k per sub-chunk (128)
constexpr int kFnRow = kKc + 8;  // fp32 per fn smem row (conflict-free B)
constexpr int kARow = kKc + 8;   // bf16 per A smem row (conflict-free ldsm)
static_assert(kKc == 16 * kWarps, "one k16 block per warp per sub-chunk");

struct Params {
  const __nv_bfloat16* x;         // [M, 4096]
  const __nv_bfloat16* residual;  // [M, 4, 4096]
  const float* post_in;           // [M, 4]
  const float* comb_in;           // [M, 4, 4]  (comb[i][o]: old i -> new o)
  const float* fn;                // [24, 16384]
  const float* hc_scale;          // [3]
  const float* hc_base;           // [24]
  const __nv_bfloat16* norm_w;    // [4096]
  __nv_bfloat16* residual_out;    // [M, 4, 4096]
  float* post_out;                // [M, 4]
  float* comb_out;                // [M, 16]
  __nv_bfloat16* layer_input;     // [M, 4096]
  float* part;                    // [S, M, 24]
  float* sq;                      // [S, M]
  int* counters;                  // [tiles], zero on entry (fused mode)
  int M;
  int sk_iters;
  int fn_tf32;
  int flags;  // diagnostics (wrong output): bit 0 skip MMAs, bit 1 skip fn
              // loads
  float rms_eps, pre_eps, sk_eps, post_mult, norm_eps;
};

// FNREG: fn B fragments come straight from global (L1/L2) into registers
// instead of a shared-memory stage, which leaves room for a deeper r / x ring.
template <int BM, int STAGES, bool FNREG>
struct Layout {
  static constexpr int kRBytes = BM * kHC * kHc * 2;
  static constexpr int kXBytes = BM * kHc * 2;
  static constexpr int kFnBytes = FNREG ? 0 : kMix * kFnRow * 4;
  static constexpr int kStageBytes = kRBytes + kXBytes + kFnBytes;
  static constexpr int kRedBytes = kWarps * BM * kMix * 4;
  static constexpr int kOlBytes = kWarps * kHidden * 2;  // pre: one row/warp
  static constexpr int kUnion0 = STAGES * kStageBytes;
  static constexpr int kUnion1 = kRedBytes > kOlBytes ? kRedBytes : kOlBytes;
  static constexpr int kUnion = kUnion0 > kUnion1 ? kUnion0 : kUnion1;
  static constexpr int kAOff = kUnion;
  static constexpr int kABytes = BM * kARow * 2;
  static constexpr int kPreOff = kAOff + kABytes;  // float [BM][4]
  static constexpr int kFlagOff = kPreOff + BM * kHC * 4;
  static constexpr int kTotal = kFlagOff + 16;
  static_assert(kRBytes % 16 == 0 && kXBytes % 16 == 0 && kFnBytes % 16 == 0);
};

// ------------------------------------------------------------------ PTX
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src,
                                           bool valid) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)),
      "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo,
                                              __nv_bfloat16 hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(lo)) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}
// fp32 -> (hi, lo) bf16 with hi + lo == v (tf32: == TF32-truncated v).
__device__ __forceinline__ void split_bf16(float v, bool tf32,
                                           __nv_bfloat16& hi,
                                           __nv_bfloat16& lo) {
  if (tf32) v = __uint_as_float(__float_as_uint(v) & 0xffffe000u);
  hi = __float2bfloat16_rn(v);
  lo = __float2bfloat16_rn(v - __bfloat162float(hi));
}
__device__ __forceinline__ float bf(uint32_t packed, int half) {
  return __uint_as_float(half ? (packed & 0xffff0000u) : (packed << 16));
}

// ---------------------------------------------- post + prenorm GEMM partials
template <int BM, int S, int STAGES, bool FNREG>
__device__ __forceinline__ void post_gemm(const Params& p, uint8_t* smem,
                                          int split, int tile) {
  using L = Layout<BM, STAGES, FNREG>;
  constexpr int kHs = kHidden / S;           // hidden columns of this split
  constexpr int kNsc = kHs / kHc;            // sub-chunks
  constexpr int kVec = BM * kHc / kThreads;  // h per thread per sub-chunk
  constexpr int kTpt = kHc / kVec;           // threads per token
  constexpr int kMT = BM / 16;
  static_assert(kVec == 4 || kVec == 8, "BM must be 32 or 64");
  static_assert(kHs % kHc == 0 && kNsc >= 1, "split too fine");

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int m0 = tile * BM;
  const int M = p.M;
  const int h_split = split * kHs;

  __nv_bfloat16* a_s = reinterpret_cast<__nv_bfloat16*>(smem + L::kAOff);

  auto stage_r = [&](int st) { return smem + st * L::kStageBytes; };
  auto stage_x = [&](int st) {
    return smem + st * L::kStageBytes + L::kRBytes;
  };
  auto stage_fn = [&](int st) {
    return reinterpret_cast<float*>(smem + st * L::kStageBytes + L::kRBytes +
                                    L::kXBytes);
  };

  auto load_stage = [&](int st, int sc) {
    const int h0 = h_split + sc * kHc;
    // r: BM tokens x 4 streams x 64 B; 16 chunks of 16 B per token, the
    // chunk index XOR-swizzled by token parity (conflict-free reads).
    uint8_t* rs = stage_r(st);
#pragma unroll
    for (int i = 0; i < BM * 16 / kThreads; ++i) {
      const int q = tid + i * kThreads;
      const int tok = q >> 4, c = q & 15;
      const int j = c >> 2, cc = c & 3;
      const bool ok = m0 + tok < M;
      const __nv_bfloat16* src =
          p.residual + (ok ? (static_cast<int64_t>(m0 + tok) * kK +
                              j * kHidden + h0 + cc * 8)
                           : 0);
      cp_async16(rs + tok * 256 + ((c ^ ((tok & 1) << 2)) << 4), src, ok);
    }
    uint8_t* xs = stage_x(st);
    for (int q = tid; q < BM * 4; q += kThreads) {
      const int tok = q >> 2, c = q & 3;
      const bool ok = m0 + tok < M;
      const __nv_bfloat16* src =
          p.x +
          (ok ? (static_cast<int64_t>(m0 + tok) * kHidden + h0 + c * 8) : 0);
      cp_async16(xs + tok * 64 + c * 16, src, ok);
    }
    if (FNREG || (p.flags & 2)) return;
    float* fs = stage_fn(st);
#pragma unroll
    for (int i = 0; i < kMix * kHC * 8 / kThreads; ++i) {
      const int q = tid + i * kThreads;
      const int n = q >> 5, rem = q & 31, j = rem >> 3, c = rem & 7;
      cp_async16(fs + n * kFnRow + j * kHc + c * 4,
                 p.fn + static_cast<int64_t>(n) * kK + j * kHidden + h0 + c * 4,
                 true);
    }
  };

  // Per-thread token for the post phase and its mixing coefficients.
  const int ptok = tid / kTpt, pv = tid % kTpt;
  const bool pvalid = m0 + ptok < M;
  float cmb[kHC][kHC], pst[kHC];
#pragma unroll
  for (int i = 0; i < kHC; ++i) {
    pst[i] =
        pvalid ? p.post_in[static_cast<int64_t>(m0 + ptok) * kHC + i] : 0.f;
#pragma unroll
    for (int o = 0; o < kHC; ++o)
      cmb[i][o] =
          pvalid ? p.comb_in[static_cast<int64_t>(m0 + ptok) * 16 + i * kHC + o]
                 : 0.f;
  }
  float sqr = 0.f;

  float acc[kMT][3][4];
#pragma unroll
  for (int a = 0; a < kMT; ++a)
#pragma unroll
    for (int b = 0; b < 3; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

  const bool tf32 = p.fn_tf32 != 0;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < kNsc) load_stage(s, s);
    cp_async_commit();
  }

  for (int sc = 0; sc < kNsc; ++sc) {
    const int st = sc % STAGES;
    cp_async_wait<STAGES - 2>();
    __syncthreads();  // stage sc visible; previous mma done with A / buffer
    {
      const int nx = sc + STAGES - 1;
      if (nx < kNsc) load_stage(nx % STAGES, nx);
      cp_async_commit();
    }
    // FNREG: this warp's fn fragments for sub-chunk sc, loaded before the
    // post phase so the L2 latency overlaps it.
    float2 fg[3][2];
    if (FNREG && !(p.flags & 2)) {
      const int g = lane >> 2, t = lane & 3, kb = warp * 16;
      const float* base =
          p.fn + (kb / kHc) * kHidden + h_split + sc * kHc + (kb % kHc) + 2 * t;
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) {
        const float* row = base + static_cast<int64_t>(nt * 8 + g) * kK;
        fg[nt][0] = __ldg(reinterpret_cast<const float2*>(row));
        fg[nt][1] = __ldg(reinterpret_cast<const float2*>(row + 8));
      }
    }

    // ---- post: r'_o = post_o * x + sum_i comb[i][o] * r_i
    {
      const int h0 = h_split + sc * kHc;
      const uint8_t* rs = stage_r(st) + ptok * 256;
      const uint8_t* xs = stage_x(st) + ptok * 64;
      float rv[kHC][kVec], xv[kVec];
#pragma unroll
      for (int j = 0; j < kHC; ++j) {
        const int boff = j * 64 + pv * kVec * 2;  // byte offset unswizzled
        const int c = boff >> 4;
        const uint8_t* ptr = rs + ((c ^ ((ptok & 1) << 2)) << 4) + (boff & 15);
        if constexpr (kVec == 8) {
          const uint4 u = *reinterpret_cast<const uint4*>(ptr);
          const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int e = 0; e < 8; ++e) rv[j][e] = bf(w[e >> 1], e & 1);
        } else {
          const uint2 u = *reinterpret_cast<const uint2*>(ptr);
          const uint32_t w[2] = {u.x, u.y};
#pragma unroll
          for (int e = 0; e < 4; ++e) rv[j][e] = bf(w[e >> 1], e & 1);
        }
      }
      if constexpr (kVec == 8) {
        const uint4 u = *reinterpret_cast<const uint4*>(xs + pv * 16);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) xv[e] = bf(w[e >> 1], e & 1);
      } else {
        const uint2 u = *reinterpret_cast<const uint2*>(xs + pv * 8);
        const uint32_t w[2] = {u.x, u.y};
#pragma unroll
        for (int e = 0; e < 4; ++e) xv[e] = bf(w[e >> 1], e & 1);
      }
#pragma unroll
      for (int o = 0; o < kHC; ++o) {
        uint32_t packed[kVec / 2];
#pragma unroll
        for (int e = 0; e < kVec; e += 2) {
          __nv_bfloat16 b2[2];
#pragma unroll
          for (int d = 0; d < 2; ++d) {
            float v = __fmaf_rn(pst[o], xv[e + d],
                                __fmul_rn(cmb[0][o], rv[0][e + d]));
#pragma unroll
            for (int i = 1; i < kHC; ++i)
              v = __fmaf_rn(cmb[i][o], rv[i][e + d], v);
            b2[d] = __float2bfloat16_rn(v);
            const float vr = __bfloat162float(b2[d]);
            sqr = __fmaf_rn(vr, vr, sqr);
          }
          packed[e >> 1] = pack_bf16(b2[0], b2[1]);
        }
        __nv_bfloat16* arow = a_s + ptok * kARow + o * kHc + pv * kVec;
        __nv_bfloat16* grow = p.residual_out +
                              static_cast<int64_t>(m0 + ptok) * kK +
                              o * kHidden + h0 + pv * kVec;
        if constexpr (kVec == 8) {
          const uint4 u =
              make_uint4(packed[0], packed[1], packed[2], packed[3]);
          *reinterpret_cast<uint4*>(arow) = u;
          if (pvalid) *reinterpret_cast<uint4*>(grow) = u;
        } else {
          const uint2 u = make_uint2(packed[0], packed[1]);
          *reinterpret_cast<uint2*>(arow) = u;
          if (pvalid) *reinterpret_cast<uint2*>(grow) = u;
        }
      }
    }
    __syncthreads();  // A tile complete

    // ---- GEMM partials: warp owns k16 block `warp` of this sub-chunk.
    if (!(p.flags & 1)) {
      const int g = lane >> 2, t = lane & 3;
      const int kb = warp * 16;
      uint32_t bh[3][2], bl[3][2];
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) {
        float2 f01, f89;
        if constexpr (FNREG) {
          f01 = fg[nt][0];
          f89 = fg[nt][1];
        } else {
          const float* row = stage_fn(st) + (nt * 8 + g) * kFnRow + kb + 2 * t;
          f01 = *reinterpret_cast<const float2*>(row);
          f89 = *reinterpret_cast<const float2*>(row + 8);
        }
        __nv_bfloat16 h0_, l0_, h1_, l1_;
        split_bf16(f01.x, tf32, h0_, l0_);
        split_bf16(f01.y, tf32, h1_, l1_);
        bh[nt][0] = pack_bf16(h0_, h1_);
        bl[nt][0] = pack_bf16(l0_, l1_);
        split_bf16(f89.x, tf32, h0_, l0_);
        split_bf16(f89.y, tf32, h1_, l1_);
        bh[nt][1] = pack_bf16(h0_, h1_);
        bl[nt][1] = pack_bf16(l0_, l1_);
      }
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt) {
        uint32_t a[4];
        ldsm_x4(a,
                a_s + (mt * 16 + (lane & 15)) * kARow + kb + (lane >> 4) * 8);
#pragma unroll
        for (int nt = 0; nt < 3; ++nt) {
          mma_bf16(acc[mt][nt], a, bl[nt][0], bl[nt][1]);
          mma_bf16(acc[mt][nt], a, bh[nt][0], bh[nt][1]);
        }
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();  // all warps done with stages and A

  // ---- reduce the 8 warp partials (fixed order) and write [S, M, 24]
  float* red = reinterpret_cast<float*>(smem);
  {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) {
        float* r0 = red + (warp * BM + mt * 16 + g) * kMix + nt * 8 + 2 * t;
        *reinterpret_cast<float2*>(r0) =
            make_float2(acc[mt][nt][0], acc[mt][nt][1]);
        *reinterpret_cast<float2*>(r0 + 8 * kMix) =
            make_float2(acc[mt][nt][2], acc[mt][nt][3]);
      }
  }
#pragma unroll
  for (int off = 1; off < kTpt; off <<= 1)
    sqr += __shfl_xor_sync(0xffffffffu, sqr, off);
  if (pv == 0 && pvalid)
    p.sq[static_cast<int64_t>(split) * M + m0 + ptok] = sqr;
  __syncthreads();
  for (int idx = tid; idx < BM * kMix; idx += kThreads) {
    const int tok = idx / kMix;
    if (m0 + tok >= M) continue;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) v += red[w * BM * kMix + idx];
    p.part[(static_cast<int64_t>(split) * M + m0) * kMix + idx] = v;
  }
}

// ---------------------------------------------- pre (mix, Sinkhorn, collapse)
// kCg: read partials / r' with ld.global.cg (written by other CTAs of this
// grid in fused mode).
template <int BM, int S, int STAGES, bool FNREG, bool kCg>
__device__ __forceinline__ void pre_tile(const Params& p, uint8_t* smem,
                                         int tile) {
  using L = Layout<BM, STAGES, FNREG>;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int m0 = tile * BM;
  const int M = p.M;
  float* pre_s = reinterpret_cast<float*>(smem + L::kPreOff);
  // Split sums staged in the (now free) union area: [BM][24] then [BM].
  float* mix_s = reinterpret_cast<float*>(smem);
  float* sqs_s = mix_s + BM * kMix;

  auto ldf = [&](const float* ptr) { return kCg ? __ldcg(ptr) : __ldg(ptr); };

  // ---- 0. sum the S split partials in split order (as TileLang does).
  for (int idx = tid; idx < BM * kMix; idx += kThreads) {
    float v = 0.f;
    if (m0 + idx / kMix < M) {
      const float* src = p.part + static_cast<int64_t>(m0) * kMix + idx;
#pragma unroll
      for (int s = 0; s < S; ++s)
        v = v + ldf(src + static_cast<int64_t>(s) * M * kMix);
    }
    mix_s[idx] = v;
  }
  if (tid < BM) {
    float v = 0.f;
    if (m0 + tid < M) {
#pragma unroll
      for (int s = 0; s < S; ++s)
        v = v + ldf(p.sq + static_cast<int64_t>(s) * M + m0 + tid);
    }
    sqs_s[tid] = v;
  }
  __syncthreads();

  // ---- 1. per token: 4 lanes, lane q owns stream / comb row q.
  if (tid < BM * 4) {
    const int tok = tid >> 2, q = tid & 3;
    const bool valid = m0 + tok < M;
    const int64_t m = m0 + tok;
    const float rms = rsqrtf(sqs_s[tok] / 16384.0f + p.rms_eps);
    float mix[6];  // pre_q, post_q, comb[q][0..3]
    const int idx[6] = {q,         kHC + q,    8 + 4 * q,
                        9 + 4 * q, 10 + 4 * q, 11 + 4 * q};
#pragma unroll
    for (int e = 0; e < 6; ++e) mix[e] = mix_s[tok * kMix + idx[e]] * rms;
    const float pre =
        (1.0f /
         (1.0f + expf(0.0f - ((mix[0] * p.hc_scale[0]) + p.hc_base[q])))) +
        p.pre_eps;
    const float post = (1.0f / (1.0f + expf(0.0f - ((mix[1] * p.hc_scale[1]) +
                                                    p.hc_base[kHC + q])))) *
                       p.post_mult;
    float cm[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
      cm[k] = (mix[2 + k] * p.hc_scale[2]) + p.hc_base[8 + 4 * q + k];
    // Same reduction shapes as the TileLang kernel (lane (j, k) butterflies:
    // rows xor 2 then 1 over k, columns xor 8 then 4 over j).
    auto row_sum = [&]() { return (cm[0] + cm[2]) + (cm[1] + cm[3]); };
    auto col_sum = [&](int k) {
      float x = cm[k] + __shfl_xor_sync(0xffffffffu, cm[k], 2);
      return x + __shfl_xor_sync(0xffffffffu, x, 1);
    };
    const float rmax = fmaxf(fmaxf(cm[0], cm[2]), fmaxf(cm[1], cm[3]));
#pragma unroll
    for (int k = 0; k < 4; ++k) cm[k] = expf(cm[k] - rmax);
    {
      const float rs = row_sum();
#pragma unroll
      for (int k = 0; k < 4; ++k) cm[k] = (cm[k] / rs) + p.sk_eps;
    }
    {
      float cs[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) cs[k] = col_sum(k);
#pragma unroll
      for (int k = 0; k < 4; ++k) cm[k] = cm[k] / (cs[k] + p.sk_eps);
    }
    // A NaN row (padding) is NaN either way; skip the remaining iterations
    // when the whole warp is NaN (IEEE division is slow on NaN), as the
    // TileLang kernel does per token. Warp-uniform for the shuffles.
    if (__any_sync(0xffffffffu, isfinite(rms))) {
      for (int it = 0; it < p.sk_iters - 1; ++it) {
        const float rs = row_sum();
#pragma unroll
        for (int k = 0; k < 4; ++k) cm[k] = cm[k] / (rs + p.sk_eps);
        float cs[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) cs[k] = col_sum(k);
#pragma unroll
        for (int k = 0; k < 4; ++k) cm[k] = cm[k] / (cs[k] + p.sk_eps);
      }
    }
    if (valid) {
      *reinterpret_cast<float4*>(p.comb_out + m * 16 + q * 4) =
          make_float4(cm[0], cm[1], cm[2], cm[3]);
      p.post_out[m * kHC + q] = post;
    }
    pre_s[tok * kHC + q] = pre;
  }
  __syncthreads();

  // ---- 2. collapse + RMSNorm: one warp per token; each lane owns 16 x 8 h.
  __nv_bfloat16* ol = reinterpret_cast<__nv_bfloat16*>(smem) + warp * kHidden;
  for (int tok = warp; tok < BM; tok += kWarps) {
    if (m0 + tok >= M) break;
    const int64_t m = m0 + tok;
    const float p0 = pre_s[tok * kHC + 0], p1 = pre_s[tok * kHC + 1],
                p2 = pre_s[tok * kHC + 2], p3 = pre_s[tok * kHC + 3];
    const uint4* rrow = reinterpret_cast<const uint4*>(p.residual_out + m * kK);
    float ss = 0.f;
#pragma unroll 4
    for (int it = 0; it < kHidden / 256; ++it) {
      const int v = it * 32 + lane;  // 16 B vector index within a stream row
      uint4 r[kHC];
#pragma unroll
      for (int j = 0; j < kHC; ++j)
        r[j] = kCg ? __ldcg(rrow + j * (kHidden / 8) + v)
                   : __ldg(rrow + j * (kHidden / 8) + v);
      const uint32_t w0[4] = {r[0].x, r[0].y, r[0].z, r[0].w};
      const uint32_t w1[4] = {r[1].x, r[1].y, r[1].z, r[1].w};
      const uint32_t w2[4] = {r[2].x, r[2].y, r[2].z, r[2].w};
      const uint32_t w3[4] = {r[3].x, r[3].y, r[3].z, r[3].w};
      uint32_t out[4];
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        __nv_bfloat16 b2[2];
#pragma unroll
        for (int d = 0; d < 2; ++d) {
          const int h = (e + d) >> 1, hf = (e + d) & 1;
          float o = __fmul_rn(p0, bf(w0[h], hf));
          o = __fmaf_rn(p1, bf(w1[h], hf), o);
          o = __fmaf_rn(p2, bf(w2[h], hf), o);
          o = __fmaf_rn(p3, bf(w3[h], hf), o);
          ss = __fmaf_rn(o, o, ss);
          b2[d] = __float2bfloat16_rn(o);
        }
        out[e >> 1] = pack_bf16(b2[0], b2[1]);
      }
      reinterpret_cast<uint4*>(ol)[v] =
          make_uint4(out[0], out[1], out[2], out[3]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float rs = rsqrtf(ss / 4096.0f + p.norm_eps);
    const uint4* wrow = reinterpret_cast<const uint4*>(p.norm_w);
    uint4* orow = reinterpret_cast<uint4*>(p.layer_input + m * kHidden);
#pragma unroll 4
    for (int it = 0; it < kHidden / 256; ++it) {
      const int v = it * 32 + lane;  // only this lane wrote ol[v]
      const uint4 o4 = reinterpret_cast<const uint4*>(ol)[v];
      const uint4 w4 = __ldg(wrow + v);
      const uint32_t ow[4] = {o4.x, o4.y, o4.z, o4.w};
      const uint32_t ww[4] = {w4.x, w4.y, w4.z, w4.w};
      uint32_t res[4];
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const float y0 = __fmul_rn(__fmul_rn(bf(ow[h], 0), rs), bf(ww[h], 0));
        const float y1 = __fmul_rn(__fmul_rn(bf(ow[h], 1), rs), bf(ww[h], 1));
        res[h] = pack_bf16(__float2bfloat16_rn(y0), __float2bfloat16_rn(y1));
      }
      orow[v] = make_uint4(res[0], res[1], res[2], res[3]);
    }
    __syncwarp();
  }
}

// mode 0: fused (last CTA per tile runs pre); 1: partials only.
template <int BM, int S, int STAGES, bool FNREG, int MODE>
__global__ void __launch_bounds__(kThreads, 2) mhc_post_gemm_kernel(Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  using L = Layout<BM, STAGES, FNREG>;
  const int split = blockIdx.x, tile = blockIdx.y;
  post_gemm<BM, S, STAGES, FNREG>(p, smem, split, tile);
  if constexpr (MODE == 0) {
    int* flag = reinterpret_cast<int*>(smem + L::kFlagOff);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) *flag = atomicAdd(p.counters + tile, 1) == S - 1;
    __syncthreads();
    if (!*flag) return;
    __threadfence();
    pre_tile<BM, S, STAGES, FNREG, true>(p, smem, tile);
  }
}

// Standalone pre: 8 tokens per CTA (one per warp) for parallelism at any M.
constexpr int kPreBM = 8;
template <int BM, int S>
__global__ void __launch_bounds__(kThreads, 2) mhc_pre_kernel(Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  pre_tile<BM, S, 1, true, false>(p, smem, blockIdx.x);
}

__global__ void __launch_bounds__(512)
    plain_copy_kernel(const uint4* src, uint4* dst, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    const uint4 a = src[i], b = src[i + stride], c = src[i + 2 * stride],
                d = src[i + 3 * stride];
    dst[i] = a;
    dst[i + stride] = b;
    dst[i + 2 * stride] = c;
    dst[i + 3 * stride] = d;
  }
  for (; i < n; i += stride) dst[i] = src[i];
}

// ---------------------------------------------------------------- host
template <typename K>
void set_smem(K kernel, int bytes) {
  // Idempotent and cheap; called on each launch so every device is covered.
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       bytes);
}

template <int BM, int S, bool FNREG>
void launch(const Params& p, int64_t mode, cudaStream_t stream) {
  // Ring depth: what fits two CTAs per SM (<= ~113 KB each on SM100/107).
  constexpr int STAGES = FNREG ? (BM == 32 ? 8 : 4) : (BM == 32 ? 3 : 2);
  using L = Layout<BM, STAGES, FNREG>;
  const int tiles = (p.M + BM - 1) / BM;
  TVM_FFI_ICHECK(tiles <= 65535) << "too many tokens";
  const dim3 grid(S, tiles);
  if (mode == 0) {
    auto k = mhc_post_gemm_kernel<BM, S, STAGES, FNREG, 0>;
    set_smem(k, L::kTotal);
    k<<<grid, kThreads, L::kTotal, stream>>>(p);
  } else {
    if (mode != 2) {
      auto k = mhc_post_gemm_kernel<BM, S, STAGES, FNREG, 1>;
      set_smem(k, L::kTotal);
      k<<<grid, kThreads, L::kTotal, stream>>>(p);
    }
    if (mode != 1) {
      using LP = Layout<kPreBM, 1, true>;
      auto k = mhc_pre_kernel<kPreBM, S>;
      set_smem(k, LP::kTotal);
      k<<<(p.M + kPreBM - 1) / kPreBM, kThreads, LP::kTotal, stream>>>(p);
    }
  }
}

#define GLM_MHCP_CHECK(t, code, ...)                           \
  TVM_FFI_ICHECK(encode_dlpack_dtype((t).dtype()) == (code) && \
                 (t).IsContiguous())                           \
      << #t " must be a contiguous " #code " tensor"

// mode 0: fused; 1: post + GEMM partials only; 2: pre only (from partials);
// 3: post + GEMM, then a separate pre launch. fn_reg: fn fragments from
// L1/L2 into registers (deeper r / x ring) instead of a shared-memory stage.
// flags (diagnostics, wrong output): bit 0 skip the MMAs, bit 1 skip the fn
// loads.
void mhc_post_pre(TensorView x, TensorView residual, TensorView post_in,
                  TensorView comb_in, TensorView fn, TensorView hc_scale,
                  TensorView hc_base, TensorView norm_w,
                  TensorView residual_out, TensorView post_out,
                  TensorView comb_out, TensorView layer_input, TensorView part,
                  TensorView sq, TensorView counters, double rms_eps,
                  double pre_eps, double sk_eps, double post_mult,
                  int64_t sk_iters, double norm_eps, int64_t splits,
                  int64_t block_m, int64_t mode, int64_t fn_tf32,
                  int64_t fn_reg, int64_t flags) {
  GLM_MHCP_CHECK(x, bfloat16_code);
  GLM_MHCP_CHECK(residual, bfloat16_code);
  GLM_MHCP_CHECK(post_in, float32_code);
  GLM_MHCP_CHECK(comb_in, float32_code);
  GLM_MHCP_CHECK(fn, float32_code);
  GLM_MHCP_CHECK(hc_scale, float32_code);
  GLM_MHCP_CHECK(hc_base, float32_code);
  GLM_MHCP_CHECK(norm_w, bfloat16_code);
  GLM_MHCP_CHECK(residual_out, bfloat16_code);
  GLM_MHCP_CHECK(post_out, float32_code);
  GLM_MHCP_CHECK(comb_out, float32_code);
  GLM_MHCP_CHECK(layer_input, bfloat16_code);
  GLM_MHCP_CHECK(part, float32_code);
  GLM_MHCP_CHECK(sq, float32_code);
  GLM_MHCP_CHECK(counters, int32_code);
  const int64_t M = x.size(0);
  TVM_FFI_ICHECK(x.numel() == M * kHidden) << "x must be [M, 4096]";
  TVM_FFI_ICHECK(residual.numel() == M * kK) << "residual must be [M, 4, 4096]";
  TVM_FFI_ICHECK(residual_out.numel() == M * kK);
  TVM_FFI_ICHECK(post_in.numel() == M * kHC && comb_in.numel() == M * 16);
  TVM_FFI_ICHECK(post_out.numel() == M * kHC && comb_out.numel() == M * 16);
  TVM_FFI_ICHECK(layer_input.numel() == M * kHidden);
  TVM_FFI_ICHECK(fn.numel() == kMix * kK) << "fn must be [24, 16384]";
  TVM_FFI_ICHECK(hc_scale.numel() == 3 && hc_base.numel() == kMix);
  TVM_FFI_ICHECK(norm_w.numel() == kHidden);
  TVM_FFI_ICHECK(part.numel() == splits * M * kMix && sq.numel() == splits * M);
  TVM_FFI_ICHECK(counters.numel() >= (M + block_m - 1) / block_m);
  TVM_FFI_ICHECK(mode >= 0 && mode <= 3);
  TVM_FFI_ICHECK(sk_iters >= 1);
  if (M == 0) return;

  Params p;
  p.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
  p.residual = static_cast<const __nv_bfloat16*>(residual.data_ptr());
  p.post_in = static_cast<const float*>(post_in.data_ptr());
  p.comb_in = static_cast<const float*>(comb_in.data_ptr());
  p.fn = static_cast<const float*>(fn.data_ptr());
  p.hc_scale = static_cast<const float*>(hc_scale.data_ptr());
  p.hc_base = static_cast<const float*>(hc_base.data_ptr());
  p.norm_w = static_cast<const __nv_bfloat16*>(norm_w.data_ptr());
  p.residual_out = static_cast<__nv_bfloat16*>(residual_out.data_ptr());
  p.post_out = static_cast<float*>(post_out.data_ptr());
  p.comb_out = static_cast<float*>(comb_out.data_ptr());
  p.layer_input = static_cast<__nv_bfloat16*>(layer_input.data_ptr());
  p.part = static_cast<float*>(part.data_ptr());
  p.sq = static_cast<float*>(sq.data_ptr());
  p.counters = static_cast<int*>(counters.data_ptr());
  p.M = static_cast<int>(M);
  p.sk_iters = static_cast<int>(sk_iters);
  p.fn_tf32 = static_cast<int>(fn_tf32);
  p.flags = static_cast<int>(flags);
  p.rms_eps = static_cast<float>(rms_eps);
  p.pre_eps = static_cast<float>(pre_eps);
  p.sk_eps = static_cast<float>(sk_eps);
  p.post_mult = static_cast<float>(post_mult);
  p.norm_eps = static_cast<float>(norm_eps);

  ffi::CUDADeviceGuard device_guard(x.device().device_id);
  const cudaStream_t stream = get_stream(x.device());
  bool ok = true;
#define GLM_MHCP_CASE(BM, S)                 \
  else if (block_m == BM && splits == S) {   \
    if (fn_reg)                              \
      launch<BM, S, true>(p, mode, stream);  \
    else                                     \
      launch<BM, S, false>(p, mode, stream); \
  }
  if (false) {
  }
  GLM_MHCP_CASE(64, 16)
  GLM_MHCP_CASE(64, 8)
  GLM_MHCP_CASE(64, 32)
  GLM_MHCP_CASE(32, 16)
  GLM_MHCP_CASE(32, 32)
  GLM_MHCP_CASE(32, 8)
  else {
    ok = false;
  }
#undef GLM_MHCP_CASE
  TVM_FFI_ICHECK(ok) << "unsupported (block_m, splits) = (" << block_m << ", "
                     << splits << ")";
  const cudaError_t e = cudaGetLastError();
  TVM_FFI_ICHECK(e == cudaSuccess)
      << "mhc_post_pre launch failed: " << cudaGetErrorString(e);
}

// Roofline helper: plain vectorised copy of src into dst (same byte size).
void copy(TensorView src, TensorView dst) {
  TVM_FFI_ICHECK(src.IsContiguous() && dst.IsContiguous());
  const int64_t nbytes = src.numel() * ((src.dtype().bits + 7) / 8);
  TVM_FFI_ICHECK(nbytes % 16 == 0 &&
                 dst.numel() * ((dst.dtype().bits + 7) / 8) == nbytes);
  ffi::CUDADeviceGuard device_guard(src.device().device_id);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                         src.device().device_id);
  plain_copy_kernel<<<4 * sms, 512, 0, get_stream(src.device())>>>(
      static_cast<const uint4*>(src.data_ptr()),
      static_cast<uint4*>(dst.data_ptr()), nbytes / 16);
  const cudaError_t e = cudaGetLastError();
  TVM_FFI_ICHECK(e == cudaSuccess) << cudaGetErrorString(e);
}

}  // namespace glm5next_mhcp

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_post_pre, glm5next_mhcp::mhc_post_pre);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(copy, glm5next_mhcp::copy);
