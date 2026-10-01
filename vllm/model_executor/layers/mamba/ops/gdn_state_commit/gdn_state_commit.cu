// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Deferred ("single-state") GDN MTP spec-decode state commit for vLLM.
// Stock (csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu): every step reads slot[acc-1] of the
// previous step and writes the state after EACH draft token to its own slot (1R + 4W x 64 KB per
// request/head); 3 of the 4 written states are never read.
// Deferred: every request owns ONE state slot (its mamba block) whose page additionally holds a
// token log (raw bf16 k/v/a/b of the last step's <=4 tokens) and a per-key-head flag L.
//   state_committed = replay(base, log[:acc]) if L else base
// The decode kernel (1) loads base, (2) replays the accepted prefix of the previous step (R = L ? acc
// : 0 tokens), (3) writes the committed state back (only if R > 0), (4) runs the new tokens with the
// state in registers (outputs only), (5) logs the new tokens and sets L = 1.
// The replay uses the SAME per-token arithmetic (same code, same thread mapping, same reduction
// order, compiled with --use_fast_math like vLLM) as the stock kernel, so the committed state is
// bit-identical to the stock kernel's slot[acc-1] state and the outputs are bit-identical to stock.
//
// Page layout per (slot, layer), from the ssm slot base (the page continues after the fp32 state):
//   [ssm fp32 HV*128*128][K bf16 [4][H][128]][V bf16 [4][HV][128]][AB bf16 [4][HV][2]][L int32 [H]]
// One CTA per (request, key head) processes its value heads (HV/H of them) so that the K log
// (per key head) has a single reader/writer.
//
// `materialize` replays n logged tokens from a source slot into a destination slot (all GDN layers
// of a KV-cache group in one launch) for every non-decode reader of GDN state: align-mode block
// migration (precopy), block-boundary checkpoints (postprocess), and non-spec (prefill / plain
// decode) reads of a slot with a pending log.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#ifndef GSC_QO
#define GSC_QO 0
#endif
#if GSC_QO
#include <cuda_fp8.h>
#endif

namespace gsc {

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxT = 4;      // logged tokens per step (1 + num_speculative_tokens)
constexpr int kMaxTok = 8;    // replay + new tokens held in smem
constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;
#ifndef GSC_MINB
#define GSC_MINB 2
#endif

struct LogLayout {
  int64_t k_off, v_off, ab_off, l_off, c_off, bytes;
};
__host__ __device__ __forceinline__ LogLayout log_layout(int H, int HV) {
  LogLayout l;
  l.k_off = 0;
  l.v_off = l.k_off + static_cast<int64_t>(kMaxT) * H * kDimK * 2;
  l.ab_off = l.v_off + static_cast<int64_t>(kMaxT) * HV * kDimV * 2;
  l.l_off = l.ab_off + static_cast<int64_t>(kMaxT) * HV * 2 * 2;
  l.c_off = l.l_off + static_cast<int64_t>(HV) * 4;
  l.bytes = l.c_off + static_cast<int64_t>(H) * 4;
  return l;
}

__device__ __forceinline__ void cp_async_16b(void* smem_ptr, const void* gmem_ptr) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

template <typename S>
__device__ __forceinline__ void copy_state_chunk(S* shared_state, const S* state, int chunk, int stage, int thread) {
  constexpr int kElementsPerCopy = 16 / sizeof(S);
  constexpr int kCopiesPerChunk = kChunkV * kDimK / kElementsPerCopy;
  for (int copy = thread; copy < kCopiesPerChunk; copy += kThreads) {
    const int element = copy * kElementsPerCopy;
    cp_async_16b(shared_state + stage * kChunkV * kDimK + element, state + chunk * kChunkV * kDimK + element);
  }
  cp_async_commit();
}

// State element I/O (same conversions as stock load_state4/store_state4). BF16 state: the committed
// state is rounded to bf16 (RN) exactly where stock stores its per-token slot; the fp32 registers are
// then replaced by the rounded values, which is what stock's next step reloads.
template <typename S>
__device__ __forceinline__ void load_h4(const S* p, float* h);
template <>
__device__ __forceinline__ void load_h4<float>(const float* p, float* h) {
  const float4 v = *reinterpret_cast<const float4*>(p);
  h[0] = v.x; h[1] = v.y; h[2] = v.z; h[3] = v.w;
}
template <>
__device__ __forceinline__ void load_h4<__nv_bfloat16>(const __nv_bfloat16* p, float* h) {
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(p);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(p + 2);
  h[0] = __bfloat162float(lo.x); h[1] = __bfloat162float(lo.y);
  h[2] = __bfloat162float(hi.x); h[3] = __bfloat162float(hi.y);
}
template <typename S>
__device__ __forceinline__ void store_h4(S* p, float* h);  // stores and rounds h in place
template <>
__device__ __forceinline__ void store_h4<float>(float* p, float* h) {
  *reinterpret_cast<float4*>(p) = make_float4(h[0], h[1], h[2], h[3]);
}
template <>
__device__ __forceinline__ void store_h4<__nv_bfloat16>(__nv_bfloat16* p, float* h) {
  const __nv_bfloat162 lo = __floats2bfloat162_rn(h[0], h[1]);
  const __nv_bfloat162 hi = __floats2bfloat162_rn(h[2], h[3]);
  *reinterpret_cast<__nv_bfloat162*>(p) = lo;
  *reinterpret_cast<__nv_bfloat162*>(p + 2) = hi;
  h[0] = __bfloat162float(lo.x); h[1] = __bfloat162float(lo.y);
  h[2] = __bfloat162float(hi.x); h[3] = __bfloat162float(hi.y);
}

__device__ __forceinline__ float sigmoid_fast(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu_fast(float x) { return x * sigmoid_fast(x); }
__device__ __forceinline__ float softplus_fast(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }
__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head, int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) return __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
  if (dt_bias_type == kDtBiasFloat16) return __half2float(static_cast<const __half*>(dt_bias)[head]);
  return static_cast<const float*>(dt_bias)[head];
}
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}
#if GSC_QO
// gb300 glue (GSC_QO=1): the decode kernel also writes the MXFP8 (E4M3 data + E8M0 scale per 32 values,
// 128x4-swizzled scales) quantization of its bf16 output for the GDN out_proj (K = HV * 128), bit-identical
// to FlashInfer mxfp8_quantize(out, is_sf_swizzled_layout=True), and zeroes the data / scales of the rows no
// request owns (graph padding rows [cu_seqlens[n], T) and the scale padding rows [T, pm)).
struct QoArgs {
  uint8_t* q;       // [>= pm, HV*128] e4m3 bytes, row stride stride_q (nullptr: QO off)
  uint8_t* sf;      // swizzled e8m0 scales of [pm, HV*4] (padded cols psc)
  int stride_q;     // row stride of q in bytes (q offsets fit in int32: rows <= 32768, K <= 65536)
  int psc;          // padded scale columns ((HV*4 + 3) / 4 * 4)
  int T;            // rows of out (graph size)
  int pm;           // T rounded up to 128
};
__device__ __forceinline__ float qo_mul_rn(float a, float b) {  // IEEE mul, no FTZ / no contraction
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ int qo_sf_word(int row, int head, int psc) {
  // byte offset of scale column j = head * 4 (+0..3) of `row` in FlashInfer's 128x4 swizzled layout (32-bit math)
  return head * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4 + (row >> 7) * (128 * psc);
}
// e8m0 exponent of a 32-value block whose |max| is amax (FlashInfer cute-dsl rule)
__device__ __forceinline__ uint32_t qo_e8m0(float amax) {
  const float nm = qo_mul_rn(amax, 1.0f / 448.0f);
  const uint32_t bits = __float_as_uint(nm);
  const uint32_t e = (bits >> 23) & 255u;
  const uint32_t mant = bits & 0x7FFFFFu;
  const uint32_t bump = (mant != 0u && !(e == 0u && mant <= 0x400000u)) ? 1u : 0u;
  uint32_t e2 = e + bump < 254u ? e + bump : 254u;
  if (!(nm > 0.0f)) e2 = 0u;
  return e2;
}
__device__ __forceinline__ uint8_t qo_e4m3(float y, uint32_t e2) {
  const float inv = e2 == 0u ? 0.0f : __uint_as_float((254u - e2) << 23);
  float v = qo_mul_rn(y, inv);
  v = fminf(fmaxf(v, -448.0f), 448.0f);
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}
// one 32-value block (value = lane + 32 * i) of one (token row, head): warp |max|, e8m0, e4m3 byte store;
// returns the e8m0 byte (all lanes)
__device__ __forceinline__ uint32_t qo_block(const QoArgs& qo, int row, int head, int lane, int i, float y) {
  float am = fabsf(y);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
  const uint32_t e2 = qo_e8m0(am);
  qo.q[row * qo.stride_q + head * kDimV + lane + 32 * i] = qo_e4m3(y, e2);
  return e2;
}
// zero q (rows < T) / scales (rows < pm) of the rows [n_tok, pm) assigned to this CTA (row = n_tok + request + k *
// n_req), spread over all threads (32 q words per row, 1 scale word per row)
__device__ __forceinline__ void qo_zero_pad_rows(const QoArgs& qo, int n_tok, int request, int n_req, int head, int tid) {
  const int first = n_tok + request;
  if (first >= qo.pm) return;
  const int nq = first < qo.T ? (qo.T - first + n_req - 1) / n_req : 0;
  for (int idx = tid; idx < nq * 32; idx += kThreads) {
    const int row = first + (idx >> 5) * n_req;
    reinterpret_cast<uint32_t*>(qo.q + row * qo.stride_q + head * kDimV)[idx & 31] = 0u;
  }
  const int ns = (qo.pm - first + n_req - 1) / n_req;
  for (int r = tid; r < ns; r += kThreads)
    *reinterpret_cast<uint32_t*>(qo.sf + qo_sf_word(first + r * n_req, head, qo.psc)) = 0u;
}
#define GSC_QO_PARAM , QoArgs qo
#define GSC_QO_ARG , qo
#else
#define GSC_QO_PARAM
#define GSC_QO_ARG
#endif
struct Sum2 { float x; float y; };
__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

// Stock per-token prep (verbatim arithmetic). Called by one warp. q may be null (replay token).
__device__ __forceinline__ void prep_qk(const __nv_bfloat16* qraw, const __nv_bfloat16* kraw, float scale,
                                        int lane, float* sq, float* sk) {
  float q_values[4];
  float k_values[4];
  float q_square = 0.0f;
  float k_square = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int dim = lane + i * 32;
    q_values[i] = qraw ? __bfloat162float(qraw[dim]) : 0.0f;
    k_values[i] = __bfloat162float(kraw[dim]);
    q_square += q_values[i] * q_values[i];
    k_square += k_values[i] * k_values[i];
  }
  const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
  const float q_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
  const float k_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int dim = lane + i * 32;
    if (sq) sq[dim] = q_values[i] * q_scale;
    sk[dim] = k_values[i] * k_scale;
  }
}
__device__ __forceinline__ void prep_gate(float a_value, float b_value, float a_log, float dtb, float* decay,
                                          float* beta) {
  const float g = -__expf(a_log) * softplus_fast(a_value + dtb);
  *decay = __expf(g);
  *beta = sigmoid_fast(b_value);
}

// Reduce-scatter version of stock's two warp_reduce_sum_pair() calls over 4 row partials
// (xor masks 16,8,4,2,1 in the same order). Every reduced value is produced by the identical
// sequence of pairwise IEEE adds as the stock butterfly (a+b == b+a), only the lane holding it
// differs: after the call lanes [8r, 8r+8) hold row r's sum. 6 SHFL instead of 20.
__device__ __forceinline__ float reduce_scatter4(float a0, float a1, float a2, float a3, int lane) {
  const bool hi16 = (lane & 16) != 0;
  const float send0 = hi16 ? a0 : a2, send1 = hi16 ? a1 : a3;
  float keep0 = hi16 ? a2 : a0, keep1 = hi16 ? a3 : a1;
  keep0 += __shfl_xor_sync(0xffffffffu, send0, 16);
  keep1 += __shfl_xor_sync(0xffffffffu, send1, 16);
  const bool hi8 = (lane & 8) != 0;
  float v = hi8 ? keep1 : keep0;
  v += __shfl_xor_sync(0xffffffffu, hi8 ? keep0 : keep1, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}

// One token of the delta rule on this warp's 4 rows x 4 k-columns (stock arithmetic, verbatim).
template <bool kOut>
__device__ __forceinline__ void token_update(float (&h)[kRowsPerWarp][4], const float* sk, const float* sq,
                                             const __nv_bfloat16* sv, float decay, float beta, int chunk,
                                             const int (&rows)[kRowsPerWarp], int k_base, int lane,
                                             __nv_bfloat16* sout) {
  const float4 k4 = *reinterpret_cast<const float4*>(&sk[k_base]);
  const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};
  float q_values[4] = {0.f, 0.f, 0.f, 0.f};
  if constexpr (kOut) {
    const float4 q4 = *reinterpret_cast<const float4*>(&sq[k_base]);
    q_values[0] = q4.x; q_values[1] = q4.y; q_values[2] = q4.z; q_values[3] = q4.w;
  }
  float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[row][i] *= decay;
      dot_hk[row] += h[row][i] * k_values[i];
    }
  }
  const float hk_mine = reduce_scatter4(dot_hk[0], dot_hk[1], dot_hk[2], dot_hk[3], lane);
  const float reduced_hk[kRowsPerWarp] = {__shfl_sync(0xffffffffu, hk_mine, 0), __shfl_sync(0xffffffffu, hk_mine, 8),
                                          __shfl_sync(0xffffffffu, hk_mine, 16),
                                          __shfl_sync(0xffffffffu, hk_mine, 24)};
  float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    const int value = chunk * kChunkV + rows[row];
    const float delta = (__bfloat162float(sv[value]) - reduced_hk[row]) * beta;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[row][i] += k_values[i] * delta;
      if constexpr (kOut) dot_hq[row] += h[row][i] * q_values[i];
    }
  }
  if constexpr (kOut) {
    const float hq_mine = reduce_scatter4(dot_hq[0], dot_hq[1], dot_hq[2], dot_hq[3], lane);
    if ((lane & 7) == 0) sout[chunk * kChunkV + rows[lane >> 3]] = __float2bfloat16(hq_mine);
  }
}

#ifndef GSC_PROBE
#define GSC_PROBE 0
#endif
#ifndef GSC_NC
#define GSC_NC 1
#endif
#ifndef GSC_F2
#define GSC_F2 0
#endif
// Packed fp32x2 (FFMA2/FMUL2). Same IEEE op per lane as the scalar FFMA/FMUL (ftz like --use_fast_math).
__device__ __forceinline__ unsigned long long p2(float a, float b) {
  return static_cast<unsigned long long>(__float_as_uint(a)) | (static_cast<unsigned long long>(__float_as_uint(b)) << 32);
}
__device__ __forceinline__ void u2(unsigned long long v, float& a, float& b) {
  a = __uint_as_float(static_cast<unsigned>(v)); b = __uint_as_float(static_cast<unsigned>(v >> 32));
}
__device__ __forceinline__ unsigned long long ffma2(unsigned long long a, unsigned long long b, unsigned long long c) {
  unsigned long long d;
  asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
  return d;
}
__device__ __forceinline__ unsigned long long fmul2(unsigned long long a, unsigned long long b) {
  unsigned long long d;
  asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}

// Generic reduce-scatter over NR = 4*NC row partials (xor masks 16,8,4,2,1 as stock). Same
// pairwise-add tree per value as stock's butterfly; afterwards lanes [r*32/NR, (r+1)*32/NR) hold row r.
template <int NR>
__device__ __forceinline__ float reduce_scatter(const float (&a)[NR], int lane) {
  float v[NR];
#pragma unroll
  for (int j = 0; j < NR; ++j) v[j] = a[j];
  int cnt = NR;
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) {
    if (cnt > 1) {
      const int half = cnt / 2;
      const bool hi = (lane & m) != 0;
#pragma unroll
      for (int j = 0; j < NR / 2; ++j) {
        if (j < half) {
          const float send = hi ? v[j] : v[j + half];
          const float keep = hi ? v[j + half] : v[j];
          v[j] = keep + __shfl_xor_sync(0xffffffffu, send, m);
        }
      }
      cnt = half;
    } else {
      v[0] += __shfl_xor_sync(0xffffffffu, v[0], m);
    }
  }
  return v[0];
}

// token_update over NC chunks at once (4*NC independent rows -> ILP), stock per-row arithmetic.
template <bool kOut, int NC>
__device__ __forceinline__ void token_update_multi(float (&h)[4 * NC][4], const float* sk, const float* sq,
                                                   const __nv_bfloat16* sv, float decay, float beta, int c0,
                                                   const int (&rows)[kRowsPerWarp], int k_base, int lane,
                                                   __nv_bfloat16* sout) {
  constexpr int NR = 4 * NC;
  const float4 k4 = *reinterpret_cast<const float4*>(&sk[k_base]);
  const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};
  float q_values[4] = {0.f, 0.f, 0.f, 0.f};
  if constexpr (kOut) {
    const float4 q4 = *reinterpret_cast<const float4*>(&sq[k_base]);
    q_values[0] = q4.x; q_values[1] = q4.y; q_values[2] = q4.z; q_values[3] = q4.w;
  }
  float dot_hk[NR];
#if GSC_F2
  {
    const unsigned long long dd = p2(decay, decay);
#pragma unroll
    for (int row = 0; row < NR; row += 2) {
#pragma unroll
      for (int i = 0; i < 4; i += 2) {  // h *= decay (FMUL), packed along i
        unsigned long long x = fmul2(p2(h[row][i], h[row][i + 1]), dd);
        u2(x, h[row][i], h[row][i + 1]);
        x = fmul2(p2(h[row + 1][i], h[row + 1][i + 1]), dd);
        u2(x, h[row + 1][i], h[row + 1][i + 1]);
      }
      unsigned long long acc = p2(0.0f, 0.0f);  // dot += h*k (FFMA), packed across rows
#pragma unroll
      for (int i = 0; i < 4; ++i) acc = ffma2(p2(h[row][i], h[row + 1][i]), p2(k_values[i], k_values[i]), acc);
      u2(acc, dot_hk[row], dot_hk[row + 1]);
    }
  }
#else
#pragma unroll
  for (int row = 0; row < NR; ++row) {
    dot_hk[row] = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[row][i] *= decay;
      dot_hk[row] += h[row][i] * k_values[i];
    }
  }
#endif
  const float hk_mine = reduce_scatter<NR>(dot_hk, lane);
  float dot_hq[NR];
#pragma unroll
  for (int row = 0; row < NR; ++row) {
    const float reduced = __shfl_sync(0xffffffffu, hk_mine, row * (32 / NR));
    const int value = (c0 + row / 4) * kChunkV + rows[row % 4];
    const float delta = (__bfloat162float(sv[value]) - reduced) * beta;
    dot_hq[row] = 0.0f;
#if GSC_F2
    {
      const unsigned long long dl = p2(delta, delta);
      unsigned long long x = ffma2(p2(k_values[0], k_values[1]), dl, p2(h[row][0], h[row][1]));
      u2(x, h[row][0], h[row][1]);
      x = ffma2(p2(k_values[2], k_values[3]), dl, p2(h[row][2], h[row][3]));
      u2(x, h[row][2], h[row][3]);
      if constexpr (kOut) {
#pragma unroll
        for (int i = 0; i < 4; ++i) dot_hq[row] += h[row][i] * q_values[i];
      }
    }
#else
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[row][i] += k_values[i] * delta;
      if constexpr (kOut) dot_hq[row] += h[row][i] * q_values[i];
    }
#endif
  }
  if constexpr (kOut) {
    const float hq_mine = reduce_scatter<NR>(dot_hq, lane);
    if ((lane & (32 / NR - 1)) == 0) {
      const int row = lane / (32 / NR);
      sout[(c0 + row / 4) * kChunkV + rows[row % 4]] = __float2bfloat16(hq_mine);
    }
  }
}

#ifndef GSC_EARLY
#define GSC_EARLY 0
#endif
#ifndef GSC_FAST
#define GSC_FAST 0
#endif
// GSC_FAST: row-per-8-lanes mapping. Each thread owns ONE value row (per chunk)
// and 16 k-columns (k = j*32 + seg*4 + e, seg = lane&7, j,e in 0..3), so each per-row dot product is
// 16 in-thread FFMAs (4 independent chains) + a 3-level xor butterfly (4,2,1) inside the 8-lane group,
// and every lane of the group ends with the full sum (no broadcast SHFL). Per token and row group:
// 6 SHFL instead of 20; FFMA count unchanged. fp32 summation order differs from stock (float-order
// class, not bit-exact). Rows: rr = warp*4 + (lane>>3) inside each 32-row chunk; NR = chunks per pass.
template <bool kOut, int NR>
__device__ __forceinline__ void token_update_fast(float (&h)[NR][16], const float* sk, const float* sq,
                                                  const __nv_bfloat16* sv, float decay, float beta, int c0, int rr,
                                                  int seg, __nv_bfloat16* sout) {
  float kv[16];
  float qv[16];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float4 k4 = *reinterpret_cast<const float4*>(&sk[j * 32 + seg * 4]);
    kv[j * 4 + 0] = k4.x; kv[j * 4 + 1] = k4.y; kv[j * 4 + 2] = k4.z; kv[j * 4 + 3] = k4.w;
    if constexpr (kOut) {
      const float4 q4 = *reinterpret_cast<const float4*>(&sq[j * 32 + seg * 4]);
      qv[j * 4 + 0] = q4.x; qv[j * 4 + 1] = q4.y; qv[j * 4 + 2] = q4.z; qv[j * 4 + 3] = q4.w;
    }
  }
  float dot[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        h[r][j * 4 + e] *= decay;
        acc[j] += h[r][j * 4 + e] * kv[j * 4 + e];
      }
    }
    dot[r] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  }
#pragma unroll
  for (int m = 4; m >= 1; m >>= 1) {
#pragma unroll
    for (int r = 0; r < NR; ++r) dot[r] += __shfl_xor_sync(0xffffffffu, dot[r], m);
  }
  float hq[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    const float delta = (__bfloat162float(sv[(c0 + r) * kChunkV + rr]) - dot[r]) * beta;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        h[r][j * 4 + e] += kv[j * 4 + e] * delta;
        if constexpr (kOut) acc[j] += h[r][j * 4 + e] * qv[j * 4 + e];
      }
    }
    hq[r] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  }
  if constexpr (kOut) {
#pragma unroll
    for (int m = 4; m >= 1; m >>= 1) {
#pragma unroll
      for (int r = 0; r < NR; ++r) hq[r] += __shfl_xor_sync(0xffffffffu, hq[r], m);
    }
    if (seg == 0) {
#pragma unroll
      for (int r = 0; r < NR; ++r) sout[(c0 + r) * kChunkV + rr] = __float2bfloat16(hq[r]);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// GSC_CK: chunked ("WY") form of the delta rule over the <= 4 replay + <= 4 new tokens.
// For a segment of n tokens starting from state S (rows v, cols k), with cumulative decay
// G_t = d_0*...*d_t and rho(s,t) = d_{s+1}*...*d_t (rho(t,t) = 1):
//   kS_t = G_t (S k_t) + sum_{s<t} rho(s,t) (k_s.k_t) delta_s,  delta_t = beta_t (v_t - kS_t)
//   o_t  = G_t (S q_t) + sum_{s<=t} rho(s,t) (k_s.q_t) delta_s
//   S_end = G_{n-1} S + sum_s rho(s,n-1) delta_s k_s^T
// which is the stock recurrence (h <- d h; kS = h.k; delta; h += delta k^T; o = h.q) re-associated.
// Per state element: 2R + 1 (replay: dots + update) + 2T (new tokens: dots only; their state is never
// needed) FMAs instead of ~4 per token, and no serial per-token dependency through the state: all dot
// products of a row are independent; only the tiny per-row scalar solve is sequential.
// fp32 everywhere (stock-level accumulation); explicit __f*_rn intrinsics (no contraction freedom), so the
// replay is bitwise identical in the decode kernel and in materialize (path independence), run to run.
// Mapping (as GSC_FAST): row rr = warp*4 + (lane>>3) of a 32-row chunk, 16 k-columns k = j*32 + seg*4 + e.
// ---------------------------------------------------------------------------------------------
#ifndef GSC_CK
#define GSC_CK 0
#endif
struct CkRep {
  float G[kMaxT];
  float W[kMaxT][kMaxT];  // rho(i,j) * (k_i.k_j), i < j
  float C[kMaxT];         // rho(j, R-1)
  float beta[kMaxT];
  float GR;               // G_{R-1}
};
struct CkNew {
  float G[kMaxT];
  float W[kMaxT][kMaxT];  // rho(s,t) * (k_s.k_t), s < t
  float Q[kMaxT][kMaxT];  // rho(s,t) * (k_s.q_t), s <= t
  float beta[kMaxT];
};

// 8-lane dot of two 128-vectors (fixed order: 4 in-thread chains, (c0+c1)+(c2+c3), xor 4,2,1).
__device__ __forceinline__ float ck_dot8(const float* x, const float* y, int seg) {
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int e = 0; e < 4; ++e) {
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = __fmaf_rn(x[j * 32 + seg * 4 + e], y[j * 32 + seg * 4 + e], acc[j]);
  }
  float s = __fadd_rn(__fadd_rn(acc[0], acc[1]), __fadd_rn(acc[2], acc[3]));
  s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 4));
  s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 2));
  s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1));
  return s;
}

// Gram entries needed by the replay segment [0,R) and the new segment [R,R+T): kk[i][j] (i<j, same segment)
// and kq[s][t] = k_{R+s}.q_t (s<=t). Called by ALL threads of the CTA (8-lane groups, uniform shuffles).
__device__ __forceinline__ void ck_gram(const float (*sk)[kDimK], const float (*sq)[kDimK], int R, int T, int tid,
                                        float (*kk)[kMaxTok], float (*kq)[kMaxT]) {
  const int grp = tid >> 3, seg = tid & 7;
  const int total = R + T;
#pragma unroll
  for (int round = 0; round < 2; ++round) {
    const int p = grp + round * (kThreads / 8);
    int i = 0, j = 0;
    bool valid = false, isq = false;
    if (p < 28) {
      // pairs (i<j<8) enumerated by j: p in [j(j-1)/2, j(j+1)/2); packed table j | i << 4
      constexpr unsigned long long kLo = 0x0000000000000000ULL;  // (unused; see kPair)
      (void)kLo;
      const int jj = p < 1 ? 1 : p < 3 ? 2 : p < 6 ? 3 : p < 10 ? 4 : p < 15 ? 5 : p < 21 ? 6 : 7;
      j = jj;
      i = p - j * (j - 1) / 2;
      valid = (j < R) || (i >= R && j < total);
    } else if (p < 28 + kMaxT * kMaxT) {
      i = (p - 28) / kMaxT;  // s
      j = (p - 28) % kMaxT;  // t
      isq = true;
      valid = sq != nullptr && i <= j && j < T;
    }
    const float* x = valid ? sk[isq ? R + i : i] : sk[0];
    const float* y = valid ? (isq ? sq[j] : sk[j]) : sk[0];
    const float d = ck_dot8(x, y, seg);
    if (valid && seg == 0) {
      if (isq) kq[i][j] = d;
      else kk[i][j] = d;
    }
  }
}

__device__ __forceinline__ void ck_coef_rep(const float* decay, const float* beta, const float (*kk)[kMaxTok], int R,
                                            CkRep* c) {
  float g = 1.0f;
  for (int j = 0; j < R; ++j) {
    g = __fmul_rn(g, decay[j]);
    c->G[j] = g;
    c->beta[j] = beta[j];
    for (int i = 0; i < j; ++i) {
      float rho = 1.0f;
      for (int r = i + 1; r <= j; ++r) rho = __fmul_rn(rho, decay[r]);
      c->W[i][j] = __fmul_rn(rho, kk[i][j]);
    }
  }
  for (int j = 0; j < R; ++j) {
    float rho = 1.0f;
    for (int r = j + 1; r < R; ++r) rho = __fmul_rn(rho, decay[r]);
    c->C[j] = rho;
  }
  c->GR = R > 0 ? c->G[R - 1] : 1.0f;
}

__device__ __forceinline__ void ck_coef_new(const float* decay, const float* beta, const float (*kk)[kMaxTok], int off,
                                            const float (*kq)[kMaxT], int T, CkNew* c) {
  float g = 1.0f;
  for (int t = 0; t < T; ++t) {
    g = __fmul_rn(g, decay[t]);
    c->G[t] = g;
    c->beta[t] = beta[t];
    for (int s = 0; s <= t; ++s) {
      float rho = 1.0f;
      for (int r = s + 1; r <= t; ++r) rho = __fmul_rn(rho, decay[r]);
      if (s < t) c->W[s][t] = __fmul_rn(rho, kk[off + s][off + t]);
      c->Q[s][t] = __fmul_rn(rho, kq[s][t]);
    }
  }
}

// Lane-parallel versions (one warp each); every entry is computed with exactly the same operations as the
// sequential ck_coef_* (products from 1.0f in increasing r), so the values are identical.
__device__ __forceinline__ void ck_coef_rep_par(const float* decay, const float* beta, const float (*kk)[kMaxTok],
                                                int R, CkRep* c, int lane) {
  if (lane < kMaxT) {
    const int j = lane;
    if (j < R) {
      float g = 1.0f;
      for (int r = 0; r <= j; ++r) g = __fmul_rn(g, decay[r]);
      c->G[j] = g;
      c->beta[j] = beta[j];
      float rho = 1.0f;
      for (int r = j + 1; r < R; ++r) rho = __fmul_rn(rho, decay[r]);
      c->C[j] = rho;
    }
  } else if (lane < kMaxT + kMaxT * kMaxT) {
    const int i = (lane - kMaxT) >> 2, j = (lane - kMaxT) & 3;
    if (i < j && j < R) {
      float rho = 1.0f;
      for (int r = i + 1; r <= j; ++r) rho = __fmul_rn(rho, decay[r]);
      c->W[i][j] = __fmul_rn(rho, kk[i][j]);
    }
  } else if (lane == kMaxT + kMaxT * kMaxT) {
    float g = 1.0f;
    for (int r = 0; r < R; ++r) g = __fmul_rn(g, decay[r]);
    c->GR = R > 0 ? g : 1.0f;
  }
}
__device__ __forceinline__ void ck_coef_new_par(const float* decay, const float* beta, const float (*kk)[kMaxTok], int off,
                                                const float (*kq)[kMaxT], int T, CkNew* c, int lane) {
  if (lane < kMaxT * kMaxT) {
    const int s = lane >> 2, t = lane & 3;
    if (s <= t && t < T) {
      float rho = 1.0f;
      for (int r = s + 1; r <= t; ++r) rho = __fmul_rn(rho, decay[r]);
      if (s < t) c->W[s][t] = __fmul_rn(rho, kk[off + s][off + t]);
      c->Q[s][t] = __fmul_rn(rho, kq[s][t]);
    }
  } else if (lane < kMaxT * kMaxT + kMaxT) {
    const int t = lane - kMaxT * kMaxT;
    if (t < T) {
      float g = 1.0f;
      for (int r = 0; r <= t; ++r) g = __fmul_rn(g, decay[r]);
      c->G[t] = g;
      c->beta[t] = beta[t];
    }
  }
}

// In-thread partial of row . x over this thread's 16 columns (same order as ck_dot8's in-thread part).
template <int NR>
__device__ __forceinline__ void ck_rowdots(const float (&h)[NR][16], const float* x, int seg, float (&d)[NR]) {
  float xv[16];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float4 a = *reinterpret_cast<const float4*>(&x[j * 32 + seg * 4]);
    xv[j * 4 + 0] = a.x; xv[j * 4 + 1] = a.y; xv[j * 4 + 2] = a.z; xv[j * 4 + 3] = a.w;
  }
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] = __fmaf_rn(h[r][j * 4 + e], xv[j * 4 + e], acc[j]);
    }
    d[r] = __fadd_rn(__fadd_rn(acc[0], acc[1]), __fadd_rn(acc[2], acc[3]));
  }
}

// Replay R (1..4) logged tokens on this thread's NR rows; h becomes the committed state (fp32, unrounded).
// sv: value vectors of this value head, [token][kDimV] with row stride kDimV (token j = replay token j).
template <int NR>
__device__ __forceinline__ void ck_replay(float (&h)[NR][16], const float (*sk)[kDimK], const __nv_bfloat16* sv,
                                          const CkRep& c, int R, const int (&row)[NR], int seg) {
  float a[kMaxT][NR];
#pragma unroll
  for (int j = 0; j < kMaxT; ++j) {
    if (j < R) ck_rowdots<NR>(h, sk[j], seg, a[j]);
  }
#pragma unroll
  for (int m = 4; m >= 1; m >>= 1) {
#pragma unroll
    for (int j = 0; j < kMaxT; ++j) {
      if (j < R) {
#pragma unroll
        for (int r = 0; r < NR; ++r) a[j][r] = __fadd_rn(a[j][r], __shfl_xor_sync(0xffffffffu, a[j][r], m));
      }
    }
  }
  float cc[kMaxT][NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    float d[kMaxT];
#pragma unroll
    for (int j = 0; j < kMaxT; ++j) {
      if (j < R) {
        float ks = __fmul_rn(c.G[j], a[j][r]);
#pragma unroll
        for (int i = 0; i < j; ++i) ks = __fmaf_rn(c.W[i][j], d[i], ks);
        d[j] = __fmul_rn(__fsub_rn(__bfloat162float(sv[j * kDimV + row[r]]), ks), c.beta[j]);
        cc[j][r] = __fmul_rn(c.C[j], d[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < kMaxT; ++j) {
    if (j < R) {
      float kv[16];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const float4 k4 = *reinterpret_cast<const float4*>(&sk[j][jj * 32 + seg * 4]);
        kv[jj * 4 + 0] = k4.x; kv[jj * 4 + 1] = k4.y; kv[jj * 4 + 2] = k4.z; kv[jj * 4 + 3] = k4.w;
      }
#pragma unroll
      for (int r = 0; r < NR; ++r) {
#pragma unroll
        for (int x = 0; x < 16; ++x) {
          const float base = j == 0 ? __fmul_rn(h[r][x], c.GR) : h[r][x];
          h[r][x] = __fmaf_rn(kv[x], cc[j][r], base);
        }
      }
    }
  }
}

// New tokens (T = 1..4) from the (committed) state h: outputs only, written to sout[t][row] (bf16).
// sk/sv point at the new segment (index t); sq[t] is the scaled q of new token t.
template <int NR>
__device__ __forceinline__ void ck_new(const float (&h)[NR][16], const float (*sk)[kDimK], const float (*sq)[kDimK],
                                       const __nv_bfloat16* sv, const CkNew& c, int T, const int (&row)[NR], int seg,
                                       __nv_bfloat16 (*sout)[kDimV]) {
  float a[kMaxT][NR];
  float b[kMaxT][NR];
#pragma unroll
  for (int t = 0; t < kMaxT; ++t) {
    if (t < T) {
      ck_rowdots<NR>(h, sk[t], seg, a[t]);
      ck_rowdots<NR>(h, sq[t], seg, b[t]);
    }
  }
#pragma unroll
  for (int m = 4; m >= 1; m >>= 1) {
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) {
      if (t < T) {
#pragma unroll
        for (int r = 0; r < NR; ++r) {
          a[t][r] = __fadd_rn(a[t][r], __shfl_xor_sync(0xffffffffu, a[t][r], m));
          b[t][r] = __fadd_rn(b[t][r], __shfl_xor_sync(0xffffffffu, b[t][r], m));
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    float d[kMaxT];
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) {
      if (t < T) {
        float ks = __fmul_rn(c.G[t], a[t][r]);
#pragma unroll
        for (int s = 0; s < t; ++s) ks = __fmaf_rn(c.W[s][t], d[s], ks);
        d[t] = __fmul_rn(__fsub_rn(__bfloat162float(sv[t * kDimV + row[r]]), ks), c.beta[t]);
        float o = __fmul_rn(c.G[t], b[t][r]);
#pragma unroll
        for (int s = 0; s <= t; ++s) o = __fmaf_rn(c.Q[s][t], d[s], o);
        if (seg == 0) sout[t][row[r]] = __float2bfloat16(o);
      }
    }
  }
}

struct Strides {
  int64_t mixed_row, a_row, b_row, gate_row, state_slot;  // state_slot in floats
};

template <int N>
__device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
// ---------------------------------------------------------------------------------------------
// GSC_CK == 2 (bf16 state): the CK math on the tensor cores (mma.sync m16n8k16 bf16, fp32 acc).
//  * dots S.x (x = replay k_j / new k_t, q_t): A = state (exactly bf16: the base slot, or the commit which
//    stock also rounds to bf16), B = x split into 3 bf16 terms (x = x1 + x2 + x3 to ~2^-24), fp32 accumulate.
//  * replay commit: S_c = G_R S + sum_j c_j k_j^T as an MMA with K = token slots: A = c split 3, B = k split 3,
//    6 cross terms (c1k1 c2k1 c1k2 | c3k1 c2k2 c1k3), accumulator initialised with G_R * S (fp32).
//  * warp w owns state rows [16w, 16w+16) for the whole kernel: its own cp.async group, __syncwarp only.
//  * smem state [128][128] bf16, 16-B chunks XOR-swizzled by (row & 7): ldmatrix and the C-layout
//    read/write-back are bank-conflict free.
// Decode and materialize call the same mm_* functions -> bitwise path independence.
// ---------------------------------------------------------------------------------------------
constexpr int kVPad = 136;  // bf16 row stride of the split vector arrays (272 B: ldmatrix conflict free)
constexpr int kKTPad = 40;  // bf16 row stride of the transposed k table (80 B)

__device__ __forceinline__ int st_idx(int row, int col) {
  return row * kDimK + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7);
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a) : "memory");
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(lo)) | (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}
// x = s0 + s1 + s2 (bf16 each), residual ~2^-24 |x|
__device__ __forceinline__ void split3(float x, __nv_bfloat16 (&s)[3]) {
  s[0] = __float2bfloat16_rn(x);
  float r = __fsub_rn(x, __bfloat162float(s[0]));
  s[1] = __float2bfloat16_rn(r);
  r = __fsub_rn(r, __bfloat162float(s[1]));
  s[2] = __float2bfloat16_rn(r);
}
// token-slot tables of the replay-commit MMA (2 k16 tiles x 4 groups of 4 slots; -1 = zero)
__device__ __forceinline__ int upd_asplit(int tile, int grp) { return grp == 3 ? -1 : (tile == 0 ? (grp == 1 ? 1 : 0) : 2 - grp); }
__device__ __forceinline__ int upd_bsplit(int tile, int grp) { return grp == 3 ? -1 : (tile == 0 ? (grp == 2 ? 1 : 0) : grp); }

// per-warp cp.async of state rows [16w, 16w+16) into the swizzled smem image (one commit group)
__device__ __forceinline__ void mm_load_rows(__nv_bfloat16* st, const __nv_bfloat16* src, int warp, int lane) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = i * 32 + lane;
    const int row = warp * 16 + (q >> 4), ch = q & 15;
    cp_async_16b(st + row * kDimK + ((ch ^ (row & 7)) << 3), src + row * kDimK + ch * 8);
  }
  cp_async_commit();
}
// coalesced write of the warp's rows from the swizzled smem image to global
__device__ __forceinline__ void mm_store_rows(__nv_bfloat16* dst, const __nv_bfloat16* st, int warp, int lane) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = i * 32 + lane;
    const int row = warp * 16 + (q >> 4), ch = q & 15;
    *reinterpret_cast<uint4*>(dst + row * kDimK + ch * 8) =
        *reinterpret_cast<const uint4*>(st + row * kDimK + ((ch ^ (row & 7)) << 3));
  }
}
// split vectors: v[s][n][k], n < nvec from src rows (fp32, stride kDimK); rows >= nvec zero. All threads.
// Rows n >= n0 (and 4 + n >= 4 + n1) are left untouched: they only feed MMA output columns that are ignored.
__device__ __forceinline__ void mm_build_vec(__nv_bfloat16 (*v)[8][kVPad], const float (*src0)[kDimK], int n0,
                                             const float (*src1)[kDimK], int n1, int tid) {
  const int rows = n0 + (src1 != nullptr ? n1 : 0);
  for (int idx = tid; idx < rows * (kDimK / 2); idx += kThreads) {
    const int r = idx >> 6, col = (idx & 63) * 2;
    const int n = r < n0 ? r : 4 + (r - n0);
    const float2 x = r < n0 ? *reinterpret_cast<const float2*>(&src0[r][col])
                            : *reinterpret_cast<const float2*>(&src1[r - n0][col]);
    __nv_bfloat16 a[3], b[3];
    split3(x.x, a);
    split3(x.y, b);
#pragma unroll
    for (int q = 0; q < 3; ++q) *reinterpret_cast<__nv_bfloat162*>(&v[q][n][col]) = __halves2bfloat162(a[q], b[q]);
  }
}
// transposed k table for the replay commit: kt[col][tile*16 + grp*4 + j] = split_{bsplit(tile,grp)}(k_j[col])
__device__ __forceinline__ void mm_build_kt(__nv_bfloat16 (*kt)[kKTPad], const float (*sk)[kDimK], int R, int tid) {
  for (int idx = tid; idx < kDimK * 32; idx += kThreads) {
    const int col = idx >> 5, slot = idx & 31;
    const int tile = slot >> 4, grp = (slot >> 2) & 3, j = slot & 3;
    const int sp = upd_bsplit(tile, grp);
    __nv_bfloat16 val = __float2bfloat16_rn(0.0f);
    if (sp >= 0 && j < R) {
      __nv_bfloat16 s[3];
      split3(sk[j][col], s);
      val = s[sp];
    }
    kt[col][slot] = val;
  }
}
// acc[4] (C layout, rows g / g+8 of the warp tile, n = 2c, 2c+1) = S[rows] . v[n]  (K = 128, 3 split terms)
__device__ __forceinline__ void mm_dots(float (&acc)[4], const __nv_bfloat16* st, const __nv_bfloat16 (*v)[8][kVPad],
                                        int warp, int lane) {
  acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
  const int arow = warp * 16 + (lane & 15);
#pragma unroll
  for (int kt = 0; kt < kDimK / 16; ++kt) {
    uint32_t a[4];
    ldsm_x4(a, st + arow * kDimK + ((((2 * kt + (lane >> 4)) ^ (arow & 7))) << 3));
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      uint32_t b[2];
      ldsm_x2(b, &v[s][lane & 7][kt * 16 + ((lane >> 3) & 1) * 8]);
      mma_bf16(acc, a, b);
    }
  }
}
// Replay R (1..4) tokens on the warp's rows and commit the bf16-rounded state into st in place.
#ifndef GSC_CK2_UPD
#define GSC_CK2_UPD 1  // 1: CUDA-core fp32 commit update (stock-level error); 0: tensor-core update (rejected:
                       //    accumulating small products into the large G_R*S accumulator loses bits)
#endif
__device__ __forceinline__ void mm_replay(__nv_bfloat16* st, const __nv_bfloat16 (*vrep)[8][kVPad],
                                          const __nv_bfloat16 (*kt)[kKTPad], const CkRep& c, int R,
                                          const __nv_bfloat16* sv, int warp, int lane, const float (*skf)[kDimK]) {
  float acc[4];
  mm_dots(acc, st, vrep, warp, lane);
  const int g = lane >> 2, cq = lane & 3, qb = lane & ~3;
  float av[2][kMaxT];
#pragma unroll
  for (int j = 0; j < kMaxT; ++j) {
    av[0][j] = __shfl_sync(0xffffffffu, acc[j & 1], qb + (j >> 1));
    av[1][j] = __shfl_sync(0xffffffffu, acc[2 + (j & 1)], qb + (j >> 1));
  }
  __nv_bfloat16 cs[2][3][kMaxT];  // [half][split][j]
  float ccf[2][kMaxT];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = warp * 16 + g + 8 * h;
    float d[kMaxT];
#pragma unroll
    for (int j = 0; j < kMaxT; ++j) {
      float cc = 0.0f;
      if (j < R) {
        float ks = __fmul_rn(c.G[j], av[h][j]);
#pragma unroll
        for (int i = 0; i < j; ++i) ks = __fmaf_rn(c.W[i][j], d[i], ks);
        d[j] = __fmul_rn(__fsub_rn(__bfloat162float(sv[j * kDimV + row]), ks), c.beta[j]);
        cc = __fmul_rn(c.C[j], d[j]);
      }
#if GSC_CK2_UPD == 0
      __nv_bfloat16 s[3];
      split3(cc, s);
      cs[h][0][j] = s[0]; cs[h][1][j] = s[1]; cs[h][2][j] = s[2];
#endif
      ccf[h][j] = cc;
    }
  }
#if GSC_CK2_UPD == 0
  // A fragments of the two k16 tiles: a0 (row g, slots 2cq..+1), a1 (row g+8), a2 (row g, slots 2cq+8..+9), a3
  uint32_t au[2][4];
  const __nv_bfloat16 zero = __float2bfloat16_rn(0.0f);
#pragma unroll
  for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
    for (int hi = 0; hi < 2; ++hi) {  // slots 2cq (+8*hi)
      const int slot = 2 * cq + 8 * hi;
      const int grp = slot >> 2, j0 = slot & 3;
      const int sp = upd_asplit(tile, grp);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        __nv_bfloat16 lo = zero, hv = zero;
#pragma unroll
        for (int s = 0; s < 3; ++s) {
          if (sp == s) {
#pragma unroll
            for (int jj = 0; jj < kMaxT; jj += 2) {
              if (j0 == jj) { lo = cs[h][s][jj]; hv = cs[h][s][jj + 1]; }
            }
          }
        }
        au[tile][2 * hi + h] = pack_bf16(lo, hv);
      }
    }
  }
#endif
  // commit: per 8-column tile, acc = G_R * S + A.B, round to bf16, write back in place
  const int r0 = warp * 16 + g, r1 = r0 + 8;
#pragma unroll 4
  for (int nt = 0; nt < kDimK / 8; ++nt) {
    const int col = nt * 8 + 2 * cq;
    __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(st + st_idx(r0, col));
    __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(st + st_idx(r1, col));
    const float2 s0 = __bfloat1622float2(*p0), s1 = __bfloat1622float2(*p1);
    float u[4] = {__fmul_rn(s0.x, c.GR), __fmul_rn(s0.y, c.GR), __fmul_rn(s1.x, c.GR), __fmul_rn(s1.y, c.GR)};
#if GSC_CK2_UPD == 1
#pragma unroll
    for (int j = 0; j < kMaxT; ++j) {
      if (j < R) {
        u[0] = __fmaf_rn(ccf[0][j], skf[j][col], u[0]);
        u[1] = __fmaf_rn(ccf[0][j], skf[j][col + 1], u[1]);
        u[2] = __fmaf_rn(ccf[1][j], skf[j][col], u[2]);
        u[3] = __fmaf_rn(ccf[1][j], skf[j][col + 1], u[3]);
      }
    }
#else
#pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
      uint32_t b[2];
      ldsm_x2(b, &kt[nt * 8 + (lane & 7)][tile * 16 + ((lane >> 3) & 1) * 8]);
      mma_bf16(u, au[tile], b);
    }
#endif
    *p0 = __floats2bfloat162_rn(u[0], u[1]);
    *p1 = __floats2bfloat162_rn(u[2], u[3]);
  }
  __syncwarp();
}
// New tokens (T = 1..4): outputs only, sout[t][row] (bf16). vnew rows 0..3 = k_t, 4..7 = q_t.
__device__ __forceinline__ void mm_new(const __nv_bfloat16* st, const __nv_bfloat16 (*vnew)[8][kVPad], const CkNew& c,
                                       int T, const __nv_bfloat16* sv, __nv_bfloat16 (*sout)[kDimV], int warp,
                                       int lane) {
  float acc[4];
  mm_dots(acc, st, vnew, warp, lane);
  const int g = lane >> 2, cq = lane & 3, qb = lane & ~3;
  float av[2][kMaxT], bv[2][kMaxT];
#pragma unroll
  for (int t = 0; t < kMaxT; ++t) {
    av[0][t] = __shfl_sync(0xffffffffu, acc[t & 1], qb + (t >> 1));
    av[1][t] = __shfl_sync(0xffffffffu, acc[2 + (t & 1)], qb + (t >> 1));
    bv[0][t] = __shfl_sync(0xffffffffu, acc[t & 1], qb + 2 + (t >> 1));
    bv[1][t] = __shfl_sync(0xffffffffu, acc[2 + (t & 1)], qb + 2 + (t >> 1));
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = warp * 16 + g + 8 * h;
    float d[kMaxT];
    float mine = 0.0f;
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) {
      if (t < T) {
        float ks = __fmul_rn(c.G[t], av[h][t]);
#pragma unroll
        for (int s = 0; s < t; ++s) ks = __fmaf_rn(c.W[s][t], d[s], ks);
        d[t] = __fmul_rn(__fsub_rn(__bfloat162float(sv[t * kDimV + row]), ks), c.beta[t]);
        float o = __fmul_rn(c.G[t], bv[h][t]);
#pragma unroll
        for (int s = 0; s <= t; ++s) o = __fmaf_rn(c.Q[s][t], d[s], o);
        if (t == cq) mine = o;
      }
    }
    if (cq < T) sout[cq][row] = __float2bfloat16(mine);
  }
}


// One CTA per (request, value head), like stock. The whole 64 KB head state is prefetched with
// 4 cp.async groups (4x the bytes in flight of stock's 2-stage pipeline). The K log is per key head
// and shared by the VPK value-head CTAs of that key head: every CTA reads the old K first, then
// atomically increments a per-(slot, key head) counter; the last reader writes the new K and resets
// the counter (no waiting, no race).
template <typename S, int VPK, bool SigmoidGate>
__global__ __launch_bounds__(kThreads, GSC_MINB) void deferred_decode_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv, const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, int state_indices_width, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, S* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate, const void* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ out, int H, int HV, int dt_bias_type, bool norm_weight_is_bf16, float scale,
    float norm_eps, Strides strides GSC_QO_PARAM) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int key_head = value_head / VPK;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int num_tokens = cu_seqlens[request + 1] - bos;
#if GSC_QO
  if (qo.q != nullptr) qo_zero_pad_rows(qo, cu_seqlens[gridDim.x], request, gridDim.x, value_head, tid);
#endif
  if (num_tokens <= 0) return;
  extern __shared__ __align__(16) unsigned char gsc_dyn_smem[];
  S* shared_state = reinterpret_cast<S*>(gsc_dyn_smem);  // [kNumChunks][kChunkV][kDimK]
  __shared__ __align__(16) float shared_q[kMaxT][kDimK];
  __shared__ __align__(16) float shared_k[kMaxTok][kDimK];
  __shared__ __nv_bfloat16 shared_v[kMaxTok][kDimV];
  __shared__ __nv_bfloat16 shared_out[kMaxT][kDimV];
  __shared__ float shared_decay[kMaxTok];
  __shared__ float shared_beta[kMaxTok];
  __shared__ int s_last;
#if GSC_CK
  static_assert(!GSC_EARLY && !GSC_FAST, "GSC_CK excludes GSC_EARLY / GSC_FAST");
  __shared__ float s_kk[kMaxTok][kMaxTok];
  __shared__ float s_kq[kMaxT][kMaxT];
  __shared__ CkRep s_rep;
  __shared__ CkNew s_new;
#endif
#if GSC_CK >= 2
  constexpr bool kMM = sizeof(S) == 2;  // tensor-core path for the bf16 state; fp32 state uses GSC_CK=1 math
  __shared__ __align__(16) __nv_bfloat16 s_vrep[3][8][kVPad];
  __shared__ __align__(16) __nv_bfloat16 s_vnew[3][8][kVPad];
#if GSC_CK2_UPD == 0
  __shared__ __align__(16) __nv_bfloat16 s_kt[kDimK][kKTPad];
#else
  __nv_bfloat16 (*s_kt)[kKTPad] = nullptr;
#endif
#else
  constexpr bool kMM = false;
#endif
#if GSC_EARLY
  // EARLY: new-token prep does not depend on state_indices / flag, so issue its
  // global loads first (new tokens live at smem index kMaxT + t; replay tokens at j < R, prepped by warps kMaxT+j).
  // Same per-token arithmetic as below (bit-exact); only the load issue order and the smem slots change.
  if (warp < num_tokens && warp < kMaxT) {
    const int t = warp;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    prep_qk(mixed_qkv + mixed_base + key_head * kDimK, mixed_qkv + mixed_base + H * kDimK + key_head * kDimK, scale,
            lane, shared_q[t], shared_k[kMaxT + t]);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      shared_v[kMaxT + t][lane + i * 32] = mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + lane + i * 32];
    if (lane == 0)
      prep_gate(__bfloat162float(a[static_cast<int64_t>(token) * strides.a_row + value_head]),
                __bfloat162float(b[static_cast<int64_t>(token) * strides.b_row + value_head]), a_log[value_head],
                load_dt_bias(dt_bias, value_head, dt_bias_type), &shared_decay[kMaxT + t], &shared_beta[kMaxT + t]);
  }
#endif
  const int base_slot = state_indices[static_cast<int64_t>(request) * state_indices_width];
  if (base_slot <= 0 || num_tokens > kMaxT) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + linear % kDimV] = __float2bfloat16(0.0f);
    }
#if GSC_QO
    if (qo.q != nullptr) {  // zero rows quantize to zero data and zero scales
      for (int linear = tid; linear < num_tokens * (kDimV / 4); linear += kThreads) {
        const int row = bos + linear / (kDimV / 4);
        reinterpret_cast<uint32_t*>(qo.q + row * qo.stride_q + value_head * kDimV)[linear % (kDimV / 4)] = 0u;
      }
      for (int t = tid; t < num_tokens; t += kThreads)
        *reinterpret_cast<uint32_t*>(qo.sf + qo_sf_word(bos + t, value_head, qo.psc)) = 0u;
    }
#endif
    return;
  }
  const LogLayout ll = log_layout(H, HV);
  S* slot_state = state + static_cast<int64_t>(base_slot) * strides.state_slot;
  char* log = reinterpret_cast<char*>(slot_state + static_cast<int64_t>(HV) * kDimV * kDimK);
  __nv_bfloat16* log_k = reinterpret_cast<__nv_bfloat16*>(log + ll.k_off);
  __nv_bfloat16* log_v = reinterpret_cast<__nv_bfloat16*>(log + ll.v_off);
  __nv_bfloat16* log_ab = reinterpret_cast<__nv_bfloat16*>(log + ll.ab_off);
  int* log_flag = reinterpret_cast<int*>(log + ll.l_off) + value_head;
  int* log_ctr = reinterpret_cast<int*>(log + ll.c_off) + key_head;
  const int flag = *log_flag;
  int accepted = num_accepted_tokens[request];
  accepted = accepted < 0 ? 0 : (accepted > kMaxT ? kMaxT : accepted);
  const int R = flag != 0 ? accepted : 0;
  const int total = R + num_tokens;
  (void)total;
#if GSC_EARLY
#define NT(t_) (kMaxT + (t_))
#else
#define NT(t_) (R + (t_))
#endif


  S* head_state = slot_state + static_cast<int64_t>(value_head) * kDimV * kDimK;
#if GSC_PROBE != 2
  if constexpr (kMM) {
    mm_load_rows(reinterpret_cast<__nv_bfloat16*>(shared_state), reinterpret_cast<const __nv_bfloat16*>(head_state),
                 warp, lane);
  } else {
#pragma unroll
    for (int chunk = 0; chunk < kNumChunks; ++chunk)
      copy_state_chunk(shared_state, head_state, chunk, chunk, tid);
  }
#else
  for (int chunk = 0; chunk < kNumChunks; ++chunk) cp_async_commit();
#endif

#if GSC_EARLY
  if (warp >= kMaxT && warp - kMaxT < R) {
    const int j = warp - kMaxT;
    {  // replay token from the log
#else
  if (warp < total) {
    const int j = warp;
    if (j < R) {  // replay token from the log
#endif
      prep_qk(nullptr, log_k + (static_cast<int64_t>(j) * H + key_head) * kDimK, scale, lane, nullptr, shared_k[j]);
      const __nv_bfloat16* vr = log_v + (static_cast<int64_t>(j) * HV + value_head) * kDimV;
#pragma unroll
      for (int i = 0; i < 4; ++i) shared_v[j][lane + i * 32] = vr[lane + i * 32];
      if (lane == 0)
        prep_gate(__bfloat162float(log_ab[(j * HV + value_head) * 2]),
                  __bfloat162float(log_ab[(j * HV + value_head) * 2 + 1]), a_log[value_head],
                  load_dt_bias(dt_bias, value_head, dt_bias_type), &shared_decay[j], &shared_beta[j]);
    }
#if !GSC_EARLY
    else {  // new token
      const int t = j - R;
      const int token = bos + t;
      const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
      prep_qk(mixed_qkv + mixed_base + key_head * kDimK, mixed_qkv + mixed_base + H * kDimK + key_head * kDimK, scale,
              lane, shared_q[t], shared_k[j]);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        shared_v[j][lane + i * 32] = mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + lane + i * 32];
      if (lane == 0)
        prep_gate(__bfloat162float(a[static_cast<int64_t>(token) * strides.a_row + value_head]),
                  __bfloat162float(b[static_cast<int64_t>(token) * strides.b_row + value_head]), a_log[value_head],
                  load_dt_bias(dt_bias, value_head, dt_bias_type), &shared_decay[j], &shared_beta[j]);
    }
#endif
  }
  __syncthreads();  // this CTA's log reads are complete
#if GSC_CK
  ck_gram(shared_k, shared_q, R, num_tokens, tid, s_kk, s_kq);
#endif
#if GSC_CK >= 2
  if constexpr (kMM) {
    mm_build_vec(s_vnew, shared_k + R, num_tokens, shared_q, num_tokens, tid);
    if (R > 0) {
      mm_build_vec(s_vrep, shared_k, R, nullptr, 0, tid);
#if GSC_CK2_UPD == 0
      mm_build_kt(s_kt, shared_k, R, tid);
#endif
    }
  }
#endif
  if (tid == 0) {
    if (VPK == 1) {
      s_last = 1;
    } else {
      __threadfence();
      const int prev = atomicAdd(log_ctr, 1);
      s_last = prev == VPK - 1;
      if (s_last) *log_ctr = 0;
    }
  }
  __syncthreads();
#if GSC_CK
  if (warp == 0) ck_coef_rep_par(shared_decay, shared_beta, s_kk, R, &s_rep, lane);
  if (warp == 1) ck_coef_new_par(shared_decay + R, shared_beta + R, s_kk, R, s_kq, num_tokens, &s_new, lane);
  // visibility: the chunk loop below starts with __syncthreads()
#endif
  if (s_last) {
    __threadfence();
    for (int linear = tid; linear < num_tokens * kDimK; linear += kThreads) {
      const int t = linear / kDimK, d = linear % kDimK;
      log_k[(static_cast<int64_t>(t) * H + key_head) * kDimK + d] =
          mixed_qkv[static_cast<int64_t>(bos + t) * strides.mixed_row + H * kDimK + key_head * kDimK + d];
    }
  }
  for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
    const int t = linear / kDimV, d = linear % kDimV;
    log_v[(static_cast<int64_t>(t) * HV + value_head) * kDimV + d] = shared_v[NT(t)][d];
  }
  if (tid < num_tokens) {
    log_ab[(tid * HV + value_head) * 2] = a[static_cast<int64_t>(bos + tid) * strides.a_row + value_head];
    log_ab[(tid * HV + value_head) * 2 + 1] = b[static_cast<int64_t>(bos + tid) * strides.b_row + value_head];
  }
  if (tid == 0) *log_flag = 1;

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) rows[row] = warp + row * kWarps;

#if GSC_EARLY
  float g_pre[4] = {0.f, 0.f, 0.f, 0.f};
  if (warp < num_tokens) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      g_pre[i] = __bfloat162float(
          output_gate[static_cast<int64_t>(bos + warp) * strides.gate_row + value_head * kDimV + lane + i * 32]);
  }
#endif
  constexpr int NC = GSC_NC;
#if GSC_CK >= 2
  if constexpr (kMM) {
    __syncthreads();  // coefficients, split vectors, k table
    cp_async_wait_group<0>();
    __syncwarp();
    __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(shared_state);
    if (R > 0) {
      mm_replay(st, s_vrep, s_kt, s_rep, R, &shared_v[0][0], warp, lane, shared_k);
      mm_store_rows(reinterpret_cast<__nv_bfloat16*>(head_state), st, warp, lane);
    }
    mm_new(st, s_vnew, s_new, num_tokens, &shared_v[R][0], shared_out, warp, lane);
  } else
#endif
  {
#pragma unroll
  for (int c0 = 0; c0 < kNumChunks; c0 += NC) {
    if (c0 + NC >= kNumChunks) cp_async_wait_group<0>();
    else if (c0 + NC == 1) cp_async_wait_group<3>();
    else if (c0 + NC == 2) cp_async_wait_group<2>();
    else cp_async_wait_group<1>();
    __syncthreads();
#if GSC_CK
    {
      const int seg = lane & 7;
      const int rr = warp * 4 + (lane >> 3);
      int row[NC];
#pragma unroll
      for (int r = 0; r < NC; ++r) row[r] = (c0 + r) * kChunkV + rr;
      float hf[NC][16];
#pragma unroll
      for (int r = 0; r < NC; ++r) {
#pragma unroll
        for (int j = 0; j < 4; ++j) load_h4<S>(&shared_state[row[r] * kDimK + j * 32 + seg * 4], &hf[r][j * 4]);
      }
      if (R > 0) {
        ck_replay<NC>(hf, shared_k, &shared_v[0][0], s_rep, R, row, seg);
#pragma unroll
        for (int r = 0; r < NC; ++r) {
#pragma unroll
          for (int j = 0; j < 4; ++j) store_h4<S>(head_state + row[r] * kDimK + j * 32 + seg * 4, &hf[r][j * 4]);
        }
      }
      ck_new<NC>(hf, shared_k + R, shared_q, &shared_v[R][0], s_new, num_tokens, row, seg, shared_out);
      continue;
    }
#endif
#if GSC_FAST
    {
      const int rr = warp * 4 + (lane >> 3);
      const int seg = lane & 7;
      float hf[NC][16];
#pragma unroll
      for (int r = 0; r < NC; ++r) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          load_h4<S>(&shared_state[((c0 + r) * kChunkV + rr) * kDimK + j * 32 + seg * 4], &hf[r][j * 4]);
      }
      for (int j = 0; j < R; ++j)
        token_update_fast<false, NC>(hf, shared_k[j], nullptr, shared_v[j], shared_decay[j], shared_beta[j], c0, rr,
                                     seg, nullptr);
      if (R > 0) {
#pragma unroll
        for (int r = 0; r < NC; ++r) {
#pragma unroll
          for (int j = 0; j < 4; ++j)
            store_h4<S>(head_state + ((c0 + r) * kChunkV + rr) * kDimK + j * 32 + seg * 4, &hf[r][j * 4]);
        }
      }
      for (int t = 0; t < num_tokens; ++t)
        token_update_fast<true, NC>(hf, shared_k[NT(t)], shared_q[t], shared_v[NT(t)], shared_decay[NT(t)],
                                    shared_beta[NT(t)], c0, rr, seg, shared_out[t]);
      continue;
    }
#endif
    float h[4 * NC][4];
#pragma unroll
    for (int row = 0; row < 4 * NC; ++row) {
      load_h4<S>(&shared_state[((c0 + row / 4) * kChunkV + rows[row % 4]) * kDimK + k_base], h[row]);
    }
#if GSC_PROBE != 1
    for (int j = 0; j < R; ++j)
      token_update_multi<false, NC>(h, shared_k[j], nullptr, shared_v[j], shared_decay[j], shared_beta[j], c0, rows,
                                    k_base, lane, nullptr);
#endif
    if (R > 0 && GSC_PROBE != 2) {
#pragma unroll
      for (int row = 0; row < 4 * NC; ++row) {
        const int value = (c0 + row / 4) * kChunkV + rows[row % 4];
        store_h4<S>(head_state + value * kDimK + k_base, h[row]);
      }
    }
#if GSC_PROBE != 1
    for (int t = 0; t < num_tokens; ++t)
      token_update_multi<true, NC>(h, shared_k[NT(t)], shared_q[t], shared_v[NT(t)], shared_decay[NT(t)],
                                   shared_beta[NT(t)], c0, rows, k_base, lane, shared_out[t]);
#endif
  }
  }  // !kMM
  __syncthreads();
  if (warp < num_tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      output_values[i] = __bfloat162float(shared_out[t][lane + i * 32]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd = rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#if GSC_QO
    float yq[4];
#endif
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
#if GSC_EARLY
      const float gate_input = g_pre[i];
#else
      const float gate_input =
          __bfloat162float(output_gate[static_cast<int64_t>(token) * strides.gate_row + value_head * kDimV + value]);
#endif
      const float gate = SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight = norm_weight_is_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(norm_weight)[value])
                                               : static_cast<const float*>(norm_weight)[value];
      const __nv_bfloat16 yb = __float2bfloat16(output_values[i] * rstd * weight * gate);
      out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + value] = yb;
#if GSC_QO
      yq[i] = __bfloat162float(yb);
#endif
    }
#if GSC_QO
    if (qo.q != nullptr) {  // after the output loop (keeps the loop's register allocation)
      uint32_t sfw = 0u;
#pragma unroll
      for (int i = 0; i < 4; ++i) sfw |= qo_block(qo, token, value_head, lane, i, yq[i]) << (8 * i);
      if (lane == 0) *reinterpret_cast<uint32_t*>(qo.sf + qo_sf_word(token, value_head, qo.psc)) = sfw;
    }
#endif
  }
}

// ---------------------------------------------------------------------------------------------
#undef NT

// ---------------------------------------------------------------------------------------------
// GSC_GB (GB300, fp32 state): deferred-commit decode kernel, BIT-EXACT with GSC_CK=0 (= the stock op's
// per-element arithmetic), restructured for sm_100/sm_103 issue and latency:
//  * row-per-8-lanes mapping: thread (warp, lane) owns ONE value row r = 4m + (lane >> 3) of a 4-row slot m
//    and the 16 k-columns j*32 + seg*4 + e (seg = lane & 7, j, e in 0..3), i.e. the stock column groups
//    g = j*8 + seg (group g = columns 4g..4g+3 = stock lane g's partial). The stock per-row reduction is a
//    sequential FFMA chain over e inside each group, then the xor butterfly over groups with masks 16,8,4,2,1.
//    Masks 16 and 8 pair the groups j <-> j^2 and j <-> j^1 inside the thread ((acc0+acc2) + (acc1+acc3)),
//    masks 4,2,1 are 3 SHFL+FADD inside the 8-lane group, so every row sum is produced by the identical
//    sequence of IEEE adds as stock (a+b == b+a) and every lane of the group ends with it (no broadcast).
//    Per token and 4 rows: 3 SHFL (stock / CK0: 10) and 2+3 FADD.
//  * packed fp32x2 (GB_F2=1): h *= decay (mul.rn.ftz.f32x2), the dot chains of groups (j0,j1) and (j2,j3)
//    (fma.rn.ftz.f32x2), h += k*delta and the q-dot chains; per lane the same IEEE op as the scalar
//    FMUL.FTZ / FFMA.FTZ (--use_fast_math, like vLLM's build). h is kept as pairs (h[2jp][e], h[2jp+1][e]);
//    k / q are stored in smem pre-paired (column d at (d&3)*32 + ((d>>2)&7)*4 + (d>>5)) so their pairs
//    come from LDS.128 directly (conflict-free: 8 segs x 16 B per load).
//  * memory: each warp streams its own row-slots through a per-warp cp.async ring of GB_D passes (no CTA
//    barrier in the token loop; a thread only reads back the 16-B chunks it copied itself), committed state
//    written straight from registers with full-sector STG.128.
//  * prologue: state cp.async first, then fixed warp roles: warps [0, T) prep the new tokens (and prefetch
//    their epilogue gate / norm weight), warps [kMaxT, kMaxT + R) the replay tokens from the log.
//  * K-log handshake (VPK = 2) and log writes as CK0; s_last broadcast inside warp 0 (one CTA barrier less).
// One CTA per (request, value head), GB_W warps; row-slot s = pass * GB_NS + n of warp w covers rows
// 4 (s * GB_W + w) .. +3. materialize is unchanged (CK0 arithmetic), so decode/materialize commits are
// bitwise identical.
// ---------------------------------------------------------------------------------------------
#ifndef GSC_GB
#define GSC_GB 0
#endif
#if GSC_GB
#ifndef GB_W
#define GB_W 8
#endif
#ifndef GB_D
#define GB_D 1
#endif
#ifndef GB_NS
#define GB_NS 2
#endif
#ifndef GB_F2
#define GB_F2 1
#endif
#ifndef GB_MINB
#define GB_MINB 3
#endif
#ifndef GB_KREG
#define GB_KREG 1  // 1: keep this token's 16 k values in registers across the dot and the update
#endif
#ifndef GB_SPEC
#define GB_SPEC 1  // 1: prep the replay tokens j < accepted before the flag is known (speculative log loads)
#endif
#ifndef GB_PROBE
#define GB_PROBE 0  // timing-only ablations (WRONG results): 1 = no token math, 2 = no state load/store, 3 = neither
#endif
#define GB_NOMATH (GB_PROBE == 1 || GB_PROBE == 3)
#define GB_NOIO (GB_PROBE == 2 || GB_PROBE == 3)
constexpr int kGbW = GB_W;
constexpr int kGbThreads = kGbW * 32;
constexpr int kGbNS = GB_NS;
constexpr int kGbSlots = kDimV / (4 * kGbW);
static_assert(kDimV % (4 * kGbW) == 0, "GB_W must divide 32");
static_assert(kGbSlots % kGbNS == 0, "GB_NS must divide the row-slots per warp");
static_assert(kGbW >= kMaxT, "GB_W >= kMaxT: warp t preps new token t and runs its epilogue");
constexpr int kGbPasses = kGbSlots / kGbNS;
constexpr int kGbD = GB_D < kGbPasses ? GB_D : kGbPasses;
constexpr int kGbPassFloats = kGbNS * 4 * kDimK;
constexpr int kGbDyn = kGbW * kGbD * kGbPassFloats * 4;

#if GB_F2
typedef unsigned long long gb2;
__device__ __forceinline__ gb2 gb_pack(float a, float b) { return p2(a, b); }
__device__ __forceinline__ float gb_lo(gb2 v) { return __uint_as_float(static_cast<unsigned>(v)); }
__device__ __forceinline__ float gb_hi(gb2 v) { return __uint_as_float(static_cast<unsigned>(v >> 32)); }
__device__ __forceinline__ gb2 gb_mul(gb2 a, gb2 b) { return fmul2(a, b); }
__device__ __forceinline__ gb2 gb_fma(gb2 a, gb2 b, gb2 c) { return ffma2(a, b, c); }
__device__ __forceinline__ gb2 gb_add(gb2 a, gb2 b) {
  gb2 d;
  asm("add.rn.ftz.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}
#else
struct gb2 { float x, y; };
__device__ __forceinline__ gb2 gb_pack(float a, float b) { return gb2{a, b}; }
__device__ __forceinline__ float gb_lo(gb2 v) { return v.x; }
__device__ __forceinline__ float gb_hi(gb2 v) { return v.y; }
__device__ __forceinline__ gb2 gb_mul(gb2 a, gb2 b) { return gb2{__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y)}; }
__device__ __forceinline__ gb2 gb_fma(gb2 a, gb2 b, gb2 c) {
  return gb2{__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y)};
}
__device__ __forceinline__ gb2 gb_add(gb2 a, gb2 b) { return gb2{__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y)}; }
#endif

__device__ __forceinline__ int gb_perm(int d) { return (d & 3) * 32 + ((d >> 2) & 7) * 4 + (d >> 5); }

__device__ __forceinline__ void gb_cp_async_16b(void* smem_ptr, const void* gmem_ptr) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_ptr) : "memory");
}

// cp.async of this thread's 16 floats of each row-slot of pass p into buf (natural [slot][4 rows][128] image)
__device__ __forceinline__ void gb_issue_pass(float* buf, const float* head_state, int p, int warp, int rq, int seg) {
#pragma unroll
  for (int n = 0; n < kGbNS; ++n) {
    const int m = (p * kGbNS + n) * kGbW + warp;
    const float* src = head_state + (4 * m + rq) * kDimK + seg * 4;
    float* dst = buf + n * (4 * kDimK) + rq * kDimK + seg * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) gb_cp_async_16b(dst + j * 32, src + j * 32);
  }
}

// one token of the delta rule on this thread's NS rows x 16 columns (stock per-element op order)
template <bool kOut>
__device__ __forceinline__ void gb_token(gb2 (&hp)[kGbNS][4][2], const float* skp, const float* sqp,
                                         const __nv_bfloat16* sv, float decay, float beta, const int (&row)[kGbNS],
                                         int seg, __nv_bfloat16* sout) {
  gb2 kp[4][2];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float4 k4 = *reinterpret_cast<const float4*>(&skp[e * 32 + seg * 4]);
    kp[e][0] = gb_pack(k4.x, k4.y);
    kp[e][1] = gb_pack(k4.z, k4.w);
  }
  const gb2 dd = gb_pack(decay, decay);
  float dot[kGbNS];
#pragma unroll
  for (int n = 0; n < kGbNS; ++n) {
    gb2 a0 = gb_pack(0.0f, 0.0f), a1 = gb_pack(0.0f, 0.0f);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      hp[n][e][0] = gb_mul(hp[n][e][0], dd);
      hp[n][e][1] = gb_mul(hp[n][e][1], dd);
      a0 = gb_fma(hp[n][e][0], kp[e][0], a0);
      a1 = gb_fma(hp[n][e][1], kp[e][1], a1);
    }
    const gb2 s1 = gb_add(a0, a1);  // groups (g + g^16) for j = 0 and j = 1
    dot[n] = gb_lo(s1) + gb_hi(s1);  // + g^8
  }
#pragma unroll
  for (int m = 4; m >= 1; m >>= 1) {
#pragma unroll
    for (int n = 0; n < kGbNS; ++n) dot[n] += __shfl_xor_sync(0xffffffffu, dot[n], m);
  }
#if !GB_KREG
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float4 k4 = *reinterpret_cast<const float4*>(&skp[e * 32 + seg * 4]);
    kp[e][0] = gb_pack(k4.x, k4.y);
    kp[e][1] = gb_pack(k4.z, k4.w);
  }
#endif
#pragma unroll
  for (int n = 0; n < kGbNS; ++n) {
    const float delta = (__bfloat162float(sv[row[n]]) - dot[n]) * beta;
    const gb2 dl = gb_pack(delta, delta);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      hp[n][e][0] = gb_fma(kp[e][0], dl, hp[n][e][0]);
      hp[n][e][1] = gb_fma(kp[e][1], dl, hp[n][e][1]);
    }
  }
  if constexpr (kOut) {
    gb2 qp[4][2];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float4 q4 = *reinterpret_cast<const float4*>(&sqp[e * 32 + seg * 4]);
      qp[e][0] = gb_pack(q4.x, q4.y);
      qp[e][1] = gb_pack(q4.z, q4.w);
    }
    float o[kGbNS];
#pragma unroll
    for (int n = 0; n < kGbNS; ++n) {
      gb2 a0 = gb_pack(0.0f, 0.0f), a1 = gb_pack(0.0f, 0.0f);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        a0 = gb_fma(hp[n][e][0], qp[e][0], a0);
        a1 = gb_fma(hp[n][e][1], qp[e][1], a1);
      }
      const gb2 s1 = gb_add(a0, a1);
      o[n] = gb_lo(s1) + gb_hi(s1);
    }
#pragma unroll
    for (int m = 4; m >= 1; m >>= 1) {
#pragma unroll
      for (int n = 0; n < kGbNS; ++n) o[n] += __shfl_xor_sync(0xffffffffu, o[n], m);
    }
    if (seg == 0) {
#pragma unroll
      for (int n = 0; n < kGbNS; ++n) sout[row[n]] = __float2bfloat16(o[n]);
    }
  }
}

template <int VPK, bool SigmoidGate>
__global__ __launch_bounds__(kGbThreads, GB_MINB) void gb_decode_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv, const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, int state_indices_width, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, float* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate, const void* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ out, int H, int HV, int dt_bias_type, bool norm_weight_is_bf16, float scale,
    float norm_eps, Strides strides) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int key_head = value_head / VPK;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int seg = lane & 7;
  const int rq = lane >> 3;
  const int bos = cu_seqlens[request];
  const int num_tokens = cu_seqlens[request + 1] - bos;
  if (num_tokens <= 0) return;
  extern __shared__ __align__(16) unsigned char gb_dyn_smem[];
  float* ring = reinterpret_cast<float*>(gb_dyn_smem) + warp * (kGbD * kGbPassFloats);
  __shared__ __align__(16) float s_kp[kMaxTok][kDimK];  // replay j at j, new token t at kMaxT + t (paired order)
  __shared__ __align__(16) float s_qp[kMaxT][kDimK];
  __shared__ __align__(16) __nv_bfloat16 s_v[kMaxTok][kDimV];
  __shared__ __align__(16) __nv_bfloat16 s_out[kMaxT][kDimV];
  __shared__ float s_decay[kMaxTok];
  __shared__ float s_beta[kMaxTok];
  const int base_slot = state_indices[static_cast<int64_t>(request) * state_indices_width];
  if (base_slot <= 0 || num_tokens > kMaxT) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kGbThreads) {
      const int token = bos + linear / kDimV;
      out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + linear % kDimV] = __float2bfloat16(0.0f);
    }
    return;
  }
  const LogLayout ll = log_layout(H, HV);
  float* slot_state = state + static_cast<int64_t>(base_slot) * strides.state_slot;
  char* log = reinterpret_cast<char*>(slot_state + static_cast<int64_t>(HV) * kDimV * kDimK);
  __nv_bfloat16* log_k = reinterpret_cast<__nv_bfloat16*>(log + ll.k_off);
  __nv_bfloat16* log_v = reinterpret_cast<__nv_bfloat16*>(log + ll.v_off);
  __nv_bfloat16* log_ab = reinterpret_cast<__nv_bfloat16*>(log + ll.ab_off);
  int* log_flag = reinterpret_cast<int*>(log + ll.l_off) + value_head;
  int* log_ctr = reinterpret_cast<int*>(log + ll.c_off) + key_head;
  float* head_state = slot_state + static_cast<int64_t>(value_head) * kDimV * kDimK;
  // (1) state: the first kGbD passes of this warp (one cp.async group per pass)
#pragma unroll
  for (int d = 0; d < kGbD; ++d) {
    if (!GB_NOIO) gb_issue_pass(ring + d * kGbPassFloats, head_state, d, warp, rq, seg);
    cp_async_commit();
  }
  const int flag = *log_flag;
  int accepted = num_accepted_tokens[request];
  accepted = accepted < 0 ? 0 : (accepted > kMaxT ? kMaxT : accepted);
  const int R = flag != 0 ? accepted : 0;
  const int Rprep = GB_SPEC ? accepted : R;  // replay tokens prepped (>= R; extra ones are never used)
  // (2) token prep (stock arithmetic; k / q stored in the paired order)
  float g_pre[4] = {0.f, 0.f, 0.f, 0.f};
  float w_pre[4] = {0.f, 0.f, 0.f, 0.f};
  for (int task = warp; task < 2 * kMaxT; task += kGbW) {
  if (task < num_tokens) {  // new token t = task
    const int t = task;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float qv[4], kv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      qv[i] = __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      kv[i] = __bfloat162float(mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      s_v[kMaxT + t][dim] = mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      if (task == warp) {  // this warp also runs token t's epilogue
        g_pre[i] = __bfloat162float(output_gate[static_cast<int64_t>(token) * strides.gate_row + value_head * kDimV + dim]);
        w_pre[i] = norm_weight_is_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(norm_weight)[dim])
                                       : static_cast<const float*>(norm_weight)[dim];
      }
    }
    float q_square = 0.0f, k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q_square += qv[i] * qv[i];
      k_square += kv[i] * kv[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      s_qp[t][gb_perm(dim)] = qv[i] * q_scale;
      s_kp[kMaxT + t][gb_perm(dim)] = kv[i] * k_scale;
    }
    if (lane == 0)
      prep_gate(__bfloat162float(a[static_cast<int64_t>(token) * strides.a_row + value_head]),
                __bfloat162float(b[static_cast<int64_t>(token) * strides.b_row + value_head]), a_log[value_head],
                load_dt_bias(dt_bias, value_head, dt_bias_type), &s_decay[kMaxT + t], &s_beta[kMaxT + t]);
  } else if (task >= kMaxT && task - kMaxT < Rprep) {  // replay token j from the log
    const int j = task - kMaxT;
    const __nv_bfloat16* kr = log_k + (static_cast<int64_t>(j) * H + key_head) * kDimK;
    const __nv_bfloat16* vr = log_v + (static_cast<int64_t>(j) * HV + value_head) * kDimV;
    float kv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      kv[i] = __bfloat162float(kr[lane + i * 32]);
      s_v[j][lane + i * 32] = vr[lane + i * 32];
    }
    float q_square = 0.0f, k_square = 0.0f;  // stock prep_qk with q = 0
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q_square += 0.0f * 0.0f;
      k_square += kv[i] * kv[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float k_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) s_kp[j][gb_perm(lane + i * 32)] = kv[i] * k_scale;
    if (lane == 0)
      prep_gate(__bfloat162float(log_ab[(j * HV + value_head) * 2]), __bfloat162float(log_ab[(j * HV + value_head) * 2 + 1]),
                a_log[value_head], load_dt_bias(dt_bias, value_head, dt_bias_type), &s_decay[j], &s_beta[j]);
  }
  }
  __syncthreads();  // token tables ready; this CTA's log reads are complete
  // (3) log update (CK0 protocol): K log written by the last of the VPK readers of this key head
  if (warp == 0) {
    int last = 1;
    if (VPK > 1) {
      int prev = 0;
      if (lane == 0) {
        __threadfence();
        prev = atomicAdd(log_ctr, 1);
        if (prev == VPK - 1) *log_ctr = 0;
      }
      last = __shfl_sync(0xffffffffu, prev, 0) == VPK - 1;
    }
    if (last) {
      __threadfence();
      for (int linear = lane; linear < num_tokens * kDimK; linear += 32) {
        const int t = linear / kDimK, d = linear % kDimK;
        log_k[(static_cast<int64_t>(t) * H + key_head) * kDimK + d] =
            mixed_qkv[static_cast<int64_t>(bos + t) * strides.mixed_row + H * kDimK + key_head * kDimK + d];
      }
    }
  }
  for (int linear = tid; linear < num_tokens * kDimV; linear += kGbThreads) {
    const int t = linear / kDimV, d = linear % kDimV;
    log_v[(static_cast<int64_t>(t) * HV + value_head) * kDimV + d] = s_v[kMaxT + t][d];
  }
  if (tid < num_tokens) {
    log_ab[(tid * HV + value_head) * 2] = a[static_cast<int64_t>(bos + tid) * strides.a_row + value_head];
    log_ab[(tid * HV + value_head) * 2 + 1] = b[static_cast<int64_t>(bos + tid) * strides.b_row + value_head];
  }
  if (tid == 0) *log_flag = 1;
  // (4) this warp's row-slots: replay R logged tokens, commit, then the new tokens (outputs only)
#pragma unroll
  for (int p = 0; p < kGbPasses; ++p) {
    cp_async_wait_group<kGbD - 1>();
    const float* buf = ring + (p % kGbD) * kGbPassFloats;
    gb2 hp[kGbNS][4][2];
    int row[kGbNS];
#pragma unroll
    for (int n = 0; n < kGbNS; ++n) {
      row[n] = 4 * ((p * kGbNS + n) * kGbW + warp) + rq;
      float4 x[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        x[j] = *reinterpret_cast<const float4*>(buf + n * (4 * kDimK) + rq * kDimK + j * 32 + seg * 4);
      hp[n][0][0] = gb_pack(x[0].x, x[1].x); hp[n][0][1] = gb_pack(x[2].x, x[3].x);
      hp[n][1][0] = gb_pack(x[0].y, x[1].y); hp[n][1][1] = gb_pack(x[2].y, x[3].y);
      hp[n][2][0] = gb_pack(x[0].z, x[1].z); hp[n][2][1] = gb_pack(x[2].z, x[3].z);
      hp[n][3][0] = gb_pack(x[0].w, x[1].w); hp[n][3][1] = gb_pack(x[2].w, x[3].w);
    }
    if (!GB_NOMATH)
      for (int j = 0; j < R; ++j)
        gb_token<false>(hp, s_kp[j], nullptr, s_v[j], s_decay[j], s_beta[j], row, seg, nullptr);
    if (R > 0 && !GB_NOIO) {
#pragma unroll
      for (int n = 0; n < kGbNS; ++n) {
        float* dst = head_state + row[n] * kDimK + seg * 4;
#pragma unroll
        for (int jp = 0; jp < 2; ++jp) {
          *reinterpret_cast<float4*>(dst + (2 * jp) * 32) =
              make_float4(gb_lo(hp[n][0][jp]), gb_lo(hp[n][1][jp]), gb_lo(hp[n][2][jp]), gb_lo(hp[n][3][jp]));
          *reinterpret_cast<float4*>(dst + (2 * jp + 1) * 32) =
              make_float4(gb_hi(hp[n][0][jp]), gb_hi(hp[n][1][jp]), gb_hi(hp[n][2][jp]), gb_hi(hp[n][3][jp]));
        }
      }
    }
    // refill this ring buffer with pass p + kGbD (its data is in registers); always commit one group per pass
    if (p + kGbD < kGbPasses && !GB_NOIO)
      gb_issue_pass(ring + (p % kGbD) * kGbPassFloats, head_state, p + kGbD, warp, rq, seg);
    cp_async_commit();
    if (!GB_NOMATH)
    for (int t = 0; t < num_tokens; ++t)
      gb_token<true>(hp, s_kp[kMaxT + t], s_qp[t], s_v[kMaxT + t], s_decay[kMaxT + t], s_beta[kMaxT + t], row, seg,
                     s_out[t]);
  }
  __syncthreads();
  // (5) gated RMSNorm epilogue (stock arithmetic)
  if (warp < num_tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      output_values[i] = __bfloat162float(s_out[t][lane + i * 32]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd = rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // g_pre / w_pre: prefetched by this warp (its first prep task is token t = warp)
      const int value = lane + i * 32;
      const float gate = SigmoidGate ? sigmoid_fast(g_pre[i]) : silu_fast(g_pre[i]);
      out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + value] =
          __float2bfloat16(output_values[i] * rstd * w_pre[i] * gate);
    }
  }
}
#endif  // GSC_GB

#if GSC_GB
#ifndef GB_KH
#define GB_KH 0  // 1: one CTA per (request, KEY head) runs both value heads (VPK = 2): shared q/k prep, no K-log atomic
#endif           // 2: auto - key-head CTAs when the call has >= GB_KH_MIN requests (half the CTAs; wins at large N)
#ifndef GB_KH_MIN
#define GB_KH_MIN 48
#endif
#if GB_KH
// slot n of a pass = value head vh0 + n (same rows); per-slot decay / beta / v / out; act[n] = slot takes this token
template <bool kOut>
__device__ __forceinline__ void gbk_token(gb2 (&hp)[2][4][2], const float* skp, const float* sqp,
                                          const __nv_bfloat16* sv0, const __nv_bfloat16* sv1, float d0, float d1,
                                          float b0, float b1, bool act0, bool act1, int row, int seg,
                                          __nv_bfloat16* so0, __nv_bfloat16* so1) {
  gb2 kp[4][2];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float4 k4 = *reinterpret_cast<const float4*>(&skp[e * 32 + seg * 4]);
    kp[e][0] = gb_pack(k4.x, k4.y);
    kp[e][1] = gb_pack(k4.z, k4.w);
  }
  const gb2 dd[2] = {gb_pack(d0, d0), gb_pack(d1, d1)};
  const bool act[2] = {act0, act1};
  float dot[2];
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    if (!act[n]) { dot[n] = 0.0f; continue; }  // warp-uniform
    gb2 a0 = gb_pack(0.0f, 0.0f), a1 = gb_pack(0.0f, 0.0f);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      hp[n][e][0] = gb_mul(hp[n][e][0], dd[n]);
      hp[n][e][1] = gb_mul(hp[n][e][1], dd[n]);
      a0 = gb_fma(hp[n][e][0], kp[e][0], a0);
      a1 = gb_fma(hp[n][e][1], kp[e][1], a1);
    }
    const gb2 s1 = gb_add(a0, a1);
    dot[n] = gb_lo(s1) + gb_hi(s1);
  }
#pragma unroll
  for (int m = 4; m >= 1; m >>= 1) {
#pragma unroll
    for (int n = 0; n < 2; ++n) dot[n] += __shfl_xor_sync(0xffffffffu, dot[n], m);
  }
  const float bt[2] = {b0, b1};
  const __nv_bfloat16* sv[2] = {sv0, sv1};
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    if (!act[n]) continue;
    const float delta = (__bfloat162float(sv[n][row]) - dot[n]) * bt[n];
    const gb2 dl = gb_pack(delta, delta);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      hp[n][e][0] = gb_fma(kp[e][0], dl, hp[n][e][0]);
      hp[n][e][1] = gb_fma(kp[e][1], dl, hp[n][e][1]);
    }
  }
  if constexpr (kOut) {  // new tokens: both slots always active
    gb2 qp[4][2];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float4 q4 = *reinterpret_cast<const float4*>(&sqp[e * 32 + seg * 4]);
      qp[e][0] = gb_pack(q4.x, q4.y);
      qp[e][1] = gb_pack(q4.z, q4.w);
    }
    float o[2];
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      gb2 a0 = gb_pack(0.0f, 0.0f), a1 = gb_pack(0.0f, 0.0f);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        a0 = gb_fma(hp[n][e][0], qp[e][0], a0);
        a1 = gb_fma(hp[n][e][1], qp[e][1], a1);
      }
      const gb2 s1 = gb_add(a0, a1);
      o[n] = gb_lo(s1) + gb_hi(s1);
    }
#pragma unroll
    for (int m = 4; m >= 1; m >>= 1) {
#pragma unroll
      for (int n = 0; n < 2; ++n) o[n] += __shfl_xor_sync(0xffffffffu, o[n], m);
    }
    if (seg == 0) {
      so0[row] = __float2bfloat16(o[0]);
      so1[row] = __float2bfloat16(o[1]);
    }
  }
}

constexpr int kGbkPasses = kDimV / (4 * kGbW);  // passes per warp (one 4-row slot per value head each)
constexpr int kGbkRing = GB_D >= 2 ? 2 : 1;  // passes staged per warp (1: refill after the replay, as gb_decode_kernel)
constexpr int kGbkDyn = kGbW * kGbkRing * (2 * 4 * kDimK) * 4;  // ring x (2 heads x 2 KB) per warp

__device__ __forceinline__ void gbk_issue_pass(float* buf, const float* hs0, const float* hs1, int p, int warp,
                                               int rq, int seg) {
  const int m = p * kGbW + warp;
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const float* src = (n == 0 ? hs0 : hs1) + (4 * m + rq) * kDimK + seg * 4;
    float* dst = buf + n * (4 * kDimK) + rq * kDimK + seg * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) gb_cp_async_16b(dst + j * 32, src + j * 32);
  }
}

template <bool SigmoidGate>
__global__ __launch_bounds__(kGbThreads, GB_MINB) void gb_kh_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv, const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, int state_indices_width, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, float* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate, const void* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ out, int H, int HV, int dt_bias_type, bool norm_weight_is_bf16, float scale,
    float norm_eps, Strides strides) {
  const int request = blockIdx.x;
  const int key_head = blockIdx.y;
  const int vh0 = key_head * 2;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int seg = lane & 7;
  const int rq = lane >> 3;
  const int bos = cu_seqlens[request];
  const int num_tokens = cu_seqlens[request + 1] - bos;
  if (num_tokens <= 0) return;
  extern __shared__ __align__(16) unsigned char gbk_dyn_smem[];
  float* ring = reinterpret_cast<float*>(gbk_dyn_smem) + warp * (kGbkRing * 2 * 4 * kDimK);
  __shared__ __align__(16) float s_kp[kMaxTok][kDimK];
  __shared__ __align__(16) float s_qp[kMaxT][kDimK];
  __shared__ __align__(16) __nv_bfloat16 s_v[2][kMaxTok][kDimV];
  __shared__ __align__(16) __nv_bfloat16 s_out[2][kMaxT][kDimV];
  __shared__ float s_decay[2][kMaxTok];
  __shared__ float s_beta[2][kMaxTok];
  const int base_slot = state_indices[static_cast<int64_t>(request) * state_indices_width];
  if (base_slot <= 0 || num_tokens > kMaxT) {
    for (int linear = tid; linear < num_tokens * 2 * kDimV; linear += kGbThreads) {
      const int token = bos + linear / (2 * kDimV);
      const int r = linear % (2 * kDimV);
      out[(static_cast<int64_t>(token) * HV + vh0 + r / kDimV) * kDimV + r % kDimV] = __float2bfloat16(0.0f);
    }
    return;
  }
  const LogLayout ll = log_layout(H, HV);
  float* slot_state = state + static_cast<int64_t>(base_slot) * strides.state_slot;
  char* log = reinterpret_cast<char*>(slot_state + static_cast<int64_t>(HV) * kDimV * kDimK);
  __nv_bfloat16* log_k = reinterpret_cast<__nv_bfloat16*>(log + ll.k_off);
  __nv_bfloat16* log_v = reinterpret_cast<__nv_bfloat16*>(log + ll.v_off);
  __nv_bfloat16* log_ab = reinterpret_cast<__nv_bfloat16*>(log + ll.ab_off);
  int* log_flag = reinterpret_cast<int*>(log + ll.l_off) + vh0;
  const float* hs0 = slot_state + static_cast<int64_t>(vh0) * kDimV * kDimK;
  const float* hs1 = hs0 + kDimV * kDimK;
  if (!GB_NOIO) gbk_issue_pass(ring, hs0, hs1, 0, warp, rq, seg);
  cp_async_commit();
  const int flag0 = log_flag[0], flag1 = log_flag[1];
  int accepted = num_accepted_tokens[request];
  accepted = accepted < 0 ? 0 : (accepted > kMaxT ? kMaxT : accepted);
  const int R0 = flag0 != 0 ? accepted : 0, R1 = flag1 != 0 ? accepted : 0;
  const int Rmax = R0 > R1 ? R0 : R1;
  // epilogue task of this warp: value head vh0 + (warp >> 2), token warp & 3 (prefetch gate + norm weight)
  const int et = warp & 3, eh = vh0 + ((warp >> 2) & 1);
  const bool has_epi = warp < 2 * kMaxT && et < num_tokens;
  float g_pre[4] = {0.f, 0.f, 0.f, 0.f};
  float w_pre[4] = {0.f, 0.f, 0.f, 0.f};
  if (has_epi) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      g_pre[i] = __bfloat162float(output_gate[static_cast<int64_t>(bos + et) * strides.gate_row + eh * kDimV + dim]);
      w_pre[i] = norm_weight_is_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(norm_weight)[dim])
                                     : static_cast<const float*>(norm_weight)[dim];
    }
  }
  for (int task = warp; task < 2 * kMaxT; task += kGbW) {
    if (task < num_tokens) {  // new token t: q / k of the key head, v / a / b of both value heads
      const int t = task;
      const int token = bos + t;
      const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
      float qv[4], kv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int dim = lane + i * 32;
        qv[i] = __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
        kv[i] = __bfloat162float(mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
        s_v[0][kMaxT + t][dim] = mixed_qkv[mixed_base + 2 * H * kDimK + vh0 * kDimV + dim];
        s_v[1][kMaxT + t][dim] = mixed_qkv[mixed_base + 2 * H * kDimK + (vh0 + 1) * kDimV + dim];
      }
      float q_square = 0.0f, k_square = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        q_square += qv[i] * qv[i];
        k_square += kv[i] * kv[i];
      }
      const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
      const float q_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
      const float k_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int dim = lane + i * 32;
        s_qp[t][gb_perm(dim)] = qv[i] * q_scale;
        s_kp[kMaxT + t][gb_perm(dim)] = kv[i] * k_scale;
      }
      if (lane < 2) {
        const int vh = vh0 + lane;
        prep_gate(__bfloat162float(a[static_cast<int64_t>(token) * strides.a_row + vh]),
                  __bfloat162float(b[static_cast<int64_t>(token) * strides.b_row + vh]), a_log[vh],
                  load_dt_bias(dt_bias, vh, dt_bias_type), &s_decay[lane][kMaxT + t], &s_beta[lane][kMaxT + t]);
      }
    } else if (task >= kMaxT && task - kMaxT < (GB_SPEC ? accepted : Rmax)) {  // replay token j from the log
      const int j = task - kMaxT;
      const __nv_bfloat16* kr = log_k + (static_cast<int64_t>(j) * H + key_head) * kDimK;
      const __nv_bfloat16* vr = log_v + (static_cast<int64_t>(j) * HV + vh0) * kDimV;
      float kv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        kv[i] = __bfloat162float(kr[lane + i * 32]);
        s_v[0][j][lane + i * 32] = vr[lane + i * 32];
        s_v[1][j][lane + i * 32] = vr[kDimV + lane + i * 32];
      }
      float q_square = 0.0f, k_square = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        q_square += 0.0f * 0.0f;
        k_square += kv[i] * kv[i];
      }
      const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
      const float k_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
      for (int i = 0; i < 4; ++i) s_kp[j][gb_perm(lane + i * 32)] = kv[i] * k_scale;
      if (lane < 2) {
        const int vh = vh0 + lane;
        prep_gate(__bfloat162float(log_ab[(j * HV + vh) * 2]), __bfloat162float(log_ab[(j * HV + vh) * 2 + 1]),
                  a_log[vh], load_dt_bias(dt_bias, vh, dt_bias_type), &s_decay[lane][j], &s_beta[lane][j]);
      }
    }
  }
  __syncthreads();  // token tables ready; the log reads of this (request, key head) are complete
  // log update: this CTA owns the key head (no counter handshake)
  for (int linear = tid; linear < num_tokens * kDimK; linear += kGbThreads) {
    const int t = linear / kDimK, d = linear % kDimK;
    log_k[(static_cast<int64_t>(t) * H + key_head) * kDimK + d] =
        mixed_qkv[static_cast<int64_t>(bos + t) * strides.mixed_row + H * kDimK + key_head * kDimK + d];
  }
  for (int linear = tid; linear < num_tokens * 2 * kDimV; linear += kGbThreads) {
    const int t = linear / (2 * kDimV), r = linear % (2 * kDimV);
    log_v[(static_cast<int64_t>(t) * HV + vh0) * kDimV + r] = s_v[r / kDimV][kMaxT + t][r % kDimV];
  }
  if (tid < 2 * num_tokens) {
    const int t = tid >> 1, vh = vh0 + (tid & 1);
    log_ab[(t * HV + vh) * 2] = a[static_cast<int64_t>(bos + t) * strides.a_row + vh];
    log_ab[(t * HV + vh) * 2 + 1] = b[static_cast<int64_t>(bos + t) * strides.b_row + vh];
  }
  if (tid < 2) log_flag[tid] = 1;
  float* hsw[2] = {slot_state + static_cast<int64_t>(vh0) * kDimV * kDimK,
                   slot_state + static_cast<int64_t>(vh0 + 1) * kDimV * kDimK};
#pragma unroll
  for (int p = 0; p < kGbkPasses; ++p) {
    cp_async_wait_group<0>();
    const float* buf = ring + (p % kGbkRing) * (2 * 4 * kDimK);
    const int row = 4 * (p * kGbW + warp) + rq;
    gb2 hp[2][4][2];
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      float4 x[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        x[j] = *reinterpret_cast<const float4*>(buf + n * (4 * kDimK) + rq * kDimK + j * 32 + seg * 4);
      hp[n][0][0] = gb_pack(x[0].x, x[1].x); hp[n][0][1] = gb_pack(x[2].x, x[3].x);
      hp[n][1][0] = gb_pack(x[0].y, x[1].y); hp[n][1][1] = gb_pack(x[2].y, x[3].y);
      hp[n][2][0] = gb_pack(x[0].z, x[1].z); hp[n][2][1] = gb_pack(x[2].z, x[3].z);
      hp[n][3][0] = gb_pack(x[0].w, x[1].w); hp[n][3][1] = gb_pack(x[2].w, x[3].w);
    }
    // ring of 2: next pass into the other buffer (consumed one pass ago), issued before the replay
    if (kGbkRing == 2) {
      if (p + 1 < kGbkPasses && !GB_NOIO) gbk_issue_pass(ring + ((p + 1) & 1) * (2 * 4 * kDimK), hs0, hs1, p + 1, warp, rq, seg);
      cp_async_commit();
    }
    if (!GB_NOMATH)
      for (int j = 0; j < Rmax; ++j)
        gbk_token<false>(hp, s_kp[j], nullptr, s_v[0][j], s_v[1][j], s_decay[0][j], s_decay[1][j], s_beta[0][j],
                         s_beta[1][j], j < R0, j < R1, row, seg, nullptr, nullptr);
    if (!GB_NOIO) {
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        if ((n == 0 ? R0 : R1) <= 0) continue;
        float* dst = hsw[n] + row * kDimK + seg * 4;
#pragma unroll
        for (int jp = 0; jp < 2; ++jp) {
          *reinterpret_cast<float4*>(dst + (2 * jp) * 32) =
              make_float4(gb_lo(hp[n][0][jp]), gb_lo(hp[n][1][jp]), gb_lo(hp[n][2][jp]), gb_lo(hp[n][3][jp]));
          *reinterpret_cast<float4*>(dst + (2 * jp + 1) * 32) =
              make_float4(gb_hi(hp[n][0][jp]), gb_hi(hp[n][1][jp]), gb_hi(hp[n][2][jp]), gb_hi(hp[n][3][jp]));
        }
      }
    }
    if (kGbkRing == 1) {  // ring of 1: refill this buffer (now in registers) after the replay / commit
      if (p + 1 < kGbkPasses && !GB_NOIO) gbk_issue_pass(ring, hs0, hs1, p + 1, warp, rq, seg);
      cp_async_commit();
    }
    if (!GB_NOMATH)
      for (int t = 0; t < num_tokens; ++t)
        gbk_token<true>(hp, s_kp[kMaxT + t], s_qp[t], s_v[0][kMaxT + t], s_v[1][kMaxT + t], s_decay[0][kMaxT + t],
                        s_decay[1][kMaxT + t], s_beta[0][kMaxT + t], s_beta[1][kMaxT + t], true, true, row, seg,
                        s_out[0][t], s_out[1][t]);
  }
  __syncthreads();
  if (has_epi) {
    const int n = (warp >> 2) & 1;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      output_values[i] = __bfloat162float(s_out[n][et][lane + i * 32]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd = rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate = SigmoidGate ? sigmoid_fast(g_pre[i]) : silu_fast(g_pre[i]);
      out[(static_cast<int64_t>(bos + et) * HV + eh) * kDimV + value] =
          __float2bfloat16(output_values[i] * rstd * w_pre[i] * gate);
    }
  }
}
#endif  // GB_KH
#endif  // GSC_GB

// ---------------------------------------------------------------------------------------------
// GSC_CK == 3 (bf16 state): GSC_CK == 2 math, bit-identical per element (same fp32 op sequence as
// mm_replay / mm_new / ck_gram / ck_coef_*_par / prep_*), restructured for issue and latency:
//  * prologue: all metadata loads before any branch; state cp.async + log/new-token loads in one round;
//    fixed warp roles (warps 0-3: replay log token j < acc, speculative on the flag; warps 4-7: new token t),
//    token slots fixed (replay 0..3, new 4..7); split tables and raw K built by the prep warps;
//  * Gram: only the <= 22 needed dots, one round;
//  * coefficients: computed redundantly per warp (per-warp smem copy) -> no CTA barrier;
//  * K-log counter atomic issued early, its result consumed only at the end;
//  * commit: lane = (row l&15, half l>>4), 8 x 16-B chunks per lane: LDS.128, f32x2 mul/fma (.ftz like the
//    fast-math scalar ops), STS.128 + direct STG.128 (no smem->global pass);
//  * one row solve per lane (replay and new tokens).
// Decode vs materialize path independence: materialize keeps mm_replay (GSC_CK == 2); per element the committed value
// is fmul(s, G_R) followed by fma(c_j, k_j, .) for j < R in the same order, then RN to bf16 -> same bits.
// ---------------------------------------------------------------------------------------------
#if GSC_CK == 3
#ifndef GSC_CK3_MAP
#define GSC_CK3_MAP 0  // commit lane map. 0: lane = (row l&15, half l>>4), chunk 8*half+i; 1: lane pairs on adjacent
                       // chunks (2i+p) of one row (full 32-B sectors per STG), rows chosen for conflict-free LDS
#endif
#ifndef GSC_CK3_STORE
#define GSC_CK3_STORE 0  // 0: direct STG.128 from the commit; 1: smem only, then coalesced mm_store_rows
#endif
#ifndef GSC_CK3_PF
#define GSC_CK3_PF 0  // > 0: warp 1 L2-prefetches (bulk) the state tile + log of block (linear id + GSC_CK3_PF)
#endif
#ifndef GSC_CK3_ORDER
#define GSC_CK3_ORDER 0  // 1: issue the log / token loads (to registers) before the state cp.async
#endif
#ifndef GSC_CK3_PFS
#define GSC_CK3_PFS 1  // with GSC_CK3_PF: 1 also prefetches the 32 KB state tile, 0 = log rows only
#endif
#ifndef GSC_CK3_F2
#define GSC_CK3_F2 1  // 1: packed f32x2 commit; 0: scalar (same bits)
#endif
#ifndef GSC_CK3_REMAP
#define GSC_CK3_REMAP 0  // [rubin-ck] 1 (row loop unrolled x2) / 2 (not unrolled): commit lane map chunk c = lane >> 1, row half hh = lane & 1, 8 rows per lane.
                         // k_j of the lane's chunk is loaded once into registers (2 LDS.128 per j instead of 16) and
                         // each row's c_j comes from a per-warp smem table written by the row-solve lanes. Same
                         // per-element op sequence (fmul(s, G_R) then fma(c_j, k_j, .) for j < R ascending, RN bf16).
#endif
#if GSC_CK3_REMAP && !GSC_CK3_F2
#error "GSC_CK3_REMAP needs GSC_CK3_F2=1"
#endif
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t pack_rn(float a, float b) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}
#if GSC_CK3_F2
__device__ __forceinline__ unsigned long long f2_mul(unsigned long long a, unsigned long long b) {
  unsigned long long d;
  asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}
__device__ __forceinline__ unsigned long long f2_fma(unsigned long long a, unsigned long long b, unsigned long long c) {
  unsigned long long d;
  asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
  return d;
}
__device__ __forceinline__ unsigned long long f2_pk(float a, float b) {
  return static_cast<unsigned long long>(__float_as_uint(a)) | (static_cast<unsigned long long>(__float_as_uint(b)) << 32);
}
__device__ __forceinline__ float f2_lo(unsigned long long v) { return __uint_as_float(static_cast<unsigned>(v)); }
__device__ __forceinline__ float f2_hi(unsigned long long v) { return __uint_as_float(static_cast<unsigned>(v >> 32)); }
#endif

// prep_qk with the normalised values returned in registers (identical arithmetic).
__device__ __forceinline__ void prep_qk3(const __nv_bfloat16* qraw, const __nv_bfloat16* kraw, float scale, int lane,
                                         float (&qo)[4], float (&ko)[4], __nv_bfloat16 (&kr)[4]) {
  float q_values[4];
  float k_values[4];
  float q_square = 0.0f;
  float k_square = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int dim = lane + i * 32;
    q_values[i] = qraw ? __bfloat162float(qraw[dim]) : 0.0f;
    kr[i] = kraw[dim];
    k_values[i] = __bfloat162float(kr[i]);
    q_square += q_values[i] * q_values[i];
    k_square += k_values[i] * k_values[i];
  }
  const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
  const float q_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
  const float k_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    qo[i] = q_values[i] * q_scale;
    ko[i] = k_values[i] * k_scale;
  }
}
// prep_qk3 on already-loaded raw values (identical arithmetic; qraw_valid=false -> q = 0 like a null qraw)
__device__ __forceinline__ void prep_qk3r(const __nv_bfloat16 (&qr)[4], bool qraw_valid, const __nv_bfloat16 (&kr)[4],
                                          float scale, int lane, float (&qo)[4], float (&ko)[4]) {
  float q_values[4];
  float k_values[4];
  float q_square = 0.0f;
  float k_square = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q_values[i] = qraw_valid ? __bfloat162float(qr[i]) : 0.0f;
    k_values[i] = __bfloat162float(kr[i]);
    q_square += q_values[i] * q_values[i];
    k_square += k_values[i] * k_values[i];
  }
  const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
  const float q_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
  const float k_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    qo[i] = q_values[i] * q_scale;
    ko[i] = k_values[i] * k_scale;
  }
}
// split3 of 4 values (dims lane + 32 i) into row n of a split table with split stride sstride (bf16 elements)
__device__ __forceinline__ void split_row(__nv_bfloat16* v, int sstride, int n, const float (&x)[4], int lane) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat16 s[3];
    split3(x[i], s);
#pragma unroll
    for (int q = 0; q < 3; ++q) v[q * sstride + n * kVPad + lane + i * 32] = s[q];
  }
}
// S[warp rows] . v[n] for n < 8 (3 split terms), split q at v + q*sstride (same MMA order as mm_dots).
__device__ __forceinline__ void mm_dots3(float (&acc)[4], const __nv_bfloat16* st, const __nv_bfloat16* v, int sstride,
                                         int warp, int lane) {
  acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
  const int arow = warp * 16 + (lane & 15);
#pragma unroll
  for (int kt = 0; kt < kDimK / 16; ++kt) {
    uint32_t a[4];
    ldsm_x4(a, st + arow * kDimK + ((((2 * kt + (lane >> 4)) ^ (arow & 7))) << 3));
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      uint32_t b[2];
      ldsm_x2(b, v + s * sstride + (lane & 7) * kVPad + kt * 16 + ((lane >> 3) & 1) * 8);
      mma_bf16(acc, a, b);
    }
  }
}
// value of C-fragment column n (0..7) of the row (g + 8h) owned by quad g, fetched into every lane of that row
__device__ __forceinline__ float cfrag_get(const float (&acc)[4], int g, int h, int n) {
  const int src = 4 * g + (n >> 1);
  const float x0 = __shfl_sync(0xffffffffu, acc[n & 1], src);
  const float x1 = __shfl_sync(0xffffffffu, acc[2 + (n & 1)], src);
  return h ? x1 : x0;
}

constexpr int kSplitRep = 4 * kVPad;   // replay split stride (rows 4..7 of a split alias the next one: ignored cols)
constexpr int kSplitNew = 8 * kVPad;

template <int VPK, bool SigmoidGate>
__global__ __launch_bounds__(kThreads, GSC_MINB) void ck3_decode_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv, const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, int state_indices_width, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, __nv_bfloat16* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate, const void* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ out, int H, int HV, int dt_bias_type, bool norm_weight_is_bf16, float scale,
    float norm_eps, Strides strides) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int key_head = value_head / VPK;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // ---- round 1: metadata (no branch before all four loads are issued)
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int base_slot = state_indices[static_cast<int64_t>(request) * state_indices_width];
  int accepted = num_accepted_tokens[request];
#if GSC_CK3_PF
  const int pb = blockIdx.x + blockIdx.y * gridDim.x + GSC_CK3_PF;
  const bool pf_on = warp == 1 && lane < 14 && pb < static_cast<int>(gridDim.x * gridDim.y);
  const int pf_req = pb % gridDim.x, pf_vh = pb / gridDim.x;
  const int pf_slot = pf_on ? state_indices[static_cast<int64_t>(pf_req) * state_indices_width] : 0;
#endif
  const int T = eos - bos;
  if (T <= 0) return;
  if (base_slot <= 0 || T > kMaxT) {
    for (int linear = tid; linear < T * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + linear % kDimV] = __float2bfloat16(0.0f);
    }
    return;
  }
  accepted = accepted < 0 ? 0 : (accepted > kMaxT ? kMaxT : accepted);

  extern __shared__ __align__(16) unsigned char gsc_ck3_smem[];
  __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(gsc_ck3_smem);  // [128][128] swizzled
  __shared__ __align__(16) float sk[kMaxTok][kDimK];     // 0..3 replay, 4..7 new
  __shared__ __align__(16) float sq[kMaxT][kDimK];
  __shared__ __align__(16) __nv_bfloat16 sv[kMaxTok][kDimV];
  __shared__ __align__(16) __nv_bfloat16 skraw[kMaxT][kDimK];
  __shared__ __align__(16) __nv_bfloat16 sout[kMaxT][kDimV];
  __shared__ __align__(16) __nv_bfloat16 svec[3 * 4 * kVPad + 3 * 8 * kVPad];  // replay splits, then new splits
  __shared__ float sdecay[kMaxTok], sbeta[kMaxTok];
  __shared__ CkRep s_rep;
  __shared__ CkNew s_new;
  __shared__ int s_last;
#if GSC_CK3_REMAP
  // [rubin-ck] per-warp c_j table, entry hh * 9 + r for warp row hh * 8 + r (padding puts rows r and 8 + r on
  // different banks)
  __shared__ __align__(16) float s_cc[kWarps][17][kMaxT];
#endif
  __nv_bfloat16* vrep = svec;
  __nv_bfloat16* vnew = svec + 3 * kSplitRep;

  // ---- round 2: state load + all token loads
  const LogLayout ll = log_layout(H, HV);
  __nv_bfloat16* slot_state = state + static_cast<int64_t>(base_slot) * strides.state_slot;
  char* log = reinterpret_cast<char*>(slot_state + static_cast<int64_t>(HV) * kDimV * kDimK);
  __nv_bfloat16* log_k = reinterpret_cast<__nv_bfloat16*>(log + ll.k_off);
  __nv_bfloat16* log_v = reinterpret_cast<__nv_bfloat16*>(log + ll.v_off);
  __nv_bfloat16* log_ab = reinterpret_cast<__nv_bfloat16*>(log + ll.ab_off);
  int* log_flag = reinterpret_cast<int*>(log + ll.l_off) + value_head;
  int* log_ctr = reinterpret_cast<int*>(log + ll.c_off) + key_head;
  __nv_bfloat16* head_state = slot_state + static_cast<int64_t>(value_head) * kDimV * kDimK;
#if GSC_CK3_ORDER == 0
  mm_load_rows(st, head_state, warp, lane);
#endif
  const int flag = *log_flag;
#if GSC_CK3_PF
  if (pf_slot > 0 && (GSC_CK3_PFS || lane > 0)) {
    const __nv_bfloat16* ps = state + static_cast<int64_t>(pf_slot) * strides.state_slot;
    const char* pl = reinterpret_cast<const char*>(ps + static_cast<int64_t>(HV) * kDimV * kDimK);
    const char* addr;
    int bytes = 256;
    if (lane == 0) { addr = reinterpret_cast<const char*>(ps + static_cast<int64_t>(pf_vh) * kDimV * kDimK); bytes = 32768; }
    else if (lane < 5) addr = pl + ll.k_off + (static_cast<int64_t>(lane - 1) * H + pf_vh / VPK) * kDimK * 2;
    else if (lane < 9) addr = pl + ll.v_off + (static_cast<int64_t>(lane - 5) * HV + pf_vh) * kDimV * 2;
    else if (lane < 13) addr = pl + ll.ab_off + ((lane - 9) * HV + pf_vh) * 4;
    else addr = pl + ll.l_off + pf_vh * 4;
    if (bytes == 32768 || lane < 9) {
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(addr), "r"(bytes) : "memory");
    } else {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(addr));
    }
  }
#endif

  uint32_t gate_raw[2] = {0u, 0u};  // new-token warps: output_gate (4 bf16) of token t for the epilogue
  __nv_bfloat16 ab_raw[2];
  // load phase (registers)
  __nv_bfloat16 qr[4], kr[4], vr[4], g4[4];
  const bool rep_w = warp < kMaxT && warp < accepted;
  const bool new_w = warp >= kMaxT && warp - kMaxT < T;
  const int t = warp - kMaxT;
  const int token = bos + t;
  const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
  if (rep_w) {
    const int j = warp;
    const __nv_bfloat16* kp = log_k + (static_cast<int64_t>(j) * H + key_head) * kDimK;
    const __nv_bfloat16* vp = log_v + (static_cast<int64_t>(j) * HV + value_head) * kDimV;
#pragma unroll
    for (int i = 0; i < 4; ++i) { kr[i] = kp[lane + i * 32]; vr[i] = vp[lane + i * 32]; }
    if (lane == 0) { ab_raw[0] = log_ab[(j * HV + value_head) * 2]; ab_raw[1] = log_ab[(j * HV + value_head) * 2 + 1]; }
  } else if (new_w) {
    const __nv_bfloat16* gp = output_gate + static_cast<int64_t>(token) * strides.gate_row + value_head * kDimV;
    const __nv_bfloat16* qp = mixed_qkv + mixed_base + key_head * kDimK;
    const __nv_bfloat16* kp = mixed_qkv + mixed_base + H * kDimK + key_head * kDimK;
    const __nv_bfloat16* vp = mixed_qkv + mixed_base + 2 * H * kDimK + value_head * kDimV;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      qr[i] = qp[lane + i * 32]; kr[i] = kp[lane + i * 32]; vr[i] = vp[lane + i * 32]; g4[i] = gp[lane + i * 32];
    }
    if (lane == 0) {
      ab_raw[0] = a[static_cast<int64_t>(token) * strides.a_row + value_head];
      ab_raw[1] = b[static_cast<int64_t>(token) * strides.b_row + value_head];
    }
  }
#if GSC_CK3_ORDER == 1
  mm_load_rows(st, head_state, warp, lane);
#endif
  // compute phase
  if (rep_w) {
    const int j = warp;
    float qo[4], ko[4];
    prep_qk3r(qr, false, kr, scale, lane, qo, ko);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sk[j][lane + i * 32] = ko[i];
      sv[j][lane + i * 32] = vr[i];
    }
    split_row(vrep, kSplitRep, j, ko, lane);
    if (lane == 0)
      prep_gate(__bfloat162float(ab_raw[0]), __bfloat162float(ab_raw[1]), a_log[value_head],
                load_dt_bias(dt_bias, value_head, dt_bias_type), &sdecay[j], &sbeta[j]);
  } else if (new_w) {
    float qo[4], ko[4];
    prep_qk3r(qr, true, kr, scale, lane, qo, ko);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sq[t][lane + i * 32] = qo[i];
      sk[kMaxT + t][lane + i * 32] = ko[i];
      skraw[t][lane + i * 32] = kr[i];
      sv[kMaxT + t][lane + i * 32] = vr[i];
    }
    split_row(vnew, kSplitNew, t, ko, lane);
    split_row(vnew, kSplitNew, 4 + t, qo, lane);
    if (lane == 0)
      prep_gate(__bfloat162float(ab_raw[0]), __bfloat162float(ab_raw[1]), a_log[value_head],
                load_dt_bias(dt_bias, value_head, dt_bias_type), &sdecay[kMaxT + t], &sbeta[kMaxT + t]);
    gate_raw[0] = pack_bf16(g4[0], g4[1]);
    gate_raw[1] = pack_bf16(g4[2], g4[3]);
  }
  __syncthreads();  // #1: token prep done; this CTA's log reads are complete
  const int R = flag != 0 ? accepted : 0;

  // ---- Gram (<= 22 dots, one round): p 0..5 replay pairs, 6..11 new pairs, 12..21 k_new.q
  {
    const int grp = tid >> 3, seg = tid & 7;
    int i = 0, j = 0;
    bool valid = false, isq = false;
    if (grp < 12) {
      const int p = grp < 6 ? grp : grp - 6;
      const int jj = p < 1 ? 1 : p < 3 ? 2 : 3;
      const int ii = p - jj * (jj - 1) / 2;
      if (grp < 6) { i = ii; j = jj; valid = jj < R; }
      else { i = kMaxT + ii; j = kMaxT + jj; valid = jj < T; }
    } else if (grp < 22) {
      // (s,t), s <= t: t = 0:(0) 1:(0,1) 2:(0..2) 3:(0..3)
      const int p = grp - 12;
      const int t = p < 1 ? 0 : p < 3 ? 1 : p < 6 ? 2 : 3;
      const int s = p - t * (t + 1) / 2;
      i = s; j = t; isq = true; valid = t < T;
    }
    const float* x = valid ? (isq ? sk[kMaxT + i] : sk[i]) : sk[0];
    const float* y = valid ? (isq ? sq[j] : sk[j]) : sk[0];
    const float d = ck_dot8(x, y, seg);
    // coefficients, same ops as ck_coef_*_par: W/Q = rho * dot (rho from 1.0f, increasing r)
    if (valid && seg == 0) {
      if (isq) {
        float rho = 1.0f;
        for (int r = i + 1; r <= j; ++r) rho = __fmul_rn(rho, sdecay[kMaxT + r]);
        s_new.Q[i][j] = __fmul_rn(rho, d);
      } else {
        const int off = grp < 6 ? 0 : kMaxT;
        float rho = 1.0f;
        for (int r = i + 1; r <= j; ++r) rho = __fmul_rn(rho, sdecay[r]);
        if (grp < 6) s_rep.W[i][j] = __fmul_rn(rho, d);
        else s_new.W[i - off][j - off] = __fmul_rn(rho, d);
      }
    }
    const int u = tid - 22 * 8;
    if (u >= 0 && u < kMaxT) {
      const int jj = u;
      if (jj < R) {
        float g = 1.0f;
        for (int r = 0; r <= jj; ++r) g = __fmul_rn(g, sdecay[r]);
        s_rep.G[jj] = g;
        s_rep.beta[jj] = sbeta[jj];
        float rho = 1.0f;
        for (int r = jj + 1; r < R; ++r) rho = __fmul_rn(rho, sdecay[r]);
        s_rep.C[jj] = rho;
      }
    } else if (u == kMaxT) {
      float g = 1.0f;
      for (int r = 0; r < R; ++r) g = __fmul_rn(g, sdecay[r]);
      s_rep.GR = R > 0 ? g : 1.0f;
    } else if (u > kMaxT && u <= 2 * kMaxT) {
      const int t = u - kMaxT - 1;
      if (t < T) {
        float g = 1.0f;
        for (int r = 0; r <= t; ++r) g = __fmul_rn(g, sdecay[kMaxT + r]);
        s_new.G[t] = g;
        s_new.beta[t] = sbeta[kMaxT + t];
      }
    }
  }
  // log writes for the new tokens (replay reads finished before #1); K log by the last CTA of the key head (end)
  int prev_ctr = 0;
  if (warp >= kMaxT) {
    const int t = warp - kMaxT;
    if (t < T) {
      const uint2 v = *reinterpret_cast<const uint2*>(&sv[kMaxT + t][lane * 4]);
      *reinterpret_cast<uint2*>(log_v + (static_cast<int64_t>(t) * HV + value_head) * kDimV + lane * 4) = v;
      if (lane == 0) {
        log_ab[(t * HV + value_head) * 2] = ab_raw[0];
        log_ab[(t * HV + value_head) * 2 + 1] = ab_raw[1];
      }
    }
    if (VPK > 1 && tid == kThreads - 32) {
      __threadfence();
      prev_ctr = atomicAdd(log_ctr, 1);
    }
  }
  if (tid == 0) *log_flag = 1;
  __syncthreads();  // #2: Gram
  const CkRep& crep = s_rep;
  const CkNew& cnew = s_new;
  cp_async_wait_group<0>();
  __syncwarp();

#if GSC_CK3_MAP == 0
  const int rr = lane & 15, half = lane >> 4;
#define CK3_CHUNK(i_) (8 * half + (i_))
#else
  const int half = lane & 1, qq = lane >> 3;
  const int rr = ((lane >> 1) & 3) * 2 + (qq & 1) + 8 * (qq >> 1);
#define CK3_CHUNK(i_) (2 * (i_) + half)
#endif
  const int g = rr & 7, h = rr >> 3;
  const int row = warp * 16 + rr;
  float acc[4];
  if (R > 0) {
    mm_dots3(acc, st, vrep, kSplitRep, warp, lane);
    float cc[kMaxT];
    {
      float d[kMaxT];
#pragma unroll
      for (int j = 0; j < kMaxT; ++j) {
        const float av = cfrag_get(acc, g, h, j);
        cc[j] = 0.0f;
        if (j < R) {
          float ks = __fmul_rn(crep.G[j], av);
#pragma unroll
          for (int i = 0; i < j; ++i) ks = __fmaf_rn(crep.W[i][j], d[i], ks);
          d[j] = __fmul_rn(__fsub_rn(__bfloat162float(sv[j][row]), ks), crep.beta[j]);
          cc[j] = __fmul_rn(crep.C[j], d[j]);
        }
      }
    }
    const float GR = crep.GR;
#if GSC_CK3_REMAP
    // [rubin-ck] commit, lane = (chunk c = lane >> 1, row half hh = lane & 1), rows hh * 8 + r (r < 8) of this warp.
    // Per element (row, col) the op sequence is unchanged: u = fmul(s, G_R) (f32x2 .ftz), then for j < R in
    // ascending order u = fma(c_j[row], k_j[col], u) (f32x2 .ftz), then RN to bf16 -> bitwise equal to the CK3 map.
    if (half == 0)
      *reinterpret_cast<float4*>(&s_cc[warp][(rr >> 3) * 9 + (rr & 7)][0]) = make_float4(cc[0], cc[1], cc[2], cc[3]);
    __syncwarp();
    {
      const int c = lane >> 1, hh = lane & 1;
      unsigned long long kp[kMaxT][4];
#pragma unroll
      for (int j = 0; j < kMaxT; ++j) {
        if (j < R) {
          const float4 k0 = *reinterpret_cast<const float4*>(&sk[j][c * 8]);
          const float4 k1 = *reinterpret_cast<const float4*>(&sk[j][c * 8 + 4]);
          kp[j][0] = f2_pk(k0.x, k0.y);
          kp[j][1] = f2_pk(k0.z, k0.w);
          kp[j][2] = f2_pk(k1.x, k1.y);
          kp[j][3] = f2_pk(k1.z, k1.w);
        }
      }
      const unsigned long long gr2 = f2_pk(GR, GR);
      const uint32_t sbase0 = static_cast<uint32_t>(__cvta_generic_to_shared(st)) + (warp * 16 + hh * 8) * (kDimK * 2);
      __nv_bfloat16* gdst0 = head_state + (warp * 16 + hh * 8) * kDimK + c * 8;
#if GSC_CK3_REMAP == 2
#pragma unroll 1
#else
#pragma unroll 2
#endif
      for (int r = 0; r < 8; ++r) {
        const uint32_t addr = sbase0 + r * (kDimK * 2) + ((c ^ r) << 4);  // row & 7 == r
        uint4 sv4;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv4.x), "=r"(sv4.y), "=r"(sv4.z), "=r"(sv4.w)
                     : "r"(addr));
        const float4 cr = *reinterpret_cast<const float4*>(&s_cc[warp][hh * 9 + r][0]);
        const float ccr[kMaxT] = {cr.x, cr.y, cr.z, cr.w};
        const uint32_t w[4] = {sv4.x, sv4.y, sv4.z, sv4.w};
        unsigned long long u[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) u[e] = f2_mul(f2_pk(bf_lo(w[e]), bf_hi(w[e])), gr2);
#pragma unroll
        for (int j = 0; j < kMaxT; ++j) {
          if (j < R) {
            const unsigned long long c2 = f2_pk(ccr[j], ccr[j]);
#pragma unroll
            for (int e = 0; e < 4; ++e) u[e] = f2_fma(c2, kp[j][e], u[e]);
          }
        }
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = pack_rn(f2_lo(u[e]), f2_hi(u[e]));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(o[0]), "r"(o[1]), "r"(o[2]), "r"(o[3])
                     : "memory");
#if GSC_CK3_STORE == 0
        *reinterpret_cast<uint4*>(gdst0 + r * kDimK) = make_uint4(o[0], o[1], o[2], o[3]);
#endif
      }
    }
    __syncwarp();
#if GSC_CK3_STORE == 1
    mm_store_rows(head_state, st, warp, lane);
#endif
#else
    // commit: 8 chunks (8 cols each) of this lane's row half
    const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(st)) + row * (kDimK * 2);
    __nv_bfloat16* gdst = head_state + row * kDimK;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = CK3_CHUNK(i);
      const uint32_t addr = sbase + ((c ^ g) << 4);
      const float* kb = &sk[0][c * 8];
      uint4 sv4;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv4.x), "=r"(sv4.y), "=r"(sv4.z), "=r"(sv4.w)
                   : "r"(addr));
      const uint32_t w[4] = {sv4.x, sv4.y, sv4.z, sv4.w};
      uint32_t o[4];
#if GSC_CK3_F2
      const unsigned long long gr2 = f2_pk(GR, GR);
      unsigned long long u[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) u[e] = f2_mul(f2_pk(bf_lo(w[e]), bf_hi(w[e])), gr2);
#pragma unroll
      for (int j = 0; j < kMaxT; ++j) {
        if (j < R) {
          const float4 k0 = *reinterpret_cast<const float4*>(kb + j * kDimK);
          const float4 k1 = *reinterpret_cast<const float4*>(kb + j * kDimK + 4);
          const unsigned long long c2 = f2_pk(cc[j], cc[j]);
          u[0] = f2_fma(c2, f2_pk(k0.x, k0.y), u[0]);
          u[1] = f2_fma(c2, f2_pk(k0.z, k0.w), u[1]);
          u[2] = f2_fma(c2, f2_pk(k1.x, k1.y), u[2]);
          u[3] = f2_fma(c2, f2_pk(k1.z, k1.w), u[3]);
        }
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) o[e] = pack_rn(f2_lo(u[e]), f2_hi(u[e]));
#else
      float u[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        u[2 * e] = __fmul_rn(bf_lo(w[e]), GR);
        u[2 * e + 1] = __fmul_rn(bf_hi(w[e]), GR);
      }
#pragma unroll
      for (int j = 0; j < kMaxT; ++j) {
        if (j < R) {
#pragma unroll
          for (int e = 0; e < 8; ++e) u[e] = __fmaf_rn(cc[j], kb[j * kDimK + e], u[e]);
        }
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) o[e] = pack_rn(u[2 * e], u[2 * e + 1]);
#endif
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(o[0]), "r"(o[1]), "r"(o[2]), "r"(o[3])
                   : "memory");
#if GSC_CK3_STORE == 0
      *reinterpret_cast<uint4*>(gdst + c * 8) = make_uint4(o[0], o[1], o[2], o[3]);
#endif
    }
    __syncwarp();
#if GSC_CK3_STORE == 1
    (void)gdst;
    mm_store_rows(head_state, st, warp, lane);
#endif
#endif  // GSC_CK3_REMAP
  }
  // ---- new tokens: outputs only
  mm_dots3(acc, st, vnew, kSplitNew, warp, lane);
  {
    float d[kMaxT];
    float mine[2] = {0.0f, 0.0f};
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) {
      const float av = cfrag_get(acc, g, h, t);
      const float bv = cfrag_get(acc, g, h, 4 + t);
      if (t < T) {
        float ks = __fmul_rn(cnew.G[t], av);
#pragma unroll
        for (int s = 0; s < t; ++s) ks = __fmaf_rn(cnew.W[s][t], d[s], ks);
        d[t] = __fmul_rn(__fsub_rn(__bfloat162float(sv[kMaxT + t][row]), ks), cnew.beta[t]);
        if ((t >> 1) == half) {
          float o = __fmul_rn(cnew.G[t], bv);
#pragma unroll
          for (int s = 0; s <= t; ++s) o = __fmaf_rn(cnew.Q[s][t], d[s], o);
          mine[t & 1] = o;
        }
      }
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int t = half * 2 + e;
      if (t < T) sout[t][row] = __float2bfloat16(mine[e]);
    }
  }
  if (VPK > 1 && tid == kThreads - 32) {
    s_last = prev_ctr == VPK - 1;
    if (prev_ctr == VPK - 1) *log_ctr = 0;
  }
  __syncthreads();  // #3
  if (warp >= kMaxT) {
    const int t = warp - kMaxT;
    if (t < T) {
      float output_values[4];
      float sum_square = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        output_values[i] = __bfloat162float(sout[t][lane + i * 32]);
        sum_square += output_values[i] * output_values[i];
      }
      sum_square = warp_reduce_sum(sum_square);
      const float rstd = rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
      const int token = bos + t;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int value = lane + i * 32;
        const uint32_t gw = gate_raw[i >> 1];
        const float gate_input = (i & 1) ? bf_hi(gw) : bf_lo(gw);
        const float gate = SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
        const float weight = norm_weight_is_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(norm_weight)[value])
                                                 : static_cast<const float*>(norm_weight)[value];
        out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + value] =
            __float2bfloat16(output_values[i] * rstd * weight * gate);
      }
    }
  } else if (VPK == 1 || s_last) {
    const int t = warp;
    if (t < T) {
      const uint2 v = *reinterpret_cast<const uint2*>(&skraw[t][lane * 4]);
      *reinterpret_cast<uint2*>(log_k + (static_cast<int64_t>(t) * H + key_head) * kDimK + lane * 4) = v;
    }
  }
}
#undef CK3_CHUNK
#endif  // GSC_CK == 3

// v3: persistent variant. Grid = min(#items, resident CTAs); each CTA walks work items
// w = blockIdx.x + k*gridDim.x, item = (request, value head). The 4 state chunks of the NEXT item are
// streamed into the chunk slots as soon as the current item releases them (slot c-1 at the start of
// chunk c, slot 3 after the token loop), so the next head's 64 KB load overlaps this head's compute,
// epilogue and the next prep. Group accounting: exactly 3 commit groups are issued after the group
// of the chunk being waited on, so every wait is cp.async.wait_group 3 (empty groups when there is
// no next item). Per-item math is identical to deferred_decode_kernel (bit-exact vs stock).
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_commit_empty() { asm volatile("cp.async.commit_group;\n" ::); }

template <typename S, int VPK, bool SigmoidGate>
__global__ __launch_bounds__(kThreads, 3) void persistent_decode_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv, const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, int state_indices_width, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, S* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate, const void* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ out, int H, int HV, int dt_bias_type, bool norm_weight_is_bf16, float scale,
    float norm_eps, Strides strides, int num_requests) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int total_items = num_requests * HV;
  const LogLayout ll = log_layout(H, HV);

  extern __shared__ __align__(16) unsigned char gsc_dyn_smem[];
  S* shared_state = reinterpret_cast<S*>(gsc_dyn_smem);  // [kNumChunks][kChunkV][kDimK]
  __shared__ __align__(16) float shared_q[kMaxT][kDimK];
  __shared__ __align__(16) float shared_k[kMaxTok][kDimK];
  __shared__ __nv_bfloat16 shared_v[kMaxTok][kDimV];
  __shared__ __nv_bfloat16 shared_out[kMaxT][kDimV];
  __shared__ float shared_decay[kMaxTok];
  __shared__ float shared_beta[kMaxTok];
  __shared__ int s_last;

  // next valid item >= w (zero-fills outputs of items without a state slot, like stock)
  auto next_valid = [&](int w) -> int {
    for (; w < total_items; w += gridDim.x) {
      const int req = w / HV, vh = w % HV;
      const int bos = cu_seqlens[req];
      const int T = cu_seqlens[req + 1] - bos;
      if (T <= 0) continue;
      const int base = state_indices[static_cast<int64_t>(req) * state_indices_width];
      if (base > 0 && T <= kMaxT) return w;
      for (int linear = tid; linear < T * kDimV; linear += kThreads) {
        const int token = bos + linear / kDimV;
        out[(static_cast<int64_t>(token) * HV + vh) * kDimV + linear % kDimV] = __float2bfloat16(0.0f);
      }
    }
    return total_items;
  };
  auto head_ptr = [&](int w) -> S* {
    const int req = w / HV, vh = w % HV;
    const int base = state_indices[static_cast<int64_t>(req) * state_indices_width];
    return state + static_cast<int64_t>(base) * strides.state_slot + static_cast<int64_t>(vh) * kDimV * kDimK;
  };

  int w = next_valid(blockIdx.x);
  if (w < total_items) {
    S* hp = head_ptr(w);
#pragma unroll
    for (int chunk = 0; chunk < kNumChunks; ++chunk) copy_state_chunk(shared_state, hp, chunk, chunk, tid);
  } else {
    return;
  }
  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) rows[row] = warp + row * kWarps;

  while (w < total_items) {
    const int request = w / HV;
    const int value_head = w % HV;
    const int key_head = value_head / VPK;
    const int bos = cu_seqlens[request];
    const int num_tokens = cu_seqlens[request + 1] - bos;
    const int base_slot = state_indices[static_cast<int64_t>(request) * state_indices_width];
    S* slot_state = state + static_cast<int64_t>(base_slot) * strides.state_slot;
    S* head_state = slot_state + static_cast<int64_t>(value_head) * kDimV * kDimK;
    char* log = reinterpret_cast<char*>(slot_state + static_cast<int64_t>(HV) * kDimV * kDimK);
    __nv_bfloat16* log_k = reinterpret_cast<__nv_bfloat16*>(log + ll.k_off);
    __nv_bfloat16* log_v = reinterpret_cast<__nv_bfloat16*>(log + ll.v_off);
    __nv_bfloat16* log_ab = reinterpret_cast<__nv_bfloat16*>(log + ll.ab_off);
    int* log_flag = reinterpret_cast<int*>(log + ll.l_off) + value_head;
    int* log_ctr = reinterpret_cast<int*>(log + ll.c_off) + key_head;
    const int flag = *log_flag;
    int accepted = num_accepted_tokens[request];
    accepted = accepted < 0 ? 0 : (accepted > kMaxT ? kMaxT : accepted);
    const int R = flag != 0 ? accepted : 0;
    const int total = R + num_tokens;

    if (warp < total) {
      const int j = warp;
      if (j < R) {
        prep_qk(nullptr, log_k + (static_cast<int64_t>(j) * H + key_head) * kDimK, scale, lane, nullptr, shared_k[j]);
        const __nv_bfloat16* vr = log_v + (static_cast<int64_t>(j) * HV + value_head) * kDimV;
#pragma unroll
        for (int i = 0; i < 4; ++i) shared_v[j][lane + i * 32] = vr[lane + i * 32];
        if (lane == 0)
          prep_gate(__bfloat162float(log_ab[(j * HV + value_head) * 2]),
                    __bfloat162float(log_ab[(j * HV + value_head) * 2 + 1]), a_log[value_head],
                    load_dt_bias(dt_bias, value_head, dt_bias_type), &shared_decay[j], &shared_beta[j]);
      } else {
        const int t = j - R;
        const int token = bos + t;
        const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
        prep_qk(mixed_qkv + mixed_base + key_head * kDimK, mixed_qkv + mixed_base + H * kDimK + key_head * kDimK,
                scale, lane, shared_q[t], shared_k[j]);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          shared_v[j][lane + i * 32] = mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + lane + i * 32];
        if (lane == 0)
          prep_gate(__bfloat162float(a[static_cast<int64_t>(token) * strides.a_row + value_head]),
                    __bfloat162float(b[static_cast<int64_t>(token) * strides.b_row + value_head]), a_log[value_head],
                    load_dt_bias(dt_bias, value_head, dt_bias_type), &shared_decay[j], &shared_beta[j]);
      }
    }
    __syncthreads();
    if (tid == 0) {
      if (VPK == 1) {
        s_last = 1;
      } else {
        __threadfence();
        const int prev = atomicAdd(log_ctr, 1);
        s_last = prev == VPK - 1;
        if (s_last) *log_ctr = 0;
      }
    }
    __syncthreads();
    if (s_last) {
      __threadfence();
      for (int linear = tid; linear < num_tokens * kDimK; linear += kThreads) {
        const int t = linear / kDimK, d = linear % kDimK;
        log_k[(static_cast<int64_t>(t) * H + key_head) * kDimK + d] =
            mixed_qkv[static_cast<int64_t>(bos + t) * strides.mixed_row + H * kDimK + key_head * kDimK + d];
      }
    }
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int t = linear / kDimV, d = linear % kDimV;
      log_v[(static_cast<int64_t>(t) * HV + value_head) * kDimV + d] = shared_v[R + t][d];
    }
    if (tid < num_tokens) {
      log_ab[(tid * HV + value_head) * 2] = a[static_cast<int64_t>(bos + tid) * strides.a_row + value_head];
      log_ab[(tid * HV + value_head) * 2 + 1] = b[static_cast<int64_t>(bos + tid) * strides.b_row + value_head];
    }
    if (tid == 0) *log_flag = 1;

    const int w_next = next_valid(w + gridDim.x);
    S* next_hp = w_next < total_items ? head_ptr(w_next) : nullptr;

#pragma unroll
    for (int chunk = 0; chunk < kNumChunks; ++chunk) {
      cp_async_wait_group<3>();
      __syncthreads();
      if (chunk > 0) {  // slot chunk-1 is free: stream the next item's chunk into it
        if (next_hp) copy_state_chunk(shared_state, next_hp, chunk - 1, chunk - 1, tid);
        else cp_async_commit_empty();
      }
      float h[4][4];
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        load_h4<S>(&shared_state[(chunk * kChunkV + rows[row]) * kDimK + k_base], h[row]);
      }
      for (int j = 0; j < R; ++j)
        token_update_multi<false, 1>(h, shared_k[j], nullptr, shared_v[j], shared_decay[j], shared_beta[j], chunk,
                                     rows, k_base, lane, nullptr);
      if (R > 0) {
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
          const int value = chunk * kChunkV + rows[row];
          store_h4<S>(head_state + value * kDimK + k_base, h[row]);
        }
      }
      for (int t = 0; t < num_tokens; ++t)
        token_update_multi<true, 1>(h, shared_k[R + t], shared_q[t], shared_v[R + t], shared_decay[R + t],
                                    shared_beta[R + t], chunk, rows, k_base, lane, shared_out[t]);
    }
    __syncthreads();
    if (next_hp) copy_state_chunk(shared_state, next_hp, kNumChunks - 1, kNumChunks - 1, tid);
    else cp_async_commit_empty();
    if (warp < num_tokens) {
      const int t = warp;
      float output_values[4];
      float sum_square = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        output_values[i] = __bfloat162float(shared_out[t][lane + i * 32]);
        sum_square += output_values[i] * output_values[i];
      }
      sum_square = warp_reduce_sum(sum_square);
      const float rstd = rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
      const int token = bos + t;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int value = lane + i * 32;
        const float gate_input =
            __bfloat162float(output_gate[static_cast<int64_t>(token) * strides.gate_row + value_head * kDimV + value]);
        const float gate = SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
        const float weight = norm_weight_is_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(norm_weight)[value])
                                                 : static_cast<const float*>(norm_weight)[value];
        out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + value] =
            __float2bfloat16(output_values[i] * rstd * weight * gate);
      }
    }
    __syncthreads();  // shared_out / shared_k/v/q reuse by the next item
    w = w_next;
  }
  cp_async_wait_group<0>();
}

// ---------------------------------------------------------------------------------------------
// materialize: dst = replay(src, log_src[:n]); L[dst] = 0.  Grid (items, H, layers).
// mode 0 (direct, in place): slot = a0[i], n = a1[i]; has_init[i] == 0 -> only clear L.
// mode 1 (precopy): cols from a0 (src col, -1 = none), a1 (dst col), a2 (token bias = acc - 1).
// mode 2 (postprocess): a0 = num_accepted, a1 = state col, a2 = num_scheduled, a3 = num_computed,
//                       a4 = num_draft (V1 fused-postprocess decision logic).
// ---------------------------------------------------------------------------------------------
struct MatArgs {
  int mode;
  const int* a0; const int* a1; const int* a2; const int* a3; const int* a4;
  const bool* has_init;
  const int* idx_map;
  const int64_t* bt_ptrs;  // per group block table base (int32 [max_reqs, max_blocks])
  int64_t bt_stride;
  int block_size;
  const int64_t* layer_state;  // per layer ssm base address
  const int64_t* layer_alog;   // per layer A_log (fp32)
  const int64_t* layer_dtb;    // per layer dt_bias
  const int* layer_group;      // per layer group index into bt_ptrs
  int64_t slot_stride_floats;
  int H, HV, dt_bias_type;
  int state_bf16;
  int num_items;  // [rubin-ck] items of the call (compact mode scans them)
  int compact;    // [rubin-ck] > 0: grid (compact, H, layers) loops over the items that pass the per-item decision
};

// [rubin-ck] compact mode: the item list holds at most kMatMaxItems items (the host falls back above that).
constexpr int kMatMaxItems = 1024;

// [rubin-ck] the layer- and head-independent part of mat_item's per-item decision (modes 1-3): false exactly when
// mat_item returns before any memory write for every (key head, layer). mat_item re-evaluates all of it.
__device__ __forceinline__ bool mat_active(const MatArgs& args, int item) {
  int r = item;
  if (args.idx_map) {
    r = args.idx_map[item];
    if (r < 0) return false;
  }
  if (args.mode == 0) return true;
  if (args.mode == 1) {
    const int src_col = args.a0[r];
    const int dst_col = args.a1[r];
    return !(src_col < 0 || src_col == dst_col);
  }
  const int acc = args.a0[r];
  const int running = args.mode == 2 ? args.a3[r] + args.a2[r] - args.a4[r] : args.a3[r] - acc + 1;
  const int new_computed = running + acc - 1;
  const int aligned = (new_computed / args.block_size) * args.block_size;
  return !(aligned < running);
}

template <typename S, int VPK>
__device__ __forceinline__ void mat_item(const MatArgs& args, const int item, const int key_head, const int layer);

template <typename S, int VPK>
__global__ __launch_bounds__(kThreads, 2) void materialize_kernel(MatArgs args) {
  mat_item<S, VPK>(args, blockIdx.x, blockIdx.y, blockIdx.z);
}

// [gx-alignc] CTAs/SM bound of the compact kernel: fp32 state 4 (the default fp32 build is 64 regs / 4 CTAs/SM, and
// 152 SMs x 4 = 608 >= the 480 (key head, layer) CTAs of one active item -> one wave), bf16 3 (rubin-ck's 80-reg build).
// Default 3 for both state types = rubin-ck as shipped (the Rubin reference build); the GB300 fp32 value 4 is selected
// with GDN_STATE_COMMIT_MATC_MINB_F32=4 (passed as -DGSC_MATC_MINB_F32 by load()).
#ifndef GSC_MATC_MINB_F32
#define GSC_MATC_MINB_F32 3
#endif
#ifndef GSC_MATC_MINB_BF16
#define GSC_MATC_MINB_BF16 3
#endif

// [rubin-ck] compact mode (separate kernel so the default kernel's code and occupancy are unchanged): every CTA
// builds the same ascending list of active items (one load round per item, spread over the CTA), then CTA x handles
// list entries x, x + gridDim.x, ... for its (key head, layer). Per-item work is the unchanged mat_item -> the same
// bytes are written as in the full (items, H, layers) grid.
template <typename S, int VPK>
__global__ __launch_bounds__(kThreads, sizeof(S) == 4 ? GSC_MATC_MINB_F32 : GSC_MATC_MINB_BF16)
void materialize_compact_kernel(MatArgs args) {
  __shared__ int s_list[kMatMaxItems];
  __shared__ int s_wsum[kWarps];
  __shared__ int s_cnt;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) s_cnt = 0;
  __syncthreads();
  for (int base = 0; base < args.num_items; base += kThreads) {
    const int it = base + tid;
    const bool act = it < args.num_items && mat_active(args, it);
    const unsigned bal = __ballot_sync(0xffffffffu, act);
    if (lane == 0) s_wsum[warp] = __popc(bal);
    __syncthreads();
    if (act) {
      int pos = s_cnt + __popc(bal & ((1u << lane) - 1u));
      for (int w = 0; w < warp; ++w) pos += s_wsum[w];
      s_list[pos] = it;
    }
    __syncthreads();
    if (tid == 0) {
      int t = 0;
      for (int w = 0; w < kWarps; ++w) t += s_wsum[w];
      s_cnt += t;
    }
    __syncthreads();
  }
  const int cnt = s_cnt;
  for (int k = blockIdx.x; k < cnt; k += gridDim.x) {
    __syncthreads();  // smem of the previous item is no longer read
    mat_item<S, VPK>(args, s_list[k], blockIdx.y, blockIdx.z);
  }
}

template <typename S, int VPK>
__device__ __forceinline__ void mat_item(const MatArgs& args, const int item, const int key_head, const int layer) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int src = 0, dst = 0, n = 0;
  bool init = true;
  int r = item;  // per-request decision arrays index (V2: req-state slot via idx_mapping)
  if (args.idx_map) {
    r = args.idx_map[item];
    if (r < 0) return;
  }
  if (args.mode == 0) {
    src = dst = args.a0[item];
    n = args.a1 ? args.a1[item] : 0;
    init = args.has_init ? args.has_init[item] : true;
  } else {
    const int g = args.layer_group[layer];
    const int* bt = reinterpret_cast<const int*>(args.bt_ptrs[g]) + static_cast<int64_t>(item) * args.bt_stride;
    int src_col, dst_col;
    if (args.mode == 1) {
      src_col = args.a0[r];
      dst_col = args.a1[r];
      if (src_col < 0 || src_col == dst_col) return;
      n = args.a2[r] + 1;
    } else {
      const int acc = args.a0[r];
      src_col = args.a1[r];
      // mode 2 (V1): computed + scheduled - draft; mode 3 (V2): post-step computed - acc + 1
      const int running = args.mode == 2 ? args.a3[r] + args.a2[r] - args.a4[r] : args.a3[r] - acc + 1;
      const int new_computed = running + acc - 1;
      const int aligned = (new_computed / args.block_size) * args.block_size;
      if (aligned < running) return;
      n = aligned - running + 1;
      dst_col = aligned / args.block_size - 1;
    }
    if (src_col < 0 || dst_col < 0) return;
    src = bt[src_col];
    dst = bt[dst_col];
  }
  if (src <= 0 || dst <= 0) return;
  const LogLayout ll = log_layout(args.H, args.HV);
  S* state = reinterpret_cast<S*>(args.layer_state[layer]);
  S* src_state = state + static_cast<int64_t>(src) * args.slot_stride_floats;
  S* dst_state = state + static_cast<int64_t>(dst) * args.slot_stride_floats;
  const int64_t ssm_floats = static_cast<int64_t>(args.HV) * kDimV * kDimK;
  char* src_log = reinterpret_cast<char*>(src_state + ssm_floats);
  char* dst_log = reinterpret_cast<char*>(dst_state + ssm_floats);
  int* src_flag = reinterpret_cast<int*>(src_log + ll.l_off) + key_head * VPK;
  int* dst_flag = reinterpret_cast<int*>(dst_log + ll.l_off) + key_head * VPK;
  int* dst_ctr = reinterpret_cast<int*>(dst_log + ll.c_off) + key_head;
  const int flag = *src_flag;
  n = n < 0 ? 0 : (n > kMaxT ? kMaxT : n);
  const int R = (flag != 0 && init) ? n : 0;
  if (R == 0 && src == dst) {
    __syncthreads();
    if (tid < VPK) dst_flag[tid] = 0;
    if (tid == 0) *dst_ctr = 0;
    return;
  }
  const __nv_bfloat16* log_k = reinterpret_cast<const __nv_bfloat16*>(src_log + ll.k_off);
  const __nv_bfloat16* log_v = reinterpret_cast<const __nv_bfloat16*>(src_log + ll.v_off);
  const __nv_bfloat16* log_ab = reinterpret_cast<const __nv_bfloat16*>(src_log + ll.ab_off);
  const float* a_log = reinterpret_cast<const float*>(args.layer_alog[layer]);
  const void* dt_bias = reinterpret_cast<const void*>(args.layer_dtb[layer]);

  __shared__ __align__(16) S shared_state[2][kChunkV][kDimK];
  __shared__ __align__(16) float shared_k[kMaxT][kDimK];
  __shared__ __nv_bfloat16 shared_v[VPK][kMaxT][kDimV];
  __shared__ float shared_decay[VPK][kMaxT];
  __shared__ float shared_beta[VPK][kMaxT];
#if GSC_CK
  __shared__ float s_kk[kMaxTok][kMaxTok];
  __shared__ CkRep s_rep[VPK];
#endif

#if GSC_CK >= 2
  constexpr bool kMM = sizeof(S) == 2;
  __shared__ __align__(16) __nv_bfloat16 s_vrep[3][8][kVPad];
#if GSC_CK2_UPD == 0
  __shared__ __align__(16) __nv_bfloat16 s_kt[kDimK][kKTPad];
#else
  __nv_bfloat16 (*s_kt)[kKTPad] = nullptr;
#endif
#else
  constexpr bool kMM = false;
#endif
  if constexpr (!kMM)
    copy_state_chunk(&shared_state[0][0][0], src_state + static_cast<int64_t>(key_head * VPK) * kDimV * kDimK, 0, 0, tid);
  if (warp < R) {
    const int j = warp;
    prep_qk(nullptr, log_k + (static_cast<int64_t>(j) * args.H + key_head) * kDimK, 1.0f, lane, nullptr, shared_k[j]);
#pragma unroll
    for (int p = 0; p < VPK; ++p) {
      const int vh = key_head * VPK + p;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        shared_v[p][j][lane + i * 32] = log_v[(static_cast<int64_t>(j) * args.HV + vh) * kDimV + lane + i * 32];
      if (lane == 0)
        prep_gate(__bfloat162float(log_ab[(j * args.HV + vh) * 2]),
                  __bfloat162float(log_ab[(j * args.HV + vh) * 2 + 1]), a_log[vh],
                  load_dt_bias(dt_bias, vh, args.dt_bias_type), &shared_decay[p][j], &shared_beta[p][j]);
    }
  }
  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) rows[row] = warp + row * kWarps;
#if GSC_CK
  __syncthreads();
  ck_gram(shared_k, nullptr, R, 0, tid, s_kk, nullptr);
  __syncthreads();
  if (warp < VPK) ck_coef_rep_par(shared_decay[warp], shared_beta[warp], s_kk, R, &s_rep[warp], lane);
  // visibility: the chunk loop below starts with __syncthreads()
#endif
#if GSC_CK >= 2
  if constexpr (kMM) {
    extern __shared__ __align__(16) unsigned char gsc_mat_dyn[];
    __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(gsc_mat_dyn);
    if (R > 0) {
      mm_build_vec(s_vrep, shared_k, R, nullptr, 0, tid);
#if GSC_CK2_UPD == 0
      mm_build_kt(s_kt, shared_k, R, tid);
#endif
    }
    __syncthreads();  // coefficients, split vectors, k table
#pragma unroll 1
    for (int p = 0; p < VPK; ++p) {
      const int value_head = key_head * VPK + p;
      mm_load_rows(st, reinterpret_cast<const __nv_bfloat16*>(src_state) + static_cast<int64_t>(value_head) * kDimV * kDimK,
                   warp, lane);
      cp_async_wait_group<0>();
      __syncwarp();
      if (R > 0) mm_replay(st, s_vrep, s_kt, s_rep[p], R, &shared_v[p][0][0], warp, lane, shared_k);
      mm_store_rows(reinterpret_cast<__nv_bfloat16*>(dst_state) + static_cast<int64_t>(value_head) * kDimV * kDimK, st,
                    warp, lane);
      __syncwarp();
    }
    __syncthreads();
    if (tid < VPK) dst_flag[tid] = 0;
    if (tid == 0) *dst_ctr = 0;
    return;
  }
#endif
#pragma unroll 1
  for (int p = 0; p < VPK; ++p) {
    const int value_head = key_head * VPK + p;
    const S* sh = src_state + static_cast<int64_t>(value_head) * kDimV * kDimK;
    S* dh = dst_state + static_cast<int64_t>(value_head) * kDimV * kDimK;
#pragma unroll
    for (int chunk = 0; chunk < kNumChunks; ++chunk) {
      cp_async_wait_all();
      __syncthreads();
      if (chunk + 1 < kNumChunks) {
        copy_state_chunk(&shared_state[0][0][0], sh, chunk + 1, (chunk + 1) & 1, tid);
      } else if (p + 1 < VPK) {
        copy_state_chunk(&shared_state[0][0][0], sh + kDimV * kDimK, 0, 0, tid);
      }
#if GSC_CK
      {
        const int seg = lane & 7;
        const int rr = warp * 4 + (lane >> 3);
        const int row[1] = {chunk * kChunkV + rr};
        float hf[1][16];
#pragma unroll
        for (int j = 0; j < 4; ++j) load_h4<S>(&shared_state[chunk & 1][rr][j * 32 + seg * 4], &hf[0][j * 4]);
        if (R > 0) ck_replay<1>(hf, shared_k, &shared_v[p][0][0], s_rep[p], R, row, seg);
#pragma unroll
        for (int j = 0; j < 4; ++j) store_h4<S>(dh + row[0] * kDimK + j * 32 + seg * 4, &hf[0][j * 4]);
        continue;
      }
#endif
      float h[kRowsPerWarp][4];
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        load_h4<S>(&shared_state[chunk & 1][rows[row]][k_base], h[row]);
      }
      for (int j = 0; j < R; ++j)
        token_update<false>(h, shared_k[j], nullptr, shared_v[p][j], shared_decay[p][j], shared_beta[p][j], chunk,
                            rows, k_base, lane, nullptr);
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        store_h4<S>(dh + value * kDimK + k_base, h[row]);
      }
    }
  }
  __syncthreads();
  if (tid < VPK) dst_flag[tid] = 0;
  if (tid == 0) *dst_ctr = 0;
}

#if GSC_QO
static void decode_core(torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias,
            torch::Tensor state_indices, torch::Tensor cu_seqlens, torch::Tensor num_accepted, torch::Tensor state,
            torch::Tensor output_gate, torch::Tensor norm_weight, torch::Tensor out, double scale, double norm_eps,
            bool sigmoid_gate, QoArgs qo) {
  TORCH_CHECK(qo.q == nullptr || (!GSC_PERSIST && !(GSC_CK == 3 && state.scalar_type() == at::kBFloat16)),
              "gdn_state_commit: fused out_proj quant (GSC_QO) needs the deferred_decode_kernel path");
#else
void decode(torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias,
            torch::Tensor state_indices, torch::Tensor cu_seqlens, torch::Tensor num_accepted, torch::Tensor state,
            torch::Tensor output_gate, torch::Tensor norm_weight, torch::Tensor out, double scale, double norm_eps,
            bool sigmoid_gate) {
#endif
  TORCH_CHECK((state.scalar_type() == at::kFloat || state.scalar_type() == at::kBFloat16) && state.dim() == 4 &&
              state.size(2) == kDimV &&
              state.size(3) == kDimK && state.stride(1) == kDimV * kDimK && state.stride(3) == 1);
  TORCH_CHECK(mixed_qkv.scalar_type() == at::kBFloat16 && mixed_qkv.stride(1) == 1);
  TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1 && a_log.scalar_type() == at::kFloat);
  TORCH_CHECK(state_indices.scalar_type() == at::kInt && state_indices.dim() == 2 && state_indices.is_contiguous());
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && cu_seqlens.is_contiguous());
  TORCH_CHECK(num_accepted.scalar_type() == at::kInt && num_accepted.is_contiguous());
  TORCH_CHECK(out.is_contiguous() && output_gate.stride(2) == 1 && output_gate.stride(1) == kDimV);
  const int HV = state.size(1);
  const int H = static_cast<int>((mixed_qkv.size(1) - static_cast<int64_t>(HV) * kDimV) / (2 * kDimK));
  const int vpk = HV / H;
  TORCH_CHECK(HV % H == 0 && (vpk == 1 || vpk == 2), "gdn_state_commit supports HV/H in {1,2}");
  const LogLayout ll = log_layout(H, HV);
  const int es = static_cast<int>(state.element_size());
  const bool sbf = state.scalar_type() == at::kBFloat16;
  TORCH_CHECK(state.stride(0) * es >= static_cast<int64_t>(HV) * kDimV * kDimK * es + ll.bytes,
              "state page too small for the token log");
  const int dtt = dt_bias.scalar_type() == at::kFloat ? kDtBiasFloat32
                  : (dt_bias.scalar_type() == at::kBFloat16 ? kDtBiasBFloat16 : kDtBiasFloat16);
  const Strides s{mixed_qkv.stride(0), a.stride(0), b.stride(0), output_gate.stride(0), state.stride(0)};
  const int n = state_indices.size(0);
  if (n == 0) return;
  const dim3 grid(n, HV);
  auto stream = c10::cuda::getCurrentCUDAStream();
  static bool attr_set[2][2][2] = {};
  const bool wbf16 = norm_weight.scalar_type() == at::kBFloat16;
#ifndef GSC_PERSIST
#define GSC_PERSIST 0
#endif
  static int resident[2][2][2] = {};
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  static int num_sms = 0;
  if (num_sms == 0) C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
#define GSC_ARGS(ST_)                                                                                                  \
      reinterpret_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),                                             \
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()), \
      a_log.data_ptr<float>(), dt_bias.data_ptr(), state_indices.data_ptr<int>(),                               \
      static_cast<int>(state_indices.size(1)), cu_seqlens.data_ptr<int>(), num_accepted.data_ptr<int>(),        \
      reinterpret_cast<ST_*>(state.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(output_gate.data_ptr()),                  \
      norm_weight.data_ptr(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, HV, dtt, wbf16,              \
      static_cast<float>(scale), static_cast<float>(norm_eps), s
#define GSC_LAUNCH(ST_, VPK_, SIG_)                                                                             \
  constexpr int kDyn = kNumChunks * kChunkV * kDimK * static_cast<int>(sizeof(ST_));                            \
  constexpr int kB = sizeof(ST_) == 2;                                                                                    \
  if (GSC_PERSIST) {                                                                                              \
    if (!resident[kB][VPK_ - 1][SIG_]) {                                                                              \
      C10_CUDA_CHECK(cudaFuncSetAttribute(persistent_decode_kernel<ST_, VPK_, SIG_>,                                   \
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, kDyn));                    \
      C10_CUDA_CHECK(cudaFuncSetAttribute(persistent_decode_kernel<ST_, VPK_, SIG_>,                                   \
                                          cudaFuncAttributePreferredSharedMemoryCarveout, 100));                  \
      int per_sm = 0;                                                                                             \
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, persistent_decode_kernel<ST_, VPK_, SIG_>, \
                                                                   kThreads, kDyn));                              \
      resident[kB][VPK_ - 1][SIG_] = per_sm * num_sms;                                                                \
    }                                                                                                             \
    const int items = n * HV;                                                                                     \
    const int g = items < resident[kB][VPK_ - 1][SIG_] ? items : resident[kB][VPK_ - 1][SIG_];                            \
    persistent_decode_kernel<ST_, VPK_, SIG_><<<g, kThreads, kDyn, stream>>>(GSC_ARGS(ST_), n);                            \
  } else {                                                                                                        \
    if (!attr_set[kB][VPK_ - 1][SIG_]) {                                                                              \
      C10_CUDA_CHECK(cudaFuncSetAttribute(deferred_decode_kernel<ST_, VPK_, SIG_>,                                     \
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, kDyn));                    \
      C10_CUDA_CHECK(cudaFuncSetAttribute(deferred_decode_kernel<ST_, VPK_, SIG_>,                                     \
                                          cudaFuncAttributePreferredSharedMemoryCarveout, 100));                  \
      attr_set[kB][VPK_ - 1][SIG_] = true;                                                                            \
    }                                                                                                             \
    deferred_decode_kernel<ST_, VPK_, SIG_><<<grid, kThreads, kDyn, stream>>>(GSC_ARGS(ST_) GSC_QO_ARG);                              \
  }
#if GSC_GB
#if GB_KH
  if (!sbf && vpk == 2 && (GB_KH == 1 || n >= GB_KH_MIN)) {  // GB300 fp32 kernel, one CTA per (request, key head)
    static bool gbk_attr[2] = {};
    const dim3 kgrid(n, H);
#define GSC_GBK_LAUNCH(SIG_)                                                                                      \
    if (!gbk_attr[SIG_]) {                                                                                        \
      C10_CUDA_CHECK(cudaFuncSetAttribute(gb_kh_kernel<SIG_>, cudaFuncAttributeMaxDynamicSharedMemorySize, kGbkDyn)); \
      C10_CUDA_CHECK(cudaFuncSetAttribute(gb_kh_kernel<SIG_>, cudaFuncAttributePreferredSharedMemoryCarveout, 100)); \
      gbk_attr[SIG_] = true;                                                                                      \
    }                                                                                                             \
    gb_kh_kernel<SIG_><<<kgrid, kGbThreads, kGbkDyn, stream>>>(GSC_ARGS(float));
    if (sigmoid_gate) { GSC_GBK_LAUNCH(true); } else { GSC_GBK_LAUNCH(false); }
#undef GSC_GBK_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#endif
  if (!sbf) {  // GB300 fp32 kernel (bit-exact with GSC_CK=0)
    static bool gb_attr[2][2] = {};
#define GSC_GB_LAUNCH(VPK_, SIG_)                                                                                 \
    if (!gb_attr[VPK_ - 1][SIG_]) {                                                                               \
      C10_CUDA_CHECK(cudaFuncSetAttribute(gb_decode_kernel<VPK_, SIG_>, cudaFuncAttributeMaxDynamicSharedMemorySize, kGbDyn)); \
      C10_CUDA_CHECK(cudaFuncSetAttribute(gb_decode_kernel<VPK_, SIG_>, cudaFuncAttributePreferredSharedMemoryCarveout, 100)); \
      gb_attr[VPK_ - 1][SIG_] = true;                                                                             \
    }                                                                                                             \
    gb_decode_kernel<VPK_, SIG_><<<grid, kGbThreads, kGbDyn, stream>>>(GSC_ARGS(float));
    if (vpk == 2) {
      if (sigmoid_gate) { GSC_GB_LAUNCH(2, true); } else { GSC_GB_LAUNCH(2, false); }
    } else {
      if (sigmoid_gate) { GSC_GB_LAUNCH(1, true); } else { GSC_GB_LAUNCH(1, false); }
    }
#undef GSC_GB_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#endif
#if GSC_CK == 3
  if (sbf) {
    static bool ck3_attr[2][2] = {};
    constexpr int kDyn3 = kDimV * kDimK * 2;
#define GSC_CK3_LAUNCH(VPK_, SIG_)                                                                                \
    if (!ck3_attr[VPK_ - 1][SIG_]) {                                                                              \
      C10_CUDA_CHECK(cudaFuncSetAttribute(ck3_decode_kernel<VPK_, SIG_>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDyn3)); \
      C10_CUDA_CHECK(cudaFuncSetAttribute(ck3_decode_kernel<VPK_, SIG_>, cudaFuncAttributePreferredSharedMemoryCarveout, 100)); \
      ck3_attr[VPK_ - 1][SIG_] = true;                                                                            \
    }                                                                                                             \
    ck3_decode_kernel<VPK_, SIG_><<<grid, kThreads, kDyn3, stream>>>(                                             \
        reinterpret_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()), a_log.data_ptr<float>(), dt_bias.data_ptr(),         \
        state_indices.data_ptr<int>(), static_cast<int>(state_indices.size(1)), cu_seqlens.data_ptr<int>(),        \
        num_accepted.data_ptr<int>(), reinterpret_cast<__nv_bfloat16*>(state.data_ptr()),                          \
        reinterpret_cast<const __nv_bfloat16*>(output_gate.data_ptr()), norm_weight.data_ptr(),                    \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, HV, dtt, wbf16, static_cast<float>(scale),            \
        static_cast<float>(norm_eps), s);
    if (vpk == 2) {
      if (sigmoid_gate) { GSC_CK3_LAUNCH(2, true); } else { GSC_CK3_LAUNCH(2, false); }
    } else {
      if (sigmoid_gate) { GSC_CK3_LAUNCH(1, true); } else { GSC_CK3_LAUNCH(1, false); }
    }
#undef GSC_CK3_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#endif
  if (sbf) {
    if (vpk == 2) {
      if (sigmoid_gate) { GSC_LAUNCH(__nv_bfloat16, 2, true); } else { GSC_LAUNCH(__nv_bfloat16, 2, false); }
    } else {
      if (sigmoid_gate) { GSC_LAUNCH(__nv_bfloat16, 1, true); } else { GSC_LAUNCH(__nv_bfloat16, 1, false); }
    }
  } else {
    if (vpk == 2) {
      if (sigmoid_gate) { GSC_LAUNCH(float, 2, true); } else { GSC_LAUNCH(float, 2, false); }
    } else {
      if (sigmoid_gate) { GSC_LAUNCH(float, 1, true); } else { GSC_LAUNCH(float, 1, false); }
    }
  }
#undef GSC_ARGS
#undef GSC_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#if GSC_QO
void decode(torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias,
            torch::Tensor state_indices, torch::Tensor cu_seqlens, torch::Tensor num_accepted, torch::Tensor state,
            torch::Tensor output_gate, torch::Tensor norm_weight, torch::Tensor out, double scale, double norm_eps,
            bool sigmoid_gate) {
  decode_core(mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens, num_accepted, state, output_gate,
              norm_weight, out, scale, norm_eps, sigmoid_gate, QoArgs{nullptr, nullptr, 0, 0, 0, 0});
}
// decode + MXFP8 quantization of out ([T, HV, 128] bf16) into q ([>= T, HV*128] e4m3 / uint8, row-contiguous) and
// sf (128x4-swizzled e8m0 scales of pm = roundup(T, 128) rows, psc padded columns); padding rows zeroed (see QoArgs).
void decode_qo(torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias,
               torch::Tensor state_indices, torch::Tensor cu_seqlens, torch::Tensor num_accepted, torch::Tensor state,
               torch::Tensor output_gate, torch::Tensor norm_weight, torch::Tensor out, double scale, double norm_eps,
               bool sigmoid_gate, torch::Tensor q, torch::Tensor sf, int64_t psc) {
  const int64_t T = out.size(0);
  const int64_t K = out.size(1) * out.size(2);
  const int64_t pm = (T + 127) / 128 * 128;
  TORCH_CHECK(out.dim() == 3 && out.size(2) == kDimV && out.is_contiguous());
  TORCH_CHECK(q.element_size() == 1 && q.dim() == 2 && q.size(1) == K && q.stride(1) == 1 && q.size(0) >= T);
  TORCH_CHECK(sf.element_size() == 1 && sf.is_contiguous() && sf.numel() >= pm * psc && psc == (K / 32 + 3) / 4 * 4);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(sf.data_ptr()) % 4 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 4 == 0 &&
              q.stride(0) % 4 == 0);
  TORCH_CHECK(pm <= 32768 && q.stride(0) * pm < (int64_t(1) << 31) && pm * psc < (int64_t(1) << 31));
  QoArgs qo{reinterpret_cast<uint8_t*>(q.data_ptr()), reinterpret_cast<uint8_t*>(sf.data_ptr()),
            static_cast<int>(q.stride(0)), static_cast<int>(psc), static_cast<int>(T), static_cast<int>(pm)};
  decode_core(mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens, num_accepted, state, output_gate,
              norm_weight, out, scale, norm_eps, sigmoid_gate, qo);
}
#endif

static void materialize_impl(int64_t compact, int64_t mode, int64_t num_items, int64_t num_layers, torch::Tensor a0,
                 c10::optional<torch::Tensor> a1,
                 c10::optional<torch::Tensor> a2, c10::optional<torch::Tensor> a3, c10::optional<torch::Tensor> a4,
                 c10::optional<torch::Tensor> has_init, c10::optional<torch::Tensor> idx_map,
                 c10::optional<torch::Tensor> bt_ptrs, int64_t bt_stride,
                 int64_t block_size, torch::Tensor layer_state, torch::Tensor layer_alog, torch::Tensor layer_dtb,
                 torch::Tensor layer_group, int64_t slot_stride_floats, int64_t H, int64_t HV, int64_t dt_bias_type,
                 int64_t state_bf16) {
  if (num_items <= 0 || num_layers <= 0) return;
  auto ip = [](const c10::optional<torch::Tensor>& t) -> const int* {
    if (!t.has_value()) return nullptr;
    TORCH_CHECK(t->scalar_type() == at::kInt && t->is_contiguous());
    return t->data_ptr<int>();
  };
  TORCH_CHECK(a0.scalar_type() == at::kInt && a0.is_contiguous());
  MatArgs m;
  m.mode = static_cast<int>(mode);
  m.a0 = a0.data_ptr<int>(); m.a1 = ip(a1); m.a2 = ip(a2); m.a3 = ip(a3); m.a4 = ip(a4);
  m.has_init = nullptr;
  if (has_init.has_value()) {
    TORCH_CHECK(has_init->scalar_type() == at::kBool && has_init->is_contiguous());
    m.has_init = has_init->data_ptr<bool>();
  }
  m.idx_map = ip(idx_map);
  m.bt_ptrs = bt_ptrs.has_value() ? bt_ptrs->data_ptr<int64_t>() : nullptr;
  m.bt_stride = bt_stride;
  m.block_size = static_cast<int>(block_size);
  m.layer_state = layer_state.data_ptr<int64_t>();
  m.layer_alog = layer_alog.data_ptr<int64_t>();
  m.layer_dtb = layer_dtb.data_ptr<int64_t>();
  m.layer_group = layer_group.data_ptr<int>();
  m.slot_stride_floats = slot_stride_floats;
  m.H = static_cast<int>(H); m.HV = static_cast<int>(HV); m.dt_bias_type = static_cast<int>(dt_bias_type);
  m.state_bf16 = static_cast<int>(state_bf16);
  m.num_items = static_cast<int>(num_items);
  // [rubin-ck] compact mode only for the per-request copy modes (1-3) and item counts that fit the smem list
  m.compact = (compact > 0 && mode != 0 && num_items <= kMatMaxItems) ? static_cast<int>(compact) : 0;
  const dim3 grid(m.compact ? std::min<int64_t>(m.compact, num_items) : num_items, H, num_layers);
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (m.state_bf16) {
    const int dyn = GSC_CK >= 2 ? kDimV * kDimK * 2 : 0;
    static bool mat_attr = false;
    if (dyn > 0 && !mat_attr) {  // static + dynamic > 48 KB needs the opt-in
      C10_CUDA_CHECK(cudaFuncSetAttribute(materialize_kernel<__nv_bfloat16, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
      C10_CUDA_CHECK(cudaFuncSetAttribute(materialize_kernel<__nv_bfloat16, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
      C10_CUDA_CHECK(cudaFuncSetAttribute(materialize_compact_kernel<__nv_bfloat16, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
      C10_CUDA_CHECK(cudaFuncSetAttribute(materialize_compact_kernel<__nv_bfloat16, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
      mat_attr = true;
    }
    if (m.compact) {
      if (HV / H == 2) materialize_compact_kernel<__nv_bfloat16, 2><<<grid, kThreads, dyn, stream>>>(m);
      else materialize_compact_kernel<__nv_bfloat16, 1><<<grid, kThreads, dyn, stream>>>(m);
    } else {
      if (HV / H == 2) materialize_kernel<__nv_bfloat16, 2><<<grid, kThreads, dyn, stream>>>(m);
      else materialize_kernel<__nv_bfloat16, 1><<<grid, kThreads, dyn, stream>>>(m);
    }
  } else if (m.compact) {
    if (HV / H == 2) materialize_compact_kernel<float, 2><<<grid, kThreads, 0, stream>>>(m);
    else materialize_compact_kernel<float, 1><<<grid, kThreads, 0, stream>>>(m);
  } else {
    if (HV / H == 2) materialize_kernel<float, 2><<<grid, kThreads, 0, stream>>>(m);
    else materialize_kernel<float, 1><<<grid, kThreads, 0, stream>>>(m);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void materialize(int64_t mode, int64_t num_items, int64_t num_layers, torch::Tensor a0, c10::optional<torch::Tensor> a1,
                 c10::optional<torch::Tensor> a2, c10::optional<torch::Tensor> a3, c10::optional<torch::Tensor> a4,
                 c10::optional<torch::Tensor> has_init, c10::optional<torch::Tensor> idx_map,
                 c10::optional<torch::Tensor> bt_ptrs, int64_t bt_stride,
                 int64_t block_size, torch::Tensor layer_state, torch::Tensor layer_alog, torch::Tensor layer_dtb,
                 torch::Tensor layer_group, int64_t slot_stride_floats, int64_t H, int64_t HV, int64_t dt_bias_type,
                 int64_t state_bf16) {
  materialize_impl(0, mode, num_items, num_layers, a0, a1, a2, a3, a4, has_init, idx_map, bt_ptrs, bt_stride, block_size,
                   layer_state, layer_alog, layer_dtb, layer_group, slot_stride_floats, H, HV, dt_bias_type, state_bf16);
}

// [rubin-ck] same as materialize, with the item-compacting grid (compact = CTAs along x per (key head, layer)).
void materialize_compact(int64_t compact, int64_t mode, int64_t num_items, int64_t num_layers, torch::Tensor a0,
                         c10::optional<torch::Tensor> a1, c10::optional<torch::Tensor> a2,
                         c10::optional<torch::Tensor> a3, c10::optional<torch::Tensor> a4,
                         c10::optional<torch::Tensor> has_init, c10::optional<torch::Tensor> idx_map,
                         c10::optional<torch::Tensor> bt_ptrs, int64_t bt_stride, int64_t block_size,
                         torch::Tensor layer_state, torch::Tensor layer_alog, torch::Tensor layer_dtb,
                         torch::Tensor layer_group, int64_t slot_stride_floats, int64_t H, int64_t HV,
                         int64_t dt_bias_type, int64_t state_bf16) {
  materialize_impl(compact, mode, num_items, num_layers, a0, a1, a2, a3, a4, has_init, idx_map, bt_ptrs, bt_stride,
                   block_size, layer_state, layer_alog, layer_dtb, layer_group, slot_stride_floats, H, HV,
                   dt_bias_type, state_bf16);
}

// [rubin-ck] materialize kernel (bf16 state, VPK 2; compact != 0 -> the compact kernel):
// blocks/SM * 1e6 + regs * 1e3 + static smem KB (+ local * 1e9)
int64_t occupancy_mat(int64_t compact) {
  constexpr int kDyn = kDimV * kDimK * 2;
  auto fn = compact ? materialize_compact_kernel<__nv_bfloat16, 2> : materialize_kernel<__nv_bfloat16, 2>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kDyn));
  int n = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, kThreads, kDyn));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, fn));
  return static_cast<int64_t>(fa.localSizeBytes) * 1000000000LL + n * 1000000 + fa.numRegs * 1000 + fa.sharedSizeBytes / 1024;
}

int64_t occupancy() {
  constexpr int kDyn = kNumChunks * kChunkV * kDimK * 4;
  C10_CUDA_CHECK(cudaFuncSetAttribute(deferred_decode_kernel<float, 2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDyn));
  C10_CUDA_CHECK(cudaFuncSetAttribute(deferred_decode_kernel<float, 2, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  int n = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, deferred_decode_kernel<float, 2, false>, kThreads, kDyn));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, deferred_decode_kernel<float, 2, false>));
  return n * 1000000 + fa.numRegs * 1000 + fa.sharedSizeBytes / 1024;
}

// bf16-state decode kernel: blocks/SM * 1e6 + regs * 1e3 + static smem KB (+ local bytes * 1e9)
int64_t occupancy_bf16() {
  constexpr int kDyn = kNumChunks * kChunkV * kDimK * 2;
  auto fn = deferred_decode_kernel<__nv_bfloat16, 2, false>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kDyn));
  C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  int n = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, kThreads, kDyn));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, fn));
  return static_cast<int64_t>(fa.localSizeBytes) * 1000000000LL + n * 1000000 + fa.numRegs * 1000 + fa.sharedSizeBytes / 1024;
}

int64_t occupancy_ck3() {
#if GSC_CK == 3
  constexpr int kDyn = kDimV * kDimK * 2;
  auto fn = ck3_decode_kernel<2, false>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kDyn));
  C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  int n = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, kThreads, kDyn));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, fn));
  return static_cast<int64_t>(fa.localSizeBytes) * 1000000000LL + n * 1000000 + fa.numRegs * 1000 + fa.sharedSizeBytes / 1024;
#else
  return -1;
#endif
}

// GB300 fp32 kernel: local bytes * 1e9 + blocks/SM * 1e6 + regs * 1e3 + static smem KB (-1 if not built)
int64_t occupancy_gb() {
#if GSC_GB
#if GB_KH
  auto fn = gb_kh_kernel<false>;
  constexpr int dyn = kGbkDyn;
#else
  auto fn = gb_decode_kernel<2, false>;
  constexpr int dyn = kGbDyn;
#endif
  C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
  C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  int n = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, kGbThreads, dyn));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, fn));
  return static_cast<int64_t>(fa.localSizeBytes) * 1000000000LL + n * 1000000 + fa.numRegs * 1000 + fa.sharedSizeBytes / 1024;
#else
  return -1;
#endif
}

int64_t log_bytes(int64_t H, int64_t HV) { return log_layout(static_cast<int>(H), static_cast<int>(HV)).bytes; }

}  // namespace gsc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &gsc::decode);
#if GSC_QO
  m.def("decode_qo", &gsc::decode_qo);
#endif
  m.def("materialize", &gsc::materialize);
  m.def("materialize_compact", &gsc::materialize_compact);
  m.def("occupancy_mat", &gsc::occupancy_mat);
  m.def("log_bytes", &gsc::log_bytes);
  m.def("occupancy", &gsc::occupancy);
  m.def("occupancy_bf16", &gsc::occupancy_bf16);
  m.def("occupancy_ck3", &gsc::occupancy_ck3);
  m.def("occupancy_gb", &gsc::occupancy_gb);
}
