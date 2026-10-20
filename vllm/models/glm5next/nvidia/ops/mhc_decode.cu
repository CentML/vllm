// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3-Flash mHC sublayer boundary at decode token counts (<= 32 rows):
// the "pre" half (coefficients + Sinkhorn + collapse + input RMSNorm) that
// follows TileLang's fused post + fp32 prenorm GEMM (mhc_fused_tilelang).
// hc_mult 4, hidden 4096, 24 mix outputs.
//
// TileLang's mhc_pre_big_fuse_with_norm runs one 96-thread CTA per token: one
// warp does post / Sinkhorn while 64 threads collapse the four 4096-wide
// streams (32 KB) and apply the RMSNorm in two passes. At decode sizes that
// is ~4.1 us per boundary on VR200 regardless of the token count (latency of
// 64 threads walking 32 KB twice), i.e. ~0.37 ms of an 89-boundary decode
// step.
//
// This kernel keeps one CTA per token but gives the collapse 16 warps: each
// of the 512 collapse threads owns one 16 B vector (8 columns) of every
// stream, keeps its 8 outputs in registers across the RMSNorm reduction
// (warp shuffles + one named barrier), and writes the normalized row once.
// Warp 16 runs post / softmax / Sinkhorn concurrently (it only produces
// post_mix / comb_mix, which the next boundary consumes).
//
// Numerics: the split sums, rms, sigmoid, softmax and Sinkhorn use the same
// operation order as the TileLang kernel (serial split order, expf, IEEE
// division, the TileLang butterfly shapes; taken over from mhc_prefill.cu,
// whose pre was checked bit-identical): post_mix / comb_mix are bit-identical
// on the same partials. The collapse is the same FMA chain (pre0 * r0, then
// fma over r1..r3) rounded to bf16; only the RMSNorm sum of squares is reduced
// in a different order, so layer_input is within 1 bf16 ulp.
//
// PDL: waits on the previous grid (griddepcontrol.wait) before reading the
// partials and triggers dependents at the end, like the TileLang kernels.
//
// Built without -use_fast_math (see mhc_decode.py).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace glm5next_mhcd {

constexpr int kHC = 4;
constexpr int kMix = kHC * (kHC + 2);  // 24
constexpr int kHidden = 4096;
constexpr int kK = kHC * kHidden;          // 16384
constexpr int kCollapseWarps = 16;         // 512 threads x 8 columns = 4096
constexpr int kCollapseThreads = kCollapseWarps * 32;
constexpr int kThreads = kCollapseThreads + 32;  // + the Sinkhorn warp
static_assert(kCollapseThreads * 8 == kHidden, "one uint4 per thread/stream");
// TileLang's fused post + GEMM uses 8 or 16 splits; mhc_prefill.cu's GEMM-only
// mode (used here for 33-256 tokens) 64.
constexpr int kMaxSplits = 64;

struct Params {
  const float* part;                  // [S, M, 24] prenorm GEMM partials
  const float* sq;                    // [S, M] sum of squares partials
  const float* hc_scale;              // [3]
  const float* hc_base;               // [24]
  const __nv_bfloat16* residual;      // [M, 4, 4096] (post output r')
  const __nv_bfloat16* norm_w;        // [4096]
  float* post_out;                    // [M, 4]
  float* comb_out;                    // [M, 16]
  __nv_bfloat16* layer_input;         // [M, 4096]
  int M, S, sk_iters;
  float rms_eps, pre_eps, sk_eps, post_mult, norm_eps;
};

__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo,
                                              __nv_bfloat16 hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(lo)) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}
__device__ __forceinline__ float bf(uint32_t packed, int half) {
  return __uint_as_float(half ? (packed & 0xffff0000u) : (packed << 16));
}
__device__ __forceinline__ float sigmoid_tl(float z) {
  return 1.0f / (1.0f + expf(0.0f - z));
}
__device__ __forceinline__ void named_barrier(int id, int threads) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(threads) : "memory");
}

// MODE 0: everything (512 collapse threads + the Sinkhorn warp); 1: collapse +
// RMSNorm only (layer_input; 512 threads); 2: post / comb only (one warp).
// 1 and 2 let the caller run Sinkhorn on a side stream: only the next boundary
// reads post / comb, so it need not sit on the critical path.
template <int MODE>
__global__ void __launch_bounds__(MODE == 2 ? 32 : (MODE == 1 ? kCollapseThreads : kThreads), 1)
    mhc_pre_decode_kernel(Params p) {
  __shared__ float mix_s[kMix];
  __shared__ float sq_s;
  __shared__ float red_s[kCollapseWarps];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int64_t m = blockIdx.x;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif

  // ---- 0. sum the S split partials in split order (as TileLang does). All
  // S loads are issued before the (ordered) sum: a dependent load chain over
  // S splits costs S L2 round trips.
  // MODE 1 needs only the pre logits and the sum of squares.
  if (tid <= kMix && (MODE != 1 || tid < kHC || tid == kMix)) {
    const float* src = tid < kMix ? p.part + m * kMix + tid : p.sq + m;
    const int64_t stride = tid < kMix ? static_cast<int64_t>(p.M) * kMix : p.M;
    float vals[kMaxSplits];
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s)
      vals[s] = s < p.S ? __ldcg(src + s * stride) : 0.f;
    float v = 0.f;
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s)
      if (s < p.S) v = v + vals[s];
    if (tid < kMix)
      mix_s[tid] = v;
    else
      sq_s = v;
  }
  __syncthreads();
  const float rms = rsqrtf(sq_s / 16384.0f + p.rms_eps);

  if (MODE == 2 || (MODE == 0 && warp == kCollapseWarps)) {
    // ---- 1. post + softmax + Sinkhorn: lane (j, k) = (lane >> 2 & 3, lane & 3)
    // owns comb[j][k] (one element per lane, as TileLang spreads them; lanes
    // 16..31 mirror 0..15 so the full-mask shuffles stay valid). Same
    // reduction shapes as the TileLang kernel: rows xor 2 then 1 over k,
    // columns xor 8 then 4 over j -> bit-identical sums.
    const int j = (lane >> 2) & 3, k = lane & 3;
    constexpr unsigned kFull = 0xffffffffu;
    auto row_sum = [&](float v) {
      const float s = v + __shfl_xor_sync(kFull, v, 2);
      return s + __shfl_xor_sync(kFull, s, 1);
    };
    auto col_sum = [&](float v) {
      const float s = v + __shfl_xor_sync(kFull, v, 8);
      return s + __shfl_xor_sync(kFull, s, 4);
    };
    float c = ((mix_s[8 + 4 * j + k] * rms) * p.hc_scale[2]) +
              p.hc_base[8 + 4 * j + k];
    float mx = fmaxf(c, __shfl_xor_sync(kFull, c, 2));
    mx = fmaxf(mx, __shfl_xor_sync(kFull, mx, 1));
    c = expf(c - mx);
    c = (c / row_sum(c)) + p.sk_eps;
    c = c / (col_sum(c) + p.sk_eps);
    // A NaN row (CUDA-graph padding) is NaN either way; skip the remaining
    // iterations as the TileLang kernel does (IEEE division is slow on NaN).
    if (isfinite(rms)) {
      for (int it = 0; it < p.sk_iters - 1; ++it) {
        c = c / (row_sum(c) + p.sk_eps);
        c = c / (col_sum(c) + p.sk_eps);
      }
    }
    if (lane < 16) {
      p.comb_out[m * 16 + lane] = c;
      if (k == 0)
        p.post_out[m * kHC + j] =
            sigmoid_tl(((mix_s[kHC + j] * rms) * p.hc_scale[1]) +
                       p.hc_base[kHC + j]) *
            p.post_mult;
    }
  } else if (MODE != 2) {
    // ---- 2. collapse + RMSNorm: thread v owns columns [8v, 8v + 8).
    float pre[kHC];
#pragma unroll
    for (int j = 0; j < kHC; ++j)
      pre[j] = sigmoid_tl((mix_s[j] * rms) * p.hc_scale[0] + p.hc_base[j]) +
               p.pre_eps;
    const int v = tid;
    const uint4* rrow = reinterpret_cast<const uint4*>(p.residual + m * kK);
    uint4 r[kHC];
#pragma unroll
    for (int j = 0; j < kHC; ++j) r[j] = __ldcg(rrow + j * (kHidden / 8) + v);
    const uint4 w4 = __ldg(reinterpret_cast<const uint4*>(p.norm_w) + v);
    const uint32_t w0[4] = {r[0].x, r[0].y, r[0].z, r[0].w};
    const uint32_t w1[4] = {r[1].x, r[1].y, r[1].z, r[1].w};
    const uint32_t w2[4] = {r[2].x, r[2].y, r[2].z, r[2].w};
    const uint32_t w3[4] = {r[3].x, r[3].y, r[3].z, r[3].w};
    float o[8];
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int h = e >> 1, hf = e & 1;
      float t = __fmul_rn(pre[0], bf(w0[h], hf));
      t = __fmaf_rn(pre[1], bf(w1[h], hf), t);
      t = __fmaf_rn(pre[2], bf(w2[h], hf), t);
      t = __fmaf_rn(pre[3], bf(w3[h], hf), t);
      ss = __fmaf_rn(t, t, ss);
      // TileLang stores the unnormalized row as bf16 before scaling.
      o[e] = __bfloat162float(__float2bfloat16_rn(t));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      ss += __shfl_xor_sync(0xffffffffu, ss, off);
    if (lane == 0) red_s[warp] = ss;
    named_barrier(1, kCollapseThreads);
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < kCollapseWarps; ++w) tot += red_s[w];
    const float rs = rsqrtf(tot / 4096.0f + p.norm_eps);
    const uint32_t ww[4] = {w4.x, w4.y, w4.z, w4.w};
    uint32_t res[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const float y0 = __fmul_rn(__fmul_rn(o[2 * h], rs), bf(ww[h], 0));
      const float y1 = __fmul_rn(__fmul_rn(o[2 * h + 1], rs), bf(ww[h], 1));
      res[h] = pack_bf16(__float2bfloat16_rn(y0), __float2bfloat16_rn(y1));
    }
    reinterpret_cast<uint4*>(p.layer_input + m * kHidden)[v] =
        make_uint4(res[0], res[1], res[2], res[3]);
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

#define GLM_MHCD_CHECK(t, code)                                \
  TVM_FFI_ICHECK(encode_dlpack_dtype((t).dtype()) == (code) && \
                 (t).IsContiguous())                           \
      << #t " must be a contiguous " #code " tensor"

void mhc_pre_decode(TensorView part, TensorView sq, TensorView hc_scale,
                    TensorView hc_base, TensorView residual, TensorView norm_w,
                    TensorView post_out, TensorView comb_out,
                    TensorView layer_input, double rms_eps, double pre_eps,
                    double sk_eps, double post_mult, int64_t sk_iters,
                    double norm_eps, int64_t pdl, int64_t mode) {
  GLM_MHCD_CHECK(part, float32_code);
  GLM_MHCD_CHECK(sq, float32_code);
  GLM_MHCD_CHECK(hc_scale, float32_code);
  GLM_MHCD_CHECK(hc_base, float32_code);
  GLM_MHCD_CHECK(residual, bfloat16_code);
  GLM_MHCD_CHECK(norm_w, bfloat16_code);
  GLM_MHCD_CHECK(post_out, float32_code);
  GLM_MHCD_CHECK(comb_out, float32_code);
  GLM_MHCD_CHECK(layer_input, bfloat16_code);
  TVM_FFI_ICHECK(mode >= 0 && mode <= 2) << "mode 0 (all), 1 (collapse), 2 (post/comb)";
  const int64_t M = residual.size(0);
  TVM_FFI_ICHECK(residual.numel() == M * kK) << "residual must be [M, 4, 4096]";
  TVM_FFI_ICHECK(part.ndim() == 3 && part.size(1) == M && part.size(2) == kMix)
      << "part must be [S, M, 24]";
  const int64_t S = part.size(0);
  TVM_FFI_ICHECK(sq.numel() == S * M) << "sq must be [S, M]";
  TVM_FFI_ICHECK(S >= 1 && S <= kMaxSplits) << "at most 64 splits";
  TVM_FFI_ICHECK(hc_scale.numel() == 3 && hc_base.numel() == kMix);
  TVM_FFI_ICHECK(norm_w.numel() == kHidden);
  // Mode 1 writes only layer_input, mode 2 only post / comb.
  TVM_FFI_ICHECK(mode == 1 ||
                 (post_out.numel() == M * kHC && comb_out.numel() == M * 16));
  TVM_FFI_ICHECK(mode == 2 || layer_input.numel() == M * kHidden);
  TVM_FFI_ICHECK(sk_iters >= 1);
  TVM_FFI_ICHECK(M <= 65535) << "one CTA per token";
  if (M == 0) return;

  Params p;
  p.part = static_cast<const float*>(part.data_ptr());
  p.sq = static_cast<const float*>(sq.data_ptr());
  p.hc_scale = static_cast<const float*>(hc_scale.data_ptr());
  p.hc_base = static_cast<const float*>(hc_base.data_ptr());
  p.residual = static_cast<const __nv_bfloat16*>(residual.data_ptr());
  p.norm_w = static_cast<const __nv_bfloat16*>(norm_w.data_ptr());
  p.post_out = static_cast<float*>(post_out.data_ptr());
  p.comb_out = static_cast<float*>(comb_out.data_ptr());
  p.layer_input = static_cast<__nv_bfloat16*>(layer_input.data_ptr());
  p.M = static_cast<int>(M);
  p.S = static_cast<int>(S);
  p.sk_iters = static_cast<int>(sk_iters);
  p.rms_eps = static_cast<float>(rms_eps);
  p.pre_eps = static_cast<float>(pre_eps);
  p.sk_eps = static_cast<float>(sk_eps);
  p.post_mult = static_cast<float>(post_mult);
  p.norm_eps = static_cast<float>(norm_eps);

  ffi::CUDADeviceGuard device_guard(residual.device().device_id);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(M));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = get_stream(residual.device());
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaError_t e;
  if (mode == 1) {
    cfg.blockDim = dim3(kCollapseThreads);
    e = cudaLaunchKernelEx(&cfg, mhc_pre_decode_kernel<1>, p);
  } else if (mode == 2) {
    cfg.blockDim = dim3(32);
    e = cudaLaunchKernelEx(&cfg, mhc_pre_decode_kernel<2>, p);
  } else {
    e = cudaLaunchKernelEx(&cfg, mhc_pre_decode_kernel<0>, p);
  }
  TVM_FFI_ICHECK(e == cudaSuccess)
      << "mhc_pre_decode launch failed: " << cudaGetErrorString(e);
}

}  // namespace glm5next_mhcd

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_pre_decode, glm5next_mhcd::mhc_pre_decode);
