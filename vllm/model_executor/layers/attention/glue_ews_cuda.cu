// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// gb300 glue: CUDA version of the full-attention QKV prologue (fused_qkv_prologue._ews_qkv_kernel): QK-RMSNorm
// (Gemma, w + 1) + partial NeoX RoPE (64 of 256 dims, optional M-RoPE sections) + FP8 query quant + FP8 paged K/V
// cache write (+ optional contiguous gate copy), ONE WARP per (token, head) instead of a 4-warp Triton program.
//
// Bit-identical to the Triton kernel (its PTX on sm_103a, Triton 3.8): every value is produced by the same sequence of
// IEEE fp32 operations:
//   * sum of squares: Triton's [256] tile is 2 elements / thread, 32 lanes, 4 warps: per-thread fma(x0, x0, x1 * x1),
//     warp butterfly xor 16, 8, 4, 2, 1 (own + partner), then (W0 + W2) + (W1 + W3) across the 4 warps. Here one warp
//     holds the 4 "Triton warps" as 4 register groups and runs the same 4 butterflies, then the same final adds;
//   * var = div.full.f32(sum, 256), rstd = rsqrt.approx.ftz.f32(var + eps) (inline PTX, as Triton emits);
//   * y = bf16((rstd * x) * (w + 1)); RoPE o1 = bf16(fma(x1, cos, -(x2 * sin))), o2 = bf16(fma(x2, cos, x1 * sin));
//   * q: e4m3 satfinite(min.NaN(max.NaN(div.full(1, q_scale) * y, -448), 448)); k/v cache: e4m3 satfinite(div.rn(y, s)).
// Built WITHOUT fast-math and with -fmad=false so nothing else is contracted or flushed.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace glue_ews {

constexpr int kD = 256;       // head dim
constexpr int kRot = 64;      // rotary dims (NeoX halves of 32)
constexpr int kHalf = kRot / 2;
constexpr int kWarpsPerCta = 8;

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float rsqrt_approx_ftz(float a) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float max_nan(float a, float b) {
  float r;
  asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float min_nan(float a, float b) {
  float r;
  asm("min.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ uint8_t e4m3(float v) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}
__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float butterfly(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

struct Args {
  const __nv_bfloat16* qkv;
  int64_t qkv_stride;
  uint8_t* q8;
  int64_t q8_stride;
  __nv_bfloat16* k_out;
  int64_t k_out_stride;
  __nv_bfloat16* gate_out;  // nullptr: no gate copy
  int64_t gate_stride;
  const __nv_bfloat16* qw;
  const __nv_bfloat16* kw;
  const __nv_bfloat16* cs;  // cos_sin_cache [max_pos, 64] bf16
  int64_t cs_stride;
  const int64_t* pos;       // [T] or [3, T]
  int64_t pos_stride_m, pos_stride_t;
  int has_mrope, mh, mw;
  const int64_t* slot;      // nullptr: no cache write
  int64_t n_slots;
  uint8_t* kc;
  uint8_t* vc;
  int64_t block_size;
  int64_t kc_sb, kc_sp, kc_sh, vc_sb, vc_sp, vc_sh;
  const float* q_scale;
  const float* k_scale;
  const float* v_scale;
  int T, nq, nkv;
  float eps;
};

__global__ __launch_bounds__(kWarpsPerCta * 32) void ews_kernel(Args a) {
  const int lane = threadIdx.x & 31;
  const int64_t item = static_cast<int64_t>(blockIdx.x) * kWarpsPerCta + (threadIdx.x >> 5);
  const int nh = a.nq + a.nkv;
  if (item >= static_cast<int64_t>(a.T) * nh) return;
  const int token = static_cast<int>(item / nh);
  const int head = static_cast<int>(item - static_cast<int64_t>(token) * nh);
  const bool is_k = head >= a.nq;
  const int lh = is_k ? head - a.nq : head;
  const __nv_bfloat16* row = a.qkv + static_cast<int64_t>(token) * a.qkv_stride;
  const __nv_bfloat16* in = row + (is_k ? a.nq * 2 * kD + lh * kD : lh * 2 * kD);
  const __nv_bfloat16* w = is_k ? a.kw : a.qw;
  // element pairs (2 (32 g + lane), +1) of the 4 Triton warps g = 0..3
  float x0[4], x1[4], w0[4], w1[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const __nv_bfloat162 xv = reinterpret_cast<const __nv_bfloat162*>(in)[32 * g + lane];
    const __nv_bfloat162 wv = reinterpret_cast<const __nv_bfloat162*>(w)[32 * g + lane];
    x0[g] = __bfloat162float(xv.x);
    x1[g] = __bfloat162float(xv.y);
    w0[g] = __bfloat162float(wv.x) + 1.0f;
    w1[g] = __bfloat162float(wv.y) + 1.0f;
  }
  float W[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) W[g] = butterfly(fmaf(x0[g], x0[g], x1[g] * x1[g]));
  const float sum = (W[0] + W[2]) + (W[1] + W[3]);
  const float rstd = rsqrt_approx_ftz(div_full(sum, 256.0f) + a.eps);
  float y0[4], y1[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    y0[g] = bf((rstd * x0[g]) * w0[g]);
    y1[g] = bf((rstd * x1[g]) * w1[g]);
  }
  // RoPE on dims 0..63 (Triton group 0): rot index i = lane; x1n = y[i] (lane i/2, comp i&1), x2n = y[32 + i]
  const int src1 = lane >> 1, src2 = 16 + (lane >> 1);
  const float a0 = __shfl_sync(0xffffffffu, y0[0], src1), a1 = __shfl_sync(0xffffffffu, y1[0], src1);
  const float b0 = __shfl_sync(0xffffffffu, y0[0], src2), b1 = __shfl_sync(0xffffffffu, y1[0], src2);
  const float xr1 = (lane & 1) ? a1 : a0;
  const float xr2 = (lane & 1) ? b1 : b0;
  int64_t p = a.pos[static_cast<int64_t>(token) * a.pos_stride_t];
  if (a.has_mrope) {
    const int m = lane % 3;
    if (m == 1 && lane < 3 * a.mh) p = a.pos[a.pos_stride_m + static_cast<int64_t>(token) * a.pos_stride_t];
    else if (m == 2 && lane < 3 * a.mw) p = a.pos[2 * a.pos_stride_m + static_cast<int64_t>(token) * a.pos_stride_t];
  }
  const __nv_bfloat16* csr = a.cs + p * a.cs_stride;
  const float c = __bfloat162float(csr[lane]);
  const float s = __bfloat162float(csr[kHalf + lane]);
  const float o1 = bf(fmaf(xr1, c, -(xr2 * s)));
  const float o2 = bf(fmaf(xr2, c, xr1 * s));
  if (!is_k) {
    const float r = div_full(1.0f, *a.q_scale);
    uint8_t* qo = a.q8 + static_cast<int64_t>(token) * a.q8_stride + lh * kD;
#pragma unroll
    for (int g = 1; g < 4; ++g) {  // pass-through dims 64..255
      const uint8_t lo = e4m3(min_nan(max_nan(r * y0[g], -448.0f), 448.0f));
      const uint8_t hi = e4m3(min_nan(max_nan(r * y1[g], -448.0f), 448.0f));
      reinterpret_cast<uint16_t*>(qo)[32 * g + lane] = static_cast<uint16_t>(lo | (hi << 8));
    }
    qo[lane] = e4m3(min_nan(max_nan(r * o1, -448.0f), 448.0f));
    qo[kHalf + lane] = e4m3(min_nan(max_nan(r * o2, -448.0f), 448.0f));
    if (a.gate_out != nullptr) {
      const uint4* gs = reinterpret_cast<const uint4*>(in + kD);
      uint4* gd = reinterpret_cast<uint4*>(a.gate_out + static_cast<int64_t>(token) * a.gate_stride + lh * kD);
      gd[lane] = gs[lane];  // 32 lanes x 16 B = 256 bf16
    }
    return;
  }
  __nv_bfloat16* ko = a.k_out + static_cast<int64_t>(token) * a.k_out_stride + lh * kD;
#pragma unroll
  for (int g = 1; g < 4; ++g) {
    __nv_bfloat162 v;
    v.x = __float2bfloat16_rn(y0[g]);
    v.y = __float2bfloat16_rn(y1[g]);
    reinterpret_cast<__nv_bfloat162*>(ko)[32 * g + lane] = v;
  }
  ko[lane] = __float2bfloat16_rn(o1);
  ko[kHalf + lane] = __float2bfloat16_rn(o2);
  if (a.slot == nullptr || token >= a.n_slots) return;
  const int64_t sl = a.slot[token];
  if (sl < 0) return;
  const int64_t blk = sl / a.block_size;
  const int64_t off = sl - blk * a.block_size;
  const float ks = *a.k_scale, vs = *a.v_scale;
  uint8_t* kd = a.kc + blk * a.kc_sb + off * a.kc_sp + lh * a.kc_sh;
#pragma unroll
  for (int g = 1; g < 4; ++g) {
    const uint8_t lo = e4m3(__fdiv_rn(y0[g], ks));
    const uint8_t hi = e4m3(__fdiv_rn(y1[g], ks));
    reinterpret_cast<uint16_t*>(kd)[32 * g + lane] = static_cast<uint16_t>(lo | (hi << 8));
  }
  kd[lane] = e4m3(__fdiv_rn(o1, ks));
  kd[kHalf + lane] = e4m3(__fdiv_rn(o2, ks));
  const __nv_bfloat16* vin = row + a.nq * 2 * kD + a.nkv * kD + lh * kD;
  uint8_t* vd = a.vc + blk * a.vc_sb + off * a.vc_sp + lh * a.vc_sh;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const __nv_bfloat162 vv = reinterpret_cast<const __nv_bfloat162*>(vin)[32 * g + lane];
    const uint8_t lo = e4m3(__fdiv_rn(__bfloat162float(vv.x), vs));
    const uint8_t hi = e4m3(__fdiv_rn(__bfloat162float(vv.y), vs));
    reinterpret_cast<uint16_t*>(vd)[32 * g + lane] = static_cast<uint16_t>(lo | (hi << 8));
  }
}

void launch(torch::Tensor qkv, torch::Tensor positions, torch::Tensor qw, torch::Tensor kw, torch::Tensor cs, double eps,
            int64_t nq, int64_t nkv, int64_t mh, int64_t mw, torch::Tensor q_scale, torch::Tensor k_scale,
            torch::Tensor v_scale, c10::optional<torch::Tensor> slot, c10::optional<torch::Tensor> k_cache,
            c10::optional<torch::Tensor> v_cache, torch::Tensor q8, torch::Tensor k_out,
            c10::optional<torch::Tensor> gate) {
  const int64_t T = qkv.size(0);
  if (T == 0) return;
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && qkv.stride(1) == 1 && qkv.stride(0) % 8 == 0);
  TORCH_CHECK(qw.is_contiguous() && kw.is_contiguous() && qw.numel() == kD && kw.numel() == kD);
  TORCH_CHECK(cs.scalar_type() == at::kBFloat16 && cs.size(1) == kRot && cs.stride(1) == 1);
  TORCH_CHECK(positions.scalar_type() == at::kLong);
  TORCH_CHECK(q_scale.scalar_type() == at::kFloat && k_scale.scalar_type() == at::kFloat &&
              v_scale.scalar_type() == at::kFloat);
  Args a{};
  a.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  a.qkv_stride = qkv.stride(0);
  a.q8 = reinterpret_cast<uint8_t*>(q8.data_ptr());
  a.q8_stride = q8.stride(0);
  a.k_out = reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr());
  a.k_out_stride = k_out.stride(0);
  a.gate_out = gate.has_value() ? reinterpret_cast<__nv_bfloat16*>(gate->data_ptr()) : nullptr;
  a.gate_stride = gate.has_value() ? gate->stride(0) : 0;
  a.qw = reinterpret_cast<const __nv_bfloat16*>(qw.data_ptr());
  a.kw = reinterpret_cast<const __nv_bfloat16*>(kw.data_ptr());
  a.cs = reinterpret_cast<const __nv_bfloat16*>(cs.data_ptr());
  a.cs_stride = cs.stride(0);
  a.pos = positions.data_ptr<int64_t>();
  if (positions.dim() == 2) {
    a.has_mrope = 1;
    a.pos_stride_m = positions.stride(0);
    a.pos_stride_t = positions.stride(1);
  } else {
    a.has_mrope = 0;
    a.pos_stride_m = 0;
    a.pos_stride_t = positions.stride(0);
  }
  a.mh = static_cast<int>(mh);
  a.mw = static_cast<int>(mw);
  if (slot.has_value()) {
    TORCH_CHECK(slot->scalar_type() == at::kLong && k_cache.has_value() && v_cache.has_value());
    a.slot = slot->data_ptr<int64_t>();
    a.n_slots = slot->size(0);
    a.kc = reinterpret_cast<uint8_t*>(k_cache->data_ptr());
    a.vc = reinterpret_cast<uint8_t*>(v_cache->data_ptr());
    a.block_size = k_cache->size(1);
    a.kc_sb = k_cache->stride(0); a.kc_sp = k_cache->stride(1); a.kc_sh = k_cache->stride(2);
    a.vc_sb = v_cache->stride(0); a.vc_sp = v_cache->stride(1); a.vc_sh = v_cache->stride(2);
    TORCH_CHECK(k_cache->element_size() == 1 && v_cache->element_size() == 1 && k_cache->stride(3) == 1 &&
                v_cache->stride(3) == 1);
  }
  a.q_scale = q_scale.data_ptr<float>();
  a.k_scale = k_scale.data_ptr<float>();
  a.v_scale = v_scale.data_ptr<float>();
  a.T = static_cast<int>(T);
  a.nq = static_cast<int>(nq);
  a.nkv = static_cast<int>(nkv);
  a.eps = static_cast<float>(eps);
  const int64_t items = T * (nq + nkv);
  const dim3 grid(static_cast<unsigned>((items + kWarpsPerCta - 1) / kWarpsPerCta));
  ews_kernel<<<grid, kWarpsPerCta * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace glue_ews

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &glue_ews::launch); }
