# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""The stock WY tensor-core GDN MTP verify kernel (``VLLM_GDN_MTP_CUDA_WY``), kept
verbatim: ``gdn_mtp_cuda`` builds and calls it unless ``VLLM_GDN_MTP_DEFER_WRITES``
or ``VLLM_GDN_MTP_VERIFY_V2`` is on (they need the extended kernel and its record
arguments), so a process without them runs the same kernel binary as before."""


def wy_source(tune: str, pdl: bool) -> str:
    """The WY kernel source with ``tune`` macros (``"1"`` or ``"K=V+..."``)."""
    defines = f"#define GWY_PDL {int(pdl)}\n"
    for item in filter(None, (x.strip() for x in tune.replace(",", "+").split("+"))):
        if item == "1":
            continue
        key, val = item.split("=")
        assert key in ("PF", "QS", "KS", "MINB"), key
        defines += f"#define GWY_{key} {int(val)}\n"
    return defines + _WY_SOURCE


_WY_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cstdint>

// Internal linkage: several builds (tunes) may be loaded in one process, and the
// per-instantiation static in launch() must not be shared between them.
namespace gwy {
namespace {

// WY form of the gated delta rule over the T tokens of one request (S_0 = the
// source state, d_t decay, b_t beta, G_t = d_0 ... d_{t-1}, rho(s, t) =
// d_{s+1} ... d_t):
//   S_{t-1} k_t = G_t S_0 k_t + sum_{s<t} rho(s, t-1) (k_s . k_t) delta_s
//   delta_t     = b_t (v_t - d_t S_{t-1} k_t)
//   S_t q_t     = G_{t+1} S_0 q_t + sum_{s<=t} rho(s, t) (k_s . q_t) delta_s
// so per value row only the 2T dots S_0 k_t, S_0 q_t are needed (one MMA
// pass), the Gram k_s . k_t, k_s . q_t is per (request, head), and the
// snapshots S_t = d_t S_{t-1} + k_t delta_t^T are the stock element update.
// Macros: GWY_PDL 1: griddepcontrol.wait first (PDL launch), GWY_PF N > 0:
// L2-prefetch the source state of the CTA N blocks ahead (0: off), GWY_KS /
// GWY_QS: bf16 split terms of k / q (3: ~fp32-exact products), GWY_MINB: CTAs
// per SM.
#ifndef GWY_PDL
#define GWY_PDL 0
#endif
#ifndef GWY_PF
#define GWY_PF 424
#endif
#ifndef GWY_KS
#define GWY_KS 3
#endif
#ifndef GWY_QS
#define GWY_QS 2
#endif
#ifndef GWY_MINB
#define GWY_MINB 4
#endif

constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kNT = 256;
constexpr int kNW = kNT / 32;
constexpr int kMaxT = 8;
constexpr int kRW = kV / kNW;       // state rows per warp (16 = one MMA M tile)
constexpr int kPad = 136;           // bf16 row stride of the split tables (272 B: ldmatrix conflict free)
constexpr int kTab = kMaxT * kPad;  // one split-term table (rows t < 8)

struct Params {
  const __nv_bfloat16* qkv;
  const __nv_bfloat16* a;
  const __nv_bfloat16* b;
  const float* a_log;
  const void* dt_bias;
  const int* si;
  const int* cu;
  const int* acc;
  __nv_bfloat16* state;
  const __nv_bfloat16* gate;
  const void* norm_w;
  __nv_bfloat16* out;
  int64_t s_qkv, s_a, s_b, s_gate, s_slot;
  int si_width, H, HV, ratio;
  int dtb_type;  // 0 fp32, 1 bf16, 2 fp16
  int norm_w_bf16, sigmoid_gate;
  float scale, eps;
};

struct Smem {
  __nv_bfloat16 st[kV * kK];  // source state; 16-B chunks XOR-swizzled by (row & 7)
  union {
    __nv_bfloat16 split[(GWY_KS + GWY_QS) * kTab];  // MMA B operand: k_t terms, then q_t terms
    __nv_bfloat16 o[kMaxT][kV];                     // bf16-rounded S_t q_t (after the MMA pass)
  } u;
  float k[kMaxT][kK];  // normalized keys
  union {
    float q[kMaxT][kK];         // normalized, scaled queries (Gram only)
    float dl[kNW][kRW][kMaxT];  // delta_t per row (after the Gram)
  } u2;
  __nv_bfloat16 v[kMaxT][kV];
  float dec[kMaxT];
  float beta[kMaxT];
  float G[kMaxT + 1];     // G[t] = d_0 ... d_{t-1}
  float W[kMaxT][kMaxT];  // W[s][t] = rho(s, t-1) (k_s . k_t), s < t
  float Q[kMaxT][kMaxT];  // Q[s][t] = rho(s, t) (k_s . q_t), s <= t
};

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __fmul2_rn(a, b);
#else
  return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
#endif
}

__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __ffma2_rn(a, b, c);
#else
  return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
#endif
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a) : "memory");
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ void unpack4(uint2 u, float (&x)[4]) {
  x[0] = bf_lo(u.x);
  x[1] = bf_hi(u.x);
  x[2] = bf_lo(u.y);
  x[3] = bf_hi(u.y);
}
// x[0..3] (dims d .. d + 3) as NS bf16 terms (x ~ s_0 + s_1 + ...; 3 terms: residual ~2^-24 |x|)
// into row t of NS tables
template <int NS>
__device__ __forceinline__ void split_store4(__nv_bfloat16* tab, int t, int d, const float (&x)[4]) {
  float r[4] = {x[0], x[1], x[2], x[3]};
#pragma unroll
  for (int s = 0; s < NS; ++s) {
    uint2 u;
    u.x = pack_bf2(r[0], r[1]);
    u.y = pack_bf2(r[2], r[3]);
    *reinterpret_cast<uint2*>(tab + s * kTab + t * kPad + d) = u;
    r[0] = __fsub_rn(r[0], bf_lo(u.x));
    r[1] = __fsub_rn(r[1], bf_hi(u.x));
    r[2] = __fsub_rn(r[2], bf_lo(u.y));
    r[3] = __fsub_rn(r[3], bf_hi(u.y));
  }
}

// Snapshots of the warp's 16 rows for a request of exactly NT tokens: lane =
// columns 4 lane .. 4 lane + 3; per element the stock update h = k_t delta_t +
// h d_t (fp32), RN to bf16, streaming 8-B stores (one 256-B row per warp store).
template <int TM, int NT>
__device__ __forceinline__ void chain(const Smem& sm, const int (&slots)[TM], __nv_bfloat16* state, int64_t s_slot,
                                      int64_t head, int warp, int lane) {
  float2 kr[NT][2];
  float dd[NT];
  __nv_bfloat16* dst[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const float4 k4 = *reinterpret_cast<const float4*>(&sm.k[t][4 * lane]);
    kr[t][0] = make_float2(k4.x, k4.y);
    kr[t][1] = make_float2(k4.z, k4.w);
    dd[t] = sm.dec[t];
    dst[t] = slots[t] > 0 ? state + static_cast<int64_t>(slots[t]) * s_slot + head + warp * kRW * kK + 4 * lane
                          : nullptr;
  }
  const __nv_bfloat16* srow = sm.st + warp * kRW * kK;
  const int lofs = ((lane >> 1) << 3) + (lane & 1) * 4;  // chunk lane / 2, half lane % 2 (unswizzled)
#pragma unroll
  for (int rr = 0; rr < kRW; ++rr) {
    const uint2 w = *reinterpret_cast<const uint2*>(srow + rr * kK + (lofs ^ ((rr & 7) << 3)));
    float2 h0 = make_float2(bf_lo(w.x), bf_hi(w.x));
    float2 h1 = make_float2(bf_lo(w.y), bf_hi(w.y));
    float dv[kMaxT];
    const float4 d0 = *reinterpret_cast<const float4*>(&sm.u2.dl[warp][rr][0]);
    dv[0] = d0.x;
    dv[1] = d0.y;
    dv[2] = d0.z;
    dv[3] = d0.w;
    if constexpr (NT > 4) {
      const float4 d1 = *reinterpret_cast<const float4*>(&sm.u2.dl[warp][rr][4]);
      dv[4] = d1.x;
      dv[5] = d1.y;
      dv[6] = d1.z;
      dv[7] = d1.w;
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const float2 dl2 = make_float2(dv[t], dv[t]);
      const float2 d2 = make_float2(dd[t], dd[t]);
      h0 = fma2(kr[t][0], dl2, mul2(h0, d2));
      h1 = fma2(kr[t][1], dl2, mul2(h1, d2));
      if (dst[t] != nullptr) __stcs(reinterpret_cast<uint2*>(dst[t] + rr * kK), make_uint2(pack_bf2(h0.x, h0.y), pack_bf2(h1.x, h1.y)));
    }
  }
}

template <int TM, int NT = 1>
__device__ __forceinline__ void chain_t(int T, const Smem& sm, const int (&slots)[TM], __nv_bfloat16* state,
                                        int64_t s_slot, int64_t head, int warp, int lane) {
  if (T == NT) chain<TM, NT>(sm, slots, state, s_slot, head, warp, lane);
  else if constexpr (NT < TM) chain_t<TM, NT + 1>(T, sm, slots, state, s_slot, head, warp, lane);
}

// TM: max tokens per request (>= si_width); requests with T > TM get zero outputs.
template <int TM>
__global__ void __launch_bounds__(kNT, TM <= 6 ? GWY_MINB : 3) wy_kernel(const Params p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int req = blockIdx.x;
  const int hv = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
#if GWY_PDL && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

  const int* si_row = p.si + req * p.si_width;
  int slots[TM];
#pragma unroll
  for (int t = 0; t < TM; ++t) slots[t] = t < p.si_width ? __ldg(si_row + t) : 0;
  const int bos = __ldg(p.cu + req);
  const int T = __ldg(p.cu + req + 1) - bos;
  const int acc = __ldg(p.acc + req);
  if (T <= 0) return;
  int src = 0;
#pragma unroll
  for (int t = 0; t < TM; ++t) src = t == acc - 1 ? slots[t] : src;
  if (src <= 0 || T > TM) {
    for (int x = tid; x < T * kV; x += kNT)
      p.out[(static_cast<int64_t>(bos + x / kV) * p.HV + hv) * kV + x % kV] = __float2bfloat16(0.0f);
    return;
  }

  // Source state -> smem (warp w: rows [16w, 16w + 16), its own commit group).
  const int64_t head = static_cast<int64_t>(hv) * kV * kK;
  {
    const __nv_bfloat16* sp = p.state + static_cast<int64_t>(src) * p.s_slot + head;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int q = i * 32 + lane;
      const int row = warp * kRW + (q >> 4), ch = q & 15;
      cp_async16(&sm.st[row * kK + ((ch ^ (row & 7)) << 3)], sp + row * kK + ch * 8);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
#if GWY_PF
  if (warp == kNW - 1) {
    const int f = blockIdx.x + blockIdx.y * gridDim.x + GWY_PF;
    if (f < static_cast<int>(gridDim.x * gridDim.y)) {
      const int freq = f % gridDim.x;
      const int fhv = f / gridDim.x;
      const int facc = __ldg(p.acc + freq);
      const int fsrc = facc >= 1 && facc <= p.si_width ? __ldg(p.si + freq * p.si_width + facc - 1) : 0;
      if (fsrc > 0) {
        constexpr int kBytes = kV * kK * 2;
        const char* fp = reinterpret_cast<const char*>(p.state + static_cast<int64_t>(fsrc) * p.s_slot +
                                                       static_cast<int64_t>(fhv) * kV * kK) +
                         lane * (kBytes / 32);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(fp), "r"(kBytes / 32) : "memory");
      }
    }
  }
#endif

  // Token loads (warp t: token t, lane: dims 4 lane .. 4 lane + 3), all issued before any use.
  const int kh = hv / p.ratio;
  uint2 qr, kr, vr, zr;
  float wv[4];
  __nv_bfloat16 ab, bb;
  if (warp < T) {
    const int64_t tok = bos + warp;
    const __nv_bfloat16* row = p.qkv + tok * p.s_qkv;
    qr = __ldg(reinterpret_cast<const uint2*>(row + kh * kK + 4 * lane));
    kr = __ldg(reinterpret_cast<const uint2*>(row + p.H * kK + kh * kK + 4 * lane));
    vr = __ldg(reinterpret_cast<const uint2*>(row + 2 * p.H * kK + hv * kV + 4 * lane));
    zr = __ldg(reinterpret_cast<const uint2*>(p.gate + tok * p.s_gate + hv * kV + 4 * lane));
    if (p.norm_w_bf16) {
      unpack4(__ldg(reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p.norm_w) + 4 * lane)), wv);
    } else {
      const float4 w4 = __ldg(reinterpret_cast<const float4*>(static_cast<const float*>(p.norm_w) + 4 * lane));
      wv[0] = w4.x;
      wv[1] = w4.y;
      wv[2] = w4.z;
      wv[3] = w4.w;
    }
    ab = p.a[tok * p.s_a + hv];
    bb = p.b[tok * p.s_b + hv];
  }
  float dtb;
  if (p.dtb_type == 1) dtb = __bfloat162float(static_cast<const __nv_bfloat16*>(p.dt_bias)[hv]);
  else if (p.dtb_type == 2) dtb = __half2float(static_cast<const __half*>(p.dt_bias)[hv]);
  else dtb = static_cast<const float*>(p.dt_bias)[hv];
  const float a_log = p.a_log[hv];

  float gact[4];  // norm weight x activated output gate (epilogue of warp t)
  if (warp < T) {
    const int t = warp;
    float qv[4], kv[4], z[4];
    unpack4(qr, qv);
    unpack4(kr, kv);
    unpack4(zr, z);
    float qq = 0.0f, kk = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      qq += qv[i] * qv[i];
      kk += kv[i] * kv[i];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      qq += __shfl_xor_sync(0xffffffffu, qq, off);
      kk += __shfl_xor_sync(0xffffffffu, kk, off);
    }
    const float qs = rsqrtf(qq + 1.0e-6f) * p.scale;
    const float ks = rsqrtf(kk + 1.0e-6f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      qv[i] *= qs;
      kv[i] *= ks;
      gact[i] = wv[i] * (p.sigmoid_gate ? sigmoid_f(z[i]) : z[i] * sigmoid_f(z[i]));
    }
    *reinterpret_cast<float4*>(&sm.u2.q[t][4 * lane]) = make_float4(qv[0], qv[1], qv[2], qv[3]);
    *reinterpret_cast<float4*>(&sm.k[t][4 * lane]) = make_float4(kv[0], kv[1], kv[2], kv[3]);
    *reinterpret_cast<uint2*>(&sm.v[t][4 * lane]) = vr;
    split_store4<GWY_KS>(sm.u.split, t, 4 * lane, kv);
    split_store4<GWY_QS>(sm.u.split + GWY_KS * kTab, t, 4 * lane, qv);
    if (lane == 0) {
      const float g = -__expf(a_log) * softplus_f(__bfloat162float(ab) + dtb);
      sm.dec[t] = __expf(g);
      sm.beta[t] = sigmoid_f(__bfloat162float(bb));
    }
  }
  __syncthreads();  // #1: token prep

  // Gram + WY coefficients: 64 groups of 4 lanes, one dot each (28 k.k pairs
  // s < t, then 36 k.q pairs s <= t; pairs with t >= T are discarded).
  {
    const int grp = tid >> 2, seg = tid & 3;
    int s, t;
    bool isq;
    if (grp < 28) {  // pairs [t (t - 1) / 2, t (t + 1) / 2): s < t
      t = static_cast<int>((1.0f + sqrtf(static_cast<float>(1 + 8 * grp))) * 0.5f);
      s = grp - t * (t - 1) / 2;
      isq = false;
    } else {  // pairs [t (t + 1) / 2, (t + 1) (t + 2) / 2): s <= t
      const int pq = grp - 28;
      t = static_cast<int>((sqrtf(static_cast<float>(1 + 8 * pq)) - 1.0f) * 0.5f);
      s = pq - t * (t + 1) / 2;
      isq = true;
    }
    const float* x = sm.k[s];
    const float* y = isq ? sm.u2.q[t] : sm.k[t];
    float4 a4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int c = 4 * (seg + 4 * j);
      const float4 x4 = *reinterpret_cast<const float4*>(x + c);
      const float4 y4 = *reinterpret_cast<const float4*>(y + c);
      a4.x = fmaf(x4.x, y4.x, a4.x);
      a4.y = fmaf(x4.y, y4.y, a4.y);
      a4.z = fmaf(x4.z, y4.z, a4.z);
      a4.w = fmaf(x4.w, y4.w, a4.w);
    }
    float d = (a4.x + a4.y) + (a4.z + a4.w);
    d += __shfl_xor_sync(0xffffffffu, d, 2);
    d += __shfl_xor_sync(0xffffffffu, d, 1);
    if (seg == 0 && t < T) {
      float rho = 1.0f;
      const int hi = isq ? t : t - 1;
      for (int r = s + 1; r <= hi; ++r) rho *= sm.dec[r];
      if (isq) sm.Q[s][t] = rho * d;
      else sm.W[s][t] = rho * d;
    }
    if (tid <= TM) {
      float g = 1.0f;
      for (int r = 0; r < tid && r < T; ++r) g *= sm.dec[r];
      sm.G[tid] = g;
    }
  }

  // S_0 k_t (columns 0..7 of ck) and S_0 q_t (cq) of this warp's 16 rows.
  float ck[4] = {0.0f, 0.0f, 0.0f, 0.0f}, cq[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncwarp();
  {
    const int arow = warp * kRW + (lane & 15);
    const __nv_bfloat16* bk = sm.u.split + (lane & 7) * kPad + ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int kt = 0; kt < kK / 16; ++kt) {
      uint32_t a[4];
      ldsm_x4(a, &sm.st[arow * kK + (((2 * kt + (lane >> 4)) ^ (arow & 7)) << 3)]);
#pragma unroll
      for (int s = 0; s < GWY_KS; ++s) {
        uint32_t b[2];
        ldsm_x2(b, bk + s * kTab + kt * 16);
        mma_bf16(ck, a, b);
      }
#pragma unroll
      for (int s = 0; s < GWY_QS; ++s) {
        uint32_t b[2];
        ldsm_x2(b, bk + (GWY_KS + s) * kTab + kt * 16);
        mma_bf16(cq, a, b);
      }
    }
  }
  __syncthreads();  // #2: coefficients; every warp is done with the split tables

  // Row solve: lane (g, tq) owns warp row g + 8 (tq & 1); the C fragment of
  // column t of that row is in lane 4 g + t / 2.
  {
    const int g = lane >> 2, tq = lane & 3, hh = tq & 1;
    const int rr = g + 8 * hh;
    const int row = warp * kRW + rr;
    float dl[kMaxT];
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) dl[t] = 0.0f;
#pragma unroll
    for (int t = 0; t < TM; ++t) {
      const int sl = 4 * g + (t >> 1);
      const float k0 = __shfl_sync(0xffffffffu, ck[t & 1], sl);
      const float k1 = __shfl_sync(0xffffffffu, ck[2 + (t & 1)], sl);
      const float q0 = __shfl_sync(0xffffffffu, cq[t & 1], sl);
      const float q1 = __shfl_sync(0xffffffffu, cq[2 + (t & 1)], sl);
      if (t < T) {
        float hk = sm.G[t] * (hh ? k1 : k0);
#pragma unroll
        for (int s = 0; s < t; ++s) hk = fmaf(sm.W[s][t], dl[s], hk);
        dl[t] = (__bfloat162float(sm.v[t][row]) - sm.dec[t] * hk) * sm.beta[t];
        float o = sm.G[t + 1] * (hh ? q1 : q0);
#pragma unroll
        for (int s = 0; s <= t; ++s) o = fmaf(sm.Q[s][t], dl[s], o);
        if (tq < 2) sm.u.o[t][row] = __float2bfloat16(o);
      }
    }
    if (tq < 2) {
      *reinterpret_cast<float4*>(&sm.u2.dl[warp][rr][0]) = make_float4(dl[0], dl[1], dl[2], dl[3]);
      if (TM > 4) *reinterpret_cast<float4*>(&sm.u2.dl[warp][rr][4]) = make_float4(dl[4], dl[5], dl[6], dl[7]);
    }
  }
  __syncthreads();  // #3: outputs, delta table

  // Gated RMSNorm epilogue: warp t, token t.
  if (warp < T) {
    const int t = warp;
    float ov[4];
    unpack4(*reinterpret_cast<const uint2*>(&sm.u.o[t][4 * lane]), ov);
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) ss += ov[i] * ov[i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float rstd = rsqrtf(ss / static_cast<float>(kV) + p.eps);
    *reinterpret_cast<uint2*>(p.out + (static_cast<int64_t>(bos + t) * p.HV + hv) * kV + 4 * lane) =
        make_uint2(pack_bf2(ov[0] * rstd * gact[0], ov[1] * rstd * gact[1]),
                   pack_bf2(ov[2] * rstd * gact[2], ov[3] * rstd * gact[3]));
  }

  chain_t<TM>(T, sm, slots, p.state, p.s_slot, head, warp, lane);
}

template <int TM>
void launch(const Params& p, dim3 grid, cudaStream_t stream) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(wy_kernel<TM>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(sizeof(Smem))));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(kNT);
  cfg.dynamicSmemBytes = sizeof(Smem);
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = GWY_PDL ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, wy_kernel<TM>, p));
}

// false if the layout contract is not met (caller keeps the csrc kernel)
bool run(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
         torch::Tensor dt_bias, torch::Tensor si, torch::Tensor cu, torch::Tensor acc,
         torch::Tensor state, torch::Tensor gate, torch::Tensor norm_w, torch::Tensor out,
         double scale, double eps, bool sigmoid_gate) {
  if (qkv.scalar_type() != at::kBFloat16 || a.scalar_type() != at::kBFloat16 ||
      b.scalar_type() != at::kBFloat16 || gate.scalar_type() != at::kBFloat16 ||
      out.scalar_type() != at::kBFloat16)
    return false;
  if (a_log.scalar_type() != at::kFloat || !a_log.is_contiguous()) return false;
  const auto dtb = dt_bias.scalar_type();
  if ((dtb != at::kFloat && dtb != at::kBFloat16 && dtb != at::kHalf) || !dt_bias.is_contiguous()) return false;
  if ((norm_w.scalar_type() != at::kFloat && norm_w.scalar_type() != at::kBFloat16) ||
      !norm_w.is_contiguous() || norm_w.numel() != kV)
    return false;
  if (state.scalar_type() != at::kBFloat16) return false;
  if (si.scalar_type() != at::kInt || cu.scalar_type() != at::kInt || acc.scalar_type() != at::kInt) return false;
  if (!si.is_contiguous() || !cu.is_contiguous() || !acc.is_contiguous()) return false;
  if (state.dim() != 4 || state.size(2) != kV || state.size(3) != kK || state.stride(3) != 1 ||
      state.stride(2) != kK || state.stride(1) != kV * kK)
    return false;
  const int HV = (int)state.size(1);
  if ((reinterpret_cast<uintptr_t>(state.data_ptr()) % 16) != 0 || state.stride(0) % 8 != 0) return false;
  if (qkv.dim() != 2 || qkv.stride(1) != 1) return false;
  const int64_t kw = qkv.size(1) - (int64_t)HV * kV;
  if (kw <= 0 || kw % (2 * kK) != 0) return false;
  const int H = (int)(kw / (2 * kK));
  if (HV % H != 0) return false;
  if (si.dim() != 2 || si.size(1) < 1 || si.size(1) > kMaxT) return false;
  const int N = (int)si.size(0);
  if (cu.numel() != N + 1 || acc.numel() != N) return false;
  const int64_t L = qkv.size(0);
  if (a.dim() != 2 || b.dim() != 2 || a.size(0) != L || b.size(0) != L || a.size(1) != HV ||
      b.size(1) != HV || a.stride(1) != 1 || b.stride(1) != 1)
    return false;
  if (a_log.numel() != HV || dt_bias.numel() != HV) return false;
  if (gate.dim() != 3 || gate.size(0) != L || gate.size(1) != HV || gate.size(2) != kV ||
      gate.stride(2) != 1 || gate.stride(1) != kV)
    return false;
  if (out.dim() != 3 || out.size(0) != L || out.size(1) != HV || out.size(2) != kV || !out.is_contiguous())
    return false;
  // 8-B vector loads of q/k/v/gate/norm weight and stores of the output
  const auto al = [](const torch::Tensor& x, int bytes) {
    return reinterpret_cast<uintptr_t>(x.data_ptr()) % bytes == 0;
  };
  if (!al(qkv, 8) || qkv.stride(0) % 4 != 0 || !al(gate, 8) || gate.stride(0) % 4 != 0 || !al(out, 8) ||
      !al(norm_w, norm_w.scalar_type() == at::kFloat ? 16 : 8))
    return false;
  if (N == 0 || L == 0) return true;
  Params p;
  p.qkv = (const __nv_bfloat16*)qkv.data_ptr();
  p.a = (const __nv_bfloat16*)a.data_ptr();
  p.b = (const __nv_bfloat16*)b.data_ptr();
  p.a_log = (const float*)a_log.data_ptr();
  p.dt_bias = dt_bias.data_ptr();
  p.si = (const int*)si.data_ptr();
  p.cu = (const int*)cu.data_ptr();
  p.acc = (const int*)acc.data_ptr();
  p.state = (__nv_bfloat16*)state.data_ptr();
  p.gate = (const __nv_bfloat16*)gate.data_ptr();
  p.norm_w = norm_w.data_ptr();
  p.out = (__nv_bfloat16*)out.data_ptr();
  p.s_qkv = qkv.stride(0);
  p.s_a = a.stride(0);
  p.s_b = b.stride(0);
  p.s_gate = gate.stride(0);
  p.s_slot = state.stride(0);
  p.si_width = (int)si.size(1);
  p.H = H;
  p.HV = HV;
  p.ratio = HV / H;
  p.dtb_type = dtb == at::kFloat ? 0 : (dtb == at::kBFloat16 ? 1 : 2);
  p.norm_w_bf16 = norm_w.scalar_type() == at::kBFloat16;
  p.sigmoid_gate = sigmoid_gate;
  p.scale = (float)scale;
  p.eps = (float)eps;
  const c10::cuda::CUDAGuard guard(qkv.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)N, (unsigned)HV);
  if (p.si_width <= 4) launch<4>(p, grid, stream);
  else if (p.si_width == 5) launch<5>(p, grid, stream);
  else if (p.si_width == 6) launch<6>(p, grid, stream);
  else launch<8>(p, grid, stream);
  return true;
}

}  // namespace
}  // namespace gwy

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &gwy::run); }
"""
