# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""GDN spec-decode (MTP) recurrence + gated RMSNorm: register-resident CUDA kernel.

Same contract as ``fused_gdn_decode_post_conv_mtp`` (csrc/libtorch_stable/gdn/
fused_gdn_decode_kernel.cu): per request, read the state of the last accepted
token (``state_indices[r, num_accepted[r] - 1]``), run the gated delta rule over
the request's tokens, write the state after token t to ``state_indices[r, t]``
(skipped if 0) and the gated-RMSNorm output; requests with an invalid source (or
more than 8 tokens) get zero outputs and write no state. Not bitwise to that
kernel (different K reduction order), within fp32 rounding of it; the output is
rounded to bf16 before the norm, as there.

Layout: one 256-thread CTA per (request, value head). Each thread keeps 8 value
rows x 8 key columns of the 128 x 128 state in registers for all tokens, so the
key reductions are 8-wide in-thread plus 4 shuffle levels (csrc: 4 columns per
lane, 5 levels, two reductions per token). Per token, one pass decays, applies
the rank-1 update and accumulates both S_t q_t (the output) and S_t k_{t+1} (the
next token's key projection; d_{t+1} is applied to the reduced scalar), in packed
fp32x2 FMAs on sm_100+. Every global load (state, state_indices row, q/k/v/a/b,
gate, norm weight) is issued before any is used. Snapshots are written straight
from registers with streaming stores. The kernel moves 1 state read + 1 write per
token (bandwidth-bound at C512 batch sizes): VR, 4 tokens/request, 77 requests:
77.0 vs 81.6 us; 48 requests 48.3 vs 55.3 us; bytes-only floor ~72 us at 77.

The CUDA source is kept in this module (``_SOURCE``) so the package ships only
Python; ``load()`` builds it with torch.utils.cpp_extension (nvcc, ninja) for the
current device's arch-specific target under VLLM_CACHE_ROOT (concurrent
processes share the build directory via cpp_extension's file lock).
"""

import hashlib
import os

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_ext: list = []  # [module] once load() built or loaded it
_tried: list = []

# VLLM_GDN_MTP_CUDA_PDL=1: build the kernel with ``griddepcontrol.wait`` as its
# first instruction and launch it with programmatic stream serialization, so
# it launches while the preceding conv1d update (which triggers its dependents
# early) still runs and waits for it before any global load. Results are
# unchanged. Off (default): the source and launch are the stock ones.
GDN_MTP_CUDA_PDL = os.environ.get("VLLM_GDN_MTP_CUDA_PDL", "0") == "1"


def build_dir(arch: str) -> str:
    return os.path.join(envs.VLLM_CACHE_ROOT, "gdn_mtp_cuda", f"sm{arch}")


def load():
    """Build (or load the cached build of) the extension for the current device."""
    if _ext:
        return _ext[0]
    major, minor = torch.cuda.get_device_capability()
    import torch.utils.cpp_extension as cpp

    arch = f"{major}{minor}{'a' if major >= 9 else ''}"
    build = build_dir(arch)
    os.makedirs(build, exist_ok=True)
    source = _pdl_source(_SOURCE) if GDN_MTP_CUDA_PDL else _SOURCE
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    src = os.path.join(build, f"gdn_mtp_cuda_{digest}.cu")
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
            name=f"_gdn_mtp_cuda_{digest}",
            sources=[src],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"],
            extra_cflags=["-O3", "-std=c++20"],
            build_directory=build,
            verbose=False,
        )
    finally:
        cpp._get_cuda_arch_flags = orig
    _ext.append(ext)
    return ext


def enable() -> bool:
    """Build (JIT, nvcc) once per process; a failed build logs a warning and
    leaves the csrc kernel in use. Returns whether the kernel is available.
    """
    if not _tried:
        _tried.append(True)
        try:
            load()
            logger.info(
                "GDN MTP decode: register-resident CUDA kernel built and enabled."
            )
        except Exception:
            logger.warning(
                "GDN MTP decode: CUDA kernel build failed; using the csrc kernel.",
                exc_info=True,
            )
    return bool(_ext)


def ready() -> bool:
    return bool(_ext)


def gdn_mtp_cuda(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    norm_eps: float,
    output_gate_activation: str,
) -> bool:
    """Run the kernel; False (nothing launched) if the layout contract is not
    met, so the caller keeps the csrc kernel. Requires ``enable()`` first.
    """
    return _ext[0].run(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        state,
        output_gate,
        norm_weight,
        out,
        float(scale),
        float(norm_eps),
        output_gate_activation == "sigmoid",
    )


_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace gmr {

constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kMaxT = 8;
constexpr int kKPT = 8;              // key columns per thread
constexpr int kRPT = 8;              // value rows per thread
constexpr int kTPR = kK / kKPT;      // threads per row (16)
constexpr int kNRG = kV / kRPT;      // row groups (16)
constexpr int kNT = kNRG * kTPR;     // threads (256)
constexpr int kNW = kNT / 32;        // warps (8): one per token in the prologue/epilogue
constexpr int kNI = kKPT / 4;        // float4 chunks per row slice (2)
constexpr int kGI = kMaxT * kV / kNT;  // gate elements per thread (4)
static_assert(kNW == kMaxT, "one warp per token");

struct Params {
  const __nv_bfloat16* qkv;
  const __nv_bfloat16* a;
  const __nv_bfloat16* b;
  const float* a_log;
  const void* dt_bias;
  const int* si;
  const int* cu;
  const int* acc;
  void* state;
  const __nv_bfloat16* gate;
  const void* norm_w;
  __nv_bfloat16* out;
  int64_t s_qkv, s_a, s_b, s_gate, s_slot;
  int si_width, H, HV, ratio;
  int dtb_type;  // 0 fp32, 1 bf16, 2 fp16
  int norm_w_bf16, sigmoid_gate;
  float scale, eps;
};

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

template <typename S>
struct Io;

template <>
struct Io<float> {
  static __device__ __forceinline__ float4 ld(const float* p) {
    return __ldg(reinterpret_cast<const float4*>(p));
  }
  static __device__ __forceinline__ void st(float* p, float4 v) {
    __stcs(reinterpret_cast<float4*>(p), v);
  }
};

template <>
struct Io<__nv_bfloat16> {
  static __device__ __forceinline__ float4 ld(const __nv_bfloat16* p) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(p));
    const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&u.x);
    const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&u.y);
    return make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
  }
  static __device__ __forceinline__ void st(__nv_bfloat16* p, float4 v) {
    const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
    const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 u;
    u.x = *reinterpret_cast<const uint32_t*>(&lo);
    u.y = *reinterpret_cast<const uint32_t*>(&hi);
    __stcs(reinterpret_cast<uint2*>(p), u);
  }
};

// Two-lane fp32 math: packed FFMA2/FMUL2 on sm_100+ (per-lane results equal
// the scalar fmaf / multiply).
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

__device__ __forceinline__ float sum4(const float2 (&x)[2]) {
  return (x[0].x + x[0].y) + (x[1].x + x[1].y);
}

template <typename S>
__global__ void __launch_bounds__(kNT, 2) mtp_kernel(const Params p) {
  __shared__ __align__(16) float s_q[kMaxT][kK];
  __shared__ __align__(16) float s_k[kMaxT][kK];
  __shared__ float s_v[kMaxT][kV];
  __shared__ float s_o[kMaxT][kV];
  __shared__ float s_g[kMaxT][kV];  // norm weight x activated output gate
  __shared__ float s_dec[kMaxT];
  __shared__ float s_beta[kMaxT];

  const int req = blockIdx.x;
  const int hv = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Independent loads (one round trip): the request's token range, accepted
  // count and whole state_indices row; the source slot is selected from the row.
  const int* si_row = p.si + req * p.si_width;
  int slots[kMaxT];
#pragma unroll
  for (int t = 0; t < kMaxT; ++t) slots[t] = t < p.si_width ? __ldg(si_row + t) : 0;
  const int bos = __ldg(p.cu + req);
  const int T = __ldg(p.cu + req + 1) - bos;
  const int acc = __ldg(p.acc + req);
  if (T <= 0) return;
  int src = 0;
#pragma unroll
  for (int t = 0; t < kMaxT; ++t) src = t == acc - 1 ? slots[t] : src;
  if (src <= 0 || T > kMaxT) {
    for (int x = tid; x < T * kV; x += kNT)
      p.out[(static_cast<int64_t>(bos + x / kV) * p.HV + hv) * kV + x % kV] = __float2bfloat16(0.0f);
    return;
  }

  // Thread (rg, j): rows rg + r * kNRG, columns i * 4 * kTPR + 4 * j + c (the 16
  // threads of a row read 256 contiguous bytes per chunk i).
  // h[r][i][0] = columns c 0..1, h[r][i][1] = columns 2..3 of chunk i.
  const int j = tid % kTPR;
  const int rg = tid / kTPR;
  const int64_t head = static_cast<int64_t>(hv) * kV * kK;
  const S* sp = static_cast<const S*>(p.state) + static_cast<int64_t>(src) * p.s_slot + head;
  float2 h[kRPT][kNI][2];
#pragma unroll
  for (int r = 0; r < kRPT; ++r) {
#pragma unroll
    for (int i = 0; i < kNI; ++i) {
      const float4 x = Io<S>::ld(sp + (rg + r * kNRG) * kK + i * 4 * kTPR + 4 * j);
      h[r][i][0] = make_float2(x.x, x.y);
      h[r][i][1] = make_float2(x.z, x.w);
    }
  }

  // Every other global load is issued before any is used, so their latencies
  // overlap the state loads: q/k/v/a/b of token `warp`, dt bias, A_log, and the
  // norm epilogue's gate and weight.
  const int kh = hv / p.ratio;
  __nv_bfloat16 qb[4], kb[4], vb[4], ab, bb;
  if (warp < T) {
    const int64_t base = static_cast<int64_t>(bos + warp) * p.s_qkv;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + 32 * i;
      qb[i] = p.qkv[base + kh * kK + dim];
      kb[i] = p.qkv[base + p.H * kK + kh * kK + dim];
      vb[i] = p.qkv[base + 2 * p.H * kK + hv * kV + dim];
    }
    ab = p.a[static_cast<int64_t>(bos + warp) * p.s_a + hv];
    bb = p.b[static_cast<int64_t>(bos + warp) * p.s_b + hv];
  }
  float dtb;
  if (p.dtb_type == 1) dtb = __bfloat162float(static_cast<const __nv_bfloat16*>(p.dt_bias)[hv]);
  else if (p.dtb_type == 2) dtb = __half2float(static_cast<const __half*>(p.dt_bias)[hv]);
  else dtb = static_cast<const float*>(p.dt_bias)[hv];
  const float a_log = p.a_log[hv];
  __nv_bfloat16 gb[kGI];
  float wv[kGI];
#pragma unroll
  for (int u = 0; u < kGI; ++u) {
    const int x = tid + u * kNT;
    if (x < T * kV) {
      const int v = x % kV;
      gb[u] = p.gate[static_cast<int64_t>(bos + x / kV) * p.s_gate + hv * kV + v];
      wv[u] = p.norm_w_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.norm_w)[v])
                            : static_cast<const float*>(p.norm_w)[v];
    }
  }

  if (warp < T) {
    const int t = warp;
    float qv[4], kv[4];
    float qq = 0.0f, kk = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      qv[i] = __bfloat162float(qb[i]);
      kv[i] = __bfloat162float(kb[i]);
      s_v[t][lane + 32 * i] = __bfloat162float(vb[i]);
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
      s_q[t][lane + 32 * i] = qv[i] * qs;
      s_k[t][lane + 32 * i] = kv[i] * ks;
    }
    if (lane == 0) {
      const float g = -__expf(a_log) * softplus_f(__bfloat162float(ab) + dtb);
      s_dec[t] = __expf(g);
      s_beta[t] = sigmoid_f(__bfloat162float(bb));
    }
  }
#pragma unroll
  for (int u = 0; u < kGI; ++u) {
    const int x = tid + u * kNT;
    if (x < T * kV) {
      const float z = __bfloat162float(gb[u]);
      s_g[x / kV][x % kV] = wv[u] * (p.sigmoid_gate ? sigmoid_f(z) : z * sigmoid_f(z));
    }
  }
  __syncthreads();

  // hkr[r]: row r of S_{t-1} k_t (before the decay d_t).
  float hkr[kRPT];
  {
    float2 a[kRPT][2];
#pragma unroll
    for (int r = 0; r < kRPT; ++r) a[r][0] = a[r][1] = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int i = 0; i < kNI; ++i) {
      const float4 k4 = *reinterpret_cast<const float4*>(s_k[0] + i * 4 * kTPR + 4 * j);
      const float2 kc[2] = {make_float2(k4.x, k4.y), make_float2(k4.z, k4.w)};
#pragma unroll
      for (int r = 0; r < kRPT; ++r) {
#pragma unroll
        for (int c = 0; c < 2; ++c) a[r][i & 1] = fma2(h[r][i][c], kc[c], a[r][i & 1]);
      }
    }
#pragma unroll
    for (int r = 0; r < kRPT; ++r) {
      hkr[r] = sum4(a[r]);
#pragma unroll
      for (int off = kTPR / 2; off > 0; off >>= 1) hkr[r] += __shfl_xor_sync(0xffffffffu, hkr[r], off);
    }
  }

  S* const state = static_cast<S*>(p.state);
  for (int t = 0; t < T; ++t) {
    const float d = s_dec[t];
    const float2 d2 = make_float2(d, d);
    const float bt = s_beta[t];
    const float* kt = s_k[t];
    const float* qt = s_q[t];
    const float* kn = s_k[t + 1 < T ? t + 1 : t];
    float2 dl[kRPT];
#pragma unroll
    for (int r = 0; r < kRPT; ++r) {
      const float x = (s_v[t][rg + r * kNRG] - d * hkr[r]) * bt;
      dl[r] = make_float2(x, x);
    }
    float2 aq[kRPT][2], ak[kRPT][2];
#pragma unroll
    for (int r = 0; r < kRPT; ++r) aq[r][0] = aq[r][1] = ak[r][0] = ak[r][1] = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int i = 0; i < kNI; ++i) {
      const float4 k4 = *reinterpret_cast<const float4*>(kt + i * 4 * kTPR + 4 * j);
      const float4 q4 = *reinterpret_cast<const float4*>(qt + i * 4 * kTPR + 4 * j);
      const float4 n4 = *reinterpret_cast<const float4*>(kn + i * 4 * kTPR + 4 * j);
      const float2 kc[2] = {make_float2(k4.x, k4.y), make_float2(k4.z, k4.w)};
      const float2 qc[2] = {make_float2(q4.x, q4.y), make_float2(q4.z, q4.w)};
      const float2 nc[2] = {make_float2(n4.x, n4.y), make_float2(n4.z, n4.w)};
#pragma unroll
      for (int r = 0; r < kRPT; ++r) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          h[r][i][c] = fma2(kc[c], dl[r], mul2(h[r][i][c], d2));
          aq[r][i & 1] = fma2(h[r][i][c], qc[c], aq[r][i & 1]);
          ak[r][i & 1] = fma2(h[r][i][c], nc[c], ak[r][i & 1]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < kRPT; ++r) {
      float o = sum4(aq[r]);
      float n = sum4(ak[r]);
#pragma unroll
      for (int off = kTPR / 2; off > 0; off >>= 1) {
        o += __shfl_xor_sync(0xffffffffu, o, off);
        n += __shfl_xor_sync(0xffffffffu, n, off);
      }
      hkr[r] = n;
      if (j == 0) s_o[t][rg + r * kNRG] = __bfloat162float(__float2bfloat16(o));
    }
    const int dst = __ldg(si_row + t);
    if (dst > 0) {
      S* dp = state + static_cast<int64_t>(dst) * p.s_slot + head;
#pragma unroll
      for (int r = 0; r < kRPT; ++r) {
#pragma unroll
        for (int i = 0; i < kNI; ++i) {
          Io<S>::st(dp + (rg + r * kNRG) * kK + i * 4 * kTPR + 4 * j,
                    make_float4(h[r][i][0].x, h[r][i][0].y, h[r][i][1].x, h[r][i][1].y));
        }
      }
    }
  }
  __syncthreads();

  if (warp < T) {
    const int t = warp;
    float ov[4];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ov[i] = s_o[t][lane + 32 * i];
      ss += ov[i] * ov[i];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float rstd = rsqrtf(ss / static_cast<float>(kV) + p.eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int v = lane + 32 * i;
      p.out[(static_cast<int64_t>(bos + t) * p.HV + hv) * kV + v] = __float2bfloat16(ov[i] * rstd * s_g[t][v]);
    }
  }
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
  const auto st = state.scalar_type();
  if (st != at::kFloat && st != at::kBFloat16) return false;
  if (si.scalar_type() != at::kInt || cu.scalar_type() != at::kInt || acc.scalar_type() != at::kInt) return false;
  if (!si.is_contiguous() || !cu.is_contiguous() || !acc.is_contiguous()) return false;
  if (state.dim() != 4 || state.size(2) != kV || state.size(3) != kK || state.stride(3) != 1 ||
      state.stride(2) != kK || state.stride(1) != kV * kK)
    return false;
  const int HV = (int)state.size(1);
  const int epc = st == at::kFloat ? 4 : 8;
  if ((reinterpret_cast<uintptr_t>(state.data_ptr()) % 16) != 0 || state.stride(0) % epc != 0) return false;
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
  p.state = state.data_ptr();
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
  auto stream = c10::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)N, (unsigned)HV);
  if (st == at::kFloat)
    mtp_kernel<float><<<grid, kNT, 0, stream>>>(p);
  else
    mtp_kernel<__nv_bfloat16><<<grid, kNT, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

}  // namespace gmr

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &gmr::run); }
"""


_PDL_LAUNCH = r"""
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(kNT);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  if (st == at::kFloat)
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, mtp_kernel<float>, p));
  else
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, mtp_kernel<__nv_bfloat16>, p));
"""


def _pdl_source(source: str) -> str:
    """``source`` with the kernel waiting on its PDL predecessor before its
    first global load and launched with programmatic stream serialization.
    """
    replacements = (
        (
            "  const int warp = tid >> 5;\n",
            "  const int warp = tid >> 5;\n"
            "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n"
            '  asm volatile("griddepcontrol.wait;" ::: "memory");\n'
            "#endif\n",
        ),
        (
            "  if (st == at::kFloat)\n"
            "    mtp_kernel<float><<<grid, kNT, 0, stream>>>(p);\n"
            "  else\n"
            "    mtp_kernel<__nv_bfloat16><<<grid, kNT, 0, stream>>>(p);\n",
            _PDL_LAUNCH,
        ),
    )
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source
