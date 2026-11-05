# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""GDN spec-decode (MTP verify) core of a small decode-only batch in one fused CUDA launch.

``gfd_kernel<1, 4, 1>`` runs, per GDN layer, the width-4 causal conv + SiLU of
the q/k/v channels (straight from the pre-conv projection rows and the
previous step's conv window, plus the window roll), the gated delta rule over
the request's T tokens with a state snapshot per token, the gated RMSNorm and
out_proj's MXFP8 activation (e4m3 + F8_128x4 UE8M0 scales, rows no request
owns zeroed). Grid (2, H, N): one 512-thread CTA per (request, value head),
the two value heads of a key head as one 2-CTA cluster that shares the q/k
conv window (the cluster barrier orders both CTAs' reads before its roll).
Requests with no tokens (FULL-graph padding) only zero their share of the
padding rows.

Numerics: bitwise to ``causal_conv1d_update`` + the register-resident MTP
kernel of ``gdn_mtp_cuda`` built with RS=1 and the fused quant (the path of
decode batches above VLLM_GDN_MTP_TRITON_MAX_REQS): each state row is reduced
exactly as that kernel does it (16 threads per row, 8 key columns each, the
same column order, packed fp32 FMAs, the RS=1 reduce-scatter tree); its q/k L2
norms, gates, norm epilogue and MXFP8 quant are copied from it; the conv keeps
the Triton kernel's op order (bf16 products, fp32 sum, ex2.approx / div.full
SiLU). Against the three-kernel chain and the one-launch Triton kernel
(``gdn_mtp_fused_decode``) of batches of <= 4 requests it differs at rounding
level (q/k L2-norm reduction order), as those two differ from each other.

It is meant for small batches: VLLM_GDN_GFD_MIN_REQS..VLLM_GDN_GFD_MAX_REQS in
qwen_gdn_linear_attn bound the batch sizes it takes.

The CUDA source is kept in this module (``_SOURCE``) so the package ships only
Python; ``enable()`` builds it with torch.utils.cpp_extension for the current
device's arch-specific target under VLLM_CACHE_ROOT, as ``gdn_mtp_cuda.build``
does (concurrent processes share the build directory via cpp_extension's file
lock).
"""

import hashlib
import os

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_ext: list = []  # [module] once enable() built or loaded it
_tried: list = []
_pdl: list = []  # [bool]: launch with programmatic stream serialization
_refused: list = []  # a layout the kernel refused (logged once)


def build_dir(arch: str) -> str:
    return os.path.join(envs.VLLM_CACHE_ROOT, "gdn_gfd_cuda", f"sm{arch}")


def build(source: str):
    """Build (or load the cached build of) ``source`` for the current device
    (``gdn_mtp_cuda.build`` with this module's directory and name).
    """
    major, minor = torch.cuda.get_device_capability()
    import torch.utils.cpp_extension as cpp

    arch = f"{major}{minor}{'a' if major >= 9 else ''}"
    bdir = build_dir(arch)
    os.makedirs(bdir, exist_ok=True)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    src = os.path.join(bdir, f"gdn_gfd_cuda_{digest}.cu")
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
            name=f"_gdn_gfd_cuda_{digest}",
            sources=[src],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"],
            extra_cflags=["-O3", "-std=c++20"],
            build_directory=bdir,
            verbose=False,
        )
    finally:
        cpp._get_cuda_arch_flags = orig


def enable() -> bool:
    """Build (JIT, nvcc) once per process; a failed build logs a warning and
    leaves the stock paths in use. Returns whether the kernel is available.
    """
    if not _tried:
        _tried.append(True)
        try:
            ext = build(_SOURCE)
            _pdl.append(bool(current_platform.is_arch_support_pdl()))
            info = list(ext.cfg_info())
            _ext.append(ext)
            logger.info(
                "GDN fused decode: gfd_kernel<1, 4, 1> built and enabled "
                "(threads %d, cluster %d, registers %d, smem %d B, local %d B, PDL %s).",
                *info,
                _pdl[0],
            )
        except Exception:
            logger.warning(
                "GDN fused decode: gfd kernel build failed; the stock decode "
                "paths stay in use.",
                exc_info=True,
            )
    return bool(_ext)


def ready() -> bool:
    return bool(_ext)


def gdn_gfd(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    activation: str,
    out_q: torch.Tensor,
    out_scale: torch.Tensor,
    scale: float,
    null_block_id: int = 0,
) -> bool:
    """Conv + SiLU, gated delta rule, gated RMSNorm and out_proj's MXFP8
    activation of the spec-decode rows in one launch; False (nothing launched)
    if the layout contract is not met, so the caller keeps the stock path.
    Requires ``enable()`` first.

    ``mixed_qkv`` [L, conv_dim] pre-conv projection rows (row-strided view;
    not modified), ``a``/``b`` [L, HV] (row-strided), ``conv_state`` [slots,
    conv_dim, state_len] (any strides; rolled in place), ``conv_weight``
    [conv_dim, 4] (no bias), ``state`` [slots, HV, V, K] bf16 (snapshots
    written at ``state_indices[r, t]``, skipped if 0), ``output_gate`` [L, HV,
    V] (row-strided), ``out_q`` [>= L, HV * V] e4m3 and ``out_scale`` the flat
    F8_128x4 scales of its rows padded to 128. int32 ``state_indices`` [N,
    T <= 8], ``cu_seqlens`` [N + 1], ``num_accepted_tokens`` [N]. K = V = 128,
    HV = 2 H. Rows ``>= cu_seqlens[N]`` get zero values and scales.
    """
    if activation not in ("silu", "swish", "sigmoid"):
        return False
    ok = _ext[0].run(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        state,
        conv_state,
        conv_weight,
        output_gate,
        norm_weight,
        out_q,
        out_scale,
        float(scale),
        float(norm_eps),
        activation == "sigmoid",
        int(null_block_id),
        _pdl[0],
    )
    if not ok and not _refused:
        _refused.append(True)
        logger.warning(
            "GDN fused decode: gfd kernel refused the layout (state %s %s, "
            "conv state %s, %d requests x %d tokens); the stock path runs.",
            tuple(state.shape),
            state.dtype,
            tuple(conv_state.shape),
            state_indices.size(0),
            state_indices.size(1) if state_indices.dim() == 2 else -1,
        )
    return ok


_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <type_traits>

namespace cg = cooperative_groups;

namespace gdn_gfd {

constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kMaxT = 8;
constexpr int kTPR = 16;     // threads per state row (8 key columns each), as gmr
constexpr int kNI = 2;       // 4-column chunks per thread and row, as gmr
constexpr int kG = 2;        // value heads per key head (checked on the host)
constexpr int kQK = 2 * kK;  // q + k conv channels of one key head

struct Params {
  const __nv_bfloat16* x;     // pre-conv projection rows [L, conv_dim], row stride s_x
  const __nv_bfloat16* a;     // [L, HV]
  const __nv_bfloat16* b;     // [L, HV]
  const float* a_log;         // [HV]
  const void* dt_bias;        // [HV]
  const int* si;              // [N, si_width]
  const int* cu;              // [N + 1]
  const int* acc;             // [N]
  __nv_bfloat16* state;       // [slots, HV, V, K]
  __nv_bfloat16* cs;          // conv state [slots, conv_dim, state_len] (strided)
  const __nv_bfloat16* cw;    // conv weight [conv_dim, 4] (strided)
  const __nv_bfloat16* gate;  // [L, HV, V], row stride s_gate
  const void* norm_w;         // [V]
  uint8_t* q;                 // e4m3 [q_rows, HV * V]
  uint8_t* sf;                // F8_128x4 UE8M0 scales of [sf_rows, HV * 4]
  __nv_bfloat16* dbg;         // optional: conv outputs, same layout as x
  uint64_t* stamp;            // optional: phase stamps, 8 per CTA
  int64_t s_x, s_a, s_b, s_gate, s_slot, cs_slot, cs_dim, cs_tok, cw_dim, cw_w;
  int si_width, H, HV, null_block;
  int dtb_type, norm_w_bf16, sigmoid_gate, silu_mode;
  int q_rows, sf_rows;
  float scale, eps;
};

// ---------------- verbatim from gmr (ops/gdn_mtp_cuda.py @ 77d1e9a77, GMR_FADD2 = 0) ----------------
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ uint32_t ue8m0(float amax) {
  const float nm = mul_rn(amax, 1.0f / 448.0f);
  const uint32_t bits = __float_as_uint(nm);
  const uint32_t e = (bits >> 23) & 255u;
  const uint32_t m = bits & 0x7FFFFFu;
  const uint32_t bump = (m != 0u && !(e == 0u && m <= 0x400000u)) ? 1u : 0u;
  const uint32_t sf = e + bump < 254u ? e + bump : 254u;
  return nm <= 0.0f ? 0u : sf;
}

__device__ __forceinline__ uint8_t e4m3(float y, uint32_t sf) {
  const float inv = __uint_as_float(sf == 0u ? 0u : (254u - sf) << 23);
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(mul_rn(y, inv), __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ int64_t sf_word(int row, int head, int HV) {
  return static_cast<int64_t>(row >> 7) * (128 * 4 * HV) + head * 512 + (row & 31) * 16 +
         ((row & 127) >> 5) * 4;
}

__device__ __forceinline__ float4 ld_bf16x4(const __nv_bfloat16* p) {
  const uint2 u = __ldg(reinterpret_cast<const uint2*>(p));
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&u.x);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&u.y);
  return make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
}

__device__ __forceinline__ void st_bf16x4(__nv_bfloat16* p, float4 v) {
  const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
  const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
  uint2 u;
  u.x = *reinterpret_cast<const uint32_t*>(&lo);
  u.y = *reinterpret_cast<const uint32_t*>(&hi);
  __stcs(reinterpret_cast<uint2*>(p), u);
}

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

template <int R, int OFF>
struct RowSum {
  static __device__ __forceinline__ float run(float* x, int j, int& rho) {
    if constexpr (R > 1) {
      constexpr int h = R / 2;
      const bool hi = (j & OFF) != 0;
#pragma unroll
      for (int k = 0; k < h; ++k) {
        const float send = hi ? x[k] : x[k + h];
        const float keep = hi ? x[k + h] : x[k];
        x[k] = keep + __shfl_xor_sync(0xffffffffu, send, OFF);
      }
      rho += hi ? h : 0;
      return RowSum<h, OFF / 2>::run(x, j, rho);
    } else if constexpr (OFF > 0) {
      x[0] += __shfl_xor_sync(0xffffffffu, x[0], OFF);
      return RowSum<1, OFF / 2>::run(x, j, rho);
    } else {
      return x[0];
    }
  }
};
// ---------------- end of the gmr copy ----------------

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(const float v) { return __bfloat162float(__float2bfloat16(v)); }

// Width-4 causal conv + SiLU of one channel and token, as _causal_conv1d_update_kernel computes and stores it.
__device__ __forceinline__ float conv_out(const __nv_bfloat16 (&u)[4], const __nv_bfloat16 (&w)[4], const int mode) {
  float acc = 0.0f;
#pragma unroll
  for (int jj = 0; jj < 4; ++jj) {
    if (mode & 4) acc = __fmaf_rn(bf(u[jj]), bf(w[jj]), acc);
    else acc = __fadd_rn(acc, rbf(__fmul_rn(bf(u[jj]), bf(w[jj]))));
  }
  const float t = __fmul_rn(-acc, 1.4426950408889634f);
  float e;
  if (mode & 1) asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(t));
  else asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  const float den = __fadd_rn(1.0f, e);
  float r;
  if (mode & 2) r = __fdiv_rn(acc, den);
  else asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(acc), "f"(den));
  return rbf(r);
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire;" ::: "memory"); }
__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

template <int C, int RPT>
struct Cfg {
  static constexpr int R = kV / C;                         // value rows per CTA
  static constexpr int NRG = R / RPT;                      // row groups (16 threads each)
  static constexpr int NT = NRG * kTPR;                    // threads
  static constexpr int NW = NT / 32;                       // warps
  static constexpr int CS = kG * C;                        // cluster: both value heads of a key head, C CTAs each
  static constexpr int NWB = kQK / CS;                     // q/k window channels each CTA rolls
  static constexpr int NGI = (kMaxT * kV + NT - 1) / NT;   // norm-gate items per thread
  static_assert(kV % C == 0 && R % RPT == 0, "rows");
  static_assert(NW >= kMaxT, "one warp per token in the prologue");
  static_assert(NT >= kQK && NT >= NWB, "one thread per q/k channel and rolled channel");
  static_assert(NT >= kQK + R || NT >= R, "v channels");
  static_assert(kTPR % RPT == 0, "reduce-scatter");
};

// Phase stamps (p.stamp != nullptr; thread 0 of each CTA): [0] globaltimer at entry, [1] clock64 at entry,
// [2] after griddepcontrol.wait, [3] after the prologue barrier (every load consumed, conv/norms/gates done),
// [4] after the token loop, [5] after the epilogue barrier, [6] clock64 at exit, [7] globaltimer at exit.
template <int C, int RPT, int MINB>
__global__ void __launch_bounds__(Cfg<C, RPT>::NT, MINB) gfd_kernel(const Params p) {
  using F = Cfg<C, RPT>;
  constexpr int R = F::R, NRG = F::NRG, NT = F::NT, NW = F::NW, NWB = F::NWB, NGI = F::NGI;
  __shared__ __align__(16) float s_q[kMaxT][kK];
  __shared__ __align__(16) float s_k[kMaxT][kK];
  __shared__ float s_v[kMaxT][R];
  __shared__ float s_o[kMaxT][R];
  __shared__ float s_full[kMaxT][kV];
  __shared__ float s_g[kMaxT][kV];
  __shared__ float s_dec[kMaxT];
  __shared__ float s_beta[kMaxT];
  __shared__ float s_n[2][NRG][RPT];
  __shared__ int s_slot[kMaxT];
  __shared__ __nv_bfloat16 s_roll[NWB][kMaxT + 2];

  const int rank = blockIdx.x;  // the cluster spans gridDim.x = 2C: rank in the cluster
  const int g = rank / C;       // value head within the key head
  const int c = rank % C;       // row block within the value head
  const int kh = blockIdx.y;
  const int req = blockIdx.z;
  const int N = static_cast<int>(gridDim.z);
  const int hv = kh * kG + g;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int mode = p.silu_mode;
  const int row0 = c * R;
  const int j = tid % kTPR;
  const int rg = tid / kTPR;
  const int64_t head = static_cast<int64_t>(hv) * kV * kK;
  uint64_t* const stamp =
      p.stamp == nullptr
          ? nullptr
          : p.stamp + 8 * ((static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
  const bool stamper = stamp != nullptr && tid == 0;
  if (stamper) {
    stamp[0] = gtimer();
    stamp[1] = clock64();
  }

  // ---- loads that need neither the indices nor this step's projections (one round trip) ----
  // Conv: one channel per thread. Threads tid < 256 own q/k channel tid of the key head; the v channel of row
  // row0 + tv belongs to thread tv + VOFF. Few live values per thread: no spill holds a load back.
  constexpr int VOFF = NT >= kQK + R ? kQK : 0;
  const bool has_qk = tid < kQK;
  const int tv = tid - VOFF;
  const bool has_v = tv >= 0 && tv < R;
  const int64_t chqk = tid < kK ? kh * kK + tid : (p.H + kh) * kK + (tid - kK);
  const int64_t chv = 2 * p.H * kK + hv * kV + row0 + tv;
  const int* si_row = p.si + req * p.si_width;
  int slots[kMaxT];
#pragma unroll
  for (int t = 0; t < kMaxT; ++t) slots[t] = t < p.si_width ? __ldg(si_row + t) : 0;
  const int bos = __ldg(p.cu + req);
  const int eos = __ldg(p.cu + req + 1);
  const int acc = __ldg(p.acc + req);
  const int nv = __ldg(p.cu + N);
  __nv_bfloat16 wqk[4], wv[4];
  if (has_qk) {
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) wqk[jj] = p.cw[chqk * p.cw_dim + jj * p.cw_w];
  }
  if (has_v) {
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) wv[jj] = p.cw[chv * p.cw_dim + jj * p.cw_w];
  }
  float dtb;
  if (p.dtb_type == 1) dtb = __bfloat162float(static_cast<const __nv_bfloat16*>(p.dt_bias)[hv]);
  else if (p.dtb_type == 2) dtb = __half2float(static_cast<const __half*>(p.dt_bias)[hv]);
  else dtb = static_cast<const float*>(p.dt_bias)[hv];
  const float a_log = p.a_log[hv];
  float nwv[NGI];
#pragma unroll
  for (int u = 0; u < NGI; ++u) {
    const int v = (tid + u * NT) % kV;
    nwv[u] = p.norm_w_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.norm_w)[v])
                           : static_cast<const float*>(p.norm_w)[v];
  }

  // ---- loads that need the indices: source state rows, conv window (one round trip) ----
  const int T = eos - bos;
  int src = 0;
#pragma unroll
  for (int t = 0; t < kMaxT; ++t) src = t == acc - 1 ? slots[t] : src;
  const int cslot = slots[0];
  if (tid == 0) {
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) s_slot[t] = slots[t];
  }
  const bool active = T > 0;
  const bool valid = active && src > 0 && T <= kMaxT && T <= p.si_width;
  const bool apply = cslot != p.null_block;
  const int64_t win = static_cast<int64_t>(cslot) * p.cs_slot + static_cast<int64_t>(acc - 1) * p.cs_tok;
  const int64_t cbase = static_cast<int64_t>(cslot) * p.cs_slot;
  float2 h[RPT][kNI][2];
  if (valid) {
    const __nv_bfloat16* sp = p.state + static_cast<int64_t>(src) * p.s_slot + head;
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
#pragma unroll
      for (int i = 0; i < kNI; ++i) {
        const float4 x = ld_bf16x4(sp + (row0 + rg + r * NRG) * kK + i * 4 * kTPR + 4 * j);
        h[r][i][0] = make_float2(x.x, x.y);
        h[r][i][1] = make_float2(x.z, x.w);
      }
    }
  }
  __nv_bfloat16 cqk[3], cv[3];
  if (active && apply && has_qk) {
#pragma unroll
    for (int pp = 0; pp < 3; ++pp) cqk[pp] = p.cs[win + chqk * p.cs_dim + pp * p.cs_tok];
  }
  if (active && apply && has_v) {
#pragma unroll
    for (int pp = 0; pp < 3; ++pp) cv[pp] = p.cs[win + chv * p.cs_dim + pp * p.cs_tok];
  }

  pdl_wait();
  if (stamper) stamp[2] = clock64();

  if (!active) {  // uniform over the cluster (one request): the padding rows, then exit
    for (int row = nv + req * C + c + warp * N * C; row < p.sf_rows; row += NW * N * C) {
      if (lane < kV / 16) {
        if (row < p.q_rows)
          reinterpret_cast<uint4*>(p.q + (static_cast<int64_t>(row) * p.HV + hv) * kV)[lane] =
              make_uint4(0u, 0u, 0u, 0u);
      } else if (lane == kV / 16) {
        *reinterpret_cast<uint32_t*>(p.sf + sf_word(row, hv, p.HV)) = 0u;
      }
    }
    return;
  }

  // ---- this step's rows (one round trip; no global store before it) ----
  const int nmine = T > c ? (T - c + C - 1) / C : 0;  // this CTA's epilogue tokens t = c + C m
  __nv_bfloat16 xqk[kMaxT], xv[kMaxT];
  if (has_qk) {
#pragma unroll
    for (int t = 0; t < kMaxT; ++t)
      if (t < T) xqk[t] = p.x[static_cast<int64_t>(bos + t) * p.s_x + chqk];
  }
  if (has_v) {
#pragma unroll
    for (int t = 0; t < kMaxT; ++t)
      if (t < T) xv[t] = p.x[static_cast<int64_t>(bos + t) * p.s_x + chv];
  }
  __nv_bfloat16 ab, bb;
  if (lane == 0 && warp < T) {
    ab = p.a[static_cast<int64_t>(bos + warp) * p.s_a + hv];
    bb = p.b[static_cast<int64_t>(bos + warp) * p.s_b + hv];
  }
  __nv_bfloat16 zv[NGI];
#pragma unroll
  for (int u = 0; u < NGI; ++u) {
    const int x = tid + u * NT;
    if (x < nmine * kV)
      zv[u] = p.gate[static_cast<int64_t>(bos + c + C * (x / kV)) * p.s_gate + hv * kV + x % kV];
  }

  // ---- conv + SiLU (Triton op order), one channel per thread; raw q/k into s_q / s_k ----
  if (has_qk) {
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) {
      if (t < T) {
        float val;
        if (apply) {
          __nv_bfloat16 u[4];
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const int pos = t + jj;  // window [c0, c1, c2, x0, x1, ...][t : t + 4]
            u[jj] = pos < 3 ? cqk[pos < 3 ? pos : 0] : xqk[pos >= 3 ? pos - 3 : 0];
          }
          val = conv_out(u, wqk, mode);
        } else {
          val = bf(xqk[t]);
        }
        if (tid < kK) s_q[t][tid] = val;
        else s_k[t][tid - kK] = val;
        if (p.dbg != nullptr && rank == 0) p.dbg[static_cast<int64_t>(bos + t) * p.s_x + chqk] = __float2bfloat16(val);
      }
    }
    if (apply && tid / NWB == rank) {  // this CTA rolls channel tid: stage [c1, c2, x0 .. x_{T-1}]
      __nv_bfloat16* rr = s_roll[tid % NWB];
      rr[0] = cqk[1];
      rr[1] = cqk[2];
#pragma unroll
      for (int t = 0; t < kMaxT; ++t)
        if (t < T) rr[2 + t] = xqk[t];
    }
  }
  if (has_v) {
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) {
      if (t < T) {
        float val;
        if (apply) {
          __nv_bfloat16 u[4];
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const int pos = t + jj;
            u[jj] = pos < 3 ? cv[pos < 3 ? pos : 0] : xv[pos >= 3 ? pos - 3 : 0];
          }
          val = conv_out(u, wv, mode);
        } else {
          val = bf(xv[t]);
        }
        s_v[t][tv] = val;
        if (p.dbg != nullptr) p.dbg[static_cast<int64_t>(bos + t) * p.s_x + chv] = __float2bfloat16(val);
      }
    }
  }
  // norm weight x activated output gate of this CTA's epilogue tokens (gmr's expression)
#pragma unroll
  for (int u = 0; u < NGI; ++u) {
    const int x = tid + u * NT;
    if (x < nmine * kV) {
      const float z = __bfloat162float(zv[u]);
      s_g[c + C * (x / kV)][x % kV] = nwv[u] * (p.sigmoid_gate ? sigmoid_f(z) : z * sigmoid_f(z));
    }
  }
  __syncthreads();
  if constexpr (C == 1) cluster_arrive_relaxed();  // this CTA has read the q/k window (rolled after the wait)
  if (apply && has_v) {  // roll this CTA's v channels (only this thread reads them)
    __nv_bfloat16* wp = p.cs + cbase + chv * p.cs_dim;
    wp[0] = cv[1];
    wp[p.cs_tok] = cv[2];
#pragma unroll
    for (int t = 0; t < kMaxT; ++t)
      if (t < T) wp[(2 + t) * p.cs_tok] = xv[t];
  }
  // ---- gmr's q/k L2 norms (same lanes, same order) and gates of token `warp` ----
  if (warp < T) {
    const int t = warp;
    float qv[4], kv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      qv[i] = s_q[t][lane + 32 * i];
      kv[i] = s_k[t][lane + 32 * i];
    }
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
      s_q[t][lane + 32 * i] = qv[i] * qs;
      s_k[t][lane + 32 * i] = kv[i] * ks;
    }
    if (lane == 0) {
      const float gg = -__expf(a_log) * softplus_f(__bfloat162float(ab) + dtb);
      s_dec[t] = __expf(gg);
      s_beta[t] = sigmoid_f(__bfloat162float(bb));
    }
  }
  __syncthreads();
  if (stamper) stamp[3] = clock64();

  // ---- gated delta rule: gmr's per-row arithmetic (RS = 1 reduce-scatter), rows rg + r * NRG of the block ----
  if (valid) {
    float hkr[RPT];
    {
      float2 a[RPT][2];
#pragma unroll
      for (int r = 0; r < RPT; ++r) a[r][0] = a[r][1] = make_float2(0.0f, 0.0f);
#pragma unroll
      for (int i = 0; i < kNI; ++i) {
        const float4 k4 = *reinterpret_cast<const float4*>(s_k[0] + i * 4 * kTPR + 4 * j);
        const float2 kc2[2] = {make_float2(k4.x, k4.y), make_float2(k4.z, k4.w)};
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
#pragma unroll
          for (int cc = 0; cc < 2; ++cc) a[r][i & 1] = fma2(h[r][i][cc], kc2[cc], a[r][i & 1]);
        }
      }
      float nvv[RPT];
#pragma unroll
      for (int r = 0; r < RPT; ++r) nvv[r] = sum4(a[r]);
      int rho_n = 0;
      const float n = RowSum<RPT, kTPR / 2>::run(nvv, j, rho_n);
      if ((j & (kTPR / RPT - 1)) == 0) s_n[1][rg][rho_n] = n;
      __syncwarp();
#pragma unroll
      for (int r = 0; r < RPT; ++r) hkr[r] = s_n[1][rg][r];
    }
    const auto token = [&](const int t, auto next) {
      constexpr bool kNext = decltype(next)::value;
      const float d = s_dec[t];
      const float2 d2 = make_float2(d, d);
      const float bt = s_beta[t];
      const float* kt = s_k[t];
      const float* qt = s_q[t];
      const float* kn = s_k[t + 1 < T ? t + 1 : t];
      float2 dl[RPT];
#pragma unroll
      for (int r = 0; r < RPT; ++r) {
        const float x = (s_v[t][rg + r * NRG] - d * hkr[r]) * bt;
        dl[r] = make_float2(x, x);
      }
      float2 aq[RPT][2], ak[RPT][2];
#pragma unroll
      for (int r = 0; r < RPT; ++r) aq[r][0] = aq[r][1] = ak[r][0] = ak[r][1] = make_float2(0.0f, 0.0f);
#pragma unroll
      for (int i = 0; i < kNI; ++i) {
        const float4 k4 = *reinterpret_cast<const float4*>(kt + i * 4 * kTPR + 4 * j);
        const float4 q4 = *reinterpret_cast<const float4*>(qt + i * 4 * kTPR + 4 * j);
        const float2 kc2[2] = {make_float2(k4.x, k4.y), make_float2(k4.z, k4.w)};
        const float2 qc2[2] = {make_float2(q4.x, q4.y), make_float2(q4.z, q4.w)};
        float2 nc[2];
        if constexpr (kNext) {
          const float4 n4 = *reinterpret_cast<const float4*>(kn + i * 4 * kTPR + 4 * j);
          nc[0] = make_float2(n4.x, n4.y);
          nc[1] = make_float2(n4.z, n4.w);
        }
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
#pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            h[r][i][cc] = fma2(kc2[cc], dl[r], mul2(h[r][i][cc], d2));
            aq[r][i & 1] = fma2(h[r][i][cc], qc2[cc], aq[r][i & 1]);
            if constexpr (kNext) ak[r][i & 1] = fma2(h[r][i][cc], nc[cc], ak[r][i & 1]);
          }
        }
      }
      // snapshot after token t (issued before the reductions; bitwise either way)
      const int dst = s_slot[t];
      if (dst > 0) {
        __nv_bfloat16* dp = p.state + static_cast<int64_t>(dst) * p.s_slot + head;
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
#pragma unroll
          for (int i = 0; i < kNI; ++i)
            st_bf16x4(dp + (row0 + rg + r * NRG) * kK + i * 4 * kTPR + 4 * j,
                      make_float4(h[r][i][0].x, h[r][i][0].y, h[r][i][1].x, h[r][i][1].y));
        }
      }
      float ov[RPT], nvv[RPT];
#pragma unroll
      for (int r = 0; r < RPT; ++r) {
        ov[r] = sum4(aq[r]);
        if constexpr (kNext) nvv[r] = sum4(ak[r]);
      }
      int rho = 0;
      const float o = RowSum<RPT, kTPR / 2>::run(ov, j, rho);
      if ((j & (kTPR / RPT - 1)) == 0) s_o[t][rg + rho * NRG] = __bfloat162float(__float2bfloat16(o));
      if constexpr (kNext) {
        int rho_n = 0;
        const float n = RowSum<RPT, kTPR / 2>::run(nvv, j, rho_n);
        if ((j & (kTPR / RPT - 1)) == 0) s_n[t & 1][rg][rho_n] = n;
        __syncwarp();
#pragma unroll
        for (int r = 0; r < RPT; ++r) hkr[r] = s_n[t & 1][rg][r];
      }
    };
    for (int t = 0; t < T; ++t) {
      if (t + 1 < T) token(t, std::true_type{});
      else token(t, std::false_type{});
    }
  }
  if (stamper) stamp[4] = clock64();
  pdl_trigger();
  // out_proj activation rows [nv, sf_rows) belong to no request: zero values and scales. CTA (req, kh, g, c)
  // zeroes head hv of rows nv + req * C + c + k * N * C, one row per warp (every (row, head) exactly once).
  for (int row = nv + req * C + c + warp * N * C; row < p.sf_rows; row += NW * N * C) {
    if (lane < kV / 16) {
      if (row < p.q_rows)
        reinterpret_cast<uint4*>(p.q + (static_cast<int64_t>(row) * p.HV + hv) * kV)[lane] = make_uint4(0u, 0u, 0u, 0u);
    } else if (lane == kV / 16) {
      *reinterpret_cast<uint32_t*>(p.sf + sf_word(row, hv, p.HV)) = 0u;
    }
  }

  // ---- every CTA of the cluster has read the q/k window; C > 1: the head's raw output rows are written ----
  if constexpr (C == 1) {
    __syncthreads();
    cluster_wait();
  } else {
    cluster_arrive();
    cluster_wait();
  }
  if (stamper) stamp[5] = clock64();
  if (apply && tid < NWB) {  // roll the q/k channels this CTA owns (staged in s_roll)
    const int e = rank * NWB + tid;
    const int64_t wch = e < kK ? kh * kK + e : (p.H + kh) * kK + (e - kK);
    __nv_bfloat16* wp = p.cs + cbase + wch * p.cs_dim;
#pragma unroll
    for (int t = 0; t < kMaxT + 2; ++t)
      if (t < T + 2) wp[t * p.cs_tok] = s_roll[tid][t];
  }
  // ---- gated RMSNorm + MXFP8 of tokens t = c + C * warp (gmr's epilogue on the gathered head) ----
  if (warp < nmine) {
    const int t = c + C * warp;
    const int row = bos + t;
    if (valid) {
      float ov[4];
      float ss = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int v = lane + 32 * i;
        const float* rp;
        if constexpr (C == 1) {
          rp = &s_o[0][0];
        } else {
          rp = cg::this_cluster().map_shared_rank(&s_o[0][0], static_cast<unsigned>(g * C + v / R));
        }
        ov[i] = rp[t * R + v % R];
        s_full[t][v] = ov[i];
        ss += ov[i] * ov[i];
      }
      __syncwarp();
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
      const float rstd = rsqrtf(ss / static_cast<float>(kV) + p.eps);
      float y[4];
      float amax = 0.0f;
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        const int v = 4 * lane + cc;
        y[cc] = __bfloat162float(__float2bfloat16(s_full[t][v] * rstd * s_g[t][v]));
        amax = fmaxf(amax, fabsf(y[cc]));
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
      const uint32_t ex = ue8m0(amax);
      uint32_t packed = 0u;
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) packed |= static_cast<uint32_t>(e4m3(y[cc], ex)) << (8 * cc);
      reinterpret_cast<uint32_t*>(p.q + (static_cast<int64_t>(row) * p.HV + hv) * kV)[lane] = packed;
      if ((lane & 7) == 0) p.sf[sf_word(row, hv, p.HV) + lane / 8] = static_cast<uint8_t>(ex);
    } else {
      // no valid source state: the quant of a zero output (gmr)
      if (lane < kV / 16)
        reinterpret_cast<uint4*>(p.q + (static_cast<int64_t>(row) * p.HV + hv) * kV)[lane] = make_uint4(0u, 0u, 0u, 0u);
      else if (lane == kV / 16)
        *reinterpret_cast<uint32_t*>(p.sf + sf_word(row, hv, p.HV)) = 0u;
    }
  }
  if constexpr (C > 1) {
    // the siblings may still read this CTA's s_o (their reads have returned once they arrive)
    cluster_arrive_relaxed();
    cluster_wait();
  }
  if (stamper) {
    stamp[6] = clock64();
    stamp[7] = gtimer();
  }
}

static bool fill_params(Params& p, torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
                        torch::Tensor dt_bias, torch::Tensor si, torch::Tensor cu, torch::Tensor acc,
                        torch::Tensor state, torch::Tensor cs, torch::Tensor cw, torch::Tensor gate,
                        torch::Tensor norm_w, torch::Tensor q, torch::Tensor sf, torch::Tensor dbg,
                        torch::Tensor stamp, double scale, double eps, bool sigmoid_gate, int64_t null_block,
                        int64_t silu_mode) {
  if (!x.is_cuda()) return false;
  for (const torch::Tensor* t : {&a, &b, &a_log, &dt_bias, &si, &cu, &acc, &state, &cs, &cw, &gate, &norm_w, &q, &sf})
    if (t->device() != x.device()) return false;
  if (x.scalar_type() != at::kBFloat16 || a.scalar_type() != at::kBFloat16 || b.scalar_type() != at::kBFloat16 ||
      gate.scalar_type() != at::kBFloat16 || state.scalar_type() != at::kBFloat16 ||
      cs.scalar_type() != at::kBFloat16 || cw.scalar_type() != at::kBFloat16)
    return false;
  if (a_log.scalar_type() != at::kFloat || !a_log.is_contiguous()) return false;
  const auto dtb = dt_bias.scalar_type();
  if ((dtb != at::kFloat && dtb != at::kBFloat16 && dtb != at::kHalf) || !dt_bias.is_contiguous()) return false;
  if ((norm_w.scalar_type() != at::kFloat && norm_w.scalar_type() != at::kBFloat16) || !norm_w.is_contiguous() ||
      norm_w.numel() != kV)
    return false;
  if (state.dim() != 4 || state.size(2) != kV || state.size(3) != kK || state.stride(3) != 1 ||
      state.stride(2) != kK || state.stride(1) != kV * kK || state.stride(0) % 8 != 0 ||
      (reinterpret_cast<uintptr_t>(state.data_ptr()) % 16) != 0)
    return false;
  const int HV = (int)state.size(1);
  if (x.dim() != 2 || x.stride(1) != 1) return false;
  const int64_t kw = x.size(1) - (int64_t)HV * kV;
  if (kw <= 0 || kw % (2 * kK) != 0) return false;
  const int H = (int)(kw / (2 * kK));
  if (HV != kG * H) return false;
  if (cs.dim() != 3 || cs.size(1) != x.size(1) || cw.dim() != 2 || cw.size(0) != x.size(1) || cw.size(1) != 4)
    return false;
  if (si.scalar_type() != at::kInt || cu.scalar_type() != at::kInt || acc.scalar_type() != at::kInt) return false;
  if (!si.is_contiguous() || !cu.is_contiguous() || !acc.is_contiguous()) return false;
  if (si.dim() != 2 || si.size(1) < 1 || si.size(1) > kMaxT) return false;
  if (cs.size(2) < 3 + si.size(1) - 1) return false;
  const int N = (int)si.size(0);
  if (cu.numel() != N + 1 || acc.numel() != N) return false;
  const int64_t L = x.size(0);
  if (a.dim() != 2 || b.dim() != 2 || a.size(0) != L || b.size(0) != L || a.size(1) != HV || b.size(1) != HV ||
      a.stride(1) != 1 || b.stride(1) != 1)
    return false;
  if (a_log.numel() != HV || dt_bias.numel() != HV) return false;
  if (gate.dim() != 3 || gate.size(0) != L || gate.size(1) != HV || gate.size(2) != kV || gate.stride(2) != 1 ||
      gate.stride(1) != kV)
    return false;
  if (q.scalar_type() != at::kFloat8_e4m3fn || q.dim() != 2 || q.size(0) < L || q.size(1) != (int64_t)HV * kV ||
      !q.is_contiguous() || (reinterpret_cast<uintptr_t>(q.data_ptr()) % 16) != 0)
    return false;
  const int64_t sf_rows = (q.size(0) + 127) / 128 * 128;
  if (sf.scalar_type() != at::kByte || !sf.is_contiguous() || sf.numel() != sf_rows * HV * 4) return false;
  const bool has_dbg = dbg.defined() && dbg.numel() > 0;
  if (has_dbg && (dbg.scalar_type() != at::kBFloat16 || dbg.sizes() != x.sizes() || dbg.strides() != x.strides()))
    return false;
  const bool has_stamp = stamp.defined() && stamp.numel() > 0;
  if (has_stamp && (stamp.scalar_type() != at::kLong || !stamp.is_contiguous())) return false;
  p.x = (const __nv_bfloat16*)x.data_ptr();
  p.a = (const __nv_bfloat16*)a.data_ptr();
  p.b = (const __nv_bfloat16*)b.data_ptr();
  p.a_log = (const float*)a_log.data_ptr();
  p.dt_bias = dt_bias.data_ptr();
  p.si = (const int*)si.data_ptr();
  p.cu = (const int*)cu.data_ptr();
  p.acc = (const int*)acc.data_ptr();
  p.state = (__nv_bfloat16*)state.data_ptr();
  p.cs = (__nv_bfloat16*)cs.data_ptr();
  p.cw = (const __nv_bfloat16*)cw.data_ptr();
  p.gate = (const __nv_bfloat16*)gate.data_ptr();
  p.norm_w = norm_w.data_ptr();
  p.q = (uint8_t*)q.data_ptr();
  p.sf = (uint8_t*)sf.data_ptr();
  p.dbg = has_dbg ? (__nv_bfloat16*)dbg.data_ptr() : nullptr;
  p.stamp = has_stamp ? (uint64_t*)stamp.data_ptr() : nullptr;
  p.s_x = x.stride(0);
  p.s_a = a.stride(0);
  p.s_b = b.stride(0);
  p.s_gate = gate.stride(0);
  p.s_slot = state.stride(0);
  p.cs_slot = cs.stride(0);
  p.cs_dim = cs.stride(1);
  p.cs_tok = cs.stride(2);
  p.cw_dim = cw.stride(0);
  p.cw_w = cw.stride(1);
  p.si_width = (int)si.size(1);
  p.H = H;
  p.HV = HV;
  p.null_block = (int)null_block;
  p.dtb_type = dtb == at::kFloat ? 0 : (dtb == at::kBFloat16 ? 1 : 2);
  p.norm_w_bf16 = norm_w.scalar_type() == at::kBFloat16;
  p.sigmoid_gate = sigmoid_gate;
  p.silu_mode = (int)silu_mode;
  p.q_rows = (int)q.size(0);
  p.sf_rows = (int)sf_rows;
  p.scale = (float)scale;
  p.eps = (float)eps;
  return true;
}

template <int C, int RPT, int MINB>
static void launch_gfd(const Params& p, int N, bool pdl, cudaStream_t stream) {
  using F = Cfg<C, RPT>;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(F::CS, p.H, N);
  cfg.blockDim = dim3(F::NT);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  int na = 0;
  attr[na].id = cudaLaunchAttributeClusterDimension;
  attr[na].val.clusterDim.x = F::CS;
  attr[na].val.clusterDim.y = 1;
  attr[na].val.clusterDim.z = 1;
  ++na;
  if (pdl) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = attr;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, gfd_kernel<C, RPT, MINB>, p));
}

// The served config: C = 1 CTA per (request, value head), RPT = 4 value rows per thread (512 threads), MINB = 1.
constexpr int kC = 1, kRPT = 4, kMINB = 1;
constexpr int kSiluMode = 0;  // ex2.approx + div.full: bitwise to _causal_conv1d_update_kernel

// False (nothing launched) when the layout contract is not met; the caller keeps the stock path.
bool run(torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias,
         torch::Tensor si, torch::Tensor cu, torch::Tensor acc, torch::Tensor state, torch::Tensor cs,
         torch::Tensor cw, torch::Tensor gate, torch::Tensor norm_w, torch::Tensor q, torch::Tensor sf,
         double scale, double eps, bool sigmoid_gate, int64_t null_block, bool pdl) {
  Params p;
  if (!fill_params(p, x, a, b, a_log, dt_bias, si, cu, acc, state, cs, cw, gate, norm_w, q, sf, torch::Tensor(),
                   torch::Tensor(), scale, eps, sigmoid_gate, null_block, kSiluMode))
    return false;
  const int N = (int)si.size(0);
  if (N == 0) return true;
  const c10::cuda::CUDAGuard guard(x.device());
  launch_gfd<kC, kRPT, kMINB>(p, N, pdl, c10::cuda::getCurrentCUDAStream());
  return true;
}

// {threads, cluster size, registers, static smem bytes, local (spill) bytes} of the served config.
std::vector<int64_t> cfg_info() {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, gfd_kernel<kC, kRPT, kMINB>));
  return {Cfg<kC, kRPT>::NT, Cfg<kC, kRPT>::CS, fa.numRegs, (int64_t)fa.sharedSizeBytes, (int64_t)fa.localSizeBytes};
}

}  // namespace gdn_gfd

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &gdn_gfd::run);
  m.def("cfg_info", &gdn_gfd::cfg_info);
}
"""
