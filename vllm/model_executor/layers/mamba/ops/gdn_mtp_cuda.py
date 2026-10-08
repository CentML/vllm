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

``run_quant`` (VLLM_GDN_MTP_FUSED_QUANT builds) writes out_proj's MXFP8
activation instead of the bf16 output: the e4m3 values and F8_128x4 UE8M0
scales that ``gdn_gated_norm_mxfp8(norm_rows=(0, 0))`` writes from the bf16
output (bitwise), including the zeros of the rows no request owns, so that
separate quant launch is dropped.

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

_ext: list = []  # [module] once load() built or loaded it (stock / tuned)
_wy: list = []  # [module] of the WY kernel (VLLM_GDN_MTP_CUDA_WY)
_tried: list = []

# VLLM_GDN_MTP_CUDA_PDL=1: build the kernel with ``griddepcontrol.wait`` as its
# first instruction and launch it with programmatic stream serialization, so
# it launches while the preceding conv1d update (which triggers its dependents
# early) still runs and waits for it before any global load. Results are
# unchanged. Off (default): the source and launch are the stock ones.
GDN_MTP_CUDA_PDL = os.environ.get("VLLM_GDN_MTP_CUDA_PDL", "0") == "1"

# VLLM_GDN_MTP_CUDA_TUNE="RS=1+LASTN=1+FADD2=1+PF=212" (items separated by "+"
# or ","): build the kernel with these tuning macros (see the GMR_* block of
# _SOURCE: rows per thread, min CTAs per SM, reduce-scatter key reductions,
# early snapshot stores, L2 prefetch distance, last-token skip, packed adds).
# Every combination computes the same per-element operations and reduction
# trees, so results are bitwise equal to the stock kernel. Empty (default):
# the stock kernel.
GDN_MTP_CUDA_TUNE = os.environ.get("VLLM_GDN_MTP_CUDA_TUNE", "")

# VLLM_GDN_MTP_FUSED_QUANT=1: also build the kernel variant whose epilogue
# writes out_proj's MXFP8 activation (e4m3 + swizzled UE8M0 scales, every row of
# the activation including the padding) instead of the bf16 output, bitwise
# equal to the separate gdn_gated_norm_mxfp8 launch it replaces in decode-only
# spec batches. Off (default): that variant is not built.
GDN_MTP_FUSED_QUANT = os.environ.get("VLLM_GDN_MTP_FUSED_QUANT", "0") == "1"
# VLLM_GDN_MTP_CUDA_WY=1 (or "K=V+..." macros of the GWY_* block of _WY_SOURCE):
# also build the WY kernel (same contract and signature, bf16 state only) and
# run it for calls with at least VLLM_GDN_MTP_CUDA_WY_MIN_REQS requests; smaller
# calls (and an fp32 state) keep the stock / tuned kernel. The source state sits
# in smem; all S_0 k_t and S_0 q_t of a request are one tensor-core pass (bf16
# state x 3-term (k) / 2-term (q) bf16 splits, fp32 accumulation); each row
# solves the T x T triangular WY system for its delta_t in registers, then
# streams the T snapshots with the stock element update S_t = d_t S_{t-1} +
# k_t delta_t. Not bitwise to the stock kernel (different reduction order:
# float-order class). Empty / "0" (default): the stock (or tuned) kernel only.
# VR (T = 5/6, vs RS=1+PF=212): WY -6 ... -18% at >= 36 requests, within
# -13 ... +6% below (grid-wave effects of 4 vs 2 CTAs per SM).
GDN_MTP_CUDA_WY = os.environ.get("VLLM_GDN_MTP_CUDA_WY", "")
if GDN_MTP_CUDA_WY == "0":
    GDN_MTP_CUDA_WY = ""
GDN_MTP_CUDA_WY_MIN_REQS = int(os.environ.get("VLLM_GDN_MTP_CUDA_WY_MIN_REQS", "36"))


def tuned_source(tune: str, pdl: bool, quant: bool = False) -> str:
    """The kernel source with the ``tune`` macros (``"K=V+..."``) prepended;
    ``quant``: also the fused-quant variant (``run_quant``).
    """
    source = _pdl_source(_SOURCE) if pdl else _SOURCE
    defines = "#define GMR_QO 1\n" if quant else ""
    for item in filter(None, (x.strip() for x in tune.replace(",", "+").split("+"))):
        key, val = item.split("=")
        assert key in ("RPT", "MINB", "RS", "EARLY_ST", "PF", "LASTN", "FADD2"), key
        defines += f"#define GMR_{key} {int(val)}\n"
    return defines + source


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


def build_dir(arch: str) -> str:
    return os.path.join(envs.VLLM_CACHE_ROOT, "gdn_mtp_cuda", f"sm{arch}")


def load():
    """Build (or load the cached build of) the extension(s) for the current device."""
    if _ext:
        return _ext[0]
    if GDN_MTP_CUDA_WY:
        _wy.append(build(wy_source(GDN_MTP_CUDA_WY, GDN_MTP_CUDA_PDL)))
    ext = build(tuned_source(GDN_MTP_CUDA_TUNE, GDN_MTP_CUDA_PDL, GDN_MTP_FUSED_QUANT))
    _ext.append(ext)
    return ext


def build(source: str):
    """Build (or load the cached build of) ``source`` for the current device."""
    major, minor = torch.cuda.get_device_capability()
    import torch.utils.cpp_extension as cpp

    arch = f"{major}{minor}{'a' if major >= 9 else ''}"
    bdir = build_dir(arch)
    os.makedirs(bdir, exist_ok=True)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    src = os.path.join(bdir, f"gdn_mtp_cuda_{digest}.cu")
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
            name=f"_gdn_mtp_cuda_{digest}",
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
    leaves the csrc kernel in use. Returns whether the kernel is available.
    """
    if not _tried:
        _tried.append(True)
        try:
            load()
            logger.info(
                "GDN MTP decode: register-resident CUDA kernel built and enabled%s%s.",
                f" (tune {GDN_MTP_CUDA_TUNE})" if GDN_MTP_CUDA_TUNE else "",
                " with the fused out_proj MXFP8 quant" if GDN_MTP_FUSED_QUANT else "",
            )
            if GDN_MTP_CUDA_WY:
                logger.info(
                    "GDN MTP decode: WY tensor-core CUDA kernel built and enabled "
                    "(VLLM_GDN_MTP_CUDA_WY=%s) for >= %d requests.",
                    GDN_MTP_CUDA_WY,
                    GDN_MTP_CUDA_WY_MIN_REQS,
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
    out_q: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
) -> bool:
    """Run the kernel; False (nothing launched) if the layout contract is not
    met, so the caller keeps the csrc kernel. Requires ``enable()`` first.

    ``out_q``/``out_scale`` (VLLM_GDN_MTP_FUSED_QUANT): write out_proj's
    MXFP8 activation into them (``[R, HV * V]`` e4m3, R >= the token rows, and
    the flat F8_128x4 scales of R rows) instead of ``out``, which is then not
    written; rows ``>= cu_seqlens[-1]`` get zero values and scales. The WY
    kernel has no fused-quant variant: a call it would take returns False with
    ``out_q`` (nothing launched), so the caller runs it without ``out_q`` and
    then the separate quant launch.
    """
    args = (
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
    )
    tail = (float(scale), float(norm_eps), output_gate_activation == "sigmoid")
    wy = bool(_wy) and state_indices.shape[0] >= GDN_MTP_CUDA_WY_MIN_REQS
    if out_q is not None:
        return not wy and _ext[0].run_quant(*args, out_q, out_scale, *tail)
    if wy and _wy[0].run(*args, *tail):
        return True
    return _ext[0].run(*args, *tail)


_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace gmr {

// Tuning macros (prepended by tuned_source(); defaults = the stock kernel):
// GMR_RPT value rows per thread (8: 256 threads, 4: 512 threads), GMR_MINB
// min CTAs per SM for __launch_bounds__, GMR_RS 1: per-token key reductions as
// a reduce-scatter (same operands and add tree per row as the xor butterfly,
// so bitwise equal; fewer shuffles) with S_t k_{t+1} shared through smem,
// GMR_EARLY_ST 1: issue the snapshot stores before the reductions, GMR_PF N > 0:
// L2-prefetch the source state of the CTA N blocks ahead, GMR_LASTN 1: skip
// S_t k_{t+1} on the last token (unused), GMR_FADD2 1: packed adds in sum4.
// GMR_QO 1: also build the fused-quant kernel (run_quant; kQ below).
#ifndef GMR_RPT
#define GMR_RPT 8
#endif
#ifndef GMR_MINB
#define GMR_MINB 2
#endif
#ifndef GMR_RS
#define GMR_RS 0
#endif
#ifndef GMR_PF
#define GMR_PF 0
#endif
#ifndef GMR_LASTN
#define GMR_LASTN 0
#endif
#ifndef GMR_FADD2
#define GMR_FADD2 0
#endif
#ifndef GMR_EARLY_ST
#define GMR_EARLY_ST 0
#endif
#ifndef GMR_QO
#define GMR_QO 0
#endif

constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kMaxT = 8;
constexpr int kKPT = 8;              // key columns per thread
constexpr int kRPT = GMR_RPT;        // value rows per thread
constexpr int kTPR = kK / kKPT;      // threads per row (16)
constexpr int kNRG = kV / kRPT;      // row groups (16)
constexpr int kNT = kNRG * kTPR;     // threads (256)
constexpr int kNW = kNT / 32;        // warps (8): one per token in the prologue/epilogue
constexpr int kNI = kKPT / 4;        // float4 chunks per row slice (2)
constexpr int kGI = kMaxT * kV / kNT;  // gate elements per thread (4)
static_assert(kNW >= kMaxT, "one warp per token");

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
  // kQ (run_quant): out_proj's MXFP8 activation is written instead of `out`.
  uint8_t* q;   // e4m3 [q_rows, HV * kV], contiguous
  uint8_t* sf;  // F8_128x4 UE8M0 scales of [sf_rows, HV * 4]
  int q_rows;   // rows of q (>= the requests' tokens)
  int sf_rows;  // q_rows rounded up to 128
};

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

// MXFP8 of out_proj's activation, bit-identical to _gdn_gated_norm_mxfp8_kernel
// (qwen_gdn_tail_ops.py; FlashInfer's mxfp8_quantize rule) on the bf16 output:
// per 32-value block the UE8M0 exponent of amax / 448 rounded up, then e4m3 (RN,
// satfinite) of value * 2^(127 - e); IEEE multiplies (no FTZ, no contraction).
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

// Byte offset of the scales of groups 4 * head .. 4 * head + 3 of `row` (one
// 32-bit word) in F8_128x4: [row / 128, group / 4, row % 32, row % 128 / 32,
// group % 4], HV * 4 groups per row (a multiple of 4: no column padding).
__device__ __forceinline__ int64_t sf_word(int row, int head, int HV) {
  return static_cast<int64_t>(row >> 7) * (128 * 4 * HV) + head * 512 + (row & 31) * 16 +
         ((row & 127) >> 5) * 4;
}

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
#if GMR_FADD2 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  // same two pair sums as below, as one packed add
  const float2 s = __fadd2_rn(make_float2(x[0].x, x[1].x), make_float2(x[0].y, x[1].y));
  return s.x + s.y;
#else
  return (x[0].x + x[0].y) + (x[1].x + x[1].y);
#endif
}

// Sum of the R row partials x over the 16 lanes of a row group, as a
// reduce-scatter: at each xor level (off 8, 4, 2, 1) a lane keeps half of its
// rows (chosen by bit `off` of j) and adds the partner's partial of the same
// row, so every row is summed with exactly the operands and order of the xor
// butterfly. Returns the sum of local row `rho` (accumulated into rho).
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

// kQ: write out_proj's MXFP8 activation (p.q, p.sf) of every row of q instead
// of the bf16 output (p.out is not written).
template <typename S, bool kQ>
__global__ void __launch_bounds__(kNT, GMR_MINB) mtp_kernel(const Params p) {
  __shared__ __align__(16) float s_q[kMaxT][kK];
  __shared__ __align__(16) float s_k[kMaxT][kK];
  __shared__ float s_v[kMaxT][kV];
  __shared__ float s_o[kMaxT][kV];
  __shared__ float s_g[kMaxT][kV];  // norm weight x activated output gate
  __shared__ float s_dec[kMaxT];
  __shared__ float s_beta[kMaxT];
#if GMR_RS
  __shared__ __align__(16) float s_n[2][kNRG][kRPT];  // S_t k_{t+1} rows, by token parity
#endif

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
  if constexpr (kQ) {
    // Rows [nv, sf_rows) belong to no request (FULL-graph padding rows, then
    // the 128-row scale padding). As _gdn_gated_norm_mxfp8_kernel does for
    // rows >= num_valid, they get zero values (rows < q_rows) and zero scales:
    // CTA (req, hv) zeroes head hv of rows nv + req + k * N, one row per warp
    // (lanes 0-7: the 128 e4m3 bytes, lane 8: the 4 scales). Issued here,
    // ahead of the state loads (VR: cheaper than in their shadow or at exit).
    const int nv = __ldg(p.cu + gridDim.x);
    const int n = static_cast<int>(gridDim.x);
    for (int row = nv + req + warp * n; row < p.sf_rows; row += kNW * n) {
      if (lane < kV / 16) {
        if (row < p.q_rows)
          reinterpret_cast<uint4*>(p.q + (static_cast<int64_t>(row) * p.HV + hv) * kV)[lane] =
              make_uint4(0u, 0u, 0u, 0u);
      } else if (lane == kV / 16) {
        *reinterpret_cast<uint32_t*>(p.sf + sf_word(row, hv, p.HV)) = 0u;
      }
    }
  }
  if (T <= 0) return;
  int src = 0;
#pragma unroll
  for (int t = 0; t < kMaxT; ++t) src = t == acc - 1 ? slots[t] : src;
  if (src <= 0 || T > kMaxT) {
    if constexpr (kQ) {
      // the quant of the zero output: zero values and scales
      for (int x = tid; x < T * (kV / 16); x += kNT)
        reinterpret_cast<uint4*>(p.q + (static_cast<int64_t>(bos + x / (kV / 16)) * p.HV + hv) * kV)[x % (kV / 16)] =
            make_uint4(0u, 0u, 0u, 0u);
      for (int x = tid; x < T; x += kNT) *reinterpret_cast<uint32_t*>(p.sf + sf_word(bos + x, hv, p.HV)) = 0u;
    } else {
      for (int x = tid; x < T * kV; x += kNT)
        p.out[(static_cast<int64_t>(bos + x / kV) * p.HV + hv) * kV + x % kV] = __float2bfloat16(0.0f);
    }
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
#if GMR_PF
  // L2 prefetch of the source state of the CTA GMR_PF blocks ahead in launch
  // order (a hint only; no effect on results), by the last warp.
  if (warp == kNW - 1) {
    const int f = blockIdx.x + blockIdx.y * gridDim.x + GMR_PF;
    if (f < static_cast<int>(gridDim.x * gridDim.y)) {
      const int freq = f % gridDim.x;
      const int fhv = f / gridDim.x;
      const int facc = __ldg(p.acc + freq);
      const int fsrc = facc >= 1 && facc <= p.si_width ? __ldg(p.si + freq * p.si_width + facc - 1) : 0;
      if (fsrc > 0) {
        constexpr int kBytes = kV * kK * static_cast<int>(sizeof(S));
        const char* fp = reinterpret_cast<const char*>(static_cast<const S*>(p.state) +
                                                       static_cast<int64_t>(fsrc) * p.s_slot +
                                                       static_cast<int64_t>(fhv) * kV * kK) +
                         lane * (kBytes / 32);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(fp), "r"(kBytes / 32) : "memory");
      }
    }
  }
#endif

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
#if GMR_RS
    // same tree as below, via RowSum; buffer 1 (token 0 writes buffer 0)
    float nv[kRPT];
#pragma unroll
    for (int r = 0; r < kRPT; ++r) nv[r] = sum4(a[r]);
    int rho_n = 0;
    const float n = RowSum<kRPT, kTPR / 2>::run(nv, j, rho_n);
    if ((j & (kTPR / kRPT - 1)) == 0) s_n[1][rg][rho_n] = n;
    __syncwarp();
#pragma unroll
    for (int r = 0; r < kRPT; r += 4) {
      const float4 n4 = *reinterpret_cast<const float4*>(&s_n[1][rg][r]);
      hkr[r] = n4.x;
      hkr[r + 1] = n4.y;
      hkr[r + 2] = n4.z;
      hkr[r + 3] = n4.w;
    }
#else
#pragma unroll
    for (int r = 0; r < kRPT; ++r) {
      hkr[r] = sum4(a[r]);
#pragma unroll
      for (int off = kTPR / 2; off > 0; off >>= 1) hkr[r] += __shfl_xor_sync(0xffffffffu, hkr[r], off);
    }
#endif
  }

  S* const state = static_cast<S*>(p.state);
  // One token; kNext: also compute S_t k_{t+1} (hkr for the next token).
  const auto token = [&](const int t, auto next) {
    constexpr bool kNext = decltype(next)::value;
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
      const float2 kc[2] = {make_float2(k4.x, k4.y), make_float2(k4.z, k4.w)};
      const float2 qc[2] = {make_float2(q4.x, q4.y), make_float2(q4.z, q4.w)};
      float2 nc[2];
      if constexpr (kNext) {
        const float4 n4 = *reinterpret_cast<const float4*>(kn + i * 4 * kTPR + 4 * j);
        nc[0] = make_float2(n4.x, n4.y);
        nc[1] = make_float2(n4.z, n4.w);
      }
#pragma unroll
      for (int r = 0; r < kRPT; ++r) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          h[r][i][c] = fma2(kc[c], dl[r], mul2(h[r][i][c], d2));
          aq[r][i & 1] = fma2(h[r][i][c], qc[c], aq[r][i & 1]);
          if constexpr (kNext) ak[r][i & 1] = fma2(h[r][i][c], nc[c], ak[r][i & 1]);
        }
      }
    }
    const auto store = [&]() {
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
    };
#if GMR_EARLY_ST
    store();
#endif
#if GMR_RS
    {
      float ov[kRPT], nv[kRPT];
#pragma unroll
      for (int r = 0; r < kRPT; ++r) {
        ov[r] = sum4(aq[r]);
        if constexpr (kNext) nv[r] = sum4(ak[r]);
      }
      int rho = 0;
      const float o = RowSum<kRPT, kTPR / 2>::run(ov, j, rho);
      if ((j & (kTPR / kRPT - 1)) == 0) s_o[t][rg + rho * kNRG] = __bfloat162float(__float2bfloat16(o));
      if constexpr (kNext) {
        int rho_n = 0;
        const float n = RowSum<kRPT, kTPR / 2>::run(nv, j, rho_n);
        if ((j & (kTPR / kRPT - 1)) == 0) s_n[t & 1][rg][rho_n] = n;
        __syncwarp();
#pragma unroll
        for (int r = 0; r < kRPT; r += 4) {
          const float4 n4 = *reinterpret_cast<const float4*>(&s_n[t & 1][rg][r]);
          hkr[r] = n4.x;
          hkr[r + 1] = n4.y;
          hkr[r + 2] = n4.z;
          hkr[r + 3] = n4.w;
        }
      }
    }
#else
#pragma unroll
    for (int r = 0; r < kRPT; ++r) {
      float o = sum4(aq[r]);
      float n = 0.0f;
      if constexpr (kNext) n = sum4(ak[r]);
#pragma unroll
      for (int off = kTPR / 2; off > 0; off >>= 1) {
        o += __shfl_xor_sync(0xffffffffu, o, off);
        if constexpr (kNext) n += __shfl_xor_sync(0xffffffffu, n, off);
      }
      if constexpr (kNext) hkr[r] = n;
      if (j == 0) s_o[t][rg + r * kNRG] = __bfloat162float(__float2bfloat16(o));
    }
#endif
#if !GMR_EARLY_ST
    store();
#endif
  };
  for (int t = 0; t < T; ++t) {
#if GMR_LASTN
    if (t + 1 < T) token(t, std::true_type{});
    else token(t, std::false_type{});
#else
    token(t, std::true_type{});
#endif
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
    if constexpr (kQ) {
      // Quantize the bf16 value the two-kernel path stores and re-reads, with
      // lane holding values 4 * lane .. 4 * lane + 3: an MXFP8 block is 8 lanes
      // (3 shuffle levels), one 32-bit e4m3 store per lane, and lane 8 * i
      // stores the scale byte of block i. Same per-value product as below.
      const int row = bos + t;
      float y[4];
      float amax = 0.0f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int v = 4 * lane + c;
        y[c] = __bfloat162float(__float2bfloat16(s_o[t][v] * rstd * s_g[t][v]));
        amax = fmaxf(amax, fabsf(y[c]));
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
      const uint32_t e = ue8m0(amax);
      uint32_t packed = 0u;
#pragma unroll
      for (int c = 0; c < 4; ++c) packed |= static_cast<uint32_t>(e4m3(y[c], e)) << (8 * c);
      reinterpret_cast<uint32_t*>(p.q + (static_cast<int64_t>(row) * p.HV + hv) * kV)[lane] = packed;
      if ((lane & 7) == 0) p.sf[sf_word(row, hv, p.HV) + lane / 8] = static_cast<uint8_t>(e);
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int v = lane + 32 * i;
        p.out[(static_cast<int64_t>(bos + t) * p.HV + hv) * kV + v] = __float2bfloat16(ov[i] * rstd * s_g[t][v]);
      }
    }
  }
}

template <typename S, bool kQ>
void launch(const dim3 grid, const cudaStream_t stream, const Params& p) {
  mtp_kernel<S, kQ><<<grid, kNT, 0, stream>>>(p);
}

// false if the layout contract is not met (caller keeps the csrc kernel).
// q, sf (run_quant): out_proj's e4m3 activation [>= L, HV * kV] and its flat
// F8_128x4 scales, written instead of `out`.
bool run_impl(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
              torch::Tensor dt_bias, torch::Tensor si, torch::Tensor cu, torch::Tensor acc,
              torch::Tensor state, torch::Tensor gate, torch::Tensor norm_w, torch::Tensor out,
              const torch::Tensor* q, const torch::Tensor* sf, double scale, double eps, bool sigmoid_gate) {
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
  int64_t sf_rows = 0;
  if (q != nullptr) {
    if (q->scalar_type() != at::kFloat8_e4m3fn || q->dim() != 2 || q->size(0) < L ||
        q->size(1) != (int64_t)HV * kV || !q->is_contiguous() ||
        (reinterpret_cast<uintptr_t>(q->data_ptr()) % 16) != 0)
      return false;
    sf_rows = (q->size(0) + 127) / 128 * 128;
    if (sf->scalar_type() != at::kByte || !sf->is_contiguous() || sf->numel() != sf_rows * HV * 4 ||
        (reinterpret_cast<uintptr_t>(sf->data_ptr()) % 4) != 0)
      return false;
    // The grid also zeroes the rows no request owns; without requests there is
    // no grid, so the caller keeps the separate quant.
    if (N == 0) return false;
  }
  if (N == 0 || (L == 0 && q == nullptr)) return true;
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
  p.q = q != nullptr ? (uint8_t*)q->data_ptr() : nullptr;
  p.sf = q != nullptr ? (uint8_t*)sf->data_ptr() : nullptr;
  p.q_rows = q != nullptr ? (int)q->size(0) : 0;
  p.sf_rows = (int)sf_rows;
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
  if (q == nullptr) {
    if (st == at::kFloat)
      launch<float, false>(grid, stream, p);
    else
      launch<__nv_bfloat16, false>(grid, stream, p);
  }
#if GMR_QO
  else if (st == at::kFloat)
    launch<float, true>(grid, stream, p);
  else
    launch<__nv_bfloat16, true>(grid, stream, p);
#endif
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

bool run(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
         torch::Tensor dt_bias, torch::Tensor si, torch::Tensor cu, torch::Tensor acc,
         torch::Tensor state, torch::Tensor gate, torch::Tensor norm_w, torch::Tensor out,
         double scale, double eps, bool sigmoid_gate) {
  return run_impl(qkv, a, b, a_log, dt_bias, si, cu, acc, state, gate, norm_w, out, nullptr, nullptr, scale,
                  eps, sigmoid_gate);
}

#if GMR_QO
bool run_quant(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
               torch::Tensor dt_bias, torch::Tensor si, torch::Tensor cu, torch::Tensor acc,
               torch::Tensor state, torch::Tensor gate, torch::Tensor norm_w, torch::Tensor out,
               torch::Tensor q, torch::Tensor sf, double scale, double eps, bool sigmoid_gate) {
  return run_impl(qkv, a, b, a_log, dt_bias, si, cu, acc, state, gate, norm_w, out, &q, &sf, scale, eps,
                  sigmoid_gate);
}
#endif

}  // namespace gmr

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &gmr::run);
#if GMR_QO
  m.def("run_quant", &gmr::run_quant);
#endif
}
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
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, mtp_kernel<S, kQ>, p));
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
            "  mtp_kernel<S, kQ><<<grid, kNT, 0, stream>>>(p);\n",
            _PDL_LAUNCH,
        ),
    )
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


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
