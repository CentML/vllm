# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""CUDA version of gdn_fused_conv_prep (SM10x), JIT-built at the GDN prefill warmup.

Same contract and bitwise-identical outputs (q/k/v, exp(g), beta, conv state) as the
Triton kernel in gdn_fused_conv_prep.py, i.e. as causal_conv1d_fn + fused_post_conv_prep:
every floating-point instruction mirrors the Triton PTX (see the source header). Faster
than the Triton kernel on VR (16-byte vector accesses, a register rolling window over the
taps; 3584 tokens: 20.4 vs 32.4 us per layer).

The CUDA source is kept in this module (``_SOURCE``) so the package ships only Python;
``load()`` writes it into the build directory and builds it with
torch.utils.cpp_extension (nvcc, ninja) for the current device's arch-specific target.
Concurrent processes share the build directory (cpp_extension's file lock): one builds,
the others wait and load. ``build_dir()`` is under VLLM_CACHE_ROOT.
"""

import hashlib
import os

import torch

from vllm import envs

_ext: list = []  # [module] once load() built or loaded it


def build_dir(arch: str, mixed_add: int) -> str:
    return os.path.join(
        envs.VLLM_CACHE_ROOT, "gdn_conv_cuda", f"sm{arch}_ma{mixed_add}"
    )


def load():
    """Build (or load the cached build of) the extension for the current device.

    MIXED_ADD (the sm_107 ``add.rn.f32x2.bf16x2.f32x2`` accumulate) is used on
    sm_107 only; it is bitwise equal to the unpack + ``add.f32`` path.
    """
    if _ext:
        return _ext[0]
    major, minor = torch.cuda.get_device_capability()
    mixed_add = int((major, minor) == (10, 7))
    import torch.utils.cpp_extension as cpp

    arch = f"{major}{minor}{'a' if major >= 9 else ''}"
    build = build_dir(arch, mixed_add)
    os.makedirs(build, exist_ok=True)
    digest = hashlib.sha256(_SOURCE.encode()).hexdigest()[:16]
    src = os.path.join(build, f"gdn_conv_cuda_{digest}.cu")
    if not os.path.exists(src):
        tmp = f"{src}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(_SOURCE)
        os.replace(tmp, src)
    orig = cpp._get_cuda_arch_flags
    cpp._get_cuda_arch_flags = lambda cflags=None: [
        f"-gencode=arch=compute_{arch},code=sm_{arch}"
    ]
    try:
        ext = cpp.load(
            name=f"_gdn_conv_cuda_{digest}_ma{mixed_add}",
            sources=[src],
            # No fast-math, no FMA contraction: every fp op is explicit PTX
            # mirroring the Triton kernel.
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "-lineinfo",
                "-fmad=false",
                f"-DGK2_MIXED_ADD={mixed_add}",
            ],
            extra_cflags=["-O3", "-std=c++20"],
            build_directory=build,
            verbose=False,
        )
    finally:
        cpp._get_cuda_arch_flags = orig
    _ext.append(ext)
    return ext


def gdn_conv_cuda_prep(
    x: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
):
    """gdn_fused_conv_prep on the CUDA kernel; None if the kernel's layout
    contract (bf16, head dims 128, width 4, 16-byte aligned rows) is not met.
    Call ``load()`` first (the warmup does).
    """
    H, K, V = num_k_heads, head_k_dim, head_v_dim
    HV = A_log.shape[0]
    if K != 128 or V != 128 or conv_weights.shape[1] != 4:
        return None
    P = x.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1
    q = torch.empty(P, H, K, dtype=x.dtype, device=x.device)
    k = torch.empty(P, H, K, dtype=x.dtype, device=x.device)
    v = torch.empty(P, HV, V, dtype=x.dtype, device=x.device)
    g = torch.empty(P, HV, dtype=torch.float32, device=x.device)
    beta = torch.empty(P, HV, dtype=torch.float32, device=x.device)
    # Tokens per half-warp: 4 (32-token CTAs) for short sequences, else 8.
    tph = 4 if 1024 * max(num_seqs, 1) > P else 8
    ok = _ext[0].run(
        x,
        conv_weights,
        conv_state,
        cache_indices.contiguous(),
        has_initial_state.contiguous(),
        cu_seqlens,
        num_seqs,
        a,
        b,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        H,
        tph,
        0,
    )
    if not ok:
        return None
    return q, k, v, g, beta


# CentML/vllm#86 (leopck), gdn_conv_cuda.cu; only the A_log pointer field of
# Params was renamed (to a_log, for the typos hook).
_SOURCE = r"""// Fused causal conv1d(+state) -> SiLU -> q/k l2norm -> gating for GDN prefill chunks.
// Bit-exact re-implementation of qwen_gdn_linear_attn.py `_gdn_fused_conv_post_conv_kernel_v2` (Triton 3.8,
// BT=32/ST=32/4 warps) for sm_107a. Every floating-point instruction mirrors the Triton PTX of that kernel:
//   taps:   mul.rn.bf16x2 (per-tap bf16 rounding) + fp32 accumulate (sm_107f add.rn.f32x2.bf16x2.f32x2,
//           verified bit-exact vs cvt+add.rn.f32), acc = (((0 + p0) + p1) + p2) + p3, p_j = x[t-3+j] * w[:, j]
//   SiLU:   div.full.f32(acc, ex2.approx.f32(acc * -log2e) + 1)  -> cvt.rn.bf16
//   l2norm: per-thread tree over 8 channels exactly as Triton's SLP-vectorized reduce (fma.rn.f32x2 contraction),
//           16-lane xor butterfly 8,4,2,1, +1e-6, sqrt.approx.ftz.f32, div.full.f32(1, .), mul.f32
//   gating: libdevice __nv_logf / __nv_expf (ftz variants, as Triton links them) transcribed from the PTX.
// Memory side: 16-B vector loads of the strided in_proj rows (one 256-B row segment per half-warp), a register
// rolling window (each input row loaded once per half-warp, +3 halo rows), 16-B stores.
#ifndef GK2_KERNEL_ONLY
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <stdint.h>

#ifndef GK2_MIXED_ADD
#define GK2_MIXED_ADD 1
#endif

namespace gk2 {

__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ float bf_lo(uint32_t p) { return __uint_as_float(p << 16); }
__device__ __forceinline__ float bf_hi(uint32_t p) { return __uint_as_float(p & 0xffff0000u); }
__device__ __forceinline__ float fadd(float a, float b) { float r; asm("add.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fmul(float a, float b) { float r; asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float ffma(float a, float b, float c) { float r; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float fdivf(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fex2(float a) { float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }

// acc(lo,hi) += bf16x2 p, fp32, round-to-nearest.
__device__ __forceinline__ void acc_add(uint64_t& acc, uint32_t p) {
#if GK2_MIXED_ADD
  asm("add.rn.f32x2.bf16x2.f32x2 %0, %1, %0;" : "+l"(acc) : "r"(p));
#else
  float lo = __uint_as_float((uint32_t)acc), hi = __uint_as_float((uint32_t)(acc >> 32));
  lo = fadd(lo, bf_lo(p));
  hi = fadd(hi, bf_hi(p));
  acc = (uint64_t)__float_as_uint(lo) | ((uint64_t)__float_as_uint(hi) << 32);
#endif
}

__device__ __forceinline__ float silu_bf16(float a) {
  // Triton: acc / (1 + tl.exp(-acc)) -> .to(bf16).to(fp32)
  float e = fex2(fmul(a, __uint_as_float(0xBFB8AA3Bu)));
  float y = fdivf(a, fadd(e, 1.0f));
  unsigned short h;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(y));
  return __uint_as_float(((uint32_t)h) << 16);
}

// SiLU of a pair, rounded to bf16x2 (packed). FAST: warp-uniformly guaranteed 2^-100 < |acc| < 87 (or 0), where
// ex2.approx.f32 == MUFU.EX2 (no subnormal fix-up: t >= -126) and div.full.f32(a, d) == a * MUFU.RCP(d)
// (d = 1 + e in [1, 2^126): no range scaling) -> identical bits with 5 instead of ~13 instructions.
template <bool FAST>
__device__ __forceinline__ uint32_t silu2_bf16(float a0, float a1) {
  float y0, y1;
  if (FAST) {
    float e0, e1, r0, r1;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e0) : "f"(fmul(a0, __uint_as_float(0xBFB8AA3Bu))));
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e1) : "f"(fmul(a1, __uint_as_float(0xBFB8AA3Bu))));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r0) : "f"(fadd(e0, 1.0f)));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r1) : "f"(fadd(e1, 1.0f)));
    y0 = fmul(a0, r0);
    y1 = fmul(a1, r1);
  } else {
    y0 = fdivf(a0, fadd(fex2(fmul(a0, __uint_as_float(0xBFB8AA3Bu))), 1.0f));
    y1 = fdivf(a1, fadd(fex2(fmul(a1, __uint_as_float(0xBFB8AA3Bu))), 1.0f));
  }
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(y1), "f"(y0));
  return r;
}
__device__ __forceinline__ bool silu_fast_ok(float a) {
  const float m = fabsf(a);
  return (m < 87.0f) && (m > 7.8886e-31f || m == 0.0f);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// libdevice __nv_logf (ftz), transcribed from the Triton PTX.
__device__ __forceinline__ float tri_logf(float x) {
  float r;
  asm("{\n\t"
      ".reg .pred p1, p2, p3;\n\t"
      ".reg .b32 xs, eadj, t0, i, e, m, fe, f, t, c;\n\t"
      "setp.lt.f32 p1, %1, 0f00800000;\n\t"
      "mul.f32 t0, %1, 0f4B000000;\n\t"
      "selp.f32 xs, t0, %1, p1;\n\t"
      "selp.f32 eadj, 0fC1B80000, 0f00000000, p1;\n\t"
      "add.s32 i, xs, -1059760811;\n\t"
      "and.b32 e, i, -8388608;\n\t"
      "sub.s32 m, xs, e;\n\t"
      "cvt.rn.f32.s32 fe, e;\n\t"
      "mov.b32 c, 0f34000000;\n\t"
      "fma.rn.ftz.f32 fe, fe, c, eadj;\n\t"
      "add.f32 f, m, 0fBF800000;\n\t"
      "mov.b32 t, 0fBE055027;\n\t"
      "mov.b32 c, 0f3E1039F6;\n\t"
      "fma.rn.ftz.f32 t, t, f, c;\n\t"
      "mov.b32 c, 0fBDF8CDCC;\n\t"
      "fma.rn.ftz.f32 t, t, f, c;\n\t"
      "mov.b32 c, 0f3E0F2955;\n\t"
      "fma.rn.ftz.f32 t, t, f, c;\n\t"
      "mov.b32 c, 0fBE2AD8B9;\n\t"
      "fma.rn.ftz.f32 t, t, f, c;\n\t"
      "mov.b32 c, 0f3E4CED0B;\n\t"
      "fma.rn.ftz.f32 t, t, f, c;\n\t"
      "mov.b32 c, 0fBE7FFF22;\n\t"
      "fma.rn.ftz.f32 t, t, f, c;\n\t"
      "mov.b32 c, 0f3EAAAA78;\n\t"
      "fma.rn.ftz.f32 t, t, f, c;\n\t"
      "mov.b32 c, 0fBF000000;\n\t"
      "fma.rn.ftz.f32 t, t, f, c;\n\t"
      "mul.f32 t, f, t;\n\t"
      "fma.rn.ftz.f32 t, t, f, f;\n\t"
      "mov.b32 c, 0f3F317218;\n\t"
      "fma.rn.ftz.f32 %0, fe, c, t;\n\t"
      "setp.lt.u32 p2, xs, 2139095040;\n\t"
      "mov.b32 c, 0f7F800000;\n\t"
      "@!p2 fma.rn.ftz.f32 %0, xs, c, c;\n\t"
      "setp.eq.f32 p3, xs, 0f00000000;\n\t"
      "selp.f32 %0, 0fFF800000, %0, p3;\n\t"
      "}"
      : "=f"(r) : "f"(x));
  return r;
}

// libdevice __nv_expf (ftz), transcribed from the Triton PTX.
__device__ __forceinline__ float tri_expf(float t) {
  float r;
  asm("{\n\t"
      ".reg .b32 a, b, c, d, s, e, k1, k2;\n\t"
      "mov.b32 k1, 0f3BBB989D;\n\t"
      "mov.b32 k2, 0f3F000000;\n\t"
      "fma.rn.ftz.f32 a, %1, k1, k2;\n\t"
      "cvt.ftz.sat.f32.f32 a, a;\n\t"
      "mov.b32 k1, 0f437C0000;\n\t"
      "mov.b32 k2, 0f4B400001;\n\t"
      "fma.rm.ftz.f32 b, a, k1, k2;\n\t"
      "add.f32 c, b, 0fCB40007F;\n\t"
      "neg.f32 c, c;\n\t"
      "mov.b32 k1, 0f3FB8AA3B;\n\t"
      "fma.rn.ftz.f32 d, %1, k1, c;\n\t"
      "mov.b32 k1, 0f32A57060;\n\t"
      "fma.rn.ftz.f32 d, %1, k1, d;\n\t"
      "shl.b32 s, b, 23;\n\t"
      "ex2.approx.ftz.f32 e, d;\n\t"
      "mul.f32 %0, e, s;\n\t"
      "}"
      : "=f"(r) : "f"(t));
  return r;
}

struct Params {
  const uint16_t* x; int64_t sx;            // [P, conv_dim] bf16, channel stride 1
  const uint16_t* w; int64_t sw_dim, sw_w;  // [conv_dim, 4] bf16
  uint16_t* cs; int64_t cs_seq, cs_dim, cs_tok;
  const void* cidx; int cidx_64;            // [num_seqs] int32/int64
  const uint8_t* hinit;                     // [num_seqs] bool
  const int32_t* cu; int num_seqs;          // [num_seqs+1]
  const uint16_t* a; const uint16_t* b; int64_t sa, sb;  // bf16 [P, HV] (row strides)
  const void* a_log; int a_log_bf16; const void* dtb; int dtb_bf16;
  uint16_t* q; uint16_t* k; uint16_t* v; float* g; float* beta;
  int H, HV;
};

__device__ __forceinline__ float ld_param_f(const void* p, int is_bf16, int i) {
  return is_bf16 ? __uint_as_float(((uint32_t)((const uint16_t*)p)[i]) << 16) : ((const float*)p)[i];
}

// RV: in-thread l2 reduction tree variant (0: fma on the first pair, 1: fma on the second pair).
template <int TPH, int RV>
__global__ void __launch_bounds__(128) conv_post_kernel(const Params p) {
  constexpr int BT = TPH * 8;
  const int tid = threadIdx.x, lane = tid & 31, l16 = tid & 15, hw = tid >> 4;
  const int grp = blockIdx.y;  // 0..H-1 q, H..2H-1 k, 2H..2H+HV-1 v
  const int pid = blockIdx.x;
  // ---- program -> (sequence, chunk): warp-parallel prefix sum over ceil(len/BT)
  int seq = -1, base = 0, s0 = 0, s1 = 0;
  for (int i0 = 0; i0 < p.num_seqs && seq < 0; i0 += 32) {
    int i = i0 + lane;
    int a0 = 0, a1 = 0, nb = 0;
    if (i < p.num_seqs) { a0 = p.cu[i]; a1 = p.cu[i + 1]; nb = (a1 - a0 + BT - 1) / BT; }
    int inc = nb;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(0xffffffffu, inc, o); if (lane >= o) inc += y; }
    unsigned hit = __ballot_sync(0xffffffffu, (i < p.num_seqs) && (pid < base + inc));
    if (hit) {
      int src = __ffs(hit) - 1;
      seq = i0 + src;
      int excl = __shfl_sync(0xffffffffu, inc - nb, src);
      s0 = __shfl_sync(0xffffffffu, a0, src);
      s1 = __shfl_sync(0xffffffffu, a1, src);
      base += excl;
    } else {
      base += __shfl_sync(0xffffffffu, inc, 31);
    }
  }
  if (seq < 0) return;
  const int chunk = pid - base;
  const int cstart = s0 + chunk * BT;
  const int64_t slot = p.cidx_64 ? ((const int64_t*)p.cidx)[seq] : (int64_t)((const int32_t*)p.cidx)[seq];
  const bool use_state = p.hinit[seq] != 0;
  const int ch = grp * 128 + l16 * 8;  // first of this thread's 8 channels (q|k|v channel space)
  const bool cs_vec = (p.cs_dim == 1);

  // ---- weights: wt[j].word[m] = (w[ch+2m, j], w[ch+2m+1, j])
  uint32_t wt[4][4];
  if (p.sw_w == 1 && p.sw_dim == 4) {
    // contiguous [conv_dim, 4]: this thread's 8 channels x 4 taps = 64 B = 4 x 16-B loads, then byte-permute
    const uint4* wp = reinterpret_cast<const uint4*>(p.w + (int64_t)ch * 4);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const uint4 u = __ldg(wp + m);  // channels ch+2m (u.x: taps 0,1; u.y: taps 2,3) and ch+2m+1 (u.z, u.w)
      wt[0][m] = __byte_perm(u.x, u.z, 0x5410);
      wt[1][m] = __byte_perm(u.x, u.z, 0x7632);
      wt[2][m] = __byte_perm(u.y, u.w, 0x5410);
      wt[3][m] = __byte_perm(u.y, u.w, 0x7632);
    }
  } else
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      uint32_t lo = p.w[(int64_t)(ch + 2 * m) * p.sw_dim + j * p.sw_w];
      uint32_t hi = p.w[(int64_t)(ch + 2 * m + 1) * p.sw_dim + j * p.sw_w];
      wt[j][m] = lo | (hi << 16);
    }

  const int t0 = cstart + hw * TPH;
  // ---- rolling window: rows t0-3 .. t0+TPH-1
  uint4 xr[TPH + 3];
  if (t0 - 3 >= s0 && t0 + TPH <= s1) {
    // interior strip (uniform per half-warp): unconditional 16-B loads, incremental addressing
    const uint16_t* xp = p.x + (int64_t)(t0 - 3) * p.sx + ch;
#pragma unroll
    for (int r = 0; r < TPH + 3; ++r) xr[r] = __ldg(reinterpret_cast<const uint4*>(xp + (int64_t)r * p.sx));
  } else {
#pragma unroll
  for (int r = 0; r < TPH + 3; ++r) {
    const int tau = t0 - 3 + r;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (tau >= s0 && tau < s1) {
      val = __ldg(reinterpret_cast<const uint4*>(p.x + (int64_t)tau * p.sx + ch));
    } else if (tau < s0 && use_state && tau >= s0 - 3 && t0 < s1) {
      const int row = 3 - (s0 - tau);
      const uint16_t* sp = p.cs + slot * p.cs_seq + (int64_t)row * p.cs_tok;
      if (cs_vec) {
        val = *reinterpret_cast<const uint4*>(sp + ch);
      } else {
        uint32_t wv[4];
#pragma unroll
        for (int m = 0; m < 4; ++m)
          wv[m] = (uint32_t)sp[(int64_t)(ch + 2 * m) * p.cs_dim] | ((uint32_t)sp[(int64_t)(ch + 2 * m + 1) * p.cs_dim] << 16);
        val = make_uint4(wv[0], wv[1], wv[2], wv[3]);
      }
    }
    xr[r] = val;
  }
  }

  // ---- final conv state (chunk 0 of each sequence): rows 0..2 = last 3 inputs (old state if seq < 3 long);
  // read early (overlaps the main loop), written after a CTA barrier at the end
  uint4 csval = make_uint4(0, 0, 0, 0);
  const int rr = tid >> 4;
  if (chunk == 0) {
    uint4& val = csval;
    if (rr < 3) {
      const int tau = s1 - 3 + rr;
      if (tau >= s0) {
        val = __ldg(reinterpret_cast<const uint4*>(p.x + (int64_t)tau * p.sx + ch));
      } else if (use_state) {
        const int row = 3 - (s0 - tau);
        const uint16_t* sp = p.cs + slot * p.cs_seq + (int64_t)row * p.cs_tok;
        if (cs_vec) {
          val = *reinterpret_cast<const uint4*>(sp + ch);
        } else {
          uint32_t wv[4];
#pragma unroll
          for (int m = 0; m < 4; ++m)
            wv[m] = (uint32_t)sp[(int64_t)(ch + 2 * m) * p.cs_dim] | ((uint32_t)sp[(int64_t)(ch + 2 * m + 1) * p.cs_dim] << 16);
          val = make_uint4(wv[0], wv[1], wv[2], wv[3]);
        }
      }
      // Triton stores (xv + sv) in bf16: -0 + +0 = +0 (canonicalize negative zeros)
      uint32_t* vw = reinterpret_cast<uint32_t*>(&val);
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        uint32_t u = vw[m];
        if ((u & 0xffffu) == 0x8000u) u &= 0xffff0000u;
        if ((u >> 16) == 0x8000u) u &= 0x0000ffffu;
        vw[m] = u;
      }
    }
  }

  const bool is_v = grp >= 2 * p.H;
  uint16_t* outp;
  int64_t orow;
  if (!is_v) {
    const bool is_k = grp >= p.H;
    outp = (is_k ? p.k : p.q) + (int64_t)(grp - (is_k ? p.H : 0)) * 128 + l16 * 8;
    orow = (int64_t)p.H * 128;
  } else {
    outp = p.v + (int64_t)(grp - 2 * p.H) * 128 + l16 * 8;
    orow = (int64_t)p.HV * 128;
  }

#pragma unroll
  for (int i = 0; i < TPH; ++i) {
    const int t = t0 + i;
    // no early exit: rows past the sequence end compute on zeros and are not stored, so every lane of the
    // warp reaches the full-mask shuffles below (a half-warp mask + divergent exit compiles to collective
    // emulation: WARPSYNC/MATCH/ENDCOLLECTIVE).
    uint64_t acc[4];
    bool ok = true;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      acc[m] = 0;  // (+0, +0)
      const uint32_t* r0 = reinterpret_cast<const uint32_t*>(&xr[i]);
      const uint32_t* r1 = reinterpret_cast<const uint32_t*>(&xr[i + 1]);
      const uint32_t* r2 = reinterpret_cast<const uint32_t*>(&xr[i + 2]);
      const uint32_t* r3 = reinterpret_cast<const uint32_t*>(&xr[i + 3]);
      acc_add(acc[m], bmul2(r0[m], wt[0][m]));
      acc_add(acc[m], bmul2(r1[m], wt[1][m]));
      acc_add(acc[m], bmul2(r2[m], wt[2][m]));
      acc_add(acc[m], bmul2(r3[m], wt[3][m]));
      ok = ok && silu_fast_ok(__uint_as_float((uint32_t)acc[m])) && silu_fast_ok(__uint_as_float((uint32_t)(acc[m] >> 32)));
    }
    uint32_t yb[4];
    if (__all_sync(0xffffffffu, ok)) {
#pragma unroll
      for (int m = 0; m < 4; ++m) yb[m] = silu2_bf16<true>(__uint_as_float((uint32_t)acc[m]), __uint_as_float((uint32_t)(acc[m] >> 32)));
    } else {
#pragma unroll
      for (int m = 0; m < 4; ++m) yb[m] = silu2_bf16<false>(__uint_as_float((uint32_t)acc[m]), __uint_as_float((uint32_t)(acc[m] >> 32)));
    }
    float y[8];
#pragma unroll
    for (int m = 0; m < 4; ++m) { y[2 * m] = bf_lo(yb[m]); y[2 * m + 1] = bf_hi(yb[m]); }
    uint4 o;
    if (!is_v) {
      float s01x, s01y, s45x, s45y;
      if (RV == 0) {
        s01x = ffma(y[0], y[0], fmul(y[2], y[2])); s01y = ffma(y[1], y[1], fmul(y[3], y[3]));
        s45x = ffma(y[4], y[4], fmul(y[6], y[6])); s45y = ffma(y[5], y[5], fmul(y[7], y[7]));
      } else {
        s01x = ffma(y[2], y[2], fmul(y[0], y[0])); s01y = ffma(y[3], y[3], fmul(y[1], y[1]));
        s45x = ffma(y[6], y[6], fmul(y[4], y[4])); s45y = ffma(y[7], y[7], fmul(y[5], y[5]));
      }
      float tot = fadd(fadd(s01x, s45x), fadd(s01y, s45y));
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) tot = fadd(tot, __shfl_xor_sync(0xffffffffu, tot, off));
      float sq;
      asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(sq) : "f"(fadd(tot, __uint_as_float(0x358637BDu))));
      const float inv = fdivf(1.0f, sq);
      o.x = pack_bf16x2(fmul(inv, y[0]), fmul(inv, y[1]));
      o.y = pack_bf16x2(fmul(inv, y[2]), fmul(inv, y[3]));
      o.z = pack_bf16x2(fmul(inv, y[4]), fmul(inv, y[5]));
      o.w = pack_bf16x2(fmul(inv, y[6]), fmul(inv, y[7]));
    } else {
      // y are exact bf16 values; packing is exact
      o = make_uint4(yb[0], yb[1], yb[2], yb[3]);
    }
    if (t < s1) *reinterpret_cast<uint4*>(outp + (int64_t)t * orow) = o;
  }

  // ---- gating (v groups): g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
  if (is_v) {
    const int hv = grp - 2 * p.H;
    const float A_log = ld_param_f(p.a_log, p.a_log_bf16, hv);
    const float dtb = ld_param_f(p.dtb, p.dtb_bf16, hv);
    for (int j = tid; j < BT; j += 128) {
      const int t = cstart + j;
      if (t >= s1) break;
      const float av = __uint_as_float(((uint32_t)p.a[(int64_t)t * p.sa + hv]) << 16);
      const float bv = __uint_as_float(((uint32_t)p.b[(int64_t)t * p.sb + hv]) << 16);
      const float xx = fadd(av, dtb);
      const float lp = tri_logf(fadd(fex2(fmul(xx, __uint_as_float(0xBFB8AA3Bu))), 1.0f));
      const float sp_pos = fadd(xx, lp);
      const float sp_neg = tri_logf(fadd(fex2(fmul(xx, __uint_as_float(0x3FB8AA3Bu))), 1.0f));
      float sp = (xx > 0.0f) ? sp_pos : sp_neg;
      sp = (xx <= 20.0f) ? sp : xx;
      const float eA = fex2(fmul(A_log, __uint_as_float(0x3FB8AA3Bu)));
      const float gv = tri_expf(fmul(sp, -eA));
      const float be = fdivf(1.0f, fadd(fex2(fmul(bv, __uint_as_float(0xBFB8AA3Bu))), 1.0f));
      p.g[(int64_t)t * p.HV + hv] = gv;
      p.beta[(int64_t)t * p.HV + hv] = be;
    }
  }

  if (chunk == 0) {
    const uint4 val = csval;
    __syncthreads();  // all reads of the old state (window preload + csval) precede the writes
    if (rr < 3) {
      uint16_t* dp = p.cs + slot * p.cs_seq + (int64_t)rr * p.cs_tok;
      if (cs_vec) {
        *reinterpret_cast<uint4*>(dp + ch) = val;
      } else {
        const uint32_t* vw = reinterpret_cast<const uint32_t*>(&val);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          dp[(int64_t)(ch + 2 * m) * p.cs_dim] = (uint16_t)(vw[m] & 0xffffu);
          dp[(int64_t)(ch + 2 * m + 1) * p.cs_dim] = (uint16_t)(vw[m] >> 16);
        }
      }
    }
  }
}

#ifndef GK2_KERNEL_ONLY
// returns false if the shape/stride contract is not met (caller falls back to Triton)
bool run(torch::Tensor x, torch::Tensor w, torch::Tensor cs, torch::Tensor cidx, torch::Tensor hinit,
         torch::Tensor cu, int64_t num_seqs, torch::Tensor a, torch::Tensor b, torch::Tensor A_log,
         torch::Tensor dt_bias, torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
         torch::Tensor beta, int64_t H, int64_t tph, int64_t rv) {
  const int64_t P = x.size(0);
  const int HV = (int)A_log.size(0);
  if (x.scalar_type() != at::kBFloat16 || w.scalar_type() != at::kBFloat16 || cs.scalar_type() != at::kBFloat16) return false;
  if (a.scalar_type() != at::kBFloat16 || b.scalar_type() != at::kBFloat16) return false;
  if (a.stride(1) != 1 || b.stride(1) != 1 || x.stride(1) != 1) return false;
  if ((x.stride(0) % 8) != 0 || (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) != 0) return false;
  if (w.size(1) != 4 || x.size(1) != (2 * H + HV) * 128) return false;
  if (w.stride(1) == 1 && w.stride(0) == 4 && (reinterpret_cast<uintptr_t>(w.data_ptr()) % 16) != 0) return false;
  if (cu.scalar_type() != at::kInt || hinit.scalar_type() != at::kBool) return false;
  if (cidx.scalar_type() != at::kInt && cidx.scalar_type() != at::kLong) return false;
  if (cs.stride(1) == 1 && ((cs.stride(0) % 8) != 0 || (cs.stride(2) % 8) != 0 ||
                            (reinterpret_cast<uintptr_t>(cs.data_ptr()) % 16) != 0)) return false;
  if (P == 0 || num_seqs == 0) return true;
  Params p;
  p.x = (const uint16_t*)x.data_ptr(); p.sx = x.stride(0);
  p.w = (const uint16_t*)w.data_ptr(); p.sw_dim = w.stride(0); p.sw_w = w.stride(1);
  p.cs = (uint16_t*)cs.data_ptr(); p.cs_seq = cs.stride(0); p.cs_dim = cs.stride(1); p.cs_tok = cs.stride(2);
  p.cidx = cidx.data_ptr(); p.cidx_64 = cidx.scalar_type() == at::kLong;
  p.hinit = (const uint8_t*)hinit.data_ptr();
  p.cu = (const int32_t*)cu.data_ptr(); p.num_seqs = (int)num_seqs;
  p.a = (const uint16_t*)a.data_ptr(); p.b = (const uint16_t*)b.data_ptr(); p.sa = a.stride(0); p.sb = b.stride(0);
  p.a_log = A_log.data_ptr(); p.a_log_bf16 = A_log.scalar_type() == at::kBFloat16;
  p.dtb = dt_bias.data_ptr(); p.dtb_bf16 = dt_bias.scalar_type() == at::kBFloat16;
  if ((A_log.scalar_type() != at::kFloat && !p.a_log_bf16) || (dt_bias.scalar_type() != at::kFloat && !p.dtb_bf16)) return false;
  p.q = (uint16_t*)q.data_ptr(); p.k = (uint16_t*)k.data_ptr(); p.v = (uint16_t*)v.data_ptr();
  p.g = (float*)g.data_ptr(); p.beta = (float*)beta.data_ptr();
  p.H = (int)H; p.HV = HV;
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  const int BT = (int)tph * 8;
  dim3 grid((unsigned)((P + BT - 1) / BT + num_seqs), (unsigned)(2 * H + HV));
#define GK2_LAUNCH(T, R) conv_post_kernel<T, R><<<grid, 128, 0, stream>>>(p)
  if (tph == 4) { if (rv) GK2_LAUNCH(4, 1); else GK2_LAUNCH(4, 0); }
  else if (tph == 8) { if (rv) GK2_LAUNCH(8, 1); else GK2_LAUNCH(8, 0); }
  else if (tph == 16) { if (rv) GK2_LAUNCH(16, 1); else GK2_LAUNCH(16, 0); }
  else return false;
#undef GK2_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

#endif  // GK2_KERNEL_ONLY
}  // namespace gk2

#ifndef GK2_KERNEL_ONLY
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &gk2::run); }
#else
template __global__ void gk2::conv_post_kernel<8, 0>(const gk2::Params);
#endif
"""
