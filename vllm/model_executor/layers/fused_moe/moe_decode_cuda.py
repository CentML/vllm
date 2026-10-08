# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Decode MXFP8 routed MoE in one launch (routing + FC1 + SwiGLU + requant + FC2), SM100+ tcgen05.

Replaces the trtllm-gen chain (routing kernel -> FC1 bmm with fused SwiGLU and
MXFP8 requant -> FC2 bmm) for small token counts (decode with MTP: T <= 32)
with one persistent cluster kernel. It reads the weights and scales exactly as
prepared for trtllm-gen (``_shuffle_mxfp8_moe_weights``: gate/up interleave and
32-row shuffle of K-major rows, 128x4-interleaved E8M0 scales) and produces the
deferred (``do_finalize=False``) output contract: unweighted BF16 GEMM2 rows per
(token, k), BF16 top-k weights and the permute map ``idx[t, k] = t * top_k + k``,
which the fused finalize consumer (``moe_finalize``) reduces in k order.

Grid = persistent clusters of C (4 or 8) CTAs, one CTA per SM. A cluster owns
whole experts (distinct experts ascending = slots; cluster c takes slots c,
c + #clusters, ...); CTA rank r of the cluster owns the intermediate slice
[r * 512 / C, (r + 1) * 512 / C) and the output rows [r * 2048 / C, ...):

1. Before ``griddepcontrol.wait``: barriers, TMEM allocation, tensor-map prefetch.
2. Every CTA: routing (top-8 of the bf16/fp32 logits, ties to the lower expert,
   fp32 softmax over the 8 = trtllm-gen Renormalize), the slot list, and all T
   activation rows as the MMA B operand (128B-swizzled K-major, N = 8/16/32
   columns >= T) plus their E8M0 scales (128x4 chunks). Then
   ``griddepcontrol.launch_dependents``.
3. Warp 4 streams the CTA's weight tiles (128 shuffled rows x 128 B, 2-D TMA
   with 128B swizzle, + the matching 512-B scale chunk) through a ring of SMEM
   stages: per expert the 8/C FC1 tiles (gate and up rows of 64 intermediate
   columns each, full K = 2048) and the 16/C FC2 tiles (full K = 512). For
   N <= 16 the order is skewed, FC1(j + 1) before FC2(j), which hides the FC1 ->
   FC2 exchange of clusters with several experts.
4. Warp 6 (SF helper) waits for the ring stages, copies their scales to TMEM
   (``tcgen05.cp``), prepares each FC2's B-operand scales and signals the issuer
   once per 2 stages. Warp 5 (one thread) only issues
   ``tcgen05.mma.kind::mxf8f6f4.block_scale`` (swap-AB: M = 128 weight rows,
   N = tokens, K = 32 per instruction, accumulators in TMEM) in K order (= the
   chain's accumulation order) and the ring commits.
5. Warps 0-3 (FC1 epilogue): TMEM -> SwiGLU (gate/up rows pair up across lanes
   l and l ^ 8 of the shuffle) -> MXFP8 requant (trtllm-gen's rule: E8M0 =
   exponent(amax) - 8, saturating E4M3) -> ``st.async`` of the quantized slice
   and its scales into the FC2 B operand of every CTA of the cluster (DSMEM,
   complete_tx on the receiver's barrier: no fences, no global round trip).
   FC2 epilogue: TMEM -> BF16 rows of the expert's routes (N <= 16: staged in
   SMEM, one bulk copy per route).

No global scratch, counters or flags.

Env: ``VLLM_MOE_DECODE_MAX_TOKENS`` (default 0 = off) / ``VLLM_MOE_DECODE_MIN_TOKENS``
(default 1): use this kernel for MXFP8 trtllm MoE calls with MIN <= T <= MAX (MAX <= 32)
that would defer finalize. VR microbench (40-layer graphs, cluster 8): it beats the chain
at T = 2..6 (T = 6 correlated routing by 1.3-1.9 us/layer) and loses at T = 1 and T >= 12.
"""

import hashlib
import os

import torch

_ext: list = []
_instances: dict = {}  # device index -> MoeDecode | None (build failed)

MAX_TOKENS = int(os.environ.get("VLLM_MOE_DECODE_MAX_TOKENS", "0"))
MIN_TOKENS = max(1, int(os.environ.get("VLLM_MOE_DECODE_MIN_TOKENS", "1")))
# SwiGLU-output requant: 1 (default) = trtllm-gen's E8M0 = exponent(amax) - 8 with a saturating
# E4M3 cast (VR probe: 1e-6 relL2 to the chain's GEMM2 rows); 0 = OCP ceil(log2(amax / 448)).
QMODE = int(os.environ.get("VLLM_MOE_DECODE_QMODE", "1"))
# CTAs per expert (cluster size): 4 or 8.
CLUSTER = int(os.environ.get("VLLM_MOE_DECODE_CLUSTER", "8"))
# Weight ring stages of 33 KB (0 = as many as fit).
STAGES = int(os.environ.get("VLLM_MOE_DECODE_STAGES", "0"))
# Persistent clusters (0 = as many as can be co-resident).
CLUSTERS = int(os.environ.get("VLLM_MOE_DECODE_CLUSTERS", "0"))
# Build macros "K=V+K=V" (MDC_FAST_EXP, MDC_DEBUG, MDC_EARLY_PDL).
TUNE = os.environ.get("VLLM_MOE_DECODE_TUNE", "")

_SOURCE = r"""
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef MDC_DEBUG
#define MDC_DEBUG 0
#endif
#ifndef MDC_FAST_EXP
#define MDC_FAST_EXP 0
#endif
#ifndef MDC_EARLY_PDL
#define MDC_EARLY_PDL 1   // griddepcontrol.launch_dependents right after routing (else at the end)
#endif

namespace mdc {
constexpr int E = 256, TOPK = 8, H = 2048, I = 512;
constexpr int KB1 = H / 32, KB2 = I / 32;    // E8M0 blocks per row: FC1 64, FC2 16
constexpr int KC1 = H / 128, KC2 = I / 128;  // 128-B (SW128 atom) K chunks: FC1 16, FC2 4
constexpr int MAXT = 32, MAXR = MAXT * TOPK;
constexpr int MAXJ = 64;                     // experts per cluster (needs >= 4 clusters)
constexpr int NW = 8, NTHR = NW * 32;        // warps 0-3 epilogue, 4 TMA producer, 5 MMA issuer, 6 SF helper
constexpr int WPROD = 4, WMMA = 5, WHLP = 6;
constexpr int SFCH = 512;                    // 128x4 E8M0 chunk (128 rows x 4 K blocks)
constexpr int KSTG = 2;                      // 128-B K chunks per ring stage: one mbarrier wait per 4*KSTG MMAs
constexpr int ACH = 128 * 128;               // one TMA box: 128 weight rows x one 128-B K chunk
constexpr int ASTG = KSTG * ACH;             // weight bytes per stage
constexpr int SFSTG = KSTG * SFCH;           // weight scale bytes per stage
constexpr int MAXST = 16;
constexpr int KGRP = 2;                      // ring stages per readiness signal to the MMA issuer
constexpr int NGRP = 16;                     // readiness barriers (ring)
constexpr int HDR = 4096;
constexpr int TMEM_COLS = 512;
constexpr int NPROF = 32;

// Skewed order FC1(j + 1) before FC2(j) (hides the FC1 -> FC2 exchange of multi-expert clusters): 2 FC1
// accumulators, 3 FC2 B operands (a push of expert j + 1 into a peer follows the peer's FC2(j - 2) completion).
__host__ __device__ constexpr bool skew(int N) { return N <= 16; }
__host__ __device__ constexpr int na1(int N) { return skew(N) ? 2 : 1; }
__host__ __device__ constexpr int nb2(int N) { return skew(N) ? 3 : 2; }
__host__ __device__ constexpr bool ostage(int N) { return N <= 16; }   // FC2 output staged + bulk-stored
// SMEM layout (offsets from the 1 KB aligned base)
__host__ __device__ constexpr int off_b1() { return HDR; }                                // [KC1][N x 128 B] SW128
__host__ __device__ constexpr int off_sfb1(int N) { return HDR + N * H; }                 // [KC1][512]
__host__ __device__ constexpr int off_b2(int N) { return off_sfb1(N) + KC1 * SFCH; }      // [nb2][KC2][N x 128 B]
__host__ __device__ constexpr int off_sfb2(int N) { return off_b2(N) + nb2(N) * N * I; }  // [2][KC2][512]
__host__ __device__ constexpr int off_sfs(int N) { return off_sfb2(N) + 2 * KC2 * SFCH; } // [nb2][C][N] u32
__host__ __device__ constexpr int off_act(int C, int N) { return off_sfs(N) + nb2(N) * C * N * 4; }   // [8/C][N][64] f32
__host__ __device__ constexpr int off_out(int C, int N) { return off_act(C, N) + (8 / C) * N * 64 * 4; }  // [N][H/C] bf16
__host__ __device__ constexpr int off_sfa(int C, int N) { return off_out(C, N) + (ostage(N) ? N * (H / C) * 2 : 0); }  // [ns][SFSTG]
__host__ __device__ constexpr int off_a(int C, int N, int ns) { return (off_sfa(C, N) + ns * SFSTG + 1023) & ~1023; }
__host__ __device__ constexpr int smem_total(int C, int N, int ns) { return off_a(C, N, ns) + ns * ASTG + 1024; }
// TMEM columns: [na1][8/C][N] FC1 acc | [2][16/C][N] FC2 acc | SFB1 | [2] SFB2 | [ns] SFA
__host__ __device__ constexpr int tm_acc2(int C, int N) { return na1(N) * (8 / C) * N; }
__host__ __device__ constexpr int tm_sfb1(int C, int N) { return tm_acc2(C, N) + 2 * (16 / C) * N; }
__host__ __device__ constexpr int tm_sfb2(int C, int N) { return tm_sfb1(C, N) + KC1 * 4; }
__host__ __device__ constexpr int tm_sfa(int C, int N) { return tm_sfb2(C, N) + 2 * KC2 * 4; }

struct Params {
  CUtensorMap tm13, tm2;                     // u8 [E*2I, H] / [E*H, I], box 128 B x 128 rows, 128B swizzle
  const void* logits;
  const uint8_t* x; const uint8_t* xs;
  const uint8_t* s13; const uint8_t* s2;
  __nv_bfloat16* out; __nv_bfloat16* topk_w; int32_t* topk_ids;
  long long* prof;                           // optional [grid][NPROF] %globaltimer stamps (microbench)
  float* dbg;                                // optional [grid][(8/C + 16/C) * 128][32]: FC1 / FC2 accumulators of
                                             // the CTA's first expert (numerics debugging)
  int T; int ns; int qmode;
};

struct Hdr {
  unsigned long long full[MAXST], empty[MAXST], grp[NGRP];
  unsigned long long acc1_full[2], acc1_empty[2], acc2_full[2], acc2_empty[2], b2full[3];
  uint32_t tmem;
  uint32_t tokmask[E];                       // per expert: tokens that selected it
  int16_t route_e[MAXR];                     // expert of route t * TOPK + k
  int16_t cexp[MAXJ];                        // this cluster's experts in order (producer warp only)
};

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(c));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n }"
               : "=r"(ok) : "r"(b), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity, int tag) {
#if MDC_DEBUG
  long long n = 0;
  while (!mbar_try(b, parity)) {
    if (++n == (1ll << 24)) {
      printf("mdc hang: blk %d tid %d tag %d parity %u\n", blockIdx.x, threadIdx.x, tag, parity);
      __trap();
    }
  }
#else
  while (!mbar_try(b, parity)) {}
#endif
}
// Waits of warps other than the MMA issuer back off between polls: busy try_wait loops of 5 warps slow the issuing
// thread's own barrier/UMMA traffic (iteration 1: a resident-data stage took ~550 ns in the kernel vs 360 ns alone).
__device__ __forceinline__ void mbar_wait_sleep(uint32_t b, uint32_t parity, uint32_t ns, int tag) {
#if MDC_DEBUG
  long long n = 0;
#endif
  while (!mbar_try(b, parity)) {
    __nanosleep(ns);
#if MDC_DEBUG
    if (++n == (1ll << 22)) {
      printf("mdc hang: blk %d tid %d tag %d parity %u\n", blockIdx.x, threadIdx.x, tag, parity);
      __trap();
    }
#endif
  }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* dst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :: "l"(dst), "r"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r;
}
__device__ __forceinline__ void st_async_v4(uint32_t a, const uint32_t (&v)[4], uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
               :: "r"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(bar) : "memory");
}
__device__ __forceinline__ void st_async_b32(uint32_t a, uint32_t v, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
               :: "r"(a), "r"(v), "r"(bar) : "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
// UMMA shared-memory descriptors (SM100, version 1): K-major 128B swizzle (8-row x 128-B atoms, SBO 1024),
// and the scale-factor source of tcgen05.cp (no swizzle, 32 rows x 16 B).
__device__ __forceinline__ uint64_t sw128_desc(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61);
}
__device__ __forceinline__ uint64_t sf_desc(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | (1ull << 16) | (8ull << 32) | (1ull << 46);
}
__device__ __forceinline__ void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t sfa, uint32_t sfb,
                                     uint32_t acc) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
               " tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p;\n}"
               :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc), "r"(sfa), "r"(sfb) : "memory");
}
__device__ __forceinline__ void tc_cp(uint32_t taddr, uint64_t desc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(desc) : "memory");
}
__device__ __forceinline__ void tc_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void tmem_ld(uint32_t ta, float (&v)[N]);
template <> __device__ __forceinline__ void tmem_ld<8>(uint32_t ta, float (&v)[8]) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(ta) : "memory");
}
template <> __device__ __forceinline__ void tmem_ld<16>(uint32_t ta, float (&v)[16]) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(ta) : "memory");
}
template <> __device__ __forceinline__ void tmem_ld<32>(uint32_t ta, float (&v)[32]) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(ta) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }

// Shuffled row -> logical row of the trtllm-gen MXFP8 weights (_shuffle_mxfp8_moe_weights: 32-row block shuffle;
// for w13 after the gate/up interleave of swap_w13_to_w31, odd = gate). Checked against the traced permutation at init.
__device__ __forceinline__ int unshuffle32(int n) { return (n & ~31) | ((n & 7) << 2) | ((n & 31) >> 3); }

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
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void st_shared_zero16(uint32_t dst) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %1, %1, %1};" :: "r"(dst), "r"(0) : "memory");
}
__device__ __forceinline__ void st_shared_zero4(uint32_t dst) {
  asm volatile("st.shared.b32 [%0], %1;" :: "r"(dst), "r"(0) : "memory");
}
// Expert of slot s (distinct experts ascending), or -1 if s >= #distinct; whole warp, after the routing barrier.
__device__ __forceinline__ int slot_expert(const uint32_t* tokmask, int s, int lane) {
  int base = 0;
#pragma unroll 1
  for (int g = 0; g < E / 32; ++g) {
    const uint32_t sel = __ballot_sync(0xffffffffu, tokmask[g * 32 + lane] != 0);
    const int c = __popc(sel);
    if (s < base + c) return g * 32 + (int)__fns(sel, 0, s - base + 1);
    base += c;
  }
  return -1;
}
// Experts of cluster cid (slots cid, cid + ncl, ... below the number of distinct experts); whole warp.
__device__ __forceinline__ int cluster_experts(const uint32_t* tokmask, int cid, int ncl, int lane) {
  int d = 0;
#pragma unroll
  for (int g = 0; g < E / 32; ++g) d += __popc(__ballot_sync(0xffffffffu, tokmask[g * 32 + lane] != 0));
  return d > cid ? (d - 1 - cid) / ncl + 1 : 0;
}
// k of route (token n, expert e)
__device__ __forceinline__ int route_k(const int16_t* route_e, int n, int e) {
  int k = 0;
#pragma unroll
  for (int kk = 0; kk < TOPK; ++kk) if (route_e[n * TOPK + kk] == e) k = kk;
  return k;
}
// One ring stage on the issuing thread: KSTG K chunks (each 128 weight rows x 128 B in a 128B-swizzled box at
// a_st + q * ACH, its scales already in TMEM at sfa_tm + 4q, copied there by the SF helper warp) against the B
// operand chunks at b0 + q * N * 128 (scales in TMEM at sfb0 + 4q); `first` starts the accumulator. 4 MMAs
// (K = 32 each) per chunk, in K order.
template <int N>
__device__ __forceinline__ void mma_stage(uint32_t d, uint32_t a_st, uint32_t sfa_tm, uint32_t b0, uint32_t sfb0,
                                          bool first) {
  constexpr uint32_t IDESC = (1u << 23) | (8u << 24) | ((uint32_t)(N >> 3) << 17);   // E4M3 x E4M3, E8M0, M128, K-major
#pragma unroll
  for (int q = 0; q < KSTG; ++q) {
    const uint64_t ad = sw128_desc(a_st + q * ACH);
    const uint64_t bd = sw128_desc(b0 + q * (N * 128));
#pragma unroll
    for (int kp = 0; kp < 4; ++kp)
      umma(d, ad + 2 * kp, bd + 2 * kp, IDESC | ((uint32_t)kp << 29) | ((uint32_t)kp << 4),
           (sfa_tm + 4 * q) | ((uint32_t)kp << 30), (sfb0 + 4 * q) | ((uint32_t)kp << 30), !(first && q == 0 && kp == 0));
  }
}

template <int C, int N, bool kF32Logits>
__global__ void __launch_bounds__(NTHR, 1) moe_tc_kernel(const __grid_constant__ Params p) {
  constexpr int F1 = 8 / C, F2 = 16 / C;                 // FC1 / FC2 M tiles per CTA
  constexpr int NA1 = na1(N), NB2 = nb2(N), LAG = skew(N) ? 1 : 0;
  constexpr int S1 = F1 * (KC1 / KSTG), S2 = F2 * (KC2 / KSTG);   // ring stages per expert: FC1 / FC2
  static_assert(S1 % KGRP == 0 && S2 % KGRP == 0, "groups must not straddle GEMMs");
  constexpr uint32_t B2TX = N * (I + 4 * C);             // bytes every CTA receives per expert
  constexpr int TACC2 = tm_acc2(C, N), TSFB1 = tm_sfb1(C, N), TSFB2 = tm_sfb2(C, N), TSFA = tm_sfa(C, N);
  constexpr int OB1 = off_b1(), OSFB1 = off_sfb1(N), OB2 = off_b2(N), OSFB2 = off_sfb2(N), OSFS = off_sfs(N);
  constexpr int OACT = off_act(C, N), OSTO = off_out(C, N), OSFA = off_sfa(C, N);
  constexpr bool OSTAGE = ostage(N);
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = smem_raw + ((1024 - (su32(smem_raw) & 1023)) & 1023);
  Hdr& h = *reinterpret_cast<Hdr*>(smem);
  const uint32_t sb = su32(smem);
  const int T = p.T, ns = p.ns;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t rank, cid, ncl;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  asm("mov.u32 %0, %%clusterid.x;" : "=r"(cid));
  asm("mov.u32 %0, %%nclusterid.x;" : "=r"(ncl));
  long long* prof = p.prof ? p.prof + blockIdx.x * NPROF : nullptr;
  if (prof && tid == 0) prof[0] = gtime();
  const uint32_t a_base = sb + off_a(C, N, ns);
  auto bar = [&](const unsigned long long& b) { return su32(&b); };

  if (tid == 0) {
    for (int i = 0; i < ns; ++i) { mbar_init(bar(h.full[i]), 1); mbar_init(bar(h.empty[i]), 1); }
    for (int g = 0; g < NGRP; ++g) mbar_init(bar(h.grp[g]), 2);   // helper: tcgen05.commit + release arrive
    for (int b = 0; b < 2; ++b) {
      mbar_init(bar(h.acc1_full[b]), 1);
      mbar_init(bar(h.acc1_empty[b]), 4);
      mbar_init(bar(h.acc2_full[b]), 1);
      mbar_init(bar(h.acc2_empty[b]), 4);
    }
    for (int b = 0; b < 3; ++b) mbar_init(bar(h.b2full[b]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == WMMA) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(su32(&h.tmem)), "r"(TMEM_COLS));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (warp == WPROD && lane == 0) {   // the weights never come from the predecessor: fetch their descriptors now
    asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&p.tm13)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&p.tm2)) : "memory");
  }
  for (int i = tid; i < E; i += NTHR) h.tokmask[i] = 0;
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  cluster_arrive_relaxed();   // peers push into this CTA only after every CTA of the cluster arrived (barrier init)
  asm volatile("griddepcontrol.wait;" ::: "memory");

  // ---- router logits first (they gate the weight stream): every token of this warp, into registers ----
  constexpr int TPW = MAXT / NW;
  uint4 lg[TPW][kF32Logits ? 2 : 1];
#pragma unroll
  for (int q = 0; q < TPW; ++q) {
    const int t = warp + q * NW;
    if (t < T) {
      if constexpr (kF32Logits) {
        const uint4* lp = reinterpret_cast<const uint4*>(static_cast<const float*>(p.logits) + (size_t)t * E + lane * 8);
        lg[q][0] = __ldcg(lp);
        lg[q][kF32Logits ? 1 : 0] = __ldcg(lp + 1);
      } else {
        lg[q][0] = __ldcg(reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.logits) + (size_t)t * E + lane * 8));
      }
    }
  }
  // ---- activations -> FC1 B operand (cp.async): N rows (zero rows >= T) x 2048 B, 128B-swizzled K-major;
  //      scales -> 128x4 chunks ----
  {
    const uint4* xg = reinterpret_cast<const uint4*>(p.x);
#pragma unroll 4
    for (int i = tid; i < N * (H / 16); i += NTHR) {
      const int t = i / (H / 16), u = i % (H / 16);
      const int kc = u >> 3, un = u & 7;
      const uint32_t dst = sb + OB1 + kc * (N * 128) + (t >> 3) * 1024 + (t & 7) * 128 + ((un ^ (t & 7)) << 4);
      if (t < T) cp_async16(dst, xg + i);
      else st_shared_zero16(dst);
    }
    for (int i = tid; i < KC1 * N; i += NTHR) {
      const int g = i / N, n = i % N;
      const uint32_t dst = sb + OSFB1 + g * SFCH + n * 16;
      if (n < T) cp_async4(dst, reinterpret_cast<const unsigned int*>(p.xs) + n * (KB1 / 4) + g);
      else st_shared_zero4(dst);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }

  // ---- routing: top-k + softmax over the top-k, one warp per token ----
#pragma unroll
  for (int q = 0; q < TPW; ++q) {
    const int t = warp + q * NW;
    if (t >= T) break;
    uint32_t key[8];
    if constexpr (kF32Logits) {
      const uint4 v0 = lg[q][0], v1 = lg[q][kF32Logits ? 1 : 0];
      const uint32_t f[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) key[j] = twiddle32(f[j]);
    } else {
      const uint4 v = lg[q][0];
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
        const uint32_t hit = (lane == (ex >> 3)) ? (1u << (ex & 7)) : 0u;
#pragma unroll
        for (int j = 0; j < 8; ++j) key[j] = ((hit >> j) & 1u) ? 0u : key[j];
      } else {
        uint32_t m = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) m = max(m, key[j]);
        const uint32_t best = __reduce_max_sync(0xffffffffu, m);
        ex = 65535 - (int)(best & 0xFFFFu);
        val = untwiddle16(best >> 16);
        const uint32_t hit = (lane == (ex >> 3)) ? (1u << (ex & 7)) : 0u;
#pragma unroll
        for (int j = 0; j < 8; ++j) key[j] = ((hit >> j) & 1u) ? 0u : key[j];
      }
      if (lane == k) { myv = val; myexp = ex; }
    }
    // softmax over the top-k (lanes 0..7 hold rank 0..7), trtllm-gen calcSoftmax order
    float mx = (lane < TOPK) ? myv : -INFINITY;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float ev = 0.f;
    if (lane < TOPK) {
#if MDC_FAST_EXP
      ev = __expf(myv - mx);
#else
      ev = expf(myv - mx);
#endif
    }
    float sum = ev;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane < TOPK) {
      h.route_e[t * TOPK + lane] = (int16_t)myexp;
      atomicOr(&h.tokmask[myexp], 1u << t);
      if (blockIdx.x == 0) {
        p.topk_w[t * TOPK + lane] = __float2bfloat16_rn(ev / sum);
        if (p.topk_ids) p.topk_ids[t * TOPK + lane] = myexp;
      }
    }
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  fence_proxy_async();   // B1 / SFB1 (cp.async / generic stores) -> tensor core
  __syncthreads();       // routing (tokmask, route_e) and B1 / SFB1 complete
#if MDC_EARLY_PDL
  // Dependents may take the SMs this grid leaves free (their griddepcontrol.wait still waits for completion).
  if (tid == 0) asm volatile("griddepcontrol.launch_dependents;");
#endif
  const uint32_t tm0 = h.tmem;
  if (prof && tid == 0) prof[1] = gtime();
  // Experts j = 0 .. J - 1 of this cluster in the step order k = 0 .. J - 1 + LAG: step k = FC1(k) (k < J), then
  // FC2(k - LAG) (k >= LAG). Every role (TMA, SF helper, MMA, epilogue) walks the same steps.

  if (warp == WPROD) {
    // ---- weight producer: per step the FC1 tiles of expert k (K chunk inner), then the FC2 tiles of k - LAG.
    //      No L2 prefetch: any prefetch depth (8/16/32 stages beyond the ring) and the old next-expert prefetch
    //      were slower on VR (714415: T=6 C8 15.90 us without, 17.48 / 18.36 / 18.52 with). ----
    const int J = cluster_experts(h.tokmask, (int)cid, (int)ncl, lane);
    {
      // this cluster's experts -> h.cexp (producer-private): slot s of the distinct experts, s = cid + j * ncl
      int base = 0;
#pragma unroll 1
      for (int g = 0; g < E / 32; ++g) {
        const uint32_t sel = __ballot_sync(0xffffffffu, h.tokmask[g * 32 + lane] != 0);
        if ((sel >> lane) & 1u) {
          const int s = base + __popc(sel & ((1u << lane) - 1u)) - (int)cid;
          if (s >= 0 && s % (int)ncl == 0) h.cexp[s / (int)ncl] = (int16_t)(g * 32 + lane);
        }
        base += __popc(sel);
      }
      __syncwarp();
    }
    if (lane == 0 && J > 0) {
      // cursor over the stage sequence: (step k, part 0 = FC1(k) / 1 = FC2(k - LAG), stage s of the part)
      struct Cur { int k, part, s; };
      auto valid = [&](const Cur& c) { return c.k < J + LAG; };
      auto fix = [&](Cur& c) {
        while (valid(c) && !(c.part == 0 ? c.k < J : c.k >= LAG)) {
          if (c.part == 0) c.part = 1; else { c.part = 0; ++c.k; }
        }
      };
      auto adv = [&](Cur& c) {
        if (++c.s < (c.part ? S2 : S1)) return;
        c.s = 0;
        if (c.part == 0) c.part = 1; else { c.part = 0; ++c.k; }
        fix(c);
      };
      // box coordinates (x, y) in the part's tensor map and the stage's scale chunks
      auto where = [&](const Cur& c, int& x, int& y, const uint8_t*& sf) {
        if (c.part == 0) {
          const int e = h.cexp[c.k], f = c.s / (KC1 / KSTG), kc = (c.s % (KC1 / KSTG)) * KSTG, t1 = rank * F1 + f;
          x = kc * 128; y = e * (2 * I) + t1 * 128;
          sf = p.s13 + (size_t)e * (2 * I * KB1) + (t1 * (KB1 / 4) + kc) * SFCH;
        } else {
          const int e = h.cexp[c.k - LAG], f = c.s / (KC2 / KSTG), kc = (c.s % (KC2 / KSTG)) * KSTG, t2 = rank * F2 + f;
          x = kc * 128; y = e * H + t2 * 128;
          sf = p.s2 + (size_t)e * (H * KB2) + (t2 * (KB2 / 4) + kc) * SFCH;
        }
      };
      Cur ld{0, 0, 0};
      fix(ld);
      for (int i = 0; valid(ld); ++i, adv(ld)) {
        const int st = i % ns;
        if (i >= ns) mbar_wait_sleep(bar(h.empty[st]), ((i / ns) & 1) ^ 1, 32, 1);
        mbar_expect_tx(bar(h.full[st]), ASTG + SFSTG);
        int x, y;
        const uint8_t* sf;
        where(ld, x, y, sf);
        const CUtensorMap* map = ld.part ? &p.tm2 : &p.tm13;
#pragma unroll
        for (int q = 0; q < KSTG; ++q) tma2d(a_base + st * ASTG + q * ACH, map, x + q * 128, y, bar(h.full[st]));
        bulk_g2s(sb + OSFA + st * SFSTG, sf, SFSTG, bar(h.full[st]));
      }
    }
    __syncwarp();
  } else if (warp == WHLP) {
    // ---- SF helper: everything the MMA issuer would otherwise wait for. Per group of KGRP ring stages: the full
    //      waits, the weight scales -> TMEM (tcgen05.cp), then tcgen05.commit + a release arrive on grp[] (the
    //      issuer's only wait). Before a GEMM's first group: its accumulator is free (acc*_empty) and, for FC2, the
    //      cluster's intermediate has arrived (b2full) and its scales are assembled and copied to TMEM. ----
    const int J = cluster_experts(h.tokmask, (int)cid, (int)ncl, lane);
    int st = 0, ph = 0, gc = 0;   // lane 0
    auto groups = [&](int nst) {
      for (int s = 0; s < nst; s += KGRP) {
#pragma unroll
        for (int g = 0; g < KGRP; ++g) {
          mbar_wait(bar(h.full[st]), ph, 3);   // one spinning thread (the issuer's own waits are gone)
          if (prof && gc == 0 && g == 0) prof[2] = gtime();
          tc_fence_after();
#pragma unroll
          for (int q = 0; q < KSTG; ++q)
            tc_cp(tm0 + TSFA + 4 * (KSTG * st + q), sf_desc(sb + OSFA + st * SFSTG + q * SFCH));
          ph ^= (st + 1 == ns);
          st = (st + 1 == ns) ? 0 : st + 1;
        }
        tc_commit(bar(h.grp[gc % NGRP]));
        mbar_arrive(bar(h.grp[gc % NGRP]));
        ++gc;
      }
    };
    if (J > 0 && lane == 0) {
#pragma unroll 1
      for (int g = 0; g < KC1; ++g) tc_cp(tm0 + TSFB1 + 4 * g, sf_desc(sb + OSFB1 + g * SFCH));
    }
    for (int k = 0; k < J + LAG; ++k) {
      if (k < J && lane == 0) {
        mbar_expect_tx(bar(h.b2full[k % NB2]), B2TX);   // the phase of expert k - NB2 completed (waited below)
        if (k >= NA1) mbar_wait_sleep(bar(h.acc1_empty[k % NA1]), ((k / NA1) - 1) & 1, 20, 9);
        groups(S1);
      }
      if (k >= LAG) {
        const int j = k - LAG, b = j % NB2;
        if (lane == 0 && j >= 2) mbar_wait_sleep(bar(h.acc2_empty[j & 1]), ((j >> 1) - 1) & 1, 20, 4);
        mbar_wait(bar(h.b2full[b]), (j / NB2) & 1, 5);
        if (prof && lane == 0) prof[4] = gtime();
        tc_fence_after();
        fence_proxy_async();
        if (lane < N) {
          const uint32_t* sfs = reinterpret_cast<const uint32_t*>(smem + OSFS) + b * C * N;
#pragma unroll
          for (int g = 0; g < KC2; ++g) {
            uint32_t w;
            if constexpr (C == 4) {
              w = sfs[g * N + lane];
            } else {
              w = (sfs[(2 * g) * N + lane] & 0xFFFFu) | (sfs[(2 * g + 1) * N + lane] << 16);
            }
            *reinterpret_cast<uint32_t*>(smem + OSFB2 + ((j & 1) * KC2 + g) * SFCH + lane * 16) = w;
          }
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) {
#pragma unroll
          for (int g = 0; g < KC2; ++g)
            tc_cp(tm0 + TSFB2 + ((j & 1) * KC2 + g) * 4, sf_desc(sb + OSFB2 + ((j & 1) * KC2 + g) * SFCH));
          groups(S2);
        }
      }
      __syncwarp();
    }
  } else if (warp == WMMA) {
    // ---- MMA issuer: lane 0 alone; per group one wait, then MMAs and per-stage ring commits only (VR probes:
    //      every mbarrier wait costs ~75 ns in the issue stream even when complete, a tcgen05.cp ~24 ns) ----
    const int J = cluster_experts(h.tokmask, (int)cid, (int)ncl, lane);
    if (lane == 0) {
      int st = 0, gc = 0;
      for (int k = 0; k < J + LAG; ++k) {
        if (k < J) {
          const uint32_t d = tm0 + (k % NA1) * (F1 * N);
#pragma unroll 1
          for (int s = 0; s < S1; s += KGRP) {
            mbar_wait(bar(h.grp[gc % NGRP]), (gc / NGRP) & 1, 6);
            tc_fence_after();
            ++gc;
#pragma unroll
            for (int g = 0; g < KGRP; ++g) {
              const int s1 = s + g, f = s1 / (KC1 / KSTG), kc = (s1 % (KC1 / KSTG)) * KSTG;
              mma_stage<N>(d + f * N, a_base + st * ASTG, tm0 + TSFA + 4 * KSTG * st, sb + OB1 + kc * (N * 128),
                           tm0 + TSFB1 + 4 * kc, kc == 0);
              tc_commit(bar(h.empty[st]));
              st = (st + 1 == ns) ? 0 : st + 1;
            }
          }
          tc_commit(bar(h.acc1_full[k % NA1]));
          if (prof) prof[3] = gtime();
        }
        if (k >= LAG) {
          const int j = k - LAG;
          const uint32_t bbase = sb + OB2 + (j % NB2) * (N * I);
          const uint32_t sfb0 = tm0 + TSFB2 + (j & 1) * KC2 * 4;
          const uint32_t d0 = tm0 + TACC2 + (j & 1) * F2 * N;
#pragma unroll 1
          for (int s = 0; s < S2; s += KGRP) {
            mbar_wait(bar(h.grp[gc % NGRP]), (gc / NGRP) & 1, 6);
            tc_fence_after();
            ++gc;
#pragma unroll
            for (int g = 0; g < KGRP; ++g) {
              const int s2 = s + g, f = s2 / (KC2 / KSTG), kc = (s2 % (KC2 / KSTG)) * KSTG;
              mma_stage<N>(d0 + f * N, a_base + st * ASTG, tm0 + TSFA + 4 * KSTG * st, bbase + kc * (N * 128),
                           sfb0 + kc * 4, kc == 0);
              tc_commit(bar(h.empty[st]));
              st = (st + 1 == ns) ? 0 : st + 1;
            }
          }
          tc_commit(bar(h.acc2_full[j & 1]));
        }
      }
      if (prof) prof[8] = J;
    }
    __syncwarp();
  } else if (warp < 4) {
    // ---- epilogue warps (TMEM lane quadrant = warp) ----
    cluster_wait();
    const int J = cluster_experts(h.tokmask, (int)cid, (int)ncl, lane);
    const uint32_t tl = tm0 + ((uint32_t)(warp * 32) << 16);
    const bool isg = (lane & 8) != 0;                                          // gate row (else up)
    const int il = 16 * warp + ((lane & 7) << 1) + ((lane >> 4) & 1);         // intermediate column in the tile
    float* act = reinterpret_cast<float*>(smem + OACT);
    long long tw_acc1 = 0;
    for (int k = 0; k < J + LAG; ++k) {
      if (k < J) {
        // FC1 epilogue of expert k: SwiGLU, requant, push to the cluster
        const int a1 = k % NA1, buf = k % NB2;
        {
          const long long t0 = prof ? gtime() : 0;
          mbar_wait_sleep(bar(h.acc1_full[a1]), (k / NA1) & 1, 32, 7);
          if (prof) tw_acc1 += gtime() - t0;
        }
        tc_fence_after();
        named_bar(1, 128);   // the previous expert's quantize pass is done with act[]
#pragma unroll
        for (int f = 0; f < F1; ++f) {
          float v[N];
          tmem_ld<N>(tl + (a1 * F1 + f) * N, v);
          tmem_wait_ld();
          if (p.dbg && k == 0) {
#pragma unroll
            for (int c = 0; c < N; ++c) p.dbg[((size_t)blockIdx.x * (F1 + F2) * 128 + f * 128 + warp * 32 + lane) * 32 + c] = v[c];
          }
#pragma unroll
          for (int c = 0; c < N / 2; ++c) {
            // lanes l / l ^ 8 hold gate / up of the same column: each computes half of the token columns
            const float send = isg ? v[c + N / 2] : v[c];
            const float recv = __shfl_xor_sync(0xffffffffu, send, 8);
            const float g = isg ? v[c] : recv;
            const float u = isg ? recv : v[c + N / 2];
            const int col = isg ? c : c + N / 2;
#if MDC_FAST_EXP
            const float sg = g / (1.f + __expf(-g));
#else
            const float sg = g / (1.f + expf(-g));
#endif
            act[(f * N + col) * 64 + il] = sg * u;
          }
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(bar(h.acc1_empty[a1]));
        named_bar(1, 128);
        // MXFP8 requant: item = (token n, tile f, 32-block b, half hh) = 16 values -> 16 B of every CTA's FC2 B operand
        constexpr int G = F1 * 4, NIT = N * G;   // NIT is a multiple of 32: warp-uniform loop
        const uint32_t b2bar = bar(h.b2full[buf]);
        for (int it = tid; it < NIT; it += 128) {
          const int n = it / G, r = it % G, f = r >> 2, b = (r >> 1) & 1, hh = r & 1;
          const float4* src = reinterpret_cast<const float4*>(act + (f * N + n) * 64 + b * 32 + hh * 16);
          float a[16];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float4 v4 = src[q];
            a[4 * q] = v4.x; a[4 * q + 1] = v4.y; a[4 * q + 2] = v4.z; a[4 * q + 3] = v4.w;
          }
          float amax = 0.f;
#pragma unroll
          for (int c = 0; c < 16; ++c) amax = fmaxf(amax, fabsf(a[c]));
          amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
          int ex;
          if (p.qmode == 1) {
            ex = (int)((__float_as_uint(amax) >> 23) & 0xFF) - 8;
          } else {
            const uint32_t bits = __float_as_uint(fmaxf(amax / 448.f, 1.17549435e-38f));
            ex = (int)((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0);
          }
          ex = min(max(ex, 1), 253);
          const float inv = __uint_as_float((uint32_t)(254 - ex) << 23);
          uint32_t q[4];
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            uint16_t lo, hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(a[4 * w + 1] * inv), "f"(a[4 * w] * inv));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(a[4 * w + 3] * inv), "f"(a[4 * w + 2] * inv));
            q[w] = (uint32_t)lo | ((uint32_t)hi << 16);
          }
          // scale word of (sender rank, token n): byte f * 2 + b, gathered from the hh == 0 lanes of the group
          uint32_t word = 0;
#pragma unroll
          for (int jj = 0; jj < 2 * F1; ++jj)
            word |= (uint32_t)__shfl_sync(0xffffffffu, ex, (lane & ~(G - 1)) + 2 * jj) << (8 * jj);
          const int i0 = ((int)rank * F1 + f) * 64 + b * 32 + hh * 16;
          const int kc = i0 >> 7, un = (i0 & 127) >> 4;
          const uint32_t off = OB2 + buf * (N * I) + kc * (N * 128) + (n >> 3) * 1024 + (n & 7) * 128 + ((un ^ (n & 7)) << 4);
          const uint32_t soff = OSFS + ((buf * C + rank) * N + n) * 4;
#pragma unroll
          for (int d = 0; d < C; ++d) {
            const uint32_t rb = mapa(b2bar, d);
            st_async_v4(mapa(sb + off, d), q, rb);
            if ((lane & (G - 1)) == 0) st_async_b32(mapa(sb + soff, d), word, rb);
          }
        }
        if (prof && tid == 0) prof[5] = gtime();
      }
      if (k >= LAG) {
        // FC2 epilogue of expert j: BF16 rows of its routes. N <= 16: staged in SMEM ([N][H / C], this CTA's
        // contiguous output columns) and written with one bulk copy per route (per-lane 2-byte stores cost
        // ~1 us per expert at C4 under the weight stream: 714415 NOSTORE probe).
        const int j = k - LAG;
        const int e = slot_expert(h.tokmask, (int)(cid + j * ncl), lane);
        const uint32_t mask = h.tokmask[e];
        if constexpr (OSTAGE) {
          if (j > 0) {
            if (tid == 0) bulk_wait_read();   // the previous expert's bulk stores are done reading the staging
            named_bar(1, 128);
          }
        }
        mbar_wait_sleep(bar(h.acc2_full[j & 1]), (j >> 1) & 1, 32, 8);
        tc_fence_after();
        if (prof && tid == 0) prof[6] = gtime();
#pragma unroll
        for (int f = 0; f < F2; ++f) {
          const int ol = f * 128 + unshuffle32(warp * 32 + lane);   // output column - rank * (H / C)
          float v[N];
          tmem_ld<N>(tl + TACC2 + ((j & 1) * F2 + f) * N, v);
          tmem_wait_ld();
          if (p.dbg && j == 0) {
#pragma unroll
            for (int c = 0; c < N; ++c)
              p.dbg[((size_t)blockIdx.x * (F1 + F2) * 128 + (F1 + f) * 128 + warp * 32 + lane) * 32 + c] = v[c];
          }
#pragma unroll
          for (int n = 0; n < N; ++n) {
            if ((mask >> n) & 1u) {
              if constexpr (OSTAGE) {
                reinterpret_cast<__nv_bfloat16*>(smem + OSTO)[n * (H / C) + ol] = __float2bfloat16_rn(v[n]);
              } else {
                p.out[(size_t)(n * TOPK + route_k(h.route_e, n, e)) * H + rank * (H / C) + ol] = __float2bfloat16_rn(v[n]);
              }
            }
          }
        }
        if constexpr (OSTAGE) fence_proxy_async();   // staging (generic stores) -> bulk copies
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(bar(h.acc2_empty[j & 1]));
        if constexpr (OSTAGE) {
          named_bar(1, 128);
          if (tid == 0) {
            for (int n = 0; n < N; ++n) {
              if ((mask >> n) & 1u)
                bulk_s2g(p.out + (size_t)(n * TOPK + route_k(h.route_e, n, e)) * H + rank * (H / C),
                         sb + OSTO + n * (H / C) * 2, (H / C) * 2);
            }
            bulk_commit();
          }
        }
      }
    }
    if constexpr (OSTAGE) {
      if (tid == 0) bulk_wait_all();   // output written before the CTA exits
    }
    if (prof && tid == 0) {
      prof[9] = gtime();
      prof[11] = tw_acc1;
    }
  }
  __syncthreads();
  if (prof && tid == 0) prof[10] = gtime();
  if (warp >= 4) cluster_wait();
  if (warp == WMMA) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tm0), "r"(TMEM_COLS));
  }
  if (prof && tid == 0) prof[7] = gtime();
#if !MDC_EARLY_PDL
  asm volatile("griddepcontrol.launch_dependents;");
#endif
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
  cuuint32_t box[2] = {128, 128};
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

static int n_cols(int T) { return T <= 8 ? 8 : (T <= 16 ? 16 : 32); }

template <int C, int N, bool F32>
static void* kernel_ptr() { return reinterpret_cast<void*>(moe_tc_kernel<C, N, F32>); }

template <int C, bool F32>
static void* kernel_for_n(int N) {
  if (N == 8) return kernel_ptr<C, 8, F32>();
  if (N == 16) return kernel_ptr<C, 16, F32>();
  return kernel_ptr<C, 32, F32>();
}

static void* kernel_for(int C, int N, bool f32) {
  TORCH_CHECK(C == 4 || C == 8, "cluster size must be 4 or 8");
  if (C == 4) return f32 ? kernel_for_n<4, true>(N) : kernel_for_n<4, false>(N);
  return f32 ? kernel_for_n<8, true>(N) : kernel_for_n<8, false>(N);
}

static void set_smem_attr(void* kern) {
  static std::unordered_map<void*, bool> done;
  if (done.count(kern)) return;
  int dev, mx;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, mx));
  done[kern] = true;
}

// Dynamic SMEM of a launch: at least half the opt-in limit + 1 KB, so a CTA (which allocates all 512 TMEM
// columns) never shares its SM with another CTA of this kernel (a co-resident cluster peer would deadlock).
static int launch_smem(int C, int N, int ns) {
  static int half = -1;
  if (half < 0) {
    int dev, mx;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    half = mx / 2 + 1024;
  }
  return std::max(smem_total(C, N, ns), half);
}

// Ring stages that fit (SMEM opt-in limit, TMEM columns, barrier slots).
int64_t max_stages(int64_t C, int64_t T, int64_t smem_max) {
  const int N = n_cols((int)T);
  int ns = 0;
  while (ns < MAXST && smem_total((int)C, N, ns + 1) <= smem_max && tm_sfa((int)C, N) + 4 * KSTG * (ns + 1) <= TMEM_COLS) ++ns;
  return ns;
}

int64_t smem_bytes(int64_t C, int64_t T, int64_t ns) { return smem_total((int)C, n_cols((int)T), (int)ns); }

// Clusters of C CTAs that can be co-resident with this kernel's SMEM.
int64_t max_clusters(int64_t C, int64_t T, int64_t ns) {
  const int N = n_cols((int)T);
  void* kern = kernel_for((int)C, N, false);
  set_smem_attr(kern);
  int dev, sms;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)((sms / C) * C));
  cfg.blockDim = dim3(NTHR);
  cfg.dynamicSmemBytes = launch_smem((int)C, N, (int)ns);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = (unsigned)C;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int n = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, kern, &cfg));
  return n;
}

void run(at::Tensor logits, at::Tensor x, at::Tensor xs, at::Tensor w13, at::Tensor s13, at::Tensor w2,
         at::Tensor s2, at::Tensor out, at::Tensor topk_w, c10::optional<at::Tensor> topk_ids,
         int64_t C, int64_t nclusters, int64_t ns, int64_t qmode, bool pdl, c10::optional<at::Tensor> prof,
         c10::optional<at::Tensor> dbg) {
  const int T = (int)x.size(0);
  const int N = n_cols(T);
  TORCH_CHECK(T >= 1 && T <= MAXT && ns >= 2 && ns <= MAXST && nclusters >= 1);
  TORCH_CHECK(tm_sfa((int)C, N) + 4 * KSTG * ns <= TMEM_COLS);
  TORCH_CHECK(x.size(1) == H && xs.numel() >= T * KB1 && logits.size(1) == E);
  TORCH_CHECK(x.is_contiguous() && xs.is_contiguous() && logits.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(w13.numel() == (int64_t)E * 2 * I * H && w2.numel() == (int64_t)E * H * I);
  TORCH_CHECK(s13.numel() == (int64_t)E * 2 * I * KB1 && s2.numel() == (int64_t)E * H * KB2);
  static_assert(sizeof(Hdr) <= HDR, "hdr");
  Params p;
  p.tm13 = cached_map(w13.data_ptr(), (uint64_t)E * 2 * I, H);
  p.tm2 = cached_map(w2.data_ptr(), (uint64_t)E * H, I);
  p.logits = logits.data_ptr();
  p.x = (const uint8_t*)x.data_ptr(); p.xs = (const uint8_t*)xs.data_ptr();
  TORCH_CHECK((E + nclusters - 1) / nclusters <= MAXJ, "too few clusters");
  p.s13 = (const uint8_t*)s13.data_ptr(); p.s2 = (const uint8_t*)s2.data_ptr();
  p.out = (__nv_bfloat16*)out.data_ptr(); p.topk_w = (__nv_bfloat16*)topk_w.data_ptr();
  p.topk_ids = topk_ids.has_value() ? (int32_t*)topk_ids->data_ptr() : nullptr;
  p.prof = prof.has_value() ? (long long*)prof->data_ptr() : nullptr;
  p.dbg = dbg.has_value() ? (float*)dbg->data_ptr() : nullptr;
  p.T = T; p.ns = (int)ns; p.qmode = (int)qmode;
  void* kern = kernel_for((int)C, N, logits.scalar_type() == at::kFloat);
  set_smem_attr(kern);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(nclusters * C));
  cfg.blockDim = dim3(NTHR);
  cfg.dynamicSmemBytes = launch_smem((int)C, N, (int)ns);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = (unsigned)C;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 2 : 1;
  void* args[] = {&p};
  C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg, kern, args));
}
}  // namespace mdc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &mdc::run);
  m.def("max_stages", &mdc::max_stages);
  m.def("smem_bytes", &mdc::smem_bytes);
  m.def("max_clusters", &mdc::max_clusters);
}
"""

E, TOPK, H, INTER = 256, 8, 2048, 512
MAXT = 32


def tuned_source(tune: str) -> str:
    defines = ""
    for item in filter(None, (x.strip() for x in tune.replace(",", "+").split("+"))):
        key, val = item.split("=")
        assert key in ("FAST_EXP", "DEBUG", "EARLY_PDL"), key
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
    """Launcher for one device (shared by all MoE layers; calls are stream-ordered, no workspace)."""

    def __init__(self, device, num_sms: int | None = None):
        self.ext = load()
        self.device = torch.device(device)
        perm13, perm2 = permutations(self.device)
        # The kernel uses the closed form of these permutations; refuse a layout it does not match.
        n = torch.arange(H, device=self.device)
        old = (n // 32) * 32 + (n % 8) * 4 + (n % 32) // 8
        cf13 = torch.where(
            old[: 2 * INTER] % 2 == 0,
            INTER + old[: 2 * INTER] // 2,
            old[: 2 * INTER] // 2,
        )
        if not (torch.equal(cf13, perm13.long()) and torch.equal(old, perm2.long())):
            raise RuntimeError(
                "trtllm-gen MXFP8 weight shuffle changed; moe_decode_cuda needs updating"
            )
        props = torch.cuda.get_device_properties(self.device)
        self.num_sms = num_sms or props.multi_processor_count
        self.smem_max = props.shared_memory_per_block_optin
        self.cluster = CLUSTER
        self.stages = STAGES  # 0 = as many as fit
        self.clusters = CLUSTERS  # 0 = as many as can be co-resident
        # [grid, 32] int64 %globaltimer stamps (microbench)
        self.prof: torch.Tensor | None = None
        # [grid, (8/C + 16/C) * 128, 32] f32 accumulators of each CTA's first expert (debug)
        self.dbg: torch.Tensor | None = None
        self._cfg: dict[tuple[int, int, int, int], tuple[int, int]] = {}
        self._idx: dict[int, torch.Tensor] = {}

    def config(self, T: int) -> tuple[int, int]:
        """(ring stages, persistent clusters) for T tokens."""
        n = 8 if T <= 8 else (16 if T <= 16 else 32)
        key = (n, self.cluster, self.stages, self.clusters)
        cfg = self._cfg.get(key)
        if cfg is None:
            ns = self.ext.max_stages(self.cluster, T, self.smem_max)
            if self.stages:
                ns = min(ns, self.stages)
            assert ns >= 2, (T, self.cluster, self.smem_max)
            ncl = min(
                self.ext.max_clusters(self.cluster, T, ns),
                self.num_sms // self.cluster,
            )
            if self.clusters:
                ncl = min(ncl, self.clusters)
            assert ncl >= 1, (T, self.cluster, ns)
            cfg = self._cfg[key] = (ns, ncl)
        return cfg

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
        ns, ncl = self.config(T)
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
            out,
            topk_w,
            topk_ids,
            self.cluster,
            ncl,
            ns,
            QMODE,
            pdl,
            self.prof,
            self.dbg,
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
                "MoE decode: single-launch tcgen05 kernel for %d <= T <= %d "
                "(cluster %d, qmode %d, tune '%s').",
                MIN_TOKENS,
                min(MAX_TOKENS, MAXT),
                inst.cluster,
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
