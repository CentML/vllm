# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Domain-local MXFP8 skinny GEMM on the SM107 block-scaled tensor cores.

``DomainMxGemm`` computes ``out[M, N] = x[M, K] @ W[N, K]^T`` (bf16 out, fp32
accumulation) for an MXFP8 weight: e4m3 ``W`` in localized memory
(``memory.localize``, 2 MiB-interleaved) plus its E8M0 1x32 scales in the
128x4-swizzled layout that FlashInfer's ``mm_mxfp8`` takes. ``x`` (bf16,
``M <= 32``) is quantized to MXFP8 inside the kernel with FlashInfer's
``mxfp8_quantize`` rule (UE8M0 = ceil of ``amax * (1/448)``, values
``x * 2^(127 - e)`` rounded to e4m3, saturating), so the products are the ones
FlashInfer forms; only the fp32 summation order differs.

Kernel (one persistent CTA per SM, 192 threads, ~206 KB SMEM):

- swap-AB: 128 weight rows are the MMA's M, the tokens its N (16 or 32,
  zero-padded). ``tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale``
  (SM107: K = 64 per instruction, idesc bit 31), fp32 accumulators in TMEM.
- warp 0 (TMA): streams 32 KB weight stages (two 128x128 B SWIZZLE_128B tensor
  boxes, L2 evict-first) plus their 1 KB scale atoms into a 5-stage
  (N = 32: 4-stage) ring; it never drains between work units.
- warps 2-5 (epilogue): first quantize x into a resident SMEM B operand
  (all 16 K blocks, SW128 K-major) and its scale atoms, then turn the TMEM
  accumulators into bf16 logits.
- warp 1 (MMA): copies the 16 B-scale atoms into TMEM once, then per stage
  2 ``tcgen05.cp`` (A scales) + 4 MMAs + commit; two row blocks (4 quarter
  accumulators each) in flight.

Work split. CTA i runs on SM i's slot (cooperative launch: one CTA per SM),
which belongs to that SM's locality domain (``topology.sm_domain``). A domain's
128-row blocks are the ones whose 2 MiB chunk lives on it. Every row block is
the fixed sum ``((q0 + q1) + (q2 + q3))`` of its four K quarters (each its own
TMEM accumulator), and the quarter is the unit of work: every SM gets an equal
contiguous range of its domain's quarters. The logits do not depend on the
split (SMs finish ~10 % apart, but the stream is DRAM-bound: the stragglers
speed up as the others finish, so unequal splits measured no faster). A row block
shared by two neighbouring ranges is completed through a workspace by whichever
CTA finishes its quarters last; the shared blocks are processed first, so these
fixups overlap the stream and every CTA ends on a whole row block. Fixup flags
are reset in-kernel: launches are CUDA-graph replayable. Weight loads start
before ``griddepcontrol.wait`` (PDL launch, ``pdl=True``); x and the flags are
read after it.

Only sm_107 (VR200) is supported: the MMA encoding is SM107-specific.
"""

from __future__ import annotations

import torch

from . import _ext
from .memory import Localized, localize, row_group_tiles
from .topology import Topology

MAX_M = 32
K_DIM = 2048
_mod: list = []


def supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


def load():
    if not _mod:
        _mod.append(_ext.build("locality_mx", _SOURCE))
    return _mod[0]


class DomainMxGemm:
    """See the module docstring. ``weight``: e4m3 (or uint8) ``[N, 2048]``,
    ``N % 128 == 0``, preferably ``Localized`` with the ``interleave`` layout
    (otherwise the row blocks are split alternately: the same kernel without
    locality, a control arm). ``scale_swizzled``: the weight's E8M0 scales in
    FlashInfer's 128x4 layout (``swizzle_mxfp8_scale``); one copy per domain is
    made here.
    """

    def __init__(
        self,
        topo: Topology,
        weight: Localized | torch.Tensor,
        scale_swizzled: torch.Tensor,
        max_m: int = MAX_M,
    ) -> None:
        w = weight.tensor if isinstance(weight, Localized) else weight
        w = w.view(torch.uint8)
        assert w.dim() == 2 and w.is_contiguous()
        self.n, self.k = w.shape
        assert self.k == K_DIM and self.n % 128 == 0, (self.n, self.k)
        assert 1 <= max_m <= MAX_M
        ext = load()
        self.w = w
        self.tmap = ext.make_tmap(w)
        sf = scale_swizzled.contiguous().view(torch.uint8).reshape(-1)
        assert sf.numel() == self.n * self.k // 32
        rb_bytes = 128 * self.k
        if isinstance(weight, Localized):
            self.sfa = (localize(sf, "dom0").tensor, localize(sf, "dom1").tensor)
            ords, cb = weight.ordinals, weight.chunk_bytes
            assert cb % rb_bytes == 0
        else:
            self.sfa = (sf, sf)
            ords, cb = None, rb_bytes
        rpc = cb // rb_bytes
        assert rpc & (rpc - 1) == 0
        self.rpc_log2 = rpc.bit_length() - 1
        # the kernel derives domain d's j-th row block arithmetically (chunks alternate between the domains)
        rbl = row_group_tiles(w, self.n, self.k, ords, cb, 128)
        for d in (0, 1):
            j = torch.arange(rbl[d].numel(), device=w.device, dtype=torch.int32)
            if not torch.equal(rbl[d], ((j // rpc) * 2 + d) * rpc + j % rpc):
                raise ValueError(
                    "weight chunks do not alternate between the two domains (use the interleave layout)"
                )
        self.nrb = (rbl[0].numel(), rbl[1].numel())
        dom = topo.sm_domain.tolist()
        assert len(dom) <= 256 and all(v in (0, 1) for v in dom)
        self.sm_dom = dom
        self.nslot = (dom.count(0), dom.count(1))
        self.max_m = max_m
        self.ws_t = (max_m + 3) // 4 * 4
        self.ws_slots = max(self.nslot) + 1
        dev = w.device
        self.ws = torch.empty(
            2 * self.ws_slots * 4 * self.ws_t * 128, dtype=torch.float32, device=dev
        )
        self.flags = torch.zeros(2 * self.ws_slots, dtype=torch.int32, device=dev)
        self._plan()
        self._keep = weight

    def _plan(self) -> None:
        """Split each domain's quarter units evenly over its SMs (contiguous
        ranges). The logits do not depend on the split:
        every row block is the fixed sum ((q0 + q1) + (q2 + q3)) of its K
        quarters.
        """
        dom = self.sm_dom
        u0, u1 = [0] * len(dom), [0] * len(dom)
        for d in (0, 1):
            sms = [s for s, v in enumerate(dom) if v == d]
            units = 4 * self.nrb[d]
            for r, s in enumerate(sms):
                u0[s], u1[s] = units * r // len(sms), units * (r + 1) // len(sms)
            # >= 3 row blocks per slot: each shared block has exactly two owners
            assert all(u1[s] - u0[s] >= 12 for s in sms), "slot below 3 row blocks"
        slot, cnt = [], [0, 0]
        for v in dom:
            slot.append((v << 15) | cnt[v])
            cnt[v] += 1
        self.plan = torch.tensor(
            slot + u0 + u1 + [self.rpc_log2, self.ws_slots], dtype=torch.int32
        )

    @property
    def extra_bytes(self) -> int:
        """Device bytes held besides the weight (scale copies, workspace)."""
        return sum(
            t.numel() * t.element_size() for t in (*self.sfa, self.ws, self.flags)
        )

    def __call__(
        self,
        x: torch.Tensor,
        out: torch.Tensor | None = None,
        pdl: bool = False,
        prof: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x``: bf16 ``[M, 2048]``, rows 16-B aligned. ``pdl``: launch with
        programmatic stream serialization. ``prof`` (diagnostics): int64
        ``[16 * num_sms]``; each CTA writes its %globaltimer stamps there (0
        start, 1 first TMA, 2 B operand ready, 3 / 4 first / last full stage,
        5 epilogue done; 6 SM id, 7 slot).
        """
        m = x.shape[0]
        assert 1 <= m <= self.max_m
        if out is None:
            out = torch.empty((m, self.n), dtype=torch.bfloat16, device=x.device)
        load().mx_head(
            x,
            out,
            self.tmap,
            self.sfa[0],
            self.sfa[1],
            self.plan,
            self.ws,
            self.ws_t,
            self.flags,
            1 if pdl else 0,
            prof if prof is not None else self.flags,
        )
        return out


_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>

namespace mxg {

#define DRV(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorName(r_, &s_); TORCH_CHECK(false, #x " failed: ", s_); } } while (0)
#define RTC(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { cudaGetLastError(); TORCH_CHECK(false, #x " failed: ", cudaGetErrorString(e_)); } } while (0)

typedef unsigned long long u64;
constexpr int K = 2048, KB = K / 128, KSB = 2;  // 16 K blocks of 128; stages of 2 K blocks (2 stages per K quarter)
constexpr int A_BLK = 16384, A_ST = KSB * A_BLK, SF_ATOM = 512, SF_ST = KSB * SF_ATOM;
constexpr int NTHREADS = 192, TMEM_COLS = 512;
constexpr uint32_t INV448 = 0x3B124925u;  // fp32(1 / 448), FlashInfer's INV_FLOAT8_E4M3_MAX

template <int NT> struct Geo {
  static constexpr int S = NT <= 16 ? 5 : 4;      // weight stages in flight (32 KB each)
  static constexpr int NBUF = 2;                  // TMEM accumulator sets, one row block (4 quarters) each
  static constexpr int B_KB = NT * 128;           // resident B bytes per K block
  static constexpr int ACC_COL = 0, SFA_COL = NBUF * 4 * NT, SFB_COL = SFA_COL + 16;
  static constexpr int OFF_B = S * A_ST, OFF_SFA = OFF_B + KB * B_KB, OFF_SFB = OFF_SFA + S * SF_ST;
  static constexpr int OFF_BAR = OFF_SFB + KB * SF_ATOM, NBAR = 2 * S + 2 * NBUF + 1;
  static constexpr int OFF_INFO = OFF_BAR + NBAR * 8;   // int [0] domain [1] slot [2] flag broadcast [3] tmem [4, 5] units
  static constexpr int SMEM = OFF_INFO + 8 * 4 + 1024;
  static_assert(SFB_COL + 4 * KB <= TMEM_COLS, "TMEM columns");
  static_assert(SMEM <= 232448, "SMEM");
};

struct Params {
  const __nv_bfloat16* x; const unsigned char* sfa[2];
  __nv_bfloat16* out; float* ws; int* flags; u64* prof;
  long long ldx, ldo;
  int M, wst, opts, rpc_log2, wsslots;
  int slot[256];          // per SM: (domain << 15) | rank within the domain
  int u0[256], u1[256];   // per SM: its quarter units [u0, u1) in the domain's row-block x quarter order
};

// ------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ u64 gtime() { u64 r; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); return r; }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mbar_expect(uint32_t a, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(tx) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t ph) {
  asm volatile("{\n\t.reg .pred P;\nLW%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t@!P bra LW%=;\n}" :: "r"(a), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar, u64 pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
               :: "r"(dst), "l"((u64)tm), "r"(c0), "r"(c1), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, u64 pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_commit(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void tc_cp_sf(uint32_t taddr, u64 sdesc) { asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(sdesc) : "memory"); }
__device__ __forceinline__ void tc_mma(uint32_t d, u64 ad, u64 bd, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}\n"
               :: "r"(d), "l"(ad), "l"(bd), "r"(idesc), "r"(acc), "r"(sfa), "r"(sfb) : "memory");
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float* v) {
  uint32_t r[16];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 16; i++) v[i] = __uint_as_float(r[i]);
}
// UMMA shared-memory descriptors (SM100/SM107 format)
__device__ __forceinline__ u64 sdesc_sw128(uint32_t saddr) {      // K-major, SWIZZLE_128B, 8-row atoms 1024 B apart
  return (u64)((saddr >> 4) & 0x7FFF) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 46) | ((u64)2 << 61);
}
__device__ __forceinline__ u64 sdesc_sf(uint32_t saddr) {         // tcgen05.cp source: 32 rows x 16 B, SBO 128 B
  return (u64)((saddr >> 4) & 0x7FFF) | ((u64)(128 >> 4) << 32) | ((u64)1 << 46);
}
// SM107 kind::mxf8f6f4 instruction descriptor: E4M3 x E4M3, K-major both, UE8M0 scales, M = 128, K = 64
__device__ __forceinline__ uint32_t idesc_mx(int n, uint32_t sfa_id, uint32_t sfb_id) {
  return (sfb_id << 4) | ((uint32_t)(n >> 3) << 17) | (1u << 23) | (1u << 27) | (sfa_id << 29) | (1u << 31);
}

// FlashInfer mxfp8_quantize: e = ue8m0 of amax * fp32(1/448) rounded up (0 for amax == 0), value scale 2^(127 - e)
__device__ __forceinline__ uint32_t e8m0_of(float amax) {
  const float nm = amax * __uint_as_float(INV448);
  const uint32_t bits = __float_as_uint(nm);
  uint32_t e = ((bits >> 23) & 255u) + ((bits & 0x7FFFFFu) != 0u ? 1u : 0u);
  e = e > 254u ? 254u : e;
  return nm <= 0.f ? 0u : e;
}
__device__ __forceinline__ float inv_scale(uint32_t e) {
  const int ne = 254 - (int)e;
  return e == 0u ? 0.f : __uint_as_float((uint32_t)(ne < 0 ? 0 : ne) << 23);
}
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return r;
}

// A slot's quarter units [U0, U1) cover the domain's row blocks j0 .. j0 + n - 1 (n >= 3). The first and the last
// may be shared with the neighbouring slots (their quarters meet in the workspace); they are processed first, so the
// cross-CTA fixups overlap the stream and every CTA ends on a whole row block. next(): row block index j in the
// domain's list, quarters [qa, qb).
struct Seg {
  int j0, n, qa0, qb1, i;
  __device__ __forceinline__ Seg(int U0, int U1)
      : j0(U0 >> 2), n(((U1 - 1) >> 2) - (U0 >> 2) + 1), qa0(U0 & 3), qb1(((U1 - 1) & 3) + 1), i(0) {}
  __device__ __forceinline__ bool next(int& j, int& qa, int& qb) {
    if (i >= n) return false;
    const int li = i == 0 ? 0 : (i == 1 ? n - 1 : i - 1);
    i++;
    j = j0 + li; qa = li == 0 ? qa0 : 0; qb = li == n - 1 ? qb1 : 4;
    return true;
  }
};

// x [M, K] bf16 -> resident B operand (SW128 K-major, rows >= M zero) + B scale atoms (rows >= M: 2^0)
template <int NT>
__device__ __forceinline__ void quantize_x(const Params& p, unsigned char* sB, unsigned char* sSFB, int et) {
  constexpr int B_KB = NT * 128;
  for (int c = et; c < KB * SF_ATOM / 16; c += 128) reinterpret_cast<uint4*>(sSFB)[c] = make_uint4(0x7F7F7F7Fu, 0x7F7F7F7Fu, 0x7F7F7F7Fu, 0x7F7F7F7Fu);
  for (int c = et; c < (NT - p.M) * KB * 8; c += 128) {
    const int t = p.M + c / (KB * 8), rest = c % (KB * 8);
    *reinterpret_cast<uint4*>(sB + (rest >> 3) * B_KB + t * 128 + ((rest & 7) << 4)) = make_uint4(0u, 0u, 0u, 0u);
  }
  named_bar(1, 128);
  const int nitem = p.M * (K / 32);
  for (int base = 0; base < nitem; base += 4 * 128) {
    uint4 v[4][4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int item = base + u * 128 + et;
      if (item < nitem) {
        const uint4* src = reinterpret_cast<const uint4*>(p.x + (size_t)(item >> 6) * p.ldx + (item & 63) * 32);
#pragma unroll
        for (int e = 0; e < 4; e++) v[u][e] = src[e];
      }
    }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int item = base + u * 128 + et;
      if (item < nitem) {
        const int t = item >> 6, blk = item & 63;
        float f[32];
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v[u][e]);
#pragma unroll
          for (int q = 0; q < 4; q++) { const float2 g = __bfloat1622float2(h[q]); f[e * 8 + 2 * q] = g.x; f[e * 8 + 2 * q + 1] = g.y; }
        }
        float amax = 0.f;
#pragma unroll
        for (int e = 0; e < 32; e++) amax = fmaxf(amax, fabsf(f[e]));
        const uint32_t sc = e8m0_of(amax);
        const float inv = inv_scale(sc);
        uint32_t w[8];
#pragma unroll
        for (int c = 0; c < 8; c++) w[c] = e4m3x2(f[4 * c] * inv, f[4 * c + 1] * inv) | (e4m3x2(f[4 * c + 2] * inv, f[4 * c + 3] * inv) << 16);
        const int kb = blk >> 2, j = (blk & 3) * 2;
        unsigned char* rowp = sB + kb * B_KB + t * 128;
        *reinterpret_cast<uint4*>(rowp + ((j ^ (t & 7)) << 4)) = make_uint4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<uint4*>(rowp + (((j + 1) ^ (t & 7)) << 4)) = make_uint4(w[4], w[5], w[6], w[7]);
        sSFB[kb * SF_ATOM + (t & 31) * 16 + (t >> 5) * 4 + (blk & 3)] = (unsigned char)sc;
      }
    }
  }
}

template <int NT>
__global__ void __launch_bounds__(NTHREADS, 1) k_mxhead(const __grid_constant__ CUtensorMap tmap, const Params p) {
  using G = Geo<NT>;
  constexpr int S = G::S;
  extern __shared__ unsigned char smem_raw[];
  // align by pointer arithmetic on the __shared__ array (an integer cast would drop the address space)
  unsigned char* sm = smem_raw + ((1024u - (sa(smem_raw) & 1023u)) & 1023u);
  unsigned char* sA = sm;
  unsigned char* sB = sm + G::OFF_B;
  unsigned char* sSFA = sm + G::OFF_SFA;
  unsigned char* sSFB = sm + G::OFF_SFB;
  u64* full = reinterpret_cast<u64*>(sm + G::OFF_BAR);
  u64* empty = full + S; u64* accf = empty + S; u64* acce = accf + G::NBUF; u64* bready = acce + G::NBUF;
  volatile int* info = reinterpret_cast<volatile int*>(sm + G::OFF_INFO);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (warp == 0) {
    // ---------------------------------------------------------------- TMA producer (also sets the CTA up)
    if (lane == 0) {
      const u64 t_start = p.prof ? gtime() : 0;
      // per-SM plan first: its loads overlap the barrier setup
      const unsigned sid = smid();
      const int v = p.slot[sid], d = v >> 15, U0 = p.u0[sid], U1 = p.u1[sid];
      asm volatile("prefetch.tensormap [%0];" :: "l"((u64)&tmap) : "memory");
      for (int s = 0; s < S; s++) { mbar_init(sa(&full[s]), 1); mbar_init(sa(&empty[s]), 1); }
      for (int b = 0; b < G::NBUF; b++) { mbar_init(sa(&accf[b]), 1); mbar_init(sa(&acce[b]), 128); }
      mbar_init(sa(bready), 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      info[0] = d; info[1] = v & 0x7FFF; info[4] = U0; info[5] = U1;
      // the other warps wait for this setup (and the TMEM allocation) on barrier 2; the producer does not
      asm volatile("barrier.arrive 2, %0;" :: "n"(NTHREADS) : "memory");
      const int rl = p.rpc_log2, rm = (1 << rl) - 1;
      u64 pol, t_first = 0;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
      uint32_t k = 0;
      Seg it(U0, U1); int j, qa, qb;
      while (it.next(j, qa, qb)) {
        // domain d's j-th row block: the weight's chunks (2^rpc_log2 row blocks each) alternate between the domains
        const int rb = ((((j >> rl) << 1) + d) << rl) + (j & rm);
        for (int s = 2 * qa; s < 2 * qb; s++, k++) {
          const int slot = (int)(k % S);
          if (k >= (uint32_t)S) mbar_wait(sa(&empty[slot]), ((k / S) - 1) & 1);
          const uint32_t fb = sa(&full[slot]), dst = sa(sA) + slot * A_ST;
          mbar_expect(fb, A_ST + SF_ST);
          tma_2d(dst, &tmap, s * KSB * 128, rb * 128, fb, pol);
          tma_2d(dst + A_BLK, &tmap, (s * KSB + 1) * 128, rb * 128, fb, pol);
          bulk_g2s(sa(sSFA) + slot * SF_ST, p.sfa[d] + (size_t)rb * (KB * SF_ATOM) + (size_t)s * SF_ST, SF_ST, fb, pol);
          if (p.prof && k == 0) t_first = gtime();
        }
      }
      if (p.prof) {
        u64* pr = p.prof + (size_t)blockIdx.x * 16;
        pr[0] = t_start; pr[1] = t_first; pr[6] = sid; pr[7] = (u64)v;
      }
    } else {
      asm volatile("barrier.arrive 2, %0;" :: "n"(NTHREADS) : "memory");
    }
  } else {
    if (warp == 1) {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(sa((const void*)&info[3])), "r"(TMEM_COLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    tc_fence_before();
    named_bar(2, NTHREADS);
    tc_fence_after();
    const int d = info[0], r = info[1], U0 = info[4], U1 = info[5];
    const uint32_t tmem = (uint32_t)info[3];
    const int rl = p.rpc_log2, rm = (1 << rl) - 1;
    u64* prof = p.prof ? p.prof + (size_t)blockIdx.x * 16 : nullptr;
    if (warp == 1) {
      // -------------------------------------------------------------- MMA issuer
      if (lane == 0) {
        mbar_wait(sa(bready), 0);
        fence_proxy_async();
        tc_fence_after();
        u64 t_full = 0, t_last = 0;
        const u64 fb0 = sdesc_sf(sa(sSFB)), fa0 = sdesc_sf(sa(sSFA));
        const u64 ad0 = sdesc_sw128(sa(sA)), bd0 = sdesc_sw128(sa(sB));
#pragma unroll 1
        for (int kb = 0; kb < KB; kb++) tc_cp_sf(tmem + G::SFB_COL + 4 * kb, fb0 + (u64)((kb * SF_ATOM) >> 4));
        const uint32_t id0 = idesc_mx(NT, 0, 0), id2 = idesc_mx(NT, 2, 2);
        uint32_t k = 0; int i = 0;
        Seg it(U0, U1); int j, qa, qb;
        while (it.next(j, qa, qb)) {
          const int b = i & 1;
          if (i >= G::NBUF) mbar_wait(sa(&acce[b]), ((i >> 1) - 1) & 1);
          tc_fence_after();
          for (int s = 2 * qa; s < 2 * qb; s++, k++) {
            const int slot = (int)(k % S);
            mbar_wait(sa(&full[slot]), (k / S) & 1);
            if (prof) { t_last = gtime(); if (k == 0) t_full = t_last; }
            tc_fence_after();
            // quarter s / 2 of the row block accumulates into its own accumulator (a fixed decomposition)
            const uint32_t dacc = tmem + G::ACC_COL + (b * 4 + (s >> 1)) * NT;
            const uint32_t sfa_col = tmem + G::SFA_COL + (k & 1) * 8;
#pragma unroll
            for (int g = 0; g < KSB; g++) {
              const int kb = s * KSB + g;
              tc_cp_sf(sfa_col + 4 * g, fa0 + (u64)((slot * SF_ST + g * SF_ATOM) >> 4));
              const u64 ad = ad0 + (u64)((slot * A_ST + g * A_BLK) >> 4), bd = bd0 + (u64)((kb * G::B_KB) >> 4);
              const uint32_t sfb = tmem + G::SFB_COL + 4 * kb;
              tc_mma(dacc, ad, bd, id0, sfa_col + 4 * g, sfb, ((s & 1) | g) != 0 ? 1u : 0u);
              tc_mma(dacc, ad + 4, bd + 4, id2, (sfa_col + 4 * g) | (2u << 30), sfb | (2u << 30), 1u);
            }
            tc_commit(sa(&empty[slot]));
          }
          tc_commit(sa(&accf[b]));
          i++;
        }
        if (prof) { prof[3] = t_full; prof[4] = t_last; }
      }
    } else {
      // -------------------------------------------------------------- x quantization, then epilogue (TMEM lane quarter = warp % 4)
      const int q = warp & 3, row = q * 32 + lane, et = tid - 64;
      griddep_wait();
      quantize_x<NT>(p, sB, sSFB, et);
      fence_proxy_async();
      named_bar(1, 128);
      if (et == 0) mbar_arrive(sa(bready));
      if (prof && et == 0) prof[2] = gtime();
      int i = 0;
      Seg it(U0, U1); int j, qa, qb;
      while (it.next(j, qa, qb)) {
        const int b = i & 1;
        mbar_wait(sa(&accf[b]), (i >> 1) & 1);
        tc_fence_after();
        float v[4][NT];
#pragma unroll
        for (int h = 0; h < 4; h++)
          if (h >= qa && h < qb)
#pragma unroll
            for (int c = 0; c < NT; c += 16) tmem_ld16(tmem + ((uint32_t)(q * 32) << 16) + G::ACC_COL + (b * 4 + h) * NT + c, v[h] + c);
        tc_fence_before();
        mbar_arrive(sa(&acce[b]));
        i++;
        const size_t col = (size_t)(((((j >> rl) << 1) + d) << rl) + (j & rm)) * 128 + row;
        if (qa == 0 && qb == 4) {
#pragma unroll
          for (int t = 0; t < NT; t++)
            if (t < p.M) p.out[(size_t)t * p.ldo + col] = __float2bfloat16((v[0][t] + v[1][t]) + (v[2][t] + v[3][t]));
          continue;
        }
        // row block shared with the previous (qa > 0) or the next slot: quarters meet in the workspace, the CTA that
        // completes the four sums them in the fixed order
        const int bnd = qa > 0 ? r : r + 1;
        float* base = p.ws + ((size_t)d * p.wsslots + bnd) * 4 * p.wst * 128 + row;
#pragma unroll
        for (int h = 0; h < 4; h++)
          if (h >= qa && h < qb)
#pragma unroll
            for (int t = 0; t < NT; t++)
              if (t < p.M) base[((size_t)h * p.wst + t) * 128] = v[h][t];
        named_bar(1, 128);
        int* flag = p.flags + d * p.wsslots + bnd;
        if (et == 0) { __threadfence(); info[2] = atomicAdd(flag, qb - qa) + (qb - qa); __threadfence(); }
        named_bar(1, 128);
        if (info[2] == 4) {
#pragma unroll
          for (int h = 0; h < 4; h++)
            if (h < qa || h >= qb)
#pragma unroll
              for (int t = 0; t < NT; t++)
                if (t < p.M) v[h][t] = __ldcg(base + ((size_t)h * p.wst + t) * 128);
#pragma unroll
          for (int t = 0; t < NT; t++)
            if (t < p.M) p.out[(size_t)t * p.ldo + col] = __float2bfloat16((v[0][t] + v[1][t]) + (v[2][t] + v[3][t]));
          if (et == 0) *flag = 0;
        }
        named_bar(1, 128);
      }
      if (prof && et == 0) prof[5] = gtime();
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"((uint32_t)info[3]), "r"(TMEM_COLS) : "memory");
  }
}

// ------------------------------------------------------------------ host
torch::Tensor make_tmap(torch::Tensor w) {
  TORCH_CHECK(w.is_cuda() && w.dim() == 2 && w.is_contiguous() && w.element_size() == 1 && w.size(1) == K);
  CUtensorMap tm;
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)w.size(0)};
  cuuint64_t strides[1] = {(cuuint64_t)K};
  cuuint32_t box[2] = {128, 128}, es[2] = {1, 1};
  DRV(cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, w.data_ptr(), dims, strides, box, es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  auto t = torch::empty({(int64_t)sizeof(CUtensorMap)}, torch::kUInt8);
  std::memcpy(t.data_ptr(), &tm, sizeof(tm));
  return t;
}

template <int NT>
static void launch(const CUtensorMap& tm, const Params& p, int grid, cudaStream_t st, int dev) {
  static bool attr[64] = {};
  auto kern = k_mxhead<NT>;
  if (!attr[dev]) { RTC(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, Geo<NT>::SMEM)); attr[dev] = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(NTHREADS); cfg.dynamicSmemBytes = Geo<NT>::SMEM; cfg.stream = st;
  // cooperative: all CTAs co-resident, one per SM (the SMEM size allows one), so every SM's slot is served exactly
  // once whatever else runs on the GPU
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeCooperative;
  at[0].val.cooperative = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = (p.opts & 1) ? 2 : 1;
  RTC(cudaLaunchKernelEx(&cfg, kern, tm, p));
}

static CUtensorMap tm_of(const torch::Tensor& t) {
  CUtensorMap tm;
  std::memcpy(&tm, t.data_ptr(), sizeof(tm));
  return tm;
}

// plan (int32, CPU): slot[g], u0[g], u1[g] (per SM), then rpc_log2, wsslots
void mx_head(torch::Tensor x, torch::Tensor out, torch::Tensor tmap, torch::Tensor sfa0, torch::Tensor sfa1, torch::Tensor plan,
             torch::Tensor ws, int64_t wst, torch::Tensor flags, int64_t opts, torch::Tensor prof) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == K && x.stride(1) == 1 && x.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(x.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, "x rows must be 16-B aligned");
  const int M = (int)x.size(0);
  TORCH_CHECK(M >= 1 && M <= 32 && M <= wst);
  TORCH_CHECK(out.dim() == 2 && out.stride(1) == 1 && out.scalar_type() == torch::kBFloat16 && out.size(0) == M);
  TORCH_CHECK(tmap.numel() == (int64_t)sizeof(CUtensorMap) && !tmap.is_cuda());
  TORCH_CHECK(!plan.is_cuda() && plan.scalar_type() == torch::kInt32 && (plan.numel() - 2) % 3 == 0);
  const int grid = (int)((plan.numel() - 2) / 3), dev = x.get_device();
  TORCH_CHECK(grid >= 2 && grid <= 256);
  const int* pl = plan.data_ptr<int>();
  Params p;
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.ldx = x.stride(0); p.M = M; p.wst = (int)wst; p.opts = (int)opts;
  p.sfa[0] = (const unsigned char*)sfa0.data_ptr(); p.sfa[1] = (const unsigned char*)sfa1.data_ptr();
  p.out = (__nv_bfloat16*)out.data_ptr(); p.ldo = out.stride(0);
  std::memset(p.slot, 0, sizeof(p.slot)); std::memset(p.u0, 0, sizeof(p.u0)); std::memset(p.u1, 0, sizeof(p.u1));
  std::memcpy(p.slot, pl, sizeof(int) * grid);
  std::memcpy(p.u0, pl + grid, sizeof(int) * grid);
  std::memcpy(p.u1, pl + 2 * grid, sizeof(int) * grid);
  p.rpc_log2 = pl[3 * grid]; p.wsslots = pl[3 * grid + 1];
  int64_t units = 0;
  for (int g = 0; g < grid; g++) {
    TORCH_CHECK(p.u1[g] - p.u0[g] >= 12 && (p.slot[g] & 0x7FFF) + 1 < p.wsslots, "slot range below 3 row blocks");
    units += p.u1[g] - p.u0[g];
  }
  TORCH_CHECK(out.size(1) * 4 >= 128 * units, "out too narrow");
  TORCH_CHECK(ws.numel() >= (int64_t)2 * p.wsslots * 4 * wst * 128);
  TORCH_CHECK(flags.is_cuda() && flags.scalar_type() == torch::kInt32 && flags.numel() >= 2 * p.wsslots);
  p.ws = ws.data_ptr<float>(); p.flags = flags.data_ptr<int>();
  p.prof = (prof.is_cuda() && prof.scalar_type() == torch::kInt64 && prof.numel() >= 16 * grid) ? (u64*)prof.data_ptr() : nullptr;
  c10::cuda::CUDAGuard guard(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  if (M <= 16) launch<16>(tm_of(tmap), p, grid, st, dev);
  else launch<32>(tm_of(tmap), p, grid, st, dev);
}

}  // namespace mxg

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_tmap", &mxg::make_tmap);
  m.def("mx_head", &mxg::mx_head);
}
"""
