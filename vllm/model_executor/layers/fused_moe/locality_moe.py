# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Locality-domain ("per-die") MXFP8 MoE for decode on Rubin-class GPUs.

``VLLM_MOE_LOCALITY_KERNEL=1`` (default off) replaces the router GEMM and the
trtllm-gen MXFP8 MoE call (routing + FC1 + FC2, deferred finalize) of decode
batches with ONE persistent kernel, ``k_moe``, whose CTAs read only expert
weights that live in their own locality domain's HBM:

- Placement (:func:`place_expert_pairs`): each expert's w13 (``[1024, 2048]``
  e4m3 = exactly one 2 MiB chunk) and each expert pair's w2 (2 x 1 MiB) are
  moved, in place of the original tensors and in the same trtllm-gen MajorK
  shuffled layout, to locality domain ``(e >> 1) & 1``. The trtllm-gen path
  (mixed / prefill steps) keeps reading the same tensors.
- Fused prologue (no separate routing kernels, no grid barrier):
  1. router GEMM ``x_bf16 @ w_router^T``: split-K tcgen05 bf16 MMAs. Work item
     (128-expert half, K slice, token tile) per CTA; the w_router slice is
     loaded before ``griddepcontrol.wait``. fp32 partials go to a scratch
     buffer and a per-tile counter is released.
  2. per token (one warp): the K-slice partials are summed in a fixed order and
     rounded to bf16 (the logits); softmax, top-8 and renormalization replicate
     FlashInfer's RenormalizeNaive block-per-token routing kernel bit for bit
     (``-use_fast_math`` arithmetic, CUB block-reduce order, lower expert id
     wins ties). The token is appended to its experts' lists (atomics) and a
     per-tile "routed" counter is released.
  3. every CTA waits for the routed counters, reads the 128 per-expert counts
     of its domain and derives its own static schedule (no plan kernel).
- MoE body: warp 0 streams weight tiles with SW128 tensor-map TMA (+ 1-D bulk
  copies of the 128x4-interleaved scale atoms) into a 5 x 32 KB ring (the
  first tiles of the CTA's static "wave-0" FC1 unit before
  ``griddepcontrol.wait``); warps 1-4 gather token rows (``cp.async``);
  warp 5 issues block-scaled tcgen05 MMAs (M = 128 weight rows, N <= 32
  tokens); warps 6-9 run the epilogue (FC1: SwiGLU + MXFP8 requant into an
  intermediate in the domain's memory; FC2: bf16 rows at ``t * 8 + k``). A CTA
  takes the domain of its SM (``%smid`` -> ``Topology.sm_domain``) and a rank
  in that domain's static round-robin schedule (FC1 units, then FC2 units;
  FC2 waits on a per-group FC1 counter).
- Output: :class:`UnfinalizedMoEOutput` with ``gemm2_permuted[T * 8, H]``,
  ``expert_weights[T, 8]`` (bf16) and the constant
  ``expanded_idx_to_permuted_idx = arange(T * 8)`` (the DLC-8 deferred-finalize
  contract).

Shapes are fixed to Qwen3.6-35B-A3B: E = 256, top-8, H = 2048, I = 512, MXFP8
(1x32 UE8M0). Decode batches of 17 ... 512 tokens (smaller batches use
FlashInfer's warp-per-token routing arithmetic, which this kernel does not
replicate).

This module imports only torch at import time, so microbenchmarks can load it
by file path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os

import torch

E, TOPK, HID, INTER = 256, 8, 2048, 512
MIN_TOKENS, MAX_TOKENS = 17, 512
MAX_SLOTS = MAX_TOKENS * TOPK + 7 * E  # per-expert slot ranges are 8-aligned
CHUNK = 2 << 20
SK_MAX = 32  # router K slices

# VLLM_MOE_LOCALITY_KERNEL=1: place MXFP8 trtllm-gen expert weights by expert
# pair and serve deferred-finalize decode calls with MIN_TOKENS <= T <=
# MAX_TOKENS by this kernel, router GEMM included (acts inside the MoE custom
# op: no compile-key change).
ENABLED = os.environ.get("VLLM_MOE_LOCALITY_KERNEL", "0") == "1"
GATE_MIN_TOKENS = max(
    MIN_TOKENS, int(os.environ.get("VLLM_MOE_LOCALITY_KERNEL_MIN_TOKENS", "17"))
)
GATE_MAX_TOKENS = min(
    MAX_TOKENS, int(os.environ.get("VLLM_MOE_LOCALITY_KERNEL_MAX_TOKENS", "512"))
)

_ext: list = []


def _cache_root() -> str:
    try:
        from vllm import envs

        return envs.VLLM_CACHE_ROOT
    except Exception:  # standalone use (microbenchmarks)
        return os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))


def load():
    """Build (or load the cached build of) the extension for the current device."""
    if _ext:
        return _ext[0]
    major, minor = torch.cuda.get_device_capability()
    import torch.utils.cpp_extension as cpp

    arch = f"{major}{minor}{'a' if major >= 9 else ''}"
    bdir = os.path.join(_cache_root(), "locality_moe", f"sm{arch}")
    os.makedirs(bdir, exist_ok=True)
    digest = hashlib.sha256(_SOURCE.encode()).hexdigest()[:16]
    src = os.path.join(bdir, f"locality_moe_{digest}.cu")
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
            name=f"_locality_moe_{digest}",
            sources=[src],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"],
            extra_cflags=["-O3", "-std=c++20"],
            extra_ldflags=["-lcuda"],
            build_directory=bdir,
            verbose=False,
        )
    finally:
        cpp._get_cuda_arch_flags = orig
    _ext.append(ext)
    return ext


# ---------------------------------------------------------------- placement
def expert_pair_plans(num_experts: int = E) -> tuple[list[int], list[int]]:
    """2 MiB chunk -> domain for w13 (one expert per chunk) and w2 (two)."""
    return [(i >> 1) & 1 for i in range(num_experts)], [
        i & 1 for i in range(num_experts // 2)
    ]


def place_expert_pairs(w13: torch.Tensor, w2: torch.Tensor):
    """Copies of w13 [E, 1024, 2048] / w2 [E, 2048, 512] (e4m3) in localized
    memory, expert pair p = e >> 1 on domain p & 1; same layout and dtype.
    """
    from vllm.model_executor.layers.locality.memory import alloc_chunks, chunk_ordinals

    assert w13.shape == (E, 2 * INTER, HID) and w2.shape == (E, HID, INTER)
    assert w13.is_contiguous() and w2.is_contiguous()
    dev = w13.device.index
    p13, p2 = expert_pair_plans()
    out = []
    for w, plan in ((w13, p13), (w2, p2)):
        nbytes = w.numel() * w.element_size()
        assert nbytes == len(plan) * CHUNK
        flat = alloc_chunks(nbytes, plan, CHUNK, dev)
        got = chunk_ordinals(flat, nbytes, CHUNK)
        if got != plan:
            raise RuntimeError(f"locality placement mismatch: {got[:8]} vs {plan[:8]}")
        flat.copy_(w.reshape(-1).view(torch.uint8))
        out.append(flat.view(w.dtype).view(w.shape))
    return out[0], out[1]


@dataclasses.dataclass
class LayerWeights:
    w13: torch.Tensor
    w2: torch.Tensor
    s13: torch.Tensor  # w13 scales (production 128x4-interleaved tensors)
    s2: torch.Tensor
    tmaps: torch.Tensor  # host bytes of the w13 / w2 tensor maps
    # router weight data_ptr -> host bytes of the three tensor maps (w13, w2, w_router)
    rmaps: dict = dataclasses.field(default_factory=dict)


def router_split(T: int) -> tuple[int, int]:
    """(K slices, tokens per tile) of the split-K router GEMM: every work item
    (expert half, K slice, token tile) fits the 48 KB SMEM operand window and
    the item count stays <= 192 (one item per CTA).
    """
    return (16, 64) if T <= 384 else (32, 192)


# ---------------------------------------------------------------- runtime
class LocalityMoE:
    """Per-device state of the locality MoE kernel (topology, scratch)."""

    def __init__(self, device: int):
        from vllm.model_executor.layers.locality.memory import alloc_chunks
        from vllm.model_executor.layers.locality.topology import get_topology

        topo = get_topology(device)
        if topo is None or topo.num_domains != 2:
            raise RuntimeError("locality MoE needs a GPU with two locality domains")
        self.device = device
        self.topo = topo
        self.nsm = topo.num_sms
        self.nd = (int(topo.domain_sms[0]), int(topo.domain_sms[1]))
        dev = torch.device("cuda", device)
        self.ext = load()
        self.ext.init(device)
        # counters (zero between calls: the last CTA of every call resets them)
        self.state = torch.zeros(self.ext.state_bytes(), dtype=torch.uint8, device=dev)
        self.lists = torch.zeros(E * MAX_TOKENS, dtype=torch.int16, device=dev)
        self.part = torch.empty(
            SK_MAX * MAX_TOKENS * E, dtype=torch.float32, device=dev
        )
        # FC1 -> FC2 intermediate: one slot space per domain, on that domain
        # (FC1 and FC2 of a group run in the same domain)
        ib = -(-MAX_SLOTS * INTER // CHUNK) * CHUNK
        sb = -(-MAX_SLOTS * (INTER // 32) // CHUNK) * CHUNK
        self.inter = alloc_chunks(
            2 * ib, [0] * (ib // CHUNK) + [1] * (ib // CHUNK), CHUNK, device
        ).view(2, ib)
        self.inter_sf = alloc_chunks(
            2 * sb, [0] * (sb // CHUNK) + [1] * (sb // CHUNK), CHUNK, device
        ).view(2, sb)
        self.idx = torch.arange(MAX_TOKENS * TOPK, dtype=torch.int32, device=dev)
        self.smdom = topo.sm_domain
        # CTAs (ranks) per domain; SMs beyond them stay free for concurrent kernels
        self.ranks = self.nd
        self.prio = 0  # launch priority attribute (0: the stream's)

    def set_launch(self, reserve_sms: int = 0, prio: int = 0) -> None:
        """Leave ``reserve_sms`` SMs (split across the domains) to concurrent
        kernels and launch with priority ``prio`` (0: the stream's).
        """
        r0 = reserve_sms // 2
        self.ranks = (self.nd[0] - r0, self.nd[1] - (reserve_sms - r0))
        self.prio = prio

    def layer_weights(
        self, w13: torch.Tensor, w2: torch.Tensor, s13: torch.Tensor, s2: torch.Tensor
    ) -> LayerWeights:
        """Kernel view of one layer: w13/w2 (as placed) and their scales."""
        return LayerWeights(w13, w2, s13, s2, self.ext.tensor_maps(w13, w2))

    def _maps(self, lw: LayerWeights, w_router: torch.Tensor) -> torch.Tensor:
        m = lw.rmaps.get(w_router.data_ptr())
        if m is None:
            assert w_router.shape == (E, HID) and w_router.dtype == torch.bfloat16
            assert w_router.is_contiguous()
            m = lw.rmaps[w_router.data_ptr()] = torch.cat(
                [lw.tmaps, self.ext.router_map(w_router)]
            )
        return m

    def forward(
        self,
        x_bf16: torch.Tensor,
        w_router: torch.Tensor,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        lw: LayerWeights,
        dbg: torch.Tensor | None = None,
        pdl: bool = True,
        logits: torch.Tensor | None = None,
    ):
        """Returns (gemm2_permuted [T*8, H] bf16, expert_weights [T, 8] bf16,
        expanded_idx_to_permuted_idx [T, 8] int32, topk_ids [T, 8] int32).
        ``dbg`` (int64 [nsm, 32]) receives per-CTA %globaltimer stamps;
        ``logits`` (bf16 [T, 256]) receives the router logits (tests);
        ``pdl=False`` launches without programmatic dependent launch.
        """
        T = x.shape[0]
        assert MIN_TOKENS <= T <= MAX_TOKENS
        sk, nt = router_split(T)
        dev = x.device
        ew = torch.empty(T, TOPK, dtype=torch.bfloat16, device=dev)
        ids = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
        out = torch.empty(T * TOPK, HID, dtype=torch.bfloat16, device=dev)
        self.ext.moe_launch(
            self._maps(lw, w_router),
            lw.s13,
            lw.s2,
            x_bf16,
            x,
            x_sf.view(torch.uint8),
            self.inter,
            self.inter_sf,
            out,
            ew,
            ids,
            self.state,
            self.lists,
            self.part,
            self.smdom,
            self.ranks[0],
            self.ranks[1],
            self.nsm,
            sk,
            nt,
            dbg,
            logits,
            pdl,
            self.prio,
        )
        return out, ew, self.idx[: T * TOPK].view(T, TOPK), ids


_RUNTIME: dict[int, LocalityMoE] = {}


def runtime(device: int) -> LocalityMoE:
    rt = _RUNTIME.get(device)
    if rt is None:
        rt = _RUNTIME[device] = LocalityMoE(device)
    return rt


# (w13 data_ptr, w2 data_ptr) of pair-placed layers -> their kernel view
_LAYERS: dict[tuple[int, int], LayerWeights] = {}


def maybe_place(
    w13: torch.Tensor, w2: torch.Tensor, s13: torch.Tensor, s2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load-time hook (trtllm-gen MXFP8 layout): pair-placed copies of w13/w2,
    which replace the originals, when the kernel is enabled and applies; else
    the inputs unchanged.
    """
    if not ENABLED or w13.shape != (E, 2 * INTER, HID) or w2.shape != (E, HID, INTER):
        return w13, w2
    from vllm.model_executor.layers.locality.topology import get_topology

    topo = get_topology(w13.device.index)
    if topo is None or topo.num_domains != 2:
        return w13, w2
    rt = runtime(w13.device.index)
    l13, l2 = place_expert_pairs(w13.contiguous(), w2.contiguous())
    _LAYERS[(l13.data_ptr(), l2.data_ptr())] = rt.layer_weights(l13, l2, s13, s2)
    return l13, l2


def try_apply(
    x_bf16: torch.Tensor,
    w_router: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
):
    """(gemm2_permuted, expert_weights, expanded_idx_to_permuted_idx) of the
    whole routed MoE (router GEMM included), or None when this call is not
    served (T outside the gate, layer not placed, or input layouts other than
    bf16 [T, H] router input + [T, H] e4m3 + linear [T, H/32] UE8M0).
    """
    T = x.shape[0]
    if not GATE_MIN_TOKENS <= T <= GATE_MAX_TOKENS:
        return None
    lw = _LAYERS.get((w13.data_ptr(), w2.data_ptr()))
    if lw is None:
        return None
    if (
        x.dim() != 2
        or x.shape[1] != HID
        or not x.is_contiguous()
        or x.dtype != torch.float8_e4m3fn
        or x_sf.numel() != T * (HID // 32)
        or not x_sf.is_contiguous()
        or x_bf16.shape != (T, HID)
        or x_bf16.dtype != torch.bfloat16
        or not x_bf16.is_contiguous()
        or w_router.shape != (E, HID)
        or w_router.dtype != torch.bfloat16
        or not w_router.is_contiguous()
    ):
        return None
    out, ew, idx, _ = runtime(x.device.index).forward(x_bf16, w_router, x, x_sf, lw)
    return out, ew, idx


_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/warp/warp_reduce.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstring>

namespace locmoe {
namespace cg = cooperative_groups;

typedef unsigned long long u64;
constexpr int E = 256, H = 2048, I = 512, TOPK = 8, TMAX = 512;
constexpr int KB_BYTES = 16384;                 // one K128 block of a 128-row tile (128 rows x 128 B, SW128)
constexpr int SF_ATOM = 512;                    // 128 rows x 4 UE8M0 (one 128x4-interleaved atom = UTCCP 32x4x4)
constexpr int KSB = 2, NMAX = 32, S = 5;        // 2 K128 blocks (32 KB of weights) per stage, 32-token B slots
constexpr int A_ST = KSB * KB_BYTES, B_ST = KSB * NMAX * 128, SF_ST = KSB * SF_ATOM;
constexpr int FC1_NKB = H / 128, FC2_NKB = I / 128, FC1_NRB = 2 * I / 128;   // K128 blocks; FC1 row blocks per expert
constexpr int UI = 6;                           // unit-info ring depth (published two units ahead)
constexpr int SCHED_MAX = 128;
constexpr int GMAX = 256;                       // groups (expert, 32-token sub-block) per domain
constexpr int NTHREADS = 320;
constexpr int TMEM_COLS = 512, ACCW = 32, NACC = 4;   // 4 MoE accumulators of N <= 32 columns, then SF columns
constexpr int RCOL = 256;                       // router accumulator: TMEM columns [256, 256 + nt)
constexpr int RX_EXTRA = 8192;                  // SMEM after the B slots: router operands (48 KB window), then the schedule
constexpr int NTILE = 8;                        // router token tiles (T <= 512)
constexpr int W13SF_E = 2 * I * H / 32, W2SF_E = H * I / 32;  // scale bytes per expert (128x4-interleaved)

// One (expert, sub-block of <= 32 tokens) of a domain: built by the producer, token entries (t * 8 + k) bulk-copied.
struct alignas(16) GRec { int e, sub, ntok, slot0; short j[32]; int gid, pad0, pad1, pad2; };   // 96 B
struct alignas(16) Info { GRec r; int kind, rb, nkb, kb0; int pad[4]; };                       // 128 B
// Cross-CTA state of one call; all zero between calls (the last CTA resets it).
struct State {
  unsigned rank[2], done, pad[29];
  unsigned q2[2 * 32];                // FC2 work queue head per domain (one 128-B line each)
  unsigned ctr[NTILE * 32];           // router partials stored, per token tile (one line each)
  unsigned tdone[NTILE * 32];         // tokens routed, per token tile
  unsigned cnt[E];                    // tokens per expert
  unsigned fc1done[2 * GMAX];         // FC1 units done, per group
};
constexpr int STATE_WORDS = (int)(sizeof(State) / 4);
struct Params {
  const uint8_t* w13sf; const uint8_t* w2sf;
  const __nv_bfloat16* xb;            // router input [T, H] bf16
  const uint8_t* x; const uint8_t* xsf;   // MXFP8 activation [T, H] e4m3, linear [T, H / 32] UE8M0
  uint8_t* interd[2]; uint8_t* intersfd[2];   // FC1 -> FC2 intermediate of domain d's groups
  __nv_bfloat16* out; __nv_bfloat16* ew; int* ids; __nv_bfloat16* logits;
  State* st; short* list; float* part;
  const signed char* smdom;
  int nd0, nd1, T, sk, nt, ntiles, nitems;
  u64* dbg;                           // optional per-CTA stamps [grid][DBGW]
};

#define RTC(x) do { cudaError_t e_ = (x); TORCH_CHECK(e_ == cudaSuccess, #x " failed: ", cudaGetErrorString(e_)); } while (0)
#define DRV(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorName(r_, &s_); TORCH_CHECK(false, #x " failed: ", s_); } } while (0)
// diagnostics: accumulate the cycles spent in `stmt` (a wait) into `acc`; dumped to Params::dbg when it is set
#define TW(acc, stmt) do { const long long t0_ = clock64(); stmt; acc += clock64() - t0_; } while (0)
constexpr int DBGW = 32;              // u64 per CTA in Params::dbg

// ------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ unsigned get_smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ u64 gtime() { u64 r; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); return r; }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mbar_expect(uint32_t a, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(tx) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t tx) { asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(tx) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t ph) {
  asm volatile("{\n\t.reg .pred P;\nLW%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t@!P bra LW%=;\n}" :: "r"(a), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"((u64)tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory"); }
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) { asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(src) : "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ void cp_arrive_noinc(uint32_t bar) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned ld_relaxed(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void red_release(unsigned* p, unsigned v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_commit(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void tc_cp_sf(uint32_t taddr, u64 sdesc) { asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(sdesc) : "memory"); }
__device__ __forceinline__ void tc_mma(uint32_t d, u64 ad, u64 bd, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}\n"
               :: "r"(d), "l"(ad), "l"(bd), "r"(idesc), "r"(acc), "r"(sfa), "r"(sfb) : "memory");
}
__device__ __forceinline__ void tc_mma_bf16(uint32_t d, u64 ad, u64 bd, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
               :: "r"(d), "l"(ad), "l"(bd), "r"(idesc), "r"(acc) : "memory");
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
__device__ __forceinline__ u64 sdesc_sf(uint32_t saddr) {         // UTCCP source: 32 rows x 16 B, no swizzle, SBO 128 B
  return (u64)((saddr >> 4) & 0x7FFF) | ((u64)(128 >> 4) << 32) | ((u64)1 << 46);
}
// SM107 kind::mxf8f6f4 instruction descriptor: E4M3 x E4M3, K-major both, UE8M0 scales, M = 128, dense K = 64
__device__ __forceinline__ uint32_t idesc_mx(int n, uint32_t sfa_id, uint32_t sfb_id) {
  return (sfb_id << 4) | ((uint32_t)(n >> 3) << 17) | (1u << 23) | (1u << 27) | (sfa_id << 29) | (1u << 31);
}
// kind::f16 instruction descriptor: BF16 x BF16 -> F32, K-major both, M = 128, K = 16 per instruction
__device__ __forceinline__ uint32_t idesc_bf16(int n) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(n >> 3) << 17) | (1u << 27);
}
__device__ __forceinline__ int warp_iscan(int v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
  return v;
}
// domain-local expert index k (0..127) of domain d -> expert id: pairs {0,1} -> d0, {2,3} -> d1, {4,5} -> d0, ...
__host__ __device__ __forceinline__ int expert_of(int d, int k) { return 4 * (k >> 1) + 2 * d + (k & 1); }
// largest k in [0, 128) with pre[k] <= v (pre non-decreasing, pre[128] > v)
__device__ __forceinline__ int find_pre(const int* pre, int v) {
  int lo = 0, hi = 127;
  while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (pre[mid] <= v) lo = mid; else hi = mid - 1; }
  return lo;
}

// ------------------------------------------------------------------ routing (one warp per token)
// FlashInfer 0.6.18 RenormalizeNaive, block-per-token kernel (routingIndicesBlockScoresKernel; used for E = 256 and
// T >= 17): SoftmaxPreprocess::applyToSmem (256 threads, thread e = expert e, cub::BlockReduce<float, 256>), the
// packed-key top-K (value descending, lower expert id first) and SumNormalizePostprocess; FlashInfer builds with
// -use_fast_math, so expf / 1/x / division are the approximate .ftz forms. Lane L holds experts 32 i + L, i.e.
// "virtual warp" i of that block, so the CUB warp reductions see the same operands in the same lanes.
__device__ __forceinline__ float f_sub_ftz(float a, float b) { float r; asm("sub.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_mul_ftz(float a, float b) { float r; asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_ex2_ftz(float a) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_rcp_ftz(float a) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_div_ftz(float a, float b) { float r; asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_max_ftz(float a, float b) { float r; asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ u64 tk_key(float v, int idx) {          // TopKRedType<float>::makeCmpVal
  unsigned b = __float_as_uint(v);
  b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);                 // cub TwiddleIn
  return ((u64)b << 32) | (unsigned)(65535 - idx);
}
__device__ __forceinline__ float tk_val(u64 k) {                   // TwiddleOut
  unsigned b = (unsigned)(k >> 32);
  b = (b & 0x80000000u) ? (b & 0x7fffffffu) : ~b;
  return __uint_as_float(b);
}
__device__ __forceinline__ u64 shfl_xor64(u64 v, int o) {
  const unsigned lo = __shfl_xor_sync(0xffffffffu, (unsigned)v, o), hi = __shfl_xor_sync(0xffffffffu, (unsigned)(v >> 32), o);
  return ((u64)hi << 32) | lo;
}

// qs (diagnostics, first token of a warp): [15] partials loaded, [16] routing math done, [17] list slots returned,
// [18] token released
__device__ void route_token(const Params& p, int t, int n, int lane, cub::WarpReduce<float>::TempStorage& wtmp, u64* qs) {
  const int T = p.T;
  // logits: fixed-order sum (s = 0, 1, ...) of the K-slice partials, one round to bf16. 16 slices of loads are issued
  // before the adds (sk is 16 or 32).
  float sc[8];
  for (int s0 = 0; s0 < p.sk; s0 += 16) {
    float v[16][8];
#pragma unroll
    for (int s = 0; s < 16; s++)
#pragma unroll
      for (int i = 0; i < 8; i++) v[s][i] = __ldcg(p.part + ((size_t)(s0 + s) * T + t) * E + i * 32 + lane);
#pragma unroll
    for (int s = 0; s < 16; s++)
#pragma unroll
      for (int i = 0; i < 8; i++) sc[i] = (s0 | s) == 0 ? v[s][i] : sc[i] + v[s][i];
  }
  if (qs && lane == 0) qs[15] = gtime() + (sc[0] != sc[0]);   // (data dependence: after the loads)
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const __nv_bfloat16 b = __float2bfloat16_rn(sc[i]);
    if (p.logits) p.logits[(size_t)t * E + i * 32 + lane] = b;
    sc[i] = __bfloat162float(b);
  }
  // softmax (SoftmaxPreprocess::applyToSmem)
  float mx = -INFINITY;
#pragma unroll
  for (int i = 0; i < 8; i++) mx = fmaxf(sc[i], mx);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float bsum = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    sc[i] = f_ex2_ftz(f_mul_ftz(f_sub_ftz(sc[i], mx), 1.4426950408889634f));
    const float a = cub::WarpReduce<float>(wtmp).Sum(sc[i]);     // lane 0: CUB warp aggregate of block warp i
    bsum = i == 0 ? a : bsum + a;                                 // ApplyWarpAggregates: in warp order
  }
  const float inv = f_rcp_ftz(__shfl_sync(0xffffffffu, bsum, 0));
  u64 cand[8];
#pragma unroll
  for (int i = 0; i < 8; i++) cand[i] = tk_key(f_mul_ftz(sc[i], inv), i * 32 + lane);
  // top-8 (reduceTopK: max packed key per round)
  u64 mine = 0;
#pragma unroll
  for (int r = 0; r < TOPK; r++) {
    u64 b = cand[0];
#pragma unroll
    for (int i = 1; i < 8; i++) b = cand[i] > b ? cand[i] : b;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { const u64 x = shfl_xor64(b, o); b = x > b ? x : b; }
#pragma unroll
    for (int i = 0; i < 8; i++) cand[i] = cand[i] == b ? 0ull : cand[i];
    mine = lane == r ? b : mine;
  }
  if (qs && lane == 0) qs[16] = gtime() + (mine == 1ull);
  // SumNormalizePostprocess (lane k holds the k-th score)
  const float v = lane < TOPK ? tk_val(mine) : 0.f;
  const float sum = cg::reduce(cg::tiled_partition<32>(cg::this_thread_block()), v, cg::plus<float>());
  if (lane < TOPK) {
    const float w = f_div_ftz(v, f_max_ftz(sum, 1e-20f));
    const int e = 65535 - (int)(mine & 0xffffu);
    p.ids[t * TOPK + lane] = e;
    p.ew[t * TOPK + lane] = __float2bfloat16_rn(w);
    const unsigned pos = atomicAdd(&p.st->cnt[e], 1u);
    p.list[e * TMAX + pos] = (short)(t * TOPK + lane);
    if (qs && lane == 0) qs[17] = gtime() + (pos > 4096u);
  }
  // the release orders this warp's list / weight stores (bar.warp.sync) before the routed count
  __syncwarp();
  if (lane == 0) { red_release(&p.st->tdone[n * 32], 1u); if (qs) qs[18] = gtime(); }
}

// ------------------------------------------------------------------ k_moe
__device__ __forceinline__ void issue_stage(uint32_t sA, uint32_t sSFA, uint32_t bar, const CUtensorMap* tm, const uint8_t* sf,
                                            int row0, int kb) {
  mbar_expect(bar, A_ST + SF_ST);
#pragma unroll
  for (int g = 0; g < KSB; g++) tma_2d(sA + g * KB_BYTES, tm, (kb + g) * 128, row0, bar);
  bulk_load(sSFA, sf + kb * SF_ATOM, SF_ST, bar);
}

__global__ void __launch_bounds__(NTHREADS, 1) k_moe(const __grid_constant__ CUtensorMap tm13, const __grid_constant__ CUtensorMap tm2,
                                                     const __grid_constant__ CUtensorMap tmr, const Params p) {
  extern __shared__ unsigned char smem_raw[];
  // align by pointer arithmetic on the __shared__ array (an integer cast would drop the address space)
  unsigned char* sm = smem_raw + ((1024u - ((uint32_t)__cvta_generic_to_shared(smem_raw) & 1023u)) & 1023u);
  unsigned char* sA = sm;
  unsigned char* sB = sA + S * A_ST;
  unsigned char* sRX = sB + S * B_ST;                    // router operands span sB + sRX; then the schedule arrays
  unsigned char* sSFA = sRX + RX_EXTRA;
  unsigned char* sSFB = sSFA + S * SF_ST;
  u64* bars = (u64*)(sSFB + S * SF_ST);
  u64* full = bars; u64* empty = bars + S; u64* accf = bars + 2 * S; u64* acce = accf + NACC; u64* inff = acce + NACC;
  u64* infe = inff + UI; u64* rfull = infe + UI; u64* racc = rfull + 1;
  Info* sinfo = (Info*)(racc + 1);
  float* sam = (float*)(sinfo + UI);                     // FC1 requant amax exchange [2 chunks][4 warps][2 parities][8 cols]
  volatile int* sdom = (volatile int*)(sam + 128);       // [0] served domain, [1] rank in it, [2] router item (-1: none)
  uint32_t* stmem = (uint32_t*)(sdom + 4);
  int* ssched = (int*)(stmem + 4);                       // this CTA's units (entry 0 = wave-0 unit)
  __nv_bfloat16* sstage = (__nv_bfloat16*)(ssched + SCHED_MAX + 4);   // FC2 epilogue: [16 tokens][128 ch] bf16
  __shared__ cub::WarpReduce<float>::TempStorage wtmp[4];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const u64 t_start = gtime();
  if (tid == 0) {
    // Rank in this SM's domain by arrival order (a CTA held back by another kernel takes the last ranks). If the
    // domain already has all its ranks (an SM ran a second CTA), take a rank of the other domain, so every rank is
    // served exactly once. Router work item 2 rank + domain: the early ranks of both domains. (Own state only:
    // allowed before the wait.)
    int d = p.smdom[get_smid()];
    d = d < 0 ? 0 : d;
    unsigned r = atomicAdd(&p.st->rank[d], 1u);
    if (r >= (unsigned)(d ? p.nd1 : p.nd0)) { d ^= 1; r = atomicAdd(&p.st->rank[d], 1u); }
    const int it = 2 * (int)r + d;
    sdom[0] = d; sdom[1] = (int)r; sdom[2] = it < p.nitems ? it : -1;
    for (int s = 0; s < S; s++) { mbar_init(sa(&full[s]), 1 + 128); mbar_init(sa(&empty[s]), 1); }
    for (int b = 0; b < NACC; b++) { mbar_init(sa(&accf[b]), 1); mbar_init(sa(&acce[b]), 4); }
    for (int i = 0; i < UI; i++) { mbar_init(sa(&inff[i]), 1); mbar_init(sa(&infe[i]), 3); }
    mbar_init(sa(rfull), 1 + 128); mbar_init(sa(racc), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 5) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(sa(stmem)), "r"(TMEM_COLS) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *stmem;
  const int d = sdom[0], rank = sdom[1], item = sdom[2];
  const int nd = d ? p.nd1 : p.nd0;
  const bool live = rank < nd;
  // router item: token tile rn, local index rj = 2 * K slice + expert half
  const bool hasr = item >= 0;
  const int rn = hasr ? item / (2 * p.sk) : 0, rj = hasr ? item % (2 * p.sk) : 0, rm = rj & 1, rs = rj >> 1;
  const int kc = H / p.sk, natom = kc / 64;              // K per slice (128 or 64) = natom SW128 atoms of 64 bf16
  const int r_t0 = rn * p.nt, r_nv = hasr ? min(p.nt, p.T - r_t0) : 0, r_np = (r_nv + 15) & ~15;
  unsigned char* sRw = sB;                               // w_router slice [128 rows][kc] (natom x 16 KB)
  unsigned char* sRx = sB + natom * 16384;               // x slice [nt rows][kc] (natom x nt x 128 B)
  u64* q_ = p.dbg ? p.dbg + (size_t)blockIdx.x * DBGW : nullptr;
  if (q_ && tid == 0) q_[9] = gtime();

  if (warp == 0) {
    // ---------------------------------------------------------------- producer
    // wave-0: static FC1 unit `rank` of the domain (expert k = rank / 8, row block rank % 8); its first S stages are
    // loaded before griddepcontrol.wait (weights are never written by a predecessor), as is the w_router slice.
    const int k0 = rank >> 3, rb0 = rank & 7, e0 = expert_of(d, k0);
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"((u64)&tm13) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"((u64)&tm2) : "memory");
      if (hasr) {
        asm volatile("prefetch.tensormap [%0];" :: "l"((u64)&tmr) : "memory");
        mbar_expect(sa(rfull), natom * 16384);
        for (int a = 0; a < natom; a++) tma_2d(sa(sRw + a * 16384), &tmr, rs * kc + a * 64, rm * 128, sa(rfull));
      }
      if (live)
        for (int s = 0; s < S; s++)
          issue_stage(sa(sA + s * A_ST), sa(sSFA + s * SF_ST), sa(&full[s]), &tm13,
                      p.w13sf + (size_t)e0 * W13SF_E + rb0 * FC1_NKB * SF_ATOM, e0 * 2 * I + rb0 * 128, KSB * s);
    }
    griddep_wait();
    if (q_ && lane == 0) q_[1] = gtime();
    // every token routed: poll the per-tile counters (one lane per tile)
    {
      const int need = lane < p.ntiles ? min(p.nt, p.T - lane * p.nt) : 0;
      for (;;) {
        const unsigned v = lane < p.ntiles ? ld_acquire(&p.st->tdone[lane * 32]) : 0u;
        if (__all_sync(0xffffffffu, (int)v >= need)) break;
        __nanosleep(32);
      }
      __syncwarp();
    }
    asm volatile("fence.proxy.async.global;" ::: "memory");   // token lists (generic stores, acquired) -> bulk copies
    if (q_ && lane == 0) q_[4] = gtime();
    // the wave-0 unit's token list, while the counts load (the slot's arrival follows with its other fields)
    if (lane == 0 && live) {
      mbar_expect_tx(sa(&inff[0]), 64u);
      bulk_load(sa(sinfo[0].r.j), p.list + (size_t)e0 * TMAX, 64u, sa(&inff[0]));
    }
    // schedule from the per-expert counts of this domain (lane: local experts 4 lane .. 4 lane + 3): wave-0 unit,
    // then the static round-robin share of the domain's FC1 list (expert order; sub-block 0 without its wave-0 row
    // blocks), then FC2 units (group order, 16 row blocks each) from the domain's work queue. Code: type << 24 |
    // k << 16 | sub << 8 | rb; type 1 FC1, 2 FC2, 3 wave-0 FC1, 4 idle wave-0.
    int* s_cnt = (int*)sRX; int* s_f0 = s_cnt + 128; int* s_g0 = s_f0 + 132; int* s_so = s_g0 + 132;
    int nstat = 0, ng = 0;
    if (live) {
      int c[4], f[4], g[4], o[4], sf_ = 0, sg_ = 0, so_ = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int k = 4 * lane + j;
        c[j] = (int)ld_relaxed(&p.st->cnt[expert_of(d, k)]);
        const int ns = (c[j] + 31) >> 5, sk0 = ns ? min(max(nd - 8 * k, 0), 8) : 0;
        f[j] = ns ? 8 * ns - sk0 : 0; g[j] = ns; o[j] = (c[j] + 7) & ~7;
        sf_ += f[j]; sg_ += g[j]; so_ += o[j];
      }
      if (q_ && lane == 0) q_[19] = gtime() + (c[0] < 0);
      int xf = warp_iscan(sf_) - sf_, xg = warp_iscan(sg_) - sg_, xo = warp_iscan(so_) - so_;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int k = 4 * lane + j;
        s_cnt[k] = c[j]; s_f0[k] = xf; s_g0[k] = xg; s_so[k] = xo;
        xf += f[j]; xg += g[j]; xo += o[j];
      }
      if (lane == 31) { s_f0[128] = xf; s_g0[128] = xg; s_so[128] = xo; }
      __syncwarp();
      const int nfc1 = s_f0[128];
      ng = s_g0[128];
      nstat = rank < nfc1 ? (nfc1 - rank + nd - 1) / nd : 0;
      nstat = nstat < SCHED_MAX - 1 ? nstat : SCHED_MAX - 1;
      for (int i = lane; i < nstat; i += 32) {
        const int m = rank + i * nd, k = find_pre(s_f0, m), off = m - s_f0[k];
        const int sk0 = min(max(nd - 8 * k, 0), 8), w0 = 8 - sk0;
        const int sub = off < w0 ? 0 : 1 + ((off - w0) >> 3), rb = off < w0 ? sk0 + off : ((off - w0) & 7);
        ssched[1 + i] = (1 << 24) | (k << 16) | (sub << 8) | rb;
      }
      if (lane == 0) ssched[0] = ((s_cnt[k0] > 0 ? 3 : 4) << 24) | (k0 << 16) | rb0;
      nstat += 1;
    }
    __syncwarp();
    if (q_ && lane == 0) q_[20] = gtime();
    if (lane == 0) {
      uint32_t k = live ? S : 0;
      int pub = 0, ndyn = 0;
      bool fin = false;
      const unsigned n2 = 16u * (unsigned)ng;              // FC2 entries of this domain
      unsigned* q2 = &p.st->q2[d * 32];
      // next FC2 entry: requested when the last static unit is published, then one publication ahead of its use
      unsigned nxt = n2;
      long long c_pinff = 0, c_pempty = 0;
      auto pub_next = [&]() {
        const int slot = pub % UI;
        if (pub >= UI) mbar_wait(sa(&infe[slot]), ((pub / UI) - 1) & 1);
        Info* f = &sinfo[slot];
        int c = -1;
        if (pub < nstat) {
          c = ssched[pub];
          if (pub == nstat - 1) nxt = atomicAdd(q2, 1u);
        } else if (nxt < n2) {
          const int gg = (int)(nxt >> 4), kk = find_pre(s_g0, gg);
          c = (2 << 24) | (kk << 16) | ((gg - s_g0[kk]) << 8) | (int)(nxt & 15);
          nxt = atomicAdd(q2, 1u);
          ndyn++;
        }
        if (c < 0) {
          f->kind = -1;
          mbar_arrive(sa(&inff[slot]));
          fin = true;
        } else {
          const int ty = c >> 24, kk = (c >> 16) & 0xff, sub = (c >> 8) & 0xff, rb = c & 0xff;
          const int e = expert_of(d, kk);
          f->r.e = e; f->r.sub = sub; f->rb = rb;
          if (ty == 4) {                                   // idle wave-0 unit: consume the prefetched stages
            f->kind = 0; f->nkb = KSB * S; f->kb0 = KSB * S; f->r.ntok = 0; f->r.slot0 = 0; f->r.gid = -1;
          } else {
            f->kind = ty == 2 ? 1 : 0; f->nkb = ty == 2 ? FC2_NKB : FC1_NKB; f->kb0 = ty == 3 ? KSB * S : 0;
            f->r.ntok = min(32, s_cnt[kk] - 32 * sub); f->r.slot0 = s_so[kk] + 32 * sub; f->r.gid = d * GMAX + s_g0[kk] + sub;
          }
          if (pub == 0) {
            mbar_arrive(sa(&inff[slot]));                  // its token list is already on the way (expect_tx above)
          } else if (ty == 4) {
            mbar_arrive(sa(&inff[slot]));
          } else {
            mbar_expect(sa(&inff[slot]), 64u);
            bulk_load(sa(f->r.j), p.list + (size_t)e * TMAX + 32 * sub, 64u, sa(&inff[slot]));
          }
        }
        pub++;
      };
      pub_next(); pub_next();
      if (q_) q_[8] = gtime();
      for (int uj = 0;; uj++) {
        const int slot = uj % UI;
        TW(c_pinff, mbar_wait(sa(&inff[slot]), (uj / UI) & 1));
        const Info* f = &sinfo[slot];
        const int kind = f->kind;
        if (kind < 0) break;
        const int e = f->r.e, rb = f->rb, nkb = f->nkb;
        const CUtensorMap* tm = kind ? &tm2 : &tm13;
        const uint8_t* sf = kind ? p.w2sf + (size_t)e * W2SF_E + rb * FC2_NKB * SF_ATOM : p.w13sf + (size_t)e * W13SF_E + rb * FC1_NKB * SF_ATOM;
        const int row0 = kind ? e * H + rb * 128 : e * 2 * I + rb * 128;
        for (int kb = f->kb0; kb < nkb; kb += KSB, k++) {
          const int s = k % S;
          if (k >= (uint32_t)S) TW(c_pempty, mbar_wait(sa(&empty[s]), ((k / S) - 1) & 1));
          issue_stage(sa(sA + s * A_ST), sa(sSFA + s * SF_ST), sa(&full[s]), tm, sf, row0, kb);
        }
        if (!fin) pub_next();
      }
      if (q_) { q_[25] = c_pinff; q_[26] = c_pempty; q_[30] = nstat; q_[31] = ndyn; }
    }
  } else if (warp <= 4) {
    // ---------------------------------------------------------------- gather (128 threads) + routing
    griddep_wait();
    const int gt = tid - 32;
    if (q_ && gt == 0) q_[10] = gtime();
    if (hasr) {
      // router x slice: tokens [r_t0, r_t0 + r_nv), columns [rs * kc, + kc) -> natom SW128 atoms of [nt rows x 128 B]
      const int cpr = kc / 8;                              // 16-B chunks per row (16 or 8)
      for (int c = gt; c < r_nv * cpr; c += 128) {
        const int row = c / cpr, cc = c % cpr, a = cc >> 3, jj = cc & 7;
        cp_async16(sa(sRx) + a * (p.nt * 128) + row * 128 + ((jj ^ (row & 7)) << 4),
                   (const uint8_t*)(p.xb + (size_t)(r_t0 + row) * H + rs * kc) + cc * 16);
      }
      cp_arrive_noinc(sa(rfull));
      if (q_ && gt == 0) q_[11] = gtime();
      // routing of this tile's tokens rj + 2 sk w (the CTAs holding the tile's items cover the whole tile)
      for (int u = rj + 2 * p.sk * (warp - 1); u < r_nv; u += 8 * p.sk) {
        if (lane == 0) while (ld_acquire(&p.st->ctr[rn * 32]) < (unsigned)(2 * p.sk)) __nanosleep(20);
        __syncwarp();
        u64* qs = (q_ && warp == 1 && u == rj) ? q_ : nullptr;
        if (qs && lane == 0) qs[14] = gtime();
        route_token(p, r_t0 + u, rn, lane, wtmp[warp - 1], qs);
      }
      if (q_ && gt == 0) q_[6] = gtime();
      mbar_wait(sa(racc), 0);                              // the router MMA has read sB
    }
    // B rows by 16-B cp.async.cg (L2), SFB words by 4-B cp.async.ca (per-expert slot ranges are 8-row aligned, so no
    // 128-B line of the intermediate scales mixes two groups); per-thread, per-stage arrival by
    // cp.async.mbarrier.arrive.noinc. Row ids are read once per unit.
    constexpr int CPR = KSB * 8;                           // 16-B chunks per B row per stage
    constexpr int MAXC = (NMAX * CPR + 127) / 128;
    constexpr int MAXF = (NMAX * KSB + 127) / 128;
    uint32_t k = 0;
    long long c_gdep = 0, c_gempty = 0;
    for (int ui = 0; live; ui++) {
      const int slot = ui % UI;
      mbar_wait(sa(&inff[slot]), (ui / UI) & 1);
      const Info* f = &sinfo[slot];
      const int kind = f->kind;
      if (kind < 0) { named_bar(1, 128); if (gt == 0) mbar_arrive(sa(&infe[slot])); break; }
      const int ntok = f->r.ntok, nkb = f->nkb;
      const uint8_t* bsrc = kind ? p.interd[d] : p.x;
      const uint8_t* bsf = kind ? p.intersfd[d] : p.xsf;
      const int bstride = kind ? I : H, bsfstride = kind ? I / 32 : H / 32;
      const uint8_t* csrc[MAXC]; uint32_t cdst[MAXC]; int nc = 0;
      const uint8_t* fsrc[MAXF]; uint32_t fdst[MAXF]; int nf = 0;
#pragma unroll
      for (int i = 0; i < MAXC; i++) {
        const int c = gt + 128 * i;
        if (c < ntok * CPR) {
          const int row = c / CPR, cc = c % CPR, blk = cc >> 3, jj = cc & 7;
          const int srow = kind ? f->r.slot0 + row : (f->r.j[row] >> 3);
          csrc[i] = bsrc + (size_t)srow * bstride + blk * 128 + jj * 16;
          cdst[i] = blk * NMAX * 128 + row * 128 + ((jj ^ (row & 7)) << 4);
          nc = i + 1;
        }
      }
#pragma unroll
      for (int i = 0; i < MAXF; i++) {
        const int c = gt + 128 * i;
        if (c < ntok * KSB) {
          const int row = c / KSB, blk = c % KSB;
          const int srow = kind ? f->r.slot0 + row : (f->r.j[row] >> 3);
          fsrc[i] = bsf + (size_t)srow * bsfstride + blk * 4;
          fdst[i] = blk * SF_ATOM + (row & 31) * 16 + (row >> 5) * 4;
          nf = i + 1;
        }
      }
      const int gid = f->r.gid;
      named_bar(1, 128);                                   // every gather thread is done with the slot
      if (gt == 0) mbar_arrive(sa(&infe[slot]));
      if (kind == 1) {
        // FC2 reads the FC1 outputs of this group (written by other CTAs): wait for its 8 FC1 units.
        if (gt == 0) { TW(c_gdep, while (ld_acquire(&p.st->fc1done[gid]) < (unsigned)FC1_NRB) __nanosleep(64)); }
        named_bar(1, 128);
      }
      for (int kb = 0; kb < nkb; kb += KSB, k++) {
        const int s = k % S;
        if (k >= (uint32_t)S) TW(c_gempty, mbar_wait(sa(&empty[s]), ((k / S) - 1) & 1));
        const uint32_t bdst = sa(sB + s * B_ST), sdst = sa(sSFB + s * SF_ST);
#pragma unroll
        for (int i = 0; i < MAXC; i++) if (i < nc) cp_async16(bdst + cdst[i], csrc[i] + kb * 128);
#pragma unroll
        for (int i = 0; i < MAXF; i++) if (i < nf) cp_async4(sdst + fdst[i], fsrc[i] + kb * 4);
        cp_arrive_noinc(sa(&full[s]));
        if (q_ && gt == 0 && ui == 0 && kb == 0) q_[21] = gtime();
      }
    }
    cp_wait_all();
    if (q_ && gt == 0) { q_[23] = c_gdep; q_[27] = c_gempty; }
  } else if (warp == 5) {
    // ---------------------------------------------------------------- MMA issuer
    if (lane == 0 && hasr) {
      // router: [128 experts] x [r_np tokens] over this K slice, kind::f16 (K = 16 per instruction: +32 B in the atom)
      mbar_wait(sa(rfull), 0);
      fence_proxy_async();                                 // cp.async (generic proxy) writes -> tcgen05 reads
      tc_fence_after();
      const uint32_t id = idesc_bf16(r_np);
      for (int a = 0; a < natom; a++) {
        const u64 ad = sdesc_sw128(sa(sRw + a * 16384)), bd = sdesc_sw128(sa(sRx + a * p.nt * 128));
#pragma unroll
        for (int kk = 0; kk < 4; kk++) tc_mma_bf16(tmem + RCOL, ad + 2 * kk, bd + 2 * kk, id, (a | kk) != 0);
      }
      tc_commit(sa(racc));
      if (q_) q_[12] = gtime();
    }
    if (lane == 0 && live) {
      uint32_t k = 0;
      long long c_macce = 0, c_mfull = 0;
      const u64 ad0 = sdesc_sw128(sa(sA)), bd0 = sdesc_sw128(sa(sB));
      const u64 fa0 = sdesc_sf(sa(sSFA)), fb0 = sdesc_sf(sa(sSFB));
      for (int ui = 0;; ui++) {
        const int slot = ui % UI;
        mbar_wait(sa(&inff[slot]), (ui / UI) & 1);
        const Info* f = &sinfo[slot];
        const int kind = f->kind, ntok = f->r.ntok, nkb = f->nkb;
        mbar_arrive(sa(&infe[slot]));
        if (kind < 0) break;
        const int b = ui % NACC;
        if (ui >= NACC) TW(c_macce, mbar_wait(sa(&acce[b]), ((ui / NACC) - 1) & 1));
        tc_fence_after();
        const int npad = (ntok + 15) & ~15;
        const uint32_t dacc = tmem + b * ACCW;
        const uint32_t id0 = idesc_mx(npad, 0, 0), id2 = idesc_mx(npad, 2, 2);
        for (int kb = 0; kb < nkb; kb += KSB, k++) {
          const int s = k % S;
          TW(c_mfull, mbar_wait(sa(&full[s]), (k / S) & 1));
          if (q_ && k == 0) q_[7] = gtime();
          fence_proxy_async();                                 // cp.async (generic proxy) writes -> tcgen05 reads
          tc_fence_after();
          if (ntok > 0) {
            const uint32_t sfa_col = tmem + NACC * ACCW + (k & 1) * 16, sfb_col = sfa_col + 8;
            const u64 ads = ad0 + (u64)((s * A_ST) >> 4), bds = bd0 + (u64)((s * B_ST) >> 4);
            const u64 fas = fa0 + (u64)((s * SF_ST) >> 4), fbs = fb0 + (u64)((s * SF_ST) >> 4);
#pragma unroll
            for (int g = 0; g < KSB; g++) {
              tc_cp_sf(sfa_col + 4 * g, fas + (u64)((g * SF_ATOM) >> 4));
              tc_cp_sf(sfb_col + 4 * g, fbs + (u64)((g * SF_ATOM) >> 4));
              const u64 ad = ads + (u64)((g * KB_BYTES) >> 4), bd = bds + (u64)((g * NMAX * 128) >> 4);
              tc_mma(dacc, ad, bd, id0, sfa_col + 4 * g, sfb_col + 4 * g, (kb | g) != 0);
              tc_mma(dacc, ad + 4, bd + 4, id2, (sfa_col + 4 * g) | (2u << 30), (sfb_col + 4 * g) | (2u << 30), 1u);
            }
            tc_commit(sa(&empty[s]));
          } else {
            mbar_arrive(sa(&empty[s]));
          }
        }
        if (ntok > 0) tc_commit(sa(&accf[b])); else mbar_arrive(sa(&accf[b]));
      }
      if (q_) { q_[28] = c_macce; q_[29] = c_mfull; }
    }
  } else {
    // ---------------------------------------------------------------- epilogue (warps 6-9; TMEM lane quarter = warp % 4)
    // Tile row p (TMEM lane) of a 128-row weight tile, trtllm-gen MajorK shuffled layout (32-row blocks):
    //   FC1 (w13, gate/up interleaved): lane l of quarter q: a = l >> 3 (even: up, odd: gate), channel
    //       rb * 64 + q * 16 + 2 * (l & 7) + (a >> 1); its partner (the other half of SwiGLU) is lane l ^ 8.
    //   FC2 (w2): hidden channel rb * 128 + q * 32 + 4 * (l & 7) + (l >> 3).
    griddep_wait();
    const int q = warp & 3, ew = warp - 6;
    if (hasr) {
      // router partials of this item: expert rm * 128 + 32 q + lane, tokens r_t0 + column -> part[rs][t][e]
      mbar_wait(sa(racc), 0);
      tc_fence_after();
      if (q_ && ew == 0 && lane == 0) q_[13] = gtime();
      float* dst = p.part + ((size_t)rs * p.T + r_t0) * E + rm * 128 + q * 32 + lane;
      for (int c0 = 0; c0 < r_np; c0 += 16) {
        float v[16];
        tmem_ld16(tmem + ((uint32_t)(q * 32) << 16) + RCOL + c0, v);
#pragma unroll
        for (int j = 0; j < 16; j++) if (c0 + j < r_nv) dst[(size_t)(c0 + j) * E] = v[j];
      }
      // bar.sync orders every epilogue thread's partial stores before the one gpu-scope release
      named_bar(2, 128);
      if (ew == 0 && lane == 0) { red_release(&p.st->ctr[rn * 32], 1u); if (q_) q_[5] = gtime(); }
    }
    const int a = lane >> 3;
    const bool isgate = a & 1;
    long long c_eaccf = 0, n_units = 0;
    for (int ui = 0; live; ui++) {
      const int slot = ui % UI;
      mbar_wait(sa(&inff[slot]), (ui / UI) & 1);
      const Info* f = &sinfo[slot];
      const int kind = f->kind;
      if (kind < 0) { named_bar(2, 128); if (ew == 0 && lane == 0) mbar_arrive(sa(&infe[slot])); break; }
      const int ntok = f->r.ntok, rb = f->rb, slot0 = f->r.slot0, gid = f->r.gid;
      const short* js = f->r.j;
      const int b = ui % NACC;
      TW(c_eaccf, mbar_wait(sa(&accf[b]), (ui / NACC) & 1));
      tc_fence_after();
      const int npad = (ntok + 15) & ~15;
      const uint32_t taddr = tmem + ((uint32_t)(q * 32) << 16) + b * ACCW;
      // Read the whole accumulator (<= 32 columns) and hand the buffer back to the MMA warp before any math or store,
      // so global-store latency and the FC1 release fence stay off the MMA's critical path.
      float vv[NMAX];
      if (npad > 0) tmem_ld16(taddr, vv);
      if (npad > 16) tmem_ld16(taddr + 16, vv + 16);
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(sa(&acce[b]));                // one arrival per epilogue warp (count 4)
#pragma unroll
      for (int cc = 0; cc < NMAX / 16; cc++) {
        if (cc * 16 >= npad) break;
        const int c0 = cc * 16;
        const float* v = vv + c0;
        if (kind == 0) {
          float pr[16];
#pragma unroll
          for (int j = 0; j < 16; j++) pr[j] = __shfl_xor_sync(0xffffffffu, v[j], 8);
          const int jb = isgate ? 8 : 0;
          const int ch = rb * 64 + q * 16 + 2 * (lane & 7) + (a >> 1), cb = q >> 1;
          float h[8], am[8];
#pragma unroll
          for (int jj = 0; jj < 8; jj++) {
            const float g = isgate ? v[8 + jj] : pr[jj], up = isgate ? pr[8 + jj] : v[jj];
            h[jj] = __fdividef(g, 1.f + __expf(-g)) * up;
            am[jj] = fabsf(h[jj]);
          }
          // amax over the 16 channels of this quarter with the same column half (lanes sharing bit 3)
#pragma unroll
          for (int o = 1; o < 32; o <<= 1) {
            if (o == 8) continue;
#pragma unroll
            for (int jj = 0; jj < 8; jj++) am[jj] = fmaxf(am[jj], __shfl_xor_sync(0xffffffffu, am[jj], o));
          }
          float* sm_ = sam + cc * 64;                          // per-chunk buffer: no reuse barrier inside a unit
          if ((lane & 0x17) == 0) {
#pragma unroll
            for (int jj = 0; jj < 8; jj++) sm_[(q * 2 + (int)isgate) * 8 + jj] = am[jj];
          }
          named_bar(3 + (q >> 1), 64);                         // only the partner warp (q ^ 1) shares the 32-ch block
#pragma unroll
          for (int jj = 0; jj < 8; jj++) am[jj] = fmaxf(am[jj], sm_[((q ^ 1) * 2 + (int)isgate) * 8 + jj]);
          unsigned char qb[8]; int e8s[8];
#pragma unroll
          for (int jj = 0; jj < 8; jj++) {
            int e8 = 0; float inv = 0.f;
            if (am[jj] > 0.f) {
              const int ex = (int)((__float_as_uint(am[jj]) >> 23) & 0xff) - 127 - 8;   // OCP: floor(log2 amax) - emax(e4m3)
              e8 = max(0, min(254, ex + 127));
              inv = exp2f((float)(127 - e8));
            }
            const __nv_fp8_e4m3 qv(h[jj] * inv);
            qb[jj] = *(const unsigned char*)&qv; e8s[jj] = e8;
          }
#pragma unroll
          for (int jj = 0; jj < 8; jj++) {
            const int col = c0 + jb + jj;
            if (col < ntok) {
              const size_t srow = (size_t)(slot0 + col);
              p.interd[d][srow * I + ch] = qb[jj];
              if ((lane & 0x17) == 0 && !(q & 1)) p.intersfd[d][srow * (I / 32) + rb * 2 + cb] = (unsigned char)e8s[jj];
            }
          }
        } else {
          // FC2: each warp owns hidden channels rb * 128 + q * 32 + [0, 32): stage its bf16 [16 tokens][32 ch] in its
          // own SMEM slice and store 64-B row segments (16-B per lane); only __syncwarp, no cross-warp barrier
          __nv_bfloat16* sw = sstage + q * 16 * 32;
          const int hl = 4 * (lane & 7) + a;
#pragma unroll
          for (int j = 0; j < 16; j++) sw[j * 32 + hl] = __float2bfloat16(v[j]);
          __syncwarp();
#pragma unroll
          for (int c = lane; c < 64; c += 32) {
            const int j = c >> 2, part = c & 3, col = c0 + j;
            if (col < ntok)
              *(uint4*)(p.out + (size_t)js[col] * H + rb * 128 + q * 32 + part * 8) = *(const uint4*)(sw + j * 32 + part * 8);
          }
          __syncwarp();
        }
      }
      // FC1: every epilogue thread's intermediate stores happen-before (bar.sync) the one gpu-scope release
      // (cumulative); the same barrier ends all reads of the slot's token list and of sam
      named_bar(2, 128);
      if (ew == 0 && lane == 0) {
        if (kind == 0 && ntok > 0) red_release(&p.st->fc1done[gid], 1u);
        mbar_arrive(sa(&infe[slot]));
      }
      n_units++;
    }
    if (q_ && ew == 0 && lane == 0) { q_[24] = c_eaccf; q_[22] = n_units; }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 5) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem), "r"(TMEM_COLS) : "memory");
  }
  // the last CTA resets the call state for the next call (every other CTA is past all its reads of it)
  if (tid == 0) {
    if (q_) { q_[0] = t_start; q_[2] = gtime(); q_[3] = (u64)get_smid() | ((u64)d << 16) | ((u64)rank << 24) | ((u64)(item + 1) << 40); }
    __threadfence();
    sdom[3] = atomicAdd(&p.st->done, 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (sdom[3]) {
    __threadfence();
    unsigned* w = (unsigned*)p.st;
    for (int i = tid; i < STATE_WORDS; i += NTHREADS) if (i != 2) w[i] = 0u;
    __threadfence();
    __syncthreads();
    if (tid == 0) { p.st->done = 0u; __threadfence(); }
  }
}

constexpr int SMEM_BYTES = 1024 + S * (A_ST + B_ST + 2 * SF_ST) + RX_EXTRA + (2 * S + 2 * NACC + 2 * UI + 2) * 8 + UI * (int)sizeof(Info) +
                           128 * 4 + 4 * 4 + 4 * 4 + (SCHED_MAX + 4) * 4 + 16 * 128 * 2;
static_assert(3 * 16384 <= S * B_ST + RX_EXTRA, "router operand window");
static_assert(16384 + 192 * 128 <= S * B_ST + RX_EXTRA, "router operand window (sk 32, nt 192)");
static_assert((128 + 3 * 132) * 4 <= RX_EXTRA, "schedule arrays");

// ------------------------------------------------------------------ host
static bool g_init[64];

void init(int64_t dev) {
  if (g_init[dev]) return;
  c10::cuda::CUDAGuard guard((int)dev);
  RTC(cudaFuncSetAttribute(k_moe, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
  int occ = 0;
  RTC(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_moe, NTHREADS, SMEM_BYTES));
  TORCH_CHECK(occ == 1, "k_moe occupancy ", occ, " (expected 1 CTA/SM)");
  g_init[dev] = true;
}

int64_t state_bytes() { return (int64_t)sizeof(State); }
int64_t smem_bytes() { return SMEM_BYTES; }

static void encode(CUtensorMap* m, const torch::Tensor& w, CUtensorMapDataType dt, uint64_t rows, uint64_t cols, uint32_t box_cols) {
  const uint64_t es = (uint64_t)w.element_size();
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && (uint64_t)w.numel() == rows * cols);
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstride[1] = {cols * es};
  cuuint32_t box[2] = {box_cols, 128};
  cuuint32_t estride[2] = {1, 1};
  DRV(cuTensorMapEncodeTiled(m, dt, 2, w.data_ptr(), gdim, gstride, box, estride,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

torch::Tensor tensor_maps(torch::Tensor w13, torch::Tensor w2) {
  TORCH_CHECK(w13.element_size() == 1 && w2.element_size() == 1);
  auto t = torch::empty({2 * (int64_t)sizeof(CUtensorMap)}, torch::dtype(torch::kUInt8));
  CUtensorMap* m = (CUtensorMap*)t.data_ptr();
  encode(&m[0], w13, CU_TENSOR_MAP_DATA_TYPE_UINT8, (uint64_t)E * 2 * I, H, 128);
  encode(&m[1], w2, CU_TENSOR_MAP_DATA_TYPE_UINT8, (uint64_t)E * H, I, 128);
  return t;
}

torch::Tensor router_map(torch::Tensor wr) {
  TORCH_CHECK(wr.dtype() == torch::kBFloat16 && wr.size(0) == E && wr.size(1) == H);
  auto t = torch::empty({(int64_t)sizeof(CUtensorMap)}, torch::dtype(torch::kUInt8));
  encode((CUtensorMap*)t.data_ptr(), wr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, E, H, 64);
  return t;
}

static cudaLaunchAttribute pdl_attr(bool pdl) {
  cudaLaunchAttribute a; memset(&a, 0, sizeof(a));
  a.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a.val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  return a;
}

void moe_launch(torch::Tensor tmaps, torch::Tensor w13sf, torch::Tensor w2sf, torch::Tensor xb, torch::Tensor x, torch::Tensor xsf,
                torch::Tensor inter, torch::Tensor intersf, torch::Tensor out, torch::Tensor ew, torch::Tensor ids,
                torch::Tensor state, torch::Tensor list, torch::Tensor part, torch::Tensor smdom,
                int64_t n0, int64_t n1, int64_t nsm, int64_t sk, int64_t nt, std::optional<torch::Tensor> dbg,
                std::optional<torch::Tensor> logits, bool pdl, int64_t prio) {
  const int64_t T = x.size(0);
  TORCH_CHECK(tmaps.device().is_cpu() && tmaps.numel() == 3 * (int64_t)sizeof(CUtensorMap));
  TORCH_CHECK(T >= 1 && T <= TMAX && x.is_contiguous() && x.size(1) == H && xsf.is_contiguous() && xsf.numel() == T * (H / 32));
  TORCH_CHECK(xb.dtype() == torch::kBFloat16 && xb.is_contiguous() && xb.size(0) == T && xb.size(1) == H);
  TORCH_CHECK(out.is_contiguous() && out.size(0) == T * TOPK && out.size(1) == H && out.dtype() == torch::kBFloat16);
  TORCH_CHECK(ew.dtype() == torch::kBFloat16 && ew.numel() == T * TOPK && ids.dtype() == torch::kInt32 && ids.numel() == T * TOPK);
  TORCH_CHECK(w13sf.numel() == (int64_t)E * W13SF_E && w2sf.numel() == (int64_t)E * W2SF_E);
  // grid = n0 + n1 CTAs (ranks per domain); nsm - grid SMs stay free for concurrent kernels (shared expert)
  TORCH_CHECK(n0 >= 64 && n1 >= 64 && n0 + n1 <= nsm && smdom.numel() == nsm);
  TORCH_CHECK((sk == 16 && nt == 64) || (sk == 32 && nt == 192));
  const int64_t ntiles = (T + nt - 1) / nt, nitems = 2 * sk * ntiles;
  TORCH_CHECK(ntiles <= NTILE && nitems <= 2 * min(n0, n1), "router split ", sk, "x", nt, " does not fit T=", T);
  TORCH_CHECK(state.numel() >= (int64_t)sizeof(State) && list.numel() >= (int64_t)E * TMAX && part.numel() >= sk * T * E);
  TORCH_CHECK(inter.dim() == 2 && intersf.dim() == 2 && inter.size(0) == 2 && intersf.size(0) == 2);
  TORCH_CHECK(inter.size(1) >= (int64_t)(TMAX * TOPK + 7 * E) * I && intersf.size(1) >= (int64_t)(TMAX * TOPK + 7 * E) * (I / 32));
  c10::cuda::CUDAGuard guard(x.device());
  CUtensorMap m[3];
  memcpy(m, tmaps.data_ptr(), sizeof(m));
  Params prm;
  prm.w13sf = (const uint8_t*)w13sf.data_ptr(); prm.w2sf = (const uint8_t*)w2sf.data_ptr();
  prm.xb = (const __nv_bfloat16*)xb.data_ptr();
  prm.x = (const uint8_t*)x.data_ptr(); prm.xsf = (const uint8_t*)xsf.data_ptr();
  for (int dd = 0; dd < 2; dd++) {
    prm.interd[dd] = (uint8_t*)inter.data_ptr() + dd * inter.stride(0);
    prm.intersfd[dd] = (uint8_t*)intersf.data_ptr() + dd * intersf.stride(0);
  }
  prm.out = (__nv_bfloat16*)out.data_ptr(); prm.ew = (__nv_bfloat16*)ew.data_ptr(); prm.ids = (int*)ids.data_ptr();
  prm.logits = nullptr;
  if (logits.has_value()) {
    TORCH_CHECK(logits->dtype() == torch::kBFloat16 && logits->is_contiguous() && logits->numel() == T * E);
    prm.logits = (__nv_bfloat16*)logits->data_ptr();
  }
  prm.st = (State*)state.data_ptr(); prm.list = (short*)list.data_ptr(); prm.part = (float*)part.data_ptr();
  prm.smdom = (const signed char*)smdom.data_ptr(); prm.nd0 = (int)n0; prm.nd1 = (int)n1;
  prm.T = (int)T; prm.sk = (int)sk; prm.nt = (int)nt; prm.ntiles = (int)ntiles; prm.nitems = (int)nitems;
  prm.dbg = dbg.has_value() ? (u64*)dbg->data_ptr() : nullptr;
  TORCH_CHECK(!dbg.has_value() || dbg->numel() * dbg->element_size() >= (n0 + n1) * DBGW * 8);
  cudaLaunchConfig_t cfg; memset(&cfg, 0, sizeof(cfg));
  cfg.gridDim = dim3((unsigned)(n0 + n1)); cfg.blockDim = dim3(NTHREADS); cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute at[2] = {pdl_attr(pdl), {}};
  at[1].id = cudaLaunchAttributePriority;
  at[1].val.priority = (int)prio;
  cfg.attrs = at; cfg.numAttrs = prio ? 2 : 1;
  RTC(cudaLaunchKernelEx(&cfg, k_moe, m[0], m[1], m[2], prm));
}

}  // namespace locmoe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &locmoe::init);
  m.def("state_bytes", &locmoe::state_bytes);
  m.def("smem_bytes", &locmoe::smem_bytes);
  m.def("tensor_maps", &locmoe::tensor_maps);
  m.def("router_map", &locmoe::router_map);
  m.def("moe_launch", &locmoe::moe_launch);
}
"""
