# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Locality-domain ("per-die") MXFP8 MoE for decode on Rubin-class GPUs.

``VLLM_MOE_LOCALITY_KERNEL=1`` (default off) replaces the trtllm-gen MXFP8 MoE
call (routing + FC1 + FC2, deferred finalize) for decode batches with one
persistent kernel whose CTAs read only expert weights that live in their own
locality domain's HBM:

- Placement (:func:`place_expert_pairs`): each expert's w13 (``[1024, 2048]``
  e4m3 = exactly one 2 MiB chunk) and each expert pair's w2 (2 x 1 MiB) are
  moved, in place of the original tensors and in the same trtllm-gen MajorK
  shuffled layout, to locality domain ``(e >> 1) & 1``. The trtllm-gen path
  (mixed / prefill steps) keeps reading the same tensors; scales stay in
  ``cudaMalloc`` memory.
- Routing: ``topk_softmax`` (renormalized softmax top-8, lower expert id wins
  ties) gives ``topk_ids`` / ``topk_weights``; ``k_plan`` (one CTA) builds the
  per-expert token lists (ascending token order), the per-domain work lists and
  ``expert_weights`` (bf16).
- ``k_moe`` (grid = #SMs, 1 CTA/SM, 320 threads, warp-specialized, PDL):
  warp 0 streams weight tiles with SW128 tensor-map TMA (+ 1-D bulk copies of
  the 128x4-interleaved scale atoms) into a 5 x 32 KB ring; warps 1-4 gather
  the expert's token rows (``cp.async``); warp 5 issues block-scaled tcgen05
  MMAs (M = 128 weight rows, N <= 32 tokens); warps 6-9 run the epilogue
  (FC1: SwiGLU + MXFP8 requant into an intermediate; FC2: bf16 rows at the
  expanded index ``t * 8 + k``). A CTA takes the domain of its SM
  (``%smid`` -> ``Topology.sm_domain``) and a rank in that domain's static
  round-robin schedule (FC1 units, then FC2 units; FC2 waits on a per-group
  FC1 counter). Before ``griddepcontrol.wait`` it fills its ring with the
  first FC1 tile of its static "wave-0" unit (weights are never written by a
  predecessor); the unit is dropped if its expert turns out to be idle.
- Output: :class:`UnfinalizedMoEOutput` with ``gemm2_permuted[T * 8, H]``,
  ``expert_weights[T, 8]`` and the constant ``expanded_idx_to_permuted_idx =
  arange(T * 8)`` (the DLC-8 deferred-finalize contract).

Shapes are fixed to Qwen3.6-35B-A3B: E = 256, top-8, H = 2048, I = 512, MXFP8
(1x32 UE8M0). Decode batches of 1 ... 512 tokens.

This module imports only torch at import time, so microbenchmarks can load it
by file path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os

import torch

E, TOPK, HID, INTER = 256, 8, 2048, 512
MAX_TOKENS = 512
MAX_SLOTS = MAX_TOKENS * TOPK + 7 * E  # per-expert slot ranges are 8-aligned
CHUNK = 2 << 20

# VLLM_MOE_LOCALITY_KERNEL=1: place MXFP8 trtllm-gen expert weights by expert
# pair and serve deferred-finalize calls with MIN_TOKENS <= T <= MAX_TOKENS by
# this kernel (acts inside the MoE custom op: no compile-key change).
ENABLED = os.environ.get("VLLM_MOE_LOCALITY_KERNEL", "0") == "1"
GATE_MIN_TOKENS = int(os.environ.get("VLLM_MOE_LOCALITY_KERNEL_MIN_TOKENS", "1"))
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


def place_scales_domain_major(s13: torch.Tensor, s2: torch.Tensor):
    """Localized copies of the (128x4-interleaved, expert-major) MXFP8 weight
    scales re-ordered domain-major: domain d's experts (local index k) at slot
    d * 128 + k, so each domain's scales fill whole 2 MiB chunks on that domain.
    The production tensors are kept for the trtllm-gen path (3 % of the weight
    bytes per layer); reading the scales from interleaved memory couples every
    stage of the per-SM ring to fabric latency.
    """
    from vllm.model_executor.layers.locality.memory import alloc_chunks, chunk_ordinals

    dev = s13.device.index
    perm = torch.tensor(
        [4 * (k >> 1) + 2 * d + (k & 1) for d in (0, 1) for k in range(E // 2)],
        device=s13.device,
    )
    out = []
    for s in (s13, s2):
        per = s.numel() // E
        nbytes = s.numel() * s.element_size()
        assert nbytes % (2 * CHUNK) == 0
        plan = [0] * (nbytes // CHUNK // 2) + [1] * (nbytes // CHUNK // 2)
        flat = alloc_chunks(nbytes, plan, CHUNK, dev)
        if chunk_ordinals(flat, nbytes, CHUNK) != plan:
            raise RuntimeError("locality placement mismatch (scales)")
        flat.view(E, per).copy_(s.reshape(E, per).view(torch.uint8)[perm])
        out.append(flat)
    return out[0], out[1]


@dataclasses.dataclass
class LayerWeights:
    tmaps: torch.Tensor  # host bytes of the w13 / w2 tensor maps
    s13: torch.Tensor  # w13 scales (domain-major localized copy or production)
    s2: torch.Tensor
    sf_dmajor: bool
    # diagnostics: pre-blocked (w13, w2) copies read with 1-D bulk copies
    blk: tuple[torch.Tensor, torch.Tensor] | None = None


# ---------------------------------------------------------------- runtime
class LocalityMoE:
    """Per-device state of the locality MoE kernel (topology, scratch)."""

    def __init__(self, device: int):
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
        self.plan = torch.zeros(self.ext.plan_bytes(), dtype=torch.uint8, device=dev)
        self.state = torch.zeros(self.ext.state_bytes(), dtype=torch.uint8, device=dev)
        # FC1 -> FC2 intermediate: one slot space per domain, on that domain
        # (FC1 and FC2 of a group run in the same domain); [1, ...] cudaMalloc
        # copies for A/B
        from vllm.model_executor.layers.locality.memory import alloc_chunks

        ib = -(-MAX_SLOTS * INTER // CHUNK) * CHUNK
        sb = -(-MAX_SLOTS * (INTER // 32) // CHUNK) * CHUNK
        self.inter = alloc_chunks(
            2 * ib, [0] * (ib // CHUNK) + [1] * (ib // CHUNK), CHUNK, device
        ).view(2, ib)
        self.inter_sf = alloc_chunks(
            2 * sb, [0] * (sb // CHUNK) + [1] * (sb // CHUNK), CHUNK, device
        ).view(2, sb)
        self.inter_glob = torch.empty(
            1, MAX_SLOTS * INTER, dtype=torch.uint8, device=dev
        )
        self.inter_sf_glob = torch.empty(
            1, MAX_SLOTS * (INTER // 32), dtype=torch.uint8, device=dev
        )
        self.idx = torch.arange(MAX_TOKENS * TOPK, dtype=torch.int32, device=dev)
        self.smdom = topo.sm_domain

    def tensor_maps(self, w13: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        """Host bytes of the two SW128 tensor maps (built once per layer)."""
        return self.ext.tensor_maps(w13, w2)

    def route(self, router_logits: torch.Tensor):
        from vllm import _custom_ops as ops

        T = router_logits.shape[0]
        dev = router_logits.device
        topk_w = torch.empty(T, TOPK, dtype=torch.float32, device=dev)
        topk_ids = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
        tok_idx = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
        ops.topk_softmax(topk_w, topk_ids, tok_idx, router_logits, True)
        return topk_ids, topk_w

    def layer_weights(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        s13: torch.Tensor,
        s2: torch.Tensor,
        localize_scales: bool = True,
    ) -> LayerWeights:
        """Kernel view of one layer: tensor maps of w13/w2 (as placed) and the
        scales (a domain-major localized copy, or the production tensors).
        """
        if localize_scales:
            ls13, ls2 = place_scales_domain_major(s13, s2)
            return LayerWeights(self.tensor_maps(w13, w2), ls13, ls2, True)
        return LayerWeights(self.tensor_maps(w13, w2), s13, s2, False)

    def forward(
        self,
        router_logits: torch.Tensor,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        lw: LayerWeights,
        topk: tuple[torch.Tensor, torch.Tensor] | None = None,
        dbg: torch.Tensor | None = None,
        pdl: bool = True,
        inter_local: bool = True,
        xrep: torch.Tensor | None = None,
    ):
        """Returns (gemm2_permuted [T*8, H] bf16, expert_weights [T, 8] bf16,
        expanded_idx_to_permuted_idx [T, 8] int32). ``dbg`` (int64 [nsm, 4])
        receives per-CTA %globaltimer stamps; ``pdl=False`` launches both
        kernels without programmatic dependent launch (diagnostics).
        """
        T = x.shape[0]
        assert 0 < T <= MAX_TOKENS
        topk_ids, topk_w = self.route(router_logits) if topk is None else topk
        ew = torch.empty(T, TOPK, dtype=torch.bfloat16, device=x.device)
        out = torch.empty(T * TOPK, HID, dtype=torch.bfloat16, device=x.device)
        self.ext.plan_launch(
            topk_ids, topk_w, ew, self.plan, T, self.nd[0], self.nd[1], pdl
        )
        self.ext.moe_launch(
            lw.tmaps,
            lw.s13,
            lw.s2,
            x,
            x_sf.view(torch.uint8),
            self.inter if inter_local else self.inter_glob,
            self.inter_sf if inter_local else self.inter_sf_glob,
            out,
            self.plan,
            self.state,
            self.smdom,
            self.nd[0],
            self.nd[1],
            self.nsm,
            dbg,
            pdl,
            lw.sf_dmajor,
            lw.blk[0] if lw.blk else None,
            lw.blk[1] if lw.blk else None,
            xrep,
        )
        return out, ew, self.idx[: T * TOPK].view(T, TOPK)


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
    """Load-time hook (trtllm-gen MXFP8 layout): pair-placed copies of w13/w2
    (which replace the originals) plus domain-major localized scale copies (the
    originals stay for trtllm-gen) when the kernel is enabled and applies; else
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
    router_logits: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
):
    """(gemm2_permuted, expert_weights, expanded_idx_to_permuted_idx), or None
    when this call is not served (T outside the gate, layer not placed, or
    input layouts other than [T, H] e4m3 + linear [T, H/32] UE8M0).
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
        or x_sf.numel() != T * (HID // 32)
        or not x_sf.is_contiguous()
        or router_logits.shape != (T, E)
    ):
        return None
    return runtime(x.device.index).forward(router_logits.contiguous(), x, x_sf, lw)


_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstring>

namespace locmoe {

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
constexpr int TMEM_COLS = 256, ACCW = 64;
constexpr int W13SF_E = 2 * I * H / 32, W2SF_E = H * I / 32;  // scale bytes per expert (128x4-interleaved)

// One (expert, sub-block of <= 32 tokens) of a domain: the unit record the producer bulk-copies into SMEM.
struct alignas(16) GRec { int e, sub, ntok, slot0; short j[32]; int gid, pad0, pad1, pad2; };   // 96 B
struct Plan {
  int ng[2], nfc1[2];                 // groups per domain; dynamic FC1 list length per domain
  int nslots, T, pad0, pad1;
  u64 ts[8];                          // k_plan %globaltimer stamps (diagnostics)
  short k2g[2][128];                  // domain-local expert -> its first group (-1: idle)
  short fc1list[2][GMAX * 8];         // dynamic FC1 units: g * 8 + rb (wave-0 units excluded)
  GRec grec[2][GMAX];
};
struct State { unsigned rank[2]; unsigned done; unsigned pad; unsigned fc1done[2 * GMAX]; };
struct alignas(16) Info { GRec r; int kind, rb, nkb, kb0; int pad[4]; };                       // 128 B
struct Params {
  const uint8_t* w13sf; const uint8_t* w2sf; const uint8_t* x; const uint8_t* xsf;
  uint8_t* interd[2]; uint8_t* intersfd[2]; __nv_bfloat16* out;   // FC1 -> FC2 intermediate of domain d's groups
  const Plan* plan; State* st; const signed char* smdom;
  int nd0, nd1;
  int sf_dmajor;                      // scales: 0 = production expert-major, 1 = domain-major localized copy
  u64* dbg;                           // optional per-CTA stamps [grid][4]: start, producer past wait, end, smid|dom|rank
  const uint8_t* blk13; const uint8_t* blk2;   // optional (diagnostics): pre-blocked weight copies, see issue_stage
  const uint8_t* xd[2]; const uint8_t* xsfd[2]; // activations read by domain d's CTAs (replicas, or both = x / xsf)
};
// scale block of expert e: expert-major (production tensors) or domain-major (localized copy: domain d's 128 experts,
// local index k, at slot d * 128 + k, so each domain's scales fill whole 2 MiB chunks of that domain)
__host__ __device__ __forceinline__ int sf_slot(int e, int dmajor) {
  return dmajor ? ((e >> 1) & 1) * 128 + (e >> 2) * 2 + (e & 1) : e;
}

#define RTC(x) do { cudaError_t e_ = (x); TORCH_CHECK(e_ == cudaSuccess, #x " failed: ", cudaGetErrorString(e_)); } while (0)
#define DRV(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorName(r_, &s_); TORCH_CHECK(false, #x " failed: ", s_); } } while (0)

// ------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ unsigned get_smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ u64 gtime() { u64 r; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); return r; }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mbar_expect(uint32_t a, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(tx) : "memory"); }
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
__device__ __forceinline__ int warp_iscan(int v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
  return v;
}
// domain-local expert index k (0..127) of domain d -> expert id: pairs {0,1} -> d0, {2,3} -> d1, {4,5} -> d0, ...
__host__ __device__ __forceinline__ int expert_of(int d, int k) { return 4 * (k >> 1) + 2 * d + (k & 1); }

// ------------------------------------------------------------------ k_plan: per-expert token lists + work lists
// One CTA of 1024 threads. The plan is built in SMEM and copied out with coalesced 16-B stores (scattered per-thread
// record stores from one SM took ~8 us). Token order within an expert is the order of shared-memory atomics, i.e.
// not fixed; outputs do not depend on it (every token is its own MMA column, K order is fixed per unit, and every
// output row t * 8 + k is written once), so results stay bitwise run-to-run.
constexpr int PLAN_HDR = (int)offsetof(Plan, k2g) + (int)sizeof(((Plan*)0)->k2g);    // header + k2g, copied whole
constexpr int PLAN_SMEM = (int)sizeof(Plan) + 8 * E * 4 + 2 * TMAX * TOPK * 2 + 2 * GMAX * 4 + 64;
__global__ void __launch_bounds__(1024, 1) k_plan(const int* __restrict__ ids, const float* __restrict__ tw,
                                                  __nv_bfloat16* __restrict__ ew, Plan* __restrict__ pl, int T, int n0, int n1) {
  extern __shared__ __align__(16) unsigned char psm[];
  Plan* P = (Plan*)psm;
  int* cnt = (int*)(psm + sizeof(Plan));
  int* fill = cnt + E; int* eoff = fill + E; int* soff = eoff + E;
  int* g0e = soff + E; int* f0e = g0e + E; int* nse = f0e + E; int* ws = nse + E;      // ws: [4][8]
  short* sid = (short*)(ws + 32);
  short* srt = sid + TMAX * TOPK;
  int* gmap = (int*)(srt + TMAX * TOPK);                                                  // [2][GMAX]: e << 8 | sub
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  griddep_wait();
  griddep_launch();                   // k_moe may start its weight prefetch now; it waits for this grid before reading pl
  const u64 t0 = gtime();
  const int N = T * TOPK;
  if (tid < E) { cnt[tid] = 0; fill[tid] = 0; }
  __syncthreads();
  for (int i = tid; i < N; i += 1024) {
    const int e = ids[i];
    sid[i] = (short)e;
    atomicAdd(&cnt[e], 1);
    ew[i] = __float2bfloat16(tw[i]);
  }
  __syncthreads();
  if (tid < 2 * 128) {
    // (a) expert order: compact offsets and 8-aligned slot offsets; (b) per domain, local expert order: groups and the
    // dynamic FC1 list (wave-0 units, the first nd static FC1 units of the domain, are run by k_moe directly)
    const int c = cnt[tid], c8 = (c + 7) & ~7;
    const int d = tid >> 7, k = tid & 127, e2 = expert_of(d, k);
    const int c2 = cnt[e2], nsub = (c2 + 31) >> 5, nd = d ? n1 : n0;
    const int skip0 = nsub ? min(max(nd - 8 * k, 0), 8) : 0;
    const int nf = 8 * nsub - skip0;
    const int a = warp_iscan(c), b = warp_iscan(c8), a2 = warp_iscan(nsub), b2 = warp_iscan(nf);
    if (lane == 31) { ws[w] = a; ws[8 + w] = b; ws[16 + w] = a2; ws[24 + w] = b2; }
    named_bar(1, 256);
    int ba = 0, bb = 0, ba2 = 0, bb2 = 0;
    for (int x = 0; x < w; x++) { ba += ws[x]; bb += ws[8 + x]; }
    for (int x = d * 4; x < w; x++) { ba2 += ws[16 + x]; bb2 += ws[24 + x]; }
    eoff[tid] = ba + a - c; soff[tid] = bb + b - c8;
    const int g0 = ba2 + a2 - nsub;
    g0e[e2] = g0; f0e[e2] = bb2 + b2 - nf; nse[e2] = nsub;
    P->k2g[d][k] = (short)(nsub ? g0 : -1);
    if (k == 127) { P->ng[d] = g0 + nsub; P->nfc1[d] = bb2 + b2; }
    if (tid == E - 1) { P->nslots = bb + b; P->T = T; }
    for (int s = 0; s < nsub; s++) gmap[d * GMAX + g0 + s] = (e2 << 8) | s;
  }
  __syncthreads();
  for (int i = tid; i < N; i += 1024) {
    const int e = sid[i];
    srt[eoff[e] + atomicAdd(&fill[e], 1)] = (short)i;
  }
  __syncthreads();
  const u64 t1 = gtime();
  const int ng0 = P->ng[0], ng1 = P->ng[1];
  for (int it = tid; it < (ng0 + ng1) * 32; it += 1024) {
    const int d = it >= ng0 * 32, gi = it - d * ng0 * 32, g = gi >> 5, q = gi & 31;
    const int m = gmap[d * GMAX + g], e = m >> 8, s = m & 255, c = cnt[e], nt = min(32, c - 32 * s);
    GRec* R = &P->grec[d][g];
    R->j[q] = q < nt ? srt[eoff[e] + 32 * s + q] : (short)0;
    if (q == 0) { R->e = e; R->sub = s; R->ntok = nt; R->slot0 = soff[e] + 32 * s; R->gid = d * GMAX + g; R->pad0 = R->pad1 = R->pad2 = 0; }
    if (q < 8) {
      const int rb = q, k = (e >> 2) * 2 + (e & 1), nd = d ? n1 : n0;
      const int skip0 = min(max(nd - 8 * k, 0), 8);
      if (s > 0) P->fc1list[d][f0e[e] + (8 - skip0) + 8 * (s - 1) + rb] = (short)(g * 8 + rb);
      else if (rb >= skip0) P->fc1list[d][f0e[e] + rb - skip0] = (short)(g * 8 + rb);
    }
  }
  __syncthreads();
  const u64 t2 = gtime();
  if (tid == 0) { P->ts[0] = t0; P->ts[1] = t1; P->ts[2] = t2; P->ts[3] = 0; }
  __syncthreads();
  // copy out: header + k2g, fc1list[d][0, nfc1[d]), grec[d][0, ng[d]) as 16-B chunks
  const int nh = PLAN_HDR / 16, nl0 = (P->nfc1[0] * 2 + 15) / 16, nl1 = (P->nfc1[1] * 2 + 15) / 16;
  const int nr0 = ng0 * (int)sizeof(GRec) / 16, nr1 = ng1 * (int)sizeof(GRec) / 16;
  constexpr int OL = (int)offsetof(Plan, fc1list) / 16, OLD = GMAX * 8 * 2 / 16;
  constexpr int OR = (int)offsetof(Plan, grec) / 16, ORD = GMAX * (int)sizeof(GRec) / 16;
  const uint4* src = (const uint4*)psm;
  uint4* dst = (uint4*)pl;
  for (int i = tid; i < nh + nl0 + nl1 + nr0 + nr1; i += 1024) {
    int o;
    if (i < nh) o = i;
    else if (i < nh + nl0) o = OL + (i - nh);
    else if (i < nh + nl0 + nl1) o = OL + OLD + (i - nh - nl0);
    else if (i < nh + nl0 + nl1 + nr0) o = OR + (i - nh - nl0 - nl1);
    else o = OR + ORD + (i - nh - nl0 - nl1 - nr0);
    dst[o] = src[o];
  }
}

// ------------------------------------------------------------------ k_moe
// blk != nullptr: diagnostics only, weights from a pre-swizzled copy with contiguous 16 KB K128 blocks per
// (expert, row block) (1-D bulk copies, the feasibility microbench's layout) instead of the production tensors.
__device__ __forceinline__ void issue_stage(uint32_t sA, uint32_t sSFA, uint32_t bar, const CUtensorMap* tm, const uint8_t* sf,
                                            int row0, int kb, const uint8_t* blk) {
  mbar_expect(bar, A_ST + SF_ST);
  if (blk) {
    bulk_load(sA, blk + (size_t)kb * KB_BYTES, A_ST, bar);
  } else {
#pragma unroll
    for (int g = 0; g < KSB; g++) tma_2d(sA + g * KB_BYTES, tm, (kb + g) * 128, row0, bar);
  }
  bulk_load(sSFA, sf + kb * SF_ATOM, SF_ST, bar);
}
__device__ __forceinline__ const uint8_t* blk_unit(const Params& p, int kind, int e, int rb) {
  if (!p.blk13) return nullptr;
  return kind ? p.blk2 + ((size_t)e * 16 + rb) * FC2_NKB * KB_BYTES : p.blk13 + ((size_t)e * 8 + rb) * FC1_NKB * KB_BYTES;
}

__global__ void __launch_bounds__(NTHREADS, 1) k_moe(const __grid_constant__ CUtensorMap tm13, const __grid_constant__ CUtensorMap tm2,
                                                     const Params p) {
  extern __shared__ unsigned char smem_raw[];
  // align by pointer arithmetic on the __shared__ array (an integer cast would drop the address space)
  unsigned char* sm = smem_raw + ((1024u - ((uint32_t)__cvta_generic_to_shared(smem_raw) & 1023u)) & 1023u);
  unsigned char* sA = sm;
  unsigned char* sB = sA + S * A_ST;
  unsigned char* sSFA = sB + S * B_ST;
  unsigned char* sSFB = sSFA + S * SF_ST;
  u64* bars = (u64*)(sSFB + S * SF_ST);
  u64* full = bars; u64* empty = bars + S; u64* accf = bars + 2 * S; u64* acce = accf + 2; u64* inff = acce + 2; u64* infe = inff + UI;
  Info* sinfo = (Info*)(infe + UI);
  float* sam = (float*)(sinfo + UI);                     // FC1 requant: per-warp partial amax [4 warps][2 parities][8 cols]
  volatile int* sdom = (volatile int*)(sam + 64);        // [0] served domain, [1] rank in it
  uint32_t* stmem = (uint32_t*)(sdom + 4);
  int* ssched = (int*)(stmem + 4);                       // this CTA's units (entry 0 = wave-0 unit)
  __nv_bfloat16* sstage = (__nv_bfloat16*)(ssched + SCHED_MAX + 4);   // FC2 epilogue: [16 tokens][128 ch] bf16

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const u64 t_start = gtime();
  if (tid == 0) {
    // Rank in this SM's domain; if the domain already has all its ranks (an SM ran a second CTA because another kernel
    // held an SM), take a rank of the other domain, so every rank is served exactly once.
    int d = p.smdom[get_smid()];
    d = d < 0 ? 0 : d;
    unsigned r = atomicAdd(&p.st->rank[d], 1u);
    if (r >= (unsigned)(d ? p.nd1 : p.nd0)) { d ^= 1; r = atomicAdd(&p.st->rank[d], 1u); }
    sdom[0] = d; sdom[1] = (int)r;
    for (int s = 0; s < S; s++) { mbar_init(sa(&full[s]), 1 + 128); mbar_init(sa(&empty[s]), 1); }
    for (int b = 0; b < 2; b++) { mbar_init(sa(&accf[b]), 1); mbar_init(sa(&acce[b]), 1); }
    for (int i = 0; i < UI; i++) { mbar_init(sa(&inff[i]), 1); mbar_init(sa(&infe[i]), 3); }
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
  const int d = sdom[0], rank = sdom[1];
  const int nd = d ? p.nd1 : p.nd0;
  const bool live = rank < nd;

  if (warp == 0) {
    // ---------------------------------------------------------------- producer
    // wave-0: static FC1 unit `rank` of the domain (expert k = rank / 8, row block rank % 8); its first S stages are
    // loaded before griddepcontrol.wait (weights are never written by a predecessor).
    const int k0 = rank >> 3, rb0 = rank & 7, e0 = expert_of(d, k0);
    if (lane == 0 && live) {
      asm volatile("prefetch.tensormap [%0];" :: "l"((u64)&tm13) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"((u64)&tm2) : "memory");
      for (int s = 0; s < S; s++)
        issue_stage(sa(sA + s * A_ST), sa(sSFA + s * SF_ST), sa(&full[s]), &tm13,
                    p.w13sf + (size_t)sf_slot(e0, p.sf_dmajor) * W13SF_E + rb0 * FC1_NKB * SF_ATOM, e0 * 2 * I + rb0 * 128, KSB * s,
                    blk_unit(p, 0, e0, rb0));
    }
    griddep_wait();
    if (p.dbg && lane == 0) p.dbg[blockIdx.x * 4 + 1] = gtime();
    const Plan* pl = p.plan;
    int nmine = 0;
    if (live) {
      const int ng = pl->ng[d], nf1 = pl->nfc1[d], tot = nf1 + 16 * ng;
      nmine = rank < tot ? (tot - rank + nd - 1) / nd : 0;
      nmine = nmine < SCHED_MAX - 1 ? nmine : SCHED_MAX - 1;
      for (int i = lane; i < nmine; i += 32) {
        const int m = rank + i * nd;
        ssched[1 + i] = m < nf1 ? (int)pl->fc1list[d][m] : (0x10000 | (m - nf1));
      }
      if (lane == 0) { const int g0 = pl->k2g[d][k0]; ssched[0] = g0 >= 0 ? (0x20000 | (g0 * 8 + rb0)) : 0x40000; }
      nmine += 1;
    }
    __syncwarp();
    if (lane == 0) {
      uint32_t k = live ? S : 0;
      int pub = 0;
      auto pub_next = [&]() {
        const int slot = pub % UI;
        if (pub >= UI) mbar_wait(sa(&infe[slot]), ((pub / UI) - 1) & 1);
        Info* f = &sinfo[slot];
        if (pub >= nmine) {
          f->kind = -1;
          mbar_arrive(sa(&inff[slot]));
        } else {
          const int c = ssched[pub];
          int g, kind, rb, nkb, kb0 = 0;
          if (c & 0x40000) { g = -1; kind = 0; rb = rb0; nkb = KSB * S; kb0 = KSB * S; }
          else if (c & 0x10000) { g = (c & 0xffff) >> 4; kind = 1; rb = c & 15; nkb = FC2_NKB; }
          else { g = (c & 0xffff) >> 3; kind = 0; rb = c & 7; nkb = FC1_NKB; kb0 = (c & 0x20000) ? KSB * S : 0; }
          f->kind = kind; f->rb = rb; f->nkb = nkb; f->kb0 = kb0;
          if (g >= 0) {
            mbar_expect(sa(&inff[slot]), (uint32_t)sizeof(GRec));
            bulk_load(sa(&f->r), &pl->grec[d][g], (uint32_t)sizeof(GRec), sa(&inff[slot]));
          } else {
            f->r.e = e0; f->r.ntok = 0; f->r.gid = -1;
            mbar_arrive(sa(&inff[slot]));
          }
        }
        pub++;
      };
      pub_next(); pub_next();
      for (int uj = 0;; uj++) {
        const int slot = uj % UI;
        mbar_wait(sa(&inff[slot]), (uj / UI) & 1);
        const Info* f = &sinfo[slot];
        const int kind = f->kind;
        if (kind < 0) break;
        const int e = f->r.e, rb = f->rb, nkb = f->nkb;
        const CUtensorMap* tm = kind ? &tm2 : &tm13;
        const size_t es = (size_t)sf_slot(e, p.sf_dmajor);
        const uint8_t* sf = kind ? p.w2sf + es * W2SF_E + rb * FC2_NKB * SF_ATOM : p.w13sf + es * W13SF_E + rb * FC1_NKB * SF_ATOM;
        const int row0 = kind ? e * H + rb * 128 : e * 2 * I + rb * 128;
        const uint8_t* blk = blk_unit(p, kind, e, rb);
        for (int kb = f->kb0; kb < nkb; kb += KSB, k++) {
          const int s = k % S;
          if (k >= (uint32_t)S) mbar_wait(sa(&empty[s]), ((k / S) - 1) & 1);
          issue_stage(sa(sA + s * A_ST), sa(sSFA + s * SF_ST), sa(&full[s]), tm, sf, row0, kb, blk);
        }
        if (pub < nmine + 1) pub_next();
      }
    }
  } else if (warp <= 4) {
    // ---------------------------------------------------------------- gather (128 threads)
    // B rows by 16-B cp.async.cg (L2), SFB words by 4-B cp.async.ca (per-expert slot ranges are 8-row aligned, so no
    // 128-B line of the intermediate scales mixes two groups); per-thread, per-stage arrival by
    // cp.async.mbarrier.arrive.noinc. Row ids are read once per unit.
    griddep_wait();
    const int gt = tid - 32;
    constexpr int CPR = KSB * 8;                           // 16-B chunks per B row per stage
    constexpr int MAXC = (NMAX * CPR + 127) / 128;
    constexpr int MAXF = (NMAX * KSB + 127) / 128;
    uint32_t k = 0;
    for (int ui = 0; live; ui++) {
      const int slot = ui % UI;
      mbar_wait(sa(&inff[slot]), (ui / UI) & 1);
      const Info* f = &sinfo[slot];
      const int kind = f->kind;
      if (kind < 0) { named_bar(1, 128); if (gt == 0) mbar_arrive(sa(&infe[slot])); break; }
      const int ntok = f->r.ntok, nkb = f->nkb;
      const uint8_t* bsrc = kind ? p.interd[d] : p.xd[d];
      const uint8_t* bsf = kind ? p.intersfd[d] : p.xsfd[d];
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
        if (gt == 0) { while (ld_acquire(&p.st->fc1done[gid]) < (unsigned)FC1_NRB) __nanosleep(64); }
        named_bar(1, 128);
      }
      for (int kb = 0; kb < nkb; kb += KSB, k++) {
        const int s = k % S;
        if (k >= (uint32_t)S) mbar_wait(sa(&empty[s]), ((k / S) - 1) & 1);
        const uint32_t bdst = sa(sB + s * B_ST), sdst = sa(sSFB + s * SF_ST);
#pragma unroll
        for (int i = 0; i < MAXC; i++) if (i < nc) cp_async16(bdst + cdst[i], csrc[i] + kb * 128);
#pragma unroll
        for (int i = 0; i < MAXF; i++) if (i < nf) cp_async4(sdst + fdst[i], fsrc[i] + kb * 4);
        cp_arrive_noinc(sa(&full[s]));
      }
    }
    cp_wait_all();
  } else if (warp == 5) {
    // ---------------------------------------------------------------- MMA issuer
    if (lane == 0 && live) {
      uint32_t k = 0;
      const u64 ad0 = sdesc_sw128(sa(sA)), bd0 = sdesc_sw128(sa(sB));
      const u64 fa0 = sdesc_sf(sa(sSFA)), fb0 = sdesc_sf(sa(sSFB));
      for (int ui = 0;; ui++) {
        const int slot = ui % UI;
        mbar_wait(sa(&inff[slot]), (ui / UI) & 1);
        const Info* f = &sinfo[slot];
        const int kind = f->kind, ntok = f->r.ntok, nkb = f->nkb;
        mbar_arrive(sa(&infe[slot]));
        if (kind < 0) break;
        const int b = ui & 1;
        if (ui >= 2) mbar_wait(sa(&acce[b]), ((ui >> 1) - 1) & 1);
        tc_fence_after();
        const int npad = (ntok + 15) & ~15;
        const uint32_t dacc = tmem + b * ACCW;
        const uint32_t id0 = idesc_mx(npad, 0, 0), id2 = idesc_mx(npad, 2, 2);
        for (int kb = 0; kb < nkb; kb += KSB, k++) {
          const int s = k % S;
          mbar_wait(sa(&full[s]), (k / S) & 1);
          fence_proxy_async();                                 // cp.async (generic proxy) writes -> tcgen05 reads
          tc_fence_after();
          if (ntok > 0) {
            const uint32_t sfa_col = tmem + 2 * ACCW + (k & 1) * 16, sfb_col = sfa_col + 8;
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
    }
  } else {
    // ---------------------------------------------------------------- epilogue (warps 6-9; TMEM lane quarter = warp % 4)
    // Tile row p (TMEM lane) of a 128-row weight tile, trtllm-gen MajorK shuffled layout (32-row blocks):
    //   FC1 (w13, gate/up interleaved): lane l of quarter q: a = l >> 3 (even: up, odd: gate), channel
    //       rb * 64 + q * 16 + 2 * (l & 7) + (a >> 1); its partner (the other half of SwiGLU) is lane l ^ 8.
    //   FC2 (w2): hidden channel rb * 128 + q * 32 + 4 * (l & 7) + (l >> 3).
    griddep_wait();
    const int q = warp & 3, ew = warp - 6, et = tid - 192;
    const int a = lane >> 3;
    const bool isgate = a & 1;
    for (int ui = 0; live; ui++) {
      const int slot = ui % UI;
      mbar_wait(sa(&inff[slot]), (ui / UI) & 1);
      const Info* f = &sinfo[slot];
      const int kind = f->kind;
      if (kind < 0) { named_bar(2, 128); if (ew == 0 && lane == 0) mbar_arrive(sa(&infe[slot])); break; }
      const int ntok = f->r.ntok, rb = f->rb, slot0 = f->r.slot0, gid = f->r.gid;
      const short* js = f->r.j;
      const int b = ui & 1;
      mbar_wait(sa(&accf[b]), (ui >> 1) & 1);
      tc_fence_after();
      const int npad = (ntok + 15) & ~15;
      const uint32_t taddr = tmem + ((uint32_t)(q * 32) << 16) + b * ACCW;
      for (int c0 = 0; c0 < npad; c0 += 16) {
        float v[16];
        tmem_ld16(taddr + c0, v);
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
          if ((lane & 0x17) == 0) {
#pragma unroll
            for (int jj = 0; jj < 8; jj++) sam[(q * 2 + (int)isgate) * 8 + jj] = am[jj];
          }
          named_bar(2, 128);
#pragma unroll
          for (int jj = 0; jj < 8; jj++) am[jj] = fmaxf(am[jj], sam[((q ^ 1) * 2 + (int)isgate) * 8 + jj]);
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
          named_bar(2, 128);                                   // sam is reused by the next chunk
        } else {
          // FC2: stage the bf16 [16 tokens][128 channels] chunk in SMEM, then 16-B coalesced row stores
          const int hl = q * 32 + 4 * (lane & 7) + a;
#pragma unroll
          for (int j = 0; j < 16; j++) sstage[j * 128 + hl] = __float2bfloat16(v[j]);
          named_bar(2, 128);
#pragma unroll
          for (int c = et; c < 256; c += 128) {
            const int j = c >> 4, part = c & 15, col = c0 + j;
            if (col < ntok)
              *(uint4*)(p.out + (size_t)js[col] * H + rb * 128 + part * 8) = *(const uint4*)(sstage + j * 128 + part * 8);
          }
          named_bar(2, 128);
        }
      }
      if (kind == 0 && ntok > 0) {
        __threadfence();
        named_bar(2, 128);
        if (ew == 0 && lane == 0) red_release(&p.st->fc1done[gid], 1u);
      }
      tc_fence_before();
      named_bar(2, 128);
      if (ew == 0 && lane == 0) { mbar_arrive(sa(&acce[b])); mbar_arrive(sa(&infe[slot])); }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 5) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem), "r"(TMEM_COLS) : "memory");
  }
  if (tid == 0) {
    if (p.dbg) {
      p.dbg[blockIdx.x * 4 + 0] = t_start; p.dbg[blockIdx.x * 4 + 2] = gtime();
      p.dbg[blockIdx.x * 4 + 3] = (u64)get_smid() | ((u64)d << 16) | ((u64)rank << 24);
    }
    // the last CTA resets the ranks and FC1 counters for the next call (every other CTA is past all its reads)
    __threadfence();
    const unsigned old = atomicAdd(&p.st->done, 1u);
    if (old == gridDim.x - 1) {
      __threadfence();
      for (int i = 0; i < 2 * GMAX; i++) p.st->fc1done[i] = 0;
      p.st->rank[0] = 0; p.st->rank[1] = 0;
      __threadfence();
      p.st->done = 0;
    }
  }
}

constexpr int SMEM_BYTES = 1024 + S * (A_ST + B_ST + 2 * SF_ST) + (2 * S + 4 + 2 * UI) * 8 + UI * (int)sizeof(Info) + 64 * 4 +
                           4 * 4 + 4 * 4 + (SCHED_MAX + 4) * 4 + 16 * 128 * 2;

// ------------------------------------------------------------------ host
static bool g_init[64];

void init(int64_t dev) {
  if (g_init[dev]) return;
  c10::cuda::CUDAGuard guard((int)dev);
  RTC(cudaFuncSetAttribute(k_moe, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
  int occ = 0;
  RTC(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_moe, NTHREADS, SMEM_BYTES));
  TORCH_CHECK(occ == 1, "k_moe occupancy ", occ, " (expected 1 CTA/SM)");
  cudaFuncAttributes fa;                    // load both functions now (lazy loading must not happen in graph capture)
  RTC(cudaFuncGetAttributes(&fa, k_plan));
  RTC(cudaFuncSetAttribute(k_plan, cudaFuncAttributeMaxDynamicSharedMemorySize, PLAN_SMEM));
  g_init[dev] = true;
}

int64_t plan_bytes() { return (int64_t)sizeof(Plan); }
int64_t state_bytes() { return (int64_t)sizeof(State); }
int64_t smem_bytes() { return SMEM_BYTES; }

static void encode(CUtensorMap* m, const torch::Tensor& w, uint64_t rows, uint64_t cols) {
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.element_size() == 1 && (uint64_t)w.numel() == rows * cols);
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstride[1] = {cols};
  cuuint32_t box[2] = {128, 128};
  cuuint32_t estride[2] = {1, 1};
  DRV(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, w.data_ptr(), gdim, gstride, box, estride,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

torch::Tensor tensor_maps(torch::Tensor w13, torch::Tensor w2) {
  auto t = torch::empty({2 * (int64_t)sizeof(CUtensorMap)}, torch::dtype(torch::kUInt8));
  CUtensorMap* m = (CUtensorMap*)t.data_ptr();
  encode(&m[0], w13, (uint64_t)E * 2 * I, H);
  encode(&m[1], w2, (uint64_t)E * H, I);
  return t;
}

static cudaLaunchAttribute pdl_attr(bool pdl) {
  cudaLaunchAttribute a; memset(&a, 0, sizeof(a));
  a.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a.val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  return a;
}

void plan_launch(torch::Tensor ids, torch::Tensor tw, torch::Tensor ew, torch::Tensor plan, int64_t T, int64_t n0, int64_t n1, bool pdl) {
  TORCH_CHECK(T > 0 && T <= TMAX && ids.dtype() == torch::kInt32 && tw.dtype() == torch::kFloat32 && ew.dtype() == torch::kBFloat16);
  TORCH_CHECK(ids.is_contiguous() && tw.is_contiguous() && ew.is_contiguous() && plan.numel() >= (int64_t)sizeof(Plan));
  c10::cuda::CUDAGuard guard(ids.device());
  cudaLaunchConfig_t cfg; memset(&cfg, 0, sizeof(cfg));
  cfg.gridDim = dim3(1); cfg.blockDim = dim3(1024); cfg.dynamicSmemBytes = PLAN_SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute at[1] = {pdl_attr(pdl)};
  cfg.attrs = at; cfg.numAttrs = 1;
  RTC(cudaLaunchKernelEx(&cfg, k_plan, (const int*)ids.data_ptr(), (const float*)tw.data_ptr(), (__nv_bfloat16*)ew.data_ptr(),
                         (Plan*)plan.data_ptr(), (int)T, (int)n0, (int)n1));
}

void moe_launch(torch::Tensor tmaps, torch::Tensor w13sf, torch::Tensor w2sf, torch::Tensor x, torch::Tensor xsf,
                torch::Tensor inter, torch::Tensor intersf, torch::Tensor out, torch::Tensor plan, torch::Tensor state,
                torch::Tensor smdom, int64_t n0, int64_t n1, int64_t nsm, std::optional<torch::Tensor> dbg, bool pdl, bool sf_dmajor,
                std::optional<torch::Tensor> blk13, std::optional<torch::Tensor> blk2, std::optional<torch::Tensor> xrep) {
  TORCH_CHECK(tmaps.device().is_cpu() && tmaps.numel() == 2 * (int64_t)sizeof(CUtensorMap));
  TORCH_CHECK(x.is_contiguous() && x.size(1) == H && xsf.is_contiguous() && xsf.numel() == x.size(0) * (H / 32));
  TORCH_CHECK(out.is_contiguous() && out.size(0) == x.size(0) * TOPK && out.size(1) == H && out.dtype() == torch::kBFloat16);
  TORCH_CHECK(w13sf.numel() == (int64_t)E * W13SF_E && w2sf.numel() == (int64_t)E * W2SF_E);
  TORCH_CHECK(n0 >= 64 && n1 >= 64 && n0 + n1 == nsm && smdom.numel() == nsm);
  c10::cuda::CUDAGuard guard(x.device());
  CUtensorMap m[2];
  memcpy(m, tmaps.data_ptr(), sizeof(m));
  Params prm;
  prm.sf_dmajor = sf_dmajor ? 1 : 0;
  prm.blk13 = blk13.has_value() ? (const uint8_t*)blk13->data_ptr() : nullptr;
  prm.blk2 = blk2.has_value() ? (const uint8_t*)blk2->data_ptr() : nullptr;
  if (xrep.has_value()) {             // [2][T * (H + H / 32)] uint8: per domain, x then x_sf
    TORCH_CHECK(xrep->numel() >= 2 * x.size(0) * (H + H / 32));
    const int64_t half = xrep->numel() / 2;
    for (int dd = 0; dd < 2; dd++) {
      prm.xd[dd] = (const uint8_t*)xrep->data_ptr() + dd * half;
      prm.xsfd[dd] = prm.xd[dd] + x.size(0) * H;
    }
  } else {
    prm.xd[0] = prm.xd[1] = (const uint8_t*)x.data_ptr();
    prm.xsfd[0] = prm.xsfd[1] = (const uint8_t*)xsf.data_ptr();
  }
  prm.w13sf = (const uint8_t*)w13sf.data_ptr(); prm.w2sf = (const uint8_t*)w2sf.data_ptr();
  prm.x = (const uint8_t*)x.data_ptr(); prm.xsf = (const uint8_t*)xsf.data_ptr();
  // inter / intersf: [2][...] (one slot space per domain, e.g. localized) or [1][...] (shared by both domains)
  TORCH_CHECK(inter.dim() == 2 && intersf.dim() == 2 && inter.size(0) == intersf.size(0) && inter.size(0) <= 2);
  TORCH_CHECK(inter.size(1) >= (int64_t)(TMAX * TOPK + 7 * E) * I && intersf.size(1) >= (int64_t)(TMAX * TOPK + 7 * E) * (I / 32));
  for (int dd = 0; dd < 2; dd++) {
    const int64_t r = inter.size(0) == 2 ? dd : 0;
    prm.interd[dd] = (uint8_t*)inter.data_ptr() + r * inter.stride(0);
    prm.intersfd[dd] = (uint8_t*)intersf.data_ptr() + r * intersf.stride(0);
  }
  prm.out = (__nv_bfloat16*)out.data_ptr(); prm.plan = (const Plan*)plan.data_ptr(); prm.st = (State*)state.data_ptr();
  prm.smdom = (const signed char*)smdom.data_ptr(); prm.nd0 = (int)n0; prm.nd1 = (int)n1;
  prm.dbg = dbg.has_value() ? (u64*)dbg->data_ptr() : nullptr;
  TORCH_CHECK(!dbg.has_value() || dbg->numel() * dbg->element_size() >= nsm * 32);
  cudaLaunchConfig_t cfg; memset(&cfg, 0, sizeof(cfg));
  cfg.gridDim = dim3((unsigned)nsm); cfg.blockDim = dim3(NTHREADS); cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute at[1] = {pdl_attr(pdl)};
  cfg.attrs = at; cfg.numAttrs = 1;
  RTC(cudaLaunchKernelEx(&cfg, k_moe, m[0], m[1], prm));
}

}  // namespace locmoe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &locmoe::init);
  m.def("plan_bytes", &locmoe::plan_bytes);
  m.def("state_bytes", &locmoe::state_bytes);
  m.def("smem_bytes", &locmoe::smem_bytes);
  m.def("tensor_maps", &locmoe::tensor_maps);
  m.def("plan_launch", &locmoe::plan_launch);
  m.def("moe_launch", &locmoe::moe_launch);
}
"""
