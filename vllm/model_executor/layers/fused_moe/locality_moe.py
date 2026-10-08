# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Locality-domain ("per-die") MXFP8 MoE for decode on Rubin-class GPUs.

``VLLM_MOE_LOCALITY_KERNEL=1`` (default off) replaces the router GEMM, the
shared expert and the trtllm-gen MXFP8 MoE call (routing + FC1 + FC2,
deferred finalize) of decode batches with ONE persistent kernel, ``k_moe``,
whose CTAs read only expert weights that live in their own locality domain's
HBM:

- Placement (:func:`place_expert_pairs`): each expert's w13 (``[1024, 2048]``
  e4m3 = exactly one 2 MiB chunk) and each expert pair's w2 (2 x 1 MiB) are
  moved, in place of the original tensors and in the same trtllm-gen MajorK
  shuffled layout, to locality domain ``(e >> 1) & 1``. The trtllm-gen path
  (mixed / prefill steps) keeps reading the same tensors. The shared expert
  gets a copy in the same layout on each domain (:func:`place_shared_expert`,
  6 MB per layer); its own linear layers keep their weights.
- One CTA per SM (``%smid`` -> ``Topology.sm_domain`` gives its domain and an
  arrival rank in it); warp 0 streams weight tiles (SW128 tensor-map TMA plus
  1-D bulk copies of the 128x4-interleaved scale atoms) into a 5 x 32 KB ring,
  warps 1-2 gather token rows (``cp.async``), warps 3-4 route, warp 5 issues
  tcgen05 MMAs, warps 6-9 run the epilogues.
- Routing (no separate kernels, no grid barrier):
  1. router GEMM ``x_bf16 @ w_router^T``: split-K tcgen05 bf16 MMAs, one work
     item (128-expert half, K slice, token tile) per CTA; the w_router slice is
     loaded before ``griddepcontrol.wait``, the x slice right after it. fp32
     partials go to a scratch buffer and a per-tile counter is released.
  2. per token (one warp of a CTA holding one of the tile's items): the K-slice
     partials are summed in a fixed order and rounded to bf16 (the logits);
     softmax, top-8 and renormalization replicate FlashInfer's
     RenormalizeNaive block-per-token routing kernel bit for bit
     (``-use_fast_math`` arithmetic, CUB block-reduce order, lower expert id
     wins ties). The token is appended to its experts' lists (atomics) and a
     per-tile "routed" counter is released. The same warp computes the shared
     expert's gate logit (``x_bf16 . w_gate``).
  3. every CTA waits for the routed counters, reads the 128 per-expert counts
     of its domain and derives its schedule: its static "wave-0" FC1 unit, a
     static round-robin share of the domain's FC1 units, then FC2 units from a
     per-domain work queue (FC2 waits on a per-group FC1 counter).
- Shared expert: tokens in 32-token groups, alternate groups per domain. Its
  FC1 units need no routing, so CTAs run a static round-robin share of them
  while the routing completes, then stream their wave-0 unit into the ring;
  its FC2 units head the domain's FC2 work queue (no CTA waits on another
  CTA's FC1 before it has scheduled its routed units).
- Output: :class:`UnfinalizedMoEOutput` with ``gemm2_permuted[T * 8, H]``,
  ``expert_weights[T, 8]`` (bf16) and the constant
  ``expanded_idx_to_permuted_idx = arange(T * 8)`` (the DLC-8 deferred-finalize
  contract), plus the ungated shared-expert output ``[T, H]`` and its gate
  logits ``[T, 1]`` for the consumer norm.

Shapes are fixed to Qwen3.6-35B-A3B: E = 256, top-8, H = 2048, I = 512, MXFP8
(1x32 UE8M0), shared expert I = 512. Decode batches of 17 ... 512 tokens
(smaller batches use FlashInfer's warp-per-token routing arithmetic, which this
kernel does not replicate).

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
SE_SLOTS = MAX_TOKENS // 2  # shared-expert rows per domain (alternate 32-token groups)
CHUNK = 2 << 20
SK_MAX = 32  # router K slices

# VLLM_MOE_LOCALITY_KERNEL=1: place MXFP8 trtllm-gen expert weights by expert
# pair and serve deferred-finalize decode calls with MIN_TOKENS <= T <=
# MAX_TOKENS by this kernel, router GEMM and shared expert included (acts
# inside the MoE custom op: no compile-key change).
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


def unswizzle_mxfp8_scale(s: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """[n, k / 32] UE8M0 scales from FlashInfer's F8_128x4 swizzled layout
    (``swizzle_mxfp8_scale``: [n / 128, k / 128, 32, 4, 4]).
    """
    ks = k // 32
    assert n % 128 == 0 and ks % 4 == 0 and s.numel() == n * ks
    v = s.reshape(n // 128, ks // 4, 32, 4, 4).view(torch.uint8)
    return v.permute(0, 3, 2, 1, 4).reshape(n, ks).contiguous()


@dataclasses.dataclass
class SharedWeights:
    """The shared expert in the routed experts' trtllm-gen layout, one copy per
    locality domain (domain d's rows at d * 1024 of ``w13`` and at d * 4096 of
    ``w2``), plus its gate weight.
    """

    w13: torch.Tensor  # [2048, 2048] e4m3
    w2: torch.Tensor  # [8192, 512] e4m3 (rows 2048..4095 of each half unused)
    s13: torch.Tensor  # [2, 1024 * 64] uint8
    s2: torch.Tensor  # [2, 2048 * 16] uint8
    w_gate: torch.Tensor  # [2048] bf16
    tmaps: torch.Tensor  # host bytes of the w13 / w2 tensor maps


def place_shared_expert(
    w_gu: torch.Tensor,
    s_gu: torch.Tensor,
    w_down: torch.Tensor,
    s_down: torch.Tensor,
    w_gate: torch.Tensor,
) -> SharedWeights:
    """Per-domain copies of the shared expert from canonical MXFP8 weights:
    gate_up [1024, 2048] e4m3 (gate rows first) with [1024, 64] UE8M0 scales,
    down [2048, 512] with [2048, 16] scales, gate [1, 2048] (or [2048]) bf16.
    """
    from vllm.model_executor.layers.locality.memory import alloc_chunks, chunk_ordinals
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        _shuffle_mxfp8_moe_weights,
        swap_w13_to_w31,
    )

    assert w_gu.shape == (2 * INTER, HID) and w_down.shape == (HID, INTER)
    dev = w_gu.device.index
    p13, p2, ps13, ps2 = _shuffle_mxfp8_moe_weights(
        swap_w13_to_w31(w_gu[None].contiguous()),
        w_down[None].contiguous(),
        swap_w13_to_w31(s_gu.view(torch.uint8).reshape(1, 2 * INTER, HID // 32)),
        s_down.view(torch.uint8).reshape(1, HID, INTER // 32),
        True,
    )
    out = []
    for w in (p13, p2):
        flat = alloc_chunks(2 * CHUNK, [0, 1], CHUNK, dev)
        if chunk_ordinals(flat, 2 * CHUNK, CHUNK) != [0, 1]:
            raise RuntimeError("locality placement mismatch (shared expert)")
        flat.zero_()
        n = w.numel()
        for d in (0, 1):
            flat[d * CHUNK : d * CHUNK + n].copy_(w.reshape(-1).view(torch.uint8))
        out.append(flat.view(torch.float8_e4m3fn))
    l13 = out[0].view(2 * 2 * INTER, HID)
    l2 = out[1].view(2 * CHUNK // INTER, INTER)
    s13 = ps13.reshape(1, -1).view(torch.uint8).expand(2, -1).contiguous()
    s2 = ps2.reshape(1, -1).view(torch.uint8).expand(2, -1).contiguous()
    ext = load()
    return SharedWeights(
        l13,
        l2,
        s13,
        s2,
        w_gate.reshape(HID).to(torch.bfloat16).contiguous(),
        ext.tensor_maps(l13, l2),
    )


@dataclasses.dataclass
class LayerWeights:
    w13: torch.Tensor
    w2: torch.Tensor
    s13: torch.Tensor  # w13 scales (production 128x4-interleaved tensors)
    s2: torch.Tensor
    tmaps: torch.Tensor  # host bytes of the w13 / w2 tensor maps
    shared: SharedWeights | None = None
    # router weight data_ptr -> host bytes of the w13, w2, w_router tensor maps
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
        # (FC1 and FC2 of a group run in the same domain); shared-expert rows
        # after the routed ones
        rows = MAX_SLOTS + SE_SLOTS
        ib = -(-rows * INTER // CHUNK) * CHUNK
        sb = -(-rows * (INTER // 32) // CHUNK) * CHUNK
        self.inter = alloc_chunks(
            2 * ib, [0] * (ib // CHUNK) + [1] * (ib // CHUNK), CHUNK, device
        ).view(2, ib)
        self.inter_sf = alloc_chunks(
            2 * sb, [0] * (sb // CHUNK) + [1] * (sb // CHUNK), CHUNK, device
        ).view(2, sb)
        self.idx = torch.arange(MAX_TOKENS * TOPK, dtype=torch.int32, device=dev)
        self.smdom = topo.sm_domain

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
        shared: bool = True,
        dbg: torch.Tensor | None = None,
        pdl: bool = True,
        logits: torch.Tensor | None = None,
    ):
        """Returns (gemm2_permuted [T*8, H] bf16, expert_weights [T, 8] bf16,
        expanded_idx_to_permuted_idx [T, 8] int32, topk_ids [T, 8] int32,
        shared [T, H] bf16 or None, shared gate logits [T, 1] bf16 or None).
        ``shared`` runs the layer's shared expert (``lw.shared``) too.
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
        sw = lw.shared if shared else None
        s_out = s_gate = None
        if sw is not None:
            s_out = torch.empty(T, HID, dtype=torch.bfloat16, device=dev)
            s_gate = torch.empty(T, 1, dtype=torch.bfloat16, device=dev)
        self.ext.moe_launch(
            self._maps(lw, w_router),
            lw.s13,
            lw.s2,
            None if sw is None else sw.tmaps,
            None if sw is None else sw.s13,
            None if sw is None else sw.s2,
            None if sw is None else sw.w_gate,
            s_out,
            s_gate,
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
            self.nd[0],
            self.nd[1],
            sk,
            nt,
            dbg,
            logits,
            pdl,
        )
        return out, ew, self.idx[: T * TOPK].view(T, TOPK), ids, s_out, s_gate


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


def is_placed(w13: torch.Tensor, w2: torch.Tensor) -> bool:
    """Whether (w13, w2) are a layer pair-placed by :func:`maybe_place`."""
    return (w13.data_ptr(), w2.data_ptr()) in _LAYERS


def _mxfp8_linear_weights(linear) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(e4m3 [N, K], UE8M0 [N, K / 32]) of a FlashInfer MXFP8 linear layer
    (weight [N, K] or its [K, N] transpose view; F8_128x4-swizzled scales).
    """
    w = getattr(linear, "weight", None)
    s = getattr(linear, "weight_scale", None)
    if w is None or s is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2:
        return None
    n = getattr(linear, "output_size_per_partition", None)
    k = getattr(linear, "input_size_per_partition", None)
    if n is None or k is None:
        return None
    if tuple(w.shape) == (k, n) and w.t().is_contiguous():
        w = w.t()
    if tuple(w.shape) != (n, k) or not w.is_contiguous() or s.numel() != n * k // 32:
        return None
    return w, unswizzle_mxfp8_scale(s, n, k)


def attach_shared_expert(w13: torch.Tensor, w2: torch.Tensor, mlp) -> bool:
    """Give the pair-placed layer (w13, w2) a per-domain copy of its shared
    expert ``mlp`` (gate_up_proj / down_proj MXFP8 linears + expert_gate), so
    the kernel runs it too. Call before CUDA-graph capture (kernel warmup).
    False when the layer is not placed or the shared expert has another form.
    """
    lw = _LAYERS.get((w13.data_ptr(), w2.data_ptr()))
    if lw is None:
        return False
    if lw.shared is not None:
        return True
    gate = getattr(mlp, "expert_gate", None)
    gu = _mxfp8_linear_weights(getattr(mlp, "gate_up_proj", None))
    dn = _mxfp8_linear_weights(getattr(mlp, "down_proj", None))
    gw = getattr(gate, "weight", None)
    if gu is None or dn is None or gw is None or gw.numel() != HID:
        return False
    if gu[0].shape != (2 * INTER, HID) or dn[0].shape != (HID, INTER):
        return False
    lw.shared = place_shared_expert(gu[0], gu[1], dn[0], dn[1], gw)
    return True


def try_apply(
    x_bf16: torch.Tensor,
    w_router: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    shared: bool,
):
    """(gemm2_permuted, expert_weights, expanded_idx_to_permuted_idx,
    (shared output, shared gate logits) or None) of the whole MoE block (router
    GEMM included; the shared expert too when ``shared``), or None when this
    call is not served: T outside the gate, layer not placed, ``shared``
    requested but not attached, or input layouts other than bf16 [T, H] router
    input + [T, H] e4m3 + linear [T, H/32] UE8M0.
    """
    T = x.shape[0]
    if not GATE_MIN_TOKENS <= T <= GATE_MAX_TOKENS:
        return None
    lw = _LAYERS.get((w13.data_ptr(), w2.data_ptr()))
    if lw is None or (shared and lw.shared is None):
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
    out, ew, idx, _, s_out, s_gate = runtime(x.device.index).forward(
        x_bf16, w_router, x, x_sf, lw, shared=shared
    )
    return out, ew, idx, (s_out, s_gate) if shared else None


_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
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
constexpr int SE = E;                           // expert id of the shared expert in unit records
constexpr int SE_GMAX = TMAX / 64;              // shared-expert groups per domain (alternate 32-token groups)
constexpr int GMAX = 256 + SE_GMAX;             // groups per domain: routed (expert, 32-token sub-block), then shared
constexpr int SE_SLOT0 = TMAX * TOPK + 7 * E;   // first shared-expert row of a domain's intermediate
constexpr int NTHREADS = 320, NGATHER = 64;     // warps 1-2 gather (128 gather threads measured no faster)
constexpr int W_ROUTE = 3, W_MMA = 5, W_EPI = 6;  // warps 3-4 route, 5 issues MMAs, 6-9 epilogue (warp % 4 = TMEM quarter)
constexpr int TMEM_COLS = 512, ACCW = 32, NACC = 4;   // 4 MoE accumulators of N <= 32 columns, then SF columns
constexpr int RCOL = 256;                       // router accumulator: TMEM columns [256, 256 + nt)
constexpr int RX_EXTRA = 8192;                  // SMEM after the B slots: router operands (48 KB window), then the schedule
constexpr int NTILE = 8;                        // router token tiles (T <= 512)
constexpr int W13SF_E = 2 * I * H / 32, W2SF_E = H * I / 32;  // scale bytes per expert (128x4-interleaved)
constexpr int SE_W2_ROWS = 4096;                // rows per domain copy of the shared expert's w2 (2 MiB chunk / 512 B)

// One unit's group (expert, sub-block of <= 32 tokens): token entries t * 8 + k (shared expert: t * 8).
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
  const uint8_t* sw13sf; const uint8_t* sw2sf;   // shared expert scales [2][W13SF_E] / [2][W2SF_E]
  const __nv_bfloat16* swg;           // shared expert gate weight [H]
  __nv_bfloat16* sout; __nv_bfloat16* sgate;     // shared expert output [T, H] (ungated), gate logits [T]
  const __nv_bfloat16* xb;            // router input [T, H] bf16
  const uint8_t* x; const uint8_t* xsf;   // MXFP8 activation [T, H] e4m3, linear [T, H / 32] UE8M0
  uint8_t* interd[2]; uint8_t* intersfd[2];   // FC1 -> FC2 intermediate of domain d's groups
  __nv_bfloat16* out; __nv_bfloat16* ew; int* ids; __nv_bfloat16* logits;
  State* st; short* list; float* part;
  const signed char* smdom;
  int nd0, nd1, T, sk, nt, ntiles, nitems, nseg;   // nseg: shared-expert 32-token groups (0: no shared expert)
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
// shared-expert units of domain d: FC1 units (8 per group) then FC2 units (16 per group) of its groups (2 j + d);
// unit m of the list -> code type << 24 | sub(j) << 8 | rb (type 5 FC1, 6 FC2)
__device__ __forceinline__ int se_code(int m, int ngd) {
  return m < 8 * ngd ? ((5 << 24) | ((m >> 3) << 8) | (m & 7)) : ((6 << 24) | (((m - 8 * ngd) >> 4) << 8) | ((m - 8 * ngd) & 15));
}

// ------------------------------------------------------------------ routing (one warp per token)
// FlashInfer 0.6.18 RenormalizeNaive, block-per-token kernel (routingIndicesBlockScoresKernel; used for E = 256 and
// T >= 17): SoftmaxPreprocess::applyToSmem (256 threads, thread e = expert e, cub::BlockReduce<float, 256>), the
// packed-key top-K (value descending, lower expert id first) and SumNormalizePostprocess; FlashInfer builds with
// -use_fast_math, so expf / 1/x / division are the approximate .ftz forms. Lane L holds experts 32 i + L, i.e.
// "virtual warp" i of that block, so the warp sums below see the CUB operands in the CUB order.
__device__ __forceinline__ float f_sub_ftz(float a, float b) { float r; asm("sub.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_mul_ftz(float a, float b) { float r; asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_ex2_ftz(float a) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_rcp_ftz(float a) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float f_div_ftz(float a, float b) { float r; asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_max_ftz(float a, float b) { float r; asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ unsigned twiddle(float v) {            // cub TwiddleIn of the float bits
  const unsigned b = __float_as_uint(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float untwiddle(unsigned b) { return __uint_as_float((b & 0x80000000u) ? (b & 0x7fffffffu) : ~b); }

// qs (diagnostics, first token of a warp): [15] partials loaded, [16] routing math done, [17] list slots returned,
// [18] token released
__device__ void route_token(const Params& p, int t, int n, int lane, u64* qs) {
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
  for (int i = 0; i < 8; i++) sc[i] = __bfloat162float(__float2bfloat16_rn(sc[i]));
  // softmax (SoftmaxPreprocess::applyToSmem): block max (order-free), e = expf(s - max), block sum = CUB
  // WarpReduceShfl::Sum of each virtual warp (offsets 1, 2, 4, 8, 16; lanes past the end keep their value) then the
  // 8 warp aggregates in order, 1 / sum, p = e * (1 / sum). The 8 shuffle trees are interleaved.
  float mx = -INFINITY;
#pragma unroll
  for (int i = 0; i < 8; i++) mx = fmaxf(sc[i], mx);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float ex[8], a[8];
#pragma unroll
  for (int i = 0; i < 8; i++) { ex[i] = f_ex2_ftz(f_mul_ftz(f_sub_ftz(sc[i], mx), 1.4426950408889634f)); a[i] = ex[i]; }
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; i++) { const float y = __shfl_down_sync(0xffffffffu, a[i], o); a[i] = lane + o < 32 ? y + a[i] : a[i]; }
  }
  float bsum = a[0];
#pragma unroll
  for (int i = 1; i < 8; i++) bsum = bsum + a[i];
  const float inv = f_rcp_ftz(__shfl_sync(0xffffffffu, bsum, 0));
  // top-8 by packed key (twiddled value, 65535 - id): each lane sorts its 8 keys, then 8 rounds of a two-step
  // redux.max (value bits, then id bits among the lanes holding that value), the owner advancing its head.
  unsigned kv[8], ki[8];
#pragma unroll
  for (int i = 0; i < 8; i++) { kv[i] = twiddle(f_mul_ftz(ex[i], inv)); ki[i] = 65535u - (unsigned)(i * 32 + lane); }
#pragma unroll
  for (int x = 0; x < 8; x++)
#pragma unroll
    for (int y = 0; y < 7 - x; y++) {
      const bool sw = kv[y] < kv[y + 1] || (kv[y] == kv[y + 1] && ki[y] < ki[y + 1]);
      const unsigned tv = sw ? kv[y + 1] : kv[y], ti = sw ? ki[y + 1] : ki[y];
      kv[y + 1] = sw ? kv[y] : kv[y + 1]; ki[y + 1] = sw ? ki[y] : ki[y + 1];
      kv[y] = tv; ki[y] = ti;
    }
  unsigned mv = 0, mi = 0;
#pragma unroll
  for (int r = 0; r < TOPK; r++) {
    const unsigned bv = __reduce_max_sync(0xffffffffu, kv[0]);
    const unsigned bi = __reduce_max_sync(0xffffffffu, kv[0] == bv ? ki[0] : 0u);
    if (lane == r) { mv = bv; mi = bi; }
    if (kv[0] == bv && ki[0] == bi) {
#pragma unroll
      for (int i = 0; i < 7; i++) { kv[i] = kv[i + 1]; ki[i] = ki[i + 1]; }
      kv[7] = 0u; ki[7] = 0u;
    }
  }
  if (qs && lane == 0) qs[16] = gtime() + (mv == 1u);
  // SumNormalizePostprocess (lane k holds the k-th score)
  const float v = lane < TOPK ? untwiddle(mv) : 0.f;
  const float sum = cg::reduce(cg::tiled_partition<32>(cg::this_thread_block()), v, cg::plus<float>());
  const int e = 65535 - (int)mi;
  if (lane < TOPK) {
    const unsigned pos = atomicAdd(&p.st->cnt[e], 1u);
    p.list[e * TMAX + pos] = (short)(t * TOPK + lane);
    if (qs && lane == 0) qs[17] = gtime() + (pos > 4096u);
  }
  // the release orders this warp's list stores (bar.warp.sync) before the routed count
  __syncwarp();
  if (lane == 0) { red_release(&p.st->tdone[n * 32], 1u); if (qs) qs[18] = gtime(); }
  if (lane < TOPK) {
    p.ids[t * TOPK + lane] = e;
    p.ew[t * TOPK + lane] = __float2bfloat16_rn(f_div_ftz(v, f_max_ftz(sum, 1e-20f)));
  }
  if (p.logits) {
#pragma unroll
    for (int i = 0; i < 8; i++) p.logits[(size_t)t * E + i * 32 + lane] = __float2bfloat16_rn(sc[i]);
  }
}

// shared-expert gate logit of token t: x_bf16[t] . w_gate (fp32), rounded to bf16
__device__ void se_gate(const Params& p, int t, int lane) {
  const uint4* xr = (const uint4*)(p.xb + (size_t)t * H);
  const uint4* wr = (const uint4*)p.swg;
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < H / 256; i++) {
    const uint4 xv = __ldcg(xr + i * 32 + lane), wv = __ldg(wr + i * 32 + lane);
    const __nv_bfloat162* xh = (const __nv_bfloat162*)&xv;
    const __nv_bfloat162* wh = (const __nv_bfloat162*)&wv;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float2 a = __bfloat1622float2(xh[j]), b = __bfloat1622float2(wh[j]);
      acc = fmaf(a.x, b.x, acc); acc = fmaf(a.y, b.y, acc);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) p.sgate[t] = __float2bfloat16_rn(acc);
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
                                                     const __grid_constant__ CUtensorMap tmr, const __grid_constant__ CUtensorMap tmx,
                                                     const __grid_constant__ CUtensorMap tms13, const __grid_constant__ CUtensorMap tms2,
                                                     const Params p) {
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
  volatile int* sdom = (volatile int*)(sam + 128);       // [0] served domain, [1] rank in it, [3] last-CTA flag
  uint32_t* stmem = (uint32_t*)(sdom + 4);
  int* ssched = (int*)(stmem + 4);                       // this CTA's static units
  __nv_bfloat16* sstage = (__nv_bfloat16*)(ssched + SCHED_MAX + 4);   // FC2 epilogue: [16 tokens][128 ch] bf16

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const u64 t_start = gtime();
  if (tid == 0) {
    for (int s = 0; s < S; s++) { mbar_init(sa(&full[s]), 1 + NGATHER); mbar_init(sa(&empty[s]), 1); }
    for (int b = 0; b < NACC; b++) { mbar_init(sa(&accf[b]), 1); mbar_init(sa(&acce[b]), 4); }
    for (int i = 0; i < UI; i++) { mbar_init(sa(&inff[i]), 1); mbar_init(sa(&infe[i]), 3); }
    mbar_init(sa(rfull), 1); mbar_init(sa(racc), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == W_MMA) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(sa(stmem)), "r"(TMEM_COLS) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *stmem;
  // router item of this CTA: token tile rn, local index rj = 2 * K slice + expert half
  const int item = (int)blockIdx.x < p.nitems ? (int)blockIdx.x : -1;
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
    // Rank in this SM's domain by arrival order. If the domain already has all its ranks (an SM ran a second CTA),
    // take a rank of the other domain, so every rank is served exactly once. (Own state: allowed before the wait.)
    int d = 0, rank = 0;
    if (lane == 0) {
      d = p.smdom[get_smid()];
      d = d < 0 ? 0 : d;
      unsigned r = atomicAdd(&p.st->rank[d], 1u);
      if (r >= (unsigned)(d ? p.nd1 : p.nd0)) { d ^= 1; r = atomicAdd(&p.st->rank[d], 1u); }
      rank = (int)r;
      sdom[0] = d; sdom[1] = rank;                       // read by the other roles after their first unit record
    }
    d = __shfl_sync(0xffffffffu, d, 0); rank = __shfl_sync(0xffffffffu, rank, 0);
    const int nd = d ? p.nd1 : p.nd0;
    const bool live = rank < nd;
    const int k0 = rank >> 3, rb0 = rank & 7, e0 = expert_of(d, k0);
    // static shared-expert units of this rank: FC1 entries rank + i nd of the domain's list (no routing needed, they
    // run in the routing window); the shared expert's FC2 units head the domain's FC2 work queue, so no CTA waits on
    // another CTA's FC1 before it has scheduled its routed units
    const int ngd = p.nseg > d ? (p.nseg - d + 1) >> 1 : 0, nse_tot = 8 * ngd;
    const int nse = live && rank < nse_tot ? (nse_tot - rank + nd - 1) / nd : 0;
    // weights of a unit: (tensor map, scales, first row, K128 blocks)
    auto unit_src = [&](int ty, int kk, int sub, int rb, const CUtensorMap*& tm, const uint8_t*& sf, int& row0, int& nkb) {
      if (ty == 5) { tm = &tms13; sf = p.sw13sf + (size_t)d * W13SF_E + rb * FC1_NKB * SF_ATOM; row0 = d * 2 * I + rb * 128; nkb = FC1_NKB; }
      else if (ty == 6) { tm = &tms2; sf = p.sw2sf + (size_t)d * W2SF_E + rb * FC2_NKB * SF_ATOM; row0 = d * SE_W2_ROWS + rb * 128; nkb = FC2_NKB; }
      else {
        const int e = expert_of(d, kk);
        if (ty == 2) { tm = &tm2; sf = p.w2sf + (size_t)e * W2SF_E + rb * FC2_NKB * SF_ATOM; row0 = e * H + rb * 128; nkb = FC2_NKB; }
        else { tm = &tm13; sf = p.w13sf + (size_t)e * W13SF_E + rb * FC1_NKB * SF_ATOM; row0 = e * 2 * I + rb * 128; nkb = FC1_NKB; }
      }
    };
    uint32_t k = 0;                                      // ring stages issued
    int pre0 = 0;                                        // K128 blocks of the first unit issued before the wait
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"((u64)&tm13) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"((u64)&tm2) : "memory");
      if (hasr) {
        asm volatile("prefetch.tensormap [%0];" :: "l"((u64)&tmr) : "memory");
        asm volatile("prefetch.tensormap [%0];" :: "l"((u64)&tmx) : "memory");
        mbar_expect_tx(sa(rfull), natom * 16384);
        for (int a = 0; a < natom; a++) tma_2d(sa(sRw + a * 16384), &tmr, rs * kc + a * 64, rm * 128, sa(rfull));
      }
      if (live) {
        // the first unit's leading stages: its first shared-expert unit, else the wave-0 unit
        const int c = nse ? se_code(rank, ngd) : ((1 << 24) | (k0 << 16) | rb0);
        const CUtensorMap* tm; const uint8_t* sf; int row0, nkb;
        unit_src(c >> 24, (c >> 16) & 0xff, (c >> 8) & 0xff, c & 0xff, tm, sf, row0, nkb);
        for (; pre0 < nkb && k < (uint32_t)S; pre0 += KSB, k++)
          issue_stage(sa(sA + k * A_ST), sa(sSFA + k * SF_ST), sa(&full[k]), tm, sf, row0, pre0);
      }
    }
    griddep_wait();
    if (q_ && lane == 0) q_[1] = gtime();
    if (lane == 0 && hasr) {
      // the x slice: tokens [r_t0, r_t0 + nt) (rows past T are zero-filled), columns [rs * kc, + kc)
      mbar_expect(sa(rfull), natom * p.nt * 128);
      for (int a = 0; a < natom; a++) tma_2d(sa(sRx + a * p.nt * 128), &tmx, rs * kc + a * 64, r_t0, sa(rfull));
      if (q_) q_[11] = gtime();
    }
    int pub = 0;
    long long c_pempty = 0;
    // publish the record of unit `pub` (code c); routed units also bulk-copy their token entries
    int* s_cnt = (int*)sRX; int* s_f0 = s_cnt + 128; int* s_g0 = s_f0 + 132; int* s_so = s_g0 + 132;
    auto publish = [&](int c, int kb0, bool list_sent) {
      const int slot = pub % UI;
      if (pub >= UI) mbar_wait(sa(&infe[slot]), ((pub / UI) - 1) & 1);
      Info* f = &sinfo[slot];
      if (c < 0) {
        f->kind = -1;
        mbar_arrive(sa(&inff[slot]));
      } else {
        const int ty = c >> 24, kk = (c >> 16) & 0xff, sub = (c >> 8) & 0xff, rb = c & 0xff;
        f->rb = rb; f->kb0 = kb0; f->r.sub = sub;
        if (ty >= 5) {                                   // shared expert: group g = 2 sub + d, all its tokens
          const int g = 2 * sub + d, nt_ = min(32, p.T - 32 * g);
          f->kind = ty == 6; f->nkb = ty == 6 ? FC2_NKB : FC1_NKB;
          f->r.e = SE; f->r.ntok = nt_; f->r.slot0 = SE_SLOT0 + 32 * sub; f->r.gid = d * GMAX + 256 + sub;
#pragma unroll
          for (int q = 0; q < 32; q++) f->r.j[q] = (short)((32 * g + q) * TOPK);
          mbar_arrive(sa(&inff[slot]));
        } else {
          const int e = expert_of(d, kk);
          f->kind = ty == 2; f->nkb = ty == 2 ? FC2_NKB : FC1_NKB; f->r.e = e;
          if (ty == 4) {                                 // idle wave-0 unit: consume the stages already issued
            f->nkb = kb0; f->r.ntok = 0; f->r.slot0 = 0; f->r.gid = -1;
            mbar_arrive(sa(&inff[slot]));
          } else {
            f->r.ntok = min(32, s_cnt[kk] - 32 * sub); f->r.slot0 = s_so[kk] + 32 * sub; f->r.gid = d * GMAX + s_g0[kk] + sub;
            if (list_sent) {
              mbar_arrive(sa(&inff[slot]));
            } else {
              mbar_expect(sa(&inff[slot]), 64u);
              bulk_load(sa(f->r.j), p.list + (size_t)e * TMAX + 32 * sub, 64u, sa(&inff[slot]));
            }
          }
        }
      }
      pub++;
    };
    // stream the stages [kb0, nkb) of a unit
    auto stream = [&](int c, int kb0) {
      if ((c >> 24) == 4) return;                        // idle wave-0 unit: nothing beyond the stages issued
      const CUtensorMap* tm; const uint8_t* sf; int row0, nkb;
      unit_src(c >> 24, (c >> 16) & 0xff, (c >> 8) & 0xff, c & 0xff, tm, sf, row0, nkb);
      for (int kb = kb0; kb < nkb; kb += KSB, k++) {
        const int s = k % S;
        if (k >= (uint32_t)S) TW(c_pempty, mbar_wait(sa(&empty[s]), ((k / S) - 1) & 1));
        issue_stage(sa(sA + s * A_ST), sa(sSFA + s * SF_ST), sa(&full[s]), tm, sf, row0, kb);
      }
    };
    // (1) shared-expert units (no routing needed), then the wave-0 unit's leading stages
    int w0 = 0;                                          // K128 blocks of the wave-0 unit issued before its record
    if (lane == 0 && live) {
      for (int i = 0; i < nse && i < 2; i++) publish(se_code(rank + i * nd, ngd), i == 0 ? pre0 : 0, false);
      for (int i = 0; i < nse; i++) {
        stream(se_code(rank + i * nd, ngd), i == 0 ? pre0 : 0);
        if (i + 2 < nse) publish(se_code(rank + (i + 2) * nd, ngd), 0, false);
      }
      const int c0 = (1 << 24) | (k0 << 16) | rb0;
      const CUtensorMap* tm; const uint8_t* sf; int row0, nkb;
      unit_src(1, k0, 0, rb0, tm, sf, row0, nkb);
      w0 = nse ? 0 : pre0;
      for (int n_ = 0; w0 < nkb && n_ < S - (nse ? 0 : pre0 / KSB); w0 += KSB, k++, n_++) {
        const int s = k % S;
        if (k >= (uint32_t)S) TW(c_pempty, mbar_wait(sa(&empty[s]), ((k / S) - 1) & 1));
        issue_stage(sa(sA + s * A_ST), sa(sSFA + s * SF_ST), sa(&full[s]), tm, sf, row0, w0);
      }
      (void)c0;
    }
    w0 = __shfl_sync(0xffffffffu, w0, 0);
    // (2) every token routed: poll the per-tile counters (one lane per tile)
    {
      const int need = lane < p.ntiles ? min(p.nt, p.T - lane * p.nt) : 0;
      for (;;) {
        const unsigned v = lane < p.ntiles ? ld_acquire(&p.st->tdone[lane * 32]) : 0u;
        if (__all_sync(0xffffffffu, (int)v >= need)) break;
        __nanosleep(20);
      }
      __syncwarp();
    }
    asm volatile("fence.proxy.async.global;" ::: "memory");   // token lists (generic stores, acquired) -> bulk copies
    if (q_ && lane == 0) q_[4] = gtime();
    // the wave-0 unit's token entries, while the counts load (its record is published below)
    if (lane == 0 && live) {
      if (pub >= UI) mbar_wait(sa(&infe[pub % UI]), ((pub / UI) - 1) & 1);
      mbar_expect_tx(sa(&inff[pub % UI]), 64u);
      bulk_load(sa(sinfo[pub % UI].r.j), p.list + (size_t)e0 * TMAX, 64u, sa(&inff[pub % UI]));
    }
    // (3) schedule from the per-expert counts of this domain (lane: local experts 4 lane .. 4 lane + 3): wave-0 unit,
    // the static round-robin share of the domain's FC1 list (expert order; sub-block 0 without its wave-0 row blocks),
    // then FC2 units (group order, 16 row blocks each) from the domain's work queue. Code: type << 24 | k << 16 |
    // sub << 8 | rb; type 1 FC1, 2 FC2, 3 wave-0 FC1, 4 idle wave-0.
    int nstat = 0, ng = 0;
    if (live) {
      int c[4], f[4], g[4], o[4], sf_ = 0, sg_ = 0, so_ = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int kk = 4 * lane + j;
        c[j] = (int)ld_relaxed(&p.st->cnt[expert_of(d, kk)]);
        const int ns = (c[j] + 31) >> 5, sk0 = ns ? min(max(nd - 8 * kk, 0), 8) : 0;
        f[j] = ns ? 8 * ns - sk0 : 0; g[j] = ns; o[j] = (c[j] + 7) & ~7;
        sf_ += f[j]; sg_ += g[j]; so_ += o[j];
      }
      if (q_ && lane == 0) q_[19] = gtime() + (c[0] < 0);
      int xf = warp_iscan(sf_) - sf_, xg = warp_iscan(sg_) - sg_, xo = warp_iscan(so_) - so_;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int kk = 4 * lane + j;
        s_cnt[kk] = c[j]; s_f0[kk] = xf; s_g0[kk] = xg; s_so[kk] = xo;
        xf += f[j]; xg += g[j]; xo += o[j];
      }
      if (lane == 31) { s_f0[128] = xf; s_g0[128] = xg; s_so[128] = xo; }
      __syncwarp();
      const int nfc1 = s_f0[128];
      ng = s_g0[128];
      nstat = rank < nfc1 ? (nfc1 - rank + nd - 1) / nd : 0;
      nstat = nstat < SCHED_MAX - 1 ? nstat : SCHED_MAX - 1;
      for (int i = lane; i < nstat; i += 32) {
        const int m = rank + i * nd, kk = find_pre(s_f0, m), off = m - s_f0[kk];
        const int sk0 = min(max(nd - 8 * kk, 0), 8), w0_ = 8 - sk0;
        const int sub = off < w0_ ? 0 : 1 + ((off - w0_) >> 3), rb = off < w0_ ? sk0 + off : ((off - w0_) & 7);
        ssched[1 + i] = (1 << 24) | (kk << 16) | (sub << 8) | rb;
      }
      if (lane == 0) ssched[0] = ((s_cnt[k0] > 0 ? 3 : 4) << 24) | (k0 << 16) | rb0;
      nstat += 1;
    }
    __syncwarp();
    if (q_ && lane == 0) q_[20] = gtime();
    // (4) routed units: records two ahead, FC2 entries from the queue one record ahead of their use; queue entries
    // [0, 16 ngd) are the shared expert's FC2 units, then the routed FC2 units in group order
    if (lane == 0) {
      int ndyn = 0, nrt = 0;
      bool fin = false;
      const unsigned nse2 = 16u * (unsigned)ngd, n2 = nse2 + 16u * (unsigned)ng;
      unsigned* q2 = &p.st->q2[d * 32];
      unsigned nxt = n2;
      int codes[UI];                                     // codes of the published, not yet streamed routed units
      auto next_code = [&]() -> int {
        int c = -1;
        if (nrt < nstat) {
          c = ssched[nrt];
          if (nrt == nstat - 1) nxt = atomicAdd(q2, 1u);
        } else if (nxt < nse2) {
          c = se_code(nse_tot + (int)nxt, ngd);
          nxt = atomicAdd(q2, 1u);
          ndyn++;
        } else if (nxt < n2) {
          const unsigned r = nxt - nse2;
          const int gg = (int)(r >> 4), kk = find_pre(s_g0, gg);
          c = (2 << 24) | (kk << 16) | ((gg - s_g0[kk]) << 8) | (int)(r & 15);
          nxt = atomicAdd(q2, 1u);
          ndyn++;
        }
        nrt++;
        return c;
      };
      auto pub_routed = [&]() {
        const int c = live ? next_code() : -1;
        codes[pub % UI] = c;
        publish(c, nrt == 1 ? w0 : 0, nrt == 1);
        if (c < 0) fin = true;
      };
      pub_routed();
      if (!fin) pub_routed();
      if (q_) q_[8] = gtime();
      for (int u = pub - (fin ? 1 : 2);; u++) {
        const int c = codes[u % UI];
        if (c < 0) break;
        stream(c, u == nse ? w0 : 0);
        if (!fin) pub_routed();
      }
      if (q_) { q_[26] = c_pempty; q_[30] = nstat; q_[31] = ndyn; q_[29] = nse; }
    }
  } else if (warp < W_ROUTE) {
    // ---------------------------------------------------------------- gather (NGATHER threads)
    // B rows by 16-B cp.async.cg (L2), SFB words by 4-B cp.async.ca (per-expert slot ranges are 8-row aligned, so no
    // 128-B line of the intermediate scales mixes two groups); per-thread, per-stage arrival by
    // cp.async.mbarrier.arrive.noinc. Row ids are read once per unit.
    griddep_wait();
    const int gt = tid - 32;
    if (q_ && gt == 0) q_[10] = gtime();
    constexpr int CPR = KSB * 8;                           // 16-B chunks per B row per stage
    constexpr int MAXC = (NMAX * CPR + NGATHER - 1) / NGATHER;
    constexpr int MAXF = (NMAX * KSB + NGATHER - 1) / NGATHER;
    uint32_t k = 0;
    long long c_gdep = 0, c_gempty = 0;
    int d = 0;
    if (hasr) mbar_wait(sa(racc), 0);                    // the router MMA has read its operands in the B slots
    for (int ui = 0;; ui++) {
      const int slot = ui % UI;
      mbar_wait(sa(&inff[slot]), (ui / UI) & 1);
      if (ui == 0) d = sdom[0];
      const Info* f = &sinfo[slot];
      const int kind = f->kind;
      if (kind < 0) { named_bar(1, NGATHER); if (gt == 0) mbar_arrive(sa(&infe[slot])); break; }
      const int ntok = f->r.ntok, nkb = f->nkb;
      const uint8_t* bsrc = kind ? p.interd[d] : p.x;
      const uint8_t* bsf = kind ? p.intersfd[d] : p.xsf;
      const int bstride = kind ? I : H, bsfstride = kind ? I / 32 : H / 32;
      const uint8_t* csrc[MAXC]; uint32_t cdst[MAXC]; int nc = 0;
      const uint8_t* fsrc[MAXF]; uint32_t fdst[MAXF]; int nf = 0;
#pragma unroll
      for (int i = 0; i < MAXC; i++) {
        const int c = gt + NGATHER * i;
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
        const int c = gt + NGATHER * i;
        if (c < ntok * KSB) {
          const int row = c / KSB, blk = c % KSB;
          const int srow = kind ? f->r.slot0 + row : (f->r.j[row] >> 3);
          fsrc[i] = bsf + (size_t)srow * bsfstride + blk * 4;
          fdst[i] = blk * SF_ATOM + (row & 31) * 16 + (row >> 5) * 4;
          nf = i + 1;
        }
      }
      const int gid = f->r.gid;
      named_bar(1, NGATHER);                               // every gather thread is done with the slot
      if (gt == 0) mbar_arrive(sa(&infe[slot]));
      if (kind == 1) {
        // FC2 reads the FC1 outputs of this group (written by other CTAs): wait for its 8 FC1 units.
        if (gt == 0) { TW(c_gdep, while (ld_acquire(&p.st->fc1done[gid]) < (unsigned)FC1_NRB) __nanosleep(40)); }
        named_bar(1, NGATHER);
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
  } else if (warp < W_MMA) {
    // ---------------------------------------------------------------- routing (64 threads)
    // tokens rj + 2 sk w of this item's tile (the CTAs holding the tile's items cover the whole tile); the shared
    // expert's gate logit first, while the router partials complete
    griddep_wait();
    if (hasr) {
      for (int u = rj + 2 * p.sk * (warp - W_ROUTE); u < r_nv; u += 4 * p.sk) {
        if (p.nseg) se_gate(p, r_t0 + u, lane);
        if (lane == 0) while (ld_acquire(&p.st->ctr[rn * 32]) < (unsigned)(2 * p.sk)) __nanosleep(20);
        __syncwarp();
        u64* qs = (q_ && warp == W_ROUTE && u == rj) ? q_ : nullptr;
        if (qs && lane == 0) qs[14] = gtime();
        route_token(p, r_t0 + u, rn, lane, qs);
      }
      if (q_ && warp == W_ROUTE && lane == 0) q_[6] = gtime();
    }
  } else if (warp == W_MMA) {
    // ---------------------------------------------------------------- MMA issuer
    if (lane == 0 && hasr) {
      // router: [128 experts] x [r_np tokens] over this K slice, kind::f16 (K = 16 per instruction: +32 B in the atom)
      mbar_wait(sa(rfull), 0);
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
    if (lane == 0) {
      uint32_t k = 0;
      long long c_macce = 0, c_mfull = 0;
      const u64 ad0 = sdesc_sw128(sa(sA)), bd0 = sdesc_sw128(sa(sB));
      const u64 fa0 = sdesc_sf(sa(sSFA)), fb0 = sdesc_sf(sa(sSFB));
      if (hasr) mbar_wait(sa(racc), 0);                  // the router MMA has read sB before any B slot is reused
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
      if (q_) { q_[28] = c_macce; q_[25] = c_mfull; }
    }
  } else {
    // ---------------------------------------------------------------- epilogue (warps 6-9; TMEM lane quarter = warp % 4)
    // Tile row p (TMEM lane) of a 128-row weight tile, trtllm-gen MajorK shuffled layout (32-row blocks):
    //   FC1 (w13, gate/up interleaved): lane l of quarter q: a = l >> 3 (even: up, odd: gate), channel
    //       rb * 64 + q * 16 + 2 * (l & 7) + (a >> 1); its partner (the other half of SwiGLU) is lane l ^ 8.
    //   FC2 (w2): hidden channel rb * 128 + q * 32 + 4 * (l & 7) + (l >> 3).
    griddep_wait();
    const int q = warp & 3, ew = warp - W_EPI;
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
    int d = 0;
    for (int ui = 0;; ui++) {
      const int slot = ui % UI;
      mbar_wait(sa(&inff[slot]), (ui / UI) & 1);
      if (ui == 0) d = sdom[0];
      const Info* f = &sinfo[slot];
      const int kind = f->kind;
      if (kind < 0) { named_bar(2, 128); if (ew == 0 && lane == 0) mbar_arrive(sa(&infe[slot])); break; }
      const int ntok = f->r.ntok, rb = f->rb, slot0 = f->r.slot0, gid = f->r.gid;
      const bool se = f->r.e == SE;
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
      // one code path per unit kind: a per-element `se` branch inside the unrolled loops costs ~3 µs (C512 CTA time)
      auto chunks = [&]<bool SEU>() {
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
            float g = isgate ? v[8 + jj] : pr[jj], up = isgate ? pr[8 + jj] : v[jj];
            if constexpr (SEU) {
              // the shared expert's production chain: bf16 gate_up GEMM output, then silu(g) * up as Inductor lowers
              // it (g / (exp(-g) + 1) * up, fp32), rounded to bf16 before the MXFP8 quant (silu_mul_mxfp8_quant)
              g = __bfloat162float(__float2bfloat16_rn(g)); up = __bfloat162float(__float2bfloat16_rn(up));
              h[jj] = __bfloat162float(__float2bfloat16_rn(g / (expf(-g) + 1.f) * up));
            } else {
              h[jj] = __fdividef(g, 1.f + __expf(-g)) * up;
            }
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
            if constexpr (SEU) {
              // FlashInfer's MXFP8 rule (the production SiLU*mul quant): sf = ceil(log2(amax / 448)) by the bits
              const float nrm = am[jj] * (1.f / 448.f);
              const unsigned bits = __float_as_uint(nrm), ex = (bits >> 23) & 255u, man = bits & 0x7FFFFFu;
              const unsigned bump = (man != 0u) && !(ex == 0u && man <= 0x400000u);
              e8 = nrm <= 0.f ? 0 : (int)min(ex + bump, 254u);
              inv = e8 == 0 ? 0.f : __uint_as_float((unsigned)(254 - e8) << 23);
            } else if (am[jj] > 0.f) {
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
          // own SMEM slice and store 64-B row segments (16-B per lane); only __syncwarp, no cross-warp barrier.
          // Routed rows go to gemm2_permuted row t * 8 + k, shared-expert rows to its output row t.
          __nv_bfloat16* sw = sstage + q * 16 * 32;
          const int hl = 4 * (lane & 7) + a;
#pragma unroll
          for (int j = 0; j < 16; j++) sw[j * 32 + hl] = __float2bfloat16(v[j]);
          __syncwarp();
          __nv_bfloat16* obase = SEU ? p.sout : p.out;
#pragma unroll
          for (int c = lane; c < 64; c += 32) {
            const int j = c >> 2, part = c & 3, col = c0 + j;
            if (col < ntok) {
              const int row = SEU ? (js[col] >> 3) : js[col];
              *(uint4*)(obase + (size_t)row * H + rb * 128 + q * 32 + part * 8) = *(const uint4*)(sw + j * 32 + part * 8);
            }
          }
          __syncwarp();
        }
      }
      };
      if (se) chunks.template operator()<true>(); else chunks.template operator()<false>();
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
  if (warp == W_MMA) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem), "r"(TMEM_COLS) : "memory");
  }
  // the last CTA resets the call state for the next call (every other CTA is past all its reads of it)
  if (tid == 0) {
    if (q_) { q_[0] = t_start; q_[2] = gtime(); q_[3] = (u64)get_smid() | ((u64)sdom[0] << 16) | ((u64)sdom[1] << 24) | ((u64)(item + 1) << 40); }
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

static void encode(CUtensorMap* m, const torch::Tensor& w, CUtensorMapDataType dt, uint64_t rows, uint64_t cols,
                   uint32_t box_cols, uint32_t box_rows) {
  const uint64_t es = (uint64_t)w.element_size();
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && (uint64_t)w.numel() == rows * cols);
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstride[1] = {cols * es};
  cuuint32_t box[2] = {box_cols, box_rows};
  cuuint32_t estride[2] = {1, 1};
  DRV(cuTensorMapEncodeTiled(m, dt, 2, w.data_ptr(), gdim, gstride, box, estride,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// w13 / w2 maps of a layer (routed experts [E * 1024, 2048] / [E * 2048, 512]; the shared expert's per-domain copies
// [2048, 2048] / [8192, 512] alike)
torch::Tensor tensor_maps(torch::Tensor w13, torch::Tensor w2) {
  TORCH_CHECK(w13.element_size() == 1 && w2.element_size() == 1 && w13.size(-1) == H && w2.size(-1) == I);
  auto t = torch::empty({2 * (int64_t)sizeof(CUtensorMap)}, torch::dtype(torch::kUInt8));
  CUtensorMap* m = (CUtensorMap*)t.data_ptr();
  encode(&m[0], w13, CU_TENSOR_MAP_DATA_TYPE_UINT8, (uint64_t)w13.numel() / H, H, 128, 128);
  encode(&m[1], w2, CU_TENSOR_MAP_DATA_TYPE_UINT8, (uint64_t)w2.numel() / I, I, 128, 128);
  return t;
}

torch::Tensor router_map(torch::Tensor wr) {
  TORCH_CHECK(wr.dtype() == torch::kBFloat16 && wr.size(0) == E && wr.size(1) == H);
  auto t = torch::empty({(int64_t)sizeof(CUtensorMap)}, torch::dtype(torch::kUInt8));
  encode((CUtensorMap*)t.data_ptr(), wr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, E, H, 64, 128);
  return t;
}

static cudaLaunchAttribute pdl_attr(bool pdl) {
  cudaLaunchAttribute a; memset(&a, 0, sizeof(a));
  a.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a.val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  return a;
}

void moe_launch(torch::Tensor tmaps, torch::Tensor w13sf, torch::Tensor w2sf, std::optional<torch::Tensor> smaps,
                std::optional<torch::Tensor> sw13sf, std::optional<torch::Tensor> sw2sf, std::optional<torch::Tensor> swg,
                std::optional<torch::Tensor> sout, std::optional<torch::Tensor> sgate, torch::Tensor xb, torch::Tensor x,
                torch::Tensor xsf, torch::Tensor inter, torch::Tensor intersf, torch::Tensor out, torch::Tensor ew,
                torch::Tensor ids, torch::Tensor state, torch::Tensor list, torch::Tensor part, torch::Tensor smdom,
                int64_t n0, int64_t n1, int64_t sk, int64_t nt, std::optional<torch::Tensor> dbg,
                std::optional<torch::Tensor> logits, bool pdl) {
  const int64_t T = x.size(0);
  TORCH_CHECK(tmaps.device().is_cpu() && tmaps.numel() == 3 * (int64_t)sizeof(CUtensorMap));
  TORCH_CHECK(T >= 1 && T <= TMAX && x.is_contiguous() && x.size(1) == H && xsf.is_contiguous() && xsf.numel() == T * (H / 32));
  TORCH_CHECK(xb.dtype() == torch::kBFloat16 && xb.is_contiguous() && xb.size(0) == T && xb.size(1) == H);
  TORCH_CHECK(out.is_contiguous() && out.size(0) == T * TOPK && out.size(1) == H && out.dtype() == torch::kBFloat16);
  TORCH_CHECK(ew.dtype() == torch::kBFloat16 && ew.numel() == T * TOPK && ids.dtype() == torch::kInt32 && ids.numel() == T * TOPK);
  TORCH_CHECK(w13sf.numel() == (int64_t)E * W13SF_E && w2sf.numel() == (int64_t)E * W2SF_E);
  TORCH_CHECK(n0 >= 64 && n1 >= 64 && n0 + n1 == smdom.numel());
  TORCH_CHECK((sk == 16 && nt == 64) || (sk == 32 && nt == 192));
  const int64_t ntiles = (T + nt - 1) / nt, nitems = 2 * sk * ntiles;
  TORCH_CHECK(ntiles <= NTILE && nitems <= n0 + n1, "router split ", sk, "x", nt, " does not fit T=", T);
  TORCH_CHECK(state.numel() >= (int64_t)sizeof(State) && list.numel() >= (int64_t)E * TMAX && part.numel() >= sk * T * E);
  TORCH_CHECK(inter.dim() == 2 && intersf.dim() == 2 && inter.size(0) == 2 && intersf.size(0) == 2);
  TORCH_CHECK(inter.size(1) >= (int64_t)(SE_SLOT0 + TMAX / 2) * I && intersf.size(1) >= (int64_t)(SE_SLOT0 + TMAX / 2) * (I / 32));
  const bool shared = smaps.has_value();
  TORCH_CHECK(shared == (sw13sf.has_value() && sw2sf.has_value() && swg.has_value() && sout.has_value() && sgate.has_value()));
  c10::cuda::CUDAGuard guard(x.device());
  CUtensorMap m[6];
  memset(m, 0, sizeof(m));
  memcpy(m, tmaps.data_ptr(), 3 * sizeof(CUtensorMap));
  // the router input map (this call's x_bf16; box [nt rows][64 columns], rows past T zero-filled)
  encode(&m[3], xb, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, (uint64_t)T, H, 64, (uint32_t)nt);
  Params prm;
  memset(&prm, 0, sizeof(prm));
  if (shared) {
    TORCH_CHECK(smaps->device().is_cpu() && smaps->numel() == 2 * (int64_t)sizeof(CUtensorMap));
    TORCH_CHECK(sw13sf->numel() == 2 * (int64_t)W13SF_E && sw2sf->numel() == 2 * (int64_t)W2SF_E);
    TORCH_CHECK(swg->dtype() == torch::kBFloat16 && swg->numel() == H && swg->is_contiguous());
    TORCH_CHECK(sout->dtype() == torch::kBFloat16 && sout->is_contiguous() && sout->numel() == T * H);
    TORCH_CHECK(sgate->dtype() == torch::kBFloat16 && sgate->is_contiguous() && sgate->numel() == T);
    memcpy(&m[4], smaps->data_ptr(), 2 * sizeof(CUtensorMap));
    prm.sw13sf = (const uint8_t*)sw13sf->data_ptr(); prm.sw2sf = (const uint8_t*)sw2sf->data_ptr();
    prm.swg = (const __nv_bfloat16*)swg->data_ptr();
    prm.sout = (__nv_bfloat16*)sout->data_ptr(); prm.sgate = (__nv_bfloat16*)sgate->data_ptr();
    prm.nseg = (int)((T + 31) / 32);
  }
  prm.w13sf = (const uint8_t*)w13sf.data_ptr(); prm.w2sf = (const uint8_t*)w2sf.data_ptr();
  prm.xb = (const __nv_bfloat16*)xb.data_ptr();
  prm.x = (const uint8_t*)x.data_ptr(); prm.xsf = (const uint8_t*)xsf.data_ptr();
  for (int dd = 0; dd < 2; dd++) {
    prm.interd[dd] = (uint8_t*)inter.data_ptr() + dd * inter.stride(0);
    prm.intersfd[dd] = (uint8_t*)intersf.data_ptr() + dd * intersf.stride(0);
  }
  prm.out = (__nv_bfloat16*)out.data_ptr(); prm.ew = (__nv_bfloat16*)ew.data_ptr(); prm.ids = (int*)ids.data_ptr();
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
  cudaLaunchAttribute at[1] = {pdl_attr(pdl)};
  cfg.attrs = at; cfg.numAttrs = 1;
  RTC(cudaLaunchKernelEx(&cfg, k_moe, m[0], m[1], m[2], m[3], m[4], m[5], prm));
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
