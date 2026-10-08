# SPDX-License-Identifier: Apache-2.0
# Vendored from FlashInfer v0.6.18.post1 flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py (#4081,
# Apache-2.0). Local change: a separate per-row RING index tensor (gRidx) addresses the k/u/g
# history rings, so the rings can live in a per-request-slot side buffer while the checkpoint state stays in the
# paged vLLM state pool (gH0 / gH0idx = page block). Everything else is unchanged; the vLLM entry point is
# adapter.py (tvm-ffi compiled launch, no FlashInfer wrapper).
"""Gated Delta-Net (GDN) decode kernel with a history ring AND per-request
state flush.

Implements the ReplaySSM chunked delta-rule decode for speculative decoding.
This is a drop-in superset of the verify-only kernel
(``gdn_decode_bf16_wy_ucache.py``): one kernel serves every iteration. Each
CTA reads its request's fill level P = hist_len[b] and either

  - P <  flush_min: runs the verify path — computes the T draft-token
    outputs and appends the T new (normed-k, u, G) entries to the ring
    (bit-identical outputs and appends to the verify kernel); or
  - P >= flush_min: additionally FOLDS the ring into the checkpoint state
    and RESTARTS the ring.

Target: CUDA tensor-core GPUs. Fixed K == V == 128.
Native draft length T in {4, 8}.

Flush semantics per (request b, v-head hv)
------------------------------------------
  S_h = e^{G_P} * S0 + sum_{j<P} w_j * u_j * k_j^T,   w_j = e^{G_P - G_j}
  - S_h is stored back to the state pool (S0 <- S_h);
  - the draft outputs y are computed via the SAME route as the verify path
    (the two factorizations are identical: hw = e^{G_P}(S0 x) +
    sum_j w_j u_j (k_j.x) == S_h x), so y matches the verify kernel;
  - the T current drafts are NOT folded: their fresh corrections U, normed
    k, and LOCAL cumulative log-decay (restarting at 0, since the reference
    checkpoint is now S_h) are appended at ring slots (base+P+s) & RING_MASK
    — PAST the fold-source window [base, base+P), so no CTA ever overwrites
    rows a sibling is still reading (ring semantics; RING_SLOTS = 32);
  - cursor commits (base slide, len reset) are CALLER-OWNED, outside the
    launch — identical to Triton commit_gdn_replayssm_spec: flush rows
    base' = (base+P) & RING_MASK, len' = accepted.
The fold runs at the CTA tail: re-stream S0 half-by-half via TMA, form
D = b_d @ khist via MMA strips (b_d = the w-scaled u tile; khist from a
register snapshot taken before the shared buffer is re-tenanted), then
S_h = bdec*S0 + D per element, stored straight to the pool (f32 accumulate,
one rounding to the state dtype).

Precisions
----------
Two element-type symbols, selected at IMPORT time via environment variables
(one dtype set per module load):

  Symbol  Dtype                     Selected by
  ------  ------------------------  ------------------------------------------
  IO      bf16 (default) | fp16     GDN_UCACHE_IO_DTYPE
  STATE   = IO (default) | fp16     GDN_UCACHE_STATE_DTYPE (fp16 requires IO=bf16)

  IO     covers q/k/v, a/b, the u and k rings, and the output.
  STATE  is the checkpoint pool only. STATE=fp16 with IO=bf16 is the MIXED
         mode: only the checkpoint carries fp16's extra mantissa bits, while
         inputs/rings/output stay bf16.

All GEMM accumulation and the gate / log-decay math run in f32 internally,
regardless of the IO/STATE dtypes.

Tensors (public entry point ``gated_delta_rule_mtp_ucache_flush``)
------------------------------------------------------------------
  Name                   Shape             Dtype   Dir      Meaning
  ---------------------  ----------------  ------  -------  ------------------------
  q, k                   [B, T, H,  K]     IO      in       draft query / key
  v                      [B, T, HV, V]     IO      in       draft values
  a, b                   [B, T, HV]        IO      in       per-token gate / beta
  A_log                  [HV]              bf16    in       per-head log-decay (cast+cached)
  dt_bias                [HV]              bf16    in       per-head time-step bias (cast+cached)
  initial_state_source   [pool, HV, V, K]  STATE   in/out   checkpoint S0 (written on flush)
  initial_state_indices  [B]               int32   in       per-request pool slot
  k_cache                [pool, H,  16, K]  IO     in/out   ring: L2-normalized keys
  u_cache                [pool, HV, 16, V]  IO     in/out   ring: correction vectors
  g_cache                [pool, HV, 16]    f32     in/out   ring: cumulative log-decay
  hist_len               [B]               int32   in       filled ring slots P per request
  output                 [B, T, HV, V]     IO      out      draft-token outputs (returned)

  Scalars: scale (float, default 1/sqrt(K)); flush_min (int, default
  W_RING - T + 1 = lazy flush). softplus_beta / softplus_threshold are FIXED
  (the kernel uses beta=1 and no threshold); the wrapper rejects any
  non-default value rather than silently ignoring it.

Ring tensors are pool-indexed via initial_state_indices and MUST be
zero-initialized at allocation. New entries are written speculatively to
slots [P, P+T) (verify path) or restarted at [0, T) (flush path); serving
code rewinds after verification by setting hist_len = P + accepted. Legal
hist_len at call time: [0, 16].

High-level flow
---------------
  1. Load and L2-normalize k, q into shared memory.
  2. Form the T x T Gram matrices K@K^T and Q@K^T.
  3. Build the WY transform Tmat via a block triangular solve.
  4. GEMM the state S0 against the packed [k; q] tile, add the history-ring
     contribution, and produce the token outputs.
  5. Compute the new corrections u and append (normed k, u, G) to the ring.
  6. On flush (P >= flush_min): fold the window into S0 and write it back;
     the caller's cursor commit then slides base past the folded window.

Implementation notes
--------------------
- The state tile is streamed with TMA into an SW128-swizzled shared buffer,
  in two K-halves to halve its shared-memory footprint.
- All SMEM->register loads for the MMAs use ldmatrix; MMA accumulators are
  written straight to SMEM (no separate C-staging buffer).
- One shared-memory region is reused for several tiles (q, ring keys, ring
  corrections, v) whose lifetimes do not overlap.
- The Gram / inverse / GEMM phases are specialized at compile time for the
  effective draft length T in {4, 8, 16}.
"""

import torch
import math
import os
import weakref
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
from cutlass import const_expr
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.arch import sync_threads
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp import MmaF16BF16Op
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as mlir_T


device = torch.device("cuda:0")

# Problem dimensions. One CTA processes a full V tile per (request, head).
T = 16
K_DIM = 128
V_DIM_C = 128  # full V tile per CTA
BK_H = 16  # K-tile for the H GEMM (multiple of 16 for mma.k=16)
EPS = 1e-6
# IO / activation / ring / state element type: bf16 (default) or fp16 via
# GDN_UCACHE_IO_DTYPE=fp16. fp16 carries 10 mantissa bits vs bf16's 7 (less
# state drift over long contexts); bf16-valued activations convert to fp16
# exactly while |x| <= 65504. Selected at IMPORT — the SMEM struct types, MMA
# operand type, and packed-pair PTX mnemonics below all specialize on it, so
# there is one dtype per module load (load the file again via importlib for a
# same-process cross-dtype comparison). Helper names keep their `bf16x2`
# suffix; in fp16 mode they operate on f16x2 pairs.
_IO_ENV = os.environ.get("GDN_UCACHE_IO_DTYPE", "bf16").strip().lower()
if _IO_ENV in ("fp16", "float16", "half"):
    io = cutlass.Float16
    IO_TORCH = torch.float16
    _CVT_F32_FROM_H = "cvt.f32.f16"  # unpack: one half -> f32
    _CVT_H2_FROM_F32 = "cvt.rn.f16x2.f32"  # pack: two f32 -> packed pair
    # raw-PTX tensor-core GEMMs (f16 and bf16 share the m16n8k16 fragment
    # layout, so ONLY the operand-type suffix changes)
    _MMA_M16N8K16_HH_F32 = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
elif _IO_ENV in ("bf16", "bfloat16"):
    io = cutlass.BFloat16
    IO_TORCH = torch.bfloat16
    _CVT_F32_FROM_H = "cvt.f32.bf16"
    _CVT_H2_FROM_F32 = "cvt.rn.bf16x2.f32"
    _MMA_M16N8K16_HH_F32 = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
else:
    raise ValueError(
        f"GDN_UCACHE_IO_DTYPE={_IO_ENV!r} unsupported: use 'bf16' or 'fp16'."
    )
# Checkpoint-state (pool) element type: defaults to the IO dtype. Set
# GDN_UCACHE_STATE_DTYPE=fp16 with bf16 IO for the MIXED mode: q/k/v inputs,
# u/k rings, and the output stay bf16; only the checkpoint is stored fp16.
# State-touching paths then run at higher fidelity: the H GEMM (A = q/k
# activations, B = state tile) converts its four shared A-fragments bf16->f16
# in registers (exact: every bf16 value is f16-representable in range) and
# issues .f16.f16 MMAs; the fold unpacks fp16 state pairs through f32 FMAs
# and repacks fp16.
_ST_ENV = os.environ.get("GDN_UCACHE_STATE_DTYPE", "").strip().lower()
if _ST_ENV in ("", _IO_ENV):
    state_ty = io
    ST_TORCH = IO_TORCH
    _ST_MIXED = False
elif _ST_ENV in ("fp16", "float16", "half") and io is cutlass.BFloat16:
    state_ty = cutlass.Float16
    ST_TORCH = torch.float16
    _ST_MIXED = True
else:
    raise ValueError(
        f"GDN_UCACHE_STATE_DTYPE={_ST_ENV!r} with IO={_IO_ENV!r} unsupported: "
        "state dtype must equal the IO dtype, or be 'fp16' with bf16 IO."
    )
if state_ty is cutlass.Float16:
    _CVT_ST2_FROM_F32 = "cvt.rn.f16x2.f32"  # pack two f32 -> state pair
    _MMA_H_F32 = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
else:
    _CVT_ST2_FROM_F32 = "cvt.rn.bf16x2.f32"
    _MMA_H_F32 = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
# _H_A_CVT_ASM (H-GEMM A-fragment bf16->f16 conversion) is defined BELOW,
# after _PACK_MIXED: in the combined state+cache-fp16 mode the packed tile is
# stored fp16 at norm time, so the H GEMM's A operand is already fp16 and no
# conversion is needed (the block becomes empty).
# Ring-cache (u/k) element type: defaults to the IO dtype. Set
# GDN_UCACHE_RING_DTYPE=fp16 with bf16 IO for the upstream-Triton-parity
# mode (the TRT-LLM PR #16464 / vLLM rule: fp16 rings under bf16
# activations). The rings hold L2-normed keys and bounded corrections —
# small dynamic range — so fp16's 10 mantissa bits beat bf16's 7 at
# the same element size. Dtype
# journey in the mixed mode, one rounding at each store:
#   u: f32 MMA accumulators -> cvt.rn.f16x2.f32 at the OutStage ring rows
#      (output rows stay IO dtype); ring appends copy the f16 bytes raw.
#   k: normed bf16 (the MMA operand) -> f16 repack at the append STG
#      (exact for in-range values; magnitudes <= 1).
#   g: fp32 always.
# Consumers: the u tile is consumed RAW f16 — the history contraction and
# the fold issue .f16.f16 MMAs with their A operands (w-scaled transposed
# scores / w-scaled u) staged f16. The khist tile is converted f16 -> bf16
# in place before the (bf16) scores GEMM, and the flush fold's STS-back
# repacks it f16 (exact in range), so the fold preserves the u values'
# fp16 fidelity while k stays at bf16 quantum (same as the bf16 mode).
_RING_ENV = os.environ.get("GDN_UCACHE_RING_DTYPE", "").strip().lower()
if _RING_ENV in ("", _IO_ENV):
    ring_ty = io
    RING_TORCH = IO_TORCH
    _RING_MIXED = False
    _CVT_RG2_FROM_F32 = _CVT_H2_FROM_F32
    _CVT_F32_FROM_RG = _CVT_F32_FROM_H
    _MMA_RING_F32 = _MMA_M16N8K16_HH_F32
elif _RING_ENV in ("fp16", "float16", "half") and io is cutlass.BFloat16:
    ring_ty = cutlass.Float16
    RING_TORCH = torch.float16
    _RING_MIXED = True
    _CVT_RG2_FROM_F32 = "cvt.rn.f16x2.f32"
    _CVT_F32_FROM_RG = "cvt.f32.f16"
    _MMA_RING_F32 = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
else:
    raise ValueError(
        f"GDN_UCACHE_RING_DTYPE={_RING_ENV!r} with IO={_IO_ENV!r} unsupported: "
        "ring dtype must equal the IO dtype, or be 'fp16' with bf16 IO."
    )
# ----- packed [k|q] tile element type ("pack_ty") -----
# In the COMBINED state+cache-fp16 mode BOTH GEMM partners of the packed tile
# are fp16 (the H GEMM's state operand and the scores GEMM's khist operand),
# so the normalized packed tile is stored fp16 DIRECTLY at norm time (fp32 ->
# f16, replacing fp32 -> bf16). This (a) gives the k-cache TRUE fp16 precision
# — normed k no longer round-trips through bf16 — and (b) eliminates the
# per-fragment bf16->f16 conversions in BOTH GEMMs, since
# the Grams / H GEMM / scores GEMM all read fp16 natively. Enabled ONLY when
# both knobs are fp16; single-knob and default paths keep pack_ty == io (bf16,
# byte-identical to before).
_PACK_MIXED = _ST_MIXED and _RING_MIXED
if _PACK_MIXED:
    pack_ty = cutlass.Float16
    _CVT_PK2_FROM_F32 = "cvt.rn.f16x2.f32"
    _CVT_F32_FROM_PK = "cvt.f32.f16"
    _MMA_PACK = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
else:
    pack_ty = io
    _CVT_PK2_FROM_F32 = _CVT_H2_FROM_F32
    _CVT_F32_FROM_PK = _CVT_F32_FROM_H
    _MMA_PACK = _MMA_M16N8K16_HH_F32
# scores-GEMM B-fragment bf16->f16 conversion: needed ONLY when the ring is
# fp16 but the packed tile is still bf16 (cache-fp16-ONLY mode). In combined
# mode the packed tile is already fp16 -> empty.
_RING_B_CVT_ASM = (
    " shl.b32 _wl, _b0, 16; and.b32 _wh, _b0, 0xFFFF0000;"
    " mov.b32 _fl, _wl; mov.b32 _fh, _wh; cvt.rn.f16x2.f32 _b0, _fh, _fl;"
    " shl.b32 _wl, _b1, 16; and.b32 _wh, _b1, 0xFFFF0000;"
    " mov.b32 _fl, _wl; mov.b32 _fh, _wh; cvt.rn.f16x2.f32 _b1, _fh, _fl;"
    if (_RING_MIXED and not _PACK_MIXED)
    else ""
)
# H-GEMM A-fragment bf16->f16 conversion: needed ONLY when the state is fp16
# but the packed tile is still bf16 (state-fp16-ONLY mode). In combined mode
# the packed tile is already fp16 -> empty.
_H_A_CVT_ASM = (
    "{ .reg .b32 _wl, _wh; .reg .f32 _fl, _fh;"
    " shl.b32 _wl, _a0, 16; and.b32 _wh, _a0, 0xFFFF0000;"
    " mov.b32 _fl, _wl; mov.b32 _fh, _wh; cvt.rn.f16x2.f32 _a0, _fh, _fl;"
    " shl.b32 _wl, _a1, 16; and.b32 _wh, _a1, 0xFFFF0000;"
    " mov.b32 _fl, _wl; mov.b32 _fh, _wh; cvt.rn.f16x2.f32 _a1, _fh, _fl;"
    " shl.b32 _wl, _a2, 16; and.b32 _wh, _a2, 0xFFFF0000;"
    " mov.b32 _fl, _wl; mov.b32 _fh, _wh; cvt.rn.f16x2.f32 _a2, _fh, _fl;"
    " shl.b32 _wl, _a3, 16; and.b32 _wh, _a3, 0xFFFF0000;"
    " mov.b32 _fl, _wl; mov.b32 _fh, _wh; cvt.rn.f16x2.f32 _a3, _fh, _fl; }"
    if (_ST_MIXED and not _PACK_MIXED)
    else ""
)
f32 = cutlass.Float32
WARP = 32
THREADS = 128
T_PAD = 16
W_RING = 16  # max history WINDOW rows (one 16-row MMA tile) — smem/tile constant
# Physical ring depth (Triton-ReplaySSM-compatible circular ring). The live
# window is [base, base+P) mod RING_SLOTS with P <= W_RING; appends land at
# (base+P+s) & RING_MASK — always PAST the window, so a flush never overwrites
# rows any sibling CTA is still reading (the old single-buffer restart race).
# Cursor commits (base slide / len reset) are CALLER-OWNED, outside the launch.
RING_SLOTS = 32
RING_MASK = RING_SLOTS - 1

# The state tile is streamed in two K-halves. Shared-memory rows are padded
# to avoid bank conflicts on ldmatrix and vectorized loads.
K_HALF = K_DIM // 2  # 64 — K-half streamed per TMA copy
K_PADDED = K_DIM + 8  # 136 — padded row stride for sK / sQ
V_PADDED = V_DIM_C + 8  # 136 — padded row stride for sV / sH (V rows, K cols)

TK = T * K_DIM
TK_PAD = T * K_PADDED
TT = T * T
BF_PAD = 24


# ---------------------------------------------------------------------------
# Small inline-PTX helpers used throughout the kernel.
# ---------------------------------------------------------------------------


def _smat_off(row, col):
    e = row * T + col
    return e ^ (
        ((e >> Int32(5)) & Int32(1)) | (((e >> Int32(6)) & Int32(1)) << Int32(3))
    )


def _ldmatrix_x4(smem_tensor, lane_id):
    addr = (
        smem_tensor.iterator.toint()
        + (lane_id % 16) * Int32(BF_PAD * 2)
        + (lane_id // 16) * Int32(16)
    )
    r = llvm.inline_asm(
        llvm.StructType.get_literal(
            [mlir_T.i32(), mlir_T.i32(), mlir_T.i32(), mlir_T.i32()]
        ),
        [addr.ir_value()],
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(mlir_T.i32(), r, [0])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [1])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [2])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [3])),
    )


def _dot_sq_bf16x2(packed_i32, acc):
    r = llvm.inline_asm(
        mlir_T.f32(),
        [acc.ir_value(), packed_i32.ir_value()],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $2;"
        f" {_CVT_F32_FROM_H} _flo, _lo;"
        f" {_CVT_F32_FROM_H} _fhi, _hi;"
        " fma.rn.f32 $0, _flo, _flo, $1;"
        " fma.rn.f32 $0, _fhi, _fhi, $0; }",
        "=f,f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Float32(r)


def _rsqrt_approx_f32(x):
    r = llvm.inline_asm(
        mlir_T.f32(),
        [x.ir_value()],
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Float32(r)


def _exp2_approx_f32(x):
    r = llvm.inline_asm(
        mlir_T.f32(),
        [x.ir_value()],
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Float32(r)


def _exp_approx_f32(x):
    return _exp2_approx_f32(x * f32(1.4426950408889634))


def _mul_bf16x2_f32(packed_i32, scalar):
    r = llvm.inline_asm(
        mlir_T.i32(),
        [packed_i32.ir_value(), scalar.ir_value()],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $1;"
        f" {_CVT_F32_FROM_H} _flo, _lo;"
        f" {_CVT_F32_FROM_H} _fhi, _hi;"
        " mul.f32 _flo, _flo, $2;"
        " mul.f32 _fhi, _fhi, $2;"
        f" {_CVT_H2_FROM_F32} $0, _fhi, _flo; }}",
        "=r,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _cp_async_bf16x8(base_addr_i64, bf16_elem_offset, smem_addr_i32):
    """cp.async.ca, 16 B (8 bf16). Uses .ca for K and Q (small reuse stream)."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            smem_addr_i32.ir_value(),
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $3, 2, $2;"
        " cp.async.ca.shared.global [$1], [_a], 16;"
        " mov.u32 $0, 0; }",
        "=r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _cp_async_bf16x8_cg(base_addr_i64, bf16_elem_offset, smem_addr_i32):
    """cp.async.cg, 16 B (8 bf16). .cg = bypass L1 (cache only at L2).
    Used for H — single-pass stream, no L1 reuse keeps L1 capacity for K/Q/V.
    Note: ptxas on this rig rejects `.L1::no_allocate` on cp.async; .cg alone
    already implies skipping L1 caching."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            smem_addr_i32.ir_value(),
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $3, 2, $2;"
        " cp.async.cg.shared.global [$1], [_a], 16;"
        " mov.u32 $0, 0; }",
        "=r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _cp_async_commit_group():
    r = llvm.inline_asm(
        mlir_T.i32(),
        [],
        "{ cp.async.commit_group; mov.u32 $0, 0; }",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _cp_async_wait_group_0():
    r = llvm.inline_asm(
        mlir_T.i32(),
        [],
        "{ cp.async.wait_group 0; mov.u32 $0, 0; }",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


# Fused gated RMSNorm epilogue helpers (VLLM_GDN_UCACHE_FUSED_NORM). The op sequence mirrors gsc
# uc_norm_kernel compiled with nvcc --use_fast_math (ftz everywhere, rsqrt/ex2 approx); the reciprocal / square-sum
# forms are selected by GDN_UCN_RCP (0 rcp.approx, 1 div.approx, 2 div.full) and GDN_UCN_SS (0 fma chain, 1 mul+add),
# Both forms can be compared against the nvcc build for bitwise equivalence.
_UCN_RCP = int(os.environ.get("GDN_UCN_RCP", "0"))
_UCN_SS = int(os.environ.get("GDN_UCN_SS", "0"))
_UCN_RCP_INSTR = {0: "rcp.approx.ftz.f32 _y, _e;", 1: "div.approx.ftz.f32 _y, 0f3F800000, _e;",
                  2: "div.full.ftz.f32 _y, 0f3F800000, _e;"}[_UCN_RCP]


def _lds_u16(smem_addr_i32):
    r = llvm.inline_asm(
        mlir_T.i32(),
        [smem_addr_i32.ir_value()],
        "{ .reg .b16 _h; ld.shared.b16 _h, [$1]; cvt.u32.u16 $0, _h; }",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _sts_u16(smem_addr_i32, val_i32):
    r = llvm.inline_asm(
        mlir_T.i32(),
        [smem_addr_i32.ir_value(), val_i32.ir_value()],
        "{ .reg .b16 _h; cvt.u16.u32 _h, $2; st.shared.b16 [$1], _h; mov.u32 $0, 0; }",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _ldg_u16(base_addr_i64, bf16_elem_offset):
    r = llvm.inline_asm(
        mlir_T.i32(),
        [base_addr_i64.ir_value(), bf16_elem_offset.ir_value()],
        "{ .reg .u64 _a; .reg .b16 _h; mad.wide.u32 _a, $2, 2, $1; ld.global.nc.b16 _h, [_a]; cvt.u32.u16 $0, _h; }",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _ldg_f32(base_addr_i64, f32_elem_offset):
    r = llvm.inline_asm(
        mlir_T.f32(),
        [base_addr_i64.ir_value(), f32_elem_offset.ir_value()],
        "{ .reg .u64 _a; mad.wide.u32 _a, $2, 4, $1; ld.global.nc.f32 $0, [_a]; }",
        "=f,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Float32(r)


def _stg_zero8(base_addr_i64, bf16_elem_offset):
    """STG.64 of zeros (4 bf16) at base + 2*offset."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [base_addr_i64.ir_value(), bf16_elem_offset.ir_value()],
        "{ .reg .u64 _a; .reg .b32 _z; mov.b32 _z, 0; mad.wide.u32 _a, $2, 2, $1;"
        " st.global.v2.b32 [_a], {_z, _z}; mov.u32 $0, 0; }",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _gnorm4(x0, x1, x2, x3, g0, g1, g2, g3, w0, w1, w2, w3, eps, sigmoid):
    """One warp, one 128-value row: lane holds values c = lane + 32 i (bf16 bits in x_i), gate bf16 bits g_i,
    weights w_i (f32). Returns the 4 gated-RMSNorm outputs as bf16 bits (uc_norm_kernel arithmetic)."""
    if _UCN_SS == 0:
        ss = ("fma.rn.ftz.f32 _ss, _v0, _v0, 0f00000000; fma.rn.ftz.f32 _ss, _v1, _v1, _ss;"
              " fma.rn.ftz.f32 _ss, _v2, _v2, _ss; fma.rn.ftz.f32 _ss, _v3, _v3, _ss;")
    else:
        ss = ("mul.rn.ftz.f32 _ss, _v0, _v0; mul.rn.ftz.f32 _t, _v1, _v1; add.rn.ftz.f32 _ss, _ss, _t;"
              " mul.rn.ftz.f32 _t, _v2, _v2; add.rn.ftz.f32 _ss, _ss, _t; mul.rn.ftz.f32 _t, _v3, _v3;"
              " add.rn.ftz.f32 _ss, _ss, _t;")
    red = "".join(f" shfl.sync.bfly.b32 _t, _ss, {o}, 31, 0xffffffff; add.ftz.f32 _ss, _ss, _t;"
                  for o in (16, 8, 4, 2, 1))
    gate = "mov.f32 _gv, _y;" if sigmoid else "mul.ftz.f32 _gv, _x, _y;"
    per = ""
    for i in range(4):
        per += (f" cvt.u16.u32 _h, ${8 + i}; cvt.f32.bf16 _x, _h; mul.ftz.f32 _a, _x, 0fBFB8AA3B;"
                f" ex2.approx.ftz.f32 _e, _a; add.ftz.f32 _e, _e, 0f3F800000; {_UCN_RCP_INSTR} {gate}"
                f" mul.ftz.f32 _o, _v{i}, _r; mul.ftz.f32 _o, _o, ${12 + i}; mul.ftz.f32 _o, _o, _gv;"
                f" cvt.rn.bf16.f32 _h, _o; cvt.u32.u16 ${i}, _h;")
    unpack = "".join(f" cvt.u16.u32 _h, ${4 + i}; cvt.f32.bf16 _v{i}, _h;" for i in range(4))
    asm = ("{ .reg .b16 _h; .reg .f32 _v0, _v1, _v2, _v3, _ss, _t, _r, _x, _a, _e, _y, _gv, _o;"
           + unpack + " " + ss + red
           + " mul.ftz.f32 _ss, _ss, 0f3C000000; add.ftz.f32 _ss, _ss, $16; rsqrt.approx.ftz.f32 _r, _ss;"
           + per + " }")
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.i32(), mlir_T.i32(), mlir_T.i32(), mlir_T.i32()]),
        [x0.ir_value(), x1.ir_value(), x2.ir_value(), x3.ir_value(), g0.ir_value(), g1.ir_value(), g2.ir_value(),
         g3.ir_value(), w0.ir_value(), w1.ir_value(), w2.ir_value(), w3.ir_value(), eps.ir_value()],
        asm,
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(mlir_T.i32(), r, [0])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [1])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [2])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [3])),
    )


def _l2_prefetch_bulk(base_addr_i64, byte_off_i64, nbytes):
    """cp.async.bulk.prefetch.L2 of nbytes (multiple of 16) at base + byte_off. Read-only hint."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [base_addr_i64.ir_value(), byte_off_i64.ir_value()],
        "{ .reg .u64 _a; add.u64 _a, $1, $2; cp.async.bulk.prefetch.L2.global [_a], " + str(int(nbytes)) + ";"
        " mov.u32 $0, 0; }",
        "=r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _qo_block(y_bf16_i32):
    """One MXFP8 block (32 values: one per lane) of a gated-norm output row, GSC_QO rule (bitwise vs
    FlashInfer 0.6.18.post1 mxfp8_quantize): yq = f32(bf16 y); amax = warp max |yq| (max.f32, xor 16..1);
    nm = mul.rn(amax, f32(1/448)); e8m0 = 0 if !(nm > 0) else min(exp(nm) + bump, 254) with bump = (mant != 0 &&
    !(exp == 0 && mant <= 0x400000)); inv = e2 ? bits((254 - e2) << 23) : 0; byte = cvt.rn.satfinite.e4m3(
    clamp(mul.rn(yq, inv), +-448)). Returns (e4m3 byte, e2) as i32."""
    asm = ("{ .reg .b16 _h; .reg .f32 _y, _a, _t, _nm, _inv, _v; .reg .b32 _b, _e, _m, _bump, _e2, _ib;"
           " .reg .pred _p, _q, _r; .reg .b16 _pk;"
           " cvt.u16.u32 _h, $2; cvt.f32.bf16 _y, _h; abs.f32 _a, _y;"
           + "".join(f" shfl.sync.bfly.b32 _t, _a, {o}, 31, 0xffffffff; max.f32 _a, _a, _t;" for o in (16, 8, 4, 2, 1))
           + " mul.rn.f32 _nm, _a, 0f3B124925; mov.b32 _b, _nm; shr.b32 _e, _b, 23; and.b32 _e, _e, 255;"
           " and.b32 _m, _b, 0x7FFFFF; setp.ne.u32 _p, _m, 0; setp.eq.u32 _q, _e, 0; setp.le.u32 _r, _m, 0x400000;"
           " and.pred _q, _q, _r; not.pred _q, _q; and.pred _p, _p, _q; selp.u32 _bump, 1, 0, _p;"
           " add.u32 _e2, _e, _bump; min.u32 _e2, _e2, 254; setp.gt.f32 _p, _nm, 0f00000000; selp.u32 _e2, _e2, 0, _p;"
           " sub.u32 _ib, 254, _e2; shl.b32 _ib, _ib, 23; mov.b32 _inv, _ib; setp.eq.u32 _p, _e2, 0;"
           " selp.f32 _inv, 0f00000000, _inv, _p; mul.rn.f32 _v, _y, _inv; max.f32 _v, _v, 0fC3E00000;"
           " min.f32 _v, _v, 0f43E00000; cvt.rn.satfinite.e4m3x2.f32 _pk, 0f00000000, _v; cvt.u32.u16 $0, _pk;"
           " and.b32 $0, $0, 255; mov.b32 $1, _e2; }")
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.i32(), mlir_T.i32()]),
        [y_bf16_i32.ir_value()],
        asm,
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(llvm.extractvalue(mlir_T.i32(), r, [0])), Int32(llvm.extractvalue(mlir_T.i32(), r, [1]))


def _stg_u8(base_addr_i64, byte_off_i64, val_i32):
    r = llvm.inline_asm(
        mlir_T.i32(),
        [base_addr_i64.ir_value(), byte_off_i64.ir_value(), val_i32.ir_value()],
        "{ .reg .u64 _a; .reg .b16 _h; add.u64 _a, $1, $2; cvt.u16.u32 _h, $3; st.global.u8 [_a], _h; mov.u32 $0, 0; }",
        "=r,l,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _stg_u32(base_addr_i64, byte_off_i64, val_i32):
    r = llvm.inline_asm(
        mlir_T.i32(),
        [base_addr_i64.ir_value(), byte_off_i64.ir_value(), val_i32.ir_value()],
        "{ .reg .u64 _a; add.u64 _a, $1, $2; st.global.u32 [_a], $3; mov.u32 $0, 0; }",
        "=r,l,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _qo_sf_off(row, head, psc):
    """byte offset of the e8m0 word (scale columns head*4..+3) of `row` in FlashInfer's 128x4 swizzled layout"""
    return (head * Int32(512) + (row & Int32(31)) * Int32(16) + ((row & Int32(127)) >> Int32(5)) * Int32(4)
            + (row >> Int32(7)) * (Int32(128) * psc))


def _ldg_b32(addr_i64):
    r = llvm.inline_asm(mlir_T.i32(), [addr_i64.ir_value()], "ld.global.b32 $0, [$1];", "=r,l",
                        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(r)


def _ldg_v4b32(addr_i64):
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.i32(), mlir_T.i32(), mlir_T.i32(), mlir_T.i32()]),
        [addr_i64.ir_value()], "ld.global.v4.b32 {$0,$1,$2,$3}, [$4];", "=r,=r,=r,=r,l",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Int32(llvm.extractvalue(mlir_T.i32(), r, [i])) for i in range(4))


def _stg_b32_addr(addr_i64, val_i32):
    r = llvm.inline_asm(mlir_T.i32(), [addr_i64.ir_value(), val_i32.ir_value()],
                        "st.global.b32 [$1], $2; mov.u32 $0, 0;", "=r,l,r",
                        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(r)


def _sts_b32(smem_addr_i32, val_i32):
    r = llvm.inline_asm(mlir_T.i32(), [smem_addr_i32.ir_value(), val_i32.ir_value()],
                        "st.shared.b32 [$1], $2; mov.u32 $0, 0;", "=r,r,r",
                        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(r)


def _atom_add_acqrel(addr_i64):
    """fence + atom.add.u32 (acq_rel, gpu scope) of 1; returns the previous value"""
    r = llvm.inline_asm(mlir_T.i32(), [addr_i64.ir_value()],
                        "fence.acq_rel.gpu; atom.global.acq_rel.gpu.add.u32 $0, [$1], 1;", "=r,l",
                        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(r)


def _conv4(s0, s1, s2, x0, x1, x2, x3, w0, w1, w2, w3):
    """vLLM _causal_conv1d_update_kernel (spec, width 4, SiLU, no bias) for ONE adjacent channel pair over
    4 tokens (paired-channel arithmetic, slow = Triton PTX path): taps mul.rn.bf16x2, f32 adds from
    0.0 in tap order, y = div.full(a, ex2.approx(a * -log2e) + 1), cvt.rn.bf16x2. seq = [s0 s1 s2 x0 x1 x2 x3] (bf16x2
    words: old conv state positions off..off+2, raw x of the 4 tokens); w = the 8 taps (ch0 t0..t3 | ch1 t0..t3)."""
    seq = ["$4", "$5", "$6", "$7", "$8", "$9", "$10"]
    a = ("{ .reg .b32 _w0, _w1, _w2, _w3, _p; .reg .f32 _a0, _a1, _f, _e, _y0, _y1;"
         " prmt.b32 _w0, $11, $13, 0x5410; prmt.b32 _w1, $11, $13, 0x7632;"
         " prmt.b32 _w2, $12, $14, 0x5410; prmt.b32 _w3, $12, $14, 0x7632;")
    for tt in range(4):
        a += " mov.f32 _a0, 0f00000000; mov.f32 _a1, 0f00000000;"
        for j in range(4):
            a += (f" mul.rn.bf16x2 _p, {seq[tt + j]}, _w{j}; shl.b32 _f, _p, 16; add.f32 _a0, _a0, _f;"
                  f" and.b32 _f, _p, 0xffff0000; add.f32 _a1, _a1, _f;")
        a += (" mul.f32 _f, _a0, 0fBFB8AA3B; ex2.approx.f32 _e, _f; add.f32 _e, _e, 0f3F800000;"
              " div.full.f32 _y0, _a0, _e;"
              " mul.f32 _f, _a1, 0fBFB8AA3B; ex2.approx.f32 _e, _f; add.f32 _e, _e, 0f3F800000;"
              " div.full.f32 _y1, _a1, _e;"
              f" cvt.rn.bf16x2.f32 ${tt}, _y1, _y0;")
    a += " }"
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.i32(), mlir_T.i32(), mlir_T.i32(), mlir_T.i32()]),
        [v.ir_value() for v in (s0, s1, s2, x0, x1, x2, x3, w0, w1, w2, w3)],
        a, "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Int32(llvm.extractvalue(mlir_T.i32(), r, [i])) for i in range(4))


def _exit_cta_if_neg(idx_i32):
    """Retire the calling thread iff idx < 0 (PTX `exit`).

    Callers pass the CTA's cache_idx so a padded CUDA-graph row (sentinel
    index < 0, batch padding) costs ~nothing instead of a full T-step verify
    against a scratch page. Must be called CTA-uniformly at kernel entry,
    BEFORE any SMEM/mbarrier/TMA/cp.async issue: every thread of the CTA sees
    the same idx and the whole CTA retires together. Rows with idx >= 0 are
    untouched."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [idx_i32.ir_value()],
        "{ .reg .pred _pexit; setp.lt.s32 _pexit, $1, 0;"
        " @_pexit exit; mov.u32 $0, 0; }",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _st_global_f32(base_addr_i64, f32_elem_offset, val_f32):
    """STG.32 of one f32 (offset in f32 elements). Used for the g_cache ring
    append; all global writes in this kernel go through inline PTX."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            base_addr_i64.ir_value(),
            f32_elem_offset.ir_value(),
            val_f32.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $2, 4, $1;"
        " st.global.f32 [_a], $3; mov.u32 $0, 0; }",
        "=r,l,r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _prefetch_l2_bf16(base_addr_i64, bf16_elem_offset):
    """(u-cache) prefetch.global.L2 of the 128-B line at base + 2*offset.
    Issued at kernel entry for the khist/u ring rows so the mid-kernel
    cp.async waves hit L2 instead of paying full DRAM latency (matters in
    the small-B latency-bound regime; at large B it only shifts the same
    bytes earlier)."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [base_addr_i64.ir_value(), bf16_elem_offset.ir_value()],
        "{ .reg .u64 _a; mad.wide.u32 _a, $2, 2, $1;"
        " prefetch.global.L2 [_a]; mov.u32 $0, 0; }",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _lds_b32(smem_addr_i32):
    """(flush) LDS.32: 4 B (2 bf16) from SMEM. Address must be 4-B aligned."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [smem_addr_i32.ir_value()],
        "ld.shared.b32 $0, [$1];",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _fold_fma_bf16x2(packed_i32, bdec_f32, d0_f32, d1_f32):
    """(flush) per-pair state-fold epilogue: unpack two half-precision S0
    values (lo = col c, hi = col c+1) from a packed i32 and return
    (lo*bdec + d0, hi*bdec + d1) as f32 — S_h = bdec*S0 + D with a single
    f32 fma per element. bf16 -> f32 via 16-bit left shift (exact, kept
    verbatim so the default-mode cubin is unchanged); fp16 -> f32 needs a
    real cvt (fp16 bits are not an f32 prefix). Keyed on the STATE dtype:
    this helper only ever unpacks checkpoint pairs."""
    if state_ty is cutlass.Float16:
        unpack_asm = (
            "{ .reg .b16 _lo, _hi; .reg .f32 _fl, _fh;"
            " mov.b32 {_lo, _hi}, $2;"
            " cvt.f32.f16 _fl, _lo;"
            " cvt.f32.f16 _fh, _hi;"
        )
    else:
        unpack_asm = (
            "{ .reg .b32 _wl, _wh; .reg .f32 _fl, _fh;"
            " shl.b32 _wl, $2, 16;"
            " and.b32 _wh, $2, 0xFFFF0000;"
            " mov.b32 _fl, _wl;"
            " mov.b32 _fh, _wh;"
        )
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32(), mlir_T.f32()]),
        [
            packed_i32.ir_value(),
            bdec_f32.ir_value(),
            d0_f32.ir_value(),
            d1_f32.ir_value(),
        ],
        unpack_asm + " fma.rn.f32 $0, _fl, $3, $4; fma.rn.f32 $1, _fh, $3, $5; }",
        "=f,=f,r,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        f32(llvm.extractvalue(mlir_T.f32(), r, [0])),
        f32(llvm.extractvalue(mlir_T.f32(), r, [1])),
    )


def _ldmatrix_x4_trans(addr_i32):
    """(flush) ldmatrix.x4.trans: A-fragments [m16, k16] for m16n8k16 read
    from a row-major [k16, m16] SMEM tile (the natural u-ring staging
    [j, v]). The caller computes the per-lane address as
      base + ((lane&7) + ((lane>>4)&1)*8) * row_stride + ((lane>>3)&1)*16
    (k-row, m-col-block bytes), which yields the fragment quadrant order
    (m0-7,k0-7), (m8-15,k0-7), (m0-7,k8-15), (m8-15,k8-15) — identical to
    the non-trans A pattern consumed by mma.m16n8k16 (cf. _h_gemm_4v)."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal(
            [mlir_T.i32(), mlir_T.i32(), mlir_T.i32(), mlir_T.i32()]
        ),
        [addr_i32.ir_value()],
        "ldmatrix.sync.aligned.x4.m8n8.trans.shared.b16 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(mlir_T.i32(), r, [0])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [1])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [2])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [3])),
    )


def _bar_sync_1_64():
    """Named barrier 1 over 64 threads (warps 2-3): orders the khist cp.async
    wave (issued and waited only by warps 2-3) before the scores GEMM's
    cross-warp ldmatrix reads, without stalling warps 0-1 which never touch
    the tile."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [],
        "{ bar.sync 1, 64; mov.u32 $0, 0; }",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _r_sub_bf16x2(packed_i32, neg_eg_f32, hw0_f32, hw1_f32):
    """R-pass pair op: unpack two v-values, fma each with (-e^{G_s}) * hw_k,
    repack to a packed pair in one i32 read-modify-write."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            packed_i32.ir_value(),
            neg_eg_f32.ir_value(),
            hw0_f32.ir_value(),
            hw1_f32.ir_value(),
        ],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $1;"
        f" {_CVT_F32_FROM_H} _flo, _lo;"
        f" {_CVT_F32_FROM_H} _fhi, _hi;"
        " fma.rn.f32 _flo, $2, $3, _flo;"
        " fma.rn.f32 _fhi, $2, $4, _fhi;"
        f" {_CVT_H2_FROM_F32} $0, _fhi, _flo; }}",
        "=r,r,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _sts_bf16x2_f32(smem_addr_i32, lo_f32, hi_f32):
    """Packed FP32 -> IO-dtype pair cast + STS.32 to SMEM. The cvt packs
    (hi, lo) into one 32-bit register; the store writes both values in a
    single 4-byte SMEM transaction."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [smem_addr_i32.ir_value(), lo_f32.ir_value(), hi_f32.ir_value()],
        "{ .reg .b32 _v;"
        f" {_CVT_H2_FROM_F32} _v, $3, $2;"
        " st.shared.b32 [$1], _v;"
        " mov.u32 $0, 0; }",
        "=r,r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _sts_st2_f32(smem_addr_i32, lo_f32, hi_f32):
    """STATE-dtype variant of ``_sts_bf16x2_f32``: packs the folded S_h f32
    pair to the checkpoint element type (fp16 in mixed mode) before the
    4-byte SMEM store. Used ONLY at the two fold pack sites."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [smem_addr_i32.ir_value(), lo_f32.ir_value(), hi_f32.ir_value()],
        "{ .reg .b32 _v;"
        f" {_CVT_ST2_FROM_F32} _v, $3, $2;"
        " st.shared.b32 [$1], _v;"
        " mov.u32 $0, 0; }",
        "=r,r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _sts_rg2_f32(smem_addr_i32, lo_f32, hi_f32):
    """RING-dtype variant of ``_sts_bf16x2_f32``: packs a U f32 pair to the
    ring element type before the 4-byte SMEM store. Used ONLY for the
    OutStage ring rows [8:8+t) (the ring appends then copy bytes raw).
    Compiles identically to the IO variant when the ring dtype is the IO
    dtype (same mnemonic)."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [smem_addr_i32.ir_value(), lo_f32.ir_value(), hi_f32.ir_value()],
        "{ .reg .b32 _v;"
        f" {_CVT_RG2_FROM_F32} _v, $3, $2;"
        " st.shared.b32 [$1], _v;"
        " mov.u32 $0, 0; }",
        "=r,r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _mul_rg2_f32(packed_i32, scalar):
    """RING-dtype variant of ``_mul_bf16x2_f32`` (the fold's w-scale over
    the u stage). Identical to the IO variant when ring dtype == IO."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [packed_i32.ir_value(), scalar.ir_value()],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $1;"
        f" {_CVT_F32_FROM_RG} _flo, _lo;"
        f" {_CVT_F32_FROM_RG} _fhi, _hi;"
        " mul.f32 _flo, _flo, $2;"
        " mul.f32 _fhi, _fhi, $2;"
        f" {_CVT_RG2_FROM_F32} $0, _fhi, _flo; }}",
        "=r,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _mul_packstore_f32(packed_bf16_i32, scalar):
    """Norm store: unpack a bf16 pair (the RAW loaded q/k), multiply by the
    fp32 inverse-norm, repack to the PACK dtype. In the combined
    state+cache-fp16 mode pack is fp16, so this rounds fp32->f16 DIRECTLY —
    the packed tile (and the k-cache snapshot taken from it) carry true fp16
    precision, and every packed-tile GEMM reads fp16 with no per-fragment
    conversion. Identical to ``_mul_bf16x2_f32`` when pack_ty == io (bf16)."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [packed_bf16_i32.ir_value(), scalar.ir_value()],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $1;"
        f" {_CVT_F32_FROM_H} _flo, _lo;"
        f" {_CVT_F32_FROM_H} _fhi, _hi;"
        " mul.f32 _flo, _flo, $2;"
        " mul.f32 _fhi, _fhi, $2;"
        f" {_CVT_PK2_FROM_F32} $0, _fhi, _flo; }}",
        "=r,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _mul_pack_f32(packed_pk_i32, scalar):
    """bdec row-scale over the (already pack-dtype) packed A-tile: unpack
    pack, multiply by e^{G_P}, repack pack. Identical to ``_mul_bf16x2_f32``
    when pack_ty == io (bf16)."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [packed_pk_i32.ir_value(), scalar.ir_value()],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $1;"
        f" {_CVT_F32_FROM_PK} _flo, _lo;"
        f" {_CVT_F32_FROM_PK} _fhi, _hi;"
        " mul.f32 _flo, _flo, $2;"
        " mul.f32 _fhi, _fhi, $2;"
        f" {_CVT_PK2_FROM_F32} $0, _fhi, _flo; }}",
        "=r,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _repack_io2_to_rg2(packed_i32):
    """Unpack an IO-dtype pair, repack as a RING-dtype pair (bf16 -> fp16 in
    the mixed mode — exact for in-range values: normed-k magnitudes <= 1).
    Used at the k-ring append STGs and the fold's khist STS-back."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [packed_i32.ir_value()],
        "{ .reg .b16 _lo, _hi; .reg .f32 _flo, _fhi;"
        " mov.b32 {_lo, _hi}, $1;"
        f" {_CVT_F32_FROM_H} _flo, _lo;"
        f" {_CVT_F32_FROM_H} _fhi, _hi;"
        f" {_CVT_RG2_FROM_F32} $0, _fhi, _flo; }}",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _lds_v4_b32(smem_addr_i32):
    """LDS.128: 16 B (8 bf16) from SMEM. Address must be 16-B aligned."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal(
            [mlir_T.i32(), mlir_T.i32(), mlir_T.i32(), mlir_T.i32()]
        ),
        [smem_addr_i32.ir_value()],
        "ld.shared.v4.b32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(mlir_T.i32(), r, [0])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [1])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [2])),
        Int32(llvm.extractvalue(mlir_T.i32(), r, [3])),
    )


def _st_global_v4_b32(base_addr_i64, bf16_elem_offset, v0, v1, v2, v3):
    """STG.128: 16 B (8 bf16) to global. Offset in bf16 elements, 16-B aligned."""
    r = llvm.inline_asm(
        mlir_T.i32(),
        [
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
            v0.ir_value(),
            v1.ir_value(),
            v2.ir_value(),
            v3.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $2, 2, $1;"
        " st.global.v4.b32 [_a], {$3,$4,$5,$6}; mov.u32 $0, 0; }",
        "=r,l,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


def _fused_ab_1mma(a_addr, b_addr, c0, c1, c2, c3):
    """ldmatrix.x4 A + ldmatrix.x2.trans B + 1 MMA."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32()] * 4),
        [
            c0.ir_value(),
            c1.ir_value(),
            c2.ir_value(),
            c3.ir_value(),
            a_addr.ir_value(),
            b_addr.ir_value(),
        ],
        "{ .reg .b32 _a<4>, _b<2>;"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8];"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$9];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3}; }",
        "=f,=f,=f,=f,0,1,2,3,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [0])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [1])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [2])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [3])),
    )


def _fused_ab_4mma_serial_brow(a_base, b_base, c0, c1, c2, c3):
    """KKT/QKT Grams — 4 sequential (ldmatrix_A + ldmatrix_B + MMA) at
    K-stride 32B. Both operands are the packed tile, so the MMA is in the
    PACK dtype (fp16 in the combined state+cache-fp16 mode, bf16 otherwise)."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32()] * 4),
        [
            c0.ir_value(),
            c1.ir_value(),
            c2.ir_value(),
            c3.ir_value(),
            a_base.ir_value(),
            b_base.ir_value(),
        ],
        "{ .reg .b32 _a<4>, _b<2>;"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9];"
        f" {_MMA_PACK}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8+32];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9+32];"
        f" {_MMA_PACK}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8+64];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9+64];"
        f" {_MMA_PACK}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8+96];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9+96];"
        f" {_MMA_PACK}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3}; }",
        "=f,=f,=f,=f,0,1,2,3,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [0])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [1])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [2])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [3])),
    )


def _qtv_4mma(a0, a1, a2, a3, b_base):
    """4 independent QT@V (ldmatrix_B_trans + MMA), B stride 16 B."""
    zero = cutlass.Float32(0.0)
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32()] * 16),
        [zero.ir_value()] * 16
        + [
            a0.ir_value(),
            a1.ir_value(),
            a2.ir_value(),
            a3.ir_value(),
            b_base.ir_value(),
        ],
        "{ .reg .b32 _b<2>;"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$0,$1,$2,$3}, {$32,$33,$34,$35}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+16];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$4,$5,$6,$7}, {$32,$33,$34,$35}, {_b0,_b1}, {$4,$5,$6,$7};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+32];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$8,$9,$10,$11}, {$32,$33,$34,$35}, {_b0,_b1}, {$8,$9,$10,$11};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+48];"
        f" {_MMA_M16N8K16_HH_F32}"
        "   {$12,$13,$14,$15}, {$32,$33,$34,$35}, {_b0,_b1}, {$12,$13,$14,$15}; }",
        "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,"
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [i])) for i in range(16)
    )


def _fused_ab_4mma_serial_brow_rgb(a_base, b_base, c0, c1, c2, c3):
    """Mixed-ring variant of ``_fused_ab_4mma_serial_brow`` for the SCORES
    GEMM only: A (the khist ring tile) is consumed RAW in the ring dtype;
    the two B fragments (the packed q/k tile, IO dtype) are converted to
    the ring dtype IN REGISTERS after each ldmatrix (exact in range), and
    the MMA issues in the ring dtype. In the default mode the cvt block is
    empty and this is byte-for-byte the plain helper."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32()] * 4),
        [
            c0.ir_value(),
            c1.ir_value(),
            c2.ir_value(),
            c3.ir_value(),
            a_base.ir_value(),
            b_base.ir_value(),
        ],
        "{ .reg .b32 _a<4>, _b<2>, _wl, _wh; .reg .f32 _fl, _fh;"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9];"
        f"{_RING_B_CVT_ASM}"
        f" {_MMA_RING_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8+32];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9+32];"
        f"{_RING_B_CVT_ASM}"
        f" {_MMA_RING_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8+64];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9+64];"
        f"{_RING_B_CVT_ASM}"
        f" {_MMA_RING_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$8+96];"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$9+96];"
        f"{_RING_B_CVT_ASM}"
        f" {_MMA_RING_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3}; }",
        "=f,=f,=f,=f,0,1,2,3,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [0])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [1])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [2])),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [3])),
    )


def _qtv_4mma_rg(a0, a1, a2, a3, b_base):
    """RING-dtype variant of ``_qtv_4mma`` — used ONLY where a ring tile is
    an MMA operand (the history contraction: A = w-scaled scores staged in
    the ring dtype, B = the u tile raw; the fold: A = w-scaled u, B = the
    khist STS-back). Identical to ``_qtv_4mma`` when ring dtype == IO."""
    zero = cutlass.Float32(0.0)
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32()] * 16),
        [zero.ir_value()] * 16
        + [
            a0.ir_value(),
            a1.ir_value(),
            a2.ir_value(),
            a3.ir_value(),
            b_base.ir_value(),
        ],
        "{ .reg .b32 _b<2>;"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36];"
        f" {_MMA_RING_F32}"
        "   {$0,$1,$2,$3}, {$32,$33,$34,$35}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+16];"
        f" {_MMA_RING_F32}"
        "   {$4,$5,$6,$7}, {$32,$33,$34,$35}, {_b0,_b1}, {$4,$5,$6,$7};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+32];"
        f" {_MMA_RING_F32}"
        "   {$8,$9,$10,$11}, {$32,$33,$34,$35}, {_b0,_b1}, {$8,$9,$10,$11};"
        " ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {_b0,_b1}, [$36+48];"
        f" {_MMA_RING_F32}"
        "   {$12,$13,$14,$15}, {$32,$33,$34,$35}, {_b0,_b1}, {$12,$13,$14,$15}; }",
        "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,"
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [i])) for i in range(16)
    )


def _h_gemm_4v(
    a_addr,
    b0_addr,
    b1_addr,
    b2_addr,
    b3_addr,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
):
    """ldmatrix_A.x4 (16x16 of K) + 4× ldmatrix_B.x2 (8x16 of V-rows × K-cols, non-trans)
       + 4× MMA accumulating into 4 separate C-tiles (one per V-group).

    A is row-major bf16 [16, 16] (rows = T, cols = K-tile of 16). The lane-row-stride
    is row_stride_bytes (passed inside the SMEM address). For sK that's K_PADDED*2=272.
    B is row-major bf16 [8, 16] non-trans (rows = V, cols = K-tile of 16). Row-stride
    is row_stride_bytes_B. B comes from the SW128-swizzled state tile
    (K_DIM*2=256 stride); callers MUST apply `_sw128_xor` to each per-lane
    B address before calling this.

    The A-fragment is shared across all 4 MMAs — 4 different B-fragments at b{0..3}_addr.
    """
    r = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_T.f32()] * 16),
        [
            c0.ir_value(),
            c1.ir_value(),
            c2.ir_value(),
            c3.ir_value(),
            c4.ir_value(),
            c5.ir_value(),
            c6.ir_value(),
            c7.ir_value(),
            c8.ir_value(),
            c9.ir_value(),
            c10.ir_value(),
            c11.ir_value(),
            c12.ir_value(),
            c13.ir_value(),
            c14.ir_value(),
            c15.ir_value(),
            a_addr.ir_value(),
            b0_addr.ir_value(),
            b1_addr.ir_value(),
            b2_addr.ir_value(),
            b3_addr.ir_value(),
        ],
        "{ .reg .b32 _a<4>, _b<2>;"
        " ldmatrix.sync.aligned.x4.m8n8.shared.b16 {_a0,_a1,_a2,_a3}, [$32];"
        f"{_H_A_CVT_ASM}"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$33];"
        f" {_MMA_H_F32}"
        "   {$0,$1,$2,$3}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$0,$1,$2,$3};"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$34];"
        f" {_MMA_H_F32}"
        "   {$4,$5,$6,$7}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$4,$5,$6,$7};"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$35];"
        f" {_MMA_H_F32}"
        "   {$8,$9,$10,$11}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$8,$9,$10,$11};"
        " ldmatrix.sync.aligned.x2.m8n8.shared.b16 {_b0,_b1}, [$36];"
        f" {_MMA_H_F32}"
        "   {$12,$13,$14,$15}, {_a0,_a1,_a2,_a3}, {_b0,_b1}, {$12,$13,$14,$15}; }",
        "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,"
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), r, [i])) for i in range(16)
    )


def _sw128_xor(addr_i32):
    """Apply SW128 (cute.make_swizzle(3, 4, 3)) XOR to a logical SMEM byte
    address.

    SW128 spec: B=3 (3 bits XORed), M=4 (target = bits 4..6), S=3 (source =
    bits 7..9). Equivalent: phys = L XOR ((L >> 3) & 0x70).

    This MUST match the swizzle the SMEM tensor was built with (see
    `_make_sH_sw128_layout_half`).
    """
    return addr_i32 ^ ((addr_i32 >> Int32(3)) & Int32(0x70))


def _make_sH_sw128_layout_half():
    """SW128 K-major BF16 layout tiled to (V_DIM_C=128, K_HALF=64).

    K_HALF=64 BF16 = exactly one 128-byte row = one SW128 swizzle period.
    The base atom (8 rows x 64 cols = 1 swizzle period) tiles 16x in V and
    1x in K. The state buffer is reused across the 2 TMA-half loads
    (single-buffer streaming), halving its SMEM footprint.
    """
    sw = cute.make_swizzle(3, 4, 3)
    base = cute.make_layout((8, 64), stride=(64, 1))
    atom = cute.make_composed_layout(sw, 0, base)
    return cute.tile_to_shape(atom, (V_DIM_C, K_HALF), order=(1, 0))


class GdnDecodeUCacheFlushKernel:
    """CuTeDSL GDN decode output + u-cache + per-request state flush."""

    def __init__(
        self,
        disable_state_update=False,
        min_blocks_per_mp=2,
        t_input=16,
        bv=None,
        n_valid=16,
        qkv_row_stride=0,
        ab_native=False,
        ab_t_stride=0,
        pdl_trigger=False,
        fused_norm=False,
        norm_sigmoid=False,
        pdl_wait=False,
        si_stride=1,
        state_slot_stride=0,
        fused_qo=False,
        fused_conv=False,
        kq_fix=False,
    ):
        assert disable_state_update, "State update not implemented in CuTeDSL kernel"
        # `bv` is accepted only for bench-script signature compatibility — this
        # kernel always consumes the full V=128 tile in one CTA so a V-split
        # path is not implemented here.
        self._disable_state_update = disable_state_update
        self._min_blocks_per_mp = min_blocks_per_mp
        self._t_input = int(t_input)
        # (native-short-T) number of valid token rows actually present in the q/k
        # gmem tensors. When n_valid < T the kernel loads only these rows via the
        # K/Q cp.async and zeros the sK/sQ[n_valid:T] smem tail itself, instead of
        # the host staging q/k into a T=16 zero-padded buffer. Default T (=16) keeps
        # the original behavior (host provides a full T-row, zero-padded tensor).
        self._n_valid = int(n_valid)
        # (strided-qkv) per-token row stride of the q/k/v gmem tensors, in elements.
        # 0 -> compact (token stride = H*K_DIM / HV*V_DIM, the staged/contiguous path).
        # >0 -> q/k/v are read directly from the fused conv-output column slices whose
        # token stride is conv_dim (= q_dim+k_dim+v_dim); the kernel loads from that
        # stride instead of requiring the host to .contiguous() them. Features within a
        # token stay contiguous (stride 1) so the smem layout / MMA path is unchanged.
        self._qkv_row_stride = int(qkv_row_stride)
        # (native-a/b) when True, a/b are the real [B, n_valid, HV] tensors (not staged into
        # T_KERNEL zero-padded buffers): batch stride = n_valid*HV and the warp-3 load/compute
        # gate uses n_valid instead of T. Tail lanes [n_valid:T] are not loaded; their gamma
        # (log_alpha=0) cannot reach the real rows through the causal prefix-sum.
        self._ab_native = bool(ab_native)
        # (strided-a/b) >0 -> a/b are regular strided views (packed a|b chunk:
        # token stride 2*HV, batch stride rows*token, feature stride 1) read
        # directly with NO host-side .contiguous(). 0 -> contiguous (HV).
        self._ab_t_stride = int(ab_t_stride)
        # Fire griddepcontrol.launch_dependents at kernel ENTRY so a dependent
        # kernel launched with use_pdl overlaps this one fully (the dependent
        # verify kernel consumes nothing we write).
        self._pdl_trigger = bool(pdl_trigger)
        # Fused gated RMSNorm (+ gate activation) epilogue: gOut is the FINAL output ([n_tok/4, 4, HV, V]
        # view of the layer's core_attn_out) and pad / rejected rows (cache_idx < 0) zero their tokens [cu[b], cu[b+1]).
        self._fused_norm = bool(fused_norm)
        self._norm_sigmoid = bool(norm_sigmoid)
        # PDL dependent of ucache_prep: before griddepcontrol.wait the CTA only L2-prefetches its state
        # head (block from the caller's state_indices[b, 0], NOT prep's outputs); everything else runs after the wait.
        self._pdl_wait = bool(pdl_wait)
        self._si_stride = int(si_stride)
        self._st_slot = int(state_slot_stride)
        # Fused out_proj MXFP8 input quant (GLUE_GSC_QO rule) in the fused-norm epilogue + the padding-row
        # zeroing of CK3's qo_zero_pad_rows (q / sf / bf16 out of rows [n_tok, pm) spread over all CTAs).
        self._fused_qo = bool(fused_qo)
        # The vLLM causal_conv1d_update (spec, width 4, SiLU, no bias) folded into this kernel: q/k/v are
        # computed from the RAW mixed_qkv + the conv state (SD layout) + taps; v-pair conv states are written by their
        # sole owner, q/k-pair conv states by the last of the VPK value-head CTAs (per-warp acq_rel counters, self-
        # resetting). mixed_qkv is not overwritten.
        self._fused_conv = bool(fused_conv)
        # kq_fix (VLLM_GDN_UCACHE_KQFIX): q/k stay RAW bf16 (exact) as MMA operands of the Grams (k.k), the
        # QK scores (q.k) and the history scores (k_hist.x); the fp32 inverse norms (q's includes K^-1/2) are applied
        # as scalars to the fp32 accumulators. The k ring stores RAW k and the u ring stores u~ = u * inv|k|, so the
        # fold S = e^G S0 + sum_j w_j u~_j k_raw_j^T keeps its form (gsc materialize unchanged). The H GEMM operand is
        # raw x * (bdec * inv|x|) rounded once to bf16 (today: rounded normalized x, re-rounded after * bdec).
        self._kq_fix = bool(kq_fix)
        if self._kq_fix:
            assert self._t_input <= 8 and not _PACK_MIXED and not _ST_MIXED, "kq_fix: t<=8, bf16 state/pack"
        if self._fused_conv:
            assert self._qkv_row_stride > 0 and self._n_valid == 4 and self._t_input == 4, "conv fold: strided T=4"
        if self._fused_qo:
            assert self._fused_norm, "fused QO needs the fused norm epilogue"
        if self._fused_norm:
            assert self._t_input == 4 and THREADS == 128, "fused norm: one warp per token row (T=4, 4 warps)"

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,
        gK: cute.Tensor,
        gV: cute.Tensor,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gAlog: cute.Tensor,
        gDtbias: cute.Tensor,
        gH0: cute.Tensor,
        gH0idx: cute.Tensor,
        gRidx: cute.Tensor,  # ring index per row
        gKC: cute.Tensor,
        gUC: cute.Tensor,
        gGC: cute.Tensor,
        gHlen: cute.Tensor,
        gBase: cute.Tensor,
        gOut: cute.Tensor,
        scale: cutlass.Float32,
        HV: cutlass.Int32,
        V_DIM: cutlass.Int32,
        H: cutlass.Int32,
        flush_min: cutlass.Int32,
        gGate: cute.Tensor,
        gNw: cute.Tensor,
        gCu: cute.Tensor,
        gate_row: cutlass.Int32,
        eps: cutlass.Float32,
        gSI: cute.Tensor,
        gQ8: cute.Tensor,
        gSF: cute.Tensor,
        qo_stride: cutlass.Int32,
        qo_psc: cutlass.Int32,
        qo_T: cutlass.Int32,
        qo_pm: cutlass.Int32,
        gCS: cute.Tensor,
        gCW: cute.Tensor,
        gNacc: cute.Tensor,
        gCtr: cute.Tensor,
        cs_seq: cutlass.Int64,
        cs_tok: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        op = MmaF16BF16Op(io, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op)
        B_val = gH0idx.layout.shape[0]
        # Build the TMA atom for the state tile. gH0 logical shape is
        # (pool, HV, V_DIM_C, K_DIM). cpasync.make_tiled_tma_atom tiles the
        # FIRST modes — we reorder modes to (V, K, HV, pool) by selecting
        # [2, 3, 1, 0] so the per-CTA tile is (V_DIM_C, K_DIM); the trailing
        # (HV, pool) modes survive tma_partition as outer iteration coords.
        # The SMEM target layout is SW128 swizzled — required for the TMA
        # descriptor.
        gH0_vkhp = cute.make_tensor(
            gH0.iterator,
            cute.select(gH0.layout, mode=[2, 3, 1, 0]),
        )
        # Half-K TMA atom — box = (V_DIM_C, K_HALF). Each CTA issues this atom
        # twice (once per K-half) into the SAME shared buffer via a 2-phase
        # mbarrier ping-pong, halving the state tile's SMEM footprint.
        sH_tma_layout = _make_sH_sw128_layout_half()
        tma_atom_h, tma_tensor_h = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gH0_vkhp,
            sH_tma_layout,
            (V_DIM_C, K_HALF),
        )
        # One CTA per (b, hv) — full V tile per CTA.
        self.kernel(
            gQ,
            gK,
            gV,
            gA,
            gB,
            gAlog,
            gDtbias,
            gH0,
            gH0idx,
            gRidx,
            gKC,
            gUC,
            gGC,
            gHlen,
            gBase,
            gOut,
            scale,
            tiled_mma,
            HV,
            V_DIM,
            H,
            tma_atom_h,
            tma_tensor_h,
            flush_min,
            gGate,
            gNw,
            gCu,
            gate_row,
            eps,
            gSI,
            gQ8,
            gSF,
            qo_stride,
            qo_psc,
            qo_T,
            qo_pm,
            gCS,
            gCW,
            gNacc,
            gCtr,
            cs_seq,
            cs_tok,
        ).launch(
            grid=(1, HV, B_val),
            block=[THREADS, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=self._min_blocks_per_mp,
            use_pdl=self._pdl_wait,
        )

    @cute.kernel
    def kernel(
        self,
        gQ: cute.Tensor,
        gK: cute.Tensor,
        gV: cute.Tensor,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gAlog: cute.Tensor,
        gDtbias: cute.Tensor,
        gH0: cute.Tensor,
        gH0idx: cute.Tensor,
        gRidx: cute.Tensor,  # ring index per row
        gKC: cute.Tensor,
        gUC: cute.Tensor,
        gGC: cute.Tensor,
        gHlen: cute.Tensor,
        gBase: cute.Tensor,
        gOut: cute.Tensor,
        scale: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        HV: cutlass.Int32,
        V_DIM: cutlass.Int32,
        H: cutlass.Int32,
        tma_atom_h: cute.CopyAtom,
        tma_tensor_h: cute.Tensor,
        flush_min: cutlass.Int32,
        gGate: cute.Tensor,
        gNw: cute.Tensor,
        gCu: cute.Tensor,
        gate_row: cutlass.Int32,
        eps: cutlass.Float32,
        gSI: cute.Tensor,
        gQ8: cute.Tensor,
        gSF: cute.Tensor,
        qo_stride: cutlass.Int32,
        qo_psc: cutlass.Int32,
        qo_T: cutlass.Int32,
        qo_pm: cutlass.Int32,
        gCS: cute.Tensor,
        gCW: cute.Tensor,
        gNacc: cute.Tensor,
        gCtr: cute.Tensor,
        cs_seq: cutlass.Int64,
        cs_tok: cutlass.Int32,
    ):
        # Strides (contiguous layout assumed). q/k/v use the batch stride of their
        # ACTUAL row count: n_valid (== T in the default/staged path, so unchanged;
        # < T in the native-short-T path where q/k/v are the real [B,n_valid,...]
        # tensors and the kernel loads n_valid rows + zeros its smem tail).
        # (strided-qkv) when qkv_row_stride>0, q/k/v are the fused conv-output column
        # slices: per-token row stride is conv_dim (shared by q/k/v) rather than the
        # compact per-tensor H*K_DIM / HV*V_DIM. Head/element strides are unchanged
        # because each token's features stay contiguous within its slice.
        _rs = self._qkv_row_stride
        _qt = _rs if _rs > 0 else H * K_DIM
        _kt = _rs if _rs > 0 else H * K_DIM
        _vt = _rs if _rs > 0 else HV * V_DIM
        cutlass.Int32(1)
        sq_h = K_DIM
        sq_t = _qt
        sq_b = self._n_valid * _qt
        cutlass.Int32(1)
        sk_h = K_DIM
        sk_t = _kt
        sk_b = self._n_valid * _kt
        cutlass.Int32(1)
        sv_hv = V_DIM
        sv_t = _vt
        sv_b = self._n_valid * _vt
        cutlass.Int32(1)
        so_hv = V_DIM
        so_t = HV * V_DIM
        # Output batch stride matches the output tensor's row count: n_valid (== T in
        # the staged path -> [B,T_KERNEL] out; == T in the native path where out is
        # the compact [B,T] tensor, valid because native is gated to T==t_disc so the
        # t_input-gated STG writes exactly T rows). Removes the caller's reshape copy.
        so_b = self._n_valid * HV * V_DIM
        # (native-a/b) a/b rows actually present in the tensor: n_valid when native (real
        # [B,n_valid,HV] passed) else T (staged T_KERNEL-row zero-padded buffer). The warp-3
        # load/compute gate below uses the same count so tail lanes never read OOB.
        _ab_rows = self._n_valid if self._ab_native else T
        _ab_t = self._ab_t_stride if self._ab_t_stride > 0 else HV
        sa_hv = cutlass.Int32(1)
        sa_t = _ab_t
        sa_b = _ab_rows * _ab_t
        sb_hv = cutlass.Int32(1)
        # b carries the SAME token stride as a (the wrapper only takes this
        # path when tuple(b.stride()) == tuple(a.stride())), so b's per-token
        # and per-batch strides must use _ab_t, not a hardcoded HV. With
        # hardcoded HV, packed/chunk-view b (stride(1) != HV — the vLLM
        # strided-qkv path) is read from the wrong rows. _ab_t == HV in the
        # compact case, so this is byte-identical there.
        sb_t = _ab_t
        sb_b = _ab_rows * _ab_t
        # State pool natural layout: (pool, HV, V, K) contiguous. Addressing is
        # handled by the TMA descriptor, so no raw GMEM strides are needed here.

        tidx, _, _ = cute.arch.thread_idx()
        _pid_vt, pid_hv, pid_b = cute.arch.block_idx()
        lane_id = tidx & 31
        warp_id = tidx // WARP

        # GQA head mapping
        i_h = pid_hv // (HV // H)
        if const_expr(self._pdl_trigger):
            cute.arch.griddepcontrol_launch_dependents()
        if const_expr(self._pdl_wait):
            _pf_blk = gSI.iterator[pid_b * Int32(self._si_stride)]
            if _pf_blk > Int32(0):
                if tidx == Int32(0):
                    _l2_prefetch_bulk(
                        gH0.iterator.toint(),
                        (Int64(_pf_blk) * Int64(self._st_slot) + Int64(pid_hv) * Int64(V_DIM_C * K_DIM)) * Int64(2),
                        V_DIM_C * K_DIM * 2,
                    )
            cute.arch.griddepcontrol_wait()
        cache_idx = gH0idx.iterator[pid_b]
        if const_expr(self._fused_qo):
            # CK3 qo_zero_pad_rows: rows n_tok + pid_b + k * n_req (k >= 0) of [n_tok, pm), head pid_hv:
            # q bytes (rows < T), bf16 out (rows < T), and the scale word (rows < pm)
            _qz_nreq = cute.arch.grid_dim()[2]
            _qz_first = gCu.iterator[_qz_nreq] + pid_b
            _qz_q = gQ8.iterator.toint()
            _qz_s = gSF.iterator.toint()
            _qz_o = gOut.iterator.toint()
            _qz_row0 = _qz_first + (tidx // Int32(32)) * _qz_nreq
            _qz_step = Int32(4) * _qz_nreq
            _qz_cnt = Int32(0)
            if _qz_row0 < qo_pm:
                _qz_cnt = (qo_pm - _qz_row0 + _qz_step - Int32(1)) // _qz_step
            for _qz_k in cutlass.range(_qz_cnt):
                _qz_row = _qz_row0 + _qz_k * _qz_step
                if _qz_row < qo_T:
                    _stg_u32(_qz_q, Int64(_qz_row) * Int64(qo_stride) + Int64(pid_hv * Int32(V_DIM_C) + (tidx & Int32(31)) * Int32(4)), Int32(0))
                    _stg_zero8(_qz_o, (_qz_row * HV + pid_hv) * Int32(V_DIM_C) + (tidx & Int32(31)) * Int32(4))
                if (tidx & Int32(31)) == Int32(0):
                    _stg_u32(_qz_s, Int64(_qo_sf_off(_qz_row, pid_hv, qo_psc)), Int32(0))
        if const_expr(self._fused_norm):
            # Rows the u-cache does not run own zero outputs for their tokens (gsc uc_norm semantics)
            if cache_idx < Int32(0):
                _z_bos = gCu.iterator[pid_b]
                _z_T = gCu.iterator[pid_b + 1] - _z_bos
                _z_t = tidx // Int32(32)
                if _z_t < _z_T:
                    _stg_zero8(
                        gOut.iterator.toint(),
                        ((_z_bos + _z_t) * HV + pid_hv) * Int32(V_DIM_C) + (tidx & Int32(31)) * Int32(4),
                    )
                    if const_expr(self._fused_qo):
                        _stg_u32(gQ8.iterator.toint(), Int64(_z_bos + _z_t) * Int64(qo_stride)
                                 + Int64(pid_hv * Int32(V_DIM_C) + (tidx & Int32(31)) * Int32(4)), Int32(0))
                        if (tidx & Int32(31)) == Int32(0):
                            _stg_u32(gSF.iterator.toint(), Int64(_qo_sf_off(_z_bos + _z_t, pid_hv, qo_psc)), Int32(0))
        # Padded CUDA-graph rows carry cache_idx < 0 (batch padding sentinel):
        # the whole CTA retires here, before any SMEM/TMA/ring work. Real rows
        # (idx >= 0) proceed unchanged.
        _exit_cta_if_neg(cache_idx)

        # Ring addressing + per-request history fill level.
        # k_cache [pool, H, W_RING, K] bf16 (L2-NORMED k),
        # u_cache [pool, HV, W_RING, V] bf16,
        # g_cache [pool, HV, W_RING] f32 (absolute log-decay since ckpt).
        P_hist = gHlen.iterator[pid_b]
        # Ring window origin (Triton cache_base semantics). All HISTORY row
        # addresses below are (ring_base + logical_row) & RING_MASK; appends
        # use (ring_base + P_hist + s) & RING_MASK. Smem/logical rows stay
        # [0, W_RING) — order is carried by the per-row g values, so the
        # rotate-at-load keeps every downstream tile/GEMM unchanged.
        ring_base = gBase.iterator[pid_b]
        # Pool/head strides from the descriptor layouts, NOT dense shape
        # products: paged serving pools (vLLM) are block-strided views whose
        # dim-0 stride spans the whole multi-component page; inner dims stay
        # dense. With contiguous pools these reduce to the old products.
        skc_pool = gKC.layout.stride[0]
        skc_h = gKC.layout.stride[1]
        suc_pool = gUC.layout.stride[0]
        suc_hv = gUC.layout.stride[1]
        sgc_pool = gGC.layout.stride[0]
        sgc_hv = gGC.layout.stride[1]
        # 64-bit per-CTA pool ELEMENT offsets: block-strided paged pools can
        # exceed 2^31 elements (cache_idx * pool_stride wraps in 32-bit).
        # These fold into the i64 byte bases / iterator indices below; all
        # remaining per-lane offsets are intra-page and stay 32-bit.
        cache_idx64 = Int64(cache_idx)
        # The rings are indexed by the row's RING slot (request-table slot), not by the state block.
        ring_idx64 = Int64(gRidx.iterator[pid_b])
        _kc_pool_e64 = ring_idx64 * skc_pool + Int64(i_h) * skc_h
        _uc_pool_e64 = ring_idx64 * suc_pool + Int64(pid_hv) * suc_hv
        _gc_pool_e64 = ring_idx64 * sgc_pool + Int64(pid_hv) * sgc_hv
        # g-ring byte base with the pool offset absorbed (stores use small
        # intra-row offsets only; do NOT re-add _gc_pool_e64 at call sites).
        _gGC_base_st = gGC.iterator.toint() + _gc_pool_e64 * 4

        # Issue the per-token gate/bias (a, b, A_log, dt_bias) loads for warp 3
        # right after pid/lane setup so they are the first instructions warp 3
        # emits — maximizing the HBM round-trip hiding window before the
        # gamma/beta math consumes them.
        _v7e_a_bf16 = f32(0.0)
        _v7e_b_bf16 = f32(0.0)
        _v7e_alog_bf16 = f32(0.0)
        _v7e_dt_bf16 = f32(0.0)
        # Load only the _ab_rows present in the a/b tensors (n_valid native,
        # T staged) so tail lanes never index past the real [B,n_valid,HV] rows.
        if warp_id == 3 and lane_id < _ab_rows:
            _v7e_a_bf16 = gA.iterator[
                pid_b * sa_b + lane_id * sa_t + pid_hv * sa_hv
            ].to(f32)
            _v7e_b_bf16 = gB.iterator[
                pid_b * sb_b + lane_id * sb_t + pid_hv * sb_hv
            ].to(f32)
            _v7e_alog_bf16 = gAlog.iterator[pid_hv].to(f32)
            _v7e_dt_bf16 = gDtbias.iterator[pid_hv].to(f32)

        # Issue the g_cache ring LDG on warp 1 (idle until the norm phase, where
        # it computes w_j/bdec from these values). Lanes >= W_RING keep 0.0;
        # g_cache is f32 so the load is direct.
        _g_hist_f32 = f32(0.0)
        if warp_id == 1 and lane_id < W_RING:
            _g_hist_f32 = gGC.iterator[
                _gc_pool_e64 + Int64((ring_base + lane_id) & Int32(RING_MASK))
            ]

        # (perf) L2-prefetch the live khist/u ring rows (rows < P; 2 x
        # 128-B lines per row) at kernel entry — the mid-kernel cp.async
        # waves then hit L2, shrinking their exposed latency at small B.
        _pf_row = (tidx & Int32(31)) >> 1
        _pf_half = (tidx & Int32(1)) * Int32(64)
        _pf_ring = (ring_base + _pf_row) & Int32(RING_MASK)
        if _pf_row < P_hist:
            if tidx < Int32(32):
                _prefetch_l2_bf16(
                    gKC.iterator.toint() + _kc_pool_e64 * 2,
                    _pf_ring * K_DIM + _pf_half,
                )
            if tidx >= Int32(32) and tidx < Int32(64):
                _prefetch_l2_bf16(
                    gUC.iterator.toint() + _uc_pool_e64 * 2,
                    _pf_ring * V_DIM + _pf_half,
                )

        smem = utils.SmemAllocator()

        @cute.struct
        class SS:
            # mbarrier for the state-tile TMA load (8B Int64). Placed first so
            # its natural 8B alignment is preserved by the prefix; the large
            # 128-aligned buffers follow. Arrival count = 1 (only one thread
            # issues the TMA + arrive; the TX-bytes complete it independently).
            h_load_mbar: cute.struct.MemRange[Int64, 1]
            k_buf: cute.struct.Align[cute.struct.MemRange[io, TK_PAD], 128]
            # q_buf and v_buf are ALIASED onto one region (qv_buf): sQ is fully
            # read before sV's cp.async fill writes the same bytes (a
            # sync_threads orders the handoff). K_PADDED == V_PADDED == 136.
            qv_buf: cute.struct.Align[cute.struct.MemRange[io, TK_PAD], 128]
            # State tile = V=128 rows x K_HALF=64 cols, SW128 swizzled,
            # single-buffered and reused across the 2 TMA half-loads via the
            # mbarrier ping-pong. Element type = state_ty (the checkpoint
            # dtype; == io except in mixed mode).
            h_buf: cute.struct.Align[
                cute.struct.MemRange[state_ty, V_DIM_C * K_HALF], 128
            ]
            tmat_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            gamma: cute.struct.Align[cute.struct.MemRange[f32, WARP], 128]
            beta: cute.struct.Align[cute.struct.MemRange[f32, WARP], 128]
            mat_fp32: cute.struct.Align[cute.struct.MemRange[f32, TT], 128]
            scratch_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            # w-scaled TRANSPOSED scores tile [W_RING, BF_PAD] bf16 — the
            # A-operand of the MMA history contraction
            # (sWScores[r_packed, j] = w_j * khist_j . packed_r).
            # (ring-fp16) typed with the RING dtype: tenant #1 (the w-scaled
            # transposed scores) is the ring-dtype contraction's A operand.
            # Tenant #2 (QT, an IO-dtype operand of the y GEMM) writes its
            # bytes through raw _sts pair-stores below, bypassing the tensor
            # element type — ldmatrix reads are untyped, so each tenant's
            # MMA sees the bytes its own staging wrote. ring_ty == io in the
            # default mode (identical layout either way: both 2 B).
            wscores_bf: cute.struct.Align[
                cute.struct.MemRange[ring_ty, W_RING * BF_PAD], 128
            ]
            # G-ring scratch: [0:16]=G_j, [16:32]=w_j, [32]=bdec.
            ghist_fp32: cute.struct.Align[cute.struct.MemRange[f32, 64 if self._kq_fix else 48], 128]

        st = smem.allocate(SS)
        sK = st.k_buf.get_tensor(cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1)))
        # sQ and sV both view the same qv_buf SMEM region (alias).
        sQ = st.qv_buf.get_tensor(cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1)))
        # State tile view: (V_DIM_C, K_HALF) SW128-swizzled (exactly one swizzle
        # period in K, so the _sw128_xor() helper applies unchanged).
        sH_layout = _make_sH_sw128_layout_half()
        sH = st.h_buf.get_tensor(sH_layout.outer, swizzle=sH_layout.inner)
        sV = st.qv_buf.get_tensor(cute.make_layout((T, V_PADDED), stride=(V_PADDED, 1)))
        sTmat = st.tmat_bf.get_tensor(cute.make_layout((T, T), stride=(BF_PAD, 1)))
        sGamma = st.gamma.get_tensor(cute.make_layout((WARP,)))
        sBeta = st.beta.get_tensor(cute.make_layout((WARP,)))
        sMat = st.mat_fp32.get_tensor(cute.make_layout((T, T), stride=(T, 1)))
        sNegL = st.scratch_bf.get_tensor(cute.make_layout((T, T), stride=(BF_PAD, 1)))
        sWScores = st.wscores_bf.get_tensor(
            cute.make_layout((W_RING, W_RING), stride=(BF_PAD, 1))
        )
        sGhist = st.ghist_fp32.get_tensor(cute.make_layout((64 if self._kq_fix else 48,)))
        # kq_fix: sGhist[48 + r] = inv|k_r| (r < 8), sGhist[56 + r] = inv|q_r| * K^-1/2; 0 for r >= t

        # ============================================================
        # mbarrier init for the state-tile TMA load. Thread 0 issues the
        # bulk-tensor copy once; all 128 threads block in mbarrier_wait before
        # the H GEMM. Arrival count = 1 (the issuing thread is the sole
        # arriver; the TX-bytes complete the barrier independently).
        # ============================================================
        mbar_h_ptr = st.h_load_mbar.data_ptr()
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_h_ptr, 1)
        cute.arch.mbarrier_init_fence()
        sync_threads()

        # Partition the TMA tensor for this CTA's (cache_idx, pid_hv).
        # tma_tensor_h logical shape (V, K, HV, pool) (mode-reordered on host).
        # flat_divide with (V_DIM_C, K_HALF) -> (V_TILE, K_TILE, V_REST=1, K_REST=2, HV, pool).
        # Two slices, one per K-half; both write to the SAME state buffer.
        gH_tiled = cute.flat_divide(tma_tensor_h, (V_DIM_C, K_HALF))
        gH_slice0 = gH_tiled[None, None, None, 0, pid_hv, cache_idx]
        gH_slice1 = gH_tiled[None, None, None, 1, pid_hv, cache_idx]
        gH_grp0 = cute.group_modes(gH_slice0, 0, 3)
        gH_grp1 = cute.group_modes(gH_slice1, 0, 3)
        sH_grp = cute.group_modes(sH, 0, 2)
        tHsH0, tHgH0 = cpasync.tma_partition(
            tma_atom_h,
            0,
            cute.make_layout(1),
            sH_grp,
            gH_grp0,
        )
        tHsH1, tHgH1 = cpasync.tma_partition(
            tma_atom_h,
            0,
            cute.make_layout(1),
            sH_grp,
            gH_grp1,
        )

        thr_mma = tiled_mma.get_slice(lane_id)
        # Build a register-only C fragment template (no SMEM).
        # partition_shape_C((T, 8)) returns the per-thread partition shape for
        # an (M=T, N=8) tile under the m16n8k16 MMA; make_fragment_C creates
        # a register fragment of that shape with the MMA's accumulator dtype (f32).
        tCsC = thr_mma.make_fragment_C(thr_mma.partition_shape_C((T, 8)))
        acc = cute.make_fragment_like(tCsC)
        _ldm_row = (lane_id % 8) + ((lane_id // 8) % 2) * Int32(8)

        EPT_TT = TT // THREADS

        # The warp-3 gamma/beta scalar loads were issued right after pid/lane
        # setup above; the _v7e_* registers are already in flight here.

        # ============================================================
        # cp.async stage 1: K + Q (8 bf16 / instr, .ca for L1 reuse)
        # ============================================================
        k_base = pid_b * sk_b + i_h * sk_h
        q_base = pid_b * sq_b + i_h * sq_h
        _gK_base = gK.iterator.toint()
        _gQ_base = gQ.iterator.toint()
        _sK_i32 = cute.recast_tensor(sK, cutlass.Int32)
        _sQ_i32 = cute.recast_tensor(sQ, cutlass.Int32)
        _kq_col_i32 = lane_id * Int32(2)
        _kpad_i32 = K_PADDED // 2
        _sK_base_async = sK.iterator.toint()
        _sQ_base_async = sQ.iterator.toint()
        _cv_yv0 = Int32(0)
        _cv_yv1 = Int32(0)
        _cv_yv2 = Int32(0)
        _cv_yv3 = Int32(0)
        if const_expr(self._fused_conv):
            # ---- conv fold: thread tidx owns q|k channel pair loc = 2*tidx of key head i_h (loc < 128: q, else k);
            # threads 0..63 also own v channel pair 2*tidx of value head pid_hv.
            _cv_x = gQ.iterator.toint()  # mixed_qkv base (the q slice starts at column 0)
            _cv_rs = Int64(_rs)
            _cv_tok0 = Int64(pid_b * Int32(4))
            _cv_off = Int64(gNacc.iterator[pid_b] - Int32(1))
            _cv_srow = gCS.iterator.toint() + (Int64(cache_idx) * cs_seq + _cv_off * Int64(cs_tok)) * Int64(2)
            _cv_cw = gCW.iterator.toint()
            _cv_loc = tidx * Int32(2)
            _cv_gch = i_h * Int32(K_DIM) + _cv_loc
            _cv_dst = _sQ_base_async + _cv_loc * Int32(2)
            if _cv_loc >= Int32(K_DIM):
                _cv_gch = H * Int32(K_DIM) + i_h * Int32(K_DIM) + _cv_loc - Int32(K_DIM)
                _cv_dst = _sK_base_async + (_cv_loc - Int32(K_DIM)) * Int32(2)
            _cv_sp = _cv_srow + Int64(_cv_gch) * Int64(2)
            _qs0 = _ldg_b32(_cv_sp)
            _qs1 = _ldg_b32(_cv_sp + Int64(cs_tok) * Int64(2))
            _qs2 = _ldg_b32(_cv_sp + Int64(cs_tok) * Int64(4))
            _qw0, _qw1, _qw2, _qw3 = _ldg_v4b32(_cv_cw + Int64(_cv_gch) * Int64(8))
            _cv_xp = _cv_x + (_cv_tok0 * _cv_rs + Int64(_cv_gch)) * Int64(2)
            _qx0 = _ldg_b32(_cv_xp)
            _qx1 = _ldg_b32(_cv_xp + _cv_rs * Int64(2))
            _qx2 = _ldg_b32(_cv_xp + _cv_rs * Int64(4))
            _qx3 = _ldg_b32(_cv_xp + _cv_rs * Int64(6))
            _vs0 = Int32(0)
            _vs1 = Int32(0)
            _vs2 = Int32(0)
            _vx0 = Int32(0)
            _vx1 = Int32(0)
            _vx2 = Int32(0)
            _vx3 = Int32(0)
            _vw0 = Int32(0)
            _vw1 = Int32(0)
            _vw2 = Int32(0)
            _vw3 = Int32(0)
            _cv_vgch = Int32(2) * H * Int32(K_DIM) + pid_hv * Int32(V_DIM_C) + tidx * Int32(2)
            _cv_vsp = _cv_srow + Int64(_cv_vgch) * Int64(2)
            if tidx < Int32(V_DIM_C // 2):
                _vs0 = _ldg_b32(_cv_vsp)
                _vs1 = _ldg_b32(_cv_vsp + Int64(cs_tok) * Int64(2))
                _vs2 = _ldg_b32(_cv_vsp + Int64(cs_tok) * Int64(4))
                _vw0, _vw1, _vw2, _vw3 = _ldg_v4b32(_cv_cw + Int64(_cv_vgch) * Int64(8))
                _cv_vxp = _cv_x + (_cv_tok0 * _cv_rs + Int64(_cv_vgch)) * Int64(2)
                _vx0 = _ldg_b32(_cv_vxp)
                _vx1 = _ldg_b32(_cv_vxp + _cv_rs * Int64(2))
                _vx2 = _ldg_b32(_cv_vxp + _cv_rs * Int64(4))
                _vx3 = _ldg_b32(_cv_vxp + _cv_rs * Int64(6))
            _qy0, _qy1, _qy2, _qy3 = _conv4(_qs0, _qs1, _qs2, _qx0, _qx1, _qx2, _qx3, _qw0, _qw1, _qw2, _qw3)
            _sts_b32(_cv_dst, _qy0)
            _sts_b32(_cv_dst + Int32(K_PADDED * 2), _qy1)
            _sts_b32(_cv_dst + Int32(2 * K_PADDED * 2), _qy2)
            _sts_b32(_cv_dst + Int32(3 * K_PADDED * 2), _qy3)
            if tidx < Int32(V_DIM_C // 2):
                _cv_yv0, _cv_yv1, _cv_yv2, _cv_yv3 = _conv4(_vs0, _vs1, _vs2, _vx0, _vx1, _vx2, _vx3,
                                                            _vw0, _vw1, _vw2, _vw3)
                # v pair: sole reader -> rolling conv-state update now (positions 0..5 = seq[1..6])
                _cv_vwp = gCS.iterator.toint() + (Int64(cache_idx) * cs_seq + Int64(_cv_vgch)) * Int64(2)
                _stg_b32_addr(_cv_vwp, _vs1)
                _stg_b32_addr(_cv_vwp + Int64(cs_tok) * Int64(2), _vs2)
                _stg_b32_addr(_cv_vwp + Int64(cs_tok) * Int64(4), _vx0)
                _stg_b32_addr(_cv_vwp + Int64(cs_tok) * Int64(6), _vx1)
                _stg_b32_addr(_cv_vwp + Int64(cs_tok) * Int64(8), _vx2)
                _stg_b32_addr(_cv_vwp + Int64(cs_tok) * Int64(10), _vx3)
            # q/k pairs are read by all VPK value-head CTAs of key head i_h: warp w of each CTA reads the same 32
            # pairs, so a per-(row, key head, warp) counter orders them; the last warp writes the rolled state.
            cute.arch.sync_warp()
            _cv_prev = Int32(0)
            _cv_ctr = gCtr.iterator.toint() + Int64((pid_b * H + i_h) * Int32(4) + warp_id) * Int64(4)
            if lane_id == Int32(0):
                _cv_prev = _atom_add_acqrel(_cv_ctr)
            _cv_prev = cute.arch.shuffle_sync(_cv_prev, Int32(0))
            if _cv_prev == (HV // H) - Int32(1):
                if lane_id == Int32(0):
                    _stg_b32_addr(_cv_ctr, Int32(0))
                _cv_qwp = gCS.iterator.toint() + (Int64(cache_idx) * cs_seq + Int64(_cv_gch)) * Int64(2)
                _stg_b32_addr(_cv_qwp, _qs1)
                _stg_b32_addr(_cv_qwp + Int64(cs_tok) * Int64(2), _qs2)
                _stg_b32_addr(_cv_qwp + Int64(cs_tok) * Int64(4), _qx0)
                _stg_b32_addr(_cv_qwp + Int64(cs_tok) * Int64(6), _qx1)
                _stg_b32_addr(_cv_qwp + Int64(cs_tok) * Int64(8), _qx2)
                _stg_b32_addr(_cv_qwp + Int64(cs_tok) * Int64(10), _qx3)
        for i in cutlass.range_constexpr(0 if self._fused_conv else T * K_DIM // (THREADS * 8)):
            _kq_group = tidx + i * THREADS
            _kq_row = _kq_group // Int32(K_DIM // 8)
            _kq_col_bf16_async = (_kq_group % Int32(K_DIM // 8)) * Int32(8)
            _smem_byte_off = _kq_row * Int32(K_PADDED * 2) + _kq_col_bf16_async * Int32(
                2
            )
            # When n_valid < T the q/k gmem tensors hold only n_valid rows; skip
            # the cp.async for rows >= n_valid (those would read OOB). The
            # sK/sQ[n_valid:T] smem tail is zeroed after the wait below.
            if const_expr(self._n_valid < T):
                if _kq_row < Int32(self._n_valid):
                    _cp_async_bf16x8(
                        _gK_base,
                        k_base + _kq_row * sk_t + _kq_col_bf16_async,
                        _sK_base_async + _smem_byte_off,
                    )
                    _cp_async_bf16x8(
                        _gQ_base,
                        q_base + _kq_row * sq_t + _kq_col_bf16_async,
                        _sQ_base_async + _smem_byte_off,
                    )
            else:
                _cp_async_bf16x8(
                    _gK_base,
                    k_base + _kq_row * sk_t + _kq_col_bf16_async,
                    _sK_base_async + _smem_byte_off,
                )
                _cp_async_bf16x8(
                    _gQ_base,
                    q_base + _kq_row * sq_t + _kq_col_bf16_async,
                    _sQ_base_async + _smem_byte_off,
                )
        _cp_async_commit_group()  # group 0 = K+Q

        # ============================================================
        # Issue the FIRST state-tile half (K=0..63); its load overlaps
        # Phase 1 + Phase 2. The SECOND half is issued later (right before
        # H GEMM half-1) once half-0 has finished reading the buffer.
        # mbarrier_arrive_and_expect_tx with V_DIM_C * K_HALF * 2 bytes.
        # cute.copy MUST stay OUTSIDE elect_one (else the GPU deadlocks).
        # ============================================================
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_h_ptr,
                    V_DIM_C * K_HALF * 2,  # 16384 B (half-tile)
                )
            cute.copy(tma_atom_h, tHgH0, tHsH0, tma_bar_ptr=mbar_h_ptr)

        # ============================================================
        # warp 3 computes gamma/beta in parallel with the cp.async pipeline,
        # consuming the gate/bias registers loaded at kernel entry.
        # ============================================================
        if warp_id == 3:
            log_alpha = f32(0.0)
            beta_val = f32(0.0)
            # Gate by _ab_rows: tail lanes [n_valid:T] are not loaded (their
            # _v7e_a/b regs are undefined), so they must keep log_alpha=beta=0. The causal
            # prefix-sum makes their (zero) contribution invisible to rows 0..n_valid-1.
            if lane_id < _ab_rows:
                a_val = _v7e_a_bf16
                b_val = _v7e_b_bf16
                A_log_val = _v7e_alog_bf16
                dt_val = _v7e_dt_bf16
                x = a_val + dt_val
                sp = cute.math.log(f32(1.0) + _exp_approx_f32(x))
                log_alpha = (f32(0.0) - _exp_approx_f32(A_log_val)) * sp
                beta_val = f32(1.0) / (f32(1.0) + _exp_approx_f32(f32(0.0) - b_val))
            cumsum = log_alpha
            for d in [1, 2, 4, 8]:
                prev = cute.arch.shuffle_sync(
                    cumsum, Int32(lane_id - d), Int32(0x0000FFFF), Int32(0x1F)
                )
                if lane_id >= d:
                    cumsum = cumsum + prev
            if lane_id < T:
                exp_g = _exp_approx_f32(cumsum)
                sGamma.iterator[T + lane_id] = exp_g
                # Store the log-domain cumsum in the free sGamma[0:T] slots so the
                # decay matrix can be formed as exp(cumsum_r - cumsum_c) directly (bounded
                # <=1 for the causal r>=c region) instead of exp(cumsum_r)*exp(-cumsum_c),
                # whose exp(-cumsum_c) overflows to inf for strong real decay (large A_log)
                # -> 0*inf = NaN. Mathematically identical, but NaN-safe.
                sGamma.iterator[lane_id] = cumsum
                sBeta.iterator[lane_id] = beta_val
                sBeta.iterator[T + lane_id] = f32(1.0) / exp_g

        # ============================================================
        # Wait for K+Q (group 0); H load is now driven by mbarrier (TMA),
        # not cp.async commit groups, so wait_group(0) is sufficient.
        # ============================================================
        _cp_async_wait_group_0()
        # (native-short-T) zero the sK/sQ tail rows [n_valid:T] that were NOT loaded
        # from gmem (q/k held only n_valid rows). This restores the documented
        # sK/sQ[..:T]=0 invariant the t_input<=8 zero-propagation proof relies on
        # (see the QT@V comment), previously supplied by the host zero-padding the
        # staged buffer. tidx in [0,THREADS=128) covers K_DIM=128 cols exactly;
        # pad cols [K_DIM:K_PADDED) are never read. The following sync_threads()
        # publishes both the loaded rows and these zeros before the L2-norm reads.
        if const_expr(self._n_valid < T):
            for _zr in cutlass.range_constexpr(self._n_valid, T):
                sK.iterator[_zr * K_PADDED + tidx] = io(0.0)
                sQ.iterator[_zr * K_PADDED + tidx] = io(0.0)
        sync_threads()

        # L2 norm for K (warps 0,1) and Q (warps 2,3) — t-aware warp skip.
        # At T<=8, warps 1 and 3 (which normalize rows 8..15) are dead-elided.
        # Rows 8..15 of sK/sQ feed only outputs that Phase-2 already gates
        # (T11 = diag(beta1) with no read of sMat[8..15, 8..15], Y/T10 skipped,
        # and the Phase-4 STG bottom half is t_input-gated).
        # Warps 0 and 2 still process 8 lane-rows even at T=4 — rows 4..7 of
        # sK/sQ are wrapper-zero-padded, so L2-norm is a harmless no-op
        # (zero * any_inv_norm = zero). Predicating individual lanes would
        # break the shuffle_sync (mask 0xFFFFFFFF requires all 32 warp lanes).
        # `self._t_input` is a Python int fixed at JIT compile time, so
        # const_expr produces 2 specializations: t_input<=8 and t_input=16.
        if const_expr(self._t_input <= 8):
            if warp_id == Int32(0):
                norm_row = lane_id // 4
                norm_quarter = lane_id % 4
                _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
                partial = f32(0.0)
                # (perf) rows >= t are zeros — their L2-norm RMW is a no-op
                # on zeros (bit-exact to skip); saves 32 LSU ops per tail
                # lane. The quad shuffles below still run on all 32 lanes
                # (tail partial stays 0; tail inv_norm is never stored).
                if norm_row < Int32(self._t_input):
                    for c in cutlass.range_constexpr(16):
                        packed = _sK_i32.iterator[_norm_off_i32 + 4 * c]
                        partial = _dot_sq_bf16x2(packed, partial)
                for d in [1, 2]:
                    other = cute.arch.shuffle_sync(
                        partial, Int32(lane_id ^ d), Int32(0xFFFFFFFF), Int32(0x1F)
                    )
                    partial = partial + other
                inv_norm = _rsqrt_approx_f32(partial + f32(EPS))
                if const_expr(self._kq_fix):
                    # keep raw k in sK; publish the inverse norm (0 for tail rows)
                    # zero rows (staged T<4 padding) get inv = 0: u~ = 0 instead of rsqrt(eps) * u, so no huge
                    # ring values (fp16 ring) and no 0 * inf; their contributions are exactly 0 either way.
                    if norm_quarter == Int32(0):
                        sGhist.iterator[48 + norm_row] = (
                            inv_norm if (norm_row < Int32(self._t_input) and partial > f32(0.0)) else f32(0.0)
                        )
                elif norm_row < Int32(self._t_input):
                    for c in cutlass.range_constexpr(16):
                        # (pack) store the normalized k in the PACK dtype
                        # (fp16 in combined mode -> true fp16 k-cache + no
                        # per-fragment GEMM conversion; bf16 otherwise).
                        _sK_i32.iterator[_norm_off_i32 + 4 * c] = _mul_packstore_f32(
                            _sK_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                        )
            if warp_id == Int32(2):
                norm_row = lane_id // 4
                norm_quarter = lane_id % 4
                _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
                partial = f32(0.0)
                # (perf) tail-lane skip as in the K norm above (bit-exact).
                if norm_row < Int32(self._t_input):
                    for c in cutlass.range_constexpr(16):
                        packed = _sQ_i32.iterator[_norm_off_i32 + 4 * c]
                        partial = _dot_sq_bf16x2(packed, partial)
                for d in [1, 2]:
                    other = cute.arch.shuffle_sync(
                        partial, Int32(lane_id ^ d), Int32(0xFFFFFFFF), Int32(0x1F)
                    )
                    partial = partial + other
                inv_norm = _rsqrt_approx_f32(partial + f32(EPS))
                inv_norm = inv_norm * scale
                if const_expr(self._kq_fix):
                    if norm_quarter == Int32(0):
                        sGhist.iterator[56 + norm_row] = (
                            inv_norm if (norm_row < Int32(self._t_input) and partial > f32(0.0)) else f32(0.0)
                        )
                    if norm_row < Int32(self._t_input):
                        for c in cutlass.range_constexpr(16):
                            # raw q dual-stored into sK rows [8:8+t) (packed tile)
                            _sK_i32.iterator[
                                _norm_off_i32 + 8 * (K_PADDED // 2) + 4 * c
                            ] = _sQ_i32.iterator[_norm_off_i32 + 4 * c]
                elif norm_row < Int32(self._t_input):
                    for c in cutlass.range_constexpr(16):
                        # (pack) normalized q in the PACK dtype (see K-norm).
                        _qn_val = _mul_packstore_f32(
                            _sQ_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                        )
                        _sQ_i32.iterator[_norm_off_i32 + 4 * c] = _qn_val
                        # (perf) fused packed-A-tile store: duplicate the
                        # normed q rows [0:t) into sK rows [8:8+t) (dual
                        # store) — removes the separate pack pass and one
                        # CTA barrier; the norm-end sync publishes both.
                        _sK_i32.iterator[
                            _norm_off_i32 + 8 * (K_PADDED // 2) + 4 * c
                        ] = _qn_val
            if warp_id == Int32(1):
                # (u-cache) publish the G ring + w_j/bdec while warps 0/2
                # L2-normalize (warp 1 is idle here at t<=8). w_j =
                # e^{G_P - G_j} for j < P else 0 — the j >= P mask is
                # MANDATORY (stale slots -> inf * 0 = NaN). G_P is lane
                # P-1's value, shuffled (all 32 lanes execute the shuffle).
                if lane_id < Int32(W_RING):
                    sGhist.iterator[lane_id] = _g_hist_f32
                _p_src = P_hist - 1 if P_hist > 0 else Int32(0)
                _gp_lane = cute.arch.shuffle_sync(
                    _g_hist_f32, _p_src, Int32(0xFFFFFFFF), Int32(0x1F)
                )
                _gp = _gp_lane if P_hist > 0 else f32(0.0)
                if lane_id < Int32(W_RING):
                    _w_j = (
                        _exp_approx_f32(_gp - _g_hist_f32)
                        if lane_id < P_hist
                        else f32(0.0)
                    )
                    sGhist.iterator[W_RING + lane_id] = _w_j
                if lane_id == Int32(0):
                    _bdec_v = _exp_approx_f32(_gp) if P_hist > 0 else f32(1.0)
                    sGhist.iterator[32] = _bdec_v
                # G-ring append: g_new[s] = G_P + cumsum_s. sGamma[0:T]
                # (log-domain cumsum, warp 3) was published by the K/Q-wait
                # sync before this norm phase.
                # (flush) flushing requests append with LOCAL decay (the
                # reference checkpoint becomes S_h, which absorbs e^{G_P}).
                # Safe this early: no CTA re-reads gGC after its entry LDG
                # (w_j live in sGhist), g rows are (cache_idx, hv)-exclusive,
                # and the target (base+P+s)&mask is past the live window.
                if lane_id < Int32(self._t_input):
                    # Ring append at (base+P+s) — PAST the live window for
                    # flush and verify rows alike. Value rule unchanged:
                    # flush rows restart with LOCAL decay (the fold absorbs
                    # e^{G_P} into the checkpoint).
                    _g_app_val = (
                        f32(0.0) if P_hist >= flush_min else _gp
                    ) + sGamma.iterator[lane_id]
                    _st_global_f32(
                        _gGC_base_st,
                        (ring_base + P_hist + lane_id) & Int32(RING_MASK),
                        _g_app_val,
                    )
        else:
            # T=16 path.
            if warp_id < 2:
                norm_row = warp_id * 8 + lane_id // 4
                norm_quarter = lane_id % 4
                _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
                partial = f32(0.0)
                for c in cutlass.range_constexpr(16):
                    packed = _sK_i32.iterator[_norm_off_i32 + 4 * c]
                    partial = _dot_sq_bf16x2(packed, partial)
                for d in [1, 2]:
                    other = cute.arch.shuffle_sync(
                        partial, Int32(lane_id ^ d), Int32(0xFFFFFFFF), Int32(0x1F)
                    )
                    partial = partial + other
                inv_norm = _rsqrt_approx_f32(partial + f32(EPS))
                for c in cutlass.range_constexpr(16):
                    _sK_i32.iterator[_norm_off_i32 + 4 * c] = _mul_bf16x2_f32(
                        _sK_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                    )
            if warp_id >= 2:
                norm_row = (warp_id - 2) * 8 + lane_id // 4
                norm_quarter = lane_id % 4
                _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
                partial = f32(0.0)
                for c in cutlass.range_constexpr(16):
                    packed = _sQ_i32.iterator[_norm_off_i32 + 4 * c]
                    partial = _dot_sq_bf16x2(packed, partial)
                for d in [1, 2]:
                    other = cute.arch.shuffle_sync(
                        partial, Int32(lane_id ^ d), Int32(0xFFFFFFFF), Int32(0x1F)
                    )
                    partial = partial + other
                inv_norm = _rsqrt_approx_f32(partial + f32(EPS))
                inv_norm = inv_norm * scale
                for c in cutlass.range_constexpr(16):
                    _sQ_i32.iterator[_norm_off_i32 + 4 * c] = _mul_bf16x2_f32(
                        _sQ_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                    )
        sync_threads()

        # ============================================================
        # (u-cache) PACKED A-TILE: the normed q rows [0:t) were already
        # dual-stored into sK rows [8:8+t) inside the warp-2 norm pass
        # (published by the sync above) — the H GEMM yields S0·k (rows
        # 0..t-1) AND S0·q (rows 8..8+t-1) from ONE streamed S0 read.
        # Safe with native-a/b at t<=8: Gram pollution from the packed
        # rows lands only where beta=0 (rows >= t) or r<c masks it.
        # k-ring append (one CTA per k-group) runs here, BEFORE the bdec
        # row-scale corrupts the normed k rows; it is a pure global write
        # of sK rows [0:t) (read-only vs the Grams below), so NO extra
        # barrier is needed — it overlaps the Grams.
        # ============================================================
        # The normed-k LDS is hoisted OUT of the append guard and kept as a
        # register snapshot: the append source must be the PRE-bdec-scale
        # values, and the ring append target (base+P+s)&mask is past every
        # sibling's fold-source window, so flush and verify rows append
        # identically right here. The LDS itself is safe for all
        # 128 threads (rows 0..15 of the sK tile); only the STG is gated.
        _gKC_base_st = gKC.iterator.toint()
        _kc_row = tidx // Int32(K_DIM // 8)
        _kc_pos = tidx % Int32(K_DIM // 8)
        _kv0, _kv1, _kv2, _kv3 = _lds_v4_b32(
            _sK_base_async + _kc_row * Int32(K_PADDED * 2) + _kc_pos * Int32(16)
        )
        # (ring-fp16) cache-fp16-ONLY mode: the packed tile is bf16, so repack
        # each bf16 pair to fp16 at the STG boundary (exact in range — normed
        # |k| <= 1). In the COMBINED mode the packed tile (hence this
        # snapshot) is ALREADY fp16 — true fp32->fp16 precision from the norm
        # pass — so the bytes go raw to the fp16 cache (no bf16 round-trip).
        # Compile-time no-op in the default mode.
        if const_expr(_RING_MIXED and not _PACK_MIXED):
            _kr0 = _repack_io2_to_rg2(_kv0)
            _kr1 = _repack_io2_to_rg2(_kv1)
            _kr2 = _repack_io2_to_rg2(_kv2)
            _kr3 = _repack_io2_to_rg2(_kv3)
        else:
            _kr0 = _kv0
            _kr1 = _kv1
            _kr2 = _kv2
            _kr3 = _kv3
        if (pid_hv % (HV // H)) == 0:
            # Ring append at (base+P+s)&mask — past the live window for flush
            # and verify rows alike (the fold reads [base, base+P); disjoint).
            if tidx < Int32(self._t_input * (K_DIM // 8)):
                _st_global_v4_b32(
                    _gKC_base_st + _kc_pool_e64 * 2,
                    ((ring_base + P_hist + _kc_row) & Int32(RING_MASK)) * K_DIM
                    + _kc_pos * Int32(8),
                    _kr0,
                    _kr1,
                    _kr2,
                    _kr3,
                )

        # KKT (warps 0-1) || QKT (warps 2-3) — direct acc → SMEM writes.
        acc.fill(f32(0.0))
        _sK_int = sK.iterator.toint()
        _sQ_int = sQ.iterator.toint()
        _rs_kpad = Int32(K_PADDED * 2)
        _lane_mod16 = lane_id & Int32(15)
        _lane_hi = (lane_id >> Int32(4)) * Int32(16)
        _lane_mod8 = lane_id % Int32(8)
        _lane_b_col = ((lane_id >> Int32(3)) & Int32(1)) * Int32(16)
        for kk_group in cutlass.range_constexpr(K_DIM // 16 // 4):
            k_group_off = kk_group * 4 * 16 * Int32(2)
            if warp_id < 2:
                col_off = warp_id * 8
                _a_base = _sK_int + _lane_mod16 * _rs_kpad + _lane_hi + k_group_off
                _b_direct = (
                    _sK_int
                    + (col_off + _lane_mod8) * _rs_kpad
                    + k_group_off
                    + _lane_b_col
                )
                acc.iterator[0], acc.iterator[1], acc.iterator[2], acc.iterator[3] = (
                    _fused_ab_4mma_serial_brow(
                        _a_base,
                        _b_direct,
                        acc.iterator[0],
                        acc.iterator[1],
                        acc.iterator[2],
                        acc.iterator[3],
                    )
                )
            if warp_id >= 2:
                col_off = (warp_id - Int32(2)) * Int32(8)
                _a_base = _sQ_int + _lane_mod16 * _rs_kpad + _lane_hi + k_group_off
                _b_direct = (
                    _sK_int
                    + (col_off + _lane_mod8) * _rs_kpad
                    + k_group_off
                    + _lane_b_col
                )
                acc.iterator[0], acc.iterator[1], acc.iterator[2], acc.iterator[3] = (
                    _fused_ab_4mma_serial_brow(
                        _a_base,
                        _b_direct,
                        acc.iterator[0],
                        acc.iterator[1],
                        acc.iterator[2],
                        acc.iterator[3],
                    )
                )
        _r0 = lane_id // 4
        _c0 = (lane_id & 3) * 2
        if const_expr(self._kq_fix):
            # sGhist[48 + i] is the inverse norm of packed-tile row i (k rows 0..7, q rows 8..15)
            if warp_id < 2:
                col_off = warp_id * 8
                _iv_r0 = sGhist.iterator[48 + _r0]
                _iv_r8 = sGhist.iterator[56 + _r0]
                _iv_c0 = sGhist.iterator[48 + col_off + _c0]
                _iv_c1 = sGhist.iterator[48 + col_off + _c0 + 1]
                acc.iterator[0] = acc.iterator[0] * _iv_r0 * _iv_c0
                acc.iterator[1] = acc.iterator[1] * _iv_r0 * _iv_c1
                acc.iterator[2] = acc.iterator[2] * _iv_r8 * _iv_c0
                acc.iterator[3] = acc.iterator[3] * _iv_r8 * _iv_c1
            if warp_id >= 2:
                col_off = (warp_id - Int32(2)) * Int32(8)
                _iv_q0 = sGhist.iterator[56 + _r0]  # sQ rows 0..7 = raw q; rows 8..15 are zero rows
                _iv_c0 = sGhist.iterator[48 + col_off + _c0]
                _iv_c1 = sGhist.iterator[48 + col_off + _c0 + 1]
                acc.iterator[0] = acc.iterator[0] * _iv_q0 * _iv_c0
                acc.iterator[1] = acc.iterator[1] * _iv_q0 * _iv_c1
                acc.iterator[2] = f32(0.0)
                acc.iterator[3] = f32(0.0)
        if warp_id < 2:
            col_off = warp_id * 8
            sMat.iterator[_smat_off(_r0, col_off + _c0)] = acc.iterator[0]
            sMat.iterator[_smat_off(_r0, col_off + _c0 + 1)] = acc.iterator[1]
            sMat.iterator[_smat_off(_r0 + 8, col_off + _c0)] = acc.iterator[2]
            sMat.iterator[_smat_off(_r0 + 8, col_off + _c0 + 1)] = acc.iterator[3]
        if warp_id >= 2:
            col_off = (warp_id - Int32(2)) * Int32(8)
            sNegL.iterator[_r0 * BF_PAD + col_off + _c0] = acc.iterator[0].to(io)
            sNegL.iterator[_r0 * BF_PAD + col_off + _c0 + 1] = acc.iterator[1].to(io)
            sNegL.iterator[(_r0 + 8) * BF_PAD + col_off + _c0] = acc.iterator[2].to(io)
            sNegL.iterator[(_r0 + 8) * BF_PAD + col_off + _c0 + 1] = acc.iterator[3].to(
                io
            )
        sync_threads()

        # ============================================================
        # (u-cache) tenant #2 of qv_buf: khist tile [W_RING, K_PADDED] via
        # cp.async.ca (the tile is re-read by the 4 sibling CTAs of the
        # k-group). sQ is dead after the Grams above. Landing is awaited
        # (wait + one barrier) after Phase 2, before the scores GEMM.
        # ============================================================
        _gKC_base_rd = gKC.iterator.toint() + _kc_pool_e64 * 2
        _kc_rd_base = Int32(0)
        # (perf) the khist wave is ISSUED AND WAITED ONLY BY WARPS 2-3 (the
        # tile's sole consumers, in the scores GEMM): warps 0-1 enter the
        # block inverse without ever stalling on ring-load latency. The
        # cross-warp (2<->3) visibility is ordered by their own wait_group
        # + the 64-thread named barrier at the scores GEMM. Rows >= P stay
        # unread (w-masked downstream).
        if warp_id >= 2:
            for _kh in cutlass.range_constexpr(W_RING * K_DIM // (64 * 8)):
                _kh_group = (tidx - Int32(64)) + _kh * Int32(64)
                _kh_row = _kh_group // Int32(K_DIM // 8)
                _kh_col = (_kh_group % Int32(K_DIM // 8)) * Int32(8)
                if _kh_row < P_hist:
                    # ring-rotate at load: gmem row (base+j)&mask lands in
                    # smem/logical row j — downstream tiles unchanged.
                    _cp_async_bf16x8(
                        _gKC_base_rd,
                        _kc_rd_base
                        + ((ring_base + _kh_row) & Int32(RING_MASK)) * K_DIM
                        + _kh_col,
                        _sQ_base_async
                        + _kh_row * Int32(K_PADDED * 2)
                        + _kh_col * Int32(2),
                    )
            _cp_async_commit_group()

        # ============================================================
        # PHASE 2: log-depth Neumann inverse
        # ============================================================
        for idx in cutlass.range_constexpr(EPT_TT):
            flat = tidx + idx * THREADS
            r = flat // T
            c = flat % T
            # (perf) tail short-circuit: entries with r >= t or c >= t are
            # structurally ZERO in both outputs under the packed-tile
            # invariants (beta = 0 tail rows, zero k/q rows, r < c masking)
            # — write the zeros directly and skip their sNegL/sMat/sGamma
            # loads. Bit-exact: the skipped loads only ever fed values
            # that reduced to 0 for these entries.
            if r < Int32(self._t_input) and c < Int32(self._t_input):
                # Stable decay: exp(cumsum_r - cumsum_c)
                # directly (<=1 for r>c) instead of
                # exp(cumsum_r)*exp(-cumsum_c) (overflows). exp_gij is only
                # consumed for r>=c (below); r<c value is discarded.
                exp_gij = (
                    f32(1.0)
                    if r == c
                    else (
                        _exp_approx_f32(sGamma.iterator[r] - sGamma.iterator[c])
                        if r > c
                        else f32(0.0)
                    )
                )
                qkt = sNegL.iterator[r * BF_PAD + c].to(f32)
                sNegL.iterator[r * BF_PAD + c] = (
                    (qkt * exp_gij).to(io) if r >= c else io(0.0)
                )
                kkt_val = sMat.iterator[_smat_off(r, c)]
                negL_val = (
                    (f32(0.0) - sBeta.iterator[r] * exp_gij * kkt_val)
                    if r > c
                    else f32(0.0)
                )
                negL_bf = negL_val.to(io)
                sTmat.iterator[r * BF_PAD + c] = negL_bf
            else:
                sNegL.iterator[r * BF_PAD + c] = io(0.0)
                sTmat.iterator[r * BF_PAD + c] = io(0.0)
        # (u-cache) NOTE: the khist tile is NOT waited here — warps 2-3 own
        # the wave and wait for it themselves at the scores GEMM, so this
        # barrier only publishes phase-2's writes.
        sync_threads()

        _r0 = lane_id // 4
        _c0 = (lane_id & 3) * 2

        # ============================================================
        # BLOCK INVERSE for T=16 — register-resident forward substitution.
        # Each lane owns one column of the 8x8 result, holding all 8 row
        # values in registers, avoiding the shared-memory store-then-load path.
        # Algorithm:
        #   T00 = solve(I - M00, diag(beta0))  [warp 0, X in regs]
        #   T11 = solve(I - M11, diag(beta1))  [warp 1, X in regs] (parallel)
        #   Y   = M10 @ T00                    [warp 0, scalar 8x8 product]
        #   T10 = solve(I - M11, Y)            [warp 1, X in regs]
        #   sTmat = [[T00, 0], [T10, T11]] (bf16)
        #
        # v14_t_aware Stage B.2: For t_input ≤ 8, M's nonzero region is the
        # top-left 8×8 only (rows/cols ≥ T are zero in K → M block-rows/cols
        # ≥ 8 are zero). Therefore at T≤8:
        #   - T11 = solve(I, diag(β1)) = diag(β1) directly (skip 28-MAC forward sub)
        #   - Y   = M10 @ T00 = 0 (skip entire 8×8 product)
        #   - T10 = solve(I, 0) = 0 (skip 28-MAC forward sub)
        # And for t_input ≤ 4: M00's bottom 4 rows are also zero, so T00 rows
        # 4..7 collapse to diag(β[r]) directly (skip 4 forward-sub iterations).
        # The const_expr branches are compile-time at JIT (3 specializations:
        # t_input ∈ {4, 8, 16}); at t_input=16 the SKIP branch is dead-code
        # eliminated. sync_threads() topology is preserved across paths.
        # ============================================================

        # === Step 1: parallel 8x8 diagonal forward substitutions (register-resident) ===
        if const_expr(self._t_input <= 8):
            # (u-cache) scores GEMM on warps 2-3, CONCURRENT with the
            # warp-0/1 block inverse: scores[j, col] = khist_j · packed_col
            # (cols 0..t-1 = ·k_hat, cols 8..8+t-1 = ·q_hat), staged
            # w-scaled + transposed + bf16 into sWScores (the contraction
            # MMA's A operand). Same MMA pattern as KKT with A = khist.
            # (perf) skipped entirely at P=0 (CTA-uniform branch): the
            # contraction is also skipped there, so sWScores has no reader.
            if warp_id >= 2 and P_hist > 0:
                # own-wave wait: each of the 64 issuing threads drains its
                # own cp.async groups, then the named barrier orders the
                # cross-warp (2<->3) SMEM visibility. Warps 0-1 are already
                # deep in the block inverse at this point.
                _cp_async_wait_group_0()
                _bar_sync_1_64()
                # (ring-fp16) the khist tile stays RAW in the ring dtype:
                # the scores GEMM below converts the PACKED tile's B
                # fragments in registers instead (exact in range) — no SMEM
                # convert pass, no extra barrier, and the flush snapshot
                # naturally captures ring-dtype bytes for the fold.
                acc.fill(f32(0.0))
                _sc_col_off = (warp_id - Int32(2)) * Int32(8)
                for _sc_g in cutlass.range_constexpr(K_DIM // 16 // 4):
                    _sc_k_off = _sc_g * 4 * 16 * Int32(2)
                    _sc_a = _sQ_int + _lane_mod16 * _rs_kpad + _lane_hi + _sc_k_off
                    _sc_b = (
                        _sK_int
                        + (_sc_col_off + _lane_mod8) * _rs_kpad
                        + _sc_k_off
                        + _lane_b_col
                    )
                    (
                        acc.iterator[0],
                        acc.iterator[1],
                        acc.iterator[2],
                        acc.iterator[3],
                    ) = _fused_ab_4mma_serial_brow_rgb(
                        _sc_a,
                        _sc_b,
                        acc.iterator[0],
                        acc.iterator[1],
                        acc.iterator[2],
                        acc.iterator[3],
                    )
                _sc_r0 = lane_id // 4
                _sc_c0 = (lane_id & 3) * 2
                # (perf) stage TRANSPOSED + w-scaled + bf16: acc element [e]
                # is C[j, col] with j = _sc_r0 (+8 for e in {2,3}); store
                # sWScores[col, j] = w_j * C[j, col]. Columns j >= P carry
                # w = 0, so the stale u rows they meet in the contraction
                # MMA contribute exact zeros.
                _sc_w_lo = sGhist.iterator[W_RING + _sc_r0]
                _sc_w_hi = sGhist.iterator[W_RING + _sc_r0 + 8]
                if const_expr(self._kq_fix):
                    # C[j, col] = k_raw_hist_j . x_raw_col -> * inv|x_col| (packed col: k 0..7, q 8..15)
                    _sc_iv0 = sGhist.iterator[48 + _sc_col_off + _sc_c0]
                    _sc_iv1 = sGhist.iterator[48 + _sc_col_off + _sc_c0 + 1]
                    acc.iterator[0] = acc.iterator[0] * _sc_iv0
                    acc.iterator[1] = acc.iterator[1] * _sc_iv1
                    acc.iterator[2] = acc.iterator[2] * _sc_iv0
                    acc.iterator[3] = acc.iterator[3] * _sc_iv1
                # (ring-fp16) staged in the RING dtype: the contraction MMA
                # pairs this A operand with the raw u tile (B), so both
                # sides carry the ring element type. `.to(ring_ty)` == the
                # old `.to(io)` in the default mode.
                sWScores.iterator[(_sc_col_off + _sc_c0) * BF_PAD + _sc_r0] = (
                    acc.iterator[0] * _sc_w_lo
                ).to(ring_ty)
                sWScores.iterator[(_sc_col_off + _sc_c0 + 1) * BF_PAD + _sc_r0] = (
                    acc.iterator[1] * _sc_w_lo
                ).to(ring_ty)
                sWScores.iterator[(_sc_col_off + _sc_c0) * BF_PAD + _sc_r0 + 8] = (
                    acc.iterator[2] * _sc_w_hi
                ).to(ring_ty)
                sWScores.iterator[(_sc_col_off + _sc_c0 + 1) * BF_PAD + _sc_r0 + 8] = (
                    acc.iterator[3] * _sc_w_hi
                ).to(ring_ty)
            # === T≤8 PATH: only T00 (warp 0) is real work; T11 = diag(β1) ===
            if warp_id == Int32(0):
                if lane_id < Int32(8):
                    _col = lane_id
                    _x_t00 = [None] * 8
                    _x_t00[0] = (
                        sBeta.iterator[Int32(0)] if _col == Int32(0) else f32(0.0)
                    )
                    if const_expr(self._t_input <= 4):
                        # Real forward-sub for rows 0..3; rows 4..7 collapse to diag(β[r])
                        for _r in cutlass.range_constexpr(1, 4):
                            _accum = (
                                sBeta.iterator[Int32(_r)]
                                if _col == Int32(_r)
                                else f32(0.0)
                            )
                            for _k in cutlass.range_constexpr(_r):
                                _m_rk = sTmat.iterator[Int32(_r * BF_PAD + _k)].to(f32)
                                _accum = _accum + _m_rk * _x_t00[_k]
                            _x_t00[_r] = _accum
                        for _r in cutlass.range_constexpr(4, 8):
                            _x_t00[_r] = (
                                sBeta.iterator[Int32(_r)]
                                if _col == Int32(_r)
                                else f32(0.0)
                            )
                    else:
                        # T=8: real forward-sub all 8 rows
                        for _r in cutlass.range_constexpr(1, 8):
                            _accum = (
                                sBeta.iterator[Int32(_r)]
                                if _col == Int32(_r)
                                else f32(0.0)
                            )
                            for _k in cutlass.range_constexpr(_r):
                                _m_rk = sTmat.iterator[Int32(_r * BF_PAD + _k)].to(f32)
                                _accum = _accum + _m_rk * _x_t00[_k]
                            _x_t00[_r] = _accum
                    # Spill T00 column to sMat[0:8, col]
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(_r, _col)] = _x_t00[_r]
            if warp_id == Int32(1):
                # T11 = diag(β1) — 28 forward-sub MACs collapse to a single column write.
                # Step 4 (sTmat stage) reads sMat[8:16, 8:16] from this region.
                if lane_id < Int32(8):
                    _col = lane_id
                    for _r in cutlass.range_constexpr(8):
                        _v = (
                            sBeta.iterator[Int32(8 + _r)]
                            if _col == Int32(_r)
                            else f32(0.0)
                        )
                        sMat.iterator[_smat_off(8 + _r, 8 + _col)] = _v
            # No barrier here: in the t<=8 path there are zero cross-warp
            # dependencies between the phase-2 barrier and the final solve
            # barrier — step 2 is a no-op skip and each warp writes disjoint
            # sMat regions that are only read after the CTA-wide barrier
            # before Step 4.

            # === Step 2 SKIP: Y = M10 @ T00 = 0 ===
            # sMat[0:8, 8:16] (top-right) — no need to write zeros: Step 4's
            # stage line `_out0_v11 = io(0.0) if (_r0_v11 < 8 and _c0_v11 >= 8)`
            # already forces sTmat top-right to 0 regardless of sMat content.

            # === Step 3 SKIP: T10 = solve(I, 0) = 0 → write zeros to sMat[8:16, 0:8] ===
            if warp_id == Int32(1):
                if lane_id < Int32(8):
                    _col = lane_id
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(8 + _r, _col)] = f32(0.0)
            sync_threads()
        else:
            # === T=16 path (block inverse) ===
            if warp_id == Int32(0):
                if lane_id < Int32(8):
                    _col = lane_id
                    # X_t00[r] = T00[r, col] (lane-private, fp32 register)
                    _x_t00 = [None] * 8
                    # Row 0: T00[0, col] = (col==0) * beta0[0]
                    _x_t00[0] = (
                        sBeta.iterator[Int32(0)] if _col == Int32(0) else f32(0.0)
                    )
                    for _r in cutlass.range_constexpr(1, 8):
                        _accum = (
                            sBeta.iterator[Int32(_r)] if _col == Int32(_r) else f32(0.0)
                        )
                        for _k in cutlass.range_constexpr(_r):
                            # M0[r, k] broadcast LDS (all 8 active lanes read same addr)
                            _m_rk = sTmat.iterator[Int32(_r * BF_PAD + _k)].to(f32)
                            _accum = _accum + _m_rk * _x_t00[_k]  # register read
                        _x_t00[_r] = _accum
                    # Spill T00 column to sMat[0:8, col] for use by Y product
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(_r, _col)] = _x_t00[_r]
            if warp_id == Int32(1):
                if lane_id < Int32(8):
                    _col = lane_id
                    _x_t11 = [None] * 8
                    _x_t11[0] = (
                        sBeta.iterator[Int32(8)] if _col == Int32(0) else f32(0.0)
                    )
                    for _r in cutlass.range_constexpr(1, 8):
                        _accum = (
                            sBeta.iterator[Int32(8 + _r)]
                            if _col == Int32(_r)
                            else f32(0.0)
                        )
                        for _k in cutlass.range_constexpr(_r):
                            _m_rk = sTmat.iterator[
                                Int32((8 + _r) * BF_PAD + 8 + _k)
                            ].to(f32)
                            _accum = _accum + _m_rk * _x_t11[_k]
                        _x_t11[_r] = _accum
                    # Spill T11 to sMat[8:16, 8:16]
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(8 + _r, 8 + _col)] = _x_t11[_r]
            sync_threads()

            # === Step 2: Y = M10 @ T00 → sMat[0:8, 8:16] ===
            # 64 outputs / 32 lanes, 2 passes × 4 rows per pass.
            if warp_id == Int32(0):
                for _p in cutlass.range_constexpr(2):
                    _i = Int32(_p * 4) + (lane_id >> Int32(3))
                    _j = lane_id & Int32(7)
                    _y_ij = f32(0.0)
                    for _k in cutlass.range_constexpr(8):
                        _m_ik = sTmat.iterator[
                            (Int32(8) + _i) * Int32(BF_PAD) + Int32(_k)
                        ].to(f32)
                        _t_kj = sMat.iterator[_smat_off(_k, _j)]
                        _y_ij = _y_ij + _m_ik * _t_kj
                    sMat.iterator[_smat_off(_i, 8 + _j)] = _y_ij
            sync_threads()

            # === Step 3: T10 = solve(I - M11, Y) → sMat[8:16, 0:8] (register-resident) ===
            if warp_id == Int32(1):
                if lane_id < Int32(8):
                    _col = lane_id
                    _x_t10 = [None] * 8
                    # Row 0: T10[0, col] = Y[0, col] = sMat[0 * T + 8 + col]
                    _x_t10[0] = sMat.iterator[_smat_off(0, 8 + _col)]
                    for _r in cutlass.range_constexpr(1, 8):
                        _accum = sMat.iterator[_smat_off(_r, 8 + _col)]
                        for _k in cutlass.range_constexpr(_r):
                            _m_rk = sTmat.iterator[
                                Int32((8 + _r) * BF_PAD + 8 + _k)
                            ].to(f32)
                            _accum = _accum + _m_rk * _x_t10[_k]
                        _x_t10[_r] = _accum
                    # Spill T10 to sMat[8:16, 0:8]
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(8 + _r, _col)] = _x_t10[_r]
            sync_threads()

        # === Step 4: stage final Tmat (bf16) to sTmat, zero top-right ===
        _flat0_v11 = tidx
        _flat1_v11 = tidx + Int32(THREADS)
        _r0_v11 = _flat0_v11 // Int32(T)
        _c0_v11 = _flat0_v11 % Int32(T)
        _r1_v11 = _flat1_v11 // Int32(T)
        _c1_v11 = _flat1_v11 % Int32(T)
        _v0_v11 = sMat.iterator[_smat_off(_r0_v11, _c0_v11)]
        _v1_v11 = sMat.iterator[_smat_off(_r1_v11, _c1_v11)]
        _out0_v11 = (
            io(0.0) if (_r0_v11 < Int32(8) and _c0_v11 >= Int32(8)) else _v0_v11.to(io)
        )
        _out1_v11 = (
            io(0.0) if (_r1_v11 < Int32(8) and _c1_v11 >= Int32(8)) else _v1_v11.to(io)
        )
        sTmat.iterator[_r0_v11 * BF_PAD + _c0_v11] = _out0_v11
        sTmat.iterator[_r1_v11 * BF_PAD + _c1_v11] = _out1_v11
        sync_threads()

        # ============================================================
        # (flush) khist REGISTER SNAPSHOT — flushing CTAs keep the whole
        # khist tile (16 rows x 128 cols bf16 = 8 i32/thread) in registers
        # across the qv_buf tenant switch: the tail fold needs khist AND the
        # u tile simultaneously, but they are sequential tenants of the same
        # shared buffer, so the snapshot avoids a second gKC round-trip at
        # the tail. Rows >= P hold stale-but-finite
        # bytes; the fold's A-operand w-mask zeroes their contribution.
        # Thread t owns row (t//8), i32 cols (t%8)*8 .. +8 (bf16 cols
        # (t%8)*16 .. +16). The gated sync orders the snapshot before any
        # thread's u cp.async can overwrite the tile (CTA-uniform branch).
        # ============================================================
        _khs0 = Int32(0)
        _khs1 = Int32(0)
        _khs2 = Int32(0)
        _khs3 = Int32(0)
        _khs4 = Int32(0)
        _khs5 = Int32(0)
        _khs6 = Int32(0)
        _khs7 = Int32(0)
        if P_hist >= flush_min:
            _khsnap_off = (tidx // Int32(8)) * Int32(K_PADDED // 2) + (
                tidx % Int32(8)
            ) * Int32(8)
            _khs0 = _sQ_i32.iterator[_khsnap_off + 0]
            _khs1 = _sQ_i32.iterator[_khsnap_off + 1]
            _khs2 = _sQ_i32.iterator[_khsnap_off + 2]
            _khs3 = _sQ_i32.iterator[_khsnap_off + 3]
            _khs4 = _sQ_i32.iterator[_khsnap_off + 4]
            _khs5 = _sQ_i32.iterator[_khsnap_off + 5]
            _khs6 = _sQ_i32.iterator[_khsnap_off + 6]
            _khs7 = _sQ_i32.iterator[_khsnap_off + 7]
            sync_threads()

        # ============================================================
        # (u-cache) tenant #3 of qv_buf: the u tile [W_RING, V_PADDED] bf16
        # via cp.async.cg (single consumer; keep L1 for k/q/v). khist's
        # last read was the scores GEMM (pre-Step-4 sync, so the tenant
        # handoff is ordered). Landing is awaited in the TMA half-1
        # shadow, right before the history contraction consumes it.
        # ============================================================
        _gUC_base = gUC.iterator.toint() + _uc_pool_e64 * 2
        _uc_base = Int32(0)
        _sV_base_async_u = sV.iterator.toint()
        for _uh in cutlass.range_constexpr(W_RING * V_DIM_C // (THREADS * 8)):
            _uh_group = tidx + _uh * THREADS
            _uh_row = _uh_group // Int32(V_DIM_C // 8)
            _uh_col = (_uh_group % Int32(V_DIM_C // 8)) * Int32(8)
            # (perf) same row < P predication as the khist wave — slots
            # >= P are w-masked; skip their DRAM reads.
            if _uh_row < P_hist:
                _cp_async_bf16x8_cg(
                    _gUC_base,
                    _uc_base
                    + ((ring_base + _uh_row) & Int32(RING_MASK)) * V_DIM
                    + _uh_col,
                    _sV_base_async_u
                    + _uh_row * Int32(V_PADDED * 2)
                    + _uh_col * Int32(2),
                )
        _cp_async_commit_group()

        # ============================================================
        # bdec fold: scale the packed A-tile rows (k rows [0:t), q rows
        # [8:8+t)) by bdec = e^{G_P} so the H GEMM directly yields
        # bdec*(S0·x); the history term joins via the contraction below.
        # The sync at the H-half-0 mbarrier wait publishes these writes.
        # ============================================================
        # Skipped at P=0: bdec = 1 there and _mul_bf16x2_f32 by 1.0 is
        # value-preserving, so the pass is a pure no-op — save the RMWs.
        if const_expr(self._kq_fix):
            _bdec_f = sGhist.iterator[32] if P_hist > 0 else f32(1.0)
            for _bs in cutlass.range_constexpr(
                2 * self._t_input * (K_DIM // 2) // THREADS
            ):
                _bs_idx = tidx + _bs * THREADS
                _bs_rr = _bs_idx // Int32(K_DIM // 2)
                _bs_col = _bs_idx % Int32(K_DIM // 2)
                _bs_row = (
                    _bs_rr
                    if _bs_rr < Int32(self._t_input)
                    else Int32(8 - self._t_input) + _bs_rr
                )
                _sK_i32.iterator[_bs_row * _kpad_i32 + _bs_col] = _mul_pack_f32(
                    _sK_i32.iterator[_bs_row * _kpad_i32 + _bs_col],
                    _bdec_f * sGhist.iterator[48 + _bs_row],
                )
        elif P_hist > 0:
            _bdec = sGhist.iterator[32]
            for _bs in cutlass.range_constexpr(
                2 * self._t_input * (K_DIM // 2) // THREADS
            ):
                _bs_idx = tidx + _bs * THREADS
                _bs_rr = _bs_idx // Int32(K_DIM // 2)
                _bs_col = _bs_idx % Int32(K_DIM // 2)
                _bs_row = (
                    _bs_rr
                    if _bs_rr < Int32(self._t_input)
                    else Int32(8 - self._t_input) + _bs_rr
                )
                # (pack) the packed A-tile is pack_ty here (fp16 in combined
                # mode) — scale in that dtype so the H GEMM stays conversion-
                # free. Identical to _mul_bf16x2_f32 when pack_ty == bf16.
                _sK_i32.iterator[_bs_row * _kpad_i32 + _bs_col] = _mul_pack_f32(
                    _sK_i32.iterator[_bs_row * _kpad_i32 + _bs_col], _bdec
                )

        # The V load is issued later, in the TMA half-1 shadow below — sV's
        # region (qv_buf) is occupied by the u tile until the history
        # contraction has read it.

        # ============================================================
        # Wait for the state tile to land in sH via mbarrier. The TMA store
        # uses the async proxy; ldmatrix uses the generic proxy —
        # fence_view_async_shared crosses the proxy boundary. We do NOT wait
        # for V here — its cp.async runs in parallel with the H GEMM below,
        # and wait_group(0) for V fires just before the QT@V consumer.
        # ============================================================
        cute.arch.mbarrier_wait(mbar_h_ptr, 0)
        cute.arch.fence_view_async_shared()
        sync_threads()

        # ============================================================
        # H GEMM: WH[16, 128] = A[16, 128] @ H^T, where A is the packed
        # [k; q] tile in sK. 4 warps x 4 V-groups (8 rows each) x 8 K-tiles
        # (16 K each) = 128 MMAs / 4 warps = 32 MMAs per warp.
        # ============================================================
        wh_acc_0 = cute.make_fragment_like(tCsC)
        wh_acc_0.fill(f32(0.0))
        wh_acc_1 = cute.make_fragment_like(tCsC)
        wh_acc_1.fill(f32(0.0))
        wh_acc_2 = cute.make_fragment_like(tCsC)
        wh_acc_2.fill(f32(0.0))
        wh_acc_3 = cute.make_fragment_like(tCsC)
        wh_acc_3.fill(f32(0.0))

        _sK_base_vl = (
            sK.iterator.toint()
        )  # A operand (packed tile in sK, K_PADDED stride)
        _sH_base_vl = (
            sH.iterator.toint()
        )  # B operand (state in sH, SW128-swizzled, half-K)
        _rs_a = Int32(K_PADDED * 2)  # 272 — sK row stride (padded, full K)
        _rs_b = Int32(K_HALF * 2)  # 128 — sH row stride (half-K, SW128)

        # Per-warp V-group base (warp_id * 32 V-rows). For B-fragment ldmatrix.x2:
        # lane_id (0..31) maps to (lane%8) row × ((lane//8)%2) 16-col group.
        _b_lane_row = lane_id % Int32(8)
        _b_col_inner = ((lane_id >> Int32(3)) & Int32(1)) * Int32(16)  # 0 or 16 bytes
        _vg_base_row = warp_id * Int32(32)

        # ============================================================
        # H GEMM HALF-0 (ka=0..3, uses sH for K=0..63). sH currently holds
        # the state's K=0..63 columns from the first TMA load.
        # ============================================================
        for ka_local in cutlass.range_constexpr(4):
            col_byte_off_a = Int32(ka_local * 16 * 2)  # sK K=0..63
            col_byte_off_b = Int32(ka_local * 16 * 2)  # sH K=0..63
            _a_addr = _sK_base_vl + _lane_mod16 * _rs_a + _lane_hi + col_byte_off_a
            _b0_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(0) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b1_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(8) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b2_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(16) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b3_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(24) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b0 = _sw128_xor(_b0_l)
            _b1 = _sw128_xor(_b1_l)
            _b2 = _sw128_xor(_b2_l)
            _b3 = _sw128_xor(_b3_l)

            _r = _h_gemm_4v(
                _a_addr,
                _b0,
                _b1,
                _b2,
                _b3,
                wh_acc_0.iterator[0],
                wh_acc_0.iterator[1],
                wh_acc_0.iterator[2],
                wh_acc_0.iterator[3],
                wh_acc_1.iterator[0],
                wh_acc_1.iterator[1],
                wh_acc_1.iterator[2],
                wh_acc_1.iterator[3],
                wh_acc_2.iterator[0],
                wh_acc_2.iterator[1],
                wh_acc_2.iterator[2],
                wh_acc_2.iterator[3],
                wh_acc_3.iterator[0],
                wh_acc_3.iterator[1],
                wh_acc_3.iterator[2],
                wh_acc_3.iterator[3],
            )
            wh_acc_0.iterator[0] = _r[0]
            wh_acc_0.iterator[1] = _r[1]
            wh_acc_0.iterator[2] = _r[2]
            wh_acc_0.iterator[3] = _r[3]
            wh_acc_1.iterator[0] = _r[4]
            wh_acc_1.iterator[1] = _r[5]
            wh_acc_1.iterator[2] = _r[6]
            wh_acc_1.iterator[3] = _r[7]
            wh_acc_2.iterator[0] = _r[8]
            wh_acc_2.iterator[1] = _r[9]
            wh_acc_2.iterator[2] = _r[10]
            wh_acc_2.iterator[3] = _r[11]
            wh_acc_3.iterator[0] = _r[12]
            wh_acc_3.iterator[1] = _r[13]
            wh_acc_3.iterator[2] = _r[14]
            wh_acc_3.iterator[3] = _r[15]

        # ============================================================
        # Issue the SECOND state-tile half (K=64..127, overwrites sH). The
        # sync_threads ensures ALL warps finished reading half-0; then warp 0
        # issues the second TMA and the mbarrier parity flips to 1.
        # ============================================================
        sync_threads()
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_h_ptr,
                    V_DIM_C * K_HALF * 2,  # 16384 B (half-tile)
                )
            cute.copy(tma_atom_h, tHgH1, tHsH1, tma_bar_ptr=mbar_h_ptr)

        # ============================================================
        # HISTORY CONTRACTION — placed in the TMA half-1 shadow. Consumes the
        # u tile (viewed through sV) and the w-scaled transposed scores tile,
        # accumulating hw[r, v] += sum_j sWScores[r, j] * u[j, v] into the
        # wh_acc fragments via one _qtv_4mma. r = {r0 (k rows -> hw_k),
        # r0+8 (q rows -> hw_q)} matches the packed A-tile row map, so the H
        # GEMM and this MMA land in the same accumulator elements.
        # ============================================================
        _cp_async_wait_group_0()
        sync_threads()
        # MMA-based history contraction:
        #   hw[r, v] += sum_j sWScores[r, j] * u[j, v]
        # A = the w-scaled transposed scores tile (bf16, staged by the scores
        # GEMM), B = the u tile (in sV's region) — the _qtv_4mma pattern,
        # whose output fragments match wh_acc exactly. Skipped at P=0
        # (CTA-uniform; wh_acc keeps only the S0 term).
        if P_hist > 0:
            _hc_a0, _hc_a1, _hc_a2, _hc_a3 = _ldmatrix_x4(sWScores, lane_id)
            _hc_b_base = (
                _sV_base_async_u + _ldm_row * Int32(V_PADDED * 2) + warp_id * Int32(64)
            )
            _hcr = _qtv_4mma_rg(_hc_a0, _hc_a1, _hc_a2, _hc_a3, _hc_b_base)
            wh_acc_0.iterator[0] = wh_acc_0.iterator[0] + _hcr[0]
            wh_acc_0.iterator[1] = wh_acc_0.iterator[1] + _hcr[1]
            wh_acc_0.iterator[2] = wh_acc_0.iterator[2] + _hcr[2]
            wh_acc_0.iterator[3] = wh_acc_0.iterator[3] + _hcr[3]
            wh_acc_1.iterator[0] = wh_acc_1.iterator[0] + _hcr[4]
            wh_acc_1.iterator[1] = wh_acc_1.iterator[1] + _hcr[5]
            wh_acc_1.iterator[2] = wh_acc_1.iterator[2] + _hcr[6]
            wh_acc_1.iterator[3] = wh_acc_1.iterator[3] + _hcr[7]
            wh_acc_2.iterator[0] = wh_acc_2.iterator[0] + _hcr[8]
            wh_acc_2.iterator[1] = wh_acc_2.iterator[1] + _hcr[9]
            wh_acc_2.iterator[2] = wh_acc_2.iterator[2] + _hcr[10]
            wh_acc_2.iterator[3] = wh_acc_2.iterator[3] + _hcr[11]
            wh_acc_3.iterator[0] = wh_acc_3.iterator[0] + _hcr[12]
            wh_acc_3.iterator[1] = wh_acc_3.iterator[1] + _hcr[13]
            wh_acc_3.iterator[2] = wh_acc_3.iterator[2] + _hcr[14]
            wh_acc_3.iterator[3] = wh_acc_3.iterator[3] + _hcr[15]

        # Next tenant of qv_buf: the V tile. The u tile's last read was the
        # loop above; one barrier orders the handoff, then the V load runs
        # here (its wait stays at the tail, so the transfer overlaps the H
        # GEMM half-1).
        sync_threads()
        _gV_base = gV.iterator.toint()
        _v_base_bf16 = pid_b * sv_b + pid_hv * sv_hv
        _v_iters = 1 if self._t_input <= 8 else (T * V_DIM_C // (THREADS * 8))
        if const_expr(self._fused_conv):
            _v_iters = 0
            if tidx < Int32(V_DIM_C // 2):
                _cv_vdst = _sV_base_async_u + tidx * Int32(4)
                _sts_b32(_cv_vdst, _cv_yv0)
                _sts_b32(_cv_vdst + Int32(V_PADDED * 2), _cv_yv1)
                _sts_b32(_cv_vdst + Int32(2 * V_PADDED * 2), _cv_yv2)
                _sts_b32(_cv_vdst + Int32(3 * V_PADDED * 2), _cv_yv3)
        for i in cutlass.range_constexpr(_v_iters):
            _v_group = tidx + i * THREADS
            _v_row = _v_group // Int32(V_DIM_C // 8)
            _v_col_bf16_async = (_v_group % Int32(V_DIM_C // 8)) * Int32(8)
            _smem_byte_off_v = _v_row * Int32(V_PADDED * 2) + _v_col_bf16_async * Int32(
                2
            )
            # (native-short-T) v holds only n_valid rows; skip rows >=
            # n_valid (OOB). Tail zeroing happens after the V wait below.
            if const_expr(self._n_valid < T):
                if _v_row < Int32(self._n_valid):
                    _cp_async_bf16x8(
                        _gV_base,
                        _v_base_bf16 + _v_row * sv_t + _v_col_bf16_async,
                        _sV_base_async_u + _smem_byte_off_v,
                    )
            else:
                _cp_async_bf16x8(
                    _gV_base,
                    _v_base_bf16 + _v_row * sv_t + _v_col_bf16_async,
                    _sV_base_async_u + _smem_byte_off_v,
                )
        _cp_async_commit_group()

        # QT = sNegL @ sTmat, computed in the half-1 shadow on warps 0-1 into
        # sWScores (dead after the contraction MMA above; the sync above
        # orders the handoff, and the R-pass barrier below publishes QT before
        # the y GEMM reads it). Associativity takes U off the output critical
        # path: y_intra = sNegL @ (sTmat @ R) = QT @ R.
        if warp_id < 2:
            acc.fill(f32(0.0))
            _qt_col_off = warp_id * 8
            _qt_a_addr = (
                sNegL.iterator.toint() + _lane_mod16 * Int32(BF_PAD * 2) + _lane_hi
            )
            _qt_b_addr = (
                sTmat.iterator.toint()
                + _ldm_row * Int32(BF_PAD * 2)
                + _qt_col_off * Int32(2)
            )
            acc.iterator[0], acc.iterator[1], acc.iterator[2], acc.iterator[3] = (
                _fused_ab_1mma(
                    _qt_a_addr,
                    _qt_b_addr,
                    acc.iterator[0],
                    acc.iterator[1],
                    acc.iterator[2],
                    acc.iterator[3],
                )
            )
            _qt_r0 = lane_id // 4
            _qt_c0 = (lane_id & 3) * 2
            # (ring-fp16) QT is an IO-dtype MMA operand (y GEMM, paired with
            # R in sV): store its bytes RAW via the IO pack helper rather
            # than typed iterator assignment — the buffer's element type is
            # the RING dtype (tenant #1), and a typed store would silently
            # convert QT's bf16 values to fp16 bit patterns that the bf16 y
            # GEMM would then misread. c0 is even, so each pair store is
            # 4-B aligned. Identical codegen in the default mode.
            _qt_sw_base = sWScores.iterator.toint()
            _sts_bf16x2_f32(
                _qt_sw_base + (_qt_r0 * BF_PAD + _qt_col_off + _qt_c0) * 2,
                acc.iterator[0],
                acc.iterator[1],
            )
            _sts_bf16x2_f32(
                _qt_sw_base + ((_qt_r0 + 8) * BF_PAD + _qt_col_off + _qt_c0) * 2,
                acc.iterator[2],
                acc.iterator[3],
            )

        # Wait for second half to land before H GEMM half-1.
        cute.arch.mbarrier_wait(mbar_h_ptr, 1)
        cute.arch.fence_view_async_shared()
        sync_threads()

        # ============================================================
        # H GEMM HALF-1 (ka=4..7, uses sH for K=64..127). sH was overwritten
        # by the second TMA, so the col offset into sH RESETS; the sK col
        # offset advances (sK still has the full K_DIM=128 layout).
        # ============================================================
        for ka_local in cutlass.range_constexpr(4):
            col_byte_off_a = Int32((4 + ka_local) * 16 * 2)  # sK K=64..127
            col_byte_off_b = Int32(ka_local * 16 * 2)  # sH K=0..63 (reset!)
            _a_addr = _sK_base_vl + _lane_mod16 * _rs_a + _lane_hi + col_byte_off_a
            _b0_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(0) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b1_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(8) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b2_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(16) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b3_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(24) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b0 = _sw128_xor(_b0_l)
            _b1 = _sw128_xor(_b1_l)
            _b2 = _sw128_xor(_b2_l)
            _b3 = _sw128_xor(_b3_l)

            _r = _h_gemm_4v(
                _a_addr,
                _b0,
                _b1,
                _b2,
                _b3,
                wh_acc_0.iterator[0],
                wh_acc_0.iterator[1],
                wh_acc_0.iterator[2],
                wh_acc_0.iterator[3],
                wh_acc_1.iterator[0],
                wh_acc_1.iterator[1],
                wh_acc_1.iterator[2],
                wh_acc_1.iterator[3],
                wh_acc_2.iterator[0],
                wh_acc_2.iterator[1],
                wh_acc_2.iterator[2],
                wh_acc_2.iterator[3],
                wh_acc_3.iterator[0],
                wh_acc_3.iterator[1],
                wh_acc_3.iterator[2],
                wh_acc_3.iterator[3],
            )
            wh_acc_0.iterator[0] = _r[0]
            wh_acc_0.iterator[1] = _r[1]
            wh_acc_0.iterator[2] = _r[2]
            wh_acc_0.iterator[3] = _r[3]
            wh_acc_1.iterator[0] = _r[4]
            wh_acc_1.iterator[1] = _r[5]
            wh_acc_1.iterator[2] = _r[6]
            wh_acc_1.iterator[3] = _r[7]
            wh_acc_2.iterator[0] = _r[8]
            wh_acc_2.iterator[1] = _r[9]
            wh_acc_2.iterator[2] = _r[10]
            wh_acc_2.iterator[3] = _r[11]
            wh_acc_3.iterator[0] = _r[12]
            wh_acc_3.iterator[1] = _r[13]
            wh_acc_3.iterator[2] = _r[14]
            wh_acc_3.iterator[3] = _r[15]

        # ============================================================
        # (u-cache) TAIL: R → U (ring append) → y = e^G∘hw_q + F^T·U
        # ============================================================
        _sV_base = sV.iterator.toint()
        _gOut_base = gOut.iterator.toint()
        _out_base = pid_b * so_b + pid_hv * so_hv
        _v_off_base = Int32(0)  # full V in one tile
        # OutStage lives in k_buf (sK — dead after the H GEMM half-1 read it
        # as the A operand; exact 4352-B fit), NOT in sH as the verify kernel
        # stages it: keeping sH untouched here leaves the S0 half-1 tile
        # RESIDENT for the tail fold, which then re-TMAs only half-0 (halving
        # the fold's state re-read). Same [16, V_PADDED] layout — rows [0:8)
        # output, [8:8+t) U.
        _sOutStage_base = sK.iterator.toint()

        # Wait for the V cp.async (tenant #4, issued in the half-1 shadow;
        # the wait drains whatever didn't finish under the H GEMM half-1).
        _cp_async_wait_group_0()
        # Zero the sV working-set tail rows [n_valid:8]. Rows [8:16) stay
        # stale-but-finite — safe: the U GEMM's A (sTmat) has exact-zero
        # cols >= t, so those rows cannot propagate.
        if const_expr(self._n_valid < T):
            for _zr in cutlass.range_constexpr(self._n_valid, 8):
                sV.iterator[_zr * V_PADDED + tidx] = io(0.0)
        sync_threads()

        # R = V − e^{G_s} ∘ hw_k, in place over sV rows [0:t). hw_k lives
        # in wh_acc elements {0,1} (fragment rows r0 = packed k rows);
        # beta is folded into sTmat, NOT applied here.
        _y_r0 = lane_id // Int32(4)
        _y_c0 = (lane_id & Int32(3)) * Int32(2)
        if _y_r0 < Int32(self._t_input):
            _neg_eg = f32(0.0) - sGamma.iterator[T + _y_r0]
            # (perf) i32-paired RMW through the sQ i32 view (same buffer and
            # row stride as sV: K_PADDED == V_PADDED) — 4 load+store pairs
            # instead of 16 scalar bf16 chains; fma(-e^G, hw, v) is the
            # fused form of the previous v - e^G*hw.
            _r_i32 = (
                _y_r0 * Int32(V_PADDED // 2)
                + warp_id * Int32(16)
                + (lane_id & Int32(3))
            )
            _sQ_i32.iterator[_r_i32] = _r_sub_bf16x2(
                _sQ_i32.iterator[_r_i32],
                _neg_eg,
                wh_acc_0.iterator[0],
                wh_acc_0.iterator[1],
            )
            _sQ_i32.iterator[_r_i32 + 4] = _r_sub_bf16x2(
                _sQ_i32.iterator[_r_i32 + 4],
                _neg_eg,
                wh_acc_1.iterator[0],
                wh_acc_1.iterator[1],
            )
            _sQ_i32.iterator[_r_i32 + 8] = _r_sub_bf16x2(
                _sQ_i32.iterator[_r_i32 + 8],
                _neg_eg,
                wh_acc_2.iterator[0],
                wh_acc_2.iterator[1],
            )
            _sQ_i32.iterator[_r_i32 + 12] = _r_sub_bf16x2(
                _sQ_i32.iterator[_r_i32 + 12],
                _neg_eg,
                wh_acc_3.iterator[0],
                wh_acc_3.iterator[1],
            )
        sync_threads()

        # y FIRST, via associativity: y_intra = QT @ R with QT = sNegL @ sTmat
        # precomputed into sWScores in the half-1 shadow (B = R). U is computed
        # AFTER the output staging (off the critical path, overlapping the flush).
        _qt_a0, _qt_a1, _qt_a2, _qt_a3 = _ldmatrix_x4(sWScores, lane_id)
        _qtv_base = _sV_base + _ldm_row * Int32(V_PADDED * 2) + warp_id * Int32(64)
        _qtvr = _qtv_4mma(_qt_a0, _qt_a1, _qt_a2, _qt_a3, _qtv_base)
        # U = sTmat @ R back-to-back with the y MMA (same B = sV(R)); its rows
        # [0:t) are staged into OutStage rows [8:8+t) below, so the single
        # pre-flush barrier publishes output AND ring-append data.
        _u_a0, _u_a1, _u_a2, _u_a3 = _ldmatrix_x4(sTmat, lane_id)
        _ur = _qtv_4mma(_u_a0, _u_a1, _u_a2, _u_a3, _qtv_base)

        _eg_yq = sGamma.iterator[T + _y_r0]
        for h_iter in cutlass.range_constexpr(4):
            h = warp_id * 4 + h_iter
            acc.iterator[0] = _qtvr[h_iter * 4]
            acc.iterator[1] = _qtvr[h_iter * 4 + 1]
            acc.iterator[2] = _qtvr[h_iter * 4 + 2]
            acc.iterator[3] = _qtvr[h_iter * 4 + 3]
            # + e^{G_s} ∘ hw_q: hw_q lives in wh_acc elements
            # {2,3} (fragment rows r0+8 = packed q rows); it lands on the
            # y elements {0,1} (output rows r0 = window tokens). Fragment
            # rows r0+8 of y are garbage — neither staged (t<=8) nor
            # stored (STG row-gated), so elements {2,3} get no add.
            if h_iter == 0:
                acc.iterator[0] = acc.iterator[0] + _eg_yq * wh_acc_0.iterator[2]
                acc.iterator[1] = acc.iterator[1] + _eg_yq * wh_acc_0.iterator[3]
            if h_iter == 1:
                acc.iterator[0] = acc.iterator[0] + _eg_yq * wh_acc_1.iterator[2]
                acc.iterator[1] = acc.iterator[1] + _eg_yq * wh_acc_1.iterator[3]
            if h_iter == 2:
                acc.iterator[0] = acc.iterator[0] + _eg_yq * wh_acc_2.iterator[2]
                acc.iterator[1] = acc.iterator[1] + _eg_yq * wh_acc_2.iterator[3]
            if h_iter == 3:
                acc.iterator[0] = acc.iterator[0] + _eg_yq * wh_acc_3.iterator[2]
                acc.iterator[1] = acc.iterator[1] + _eg_yq * wh_acc_3.iterator[3]
            # SMEM-staged epilogue: stage the [T,128] tile in SMEM (h_buf — sH
            # is dead after the half-1 H GEMM; the sync at the QT@V wait above
            # orders all warps past it), then flush with fully-coalesced 16-B
            # STGs below. The STS pattern (word = 68*r + 4*h + lane%4) is
            # bank-conflict-free, avoiding the uncoalesced fragment-direct
            # 4-B stores.
            _out_r0 = lane_id // 4
            _out_c0 = (lane_id & 3) * 2
            _stg_col = h * 8 + _out_c0
            _sts_bf16x2_f32(
                _sOutStage_base + (_out_r0 * V_PADDED + _stg_col) * 2,
                acc.iterator[0],
                acc.iterator[1],
            )
            if const_expr(self._t_input > 8):
                _sts_bf16x2_f32(
                    _sOutStage_base + ((_out_r0 + 8) * V_PADDED + _stg_col) * 2,
                    acc.iterator[2],
                    acc.iterator[3],
                )

        # Stage U rows [0:t) at OutStage rows [8:8+t) (dead rows at t<=8):
        # frag elements {0,1} hold U row _y_r0. Published by the same
        # pre-flush barrier as the output rows.
        if _y_r0 < Int32(self._t_input):
            for _ug in cutlass.range_constexpr(4):
                _uu_col = (warp_id * Int32(4) + Int32(_ug)) * Int32(8) + _y_c0
                # (ring-fp16) U packs f32 -> RING dtype here — the single
                # rounding on the u path (the ring appends copy raw bytes).
                if const_expr(self._kq_fix):
                    _u_iv = sGhist.iterator[48 + _y_r0]
                    _sts_rg2_f32(
                        _sOutStage_base + ((8 + _y_r0) * V_PADDED + _uu_col) * 2,
                        _ur[_ug * 4] * _u_iv,
                        _ur[_ug * 4 + 1] * _u_iv,
                    )
                else:
                    _sts_rg2_f32(
                        _sOutStage_base + ((8 + _y_r0) * V_PADDED + _uu_col) * 2,
                        _ur[_ug * 4],
                        _ur[_ug * 4 + 1],
                    )

        # Coalesced flush: lanes write consecutive vector chunks; the shared
        # memory loads span a full bank period.
        sync_threads()
        if const_expr(self._fused_norm):
            # Gated RMSNorm in place over the staged bf16 output rows: warp w = token row w (t = 4).
            _nsm = _sOutStage_base + warp_id * Int32(V_PADDED * 2) + lane_id * Int32(2)
            _nx0 = _lds_u16(_nsm)
            _nx1 = _lds_u16(_nsm + Int32(64))
            _nx2 = _lds_u16(_nsm + Int32(128))
            _nx3 = _lds_u16(_nsm + Int32(192))
            _ngb = gGate.iterator.toint()
            _ngo = (pid_b * Int32(4) + warp_id) * gate_row + pid_hv * Int32(V_DIM_C) + lane_id
            _ng0 = _ldg_u16(_ngb, _ngo)
            _ng1 = _ldg_u16(_ngb, _ngo + Int32(32))
            _ng2 = _ldg_u16(_ngb, _ngo + Int32(64))
            _ng3 = _ldg_u16(_ngb, _ngo + Int32(96))
            _nwb = gNw.iterator.toint()
            _nw0 = _ldg_f32(_nwb, lane_id)
            _nw1 = _ldg_f32(_nwb, lane_id + Int32(32))
            _nw2 = _ldg_f32(_nwb, lane_id + Int32(64))
            _nw3 = _ldg_f32(_nwb, lane_id + Int32(96))
            _no0, _no1, _no2, _no3 = _gnorm4(
                _nx0, _nx1, _nx2, _nx3, _ng0, _ng1, _ng2, _ng3, _nw0, _nw1, _nw2, _nw3, eps, self._norm_sigmoid
            )
            _sts_u16(_nsm, _no0)
            _sts_u16(_nsm + Int32(64), _no1)
            _sts_u16(_nsm + Int32(128), _no2)
            _sts_u16(_nsm + Int32(192), _no3)
            if const_expr(self._fused_qo):
                # token row of this warp: pid_b * 4 + warp (fused path: every decode row has 4 tokens at cu = 4b)
                _qrow = pid_b * Int32(4) + warp_id
                _qb = gQ8.iterator.toint()
                _qoff = Int64(_qrow) * Int64(qo_stride) + Int64(pid_hv * Int32(V_DIM_C) + lane_id)
                _q0, _e0 = _qo_block(_no0)
                _q1, _e1 = _qo_block(_no1)
                _q2, _e2 = _qo_block(_no2)
                _q3, _e3 = _qo_block(_no3)
                _stg_u8(_qb, _qoff, _q0)
                _stg_u8(_qb, _qoff + Int64(32), _q1)
                _stg_u8(_qb, _qoff + Int64(64), _q2)
                _stg_u8(_qb, _qoff + Int64(96), _q3)
                if lane_id == Int32(0):
                    _stg_u32(gSF.iterator.toint(), Int64(_qo_sf_off(_qrow, pid_hv, qo_psc)),
                             _e0 | (_e1 << Int32(8)) | (_e2 << Int32(16)) | (_e3 << Int32(24)))
            sync_threads()
        for _fl_pass in cutlass.range_constexpr(2 if self._t_input > 8 else 1):
            _fl_chunk = _fl_pass * 128 + tidx
            _fl_row = _fl_chunk // 16
            _fl_pos = _fl_chunk & 15
            _fl_lds = _sOutStage_base + _fl_row * Int32(V_PADDED * 2) + _fl_pos * 16
            _fl_off = _out_base + _fl_row * so_t + _v_off_base + _fl_pos * 8
            # LDS hoisted out of the runtime guard: tuple-unpack inside an
            # if-region trips a DSL region-type error, and reading staged
            # garbage rows (>= t_input) is harmless — only the STG is gated.
            _v0, _v1, _v2, _v3 = _lds_v4_b32(_fl_lds)
            if const_expr(self._t_input >= 8):
                _st_global_v4_b32(_gOut_base, _fl_off, _v0, _v1, _v2, _v3)
            else:
                if _fl_row < Int32(self._t_input):
                    _st_global_v4_b32(_gOut_base, _fl_off, _v0, _v1, _v2, _v3)

        # Ring append straight from OutStage rows [8:8+t) — same barrier and
        # staging tile as the output, so this overlaps the output STGs above
        # (no extra pass, no extra barrier). Flush rows append here too: the
        # ring target (base+P+s)&mask is PAST the fold-source window
        # [base, base+P), so there is no write-after-read hazard.
        if tidx < Int32(self._t_input * (V_DIM_C // 8)):
            _uf_row = tidx // Int32(V_DIM_C // 8)
            _uf_pos = tidx % Int32(V_DIM_C // 8)
            _uv0, _uv1, _uv2, _uv3 = _lds_v4_b32(
                _sOutStage_base
                + (8 + _uf_row) * Int32(V_PADDED * 2)
                + _uf_pos * Int32(16)
            )
            _st_global_v4_b32(
                _gUC_base,
                _uc_base
                + ((ring_base + P_hist + _uf_row) & Int32(RING_MASK)) * V_DIM
                + _uf_pos * Int32(8),
                _uv0,
                _uv1,
                _uv2,
                _uv3,
            )

        # ============================================================
        # PER-REQUEST STATE FOLD + WRITE-BACK (ring semantics: NO restart
        # writes — this step's tokens were appended at (base+P+s)&mask in
        # the main pipeline, past the fold-source window [base, base+P);
        # cursor slide/reset is the CALLER's commit, outside the launch).
        # Runs only for CTAs whose request crossed flush_min. The predicate
        # is CTA-uniform (P_hist is per-request), so the sync_threads and
        # mbarrier waits inside this branch are safe. Ordering:
        #   1. reload the OLD u window rows into k_buf (dead after the H
        #      GEMM) and restore the khist snapshot into qv_buf (dead after
        #      the y/U GEMMs).
        #   2. b_d = w_j * u_j in place over the u stage (w = 0 for
        #      j >= P zeroes the k16 padding and any stale bytes).
        #   3. re-TMA S0 half-by-half into sH (mbarrier phases continue at
        #      literal parities 0, 1) and fold: D = b_d @ khist via
        #      _qtv_4mma strips, then S_h = bdec*S0 + D per element, STG
        #      straight to the state pool.
        # ============================================================
        if P_hist >= flush_min:
            # Order all earlier OutStage reads (output STGs + ring appends)
            # before the u reload re-tenants k_buf below.
            sync_threads()
            # --- 1) u reload (rows < P; .cg) into k_buf + khist snapshot
            #        restore into qv_buf ---
            for _fr in cutlass.range_constexpr(2):
                _fr_group = tidx + _fr * THREADS
                _fr_row = _fr_group // Int32(V_DIM_C // 8)
                _fr_col = (_fr_group % Int32(V_DIM_C // 8)) * Int32(8)
                if _fr_row < P_hist:
                    _cp_async_bf16x8_cg(
                        _gUC_base,
                        _uc_base
                        + ((ring_base + _fr_row) & Int32(RING_MASK)) * V_DIM
                        + _fr_col,
                        _sK_base_async
                        + _fr_row * Int32(K_PADDED * 2)
                        + _fr_col * Int32(2),
                    )
            _cp_async_commit_group()
            _kh_sts = (tidx // Int32(8)) * Int32(K_PADDED // 2) + (
                tidx % Int32(8)
            ) * Int32(8)
            # (ring-fp16) the snapshot holds RAW ring-dtype bytes (the tile
            # is never converted in SMEM), so plain stores are correct in
            # every mode — the fold MMA consumes the ring dtype directly.
            _sQ_i32.iterator[_kh_sts + 0] = _khs0
            _sQ_i32.iterator[_kh_sts + 1] = _khs1
            _sQ_i32.iterator[_kh_sts + 2] = _khs2
            _sQ_i32.iterator[_kh_sts + 3] = _khs3
            _sQ_i32.iterator[_kh_sts + 4] = _khs4
            _sQ_i32.iterator[_kh_sts + 5] = _khs5
            _sQ_i32.iterator[_kh_sts + 6] = _khs6
            _sQ_i32.iterator[_kh_sts + 7] = _khs7
            _cp_async_wait_group_0()
            sync_threads()
            # --- 2) b_d = w_j * u_j in place over the u stage (k_buf) ---
            for _ws in cutlass.range_constexpr(9):
                _ws_g = tidx + _ws * THREADS
                _ws_r = _ws_g // Int32(V_PADDED // 2)
                _ws_c = _ws_g % Int32(V_PADDED // 2)
                if _ws_r < Int32(W_RING):
                    _w_row = sGhist.iterator[W_RING + _ws_r]
                    _sK_i32.iterator[_ws_r * _kpad_i32 + _ws_c] = _mul_rg2_f32(
                        _sK_i32.iterator[_ws_r * _kpad_i32 + _ws_c], _w_row
                    )
            # --- 5) fold: HALF-1 FIRST from the RESIDENT sH (OutStage
            #        moved to sK, so S0 cols [64:128) survived the main
            #        pipeline), then ONE re-TMA for half-0 (halving the
            #        fold's state re-read). ALL FOUR warps fold each half
            #        (2 strips/warp x 2 column spans), halving the per-half
            #        MMA latency. Epilogue: S_h pairs are STS'd back to the
            #        SAME swizzled sH bytes (bit-identical values to a direct
            #        store), then flushed to the pool with fully-coalesced
            #        16-B chunks. ---
            _bdec_t = sGhist.iterator[32]
            # Pool/head strides from the layout (block-strided paged pools);
            # folded into the i64 byte base — pool offsets can exceed 2^31
            # elements, so the 32-bit path keeps only intra-page offsets.
            _gH0_st = (
                gH0.iterator.toint()
                + (
                    cache_idx64 * gH0.layout.stride[0]
                    + Int64(pid_hv) * gH0.layout.stride[1]
                )
                * 2
            )
            _h0_elem_base = Int32(0)
            _fa_row = (lane_id & Int32(7)) + ((lane_id >> Int32(4)) & Int32(1)) * Int32(
                8
            )
            _fa_colb = ((lane_id >> Int32(3)) & Int32(1)) * Int32(16)
            _fc_r0 = lane_id // Int32(4)
            _fc_c0 = (lane_id & Int32(3)) * Int32(2)
            for _fh in (1, 0):
                # Barrier BEFORE anything in the iteration: for half-1 it
                # publishes the w-scaled u stage to the fold's ldmatrix
                # readers; for half-0 it is LOAD-BEARING against the TMA —
                # every warp must have finished its half-1 coalesced-flush
                # LDS reads of sH before warp 0's re-TMA overwrites the
                # tile; otherwise per-request state corruption is possible.
                sync_threads()
                if _fh == 0:
                    # sH half-1 fully consumed: re-TMA half-0. Main
                    # pipeline used mbar phases 0 and 1; this third
                    # completion is parity 0 again (literal).
                    if warp_id == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar_h_ptr,
                                V_DIM_C * K_HALF * 2,
                            )
                        cute.copy(tma_atom_h, tHgH0, tHsH0, tma_bar_ptr=mbar_h_ptr)
                    cute.arch.mbarrier_wait(mbar_h_ptr, 0)
                    cute.arch.fence_view_async_shared()
                    sync_threads()
                for _fs2 in cutlass.range_constexpr(2):
                    _fs = warp_id * Int32(2) + Int32(_fs2)
                    for _fspan in cutlass.range_constexpr(2):
                        _fa0, _fa1, _fa2, _fa3 = _ldmatrix_x4_trans(
                            _sK_base_async
                            + _fa_row * Int32(K_PADDED * 2)
                            + _fs * Int32(32)
                            + _fa_colb
                        )
                        _frr = _qtv_4mma_rg(
                            _fa0,
                            _fa1,
                            _fa2,
                            _fa3,
                            _sQ_base_async
                            + _ldm_row * Int32(K_PADDED * 2)
                            + Int32(_fh * 128 + _fspan * 64),
                        )
                        for _ft in cutlass.range_constexpr(4):
                            _f_col = Int32(_fspan * 32 + _ft * 8) + _fc_c0
                            _f_row0 = _fs * Int32(16) + _fc_r0
                            _sh_a0 = _sw128_xor(
                                _sH_base_vl
                                + _f_row0 * Int32(K_HALF * 2)
                                + _f_col * Int32(2)
                            )
                            _s0p = _lds_b32(_sh_a0)
                            _sf0, _sf1 = _fold_fma_bf16x2(
                                _s0p, _bdec_t, _frr[_ft * 4], _frr[_ft * 4 + 1]
                            )
                            _sts_st2_f32(_sh_a0, _sf0, _sf1)
                            _sh_a1 = _sw128_xor(
                                _sH_base_vl
                                + (_f_row0 + Int32(8)) * Int32(K_HALF * 2)
                                + _f_col * Int32(2)
                            )
                            _s1p = _lds_b32(_sh_a1)
                            _sf2, _sf3 = _fold_fma_bf16x2(
                                _s1p, _bdec_t, _frr[_ft * 4 + 2], _frr[_ft * 4 + 3]
                            )
                            _sts_st2_f32(_sh_a1, _sf2, _sf3)
                sync_threads()
                # coalesced flush of the updated half: consecutive lanes
                # write consecutive 16-B chunks (8 chunks per 128-B SMEM
                # row = one swizzle period; the XOR is applied per chunk).
                for _fc in cutlass.range_constexpr(8):
                    _f_chunk = tidx + _fc * THREADS
                    _f_row = _f_chunk >> 3
                    _f_pos = _f_chunk & Int32(7)
                    _cv0, _cv1, _cv2, _cv3 = _lds_v4_b32(
                        _sw128_xor(
                            _sH_base_vl
                            + _f_row * Int32(K_HALF * 2)
                            + _f_pos * Int32(16)
                        )
                    )
                    _st_global_v4_b32(
                        _gH0_st,
                        _h0_elem_base
                        + _f_row * K_DIM
                        + Int32(_fh * K_HALF)
                        + _f_pos * Int32(8),
                        _cv0,
                        _cv1,
                        _cv2,
                        _cv3,
                    )
            # (ring) no tail k restart: this step's normed k was appended at
            # (base+P+s)&mask by the k-group leader in the main pipeline —
            # past every sibling's fold-source window, so no inter-CTA
            # ordering is required.


# ============================================================================
# Public entry point — gated_delta_rule_mtp_ucache_flush: the draft-token
# decode output PLUS the (k_cache, u_cache, g_cache, hist_len) history-ring
# append AND per-request state flush (see the module docstring). Native
# T in {4, 8} only.
#
# Consumes the state pool in its natural (pool, HV, V, K) layout (no external
# prepack) and computes the decode output. Requests below flush_min take the
# verify path (state read-only); requests at or above flush_min additionally
# fold the ring into the state and write it back. State-update flags not
# covered by this contract raise NotImplementedError so a mis-routed caller
# fails loudly rather than silently returning wrong results.
#
# Input/output contract (IO = bf16 or fp16; STATE = IO or fp16, see module doc):
#   A_log [HV], a [B,T,HV], dt_bias [HV], q/k [B,T,H,K], v [B,T,HV,V], b [B,T,HV],
#   initial_state_source [pool,HV,V,K] STATE, initial_state_indices [B] int32,
#   scale, output [B,T,HV,V] -> returns output [B,T,HV,V] IO.
# Requires SM90+ (TMA + mbarrier); K == V == 128.
# ============================================================================

_CACHE: dict = {}
# Persistent pre-zeroed T=16 input staging buffers for the T<16 path, keyed by
# (device, B, H, HK, HV, K, V, dtype, T). Reused across calls so short-T decode
# pays only a T-row copy-in (no per-call F.pad realloc/re-zero).
_STAGE: dict = {}
# When False, the T<16 path assumes the staging buffers already hold the current
# inputs and skips the per-call copy-in. Set this only when the producer writes
# q/k/v/a/b directly into the persistent T=16 buffers (the fixed-buffer serving
# pattern) or to benchmark the bare kernel. Default True = always safe drop-in.
_RESTAGE = True
# When set, the T<T_KERNEL path passes q/k to the kernel as the real [B,T,...]
# tensors (no host staging copy) and the kernel loads only those T rows + zeros
# its sK/sQ smem tail, removing the two big q/k gmem->gmem staging copies. v/a/b
# stay staged. Set SGLANG_GDN_WY_NATIVE_T=0 to restore full staging.
import os as _os

_NATIVE_T = _os.environ.get("SGLANG_GDN_WY_NATIVE_T", "1") != "0"
# (strided-qkv) read q/k/v directly from the fused conv-output column slices (token
# stride = conv_dim) instead of .contiguous()-materializing them. Removes the 3 big
# q/k/v copies from the verify region. Only valid on the native path (T in {4,8}).
_STRIDED_QKV = _os.environ.get("SGLANG_GDN_WY_STRIDED_QKV", "0") != "0"
# (native-a/b) read a/b directly from the real [B, n_valid, HV] tensors instead of staging
# them into T_KERNEL-row zero-padded buffers (removes the 2 a/b staging copies). Bit-exact on
# the compact [B,T] output: gamma is a causal prefix-sum, so the unloaded tail rows (which get
# log_alpha=0 instead of the staged-zero value) cannot affect rows 0..n_valid-1, and the tail
# output is discarded. Native path only (T in {4,8}).
# ON by default alongside _NATIVE_T (same validation); SGLANG_GDN_WY_NATIVE_AB=0
# restores a/b staging.
_NATIVE_AB = _os.environ.get("SGLANG_GDN_WY_NATIVE_AB", "1") != "0"
# Cache the bf16 cast of the per-layer CONSTANT weights A_log/dt_bias, keyed by
# storage identity (data_ptr, shape). They are persistent tensors passed every verify
# call; caching turns the per-call `.to(bf16)` into a one-time (warm-up) cast that does
# not appear in the captured CUDA graph. Safe for inference (weights never change).
_BF16_CACHE: dict = {}


def _cached_bf16(t):
    """Cast-cache to the module IO dtype (bf16 default, fp16 when
    GDN_UCACHE_IO_DTYPE=fp16); name keeps the historical bf16 suffix."""
    if t.dtype == IO_TORCH and t.is_contiguous():
        return t
    # Key by the SOURCE TENSOR OBJECT's identity, evicted when the object dies —
    # NOT by data_ptr: the caching allocator recycles freed storage, so a
    # data_ptr key can return a STALE cast for a brand-new tensor that landed on
    # a recycled allocation (silent wrong A_log/dt_bias whenever the caller
    # recreates them, e.g. benches/tests). id() is safe here because the
    # weakref.finalize pop runs during the referent's destruction, before CPython
    # can reuse the id. Serving keeps the fast path: per-layer weights are
    # persistent objects, so hits return the same bf16 tensor (stable address —
    # required for CUDA-graph replay). In-place mutation of a cached source
    # tensor is not detected (same limitation as the previous key; these are
    # frozen inference weights).
    key = id(t)
    c = _BF16_CACHE.get(key)
    if c is None:
        c = t.to(IO_TORCH).contiguous()
        _BF16_CACHE[key] = c
        weakref.finalize(t, _BF16_CACHE.pop, key, None)
    return c


# The FlashInfer wrapper (gated_delta_rule_mtp_ucache_flush) is not vendored: vLLM launches the kernel
# through adapter.ucache_decode (separate ring index, tvm-ffi compiled launch, caller-owned cursors).


__all__ = [
    "GdnDecodeUCacheFlushKernel",
    "K_DIM",
    "V_DIM_C",
    "W_RING",
    "RING_SLOTS",
    "RING_MASK",
]
