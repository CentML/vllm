# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# mypy: ignore-errors
# fmt: off
# Kernel Factory solution, verbatim below this header (do not edit; swap the whole file).
#   campaign  8vatkk15p51kb4gxsgkgmwm7km round 5, definition qwen36_moe_mxfp8_lowm_e256k8_h2048i512
#   candidate 23e8e69e4ecc43ae62d43908fe1ec9ad0efc055913acc7a29d1fcf39e5c61af3 (moe_v4_tma_sched, author 8vatkk15p51kb4gxsgkgmwm7km-r005-a001)
#   gate      mwv-s64+stress+tie-v2 PASS on 0103: Sigma graph 0.785, env 0.754, no slow T, tie PASS (E/runs/kfmoeA/winner-validate/gate-0103-f-23e8e69e.md)
#   source sha256 41dcbab6c33be9425974eab1928ab2ae81514b195517dc3e710d9670969bfe49 (bytes after this header)
# Retrieve: kf --format json campaign kernel show 23e8e69e4ecc43ae62d43908fe1ec9ad0efc055913acc7a29d1fcf39e5c61af3  (.source -> sources[kernel.py].content)
# Runtime: vllm/model_executor/layers/fused_moe/kf_lowm_moe.py uses setup()/run()/compile_for() from here.
# ---- verbatim KF source below ----
"""Single-launch routed MoE block (Qwen3.6-35B-A3B, MXFP8, decode) for Rubin, written in CuTe DSL.

One persistent CTA per SM:
  1. router GEMM (bf16, CUDA cores; each CTA owns 1-2 experts' logits)  -> grid barrier
  2. every CTA recomputes softmax/top-8/renorm + the expert/token tables redundantly (no metadata kernel)
  3. a producer warp streams weight tiles (cp.async.bulk -> smem ring); 8 consumer warps use native
     FP8 mma.sync m16n8k32 with per-32-block UE8M0 scales applied to fp32 accumulators.
     FC1: 16-row tiles (one gate/up-paired m16 tile, full K) distributed round-robin over all CTAs;
          SwiGLU in the epilogue, fp32 act -> global, and one ready-counter increment per FC1 tile.
          FC2 loader warps perform vectorized OCP requantization after all expert tiles are ready.
          For concentrated routes, two independent three-warp groups share each expert's token rows.
     FC2: CTA c < 128 owns 16 output rows of H for ALL experts (so the top-8 combine stays in-CTA,
          no partial rows / finalize); it waits on the per-expert ready counters.
  4. loader/epilogue warps self-reset counters while MMA warps finish deterministic bf16 output stores.
"""
import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32
from cutlass.cutlass_dsl import T as MT
from cutlass._mlir.dialects import llvm
import cuda.bindings.driver as cuda
from cutlass.experimental.cuda import TensorMap, TensorMapSwizzle, create_tensor_map_tiled

NCW = 8                    # MMA consumer warps 0..7 (warp w owns k-chunks 2w, 2w+1 of an FC1 tile)
WP = 8                     # weight producer warp
WL = 9                     # FC2 operand loader/requant warps WL..WL+NL-1
NL = 6
WE = WL + NL               # FC1 epilogue warps WE..WE+NE-1
NE = 4
NTH = (WE + NE) * 32
READY_TILES = 64           # FC1 m-tiles per expert
SLOT = 37888               # ring slot bytes (FC1: 16 rows x 2064 B padded + 4 KB scales; FC2: 4 x 8 KB swizzled + 4 KB)
XST = 2064                 # padded smem row stride for 2048-byte rows (conflict-free ldmatrix)
XHST = 2064                # native E4M3 token rows with 16 B padding
W2ST = 528                 # padded smem row stride for 512-byte rows
FC1_SF_OFF = 16 * XST
FC2_SF_OFF = 32768
FC1_BYTES = 16 * 2048 + 16 * 256
FC2_EXP_BYTES = 16 * 512 + 4 * 256
W13_EXP = 1024 * 2048
W13SF_EXP = 1024 * 64
W2_EXP = 2048 * 512
W2SF_EXP = 2048 * 16
CNT_BAR, CNT_DONE, CNT_READY, CNT_BLK = 0, 1, 32, 512
CNT_WORDS = 512 + 256 * 16
CNT_DBG = 8192
CNT_ALLOC = CNT_DBG + 512 * 512
DBG = False
PROF = False           # timeline instrumentation (printf from the last CTA); off for timing
NS_DEFAULT = 6
SMEM_SOFT = 232448     # ring depth is chosen to fit here: deeper rings measured slower (DRAM locality)
SMEM_HARD = 333824     # VR100 launches up to 334848 B of dynamic smem (verified); used only if needed
SMEM_PAD = 0
POLL_NS = 128          # back-off between polls of the per-expert ready counters
PF_ITEMS = 0           # L2 prefetch distance beyond the smem ring, in ring-depth units
NS_TARGET = (4, 4, 4, 4)
def npb_for(T):
    return 2 if T <= 18 else 1   # FC1 partial-sum buffers (decoupled from the weight ring)


SC_OWN1, SC_FC2, SC_NON1, SC_TAIL = 24, 19, 20, 70   # schedule cost weights (see the producer warp)
FC1_TMA = True         # FC1 weight tile = one 128B-swizzled TMA box (1 op instead of 16 row copies)
EARLY_ISSUE = True     # first ring stages issued by the producer right after top-8 selection
SCHED_SLACK = 2        # FC1 tiles the non-owner CTAs should finish ahead of the owners (FC1->FC2 hand-off)
XSTAGE_T = 8           # T > XSTAGE_T: router input staged in smem (avoids register spills)
SPIN_CAP = 1 << 21   # fast polling phase; every wait has an unconditional blocking fallback
LOG2E = 1.4426950408889634
SORT8 = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (1, 2), (5, 6),
         (0, 4), (3, 7), (1, 5), (2, 6), (1, 4), (3, 6), (2, 4), (3, 5), (3, 4))


# ----------------------------------------------------------------------------------------------
# inline PTX helpers (straight-line; called while tracing the kernel)
# ----------------------------------------------------------------------------------------------
def _ir(v):
    return v.ir_value()


def _asm(outs, ins, code, cons, se=True):
    vals = [_ir(v) for v in ins]
    if not outs:
        llvm.inline_asm(None, vals, code, cons, has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT)
        return None
    if len(outs) == 1:
        r = llvm.inline_asm(outs[0].mlir_type, vals, code, cons, has_side_effects=se,
                            is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
        return outs[0](r)
    st = llvm.StructType.get_literal([o.mlir_type for o in outs])
    r = llvm.inline_asm(st, vals, code, cons, has_side_effects=se, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT)
    return [o(llvm.extractvalue(o.mlir_type, r, [i])) for i, o in enumerate(outs)]


def i2f(x):
    return _asm([Float32], [x], "mov.b32 $0, $1;", "=f,r", se=False)


def f2i(x):
    return _asm([Int32], [x], "mov.b32 $0, $1;", "=r,f", se=False)


def imin(a, b):
    return _asm([Int32], [a, b], "min.s32 $0, $1, $2;", "=r,r,r", se=False)


def imax(a, b):
    return _asm([Int32], [a, b], "max.s32 $0, $1, $2;", "=r,r,r", se=False)


def umin(a, b):
    return _asm([Int32], [a, b], "min.u32 $0, $1, $2;", "=r,r,r", se=False)


def umax(a, b):
    return _asm([Int32], [a, b], "max.u32 $0, $1, $2;", "=r,r,r", se=False)


def popc(a):
    return _asm([Int32], [a], "popc.b32 $0, $1;", "=r,r", se=False)


def isel(c, a, b):
    """c (Int32) != 0 ? a : b"""
    return _asm([Int32], [c, a, b], "{ .reg .pred p; setp.ne.s32 p, $1, 0; selp.b32 $0, $2, $3, p; }",
                "=r,r,r,r", se=False)


def fsel(c, a, b):
    return _asm([Float32], [c, a, b], "{ .reg .pred p; setp.ne.s32 p, $1, 0; selp.f32 $0, $2, $3, p; }",
                "=f,r,f,f", se=False)


def eq_i(a, b):
    """(a == b) as Int32 0/1"""
    return _asm([Int32], [a, b], "{ .reg .pred p; setp.eq.s32 p, $1, $2; selp.b32 $0, 1, 0, p; }",
                "=r,r,r", se=False)


def lt_i(a, b):
    """(a < b) as Int32 0/1 (signed)"""
    return _asm([Int32], [a, b], "{ .reg .pred p; setp.lt.s32 p, $1, $2; selp.b32 $0, 1, 0, p; }",
                "=r,r,r", se=False)


def sf_f32(word, j):
    """UE8M0 byte j of word -> fp32 2^(b-127) (b >= 1)."""
    sel = 0x444 | (j << 12)
    return _asm([Float32], [word],
                "{ .reg .b32 z, t; mov.b32 z, 0; prmt.b32 t, $1, z, %d; shr.b32 t, t, 1; mov.b32 $0, t; }" % sel,
                "=f,r", se=False)


def ex2(x):
    return _asm([Float32], [x], "ex2.approx.ftz.f32 $0, $1;", "=f,f", se=False)


def frcp(x):
    return _asm([Float32], [x], "rcp.rn.f32 $0, $1;", "=f,f", se=False)


def fdiv(a, b):
    return _asm([Float32], [a, b], "div.rn.f32 $0, $1, $2;", "=f,f,f", se=False)


def ffma(a, b, c):
    return _asm([Float32], [a, b, c], "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", se=False)


def bf16_round(x):
    return _asm([Float32], [x], "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }", "=f,f", se=False)


def e4m3_byte(x):
    return _asm([Int32], [x, Float32(0.0)],
                "{ .reg .b16 h; cvt.rn.satfinite.e4m3x2.f32 h, $2, $1; cvt.u32.u16 $0, h; }", "=r,f,f", se=False)


# --- shared memory
def lds(a):
    return _asm([Int32], [a], "ld.shared.b32 $0, [$1];", "=r,r")


def ldsf(a):
    return _asm([Float32], [a], "ld.shared.f32 $0, [$1];", "=f,r")


def lds4i(a):
    return _asm([Int32] * 4, [a], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r")


def lds4f(a):
    return _asm([Float32] * 4, [a], "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r")


def sts_u8(a, v):
    _asm([], [a, v], "st.shared.u8 [$0], $1;", "r,r")


def sts(a, v):
    _asm([], [a, v], "st.shared.b32 [$0], $1;", "r,r")


def stsf(a, v):
    _asm([], [a, v], "st.shared.f32 [$0], $1;", "r,f")


def sts4f(a, v0, v1, v2, v3):
    _asm([], [a, v0, v1, v2, v3], "st.shared.v4.f32 [$0], {$1, $2, $3, $4};", "r,f,f,f,f")


def stsf_if(c, a, v):
    _asm([], [c, a, v], "{ .reg .pred p; setp.ne.s32 p, $0, 0; @p st.shared.f32 [$1], $2; }", "r,r,f")


def stg_if(c, p, v):
    _asm([], [c, p, v], "{ .reg .pred p; setp.ne.s32 p, $0, 0; @p st.global.b32 [$1], $2; }", "r,l,r")


def red_or_s(a, v):
    _asm([], [a, v], "red.shared.or.b32 [$0], $1;", "r,r")


def atom_add_s(a, v):
    return _asm([Int32], [a, v], "atom.shared.add.u32 $0, [$1], $2;", "=r,r,r")


def ldsm4(a):
    return _asm([Int32] * 4, [a], "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
                "=r,=r,=r,=r,r")


# --- global memory
def ldg4_nc(p):
    return _asm([Int32] * 4, [p], "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l")


def ldg4_cg(p):
    return _asm([Int32] * 4, [p], "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l")


def ldg2_cg(p):
    return _asm([Int32] * 2, [p], "ld.global.cg.v2.b32 {$0, $1}, [$2];", "=r,=r,l")


def ldgf_cg(p):
    return _asm([Float32], [p], "ld.global.cg.f32 $0, [$1];", "=f,l")


def stgf(p, v):
    _asm([], [p, v], "st.global.f32 [$0], $1;", "l,f")


def stg_u8(p, v):
    _asm([], [p, v], "st.global.u8 [$0], $1;", "l,r")


def stg_bf16(p, v):
    _asm([], [p, v], "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.global.b16 [$0], h; }", "l,f")


def ldg4f_cg(p):
    return _asm([Float32] * 4, [p], "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,l")


def fmax_abs(a, b):
    return _asm([Float32], [a, b], "{ .reg .f32 x, y; abs.f32 x, $1; abs.f32 y, $2; max.f32 $0, x, y; }", "=f,f,f", se=False)


def fmax(a, b):
    return _asm([Float32], [a, b], "max.f32 $0, $1, $2;", "=f,f,f", se=False)


def e4m3x4(a, b, c, d, s):
    """4 floats * s -> 4 packed e4m3 bytes (a in byte 0), RNE, saturating"""
    return _asm([Int32], [a, b, c, d, s],
                "{ .reg .f32 x0, x1, x2, x3; .reg .b16 l, h;\n"
                "mul.rn.f32 x0, $1, $5; mul.rn.f32 x1, $2, $5; mul.rn.f32 x2, $3, $5; mul.rn.f32 x3, $4, $5;\n"
                "cvt.rn.satfinite.e4m3x2.f32 l, x1, x0; cvt.rn.satfinite.e4m3x2.f32 h, x3, x2;\n"
                "mov.b32 $0, {l, h}; }", "=r,f,f,f,f,f", se=False)


def shfl_xor_f(v, m):
    return _asm([Float32], [v], "shfl.sync.bfly.b32 $0, $1, %d, 0x1f, 0xffffffff;" % m, "=f,f")


def st_rlx_f(p, v):
    _asm([], [p, v], "st.relaxed.gpu.global.f32 [$0], $1;", "l,f")


def st_rlx4(p, v):
    _asm([], [p, v, v, v, v], "st.relaxed.gpu.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r")


def stg_rlx(p, v):
    _asm([], [p, v], "st.relaxed.gpu.global.b32 [$0], $1;", "l,r")


def ld_acq(p):
    return _asm([Int32], [p], "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l")


def red_add_rlx(p, v):
    _asm([], [p, v], "red.relaxed.gpu.global.add.u32 [$0], $1;", "l,r")


def ld_rlx(p):
    return _asm([Int32], [p], "ld.relaxed.gpu.global.b32 $0, [$1];", "=r,l")


def red_add_rel(p, v):
    _asm([], [p, v], "red.release.gpu.global.add.u32 [$0], $1;", "l,r")


def atom_add_ar(p, v):
    return _asm([Int32], [p, v], "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;", "=r,l,r")


def clk64():
    if not DBG:
        return Int64(0)
    return _asm([Int64], [], "mov.u64 $0, %clock64;", "=l")


def gtimer():
    if not PROF:
        return Int64(0)
    return _asm([Int64], [], "mov.u64 $0, %globaltimer;", "=l")


def dbg_put(p_cnt, cta, k, v):
    if not PROF:
        return
    _asm([], [p_cnt + Int64(4 * CNT_DBG) + Int64(cta) * 2048 + Int64(8) * Int64(k), v], "st.global.u64 [$0], $1;", "l,l")


def ldg_u64(p):
    return _asm([Int64], [p], "ld.relaxed.gpu.global.u64 $0, [$1];", "=l,l")


def smin64(a, b):
    return _asm([Int64], [a, b], "min.s64 $0, $1, $2;", "=l,l,l", se=False)


def smax64(a, b):
    return _asm([Int64], [a, b], "max.s64 $0, $1, $2;", "=l,l,l", se=False)


def l2_evict_first_policy():
    return _asm([Int64], [], "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", "=l")


def tma3h(dst, tmap, c0, c1, c2, mbar, pol):
    _asm([], [dst, tmap, c0, c1, c2, mbar, pol],
         "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint "
         "[$0], [$1, {$2, $3, $4}], [$5], $6;", "r,l,r,r,r,r,l")


def tma4h(dst, tmap, c0, c1, c2, c3, mbar, pol):
    _asm([], [dst, tmap, c0, c1, c2, c3, mbar, pol],
         "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint "
         "[$0], [$1, {$2, $3, $4, $5}], [$6], $7;", "r,l,r,r,r,r,r,l")


def tma3(dst, tmap, c0, c1, c2, mbar):
    _asm([], [dst, tmap, c0, c1, c2, mbar],
         "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4}], [$5];",
         "r,l,r,r,r,r")


def tma4(dst, tmap, c0, c1, c2, c3, mbar):
    _asm([], [dst, tmap, c0, c1, c2, c3, mbar],
         "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4, $5}], [$6];",
         "r,l,r,r,r,r,r")


def nanosleep(ns):
    _asm([], [], "nanosleep.u32 %d;" % ns, "")


def l2_prefetch(p, nbytes):
    _asm([], [p, nbytes], "cp.async.bulk.prefetch.L2.global [$0], $1;", "l,r")


def prefetch_tmap(tmap):
    _asm([], [tmap], "prefetch.tensormap [$0];", "l")


def swz_addr(base, row, kb, half):
    """smem address of (row, k-bytes kb*32+16*half..+15) inside a [kchunk][16 rows][128B] 128B-swizzled tile"""
    c16 = ((kb & 3) << 1) | half
    return base + (kb >> 2) * 2048 + row * 128 + ((c16 ^ (row & 7)) << 4)


def fence_gpu():
    _asm([], [], "fence.acq_rel.gpu;", "")


def fence_cta():
    _asm([], [], "fence.acq_rel.cta;", "")


def syncwarp():
    _asm([], [], "bar.warp.sync 0xffffffff;", "")


def bar_sync(bid, n):
    _asm([], [], "bar.sync %d, %d;" % (bid, n), "")


# --- warp collectives
def shfl_bfly_f(v, m):
    return _asm([Float32], [v], "shfl.sync.bfly.b32 $0, $1, %d, 0x1f, 0xffffffff;" % m, "=f,f")


def shfl_up_i(v, d):
    return _asm([Int32], [v], "shfl.sync.up.b32 $0, $1, %d, 0x0, 0xffffffff;" % d, "=r,r")


def shfl_idx_i(v, src):
    return _asm([Int32], [v], "shfl.sync.idx.b32 $0, $1, %d, 0x1f, 0xffffffff;" % src, "=r,r")


def shfl_idx_f(v, src):
    return _asm([Float32], [v], "shfl.sync.idx.b32 $0, $1, %d, 0x1f, 0xffffffff;" % src, "=f,f")


def redux_min_u(v):
    return _asm([Int32], [v], "redux.sync.min.u32 $0, $1, 0xffffffff;", "=r,r")


def redux_max_u(v):
    return _asm([Int32], [v], "redux.sync.max.u32 $0, $1, 0xffffffff;", "=r,r")


# --- mbarrier / bulk copy
def mbar_init(a, cnt):
    _asm([], [a, Int32(cnt)], "mbarrier.init.shared::cta.b64 [$0], $1;", "r,r")


def mbar_init_fence():
    _asm([], [], "fence.mbarrier_init.release.cluster;", "")


def mbar_expect(a, nbytes):
    _asm([], [a, nbytes], "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;", "r,r")


def mbar_arrive(a):
    _asm([], [a], "mbarrier.arrive.shared::cta.b64 _, [$0];", "r")


def mbar_arrive_n(a, n):
    _asm([], [a], "mbarrier.arrive.shared::cta.b64 _, [$0], %d;" % n, "r")


def mbar_try(a, par):
    return _asm([Int32], [a, par],
                "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }",
                "=r,r,r")


def bulk_g2s_h(dst, src, nbytes, mbar, pol):
    _asm([], [dst, src, nbytes, mbar, pol],
         "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], $2, [$3], $4;",
         "r,l,r,r,l")


def bulk_g2s(dst, src, nbytes, mbar):
    _asm([], [dst, src, nbytes, mbar],
         "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
         "r,l,r,r")


# --- tensor core
def mma_e4m3(a, b0, b1, z):
    return _asm([Float32] * 4, [a[0], a[1], a[2], a[3], b0, b1, z],
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, "
                "{$8, $9}, {$10, $10, $10, $10};",
                "=f,=f,=f,=f,r,r,r,r,r,r,f", se=False)


def mma_k32(a, b0, b1):
    """D = A(16x32 e4m3) * B(32x8 e4m3), fp32 out, as two chained f16 m16n8k16 (e4m3 -> f16 is exact)."""
    return _asm([Float32] * 4, [a[0], a[1], a[2], a[3], b0, b1],
                "{ .reg .b16 l0, h0, l1, h1, l2, h2, l3, h3, lb0, hb0, lb1, hb1;\n"
                ".reg .b32 A0, A1, A2, A3, A4, A5, A6, A7, B0, B1, B2, B3;\n"
                ".reg .f32 z;\n"
                "mov.b32 {l0, h0}, $4; mov.b32 {l1, h1}, $5; mov.b32 {l2, h2}, $6; mov.b32 {l3, h3}, $7;\n"
                "mov.b32 {lb0, hb0}, $8; mov.b32 {lb1, hb1}, $9;\n"
                "cvt.rn.f16x2.e4m3x2 A0, l0; cvt.rn.f16x2.e4m3x2 A1, l1; cvt.rn.f16x2.e4m3x2 A2, h0; cvt.rn.f16x2.e4m3x2 A3, h1;\n"
                "cvt.rn.f16x2.e4m3x2 A4, l2; cvt.rn.f16x2.e4m3x2 A5, l3; cvt.rn.f16x2.e4m3x2 A6, h2; cvt.rn.f16x2.e4m3x2 A7, h3;\n"
                "cvt.rn.f16x2.e4m3x2 B0, lb0; cvt.rn.f16x2.e4m3x2 B1, hb0; cvt.rn.f16x2.e4m3x2 B2, lb1; cvt.rn.f16x2.e4m3x2 B3, hb1;\n"
                "mov.f32 z, 0f00000000;\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {A0, A1, A2, A3}, {B0, B1}, {z, z, z, z};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {A4, A5, A6, A7}, {B2, B3}, {$0, $1, $2, $3};\n"
                "}",
                "=f,=f,=f,=f,r,r,r,r,r,r", se=False)


def mma_k32_h(a, b):
    """D = A(16x32 e4m3, converted in registers) * B(32x8, already k-permuted f16 in smem): two chained HMMA."""
    return _asm([Float32] * 4, [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]],
                "{ .reg .b16 l0, h0, l1, h1, l2, h2, l3, h3;\n"
                ".reg .b32 A0, A1, A2, A3, A4, A5, A6, A7;\n"
                ".reg .f32 z;\n"
                "mov.b32 {l0, h0}, $4; mov.b32 {l1, h1}, $5; mov.b32 {l2, h2}, $6; mov.b32 {l3, h3}, $7;\n"
                "cvt.rn.f16x2.e4m3x2 A0, l0; cvt.rn.f16x2.e4m3x2 A1, l1; cvt.rn.f16x2.e4m3x2 A2, h0; cvt.rn.f16x2.e4m3x2 A3, h1;\n"
                "cvt.rn.f16x2.e4m3x2 A4, l2; cvt.rn.f16x2.e4m3x2 A5, l3; cvt.rn.f16x2.e4m3x2 A6, h2; cvt.rn.f16x2.e4m3x2 A7, h3;\n"
                "mov.f32 z, 0f00000000;\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {A0, A1, A2, A3}, {$8, $9}, {z, z, z, z};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {A4, A5, A6, A7}, {$10, $11}, {$0, $1, $2, $3};\n"
                "}",
                "=f,=f,=f,=f,r,r,r,r,r,r,r,r", se=False)


def scale_acc(dv, fw0, fw1, fx0, fx1, a):
    """a += D * (fw (rows g, g+8) x fx (cols c0, c1)) with packed f32x2 math"""
    return _asm([Float32] * 4, [dv[0], dv[1], dv[2], dv[3], fw0, fw1, fx0, fx1, a[0], a[1], a[2], a[3]],
                "{ .reg .b64 X, F0, F1, S0, S1, D01, D23, A01, A23;\n"
                "mov.b64 X, {$10, $11}; mov.b64 F0, {$8, $8}; mov.b64 F1, {$9, $9};\n"
                "mul.rn.f32x2 S0, F0, X; mul.rn.f32x2 S1, F1, X;\n"
                "mov.b64 D01, {$4, $5}; mov.b64 D23, {$6, $7}; mov.b64 A01, {$12, $13}; mov.b64 A23, {$14, $15};\n"
                "fma.rn.f32x2 A01, D01, S0, A01; fma.rn.f32x2 A23, D23, S1, A23;\n"
                "mov.b64 {$0, $1}, A01; mov.b64 {$2, $3}, A23; }",
                "=f,=f,=f,=f,f,f,f,f,f,f,f,f,f,f,f,f", se=False)


def cvt_e4m3x4_f16(w):
    """4 e4m3 bytes -> (f16x2 of bytes 0,1 ; f16x2 of bytes 2,3)"""
    return _asm([Int32] * 2, [w], "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; cvt.rn.f16x2.e4m3x2 $0, lo; "
                "cvt.rn.f16x2.e4m3x2 $1, hi; }", "=r,=r,r", se=False)


def lds_u8(a):
    return _asm([Int32], [a], "ld.shared.u8 $0, [$1];", "=r,r")


def sts4i(a, v0, v1, v2, v3):
    _asm([], [a, v0, v1, v2, v3], "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r")


def ld_rlx4(p):
    return _asm([Int32] * 4, [p], "ld.relaxed.gpu.global.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l")


def any_nan4(v):
    return _asm([Int32], [v[0], v[1], v[2], v[3]],
                "{ .reg .pred p0, p1, p2, p3; setp.eq.s32 p0, $1, 0x7fffffff; setp.eq.s32 p1, $2, 0x7fffffff;\n"
                "setp.eq.s32 p2, $3, 0x7fffffff; setp.eq.s32 p3, $4, 0x7fffffff; or.pred p0, p0, p1; or.pred p2, p2, p3;\n"
                "or.pred p0, p0, p2; selp.b32 $0, 1, 0, p0; }", "=r,r,r,r,r", se=False)


def vote_any(v):
    return _asm([Int32], [v], "{ .reg .pred p, q; setp.ne.s32 p, $1, 0; vote.sync.any.pred q, p, 0xffffffff; selp.b32 $0, 1, 0, q; }",
                "=r,r")


def dot8_bf16(w, x, acc):
    for i in range(4):
        wl = i2f(w[i] << 16)
        wh = i2f(w[i] & Int32(-65536))
        xl = i2f(x[i] << 16)
        xh = i2f(x[i] & Int32(-65536))
        acc = ffma(wl, xl, acc)
        acc = ffma(wh, xh, acc)
    return acc


def transpose_reduce(vals, lane, T, zf):
    """Warp-sum T per-lane partials: each butterfly stage halves the token list (P-1+5-log2P shuffles).
    Returns (sum, token index held by this lane, write-enable)."""
    P = 1
    while P < T:
        P *= 2
    vals = list(vals) + [zf] * (P - T)
    mk = 16
    while len(vals) > 1:
        half = len(vals) // 2
        hib = isel(lane & mk, Int32(1), Int32(0))
        nv = []
        for ii in range(half):
            snd = fsel(hib, vals[ii], vals[ii + half])
            kp = fsel(hib, vals[ii + half], vals[ii])
            nv.append(kp + shfl_bfly_f(snd, mk))
        vals = nv
        mk //= 2
    d0 = vals[0]
    while mk >= 1:
        d0 = d0 + shfl_bfly_f(d0, mk)
        mk //= 2
    lsh = 5 - (P.bit_length() - 1)
    tok = lane >> lsh
    wok = eq_i(lane & Int32((1 << lsh) - 1), Int32(0)) & lt_i(tok, Int32(T))
    return d0, tok, wok


def fence_proxy_async():
    _asm([], [], "fence.proxy.async.global;", "")


def smem_layout(T, NS):
    """Byte offsets inside the dynamic shared memory window."""
    o = {}
    off = NS * SLOT
    S = 8 * T                      # max (token, k) rows == max expert slots
    for name, size in (("XH", T * XHST), ("XSF", T * 64), ("XF", T * 256), ("ACTB", S * W2ST), ("ACTSF", S * 64 + 64),
                       ("MB", (3 * NS + 2 + S + 2 * npb_for(T)) * 8), ("PB", npb_for(T) * 4096 * ((T + 7) // 8)),
                       ("TOPE", 32 * T), ("TOPW", 32 * T), ("ACTV", 32), ("TOKM", 1024), ("SLOTOF", 1024),
                       ("SLEXP", 4 * S), ("SLN", 4 * S), ("ROFF", 4 * S + 16), ("RTOK", 4 * S), ("RW", 4 * S),
                       ("WTOT", 64), ("MISC", 64), ("SCNT", 4 * NS), ("RRED", 64 * T), ("OACC", 64 * NCW * T), ("FLAG", 64), ("PAD", SMEM_PAD)):
        off = (off + 127) // 128 * 128
        o[name] = off
        off += size
    o["TOTAL"] = (off + 127) // 128 * 128
    return o


def pick_ns(T, soft, hard):
    # The native activation layout permits a four-stage weight pipeline; reduce only to fit device smem.
    target = NS_TARGET[0] if T <= 4 else (NS_TARGET[1] if T <= 8 else (NS_TARGET[2] if T <= 16 else NS_TARGET[3]))
    for ns in range(target, 1, -1):
        if smem_layout(T, ns)["TOTAL"] <= hard:
            return ns
    return 2


# ----------------------------------------------------------------------------------------------
# the kernel
# ----------------------------------------------------------------------------------------------
@cute.kernel
def moe_kernel(p_xb: Int64, p_wr: Int64, p_x8: Int64, p_xsf: Int64, p_w13: Int64, p_w13sf: Int64,
               p_w2: Int64, p_w2sf: Int64, p_out: Int64, p_cnt: Int64, p_lg: Int64, p_act: Int64,
               p_aq: Int64, p_asf: Int64, d13: cutlass.GridConstant[TensorMap],
               d13s: cutlass.GridConstant[TensorMap], d2: cutlass.GridConstant[TensorMap],
               d2s: cutlass.GridConstant[TensorMap], T: cutlass.Constexpr, NS: cutlass.Constexpr, GS: cutlass.Constexpr):
    L = smem_layout(T, NS)
    NTM = (T + 7) // 8
    NPB = npb_for(T)
    t13 = Int64(d13.get_ptr().toint())
    t13s = Int64(d13s.get_ptr().toint())
    t2 = Int64(d2.get_ptr().toint())
    t2s = Int64(d2s.get_ptr().toint())
    tid0, _, _ = cute.arch.thread_idx()
    cta0, _, _ = cute.arch.block_idx()
    G0, _, _ = cute.arch.grid_dim()
    tid = Int32(tid0)
    cta = Int32(cta0)
    G = Int32(GS)  # compiled from the device SM count read by setup
    warp = tid >> 5
    lane = tid & 31
    sb = Int32(cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024).toint())
    MB = sb + L["MB"]
    FULLB = MB                     # [NS]   weights landed
    EMPTYB = MB + 8 * NS           # [NS]   slot free
    PFULLB = MB + 16 * NS          # [NS]   FC1 partials ready for the epilogue warps
    XBAR = MB + 24 * NS            # x tokens landed
    ABAR = MB + 24 * NS + 8        # [8T]   requantized act of expert slot landed in smem
    XBBAR = ABAR + 64 * T          # router input landed
    PBEB = XBBAR + 8               # [NPB]  partial buffer consumed by the epilogue
    zf = Float32(0.0)
    if tid == 0:
        tz = gtimer()
        for kk in cutlass.range_constexpr(30):
            dbg_put(p_cnt, cta, kk, tz)

    # ---------------- phase 0: router operand loads first (longest latency), then barriers / tables ----------------
    nexp = isel(lt_i(cta, Int32(256)), (Int32(255) - cta) // G + 1, Int32(0))
    jr = tid >> 8                 # threads 0..255 -> expert cta, 256..511 -> expert cta+G
    kr = tid & 255
    rw = (Int32(0), Int32(0), Int32(0), Int32(0))
    if tid < 512 and cta + jr * G < 256:
        rw = tuple(ldg4_nc(p_wr + Int64(cta + jr * G) * 4096 + Int64(kr) * 16))
    rx = [ldg4_nc(p_xb + Int64(t * 4096) + Int64(kr) * 16) for t in range(T if T <= XSTAGE_T else 0)]
    if tid < NS:
        mbar_init(FULLB + 8 * tid, 1)
        mbar_init(EMPTYB + 8 * tid, NCW)
        mbar_init(PFULLB + 8 * tid, NCW)
    if tid < NPB:
        mbar_init(PBEB + 8 * tid, 1)
    if tid < 8 * T:
        mbar_init(ABAR + 8 * tid, 3)
    if tid == 0:
        mbar_init(XBAR, 1)
        mbar_init(XBBAR, 1)
        mbar_init_fence()
        if cutlass.const_expr(T > XSTAGE_T):
            # router input x_bf16 [T, 2048] -> smem (the FC2 operand area is idle until routing is done)
            mbar_expect(XBBAR, Int32(T * 4096))
            bulk_g2s(sb + L["ACTB"], p_xb, Int32(T * 4096), XBBAR)
    if tid < 256:
        sts(sb + L["TOKM"] + 4 * tid, Int32(0))
        sts(sb + L["SLOTOF"] + 4 * tid, Int32(0))
    if tid < 8:
        sts(sb + L["ACTV"] + 4 * tid, Int32(0))
    if tid < NS:
        sts(sb + L["SCNT"] + 4 * tid, Int32(0))
    for k in cutlass.range_constexpr((16 * NCW * T + NTH - 1) // NTH):
        idx = tid + NTH * k
        if idx < 16 * NCW * T:
            sts(sb + L["OACC"] + 4 * idx, Int32(0))
    mbar_init_fence()
    cute.arch.sync_threads()

    if warp == WP:
        if lane == 0:
            prefetch_tmap(t13)
            prefetch_tmap(t13s)
            prefetch_tmap(t2)
            prefetch_tmap(t2s)
            mbar_expect(XBAR, Int32(T * 2048 + T * 64))
        syncwarp()
        for r in cutlass.range_constexpr((T + 31) // 32):
            row = lane + 32 * r
            if row < T:
                bulk_g2s(sb + L["XH"] + row * XHST, p_x8 + Int64(row) * 2048, Int32(2048), XBAR)
        if lane == 0:
            bulk_g2s(sb + L["XSF"], p_xsf, Int32(T * 64), XBAR)

    # ---------------- phase 1: router logits for experts cta + j*G (two experts per pass) ----------------
    # Each logit is published once, then a grid barrier makes the current logits visible to every CTA.
    if cutlass.const_expr(T > XSTAGE_T):
        okb = mbar_try(XBBAR, Int32(0))
        spinb = Int32(0)
        while okb == 0 and spinb < SPIN_CAP:
            okb = mbar_try(XBBAR, Int32(0))
            spinb = spinb + 1
        while okb == 0:
            okb = mbar_try(XBBAR, Int32(0))
    for jp in range((nexp + 1) // 2):
        if jp > 0:
            rw = tuple(ldg4_nc(p_wr + Int64(imin(cta + (2 * jp + jr) * G, Int32(255))) * 4096 + Int64(kr) * 16))
        if tid < 512 and 2 * jp + jr < nexp:
            vals = []
            for t in cutlass.range_constexpr(T):
                if cutlass.const_expr(T > XSTAGE_T):
                    xv = lds4i(sb + L["ACTB"] + t * 4096 + kr * 16)
                    vals.append(dot8_bf16(rw, xv, zf))
                else:
                    vals.append(dot8_bf16(rw, rx[t], zf))
            if tid == 0 and jp == 0:
                dbg_put(p_cnt, cta, 30, gtimer() + Int64(f2i(vals[0]) & 0))
            d0, tok, wok = transpose_reduce(vals, lane, T, zf)
            stsf_if(wok, sb + L["RRED"] + 4 * (warp * T + tok), d0)
        cute.arch.sync_threads()
        if tid < 2 * T:
            tt = tid >> 1
            jj = tid & 1
            ee = cta + (2 * jp + jj) * G
            if ee < 256:
                acc = zf
                for w in cutlass.range_constexpr(8):
                    acc = acc + ldsf(sb + L["RRED"] + 4 * ((8 * jj + w) * T + tt))
                st_rlx_f(p_lg + Int64(tt * 256 + ee) * 4, bf16_round(acc))
        cute.arch.sync_threads()
    # Grid barrier without a release fence: the arrival may overtake this CTA's logit stores, so the routing
    # warps re-read any logit word that still holds the NaN sentinel (the last CTA re-arms it at the end).
    if tid == 0:
        dbg_put(p_cnt, cta, 1, gtimer())
        red_add_rlx(p_cnt + 4 * CNT_BAR, Int32(1))
        bv = ld_rlx(p_cnt + 4 * CNT_BAR)
        spin = Int32(0)
        while bv < G and spin < SPIN_CAP:
            nanosleep(64)
            bv = ld_rlx(p_cnt + 4 * CNT_BAR)
            spin = spin + 1
        while bv < G:
            nanosleep(64)
            bv = ld_rlx(p_cnt + 4 * CNT_BAR)
        dbg_put(p_cnt, cta, 2, gtimer())
    cute.arch.sync_threads()

    # ---------------- phase 2: routing (redundant per CTA) ----------------
    for u in cutlass.range_constexpr((T + NTH // 32 - 1) // (NTH // 32)):
        t = warp + (NTH // 32) * u
        if t < T:
            plg = p_lg + Int64(t * 256 + 8 * lane) * 4
            q0 = ld_rlx4(plg)
            q1 = ld_rlx4(plg + 16)
            v0 = q0[0]
            v1 = q0[1]
            v2 = q0[2]
            v3 = q0[3]
            v4 = q1[0]
            v5 = q1[1]
            v6 = q1[2]
            v7 = q1[3]
            miss = vote_any(any_nan4(q0) | any_nan4(q1))
            while miss != 0:
                nanosleep(64)
                r0 = ld_rlx4(plg)
                r1 = ld_rlx4(plg + 16)
                v0 = r0[0]
                v1 = r0[1]
                v2 = r0[2]
                v3 = r0[3]
                v4 = r1[0]
                v5 = r1[1]
                v6 = r1[2]
                v7 = r1[3]
                miss = vote_any(any_nan4(r0) | any_nan4(r1))
            vb = [v0, v1, v2, v3, v4, v5, v6, v7]
            if tid == 0:
                dbg_put(p_cnt, cta, 31, gtimer() + Int64(v0 & 0))
            if cutlass.const_expr(DBG):
                if tid == 0:
                    dbg_put(p_cnt, cta, 12, gtimer())
            keys = []
            for i in cutlass.range_constexpr(8):
                b = vb[i]
                o = b ^ isel(lt_i(b, Int32(0)), Int32(-1), Int32(-2147483648))
                keys.append((o & Int32(-65536)) | (Int32(255 - i) - 8 * lane))
            for ci in cutlass.range_constexpr(19):
                ia = SORT8[ci][0]
                ib = SORT8[ci][1]
                hi = umax(keys[ia], keys[ib])
                lo = umin(keys[ia], keys[ib])
                keys[ia] = hi
                keys[ib] = lo
            sel = []
            for k in cutlass.range_constexpr(8):
                mk = redux_max_u(keys[0])
                hit = eq_i(keys[0], mk)
                for i in cutlass.range_constexpr(7):
                    keys[i] = isel(hit, keys[i + 1], keys[i])
                keys[7] = isel(hit, Int32(0), keys[7])
                sel.append(mk)
            # Each selected lane evaluates its own exponential; warp shuffles
            # reproduce the original k=0..7 serial FP32 normalization sum.
            myk = sel[0]
            for k in cutlass.range_constexpr(8):
                myk = isel(eq_i(lane, Int32(k)), sel[k], myk)
            kb = myk & Int32(-65536)
            fb = isel(lt_i(kb, Int32(0)), kb ^ Int32(-2147483648), (kb ^ Int32(-1)) & Int32(-65536))
            myval = i2f(fb)
            maxval = shfl_idx_f(myval, 0)
            myv = ex2((myval - maxval) * Float32(LOG2E))
            ssum = zf
            for k in cutlass.range_constexpr(8):
                ssum = ssum + shfl_idx_f(myv, k)
            if tid == 0:
                dbg_put(p_cnt, cta, 32, gtimer() + Int64(f2i(ssum) & 0))
            if lane < 8:
                ee = Int32(255) - (myk & Int32(255))
                sts(sb + L["TOPE"] + 4 * (t * 8 + lane), ee)
                stsf(sb + L["TOPW"] + 4 * (t * 8 + lane), fdiv(myv, ssum))
                red_or_s(sb + L["ACTV"] + 4 * (ee >> 5), Int32(1) << (ee & 31))
                red_or_s(sb + L["TOKM"] + 4 * ee, Int32(1) << t)
    cute.arch.sync_threads()
    if tid == 0:
        dbg_put(p_cnt, cta, 35, gtimer())
    if cutlass.const_expr(DBG):
        if tid == 0:
            dbg_put(p_cnt, cta, 13, gtimer())

    # Only the eight expert warps build the prefix metadata. Explicit initial
    # values carry the live scan results through staged branches and the barrier.
    e_me = tid & 255
    isact = Int32(0)
    slot_me = Int32(0)
    ne_me = Int32(0)
    incl = Int32(0)
    if tid < 256:
        aw = [lds(sb + L["ACTV"] + 4 * i) for i in range(8)]
        wsel = e_me >> 5
        bit = e_me & 31
        pre = Int32(0)
        myw = Int32(0)
        for i in cutlass.range_constexpr(8):
            pre = pre + isel(lt_i(Int32(i), wsel), popc(aw[i]), Int32(0))
            myw = myw | isel(eq_i(wsel, Int32(i)), aw[i], Int32(0))
        isact = (myw >> bit) & 1
        lowmask = (Int32(1) << bit) - 1
        slot_me = pre + popc(myw & lowmask)
        tm_me = lds(sb + L["TOKM"] + 4 * e_me)
        ne_me = popc(tm_me)
        incl = ne_me
        for di in cutlass.range_constexpr(5):
            y = shfl_up_i(incl, 1 << di)
            incl = incl + isel(lt_i(Int32((1 << di) - 1), lane), y, Int32(0))
        if lane == 31:
            sts(sb + L["WTOT"] + 4 * warp, incl)
    # The producer warp overlaps D/R selection with the expert prefix scan.
    if warp == WP:
        ds = Int32(0)
        for i in cutlass.range_constexpr(8):
            ds = ds + popc(lds(sb + L["ACTV"] + 4 * i))
        nns = G - imin(G, Int32(128))
        dn = imax(nns, Int32(1))
        ms = ds * 64
        rf = (ds * (256 - nns)) // (4 * G)
        rm = (ms + G - 1) // G
        # measured per-item costs (0.05 us units): owner FC1 tile, FC2 quad, non-owner FC1 tile; the
        # non-owners' last FC1 tiles are followed by the hand-off tail (epilogue, ready poll, requant, FC2)
        rc_sched = imax(Int32(0), imin(rf - 12 + lane, rm))
        co = Int32(SC_OWN1) * rc_sched + Int32(SC_FC2) * ((ds + 3) >> 2)
        cn = Int32(SC_NON1) * (rc_sched + (imax(ms - rc_sched * G, Int32(0)) + dn - 1) // dn) + Int32(SC_TAIL)
        key = (imax(co, cn) << 5) | lane
        winner = redux_min_u(key) & 31
        chosen = imax(Int32(0), imin(rf - 12 + winner, rm))
        chosen = isel(lt_i(Int32(0), nns), chosen, rm)
        # With concentrated routes, each FC1 tile has several token groups.
        # Use all resident CTAs for the balanced FC1 prefix before FC2.
        if cutlass.const_expr(T > 8):
            chosen = isel(lt_i(ds, Int32(17)), rm, chosen)
        # Issue this CTA's first FC1 ring stages now (before the slot tables exist): lane l covers experts
        # 8l..8l+7 of the active bitmask, so the k-th active expert is found with one scan + one reduction.
        npre = Int32(0)
        if cutlass.const_expr(EARLY_ISSUE):
            wd = lds(sb + L["ACTV"] + 4 * (lane >> 2))
            byt = (wd >> (8 * (lane & 3))) & 255
            bc = popc(byt)
            binc = bc
            for di in cutlass.range_constexpr(5):
                yv = shfl_up_i(binc, 1 << di)
                binc = binc + isel(lt_i(Int32((1 << di) - 1), lane), yv, Int32(0))
            bex = binc - bc
            nr_e = isel(lt_i(cta, ms), imax(Int32(0), imin(chosen, (ms - cta + G - 1) // G)), Int32(0))
            npre = imin(nr_e, Int32(NS))
            epol = l2_evict_first_policy()
            for ip in cutlass.range_constexpr(NS):
                if ip < npre:
                    mpre = ip * G + cta
                    kq = mpre >> 6
                    rq = kq - bex
                    pos = Int32(0)
                    cq = Int32(0)
                    for bb in cutlass.range_constexpr(8):
                        bit = (byt >> bb) & 1
                        pos = isel(bit & eq_i(cq, rq), Int32(bb), pos)
                        cq = cq + bit
                    own = lt_i(kq, binc) & (Int32(1) - lt_i(kq, bex))
                    epre = redux_max_u(isel(own, 8 * lane + pos + 1, Int32(0))) - 1
                    rpre = (mpre & 63) * 16
                    if lane == 0:
                        mbar_expect(FULLB + 8 * ip, Int32(FC1_BYTES))
                        if cutlass.const_expr(FC1_TMA):
                            tma3h(sb + ip * SLOT, t13, Int32(0), epre * 1024 + rpre, Int32(0), FULLB + 8 * ip, epol)
                    syncwarp()
                    if cutlass.const_expr(not FC1_TMA):
                        if lane < 16:
                            bulk_g2s_h(sb + ip * SLOT + lane * XST, p_w13 + Int64(epre) * W13_EXP + Int64((rpre + lane) * 2048),
                                       Int32(2048), FULLB + 8 * ip, epol)
                    if lane == 16:
                        tma4(sb + ip * SLOT + FC1_SF_OFF, t13s, Int32(0), (rpre >> 4) & 1, Int32(0),
                             epre * 8 + (rpre >> 7), FULLB + 8 * ip)
        if lane == 0:
            sts(sb + L["MISC"], ds)
            sts(sb + L["MISC"] + 4, chosen)
            sts(sb + L["MISC"] + 8, npre)
    cute.arch.sync_threads()
    if tid == 0:
        dbg_put(p_cnt, cta, 33, gtimer())
    if tid < 256:
        wt = [lds(sb + L["WTOT"] + 4 * i) for i in range(8)]
        base = Int32(0)
        for i in cutlass.range_constexpr(8):
            base = base + isel(lt_i(Int32(i), warp), wt[i], Int32(0))
        excl = base + incl - ne_me
        if isact != 0:
            sts(sb + L["SLOTOF"] + 4 * e_me, slot_me)
            sts(sb + L["SLEXP"] + 4 * slot_me, e_me)
            sts(sb + L["SLN"] + 4 * slot_me, ne_me)
            sts(sb + L["ROFF"] + 4 * slot_me, excl)
    cute.arch.sync_threads()
    if tid == 0:
        dbg_put(p_cnt, cta, 34, gtimer())
    # Prime the first shared-memory weight slot during token-table construction.
    if warp == (WP if not EARLY_ISSUE else 1 << 20):
        early_d = lds(sb + L["MISC"])
        early_r = lds(sb + L["MISC"] + 4)
        early_no = imin(G, Int32(128))
        early_m = isel(lt_i(Int32(0), early_r), cta, early_r * G + cta - early_no)
        early_fc1 = Int32(0)
        if early_r > 0 and cta < early_d * 64:
            early_fc1 = Int32(1)
        if early_r == 0 and cta >= early_no and early_m < early_d * 64:
            early_fc1 = Int32(1)
        early_pol = l2_evict_first_policy()
        if early_fc1 != 0:
            early_e = lds(sb + L["SLEXP"] + 4 * (early_m >> 6))
            early_row = (early_m & 63) * 16
            if lane == 0:
                mbar_expect(FULLB, Int32(FC1_BYTES))
            syncwarp()
            if lane < 16:
                bulk_g2s_h(sb + lane * XST, p_w13 + Int64(early_e) * W13_EXP + Int64((early_row + lane) * 2048),
                           Int32(2048), FULLB, early_pol)
            if lane == 16:
                tma4(sb + FC1_SF_OFF, t13s, Int32(0), (early_row >> 4) & 1, Int32(0),
                     early_e * 8 + (early_row >> 7), FULLB)
        else:
            if cta < 128:
                early_nv = imin(Int32(4), early_d)
                if lane == 0:
                    mbar_expect(FULLB, early_nv * FC2_EXP_BYTES)
                if lane < early_nv:
                    early_e2 = lds(sb + L["SLEXP"] + 4 * lane)
                    tma3h(sb + lane * 8192, t2, Int32(0), early_e2 * 2048 + 16 * cta, Int32(0), FULLB, early_pol)
                    tma4(sb + FC2_SF_OFF + lane * 1024, t2s, Int32(0), cta & 1, Int32(0),
                         early_e2 * 16 + (cta >> 3), FULLB)
    if tid < 8 * T:
        tt = tid >> 3
        ee2 = lds(sb + L["TOPE"] + 4 * tid)
        ww2 = ldsf(sb + L["TOPW"] + 4 * tid)
        s2 = lds(sb + L["SLOTOF"] + 4 * ee2)
        tm2 = lds(sb + L["TOKM"] + 4 * ee2)
        n2 = popc(tm2 & ((Int32(1) << tt) - 1))
        r2 = lds(sb + L["ROFF"] + 4 * s2) + n2
        sts(sb + L["RTOK"] + 4 * r2, tt)
        stsf(sb + L["RW"] + 4 * r2, ww2)
    cute.arch.sync_threads()
    D = lds(sb + L["MISC"])
    if tid == 0:
        dbg_put(p_cnt, cta, 3, gtimer())

    # ---------------- phase 3: static work schedule (identical in every thread) ----------------
    M1 = D * 64
    NO = imin(G, Int32(128))
    NN = G - NO
    nNN = imax(NN, Int32(1))
    R = lds(sb + L["MISC"] + 4)
    nr = imax(Int32(0), imin(R, (M1 - cta + G - 1) // G))
    nr = isel(lt_i(cta, M1), nr, Int32(0))
    E1 = imax(M1 - R * G, Int32(0))
    xo = cta - NO
    nx = imax(Int32(0), (E1 - xo + nNN - 1) // nNN)
    nx = isel(lt_i(xo, Int32(0)), Int32(0), nx)
    nx = isel(lt_i(Int32(0), NN), nx, Int32(0))
    n1 = nr + nx
    nsl = isel(lt_i(cta, Int32(128)), (Int32(127) - cta) // G + 1, Int32(0))
    nq = (D + 3) >> 2
    n_items = n1 + nsl * nq

    # ---------------- phase 4a: weight producer warp ----------------
    if warp == WP:
        npre_l = Int32(1)
        if cutlass.const_expr(EARLY_ISSUE):
            npre_l = lds(sb + L["MISC"] + 8)
        pol = l2_evict_first_policy()
        wemp = Int64(0)
        for i in range(n_items):
            s = i % NS
            ph = (i // NS) & 1
            twe = gtimer()
            if i >= NS:
                ok = mbar_try(EMPTYB + 8 * s, ph ^ 1)
                spin = Int32(0)
                while ok == 0 and spin < SPIN_CAP:
                    ok = mbar_try(EMPTYB + 8 * s, ph ^ 1)
                    spin = spin + 1
                while ok == 0:
                    ok = mbar_try(EMPTYB + 8 * s, ph ^ 1)
            wemp = wemp + (gtimer() - twe)
            fullb = FULLB + 8 * s
            dst = sb + s * SLOT
            if i >= npre_l:
                if i < n1:
                    m = isel(lt_i(i, nr), i * G + cta, R * G + (i - nr) * NN + xo)
                    slot_e = m >> 6
                    mt = m & 63
                    e = lds(sb + L["SLEXP"] + 4 * slot_e)
                    r0 = mt * 16
                    if lane == 0:
                        mbar_expect(fullb, Int32(FC1_BYTES))
                        if cutlass.const_expr(FC1_TMA):
                            tma3h(dst, t13, Int32(0), e * 1024 + r0, Int32(0), fullb, pol)
                    syncwarp()
                    if cutlass.const_expr(not FC1_TMA):
                        if lane < 16:
                            # one contiguous 2 KB row per lane (sequential DRAM pages), padded smem rows
                            bulk_g2s_h(dst + lane * XST, p_w13 + Int64(e) * W13_EXP + Int64((r0 + lane) * 2048),
                                       Int32(2048), fullb, pol)
                    if lane == 16:
                        tma4(dst + FC1_SF_OFF, t13s, Int32(0), (r0 >> 4) & 1, Int32(0), e * 8 + (r0 >> 7), fullb)
                else:
                    jq = i - n1
                    sl = cta + (jq // nq) * G
                    qd = jq % nq
                    nval = imin(Int32(4), D - 4 * qd)
                    j0 = 16 * sl
                    if lane == 0:
                        mbar_expect(fullb, nval * FC2_EXP_BYTES)
                    if lane < nval:
                        e = lds(sb + L["SLEXP"] + 4 * (4 * qd + lane))
                        tma3h(dst + lane * 8192, t2, Int32(0), e * 2048 + j0, Int32(0), fullb, pol)
                        tma4(dst + FC2_SF_OFF + lane * 1024, t2s, Int32(0), (j0 >> 4) & 1, Int32(0), e * 16 + (j0 >> 7),
                             fullb)
            for pf in cutlass.range_constexpr(PF_ITEMS):
                jn = i + NS * (pf + 1)
                if jn < n_items:
                    if jn < n1:
                        mp = isel(lt_i(jn, nr), jn * G + cta, R * G + (jn - nr) * NN + xo)
                        ep = lds(sb + L["SLEXP"] + 4 * (mp >> 6))
                        rp = (mp & 63) * 16
                        if lane == 0:
                            l2_prefetch(p_w13 + Int64(ep) * W13_EXP + Int64(rp * 2048), Int32(32768))
                        if lane == 1:
                            l2_prefetch(p_w13sf + Int64(ep) * W13SF_EXP + Int64((rp >> 7) * 8192), Int32(8192))
                    else:
                        jqp = jn - n1
                        slp = cta + (jqp // nq) * G
                        qdp = jqp % nq
                        nvp = imin(Int32(4), D - 4 * qdp)
                        if lane < nvp:
                            ep2 = lds(sb + L["SLEXP"] + 4 * (4 * qdp + lane))
                            l2_prefetch(p_w2 + Int64(ep2) * W2_EXP + Int64(16 * slp * 512), Int32(8192))
                            l2_prefetch(p_w2sf + Int64(ep2) * W2SF_EXP + Int64(((16 * slp) >> 7) * 2048), Int32(2048))
            if cutlass.const_expr(DBG):
                if lane == 0 and i < 24:
                    dbg_put(p_cnt, cta, 16 + i, gtimer())
        if lane == 0:
            dbg_put(p_cnt, cta, 11, wemp)

    # ---------------- phase 4b: FC2 operand loader warps (owners only): fp32 act -> OCP MXFP8 in smem ----------------
    if warp >= WL and warp < WL + NL:
        if nsl > 0:
            lw = warp - WL
            dense_loader = Int32(0)
            if cutlass.const_expr(T > 8):
                dense_loader = lt_i(D, Int32(17))
            if dense_loader != 0:
                # Two three-warp groups process experts independently.
                # All three participants arrive, including those with no rows.
                eg = lw // 3
                rq_lane = lw % 3
                for eq in range((D - eg + 1) // 2):
                    se = eg + eq * 2
                    rdy = ld_acq(p_cnt + Int64(4 * (CNT_READY + se)))
                    spin = Int32(0)
                    while rdy < READY_TILES and spin < SPIN_CAP:
                        nanosleep(POLL_NS)
                        rdy = ld_acq(p_cnt + Int64(4 * (CNT_READY + se)))
                        spin = spin + 1
                    while rdy < READY_TILES:
                        nanosleep(POLL_NS)
                        rdy = ld_acq(p_cnt + Int64(4 * (CNT_READY + se)))
                    ne = lds(sb + L["SLN"] + 4 * se)
                    ro = lds(sb + L["ROFF"] + 4 * se)
                    # One lane converts four adjacent values. Each eight-lane subgroup owns one 32-wide block.
                    nrq = imax(Int32(0), (ne - rq_lane + 2) // 3)
                    for rq in range(nrq):
                        rc = rq_lane + rq * 3
                        row = ro + rc
                        av = [ldg4f_cg(p_act + Int64((row * 512 + 128 * b4 + 4 * lane) * 4))
                              for b4 in range(4)]
                        for b4 in cutlass.range_constexpr(4):
                            vv = av[b4]
                            mx = fmax_abs(fmax_abs(vv[0], vv[1]), fmax_abs(vv[2], vv[3]))
                            for sh in cutlass.range_constexpr(3):
                                mx = fmax(mx, shfl_xor_f(mx, 1 << sh))
                            am = f2i(mx)
                            ex = imax((am >> 23) & 255, Int32(8))
                            inv = i2f((Int32(262) - ex) << 23)
                            qb = e4m3x4(vv[0], vv[1], vv[2], vv[3], inv)
                            sts(sb + L["ACTB"] + row * W2ST + 128 * b4 + 4 * lane, qb)
                            if (lane & 7) == 0:
                                sts(sb + L["ACTSF"] + row * 64 + 4 * (4 * b4 + (lane >> 3)), (ex - 8) << 23)
                    fence_cta()
                    syncwarp()
                    if lane == 0:
                        mbar_arrive(ABAR + 8 * se)
                        dbg_put(p_cnt, cta, 20 + lw, gtimer())
            else:
                cntL = imax(Int32(0), (D - lw + NL - 1) // NL)
                for kk2 in range(cntL):
                    se = lw + kk2 * NL
                    rdy = ld_acq(p_cnt + Int64(4 * (CNT_READY + se)))
                    spin = Int32(0)
                    while rdy < READY_TILES and spin < SPIN_CAP:
                        nanosleep(POLL_NS)
                        rdy = ld_acq(p_cnt + Int64(4 * (CNT_READY + se)))
                        spin = spin + 1
                    while rdy < READY_TILES:
                        nanosleep(POLL_NS)
                        rdy = ld_acq(p_cnt + Int64(4 * (CNT_READY + se)))
                    ne = lds(sb + L["SLN"] + 4 * se)
                    ro = lds(sb + L["ROFF"] + 4 * se)
                    # One lane converts four adjacent values. Each eight-lane subgroup owns one 32-wide block.
                    for rc in range(ne):
                        row = ro + rc
                        av = [ldg4f_cg(p_act + Int64((row * 512 + 128 * b4 + 4 * lane) * 4))
                              for b4 in range(4)]
                        for b4 in cutlass.range_constexpr(4):
                            vv = av[b4]
                            mx = fmax_abs(fmax_abs(vv[0], vv[1]), fmax_abs(vv[2], vv[3]))
                            for sh in cutlass.range_constexpr(3):
                                mx = fmax(mx, shfl_xor_f(mx, 1 << sh))
                            am = f2i(mx)
                            ex = imax((am >> 23) & 255, Int32(8))
                            inv = i2f((Int32(262) - ex) << 23)
                            qb = e4m3x4(vv[0], vv[1], vv[2], vv[3], inv)
                            sts(sb + L["ACTB"] + row * W2ST + 128 * b4 + 4 * lane, qb)
                            if (lane & 7) == 0:
                                sts(sb + L["ACTSF"] + row * 64 + 4 * (4 * b4 + (lane >> 3)), (ex - 8) << 23)
                    fence_cta()
                    syncwarp()
                    if lane == 0:
                        mbar_arrive_n(ABAR + 8 * se, 3)
                        dbg_put(p_cnt, cta, 20 + lw, gtimer())

    # ---------------- phase 4c: FC1 epilogue warps (cross-warp sum, SwiGLU, fp32 act out, tile count) ----------------
    if warp >= WE:
        h = warp - WE
        g = lane >> 2
        tq = lane & 3
        # every ring slot is served by exactly one epilogue warp, in item order, so an mbarrier parity
        # can never alias a phase two uses ahead
        for i in range(n1):
            s = i % NS
            if s % NE == h:
                ph = (i // NS) & 1
                okp = mbar_try(PFULLB + 8 * s, ph)
                spin = Int32(0)
                while okp == 0 and spin < SPIN_CAP:
                    okp = mbar_try(PFULLB + 8 * s, ph)
                    spin = spin + 1
                while okp == 0:
                    okp = mbar_try(PFULLB + 8 * s, ph)
                dst = sb + s * SLOT
                m = isel(lt_i(i, nr), i * G + cta, R * G + (i - nr) * NN + xo)
                slot_e = m >> 6
                mt = m & 63
                ne = lds(sb + L["SLN"] + 4 * slot_e)
                ro = lds(sb + L["ROFF"] + 4 * slot_e)
                iidx = 16 * (mt >> 1) + 2 * g + (mt & 1)
                for nt in cutlass.range_constexpr(NTM):
                    if nt * 8 < ne:
                        s0 = zf
                        s1 = zf
                        s2 = zf
                        s3 = zf
                        for w in cutlass.range_constexpr(NCW):
                            pv = lds4f(sb + L["PB"] + (i % NPB) * (4096 * NTM) + 4096 * nt + 512 * w + lane * 16)
                            s0 = s0 + pv[0]
                            s1 = s1 + pv[1]
                            s2 = s2 + pv[2]
                            s3 = s3 + pv[3]
                        # rows g = up_i, g+8 = gate_i ; cols = slot tokens 2tq, 2tq+1
                        act0 = fdiv(s2 * s0, Float32(1.0) + ex2(s2 * Float32(-LOG2E)))
                        act1 = fdiv(s3 * s1, Float32(1.0) + ex2(s3 * Float32(-LOG2E)))
                        cc0 = Int32(8 * nt) + 2 * tq
                        if cc0 < ne:
                            stgf(p_act + Int64(((ro + cc0) * 512 + iidx) * 4), act0)
                        if cc0 + 1 < ne:
                            stgf(p_act + Int64(((ro + cc0 + 1) * 512 + iidx) * 4), act1)
                syncwarp()
                if lane == 0:
                    mbar_arrive(PBEB + 8 * (i % NPB))
                    if cutlass.const_expr(DBG):
                        if i < 24:
                            dbg_put(p_cnt, cta, 88 + i, gtimer())
                if lane == 0:
                    red_add_rel(p_cnt + Int64(4 * (CNT_READY + slot_e)), Int32(1))
                    dbg_put(p_cnt, cta, 12 + h, gtimer())
                    if cutlass.const_expr(DBG):
                        if i < 16:
                            dbg_put(p_cnt, cta, 112 + i, gtimer())

    # ---------------- phase 4e: self-reset off the critical path ----------------
    # Once this CTA's loader and epilogue warps are done with the global counters (grid barrier passed, all
    # ready-counter increments and reads issued), it signals DONE; the last CTA to signal re-arms the counters.
    # The MMA warps keep running and finish with plain output stores (no end-of-kernel atomic round trip).
    if warp >= WL:
        bar_sync(3, (NL + NE) * 32)
        if warp == WL:
            od = Int32(0)
            if lane == 0:
                od = atom_add_ar(p_cnt + 4 * CNT_DONE, Int32(1))
            od = shfl_idx_i(od, 0)
            if od == G - 1:
                for k in cutlass.range_constexpr((8 * T + 31) // 32):
                    idx = lane + 32 * k
                    if idx < D:
                        stg_rlx(p_cnt + Int64(4 * (CNT_READY + idx)), Int32(0))
                for k in cutlass.range_constexpr(T * 256 // 128):
                    st_rlx4(p_lg + Int64((lane + 32 * k) * 16), Int32(0x7FFFFFFF))
                if lane == 0:
                    stg_rlx(p_cnt + 4 * CNT_DONE, Int32(0))
                    stg_rlx(p_cnt + 4 * CNT_BAR, Int32(0))

    # ---------------- phase 4d: MMA consumer warps ----------------
    if warp < NCW:
        g = lane >> 2
        tq = lane & 3
        ra = (lane & 7) + ((lane >> 3) & 1) * 8
        ca = (lane >> 4) * 16
        xboff = ((lane >> 3) & 1) * 16 + (lane >> 4) * 32
        okx = mbar_try(XBAR, Int32(0))
        spin = Int32(0)
        while okx == 0 and spin < SPIN_CAP:
            okx = mbar_try(XBAR, Int32(0))
            spin = spin + 1
        while okx == 0:
            okx = mbar_try(XBAR, Int32(0))
        # Keep activations in native E4M3 and expand only UE8M0 block scales.
        for k in cutlass.range_constexpr((64 * T + NCW * 32 - 1) // (NCW * 32)):
            idx = tid + NCW * 32 * k
            if idx < 64 * T:
                sts(sb + L["XF"] + 4 * idx, lds_u8(sb + L["XSF"] + idx) << 23)
        bar_sync(2, NCW * 32)
        abase = ra * 128
        arow = ra * XST + (lane >> 4) * 16
        r7 = ra & 7
        hsel = lane >> 4
        swo = [(((jx << 1) | hsel) ^ r7) << 4 for jx in range(4)]
        wfull = Int64(0)
        wabar = Int64(0)
        tfc2 = Int64(0)
        for i in range(n_items):
            s = i % NS
            ph = (i // NS) & 1
            twf = gtimer()
            okf = mbar_try(FULLB + 8 * s, ph)
            spin2 = Int32(0)
            while okf == 0 and spin2 < SPIN_CAP:
                okf = mbar_try(FULLB + 8 * s, ph)
                spin2 = spin2 + 1
            while okf == 0:
                okf = mbar_try(FULLB + 8 * s, ph)
            wfull = wfull + (gtimer() - twf)
            if tid == 0 and i == 0:
                dbg_put(p_cnt, cta, 4, gtimer())
            if tid == 0 and i < 28:
                dbg_put(p_cnt, cta, 40 + i, gtimer())
            if cutlass.const_expr(DBG):
                if tid == 0 and i < 24:
                    dbg_put(p_cnt, cta, 40 + i, gtimer())
            dst = sb + s * SLOT
            ck0 = clk64()
            is1 = lt_i(i, n1)
            m = isel(lt_i(i, nr), i * G + cta, R * G + (i - nr) * NN + xo)
            jq = imax(i - n1, Int32(0))
            sl = cta
            qd = Int32(0)
            if is1 == 0:
                sl = cta + (jq // nq) * G
                qd = jq % nq
            if is1 != 0:
                # ======== FC1 tile: 16 stored rows (8 gate/up pairs) x K=2048; warp owns 8 k-blocks
                slot_e = m >> 6
                mt = m & 63
                ne = lds(sb + L["SLN"] + 4 * slot_e)
                ro = lds(sb + L["ROFF"] + 4 * slot_e)
                qsf = (mt >> 1) & 3
                swa = [lds(dst + FC1_SF_OFF + (2 * warp + cc) * 256 + (g + 8 * rh) * 16 + qsf * 4)
                       for rh in range(2) for cc in range(2)]
                accs = tuple(zf for _ in range(4 * NTM))
                for nt in cutlass.range_constexpr(NTM):
                    if nt * 8 < ne:
                        nb = imin(Int32(8 * nt) + (lane & 7), ne - 1)
                        tokb = lds(sb + L["RTOK"] + 4 * (ro + nb))
                        xaddr = sb + L["XH"] + tokb * XHST + 256 * warp + 16 * (lane >> 3)
                        c0 = imin(Int32(8 * nt) + 2 * tq, ne - 1)
                        c1 = imin(Int32(8 * nt) + 2 * tq + 1, ne - 1)
                        tk0 = lds(sb + L["RTOK"] + 4 * (ro + c0))
                        tk1 = lds(sb + L["RTOK"] + 4 * (ro + c1))
                        sx0 = [lds4f(sb + L["XF"] + tk0 * 256 + 32 * warp + 16 * hh) for hh in range(2)]
                        sx1 = [lds4f(sb + L["XF"] + tk1 * 256 + 32 * warp + 16 * hh) for hh in range(2)]
                        a0 = zf
                        a1 = zf
                        a2 = zf
                        a3 = zf
                        abk = dst + arow + 256 * warp
                        abt = dst + abase + 4096 * warp
                        for hg in cutlass.range_constexpr(2):
                            # 4 independent k-blocks in flight: loads, then MMAs, then block scales
                            bq = [ldsm4(xaddr + 128 * hg + 64 * p) for p in range(2)]
                            if cutlass.const_expr(FC1_TMA):
                                afs = [ldsm4(abt + 2048 * hg + swo[j]) for j in range(4)]
                            else:
                                afs = [ldsm4(abk + 128 * hg + 32 * j) for j in range(4)]
                            dvs = [mma_e4m3(afs[j], bq[j >> 1][2 * (j & 1)], bq[j >> 1][2 * (j & 1) + 1], zf)
                                   for j in range(4)]
                            for j in cutlass.range_constexpr(4):
                                acc4 = scale_acc(dvs[j], sf_f32(swa[hg], j), sf_f32(swa[2 + hg], j),
                                                 sx0[hg][j], sx1[hg][j], [a0, a1, a2, a3])
                                a0 = acc4[0]
                                a1 = acc4[1]
                                a2 = acc4[2]
                                a3 = acc4[3]
                        accs = accs[:4 * nt] + (a0, a1, a2, a3) + accs[4 * nt + 4:]
                pbi = i % NPB
                if i >= NPB:
                    okq = mbar_try(PBEB + 8 * pbi, ((i // NPB) & 1) ^ 1)
                    spinq = Int32(0)
                    while okq == 0 and spinq < SPIN_CAP:
                        okq = mbar_try(PBEB + 8 * pbi, ((i // NPB) & 1) ^ 1)
                        spinq = spinq + 1
                    while okq == 0:
                        okq = mbar_try(PBEB + 8 * pbi, ((i // NPB) & 1) ^ 1)
                pbuf = sb + L["PB"] + pbi * (4096 * NTM)
                for nt in cutlass.range_constexpr(NTM):
                    sts4f(pbuf + 4096 * nt + 512 * warp + lane * 16,
                          accs[4 * nt], accs[4 * nt + 1], accs[4 * nt + 2], accs[4 * nt + 3])
            else:
                # ======== FC2 quad: 4 experts x 16 rows of H; warp = (expert jj, k-half kh)
                jj = warp & 3
                kh = warp >> 2
                slot_e = 4 * qd + jj
                if slot_e < D:
                    twa = gtimer()
                    oka = mbar_try(ABAR + 8 * slot_e, Int32(0))
                    spin3 = Int32(0)
                    while oka == 0 and spin3 < SPIN_CAP:
                        oka = mbar_try(ABAR + 8 * slot_e, Int32(0))
                        spin3 = spin3 + 1
                    while oka == 0:
                        oka = mbar_try(ABAR + 8 * slot_e, Int32(0))
                    wabar = wabar + (gtimer() - twa)
                    tfc2 = gtimer()
                    ne = lds(sb + L["SLN"] + 4 * slot_e)
                    ro = lds(sb + L["ROFF"] + 4 * slot_e)
                    edst = dst + jj * 8192
                    qsf = (sl & 7) >> 1
                    swa = [lds(dst + FC2_SF_OFF + jj * 1024 + (2 * kh + cc) * 256 + (g + 8 * rh) * 16 + qsf * 4)
                           for rh in range(2) for cc in range(2)]
                    oa = sb + L["OACC"] + 4 * (warp * 16 * T)
                    for nt in cutlass.range_constexpr(NTM):
                        if nt * 8 < ne:
                            nb = imin(Int32(8 * nt) + (lane & 7), ne - 1)
                            baddr = sb + L["ACTB"] + (ro + nb) * W2ST + xboff + kh * 256
                            cc0 = Int32(8 * nt) + 2 * tq
                            cc1 = cc0 + 1
                            c0 = imin(cc0, ne - 1)
                            c1 = imin(cc1, ne - 1)
                            sx0 = [lds4f(sb + L["ACTSF"] + (ro + c0) * 64 + 32 * kh + 16 * hh) for hh in range(2)]
                            sx1 = [lds4f(sb + L["ACTSF"] + (ro + c1) * 64 + 32 * kh + 16 * hh) for hh in range(2)]
                            a0 = zf
                            a1 = zf
                            a2 = zf
                            a3 = zf
                            abk = edst + abase + 4096 * kh
                            for hg in cutlass.range_constexpr(2):
                                bfa = ldsm4(baddr + (2 * hg) * 64)
                                bfb = ldsm4(baddr + (2 * hg + 1) * 64)
                                afs = [ldsm4(abk + 2048 * hg + swo[j]) for j in range(4)]
                                dvs = [mma_e4m3(afs[0], bfa[0], bfa[1], zf), mma_e4m3(afs[1], bfa[2], bfa[3], zf),
                                       mma_e4m3(afs[2], bfb[0], bfb[1], zf), mma_e4m3(afs[3], bfb[2], bfb[3], zf)]
                                for j in cutlass.range_constexpr(4):
                                    acc4 = scale_acc(dvs[j], sf_f32(swa[hg], j), sf_f32(swa[2 + hg], j),
                                                     sx0[hg][j], sx1[hg][j], [a0, a1, a2, a3])
                                    a0 = acc4[0]
                                    a1 = acc4[1]
                                    a2 = acc4[2]
                                    a3 = acc4[3]
                            wr0 = ldsf(sb + L["RW"] + 4 * (ro + c0))
                            wr1 = ldsf(sb + L["RW"] + 4 * (ro + c1))
                            tk0 = lds(sb + L["RTOK"] + 4 * (ro + c0))
                            tk1 = lds(sb + L["RTOK"] + 4 * (ro + c1))
                            if cc0 < ne:
                                pa = oa + 4 * (g * T + tk0)
                                stsf(pa, ffma(a0, wr0, ldsf(pa)))
                                pb = oa + 4 * ((g + 8) * T + tk0)
                                stsf(pb, ffma(a2, wr0, ldsf(pb)))
                            if cc1 < ne:
                                pa = oa + 4 * (g * T + tk1)
                                stsf(pa, ffma(a1, wr1, ldsf(pa)))
                                pb = oa + 4 * ((g + 8) * T + tk1)
                                stsf(pb, ffma(a3, wr1, ldsf(pb)))

            # ---- hand-back: every MMA warp arrives once; the barriers complete after NCW arrivals
            ck1 = clk64()
            syncwarp()
            if lane == 0:
                if is1 != 0:
                    mbar_arrive(PFULLB + 8 * s)
                mbar_arrive(EMPTYB + 8 * s)
                if cutlass.const_expr(DBG):
                    if warp == 0 and i < 24:
                        dbg_put(p_cnt, cta, 64 + i, gtimer())
            if cutlass.const_expr(DBG):
                ck2 = clk64()
                if lane == 0 and i < 16:
                    dbg_put(p_cnt, cta, 128 + warp * 16 + i, ck1 - ck0)
                pass
            if tid == 0 and i == n1 - 1:
                dbg_put(p_cnt, cta, 5, gtimer())
            # end of a slice: flush the CTA's 16 x T output block (fixed-order sum over warps)
            if is1 == 0:
                if qd == nq - 1:
                    bar_sync(1, NCW * 32)
                    for k in cutlass.range_constexpr((16 * T + NCW * 32 - 1) // (NCW * 32)):
                        idx = tid + NCW * 32 * k
                        if idx < 16 * T:
                            rr = idx & 15
                            tt = idx >> 4
                            v = zf
                            for w in cutlass.range_constexpr(NCW):
                                pa = sb + L["OACC"] + 4 * ((w * 16 + rr) * T + tt)
                                v = v + ldsf(pa)
                                stsf(pa, zf)
                            jrow = 16 * sl + rr
                            hcol = 32 * (jrow >> 5) + 4 * (jrow & 7) + ((jrow & 31) >> 3)
                            stg_bf16(p_out + Int64((tt * 2048 + hcol) * 2), v)
                    bar_sync(1, NCW * 32)
        if tid == 0:
            dbg_put(p_cnt, cta, 8, wfull)
            dbg_put(p_cnt, cta, 9, wabar)
            dbg_put(p_cnt, cta, 10, tfc2)

    if tid == 0:
        dbg_put(p_cnt, cta, 6, gtimer())
        dbg_put(p_cnt, cta, 7, Int64(D) * Int64(4294967296) + Int64(n1) * 65536 + Int64(n_items))
    if cutlass.const_expr(DBG):
        if lane == 0:
            if warp == WP:
                dbg_put(p_cnt, cta, 7, gtimer())
            if warp == WE:
                dbg_put(p_cnt, cta, 8, gtimer())
            if warp == WL:
                dbg_put(p_cnt, cta, 9, gtimer())
            if warp == 0:
                dbg_put(p_cnt, cta, 11, Int64(n1) * 65536 + Int64(n_items) * 256 + Int64(D))


@cute.jit
def moe_host(p_xb: Int64, p_wr: Int64, p_x8: Int64, p_xsf: Int64, p_w13: Int64, p_w13sf: Int64,
             p_w2: Int64, p_w2sf: Int64, p_out: Int64, p_cnt: Int64, p_lg: Int64, p_act: Int64,
             p_aq: Int64, p_asf: Int64, G: Int32, stream: cuda.CUstream,
             T: cutlass.Constexpr, NS: cutlass.Constexpr, GS: cutlass.Constexpr):
    L = smem_layout(T, NS)
    E = 256
    # W13 [E*1024 rows][16 kchunks][128 B] viewed as (128 B, rows, kchunk): one box = 16 rows x 2048 B,
    # landing in smem as [kchunk][row][128 B] with the 128B swizzle keyed on the row -> conflict-free ldmatrix
    d13 = create_tensor_map_tiled(p_w13, cutlass.Float8E4M3FN, [128, E * 1024, 16], [2048 // 16, 128 // 16],
                                  [128, 16, 16], swizzle=TensorMapSwizzle.s128b)
    # W13 scales (F8 128x4 interleave): (256 B half-chunk, half, kchunk4 x16, 128-row block)
    d13s = create_tensor_map_tiled(p_w13sf, cutlass.Uint8, [256, 2, 16, E * 8], [256 // 16, 512 // 16, 8192 // 16],
                                   [256, 1, 16, 1])
    d2 = create_tensor_map_tiled(p_w2, cutlass.Float8E4M3FN, [128, E * 2048, 4], [512 // 16, 128 // 16],
                                 [128, 16, 4], swizzle=TensorMapSwizzle.s128b)
    d2s = create_tensor_map_tiled(p_w2sf, cutlass.Uint8, [256, 2, 4, E * 16], [256 // 16, 512 // 16, 2048 // 16],
                                  [256, 1, 4, 1])
    moe_kernel(p_xb, p_wr, p_x8, p_xsf, p_w13, p_w13sf, p_w2, p_w2sf, p_out, p_cnt, p_lg, p_act, p_aq, p_asf,
               d13, d13s, d2, d2s, T, NS, GS).launch(grid=[G, 1, 1], block=[NTH, 1, 1], smem=L["TOTAL"], stream=stream)


def compile_for(T, NS=NS_DEFAULT, GS=1, opts=None):
    z = Int64(0)
    args = [z] * 14 + [Int32(1), cuda.CUstream(0)]
    if opts:
        return cute.compile(moe_host, *args, T, NS, GS, options=opts)
    return cute.compile(moe_host, *args, T, NS, GS)


# ----------------------------------------------------------------------------------------------
# host API
# ----------------------------------------------------------------------------------------------
class _State:
    pass


def _smem_optin(dev):
    """Max dynamic shared memory per block (opt-in), asked from the driver first."""
    notes = []
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    vals = {}
    try:
        err, d = cuda.cuDeviceGet(idx)
        for nm in ("CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
                   "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR"):
            err2, v = cuda.cuDeviceGetAttribute(getattr(cuda.CUdevice_attribute, nm), d)
            vals[nm[-6:]] = (int(err2), int(v))
    except Exception as ex:
        notes.append("drv:%s" % repr(ex)[:60])
    notes.append(str(vals))
    try:
        notes.append("torch=%d" % int(torch.cuda.get_device_properties(dev).shared_memory_per_block_optin))
    except Exception as ex:
        notes.append("torch:%s" % type(ex).__name__)
    e, v = vals.get("_OPTIN", (1, 0))
    if e == 0 and v > 0:
        return v, " ".join(notes)
    return 232448, " ".join(notes)


def setup(context):
    T = int(context["axes"]["T"])
    dev = torch.device(context.get("device", "cuda"))
    props = torch.cuda.get_device_properties(dev)
    smem_max, st_note = _smem_optin(dev)
    st = _State()
    st.T = T
    st.G = props.multi_processor_count
    st.smem_max = smem_max
    st.note = st_note
    hard = smem_max
    if tuple(torch.cuda.get_device_capability(dev)) == (10, 7):
        hard = max(smem_max, SMEM_HARD)
    st.NS = pick_ns(T, min(SMEM_SOFT, hard), hard)
    st.fn = compile_for(T, st.NS, st.G)
    S = 8 * T
    st.cnt = torch.zeros(CNT_ALLOC, dtype=torch.int32, device=dev)
    st.lg = torch.full((T * 256,), 0x7FFFFFFF, dtype=torch.int32, device=dev)   # NaN sentinel
    st.act = torch.zeros(S * 512, dtype=torch.float32, device=dev)
    st.aq = torch.zeros(S * 512, dtype=torch.uint8, device=dev)
    st.asf = torch.zeros(S * 16 + 16, dtype=torch.uint8, device=dev)
    torch.cuda.synchronize(dev)
    return {"state": st, "input_buffers": {}}


_NCALL = [0]


def _prof_dump(st):
    import numpy as np
    torch.cuda.synchronize()
    G = st.G
    d = st.cnt[CNT_DBG:].view(torch.int64).view(-1, 256)[:G].cpu().numpy().astype(np.int64)
    t0 = d[:, 0].min()
    rel = lambda k: (d[:, k] - t0) / 1000.0
    D = int(d[0, 7] >> 32)
    own = np.arange(G) < min(G, 128)
    s = "PROF call=%d T=%d D=%d |" % (_NCALL[0], st.T, D)
    for k, nm in ((1, "rt"), (2, "bar"), (3, "tab"), (4, "1st"), (5, "fc1end"), (6, "end")):
        s += " %s o%.2f/%.2f n%.2f/%.2f" % (nm, rel(k)[own].mean(), rel(k)[own].max(), rel(k)[~own].mean() if (~own).any() else 0,
                                          rel(k)[~own].max() if (~own).any() else 0)
    s += " | rdot %.2f lgld %.2f top8 %.2f sync1 %.2f scan %.2f abar %.2f" % tuple(rel(k).mean() for k in (30, 31, 32, 35, 33, 34))
    ep = np.max(d[:, 12:16], axis=1)
    s += " | lastrdy %.2f" % ((ep.max() - t0) / 1000.0)
    s += " | wfull o%.2f n%.2f wabar o%.2f wemp o%.2f n%.2f lastfc2 o%.2f" % (
        d[own, 8].mean() / 1000, d[~own, 8].mean() / 1000 if (~own).any() else 0, d[own, 9].mean() / 1000,
        d[own, 11].mean() / 1000, d[~own, 11].mean() / 1000 if (~own).any() else 0, rel(10)[own].max())
    n1 = (d[:, 7] >> 16) & 0xFFFF
    ni = d[:, 7] & 0xFFFF
    s += " | n1 o%d-%d n%d-%d items o%d" % (n1[own].min(), n1[own].max(), n1[~own].min() if (~own).any() else 0,
                                          n1[~own].max() if (~own).any() else 0, ni[own].max())
    ld = (d[own, 20:26] - t0) / 1000.0
    s += " | ldr %.2f" % ld.max()
    c = int(np.argmax(d[:, 6]))
    s += " | slowcta %d items:" % c + ",".join("%.1f" % ((d[c, 40 + i] - t0) / 1000.0) for i in range(min(int(ni[c]), 28)))
    print(s, flush=True)


def run(x_bf16, w_router, x_fp8, x_sf, w13, w13_sf, w2, w2_sf, state, out):
    st = state
    if isinstance(st, dict):
        st = st["state"]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    st.fn(Int64(x_bf16.data_ptr()), Int64(w_router.data_ptr()), Int64(x_fp8.data_ptr()), Int64(x_sf.data_ptr()),
          Int64(w13.data_ptr()), Int64(w13_sf.data_ptr()), Int64(w2.data_ptr()), Int64(w2_sf.data_ptr()),
          Int64(out.data_ptr()), Int64(st.cnt.data_ptr()), Int64(st.lg.data_ptr()),
          Int64(st.act.data_ptr()), Int64(st.aq.data_ptr()), Int64(st.asf.data_ptr()),
          Int32(st.G), stream)
    if PROF:
        _NCALL[0] += 1
        n = _NCALL[0]
        if (n <= 8 or n % 97 == 0) and not torch.cuda.is_current_stream_capturing():
            _prof_dump(st)
