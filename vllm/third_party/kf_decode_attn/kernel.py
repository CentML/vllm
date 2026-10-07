# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# mypy: ignore-errors
# fmt: off
# Kernel Factory solution, verbatim below this header (do not edit; swap the whole file).
#   campaign  jkzbdr4bg92wn8vc5rjshqn0y0 (phaseB-cutedsl-0103b), artifact paged-fp8-decode-attn-h16kv2d256-p128-graph
#   candidate 10553759f57f7e9cfad757bb6592d846514738ab8a21a615d411e3ee28d06b6d (KF 1.58x, 26.29 vs 41.46 us geomean)
#   definition paged_fp8_decode_attn_h16kv2d256_p128_graph (MTP k=4: Q in {5, 1}; Q <= 8 supported by the kernel)
#   source sha256 27d56ca9c04c7800b8cc54765ad96f881ed8deff56ba2a4d37cc25eb0685ecee (bytes after this header)
# Retrieve: kf campaign kernel show 10553759f57f7e9cfad757bb6592d846514738ab8a21a615d411e3ee28d06b6d --source kernel.py
# runtime.py uses _compile/constants from here (setup()/run() are the solution's own glue, used by tests).
# Paged-KV FP8 decode/verify attention for Rubin (sm_107a), CuTe DSL.
#
# One persistent launch per call (grid = #SMs).  Every CTA derives the KV split from the device
# seq_lens: the work unit is one (request, kv-head) stream split into page chunks so that all CTAs
# stream about the same number of KV pages.  Per CTA:
#   * TMA warp streams K and V pages (128 tokens x 256 B each) through an NS-stage smem ring.
#   * MMA warp issues S = Q K^T (tcgen05 FP8, M=128 rows, N=128 tokens, K=256) into TMEM and
#     O += P V (A = P from TMEM, B = V from smem, N=256) into TMEM.
#   * 8 softmax warps: warp w owns TMEM lanes 32*(w%4).. (one query row per lane) and token half
#     w//4 of every page (the two halves exchange their block max through smem).  Every quadrant
#     holds two 16-lane groups; lane 32*q + 16*g + i carries query i of head 2*q + g ("hi" copy)
#     and lane 32*q + 16*g + 8 + i the "lo" copy of the same row, so one 16x256b TMEM load hands
#     each thread both copies of an O element (register combine, no smem staging).  Hi rows
#     store P_hi = e4m3(x), lo rows store P_lo = e4m3((x - P_hi) * 16), so O = O_hi + O_lo / 16
#     carries ~8 significant bits of P while V stays FP8 (no V conversion).  x = exp2(s*c - m_ref +
#     log2(448) - TAU) with a lazily updated row max m_ref (O rescaled only when the max grows by
#     more than TAU).
#   * Split streams are merged inside the launch: partial (O, m, l) go to a gmem workspace, the
#     stream's CTAs meet on a self-resetting counter and each merges 1/cs of the outputs in a fixed
#     order (deterministic).
import math

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, Float32, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
from cutlass.cute.nvgpu.tcgen05 import OperandSource, CtaGroup
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.utils import rubin_helpers as sm107_utils
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
import cutlass.experimental.cuda as cuda_exp
from cutlass.experimental import primitives as prims

HQ, HKV, D, PAGE, GROUP, MAXP = 16, 2, 256, 128, 8, 2057
NS = 2                      # K/V smem pipeline stages, with independent K release
TAU = 4.0                   # lazy-rescale threshold (log2 units): P <= 2^TAU relative to m_ref
SCALE_LOG2 = math.log2(448.0) - TAU
LOG2E = 1.4426950408889634
MAX_GRID = 256
STAGE_GAP = 800             # ns between the first and second page (BL > 1)
SPEC_GAP = 800              # ns between speculative page issues (BL == 1)
MW = 8                      # warps running the split merge
SPEC = 2                    # BL == 1: first K/V pages issued straight from seq_lens[0] (no split math)
W_SIZE = 1024               # merge weight table (floats)
NEG_INF = -math.inf
EVICT_FIRST = 0x12F0000000000000
EVICT_LAST = 0x14F0000000000000
EVICT_NORMAL = 0x1000000000000000
BOX_MIN_CS = 32             # merge fetch: 2-D boxes only above this many partials (exact bulk copies below)
KV_STAGE = PAGE * D         # bytes per K (or V) ring stage: two 128-row x 128 B swizzle boxes
KV_BOX = PAGE * 128         # bytes per box
BT_PER = 128                # BL >= 2: first-page block-table entries staged in smem per request (>= grid / HKV)


# ----------------------------------------------------------------------------------------------
# small PTX helpers
# ----------------------------------------------------------------------------------------------
@dsl_user_op
def pack_e4m3x4(a, b, c, d, *, loc=None, ip=None) -> Uint32:
    """bytes [a, b, c, d] (a in the low byte) = e4m3(f16(x)), RN, satfinite."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
             Float32(c).ir_value(loc=loc, ip=ip), Float32(d).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 xab, xcd;\n\t.reg .b16 l, h;\n\t"
            "cvt.rn.f16x2.f32 xab, $2, $1;\n\t"
            "cvt.rn.f16x2.f32 xcd, $4, $3;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 l, xab;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 h, xcd;\n\t"
            "mov.b32 $0, {l, h};\n\t}",
            "=r,f,f,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def pack_e4m3x4_residual(a, b, c, d, *, loc=None, ip=None) -> Uint32:
    """e4m3((f16(x) - e4m3(f16(x))) * 16) for x in [a, b, c, d] (exact residual in f16)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
             Float32(c).ir_value(loc=loc, ip=ip), Float32(d).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 xab, xcd, hab, hcd, k16;\n\t.reg .b16 eab, ecd;\n\t"
            "mov.b32 k16, 0x4C004C00;\n\t"
            "cvt.rn.f16x2.f32 xab, $2, $1;\n\t"
            "cvt.rn.f16x2.f32 xcd, $4, $3;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 eab, xab;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 ecd, xcd;\n\t"
            "cvt.rn.f16x2.e4m3x2 hab, eab;\n\t"
            "cvt.rn.f16x2.e4m3x2 hcd, ecd;\n\t"
            "sub.rn.f16x2 xab, xab, hab;\n\t"
            "sub.rn.f16x2 xcd, xcd, hcd;\n\t"
            "mul.rn.f16x2 xab, xab, k16;\n\t"
            "mul.rn.f16x2 xcd, xcd, k16;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 eab, xab;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 ecd, xcd;\n\t"
            "mov.b32 $0, {eab, ecd};\n\t}",
            "=r,f,f,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def pack_e4m3x4_sel(a, b, c, d, sel_hi, *, loc=None, ip=None) -> Uint32:
    """sel_hi != 0: e4m3(f16(x)) (as pack_e4m3x4); else the residual e4m3((f16(x) - e4m3(f16(x))) * 16).
    Hi and lo copies of a row share a warp, so both are computed and selected without divergence."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
             Float32(c).ir_value(loc=loc, ip=ip), Float32(d).ir_value(loc=loc, ip=ip),
             Uint32(sel_hi).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 xab, xcd, hab, hcd, k16, hiw, low;\n\t.reg .b16 eab, ecd, lab, lcd;\n\t"
            ".reg .pred p;\n\t"
            "mov.b32 k16, 0x4C004C00;\n\t"
            "cvt.rn.f16x2.f32 xab, $2, $1;\n\t"
            "cvt.rn.f16x2.f32 xcd, $4, $3;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 eab, xab;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 ecd, xcd;\n\t"
            "mov.b32 hiw, {eab, ecd};\n\t"
            "cvt.rn.f16x2.e4m3x2 hab, eab;\n\t"
            "cvt.rn.f16x2.e4m3x2 hcd, ecd;\n\t"
            "sub.rn.f16x2 xab, xab, hab;\n\t"
            "sub.rn.f16x2 xcd, xcd, hcd;\n\t"
            "mul.rn.f16x2 xab, xab, k16;\n\t"
            "mul.rn.f16x2 xcd, xcd, k16;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 lab, xab;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 lcd, xcd;\n\t"
            "mov.b32 low, {lab, lcd};\n\t"
            "setp.ne.u32 p, $5, 0;\n\t"
            "selp.b32 $0, hiw, low, p;\n\t}",
            "=r,f,f,f,f,r",
            has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def tmem_ld_hilo_x8(taddr, *, loc=None, ip=None):
    """tcgen05.ld 16x256b.x8 from 16 TMEM lanes: thread t gets lane t/4 (hi copy) and lane t/4 + 8 (lo copy)
    at columns 8k + 2(t%4) + {0, 1}, k < 8.  Returns the 16 combined values hi + lo / 16 (column order)."""
    regs = ", ".join(f"r{i}" for i in range(32))
    body = ["{", ".reg .b32 " + regs + ";", ".reg .b64 hi, lo, result, scale;",
            "mov.b64 scale, {0f3D800000, 0f3D800000};",
            "tcgen05.ld.sync.aligned.16x256b.x8.b32 {" + regs + "}, [$16];",
            "tcgen05.wait::ld.sync.aligned;"]
    for k in range(8):
        body.append(f"mov.b64 hi, {{r{4 * k}, r{4 * k + 1}}};")
        body.append(f"mov.b64 lo, {{r{4 * k + 2}, r{4 * k + 3}}};")
        body.append("fma.rn.f32x2 result, lo, scale, hi;")
        body.append(f"mov.b64 {{${2 * k}, ${2 * k + 1}}}, result;")
    body.append("}")
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 16),
        [Int32(taddr).ir_value(loc=loc, ip=ip)],
        "\n\t".join(body),
        ",".join(["=f"] * 16) + ",r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(16)]


@dsl_user_op
def weighted_fma4(w, x0, x1, x2, x3, a0, a1, a2, a3, *, loc=None, ip=None):
    """Two full-precision SIMD pairs, preserving each element's FP32 FMA order."""
    args = [w, x0, x1, x2, x3, a0, a1, a2, a3]
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Float32(x).ir_value(loc=loc, ip=ip) for x in args],
        "{ .reg .b64 wp, xp0, xp1, ap0, ap1;\n"
        "mov.b64 wp, {$4, $4};\n"
        "mov.b64 xp0, {$5, $6}; mov.b64 xp1, {$7, $8};\n"
        "mov.b64 ap0, {$9, $10}; mov.b64 ap1, {$11, $12};\n"
        "fma.rn.f32x2 ap0, wp, xp0, ap0;\n"
        "fma.rn.f32x2 ap1, wp, xp1, ap1;\n"
        "mov.b64 {$0, $1}, ap0; mov.b64 {$2, $3}, ap1; }",
        ",".join(["=f"] * 4 + ["f"] * 9),
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def add_f32x4(a0, a1, a2, a3, b0, b1, b2, b3, *, loc=None, ip=None):
    """Four independent FP32 additions; each sum chain retains its scalar operation order."""
    args = [a0, a1, a2, a3, b0, b1, b2, b3]
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Float32(x).ir_value(loc=loc, ip=ip) for x in args],
        "{ .reg .b64 a01, a23, b01, b23;\n"
        "mov.b64 a01, {$4, $5}; mov.b64 a23, {$6, $7};\n"
        "mov.b64 b01, {$8, $9}; mov.b64 b23, {$10, $11};\n"
        "add.rn.f32x2 a01, a01, b01; add.rn.f32x2 a23, a23, b23;\n"
        "mov.b64 {$0, $1}, a01; mov.b64 {$2, $3}, a23; }",
        ",".join(["=f"] * 4 + ["f"] * 8),
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_global_f32x2(addr: Int64, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v2.f32 {$0, $1}, [$2];", "=f,=f,l",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def st_global_f32x2(addr: Int64, a, b, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [Int64(addr).ir_value(loc=loc, ip=ip), Float32(a).ir_value(loc=loc, ip=ip),
               Float32(b).ir_value(loc=loc, ip=ip)],
        "st.global.v2.f32 [$0], {$1, $2};", "l,f,f",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_global_bf16x2(addr: Int64, a, b, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [Int64(addr).ir_value(loc=loc, ip=ip), Float32(a).ir_value(loc=loc, ip=ip),
               Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 v;\n\tcvt.rn.bf16x2.f32 v, $2, $1;\n\tst.global.b32 [$0], v;\n\t}", "l,f,f",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def pack_e4m3x4_own_send(a, b, c, d, sel_hi, *, loc=None, ip=None):
    """hi = e4m3(f16(x)), lo = e4m3((f16(x) - hi) * 16) for x in [a, b, c, d] (a in the low byte).
    Returns (own, send): sel_hi != 0 -> (hi, lo), else (lo, hi)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
         Float32(c).ir_value(loc=loc, ip=ip), Float32(d).ir_value(loc=loc, ip=ip),
         Uint32(sel_hi).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 xab, xcd, hab, hcd, k16, hiw, low;\n\t.reg .b16 eab, ecd, lab, lcd;\n\t"
        ".reg .pred p;\n\t"
        "mov.b32 k16, 0x4C004C00;\n\t"
        "cvt.rn.f16x2.f32 xab, $3, $2;\n\t"
        "cvt.rn.f16x2.f32 xcd, $5, $4;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 eab, xab;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 ecd, xcd;\n\t"
        "mov.b32 hiw, {eab, ecd};\n\t"
        "cvt.rn.f16x2.e4m3x2 hab, eab;\n\t"
        "cvt.rn.f16x2.e4m3x2 hcd, ecd;\n\t"
        "sub.rn.f16x2 xab, xab, hab;\n\t"
        "sub.rn.f16x2 xcd, xcd, hcd;\n\t"
        "mul.rn.f16x2 xab, xab, k16;\n\t"
        "mul.rn.f16x2 xcd, xcd, k16;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 lab, xab;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 lcd, xcd;\n\t"
        "mov.b32 low, {lab, lcd};\n\t"
        "setp.ne.u32 p, $6, 0;\n\t"
        "selp.b32 $0, hiw, low, p;\n\t"
        "selp.b32 $1, low, hiw, p;\n\t}",
        "=r,=r,f,f,f,f,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)))


@dsl_user_op
def sel8(flag, va, vb, ty: str, *, loc=None, ip=None):
    """flag != 0 ? va[i] : vb[i] for 8 values (ty 'f' = Float32, 'r' = Uint32), one predicate."""
    lt = T.f32() if ty == "f" else T.i32()
    cls = Float32 if ty == "f" else Uint32
    body = ["{", ".reg .pred p;", "setp.ne.u32 p, $8, 0;"]
    for i in range(8):
        body.append(f"selp.b32 ${i}, ${9 + i}, ${17 + i}, p;")
    body.append("}")
    res = llvm.inline_asm(
        llvm.StructType.get_literal([lt] * 8),
        [Uint32(flag).ir_value(loc=loc, ip=ip)] + [cls(v).ir_value(loc=loc, ip=ip) for v in va]
        + [cls(v).ir_value(loc=loc, ip=ip) for v in vb],
        "\n\t".join(body),
        ",".join([f"={ty}"] * 8) + ",r," + ",".join([ty] * 16),
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [cls(llvm.extractvalue(lt, res, [i], loc=loc, ip=ip)) for i in range(8)]


@dsl_user_op
def fmax3(a, b, c, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
             Float32(c).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2, $3;", "=f,f,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def nanosleep_ns(ns: int):
    llvm.inline_asm(None, [], f"nanosleep.u32 {ns};", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def publish_arrival(addr: Int64, limit: Int32, *, loc=None, ip=None):
    """Release-only wrapping increment: no acquire stall on the publisher; its own later acquire poll
    of the same counter observes the increment (same-location program order)."""
    llvm.inline_asm(
        None, [addr.ir_value(loc=loc, ip=ip), limit.ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.inc.u32 [$0], $1;", "l,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def bulk_g2s(gmem_addr: Int64, smem_addr: Int32, mbar_addr: Int32, nbytes: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(gmem_addr).ir_value(loc=loc, ip=ip), Int32(smem_addr).ir_value(loc=loc, ip=ip),
         Int32(mbar_addr).ir_value(loc=loc, ip=ip), Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$1], [$0], $3, [$2];",
        "l,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def prefetch_l2(gmem_addr: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        "prefetch.global.L2 [$0];", "l",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def prefetch_l2_bulk(gmem_addr: Int64, nbytes: int, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        f"cp.async.bulk.prefetch.L2.global [$0], {nbytes};", "l",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def warp_max(v: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [v.ir_value(loc=loc, ip=ip)],
        "redux.sync.max.f32 $0, $1, 0xffffffff;", "=f,f",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def warp_isum(v: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [v.ir_value(loc=loc, ip=ip)],
        "redux.sync.add.s32 $0, $1, 0xffffffff;", "=r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@cute.jit
def warp_sum(v: Float32) -> Float32:
    for i in cutlass.range_constexpr(5):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << i)
    return v


def gptr(dtype, addr, align=16):
    return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)


def sptr(dtype, addr, align=16):
    return cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=align)


# ----------------------------------------------------------------------------------------------
# kernel
# ----------------------------------------------------------------------------------------------
SM_WARPS = 8                # softmax/epilogue warps: warp w owns TMEM lanes 32*(w%4).. and token half w//4
NUM_SM_THREADS = SM_WARPS * 32


class PagedDecodeAttn:
    def __init__(self, Q: int, BL: int, grid: int, bl1: bool = False, unsplit: bool = False):
        self.BL = BL
        self.grid = grid
        self.Q = Q
        self.unsplit = unsplit
        self.bl1 = bl1                           # BL == 1 specialization (speculative start)
        # BL >= 2: each request's first-page block-table entries are loaded with seq_lens and staged in
        # the (idle) merge-weight table, so the first K/V loads need no table round trip after the split
        self.bt_smem = (not bl1) and (not unsplit) and BL * BT_PER <= W_SIZE
        self.R = Q * GROUP                       # query rows per (request, kv head) stream
        self.RH = self.R // 2
        assert Q <= 8
        self.n_act_warps = SM_WARPS              # every softmax warp holds live rows
        self.tma_warp = SM_WARPS
        self.mma_warp = SM_WARPS + 1
        self.threads = (SM_WARPS + 2) * 32
        self.mma_tiler_qk = (128, 128, 256)
        self.mma_tiler_pv = (128, 256, 128)

    @cute.jit
    def __call__(self, q_u8: cute.Tensor, kv_u8: cute.Tensor, bt: cute.Tensor, sl: cute.Tensor,
                 out: cute.Tensor, po: cute.Tensor, pml: cute.Tensor, cnt: cute.Tensor,
                 bmm1: Float32, bmm2: Float32, grid_size: Int32, kv_policy: Int64, stream):
        fp8 = cutlass.Float8E4M3FN
        n_pages = kv_u8.shape[0]
        kv_ptr = cute.recast_ptr(kv_u8.iterator, dtype=fp8)
        # K: (tok, dim, head, page);  V: (dim, tok, head, page)  (V is the MN-major B of P@V)
        mK = cute.make_tensor(kv_ptr, cute.make_layout((PAGE, D, HKV, n_pages),
                                                       stride=(2 * D, 1, PAGE * 2 * D, HKV * PAGE * 2 * D)))
        mV = cute.make_tensor(kv_ptr + D, cute.make_layout((D, PAGE, HKV, n_pages),
                                                           stride=(1, 2 * D, PAGE * 2 * D, HKV * PAGE * 2 * D)))

        tiled_mma_qk = sm107_utils.make_trivial_tiled_mma(
            fp8, fp8, OperandMajorMode.K, OperandMajorMode.K, Float32, CtaGroup.ONE, (128, 128, 64))
        tiled_mma_pv = sm107_utils.make_trivial_tiled_mma(
            fp8, fp8, OperandMajorMode.K, OperandMajorMode.MN, Float32, CtaGroup.ONE, (128, 256, 64),
            a_source=OperandSource.TMEM)

        sQ_layout = sm100_utils.make_smem_layout_a(tiled_mma_qk, self.mma_tiler_qk, fp8, 1)
        sK_layout = sm100_utils.make_smem_layout_b(tiled_mma_qk, self.mma_tiler_qk, fp8, NS)
        sV_layout = sm100_utils.make_smem_layout_b(tiled_mma_pv, self.mma_tiler_pv, fp8, NS)
        tP_layout = sm100_utils.make_smem_layout_a(tiled_mma_pv, self.mma_tiler_pv, fp8, 1)

        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma_qk.thr_id.shape,))
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(CtaGroup.ONE)
        tma_atom_K, mK_tma = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, mK, cute.select(sK_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk,
            cluster_layout_vmnk.shape)
        tma_atom_V, mV_tma = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, mV, cute.select(sV_layout, mode=[0, 1, 2]), self.mma_tiler_pv, tiled_mma_pv,
            cluster_layout_vmnk.shape)

        # K/V loads use our own descriptor over the whole page row (512 B = K | V per token), so
        # the TMA can promote L2 fills to 256 B: a K (or V) row is two 128 B swizzle boxes, and the
        # first box's fill brings in the second one with one DRAM access instead of two.
        kv_tmap = cuda_exp.create_tensor_map_tiled(
            kv_u8.iterator.toint(), cutlass.Uint8, [2 * D, PAGE, HKV, n_pages],
            [2 * D // 16, PAGE * 2 * D // 16, HKV * PAGE * 2 * D // 16], [128, 128, 1, 1],
            swizzle=cuda_exp.TensorMapSwizzle.s128b, l2_promotion=cuda_exp.TensorMapL2Promotion.l2_256b)

        # partial-O TMA store: gmem partial [slot*R + row, 256] fp32, smem tiles of R x 32 (128B swizzle)
        R = self.R
        po2d = cute.make_tensor(po.iterator, cute.make_layout((po.shape[0] * R, D), stride=(D, 1)))
        sPO_tile = cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0,
                                             cute.make_layout((R, 32), stride=(32, 1)))
        sPO_layout = cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0,
                                               cute.make_layout((R, 32, 8), stride=(32, 1, R * 32)))
        tma_atom_PO, mPO_tma = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), po2d, sPO_tile, (R, 32))
        # merge fetch: partial-major view (R*256 floats, slot) -> boxes of 64 floats x 32 slots
        pm2d = cute.make_tensor(po.iterator, cute.make_layout((R * D, po.shape[0]), stride=(1, R * D)))
        tma_atom_M, mPM_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), pm2d, cute.make_layout((64, 32), stride=(1, 64)), (64, 32))

        @cute.struct
        class SharedStorage:
            bar_kfull: cute.struct.MemRange[Int64, NS]
            bar_vfull: cute.struct.MemRange[Int64, NS]
            bar_kvempty: cute.struct.MemRange[Int64, NS]
            bar_sfull: cute.struct.MemRange[Int64, NS]
            bar_pfull: cute.struct.MemRange[Int64, 2]
            bar_odone: cute.struct.MemRange[Int64, 1]
            bar_ofinal: cute.struct.MemRange[Int64, 1]
            bar_qfull: cute.struct.MemRange[Int64, 1]
            bar_merge: cute.struct.MemRange[Int64, 1]
            tmem_addr: cute.struct.MemRange[Int32, 4]
            sMx: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 2 * 128], 16]   # [buf][half][lane]
            sLp: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 128], 16]       # [half][lane]
            sW: cute.struct.Align[cute.struct.MemRange[Float32, W_SIZE], 16]
            sQ: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(sQ_layout)], 1024]
            # The rings are reused by a 96 KiB merge fetch and two FP32 epilogue images.
            sK: cute.struct.Align[cute.struct.MemRange[fp8, max(cute.cosize(sK_layout), 96 * 1024)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[fp8, max(cute.cosize(sV_layout), R * D * 8)], 1024]

        self.kernel(
            tiled_mma_qk, tiled_mma_pv, tma_atom_K, mK_tma, tma_atom_V, mV_tma,
            tma_atom_PO, mPO_tma, tma_atom_M, mPM_tma, kv_tmap,
            q_u8, bt, sl, out, po, pml, cnt, bmm1 * LOG2E, bmm2, kv_policy,
            kv_u8.iterator.toint(), n_pages,
            sQ_layout, sK_layout, sV_layout, tP_layout, sPO_layout, SharedStorage,
        ).launch(
            grid=[grid_size, 1, 1],
            block=[self.threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(self, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma,
               tma_atom_K: cute.CopyAtom, mK_tma: cute.Tensor, tma_atom_V: cute.CopyAtom, mV_tma: cute.Tensor,
               tma_atom_PO: cute.CopyAtom, mPO_tma: cute.Tensor,
               tma_atom_M: cute.CopyAtom, mPM_tma: cute.Tensor,
               kv_tmap: cutlass.GridConstant[cuda_exp.TensorMap],
               mQ: cute.Tensor, mBT: cute.Tensor, mSL: cute.Tensor, mOut: cute.Tensor,
               mPO: cute.Tensor, mPML: cute.Tensor, mCnt: cute.Tensor,
               scale_log2: Float32, bmm2: Float32, kv_policy: Int64, kv_base: Int64, kv_pages: Int32,
               sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout,
               sV_layout: cute.ComposedLayout, tP_layout: cute.ComposedLayout,
               sPO_layout: cute.ComposedLayout, SharedStorage: cutlass.Constexpr):
        Qn = self.Q
        R = self.R
        RH = self.RH
        tidx = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        bid = cute.arch.block_idx()[0]
        G = Int32(self.grid)
        BL = Int32(self.BL)
        pg_spec = Int32(0)
        sl0 = Int32(0)
        # Load the device work plan inputs before descriptor and barrier setup.
        # All warps reuse these values for splitting; no second dependent load.
        L_l = Int32(0)
        if const_expr(not self.bl1 and not self.unsplit):
            if lane < BL:
                L_l = mSL[lane]
        bt_first = []
        if const_expr(self.bt_smem):
            # entries [0, grid / HKV) of every request row: a split's first page index is below that
            for bt_t in cutlass.range_constexpr((self.BL * BT_PER + NUM_SM_THREADS - 1) // NUM_SM_THREADS):
                bt_e = tidx + bt_t * NUM_SM_THREADS
                bt_v = Int32(0)
                if bt_e < self.BL * BT_PER and warp < SM_WARPS:
                    if bt_e % BT_PER < G // HKV:
                        bt_v = mBT[bt_e // BT_PER, bt_e % BT_PER]
                bt_first.append(bt_v)
        if const_expr(self.bl1):
            # seq_lens[0] and the block-table entries of this split's first pages: both loads in flight
            # together (the table index is < G <= 256, inside the 2057-wide row; entries are only used
            # for pages below ceil(seq_len / 128))
            sl0 = mSL[0]
            if warp == self.tma_warp:
                if lane < SPEC:
                    pg_spec = mBT[0, bid // HKV + lane * (G // HKV)]

        # ---- shared storage, barriers (TMA warp) and, for BL == 1, the speculative K/V loads: as early in
        # the instruction stream as possible
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        bar_kfull = storage.bar_kfull.data_ptr()
        bar_vfull = storage.bar_vfull.data_ptr()
        bar_kvempty = storage.bar_kvempty.data_ptr()
        bar_sfull = storage.bar_sfull.data_ptr()
        bar_pfull = storage.bar_pfull.data_ptr()
        bar_odone = storage.bar_odone.data_ptr()
        bar_ofinal = storage.bar_ofinal.data_ptr()
        bar_qfull = storage.bar_qfull.data_ptr()
        bar_merge = storage.bar_merge.data_ptr()
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        if warp == self.tma_warp:
            if lane == 0:
                for i in cutlass.range_constexpr(NS):
                    cute.arch.mbarrier_init(bar_kfull + i, 1)
                    cute.arch.mbarrier_init(bar_vfull + i, 1)
                    cute.arch.mbarrier_init(bar_kvempty + i, 1)
                    cute.arch.mbarrier_init(bar_sfull + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(bar_pfull + i, self.n_act_warps)
                cute.arch.mbarrier_init(bar_odone, 1)
                cute.arch.mbarrier_init(bar_ofinal, 1)
                cute.arch.mbarrier_init(bar_qfull, SM_WARPS)
                cute.arch.mbarrier_init(bar_merge, 1)
            cute.arch.mbarrier_init_fence()
            cpasync.prefetch_descriptor(tma_atom_M)
            if const_expr(not self.bl1):
                # the K/V descriptor is cold after the L2 flush: fetch it while seq_lens is in flight
                prims.prefetch_tensormap(kv_tmap.get_ptr())
            if const_expr(self.bl1):
                sp_h = bid % HKV
                if bid < (G // HKV) * HKV:
                    prims.prefetch_tensormap(kv_tmap.get_ptr())
                    gK0 = cute.local_tile(mK_tma[None, None, sp_h, None], cute.select(self.mma_tiler_qk, mode=[1, 2]),
                                          (None, 0, None))
                    gV0 = cute.local_tile(mV_tma[None, None, sp_h, None], cute.select(self.mma_tiler_pv, mode=[1, 2]),
                                          (0, None, None))
                    tKsK0, tKgK0 = cpasync.tma_partition(tma_atom_K, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3),
                                                         cute.group_modes(thr_mma_qk.partition_B(gK0), 0, 3))
                    tVsV0, tVgV0 = cpasync.tma_partition(tma_atom_V, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3),
                                                         cute.group_modes(thr_mma_pv.partition_B(gV0), 0, 3))
                    # split k of kv head sp_h takes pages k, k + cs, ... with cs = min(pages, half): its
                    # j-th page (j < SPEC) is k + j * half and exists iff that index is < pages
                    sp_np = (sl0 + (PAGE - 1)) // PAGE
                    sp_k = bid // HKV
                    for sp in cutlass.range_constexpr(SPEC):
                        if const_expr(sp > 0 and SPEC_GAP > 0):
                            # let every CTA's first page into the DRAM queues before the second ones
                            nanosleep_ns(SPEC_GAP)
                        pgs = cute.arch.shuffle_sync(pg_spec, sp)
                        if sp_k + sp * (G // HKV) < sp_np:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(bar_kfull + sp, PAGE * D)
                                for kb in cutlass.range_constexpr(2):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        storage.sK.data_ptr() + (sp * KV_STAGE + kb * KV_BOX), kv_tmap.get_ptr(),
                                        [128 * kb, 0, sp_h, pgs], bar_kfull + sp, l2_cache_hint=kv_policy)
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(bar_vfull + sp, PAGE * D)
                                for kb in cutlass.range_constexpr(2):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        storage.sV.data_ptr() + (sp * KV_STAGE + kb * KV_BOX), kv_tmap.get_ptr(),
                                        [D + 128 * kb, 0, sp_h, pgs], bar_vfull + sp, l2_cache_hint=kv_policy)

        if const_expr(not self.bl1):
            # Strided splits start at the first pages of every request: adjacent CTAs prefetch
            # opposite KV heads of one page into L2 early, if that page exists (seq_lens and the
            # block-table entry are loaded together; padded rows prefetch nothing).
            if warp == self.tma_warp:
                pf_b = (bid // HKV) % BL
                pf_p = bid // (BL * HKV)
                if const_expr(self.unsplit):
                    pf_len = mSL[pf_b]
                else:
                    pf_len = cute.arch.shuffle_sync(L_l, pf_b)
                if lane == 0:
                    pf_pg = mBT[pf_b, pf_p]
                    if pf_p < (pf_len + (PAGE - 1)) // PAGE:
                        pf_pg = cutlass.min(cutlass.max(pf_pg, Int32(0)), kv_pages - 1)
                        pf_addr = kv_base + Int64(pf_pg) * (HKV * PAGE * 2 * D) + Int64(bid % HKV) * (PAGE * 2 * D)
                        prefetch_l2_bulk(pf_addr, 65536)
        # Warm L2 with this CTA's slice of the block tables and q while seq_lens is in flight.
        bt_lines = (BL * (MAXP * 4) + 127) // 128
        per_cta = (bt_lines + G - 1) // G
        if tidx < per_cta:
            ln = bid * per_cta + tidx
            if ln < bt_lines:
                prefetch_l2(mBT.iterator.toint() + Int64(ln) * 128)
        q_lines = BL * (Qn * HQ * D // 128)
        per_cta_q = (q_lines + G - 1) // G
        if tidx >= 64 and tidx < 64 + per_cta_q:
            lnq = bid * per_cta_q + (tidx - 64)
            if lnq < q_lines:
                prefetch_l2(mQ.iterator.toint() + Int64(lnq) * 128)
        # The merge counters and the split row statistics are cold after the L2 flush: bring them in
        # now, so the first arrival's atomic and the statistics stores/loads of the merge tail hit L2.
        st_lines = (mPML.shape[0] * mPML.shape[1] * 8 + 127) // 128
        per_cta_s = (st_lines + G - 1) // G
        if tidx >= 128 and tidx < 128 + per_cta_s:
            lns = bid * per_cta_s + (tidx - 128)
            if lns < st_lines:
                prefetch_l2(mPML.iterator.toint() + Int64(lns) * 128)
        cnt_lines = (mCnt.shape[0] * 4 + 127) // 128
        if tidx == 192 and bid < cnt_lines:
            prefetch_l2(mCnt.iterator.toint() + Int64(bid) * 128)

        # ------------------------------------------------------------------ work assignment
        half = G // HKV
        if const_expr(self.unsplit):
            # General-batch fallback: every request/head stream has one CTA.  It never
            # waits on another CTA, so any number of streams can run in successive waves.
            my_b = bid // HKV
            my_h = bid % HKV
            my_k = Int32(0)
            my_cs = Int32(1)
            my_L = mSL[my_b]
            my_np = (my_L + (PAGE - 1)) // PAGE
            if my_L == 0:
                my_b = Int32(-1)
        elif const_expr(self.bl1):
            # BL == 1: adjacent CTAs own the two heads of a page; split k takes pages k, k + cs, ... with
            # cs = min(pages, half), so the first pages (k, k + half) follow from seq_lens[0] directly.
            my_b = Int32(-1)
            if bid < half * HKV:
                my_b = Int32(0)
            my_h = bid % HKV
            my_k = bid // HKV
            my_cs = Int32(0)
            my_np = Int32(0)
            my_L = Int32(0)
        else:
            # Everything below depends only on the device seq_lens (identical in every warp).
            # Lane b < BL handles request b; sums / prefix sums over requests are warp shuffles.
            np_l = (L_l + (PAGE - 1)) // PAGE
            tot = warp_isum(np_l) * HKV
            chunk = cutlass.max((tot + G - 1) // G, Int32(1))
            cs_l = (np_l + chunk - 1) // chunk
            cnt_streams = warp_isum(cs_l) * HKV
            while cnt_streams > G:
                chunk = chunk + 1
                cs_l = (np_l + chunk - 1) // chunk
                cnt_streams = warp_isum(cs_l) * HKV
            # CTA order: request-major, then split, then KV head.
            w_l = cs_l * HKV
            inc_l = w_l
            for i in cutlass.range_constexpr((self.BL - 1).bit_length()):
                up = cute.arch.shuffle_sync_up(inc_l, 1 << i)
                if lane >= (1 << i):
                    inc_l = inc_l + up
            exc_l = inc_l - w_l
            hit = Boolean(False)
            if bid >= exc_l:
                if bid < inc_l:
                    hit = Boolean(True)
            ballot = cute.arch.vote_ballot_sync(hit)
            my_b = Int32(-1)
            my_h = Int32(0)
            my_k = Int32(0)
            my_cs = Int32(0)
            my_np = Int32(0)
            my_L = Int32(0)
            if ballot != 0:
                bsel = Int32(31) - Int32(cute.arch.clz(ballot))
                exc_b = cute.arch.shuffle_sync(exc_l, bsel)
                cs_b = cute.arch.shuffle_sync(cs_l, bsel)
                L_b = cute.arch.shuffle_sync(L_l, bsel)
                off_b = bid - exc_b
                my_b = bsel
                my_h = off_b % HKV
                my_k = off_b // HKV
                my_cs = cs_b
                my_np = (L_b + (PAGE - 1)) // PAGE
                my_L = L_b

        if const_expr(not self.bl1):
            # Graph-padded requests (seq_len 0) get zero rows (cheap: one predicated 16B store per thread),
            # so the whole output is defined and bitwise reproducible.
            req_chunks = Qn * HQ * D * 2 // 16
            for zi in cutlass.range(bid * self.threads + tidx, BL * req_chunks, G * self.threads):
                zb = zi // req_chunks
                if mSL[zb] == 0:
                    zv = cute.make_rmem_tensor(4, Uint32)
                    zv.fill(0)
                    cute.autovec_copy(zv, cute.make_tensor(gptr(Uint32, mOut.iterator.toint() + Int64(zi) * 16),
                                                           cute.make_layout(4)))

        if my_b >= 0:
            # Keep partial storage head-major while issuing adjacent heads together.
            if const_expr(self.bl1):
                bid = my_h * half + my_k
            elif const_expr(not self.unsplit):
                bid = bid - (my_k * HKV + my_h) + my_h * my_cs + my_k
            mK_cur = mK_tma[None, None, my_h, None]
            mV_cur = mV_tma[None, None, my_h, None]
            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None))
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None))
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            tKsK, tKgK = cpasync.tma_partition(tma_atom_K, 0, cute.make_layout(1),
                                               cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
            tVsV, tVgV = cpasync.tma_partition(tma_atom_V, 0, cute.make_layout(1),
                                               cute.group_modes(sV, 0, 3), cute.group_modes(tOgV, 0, 3))

            sBT = cute.make_tensor(cute.recast_ptr(storage.sW.data_ptr(), dtype=Int32), cute.make_layout(W_SIZE))
            if const_expr(self.bt_smem):
                if warp < SM_WARPS:
                    for bt_t in cutlass.range_constexpr(len(bt_first)):
                        bt_e = tidx + bt_t * NUM_SM_THREADS
                        if bt_e < self.BL * BT_PER:
                            sBT[bt_e] = bt_first[bt_t]
            # Copy Q straight into its swizzled MMA tile while TMEM is allocated.
            q_base = mQ.iterator.toint() + Int64(((my_b * Qn) * HQ + my_h * GROUP) * D)
            c16 = tidx % 16                     # this thread's 16-byte column chunk
            QT = 128 * 16 // NUM_SM_THREADS     # rows per thread
            if warp < SM_WARPS:
                for t in cutlass.range_constexpr(QT):
                    rr = tidx // 16 + (NUM_SM_THREADS // 16) * t
                    # TMEM lane rr = 32*qd + 16*gq + 8*lo + i holds stream row i*8 + 2*qd + gq (query i,
                    # head 2*qd + gq of the group) for i < Q; lo = 1 is the residual copy of the same row
                    src_row = Int32(0)
                    src_bytes = Int32(0)
                    if rr % 8 < Qn:
                        src_row = (rr % 8) * GROUP + 2 * (rr // 32) + (rr % 32) // 16
                        src_bytes = Int32(16)
                    src_q = gptr(cutlass.Uint8, q_base + Int64(
                        ((src_row // GROUP) * HQ + src_row % GROUP) * D + c16 * 16))
                    off = (c16 // 8) * 16384 + rr * 128 + (((c16 % 8) ^ (rr % 8)) * 16)
                    prims.cp_async_shared_global(storage.sQ.data_ptr() + off, src_q,
                                                 16, "ca", cp_size=src_bytes)
                prims.cp_async_commit_group()

            if warp == self.mma_warp:
                cute.arch.alloc_tmem(512, storage.tmem_addr.data_ptr(), arch="sm_107")
            cute.arch.sync_threads()
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=storage.tmem_addr.data_ptr())
            if const_expr(self.bl1):
                my_L = sl0
                my_np = (sl0 + (PAGE - 1)) // PAGE
                my_cs = cutlass.min(my_np, half)
            # strided pages: split k takes pages k, k + cs, k + 2 cs, ... (all splits of a stream sweep
            # the KV in lockstep, so concurrently read pages are close together)
            my_act = my_k < my_cs
            n_blk = Int32(0)
            if my_act:
                n_blk = (my_np - my_k + my_cs - 1) // my_cs
            j_first = 0
            if const_expr(self.bl1):
                j_first = SPEC
            tStS_fake = thr_mma_qk.make_fragment_C(thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2]))
            tStS0 = cute.make_tensor(tmem_ptr, tStS_fake.layout)
            tOtO_fake = thr_mma_pv.make_fragment_C(thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2]))
            tOtO = cute.make_tensor(tmem_ptr + 256, tOtO_fake.layout)

            # ================================================================== TMA producer
            if warp == self.tma_warp:
                pg_reg = Int32(0)
                for j in cutlass.range(j_first, n_blk):
                    jr = j - j_first
                    if jr % 32 == 0:
                        if j + lane < n_blk:
                            pg_reg = mBT[my_b, my_k + (j + lane) * my_cs]
                    pg = Int32(0)
                    if const_expr(self.bt_smem):
                        if j == 0:
                            pg = sBT[my_b * BT_PER + my_k]
                        else:
                            pg = cute.arch.shuffle_sync(pg_reg, jr % 32)
                    else:
                        pg = cute.arch.shuffle_sync(pg_reg, jr % 32)
                    if const_expr(STAGE_GAP > 0 and not self.bl1):
                        # stagger the startup burst: every CTA's first page(s) queue ahead of later ones
                        if j == 1:
                            nanosleep_ns(STAGE_GAP)
                    stage = j % NS
                    if j >= NS:
                        # QK completion releases this K stage. Barrier stages follow
                        # K, independently of the double-buffered TMEM score storage.
                        cute.arch.mbarrier_wait(bar_sfull + stage, ((j // NS) - 1) & 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(bar_kfull + stage, PAGE * D)
                        for kb in cutlass.range_constexpr(2):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                storage.sK.data_ptr() + (stage * KV_STAGE + kb * KV_BOX), kv_tmap.get_ptr(),
                                [128 * kb, 0, my_h, pg], bar_kfull + stage, l2_cache_hint=kv_policy)
                    if j >= NS:
                        cute.arch.mbarrier_wait(bar_kvempty + stage, ((j // NS) - 1) & 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(bar_vfull + stage, PAGE * D)
                        for kb in cutlass.range_constexpr(2):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                storage.sV.data_ptr() + (stage * KV_STAGE + kb * KV_BOX), kv_tmap.get_ptr(),
                                [D + 128 * kb, 0, my_h, pg], bar_vfull + stage, l2_cache_hint=kv_policy)

            # ================================================================== MMA issuer
            elif warp == self.mma_warp and my_act:
                tSrQ = tiled_mma_qk.make_fragment_A(sQ)
                tSrK = tiled_mma_qk.make_fragment_B(sK)
                tOrV = tiled_mma_pv.make_fragment_B(sV)
                tP = cute.make_tensor(tmem_ptr, tP_layout.outer)
                tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
                cute.arch.mbarrier_wait(bar_qfull, 0)
                for j in cutlass.range(n_blk + 1):
                    if j < n_blk:
                        stage = j % NS
                        buf = j % 2
                        cute.arch.mbarrier_wait(bar_kfull + stage, (j // NS) & 1)
                        tStS_cur = cute.make_tensor(tStS0.iterator + buf * 128, tStS0.layout)
                        for kb in cutlass.range_constexpr(4):
                            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kb != 0)
                            cute.gemm(tiled_mma_qk, tStS_cur, tSrQ[None, None, kb, 0],
                                      tSrK[None, None, kb, stage], tStS_cur)
                        with cute.arch.elect_one():
                            tcgen05.commit(bar_sfull + stage)
                    if j >= 1:
                        jj = j - 1
                        stage = jj % NS
                        buf = jj % 2
                        cute.arch.mbarrier_wait(bar_vfull + stage, (jj // NS) & 1)
                        cute.arch.mbarrier_wait(bar_pfull + buf, (jj // 2) & 1)
                        tOrP_cur = cute.make_tensor(tOrP.iterator + buf * (128 * 4), tOrP.layout)
                        for kb in cutlass.range_constexpr(2):
                            if const_expr(kb == 0):
                                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, jj > 0)
                            else:
                                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(tiled_mma_pv, tOtO, tOrP_cur[None, None, kb],
                                      tOrV[None, None, kb, stage], tOtO)
                        with cute.arch.elect_one():
                            tcgen05.commit(bar_kvempty + stage)
                            tcgen05.commit(bar_odone)
                            if jj == n_blk - 1:
                                # single-use completion of the last P V: odone's parity alone cannot
                                # tell PV(n-1) from PV(n-3) when softmax warps drift by a page
                                tcgen05.commit(bar_ofinal)
            # ================================================================== softmax / epilogue
            elif warp < SM_WARPS and my_act:
                # warp w: TMEM lane quadrant qd = w % 4 (rows 32*qd + lane), token half hf = w // 4
                qd = warp % 4
                hf = warp // 4
                tsl = tidx % 128                    # TMEM lane == query row slot
                row = tsl
                # lane 16*gq + 8*lo + i: query i of head 2*qd + gq; hi copy (lo = 0) and residual copy (lo = 1)
                # sit 8 lanes apart, so one 16x256b TMEM load hands a thread both halves of an O element
                lane_hi = (lane % 16) < 8
                sel_hi = Uint32(0)
                if lane_hi:
                    sel_hi = Uint32(1)
                row_live = (lane % 8) < Qn
                r = (lane % 8) * GROUP + 2 * qd + lane // 16     # stream row (qi * 8 + g)
                if (lane % 8) >= Qn:
                    r = Int32(0)
                qi = r // GROUP
                g = r % GROUP
                quad_live = Boolean(True)

                # Complete each warp's Q copies before notifying the MMA issuer.
                prims.cp_async_wait_group(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(bar_qfull)

                # ---- TMEM copy atoms (each warp handles 64 S columns / 128 O columns of its rows)
                ld_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(64)), Float32)
                tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
                tSh_layout = cute.composition(tStS0.layout, cute.make_layout((128, 64)))
                thr_ld_s = tcgen05.make_tmem_copy(ld_atom, cute.make_tensor(tmem_ptr, tSh_layout)).get_slice(tsl)
                tSrS = cute.make_rmem_tensor(
                    thr_ld_s.partition_D(cute.composition(tScS, cute.make_layout((128, 64)))).shape, Float32)
                st64_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(64)), Float32)
                thr_st_s = tcgen05.make_tmem_copy(st64_atom, cute.make_tensor(tmem_ptr, tSh_layout)).get_slice(tsl)
                tPh_layout = cute.composition(tStS0.layout, cute.make_layout((128, 16)))
                st16_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32)
                thr_st_p = tcgen05.make_tmem_copy(st16_atom, cute.make_tensor(tmem_ptr, tPh_layout)).get_slice(tsl)
                tSrP = cute.make_rmem_tensor(
                    thr_st_p.partition_S(cute.composition(tScS, cute.make_layout((128, 16)))).shape, Float32)
                tSrP_u32 = cute.make_tensor(cute.recast_ptr(tSrP.iterator, dtype=Uint32), tSrP.layout)
                tOcO = thr_mma_pv.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
                tOe_layout = cute.composition(tOtO.layout, cute.make_layout((128, 64)))
                thr_ld_e = tcgen05.make_tmem_copy(ld_atom, cute.make_tensor(tmem_ptr + 256, tOe_layout)).get_slice(tsl)
                thr_st_e = tcgen05.make_tmem_copy(st64_atom, cute.make_tensor(tmem_ptr + 256, tOe_layout)).get_slice(tsl)
                tErA = cute.make_rmem_tensor(
                    thr_ld_e.partition_D(cute.composition(tOcO, cute.make_layout((128, 64)))).shape, Float32)

                sMx = storage.sMx.get_tensor(cute.make_layout(2 * 2 * 128))
                limit = my_L - Qn + qi            # last key this row may attend
                m_ref = Float32(NEG_INF)
                l_sum = Float32(0.0)
                pair_bar = 2 + qd                 # named barrier shared by warps qd and qd + 4

                if quad_live:
                    for j in cutlass.range(n_blk):
                        buf = j % 2
                        cute.arch.mbarrier_wait(bar_sfull + (j % NS), (j // NS) & 1)
                        tSh_cur = cute.make_tensor(tmem_ptr + buf * 128 + hf * 64, tSh_layout)
                        cute.copy(thr_ld_s, thr_ld_s.partition_S(tSh_cur), tSrS)
                        tok0 = (my_k + j * my_cs) * PAGE + hf * 64
                        if tok0 + 63 > my_L - Qn:
                            for i in cutlass.range_constexpr(64):
                                if tok0 + i > limit:
                                    tSrS[i] = Float32(NEG_INF)
                        # The hi lane and the lo lane of a row (8 lanes apart) split the warp's 64 tokens:
                        # hi takes [0, 32), lo takes [32, 64); each computes both P parts of its tokens.
                        ys = []
                        for gq8 in cutlass.range_constexpr(4):
                            ys.extend(sel8(sel_hi, [tSrS[8 * gq8 + i] for i in range(8)],
                                           [tSrS[32 + 8 * gq8 + i] for i in range(8)], "f"))
                        # max over this thread's 32 tokens, then the pair lane, then the partner warp
                        mx0 = cute.arch.fmax(ys[0], ys[1])
                        mx1 = cute.arch.fmax(ys[2], ys[3])
                        mx2 = cute.arch.fmax(ys[4], ys[5])
                        mx3 = cute.arch.fmax(ys[6], ys[7])
                        for i in cutlass.range_constexpr(1, 4):
                            b8 = 8 * i
                            mx0 = fmax3(mx0, ys[b8 + 0], ys[b8 + 1])
                            mx1 = fmax3(mx1, ys[b8 + 2], ys[b8 + 3])
                            mx2 = fmax3(mx2, ys[b8 + 4], ys[b8 + 5])
                            mx3 = fmax3(mx3, ys[b8 + 6], ys[b8 + 7])
                        mxh = fmax3(fmax3(mx0, mx1, mx2), mx3, mx3)
                        mxh = cute.arch.fmax(mxh, cute.arch.shuffle_sync_bfly(mxh, offset=8))
                        sMx[(buf * 2 + hf) * 128 + tsl] = mxh
                        cute.arch.barrier(barrier_id=pair_bar, number_of_threads=64)
                        mxo = sMx[(buf * 2 + (1 - hf)) * 128 + tsl]
                        m_blk = cute.arch.fmax(mxh, mxo) * scale_log2
                        need = m_blk > m_ref + TAU
                        alpha = Float32(1.0)
                        if need:
                            alpha = cute.arch.exp2(m_ref - m_blk)
                            m_ref = m_blk
                        l_sum = l_sum * alpha
                        need_live = Boolean(False)
                        if row_live:
                            need_live = need
                        any_need = cute.arch.vote_any_sync(need_live)
                        if j > 0 and any_need:
                            # this warp's half of the O rows must be rescaled: wait for P V of block j-1
                            cute.arch.mbarrier_wait(bar_odone, (j - 1) & 1)
                            for cc in cutlass.range(2, unroll=1):
                                tOt_c = cute.make_tensor(tmem_ptr + 256 + hf * 128 + cc * 64, tOe_layout)
                                cute.copy(thr_ld_e, thr_ld_e.partition_S(tOt_c), tErA)
                                for i in cutlass.range_constexpr(64):
                                    tErA[i] = tErA[i] * alpha
                                cute.copy(thr_st_e, tErA, thr_st_e.partition_D(tOt_c))
                            cute.arch.fence_view_async_tmem_store()
                        m_use = m_ref
                        if m_ref == Float32(NEG_INF):
                            m_use = Float32(0.0)
                        neg_off = SCALE_LOG2 - m_use
                        # P_hi = e4m3(x) goes to the hi lane, P_lo = e4m3((x - P_hi) * 16) to the lo lane: each
                        # thread keeps one part of its 32 tokens and swaps the other with its pair lane.  The
                        # partial row sums (32 tokens each) are combined at the end.
                        s0 = Float32(0.0)
                        s1 = Float32(0.0)
                        s2 = Float32(0.0)
                        s3 = Float32(0.0)
                        p_own = []
                        p_recv = []
                        for i in cutlass.range_constexpr(8):
                            x0 = cute.arch.exp2(ys[4 * i] * scale_log2 + neg_off)
                            x1 = cute.arch.exp2(ys[4 * i + 1] * scale_log2 + neg_off)
                            x2 = cute.arch.exp2(ys[4 * i + 2] * scale_log2 + neg_off)
                            x3 = cute.arch.exp2(ys[4 * i + 3] * scale_log2 + neg_off)
                            s0 = s0 + x0
                            s1 = s1 + x1
                            s2 = s2 + x2
                            s3 = s3 + x3
                            w_own, w_send = pack_e4m3x4_own_send(x0, x1, x2, x3, sel_hi)
                            p_own.append(w_own)
                            p_recv.append(Uint32(cute.arch.shuffle_sync_bfly(w_send, offset=8)))
                        l_sum = l_sum + ((s0 + s1) + (s2 + s3))
                        # hi lane: words [own(0-31) | recv(32-63)];  lo lane: [recv(0-31) | own(32-63)]
                        w_lo = sel8(sel_hi, p_own, p_recv, "r")
                        w_hi = sel8(sel_hi, p_recv, p_own, "r")
                        for i in cutlass.range_constexpr(8):
                            tSrP_u32[i] = w_lo[i]
                            tSrP_u32[8 + i] = w_hi[i]
                        tPh_cur = cute.make_tensor(tmem_ptr + buf * 128 + hf * 16, tPh_layout)
                        cute.copy(thr_st_p, tSrP, thr_st_p.partition_D(tPh_cur))
                        cute.arch.fence_view_async_tmem_store()
                        cute.arch.sync_warp()
                        if lane == 0:
                            cute.arch.mbarrier_arrive(bar_pfull + buf)
                    # m_ref / l are final: publish the row statistics now (stores retire during the epilogue)
                    sLp0 = storage.sLp.get_tensor(cute.make_layout(2 * 128))
                    sLp0[hf * 128 + tsl] = l_sum
                    cute.arch.barrier(barrier_id=pair_bar, number_of_threads=64)
                    if hf == 0 and row_live and lane_hi:
                        # this hi lane's 32 tokens + its lo lane's 32 tokens, for both token halves
                        l_tot = (sLp0[tsl] + sLp0[tsl + 8]) + (sLp0[128 + tsl] + sLp0[128 + tsl + 8])
                        if my_cs > 1:
                            # Interleave each slot's max and denominator for one vector transaction.
                            pml_addr = mPML.iterator.toint() + Int64(r * mPO.shape[0] + bid) * 8
                            st_global_f32x2(pml_addr, m_ref, l_tot)
                        else:
                            sLs0 = storage.sW.get_tensor(cute.make_layout(W_SIZE))
                            sLs0[r] = bmm2 / l_tot
                    # all P V done
                    cute.arch.mbarrier_wait(bar_ofinal, 0)
                # ---- epilogue: O = O_hi + O_lo / 16.  Warp (qd, hf) owns O columns [hf*128, +128) of the two
                # 16-lane groups of its quadrant; a 16x256b TMEM load gives thread t the hi lane t/4 and the lo
                # lane t/4 + 8 at the same two columns, so the combine is in registers and goes straight to
                # gmem (fp32 partial, or bf16 `out` for an unsplit stream).
                tm_base = storage.tmem_addr.get_tensor(cute.make_layout(4))[0]
                e_i = lane // 4                                   # query of this thread's group row
                e_live = e_i < Qn
                e_c2 = (lane % 4) * 2
                if my_cs > 1:
                    po_row0 = mPO.iterator.toint() + Int64(bid * R * D) * 4
                    for gq in cutlass.range_constexpr(2):
                        e_r = e_i * GROUP + 2 * qd + gq
                        lane_addr = tm_base + ((32 * qd + 16 * gq) << 16) + 256 + hf * 128
                        for cc in cutlass.range_constexpr(2):
                            ov = tmem_ld_hilo_x8(lane_addr + cc * 64)
                            if e_live:
                                for k in cutlass.range_constexpr(8):
                                    col = hf * 128 + cc * 64 + 8 * k + e_c2
                                    st_global_f32x2(po_row0 + Int64(e_r * D + col) * 4, ov[2 * k], ov[2 * k + 1])
                else:
                    # unsplit stream: bmm2 / l per row comes from the hf = 0 warps through smem
                    cute.arch.barrier(barrier_id=1, number_of_threads=NUM_SM_THREADS)
                    sLs = storage.sW.get_tensor(cute.make_layout(W_SIZE))
                    for gq in cutlass.range_constexpr(2):
                        e_r = e_i * GROUP + 2 * qd + gq
                        sc = sLs[cutlass.min(e_r, R - 1)]
                        o_row0 = mOut.iterator.toint() + Int64(((my_b * Qn + e_i) * HQ + my_h * GROUP + 2 * qd + gq) * D) * 2
                        lane_addr = tm_base + ((32 * qd + 16 * gq) << 16) + 256 + hf * 128
                        for cc in cutlass.range_constexpr(2):
                            ov = tmem_ld_hilo_x8(lane_addr + cc * 64)
                            if e_live:
                                for k in cutlass.range_constexpr(8):
                                    col = hf * 128 + cc * 64 + 8 * k + e_c2
                                    st_global_bf16x2(o_row0 + Int64(col) * 2, ov[2 * k] * sc, ov[2 * k + 1] * sc)
            # ------------------------------------------------------------------ teardown TMEM
            cute.arch.sync_threads()
            if warp == self.mma_warp:
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(tmem_ptr, 512, arch="sm_107")
            # The teardown barrier also carries all partial stores to the
            # releasing arrival. TMEM deallocation overlaps publish and merge.
            if my_cs > 1 and my_act and tidx == 0:
                arrival_ptr = mCnt.iterator + (my_b * HKV + my_h)
                publish_arrival(arrival_ptr.toint(), my_cs - 1)

            # ------------------------------------------------------------------ split merge
            merge_live = Boolean(True)
            if const_expr(Qn == 1):
                if my_cs > 32:
                    merge_live = ((my_k + 1) * (R * 4)) // my_cs > (my_k * (R * 4)) // my_cs
            if my_cs > 1 and my_act and warp < MW and merge_live:
                s_idx = my_b * HKV + my_h
                f = bid - my_k                      # first CTA of this stream
                cnt_ptr = mCnt.iterator + s_idx   # last arrival releases and resets the stream
                G4 = R * 64
                g0 = (my_k * G4) // my_cs
                g1 = ((my_k + 1) * G4) // my_cs
                if my_cs > 32:
                    # Assign whole 64-float TMA columns (no overlapping box fetches). Q=1 owners with
                    # empty intervals have already arrived and skip merging.
                    g0 = 16 * ((my_k * (R * 4)) // my_cs)
                    g1 = 16 * (((my_k + 1) * (R * 4)) // my_cs)
                e0 = g0 * 4
                ne = (g1 - g0) * 4
                nbytes = ne * 4
                r0 = g0 // 64
                r1 = (g1 - 1) // 64
                stage_k = storage.sK.data_ptr().toint()
                ncb = (ne + 63) // 64
                nrb = (my_cs + 31) // 32
                use_box = my_cs > BOX_MIN_CS and ncb * nrb <= 12
                fetch_bytes = my_cs * nbytes
                if use_box:
                    fetch_bytes = ncb * nrb * (64 * 32 * 4)
                mM_cur = cute.domain_offset((e0, f), mPM_tma)
                gM = cute.local_tile(mM_cur, (64, 32), (None, None))
                sMb = cute.make_tensor(sptr(Float32, stage_k, align=1024),
                                       cute.make_layout((64, 32, 12), stride=(1, 64, 2048)))
                tMsM, tMgM = cpasync.tma_partition(tma_atom_M, 0, cute.make_layout(1),
                                                   cute.group_modes(sMb, 0, 2), cute.group_modes(gM, 0, 2))
                # The last slot box is aligned backward to end at the stream's last partial (cs > 32 whenever
                # boxes are used), so no box reads another stream's or an unused (cold) slot.  It lands at
                # its partials' own staging rows; the overlap with the previous box carries identical data.
                l_off = cutlass.max(my_cs - 32, 0)
                mM_last = cute.domain_offset((e0, f + l_off), mPM_tma)
                gM_last = cute.local_tile(mM_last, (64, 32), (None, None))
                sMl = cute.make_tensor(sptr(Float32, stage_k + l_off * 256, align=256),
                                       cute.make_layout((64, 32, 12), stride=(1, 64, 2048)))
                tMsL, tMgL = cpasync.tma_partition(tma_atom_M, 0, cute.make_layout(1),
                                                   cute.group_modes(sMl, 0, 2), cute.group_modes(gM_last, 0, 2))
                if tidx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_merge, fetch_bytes)
                    # acquire polls: the load that sees the last arrival orders the partial reads after it
                    seen = cute.arch.load(cnt_ptr, Int32, sem="acquire", scope="gpu")
                    while seen != 0 and seen < my_cs:
                        seen = cute.arch.load(cnt_ptr, Int32, sem="acquire", scope="gpu")
                cute.arch.barrier(barrier_id=6, number_of_threads=MW * 32)
                cute.arch.fence_proxy("async.global")
                # fetch the cs partial slices [e0, e0+ne) into smem with TMA:
                #   cs <= 16 : one bulk copy per partial  -> stage[k][ne]
                #   cs >  16 : 2-D boxes of 64 floats x 32 partials -> box[cb][rb][32][64]
                if use_box:
                    if warp == 0:
                        for cb in cutlass.range(ncb):
                            for rb in cutlass.range(nrb - 1):
                                cute.copy(tma_atom_M, tMgM[None, cb, rb], tMsM[None, cb * nrb + rb],
                                          tma_bar_ptr=bar_merge)
                            cute.copy(tma_atom_M, tMgL[None, cb, 0], tMsL[None, cb * nrb],
                                      tma_bar_ptr=bar_merge)
                else:
                    po_base = mPO.iterator.toint()
                    for kk in cutlass.range(tidx, my_cs, MW * 32):
                        bulk_g2s(po_base + (Int64(f + kk) * (R * D) + e0) * 4,
                                 stage_k + kk * nbytes, bar_merge.toint(), nbytes)
                # weights W[row][k] = bmm2 * exp2(m_k - M) / L  (all loads issued before first use)
                sW = storage.sW.get_tensor(cute.make_layout(W_SIZE))
                wstr = (my_cs + 3) // 4 * 4             # weight row stride: 16B-aligned rows for vector loads
                if my_cs <= 32:
                    # lane = partial, warp w handles rows r0 + w + MW*i  (a CTA covers < 32 rows)
                    kk = cutlass.min(lane, my_cs - 1)
                    if r0 + warp <= r1:
                        mk = cute.make_rmem_tensor(32 // MW, Float32)
                        lk = cute.make_rmem_tensor(32 // MW, Float32)
                        for i in cutlass.range_constexpr(32 // MW):
                            rr = cutlass.min(r0 + warp + MW * i, r1)
                            mv, lv = ld_global_f32x2(mPML.iterator.toint() +
                                                     Int64(rr * mPO.shape[0] + f + kk) * 8)
                            mk[i] = mv
                            lk[i] = lv
                        for i in cutlass.range_constexpr(32 // MW):
                            rr = r0 + warp + MW * i
                            if rr <= r1:
                                mv = mk[i]
                                lv = lk[i]
                                if lane >= my_cs:
                                    mv = Float32(NEG_INF)
                                    lv = Float32(0.0)
                                M = warp_max(mv)
                                wv = cute.arch.exp2(mv - M)
                                Ls = warp_sum(wv * lv)
                                if lane < my_cs:
                                    sW[(rr - r0) * wstr + lane] = wv * (bmm2 / Ls)
                else:
                    # few rows (<= 3), lanes split the partials
                    rr = r0 + warp
                    if rr <= r1:
                        mk = cute.make_rmem_tensor(MAX_GRID // HKV // 32, Float32)
                        lk = cute.make_rmem_tensor(MAX_GRID // HKV // 32, Float32)
                        for i in cutlass.range_constexpr(MAX_GRID // HKV // 32):
                            kk = cutlass.min(lane + 32 * i, my_cs - 1)
                            mv, lv = ld_global_f32x2(mPML.iterator.toint() +
                                                     Int64(rr * mPO.shape[0] + f + kk) * 8)
                            mk[i] = mv
                            lk[i] = lv
                        mloc = Float32(NEG_INF)
                        for i in cutlass.range_constexpr(MAX_GRID // HKV // 32):
                            if lane + 32 * i >= my_cs:
                                mk[i] = Float32(NEG_INF)
                                lk[i] = Float32(0.0)
                            mloc = cute.arch.fmax(mloc, mk[i])
                        M = warp_max(mloc)
                        wl = Float32(0.0)
                        for i in cutlass.range_constexpr(MAX_GRID // HKV // 32):
                            mk[i] = cute.arch.exp2(mk[i] - M)
                            wl = wl + mk[i] * lk[i]
                        Ls = warp_sum(wl)
                        inv = bmm2 / Ls
                        for i in cutlass.range_constexpr(MAX_GRID // HKV // 32):
                            kk = lane + 32 * i
                            if kk < my_cs:
                                sW[(rr - r0) * wstr + kk] = mk[i] * inv
                cute.arch.barrier(barrier_id=6, number_of_threads=MW * 32)
                cute.arch.mbarrier_wait(bar_merge, 0)
                stage_t = cute.make_tensor(sptr(Float32, stage_k), cute.make_layout(64 * 1024))
                hb = my_h * GROUP
                # Vector merge (slices of <= 256 float4): lane = float4 of the slice, warp w = contiguous
                # block of partials (one 16B shared load per partial); the MW warp sums meet in the idle V
                # ring and are added in warp order (deterministic).
                f_nv = ne // 4
                f_ok = f_nv <= 256
                f_ch = (my_cs + MW - 1) // MW
                f_lo = cutlass.min(warp * f_ch, my_cs)
                f_hi = cutlass.min(f_lo + f_ch, my_cs)
                f_red = storage.sV.data_ptr().toint()
                if f_ok:
                    for vg in cutlass.range(0, f_nv, 32):
                        fv = cutlass.min(vg + lane, f_nv - 1)
                        fe = fv * 4
                        fwrow = (((e0 + fe) // D) - r0) * wstr
                        festep = ne
                        febase = fe
                        if use_box:
                            festep = 64
                            febase = (fe // 64) * (nrb * 2048) + (fe % 64)
                        f_acc = cute.make_rmem_tensor(4, Float32)
                        f_acc.fill(0.0)
                        f_x4 = cute.make_rmem_tensor(4, Float32)
                        for kk in cutlass.range(f_lo, f_hi, unroll=4):
                            fw = sW[fwrow + kk]
                            cute.autovec_copy(cute.make_tensor(sptr(Float32, stage_k + (febase + kk * festep) * 4),
                                                               cute.make_layout(4)), f_x4)
                            fv4 = weighted_fma4(fw, f_x4[0], f_x4[1], f_x4[2], f_x4[3],
                                               f_acc[0], f_acc[1], f_acc[2], f_acc[3])
                            for c in cutlass.range_constexpr(4):
                                f_acc[c] = fv4[c]
                        if vg + lane < f_nv:
                            cute.autovec_copy(f_acc, cute.make_tensor(
                                sptr(Float32, f_red + (warp * f_nv + fv) * 16), cute.make_layout(4)))
                    cute.arch.barrier(barrier_id=6, number_of_threads=MW * 32)
                    for fv2 in cutlass.range(tidx, f_nv, MW * 32):
                        f_tot = cute.make_rmem_tensor(4, Float32)
                        f_tot.fill(0.0)
                        f_y4 = cute.make_rmem_tensor(4, Float32)
                        for w2 in cutlass.range_constexpr(MW):
                            cute.autovec_copy(cute.make_tensor(sptr(Float32, f_red + (w2 * f_nv + fv2) * 16),
                                                               cute.make_layout(4)), f_y4)
                            f_sum4 = add_f32x4(f_tot[0], f_tot[1], f_tot[2], f_tot[3],
                                               f_y4[0], f_y4[1], f_y4[2], f_y4[3])
                            for c in cutlass.range_constexpr(4):
                                f_tot[c] = f_sum4[c]
                        f_ob = cute.make_rmem_tensor(4, cutlass.BFloat16)
                        for c in cutlass.range_constexpr(4):
                            f_ob[c] = f_tot[c].to(cutlass.BFloat16)
                        fge2 = e0 + fv2 * 4
                        frr2 = fge2 // D
                        fo_off = Int64(((my_b * Qn + frr2 // GROUP) * HQ + hb + frr2 % GROUP) * D + fge2 % D)
                        cute.autovec_copy(f_ob, cute.make_tensor(
                            gptr(cutlass.BFloat16, mOut.iterator.toint() + fo_off * 2, align=8), cute.make_layout(4)))
                ne_s = ne
                if f_ok:
                    ne_s = Int32(0)
                for e in cutlass.range(tidx, ne_s, MW * 32):
                    ge = e0 + e
                    rr = ge // D
                    col = ge % D
                    wrow = (rr - r0) * wstr
                    # box layout: partial kk of element e sits at ebase + kk*64 (2048 = 32*64 per box)
                    estep = ne
                    ebase = e
                    if use_box:
                        estep = 64
                        ebase = (e // 64) * (nrb * 2048) + (e % 64)
                    a0 = Float32(0.0)
                    a1 = Float32(0.0)
                    a2 = Float32(0.0)
                    a3 = Float32(0.0)
                    if my_cs >= 16:
                        sw_base = storage.sW.data_ptr().toint()
                        for k4 in cutlass.range(my_cs // 4, unroll=2):
                            kk = k4 * 4
                            w4 = cute.make_rmem_tensor(4, Float32)
                            cute.autovec_copy(cute.make_tensor(sptr(Float32, sw_base + (wrow + kk) * 4),
                                                               cute.make_layout(4)), w4)
                            a0 = a0 + w4[0] * stage_t[ebase + kk * estep]
                            a1 = a1 + w4[1] * stage_t[ebase + (kk + 1) * estep]
                            a2 = a2 + w4[2] * stage_t[ebase + (kk + 2) * estep]
                            a3 = a3 + w4[3] * stage_t[ebase + (kk + 3) * estep]
                        for kr in cutlass.range(my_cs - my_cs % 4, my_cs):
                            a0 = a0 + sW[wrow + kr] * stage_t[ebase + kr * estep]
                    else:
                        for k2 in cutlass.range(my_cs // 2, unroll=2):
                            kk = k2 * 2
                            a0 = a0 + sW[wrow + kk] * stage_t[ebase + kk * estep]
                            a1 = a1 + sW[wrow + kk + 1] * stage_t[ebase + (kk + 1) * estep]
                        if my_cs % 2 == 1:
                            a0 = a0 + sW[wrow + my_cs - 1] * stage_t[ebase + (my_cs - 1) * estep]
                    a = (a0 + a1) + (a2 + a3)
                    o_off = ((my_b * Qn + rr // GROUP) * HQ + hb + rr % GROUP) * D + col
                    mOut[o_off] = a.to(cutlass.BFloat16)
# ----------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------
_COMPILED = {}


def _compile(Q: int, BL: int, grid: int, bl1: bool, unsplit: bool = False):
    key = (Q, BL, grid, bl1, unsplit)
    if key in _COMPILED:
        return _COMPILED[key]
    R = Q * GROUP
    bl = BL
    npg = cute.sym_int32()
    gmax = 1 if unsplit else grid
    q_f = make_fake_compact_tensor(cutlass.Uint8, (bl, Q, HQ, D), stride_order=(3, 2, 1, 0), assumed_align=16)
    kv_f = make_fake_compact_tensor(cutlass.Uint8, (npg, HKV, PAGE, 2 * D), stride_order=(3, 2, 1, 0),
                                    assumed_align=16)
    bt_f = make_fake_compact_tensor(Int32, (bl, MAXP), stride_order=(1, 0), assumed_align=4)
    sl_f = make_fake_compact_tensor(Int32, (bl,), assumed_align=4)
    out_f = make_fake_compact_tensor(cutlass.BFloat16, (bl * Q * HQ * D,), assumed_align=16)
    po_f = make_fake_compact_tensor(Float32, (gmax, R, D), stride_order=(2, 1, 0), assumed_align=16)
    pml_f = make_fake_compact_tensor(Float32, (R, gmax, 2), stride_order=(2, 1, 0), assumed_align=16)
    ncnt = 1 if unsplit else BL * HKV
    cnt_f = make_fake_compact_tensor(Int32, (ncnt,), assumed_align=16)
    k = PagedDecodeAttn(Q, BL, grid, bl1, unsplit)
    fn = cute.compile(k, q_f, kv_f, bt_f, sl_f, out_f, po_f, pml_f, cnt_f, Float32(0.0625), Float32(1.0),
                      Int32(1), Int64(EVICT_NORMAL), make_fake_stream(use_tvm_ffi_env_stream=True),
                      options="--enable-tvm-ffi")
    _COMPILED[key] = fn
    return fn


def setup(context):
    axes = context["axes"]
    Q = int(axes["Q"])
    BL = int(axes["BL"])
    dev = torch.device(context["device"])
    n_sm = torch.cuda.get_device_properties(dev).multi_processor_count
    grid = min(n_sm, MAX_GRID)
    # The warp scheduler covers at most 32 requests, and its minimum allocation
    # of two CTAs per live request must fit simultaneously.  Other capture sizes
    # use independent unsplit streams, with bounded scratch regardless of BL.
    unsplit = BL > 32 or BL * HKV > grid
    if unsplit:
        grid = BL * HKV
    slots = 1 if unsplit else grid
    ncnt = 1 if unsplit else BL * HKV
    R = Q * GROUP
    fn = _compile(Q, BL, grid, BL == 1 and not unsplit, unsplit)
    po = torch.empty((slots, R, D), dtype=torch.float32, device=dev)
    pml = torch.empty((R, slots, 2), dtype=torch.float32, device=dev)
    cnt = torch.zeros((ncnt,), dtype=torch.int32, device=dev)
    # L2 policy for the streamed K/V: evict-first keeps the one-pass KV stream from displacing (and
    # writing back) other L2 lines; measured faster for one and two requests alike.
    kv_policy = EVICT_FIRST
    return {"fn": fn, "grid": grid, "po": po, "pml": pml, "cnt": cnt, "kv_policy": kv_policy}


def run(q, kv_cache, block_tables, seq_lens, bmm1_scale, bmm2_scale, state, out):
    st = state
    st["fn"](q.view(torch.uint8), kv_cache.view(torch.uint8), block_tables, seq_lens, out.view(-1),
             st["po"], st["pml"], st["cnt"], float(bmm1_scale), float(bmm2_scale), st["grid"], st["kv_policy"])
