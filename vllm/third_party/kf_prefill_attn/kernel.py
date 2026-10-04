# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# mypy: ignore-errors
# fmt: off
# Kernel Factory solution, verbatim below this header (do not edit; swap the whole file).
#   campaign  cm1s2rr9s16sf16knj2mbhshrc (phaseA-cutedsl-0103k), artifact paged-fp8-prefill-attn-h16kv2d256-p128
#   candidate 511e91b9dfb482911baf062588595be45efafd7cecab9a51f54ec68a53395501
#   solution paged_fp8_prefill_cpasync_merge, definition paged_fp8_prefill_attn_h16kv2d256_p128 (sha256 d710c49f...)
#   source sha256 85d85ed593c29ba15137e60a511ca0fda3fe60d61a092e3cc5eab7922820589f (bytes after this header)
# Retrieve: kf campaign results cm1s2rr9s16sf16knj2mbhshrc --output-dir <dir>  (or kf campaign kernel show <id> --source kernel.py)
# The host planner of this file (setup) is reimplemented exactly in C++ by planner.py; runtime.py uses
# only PagedFp8Attn/_OPS/constants from here.
# Paged-KV FP8 causal chunked-prefill attention for Rubin (sm_107a), CuTe DSL.
#
#   16 q heads / 2 kv heads (GQA 8), head_dim 256, FP8 e4m3 Q and paged FP8 KV
#   (page 128, K|V packed per token), BF16 out, varlen batch of B requests.
#
# One persistent warp-specialized kernel.  A work item is
# (request, 16-token q-tile x 8 grouped heads = 128 MMA rows, kv head, kv split).
# The host builds a length-balanced work list in setup(); split partials are
# merged inside the same launch by the last-arriving CTA of the group through a
# self-resetting atom.inc counter.
import heapq
import math


import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.cute.nvgpu import OperandMajorMode
import cutlass.utils as utils
import cutlass.utils.rubin_helpers as rubin_utils
import cutlass.utils.blackwell_helpers as sm1xx_utils
from cutlass.cute.runtime import from_dlpack
from cutlass import Float32, Int32, Int64, Boolean, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

HQ = 16
HKV = 2
GROUP = 8
HDIM = 256
PAGE = 128
MAXP = 2057

NTILE = 128          # keys per KV tile == page size
MROW = 128           # MMA rows per CTA (16 tokens x 8 grouped heads)
KVSTAGE_MAX = 6      # mbarrier/offset budget; a specialization may use fewer
NTHREAD = 576          # 16 compute warps + one MMA warp + one TMA warp
WARP_MMA = 16
WARP_LOAD = 17
NCOMPW = 16         # compute warps = 4 column segments x 4 lane groups
NSEG = 4
HALFN = NTILE // NSEG
PLAN_F = 12          # int32 fields per work item
CHUNK = 32           # columns of O handled per TMEM round trip
# The lazy-anchor O rescale is the only place a compute thread holds a second
# live fragment.  Halving its width frees the registers that keeping the packed
# probabilities alive past the P release costs, so the row sum can move off the
# chain that gates the next PV MMA without spilling.
CORRCH = 16
CORRCH_LEAN = 4      # preserve the current champion's cold rescale fragment
SUMN = 8              # N of the row-sum UMMA (P x ones) that replaces the
# per-tile register row sum.  The compute warps spend 16 packed adds per thread
# per KV tile summing the probabilities; the tensor pipe does the same reduction
# as a rank-8 UMMA costing ~1.5% of the tile's MMA work, and the accumulator
# then lives in tmem next to O, so the per-item cross-segment shared-memory
# reduction disappears as well.
NMRG = 4             # column bands a split group's partials are cut into
MCOL = HDIM // NMRG  # output columns one band covers
# A group is reduced by ntask in {1,2,4} CTAs, each owning NMRG//ntask adjacent
# bands.  A merge task costs far more than the bytes it moves -- it polls the
# group's arrival counter, rendezvouses all 16 compute warps, then walks the
# per-split row max and sum -- so cutting a group four ways multiplies that
# fixed cost by four.  Measured at eight ways it costs 4-6.5% on launches with
# a few hundred groups.  The host therefore spends extra tasks only on the
# groups where they shorten the tail rather than lengthen the grid.
MRG_F = 8            # int32 fields per merge task
BIGLIM = 1 << 30     # atom.inc limit used when no wrap is wanted
POV = 4              # head-dim columns of one partial-O row kept contiguous
# The split merge streams its partial bands through the CTA's Q/K/V buffers,
# which sit idle once the streaming rows are done.  One stage holds one split's
# band for every merge thread (64 B of O + the row max/sum), so the read-back
# keeps MRG_STAGES splits in flight instead of the two a register prefetch fits.
MRG_STAGES_MAX = 4
# A split piece writes 128 KiB of partial O and the merging CTA reads one
# such block per split.  With the natural (row, column) partial layout the
# merge issues one 4-byte load per column, so a single CTA's outstanding
# load budget covers only 4 B per request.  Grouping four head-dim columns
# of a row makes every access a 16-byte vector: same bytes, a quarter of
# the requests, four times the bytes in flight per CTA.
LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
P_PRESCALE = 6.0     # conservative path: P stored as exp2(x + 6)
CORR_TAU = 2.75      # conservative path keeps headroom for a lazy max update
P_MIXED_PRESCALE = 7.5  # expanded P for large mixed-request launches
P_MIXED_TAU = 1.25
P_LONG_PRESCALE = 7.0   # more lazy headroom for very long per-request prefill
P_LONG_TAU = 1.75       # each prescale + tau = 8.75, below e4m3 max
PCOL_T = NTILE // 32 * 8   # tmem columns taken by one fp8 P tile
NEG_BIG = -1.0e30
ROLES = 7
LVL = 15

MB_Q = 0
MB_QE = 1
MB_K = 2
MB_V = MB_K + KVSTAGE_MAX
MB_KE = MB_V + KVSTAGE_MAX
MB_VE = MB_KE + KVSTAGE_MAX
MB_S = MB_VE + KVSTAGE_MAX
MB_P = MB_S + 2
MB_PV = MB_P + 2
MB_EPI = MB_PV + 2
MB_QT = MB_EPI + 1     # qtm: Q half written to tmem by the compute warps
MB_N = MB_QT + 1       # mbarriers actually used; sizes the smem array
# qtm: the first QTM_KB K64 blocks of Q*K^T read Q from tmem instead of smem.
# The packed two-term P forms take a third block (QTM_KB_PACKED): their P and
# residual terms use only the first 64 columns of an S/P slot, so the slots can
# shift up by 16 more columns and still keep every A operand below column 512.
QTM = True
QTM_KB = 2
QTM_KB_PACKED = 3


@dsl_user_op
def _atom_inc(addr: Int64, limit: Int32, *, loc=None, ip=None) -> Int32:
    """old = atomicInc(*addr, limit); wraps back to 0 at old == limit."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(limit).ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.inc.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _pack2(a: Float32, b: Float32, *, loc=None, ip=None) -> Int64:
    """Pack two fp32 lanes into one f32x2 operand."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mov.b64 $0, {$1, $2};",
            "=l,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _sm_pair(s2: Int64, sc2: Int64, sh2: Int64,
             *, loc=None, ip=None) -> Int32:
    """Transform two scores with Rubin's packed fp16x2 exp2 path.

    The fp32 -> fp16 step rounds to nearest instead of toward zero.  RZ drags
    every exponent argument down by up to one fp16 ulp, and because that bias
    is systematic it does not average out along a row the way the e4m3 P noise
    does -- it just eats error budget the residual tiles have to buy back."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(s2).ir_value(loc=loc, ip=ip), Int64(sc2).ir_value(loc=loc, ip=ip),
             Int64(sh2).ir_value(loc=loc, ip=ip)],
            "{ .reg .b64 u;\n\t"
            ".reg .f32 ra, rb;\n\t"
            ".reg .b32 t;\n\t"
            "fma.rn.f32x2 u, $1, $2, $3;\n\t"
            "mov.b64 {ra, rb}, u;\n\t"
            "cvt.rn.f16x2.f32 t, rb, ra;\n\t"
            "ex2.approx.f16x2 $0, t; }",
            "=r,l,l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _affine_pair(s2: Int64, sc2: Int64, sh2: Int64, *, loc=None, ip=None) -> Int64:
    """Apply the FP32 softmax scale and shift to two lanes in one instruction."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Int64(s2).ir_value(loc=loc, ip=ip), Int64(sc2).ir_value(loc=loc, ip=ip),
             Int64(sh2).ir_value(loc=loc, ip=ip)],
            "fma.rn.f32x2 $0, $1, $2, $3;",
            "=l,l,l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _acc_f16x2(acc: Int64, h: Int32, *, loc=None, ip=None) -> Int64:
    """Promote two fp16 probabilities and accumulate them into fp32x2."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Int64(acc).ir_value(loc=loc, ip=ip), Int32(h).ir_value(loc=loc, ip=ip)],
            "add.rn.f32x2.f16x2.f32x2 $0, $2, $1;",
            "=l,l,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _hadd_f32x2(acc: Int64, *, loc=None, ip=None) -> Float32:
    """Horizontally sum an fp32x2 accumulator."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(acc).ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 ra, rb;\n\t"
            "mov.b64 {ra, rb}, $1;\n\t"
            "add.f32 $0, ra, rb; }",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _mul_f32x2(a: Int64, b: Int64, *, loc=None, ip=None) -> Int64:
    """Rescale two packed fp32 row-sum lanes in one instruction."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Int64(a).ir_value(loc=loc, ip=ip), Int64(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32x2 $0, $1, $2;",
            "=l,l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _add_f32x2(a: Int64, b: Int64, *, loc=None, ip=None) -> Int64:
    """Add two packed fp32 row-sum lanes in one instruction."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Int64(a).ir_value(loc=loc, ip=ip), Int64(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32x2 $0, $1, $2;",
            "=l,l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cvt_e4m3x4(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """Convert four packed fp16 probabilities to four e4m3 bytes."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 pl, ph;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 pl, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 ph, $2;\n\t"
            "mov.b32 $0, {pl, ph}; }",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _resid_e4m3x4(a: Int32, b: Int32, p: Int32, *, loc=None, ip=None) -> Int32:
    """Quantize four packed residual probabilities to e4m3."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip),
             Int32(p).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 ql, qh;\n\t"
            ".reg .b32 fl, fh, dl, dh;\n\t"
            "mov.b32 {ql, qh}, $3;\n\t"
            "cvt.rn.f16x2.e4m3x2 fl, ql;\n\t"
            "cvt.rn.f16x2.e4m3x2 fh, qh;\n\t"
            "sub.f16x2 dl, $1, fl;\n\t"
            "sub.f16x2 dh, $2, fh;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 ql, dl;\n\t"
            "cvt.rn.satfinite.e4m3x2.f16x2 qh, dh;\n\t"
            "mov.b32 $0, {ql, qh}; }",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _add_h2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """Packed fp16 add: one 32-bit ALU op where the mixed-precision f32x2
    accumulate needs a 64-bit one."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "add.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _tree_sum_h2(hp, n):
    """Balanced fp16 tree over n packed pairs; returns one f16x2 partial.

    Every probability is at or below 2^8.75 by construction of the prescale, so
    a 32-wide half sums to at most 1.4e4 -- an order of magnitude inside the fp16
    range -- and the five rounding levels contribute ~5e-4 relative, which the
    fp32 cross-tile accumulation then averages down."""
    cur = [_add_h2(hp[2 * i], hp[2 * i + 1]) for i in range(n // 2)]
    while len(cur) > 1:
        cur = [_add_h2(cur[2 * i], cur[2 * i + 1]) for i in range(len(cur) // 2)]
    return cur[0]


@dsl_user_op
def _scale_cvt_bf16x2(ab: Int64, s2: Int64, *, loc=None, ip=None) -> Int32:
    """Scale two fp32 outputs and pack them as bf16 in two instructions."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(ab).ir_value(loc=loc, ip=ip), Int64(s2).ir_value(loc=loc, ip=ip)],
            "{ .reg .b64 u;\n\t"
            ".reg .f32 x, y;\n\t"
            "mul.rn.f32x2 u, $1, $2;\n\t"
            "mov.b64 {x, y}, u;\n\t"
            "cvt.rn.bf16x2.f32 $0, y, x; }",
            "=r,l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _bar_red_gt(a: Float32, b: Float32, *, loc=None, ip=None) -> Int32:
    """Barrier over the compute warps that also counts the threads with a > b.

    The running row maximum only has to move when some lane sees a score beyond
    the e4m3 head-room, which after the first tile of an item is essentially
    never.  One CTA-wide popc reduction decides that for the whole tile, so the
    common path drops the shared-memory round trip entirely while keeping the
    very same barrier that orders the score reads against the packed-P writes
    that alias them in tmem."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "{ .reg .pred q;\n\t"
            "setp.gt.f32 q, $1, $2;\n\t"
            "barrier.red.popc.u32 $0, 1, " + str(32 * NCOMPW) + ", q; }",
            "=r,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _fence_gpu(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None, [], "fence.acq_rel.gpu;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_acq(addr: Int64, *, loc=None, ip=None) -> Int32:
    """Acquiring 32-bit load used to poll a split group's arrival counter."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _st_rel(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
    """Releasing 32-bit store that returns a counter to its initial state."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.release.gpu.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _cp_async16(saddr: Int32, gaddr: Int64, *, loc=None, ip=None) -> None:
    """16-byte global->shared async copy that bypasses L1 (split partials are
    written by other SMs and live in L2)."""
    llvm.inline_asm(
        None,
        [Int32(saddr).ir_value(loc=loc, ip=ip), Int64(gaddr).ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _cp_async4(saddr: Int32, gaddr: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(saddr).ir_value(loc=loc, ip=ip), Int64(gaddr).ir_value(loc=loc, ip=ip)],
        "cp.async.ca.shared.global [$0], [$1], 4;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class PagedFp8Attn:
    def __init__(self, resid_period, use_fused_max=False, use_packed=False,
                 use_item_mask=False, kvstage=3, po_bf16=False, mcast=False,
                 p_prescale=P_PRESCALE, corr_tau=CORR_TAU, twocta=False,
                 row_pred=False, tsum=False, skew_load=False, deep=False,
                 qtm=False):
        self.resid_period = resid_period
        # deep: run Q*K^T two tiles ahead of P*V (see mma_loop_deep).  Measured
        # 1.0-1.6% faster on launches whose CTA pairs stream >= 4000 KV tiles
        # and 5-14% slower below that, so the host gates it by stream length.
        self.deep = deep
        self.tsum = tsum
        # mcast: two CTAs of a cluster take the 16-token halves of one 32-token
        # q-tile and TMA-multicast the shared K/V pages.  Each CTA then pulls
        # only half the page bytes across the L2 network while still filling its
        # own 128-row MMA tile, which doubles the rows served per KV byte.
        self.mcast = mcast
        self.kvstage = kvstage
        self.po_bf16 = po_bf16
        self.use_fused_max = use_fused_max
        self.use_packed = use_packed
        self.use_item_mask = use_item_mask
        self.p_prescale = p_prescale
        self.corr_tau = corr_tau
        self.row_pred = row_pred
        # The lazy-anchor O rescale is cold, but ptxas still reserves its
        # fragment in the loop that owns it.  Halve it exactly where the
        # softmax working set is also trimmed below.
        self.corrch = CORRCH_LEAN if row_pred else CORRCH
        self.skew_load = skew_load
        # twocta: one tcgen05 MMA spans the CTA pair (M = 256 rows).  Each CTA
        # then TMA-loads only its N-half of K and of V, so the per-SM K|V fill
        # per 128-key tile drops from 64 KiB to 32 KiB while the MMA work per SM
        # is unchanged.  That is the measured wall on the long-query launches:
        # ~67 B/clk/SM of fill against a ~73 B/clk/SM ceiling, with the tensor
        # pipe only ~73% busy.
        self.twocta = twocta
        self.clustered = mcast or twocta
        self.cta_group = (tcgen05.CtaGroup.TWO if twocta
                          else tcgen05.CtaGroup.ONE)
        self.mma_m = 2 * MROW if twocta else MROW
        self.atom_sm = 2 if twocta else 1
        # barriers the pair-leader has to broadcast to both CTAs
        self.pair_mask = 3 if twocta else None
        self.qk_mma_tiler = (self.mma_m, NTILE, HDIM)
        self.pv_mma_tiler = (self.mma_m, HDIM, NTILE)
        self.kv_rel_mask = 3 if self.clustered else None
        self.tmem_o = 0
        self.tmem_s0 = 256
        self.tmem_s1 = 384
        self.tmem_cols = 512
        # qtm: on the collective pipeline Q*K^T streams its A operand (Q,
        # 32 KiB/CTA) from smem on every KV tile, next to 16 KiB of K, 16 KiB
        # of V and 32 KiB of TMA fill, which leaves the tile on the smem port
        # rather than on the FP8 tensor rate.  Keep the first half of Q's head
        # dim in tmem.  A-operand tmem columns must sit below 512, so Q takes
        # 256..287 and the S/P slots shift up by 32; S1's last 32 fp32 score
        # columns (512..543) and the row sum (544..) live in the second
        # 64-column allocation.
        self.qtm = qtm and twocta
        self.tmem_cols = 512
        self.set_qtm_kb(QTM_KB_PACKED if use_packed else QTM_KB)
        # A CtaGroup.TWO UMMA splits B in N across the pair, so the collective
        # tiler needs 2 x the smallest legal per-CTA operand width.
        # A CtaGroup.TWO UMMA requires 16 <= N <= 256 with N % 16 == 0; the
        # one-CTA form accepts N = 8.
        self.sumn = 2 * SUMN if twocta else SUMN
        self.num_regs_compute = 120
        self.num_regs_other = 24

    def set_qtm_kb(self, kb):
        # Tmem column map for kb K64 blocks of Q held in tmem (qtm only).
        self.tmem_o = 0
        self.tmem_s0 = 256
        self.tmem_s1 = 384
        self.qtm_kb = kb
        self.tmem_q = 256
        self.tmem_r = 0
        if self.qtm:
            # Q takes 16 columns per K64 block; with three blocks the layout is
            # O 0..255 | Q 256..303 | S0 304..431 | S1 432..559 | sum 560..575,
            # which is exactly Rubin's 576 columns.
            self.tmem_s0 = self.tmem_q + 16 * self.qtm_kb
            self.tmem_s1 = self.tmem_s0 + 128
            self.tmem_r = self.tmem_s1 + 128 - 512
        # Rubin carries 576 tmem columns per SM; O + the two S/P slots already
        # fill a 512-column allocation, so the row-sum accumulator takes a
        # second, smaller allocation (the unit is 32 columns and allocations
        # must not grow, which 512 -> 32 respects).
        self.tmem_sum_cols = 64 if self.qtm else 32
        self.has_alloc2 = self.tsum or self.qtm
        return self

    # ---------------------------------------------------------------- host ---
    @cute.jit
    def __call__(
        self,
        rQ: cute.Tensor,
        rKV: cute.Tensor,
        rBT: cute.Tensor,
        rO: cute.Tensor,
        rPlan: cute.Tensor,
        rBin: cute.Tensor,
        rPO: cute.Tensor,
        rPML: cute.Tensor,
        rMrg: cute.Tensor,
        rMBin: cute.Tensor,
        cnt_base: Int64,
        scale_log2: Float32,
        out_scale: Float32,
        nwork: Int32,
        nslot: Int32,
        grid: cutlass.Constexpr,
        stream,
    ):
        ntok = rQ.shape[0]
        npage = rKV.shape[0]
        nreq = rBT.shape[0]

        mQ = cute.make_tensor(
            rQ.iterator,
            cute.make_layout(((GROUP, ntok), HDIM, HKV),
                             stride=((HDIM, HQ * HDIM), 1, GROUP * HDIM)))
        mO = cute.make_tensor(
            rO.iterator,
            cute.make_layout(((GROUP, ntok), (8, HDIM // 8), HKV),
                             stride=((HDIM, HQ * HDIM), (1, 8), GROUP * HDIM)))
        mBT = cute.make_tensor(
            rBT.iterator, cute.make_layout((nreq, MAXP), stride=(MAXP, 1)))
        mPlan = cute.make_tensor(
            rPlan.iterator, cute.make_layout((nwork, PLAN_F), stride=(PLAN_F, 1)))
        nbin = (grid // 2) if self.clustered else grid
        mBin = cute.make_tensor(
            rBin.iterator, cute.make_layout(nbin + 1))
        mPO = cute.make_tensor(
            rPO.iterator,
            cute.make_layout((MROW, (POV, HDIM // POV), nslot),
                             stride=(POV, (1, POV * MROW), MROW * HDIM)))
        mPML = cute.make_tensor(
            rPML.iterator, cute.make_layout((MROW, 2, nslot), stride=(1, MROW, 2 * MROW)))
        mMrg = cute.make_tensor(
            rMrg.iterator, cute.make_layout((rMrg.shape[0], MRG_F), stride=(MRG_F, 1)))
        mMBin = cute.make_tensor(
            rMBin.iterator, cute.make_layout(grid + 1))

        qk_mma = rubin_utils.make_trivial_tiled_mma(
            cutlass.Float8E4M3FN, cutlass.Float8E4M3FN,
            OperandMajorMode.K, OperandMajorMode.K,
            Float32, self.cta_group, (self.mma_m, NTILE, 64),
        )
        qk_mma_ts = rubin_utils.make_trivial_tiled_mma(
            cutlass.Float8E4M3FN, cutlass.Float8E4M3FN,
            OperandMajorMode.K, OperandMajorMode.K,
            Float32, self.cta_group, (self.mma_m, NTILE, 64),
            a_source=tcgen05.OperandSource.TMEM,
        )
        tQt_layout = sm1xx_utils.make_smem_layout_a(
            qk_mma_ts, (self.mma_m, NTILE, 128), cutlass.Float8E4M3FN, 1)
        mQg = cute.make_tensor(
            rQ.iterator,
            cute.make_layout(((GROUP, ntok), (32, HDIM // 32), HKV),
                             stride=((HDIM, HQ * HDIM), (1, 32), GROUP * HDIM)))
        pv_mma = rubin_utils.make_trivial_tiled_mma(
            cutlass.Float8E4M3FN, cutlass.Float8E4M3FN,
            OperandMajorMode.K, OperandMajorMode.MN,
            Float32, self.cta_group, (self.mma_m, HDIM, 64),
            a_source=tcgen05.OperandSource.TMEM,
        )
        pv_mma_fill = rubin_utils.make_trivial_tiled_mma(
            cutlass.Float8E4M3FN, cutlass.Float8E4M3FN,
            OperandMajorMode.K, OperandMajorMode.MN,
            Float32, self.cta_group, (self.mma_m, HDIM, 64),
            a_source=tcgen05.OperandSource.TMEM,
            b_collector_op=tcgen05.CollectorOp.FILL,
        )
        pv_mma_last = rubin_utils.make_trivial_tiled_mma(
            cutlass.Float8E4M3FN, cutlass.Float8E4M3FN,
            OperandMajorMode.K, OperandMajorMode.MN,
            Float32, self.cta_group, (self.mma_m, HDIM, 64),
            a_source=tcgen05.OperandSource.TMEM,
            b_collector_op=tcgen05.CollectorOp.LASTUSE,
        )

        sum_mma = rubin_utils.make_trivial_tiled_mma(
            cutlass.Float8E4M3FN, cutlass.Float8E4M3FN,
            OperandMajorMode.K, OperandMajorMode.K,
            Float32, self.cta_group, (self.mma_m, self.sumn, 64),
            a_source=tcgen05.OperandSource.TMEM,
        )
        sOnes_layout = sm1xx_utils.make_smem_layout_b(
            sum_mma, (self.mma_m, self.sumn, NTILE), cutlass.Float8E4M3FN, 1)

        sQ_layout = sm1xx_utils.make_smem_layout_a(
            qk_mma, self.qk_mma_tiler, cutlass.Float8E4M3FN, 1)
        sK_layout = sm1xx_utils.make_smem_layout_b(
            qk_mma, self.qk_mma_tiler, cutlass.Float8E4M3FN, self.kvstage)
        sV_layout = sm1xx_utils.make_smem_layout_b(
            pv_mma, self.pv_mma_tiler, cutlass.Float8E4M3FN, self.kvstage)
        tP_layout = sm1xx_utils.make_smem_layout_a(
            pv_mma, self.pv_mma_tiler, cutlass.Float8E4M3FN, 1)

        if cutlass.const_expr(self.twocta):
            # cluster (2,1) at CtaGroup.TWO: the pair splits A in M and B in N
            # internally, so neither operand multicasts -- both atoms are the
            # plain 2-SM flavour and every stage barrier sees the pair's bytes.
            cluster_vmnk = (2, 1, 1, 1)
            kv_cluster = (2, 1, 1, 1)
            tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
            kv_op = tma_op
        else:
            cluster_vmnk = (1, 1, 1, 1)
            tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
            if cutlass.const_expr(self.mcast):
                kv_cluster = (1, 2, 1, 1)
                kv_op = cpasync.CopyBulkTensorTileG2SMulticastOp(
                    tcgen05.CtaGroup.ONE)
            else:
                kv_cluster = cluster_vmnk
                kv_op = tma_op

        tma_atom_q, tQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, mQ, cute.select(sQ_layout, mode=[0, 1, 2]),
            self.qk_mma_tiler, qk_mma, cluster_vmnk)
        mK = cute.make_tensor(
            rKV.iterator,
            cute.make_layout((PAGE, HDIM, HKV, npage),
                             stride=(2 * HDIM, 1, PAGE * 2 * HDIM, HKV * PAGE * 2 * HDIM)))
        tma_atom_k, tK = cute.nvgpu.make_tiled_tma_atom_B(
            kv_op, mK, cute.select(sK_layout, mode=[0, 1, 2]),
            self.qk_mma_tiler, qk_mma, kv_cluster)
        mV = cute.make_tensor(
            rKV.iterator + HDIM,
            cute.make_layout((HDIM, PAGE, HKV, npage),
                             stride=(1, 2 * HDIM, PAGE * 2 * HDIM, HKV * PAGE * 2 * HDIM)))
        tma_atom_v, tV = cute.nvgpu.make_tiled_tma_atom_B(
            kv_op, mV, cute.select(sV_layout, mode=[0, 1, 2]),
            self.pv_mma_tiler, pv_mma, kv_cluster)

        # A cta_group::2 TMA credits BOTH CTAs' transaction counters, so each
        # stage barrier expects the pair's bytes, not this CTA's half.
        self.tma_bytes_q = self.atom_sm * cute.size_in_bytes(
            cutlass.Float8E4M3FN, cute.select(sQ_layout, mode=[0, 1, 2]))
        self.tma_bytes_k = self.atom_sm * cute.size_in_bytes(
            cutlass.Float8E4M3FN, cute.select(sK_layout, mode=[0, 1, 2]))
        self.tma_bytes_v = self.atom_sm * cute.size_in_bytes(
            cutlass.Float8E4M3FN, cute.select(sV_layout, mode=[0, 1, 2]))

        # The constant row-sum operand only exists on the pipelines that use it:
        # carrying its shared-memory block (and the padding its 1 KiB alignment
        # adds) on the one-CTA pipelines measured 20-45% slower on short-stream
        # launches even with the row-sum UMMA itself compiled out.
        if cutlass.const_expr(self.tsum):
            @cute.struct
            class SharedStorage:
                sQ: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(sQ_layout)], 1024]
                sK: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(sK_layout)], 1024]
                sV: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(sV_layout)], 1024]
                sOnes: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float8E4M3FN,
                                         cute.cosize(sOnes_layout)], 1024]
                mbar: cute.struct.MemRange[Int64, MB_N]
                tmem_holder: Int32
                tmem_holder2: Int32
                sAux: cute.struct.MemRange[Int32, 4]
                sRed: cute.struct.MemRange[Float32, 32 * NCOMPW]
        else:
            @cute.struct
            class SharedStorage:
                sQ: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(sQ_layout)], 1024]
                sK: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(sK_layout)], 1024]
                sV: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(sV_layout)], 1024]
                mbar: cute.struct.MemRange[Int64, MB_N]
                tmem_holder: Int32
                tmem_holder2: Int32
                sAux: cute.struct.MemRange[Int32, 4]
                sRed: cute.struct.MemRange[Float32, 32 * NCOMPW]

        self.shared_storage = SharedStorage
        mrg_region = (cute.cosize(sQ_layout) + cute.cosize(sK_layout) +
                      cute.cosize(sV_layout))
        mrg_stage_b = (MCOL // NSEG + 2) * 4 * 32 * NCOMPW
        self.mrg_stages = min(MRG_STAGES_MAX, mrg_region // mrg_stage_b)

        self.kernel(
            qk_mma, pv_mma, pv_mma_fill, pv_mma_last, sum_mma, qk_mma_ts,
            tma_atom_q, tQ, tma_atom_k, tK, tma_atom_v, tV,
            mBT, mO, mQg, mPlan, mBin, mPO, mPML, mMrg, mMBin, cnt_base,
            scale_log2, out_scale, nwork,
            sQ_layout, sK_layout, sV_layout, tP_layout, sOnes_layout,
            tQt_layout,
        ).launch(
            grid=[grid, 1, 1],
            block=[NTHREAD, 1, 1],
            cluster=[2, 1, 1] if self.clustered else [1, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # -------------------------------------------------------------- device ---
    @cute.kernel
    def kernel(
        self,
        qk_mma: cute.TiledMma,
        pv_mma: cute.TiledMma,
        pv_mma_fill: cute.TiledMma,
        pv_mma_last: cute.TiledMma,
        sum_mma: cute.TiledMma,
        qk_mma_ts: cute.TiledMma,
        tma_atom_q: cute.CopyAtom, tQ: cute.Tensor,
        tma_atom_k: cute.CopyAtom, tK: cute.Tensor,
        tma_atom_v: cute.CopyAtom, tV: cute.Tensor,
        mBT: cute.Tensor,
        mO: cute.Tensor,
        mQg: cute.Tensor,
        mPlan: cute.Tensor,
        mBin: cute.Tensor,
        mPO: cute.Tensor,
        mPML: cute.Tensor,
        mMrg: cute.Tensor,
        mMBin: cute.Tensor,
        cnt_base: Int64,
        scale_log2: Float32,
        out_scale: Float32,
        nwork: Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sOnes_layout: cute.ComposedLayout,
        tQt_layout: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.clustered):
            rank = cute.arch.make_warp_uniform(Int32(bidx & 1))
            cbin = bidx >> 1
        else:
            rank = Int32(0)
            cbin = bidx
        w0 = mBin[cbin]
        nit = mBin[cbin + 1] - w0

        if warp_idx == WARP_LOAD:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mbar = storage.mbar.data_ptr()
        sAux = storage.sAux.get_tensor(cute.make_layout(4))
        sRed = storage.sRed.get_tensor(cute.make_layout(32 * NCOMPW))
        sQ = cute.make_tensor(
            cute.recast_ptr(storage.sQ.data_ptr(), sQ_layout.inner), sQ_layout.outer)
        sK = cute.make_tensor(
            cute.recast_ptr(storage.sK.data_ptr(), sK_layout.inner), sK_layout.outer)
        sV = cute.make_tensor(
            cute.recast_ptr(storage.sV.data_ptr(), sV_layout.inner), sV_layout.outer)
        MST = const_expr(max(self.mrg_stages, 1))
        MNV = const_expr(MCOL // NSEG // POV)
        MNT = const_expr(32 * NCOMPW)
        mrg_ptr = cute.recast_ptr(storage.sQ.data_ptr(), dtype=Float32)
        sMrg = cute.make_tensor(
            mrg_ptr,
            cute.make_layout((POV, MNT, MNV, MST),
                             stride=(1, POV, POV * MNT, POV * MNT * MNV)))
        sMrgML = cute.make_tensor(
            mrg_ptr + MST * POV * MNT * MNV,
            cute.make_layout((2, MNT, MST), stride=(MNT, 1, 2 * MNT)))
        if cutlass.const_expr(self.tsum):
            ones_ptr = storage.sOnes.data_ptr()
        else:
            ones_ptr = storage.sQ.data_ptr()
        sOnes = cute.make_tensor(
            cute.recast_ptr(ones_ptr, sOnes_layout.inner), sOnes_layout.outer)
        if cutlass.const_expr(self.tsum):
            # Every byte of the row-sum operand is e4m3 1.0 (0x38), so the
            # operand swizzle is irrelevant and one constant store per word
            # fills it for the whole kernel.
            nw1 = const_expr(cute.cosize(sOnes_layout) // 4)
            ones32 = cute.make_tensor(
                cute.recast_ptr(ones_ptr, dtype=Int32),
                cute.make_layout(nw1))
            for r1 in cutlass.range_constexpr((nw1 + NTHREAD - 1) // NTHREAD):
                if tidx + r1 * NTHREAD < nw1:
                    ones32[tidx + r1 * NTHREAD] = Int32(0x38383838)

        if tidx == 0:
            cute.arch.mbarrier_init(mbar + MB_Q, 1)
            cute.arch.mbarrier_init(mbar + MB_QE, 1)
            nsig = const_expr(2 * NCOMPW if self.twocta else NCOMPW)
            cute.arch.mbarrier_init(mbar + MB_EPI, nsig)
            cute.arch.mbarrier_init(mbar + MB_QT, nsig)
            # twocta: only the pair-leader issues UMMA, and its commits carry
            # the pair mask, so each release barrier sees exactly one arrival.
            nrel = const_expr(1 if self.twocta else (2 if self.mcast else 1))
            for i in cutlass.range_constexpr(self.kvstage):
                cute.arch.mbarrier_init(mbar + MB_K + i, 1)
                cute.arch.mbarrier_init(mbar + MB_V + i, 1)
                cute.arch.mbarrier_init(mbar + MB_KE + i, nrel)
                cute.arch.mbarrier_init(mbar + MB_VE + i, nrel)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(mbar + MB_S + i, 1)
                cute.arch.mbarrier_init(mbar + MB_P + i, nsig)
                cute.arch.mbarrier_init(mbar + MB_PV + i, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if cutlass.const_expr(self.clustered):
            # Both CTAs publish their mbarriers before either one can be the
            # target of a multicast TMA or of the peer's tcgen05 release.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        if warp_idx == WARP_MMA:
            cute.arch.alloc_tmem(self.tmem_cols, storage.tmem_holder,
                                 is_two_cta=self.twocta)
            if cutlass.const_expr(self.has_alloc2):
                cute.arch.alloc_tmem(self.tmem_sum_cols, storage.tmem_holder2,
                                     is_two_cta=self.twocta)
            cute.arch.relinquish_tmem_alloc_permit(
                is_two_cta=self.twocta)
        cute.arch.sync_threads()
        if cutlass.const_expr(self.twocta):
            # The leader's cooperative UMMA writes the peer's tmem, so the pair
            # must both hold their columns before any MMA can issue.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        thr_qk = qk_mma.get_slice(0)
        thr_pv = pv_mma.get_slice(0)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holder)
        tStS_fake = thr_qk.make_fragment_C(
            thr_qk.partition_shape_C((self.mma_m, NTILE)))
        tOtO_fake = thr_pv.make_fragment_C(
            thr_pv.partition_shape_C((self.mma_m, HDIM)))
        tOtO = cute.make_tensor(tmem_ptr + self.tmem_o, tOtO_fake.layout)
        tStS0 = cute.make_tensor(tmem_ptr + self.tmem_s0, tStS_fake.layout)
        thr_sum = sum_mma.get_slice(0)
        tRtR_fake = thr_sum.make_fragment_C(
            thr_sum.partition_shape_C((self.mma_m, self.sumn)))
        if cutlass.const_expr(self.has_alloc2):
            tmem_ptr2 = cute.arch.retrieve_tmem_ptr(
                Float32, alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holder2)
        else:
            tmem_ptr2 = tmem_ptr
        if cutlass.const_expr(self.tsum):
            tRtR = cute.make_tensor(tmem_ptr2 + self.tmem_r, tRtR_fake.layout)
        else:
            tRtR = cute.make_tensor(tmem_ptr, tRtR_fake.layout)

        if warp_idx == WARP_MMA:
            if cutlass.const_expr(ROLES & 1):
                if cutlass.const_expr(self.twocta):
                    # Only the pair-leader issues the cooperative UMMA; the peer
                    # just feeds its operand halves and runs its own epilogue.
                    if rank == 0:
                        if cutlass.const_expr(self.deep):
                            self.mma_loop_deep(
                                qk_mma, pv_mma, pv_mma_fill, pv_mma_last,
                                sum_mma, qk_mma_ts, tQt_layout, sQ, sK, sV, sOnes, tStS0, tOtO,
                                tRtR, tmem_ptr,
                                tP_layout, mPlan, mbar, w0, nit)
                        else:
                            self.mma_loop(
                                qk_mma, pv_mma, pv_mma_fill, pv_mma_last,
                                sum_mma, qk_mma_ts, tQt_layout, sQ, sK, sV, sOnes, tStS0, tOtO,
                                tRtR, tmem_ptr,
                                tP_layout, mPlan, mbar, w0, nit)
                else:
                    self.mma_loop(qk_mma, pv_mma, pv_mma_fill, pv_mma_last,
                                  sum_mma, qk_mma_ts, tQt_layout, sQ, sK, sV, sOnes, tStS0, tOtO,
                                  tRtR, tmem_ptr,
                                  tP_layout, mPlan, mbar, w0, nit)
            if cutlass.const_expr(not self.twocta):
                if cutlass.const_expr(self.has_alloc2):
                    cute.arch.dealloc_tmem(tmem_ptr2, self.tmem_sum_cols)
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols)
        elif warp_idx == WARP_LOAD:
            if cutlass.const_expr(ROLES & 2):
                self.load_loop(qk_mma, pv_mma, tma_atom_q, tQ, tma_atom_k, tK,
                               tma_atom_v, tV, sQ, sK, sV, mBT, mPlan, mbar,
                               w0, nit, rank)
        elif warp_idx < NCOMPW:
            if cutlass.const_expr(ROLES & 4):
                self.compute_loop(qk_mma, pv_mma, tStS0, tOtO, tRtR, sAux,
                                  sRed, sMrg, sMrgML, mPlan, mO, mQg, mPO, mPML,
                                  mMrg, mMBin,
                                  cnt_base, mbar,
                                  scale_log2, out_scale, w0, nit, tidx, rank,
                                  bidx)
        if cutlass.const_expr(self.clustered):
            # No CTA may retire while its peer can still read its smem or tmem.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        if cutlass.const_expr(self.twocta):
            # The peer's tmem is a target of the leader's UMMA, so neither CTA
            # may free its columns before the pair has drained.
            if warp_idx == WARP_MMA:
                if cutlass.const_expr(self.has_alloc2):
                    cute.arch.dealloc_tmem(tmem_ptr2, self.tmem_sum_cols,
                                           is_two_cta=True)
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols,
                                       is_two_cta=True)

    # ----------------------------------------------------------- load warp ---
    @cute.jit
    def load_loop(self, qk_mma, pv_mma, tma_atom_q, tQ, tma_atom_k, tK,
                  tma_atom_v, tV, sQ, sK, sV, mBT, mPlan, mbar,
                  w0: Int32, nit: Int32, rank: Int32):
        # Under CtaGroup.TWO the MMA V-coordinate selects this CTA's M-half of Q
        # and its N-half of K / V, so every gmem partition is rank-sliced.
        # Under cta_group::2 the pair's K/V/Q loads all complete on the LEADER
        # CTA's transaction barrier, so only the leader arms it -- for the
        # pair's bytes.  The peer never arms and never waits those barriers,
        # which is what keeps its arrival counter from running away.
        mma_v = rank if self.twocta else Int32(0)
        arm = (rank == 0) if self.twocta else None
        thr_qk = qk_mma.get_slice(mma_v)
        thr_pv = pv_mma.get_slice(mma_v)
        gK_all = cute.local_tile(tK, cute.select(self.qk_mma_tiler, mode=[1, 2]),
                                 (None, 0, None, None))
        gV_all = cute.local_tile(tV, cute.select(self.pv_mma_tiler, mode=[1, 2]),
                                 (0, None, None, None))
        sQg = cute.group_modes(sQ, 0, 3)
        sKg = cute.group_modes(sK, 0, 3)
        sVg = cute.group_modes(sV, 0, 3)
        if cutlass.const_expr(self.mcast):
            # Each CTA fetches one half of the page and broadcasts it to both,
            # so the bytes crossing the L2 network per SM halve.
            kv_cta_layout = cute.make_layout(2)
            kv_coord = rank
            kv_mask = cutlass.Int16(3)
        else:
            # twocta also lands here: the pair splits K|V in N internally, so
            # the TMA is the plain 2-SM flavour with no multicast group.
            kv_cta_layout = cute.make_layout(1)
            kv_coord = Int32(0)
            kv_mask = cutlass.Int16(0)
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_k, kv_coord, kv_cta_layout, sKg,
            cute.group_modes(thr_qk.partition_B(gK_all), 0, 3))
        tVsV, tVgV = cpasync.tma_partition(
            tma_atom_v, kv_coord, kv_cta_layout, sVg,
            cute.group_modes(thr_pv.partition_B(gV_all), 0, 3))

        # K is consumed one tile ahead of V (QK(t+1) is issued before PV(t)).
        # Short launches use independent cursors so TMA request order follows
        # those deadlines: K0, K1, V0, K2, V1, ... .
        kstage = Int32(0)
        kphase = Int32(1)
        vstage = Int32(0)
        vphase = Int32(1)
        qphase = Int32(1)
        for wi in cutlass.range(nit):
            w = w0 + wi
            tok_base = mPlan[w, 0]
            h_kv = mPlan[w, 2]
            b = mPlan[w, 3]
            kv_begin = mPlan[w, 4]
            kv_ntile = mPlan[w, 5]
            if cutlass.const_expr(self.mcast):
                if mPlan[w, 1] > MROW // GROUP:
                    tok_base = tok_base + rank * (MROW // GROUP)

            gQ_all = cute.domain_offset(((0, tok_base), 0, 0), tQ)
            gQ = cute.local_tile(gQ_all[None, None, h_kv],
                                 cute.select(self.qk_mma_tiler, mode=[0, 2]), (0, 0))
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_q, 0, cute.make_layout(1), sQg,
                cute.group_modes(thr_qk.partition_A(gQ), 0, 3))
            cute.arch.mbarrier_wait(mbar + MB_QE, qphase)
            if cutlass.const_expr(self.twocta):
                if arm:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar + MB_Q, self.tma_bytes_q)
            else:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar + MB_Q, self.tma_bytes_q)
            cute.copy(tma_atom_q, tQgQ, tQsQ[None, 0], tma_bar_ptr=mbar + MB_Q)
            qphase = qphase ^ 1

            pg0 = kv_begin >> 7
            page = mBT[b, pg0]

            if cutlass.const_expr(self.skew_load):
                cute.arch.mbarrier_wait(mbar + MB_KE + kstage, kphase)
                if cutlass.const_expr(self.twocta):
                    if arm:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar + MB_K + kstage, self.tma_bytes_k)
                else:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar + MB_K + kstage, self.tma_bytes_k)
                if cutlass.const_expr(self.mcast):
                    cute.copy(tma_atom_k, tKgK[None, 0, h_kv, page],
                              tKsK[None, kstage],
                              tma_bar_ptr=mbar + MB_K + kstage,
                              mcast_mask=kv_mask)
                else:
                    cute.copy(tma_atom_k, tKgK[None, 0, h_kv, page],
                              tKsK[None, kstage],
                              tma_bar_ptr=mbar + MB_K + kstage)
                kstage = kstage + 1
                if kstage == self.kvstage:
                    kstage = Int32(0)
                    kphase = kphase ^ 1

            for t in cutlass.range(kv_ntile):
                page_next = page
                if t + 1 < kv_ntile:
                    page_next = mBT[b, pg0 + t + 1]
                if cutlass.const_expr(self.skew_load):
                    if t + 1 < kv_ntile:
                        cute.arch.mbarrier_wait(mbar + MB_KE + kstage, kphase)
                        if cutlass.const_expr(self.twocta):
                            if arm:
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_arrive_and_expect_tx(
                                        mbar + MB_K + kstage, self.tma_bytes_k)
                        else:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mbar + MB_K + kstage, self.tma_bytes_k)
                        if cutlass.const_expr(self.mcast):
                            cute.copy(tma_atom_k, tKgK[None, 0, h_kv, page_next],
                                      tKsK[None, kstage],
                                      tma_bar_ptr=mbar + MB_K + kstage,
                                      mcast_mask=kv_mask)
                        else:
                            cute.copy(tma_atom_k, tKgK[None, 0, h_kv, page_next],
                                      tKsK[None, kstage],
                                      tma_bar_ptr=mbar + MB_K + kstage)
                        kstage = kstage + 1
                        if kstage == self.kvstage:
                            kstage = Int32(0)
                            kphase = kphase ^ 1
                else:
                    cute.arch.mbarrier_wait(mbar + MB_KE + kstage, kphase)
                    if cutlass.const_expr(self.twocta):
                        if arm:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mbar + MB_K + kstage, self.tma_bytes_k)
                    else:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar + MB_K + kstage, self.tma_bytes_k)
                    if cutlass.const_expr(self.mcast):
                        cute.copy(tma_atom_k, tKgK[None, 0, h_kv, page],
                                  tKsK[None, kstage],
                                  tma_bar_ptr=mbar + MB_K + kstage,
                                  mcast_mask=kv_mask)
                    else:
                        cute.copy(tma_atom_k, tKgK[None, 0, h_kv, page],
                                  tKsK[None, kstage],
                                  tma_bar_ptr=mbar + MB_K + kstage)
                    kstage = kstage + 1
                    if kstage == self.kvstage:
                        kstage = Int32(0)
                        kphase = kphase ^ 1

                cute.arch.mbarrier_wait(mbar + MB_VE + vstage, vphase)
                if cutlass.const_expr(self.twocta):
                    if arm:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar + MB_V + vstage, self.tma_bytes_v)
                else:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar + MB_V + vstage, self.tma_bytes_v)
                if cutlass.const_expr(self.mcast):
                    cute.copy(tma_atom_v, tVgV[None, 0, h_kv, page],
                              tVsV[None, vstage],
                              tma_bar_ptr=mbar + MB_V + vstage,
                              mcast_mask=kv_mask)
                else:
                    cute.copy(tma_atom_v, tVgV[None, 0, h_kv, page],
                              tVsV[None, vstage],
                              tma_bar_ptr=mbar + MB_V + vstage)
                vstage = vstage + 1
                if vstage == self.kvstage:
                    vstage = Int32(0)
                    vphase = vphase ^ 1
                page = page_next

    # ------------------------------------------------------------ mma warp ---
    @cute.jit
    def mma_loop(self, qk_mma, pv_mma, pv_mma_fill, pv_mma_last, sum_mma,
                 qk_mma_ts, tQt_layout,
                 sQ, sK, sV, sOnes, tStS0, tOtO, tRtR, tmem_ptr,
                 tP_layout, mPlan, mbar, w0: Int32, nit: Int32):
        thr_pv = pv_mma.get_slice(0)
        tSrQ = qk_mma.make_fragment_A(sQ)[None, None, None, 0]
        tSrQt = None
        tSrQt2 = None
        tQblk = None
        if cutlass.const_expr(self.qtm):
            thr_qkt = qk_mma_ts.get_slice(0)
            tQt0 = cute.make_tensor(tmem_ptr, tQt_layout.outer)
            tSrQt0 = thr_qkt.make_fragment_A(tQt0)[None, None, None, 0]
            tSrQt = cute.make_tensor(tSrQt0.iterator + 4 * self.tmem_q,
                                     tSrQt0.layout)
            tQblk = [tSrQt[None, None, 0], tSrQt[None, None, 1]]
            if cutlass.const_expr(self.qtm_kb > 2):
                tSrQt2 = cute.make_tensor(tSrQt.iterator + 4 * 32,
                                          tSrQt[None, None, 0].layout)
                tQblk.append(tSrQt2)
        tSrK = qk_mma.make_fragment_B(sK)
        tOrV = pv_mma.make_fragment_B(sV)
        tP0 = cute.make_tensor(tmem_ptr, tP_layout.outer)
        tOrP0 = thr_pv.make_fragment_A(tP0)[None, None, None, 0]
        s_layout = tStS0.layout
        p_layout = tOrP0.layout
        s_iter = tStS0.iterator
        p_iter = tOrP0.iterator
        nk_qk = const_expr(cute.size(tSrQ, mode=[2]))
        nk_pv = const_expr(cute.size(tOrP0, mode=[2]))
        thr_sum = sum_mma.get_slice(0)
        tOnes = sum_mma.make_fragment_B(sOnes)
        tRrP0 = thr_sum.make_fragment_A(tP0)[None, None, None, 0]
        ps_layout = tRrP0.layout
        ps_iter = tRrP0.iterator
        nk_sum = const_expr(cute.size(tRrP0, mode=[2]))
        nterm_r = const_expr(2 if self.use_packed else 3)

        sq = Int32(0)        # kv stage consumed by Q*K^T (leads)
        sqph = Int32(0)
        sp = Int32(0)        # kv stage consumed by P*V (trails by one tile)
        spph = Int32(0)
        qphase = Int32(0)
        epiphase = Int32(0)
        gtq = Int32(0)       # global tile index of the issued Q*K^T
        gtp = Int32(0)       # global tile index of the issued P*V
        for wi in cutlass.range(nit):
            w = w0 + wi
            kv_ntile = mPlan[w, 5]
            resid_mask = mPlan[w, 11]
            if cutlass.const_expr(not self.use_packed):
                resid_mask = Int32(0)
            cute.arch.mbarrier_wait(mbar + MB_Q, qphase)
            if cutlass.const_expr(self.qtm):
                cute.arch.mbarrier_wait(mbar + MB_QT, qphase)
            qphase = qphase ^ 1

            # --- prologue: issue S = Q*K^T for the first tile ---
            jq = gtq & 1
            cute.arch.mbarrier_wait(mbar + MB_K + sq, sqph)
            # P*V(gtq-2) already waited on this S/P slot at the same phase in
            # this warp's program order, so a second phase check cannot block.
            tStSi = cute.make_tensor(s_iter + jq * 128, s_layout)
            for kp in cutlass.range_constexpr(nk_qk):
                if cutlass.const_expr(kp < (self.qtm_kb if self.qtm else 0)):
                    qk_mma_ts.set(tcgen05.Field.ACCUMULATE, const_expr(kp != 0))
                    cute.gemm(qk_mma_ts, tStSi, tQblk[kp],
                              tSrK[None, None, kp, sq], tStSi)
                else:
                    qk_mma.set(tcgen05.Field.ACCUMULATE, const_expr(kp != 0))
                    cute.gemm(qk_mma, tStSi, tSrQ[None, None, kp],
                              tSrK[None, None, kp, sq], tStSi)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MB_S + jq, self.pair_mask, self.cta_group)
                tcgen05.commit(mbar + MB_KE + sq, self.kv_rel_mask,
                               self.cta_group)
            sq = sq + 1
            if sq == self.kvstage:
                sq = Int32(0)
                sqph = sqph ^ 1
            gtq = gtq + 1
            if wi > 0:
                cute.arch.mbarrier_wait(mbar + MB_EPI, epiphase)
                epiphase = epiphase ^ 1

            # --- steady state: issue QK(t+1) then PV(t) so the MMA never idles ---
            for t in cutlass.range(kv_ntile - 1):
                jq = gtq & 1
                cute.arch.mbarrier_wait(mbar + MB_K + sq, sqph)
                tStSi = cute.make_tensor(s_iter + jq * 128, s_layout)
                for kp in cutlass.range_constexpr(nk_qk):
                    if cutlass.const_expr(kp < (self.qtm_kb if self.qtm else 0)):
                        qk_mma_ts.set(tcgen05.Field.ACCUMULATE, const_expr(kp != 0))
                        cute.gemm(qk_mma_ts, tStSi, tQblk[kp],
                                  tSrK[None, None, kp, sq], tStSi)
                    else:
                        qk_mma.set(tcgen05.Field.ACCUMULATE, const_expr(kp != 0))
                        cute.gemm(qk_mma, tStSi, tSrQ[None, None, kp],
                                  tSrK[None, None, kp, sq], tStSi)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MB_S + jq, self.pair_mask,
                                   self.cta_group)
                    tcgen05.commit(mbar + MB_KE + sq, self.kv_rel_mask,
                                   self.cta_group)
                sq = sq + 1
                if sq == self.kvstage:
                    sq = Int32(0)
                    sqph = sqph ^ 1
                gtq = gtq + 1

                jp = gtp & 1
                cute.arch.mbarrier_wait(mbar + MB_V + sp, spph)
                cute.arch.mbarrier_wait(mbar + MB_P + jp, (gtp >> 1) & 1)
                tOrPi = cute.make_tensor(p_iter + 4 * (self.tmem_s0 + jp * 128),
                                         p_layout)
                tOrLi = cute.make_tensor(
                    p_iter + 4 * (self.tmem_s0 + jp * 128 + PCOL_T), p_layout)
                tOrXi = cute.make_tensor(
                    p_iter + 4 * (self.tmem_s0 + jp * 128 + 2 * PCOL_T), p_layout)
                if cutlass.const_expr(self.resid_period != 0):
                    if cutlass.const_expr(self.use_item_mask):
                        use_resid = (gtp & resid_mask) == 0
                    else:
                        use_resid = gtp % self.resid_period == 0
                    if use_resid:
                        for kp in cutlass.range_constexpr(nk_pv):
                            if const_expr(kp == 0):
                                pv_mma_fill.set(tcgen05.Field.ACCUMULATE, t != 0)
                            else:
                                pv_mma_fill.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(pv_mma_fill, tOtO, tOrPi[None, None, kp],
                                      tOrV[None, None, kp, sp], tOtO)
                            pv_mma_last.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(pv_mma_last, tOtO, tOrLi[None, None, kp],
                                      tOrV[None, None, kp, sp], tOtO)
                            if cutlass.const_expr(not self.use_packed):
                                pv_mma_last.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(pv_mma_last, tOtO, tOrXi[None, None, kp],
                                          tOrV[None, None, kp, sp], tOtO)
                        if cutlass.const_expr(self.tsum):
                            # The V stage is free the moment P*V has read it; the
                            # row-sum UMMA below touches only P and the constant
                            # operand, and on the fill-bound one-CTA pipelines the
                            # stage release is the critical path.
                            with cute.arch.elect_one():
                                tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                               self.cta_group)
                            # Row sums of this tile's e4m3 P as a narrow UMMA: the
                            # denominator is then the sum of exactly the probabilities
                            # the P*V numerator consumed.
                            for tm in cutlass.range_constexpr(nterm_r):
                                tRrPi = cute.make_tensor(
                                    ps_iter + 4 * (self.tmem_s0 + jp * 128
                                                   + tm * PCOL_T), ps_layout)
                                for kq in cutlass.range_constexpr(nk_sum):
                                    if const_expr(kq == 0 and tm == 0):
                                        sum_mma.set(tcgen05.Field.ACCUMULATE, t != 0)
                                    else:
                                        sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                              tOnes[None, None, kq, 0], tRtR)
                    else:
                        for kp in cutlass.range_constexpr(nk_pv):
                            if const_expr(kp == 0):
                                pv_mma.set(tcgen05.Field.ACCUMULATE, t != 0)
                            else:
                                pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(pv_mma, tOtO, tOrPi[None, None, kp],
                                      tOrV[None, None, kp, sp], tOtO)
                        if cutlass.const_expr(self.tsum):
                            # The V stage is free the moment P*V has read it; the
                            # row-sum UMMA below touches only P and the constant
                            # operand, and on the fill-bound one-CTA pipelines the
                            # stage release is the critical path.
                            with cute.arch.elect_one():
                                tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                               self.cta_group)
                            # Row sums of this tile's e4m3 P as a narrow UMMA: the
                            # denominator is then the sum of exactly the probabilities
                            # the P*V numerator consumed.
                            for tm in cutlass.range_constexpr(1):
                                tRrPi = cute.make_tensor(
                                    ps_iter + 4 * (self.tmem_s0 + jp * 128
                                                   + tm * PCOL_T), ps_layout)
                                for kq in cutlass.range_constexpr(nk_sum):
                                    if const_expr(kq == 0 and tm == 0):
                                        sum_mma.set(tcgen05.Field.ACCUMULATE, t != 0)
                                    else:
                                        sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                              tOnes[None, None, kq, 0], tRtR)
                else:
                    for kp in cutlass.range_constexpr(nk_pv):
                        if const_expr(kp == 0):
                            pv_mma.set(tcgen05.Field.ACCUMULATE, t != 0)
                        else:
                            pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(pv_mma, tOtO, tOrPi[None, None, kp],
                                  tOrV[None, None, kp, sp], tOtO)
                    if cutlass.const_expr(self.tsum):
                        # The V stage is free the moment P*V has read it; the
                        # row-sum UMMA below touches only P and the constant
                        # operand, and on the fill-bound one-CTA pipelines the
                        # stage release is the critical path.
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                           self.cta_group)
                        # Row sums of this tile's e4m3 P as a narrow UMMA: the
                        # denominator is then the sum of exactly the probabilities
                        # the P*V numerator consumed.
                        for tm in cutlass.range_constexpr(1):
                            tRrPi = cute.make_tensor(
                                ps_iter + 4 * (self.tmem_s0 + jp * 128
                                               + tm * PCOL_T), ps_layout)
                            for kq in cutlass.range_constexpr(nk_sum):
                                if const_expr(kq == 0 and tm == 0):
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, t != 0)
                                else:
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                          tOnes[None, None, kq, 0], tRtR)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MB_PV + jp, self.pair_mask,
                                   self.cta_group)
                    if cutlass.const_expr(not self.tsum):
                        tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                       self.cta_group)
                sp = sp + 1
                if sp == self.kvstage:
                    sp = Int32(0)
                    spph = spph ^ 1
                gtp = gtp + 1

            # --- epilogue: last P*V of this work item ---
            jp = gtp & 1
            cute.arch.mbarrier_wait(mbar + MB_V + sp, spph)
            cute.arch.mbarrier_wait(mbar + MB_P + jp, (gtp >> 1) & 1)
            tOrPi = cute.make_tensor(p_iter + 4 * (self.tmem_s0 + jp * 128), p_layout)
            tOrLi = cute.make_tensor(
                p_iter + 4 * (self.tmem_s0 + jp * 128 + PCOL_T), p_layout)
            tOrXi = cute.make_tensor(
                p_iter + 4 * (self.tmem_s0 + jp * 128 + 2 * PCOL_T), p_layout)
            if cutlass.const_expr(self.resid_period != 0):
                if cutlass.const_expr(self.use_item_mask):
                    use_resid = (gtp & resid_mask) == 0
                else:
                    use_resid = gtp % self.resid_period == 0
                if use_resid:
                    for kp in cutlass.range_constexpr(nk_pv):
                        if const_expr(kp == 0):
                            pv_mma_fill.set(
                                tcgen05.Field.ACCUMULATE, kv_ntile != 1)
                        else:
                            pv_mma_fill.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(pv_mma_fill, tOtO, tOrPi[None, None, kp],
                                  tOrV[None, None, kp, sp], tOtO)
                        pv_mma_last.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(pv_mma_last, tOtO, tOrLi[None, None, kp],
                                  tOrV[None, None, kp, sp], tOtO)
                        if cutlass.const_expr(not self.use_packed):
                            pv_mma_last.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(pv_mma_last, tOtO, tOrXi[None, None, kp],
                                      tOrV[None, None, kp, sp], tOtO)
                    if cutlass.const_expr(self.tsum):
                        # The V stage is free the moment P*V has read it; the
                        # row-sum UMMA below touches only P and the constant
                        # operand, and on the fill-bound one-CTA pipelines the
                        # stage release is the critical path.
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                           self.cta_group)
                        # Row sums of this tile's e4m3 P as a narrow UMMA: the
                        # denominator is then the sum of exactly the probabilities
                        # the P*V numerator consumed.
                        for tm in cutlass.range_constexpr(nterm_r):
                            tRrPi = cute.make_tensor(
                                ps_iter + 4 * (self.tmem_s0 + jp * 128
                                               + tm * PCOL_T), ps_layout)
                            for kq in cutlass.range_constexpr(nk_sum):
                                if const_expr(kq == 0 and tm == 0):
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 1)
                                else:
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                          tOnes[None, None, kq, 0], tRtR)
                else:
                    for kp in cutlass.range_constexpr(nk_pv):
                        if const_expr(kp == 0):
                            pv_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 1)
                        else:
                            pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(pv_mma, tOtO, tOrPi[None, None, kp],
                                  tOrV[None, None, kp, sp], tOtO)
                    if cutlass.const_expr(self.tsum):
                        # The V stage is free the moment P*V has read it; the
                        # row-sum UMMA below touches only P and the constant
                        # operand, and on the fill-bound one-CTA pipelines the
                        # stage release is the critical path.
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                           self.cta_group)
                        # Row sums of this tile's e4m3 P as a narrow UMMA: the
                        # denominator is then the sum of exactly the probabilities
                        # the P*V numerator consumed.
                        for tm in cutlass.range_constexpr(1):
                            tRrPi = cute.make_tensor(
                                ps_iter + 4 * (self.tmem_s0 + jp * 128
                                               + tm * PCOL_T), ps_layout)
                            for kq in cutlass.range_constexpr(nk_sum):
                                if const_expr(kq == 0 and tm == 0):
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 1)
                                else:
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                          tOnes[None, None, kq, 0], tRtR)
            else:
                for kp in cutlass.range_constexpr(nk_pv):
                    if const_expr(kp == 0):
                        pv_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 1)
                    else:
                        pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(pv_mma, tOtO, tOrPi[None, None, kp],
                              tOrV[None, None, kp, sp], tOtO)
                if cutlass.const_expr(self.tsum):
                    # The V stage is free the moment P*V has read it; the
                    # row-sum UMMA below touches only P and the constant
                    # operand, and on the fill-bound one-CTA pipelines the
                    # stage release is the critical path.
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                       self.cta_group)
                    # Row sums of this tile's e4m3 P as a narrow UMMA: the
                    # denominator is then the sum of exactly the probabilities
                    # the P*V numerator consumed.
                    for tm in cutlass.range_constexpr(1):
                        tRrPi = cute.make_tensor(
                            ps_iter + 4 * (self.tmem_s0 + jp * 128
                                           + tm * PCOL_T), ps_layout)
                        for kq in cutlass.range_constexpr(nk_sum):
                            if const_expr(kq == 0 and tm == 0):
                                sum_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 1)
                            else:
                                sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                      tOnes[None, None, kq, 0], tRtR)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MB_PV + jp, self.pair_mask, self.cta_group)
                if cutlass.const_expr(not self.tsum):
                    tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                   self.cta_group)
                tcgen05.commit(mbar + MB_QE, self.pair_mask, self.cta_group)
            sp = sp + 1
            if sp == self.kvstage:
                sp = Int32(0)
                spph = spph ^ 1
            gtp = gtp + 1
        if nit > 0:
            cute.arch.mbarrier_wait(mbar + MB_EPI, epiphase)


    # --------------------------------------------- mma warp, two-tile lead ---
    @cute.jit
    def mma_loop_deep(self, qk_mma, pv_mma, pv_mma_fill, pv_mma_last, sum_mma,
                      qk_mma_ts, tQt_layout,
                      sQ, sK, sV, sOnes, tStS0, tOtO, tRtR, tmem_ptr,
                      tP_layout, mPlan, mbar, w0: Int32, nit: Int32):
        """Q*K^T runs two tiles ahead of P*V instead of one.

        Per item the schedule is [QK(0), QK(1)] then {PV(t), QK(t+2)}.  P*V(t)
        still precedes QK(t+2), so the fp8 P tile keeps aliasing the score slot
        it was produced from and the tmem layout is untouched; what changes is
        that the softmax warps now have a whole tile of UMMA between S(t) and
        P*V(t) rather than half of one.  Only the longest CtaGroup.TWO streams
        route here: shorter ones pay the extra ring stage in L2 residency and
        lose more than the latency slack wins."""
        thr_pv = pv_mma.get_slice(0)
        tSrQ = qk_mma.make_fragment_A(sQ)[None, None, None, 0]
        tSrQt = None
        tSrQt2 = None
        tQblk = None
        if cutlass.const_expr(self.qtm):
            thr_qkt = qk_mma_ts.get_slice(0)
            tQt0 = cute.make_tensor(tmem_ptr, tQt_layout.outer)
            tSrQt0 = thr_qkt.make_fragment_A(tQt0)[None, None, None, 0]
            tSrQt = cute.make_tensor(tSrQt0.iterator + 4 * self.tmem_q,
                                     tSrQt0.layout)
            tQblk = [tSrQt[None, None, 0], tSrQt[None, None, 1]]
            if cutlass.const_expr(self.qtm_kb > 2):
                tSrQt2 = cute.make_tensor(tSrQt.iterator + 4 * 32,
                                          tSrQt[None, None, 0].layout)
                tQblk.append(tSrQt2)
        tSrK = qk_mma.make_fragment_B(sK)
        tOrV = pv_mma.make_fragment_B(sV)
        tP0 = cute.make_tensor(tmem_ptr, tP_layout.outer)
        tOrP0 = thr_pv.make_fragment_A(tP0)[None, None, None, 0]
        s_layout = tStS0.layout
        p_layout = tOrP0.layout
        s_iter = tStS0.iterator
        p_iter = tOrP0.iterator
        nk_qk = const_expr(cute.size(tSrQ, mode=[2]))
        nk_pv = const_expr(cute.size(tOrP0, mode=[2]))
        thr_sum = sum_mma.get_slice(0)
        tOnes = sum_mma.make_fragment_B(sOnes)
        tRrP0 = thr_sum.make_fragment_A(tP0)[None, None, None, 0]
        ps_layout = tRrP0.layout
        ps_iter = tRrP0.iterator
        nk_sum = const_expr(cute.size(tRrP0, mode=[2]))
        nterm_r = const_expr(2 if self.use_packed else 3)

        sq = Int32(0)
        sqph = Int32(0)
        sp = Int32(0)
        spph = Int32(0)
        qphase = Int32(0)
        epiphase = Int32(0)
        gtq = Int32(0)
        gtp = Int32(0)
        # Bind every value the staged branches below rebind, so a one-tile work
        # item (which skips the second lead Q*K^T and the first drain P*V)
        # rejoins with the same types the tracer saw going in.
        jq = Int32(0)
        jp = Int32(0)
        use_resid = Boolean(True)
        # Seed every loop-carried TMEM view with the same dynamic-offset form
        # used in the staged loop body.  A constant offset preserves align<16>
        # while the dynamic jq/jp offset does not, and scf.for requires the
        # complete memref type (including alignment) to be invariant.
        tStSi = cute.make_tensor(s_iter + jq * 128, s_layout)
        tOrPi = cute.make_tensor(
            p_iter + 4 * (self.tmem_s0 + jp * 128), p_layout)
        tOrLi = cute.make_tensor(
            p_iter + 4 * (self.tmem_s0 + jp * 128 + PCOL_T), p_layout)
        tOrXi = cute.make_tensor(
            p_iter + 4 * (self.tmem_s0 + jp * 128 + 2 * PCOL_T), p_layout)
        tRrPi = cute.make_tensor(
            ps_iter + 4 * (self.tmem_s0 + jp * 128), ps_layout)
        for wi in cutlass.range(nit):
            w = w0 + wi
            kv_ntile = mPlan[w, 5]
            resid_mask = mPlan[w, 11]
            if cutlass.const_expr(not self.use_packed):
                resid_mask = Int32(0)
            cute.arch.mbarrier_wait(mbar + MB_Q, qphase)
            if cutlass.const_expr(self.qtm):
                cute.arch.mbarrier_wait(mbar + MB_QT, qphase)
            qphase = qphase ^ 1

            # --- prologue: fill the two-tile lead ---
            jq = gtq & 1
            cute.arch.mbarrier_wait(mbar + MB_K + sq, sqph)
            tStSi = cute.make_tensor(s_iter + jq * 128, s_layout)
            for kp in cutlass.range_constexpr(nk_qk):
                if cutlass.const_expr(kp < (self.qtm_kb if self.qtm else 0)):
                    qk_mma_ts.set(tcgen05.Field.ACCUMULATE, const_expr(kp != 0))
                    cute.gemm(qk_mma_ts, tStSi, tQblk[kp],
                              tSrK[None, None, kp, sq], tStSi)
                else:
                    qk_mma.set(tcgen05.Field.ACCUMULATE, const_expr(kp != 0))
                    cute.gemm(qk_mma, tStSi, tSrQ[None, None, kp],
                              tSrK[None, None, kp, sq], tStSi)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MB_S + jq, self.pair_mask, self.cta_group)
                tcgen05.commit(mbar + MB_KE + sq, self.kv_rel_mask, self.cta_group)
            sq = sq + 1
            if sq == self.kvstage:
                sq = Int32(0)
                sqph = sqph ^ 1
            gtq = gtq + 1
            # A one-tile work item has no second lead tile.
            if kv_ntile >= 2:
                jq = gtq & 1
                cute.arch.mbarrier_wait(mbar + MB_K + sq, sqph)
                tStSi = cute.make_tensor(s_iter + jq * 128, s_layout)
                for kp in cutlass.range_constexpr(nk_qk):
                    if cutlass.const_expr(kp < (self.qtm_kb if self.qtm else 0)):
                        qk_mma_ts.set(tcgen05.Field.ACCUMULATE, const_expr(kp != 0))
                        cute.gemm(qk_mma_ts, tStSi, tQblk[kp],
                                  tSrK[None, None, kp, sq], tStSi)
                    else:
                        qk_mma.set(tcgen05.Field.ACCUMULATE, const_expr(kp != 0))
                        cute.gemm(qk_mma, tStSi, tSrQ[None, None, kp],
                                  tSrK[None, None, kp, sq], tStSi)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MB_S + jq, self.pair_mask, self.cta_group)
                    tcgen05.commit(mbar + MB_KE + sq, self.kv_rel_mask, self.cta_group)
                sq = sq + 1
                if sq == self.kvstage:
                    sq = Int32(0)
                    sqph = sqph ^ 1
                gtq = gtq + 1
            if wi > 0:
                cute.arch.mbarrier_wait(mbar + MB_EPI, epiphase)
                epiphase = epiphase ^ 1

            # --- steady state: one P*V then one Q*K^T per tile ---
            nmid = kv_ntile - 2
            if nmid < 0:
                nmid = Int32(0)
            for t in cutlass.range(nmid):
                jp = gtp & 1
                cute.arch.mbarrier_wait(mbar + MB_V + sp, spph)
                cute.arch.mbarrier_wait(mbar + MB_P + jp, (gtp >> 1) & 1)
                tOrPi = cute.make_tensor(p_iter + 4 * (self.tmem_s0 + jp * 128), p_layout)
                tOrLi = cute.make_tensor(
                    p_iter + 4 * (self.tmem_s0 + jp * 128 + PCOL_T), p_layout)
                tOrXi = cute.make_tensor(
                    p_iter + 4 * (self.tmem_s0 + jp * 128 + 2 * PCOL_T), p_layout)
                if cutlass.const_expr(self.resid_period != 0):
                    if cutlass.const_expr(self.use_item_mask):
                        use_resid = (gtp & resid_mask) == 0
                    else:
                        use_resid = gtp % self.resid_period == 0
                    if use_resid:
                        for kp in cutlass.range_constexpr(nk_pv):
                            if const_expr(kp == 0):
                                pv_mma_fill.set(tcgen05.Field.ACCUMULATE, t != 0)
                            else:
                                pv_mma_fill.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(pv_mma_fill, tOtO, tOrPi[None, None, kp],
                                      tOrV[None, None, kp, sp], tOtO)
                            pv_mma_last.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(pv_mma_last, tOtO, tOrLi[None, None, kp],
                                      tOrV[None, None, kp, sp], tOtO)
                            if cutlass.const_expr(not self.use_packed):
                                pv_mma_last.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(pv_mma_last, tOtO, tOrXi[None, None, kp],
                                          tOrV[None, None, kp, sp], tOtO)
                        if cutlass.const_expr(self.tsum):
                            with cute.arch.elect_one():
                                tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                               self.cta_group)
                            for tm in cutlass.range_constexpr(nterm_r):
                                tRrPi = cute.make_tensor(
                                    ps_iter + 4 * (self.tmem_s0 + jp * 128 + tm * PCOL_T),
                                    ps_layout)
                                for kq in cutlass.range_constexpr(nk_sum):
                                    if const_expr(kq == 0 and tm == 0):
                                        sum_mma.set(tcgen05.Field.ACCUMULATE, t != 0)
                                    else:
                                        sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                              tOnes[None, None, kq, 0], tRtR)
                    else:
                        for kp in cutlass.range_constexpr(nk_pv):
                            if const_expr(kp == 0):
                                pv_mma.set(tcgen05.Field.ACCUMULATE, t != 0)
                            else:
                                pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(pv_mma, tOtO, tOrPi[None, None, kp],
                                      tOrV[None, None, kp, sp], tOtO)
                        if cutlass.const_expr(self.tsum):
                            with cute.arch.elect_one():
                                tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                               self.cta_group)
                            for tm in cutlass.range_constexpr(1):
                                tRrPi = cute.make_tensor(
                                    ps_iter + 4 * (self.tmem_s0 + jp * 128 + tm * PCOL_T),
                                    ps_layout)
                                for kq in cutlass.range_constexpr(nk_sum):
                                    if const_expr(kq == 0 and tm == 0):
                                        sum_mma.set(tcgen05.Field.ACCUMULATE, t != 0)
                                    else:
                                        sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                              tOnes[None, None, kq, 0], tRtR)
                else:
                    for kp in cutlass.range_constexpr(nk_pv):
                        if const_expr(kp == 0):
                            pv_mma.set(tcgen05.Field.ACCUMULATE, t != 0)
                        else:
                            pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(pv_mma, tOtO, tOrPi[None, None, kp],
                                  tOrV[None, None, kp, sp], tOtO)
                    if cutlass.const_expr(self.tsum):
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                           self.cta_group)
                        for tm in cutlass.range_constexpr(1):
                            tRrPi = cute.make_tensor(
                                ps_iter + 4 * (self.tmem_s0 + jp * 128 + tm * PCOL_T),
                                ps_layout)
                            for kq in cutlass.range_constexpr(nk_sum):
                                if const_expr(kq == 0 and tm == 0):
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, t != 0)
                                else:
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                          tOnes[None, None, kq, 0], tRtR)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MB_PV + jp, self.pair_mask, self.cta_group)
                    if cutlass.const_expr(not self.tsum):
                        tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                       self.cta_group)
                sp = sp + 1
                if sp == self.kvstage:
                    sp = Int32(0)
                    spph = spph ^ 1
                gtp = gtp + 1
                jq = gtq & 1
                cute.arch.mbarrier_wait(mbar + MB_K + sq, sqph)
                tStSi = cute.make_tensor(s_iter + jq * 128, s_layout)
                for kp in cutlass.range_constexpr(nk_qk):
                    if cutlass.const_expr(kp < (self.qtm_kb if self.qtm else 0)):
                        qk_mma_ts.set(tcgen05.Field.ACCUMULATE, const_expr(kp != 0))
                        cute.gemm(qk_mma_ts, tStSi, tQblk[kp],
                                  tSrK[None, None, kp, sq], tStSi)
                    else:
                        qk_mma.set(tcgen05.Field.ACCUMULATE, const_expr(kp != 0))
                        cute.gemm(qk_mma, tStSi, tSrQ[None, None, kp],
                                  tSrK[None, None, kp, sq], tStSi)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MB_S + jq, self.pair_mask, self.cta_group)
                    tcgen05.commit(mbar + MB_KE + sq, self.kv_rel_mask, self.cta_group)
                sq = sq + 1
                if sq == self.kvstage:
                    sq = Int32(0)
                    sqph = sqph ^ 1
                gtq = gtq + 1

            # --- drain: the two lead tiles' P*V ---
            if kv_ntile >= 2:
                jp = gtp & 1
                cute.arch.mbarrier_wait(mbar + MB_V + sp, spph)
                cute.arch.mbarrier_wait(mbar + MB_P + jp, (gtp >> 1) & 1)
                tOrPi = cute.make_tensor(p_iter + 4 * (self.tmem_s0 + jp * 128), p_layout)
                tOrLi = cute.make_tensor(
                    p_iter + 4 * (self.tmem_s0 + jp * 128 + PCOL_T), p_layout)
                tOrXi = cute.make_tensor(
                    p_iter + 4 * (self.tmem_s0 + jp * 128 + 2 * PCOL_T), p_layout)
                if cutlass.const_expr(self.resid_period != 0):
                    if cutlass.const_expr(self.use_item_mask):
                        use_resid = (gtp & resid_mask) == 0
                    else:
                        use_resid = gtp % self.resid_period == 0
                    if use_resid:
                        for kp in cutlass.range_constexpr(nk_pv):
                            if const_expr(kp == 0):
                                pv_mma_fill.set(tcgen05.Field.ACCUMULATE, kv_ntile != 2)
                            else:
                                pv_mma_fill.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(pv_mma_fill, tOtO, tOrPi[None, None, kp],
                                      tOrV[None, None, kp, sp], tOtO)
                            pv_mma_last.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(pv_mma_last, tOtO, tOrLi[None, None, kp],
                                      tOrV[None, None, kp, sp], tOtO)
                            if cutlass.const_expr(not self.use_packed):
                                pv_mma_last.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(pv_mma_last, tOtO, tOrXi[None, None, kp],
                                          tOrV[None, None, kp, sp], tOtO)
                        if cutlass.const_expr(self.tsum):
                            with cute.arch.elect_one():
                                tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                               self.cta_group)
                            for tm in cutlass.range_constexpr(nterm_r):
                                tRrPi = cute.make_tensor(
                                    ps_iter + 4 * (self.tmem_s0 + jp * 128 + tm * PCOL_T),
                                    ps_layout)
                                for kq in cutlass.range_constexpr(nk_sum):
                                    if const_expr(kq == 0 and tm == 0):
                                        sum_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 2)
                                    else:
                                        sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                              tOnes[None, None, kq, 0], tRtR)
                    else:
                        for kp in cutlass.range_constexpr(nk_pv):
                            if const_expr(kp == 0):
                                pv_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 2)
                            else:
                                pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(pv_mma, tOtO, tOrPi[None, None, kp],
                                      tOrV[None, None, kp, sp], tOtO)
                        if cutlass.const_expr(self.tsum):
                            with cute.arch.elect_one():
                                tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                               self.cta_group)
                            for tm in cutlass.range_constexpr(1):
                                tRrPi = cute.make_tensor(
                                    ps_iter + 4 * (self.tmem_s0 + jp * 128 + tm * PCOL_T),
                                    ps_layout)
                                for kq in cutlass.range_constexpr(nk_sum):
                                    if const_expr(kq == 0 and tm == 0):
                                        sum_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 2)
                                    else:
                                        sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                              tOnes[None, None, kq, 0], tRtR)
                else:
                    for kp in cutlass.range_constexpr(nk_pv):
                        if const_expr(kp == 0):
                            pv_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 2)
                        else:
                            pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(pv_mma, tOtO, tOrPi[None, None, kp],
                                  tOrV[None, None, kp, sp], tOtO)
                    if cutlass.const_expr(self.tsum):
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                           self.cta_group)
                        for tm in cutlass.range_constexpr(1):
                            tRrPi = cute.make_tensor(
                                ps_iter + 4 * (self.tmem_s0 + jp * 128 + tm * PCOL_T),
                                ps_layout)
                            for kq in cutlass.range_constexpr(nk_sum):
                                if const_expr(kq == 0 and tm == 0):
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 2)
                                else:
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                          tOnes[None, None, kq, 0], tRtR)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MB_PV + jp, self.pair_mask, self.cta_group)
                    if cutlass.const_expr(not self.tsum):
                        tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                       self.cta_group)
                sp = sp + 1
                if sp == self.kvstage:
                    sp = Int32(0)
                    spph = spph ^ 1
                gtp = gtp + 1

            jp = gtp & 1
            cute.arch.mbarrier_wait(mbar + MB_V + sp, spph)
            cute.arch.mbarrier_wait(mbar + MB_P + jp, (gtp >> 1) & 1)
            tOrPi = cute.make_tensor(p_iter + 4 * (self.tmem_s0 + jp * 128), p_layout)
            tOrLi = cute.make_tensor(
                p_iter + 4 * (self.tmem_s0 + jp * 128 + PCOL_T), p_layout)
            tOrXi = cute.make_tensor(
                p_iter + 4 * (self.tmem_s0 + jp * 128 + 2 * PCOL_T), p_layout)
            if cutlass.const_expr(self.resid_period != 0):
                if cutlass.const_expr(self.use_item_mask):
                    use_resid = (gtp & resid_mask) == 0
                else:
                    use_resid = gtp % self.resid_period == 0
                if use_resid:
                    for kp in cutlass.range_constexpr(nk_pv):
                        if const_expr(kp == 0):
                            pv_mma_fill.set(tcgen05.Field.ACCUMULATE, kv_ntile != 1)
                        else:
                            pv_mma_fill.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(pv_mma_fill, tOtO, tOrPi[None, None, kp],
                                  tOrV[None, None, kp, sp], tOtO)
                        pv_mma_last.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(pv_mma_last, tOtO, tOrLi[None, None, kp],
                                  tOrV[None, None, kp, sp], tOtO)
                        if cutlass.const_expr(not self.use_packed):
                            pv_mma_last.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(pv_mma_last, tOtO, tOrXi[None, None, kp],
                                      tOrV[None, None, kp, sp], tOtO)
                    if cutlass.const_expr(self.tsum):
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                           self.cta_group)
                        for tm in cutlass.range_constexpr(nterm_r):
                            tRrPi = cute.make_tensor(
                                ps_iter + 4 * (self.tmem_s0 + jp * 128 + tm * PCOL_T),
                                ps_layout)
                            for kq in cutlass.range_constexpr(nk_sum):
                                if const_expr(kq == 0 and tm == 0):
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 1)
                                else:
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                          tOnes[None, None, kq, 0], tRtR)
                else:
                    for kp in cutlass.range_constexpr(nk_pv):
                        if const_expr(kp == 0):
                            pv_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 1)
                        else:
                            pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(pv_mma, tOtO, tOrPi[None, None, kp],
                                  tOrV[None, None, kp, sp], tOtO)
                    if cutlass.const_expr(self.tsum):
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                           self.cta_group)
                        for tm in cutlass.range_constexpr(1):
                            tRrPi = cute.make_tensor(
                                ps_iter + 4 * (self.tmem_s0 + jp * 128 + tm * PCOL_T),
                                ps_layout)
                            for kq in cutlass.range_constexpr(nk_sum):
                                if const_expr(kq == 0 and tm == 0):
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 1)
                                else:
                                    sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                          tOnes[None, None, kq, 0], tRtR)
            else:
                for kp in cutlass.range_constexpr(nk_pv):
                    if const_expr(kp == 0):
                        pv_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 1)
                    else:
                        pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(pv_mma, tOtO, tOrPi[None, None, kp],
                              tOrV[None, None, kp, sp], tOtO)
                if cutlass.const_expr(self.tsum):
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                       self.cta_group)
                    for tm in cutlass.range_constexpr(1):
                        tRrPi = cute.make_tensor(
                            ps_iter + 4 * (self.tmem_s0 + jp * 128 + tm * PCOL_T),
                            ps_layout)
                        for kq in cutlass.range_constexpr(nk_sum):
                            if const_expr(kq == 0 and tm == 0):
                                sum_mma.set(tcgen05.Field.ACCUMULATE, kv_ntile != 1)
                            else:
                                sum_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(sum_mma, tRtR, tRrPi[None, None, kq],
                                      tOnes[None, None, kq, 0], tRtR)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MB_PV + jp, self.pair_mask, self.cta_group)
                if cutlass.const_expr(not self.tsum):
                    tcgen05.commit(mbar + MB_VE + sp, self.kv_rel_mask,
                                   self.cta_group)
                tcgen05.commit(mbar + MB_QE, self.pair_mask, self.cta_group)
            sp = sp + 1
            if sp == self.kvstage:
                sp = Int32(0)
                spph = spph ^ 1
            gtp = gtp + 1
        if nit > 0:
            cute.arch.mbarrier_wait(mbar + MB_EPI, epiphase)

    # ------------------------------------------------------ softmax body ---
    def _pack_exp(self, frg64, sc2, sh2, hp):
        """Packed exp2 for one 32-column score slice."""
        for i in range(HALFN // 2):
            hp[i] = _sm_pair(frg64[i], sc2, sh2)

    def _reload_scores(self, tiled_ld, frg_src, shp):
        """Read this tile's scores out of tmem a second time.

        On the row-predicated variants the loaded fp32 score fragment is then
        allowed to die pair by pair inside _pack_exp, so the 16 compute warps
        carry roughly 33 instead of 56 of their 112 registers through the KV
        loop.  Only the rare tile whose lazy anchor moves wants the scores
        again, and it can simply read them again: nothing overwrites the slot
        until this warp stores P into the columns that alias it.  The read is
        issued at the CTA-uniform vcnt level because tcgen05.ld is a
        warp-collective instruction and 'moved' is per row."""
        rfrg = cute.make_rmem_tensor(shp, Float32)
        if cutlass.const_expr(self.use_fused_max):
            rmax = cute.make_rmem_tensor(cute.make_layout(1), Float32)
            cute.copy_atom_call(
                tiled_ld,
                frg_src[((None, 0), 0), 0, 0],
                (rfrg[(None, 0), 0, 0], rmax))
        else:
            cute.copy(tiled_ld, frg_src, rfrg)
        cute.arch.fence_view_async_tmem_load()
        return rfrg

    # ------------------------------------------------------- compute warps ---
    @cute.jit
    def compute_loop(self, qk_mma, pv_mma, tStS0, tOtO, tRtR, sAux, sRed,
                     sMrg, sMrgML,
                     mPlan, mO, mQg, mPO, mPML, mMrg, mMBin, cnt_base, mbar,
                     scale_log2: Float32, out_scale: Float32,
                     w0: Int32, nit: Int32, tidx: Int32, rank: Int32,
                     bidx: Int32):
        # Two warpgroups split the 128-wide S tile by columns; each covers all
        # 128 tmem lanes so every row keeps a single owner thread per half.
        half = tidx // MROW
        lane = tidx % MROW
        NCT = 32 * NCOMPW

        thr_qk = qk_mma.get_slice(0)
        tScS = thr_qk.partition_C(
            cute.make_identity_tensor((self.mma_m, NTILE)))
        tStSh = cute.make_tensor(
            tStS0.iterator, cute.composition(tStS0.layout, cute.make_layout((MROW, HALFN))))
        tScSh = cute.make_tensor(
            tScS.iterator, cute.composition(tScS.layout, cute.make_layout((MROW, HALFN))))
        if cutlass.const_expr(self.use_fused_max):
            tmem_ld_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.Repetition(32), redOp=tcgen05.TmemLoadRedOp.MAX),
                Float32)
        else:
            tmem_ld_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32)
        tiled_ld = tcgen05.make_tmem_copy(tmem_ld_atom, tStSh)
        thr_ld = tiled_ld.get_slice(lane)
        tScS_ld = thr_ld.partition_D(tScSh)
        row = tScS_ld[0][0]
        s_src = thr_ld.partition_S(tStSh)
        s_src_layout = s_src.layout
        s_src_iter = s_src.iterator

        PCH = HALFN // 32 * 8          # tmem columns of one fp8 P half
        tStPh = cute.make_tensor(
            tStS0.iterator, cute.composition(tStS0.layout, cute.make_layout((MROW, PCH))))
        tScPh = cute.make_tensor(
            tScS.iterator, cute.composition(tScS.layout, cute.make_layout((MROW, PCH))))
        tmem_st_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(HALFN // 32 * 8)), Float32)
        tiled_st = tcgen05.make_tmem_copy(tmem_st_atom, tStPh)
        thr_st = tiled_st.get_slice(lane)
        tScP_st = thr_st.partition_S(tScPh)
        p_dst = thr_st.partition_D(tStPh)
        p_dst_layout = p_dst.layout
        p_dst_iter = p_dst.iterator

        tOtO_c = cute.make_tensor(
            tOtO.iterator, cute.composition(tOtO.layout, cute.make_layout((MROW, CHUNK))))
        tOcO = cute.make_tensor(
            tScS.iterator, cute.composition(tScS.layout, cute.make_layout((MROW, CHUNK))))
        o_ld_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(CHUNK)), Float32)
        o_tiled_ld = tcgen05.make_tmem_copy(o_ld_atom, tOtO_c)
        o_thr_ld = o_tiled_ld.get_slice(lane)
        o_ld_c = o_thr_ld.partition_D(tOcO)
        o_st_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(CHUNK)), Float32)
        o_tiled_st = tcgen05.make_tmem_copy(o_st_atom, tOtO_c)
        o_thr_st = o_tiled_st.get_slice(lane)

        tOtO_r = cute.make_tensor(
            tOtO.iterator, cute.composition(tOtO.layout, cute.make_layout((MROW, self.corrch))))
        tOcO_r = cute.make_tensor(
            tScS.iterator, cute.composition(tScS.layout, cute.make_layout((MROW, self.corrch))))
        r_tiled_ld = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(self.corrch)), Float32), tOtO_r)
        r_thr_ld = r_tiled_ld.get_slice(lane)
        r_ld_c = r_thr_ld.partition_D(tOcO_r)
        r_tiled_st = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(self.corrch)), Float32), tOtO_r)
        r_thr_st = r_tiled_st.get_slice(lane)

        tRtR_c = cute.make_tensor(
            tRtR.iterator, cute.composition(tRtR.layout, cute.make_layout((MROW, 1))))
        tRcR = cute.make_tensor(
            tScS.iterator, cute.composition(tScS.layout, cute.make_layout((MROW, 1))))
        sum_tiled_ld = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(1)), Float32), tRtR_c)
        sum_thr_ld = sum_tiled_ld.get_slice(lane)
        sum_ld_c = sum_thr_ld.partition_D(tRcR)
        sum_tiled_st = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(1)), Float32), tRtR_c)
        sum_thr_st = sum_tiled_st.get_slice(lane)

        NCHUNK = HDIM // CHUNK // NSEG   # O column chunks owned by one segment
        cbase = half * NCHUNK
        jrow = row >> 3
        grow = row & 7
        neg = Float32(NEG_BIG)
        sc2 = _pack2(scale_log2, scale_log2)
        zero2 = _pack2(Float32(0.0), Float32(0.0))
        tau_raw = Float32(self.corr_tau) / scale_log2
        gt = Int32(0)
        for wi in cutlass.range(nit):
            w = w0 + wi
            tok_base = mPlan[w, 0]
            n_tok = mPlan[w, 1]
            h_kv = mPlan[w, 2]
            kv_begin = mPlan[w, 4]
            kv_ntile = mPlan[w, 5]
            resid_mask = mPlan[w, 11]
            if cutlass.const_expr(not self.use_packed):
                resid_mask = Int32(0)
            causal = mPlan[w, 6]
            slot = mPlan[w, 7]
            slot_base = mPlan[w, 8]
            nsplit = mPlan[w, 9]
            grp = mPlan[w, 10]
            if cutlass.const_expr(self.clustered):
                # A plan row carries a 32-token q-tile; this CTA owns the
                # 16-token half selected by its rank in the cluster.
                if n_tok > MROW // GROUP:
                    tok_base = tok_base + rank * (MROW // GROUP)
                    causal = causal + rank * (MROW // GROUP)
                n_tok = n_tok - rank * (MROW // GROUP)
                if n_tok > MROW // GROUP:
                    n_tok = Int32(MROW // GROUP)
                if slot >= 0:
                    slot = slot + rank * nsplit
                    slot_base = slot_base + rank * nsplit
                    grp = grp * 2 + rank

            if cutlass.const_expr(self.qtm):
                # Every Q*K^T of the previous item completed before its last
                # S was read, so the tmem Q columns are free.  Row = lane holds
                # token row >> 3, grouped head row & 7; this thread writes the
                # 32 head-dim bytes [32 * half, 32 * half + 32) as 8 columns.
                qfrg = cute.make_rmem_tensor(cute.make_layout(32),
                                             cutlass.Float8E4M3FN)
                q32 = cute.make_tensor(
                    cute.recast_ptr(qfrg.iterator, dtype=Int32),
                    cute.make_layout(8))
                for i in cutlass.range_constexpr(8):
                    q32[i] = Int32(0)
                if jrow < n_tok:
                    cute.autovec_copy(
                        mQg[((grow, tok_base + jrow), (None, half), h_kv)], qfrg)
                qf32 = cute.make_tensor(
                    cute.recast_ptr(qfrg.iterator, dtype=Float32), tScP_st.shape)
                cute.copy(tiled_st, qf32,
                          cute.make_tensor(
                              p_dst_iter + (self.tmem_q - self.tmem_s0) + half * PCH,
                              p_dst_layout))
                if cutlass.const_expr(self.qtm_kb > 2):
                    # Head dims 128..191 are two more 32-byte chunks; the
                    # first two column segments of each lane group take them.
                    if half < 2:
                        for i in cutlass.range_constexpr(8):
                            q32[i] = Int32(0)
                        if jrow < n_tok:
                            cute.autovec_copy(
                                mQg[((grow, tok_base + jrow), (None, half + 4), h_kv)],
                                qfrg)
                        cute.copy(tiled_st, qf32,
                                  cute.make_tensor(
                                      p_dst_iter + (self.tmem_q - self.tmem_s0)
                                      + (half + 4) * PCH,
                                      p_dst_layout))
                cute.arch.fence_view_async_tmem_store()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mbar + MB_QT, Int32(0))

            row_safe = Float32(0.0)
            row_sum = Float32(0.0)
            # Four independent packed-fp32 chains break the 16-promotion
            # dependency chain.  They live for the whole work item, so the
            # horizontal reduction is paid once rather than once per KV tile.
            if cutlass.const_expr(not self.tsum):
                rs2a = zero2
                rs2b = zero2
                rs2c = zero2
                rs2d = zero2
            thr = Float32(NEG_BIG)
            shift = Float32(self.p_prescale)
            sh2 = _pack2(shift, shift)

            # Only the last q_len keys of a request are masked, so every tile
            # ending at or before causal - 127 is dense for all 512 lanes.
            # Compiling that prefix without the mask stops ptxas if-converting
            # 32 selects and a 32-wide max into every KV tile of the stream;
            # there the fused tmem-load reduction already supplies pmax.
            dfull = causal - kv_begin - (NTILE - 1)
            nfull = Int32(0)
            if dfull >= 0:
                nfull = (dfull >> 7) + 1
            if nfull > kv_ntile:
                nfull = kv_ntile

            for t in cutlass.range(nfull):
                j = gt & 1
                ph = (gt >> 1) & 1
                cute.arch.mbarrier_wait(mbar + MB_S + j, ph)
                frg = cute.make_rmem_tensor(tScS_ld.shape, Float32)
                frg_src = cute.make_tensor(
                    s_src_iter + j * 128 + half * HALFN, s_src_layout)
                if cutlass.const_expr(self.use_fused_max):
                    frg_max = cute.make_rmem_tensor(cute.make_layout(1), Float32)
                    cute.copy_atom_call(
                        tiled_ld,
                        frg_src[((None, 0), 0), 0, 0],
                        (frg[(None, 0), 0, 0], frg_max))
                    pmax = frg_max[0]
                else:
                    cute.copy(tiled_ld, frg_src, frg)
                cute.arch.fence_view_async_tmem_load()

                corr = Float32(1.0)
                if cutlass.const_expr(self.use_packed):
                    # The running row max moves on the first tile of a work item
                    # and essentially never again, so run the whole exp2/pack
                    # chain on the shift carried in from the previous tile and
                    # only then take the CTA rendezvous that guards the aliased
                    # tmem P columns.  That lifts the barrier out of the
                    # tmem-load -> exp2 -> tmem-store critical path and overlaps
                    # it with the other warps' arithmetic; the rare tile that
                    # really does move the max just redoes the chain.
                    frg64 = cute.make_tensor(
                        cute.recast_ptr(frg.iterator, dtype=Int64),
                        cute.make_layout(HALFN // 2))
                    hp = cute.make_rmem_tensor(cute.make_layout(HALFN // 2), Int32)
                    pfrg = cute.make_rmem_tensor(tScS_ld.shape, cutlass.Float8E4M3FN)
                    p32 = cute.make_tensor(
                        cute.recast_ptr(pfrg.iterator, dtype=Int32),
                        cute.make_layout(HALFN // 4))
                    self._pack_exp(frg64, sc2, sh2, hp)
                    for i in cutlass.range_constexpr(HALFN // 4):
                        p32[i] = _cvt_e4m3x4(hp[2 * i], hp[2 * i + 1])
                    vcnt = _bar_red_gt(pmax, thr)
                    if vcnt != 0:
                        sRed[tidx] = pmax
                        cute.arch.barrier(barrier_id=1, number_of_threads=NCT)
                        tile_max = sRed[lane]
                        for q in cutlass.range_constexpr(1, NSEG):
                            other = sRed[lane + q * MROW]
                            if other > tile_max:
                                tile_max = other
                        moved = tile_max > thr
                        if moved:
                            old_safe = row_safe
                            row_safe = tile_max
                            if tile_max < Float32(0.5 * NEG_BIG):
                                row_safe = Float32(0.0)
                            thr = tile_max + tau_raw
                            shift = Float32(0.0) - row_safe * scale_log2 + Float32(
                                self.p_prescale)
                            sh2 = _pack2(shift, shift)
                            corr = cute.math.exp2(
                                (old_safe - row_safe) * scale_log2, fastmath=True)
                            if cutlass.const_expr(not self.tsum):
                                c2 = _pack2(corr, corr)
                                rs2a = _mul_f32x2(rs2a, c2)
                                rs2b = _mul_f32x2(rs2b, c2)
                                rs2c = _mul_f32x2(rs2c, c2)
                                rs2d = _mul_f32x2(rs2d, c2)
                        if cutlass.const_expr(self.row_pred):
                            rfrg = self._reload_scores(
                                tiled_ld, frg_src, tScS_ld.shape)
                            rfrg64 = cute.make_tensor(
                                cute.recast_ptr(rfrg.iterator, dtype=Int64),
                                cute.make_layout(HALFN // 2))
                            if moved:
                                self._pack_exp(rfrg64, sc2, sh2, hp)
                                for i in cutlass.range_constexpr(HALFN // 4):
                                    p32[i] = _cvt_e4m3x4(
                                        hp[2 * i], hp[2 * i + 1])
                        else:
                            self._pack_exp(frg64, sc2, sh2, hp)
                            for i in cutlass.range_constexpr(HALFN // 4):
                                p32[i] = _cvt_e4m3x4(
                                    hp[2 * i], hp[2 * i + 1])
                else:
                    vcnt = _bar_red_gt(pmax, thr)
                    if vcnt != 0:
                        sRed[tidx] = pmax
                        cute.arch.barrier(barrier_id=1, number_of_threads=NCT)
                        tile_max = sRed[lane]
                        for q in cutlass.range_constexpr(1, NSEG):
                            other = sRed[lane + q * MROW]
                            if other > tile_max:
                                tile_max = other
                        moved = tile_max > thr
                        if moved:
                            old_safe = row_safe
                            row_safe = tile_max
                            if tile_max < Float32(0.5 * NEG_BIG):
                                row_safe = Float32(0.0)
                            thr = tile_max + tau_raw
                            shift = Float32(0.0) - row_safe * scale_log2 + Float32(
                                self.p_prescale)
                            sh2 = _pack2(shift, shift)
                            corr = cute.math.exp2(
                                (old_safe - row_safe) * scale_log2, fastmath=True)
                if cutlass.const_expr(not self.use_packed):
                    sh2 = _pack2(shift, shift)
                    frg64 = cute.make_tensor(
                        cute.recast_ptr(frg.iterator, dtype=Int64),
                        cute.make_layout(HALFN // 2))
                    for i in cutlass.range_constexpr(HALFN // 2):
                        frg64[i] = _affine_pair(frg64[i], sc2, sh2)
                    for i in cutlass.range_constexpr(HALFN):
                        frg[i] = cute.math.exp2(frg[i], fastmath=True)
                    if cutlass.const_expr(not self.tsum):
                        phalf = frg.load().reduce(
                            cute.ReductionOp.ADD, Float32(0.0), 0)
                        row_sum = row_sum * corr + phalf

                if cutlass.const_expr(not self.use_packed):
                    pfrg = cute.make_rmem_tensor(tScS_ld.shape, cutlass.Float8E4M3FN)
                    pfrg.store(frg.load().to(cutlass.Float8E4M3FN))
                pf32 = cute.make_tensor(
                    cute.recast_ptr(pfrg.iterator, dtype=Float32), tScP_st.shape)
                cute.copy(tiled_st, pf32,
                          cute.make_tensor(p_dst_iter + j * 128 + half * PCH, p_dst_layout))
                if cutlass.const_expr(self.resid_period != 0):
                    if cutlass.const_expr(self.use_item_mask):
                        use_resid = (gt & resid_mask) == 0
                    else:
                        use_resid = gt % self.resid_period == 0
                    if use_resid:
                        lfrg = cute.make_rmem_tensor(tScS_ld.shape, cutlass.Float8E4M3FN)
                        if cutlass.const_expr(self.use_packed):
                            l32 = cute.make_tensor(
                                cute.recast_ptr(lfrg.iterator, dtype=Int32),
                                cute.make_layout(HALFN // 4))
                            for i in cutlass.range_constexpr(HALFN // 4):
                                l32[i] = _resid_e4m3x4(
                                    hp[2 * i], hp[2 * i + 1], p32[i])
                        else:
                            fsub = cute.logical_divide(frg, cute.make_layout(32))
                            psub = cute.logical_divide(pfrg, cute.make_layout(32))
                            for c in cutlass.range_constexpr(HALFN // 32):
                                fsub[None, c].store(
                                    fsub[None, c].load() -
                                    psub[None, c].load().to(Float32))
                            lfrg.store(frg.load().to(cutlass.Float8E4M3FN))
                        lf32 = cute.make_tensor(
                            cute.recast_ptr(lfrg.iterator, dtype=Float32), tScP_st.shape)
                        cute.copy(tiled_st, lf32,
                                  cute.make_tensor(
                                      p_dst_iter + j * 128 + PCOL_T + half * PCH,
                                      p_dst_layout))
                        if cutlass.const_expr(not self.use_packed):
                            xfrg = cute.make_rmem_tensor(
                                tScS_ld.shape, cutlass.Float8E4M3FN)
                            fsub = cute.logical_divide(frg, cute.make_layout(32))
                            lsub = cute.logical_divide(lfrg, cute.make_layout(32))
                            for c in cutlass.range_constexpr(HALFN // 32):
                                fsub[None, c].store(
                                    fsub[None, c].load() -
                                    lsub[None, c].load().to(Float32))
                            xfrg.store(frg.load().to(cutlass.Float8E4M3FN))
                            xf32 = cute.make_tensor(
                                cute.recast_ptr(xfrg.iterator, dtype=Float32),
                                tScP_st.shape)
                            cute.copy(
                                tiled_st, xf32,
                                cute.make_tensor(
                                    p_dst_iter + j * 128 + 2 * PCOL_T + half * PCH,
                                    p_dst_layout))
                cute.arch.fence_view_async_tmem_store()

                if vcnt != 0:
                    if cutlass.const_expr(LVL & 2) and t != 0:
                        if cute.arch.vote_ballot_sync(corr < Float32(1.0)) != 0:
                            cute.arch.mbarrier_wait(
                                mbar + MB_PV + ((gt - 1) & 1), ((gt - 1) >> 1) & 1)
                            for c in cutlass.range_constexpr(
                                    NCHUNK * (CHUNK // self.corrch)):
                                ofrg = cute.make_rmem_tensor(r_ld_c.shape, Float32)
                                src = cute.make_tensor(
                                    tOtO_r.iterator + cbase * CHUNK + c * self.corrch,
                                    tOtO_r.layout)
                                cute.copy(r_tiled_ld, r_thr_ld.partition_S(src), ofrg)
                                cute.arch.fence_view_async_tmem_load()
                                for i in cutlass.range_constexpr(self.corrch):
                                    ofrg[i] = ofrg[i] * corr
                                cute.copy(r_tiled_st, ofrg, r_thr_st.partition_D(src))
                            if cutlass.const_expr(self.tsum):
                                # One owner per row, or the shared accumulator
                                # would be scaled once per column segment.
                                if half == 0:
                                    rsf = cute.make_rmem_tensor(
                                        sum_ld_c.shape, Float32)
                                    cute.copy(sum_tiled_ld,
                                              sum_thr_ld.partition_S(tRtR_c), rsf)
                                    cute.arch.fence_view_async_tmem_load()
                                    rsf[0] = rsf[0] * corr
                                    cute.copy(sum_tiled_st, rsf,
                                              sum_thr_st.partition_D(tRtR_c))
                            cute.arch.fence_view_async_tmem_store()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    if cutlass.const_expr(self.twocta):
                        cute.arch.mbarrier_arrive(mbar + MB_P + j, Int32(0))
                    else:
                        cute.arch.mbarrier_arrive(mbar + MB_P + j)
                if cutlass.const_expr(self.use_packed and not self.tsum):
                    # The row sum is only read when the work item ends, so every
                    # instruction of it placed before the P release is pure added
                    # latency on the one chain that gates the next PV MMA.
                    for i in cutlass.range_constexpr(HALFN // 8):
                        rs2a = _acc_f16x2(rs2a, hp[4 * i])
                        rs2b = _acc_f16x2(rs2b, hp[4 * i + 1])
                        rs2c = _acc_f16x2(rs2c, hp[4 * i + 2])
                        rs2d = _acc_f16x2(rs2d, hp[4 * i + 3])
                gt = gt + 1
            for tmask in cutlass.range(kv_ntile - nfull):
                t = nfull + tmask
                j = gt & 1
                ph = (gt >> 1) & 1
                k0 = kv_begin + t * NTILE
                cute.arch.mbarrier_wait(mbar + MB_S + j, ph)
                frg = cute.make_rmem_tensor(tScS_ld.shape, Float32)
                frg_src = cute.make_tensor(
                    s_src_iter + j * 128 + half * HALFN, s_src_layout)
                if cutlass.const_expr(self.use_fused_max):
                    frg_max = cute.make_rmem_tensor(cute.make_layout(1), Float32)
                    cute.copy_atom_call(
                        tiled_ld,
                        frg_src[((None, 0), 0), 0, 0],
                        (frg[(None, 0), 0, 0], frg_max))
                    pmax = frg_max[0]
                else:
                    cute.copy(tiled_ld, frg_src, frg)
                cute.arch.fence_view_async_tmem_load()

                lim = causal + jrow - k0 + 1 - half * HALFN
                if lim < HALFN:
                    for i in cutlass.range_constexpr(HALFN):
                        frg[i] = frg[i] if i < lim else neg

                if cutlass.const_expr(not self.use_fused_max):
                    pmax = frg.load().reduce(cute.ReductionOp.MAX, Float32(NEG_BIG), 0)
                else:
                    if lim < HALFN:
                        pmax = frg.load().reduce(
                            cute.ReductionOp.MAX, Float32(NEG_BIG), 0)

                corr = Float32(1.0)
                if cutlass.const_expr(self.use_packed):
                    # The running row max moves on the first tile of a work item
                    # and essentially never again, so run the whole exp2/pack
                    # chain on the shift carried in from the previous tile and
                    # only then take the CTA rendezvous that guards the aliased
                    # tmem P columns.  That lifts the barrier out of the
                    # tmem-load -> exp2 -> tmem-store critical path and overlaps
                    # it with the other warps' arithmetic; the rare tile that
                    # really does move the max just redoes the chain.
                    frg64 = cute.make_tensor(
                        cute.recast_ptr(frg.iterator, dtype=Int64),
                        cute.make_layout(HALFN // 2))
                    hp = cute.make_rmem_tensor(cute.make_layout(HALFN // 2), Int32)
                    pfrg = cute.make_rmem_tensor(tScS_ld.shape, cutlass.Float8E4M3FN)
                    p32 = cute.make_tensor(
                        cute.recast_ptr(pfrg.iterator, dtype=Int32),
                        cute.make_layout(HALFN // 4))
                    self._pack_exp(frg64, sc2, sh2, hp)
                    for i in cutlass.range_constexpr(HALFN // 4):
                        p32[i] = _cvt_e4m3x4(hp[2 * i], hp[2 * i + 1])
                    vcnt = _bar_red_gt(pmax, thr)
                    if vcnt != 0:
                        sRed[tidx] = pmax
                        cute.arch.barrier(barrier_id=1, number_of_threads=NCT)
                        tile_max = sRed[lane]
                        for q in cutlass.range_constexpr(1, NSEG):
                            other = sRed[lane + q * MROW]
                            if other > tile_max:
                                tile_max = other
                        moved = tile_max > thr
                        if moved:
                            old_safe = row_safe
                            row_safe = tile_max
                            if tile_max < Float32(0.5 * NEG_BIG):
                                row_safe = Float32(0.0)
                            thr = tile_max + tau_raw
                            shift = Float32(0.0) - row_safe * scale_log2 + Float32(
                                self.p_prescale)
                            sh2 = _pack2(shift, shift)
                            corr = cute.math.exp2(
                                (old_safe - row_safe) * scale_log2, fastmath=True)
                            if cutlass.const_expr(not self.tsum):
                                c2 = _pack2(corr, corr)
                                rs2a = _mul_f32x2(rs2a, c2)
                                rs2b = _mul_f32x2(rs2b, c2)
                                rs2c = _mul_f32x2(rs2c, c2)
                                rs2d = _mul_f32x2(rs2d, c2)
                        if cutlass.const_expr(self.row_pred):
                            rfrg = self._reload_scores(
                                tiled_ld, frg_src, tScS_ld.shape)
                            if lim < HALFN:
                                for i in cutlass.range_constexpr(HALFN):
                                    rfrg[i] = rfrg[i] if i < lim else neg
                            rfrg64 = cute.make_tensor(
                                cute.recast_ptr(rfrg.iterator, dtype=Int64),
                                cute.make_layout(HALFN // 2))
                            if moved:
                                self._pack_exp(rfrg64, sc2, sh2, hp)
                                for i in cutlass.range_constexpr(HALFN // 4):
                                    p32[i] = _cvt_e4m3x4(
                                        hp[2 * i], hp[2 * i + 1])
                        else:
                            self._pack_exp(frg64, sc2, sh2, hp)
                            for i in cutlass.range_constexpr(HALFN // 4):
                                p32[i] = _cvt_e4m3x4(
                                    hp[2 * i], hp[2 * i + 1])
                else:
                    vcnt = _bar_red_gt(pmax, thr)
                    if vcnt != 0:
                        sRed[tidx] = pmax
                        cute.arch.barrier(barrier_id=1, number_of_threads=NCT)
                        tile_max = sRed[lane]
                        for q in cutlass.range_constexpr(1, NSEG):
                            other = sRed[lane + q * MROW]
                            if other > tile_max:
                                tile_max = other
                        if tile_max > thr:
                            old_safe = row_safe
                            row_safe = tile_max
                            if tile_max < Float32(0.5 * NEG_BIG):
                                row_safe = Float32(0.0)
                            thr = tile_max + tau_raw
                            shift = Float32(0.0) - row_safe * scale_log2 + Float32(
                                self.p_prescale)
                            sh2 = _pack2(shift, shift)
                            corr = cute.math.exp2(
                                (old_safe - row_safe) * scale_log2, fastmath=True)
                if cutlass.const_expr(not self.use_packed):
                    sh2 = _pack2(shift, shift)
                    frg64 = cute.make_tensor(
                        cute.recast_ptr(frg.iterator, dtype=Int64),
                        cute.make_layout(HALFN // 2))
                    for i in cutlass.range_constexpr(HALFN // 2):
                        frg64[i] = _affine_pair(frg64[i], sc2, sh2)
                    for i in cutlass.range_constexpr(HALFN):
                        frg[i] = cute.math.exp2(frg[i], fastmath=True)
                    if cutlass.const_expr(not self.tsum):
                        phalf = frg.load().reduce(
                            cute.ReductionOp.ADD, Float32(0.0), 0)
                        row_sum = row_sum * corr + phalf

                if cutlass.const_expr(not self.use_packed):
                    pfrg = cute.make_rmem_tensor(tScS_ld.shape, cutlass.Float8E4M3FN)
                    pfrg.store(frg.load().to(cutlass.Float8E4M3FN))
                pf32 = cute.make_tensor(
                    cute.recast_ptr(pfrg.iterator, dtype=Float32), tScP_st.shape)
                cute.copy(tiled_st, pf32,
                          cute.make_tensor(p_dst_iter + j * 128 + half * PCH, p_dst_layout))
                if cutlass.const_expr(self.resid_period != 0):
                    if cutlass.const_expr(self.use_item_mask):
                        use_resid = (gt & resid_mask) == 0
                    else:
                        use_resid = gt % self.resid_period == 0
                    if use_resid:
                        lfrg = cute.make_rmem_tensor(tScS_ld.shape, cutlass.Float8E4M3FN)
                        if cutlass.const_expr(self.use_packed):
                            l32 = cute.make_tensor(
                                cute.recast_ptr(lfrg.iterator, dtype=Int32),
                                cute.make_layout(HALFN // 4))
                            for i in cutlass.range_constexpr(HALFN // 4):
                                l32[i] = _resid_e4m3x4(
                                    hp[2 * i], hp[2 * i + 1], p32[i])
                        else:
                            fsub = cute.logical_divide(frg, cute.make_layout(32))
                            psub = cute.logical_divide(pfrg, cute.make_layout(32))
                            for c in cutlass.range_constexpr(HALFN // 32):
                                fsub[None, c].store(
                                    fsub[None, c].load() -
                                    psub[None, c].load().to(Float32))
                            lfrg.store(frg.load().to(cutlass.Float8E4M3FN))
                        lf32 = cute.make_tensor(
                            cute.recast_ptr(lfrg.iterator, dtype=Float32), tScP_st.shape)
                        cute.copy(tiled_st, lf32,
                                  cute.make_tensor(
                                      p_dst_iter + j * 128 + PCOL_T + half * PCH,
                                      p_dst_layout))
                        if cutlass.const_expr(not self.use_packed):
                            xfrg = cute.make_rmem_tensor(
                                tScS_ld.shape, cutlass.Float8E4M3FN)
                            fsub = cute.logical_divide(frg, cute.make_layout(32))
                            lsub = cute.logical_divide(lfrg, cute.make_layout(32))
                            for c in cutlass.range_constexpr(HALFN // 32):
                                fsub[None, c].store(
                                    fsub[None, c].load() -
                                    lsub[None, c].load().to(Float32))
                            xfrg.store(frg.load().to(cutlass.Float8E4M3FN))
                            xf32 = cute.make_tensor(
                                cute.recast_ptr(xfrg.iterator, dtype=Float32),
                                tScP_st.shape)
                            cute.copy(
                                tiled_st, xf32,
                                cute.make_tensor(
                                    p_dst_iter + j * 128 + 2 * PCOL_T + half * PCH,
                                    p_dst_layout))
                cute.arch.fence_view_async_tmem_store()

                if vcnt != 0:
                    if cutlass.const_expr(LVL & 2) and t != 0:
                        if cute.arch.vote_ballot_sync(corr < Float32(1.0)) != 0:
                            cute.arch.mbarrier_wait(
                                mbar + MB_PV + ((gt - 1) & 1), ((gt - 1) >> 1) & 1)
                            for c in cutlass.range_constexpr(
                                    NCHUNK * (CHUNK // self.corrch)):
                                ofrg = cute.make_rmem_tensor(r_ld_c.shape, Float32)
                                src = cute.make_tensor(
                                    tOtO_r.iterator + cbase * CHUNK + c * self.corrch,
                                    tOtO_r.layout)
                                cute.copy(r_tiled_ld, r_thr_ld.partition_S(src), ofrg)
                                cute.arch.fence_view_async_tmem_load()
                                for i in cutlass.range_constexpr(self.corrch):
                                    ofrg[i] = ofrg[i] * corr
                                cute.copy(r_tiled_st, ofrg, r_thr_st.partition_D(src))
                            if cutlass.const_expr(self.tsum):
                                # One owner per row, or the shared accumulator
                                # would be scaled once per column segment.
                                if half == 0:
                                    rsf = cute.make_rmem_tensor(
                                        sum_ld_c.shape, Float32)
                                    cute.copy(sum_tiled_ld,
                                              sum_thr_ld.partition_S(tRtR_c), rsf)
                                    cute.arch.fence_view_async_tmem_load()
                                    rsf[0] = rsf[0] * corr
                                    cute.copy(sum_tiled_st, rsf,
                                              sum_thr_st.partition_D(tRtR_c))
                            cute.arch.fence_view_async_tmem_store()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    if cutlass.const_expr(self.twocta):
                        cute.arch.mbarrier_arrive(mbar + MB_P + j, Int32(0))
                    else:
                        cute.arch.mbarrier_arrive(mbar + MB_P + j)
                if cutlass.const_expr(self.use_packed and not self.tsum):
                    # The row sum is only read when the work item ends, so every
                    # instruction of it placed before the P release is pure added
                    # latency on the one chain that gates the next PV MMA.
                    for i in cutlass.range_constexpr(HALFN // 8):
                        rs2a = _acc_f16x2(rs2a, hp[4 * i])
                        rs2b = _acc_f16x2(rs2b, hp[4 * i + 1])
                        rs2c = _acc_f16x2(rs2c, hp[4 * i + 2])
                        rs2d = _acc_f16x2(rs2d, hp[4 * i + 3])
                gt = gt + 1

            if cutlass.const_expr(self.tsum):
                # The UMMA already reduced all 128 columns of every tile into
                # one tmem column, so the per-item cross-segment shared-memory
                # reduction and its two CTA rendezvous disappear with it.
                cute.arch.mbarrier_wait(
                    mbar + MB_PV + ((gt - 1) & 1), ((gt - 1) >> 1) & 1)
                rfrg = cute.make_rmem_tensor(sum_ld_c.shape, Float32)
                cute.copy(sum_tiled_ld, sum_thr_ld.partition_S(tRtR_c), rfrg)
                cute.arch.fence_view_async_tmem_load()
                row_sum = rfrg[0]
            else:
                if cutlass.const_expr(self.use_packed):
                    row_sum = _hadd_f32x2(
                        _add_f32x2(_add_f32x2(rs2a, rs2b),
                                   _add_f32x2(rs2c, rs2d)))
                sRed[tidx] = row_sum
                cute.arch.barrier(barrier_id=1, number_of_threads=NCT)
                row_sum = sRed[lane]
                for q in cutlass.range_constexpr(1, NSEG):
                    row_sum = row_sum + sRed[lane + q * MROW]
                cute.arch.barrier(barrier_id=1, number_of_threads=NCT)
                cute.arch.mbarrier_wait(
                    mbar + MB_PV + ((gt - 1) & 1), ((gt - 1) >> 1) & 1)
            if slot < 0:
                rinv = out_scale / row_sum
                rinv2 = _pack2(rinv, rinv)
                for c in cutlass.range_constexpr(NCHUNK):
                    ofrg = cute.make_rmem_tensor(o_ld_c.shape, Float32)
                    src = cute.make_tensor(
                        tOtO_c.iterator + (cbase + c) * CHUNK, tOtO_c.layout)
                    cute.copy(o_tiled_ld, o_thr_ld.partition_S(src), ofrg)
                    cute.arch.fence_view_async_tmem_load()
                    if jrow < n_tok:
                        o64 = cute.make_tensor(
                            cute.recast_ptr(ofrg.iterator, dtype=Int64),
                            cute.make_layout(CHUNK // 2))
                        for v in cutlass.range_constexpr(CHUNK // 8):
                            obf = cute.make_rmem_tensor(cute.make_layout(8), cutlass.BFloat16)
                            o32 = cute.make_tensor(
                                cute.recast_ptr(obf.iterator, dtype=Int32),
                                cute.make_layout(4))
                            for i in cutlass.range_constexpr(4):
                                o32[i] = _scale_cvt_bf16x2(o64[v * 4 + i], rinv2)
                            cute.autovec_copy(
                                obf, mO[((grow, tok_base + jrow),
                                        (None, (cbase + c) * (CHUNK // 8) + v), h_kv)])
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    if cutlass.const_expr(self.twocta):
                        cute.arch.mbarrier_arrive(mbar + MB_EPI, Int32(0))
                    else:
                        cute.arch.mbarrier_arrive(mbar + MB_EPI)
            else:
                for c in cutlass.range_constexpr(NCHUNK):
                    ofrg = cute.make_rmem_tensor(o_ld_c.shape, Float32)
                    src = cute.make_tensor(
                        tOtO_c.iterator + (cbase + c) * CHUNK, tOtO_c.layout)
                    cute.copy(o_tiled_ld, o_thr_ld.partition_S(src), ofrg)
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(self.po_bf16):
                        for i in cutlass.range_constexpr(CHUNK):
                            mPO[row, (cbase + c) * CHUNK + i, slot] = ofrg[i].to(
                                cutlass.BFloat16)
                    else:
                        osub = cute.logical_divide(ofrg, cute.make_layout(POV))
                        cg0 = ((cbase + c) * CHUNK) // POV
                        for v in cutlass.range_constexpr(CHUNK // POV):
                            cute.autovec_copy(
                                osub[None, v], mPO[row, (None, cg0 + v), slot])
                if half == 0:
                    mPML[row, 0, slot] = row_safe
                    mPML[row, 1, slot] = row_sum
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    if cutlass.const_expr(self.twocta):
                        cute.arch.mbarrier_arrive(mbar + MB_EPI, Int32(0))
                    else:
                        cute.arch.mbarrier_arrive(mbar + MB_EPI)
                cute.arch.barrier(barrier_id=1, number_of_threads=NCT)
                if tidx == 0:
                    # Counts arrivals without wrapping: the merge tasks poll it
                    # for == nsplit and the last of them stores it back to 0.
                    sAux[0] = _atom_inc(cnt_base + Int64(grp) * Int64(8),
                                        Int32(BIGLIM))
                cute.arch.barrier(barrier_id=1, number_of_threads=NCT)

        # ------------------------------------------------- split merge ------
        # The partials of one split group are reduced by NMRG CTAs, each owning
        # a disjoint MCOL-wide column band of the 128-row tile, instead of by
        # the single last-arriving CTA.  A group's merge is a plan row of its
        # own, placed after every streaming row this CTA owns, so a CTA never
        # blocks on a group while still holding work another CTA waits for.
        if cutlass.const_expr(LVL & 8):
            MPT = const_expr(MCOL // NSEG)       # merge columns per thread
            m0 = mMBin[bidx]
            nmt = mMBin[bidx + 1] - m0
            for mt in cutlass.range(nmt):
                m = m0 + mt
                tok_base = mMrg[m, 0]
                n_tok = mMrg[m, 1]
                h_kv = mMrg[m, 2]
                slot_base = mMrg[m, 3]
                nsplit = mMrg[m, 4]
                grp = mMrg[m, 5]
                kcol = mMrg[m, 6]
                ntask = mMrg[m, 7]
                cnt_a = cnt_base + Int64(grp) * Int64(8)
                # Only warp 0 polls, so the spin is warp-uniform and the
                # named barrier below is reached convergently by every warp.
                want = nsplit
                if tidx >= 32:
                    want = Int32(0)
                seen = _ld_acq(cnt_a)
                while seen < want:
                    seen = _ld_acq(cnt_a)
                if tidx == 0:
                    # Every task of this group has cleared the poll once the
                    # departure counter wraps, so the last one can hand the
                    # arrival counter back to the next launch right here.
                    last = _atom_inc(cnt_a + Int64(4), ntask - Int32(1))
                    if last == ntask - Int32(1):
                        _st_rel(cnt_a, Int32(0))
                cute.arch.barrier(barrier_id=1, number_of_threads=NCT)
                if cutlass.const_expr(self.po_bf16 or self.mrg_stages < 1):
                    gmax = Float32(NEG_BIG)
                    for s in cutlass.range(nsplit, unroll=4):
                        v = mPML[row, 0, slot_base + s]
                        if v > gmax:
                            gmax = v
                    c0 = kcol * MCOL + half * MPT
                    acc = cute.make_rmem_tensor(cute.make_layout(MPT), Float32)
                    for i in cutlass.range_constexpr(MPT):
                        acc[i] = Float32(0.0)
                    gsum = Float32(0.0)
                    for s in cutlass.range(nsplit, unroll=2):
                        f = cute.math.exp2(
                            (mPML[row, 0, slot_base + s] - gmax) * scale_log2,
                            fastmath=True)
                        # The denominator rides the same exp2 and the same ascending
                        # split order the partial-O accumulation already walks, so
                        # the separate row-sum pass over mPML disappears.
                        gsum = gsum + mPML[row, 1, slot_base + s] * f
                        if cutlass.const_expr(self.po_bf16):
                            for i in cutlass.range_constexpr(MPT):
                                acc[i] = acc[i] + mPO[
                                    row, c0 + i, slot_base + s].to(Float32) * f
                        else:
                            for v in cutlass.range_constexpr(MPT // POV):
                                pfr = cute.make_rmem_tensor(
                                    cute.make_layout(POV), Float32)
                                cute.autovec_copy(
                                    mPO[row, (None, c0 // POV + v), slot_base + s],
                                    pfr)
                                for i in cutlass.range_constexpr(POV):
                                    acc[v * POV + i] = acc[v * POV + i] + pfr[i] * f
                    rinv = out_scale / gsum
                    rinv2 = _pack2(rinv, rinv)
                    if jrow < n_tok:
                        a64 = cute.make_tensor(
                            cute.recast_ptr(acc.iterator, dtype=Int64),
                            cute.make_layout(MPT // 2))
                        for v2 in cutlass.range_constexpr(MPT // 8):
                            obf = cute.make_rmem_tensor(
                                cute.make_layout(8), cutlass.BFloat16)
                            o32 = cute.make_tensor(
                                cute.recast_ptr(obf.iterator, dtype=Int32),
                                cute.make_layout(4))
                            for i in cutlass.range_constexpr(4):
                                o32[i] = _scale_cvt_bf16x2(a64[v2 * 4 + i], rinv2)
                            cute.autovec_copy(
                                obf, mO[((grow, tok_base + jrow),
                                        (None, c0 // 8 + v2), h_kv)])
                    # A task that owns more than one band replays the weighted sum
                    # on the remaining ones.  gmax and the denominator are column
                    # independent, so only the partial-O accumulation repeats; a
                    # one-band task (the host's choice whenever CTAs are free)
                    # skips this loop entirely and keeps the fused single-pass form.
                    for bb in cutlass.range(NMRG // ntask - Int32(1)):
                        cb = (kcol + Int32(1) + bb) * MCOL + half * MPT
                        for i in cutlass.range_constexpr(MPT):
                            acc[i] = Float32(0.0)
                        for s in cutlass.range(nsplit, unroll=2):
                            f = cute.math.exp2(
                                (mPML[row, 0, slot_base + s] - gmax) * scale_log2,
                                fastmath=True)
                            if cutlass.const_expr(self.po_bf16):
                                for i in cutlass.range_constexpr(MPT):
                                    acc[i] = acc[i] + mPO[
                                        row, cb + i, slot_base + s].to(Float32) * f
                            else:
                                for v in cutlass.range_constexpr(MPT // POV):
                                    pfr = cute.make_rmem_tensor(
                                        cute.make_layout(POV), Float32)
                                    cute.autovec_copy(
                                        mPO[row, (None, cb // POV + v),
                                            slot_base + s], pfr)
                                    for i in cutlass.range_constexpr(POV):
                                        acc[v * POV + i] = (acc[v * POV + i]
                                                            + pfr[i] * f)
                        if jrow < n_tok:
                            a64b = cute.make_tensor(
                                cute.recast_ptr(acc.iterator, dtype=Int64),
                                cute.make_layout(MPT // 2))
                            for v2 in cutlass.range_constexpr(MPT // 8):
                                obf = cute.make_rmem_tensor(
                                    cute.make_layout(8), cutlass.BFloat16)
                                o32 = cute.make_tensor(
                                    cute.recast_ptr(obf.iterator, dtype=Int32),
                                    cute.make_layout(4))
                                for i in cutlass.range_constexpr(4):
                                    o32[i] = _scale_cvt_bf16x2(a64b[v2 * 4 + i],
                                                               rinv2)
                                cute.autovec_copy(
                                    obf, mO[((grow, tok_base + jrow),
                                            (None, cb // 8 + v2), h_kv)])

                else:
                    # Pipelined read-back: each thread cp.asyncs its own 64 B of
                    # the band plus its row's (max, sum) for split s + MST
                    # while it folds split s, so MST splits are in flight per
                    # thread and no cross-thread barrier is needed.  The row
                    # max is taken online in the same ascending split order,
                    # which removes the separate max pass over mPML.
                    MSTG = const_expr(self.mrg_stages)
                    NVEC = const_expr(MPT // POV)
                    PO_SLOT_B = const_expr(MROW * HDIM * 4)
                    PO_VEC_B = const_expr(POV * MROW * 4)
                    ML_SLOT_B = const_expr(2 * MROW * 4)
                    S_VEC_B = const_expr(POV * NCT * 4)
                    S_PO_STG_B = const_expr(NVEC * S_VEC_B)
                    S_ML_STG_B = const_expr(2 * NCT * 4)
                    s_po = Int32(sMrg.iterator.toint()) + tidx * Int32(POV * 4)
                    s_ml = Int32(sMrgML.iterator.toint()) + tidx * Int32(4)
                    ml_t = (mPML.iterator.toint()
                            + Int64(slot_base) * Int64(ML_SLOT_B)
                            + Int64(row * 4))
                    if mt == 0:
                        # The buffers were last written by TMA and read by
                        # UMMA; order those before the generic-proxy refill.
                        cute.arch.fence_proxy("async.shared", space="cta")
                    for bb in cutlass.range(NMRG // ntask):
                        c0 = (kcol + bb) * MCOL + half * MPT
                        po_t = (mPO.iterator.toint()
                                + Int64(slot_base) * Int64(PO_SLOT_B)
                                + Int64((c0 // POV) * PO_VEC_B + row * (POV * 4)))
                        for p in cutlass.range_constexpr(MSTG):
                            if p < nsplit:
                                g = po_t + Int64(p * PO_SLOT_B)
                                for v in cutlass.range_constexpr(NVEC):
                                    _cp_async16(s_po + Int32(p * S_PO_STG_B + v * S_VEC_B),
                                                g + Int64(v * PO_VEC_B))
                                gm = ml_t + Int64(p * ML_SLOT_B)
                                _cp_async4(s_ml + Int32(p * S_ML_STG_B), gm)
                                _cp_async4(s_ml + Int32(p * S_ML_STG_B + NCT * 4),
                                           gm + Int64(MROW * 4))
                            cute.arch.cp_async_commit_group()
                        acc = cute.make_rmem_tensor(cute.make_layout(MPT), Float32)
                        for i in cutlass.range_constexpr(MPT):
                            acc[i] = Float32(0.0)
                        gmax = Float32(NEG_BIG)
                        gsum = Float32(0.0)
                        st = Int32(0)
                        for s in cutlass.range(nsplit):
                            cute.arch.cp_async_wait_group(MSTG - 1)
                            m_s = sMrgML[0, tidx, st]
                            l_s = sMrgML[1, tidx, st]
                            m_new = gmax
                            if m_s > m_new:
                                m_new = m_s
                            a = cute.math.exp2((gmax - m_new) * scale_log2,
                                               fastmath=True)
                            f = cute.math.exp2((m_s - m_new) * scale_log2,
                                               fastmath=True)
                            gsum = gsum * a + l_s * f
                            for v in cutlass.range_constexpr(NVEC):
                                pfr = cute.make_rmem_tensor(
                                    cute.make_layout(POV), Float32)
                                cute.autovec_copy(sMrg[(None, tidx, v, st)], pfr)
                                for i in cutlass.range_constexpr(POV):
                                    acc[v * POV + i] = (acc[v * POV + i] * a
                                                        + pfr[i] * f)
                            gmax = m_new
                            sn = s + Int32(MSTG)
                            if sn < nsplit:
                                g = po_t + Int64(sn) * Int64(PO_SLOT_B)
                                sp = s_po + st * Int32(S_PO_STG_B)
                                for v in cutlass.range_constexpr(NVEC):
                                    _cp_async16(sp + Int32(v * S_VEC_B),
                                                g + Int64(v * PO_VEC_B))
                                gm = ml_t + Int64(sn) * Int64(ML_SLOT_B)
                                sm_ = s_ml + st * Int32(S_ML_STG_B)
                                _cp_async4(sm_, gm)
                                _cp_async4(sm_ + Int32(NCT * 4), gm + Int64(MROW * 4))
                            cute.arch.cp_async_commit_group()
                            st = st + 1
                            if st == Int32(MSTG):
                                st = Int32(0)
                        rinv = out_scale / gsum
                        rinv2 = _pack2(rinv, rinv)
                        if jrow < n_tok:
                            a64 = cute.make_tensor(
                                cute.recast_ptr(acc.iterator, dtype=Int64),
                                cute.make_layout(MPT // 2))
                            for v2 in cutlass.range_constexpr(MPT // 8):
                                obf = cute.make_rmem_tensor(
                                    cute.make_layout(8), cutlass.BFloat16)
                                o32 = cute.make_tensor(
                                    cute.recast_ptr(obf.iterator, dtype=Int32),
                                    cute.make_layout(4))
                                for i in cutlass.range_constexpr(4):
                                    o32[i] = _scale_cvt_bf16x2(a64[v2 * 4 + i], rinv2)
                                cute.autovec_copy(
                                    obf, mO[((grow, tok_base + jrow),
                                            (None, c0 // 8 + v2), h_kv)])

# =============================================================== host glue ===
def _mk(mcast, kvstage, row_pred=False, skew_load=False):
    # Default, mixed-long, very-long, and precise runtime-scale forms for one
    # pipeline shape.  These one-CTA pipelines move a full 64 KiB of K|V per
    # 128-key tile against ~8.4 M MACs, so they sit on the fill ceiling, not on
    # compute-warp issue: the row-sum UMMA measured 13-53% slower there (its
    # tmem P read costs the same as P*V's however narrow N is) while buying
    # nothing.  They keep the register row sum.  The two expanded-P forms reduce residual PV MMAs only
    # where long-query tensor work dominates their extra numerical headroom.
    return [
        PagedFp8Attn(5, True, use_packed=True, use_item_mask=True,
                     kvstage=kvstage, mcast=mcast, row_pred=row_pred,
                     skew_load=skew_load),
        PagedFp8Attn(16, True, use_packed=True, use_item_mask=True,
                     kvstage=kvstage, mcast=mcast,
                     p_prescale=P_MIXED_PRESCALE, corr_tau=P_MIXED_TAU,
                     row_pred=row_pred, skew_load=skew_load),
        PagedFp8Attn(16, True, use_packed=True, use_item_mask=True,
                     kvstage=kvstage, mcast=mcast,
                     p_prescale=P_LONG_PRESCALE, corr_tau=P_LONG_TAU,
                     row_pred=row_pred, skew_load=skew_load),
        PagedFp8Attn(1, True, use_packed=False, kvstage=kvstage, mcast=mcast,
                     row_pred=row_pred, skew_load=skew_load),
        PagedFp8Attn(5, True, use_packed=True, kvstage=kvstage, mcast=mcast,
                     row_pred=row_pred, skew_load=skew_load),
    ]


def _mk2(kvstage, row_pred=False, skew_load=False):
    # Same four numeric forms on the CtaGroup.TWO pipeline.  The pair's MMA
    # covers 256 rows, so each CTA streams only its N-half of K and of V.  That
    # halves the fill and leaves these launches bound by compute-warp issue, so
    # here the row sum is worth moving onto the tensor core.
    return [
        PagedFp8Attn(5, True, use_packed=True, use_item_mask=True,
                     kvstage=kvstage, twocta=True, qtm=QTM, row_pred=row_pred,
                     tsum=True, skew_load=skew_load),
        PagedFp8Attn(16, True, use_packed=True, use_item_mask=True,
                     kvstage=kvstage, twocta=True, qtm=QTM,
                     p_prescale=P_MIXED_PRESCALE, corr_tau=P_MIXED_TAU,
                     row_pred=row_pred, tsum=True, skew_load=skew_load),
        PagedFp8Attn(16, True, use_packed=True, use_item_mask=True,
                     kvstage=kvstage, twocta=True, qtm=QTM,
                     p_prescale=P_LONG_PRESCALE, corr_tau=P_LONG_TAU,
                     row_pred=row_pred, tsum=True, skew_load=skew_load),
        # The precise runtime-scale form releases three e4m3 P components per
        # tile, so the tensor-core row sum would cost three UMMAs there; it
        # keeps the register sum.
        PagedFp8Attn(1, True, use_packed=False, kvstage=kvstage, twocta=True, qtm=QTM,
                     row_pred=row_pred, skew_load=skew_load),
        PagedFp8Attn(5, True, use_packed=True, kvstage=kvstage, twocta=True, qtm=QTM,
                     row_pred=row_pred, tsum=True, skew_load=skew_load),
    ]


# mode 0: single-CTA pages.  mode 1: clustered multicast pages, 3-deep ring.
# mode 2: clustered pages with a 4-deep ring, which only pays off once a CTA
# streams one very long piece at a time; on short split pieces the extra
# prefetch distance evicts KV that neighbouring CTAs still want.
# mode 3: CtaGroup.TWO collective tile.  Keep the 3-tile lookahead of mode 1:
# the collective tile already halves the per-stage bytes, and prefetching a
# fourth tile ahead only evicts KV that the neighbouring pairs still want.
def _mk2d(kvstage, row_pred=False, skew_load=False):
    # CtaGroup.TWO with the two-tile Q*K^T lead and one more ring stage, so the
    # loader keeps its prefetch distance now that P*V trails Q*K^T by two.
    return [
        PagedFp8Attn(5, True, use_packed=True, use_item_mask=True,
                     kvstage=kvstage, twocta=True, qtm=QTM, row_pred=row_pred,
                     tsum=True, skew_load=skew_load, deep=True),
        PagedFp8Attn(16, True, use_packed=True, use_item_mask=True,
                     kvstage=kvstage, twocta=True, qtm=QTM,
                     p_prescale=P_MIXED_PRESCALE, corr_tau=P_MIXED_TAU,
                     row_pred=row_pred, tsum=True, skew_load=skew_load,
                     deep=True),
        PagedFp8Attn(16, True, use_packed=True, use_item_mask=True,
                     kvstage=kvstage, twocta=True, qtm=QTM,
                     p_prescale=P_LONG_PRESCALE, corr_tau=P_LONG_TAU,
                     row_pred=row_pred, tsum=True, skew_load=skew_load,
                     deep=True),
        PagedFp8Attn(1, True, use_packed=False, kvstage=kvstage, twocta=True, qtm=QTM,
                     row_pred=row_pred, skew_load=skew_load, deep=True),
        PagedFp8Attn(5, True, use_packed=True, kvstage=kvstage, twocta=True, qtm=QTM,
                     row_pred=row_pred, tsum=True, skew_load=skew_load,
                     deep=True),
    ]


def _all_ops(skew_load):
    return (
        _mk(False, 3, skew_load=skew_load) +
        _mk(False, 3, True, skew_load) +
        _mk(True, 3, skew_load=skew_load) +
        _mk(True, 3, True, skew_load) +
        _mk(True, 4, skew_load=skew_load) +
        _mk(True, 4, True, skew_load) +
        _mk2(3, skew_load=skew_load) +
        _mk2(3, True, skew_load) +
        _mk2d(4, skew_load=skew_load) +
        _mk2d(4, True, skew_load) +
        # mode 5: the two-tile Q*K^T lead on the plain three-stage ring.  The
        # four-stage deep form measured 5-14% slower below 4000 KV tiles per
        # pair, and the stated cause is the fourth ring stage evicting K|V that
        # neighbouring pairs still want -- not the lead itself.  Keeping the
        # ring at three stages isolates the lead, which is what buys the
        # softmax warps a whole tile of UMMA between S(t) and P*V(t).
        _mk2d(3, skew_load=skew_load) +
        _mk2d(3, True, skew_load)
    )


NOPS = len(_all_ops(False))
# Index 2 * NOPS onward: the same forms with only half of Q in tmem.  A third
# tmem Q block draws more power per tile, which costs clock on launches long
# enough to sit at the board power cap; those keep the two-block layout.
_OPS = (_all_ops(False) + _all_ops(True) +
        [op.set_qtm_kb(QTM_KB) for op in _all_ops(False) + _all_ops(True)])
DEEP_PIECE = 500.0   # work-weighted mean KV tiles per piece to earn mode 2
# KV tiles one CTA streams, averaged over the persistent grid, at which the
# CtaGroup.TWO collective tile earns its keep.  The collective tile halves the
# per-SM K|V shared-memory fill but lock-steps the two CTAs of a pair on every
# tile, so it only pays once each CTA streams long enough to sit at the
# steady-state fill ceiling; shorter launches are tail- and latency-bound and
# the multicast pipeline, whose CTAs advance independently, is faster there.
LONG_Q_ROWS = 24 * 1024
VERY_LONG_MEAN_Q = 4 * 1024
# A 128-row tile pulls 64 KiB of K|V per KV tile, enough to leave the L2 read
# path near its per-SM ceiling.  Pairing two 16-token q-tiles in a 2-CTA cluster
# and multicasting the pages halves the L2 reads per tile, which measures ~2% per
# tile; a launch whose q-tiles do not pair evenly pays more than that in padding,
# so the ratio below is the measured gain rather than the modelled one.
MCAST_TILE_RATIO = 0.98
TWOCTA_TILES_PER_CTA = 1800.0
# Work-weighted KV tiles per split piece above which the isolated two-tile
# Q*K^T lead earns its keep on the three-stage collective ring.  The lead costs
# a two-tile prologue and a one-tile drain per piece and buys the softmax warps
# a whole UMMA of slack per tile, so the trade is set by the piece length, not
# by how many tiles a CTA streams in total: measured -7.1% at 1063 tiles per
# piece and -2.8% at 1037, against +0.4..0.8% at 692 and below.
LEAD_PIECE = 900.0
# KV tiles per CTA pair above which the two-tile Q*K^T lead pays.  Below it the
# extra ring stage costs more in L2 residency than the latency slack wins.
DEEP_TILES_PER_CTA = 4000.0
SKEW_Q_ROWS = 2 * 1024
REPLAY_Q_ROWS = 10 * 1024
_COMPILED = [None] * (4 * NOPS + 10)
# A launch whose busiest merging CTA has to read back more partial bytes than it
# ever streamed of KV is dominated by that serial tail, and there bf16 partials
# halve it for ~2e-4 relative error.  Everywhere else the extra pack/unpack in
# the epilogue costs more than the traffic it saves, so those keep fp32.
PO_BF16_RATIO = 0.12
_NSM = [0]
NSLOT_CAP = 1520      # 191 MiB of partials, under the 197 MiB scratch cap
C_ITEM = 2.5      # fixed per-work-item cost, in units of one 128-key KV tile
C_SPLIT = 0.5     # sub-threshold per-piece scheduling cost; preserves split counts
C_MERGE = 5.0     # cost of reading one whole split partial, in KV-tile units
_SCOREMODE = [0]  # 0: score plans for a single merger, 1: for the NMRG tasks
C_MPRE = 0.2      # per-partial cost of a merge task's row max/sum pre-pass
C_MTASK = 10.0    # fixed cost of one merge task, in KV-tile units: the counter
# poll, the 16-warp rendezvous and the per-split row max/sum walk are all
# latency, not bandwidth, so they do not shrink with the band width.  Measured
# by doubling the tasks per group: +4.0 to +6.5% on four launches with 150-330
# groups, which puts one task at ~5 us against a ~0.43 us KV tile.
# Measured extra cost of a KV tile that also runs the residual-P correction:
# one more P*V UMMA plus the residual requantize wave, ~0.36 of a plain tile.
# The schedule below gives different streams different correction densities, so
# the LPT pack has to balance weighted tiles or the densely corrected streams
# pile up on a few CTAs.
RESID_ALPHA = 0.36
_KGRID = tuple(sorted(set(
    (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0) +
    tuple(2.0 ** (i / 8.0) for i in range(41)))))
_BASE_WAVE_GRID = (0.0625, 0.125, 0.25, 1.0 / 3.0, 0.5,
                   1, 2, 3, 4, 6, 8, 12, 16)
_DENSE_WAVE_GRID = tuple(sorted(set(
    _BASE_WAVE_GRID +
    tuple(i / 16.0 for i in range(1, 17)) +
    tuple(i / 4.0 for i in range(5, 33)) +
    tuple(i / 2.0 for i in range(17, 33)))))


def _score_last(bins):
    """Legacy makespan: only the last-arriving split CTA pays the merge.

    The distributed merge below is strictly cheaper, so scoring the plan search
    with this model is conservative: it keeps the split counts the single-CTA
    merge earned and lets the parallel merge bank the difference, instead of
    cutting KV further and paying NMRG task set-ups for every new piece."""
    pos = [0] * len(bins)
    arrived = {}
    events = []
    for cta, jobs in enumerate(bins):
        if jobs:
            heapq.heappush(events, (jobs[0][0], cta))
    makespan = 0.0
    while events:
        done_at, cta = heapq.heappop(events)
        rec = bins[cta][pos[cta]][1]
        nsplit = rec[9]
        if nsplit > 1:
            grp = rec[10]
            count = arrived.get(grp, 0) + 1
            arrived[grp] = count
            if count == nsplit:
                done_at += C_MERGE * nsplit
        pos[cta] += 1
        if pos[cta] < len(bins[cta]):
            heapq.heappush(events, (done_at + bins[cta][pos[cta]][0], cta))
        makespan = max(makespan, done_at)
    return makespan


def _score_bins(bins, nsm, dup):
    """Simulate the persistent grid and lay out the split-merge tasks.

    Every CTA first streams the parts of its LPT bin; only then does it run the
    merge tasks the host gave it.  A group of `ns` split partials is reduced by
    NMRG CTAs, each owning a disjoint column band, so the serial read-back that
    used to sit on the single last-arriving CTA is cut NMRG ways and can be
    placed on CTAs that would otherwise idle at the tail.  Deferring every merge
    behind a CTA's own streaming rows is what makes the wait safe: no CTA can
    block on a group while still holding a part some other CTA waits for.

    Returns (makespan, mrg_rows, mrg_starts)."""
    ngrid = nsm * dup
    free = [0.0] * ngrid
    ready = {}
    info = {}
    for i, jobs in enumerate(bins):
        cum = 0.0
        for cost, rec in jobs:
            cum += cost
            ns = rec[9]
            if ns > 1:
                for r in range(dup):
                    g = rec[10] * dup + r
                    nt = rec[1]
                    tb = rec[0]
                    if dup > 1:
                        if nt > 16:
                            tb = tb + r * 16
                        nt = nt - r * 16
                        if nt > 16:
                            nt = 16
                    if g not in info:
                        info[g] = (tb, nt, rec[2], rec[8] + r * ns, ns)
                    if cum > ready.get(g, 0.0):
                        ready[g] = cum
        for r in range(dup):
            free[i * dup + r] = cum
    if not info:
        return max(free), [], [0] * (ngrid + 1)
    heap = [(free[c], c) for c in range(ngrid)]
    heapq.heapify(heap)
    per = [[] for _ in range(ngrid)]
    order = sorted(info, key=lambda k: (ready[k], k))
    for gi, g in enumerate(order):
        rem = len(order) - gi
        tb, nt, h, sb, ns = info[g]
        band = C_MERGE * ns / NMRG + C_MPRE * ns
        rdy = ready[g]
        # Pick how many CTAs this group's read-back is cut across.  Eight
        # one-band tasks finish a group in an eighth of the time but need eight
        # CTAs free at once; when the grid still has streaming rows to run, one
        # task over all eight bands lands earlier.  Score both ends and the two
        # powers between against the actual free times.
        cand = [heapq.heappop(heap) for _ in range(NMRG)]   # ascending free time
        best = None
        for nt_try in (1, 2, 4):
            cost_try = band * (NMRG // nt_try) + C_MTASK
            fr = cand[nt_try - 1][0]
            fin_try = (fr if fr > rdy else rdy) + cost_try
            # Splitting a group costs nt_try set-ups instead of one.  That
            # CTA time is unavailable to every group still waiting, which on a
            # grid with many groups lengthens the whole tail by the aggregate
            # over the remaining ones; weigh it against the gain on this group.
            fin_try += rem * nt_try * cost_try / ngrid
            if best is None or fin_try < best[0]:
                best = (fin_try, nt_try, cost_try)
        ntask = best[1]
        cost = best[2]
        nband = NMRG // ntask
        for k in range(ntask):
            fr, c = cand[k]
            fin = (fr if fr > rdy else rdy) + cost
            per[c].append((tb, nt, h, sb, ns, g, k * nband, ntask))
            free[c] = fin
            heapq.heappush(heap, (fin, c))
        for k in range(ntask, NMRG):
            heapq.heappush(heap, cand[k])
    rows = []
    starts = [0] * (ngrid + 1)
    for c in range(ngrid):
        rows.extend(per[c])
        starts[c + 1] = len(rows)
    return max(free), rows, starts


RESID_MAXK = 10         # deepest thinning any stream may take (1 in 1024)
# A schedule pays off only when it is mostly a thinning: a tile whose period
# shrinks costs strictly more time, and measurement shows that densifying a
# large share of the tiles loses even when the launch total falls.  Only accept
# a schedule whose denser-than-uniform tiles are a small minority, which is the
# regime the 1/N^2 argument was derived for -- a short stream is cheap exactly
# because it is few tiles.
RESID_DENSE_FRAC = 0.15


def _resid_mask(npg):
    """Placeholder cadence; setup() rewrites every plan row with _mask_k."""
    return 7


def _mask_k(npg, sched):
    """log2 of the residual-P correction period for a stream of npg KV tiles.

    The e4m3 P quantization is the kernel's whole error: every probability
    carries the same 2^-4/m relative round-off, so a row that attends N keys
    ends up with |dO| ~ 0.0265 * |O| and |O| ~ sigma / sqrt(N).  The variance a
    corrected tile removes is therefore proportional to 1/N^2 while its cost is
    flat, which makes correction on a short stream hundreds of times more
    valuable than on a long one, so the period grows as a power of npg.  The
    exponent is searched with the scale: when one long request owns nearly the
    whole relative-L2 budget the npg^-2 fall-off is far too slow, and only a
    steeper law puts the budget on the short streams without also paying to
    densify the long one."""
    scale, alpha = sched
    want = scale / float(npg) ** alpha
    if want >= 1.0:
        return 0
    k = 0
    while k < RESID_MAXK and (1.0 / (1 << (k + 1))) >= want:
        k += 1
    return k


_RESID_EXPONENTS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0)


def _resid_scale(qls, Ls, f_cur):
    """Cheapest power-law correction schedule that is no worse than the uniform
    cadence f_cur on either the fp32-oracle relative L2 norm or the worst-row
    absolute error.  Returns None to keep a uniform power-of-two cadence.

    Streams are bucketed by npg before the search: the period is a function of
    npg alone, so one bucket per distinct stream length carries the same three
    sums the feasibility test needs and the two-parameter sweep costs no more
    than the old one-parameter sweep over every q-tile."""
    grp = {}
    for b in range(len(qls)):
        ql = int(qls[b])
        L = int(Ls[b])
        for qt in range(0, ql, 16):
            n = min(16, ql - qt)
            causal = L - ql + qt
            npg = max(1, (causal + n + PAGE - 1) // PAGE)
            N = max(causal + n, 1)
            r = grp.get(npg)
            if r is None:
                grp[npg] = [n / float(N), float(n * npg), N]
            else:
                r[0] += n / float(N)
                r[1] += float(n * npg)
                if N < r[2]:
                    r[2] = N
    if not grp:
        return None
    gs = sorted(grp.items())
    ref = float(max(grp))
    varc = sum(r[0] for _, r in gs)
    worst0 = max(math.sqrt((1.0 - f_cur) / r[2]) for _, r in gs)
    tiles = sum(r[1] for _, r in gs)
    dense_cap = RESID_DENSE_FRAC * tiles
    lim = math.sqrt(1.0 - f_cur)
    best = None
    # Only accept a schedule that is strictly cheaper than the uniform cadence
    # this launch would otherwise run; otherwise the plain AOT form is kept.
    best_cost = tiles * f_cur
    for alpha in _RESID_EXPONENTS:
        base = alpha * math.log(ref)
        for e in range(-40, 41):
            sched = (math.exp(base + (e / 2.0) * _LN2) / 16.0, alpha)
            rem = 0.0
            cost = 0.0
            worst = 0.0
            dense = 0.0
            for npg, r in gs:
                f = 1.0 / (1 << _mask_k(npg, sched))
                rem += r[0] * f
                cost += r[1] * f
                if f > f_cur:
                    dense += r[1]
                w = math.sqrt(max(0.0, 1.0 - f) / r[2])
                if w > worst:
                    worst = w
            if cost >= best_cost or worst > worst0 or dense > dense_cap:
                continue
            if math.sqrt(max(0.0, 1.0 - rem / varc)) <= lim:
                best = sched
                best_cost = cost
    return best


def _wave_splits(items, nsm, wave):
    """Per-item piece counts whose total is wave*nsm, allotted in proportion to
    each item's attended KV length (largest-remainder rounding)."""
    total = sum(it[5] for it in items)
    if total <= 0:
        return None
    target = wave * nsm
    raw = [it[5] * target / total for it in items]
    ns = [max(1, min(it[5], int(r))) for it, r in zip(items, raw)]
    short = target - sum(ns)
    if short > 0:
        order = sorted(range(len(items)), key=lambda i: -(raw[i] - int(raw[i])))
        k = 0
        while short > 0 and k < len(order):
            i = order[k]
            if ns[i] < items[i][5]:
                ns[i] += 1
                short -= 1
            k += 1
    return ns


def _stream_neff(groups):
    """Work-weighted count of KV streams a CTA group is reading at once."""
    tot = 0.0
    sq = 0.0
    for v in groups:
        tot += v
        sq += v * v
    return (tot * tot / sq) if sq > 0.0 else 1.0


def _filler_splits(items, nbin, k, ns):
    """Cut the k costliest items into ns pieces each and leave the rest whole.

    LPT over many near-equal items quantises: a launch with 18.2 items per CTA
    runs 19 of them on the busiest bin, a 4% tail that no uniform chunk can
    remove because cutting every item at once blows the partial-slot budget.  A
    handful of finely cut items is enough filler to level the bins, and only
    those items pay split and merge traffic.
    """
    if k <= 0 or ns <= 1 or not items:
        return None
    order = sorted(range(len(items)), key=lambda i: (-items[i][5], i))
    sp = [1] * len(items)
    cut = False
    for i in order[:k]:
        if items[i][5] >= ns:
            sp[i] = ns
            cut = True
    return sp if cut else None


def _pack(items, chunk, nsm, splits=None, dup=1, streamwave=False, wfun=None):
    """Cut every work item into parts, then LPT-pack the parts onto nsm
    persistent CTAs.  `chunk` caps the pages per part; `splits` instead gives an
    explicit per-item piece count.  Returns (makespan, rows, bin_starts, nslot,
    ngrp, nstream) or None when the partial-output slot budget is exceeded.

    With `streamwave`, the parts are cut into waves of whole (request, kv head)
    KV streams -- each wave at least one part per CTA -- and the waves are
    LPT-packed in turn.  Inside a wave the order is the same cost-descending LPT
    order as before, so the CTAs stay aligned on the same page offsets; across
    waves the whole grid walks the streams together.  A page is then live in L2
    while every CTA that wants it reads it, instead of being refetched once per
    group of CTAs that happen to share a stream, which is what a batch of
    similar-length requests pays in DRAM traffic."""
    parts = []
    nslot = 0
    grp = 0
    for idx, (tok_base, n_tok, h, b, causal, npg) in enumerate(items):
        ns = splits[idx] if splits is not None else (npg + chunk - 1) // chunk
        wm = wfun(npg) if wfun is not None else 1.0
        if ns <= 1:
            parts.append((float(npg) * wm + C_ITEM,
                          (tok_base, n_tok, h, b, 0, npg, causal, -1, 0, 1, -1,
                           _resid_mask(npg))))
        else:
            per = (npg + ns - 1) // ns
            ns = (npg + per - 1) // per
            sb = nslot
            nslot += dup * ns
            if nslot > NSLOT_CAP:
                return None
            for s in range(ns):
                p0 = s * per
                pw = min(per, npg - p0)
                parts.append((float(pw) * wm + C_ITEM + C_SPLIT,
                              (tok_base, n_tok, h, b, p0 * PAGE, pw, causal,
                               sb + s, sb, ns, grp, _resid_mask(pw))))
            grp += 1
    streams = {}
    for cost, rec in parts:
        streams.setdefault((rec[3], rec[2]), []).append((cost, rec))
    if streamwave:
        order = sorted(streams, key=lambda k: (-sum(c for c, _ in streams[k]), k))
        groups = []
        cur = []
        curw = []
        for k in order:
            cur.extend(streams[k])
            curw.append(sum(c for c, _ in streams[k]))
            if len(cur) >= nsm:
                groups.append((cur, curw))
                cur = []
                curw = []
        if cur:
            if groups:
                groups[-1][0].extend(cur)
                groups[-1][1].extend(curw)
            else:
                groups.append((cur, curw))
        waves = []
        tot = 0.0
        neff = 0.0
        for g, gw in groups:
            w = sum(gw)
            tot += w
            neff += w * _stream_neff(gw)
            waves.append(sorted(g, key=lambda p: -p[0]))
        nstream = (neff / tot) if tot > 0.0 else 1.0
    else:
        waves = [sorted(parts, key=lambda p: -p[0])]
        nstream = _stream_neff([sum(c for c, _ in v) for v in streams.values()])
    load = [0.0] * nsm
    bins = [[] for _ in range(nsm)]
    heap = [(0.0, i) for i in range(nsm)]
    heapq.heapify(heap)
    for wave in waves:
        for cost, rec in wave:
            l, i = heapq.heappop(heap)
            bins[i].append((cost, rec))
            heapq.heappush(heap, (l + cost, i))
            load[i] = l + cost
    rows = []
    starts = [0] * (nsm + 1)
    for i in range(nsm):
        rows.extend(rec for _, rec in bins[i])
        starts[i + 1] = len(rows)
    ms_new, mrg, mstart = _score_bins(bins, nsm, dup)
    ms = ms_new if _SCOREMODE[0] else _score_last(bins)
    return ms, rows, starts, nslot, max(grp, 1), nstream, mrg, mstart


def _build_plan(qls, Ls, nsm, pair=False, wfun=None):
    """Host work list.  A work item is (request, 16-token q-tile x 8 grouped
    heads, kv head, kv split).  Splitting granularity is searched so that the
    LPT makespan over the persistent grid is minimal: too coarse leaves whole
    CTAs idle at the tail, too fine pays the split-merge traffic."""
    step = 32 if pair else 16
    dup = 2 if pair else 1
    nbin = max(1, nsm // 2) if pair else nsm
    items = []
    off = 0
    for b in range(len(qls)):
        ql = int(qls[b])
        L = int(Ls[b])
        for qt in range(0, ql, step):
            n_tok = min(step, ql - qt)
            causal = L - ql + qt
            npg = (causal + n_tok + PAGE - 1) // PAGE
            for h in range(HKV):
                items.append((off + qt, n_tok, h, b, causal, npg))
        off += ql
    if not items:
        return None
    nsm = nbin
    total = sum(it[5] + C_ITEM for it in items)
    best = None
    best_cfg = None
    # Sub-unit waves matter when a launch has far fewer work items than SMs.
    # The old grid could never emit fewer parts than CTAs, so a 6-item launch
    # was cut 36 ways and one CTA then read 36 x 128 KiB of partials back
    # serially; leaving CTAs idle beats paying that merge tail.
    for wave in _BASE_WAVE_GRID:
        sp = _wave_splits(items, nsm, wave)
        if sp is None:
            continue
        res = _pack(items, 0, nsm, splits=sp, dup=dup, wfun=wfun)
        if res is not None and (best is None or res[0] < best[0]):
            best = res
            best_cfg = (0, sp)
    seen = set()
    for k in _KGRID:
        raw_chunk = max(1, int(math.ceil((total / nsm) / k)))
        chunk = max(80, raw_chunk)
        if chunk != raw_chunk:
            floor_res = _pack(items, chunk, nsm, dup=dup, wfun=wfun)
            if floor_res is not None and len(floor_res[1]) < nsm:
                chunk = raw_chunk
        if chunk in seen:
            continue
        seen.add(chunk)
        res = _pack(items, chunk, nsm, dup=dup, wfun=wfun)
        if res is not None and (best is None or res[0] < best[0]):
            best = res
            best_cfg = (chunk, None)
    if best is None:
        best_cfg = (max(it[5] for it in items), None)
        best = _pack(items, best_cfg[0], nsm, dup=dup, wfun=wfun)

    # Fine wave steps can remove a merge-tail cliff, but accepting every tiny
    # model win overfits the fixed-cost estimate and can add split traffic.
    # Keep the established plan as a control and accept a dense-wave plan only
    # when its predicted win is material and it does not grow split scratch.
    # If it underfills the grid, require a much larger modeled win as evidence
    # that the reduced merge tail outweighs leaving CTAs idle.
    base = best
    dense = best
    dense_cfg = best_cfg
    for wave in _DENSE_WAVE_GRID:
        sp = _wave_splits(items, nsm, wave)
        if sp is None:
            continue
        res = _pack(items, 0, nsm, splits=sp, dup=dup, wfun=wfun)
        if res is not None and res[0] < dense[0]:
            dense = res
            dense_cfg = (0, sp)
    base_active = sum(base[2][i + 1] > base[2][i] for i in range(nsm))
    dense_active = sum(dense[2][i + 1] > dense[2][i] for i in range(nsm))
    if (dense[0] <= base[0] * 0.99 and dense[3] <= base[3] and
            (dense_active >= base_active or dense[0] <= base[0] * 0.95)):
        best = dense
        best_cfg = dense_cfg
    # The LPT tail that is left is pure wave quantisation: the items are all
    # the same length, so the busiest bin simply holds one more of them than
    # the average.  Cutting a few of the longest ones into filler pieces levels
    # that without the scratch a uniform cut would need.  Accept only a
    # material model win so the established plan stays the default wherever the
    # bins were already level.  Only where every CTA already holds work: with
    # fewer parts than bins the grid is underfilled rather than quantised, the
    # wave search above owns that regime, and extra cuts there just buy merge
    # traffic (measured +4.9% on a 28-item launch).
    fgrid = (max(1, nsm // 4), max(1, nsm // 2), nsm, 2 * nsm) if len(items) >= nsm else ()
    for fk in fgrid:
        for fns in (2, 3, 4, 6, 8):
            sp = _filler_splits(items, nsm, fk, fns)
            if sp is None:
                continue
            res = _pack(items, 0, nsm, splits=sp, dup=dup, wfun=wfun)
            if res is not None and res[0] < best[0] * 0.995:
                best = res
                best_cfg = (0, sp)

    # Same cut, walked stream by stream.  Every CTA then pulls the pages its
    # neighbours are pulling, so one DRAM fetch of a page feeds the whole grid
    # instead of only the CTAs that happen to share that KV stream.  The waves
    # cost a little LPT freedom, so take the order only when the modelled
    # makespan barely moves and the stream count really collapses -- a batch of
    # same-length requests goes from ~16 concurrent streams to ~1, while a
    # launch already dominated by one request is a no-op and is rejected here.
    if best_cfg is not None:
        alt = _pack(items, best_cfg[0], nsm, splits=best_cfg[1], dup=dup,
                    streamwave=True, wfun=wfun)
        if (alt is not None and alt[0] <= best[0] * 1.01 and
                alt[3] <= best[3] and best[5] >= 4.0 and
                alt[5] * 2.0 <= best[5]):
            best = alt
    return (best[1], best[2], max(best[3], 1), best[4], best[0],
            best[6], best[7])


def setup(context):
    dev = torch.device(context["device"])
    qls = context["setup_inputs"]["query_lens_cpu"].tolist()
    Ls = context["setup_inputs"]["seq_lens_cpu"].tolist()
    if _NSM[0] == 0:
        _NSM[0] = torch.cuda.get_device_properties(dev).multi_processor_count

    nsm = _NSM[0]
    total_q = sum(qls)
    # The residual-P schedule is decided before the pack so the LPT bins can be
    # balanced in weighted tiles.
    f_cur = (1.0 / 16.0) if total_q >= LONG_Q_ROWS else (1.0 / 8.0)
    rscale = _resid_scale(qls, Ls, f_cur)
    kflat = 0
    while (1.0 / (1 << (kflat + 1))) >= f_cur:
        kflat += 1
    if rscale is None:
        # No schedule: leave the pack exactly as it was.  A uniform weight is
        # not neutral here -- it would rescale every tile against the fixed
        # per-item, per-split and per-merge costs and move the plan search.
        wfun = None
    else:
        # Normalised so a tile running the uniform cadence still weighs one,
        # which keeps C_ITEM / C_SPLIT / C_MERGE on their calibrated scale.
        wnorm = 1.0 + RESID_ALPHA * f_cur
        wfun = (lambda npg: (1.0 + RESID_ALPHA /
                             float(1 << _mask_k(npg, rscale))) / wnorm)
    flat_plan = _build_plan(qls, Ls, nsm, pair=False, wfun=wfun)
    pair_plan = _build_plan(qls, Ls, nsm, pair=True, wfun=wfun) if nsm >= 2 else None
    # The single-merger model answers "how many ways may I cut this stream
    # before one CTA's serial read-back costs more than the parallelism buys".
    # With the read-back spread over NMRG CTAs that answer is too coarse only
    # where it left the grid visibly idle, so re-search in the distributed
    # currency exactly there -- a handful of very-few-item launches.
    pick = pair_plan if (pair_plan is not None and
                         pair_plan[4] * MCAST_TILE_RATIO < flat_plan[4]) else flat_plan
    nbin0 = (nsm // 2) if pick is pair_plan else nsm
    st0 = pick[1]
    act0 = sum(st0[i + 1] > st0[i] for i in range(nbin0))
    if act0 * 4 < nbin0 * 3:
        _SCOREMODE[0] = 1
        try:
            flat_plan = _build_plan(qls, Ls, nsm, pair=False, wfun=wfun)
            if nsm >= 2:
                pair_plan = _build_plan(qls, Ls, nsm, pair=True, wfun=wfun)
        finally:
            _SCOREMODE[0] = 0
    # Clustered pages are cheaper per KV tile but a request whose q-tiles do not
    # pair evenly makes one CTA of the cluster stream for rows it will not emit.
    # Compare the two modelled makespans in their own per-tile currency.
    use_mcast = (pair_plan is not None and
                 pair_plan[4] * MCAST_TILE_RATIO < flat_plan[4])
    plan, starts, nslot, ngrp, _, mrg, mstart = (
        pair_plan if use_mcast else flat_plan)
    nwork = max(len(plan), 1)
    gcnt = {}
    for r in plan:
        if r[9] > 1:
            gcnt[r[10]] = gcnt.get(r[10], 0) + 1
    max_merge = max(gcnt.values()) if gcnt else 0
    tiles_tot = sum(r[5] for r in plan)
    merge_ratio = (max_merge * C_MERGE) / max(tiles_tot / _NSM[0], 1e-9)
    # Split partials participate in the mathematically sensitive fixed-order
    # merge.  Keep them fp32 for every launch; bf16 partials passed the search
    # seeds but lost the required held-out/non-unit-scale margin.
    po_bf16 = False
    tiles = [float(r[5]) for r in plan] or [0.0]
    piece_len = sum(t * t for t in tiles) / max(sum(tiles), 1.0)
    deep = use_mcast and piece_len >= DEEP_PIECE
    # A paired launch streams tiles_tot KV tiles over grid/2 CTA pairs, and both
    # CTAs of a pair walk every tile of their pieces.
    tiles_per_cta = tiles_tot / max(_NSM[0] // 2, 1)
    # The collective also wins before its generic steady-state crossover when
    # the launch is underfilled but each item has a long prefix: halving each
    # SM's K|V fill outweighs pair lock-step.  Keep the gates length-derived so
    # they generalize to the production trace rather than naming workloads.
    cta2_underfilled = (total_q < 6000 and
                        tiles_per_cta >= 1500.0)
    cta2_long_prefix_tail = (total_q < 3000 and
                             max(Ls, default=0) >= 160000 and
                             tiles_per_cta >= 1100.0)
    # B=1 short-query launches need a split that fills the GPU; this is also
    # the acceptance-gate regime where the unsplit context path regresses.
    cta2_single_short = (len(qls) == 1 and total_q <= 128 and
                         max(Ls, default=0) >= 8192)
    # Launches that carry real query work per request stream long enough per CTA
    # to sit at the steady-state fill ceiling, where the collective tile's
    # halved K|V fill beats the independent multicast CTAs.  Short-query
    # launches over long prefixes stay on multicast: they split far more, and
    # the pair lock-step costs more there than the fill it saves.
    cta2_long_query = (tiles_per_cta >= 600.0 and total_q >= 2048)
    if use_mcast:
        if tiles_per_cta >= DEEP_TILES_PER_CTA:
            mode = 4
        elif (tiles_per_cta >= TWOCTA_TILES_PER_CTA or cta2_underfilled or
              cta2_long_prefix_tail or cta2_single_short or
              cta2_long_query):
            # The lead only pays on long split pieces, which alone amortize
            # its two-tile prologue and one-tile drain; pieces below that run
            # the plain collective pipeline.
            mode = 5 if piece_len >= LEAD_PIECE else 3
        else:
            mode = 2 if deep else 1
    else:
        mode = 0
    mode_base = 10 * mode
    # Form 4 is the plain 1-in-5 cadence; it serves launches whose streams are
    # all one length, where no power-of-two schedule beats it.
    variant = mode_base + (0 if rscale is not None else 4)
    if total_q >= LONG_Q_ROWS:
        variant = mode_base + (2 if total_q >= VERY_LONG_MEAN_Q * len(qls) else 1)
    # Large-query launches amortize a second AOT specialization and benefit
    # when only rows whose lazy anchor moved replay exp2 and FP8 packing.
    # Keep the established unconditional replay on medium launches, where the
    # extra predicate and code footprint measured slower.
    # On medium CTA2 launches, row predication pays when query work is spread
    # across requests (or one sufficiently long B=1 request): their long
    # per-row histories make the CTA anchor vote much more frequent than an
    # individual row move.  Highly skewed mixtures keep unconditional replay;
    # their many tiny requests measured faster without the extra predicate.
    max_q = max(qls, default=0)
    balanced_cta2_replay = (
        mode >= 3 and total_q >= 2 * 1024 and
        (len(qls) == 1 or 2 * max_q <= total_q))
    if total_q >= REPLAY_Q_ROWS or balanced_cta2_replay:
        variant = variant + 5
    precise_variant = mode_base + 3
    # Short launches prioritize the next QK page ahead of the older V page;
    # larger mixed launches retain interleaved K(t), V(t) TMA issue order.
    # Short launches prioritize the next QK page ahead of the older V page.
    # Widening this gate to 8k rows measured 7-11% worse on c157/c179.
    if total_q <= SKEW_Q_ROWS:
        variant = variant + NOPS
        precise_variant = precise_variant + NOPS
    if total_q >= LONG_Q_ROWS:
        variant = variant + 2 * NOPS
        precise_variant = precise_variant + 2 * NOPS
    grid = ((nsm // 2) * 2) if use_mcast else nsm
    # Spend the residual-P correction budget where it buys the most accuracy.
    # The uniform cadence this form would otherwise run is the reference; the
    # schedule below is accepted only when it is cheaper AND no worse on both
    # correctness metrics, so the numerics never regress.
    flat = torch.zeros((nwork, PLAN_F), dtype=torch.int32)
    for i, r in enumerate(plan):
        for f in range(PLAN_F):
            flat[i, f] = r[f]
        if rscale is not None:
            npg_full = max(1, (int(r[6]) + int(r[1]) + PAGE - 1) // PAGE)
            flat[i, 11] = (1 << _mask_k(npg_full, rscale)) - 1
        else:
            flat[i, 11] = (1 << kflat) - 1
    plan_t = flat.to(dev, non_blocking=False)
    bin_t = torch.tensor(starts, dtype=torch.int32).to(dev, non_blocking=False)
    po = torch.empty((nslot, HDIM, MROW),
                     dtype=torch.bfloat16 if po_bf16 else torch.float32, device=dev)
    pml = torch.empty((nslot, 2, MROW), dtype=torch.float32, device=dev)
    # Two self-resetting counters per split group: arrivals of the split
    # partials, then departures of the NMRG merge tasks that consume them.
    cnt = torch.zeros((max(4 * ngrp, 4) + 4,), dtype=torch.int32, device=dev)
    nmrg = max(len(mrg), 1)
    mflat = torch.zeros((nmrg, MRG_F), dtype=torch.int32)
    for i, r in enumerate(mrg):
        for f in range(MRG_F):
            mflat[i, f] = r[f]
    mrg_t = mflat.to(dev, non_blocking=False)
    mbin_t = torch.tensor(mstart if mstart else [0] * (grid + 1),
                          dtype=torch.int32).to(dev, non_blocking=False)
    torch.cuda.synchronize()

    st = {
        "plan": plan_t,
        "bins": bin_t,
        "rbin": from_dlpack(bin_t, assumed_align=16).mark_layout_dynamic(leading_dim=0),
        "po": po,
        "pml": pml,
        "cnt": cnt,
        "cnt_base": int(cnt.data_ptr()),
        "nwork": nwork,
        "nslot": nslot,
        "variant": variant,
        "precise_variant": precise_variant,
        "rplan": from_dlpack(plan_t, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        "rpo": from_dlpack(po, assumed_align=16).mark_layout_dynamic(leading_dim=2),
        "rpml": from_dlpack(pml, assumed_align=16).mark_layout_dynamic(leading_dim=2),
        "mrg": mrg_t,
        "mbin": mbin_t,
        "rmrg": from_dlpack(mrg_t, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        "rmbin": from_dlpack(mbin_t, assumed_align=16).mark_layout_dynamic(leading_dim=0),
    }
    if _COMPILED[variant] is None or _COMPILED[precise_variant] is None:
        # Compile ahead of the timed path: every tensor mode that varies across
        # launches is staged, so one binary serves every shape.
        import cuda.bindings.driver as cuda
        dq = torch.zeros(16, HQ, HDIM, dtype=torch.float8_e4m3fn, device=dev)
        dkv = torch.zeros(2, HKV, PAGE, 2 * HDIM, dtype=torch.float8_e4m3fn, device=dev)
        dbt = torch.zeros(1, MAXP, dtype=torch.int32, device=dev)
        do = torch.zeros(16, HQ, HDIM, dtype=torch.bfloat16, device=dev)
        # Compile both runtime-scale modes ahead of the timed path.  The packed
        # mode is the fast default; precise_variant keeps FP32 exp/sum math for
        # high-sensitivity layer scales.
        for v in (variant, precise_variant):
            _COMPILED[v] = cute.compile(
                _OPS[v],
                from_dlpack(dq, assumed_align=16).mark_layout_dynamic(leading_dim=2),
                from_dlpack(dkv, assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(dbt, assumed_align=16).mark_layout_dynamic(leading_dim=1),
                from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
                st["rplan"], st["rbin"], st["rpo"], st["rpml"],
                st["rmrg"], st["rmbin"],
                cutlass.Int64(st["cnt_base"]), cutlass.Float32(0.0625), cutlass.Float32(1.0),
                cutlass.Int32(nwork), cutlass.Int32(nslot), grid,
                cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        del dq, dkv, dbt, do
        torch.cuda.synchronize()
    return st


def run(q, kv_cache, block_tables, seq_lens, cu_seqlens_q, query_lens_cpu,
        seq_lens_cpu, bmm1_scale, bmm2_scale, state, out):
    rQ = from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2)
    rKV = from_dlpack(kv_cache, assumed_align=16).mark_layout_dynamic(leading_dim=3)
    rBT = from_dlpack(block_tables, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    rO = from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=2)
    scale_log2 = cutlass.Float32(float(bmm1_scale) * LOG2E)
    out_scale = cutlass.Float32(float(bmm2_scale))
    nwork = cutlass.Int32(state["nwork"])
    nslot = cutlass.Int32(state["nslot"])
    cnt_base = cutlass.Int64(state["cnt_base"])
    import cuda.bindings.driver as cuda
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    precise = float(bmm1_scale) > 0.0625 or float(bmm2_scale) > 1.0
    variant = state["precise_variant"] if precise else state["variant"]
    _COMPILED[variant](
        rQ, rKV, rBT, rO, state["rplan"], state["rbin"], state["rpo"],
        state["rpml"], state["rmrg"], state["rmbin"],
        cnt_base, scale_log2, out_scale, nwork, nslot, stream)
