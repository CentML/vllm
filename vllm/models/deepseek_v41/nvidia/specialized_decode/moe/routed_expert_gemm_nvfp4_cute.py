# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 (W4A4) routed-expert GEMMs of the DeepSeek-V4.1 low-batch MoE.

The NVFP4 counterparts of ``RoutedExpertGemm1`` / ``RoutedExpertGemm2``
(``routed_expert_gemm_cute.py``), for the tensors vLLM prepares for FlashInfer's
TRT-LLM-gen NVFP4 MoE (``TrtLlmNvFp4ExpertsModular``). The persistent
warp-specialized GEMM, the routing record and the accumulator handling are the
base class's; this module changes the operands and the two epilogues:

* Operands: E2M1 weights (A) times E2M1 activations (B), E4M3 block scales over
  16 values (``tcgen05`` ``kind::mxf4nvf4``). Blackwell/Rubin MMAs cannot mix
  MXFP8 activations with NVFP4 weights (that kind needs UE8M0 scales over 32
  values), so the activations are NVFP4 too, as in the TRT-LLM-gen path.
* The intermediate is not padded: vLLM pads a gated NVFP4 intermediate to a
  multiple of 64 only, so at TP4 it is 576 (w13 has 1152 rows, nine 128-row
  tiles; GEMM2's K is 576).
* GEMM1's epilogue: the per-expert global scales are applied after the
  block-scaled MMA (``alpha`` = ``g1_alphas`` on the gate, ``scale_c`` =
  ``g1_alphas * a2_gscale`` on the up rows), after clamping the raw accumulator
  at the folded limit (``gemm1_clamp_limit`` = limit / ``g1_alphas``), then
  ``h = up * gate * sigmoid(gate)``, quantized to NVFP4 with an E4M3 scale per
  16 columns (amax / 6, global scale 1: ``scale_c`` already holds ``a2_gscale``).
  One 16-column block is exactly one epilogue warp's columns, so the block amax
  needs no exchange between warps.
* GEMM2's epilogue scales the accumulator by the expert's ``g2_alphas`` before
  rounding to BF16.

H's block scales use the per-tile layout of B's scales: one 128-row atom per
8-row tile, of which rows 0-7 are used; with 16-value blocks an atom covers 64
columns (``HS[c * K + (k // 64) * 512 + r * 16 + (k // 16) % 4]``).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .routed_expert_gemm_cute import (
    TILE,
    RoutedExpertGemm,
    _asm,
    abs_max_x2,
    unshuffled_row,
)

SF_VEC = 16
# Columns one 128-row scale atom covers (four scales per row).
ATOM_K = 4 * SF_VEC
# E4M3 1.0 in each byte: the scale of B rows past the routed input.
E4M3_ONES = 0x38383838


def tile_scale_bytes_nvfp4(k: int) -> int:
    """Bytes of one 8-row tile's NVFP4 block scales over ``k`` columns (one
    512-byte atom per 64 columns)."""
    return k // ATOM_K * 512


# Clamp the raw accumulators (up to [-limit, limit], gate to at most limit),
# apply the global scales (up by $8, gate by $9), then up * gate * sigmoid(gate)
# with the instruction sequence of the MXFP8 epilogue (_SWIGLU_X2).
_SWIGLU_NV_X2 = """{
.reg .pred sw_p;
.reg .f32 sw_u0, sw_u1, sw_g0, sw_g1, sw_x0, sw_x1;
.reg .b32 sw_r0, sw_r1;
.reg .b64 sw_up, sw_gate, sw_one, sw_zero, sw_l2e, sw_neg, sw_f, sw_a, sw_d;
setp.lt.ftz.f32 sw_p, $6, $2;
selp.f32 sw_u0, $6, $2, sw_p;
setp.lt.ftz.f32 sw_p, $2, $7;
selp.f32 sw_u0, $7, sw_u0, sw_p;
setp.lt.ftz.f32 sw_p, $6, $3;
selp.f32 sw_u1, $6, $3, sw_p;
setp.lt.ftz.f32 sw_p, $3, $7;
selp.f32 sw_u1, $7, sw_u1, sw_p;
setp.lt.ftz.f32 sw_p, $6, $4;
selp.f32 sw_g0, $6, $4, sw_p;
setp.lt.ftz.f32 sw_p, $4, 0fFF7FFFFF;
selp.f32 sw_g0, 0fFF7FFFFF, sw_g0, sw_p;
setp.lt.ftz.f32 sw_p, $6, $5;
selp.f32 sw_g1, $6, $5, sw_p;
setp.lt.ftz.f32 sw_p, $5, 0fFF7FFFFF;
selp.f32 sw_g1, 0fFF7FFFFF, sw_g1, sw_p;
mul.rn.ftz.f32 sw_u0, sw_u0, $8;
mul.rn.ftz.f32 sw_u1, sw_u1, $8;
mul.rn.ftz.f32 sw_g0, sw_g0, $9;
mul.rn.ftz.f32 sw_g1, sw_g1, $9;
mov.b32 sw_r0, sw_u0;
mov.b32 sw_r1, sw_u1;
mov.b64 sw_up, {sw_r0, sw_r1};
mov.b32 sw_r0, sw_g0;
mov.b32 sw_r1, sw_g1;
mov.b64 sw_gate, {sw_r0, sw_r1};
mov.b64 sw_one, 4575657222473777152;
mov.b64 sw_zero, 0;
mov.b64 sw_l2e, 4591606991553079867;
mov.b64 sw_neg, -4647714812233515008;
fma.rn.f32x2 sw_f, sw_up, sw_one, sw_zero;
mul.rn.f32x2 sw_a, sw_gate, sw_l2e;
mul.rn.f32x2 sw_a, sw_neg, sw_a;
mov.b64 {sw_r0, sw_r1}, sw_a;
mov.b32 sw_x0, sw_r0;
mov.b32 sw_x1, sw_r1;
ex2.approx.ftz.f32 sw_x0, sw_x0;
ex2.approx.ftz.f32 sw_x1, sw_x1;
mov.b32 sw_r0, sw_x0;
mov.b32 sw_r1, sw_x1;
mov.b64 sw_d, {sw_r0, sw_r1};
add.f32x2 sw_d, sw_one, sw_d;
mov.b64 {sw_r0, sw_r1}, sw_d;
mov.b32 sw_x0, sw_r0;
mov.b32 sw_x1, sw_r1;
rcp.approx.ftz.f32 sw_x0, sw_x0;
rcp.approx.ftz.f32 sw_x1, sw_x1;
mov.b32 sw_r0, sw_x0;
mov.b32 sw_r1, sw_x1;
mov.b64 sw_d, {sw_r0, sw_r1};
mul.rn.f32x2 sw_d, sw_gate, sw_d;
mul.rn.f32x2 sw_d, sw_f, sw_d;
mov.b64 {sw_r0, sw_r1}, sw_d;
mov.b32 $0, sw_r0;
mov.b32 $1, sw_r1;
}"""

# Two blocks' amax to their E4M3 scales (amax / 6, round to nearest, saturating)
# and the FP32 reciprocals of the decoded scales (0 for a zero scale). $0 holds
# the two scale bytes, block a in bits 0-7.
_NV_SCALES_X2 = """{
.reg .pred ns_p;
.reg .f32 ns_a, ns_b, ns_fa, ns_fb;
.reg .b16 ns_s, ns_ha, ns_hb;
.reg .b32 ns_h2;
mul.rn.f32 ns_a, $3, 0f3E2AAAAB;
mul.rn.f32 ns_b, $4, 0f3E2AAAAB;
cvt.rn.satfinite.e4m3x2.f32 ns_s, ns_b, ns_a;
cvt.rn.f16x2.e4m3x2 ns_h2, ns_s;
mov.b32 {ns_ha, ns_hb}, ns_h2;
cvt.f32.f16 ns_fa, ns_ha;
cvt.f32.f16 ns_fb, ns_hb;
rcp.approx.ftz.f32 $1, ns_fa;
setp.eq.f32 ns_p, ns_fa, 0f00000000;
@ns_p mov.f32 $1, 0f00000000;
rcp.approx.ftz.f32 $2, ns_fb;
setp.eq.f32 ns_p, ns_fb, 0f00000000;
@ns_p mov.f32 $2, 0f00000000;
cvt.u32.u16 $0, ns_s;
}"""


@dsl_user_op
def clamped_scaled_swiglu_x2(
    up0, up1, gate0, gate1, limit, neg_limit, scale_up, scale_gate, *, loc=None, ip=None
):
    """``h`` of two tokens (``_SWIGLU_NV_X2``)."""
    result = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            Float32(v).ir_value(loc=loc, ip=ip)
            for v in (up0, up1, gate0, gate1, limit, neg_limit, scale_up, scale_gate)
        ],
        _SWIGLU_NV_X2,
        "=f,=f,f,f,f,f,f,f,f,f",
        loc,
        ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [index], loc=loc, ip=ip))
        for index in range(2)
    )


@dsl_user_op
def nvfp4_scales_x2(amax_a, amax_b, *, loc=None, ip=None):
    """The E4M3 scale bytes of two blocks (block a in bits 0-7) and the
    inverses of their decoded values (``_NV_SCALES_X2``)."""
    result = _asm(
        llvm.StructType.get_literal([T.i32(), T.f32(), T.f32()]),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in (amax_a, amax_b)],
        _NV_SCALES_X2,
        "=r,=f,=f,f,f",
        loc,
        ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
    )


@dsl_user_op
def quantize_e2m1x2(lo, hi, inverse, *, loc=None, ip=None) -> Uint32:
    """``lo`` and ``hi`` times the inverse scale (subnormals flushed) as one
    E2M1 byte, ``lo`` in bits 0-3, in the low byte of the result."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (lo, hi, inverse)],
            "{\n.reg .f32 q_a, q_b;\n.reg .b8 q_q;\n"
            "mul.ftz.f32 q_a, $1, $3;\nmul.ftz.f32 q_b, $2, $3;\n"
            "cvt.rn.satfinite.e2m1x2.f32 q_q, q_b, q_a;\n"
            "mov.b32 $0, {q_q, q_q, q_q, q_q};\n}",
            "=r,f,f,f",
            loc,
            ip,
        )
    )


class RoutedExpertGemm1Nvfp4(RoutedExpertGemm):
    """GEMM1: ``h = NVFP4(clamped SwiGLU(x @ w13[expert]^T))`` of each token
    tile, in the tile's permuted rows (module docstring).

    B is the whole routed input ``x`` (at most eight rows; rows past its end
    read as zero) with its linear E4M3 scales, which the B load warp writes
    into the layout the MMA reads. Accumulator column ``t`` is token ``t``;
    the epilogue writes it to the tile row that token was routed to
    (``record[4 + 2 C + 8 c + r] == t``), if any.
    """

    route_before_wait = False

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        cta_k: int = 256,
        num_ab_stage: int = 8,
    ) -> None:
        assert (2 * intermediate) % 128 == 0 and intermediate % ATOM_K == 0
        super().__init__(
            2 * intermediate, 2 * intermediate // 128, hidden, cta_k, num_ab_stage
        )
        self.sf_vec_size = SF_VEC
        self.intermediate = intermediate

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        b_scale_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        hs_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        clamp_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        scale_c_ptr: cute.Pointer,
        rows: Int32,
        experts: Int32,
        cta_words: Int32,
        h_rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w13 (2 x intermediate, hidden, experts), K-major E2M1; B: x
        (rows, hidden) E2M1, ``b_scale_ptr`` its linear E4M3 scales as Uint32
        words (rows, hidden / 64); H (``h_rows``, intermediate / 2) bytes; HS in
        the per-tile layout; ``clamp``, ``alpha``, ``scale_c`` FP32 per
        expert; ``ctas`` persistent CTAs."""
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((self.m, self.k, experts), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((rows, self.k, 1), order=(1, 0, 2))
        )
        words = self.k // ATOM_K
        b_scale = cute.make_tensor(
            b_scale_ptr, cute.make_layout((rows, words), stride=(words, 1))
        )
        h = cute.make_tensor(h_ptr, cute.make_layout(h_rows * (self.intermediate // 2)))
        hs = cute.make_tensor(
            hs_ptr,
            cute.make_layout(
                (h_rows + TILE - 1) // TILE * tile_scale_bytes_nvfp4(self.intermediate)
            ),
        )
        clamp = cute.make_tensor(clamp_ptr, cute.make_layout(experts))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(experts))
        scale_c = cute.make_tensor(scale_c_ptr, cute.make_layout(experts))
        route = cute.make_tensor(
            route_ptr, cute.make_layout(4 + 2 * cta_words + h_rows)
        )
        self._launch(
            a,
            sfa,
            b,
            None,
            route,
            (h, hs, clamp, alpha, scale_c),
            (b_scale,),
            cta_words,
            ctas,
            stream,
        )

    @cute.jit
    def b_tile_index(self, n_tile: Int32) -> Int32:
        """Every item multiplies all of ``x``."""
        return Int32(0)

    @cute.jit
    def write_b_scales(
        self, sSFB: cute.Tensor, stage: Int32, k_tile: Int32, b_side: tuple
    ):
        """Copy the stage's scales of ``x`` (four per 32-bit word, one word per
        64 columns) into rows 0-7 of B's scale atoms; rows past ``x``'s end
        get scale 1."""
        (b_scale,) = b_side
        atoms = self.cta_k // ATOM_K
        stage_bytes = 512 * atoms
        words = cute.make_tensor(
            cute.recast_ptr(sSFB.iterator, dtype=Uint32),
            cute.make_layout(self.num_ab_stage * stage_bytes // 4),
        )
        lane = cute.arch.lane_idx()
        for slot in cutlass.range_constexpr((TILE * atoms + 31) // 32):
            index = lane + 32 * slot
            if index < TILE * atoms:
                row = index // atoms
                atom = index - row * atoms
                value = Uint32(E4M3_ONES)
                if row < cute.size(b_scale, mode=[0]):
                    value = Uint32(b_scale[row, k_tile * atoms + atom])
                words[(stage * stage_bytes + atom * 512 + row * 16) // 4] = value
        # The MMA warp copies them to tensor memory through the async proxy.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()

    @cute.jit
    def store(
        self,
        tDrAcc: cute.Tensor,
        tDcC: cute.Tensor,
        epi: tuple,
        mRoute: cute.Tensor,
        cta_words: Int32,
        n_tile: Int32,
        m_tile: Int32,
        item: Int32,
        exchange: cute.Tensor,
    ):
        """H and HS of the item. The thread holds, for tokens ``t, t + 1``
        (``t`` = 2 x lane % 4), the up and gate rows of H columns ``p``
        (registers 0-3) and ``p + 1`` (4-7), ``p`` = 64 x m_tile + 16 x warp +
        2 x lane / 4: the warp's 16 columns are one NVFP4 block."""
        mH, mHS, mClamp, mAlpha, mScaleC = epi
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - self.epilogue_warp
        lane = cute.arch.lane_idx()
        token = (lane % 4) * 2
        expert = mRoute[4 + n_tile]
        limit = mClamp[expert]
        alpha = mAlpha[expert]
        scale_c = mScaleC[expert]
        row0 = n_tile * TILE
        valid = mRoute[4 + cta_words + n_tile] - row0
        # The tile rows of the thread's two tokens (-1: not routed here).
        row_a = Int32(-1)
        row_b = Int32(-1)
        for r in cutlass.range_constexpr(TILE):
            if r < valid:
                routed = mRoute[4 + 2 * cta_words + row0 + r]
                if routed == token:
                    row_a = Int32(r)
                if routed == token + 1:
                    row_b = Int32(r)

        lo_a, lo_b = clamped_scaled_swiglu_x2(
            tDrAcc[0], tDrAcc[1], tDrAcc[2], tDrAcc[3], limit, -limit, scale_c, alpha
        )
        hi_a, hi_b = clamped_scaled_swiglu_x2(
            tDrAcc[4], tDrAcc[5], tDrAcc[6], tDrAcc[7], limit, -limit, scale_c, alpha
        )
        amax_a = abs_max_x2(lo_a, hi_a)
        amax_b = abs_max_x2(lo_b, hi_b)
        # The block: the 8 lanes of the warp holding the same two tokens.
        for shift in cutlass.range_constexpr(4, 1, -1):
            amax_a = cute.arch.fmax(
                amax_a, cute.arch.shuffle_sync_bfly(amax_a, 1 << shift), ftz=True
            )
            amax_b = cute.arch.fmax(
                amax_b, cute.arch.shuffle_sync_bfly(amax_b, 1 << shift), ftz=True
            )
        codes, inverse_a, inverse_b = nvfp4_scales_x2(amax_a, amax_b)

        h_row = self.intermediate // 2
        column = m_tile * 32 + warp * 8 + lane // 4
        if row_a >= 0:
            mH[(row0 + row_a) * h_row + column] = Uint8(
                quantize_e2m1x2(lo_a, hi_a, inverse_a) & Uint32(0xFF)
            )
        if row_b >= 0:
            mH[(row0 + row_b) * h_row + column] = Uint8(
                quantize_e2m1x2(lo_b, hi_b, inverse_b) & Uint32(0xFF)
            )
        if lane < 4:
            tile_bytes = tile_scale_bytes_nvfp4(self.intermediate)
            # Block m_tile * 4 + warp: atom m_tile, byte warp of the row.
            base = n_tile * tile_bytes + m_tile * 512 + warp
            if row_a >= 0:
                mHS[base + row_a * 16] = Uint8(codes & Uint32(0xFF))
            if row_b >= 0:
                mHS[base + row_b * 16] = Uint8((codes >> Uint32(8)) & Uint32(0xFF))


class RoutedExpertGemm2Nvfp4(RoutedExpertGemm):
    """GEMM2: the BF16 rows ``g2_alphas[expert] * (h @ w2[expert]^T)`` of each
    token tile, in the tile's permuted order."""

    def __init__(
        self, hidden: int, intermediate: int, cta_k: int = 64, num_ab_stage: int = 16
    ) -> None:
        assert intermediate % cta_k == 0
        super().__init__(hidden, hidden // 128, intermediate, cta_k, num_ab_stage)
        self.sf_vec_size = SF_VEC

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        rows: Int32,
        experts: Int32,
        cta_words: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w2 (hidden, intermediate, experts), K-major E2M1; B: H (rows,
        intermediate) E2M1, K-major, ``rows`` the padded rows; C: (rows,
        hidden) BF16; ``alpha`` FP32 per expert; ``ctas`` persistent CTAs."""
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((self.m, self.k, experts), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((rows, self.k, 1), order=(1, 0, 2))
        )
        padded = cute.ceil_div(rows, TILE) * 128
        sfb = cute.make_tensor(
            sfb_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                cute.make_ordered_layout((padded, self.k, 1), order=(1, 0, 2)).shape,
                self.sf_vec_size,
            ),
        )
        c = cute.make_tensor(
            c_ptr, cute.make_ordered_layout((self.m, rows), order=(0, 1))
        )
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(experts))
        route = cute.make_tensor(route_ptr, cute.make_layout(4 + 2 * cta_words))
        self._launch(a, sfa, b, sfb, route, (c, alpha), (), cta_words, ctas, stream)

    @cute.jit
    def store(
        self,
        tDrAcc: cute.Tensor,
        tDcC: cute.Tensor,
        epi: tuple,
        mRoute: cute.Tensor,
        cta_words: Int32,
        n_tile: Int32,
        m_tile: Int32,
        item: Int32,
        exchange: cute.Tensor,
    ):
        """Scale by the expert's ``g2_alphas``, round to BF16 (nearest even) and
        store the tile's rows below its limit at their unshuffled hidden
        index."""
        mC, mAlpha = epi
        alpha = mAlpha[mRoute[4 + n_tile]]
        row_limit = mRoute[4 + cta_words + n_tile]
        tDrS = cute.make_rmem_tensor(tDrAcc.shape, cutlass.Float32)
        for i in cutlass.range_constexpr(cute.size(tDrAcc)):
            tDrS[i] = tDrAcc[i] * alpha
        tDrC = cute.make_rmem_tensor(tDrAcc.shape, cutlass.BFloat16)
        tDrC.store(tDrS.load().to(cutlass.BFloat16))
        for i in cutlass.range_constexpr(cute.size(tDrC)):
            coord = tDcC[i]
            if coord[1] < row_limit:
                mC[unshuffled_row(coord[0]), coord[1]] = tDrC[i]
