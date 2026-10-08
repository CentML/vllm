# SPDX-License-Identifier: Apache-2.0
"""vLLM launch of the vendored FlashInfer u-cache flush GDN decode kernel (gdn_ucache_flush.py).

Contract (one call per GDN layer per step; all tensors on one device):
  q, k     [n, 4, H, 128]  bf16 compact (staged by gsc ucache_prep)    v [n, 4, HV, 128] bf16 compact
  a, b     [n, 4, HV]      bf16 compact                                 A_log [HV] fp32, dt_bias [HV]
  state    [pool, HV, 128, 128] bf16, block-strided view of the vLLM page (checkpoint S0; written on flush rows)
  sidx     [n] int32  state block per row (< 0: padded row, the CTA exits before any memory access)
  ridx     [n] int32  ring slot per row (request-table slot)
  kring    [S, H, 32, 128] bf16, uring [S, HV, 32, 128] bf16, gring [S, HV, 32] fp32 (zero-initialised)
  hist, base [n] int32  ring window length P (<= 16) / origin (< 32), computed by ucache_prep (caller-owned cursors)
  out      [n, 4, HV, 128] bf16  raw delta-rule outputs (the gated RMSNorm runs in gsc ucache_norm)
The kernel is compiled once per (dtype set, HV, H, min_blocks) with tvm-ffi (low host cost per launch); n, pool and
the pool strides are dynamic, so one cubin serves every batch size, layer and graph size.
"""
import functools
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MBP = int(os.environ.get("VLLM_GDN_UCACHE_MBP", "7"))
_COMPILED: dict = {}


@functools.cache
def _mods():
    import cuda.bindings.driver as cuda  # noqa: F401
    import cutlass  # noqa: F401
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from . import gdn_ucache_flush as K

    # The ring may be fp16 (GDN_UCACHE_RING_DTYPE=fp16; gsc allocates its rings to match).
    assert K.IO_TORCH == torch.bfloat16 and K.ST_TORCH == torch.bfloat16 and K.RING_TORCH in (torch.bfloat16, torch.float16), (
        "u-cache kernel must be imported with bf16 IO/state/ring dtypes (unset GDN_UCACHE_*_DTYPE)")
    return cute, from_dlpack, K


def _dyn0(fd, t):
    x = fd(t, assumed_align=16)
    x.mark_compact_shape_dynamic(mode=0, stride_order=tuple(range(t.dim())), divisibility=1)
    return x


def _anyl(fd, t):
    """strided inputs: the kernel addresses them from the base pointer with compile-time strides (qkv_row_stride /
    ab_t_stride), so the tensor layout is irrelevant -> fully dynamic (one cubin for every batch size)"""
    return fd(t, assumed_align=16 if t.dtype != torch.float32 else 4).mark_layout_dynamic()


def _compile(key, args_t, scale, HV, H, flush_min, stream, qkv_rs, ab_ts, fused=False, sigmoid=False,
             norm_args=None, si=None, pdl=False, qo_args=None, cv_args=None, kq_fix=False):
    cute, fd, K = _mods()
    (q, k, v, a, b, A_log, dt_bias, state, sidx, ridx, kring, uring, gring, hist, base, out) = args_t
    gate, nw, cu, gate_row, eps = norm_args
    # block-strided page view: STATIC layout (the state TMA partition needs static modes; FlashInfer's own wrapper
    # uses static descriptors for paged pools too). The pool shape / block stride are in the compile key; they are
    # fixed for a server's lifetime (all GDN layers share them), so this compiles once.
    assert state.stride(3) == 1 and state.stride(2) == 128 and state.stride(1) == 128 * 128
    assert (state.stride(0) * 2) % 16 == 0 and state.data_ptr() % 16 == 0
    st = fd(state, assumed_align=16)
    if qkv_rs:
        qkv_ab = [_anyl(fd, q), _anyl(fd, k), _anyl(fd, v),
                  fd(a, assumed_align=2).mark_layout_dynamic(), fd(b, assumed_align=2).mark_layout_dynamic()]
    else:
        qkv_ab = [_dyn0(fd, q), _dyn0(fd, k), _dyn0(fd, v), _dyn0(fd, a), _dyn0(fd, b)]
    cargs = [*qkv_ab,
             fd(A_log, assumed_align=4), fd(dt_bias, assumed_align=2), st,
             fd(sidx, assumed_align=4).mark_layout_dynamic(), fd(ridx, assumed_align=4).mark_layout_dynamic(),
             _dyn0(fd, kring), _dyn0(fd, uring), _dyn0(fd, gring),
             fd(hist, assumed_align=4).mark_layout_dynamic(), fd(base, assumed_align=4).mark_layout_dynamic(),
             _dyn0(fd, out), scale, HV, 128, H, flush_min,
             fd(gate, assumed_align=2).mark_layout_dynamic(), fd(nw, assumed_align=4).mark_layout_dynamic(),
             fd(cu, assumed_align=4).mark_layout_dynamic(), int(gate_row), float(eps),
             fd(si, assumed_align=4).mark_layout_dynamic(),
             fd(qo_args[0], assumed_align=4).mark_layout_dynamic(), fd(qo_args[1], assumed_align=4).mark_layout_dynamic(),
             *[int(x) for x in qo_args[2:6]],  # (stride, psc, T, pm); qo_args[6] is the compile-time flag
             fd(cv_args[0], assumed_align=4).mark_layout_dynamic(), fd(cv_args[1], assumed_align=16).mark_layout_dynamic(),
             fd(cv_args[2], assumed_align=4).mark_layout_dynamic(), fd(cv_args[3], assumed_align=4).mark_layout_dynamic(),
             int(cv_args[4]), int(cv_args[5]), stream]
    kern = K.GdnDecodeUCacheFlushKernel(disable_state_update=True, min_blocks_per_mp=_MBP, t_input=4, n_valid=4,
                                        qkv_row_stride=int(qkv_rs), ab_native=True, ab_t_stride=int(ab_ts),
                                        pdl_trigger=False, fused_norm=bool(fused), norm_sigmoid=bool(sigmoid),
                                        pdl_wait=bool(pdl), si_stride=int(si.stride(0)) if pdl else 1,
                                        state_slot_stride=int(state.stride(0)), fused_qo=bool(qo_args[6]),
                                        fused_conv=bool(cv_args[6]), kq_fix=bool(kq_fix))
    if torch.cuda.is_current_stream_capturing():
        logger.warning("gdn ucache: compiling the decode kernel during CUDA graph capture (%s)", key)
    logger.info("gdn ucache: compiling the u-cache flush decode kernel %s", key)
    return cute.compile(kern, *cargs, options="--enable-tvm-ffi --opt-level 3")


_DUMMY: dict = {}


def _norm_dummies(out):
    """unfused launches still pass the (unused) fused-norm operands"""
    d = _DUMMY.get(out.device)
    if d is None:
        d = _DUMMY[out.device] = (torch.zeros(128, dtype=torch.bfloat16, device=out.device),
                                  torch.zeros(128, dtype=torch.float32, device=out.device),
                                  torch.zeros(2, dtype=torch.int32, device=out.device))
    return (d[0], d[1], d[2], 0, 0.0)


def ucache_decode(q, k, v, a, b, A_log, dt_bias, state, sidx, ridx, kring, uring, gring, hist, base, out,
                  scale: float, flush_min: int, qkv_rs: int = 0, ab_ts: int = 0, H: int = 0, norm=None,
                  pdl_si=None, qo=None, conv=None, compile_only: bool = False, kq_fix: bool = False) -> None:
    """kq_fix (VLLM_GDN_UCACHE_KQFIX): raw q/k MMA operands + fp32 inverse-norm scalars; the k ring holds
    raw k and the u ring u~ = u * inv|k| (fold form unchanged). Rings written with and without kq_fix are NOT
    interchangeable: the flag must be process-wide (it is part of the compile key)."""
    """norm = (gate [n_tok, HV, 128] bf16 (stride(1) == 128, stride(2) == 1), weight [128] fp32, cu_seqlens [n+1]
    int32, eps, sigmoid) -> fused gated RMSNorm: `out` is then the FINAL [n_tok/4, 4, HV, 128] output and
    rows with sidx < 0 get zeros for their tokens (gsc ucache_norm semantics). Strided (qkv_rs > 0) inputs only."""
    """qkv_rs == 0: q/k/v/a/b are the compact staged [n, 4, ...] tensors. qkv_rs > 0 (strided): q/k/v are the column
    slices [T, H*128] / [T, HV*128] of the decode rows' mixed_qkv (row r = tokens 4r..4r+3, token stride qkv_rs),
    a/b are [T, HV] with token stride ab_ts; nothing is staged (ucache_prep verified cu_seqlens[r] == 4r)."""
    import cuda.bindings.driver as cuda

    n = sidx.size(0)
    if n == 0:
        return
    HV = out.size(2)
    if not qkv_rs:
        H = q.size(2)
        assert q.size(1) == 4 and q.size(3) == 128 and v.size(3) == 128
    assert H > 0 and out.size(1) == 4 and out.size(3) == 128
    assert q.dtype == torch.bfloat16 and state.dtype == torch.bfloat16 and A_log.dtype == torch.float32
    assert 1 <= flush_min <= 13
    # model parameters (A_log / dt_bias are nn.Parameters) cannot be exported through DLPack while they require grad
    A_log, dt_bias = A_log.detach(), dt_bias.detach()
    fused = norm is not None
    if fused:
        gate, nw, cu, eps, sigmoid = norm
        assert qkv_rs > 0 and gate.stride(2) == 1 and gate.stride(1) == 128 and nw.dtype == torch.float32
        norm_args = (gate, nw, cu, int(gate.stride(0)), float(eps))
    else:
        sigmoid = False
        norm_args = _norm_dummies(out)
    # qo = (q [>= pm, HV*128] uint8, sf [>= pm*psc] uint8, psc, T, pm): fused out_proj MXFP8 quant
    if qo is not None:
        q8, sf8 = qo[0].view(torch.uint8), qo[1].view(torch.uint8)  # e4m3 / e8m0 buffers as raw bytes
        assert norm is not None and q8.dim() == 2 and q8.stride(1) == 1 and sf8.is_contiguous()
        assert q8.stride(0) % 4 == 0 and q8.data_ptr() % 4 == 0 and sf8.data_ptr() % 4 == 0
        qo_args = (q8, sf8, int(q8.stride(0)), int(qo[2]), int(qo[3]), int(qo[4]), True)
    else:
        d8 = _DUMMY.setdefault((out.device, "u8"), torch.zeros(16, dtype=torch.uint8, device=out.device))
        qo_args = (d8.view(1, 16), d8, 16, 4, 0, 0, False)
    # conv = (conv_state SD [slots, state_len, dim] bf16 (dim stride 1), weight [dim, 4] bf16, num_accepted
    # [n] int32): fold the causal_conv1d_update into the kernel; q/k/v must then be the RAW mixed_qkv slices.
    if conv is not None:
        cs, cw, nacc_t = conv
        assert norm is not None and qkv_rs > 0 and cs.dtype == torch.bfloat16 and cs.stride(2) == 1
        assert cw.dtype == torch.bfloat16 and cw.is_contiguous() and cw.size(1) == 4 and nacc_t.dtype == torch.int32
        ctr = _DUMMY.get((out.device, "cvctr"))
        if ctr is None or ctr.numel() < sidx.size(0) * H * 4:
            ctr = _DUMMY[(out.device, "cvctr")] = torch.zeros(max(4096, sidx.size(0)) * max(H, 1) * 4,
                                                             dtype=torch.int32, device=out.device)
        cv_args = (cs, cw, nacc_t, ctr, int(cs.stride(0)), int(cs.stride(1)), True)
    else:
        dcs = _DUMMY.setdefault((out.device, "cvbf"), torch.zeros(8, dtype=torch.bfloat16, device=out.device))
        di = _DUMMY.setdefault((out.device, "cvi"), torch.zeros(4, dtype=torch.int32, device=out.device))
        cv_args = (dcs.view(1, 1, 8), dcs.view(2, 4), di, di, 8, 8, False)
    pdl = pdl_si is not None  # PDL dependent of ucache_prep; pdl_si = the caller's state_indices [n, w]
    si = pdl_si.view(-1) if pdl else _norm_dummies(out)[2]
    key = (q.device.index, H, HV, str(dt_bias.dtype), _MBP, tuple(state.shape), tuple(state.stride()), int(qkv_rs),
           int(ab_ts), fused, bool(sigmoid), pdl, int(pdl_si.stride(0)) if pdl else 1, bool(qo_args[6]),
           bool(cv_args[6]), bool(kq_fix))
    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    args_t = (q, k, v, a, b, A_log, dt_bias, state, sidx, ridx, kring, uring, gring, hist, base, out)
    c = _COMPILED.get(key)
    if compile_only and c is not None:
        return
    if c is None:
        c = _COMPILED[key] = _compile(key, args_t, float(scale), HV, H, int(flush_min), stream, qkv_rs, ab_ts,
                                      fused, sigmoid, norm_args, si, pdl, qo_args, cv_args, kq_fix)
    if compile_only:  # Warm a variant before serving needs it (no launch).
        return
    c(*args_t, float(scale), HV, 128, H, int(flush_min), *norm_args, si, qo_args[0], qo_args[1], *qo_args[2:6],
      cv_args[0], cv_args[1], cv_args[2], cv_args[3], int(cv_args[4]), int(cv_args[5]), stream)
