# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN spec-decode (MTP) on FlashInfer's ReplaySSM ring (VLLM_GDN_MTP_RING=1, H46).

FlashInfer 0.6.18.post1 ``gated_delta_rule_mtp_ucache_flush`` keeps, per request, a
checkpoint state S0 plus a ring of the accepted tokens' (normed k, u, cumulative
log-decay) and writes S0 back only when the ring is folded (flush), instead of one
state snapshot per draft token. Its native draft length is T in {4, 8}; the k + 1
verify tokens are gathered into zero-padded ``[B, T_pad, ...]`` buffers.

This module holds the per-layer helpers: the fused gather of the spec tokens into
the padded kernel layout.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

GDN_MTP_RING = os.environ.get("VLLM_GDN_MTP_RING", "0") == "1"
# VLLM_GDN_MTP_RING_LEAN=1 (H102, needs VLLM_GDN_MTP_RING=1): the align-copy
# replacements (PRE/POST, ring_copy_work) run on a fixed grid of
# _RING_COPY_SLOTS programs per (layer, tile) that loop over the compact work
# list, instead of one program row per batch row (num_reqs x 30 x 128 programs
# that nearly all exit at the count check).
GDN_MTP_RING_LEAN = os.environ.get("VLLM_GDN_MTP_RING_LEAN", "0") == "1"
_RING_COPY_SLOTS = 4
# VLLM_GDN_MTP_RING_CONV_OUT=1 (H110 stage 2a, needs VLLM_GDN_MTP_RING=1): the
# spec-decode conv writes its rows straight into the ring's padded q/k/v
# buffers and copies a/b beside them (_ring_conv_kernel, ring_conv), so the
# _ring_gather_kernel pass (a copy of the conv output) is skipped.
GDN_MTP_RING_CONV_OUT = os.environ.get("VLLM_GDN_MTP_RING_CONV_OUT", "0") == "1"


@triton.jit
def _ring_gather_kernel(
    mqkv_ptr,  # [L, 2 * HK + HVV] (row stride s_m)
    s_m,
    a_ptr,  # [L, HV] (row stride s_a)
    s_a,
    b_ptr,  # [L, HV] (row stride s_b)
    s_b,
    cu_ptr,  # [B + 1] int32 token offsets of the B requests
    q_ptr,  # [B, TP, HK] contiguous
    k_ptr,  # [B, TP, HK]
    v_ptr,  # [B, TP, HVV]
    ao_ptr,  # [B, TP, HV]
    bo_ptr,  # [B, TP, HV]
    HK: tl.constexpr,
    HVV: tl.constexpr,
    HV: tl.constexpr,
    TP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Program (r, t, c): column chunk c of token t of request r -> padded row
    r * TP + t; rows t >= T_r (and FULL-graph padding requests) are zero-filled."""
    r = tl.program_id(0)
    t = tl.program_id(1)
    c = tl.program_id(2)
    bos = tl.load(cu_ptr + r)
    n = tl.load(cu_ptr + r + 1) - bos
    valid = t < n
    src = (bos + t).to(tl.int64)
    dst = (r * TP + t).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    x = tl.load(mqkv_ptr + src * s_m + c * BLOCK + cols, mask=valid, other=0.0)
    if c * BLOCK < HK:
        tl.store(q_ptr + dst * HK + c * BLOCK + cols, x)
    elif c * BLOCK < 2 * HK:
        tl.store(k_ptr + dst * HK + (c * BLOCK - HK) + cols, x)
    else:
        tl.store(v_ptr + dst * HVV + (c * BLOCK - 2 * HK) + cols, x)
    if c == 0:
        hcols = tl.arange(0, HV)
        tl.store(ao_ptr + dst * HV + hcols, tl.load(a_ptr + src * s_a + hcols, mask=valid, other=0.0))
        tl.store(bo_ptr + dst * HV + hcols, tl.load(b_ptr + src * s_b + hcols, mask=valid, other=0.0))


def ring_gather(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    t_pad: int,
    out: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Gather the spec tokens (request r: rows [cu[r], cu[r+1])) of ``mixed_qkv``
    and ``a``/``b`` into zero-padded contiguous q/k ``[B, t_pad, H, K]``, v
    ``[B, t_pad, HV, V]``, a/b ``[B, t_pad, HV]`` (``out``: reuse these buffers)."""
    B = cu_seqlens.numel() - 1
    HK = num_k_heads * head_k_dim
    HVV = num_v_heads * head_v_dim
    assert mixed_qkv.stride(-1) == 1 and a.stride(-1) == 1 and b.stride(-1) == 1
    assert mixed_qkv.shape[-1] == 2 * HK + HVV and a.shape[-1] == num_v_heads
    block = 1024
    assert HK % block == 0 and HVV % block == 0
    nchunk = (2 * HK + HVV) // block
    if out is None:
        dev, dt = mixed_qkv.device, mixed_qkv.dtype
        out = (
            torch.empty(B, t_pad, num_k_heads, head_k_dim, device=dev, dtype=dt),
            torch.empty(B, t_pad, num_k_heads, head_k_dim, device=dev, dtype=dt),
            torch.empty(B, t_pad, num_v_heads, head_v_dim, device=dev, dtype=dt),
            torch.empty(B, t_pad, num_v_heads, device=dev, dtype=a.dtype),
            torch.empty(B, t_pad, num_v_heads, device=dev, dtype=b.dtype),
        )
    q, k, v, ao, bo = out
    if B > 0:
        _ring_gather_kernel[(B, t_pad, nchunk)](
            mixed_qkv, mixed_qkv.stride(0), a, a.stride(0), b, b.stride(0), cu_seqlens,
            q, k, v, ao, bo,
            HK=HK, HVV=HVV, HV=num_v_heads, TP=t_pad, BLOCK=block, num_warps=2,
        )
    return out


@triton.jit
def _ring_compact_kernel(
    y_ptr,  # [B, TP, W] contiguous (W = HV * V)
    cu_ptr,
    x_ptr,  # [L, W] contiguous
    W: tl.constexpr,
    TP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    t = tl.program_id(1)
    c = tl.program_id(2)
    bos = tl.load(cu_ptr + r)
    n = tl.load(cu_ptr + r + 1) - bos
    if t >= n:
        return
    cols = c * BLOCK + tl.arange(0, BLOCK)
    src = (r * TP + t).to(tl.int64)
    dst = (bos + t).to(tl.int64)
    tl.store(x_ptr + dst * W + cols, tl.load(y_ptr + src * W + cols))


def ring_compact(y: torch.Tensor, cu_seqlens: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The kernel's padded output ``[B, t_pad, HV, V]`` -> token rows ``x[L, HV, V]``."""
    B, TP = y.shape[0], y.shape[1]
    W = y.shape[2] * y.shape[3]
    assert y.is_contiguous() and x.is_contiguous() and W % 1024 == 0
    if B > 0:
        _ring_compact_kernel[(B, TP, W // 1024)](y, cu_seqlens, x, W=W, TP=TP, BLOCK=1024, num_warps=2)
    return x


# --------------------------------------------------------------------------
# Ring runtime (design P). Per request slot (the V2 runner's persistent
# req_state index, < max_num_reqs) and GDN layer: a checkpoint S0 and the
# k/u/g rings, outside vLLM's block pool. Cursors (hist_len, cache_base) and
# ownership are per slot and shared by every layer. A slot is "vLLM-owned"
# (its state lives in vLLM's running block, as without the ring) until its
# first spec-decode step imports it; a non-spec step exports it back. The
# align-mode copies read blocks: for ring-owned slots they become folds of the
# ring (_ring_state_kernel).
# --------------------------------------------------------------------------
import numpy as np  # noqa: E402

_STATE: list = []  # [RingState]
_MODE_POST, _MODE_PRE, _MODE_IMPORT, _MODE_EXPORT = 0, 1, 2, 3


def ring_t_pad(num_tokens: int) -> int:
    return 4 if num_tokens <= 4 else 8


class RingState:
    """Process-wide ring state: slot cursors and flags, the shared padded
    buffers, and the registered layers' pools."""

    def __init__(self, num_slots: int, device: torch.device):
        import flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_flush as uc

        self.uc = uc
        self.ring_slots = uc.RING_SLOTS
        self.w_ring = uc.W_RING
        self.num_slots = num_slots
        self.device = device
        i32 = dict(dtype=torch.int32, device=device)
        self.hl = torch.zeros(num_slots, **i32)  # committed cursors per slot
        self.cb = torch.zeros(num_slots, **i32)
        self.hl0 = torch.zeros(num_slots, **i32)  # this step's cursors per slot
        self.cb0 = torch.zeros(num_slots, **i32)
        self.ran = torch.zeros(num_slots, **i32)  # slot ran the ring this step
        self.owner = torch.zeros(num_slots, **i32)  # 1: ring-owned
        self.owner_np = np.zeros(num_slots, dtype=np.bool_)
        # per spec row of the step (row order of the GDN spec metadata)
        self.slots_step = torch.full((num_slots,), -1, **i32)
        self.hl_step = torch.zeros(num_slots, **i32)
        self.cb_step = torch.zeros(num_slots, **i32)
        self.spec_np = torch.zeros(num_slots, dtype=torch.int32, pin_memory=True)
        self.import_np = torch.zeros(num_slots, dtype=torch.int32, pin_memory=True)
        self.spec_dev = torch.zeros(num_slots, **i32)
        self.import_dev = torch.zeros(num_slots, **i32)
        self.items_np = torch.zeros(2 * num_slots, dtype=torch.int32, pin_memory=True)
        self.items_dev = torch.zeros(2 * num_slots, **i32)
        self.layers: list = []  # registered GDN layers (in registration order)
        self.bufs: tuple | None = None  # padded gather / output buffers
        self.t_pad = 0
        self.prepared = False
        self.bound = False
        self._calls: dict = {}
        logger.info(
            "GDN MTP ring (VLLM_GDN_MTP_RING=1, H46): FlashInfer ucache_flush, "
            "%d request slots per layer, W_RING %d.", num_slots, self.w_ring
        )

    # ---------------------------------------------------------------- layers
    def register(self, layer, num_k_heads, head_k_dim, num_v_heads, head_v_dim, t_pad):
        """Pools of ``layer`` ([num_slots, ...]) and its bf16 A_log / dt_bias
        (the kernel reads them as bf16; the checkpoint stores bf16, so exact)."""
        assert head_k_dim == 128 and head_v_dim == 128, "ring kernel: K = V = 128"
        if getattr(layer, "_ring", None) is not None:
            return
        n, dev, rs = self.num_slots, self.device, self.ring_slots
        bf = dict(dtype=torch.bfloat16, device=dev)
        a_log = layer.A_log.detach().to(torch.bfloat16).contiguous()
        dt_bias = layer.dt_bias.detach().to(torch.bfloat16).contiguous()
        assert torch.equal(a_log.float(), layer.A_log.detach().float()), (
            "VLLM_GDN_MTP_RING: A_log is not bf16-exact (the ring kernel reads bf16)"
        )
        assert torch.equal(dt_bias.float(), layer.dt_bias.detach().float()), (
            "VLLM_GDN_MTP_RING: dt_bias is not bf16-exact"
        )
        layer._ring = dict(
            s0=torch.zeros(n, num_v_heads, head_v_dim, head_k_dim, **bf),
            kc=torch.zeros(n, num_k_heads, rs, head_k_dim, **bf),
            uc=torch.zeros(n, num_v_heads, rs, head_v_dim, **bf),
            gc=torch.zeros(n, num_v_heads, rs, dtype=torch.float32, device=dev),
            a_log=a_log,
            dt_bias=dt_bias,
            H=num_k_heads,
            HV=num_v_heads,
            index=len(self.layers),
        )
        self.layers.append(layer)
        if self.bufs is None:
            self.t_pad = t_pad
            self.bufs = (
                torch.zeros(n, t_pad, num_k_heads, head_k_dim, **bf),
                torch.zeros(n, t_pad, num_k_heads, head_k_dim, **bf),
                torch.zeros(n, t_pad, num_v_heads, head_v_dim, **bf),
                torch.zeros(n, t_pad, num_v_heads, **bf),
                torch.zeros(n, t_pad, num_v_heads, **bf),
                torch.zeros(n, t_pad, num_v_heads, head_v_dim, **bf),  # output
            )
        assert self.t_pad == t_pad

    @property
    def flush_min(self) -> int:
        return self.w_ring - self.t_pad + 1

    # ---------------------------------------------------------------- kernel
    def call(self, layer, B: int, scale: float) -> torch.Tensor:
        """The ring kernel on the gathered rows [:B] of ``layer``; returns the
        raw output view [B, t_pad, HV, V]. The first call per (layer, B) goes
        through FlashInfer's wrapper (compiles, validates); later calls reuse
        its compiled kernel and the argument descriptors (all buffers are
        persistent), avoiding the wrapper's per-call host work."""
        uc = self.uc
        r = layer._ring
        q, k, v, a, b, y = (t[:B] for t in self.bufs)
        idx, hl, cb = self.slots_step[:B], self.hl_step[:B], self.cb_step[:B]
        key = (r["index"], B)
        c = self._calls.get(key)
        stream = uc.cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if c is None:
            uc.gated_delta_rule_mtp_ucache_flush(
                r["a_log"], a, r["dt_bias"], q=q, k=k, v=v, b=b,
                initial_state_source=r["s0"], initial_state_indices=idx,
                k_cache=r["kc"], u_cache=r["uc"], g_cache=r["gc"], hist_len=hl,
                cache_base=cb, scale=scale, output=y, flush_min=self.flush_min,
                restart_hist_on_flush=False,
            )
            HV, H = r["HV"], r["H"]
            num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
            mbp = max(1, min(-(-(HV * B) // num_sms) + 1, 7))
            if os.environ.get("GDN_WY_MBP"):
                mbp = int(os.environ["GDN_WY_MBP"])
            t = self.t_pad
            ckey = (
                "ucache-flush-v2", str(uc.IO_TORCH), str(uc.ST_TORCH),
                str(uc.RING_TORCH), str(q.device), mbp, t, t, 0, True,
                a.stride(1), False, HV, H, 128,
            )
            fn = uc._CACHE.get(ckey)
            assert fn is not None, f"ring fast path: no compiled kernel for {ckey}"

            def mk(x):
                return uc.from_dlpack(x, 16)

            def mk_dyn(x):
                return mk(x).mark_compact_shape_dynamic(
                    mode=0, stride_order=tuple(range(x.dim())), divisibility=1
                )

            args = [
                mk_dyn(q), mk_dyn(k), mk_dyn(v), mk_dyn(a), mk_dyn(b),
                mk(r["a_log"]), mk(r["dt_bias"]), mk_dyn(r["s0"]), mk_dyn(idx),
                mk_dyn(r["kc"]), mk_dyn(r["uc"]), mk_dyn(r["gc"]), mk_dyn(hl),
                mk_dyn(cb), mk_dyn(y), scale, HV, 128, H, int(self.flush_min),
            ]
            # keep the tensors alive with their descriptors
            self._calls[key] = (fn, args, (q, k, v, a, b, y, idx, hl, cb))
            if len(self._calls) == 1:
                logger.info(
                    "GDN MTP ring: first kernel call (T_pad %d, flush_min %d, "
                    "%d layers).", self.t_pad, self.flush_min, len(self.layers)
                )
            return y
        fn, args, _ = c
        fn(*args, stream)
        return y


def ring_state() -> RingState | None:
    return _STATE[0] if _STATE else None


def ring_register(layer, num_slots, num_k_heads, head_k_dim, num_v_heads,
                  head_v_dim, num_spec_tokens) -> None:
    """Called by each GDN layer's warm-up (weights loaded, before the KV cache
    is sized): creates the process state and the layer's pools."""
    if not _STATE:
        _STATE.append(RingState(num_slots, layer.A_log.device))
    _STATE[0].register(layer, num_k_heads, head_k_dim, num_v_heads, head_v_dim,
                       ring_t_pad(num_spec_tokens + 1))


def ring_core(layer, mixed_qkv, a, b, cu_seqlens, num_requests: int) -> torch.Tensor:
    """Gather + ring kernel for the spec rows (post-conv ``mixed_qkv`` rows,
    request r at [cu[r], cu[r+1])); returns the raw output [B, t_pad, HV, V]."""
    st = _STATE[0]
    r = layer._ring
    conv_rows = layer.__dict__.pop("_ring_conv_rows", None)
    ab_rows = layer.__dict__.pop("_ring_ab_rows", None)
    if conv_rows is None:
        ring_gather(mixed_qkv, a, b, cu_seqlens, r["H"], 128, r["HV"], 128, st.t_pad,
                    out=tuple(x[:num_requests] for x in st.bufs[:5]))
    else:
        # VLLM_GDN_MTP_RING_CONV_OUT: ring_conv already wrote the padded q/k/v
        # rows; the a/b rows were copied at the b/a join, else copy them here.
        assert conv_rows == num_requests, (conv_rows, num_requests)
        if ab_rows is None:
            ring_ab(layer, a, b, cu_seqlens, num_requests, mark=False)
        else:
            assert ab_rows == num_requests, (ab_rows, num_requests)
    return st.call(layer, num_requests, layer.head_k_dim ** -0.5)


@triton.jit
def _ring_ab_kernel(a_ptr, s_a, b_ptr, s_b, cu_ptr, ao_ptr, bo_ptr,
                    HV: tl.constexpr, TP: tl.constexpr):
    """_ring_gather_kernel's a/b writes: request r's rows t < n -> padded row
    r * TP + t, zero rows t >= n."""
    r = tl.program_id(0)
    bos = tl.load(cu_ptr + r)
    n = tl.load(cu_ptr + r + 1) - bos
    t = tl.arange(0, TP)
    h = tl.arange(0, HV)
    tok = (t < n)[:, None]
    src = (bos + t).to(tl.int64)[:, None]
    dst = (r * TP + t).to(tl.int64)[:, None] * HV + h[None, :]
    tl.store(ao_ptr + dst, tl.load(a_ptr + src * s_a + h[None, :], mask=tok, other=0.0))
    tl.store(bo_ptr + dst, tl.load(b_ptr + src * s_b + h[None, :], mask=tok, other=0.0))


_ring_ab_launch: list = []


def ring_ab(layer, a: torch.Tensor, b: torch.Tensor, cu_seqlens: torch.Tensor,
            num_requests: int, mark: bool = True) -> None:
    """VLLM_GDN_MTP_RING_CONV_OUT: the ring's padded a/b rows of the spec
    requests (request r at rows [cu[r], cu[r+1]) of ``a``/``b``), i.e. the
    a/b part of ring_gather. Launched on the current stream: the aux stream
    behind the b/a GEMM when called from the deferred join, so it overlaps the
    conv. ``mark``: tell ring_core it is done."""
    st = _STATE[0]
    hv = layer._ring["HV"]
    assert a.stride(-1) == 1 and b.stride(-1) == 1 and a.size(-1) == hv == b.size(-1)
    if not _ring_ab_launch:
        from vllm.model_executor.layers.mamba.ops.gdn_host_trim import launcher

        _ring_ab_launch.append(launcher(_ring_ab_kernel))
    if num_requests > 0:
        _ring_ab_launch[0][(num_requests,)](
            a, a.stride(0), b, b.stride(0), cu_seqlens, st.bufs[3], st.bufs[4],
            HV=hv, TP=st.t_pad, num_warps=1,
        )
    if mark:
        layer._ring_ab_rows = num_requests


@triton.jit(do_not_specialize_on_alignment=["num_cache_lines"])
def _ring_conv_kernel(
    x_ptr,  # (num_tokens, dim)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,  # (batch + 1)
    q_ptr,  # ring buffers [*, TP, HK] / [*, TP, HK] / [*, TP, HVV]
    k_ptr,
    v_ptr,
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.int64,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    null_block_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HK: tl.constexpr,
    HVV: tl.constexpr,
    TP: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """VLLM_GDN_MTP_RING_CONV_OUT: causal_conv1d's _causal_conv1d_update_kernel
    (varlen spec decoding, SiLU, null-block skip; the same loads, arithmetic
    and rounding) storing request r's token t into ring row r * TP + t of
    q/k/v instead of x, plus _ring_gather_kernel's other q/k/v writes: zero
    rows t >= n, and for a null-block request (which the conv skips) its raw
    x rows. It reads no b/a (ring_ab copies those). Grid (batch, dim //
    BLOCK_N)."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    c0 = tl.program_id(1) * BLOCK_N
    if c0 < HK:
        o_base = q_ptr
    elif c0 < 2 * HK:
        o_base = k_ptr
    else:
        o_base = v_ptr
    o_col = idx_feats - tl.where(c0 < HK, 0, tl.where(c0 < 2 * HK, HK, 2 * HK))
    o_w = tl.where(c0 < 2 * HK, HK, HVV)
    o_row0 = idx_seq.to(tl.int64) * TP
    rows_tp = tl.arange(0, TP)
    bos = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    n_tok = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64) - bos
    # _ring_gather_kernel: rows t >= n are zero
    pad = (rows_tp >= n_tok)[:, None] & (idx_feats < dim)[None, :]
    tl.store(
        o_base + (o_row0 + rows_tp)[:, None] * o_w + o_col[None, :],
        tl.zeros((TP, BLOCK_N), dtype=o_base.dtype.element_ty),
        mask=pad,
    )

    conv_states_input_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices).to(
        tl.int64
    )
    if conv_states_input_coord == null_block_id:
        # the conv skips this request; the gather would copy its raw rows
        raw = (rows_tp < n_tok)[:, None] & (idx_feats < dim)[None, :]
        xr = tl.load(
            x_ptr + (bos + rows_tp)[:, None] * stride_x_token + (idx_feats * stride_x_dim)[None, :],
            mask=raw,
        )
        tl.store(o_base + (o_row0 + rows_tp)[:, None] * o_w + o_col[None, :], xr, mask=raw)
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        return

    query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
    state_len = state_len - (seqlen - (query_end_index - query_start_index))
    seqlen = query_end_index - query_start_index
    x_offset = query_start_index * stride_x_token

    if query_start_index == query_end_index:
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        return

    conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1

    conv_states_base = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        conv_states_ptrs = prior_tokens + 4 * stride_conv_state_tok
        col4 = tl.load(conv_states_ptrs, mask_w, 0.0)

    idx_tokens = tl.arange(0, NP2_STATELEN)
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + 1) * stride_conv_state_tok)[:, None]
    )
    mask = (
        (conv_states_input_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)

    x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_states_offset = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices).to(
        tl.int64
    )
    conv_state_ptrs_target = (
        conv_state_ptr
        + (conv_states_offset * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )[None, :] + (idx_tokens * stride_conv_state_tok)[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    w_base = w_ptr + (idx_feats * stride_w_dim)
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_ptrs = w_base + (4 * stride_w_width)
        w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_ptrs = w_base + (5 * stride_w_width)
        w_col5 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base
    mask_x_1d = idx_feats < dim

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    for idx_token in tl.range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim)
        o_ptrs = o_base + (o_row0 + idx_token) * o_w + o_col

        tl.store(o_ptrs, acc, mask=mask_1d)


_ring_conv_launch: list = []


def ring_conv(layer, x, conv_state, weight, bias, activation, conv_state_indices,
              num_accepted_tokens, query_start_loc, max_query_len: int,
              num_requests: int) -> bool:
    """VLLM_GDN_MTP_RING_CONV_OUT: ``causal_conv1d_update(x, conv_state,
    weight, bias, activation, conv_state_indices=..., num_accepted_tokens=...,
    query_start_loc=..., max_query_len=...)`` for the spec rows of ``layer``,
    its output written into the ring's padded q/k/v rows, which is what
    ring_core's gather would copy (its a/b rows: ring_ab). Returns False
    (nothing launched) when the form does not apply; then the caller runs the
    conv as before."""
    st = ring_state()
    r = getattr(layer, "_ring", None)
    if st is None or r is None or st.bufs is None:
        return False
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    hk, hvv = r["H"] * 128, r["HV"] * 128
    batch = conv_state_indices.size(0)
    _, width = weight.shape
    if (
        activation not in ("silu", "swish")
        or x.dim() != 2
        or x.dtype != torch.bfloat16
        or conv_state.dtype != x.dtype
        or x.size(1) != 2 * hk + hvv
        or hk % 256 or hvv % 256
        or batch != num_requests
        or max_query_len > st.t_pad
        or not 2 <= width <= 6
    ):
        return False
    if not _ring_conv_launch:
        from vllm.model_executor.layers.mamba.ops.gdn_host_trim import (
            arch_support_pdl,
            launcher,
        )

        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

        _ring_conv_launch.extend(
            (launcher(_ring_conv_kernel), arch_support_pdl(), NULL_BLOCK_ID)
        )
        logger.info(
            "GDN MTP ring conv out (VLLM_GDN_MTP_RING_CONV_OUT=1, H110 2a): the "
            "spec conv writes the padded ring rows, the a/b rows are copied "
            "behind the b/a GEMM; _ring_gather_kernel skipped."
        )
    launch, pdl, null_block_id = _ring_conv_launch
    num_cache_lines = conv_state.size(0)
    state_len = width - 1 + (max_query_len - 1)
    q, k, v = st.bufs[:3]
    stride_x_token, stride_x_dim = x.stride()
    stride_w_dim, stride_w_width = weight.stride()
    s_seq, s_dim, s_tok = conv_state.stride()
    if batch > 0:
        launch[(batch, x.size(1) // 256)](
            x, weight, bias, conv_state, conv_state_indices, num_accepted_tokens,
            query_start_loc, q, k, v,
            batch, x.size(1), max_query_len, state_len, num_cache_lines,
            stride_x_dim, stride_x_token, stride_w_dim, stride_w_width,
            s_seq, s_dim, s_tok, conv_state_indices.stride(0), null_block_id,
            HAS_BIAS=bias is not None, KERNEL_WIDTH=width,
            NP2_STATELEN=triton.next_power_of_2(state_len), BLOCK_N=256,
            HK=hk, HVV=hvv, TP=st.t_pad, launch_pdl=pdl,
        )
    layer._ring_conv_rows = num_requests
    return True


@triton.jit
def _ring_xmap_kernel(cu_ptr, xmap_ptr, num_rows, TP: tl.constexpr, BLOCK: tl.constexpr):
    """xmap[cu[r] + t] = r * TP + t for the spec token rows; -1 elsewhere."""
    r = tl.program_id(0)
    bos = tl.load(cu_ptr + r)
    n = tl.load(cu_ptr + r + 1) - bos
    t = tl.arange(0, BLOCK)
    tl.store(xmap_ptr + bos + t, r * TP + t, mask=t < n)


def ring_xmap(attn_metadata, cu_seqlens, num_requests: int, num_rows: int) -> torch.Tensor:
    """Row map of the padded output for the norm (built once per metadata)."""
    c = attn_metadata.__dict__.get("_ring_xmap")
    if c is not None and c[0] == (num_requests, num_rows):
        return c[1]
    st = _STATE[0]
    xmap = torch.full((num_rows,), -1, dtype=torch.int32, device=cu_seqlens.device)
    if num_requests > 0:
        _ring_xmap_kernel[(num_requests,)](cu_seqlens, xmap, num_rows, TP=st.t_pad,
                                           BLOCK=st.t_pad, num_warps=1)
    attn_metadata.__dict__["_ring_xmap"] = ((num_requests, num_rows), xmap)
    return xmap


@triton.jit
def _ring_state_kernel(
    work_ptr,  # POST/PRE: [n, 4] (row, src_col, dst_col, bias); IMPORT/EXPORT: [n, 2] (row, slot)
    count_ptr,  # POST/PRE: [1] entries in work
    idx_map_ptr,  # POST/PRE: batch row -> slot
    bt_ptrs_ptr,  # [groups] int64 addresses of the block tables
    bt_stride,
    ssm_base_ptr,  # [layers] int64 address of kv_cache[1]
    ssm_stride_ptr,  # [layers] elements per block
    grp_ptr,  # [layers] int32 group index
    s0_base_ptr,  # [layers] int64 addresses of the ring pools
    kc_base_ptr,
    uc_base_ptr,
    gc_base_ptr,
    flag_ptr,  # POST: ran[slot]; PRE: owner[slot]
    hl_ptr,  # POST: hl0; EXPORT: hl
    cb_ptr,  # POST: cb0; EXPORT: cb
    state_idx_ptr,  # IMPORT/EXPORT: running block column per slot
    acc_ptr,  # IMPORT: accepted tokens per slot
    MODE: tl.constexpr,
    FMIN: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    RS: tl.constexpr,
    VT: tl.constexpr,
):
    """One (item, GDN layer, head, V-tile) of the ring's state traffic:
    POST: align postprocess copy (boundary state) -> fold of the ring for slots
          that ran it this step, else the block copy it replaces;
    PRE:  align pre-copy -> skipped for ring-owned slots, else the block copy;
    IMPORT: vLLM's source block (column state_idx + acc - 1) -> the slot's S0;
    EXPORT: fold of the slot's ring -> its running block (column state_idx).
    The fold is S0 (+) the window rows [start, start + n) mod RS:
    exp(G) S0 + sum_j exp(G - g_j) u_j k_j^T, G = g of the window's last row,
    in fp32, one rounding to bf16."""
    i = tl.program_id(0)
    layer = tl.program_id(1)
    pid2 = tl.program_id(2)
    if MODE == 0 or MODE == 1:
        if i >= tl.load(count_ptr):
            return
        row = tl.load(work_ptr + i * 4)
        src_col = tl.load(work_ptr + i * 4 + 1)
        dst_col = tl.load(work_ptr + i * 4 + 2)
        bias = tl.load(work_ptr + i * 4 + 3)
        slot = tl.load(idx_map_ptr + row).to(tl.int32)
    else:
        row = tl.load(work_ptr + i * 2)
        slot = tl.load(work_ptr + i * 2 + 1)
        src_col = 0
        dst_col = 0
        bias = 0
    hv = pid2 // (V // VT)
    vt = pid2 % (V // VT)
    grp = tl.load(grp_ptr + layer)
    bt = tl.load(bt_ptrs_ptr + grp).to(tl.pointer_type(tl.int32)) + row.to(tl.int64) * bt_stride
    ssm = tl.load(ssm_base_ptr + layer).to(tl.pointer_type(tl.bfloat16))
    sstr = tl.load(ssm_stride_ptr + layer)
    rows = vt * VT + tl.arange(0, VT)
    cols = tl.arange(0, K)
    off = hv * V * K + rows[:, None] * K + cols[None, :]
    s0 = (
        tl.load(s0_base_ptr + layer).to(tl.pointer_type(tl.bfloat16))
        + slot.to(tl.int64) * (HV * V * K)
    )
    use_ring = 0
    start = 0
    n = 0
    src_blk = 0
    dst_blk = 0
    if MODE == 0:
        dst_blk = tl.load(bt + dst_col)
        if tl.load(flag_ptr + slot) != 0:
            use_ring = 1
            p = tl.load(hl_ptr + slot)
            c = tl.load(cb_ptr + slot)
            if p >= FMIN:  # flushed this step: S0 holds the folded window
                start = (c + p) & (RS - 1)
                n = bias + 1
            else:
                start = c
                n = p + bias + 1
        else:
            src_blk = tl.load(bt + src_col + bias)
    elif MODE == 1:
        if tl.load(flag_ptr + slot) != 0:
            return
        src_blk = tl.load(bt + src_col + bias)
        dst_blk = tl.load(bt + dst_col)
    elif MODE == 2:
        src_blk = tl.load(bt + tl.load(state_idx_ptr + slot) + tl.load(acc_ptr + slot) - 1)
    else:
        use_ring = 1
        start = tl.load(cb_ptr + slot)
        n = tl.load(hl_ptr + slot)
        dst_blk = tl.load(bt + tl.load(state_idx_ptr + slot))
    if use_ring != 0:
        x = tl.load(s0 + off).to(tl.float32)
        if n > 0:
            gcp = (
                tl.load(gc_base_ptr + layer).to(tl.pointer_type(tl.float32))
                + (slot.to(tl.int64) * HV + hv) * RS
            )
            kcp = (
                tl.load(kc_base_ptr + layer).to(tl.pointer_type(tl.bfloat16))
                + (slot.to(tl.int64) * H + hv // (HV // H)) * (RS * K)
            )
            ucp = (
                tl.load(uc_base_ptr + layer).to(tl.pointer_type(tl.bfloat16))
                + (slot.to(tl.int64) * HV + hv) * (RS * V)
            )
            g_last = tl.load(gcp + ((start + n - 1) & (RS - 1)))
            x = x * tl.exp(g_last)
            for j in range(0, n):
                rr = (start + j) & (RS - 1)
                w = tl.exp(g_last - tl.load(gcp + rr))
                u = tl.load(ucp + rr * V + rows).to(tl.float32)
                kk = tl.load(kcp + rr * K + cols).to(tl.float32)
                x += (w * u)[:, None] * kk[None, :]
    else:
        x = tl.load(ssm + src_blk.to(tl.int64) * sstr + off).to(tl.float32)
    if MODE == 2:
        tl.store(s0 + off, x.to(tl.bfloat16))
    else:
        tl.store(ssm + dst_blk.to(tl.int64) * sstr + off, x.to(tl.bfloat16))


@triton.jit
def _ring_copy_slots_kernel(
    work_ptr,  # [n, 4] (row, src_col, dst_col, bias)
    count_ptr,  # [1] entries in work
    idx_map_ptr,  # batch row -> slot
    bt_ptrs_ptr,
    bt_stride,
    ssm_base_ptr,
    ssm_stride_ptr,
    grp_ptr,
    s0_base_ptr,
    kc_base_ptr,
    uc_base_ptr,
    gc_base_ptr,
    flag_ptr,  # POST: ran[slot]; PRE: owner[slot]
    hl_ptr,  # POST: hl0
    cb_ptr,  # POST: cb0
    MODE: tl.constexpr,  # 0 POST, 1 PRE
    FMIN: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    RS: tl.constexpr,
    VT: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """VLLM_GDN_MTP_RING_LEAN: the POST/PRE modes of _ring_state_kernel on a
    fixed grid (SLOTS, layers, HV * V // VT). Program (s, layer, tile) handles
    work entries s, s + SLOTS, ... < count with _ring_state_kernel's per-entry
    loads, fold order and rounding (the same bits), so an empty list costs
    SLOTS instead of num_reqs programs per (layer, tile)."""
    layer = tl.program_id(1)
    pid2 = tl.program_id(2)
    hv = pid2 // (V // VT)
    vt = pid2 % (V // VT)
    grp = tl.load(grp_ptr + layer)
    bt0 = tl.load(bt_ptrs_ptr + grp).to(tl.pointer_type(tl.int32))
    ssm = tl.load(ssm_base_ptr + layer).to(tl.pointer_type(tl.bfloat16))
    sstr = tl.load(ssm_stride_ptr + layer)
    rows = vt * VT + tl.arange(0, VT)
    cols = tl.arange(0, K)
    off = hv * V * K + rows[:, None] * K + cols[None, :]
    count = tl.load(count_ptr)
    for i in range(tl.program_id(0), count, SLOTS):
        row = tl.load(work_ptr + i * 4)
        src_col = tl.load(work_ptr + i * 4 + 1)
        dst_col = tl.load(work_ptr + i * 4 + 2)
        bias = tl.load(work_ptr + i * 4 + 3)
        slot = tl.load(idx_map_ptr + row).to(tl.int32)
        bt = bt0 + row.to(tl.int64) * bt_stride
        flag = tl.load(flag_ptr + slot)
        if MODE == 0:
            dst_blk = tl.load(bt + dst_col)
            if flag != 0:
                s0 = (
                    tl.load(s0_base_ptr + layer).to(tl.pointer_type(tl.bfloat16))
                    + slot.to(tl.int64) * (HV * V * K)
                )
                p = tl.load(hl_ptr + slot)
                c = tl.load(cb_ptr + slot)
                # flushed this step: S0 holds the folded window
                start = tl.where(p >= FMIN, (c + p) & (RS - 1), c)
                n = tl.where(p >= FMIN, bias + 1, p + bias + 1)
                x = tl.load(s0 + off).to(tl.float32)
                if n > 0:
                    gcp = (
                        tl.load(gc_base_ptr + layer).to(tl.pointer_type(tl.float32))
                        + (slot.to(tl.int64) * HV + hv) * RS
                    )
                    kcp = (
                        tl.load(kc_base_ptr + layer).to(tl.pointer_type(tl.bfloat16))
                        + (slot.to(tl.int64) * H + hv // (HV // H)) * (RS * K)
                    )
                    ucp = (
                        tl.load(uc_base_ptr + layer).to(tl.pointer_type(tl.bfloat16))
                        + (slot.to(tl.int64) * HV + hv) * (RS * V)
                    )
                    g_last = tl.load(gcp + ((start + n - 1) & (RS - 1)))
                    x = x * tl.exp(g_last)
                    for j in range(0, n):
                        rr = (start + j) & (RS - 1)
                        w = tl.exp(g_last - tl.load(gcp + rr))
                        u = tl.load(ucp + rr * V + rows).to(tl.float32)
                        kk = tl.load(kcp + rr * K + cols).to(tl.float32)
                        x += (w * u)[:, None] * kk[None, :]
                tl.store(ssm + dst_blk.to(tl.int64) * sstr + off, x.to(tl.bfloat16))
            else:
                src_blk = tl.load(bt + src_col + bias)
                x = tl.load(ssm + src_blk.to(tl.int64) * sstr + off).to(tl.float32)
                tl.store(ssm + dst_blk.to(tl.int64) * sstr + off, x.to(tl.bfloat16))
        else:
            if flag == 0:
                src_blk = tl.load(bt + src_col + bias)
                dst_blk = tl.load(bt + dst_col)
                x = tl.load(ssm + src_blk.to(tl.int64) * sstr + off).to(tl.float32)
                tl.store(ssm + dst_blk.to(tl.int64) * sstr + off, x.to(tl.bfloat16))


@triton.jit
def _ring_prep_kernel(spec_ptr, imp_ptr, n_spec, slots_step_ptr, hl_step_ptr,
                      cb_step_ptr, hl_ptr, cb_ptr, hl0_ptr, cb0_ptr, ran_ptr,
                      owner_ptr, num_slots, BLOCK: tl.constexpr):
    """Per spec row i < n_spec (slot s): this step's cursors (0 for an import)
    -> hl_step/cb_step[i] and hl0/cb0[s]; ran[s] = owner[s] = 1. Rows past
    n_spec (FULL-graph padding) get slot -1, which the kernel skips."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < num_slots
    spec = m & (i < n_spec)
    s = tl.load(spec_ptr + i, mask=spec, other=-1)
    ok = spec & (s >= 0)
    imp = tl.load(imp_ptr + i, mask=ok, other=0)
    h = tl.where(imp != 0, 0, tl.load(hl_ptr + s, mask=ok, other=0))
    c = tl.where(imp != 0, 0, tl.load(cb_ptr + s, mask=ok, other=0))
    tl.store(slots_step_ptr + i, tl.where(ok, s, -1), mask=m)
    tl.store(hl_step_ptr + i, h, mask=m)
    tl.store(cb_step_ptr + i, c, mask=m)
    tl.store(hl0_ptr + s, h, mask=ok)
    tl.store(cb0_ptr + s, c, mask=ok)
    tl.store(ran_ptr + s, tl.full((BLOCK,), 1, tl.int32), mask=ok)
    tl.store(owner_ptr + s, tl.full((BLOCK,), 1, tl.int32), mask=ok)


@triton.jit
def _ring_commit_kernel(ran_ptr, acc_ptr, hl0_ptr, cb0_ptr, hl_ptr, cb_ptr,
                        num_slots, FMIN: tl.constexpr, RS: tl.constexpr,
                        BLOCK: tl.constexpr):
    """Cursor commit of the slots that ran the ring (Triton
    commit_gdn_replayssm_spec): flushed rows base' = (base + P) & (RS - 1),
    len' = acc; others len' = P + acc. acc = accepted tokens of the step."""
    s = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = (s < num_slots) & (tl.load(ran_ptr + s, mask=s < num_slots, other=0) != 0)
    p = tl.load(hl0_ptr + s, mask=m, other=0)
    c = tl.load(cb0_ptr + s, mask=m, other=0)
    a = tl.load(acc_ptr + s, mask=m, other=0)
    fl = p >= FMIN
    tl.store(cb_ptr + s, tl.where(fl, (c + p) & (RS - 1), c), mask=m)
    tl.store(hl_ptr + s, tl.where(fl, a, p + a), mask=m)


def _h2d(x: np.ndarray, device) -> torch.Tensor:
    # pinned staging: the caching host allocator keeps it until the copy ran
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.int32)).pin_memory().to(
        device, non_blocking=True
    )


def _bind(st: RingState, entries, bt_ptrs: torch.Tensor, bt_stride: int) -> None:
    """Per-layer addresses for _ring_state_kernel; entries: (layer, group
    index) of every registered GDN layer, from the align copy context."""
    dev = st.device
    layers = [layer for layer, _ in entries]
    assert {id(x) for x in layers} == {id(x) for x in st.layers}, (
        "VLLM_GDN_MTP_RING: every ring layer must be an align-mode mamba state"
    )
    for layer in layers:
        ssm = layer.kv_cache[1]
        assert ssm.dim() == 4 and ssm.stride(3) == 1 and ssm.stride(2) == ssm.size(3)
        assert ssm.stride(1) == ssm.size(2) * ssm.size(3) and ssm.dtype == torch.bfloat16

    def i64(vals):
        return torch.tensor(vals, dtype=torch.int64, device=dev)

    st.k_layers = len(layers)
    st.k_ssm = i64([x.kv_cache[1].data_ptr() for x in layers])
    st.k_ssm_stride = i64([x.kv_cache[1].stride(0) for x in layers])
    st.k_grp = torch.tensor([g for _, g in entries], dtype=torch.int32, device=dev)
    st.k_s0 = i64([x._ring["s0"].data_ptr() for x in layers])
    st.k_kc = i64([x._ring["kc"].data_ptr() for x in layers])
    st.k_uc = i64([x._ring["uc"].data_ptr() for x in layers])
    st.k_gc = i64([x._ring["gc"].data_ptr() for x in layers])
    st.bt_ptrs = bt_ptrs
    st.bt_stride = int(bt_stride)
    st.bound = True


def _state_launch(st: RingState, mode: int, n_items: int, work, count, idx_map,
                  flag, hl, cb, state_idx, acc) -> None:
    if n_items <= 0:
        return
    r = st.layers[0]._ring
    vt = 32
    dummy = st.ran
    _ring_state_kernel[(n_items, st.k_layers, r["HV"] * (128 // vt))](
        work, dummy if count is None else count, dummy if idx_map is None else idx_map,
        st.bt_ptrs, st.bt_stride, st.k_ssm, st.k_ssm_stride, st.k_grp,
        st.k_s0, st.k_kc, st.k_uc, st.k_gc,
        dummy if flag is None else flag, dummy if hl is None else hl,
        dummy if cb is None else cb, dummy if state_idx is None else state_idx,
        dummy if acc is None else acc,
        MODE=mode, FMIN=st.flush_min, H=r["H"], HV=r["HV"], K=128, V=128,
        RS=st.ring_slots, VT=vt, num_warps=4,
    )


# ------------------------------------------------------------------ hooks
def ring_bind_copy_context(entries, bt_ptrs: torch.Tensor, bt_stride: int) -> None:
    """From MambaSpecDecodeGPUContext._populate_metadata (once)."""
    st = ring_state()
    if st is not None and not getattr(st, "bound", False):
        _bind(st, entries, bt_ptrs, bt_stride)


def _copy_slots_launch(st: RingState, mode: int, work, count, idx_map, flag,
                       hl, cb, slots: int = _RING_COPY_SLOTS) -> None:
    r = st.layers[0]._ring
    vt = 32
    dummy = st.ran
    _ring_copy_slots_kernel[(slots, st.k_layers, r["HV"] * (128 // vt))](
        work, count, idx_map, st.bt_ptrs, st.bt_stride, st.k_ssm, st.k_ssm_stride,
        st.k_grp, st.k_s0, st.k_kc, st.k_uc, st.k_gc, flag,
        dummy if hl is None else hl, dummy if cb is None else cb,
        MODE=mode, FMIN=st.flush_min, H=r["H"], HV=r["HV"], K=128, V=128,
        RS=st.ring_slots, VT=vt, SLOTS=slots, num_warps=4,
    )


def ring_copy_work(work, count, num_reqs: int, idx_mapping, is_precopy: bool) -> None:
    """The GDN SSM-state part of an align compact copy (the compact copy
    kernel skips those states under VLLM_GDN_MTP_RING)."""
    st = ring_state()
    assert st is not None and st.bound and idx_mapping is not None
    if GDN_MTP_RING_LEAN:
        if not getattr(st, "lean_logged", False):
            st.lean_logged = True
            tiles = st.layers[0]._ring["HV"] * 4
            logger.info(
                "GDN MTP ring lean align copies (VLLM_GDN_MTP_RING_LEAN=1, H102): "
                "PRE/POST on a fixed grid of %d work slots x %d layers x %d "
                "tiles = %d programs per launch (num_reqs-sized grid: %d at "
                "this step's %d rows).", _RING_COPY_SLOTS, st.k_layers, tiles,
                _RING_COPY_SLOTS * st.k_layers * tiles,
                num_reqs * st.k_layers * tiles, num_reqs,
            )
        if is_precopy:
            _copy_slots_launch(st, _MODE_PRE, work, count, idx_mapping, st.owner,
                               None, None)
        else:
            _copy_slots_launch(st, _MODE_POST, work, count, idx_mapping, st.ran,
                               st.hl0, st.cb0)
        return
    if is_precopy:
        _state_launch(st, _MODE_PRE, num_reqs, work, count, idx_mapping, st.owner,
                      None, None, None, None)
    else:
        _state_launch(st, _MODE_POST, num_reqs, work, count, idx_mapping, st.ran,
                      st.hl0, st.cb0, None, None)


def ring_on_add_request(req_index: int) -> None:
    st = ring_state()
    if st is not None:
        st.owner_np[req_index] = False
        st.owner[req_index].fill_(0)


def ring_step_prepare(idx_mapping_np: np.ndarray, spec_rows: np.ndarray,
                      other_rows: np.ndarray, state_idx: torch.Tensor,
                      acc: torch.Tensor) -> None:
    """Before the forward (after the align pre-copy): export ring-owned rows
    that take a non-spec path this step, import vLLM-owned spec rows, and set
    the step's per-row slots and cursors. ``spec_rows``: batch rows of the
    spec decodes in the GDN metadata's row order; ``other_rows``: the other
    real rows; ``state_idx`` / ``acc``: per-slot running column and accepted
    tokens (after the pre-copy's reset)."""
    st = ring_state()
    if st is None:
        return
    dev = st.device
    st.ran.zero_()
    if len(other_rows):
        oslots = idx_mapping_np[other_rows]
        exp = st.owner_np[oslots]
        if exp.any():
            assert st.bound, "VLLM_GDN_MTP_RING: export before the copy context"
            items = np.stack([other_rows[exp], oslots[exp]], 1).reshape(-1)
            _state_launch(st, _MODE_EXPORT, int(exp.sum()), _h2d(items, dev), None,
                          None, None, st.hl, st.cb, state_idx, None)
            ex = _h2d(oslots[exp], dev).long()
            st.owner.index_fill_(0, ex, 0)
            st.owner_np[oslots[exp]] = False
    n_spec = len(spec_rows)
    if n_spec:
        slots = idx_mapping_np[spec_rows].astype(np.int32)
        imp = ~st.owner_np[slots]
        if imp.any():
            assert st.bound, "VLLM_GDN_MTP_RING: import before the copy context"
            items = np.stack([spec_rows[imp], slots[imp]], 1).reshape(-1)
            _state_launch(st, _MODE_IMPORT, int(imp.sum()), _h2d(items, dev), None,
                          None, None, None, None, state_idx, acc)
        st.owner_np[slots] = True
        spec = _h2d(slots, dev)
        impd = _h2d(imp.astype(np.int32), dev)
    else:
        spec = impd = st.spec_dev
    block = 256
    _ring_prep_kernel[(triton.cdiv(st.num_slots, block),)](
        spec, impd, n_spec, st.slots_step, st.hl_step, st.cb_step, st.hl, st.cb,
        st.hl0, st.cb0, st.ran, st.owner, st.num_slots, BLOCK=block, num_warps=4,
    )
    st.prepared = n_spec > 0


def ring_step_commit(acc: torch.Tensor) -> None:
    """After sampling, before the align postprocess (which resets acc to 1 for
    self-copy rows): commit the cursors of the slots that ran the ring."""
    st = ring_state()
    if st is None or not st.prepared:
        return
    block = 256
    _ring_commit_kernel[(triton.cdiv(st.num_slots, block),)](
        st.ran, acc, st.hl0, st.cb0, st.hl, st.cb, st.num_slots,
        FMIN=st.flush_min, RS=st.ring_slots, BLOCK=block, num_warps=4,
    )
    st.prepared = False


def ring_step_dummy() -> None:
    """Dummy / capture steps: no slot runs the ring (all rows -1), so a dummy
    forward (or a FULL replay of one) never touches a real slot's state."""
    st = ring_state()
    if st is None:
        return
    st.slots_step.fill_(-1)
    st.prepared = False


def ring_spec_rows(num_decode_draft_tokens_np: np.ndarray, num_reqs: int):
    """(spec rows, other real rows) in the GDN metadata's order: rows with a
    draft count >= 0 are spec decodes (ascending), unless no spec row has a
    draft token (then none is, as GDNAttentionMetadataBuilder._split_batch)."""
    d = num_decode_draft_tokens_np[:num_reqs]
    mask = d >= 0
    if not mask.any() or int(d[mask].sum()) == 0:
        mask = np.zeros(num_reqs, dtype=np.bool_)
    rows = np.arange(num_reqs)
    return rows[mask], rows[~mask]
