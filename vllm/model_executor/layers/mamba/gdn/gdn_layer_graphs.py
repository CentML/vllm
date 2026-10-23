# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501 - the Triton pack kernel is kept compact
"""CUDA-graph replay of the eager GDN core of mixed and prefill-only steps.

With ``VLLM_GDN_LAYER_GRAPHS=1`` the GDN core op of a PIECEWISE step (prefill
rows, optionally preceded by a block of MTP spec-decode rows) replays one CUDA
graph per GDN layer instead of issuing its ~6-9 eager launches. The op stays a
splitting op, so the compiled pieces and every other kernel are unchanged.
Idea from CentML/vllm#138 (gdn_layer_graphs.py, MODE=layer); this is a port
onto the snapshot-MTP stack (no deferred state commit, no step plan).

Exactness. A graph replays the eager path's kernels with the eager path's
compile-time choices; only the per-step data moves to the device:

* per KV-cache group and step, one pack launch copies the step's spec slots,
  accepted counts and cu_seqlens (padded to a request bucket with zero-length
  requests on the null slot), the prefill cu_seqlens rebased to absolute token
  rows, the prefill state slots / initial-state flags, and
  ``[num_valid, norm_lo, norm_hi]`` into fixed-address buffers of the group;
* every launch-shaping host int of the eager path is part of the graph key
  (the "variant"): padded token count ``T`` (the PIECEWISE size), number of
  prefill sequences (exact: the CUDA conv, the zero-fresh kernel and the
  V-split kernel see the same sequence count, grid and tiles as eager),
  V-split factor, CUDA-conv tokens per half-warp, the spec kernel family and
  its Triton launch config (request bucket), and the norm kernel's warps;
* row offsets that differ per step are read on the device: the CUDA conv and
  the V-split kernel take absolute cu_seqlens over the full ``mixed_qkv`` /
  ``core_attn_out`` rows instead of eager's row-sliced views, the spec kernels
  read their cu_seqlens (spec rows lead at row 0), and the gated-norm MXFP8
  kernel reads ``[num_valid, norm_lo, norm_hi]`` (``ROWS_FROM_PTR``);
* padding only adds zero-length spec requests on ``NULL_BLOCK_ID`` (the conv
  update, the Triton and the CUDA MTP kernels return before any access) and
  surplus CTAs of the CUDA conv grid (unmapped, they return).

Eligible steps (else the eager path runs, unchanged): graphs armed (after
vLLM's own capture phase), PIECEWISE runtime mode, out_proj takes the MXFP8
activation, ``T <= T_MAX``, prefill rows without non-spec decodes and without
an in-forward prefill checkpoint, at most ``NP`` prefill sequences, the spec
block (if any) leading at row 0 and taking the MTP-kernel half of the mixed
path, at most ``NR`` spec requests (CUDA MTP kernel above
``VLLM_GDN_MTP_TRITON_MAX_REQS``), the CUDA conv kernel and the non-CP V-split
kernel (``v_split > 1``) being what eager would launch. ``VLLM_GDN_MIXED_FORK``
steps replay serially (the fork only reorders the same kernels).

Graphs are captured lazily, the first time a (layer, variant, buffer
addresses, state pool) key is seen while serving, after one eager launch of
the same sequence on "null" buffer contents (zero-length everything: compiles
and loads every kernel outside the capture, touches no live state). All layer
graphs share one private memory pool (intermediates of at most ``T_MAX`` rows).

Knobs (``VLLM_GDN_LAYER_GRAPHS_*``): ``T_MAX`` (4096) padded token bound,
``NP`` (4) prefill sequences, ``NR`` (64) spec requests, ``MAX_GRAPHS``
(6000) capture cap, ``CHECK`` (0) the first N replays are compared bitwise
with the eager path on the same inputs and state (state restored in between;
a mismatch logs, keeps the eager result and disables the graphs),
``LOG_EVERY`` (2000) stats period in planned steps.
"""

import dataclasses
import os
import time

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.gdn_host_trim import GDN_HOST_TRIM
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)

_P = "VLLM_GDN_LAYER_GRAPHS"
ENABLED = os.environ.get(_P, "0") == "1"
T_MAX = int(os.environ.get(_P + "_T_MAX", "4096"))
NP = int(os.environ.get(_P + "_NP", "4"))
NR = int(os.environ.get(_P + "_NR", "64"))
MAX_GRAPHS = int(os.environ.get(_P + "_MAX_GRAPHS", "6000"))
LOG_EVERY = int(os.environ.get(_P + "_LOG_EVERY", "2000"))
_CHECK = [int(os.environ.get(_P + "_CHECK", "0"))]

_ARMED = [False]  # set after vLLM's capture phase (real KV cache, real buffers)
_OFF = [False]  # disabled after a CHECK mismatch
_BYPASS = [False]  # CHECK: the eager re-run of a replayed layer
_KEY = "_gdn_layer_graph_step"  # md.__dict__: _Step or ineligibility reason
# md.__dict__: set once a layer graph of the group ran this step (so a group's
# eager layers do not re-zero fresh rows a replay already wrote:
# VLLM_GDN_HOST_TRIM3 ZERO).
REPLAYED_KEY = "_gdn_layer_graph_replayed"
_BUFS: dict = {}  # layer names of a KV-cache group -> _GroupBufs
_GRAPHS: dict = {}  # graph key -> torch.cuda.CUDAGraph
_FAILED: set = set()  # graph keys whose capture failed (stay eager)
_WS: dict = {}  # (device, HQ, HV) -> V-split workspace
_CAP: list = [None, None]  # capture stream, private graph pool
STATS: dict = {}

# Spec-kernel families of the mixed path's MTP half.
SPEC_NONE, SPEC_TRITON, SPEC_CUDA = 0, 1, 2


def _gdn():
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    return qwen_gdn_linear_attn


def _stat(name: str, n: int = 1) -> None:
    STATS[name] = STATS.get(name, 0) + n


def arm() -> None:
    """Called by the model runner after its CUDA-graph capture phase: from
    now on the KV cache and the piecewise graphs' buffers are final.
    """
    if ENABLED and not _ARMED[0]:
        _ARMED[0] = True
        logger.info(
            "GDN layer graphs armed: T_MAX=%d NP=%d NR=%d MAX_GRAPHS=%d CHECK=%d",
            T_MAX,
            NP,
            NR,
            MAX_GRAPHS,
            _CHECK[0],
        )


def drop_graphs() -> None:
    """Drop every layer graph, its private pool and the group buffers (for a
    KV-cache re-initialization; the next eligible steps capture anew).
    """
    torch.accelerator.synchronize()
    _GRAPHS.clear()
    _FAILED.clear()
    _BUFS.clear()
    _WS.clear()
    _CAP[1] = None


# Inputs are the step metadata's tensors (slices at any offset): no alignment
# specialization, so the warmup's launches compile the serving variants.
# fmt: off
@triton.jit(do_not_specialize=["S", "N", "nr", "npf", "lo", "hi", "si_s0", "ns_s0", "ps_s0"],
            do_not_specialize_on_alignment=["spec_si", "spec_cu", "spec_acc", "p_cu", "ns_si", "ns_hi", "p_si", "p_hi"])
def _pack_kernel(spec_si, si_s0, spec_cu, spec_acc, p_cu, ns_si, ns_s0, ns_hi, p_si, ps_s0, p_hi,
                 si_out, acc_out, cu_s_out, cu_p_out, ci_conv, hi_conv, ci_chunk, hi_chunk, rows,
                 S, N, nr, npf, lo, hi,
                 W: tl.constexpr, NRC: tl.constexpr, NPC: tl.constexpr, BLOCK: tl.constexpr,
                 HAS_SPEC: tl.constexpr):
    """One step of one KV-cache group into the group's fixed buffers.
    Spec: slots [NRC, W] (pads NULL_BLOCK_ID = 0), accepted [NRC] (pads 1),
    cu [NRC + 1] (pads = S: zero-length). Prefill: cu rebased to absolute rows
    (+ S) [NPC + 1] (pads = N), conv / chunk slots and initial-state flags
    [NPC] (pads slot 0, flag 1). rows = [N, lo, hi].
    """
    offs = tl.arange(0, BLOCK)
    if HAS_SPEC:
        mr = offs < nr
        for w in tl.static_range(W):
            s = tl.load(spec_si + offs * si_s0 + w, mask=mr, other=0)
            tl.store(si_out + offs * W + w, s, mask=offs < NRC)
        a = tl.load(spec_acc + offs, mask=mr, other=1)
        tl.store(acc_out + offs, a, mask=offs < NRC)
        c = tl.load(spec_cu + offs, mask=offs <= nr, other=0)
        tl.store(cu_s_out + offs, tl.where(offs <= nr, c, S), mask=offs <= NRC)
    c = tl.load(p_cu + offs, mask=offs <= npf, other=0) + S
    tl.store(cu_p_out + offs, tl.where(offs <= npf, c, N), mask=offs <= NPC)
    mp = offs < npf
    s = tl.load(ns_si + offs * ns_s0, mask=mp, other=0)
    tl.store(ci_conv + offs, s, mask=offs < NPC)
    h = tl.load(ns_hi + offs, mask=mp, other=1)
    tl.store(hi_conv + offs, h, mask=offs < NPC)
    s = tl.load(p_si + offs * ps_s0, mask=mp, other=0)
    tl.store(ci_chunk + offs, s, mask=offs < NPC)
    h = tl.load(p_hi + offs, mask=mp, other=1)
    tl.store(hi_chunk + offs, h, mask=offs < NPC)
    tl.store(rows + offs, tl.where(offs == 0, N, tl.where(offs == 1, lo, hi)), mask=offs < 3)
# fmt: on


class _GroupBufs:
    """Fixed-address per-step buffers of one KV-cache group (its layers share
    the step's metadata). Never freed or replaced: captured graphs bake their
    addresses.
    """

    def __init__(self, dev, si_dtype, ns_dtype, ps_dtype):
        i32 = dict(dtype=torch.int32, device=dev)
        self.dev = dev
        self.dtypes = (si_dtype, ns_dtype, ps_dtype)
        self._si: dict = {}  # spec width -> [NR, width] slots
        self.acc = torch.ones(NR, **i32)
        self.cu_s = torch.zeros(NR + 1, **i32)
        self.cu_p = torch.zeros(NP + 1, **i32)
        self.ci_conv = torch.zeros(NP, dtype=ns_dtype, device=dev)
        self.hi_conv = torch.ones(NP, dtype=torch.bool, device=dev)
        self.ci_chunk = torch.zeros(NP, dtype=ps_dtype, device=dev)
        self.hi_chunk = torch.ones(NP, dtype=torch.bool, device=dev)
        self.rows = torch.zeros(3, **i32)

    def si(self, width: int) -> torch.Tensor:
        t = self._si.get(width)
        if t is None:
            t = self._si[width] = torch.full(
                (NR, width), NULL_BLOCK_ID, dtype=self.dtypes[0], device=self.dev
            )
        return t

    def tensors(self):
        return (
            *self._si.values(),
            self.acc,
            self.cu_s,
            self.cu_p,
            self.ci_conv,
            self.hi_conv,
            self.ci_chunk,
            self.hi_chunk,
            self.rows,
        )

    def fill_null(self) -> None:
        """Zero-length everything: every kernel of a layer graph is a no-op
        on live state (spec requests on the null slot, prefill sequences
        empty, num_valid 0).
        """
        for t in self._si.values():
            t.fill_(NULL_BLOCK_ID)
        self.acc.fill_(1)
        self.cu_s.zero_()
        self.cu_p.zero_()
        self.ci_conv.fill_(NULL_BLOCK_ID)
        self.hi_conv.fill_(True)
        self.ci_chunk.fill_(NULL_BLOCK_ID)
        self.hi_chunk.fill_(True)
        self.rows.zero_()


def _launch_pack(
    gb: _GroupBufs,
    si_buf: torch.Tensor,
    si: torch.Tensor | None,
    spec_cu: torch.Tensor | None,
    spec_acc: torch.Tensor | None,
    pcu: torch.Tensor,
    ns_si: torch.Tensor,
    ns_hi: torch.Tensor,
    ps_si: torch.Tensor,
    ps_hi: torch.Tensor,
    S: int,
    N: int,
    nr: int,
    npf: int,
    norm_rows: tuple[int, int],
) -> None:
    """One step of one KV-cache group into ``gb`` (spec inputs None: no spec
    rows).
    """
    has_spec = si is not None
    dummy = gb.cu_s
    _pack_kernel[(1,)](
        si if has_spec else dummy,
        si.stride(0) if has_spec else 0,
        spec_cu if has_spec else dummy,
        spec_acc if has_spec else dummy,
        pcu,
        ns_si,
        ns_si.stride(0),
        ns_hi,
        ps_si,
        ps_si.stride(0),
        ps_hi,
        si_buf,
        gb.acc,
        gb.cu_s,
        gb.cu_p,
        gb.ci_conv,
        gb.hi_conv,
        gb.ci_chunk,
        gb.hi_chunk,
        gb.rows,
        S,
        N,
        nr,
        npf,
        norm_rows[0],
        norm_rows[1],
        W=si_buf.size(1),
        NRC=NR,
        NPC=NP,
        BLOCK=triton.next_power_of_2(max(NR, NP) + 1),
        HAS_SPEC=has_spec,
        num_warps=4,
    )


def warm_triton_kernels(layer, num_spec: int, x_dtype: torch.dtype) -> None:
    """Compile at start-up the Triton specializations that only the layer
    graphs launch (else the null warm-up of a variant's first capture compiles
    them while serving): the pack kernel with and without spec rows, the
    spec-row ``causal_conv1d_update`` at every padded request bucket on the
    group buffers, and the ``ROWS_FROM_PTR`` gated-norm MXFP8 kernel in both
    warp classes, with and without PDL (``T`` below / at 4096). Arguments
    mirror ``_plan`` / ``graph_core``; nothing live is touched (null slots,
    zero-length sequences, ``num_valid`` 0 into scratch outputs).
    """
    M = _gdn()
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        gdn_gated_norm_mxfp8,
        gdn_mxfp8_scale_numel,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    dev = layer.A_log.device
    i32 = dict(dtype=torch.int32, device=dev)
    gb = _GroupBufs(dev, torch.int32, torch.int32, torch.int32)
    pcu = torch.zeros(NP + 1, **i32)
    slots = torch.full((NP,), NULL_BLOCK_ID, **i32)
    flags = torch.ones(NP, dtype=torch.bool, device=dev)
    W = num_spec + 1
    for has_spec in (False, True) if num_spec > 0 else (False,):
        width = W if has_spec else 1
        _launch_pack(
            gb,
            gb.si(width),
            torch.full((NR, width), NULL_BLOCK_ID, **i32) if has_spec else None,
            torch.zeros(NR + 1, **i32) if has_spec else None,
            torch.ones(NR, **i32) if has_spec else None,
            pcu,
            slots,
            flags,
            slots,
            flags,
            0,
            0,
            0,
            0,
            (0, 0),
        )
    qkv_size = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
    vdim = layer.value_dim // layer.tp_size
    HV, V = layer.A_log.shape[0], layer.head_v_dim
    if num_spec > 0:
        conv_state, conv_w = M._host_layer_views(layer)
        mixed_qkv = torch.zeros(1, qkv_size + vdim, dtype=x_dtype, device=dev).split(
            [qkv_size, vdim], dim=-1
        )[0]
        si = gb.si(W)
        for nrb in sorted({_spec_bucket(M, n)[1] for n in range(1, NR + 1)}):
            causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_w,
                layer.conv1d.bias,
                layer.activation,
                conv_state_indices=si[:nrb, 0],
                num_accepted_tokens=gb.acc[:nrb],
                query_start_loc=gb.cu_s[: nrb + 1],
                max_query_len=W,
                validate_data=False,
            )
    for T in sorted({min(T_MAX, 4095), T_MAX}):
        mixed_qkvz = torch.zeros(T, qkv_size + vdim, dtype=x_dtype, device=dev)
        gate = mixed_qkvz.split([qkv_size, vdim], dim=-1)[1].reshape(T, -1, V)
        core_attn_out = torch.zeros(T, HV, V, dtype=x_dtype, device=dev)
        out_q = torch.empty(T, HV * V, dtype=torch.float8_e4m3fn, device=dev)
        out_scale = torch.empty(
            gdn_mxfp8_scale_numel(T, HV * V), dtype=torch.uint8, device=dev
        )
        for norm_rows in ((0, 0), (0, 1), (0, T)):
            gdn_gated_norm_mxfp8(
                core_attn_out,
                gate,
                layer.norm.weight,
                layer.norm.eps,
                layer.norm.activation,
                out_q,
                out_scale,
                norm_rows,
                gb.rows[:1],
                rows=gb.rows,
            )


@dataclasses.dataclass
class _Step:
    """A packed step of one KV-cache group: variant (graph-key part shared
    by the group's layers), buffers, and the host values of the variant.
    """

    variant: tuple
    bufs: _GroupBufs
    width: int
    nr: int
    nrb: int
    npf: int
    spec: int
    tph: int
    vsf: int
    norm_rows: tuple[int, int]
    mtp_cfg: tuple[int, int] | None
    T: int
    zero: bool


def _spec_bucket(M, n: int) -> tuple[int, int, tuple[int, int] | None]:
    """(kernel family, padded request count, Triton (BLOCK_V, num_warps)) of
    ``n`` spec requests. Eager picks the Triton recurrence up to
    VLLM_GDN_MTP_TRITON_MAX_REQS requests with a config that depends on n,
    else the CUDA MTP kernel; a bucket never crosses either choice.
    """
    from vllm.model_executor.layers.mamba.ops.gdn_mtp_decode import (
        gdn_mtp_launch_config,
    )

    tmax = M.GDN_MTP_TRITON_MAX_REQUESTS
    if n <= tmax:
        cfg = gdn_mtp_launch_config(n)
        nb = n
        while nb + 1 <= tmax and gdn_mtp_launch_config(nb + 1) == cfg:
            nb += 1
        return SPEC_TRITON, nb, cfg
    nb = 1 << (n - 1).bit_length()
    return SPEC_CUDA, min(max(nb, tmax + 1), NR), None


def _group_names(fc, md) -> tuple:
    return tuple(n for n, m in fc.attn_metadata.items() if m is md)


def _plan(layer, md, fc, core_attn_out: torch.Tensor):
    """Eligibility of the step for this metadata object (one KV-cache group);
    packs it on success. Returns a _Step or the reason string.
    """
    M = _gdn()
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        _gdn_norm_mxfp8_num_warps,
    )
    from vllm.model_executor.layers.mamba.ops import gdn_conv_cuda, gdn_mtp_cuda
    from vllm.model_executor.layers.mamba.ops import gdn_fused_conv_prep as gfcp

    T = core_attn_out.size(0)
    if T > T_MAX:
        return "tokens"
    if md.num_prefills <= 0:
        return "no_prefill"
    if md.num_decodes != 0:
        return "non_spec_decodes"
    if md.prefill_checkpoint is not None:
        return "checkpoint"
    npf, P = md.num_prefills, md.num_prefill_tokens
    if npf > NP:
        return "num_prefills"
    ssm = layer.kv_cache[1]
    has_spec = md.spec_sequence_masks is not None
    if has_spec:
        nr, S = md.num_spec_decodes, md.num_spec_decode_tokens
        si = md.spec_state_indices_tensor
        if (
            nr <= 0
            or md.spec_token_start != 0
            or md.non_spec_token_start != S
            or not layer._can_use_fused_gdn_mtp_decode(md)
        ):
            return "spec_layout"
        if nr > NR:
            return "num_spec"
        spec, nrb, mtp_cfg = _spec_bucket(M, nr)
        if spec == SPEC_CUDA and not gdn_mtp_cuda.ready():
            return "mtp_csrc"
        if not si.is_contiguous():
            return "spec_layout"
    else:
        nr = S = nrb = 0
        spec, mtp_cfg, si = SPEC_NONE, None, None
    N = S + P
    if md.num_actual_tokens != N:
        return "rows"
    # Prefill half: CUDA fused conv prep + in-place V-split chunk kernel.
    if not (
        M._fused_conv_prep_applies(layer, md)
        and gfcp._cuda_kernel_ready
        and gfcp._cuda_kernel_ready[0] is gdn_conv_cuda.gdn_conv_cuda_prep
        and gdn_conv_cuda._ext
    ):
        return "conv"
    pcu = md.prefill_query_start_loc
    if pcu is not md.non_spec_query_start_loc or pcu.dtype != torch.int32:
        return "prefill_md"
    if (
        not M._gdn_vsplit_ready
        or not layer.chunk_gated_delta_rule.updates_state_in_place(ssm.dtype)
        or not layer.chunk_gated_delta_rule.expects_exp_g
    ):
        return "chunk"
    if not (npf > 1 or M._gdn_fi_non_cp_max_tokens() >= P):
        return "cp"
    HV = layer.num_v_heads // layer.tp_size
    vsf = M._gdn_vsplit_ready[0].choose_vsplit(
        npf, P, P if npf == 1 else md.prefill_max_seqlen, hv=HV
    )
    if vsf <= 1:
        return "vsplit1"
    # Eager CUDA conv launch config (gdn_conv_cuda_prep).
    tph = 4 if 1024 * max(npf, 1) > P else 8
    # Eager zeroes the fresh pool rows unless the builder saw on the host
    # that every prefill row has an initial state (VLLM_GDN_HOST_TRIM).
    zero = not (GDN_HOST_TRIM and md.prefill_all_initial_state)
    norm_rows = (S, N) if spec == SPEC_CUDA else (0, N)
    nw = _gdn_norm_mxfp8_num_warps(
        norm_rows[1] - norm_rows[0],
        triton.next_power_of_2(HV),
        core_attn_out.device,
    )
    names = _group_names(fc, md)
    ns_si = md.non_spec_state_indices_tensor
    ps_si = md.prefill_state_indices
    ns_hi = md.has_initial_state
    ps_hi = md.prefill_has_initial_state
    if (
        ns_si.dim() != 1
        or ps_si.dim() != 1
        or not ns_hi.is_contiguous()
        or not ps_hi.is_contiguous()
        or ns_hi.dtype != torch.bool
        or ps_hi.dtype != torch.bool
    ):
        return "prefill_md"
    if has_spec and not (
        md.spec_query_start_loc.is_contiguous()
        and md.num_accepted_tokens.is_contiguous()
    ):
        return "spec_layout"
    W = si.size(1) if has_spec else 1
    dt = (si.dtype if has_spec else torch.int32, ns_si.dtype, ps_si.dtype)
    gb = _BUFS.get(names)
    if gb is None:
        gb = _BUFS[names] = _GroupBufs(core_attn_out.device, *dt)
    elif gb.dtypes[1:] != dt[1:] or (has_spec and gb.dtypes[0] != dt[0]):
        return "dtype"
    si_buf = gb.si(W)
    _launch_pack(
        gb,
        si_buf,
        si if has_spec else None,
        md.spec_query_start_loc if has_spec else None,
        md.num_accepted_tokens if has_spec else None,
        pcu,
        ns_si,
        ns_hi,
        ps_si,
        ps_hi,
        S,
        N,
        nr,
        npf,
        norm_rows,
    )
    variant = (T, spec, W, nrb, npf, vsf, tph, nw, zero)
    return _Step(
        variant=variant,
        bufs=gb,
        width=W,
        nr=nr,
        nrb=nrb,
        npf=npf,
        spec=spec,
        tph=tph,
        vsf=vsf,
        norm_rows=norm_rows,
        mtp_cfg=mtp_cfg,
        T=T,
        zero=zero,
    )


def _workspace(dev, HQ: int, HV: int) -> torch.Tensor:
    from vllm.third_party.flashinfer_gdn_vsplit.adapter import _num_sm
    from vllm.third_party.flashinfer_gdn_vsplit.gdn_chunked_vs import (
        GatedDeltaNetChunkedKernel,
    )

    key = (dev, HQ, HV)
    ws = _WS.get(key)
    if ws is None:
        idx = (
            dev.index
            if dev.index is not None
            else torch.accelerator.current_device_index()
        )
        ws = _WS[key] = torch.empty(
            GatedDeltaNetChunkedKernel.get_workspace_size(
                _num_sm(idx), NP, HQ, HV, True
            ),
            dtype=torch.int8,
            device=dev,
        )
    return ws


def graph_core(
    layer,
    st: _Step,
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    out_q: torch.Tensor,
    out_scale: torch.Tensor,
) -> None:
    """The eager mixed / prefill-only GDN core of one layer
    (``_forward_core_fused_norm`` with ``out_q``) on the group's packed
    buffers; captured into the layer graph (also run eagerly by the null
    warm-up and by tests).
    """
    M = _gdn()
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        gdn_gated_norm_mxfp8,
        zero_fresh_state_rows,
    )
    from vllm.model_executor.layers.mamba.ops import gdn_conv_cuda, gdn_mtp_cuda
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.model_executor.layers.mamba.ops.gdn_mtp_decode import (
        gdn_mtp_recurrence,
    )

    gb = st.bufs
    T = core_attn_out.size(0)
    qkv_size = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
    mixed_qkv, gate_flat = mixed_qkvz.split(
        [qkv_size, layer.value_dim // layer.tp_size], dim=-1
    )
    output_gate = gate_flat.reshape(gate_flat.size(0), -1, layer.head_v_dim)
    b, a = layer.split_ba(ba)
    conv_state, conv_w = M._host_layer_views(layer)
    ssm = layer.kv_cache[1]
    scale = layer.head_k_dim**-0.5
    if st.spec != SPEC_NONE:
        # Spec rows [0, S): conv window update in place, then the MTP
        # recurrence (_forward_core_decode_spec_fused_norm).
        nrb = st.nrb
        si = gb.si(st.width)
        mq = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_w,
            layer.conv1d.bias,
            layer.activation,
            conv_state_indices=si[:nrb, 0],
            num_accepted_tokens=gb.acc[:nrb],
            query_start_loc=gb.cu_s[: nrb + 1],
            max_query_len=st.width,
            validate_data=False,
        )
        if st.spec == SPEC_TRITON:
            assert st.mtp_cfg is not None
            gdn_mtp_recurrence(
                mq,
                a,
                b,
                layer.A_log,
                layer.dt_bias,
                si[:nrb],
                gb.cu_s[: nrb + 1],
                gb.acc[:nrb],
                ssm,
                core_attn_out,
                scale=scale,
                block_v=st.mtp_cfg[0],
                num_warps=st.mtp_cfg[1],
            )
        else:
            ok = gdn_mtp_cuda.gdn_mtp_cuda(
                mq,
                a,
                b,
                layer.A_log,
                layer.dt_bias,
                si[:nrb],
                gb.cu_s[: nrb + 1],
                gb.acc[:nrb],
                ssm,
                output_gate,
                layer.norm.weight,
                core_attn_out,
                scale,
                layer.layer_norm_epsilon,
                layer.norm.activation,
            )
            if not ok:
                raise RuntimeError("GDN layer graphs: CUDA MTP kernel contract")
    # Prefill rows [S, N): CUDA conv + post-conv prep over absolute rows
    # (gdn_conv_cuda_prep), fresh-state zeroing, in-place V-split chunk.
    H = layer.num_k_heads // layer.tp_size
    HV = layer.A_log.shape[0]
    K, V = layer.head_k_dim, layer.head_v_dim
    dev = mixed_qkv.device
    q = torch.empty(T, H, K, dtype=mixed_qkv.dtype, device=dev)
    k = torch.empty(T, H, K, dtype=mixed_qkv.dtype, device=dev)
    v = torch.empty(T, HV, V, dtype=mixed_qkv.dtype, device=dev)
    gb_ts = 0
    if envs.VLLM_GDN_VSPLIT_HEAD_MAJOR:
        gb_ts = max(64, ((T + 63) // 64) * 64)
        g = torch.empty(HV, gb_ts, dtype=torch.float32, device=dev)[:, :T].t()
        beta = torch.empty(HV, gb_ts, dtype=torch.float32, device=dev)[:, :T].t()
    else:
        g = torch.empty(T, HV, dtype=torch.float32, device=dev)
        beta = torch.empty(T, HV, dtype=torch.float32, device=dev)
    npf = st.npf
    pdl = envs.VLLM_GDN_VSPLIT_PDL
    # Resolve scratch before the producer; no intervening GPU work is allowed
    # between its early trigger and the chunk's dependency wait. Fresh SSM
    # zeroing is independent of the conv-state update and moves before conv.
    ws = _workspace(dev, H, HV) if pdl else None
    if pdl and st.zero:
        zero_fresh_state_rows(ssm, gb.ci_chunk[:npf], gb.hi_chunk[:npf])
    ok = gdn_conv_cuda._ext[0].run(
        mixed_qkv,
        conv_w,
        conv_state,
        gb.ci_conv[:npf],
        gb.hi_conv[:npf],
        gb.cu_p[: npf + 1],
        npf,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        q,
        k,
        v,
        g,
        beta,
        H,
        st.tph,
        0,
        gb_ts=gb_ts,
        pdl=int(pdl),
    )
    if not ok:
        raise RuntimeError("GDN layer graphs: CUDA conv contract")
    if st.zero and not pdl:
        zero_fresh_state_rows(ssm, gb.ci_chunk[:npf], gb.hi_chunk[:npf])
    M._gdn_vsplit_ready[0].chunk_gated_delta_rule_vsplit(
        q,
        k,
        v,
        g,
        beta,
        core_attn_out,
        gb.cu_p[: npf + 1],
        ssm,
        ssm,
        K**-0.5,
        state_indices=gb.ci_chunk[:npf],
        v_split=st.vsf,
        workspace=ws if pdl else _workspace(dev, H, HV),
        pdl=pdl,
        h0_late=True,
    )
    # Gated RMSNorm + out_proj's MXFP8 activation (_gated_norm_mxfp8) with
    # [num_valid, norm_lo, norm_hi] read on the device.
    assert layer.norm.bias is None and layer.norm.norm_before_gate
    gdn_gated_norm_mxfp8(
        core_attn_out,
        output_gate,
        layer.norm.weight,
        layer.norm.eps,
        layer.norm.activation,
        out_q,
        out_scale,
        st.norm_rows,
        gb.rows[:1],
        rows=gb.rows,
    )


def _capture(layer, st, mixed_qkvz, ba, core_attn_out, out_q, out_scale):
    dev = core_attn_out.device
    if _CAP[0] is None:
        _CAP[0] = torch.cuda.Stream(device=dev)
    if _CAP[1] is None or not _GRAPHS:
        # A private pool lives as long as a graph uses it; never reuse the
        # handle of one that drained (e.g. after a failed first capture).
        _CAP[1] = torch.cuda.graph_pool_handle()
    gb = st.bufs
    # Null warm-up on the real buffers' addresses (same Triton
    # specializations, every module loaded, V-split compiled): save the
    # packed contents, run on zero-length contents, restore.
    saved = [t.clone() for t in gb.tensors()]
    gb.fill_null()
    try:
        graph_core(layer, st, mixed_qkvz, ba, core_attn_out, out_q, out_scale)
    finally:
        for t, s in zip(gb.tensors(), saved):
            t.copy_(s)
    g = torch.cuda.CUDAGraph()
    s = _CAP[0]
    cur = torch.cuda.current_stream()
    s.wait_stream(cur)
    with torch.cuda.stream(s):
        g.capture_begin(pool=_CAP[1], capture_error_mode="thread_local")
        try:
            graph_core(layer, st, mixed_qkvz, ba, core_attn_out, out_q, out_scale)
        finally:
            g.capture_end()
    cur.wait_stream(s)
    return g


def try_replay(
    layer,
    md,
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    out_q: torch.Tensor | None,
    out_scale: torch.Tensor | None,
) -> bool:
    """Start of ``_forward_core_fused_norm_packed`` with GDN metadata. True:
    this layer's core was replayed from its graph (the eager path must not
    run); False: run the eager path.
    """
    if not _ARMED[0] or _OFF[0] or _BYPASS[0] or out_q is None or out_scale is None:
        return False
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    # The op may run outside a model forward (direct layer calls, e.g. kernel
    # tests after a runner armed the graphs in the same process): eager.
    if not is_forward_context_available():
        return False
    fc = get_forward_context()
    if fc.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE:
        return False
    st = md.__dict__.get(_KEY)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        try:
            st = _plan(layer, md, fc, core_attn_out)
        except Exception as e:  # noqa: BLE001 - never half-pack a step
            st = "plan_error"
            _stat("plan_errors")
            if STATS["plan_errors"] <= 3:
                logger.warning("GDN layer graphs: plan error -> eager: %r", e)
        md.__dict__[_KEY] = st
        _stat("steps")
        if isinstance(st, str):
            _stat("ineligible_" + st)
        else:
            _stat("packed")
        if STATS["steps"] % LOG_EVERY == 1:
            logger.info("GDN layer graphs stats: %s", dict(sorted(STATS.items())))
    if isinstance(st, str) or core_attn_out.size(0) != st.T:
        return False
    key = (
        layer.prefix,
        st.variant,
        mixed_qkvz.data_ptr(),
        ba.data_ptr(),
        core_attn_out.data_ptr(),
        out_q.data_ptr(),
        out_scale.data_ptr(),
        layer.kv_cache[0].data_ptr(),
        layer.kv_cache[1].data_ptr(),
    )
    layer._in_proj_ba_join()
    graph = _GRAPHS.get(key)
    if graph is None:
        if key in _FAILED or len(_GRAPHS) >= MAX_GRAPHS:
            _stat("eager_uncaptured")
            return False
        t0 = time.perf_counter()
        try:
            graph = _capture(layer, st, mixed_qkvz, ba, core_attn_out, out_q, out_scale)
        except Exception as e:  # noqa: BLE001 - keep this key on the eager path
            _FAILED.add(key)
            _stat("capture_errors")
            if STATS["capture_errors"] <= 3:
                logger.warning(
                    "GDN layer graphs: capture failed (%s, variant %s) -> eager: %r",
                    layer.prefix,
                    st.variant,
                    e,
                )
            return False
        _GRAPHS[key] = graph
        _stat("captured")
        _stat("capture_us", int((time.perf_counter() - t0) * 1e6))
    md.__dict__[REPLAYED_KEY] = True
    if _CHECK[0] > 0:
        _CHECK[0] -= 1
        return _check(
            layer, md, st, graph, mixed_qkvz, ba, core_attn_out, out_q, out_scale
        )
    graph.replay()
    _stat("replays")
    return True


def _touched_slots(md, st) -> torch.Tensor:
    parts = []
    if st.spec != SPEC_NONE:
        parts.append(md.spec_state_indices_tensor[: st.nr].reshape(-1).long())
    parts.append(md.non_spec_state_indices_tensor[: st.npf].long())
    parts.append(md.prefill_state_indices[: st.npf].long())
    slots = torch.unique(torch.cat(parts))
    return slots[slots != NULL_BLOCK_ID]


def _check(
    layer, md, st, graph, mixed_qkvz, ba, core_attn_out, out_q, out_scale
) -> bool:
    """VLLM_GDN_LAYER_GRAPHS_CHECK: replay vs the eager path on the same
    inputs and state, bitwise. Keeps the eager result.
    """
    conv, ssm = layer.kv_cache[0], layer.kv_cache[1]
    slots = _touched_slots(md, st)
    N = md.num_actual_tokens
    outs = (mixed_qkvz, core_attn_out, out_q, out_scale)
    pre = [t.clone() for t in outs]
    pre_conv, pre_ssm = conv[slots].clone(), ssm[slots].clone()
    graph.replay()
    got = [t.clone() for t in outs]
    got_conv, got_ssm = conv[slots].clone(), ssm[slots].clone()
    for t, p in zip(outs, pre):
        t.copy_(p)
    conv[slots] = pre_conv
    ssm[slots] = pre_ssm
    _BYPASS[0] = True
    try:
        layer._forward_core_fused_norm_packed(
            mixed_qkvz, ba, core_attn_out, out_q=out_q, out_scale=out_scale
        )
    finally:
        _BYPASS[0] = False
    bad = []
    pairs = (
        ("qkvz", mixed_qkvz, got[0]),
        ("core_out", core_attn_out[:N], got[1][:N]),
        ("out_q", out_q, got[2]),
        ("out_scale", out_scale, got[3]),
        ("conv", conv[slots], got_conv),
        ("ssm", ssm[slots], got_ssm),
    )
    for name, x, y in pairs:
        if not torch.equal(x.view(torch.uint8), y.view(torch.uint8)):
            bad.append(name)
    _stat("check")
    if bad:
        _stat("check_fail")
        _OFF[0] = True
        logger.warning(
            "GDN layer graphs: CHECK MISMATCH %s variant=%s S=%d P=%d nseq=%d: %s; "
            "keeping the eager result, layer graphs disabled",
            layer.prefix,
            st.variant,
            md.num_spec_decode_tokens,
            md.num_prefill_tokens,
            st.npf,
            bad,
        )
    elif STATS["check"] in (1, 10, 100, 1000) or _CHECK[0] == 0:
        logger.info(
            "GDN layer graphs: check ok #%d (%s variant=%s)",
            STATS["check"],
            layer.prefix,
            st.variant,
        )
    return True
