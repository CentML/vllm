# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused GDN attention-metadata builder for spec-decode steps (opt-in).

Per engine step vLLM builds GDN metadata once per GDN KV-cache group (Qwen3.5 /
Qwen3.6-35B-A3B: 3 groups x 10 layers). For MTP spec-decode steps
GDNAttentionMetadataBuilder._build_full issues ~20-60 small torch kernels and
several H2D copies per group: mamba_get_block_table_tensor (sub, floor-div,
clamp, arange, add, cast, gather), boolean-mask indexing with CPU masks
(nonzero + H2D + index), repeat_interleave + stable argsort for the spec /
non-spec token permutation, zeros + cumsum for the query_start_loc splits,
compute_num_computed_tokens, the deferred-state-commit post-step and, for
decode-only steps, copy_ / fill_ into the builder's persistent FULL-cudagraph
buffers.

With VLLM_GDN_FUSED_MD=1 the two spec-decode step shapes that dominate serving
are built by ONE Triton launch per group:

  MIXED      spec-decode requests first (spec prefix) + prefill (and
             reclassified 1-token) requests; PIECEWISE step; fresh outputs.
  SPEC_ONLY  spec-decode requests only (+ zero-length padding rows at the back)
             with FULL cudagraphs: writes the builder's persistent buffers.

Every output is integer / bool metadata computed with the same integer
arithmetic from the same GPU inputs (query_start_loc, seq_lens, block table,
num_accepted_tokens), so it is bit-identical by construction; tensors have the
same shapes, dtypes, strides and aliasing as the regular build (views of one
permutation tensor, prefill_* aliases, the deferred-commit expand view,
persistent-buffer views). All CPU-side values (counts, CPU cumsums,
prefill_max_seqlen) and the FLA chunk / causal-conv1d metadata calls (deferred
under GGM_LAZY=1 exactly like the regular build) run the regular code. Any
other step (no spec rows, non-prefix spec rows, non-FULL spec-only, capture,
unexpected dtypes / layouts) returns None and the regular build runs.

Env (defaults in brackets):
  VLLM_GDN_FUSED_MD [0]      1 = enable (GS3MD=1 is accepted as an alias).
  VLLM_GDN_FUSED_MD_MODES    [mixed,spec].
  VLLM_GDN_FUSED_MD_CHECK    [0]: the first N fused builds per mode also run
                             the regular build and compare every field (values,
                             dtype, shape, stride, aliasing, persistent-buffer
                             bytes); on a mismatch the regular result is used
                             and the fused path is disabled.
  VLLM_GDN_FUSED_MD_MAT      [1]: with the GDN step plan's group materialize,
                             pass views of the fused accepted counts / initial-
                             state flags instead of zeros + slice copies
                             (identical values; checked for numel >= items).
  VLLM_GDN_FUSED_MD_LOG_EVERY [20000] builds.

Port of the Rubin study's F108 (glue-sol3 gs3md, py-md2), validated there as
bit-exact on 588 unit cases and 73,901 in-serving checked builds.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

ENABLED = (
    os.environ.get("VLLM_GDN_FUSED_MD", os.environ.get("GS3MD", "0")) == "1"
)
MODES = set(os.environ.get("VLLM_GDN_FUSED_MD_MODES", "mixed,spec").split(","))
_CHECK0 = int(os.environ.get("VLLM_GDN_FUSED_MD_CHECK", "0"))
CHECK = {"mixed": _CHECK0, "spec": _CHECK0}
MAT = ENABLED and os.environ.get("VLLM_GDN_FUSED_MD_MAT", "1") == "1"
LOG_EVERY = int(os.environ.get("VLLM_GDN_FUSED_MD_LOG_EVERY", "20000"))
STATS: dict = {
    "calls": 0,
    "mixed": 0,
    "spec": 0,
    "fallback": {},
    "check_ok": 0,
    "check_fail": 0,
    "mat_views": 0,
}
_DISABLED = [False]
_IN_CAPTURE = [False]
_FLAG = "_fused_md"


def _jit(ints, ptrs):
    # no value / alignment specialization: one compiled variant per constexpr
    # combination
    try:
        return triton.jit(
            do_not_specialize=ints + ptrs, do_not_specialize_on_alignment=ptrs
        )
    except TypeError:
        return triton.jit(do_not_specialize=ints + ptrs)


# fmt: off
@_jit(["bt_stride", "R", "NR", "N_NS_TOK", "BS"],
      ["QSL", "SEQ", "BT", "NACC", "SMASK", "TOKIDX", "SSI", "NSSI", "SQSL", "NSQSL", "NACCS", "HASINIT", "GSCNS"])
def _fused_md_mixed_kernel(QSL, SEQ, BT, bt_stride, NACC,
                           SMASK, TOKIDX, SSI, NSSI, SQSL, NSQSL, NACCS, HASINIT, GSCNS,
                           R, NR, N_NS_TOK, BS,
                           W: tl.constexpr, W_PAD: tl.constexpr, ALIGN: tl.constexpr, R_PAD: tl.constexpr,
                           TB: tl.constexpr, GSC: tl.constexpr):
    """One program per request r (spec rows are r < NR). Replicates, for request r:
    mamba_get_block_table_tensor (align: start = clamp((seq-1)//bs, 0); cols start..start+W-1), the CPU-mask
    compactions bt[mask,:W] / bt[~mask,0] / nacc[mask] / nacc[~mask] / (seq-qlen>0)[~mask], the zero+cumsum query
    start locs of both sides, the H2D'd spec mask, and argsort(repeat_interleave(mask, qlen), stable) (non-spec
    token positions first, then spec token positions, each in increasing order)."""
    r = tl.program_id(0)
    offs = tl.arange(0, R_PAD)
    m = offs < R
    q0 = tl.load(QSL + offs, mask=m, other=0)
    q1 = tl.load(QSL + offs + 1, mask=m, other=0)
    lens = q1 - q0
    before = m & (offs < r)
    sb = before & (offs < NR)
    nb = before & (offs >= NR)
    stok = tl.sum(tl.where(sb, lens, 0), axis=0)
    ntok = tl.sum(tl.where(nb, lens, 0), axis=0)
    my_q0 = tl.load(QSL + r)
    my_len = tl.load(QSL + r + 1) - my_q0
    seq = tl.load(SEQ + r)
    if ALIGN:
        start = tl.maximum((seq - 1) // BS, 0)
    else:
        start = seq * 0
    row = BT + r.to(tl.int64) * bt_stride
    is_spec = r < NR
    if r == 0:
        tl.store(SQSL, stok * 0)
        tl.store(NSQSL, stok * 0)
    tl.store(SMASK + r, is_spec)
    base = tl.where(is_spec, N_NS_TOK + stok, ntok)
    for i0 in range(0, my_len, TB):
        ii = i0 + tl.arange(0, TB)
        mk = ii < my_len
        tl.store(TOKIDX + (base + ii).to(tl.int64), (my_q0 + ii).to(tl.int64), mask=mk)
    if is_spec:
        jj = tl.arange(0, W_PAD)
        mw = jj < W
        v = tl.load(row + start + jj, mask=mw, other=0)
        tl.store(SSI + r * W + jj, v, mask=mw)
        tl.store(NACCS + r, tl.load(NACC + r))
        tl.store(SQSL + r + 1, stok + my_len)
    else:
        k = r - NR
        tl.store(NSSI + k, tl.load(row + start))
        tl.store(HASINIT + k, (seq - my_len) > 0)
        tl.store(NSQSL + k + 1, ntok + my_len)
        if GSC:
            tl.store(GSCNS + k, tl.load(NACC + r))


@_jit(["bt_stride", "ssi_ld", "NR", "STS", "BS", "NULL_ID"],
      ["QSL", "SEQ", "BT", "NACC", "SSI_B", "SMASK_B", "STI_B", "SQSL_B", "NACC_B"])
def _fused_md_spec_kernel(QSL, SEQ, BT, bt_stride, NACC,
                          SSI_B, ssi_ld, SMASK_B, STI_B, SQSL_B, NACC_B,
                          NR, STS, BS, NULL_ID,
                          W: tl.constexpr, WF: tl.constexpr, WF_PAD: tl.constexpr, ALIGN: tl.constexpr,
                          TB: tl.constexpr):
    """One program per row r < batch_size of the FULL-cudagraph persistent buffers (spec rows r < NR). Replicates
    self.X[:NR].copy_(...) (with the [NR,1] -> [NR,num_spec+1] broadcast when W == 1) + self.X[:bs][NR:].fill_(...),
    spec_token_indx = arange(STS) (int32), spec_query_start_loc = qsl[:NR+1] then filled with qsl[NR]."""
    r = tl.program_id(0)
    jj = tl.arange(0, WF_PAD)
    mw = jj < WF
    if r < NR:
        seq = tl.load(SEQ + r)
        if ALIGN:
            start = tl.maximum((seq - 1) // BS, 0)
        else:
            start = seq * 0
        row = BT + r.to(tl.int64) * bt_stride
        if W == 1:
            v = tl.load(row + start + jj * 0, mask=mw, other=0)
        else:
            v = tl.load(row + start + jj, mask=mw, other=0)
        tl.store(SSI_B + r * ssi_ld + jj, v, mask=mw)
        tl.store(SMASK_B + r, r >= 0)
        tl.store(NACC_B + r, tl.load(NACC + r))
        tl.store(SQSL_B + r + 1, tl.load(QSL + r + 1))
    else:
        tl.store(SSI_B + r * ssi_ld + jj, tl.zeros([WF_PAD], tl.int32) + NULL_ID, mask=mw)
        tl.store(SMASK_B + r, r < 0)
        tl.store(NACC_B + r, tl.full([], 1, tl.int32))
        tl.store(SQSL_B + r + 1, tl.load(QSL + NR))
    if r == 0:
        tl.store(SQSL_B, tl.load(QSL))
        for t0 in range(0, STS, TB):
            tt = t0 + tl.arange(0, TB)
            tl.store(STI_B + tt, tt.to(tl.int32), mask=tt < STS)
# fmt: on


def _fb(reason):
    d = STATS["fallback"]
    d[reason] = d.get(reason, 0) + 1


def _pad(n, lo=16):
    p = lo
    while p < n:
        p *= 2
    return p


def _counts(qsl_cpu, mask_cpu, nr):
    """The regular build's CPU arithmetic of the spec branch."""
    non_spec_mask_cpu = ~mask_cpu
    query_lens_cpu = qsl_cpu[1:] - qsl_cpu[:-1]
    non_spec_query_lens_cpu = query_lens_cpu[non_spec_mask_cpu]
    num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
    num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
    num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
    num_decode_tokens = num_decodes
    num_prefill_tokens = non_spec_query_lens_cpu.sum().item() - num_decode_tokens
    num_spec_decode_tokens = (
        query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
    )
    if num_decodes > 0 and nr > 0:
        num_prefills += num_decodes
        num_prefill_tokens += num_decode_tokens
        num_decodes = 0
        num_decode_tokens = 0
    return (
        non_spec_mask_cpu,
        query_lens_cpu,
        num_decodes,
        num_prefills,
        num_decode_tokens,
        num_prefill_tokens,
        num_spec_decode_tokens,
    )


def _ga():
    from vllm.v1.attention.backends import gdn_attn

    return gdn_attn


def _fast(b, m, num_accepted_tokens, ndd):
    """Fused build or None (-> regular build)."""
    ga = _ga()
    if not b.use_spec_decode or ndd is None or num_accepted_tokens is None:
        _fb("no_spec_inputs")
        return None
    qsl, qsl_cpu = m.query_start_loc, m.query_start_loc_cpu
    seq_lens, bt_in = m.seq_lens, m.block_table_tensor
    R = ndd.numel()
    if (
        qsl.numel() != R + 1
        or qsl_cpu.numel() != R + 1
        or seq_lens.numel() != R
        or num_accepted_tokens.numel() != R
        or m.num_reqs != R
    ):
        _fb("shape")
        return None
    if (
        qsl.dtype != torch.int32
        or seq_lens.dtype != torch.int32
        or bt_in.dtype != torch.int32
        or num_accepted_tokens.dtype != torch.int32
        or not qsl.is_cuda
        or bt_in.dim() != 2
        or bt_in.stride(1) != 1
        or not qsl.is_contiguous()
        or not seq_lens.is_contiguous()
        or not num_accepted_tokens.is_contiguous()
        or R == 0
        or R > 4096
    ):
        _fb("layout")
        return None
    mask_cpu = ndd >= 0
    nr = mask_cpu.sum().item()
    if nr == 0 or ndd[mask_cpu].sum().item() == 0:
        _fb("no_spec_rows")
        return None
    if not bool(mask_cpu[:nr].all().item()):
        _fb("spec_not_prefix")
        return None
    spec = b.kv_cache_spec
    mode = b.vllm_config.cache_config.mamba_cache_mode
    if mode in ("all", "none"):
        align = False
        if bt_in.size(0) != R:
            _fb("bt_rows")
            return None
        bt_cols = bt_in.size(1)
    else:
        align = True
        if not isinstance(spec, ga.MambaSpec) or bt_in.size(0) < R:
            _fb("align_spec")
            return None
        bt_cols = 1 + spec.num_speculative_blocks
    W = min(b.num_spec + 1, bt_cols)
    (
        non_spec_mask_cpu,
        query_lens_cpu,
        num_decodes,
        num_prefills,
        num_decode_tokens,
        num_prefill_tokens,
        num_spec_decode_tokens,
    ) = _counts(qsl_cpu, mask_cpu, nr)
    dev = qsl.device
    BS = spec.block_size
    gsc = ga._GDN_STATE_COMMIT_DEFERRED
    if num_prefills == 0 and num_decodes == 0:
        # ---------------- SPEC_ONLY (FULL-cudagraph persistent buffers) -------
        if "spec" not in MODES:
            _fb("mode_off_spec")
            return None
        batch_size = m.num_reqs
        if not (
            b.use_full_cuda_graph
            and nr <= b.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= b.decode_cudagraph_max_bs
            and batch_size <= b.decode_cudagraph_max_bs
        ):
            _fb("spec_not_full")
            return None
        WF = b.num_spec + 1
        if W != 1 and W != WF:
            _fb("spec_width")
            return None
        sts = min(nr * (b.num_spec + 1), qsl_cpu[-1].item())
        ssi_b = b.spec_state_indices_tensor
        if (
            ssi_b.stride(1) != 1
            or ssi_b.size(1) != WF
            or b.num_accepted_tokens.dtype != torch.int32
        ):
            _fb("spec_buf_layout")
            return None
        _fused_md_spec_kernel[(batch_size,)](
            qsl,
            seq_lens,
            bt_in,
            bt_in.stride(0),
            num_accepted_tokens,
            ssi_b,
            ssi_b.stride(0),
            b.spec_sequence_masks,
            b.spec_token_indx,
            b.spec_query_start_loc,
            b.num_accepted_tokens,
            nr,
            sts,
            BS,
            ga.NULL_BLOCK_ID,
            W=W,
            WF=WF,
            WF_PAD=_pad(WF, 1),
            ALIGN=align,
            TB=1024,
            num_warps=1,
        )
        md = ga.GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=nr,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=None,
            chunk_indices=None,
            chunk_offsets=None,
            prefill_query_start_loc=None,
            prefill_state_indices=None,
            prefill_has_initial_state=None,
            prefill_max_seqlen=0,
            spec_query_start_loc=b.spec_query_start_loc[: batch_size + 1],
            non_spec_query_start_loc=None,
            spec_state_indices_tensor=ssi_b[:batch_size],
            non_spec_state_indices_tensor=None,
            spec_sequence_masks=b.spec_sequence_masks[:batch_size],
            spec_token_indx=b.spec_token_indx[:sts],
            non_spec_token_indx=b.non_spec_token_indx[:0],
            num_accepted_tokens=b.num_accepted_tokens[:batch_size],
            spec_tokens_are_prefix=False,
            nums_dict=None,
            batch_ptr=None,
            token_chunk_offset_ptr=None,
        )
        if gsc:
            _gsc_post(b, md, None)
        md.__dict__[_FLAG] = True
        STATS["spec"] += 1
        return md
    # ---------------- MIXED (PIECEWISE; fresh tensors) ------------------------
    if "mixed" not in MODES:
        _fb("mode_off_mixed")
        return None
    N = qsl_cpu[-1].item()
    n_ns_tok = num_prefill_tokens + num_decode_tokens
    nns = R - nr
    smask = torch.empty(R, dtype=torch.bool, device=dev)
    index = torch.empty(N, dtype=torch.int64, device=dev)
    ssi = torch.empty((nr, W), dtype=torch.int32, device=dev)
    nssi = torch.empty(nns, dtype=torch.int32, device=dev)
    sqsl = torch.empty(nr + 1, dtype=torch.int32, device=dev)
    nsqsl = torch.empty(nns + 1, dtype=torch.int32, device=dev)
    naccs = torch.empty(nr, dtype=torch.int32, device=dev)
    hasinit = torch.empty(nns, dtype=torch.bool, device=dev)
    gscns = torch.empty(nns if gsc else 1, dtype=torch.int32, device=dev)
    _fused_md_mixed_kernel[(R,)](
        qsl,
        seq_lens,
        bt_in,
        bt_in.stride(0),
        num_accepted_tokens,
        smask,
        index,
        ssi,
        nssi,
        sqsl,
        nsqsl,
        naccs,
        hasinit,
        gscns,
        R,
        nr,
        n_ns_tok,
        BS,
        W=W,
        W_PAD=_pad(W, 1),
        ALIGN=align,
        R_PAD=256 if R <= 256 else _pad(R),
        TB=1024,
        GSC=gsc,
        num_warps=4,
    )
    non_spec_token_indx = index[:n_ns_tok]
    spec_token_indx = index[n_ns_tok:]
    # CPU side: the regular build's code
    non_spec_query_start_loc_cpu = torch.zeros(
        query_lens_cpu.size(0) - nr + 1, dtype=torch.int32
    )
    torch.cumsum(
        query_lens_cpu[non_spec_mask_cpu], dim=0, out=non_spec_query_start_loc_cpu[1:]
    )
    prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
    prefill_max_seqlen = int(
        (prefill_query_start_loc_cpu[1:] - prefill_query_start_loc_cpu[:-1]).max()
    )
    from vllm.model_executor.layers.mamba.gdn import gdn_step_plan

    lazy: list | None = [] if gdn_step_plan.LAZY else None
    if lazy is not None:
        lazy.append(
            (
                b._build_chunk_metadata,
                "chunk",
                (nsqsl, prefill_query_start_loc_cpu, dev),
                {},
            )
        )
        chunk_indices, chunk_offsets = None, None
        lazy.append(
            (
                ga.compute_causal_conv1d_metadata,
                "ccm",
                (non_spec_query_start_loc_cpu,),
                {"device": dev},
            )
        )
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
    else:
        chunk_indices, chunk_offsets = b._build_chunk_metadata(
            nsqsl, prefill_query_start_loc_cpu, dev
        )
        nums_dict, batch_ptr, token_chunk_offset_ptr = (
            ga.compute_causal_conv1d_metadata(non_spec_query_start_loc_cpu, device=dev)
        )
    md = ga.GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_spec_decodes=nr,
        num_spec_decode_tokens=num_spec_decode_tokens,
        num_actual_tokens=m.num_actual_tokens,
        has_initial_state=hasinit,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        prefill_query_start_loc=nsqsl,
        prefill_state_indices=nssi,
        prefill_has_initial_state=hasinit,
        prefill_max_seqlen=prefill_max_seqlen,
        spec_query_start_loc=sqsl,
        non_spec_query_start_loc=nsqsl,
        spec_state_indices_tensor=ssi,
        non_spec_state_indices_tensor=nssi,
        spec_sequence_masks=smask,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=naccs,
        spec_tokens_are_prefix=True,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )
    if gsc:
        _gsc_post(b, md, gscns)
    if lazy:
        gdn_step_plan.defer_metadata(md, lazy)
    md.__dict__[_FLAG] = True
    STATS["mixed"] += 1
    return md


def _gsc_post(b, md, gscns):
    """gdn_state_commit.postprocess_metadata with the non-spec accepted-count
    gather precomputed by the kernel (num_accepted_tokens[mask][:n], mask =
    the non-spec rows, n = their count).
    """
    w = b.num_spec + 1
    si = md.spec_state_indices_tensor
    if si is not None and si.size(1) < w:
        md.spec_state_indices_tensor = si[:, :1].expand(si.size(0), w)
    md.gsc_non_spec_num_accepted = (
        gscns if md.non_spec_state_indices_tensor is not None else None
    )


def _tensor_desc(t):
    return (tuple(t.shape), t.dtype, t.device, tuple(t.stride()), t.storage_offset())


def _compare(a, b, buf_views):
    """a = regular md, b = fused md. Returns the list of mismatching fields."""
    bad = []
    ka = {k for k in a.__dict__ if not k.startswith("_")}
    kb = {k for k in b.__dict__ if not k.startswith("_")}
    if ka != kb:
        bad.append(f"keys:{sorted(ka ^ kb)}")
    tens = {}
    for k in sorted(ka & kb):
        x, y = a.__dict__[k], b.__dict__[k]
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
                bad.append(f"{k}:type")
                continue
            if _tensor_desc(x)[:4] != _tensor_desc(y)[:4]:
                bad.append(f"{k}:desc {_tensor_desc(x)} vs {_tensor_desc(y)}")
                continue
            yv = buf_views.get(k, y)
            if not torch.equal(x, yv):
                bad.append(f"{k}:values")
            tens[k] = (x, y)
        elif k == "nums_dict":
            if (x is None) != (y is None):
                bad.append(f"{k}:none")
        elif x != y:
            bad.append(f"{k}:{x!r} vs {y!r}")
    for lk in ("_step_plan_lazy",):
        la, lb = a.__dict__.get(lk), b.__dict__.get(lk)
        if (la is None) != (lb is None) or (
            la is not None and [x[1] for x in la] != [x[1] for x in lb]
        ):
            bad.append(f"{lk}:kinds")
    ks = sorted(tens)
    for i, k1 in enumerate(ks):
        for k2 in ks[i + 1 :]:
            x1, y1 = tens[k1]
            x2, y2 = tens[k2]
            if (x1 is x2) != (y1 is y2):
                bad.append(f"alias:{k1}/{k2}")
            sa = (
                x1.untyped_storage().data_ptr() == x2.untyped_storage().data_ptr()
                and x1.numel()
                and x2.numel()
            )
            sb = (
                y1.untyped_storage().data_ptr() == y2.untyped_storage().data_ptr()
                and y1.numel()
                and y2.numel()
            )
            if bool(sa) != bool(sb):
                bad.append(f"storage:{k1}/{k2}")
    return bad


_BUF_KEYS = (
    "spec_state_indices_tensor",
    "spec_sequence_masks",
    "spec_token_indx",
    "spec_query_start_loc",
    "num_accepted_tokens",
    "non_spec_token_indx",
)


def _checked(b, regular, m, nacc, ndd):
    bufs = {k: getattr(b, k) for k in _BUF_KEYS}
    snap = {k: v.clone() for k, v in bufs.items()}
    before = STATS["mixed"]
    mine = _fast(b, m, nacc, ndd)
    if mine is None:
        return None
    mode = "mixed" if STATS["mixed"] != before else "spec"
    if CHECK[mode] <= 0:
        return mine
    CHECK[mode] -= 1
    views = {}
    after = None
    if mode == "spec":
        # fused wrote the persistent buffers: keep a copy, restore the
        # pre-state, run the regular build
        after = {k: v.clone() for k, v in bufs.items()}
        for k, v in bufs.items():
            v.copy_(snap[k])
        for f in _BUF_KEYS:
            t = mine.__dict__.get(f)
            if isinstance(t, torch.Tensor):
                off = t.storage_offset() - bufs[f].storage_offset()
                views[f] = after[f].as_strided(
                    t.shape, t.stride(), after[f].storage_offset() + off
                )
    ref = regular()
    bad = _compare(ref, mine, views)
    if mode == "spec":
        full = [k for k in bufs if not torch.equal(bufs[k], after[k])]
        if full:
            bad.append(f"persistent-buffer bytes differ: {full}")
    if bad:
        STATS["check_fail"] += 1
        _DISABLED[0] = True
        logger.warning(
            "GDN fused metadata: CHECK MISMATCH mode=%s fields=%s; using the "
            "regular result, fused path disabled",
            mode,
            bad,
        )
        return ref
    STATS["check_ok"] += 1
    if STATS["check_ok"] in (1, 10, 100, 1000) or CHECK[mode] == 0:
        logger.info(
            "GDN fused metadata: check ok mode=%s (#%d; left mixed=%d spec=%d)",
            mode,
            STATS["check_ok"],
            CHECK["mixed"],
            CHECK["spec"],
        )
    # spec: the regular build rewrote the persistent buffers identically
    return ref if mode == "spec" else mine


def build(b, regular, m, num_accepted_tokens, num_decode_draft_tokens_cpu):
    """GDNAttentionMetadataBuilder.build with VLLM_GDN_FUSED_MD=1: the fused
    metadata, or None (the caller runs the regular build). regular() runs the
    regular build of the same step (CHECK mode only).
    """
    if _DISABLED[0] or _IN_CAPTURE[0]:
        return None
    STATS["calls"] += 1
    if LOG_EVERY and STATS["calls"] % LOG_EVERY == 0:
        logger.info("GDN fused metadata stats: %s", STATS)
    try:
        if torch.cuda.is_current_stream_capturing():
            _fb("capturing")
            return None
        if CHECK["mixed"] > 0 or CHECK["spec"] > 0:
            return _checked(
                b, regular, m, num_accepted_tokens, num_decode_draft_tokens_cpu
            )
        return _fast(b, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
    except Exception as e:  # noqa: BLE001 - never break serving
        _DISABLED[0] = True
        logger.warning("GDN fused metadata: error, disabled: %r", e)
        return None


class capture_guard:
    """build_for_cudagraph_capture: the fused path is never used for capture
    metadata.
    """

    def __enter__(self):
        _IN_CAPTURE[0] = True

    def __exit__(self, *exc):
        _IN_CAPTURE[0] = False
        return False


def mat_inputs(md, items):
    """VLLM_GDN_FUSED_MD_MAT: (accepted counts, initial-state flags) views for
    the group materialize of a fused md, or None (the caller builds zeros +
    slice copies). Identical values whenever numel >= items.
    """
    if not MAT or not md.__dict__.get(_FLAG):
        return None
    n_src = getattr(md, "gsc_non_spec_num_accepted", None)
    has_init = md.has_initial_state
    if (
        n_src is None
        or has_init is None
        or n_src.numel() < items
        or has_init.numel() < items
        or n_src.dtype != torch.int32
        or has_init.dtype != torch.bool
    ):
        return None
    STATS["mat_views"] += 1
    return n_src[:items], has_init[:items]


def log_config() -> None:
    if ENABLED:
        logger.info(
            "GDN fused metadata builder enabled: modes=%s check=%d mat=%d",
            sorted(MODES),
            _CHECK0,
            int(MAT),
        )

