# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph replay of the GDN core of mixed (prefill + MTP) steps.

Extends the GDN step plan (gdn_step_plan, GGM=1). With VLLM_GDN_LAYER_GRAPHS=1
everything that depends on the step moves to the GDN metadata builder (once per
KV-cache group per step, before the forward, same stream):

  * a pack kernel copies the step's metadata into fixed-address, fixed-size
    buffers of the group: spec slots / accepted counts / cu_seqlens padded to
    NR requests (zero-length pads), prefill cu_seqlens rebased to absolute token
    rows and padded to NP zero-length sequences, one cu_seqlens copy per V-split
    variant (the unselected one all-empty), and S / N;
  * the deferred-commit materialize of all layers of the group and the
    fresh-slot zeroing of all layers (one launch each);
  * eligibility: only a packed step carries the graph flag; every PIECEWISE step
    with GDN metadata that is not packed (CP routing, more than NP prefill
    sequences, more than NR spec requests, non-prefix spec batches, ...) runs
    with cudagraph_runtime_mode NONE, i.e. the eager step-plan path.

The captured per-layer sequence launches the kernels of the eager path, fed the
padded buffers: conv update (spec rows, NR padded requests) -> deferred-commit
MTP decode -> CUDA conv + post-conv (NP padded sequences, absolute rows) ->
V-split chunk kernel for v_split 1 and 2 (the unselected one has no work) ->
gated RMSNorm + MXFP8 quant of rows [S, N) and MXFP8 quant of rows [0, S) and
[N, pad(T)) (norm-quant fusion math, row ranges read on device).

VLLM_GDN_LAYER_GRAPHS_MODE:
  "layer": the GDN core op stays a splitting op (the compiled partition, and so
      every other kernel, is unchanged) and each GDN layer replays its own CUDA
      graph (one per layer, padded token count and buffer addresses).
  "piece" (default): the GDN op must be removed from the compilation
      splitting_ops by the user; it is then captured inside the piecewise graphs.
      Not bit-exact vs the eager path (the changed partition changes compiled
      numerics of other ops for some sizes).
Other knobs (VLLM_GDN_LAYER_GRAPHS_*): NR (256) / NP (16) / T_MAX (16384) static
bounds, TPH (8) CUDA-conv tokens per half warp, V2ONLY (capture only the
v_split=2 kernel and feed it every step), VSF_KEY (layer mode: one graph per
V-split variant), DEVICE_NB (use the device_nb V-split kernel copy: padded
sequences cost nothing), MERGED_NQ (1) / MERGED_NQ_MAX_T (8192) / NQ_ALL_PROGS
(2048) / NQ_TILES (64) norm-quant launch shapes, FORCE_EAGER (diagnostic: never
replay), FIX_PROFILE (1: drop every layer graph after vLLM's CUDA-graph memory
profiling and key graphs by the conv / SSM state pointers, so no graph recorded
against the throwaway profiling KV cache is ever replayed).
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.norm_quant_kernels import _mx_epilogue
from vllm.model_executor.layers.mamba.gdn import gdn_step_plan
from vllm.model_executor.layers.mamba.gdn.gdn_step_plan import (
    _TABLES,
    STATS,
    _gdn_zero_state_slots_layers_kernel,
    _GroupTable,
)
from vllm.triton_utils import tl, triton

# ruff: noqa: E501, SIM108, D209 - the Triton kernel sources below are kept verbatim

logger = init_logger(__name__)

_P = "VLLM_GDN_LAYER_GRAPHS"
ENABLED = gdn_step_plan.ENABLED and os.environ.get(_P, "0") == "1"
NR = int(os.environ.get(_P + "_NR", "256"))
NP = int(os.environ.get(_P + "_NP", "16"))
T_MAX = int(os.environ.get(_P + "_T_MAX", "16384"))
TPH = int(os.environ.get(_P + "_TPH", "8"))
V2ONLY = os.environ.get(_P + "_V2ONLY", "0") == "1"
VSF_KEY = os.environ.get(_P + "_VSF_KEY", "0") == "1"
NQ_TILES = int(os.environ.get(_P + "_NQ_TILES", "64"))
MERGED_NQ = os.environ.get(_P + "_MERGED_NQ", "1") == "1"
MERGED_NQ_MAX_T = int(os.environ.get(_P + "_MERGED_NQ_MAX_T", "8192"))
NQ_ALL_PROGS = int(os.environ.get(_P + "_NQ_ALL_PROGS", "2048"))
FORCE_EAGER = os.environ.get(_P + "_FORCE_EAGER", "0") == "1"
MODE = os.environ.get(_P + "_MODE", "piece").strip().lower()
DEVNB = os.environ.get(_P + "_DEVICE_NB", "0") == "1"
FIX_PROFILE = os.environ.get(_P + "_FIX_PROFILE", "1") == "1"
# Layer mode: a PIECEWISE step whose GDN metadata was not packed (CP routing,
# > T_MAX tokens, > NP prefill sequences, ...) keeps its PIECEWISE graphs and
# runs the GDN core on the eager step-plan path (1, default), instead of
# running the whole forward with cudagraph_runtime_mode NONE (0, the original
# behaviour; piece mode always needs NONE). Layer graphs are then captured
# only inside vLLM's capture phase (never while serving an unpacked step).
KEEP_PW = os.environ.get(_P + "_KEEP_PIECEWISE", "1") == "1"
# gdnp: VLLM_GDNP_SPEC_OVERLAP=1|2 captures the spec-decode part of the layer
# (conv update + deferred-commit decode of rows [0, S), spec slots) on a side
# stream, concurrent with the prefill part (CUDA conv + V-split chunk kernel of
# rows [S, N), prefill slots), joined before the norm-quant kernel. The two
# parts touch disjoint rows / state slots and run the same kernels, so the
# result is bitwise identical; the V-split kernel is latency-bound on
# num_prefills * HV * v_split CTAs and leaves SMs idle that the bandwidth-bound
# decode can use. Only inside graph capture (eager calls stay serial).
# 1 = fork at the layer start; 2 = fork after the prefill conv kernel.
SPEC_OVERLAP_MODE = int(os.environ.get("VLLM_GDNP_SPEC_OVERLAP", "0") or 0)
SPEC_OVERLAP = SPEC_OVERLAP_MODE in (1, 2)
_SIDE: dict = {}  # device -> side capture stream
# In-serving check: the first N layer replays of packed steps also run the
# eager path on the same inputs / state pages (restored in between) and compare
# outputs, MXFP8 stash buffers and the touched state pages bitwise; a mismatch
# keeps the eager result and disables the layer graphs.
_CHECK = [int(os.environ.get(_P + "_CHECK", "0"))]
MTPW = 4  # 1 + num_speculative_tokens (checked against the metadata)

_GB: dict = {}  # group key (layer names) -> _GroupBufs
_L2G: dict = {}  # layer name -> _GroupBufs
_SHARED: dict = {}  # conv-output buffers, shared by all groups (layers run in order)
_CAP: list = [None, None]  # layer mode: capture stream, private graph pool
_PHASE = ["real"]
_IN_CAPTURE = [False]  # inside GPUModelRunner.capture_model (both phases)
_BYPASS = [False]  # check mode: the eager re-run of a layer
_OFF = [False]  # disabled after a check mismatch
_GRAPH = "_step_plan_graph"  # md flag: this step was packed for the graphs
_ZEROED = "_step_plan_zeroed"  # md flag: fresh slots already zeroed
_VSF = "_step_plan_vsf"  # md: the step's v_split


def check_config() -> None:
    """Called once when the Qwen GDN layer module is imported."""
    if ENABLED:
        logger.info(
            "GDN layer graphs enabled: MODE=%s DEVICE_NB=%d NR=%d NP=%d T_MAX=%d "
            "TPH=%d V2ONLY=%d FIX_PROFILE=%d SPEC_OVERLAP=%d",
            MODE,
            int(DEVNB),
            NR,
            NP,
            T_MAX,
            TPH,
            int(V2ONLY),
            int(FIX_PROFILE),
            SPEC_OVERLAP_MODE,
        )


# fmt: off
@triton.jit(do_not_specialize=["S", "N", "nr", "npf", "VS2", "spec_si_s0"])
def _gdn_graph_pack_kernel(spec_si, spec_si_s0, spec_cu, nacc_src, ns_cu, slots_src, hi_src,
                           conv_si, dec_si, nacc, cu_s, cu_abs, cu_v1, cu_v2, ci, hi, sn,
                           S, N, nr, npf, VS2, NRC: tl.constexpr, NPC: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < nr
    si = tl.load(spec_si + offs * spec_si_s0, mask=m, other=0)
    tl.store(conv_si + offs, si, mask=offs < NRC)
    tl.store(dec_si + offs, si, mask=offs < NRC)
    na = tl.load(nacc_src + offs, mask=m, other=1)
    tl.store(nacc + offs, na, mask=offs < NRC)
    c = tl.load(spec_cu + offs, mask=offs <= nr, other=0)
    c = tl.where(offs <= nr, c, S)
    tl.store(cu_s + offs, c, mask=offs <= NRC)
    cp = tl.load(ns_cu + offs, mask=offs <= npf, other=0) + S
    cp = tl.where(offs <= npf, cp, N)
    tl.store(cu_abs + offs, cp, mask=offs <= NPC)
    tl.store(cu_v1 + offs, tl.where(VS2 == 0, cp, N), mask=offs <= NPC)
    tl.store(cu_v2 + offs, tl.where(VS2 == 1, cp, N), mask=offs <= NPC)
    # [NPC + 1]: number of valid sequences for the device_nb V-split kernel (0 = the unselected variant)
    tl.store(cu_v1 + NPC + 1 + offs, tl.where(VS2 == 0, npf, 0), mask=offs < 1)
    tl.store(cu_v2 + NPC + 1 + offs, tl.where(VS2 == 1, npf, 0), mask=offs < 1)
    mp = offs < npf
    sl = tl.load(slots_src + offs, mask=mp, other=0)
    tl.store(ci + offs, sl.to(tl.int32), mask=offs < NPC)
    h = tl.load(hi_src + offs, mask=mp, other=1)
    tl.store(hi + offs, h != 0, mask=offs < NPC)
    tl.store(sn + offs, tl.where(offs == 0, S, N), mask=offs < 2)


@triton.jit
def _gdn_graph_norm_quant_kernel(x_ptr, z_ptr, w_ptr, Q, SF_SWZ, SN, stride_z_tok, stride_q, eps,
                                 HV: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
                                 SIGMOID_GATE: tl.constexpr, PADDED_SF_COLS: tl.constexpr):
    """== NQF _nqf_gdn_gated_rmsnorm_quant_kernel on rows [S, N) of core_attn_out (S, N read on device).
    Grid-stride over BT-row tiles (grid.x is bounded, not T-sized): per-row math unchanged."""
    S = tl.load(SN)
    N = tl.load(SN + 1)
    i_h = tl.program_id(1)
    offs_d = tl.arange(0, D)
    w = tl.load(w_ptr + offs_d).to(tl.float32)
    for t0 in range(S + tl.program_id(0) * BT, N, tl.num_programs(0) * BT):
        offs_t = t0 + tl.arange(0, BT)
        mask = (offs_t < N)[:, None]
        row = offs_t.to(tl.int64)[:, None]
        xo = row * (HV * D) + i_h * D + offs_d[None, :]
        x = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + row * stride_z_tok + i_h * D + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / D
        rstd = tl.rsqrt(var + eps)
        y = x * rstd[:, None] * w[None, :]
        if SIGMOID_GATE:
            y = y * tl.sigmoid(z)
        else:
            y = y * (z * tl.sigmoid(z))
        yb = y.to(x_ptr.dtype.element_ty)
        tl.store(x_ptr + xo, yb, mask=mask)
        _mx_epilogue(yb, row, i_h * D + offs_d[None, :], mask, mask, Q, SF_SWZ, stride_q, i_h * (D // 32), BT, D,
                     PADDED_SF_COLS)


@triton.jit(do_not_specialize=["M", "PM"])
def _gdn_graph_quant_rows_kernel(X, Q, SF_SWZ, SN, M, PM, stride_x, stride_q,
                                 NCOL: tl.constexpr, BN: tl.constexpr, XB: tl.constexpr, PADDED_SF_COLS: tl.constexpr):
    """== NQF _nqf_quant_rows_kernel on rows [0, S) u [N, PM) (rows >= M: zero scales, no data)."""
    S = tl.load(SN)
    N = tl.load(SN + 1)
    v = tl.program_id(0).to(tl.int64) * XB + tl.arange(0, XB)[:, None]
    total = S + (PM - N)
    if tl.program_id(0) * XB >= total:
        return
    row = tl.where(v < S, v, N + (v - S))
    in_rng = v < total
    dmask = in_rng & (row < M)
    for c0 in tl.static_range(0, NCOL, BN):
        cols = c0 + tl.arange(0, BN)[None, :]
        yb = tl.load(X + row * stride_x + cols, dmask, other=0.0)
        _mx_epilogue(yb, row, cols, dmask, in_rng, Q, SF_SWZ, stride_q, c0 // 32, XB, BN, PADDED_SF_COLS)


@triton.jit(do_not_specialize=["M", "PM"])
def _gdn_graph_norm_quant_all_kernel(x_ptr, z_ptr, w_ptr, Q, SF_SWZ, SN, M, PM, stride_z_tok, stride_q, eps,
                                     HV: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, XB: tl.constexpr, BN: tl.constexpr,
                                     SIGMOID_GATE: tl.constexpr, PADDED_SF_COLS: tl.constexpr):
    """One launch = _gdn_graph_norm_quant_kernel (gated RMSNorm + MXFP8 of rows [S, N), BT-row x head tiles) followed by
    _gdn_graph_quant_rows_kernel (plain MXFP8 of rows [0, S) u [N, PM), XB-row x NCOL tiles in BN chunks), as a grid-stride loop over
    the concatenated tile space. Disjoint rows / scale blocks; per-row math identical to the two kernels."""
    S = tl.load(SN)
    N = tl.load(SN + 1)
    NCOL: tl.constexpr = HV * D
    n_norm = tl.cdiv(N - S, BT) * HV
    n_q = tl.cdiv(S + (PM - N), XB)  # quant tiles = XB rows x the full NCOL row (same tiling as quant_rows)
    offs_d = tl.arange(0, D)
    w = tl.load(w_ptr + offs_d).to(tl.float32)
    for it in range(tl.program_id(0), n_norm + n_q, tl.num_programs(0)):
        if it < n_norm:
            i_t = it // HV
            i_h = it - i_t * HV
            offs_t = S + i_t * BT + tl.arange(0, BT)
            mask = (offs_t < N)[:, None]
            row = offs_t.to(tl.int64)[:, None]
            xo = row * NCOL + i_h * D + offs_d[None, :]
            x = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)
            z = tl.load(z_ptr + row * stride_z_tok + i_h * D + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            var = tl.sum(x * x, axis=1) / D
            rstd = tl.rsqrt(var + eps)
            y = x * rstd[:, None] * w[None, :]
            if SIGMOID_GATE:
                y = y * tl.sigmoid(z)
            else:
                y = y * (z * tl.sigmoid(z))
            yb = y.to(x_ptr.dtype.element_ty)
            tl.store(x_ptr + xo, yb, mask=mask)
            _mx_epilogue(yb, row, i_h * D + offs_d[None, :], mask, mask, Q, SF_SWZ, stride_q, i_h * (D // 32), BT, D,
                         PADDED_SF_COLS)
        else:
            # distinct names: Triton requires same-typed values for names bound in both branches
            j = it - n_norm
            v = j.to(tl.int64) * XB + tl.arange(0, XB)[:, None]
            total = S + (PM - N)
            q_row = tl.where(v < S, v, N + (v - S))
            in_rng = v < total
            dmask = in_rng & (q_row < M)
            for c0 in tl.static_range(0, NCOL, BN):
                cols = c0 + tl.arange(0, BN)[None, :]
                q_yb = tl.load(x_ptr + q_row * NCOL + cols, dmask, other=0.0)
                _mx_epilogue(q_yb, q_row, cols, dmask, in_rng, Q, SF_SWZ, stride_q, c0 // 32, XB, BN,
                             PADDED_SF_COLS)
# fmt: on


class _GroupBufs:
    def __init__(self, names, layers, dev):
        i32 = dict(dtype=torch.int32, device=dev)
        self.names = names
        self.layers = layers
        self.conv_si = torch.zeros(NR, **i32)
        self.dec_si = torch.zeros(NR, 1, **i32)
        self.nacc = torch.ones(NR, **i32)
        self.cu_s = torch.zeros(NR + 1, **i32)
        self.cu_abs = torch.zeros(NP + 1, **i32)
        # [NP + 1] = valid sequence count (device_nb)
        self.cu_v1_full = torch.zeros(NP + 2, **i32)
        self.cu_v2_full = torch.zeros(NP + 2, **i32)
        self.cu_v1 = self.cu_v1_full[: NP + 1]
        self.cu_v2 = self.cu_v2_full[: NP + 1]
        self.ws = None
        # (layer prefix, T, input/output pointers, ...) -> (CUDAGraph, key, phase)
        self.graphs = {}
        self.ci = torch.zeros(NP, **i32)
        self.hi = torch.ones(NP, dtype=torch.bool, device=dev)
        self.sn = torch.zeros(2, **i32)
        self.warm = False


def _shared(dev, H, HV):
    key = (dev, H, HV)
    s = _SHARED.get(key)
    if s is None:
        bf = dict(dtype=torch.bfloat16, device=dev)
        f32 = dict(dtype=torch.float32, device=dev)
        s = _SHARED[key] = (
            torch.zeros(T_MAX, H, 128, **bf),
            torch.zeros(T_MAX, H, 128, **bf),
            torch.zeros(T_MAX, HV, 128, **bf),
            torch.zeros(T_MAX, HV, **f32),
            torch.zeros(T_MAX, HV, **f32),
        )
    return s


def _vs_mod(mod):
    if DEVNB:
        import vllm.third_party.flashinfer_gdn_vsplit_devnb as devnb

        return devnb
    return mod._GDN_VSPLIT_STATE["mod"]


def _vs_call(mod, gb, q, k, v, g, beta, out, cu, ssm, scale, vsf):
    if DEVNB:
        if gb.ws is None:
            from vllm.third_party.flashinfer_gdn_vsplit_devnb.gdn_chunked_vs import (
                GatedDeltaNetChunkedKernel as _K,
            )

            nsm = torch.cuda.get_device_properties(q.device).multi_processor_count
            gb.ws = torch.empty(
                _K.get_workspace_size(nsm, NP, q.size(1), v.size(1), True),
                dtype=torch.int8,
                device=q.device,
            )
        _vs_mod(mod).chunk_gated_delta_rule_vsplit(
            q,
            k,
            v,
            g,
            beta,
            out,
            cu,
            ssm,
            ssm,
            scale,
            state_indices=gb.ci,
            v_split=vsf,
            device_nb=True,
            workspace=gb.ws,
        )
    else:
        _vs_mod(mod).chunk_gated_delta_rule_vsplit(
            q,
            k,
            v,
            g,
            beta,
            out,
            cu,
            ssm,
            ssm,
            scale,
            state_indices=gb.ci,
            v_split=vsf,
        )


def triton_next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def triton_cdiv(a, b):
    return (a + b - 1) // b


# ----------------------------------------------------------------------------
# metadata builder (per KV-cache group and step)
# ----------------------------------------------------------------------------
def _gb_of(builder):
    mod = gdn_step_plan._gdn()
    key = tuple(builder.layer_names)
    gb = _GB.get(key)
    if gb is None:
        sfc = builder.vllm_config.compilation_config.static_forward_context
        layers = [sfc[n] for n in builder.layer_names]
        if not layers or not all(
            isinstance(L, mod.QwenGatedDeltaNetAttention) for L in layers
        ):
            return None
        gb = _GB[key] = _GroupBufs(key, layers, layers[0].A_log.device)
        for n in key:
            _L2G[n] = gb
    return gb


def _eligible(L, md):
    """Same predicates as the eager path (deferred-commit prologue + zero-copy
    mixed path) plus the graphs' static bounds; None when eligible.
    """
    mod = gdn_step_plan._gdn()
    if not isinstance(md, mod.GDNAttentionMetadata) or md.num_prefills <= 0:
        return "no_prefill"
    if L._can_use_fused_gdn_mtp_decode(md) and md.num_prefills == 0:
        return "decode_only"
    if not L._can_use_mixed_fastpath(md):
        return "not_mixed_fastpath"
    if (
        mod._GDN_VERIFY_LEFT[0] > 0
        or mod._GDN_MIXED_SPEC_TRITON
        or not mod._GDN_FUSED_CONV
        or not getattr(mod, "_GDN_CONV_CUDA", False)
        or not mod._GDN_FI_STATE_POOL
    ):
        return "config"
    if not mod._GDN_CONV_CUDA_MOD or mod._GDN_CONV_CUDA_MOD[0] is None:
        return "conv_not_loaded"
    if (
        not mod._GDN_FI_VSPLIT
        or mod._GDN_FI_VSPLIT_CHECK
        or mod._GDN_VSPLIT_STATE.get("disabled")
        or "mod" not in mod._GDN_VSPLIT_STATE
    ):
        return "vsplit_not_ready"
    if md.num_spec_decodes > NR or md.num_prefills > NP:
        return "bounds"
    if md.num_actual_tokens > T_MAX:
        return "tokens"
    si = md.spec_state_indices_tensor
    if md.num_spec_decodes > 0 and (si is None or si.size(1) != MTPW):
        return "spec_width"
    n = md.prefill_query_start_loc.numel() - 1
    if n != md.num_prefills or md.non_spec_state_indices_tensor.numel() < n:
        return "prefill_md"
    if mod._gdn_fi_want_cp(
        n, md.num_prefill_tokens, getattr(md, "prefill_max_seqlen", 0)
    ):
        return "cp"
    return None


def _ensure_modules():
    """Load the CUDA conv and V-split modules the way the eager path does on
    its first call (no numerics).
    """
    mod = gdn_step_plan._gdn()
    if not mod._GDN_CONV_CUDA_MOD:
        from vllm.model_executor.layers.mamba.ops import gdn_conv_cuda as _gcc

        _gcc.load()
        mod._GDN_CONV_CUDA_MOD.append(_gcc)
    st = mod._GDN_VSPLIT_STATE
    if "mod" not in st and not st.get("disabled"):
        import vllm.third_party.flashinfer_gdn_vsplit as gdn_vsplit

        st["mod"] = gdn_vsplit
        st.setdefault("calls", 0)


def after_build(builder, md) -> None:
    """End of GDNAttentionMetadataBuilder.build: pack the step for the graphs."""
    try:
        _after_build(builder, md)
    except Exception as e:  # noqa: BLE001 - never leave a half-packed step on the graph path
        md.__dict__.pop(_GRAPH, None)
        STATS["graph_build_errors"] = STATS.get("graph_build_errors", 0) + 1
        if STATS["graph_build_errors"] <= 3:
            logger.warning("GDN layer graphs: build error -> eager: %r", e)


def _after_build(builder, md):
    if _OFF[0]:
        return
    mod = gdn_step_plan._gdn()
    gsc = gdn_step_plan._gsc()
    if not isinstance(md, mod.GDNAttentionMetadata):
        return
    gb = _gb_of(builder)
    if gb is None:
        return
    L = gb.layers[0]
    kv = getattr(L, "kv_cache", None)
    if not kv or not isinstance(kv[1], torch.Tensor) or kv[1].dim() != 4:
        return
    if not gb.warm:
        _ensure_modules()
        gsc._layer_check(L)
        _warm(gb, L)
    if md.num_prefills <= 0:
        # decode-only: FULL graph / eager; a PIECEWISE step without the flag
        # runs eagerly
        return
    why = _eligible(L, md)
    STATS["graph_steps"] = STATS.get("graph_steps", 0) + 1
    if STATS["graph_steps"] % 3000 == 1:
        logger.info("GDN layer graphs stats: %s", {k: v for k, v in STATS.items()})
    if why is not None:
        d = STATS.setdefault("graph_ineligible", {})
        d[why] = d.get(why, 0) + 1
        return
    S, P = md.num_spec_decode_tokens, md.num_prefill_tokens
    N = S + P
    nr, npf = md.num_spec_decodes, md.num_prefills
    HV = L.num_v_heads // L.tp_size
    vs = mod._GDN_VSPLIT_STATE["mod"]
    vsf = (
        2
        if V2ONLY
        else vs.choose_vsplit(npf, P, int(getattr(md, "prefill_max_seqlen", 0)), hv=HV)
    )
    dummy = gb.cu_s
    si = md.spec_state_indices_tensor if nr > 0 else dummy
    slots = (
        md.non_spec_state_indices_tensor.contiguous()
    )  # may be a strided block_table column
    hi = md.has_initial_state.contiguous()
    ns_cu = md.non_spec_query_start_loc.contiguous()
    _gdn_graph_pack_kernel[(1,)](
        si,
        si.stride(0) if nr > 0 else 0,
        md.spec_query_start_loc.contiguous() if nr > 0 else dummy,
        md.num_accepted_tokens.contiguous() if nr > 0 else dummy,
        ns_cu,
        slots,
        hi,
        gb.conv_si,
        gb.dec_si,
        gb.nacc,
        gb.cu_s,
        gb.cu_abs,
        gb.cu_v1_full,
        gb.cu_v2_full,
        gb.ci,
        gb.hi,
        gb.sn,
        S,
        N,
        nr,
        npf,
        int(vsf == 2),
        NRC=NR,
        NPC=NP,
        BLOCK=max(triton_next_pow2(NR + 1), triton_next_pow2(NP + 1)),
        num_warps=4,
    )
    # hoisted per-step state work (same as the step plan, now before the forward)
    key = (gb.names, tuple(Lx.kv_cache[1].data_ptr() for Lx in gb.layers))
    gt = _TABLES.get(key)
    if gt is None:
        gt = _TABLES[key] = _GroupTable(gb.layers, gsc)
    gdn_step_plan._materialize_rows(md, gb.layers, gt)
    pslots = md.prefill_state_indices.contiguous()
    phi = md.prefill_has_initial_state.contiguous()
    if pslots.numel() > 0:
        _gdn_zero_state_slots_layers_kernel[(pslots.numel(), gt.hv, gt.n)](
            gt.ptrs,
            pslots,
            phi,
            gt.slot_stride,
            HEAD_ELEMS=gt.head_elems,
            BLOCK=4096,
            BF16=gt.dtype == torch.bfloat16,
            num_warps=4,
        )
    md.__dict__[_ZEROED] = True
    md.__dict__[_VSF] = vsf
    md.__dict__[_GRAPH] = True
    STATS["graph_packed"] = STATS.get("graph_packed", 0) + 1
    STATS[f"graph_vsf{vsf}"] = STATS.get(f"graph_vsf{vsf}", 0) + 1


def _warm(gb, L):
    """Compile both V-split variants for the pool's exact key before any capture
    (no work: empty cu_seqlens).
    """
    mod = gdn_step_plan._gdn()
    ssm = L.kv_cache[1]
    H, HV = L.num_k_heads // L.tp_size, L.num_v_heads // L.tp_size
    q, k, v, g, beta = _shared(ssm.device, H, HV)
    out = torch.zeros(8, HV, 128, dtype=torch.bfloat16, device=ssm.device)
    for vsf in (1, 2):
        # gb buffers are all zero here: empty cu and (device_nb) zero valid
        # sequences -> no work
        _vs_call(
            mod,
            gb,
            q[:8],
            k[:8],
            v[:8],
            g[:8],
            beta[:8],
            out,
            gb.cu_v1 if vsf == 1 else gb.cu_v2,
            ssm,
            128**-0.5,
            vsf,
        )
    Tw = 32
    qkvz = torch.zeros(
        Tw,
        (2 * L.key_dim + 2 * L.value_dim) // L.tp_size,
        dtype=torch.bfloat16,
        device=ssm.device,
    )
    ba = torch.zeros(Tw, 2 * HV, dtype=torch.bfloat16, device=ssm.device)
    cao = torch.zeros(Tw, HV, 128, dtype=torch.bfloat16, device=ssm.device)
    _graph_core(L, gb, qkvz, ba, cao, stash=False)
    torch.cuda.synchronize()
    gb.warm = True
    logger.info(
        "GDN layer graphs: group warmed: %d layers, NR=%d NP=%d T_MAX=%d",
        len(gb.layers),
        NR,
        NP,
        T_MAX,
    )


# ----------------------------------------------------------------------------
# model runner: forced eager steps, CUDA-graph memory profiling
# ----------------------------------------------------------------------------
def forward_context_mode(attn_metadata, mode):
    """cudagraph_runtime_mode for set_forward_context: a PIECEWISE step whose
    GDN metadata was not packed for the graphs runs with NONE (eager).
    """
    from vllm.config import CUDAGraphMode

    if MODE != "piece" and KEEP_PW and not FORCE_EAGER:
        # layer mode: unpacked steps run the eager GDN core between the
        # PIECEWISE graphs (the GDN op is still a splitting op)
        return mode
    if mode == CUDAGraphMode.PIECEWISE and isinstance(attn_metadata, dict):
        GDNMD = gdn_step_plan._gdn().GDNAttentionMetadata
        seen = set()
        for m in attn_metadata.values():
            if id(m) in seen:
                continue
            seen.add(id(m))
            if isinstance(m, GDNMD) and (FORCE_EAGER or not m.__dict__.get(_GRAPH)):
                STATS["graph_forced_eager"] = STATS.get("graph_forced_eager", 0) + 1
                return CUDAGraphMode.NONE
        STATS["graph_piecewise"] = STATS.get("graph_piecewise", 0) + 1
    return mode


def capture_model(impl):
    """GPUModelRunner.capture_model: marks vLLM's capture phase (profiling and
    real), the only time layer graphs of unpacked (dummy) steps are captured
    when KEEP_PIECEWISE=1.
    """
    _IN_CAPTURE[0] = True
    try:
        return impl()
    finally:
        _IN_CAPTURE[0] = False
        _qo_capture_stats()


def _qo_capture_stats() -> None:
    """gb300-fuse engage proof (GLUE_GSC_QO): after vLLM's capture phase, how many GDN decode calls took the fused
    MXFP8 quant (decode_qo), how many FULL-graph pad-row zeroings were deferred to the decode kernel, skipped (the
    QO kernel zeroed them) or ran late (QO not taken, e.g. > GLUE_GSC_QO_MAXT rows). Per layer = / 30 GDN layers."""
    if os.environ.get("GLUE_GSC_QO", "0") != "1":
        return
    try:
        from vllm.model_executor.layers.fusion import norm_quant as _nq
        from vllm.model_executor.layers.mamba.gdn import gdn_out_alloc as _ga
        from vllm.model_executor.layers.mamba.ops import gdn_state_commit as _gsc

        logger.info(
            "[fuse] QO capture stats: gsc decode_calls=%s decode_qo_calls=%s | out_alloc deferred=%s skipped=%s "
            "late_zero=%s | norm_quant gdn_qo=%s",
            _gsc.STATS.get("decode_calls"), _gsc.STATS.get("decode_qo_calls", 0), _ga.STATS.get("deferred"),
            _ga.STATS.get("skipped"), _ga.STATS.get("late_zero"), _nq.STATS.get("gdn_qo", 0),
        )
    except Exception as ex:  # noqa: BLE001 (diagnostic only)
        logger.warning("[fuse] QO capture stats unavailable: %s", ex)


def _drop_graphs(reason):
    torch.cuda.synchronize()
    n = 0
    for gb in {id(g): g for g in _GB.values()}.values():
        n += len(gb.graphs)
        gb.graphs.clear()
        gb.ws = None  # allocated inside a profiling-phase capture (old private pool)
    _SHARED.clear()  # same
    # fresh capture stream + private pool (never re-incref a pool that drained to 0)
    _CAP[0] = _CAP[1] = None
    torch.cuda.synchronize()
    STATS["graphs_dropped"] = STATS.get("graphs_dropped", 0) + n
    logger.info("GDN layer graphs: dropped %d layer graphs (%s)", n, reason)


def profile_cudagraph_memory(impl):
    """VLLM's CUDA-graph memory profiling captures every graph once against a
    throwaway minimal KV cache and throwaway buffers. Layer graphs recorded then
    bake in the throwaway GDN state / buffer addresses: drop them afterwards
    (VLLM_GDN_LAYER_GRAPHS_FIX_PROFILE=1).
    """
    _PHASE[0] = "prof"
    try:
        return impl()
    finally:
        _PHASE[0] = "real"
        if FIX_PROFILE:
            _drop_graphs("after vLLM CUDA-graph memory profiling")
        else:
            logger.info(
                "GDN layer graphs: FIX_PROFILE=0: keeping %d profiling-phase layer "
                "graphs (diagnostic)",
                sum(len(g.graphs) for g in _GB.values()),
            )


# ----------------------------------------------------------------------------
# the capturable core
# ----------------------------------------------------------------------------
def forward_packed(layer, mixed_qkvz, ba, core_attn_out) -> bool:
    """Start of QwenGatedDeltaNetAttention._forward_core_fused_norm_packed.
    Returns True when the call was served by a layer graph (captured or
    replayed); False: the caller runs the regular path.
    """
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    if MODE == "piece":
        if torch.cuda.is_current_stream_capturing():
            fc = get_forward_context()
            if fc.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                gb = _L2G.get(layer.prefix)
                if gb is None or not gb.warm:
                    raise RuntimeError(
                        f"GDN layer graphs: PIECEWISE capture of {layer.prefix} before "
                        "its graph buffers were set up (the eager path must never be "
                        "captured)"
                    )
                _graph_core(layer, gb, mixed_qkvz, ba, core_attn_out)
                return True
        return False
    # ---- layer mode: this op is still a splitting op (runs between vLLM's
    # piecewise graphs) ----
    fc = get_forward_context()
    if (
        _BYPASS[0]
        or _OFF[0]
        or getattr(fc, "cudagraph_runtime_mode", None) != CUDAGraphMode.PIECEWISE
        or torch.cuda.is_current_stream_capturing()
    ):
        return False
    gb = _L2G.get(layer.prefix)
    raw = fc.attn_metadata
    md = raw.get(layer.prefix) if isinstance(raw, dict) else None
    if gb is None or not gb.warm or md is None:
        return False
    T = core_attn_out.size(0)
    if T > T_MAX:
        # MNBT > T_MAX (GB300 recipe: MNBT 32768): no graph for this size; the
        # step is never packed (_eligible "tokens")
        return False
    if KEEP_PW and not md.__dict__.get(_GRAPH) and not _IN_CAPTURE[0]:
        # unpacked step while serving: eager step-plan path, no capture
        STATS["graph_unpacked_pw"] = STATS.get("graph_unpacked_pw", 0) + 1
        return False
    # keyed by the input/output addresses too: vLLM may hold more than one
    # piecewise graph per size, each with its own static buffers
    vsel = (md.__dict__.get(_VSF, 2) if V2ONLY is False else 2) if VSF_KEY else None
    key = (
        layer.prefix,
        T,
        mixed_qkvz.data_ptr(),
        ba.data_ptr(),
        core_attn_out.data_ptr(),
        mixed_qkvz.stride(0),
        ba.stride(0),
    )
    if FIX_PROFILE:
        key = key + (layer.kv_cache[0].data_ptr(), layer.kv_cache[1].data_ptr())
    key = key + (vsel,)
    ent = gb.graphs.get(key)
    if ent is None:
        ent = gb.graphs[key] = (
            _capture(layer, gb, mixed_qkvz, ba, core_attn_out, vsel),
            key,
            _PHASE[0],
        )
        STATS["layer_graphs_captured"] = STATS.get("layer_graphs_captured", 0) + 1
        k_ph = "layer_graphs_captured_" + _PHASE[0]
        STATS[k_ph] = STATS.get(k_ph, 0) + 1
        if VSF_KEY and not md.__dict__.get(_GRAPH):
            # vLLM's capture phase: also capture the other V-split variant now
            # (no capture while serving)
            for other in (1, 2):
                k2 = key[:-1] + (other,)
                if k2 not in gb.graphs:
                    gb.graphs[k2] = (
                        _capture(layer, gb, mixed_qkvz, ba, core_attn_out, other),
                        k2,
                        _PHASE[0],
                    )
                    STATS["layer_graphs_captured"] += 1
    elif ent[2] == "prof" and _PHASE[0] == "real":
        n = STATS["stale_graph_hits"] = STATS.get("stale_graph_hits", 0) + 1
        if n <= 5 or n in (100, 1000, 10000):
            logger.warning(
                "GDN layer graphs: STALE profiling-phase graph hit #%d: %s T=%d "
                "md_packed=%s",
                n,
                layer.prefix,
                T,
                bool(md.__dict__.get(_GRAPH)),
            )
        elif md.__dict__.get(_GRAPH):
            STATS["layer_graphs_captured_serving"] = (
                STATS.get("layer_graphs_captured_serving", 0) + 1
            )
    if not md.__dict__.get(_GRAPH):
        # vLLM's own capture phase (dummy metadata) or a step that was not packed:
        # the regular path
        return False
    if _CHECK[0] > 0:
        _CHECK[0] -= 1
        _check(layer, md, mixed_qkvz, ba, core_attn_out, ent[0])
    else:
        ent[0].replay()
    STATS["layer_replays"] = STATS.get("layer_replays", 0) + 1
    return True


def _pages_u8(layer):
    """uint8 [n_slots, page] view of the layer's GDN state pages (conv | ssm |
    deferred-commit log), or None.
    """
    conv, ssm = layer.kv_cache[0], layer.kv_cache[1]
    page = ssm.stride(0) * ssm.element_size()
    if conv.stride(0) * conv.element_size() != page:
        return None
    st = ssm.untyped_storage()
    off = min(conv.data_ptr(), ssm.data_ptr()) - st.data_ptr()
    n = min(ssm.size(0), (st.nbytes() - off) // page)
    v = torch.empty(0, dtype=torch.uint8, device=ssm.device)
    v.set_(st, off, (n, page), (page, 1))
    return v


def _check(layer, md, mixed_qkvz, ba, core_attn_out, graph):
    """VLLM_GDN_LAYER_GRAPHS_CHECK: graph replay vs the eager path, bitwise."""
    from vllm.model_executor.layers.fusion import norm_quant

    T = core_attn_out.size(0)
    HV = layer.num_v_heads // layer.tp_size
    qb, sf, psc = norm_quant._gdn_bufs(HV * layer.head_v_dim, core_attn_out.device)
    pm = (T + 127) // 128 * 128
    nr, npf = md.num_spec_decodes, md.num_prefills
    parts = []
    if nr > 0:
        parts.append(md.spec_state_indices_tensor[:nr, 0].long())
    parts.append(md.prefill_state_indices[:npf].long())
    slots = torch.unique(torch.cat(parts))
    pages = _pages_u8(layer)
    bufs = [mixed_qkvz, ba, core_attn_out, qb[:T], sf[: pm * psc]]
    pre = [t.clone() for t in bufs]
    pre_pages = pages[slots].clone() if pages is not None else None
    graph.replay()
    got = [t.clone() for t in bufs]
    got_pages = pages[slots].clone() if pages is not None else None
    for t, x in zip(bufs, pre):
        t.copy_(x)
    if pages is not None:
        pages[slots] = pre_pages
    _BYPASS[0] = True
    try:
        layer._forward_core_fused_norm_packed(mixed_qkvz, ba, core_attn_out)
    finally:
        _BYPASS[0] = False
    bad = []
    for name, t, g in zip(("qkvz", "ba", "out", "q", "sf"), bufs, got):
        a8 = t.contiguous().view(torch.uint8)
        b8 = g.contiguous().view(torch.uint8)
        if not torch.equal(a8, b8):
            bad.append(f"{name}:{int((a8 != b8).sum())}")
    if pages is not None:
        now = pages[slots]
        if not torch.equal(now, got_pages):
            rows = (now != got_pages).any(dim=1).nonzero().flatten()[:4].tolist()
            bad.append(f"pages:rows{rows}")
    n = STATS["graph_check"] = STATS.get("graph_check", 0) + 1
    if bad:
        STATS["graph_check_fail"] = STATS.get("graph_check_fail", 0) + 1
        _OFF[0] = True
        logger.warning(
            "GDN layer graphs: CHECK MISMATCH %s T=%d S=%d P=%d nseq=%d %s; keeping "
            "the eager result, layer graphs disabled",
            layer.prefix,
            T,
            md.num_spec_decode_tokens,
            md.num_prefill_tokens,
            npf,
            bad,
        )
    elif n in (1, 10, 100, 1000) or _CHECK[0] == 0:
        logger.info(
            "GDN layer graphs: check ok #%d (%s T=%d S=%d P=%d nseq=%d)",
            n,
            layer.prefix,
            T,
            md.num_spec_decode_tokens,
            md.num_prefill_tokens,
            npf,
        )


def _capture(layer, gb, mixed_qkvz, ba, core_attn_out, only_vsf=None):
    g = torch.cuda.CUDAGraph()
    s = _CAP[0]
    if s is None:
        s = _CAP[0] = torch.cuda.Stream(device=core_attn_out.device)
        _CAP[1] = torch.cuda.graph_pool_handle()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g.capture_begin(pool=_CAP[1], capture_error_mode="thread_local")
        try:
            _graph_core(
                layer, gb, mixed_qkvz, ba, core_attn_out, stash=False, only_vsf=only_vsf
            )
        finally:
            g.capture_end()
    torch.cuda.current_stream().wait_stream(s)
    return g


def _graph_core(layer, gb, mixed_qkvz, ba, core_attn_out, stash=True, only_vsf=None):
    from vllm.model_executor.layers.fusion import norm_quant

    mod = gdn_step_plan._gdn()
    gsc = gdn_step_plan._gsc()
    T = core_attn_out.size(0)
    assert (
        T <= T_MAX
        and core_attn_out.is_contiguous()
        and core_attn_out.dtype == torch.bfloat16
    )
    H, HV = layer.num_k_heads // layer.tp_size, layer.num_v_heads // layer.tp_size
    qkv_size = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
    mixed_qkv, gate_flat = mixed_qkvz.split(
        [qkv_size, layer.value_dim // layer.tp_size], dim=-1
    )
    output_gate = gate_flat.reshape(gate_flat.size(0), -1, layer.head_v_dim)
    b, a = layer.split_ba(ba)
    ssm = layer.kv_cache[1]
    conv_state = (
        layer.kv_cache[0]
        if mod.is_conv_state_dim_first()
        else layer.kv_cache[0].transpose(-1, -2)
    )
    conv_w = layer.conv1d.weight.view(
        layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
    )
    q, k, v, g, beta = (t[:T] for t in _shared(ssm.device, H, HV))

    def spec_part():
        # spec rows [0, S): conv update (in place) + deferred-commit decode (fused
        # gated RMSNorm), NR padded requests
        mq = mod.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_w,
            layer.conv1d.bias,
            layer.activation,
            conv_state_indices=gb.conv_si,
            num_accepted_tokens=gb.nacc,
            query_start_loc=gb.cu_s,
            max_query_len=MTPW,
            validate_data=False,
        )
        gsc.load().decode(
            mq,
            a,
            b,
            layer.A_log,
            layer.dt_bias,
            gb.dec_si,
            gb.cu_s,
            gb.nacc,
            ssm,
            output_gate,
            layer.norm.weight,
            core_attn_out,
            float(layer.head_k_dim**-0.5),
            float(layer.layer_norm_epsilon),
            layer.norm.activation == "sigmoid",
        )

    def conv_part():
        # prefill rows [S, N): conv + post-conv into absolute rows of the shared buffers
        gcc = mod._GDN_CONV_CUDA_MOD[0]
        ok = gcc.load().run(
            mixed_qkv,
            conv_w,
            conv_state,
            gb.ci,
            gb.hi,
            gb.cu_abs,
            NP,
            a,
            b,
            layer.A_log,
            layer.dt_bias,
            q,
            k,
            v,
            g,
            beta,
            int(H),
            int(TPH),
            int(gcc.RV),
        )
        assert ok, "GDN layer graphs: CUDA conv refused the graph-mode contract"

    def chunk_part():
        variants = ((2, gb.cu_v2),) if V2ONLY else ((1, gb.cu_v1), (2, gb.cu_v2))
        if only_vsf is not None:
            variants = tuple(x for x in variants if x[0] == only_vsf)
        for vsf, cu in variants:
            _vs_call(
                mod,
                gb,
                q,
                k,
                v,
                g,
                beta,
                core_attn_out,
                cu,
                ssm,
                layer.head_k_dim**-0.5,
                vsf,
            )

    if SPEC_OVERLAP and torch.cuda.is_current_stream_capturing():
        # gdnp: spec part on a side capture stream (disjoint rows / slots).
        # mode 1: forked at the layer start (concurrent with conv + chunk);
        # mode 2: forked after the prefill conv (concurrent with the chunk kernel,
        #         which is then dispatched before the decode fills the SMs).
        main = torch.cuda.current_stream()
        side = _SIDE.get(main.device)
        if side is None:
            side = _SIDE[main.device] = torch.cuda.Stream(device=main.device)
        if SPEC_OVERLAP_MODE == 2:
            conv_part()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            spec_part()
        if SPEC_OVERLAP_MODE != 2:
            conv_part()
        chunk_part()
        # join: the norm-quant kernel reads rows [0, S) written by the decode
        main.wait_stream(side)
    else:
        spec_part()
        conv_part()
        chunk_part()
    # gated RMSNorm + MXFP8 quant (norm-quant fusion math) with device-side row
    # ranges, then the stash for out_proj
    K_ = HV * layer.head_v_dim
    qb, sf, psc = norm_quant._gdn_bufs(K_, core_attn_out.device)
    assert qb.size(0) >= T
    BT = 16
    pm = (T + 127) // 128 * 128
    x2 = core_attn_out.view(T, K_)
    if MERGED_NQ and T <= MERGED_NQ_MAX_T:
        _gdn_graph_norm_quant_all_kernel[(NQ_ALL_PROGS,)](
            core_attn_out,
            output_gate,
            layer.norm.weight,
            qb,
            sf,
            gb.sn,
            T,
            pm,
            output_gate.stride(0),
            qb.stride(0),
            layer.layer_norm_epsilon,
            HV=HV,
            D=layer.head_v_dim,
            BT=BT,
            XB=4,
            BN=512,
            SIGMOID_GATE=(layer.norm.activation == "sigmoid"),
            PADDED_SF_COLS=psc,
            num_warps=4,
        )
    else:
        _gdn_graph_norm_quant_kernel[(min(triton_cdiv(T, BT), NQ_TILES), HV)](
            core_attn_out,
            output_gate,
            layer.norm.weight,
            qb,
            sf,
            gb.sn,
            output_gate.stride(0),
            qb.stride(0),
            layer.layer_norm_epsilon,
            HV=HV,
            D=layer.head_v_dim,
            BT=BT,
            SIGMOID_GATE=(layer.norm.activation == "sigmoid"),
            PADDED_SF_COLS=psc,
            num_warps=4,
        )
        XB, BN, nw = 4, 512, 2
        _gdn_graph_quant_rows_kernel[(triton_cdiv(pm, XB),)](
            x2,
            qb,
            sf,
            gb.sn,
            T,
            pm,
            x2.stride(0),
            qb.stride(0),
            NCOL=K_,
            BN=BN,
            XB=XB,
            PADDED_SF_COLS=psc,
            num_warps=nw,
        )
    if not stash:
        return
    norm_quant._stash_set(x2, qb[:T], sf[: pm * psc], None)
    STATS["graph_layers_captured"] = STATS.get("graph_layers_captured", 0) + 1
