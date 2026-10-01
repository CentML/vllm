# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred ("single-state") GDN MTP state commit.

Enabled with GDN_STATE_COMMIT=1 (read in every process; default off). With MTP k speculative
tokens the stock GDN decode op writes the recurrent state after every draft token into its own
slot (1 + k slots per request), although only the accepted one is ever read again. Here every
request keeps ONE state slot per GDN KV-cache group; its page additionally holds a small token log
(raw bf16 k/v/a/b of the last step's <= MAX_T tokens plus a per-key-head flag). Each decode step
replays the accepted prefix of the previous step from the log, commits the state once, runs the
new tokens with the state in registers (outputs only) and logs them. ``materialize`` commits
pending logs in place for every other reader of GDN state.

Integration points (direct, gated edits; all are no-ops unless GDN_STATE_COMMIT=1):

  model_executor/layers/mamba/mamba_utils.py
    MambaStateShapeCalculator.gated_delta_net_state_shape / MambaStateDtypeCalculator.
      gated_delta_net_state_dtype -> append the in-page token log state (bf16 [log_bytes // 2]).
    MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func -> (conv,) only: the stock align
      copy kernels move conv state only; GDN ssm state + log are moved by `materialize`.
  model_executor/layers/mamba/abstract.py
    MambaBase.get_kv_cache_spec -> adjust_kv_cache_spec(): GDN MambaSpec with
      num_speculative_blocks = 0 (1 state block per running request and GDN group).
  v1/attention/backends/gdn_attn.py
    GDNAttentionMetadataBuilder.build -> postprocess_metadata(): spec_state_indices broadcast to
      width 1 + num_spec (all columns = the single base slot) and `gsc_non_spec_num_accepted`
      for non-spec readers.
  _custom_ops.fused_gdn_decode_post_conv_mtp -> decode() (decode-only CUDA-graph path and the
      mixed-batch spec path of the Qwen GDN layer).
  model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
    QwenGatedDeltaNetAttention.get_state_dtype -> (conv, ssm) only for layer-level consumers.
    QwenGatedDeltaNetAttention._forward_core_fused_norm / _forward_core -> fused_norm_prologue()
      / forward_core_prologue(): before any non-spec (prefill / plain decode) read of GDN state,
      materialize pending logs of those slots in place.
  v1/worker/mamba_utils.py
    MambaSpecDecodeGPUContext.initialize_from_forward_context -> _build_worker_table();
    run_fused_precopy / run_fused_postprocess / run_fused_postprocess_align -> worker_*():
      after the stock (conv-only) copies, materialize ssm state for align-mode block migration
      and block-boundary checkpoints.

GDN_STATE_COMMIT_LAYOUT_ONLY=1 is a control mode: same page layout and block size, but stock
speculative blocks, stock copies and the stock decode kernel.

Kernel: gdn_state_commit.cu (next to this file), JIT-built with torch.utils.cpp_extension for the
device arch into $GDN_STATE_COMMIT_BUILD_DIR (default ~/.cache/gdn_state_commit); the build is
cached there (ninja + file lock). Requirements: mamba_cache_mode="align", head dims 128,
fp32 or bf16 SSM state, 1 + num_speculative_tokens <= MAX_T, int32 state indices / cu_seqlens /
accepted-token counts, and the CUDA mixed-batch spec path (VLLM_GDN_MIXED_SPEC_TRITON=0).
"""

import copy
import dataclasses
import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn import gdn_step_plan

logger = init_logger(__name__)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None
_exts = {}
MAX_T = 4
_DEBUG = os.environ.get("GDN_STATE_COMMIT_DEBUG", "0") == "1"


def _dbg(what):
    if _DEBUG and not torch.cuda.is_current_stream_capturing():
        try:
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"[gdn_state_commit] CUDA error right after {what}") from e


STATS = {"decode_calls": 0, "materialize_calls": 0, "precopy_calls": 0, "postprocess_calls": 0}


def layout_only() -> bool:
    """GDN_STATE_COMMIT_LAYOUT_ONLY=1: control arm - same page layout / block size as the deferred
    commit, but stock 1+k speculative blocks, stock copies and the stock kernel."""
    return os.environ.get("GDN_STATE_COMMIT_LAYOUT_ONLY", "0") == "1"


def enabled() -> bool:
    return os.environ.get("GDN_STATE_COMMIT", "0") == "1"


def decode_routed() -> bool:
    """True when vllm._custom_ops.fused_gdn_decode_post_conv_mtp dispatches to decode()
    (GDN_STATE_COMMIT=1 and not the layout-only control mode)."""
    from vllm import _custom_ops

    return bool(_custom_ops._GDN_STATE_COMMIT_DECODE)


def log_bytes(H: int, HV: int) -> int:
    # K [4][H][128] bf16 + V [4][HV][128] bf16 + AB [4][HV][2] bf16 + L [HV] int32 + ctr [H] int32
    return MAX_T * H * 128 * 2 + MAX_T * HV * 128 * 2 + MAX_T * HV * 2 * 2 + HV * 4 + H * 4


# ------------------------------------------------------------------------------
# load-time guard: tokens per spec-decode row (1 + num_speculative_tokens) <= MAX_T
# ------------------------------------------------------------------------------
# The decode kernels log at most kMaxT (= MAX_T, asserted against the compiled
# kernel in load()) tokens per spec-decode row; a longer row takes an early-exit
# branch that writes zeros as the GDN output without any error. So with
# GDN_STATE_COMMIT=1 the engine refuses to start when num_speculative_tokens + 1
# > MAX_T: in EngineCore.__init__ (before the model loads) and in
# GDNAttentionMetadataBuilder.__init__ (on the width the metadata post-step and
# the kernel see). The layout-only control mode runs the stock kernels and is not
# limited. GDN_STATE_COMMIT_GUARD=0 disables the check (default 1).
_GUARD_SEEN: set = set()


def guard_enabled() -> bool:
    return enabled() and os.environ.get("GDN_STATE_COMMIT_GUARD", "1") != "0"


def check_num_speculative_tokens(num_spec, site: str) -> int:
    """Raise RuntimeError if 1 + num_spec tokens per spec-decode row exceed MAX_T.

    Returns the row width. Logs the accepted width once per (site, width).
    """
    k = int(num_spec or 0)
    t = k + 1
    if layout_only():
        if (site, "layout_only") not in _GUARD_SEEN:
            _GUARD_SEEN.add((site, "layout_only"))
            logger.info(
                "gdn_state_commit guard: OK (%s): GDN_STATE_COMMIT_LAYOUT_ONLY=1 "
                "(stock kernels), num_speculative_tokens=%d not limited",
                site,
                k,
            )
        return t
    if t > MAX_T:
        msg = (
            f"refusing to start ({site}): num_speculative_tokens={k} -> {t} "
            f"tokens per spec row > gdn_state_commit MAX_T={MAX_T}. With "
            "GDN_STATE_COMMIT=1 the deferred-commit decode kernel would write "
            "zero GDN output for such rows. Use num_speculative_tokens <= "
            f"{MAX_T - 1}, disable GDN_STATE_COMMIT, or use a kernel build with "
            f"MAX_T >= {t}."
        )
        logger.error("gdn_state_commit guard: %s", msg)
        raise RuntimeError("gdn_state_commit guard: " + msg)
    if (site, t) not in _GUARD_SEEN:
        _GUARD_SEEN.add((site, t))
        logger.info(
            "gdn_state_commit guard: OK (%s): num_speculative_tokens=%d -> %d "
            "tokens/row <= MAX_T=%d",
            site,
            k,
            t,
            MAX_T,
        )
    return t


def load(minb=None, gb=None):
    """minb: CTAs/SM launch bound of the decode kernel (env GDN_STATE_COMMIT_MINB, default 2).
    gb: override GDN_STATE_COMMIT_GB (the in-serving check loads the CK0 reference with gb=0)."""
    global _ext
    if minb is None:
        minb = int(os.environ.get("GDN_STATE_COMMIT_MINB", "2"))
        if _ext is not None and gb is None:
            return _ext
    key = minb if gb is None else (minb, int(gb))
    if key in _exts:
        return _exts[key]
    import torch.utils.cpp_extension as cpp

    build = os.environ.get(
        "GDN_STATE_COMMIT_BUILD_DIR", os.path.expanduser("~/.cache/gdn_state_commit")
    )
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    nc = int(os.environ.get("GDN_STATE_COMMIT_NC", "1"))
    f2 = int(os.environ.get("GDN_STATE_COMMIT_F2", "0"))
    ps = int(os.environ.get("GDN_STATE_COMMIT_PERSIST", "0"))
    pr = int(os.environ.get("GDN_STATE_COMMIT_PROBE", "0"))
    fx = int(os.environ.get("GDN_STATE_COMMIT_FAST", "0"))  # row-per-8-lanes kernel (float-order)
    ea = int(os.environ.get("GDN_STATE_COMMIT_EARLY", "0"))  # early new-token loads (bit-exact)
    ck = int(os.environ.get("GDN_STATE_COMMIT_CK", "0"))  # chunked (WY) replay/new-token math
    pdl = int(os.environ.get("GDN_STATE_COMMIT_PDL", "0"))  # programmatic dependent launch
    upd = int(os.environ.get("GDN_STATE_COMMIT_CK2_UPD", "1"))  # 1: CUDA-core commit update (default)
    c3f2 = int(os.environ.get("GDN_STATE_COMMIT_CK3_F2", "1"))  # CK=3: packed f32x2 commit
    c3m = int(os.environ.get("GDN_STATE_COMMIT_CK3_MAP", "0"))  # CK=3: commit lane map
    c3s = int(os.environ.get("GDN_STATE_COMMIT_CK3_STORE", "0"))  # CK=3: state store path
    c3p = int(os.environ.get("GDN_STATE_COMMIT_CK3_PF", "0"))  # CK=3: L2 prefetch distance (blocks)
    c3o = int(os.environ.get("GDN_STATE_COMMIT_CK3_ORDER", "0"))  # CK=3: token loads before state
    c3ps = int(os.environ.get("GDN_STATE_COMMIT_CK3_PFS", "1"))  # CK=3: prefetch includes the state
    # [rubin-ck] CK=3: commit lane remap (k of the lane's chunk in registers; bitwise equal to CK=3)
    c3r = int(os.environ.get("GDN_STATE_COMMIT_CK3_REMAP", "0"))
    tag = f"b{minb}_nc{nc}_f{f2}_p{ps}_r{pr}_x{fx}_e{ea}_ck{ck}_pdl{pdl}" + (f"_u{upd}" if upd != 1 else "") + (
        f"_c3f{c3f2}m{c3m}s{c3s}p{c3p}o{c3o}ps{c3ps}" + (f"r{c3r}" if c3r else "") + "_v3" if ck == 3 else "_v4") + ("_mc2")
    # compact materialize CTAs/SM bound for fp32 state (default 3 = Rubin rubin-ck build; GB300 [gx-alignc] uses 4)
    matc_f32 = int(os.environ.get("GDN_STATE_COMMIT_MATC_MINB_F32", "3"))
    if matc_f32 != 3:
        tag += f"_matc{matc_f32}"
    # GB300 fp32 decode kernel (GDN_STATE_COMMIT_GB=1): bit-exact with CK=0, used for the fp32 state only
    gb_env = gb
    gb = int(os.environ.get("GDN_STATE_COMMIT_GB", "0")) if gb_env is None else int(gb_env)
    gbo = {k: int(os.environ.get(f"GDN_STATE_COMMIT_GB_{k}", v))
           for k, v in (("W", "8"), ("D", "2"), ("NS", "1"), ("F2", "1"), ("MINB", "4"), ("KREG", "1"))}
    gb_flags = []
    if gb:
        tag += "_gb{W}w{D}d{NS}n{F2}f{MINB}m{KREG}k".format(**gbo)
        gb_flags = ["-DGSC_GB=1"] + [f"-DGB_{k}={v}" for k, v in gbo.items()]
    build = os.path.join(build, f"sm{arch}_{tag}")
    os.makedirs(build, exist_ok=True)
    orig = cpp._get_cuda_arch_flags
    # torch's arch-list parser does not know "10.7f"-style arch entries; pin the device arch.
    cpp._get_cuda_arch_flags = lambda cflags=None: [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    try:
        ext = cpp.load(
            name=f"_gdn_state_commit_{tag}",
            sources=[os.path.join(_HERE, "gdn_state_commit.cu")],
            # --use_fast_math: same as vLLM's build of fused_gdn_decode_kernel.cu (bit-exactness)
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++20", "-lineinfo", f"-DGSC_MINB={minb}", f"-DGSC_NC={nc}", f"-DGSC_F2={f2}", f"-DGSC_PERSIST={ps}", f"-DGSC_PROBE={pr}", f"-DGSC_FAST={fx}", f"-DGSC_EARLY={ea}", f"-DGSC_CK={ck}", f"-DGSC_PDL={pdl}", f"-DGSC_CK2_UPD={upd}", f"-DGSC_CK3_F2={c3f2}", f"-DGSC_CK3_MAP={c3m}", f"-DGSC_CK3_STORE={c3s}", f"-DGSC_CK3_PF={c3p}", f"-DGSC_CK3_ORDER={c3o}", f"-DGSC_CK3_PFS={c3ps}", f"-DGSC_CK3_REMAP={c3r}", f"-DGSC_MATC_MINB_F32={matc_f32}"] + gb_flags,
            extra_cflags=["-O3", "-std=c++20"],
            build_directory=build,
            verbose=False,
        )
    finally:
        cpp._get_cuda_arch_flags = orig
    assert ext.log_bytes(16, 32) == log_bytes(16, 32)
    _exts[key] = ext
    if gb_env is None and minb == int(os.environ.get("GDN_STATE_COMMIT_MINB", "2")):
        _ext = ext
    logger.info_once("gdn_state_commit enabled: deferred GDN state commit kernel loaded (%s)", tag)
    return ext


# ----------------------------------------------------------------------------------------------
# kernels (python entry points)
# ----------------------------------------------------------------------------------------------
# GB300 fp32 kernel in-serving check (GDN_STATE_COMMIT_GB_CHECK=N, diagnostics only): every decode call
# (eager or CUDA-graph replay) also runs the CK0 reference kernel on a gathered copy of the rows' pages
# (ssm state + token log) and counts mismatching state/log words and output elements on the device; every
# N-th eager call logs the counters. Doubles the decode traffic: never use it in a scored run.
_GB_CHECK = int(os.environ.get("GDN_STATE_COMMIT_GB_CHECK", "0"))
_GB_CHK = {}


def _gb_check_counters(device):
    c = _GB_CHK.get("cnt")
    if c is None:
        c = _GB_CHK["cnt"] = torch.zeros(4, dtype=torch.int64, device=device)
    return c


def _gb_checked_decode(ext, args, state, state_indices, cu_seqlens, out):
    """Run the GB kernel (ext) and the CK0 reference on a gathered page copy; accumulate mismatches."""
    ref = load(gb=0)
    n = state_indices.size(0)
    HV = state.size(1)
    H = (args[0].size(1) - HV * 128) // 256  # mixed_qkv = [q (H*128) | k (H*128) | v (HV*128)]
    words = (state[0].numel() * state.element_size() + log_bytes(H, HV)) // 4
    raw = torch.as_strided(state.view(torch.int32) if state.dtype == torch.float32 else state, (state.size(0), words),
                           (state.stride(0), 1))
    idx = state_indices[:, 0].long()
    valid = idx > 0
    gidx = torch.where(valid, idx, torch.zeros_like(idx))
    sp = torch.empty((n + 1, words), dtype=raw.dtype, device=raw.device)
    sp[1:] = raw[gidx]
    sstate = torch.as_strided(sp.view(state.dtype), (n + 1,) + tuple(state.shape[1:]),
                              (words,) + tuple(state.stride()[1:]))
    sidx = torch.where(valid, torch.arange(1, n + 1, device=idx.device, dtype=torch.int32),
                       torch.zeros(n, device=idx.device, dtype=torch.int32))[:, None].contiguous()
    out2 = torch.empty_like(out)
    a = list(args)
    a[5], a[8], a[11] = sidx, sstate, out2
    ref.decode(*a)
    ext.decode(*args)
    cnt = _gb_check_counters(out.device)
    cnt[0] += 1
    cnt[1] += ((raw[gidx] != sp[1:]) & valid[:, None]).sum()
    tok = torch.arange(out.size(0), device=out.device) < cu_seqlens[n].long()
    cnt[2] += ((out.view(torch.int16) != out2.view(torch.int16)).flatten(1).any(1) & tok).sum()
    cnt[3] += valid.sum()
    if not torch.cuda.is_current_stream_capturing():
        _GB_CHK["eager"] = _GB_CHK.get("eager", 0) + 1
        if _GB_CHK["eager"] % _GB_CHECK == 0:
            c = cnt.tolist()
            logger.warning("gdn_state_commit GB check: kernel calls %d, rows %d, state/log word mismatches %d, "
                           "output token-row mismatches %d", c[0], c[3], c[1], c[2])


def decode(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state,
           output_gate, norm_weight, out=None, scale=128**-0.5, norm_eps=1e-5,
           output_gate_activation="silu"):
    if out is None:
        out = torch.empty_like(output_gate)
    if not state_indices.is_contiguous():
        state_indices = state_indices[:, :1].contiguous()
    STATS["decode_calls"] += 1
    args = (mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens,
            state, output_gate, norm_weight, out, float(scale), float(norm_eps),
            output_gate_activation == "sigmoid")
    if _GB_CHECK and state.dtype == torch.float32 and os.environ.get("GDN_STATE_COMMIT_GB", "0") == "1" \
            and state_indices.size(0) > 0:
        _gb_checked_decode(load(), args, state, state_indices, cu_seqlens, out)
    else:
        load().decode(*args)
    _dbg(f"decode N={state_indices.size(0)} w={state_indices.size(1)}")
    return out


def _dt_type(t):
    return 0 if t.dtype == torch.float32 else (1 if t.dtype == torch.bfloat16 else 2)


class LayerTable:
    """Per-layer pointers (ssm state base, A_log, dt_bias) + group index for `materialize`."""

    def __init__(self, entries, device):
        # entries: list of (ssm_state_tensor, A_log, dt_bias, group_idx)
        assert entries, "no GDN layers"
        st = entries[0][0]
        self.H = None
        self.HV = st.size(1)
        self.slot_stride = st.stride(0)
        for s, al, db, _ in entries:
            assert s.stride(0) == self.slot_stride and s.size(1) == self.HV
            assert al.dtype == torch.float32 and al.is_contiguous() and db.is_contiguous()
        self.dt_type = _dt_type(entries[0][2])
        self.state_bf16 = int(st.dtype == torch.bfloat16)
        assert st.dtype in (torch.float32, torch.bfloat16)
        self.state = torch.tensor([e[0].data_ptr() for e in entries], dtype=torch.int64, device=device)
        self.alog = torch.tensor([e[1].data_ptr() for e in entries], dtype=torch.int64, device=device)
        self.dtb = torch.tensor([e[2].data_ptr() for e in entries], dtype=torch.int64, device=device)
        self.group = torch.tensor([e[3] for e in entries], dtype=torch.int32, device=device)
        self.n = len(entries)
        self._keep = entries


def materialize(mode, num_items, table: LayerTable, H, a0, a1=None, a2=None, a3=None, a4=None,
                has_init=None, bt_ptrs=None, bt_stride=0, block_size=0, idx_map=None, compact=0):
    """compact > 0 ([rubin-ck], modes 1-3): grid (compact, H, layers) over only the items that pass the per-item
    decision instead of (num_items, H, layers) mostly-empty CTAs. Same bytes written (bitwise)."""
    if num_items <= 0:
        return
    STATS["materialize_calls"] += 1
    i32 = lambda t: None if t is None else t.to(torch.int32).contiguous()  # noqa: E731
    args = (int(mode), int(num_items), int(table.n), i32(a0), i32(a1), i32(a2), i32(a3),
            i32(a4), None if has_init is None else has_init.to(torch.bool).contiguous(),
            i32(idx_map), bt_ptrs, int(bt_stride), int(block_size), table.state, table.alog, table.dtb,
            table.group, int(table.slot_stride), int(H), int(table.HV), int(table.dt_type), int(table.state_bf16))
    ext = load()
    if compact and int(mode) != 0 and hasattr(ext, "materialize_compact"):
        logger.info_once("gdn_state_commit: compact align materialize active (VLLM_MAMBA_ALIGN_COMPACT=%d)",
                         int(compact))
        ext.materialize_compact(int(compact), *args)
    else:
        ext.materialize(*args)
    _dbg(f"materialize mode={mode} items={num_items} layers={table.n}")


# ----------------------------------------------------------------------------------------------
# page layout / KV cache spec (model_executor/layers/mamba/{mamba_utils,abstract}.py)
# ----------------------------------------------------------------------------------------------
def adjust_kv_cache_spec(spec, vllm_config):
    """GDN MambaSpec: token-log dtype appended, no speculative blocks (unless layout-only)."""
    if spec is not None and getattr(spec.mamba_type, "name", str(spec.mamba_type)) == "GDN_ATTN":
        assert vllm_config.cache_config.mamba_cache_mode == "align", (
            "gdn_state_commit supports mamba_cache_mode=align (prefix caching) only")
        dtypes = tuple(spec.dtypes)
        if len(dtypes) < len(spec.shapes):  # layer get_state_dtype hides the log dtype
            dtypes = dtypes + (torch.bfloat16,) * (len(spec.shapes) - len(dtypes))
        spec = dataclasses.replace(
            spec, dtypes=dtypes,
            num_speculative_blocks=spec.num_speculative_blocks if layout_only() else 0)
        logger.info_once("gdn_state_commit enabled: GDN state page holds the token log (layout_only=%s)",
                         layout_only())
    return spec


# ----------------------------------------------------------------------------------------------
# GDN metadata (v1/attention/backends/gdn_attn.py)
# ----------------------------------------------------------------------------------------------
def postprocess_metadata(builder, md, num_accepted_tokens=None, num_decode_draft_tokens_cpu=None):
    """Runs at the end of GDNAttentionMetadataBuilder.build (not in layout-only mode)."""
    w = builder.num_spec + 1
    si = md.spec_state_indices_tensor
    if si is not None and si.size(1) < w:
        md.spec_state_indices_tensor = si[:, :1].expand(si.size(0), w)
    ns = None
    if num_accepted_tokens is not None and md.non_spec_state_indices_tensor is not None:
        n = md.non_spec_state_indices_tensor.size(0)
        if num_decode_draft_tokens_cpu is not None and md.spec_sequence_masks is not None:
            mask = num_decode_draft_tokens_cpu < 0
            ns = num_accepted_tokens[: mask.numel()][mask][:n]
        else:
            ns = num_accepted_tokens[:n]
    md.gsc_non_spec_num_accepted = ns
    return md


# ----------------------------------------------------------------------------------------------
# Qwen GDN layer (model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)
# ----------------------------------------------------------------------------------------------
def _layer_check(layer):
    # Rebuild whenever the KV cache is (re)bound: profiling runs use a placeholder cache.
    if getattr(layer, "_gsc_checked", None) == layer.kv_cache[1].data_ptr():
        return
    kv = layer.kv_cache
    conv, ssm = kv[0], kv[1]
    ssm_bytes = ssm[0].numel() * ssm.element_size()
    H = layer.num_k_heads // layer.tp_size
    page = ssm.stride(0) * ssm.element_size()
    assert conv.stride(0) * conv.element_size() == page
    # the token log lives in the page right after the ssm state
    assert (ssm.data_ptr() - conv.data_ptr()) + ssm_bytes + log_bytes(H, ssm.size(1)) <= page, (
        f"GDN page {page} B too small for conv+ssm+log; is GDN_STATE_COMMIT=1 set in every process?")
    layer._gsc_table = LayerTable([(ssm, layer.A_log, layer.dt_bias, 0)], ssm.device)
    layer._gsc_H = H
    layer._gsc_checked = ssm.data_ptr()


def materialize_non_spec(layer, md):
    """In-place commit of pending logs for every non-spec reader (prefill / plain decode) slot.

    Single entry for every caller. With VLLM_GDN_GROUP_MATERIALIZE (GGM_OG2=1) all
    GDN layers of the KV-cache group are committed in one launch (gdn_step_plan).
    """
    if gdn_step_plan.GROUP_MATERIALIZE:
        return gdn_step_plan.group_materialize_non_spec(
            layer, md, _materialize_non_spec_layer
        )
    return _materialize_non_spec_layer(layer, md)


def _materialize_non_spec_layer(layer, md):
    """materialize_non_spec for one layer."""
    done = md.__dict__.setdefault("_gsc_done", set())
    if id(layer) in done:
        return
    done.add(id(layer))
    slots = md.non_spec_state_indices_tensor
    # Only real non-spec requests (zero-length / padded rows are at the back and may hold stale
    # block ids of other requests' live slots: never touch them).
    items = md.num_prefills + md.num_decodes
    if slots is None or items <= 0:
        return
    items = min(items, slots.size(0))
    _layer_check(layer)
    n_src = getattr(md, "gsc_non_spec_num_accepted", None)
    n = torch.zeros(items, dtype=torch.int32, device=slots.device)
    if n_src is not None and n_src.numel() > 0:
        k = min(items, n_src.numel())
        n[:k] = n_src[:k]
    has_init = md.has_initial_state
    hi = None
    if has_init is not None:
        hi = torch.zeros(items, dtype=torch.bool, device=slots.device)
        k = min(items, has_init.numel())
        hi[:k] = has_init[:k]
    if _DEBUG and not torch.cuda.is_current_stream_capturing():
        logger.info("gdn_state_commit debug: mat0 items=%d slots=%s max=%d nslots=%d n_src=%s has_init=%s "
                    "P=%d D=%d S=%d", items, tuple(slots.shape), int(slots[:items].max()),
                    layer.kv_cache[1].size(0), None if n_src is None else tuple(n_src.shape),
                    None if has_init is None else tuple(has_init.shape), md.num_prefills,
                    md.num_decodes, md.num_spec_decodes)
    materialize(0, items, layer._gsc_table, layer._gsc_H, slots[:items], n, has_init=hi)


def fused_norm_prologue(layer, md, mixed_qkv, b, a, output_gate, core_attn_out) -> bool:
    """Start of QwenGatedDeltaNetAttention._forward_core_fused_norm (md = this layer's GDN
    metadata). Returns True when the whole call was handled here (the caller returns)."""
    _layer_check(layer)
    decode_only = layer._can_use_fused_gdn_mtp_decode(md) and md.num_prefills == 0
    if not decode_only:
        if md.num_spec_decodes > 0 and not layer._can_use_mixed_fastpath(md):
            # Spec requests are not a token prefix (e.g. a <=4-token prefill chunk was
            # reordered among the decodes): permute tokens spec-first, run the zero-copy
            # mixed path (-> deferred kernel), scatter back.
            md2 = copy.copy(md)
            md2.spec_tokens_are_prefix = True
            if (md.spec_token_indx is None or md.non_spec_token_indx is None
                    or not layer._can_use_mixed_fastpath(md2)):
                raise RuntimeError(
                    "[gdn_state_commit] spec-decode batch would take the FLA spec path "
                    f"(num_decodes={md.num_decodes}); unsupported with deferred state commit")
            STATS["permuted_mixed"] = STATS.get("permuted_mixed", 0) + 1
            materialize_non_spec(layer, md)
            perm = torch.cat([md.spec_token_indx, md.non_spec_token_indx]).long()
            n = perm.numel()
            out_p = torch.empty((n,) + tuple(core_attn_out.shape[1:]), dtype=core_attn_out.dtype,
                                device=core_attn_out.device)
            layer._forward_core_mixed_fastpath(
                mixed_qkv=mixed_qkv.index_select(0, perm), b=b.index_select(0, perm),
                a=a.index_select(0, perm), output_gate=output_gate.index_select(0, perm),
                core_attn_out=out_p, attn_metadata=md2)
            core_attn_out.index_copy_(0, perm, out_p)
            return True
        materialize_non_spec(layer, md)
    return False


def forward_core_prologue(layer, md) -> None:
    """Start of QwenGatedDeltaNetAttention._forward_core (md = this layer's GDN metadata)."""
    if md.num_spec_decodes > 0:
        raise RuntimeError("[gdn_state_commit] FLA spec path unsupported with deferred commit")
    materialize_non_spec(layer, md)


# ----------------------------------------------------------------------------------------------
# worker align-mode copies (v1/worker/mamba_utils.py MambaSpecDecodeGPUContext)
# ----------------------------------------------------------------------------------------------
def align_compact() -> int:
    """[rubin-ck] VLLM_MAMBA_ALIGN_COMPACT=G (default 0 = off): the align-mode precopy / postprocess state copies
    (stock conv copies in v1/worker/mamba_utils.py and the GDN ssm materialize here) run a grid of G CTAs per
    (state, tile) / (key head, layer) over only the requests whose copy decision is taken, instead of one CTA per
    request. Exact: the same bytes are written."""
    return int(os.environ.get("VLLM_MAMBA_ALIGN_COMPACT", "0"))


class _WorkerTables:
    table = None
    H = None


def _build_worker_table(ctx, kv_cache_config, forward_context):
    entries = []
    H = None
    for g, gid in enumerate(ctx.mamba_group_ids):
        for name in kv_cache_config.kv_cache_groups[gid].layer_names:
            layer = forward_context.get(name)
            if layer is None or not hasattr(layer, "A_log") or len(getattr(layer, "kv_cache", ())) < 2:
                continue
            _layer_check(layer)
            H = layer._gsc_H
            entries.append((layer.kv_cache[1], layer.A_log, layer.dt_bias, g))
    if entries:
        _WorkerTables.table = LayerTable(entries, entries[0][0].device)
        _WorkerTables.H = H
        logger.warning("gdn_state_commit: worker materialize table: %d GDN layers in %d groups",
                       len(entries), len(ctx.mamba_group_ids))


def worker_precopy(ctx, num_reqs, state_idx_gpu, src_col_gpu, token_bias_gpu, idx_mapping):
    """End of MambaSpecDecodeGPUContext.run_fused_precopy."""
    t = _WorkerTables.table
    if num_reqs == 0 or not ctx.is_initialized or t is None:
        return
    STATS["precopy_calls"] += 1
    materialize(1, num_reqs, t, _WorkerTables.H, src_col_gpu, state_idx_gpu, token_bias_gpu,
                bt_ptrs=ctx.block_table_ptrs, bt_stride=ctx.block_table_stride_req,
                block_size=ctx.block_size, idx_map=idx_mapping, compact=align_compact())


def worker_postprocess(ctx, num_reqs, num_accepted_tokens_gpu, mamba_state_idx_gpu,
                       num_scheduled_tokens_gpu, num_computed_tokens_gpu, num_draft_tokens_gpu):
    """End of MambaSpecDecodeGPUContext.run_fused_postprocess (V1 runner). The stock kernel only
    writes num_accepted_tokens_out; its inputs are unchanged."""
    t = _WorkerTables.table
    if num_reqs == 0 or not ctx.is_initialized or t is None:
        return
    STATS["postprocess_calls"] += 1
    materialize(2, num_reqs, t, _WorkerTables.H, num_accepted_tokens_gpu, mamba_state_idx_gpu,
                num_scheduled_tokens_gpu, num_computed_tokens_gpu, num_draft_tokens_gpu,
                bt_ptrs=ctx.block_table_ptrs, bt_stride=ctx.block_table_stride_req,
                block_size=ctx.block_size, compact=align_compact())


def worker_postprocess_align(ctx, num_reqs, num_accepted_tokens_gpu, state_idx_gpu,
                             new_num_computed_tokens_gpu, idx_mapping):
    """End of MambaSpecDecodeGPUContext.run_fused_postprocess_align (V2 runner). The stock kernel
    reads a snapshot (ctx.num_accepted_tokens_out) and resets num_accepted_tokens_gpu in place;
    the snapshot keeps this step's accept counts."""
    t = _WorkerTables.table
    if num_reqs == 0 or not ctx.is_initialized or t is None:
        return
    STATS["postprocess_calls"] += 1
    materialize(3, num_reqs, t, _WorkerTables.H, ctx.num_accepted_tokens_out, state_idx_gpu,
                None, new_num_computed_tokens_gpu, None,
                bt_ptrs=ctx.block_table_ptrs, bt_stride=ctx.block_table_stride_req,
                block_size=ctx.block_size, idx_map=idx_mapping, compact=align_compact())
