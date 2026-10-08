# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""U-cache GDN MTP decode on qwen-v2 (VLLM_GDN_DECODE_UCACHE=1; default off; FLOAT-ORDER vs the
stock gdn_mtp_cuda kernel; HIGH band only).

At steps with at least VLLM_GDN_UCACHE_MIN_ROWS (default 32) spec-decode rows, the bf16 GDN MTP recurrence runs
FlashInfer #4081's ReplaySSM u-cache kernel (vendored: vllm/third_party/flashinfer_gdn_ucache) instead of
gdn_mtp_cuda. Deferred-state decode adapted to qwen-v2's in-place snapshot layout:

* Checkpoint + ring. The request's spec column 0 holds a checkpoint S0; the tokens since S0 live in a per-layer,
  per-request-slot ring (k / u / cumulative log-decay; slot = the V2 runner's request-state index). The kernel folds
  the ring into S0 only every few steps (VLLM_GDN_UCACHE_FLUSH_MIN, default 13). Columns 1..num_spec are not written
  by u-cache steps.
* Validity. ring valid for (layer, block X, slot r) <=> active[r] and tags[layer][X][vh] == {cursor count, r}. The
  tags side table replaces gsc's in-page flag words (qwen-v2 pages carry no log). active[r] is cleared when a request
  takes slot r (reset_slot) and when a request leaves the u-cache band (prepass).
* Readers outside the decode kernel:
    R1 align precopy (block change)     -> fold_r1: fold(src checkpoint, n = acc) into the new block; the stock
                                           precopy skips the temporal copy of active slots
    R2 align postprocess (boundary)     -> fold_r2: fold(src checkpoint, n = bias + 1) into the boundary block (in
                                           place when src == dst, also for bias 0, so the published block holds the
                                           full state); the stock postprocess skips the temporal copy of active slots
    R3 next spec step                   -> native (ucache_prep commits acc tokens)
    R4 non-spec decode                  -> not reachable: gdnfix (VLLM_GDN_ZERO_DRAFT_AS_SPEC=1, required) runs every
                                           decode row as a spec row; T < 4 rows take the staged path (zero-padded)
    R5 prefill with initial state       -> reads blocks written by prefill / R1 / R2 (full states)
  plus the band switch: prepass (one launch per step, outside CUDA graphs): entering the band (stock layout) copies
  the state of the last accepted token into column 0 (rows with acc > 1); leaving it materializes column acc - 1
  and clears active.
* NULL / capture: rows with state block <= 0 (NULL_BLOCK_ID padding) are skipped by prep and get zero outputs, as
  gdn_mtp_cuda. Every launch is graph-safe (fixed pointers; per-step data in persistent buffers).

Exactness: the prep / norm / fold arithmetic is gsc's verbatim (bitwise target: tests/kernels/mamba/
test_gdn_ucache_qv2.py vs gsc u-cache); the u-cache class is float-order vs gdn_mtp_cuda.
"""

import hashlib
import os
import weakref

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
MAX_T = 4
ENABLED = os.environ.get("VLLM_GDN_DECODE_UCACHE", "0") == "1"
MIN_ROWS = int(os.environ.get("VLLM_GDN_UCACHE_MIN_ROWS", "32"))
FLUSH_MIN = int(os.environ.get("VLLM_GDN_UCACHE_FLUSH_MIN", "13"))
FUSED_NORM = os.environ.get("VLLM_GDN_UCACHE_FUSED_NORM", "1") == "1"
STRIDED = os.environ.get("VLLM_GDN_UCACHE_STRIDED", "1") == "1"
ERR_EVERY = int(os.environ.get("VLLM_GDN_UCACHE_ERR_EVERY", "2000"))
DEBUG = os.environ.get("VLLM_GDN_UCACHE_DEBUG", "0") == "1"  # counters [0..2] != 0 -> raise (eager checks)
# KQ normalization (VLLM_GDN_UCACHE_KQFIX, default 1): raw q/k MMA operands + fp32 inverse norms; k ring
# = raw k, u ring = u * inv|k| (fold form unchanged). Process-wide: rings written with and without it are not
# interchangeable.
# Levels: 0 = normalized operands; 1 = raw operands; 2 = raw operands + hi/lo bf16 U MMA operands
# (u rounded once into the ring; outputs value-identical to 1). The vendored adapter takes the int (True == 1).
KQFIX = int(os.environ.get("VLLM_GDN_UCACHE_KQFIX", "1"))  # Level 2 remains experimental: non-finite ring values observed.
assert KQFIX in (0, 1, 2), f"VLLM_GDN_UCACHE_KQFIX must be 0, 1 or 2 (got {KQFIX})"
# GDN_UCACHE_RING_DTYPE (read by the vendored kernel at its import, which happens after this module): fp16 rings are
# the port's default; "bf16" selects bf16 rings.
if ENABLED:
    os.environ.setdefault("GDN_UCACHE_RING_DTYPE", "fp16")
RING_F16 = os.environ.get("GDN_UCACHE_RING_DTYPE", "").strip().lower() in ("fp16", "float16", "half")
RING_DTYPE = torch.float16 if RING_F16 else torch.bfloat16
STATS: dict = {}


class _Step:
    """Per-step band decision, written by MambaHybridModelState.prepare_attn (also for CUDA-graph capture) and read
    by the GDN layers' spec path, so the captured graphs and the prepass agree."""

    band = False  # this step's spec decode runs the u-cache
    uniform = True  # every spec row has 4 tokens (strided / fused-norm path eligible)


STEP = _Step()


class _State:
    def __init__(self):
        self.slots = 0
        self.rows = 0
        self.slot = None  # int32 [rows]: request-state slot of each spec row of the step (fixed pointer)
        self.active = None  # int32 [slots]
        self.stage = None
        self.layers = []  # registered GDN layers, construction order
        self.ext = None
        self.calls = 0
        self.table = None  # worker fold table (built once the KV cache and block tables exist)


UC = _State()


def enabled() -> bool:
    return ENABLED


# ------------------------------------------------------------------------------------------------- build
def load():
    """JIT-build the device helpers (startup only)."""
    if UC.ext is not None:
        return UC.ext
    import torch.utils.cpp_extension as cpp

    src = os.path.join(_HERE, "gdn_ucache.cu")
    with open(src, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:12]
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    from vllm import envs

    build = os.path.join(envs.VLLM_CACHE_ROOT, "gdn_ucache", f"sm{arch}_{digest}")
    os.makedirs(build, exist_ok=True)
    orig = cpp._get_cuda_arch_flags
    cpp._get_cuda_arch_flags = lambda cflags=None: [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    try:
        # --use_fast_math: same flags as gsc's build (bitwise target)
        UC.ext = cpp.load(name=f"_gdn_ucache_{digest}", sources=[src],
                          extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++20", "-lineinfo"],
                          extra_cflags=["-O3", "-std=c++20"], build_directory=build, verbose=False)
    finally:
        cpp._get_cuda_arch_flags = orig
    logger.info("gdn_ucache: device helpers built (%s)", digest)
    return UC.ext


# ------------------------------------------------------------------------------------------------- layers
def register_layer(layer, H: int, HV: int, vllm_config) -> None:
    """QwenGatedDeltaNetAttention.__init__ (VLLM_GDN_DECODE_UCACHE=1): allocate the layer's rings before KV-cache
    profiling. Ring bytes per layer and slot: k 16*32*128*2 + u HV*32*128*2 + g HV*32*4 (+ cursor)."""
    dev = torch.device("cuda", torch.cuda.current_device())
    if UC.slot is None:
        _check_config(vllm_config)
        sc = vllm_config.scheduler_config
        UC.slots = int(sc.max_num_seqs)
        sizes = list(getattr(vllm_config.compilation_config, "cudagraph_capture_sizes", None) or [])
        UC.rows = max([UC.slots] + [int(x) // MAX_T + 1 for x in sizes]) + 1
        R = UC.rows
        UC.slot = torch.full((R,), -1, dtype=torch.int32, device=dev)
        UC.active = torch.zeros((UC.slots,), dtype=torch.int32, device=dev)
        bf = torch.bfloat16
        UC.stage = dict(
            q=torch.zeros((R, MAX_T, H, 128), dtype=bf, device=dev),
            k=torch.zeros((R, MAX_T, H, 128), dtype=bf, device=dev),
            v=torch.zeros((R, MAX_T, HV, 128), dtype=bf, device=dev),
            a=torch.zeros((R, MAX_T, HV), dtype=bf, device=dev),
            b=torch.zeros((R, MAX_T, HV), dtype=bf, device=dev),
            o=torch.zeros((R, MAX_T, HV, 128), dtype=bf, device=dev),
            sidx=torch.full((R,), -1, dtype=torch.int32, device=dev),
            ridx=torch.zeros((R,), dtype=torch.int32, device=dev),
            hist=torch.zeros((R,), dtype=torch.int32, device=dev),
            base=torch.zeros((R,), dtype=torch.int32, device=dev))
        load()
        logger.info("gdn_ucache: GDN u-cache decode ON for steps with >= %d spec rows (VLLM_GDN_DECODE_UCACHE=1, "
                    "float-order vs gdn_mtp_cuda): %d ring slots, %d staging rows, flush_min %d, fused norm %d, "
                    "kq-fix %d, ring %s", MIN_ROWS, UC.slots, R, FLUSH_MIN, int(FUSED_NORM), int(KQFIX), RING_DTYPE)
    S = UC.slots
    layer._gdn_ucache = dict(
        kr=torch.zeros((S, H, 32, 128), dtype=RING_DTYPE, device=dev),
        ur=torch.zeros((S, HV, 32, 128), dtype=RING_DTYPE, device=dev),
        gr=torch.zeros((S, HV, 32), dtype=torch.float32, device=dev),
        cur=torch.zeros((S, 4), dtype=torch.int32, device=dev),
        tags=None, H=H, HV=HV)
    UC.layers.append(layer)


def _check_config(vllm_config) -> None:
    """Fail closed at start-up on the configurations the port does not cover."""
    why = []
    if not getattr(vllm_config, "use_v2_model_runner", False):
        why.append("needs the V2 model runner (slot = request-state index; R1/R2 hooks are V2's)")
    cc = vllm_config.cache_config
    if getattr(cc, "mamba_cache_mode", None) != "align":
        why.append(f"needs mamba_cache_mode=align (got {getattr(cc, 'mamba_cache_mode', None)})")
    if str(getattr(cc, "mamba_ssm_cache_dtype", "auto")) not in ("auto", "bfloat16"):
        why.append(f"needs a bf16 SSM state (mamba_ssm_cache_dtype={cc.mamba_ssm_cache_dtype})")
    k = int(getattr(vllm_config, "num_speculative_tokens", 0) or 0)
    if not 1 <= k <= MAX_T - 1:
        why.append(f"needs 1 <= num_speculative_tokens <= {MAX_T - 1} (got {k})")
    # same env reads (and defaults) as gdn_zero_draft.ZERO_DRAFT_AS_SPEC / mamba_utils.MAMBA_COPY_COMPACT
    if os.environ.get("VLLM_GDN_ZERO_DRAFT_AS_SPEC", "1") != "1":
        why.append("needs VLLM_GDN_ZERO_DRAFT_AS_SPEC=1 (no non-spec decode readers of the checkpoint)")
    if os.environ.get("VLLM_MAMBA_COPY_COMPACT", "0") == "1":
        why.append("VLLM_MAMBA_COPY_COMPACT=1 is not covered (its copies do not skip u-cache slots)")
    if why:
        raise RuntimeError("[gdn_ucache] VLLM_GDN_DECODE_UCACHE=1: " + "; ".join(why))
    if k != MAX_T - 1:
        logger.warning("gdn_ucache: num_speculative_tokens=%d: every u-cache call takes the staged path", k)


def _tags(layer, state: torch.Tensor) -> torch.Tensor:
    """The layer's tag table [blocks, HV] int32 (allocated once its ssm pool exists; never under capture)."""
    b = layer._gdn_ucache
    t = b["tags"]
    if t is None or t.size(0) != state.size(0):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gdn_ucache] tag table first needed under CUDA-graph capture; an eager call must "
                               "precede capture")
        t = b["tags"] = torch.zeros((state.size(0), b["HV"]), dtype=torch.int32, device=state.device)
        UC.table = None  # rebuild the worker table with the new pointer
    return t


def reset_slot(req_index: int) -> None:
    """MambaHybridModelState.add_request: a new (or resumed) request takes slot req_index."""
    if UC.active is not None:
        UC.active[req_index].fill_(0)


# ------------------------------------------------------------------------------------------------- step
def set_step(band: bool, uniform: bool) -> None:
    STEP.band = bool(band) and ENABLED
    STEP.uniform = bool(uniform)


def fill_slots(idx_mapping: torch.Tensor, num_spec: int) -> None:
    """Request-state slot of each spec row (spec rows lead the batch); the remaining rows -1."""
    if UC.slot is None:
        return
    n = min(int(num_spec), UC.slot.numel(), idx_mapping.numel())
    if n:
        UC.slot[:n].copy_(idx_mapping[:n], non_blocking=True)
    if n < UC.slot.numel():
        UC.slot[n:].fill_(-1)


_NW32: dict = {}


def _norm_w32(w):
    """Cache an fp32 norm weight only for the same live, unmodified source tensor.

    Storage addresses may be reused after a tensor dies; an address-only cache
    could then return another parameter's values.
    """
    if w.dtype == torch.float32 and w.is_contiguous():
        return w
    key = (w.data_ptr(), w.dtype, w.device, getattr(w, "_version", 0))
    hit = _NW32.get(key)
    if hit is not None and hit[0]() is w:
        return hit[1]
    for k in [k for k, v in _NW32.items() if v[0]() is None]:  # drop entries of freed weights
        del _NW32[k]
    t = w.detach().float().contiguous()  # bf16 -> fp32 is exact (uc_norm converts the same way)
    _NW32[key] = (weakref.ref(w), t)
    return t


_PRECOMPILED: set = set()
# kq_fix is passed only when on, so the default build also runs with a pre-kq-fix vendored adapter
_KQ = {"kq_fix": KQFIX} if KQFIX else {}


def _precompile_fallback(ucache_decode, state, A_log, dt_bias, b, H, scale):
    """Compile (never launch) the staged / unfused variant on the first eager call for this pool (no mid-run JIT)."""
    key = (state.device.index, state.data_ptr(), tuple(state.shape), tuple(state.stride()), int(H))
    if key in _PRECOMPILED or torch.cuda.is_current_stream_capturing():
        return
    _PRECOMPILED.add(key)
    st = UC.stage
    try:
        ucache_decode(st["q"][:1], st["k"][:1], st["v"][:1], st["a"][:1], st["b"][:1], A_log, dt_bias, state,
                      st["sidx"][:1], st["ridx"][:1], b["kr"], b["ur"], b["gr"], st["hist"][:1], st["base"][:1],
                      st["o"][:1], float(scale), int(FLUSH_MIN), compile_only=True, **_KQ)
    except Exception as e:  # noqa: BLE001 - a warm-up miss only costs a lazy compile later, never correctness
        logger.warning_once("gdn_ucache: fallback precompile failed (%r); it compiles lazily", e)


def decode(layer, mixed_qkv, a, b, state_indices, cu_seqlens, num_accepted, state, output_gate, norm_weight, out,
           scale: float, norm_eps: float, sigmoid: bool) -> None:
    """Spec-decode recurrence + gated RMSNorm of the step's spec rows into `out` [tokens, HV, 128] (normed).
    state_indices [n, 1 + k] (column 0 = checkpoint block), cu_seqlens [n + 1], num_accepted [n]; the request-state
    slot of row i is UC.slot[i] (fill_slots)."""
    n = state_indices.size(0)
    if n == 0:
        return
    if n > UC.rows:
        raise RuntimeError(f"[gdn_ucache] {n} decode rows > {UC.rows} staging rows")
    assert state.dtype == torch.bfloat16, "u-cache needs the bf16 GDN state"
    bufs = layer._gdn_ucache
    tags = _tags(layer, state)
    st = UC.stage
    HV = state.size(1)
    H = (mixed_qkv.size(1) - HV * 128) // 256
    strided = (STRIDED and STEP.uniform and mixed_qkv.stride(1) == 1 and mixed_qkv.stride(0) % 8 == 0
               and mixed_qkv.data_ptr() % 16 == 0 and a.stride(1) == 1 and tuple(b.stride()) == tuple(a.stride())
               and mixed_qkv.size(0) >= 4 * n and a.size(0) >= 4 * n and b.size(0) >= 4 * n)
    ext = UC.ext
    ext.ucache_prep(mixed_qkv, a, b, state_indices, cu_seqlens, num_accepted, UC.slot, bufs["cur"], tags, UC.active,
                    HV, st["q"], st["k"], st["v"], st["a"], st["b"], st["sidx"], st["ridx"], st["hist"], st["base"],
                    int(FLUSH_MIN), not strided)
    from vllm.third_party.flashinfer_gdn_ucache.adapter import ucache_decode

    A_log, dt_bias = layer.A_log, layer.dt_bias
    _precompile_fallback(ucache_decode, state, A_log, dt_bias, bufs, H, scale)
    fused = (FUSED_NORM and strided and out.is_contiguous() and out.dim() == 3 and out.size(0) >= 4
             and out.size(0) % 4 == 0 and output_gate.stride(2) == 1 and output_gate.stride(1) == 128
             and output_gate.dtype == torch.bfloat16)
    hk = H * 128
    if fused:
        ucache_decode(mixed_qkv[:, :hk], mixed_qkv[:, hk:2 * hk], mixed_qkv[:, 2 * hk:2 * hk + HV * 128], a, b,
                      A_log, dt_bias, state, st["sidx"][:n], st["ridx"][:n], bufs["kr"], bufs["ur"], bufs["gr"],
                      st["hist"][:n], st["base"][:n], out.view(-1, 4, HV, 128), float(scale), int(FLUSH_MIN),
                      qkv_rs=int(mixed_qkv.stride(0)), ab_ts=int(a.stride(0)), H=int(H),
                      norm=(output_gate, _norm_w32(norm_weight), cu_seqlens, float(norm_eps), bool(sigmoid)), **_KQ)
        STATS["fused"] = STATS.get("fused", 0) + 1
    else:
        if strided:
            ucache_decode(mixed_qkv[:, :hk], mixed_qkv[:, hk:2 * hk], mixed_qkv[:, 2 * hk:2 * hk + HV * 128], a, b,
                          A_log, dt_bias, state, st["sidx"][:n], st["ridx"][:n], bufs["kr"], bufs["ur"], bufs["gr"],
                          st["hist"][:n], st["base"][:n], st["o"][:n], float(scale), int(FLUSH_MIN),
                          qkv_rs=int(mixed_qkv.stride(0)), ab_ts=int(a.stride(0)), H=int(H), **_KQ)
        else:
            ucache_decode(st["q"][:n], st["k"][:n], st["v"][:n], st["a"][:n], st["b"][:n], A_log, dt_bias, state,
                          st["sidx"][:n], st["ridx"][:n], bufs["kr"], bufs["ur"], bufs["gr"], st["hist"][:n],
                          st["base"][:n], st["o"][:n], float(scale), int(FLUSH_MIN), **_KQ)
            STATS["staged"] = STATS.get("staged", 0) + 1
        ext.ucache_norm(st["o"], cu_seqlens, st["sidx"], output_gate, norm_weight, out, float(norm_eps),
                        bool(sigmoid), int(n))
    STATS["calls"] = STATS.get("calls", 0) + 1
    _check_errors()


def _check_errors() -> None:
    UC.calls += 1
    if ERR_EVERY <= 0 or UC.calls % ERR_EVERY or torch.cuda.is_current_stream_capturing():
        return
    e = UC.ext.ucache_errors(False).tolist()
    bad = e[0] or e[1] or e[2]
    if DEBUG and bad:
        raise RuntimeError(f"[gdn_ucache] invariant violated: counters {e[:8]}, first rejected row {e[8:16]}")
    (logger.warning if bad else logger.info)(
        "gdn_ucache counters: rows not run %d (bad slot %d, cu != 4 row %d, T %d), lost-history rows %d, prepass "
        "lost-state rows %d, fresh rows %d, fold fallback copies %d (after %d eager calls; paths %s); first rejected "
        "row %s", e[0], e[4], e[5], e[6], e[1], e[2], e[3], e[7], UC.calls, STATS, e[8:16])


def errors(reset: bool = False) -> list:
    return UC.ext.ucache_errors(reset).tolist() if UC.ext is not None else [0] * 16


# ------------------------------------------------------------------------------------------------- worker folds
class _Table:
    """Per-layer pointers for the multi-layer fold kernel; block tables of the align context's mamba groups."""

    def __init__(self, layers, groups, bt_ptrs, bt_stride, block_size, dev):
        L = len(layers)
        ptrs = torch.zeros((6, L), dtype=torch.int64)
        self.slot_stride = None
        for i, lay in enumerate(layers):
            b = lay._gdn_ucache
            ssm = lay.kv_cache[1]
            tags = _tags(lay, ssm)
            for j, t in enumerate((ssm, b["kr"], b["ur"], b["gr"], b["cur"], tags)):
                p = t.data_ptr()
                ptrs[j, i] = p if p < (1 << 63) else p - (1 << 64)
            ss = int(ssm.stride(0))
            assert self.slot_stride in (None, ss), "GDN ssm pools must share the block stride"
            assert ssm.stride(1) == 128 * 128 and ssm.stride(2) == 128 and ssm.stride(3) == 1
            self.slot_stride = ss
        self.ptrs = ptrs.to(dev)
        self.group = torch.tensor(groups, dtype=torch.int32, device=dev)
        self.bt_ptrs = bt_ptrs
        self.bt_stride = int(bt_stride)
        self.block_size = int(block_size)
        self.H = layers[0]._gdn_ucache["H"]
        self.HV = layers[0]._gdn_ucache["HV"]


def bind_worker(ctx, kv_cache_config, forward_context) -> None:
    """MambaSpecDecodeGPUContext._populate_metadata (once; KV cache bound): the fold table of the u-cache layers."""
    if not ENABLED or not UC.layers:
        return
    names = {id(lay): None for lay in UC.layers}
    layers, groups = [], []
    for g_local, gid in enumerate(ctx.mamba_group_ids):
        for name in kv_cache_config.kv_cache_groups[gid].layer_names:
            lay = forward_context[name]
            if id(lay) in names:
                layers.append(lay)
                groups.append(g_local)
    if len(layers) != len(UC.layers):
        raise RuntimeError(f"[gdn_ucache] {len(UC.layers)} u-cache layers, {len(layers)} found in the mamba groups")
    UC.table = _Table(layers, groups, ctx.block_table_ptrs, ctx.block_table_stride_req, ctx.block_size,
                      UC.active.device)
    UC.table_src = (ctx, kv_cache_config, forward_context)
    logger.info("gdn_ucache: worker fold table bound (%d layers, %d groups)", len(layers), len(set(groups)))


def _table():
    if UC.table is None and getattr(UC, "table_src", None) is not None:
        bind_worker(*UC.table_src)
    return UC.table


def _fold(mode: int, num_rows: int, idx_mapping, **kw) -> None:
    t = _table()
    if t is None or num_rows <= 0:
        return
    if idx_mapping.dtype != torch.int64 or not idx_mapping.is_contiguous():
        idx_mapping = idx_mapping.long().contiguous()
    UC.ext.ucache_fold(mode, int(num_rows), idx_mapping, kw.get("state_idx"), kw.get("src_col"), kw.get("src_off"),
                       kw.get("num_accepted"), kw.get("num_computed"), kw.get("seq_lens"), t.block_size, t.bt_ptrs,
                       t.bt_stride, t.ptrs, t.group, t.slot_stride, t.H, t.HV, UC.active, RING_F16)


def fold_r1(num_reqs: int, state_idx, src_col, src_off, idx_mapping) -> None:
    """Before the stock align precopy (run_fused_precopy): active slots' temporal state."""
    if ENABLED:
        _fold(1, num_reqs, idx_mapping, state_idx=state_idx, src_col=src_col, src_off=src_off)


def fold_r2(num_reqs: int, num_accepted, state_idx, new_num_computed, idx_mapping) -> None:
    """Before the stock align postprocess (run_fused_postprocess_align, which then resets acc on in-place rows)."""
    if ENABLED:
        _fold(2, num_reqs, idx_mapping, state_idx=state_idx, num_accepted=num_accepted, num_computed=new_num_computed)


def prepass(num_spec: int, idx_mapping, seq_lens, num_accepted_by_req) -> None:
    """MambaHybridModelState.prepare_attn (real steps, after R1, before the forward; outside CUDA graphs)."""
    if not ENABLED or num_spec <= 0:
        return
    if STEP.band:
        _fold(3, num_spec, idx_mapping, seq_lens=seq_lens, num_accepted=num_accepted_by_req)
    else:
        _fold(4, num_spec, idx_mapping, seq_lens=seq_lens, num_accepted=num_accepted_by_req)
        if _table() is not None:
            UC.ext.ucache_clear(idx_mapping.long().contiguous(), int(num_spec), UC.active)


def active_or_dummy(device) -> torch.Tensor:
    """The stock align kernels' skip mask (UC_SKIP); a 1-element zero tensor when off."""
    if UC.active is not None:
        return UC.active
    return torch.zeros(1, dtype=torch.int32, device=device)


# ------------------------------------------------------------------------------------------------- CHECK
def snapshot(layers) -> list:
    """VLLM_GDN_LAYER_GRAPHS_CHECK: rings / cursors / tags of `layers` plus the active bits (clone)."""
    if not ENABLED:
        return []
    out = [UC.active.clone()]
    for lay in layers:
        b = getattr(lay, "_gdn_ucache", None)
        if b is None:
            continue
        out.append((lay, {k: v.clone() for k, v in b.items() if isinstance(v, torch.Tensor)}))
    return out


def restore(snap: list) -> None:
    if not snap:
        return
    UC.active.copy_(snap[0])
    for lay, saved in snap[1:]:
        for k, v in saved.items():
            lay._gdn_ucache[k].copy_(v)
