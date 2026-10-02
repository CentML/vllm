# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[F122] Worker side of the in-step GDN prefill checkpoints (VLLM_MAMBA_TAIL_CKPT=1).

The scheduler ran a prompt chunk [start, end) through positions whose GDN state
the align-mode prefix cache needs (see vllm/v1/core/sched/mamba_inline_ckpt.py):

  kind 0 (block boundary b): the state at b belongs in the block-table column
      that held the chunk's initial state (block X). The align-mode pre-copy
      moved the state at `start` to the running column, so X still holds it.
  kind 1 (partial tail T): a reserved block D, pre-filled before the forward
      with the page (conv | ssm | log) holding the state at `start` (or zero for
      a fresh prompt).

The step's GDN core is unchanged (CUDA graph or eager, whatever path the step
takes). After each GDN layer's core, every checkpoint is produced by re-running
that layer's split-flow chunk in place on its block: the CUDA conv + post-conv
over the rows [start, pos) with the block's conv state as the initial window
(it writes the conv window at `pos`, as the split flow's chunk does), then the
chunked delta rule over the same rows with the block's SSM state as the initial
state (pool in place). Same kernels, same chunk grid from `start`, same inputs
and initial state as the split flow's chunk that ends at `pos`, so the stored
states and conv windows are the split flow's. The replayed outputs are thrown
away; the log flags of X / D stay clean (copied from a committed page).

VLLM_MAMBA_TAIL_CKPT_VERIFY=N: for the first N (layer, request) pairs, re-run the
split flow as a chain of segments [start, c1) [c1, c2) ... [cn, end) on a scratch
pool from the state at `start` and compare every checkpoint page with it bitwise
(ssm and conv window); the final state is compared with the running block and the
chain's outputs with the step's outputs (max |diff|, informational: the step ran
the row unsplit).
"""

import os

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_MAMBA_TAIL_CKPT", "0") == "1"
VERIFY = [int(os.environ.get("VLLM_MAMBA_TAIL_CKPT_VERIFY", "0") or 0)]
LOG_EVERY = max(1, int(os.environ.get("VLLM_MAMBA_TAIL_CKPT_LOG_EVERY", "2000") or 2000))
# >= 2 checkpoints of one KV-cache group in a step: one gathered conv + chunk launch
# per layer instead of one pair per checkpoint (same per-sequence math)
BATCH = os.environ.get("VLLM_MAMBA_TAIL_CKPT_BATCH", "1") == "1"

KIND_BLOCK, KIND_TAIL, KIND_RUN, KIND_SPLIT = 0, 1, 2, 3  # = mamba_inline_ckpt.KIND_*

STATS = {
    "steps": 0,
    "rows": 0,
    "block_ckpts": 0,
    "tail_ckpts": 0,
    "zero_init": 0,
    "replays": 0,
    "replay_tokens": 0,
    "verify_layers": 0,
    "verify_mismatch": 0,
}

_LAYER_GROUP: dict[str, int] = {}
_MODS: dict = {}


class _Step:
    """One forward's checkpoints. rows: per request (a0, start, end, ckpts), a0 =
    the request's first row in the step's token batch. per_group: gid -> the
    records (a0, slot, kind, pos, start, end, zero_init) (VERIFY); waves: gid ->
    list of launches; copies: gid -> page copies (src, dst) after the first wave;
    zeros: gid -> pages to zero-fill before the first wave."""

    __slots__ = ("rows", "dev", "per_group", "waves", "copies", "zeros", "bufs",
                 "lmax", "_keep", "const", "stream", "jobs", "graph", "short", "pin")

    def __init__(self, rows, dev):
        self.rows = rows
        self.dev = dev
        self.per_group: dict[int, list] = {}
        self.waves: dict[int, list] = {}
        self.copies: dict[int, list] = {}
        self.zeros: dict[int, list] = {}
        self.bufs = None
        self.lmax = 0
        self._keep = []
        self.const = None  # (cu3 + count, has_init False) for the 3-row conv-window launch
        self.stream = None  # cuda-python stream handle for the direct kernel calls
        self.jobs: dict[int, list] = {}  # gid -> _plan_jobs
        self.graph: dict[int, tuple] = {}  # gid -> replay-graph spec (graph-served groups)
        self.short = False  # a segment shorter than the conv window (3 rows): no fast path
        self.pin = False


_STEP: list = [None]


def _gdn():
    m = _MODS.get("gdn")
    if m is None:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as m

        _MODS["gdn"] = m
    return m


def _vsplit(devnb: bool):
    key = "vs_devnb" if devnb else "vs"
    m = _MODS.get(key)
    if m is None:
        if devnb:
            import vllm.third_party.flashinfer_gdn_vsplit_devnb as m
        else:
            import vllm.third_party.flashinfer_gdn_vsplit as m
        _MODS[key] = m
    return m


def _use_devnb() -> bool:
    """The kernel copy the split flow's chunks run on at low concurrency: the
    GDN layer graphs' device_nb V-split copy (compiled at graph capture), else
    the regular V-split copy (both bitwise identical by construction)."""
    v = _MODS.get("devnb_on")
    if v is None:
        try:
            from vllm.model_executor.layers.mamba.gdn import gdn_layer_graphs

            v = _MODS["devnb_on"] = bool(gdn_layer_graphs.ENABLED and gdn_layer_graphs.DEVNB)
        except ImportError:
            # [F122] adapt: the layer-graph module (row 14 / #88 family) may not
            # be present; without it there is no device_nb V-split copy to
            # prefer (the two copies are bitwise identical by construction).
            v = _MODS["devnb_on"] = False
    return v


def _conv():
    m = _MODS.get("conv")
    if m is None:
        from vllm.model_executor.layers.mamba.ops import gdn_conv_cuda as m

        _MODS["conv"] = m
    return m


def set_layer_groups(kv_cache_config) -> None:
    """layer name -> KV-cache group id for the GDN (Mamba) groups."""
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    _LAYER_GROUP.clear()
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            names = [n for n, s in spec.kv_cache_specs.items() if isinstance(s, MambaSpec)]
        elif isinstance(spec, MambaSpec):
            names = list(group.layer_names)
        else:
            continue
        for n in names:
            _LAYER_GROUP[n] = gid


def end_step() -> None:
    _STEP[0] = None


def _plan_jobs(rows, gid):
    """Per request of the step: the split flow's chunk chain that produces this
    group's checkpoints. A tail checkpoint D (pre-filled with the state at start)
    replays [start, b) [b, T) when a block boundary b lies before T (the split
    flow restarts its chunk grid at b), else [start, T); a cached block state X
    is a page copy of D after [start, b), or its own replay [start, b) in place
    when there is no tail checkpoint. Returns jobs (target slot, zero_init,
    [(row offset, length, has_init), ...], copy-after-first-segment slot)."""
    jobs = []
    for a0, start, end, cks in rows:
        b = 0
        x = d = None
        zero = False
        for pos, kind, zero_init, blocks in cks:
            if kind in (KIND_BLOCK, KIND_SPLIT):
                b = pos
                if kind == KIND_BLOCK:
                    x = blocks[gid]
            elif kind == KIND_TAIL:
                t, d, zero = pos, blocks[gid], bool(zero_init)
        if d is not None:
            cuts = [start] + ([b] if start < b < t else []) + [t]
            segs = [
                (a0 + lo - start, hi - lo, (lo > start) or (start > 0 and not zero))
                for lo, hi in zip(cuts, cuts[1:])
            ]
            copy_to = x if (x is not None and len(cuts) == 3) else None
            jobs.append((d, zero, segs, copy_to))
            if x is not None and copy_to is None:
                raise RuntimeError(f"[F122] block checkpoint {b} not before tail {t}")
        elif x is not None:
            jobs.append((x, False, [(a0, b - start, start > 0)], None))
    return jobs


def _build_waves(st, jobs):
    """Eager launch plan of a group's replay (device metadata per launch)."""
    device = st.dev
    pin = st.pin

    def dev_i32(vals):
        t = torch.tensor(vals, dtype=torch.int32)
        if pin:
            t = t.pin_memory()
        d = t.to(device, non_blocking=True)
        st._keep += [t, d]
        return d

    def dev_bool(vals):
        t = torch.tensor(vals, dtype=torch.bool)
        if pin:
            t = t.pin_memory()
        d = t.to(device, non_blocking=True)
        st._keep += [t, d]
        return d

    if st.const is None:
        c = dev_i32([0, 3, 1])
        st.const = (c, dev_bool([False]))
    waves = []
    for w in range(max(len(j[2]) for j in jobs)):
        segs = [(j[0], j[2][w]) for j in jobs if len(j[2]) > w]
        ones = []
        for slot, (r0, L, hi) in segs:
            m = dev_i32([slot, 0, L, 1])  # slot | cu_seqlens | device_nb count
            ones.append((r0, L, m[0:1], dev_bool([hi]), m[1:3]))
            st.lmax = max(st.lmax, L)
        if len(segs) == 1 or not BATCH:
            waves.append(("one", ones))
        else:
            idx = torch.cat(
                [torch.arange(r0, r0 + L, dtype=torch.int64) for _, (r0, L, _) in segs]
            )
            if pin:
                idx = idx.pin_memory()
            idx_d = idx.to(device, non_blocking=True)
            st._keep += [idx, idx_d]
            cu = [0]
            for _, (_, L, _) in segs:
                cu.append(cu[-1] + L)
            R = len(segs)
            m = dev_i32([slot for slot, _ in segs] + cu + [R])
            # fast path: the last 3 rows of every segment (conv window at its cut)
            idx3 = torch.tensor(
                [r0 + L - 3 + j for _, (r0, L, _) in segs for j in range(3)], dtype=torch.int64
            )
            if pin:
                idx3 = idx3.pin_memory()
            idx3_d = idx3.to(device, non_blocking=True)
            st._keep += [idx3, idx3_d]
            c3 = dev_i32([3 * i for i in range(R + 1)] + [R])
            f3 = dev_bool([False] * R)
            waves.append(
                (
                    "batch",
                    (idx_d, m[:R], dev_bool([hi for _, (_, _, hi) in segs]),
                     m[R : 2 * R + 1], R, cu[R], max(L for _, (_, L, _) in segs), ones,
                     idx3_d, c3, f3),
                )
            )
            st.lmax = max(st.lmax, cu[R])
        if w == 0:
            waves.append(("copy", None))
    return waves


def begin_step(scheduler_output, input_batch, kv_cache_config, device) -> None:
    """Model runner, after prepare_inputs / the align pre-copy and before the
    forward: map this step's checkpoints onto token rows and launch plans."""
    _STEP[0] = None
    ck = getattr(scheduler_output, "mamba_inline_ckpts", None) if scheduler_output else None
    if not ck:
        return
    if not _LAYER_GROUP:
        set_layer_groups(kv_cache_config)
    req_ids = input_batch.req_ids
    qsl = input_batch.query_start_loc_np
    nst = input_batch.num_scheduled_tokens
    rows = []
    for req_id, (start, end, ckpts) in ck.items():
        i = req_ids.index(req_id)
        a0 = int(qsl[i])
        n = int(nst[i])
        if n != end - start:
            raise RuntimeError(
                f"[F122] request {req_id}: scheduled {n} tokens, checkpoint plan "
                f"[{start}, {end})"
            )
        rows.append((a0, start, end, ckpts))
    st = _Step(rows, device)
    gids = sorted({g for _, _, _, cks in rows for c in cks for g in c[3]})
    st.pin = torch.cuda.is_available()
    for gid in gids:
        st.per_group[gid] = [
            (a0, blocks[gid], kind, pos, start, end, bool(zero_init))
            for a0, start, end, cks in rows
            for pos, kind, zero_init, blocks in cks
        ]
        jobs = _plan_jobs(rows, gid)
        if not jobs:
            continue
        st.jobs[gid] = jobs
        st.zeros[gid] = [j[0] for j in jobs if j[1]]
        st.copies[gid] = [(j[0], j[3]) for j in jobs if j[3] is not None]
        st.short |= any(L < 3 for j in jobs for _, L, _ in j[2])
        spec = _graph_spec(jobs) if GRAPH and not st.short else None
        if spec is not None:
            st.graph[gid] = spec
        else:
            st.waves[gid] = _build_waves(st, jobs)
    if st.graph:
        _fill_graph_meta(st, device, len(kv_cache_config.kv_cache_groups))
    _STEP[0] = st
    STATS["steps"] += 1
    STATS["rows"] += len(rows)
    for _, _, _, cks in rows:
        for c in cks:
            STATS["block_ckpts"] += int(c[1] == KIND_BLOCK)
            STATS["block_splits"] = STATS.get("block_splits", 0) + int(c[1] == KIND_SPLIT)
            STATS["tail_ckpts"] += int(c[1] == KIND_TAIL)
            STATS["zero_init"] += int(c[1] == KIND_TAIL and c[2])
    if STATS["steps"] <= 3 or STATS["steps"] % LOG_EVERY == 0:
        logger.info(
            "[F122] worker step #%d rows=%s devnb=%d stats=%s",
            STATS["steps"],
            [(a0, s_, e, [(c[0], c[1], c[2]) for c in cks]) for a0, s_, e, cks in rows],
            int(_use_devnb()),
            dict(STATS),
        )


def _pages_u8(layer):
    conv, ssm = layer.kv_cache[0], layer.kv_cache[1]
    page = ssm.stride(0) * ssm.element_size()
    assert conv.stride(0) * conv.element_size() == page
    stg = ssm.untyped_storage()
    off = min(conv.data_ptr(), ssm.data_ptr()) - stg.data_ptr()
    n = min(ssm.size(0), (stg.nbytes() - off) // page)
    v = torch.empty(0, dtype=torch.uint8, device=ssm.device)
    v.set_(stg, off, (n, page), (page, 1))
    return v


def _layer_views(layer):
    mod = _gdn()
    conv_state = (
        layer.kv_cache[0]
        if mod.is_conv_state_dim_first()
        else layer.kv_cache[0].transpose(-1, -2)
    )
    conv_w = layer.conv1d.weight.view(
        layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
    )
    return conv_state, layer.kv_cache[1], conv_w


def run_chunk(layer, x, a, b, conv_state, ssm, slot_t, hi_t, cu_t, L, bufs=None,
              ws=None, devnb=None, nseq=1, maxlen=None):
    """The split flow's GDN chunk over the rows x/a/b ([L, ...]) in place on the
    pool slot `slot_t`: CUDA conv + post-conv (initial conv window from the slot
    if hi_t, final window written to the slot), then the chunked delta rule with
    the slot's state as the initial state (pool in place). cu_t = [0, L] (with the
    device_nb count 1 right after it). Returns the (thrown-away) outputs."""
    mod = _gdn()
    H = layer.num_k_heads // layer.tp_size
    HV = layer.num_v_heads // layer.tp_size
    K, V = layer.head_k_dim, layer.head_v_dim
    conv_w = layer.conv1d.weight.view(
        layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
    )
    dev = x.device
    if bufs is None:
        bufs = _alloc_bufs(L, H, HV, K, V, x.dtype, dev)
    q, k, v, g, beta, out = (t[:L] for t in bufs)
    gcc = _conv()
    if mod._GDN_CONV_CUDA_TPH == "auto":
        tph = 4 if L < 1024 * int(nseq) else 8
    else:
        tph = int(mod._GDN_CONV_CUDA_TPH)
    ok = gcc.load().run(
        x,
        conv_w,
        conv_state,
        slot_t,
        hi_t,
        cu_t,
        int(nseq),
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
        int(tph),
        int(gcc.RV),
    )
    if not ok:
        STATS["conv_fallback"] = STATS.get("conv_fallback", 0) + 1
        q, k, v, g, beta = mod.gdn_fused_conv_post_conv(
            x,
            conv_w,
            conv_state,
            slot_t,
            hi_t,
            cu_t,
            int(nseq),
            a,
            b,
            layer.A_log,
            layer.dt_bias,
            H,
            K,
            V,
            use_cuda=False,
        )
    scale = 1.0 / (K**0.5)
    if devnb is None:
        devnb = _use_devnb()
    vs = _vsplit(devnb)
    vsf = max(int(vs.choose_vsplit(int(nseq), L, int(maxlen or L), hv=HV)), 1)
    if devnb:
        if ws is None:
            from vllm.third_party.flashinfer_gdn_vsplit_devnb.gdn_chunked_vs import (
                GatedDeltaNetChunkedKernel as _K,
            )

            nsm = torch.cuda.get_device_properties(dev).multi_processor_count
            ws = torch.empty(
                _K.get_workspace_size(nsm, int(nseq), q.size(1), v.size(1), True),
                dtype=torch.int8,
                device=dev,
            )
        vs.chunk_gated_delta_rule_vsplit(
            q, k, v, g, beta, out, cu_t, ssm, ssm, scale,
            state_indices=slot_t, v_split=vsf, device_nb=True, workspace=ws,
        )
    else:
        vs.chunk_gated_delta_rule_vsplit(
            q, k, v, g, beta, out, cu_t, ssm, ssm, scale,
            state_indices=slot_t, v_split=vsf,
        )
    return out


def _alloc_bufs(L, H, HV, K, V, dtype, dev):
    return (
        torch.empty(L, H, K, dtype=dtype, device=dev),
        torch.empty(L, H, K, dtype=dtype, device=dev),
        torch.empty(L, HV, V, dtype=dtype, device=dev),
        torch.empty(L, HV, dtype=torch.float32, device=dev),
        torch.empty(L, HV, dtype=torch.float32, device=dev),
        torch.empty(L, HV, V, dtype=dtype, device=dev),
    )


# graph fast path (VLLM_MAMBA_TAIL_CKPT_FAST=1, default): layers served by a GDN layer graph
FAST = os.environ.get("VLLM_MAMBA_TAIL_CKPT_FAST", "1") == "1"
_DIRECT: dict = {}


def _graph_conv_outputs(layer, mixed_qkv):
    """(q, k, v, g, beta) with this step's conv outputs of `layer` at absolute rows,
    if its core was served by a GDN layer graph in this call, else None."""
    if not getattr(layer, "_f122_graph", False):
        return None
    from vllm.model_executor.layers.mamba.gdn import gdn_layer_graphs

    H = layer.num_k_heads // layer.tp_size
    HV = layer.num_v_heads // layer.tp_size
    sh = gdn_layer_graphs._SHARED.get((mixed_qkv.device, H, HV))
    if sh is None or sh[0].size(0) < mixed_qkv.size(0):
        return None
    return sh


def _page_copies(layer, st, gid):
    cps = st.copies.get(gid)
    if cps:
        pages = _pages_u8(layer)
        for src, dst in cps:
            pages[dst].copy_(pages[src])
        STATS["page_copies"] = STATS.get("page_copies", 0) + len(cps)


def _direct_kernel(q, v, ssm, slot_t, vsf, nseq=1):
    """The device_nb V-split adapter's compiled kernel for this key (compiled by the
    GDN layer graphs at capture), or None. Same key formula as the adapter."""
    from vllm.third_party.flashinfer_gdn_vsplit_devnb import adapter as ad

    HQ, HV = q.size(1), v.size(1)
    dev = q.device.index if q.device.index is not None else torch.cuda.current_device()
    key = (dev, ad._num_sm(dev), str(q.dtype), str(ssm.dtype), HQ, HV, HQ >= HV, True, True, True,
           str(slot_t.dtype), tuple(ssm.stride()[1:]), tuple(ssm.stride()[1:]), int(vsf), True,
           ad._cg0_split(vsf, True), bool(ad._C1_REORDER))
    hit = _DIRECT.get((key, int(nseq)))
    if hit is None:
        c = ad._cache(*key)
        if "compiled" not in c:
            return None
        ws = torch.empty(
            ad.GatedDeltaNetChunkedKernel.get_workspace_size(ad._num_sm(dev), int(nseq), HQ, HV, True),
            dtype=torch.int8, device=q.device)
        hit = _DIRECT[(key, int(nseq))] = (c["compiled"], ws)
    return hit


def _chunk_shared(layer, shared, r0, L, ssm, slot_t, cu_t, st):
    """Chunked delta rule over the absolute rows [r0, r0 + L) of the layer graph's
    conv outputs, in place on the pool slot (initial state = the slot's state)."""
    q, k, v, g, beta = (t[r0 : r0 + L] for t in shared)
    HV = layer.num_v_heads // layer.tp_size
    _run_direct(q, k, v, g, beta, st.bufs[5][:L], cu_t, ssm, slot_t, 1, L, st, HV,
                1.0 / (layer.head_k_dim**0.5))


def _run_direct(q, k, v, g, beta, out, cu, ssm, slots, nseq, maxlen, st, HV, scale):
    vs = _vsplit(True)
    vsf = max(int(vs.choose_vsplit(int(nseq), q.size(0), int(maxlen), hv=HV)), 1)
    d = _direct_kernel(q, v, ssm, slots, vsf, nseq)
    if d is not None:
        if st.stream is None:
            import cuda.bindings.driver as cuda

            st.stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
        d[0](q, k, v, g, beta, out, cu, ssm, ssm, slots, None, None, 0, scale, d[1], st.stream)
        STATS["direct_calls"] = STATS.get("direct_calls", 0) + 1
    else:
        vs.chunk_gated_delta_rule_vsplit(q, k, v, g, beta, out, cu, ssm, ssm, scale,
                                         state_indices=slots, v_split=vsf, device_nb=True)
        STATS["adapter_calls"] = STATS.get("adapter_calls", 0) + 1


def _chunk_gathered(layer, shared, idx_d, Ltot, R, maxl, ssm, slots_d, cu_d, st):
    q, k, v, g, beta = (t.index_select(0, idx_d) for t in shared)
    HV = layer.num_v_heads // layer.tp_size
    _run_direct(q, k, v, g, beta, st.bufs[5][:Ltot], cu_d, ssm, slots_d, R, maxl, st, HV,
                1.0 / (layer.head_k_dim**0.5))


def _conv_windows_gathered(layer, mixed_qkv, a, b, conv_state, idx3_d, R, slots_d, c3, f3, st):
    H = layer.num_k_heads // layer.tp_size
    q, k, v, g, beta = (t[: 3 * R] for t in st.bufs[:5])
    conv_w = layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2))
    gcc = _conv()
    ok = gcc.load().run(mixed_qkv.index_select(0, idx3_d), conv_w, conv_state, slots_d, f3,
                        c3[: R + 1], int(R), a.index_select(0, idx3_d), b.index_select(0, idx3_d),
                        layer.A_log, layer.dt_bias, q, k, v, g, beta, int(H), 4, int(gcc.RV))
    if not ok:
        raise RuntimeError("[F122] CUDA conv refused the gathered conv-window launch")


def _conv_window(layer, mixed_qkv, a, b, conv_state, end, slot_t, st):
    """The conv window at `end` (last 3 inputs x[end-3:end], as the conv kernel stores
    it at a sequence end) into the slot: the conv kernel over those 3 rows alone."""
    cu3, hi0 = st.const
    H = layer.num_k_heads // layer.tp_size
    q, k, v, g, beta = (t[:3] for t in st.bufs[:5])
    conv_w = layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2))
    gcc = _conv()
    ok = gcc.load().run(mixed_qkv[end - 3 : end], conv_w, conv_state, slot_t, hi0, cu3[:2], 1,
                        a[end - 3 : end], b[end - 3 : end], layer.A_log, layer.dt_bias,
                        q, k, v, g, beta, int(H), 4, int(gcc.RV))
    if not ok:
        raise RuntimeError("[F122] CUDA conv refused the 3-row conv-window launch")


# ----------------------------------------------------------------------------------------------
# Replay graphs (VLLM_MAMBA_TAIL_CKPT_GRAPH=1, default). On a layer-graph step, the replay of one
# layer is one row gather (the last 3 input rows before every cut) plus one CUDA graph replay: page
# zero-fills, the first wave's device_nb chunk launches over absolute rows of the shared conv-output
# buffers (full T_MAX views: the graph does not depend on the step's token count), the page copies
# D -> X, the second wave's chunk launches, then one conv-window launch writing every cut's window
# (X's window at b after the copy, D's at T). All per-step values (rows, slots, pages) are device
# metadata, written for every group with one host-to-device copy in begin_step; a graph is keyed by
# the layer and the step's launch counts, captured on first use. Same kernels as the eager fast path;
# fixed v_split 2 (the V-split kernel is bitwise identical across v_split); the conv-window launch
# gets zero a / b rows (they only feed its thrown-away g / beta outputs, not the window).
# ----------------------------------------------------------------------------------------------
GRAPH = os.environ.get("VLLM_MAMBA_TAIL_CKPT_GRAPH", "1") == "1"
R_MAX, W_MAX, Z_MAX, C_MAX = 4, 2, 4, 4
NW_MAX = R_MAX + C_MAX  # conv windows: one per job + one per page copy
_O_CU = 0  # (w, r) -> [row0, row0 + L, 1 (device_nb count)]
_O_SLOT = _O_CU + W_MAX * R_MAX * 3  # (w, r) -> slot of the chunk launch
_O_CSLOT = _O_SLOT + W_MAX * R_MAX  # window i -> slot
_O_CU3 = _O_CSLOT + NW_MAX  # [0, 3, ..., 3 NW_MAX]: conv-window cu_seqlens
_O_IDX = _O_CU3 + NW_MAX + 1  # window i -> its 3 input rows
_O_CP = _O_IDX + 3 * NW_MAX  # [src, dst] x C_MAX page copies after wave 0
_O_Z = _O_CP + 2 * C_MAX  # Z_MAX pages zero-filled first
_META = (_O_Z + Z_MAX + 3) // 4 * 4
_GR: dict = {}  # (layer, spec, addresses) -> CUDAGraph
_GBUF: dict = {}  # device index -> _GraphBufs


class _GraphBufs:
    """Per device: the groups' replay metadata (created in begin_step) and, from the first
    graph-served layer on, the static buffers the graphs use."""

    def __init__(self, dev, n_groups):
        self.dev = dev
        self.meta = torch.zeros(n_groups, _META, dtype=torch.int32, device=dev)
        self.meta_np = np.zeros((n_groups, _META), dtype=np.int32)
        self.ready = False

    def setup(self, layer, shared, wz, wba):
        dev = self.dev
        H = layer.num_k_heads // layer.tp_size
        HV = layer.num_v_heads // layer.tp_size
        n = 3 * NW_MAX
        self.sz = torch.zeros(n, wz, dtype=torch.bfloat16, device=dev)  # gathered input rows
        self.ab = torch.zeros(n, wba, dtype=torch.bfloat16, device=dev)  # zero ba rows (split_ba)
        # chunk outputs (thrown away) at absolute rows; conv-window outputs (thrown away)
        self.out = torch.empty(shared[0].size(0), HV, layer.head_v_dim, dtype=torch.bfloat16,
                               device=dev)
        self.c3 = _alloc_bufs(n, H, HV, layer.head_k_dim, layer.head_v_dim, torch.bfloat16, dev)[:5]
        self.f3 = torch.zeros(NW_MAX, dtype=torch.bool, device=dev)
        self.stream = torch.cuda.Stream(device=dev)
        self.pool = torch.cuda.graph_pool_handle()
        self.ready = True


def _dev_key(device):
    d = torch.device(device)
    return d.index if d.index is not None else torch.cuda.current_device()


def _graph_spec(jobs):
    """(launches per wave, zero-fills, page copies, conv windows) if the step fits the graphs."""
    W = max(len(j[2]) for j in jobs)
    counts = tuple(sum(1 for j in jobs if len(j[2]) > w) for w in range(W))
    nz = sum(1 for j in jobs if j[1])
    nc = sum(1 for j in jobs if j[3] is not None)
    if W > W_MAX or counts[0] > R_MAX or nz > Z_MAX or nc > C_MAX:
        STATS["graph_misfit"] = STATS.get("graph_misfit", 0) + 1
        return None
    return counts, nz, nc, len(jobs) + nc


def _fill_graph_meta(st, device, n_groups):
    dk = _dev_key(device)
    gb = _GBUF.get(dk)
    if gb is None:
        gb = _GBUF[dk] = _GraphBufs(torch.device("cuda", dk), n_groups)
    m = gb.meta_np
    for gid in st.graph:
        jobs = st.jobs[gid]
        row = m[gid]
        row[:] = 0  # unused entries: row 0 / slot 0 (the null block)
        row[_O_CU3 : _O_CU3 + NW_MAX + 1] = np.arange(NW_MAX + 1, dtype=np.int32) * 3
        for w in range(max(len(j[2]) for j in jobs)):
            n = 0
            for j in jobs:
                if len(j[2]) <= w:
                    continue
                r0, L, _ = j[2][w]
                o = _O_CU + (w * R_MAX + n) * 3
                row[o : o + 3] = (r0, r0 + L, 1)
                row[_O_SLOT + w * R_MAX + n] = j[0]
                n += 1
        wins = []  # (slot, end row): X's window at b (D chains with a copy), each job's last cut
        for j in jobs:
            if j[3] is not None:
                wins.append((j[3], j[2][0][0] + j[2][0][1]))
            r0, L, _ = j[2][-1]
            wins.append((j[0], r0 + L))
        for i, (slot, e) in enumerate(wins):
            row[_O_CSLOT + i] = slot
            row[_O_IDX + 3 * i : _O_IDX + 3 * i + 3] = (e - 3, e - 2, e - 1)
        for i, (src, dst) in enumerate(st.copies[gid]):
            row[_O_CP + 2 * i : _O_CP + 2 * i + 2] = (src, dst)
        for i, slot in enumerate(st.zeros[gid]):
            row[_O_Z + i] = slot
    h = torch.empty(m.shape, dtype=torch.int32, pin_memory=st.pin)
    h.numpy()[:] = m
    gb.meta.copy_(h, non_blocking=True)
    st._keep.append(h)


@triton.jit
def _page_copy_kernel(words, lst, page_words, stride_words, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    t = tl.program_id(1)
    src = tl.load(lst + 2 * i).to(tl.int64)
    dst = tl.load(lst + 2 * i + 1).to(tl.int64)
    offs = t * BLOCK + tl.arange(0, BLOCK)
    m = offs < page_words
    v = tl.load(words + src * stride_words + offs, mask=m)
    tl.store(words + dst * stride_words + offs, v, mask=m)


@triton.jit
def _page_zero_kernel(words, lst, page_words, stride_words, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    t = tl.program_id(1)
    dst = tl.load(lst + i).to(tl.int64)
    offs = t * BLOCK + tl.arange(0, BLOCK)
    m = offs < page_words
    tl.store(words + dst * stride_words + offs, tl.zeros([BLOCK], dtype=tl.int32), mask=m)


_PAGE_BLOCK = 4096


def _graph_bufs(layer, gb, shared, mq_full, ba):
    if mq_full.dtype != torch.bfloat16 or ba.dtype != torch.bfloat16:
        return None
    if not gb.ready:
        gb.setup(layer, shared, mq_full.size(1), ba.size(1))
        # compile the page kernels outside any capture, with the real page view and list
        # alignment: copy the null block (slot 0) onto itself, zero-fill it
        words = _pages_u8(layer).view(torch.int32)
        lst = torch.zeros(_META, dtype=torch.int32, device=gb.dev)
        _page_copy_kernel[(1, 1)](words, lst[_O_CP:], words.size(1), words.stride(0),
                                  BLOCK=_PAGE_BLOCK, num_warps=4)
        _page_zero_kernel[(1, 1)](words, lst[_O_Z:], words.size(1), words.stride(0),
                                  BLOCK=_PAGE_BLOCK, num_warps=4)
    if mq_full.size(1) != gb.sz.size(1) or ba.size(1) != gb.ab.size(1):
        return None
    return gb


def _graph_body(layer, gid, spec, gb, conv_state, ssm, shared, kern, cstream):
    counts, nz, nc, nw = spec
    row = gb.meta[gid]
    H = layer.num_k_heads // layer.tp_size
    qkv_size = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
    scale = 1.0 / (layer.head_k_dim**0.5)
    conv_w = layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2))
    words = _pages_u8(layer).view(torch.int32)
    pw = words.size(1)
    grid_t = triton.cdiv(pw, _PAGE_BLOCK)
    q, k, v, g, beta = shared
    if nz:
        _page_zero_kernel[(nz, grid_t)](words, row[_O_Z:], pw, words.stride(0), BLOCK=_PAGE_BLOCK,
                                        num_warps=4)
    for w, n in enumerate(counts):
        for r in range(n):
            o = _O_CU + (w * R_MAX + r) * 3
            sl = _O_SLOT + w * R_MAX + r
            kern[0](q, k, v, g, beta, gb.out, row[o : o + 2], ssm, ssm, row[sl : sl + 1], None, None,
                    0, scale, kern[1], cstream)
        if w == 0 and nc:
            _page_copy_kernel[(nc, grid_t)](words, row[_O_CP:], pw, words.stride(0),
                                            BLOCK=_PAGE_BLOCK, num_warps=4)
    b3, a3 = layer.split_ba(gb.ab[: 3 * nw])
    q3, k3, v3, g3, be3 = (t[: 3 * nw] for t in gb.c3)
    ok = _conv().load().run(gb.sz[: 3 * nw, :qkv_size], conv_w, conv_state,
                            row[_O_CSLOT : _O_CSLOT + nw], gb.f3[:nw], row[_O_CU3 : _O_CU3 + nw + 1],
                            nw, a3, b3, layer.A_log, layer.dt_bias, q3, k3, v3, g3, be3, int(H), 4,
                            int(_conv().RV))
    assert ok, "[F122] CUDA conv refused the conv-window launch (replay graph)"


def _graph_replay(layer, gid, spec, conv_state, ssm, shared, mq_full, ba) -> bool:
    """This layer's replay of the step: one row gather + one graph replay; False if the graph
    path is unavailable for it."""
    gb = _GBUF.get(_dev_key(ssm.device))
    if gb is None or gid >= gb.meta.size(0):
        return False
    key = (layer.prefix, spec, shared[0].data_ptr(), ssm.data_ptr(), conv_state.data_ptr(),
           gb.meta.data_ptr(), mq_full.size(1), ba.size(1))
    gr = _GR.get(key)
    if gr is None:
        if _pages_u8(layer).size(1) % 4 or _graph_bufs(layer, gb, shared, mq_full, ba) is None:
            return False
        kern = _direct_kernel(shared[0], shared[2], ssm, gb.meta[gid, _O_SLOT : _O_SLOT + 1], 2, 1)
        if kern is None:
            STATS["graph_nokernel"] = STATS.get("graph_nokernel", 0) + 1
            return False
        import cuda.bindings.driver as cuda

        gr = torch.cuda.CUDAGraph()
        s = gb.stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            gr.capture_begin(pool=gb.pool, capture_error_mode="thread_local")
            try:
                _graph_body(layer, gid, spec, gb, conv_state, ssm, shared, kern,
                            cuda.CUstream(s.cuda_stream))
            finally:
                gr.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        _GR[key] = gr
        STATS["graphs_captured"] = STATS.get("graphs_captured", 0) + 1
    n = 3 * spec[3]
    torch.index_select(mq_full, 0, gb.meta[gid, _O_IDX : _O_IDX + n], out=gb.sz[:n])
    gr.replay()
    return True


def after_core(layer, mixed_qkv, b, a, core_attn_out=None, raw=None) -> None:
    """End of the GDN core custom op of `layer` (any path): write this step's
    in-step checkpoints of the layer's KV-cache group."""
    st = _STEP[0]
    if st is None:
        return
    gid = _LAYER_GROUP.get(layer.prefix)
    if gid is None:
        raise RuntimeError(f"[F122] GDN layer {layer.prefix} has no KV-cache group")
    if gid not in st.jobs:
        return
    conv_state, ssm, _ = _layer_views(layer)
    pages = None
    verify = VERIFY[0] > 0 and not torch.cuda.is_current_stream_capturing()
    recs = st.per_group.get(gid, [])
    snaps = None
    if verify:
        pages = _pages_u8(layer)
        snaps = {r[1]: pages[r[1]].clone() for r in recs if r[2] != KIND_SPLIT}
    spec = st.graph.get(gid)
    if spec is not None and raw is not None:
        shared = _graph_conv_outputs(layer, mixed_qkv)
        if shared is not None and _graph_replay(layer, gid, spec, conv_state, ssm, shared, *raw):
            STATS["graph_layers"] = STATS.get("graph_layers", 0) + 1
            STATS["replays"] += sum(spec[0])
            if verify:
                _verify(layer, st, gid, recs, snaps, mixed_qkv, b, a, core_attn_out)
            return
    waves = st.waves.get(gid)
    if waves is None:
        waves = st.waves[gid] = _build_waves(st, st.jobs[gid])
        STATS["graph_fallback_layers"] = STATS.get("graph_fallback_layers", 0) + 1
    if st.bufs is None:
        H = layer.num_k_heads // layer.tp_size
        HV = layer.num_v_heads // layer.tp_size
        # one set of conv-output / scratch-output buffers per step, reused by
        # every layer and launch (stream-ordered)
        st.bufs = _alloc_bufs(
            st.lmax, H, HV, layer.head_k_dim, layer.head_v_dim, mixed_qkv.dtype,
            mixed_qkv.device,
        )
    zeros = st.zeros.get(gid)
    if zeros:
        if pages is None:
            pages = _pages_u8(layer)
        for slot in zeros:
            pages[slot].zero_()
    shared = _graph_conv_outputs(layer, mixed_qkv) if FAST and not st.short else None
    if shared is not None:
        # this layer's core ran as a GDN layer graph: its conv + post-conv outputs
        # (q, k, v, g, beta at absolute rows) are still in the shared buffers, and
        # the replay rows' values equal the split flow's chunk inputs (per-token
        # conv, same initial window). Only the chunk recurrence runs again (direct
        # kernel call); the conv window at each cut is written by the conv kernel
        # over the cut's last 3 rows (no initial window needed).
        for kind, w in waves:
            if kind == "copy":
                _page_copies(layer, st, gid)
                continue
            if kind == "batch":
                # >= 2 segments: rows gathered once, one chunk launch, one conv-window launch
                idx_d, slots_d, _, cu_d, R, Ltot, maxl, _, idx3_d, c3, f3 = w
                _chunk_gathered(layer, shared, idx_d, Ltot, R, maxl, ssm, slots_d, cu_d, st)
                _conv_windows_gathered(layer, mixed_qkv, a, b, conv_state, idx3_d, R, slots_d, c3, f3, st)
                STATS["replays"] += R
                STATS["replay_tokens"] += Ltot
                STATS["batched_launches"] = STATS.get("batched_launches", 0) + 1
                continue
            for r0, L, slot_t, hi_t, cu_t in w:
                _chunk_shared(layer, shared, r0, L, ssm, slot_t, cu_t, st)
                _conv_window(layer, mixed_qkv, a, b, conv_state, r0 + L, slot_t, st)
                STATS["replays"] += 1
                STATS["replay_tokens"] += L
        STATS["fast_layers"] = STATS.get("fast_layers", 0) + 1
        if verify:
            _verify(layer, st, gid, recs, snaps, mixed_qkv, b, a, core_attn_out)
        return
    for kind, w in waves:
        if kind == "one":
            for r0, L, slot_t, hi_t, cu_t in w:
                run_chunk(layer, mixed_qkv[r0 : r0 + L], a[r0 : r0 + L], b[r0 : r0 + L],
                          conv_state, ssm, slot_t, hi_t, cu_t, L, bufs=st.bufs)
                STATS["replays"] += 1
                STATS["replay_tokens"] += L
        elif kind == "batch":
            idx_d, slots_d, hi_d, cu_d, R, Ltot, maxl = w[:7]
            run_chunk(layer, mixed_qkv.index_select(0, idx_d), a.index_select(0, idx_d),
                      b.index_select(0, idx_d), conv_state, ssm, slots_d, hi_d, cu_d, Ltot,
                      bufs=st.bufs, nseq=R, maxlen=maxl)
            STATS["replays"] += R
            STATS["replay_tokens"] += Ltot
            STATS["batched_launches"] = STATS.get("batched_launches", 0) + 1
        else:  # page copies after the first wave: D (state at b) -> X
            _page_copies(layer, st, gid)
    if verify:
        _verify(layer, st, gid, recs, snaps, mixed_qkv, b, a, core_attn_out)


def _scratch_views(layer, page_src):
    """A 1-slot scratch pool with the layer's page layout (conv | ssm | log),
    initialised from the uint8 page `page_src`: (bytes, conv view, ssm view)."""
    mod = _gdn()
    conv, ssm = layer.kv_cache[0], layer.kv_cache[1]
    page = page_src.numel()
    dev = ssm.device
    raw = torch.empty(page, dtype=torch.uint8, device=dev)
    raw.copy_(page_src)
    stg = raw.untyped_storage()
    base = min(conv.data_ptr(), ssm.data_ptr())
    c_off = (conv.data_ptr() - base) // conv.element_size()
    s_off = (ssm.data_ptr() - base) // ssm.element_size()
    c = torch.empty(0, dtype=conv.dtype, device=dev).set_(stg)
    c = torch.as_strided(
        c, (1,) + tuple(conv.shape[1:]), (page // conv.element_size(),) + tuple(conv.stride()[1:]), c_off
    )
    s = torch.empty(0, dtype=ssm.dtype, device=dev).set_(stg)
    s = torch.as_strided(
        s, (1,) + tuple(ssm.shape[1:]), (page // ssm.element_size(),) + tuple(ssm.stride()[1:]), s_off
    )
    c_v = c if mod.is_conv_state_dim_first() else c.transpose(-1, -2)
    return raw, c_v, s


def _verify(layer, st, gid, recs, snaps, mixed_qkv, b, a, core_attn_out):
    """Split-flow chain on a scratch page vs the in-step checkpoints (bitwise).
    recs: (a0, slot, kind, pos, start, end, zero_init)."""
    conv, ssm = layer.kv_cache[0], layer.kv_cache[1]
    pages = _pages_u8(layer)
    base = min(conv.data_ptr(), ssm.data_ptr())
    c0 = conv.data_ptr() - base
    c1 = c0 + conv[0].numel() * conv.element_size()
    s0 = ssm.data_ptr() - base
    s1 = s0 + ssm[0].numel() * ssm.element_size()
    dev = ssm.device
    by_req: dict = {}
    for r in recs:
        by_req.setdefault((r[0], r[4], r[5]), []).append(r)
    for (a0, start, end), rs in by_req.items():
        if VERIFY[0] <= 0:
            return
        VERIFY[0] -= 1
        rs = sorted(rs, key=lambda r: (r[3], r[2]))
        cuts = [r for r in rs if r[2] in (KIND_BLOCK, KIND_SPLIT, KIND_TAIL)]
        run = [r for r in rs if r[2] == KIND_RUN]
        bad = []
        # the state at `start`: D's pre-copied page, else X before its replay
        d_init = next((snaps[r[1]] for r in cuts if r[2] == KIND_TAIL), None)
        x_init = next((snaps[r[1]] for r in cuts if r[2] == KIND_BLOCK), None)
        zero = any(r[6] for r in cuts if r[2] == KIND_TAIL)
        if d_init is not None and x_init is not None and not zero:
            if not torch.equal(d_init[s0:s1], x_init[s0:s1]):
                bad.append("precopy_ssm")
            if not torch.equal(d_init[c0:c1], x_init[c0:c1]):
                bad.append("precopy_conv")
        init = d_init if d_init is not None else x_init
        if init is None:
            continue
        raw, s_conv, s_ssm = _scratch_views(layer, init)
        if zero:
            raw.zero_()
        slot0 = torch.zeros(1, dtype=torch.int32, device=dev)
        cur = start
        for r in cuts + [None]:
            pos = end if r is None else r[3]
            L = pos - cur
            if L <= 0:
                continue
            hi = torch.tensor([cur > start or (start > 0 and not zero)], device=dev)
            cu = torch.tensor([0, L, 1], dtype=torch.int32, device=dev)
            lo, hi_row = a0 + cur - start, a0 + pos - start
            # the other V-split kernel copy than the replay: also checks that the
            # graph (device_nb) and eager kernels store the same bits
            run_chunk(layer, mixed_qkv[lo:hi_row], a[lo:hi_row], b[lo:hi_row], s_conv,
                      s_ssm, slot0, hi, cu[:2], L, devnb=not _use_devnb())
            cur = pos
            if r is None or r[2] == KIND_SPLIT:
                continue
            got = pages[r[1]]
            if not torch.equal(got[s0:s1], raw[s0:s1]):
                d = (got[s0:s1].view(torch.float32) - raw[s0:s1].view(torch.float32)).abs()
                bad.append(f"ssm@{pos}:max{float(d.max()):.3g}")
            if not torch.equal(got[c0:c1], raw[c0:c1]):
                bad.append(f"conv@{pos}")
        # final state vs the running block (the step ran the row unsplit:
        # informational, float-order)
        fin = "n/a"
        if run:
            got = pages[run[0][1]]
            if torch.equal(got[s0:s1], raw[s0:s1]):
                fin = "bitwise"
            else:
                d = (got[s0:s1].view(torch.float32) - raw[s0:s1].view(torch.float32)).abs()
                fin = f"max|d|={float(d.max()):.3g}"
            STATS["verify_final_bitwise"] = STATS.get("verify_final_bitwise", 0) + int(
                fin == "bitwise"
            )
        STATS["verify_layers"] += 1
        desc = [(r[3], r[2]) for r in cuts]
        if bad:
            STATS["verify_mismatch"] += 1
            logger.warning(
                "[F122] VERIFY MISMATCH layer=%s group=%d start=%d end=%d ckpts=%s bad=%s",
                layer.prefix, gid, start, end, desc, bad,
            )
        elif (STATS["verify_layers"] <= 5 or STATS["verify_layers"] % 100 == 0
              or any(r[2] in (KIND_BLOCK, KIND_SPLIT) for r in cuts) and STATS["verify_layers"] % 10 == 0):
            logger.info(
                "[F122] verify ok #%d layer=%s group=%d start=%d end=%d ckpts=%s final=%s",
                STATS["verify_layers"], layer.prefix, gid, start, end, desc, fin,
            )
