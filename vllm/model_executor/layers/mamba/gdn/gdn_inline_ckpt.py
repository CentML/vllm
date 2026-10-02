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

import torch

from vllm.logger import init_logger

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
                 "lmax", "_keep")

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
    pin = torch.cuda.is_available()

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

    for gid in gids:
        st.per_group[gid] = [
            (a0, blocks[gid], kind, pos, start, end, bool(zero_init))
            for a0, start, end, cks in rows
            for pos, kind, zero_init, blocks in cks
        ]
        jobs = _plan_jobs(rows, gid)
        if not jobs:
            continue
        st.zeros[gid] = [j[0] for j in jobs if j[1]]
        st.copies[gid] = [(j[0], j[3]) for j in jobs if j[3] is not None]
        waves = []
        for w in range(max(len(j[2]) for j in jobs)):
            segs = [(j[0], j[2][w]) for j in jobs if len(j[2]) > w]
            if len(segs) == 1 or not BATCH:
                for slot, (r0, L, hi) in segs:
                    m = dev_i32([slot, 0, L, 1])  # slot | cu_seqlens | device_nb count
                    waves.append(("one", (r0, L, m[0:1], dev_bool([hi]), m[1:3])))
                    st.lmax = max(st.lmax, L)
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
                waves.append(
                    (
                        "batch",
                        (idx_d, m[:R], dev_bool([hi for _, (_, _, hi) in segs]),
                         m[R : 2 * R + 1], R, cu[R], max(L for _, (_, L, _) in segs)),
                    )
                )
                st.lmax = max(st.lmax, cu[R])
            if w == 0:
                waves.append(("copy", None))
        st.waves[gid] = waves
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


def after_core(layer, mixed_qkv, b, a, core_attn_out=None) -> None:
    """End of the GDN core custom op of `layer` (any path): write this step's
    in-step checkpoints of the layer's KV-cache group."""
    st = _STEP[0]
    if st is None:
        return
    gid = _LAYER_GROUP.get(layer.prefix)
    if gid is None:
        raise RuntimeError(f"[F122] GDN layer {layer.prefix} has no KV-cache group")
    waves = st.waves.get(gid)
    if not waves:
        return
    conv_state, ssm, _ = _layer_views(layer)
    pages = None
    verify = VERIFY[0] > 0 and not torch.cuda.is_current_stream_capturing()
    recs = st.per_group.get(gid, [])
    snaps = None
    if verify:
        pages = _pages_u8(layer)
        snaps = {r[1]: pages[r[1]].clone() for r in recs if r[2] != KIND_SPLIT}
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
    for kind, w in waves:
        if kind == "one":
            r0, L, slot_t, hi_t, cu_t = w
            run_chunk(layer, mixed_qkv[r0 : r0 + L], a[r0 : r0 + L], b[r0 : r0 + L],
                      conv_state, ssm, slot_t, hi_t, cu_t, L, bufs=st.bufs)
            STATS["replays"] += 1
            STATS["replay_tokens"] += L
        elif kind == "batch":
            idx_d, slots_d, hi_d, cu_d, R, Ltot, maxl = w
            run_chunk(layer, mixed_qkv.index_select(0, idx_d), a.index_select(0, idx_d),
                      b.index_select(0, idx_d), conv_state, ssm, slots_d, hi_d, cu_d, Ltot,
                      bufs=st.bufs, nseq=R, maxlen=maxl)
            STATS["replays"] += R
            STATS["replay_tokens"] += Ltot
            STATS["batched_launches"] = STATS.get("batched_launches", 0) + 1
        else:  # page copies after the first wave: D (state at b) -> X
            cps = st.copies.get(gid)
            if cps:
                if pages is None:
                    pages = _pages_u8(layer)
                for src, dst in cps:
                    pages[dst].copy_(pages[src])
                STATS["page_copies"] = STATS.get("page_copies", 0) + len(cps)
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
