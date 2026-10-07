# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated capture of real MoE routing for offline analysis (instrument only).

Enabled by ``VLLM_ROUTING_CAPTURE_DIR=<dir>``; when unset nothing here runs.

Every MoE call (target layers and MTP draft layers, eager or inside CUDA graphs)
launches one Triton kernel after the routed experts. It reads the call's router
logits, computes the top-``K+1`` logits per token (ties -> lowest expert id) and
appends ``K`` expert ids (int16) and ``K+1`` logit values (fp32) per token to a
device ring, plus one call record ``(kind=1, layer tag, T, ring row)`` to a
device metadata ring. ``T`` is the token count the MoE kernel sees (CUDA-graph
padding included). Softmax is monotonic, so the top-``K`` logits are the
top-``K`` routed experts of softmax -> top-k -> renormalize; the renormalized
weights are ``softmax(vals[:, :K])`` and ``vals[:, K-1] == vals[:, K]`` flags a
top-k boundary tie.

The host side only acts inside one time window, ``VLLM_ROUTING_CAPTURE_DELAY_S``
(default 900) seconds after this module is imported (process start, the same
reference as ``nsys --delay``), for ``VLLM_ROUTING_CAPTURE_DUR_S`` (default 20)
seconds. Inside the window every ``set_forward_context`` entry writes a forward
record (kind=0) through the same device ring, so records and calls are ordered
on the device as they ran. The host drains the rings every few forwards
(one stream sync per drain) and at window end writes
``<dir>/rcap-<host>-pid<pid>.npz`` (format: ``finish``). Calls with more than
``VLLM_ROUTING_CAPTURE_ROWS_MAX`` (default 1024) tokens keep only their per-expert
token histogram.
"""

from __future__ import annotations

import json
import os
import socket
import time

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

CAPTURE_DIR = os.environ.get("VLLM_ROUTING_CAPTURE_DIR", "")
ENABLED = bool(CAPTURE_DIR)
_DELAY_S = float(os.environ.get("VLLM_ROUTING_CAPTURE_DELAY_S", "900"))
_DUR_S = float(os.environ.get("VLLM_ROUTING_CAPTURE_DUR_S", "20"))
_ROWS_MAX = int(os.environ.get("VLLM_ROUTING_CAPTURE_ROWS_MAX", "1024"))
_RING_ROWS = 1 << 22  # token rows: 4M x (16 B ids + 36 B vals) = 208 MiB
_RING_CALLS = 1 << 18  # metadata records: 256k x 96 B = 24 MiB
_TOPK = 8
_NVALS = _TOPK + 1
_META_W = 12
_BLOCK_ROWS = 16
_DRAIN_FORWARDS = 512
_T0 = time.monotonic()

KIND_FORWARD = 0
KIND_CALL = 1
KIND_INIT = 3


@triton.jit(do_not_specialize=["stride_t", "T", "tag", "nprog"])
def _rcap_call_kernel(
    logits_ptr,
    stride_t,
    T,
    ids_ptr,
    vals_ptr,
    meta_ptr,
    state_ptr,
    tag,
    nprog,
    R: tl.constexpr,
    NC: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    KV: tl.constexpr,
    MW: tl.constexpr,
    BR: tl.constexpr,
):
    pid = tl.program_id(0)
    # state: [ring rows written, records written, programs done]. Every program
    # reads the row cursor before its acq_rel atomic; the last one to finish
    # advances the cursors, so no program reads a moved cursor.
    base = tl.load(state_ptr)
    rows = pid * BR + tl.arange(0, BR)
    cols = tl.arange(0, E)
    rmask = rows < T
    x = tl.load(
        logits_ptr + rows[:, None].to(tl.int64) * stride_t + cols[None, :],
        mask=rmask[:, None],
        other=float("-inf"),
    ).to(tl.float32)
    dst = (base + rows) % R
    for k in tl.static_range(KV):
        m = tl.max(x, axis=1)
        i = tl.argmax(x, axis=1)
        if k < K:
            tl.store(ids_ptr + dst * K + k, i.to(tl.int16), mask=rmask)
        tl.store(vals_ptr + dst * KV + k, m, mask=rmask)
        x = tl.where(cols[None, :] == i[:, None], float("-inf"), x)
    done = tl.atomic_add(state_ptr + 2, 1)
    if done == nprog - 1:
        c = tl.load(state_ptr + 1)
        j = tl.arange(0, 16)
        rec = tl.where(
            j == 0,
            1,
            tl.where(j == 1, tag, tl.where(j == 2, T, tl.where(j == 3, base, 0))),
        )
        tl.store(meta_ptr + (c % NC) * MW + j, rec.to(tl.int64), mask=j < MW)
        tl.store(state_ptr, base + T)
        tl.store(state_ptr + 1, c + 1)
        tl.store(state_ptr + 2, 0)


@triton.jit(do_not_specialize=[f"v{i}" for i in range(12)])
def _rcap_record_kernel(
    meta_ptr,
    state_ptr,
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    v9,
    v10,
    v11,
    NC: tl.constexpr,
    MW: tl.constexpr,
):
    c = tl.load(state_ptr + 1)
    row = meta_ptr + (c % NC) * MW
    tl.store(row + 0, v0.to(tl.int64))
    tl.store(row + 1, v1.to(tl.int64))
    tl.store(row + 2, v2.to(tl.int64))
    tl.store(row + 3, v3.to(tl.int64))
    tl.store(row + 4, v4.to(tl.int64))
    tl.store(row + 5, v5.to(tl.int64))
    tl.store(row + 6, v6.to(tl.int64))
    tl.store(row + 7, v7.to(tl.int64))
    tl.store(row + 8, v8.to(tl.int64))
    tl.store(row + 9, v9.to(tl.int64))
    tl.store(row + 10, v10.to(tl.int64))
    tl.store(row + 11, v11.to(tl.int64))
    tl.store(state_ptr + 1, c + 1)


_CG_MODE = {"NONE": 0, "PIECEWISE": 1, "FULL": 2}


def _i32(v) -> int:
    return -1 if v is None else max(-(2**31), min(int(v), 2**31 - 1))


class RoutingCapture:
    def __init__(self) -> None:
        self.ready = False
        self.disabled = False
        self.active = False
        self.finished = False
        self.tags: dict[str, int] = {}
        self.seq = 0
        self.t_start = 0.0
        self.pending_rows = 0
        self.forwards_since_drain = 0
        self.chunks: list[tuple[int, np.ndarray, int, np.ndarray, np.ndarray]] = []
        self.lost_records = 0
        self.lost_rows = 0
        self.drains = 0

    # ---- device side -------------------------------------------------------

    def _allocate(self, device: torch.device) -> None:
        self.device = device
        self.ids = torch.zeros((_RING_ROWS, _TOPK), dtype=torch.int16, device=device)
        self.vals = torch.zeros(
            (_RING_ROWS, _NVALS), dtype=torch.float32, device=device
        )
        self.meta = torch.zeros(
            (_RING_CALLS, _META_W), dtype=torch.int64, device=device
        )
        self.state = torch.zeros(4, dtype=torch.int64, device=device)
        # Compile the record kernel now (profile run), not inside the window.
        self._record(KIND_INIT, *([0] * 11))
        self.ready = True
        logger.info(
            "routing capture: rings allocated (%d rows, %d records) on %s; "
            "window %.0f s + %.0f s after start, dir %s",
            _RING_ROWS,
            _RING_CALLS,
            device,
            _DELAY_S,
            _DUR_S,
            CAPTURE_DIR,
        )

    def _tag(self, layer_name: str) -> int:
        tag = self.tags.get(layer_name)
        if tag is None:
            from vllm.model_executor.models.utils import extract_layer_index

            draft = "mtp" in layer_name.split(".")
            tag = extract_layer_index(layer_name) + (1000 if draft else 0)
            assert tag not in self.tags.values(), (layer_name, self.tags)
            self.tags[layer_name] = tag
        return tag

    def record_call(
        self, layer_name: str, router_logits: torch.Tensor, num_experts: int
    ) -> None:
        if self.disabled:
            return
        T = router_logits.shape[0]
        if T == 0:
            return
        if not self.ready:
            if torch.cuda.is_current_stream_capturing():
                logger.warning(
                    "routing capture: first MoE call is under graph "
                    "capture; capture disabled"
                )
                self.disabled = True
                return
            assert num_experts & (num_experts - 1) == 0, num_experts
            self.num_experts = num_experts
            self._allocate(router_logits.device)
        assert router_logits.dim() == 2 and router_logits.stride(1) == 1
        assert router_logits.shape[1] >= num_experts
        nprog = triton.cdiv(T, _BLOCK_ROWS)
        _rcap_call_kernel[(nprog,)](
            router_logits,
            router_logits.stride(0),
            T,
            self.ids,
            self.vals,
            self.meta,
            self.state,
            self._tag(layer_name),
            nprog,
            R=_RING_ROWS,
            NC=_RING_CALLS,
            E=num_experts,
            K=_TOPK,
            KV=_NVALS,
            MW=_META_W,
            BR=_BLOCK_ROWS,
        )

    def _record(self, *vals: int) -> None:
        _rcap_record_kernel[(1,)](
            self.meta, self.state, *[_i32(v) for v in vals], NC=_RING_CALLS, MW=_META_W
        )

    # ---- host side (window only) --------------------------------------------

    def forward_begin(self, ctx, num_tokens: int | None) -> None:
        if not self.ready or self.finished or self.disabled:
            return
        now = time.monotonic() - _T0
        if now < _DELAY_S or torch.cuda.is_current_stream_capturing():
            return
        if not self.active:
            st = self.state.cpu().tolist()
            self.r_drained, self.c_drained = st[0], st[1]
            self.t_start = now
            self.active = True
            logger.info(
                "routing capture: window open at %.1f s (rows %d, records %d)",
                now,
                st[0],
                st[1],
            )
        md = ctx.attn_metadata
        if isinstance(md, list):
            md = md[0] if md else None
        live = n_dec = n_dec_tok = n_pf = n_pf_tok = None
        if isinstance(md, dict):
            for m in md.values():
                if hasattr(m, "num_actual_tokens") and hasattr(m, "num_prefills"):
                    live = m.num_actual_tokens
                    n_dec, n_dec_tok = m.num_decodes, m.num_decode_tokens
                    n_pf, n_pf_tok = m.num_prefills, m.num_prefill_tokens
                    break
        bd = ctx.batch_descriptor
        self._record(
            KIND_FORWARD,
            self.seq,
            num_tokens,
            live,
            _CG_MODE.get(ctx.cudagraph_runtime_mode.name, 9),
            None if bd is None else int(bd.uniform),
            None if bd is None else bd.num_reqs,
            int((now - self.t_start) * 1e6),
            n_dec,
            n_dec_tok,
            n_pf,
            n_pf_tok,
        )
        self.seq += 1
        self.pending_rows += (num_tokens or 0) * 41
        self.forwards_since_drain += 1

    def forward_end(self) -> None:
        if not self.active or self.finished:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        if time.monotonic() - _T0 - self.t_start >= _DUR_S:
            self._drain()
            self.finish()
        elif (
            self.pending_rows > _RING_ROWS // 3
            or self.forwards_since_drain >= _DRAIN_FORWARDS
        ):
            self._drain()

    def _drain(self) -> None:
        st = self.state.cpu().tolist()  # blocking copy: syncs the stream
        r, c = st[0], st[1]
        c0 = self.c_drained
        if c - c0 > _RING_CALLS:
            self.lost_records += c - c0 - _RING_CALLS
            c0 = c - _RING_CALLS
        r0 = self.r_drained
        if r - r0 > _RING_ROWS:
            self.lost_rows += r - r0 - _RING_ROWS
            r0 = r - _RING_ROWS
        cidx = torch.arange(c0, c, device=self.device) % _RING_CALLS
        ridx = torch.arange(r0, r, device=self.device) % _RING_ROWS
        meta = self.meta[cidx].cpu().numpy()
        ids = self.ids[ridx].cpu().numpy()
        vals = self.vals[ridx].cpu().numpy()
        self.chunks.append((c0, meta, r0, ids, vals))
        self.c_drained, self.r_drained = c, r
        self.pending_rows = 0
        self.forwards_since_drain = 0
        self.drains += 1

    def finish(self) -> None:
        """Write the window to ``<dir>/rcap-<host>-pid<pid>.npz``.

        Arrays:
          meta [N, 12] int64, device order. kind 0 = forward record
            [0, seq, num_tokens (padded, as passed to set_forward_context),
             live tokens (attn num_actual_tokens), cudagraph mode (0 none,
             1 piecewise, 2 full), batch uniform, batch num_reqs, t_us since
             window open, num_decodes, num_decode_tokens, num_prefills,
             num_prefill_tokens]
            (-1 = unknown); kind 1 = MoE call [1, tag, T, ring row, 0...]
            with tag = layer index, + 1000 for MTP draft layers.
          rowoff [N] int64: first row of the call in ids/vals, -1 if the call's
            rows were not kept (T > rows_max or outside the drained range).
          ids [M, 8] int16 top-8 expert ids per token (descending logit).
          vals [M, 9] fp32 top-9 router logits per token.
          hist [H, E] int32 per-expert routed-token counts of calls with
            T > rows_max; hist_rec [H] int64 = their index into meta.
        """
        self.finished = True
        self.active = False
        if not self.chunks:
            logger.warning("routing capture: window closed with nothing drained")
            return
        meta = np.concatenate([ch[1] for ch in self.chunks])
        ids_all = np.concatenate([ch[3] for ch in self.chunks])
        vals_all = np.concatenate([ch[4] for ch in self.chunks])
        # Chunk rows are contiguous unless a drain lost rows; map each call's
        # absolute ring row to its index in ids_all.
        starts = np.asarray([ch[2] for ch in self.chunks], np.int64)
        lens = np.asarray([len(ch[3]) for ch in self.chunks], np.int64)
        offs = np.cumsum(lens) - lens
        calls = np.nonzero(meta[:, 0] == KIND_CALL)[0]
        T = meta[calls, 2]
        base = meta[calls, 3]
        ci = np.searchsorted(starts, base, side="right") - 1
        cic = np.maximum(ci, 0)
        ok = (ci >= 0) & (base + T <= starts[cic] + lens[cic])
        src0 = offs[cic] + base - starts[cic]
        small = ok & (T <= _ROWS_MAX)
        Ts = T[small]
        excl = np.cumsum(Ts) - Ts
        out_rows = int(Ts.sum())
        rowoff = np.full(len(meta), -1, np.int64)
        rowoff[calls[small]] = excl
        src = np.repeat(src0[small] - excl, Ts) + np.arange(out_rows)
        keep_ids, keep_vals = ids_all[src], vals_all[src]
        big = ok & ~small
        hist_rec = calls[big]
        hist = np.zeros((len(hist_rec), self.num_experts), np.int32)
        for h, (s0, tk) in enumerate(zip(src0[big], T[big])):
            hist[h] = np.bincount(
                ids_all[s0 : s0 + tk].ravel().astype(np.int64),
                minlength=self.num_experts,
            )
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        stem = os.path.join(
            CAPTURE_DIR, f"rcap-{socket.gethostname()}-pid{os.getpid()}"
        )
        info = {
            "tags": self.tags,
            "delay_s": _DELAY_S,
            "dur_s": _DUR_S,
            "rows_max": _ROWS_MAX,
            "ring_rows": _RING_ROWS,
            "ring_records": _RING_CALLS,
            "drains": self.drains,
            "lost_records": self.lost_records,
            "lost_rows": self.lost_rows,
            "calls_dropped": int((~ok).sum()),
            "forwards": self.seq,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(self.device),
        }
        np.savez(
            stem + ".tmp.npz",
            meta=meta,
            rowoff=rowoff,
            ids=keep_ids,
            vals=keep_vals,
            hist=hist,
            hist_rec=hist_rec,
            info=np.asarray(json.dumps(info)),
        )
        os.replace(stem + ".tmp.npz", stem + ".npz")
        self.chunks = []
        logger.info(
            "ROUTING_CAPTURE_DONE %s.npz records=%d kept_rows=%d hist_calls=%d "
            "lost_records=%d lost_rows=%d",
            stem,
            len(meta),
            out_rows,
            len(hist),
            self.lost_records,
            self.lost_rows,
        )


CAPTURE = RoutingCapture()
