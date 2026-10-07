# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CUDA graphs for the spec-decode verify sampler of decode steps.

On GB300 at low concurrency the decode step is host-bound: after the FULL-graph
forward, ``GPUModelRunner.sample`` (target lm_head, sampling params, rejection
sampling, accepted counts) launches ~30 eager ops
whose Python + launch time is ~1 ms per step on Grace. This module captures
that call once per batch shape and sampling-parameter signature and replays it.

What is captured: exactly ``runner.sample(hidden_states, input_batch, None)``,
the same Python code, kernels and launch arguments as the eager path. Only the
per-step device tensors of the InputBatch that are freshly allocated every step
(idx_mapping, expanded_idx_mapping, expanded_local_pos, cu_num_logits,
logits_indices) are copied into static buffers before the replay; all other
device inputs (hidden states of the FULL decode graph, input_ids / positions /
seq_lens buffers, per-request sampling state) are persistent tensors whose
addresses are part of the graph key. The outputs are cloned after the replay so
their lifetime matches the eager path (the async D2H output copy and the
drafter read them).

Randomness: the V2 sampler and rejection sampler draw with Triton
``tl.rand(seed, offset)`` from the per-request seed tensor and token positions
(device tensors read at replay); there is no torch generator state, so replays
draw exactly what the eager path draws. Batches with explicitly seeded requests
still run eagerly (conservative).

Host decisions inside the captured call (top-k max,
penalty / logit-bias / bad-words / thinking-budget / logprobs branches, chunking)
are functions of the batch's per-request sampling parameters; the graph key holds
the set of distinct per-request parameter tuples plus the batch shape, so a
replay always follows the branches of its capture.

Eligible steps: decode-only spec-decode verify steps (no prefill rows, no
grammar, no draft logits, no watermark, no adaptive verification, no logprobs,
no batch sharding / PCP), num_reqs <= VLLM_SAMPLER_GRAPH_MAX_REQS.

Environment:
    VLLM_SAMPLER_GRAPH=1               enable (default off)
    VLLM_SAMPLER_GRAPH_MAX_REQS=64     largest captured batch
    VLLM_SAMPLER_GRAPH_MAX_GRAPHS=96   cap on captured graphs (then eager)
    VLLM_SAMPLER_GRAPH_WARM=2          eager runs of a key before it is captured
    VLLM_SAMPLER_GRAPH_CHECK=N         for the first N replays also run the
                                       eager path first and compare sampled
                                       tokens / num_sampled / num_rejected
                                       bitwise (mismatch -> disable, warn)
    VLLM_SAMPLER_GRAPH_LOG=N           log counters every N replays (0 = off)
"""

import os
from dataclasses import replace

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_P = "VLLM_SAMPLER_GRAPH"
ENABLED = os.environ.get(_P, "0") == "1"
MAX_REQS = int(os.environ.get(_P + "_MAX_REQS", "64"))
MAX_GRAPHS = int(os.environ.get(_P + "_MAX_GRAPHS", "96"))
WARM = int(os.environ.get(_P + "_WARM", "2"))
_CHECK = [int(os.environ.get(_P + "_CHECK", "0"))]
# Optional: N read from this file at engine start (lets a CHECK smoke share the
# scored arm's env, i.e. its warm caches).
_CHECK_FILE = os.environ.get(_P + "_CHECK_FILE", "")
if _CHECK_FILE and os.path.isfile(_CHECK_FILE):
    try:
        _CHECK[0] = max(_CHECK[0], int(open(_CHECK_FILE).read().strip() or 0))
    except (OSError, ValueError):
        pass
LOG_EVERY = int(os.environ.get(_P + "_LOG", "0"))
# Start-up pre-capture (GB300 lowc2): "1-8" or "1,2,4" = decode batch sizes to
# capture at engine start, for the sampling parameters in
# VLLM_SAMPLER_GRAPH_PRECAPTURE_PARAMS ("temperature=1.0,top_k=20,top_p=0.95";
# SamplingParams keyword arguments). The keys are built exactly like serving
# keys (same hidden-state / input buffers, parameter signature, state-buffer
# addresses for every pool-slot parity), so a pre-captured graph is replayed
# only when a serving step has an identical key; otherwise the lazy path runs
# as before. Default off.
PRECAPTURE = os.environ.get(_P + "_PRECAPTURE", "")
PRECAPTURE_PARAMS = os.environ.get(_P + "_PRECAPTURE_PARAMS", "")

STATS: dict[str, int] = {}


def _inc(k: str, n: int = 1) -> None:
    STATS[k] = STATS.get(k, 0) + n


class SamplerGraphs:
    def __init__(self, runner) -> None:
        self.runner = runner
        dev = runner.device
        L = MAX_REQS * (runner.num_speculative_steps + 1)
        self.max_logits = L
        i32, i64 = torch.int32, torch.int64
        # Static copies of the per-step InputBatch tensors (dtypes fixed at
        # first use to the eager tensors' dtypes).
        self._dtypes: dict[str, torch.dtype] = {}
        self._bufs: dict[str, torch.Tensor] = {}
        self._dev = dev
        self._sizes = {
            "idx_mapping": MAX_REQS,
            "expanded_idx_mapping": L,
            "expanded_local_pos": L,
            "cu_num_logits": MAX_REQS + 1,
            "logits_indices": L,
        }
        del i32, i64
        self.graphs: dict[tuple, tuple] = {}
        self.seen: dict[tuple, int] = {}
        self.bad_keys: set[tuple] = set()
        self.pool = None
        self.stream = None
        self.disabled = False
        self._getters = None
        self._last_scratch = None
        self._realloc_seen = False
        self.precaptured: set[tuple] = set()
        self._in_precapture = False
        logger.info(
            "sampler graphs enabled: max_reqs=%d max_graphs=%d warm=%d check=%d",
            MAX_REQS,
            MAX_GRAPHS,
            WARM,
            _CHECK[0],
        )

    # ---------------------------------------------------------------- helpers
    def _buf(self, name: str, src: torch.Tensor) -> torch.Tensor:
        b = self._bufs.get(name)
        if b is None or b.dtype != src.dtype:
            b = torch.zeros(self._sizes[name], dtype=src.dtype, device=self._dev)
            self._bufs[name] = b
        return b

    def _state_ptr_getters(self):
        """Getters for the device addresses of every per-request state buffer the
        sampler may read. UvaBackedTensor.gpu rotates through a buffer pool on
        every staged write, so the graph key must hold the current addresses."""
        r = self.runner
        smp = r.sampler
        objs = [smp, getattr(smp, "sampling_states", None),
                getattr(smp, "penalties_state", None),
                getattr(smp, "logit_bias_state", None),
                getattr(smp, "bad_words_state", None),
                getattr(smp, "thinking_budget_state", None),
                getattr(smp, "logprob_token_ids_state", None),
                getattr(smp, "req_states", None), r.req_states, r.rejection_sampler]
        getters = []
        seen = set()
        for o in objs:
            if o is None or id(o) in seen:
                continue
            seen.add(id(o))
            for name, v in vars(o).items():
                if isinstance(v, torch.Tensor):
                    if v.is_cuda:
                        getters.append((o, name, False))
                elif isinstance(getattr(v, "gpu", None), torch.Tensor):
                    getters.append((o, name, True))
        return getters

    @staticmethod
    def _scratch_ptrs() -> tuple:
        """Addresses of the module-level scratch buffers the sampling kernels
        cache and REALLOCATE when a larger batch arrives (e.g. an eager step with
        more logits rows): a graph must never replay against a freed buffer.
        Modules that are not imported (or lack the cache) contribute nothing."""
        import sys

        out = []
        for modname, attr in (
            ("vllm.v1.worker.gpu.sample.spec_topk_topp", "_SCRATCH"),
            ("vllm.v1.sample.ops.topk_topp_triton", "_TRITON_BUFFER_CACHE"),
            ("vllm.v1.sample.ops.topk_topp_triton", "_TRITON_SPLIT_CACHE"),
            ("vllm.v1.sample.ops.topk_topp_triton", "_TRITON_TABLE_CACHE"),
        ):
            mod = sys.modules.get(modname)
            d = getattr(mod, attr, None) if mod is not None else None
            if not isinstance(d, dict):
                continue
            for k in sorted(d, key=repr):
                v = d[k]
                if isinstance(v, dict):
                    vals = v.values()
                elif isinstance(v, (tuple, list)):
                    vals = v
                else:
                    vals = (v,)
                for t in vals:
                    if isinstance(t, torch.Tensor):
                        out.append(t.data_ptr())
        return tuple(out)

    def _state_ptrs(self) -> tuple:
        if self._getters is None:
            self._getters = self._state_ptr_getters()
            logger.info("sampler graphs: %d state buffers in the graph key",
                        len(self._getters))
        out = []
        for o, name, sub in self._getters:
            v = getattr(o, name)
            t = v.gpu if sub else v
            out.append(t.data_ptr() if isinstance(t, torch.Tensor) else 0)
        scratch = self._scratch_ptrs()
        if self._last_scratch is not None and scratch != self._last_scratch:
            _inc("scratch_realloc")
            self._realloc_seen = True
            logger.info(
                "sampler graphs: sampling scratch buffers reallocated (#%d); graphs "
                "keyed on the old addresses are no longer replayed; stats %s",
                STATS["scratch_realloc"],
                STATS,
            )
        self._last_scratch = scratch
        return tuple(out) + scratch

    def _signature(self, input_batch) -> tuple | None:
        r = self.runner
        smp = r.sampler
        st = smp.sampling_states
        idx = input_batch.idx_mapping_np
        if st.any_explicit_seed(idx):
            return None
        cols = [
            st.temperature.np[idx],
            st.top_k.np[idx].astype(np.float64),
            st.top_p.np[idx],
            st.min_p.np[idx],
            st.num_logprobs[idx].astype(np.float64),
            smp.penalties_state.use_penalty[idx].astype(np.float64),
            smp.penalties_state.repetition_penalty.np[idx],
            smp.penalties_state.frequency_penalty.np[idx],
            smp.penalties_state.presence_penalty.np[idx],
            smp.logit_bias_state.use_logit_bias[idx].astype(np.float64),
            smp.bad_words_state.num_bad_words.np[idx].astype(np.float64),
            smp.needs_logits_processing[idx].astype(np.float64),
        ]
        tb = smp.thinking_budget_state
        if tb.enabled:
            cols.append(tb.use_thinking_budget[idx].astype(np.float64))
        m = np.stack([np.asarray(c, dtype=np.float64) for c in cols], axis=1)
        return tuple(sorted({tuple(row) for row in m.tolist()}))

    def _eligible_key(self, hidden_states, input_batch, grammar_output):
        r = self.runner
        if self.disabled or grammar_output is not None:
            return None
        if r.rejection_sampler is None or r.speculator is None:
            return None
        if (
            r.batch_sharder is not None
            or r.pcp_manager is not None
            or getattr(r, "pp_handler", None) is not None
        ):
            return None
        rs = r.rejection_sampler
        if (
            rs.watermark_key is not None
            or rs.enable_adaptive_verification
            or rs.use_block_verification
            or rs.synthetic_conditional_rates is not None
            or r.sampler.compute_nans
        ):
            return None
        if getattr(r.speculator, "draft_logits", None) is not None:
            return None
        B = input_batch.num_reqs
        if (
            B == 0
            or B > MAX_REQS
            or input_batch.has_prefill
            or input_batch.num_draft_tokens == 0
        ):
            return None
        nl = int(input_batch.logits_indices.shape[0])
        if nl > self.max_logits or int(input_batch.cu_num_logits.shape[0]) != B + 1:
            return None
        sig = self._signature(input_batch)
        if sig is None:
            return None
        if self.precaptured and not self._in_precapture and not STATS.get(
            "_sig_logged"
        ):
            STATS["_sig_logged"] = 1
            pre_sigs = {k[9] for k in self.precaptured}
            if sig not in pre_sigs:
                logger.warning(
                    "sampler graph pre-capture: first serving signature %s differs "
                    "from the pre-captured %s (pre-captured graphs will not be "
                    "replayed; lazy capture continues)",
                    sig,
                    sorted(pre_sigs),
                )
            else:
                logger.info("sampler graph pre-capture: serving signature matches")
        if any(row[4] != -1 for row in sig):  # logprobs requested
            return None
        return (
            B,
            nl,
            hidden_states.data_ptr(),
            tuple(hidden_states.shape),
            hidden_states.stride(0),
            input_batch.input_ids.data_ptr(),
            input_batch.positions.data_ptr(),
            input_batch.seq_lens.data_ptr(),
            int(input_batch.seq_lens.shape[0]),
            sig,
            self._state_ptrs(),
        )

    def _static_batch(self, input_batch, copy: bool):
        kw = {}
        for name in self._sizes:
            src = getattr(input_batch, name)
            b = self._buf(name, src)[: src.shape[0]]
            if copy:
                b.copy_(src, non_blocking=True)
            kw[name] = b
        return replace(input_batch, **kw)

    def _capture(self, key, hidden_states, input_batch):
        r = self.runner
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
            self.stream = torch.cuda.Stream(device=self._dev)
        ib = self._static_batch(input_batch, copy=True)
        g = torch.cuda.CUDAGraph()
        self.stream.wait_stream(torch.cuda.current_stream())
        # Manual capture (no synchronize / gc / empty_cache of torch.cuda.graph):
        # this runs mid-serving. thread_local: other threads' CUDA calls do not
        # invalidate the capture.
        out = None
        err = None
        with torch.cuda.stream(self.stream):
            g.capture_begin(pool=self.pool, capture_error_mode="thread_local")
            try:
                out, ns, nr = r.sample(hidden_states, ib, None)
            except Exception as e:  # never break serving
                err = e
            finally:
                try:
                    g.capture_end()
                except Exception as e:
                    err = err or e
        torch.cuda.current_stream().wait_stream(self.stream)
        if err is not None or out is None:
            logger.warning("sampler graph capture failed for B=%d: %r", key[0], err)
            self.bad_keys.add(key)
            _inc("capture_failed")
            return None
        if (
            out.logprobs_tensors is not None
            or out.num_nans is not None
            or out.sampling_mask_tensors is not None
        ):
            self.bad_keys.add(key)
            _inc("capture_unsupported_outputs")
            return None
        ent = (g, ib, out, ns, nr)
        self.graphs[key] = ent
        _inc("captured")
        return ent

    def _replay(self, ent, input_batch):
        g, ib, out, ns, nr = ent
        for name in self._sizes:
            getattr(ib, name).copy_(getattr(input_batch, name), non_blocking=True)
        g.replay()
        sampled = out.sampled_token_ids.clone()
        num_sampled = ns.clone()
        num_rejected = nr.clone()
        so = replace(
            out,
            sampled_token_ids=sampled,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        return so, num_sampled, num_rejected

    # ------------------------------------------------------------ pre-capture
    @staticmethod
    def _parse_sizes(spec: str) -> list[int]:
        out: set[int] = set()
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
        return sorted(b for b in out if 0 < b <= MAX_REQS)

    @staticmethod
    def _parse_params(spec: str):
        from vllm.sampling_params import SamplingParams

        kw = {}
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            k, v = part.split("=", 1)
            v = v.strip()
            try:
                kw[k.strip()] = int(v)
            except ValueError:
                kw[k.strip()] = float(v)
        return SamplingParams(**kw)

    def _synthetic_decode_batch(self, B: int, row: int):
        """A B-request spec-decode verify batch shaped exactly like the one
        prepare_inputs builds for B decode requests with num_speculative_steps
        drafts each (same persistent input buffers, dtypes and logits layout);
        every request maps to request-state row ``row``."""
        from vllm.v1.worker.gpu.input_batch import InputBatch

        r = self.runner
        k = r.num_speculative_steps + 1  # logits rows per request
        n = B * k
        dev = self._dev
        ib = InputBatch.make_dummy(B, n, r.input_buffers, max_query_len=k)
        # Real decode tokens (make_dummy marks every row as padding).
        r.input_buffers.is_padding[:n].fill_(False)
        idx = torch.full((B,), row, dtype=torch.int64, device=dev)
        num_sched = np.full(B, k, dtype=np.int32)
        cu_np = np.arange(0, n + 1, k, dtype=np.int32)
        return replace(
            ib,
            idx_mapping=idx,
            idx_mapping_np=np.full(B, row, dtype=np.intp),
            expanded_idx_mapping=torch.full((n,), row, dtype=torch.int64, device=dev),
            expanded_local_pos=torch.arange(k, dtype=torch.int32, device=dev).repeat(B),
            num_scheduled_tokens=num_sched,
            num_draft_tokens=B * (k - 1),
            num_draft_tokens_per_req=np.full(B, k - 1, dtype=np.int32),
            cu_num_logits=torch.from_numpy(cu_np).to(dev),
            cu_num_logits_np=cu_np,
            logits_indices=torch.arange(n, dtype=torch.int64, device=dev),
            num_computed_tokens_np=np.full(B, 1, dtype=np.int32),
            prefill_len_np=np.full(B, 1, dtype=np.int32),
            num_computed_prefill_tokens_np=np.full(B, 1, dtype=np.int32),
            is_prefilling_np=np.zeros(B, dtype=bool),
            has_prefill=False,
            max_query_len=None,
        )

    def precapture(self) -> None:
        """Capture the verify-sampler graphs of B = PRECAPTURE sizes at engine
        start, for every pool-slot parity of the rotating request / sampler
        state buffers, using a scratch request-state row (the last row; any
        real request that later takes it overwrites all of its state)."""
        r = self.runner
        sizes = self._parse_sizes(PRECAPTURE)
        if not sizes or not PRECAPTURE_PARAMS:
            return
        cgm = r.cudagraph_manager
        hs = getattr(cgm, "hidden_states", None)
        k = r.num_speculative_steps + 1
        if hs is None or r.sampler is None or r.rejection_sampler is None:
            logger.warning("sampler graph pre-capture skipped (no FULL graph output)")
            return
        sizes = [b for b in sizes if b * k <= hs.shape[0]]
        sp = self._parse_params(PRECAPTURE_PARAMS)
        self._in_precapture = True
        try:
            self._precapture(sizes, sp, hs, k)
        finally:
            self._in_precapture = False

    def _precapture(self, sizes, sp, hs, k) -> None:
        r = self.runner
        row = r.max_num_reqs - 1
        smp = r.sampler
        rs = r.req_states
        # A 1-token "prompt" on the scratch row, so per-request state that is
        # initialised from the prompt (e.g. the penalties bincount) is well-formed.
        rs.prompt_len.np[row] = 1
        rs.prefill_len.np[row] = 1
        rs.prompt_len.copy_to_uva()
        rs.prefill_len.copy_to_uva()
        smp.add_request(row, 1, sp)
        if hasattr(smp, "_lowc2_dirty"):
            smp._lowc2_dirty = True
        smp.apply_staged_writes()
        t0 = __import__("time").perf_counter()
        batches = {b: self._synthetic_decode_batch(b, row) for b in sizes}
        # Eager runs first (largest first): lazy allocations of the sampling
        # scratch buffers happen outside any capture, and every graph is keyed
        # on the final scratch addresses.
        for b in reversed(sizes):
            r.sample(hs[: b * k], batches[b], None)
        torch.cuda.current_stream().synchronize()

        def rotate_sampler():
            # Same call serving makes (rotates every sampler-state pool together;
            # with the lowc2 dirty gate, mark dirty so it really rotates).
            if hasattr(smp, "_lowc2_dirty"):
                smp._lowc2_dirty = True
            smp.apply_staged_writes()

        def rotate_reqs():
            rs.prompt_len.copy_to_uva()
            rs.prefill_len.copy_to_uva()

        n_s = smp.sampling_states.temperature.pool.max_concurrency
        n_r = rs.prompt_len.pool.max_concurrency
        n_cap = 0
        for _ in range(n_s):
            for _ in range(n_r):
                for b in reversed(sizes):
                    ib = batches[b]
                    key = self._eligible_key(hs[: b * k], ib, None)
                    if key is None or key in self.graphs:
                        continue
                    if len(self.graphs) >= MAX_GRAPHS:
                        break
                    if self._capture(key, hs[: b * k], ib) is not None:
                        self.precaptured.add(key)
                        n_cap += 1
                rotate_reqs()
            rotate_sampler()
        torch.cuda.current_stream().synchronize()
        _inc("precaptured", n_cap)
        logger.info(
            "sampler graph pre-capture: %d graphs for B=%s (params %s, row %d) in "
            "%.2f s; signature %s",
            n_cap,
            sizes,
            PRECAPTURE_PARAMS,
            row,
            __import__("time").perf_counter() - t0,
            sorted({kk[9] for kk in self.precaptured}),
        )

    # ------------------------------------------------------------------- main
    def sample(self, hidden_states, input_batch, grammar_output):
        r = self.runner
        key = self._eligible_key(hidden_states, input_batch, grammar_output)
        _inc("steps")
        if key is None or key in self.bad_keys:
            _inc("eager_ineligible")
            if not input_batch.has_prefill:
                _inc("eager_ineligible_decode")
            return r.sample(hidden_states, input_batch, grammar_output)
        ent = self.graphs.get(key)
        if ent is None:
            if len(self.seen) > 4096:
                self.seen.clear()
            n = self.seen.get(key, 0) + 1
            self.seen[key] = n
            res = r.sample(hidden_states, input_batch, grammar_output)
            if n > WARM and len(self.graphs) < MAX_GRAPHS:
                self._capture(key, hidden_states, input_batch)
            _inc("eager_warm")
            return res
        if _CHECK[0] > 0:
            _CHECK[0] -= 1
            ref, ref_ns, ref_nr = r.sample(hidden_states, input_batch, grammar_output)
            so, ns, nr = self._replay(ent, input_batch)
            # sampled_token_ids is allocated with new_empty: only the first
            # num_sampled entries of each row are defined (and read downstream).
            ok_ns = torch.equal(ref_ns, ns)
            ok_nr = torch.equal(ref_nr, nr)
            ok_tok = False
            if ok_ns:
                st_ref, st_g = ref.sampled_token_ids, so.sampled_token_ids
                cols = torch.arange(st_ref.shape[1], device=st_ref.device)
                valid = cols[None, :] < ns[: st_ref.shape[0], None].to(cols.dtype)
                ok_tok = st_ref.shape == st_g.shape and torch.equal(
                    torch.where(valid, st_ref, 0), torch.where(valid, st_g, 0)
                )
            ok = ok_ns and ok_nr and ok_tok
            if ok_ns and st_ref.shape == st_g.shape:
                diff = st_ref != st_g
                n_in = int((diff & valid).sum().item())
                n_out = int((diff & ~valid).sum().item())
                if n_out:
                    _inc("check_diff_outside_mask")
                if n_in or (n_out and not STATS.get("_logged_outside")):
                    STATS["_logged_outside"] = 1
                    logger.info(
                        "sampler graph CHECK detail (B=%d, logits=%d): differing token "
                        "positions inside num_sampled mask %d, outside %d; num_sampled "
                        "eager %s graph %s; num_rejected eager %s graph %s",
                        key[0],
                        key[1],
                        n_in,
                        n_out,
                        ref_ns.tolist(),
                        ns.tolist(),
                        ref_nr.tolist(),
                        nr.tolist(),
                    )
            elif not ok_ns or not ok_nr:
                logger.warning(
                    "sampler graph CHECK detail (B=%d): num_sampled eager %s graph %s; "
                    "num_rejected eager %s graph %s",
                    key[0],
                    ref_ns.tolist(),
                    ns.tolist(),
                    ref_nr.tolist(),
                    nr.tolist(),
                )
            _inc("check_ok" if ok else "check_mismatch")
            if ok and self._realloc_seen:
                _inc("check_ok_after_realloc")
            if ok:
                _inc("check_tokens", int(ns.sum().item()))
            if not ok:
                self.disabled = True
                logger.warning(
                    "sampler graph CHECK MISMATCH (B=%d, logits=%d; num_sampled %s, "
                    "num_rejected %s, tokens %s): disabling sampler graphs; stats %s",
                    key[0],
                    key[1],
                    ok_ns,
                    ok_nr,
                    ok_tok,
                    STATS,
                )
                return ref, ref_ns, ref_nr
            if _CHECK[0] == 0:
                logger.info("sampler graph check done: %s", STATS)
            return so, ns, nr
        _inc("replay")
        _inc(f"replay_B{key[0]}")
        if key in self.precaptured:
            _inc("replay_precaptured")
        if self._realloc_seen:
            _inc("replay_after_realloc")
        if LOG_EVERY and STATS["replay"] % LOG_EVERY == 0:
            dec = STATS["steps"] - STATS.get("eager_ineligible", 0) + STATS.get(
                "eager_ineligible_decode", 0
            )
            logger.info(
                "sampler graph stats %s graphs=%d keys_seen=%d hit_rate(decode)=%.3f",
                STATS,
                len(self.graphs),
                len(self.seen),
                STATS["replay"] / max(dec, 1),
            )
        return self._replay(ent, input_batch)
