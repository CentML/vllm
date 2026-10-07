# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime for the Kernel Factory paged-FP8 decode/verify attention kernel.

Kernel: ``vllm/third_party/kf_decode_attn/kernel.py`` (verbatim KF solution). One
persistent launch per call; every CTA derives its KV split from the device
``seq_lens``, so a launch captured in a FULL CUDA graph stays correct at replay for
any lengths (no host plan, nothing per step but the block-table copy below).

A compiled form is fixed by (Q, BL): query tokens per request and launched
(graph-padded) batch. ``warmup()`` AOT-compiles (TVM-FFI) every form the server
can reach before CUDA-graph capture: Q in {1, 1 + num_speculative_tokens} (the
MTP verify / draft step-1 width and the later draft steps), BL in 1..max_bl. All
forms share one set of split partials and one self-resetting counter array
(launches on a stream never overlap).

Contract (checked by the FlashInfer backend before each launch; else trtllm-gen):
  q [BL*Q, 16, 256] FP8 contiguous, out [BL*Q, 16, 256] BF16 contiguous, the
  per-layer KV cache [P, 2, 128, 512] FP8 (K|V packed per token), seq_lens [BL]
  int32 (graph-padded rows 0), block table rows of stride MAXP = 2057 (vLLM's
  narrower table is copied into a persistent MAXP-stride buffer once per step by
  the metadata builder), no window / sinks / soft cap / LSE, BF16 output.
"""

from __future__ import annotations

import time
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

HQ, HKV, HDIM, PAGE, GROUP, MAXP = 16, 2, 256, 128, 8, 2057
MAX_Q = 8  # PagedDecodeAttn asserts Q <= 8 (8 query rows per TMEM lane group)


def parse_qlens(spec: str) -> frozenset[int] | None:
    """``VLLM_KF_DECODE_ATTN_QLENS``: comma list of query widths routed to the
    kernel; empty = every width the server uses.
    """
    spec = spec.strip()
    if not spec:
        return None
    qs = frozenset(int(x) for x in spec.split(",") if x.strip())
    bad = [q for q in qs if not 1 <= q <= MAX_Q]
    if bad:
        raise ValueError(f"VLLM_KF_DECODE_ATTN_QLENS={spec!r}: widths must be 1..8")
    return qs


class KfDecodeAttn:
    def __init__(self, device: torch.device, qlens: set[int], max_bl: int):
        from vllm.third_party.kf_decode_attn import kernel as kmod

        self.k = kmod
        self.device = device
        nsm = torch.cuda.get_device_properties(device).multi_processor_count
        self.grid = min(nsm, kmod.MAX_GRID)
        # The kernel's split forms need BL <= 32 and two CTAs per (request, kv
        # head) (setup(): otherwise "unsplit" forms with per-BL scratch).
        self.max_bl = max(1, min(max_bl, 32, self.grid // HKV))
        self.qlens = frozenset(q for q in qlens if 1 <= q <= MAX_Q)
        rmax = max(self.qlens) * GROUP
        self.po = torch.empty(
            self.grid * rmax * HDIM, dtype=torch.float32, device=device
        )
        self.pml = torch.empty(rmax * self.grid * 2, dtype=torch.float32, device=device)
        # Merge arrival counters: reset to 0 by the last CTA of every stream.
        self.cnt = torch.zeros(self.max_bl * HKV, dtype=torch.int32, device=device)
        self.forms: dict[tuple[int, int], tuple[Any, ...]] = {}
        self.stats: dict[str, int] = {
            "steps": 0,  # builder steps that routed their decode rows here
            "launches": 0,  # eager launches + graph captures (replays are not seen)
            "skip_qlen": 0,  # decode steps kept on trtllm-gen: Q not routed
            "skip_batch": 0,  # ... launched batch above max_bl
            "skip_ragged": 0,  # ... non-uniform query lengths
            "skip_contract": 0,  # per-layer contract check failed (forward)
        }
        self.steps_by_form: dict[tuple[int, int], int] = {}
        self.log_every = envs.VLLM_KF_DECODE_ATTN_LOG_EVERY
        self._since_log = 0

    def form_args(self, q: int, bl: int) -> tuple[Any, ...]:
        """(fn, po, pml, cnt) for one form: the solution's setup() with shared
        scratch (split form: grid = min(SMs, MAX_GRID), BL == 1 specialization).
        """
        r = q * GROUP
        po = self.po[: self.grid * r * HDIM].view(self.grid, r, HDIM)
        pml = self.pml[: r * self.grid * 2].view(r, self.grid, 2)
        fn = self.k._compile(q, bl, self.grid, bl == 1, False)
        return fn, po, pml, self.cnt[: bl * HKV]

    def warmup(self) -> None:
        """Compile every (Q, BL) form, then launch each once on a one-page dummy
        cache so its module is loaded before CUDA-graph capture.
        """
        t0 = time.perf_counter()
        for q in sorted(self.qlens):
            for bl in range(1, self.max_bl + 1):
                self.forms[(q, bl)] = self.form_args(q, bl)
        t1 = time.perf_counter()
        dev = self.device
        fp8 = torch.float8_e4m3fn
        kv = torch.zeros(1, HKV, PAGE, 2 * HDIM, dtype=fp8, device=dev)
        bt = torch.zeros(self.max_bl, MAXP, dtype=torch.int32, device=dev)
        for (q, bl), (fn, po, pml, cnt) in self.forms.items():
            qq = torch.zeros(bl, q, HQ, HDIM, dtype=fp8, device=dev)
            out = torch.empty(bl * q * HQ * HDIM, dtype=torch.bfloat16, device=dev)
            sl = torch.full((bl,), PAGE, dtype=torch.int32, device=dev)
            fn(qq.view(torch.uint8), kv.view(torch.uint8), bt[:bl], sl, out,
               po, pml, cnt, 0.0625, 1.0, self.grid, self.k.EVICT_FIRST)  # fmt: skip
        torch.accelerator.synchronize()
        if not bool((self.cnt == 0).all()):
            raise RuntimeError("KF decode attention: merge counters not reset")
        logger.info(
            "KF decode attention: %d kernel forms (Q %s x BL 1..%d, grid %d) compiled "
            "in %.1fs, loaded in %.1fs.",
            len(self.forms),
            sorted(self.qlens),
            self.max_bl,
            self.grid,
            t1 - t0,
            time.perf_counter() - t1,
        )

    def route(self, q: int, bl: int, uniform: bool) -> bool:
        """Builder: whether this step's decode rows run here (counts the outcome)."""
        if not uniform:
            key = "skip_ragged"
        elif q not in self.qlens:
            key = "skip_qlen"
        elif bl > self.max_bl:
            key = "skip_batch"
        else:
            key = "steps"
            self.steps_by_form[(q, bl)] = self.steps_by_form.get((q, bl), 0) + 1
        self.stats[key] += 1
        self._since_log += 1
        if self.log_every > 0 and self._since_log >= self.log_every:
            self._since_log = 0
            logger.info(
                "KF decode attention stats %s; steps by (Q, BL) %s",
                self.stats,
                dict(sorted(self.steps_by_form.items())),
            )
        return key == "steps"

    def launch(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        out: torch.Tensor,
        q_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
    ) -> bool:
        """One layer's decode attention into ``out``; False = not launched."""
        bl = seq_lens.shape[0]
        f = self.forms.get((q_len, bl))
        if f is None:
            return False
        fn, po, pml, cnt = f
        fn(q.view(torch.uint8), kv.view(torch.uint8), block_tables, seq_lens,
           out.view(-1), po, pml, cnt, float(bmm1_scale), float(bmm2_scale),
           self.grid, self.k.EVICT_FIRST)  # fmt: skip
        self.stats["launches"] += 1
        if self.stats["launches"] == 1:
            logger.info("KF decode attention: first launch (Q %d, BL %d).", q_len, bl)
        return True


_RUNTIME: dict[int, KfDecodeAttn | None] = {}


def get_runtime(device: torch.device | str) -> KfDecodeAttn | None:
    """The process' runtime once ``warmup_runtime`` built it, else None."""
    return _RUNTIME.get(torch.device(device).index or 0)


def warmup_runtime(device: torch.device, qlens: set[int]) -> bool:
    """Create and warm the runtime (startup only). A failure logs and keeps
    trtllm-gen.
    """
    idx = torch.device(device).index or 0
    if idx in _RUNTIME:
        return _RUNTIME[idx] is not None
    try:
        rt = KfDecodeAttn(device, qlens, envs.VLLM_KF_DECODE_ATTN_MAX_BL)
        rt.warmup()
        _RUNTIME[idx] = rt
    except Exception:
        logger.warning(
            "KF decode attention unavailable; trtllm-gen decode stays in use.",
            exc_info=True,
        )
        _RUNTIME[idx] = None
    return _RUNTIME[idx] is not None
