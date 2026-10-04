# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime for the Kernel Factory paged-FP8 prefill attention kernel.

Kernel: ``vllm/third_party/kf_prefill_attn/kernel.py`` (verbatim KF solution).
Contract (the FlashInfer trtllm-gen prefill call site in ``flashinfer.py``): SM107,
FP8 e4m3 Q ``[T, 16, 256]`` (contiguous), paged FP8 KV ``[P, 2, 128, 512]`` (HND,
K|V packed, contiguous pages), block table ``[B, 2057]`` int32 (row stride 2057),
causal attention over a cached prefix, BF16 out ``[T, 16, 256]`` (contiguous),
``bmm1_scale`` = softmax scale x q_scale x k_scale, ``bmm2_scale`` = v_scale.
No window, sinks or soft cap.

Per scheduler step (metadata builder, once for all attention layers): ``plan()``
runs the solution's host planner (exact C++ port, ``planner.py``, GIL released)
into a pinned ring slot and copies it to a device ring slot (non-blocking, current
stream). No synchronization, allocation or memset per step.
Per layer (forward): ``launch()`` calls the AOT-compiled CuTe DSL kernel through
TVM-FFI (a few us of host time). Split partials live in the upper half of the
shared trtllm workspace (FlashInfer keeps its semaphores at the start of the lower
half and caps its partials at half of it); the merge counters self-reset.
``warmup()`` builds the planner and compiles every kernel form the planner can
pick, so nothing compiles while serving; a missing form makes ``plan()`` return
None (the caller keeps FlashInfer).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

HQ, HKV, HDIM, PAGE, MAXP = 16, 2, 256, 128, 2057
PLAN_F, MRG_F = 12, 8
_RING = 4
_CAP_WORK = 32768
_CAP_MRG = 32768
_CNT_INTS = 16384
# The packed (fast) kernel forms cover bmm1 <= 1/16 and bmm2 <= 1 (the solution's
# run() switches to its precise form above that).
_UNIT_BMM1 = 0.0625


@dataclass
class KfPrefillPlan:
    """One step's work list on the device (shared by every attention layer)."""

    variant: int
    precise_variant: int
    nwork: int
    nslot: int
    plan: torch.Tensor
    bins: torch.Tensor
    mrg: torch.Tensor
    mbin: torch.Tensor
    po: torch.Tensor
    pml: torch.Tensor


def search_grids(kmod: Any, preset: str) -> tuple[tuple, tuple, tuple, bool]:
    """Planner search space: ``full`` = the solution's own search; ``r2``/``r3`` =
    base wave grid only (r3: waves 1/4..4). Offline (VR, 197 real launches) r2/r3
    cost +0.7/+1.3% of kernel time vs full, for 1/5 and 1/10 of the planner time.
    """
    if preset == "full":
        return (
            tuple(kmod._KGRID),
            tuple(kmod._BASE_WAVE_GRID),
            tuple(kmod._DENSE_WAVE_GRID),
            True,
        )
    if preset == "r2":
        return ((), tuple(kmod._BASE_WAVE_GRID), (), False)
    if preset == "r3":
        return ((), (0.25, 0.5, 1, 2, 4), (), False)
    raise ValueError(f"VLLM_KF_PREFILL_ATTN_SEARCH={preset!r}: use full, r2 or r3")


def reachable_variants(nops: int, precise: bool) -> list[int]:
    """Kernel forms ``setup()`` can pick: mode m in 0..5 (base 10 m); packed forms
    0/4 (+5 replay) on the plain and skewed-load (+NOPS) tables, long forms 1/2 (+5)
    on the half-Q tmem table (+2 NOPS); the precise form 3 on all three tables.
    """
    out = []
    for m in range(6):
        b = 10 * m
        out += [b + k for k in (0, 4, 5, 9)]
        out += [nops + b + k for k in (0, 4, 5, 9)]
        out += [2 * nops + b + k for k in (1, 2, 6, 7)]
        if precise:
            out += [b + 3, nops + b + 3, 2 * nops + b + 3]
    return sorted(out)


class KfPrefillAttn:
    def __init__(self, device: torch.device, workspace: torch.Tensor):
        from vllm.third_party.kf_prefill_attn import kernel as kmod
        from vllm.v1.attention.ops.kf_prefill_attn import planner

        self.k = kmod
        self.planner = planner
        self.device = device
        props = torch.cuda.get_device_properties(device)
        self.nsm = props.multi_processor_count
        self.grids = search_grids(kmod, envs.VLLM_KF_PREFILL_ATTN_SEARCH)
        self.min_kv = envs.VLLM_KF_PREFILL_ATTN_MIN_KV
        nb = self.nsm + 2
        # Split partials: fp32 [nslot, 256, 128] + stats [nslot, 2, 128] in the
        # upper half of the trtllm workspace.
        half = workspace.numel() // 2
        self.pml_off = kmod.NSLOT_CAP * HDIM * kmod.MROW * 4
        need = self.pml_off + kmod.NSLOT_CAP * 2 * kmod.MROW * 4
        if half < need:
            raise RuntimeError(f"trtllm workspace half {half} B < {need} B partials")
        self.scratch = workspace[half:]
        self.cnt = torch.zeros(_CNT_INTS, dtype=torch.int32, device=device)
        self.cnt_base = int(self.cnt.data_ptr())
        self.host: list[tuple[torch.Tensor, ...]] = []
        self.dev: list[tuple[torch.Tensor, ...]] = []
        self.events: list[torch.Event | None] = [None] * _RING
        for _ in range(_RING):
            self.host.append(
                (
                    torch.empty(
                        (_CAP_WORK, PLAN_F), dtype=torch.int32, pin_memory=True
                    ),
                    torch.empty(nb, dtype=torch.int32, pin_memory=True),
                    torch.empty((_CAP_MRG, MRG_F), dtype=torch.int32, pin_memory=True),
                    torch.empty(nb, dtype=torch.int32, pin_memory=True),
                )
            )
            self.dev.append(
                tuple(torch.empty_like(t, device=device) for t in self.host[-1])
            )
        self.slot = 0
        self.compiled: dict[int, Any] = {}
        self.stats = {"plans": 0, "fallback_capacity": 0, "fallback_variant": 0}
        self.plan_us = 0.0

    def _partials(self, nslot: int) -> tuple[torch.Tensor, torch.Tensor]:
        mrow = self.k.MROW
        po = self.scratch[: nslot * HDIM * mrow * 4].view(torch.float32)
        pml = self.scratch[self.pml_off : self.pml_off + nslot * 2 * mrow * 4]
        return po.view(nslot, HDIM, mrow), pml.view(torch.float32).view(nslot, 2, mrow)

    def warmup(self, precise: bool, variants: list[int] | None = None) -> None:
        """Build the C++ planner and AOT-compile (TVM-FFI) every reachable kernel
        form (``variants`` restricts the set; tests only), then load each module
        with one launch on an empty work list.
        """
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        def dyn(t: torch.Tensor, d: int) -> Any:
            return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=d)

        t0 = time.perf_counter()
        self.planner.build()
        t1 = time.perf_counter()
        k = self.k
        dev = self.device
        fp8 = torch.float8_e4m3fn
        dq = torch.zeros(16, HQ, HDIM, dtype=fp8, device=dev)
        dkv = torch.zeros(2, HKV, PAGE, 2 * HDIM, dtype=fp8, device=dev)
        dbt = torch.zeros(1, MAXP, dtype=torch.int32, device=dev)
        do = torch.zeros(16, HQ, HDIM, dtype=torch.bfloat16, device=dev)
        plan, bins, mrg, mbin = self.dev[0]
        po, pml = self._partials(1)
        args = (
            dyn(dq, 2),
            dyn(dkv, 3),
            dyn(dbt, 1),
            dyn(do, 2),
            dyn(plan[:1], 1),
            dyn(bins, 0),
            dyn(po, 2),
            dyn(pml, 2),
            dyn(mrg[:1], 1),
            dyn(mbin, 0),
            cutlass.Int64(self.cnt_base),
            cutlass.Float32(0.0625),
            cutlass.Float32(1.0),
            cutlass.Int32(1),
            cutlass.Int32(1),
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if variants is None:
            variants = reachable_variants(k.NOPS, precise)
        for v in variants:
            if v in self.compiled:
                continue
            mode = (v % k.NOPS) // 10
            grid = self.nsm if mode == 0 else (self.nsm // 2) * 2
            self.compiled[v] = cute.compile(
                k._OPS[v], *args, grid, stream, options="--enable-tvm-ffi"
            )
        # Every CTA bin and merge bin empty: no CTA reads Q/KV or writes out.
        bins.zero_()
        mbin.zero_()
        for fn in self.compiled.values():
            fn(dq, dkv, dbt, do, plan[:1], bins, po, pml, mrg[:1], mbin,
               self.cnt_base, 0.0625 * k.LOG2E, 1.0, 1, 1, stream)  # fmt: skip
        torch.accelerator.synchronize()
        if not bool((self.cnt == 0).all()):
            raise RuntimeError("KF prefill attention: merge counters not reset")
        logger.info(
            "KF prefill attention: planner built in %.1fs, %d kernel forms "
            "compiled in %.1fs (search=%s, min_kv=%d).",
            t1 - t0,
            len(self.compiled),
            time.perf_counter() - t1,
            envs.VLLM_KF_PREFILL_ATTN_SEARCH,
            self.min_kv,
        )

    def plan(self, query_lens: list[int], seq_lens: list[int]) -> KfPrefillPlan | None:
        """Plan one step's prefill launch; None keeps FlashInfer for this step."""
        if not self.compiled or max(seq_lens) < self.min_kv:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        i = self.slot
        ev = self.events[i]
        if ev is not None:
            ev.synchronize()  # this pinned slot's H2D (RING steps ago) has finished
        hp, hb, hm, hmb = self.host[i]
        t0 = time.perf_counter()
        res = self.planner.plan_into(
            query_lens, seq_lens, self.nsm, hp, hb, hm, hmb, *self.grids
        )
        self.plan_us += (time.perf_counter() - t0) * 1e6
        nwork, nbins, nmrg, nmbin, nslot, ngrp, variant, precise_variant = res[:8]
        if nwork < 0 or nslot > self.k.NSLOT_CAP or 4 * max(ngrp, 1) + 4 > _CNT_INTS:
            self.stats["fallback_capacity"] += 1
            return None
        if variant not in self.compiled:
            self.stats["fallback_variant"] += 1
            return None
        dp, db, dm, dmb = self.dev[i]
        dp[:nwork].copy_(hp[:nwork], non_blocking=True)
        db[:nbins].copy_(hb[:nbins], non_blocking=True)
        dm[:nmrg].copy_(hm[:nmrg], non_blocking=True)
        dmb[:nmbin].copy_(hmb[:nmbin], non_blocking=True)
        if ev is None:
            ev = self.events[i] = torch.Event()
        ev.record()
        self.slot = (i + 1) % _RING
        self.stats["plans"] += 1
        po, pml = self._partials(nslot)
        return KfPrefillPlan(
            variant=variant,
            precise_variant=precise_variant,
            nwork=nwork,
            nslot=nslot,
            plan=dp[:nwork],
            bins=db[:nbins],
            mrg=dm[:nmrg],
            mbin=dmb[:nmbin],
            po=po,
            pml=pml,
        )

    def launch(
        self,
        plan: KfPrefillPlan,
        q: torch.Tensor,
        kv: torch.Tensor,
        block_tables: torch.Tensor,
        out: torch.Tensor,
        bmm1_scale: float,
        bmm2_scale: float,
    ) -> bool:
        """Run one layer's prefill attention into ``out``; False = fall back."""
        precise = bmm1_scale > _UNIT_BMM1 or bmm2_scale > 1.0
        fn = self.compiled.get(plan.precise_variant if precise else plan.variant)
        if fn is None:
            return False
        import cuda.bindings.driver as cuda

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        fn(q, kv, block_tables, out, plan.plan, plan.bins, plan.po, plan.pml,
           plan.mrg, plan.mbin, self.cnt_base, float(bmm1_scale) * self.k.LOG2E,
           float(bmm2_scale), plan.nwork, plan.nslot, stream)  # fmt: skip
        return True


_RUNTIME: dict[int, KfPrefillAttn | None] = {}


def kf_supported_kv(kv: torch.Tensor) -> bool:
    """The kernel hard-codes the page geometry; check the per-layer cache view."""
    return (
        kv.dtype == torch.float8_e4m3fn
        and kv.dim() == 4
        and tuple(kv.shape[1:]) == (HKV, PAGE, 2 * HDIM)
        and kv.stride() == (HKV * PAGE * 2 * HDIM, PAGE * 2 * HDIM, 2 * HDIM, 1)
    )


def get_runtime(device: torch.device | str) -> KfPrefillAttn | None:
    """The process' runtime once ``warmup_runtime`` built it, else None."""
    return _RUNTIME.get(torch.device(device).index or 0)


def warmup_runtime(
    device: torch.device, workspace: torch.Tensor, precise: bool
) -> bool:
    """Create and warm the runtime (startup only). A failure logs and keeps
    FlashInfer.
    """
    idx = torch.device(device).index or 0
    if idx in _RUNTIME:
        return _RUNTIME[idx] is not None
    try:
        rt = KfPrefillAttn(device, workspace)
        rt.warmup(precise)
        _RUNTIME[idx] = rt
    except Exception:
        logger.warning(
            "KF prefill attention unavailable; FlashInfer prefill stays in use.",
            exc_info=True,
        )
        _RUNTIME[idx] = None
    return _RUNTIME[idx] is not None
