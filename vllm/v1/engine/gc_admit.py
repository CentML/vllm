# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze-on-admit GC policy for the EngineCore process (``VLLM_GC_FREEZE_ADMIT``,
default off) plus env-gated GC observability (``VLLM_GC_STATS_S``,
``VLLM_GC_NVTX``), both independent of the policy.

Why: in steady-state serving the automatic GC passes collect nothing, but each
pass re-traverses the long-lived per-request lists (prompt / all-token lists of
tens of thousands of ints). Young passes right after an admission walk the new
request's lists; every gen-2 pass walks the lists of every live request. On
VR C512 (128 live requests per engine) that shows up as ~37-63 ms GPU-idle host
stalls a few times per minute and ~1-3 ms stalls in admission steps.

``gc.freeze()`` (O(1): it splices the generation lists into the permanent
generation) right after a request is built moves those objects out of every
later traversal. Frozen objects are still freed by reference counting when the
request finishes, so only cyclic garbage reachable at freeze time could be
retained; the rate-limited, idle-only leak guard bounds that. Exact: no tensor
and no GPU launch is touched; object lifetimes are unchanged while GC collects
nothing (check the ``collected`` counts of the stats line).

Env (read once at import):
    VLLM_GC_FREEZE_ADMIT=1             EngineCore: gc.freeze() after each ADD
                                       request is preprocessed (input thread)
    VLLM_GC_FREEZE_ADMIT_GUARD_S=S     leak guard period, idle only (default 30)
    VLLM_GC_FREEZE_ADMIT_GUARD_RSS_MB  RSS growth since the last guard that
                                       triggers unfreeze + collect + refreeze
                                       (default 4096)
    VLLM_GC_FREEZE_ADMIT_COUNT_S=S     at most every S idle seconds also count
                                       the permanent generation (default 300)
    VLLM_GC_FREEZE_ADMIT_GUARD=N       frozen-object growth that triggers the
                                       same (default 4,000,000)
    VLLM_GC_STATS_S=S                  every S seconds log one line of GC passes
                                       per generation (count, total / max ms,
                                       collected; main thread vs others)
    VLLM_GC_NVTX=1                     one NVTX range ``gc<gen>`` per GC pass on
                                       the thread that runs it (Nsight
                                       attribution of host stalls)
"""

import gc
import os
import threading
import time
from collections.abc import Callable

from vllm.logger import init_logger

logger = init_logger(__name__)

MODE = int(os.environ.get("VLLM_GC_FREEZE_ADMIT", "0") or 0)
GUARD_S = float(os.environ.get("VLLM_GC_FREEZE_ADMIT_GUARD_S", "30"))
GUARD_RSS = int(
    float(os.environ.get("VLLM_GC_FREEZE_ADMIT_GUARD_RSS_MB", "4096")) * 2**20
)
COUNT_S = float(os.environ.get("VLLM_GC_FREEZE_ADMIT_COUNT_S", "300"))
GUARD = int(os.environ.get("VLLM_GC_FREEZE_ADMIT_GUARD", "4000000"))
STATS_S = float(os.environ.get("VLLM_GC_STATS_S", "0") or 0)
NVTX = os.environ.get("VLLM_GC_NVTX", "0") == "1"


def _rss() -> int | None:
    try:
        with open("/proc/self/statm", "rb") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


class _State:
    base: int | None = None  # permanent-generation size after the last guard
    base_rss: int | None = None  # process RSS after the last guard run / init
    admits = 0
    last_guard = 0.0
    last_count = 0.0
    guard_runs = 0


S = _State()


class _Timing:
    role = ""
    main_tid: int | None = None
    t0: dict[int, float] = {}
    acc: dict[tuple[int, bool], list] = {}  # (gen, main?) -> [n, s, max_s, collected]
    win0 = 0.0


TM = _Timing()


def _stats_cb(phase: str, info: dict) -> None:
    tid = threading.get_ident()
    if phase == "start":
        TM.t0[tid] = time.perf_counter()
        return
    t0 = TM.t0.pop(tid, None)
    if t0 is None:
        return
    now = time.perf_counter()
    d = now - t0
    key = (min(info.get("generation", 0), 2), tid == TM.main_tid)
    a = TM.acc.setdefault(key, [0, 0.0, 0.0, 0])
    a[0] += 1
    a[1] += d
    a[2] = max(a[2], d)
    a[3] += info.get("collected", 0)
    if now - TM.win0 < STATS_S:
        return
    win = now - TM.win0
    parts = []
    for g in (0, 1, 2):
        for m in (True, False):
            x = TM.acc.get((g, m))
            if x:
                parts.append(
                    f"gen{g}{'' if m else '(other)'} n={x[0]} "
                    f"total={1e3 * x[1]:.1f}ms max={1e3 * x[2]:.1f}ms "
                    f"collected={x[3]}"
                )
    all_ms = sum(v[1] for v in TM.acc.values()) * 1e3
    logger.info(
        "[gcf] %s gc timing %.0fs: %s; all-thread GC %.1f ms (%.2f%%)",
        TM.role,
        win,
        "; ".join(parts) or "none",
        all_ms,
        100 * all_ms / 1e3 / max(win, 1e-9),
    )
    TM.acc = {}
    TM.win0 = now


def _make_nvtx_cb() -> Callable[[str, dict], None]:
    import torch

    push, pop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop
    names = ("gc0", "gc1", "gc2")

    def cb(phase: str, info: dict) -> None:
        if phase == "start":
            push(names[min(info.get("generation", 0), 2)])
        else:
            pop()

    cb._vllm_gc_nvtx = True  # type: ignore[attr-defined]
    return cb


def init(role: str = "engine") -> None:
    """Install the GC observability callbacks (any process) and, for the
    EngineCore (role ``engine``), arm the freeze-on-admit policy. Call once,
    after the start-up heap is frozen.
    """
    TM.role = role
    if NVTX and not any(getattr(cb, "_vllm_gc_nvtx", False) for cb in gc.callbacks):
        gc.callbacks.append(_make_nvtx_cb())
        logger.info("[gcf] %s: NVTX range per GC pass", role)
    if STATS_S > 0 and _stats_cb not in gc.callbacks:
        TM.main_tid = threading.get_ident()
        TM.win0 = time.perf_counter()
        gc.callbacks.append(_stats_cb)
        logger.info("[gcf] %s: gc timing line every %.0f s", role, STATS_S)
    if MODE <= 0 or role != "engine":
        return
    S.base = gc.get_freeze_count()
    S.base_rss = _rss()
    S.last_guard = S.last_count = time.monotonic()
    logger.info(
        "[gcf] GC freeze-on-admit on: frozen=%d rss=%.0f MB thresholds=%s "
        "guard: rss +%d MB / frozen +%d (idle only, every >=%.0f / %.0f s)",
        S.base,
        (S.base_rss or 0) / 2**20,
        gc.get_threshold(),
        GUARD_RSS // 2**20,
        GUARD,
        GUARD_S,
        COUNT_S,
    )


def on_admit() -> None:
    """After a new Request is built: move all tracked objects to the permanent
    generation (O(1)).
    """
    if MODE <= 0:
        return
    gc.freeze()
    S.admits += 1


def on_idle() -> None:
    """Engine idle (no work): rate-limited leak guard and stats line."""
    if MODE <= 0 or S.base is None:
        return
    now = time.monotonic()
    if now - S.last_guard < GUARD_S:
        return
    S.last_guard = now
    rss = _rss()
    trig = rss is not None and S.base_rss is not None and rss - S.base_rss > GUARD_RSS
    frozen = None
    if trig or now - S.last_count >= COUNT_S:
        S.last_count = now
        frozen = gc.get_freeze_count()
        stats = gc.get_stats()
        logger.info(
            "[gcf] stats: admits=%d frozen=%d (+%d) rss=%.0f MB (+%.0f) "
            "gc collections=%s collected=%s guard_runs=%d",
            S.admits,
            frozen,
            frozen - S.base,
            (rss or 0) / 2**20,
            ((rss or 0) - (S.base_rss or 0)) / 2**20,
            [s["collections"] for s in stats],
            [s["collected"] for s in stats],
            S.guard_runs,
        )
        trig = trig or frozen - S.base > GUARD
    if not trig:
        return
    t0 = time.perf_counter()
    gc.unfreeze()
    collected = gc.collect()
    gc.freeze()
    S.base = gc.get_freeze_count()
    S.base_rss = _rss()
    S.guard_runs += 1
    logger.warning(
        "[gcf] leak guard: rss %.0f MB, frozen %s before; unfreeze + collect "
        "found %d objects (%.1f ms); refrozen %d, rss now %.0f MB",
        (rss or 0) / 2**20,
        frozen,
        collected,
        1e3 * (time.perf_counter() - t0),
        S.base,
        (S.base_rss or 0) / 2**20,
    )
