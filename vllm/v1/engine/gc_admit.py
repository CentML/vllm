# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze-on-admit GC policy for the EngineCore process (``VLLM_GC_FREEZE_ADMIT``, default off) and an
env-gated GC timing line (``VLLM_GC_STATS_S``, default off; independent of the policy).

Why: on the Rubin HE2 stack every automatic GC pass in steady-state serving collects 0 objects (hostprobe
gcinfo, r32 + r256, 2,793 passes incl. 19 gen-2 passes; GB300 C384 the same), yet GC costs 2.5-2.7% of the
EngineCore main thread at r256 (gen0 ~0.9 ms, gen1 ~8 ms, gen2 ~50 ms, max 84 ms), 93% of it in mixed (admission)
steps, ~1.1 ms per admitted request. The passes keep re-traversing the long-lived per-request lists (76-93K-token
prompt / all-token lists): the young passes right after admission and every gen-2 pass while the request lives.
``gc.freeze()`` (O(1) in CPython: it splices the generation lists into the permanent generation) right after a
request is built moves those objects out of every later traversal. Frozen objects are still freed by reference
counting when the request finishes (deallocation unlinks them from the permanent list), so only cyclic garbage
reachable at freeze time could be retained; none is produced in steady state (0 collected).

Exact: GC never collected anything in serving, so object lifetimes, allocation order and every GPU launch are
unchanged; this only removes traversal work. The F88 GC pump moved the same traversals into GPU-wait windows (and
forced bigger passes when no window came); this removes them instead. Unlike GB300's F42 combo it starts no thread,
does not touch ``sys.setswitchinterval`` and adds nothing to the request path but one O(1) call per admission.

Env:
    VLLM_GC_FREEZE_ADMIT=1             freeze after each ADD request is preprocessed (input thread)
    VLLM_GC_FREEZE_ADMIT_GUARD_S=S     leak guard: at most every S seconds (default 30), and only while the engine is
                                       idle, compare the process RSS (/proc/self/statm, ~10 us) with the RSS after
                                       the last guard run
    VLLM_GC_FREEZE_ADMIT_GUARD_RSS_MB  RSS growth that triggers unfreeze + full collect + refreeze (default 4096)
    VLLM_GC_FREEZE_ADMIT_COUNT_S=S     at most every S idle seconds (default 300) also count the permanent generation
                                       (an O(n) list walk, a few ms) for the stats line and the object trigger
    VLLM_GC_FREEZE_ADMIT_GUARD=N       frozen-object growth that triggers the same (default 4,000,000)
    VLLM_GC_STATS_S=S                  every S seconds log one line of GC passes per generation (count, total / max
                                       ms, collected; main thread vs other threads). gc.callbacks timing, ~1 us per
                                       pass. Works with or without the freeze policy.
"""

import gc
import os
import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

MODE = int(os.environ.get("VLLM_GC_FREEZE_ADMIT", "0") or 0)
GUARD_S = float(os.environ.get("VLLM_GC_FREEZE_ADMIT_GUARD_S", "30"))
GUARD_RSS = int(float(os.environ.get("VLLM_GC_FREEZE_ADMIT_GUARD_RSS_MB", "4096")) * 2**20)
COUNT_S = float(os.environ.get("VLLM_GC_FREEZE_ADMIT_COUNT_S", "300"))
GUARD = int(os.environ.get("VLLM_GC_FREEZE_ADMIT_GUARD", "4000000"))
STATS_S = float(os.environ.get("VLLM_GC_STATS_S", "0") or 0)


def _rss() -> int | None:
    try:
        with open("/proc/self/statm", "rb") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        try:
            import psutil

            return psutil.Process().memory_info().rss
        except Exception:
            return None


class _State:
    base: int | None = None        # permanent-generation size after the last guard run / init
    base_rss: int | None = None    # process RSS after the last guard run / init
    admits = 0
    last_guard = 0.0
    last_count = 0.0
    guard_runs = 0


S = _State()


class _Timing:
    main_tid: int | None = None
    t0: dict = {}
    acc: dict = {}                 # (gen, main?) -> [n, total_s, max_s, collected]
    win0 = 0.0


TM = _Timing()


def _gc_cb(phase: str, info: dict) -> None:
    tid = threading.get_ident()
    if phase == "start":
        TM.t0[tid] = time.perf_counter()
        return
    t0 = TM.t0.pop(tid, None)
    if t0 is None:
        return
    now = time.perf_counter()
    d = now - t0
    a = TM.acc.setdefault((min(info.get("generation", 0), 2), tid == TM.main_tid), [0, 0.0, 0.0, 0])
    a[0] += 1
    a[1] += d
    a[2] = max(a[2], d)
    a[3] += info.get("collected", 0)
    if now - TM.win0 >= STATS_S:
        win = now - TM.win0
        parts = []
        for g in (0, 1, 2):
            for m in (True, False):
                x = TM.acc.get((g, m))
                if x:
                    parts.append(f"gen{g}{'' if m else '(other)'} n={x[0]} total={1e3 * x[1]:.1f}ms "
                                 f"max={1e3 * x[2]:.1f}ms collected={x[3]}")
        main_ms = sum(v[1] for (g, m), v in TM.acc.items() if m) * 1e3
        logger.info("[gcf] gc timing %.0fs: %s; main-thread GC %.1f ms (%.2f%%)", win, "; ".join(parts) or "none",
                    main_ms, 100 * main_ms / 1e3 / max(win, 1e-9))
        TM.acc = {}
        TM.win0 = now


def init() -> None:
    """Call once, on the EngineCore main thread, after the start-up heap is frozen (EngineCore.__init__)."""
    if STATS_S > 0:
        TM.main_tid = threading.get_ident()
        TM.win0 = time.perf_counter()
        gc.callbacks.append(_gc_cb)
        logger.info("[gcf] gc timing line every %.0f s", STATS_S)
    if MODE <= 0:
        return
    S.base = gc.get_freeze_count()
    S.base_rss = _rss()
    S.last_guard = S.last_count = time.monotonic()
    logger.info(
        "[gcf] GC freeze-on-admit on: mode=%d frozen=%d rss=%.0f MB thresholds=%s guard: rss +%d MB / frozen +%d "
        "(idle only, every >=%.0f / %.0f s)", MODE, S.base, (S.base_rss or 0) / 2**20, gc.get_threshold(),
        GUARD_RSS // 2**20, GUARD, GUARD_S, COUNT_S,
    )


def on_admit() -> None:
    """After a new Request is built: move all tracked objects to the permanent generation (O(1))."""
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
            "[gcf] stats: admits=%d frozen=%d (+%d) rss=%.0f MB (+%.0f) gc collections=%s collected=%s guard_runs=%d",
            S.admits, frozen, frozen - S.base, (rss or 0) / 2**20, ((rss or 0) - (S.base_rss or 0)) / 2**20,
            [s["collections"] for s in stats], [s["collected"] for s in stats], S.guard_runs,
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
        "[gcf] leak guard: rss %.0f MB, frozen %s before; unfreeze + collect found %d objects (%.1f ms); "
        "refrozen %d, rss now %.0f MB", (rss or 0) / 2**20, frozen, collected, 1e3 * (time.perf_counter() - t0),
        S.base, (S.base_rss or 0) / 2**20,
    )
