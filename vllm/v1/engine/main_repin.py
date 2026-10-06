# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GB300 study (fix-repin): adaptive re-pinning of the EngineCore main thread.

On GB300 (Grace) nodes the busy EngineCore main thread intermittently enters a
"slow mode" (+25-45% host time per engine step, 2.5-8x LLC misses, same
clocks) on the cores of ONE socket; which socket differs per node, so a fixed
CPU map is not portable.  Moving only the main thread to a core on the other
socket resets it within seconds; moves inside the same socket do not (fix-host,
2026-10-01).

This watchdog measures the main thread's own host time per engine cycle (wall
time between returning from the blocking wait for step N-1's result and
starting the blocking wait for step N's result: update + schedule + launch),
on pure-decode cycles keyed by the decode batch size B.  The reference per B is
the best (lowest) 64-sample median seen by this worker or published by the
other DP workers of the node (shared JSON files).  When the 1-s window median
of host/reference stays above 1 + THR for HOLD seconds, the main thread (only
the calling thread: sched_setaffinity(0) acts on the calling thread on Linux)
is pinned to one core from the candidate pool, checked after SETTLE seconds,
and moved on if still slow (hysteresis: cooldown, bad-core TTL, exponential
backoff after failed episodes).  Default off.  No numerics change.

Env (all optional except the switch):
  VLLM_ENGINE_MAIN_REPIN=1|observe   enable; "observe" detects and logs only
  VLLM_ENGINE_MAIN_REPIN_POOL=remote comma-separated candidate tiers, tried in
        order within an episode: "own" = the thread's affinity at busy-loop
        start (the worker's taskset mask); "remote" = idle cores of the OTHER
        DP workers' masks on another NUMA node (never a core outside every
        worker mask, i.e. never the frontend/client cores, never a peer's
        current main-thread core); "list" = VLLM_ENGINE_MAIN_REPIN_CPUS.
  VLLM_ENGINE_MAIN_REPIN_CPUS=a-b+c  explicit candidate cores for tier "list"
  VLLM_ENGINE_MAIN_REPIN_THR=0.20    slow if median(host/ref) > 1 + THR
  VLLM_ENGINE_MAIN_REPIN_OK=0.15     recovered if median(host/ref) <= 1 + OK
                                     (a cross-socket main thread costs ~+8%)
  VLLM_ENGINE_MAIN_REPIN_HOLD_S=5    continuous slow time before acting
  VLLM_ENGINE_MAIN_REPIN_SETTLE_S=8  observation time after a move
  VLLM_ENGINE_MAIN_REPIN_COOLDOWN_S=20  minimum time between episodes
  VLLM_ENGINE_MAIN_REPIN_TRIES=3     moves per episode before backing off
  VLLM_ENGINE_MAIN_REPIN_OWN_TRIES=2 max moves inside tier "own" per episode
  VLLM_ENGINE_MAIN_REPIN_MAX_B=16    only pure-decode cycles with B <= MAX_B
  VLLM_ENGINE_MAIN_REPIN_PEER_DIR=/tmp/vllm-main-repin  ("" = no peers)
  VLLM_ENGINE_MAIN_REPIN_STATUS_S=60 status line period (0 = off)
  VLLM_ENGINE_MAIN_REPIN_DIAG=1      log per-CPU IPI rates + threads on the
                                     slow core at each episode
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

_WIN_S = 1.0  # evaluation window
_MIN_WIN_N = 24  # samples with a reference needed for a window verdict
_REF_N = 64  # samples per reference median
_PEER_S = 5.0  # peer file exchange period
_PEER_FRESH_S = 30.0
_PEER_SLACK = 0.05
_BAD_TTL_S = 120.0
_MAX_HOST_MS = 50.0  # longer "host" gaps are idle waits, not engine cycles
_BUSY_MAX = 0.30  # a candidate core busier than this over the hold window is skipped


def _parse_cpus(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.replace(",", "+").split("+"):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def _cpu_nodes() -> dict[int, int]:
    nodes: dict[int, int] = {}
    for path in glob.glob("/sys/devices/system/node/node[0-9]*/cpulist"):
        try:
            node = int(path.split("/node")[-1].split("/")[0])
            with open(path) as f:
                for c in _parse_cpus(f.read().strip()):
                    nodes[c] = node
        except (OSError, ValueError):
            continue
    return nodes


def _proc_stat_busy() -> dict[int, tuple[int, int]]:
    """Per-CPU (busy, total) jiffies from /proc/stat."""
    out: dict[int, tuple[int, int]] = {}
    try:
        with open("/proc/stat") as f:
            for line in f:
                if not line.startswith("cpu"):
                    break
                if line.startswith("cpu "):
                    continue
                parts = line.split()
                v = [int(x) for x in parts[1:9]]
                idle = v[3] + v[4]
                out[int(parts[0][3:])] = (sum(v) - idle, sum(v))
    except (OSError, ValueError):
        pass
    return out


def _proc_ipis() -> dict[str, list[int]]:
    """Per-CPU counts of the IPI rows of /proc/interrupts (arm64: IPI0..)."""
    out: dict[str, list[int]] = {}
    try:
        with open("/proc/interrupts") as f:
            ncpu = len(f.readline().split())
            for line in f:
                s = line.lstrip()
                if not s.startswith("IPI"):
                    continue
                parts = s.split()
                out[parts[0].rstrip(":")] = [int(x) for x in parts[1 : 1 + ncpu]]
    except (OSError, ValueError):
        pass
    return out


def _current_cpu() -> int:
    try:
        with open("/proc/thread-self/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[36])
    except (OSError, ValueError, IndexError):
        return -1


def _threads_on_cpu(cpu: int) -> list[str]:
    """Threads of this process whose last CPU is `cpu` (name:tid)."""
    out = []
    for path in glob.glob("/proc/self/task/*/stat"):
        try:
            with open(path) as f:
                s = f.read()
            head, rest = s.rsplit(")", 1)
            if int(rest.split()[36]) == cpu:
                out.append(f"{head.split('(', 1)[1]}:{head.split()[0]}")
        except (OSError, ValueError, IndexError):
            continue
    return out


def _worker_index() -> str:
    port = os.environ.get("DYN_SYSTEM_PORT", "")
    idx = os.environ.get("VLLM_ENGINE_WORKER_INDEX") or (
        str(int(port) - 18081) if port.isdigit() else ""
    )
    return idx or f"p{os.getpid()}"


class MainThreadRepin:
    """Per-EngineCore watchdog; all methods run on the busy-loop thread."""

    def __init__(self, mode: str) -> None:
        env = os.environ.get
        self.observe = mode.strip().lower() == "observe"
        self.thr = float(env("VLLM_ENGINE_MAIN_REPIN_THR", "0.20"))
        self.ok = float(env("VLLM_ENGINE_MAIN_REPIN_OK", "0.15"))
        self.hold_s = float(env("VLLM_ENGINE_MAIN_REPIN_HOLD_S", "5"))
        self.settle_s = float(env("VLLM_ENGINE_MAIN_REPIN_SETTLE_S", "8"))
        self.cooldown_s = float(env("VLLM_ENGINE_MAIN_REPIN_COOLDOWN_S", "20"))
        self.tries = int(env("VLLM_ENGINE_MAIN_REPIN_TRIES", "3"))
        self.own_tries = int(env("VLLM_ENGINE_MAIN_REPIN_OWN_TRIES", "2"))
        self.max_b = int(env("VLLM_ENGINE_MAIN_REPIN_MAX_B", "16"))
        self.max_tok = int(env("VLLM_ENGINE_MAIN_REPIN_MAX_TOK_PER_REQ", "8"))
        self.status_s = float(env("VLLM_ENGINE_MAIN_REPIN_STATUS_S", "60"))
        self.diag = env("VLLM_ENGINE_MAIN_REPIN_DIAG", "1") == "1"
        self.pool = [
            t.strip()
            for t in env("VLLM_ENGINE_MAIN_REPIN_POOL", "remote").split(",")
            if t.strip()
        ]
        self.list_cpus = _parse_cpus(env("VLLM_ENGINE_MAIN_REPIN_CPUS", ""))
        self.peer_dir = env("VLLM_ENGINE_MAIN_REPIN_PEER_DIR", "/tmp/vllm-main-repin")
        self.idx = _worker_index()
        self.tag = f"[main-repin w{self.idx}]"
        self.home = sorted(os.sched_getaffinity(0))
        self.nodes = _cpu_nodes()
        home_nodes = {self.nodes.get(c, -1) for c in self.home}
        self.home_node = min(home_nodes) if len(home_nodes) == 1 else -1

        # measurement state
        self.t_last_ret: float | None = None
        self.n_launch = 0
        self.launch_b = 0
        self.prev_b = 0  # class of the step whose output the cycle processes
        self.ref_buf: dict[int, list[float]] = {}
        self.ref_own: dict[int, float] = {}
        self.ref_peer: dict[int, float] = {}
        self.win: list[float] = []  # host/ref ratios in the current window
        self.win_t0 = time.perf_counter()
        self.settle: list[float] = []
        # state machine
        self.state = "normal"  # normal | settle
        self.slow_since: float | None = None
        self.last_verdict_t = self.win_t0
        self.next_episode_t = 0.0
        self.episode_moves = 0
        self.episode_own = 0
        self.episode_from = -1
        self.fail_episodes = 0
        self.move_t = 0.0
        self.bad_until: dict[int, float] = {}
        self.snap_stat: dict[int, tuple[int, int]] = {}
        self.snap_ipi: dict[str, list[int]] = {}
        self.snap_t = 0.0
        self.last_ratio = 0.0
        # counters
        self.t_start = self.win_t0
        self.slow_s = 0.0
        self.windows = 0
        self.slow_windows = 0
        self.moves = 0
        self.recovered = 0
        self.failed = 0
        self.next_status_t = self.win_t0 + (self.status_s or 1e18)
        self.next_peer_t = 0.0
        self.peers: dict[str, dict] = {}
        self._remote_cache_t = -1.0
        self._remote_cache: list[dict] = []
        self.own_disabled = False
        self.cur_cpu = _current_cpu()
        self.main_tid = threading.get_native_id()
        self.own_failed_episodes = 0
        self.episode_tiers: list[str] = []
        logger.info(
            "%s enabled (mode=%s pool=%s thr=%.2f hold=%.0fs settle=%.0fs "
            "max_b=%d home=%s node=%d peer_dir=%s)",
            self.tag,
            "observe" if self.observe else "repin",
            ",".join(self.pool),
            self.thr,
            self.hold_s,
            self.settle_s,
            self.max_b,
            _fmt_cpus(self.home),
            self.home_node,
            self.peer_dir or "-",
        )

    # ------------------------------------------------------------------ hooks
    def _decode_b(self, so) -> int:
        n = len(so.num_scheduled_tokens)
        if n == 0 or so.scheduled_new_reqs:
            return 0
        if so.total_num_scheduled_tokens > n * self.max_tok:
            return 0
        return n

    def note_launch(self, scheduler_output) -> None:
        self.n_launch += 1
        self.launch_b = self._decode_b(scheduler_output)

    def on_cycle(self, t_wait0: float, t_wait1: float, popped_output) -> None:
        """Called after the blocking wait for a step's result returned."""
        last = self.t_last_ret
        self.t_last_ret = t_wait1
        if last is None:  # first cycle: anchor the clocks to the hook's clock
            self.win_t0 = self.t_start = self.last_verdict_t = t_wait1
            self.next_status_t = t_wait1 + (self.status_s or 1e18)
        b_pop = self._decode_b(popped_output)
        b_upd, self.prev_b = self.prev_b, b_pop
        n_launch, self.n_launch = self.n_launch, 0
        if last is not None and n_launch == 1:
            b = self.launch_b
            host = (t_wait0 - last) * 1e3
            if 0 < b <= self.max_b and b == b_upd and 0.0 < host < _MAX_HOST_MS:
                self._sample(b, host)
        if t_wait1 - self.win_t0 >= _WIN_S:
            self._evaluate(t_wait1)

    # ------------------------------------------------------------ measurement
    def _sample(self, b: int, host: float) -> None:
        buf = self.ref_buf.setdefault(b, [])
        buf.append(host)
        if len(buf) >= _REF_N:
            med = statistics.median(buf)
            buf.clear()
            if med < self.ref_own.get(b, 1e9):
                self.ref_own[b] = med
        ref = self._ref(b)
        if ref:
            self.win.append(host / ref)

    def _ref(self, b: int) -> float:
        own = self.ref_own.get(b)
        peer = self.ref_peer.get(b)
        if peer is not None:
            peer *= 1.0 + _PEER_SLACK
            return min(own, peer) if own is not None else peer
        return own or 0.0

    def _evaluate(self, now: float) -> None:
        dt = now - self.win_t0
        win, self.win, self.win_t0 = self.win, [], now
        if now >= self.next_peer_t:
            self._exchange_peers(now)
        ratio = statistics.median(win) if len(win) >= _MIN_WIN_N else None
        if ratio is not None:
            self.windows += 1
            self.last_ratio = ratio
            self.last_verdict_t = now
            if ratio > 1.0 + self.thr:
                self.slow_windows += 1
                self.slow_s += dt
        if self.state == "settle":
            if ratio is not None and now - self.move_t >= 1.0:
                self.settle.append(ratio)
            if now - self.move_t >= self.settle_s:
                self._settle_verdict(now)
        else:
            self._normal(now, ratio)
        if now >= self.next_status_t:
            self._status(now)

    def _normal(self, now: float, ratio: float | None) -> None:
        if ratio is None:
            if self.slow_since is not None and now - self.last_verdict_t > 10.0:
                self.slow_since = None  # load went away
            return
        if ratio <= 1.0 + self.thr:
            self.slow_since = None
            return
        if self.slow_since is None:
            self.slow_since = now
            if not self.observe:
                self.snap_stat = _proc_stat_busy()
            if self.diag and now - self.snap_t > 20.0:
                # /proc/interrupts is large on 144 CPUs: read it off-thread
                self.snap_t = now
                self.snap_ipi = {}
                threading.Thread(
                    target=self._snap_ipi_bg, name="main-repin-diag", daemon=True
                ).start()
            return
        if now - self.slow_since < self.hold_s or now < self.next_episode_t:
            return
        # start an episode
        self.episode_moves = 0
        self.episode_own = 0
        self.episode_tiers = []
        self.episode_from = self.cur_cpu = _current_cpu()
        self._log_episode(now, ratio)
        if self.observe:
            self.slow_since = None
            self.next_episode_t = now + self.cooldown_s
            return
        if self.episode_from >= 0:
            self.bad_until[self.episode_from] = now + _BAD_TTL_S
        self._move(now, ratio)

    def _settle_verdict(self, now: float) -> None:
        self.state = "normal"
        self.slow_since = None
        cpu = _current_cpu()
        if len(self.settle) < 2:
            logger.info(
                "%s settle on cpu %d inconclusive (no decode load); staying",
                self.tag,
                cpu,
            )
            self.next_episode_t = now + self.cooldown_s
            return
        r = statistics.median(self.settle)
        if r <= 1.0 + self.ok:
            self.recovered += 1
            self.fail_episodes = 0
            self.next_episode_t = now + self.cooldown_s
            if self.episode_tiers[-1:] == ["remote"] and "own" in self.episode_tiers:
                self.own_failed_episodes += 1
                if self.own_failed_episodes >= 2 and not self.own_disabled:
                    self.own_disabled = True
                    logger.info(
                        "%s own-mask moves failed in %d episodes that remote moves "
                        "fixed; skipping tier own from now on",
                        self.tag,
                        self.own_failed_episodes,
                    )
            logger.info(
                "%s RECOVERED on cpu %d: host/ref %.2f (episode from cpu %d, %d move(s))",
                self.tag,
                cpu,
                r,
                self.episode_from,
                self.episode_moves,
            )
            return
        self.bad_until[cpu] = now + _BAD_TTL_S
        if self.episode_moves < self.tries:
            logger.info(
                "%s still slow on cpu %d: host/ref %.2f -> next core", self.tag, cpu, r
            )
            self._move(now, r)
            return
        self.failed += 1
        self.fail_episodes += 1
        backoff = min(600.0, 60.0 * 2 ** (self.fail_episodes - 1))
        self.next_episode_t = now + backoff
        logger.info(
            "%s episode FAILED after %d moves (host/ref %.2f on cpu %d); "
            "backing off %.0f s",
            self.tag,
            self.episode_moves,
            r,
            cpu,
            backoff,
        )

    # ------------------------------------------------------------------ moves
    def _candidates(self, now: float, cur: int) -> tuple[str, list[int]]:
        busy = {}
        if self.snap_stat:
            cur_stat = _proc_stat_busy()
            for c, (b1, t1) in cur_stat.items():
                b0, t0 = self.snap_stat.get(c, (b1, t1))
                busy[c] = (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0
        peer_main = {
            p.get("cpu", -1)
            for p in self.peers.values()
            if now - p.get("_seen", 0) < _PEER_FRESH_S
        }
        owner_of: dict[int, dict] = {}
        for tier in self.pool:
            if tier == "own":
                if self.episode_own >= self.own_tries or self.own_disabled:
                    continue
                cands = list(self.home)
            elif tier == "remote":
                if self.home_node < 0:
                    continue
                cands = []
                for owner in self._remote_masks(now):
                    cands.extend(
                        c for c in owner.get("home", []) if c not in self.home
                    )
                    owner_of.update({c: owner for c in owner.get("home", [])})
            elif tier == "list":
                cands = list(self.list_cpus)
            else:
                continue
            cands = [
                c
                for c in cands
                if c != cur
                and c not in peer_main
                and self.bad_until.get(c, 0.0) <= now
                and busy.get(c, 0.0) <= _BUSY_MAX
            ]
            if cands:
                if tier == "remote":
                    # keep the preferred neighbour mask first, least busy inside it
                    rank = {id(o): i for i, o in enumerate(self._remote_masks(now))}
                    cands.sort(
                        key=lambda c: (
                            rank.get(id(owner_of.get(c)), 99),
                            round(busy.get(c, 0.0) / 0.05),
                            -c,
                        )
                    )
                else:
                    # least busy (0.05 granularity), then round-robin starting half
                    # the pool away from `cur` (spread over the socket's mesh)
                    n = len(cands) + 1
                    pos = sorted(cands + [cur])
                    p0 = pos.index(cur)
                    cands.sort(
                        key=lambda c: (
                            round(busy.get(c, 0.0) / 0.05),
                            (pos.index(c) - p0 - n // 2) % n,
                        )
                    )
                return tier, cands
        return "", []

    def _remote_masks(self, now: float) -> list[dict]:
        """Fresh peers on another NUMA node whose mask hosts no foreign main
        thread, in preference order: the peer at the same rank among its node's
        workers as this worker among its own (w0 -> first remote, w1 -> second)
        first, so two slow workers of one socket never pick the same mask."""
        if self._remote_cache_t == now:
            return self._remote_cache
        fresh = [
            p for p in self.peers.values() if now - p.get("_seen", 0) < _PEER_FRESH_S
        ]
        mains = {str(p.get("idx")): p.get("cpu", -1) for p in fresh}
        out = []
        for p in fresh:
            if p.get("node", -1) in (-1, self.home_node):
                continue
            home = set(p.get("home", []))
            foreign = [
                i for i, c in mains.items() if i != str(p.get("idx")) and c in home
            ]
            if not foreign:
                out.append(p)
        out.sort(key=lambda p: str(p.get("idx")))
        mine = sorted(
            [str(p.get("idx")) for p in fresh if p.get("node") == self.home_node]
            + [str(self.idx)]
        )
        if out:
            k = mine.index(str(self.idx)) % len(out)
            out = out[k:] + out[:k]
        self._remote_cache_t, self._remote_cache = now, out
        return out

    def _move(self, now: float, ratio: float) -> None:
        cur = _current_cpu()
        if "remote" in self.pool:
            self._exchange_peers(now)  # fresh peer main-thread cores
        tier, cands = self._candidates(now, cur)
        if not cands:
            self.failed += 1
            self.fail_episodes += 1
            backoff = min(600.0, 60.0 * 2 ** (self.fail_episodes - 1))
            self.next_episode_t = now + backoff
            logger.info(
                "%s no candidate core (pool=%s, peers=%d); backing off %.0f s",
                self.tag,
                ",".join(self.pool),
                len(self.peers),
                backoff,
            )
            return
        target = cands[0]
        try:
            os.sched_setaffinity(0, {target})
        except OSError as e:
            self.bad_until[target] = now + _BAD_TTL_S
            logger.info("%s sched_setaffinity(%d) failed: %s", self.tag, target, e)
            self.next_episode_t = now + self.cooldown_s
            return
        self.cur_cpu = target
        self.moves += 1
        self.episode_moves += 1
        self.episode_tiers.append(tier)
        if tier == "own":
            self.episode_own += 1
        owner = ""
        if tier == "remote":
            for p in self.peers.values():
                if target in p.get("home", []):
                    owner = f" owner w{p.get('idx')} mask {_fmt_cpus(p.get('home', []))}"
        self.state = "settle"
        self.settle = []
        self.move_t = now
        self.snap_stat = _proc_stat_busy()  # core load during settle -> next pick
        self.win = []
        self.win_t0 = now
        self._publish(now)
        logger.info(
            "%s MOVE cpu %d -> %d (tier %s%s, node %d -> %d) host/ref %.2f "
            "slow %.1fs, move %d of episode",
            self.tag,
            cur,
            target,
            tier,
            owner,
            self.nodes.get(cur, -1),
            self.nodes.get(target, -1),
            ratio,
            now - (self.slow_since or now),
            self.episode_moves,
        )

    # ------------------------------------------------------------ peers/logs
    def _publish(self, now: float) -> None:
        if not self.peer_dir:
            return
        try:
            os.makedirs(self.peer_dir, exist_ok=True)
            path = os.path.join(self.peer_dir, f"w{self.idx}.json")
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "idx": self.idx,
                        "pid": os.getpid(),
                        "cpu": _current_cpu(),
                        "home": self.home,
                        "node": self.home_node,
                        "ref": {str(b): v for b, v in self.ref_own.items()},
                        "t": time.time(),
                    },
                    f,
                )
            os.replace(tmp, path)
        except OSError:
            pass

    def _exchange_peers(self, now: float) -> None:
        self.next_peer_t = now + _PEER_S
        if not self.peer_dir:
            return
        self._publish(now)
        peers: dict[str, dict] = {}
        ref: dict[int, float] = {}
        wall = time.time()
        for path in glob.glob(os.path.join(self.peer_dir, "w*.json")):
            try:
                with open(path) as f:
                    p = json.load(f)
            except (OSError, ValueError):
                continue
            if p.get("idx") == self.idx or wall - p.get("t", 0) > _PEER_FRESH_S:
                continue
            p["_seen"] = now
            peers[str(p.get("idx"))] = p
            for b, v in p.get("ref", {}).items():
                b = int(b)
                if v < ref.get(b, 1e9):
                    ref[b] = v
        self.peers = peers
        self.ref_peer = ref
        self._remote_cache_t = -1.0

    def _bg_affinity(self) -> None:
        """Helper threads inherit the creator's affinity: keep them off the
        main thread's core (they run on the worker's own mask)."""
        try:
            cur = self.cur_cpu
            os.sched_setaffinity(0, set(self.home) - {cur} or set(self.home))
        except OSError:
            pass

    def _snap_ipi_bg(self) -> None:
        self._bg_affinity()
        self.snap_ipi = _proc_ipis()

    def _fix_inherited_bg(self, main_tid: int, cpu: int) -> None:
        """Threads created by the main thread after a move inherited its
        single-core mask; give them the worker's own mask back."""
        self._bg_affinity()
        fixed = []
        for path in glob.glob("/proc/self/task/*"):
            try:
                tid = int(os.path.basename(path))
                if tid in (main_tid, threading.get_native_id()):
                    continue
                if os.sched_getaffinity(tid) == {cpu}:
                    os.sched_setaffinity(tid, set(self.home))
                    fixed.append(tid)
            except (OSError, ValueError):
                continue
        if fixed:
            logger.info(
                "%s reset %d helper thread(s) that inherited the main thread's "
                "core %d to the worker mask",
                self.tag,
                len(fixed),
                cpu,
            )

    def _log_episode(self, now: float, ratio: float) -> None:
        cur = self.episode_from
        msg = (
            f"{self.tag} SLOW on cpu {cur} (node {self.nodes.get(cur, -1)}): "
            f"host/ref {ratio:.2f} for {now - (self.slow_since or now):.1f}s; "
            f"ref B1 own {self.ref_own.get(1, 0):.2f} peer "
            f"{self.ref_peer.get(1, 0):.2f} ms"
        )
        if self.observe:
            msg += " (observe: no move)"
        logger.info(msg)
        if self.diag and self.snap_ipi and cur >= 0:
            snap, t0 = self.snap_ipi, self.snap_t
            self.snap_ipi = {}
            threading.Thread(
                target=self._diag_bg,
                args=(cur, snap, t0),
                name="main-repin-diag",
                daemon=True,
            ).start()

    def _diag_bg(self, cur: int, snap: dict[str, list[int]], t0: float) -> None:
        """IPI rates on the slow core since slow onset + threads last on it."""
        self._bg_affinity()
        try:
            ipi = _proc_ipis()
            dt = max(time.perf_counter() - t0, 1e-3)
            parts = []
            for name in sorted(ipi):
                a, b = snap.get(name), ipi.get(name)
                if not a or not b or cur >= len(b):
                    continue
                rates = [(y - x) / dt for x, y in zip(a, b)]
                home = sorted(rates[c] for c in self.home if c < len(rates))
                top = max(range(len(rates)), key=rates.__getitem__)
                parts.append(
                    f"{name} {rates[cur]:.0f}/s (home median "
                    f"{home[len(home) // 2] if home else 0:.0f}, node max "
                    f"{rates[top]:.0f}@cpu{top})"
                )
            logger.info(
                "%s diag cpu %d over %.1fs: %s; threads last on cpu: %s",
                self.tag,
                cur,
                dt,
                ", ".join(parts) or "-",
                ",".join(_threads_on_cpu(cur)[:10]) or "-",
            )
        except Exception:  # diagnostics only
            logger.debug("%s diag failed", self.tag, exc_info=True)

    def _status(self, now: float) -> None:
        self.next_status_t = now + self.status_s
        if self.moves:
            threading.Thread(
                target=self._fix_inherited_bg,
                args=(self.main_tid, self.cur_cpu),
                name="main-repin-fix",
                daemon=True,
            ).start()
        refs = " ".join(
            f"B{b}:{self.ref_own[b]:.2f}" for b in sorted(self.ref_own)[:4]
        )
        logger.info(
            "%s status: cpu %d ratio %.2f slow %.0f/%.0f s (%d/%d windows) "
            "moves %d recovered %d failed %d refs %s peers %d",
            self.tag,
            _current_cpu(),
            self.last_ratio,
            self.slow_s,
            now - self.t_start,
            self.slow_windows,
            self.windows,
            self.moves,
            self.recovered,
            self.failed,
            refs or "-",
            len(self.peers),
        )


def _fmt_cpus(cpus: list[int]) -> str:
    if not cpus:
        return "-"
    out, lo, prev = [], cpus[0], cpus[0]
    for c in cpus[1:] + [None]:
        if c is not None and c == prev + 1:
            prev = c
            continue
        out.append(f"{lo}-{prev}" if prev > lo else str(lo))
        if c is not None:
            lo = prev = c
    return ",".join(out)


def maybe_create() -> MainThreadRepin | None:
    """Create the watchdog on the busy-loop thread if VLLM_ENGINE_MAIN_REPIN is set."""
    mode = os.environ.get("VLLM_ENGINE_MAIN_REPIN", "")
    if mode.strip().lower() in ("", "0", "false", "off"):
        return None
    try:
        return MainThreadRepin(mode)
    except Exception:  # never let the watchdog stop the engine
        logger.exception("[main-repin] init failed; watchdog disabled")
        return None
