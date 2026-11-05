# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step counters of the die-local KV decode split (log only).

``VLLM_DIE_LOCAL_KV_STATS=N`` logs a summary every N engine steps; unset or 0
turns every hook into an immediate return.

Every decode-attention site the die-local split covers calls ``note`` each time
its Python runs, with its rows, query length per request and either "split"
(one launch per die) or the reason the launch stayed whole-GPU. The sites, in
``FlashInferImpl``, are the main trtllm-gen decode site (MTP verify and every
draft decode pass) and the two draft-prefill row sites.

Under a FULL CUDA graph that Python runs only while the graph is captured, so
``capture_begin`` / ``bind`` file a capture's notes under its graph and
``replayed`` counts them on every replay. Notes made outside a capture are eager
launches (PIECEWISE and NONE steps).

A step runs from one ``step_begin`` (``GPUModelRunner.execute_model``, with the
target batch's CUDA-graph mode and capture size) to the next, so it includes the
drafts that step proposes. Nothing here touches a tensor or a CUDA stream.
"""

import os
from collections import Counter
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_DIE_LOCAL_KV_STATS"


def _every() -> int:
    raw = os.environ.get(ENV, "0") or "0"
    try:
        return max(int(raw), 0)
    except ValueError:
        logger.warning("%s=%r is not an integer: die-local KV stats off", ENV, raw)
        return 0


EVERY = _every()
ON = EVERY > 0

# A capture-time note: (part, site, q_len, state).
# - part: "t" (target layer) or "d" (MTP draft layer).
# - site: "dec" (main decode site), "dpr" / "dpl" (draft-prefill rows / last
#   rows).
# - state: "split", or why not (see ``why``).
Note = tuple[str, str, int, str]
# A launch: (src, *Note); src is "graph" (FULL replay), "eager", or "cap?" (made
# inside a capture that no hook opened).
Launch = tuple[str, str, str, int, str]


def part_of(layer_name: str) -> str:
    """Return "d" for the MTP draft model's attention layers, else "t"."""
    return "d" if layer_name.startswith("mtp.") or ".mtp." in layer_name else "t"


def why(split: bool, active: bool, rows_ok: bool, eligible: bool, raised: bool) -> str:
    """State of one decode launch, in the order the split gate checks it.

    Args:
        split: The launch runs once per die.
        active: Die-local KV is mapped in this worker.
        rows_ok: At least VLLM_DIE_LOCAL_KV_MIN_ROWS rows.
        eligible: Not excluded (LSE, KF decode, DCP, NVFP4 KV, mock cache).
        raised: The vLLM split-KV policy raised the KV split.
    """
    if split:
        return "split"
    if not active:
        return "off"
    if not rows_ok:
        return "rows"
    if not eligible:
        return "excluded"
    if not raised:
        return "unraised"
    return "unknown"


@dataclass
class _Step:
    dummy: bool
    mode: str  # target CUDA-graph mode: FULL / PIECEWISE / NONE
    reqs: int
    tokens: int
    q: int  # uniform tokens per request (1 + k); 0 when not uniform
    graph_tokens: int  # padded tokens (FULL: the graph's capture size)
    graph_reqs: int  # padded requests (0: no request padding)
    launches: dict[Launch, list[int]] = field(default_factory=dict)  # [n, rows]
    replays: int = 0


@dataclass
class _Agg:
    steps: int = 0
    dummy: Counter = field(default_factory=Counter)
    mode: Counter = field(default_factory=Counter)
    q: Counter = field(default_factory=Counter)
    sizes: Counter = field(default_factory=Counter)
    pad: int = 0
    reqs: int = 0
    reqs_min: int = 1 << 30
    reqs_max: int = 0
    verify: Counter = field(default_factory=Counter)
    draft: Counter = field(default_factory=Counter)
    launches: dict[Launch, list[int]] = field(default_factory=dict)
    replays: int = 0


_capturing = False
_cap: dict[Note, list[int]] = {}
# id(graph) -> the notes made while it was captured: (note, launches, rows).
_records: dict[int, tuple[tuple[Note, int, int], ...]] = {}
_step: _Step | None = None
_agg = _Agg()
_done = 0  # steps already logged
_split_total = {"verify": 0, "draft": 0}


def _add(table: dict, key: tuple, n: int, rows: int) -> None:
    v = table.get(key)
    if v is None:
        table[key] = [n, rows]
    else:
        v[0] += n
        v[1] += rows


def note(part: str, site: str, rows: int, q_len: int, state: str) -> None:
    """Record one decode launch decision where its site's Python runs."""
    if not ON:
        return
    if _capturing:
        _add(_cap, (part, site, q_len, state), 1, rows)
        return
    if _step is None:
        return  # before the first step: warmup and profiling
    src = "cap?" if torch.cuda.is_current_stream_capturing() else "eager"
    _add(_step.launches, (src, part, site, q_len, state), 1, rows)


def capture_begin() -> None:
    """Call right before a FULL CUDA graph capture."""
    global _capturing
    if not ON:
        return
    _cap.clear()
    _capturing = True


def bind(graph: object) -> None:
    """Call right after a FULL capture: file its notes under ``graph``."""
    global _capturing
    if not ON:
        return
    _capturing = False
    _records[id(graph)] = tuple((k, v[0], v[1]) for k, v in _cap.items())
    _cap.clear()


def replayed(graph: object) -> None:
    """Call on every replay of a FULL CUDA graph."""
    if not ON or _step is None:
        return
    _step.replays += 1
    rec = _records.get(id(graph))
    if rec is None:
        _add(_step.launches, ("graph", "?", "?", 0, "unbound"), 1, 0)
        return
    for (part, site, q_len, state), n, rows in rec:
        _add(_step.launches, ("graph", part, site, q_len, state), n, rows)


def step_begin(
    dummy: bool,
    mode: str,
    reqs: int,
    tokens: int,
    q: int | None,
    graph_tokens: int,
    graph_reqs: int | None,
) -> None:
    """Close the previous step (its target forward and drafts); open this one.

    Called once per execute_model, after the CUDA-graph dispatch.
    """
    global _step
    if not ON:
        return
    if _step is None:
        logger.info(
            "die-local KV stats: on, a summary every %d engine steps (%s)",
            EVERY,
            ENV,
        )
    else:
        _fold(_step)
    _step = _Step(dummy, mode, reqs, tokens, q or 0, graph_tokens, graph_reqs or 0)


def _verdict(launches: dict[Launch, list[int]], part: str, site: str | None) -> str:
    """Return "<src> <state>" for one step's launches of a part (and site)."""
    srcs: set[str] = set()
    states: Counter = Counter()
    for (src, p, s, _q, state), (n, _rows) in launches.items():
        if p == part and (site is None or s == site):
            srcs.add(src)
            states[state] += n
    if not states:
        return "none"
    src = "+".join(sorted(srcs))
    if set(states) == {"split"}:
        return f"{src} split"
    if "split" in states:
        return f"{src} partial"
    return f"{src} no-{states.most_common(1)[0][0]}"


def _fold(step: _Step) -> None:
    a = _agg
    if step.dummy:
        a.dummy[step.mode] += 1
        return
    a.steps += 1
    a.mode[step.mode] += 1
    a.q[step.q] += 1
    a.reqs += step.reqs
    a.reqs_min = min(a.reqs_min, step.reqs)
    a.reqs_max = max(a.reqs_max, step.reqs)
    if step.mode == "FULL":
        a.sizes[(step.graph_tokens, step.graph_reqs)] += 1
        a.pad += step.graph_tokens - step.tokens
    verify = _verdict(step.launches, "t", "dec")
    draft = _verdict(step.launches, "d", None)
    a.verify[verify] += 1
    a.draft[draft] += 1
    _split_total["verify"] += verify.endswith(" split")
    _split_total["draft"] += draft.endswith(" split")
    for key, (n, rows) in step.launches.items():
        _add(a.launches, key, n, rows)
    a.replays += step.replays
    if a.steps >= EVERY:
        _log()


def _log() -> None:
    global _agg, _done
    a = _agg
    first, last = _done + 1, _done + a.steps
    _done = last
    sizes = " ".join(f"{t}/{r}:{n}" for (t, r), n in a.sizes.most_common(8))
    logger.info(
        "die-local KV stats [steps] %d-%d: target mode %s, dummy %s; q (1+k, 0 = "
        "not uniform) %s; requests mean %.1f min %d max %d; FULL graphs "
        "(tokens/requests:steps) %s (%d sizes), padding %.1f tokens per FULL step",
        first,
        last,
        dict(a.mode),
        dict(a.dummy),
        dict(sorted(a.q.items())),
        a.reqs / max(a.steps, 1),
        a.reqs_min if a.steps else 0,
        a.reqs_max,
        sizes,
        len(a.sizes),
        a.pad / max(a.mode["FULL"], 1),
    )
    logger.info(
        "die-local KV stats [split] %d-%d: verify %s; draft %s; since start: "
        "verify split in %d of %d steps, draft split in %d of %d",
        first,
        last,
        dict(a.verify.most_common()),
        dict(a.draft.most_common()),
        _split_total["verify"],
        last,
        _split_total["draft"],
        last,
    )
    launches = ", ".join(
        f"{src} {part} {site} q{q} {state}: {n}/{rows / max(n, 1):.1f}"
        for (src, part, site, q, state), (n, rows) in sorted(
            a.launches.items(), key=lambda kv: -kv[1][0]
        )
    )
    logger.info(
        "die-local KV stats [launches] %d-%d (src part site q state: "
        "launches/mean rows): %s; FULL graph replays %d",
        first,
        last,
        launches,
        a.replays,
    )
    _agg = _Agg()
