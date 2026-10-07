# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned, fingerprinted FlashInfer autotune tables.

FlashInfer's AutoTuner stores one winning tactic per file key
``(op, runner class name, bucketed shapes, extras)``. The key says nothing about
what the tactic means, so a cached or pinned entry silently goes stale when:

- the runner's tactic list changes: FlashInfer version or candidate lists, SM
  count (``can_implement``), vLLM tactic wrappers (``VLLM_FLASHINFER_MXFP8_K64``
  and its tactic list / min M, extra split-K or cluster filters), or a K=64
  tactic on an image without the sm_107 kernel (it falls back to the default
  tactic at run time);
- the kernel behind an unchanged tactic changes: FlashInfer kernel sources,
  the ``vllm/lcd_pdl`` overlay and its ``PFW*`` knobs, the sm_107 backport, a
  prebuilt MoE module (``GS2_ROUTE_PREBUILT``) whose config indices mean
  different kernels;
- the runner instance differs under the same shapes (e.g. MoE top-k,
  intermediate size: not part of FlashInfer's key);
- the GPU differs (name, SM count, compute capability).

With ``VLLM_FLASHINFER_AUTOTUNE_FILE`` set, :func:`run_pinned_autotune` replaces
FlashInfer's own cache load/save in the kernel warmup:

1. The pinned file and the per-config cache file are parsed into one index of
   candidates per file key (pinned first). Nothing is handed to FlashInfer
   unchecked.
2. During the warmup's tuning passes ``AutoTuner.search_cache`` is wrapped.
   When FlashInfer finds no entry, the wrapper validates the indexed
   candidates for that key against this process: the op fingerprint (group
   id), the runner attributes, and the tactic list at the key's bucket shape
   (``get_valid_tactics`` on synthesized inputs, after FlashInfer's
   blocklist), then the tactic itself, then runs it once at that shape (a
   failure rejects it; JIT kernels compile here, before CUDA-graph capture,
   as profiling would have compiled them). The first valid candidate is
   installed in ``AutoTuner._file_configs``; invalid ones are counted and
   logged as stale. A key without a valid candidate is a miss: it is tuned
   (or, with ``VLLM_FLASHINFER_AUTOTUNE_STRICT=1``, runs the fallback tactic
   and start-up fails after the warmup with the list of misses). The wrappers
   are removed after the passes: serving has no extra Python on its path, and
   an entry that was not validated is never in FlashInfer's table.
3. One writer: the world leader holds an exclusive ``flock`` on
   ``<cache file>.lock`` from reading the cache file to writing it, so DP
   workers that share a cache dir (separate engine processes) run their
   autotune pass one after another. The first tunes the misses and atomically
   replaces the cache file; the others load those entries and tune nothing, so
   every worker ends with the same table. Nothing is written when nothing was
   tuned.
4. Each worker logs hits / stale / misses / unused entries and a digest of its
   effective table (identical digests = identical tables).

File format (FlashInfer-compatible; FlashInfer's own ``load_configs`` reads the
top-level entries and ignores the rest)::

    {"_metadata": {FlashInfer metadata},
     "_records": {"vllm_autotune": {
         "schema": 2,
         "host": {flashinfer_version, flashinfer_git, gpu, sm_count, cc, cuda...},
         "groups": {gid: {"op", "runner", "fp", "parts"}},
         "alt": {file_key: [[runner, tactic, meta], ...]},   # other variants
         "provenance": {...}}},
     "<file_key>": [runner, tactic, {"g": gid, "tl": tactic-list hash,
                                     "nt": n tactics, "ra": runner attrs hash,
                                     "us": best us, "us2": runner-up us, ...}],
     "_generation": sha256}

A group id is ``op|runner class|fingerprint`` where the fingerprint hashes the
runner's source modules, the op's kernel source modules (as loaded, so
overlays count), binaries named by envs, and the op's envs. One file can hold
several variants of an op (e.g. ``PFW`` on and off); a process uses the one
whose group id matches. Files without the ``vllm_autotune`` record (FlashInfer
format, e.g. a #148 pin) are read as legacy: their ``_metadata`` must match
the runtime and each tactic must be offered by the runner at that shape.

Generation (``VLLM_FLASHINFER_AUTOTUNE_RECORD=<dir>``): every key is profiled
(pinned file and cache ignored); each tactic is timed as
``VLLM_FLASHINFER_AUTOTUNE_RECORD_ROUNDS`` CUDA-graph replays and the median is
used; for ``mxfp8_gemm`` only the weights (B and its scales) rotate through
copies larger than L2 while the activation and output stay resident (decode
serving reads activations just produced by the previous kernel and weights
from DRAM); other ops keep FlashInfer's cold-L2 inputs. Each worker writes its
samples; ``python -m vllm.model_executor.warmup.flashinfer_autotune_pin merge``
picks per key the tactic with the lowest median-over-workers of the per-worker
medians and writes the pinned file.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import enum
import errno
import glob
import hashlib
import importlib.util
import inspect
import json
import math
import os
import socket
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

NAMESPACE = "vllm_autotune"
SCHEMA = 2
RECORD_KIND = "vllm_autotune_record"
_FI_META = "_metadata"
_FI_RECORDS = "_records"
_FI_GENERATION = "_generation"

# FlashInfer's own metadata keys (``autotuner._collect_metadata``).
FI_META_KEYS = (
    "flashinfer_version",
    "cuda_version",
    "cublas_version",
    "cudnn_version",
    "cudnn_frontend_version",
    "gpu",
)
# A table whose host differs on any of these is rejected as a whole.
HOST_KEYS = FI_META_KEYS + ("flashinfer_git", "sm_count", "cc")

# Warnings with per-key details; the rest go to DEBUG (counts are always logged).
_MAX_KEY_WARNINGS = 12
# The PinSession of the last run_pinned_autotune() in this process (diagnostics).
LAST_SESSION: PinSession | None = None


# --------------------------------------------------------------------------
# Pure helpers (no torch / FlashInfer): file format, hashing, merge.
# --------------------------------------------------------------------------


def tactic_to_json(tactic: Any) -> Any:
    """JSON form of a tactic (tuples and foreign iterables -> lists)."""
    if isinstance(tactic, (tuple, list)):
        return [tactic_to_json(v) for v in tactic]
    if hasattr(tactic, "__iter__") and not isinstance(tactic, (str, bytes, dict)):
        return [tactic_to_json(v) for v in tactic]
    if isinstance(tactic, bool):
        return tactic
    if isinstance(tactic, int):
        return int(tactic)
    return tactic


def json_to_tactic(value: Any) -> Any:
    """Inverse of :func:`tactic_to_json` (lists -> tuples), as FlashInfer."""
    if isinstance(value, list):
        return tuple(json_to_tactic(v) for v in value)
    return value


def stable_hash(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def tactic_list_hash(tactics: Sequence[Any]) -> str:
    return stable_hash([tactic_to_json(t) for t in tactics])


def make_gid(op: str, runner: str, fp: str) -> str:
    return f"{op}|{runner}|{fp}"


def file_sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclasses.dataclass
class Candidate:
    runner: str
    tactic: Any  # JSON form
    meta: dict[str, Any]
    source: str  # "pinned" | "cache"

    @property
    def gid(self) -> str | None:
        return self.meta.get("g")


@dataclasses.dataclass
class Table:
    path: str
    source: str
    sha256: str
    fi_meta: dict[str, Any] | None
    host: dict[str, Any] | None  # None: legacy FlashInfer-format file
    groups: dict[str, dict[str, Any]]
    entries: dict[str, list[Candidate]]
    provenance: dict[str, Any]

    @property
    def legacy(self) -> bool:
        return self.host is None

    def num_entries(self) -> int:
        return sum(len(v) for v in self.entries.values())


def parse_table(
    obj: Any, *, path: str = "", source: str = "pinned", sha256: str = ""
) -> Table:
    """Parse a pinned / cache table (schema 2 or legacy FlashInfer format)."""
    if not isinstance(obj, dict):
        raise ValueError(f"{path or 'autotune table'}: not a JSON object")
    obj = dict(obj)
    fi_meta = obj.pop(_FI_META, None)
    obj.pop(_FI_GENERATION, None)
    records = obj.pop(_FI_RECORDS, None) or {}
    rec = records.get(NAMESPACE) if isinstance(records, dict) else None
    host: dict[str, Any] | None = None
    groups: dict[str, dict[str, Any]] = {}
    alt: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    if rec is not None:
        if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
            schema = rec.get("schema") if isinstance(rec, dict) else rec
            raise ValueError(
                f"{path or 'autotune table'}: unsupported {NAMESPACE} schema "
                f"{schema!r} (expected {SCHEMA})"
            )
        host = dict(rec.get("host") or {})
        groups = dict(rec.get("groups") or {})
        alt = dict(rec.get("alt") or {})
        provenance = dict(rec.get("provenance") or {})
    entries: dict[str, list[Candidate]] = {}

    def add(file_key: str, value: Any) -> None:
        if not (
            isinstance(value, list) and len(value) >= 2 and isinstance(value[0], str)
        ):
            raise ValueError(f"{path or 'autotune table'}: bad entry {file_key!r}")
        meta = value[2] if len(value) > 2 and isinstance(value[2], dict) else {}
        entries.setdefault(file_key, []).append(
            Candidate(value[0], value[1], dict(meta), source)
        )

    for file_key, value in obj.items():
        if not file_key.startswith("_"):
            add(file_key, value)
    for file_key, values in alt.items():
        for value in values:
            add(file_key, value)
    return Table(
        path=path,
        source=source,
        sha256=sha256,
        fi_meta=fi_meta if isinstance(fi_meta, dict) else None,
        host=host,
        groups=groups,
        entries=entries,
        provenance=provenance,
    )


def read_table(path: str | os.PathLike, source: str = "pinned") -> Table:
    data = Path(path).read_bytes()
    return parse_table(
        json.loads(data),
        path=str(path),
        source=source,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def build_table_json(
    *,
    fi_meta: dict[str, Any],
    host: dict[str, Any],
    groups: dict[str, dict[str, Any]],
    entries: dict[str, list[list[Any]]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Serialize a table. ``entries[file_key]`` lists ``[runner, tactic, meta]``
    candidates; the first is the top-level (FlashInfer-visible) one.
    """
    ordered: dict[str, Any] = {_FI_META: dict(fi_meta)}
    rec: dict[str, Any] = {
        "schema": SCHEMA,
        "host": dict(host),
        "groups": {g: groups[g] for g in sorted(groups)},
        "provenance": provenance,
    }
    alt = {k: v[1:] for k, v in sorted(entries.items()) if len(v) > 1}
    if alt:
        rec["alt"] = alt
    ordered[_FI_RECORDS] = {NAMESPACE: rec}
    for file_key in sorted(entries):
        ordered[file_key] = entries[file_key][0]
    source = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    ordered[_FI_GENERATION] = hashlib.sha256(source.encode()).hexdigest()
    return ordered


def write_json_atomic(path: str | os.PathLike, obj: Any) -> str:
    """Write ``obj`` as JSON via a temp file + ``os.replace``; returns sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1) + "\n").encode()
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return hashlib.sha256(data).hexdigest()


def host_mismatches(table: Table, host: dict[str, str]) -> dict[str, tuple[Any, Any]]:
    """``key -> (table, runtime)`` for the host fields that differ."""
    keys: tuple[str, ...]
    if table.host is not None:
        saved, keys = table.host, HOST_KEYS
    else:
        saved, keys = table.fi_meta or {}, FI_META_KEYS
    out = {}
    for k in keys:
        if str(saved.get(k)) not in (str(host.get(k)), "*"):
            out[k] = (saved.get(k), host.get(k))
    return out


def diff_parts(old: dict[str, Any] | None, new: dict[str, Any]) -> str:
    """Short description of fingerprint parts that differ."""
    if old is None:
        return "unknown group"
    out = []

    def walk(a: Any, b: Any, prefix: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k), b.get(k), f"{prefix}{k}.")
        elif a != b:
            out.append(f"{prefix[:-1]}: {a} -> {b}")

    walk(old, new, "")
    return "; ".join(out[:6]) + ("; ..." if len(out) > 6 else "")


@contextlib.contextmanager
def interprocess_lock(path: Path) -> Iterator[tuple[float, bool]]:
    """Exclusive ``flock`` on ``path``; yields (seconds waited, locked)."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    locked = False
    t0 = time.monotonic()
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
        except OSError as e:
            if e.errno not in (
                errno.ENOLCK,
                errno.EOPNOTSUPP,
                errno.ENOSYS,
                errno.EINVAL,
            ):
                raise
            logger.warning(
                "FlashInfer autotune: cannot lock %s (%s); DP workers sharing "
                "this cache dir may tune uncovered keys independently",
                path,
                e,
            )
        yield time.monotonic() - t0, locked
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# --------------------------------------------------------------------------
# Runtime fingerprints (torch / FlashInfer imported lazily).
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _OpRule:
    # Modules whose source identifies the kernels behind the op's tactics.
    kernel_modules: tuple[str, ...] = ()
    # Env-name prefixes that select or modify those kernels / tactic lists.
    env_prefixes: tuple[str, ...] = ()
    # Envs naming a binary whose content identifies the kernels.
    binary_envs: tuple[str, ...] = ()


_MXFP8_RULE = _OpRule(
    kernel_modules=(
        "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100",
        "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_common",
        "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_splitk",
        "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm107",
        "flashinfer.gemm.kernels.utils",
    ),
    env_prefixes=("VLLM_FLASHINFER_MXFP8_", "VLLM_MXFP8_", "LCD_FI_", "PFW"),
)
_MOE_RULE = _OpRule(
    env_prefixes=("GS2_", "VLLM_FI_SM107_MOE_"),
    binary_envs=("GS2_ROUTE_PREBUILT",),
)


def _op_rule(op: str) -> _OpRule:
    if op == "mxfp8_gemm":
        return _MXFP8_RULE
    if "moe" in op:
        return _MOE_RULE
    return _OpRule()


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def _cached_file_sha(path: str) -> str:
    try:
        st = os.stat(path)
    except OSError:
        return "absent"
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = file_sha256(path)[:16]
    return _SHA_CACHE[key]


def _module_source_sha(name: str) -> str:
    """Sha of the source this process uses (or would use) for ``name``."""
    mod = sys.modules.get(name)
    origin = getattr(mod, "__file__", None) if mod is not None else None
    if origin is None:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError):
            spec = None
        origin = spec.origin if spec is not None and spec.has_location else None
    if not origin:
        return "absent"
    return _cached_file_sha(origin)


def _env_parts(prefixes: tuple[str, ...]) -> dict[str, str]:
    if not prefixes:
        return {}
    import vllm.envs as envs

    out = {}
    # Registered vLLM envs by effective value (unset == default).
    for name in envs.environment_variables:
        if name.startswith(prefixes):
            out[name] = str(getattr(envs, name))
    for name, value in os.environ.items():
        if name.startswith(prefixes) and name not in out:
            out[name] = value
    return dict(sorted(out.items()))


def _runner_source_parts(cls: type) -> dict[str, str]:
    out = {}
    for klass in cls.__mro__:
        if klass.__module__ in ("builtins", "abc") or klass.__name__ == "TunableRunner":
            continue
        try:
            path = inspect.getsourcefile(klass)
        except TypeError:
            path = None
        out[f"{klass.__module__}.{klass.__qualname__}"] = (
            _cached_file_sha(path) if path else "absent"
        )
    return out


def op_fingerprint_parts(op: str, runner: Any) -> dict[str, Any]:
    """Everything (besides host and per-entry tactic list) that decides what
    a tactic of ``op`` on ``runner`` runs.
    """
    rule = _op_rule(op)
    parts: dict[str, Any] = {
        "runner_src": _runner_source_parts(type(runner)),
        "kernels": {m: _module_source_sha(m) for m in rule.kernel_modules},
        "env": _env_parts(rule.env_prefixes),
        "binaries": {
            e: _cached_file_sha(os.environ[e])
            for e in rule.binary_envs
            if os.environ.get(e)
        },
    }
    if op == "mxfp8_gemm":
        tb = sys.modules.get(
            "vllm.model_executor.kernels.linear.mxfp8.flashinfer_tune_buckets"
        )
        installed = tb.installed_buckets() if tb is not None else None
        parts["tune_buckets"] = list(installed or ())
    return parts


def runner_attrs_hash(runner: Any) -> str:
    """Hash of the runner's plain configuration attributes (e.g. MoE top-k,
    intermediate size), which FlashInfer's file key does not contain.
    """
    vals: dict[str, Any] = {}
    for k, v in sorted(vars(runner).items()):
        if k.startswith("_") or k.endswith("_cache"):
            continue
        if isinstance(v, enum.Enum):
            vals[k] = f"{type(v).__name__}.{v.name}"
        elif isinstance(v, (bool, int, float, str, type(None))):
            vals[k] = v
        elif isinstance(v, tuple) and all(
            isinstance(x, (bool, int, float, str, type(None))) for x in v
        ):
            vals[k] = list(v)
    return stable_hash(vals)


def runtime_host() -> dict[str, str]:
    import torch
    from flashinfer.autotuner import _collect_metadata

    host = {k: str(v) for k, v in _collect_metadata().items()}
    try:
        from flashinfer._build_meta import __git_commit__ as git
    except Exception:
        git = "unknown"
    props = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    host.update(
        flashinfer_git=str(git),
        sm_count=str(props.multi_processor_count),
        cc=f"{props.major}.{props.minor}",
    )
    return host


def _gpu_uuid() -> str:
    import torch

    try:
        return str(
            torch.cuda.get_device_properties(
                torch.accelerator.current_device_index()
            ).uuid
        )
    except Exception:
        return "unknown"


class _Groups:
    """Memoized op fingerprints per (op, runner class)."""

    def __init__(self) -> None:
        self.groups: dict[str, dict[str, Any]] = {}
        self._by_cls: dict[tuple[str, type], str] = {}

    def gid(self, op: str, runner: Any) -> str:
        k = (op, type(runner))
        gid = self._by_cls.get(k)
        if gid is None:
            parts = op_fingerprint_parts(op, runner)
            fp = stable_hash(parts)
            gid = make_gid(op, type(runner).__name__, fp)
            self.groups[gid] = {
                "op": op,
                "runner": type(runner).__name__,
                "fp": fp,
                "parts": parts,
            }
            self._by_cls[k] = gid
        return gid


def _profile_for_key(tuning_config: Any, inputs: list[Any], nearest: Any) -> Any:
    """FlashInfer optimization profile at a cache key's bucket shapes."""
    import torch
    from flashinfer.autotuner import DynamicDim, OptimizationProfile, StaticDim

    shapes = [
        [StaticDim(x) for x in t.size()]
        if isinstance(t, torch.Tensor)
        else [StaticDim(0)]
        for t in inputs
    ]
    p = OptimizationProfile(shapes, [None] * len(inputs))
    for idx, init in tuning_config.tensor_initializers:
        p.tensor_initializers[idx] = init
    for spec in tuning_config.dynamic_tensor_specs:
        v = nearest[spec.input_idx[0]][spec.dim_idx[0]]
        for i, d in zip(spec.input_idx, spec.dim_idx, strict=True):
            p.shapes[i][d] = DynamicDim(v, v, v)
    for c in tuning_config.constraint_specs:
        v = c.infer_shape(p.get_opt_shapes())
        p.shapes[c.input_idx][c.dim_idx] = DynamicDim(v, v, v)
    return p


def key_tactics(
    tuner: Any, op: str, runner: Any, key: Any, tuning_config: Any, inputs: list[Any]
) -> tuple[list[Any], list[Any]]:
    """The runner's tactic list (JSON form) at the key's bucket, as FlashInfer's
    tuning loop would profile it (synthesized inputs, pre-hook, blocklist), and
    the synthesized inputs.
    """
    p = _profile_for_key(tuning_config, inputs, key.nearest_profile)
    tensors = tuner._prepare_input_tensors(p, inputs)
    if tuning_config.inputs_pre_hook is not None:
        tensors = list(tuning_config.inputs_pre_hook(tensors))
    valid = list(runner.get_valid_tactics(tensors, p))
    valid = tuner._blocklist.filter(op, runner, valid)
    return [tactic_to_json(t) for t in valid], tensors


# --------------------------------------------------------------------------
# Pinned load / validation session.
# --------------------------------------------------------------------------


def _reason_class(reason: str) -> str:
    return reason.split(":", 1)[0].split(" (", 1)[0]


class PinSession:
    """Validates pinned candidates lazily from ``AutoTuner.search_cache``
    while installed (the kernel warmup's tuning passes).
    """

    def __init__(
        self,
        tables: Sequence[Table],
        *,
        strict: bool = False,
        host: dict[str, str] | None = None,
    ) -> None:
        self.strict = strict
        self.host = runtime_host() if host is None else host
        self.tables = list(tables)
        self.stats: Counter[str] = Counter()
        self.stale_reasons: Counter[str] = Counter()
        self.table_groups: dict[str, dict[str, Any]] = {}
        self.index: dict[str, list[Candidate]] = {}
        self.rejected_tables: list[tuple[Table, dict[str, tuple[Any, Any]]]] = []
        for t in self.tables:
            mism = host_mismatches(t, self.host)
            if mism:
                self.rejected_tables.append((t, mism))
                self.stats["stale_host"] += t.num_entries()
                logger.warning(
                    "FlashInfer autotune %s table %s (sha256 %s) was made on a "
                    "different host/build and is NOT used (%d entries): %s",
                    t.source,
                    t.path,
                    t.sha256[:12],
                    t.num_entries(),
                    "; ".join(f"{k}: {a} != {b}" for k, (a, b) in mism.items()),
                )
                continue
            for g, info in t.groups.items():
                self.table_groups.setdefault(g, info)
            for fk, cands in t.entries.items():
                self.index.setdefault(fk, []).extend(cands)
        self.groups = _Groups()
        # file_key -> "hit" | "miss" | "strict"
        self.decided: dict[str, str] = {}
        # file_key -> [runner, tactic(json), meta] of accepted candidates.
        self.accepted: dict[str, list[Any]] = {}
        # file_key -> meta of keys with no valid candidate (tuned here).
        self.misses: dict[str, dict[str, Any]] = {}
        self._warned = 0
        self._tuner: Any = None

    # -- hooks (installed for the tuning passes only) -------------------------
    def install(self, tuner: Any) -> None:
        """Wrap ``tuner.search_cache`` (validation) and ``tuner.choose_one``
        (to see the runner kwargs for the warm-up call).
        """
        cls = type(tuner)
        orig_search = cls.search_cache.__get__(tuner)
        orig_choose = cls.choose_one.__get__(tuner)
        session = self
        local = threading.local()

        def search_cache(custom_op, runners, input_shapes, tuning_config, inputs=None):
            res = orig_search(custom_op, runners, input_shapes, tuning_config, inputs)
            if res[0]:
                return res
            return session.lookup(
                tuner,
                custom_op,
                runners,
                input_shapes,
                tuning_config,
                inputs,
                res,
                getattr(local, "kwargs", None) or {},
            )

        def choose_one(custom_op, runners, tuning_config, inputs, **kwargs):
            prev = getattr(local, "kwargs", None)
            local.kwargs = kwargs
            try:
                return orig_choose(custom_op, runners, tuning_config, inputs, **kwargs)
            finally:
                local.kwargs = prev

        tuner.search_cache = search_cache
        tuner.choose_one = choose_one
        self._tuner = tuner

    def uninstall(self) -> None:
        """Restore FlashInfer's methods. Only accepted entries are in
        ``AutoTuner._file_configs``; a key first seen later misses as usual.
        """
        if self._tuner is not None:
            for name in ("search_cache", "choose_one"):
                with contextlib.suppress(AttributeError):
                    delattr(self._tuner, name)
            self._tuner = None

    def lookup(
        self,
        tuner: Any,
        op: str,
        runners: list[Any],
        input_shapes: Any,
        tuning_config: Any,
        inputs: list[Any] | None,
        res: tuple[bool, int, Any, Any],
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[bool, int, Any, Any]:
        import torch
        from flashinfer.autotuner import AutoTuner

        new_misses: list[str] = []
        for r_id, r in enumerate(runners):
            extras = r.get_cache_key_extras(inputs) if inputs is not None else ()
            key = AutoTuner._get_cache_key(op, r, input_shapes, tuning_config, extras)
            fk = key.file_key
            state = self.decided.get(fk)
            if state == "strict":
                return (True, 0, -1, None)
            if state is not None:
                continue
            runner_name = type(r).__name__
            cands = [c for c in self.index.get(fk, ()) if c.runner == runner_name]
            if not tuner.is_tuning_mode:
                # Outside the tuning passes FlashInfer would not tune either.
                self.decided[fk] = "miss"
                self.stats["untuned_lookup"] += 1
                continue
            if torch.cuda.is_current_stream_capturing():
                # Synthesizing inputs would enter the captured graph.
                if cands:
                    self._reject(fk, op, ["unvalidated: lookup during graph capture"])
                self.decided[fk] = "miss"
                continue
            info, tensors = self._key_info(tuner, op, r, key, tuning_config, inputs)
            reasons: list[str] = []
            for c in cands:
                why = self._check(c, info)
                if why is None:
                    tactic = json_to_tactic(c.tactic)
                    why = self._warm_up(r, tactic, tensors, kwargs or {})
                if why is None:
                    tuner._file_configs[fk] = (c.runner, tactic)
                    meta = dict(c.meta)
                    if c.gid is None:  # legacy: adopt this process's identity
                        meta.update(g=info["g"], tl=info["tl"], nt=info["nt"])
                        meta["ra"] = info["ra"]
                    meta["src"] = meta.get("src", c.source)
                    self.accepted[fk] = [c.runner, tactic_to_json(c.tactic), meta]
                    self.decided[fk] = "hit"
                    self.stats[
                        f"hit_{c.source}{'_legacy' if c.gid is None else ''}"
                    ] += 1
                    if reasons:
                        self.stats["variant_skipped"] += len(reasons)
                    return (True, r_id, tactic, None)
                reasons.append(why)
            del tensors
            if reasons:
                self._reject(fk, op, reasons)
            self.decided[fk] = "miss"
            self.misses[fk] = {
                "op": op,
                "runner": runner_name,
                "g": info["g"],
                "tl": info["tl"],
                "nt": info["nt"],
                "ra": info["ra"],
            }
            new_misses.append(fk)
        if new_misses:
            self.stats["miss"] += len(new_misses)
            if self._warned < _MAX_KEY_WARNINGS:
                self._warned += 1
                logger.warning(
                    "FlashInfer autotune: no valid pinned entry for %s; %s",
                    new_misses[0],
                    "running the fallback tactic (strict)"
                    if self.strict
                    else "tuning it at start-up",
                )
            if self.strict:
                for fk in new_misses:
                    self.decided[fk] = "strict"
                return (True, 0, -1, None)
        return res

    def _key_info(
        self,
        tuner: Any,
        op: str,
        runner: Any,
        key: Any,
        tuning_config: Any,
        inputs: list[Any] | None,
    ) -> tuple[dict[str, Any], list[Any] | None]:
        gid = self.groups.gid(op, runner)
        info: dict[str, Any] = {
            "g": gid,
            "ra": runner_attrs_hash(runner),
            "tl": None,
            "nt": None,
            "tactics": None,
        }
        if inputs is None:
            info["error"] = "no inputs"
            return info, None
        try:
            tactics, tensors = key_tactics(
                tuner, op, runner, key, tuning_config, inputs
            )
        except Exception as e:  # validation must not break start-up
            info["error"] = f"{type(e).__name__}: {e}"
            return info, None
        info.update(tl=tactic_list_hash(tactics), nt=len(tactics), tactics=tactics)
        return info, tensors

    @staticmethod
    def _warm_up(
        runner: Any, tactic: Any, tensors: list[Any] | None, kwargs: dict[str, Any]
    ) -> str | None:
        """Run the accepted tactic once at the key's shape before CUDA-graph
        capture, as FlashInfer's profiling would have: JIT kernels (CuTe DSL,
        K=64) compile now, not during capture (where the K=64 wrapper falls
        back to the default tactic) or at run time.
        """
        import torch

        try:
            if "do_preparation" in inspect.signature(runner.forward).parameters:
                runner(tensors, tactic=-1, do_preparation=True, **kwargs)
            runner(tensors, tactic=tactic, **kwargs)
            torch.cuda.current_stream().synchronize()
        except Exception as e:
            with contextlib.suppress(Exception):
                torch.accelerator.synchronize()
            with contextlib.suppress(Exception):
                torch.cuda.cudart().cudaGetLastError()
            return f"warm-up failed: {type(e).__name__}: {e}"
        return None

    def _check(self, c: Candidate, info: dict[str, Any]) -> str | None:
        if info.get("error"):
            return f"unvalidated: {info['error']}"
        if c.gid is not None:
            if c.gid != info["g"]:
                old = self.table_groups.get(c.gid, {}).get("parts")
                new = self.groups.groups[info["g"]]["parts"]
                return f"fingerprint: {diff_parts(old, new)}"
            if c.meta.get("ra") is not None and c.meta["ra"] != info["ra"]:
                return "runner attributes: configuration differs"
            if c.meta.get("tl") is not None and c.meta["tl"] != info["tl"]:
                return (
                    f"tactic list: changed ({c.meta.get('nt')} -> {info['nt']} tactics)"
                )
        if tactic_to_json(c.tactic) not in info["tactics"]:
            return f"tactic not offered: {c.tactic}"
        return None

    def _reject(self, fk: str, op: str, reasons: list[str]) -> None:
        self.stats["stale"] += 1
        for r in set(map(_reason_class, reasons)):
            self.stale_reasons[f"{op}: {r}"] += 1
        if self._warned < _MAX_KEY_WARNINGS:
            self._warned += 1
            logger.warning(
                "FlashInfer autotune: stale pinned entry rejected for %s: %s",
                fk,
                " | ".join(reasons),
            )
        else:
            logger.debug("Stale pinned entry %s: %s", fk, " | ".join(reasons))

    # -- results ------------------------------------------------------------
    def tuned_entries(self, tuner: Any) -> dict[str, list[Any]]:
        """Entries tuned in this process for keys without a valid candidate."""
        out = {}
        for key, (tactic, _) in list(tuner.profiling_cache.items()):
            meta = self.misses.get(key.file_key)
            if meta is None:
                continue
            m = {k: meta[k] for k in ("g", "tl", "nt", "ra")}
            m["src"] = "tuned"
            out[key.file_key] = [key.runner_class_name, tactic_to_json(tactic), m]
        return out

    def effective_digest(self, tuner: Any) -> str:
        items = [
            (fk, runner, tactic_to_json(tactic))
            for fk, (runner, tactic) in tuner._file_configs.items()
        ]
        items += [
            (k.file_key, k.runner_class_name, tactic_to_json(t))
            for k, (t, _) in tuner.profiling_cache.items()
        ]
        return stable_hash(sorted(items, key=lambda x: (x[0], x[1])))

    def summary(self, tuner: Any) -> dict[str, Any]:
        unused: Counter[str] = Counter()
        for fk, cands in self.index.items():
            if fk not in self.decided:
                unused[fk.split(",", 1)[0].strip("('\"")] += 1
        hits = sum(v for k, v in self.stats.items() if k.startswith("hit_"))
        return {
            "keys": hits + self.stats["miss"],
            "hits": hits,
            "hit_detail": {
                k[4:]: v for k, v in sorted(self.stats.items()) if k.startswith("hit_")
            },
            "stale": self.stats["stale"],
            "stale_reasons": dict(self.stale_reasons),
            "stale_host_entries": self.stats["stale_host"],
            "misses": self.stats["miss"],
            "untuned_lookups": self.stats["untuned_lookup"],
            "unused": dict(unused),
            "digest": self.effective_digest(tuner),
        }

    def cache_payload(self, tuner: Any) -> dict[str, Any]:
        entries = {fk: [v] for fk, v in self.accepted.items()}
        for fk, v in self.tuned_entries(tuner).items():
            entries[fk] = [v]
        gids = {v[0][2]["g"] for v in entries.values()}
        groups = {}
        for g in gids:
            info = self.groups.groups.get(g) or self.table_groups.get(g)
            if info is not None:
                groups[g] = info
        return build_table_json(
            fi_meta={k: self.host.get(k) for k in FI_META_KEYS},
            host=self.host,
            groups=groups,
            entries=entries,
            provenance={
                "kind": "cache",
                "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
            },
        )


def _load_tables(
    pinned: Path | None, cache_bytes: bytes | None, cache_path: Path
) -> list[Table]:
    tables = []
    if pinned is not None:
        if pinned.is_file():
            tables.append(read_table(pinned, "pinned"))
        else:
            logger.warning(
                "VLLM_FLASHINFER_AUTOTUNE_FILE=%s does not exist; every "
                "FlashInfer autotune key is a miss",
                pinned,
            )
    if cache_bytes is not None:
        try:
            tables.append(
                parse_table(
                    json.loads(cache_bytes),
                    path=str(cache_path),
                    source="cache",
                    sha256=hashlib.sha256(cache_bytes).hexdigest(),
                )
            )
        except ValueError as e:
            logger.warning(
                "FlashInfer autotune cache %s is unreadable and ignored: %s",
                cache_path,
                e,
            )
    return tables


def _worker_tag(runner: Any) -> str:
    import torch

    dp = getattr(getattr(runner, "parallel_config", None), "data_parallel_rank", None)
    return (
        f"host={socket.gethostname()} pid={os.getpid()} "
        f"cuda={torch.accelerator.current_device_index()} gpu={_gpu_uuid()[:13]}"
        + (f" dp_rank={dp}" if dp is not None else "")
    )


def run_pinned_autotune(
    runner: Any,
    *,
    world: GroupCoordinator,
    tuner: Any,
    cache_path: Path,
    pinned_path: Path,
    strict: bool,
    run_passes: Callable[[], None],
) -> dict[str, Any]:
    """The kernel warmup's FlashInfer autotune with a pinned table."""
    global LAST_SESSION
    is_leader = world.rank_in_group == 0
    lock = (
        interprocess_lock(cache_path.with_name(cache_path.name + ".lock"))
        if is_leader
        else contextlib.nullcontext((0.0, False))
    )
    with lock as (waited, locked):
        cache_bytes = None
        if is_leader and cache_path.exists():
            cache_bytes = cache_path.read_bytes()
        if world.world_size > 1:
            cache_bytes = world.broadcast_object(cache_bytes, src=0)
        tables = _load_tables(pinned_path, cache_bytes, cache_path)
        session = PinSession(tables, strict=strict)
        LAST_SESSION = session
        session.install(tuner)
        try:
            run_passes()
        finally:
            session.uninstall()
        summary = session.summary(tuner)
        written = None
        if is_leader and not strict and session.tuned_entries(tuner):
            written = write_json_atomic(cache_path, session.cache_payload(tuner))
    pinned = next((t for t in tables if t.source == "pinned"), None)
    logger.info(
        "FlashInfer autotune pin [%s]: file %s sha256=%s | keys %d: hits %d %s, "
        "stale %d %s, misses %d%s | unused pinned entries %s | stale-host "
        "entries %d | table digest %s | lock waited %.1fs%s%s",
        _worker_tag(runner),
        pinned_path,
        pinned.sha256[:12] if pinned else "-",
        summary["keys"],
        summary["hits"],
        summary["hit_detail"],
        summary["stale"],
        summary["stale_reasons"] or "",
        summary["misses"],
        " (strict: fallback tactic)" if strict else " (tuned)",
        summary["unused"] or 0,
        summary["stale_host_entries"],
        summary["digest"],
        waited,
        "" if locked or not is_leader else " (unlocked)",
        f" | wrote {cache_path} sha256={written[:12]}" if written else "",
    )
    if strict and summary["misses"]:
        missing = [fk for fk, d in session.decided.items() if d == "strict"]
        raise RuntimeError(
            f"VLLM_FLASHINFER_AUTOTUNE_STRICT=1: {summary['misses']} FlashInfer "
            f"autotune keys have no valid entry in {pinned_path} "
            f"(stale {summary['stale']}: {summary['stale_reasons']}); first: "
            + "; ".join(missing[:10])
        )
    return summary


# --------------------------------------------------------------------------
# Generation: robust per-worker measurements.
# --------------------------------------------------------------------------


class RecordSession:
    """Profiles every key with repeated timings and records all samples."""

    def __init__(self, out_dir: Path, rounds: int) -> None:
        self.out_dir = out_dir
        self.rounds = max(int(rounds), 1)
        self.keys: dict[str, dict[str, Any]] = {}
        self.groups = _Groups()
        self._local = threading.local()
        self._tuner: Any = None
        self._orig: dict[str, Any] = {}

    def install(self, tuner: Any) -> None:
        self._tuner = tuner
        for name in (
            "choose_one",
            "_profile_single_kernel",
            "_prepare_input_tensors_with_batches",
        ):
            self._orig[name] = getattr(tuner, name)
        tuner.choose_one = self._choose_one
        tuner._profile_single_kernel = self._profile
        tuner._prepare_input_tensors_with_batches = self._prepare

    def uninstall(self) -> None:
        for name in self._orig:
            with contextlib.suppress(AttributeError):
                delattr(self._tuner, name)
        self._orig.clear()

    def _choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
        prev = getattr(self._local, "op", None)
        self._local.op = custom_op
        try:
            return self._orig["choose_one"](
                custom_op, runners, tuning_config, inputs, **kwargs
            )
        finally:
            self._local.op = prev

    def _prepare(self, inputs, tuning_config):
        batches = None
        if (
            getattr(self._local, "op", None) == "mxfp8_gemm"
            and tuning_config.use_cold_l2_cache
        ):
            batches = self._weights_cold_batches(inputs)
        self._local.l2 = "weights-cold" if batches is not None else "flashinfer"
        if batches is None:
            batches = self._orig["_prepare_input_tensors_with_batches"](
                inputs, tuning_config
            )
        return batches

    def _weights_cold_batches(self, inputs: list[Any]) -> list[list[Any]] | None:
        """mm_mxfp8 inputs (a, b, a_sf, b_sf, dtype, out, ws): rotate the weight
        and its scales through copies totalling > 2.5x L2 and keep a / out / ws
        resident, as in decode serving. None (FlashInfer's cold inputs) when the
        activation and output alone exceed half of L2 (large M: nothing stays
        resident in serving either).
        """
        import torch

        a, b, b_sf, out = inputs[0], inputs[1], inputs[3], inputs[5]
        l2 = self._tuner._get_l2_cache_size_in_bytes()
        act = sum(
            t.numel() * t.element_size()
            for t in (a, inputs[2], out)
            if isinstance(t, torch.Tensor)
        )
        if act >= 0.5 * l2:
            return None
        per = b.numel() * b.element_size() + b_sf.numel() * b_sf.element_size()
        copies = min(max(math.ceil(2.5 * l2 / max(per, 1)), 2), 256)
        batches = []
        for _ in range(copies):
            nb = torch.empty_strided(
                b.size(), b.stride(), dtype=b.dtype, device=b.device
            )
            nb.copy_(b)
            x = list(inputs)
            x[1], x[3] = nb, b_sf.clone()
            batches.append(x)
        return batches

    def _profile(
        self,
        runner: Any,
        inputs: list[Any],
        tactic: Any,
        tuning_config: Any,
        input_tensor_batches: list[list[Any]] | None = None,
        **kwargs: Any,
    ) -> float:
        import flashinfer.autotuner.autotuner as fa
        import torch
        from flashinfer.autotuner import AutoTuner

        tuner = self._tuner
        op = getattr(self._local, "op", None) or "?"
        shapes = tuple(
            tuple(t.size()) if isinstance(t, torch.Tensor) else (0,) for t in inputs
        )
        key = AutoTuner._get_cache_key(
            op, runner, shapes, tuning_config, runner.get_cache_key_extras(inputs)
        )
        rec = self.keys.get(key.file_key)
        if rec is None:
            rec = self.keys[key.file_key] = {
                "op": op,
                "runner": type(runner).__name__,
                "g": self.groups.gid(op, runner),
                "ra": runner_attrs_hash(runner),
                "l2": getattr(self._local, "l2", None),
                "tactics": [],
                "samples": [],
            }
        tj = tactic_to_json(tactic)
        if tj in rec["tactics"]:
            idx = rec["tactics"].index(tj)
        else:
            rec["tactics"].append(tj)
            rec["samples"].append(None)
            idx = len(rec["tactics"]) - 1

        if input_tensor_batches is None:
            input_tensor_batches = tuner._prepare_input_tensors_with_batches(
                inputs, tuning_config
            )
        n = max(tuner.repeat, len(input_tensor_batches))
        stream = torch.cuda.current_stream()
        samples: list[float] = []
        exc: BaseException | None = None
        value = float("inf")
        try:
            with fa._profile_measurement_scope(), torch.cuda.stream(stream):
                for _ in range(tuner.warmup):
                    runner(input_tensor_batches[-1], tactic=tactic, **kwargs)

                def run() -> None:
                    for i in range(n):
                        runner(
                            input_tensor_batches[i % len(input_tensor_batches)],
                            tactic=tactic,
                            **kwargs,
                        )

                graph = None
                if tuning_config.use_cuda_graph:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                stream.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                delay = (
                    tuner._CUDA_GRAPH_DELAY_MICRO_SECS
                    if graph is not None
                    else tuner.stream_delay_micro_secs
                )
                for _ in range(self.rounds):
                    fa.delay_kernel(delay)
                    start.record()
                    if graph is not None:
                        graph.replay()
                    else:
                        run()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / n)
            value = statistics.median(samples)
        except BaseException as e:  # noqa: BLE001 - re-raised below
            exc = e
        group = fa._tune_process_group
        if group is not None:
            # Same collective contract as FlashInfer's _profile_single_kernel.
            import torch.distributed as dist

            backend = str(dist.get_backend(group)).lower()
            t = torch.tensor(
                [value],
                dtype=torch.float64,
                device="cuda" if backend == "nccl" else "cpu",
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
            value = t.item() / dist.get_world_size(group)
        rec["samples"][idx] = (
            [round(s * 1000.0, 4) for s in samples] if exc is None else None
        )
        if exc is not None:
            raise exc
        return value

    def write(self, runner: Any) -> Path:
        import torch

        host = runtime_host()
        keys = {}
        for fk, rec in self.keys.items():
            keys[fk] = dict(rec, tl=tactic_list_hash(rec["tactics"]))
        dp = getattr(
            getattr(runner, "parallel_config", None), "data_parallel_rank", None
        )
        payload = {
            "kind": RECORD_KIND,
            "schema": SCHEMA,
            "host": host,
            "fi_meta": {k: host.get(k) for k in FI_META_KEYS},
            "groups": self.groups.groups,
            "worker": {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "cuda_device": torch.accelerator.current_device_index(),
                "gpu_uuid": _gpu_uuid(),
                "dp_rank": dp,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            "method": {
                "rounds": self.rounds,
                "repeat": self._tuner.repeat if self._tuner else None,
                "warmup": self._tuner.warmup if self._tuner else None,
                "stat": "median over rounds of the per-call mean of one "
                "CUDA-graph replay",
                "l2": "mxfp8_gemm: weights rotate (>2.5x L2), activations/output "
                "resident; other ops: FlashInfer cold-L2 inputs",
                "unit": "us",
            },
            "env": {
                k: v
                for k, v in sorted(os.environ.items())
                if k.startswith(("VLLM_", "LCD_FI_", "PFW", "GS2_", "FLASHINFER_"))
            },
            "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "keys": keys,
        }
        name = f"record-{socket.gethostname()}-{_gpu_uuid()[:13]}-{os.getpid()}.json"
        path = self.out_dir / name
        sha = write_json_atomic(path, payload)
        logger.info(
            "FlashInfer autotune record [%s]: %d keys, %d groups -> %s (sha256 %s)",
            _worker_tag(runner),
            len(keys),
            len(self.groups.groups),
            path,
            sha[:12],
        )
        return path


def run_record_autotune(
    runner: Any,
    *,
    tuner: Any,
    out_dir: Path,
    rounds: int,
    run_passes: Callable[[], None],
) -> Path:
    session = RecordSession(out_dir, rounds)
    session.install(tuner)
    try:
        run_passes()
    finally:
        session.uninstall()
    return session.write(runner)


# --------------------------------------------------------------------------
# Merge (offline, CPU only).
# --------------------------------------------------------------------------


def _median(xs: Sequence[float]) -> float:
    return statistics.median(xs) if xs else math.inf


def merge_records(
    sets: Sequence[tuple[str, Sequence[dict[str, Any]]]],
    *,
    primary: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge per-worker records into a pinned table.

    ``sets`` = [(name, [record, ...]), ...], one set per configuration (e.g.
    C512 env, C32 env). Per (group, key) every worker's samples of a tactic are
    reduced to their median; the pick is the tactic with the lowest median over
    workers (ties: first in tactic order). Returns (table JSON, report).
    """
    if not sets or not any(recs for _, recs in sets):
        raise ValueError("no records")
    primary = primary or sets[0][0]
    host = None
    fi_meta = None
    groups: dict[str, dict[str, Any]] = {}
    per_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    set_groups: dict[str, set[str]] = {}
    workers: dict[str, list[dict[str, Any]]] = {}
    for name, recs in sets:
        set_groups[name] = set()
        workers[name] = []
        for rec in recs:
            if rec.get("kind") != RECORD_KIND or rec.get("schema") != SCHEMA:
                raise ValueError(f"not a schema-{SCHEMA} record: {rec.get('kind')}")
            if host is None:
                host, fi_meta = rec["host"], rec["fi_meta"]
            else:
                diff = {
                    k: (host.get(k), rec["host"].get(k))
                    for k in HOST_KEYS
                    if host.get(k) != rec["host"].get(k)
                }
                if diff:
                    raise ValueError(f"records from different hosts/builds: {diff}")
            workers[name].append(rec["worker"])
            for g, info in rec["groups"].items():
                if g in groups and groups[g]["fp"] != info["fp"]:
                    raise ValueError(f"group {g} fingerprint conflict")
                groups.setdefault(g, info)
            for fk, k in rec["keys"].items():
                set_groups[name].add(k["g"])
                per_key.setdefault((k["g"], fk), []).append(k)
    if primary not in set_groups:
        raise ValueError(f"primary set {primary!r} not among {list(set_groups)}")

    entries: dict[str, list[list[Any]]] = {}
    report_keys = []
    for (g, fk), ks in sorted(per_key.items()):
        tls = {k["tl"] for k in ks}
        if len(tls) != 1:
            raise ValueError(f"{fk} ({g}): workers saw different tactic lists {tls}")
        tactics = ks[0]["tactics"]
        ras = {k["ra"] for k in ks}
        if len(ras) != 1:
            raise ValueError(f"{fk} ({g}): workers have different runner attributes")
        agg = []
        worker_picks = []
        for k in ks:
            meds = [
                _median(s) if s else math.inf
                for s in (k["samples"] + [None] * (len(tactics) - len(k["samples"])))
            ]
            worker_picks.append(min(range(len(meds)), key=lambda i: (meds[i], i)))
            agg.append(meds)
        per_tactic = [
            _median([w[i] for w in agg])
            if all(math.isfinite(w[i]) for w in agg)
            else math.inf
            for i in range(len(tactics))
        ]
        order = sorted(range(len(tactics)), key=lambda i: (per_tactic[i], i))
        best = order[0]
        if not math.isfinite(per_tactic[best]):
            raise ValueError(f"{fk} ({g}): no tactic ran on every worker")
        second = order[1] if len(order) > 1 else None
        us = per_tactic[best]
        us2 = per_tactic[second] if second is not None else None
        spread = [w[best] for w in agg]
        meta = {
            "g": g,
            "tl": next(iter(tls)),
            "nt": len(tactics),
            "ra": next(iter(ras)),
            "us": round(us, 3),
            "us2": round(us2, 3) if us2 is not None and math.isfinite(us2) else None,
            "w": len(ks),
            "agree": sum(p == best for p in worker_picks),
            "wspread": round((max(spread) - min(spread)) / us, 4) if us > 0 else 0.0,
        }
        cand = [ks[0]["runner"], tactics[best], meta]
        slot = entries.setdefault(fk, [])
        if g in set_groups[primary]:
            slot.insert(0, cand)
        else:
            slot.append(cand)
        report_keys.append(
            {
                "file_key": fk,
                "g": g,
                "op": ks[0]["op"],
                "tactic": tactics[best],
                "us": meta["us"],
                "us2": meta["us2"],
                "margin": round((us2 - us) / us, 4)
                if us2 is not None and math.isfinite(us2) and us > 0
                else None,
                "w": len(ks),
                "agree": meta["agree"],
                "wspread": meta["wspread"],
            }
        )
    provenance = {
        "kind": "pinned",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary": primary,
        "sets": {
            name: {
                "groups": sorted(set_groups[name]),
                "workers": [
                    f"{w.get('hostname')}:{str(w.get('gpu_uuid'))[:13]}"
                    for w in workers[name]
                ],
            }
            for name in set_groups
        },
        "method": "per key: tactic with the lowest median over workers of each "
        "worker's median over CUDA-graph replays (vLLM record mode)",
    }
    assert host is not None and fi_meta is not None
    table = build_table_json(
        fi_meta=fi_meta,
        host=host,
        groups={g: groups[g] for g in {e[2]["g"] for v in entries.values() for e in v}},
        entries=entries,
        provenance=provenance,
    )
    return table, {"keys": report_keys, "provenance": provenance}


def describe_table(table: Table) -> list[str]:
    kind = "legacy FlashInfer format" if table.legacy else f"schema {SCHEMA}"
    lines = [
        (
            f"{table.path}: sha256 {table.sha256[:12]} {kind}, "
            f"{len(table.entries)} keys, {table.num_entries()} candidates"
        )
    ]
    if table.host:
        lines.append("host: " + json.dumps(table.host, sort_keys=True))
    per: Counter[str] = Counter()
    for fk, cands in table.entries.items():
        for c in cands:
            per[c.gid or f"{fk.split(',', 1)[0].strip('(')}|{c.runner}|legacy"] += 1
    for g, n in sorted(per.items()):
        parts = table.groups.get(g, {}).get("parts", {})
        env = parts.get("env", {}) if isinstance(parts, dict) else {}
        lines.append(f"  {n:4d}  {g}  env={env}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m vllm.model_executor.warmup.flashinfer_autotune_pin"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="merge per-worker records into a pinned file")
    m.add_argument(
        "--set",
        action="append",
        required=True,
        metavar="NAME=GLOB",
        help="one configuration's record files (repeatable)",
    )
    m.add_argument("--primary", help="set whose groups are the top-level entries")
    m.add_argument("--out", required=True)
    m.add_argument("--report", help="per-key report (JSON)")
    s = sub.add_parser("show", help="summarize pinned / cache files")
    s.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "merge":
        sets = []
        for item in args.set:
            name, _, pattern = item.partition("=")
            files = sorted(glob.glob(pattern))
            if not files:
                raise SystemExit(f"--set {item}: no files")
            sets.append((name, [json.loads(Path(f).read_text()) for f in files]))
        table, report = merge_records(sets, primary=args.primary)
        sha = write_json_atomic(args.out, table)
        if args.report:
            write_json_atomic(args.report, report)
        keys = report["keys"]
        close = sum(1 for k in keys if k["margin"] is not None and k["margin"] < 0.02)
        differ = sum(k["agree"] < k["w"] for k in keys)
        print(
            f"wrote {args.out} sha256 {sha}: {len(keys)} (group, key) picks, "
            f"{close} within 2% of the runner-up, "
            f"{differ} where some worker's own pick differs"
        )
        for line in describe_table(read_table(args.out)):
            print(line)
    else:
        for f in args.files:
            for line in describe_table(read_table(f)):
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
