# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only, sha-checked FlashInfer autotune cache (``EWARM_AUTOTUNE_RO=1``).

``flashinfer_autotune()`` writes the autotune cache file back on every start,
twice: ``write_flashinfer_autotune_cache(cache_path, <bytes just read>)`` and
``AutoTuner.save_configs(cache_path)`` after the tuning pass. Neither is
guarded, so on a read-only cache directory the ``PermissionError`` aborts
engine start-up. Tuned tactics are configuration: pinning a read-only copy
keeps kernel selection reproducible across starts and servers.

With ``EWARM_AUTOTUNE_RO=1``:

- the write-back of the bytes just read is skipped; the bytes are checked
  against ``EWARM_AUTOTUNE_MANIFEST`` (lines ``<sha256>  <hash-dir>/<file>``).
  A missing entry or a sha mismatch is an error.
- ``AutoTuner.save_configs`` is not called. Instead, entries tuned at start-up
  that are not in the pinned file (cache misses) are counted; any such entry,
  or a pinned file that was never loaded and verified, is an error.
- Loading is unchanged: ``AutoTuner.load_configs`` reads the pinned file.

Errors raise ``RuntimeError`` unless ``VLLM_AUTOTUNE_RO_STRICT=0`` (default
``1``), in which case they are only logged.
"""

import hashlib
import os
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

PINNED_AUTOTUNE_RO = os.environ.get("EWARM_AUTOTUNE_RO", "0") == "1"
_STRICT = os.environ.get("VLLM_AUTOTUNE_RO_STRICT", "1") == "1"
_MANIFEST = os.environ.get("EWARM_AUTOTUNE_MANIFEST", "")
_STATE = {"checked": 0}


def log_read_only_enabled() -> None:
    logger.info_once(
        "vLLM autotune cache write-back disabled (EWARM_AUTOTUNE_RO=1)"
    )
    logger.info_once(
        "FlashInfer AutoTuner.save_configs disabled (EWARM_AUTOTUNE_RO=1)"
    )


def _manifest() -> dict[str, str]:
    out: dict[str, str] = {}
    if _MANIFEST:
        with open(_MANIFEST) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    out[parts[1]] = parts[0]
    return out


def _fail(msg: str) -> None:
    logger.error("EWARM_AUTOTUNE_RO ERROR: %s", msg)
    if _STRICT:
        raise RuntimeError("EWARM_AUTOTUNE_RO: " + msg)


def verify_pinned_autotune_file(cache_path: Path, contents: bytes) -> None:
    """Replaces the write-back of the bytes just read from ``cache_path``."""
    rel = f"{cache_path.parent.name}/{cache_path.name}"
    sha = hashlib.sha256(contents).hexdigest()
    want = _manifest().get(rel)
    if want is None:
        _fail(f"pinned autotune file {rel} is not in the manifest {_MANIFEST}")
    elif want != sha:
        _fail(f"pinned autotune file {rel} sha256 {sha[:16]} != manifest {want[:16]}")
    else:
        _STATE["checked"] += 1
        logger.info(
            "EWARM_AUTOTUNE_RO pinned autotune file verified: %s sha256=%s "
            "(%d bytes); write-back skipped (read-only)",
            cache_path,
            sha[:16],
            len(contents),
        )


def check_no_startup_tuning(tuner: Any, path: str) -> None:
    """Replaces ``tuner.save_configs(path)``: nothing is written; fails if
    the pinned file was not loaded / verified or if any entry was tuned at
    start-up (not found in the pinned file)."""
    extra = []
    with tuner._lock:
        n_file = len(tuner._file_configs)
        for ck in list(tuner.profiling_cache.keys()):
            fk = getattr(ck, "file_key", None)
            if fk is not None and fk not in tuner._file_configs:
                extra.append(fk)
    if n_file == 0 or _STATE["checked"] == 0:
        _fail(
            f"save_configs({path}): the pinned autotune file was not "
            f"loaded/verified (file configs={n_file}, "
            f"verified={_STATE['checked']})"
        )
    if extra:
        _fail(
            f"save_configs({path}): {len(extra)} entries were tuned at start-up "
            f"and are NOT in the pinned file: {[str(e)[:120] for e in extra[:4]]}"
        )
    logger.info(
        "EWARM_AUTOTUNE_RO save_configs skipped (pinned read-only cache): %s; "
        "file configs=%d, start-up tuned=%d",
        path,
        n_file,
        len(extra),
    )
