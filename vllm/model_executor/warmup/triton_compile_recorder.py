# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton compile recorder (diagnostics; ``VLLM_JIT_WARMUP_RECORD_DIR=<dir>``).

Chains ``triton.knobs.runtime.jit_post_compile_hook`` and appends one JSON
line per Triton compile to ``<dir>/rec-<pid>.jsonl``: kernel module and
qualname, Triton's own ``specialization_data`` (the JSON that
``JITFunction.preload()`` consumes), the cache key and whether the JIT monitor
was already active (i.e. the compile happened during inference). The records
of kernels that compiled after start-up are the input for the recorded-key
JIT warmup (``EWARM_KEYS``, see ``engine_jit_warmup.py``).
"""

import json
import os
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_RECORD_DIR = os.environ.get("VLLM_JIT_WARMUP_RECORD_DIR", "")
_INSTALLED = False


def _fn_id(fn: Any) -> tuple[str, str]:
    jf = getattr(fn, "jit_function", None) or fn
    inner = getattr(jf, "fn", None)
    mod = getattr(inner, "__module__", None) or getattr(jf, "__module__", "?")
    name = getattr(inner, "__qualname__", None) or getattr(jf, "__name__", "?")
    return mod, name


def maybe_install_triton_compile_recorder() -> None:
    """Chain ``triton.knobs.runtime.jit_post_compile_hook`` and append one JSON
    line per Triton compile to ``<VLLM_JIT_WARMUP_RECORD_DIR>/rec-<pid>.jsonl``.
    No-op unless ``VLLM_JIT_WARMUP_RECORD_DIR`` is set."""
    global _INSTALLED
    if not _RECORD_DIR or _INSTALLED:
        return
    try:
        from triton import knobs
    except Exception as e:  # no triton: nothing to record
        logger.warning("Triton compile recorder: triton knobs unavailable: %r", e)
        return
    os.makedirs(_RECORD_DIR, exist_ok=True)
    path = os.path.join(_RECORD_DIR, f"rec-{os.getpid()}.jsonl")
    prev = knobs.runtime.jit_post_compile_hook

    def hook(*args: Any, **kwargs: Any) -> Any:
        try:
            comp = kwargs.get("compile") or {}
            mod, name = _fn_id(kwargs.get("fn"))
            try:
                from vllm.utils import jit_monitor

                active: bool | None = bool(jit_monitor.is_active())
            except Exception:
                active = None
            rec = {
                "t": time.time(),
                "module": mod,
                "name": name,
                "active": active,
                "is_warmup": comp.get("is_warmup"),
                "key": str(comp.get("key")),
                "specialization_data": comp.get("specialization_data"),
                "compile_keys": sorted(map(str, comp.keys())),
                "compile": {
                    str(k): str(v)[:4000] for k, v in comp.items() if k not in ("fn",)
                },
            }
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # recording must never break serving
            logger.warning("Triton compile recorder: record failed: %r", e)
        if prev is not None:
            return prev(*args, **kwargs)
        return False

    knobs.runtime.jit_post_compile_hook = hook
    _INSTALLED = True
    logger.info("Triton compile recorder: recording Triton compiles to %s", path)
