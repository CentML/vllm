# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Additional in-engine JIT warmup registrations (``EWARM=1``).

Registers extra kernels with the runner's ``JitWarmupRegistry`` right before
``kernel_warmup()`` runs ``registry.warmup()``, so they are compiled during
engine start-up (before CUDA-graph capture and before the JIT monitor is
activated) instead of on first use during inference. Three groups:

1. The sampler's top-k/top-p Triton kernels
   (``register_top_k_top_p_warmups``). Idempotent: a no-op when the model
   runner already registered them.
2. Recorded Triton keys: one ``TritonPreloadKernel`` per kernel found in the
   JSONL files under ``EWARM_KEYS``. Each compile key is a Triton
   ``specialization_data`` record (the JSON consumed by
   ``JITFunction.preload()``), captured from real runs with the Triton
   compile recorder (see below). ``preload()`` compiles the kernel from its
   current source (or loads it from the Triton disk cache) and installs the
   binary in the kernel's in-memory cache under the recorded key, so the
   first runtime launch with that specialization does not JIT. A key that
   fails to preload is logged and skipped; it never aborts start-up.
3. FlashInfer GDN chunked prefill (CuTe DSL): runs
   ``flashinfer.gdn_prefill.chunk_gated_delta_rule`` once on the
   context-parallel path (one long sequence) and once on the non-CP path
   (several balanced sequences) with the indexed state pool, on scratch
   tensors that have the exact layout of the live GDN SSM state pool
   (``torch.empty_strided`` with the pool's shape, strides and dtype).
   FlashInfer caches the compiled CuTe kernels in-process; nothing in the
   live cache is touched.

Recorded key files (``EWARM_KEYS/*.jsonl``) hold one JSON object per line::

    {"module": "<module defining the kernel>",
     "name": "<kernel qualname in that module>",
     "specialization_data": "<Triton specialization_data JSON string>"}

``JITFunction.preload()`` (Triton 3.8) rejects a record whose
``specialization_data["name"]`` differs from ``f"{fn.__module__}.{qualname}"``
of the kernel it is called on, and it requires the recorded GPU target to
match. The in-memory kernel cache key (``specialization_data["key"]``) is the
argument specialization plus launch options and does not contain the module
path, and the on-disk Triton cache key hashes the kernel source (and its line
number), not the module path. Keys recorded for a kernel that later moved to
another module can therefore be reused by remapping ``module`` / ``name``
through ``TRITON_KEY_MODULE_ALIASES``, as long as the kernel's parameters and
launch options are unchanged; otherwise re-record them.

Environment variables:

- ``EWARM``: ``1`` enables these registrations (default ``0``).
- ``EWARM_KEYS``: directory with recorded key files (``*.jsonl``); unset or
  empty disables group 2.
- ``VLLM_JIT_WARMUP_GDN_PREFILL``: ``0`` disables group 3 (default ``1``).

Key files are produced with the Triton compile recorder
(``vllm/model_executor/warmup/triton_compile_recorder.py``,
``VLLM_JIT_WARMUP_RECORD_DIR``): keep the records of kernels that compiled
after start-up and reduce them to ``module`` / ``name`` /
``specialization_data``, one file per kernel.
"""

import glob
import importlib
import json
import os
import sys
import time
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry, VllmJitKernel

logger = init_logger(__name__)

EWARM_ENABLED = os.environ.get("EWARM", "0") == "1"
_GDN = os.environ.get("VLLM_JIT_WARMUP_GDN_PREFILL", "1") == "1"
_KEYS = os.environ.get("EWARM_KEYS", "")

# Recorded module name -> module that defines the kernel now (same qualname),
# or "module:qualname" -> "module:qualname" for a kernel that was also
# renamed. Applied to key records before import / preload() so that key files
# recorded before a kernel moved into vLLM keep working.
# TODO: add an entry for every recorded kernel that moved; e.g.
#   "old_pkg.kernels": "vllm.model_executor.layers.quantization.kernels",
TRITON_KEY_MODULE_ALIASES: dict[str, str] = {}

STATS = {
    "triton_keys": 0,
    "triton_ok": 0,
    "triton_fail": 0,
    "gdn_ok": 0,
    "gdn_fail": 0,
    "topkp": 0,
}


def _alias(module: str, name: str) -> tuple[str, str]:
    new = TRITON_KEY_MODULE_ALIASES.get(f"{module}:{name}")
    if new is not None:
        new_module, _, new_name = new.partition(":")
        return new_module, new_name or name
    return TRITON_KEY_MODULE_ALIASES.get(module, module), name


def _alias_specialization_data(sd: str, module: str, name: str) -> str:
    obj = json.loads(sd)
    obj["name"] = f"{module}.{name}"
    # Constexpr arguments that are JIT functions are resolved by
    # "module:qualname" at preload time.
    vals = obj.get("constant_vals") or []
    for i, val in enumerate(vals):
        if isinstance(val, dict) and "jit_function" in val:
            m, _, q = val["jit_function"].partition(":")
            m, q = _alias(m, q)
            vals[i] = {"jit_function": f"{m}:{q}"}
    return json.dumps(obj)


def _load_keys() -> dict[tuple[str, str], list[str]]:
    """{(module, qualname): [specialization_data, ...]} from EWARM_KEYS/*.jsonl
    (deduplicated, file and line order kept)."""
    keys: dict[tuple[str, str], list[str]] = {}
    if not _KEYS:
        return keys
    for path in sorted(glob.glob(os.path.join(_KEYS, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sd = rec.get("specialization_data")
                if not sd:
                    continue
                module, name = _alias(rec["module"], rec["name"])
                if (module, name) != (rec["module"], rec["name"]):
                    sd = _alias_specialization_data(sd, module, name)
                lst = keys.setdefault((module, name), [])
                if sd not in lst:
                    lst.append(sd)
    return keys


def _resolve_jit(module: str, name: str) -> Any:
    obj: Any = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    # Unwrap Autotuner / Heuristics wrappers down to the JITFunction.
    while not hasattr(obj, "preload") and hasattr(obj, "fn"):
        obj = obj.fn
    return obj


class TritonPreloadKernel(VllmJitKernel[str]):
    """One Triton JIT kernel; compile keys are Triton specialization_data."""

    CompileKey = str

    def __init__(self, module: str, name: str, keys: list[str]) -> None:
        super().__init__()
        self.module, self.name, self.keys = module, name, tuple(keys)

    def get_warmup_keys(self, *args: Any, **kwargs: Any) -> list[str]:
        return list(self.keys)

    def compile(self, compile_key: str) -> None:
        try:
            jf = _resolve_jit(self.module, self.name)
            jf.preload(compile_key)
            STATS["triton_ok"] += 1
        except Exception as e:
            # A stale key must not abort engine start-up; log it.
            STATS["triton_fail"] += 1
            logger.warning(
                "EWARM preload failed for %s.%s: %r", self.module, self.name, e
            )
        self._compiled_cache[compile_key] = True


class FiGdnPrefillWarmup(VllmJitKernel[str]):
    """FlashInfer GDN chunked prefill (CuTe DSL) on the live state-pool
    layout; compile keys "cp" and "noncp"."""

    CompileKey = str

    def __init__(self, vllm_config: Any) -> None:
        super().__init__()
        self.vllm_config = vllm_config

    def get_warmup_keys(self, *args: Any, **kwargs: Any) -> list[str]:
        return ["cp", "noncp"]

    def _pool(self) -> tuple[Any, torch.Tensor | None]:
        from vllm.model_executor.warmup.qwen_triton_warmup import (
            _iter_qwen_gdn_layers,
            _split_qwen_gdn_cache,
        )

        ctx = self.vllm_config.compilation_config.static_forward_context
        for layer in _iter_qwen_gdn_layers(ctx):
            kv = getattr(layer, "kv_cache", None)
            if isinstance(kv, (list, tuple)) and kv and not torch.is_tensor(kv[0]):
                kv = kv[0]
            split = _split_qwen_gdn_cache(kv)
            if split is not None:
                return layer, split[1]
        return None, None

    @torch.inference_mode()
    def compile(self, compile_key: str) -> None:
        try:
            from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_gdn

            layer, pool = self._pool()
            if pool is None:
                logger.info("EWARM GDN warmup skipped: no GDN SSM pool found")
                self._compiled_cache[compile_key] = True
                return
            dev = pool.device
            # Scratch pool: 2 slots with the live pool's strides / dtype
            # (slot 1 is used; nothing live is touched).
            scratch = torch.empty_strided(
                (2,) + tuple(pool.shape[1:]),
                pool.stride(),
                dtype=pool.dtype,
                device=dev,
            )
            scratch.zero_()
            HV, V, K = pool.shape[1], pool.shape[2], pool.shape[3]
            H = int(layer.num_k_heads // max(getattr(layer, "tp_size", 1), 1))
            if compile_key == "cp":
                # One long sequence -> context-parallel path.
                lens = [16384]
            else:
                # Balanced multi-sequence batch -> non-CP kernel.
                lens = [2048, 2048, 2048, 2048]
            T = sum(lens)
            cu = torch.tensor(
                [0] + list(torch.tensor(lens).cumsum(0).tolist()),
                dtype=torch.int32,
                device=dev,
            )
            q = torch.randn(T, H, K, device=dev, dtype=torch.bfloat16) * 0.05
            k = torch.randn(T, H, K, device=dev, dtype=torch.bfloat16) * 0.05
            v = torch.randn(T, HV, V, device=dev, dtype=torch.bfloat16) * 0.05
            g = torch.full((T, HV), 0.99, device=dev, dtype=torch.float32)
            beta = torch.full((T, HV), 0.5, device=dev, dtype=torch.float32)
            out = torch.empty(T, HV, V, device=dev, dtype=torch.bfloat16)
            slots = torch.ones(len(lens), dtype=torch.int32, device=dev)
            kw: dict[str, Any] = {}
            if compile_key == "cp":
                # Pass max_seqlen exactly like the GDN layer does, when the
                # layer module provides the helper.
                mod = sys.modules.get(
                    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
                )
                fn = getattr(mod, "_gdn_fi_maxlen_kw", None) if mod else None
                if fn is not None:

                    class _MD:  # only prefill_max_seqlen is read
                        prefill_max_seqlen = max(lens)

                    kw = fn(True, _MD())
            fi_gdn(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=scratch,
                output_final_state=True,
                cu_seqlens=cu,
                output=out,
                output_state=scratch,
                use_cp=(compile_key == "cp"),
                state_indices=slots,
                **kw,
            )
            torch.cuda.synchronize(dev)
            del scratch, q, k, v, g, beta, out
            STATS["gdn_ok"] += 1
        except Exception as e:
            STATS["gdn_fail"] += 1
            logger.warning("EWARM GDN %s warmup failed: %r", compile_key, e)
        self._compiled_cache[compile_key] = True


_INSTANCES: list[VllmJitKernel[Any]] = []


def _register_all(registry: JitWarmupRegistry) -> None:
    try:
        from vllm.v1.sample.ops.topk_topp_sampler import (
            register_top_k_top_p_warmups,
        )

        register_top_k_top_p_warmups()
        STATS["topkp"] = 1
    except Exception as e:
        logger.warning("EWARM top-k/top-p warmup registration failed: %r", e)
    keys = _load_keys()
    for (module, name), lst in keys.items():
        kern = TritonPreloadKernel(module, name, lst)
        _INSTANCES.append(kern)
        kern.register_warmup()
        STATS["triton_keys"] += len(lst)
    if _GDN:
        gdn = FiGdnPrefillWarmup(registry.vllm_config)
        _INSTANCES.append(gdn)
        gdn.register_warmup()
    logger.info(
        "EWARM in-engine JIT warmup registered: top-k/top-p=%d, "
        "triton kernels=%d (%d keys) from %s, gdn=%d",
        STATS["topkp"],
        len(keys),
        STATS["triton_keys"],
        _KEYS or "<none>",
        int(_GDN),
    )


def register_engine_warmups(registry: JitWarmupRegistry) -> float:
    """Register the extra warmups with ``registry``; returns the start time
    for :func:`log_engine_warmup_done`. Never raises."""
    t0 = time.perf_counter()
    try:
        with registry.activate():
            _register_all(registry)
    except Exception as e:  # never block start-up
        logger.warning("EWARM registration failed: %r", e)
    return t0


def log_engine_warmup_done(t0: float) -> None:
    logger.info(
        "EWARM JIT warmup done in %.1fs: %s", time.perf_counter() - t0, STATS
    )
