# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

# FlashInfer autotune-file metadata keys compared against the runtime.
_PINNED_METADATA_KEYS = (
    "flashinfer_version",
    "cuda_version",
    "cublas_version",
    "cudnn_version",
    "cudnn_frontend_version",
    "gpu",
)
# Cache files already handled by seed_pinned_autotune_file (one log each).
_SEEDED: set[Path] = set()


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    config_hash = runner.vllm_config.compute_hash(include_version=False)
    return hashlib.sha256(config_hash.encode()).hexdigest()


def resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "autotune_configs.json"
    if envs.VLLM_FLASHINFER_AUTOTUNE_FILE:
        seed_pinned_autotune_file(
            cache_file, Path(envs.VLLM_FLASHINFER_AUTOTUNE_FILE).expanduser()
        )
    return cache_file


def _runtime_autotune_metadata() -> dict[str, str] | None:
    """FlashInfer's own autotune-file metadata for this process.

    None if this FlashInfer build does not expose it.
    """
    for name in ("flashinfer.autotuner.autotuner", "flashinfer.autotuner"):
        try:
            import importlib

            collect = getattr(importlib.import_module(name), "_collect_metadata", None)
        except Exception:
            # Optional: an older or missing FlashInfer just skips the comparison.
            collect = None
        if collect is not None:
            try:
                return {k: str(v) for k, v in collect().items()}
            except Exception as exc:
                logger.warning(
                    "FlashInfer autotune metadata unavailable (%s); "
                    "pinned file metadata not compared",
                    exc,
                )
                return None
    return None


def compare_autotune_metadata(
    pinned: dict, runtime: dict[str, str] | None
) -> list[str]:
    """Return ``key: pinned != runtime`` strings for the compared keys."""
    if runtime is None:
        return []
    out = []
    for key in _PINNED_METADATA_KEYS:
        if key in pinned and key in runtime and str(pinned[key]) != runtime[key]:
            out.append(f"{key}: {pinned[key]} != {runtime[key]}")
    return out


def seed_pinned_autotune_file(cache_file: Path, pinned: Path) -> None:
    """Seed ``cache_file`` from the pinned autotune file
    (``VLLM_FLASHINFER_AUTOTUNE_FILE``) if it does not exist yet.

    The cache directory depends on the engine config hash, so the pinned file
    is copied into whatever directory vLLM resolved. An existing cache file is
    left unchanged (its sha256 is logged next to the pinned file's). Shapes
    missing from the pinned file autotune normally, and the merged result is
    saved to ``cache_file`` as usual. A file whose ``_metadata`` differs from
    the runtime is still installed (with a warning); FlashInfer decides at
    ``load_configs`` whether to use it.
    """
    if cache_file in _SEEDED:
        return
    _SEEDED.add(cache_file)
    if not pinned.is_file():
        logger.warning(
            "VLLM_FLASHINFER_AUTOTUNE_FILE=%s does not exist; FlashInfer "
            "autotunes at start-up (cache %s)",
            pinned,
            cache_file,
        )
        return
    contents = pinned.read_bytes()
    sha = hashlib.sha256(contents).hexdigest()
    try:
        meta = json.loads(contents).get("_metadata", {}) or {}
    except (ValueError, AttributeError):
        logger.warning(
            "VLLM_FLASHINFER_AUTOTUNE_FILE=%s is not a JSON object; not used",
            pinned,
        )
        return
    if cache_file.exists():
        cur = hashlib.sha256(cache_file.read_bytes()).hexdigest()
        logger.info(
            "FlashInfer autotune cache file already present: %s sha256=%s "
            "(pinned %s sha256=%s, %s); left unchanged",
            cache_file,
            cur,
            pinned,
            sha,
            "same" if cur == sha else "different",
        )
        return
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_name(f".{cache_file.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(pinned, tmp)
        os.replace(tmp, cache_file)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise
    logger.info(
        "Seeded FlashInfer autotune cache %s from pinned file %s sha256=%s "
        "_metadata=%s",
        cache_file,
        pinned,
        sha,
        json.dumps(meta, sort_keys=True),
    )
    diffs = compare_autotune_metadata(meta, _runtime_autotune_metadata())
    if diffs:
        logger.warning(
            "Pinned FlashInfer autotune file %s was recorded on a different "
            "runtime (%s); FlashInfer may reject it and autotune instead",
            pinned,
            "; ".join(diffs),
        )


def sync_flashinfer_autotune_cache(
    runner: "GPUModelRunner",
    group: "GroupCoordinator",
) -> None:
    cache: bytes | str | None = None
    if (
        group.rank_in_group == 0
        and runner.vllm_config.kernel_config.enable_flashinfer_autotune
    ):
        try:
            from vllm.platforms import current_platform
            from vllm.utils.flashinfer import has_flashinfer

            if has_flashinfer() and current_platform.has_device_capability(90):
                from flashinfer.autotuner import AutoTuner

                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "autotune_configs.json"
                    AutoTuner.get().save_configs(str(path))
                    cache = path.read_bytes()
        except Exception as exc:
            cache = f"{type(exc).__name__}: {exc}"

    cache = group.broadcast_object(cache)
    if isinstance(cache, str):
        raise RuntimeError(f"Failed to serialize FlashInfer autotune state: {cache}")
    if cache is None or group.rank_in_group == 0:
        return

    from flashinfer.autotuner import AutoTuner

    with tempfile.NamedTemporaryFile() as f:
        f.write(cache)
        f.flush()
        if not AutoTuner.get().load_configs(f.name):
            raise RuntimeError("FlashInfer autotune cache is incompatible")


def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=cache_path.parent, suffix=".tmp", prefix=f".{cache_path.name}."
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        os.replace(tmp_path, cache_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
