# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_FLASHINFER_AUTOTUNE_FILE: seeding of the resolved autotune cache file."""

import hashlib
import json
from types import SimpleNamespace

import pytest

import vllm.model_executor.warmup.flashinfer_autotune_cache as fac

PINNED = {
    "_metadata": {"flashinfer_version": "0.6.18.post1", "gpu": "GPU-A"},
    "('op', 'Runner', ((1, 2),), ())": ["Runner", [3, 4]],
}


def _runner(config_hash: str):
    cfg = SimpleNamespace(compute_hash=lambda include_version=False: config_hash)
    return SimpleNamespace(vllm_config=cfg)


@pytest.fixture
def env(tmp_path, monkeypatch):
    pinned = tmp_path / "pinned.json"
    pinned.write_text(json.dumps(PINNED))
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(fac, "_SEEDED", set())
    monkeypatch.setattr(fac, "_runtime_autotune_metadata", lambda: None)
    return pinned


def test_unset_keeps_default_behaviour(env, monkeypatch):
    monkeypatch.delenv("VLLM_FLASHINFER_AUTOTUNE_FILE", raising=False)
    path = fac.resolve_flashinfer_autotune_file(_runner("cfg-a"))
    assert path.name == "autotune_configs.json"
    assert path.parent.is_dir()
    assert not path.exists()


@pytest.mark.parametrize("config_hash", ["cfg-a", "cfg-b"])
def test_seeds_whatever_hash_dir(env, monkeypatch, config_hash):
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_FILE", str(env))
    path = fac.resolve_flashinfer_autotune_file(_runner(config_hash))
    assert path.parent.name == hashlib.sha256(config_hash.encode()).hexdigest()
    assert path.read_bytes() == env.read_bytes()
    assert not list(path.parent.glob(".*.tmp"))


def test_existing_cache_file_is_left_unchanged(env, monkeypatch):
    monkeypatch.delenv("VLLM_FLASHINFER_AUTOTUNE_FILE", raising=False)
    path = fac.resolve_flashinfer_autotune_file(_runner("cfg-a"))
    path.write_text('{"tuned": 1}')
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_FILE", str(env))
    assert fac.resolve_flashinfer_autotune_file(_runner("cfg-a")) == path
    assert path.read_text() == '{"tuned": 1}'


def test_missing_pinned_file_does_not_fail(env, monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_FILE", str(tmp_path / "absent.json"))
    path = fac.resolve_flashinfer_autotune_file(_runner("cfg-a"))
    assert not path.exists()


def test_metadata_mismatch_warns_but_installs(env, monkeypatch):
    monkeypatch.setattr(
        fac,
        "_runtime_autotune_metadata",
        lambda: {"flashinfer_version": "0.6.18.post1", "gpu": "GPU-B"},
    )
    warnings = []
    monkeypatch.setattr(fac.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_FILE", str(env))
    path = fac.resolve_flashinfer_autotune_file(_runner("cfg-a"))
    assert path.read_bytes() == env.read_bytes()
    assert len(warnings) == 1 and "gpu: GPU-A != GPU-B" in warnings[0]


def test_compare_autotune_metadata():
    pinned = {"flashinfer_version": "1", "cublas_version": "13.7.0", "x": "y"}
    assert fac.compare_autotune_metadata(pinned, None) == []
    assert fac.compare_autotune_metadata(pinned, dict(pinned)) == []
    assert fac.compare_autotune_metadata(
        pinned, {"flashinfer_version": "1", "cublas_version": "13.8.0"}
    ) == ["cublas_version: 13.7.0 != 13.8.0"]


def test_env_registered(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_FLASHINFER_AUTOTUNE_FILE", raising=False)

    assert "VLLM_FLASHINFER_AUTOTUNE_FILE" in envs.environment_variables
    assert envs.environment_variables["VLLM_FLASHINFER_AUTOTUNE_FILE"]() is None
