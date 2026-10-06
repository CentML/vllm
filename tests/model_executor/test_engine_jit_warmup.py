# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import threading
from pathlib import Path

import pytest

from vllm.model_executor.warmup import engine_jit_warmup as ewarm
from vllm.model_executor.warmup import pinned_autotune_cache as pinned

pytestmark = pytest.mark.cpu_test


def _key_record(module: str, name: str, extra: str = "") -> str:
    sd = json.dumps({"name": f"{module}.{name}", "key": f"k{extra}"})
    return json.dumps({"module": module, "name": name, "specialization_data": sd})


def test_load_keys_dedup_and_alias(tmp_path, monkeypatch):
    (tmp_path / "a.jsonl").write_text(
        "\n".join(
            [
                _key_record("pkg.mod", "_kern"),
                _key_record("pkg.mod", "_kern"),
                _key_record("pkg.mod", "_kern", "2"),
                "",
            ]
        )
    )
    (tmp_path / "b.jsonl").write_text(_key_record("old.mod", "_other"))
    monkeypatch.setattr(ewarm, "_KEYS", str(tmp_path))
    monkeypatch.setattr(ewarm, "TRITON_KEY_MODULE_ALIASES", {"old.mod": "new.mod"})

    keys = ewarm._load_keys()

    assert list(keys) == [("pkg.mod", "_kern"), ("new.mod", "_other")]
    assert len(keys[("pkg.mod", "_kern")]) == 2
    # Unaliased records are passed through verbatim.
    assert keys[("pkg.mod", "_kern")][0] == json.loads(
        _key_record("pkg.mod", "_kern")
    )["specialization_data"]
    # Aliased records carry the new full name that preload() checks.
    (moved,) = keys[("new.mod", "_other")]
    assert json.loads(moved)["name"] == "new.mod._other"


def test_load_keys_without_dir(monkeypatch):
    monkeypatch.setattr(ewarm, "_KEYS", "")
    assert ewarm._load_keys() == {}


class _FakeCacheKey:
    def __init__(self, file_key: str) -> None:
        self.file_key = file_key


class _FakeTuner:
    def __init__(self, file_keys, tuned_keys):
        self._lock = threading.RLock()
        self._file_configs = {k: ("runner", 0) for k in file_keys}
        self.profiling_cache = {_FakeCacheKey(k): None for k in tuned_keys}


def _pin(tmp_path: Path, monkeypatch, contents: bytes, sha: str | None = None):
    cache_path = tmp_path / "hashdir" / "autotune_configs.json"
    manifest = tmp_path / "MANIFEST.sha256"
    sha = sha or hashlib.sha256(contents).hexdigest()
    manifest.write_text(f"{sha}  hashdir/autotune_configs.json\n")
    monkeypatch.setattr(pinned, "_MANIFEST", str(manifest))
    monkeypatch.setattr(pinned, "_STRICT", True)
    monkeypatch.setattr(pinned, "_STATE", {"checked": 0})
    return cache_path


def test_pinned_autotune_verified_and_no_startup_tuning(tmp_path, monkeypatch):
    cache_path = _pin(tmp_path, monkeypatch, b"{}")
    pinned.verify_pinned_autotune_file(cache_path, b"{}")
    assert pinned._STATE["checked"] == 1
    assert not cache_path.exists()  # nothing written
    pinned.check_no_startup_tuning(_FakeTuner(["a", "b"], ["a"]), str(cache_path))


def test_pinned_autotune_sha_mismatch_raises(tmp_path, monkeypatch):
    cache_path = _pin(tmp_path, monkeypatch, b"{}", sha="0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        pinned.verify_pinned_autotune_file(cache_path, b"{}")


def test_pinned_autotune_startup_tuning_raises(tmp_path, monkeypatch):
    cache_path = _pin(tmp_path, monkeypatch, b"{}")
    pinned.verify_pinned_autotune_file(cache_path, b"{}")
    with pytest.raises(RuntimeError, match="tuned at start-up"):
        pinned.check_no_startup_tuning(
            _FakeTuner(["a"], ["a", "new"]), str(cache_path)
        )


def test_pinned_autotune_not_loaded_raises(tmp_path, monkeypatch):
    cache_path = _pin(tmp_path, monkeypatch, b"{}")
    with pytest.raises(RuntimeError, match="not loaded/verified"):
        pinned.check_no_startup_tuning(_FakeTuner(["a"], []), str(cache_path))
