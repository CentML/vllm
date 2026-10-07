# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of how flashinfer_exact_routing.maybe_install selects the module
(JIT build, persistent cache, prebuilt) with a fake FlashInfer module spec."""

import dataclasses
import importlib.util
from pathlib import Path

import pytest

flashinfer = pytest.importorskip("flashinfer")


def _load_module():
    # Load the file directly: importing the fused_moe package needs the CUDA
    # extensions, this module only needs vllm.envs and vllm.logger.
    path = (
        Path(__file__).parents[3]
        / "vllm/model_executor/layers/fused_moe/flashinfer_exact_routing.py"
    )
    spec = importlib.util.spec_from_file_location("_flashinfer_exact_routing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fer = _load_module()

CSRC = Path(flashinfer.__file__).parent / "data" / "csrc"
ROUTING_DIR = CSRC / "fused_moe" / "trtllm_backend"


@dataclasses.dataclass
class FakeSpec:
    """Mirrors the JitSpecNvcc load protocol: ``try_load`` loads the AOT
    artifact if present, ``build`` writes the JIT library."""

    name: str
    sources: list
    root: Path

    @property
    def jit_library_path(self) -> Path:
        return self.root / "jit" / self.name / f"{self.name}.so"

    @property
    def aot_path(self) -> Path:
        return self.root / "aot" / f"{self.name}.so"

    def try_load(self):
        if self.aot_path.exists():
            return self.load(self.aot_path)
        return None

    def load(self, path=None):
        path = Path(path or self.jit_library_path)
        return (path, path.read_bytes())

    def build(self):
        self.jit_library_path.parent.mkdir(parents=True, exist_ok=True)
        self.jit_library_path.write_bytes(b"built:" + self.name.encode())

    def build_and_load(self):
        module = self.try_load()
        if module is not None:
            return module
        self.build()
        return self.load()


@pytest.fixture
def install(monkeypatch, tmp_path):
    """Return ``gen(**env) -> spec``: runs maybe_install with the given
    environment against a fake stock spec and returns the generated spec."""
    from flashinfer.fused_moe import core
    from flashinfer.jit import env as jit_env

    sources = [
        ROUTING_DIR / "trtllm_fused_moe_routing_common.cu",
        ROUTING_DIR / "trtllm_fused_moe_routing_custom.cu",
        CSRC / "trtllm_fused_moe_kernel_launcher.cu",
    ]
    for p in sources:
        if not p.is_file():
            pytest.skip(f"{p} not installed")

    def stock_gen(enable_rubin=False):
        return FakeSpec("fused_moe_trtllm_sm100", list(sources), tmp_path / "ws")

    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "gen")
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a: (10, 7))
    monkeypatch.setattr(core, "gen_trtllm_gen_fused_moe_sm100_module", stock_gen)

    def run(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(fer, "ENABLED", True)
        monkeypatch.setattr(fer, "PREBUILT", env.get("GS2_ROUTE_PREBUILT"))
        monkeypatch.setattr(fer, "_installed", False)
        monkeypatch.setattr(core, "gen_trtllm_gen_fused_moe_sm100_module", stock_gen)
        fer.maybe_install()
        return core.gen_trtllm_gen_fused_moe_sm100_module()

    return run


def test_module_cache_reused_across_workspaces(install, tmp_path):
    cache = tmp_path / "cache"
    first = install(VLLM_FLASHINFER_MOE_ROUTING_MODULE_CACHE=str(cache))
    path, data = first.build_and_load()
    assert path == first.jit_library_path
    cached = list(cache.glob(f"{first.name}-*/{first.name}.so"))
    assert len(cached) == 1 and cached[0].read_bytes() == data

    # A fresh workspace loads the cached build instead of rebuilding.
    first.jit_library_path.unlink()
    second = install(VLLM_FLASHINFER_MOE_ROUTING_MODULE_CACHE=str(cache))
    assert second.try_load() == (second.jit_library_path, data)


def test_prebuilt_wins_over_module_cache(install, tmp_path):
    cache = tmp_path / "cache"
    cached_spec = install(VLLM_FLASHINFER_MOE_ROUTING_MODULE_CACHE=str(cache))
    cached_spec.build_and_load()
    assert list(cache.glob("*/*.so"))  # a cache entry the prebuilt must beat

    prebuilt = tmp_path / "prebuilt.so"
    prebuilt.write_bytes(b"prebuilt")
    spec = install(
        VLLM_FLASHINFER_MOE_ROUTING_MODULE_CACHE=str(cache),
        GS2_ROUTE_PREBUILT=str(prebuilt),
    )
    assert spec.name == cached_spec.name
    assert spec.build_and_load() == (prebuilt, b"prebuilt")
