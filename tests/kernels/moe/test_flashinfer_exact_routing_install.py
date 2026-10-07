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

    stock_gate = core._device_support_moe_pdl
    monkeypatch.setattr(core, "_device_support_moe_pdl", stock_gate)
    for var in (
        "VLLM_MOE_PDL_FC",
        "VLLM_FI_SM107_MOE_PDL_MAX_TOKENS",
        "VLLM_FLASHINFER_MOE_ROUTING_MODULE_CACHE",
    ):
        monkeypatch.delenv(var, raising=False)

    def run(gs2=True, **env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(fer, "ENABLED", gs2)
        monkeypatch.setattr(fer, "PREBUILT", env.get("GS2_ROUTE_PREBUILT"))
        monkeypatch.setattr(fer, "_installed", False)
        monkeypatch.setattr(fer, "_pdl_fc_installed", False)
        monkeypatch.setattr(fer, "_PDL_MODULE_READY", False)
        monkeypatch.setattr(core, "gen_trtllm_gen_fused_moe_sm100_module", stock_gen)
        monkeypatch.setattr(core, "_device_support_moe_pdl", stock_gate)
        fer.maybe_install()
        return core.gen_trtllm_gen_fused_moe_sm100_module()

    run.core = core
    run.stock_gate = stock_gate
    run.stock = {Path(p).name: Path(p) for p in sources}

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


LAUNCHER = "trtllm_fused_moe_kernel_launcher.cu"
ROUTING = ("trtllm_fused_moe_routing_common.cu", "trtllm_fused_moe_routing_custom.cu")


def _by_name(spec):
    return {Path(p).name: Path(p) for p in spec.sources}


def _assert_launcher_patched(path: Path):
    data = path.read_bytes()
    n_sites = data.count(b"routing_runner.run(")
    assert n_sites == 4
    assert data.count(b"/*enable_pdl=*/false);  // [vllm moe-pdl-fc]") == n_sites


def test_gs2_route_alone_leaves_launcher_and_pdl_gate(install):
    spec = install(gs2=True)
    assert spec.name == "fused_moe_trtllm_sm100_exact_routing"
    srcs = _by_name(spec)
    assert srcs[LAUNCHER] == install.stock[LAUNCHER]
    assert all(srcs[n] != install.stock[n] for n in ROUTING)
    assert install.core._device_support_moe_pdl is install.stock_gate


@pytest.mark.parametrize("gs2", [False, True])
def test_pdl_fc_patches_launcher_and_composes(install, gs2):
    spec = install(gs2=gs2, VLLM_MOE_PDL_FC="1")
    stock = "fused_moe_trtllm_sm100"
    assert spec.name == (f"{stock}_exact_routing_pdlfc" if gs2 else f"{stock}_pdlfc")
    srcs = _by_name(spec)
    _assert_launcher_patched(srcs[LAUNCHER])
    assert all((srcs[n] != install.stock[n]) == gs2 for n in ROUTING)
    assert install.core._device_support_moe_pdl is fer._moe_pdl_fc_supported


def test_pdl_fc_gate_closed_until_patched_module_generated(install, monkeypatch):
    core = install.core
    monkeypatch.setattr(core, "device_support_pdl", lambda device: True)
    monkeypatch.setenv("VLLM_MOE_PDL_FC", "1")
    monkeypatch.setattr(fer, "_pdl_fc_installed", False)
    monkeypatch.setattr(fer, "_PDL_MODULE_READY", False)
    fer.maybe_install()
    assert core._device_support_moe_pdl("cuda:0") is False
    core.gen_trtllm_gen_fused_moe_sm100_module()
    assert core._device_support_moe_pdl("cuda:0") is True


@pytest.mark.parametrize("conflict", ["max_tokens", "prebuilt"])
def test_pdl_fc_ignored_with_conflicting_setting(install, tmp_path, conflict):
    env = {"VLLM_MOE_PDL_FC": "1"}
    if conflict == "max_tokens":
        env["VLLM_FI_SM107_MOE_PDL_MAX_TOKENS"] = "1024"
    else:
        prebuilt = tmp_path / "prebuilt.so"
        prebuilt.write_bytes(b"prebuilt")
        env["GS2_ROUTE_PREBUILT"] = str(prebuilt)
    spec = install(gs2=True, **env)
    assert spec.name == "fused_moe_trtllm_sm100_exact_routing"
    assert _by_name(spec)[LAUNCHER] == install.stock[LAUNCHER]
    assert install.core._device_support_moe_pdl is install.stock_gate


def test_launcher_patch_refuses_mismatched_source(tmp_path):
    stock = CSRC / LAUNCHER
    if not stock.is_file():
        pytest.skip(f"{stock} not installed")
    lines = stock.read_bytes().split(b"\n")
    # Drop the first routing call's context so the hunk no longer matches.
    i = next(k for k, line in enumerate(lines) if b"replay_ptr, enable_pdl);" in line)
    bad = tmp_path / "src" / LAUNCHER
    bad.parent.mkdir()
    bad.write_bytes(b"\n".join(lines[:i] + [b"  enable_pdl);"] + lines[i + 1 :]))
    with pytest.raises(ValueError, match="context mismatch"):
        fer.patched_launcher_source(bad, tmp_path / "out")
