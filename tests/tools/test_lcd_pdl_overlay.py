# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual-source compatibility checks, without requiring a CUDA device.

For both parents, set VLLM_TEST_FLASHINFER_SOURCES to a directory containing
stock/dense_blockscaled_gemm_sm100.py and
installer_patched/dense_blockscaled_gemm_sm100.py. For Rubin inheritance also
provide installer_patched/dense_blockscaled_gemm_sm107.py (or set
VLLM_TEST_FLASHINFER_SM107_SOURCE to that file separately). These must be the
real pinned sources, not generated stand-ins. Otherwise tests use the installed
FlashInfer sources, skipping only the unavailable source variant. Constructor
checks additionally require the real FlashInfer/CuTe DSL dependencies.
"""

import hashlib
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PERSISTENT = "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100"
SPLITK = "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_splitk"


def _load_module(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def overlay(monkeypatch):
    # Load just the import hook; no vLLM import or CUDA initialization is needed.
    monkeypatch.setenv("LCD_FI_OVERLAY", "0")
    monkeypatch.setenv("LCD_FI_OVERLAY_BASE_CHECK", "1")
    return _load_module(
        "_lcd_pdl_test_hook", ROOT / "vllm/lcd_pdl/__init__.py", monkeypatch
    )


@pytest.fixture
def installer(monkeypatch):
    return _load_module(
        "_lcd_pdl_test_installer",
        ROOT / "tools/install_flashinfer_sm107.py",
        monkeypatch,
    )


def _source_path(variant, name, installer):
    index = 0 if variant == "stock" else 1
    expected = installer.SOURCES[name][index]
    fixtures = os.environ.get("VLLM_TEST_FLASHINFER_SOURCES")
    child_fixture = os.environ.get("VLLM_TEST_FLASHINFER_SM107_SOURCE")
    if name == "dense_blockscaled_gemm_sm107.py" and child_fixture:
        path = Path(child_fixture)
        assert path.is_file(), f"Pinned SM107 source unavailable: {path}"
    elif fixtures:
        path = Path(fixtures) / variant / name
        if not path.is_file():
            pytest.skip(f"Actual {variant} source fixture unavailable: {path}")
    else:
        try:
            dist = importlib.metadata.distribution("flashinfer-python")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("FlashInfer source fixtures or installation required")
        path = Path(str(dist.locate_file(f"flashinfer/gemm/kernels/{name}")))
        if not path.is_file():
            pytest.skip(f"Installed source unavailable: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            pytest.skip(f"Installed source is not the {variant} variant: {path}")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, (
        f"Fixture must contain the real pinned {variant} source: {path}"
    )
    return path


@pytest.mark.parametrize("variant", ["stock", "installer_patched"])
@pytest.mark.parametrize("guard_env", [None, "1"])
def test_changed_real_parent_is_rejected(
    variant, guard_env, installer, overlay, monkeypatch, tmp_path
):
    parent = _source_path(variant, "dense_blockscaled_gemm_sm100.py", installer)
    # Even a nonfunctional source change must fail the exact-source guard.
    (tmp_path / parent.name).write_bytes(parent.read_bytes() + b"\n# changed\n")
    if guard_env is None:
        monkeypatch.delenv("LCD_FI_OVERLAY_BASE_CHECK", raising=False)
    else:
        monkeypatch.setenv("LCD_FI_OVERLAY_BASE_CHECK", guard_env)
    assert overlay._OverlayFinder().find_spec(PERSISTENT, [str(tmp_path)]) is None


@pytest.mark.parametrize("fullname", [PERSISTENT, SPLITK])
@pytest.mark.parametrize("guard_env", [None, "1"])
def test_missing_and_unsupported_sources_never_overlay(
    fullname, guard_env, overlay, monkeypatch, tmp_path
):
    if guard_env is None:
        monkeypatch.delenv("LCD_FI_OVERLAY_BASE_CHECK", raising=False)
    else:
        monkeypatch.setenv("LCD_FI_OVERLAY_BASE_CHECK", guard_env)
    finder = overlay._OverlayFinder()
    assert finder.find_spec(fullname, [str(tmp_path)]) is None
    name = fullname.rsplit(".", 1)[1] + ".py"
    (tmp_path / name).write_text("# unapproved FlashInfer source\n")
    importlib.invalidate_caches()
    assert finder.find_spec(fullname, [str(tmp_path)]) is None


@pytest.mark.parametrize("variant", ["stock", "installer_patched"])
def test_real_pinned_rubin_consumer_gets_barriers_on_both_approved_bases(
    variant, installer, overlay, monkeypatch
):
    # The real child consumes NamedBarrier objects from its replaced parent.
    # Exercise that contract with both actual approved source installations.
    parent = _source_path(variant, "dense_blockscaled_gemm_sm100.py", installer)
    child = _source_path(
        "installer_patched", "dense_blockscaled_gemm_sm107.py", installer
    )
    pipeline = pytest.importorskip("cutlass.pipeline")
    pytest.importorskip("flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_common")
    spec = overlay._OverlayFinder().find_spec(PERSISTENT, [str(parent.parent)])
    parent_module = _load_module(PERSISTENT, spec.origin, monkeypatch)
    subclass = _load_module(
        "flashinfer.gemm.kernels._lcd_pdl_test_sm107", child, monkeypatch
    ).Sm107BlockScaledPersistentDenseGemmKernel
    assert issubclass(subclass, parent_module.Sm100BlockScaledPersistentDenseGemmKernel)
    kernel = subclass(32, (128, 128, 128), (256, 128, 256), (2, 1))
    assert isinstance(kernel.epilog_sync_barrier, pipeline.NamedBarrier)
    assert isinstance(kernel.tmem_alloc_barrier, pipeline.NamedBarrier)
    assert kernel.epilog_sync_barrier.barrier_id == kernel.epilog_sync_bar_id
    assert kernel.tmem_alloc_barrier.barrier_id == kernel.tmem_ptr_sync_bar_id
    assert kernel.epilog_sync_barrier.num_threads == 128
    assert kernel.tmem_alloc_barrier.num_threads == 160
