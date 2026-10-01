# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MXFP8 producers' early PDL trigger is off whenever Inductor PDL is on."""

import pytest
import torch._inductor
from torch._inductor import config as inductor_config

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fusion import mxfp8_pdl


@pytest.fixture(autouse=True)
def _restore_policy():
    saved = mxfp8_pdl._early_trigger
    yield
    mxfp8_pdl._early_trigger = saved


@pytest.mark.parametrize("global_pdl", [False, True])
@pytest.mark.parametrize("entry", [None, False, True])
def test_inductor_pdl_enabled_precedence(monkeypatch, global_pdl, entry) -> None:
    monkeypatch.setattr(inductor_config.triton, "enable_pdl", global_pdl)
    cfg = {} if entry is None else {mxfp8_pdl.INDUCTOR_PDL_KEY: entry}
    expected = global_pdl if entry is None else entry
    assert mxfp8_pdl.inductor_pdl_enabled(cfg) is expected
    assert mxfp8_pdl.configure_mxfp8_producer_early_trigger(cfg) is (not expected)
    assert mxfp8_pdl.mxfp8_producer_early_trigger() is (not expected)


def test_unreadable_inductor_config_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(torch._inductor, "config", object())
    assert mxfp8_pdl.inductor_pdl_enabled({})
    assert not mxfp8_pdl.configure_mxfp8_producer_early_trigger({})
    assert not mxfp8_pdl.mxfp8_producer_early_trigger()


def test_policy_error_fails_closed(monkeypatch) -> None:
    mxfp8_pdl._early_trigger = True

    def boom(_cfg):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(mxfp8_pdl, "inductor_pdl_enabled", boom)
    assert not mxfp8_pdl.configure_mxfp8_producer_early_trigger({})
    assert not mxfp8_pdl.mxfp8_producer_early_trigger()


@pytest.mark.parametrize("inductor_pdl", [False, True])
def test_set_current_vllm_config_sets_policy(monkeypatch, inductor_pdl) -> None:
    monkeypatch.setattr(inductor_config.triton, "enable_pdl", False)
    mxfp8_pdl._early_trigger = inductor_pdl
    config = VllmConfig(
        compilation_config=CompilationConfig(
            inductor_compile_config={mxfp8_pdl.INDUCTOR_PDL_KEY: inductor_pdl}
        )
    )
    with set_current_vllm_config(config):
        assert mxfp8_pdl.mxfp8_producer_early_trigger() is (not inductor_pdl)
