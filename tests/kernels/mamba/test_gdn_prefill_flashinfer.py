# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    pytest.skip(
        reason="FlashInfer GDN prefill is not supported on ROCm.",
        allow_module_level=True,
    )

import flashinfer.gdn_prefill  # noqa: E402

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fi_chunk_gated_delta_rule,
)  # noqa: E402


@pytest.mark.parametrize("backend", [None, "sm107"])
def test_flashinfer_gdn_prefill_preserves_metadata(monkeypatch, backend):
    captured_cu_seqlens = None
    captured_kwargs = {}

    def fake_chunk_gated_delta_rule(**kwargs):
        nonlocal captured_cu_seqlens
        captured_cu_seqlens = kwargs["cu_seqlens"]
        captured_kwargs.update(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(
        flashinfer.gdn_prefill,
        "chunk_gated_delta_rule",
        fake_chunk_gated_delta_rule,
    )
    q = torch.zeros(1, 2, 1, 2)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

    output, final_state = fi_chunk_gated_delta_rule(
        q=q,
        k=q,
        v=q,
        g=torch.zeros(1, 2, 1),
        beta=torch.zeros(1, 2, 1),
        initial_state=torch.zeros(1, 1, 2, 2),
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        backend=backend,
    )

    assert captured_cu_seqlens is not None
    assert captured_cu_seqlens.dtype == torch.int64
    assert output.shape == q.shape
    assert final_state is None

    if backend is None:
        assert "backend" not in captured_kwargs
    else:
        assert captured_kwargs["backend"] == "sm107"
        assert captured_kwargs["use_cp"] is False


@pytest.mark.parametrize(
    "capability,head_dim,cuda_major,requested,expected",
    [
        (107, 128, 13, "flashinfer_sm107", "flashinfer_sm107"),
        (107, 128, 13, "auto", "flashinfer"),
        (107, 128, 13, "flashinfer", "flashinfer"),
        (100, 128, 13, "flashinfer_sm107", "triton"),
        (103, 128, 13, "flashinfer_sm107", "triton"),
        (120, 128, 13, "flashinfer_sm107", "triton"),
        (90, 128, 13, "flashinfer_sm107", "triton"),
        (107, 64, 13, "flashinfer_sm107", "triton"),
        (107, 128, 12, "flashinfer_sm107", "triton"),
    ],
)
@pytest.mark.parametrize(
    "native_dependency", ["available", "missing_api", "missing_kernel"]
)
def test_gdn_prefill_backend_is_explicit_and_arch_specific(
    monkeypatch,
    capability,
    head_dim,
    cuda_major,
    requested,
    expected,
    native_dependency,
):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _resolve_gdn_prefill_backend,
    )

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda cc: cc == capability
    )
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda cc: cc // 10 == capability // 10,
    )
    monkeypatch.setattr(current_platform, "get_cuda_runtime_major", lambda: cuda_major)

    def native_api(*, backend: str = "auto") -> None:
        pass

    def legacy_api() -> None:
        pass

    monkeypatch.setattr(
        flashinfer.gdn_prefill,
        "chunk_gated_delta_rule",
        native_api if native_dependency != "missing_api" else legacy_api,
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.gdn_kernels.rubin.gated_delta_net_chunked",
        SimpleNamespace(RubinGatedDeltaNetChunkedKernel=SimpleNamespace(arch="sm_107"))
        if native_dependency != "missing_kernel"
        else None,
    )
    config = Mock(
        additional_config={"gdn_prefill_backend": requested},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(linear_key_head_dim=head_dim)
        ),
    )
    if expected == "flashinfer_sm107" and native_dependency != "available":
        with pytest.raises(RuntimeError, match="source-built FlashInfer"):
            _resolve_gdn_prefill_backend(config)
    else:
        assert _resolve_gdn_prefill_backend(config) == (requested, expected)
