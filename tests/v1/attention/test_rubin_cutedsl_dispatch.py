# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only contract tests; kernel correctness is tested in FlashInfer.

The adapters must preserve token indptr, strided KV views, folded input scales,
and the whole swizzled FP4 scale buffer with its row offset. Mocking the kernel
is the cheapest way to catch metadata changes without requiring new artifacts.
"""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops import rubin_cutedsl_prefill, vit_attn_wrappers


@pytest.fixture
def prefill_args(monkeypatch):
    monkeypatch.setattr(
        rubin_cutedsl_prefill,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True, is_device_capability=lambda cc: cc == 107
        ),
    )
    pool = torch.empty(4, 4, 16, 256, dtype=torch.float8_e4m3fn)
    return dict(
        query=torch.empty(4, 64, 128, dtype=torch.float8_e4m3fn),
        kv_cache=pool.split(128, dim=-1),
        output=torch.empty(4, 64, 64, dtype=torch.uint8),
        output_block_scale=torch.empty(128, 512, dtype=torch.float8_e4m3fn),
        o_sf_scale=256.0,
        o_sf_start_index=3,
        block_tables=torch.tensor([[3, 1], [2, 0]], dtype=torch.int32),
        seq_lens=torch.tensor([17, 9], dtype=torch.int32),
        cum_seq_lens_q=torch.tensor([0, 1, 4], dtype=torch.int32),
        cum_seq_lens_kv=torch.tensor([0, 17, 26], dtype=torch.int32),
        max_q_len=3,
        max_kv_len=17,
        bmm1_scale=0.125 * 0.5 * 0.25,
        bmm2_scale=2.0,
        causal=True,
        window_left=-1,
        sinks=None,
        logits_soft_cap=None,
    )


def test_prefill_preserves_strides_scales_and_mixed_batch_offset(
    monkeypatch, prefill_args
):
    kernel = Mock()
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.attention.cute_dsl",
        SimpleNamespace(cute_dsl_fmha_paged_prefill=kernel),
    )
    assert rubin_cutedsl_prefill.try_rubin_cutedsl_prefill(**prefill_args)
    args, kwargs = kernel.call_args
    assert args[1] is prefill_args["kv_cache"][0]
    assert args[1].stride(2) == 256  # Preserve split-K/V storage, no cache copy.
    assert args[5] is prefill_args["cum_seq_lens_kv"]
    assert kwargs["output_block_scale"] is prefill_args["output_block_scale"]
    assert kwargs["o_sf_start_index"] == 3
    assert kwargs["o_sf_scale"] == 256.0
    assert kwargs["sm_scale"] == prefill_args["bmm1_scale"]
    assert kwargs["scale_v"] == 2.0
    assert kwargs["use_fp16_softmax"] is True


@pytest.mark.parametrize("unsupported", ["window", "dequantized_kv", "fp8_output"])
def test_unsupported_prefill_keeps_existing_backend(prefill_args, unsupported):
    if unsupported == "window":
        prefill_args["window_left"] = 64
    elif unsupported == "dequantized_kv":
        prefill_args["kv_cache"] = torch.empty(4, 2, 4, 16, 128)
    else:
        prefill_args["output"] = torch.empty(4, 64, 128, dtype=torch.float8_e4m3fn)
    assert not rubin_cutedsl_prefill.try_rubin_cutedsl_prefill(**prefill_args)


def test_selected_prefill_does_not_hide_missing_artifact(monkeypatch, prefill_args):
    kernel = Mock(side_effect=FileNotFoundError("new FP4 artifact not published"))
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.attention.cute_dsl",
        SimpleNamespace(cute_dsl_fmha_paged_prefill=kernel),
    )
    with pytest.raises(FileNotFoundError, match="not published"):
        rubin_cutedsl_prefill.try_rubin_cutedsl_prefill(**prefill_args)


def test_bf16_prefill_does_not_pass_fp4_scale_offset(monkeypatch, prefill_args):
    kernel = Mock()
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.attention.cute_dsl",
        SimpleNamespace(cute_dsl_fmha_paged_prefill=kernel),
    )
    prefill_args["output"] = torch.empty(4, 64, 128, dtype=torch.bfloat16)
    prefill_args["output_block_scale"] = None
    prefill_args["o_sf_scale"] = None
    assert rubin_cutedsl_prefill.try_rubin_cutedsl_prefill(**prefill_args)
    assert kernel.call_args.kwargs["o_sf_start_index"] == 0


def test_vit_metadata_is_token_indptr_not_cudnn_element_offsets(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_RUBIN_CUTEDSL_VIT", True)
    offsets = np.array([0, 72, 200], dtype=np.int32)
    backend = AttentionBackendEnum.FLASHINFER
    assert MMEncoderAttention.compute_max_seqlen(backend, offsets) == 128
    assert (
        MMEncoderAttention.maybe_compute_seq_lens(backend, offsets, torch.device("cpu"))
        is None
    )
    result = MMEncoderAttention.maybe_recompute_cu_seqlens(
        backend,
        offsets,
        1152,
        1,
        torch.device("cpu"),
        fp8_padded_hidden_size=1280,
    )
    torch.testing.assert_close(result, torch.from_numpy(offsets))


def test_vit_passes_logical_d72_with_padded_storage_and_host_scales(monkeypatch):
    kernel = Mock()
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.attention.cute_dsl",
        SimpleNamespace(cute_dsl_fmha_vit=kernel),
    )
    q = torch.empty(1, 200, 16, 80, dtype=torch.float8_e4m3fn)
    # Captured encoder buffers include empty sequence slots and unused tokens.
    indptr = torch.tensor([0, 72, 190, 190], dtype=torch.int32)
    result = vit_attn_wrappers.rubin_cutedsl_vit_wrapper(
        q,
        q,
        q,
        indptr,
        torch.tensor(128),
        72,
        0.125,
        2.0,
        3.0,
        4.0,
    )
    args, kwargs = kernel.call_args
    assert args[0].shape == (200, 16, 80)
    assert args[4] is indptr
    assert kwargs["logical_head_dim"] == 72
    assert kwargs["scale_q"] == 2.0
    assert kwargs["scale_k"] == 3.0
    assert kwargs["scale_v"] == 4.0
    assert result.shape == q.shape
    assert result.dtype == torch.bfloat16
