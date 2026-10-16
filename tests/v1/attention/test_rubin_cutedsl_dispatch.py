# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only contract tests; kernel correctness is tested in FlashInfer.

The adapters must preserve token indptr, strided KV views, folded input scales,
and the whole swizzled FP4 scale buffer with its row offset. Mocking the kernel
is the cheapest way to catch metadata changes without requiring new artifacts.
"""

import sys
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock, PropertyMock

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops import rubin_cutedsl_prefill, vit_attn_wrappers
from vllm.v1.kv_cache_layout import KVCacheLayout

TUNED_POLICY = {
    "scheduler": "auto",
    "sequence_barrier": False,
    "mma_rowsum": True,
    "packed_nvfp4_epilogue": True,
}


@pytest.fixture(autouse=True)
def reset_prefill_policy():
    """The prefill policy and kernel options are cached per process."""
    caches = (
        rubin_cutedsl_prefill._tuned_prefill_policy,
        rubin_cutedsl_prefill._tuned_kernel_options,
    )
    for cached in caches:
        cached.cache_clear()
    yield
    for cached in caches:
        cached.cache_clear()


def fake_flashinfer(monkeypatch, kernel, select_query_layout=None):
    """Install a fake FlashInfer, with the tuned API if a selector is given."""
    api = {"cute_dsl_fmha_paged_prefill": kernel}
    if select_query_layout is not None:
        api["RUBIN_GQA_TUNED_POLICY"] = MappingProxyType(TUNED_POLICY)
        api["rubin_paged_prefill_query_layout"] = select_query_layout
    monkeypatch.setitem(
        sys.modules, "flashinfer.attention.cute_dsl", SimpleNamespace(**api)
    )


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


@pytest.mark.parametrize("fp4_output", [True, False])
def test_tuned_policy_selects_query_layout_once_per_batch(
    monkeypatch, prefill_args, fp4_output
):
    """The tuned policy passes FlashInfer's layout for the host query lengths,
    selected once per batch rather than once per layer."""
    monkeypatch.setattr(envs, "VLLM_RUBIN_CUTEDSL_PREFILL_POLICY", "tuned")
    kernel = Mock()
    select_query_layout = Mock(return_value=("uniform", "tma"))
    fake_flashinfer(monkeypatch, kernel, select_query_layout)
    if not fp4_output:
        prefill_args["output"] = torch.empty(4, 64, 128, dtype=torch.bfloat16)
        prefill_args["output_block_scale"] = None
        prefill_args["o_sf_scale"] = None
    prefill_args["query_lens_cpu"] = (1, 3)
    for _layer in range(3):
        assert rubin_cutedsl_prefill.try_rubin_cutedsl_prefill(**prefill_args)
    select_query_layout.assert_called_once_with((1, 3), fp4_output=fp4_output)
    kwargs = kernel.call_args.kwargs
    assert {name: kwargs[name] for name in TUNED_POLICY} == TUNED_POLICY
    assert (kwargs["query_layout"], kwargs["output_store"]) == ("uniform", "tma")


@pytest.mark.parametrize(
    "policy, tuned_api", [("legacy", True), ("tuned", False)], ids=["default", "old"]
)
def test_legacy_policy_keeps_original_export(
    monkeypatch, prefill_args, policy, tuned_api
):
    """The default policy, and the tuned one on a FlashInfer without the tuned
    API, keep the original packed-query export call."""
    monkeypatch.setattr(envs, "VLLM_RUBIN_CUTEDSL_PREFILL_POLICY", policy)
    kernel = Mock()
    select_query_layout = Mock(return_value=("uniform", "tma"))
    fake_flashinfer(monkeypatch, kernel, select_query_layout if tuned_api else None)
    prefill_args["query_lens_cpu"] = (1, 3)
    assert rubin_cutedsl_prefill.try_rubin_cutedsl_prefill(**prefill_args)
    select_query_layout.assert_not_called()
    extra = {"query_layout", "output_store", *TUNED_POLICY}
    assert not extra & kernel.call_args.kwargs.keys()


class _ReachedKVCache(Exception):
    pass


@pytest.mark.parametrize("trtllm_prefill", [True, False])
def test_prefill_gate_warns_only_off_the_trtllm_route(monkeypatch, trtllm_prefill):
    """Opting in must not read removed metadata fields such as `use_cascade`,
    and only batches off the TRTLLM prefill route report a fallback."""
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends import flashinfer as fi_backend

    monkeypatch.setattr(envs, "VLLM_RUBIN_CUTEDSL_PREFILL", True)
    warning_once = Mock()
    monkeypatch.setattr(fi_backend.logger, "warning_once", warning_once)
    impl = object.__new__(fi_backend.FlashInferImpl)
    impl.bmm1_scale = impl.bmm2_scale = 1.0
    impl.is_kvcache_nvfp4 = False
    # The gate runs before the KV cache is first touched; stop there.
    kv_cache = Mock()
    type(kv_cache).dtype = PropertyMock(side_effect=_ReachedKVCache)
    prefill_cls = fi_backend.TRTLLMPrefill if trtllm_prefill else fi_backend.FIPrefill
    metadata = fi_backend.FlashInferMetadata(
        kv_cache_layout=KVCacheLayout.LBHNC,
        num_actual_tokens=4,
        slot_mapping=torch.arange(4),
        q_data_type_prefill=torch.float8_e4m3fn,
        q_data_type_decode=torch.float8_e4m3fn,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=4,
        causal=True,
        prefill=Mock(spec=prefill_cls),
        decode=None,
    )
    query = torch.empty(4, 64, 128, dtype=torch.float8_e4m3fn)
    output = torch.empty(4, 64 * 128, dtype=torch.bfloat16)
    with pytest.raises(_ReachedKVCache):
        impl.forward(Mock(), query, None, None, kv_cache, metadata, output)
    assert warning_once.called == (not trtllm_prefill)


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


def _fake_persistent_vit_module(kernel):
    return SimpleNamespace(
        cute_dsl_fmha_vit=kernel,
        cute_dsl_fmha_vit_work_count=lambda lens, heads: (
            heads * sum((n + 255) // 256 for n in lens)
        ),
        RUBIN_VIT_TUNED_POLICY={"persistent": True, "mma_rowsum": True},
        rubin_vit_use_work_queue=lambda capacity: capacity > 8192,
    )


def test_vit_work_counter_counts_q_tiles_of_padded_slots(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "flashinfer.attention.cute_dsl", _fake_persistent_vit_module(None)
    )
    backend = AttentionBackendEnum.FLASHINFER
    # Lengths 300, 0 (empty slot) and 1024 -> 2 + 0 + 4 Q tiles of 256 rows.
    offsets = np.array([0, 300, 300, 1324], dtype=np.int32)
    compute = MMEncoderAttention.maybe_compute_vit_work_counter
    cpu = torch.device("cpu")
    monkeypatch.setattr(envs, "VLLM_RUBIN_CUTEDSL_VIT", True)
    monkeypatch.setattr(envs, "VLLM_RUBIN_CUTEDSL_VIT_PERSISTENT", False)
    assert compute(backend, offsets, 16, cpu) is None
    monkeypatch.setattr(envs, "VLLM_RUBIN_CUTEDSL_VIT_PERSISTENT", True)
    counter = compute(backend, offsets, 16, cpu)
    torch.testing.assert_close(counter, torch.tensor([0, 96], dtype=torch.int32))
    assert compute(AttentionBackendEnum.TORCH_SDPA, offsets, 16, cpu) is None
    monkeypatch.setattr(envs, "VLLM_RUBIN_CUTEDSL_VIT", False)
    assert compute(backend, offsets, 16, cpu) is None


@pytest.mark.parametrize(
    "capacity,with_counter,dynamic",
    [(8192, True, False), (8193, True, True), (8193, False, False)],
)
def test_vit_persistent_policy_and_work_queue_choice(
    monkeypatch, capacity, with_counter, dynamic
):
    kernel = Mock()
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.attention.cute_dsl",
        _fake_persistent_vit_module(kernel),
    )
    q = torch.empty(capacity, 16, 80, dtype=torch.float8_e4m3fn)
    indptr = torch.tensor([0, 72, 190, 190], dtype=torch.int32)
    counter = torch.tensor([0, 32], dtype=torch.int32) if with_counter else None
    result = vit_attn_wrappers.rubin_cutedsl_vit_wrapper(
        q, q, q, indptr, torch.tensor(capacity), 72, 0.125, 2.0, 3.0, 4.0,
        True, counter,
    )  # fmt: skip
    args, kwargs = kernel.call_args
    # The graph capacity, not the longest sequence, is the launch's max_seqlen.
    assert args[5] == capacity
    assert kwargs["persistent"] is True
    assert kwargs["mma_rowsum"] is True
    assert kwargs.get("work_counter") is (counter if dynamic else None)
    # The persistent kernel writes compact logical-width (D72) rows.
    assert args[3].shape == (capacity, 16, 72)
    assert result.shape == (capacity, 16, 72)
    assert result.dtype == torch.bfloat16


def test_vit_default_path_stays_nonpersistent(monkeypatch):
    kernel = Mock()
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.attention.cute_dsl",
        _fake_persistent_vit_module(kernel),
    )
    q = torch.empty(9000, 16, 80, dtype=torch.float8_e4m3fn)
    indptr = torch.tensor([0, 72], dtype=torch.int32)
    counter = torch.tensor([0, 16], dtype=torch.int32)
    vit_attn_wrappers.rubin_cutedsl_vit_wrapper(
        q, q, q, indptr, torch.tensor(9000), 72, 0.125, 1.0, 1.0, 1.0,
        False, counter,
    )  # fmt: skip
    _, kwargs = kernel.call_args
    assert "persistent" not in kwargs
    assert "work_counter" not in kwargs
    assert kernel.call_args.args[3].shape == q.shape
    for persistent, width in ((False, 80), (True, 72)):
        fake = vit_attn_wrappers.rubin_cutedsl_vit_wrapper_fake(
            q, q, q, indptr, torch.tensor(9000), 72, 0.125, 1.0, 1.0, 1.0,
            persistent, None,
        )  # fmt: skip
        assert fake.shape == (9000, 16, width)
        assert fake.dtype == torch.bfloat16
