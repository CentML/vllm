# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer metadata reuse across fused autoregressive draft decode steps.

The fused draft loop builds the attention metadata once for draft step 1 and
records all remaining steps in one CUDA graph, advancing positions, sequence
lengths and slot mappings in place between steps. These tests check that this
matches rebuilding the metadata before every step, as the per-step loop does.
"""

import dataclasses
from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import set_current_vllm_config
from vllm.utils.flashinfer import supports_trtllm_attention
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends import flashinfer as flashinfer_backend
from vllm.v1.attention.backends.flashinfer import (
    FlashInferDecodeKernel,
    FlashInferImpl,
    FlashInferMetadataBuilder,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID, PerLayerParameters
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

MODEL = "Qwen/Qwen3-0.6B"
# Per-rank attention shape of the Qwen3.6-35B-A3B MTP layer at TP=1.
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 256
BLOCK_SIZE = 16
NUM_BLOCKS = 128
LAYER_NAME = "draft.layers.0.self_attn.attn"
# Tokens already in the KV cache per request before draft step 1. 15 moves the
# second draft token into a new block.
CONTEXT_LENS = [15, 40, 300]


def _make_config(kv_cache_dtype: str):
    vllm_config = create_vllm_config(
        model_name=MODEL,
        max_model_len=1024,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=NUM_BLOCKS,
        hf_config_override={
            "num_attention_heads": NUM_Q_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "head_dim": HEAD_SIZE,
        },
    )
    vllm_config.cache_config.cache_dtype = kv_cache_dtype
    # HND, as required by trtllm-gen.
    vllm_config.cache_config.kv_cache_layout = "LBHNC"
    kv_cache_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.uint8 if kv_cache_dtype == "fp8" else torch.bfloat16,
        kv_quant_mode=(
            KVQuantMode.FP8_PER_TENSOR if kv_cache_dtype == "fp8" else KVQuantMode.NONE
        ),
    )
    return vllm_config, kv_cache_spec


def _per_layer_parameters(vllm_config, layer_names, impl_cls):
    return {
        name: PerLayerParameters(
            window_left=-1,
            logits_soft_cap=0.0,
            sm_scale=HEAD_SIZE**-0.5,
            has_sinks=False,
        )
        for name in layer_names
    }


class _DraftBatch:
    """Persistent draft decode buffers, laid out as the speculator keeps them.

    Padded requests have an empty query, sequence length 0 and PAD_SLOT_ID.
    """

    def __init__(self, num_reqs_padded: int, device: torch.device):
        num_reqs = len(CONTEXT_LENS)
        self.num_reqs = num_reqs
        self.num_reqs_padded = num_reqs_padded
        max_blocks = NUM_BLOCKS // num_reqs_padded
        self.block_table = (
            torch.randperm(NUM_BLOCKS, dtype=torch.int32)[
                : num_reqs_padded * max_blocks
            ]
            .view(num_reqs_padded, max_blocks)
            .to(device)
        )
        self.context_lens = torch.tensor(CONTEXT_LENS, dtype=torch.int64, device=device)
        self.positions = torch.zeros(num_reqs_padded, dtype=torch.int64, device=device)
        self.seq_lens = torch.zeros(num_reqs_padded, dtype=torch.int32, device=device)
        self.slot_mapping = torch.full(
            (num_reqs_padded,), PAD_SLOT_ID, dtype=torch.int64, device=device
        )
        qsl = list(range(num_reqs + 1)) + [num_reqs] * (num_reqs_padded - num_reqs)
        self.query_start_loc_cpu = torch.tensor(qsl, dtype=torch.int32)
        self.query_start_loc = self.query_start_loc_cpu.to(device)
        # Constant across draft steps, like the speculator's draft_max_seq_len.
        self.max_seq_len = max(CONTEXT_LENS) + 3

    def reset_to_step_1(self) -> None:
        n = self.num_reqs
        self.positions[:n].copy_(self.context_lens)
        self.seq_lens[:n].copy_(self.context_lens + 1)
        self._compute_slot_mapping()

    def advance(self) -> None:
        """Move to the next draft step on the GPU, capture-safe."""
        n = self.num_reqs
        self.positions[:n] += 1
        self.seq_lens[:n] += 1
        self._compute_slot_mapping()

    def _compute_slot_mapping(self) -> None:
        n = self.num_reqs
        positions = self.positions[:n]
        blocks = self.block_table[:n].gather(1, (positions // BLOCK_SIZE)[:, None])
        self.slot_mapping[:n].copy_(blocks[:, 0] * BLOCK_SIZE + positions % BLOCK_SIZE)

    def common_attn_metadata(self, step: int) -> CommonAttentionMetadata:
        n = self.num_reqs_padded
        seq_lens_upper_bound = torch.zeros(n, dtype=torch.int32)
        seq_lens_upper_bound[: self.num_reqs] = (
            torch.tensor(CONTEXT_LENS, dtype=torch.int32) + step
        )
        return CommonAttentionMetadata(
            query_start_loc=self.query_start_loc,
            query_start_loc_cpu=self.query_start_loc_cpu,
            seq_lens=self.seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_upper_bound,
            num_reqs=n,
            # FULL CUDA graphs attend over the padded token count.
            num_actual_tokens=n,
            max_query_len=1,
            max_seq_len=self.max_seq_len,
            block_table_tensor=self.block_table,
            slot_mapping=self.slot_mapping,
            causal=True,
        )


def _assert_same_metadata(actual, expected, path="metadata") -> None:
    """Equal host fields and identical device storage for tensor fields."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert (
            actual.data_ptr(),
            actual.shape,
            actual.stride(),
            actual.dtype,
        ) == (
            expected.data_ptr(),
            expected.shape,
            expected.stride(),
            expected.dtype,
        ), path
    elif dataclasses.is_dataclass(expected):
        assert type(actual) is type(expected), path
        for field in dataclasses.fields(expected):
            _assert_same_metadata(
                getattr(actual, field.name),
                getattr(expected, field.name),
                f"{path}.{field.name}",
            )
    else:
        assert actual == expected, path


@pytest.mark.parametrize(
    "kernel", [FlashInferDecodeKernel.XQA, FlashInferDecodeKernel.TRTLLM_GEN]
)
@pytest.mark.parametrize("num_reqs_padded", [3, 4])
def test_draft_decode_metadata_after_update_matches_fresh_build(
    monkeypatch, kernel, num_reqs_padded
):
    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        FlashInferMetadataBuilder,
        "_get_flashinfer_trtllm_api_decode_kernel",
        staticmethod(lambda: kernel),
    )
    monkeypatch.setattr(
        flashinfer_backend, "get_per_layer_parameters", _per_layer_parameters
    )
    vllm_config, kv_cache_spec = _make_config("fp8")
    device = torch.device("cpu")
    with set_current_vllm_config(vllm_config):
        builder = FlashInferMetadataBuilder(
            kv_cache_spec, [LAYER_NAME], vllm_config, device
        )
    assert builder.flashinfer_trtllm_api_decode_kernel == kernel
    assert builder.supports_draft_decode_metadata_update

    batch = _DraftBatch(num_reqs_padded, device)
    batch.reset_to_step_1()
    metadata = builder.build(0, batch.common_attn_metadata(step=1))

    for step in (2, 3):
        batch.advance()
        builder.update_draft_decode_metadata(metadata)
        fresh = builder.build(0, batch.common_attn_metadata(step=step))
        _assert_same_metadata(metadata, fresh)


@pytest.mark.parametrize(
    ("trtllm_decode", "dcp_world_size", "expected"),
    [
        (True, 1, True),
        # Native decode plans on the host from the sequence lengths.
        (False, 1, False),
        # DCP decodes over rank-local lengths the draft loop does not advance.
        (True, 2, False),
    ],
)
def test_draft_decode_metadata_update_support(
    monkeypatch, trtllm_decode, dcp_world_size, expected
):
    monkeypatch.setattr(
        flashinfer_backend,
        "can_use_trtllm_attention",
        lambda *args, **kwargs: trtllm_decode,
    )
    monkeypatch.setattr(
        FlashInferMetadataBuilder,
        "_get_flashinfer_trtllm_api_decode_kernel",
        staticmethod(lambda: FlashInferDecodeKernel.TRTLLM_GEN),
    )
    monkeypatch.setattr(
        flashinfer_backend, "get_per_layer_parameters", _per_layer_parameters
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=dcp_world_size, rank_in_group=0),
    )
    vllm_config, kv_cache_spec = _make_config("fp8")
    with set_current_vllm_config(vllm_config):
        builder = FlashInferMetadataBuilder(
            kv_cache_spec, [LAYER_NAME], vllm_config, torch.device("cpu")
        )
    assert builder.supports_draft_decode_metadata_update is expected


class _Layer:
    def __init__(self, device: torch.device):
        one = torch.tensor(1.0, dtype=torch.float32, device=device)
        self._q_scale = one
        self._k_scale = one
        self._v_scale = one
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0
        self._o_scale_float = None


@pytest.mark.skipif(
    not supports_trtllm_attention(is_prefill=False),
    reason="Needs the FlashInfer XQA or trtllm-gen decode kernel.",
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("num_reqs_padded", [3, 4])
@torch.inference_mode()
def test_fused_draft_decode_graph_matches_per_step_builds(
    monkeypatch, kv_cache_dtype, num_reqs_padded
):
    """Two draft steps in one CUDA graph with the step-1 metadata produce the
    same attention output and KV cache, bit for bit, as rebuilding the metadata
    before every step.
    """
    monkeypatch.setattr(
        flashinfer_backend, "get_per_layer_parameters", _per_layer_parameters
    )
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    vllm_config, kv_cache_spec = _make_config(kv_cache_dtype)
    dtype = torch.bfloat16
    num_steps = 2

    with set_current_vllm_config(vllm_config):
        builder = FlashInferMetadataBuilder(
            kv_cache_spec, [LAYER_NAME], vllm_config, device
        )
        impl = FlashInferImpl(
            num_heads=NUM_Q_HEADS,
            head_size=HEAD_SIZE,
            scale=HEAD_SIZE**-0.5,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
        )
    assert builder.supports_draft_decode_metadata_update
    layer = _Layer(device)

    cache_shape = (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE)
    initial_kv_cache = torch.randn(cache_shape, device=device)
    if kv_cache_dtype == "fp8":
        initial_kv_cache = initial_kv_cache.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        initial_kv_cache = initial_kv_cache.to(dtype)
    kv_cache = initial_kv_cache.clone()

    n = num_reqs_padded
    query = torch.randn(
        num_steps, n, NUM_Q_HEADS, HEAD_SIZE, dtype=dtype, device=device
    )
    key = torch.randn(num_steps, n, NUM_KV_HEADS, HEAD_SIZE, dtype=dtype, device=device)
    value = torch.randn_like(key)
    batch = _DraftBatch(num_reqs_padded, device)

    def draft_step(metadata, step: int, out: torch.Tensor) -> None:
        impl.do_kv_cache_update(
            layer, key[step], value[step], kv_cache, metadata.slot_mapping
        )
        impl.forward(
            layer, query[step], key[step], value[step], kv_cache, metadata, output=out
        )

    # Per-step loop: rebuild the metadata before every draft step.
    expected = torch.zeros(
        num_steps, n, NUM_Q_HEADS, HEAD_SIZE, dtype=dtype, device=device
    )
    batch.reset_to_step_1()
    for step in range(num_steps):
        if step > 0:
            batch.advance()
        metadata = builder.build(0, batch.common_attn_metadata(step=step + 1))
        draft_step(metadata, step, expected[step])
    expected_kv_cache = kv_cache.clone()

    # Fused loop: one build for draft step 1, then all steps in one graph.
    actual = torch.zeros_like(expected)

    def fused_steps(metadata) -> None:
        for step in range(num_steps):
            if step > 0:
                batch.advance()
                builder.update_draft_decode_metadata(metadata)
            draft_step(metadata, step, actual[step])

    kv_cache.copy_(initial_kv_cache)
    batch.reset_to_step_1()
    metadata = builder.build(0, batch.common_attn_metadata(step=1))
    # Warm up outside the graph, then restore the step-1 state.
    fused_steps(metadata)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_steps(metadata)
    kv_cache.copy_(initial_kv_cache)
    batch.reset_to_step_1()
    actual.zero_()
    graph.replay()
    torch.accelerator.synchronize()

    assert torch.equal(actual[:, : batch.num_reqs], expected[:, : batch.num_reqs])
    assert torch.equal(kv_cache, expected_kv_cache)
