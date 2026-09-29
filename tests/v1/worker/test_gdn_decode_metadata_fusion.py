# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused GDN decode metadata path must leave every GDN builder in the same
state as the per-builder GDNAttentionMetadataBuilder.build() path: identical
metadata and bit-identical persistent buffers, including stale rows, across a
sequence of decode, spec-decode, padded and mixed steps.
"""

from dataclasses import dataclass, fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
    GDNFusedDecodeStep,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.utils import AttentionGroup

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Needs CUDA")

MAMBA_BLOCK_SIZE = 16
MAX_NUM_SEQS = 8
MAX_NUM_REQS = 16
BLOCK_TABLE_WIDTH = 64
# Group 0 stands for the full-attention group; groups 1..3 hold GDN layers.
NUM_KV_CACHE_GROUPS = 4
PERSISTENT_BUFFERS = (
    "spec_state_indices_tensor",
    "non_spec_state_indices_tensor",
    "spec_sequence_masks",
    "spec_token_indx",
    "non_spec_token_indx",
    "spec_query_start_loc",
    "non_spec_query_start_loc",
    "num_accepted_tokens",
)


@dataclass
class Step:
    """One batch: per real request (query_len, num_drafts, seq_len, prefilling)."""

    reqs: list[tuple[int, int, int, bool]]
    num_reqs_padded: int
    full_cudagraph: bool


def _decode(num_drafts: int, seq_len: int) -> tuple[int, int, int, bool]:
    return (num_drafts + 1, num_drafts, seq_len, False)


SPEC_STEPS = [
    # Uniform spec decode, no request padding.
    Step([_decode(3, 45), _decode(3, 16), _decode(3, 17), _decode(3, 300)], 4, True),
    # Uniform spec decode padded to a larger captured batch.
    Step([_decode(3, 33 + 7 * i) for i in range(5)], 8, True),
    # Varying draft counts: PIECEWISE, unpadded, still the persistent path.
    Step([_decode(3, 70), _decode(1, 64), _decode(2, 5), _decode(3, 129)], 4, False),
    # Chunked prefill tail after spec decodes: build()'s own path.
    Step([_decode(3, 90), _decode(3, 91), (7, 0, 200, True)], 3, False),
    # A non-spec decode among spec decodes is reclassified as a prefill.
    Step([_decode(3, 48), (1, 0, 49, False), _decode(2, 50)], 3, False),
    # Spec decode after the mixed steps, with padding again.
    Step([_decode(3, 17 + 16 * i) for i in range(3)], 4, True),
    # More spec tokens than decode_cudagraph_max_bs: no persistent buffers.
    Step([_decode(3, 20 + i) for i in range(9)], 9, False),
    # Decode without drafts (non-spec path), padded.
    Step([(1, 0, 40 + 9 * i, False) for i in range(6)], 8, True),
    Step([_decode(3, 100), _decode(3, 101)], 2, True),
]
NON_SPEC_STEPS = [
    Step([(1, 0, 40 + 9 * i, False) for i in range(6)], 8, True),
    Step([(1, 0, 33, False), (1, 0, 17, False), (5, 0, 64, True)], 3, False),
    Step([(1, 0, 16 * (i + 1), False) for i in range(3)], 3, True),
    Step([(1, 0, 16 * (i + 1) + 1, False) for i in range(7)], 8, True),
]


def _make_state(num_spec: int, mamba_cache_mode: str, seed: int):
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=MAMBA_BLOCK_SIZE,
        max_num_seqs=MAX_NUM_SEQS,
    )
    if num_spec:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=num_spec
        )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    device = torch.device("cuda")
    spec = MambaSpec(
        block_size=MAMBA_BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_spec,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    attn_groups: list[list[AttentionGroup]] = [[]]
    for group_id in range(1, NUM_KV_CACHE_GROUPS):
        layer = f"layers.{group_id}.linear_attn"
        builder = GDNAttentionMetadataBuilder(spec, [layer], vllm_config, device)
        # Stale contents must survive exactly where build() leaves them.
        for name in PERSISTENT_BUFFERS:
            buf = getattr(builder, name)
            if buf.dtype == torch.bool:
                buf.copy_(
                    torch.rand(buf.shape, generator=generator, device=device) < 0.5
                )
            else:
                buf.copy_(
                    torch.randint(
                        -9, 9999, buf.shape, generator=generator, device=device
                    )
                )
        attn_groups.append(
            [AttentionGroup(GDNAttentionBackend, [layer], spec, group_id, [builder])]
        )

    state = object.__new__(MambaHybridModelState)
    state.vllm_config = vllm_config
    state.device = device
    state.max_model_len = vllm_config.model_config.max_model_len
    # The model state's align context serves only KDA builders.
    state._align_mode = False
    state.recoverssm = None
    state._gdn_decode_fusions = {}
    state.num_accepted_tokens_gpu = torch.randint(
        1, num_spec + 2, (MAX_NUM_REQS,), generator=generator, device=device
    ).to(torch.int32)
    return state, attn_groups


def _make_inputs(step: Step, rng: np.random.Generator, device: torch.device):
    num_reqs = len(step.reqs)
    padded = step.num_reqs_padded
    query_lens = np.array([r[0] for r in step.reqs], dtype=np.int32)
    num_drafts = np.array([r[1] for r in step.reqs], dtype=np.int32)
    seq_lens = np.array([r[2] for r in step.reqs], dtype=np.int32)
    num_tokens = int(query_lens.sum())

    query_start_loc_np = np.full(padded + 1, num_tokens, dtype=np.int32)
    query_start_loc_np[0] = 0
    np.cumsum(query_lens, out=query_start_loc_np[1 : num_reqs + 1])
    query_start_loc = torch.full(
        (MAX_NUM_REQS + 1,), num_tokens, dtype=torch.int32, device=device
    )
    query_start_loc[: padded + 1] = torch.from_numpy(query_start_loc_np)
    seq_lens_gpu = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=device)
    seq_lens_gpu[:num_reqs] = torch.from_numpy(seq_lens)

    block_tables = []
    for _ in range(NUM_KV_CACHE_GROUPS):
        table = torch.from_numpy(
            rng.integers(1, 1 << 20, (MAX_NUM_REQS, BLOCK_TABLE_WIDTH), dtype=np.int32)
        ).to(device)
        # gather_block_tables zeroes the padded rows.
        table[num_reqs:padded] = 0
        block_tables.append(table[:padded])

    num_draft_tokens_per_req = num_drafts if num_drafts.any() else None
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_reqs_after_padding=padded,
        num_tokens=num_tokens,
        num_tokens_after_padding=max(num_tokens, padded * int(query_lens.max())),
        idx_mapping=torch.from_numpy(
            rng.permutation(MAX_NUM_REQS)[:num_reqs].astype(np.int64)
        ).to(device),
        query_start_loc=query_start_loc[: padded + 1],
        query_start_loc_np=query_start_loc_np,
        num_scheduled_tokens=query_lens,
        num_draft_tokens_per_req=num_draft_tokens_per_req,
        seq_lens=seq_lens_gpu[:padded],
        seq_lens_cpu_upper_bound=torch.from_numpy(
            np.pad(seq_lens, (0, padded - num_reqs))
        ),
        is_prefilling_np=np.array([r[3] for r in step.reqs], dtype=np.bool_),
        dcp_local_seq_lens=None,
        positions=torch.zeros(num_tokens, dtype=torch.int64, device=device),
        prompt_lens=None,
    )
    return input_batch, tuple(block_tables)


def _prepare_attn(state, attn_groups, step, input_batch, block_tables):
    return state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.FULL
        if step.full_cudagraph
        else CUDAGraphMode.NONE,
        block_tables=block_tables,
        slot_mappings=torch.zeros(
            NUM_KV_CACHE_GROUPS, 1, dtype=torch.int64, device=block_tables[0].device
        ),
        attn_groups=attn_groups,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[None] * NUM_KV_CACHE_GROUPS),
    )


def _buffer_view(builder, t: torch.Tensor) -> tuple[str, int] | None:
    """(persistent buffer name, element offset) if t is a view of one."""
    for name in PERSISTENT_BUFFERS:
        buf = getattr(builder, name)
        if t.untyped_storage().data_ptr() == buf.untyped_storage().data_ptr():
            return name, t.storage_offset()
    return None


def _assert_same_metadata(
    ref: GDNAttentionMetadata,
    ref_builder: GDNAttentionMetadataBuilder,
    new: GDNAttentionMetadata,
    new_builder: GDNAttentionMetadataBuilder,
) -> None:
    for field in fields(GDNAttentionMetadata):
        a, b = getattr(ref, field.name), getattr(new, field.name)
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor), field.name
            assert (a.shape, a.stride(), a.dtype) == (b.shape, b.stride(), b.dtype)
            assert torch.equal(a.cpu(), b.cpu()), field.name
            assert _buffer_view(ref_builder, a) == _buffer_view(new_builder, b), (
                field.name
            )
        elif field.name == "nums_dict":
            assert (a is None) == (b is None)
        else:
            assert a == b, field.name


@pytest.mark.parametrize(
    ("num_spec", "mamba_cache_mode", "steps"),
    [
        (3, "align", SPEC_STEPS),
        (3, "none", SPEC_STEPS),
        (0, "align", NON_SPEC_STEPS),
    ],
)
def test_fused_gdn_decode_metadata_matches_per_builder_build(
    num_spec: int,
    mamba_cache_mode: str,
    steps: list[Step],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref_state, ref_groups = _make_state(num_spec, mamba_cache_mode, seed=0)
    new_state, new_groups = _make_state(num_spec, mamba_cache_mode, seed=0)
    rng = np.random.default_rng(0)
    device = torch.device("cuda")

    fused_writes: list[bool] = []
    write = GDNFusedDecodeStep._write_decode_buffers

    def record_write(self, *args):
        fused_writes.append(write(self, *args))
        return fused_writes[-1]

    monkeypatch.setattr(GDNFusedDecodeStep, "_write_decode_buffers", record_write)
    num_fused_steps = 0
    for step in steps:
        input_batch, block_tables = _make_inputs(step, rng, device)
        with monkeypatch.context() as m:
            m.setattr(mamba_hybrid.GDNDecodeMetadataFusion, "create", lambda *_: None)
            ref = _prepare_attn(ref_state, ref_groups, step, input_batch, block_tables)
        fused_writes.clear()
        new = _prepare_attn(new_state, new_groups, step, input_batch, block_tables)
        torch.accelerator.synchronize()

        assert ref.keys() == new.keys()
        for ref_group, new_group in zip(ref_groups[1:], new_groups[1:]):
            ref_builder = ref_group[0].get_metadata_builder()
            new_builder = new_group[0].get_metadata_builder()
            layer = ref_group[0].layer_names[0]
            _assert_same_metadata(ref[layer], ref_builder, new[layer], new_builder)
            for name in PERSISTENT_BUFFERS:
                assert torch.equal(
                    getattr(ref_builder, name), getattr(new_builder, name)
                ), name

        # One fused write attempt per step, covering exactly the batches whose
        # build() writes the persistent buffers.
        ref_builder = ref_groups[1][0].get_metadata_builder()
        ref_meta = ref[ref_groups[1][0].layer_names[0]]
        state_indices = (
            ref_meta.spec_state_indices_tensor
            if ref_meta.num_spec_decodes
            else ref_meta.non_spec_state_indices_tensor
        )
        assert state_indices is not None
        uses_buffers = _buffer_view(ref_builder, state_indices) is not None
        assert fused_writes == [uses_buffers]
        num_fused_steps += uses_buffers
    # The sequences cover both the fused path and build()'s own path.
    assert 0 < num_fused_steps < len(steps)
