# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Internal prefill checkpoints of ``QwenGatedDeltaNetAttention._forward_core``.

With ``num_prefill_checkpoint_blocks`` set (align-mode prefix caching), one
forward over a prefill chunk must leave, for every sequence the positional rule
selects, the conv/ssm state at its checkpoint in the checkpoint block (column
``cdiv(seq_len, B) - 2``) and still produce the outputs and final states of the
plain forward. The reference runs the same sequences the way the scheduler
splits them without checkpoints: a chunk ending at the checkpoint, then a
chunk with the rest.

Both runs go through the real metadata builder and ``_forward_core``.
"""

from __future__ import annotations

import dataclasses
import types
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="GDN kernels need CUDA", allow_module_level=True)

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.utils.math_utils import cdiv  # noqa: E402
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
)

H = 4  # num key heads
HV = 8  # num value heads
K = 128  # head_k_dim
V = 128  # head_v_dim
CONV_KERNEL = 4
CONV_DIM = 2 * H * K + HV * V
BLOCK = 256  # mamba block size
UNIT = 32  # prefix_match_unit
PREFIX = "model.layers.0.linear_attn"

# (query_start, seq_len) of each prefill, against BLOCK = 256 and UNIT = 32:
PREFILLS = [
    (288, 600),  # starts mid-block, ends one block later: checkpoint in the
    #               initial-state block
    (0, 700),  # fresh prompt: checkpoint in a new block
    (544, 700),  # stays inside one block: no checkpoint
    (256, 1000),  # block-aligned start: checkpoint in a new block
    (512, 700),  # block-aligned start, one block later: no checkpoint
]
CHECKPOINTED = (0, 1, 3)
# One checkpoint (prefill 1, after an unsplit prefill): the tail is a token
# range, rerun on slices instead of gathered rows.
LAYOUTS = {
    "three": (PREFILLS, CHECKPOINTED),
    "one": ([PREFILLS[2], PREFILLS[3], PREFILLS[4]], (1,)),
    # Aligned private-initial alias between cold one-block rows. The cold rows
    # must never acquire a negative checkpoint column under copy capability.
    "aligned": ([(0, 200), (512, 700), (0, 128)], (1,)),
    # Eagle's desired tail is already the initial state; export boundary512
    # instead, with the original tail retained by the manager's immutable copy.
    "initial_tail": ([(0, 200), (480, 530), (0, 128)], (1,)),
}


def _make_vllm_config(backend: str):
    cfg = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK,
        max_model_len=4096,
        hf_config_override={"linear_key_head_dim": K},
    )
    cfg.additional_config = {"gdn_prefill_backend": backend}
    cfg.cache_config.mamba_cache_mode = "align"
    cfg.cache_config.prefix_match_unit = UNIT
    return cfg


def _make_builder(
    vllm_config, checkpoints: bool, drop_eagle: bool, device, copy_initial: bool = False
):
    spec = MambaSpec(
        block_size=BLOCK,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=int(checkpoints),
        prefill_checkpoint_alignment=1 if checkpoints else None,
        prefill_checkpoint_reuses_initial_block=checkpoints,
        prefill_checkpoint_copies_initial_block=checkpoints and copy_initial,
    )
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=spec,  # type: ignore[arg-type]
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    builder.checkpoint_drop_eagle_block = drop_eagle
    return builder


def _build_layer(vllm_config, conv_state, ssm_state, weights):
    """A minimal object that runs the real ``_forward_core`` bound to it."""
    A_log, dt_bias, conv_weight, conv_bias = weights
    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.enable_packed_recurrent_decode = False
    layer.tp_size = 1
    layer.num_k_heads = H
    layer.num_v_heads = HV
    layer.head_k_dim = K
    layer.head_v_dim = V
    layer.key_dim = H * K
    layer.value_dim = HV * V
    layer.activation = "silu"
    layer.A_log = A_log
    layer.dt_bias = dt_bias
    layer.conv1d = types.SimpleNamespace(weight=conv_weight, bias=conv_bias)
    layer.kv_cache = (conv_state, ssm_state)
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
        "_run_prefill_chunk",
        "_chunk_prefill_with_checkpoint",
    ):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    layer._store_conv_checkpoint = QwenGatedDeltaNetAttention._store_conv_checkpoint
    return layer


def _forward(vllm_config, builder, block_table, segments, pools, weights, x):
    """Run one step over ``segments`` [(row, query_start, seq_len)] (rows are
    batch rows of ``block_table``) and return (metadata, per-row outputs).
    """
    mixed_qkv, a, b = x
    seq_lens = [end for _, _, end in segments]
    query_lens = [end - start for _, start, end in segments]
    device = block_table.device
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens), BLOCK, device
    )
    common = dataclasses.replace(
        common, block_table_tensor=block_table[[r for r, _, _ in segments]]
    )
    with set_current_vllm_config(vllm_config):
        meta = builder.build(common_prefix_len=0, common_attn_metadata=common)
    tokens = [torch.arange(s, e, device=device) for _, s, e in segments]
    inputs = [
        torch.cat([t[r][tok] for (r, _, _), tok in zip(segments, tokens)])
        for t in (mixed_qkv, a, b)
    ]
    num_tokens = sum(query_lens)
    out = torch.zeros(num_tokens, HV, V, dtype=torch.bfloat16, device=device)
    conv_state, ssm_state = pools
    layer = _build_layer(vllm_config, conv_state, ssm_state, weights)
    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})
    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        layer._forward_core(
            mixed_qkv=inputs[0], b=inputs[2], a=inputs[1], core_attn_out=out
        )
    per_row = {}
    offset = 0
    for (row, _, _), q in zip(segments, query_lens):
        per_row[row] = out[offset : offset + q]
        offset += q
    return meta, per_row


def _slot(block_table, row, num_tokens):
    return int(block_table[row, (num_tokens - 1) // BLOCK])


def _conv_window(conv_pool: torch.Tensor) -> torch.Tensor:
    """The ``[..., dim, width - 1]`` part of the conv pool that the prefill and
    decode conv kernels read as the initial state (with MTP the pool rows are
    ``width - 1 + num_spec`` long; the extra columns are spec-decode scratch).
    """
    view = conv_pool if is_conv_state_dim_first() else conv_pool.transpose(-1, -2)
    return view[..., : CONV_KERNEL - 1]


@pytest.mark.parametrize("layout", ["three", "one", "aligned", "initial_tail"])
@pytest.mark.parametrize("num_spec", [0, 3])
@pytest.mark.parametrize("drop_eagle", [False, True])
@pytest.mark.parametrize("num_decodes", [0, 2])
@pytest.mark.parametrize("backend", ["auto", "triton"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_prefill_checkpoint_matches_split_prefill(
    drop_eagle: bool,
    num_decodes: int,
    backend: str,
    num_spec: int,
    layout: str,
    state_dtype: torch.dtype,
) -> None:
    """With a bf16 pool the checkpoint holds the state rounded to bf16 and the
    tail resumes from it, exactly as the second chunk of the split prefill
    (and a later prefix hit on the checkpoint) does.
    """
    if backend == "triton" and current_platform.is_device_capability_family(100):
        pytest.skip("The Triton/FLA chunk kernel is unsupported on SM10x")
    torch.manual_seed(0)
    device = torch.device("cuda")
    vllm_config = _make_vllm_config(backend)
    ckpt_builder = _make_builder(
        vllm_config,
        True,
        drop_eagle,
        device,
        copy_initial=layout in ("aligned", "initial_tail"),
    )
    ref_builder = _make_builder(vllm_config, False, drop_eagle, device)

    # Decodes (1 token, with context) first, then the prefills.
    prefills, checkpointed = LAYOUTS[layout]
    requests = [(63, 64)] * num_decodes + prefills
    max_blocks = cdiv(max(end for _, end in requests), BLOCK)
    num_rows = len(requests)
    block_table = (
        torch.arange(num_rows * max_blocks, dtype=torch.int32, device=device).view(
            num_rows, max_blocks
        )
        + 1
    )
    pool_size = num_rows * max_blocks + 1
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, CONV_KERNEL, num_spec=num_spec
    )
    conv0 = torch.randn(pool_size, *conv_shape, dtype=torch.bfloat16, device=device)
    ssm0 = torch.randn(pool_size, *ssm_shape, device=device).to(state_dtype)
    conv0 *= 0.05
    ssm0 *= 0.05

    weights = (
        torch.randn(HV, dtype=torch.float32, device=device) * 0.1,
        torch.randn(HV, dtype=torch.float32, device=device) * 0.1,
        torch.randn(CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device)
        * 0.1,
        torch.randn(CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1,
    )
    longest = max(end for _, end in requests)
    x = (
        torch.randn(num_rows, longest, CONV_DIM, dtype=torch.bfloat16, device=device)
        * 0.1,
        torch.randn(num_rows, longest, HV, dtype=torch.bfloat16, device=device) * 0.1,
        torch.randn(num_rows, longest, HV, dtype=torch.bfloat16, device=device) * 0.1,
    )

    positions = {}
    for i in checkpointed:
        start, end = prefills[i]
        positions[num_decodes + i] = get_mamba_prefill_checkpoint_position(
            end,
            UNIT,
            drop_eagle_block=drop_eagle,
            query_start=start,
            mamba_block_size=BLOCK,
            copy_initial_block=(
                ckpt_builder.kv_cache_spec.prefill_checkpoint_copies_initial_block
            ),
        )

    # ---- One forward with internal checkpoints ----
    conv_c, ssm_c = conv0.clone(), ssm0.clone()
    segments = [(r, s, e) for r, (s, e) in enumerate(requests)]
    meta, out_c = _forward(
        vllm_config,
        ckpt_builder,
        block_table,
        segments,
        (conv_c, ssm_c),
        weights,
        x,
    )
    assert meta.prefill_checkpoint is not None
    assert meta.prefill_checkpoint.checkpoint_state_indices.numel() == len(checkpointed)
    assert (meta.prefill_checkpoint.tail_token_range is not None) == (
        len(checkpointed) == 1
    )

    # ---- Reference: split at each checkpoint, two forwards ----
    conv_r, ssm_r = conv0.clone(), ssm0.clone()
    first = [(r, s, positions.get(r, e)) for r, (s, e) in enumerate(requests)]
    for r, s, e in first:
        # Both runs read the initial state from the block of their chunk end.
        if s > 0:
            src = _slot(block_table, r, requests[r][1])
            dst = _slot(block_table, r, e)
            conv_r[dst] = conv0[src]
            ssm_r[dst] = ssm0[src]
    meta_a, out_a = _forward(
        vllm_config,
        ref_builder,
        block_table,
        first,
        (conv_r, ssm_r),
        weights,
        x,
    )
    assert meta_a.prefill_checkpoint is None
    # States at the checkpoints, before the second chunk moves on.
    ckpt_conv = {}
    ckpt_ssm = {}
    for r, pos in positions.items():
        slot = _slot(block_table, r, pos)
        ckpt_conv[r] = conv_r[slot].clone()
        ckpt_ssm[r] = ssm_r[slot].clone()
        # The align pre-copy moves the state into the next chunk's block.
        dst = _slot(block_table, r, requests[r][1])
        if dst != slot:
            conv_r[dst] = conv_r[slot]
            ssm_r[dst] = ssm_r[slot]
    second = [(r, pos, requests[r][1]) for r, pos in positions.items()]
    _, out_b = _forward(
        vllm_config,
        ref_builder,
        block_table,
        second,
        (conv_r, ssm_r),
        weights,
        x,
    )

    # The Triton/FLA chunk kernel treats every varlen sequence the same way
    # whatever the batch, so the split is bitwise. FlashInfer picks its kernel
    # by batch shape; compare by relative L2 error there (a checkpoint taken at
    # the wrong position is off by O(1)).
    exact = backend == "triton"

    def check(actual: torch.Tensor, expected: torch.Tensor) -> None:
        if exact:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            return
        err = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert err < 1e-2, err

    for r, (s, e) in enumerate(requests):
        running = _slot(block_table, r, e)
        if r in positions:
            ref_out = torch.cat([out_a[r], out_b[r]])
            ckpt_slot = int(block_table[r, cdiv(e, BLOCK) - 2])
            assert ckpt_slot != running
            # Conv checkpoints are copies of the conv inputs: always exact.
            torch.testing.assert_close(
                _conv_window(conv_c[ckpt_slot]),
                _conv_window(ckpt_conv[r]),
                atol=0,
                rtol=0,
            )
            check(ssm_c[ckpt_slot], ckpt_ssm[r])
        else:
            ref_out = out_a[r]
        check(out_c[r], ref_out)
        torch.testing.assert_close(conv_c[running], conv_r[running], atol=0, rtol=0)
        check(ssm_c[running], ssm_r[running])
