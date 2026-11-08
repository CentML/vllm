# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer: the wiring of ``kda_prefill_conv=planar``.

tests/models/glm5next/test_kda_prefill_conv.py covers the kernel alone. Here
the layer's ``_forward`` runs one mixed chunked-prefill step (decode rows next
to a fresh and a continued prompt) with the stock conv and with the planar
conv on cloned caches, through the Triton chunk path and, where it is
available, FlashKDA; outputs and conv states must agree bit for bit. The
dispatch guard must send a short step to the stock kernel, and the option
resolver must reject what it should.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

D = 128
HEADS = 2
LOWER_BOUND = -5.0
# Decode rows first, then one fresh and one continued prompt. 8.3K prefill
# tokens clear the dispatch guard's bar at 2 heads (8192 + 6 per decode row).
LONG_STEP = ([1, 1, 1, 5000, 3300], [True, True, True, False, True], [1, 2, 3, 4, 5])
SHORT_STEP = ([1, 1, 300, 200], [True, True, False, True], [1, 2, 3, 4])


def _make_layer(kda, backend: str, dim_first: bool, device, monkeypatch):
    """A layer assembled without __init__, as test_kda_recurrent.py does."""
    from vllm.v1.worker.workspace import WorkspaceManager

    width = 3 * HEADS * D
    layer = object.__new__(kda.Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer.0"
    layer.local_num_heads = HEADS
    layer.head_dim = D
    layer.conv_size = 4
    layer.local_projection_size = HEADS * D
    layer.kda_safe_gate = True
    layer.kda_lower_bound = LOWER_BOUND
    layer.kda_prefill_backend = backend
    layer._conv_state_dim_first = dim_first
    layer.conv1d = SimpleNamespace(
        weight=torch.randn(width, 1, 4, device=device) * 0.1, bias=None
    )
    layer.A_log = torch.randn(1, 1, HEADS, 1, device=device) * 0.1
    layer.dt_bias = torch.randn(HEADS * D, device=device) * 0.1
    layer._flashkda_buffer_specs = None
    layer._checkpoint_exporter = None
    if backend == "flashkda":
        from vllm.model_executor.layers.mamba.kda_checkpoint import (
            FlashKDAPrefillCheckpointExporter,
        )

        layer._checkpoint_exporter = FlashKDAPrefillCheckpointExporter(state_len=3)
        monkeypatch.setattr(
            kda, "current_workspace_manager", lambda: WorkspaceManager(device)
        )
    return layer


def _flashkda_buffer_specs(step):
    """The layer's FlashKDA workspace buffers, sized for one step as __init__
    would size them for the whole batch."""
    num_tokens, num_seqs = sum(step[0]), len(step[0])
    state_shape = (num_seqs, HEADS, D, D)
    return (
        (state_shape, torch.float32),
        (state_shape, torch.float32),
        (
            (torch.ops._flashkda_C.get_workspace_size(num_tokens, HEADS, num_seqs),),
            torch.uint8,
        ),
        ((1, num_tokens, HEADS, D), torch.bfloat16),
    )


def _metadata(step, device):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    lens, has_init, slots = step
    cu_cpu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()]).int()
    nums, batch_ptr, chunk_offsets = compute_causal_conv1d_metadata(
        cu_cpu, device=device
    )
    num_decodes = sum(n == 1 for n in lens)
    return GDNAttentionMetadata(
        num_prefills=len(lens) - num_decodes,
        num_prefill_tokens=sum(lens) - num_decodes,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=sum(lens),
        has_initial_state=torch.tensor(has_init, device=device),
        non_spec_query_start_loc=cu_cpu.to(device),
        non_spec_state_indices_tensor=torch.tensor(
            slots, dtype=torch.int32, device=device
        ),
        nums_dict=nums,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=chunk_offsets,
    )


def _run_step(kda, layer, step, conv, recurrent, conv_option, monkeypatch, device):
    """One _forward on clones of the caches; returns (out, conv, recurrent)."""
    lens = step[0]
    num_tokens, width = sum(lens), 3 * HEADS * D
    torch.manual_seed(1)
    # q|k|v is the leading, row-strided slice of the fused projection.
    projected = torch.randn(
        num_tokens, width + HEADS + 2 * D, device=device, dtype=torch.bfloat16
    )
    qkv, beta = projected[:, :width], projected[:, width : width + HEADS].unsqueeze(0)
    gate = torch.randn(1, num_tokens, HEADS, D, device=device, dtype=torch.bfloat16)
    conv, recurrent = conv.clone(), recurrent.clone()
    layer.kv_cache = (conv, recurrent)
    layer.kda_prefill_conv = conv_option
    if layer.kda_prefill_backend == "flashkda":
        layer._flashkda_buffer_specs = _flashkda_buffer_specs(step)
    monkeypatch.setattr(
        kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.prefix: _metadata(step, device)}),
    )
    out = torch.empty(1, num_tokens, HEADS, D, device=device, dtype=torch.bfloat16)
    layer._forward(qkv, gate, beta, out)
    return out, conv, recurrent


@pytest.mark.parametrize("backend", ["triton", "flashkda"])
@pytest.mark.parametrize("dim_first", [False, True], ids=["SD", "DS"])
@torch.inference_mode()
def test_planar_prefill_conv_matches_stock_in_layer(monkeypatch, backend, dim_first):
    from vllm.models.glm5next.common import kda

    if backend == "flashkda":
        if (
            kda._resolve_kda_prefill_backend("auto", D, torch.bfloat16, LOWER_BOUND)
            != "flashkda"
        ):
            pytest.skip("requires FlashKDA")
        import vllm._flashkda_C  # noqa: F401

    torch.manual_seed(42)
    device = torch.device("cuda")
    layer = _make_layer(kda, backend, dim_first, device, monkeypatch)
    width, num_slots = 3 * HEADS * D, 6
    conv_shape = (width, 3) if dim_first else (3, width)
    conv = torch.randn(num_slots, *conv_shape, device=device, dtype=torch.bfloat16)
    recurrent = torch.randn(num_slots, HEADS, D, D, device=device) * 0.1

    planar_calls = []
    planar = kda.kda_causal_conv1d_planar

    def counting_planar(*args, **kwargs):
        planar_calls.append(1)
        return planar(*args, **kwargs)

    monkeypatch.setattr(kda, "kda_causal_conv1d_planar", counting_planar)

    args = (kda, layer, LONG_STEP, conv, recurrent)
    out_stock, conv_stock, rec_stock = _run_step(
        *args, "causal_conv1d", monkeypatch, device
    )
    assert not planar_calls
    out_planar, conv_planar, rec_planar = _run_step(
        *args, "planar", monkeypatch, device
    )
    assert len(planar_calls) == 1
    assert torch.equal(conv_planar, conv_stock)
    assert torch.equal(out_planar, out_stock)
    assert torch.equal(rec_planar, rec_stock)

    # Short step: the dispatch guard keeps the stock kernel.
    args = (kda, layer, SHORT_STEP, conv, recurrent)
    out_stock, conv_stock, _ = _run_step(*args, "causal_conv1d", monkeypatch, device)
    out_guarded, conv_guarded, _ = _run_step(*args, "planar", monkeypatch, device)
    assert len(planar_calls) == 1
    assert torch.equal(conv_guarded, conv_stock)
    assert torch.equal(out_guarded, out_stock)


def test_resolve_kda_prefill_conv():
    from vllm.models.glm5next.common import kda

    assert kda._resolve_kda_prefill_conv("causal_conv1d", 4) == "causal_conv1d"
    assert kda._resolve_kda_prefill_conv("planar", 4) == "planar"
    with pytest.raises(ValueError, match="causal_conv1d"):
        kda._resolve_kda_prefill_conv("fused", 4)
    with pytest.raises(ValueError, match="width 4"):
        kda._resolve_kda_prefill_conv("planar", 3)
