# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-path tests for fused Qwen3.5 GDN MTP decode."""

from __future__ import annotations

import types
from typing import cast
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.has_device_capability(80)):
    pytest.skip(
        reason="Fused GDN MTP decode requires CUDA compute capability 8.0+.",
        allow_module_level=True,
    )

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (  # noqa: E402
    rmsnorm_fn,
)
from vllm.utils.torch_utils import _encode_layer_name  # noqa: E402
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

NUM_SPEC = 3
SPEC_TOKENS = NUM_SPEC + 1
H = 1
HV = 8
K = 128
V = 128
CONV_KERNEL = 4
CONV_DIM = 2 * H * K + HV * V
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"
EPS = 1e-6


class _TestGatedNorm:
    def __init__(self, weight: torch.Tensor, activation: str) -> None:
        self.weight = weight
        self.bias = None
        self.eps = EPS
        self.group_size = None
        self.norm_before_gate = True
        self.activation = activation

    def __call__(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return rmsnorm_fn(
            x,
            self.weight,
            None,
            z=z,
            eps=self.eps,
            norm_before_gate=True,
            activation=self.activation,
        )


def _make_vllm_config():
    config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    config.additional_config = {"gdn_prefill_backend": "cutedsl"}
    config.cache_config.mamba_cache_mode = "none"
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    return config


def _build_layer(
    vllm_config,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    output_gate_activation: str,
):
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=False,
        disable_tp_for_ba_proj=False,
        tp_size=1,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=K,
        value_dim=HV * V,
        activation="silu",
        A_log=a_log,
        dt_bias=dt_bias,
        conv1d=types.SimpleNamespace(weight=conv_weight, bias=None),
        kv_cache=(conv_state, ssm_state),
        norm=_TestGatedNorm(norm_weight, output_gate_activation),
        layer_norm_epsilon=EPS,
        gdn_decode_kernel="cuda",
        _fused_decode_counters=None,
        _ba_pending=False,
    )
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
        "_forward_core_decode_spec_post_conv_fused_norm",
        "_forward_core_decode_spec_fused_norm",
        "_can_use_fused_gdn_mtp_decode",
        "_rms_norm_gated_cuda",
        "_rms_norm_gated_strided_gate_cuda",
        "_forward_core_fused_norm",
        "_forward_core_fused_norm_packed",
        "_forward_core_decode_spec_one_launch",
        "_gated_norm_mxfp8",
        "split_ba",
        "_in_proj_ba_join",
    ):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


@pytest.mark.parametrize(
    "num_v_heads,expected",
    [
        pytest.param(2, True, id="ratio1"),
        pytest.param(4, True, id="ratio2"),
        pytest.param(6, True, id="ratio3"),
        pytest.param(8, True, id="ratio4"),
        pytest.param(16, True, id="ratio8"),
        pytest.param(5, False, id="non-integral-ratio"),
        pytest.param(10, False, id="unsupported-ratio5"),
    ],
)
def test_fused_mtp_head_ratio_guard(num_v_heads: int, expected: bool) -> None:
    if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"):
        pytest.skip("fused GDN decode MTP op is not built")

    layer = types.SimpleNamespace(
        num_k_heads=2,
        num_v_heads=num_v_heads,
        kv_cache=(None, torch.empty(1, dtype=torch.float32, device="cuda")),
        gdn_decode_kernel="cuda",
    )
    attn_metadata = types.SimpleNamespace(
        spec_state_indices_tensor=torch.ones(
            1, SPEC_TOKENS, dtype=torch.int32, device="cuda"
        ),
        spec_sequence_masks=object(),
        num_decodes=0,
        num_spec_decodes=1,
    )

    assert (
        QwenGatedDeltaNetAttention._can_use_fused_gdn_mtp_decode(
            cast(QwenGatedDeltaNetAttention, layer),
            cast(GDNAttentionMetadata, attn_metadata),
        )
        is expected
    )


@torch.inference_mode()
def test_fused_forward_uses_packed_entrypoint() -> None:
    """Fused mode keeps projected QKVZ and BA packed through the model op."""
    device = torch.device("cuda")
    num_tokens = 3
    hidden_states = torch.empty(num_tokens, 1, dtype=torch.bfloat16, device=device)
    mixed_qkvz = torch.randn(
        num_tokens, CONV_DIM + HV * V, dtype=torch.bfloat16, device=device
    )
    ba = torch.randn(num_tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_fused_gdn_decode=True,
        gdn_out_mxfp8=False,
        norm=types.SimpleNamespace(
            weight=torch.empty(V, dtype=torch.bfloat16, device=device)
        ),
        num_v_heads=HV,
        tp_size=1,
        head_v_dim=V,
        in_proj_qkvz=lambda _: (mixed_qkvz, None),
        in_proj_ba=lambda _: (ba, None),
        out_proj=lambda x: (x, None),
        _ba_stream=None,
    )
    layer._in_projections = types.MethodType(
        QwenGatedDeltaNetAttention._in_projections, layer
    )
    layer.forward_cuda = types.MethodType(
        QwenGatedDeltaNetAttention.forward_cuda, layer
    )

    def packed_op(
        actual_qkvz: torch.Tensor,
        actual_ba: torch.Tensor,
        output: torch.Tensor,
        *,
        layer_name: str,
    ) -> None:
        assert actual_qkvz is mixed_qkvz
        assert actual_ba is ba
        assert layer_name == _encode_layer_name(PREFIX)
        output.fill_(1)

    with (
        patch.object(
            torch.ops.vllm,
            "qwen_gdn_attention_core_fused_norm_packed",
            side_effect=packed_op,
        ) as packed_mock,
        patch.object(torch.ops.vllm, "qwen_gdn_attention_core") as triton_mock,
    ):
        output = layer.forward_cuda(hidden_states)

    packed_mock.assert_called_once()
    triton_mock.assert_not_called()
    torch.testing.assert_close(output, torch.ones_like(output))


@pytest.mark.parametrize(
    "seq_lens,query_lens,draft_tokens,expected_fused_calls",
    [
        pytest.param([128], [SPEC_TOKENS], [NUM_SPEC], 1, id="pure-mtp"),
        pytest.param(
            [128, 96],
            [SPEC_TOKENS, 64],
            [NUM_SPEC, -1],
            1,
            id="mixed-mtp",
        ),
        pytest.param(
            [128, 96, 128],
            [SPEC_TOKENS, 64, SPEC_TOKENS],
            [NUM_SPEC, -1, NUM_SPEC],
            0,
            id="mixed-interleaved-falls-back",
        ),
        pytest.param([96], [64], [-1], 0, id="pure-prefill"),
        pytest.param([128], [1], [-1], 0, id="pure-decode"),
    ],
)
@pytest.mark.parametrize("output_gate_activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("decode_backend", ["cuda", "cuda_jit", "triton"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_fused_model_path_matches_reference(
    seq_lens: list[int],
    query_lens: list[int],
    draft_tokens: list[int],
    expected_fused_calls: int,
    output_gate_activation: str,
    decode_backend: str,
    state_dtype: torch.dtype,
) -> None:
    """Fused MTP (decode-only batches and the contiguous spec block of mixed
    batches) and its interleaved/prefill/decode fallbacks match the reference,
    with the spec recurrence in the csrc CUDA MTP kernel or the JIT
    register-resident CUDA kernel (gated norm inside), or in the Triton
    value-split kernel (gated norm in the following norm launch).
    """
    if decode_backend == "cuda" and not hasattr(
        torch.ops._C, "fused_gdn_decode_post_conv_mtp"
    ):
        pytest.skip("fused GDN decode MTP op is not built")
    if decode_backend == "cuda_jit" and not qwen_gdn_linear_attn.gdn_mtp_cuda.enable():
        pytest.skip("GDN MTP CUDA kernel did not build")
    torch.manual_seed(1)
    device = torch.device("cuda")
    vllm_config = _make_vllm_config()
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(vllm_config):
        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=torch.ones(
                batch.batch_size, dtype=torch.int32, device=device
            ),
            num_decode_draft_tokens_cpu=torch.tensor(draft_tokens, dtype=torch.int32),
        )

    state_indices = [
        indices
        for indices in (
            metadata.spec_state_indices_tensor,
            metadata.non_spec_state_indices_tensor,
        )
        if indices is not None and indices.numel() > 0
    ]
    pool_size = max(int(indices.max().item()) for indices in state_indices) + 1
    conv_state_shape, temporal_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, H, HV, K, V, CONV_KERNEL, NUM_SPEC
        )
    )
    conv_state_seed = 0.05 * torch.randn(
        pool_size, *conv_state_shape, dtype=torch.bfloat16, device=device
    )
    ssm_state_seed = (
        0.01 * torch.randn(pool_size, *temporal_state_shape, device=device)
    ).to(state_dtype)
    a_log = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    conv_weight = 0.1 * torch.randn(
        CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device
    )
    norm_weight = torch.randn(V, dtype=torch.float32, device=device)
    num_tokens = batch.compute_num_tokens()
    mixed_qkv = 0.1 * torch.randn(
        num_tokens, CONV_DIM, dtype=torch.bfloat16, device=device
    )
    b = 0.1 * torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device)
    a = 0.1 * torch.randn_like(b)
    output_gate = 0.1 * torch.randn(
        num_tokens, HV, V, dtype=torch.bfloat16, device=device
    )
    mixed_qkvz = torch.cat((mixed_qkv, output_gate.flatten(1)), dim=-1)
    ba = torch.cat((b, a), dim=-1)
    context = types.SimpleNamespace(attn_metadata={PREFIX: metadata})

    reference_layer = _build_layer(
        vllm_config,
        conv_state_seed.clone(),
        ssm_state_seed.clone(),
        a_log,
        dt_bias,
        conv_weight,
        norm_weight,
        output_gate_activation,
    )
    reference_out = torch.zeros_like(output_gate)
    with patch.object(
        qwen_gdn_linear_attn, "get_forward_context", return_value=context
    ):
        reference_layer._forward_core(
            mixed_qkv=mixed_qkv.clone(),
            b=b,
            a=a,
            core_attn_out=reference_out,
        )
    reference_out = reference_layer.norm(reference_out, output_gate)

    fused_layer = _build_layer(
        vllm_config,
        conv_state_seed.clone(),
        ssm_state_seed.clone(),
        a_log,
        dt_bias,
        conv_weight,
        norm_weight,
        output_gate_activation,
    )
    context.no_compile_layers = {PREFIX: fused_layer}
    fused_out = torch.zeros_like(output_gate)
    use_triton = decode_backend == "triton"
    gmc = qwen_gdn_linear_attn.gdn_mtp_cuda
    with (
        patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=context),
        patch.object(
            qwen_gdn_linear_attn,
            "GDN_MTP_TRITON_MAX_REQUESTS",
            1 << 20 if use_triton else 0,
        ),
        patch.object(gmc, "ready", return_value=decode_backend == "cuda_jit"),
        patch.object(gmc, "gdn_mtp_cuda", wraps=gmc.gdn_mtp_cuda) as jit_mock,
        patch.object(
            qwen_gdn_linear_attn.ops,
            "fused_gdn_decode_post_conv_mtp",
            wraps=qwen_gdn_linear_attn.ops.fused_gdn_decode_post_conv_mtp,
        ) as cuda_mock,
        patch.object(
            qwen_gdn_linear_attn,
            "gdn_mtp_recurrence",
            wraps=qwen_gdn_linear_attn.gdn_mtp_recurrence,
        ) as triton_mock,
    ):
        torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
            mixed_qkvz.clone(),
            ba,
            fused_out,
            layer_name=_encode_layer_name(PREFIX),
        )

    mocks = {"cuda": cuda_mock, "cuda_jit": jit_mock, "triton": triton_mock}
    for backend, mock in mocks.items():
        expected = expected_fused_calls if backend == decode_backend else 0
        assert mock.call_count == expected, backend
    torch.testing.assert_close(
        fused_layer.kv_cache[0], reference_layer.kv_cache[0], atol=0, rtol=0
    )
    torch.testing.assert_close(fused_out, reference_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        fused_layer.kv_cache[1],
        reference_layer.kv_cache[1],
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_requests", [1, 3])
@torch.inference_mode()
def test_one_launch_model_path_matches_three_kernels(
    num_requests: int, state_dtype: torch.dtype
) -> None:
    """Decode-only spec batches of at most VLLM_GDN_FUSED_DECODE_MAX_REQS
    requests take the one-launch core (gdn_mtp_fused_decode) and produce
    out_proj's MXFP8 activation, conv window and SSM states of the
    conv -> Triton recurrence -> gated-norm MXFP8 path; larger batches keep
    the three-kernel path.
    """
    torch.manual_seed(2)
    device = torch.device("cuda")
    vllm_config = _make_vllm_config()
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    batch = BatchSpec(
        seq_lens=[128, 96, 64][:num_requests], query_lens=[SPEC_TOKENS] * num_requests
    )
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(vllm_config):
        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=torch.tensor(
                [1, 3, 2][:num_requests], dtype=torch.int32, device=device
            ),
            num_decode_draft_tokens_cpu=torch.full(
                (num_requests,), NUM_SPEC, dtype=torch.int32
            ),
        )
    assert metadata.num_prefills == 0 and metadata.num_spec_decodes == num_requests

    assert metadata.spec_state_indices_tensor is not None
    pool_size = int(metadata.spec_state_indices_tensor.max().item()) + 1
    conv_state_shape, temporal_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, H, HV, K, V, CONV_KERNEL, NUM_SPEC
        )
    )
    conv_state_seed = 0.5 * torch.randn(
        pool_size, *conv_state_shape, dtype=torch.bfloat16, device=device
    )
    ssm_state_seed = (
        0.05 * torch.randn(pool_size, *temporal_state_shape, device=device)
    ).to(state_dtype)
    a_log = torch.log(torch.empty(HV, device=device).uniform_(1, 16))
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    conv_weight = 0.5 * torch.randn(
        CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device
    )
    norm_weight = (1 + 0.1 * torch.randn(V, device=device)).to(torch.bfloat16)
    num_tokens = batch.compute_num_tokens()
    mixed_qkvz = torch.randn(
        num_tokens, CONV_DIM + HV * V, dtype=torch.bfloat16, device=device
    )
    ba = torch.randn(num_tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    context = types.SimpleNamespace(attn_metadata={PREFIX: metadata})

    def run(one_launch: bool):
        layer = _build_layer(
            vllm_config,
            conv_state_seed.clone(),
            ssm_state_seed.clone(),
            a_log,
            dt_bias,
            conv_weight,
            norm_weight,
            "silu",
        )
        layer._fused_decode_counters = torch.zeros(
            4 * H, dtype=torch.int32, device=device
        )
        context.no_compile_layers = {PREFIX: layer}
        out_q = torch.empty(
            num_tokens, HV * V, dtype=torch.float8_e4m3fn, device=device
        )
        out_scale = torch.empty(
            qwen_gdn_linear_attn.gdn_mxfp8_scale_numel(num_tokens, HV * V),
            dtype=torch.uint8,
            device=device,
        )
        with (
            patch.object(
                qwen_gdn_linear_attn, "get_forward_context", return_value=context
            ),
            patch.object(qwen_gdn_linear_attn, "GDN_MTP_TRITON_MAX_REQUESTS", 4),
            patch.object(
                qwen_gdn_linear_attn,
                "GDN_FUSED_DECODE_MAX_REQUESTS",
                4 if one_launch else 0,
            ),
            patch.object(
                qwen_gdn_linear_attn,
                "gdn_mtp_fused_decode",
                wraps=qwen_gdn_linear_attn.gdn_mtp_fused_decode,
            ) as fused_mock,
        ):
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                mixed_qkvz.clone(),
                ba,
                torch.empty(num_tokens, HV, V, dtype=torch.bfloat16, device=device),
                layer_name=_encode_layer_name(PREFIX),
                out_q=out_q,
                out_scale=out_scale,
            )
        assert fused_mock.call_count == int(one_launch)
        assert int(layer._fused_decode_counters.abs().sum()) == 0
        return layer, out_q, out_scale

    ref_layer, ref_q, ref_scale = run(one_launch=False)
    layer, out_q, out_scale = run(one_launch=True)
    assert torch.equal(layer.kv_cache[0], ref_layer.kv_cache[0])
    if state_dtype == torch.bfloat16:
        # Both store the fp32 result rounded to nearest even: one bf16 step
        # apart where the fp32 results straddle a rounding boundary.
        torch.testing.assert_close(
            layer.kv_cache[1], ref_layer.kv_cache[1], rtol=2**-7, atol=1e-6
        )
        assert (layer.kv_cache[1] != ref_layer.kv_cache[1]).float().mean() < 1e-3
    else:
        torch.testing.assert_close(
            layer.kv_cache[1], ref_layer.kv_cache[1], rtol=1e-5, atol=1e-6
        )
    assert torch.equal(out_scale, ref_scale)
    flips = out_q.view(torch.uint8) != ref_q.view(torch.uint8)
    assert flips.float().mean() < 1e-3


@pytest.mark.parametrize("prefill_first", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_mixed_fork_matches_serial(
    prefill_first: bool, state_dtype: torch.dtype
) -> None:
    """VLLM_GDN_MIXED_FORK: a mixed batch whose spec block and prefill rows run on
    two streams leaves the output and both state pools bitwise equal to the
    one-stream path, with either half on the side stream.
    """
    torch.manual_seed(3)
    device = torch.device("cuda")
    vllm_config = _make_vllm_config()
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    # Two spec requests lead the batch, then two prefills (one fresh).
    batch = BatchSpec(
        seq_lens=[128, 112, 96, 160], query_lens=[SPEC_TOKENS, SPEC_TOKENS, 64, 160]
    )
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(vllm_config):
        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=torch.tensor(
                [2, 1, 1, 1], dtype=torch.int32, device=device
            ),
            num_decode_draft_tokens_cpu=torch.tensor(
                [NUM_SPEC, NUM_SPEC, -1, -1], dtype=torch.int32
            ),
        )
    assert metadata.num_prefills == 2 and metadata.spec_token_start is not None
    indices = [
        t
        for t in (
            metadata.spec_state_indices_tensor,
            metadata.non_spec_state_indices_tensor,
        )
        if t is not None and t.numel() > 0
    ]
    pool_size = max(int(t.max().item()) for t in indices) + 1
    conv_state_shape, temporal_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, H, HV, K, V, CONV_KERNEL, NUM_SPEC
        )
    )
    conv_state_seed = 0.05 * torch.randn(
        pool_size, *conv_state_shape, dtype=torch.bfloat16, device=device
    )
    ssm_state_seed = (
        0.01 * torch.randn(pool_size, *temporal_state_shape, device=device)
    ).to(state_dtype)
    a_log = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    conv_weight = 0.1 * torch.randn(
        CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device
    )
    norm_weight = torch.randn(V, dtype=torch.float32, device=device)
    num_tokens = batch.compute_num_tokens()
    mixed_qkvz = 0.1 * torch.randn(
        num_tokens, CONV_DIM + HV * V, dtype=torch.bfloat16, device=device
    )
    ba = 0.1 * torch.randn(num_tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    context = types.SimpleNamespace(attn_metadata={PREFIX: metadata})

    def run(fork: bool):
        layer = _build_layer(
            vllm_config,
            conv_state_seed.clone(),
            ssm_state_seed.clone(),
            a_log,
            dt_bias,
            conv_weight,
            norm_weight,
            "silu",
        )
        context.no_compile_layers = {PREFIX: layer}
        out = torch.zeros(num_tokens, HV, V, dtype=torch.bfloat16, device=device)
        with (
            patch.object(
                qwen_gdn_linear_attn, "get_forward_context", return_value=context
            ),
            patch.object(qwen_gdn_linear_attn, "GDN_MIXED_FORK", fork),
            patch.object(
                qwen_gdn_linear_attn,
                "GDN_MIXED_FORK_PREFILL_FIRST",
                0 if prefill_first else 1 << 30,
            ),
        ):
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                mixed_qkvz.clone(),
                ba,
                out,
                layer_name=_encode_layer_name(PREFIX),
            )
        torch.accelerator.synchronize()
        return layer, out

    ref_layer, ref_out = run(fork=False)
    layer, out = run(fork=True)
    # The fork path ran (it creates the side streams on first use).
    assert qwen_gdn_linear_attn._gdn_mixed_fork_state
    assert torch.equal(out, ref_out)
    assert torch.equal(layer.kv_cache[0], ref_layer.kv_cache[0])
    assert torch.equal(layer.kv_cache[1], ref_layer.kv_cache[1])
