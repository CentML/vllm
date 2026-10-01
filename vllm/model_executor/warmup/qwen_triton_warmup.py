# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Qwen Triton kernels from the loaded model's compile keys."""

import itertools
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

_QWEN_MODEL_TYPES = frozenset(
    {
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)

# Covers L=1 constexpr, non-divisible runtime L, and divisible runtime L.
_FLA_POST_CONV_WARMUP_LENGTHS = (1, 2, 16)


@dataclass(frozen=True)
class _QwenGDNWarmupConfig:
    h: int
    hv: int
    k: int
    v: int
    conv_kernel_size: int
    conv_state: torch.Tensor
    conv_dtype: torch.dtype
    norm_weight_dtype: torch.dtype
    norm_before_gate: bool
    norm_activation: str
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    state_stride_token: int
    state_dtype: torch.dtype
    # The FlashInfer prefill path makes the post-conv kernel emit exp(g).
    post_conv_output_g_exp: bool = False
    # out_proj consumes the MXFP8 activation written by the core op.
    gdn_out_mxfp8: bool = False
    # Prefill updates the SSM state pool in place (FlashInfer state_indices).
    state_in_place: bool = False

    @property
    def conv_dim(self) -> int:
        return 2 * self.h * self.k + self.hv * self.v


def _is_non_empty_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and value.numel() > 0


def _is_qwen_gdn_layer(module: object) -> bool:
    return all(
        hasattr(module, attr)
        for attr in (
            "num_k_heads",
            "num_v_heads",
            "head_k_dim",
            "head_v_dim",
            "conv_kernel_size",
            "tp_size",
            "kv_cache",
            "norm",
            "A_log",
            "dt_bias",
        )
    )


def _iter_qwen_gdn_layers(static_forward_context: object):
    if not isinstance(static_forward_context, dict):
        return

    for module in static_forward_context.values():
        if _is_qwen_gdn_layer(module):
            yield module


def _split_qwen_gdn_cache(kv_cache: object) -> tuple[torch.Tensor, torch.Tensor] | None:
    if isinstance(kv_cache, (list, tuple)) and len(kv_cache) >= 2:
        conv_cache, ssm_state = kv_cache[:2]
        if _is_non_empty_tensor(conv_cache) and _is_non_empty_tensor(ssm_state):
            return conv_cache, ssm_state

    if isinstance(kv_cache, torch.Tensor) and kv_cache.size(0) >= 2:
        conv_cache = kv_cache[0]
        ssm_state = kv_cache[1]
        if _is_non_empty_tensor(conv_cache) and _is_non_empty_tensor(ssm_state):
            return conv_cache, ssm_state
    return None


def _qwen_gdn_warmup_config(
    static_forward_context: object,
) -> _QwenGDNWarmupConfig | None:
    found_layer = False
    for layer in _iter_qwen_gdn_layers(static_forward_context):
        found_layer = True
        cache_tensors = _split_qwen_gdn_cache(getattr(layer, "kv_cache", None))
        if cache_tensors is None:
            continue

        conv_cache, ssm_state = cache_tensors
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        conv_state = (
            conv_cache if is_conv_state_dim_first() else conv_cache.transpose(-1, -2)
        )
        tp_size = int(layer.tp_size)
        h = int(layer.num_k_heads) // tp_size
        hv = int(layer.num_v_heads) // tp_size
        norm = layer.norm

        return _QwenGDNWarmupConfig(
            h=h,
            hv=hv,
            k=int(layer.head_k_dim),
            v=int(layer.head_v_dim),
            conv_kernel_size=int(layer.conv_kernel_size),
            conv_state=conv_state,
            conv_dtype=conv_state.dtype,
            norm_weight_dtype=norm.weight.dtype,
            norm_before_gate=bool(norm.norm_before_gate),
            norm_activation=str(norm.activation),
            a_log=layer.A_log,
            dt_bias=layer.dt_bias,
            state_stride_token=int(ssm_state.stride(0)),
            state_dtype=ssm_state.dtype,
            post_conv_output_g_exp=bool(
                getattr(
                    getattr(layer, "chunk_gated_delta_rule", None),
                    "expects_exp_g",
                    False,
                )
            ),
            gdn_out_mxfp8=bool(getattr(layer, "gdn_out_mxfp8", False)),
            state_in_place=bool(
                layer.chunk_gated_delta_rule.updates_state_in_place(ssm_state.dtype)
            ),
        )

    if found_layer:
        logger.info("Skipping Qwen GDN Triton warmup: no bound Qwen GDN cache found.")
    else:
        logger.info("Skipping Qwen GDN Triton warmup: no Qwen GDN layer found.")
    return None


def _warm_mxfp8_producers(runner: "GPUModelRunner", device: torch.device) -> None:
    """Every launch config of the SiLU-and-mul and attention-gate MXFP8
    producers: their tile and warp counts switch with the token count, so the
    dummy runs alone leave mixed-batch sizes to compile at runtime.
    """
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    silu_dims: set[int] = set()
    gate_shapes: set[tuple[int, int]] = set()
    for module in runner.get_model().modules():
        down_proj = getattr(module, "down_proj", None)
        if (
            isinstance(getattr(module, "act_fn", None), SiluAndMul)
            and down_proj is not None
            and get_input_quant_key(down_proj) == kMxfp8Dynamic
        ):
            silu_dims.add(down_proj.input_size_per_partition)
        if getattr(module, "attn_gate_mxfp8", False):
            gate_shapes.add((module.num_heads, module.head_dim))
    bf16 = torch.bfloat16
    if silu_dims:
        from vllm.model_executor.layers.fusion import silu_mul_mxfp8_quant as silu

        for d in silu_dims:
            for num_tokens in silu.LAUNCH_CONFIG_TOKEN_COUNTS:
                x = torch.zeros((num_tokens, 2 * d), dtype=bf16, device=device)
                silu.silu_mul_mxfp8_quant(x)
    if gate_shapes:
        from vllm.model_executor.layers.fusion import attn_gate_mxfp8_quant as gate

        for heads, head_dim in gate_shapes:
            for num_tokens in gate.LAUNCH_CONFIG_TOKEN_COUNTS:
                shape = (num_tokens, heads, head_dim)
                attn = torch.zeros(shape, dtype=bf16, device=device)
                # The gate half of q_gate, read strided as the layer does.
                q_gate = torch.zeros(
                    (num_tokens, heads, 2, head_dim), dtype=bf16, device=device
                )
                gate.attn_gate_mxfp8_quant(attn, q_gate[:, :, 1])


def _warm_gated_rms_norm_kernel(
    device: torch.device,
    config: _QwenGDNWarmupConfig,
    max_num_tokens: int,
    x_dtype: torch.dtype,
) -> None:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        gdn_norm_launch_config,
    )
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        warmup_layer_norm_fwd,
    )

    # Decode-only batches normalize T * HV rows of V; batches with prefill
    # normalize T rows of HV * V (one group per head), with the layer's launch
    # config.
    for rows_per_token in (config.hv, 1):
        launch_config = None
        if rows_per_token == 1:
            launch_config = partial(gdn_norm_launch_config, device=device)
        warmup_layer_norm_fwd(
            max_num_tokens=max_num_tokens,
            rows_per_token=rows_per_token,
            group_size=config.v,
            x_dtype=x_dtype,
            weight_dtype=config.norm_weight_dtype,
            device=device,
            norm_before_gate=config.norm_before_gate,
            is_rms_norm=True,
            activation=config.norm_activation,
            launch_config=launch_config,
        )


def _warm_gdn_gated_norm_mxfp8_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig, x_dtype: torch.dtype
) -> None:
    if not config.gdn_out_mxfp8:
        return
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        gdn_gated_norm_mxfp8,
        gdn_mxfp8_scale_numel,
    )

    hidden = config.hv * config.v
    weight = torch.ones(config.v, dtype=config.norm_weight_dtype, device=device)
    # Both num_warps classes, with the valid row count as a host int (mixed
    # batches) and as a device scalar (decode-only batches).
    for num_rows in (1, 1024):
        x = torch.zeros((num_rows, config.hv, config.v), dtype=x_dtype, device=device)
        q = torch.empty((num_rows, hidden), dtype=torch.float8_e4m3fn, device=device)
        scale = torch.empty(
            gdn_mxfp8_scale_numel(num_rows, hidden), dtype=torch.uint8, device=device
        )
        for num_valid in (
            num_rows,
            torch.full((1,), num_rows, dtype=torch.int32, device=device),
        ):
            gdn_gated_norm_mxfp8(
                x,
                x,
                weight,
                1e-6,
                config.norm_activation,
                q,
                scale,
                (0, num_rows),
                num_valid,
            )


def _warm_zero_fresh_state_rows_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    if not config.state_in_place:
        return
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_tail_ops import (
        zero_fresh_state_rows,
    )

    pool = torch.empty(
        (1, config.hv, config.k, config.v), dtype=config.state_dtype, device=device
    )
    zero_fresh_state_rows(
        pool,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.bool, device=device),
    )


def _warm_causal_conv1d_fwd_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

    x_storage = torch.empty(
        (1, config.conv_dim), dtype=config.conv_dtype, device=device
    )
    x = x_storage.t()
    weight = torch.empty(
        (config.conv_dim, config.conv_kernel_size),
        dtype=config.conv_dtype,
        device=device,
    )
    cache_indices = torch.full((1,), NULL_BLOCK_ID, dtype=torch.int32, device=device)
    has_initial_state = torch.empty(1, dtype=torch.bool, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)

    causal_conv1d_fn(
        x,
        weight,
        None,
        config.conv_state,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=NULL_BLOCK_ID,
        metadata=None,
        validate_data=False,
    )


def _warm_fused_post_conv_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (  # noqa: E501
        fused_post_conv_prep,
    )

    qkv_dim = 2 * config.h * config.k + config.hv * config.v
    for length in _FLA_POST_CONV_WARMUP_LENGTHS:
        conv_output = torch.empty(
            (length, qkv_dim), dtype=config.conv_dtype, device=device
        )
        a = torch.empty((length, config.hv), dtype=config.conv_dtype, device=device)
        b = torch.empty_like(a)

        fused_post_conv_prep(
            conv_output,
            a,
            b,
            config.a_log,
            config.dt_bias,
            config.h,
            config.k,
            config.v,
            apply_l2norm=True,
            output_g_exp=config.post_conv_output_g_exp,
        )


def _warm_fused_sigmoid_gating_delta_rule_update_kernel(
    device: torch.device,
    config: _QwenGDNWarmupConfig,
) -> None:
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    q = torch.empty((1, 1, config.h, config.k), dtype=config.conv_dtype, device=device)
    k = torch.empty_like(q)
    v = torch.empty((1, 1, config.hv, config.v), dtype=config.conv_dtype, device=device)
    a = torch.empty((1, 1, config.hv), dtype=config.conv_dtype, device=device)
    b = torch.empty_like(a)
    state = torch.empty(
        (1, config.state_stride_token),
        dtype=config.state_dtype,
        device=device,
    )
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    ssm_state_indices = torch.empty((1, 1), dtype=torch.int32, device=device)
    ssm_state_indices.zero_()

    fused_sigmoid_gating_delta_rule_update(
        A_log=config.a_log,
        a=a,
        b=b,
        dt_bias=config.dt_bias,
        q=q,
        k=k,
        v=v,
        beta=1.0,
        threshold=20.0,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
    )


# Sequence counts of the FlashInfer GDN prefill warmup in the state-pool form.
# FlashInfer specializes its context-parallel (CP) fixup kernel on indexed
# state I/O and on its CTA shape, which it picks from num_seqs * heads (one
# CTA wave or more), so cover single and multi-sequence CP batches.
_FI_GDN_POOL_WARMUP_NUM_SEQS = (1, 2, 3, 4, 8, 16)
_FI_GDN_POOL_WARMUP_MULTI_SEQ_LEN = 1024


def _warm_fi_gdn_prefill_state_pool(runner: "GPUModelRunner") -> None:
    """Run FlashInfer GDN prefill as serving does: through the layer's op with
    ``state_indices`` into a scratch pool with the live pool's layout.

    The layer's own warmup runs during profiling, before the state pool exists,
    and passes a fresh state without ``state_indices``; the CP fixup kernel of
    the indexed form then JIT-compiles at the first long prefill under load.
    Nothing in the live cache is touched.
    """
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        GDN_FI_NON_CP_MAX_TOKENS,
    )
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (  # noqa: E501
        fused_post_conv_prep,
    )

    for layer in _iter_qwen_gdn_layers(
        runner.compilation_config.static_forward_context
    ):
        cache = _split_qwen_gdn_cache(getattr(layer, "kv_cache", None))
        if cache is not None:
            pool = cache[1]
            break
    else:
        return
    op = getattr(layer, "chunk_gated_delta_rule", None)
    if op is None or not op.updates_state_in_place(pool.dtype):
        return

    device = pool.device
    dtype = runner.model_config.dtype
    h = int(layer.num_k_heads) // int(layer.tp_size)
    hv = int(layer.num_v_heads) // int(layer.tp_size)
    k_dim, v_dim = int(layer.head_k_dim), int(layer.head_v_dim)
    # One FLA chunk (non-CP), one sequence over the non-CP limit (CP), then
    # multi-sequence batches (CP while FlashInfer's heuristic picks it).
    batches = [[64]]
    if GDN_FI_NON_CP_MAX_TOKENS > 0:
        batches.append([GDN_FI_NON_CP_MAX_TOKENS + 64])
    batches += [
        [_FI_GDN_POOL_WARMUP_MULTI_SEQ_LEN] * n
        for n in _FI_GDN_POOL_WARMUP_NUM_SEQS
        if n > 1
    ]
    max_seqs = max(len(lens) for lens in batches)
    scratch = torch.empty_strided(
        (max_seqs + 1,) + tuple(pool.shape[1:]),
        pool.stride(),
        dtype=pool.dtype,
        device=device,
    )
    scratch.zero_()
    for lens in batches:
        num_tokens = sum(lens)
        mixed_qkv = torch.randn(
            num_tokens, 2 * h * k_dim + hv * v_dim, device=device, dtype=dtype
        )
        a = torch.randn(num_tokens, hv, device=device, dtype=dtype)
        b = torch.randn_like(a)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=mixed_qkv,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            num_k_heads=h,
            head_k_dim=k_dim,
            head_v_dim=v_dim,
            apply_l2norm=True,
            output_g_exp=op.expects_exp_g,
        )
        cu_seqlens = torch.tensor(
            [0, *itertools.accumulate(lens)], dtype=torch.int32, device=device
        )
        state_indices = torch.arange(1, len(lens) + 1, dtype=torch.int32, device=device)
        out = torch.empty(num_tokens, hv, v_dim, dtype=v.dtype, device=device)
        op(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=scratch,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            core_attn_out=out,
            state_indices=state_indices,
        )
    _synchronize_device(device)
    logger.info(
        "Warmed FlashInfer GDN prefill (state-pool form) for batches %s.",
        [(len(lens), lens[0]) for lens in batches],
    )


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize(device)


@torch.inference_mode()
def qwen_triton_warmup(
    runner: "GPUModelRunner",
    model_config: "ModelConfig",
) -> None:
    """Warm Qwen GDN Triton kernels reported by the JIT monitor."""
    model_type = getattr(model_config.hf_text_config, "model_type", "") or getattr(
        model_config.hf_config, "model_type", ""
    )
    if model_type not in _QWEN_MODEL_TYPES:
        return

    device = runner.device
    _warm_mxfp8_producers(runner, device)
    logger.info("Warming up Qwen GDN Triton kernels for model_type=%s.", model_type)

    gdn_config = _qwen_gdn_warmup_config(
        runner.compilation_config.static_forward_context
    )
    if gdn_config is None:
        _synchronize_device(device)
        return

    max_num_tokens = max(1, int(runner.max_num_tokens))
    _warm_gated_rms_norm_kernel(device, gdn_config, max_num_tokens, model_config.dtype)
    _warm_gdn_gated_norm_mxfp8_kernel(device, gdn_config, model_config.dtype)
    _warm_zero_fresh_state_rows_kernel(device, gdn_config)
    _warm_causal_conv1d_fwd_kernel(device, gdn_config)
    _warm_fused_post_conv_kernel(device, gdn_config)
    # Pooling only runs full prefills; the decode update kernel is unused.
    if not runner.is_pooling_model:
        _warm_fused_sigmoid_gating_delta_rule_update_kernel(device, gdn_config)
    _warm_fi_gdn_prefill_state_pool(runner)
    _synchronize_device(device)
