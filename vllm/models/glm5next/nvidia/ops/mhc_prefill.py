# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash mHC sublayer boundary for large token counts, one launch.

Fuses the previous hc_post, the next hc_pre (prenorm GEMM on tensor cores,
Sinkhorn, collapse) and the input RMSNorm for hc_mult 4 / hidden 4096, which
vLLM's TileLang path runs as three kernels above 32 tokens. The CUDA source
next to this file is built on first use with FlashInfer's JIT (nvcc, TVM-FFI
binding), without fast math so expf / division match the TileLang kernels; the
module name carries the source hash. See mhc_prefill.cu for the design and
the numerics.
"""

import functools
import hashlib
from pathlib import Path

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_SOURCE = Path(__file__).with_name("mhc_prefill.cu")

HIDDEN_SIZE = 4096
HC_MULT = 4
MIX_SIZE = HC_MULT * (HC_MULT + 2)

# Supported (block_m, splits) instantiations, each with fn staged through
# shared memory or loaded into registers (fn_reg).
CONFIGS = ((64, 16), (64, 8), (64, 32), (32, 16), (32, 32), (32, 8))
DEFAULT_CONFIG = (32, 16)
DEFAULT_FN_REG = True
# Below this many tokens the TileLang path keeps r' in L2 between its three
# kernels and the single-pass kernel's per-tile tail costs more than it saves.
DEFAULT_MIN_TOKENS = 8192
# mode: 0 fused (one launch), 1 post + GEMM partials only, 2 pre only (from
# partials), 3 = 1 then 2 (two launches).
MODES = {"fused": 0, "gemm": 1, "pre": 2, "split": 3}


@functools.cache
def _module():
    from flashinfer.jit.core import gen_jit_spec

    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    spec = gen_jit_spec(
        f"glm5next_mhc_prefill_{digest}",
        [_SOURCE],
        extra_cuda_cflags=["-lineinfo"],
    )
    # FlashInfer builds with -use_fast_math; the Sinkhorn / sigmoid must use
    # IEEE division and the accurate expf, as the TileLang kernels do.
    fast_math = ("-use_fast_math", "--use_fast_math")
    spec.extra_cuda_cflags = [f for f in spec.extra_cuda_cflags if f not in fast_math]
    return spec.build_and_load()


def load() -> None:
    """Build (or load the cached build of) the kernel module."""
    _module()


def is_supported(hidden_size: int, hc_mult: int) -> bool:
    if hidden_size != HIDDEN_SIZE or hc_mult != HC_MULT:
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 0)


def copy(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Roofline helper: plain vectorised copy (same byte size)."""
    _module().copy(src, dst)


def mhc_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    *,
    block_m: int = DEFAULT_CONFIG[0],
    splits: int = DEFAULT_CONFIG[1],
    mode: str = "fused",
    fn_tf32: bool = True,
    fn_reg: bool = DEFAULT_FN_REG,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    residual_out: torch.Tensor | None = None,
    flags: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """hc_post(x, residual, post_mix, comb_mix) then hc_pre + RMSNorm.

    Shapes: x [M, 4096] bf16; residual [M, 4, 4096] bf16; post_mix [M, 4(, 1)]
    and comb_mix [M, 4, 4] fp32 (comb[i][o]: old stream i -> new stream o);
    fn [24, 16384] fp32; hc_scale [3], hc_base [24] fp32; norm_weight [4096]
    bf16. Returns (residual_cur [M, 4, 4096], post_mix_cur [M, 4, 1],
    comb_mix_cur [M, 4, 4], layer_input_cur [M, 4096]) like
    ``mhc_fused_post_pre_tilelang``.

    ``fn_tf32`` (default) truncates fn to TF32 as the DeepGEMM prenorm GEMM of
    the TileLang path does; False keeps 16 mantissa bits. ``workspace`` /
    ``residual_out`` are for the benchmark's split modes; ``flags`` bit 0
    skips the MMAs (memory-only diagnostic, wrong output).
    """
    num_tokens = x.shape[0]
    dev = x.device
    if residual_out is None:
        residual_out = torch.empty_like(residual)
    post_out = torch.empty(num_tokens, HC_MULT, 1, dtype=torch.float32, device=dev)
    comb_out = torch.empty(
        num_tokens, HC_MULT, HC_MULT, dtype=torch.float32, device=dev
    )
    layer_input = torch.empty(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=dev)
    if workspace is None:
        part = torch.empty(
            splits, num_tokens, MIX_SIZE, dtype=torch.float32, device=dev
        )
        sq = torch.empty(splits, num_tokens, dtype=torch.float32, device=dev)
    else:
        part, sq = workspace
    mode_id = MODES[mode]
    tiles = -(-num_tokens // block_m)
    if mode_id == 0:
        counters = torch.zeros(max(tiles, 1), dtype=torch.int32, device=dev)
    else:
        counters = torch.empty(max(tiles, 1), dtype=torch.int32, device=dev)
    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    _module().mhc_post_pre(
        x.contiguous(),
        residual.contiguous(),
        post_mix.contiguous(),
        comb_mix.contiguous(),
        fn.contiguous(),
        hc_scale.contiguous(),
        hc_base.contiguous(),
        norm_weight.contiguous(),
        residual_out,
        post_out,
        comb_out,
        layer_input,
        part,
        sq,
        counters,
        float(rms_eps),
        float(hc_pre_eps),
        float(hc_sinkhorn_eps),
        float(hc_post_mult_value),
        int(sinkhorn_repeat),
        float(norm_eps),
        int(splits),
        int(block_m),
        mode_id,
        int(bool(fn_tf32)),
        int(bool(fn_reg)),
        int(flags),
    )
    return residual_out, post_out, comb_out, layer_input


def glm5next_mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
    min_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mhc_fused_post_pre_tilelang`` with the single-pass kernel at and
    above ``min_tokens`` tokens."""
    if x.shape[0] < min_tokens:
        from vllm.model_executor.kernels.mhc.tilelang import (
            mhc_fused_post_pre_tilelang,
        )

        return mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    return mhc_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
    )


def _glm5next_mhc_fused_post_pre_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
    min_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = x.shape[0]
    return (
        torch.empty_like(residual),
        x.new_empty(num_tokens, HC_MULT, 1, dtype=torch.float32),
        x.new_empty(num_tokens, HC_MULT, HC_MULT, dtype=torch.float32),
        torch.empty_like(x),
    )


direct_register_custom_op(
    op_name="glm5next_mhc_fused_post_pre",
    op_func=glm5next_mhc_fused_post_pre,
    mutates_args=[],
    fake_impl=_glm5next_mhc_fused_post_pre_fake,
)
