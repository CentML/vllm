# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash mHC sublayer boundary for large token counts, one launch.

Two backends for ``glm5next_mhc_fused_post_pre`` at and above a token count:
"deepgemm" runs DeepGEMM's mega_mhc (non-shifted, BF16 output), "cuda" runs
the kernel below. On VR200 (job 803407) mega_mhc is 1.10-1.26x faster than
the TileLang path from 1024 tokens on; the CUDA kernel is slower there
(issue-bound at 2 CTAs/SM) and kept as an experimental option.

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
CONFIGS = ((64, 16), (64, 8), (64, 32), (32, 16), (32, 32), (32, 8), (32, 64))
# Fastest single-launch config on VR200 (job 803407), still 0.96x of TileLang.
DEFAULT_CONFIG = (64, 16)
DEFAULT_FN_REG = False
# Per backend: the token count from which it replaces TileLang (VR200, job
# 803407: mega_mhc 1.01x at 256 tokens, 1.10x at 1024, 1.14-1.26x above).
DEFAULT_MIN_TOKENS = {"deepgemm": 1024, "cuda": 8192}
BACKENDS = tuple(DEFAULT_MIN_TOKENS)
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


def is_supported(backend: str, hidden_size: int, hc_mult: int) -> bool:
    if hidden_size != HIDDEN_SIZE or hc_mult != HC_MULT:
        return False
    if not torch.cuda.is_available():
        return False
    if backend == "cuda":
        return torch.cuda.get_device_capability() >= (8, 0)
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import _import_deep_gemm, is_deep_gemm_supported

    if not (
        is_deep_gemm_supported() and current_platform.is_device_capability_family(100)
    ):
        return False
    deep_gemm = _import_deep_gemm()
    return deep_gemm is not None and callable(getattr(deep_gemm, "mega_mhc", None))


def mhc_post_pre_deep_gemm(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The same boundary with DeepGEMM mega_mhc (non-shifted, BF16 output)."""
    from vllm.utils.deep_gemm import mega_mhc

    num_tokens = x.shape[0]
    new_residual = torch.empty_like(residual)
    new_post = x.new_empty(num_tokens, HC_MULT, 1, dtype=torch.float32)
    new_comb = x.new_empty(num_tokens, HC_MULT, HC_MULT, dtype=torch.float32)
    layer_input = torch.empty_like(x)
    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    mega_mhc(
        x=x,
        residual=residual,
        shifted_prev_mix=None,
        post_mix=post_mix.view(num_tokens, HC_MULT, 1),
        comb_res_mix=comb_mix.view(num_tokens, HC_MULT, HC_MULT),
        fn=fn,
        mix_scales=hc_scale,
        mix_bases=hc_base,
        hc_mult=HC_MULT,
        hc_norm_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_post_scale=hc_post_mult_value,
        sinkhorn_eps=hc_sinkhorn_eps,
        num_sinkhorn_iters=sinkhorn_repeat,
        rmsnorm_weight=norm_weight,
        rmsnorm_eps=norm_eps,
        rmsnorm_scale=1.0,
        new_residual=new_residual,
        new_prev_mix=None,
        new_post_mix=new_post,
        new_comb_res_mix=new_comb,
        y_bf16=layer_input,
    )
    return new_residual, new_post, new_comb, layer_input


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
    fn_tf32: bool = False,
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

    ``fn_tf32`` truncates fn to TF32; the default keeps 16 mantissa bits, which
    lands closer to the TileLang path's DeepGEMM TF32 GEMM on VR200 (post/comb
    within 4.6e-4 vs 9.0e-4 truncated, job 803407). ``workspace`` /
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
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mhc_fused_post_pre_tilelang`` with a single-launch ``backend`` at and
    above ``min_tokens`` tokens. DeepGEMM's mega_mhc keeps per-stream
    workspace that cannot be created while capturing, so captured graphs keep
    TileLang for it."""
    if x.shape[0] < min_tokens or (
        backend == "deepgemm" and torch.cuda.is_current_stream_capturing()
    ):
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
    run = mhc_post_pre_deep_gemm if backend == "deepgemm" else mhc_post_pre
    return run(
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
    backend: str,
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
