# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash mHC sublayer boundary at decode token counts (<= 32 rows).

TileLang's fused post + fp32 prenorm GEMM (``mhc_fused_tilelang``) stays; its
second kernel, ``mhc_pre_big_fuse_with_norm`` (coefficients, Sinkhorn,
collapse, input RMSNorm; one 96-thread CTA per token, ~4.1 us at any decode
size on VR200), is replaced by ``mhc_decode.cu``: one 544-thread CTA per
token, the collapse spread over 16 warps with the row kept in registers
across the RMSNorm reduction, and Sinkhorn on a separate warp. post / comb
are bit-identical on the same partials; layer_input is within 1 bf16 ulp
(sum of squares reduced in another order). Opt-in via
``additional_config.mhc_decode_backend = "cuda"``.

The CUDA source next to this file is built on first use with FlashInfer's JIT
(nvcc, TVM-FFI binding) without fast math, so expf / division match TileLang.
"""

import functools
import hashlib
from pathlib import Path

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_SOURCE = Path(__file__).with_name("mhc_decode.cu")

HIDDEN_SIZE = 4096
HC_MULT = 4
MIX_SIZE = HC_MULT * (HC_MULT + 2)
# The TileLang fused post + GEMM covers up to this many tokens.
MAX_TOKENS = 32
# Run the Sinkhorn half on a side stream (set from additional_config by the
# model; the caller must call join() before reading post / comb).
SIDE_STREAM = False


@functools.cache
def _module():
    from flashinfer.jit.core import gen_jit_spec

    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    spec = gen_jit_spec(
        f"glm5next_mhc_decode_{digest}",
        [_SOURCE],
        extra_cuda_cflags=["-lineinfo"],
    )
    fast_math = ("-use_fast_math", "--use_fast_math")
    spec.extra_cuda_cflags = [f for f in spec.extra_cuda_cflags if f not in fast_math]
    return spec.build_and_load()


def load() -> None:
    """Build (or load the cached build of) the kernel module."""
    _module()


def is_supported(hidden_size: int, hc_mult: int) -> bool:
    return (
        hidden_size == HIDDEN_SIZE
        and hc_mult == HC_MULT
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (9, 0)
    )


@functools.cache
def _pdl() -> bool:
    from vllm.platforms import current_platform

    return bool(current_platform.is_arch_support_pdl())


def mhc_pre_decode(
    part: torch.Tensor,
    sq: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual_cur: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    pdl: bool | None = None,
    mode: int = 0,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mhc_pre_big_fuse_with_norm`` on split partials ``part`` [S, M, 24] and
    ``sq`` [S, M]; returns (post_mix [M, 4], comb_mix [M, 16], layer_input
    [M, 4096]). ``mode`` 0 writes all three, 1 only layer_input, 2 only
    post_mix / comb_mix (into ``out`` when given)."""
    num_tokens = residual_cur.shape[0]
    dev = residual_cur.device
    if out is None:
        post = torch.empty(num_tokens, HC_MULT, dtype=torch.float32, device=dev)
        comb = torch.empty(
            num_tokens, HC_MULT * HC_MULT, dtype=torch.float32, device=dev
        )
        layer_input = torch.empty(
            num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=dev
        )
    else:
        post, comb, layer_input = out
    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    _module().mhc_pre_decode(
        part,
        sq,
        hc_scale,
        hc_base,
        residual_cur,
        norm_weight,
        post,
        comb,
        layer_input,
        float(rms_eps),
        float(hc_pre_eps),
        float(hc_sinkhorn_eps),
        float(hc_post_mult_value),
        int(sinkhorn_repeat),
        float(norm_eps),
        int(_pdl() if pdl is None else pdl),
        int(mode),
    )
    return post, comb, layer_input


class _SideStream:
    """Per-device side stream for the Sinkhorn half of each boundary, and the
    event the next mHC op on the main stream must wait for (``join``)."""

    def __init__(self) -> None:
        self.stream: torch.cuda.Stream | None = None
        self.fork = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.pending = False


_SIDE: dict[int, _SideStream] = {}


def _side() -> _SideStream:
    dev = torch.cuda.current_device()
    side = _SIDE.get(dev)
    if side is None:
        side = _SIDE[dev] = _SideStream()
        side.stream = torch.cuda.Stream()
    return side


def join() -> None:
    """Make the current stream wait for the last side-stream Sinkhorn (call
    before anything reads the post / comb mix of a decode boundary)."""
    side = _SIDE.get(torch.cuda.current_device()) if _SIDE else None
    if side is not None and side.pending:
        torch.cuda.current_stream().wait_event(side.done)
        side.pending = False


def _side_stream_allowed() -> bool:
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    # Breakable (PIECEWISE) capture replays segments eagerly in a Python loop;
    # keep everything on one stream there, as maybe_execute_in_parallel does.
    return not BreakableCUDAGraphCapture.is_active()


def glm5next_mhc_decode_post_pre(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mhc_fused_post_pre_tilelang`` (with ``norm_weight``) for <= 32 tokens:
    TileLang fused post + prenorm GEMM, then the CUDA decode pre kernel. Larger
    calls take the TileLang path unchanged."""
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_FUSED_TILELANG_KERNEL,
        mhc_fused_post_pre_split_config,
    )

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.numel() // (hc_mult * hidden_size)
    if (
        num_tokens > MAX_TOKENS
        or mhc_fused_post_pre_split_config(num_tokens, hidden_size, hc_mult) is None
    ):
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
    join()
    outer_shape = residual.shape[:-2]
    hc_mult3 = hc_mult * (hc_mult + 2)
    gemm_out_mul, gemm_out_sqrsum, residual_cur = _MHC_FUSED_TILELANG_KERNEL(
        comb_res_mix.reshape(num_tokens, hc_mult, hc_mult),
        residual.reshape(num_tokens, hc_mult, hidden_size),
        post_layer_mix.reshape(num_tokens, hc_mult),
        x.reshape(num_tokens, hidden_size),
        fn.view(hc_mult3, hc_mult, hidden_size),
        hc_mult,
        hidden_size,
        hc_mult3,
    )
    eps = (rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
           norm_eps)
    if SIDE_STREAM and _side_stream_allowed():
        # Sinkhorn (post / comb, read only by the next boundary) on the side
        # stream; collapse + RMSNorm (layer_input) stays on the critical path.
        side = _side()
        dev = residual_cur.device
        post = torch.empty(num_tokens, HC_MULT, dtype=torch.float32, device=dev)
        comb = torch.empty(
            num_tokens, HC_MULT * HC_MULT, dtype=torch.float32, device=dev
        )
        main = torch.cuda.current_stream()
        side.fork.record(main)
        with torch.cuda.stream(side.stream):
            side.stream.wait_event(side.fork)
            mhc_pre_decode(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base,
                           residual_cur, norm_weight, *eps, pdl=False, mode=2,
                           out=(post, comb, residual_cur))
            side.done.record(side.stream)
        # The side stream reads these after the main stream lets them go.
        for t in (gemm_out_mul, gemm_out_sqrsum, post, comb):
            t.record_stream(side.stream)
        side.pending = True
        _, _, layer_input = mhc_pre_decode(gemm_out_mul, gemm_out_sqrsum, hc_scale,
                                           hc_base, residual_cur, norm_weight, *eps,
                                           mode=1)
    else:
        post, comb, layer_input = mhc_pre_decode(gemm_out_mul, gemm_out_sqrsum,
                                                 hc_scale, hc_base, residual_cur,
                                                 norm_weight, *eps)
    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post.view(*outer_shape, hc_mult, 1),
        comb.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _glm5next_mhc_decode_post_pre_fake(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outer_shape = residual.shape[:-2]
    return (
        torch.empty_like(residual),
        x.new_empty(*outer_shape, HC_MULT, 1, dtype=torch.float32),
        x.new_empty(*outer_shape, HC_MULT, HC_MULT, dtype=torch.float32),
        torch.empty_like(x),
    )


direct_register_custom_op(
    op_name="glm5next_mhc_decode_post_pre",
    op_func=glm5next_mhc_decode_post_pre,
    fake_impl=_glm5next_mhc_decode_post_pre_fake,
)
