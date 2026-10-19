# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipelined GLM-5.3-Flash KDA single-token decode (one launch per layer).

Fuses the depthwise conv + SiLU, q/k L2 norm, bounded gate, beta sigmoid, one
gated delta-rule step on the recurrent state [slot, H, 128, 128] (fp32 or
bf16) and the gated RMSNorm. The CUDA source next to this file is built on
first use with FlashInfer's JIT (nvcc, TVM-FFI binding); the module name
carries the source hash, so an edited kernel is never served from a stale
cache. See kda_decode_pipelined.cu for the design.
"""

import functools
import hashlib
from pathlib import Path

import torch

_SOURCE = Path(__file__).with_name("kda_decode_pipelined.cu")


@functools.cache
def _module():
    from flashinfer.jit.core import gen_jit_spec

    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    spec = gen_jit_spec(
        f"glm5next_kda_decode_pipelined_{digest}",
        [_SOURCE],
        extra_cuda_cflags=["-lineinfo"],
    )
    return spec.build_and_load()


def load() -> None:
    """Build (or load the cached build of) the kernel module."""
    _module()


def get_kernel():
    """The raw TVM-FFI entry point (same positional arguments as
    :func:`kda_decode`), for callers that want to skip the Python wrapper."""
    return _module().kda_decode


def default_launch_config(device: torch.device) -> tuple[int, int]:
    """(ring depth, SM count) the kernel picks by default on ``device``."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    mod = _module()
    return int(mod.default_ring_for_device(index)), int(mod.sm_count_for_device(index))


def stream_copy(
    src: torch.Tensor, dst: torch.Tensor, ring: int = 0, ctas: int = 0, mode: int = 0
) -> None:
    """Roofline helper: copy ``src`` to ``dst`` with the kernel's streaming
    pattern, no math. mode 0: bulk loads into the shared-memory ring + warp
    stores; 1: ring + bulk (TMA) stores; 2: plain vectorised copy kernel."""
    _module().stream_copy(src, dst, int(ring), int(ctas), int(mode))


def kda_decode(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor,
    lower_bound: float,
    norm_eps: float,
    ring: int = 0,
    ctas: int = 0,
    flags: int = -1,
) -> None:
    """One decode step for ``x.shape[0]`` rows, written into ``output``.

    x: [rows, 3*H*128] bf16 (row-strided view OK); conv_weight: [3*H*128, 4]
    fp32 (vLLM's merged q|k|v weight); conv_state: [slots, 3*H*128, 3] bf16 in
    either layout (DS or SD, any strides); raw_gate / output_gate:
    [(1,) rows, H, 128] bf16; raw_beta: [(1,) rows, H] bf16; A_log: [H] fp32;
    dt_bias: [H*128] fp32; state_indices: [rows] int32 (<= 0: null row, zero
    output, no cache write); state: [slots, H, 128, 128] fp32 or bf16;
    norm_weight: [128] fp32; output: [(1,) rows, H, 128] bf16.
    ``ring``/``ctas`` override the shared-memory ring depth and the grid size
    (0 = defaults: 6 slots or the deepest that fits, one CTA per SM); ``flags`` picks
    the variant (0: half-warp rows; 1: quarter-warp rows with packed fp32x2
    math; -1 = the default, 1).
    """
    _module().kda_decode(
        x,
        conv_weight,
        conv_state,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_indices,
        state,
        output_gate,
        norm_weight,
        output,
        float(lower_bound),
        float(norm_eps),
        int(ring),
        int(ctas),
        int(flags),
    )
