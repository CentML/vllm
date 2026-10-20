# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash mHC decode boundary with the TP all-reduce fused in.

At <= 32 tokens every TP4 / TP8 sublayer boundary runs FlashInfer's MNNVL
one-shot all-reduce of the o_proj / MoE output (~5 us on VR200) and then
TileLang's fused hc_post + fp32 prenorm GEMM partials. ``mhc_ar_post.cu``
does both in one launch on FlashInfer's own MNNVL workspace and Lamport
protocol (so it interleaves with FlashInfer all-reduces on the same group) and
returns the same (GEMM partials, sum-of-squares partials, residual) as
TileLang, bit for bit; the CUDA decode pre (mhc_decode.cu) follows unchanged.
Opt-in via ``additional_config.mhc_decode_ar_fusion = True`` (needs
``mhc_decode_backend = "cuda"``).

The CUDA source next to this file is built on first use with FlashInfer's JIT
(nvcc, TVM-FFI binding) without fast math.
"""

import functools
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch

_SOURCE = Path(__file__).with_name("mhc_ar_post.cu")

HIDDEN_SIZE = 4096
HC_MULT = 4
MIX_SIZE = HC_MULT * (HC_MULT + 2)
# TileLang's fused post + GEMM (and so this kernel) covers up to this many tokens.
MAX_TOKENS = 32
WORLD_SIZES = (2, 4, 8)

# (tokens per CTA, outputs per CTA) by token count; any choice is bit-identical
# (see mhc_ar_post.cu), only the speed differs. Initial guess, tuned on VR200.
_DEFAULT_TILES = {1: (1, 6), 2: (1, 6), 4: (2, 6), 8: (2, 6), 16: (4, 6), 32: (4, 6)}
_SUPPORTED_TILES = {
    256: ((1, 2), (1, 6), (1, 24), (2, 6), (4, 6)),
    128: ((1, 2), (1, 6), (2, 6), (4, 6), (2, 3), (4, 3)),
}


def tilelang_split(num_tokens: int) -> tuple[int, int]:
    """(threads, splits) of mhc_fused_tilelang at this token count (its
    mhc_fused_post_pre_split_config for hidden 4096, hc_mult 4)."""
    return (256, 16) if num_tokens <= 4 else (128, 8)


def _tile_override() -> dict[int, tuple[int, int]]:
    """``VLLM_GLM53_MHC_AR_TILES="1:1x6,8:2x6"`` (tokens:tbxtile_n) for tuning."""
    spec = os.environ.get("VLLM_GLM53_MHC_AR_TILES", "")
    out = {}
    for item in filter(None, spec.split(",")):
        m, cfg = item.split(":")
        tb, tn = cfg.split("x")
        out[int(m)] = (int(tb), int(tn))
    return out


@functools.cache
def tiles(num_tokens: int) -> tuple[int, int]:
    table = {**_DEFAULT_TILES, **_tile_override()}
    key = min((m for m in table if m >= num_tokens), default=max(table))
    tb, tn = table[key]
    nthr, _ = tilelang_split(num_tokens)
    if (tb, tn) not in _SUPPORTED_TILES[nthr]:
        tb, tn = (1, 6)
    return tb, tn


# hc_post contraction order matching TileLang's build (mhc_ar_post.cu,
# GLM_MHCAR_X_FIRST); the VR200 check confirms it against on-device TileLang.
X_FIRST = int(os.environ.get("VLLM_GLM53_MHC_AR_X_FIRST", "0"))


def _module(world: int, x_first: int | None = None):
    return _build(world, X_FIRST if x_first is None else x_first)


@functools.cache
def _build(world: int, x_first: int):
    """One JIT module per (world size, contraction variant)."""
    from flashinfer.jit.core import gen_jit_spec

    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    spec = gen_jit_spec(
        f"glm5next_mhc_ar_post_{digest}_w{world}_x{x_first}",
        [_SOURCE],
        extra_cuda_cflags=[
            "-lineinfo",
            f"-DGLM_MHCAR_WORLD={world}",
            f"-DGLM_MHCAR_X_FIRST={x_first}",
        ],
    )
    fast_math = ("-use_fast_math", "--use_fast_math")
    spec.extra_cuda_cflags = [f for f in spec.extra_cuda_cflags if f not in fast_math]
    return spec.build_and_load()


def load(world: int) -> None:
    """Build (or load the cached build of) the module for this world size."""
    _module(world)


@functools.cache
def _pdl() -> bool:
    from vllm.platforms import current_platform

    return bool(current_platform.is_arch_support_pdl())


@dataclass(frozen=True)
class LamportWorkspace:
    """The pieces of a FlashInfer MNNVL all-reduce workspace the kernel uses."""

    mcast_ptr: int  # 0: unicast stores to every rank (tests)
    uc_ptrs_dev: int
    flags: torch.Tensor  # uint32 [9]
    buffer_size_bytes: int
    world_size: int
    rank: int

    def fits(self, num_tokens: int) -> bool:
        return num_tokens * HIDDEN_SIZE * self.world_size * 2 <= self.buffer_size_bytes


def mhc_ar_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    fn: torch.Tensor,
    ws: LamportWorkspace | None,
    x_out: torch.Tensor | None = None,
    tile: tuple[int, int] | None = None,
    pdl: bool | None = None,
    resident_ranks: int = 1,
    x_first: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-reduce of the TP partial ``x`` [M, 4096] (``ws`` None: world size 1,
    no reduction) fused with mhc_fused_tilelang: returns (yp [S, M, 24], rp
    [S, M], residual_out [M, 4, 4096]). ``x_out`` (tests) receives the reduced
    x; ``resident_ranks`` > 1 when several ranks share one GPU (tests);
    ``x_first`` picks the hc_post contraction variant (tests, see X_FIRST)."""
    num_tokens = x.shape[0]
    dev = x.device
    nthr, splits = tilelang_split(num_tokens)
    tb, tn = tile if tile is not None else tiles(num_tokens)
    yp = torch.empty(splits, num_tokens, MIX_SIZE, dtype=torch.float32, device=dev)
    rp = torch.empty(splits, num_tokens, dtype=torch.float32, device=dev)
    residual_out = torch.empty(
        num_tokens, HC_MULT, HIDDEN_SIZE, dtype=torch.bfloat16, device=dev
    )
    if ws is None:
        world, rank, mcast, ucs, flags = 1, 0, 0, 0, _empty_u32(dev)
    else:
        world, rank = ws.world_size, ws.rank
        mcast, ucs, flags = ws.mcast_ptr, ws.uc_ptrs_dev, ws.flags
    _module(world, x_first).mhc_ar_post(
        x.reshape(num_tokens, HIDDEN_SIZE),
        residual.reshape(num_tokens, HC_MULT, HIDDEN_SIZE),
        post_mix.reshape(num_tokens, HC_MULT),
        comb_mix.reshape(num_tokens, HC_MULT * HC_MULT),
        fn.reshape(MIX_SIZE, HC_MULT * HIDDEN_SIZE),
        yp,
        rp,
        residual_out,
        x_out if x_out is not None else _empty_bf16(dev),
        int(mcast),
        int(ucs),
        flags,
        int(world),
        int(rank),
        nthr,
        tb,
        tn,
        int(_pdl() if pdl is None else pdl),
        int(resident_ranks),
    )
    return yp, rp, residual_out


@functools.cache
def _empty_u32(dev: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint32, device=dev)


@functools.cache
def _empty_bf16(dev: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.bfloat16, device=dev)


# ---------------------------------------------------------------------------
# vLLM integration: the TP group's FlashInfer MNNVL workspace.

_WS: LamportWorkspace | None = None
_WS_RESOLVED = False


def tp_workspace(x: torch.Tensor) -> LamportWorkspace | None:
    """The TP group's FlashInfer MNNVL all-reduce workspace, or None when the
    group does not run FlashInfer MNNVL all-reduce (then the caller keeps the
    separate all-reduce). Collective on first use (workspace creation), so
    every TP rank must call it at the same point, which the model does."""
    global _WS, _WS_RESOLVED
    if _WS_RESOLVED:
        return _WS
    from vllm.distributed import get_tp_group
    from vllm.distributed.device_communicators import flashinfer_all_reduce as fiar

    comm = getattr(get_tp_group().device_communicator, "fi_ar_comm", None)
    ws = None
    if comm is not None and not comm.disabled and comm.world_size in WORLD_SIZES:
        # Creates the shared workspace on first use (as the first all-reduce
        # would); every rank reaches this with the same shape.
        comm.should_use_fi_ar(x.reshape(-1, HIDDEN_SIZE))
        fi_ws = fiar._fi_ar_workspace
        if (
            fi_ws is not None
            and fi_ws.backend == "mnnvl"
            and getattr(fi_ws, "mc_ptr", 0)
        ):
            ws = LamportWorkspace(
                mcast_ptr=int(fi_ws.mc_ptr),
                uc_ptrs_dev=int(fi_ws.uc_ptrs_dev),
                flags=fi_ws.buffer_flags,
                buffer_size_bytes=int(fi_ws.buffer_size_bytes),
                world_size=int(comm.world_size),
                rank=int(comm.rank),
            )
    _WS, _WS_RESOLVED = ws, True
    return ws
