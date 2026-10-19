# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Large-M path of the TRT-LLM NVFP4 MoE: run FlashInfer's CuTe-DSL MoE on the
same weight tensors.

At large prefill row counts (e.g. agg DEP2 with 65536 tokens per rank, i.e.
131072 dispatched rows), FlashInfer's CuTe-DSL NVFP4 MoE (gather + SwiGLU + FP4
FC1, FC2 + fused finalize) with a fixed tile-128 / N-256 tactic is ~15 % faster
than the tuned trtllm-gen MoE on VR200 (SM107), while trtllm-gen stays better
for decode. Keeping a second CuTe-DSL copy of the expert weights is not
affordable, so this path re-lays the TRT-LLM tensors out per call into one
scratch buffer shared by all layers.

Both layouts hold the same NVFP4 bytes and E4M3 block scales, and differ only
by row permutations inside 128-row (W13) / 32-row (W2) blocks:
  * TRT-LLM W13 row j  = [up; gate] row p13[j]  (gated-act 8-row interleave +
    shuffle_matrix_a), CuTe-DSL W13 row r = [up; gate] row pc[r] (64-row
    up/gate interleave);
  * TRT-LLM W2 row j = row p2[j] (shuffle_matrix_a), CuTe-DSL W2 is unpermuted;
  * block scales use the same row permutation as their weights, then the same
    128x4 swizzle. A row permutation inside 128-row blocks moves whole 4-byte
    groups of the swizzled layout, so scales are re-indexed as int32.
The re-layout reads and writes ~2 GB per call for GLM-5.3 (144 local experts,
H 4096, I 2048): ~0.3 ms on VR200, against ~0.6 ms saved at 131072 rows.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# (tile_size, gemm1, gemm2); gemm = (mma_tiler, mma_inst_shape, cluster, raster)
_TACTICS = {
    "t128_n256_n256": (
        128,
        ((128, 256, 256), (128, 256, 128), (1, 1), False),
        ((128, 256, 256), (128, 256, 128), (1, 1), False),
    ),
    "t128_n128_n128": (
        128,
        ((128, 128, 256), (128, 128, 128), (1, 1), False),
        ((128, 128, 256), (128, 128, 128), (1, 1), False),
    ),
    "t128_n256_n128": (
        128,
        ((128, 256, 256), (128, 256, 128), (1, 1), False),
        ((128, 128, 256), (128, 128, 128), (1, 1), False),
    ),
}

_index_cache: dict[tuple, tuple[torch.Tensor, ...]] = {}


@triton.jit
def _row_gather_kernel(
    src_ptr, idx_ptr, dst_ptr, N, C: tl.constexpr, ROWS: tl.constexpr
):
    """dst[e, r, :] = src[e, idx[r], :] for [E, N, C] int64 tensors."""
    e = tl.program_id(1).to(tl.int64)
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, C)
    src_r = tl.load(idx_ptr + r).to(tl.int64)
    base = e * N * C
    vals = tl.load(src_ptr + base + src_r[:, None] * C + c[None, :])
    tl.store(dst_ptr + base + r.to(tl.int64)[:, None] * C + c[None, :], vals)


def _gather_rows(src: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor) -> None:
    """Row gather along dim 1 of a contiguous [E, N, bytes] tensor at copy speed."""
    num_e, n = src.shape[0], src.shape[1]
    s64 = src.view(torch.uint8).reshape(num_e, n, -1).view(torch.int64)
    d64 = dst.view(torch.uint8).reshape(num_e, n, -1).view(torch.int64)
    rows = 8
    assert n % rows == 0 and s64.is_contiguous() and d64.is_contiguous()
    _row_gather_kernel[(n // rows, num_e)](
        s64, idx, d64, n, C=s64.shape[2], ROWS=rows, num_warps=4
    )


def tactic_from_name(name: str):
    """None (FlashInfer autotuner / default) for "auto", else a forced tactic."""
    if name == "auto":
        return None
    if name not in _TACTICS:
        raise ValueError(
            f"Unknown CuTe-DSL MoE tactic {name!r}; choose from {['auto', *_TACTICS]}"
        )
    return _TACTICS[name]


def _swizzled_row_gather(row_src: torch.Tensor, n_cols4: int) -> torch.Tensor:
    """int32-unit gather map for 128x4-swizzled scales: dst_swz = src_swz[map]
    where the linear (unswizzled) rows satisfy dst[r] = src[row_src[r]] and
    each row holds n_cols4 groups of 4 scale bytes (contiguous in both)."""
    n_rows = row_src.numel()
    dev = row_src.device
    n4 = n_rows * n_cols4
    # linear int32 unit (r * n_cols4 + k4) stored at each swizzled int32 slot
    lin_of_swz = (
        torch.arange(n4, device=dev, dtype=torch.int64)
        .view(n_rows // 128, 4, 32, n_cols4)
        .permute(0, 3, 2, 1)
        .reshape(-1)
    )
    swz_of_lin = torch.empty_like(lin_of_swz)
    swz_of_lin[lin_of_swz] = torch.arange(n4, device=dev, dtype=torch.int64)
    r, k4 = lin_of_swz // n_cols4, lin_of_swz % n_cols4
    return swz_of_lin[row_src[r] * n_cols4 + k4].contiguous()


def _relayout_indices(num_rows13: int, hidden: int, inter: int, device):
    key = (num_rows13, hidden, inter, str(device))
    if key not in _index_cache:
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
            interleave_linear_and_gate,
        )

        epilogue_tile_m = 128  # as in prepare_static_weights_for_trtllm_fp4_moe
        p13 = _maybe_get_cached_w3_w1_permute_indices(
            {},
            torch.empty(num_rows13, hidden // 2, dtype=torch.uint8),
            epilogue_tile_m,
            is_gated_act_gemm=True,
        ).to(device)
        inv13 = torch.empty_like(p13)
        inv13[p13] = torch.arange(num_rows13, device=device, dtype=p13.dtype)
        pc = interleave_linear_and_gate(
            torch.arange(num_rows13, device=device, dtype=torch.int64),
            group_size=64,
            dim=0,
        )
        rows13 = inv13[pc].long().contiguous()  # cute row r <- trt row rows13[r]
        p2 = get_w2_permute_indices_with_cache(
            {}, torch.empty(hidden, inter // 2, dtype=torch.uint8), epilogue_tile_m
        ).to(device)
        rows2 = torch.empty_like(p2)
        rows2[p2] = torch.arange(hidden, device=device, dtype=p2.dtype)
        rows2 = rows2.long().contiguous()  # natural row n <- trt row rows2[n]
        sf13 = _swizzled_row_gather(rows13, hidden // 64)
        sf2 = _swizzled_row_gather(rows2, inter // 64)
        _index_cache[key] = (rows13, rows2, sf13, sf2)
    return _index_cache[key]


def supported(experts, hidden_states: torch.Tensor) -> bool:
    """Static eligibility of a TrtLlmNvFp4Experts instance for this path."""
    qc = experts.quant_config
    w1_scale = qc.w1_scale
    if experts.expects_unquantized_inputs or w1_scale is None:
        return False
    if experts.gemm1_alpha is not None or experts.gemm1_beta is not None:
        return False  # Rubin CuTe-DSL kernel: plain (clamped) SwiGLU only
    num_e, rows13, hp = w1_scale.shape
    hidden = experts.hidden_dim
    inter = experts.intermediate_size_per_partition
    return (
        rows13 == 2 * inter
        and hp * 16 == hidden
        and hidden % 256 == 0
        and inter % 128 == 0
        and hidden_states.dtype == torch.uint8
        and hidden_states.shape[-1] * 2 == hidden
    )


def apply_cutedsl_large_m(
    experts,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int,
    a1q_scale: torch.Tensor,
    tactic,
    swiglu_limit: float | None,
) -> bool:
    """Finalized MoE into ``output`` with the CuTe-DSL kernels, reading the
    TRT-LLM weight tensors through a per-call re-layout into shared scratch.
    Returns False (nothing launched) if the scratch cannot be reserved."""
    from flashinfer.fused_moe import ActivationType
    from flashinfer.fused_moe.cute_dsl.fused_moe import cute_dsl_fused_moe

    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        nvfp4_swizzled_scale_to_cutedsl_mma_view,
    )
    from vllm.v1.worker.workspace import current_workspace_manager

    qc = experts.quant_config
    num_e = w1.shape[0]
    hidden = experts.hidden_dim
    inter = experts.intermediate_size_per_partition
    rows13 = 2 * inter
    dev = w1.device
    idx13, idx2, sf13, sf2 = _relayout_indices(rows13, hidden, inter, dev)

    shapes = (
        ((num_e, rows13, hidden // 2), torch.uint8),
        ((num_e, rows13 * hidden // 64), torch.int32),
        ((num_e, hidden, inter // 2), torch.uint8),
        ((num_e, hidden * inter // 64), torch.int32),
    )
    try:
        w13_c, s13_c, w2_c, s2_c = current_workspace_manager().get_persistent_resource(
            ("trtllm_nvfp4_cutedsl_large_m", shapes, str(dev)),
            lambda: tuple(torch.empty(s, dtype=d, device=dev) for s, d in shapes),
        )
    except AssertionError as e:
        # Workspace locked before any large-M call reserved the scratch
        # (warmup never reached the threshold): stay on trtllm-gen.
        logger.warning_once("CuTe-DSL large-M MoE path disabled: %s", e)
        return False
    _gather_rows(w1, idx13, w13_c)
    torch.index_select(
        qc.w1_scale.view(torch.int32).reshape(num_e, -1), 1, sf13, out=s13_c
    )
    _gather_rows(w2, idx2, w2_c)
    torch.index_select(
        qc.w2_scale.view(torch.int32).reshape(num_e, -1), 1, sf2, out=s2_c
    )
    w1_sf = nvfp4_swizzled_scale_to_cutedsl_mma_view(
        s13_c.view(torch.float8_e4m3fn).view(num_e, rows13, hidden // 16)
    )
    w2_sf = nvfp4_swizzled_scale_to_cutedsl_mma_view(
        s2_c.view(torch.float8_e4m3fn).view(num_e, hidden, inter // 16)
    )
    kwargs = {}
    if swiglu_limit is not None:
        kwargs["swiglu_limit"] = float(swiglu_limit)
    cute_dsl_fused_moe(
        hidden_states,
        a1q_scale.view(torch.float8_e4m3fn).reshape(hidden_states.shape[0], -1, 1),
        topk_ids.to(torch.int32),
        topk_weights.float(),
        w13_c,
        w1_sf,
        qc.g1_alphas,
        qc.a2_gscale,
        w2_c,
        w2_sf,
        qc.g2_alphas,
        global_num_experts,
        experts.topk,
        experts.local_num_experts,
        experts.ep_rank * experts.local_num_experts,
        moe_output=output,
        activation_type=ActivationType.Swiglu.value,
        tactic=tactic,
        **kwargs,
    )
    return True
