# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team
#
# Adapted from FlashInfer (flashinfer/gdn_kernels/blackwell/gdn_prefill.py,
# chunk_gated_delta_rule_sm100), licensed under the Apache License, Version 2.0.
"""Host adapter for the V-split non-CP GDN chunked-prefill kernel.

Mirrors FlashInfer's gdn_kernels/blackwell/gdn_prefill.py (chunk_gated_delta_rule_sm100) but
compiles GatedDeltaNetChunkedKernel with a ``v_split`` factor: work item = (seq, v_head, v_slice),
each CTA owns DV/v_split rows of the state and of the output. v_split=1 is the unmodified kernel.
Compiles with plain cute.compile (no on-disk cache), so it works with the installed FlashInfer
0.6.18 package.
"""

import functools
import os
from typing import Optional

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from vllm.logger import init_logger

from .gdn_chunked_vs import GatedDeltaNetChunkedKernel

logger = init_logger(__name__)

@functools.cache
def _num_sm(dev_index: int) -> int:
    return torch.cuda.get_device_properties(dev_index).multi_processor_count


@functools.cache
def _cache(*key):
    return {}


# Key of the latest call; the GDN step plan's direct launch (GGM_VSDIRECT) checks
# its own rebuilt key against it (gdn_step_plan._vs_direct).
_LAST_KEY: list = [None]


_CG0_SPLIT = os.environ.get("VLLM_GDN_VSPLIT_CG0SPLIT", "0") == "1"
# VLLM_GDN_VSPLIT_C1REORDER=1 (default off): issue the state-update GEMM (KV) before the output GEMM (QKV) and let
# compute group 1 drain chunk c's output after it has published state c+1 (exact: same ops, different order).
_C1_REORDER = os.environ.get("VLLM_GDN_VSPLIT_C1REORDER", "0") == "1"


def _cg0_split(v_split: int, use_init: bool) -> bool:
    """VLLM_GDN_VSPLIT_CG0SPLIT=1 (default off): v_split=2 launches run as 2-CTA clusters (the two V slices of a head)
    whose CG0 warpgroups split the V-independent WY prep (A_inv / W_qkv) by pair and share it through DSMEM.
    Bit-identical to the unsplit kernel (same MMAs, same reduction order, same bf16 rounding points)."""
    return _CG0_SPLIT and int(v_split) == 2 and bool(use_init)


def _io(t):
    return {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}[t]


def _st(t):
    return {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16,
            torch.float16: cutlass.Float16}[t]


def _mark_state(s_cute, use_state_indices: bool, DK: int):
    if use_state_indices:
        s_cute.mark_layout_dynamic()
    else:
        s_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
            mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)


def chunk_gated_delta_rule_vsplit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    scale: float,
    state_indices: Optional[torch.Tensor] = None,
    v_split: int = 2,
) -> None:
    """Same contract as flashinfer chunk_gated_delta_rule_sm100 (no checkpoints).
    q/k: [T, HQ, 128] bf16, v/output: [T, HV, 128], gate (=exp(g)) / beta: [T, HV] fp32,
    cu_seqlens int32 [B+1], states [N, HV, 128(V), 128(K)] fp32 (pool if state_indices)."""
    HQ, HV, DK = q.size(1), v.size(1), q.size(2)
    assert DK == 128 and v.size(2) == 128
    assert cu_seqlens.dtype == torch.int32
    is_GQA = HQ >= HV
    use_init = initial_state is not None
    store_final = output_state is not None
    use_idx = state_indices is not None
    st_dtype = (initial_state if use_init else output_state).dtype if (use_init or store_final) else torch.float32
    B = cu_seqlens.size(0) - 1
    dev = q.device.index if q.device.index is not None else torch.cuda.current_device()
    num_sm = _num_sm(dev)
    key = (dev, num_sm, str(q.dtype), str(st_dtype), HQ, HV, is_GQA, use_init, store_final, use_idx,
           str(state_indices.dtype) if use_idx else "none",
           tuple(initial_state.stride()[1:]) if (use_idx and use_init) else None,
           tuple(output_state.stride()[1:]) if (use_idx and store_final) else None,
           int(v_split), _cg0_split(v_split, use_init), _C1_REORDER and use_init)
    _LAST_KEY[0] = key
    c = _cache(*key)
    dv = 128 // v_split
    if "compiled" not in c:
        if _cg0_split(v_split, use_init) or (_C1_REORDER and use_init):
            logger.info("V-split GDN chunk kernel: compiling v_split=%d cg0_split=%s c1_reorder=%s",
                        v_split, _cg0_split(v_split, use_init), _C1_REORDER and use_init)
        gdn = GatedDeltaNetChunkedKernel(
            io_dtype=_io(q.dtype), inverse_dtype=_io(q.dtype), acc_dtype=cutlass.Float32,
            state_dtype=_st(st_dtype),
            mma_tiler_qk=(64, 64, 128), mma_tiler_qs=(dv, 64, 128), mma_tiler_qkv=(dv, 64, 64),
            mma_tiler_kv=(dv, 128, 64), max_active_clusters=num_sm, num_sm=num_sm, is_GQA=is_GQA,
            use_initial_state=use_init, store_final_state=store_final, enable_checkpoints=False,
            is_persistent=True, v_split=v_split,
            cg0_split=_cg0_split(v_split, use_init), c1_reorder=_C1_REORDER and use_init)

        def dyn(t, nd):
            x = from_dlpack(t, assumed_align=16)
            x.mark_compact_shape_dynamic(mode=0, stride_order=tuple(range(nd)), divisibility=1)
            return x
        qc, kc, vc = dyn(q, 3), dyn(k, 3), dyn(v, 3)
        gc, bc, oc = dyn(gate, 2), dyn(beta, 2), dyn(output, 3)
        cuc = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()
        sic = soc = sidx = None
        if use_init:
            sic = from_dlpack(initial_state, assumed_align=16); _mark_state(sic, use_idx, DK)
        if store_final:
            soc = from_dlpack(output_state, assumed_align=16); _mark_state(soc, use_idx, DK)
        if use_idx:
            sidx = from_dlpack(state_indices, assumed_align=4).mark_layout_dynamic()
        ws = torch.empty(GatedDeltaNetChunkedKernel.get_workspace_size(num_sm, B, HQ, HV, True),
                         dtype=torch.int8, device=q.device)
        wsc = from_dlpack(ws, assumed_align=16)
        stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
        c["compiled"] = cute.compile(
            gdn, qc, kc, vc, gc, bc, oc, cuc, sic, soc, sidx, None, None, 0, scale, wsc, stream,
            options="--enable-tvm-ffi --opt-level 3")
    ws = torch.empty(GatedDeltaNetChunkedKernel.get_workspace_size(num_sm, B, HQ, HV, True),
                     dtype=torch.int8, device=q.device)
    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    c["compiled"](q, k, v, gate, beta, output, cu_seqlens,
                  initial_state if use_init else None, output_state if store_final else None,
                  state_indices if use_idx else None, None, None, 0, scale, ws, stream)
