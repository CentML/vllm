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
from typing import Optional

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.gdn_host_trim import GDN_HOST_TRIM

from .gdn_chunked_vs import GatedDeltaNetChunkedKernel

logger = init_logger(__name__)


@functools.cache
def _num_sm(dev_index: int) -> int:
    return torch.cuda.get_device_properties(dev_index).multi_processor_count


@functools.cache
def _cache(*key):
    return {}


# VLLM_GDN_HOST_TRIM: compiled kernel by raw key; (workspace, CUstream) by
# raw stream handle (the persistent kernel's workspace size depends only on
# the SM count).
_fast_compiled: dict = {}
_fast_ws: dict = {}


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
    cu_seqlens int32 [B+1], states [N, HV, 128(V), 128(K)] fp32 or bf16 (pool if state_indices)."""
    HQ, HV, DK = q.size(1), v.size(1), q.size(2)
    assert DK == 128 and v.size(2) == 128
    assert cu_seqlens.dtype == torch.int32
    use_init = initial_state is not None
    store_final = output_state is not None
    use_idx = state_indices is not None
    if GDN_HOST_TRIM:
        # Same compiled kernel and arguments; the key, the workspace (scratch
        # for per-CTA TMA descriptors, written before use) and the stream
        # object are looked up instead of rebuilt per call.
        dev = q.device.index
        fkey = (
            dev, q.dtype,
            (initial_state if use_init else output_state).dtype
            if (use_init or store_final) else None,
            HQ, HV, use_init, store_final,
            state_indices.dtype if use_idx else None,
            initial_state.stride()[1:] if (use_idx and use_init) else None,
            output_state.stride()[1:] if (use_idx and store_final) else None,
            v_split,
        )
        compiled = _fast_compiled.get(fkey)
        if compiled is not None:
            handle = torch._C._cuda_getCurrentRawStream(dev)
            ws_stream = _fast_ws.get(handle)
            if ws_stream is None:
                ws = torch.empty(
                    GatedDeltaNetChunkedKernel.get_workspace_size(
                        _num_sm(dev), cu_seqlens.size(0) - 1, HQ, HV, True),
                    dtype=torch.int8, device=q.device)
                ws_stream = (ws, cuda.CUstream(handle))
                _fast_ws[handle] = ws_stream
            compiled(q, k, v, gate, beta, output, cu_seqlens,
                     initial_state if use_init else None,
                     output_state if store_final else None,
                     state_indices if use_idx else None, None, None, 0, scale,
                     ws_stream[0], ws_stream[1])
            return
    is_GQA = HQ >= HV
    st_dtype = (initial_state if use_init else output_state).dtype if (use_init or store_final) else torch.float32
    B = cu_seqlens.size(0) - 1
    dev = q.device.index if q.device.index is not None else torch.cuda.current_device()
    num_sm = _num_sm(dev)
    key = (dev, num_sm, str(q.dtype), str(st_dtype), HQ, HV, is_GQA, use_init, store_final, use_idx,
           str(state_indices.dtype) if use_idx else "none",
           tuple(initial_state.stride()[1:]) if (use_idx and use_init) else None,
           tuple(output_state.stride()[1:]) if (use_idx and store_final) else None,
           int(v_split))
    c = _cache(*key)
    dv = 128 // v_split
    if "compiled" not in c:
        # Serving must never get here after the prefill warmup; the log line
        # makes an unwarmed variant visible.
        logger.info("gdn_vsplit: compiling V-split kernel, key=%s", key)
        gdn = GatedDeltaNetChunkedKernel(
            io_dtype=_io(q.dtype), inverse_dtype=_io(q.dtype), acc_dtype=cutlass.Float32,
            state_dtype=_st(st_dtype),
            mma_tiler_qk=(64, 64, 128), mma_tiler_qs=(dv, 64, 128), mma_tiler_qkv=(dv, 64, 64),
            mma_tiler_kv=(dv, 128, 64), max_active_clusters=num_sm, num_sm=num_sm, is_GQA=is_GQA,
            use_initial_state=use_init, store_final_state=store_final, enable_checkpoints=False,
            is_persistent=True, v_split=v_split)

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
    if GDN_HOST_TRIM:
        _fast_compiled[fkey] = c["compiled"]
