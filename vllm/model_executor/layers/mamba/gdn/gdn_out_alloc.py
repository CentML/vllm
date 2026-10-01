# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN core_attn_out allocation that zeroes only the padding rows.

The fused-decode branch of QwenGatedDeltaNetAttention.forward_cuda allocated
core_attn_out ([num_tokens_padded, HV, D] bf16) with torch.zeros (one
Inductor zeros kernel per GDN layer). The GDN core op writes every real row
[0, num_actual) on every path, but padding rows [num_actual, padded) are
never written and must stay finite (vLLM PR 28182: garbage/NaN pad rows
reach out_proj, the residual stream and the MoE router of later layers).

vllm::ews_gdn_out_alloc(hidden_states, H, D) returns torch.empty(...) and
launches one tiny Triton kernel that zeroes row r iff slot_mapping[r] < 0.
Model-runner v2 marks exactly the padding rows [actual_num_tokens, max) of
its persistent slot-mapping buffer with PAD_SLOT_ID=-1 for every KV-cache
group, and that buffer has a fixed address, so the kernel is CUDA-graph safe
(piecewise and FULL). Real tokens always have a slot >= 0. Without a
forward-context slot mapping (profile / dummy runs) all rows are zeroed
(== stock). Outputs are bit-identical to stock on real rows (they are
overwritten by the core op either way) and pad rows are zero as before.

Env: EWS=1 and VLLM_GDN_OUT_ZERO_PAD_ROWS_ONLY (default 1) enable it.
"""

# ruff: noqa: E501
# Kernel source kept verbatim (Triton cache keys hash the source).
# fmt: off
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.lcd_pdl.triton_switch import lcd_pdl_triton_on as _lcd_pdl_on  # noqa: E402

logger = init_logger(__name__)

ENABLED = (os.environ.get("EWS", "0") == "1"
           and os.environ.get("VLLM_GDN_OUT_ZERO_PAD_ROWS_ONLY", "1") == "1")
# gb300 glue: with GLUE_GSC_QO=1 the GDN decode kernel writes the out_proj MXFP8 input of every row (padding rows
# zeroed), so in FULL (decode-only) CUDA graphs the bf16 padding rows are never read and their zeroing is deferred
# to the GDN core op, which zeroes them only if the fused quant did not run (zero_pad_rows_late).
# GLUE_GSC_QO_NOZERO=0 keeps the zeroing kernel.
NOZERO_FULL = (ENABLED and os.environ.get("GLUE_GSC_QO", "0") == "1"
               and os.environ.get("GLUE_GSC_QO_NOZERO", "1") == "1")
STATS = {"deferred": 0, "late_zero": 0, "skipped": 0}
_DEFERRED: dict = {}  # core_attn_out data_ptr -> slot mapping (or None) whose pad-row zeroing was deferred


@triton.jit
def _zero_pad_rows_kernel(out_ptr, slot_ptr, row_elems, BLOCK: tl.constexpr, HAS_SLOT: tl.constexpr, launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    r = tl.program_id(0)
    if HAS_SLOT:
        s = tl.load(slot_ptr + r)
        pad = s < 0
    else:
        pad = True
    if pad:
        base = out_ptr + r.to(tl.int64) * row_elems
        for off in range(0, row_elems, BLOCK):
            o = off + tl.arange(0, BLOCK)
            tl.store(base + o, tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty), mask=o < row_elems)


def _any_slot_mapping(T):
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            return None
        sm = get_forward_context().slot_mapping
        if isinstance(sm, list):
            sm = sm[0] if sm else None
        if not sm:
            return None
        for v in sm.values():
            if v is not None and v.dim() == 1 and v.shape[0] >= T and v.dtype in (torch.int64, torch.int32):
                return v
    except Exception:  # no usable slot mapping: zero every row (== stock)
        return None
    return None


def gdn_out_alloc_impl(like: torch.Tensor, h: int, d: int) -> torch.Tensor:
    T = like.shape[0]
    out = torch.empty((T, h, d), dtype=like.dtype, device=like.device)
    if T == 0:
        return out
    slot = _any_slot_mapping(T)
    if NOZERO_FULL and _full_graph_forward():
        _DEFERRED[out.data_ptr()] = slot
        STATS["deferred"] += 1
        return out
    _zero_pad_rows_kernel[(T,)](out, slot if slot is not None else out, h * d, BLOCK=1024,
                                HAS_SLOT=slot is not None, num_warps=4, launch_pdl=_lcd_pdl_on())
    return out


def _full_graph_forward() -> bool:
    """True inside a FULL-cudagraph forward (decode-only batches in FULL_AND_PIECEWISE)."""
    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available

        return (is_forward_context_available()
                and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL)
    except Exception:  # no forward context: keep the zeroing
        return False


def zero_pad_rows_late(out: torch.Tensor, quant_done: bool) -> None:
    """Called by the GDN core op right after the core ran on `out` ([T, h, d]): if this tensor's pad-row zeroing
    was deferred by gdn_out_alloc_impl and the decode kernel did NOT write the quantized rows itself
    (quant_done False), zero the pad rows now (same kernel, before anything reads them)."""
    if not _DEFERRED:
        return
    k = out.data_ptr()
    if k not in _DEFERRED:
        return
    slot = _DEFERRED.pop(k)
    if quant_done:
        STATS["skipped"] += 1
        return
    STATS["late_zero"] += 1
    T = out.shape[0]
    row = out[0].numel()
    _zero_pad_rows_kernel[(T,)](out, slot if slot is not None else out, row, BLOCK=1024,
                                HAS_SLOT=slot is not None, num_warps=4)


def gdn_out_alloc_fake(like: torch.Tensor, h: int, d: int) -> torch.Tensor:
    return like.new_empty((like.shape[0], h, d))


_REG = [False]


def register_op():
    if _REG[0]:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(op_name="ews_gdn_out_alloc", op_func=gdn_out_alloc_impl, mutates_args=[],
                              fake_impl=gdn_out_alloc_fake)
    _REG[0] = True


if ENABLED:
    register_op()
    logger.info("GDN core_attn_out: empty + padding-row zeroing enabled")

