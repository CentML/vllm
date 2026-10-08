# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed BF16 routing for the opt-in folded W4A8 expert path."""
import torch
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as _ld

@triton.jit
def _route_fold_packed_kernel(lg_ptr, ids_ptr, w_ptr, M, stride_l,
                              E: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, BT: tl.constexpr, launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # bf16 logits only: fp32(bf16) has 16 zero low bits, so (order-preserving int32 key | (E-1-idx)) is a unique
    # sortable key -> one max-reduction per top-k round (ties -> lowest expert index), value recovered from the key.
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)
    rmask = rows < M
    cols = tl.arange(0, E)
    v = tl.load(lg_ptr + rows[:, None] * stride_l + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    b = v.to(tl.int32, bitcast=True)
    key = b ^ ((b >> 31) & 0x7FFFFFFF)
    key = (key & -65536) | (E - 1 - cols)[None, :]
    kidx = tl.arange(0, KP)
    sel_v = tl.zeros([BT, KP], dtype=tl.float32)
    sel_i = tl.zeros([BT, KP], dtype=tl.int32)
    for j in tl.static_range(K):
        kmax = tl.max(key, axis=1)
        i = (E - 1) - (kmax & 0xFFFF)
        hb = kmax & -65536
        vb = (hb ^ ((hb >> 31) & 0x7FFFFFFF)) & -65536
        val = vb.to(tl.float32, bitcast=True)
        sel_v = tl.where(kidx[None, :] == j, val[:, None], sel_v)
        sel_i = tl.where(kidx[None, :] == j, i[:, None], sel_i)
        key = tl.where(key == kmax[:, None], -2147483647 - 1, key)
    mx = tl.max(sel_v, axis=1)
    p = tl.where(kidx[None, :] < K, tl.exp(sel_v - mx[:, None]), 0.0)
    p = p / tl.sum(p, axis=1)[:, None]
    gs = tl.load(lg_ptr + rows * stride_l + E, mask=rmask, other=0.0).to(tl.float32)
    s = _ld.div_rn(1.0, 1.0 + _ld.exp(-gs))
    p = tl.where(kidx[None, :] == K, s[:, None], p)
    sel_i = tl.where(kidx[None, :] == K, E, sel_i)
    om = rmask[:, None] & (kidx[None, :] <= K)
    offs = rows[:, None] * (K + 1) + kidx[None, :]
    tl.store(ids_ptr + offs, sel_i, mask=om)
    tl.store(w_ptr + offs, p.to(w_ptr.dtype.element_ty), mask=om)



def route(logits: torch.Tensor):
    """BF16 logits [M,264] -> BF16 weights and int32 IDs [M,9]."""
    m = logits.shape[0]
    ids = torch.empty((m, 9), dtype=torch.int32, device=logits.device)
    weights = torch.empty((m, 9), dtype=torch.bfloat16, device=logits.device)
    if m:
        bt = 8 if m >= 4096 else 4
        _route_fold_packed_kernel[(triton.cdiv(m, bt),)](
            logits, ids, weights, m, logits.stride(0),
            E=256, K=8, KP=16, BT=bt, num_warps=4, launch_pdl=False,
        )
    return ids, weights
