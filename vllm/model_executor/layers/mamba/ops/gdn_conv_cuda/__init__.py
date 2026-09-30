# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA (sm_107a) fused conv1d + SiLU + q/k l2norm + gating for GDN prefill chunks.

Bit-exact drop-in for qwen_gdn_linear_attn.gdn_fused_conv_post_conv (Triton v2 kernel). JIT-built with
torch.utils.cpp_extension from gdn_conv_cuda.cu (next to this file) into $GDN_CONV_CUDA_BUILD_DIR
(default ~/.cache/gdn_conv_cuda; compiled on first use, cached afterwards). Knobs: GDN_CONV_CUDA_TPH
(tokens per half-warp: 4/8/16, default 8 -> 64-token CTAs; the vLLM call site passes its own value from
VLLM_GDN_CONV_CUDA_TPH), GDN_CONV_CUDA_MIXED_ADD (1 = sm_107f add.rn.f32x2.bf16x2.f32x2, 0 = unpack +
add.f32).
"""
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_exts = {}
TPH = int(os.environ.get("GDN_CONV_CUDA_TPH", "8"))
RV = int(os.environ.get("GDN_CONV_CUDA_RV", "0"))  # l2 reduce-tree variant matching Triton (validated: 0)
STATS = {"calls": 0, "fallbacks": 0}


def load(mixed_add=None):
    if mixed_add is None:
        mixed_add = int(os.environ.get("GDN_CONV_CUDA_MIXED_ADD", "1"))
    if mixed_add in _exts:
        return _exts[mixed_add]
    import torch.utils.cpp_extension as cpp

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}{'a' if major >= 9 else ''}"
    if mixed_add and (major, minor) != (10, 7):
        mixed_add = 0
    build = os.path.join(
        os.environ.get("GDN_CONV_CUDA_BUILD_DIR", os.path.expanduser("~/.cache/gdn_conv_cuda")),
        f"sm{arch}_ma{mixed_add}",
    )
    os.makedirs(build, exist_ok=True)
    orig = cpp._get_cuda_arch_flags
    cpp._get_cuda_arch_flags = lambda cflags=None: [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    try:
        ext = cpp.load(
            name=f"_gdn_conv_cuda_ma{mixed_add}",
            sources=[os.path.join(_HERE, "gdn_conv_cuda.cu")],
            # no fast-math, no FMA contraction: every fp op is explicit PTX mirroring the Triton kernel
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo", "-fmad=false", f"-DGK2_MIXED_ADD={mixed_add}"],
            extra_cflags=["-O3", "-std=c++20"],
            build_directory=build,
            verbose=False,
        )
    finally:
        cpp._get_cuda_arch_flags = orig
    _exts[mixed_add] = ext
    return ext


def fused_conv_post_conv(x, conv_weights, conv_state, cache_indices, has_initial_state, cu_seqlens,
                         num_seqs, a, b, A_log, dt_bias, num_k_heads, head_k_dim, head_v_dim,
                         tph=None, rv=None, mixed_add=None):
    """Same contract/outputs as gdn_fused_conv_post_conv; returns None if the fast-path contract is not met."""
    H, K, V = num_k_heads, head_k_dim, head_v_dim
    HV = A_log.shape[0]
    if K != 128 or V != 128 or conv_weights.shape[1] != 4 or x.dim() != 2:
        STATS["fallbacks"] += 1
        return None
    P = x.shape[0]
    q = torch.empty(P, H, K, dtype=x.dtype, device=x.device)
    k = torch.empty(P, H, K, dtype=x.dtype, device=x.device)
    v = torch.empty(P, HV, V, dtype=x.dtype, device=x.device)
    g = torch.empty(P, HV, dtype=torch.float32, device=x.device)
    beta = torch.empty(P, HV, dtype=torch.float32, device=x.device)
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    ok = load(mixed_add).run(
        x, conv_weights, conv_state, cache_indices.contiguous(), has_initial_state.contiguous(),
        cu_seqlens.contiguous(), int(num_seqs), a, b, A_log, dt_bias, q, k, v, g, beta, int(H),
        int(TPH if tph is None else tph), int(RV if rv is None else rv))
    if not ok:
        STATS["fallbacks"] += 1
        return None
    STATS["calls"] += 1
    return q, k, v, g, beta
