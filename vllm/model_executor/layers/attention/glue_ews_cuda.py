# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gb300 glue: CUDA QKV prologue (glue_ews_cuda.cu, one warp per (token, head)), bit-identical to
fused_qkv_prologue._ews_qkv_kernel. Enabled by GLUE_EWS_CUDA=1 (default off); JIT-built with
torch.utils.cpp_extension for the device arch into $GLUE_EWS_CUDA_BUILD_DIR (default ~/.cache/glue_ews_cuda).
Supported: bf16 qkv, head_dim 256, rotary_dim 64, Gemma (w + 1) norms, int64 positions ([T] or M-RoPE [3, T]),
fp8 e4m3 paged KV cache; anything else keeps the Triton kernel."""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("GLUE_EWS_CUDA", "0") == "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None


def load():
    global _ext
    if _ext is not None:
        return _ext
    import torch.utils.cpp_extension as cpp

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    build = os.path.join(
        os.environ.get("GLUE_EWS_CUDA_BUILD_DIR", os.path.expanduser("~/.cache/glue_ews_cuda")), f"sm{arch}_v2"
    )
    os.makedirs(build, exist_ok=True)
    orig = cpp._get_cuda_arch_flags
    cpp._get_cuda_arch_flags = lambda cflags=None: [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    try:
        _ext = cpp.load(
            name="_glue_ews_cuda_v2",
            sources=[os.path.join(_HERE, "glue_ews_cuda.cu")],
            # no fast-math, no FMA contraction: every op is written explicitly to match the Triton PTX
            extra_cuda_cflags=["-O3", "-std=c++20", "-fmad=false", "-lineinfo"],
            extra_cflags=["-O3", "-std=c++20"],
            build_directory=build,
            verbose=False,
        )
    finally:
        cpp._get_cuda_arch_flags = orig
    logger.info_once("[glue] CUDA QKV prologue loaded (GLUE_EWS_CUDA)")
    return _ext


def supported(qkv, head_dim, rotary_dim, norm_beta, positions, k_cache, cos_sin_cache=None) -> bool:
    return (
        (cos_sin_cache is None or (cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.shape[-1] == 64
                                   and cos_sin_cache.stride(-1) == 1))
        and qkv.dtype == torch.bfloat16
        and qkv.stride(-1) == 1
        and qkv.stride(0) % 8 == 0
        and head_dim == 256
        and rotary_dim == 64
        and norm_beta == 1.0
        and positions.dtype == torch.int64
        and (k_cache is None or k_cache.element_size() == 1)
    )


def launch(qkv, positions, q_weight, k_weight, cos_sin_cache, eps, num_q_heads, num_kv_heads, mrope_section,
           q_scale, k_scale, v_scale, slot_mapping, k_cache, v_cache, gate_copy=True, pdl=False):
    """Same results as fused_qkv_prologue.launch: (q_fp8 [T, H*D], k_out bf16 [T, KVH*D], gate bf16 [T, H*D] or
    None when gate_copy is False); writes K/V to the cache when slot_mapping is not None.
    pdl (gb300-fuse, GLUE_EWS_CUDA_PDL): launch with programmatic stream serialization; the kernel waits
    (griddepcontrol.wait) before any global access and triggers its dependents after its last qkv read."""
    T = qkv.shape[0]
    dev = qkv.device
    D = 256
    q8 = torch.empty((T, num_q_heads * D), dtype=torch.float8_e4m3fn, device=dev)
    k_out = torch.empty((T, num_kv_heads * D), dtype=qkv.dtype, device=dev)
    gate = torch.empty((T, num_q_heads * D), dtype=qkv.dtype, device=dev) if gate_copy else None
    if T == 0:
        return q8, k_out, gate
    mh = mw = 0
    if positions.ndim == 2:
        mh, mw = mrope_section[1], mrope_section[2]
    kc = vc = None
    if slot_mapping is not None:
        kc = k_cache.view(torch.uint8)
        vc = v_cache.view(torch.uint8)
    load().launch(qkv, positions, q_weight, k_weight, cos_sin_cache, float(eps), num_q_heads, num_kv_heads, mh, mw,
                  q_scale, k_scale, v_scale, slot_mapping, kc, vc, q8.view(torch.uint8), k_out, gate, bool(pdl))
    return q8, k_out, gate
