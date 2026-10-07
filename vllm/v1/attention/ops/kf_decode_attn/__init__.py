# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel Factory paged-FP8 decode/verify attention, SM107 (Qwen3.6-35B-A3B geometry).

vllm/third_party/kf_decode_attn/kernel.py
            the KF solution, verbatim (CuTe DSL kernel + its host glue)
runtime.py  warmup (AOT compile of every (Q, BL) form), per-layer launch (TVM-FFI);
            used by vllm/v1/attention/backends/flashinfer.py when VLLM_KF_DECODE_ATTN=1
"""
