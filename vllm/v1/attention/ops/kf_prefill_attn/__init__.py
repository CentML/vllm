# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel Factory paged-FP8 prefill attention for SM107 (Qwen3.6-35B-A3B geometry).

vllm/third_party/kf_prefill_attn/kernel.py
            the KF solution, verbatim (CuTe DSL kernel + its Python host planner)
planner.py  exact C++ port of that planner (CPU, built at warmup)
runtime.py  per-step plan, per-layer launch (TVM-FFI), warmup; used by
            vllm/v1/attention/backends/flashinfer.py when VLLM_KF_PREFILL_ATTN=1
"""
