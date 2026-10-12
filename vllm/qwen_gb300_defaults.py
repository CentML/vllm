# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Built-in defaults for this branch's Qwen3.6 GB300 serving stack.

Each setting is applied with ``os.environ.setdefault`` before any other vLLM module
is imported, so an explicit environment value still wins (for example ``KEY=0``
turns a feature off). Build/cache directories default to ``$VLLM_CACHE_ROOT/qwen-gb300``.
"""

import os

_CACHE_ROOT = os.environ.get("VLLM_CACHE_ROOT") or os.path.join(
    os.path.expanduser(os.environ.get("XDG_CACHE_HOME", "~/.cache")), "vllm")

DEFAULTS: dict[str, str] = {
    # CUDA-graph capture hardening
    "CFIX": "1",
    "CFIX_RESET": "1",
    # GDN deferred state commit
    "GDN_STATE_COMMIT": "1",
    "GDN_STATE_COMMIT_BUILD_DIR": os.path.join(_CACHE_ROOT, "qwen-gb300", "gsc-build"),
    "GDN_STATE_COMMIT_GB": "1",
    "GDN_STATE_COMMIT_GB_QO": "1",
    "GDN_STATE_COMMIT_MINB": "4",
    # GDN step plan and spec/prefill overlap
    "GGM": "1",
    "GGM_LAZY": "1",
    "GGM_MDCHECK": "300",
    "GGM_MDREUSE": "1",
    "GGM_OG2": "1",
    "GGM_VSDIRECT": "1",
    "VLLM_GDN_FUSED_MD": "1",
    # GDN layer graphs
    "VLLM_GDNP_SPEC_OVERLAP": "2",
    "VLLM_GDN_LAYER_GRAPHS": "1",
    "VLLM_GDN_LAYER_GRAPHS_DEVICE_NB": "1",
    "VLLM_GDN_LAYER_GRAPHS_MODE": "layer",
    "VLLM_GDN_LAYER_GRAPHS_TPH": "4",
    "VLLM_GDN_LAYER_GRAPHS_VSF_KEY": "1",
    # GDN conv and chunked prefill
    "GDN_CONV_CUDA_BUILD_DIR": os.path.join(_CACHE_ROOT, "qwen-gb300", "gdn-conv-build"),
    "GDN_CONV_CUDA_MIXED_ADD": "0",
    "VLLM_GDN_CONV_CUDA": "1",
    "VLLM_GDN_FI_CP_POOL": "1",
    "VLLM_GDN_FI_VSPLIT": "1",
    "VLLM_GDN_FI_VSPLIT_V1": "1",
    "VLLM_GDN_VSPLIT_C1REORDER": "1",
    "VLLM_GDN_VSPLIT_CG0SPLIT": "1",
    # KV copy, prefix cache and Mamba checkpoints
    "VLLM_FUSED_KV_BLOCK_COPY_MULTI": "1",
    "VLLM_FUSED_KV_BLOCK_COPY_TABLE_CACHE": "1",
    "VLLM_MAMBA_DROP_SUPERSEDED_STATE": "1",
    "VLLM_MAMBA_PIN_OWN_CKPT": "1",
    "VLLM_MAMBA_TAIL_CKPT": "1",
    "VLLM_MAMBA_TAIL_CKPT_MAX_RUNNING": "60",
    "VLLM_SUFFIX_TOKEN_STAGING": "256",
    # Norm/quant fusion and MoE
    "F121": "1",
    "GS2_ROUTE": "1",
    "GS2_ROUTE_LOG": "1",
    "GS2_ROUTE_MAXN": "11520",
    "GS2_ROUTE_TAG": "gs2r3",
    "LCD2_BF16": "tiny",
    "LCD2_PDL": "1",
    "NQF": "1",
    "QGF_FIN": "1",
    "SEG_FOLD": "1",
    "VLLM_FLASHINFER_MOE_ROUTING_MODULE_CACHE": os.path.join(_CACHE_ROOT, "qwen-gb300", "fi-moe-modules"),
    "VLLM_MOE_FINALIZE_FOLD_MAX_M": "1000000",
    "VLLM_SEG_MODE": "scale",
    # Elementwise and glue kernels
    "EWS": "1",
    "EWS_QKV_TPP": "2",
    "GLUE_EWS_CUDA": "1",
    "GLUE_EWS_CUDA_BUILD_DIR": os.path.join(_CACHE_ROOT, "qwen-gb300", "ews-build"),
    "GLUE_EWS_CUDA_MIN_T": "1024",
    "GLUE_EWS_CUDA_PDL": "1",
    "GLUE_EWS_CUDA_PW_MIN_T": "256",
    "GLUE_EWS_NOGATE": "1",
    "GLUE_EWS_TPP_SPLIT": "1024",
    "GLUE_FNQ_F0": "1",
    "GLUE_FNQ_XB": "1",
    "GLUE_GMRR": "1",
    "GLUE_GNQ_CFG": "1,2048",
    "GLUE_GSC_QO": "1",
    "GLUE_LAZY": "1",
    "GLUE_NQRR": "1",
    # Attention (prefill routing, decode split-KV, prefill/decode overlap)
    "FI_DECODE_SPLITKV": "auto",
    "FMHA107": "1",
    "FMHA107_F107": "0",
    "FMHA107_LOG": "1",
    "FMHA107_MIN_KV": "0",
    "FMHA107_MIN_TOKENS": "512",
    "FMHA_GEN": "1",
    "FMHA_GEN_RULE": "gb300",
    "VLLM_ATTN_PD_OVERLAP": "1",
    "VLLM_ATTN_PD_WAVE_GATE": "1",
    # Programmatic dependent launch
    "LCD_FI_OVERLAY": "1",
    "LCD_PDL_TRITON": "1",
    "PFW": "1",
    "PFWS": "1",
    "PFWS_TRIG": "1",
    "PFW_TRIG": "1",
    # Spec decode and sampling
    "VLLM_DRAFT_LMH_DTYPE": "mxfp8",
    "VLLM_DRAFT_LMH_VOCAB": "0-98304,248044-248070",
    "VLLM_MTP_DRAFT_PREFILL_PRUNE": "1",
    "VLLM_SAMPLER_GRAPH": "1",
    "VLLM_SAMPLER_GRAPH_PRECAPTURE": "1-8",
    "VLLM_SAMPLER_GRAPH_PRECAPTURE_PARAMS": "temperature=1.0,top_k=20,top_p=0.95,presence_penalty=1.5",
    "VLLM_SAMPLER_STATE_DIRTY": "1",
    "VLLM_SPARSE_VERIFY_SAMPLING": "1",
    # Engine host path
    "FEH": "1",
    "VLLM_FEH_MEMO": "1",
}

for _key, _value in DEFAULTS.items():
    os.environ.setdefault(_key, _value)
