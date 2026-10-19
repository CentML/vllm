# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in online FP8 for the GLM-5.3 linear layers a checkpoint keeps in BF16.

The GLM-5.3-Flash NVFP4 (ModelOpt) checkpoint leaves the attention projections
(KDA and DSA/MLA), the DSA indexer query projection and the MoE shared expert
in BF16. ``additional_config`` can run selected groups of them as FP8 GEMMs:

    {"glm53_fp8_linear": ["kda_in", "kda_out", "mla", "indexer", "shared_expert"],
     "glm53_fp8_linear_scheme": "per_channel"}

``glm53_fp8_linear`` is a list or a comma-separated string of the groups below
(or the aliases ``kda``, ``mla``, ``indexer``, ``all``). Empty / unset keeps
every layer in BF16 (default).

``glm53_fp8_linear_scheme`` (post-training quantization, no calibration data):

* ``per_channel`` (default): vLLM's ``Fp8PtpcOnlineLinearMethod``. The BF16
  weight is quantized to FP8 E4M3 once, while it is loaded, with one absmax
  scale per output channel; activations get a dynamic per-token FP8 scale on
  every call. The GEMM kernel comes from ``init_fp8_linear_kernel`` (CUTLASS
  ``cutlass_scaled_mm`` on SM10x) and can be overridden natively with
  ``kernel_config.linear_backend_per_quant = {"fp8_w8a8": ...}``.
* ``per_block``: vLLM's ``Fp8PerBlockOnlineLinearMethod``, 128x128 weight
  blocks and dynamic 1x128 activation groups (DeepGEMM / FlashInfer / CUTLASS
  block-scaled kernels). Not valid for ``kda_in``: its beta shard has 64 rows,
  so a 128-row weight block would mix beta and f_a rows.

The layers are created with vLLM's online quantization methods, so loading
works as for ``--quantization fp8_per_channel``: the weight is allocated on the
meta device, each shard's weight_loader call is buffered, and once every shard
of a layer has arrived the layer is materialized in BF16, loaded with the
original (merged / replicated-shard aware) weight loaders, quantized in
``process_weights_after_loading`` and the BF16 weight is released. Peak extra
memory is one layer's BF16 weight.

Never quantized here: the MoE router (``mlp.gate``, fp32), ``lm_head``, the
embeddings, the indexer ``wk_weights_proj`` (its fp32 head-weight half is read
from ``.weight``), ``kv_b_proj`` (absorbed into the W_UK / W_UV bmm weights at
load), the KDA conv weights and the MTP layer.
"""

import re
from typing import TYPE_CHECKING

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)

# group -> layers it covers
GLM53_FP8_LINEAR_GROUPS: dict[str, str] = {
    "kda_in": "KDA in_proj_qkvbfg_a (merged q|k|v|beta|f_a|g_a)",
    "kda_gate": "KDA f_b_proj and g_b_proj (gate up-projections, K=128)",
    "kda_out": "KDA o_proj",
    "mla_qkv_a": "DSA fused_qkv_a_proj (q_a | kv_a)",
    "mla_q_b": "DSA q_b_proj",
    "mla_out": "DSA o_proj",
    "indexer_q": "DSA indexer wq_b",
    "shared_expert": "MoE shared expert gate_up_proj and down_proj",
}

_ALIASES: dict[str, tuple[str, ...]] = {
    "kda": ("kda_in", "kda_out"),
    "mla": ("mla_qkv_a", "mla_q_b", "mla_out"),
    "indexer": ("indexer_q",),
    # kda_gate feeds the decay gate and gains little (K=128): name it explicitly.
    "all": tuple(g for g in GLM53_FP8_LINEAR_GROUPS if g != "kda_gate"),
}

_SCHEMES = ("per_channel", "per_block")

_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)(?:\.|$)")

# One online quantization config per scheme and process, shared by all layers.
_ONLINE_CONFIGS: dict[str, "QuantizationConfig"] = {}


def _parse_settings(vllm_config: VllmConfig) -> tuple[frozenset[str], str]:
    additional_config = vllm_config.additional_config
    if not isinstance(additional_config, dict):
        return frozenset(), "per_channel"
    raw = additional_config.get("glm53_fp8_linear") or []
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "additional_config.glm53_fp8_linear must be a list or a "
            f"comma-separated string, got {type(raw).__name__}"
        )
    groups: set[str] = set()
    for item in raw:
        name = str(item).strip()
        if not name:
            continue
        if name in _ALIASES:
            groups.update(_ALIASES[name])
        elif name in GLM53_FP8_LINEAR_GROUPS:
            groups.add(name)
        else:
            raise ValueError(
                f"Unknown glm53_fp8_linear group {name!r}; expected one of "
                f"{sorted(GLM53_FP8_LINEAR_GROUPS)} or aliases {sorted(_ALIASES)}"
            )
    scheme = str(additional_config.get("glm53_fp8_linear_scheme", "per_channel"))
    if scheme not in _SCHEMES:
        raise ValueError(
            f"Unknown glm53_fp8_linear_scheme {scheme!r}; expected one of {_SCHEMES}"
        )
    if scheme == "per_block" and "kda_in" in groups:
        raise ValueError(
            "glm53_fp8_linear_scheme=per_block cannot quantize kda_in: the "
            "merged KDA projection has a 64-row beta shard, so 128-row weight "
            "blocks would straddle shards. Use per_channel or drop kda_in."
        )
    return frozenset(groups), scheme


def _online_config(scheme: str) -> "QuantizationConfig":
    config = _ONLINE_CONFIGS.get(scheme)
    if config is None:
        from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
        from vllm.model_executor.layers.quantization.online.base import (
            OnlineQuantizationConfig,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8Static128BlockSym,
            kFp8StaticChannelSym,
        )

        weight_key = (
            kFp8StaticChannelSym if scheme == "per_channel" else kFp8Static128BlockSym
        )
        config = OnlineQuantizationConfig(
            QuantizationConfigArgs(linear=QuantSpec(weight=weight_key))
        )
        _ONLINE_CONFIGS[scheme] = config
    return config


def glm53_fp8_linear_quant_config(
    group: str,
    prefix: str,
    vllm_config: VllmConfig | None = None,
) -> "QuantizationConfig | None":
    """Return the online FP8 quant config for a layer of ``group`` at
    ``prefix`` when ``additional_config.glm53_fp8_linear`` selects that group,
    else None (the caller keeps its own quant config, i.e. BF16)."""
    assert group in GLM53_FP8_LINEAR_GROUPS, group
    if vllm_config is None:
        vllm_config = get_current_vllm_config()
    groups, scheme = _parse_settings(vllm_config)
    if group not in groups:
        return None
    # Main decoder layers only: the MTP layer sits at index >= num_hidden_layers
    # and is excluded by the checkpoint's own ignore list policy too.
    match = _LAYER_INDEX_RE.search(prefix)
    num_hidden_layers = getattr(
        vllm_config.model_config.hf_text_config, "num_hidden_layers", None
    )
    if match is None or (
        num_hidden_layers is not None and int(match.group(1)) >= num_hidden_layers
    ):
        return None
    logger.info_once(
        "GLM-5.3 online FP8 linear (%s weights, dynamic %s activations) for "
        "groups: %s. Router, lm_head, embeddings, kv_b_proj, indexer "
        "wk_weights_proj and MTP stay unquantized.",
        "per-output-channel" if scheme == "per_channel" else "128x128 block",
        "per-token" if scheme == "per_channel" else "1x128 group",
        ", ".join(sorted(groups)),
        scope="global",
    )
    return _online_config(scheme)


def glm53_shared_expert_quant_config(
    quant_config: "QuantizationConfig | None", prefix: str
) -> "QuantizationConfig | None":
    """Quant config for the shared-expert MLP at ``prefix``: the online FP8
    config when selected and the checkpoint keeps the shared expert in BF16,
    else ``quant_config`` unchanged."""
    fp8_config = glm53_fp8_linear_quant_config("shared_expert", prefix)
    if fp8_config is None:
        return quant_config
    if quant_config is not None:
        is_layer_excluded = getattr(quant_config, "is_layer_excluded", None)
        if is_layer_excluded is None or not all(
            is_layer_excluded(f"{prefix}.{name}")
            for name in ("gate_up_proj", "down_proj")
        ):
            raise ValueError(
                "glm53_fp8_linear=shared_expert needs a checkpoint whose shared "
                f"expert is unquantized (BF16); {prefix} is quantized by "
                f"{quant_config.get_name()}."
            )
    return fp8_config
