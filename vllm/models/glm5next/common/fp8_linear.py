# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in FP8 for the GLM-5.3 linear layers a checkpoint keeps in BF16.

The GLM-5.3-Flash NVFP4 (ModelOpt) checkpoint leaves the attention projections
(KDA and DSA/MLA), the DSA indexer query projection and the MoE shared expert
in BF16. ``additional_config`` can run selected groups of them as FP8 GEMMs:

    {"glm53_fp8_linear": ["kda_in", "kda_out", "mla", "indexer", "shared_expert"],
     "glm53_fp8_linear_scheme": "hybrid_tensor",
     "glm53_fp8_linear_min_tokens": 2048}

``glm53_fp8_linear`` is a list or a comma-separated string of the groups below
(or the aliases ``kda``, ``mla``, ``indexer``, ``all``). Empty / unset keeps
every layer in BF16 (default).

``glm53_fp8_linear_scheme`` (post-training quantization, no calibration data):

* ``hybrid_tensor`` (default, recommended): the BF16 weight is kept and used
  unchanged for calls with fewer than ``glm53_fp8_linear_min_tokens`` tokens
  (default 2048), so decode and every CUDA-graph-captured size (<= 1024 by
  default) run exactly the stock BF16 GEMM. At load, an FP8 E4M3 copy with one
  static per-tensor scale (weight absmax / 448) is added; calls with at least
  ``min_tokens`` tokens (prefill / mixed steps, which run eagerly under
  breakable CUDA graphs) quantize the activation with a dynamic per-tensor
  scale computed on device (no host sync) and run ``torch._scaled_mm``
  (cuBLASLt) with BF16 output. Costs one extra byte per selected weight
  element (logged after loading). With tensor parallelism every rank uses the
  absmax of its own shard: each rank's GEMM is self-contained before the
  all-reduce, so local scales are exact.
* ``per_channel``: vLLM's ``Fp8PtpcOnlineLinearMethod`` for every M. The BF16
  weight is replaced by FP8 E4M3 with one absmax scale per output channel and
  activations get a dynamic per-token scale; CUTLASS ``cutlass_scaled_mm`` on
  SM10x. The BF16 weight is freed, but decode also runs in FP8.

Never quantized here: the MoE router (``mlp.gate``, fp32), ``lm_head``, the
embeddings, the indexer ``wk_weights_proj`` (its fp32 head-weight half is read
from ``.weight``), ``kv_b_proj`` (absorbed into the W_UK / W_UV bmm weights at
load), the KDA conv weights and the MTP layer.
"""

import re
from typing import cast

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)

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

_SCHEMES = ("hybrid_tensor", "per_channel")
_DEFAULT_MIN_TOKENS = 2048

_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)(?:\.|$)")

# One quant config per (scheme, min_tokens) and process, shared by all layers.
_CONFIGS: dict[tuple[str, int], QuantizationConfig] = {}

# hybrid_tensor: layers holding an extra FP8 weight copy and its bytes.
_HYBRID_FP8_COPY = {"layers": 0, "bytes": 0}


def _parse_settings(vllm_config: VllmConfig) -> tuple[frozenset[str], str, int]:
    additional_config = vllm_config.additional_config
    if not isinstance(additional_config, dict):
        return frozenset(), _SCHEMES[0], _DEFAULT_MIN_TOKENS
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
    scheme = str(additional_config.get("glm53_fp8_linear_scheme", _SCHEMES[0]))
    if scheme not in _SCHEMES:
        raise ValueError(
            f"Unknown glm53_fp8_linear_scheme {scheme!r}; expected one of {_SCHEMES}"
        )
    min_tokens = int(
        additional_config.get("glm53_fp8_linear_min_tokens", _DEFAULT_MIN_TOKENS)
    )
    if min_tokens < 1:
        raise ValueError(f"glm53_fp8_linear_min_tokens must be >= 1, got {min_tokens}")
    return frozenset(groups), scheme, min_tokens


@register_weight_loader_v2_supported_method
class Glm53HybridFp8LinearMethod(UnquantizedLinearMethod):
    """BF16 linear (stock ``UnquantizedLinearMethod``: same weight, loaders
    and GEMM) plus an FP8 per-tensor copy of the weight used for calls with at
    least ``min_tokens`` tokens.

    The branch is taken in Python on the token count. Graph capture records
    whichever branch its (static) size selects, and both branches are
    capture-safe; GLM-5.3 runs without torch.compile (breakable CUDA graphs).
    """

    # The FP8 copy is derived from the loaded BF16 weight.
    supports_pre_processed_weights = False

    def __init__(self, min_tokens: int) -> None:
        super().__init__()
        self.min_tokens = min_tokens

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        weight = layer.weight.data  # [N, K] bf16
        if weight.shape[0] % 16 or weight.shape[1] % 16:
            # torch._scaled_mm (cuBLASLt FP8) needs N and K divisible by 16;
            # e.g. the merged KDA in_proj is 3336 wide per rank at TP8. Such a
            # layer simply stays BF16.
            layer.weight_fp8 = None
            _HYBRID_FP8_COPY["skipped"] = _HYBRID_FP8_COPY.get("skipped", 0) + 1
            return
        fp8_max = get_fp8_min_max()[1]
        scale = (weight.abs().amax().float() / fp8_max).clamp_min(1e-12).reshape(1)
        qweight, _ = ops.scaled_fp8_quant(weight, scale=scale)
        first = getattr(layer, "weight_fp8", None) is None
        # [K, N] column-major view, the layout torch._scaled_mm wants for B.
        layer.register_buffer("weight_fp8", qweight.t(), persistent=False)
        layer.register_buffer("weight_fp8_scale", scale.reshape(()), persistent=False)
        if first:
            _HYBRID_FP8_COPY["layers"] += 1
            _HYBRID_FP8_COPY["bytes"] += qweight.numel() * qweight.element_size()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = x.numel() // x.shape[-1]
        if (
            num_tokens < self.min_tokens
            or envs.VLLM_BATCH_INVARIANT
            or getattr(layer, "weight_fp8", None) is None
        ):
            return super().apply(layer, x, bias)
        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        # Dynamic per-tensor scale: absmax reduced on device, no host sync.
        x_q, x_scale = ops.scaled_fp8_quant(x_2d)
        out = torch._scaled_mm(
            x_q,
            layer.weight_fp8,
            scale_a=x_scale.reshape(()),
            scale_b=layer.weight_fp8_scale,
            bias=bias,
            out_dtype=x.dtype,
        )
        return out.view(*x.shape[:-1], out.shape[-1])


class Glm53HybridFp8Config(QuantizationConfig):
    """Quant config handed only to the selected GLM-5.3 linear layers."""

    def __init__(self, min_tokens: int) -> None:
        super().__init__()
        self.min_tokens = min_tokens

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return cast(QuantizationMethods, "glm53_hybrid_fp8")

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "Glm53HybridFp8Config":
        raise NotImplementedError("set via additional_config.glm53_fp8_linear")

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return Glm53HybridFp8LinearMethod(self.min_tokens)
        return None


def _quant_config(scheme: str, min_tokens: int) -> QuantizationConfig:
    key = (scheme, min_tokens if scheme == "hybrid_tensor" else 0)
    config = _CONFIGS.get(key)
    if config is None:
        if scheme == "hybrid_tensor":
            config = Glm53HybridFp8Config(min_tokens)
        else:
            from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
            from vllm.model_executor.layers.quantization.online.base import (
                OnlineQuantizationConfig,
            )
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                kFp8StaticChannelSym,
            )

            config = OnlineQuantizationConfig(
                QuantizationConfigArgs(linear=QuantSpec(weight=kFp8StaticChannelSym))
            )
        _CONFIGS[key] = config
    return config


def glm53_fp8_linear_quant_config(
    group: str,
    prefix: str,
    vllm_config: VllmConfig | None = None,
) -> QuantizationConfig | None:
    """Return the FP8 quant config for a layer of ``group`` at ``prefix`` when
    ``additional_config.glm53_fp8_linear`` selects that group, else None (the
    caller keeps its own quant config, i.e. BF16)."""
    assert group in GLM53_FP8_LINEAR_GROUPS, group
    if vllm_config is None:
        vllm_config = get_current_vllm_config()
    groups, scheme, min_tokens = _parse_settings(vllm_config)
    if group not in groups:
        return None
    # Main decoder layers only: the MTP layer sits at index >= num_hidden_layers.
    match = _LAYER_INDEX_RE.search(prefix)
    num_hidden_layers = getattr(
        vllm_config.model_config.hf_text_config, "num_hidden_layers", None
    )
    if match is None or (
        num_hidden_layers is not None and int(match.group(1)) >= num_hidden_layers
    ):
        return None
    if scheme == "hybrid_tensor":
        logger.info_once(
            "GLM-5.3 FP8 linear (hybrid_tensor) for groups: %s. Calls with >= %d "
            "tokens use FP8 (static per-tensor weight, dynamic per-tensor "
            "activation, torch._scaled_mm); smaller calls use the BF16 weight. "
            "Router, lm_head, embeddings, kv_b_proj, indexer wk_weights_proj "
            "and MTP stay unquantized.",
            ", ".join(sorted(groups)),
            min_tokens,
            scope="global",
        )
    else:
        logger.info_once(
            "GLM-5.3 FP8 linear (per_channel: per-output-channel weights, "
            "dynamic per-token activations) for groups: %s. Router, lm_head, "
            "embeddings, kv_b_proj, indexer wk_weights_proj and MTP stay "
            "unquantized.",
            ", ".join(sorted(groups)),
            scope="global",
        )
    return _quant_config(scheme, min_tokens)


def glm53_shared_expert_quant_config(
    quant_config: QuantizationConfig | None, prefix: str
) -> QuantizationConfig | None:
    """Quant config for the shared-expert MLP at ``prefix``: the FP8 config
    when selected and the checkpoint keeps the shared expert in BF16, else
    ``quant_config`` unchanged."""
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


def log_glm53_fp8_linear_memory() -> None:
    """Log the extra memory of the hybrid_tensor FP8 weight copies (once,
    after loading)."""
    if _HYBRID_FP8_COPY["layers"]:
        logger.info_once(
            "GLM-5.3 FP8 linear (hybrid_tensor): %d layers keep an FP8 weight "
            "copy next to BF16, +%.2f GiB on this GPU.",
            _HYBRID_FP8_COPY["layers"],
            _HYBRID_FP8_COPY["bytes"] / 2**30,
            scope="local",
        )
    if _HYBRID_FP8_COPY.get("skipped"):
        logger.info_once(
            "GLM-5.3 FP8 linear (hybrid_tensor): %d layers stay BF16 (weight "
            "dims not divisible by 16 for torch._scaled_mm).",
            _HYBRID_FP8_COPY["skipped"],
            scope="local",
        )
