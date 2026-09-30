# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fusion of the attention output gate into o_proj's MXFP8 quant.

A chain

    y = aten.mul(attn, aten.sigmoid(gate))      # [T, K] or [T, H, D]
    q, sf = vllm.mxfp8_quantize(view(y), is_sf_swizzled_layout=True)

becomes one ``vllm.attn_gate_mxfp8_quant(attn, gate)``, whose e4m3 values and
swizzled scales are bit-identical to the quant of the Inductor gate kernel's
bf16 output. ``gate`` may stay a strided view of the QKV projection. Only
chains whose bf16 product feeds nothing else are fused, and only when both
inputs are materialized anyway, so no Inductor fusion is traded away.
"""

import math
import operator

import torch
from torch import fx
from torch._guards import detect_fake_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true

import vllm.model_executor.layers.fusion.attn_gate_mxfp8_quant  # noqa: F401
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import MXFP8_BLOCK

from ..vllm_inductor_pass import VllmInductorPass
from .rms_norm_mxfp8_fusion import (
    _QUANT,
    _getitems,
    _is_swizzled_quant,
    _materialized,
    _val,
)

logger = init_logger(__name__)

aten = torch.ops.aten
_FUSED = torch.ops.vllm.attn_gate_mxfp8_quant.default
_RESHAPES = (aten.view.default, aten.reshape.default, aten._unsafe_view.default)
# Pure views a strided gate may be taken through (e.g. q_gate[:, :, 1]).
_VIEW_OPS = (
    *_RESHAPES,
    aten.slice.Tensor,
    aten.select.int,
    aten.as_strided.default,
    aten.alias.default,
    aten.unsqueeze.default,
    aten.squeeze.dim,
)


def _view_base(node: fx.Node) -> fx.Node:
    while (
        node.op == "call_function"
        and node.target in _VIEW_OPS
        and isinstance(node.args[0], fx.Node)
    ):
        node = node.args[0]
    return node


def _bf16(node: fx.Node) -> torch.Tensor | None:
    val = _val(node)
    if isinstance(val, torch.Tensor) and val.dtype == torch.bfloat16:
        return val
    return None


def _same_shape(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dim() == b.dim() and all(
        statically_known_true(x == y) for x, y in zip(a.shape, b.shape)
    )


class AttnGateMxfp8QuantFusionPass(VllmInductorPass):
    """Replace attn * sigmoid(gate) -> swizzled mxfp8_quantize with one op."""

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = 0
        for quant in list(graph.find_nodes(op="call_function", target=_QUANT)):
            if self._fuse(graph, quant):
                self.matched_count += 1
        logger.debug(
            "%s fused %d attention gate -> MXFP8 quant chains",
            self.pass_name,
            self.matched_count,
        )

    def _fuse(self, graph: fx.Graph, quant: fx.Node) -> bool:
        if not _is_swizzled_quant(quant) or not isinstance(quant.args[0], fx.Node):
            return False
        x_val = _bf16(quant.args[0])
        if x_val is None or x_val.dim() != 2:
            return False
        # Single-use reshapes between the product and the quant.
        views: list[fx.Node] = []
        mul = quant.args[0]
        while (
            mul.op == "call_function"
            and mul.target in _RESHAPES
            and isinstance(mul.args[0], fx.Node)
            and len(mul.users) == 1
        ):
            views.append(mul)
            mul = mul.args[0]
        if not (
            mul.op == "call_function"
            and mul.target is aten.mul.Tensor
            and not mul.kwargs
            and len(mul.users) == 1
            and len(mul.args) == 2
            and all(isinstance(a, fx.Node) for a in mul.args)
        ):
            return False
        sig_idx = next(
            (
                i
                for i, a in enumerate(mul.args)
                if a.target is aten.sigmoid.default and len(a.users) == 1
            ),
            None,
        )
        if sig_idx is None:
            return False
        sig = mul.args[sig_idx]
        attn, gate = mul.args[1 - sig_idx], sig.args[0]
        if not isinstance(gate, fx.Node):
            return False
        y_val, a_val, g_val = _bf16(mul), _bf16(attn), _bf16(gate)
        if y_val is None or a_val is None or g_val is None:
            return False
        if y_val.dim() not in (2, 3):
            return False
        if not (_same_shape(a_val, y_val) and _same_shape(g_val, y_val)):
            return False
        hidden = math.prod(y_val.shape[1:])
        if not isinstance(hidden, int) or hidden % (4 * MXFP8_BLOCK):
            return False
        if not statically_known_true(x_val.shape[-1] == hidden):
            return False
        if not (
            statically_known_true(a_val.is_contiguous())
            and statically_known_true(g_val.stride(-1) == 1)
        ):
            return False
        if not (_materialized(_view_base(attn)) and _materialized(_view_base(gate))):
            return False
        items = _getitems(quant)
        if items is None:
            return False

        fake_mode = detect_fake_mode([a_val, g_val])
        if fake_mode is None:
            return False
        with fake_mode:
            fake = _FUSED(a_val, g_val)
        with graph.inserting_before(quant):
            fused = graph.call_function(_FUSED, (attn, gate))
            fused.meta["val"] = fake
            outs = []
            for i, val in enumerate(fake):
                out = graph.call_function(operator.getitem, (fused, i))
                out.meta["val"] = val
                outs.append(out)
        for idx, item in items.items():
            item.replace_all_uses_with(outs[idx])
            graph.erase_node(item)
        graph.erase_node(quant)
        for node in (*views, mul, sig):
            if not node.users:
                graph.erase_node(node)
        return True
