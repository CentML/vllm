# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fusion of an (add +) RMSNorm into its MXFP8 activation quant.

Before vLLM IR lowering, a chain

    x = aten.add(a, b)                                  # optional, e.g. MoE combine
    y, r = vllm_ir.fused_add_rms_norm(x, residual, w.float() + 1.0, eps)
    q, sf = vllm.mxfp8_quantize(view(y), is_sf_swizzled_layout=True)

becomes one ``vllm.add_rms_norm_mxfp8_quant(a, b, residual, w, eps, 1.0, ...)``.
When one summand is the deferred Qwen shared-expert gate
``apply_shared_expert_gate(g, shared)``, the gate chain is absorbed as well
and the op receives ``shared`` and ``g``. When the other summand is
``vllm.moe_finalize`` of a deferred MoE output used only here, the reduction
moves into the norm as well (``vllm.moe_finalize_add_rms_norm_mxfp8_quant``),
so the routed output is never written.
Its e4m3 values and swizzled scales are bit-identical to what
``mxfp8_quantize`` produces from the fused op's own bf16 output, so every
swizzled MXFP8 consumer is unaffected. ``y`` is still written when anything
other than the quant reads it (e.g. the bf16 GDN ``in_proj_ba``).

Only norm inputs that are materialized anyway (graph inputs, multi-output op
results, extern GEMMs) are accepted, so no Inductor fusion into the norm (e.g.
the embedding gather) is traded for the opaque op.
"""

import operator
from typing import Any

import torch
from torch import fx
from torch._guards import detect_fake_mode
from torch._inductor.pattern_matcher import fwd_only
from torch._ops import OpOverload
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

import vllm.model_executor.layers.quantization.utils.mxfp8_utils  # noqa: F401
from vllm import ir
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.rms_norm_mxfp8_quant import MXFP8_BLOCK
from vllm.model_executor.layers.fusion.shared_expert_gate import (
    apply_shared_expert_gate,
)

from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

aten = torch.ops.aten
_QUANT = torch.ops.vllm.mxfp8_quantize.default
_FUSED = torch.ops.vllm.add_rms_norm_mxfp8_quant.default
_FUSED_MOE = torch.ops.vllm.moe_finalize_add_rms_norm_mxfp8_quant.default
_MOE_FINALIZE = torch.ops.vllm.moe_finalize.default
_FUSED_ADD_RMS_NORM = ir.ops.fused_add_rms_norm.torch_op
_RMS_NORM = ir.ops.rms_norm.torch_op
_VIEWS = (aten.view.default, aten.reshape.default, aten._unsafe_view.default)
_TO_DTYPE = (torch.ops.prims.convert_element_type.default, aten._to_copy.default)
_EXTERN_ATEN = (aten.mm.default, aten.addmm.default, aten.bmm.default)


def _val(node: fx.Node) -> Any:
    return node.meta.get("val")


def _same_shape(a: fx.Node, b: fx.Node) -> bool:
    sa, sb = _val(a).shape, _val(b).shape
    return len(sa) == len(sb) and all(
        statically_known_true(x == y) for x, y in zip(sa, sb)
    )


def _is_noop_view(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target in _VIEWS
        and isinstance(node.args[0], fx.Node)
        and _same_shape(node, node.args[0])
    )


def _through_views(node: fx.Node) -> tuple[fx.Node, list[fx.Node]]:
    """Walk up same-shape views; return the source and the views, nearest first."""
    views = []
    while _is_noop_view(node):
        views.append(node)
        node = node.args[0]
    return node, views


def _materialized(node: fx.Node) -> bool:
    """Whether Inductor writes ``node`` to memory regardless of this fusion."""
    if node.op == "placeholder":
        return True
    if node.op != "call_function":
        return False
    if node.target is operator.getitem or node.target in _EXTERN_ATEN:
        return True
    return isinstance(node.target, OpOverload) and node.target.namespace not in (
        "aten",
        "prims",
    )


def _is_bf16_2d(node: Any) -> bool:
    val = _val(node) if isinstance(node, fx.Node) else None
    return (
        isinstance(val, torch.Tensor)
        and val.dtype == torch.bfloat16
        and val.dim() == 2
        and statically_known_true(val.stride(-1) == 1)
    )


def _arg(node: fx.Node, idx: int, name: str, default: Any) -> Any:
    if len(node.args) > idx:
        return node.args[idx]
    return node.kwargs.get(name, default)


def _getitems(node: fx.Node) -> dict[int, fx.Node] | None:
    """Map output index -> getitem user, or None if used any other way."""
    items: dict[int, fx.Node] = {}
    for user in node.users:
        if user.target is not operator.getitem or user.args[1] in items:
            return None
        items[user.args[1]] = user
    return items


def _is_swizzled_quant(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is _QUANT
        and _arg(node, 1, "is_sf_swizzled_layout", False) is True
        # Any alignment <= 32 pads nothing when hidden % 32 == 0.
        and _arg(node, 2, "alignment", 0) in (0, 32)
        and _getitems(node) is not None
    )


def _gemma_weight(weight: fx.Node) -> fx.Node | None:
    """Return w if ``weight`` is ``w.float() + 1`` (GemmaRMSNorm), else None."""
    if not (
        weight.op == "call_function"
        and weight.target is aten.add.Tensor
        and not weight.kwargs
        and len(weight.args) == 2
        and isinstance(weight.args[0], fx.Node)
        and not isinstance(weight.args[1], (fx.Node, bool))
        and weight.args[1] == 1
    ):
        return None
    cast = weight.args[0]
    if cast.op != "call_function" or cast.target not in _TO_DTYPE:
        return None
    if _val(cast).dtype != torch.float32 or not isinstance(cast.args[0], fx.Node):
        return None
    return cast.args[0]


def _shared_gate_matcher() -> SubgraphMatcher:
    """Matcher for ``apply_shared_expert_gate(g, s)`` in a post-grad graph.

    The pattern is traced from the helper the MoE runner emits, with the same
    decompositions, so the two stay in sync.
    """
    example = [
        torch.empty(8, 1, dtype=torch.bfloat16),
        torch.empty(8, 64, dtype=torch.bfloat16),
    ]
    pattern = fwd_only(apply_shared_expert_gate, example).graph
    pattern.eliminate_dead_code()
    return SubgraphMatcher(
        pattern,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=False,
    )


class RMSNormMxfp8QuantFusionPass(VllmInductorPass):
    """Replace (add +) RMSNorm -> swizzled mxfp8_quantize with one fused op."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self._gate_matcher = _shared_gate_matcher()

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = 0
        self._erased: set[fx.Node] = set()
        self._gates = self._find_shared_gates(graph)
        for quant in list(graph.find_nodes(op="call_function", target=_QUANT)):
            if quant not in self._erased and self._fuse(graph, quant):
                self.matched_count += 1
        logger.debug(
            "%s fused %d norm -> MXFP8 quant chains",
            self.pass_name,
            self.matched_count,
        )

    def _find_shared_gates(
        self, graph: fx.Graph
    ) -> dict[fx.Node, tuple[fx.Node, fx.Node, list[fx.Node]]]:
        """Map each gated shared-expert output to (shared, gate logits, chain)."""
        gates = {}
        for match in self._gate_matcher.match(graph):
            (out,) = match.returning_nodes
            gate, shared = match.placeholder_nodes
            chain = [
                node
                for pattern_node, node in match.nodes_map.items()
                if pattern_node.op not in ("placeholder", "output")
            ]
            g_val = _val(gate)
            if (
                len(out.users) == 1
                and _is_bf16_2d(shared)
                and isinstance(g_val, torch.Tensor)
                and g_val.dtype == torch.bfloat16
                and g_val.dim() == 2
                and statically_known_true(g_val.shape[1] == 1)
                and statically_known_true(g_val.shape[0] == _val(shared).shape[0])
            ):
                gates[out] = (shared, gate, chain)
        return gates

    def _erase(self, graph: fx.Graph, node: fx.Node) -> None:
        graph.erase_node(node)
        self._erased.add(node)

    def _fuse(self, graph: fx.Graph, quant: fx.Node) -> bool:
        if not _is_swizzled_quant(quant) or not isinstance(quant.args[0], fx.Node):
            return False
        normed, _ = _through_views(quant.args[0])
        if normed.op != "call_function":
            return False
        if normed.target is operator.getitem and normed.args[1] == 0:
            norm = normed.args[0]
            if norm.target is not _FUSED_ADD_RMS_NORM or len(norm.args) < 4:
                return False
            x, residual, weight, eps = norm.args[:4]
            variance_size = _arg(norm, 4, "variance_size", None)
            norm_items = _getitems(norm)
        elif normed.target is _RMS_NORM:
            norm = normed
            if len(norm.args) < 3:
                return False
            x, weight, eps = norm.args[:3]
            residual = None
            variance_size = _arg(norm, 3, "variance_size", None)
            norm_items = {0: norm}
        else:
            return False
        if variance_size is not None or norm_items is None:
            return False
        if not (_is_bf16_2d(x) and _is_bf16_2d(normed)):
            return False
        hidden = _val(x).shape[-1]
        if not isinstance(hidden, int) or hidden % MXFP8_BLOCK:
            return False
        if residual is not None and not (
            _is_bf16_2d(residual) and _same_shape(residual, x)
        ):
            return False

        # Absorb a single-use add feeding the norm (the MoE shared + routed
        # combine), which Inductor would otherwise fuse into the norm. When the
        # shared summand is the deferred shared-expert gate, the kernel applies
        # the gate as well.
        x_src, x_views = _through_views(x)
        add, x2, gate = None, None, None
        gate_chain: list[fx.Node] = []
        if (
            x_src.op == "call_function"
            and x_src.target is aten.add.Tensor
            and not x_src.kwargs
            and len(x_src.users) == 1
            and all(len(v.users) == 1 for v in x_views)
            and all(isinstance(arg, fx.Node) for arg in x_src.args)
        ):
            a, b = (_through_views(arg)[0] for arg in x_src.args)
            if all(_is_bf16_2d(t) and _same_shape(t, x) for t in (a, b)):
                add, x_src, x2 = x_src, a, b
                for summand, other in ((a, b), (b, a)):
                    matched = self._gates.get(summand)
                    if matched is not None and _same_shape(matched[0], x):
                        x_src, (x2, gate, gate_chain) = other, matched
                        break
        if not all(_materialized(t) for t in (x_src, x2, gate) if t is not None):
            return False
        if not (_is_bf16_2d(x_src) and _same_shape(x_src, x)):
            return False

        if not isinstance(weight, fx.Node) or not isinstance(
            _val(weight), torch.Tensor
        ):
            return False
        gemma_weight = _gemma_weight(weight)
        if gemma_weight is not None:
            weight, weight_offset = gemma_weight, 1.0
        elif _val(weight).dtype == torch.float32 and _materialized(weight):
            weight_offset = 0.0
        else:
            # A low-precision weight rounds x before the multiply; unsupported.
            return False
        w_val = _val(weight)
        if (
            w_val.dim() != 1
            or not statically_known_true(w_val.shape[0] == hidden)
            or not w_val.is_contiguous()
        ):
            return False

        # All swizzled quants of this normed tensor share the fused outputs.
        quants: list[tuple[fx.Node, dict[int, fx.Node]]] = []
        normed_views: list[fx.Node] = []
        store_normed = False
        stack = [normed]
        while stack:
            for user in stack.pop().users:
                items = _getitems(user) if _is_swizzled_quant(user) else None
                if items is not None:
                    quants.append((user, items))
                elif _is_noop_view(user):
                    normed_views.append(user)
                    stack.append(user)
                else:
                    store_normed = True

        # A deferred MoE reduction read only by this norm moves into the op.
        target, finalize = _FUSED, None
        if (
            x_src.op == "call_function"
            and x_src.target is _MOE_FINALIZE
            and len(x_src.users) == 1
            and all(isinstance(a, fx.Node) for a in x_src.args)
        ):
            target, finalize = _FUSED_MOE, x_src
        x_args = (x_src,) if finalize is None else tuple(finalize.args)
        args = (*x_args, x2, residual, weight, eps, weight_offset, store_normed, gate)
        fake_mode = detect_fake_mode(
            [_val(n) for n in (*x_args, x2, residual, weight, gate) if n is not None]
        )
        if fake_mode is None:
            return False
        with fake_mode:
            fake = target(*(_val(a) if isinstance(a, fx.Node) else a for a in args))
        with graph.inserting_before(norm):
            fused = graph.call_function(target, args)
            fused.meta["val"] = fake
            outs = []
            for i, val in enumerate(fake):
                out = graph.call_function(operator.getitem, (fused, i))
                out.meta["val"] = val
                outs.append(out)

        for q, items in quants:
            for idx, item in items.items():
                item.replace_all_uses_with(outs[2 + idx])
                self._erase(graph, item)
            self._erase(graph, q)
        if residual is not None and 1 in norm_items:
            norm_items[1].replace_all_uses_with(outs[1])
        if store_normed:
            normed.replace_all_uses_with(outs[0])

        # Drop the nodes the fused op made dead, consumers first.
        order = {n: i for i, n in enumerate(graph.nodes)}
        dead = sorted(normed_views, key=order.__getitem__, reverse=True)
        dead += [*norm_items.values(), norm, *x_views]
        if add is not None:
            dead.append(add)
        if finalize is not None:
            dead.append(finalize)
        dead += sorted(gate_chain, key=order.__getitem__, reverse=True)
        for node in dict.fromkeys(dead):
            if node not in self._erased and not node.users:
                self._erase(graph, node)
        return True
