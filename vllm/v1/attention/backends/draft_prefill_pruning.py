# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in pruning of the MTP draft layer's attention in draft prefill.

In a draft prefill that does not run as a FULL CUDA graph (i.e. every mixed
prefill + decode iteration), the MTP draft model computes attention for every
scheduled token, but only one row per request is ever used: the hidden state
at ``speculator.last_token_indices`` is sampled and propagated, and the MTP
layer is the only layer of the draft model. With pruning enabled, the draft
layer's FlashInfer attention computes just those rows with a single
trtllm-gen decode launch (``q_len_per_req=1``, ``seq_len = position + 1``) and
zero-fills the other rows. KV-cache writes are a separate op and still cover
every token, so later draft steps and iterations see a complete cache.

Numerics: float-order differences on the sampled rows (different kernel and
split), draft tokens only; the target model's distribution is unchanged
(drafts are verified by rejection sampling).

Only FlashInfer attention implementations honour the flag; configurations the
pruned path does not support (cascade attention, NVFP4 KV cache, DCP, fused
output quantization, non-16-bit outputs, unknown KV-cache group) fall back to
the regular forward.

Environment:
    VLLM_MTP_DRAFT_PREFILL_PRUNE=1         enable (default off)
    VLLM_MTP_DRAFT_PREFILL_PRUNE_CHECK=N   for the first N pruned calls also
                                           run the regular forward and log the
                                           max abs difference on the pruned
                                           rows (default 0)
    VLLM_MTP_DRAFT_PREFILL_PRUNE_CHECK_EVERY=K
                                           also check every K-th pruned call
                                           once the speculator is captured
                                           (default 0 = off)
"""

import contextlib
import os
from typing import Any

import torch
import torch.nn as nn

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_MTP_DRAFT_PREFILL_PRUNE", "0") == "1"
_CHECK = int(os.environ.get("VLLM_MTP_DRAFT_PREFILL_PRUNE_CHECK", "0") or 0)
_CHECK_EVERY = int(os.environ.get("VLLM_MTP_DRAFT_PREFILL_PRUNE_CHECK_EVERY", "0") or 0)


class _Ctx:
    active = False
    spec: Any = None
    num_reqs = 0
    calls = 0
    fallbacks = 0
    checked = 0
    ready = False


CTX = _Ctx()

# layer name -> KV-cache group id of the draft attention layer.
_GROUP_IDS: dict[str, int | None] = {}


def install(draft_model: nn.Module) -> None:
    """Enable pruning on every attention layer of a loaded MTP draft model."""
    from vllm.model_executor.layers.attention.attention import Attention

    n = 0
    for name, m in draft_model.named_modules():
        if isinstance(m, Attention):
            layer_name = getattr(m, "layer_name", name)
            m.impl.draft_prefill_pruning_layer = layer_name
            logger.info(
                "draft prefill pruning installed on %s (%s)",
                layer_name,
                type(m.impl).__name__,
            )
            n += 1
    logger.info("draft attention layers: %d (draft prefill pruning=%s)", n, ENABLED)


def mark_ready() -> None:
    """Called when the speculator finished CUDA graph capture."""
    CTX.ready = True
    logger.info("speculator capture done; draft prefill pruning armed")


def prefill_scope(
    speculator: Any,
    num_reqs: int,
    attn_metadata: dict[str, Any] | None,
    cudagraph_runtime_mode: CUDAGraphMode,
) -> contextlib.AbstractContextManager:
    """Context for the draft model forward of a draft prefill.

    Pruning is active only for eager / piecewise draft prefills after capture.
    """
    use = (
        ENABLED
        and CTX.ready
        and attn_metadata is not None
        and not torch.cuda.is_current_stream_capturing()
        and cudagraph_runtime_mode != CUDAGraphMode.FULL
    )
    if not use:
        return contextlib.nullcontext()
    return _active_scope(speculator, num_reqs)


@contextlib.contextmanager
def _active_scope(speculator: Any, num_reqs: int):
    CTX.active = True
    CTX.spec = speculator
    CTX.num_reqs = num_reqs
    try:
        yield
    finally:
        CTX.active = False


def _kv_cache_group_id(layer_name: str, speculator: Any) -> int | None:
    gid = _GROUP_IDS.get(layer_name)
    if gid is None:
        for i, g in enumerate(speculator.kv_cache_config.kv_cache_groups):
            if layer_name in g.layer_names:
                gid = i
                break
        _GROUP_IDS[layer_name] = gid
        logger.info("draft prefill pruning %s: kv cache group %s", layer_name, gid)
    return gid


def pruned_forward(
    impl: Any,
    layer: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Pruned FlashInfer attention for the draft prefill rows that are used.

    Called from ``FlashInferImpl.forward`` while ``CTX.active``. Returns the
    filled ``output``, or None when the regular forward must run instead.
    """
    from vllm.v1.attention.backends import flashinfer as fib

    md = attn_metadata
    ok = (
        md is not None
        and output_scale is None
        and impl.bmm1_scale is not None
        and not getattr(md, "use_cascade", False)
        and not impl.is_kvcache_nvfp4
        and impl.dcp_world_size == 1
        and output.dtype in (torch.bfloat16, torch.float16)
    )
    sp = CTX.spec
    gid = _kv_cache_group_id(impl.draft_prefill_pruning_layer, sp) if ok else None
    if gid is None:
        CTX.fallbacks += 1
        return None
    n = CTX.num_reqs
    lti = sp.last_token_indices[:n]
    seq = (sp.input_buffers.positions[lti] + 1).to(torch.int32)
    bt = sp.block_tables.input_block_tables[gid][:n]
    q = query[lti]
    q = impl.maybe_quant_query(q, md.q_data_type_decode, layer._q_scale)
    q = fib.canonicalize_singleton_dim_strides(q.contiguous())
    kvc = kv_cache
    if kvc.dtype == torch.uint8 and impl.kv_cache_dtype in (
        "fp8",
        "fp8_e4m3",
        torch.float8_e4m3fn,
    ):
        kvc = kvc.view(torch.float8_e4m3fn)
    elif kvc.dtype == torch.uint8 and impl.kv_cache_dtype in (
        "fp8_e5m2",
        torch.float8_e5m2,
    ):
        kvc = kvc.view(torch.float8_e5m2)
    kvp = fib.canonicalize_singleton_dim_strides(
        kvc.permute(*impl.kv_cache_layout.layer_view_order)
    )
    kv_tuple = kvp.split(impl.head_size, dim=-1)
    out = torch.empty(
        (n, impl.num_heads, impl.head_size), dtype=output.dtype, device=output.device
    )
    dec = md.decode
    backend = (
        dec.kernel.value
        if isinstance(dec, fib.FlashInferTrtllmAPIDecode)
        else "trtllm-gen"
    )
    fib.trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=kv_tuple,
        workspace_buffer=fib._get_trtllm_workspace_buffer(),
        block_tables=bt,
        seq_lens=seq,
        max_seq_len=int(sp.draft_max_seq_len),
        bmm1_scale=impl.bmm1_scale,
        bmm2_scale=impl.bmm2_scale,
        window_left=impl.window_left,
        sinks=impl.sinks,
        o_sf_scale=impl.o_sf_scale,
        out=out,
        kv_layout=fib.get_flashinfer_layout_string(impl.kv_cache_layout),
        backend=backend,
        q_len_per_req=1,
    )
    T = md.num_actual_tokens
    if CTX.checked < _CHECK or (
        _CHECK_EVERY and CTX.calls % _CHECK_EVERY == 0 and CTX.ready and n > 8
    ):
        # Validation mode: compare against the regular forward.
        ref = torch.empty_like(output)
        CTX.active = False
        try:
            impl.forward(layer, query, key, value, kv_cache, attn_metadata, ref)
        finally:
            CTX.active = True
        r = ref[:T].view(T, impl.num_heads, impl.head_size)[lti].float()
        d = (r - out.float()).abs()
        rm = d.amax(dim=(1, 2))
        logger.info(
            "draft prefill pruning check %d call=%d: T=%d n=%d maxabs=%.3e "
            "meanabs=%.3e refmax=%.3e bitexact_rows=%d/%d rows>1e-2=%d "
            "rows>0.1=%d ndec=%d npre=%d",
            CTX.checked,
            CTX.calls,
            T,
            n,
            d.max().item(),
            d.mean().item(),
            r.abs().max().item(),
            (rm == 0).sum().item(),
            n,
            (rm > 1e-2).sum().item(),
            (rm > 0.1).sum().item(),
            md.num_decodes,
            md.num_prefills,
        )
        CTX.checked += 1
    o = output[:T].view(T, impl.num_heads, impl.head_size)
    o.zero_()
    o.index_copy_(0, lti, out)
    CTX.calls += 1
    if CTX.calls in (1, 10, 100, 1000, 10000, 100000):
        logger.info(
            "draft prefill pruning calls=%d fallbacks=%d last T=%d n=%d backend=%s",
            CTX.calls,
            CTX.fallbacks,
            T,
            n,
            backend,
        )
    return output
