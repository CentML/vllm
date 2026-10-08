# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``VLLM_LOCALITY_LM_HEAD``: lm_head weights in domain-local memory, read by
the domain-aware skinny GEMM (``skinny.DomainGemm``).

On a GPU with two locality domains (VR200), after the model and the drafter
are loaded (``maybe_enable``, called by the model runner before memory
profiling):

- target head (BF16, ``[vocab, hidden]``): the weight is copied into a
  2 MiB-interleaved localized range (chunk i on domain i % 2) that replaces
  ``lm_head.weight``'s storage (the drafter shares this module), and the
  ``cudaMalloc`` original is freed. Logits for M <= ``TARGET_MAX_M`` rows run on
  the domain kernel; larger M keep the cuBLAS GEMM, now reading the
  interleaved localized weight.
- MXFP8 draft head (``Mxfp8DraftLmHead``, ``VLLM_MTP_DRAFT_LM_HEAD_MXFP8``): the
  e4m3 weight moves to a localized range the same way and the domain kernel
  serves M <= ``DRAFT_MAX_M`` rows (x quantized to e4m3 in the kernel with one
  scale per row); larger M keep the FlashInfer GEMM on the localized weight.

Values: ``1`` / ``target`` (target head only), ``draft``, ``both``; unset / ``0``:
off (no code path changes). Numerics: fp32 accumulation in a different order than cuBLAS /
FlashInfer (not bitwise; logits within bf16 rounding for the target head).
Memory: no extra copies (padding to 2 MiB chunks only, plus two 16 MB copies
of the draft head's E8M0 scales, one per domain).
Knobs: ``VLLM_LOCALITY_LM_HEAD_TARGET_MAX_M`` (default 16: the domain kernel beats
cuBLAS up to M = 16 on VR200, -11 % at M <= 8),
``VLLM_LOCALITY_LM_HEAD_DRAFT_MAX_M`` (default 32), ``..._CFG16`` / ``..._CFG8``
(stage configs, see ``_ext``).
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MODE = os.environ.get("VLLM_LOCALITY_LM_HEAD", "0")
TARGET_ON = MODE in ("1", "target", "both")
# the MXFP8 draft kernel is slower than FlashInfer today (54 vs 43 us at M = 1): opt-in only
DRAFT_ON = MODE in ("draft", "both")
TARGET_MAX_M = int(os.environ.get("VLLM_LOCALITY_LM_HEAD_TARGET_MAX_M", "16"))
DRAFT_MAX_M = int(os.environ.get("VLLM_LOCALITY_LM_HEAD_DRAFT_MAX_M", "32"))
CFG16 = int(os.environ.get("VLLM_LOCALITY_LM_HEAD_CFG16", "0")) or None
CFG8 = int(os.environ.get("VLLM_LOCALITY_LM_HEAD_CFG8", "0")) or None


def enabled() -> bool:
    return TARGET_ON or DRAFT_ON


class LocalityLmHead:
    """Target (BF16) head: ``__call__`` returns logits, or None above max_m."""

    def __init__(self, lm_head: torch.nn.Module, topo, max_m: int) -> None:
        from .memory import localize
        from .skinny import BF16_MAX_M, DomainGemm

        w = lm_head.weight
        loc = localize(w.data, "interleave")
        # every user of lm_head.weight (cuBLAS fallback, drafter) now reads the
        # localized copy; the cudaMalloc original is released with its last ref
        w.data = loc.tensor
        self.gemm = DomainGemm(topo, loc, cfg=CFG16)
        self.max_m = min(max_m, BF16_MAX_M)
        self.loc = loc

    def __call__(self, x: torch.Tensor) -> torch.Tensor | None:
        x2 = x.reshape(-1, x.shape[-1])
        m = x2.shape[0]
        if m == 0 or m > self.max_m or x2.stride(-1) != 1:
            return None
        return self.gemm(x2).view(*x.shape[:-1], self.gemm.n)

    def warmup_sizes(self) -> list[int]:
        return [m for m in (1, 8, 16, 24, 32) if m <= self.max_m]


def _target_language_model(model: torch.nn.Module) -> torch.nn.Module | None:
    for mod in (getattr(model, "language_model", None), model):
        if mod is not None and hasattr(mod, "lm_head") and hasattr(mod, "logits_processor"):
            return mod
    return None


def _check_head(head, lp) -> str | None:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
    )

    if not isinstance(head, ParallelLMHead) or head.tp_size != 1:
        return "needs an unsharded lm_head (TP=1)"
    if not isinstance(head.quant_method, UnquantizedEmbeddingMethod):
        return "lm_head is quantized"
    w = head.weight
    if w.dtype != torch.bfloat16 or w.shape[1] != 2048 or w.shape[0] % 16:
        return f"needs a BF16 [N % 16 == 0, 2048] head, got {w.dtype} {tuple(w.shape)}"
    if lp.scale != 1.0 or lp.soft_cap is not None or lp.logits_as_input:
        return "logits scaling/soft-cap is not supported"
    if lp.head_dtype not in (None, torch.bfloat16):
        return "head_dtype must be the model dtype"
    return None


def maybe_enable(model: torch.nn.Module, drafter_model: torch.nn.Module | None) -> None:
    """Localize the target / draft heads (see module docstring). Call once,
    after the model and the drafter (lm_head sharing) are loaded and before
    memory profiling."""
    if not enabled():
        return
    from .topology import get_topology

    topo = get_topology(torch.cuda.current_device())
    if topo is None:
        logger.warning("VLLM_LOCALITY_LM_HEAD ignored: the GPU has no 2 locality domains.")
        return
    logger.info("Locality topology: %s", topo.summary())
    if TARGET_ON:
        lm = _target_language_model(model)
        reason = "no lm_head/logits_processor" if lm is None else _check_head(lm.lm_head, lm.logits_processor)
        if reason is None and lm.logits_processor.org_vocab_size != lm.lm_head.weight.shape[0]:
            reason = "padded vocab"
        if reason is not None:
            logger.warning("VLLM_LOCALITY_LM_HEAD target head not localized: %s.", reason)
        else:
            lm.locality_lm_head = LocalityLmHead(lm.lm_head, topo, TARGET_MAX_M)
            loc = lm.locality_lm_head.loc
            logger.info(
                "Locality target lm_head %s: %d chunks of %d MiB (dom0 %d / dom1 %d), domain kernel for M <= %d.",
                tuple(lm.lm_head.weight.shape), len(loc.ordinals), loc.chunk_bytes >> 20,
                loc.ordinals.count(0), loc.ordinals.count(1), lm.locality_lm_head.max_m,
            )
    if DRAFT_ON and drafter_model is not None:
        from vllm.model_executor.kernels.linear.mxfp8_draft_head import Mxfp8DraftLmHead

        n = 0
        for module in drafter_model.modules():
            head = getattr(module, "draft_lm_head_mxfp8", None)
            if isinstance(head, Mxfp8DraftLmHead):
                if head.enable_locality(topo, module.lm_head.weight, DRAFT_MAX_M, CFG8):
                    n += 1
                else:
                    logger.warning("VLLM_LOCALITY_LM_HEAD draft: MXFP8 head differs from the shared lm_head; not localized.")
        if n:
            logger.info("Locality MXFP8 draft lm_head: domain kernel for M <= %d.", DRAFT_MAX_M)
        else:
            logger.warning("VLLM_LOCALITY_LM_HEAD draft: no MXFP8 draft head (set VLLM_MTP_DRAFT_LM_HEAD_MXFP8=1).")
    torch.cuda.empty_cache()


@torch.inference_mode()
def warmup(model: torch.nn.Module, drafter_model: torch.nn.Module | None) -> list[str]:
    """Launch every kernel specialization serving can reach (module load and
    attribute setup happen here, not during serving)."""
    done = []
    lm = _target_language_model(model)
    head = getattr(lm, "locality_lm_head", None) if lm is not None else None
    if isinstance(head, LocalityLmHead):
        for m in head.warmup_sizes():
            x = torch.randn(m, head.gemm.k, device="cuda", dtype=torch.bfloat16)
            head(x)
        done.append(f"target M={head.warmup_sizes()}")
    if drafter_model is not None:
        for module in drafter_model.modules():
            dh = getattr(module, "draft_lm_head_mxfp8", None)
            if dh is not None and getattr(dh, "loc_gemm", None) is not None:
                ms = [m for m in (1, 8, 16, 32, 64) if m <= dh.loc_max_m]
                for m in ms:
                    x = torch.randn(m, dh.loc_gemm.k, device="cuda", dtype=torch.bfloat16)
                    dh.loc_gemm(x)
                done.append(f"draft M={ms}")
    torch.cuda.synchronize()
    return done
