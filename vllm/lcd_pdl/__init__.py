# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in PDL weight prefetch for FlashInfer's CuTe-DSL dense MXFP8 GEMMs.

Both overlays are derived from FlashInfer 0.7.2's stock sources and are
installed only over those exact files (sha256 guard below). The persistent
overlay is also the parent of FlashInfer's native SM107 subclass
(``dense_blockscaled_gemm_sm107.py``), which overrides its kernel: on that
path the overlay supplies the stock parent interface, not weight-prefetch or
early-trigger changes.

On the SM100/SM103 paths (and on sm_107 while ``flashinfer_mxfp8_k64`` keeps
FlashInfer's Sm100 runner), every dense MXFP8 linear at decode sizes
(in_proj_qkvz, out_proj, qkv_proj,
o_proj, shared expert, MTP layer, MXFP8 draft lm_head) runs FlashInfer's
``Sm100BlockScaledPersistentDenseGemmKernel`` or
``Sm100BlockScaledSplitKGemmKernel``. They are launched with programmatic
dependent launch, and the vLLM MXFP8 producers trigger their dependents right
after their own wait, but the GEMMs only start streaming weights after
``griddepcontrol_wait``. The overlay modules here are the stock files plus:

* weight L2 prefetch (``PFW=1`` persistent, ``PFWS=1`` split-K): the TMA warp
  issues ``cp.async.bulk.prefetch.tensor`` for this CTA's weight tile (all
  k-blocks) and its scale factors BEFORE ``griddepcontrol_wait``. Weights are
  never written by a predecessor and a prefetch is only an L2 hint, so the
  results are bit-identical;
* early trigger (``PFW_TRIG=1`` / ``PFWS_TRIG=1``):
  ``griddepcontrol_launch_dependents`` right after the wait. Safe only if
  every PDL consumer of the GEMM output waits before touching it and writes
  nothing before its wait (audited for vLLM's current consumers; soak before
  adopting).
* weights into shared memory before the wait (``LCD_FI_PREWAIT=1``
  persistent, ``LCD_FI_PREWAITS=1`` split-K): the TMA warp arms the first
  ``min(stages, k-blocks)`` pipeline stages of the CTA's first tile and issues
  their weight copies (B + B-scale, or A + A-scale under swap-AB) before
  ``griddepcontrol_wait``; TMEM is allocated before it; only the TMA warp
  waits, then adds the activation copies of the armed stages. Weights are never
  written by a predecessor, the MMA/epilogue warps follow the wait through the
  stage mbarriers, so results are bit-identical;
* trigger placement (``LCD_FI_TRIG`` persistent / ``LCD_FI_TRIGS`` split-K):
  ``0`` stock (last instruction), ``1`` after the CTA's mainloop (MMA warp),
  ``2`` after the last C store is issued (epilogue warps). Same consumer
  requirement as the early trigger.

The pre-wait weight copies and prefetches slice the operands exactly as the
stock TMA loop does (narrow N tiles share one 128-wide SFB tile); the
persistent kernel's pre-armed copies carry its stock L2 eviction policies.

Env (read when the overlay modules are imported; baked into the kernels):
  LCD_FI_OVERLAY=1  install the overlays (``vllm/__init__.py``). Default off.
  PFW, PFW_KB (0 = all k-blocks), PFW_TILES (first tiles per CTA), PFW_TRIG
  PFWS, PFWS_TRIG
  LCD_FI_PREWAIT, LCD_FI_TRIG, LCD_FI_PREWAITS, LCD_FI_TRIGS (default 0)
  LCD_FI_PREWAIT_STAGES (max armed stages, 0 = all), LCD_FI_PREWAIT_WAITALL
  (MMA/epilogue warps also sleep in griddepcontrol.wait after their setup)
  LCD_FI_OVERLAY_BASE_CHECK=0  skip the base-file sha256 guard (do not).

A FlashInfer module is replaced only if the file it would have loaded is the
exact base the overlay was derived from (sha256 guard), so a different
FlashInfer build is left alone (logged).
"""

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_MAP = {
    "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100": (
        "fi_dense_blockscaled_gemm_sm100.py"
    ),
    "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_splitk": (
        "fi_dense_blockscaled_gemm_sm100_splitk.py"
    ),
}
# Stock FlashInfer 0.7.2 sources the overlays were derived from. Accept no
# other source revisions.
_BASE_SHA = {
    "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100": (
        "8e379737ac18447c6ca5869e4a32fc006ac55f0fb7f544f25ad8e0f69594cfce",
    ),
    "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_splitk": (
        "9d649776c24658c739957c5173145ba972c790f40c6f68845cf27de93a1d1d7e",
    ),
}
SKIPPED: list = []
LOADED: list = []
INSTALLED = [False]


def _log(msg: str) -> None:
    print(f"[lcd-pdl] {msg}", file=sys.stderr, flush=True)


def _base_ok(fullname, path) -> bool:
    want = _BASE_SHA[fullname]
    if os.environ.get("LCD_FI_OVERLAY_BASE_CHECK", "1") != "1":
        return True
    try:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        with open(spec.origin, "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
    except Exception as e:  # noqa: BLE001 - cannot prove the base: do not overlay
        _log(f"NOT overlaying {fullname}: cannot hash the original module ({e!r})")
        return False
    if got not in want:
        _log(
            f"NOT overlaying {fullname}: original {spec.origin} sha256 "
            f"{got[:16]} not in approved bases "
            f"{', '.join(digest[:16] for digest in want)}"
        )
        return False
    return True


class _OverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        f = _MAP.get(fullname)
        if f is None:
            return None
        if not _base_ok(fullname, path):
            SKIPPED.append(fullname)
            return None
        LOADED.append(fullname)
        e = os.environ.get
        _log(
            f"FlashInfer overlay {fullname} <- vllm/lcd_pdl/{f} "
            f"(PFW={e('PFW', '0')}/{e('PFW_TRIG', '0')} "
            f"PFWS={e('PFWS', '0')}/{e('PFWS_TRIG', '0')} "
            f"PREWAIT={e('LCD_FI_PREWAIT', '0')}/{e('LCD_FI_TRIG', '0')} "
            f"PREWAITS={e('LCD_FI_PREWAITS', '0')}/{e('LCD_FI_TRIGS', '0')} "
            f"stages={e('LCD_FI_PREWAIT_STAGES', '0')} "
            f"waitall={e('LCD_FI_PREWAIT_WAITALL', '0')}, "
            f"pid {os.getpid()})"
        )
        return importlib.util.spec_from_file_location(fullname, os.path.join(_DIR, f))


def install() -> None:
    if INSTALLED[0]:
        return
    pre = [n for n in _MAP if n in sys.modules]
    if pre:
        _log(f"NOT installed: already imported {pre}")
        return
    sys.meta_path.insert(0, _OverlayFinder())
    INSTALLED[0] = True


if os.environ.get("LCD_FI_OVERLAY", "0") == "1":
    install()
