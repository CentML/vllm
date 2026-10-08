# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in PDL weight prefetch for FlashInfer's CuTe-DSL dense MXFP8 GEMMs.

The persistent overlay supports both FlashInfer 0.6.18.post1's stock source
and the SM107 backport installed by ``tools/install_flashinfer_sm107.py``.
Those approved parents differ only by two NamedBarrier attributes required
by the Rubin subclass; the overlay retains them without changing the SM100
kernel. The split-K overlay retains its stock 0.6.18.post1 base.
The native SM107 subclass overrides its kernel: this supplies its compatible
parent interface, not weight-prefetch or early-trigger changes to that kernel.

On the SM100/SM103 paths, every dense MXFP8 linear at decode sizes
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

Env (read when the overlay modules are imported; baked into the kernels):
  LCD_FI_OVERLAY=1  install the overlays (``vllm/__init__.py``). Default off.
  PFW, PFW_KB (0 = all k-blocks), PFW_TILES (first tiles per CTA), PFW_TRIG
  PFWS, PFWS_TRIG
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
# The pinned SM107 parent differs from stock only by the NamedBarrier objects
# retained in the shared persistent overlay. Accept no other source revisions.
# Backport: flashinfer-ai/flashinfer@2b16e3c765fd4bf79339a4b2536a2de68a9c85ee.
_BASE_SHA = {
    "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100": (
        "aad93031b1145c43195d1af2cbeb310c91df5c000310df26dfea768be949f705",
        "64b83236fa69202c97e015fedf298fe5abdbf3f464609fb44a5e7a92158dd79b",
    ),
    "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_splitk": (
        "ca08b62a377b43aaeea30ec2b7a24b92515f72e3b48f5398e9af9b4c360edc9e",
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
        _log(
            f"FlashInfer overlay {fullname} <- vllm/lcd_pdl/{f} "
            f"(PFW={os.environ.get('PFW', '0')}/{os.environ.get('PFW_TRIG', '0')} "
            f"PFWS={os.environ.get('PFWS', '0')}/{os.environ.get('PFWS_TRIG', '0')}, "
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
