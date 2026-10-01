# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in PDL weight prefetch for FlashInfer's CuTe-DSL dense MXFP8 GEMMs
(GB300 low-concurrency decode latency, study tag ``lcd``; port of the Rubin
pdl-why "pw-tmgs" GEMM overlays to a vLLM-tree arm).

At decode sizes every dense MXFP8 linear (in_proj_qkvz, out_proj, qkv_proj,
o_proj, MTP layer) runs FlashInfer's ``Sm100BlockScaledPersistentDenseGemmKernel``
or ``Sm100BlockScaledSplitKGemmKernel``. They are launched with programmatic
dependent launch but only start streaming their weights after
``griddepcontrol_wait``, i.e. after the producer of the activation has fully
finished. The two overlay modules in this package are the FlashInfer
0.6.18.post1 kernel files (image d9612d2e) with two additions:

* weight L2 prefetch: the TMA warp issues ``cp.async.bulk.prefetch.tensor``
  for this CTA's weight tile (all k-blocks) and its scale factors BEFORE
  ``griddepcontrol_wait``. Weights are never written by a predecessor, and
  activations / activation scales are only touched after the wait, so the
  math and the results are unchanged (bit-exact).
* early trigger: ``griddepcontrol_launch_dependents`` right after the wait,
  so the next PDL-launched kernel can run its own prologue / prefetch while
  this GEMM computes (every PDL consumer waits before reading the output).

Env (read when the overlay modules are imported; baked into the compiled
kernels):
  LCD_FI_OVERLAY=1  install the overlays (vllm/__init__.py imports this
                    package first thing when set). Default off.
  PFW=1 PFW_TRIG=1  persistent kernel: prefetch / early trigger
                    (PFW_KB=0 all k-blocks, PFW_TILES=1 first tile per CTA)
  PFWS=1 PFWS_TRIG=1  split-K kernel: prefetch / early trigger

The overlay is applied by a meta-path finder that maps the two FlashInfer
module names to the files here; FlashInfer imports them lazily (first
mm_mxfp8 with the cute-dsl backend), after vLLM's own import. If FlashInfer
already imported them, nothing is replaced (logged).
"""

import importlib.abc
import importlib.util
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_MAP = {
    "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100": "fi_dense_blockscaled_gemm_sm100.py",
    "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_splitk": "fi_dense_blockscaled_gemm_sm100_splitk.py",
}
INSTALLED = [False]
LOADED: list = []


class _OverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        f = _MAP.get(fullname)
        if f is None:
            return None
        LOADED.append(fullname)
        print(
            f"[lcd-pdl] FlashInfer overlay {fullname} <- vllm/lcd_pdl/{f} "
            f"(PFW={os.environ.get('PFW', '0')}/{os.environ.get('PFW_TRIG', '0')} "
            f"PFWS={os.environ.get('PFWS', '0')}/{os.environ.get('PFWS_TRIG', '0')}, pid {os.getpid()})",
            file=sys.stderr,
            flush=True,
        )
        return importlib.util.spec_from_file_location(fullname, os.path.join(_DIR, f))


def install() -> None:
    if INSTALLED[0]:
        return
    pre = [n for n in _MAP if n in sys.modules]
    if pre:
        print(f"[lcd-pdl] NOT installed: already imported {pre}", file=sys.stderr, flush=True)
        return
    sys.meta_path.insert(0, _OverlayFinder())
    INSTALLED[0] = True


if os.environ.get("LCD_FI_OVERLAY", "0") == "1":
    install()
