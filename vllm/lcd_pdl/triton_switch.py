# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Switch for the opt-in PDL launch of the Qwen3.5 fusion Triton kernels
(study tag ``lcd``; port of the Rubin pdl-why "pw-t" Triton part).

Kernels patched with ``launch_pdl: tl.constexpr = False`` start with
``gdc_wait()`` (before any global memory access, so RAW and WAR hazards on
the predecessor are covered) and ``gdc_launch_dependents()`` (early trigger:
a PDL-launched successor may run its pre-wait prologue, e.g. the CuTe GEMM
weight prefetch, while this kernel runs; it still waits for this kernel to
complete before reading its outputs). Bit-exact by construction.

Env: LCD_PDL_TRITON=1 enables (default off: launch_pdl=False, the kernels
compile without the PDL instructions). Requires compute capability >= 9.
"""

import functools
import os


@functools.cache
def lcd_pdl_triton_on() -> bool:
    if os.environ.get("LCD_PDL_TRITON", "0") != "1":
        return False
    try:
        import torch

        return torch.cuda.get_device_capability()[0] >= 9
    except Exception:
        return False
