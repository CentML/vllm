# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in programmatic dependent launch (PDL) for the eager step glue kernels.

A decode step launches about 20 small Triton kernels eagerly between its CUDA
graphs: input preparation before the target graph, then the rejection sampler,
the post-update and the draft input preparation. The host is far ahead of the
GPU, so these launches are already queued, but each kernel boundary on the
stream still costs about 0.7-0.9 us of GPU idle (the next grid is only
launched once the previous one has fully retired).

With ``VLLM_GLUE_PDL=1`` these kernels are launched with programmatic stream
serialization. Each one first executes ``griddepcontrol.wait`` (which returns
only once the preceding grid has completed and its memory is visible), before
any global load or store, and then ``griddepcontrol.launch_dependents``, so the
next kernel is launched and made resident while this one runs. Same kernels,
same arguments, same arithmetic: results are bitwise identical to the flag-off
path; only the launch overlap changes.

The early trigger is only safe if every PDL-launched kernel that may follow a
glue kernel waits before its first global load. That is the same condition as
the MXFP8 producers' early trigger (``mxfp8_pdl.py``: it does not hold with
Inductor PDL on), so the glue kernels use PDL only while that policy allows the
early trigger; otherwise they launch as before.

Environment:
    VLLM_GLUE_PDL=1     enable (default off; needs SM90+)
"""

import os

from vllm.model_executor.layers.fusion.mxfp8_pdl import mxfp8_producer_early_trigger
from vllm.platforms import current_platform

_ENV = os.environ.get("VLLM_GLUE_PDL", "0") == "1"
_arch: list[bool] = []


def glue_pdl() -> bool:
    """Whether the glue kernels launch with PDL (wait first, trigger early)."""
    if not _ENV:
        return False
    if not _arch:
        _arch.append(current_platform.is_arch_support_pdl())
    return _arch[0] and mxfp8_producer_early_trigger()
