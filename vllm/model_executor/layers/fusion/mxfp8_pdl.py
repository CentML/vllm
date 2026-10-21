# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whether the fused MXFP8 producer kernels trigger their PDL dependents early.

The producers (RMSNorm, SiLU-mul, attention-gate and GDN gated-norm -> MXFP8)
call ``griddepcontrol.launch_dependents`` right after their own
``griddepcontrol.wait`` so the consumer MXFP8 GEMM can launch while they run.
That is only safe if every PDL-launched kernel that may follow waits before its
first global load. Inductor's PDL codegen (``triton.enable_pdl``) issues loads
before ``gdc_wait``, so with Inductor PDL on the early trigger is disabled.

The policy is set from the vLLM config when it is installed
(``set_current_vllm_config``, in every process) and fails closed: until a
config has been seen, or if Inductor's PDL setting cannot be read, there is no
early trigger.
"""

from collections.abc import Mapping
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

INDUCTOR_PDL_KEY = "triton.enable_pdl"

_early_trigger = False


def inductor_pdl_enabled(inductor_compile_config: Mapping[str, Any]) -> bool:
    """Whether Inductor launches its Triton kernels with PDL.

    An explicit ``inductor_compile_config`` entry wins; otherwise Inductor's
    global default (``TORCHINDUCTOR_ENABLE_PDL``) applies. Unknown -> True.
    """
    if INDUCTOR_PDL_KEY in inductor_compile_config:
        return bool(inductor_compile_config[INDUCTOR_PDL_KEY])
    try:
        from torch._inductor import config as inductor_config

        return bool(inductor_config.triton.enable_pdl)
    except Exception:
        return True


def configure_mxfp8_producer_early_trigger(
    inductor_compile_config: Mapping[str, Any],
) -> bool:
    """Set the policy from a compilation config's Inductor options."""
    global _early_trigger
    try:
        enabled = not inductor_pdl_enabled(inductor_compile_config)
    except Exception:
        enabled = False
    _early_trigger = enabled
    logger.info_once("MXFP8 producer early PDL trigger: %s", "on" if enabled else "off")
    return enabled


def mxfp8_producer_early_trigger() -> bool:
    """Whether the MXFP8 producers may trigger their dependents early."""
    return _early_trigger
