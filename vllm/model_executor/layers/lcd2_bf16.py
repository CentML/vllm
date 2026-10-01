# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in single-kernel small-M BF16 GEMMs for the decode router and the GDN
``in_proj_ba`` projection (GB300 study tag ``lcd2``).

On GB300 (sm_103) cuBLAS runs these decode-size BF16 GEMMs (x [M, 2048] with
the 264-row SEG-folded router weight, or the 64-row ``in_proj_ba`` weight) as
two kernels: an nvjet split-K (or cutlass) GEMM plus a separate
``cublasLt::splitKreduce_kernel``. FlashInfer's TinyGEMM2 router GEMM
(``flashinfer.gemm.routergemm.tinygemm_bf16``, a TensorRT-LLM tinygemm2 port:
TMA + mma.sync, fp32 accumulation, no split-K, no atomics, no autotuner) does
the same ``x @ W^T`` in one kernel with a bf16 output.

Numerics: float-order (the fp32 accumulation order of each output before the
bf16 rounding differs from cuBLAS); deterministic. Token counts above
``LCD2_MAXM`` (prefill / mixed steps) keep ``F.linear`` (bit-identical to the
stock path there).

Environment (default off):

* ``LCD2_BF16`` = ``tiny`` enables the TinyGEMM2 path (``0`` = off).
* ``LCD2_RTR`` (default ``1``): the SEG-folded router GEMM (needs SEG_FOLD=1);
  the 264-row weight is zero-padded to 272 rows (TinyGEMM2 needs N % 16 == 0)
  and the routing kernel reads the [M, 272] logits through its row stride.
* ``LCD2_BA`` (default ``1``): the GDN ``in_proj_ba`` GEMM (custom op
  ``torch.ops.vllm.lcd2_bf16_linear``).
* ``LCD2_MAXM`` (default 64): largest token count that uses TinyGEMM2.
* ``LCD2_PDL`` (default 0): launch TinyGEMM2 with programmatic dependent launch.
"""

import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

BACKEND = os.environ.get("LCD2_BF16", "0")
ENABLED = BACKEND == "tiny"
RTR = ENABLED and os.environ.get("LCD2_RTR", "1") == "1"
BA = ENABLED and os.environ.get("LCD2_BA", "1") == "1"
MAXM = int(os.environ.get("LCD2_MAXM", "64"))
PDL = os.environ.get("LCD2_PDL", "0") == "1"

STATS = {"rtr_tiny": 0, "rtr_stock": 0, "ba_tiny": 0, "ba_stock": 0}
_SEEN: set = set()
_STATE = {"ok": None}


def _note(site: str, M: int, tiny: bool) -> None:
    STATS[f"{site}_{'tiny' if tiny else 'stock'}"] += 1
    if (site, M, tiny) not in _SEEN:
        _SEEN.add((site, M, tiny))
        logger.info("[lcd2] %s M=%d -> %s", site, M, "tinygemm2" if tiny else "F.linear")


def _tiny():
    from flashinfer.gemm.routergemm import tinygemm_bf16

    return tinygemm_bf16


def available() -> bool:
    """Load (or JIT-build) the TinyGEMM2 module once, at model load time."""
    if _STATE["ok"] is None:
        try:
            fn = _tiny()
            x = torch.zeros(1, 64, dtype=torch.bfloat16, device="cuda")
            w = torch.zeros(16, 64, dtype=torch.bfloat16, device="cuda")
            b = torch.zeros(16, dtype=torch.bfloat16, device="cuda")
            o = torch.empty(1, 16, dtype=torch.bfloat16, device="cuda")
            fn(x, w, o, b, use_pdl=PDL)
            torch.cuda.synchronize()
            _STATE["ok"] = True
            logger.info("[lcd2] TinyGEMM2 BF16 path ON (router=%s, in_proj_ba=%s, max M=%d, pdl=%s)", RTR, BA, MAXM, PDL)
        except Exception as e:  # noqa: BLE001 - fall back to cuBLAS, say why
            _STATE["ok"] = False
            logger.warning("[lcd2] TinyGEMM2 unavailable (%r): router / in_proj_ba stay on cuBLAS", e)
    return bool(_STATE["ok"])


def prep_router(st: dict) -> None:
    """Zero-pad the SEG-folded router weight [264, K] to [272, K] (+ zero bias)."""
    if not RTR or not available():
        return
    w = st["w264"]
    n = (w.shape[0] + 15) // 16 * 16
    wp = torch.zeros(n, w.shape[1], dtype=w.dtype, device=w.device)
    wp[: w.shape[0]].copy_(w)
    st["lcd2_w"] = wp
    st["lcd2_b"] = torch.zeros(n, dtype=w.dtype, device=w.device)


def router_logits(x: torch.Tensor, st: dict) -> torch.Tensor:
    """Router logits for the SEG-folded MoE: [M, 272] bf16 (cols 264.. are 0)
    via TinyGEMM2 for small M, else ``F.linear(x, w264)`` ([M, 264])."""
    M = x.shape[0]
    w = st.get("lcd2_w")
    tiny = (w is not None and 0 < M <= MAXM and x.dtype == torch.bfloat16 and x.is_contiguous())
    _note("router", M, tiny)
    if not tiny:
        return F.linear(x, st["w264"])
    out = torch.empty(M, w.shape[0], dtype=x.dtype, device=x.device)
    _tiny()(x, w, out, st["lcd2_b"], use_pdl=PDL)
    return out


def _ba_impl(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = x.shape[0]
    tiny = (0 < M <= MAXM and _STATE["ok"] is True and x.is_contiguous())
    _note("ba", M, tiny)
    if not tiny:
        return F.linear(x, w)
    out = torch.empty(M, w.shape[0], dtype=x.dtype, device=x.device)
    _tiny()(x, w, out, b, use_pdl=PDL)
    return out


def _ba_fake(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)


_REGISTERED = [False]


def register_ops() -> None:
    if _REGISTERED[0]:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(op_name="lcd2_bf16_linear", op_func=_ba_impl, fake_impl=_ba_fake)
    _REGISTERED[0] = True


def ba_eligible(weight: torch.Tensor) -> bool:
    return (BA and weight.dtype == torch.bfloat16 and weight.dim() == 2 and weight.shape[0] % 16 == 0
            and weight.shape[1] % 64 == 0)


if BA:
    register_ops()
