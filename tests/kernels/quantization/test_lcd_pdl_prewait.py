# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The lcd_pdl CuTe MXFP8 GEMM overlay's pre-wait weight staging and trigger
placements (LCD_FI_PREWAIT[S], LCD_FI_TRIG[S], LCD_FI_PREWAIT_STAGES,
LCD_FI_PREWAIT_WAITALL) give the same bits as the overlay with them off.

Each variant runs in a fresh process because the overlay env is baked into the
kernels at import. Every case is a PDL chain captured in a CUDA graph:
early-trigger MXFP8 producer -> GEMM (explicit tactic) -> PDL consumer, replayed
on changing inputs, so a GEMM that read its activation before the producer
finished, or staged the wrong operand, changes the digests.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

# (N, K, M, tactic): split-K s2/s4, 1-CTA persistent swap-AB, 2-CTA swap-AB,
# 2-CTA without swap (weights = B operand), 2-CTA with a (2, 4) multicast
# cluster, and a multi-tile persistent case (M = 290).
CASES = [
    (12288, 2048, 6, [[128, 8], [1, 1], True, False, 2]),
    (2048, 4096, 6, [[128, 8], [1, 1], True, False, 4]),
    (12288, 2048, 6, [[128, 32], [1, 1], True, False, 1]),
    (12288, 2048, 6, [[256, 64], [2, 1], True, False, 1]),
    (12288, 2048, 6, [[256, 128], [2, 1], False, False, 1]),
    (12288, 2048, 6, [[256, 64], [2, 4], True, False, 1]),
    (12288, 2048, 290, [[256, 192], [2, 1], True, False, 1]),
]

VARIANTS = {
    "prewait": {"LCD_FI_PREWAIT": "1", "LCD_FI_PREWAITS": "1"},
    "prewait_trig1": {
        "LCD_FI_PREWAIT": "1",
        "LCD_FI_PREWAITS": "1",
        "LCD_FI_TRIG": "1",
        "LCD_FI_TRIGS": "1",
    },
    "prewait_trig2_stages2_waitall": {
        "LCD_FI_PREWAIT": "1",
        "LCD_FI_PREWAITS": "1",
        "LCD_FI_TRIG": "2",
        "LCD_FI_TRIGS": "2",
        "LCD_FI_PREWAIT_STAGES": "2",
        "LCD_FI_PREWAIT_WAITALL": "1",
    },
    "prewait_pfw": {
        "PFW": "1",
        "PFWS": "1",
        "LCD_FI_PREWAIT": "1",
        "LCD_FI_PREWAITS": "1",
    },
}

_CHILD = r"""
import hashlib, json, sys
import torch
import vllm  # noqa: F401  (LCD_FI_OVERLAY=1: installs vllm.lcd_pdl)
import vllm.lcd_pdl as ovl
import flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100 as pk
from flashinfer.gemm.gemm_base import _cute_dsl_gemm_mxfp8_runner
from vllm.model_executor.layers.fusion.mxfp8_pdl import (
    configure_mxfp8_producer_early_trigger,
)
from vllm.model_executor.layers.fusion.silu_mul_mxfp8_quant import silu_mul_mxfp8_quant
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)

configure_mxfp8_producer_early_trigger({"triton.enable_pdl": False})
major, minor = torch.cuda.get_device_capability()
runner = _cute_dsl_gemm_mxfp8_runner(major, minor, True, torch.bfloat16)
res = {"loaded": sorted(ovl.LOADED), "prewait": bool(getattr(pk, "_PREWAIT", False)),
       "digests": []}
for i, (n, k, m, t) in enumerate(json.loads(sys.argv[1])):
    tactic = tuple(tuple(x) if isinstance(x, list) else x for x in t)
    g = torch.Generator(device="cuda").manual_seed(1000 + i)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    wq, ws = mxfp8_e4m3_quantize(w)
    ws = swizzle_mxfp8_scale(ws.view(n, k // 32), M=n, K=k).contiguous()
    src = torch.empty(m, 2 * k, device="cuda", dtype=torch.bfloat16)
    inputs = [torch.randn(m, 2 * k, device="cuda", generator=g).to(torch.bfloat16)
              for _ in range(4)]
    width = min(n, 4096)

    def chain():
        q, s = silu_mul_mxfp8_quant(src)
        y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        runner.forward([q, wq.t(), s, ws, torch.bfloat16, y, None], tactic=tactic)
        q2, s2 = silu_mul_mxfp8_quant(y[:, :width])
        return y, q2, s2

    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        src.copy_(inputs[0])
        chain()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=st):
            outs = chain()
    torch.cuda.synchronize()
    h = hashlib.sha256()
    for r in range(8):
        src.copy_(inputs[r % len(inputs)])
        graph.replay()
        torch.cuda.synchronize()
        for o in outs:
            h.update(o.view(torch.uint8).cpu().numpy().tobytes())
    res["digests"].append(h.hexdigest())
print("RESULT " + json.dumps(res))
"""


def _run(extra_env: dict[str, str]) -> dict:
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("LCD_FI_") or k in ("PFW", "PFWS", "PFW_TRIG", "PFWS_TRIG"):
            del env[k]
    env.update({"LCD_FI_OVERLAY": "1", **extra_env})
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(CASES)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT ") :])


@pytest.fixture(scope="module")
def reference() -> dict:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("CuTe-DSL MXFP8 GEMM overlay needs an SM10x GPU")
    pytest.importorskip("flashinfer")
    ref = _run({})
    if len(ref["loaded"]) != 2:
        pytest.skip(f"overlay not applied to this FlashInfer build: {ref['loaded']}")
    assert not ref["prewait"]
    return ref


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_prewait_and_trigger_are_bitwise(reference: dict, variant: str) -> None:
    got = _run(VARIANTS[variant])
    assert got["loaded"] == reference["loaded"]
    assert got["prewait"]
    assert got["digests"] == reference["digests"], [
        (case, a == b)
        for case, a, b in zip(CASES, got["digests"], reference["digests"])
    ]
