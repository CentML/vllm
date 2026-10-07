# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin MXFP8 GEMM backends (mxfp8/rubin.py) compute the cute-dsl path's math:
same operands and block scales, FP32 accumulation; only the K summation order
differs, so outputs agree to BF16 rounding."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability(107)),
    reason="Rubin (sm_107) only",
)


@pytest.mark.parametrize("k,n", [(8192, 5120), (5120, 1792), (1280, 32768)])
@pytest.mark.parametrize("m", [130, 2046, 3072])
def test_rubin_mxfp8_backends_match_cute_dsl(k, n, m):
    from flashinfer import mm_mxfp8, mxfp8_quantize

    from vllm.model_executor.kernels.linear.mxfp8 import rubin as R

    g = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn(n, k, device="cuda", generator=g).bfloat16() * 0.02
    a = torch.randn(m, k, device="cuda", generator=g).bfloat16()
    w_q, w_sf = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    a_q, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout=True)
    b = w_q.t()
    ref = mm_mxfp8(a_q, b, a_sf, w_sf, out_dtype=torch.bfloat16, backend="cute-dsl")
    outs = [R.cublaslt_mm_mxfp8(a_q, b, a_sf, w_sf, torch.bfloat16)]
    for name in ("t256x256_i256_c2x1_s0_p0", "t512x256_i256_c2x1_s0_p0"):
        t = R.Sm107Tactic.parse(name)
        if R.sm107_can_implement(t, m, n, k):
            outs.append(R.sm107_mm_mxfp8(a_q, b, a_sf, w_sf, torch.bfloat16, t))
    den = ref.float().abs().max()
    for out in outs:
        assert ((out.float() - ref.float()).abs().max() / den).item() < 1e-2


# (N, K) keys of rubin._AUTO_TABLE whose entries are checked one by one:
# the DSV4.1 shared expert's gate_up / down (separate linears when the routed
# experts run FlashInfer MegaMoE).
_TABLE_SHAPES = [(4608, 5120), (5120, 2304)]


@pytest.mark.parametrize("n,k", _TABLE_SHAPES)
def test_rubin_auto_table_entries_dispatch_and_match_cute_dsl(n, k):
    """Every _AUTO_TABLE entry of these shapes is what select_backend picks at
    its M, its Sm107 tactic can implement that GEMM (otherwise the dispatch
    silently falls back to cute-dsl), and its output matches cute-dsl."""
    from flashinfer import mm_mxfp8, mxfp8_quantize

    from vllm.model_executor.kernels.linear.mxfp8 import rubin as R

    g = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn(n, k, device="cuda", generator=g).bfloat16() * 0.02
    w_q, w_sf = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    b = w_q.t()
    for m, choice in R._AUTO_TABLE[(n, k)]:
        assert R._lookup(R._AUTO_TABLE, m, n, k) == choice
        a = torch.randn(m, k, device="cuda", generator=g).bfloat16()
        a_q, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout=True)
        ref = mm_mxfp8(a_q, b, a_sf, w_sf, out_dtype=torch.bfloat16, backend="cute-dsl")
        if choice == "cublaslt":
            out = R.cublaslt_mm_mxfp8(a_q, b, a_sf, w_sf, torch.bfloat16)
        else:
            t = R.Sm107Tactic.parse(choice)
            assert R.sm107_can_implement(t, m, n, k), (m, choice)
            out = R.sm107_mm_mxfp8(a_q, b, a_sf, w_sf, torch.bfloat16, t)
        den = ref.float().abs().max()
        err = ((out.float() - ref.float()).abs().max() / den).item()
        assert err < 1e-2, (m, choice, err)
