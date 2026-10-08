# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_LOCALITY_LM_HEAD kernels on a GPU with two locality domains (VR200):
the MXFP8 draft head (locality.mxgemm.DomainMxGemm) against FlashInfer's
mm_mxfp8 and the exact product of the same MXFP8 operands, and the BF16 target
head (locality.skinny.DomainGemm) against cuBLAS.
"""

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)
if torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the locality MXFP8 head kernel is SM107-only", allow_module_level=True)

from vllm.model_executor.layers.locality import (  # noqa: E402
    DomainGemm,
    get_topology,
    localize,
    mxgemm,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.utils import flashinfer as vllm_flashinfer  # noqa: E402

# >= 3 row blocks of 128 per SM and domain: 1024 rows per 2 MiB chunk, 128 chunks
VOCAB, HIDDEN = 131072, 2048


@pytest.fixture(scope="module")
def topo():
    t = get_topology(torch.accelerator.current_device_index())
    if t is None:
        pytest.skip("the GPU has no two locality domains")
    return t


@pytest.fixture(scope="module")
def mx(topo):
    torch.manual_seed(0)
    w = torch.randn(VOCAB, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    wq, ws = mxfp8_e4m3_quantize(w)
    ws = ws.view(torch.uint8).reshape(VOCAB, HIDDEN // 32)
    ws_sw = swizzle_mxfp8_scale(ws, M=VOCAB, K=HIDDEN).contiguous()
    loc = localize(wq.view(torch.uint8), "interleave")
    g = mxgemm.DomainMxGemm(topo, loc, ws_sw)
    g_plain = mxgemm.DomainMxGemm(topo, wq.view(torch.uint8), ws_sw)
    wdq = (
        wq.float().view(VOCAB, HIDDEN // 32, 32)
        * torch.exp2(ws.float() - 127).view(VOCAB, HIDDEN // 32, 1)
    ).view(VOCAB, HIDDEN)
    return g, g_plain, wq, ws_sw, wdq


def _exact(x: torch.Tensor, wdq: torch.Tensor) -> torch.Tensor:
    """fp32 product of the MXFP8-quantized operands (x with FlashInfer's rule)."""
    xq, xs = mxfp8_e4m3_quantize(x)
    m = x.shape[0]
    xs = xs.view(torch.uint8).reshape(m, HIDDEN // 32)
    scale = torch.exp2(xs.float() - 127).view(m, HIDDEN // 32, 1)
    xdq = (xq.float().view(m, HIDDEN // 32, 32) * scale).view(m, HIDDEN)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return xdq @ wdq.t()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@pytest.mark.parametrize("m", [1, 2, 7, 16, 17, 32])
@pytest.mark.parametrize("scale", [1.0, 1e-2])
def test_mx_head_matches_mxfp8_product(mx, m: int, scale: float):
    g, g_plain, wq, ws_sw, wdq = mx
    torch.manual_seed(m)
    x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16) * scale
    out = g(x)
    exact = _exact(x, wdq)
    # same products as FlashInfer, fp32 sums in another order, then bf16 rounding
    atol = 1e-4 * exact.abs().max().item()
    torch.testing.assert_close(out.float(), exact, rtol=2**-7, atol=atol)
    xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    fi = vllm_flashinfer.mm_mxfp8(
        xq, wq.t(), xs, ws_sw, out_dtype=torch.bfloat16, backend="cute-dsl"
    )
    rel = ((out.float() - exact).norm() / exact.norm()).item()
    rel_fi = ((fi.float() - exact).norm() / exact.norm()).item()
    assert rel <= rel_fi * 1.01 + 1e-6, (rel, rel_fi)
    assert torch.equal(out.argmax(-1), fi.argmax(-1))
    # deterministic, and independent of the work split / placement
    assert torch.equal(g(x), out)
    assert torch.equal(g_plain(x), out)
    assert torch.equal(g(x, pdl=True), out)


@pytest.mark.parametrize("pdl", [False, True])
def test_mx_head_cuda_graph(mx, pdl: bool):
    g = mx[0]
    for m in (2, 17):
        x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, VOCAB, device="cuda", dtype=torch.bfloat16)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            g(x, out, pdl=pdl)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g(x, out, pdl=pdl)
            g(x, out, pdl=pdl)
        for _ in range(3):
            x.copy_(torch.randn_like(x))
            out.zero_()
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(out, g(x))
        # every fixup flag was reset by the launch that used it
        assert int(g.flags.count_nonzero()) == 0


@pytest.mark.parametrize("m", [1, 6, 16])
def test_bf16_target_head_tracks_cublas(topo, m: int):
    torch.manual_seed(1)
    w = torch.randn(VOCAB, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    g = DomainGemm(topo, localize(w, "interleave"))
    x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ w.float().t()
    out = g(x)
    cb = F.linear(x, w)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    rel_cb = ((cb.float() - ref).norm() / ref.norm()).item()
    assert rel <= rel_cb * 1.05 + 1e-6, (rel, rel_cb)
    assert torch.equal(g(x), out)
    assert torch.equal(out.argmax(-1), cb.argmax(-1))


def test_draft_head_dispatch(topo):
    from vllm.model_executor.kernels.linear.mxfp8_draft_head import Mxfp8DraftLmHead

    torch.manual_seed(2)
    w = torch.randn(VOCAB, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    head = Mxfp8DraftLmHead(w)
    x_small = torch.randn(4, HIDDEN, device="cuda", dtype=torch.bfloat16)
    x_big = torch.randn(24, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref_small, ref_big = head(x_small, w), head(x_big, w)  # FlashInfer before
    assert head.enable_locality(topo, max_m=16, pdl=True) is None
    assert head.loc_max_m == 16
    out_small = head(x_small, w)  # domain kernel
    # M > 16: FlashInfer on the localized weight and the domain-0 scales
    out_big = head(x_big, w)
    assert torch.equal(out_big, ref_big)
    torch.testing.assert_close(
        out_small.float(), ref_small.float(), rtol=2**-7, atol=1e-3
    )
    assert torch.equal(out_small.argmax(-1), ref_small.argmax(-1))
