# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mid-M split-K MXFP8 tactics (flashinfer_mxfp8_splitk): offer rules, runner
key compatibility, and on SM10x numerics vs the stock tactic + determinism."""

import types

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import (
    flashinfer_mxfp8_splitk as sk,
)


class _T:
    def __init__(self, *shape, contiguous=True):
        self.shape = shape
        self._c = contiguous

    def is_contiguous(self):
        return self._c


class _BaseRunner:
    def get_valid_tactics(self, inputs, profile):
        return [
            ((128, 64), (1, 1), True, False, 1),
            ((256, 64), (2, 1), False, False, 1),
            ((256, 192), (4, 2), False, False, 1),
        ]

    def forward(self, inputs, tactic=None, do_preparation=False, **kw):
        return ("base", tactic)


def _inputs(m, k=4096, n=2048, out_contiguous=True):
    return [_T(m, k), _T(k, n), None, None, None, _T(m, n, contiguous=out_contiguous), None]


def _install(monkeypatch, tactics, max_m=2048, cap=0, cap_min_m=513):
    monkeypatch.setattr(sk, "_max_active_clusters", lambda s: {2: 106, 4: 46}[s])
    gb = types.SimpleNamespace(
        _cute_dsl_gemm_mxfp8_runner=lambda *a: _BaseRunner(), run_calls=[]
    )
    monkeypatch.setattr(
        sk, "run", lambda gb_, inputs, tn, s, pdl: gb.run_calls.append((tn, s, pdl))
    )
    sk.install(gb, tactics, max_m, cap, cap_min_m)
    return gb, gb._cute_dsl_gemm_mxfp8_runner(10, 7, True, torch.bfloat16)


def test_parse_rejects_unsupported():
    assert sk.parse_tactics("64,2;128,4") == [(64, 2), (128, 4)]
    for bad in ("32,2", "64,8", "256,2"):
        with pytest.raises(ValueError):
            sk.parse_tactics(bad)


def test_offer_boundaries(monkeypatch):
    monkeypatch.setattr(sk, "_max_active_clusters", lambda s: {2: 106, 4: 46}[s])
    n, k = 2048, 4096
    assert not sk.offer(32, n, k, 64, 2, 2048)  # stock split-K owns M <= 32
    assert sk.offer(33, n, k, 64, 2, 2048)
    assert not sk.offer(64, n, k, 64, 2, 48)  # above MAX_M
    assert not sk.offer(64, n, 512 + 128, 64, 4, 2048)  # K % (128 s) != 0
    # One wave: 16 weight tiles x ceil(M / tile_n) clusters <= max active clusters.
    assert sk.offer(6 * 64, n, k, 64, 2, 2048)  # 96 <= 106
    assert not sk.offer(7 * 64, n, k, 64, 2, 2048)  # 112 > 106
    assert sk.offer(2 * 128, n, k, 128, 4, 2048)  # 32 <= 46
    assert not sk.offer(3 * 128, n, k, 128, 4, 2048)  # 48 > 46


def test_runner_keeps_class_name_and_dispatches(monkeypatch):
    gb, r = _install(monkeypatch, [(64, 2), (128, 4)])
    # Autotune cache/file keys use the runner class name: must stay the base's.
    assert type(r).__name__ == _BaseRunner.__name__
    t = r.get_valid_tactics(_inputs(128), None)
    assert (1072, 64, 2) in t and (1072, 128, 4) in t
    assert all(x[0] != 1072 for x in r.get_valid_tactics(_inputs(32), None))
    assert all(x[0] != 1072 for x in r.get_valid_tactics(_inputs(128, out_contiguous=False), None))
    r.forward(_inputs(128), tactic=(1072, 128, 4))
    assert gb.run_calls == [(128, 4, True)]
    assert r.forward(_inputs(128), tactic=7) == ("base", 7)


def test_cluster_cap(monkeypatch):
    _, r = _install(monkeypatch, [], cap=2, cap_min_m=513)
    assert len(r.get_valid_tactics(_inputs(512), None)) == 3
    t = r.get_valid_tactics(_inputs(1024), None)
    assert ((256, 192), (4, 2), False, False, 1) not in t and len(t) == 2


def _sm10x_cute_dsl():
    if not torch.cuda.is_available():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(100):
        return False
    try:
        import cutlass  # noqa: F401
        import flashinfer.gemm.gemm_base  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _sm10x_cute_dsl(), reason="needs SM10x + FlashInfer CuTe DSL")
@pytest.mark.parametrize("n,k", [(2048, 4096), (1024, 2048)])
@pytest.mark.parametrize("m", [33, 64, 290])
@pytest.mark.parametrize("tile_n,s", [(64, 2), (64, 4), (128, 2), (128, 4)])
def test_numerics_vs_stock_and_deterministic(n, k, m, tile_n, s):
    import flashinfer.gemm.gemm_base as gb

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
        swizzle_mxfp8_scale,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    x = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
    wq, ws = mxfp8_e4m3_quantize(w)
    ws = swizzle_mxfp8_scale(ws.view(n, k // 32), M=n, K=k).contiguous()
    xq, xs = mxfp8_e4m3_quantize(x)
    xs = swizzle_mxfp8_scale(xs.view(m, k // 32), M=m, K=k).contiguous()
    runner = gb._cute_dsl_gemm_mxfp8_runner(10, 0, True, torch.bfloat16)

    def gemm(tactic):
        out = torch.full((m, n), 7.0, device="cuda", dtype=torch.bfloat16)
        inputs = [xq, wq.t(), xs, ws, torch.bfloat16, out, None]
        if sk.is_splitk_tactic(tactic):
            return sk.run(gb, inputs, tactic[1], tactic[2], True)
        return runner.forward(inputs, tactic=tactic)

    ref = gemm(((128, tile_n), (1, 1), True, False, 1)).float()
    t = sk.encode_tactic(tile_n, s)
    o1, o2 = gemm(t), gemm(t)
    torch.cuda.synchronize()
    # Fixed fp32 reduction order: bitwise run to run.
    assert torch.equal(o1, o2)
    # Only the fp32 summation order differs from the unsplit tactic.
    d = (o1.float() - ref).abs()
    assert (d.norm() / ref.norm()).item() < 1e-4
    assert d.max().item() <= 2 * torch.finfo(torch.bfloat16).eps * ref.abs().max().item()
