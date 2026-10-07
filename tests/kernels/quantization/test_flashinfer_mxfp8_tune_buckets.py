# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MXFP8_TUNE_BUCKETS: bucket grammar, round-up mapping, the FlashInfer
config override, and the warmup pass against FlashInfer's real AutoTuner.
"""

import pytest
import torch

pytest.importorskip("flashinfer.gemm.gemm_base")

import flashinfer.gemm.gemm_base as gemm_base  # noqa: E402
from flashinfer.autotuner import AutoTuner, TunableRunner, autotune  # noqa: E402
from flashinfer.fused_moe.utils import (  # noqa: E402
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)

from vllm.model_executor.kernels.linear.mxfp8 import (  # noqa: E402
    flashinfer_tune_buckets as tb,
)

C512 = tuple(range(272, 513, 16))


@pytest.fixture(autouse=True)
def _restore():
    tb.uninstall()
    yield
    tb.uninstall()
    AutoTuner.get().clear_cache()


@pytest.mark.parametrize(
    ("spec", "capture", "max_m", "want"),
    [
        (None, (), None, ()),
        ("", (), None, ()),
        ("272-512:16", (), None, C512),
        ("288, 320,288", (), None, (288, 320)),
        ("300-330:16", (), None, (300, 316)),
        ("capture:257-512", (8, 256, 264, 272, 512, 520), None, (264, 272, 512)),
        ("capture", (8, 264), None, (8, 264)),
        ("capture:257-512,600", (264,), None, (264, 600)),
        ("272-512:16", (), 300, (272, 288)),
    ],
)
def test_parse(spec, capture, max_m, want):
    assert tb.parse_tune_buckets(spec, capture, max_m) == want


@pytest.mark.parametrize(
    "spec", ["abc", "512-272:16", "272-512:0", "272-512", "0", "-16", "capture:300"]
)
def test_parse_rejects(spec):
    with pytest.raises(ValueError, match="VLLM_MXFP8_TUNE_BUCKETS"):
        tb.parse_tune_buckets(spec)


@pytest.mark.parametrize(
    ("x", "want"),
    [
        (1, 1),
        (200, 256),
        (256, 256),
        (257, 272),
        (288, 288),
        (289, 304),
        (300, 304),
        (512, 512),
        (513, 768),
        (3000, 3072),
        (9000, 16384),
    ],
)
def test_map_points(x, want):
    assert tb.map_to_bucket(x, C512) == want


@pytest.mark.parametrize("extras", [C512, (24, 40, 300, 1000, 5000), (1,)])
def test_map_rounds_up_to_smallest_profiled_bucket(extras):
    """Every runtime M maps to the smallest bucket >= M that the tuner
    profiles (so a cache miss cannot come from the mapping), and M outside
    the extras' reach maps exactly as FlashInfer does.
    """
    max_m = 8192
    gen = tb.gen_buckets(max_m, extras)
    assert set(extras) <= set(gen)
    assert set(get_hybrid_num_tokens_buckets(max_m)) <= set(gen)
    for x in range(1, max_m + 1):
        b = tb.map_to_bucket(x, extras)
        assert b >= x and b in gen, x
        assert not [g for g in gen if x <= g < b], x
        if x > max(extras):
            assert b == map_to_hybrid_bucket_uncapped(x)


def test_install_patches_only_cute_dsl_config():
    orig_cute = gemm_base._MM_MXFP8_CUTE_DSL_TUNING_CONFIG
    orig_shared = gemm_base._MM_MXFP8_TUNING_CONFIG
    assert tb.install(C512)
    cfg = gemm_base._MM_MXFP8_CUTE_DSL_TUNING_CONFIG
    assert gemm_base._MM_MXFP8_TUNING_CONFIG is orig_shared
    assert cfg is not orig_cute
    # Everything but the dynamic-M spec is kept (cold L2, CUDA-graph timing,
    # the scale/out constraints).
    assert cfg.use_cuda_graph and cfg.use_cold_l2_cache
    assert cfg.constraint_specs == orig_cute.constraint_specs
    (spec,) = cfg.dynamic_tensor_specs
    assert spec.map_to_tuning_buckets(288) == 288
    assert spec.map_to_tuning_buckets(289) == 304
    assert 272 in spec.gen_tuning_buckets(8192)
    # Idempotent (same object), a different list is refused.
    assert tb.install(list(reversed(C512)))
    assert gemm_base._MM_MXFP8_CUTE_DSL_TUNING_CONFIG is cfg
    with pytest.raises(RuntimeError):
        tb.install((288,))
    tb.uninstall()
    assert gemm_base._MM_MXFP8_CUTE_DSL_TUNING_CONFIG is orig_cute


def test_install_refuses_unknown_config(monkeypatch):
    import dataclasses

    cfg = gemm_base._MM_MXFP8_CUTE_DSL_TUNING_CONFIG
    (spec,) = cfg.dynamic_tensor_specs
    other = dataclasses.replace(
        cfg,
        dynamic_tensor_specs=(
            dataclasses.replace(spec, map_to_tuning_buckets=lambda x: x),
        ),
    )
    monkeypatch.setattr(gemm_base, "_MM_MXFP8_CUTE_DSL_TUNING_CONFIG", other)
    assert not tb.install(C512)
    assert gemm_base._MM_MXFP8_CUTE_DSL_TUNING_CONFIG is other
    assert tb.installed_buckets() is None


class _FakeMxfp8Runner(TunableRunner):
    """Stands in for the CuTe-DSL runner (SM100-only kernels): same inputs
    layout, records the M of every profile it is asked for.
    """

    profiled_m: list[int] = []

    def get_valid_tactics(self, inputs, profile):
        _FakeMxfp8Runner.profiled_m.append(inputs[0].shape[0])
        return [0, 1]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        out = inputs[5]
        out.fill_(float(tactic))
        return out


def _inputs(m, k, n, dev):
    a = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=dev)
    a_scale = torch.empty(
        gemm_base._mxfp8_swizzled_scale_len(m, k, gemm_base.SfLayout.layout_128x4),
        dtype=torch.uint8,
        device=dev,
    )
    b = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=dev).t()
    b_scale = torch.empty(n * k // 32, dtype=torch.uint8, device=dev)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    ws = torch.empty(1 << 20, dtype=torch.uint8, device=dev)
    return [a, b, a_scale, b_scale, torch.bfloat16, out, ws]


def _choose(inputs):
    """What flashinfer.mm_mxfp8(backend="cute-dsl") does, with the fake runner."""
    return AutoTuner.get().choose_one(
        custom_op="mxfp8_gemm",
        runners=[_FakeMxfp8Runner()],
        tuning_config=gemm_base._MM_MXFP8_CUTE_DSL_TUNING_CONFIG,
        inputs=inputs,
    )


def _hit(m, k, n, dev):
    inputs = _inputs(m, k, n, dev)
    tuner = AutoTuner.get()
    shapes = tuple(tuner._get_input_sizes(inputs))
    hit, _, _, _ = tuner.search_cache(
        "mxfp8_gemm",
        [_FakeMxfp8Runner()],
        shapes,
        gemm_base._MM_MXFP8_CUTE_DSL_TUNING_CONFIG,
        inputs=inputs,
    )
    return hit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (timing)")
@pytest.mark.parametrize("extra_pass", [True, False])
def test_warmup_profiles_extra_buckets_before_runtime(monkeypatch, extra_pass):
    """Emulates kernel_warmup.flashinfer_autotune: the dummy run tunes the
    hybrid list (explicit override), then the extra pass tunes the recorded
    shapes at the extra buckets. Afterwards every decode M in (256, 512]
    hits a profiled entry; without the pass the extra buckets miss.
    """
    import vllm.utils.flashinfer as fi_utils
    from vllm.model_executor.layers.quantization.utils import mxfp8_utils

    dev = torch.device("cuda")
    n, k, max_m = 256, 128, 1024
    AutoTuner.get().clear_cache()
    _FakeMxfp8Runner.profiled_m = []

    def fake_mm_mxfp8(a, b, a_scale, b_scale, out_dtype, backend):
        assert backend == "cute-dsl"
        inputs = _inputs(a.shape[0], a.shape[1], b.shape[1], dev)
        inputs[0], inputs[1], inputs[2], inputs[3] = a, b, a_scale, b_scale
        _choose(inputs)

    def fake_quant(x, is_sf_swizzled_layout=False):
        m, kk = x.shape
        return _inputs(m, kk, n, dev)[0], _inputs(m, kk, n, dev)[2]

    monkeypatch.setattr(fi_utils, "mm_mxfp8", fake_mm_mxfp8, raising=False)
    monkeypatch.setattr(mxfp8_utils, "mxfp8_e4m3_quantize", fake_quant)
    monkeypatch.setattr(tb, "_resolve_from_env", lambda: C512)

    layer = torch.nn.Module()
    w = _inputs(1, k, n, dev)
    layer.weight, layer.weight_scale = w[1], w[3]
    tb.maybe_install(layer, torch.bfloat16)
    assert tb.installed_buckets() == C512

    with torch.inference_mode(), autotune(True):
        with autotune(tuning_buckets=get_hybrid_num_tokens_buckets(max_m)):
            _choose(_inputs(max_m, k, n, dev))  # the dummy run's GEMM
        hybrid_profiled = set(_FakeMxfp8Runner.profiled_m)
        if extra_pass:
            assert tb.autotune_extra_buckets(max_m) == [(n, k)]
    assert hybrid_profiled == set(get_hybrid_num_tokens_buckets(max_m))
    extra_profiled = set(_FakeMxfp8Runner.profiled_m) - hybrid_profiled
    assert extra_profiled == (set(C512) - hybrid_profiled if extra_pass else set())

    for m in (1, 32, 200, 256, 500, 600, 1024):  # hybrid buckets: always tuned
        assert _hit(m, k, n, dev), m
    for m in (257, 272, 288, 300, 320, 400, 496):  # extra buckets
        assert _hit(m, k, n, dev) == extra_pass, m
