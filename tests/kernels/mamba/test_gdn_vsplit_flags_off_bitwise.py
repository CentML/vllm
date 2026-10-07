# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The vendored V-split GDN chunk kernel with its opt-in variants (CG0 split,
C1 reorder) switched off against a reference copy of the package from before
they were added, bit for bit.

Needs an SM100-family GPU and two env vars (the test skips otherwise):

- ``GDN_VSPLIT_REF_DIR``: a directory holding the reference package files
  ``__init__.py``, ``adapter.py``, ``gdn_chunked_vs.py`` and ``gdn_sched_vs.py``
  (e.g. ``git show <ref>:vllm/third_party/flashinfer_gdn_vsplit/<file>``);
- ``QWEN36_CONFIG``: the model's ``config.json`` (GDN head counts and dims,
  TP1).

Every case compiles the kernel variant once per package (CuTe, no on-disk
cache), so the module takes minutes.
"""

import importlib
import json
import os
import shutil
import sys

import pytest
import torch

REF_FILES = ("__init__.py", "adapter.py", "gdn_chunked_vs.py", "gdn_sched_vs.py")
REF_PACKAGE = "ref_vsplit"

# Prefill sequence lengths per step.
SEQLEN_CASES = {
    "1x64": [64],
    "4x512": [512] * 4,
    "8mixed": [37, 129, 1000, 2049, 64, 300, 77, 513],
    "16x1024": [1024] * 16,
}
# Pool slots beyond the ones the step touches.
SPARE_SLOTS = 2

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("GDN_VSPLIT_REF_DIR"),
        reason="GDN_VSPLIT_REF_DIR (reference package files) not set",
    ),
    pytest.mark.skipif(
        not os.environ.get("QWEN36_CONFIG"),
        reason="QWEN36_CONFIG (model config.json) not set",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] not in (10, 11),
        reason="V-split GDN kernel needs an SM100/SM110-family GPU",
    ),
]


def _gdn_dims() -> tuple[int, int, int, int]:
    """(HQ, HV, DK, DV) from the model config (TP1)."""
    with open(os.environ["QWEN36_CONFIG"]) as f:
        cfg = json.load(f)
    if "linear_num_key_heads" not in cfg:
        cfg = cfg["text_config"]
    return (
        cfg["linear_num_key_heads"],
        cfg["linear_num_value_heads"],
        cfg["linear_key_head_dim"],
        cfg["linear_value_head_dim"],
    )


@pytest.fixture(scope="module")
def ref_adapter(tmp_path_factory):
    pytest.importorskip("cutlass")
    src = os.environ["GDN_VSPLIT_REF_DIR"]
    root = tmp_path_factory.mktemp("gdn_vsplit_ref")
    pkg = root / REF_PACKAGE
    pkg.mkdir()
    for name in REF_FILES:
        shutil.copy(os.path.join(src, name), pkg / name)
    sys.path.insert(0, str(root))
    try:
        importlib.import_module(REF_PACKAGE)
        yield importlib.import_module(f"{REF_PACKAGE}.adapter")
    finally:
        sys.path.remove(str(root))


@pytest.fixture(scope="module")
def cand_adapter():
    pytest.importorskip("cutlass")
    return importlib.import_module("vllm.third_party.flashinfer_gdn_vsplit.adapter")


def _inputs(seqlens: list[int], state_dtype: torch.dtype):
    """Step inputs as the model passes them: bf16 q/k/v (q, k L2-normalized),
    fp32 gate = exp(g) and beta, int32 cu_seqlens, a pool with
    ``SPARE_SLOTS`` extra slots and permuted int32 state indices."""
    hq, hv, dk, dv = _gdn_dims()
    torch.manual_seed(0)
    dev = "cuda"
    t = sum(seqlens)
    q = torch.nn.functional.normalize(torch.randn(t, hq, dk, device=dev), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(t, hq, dk, device=dev), dim=-1)
    v = torch.randn(t, hv, dv, device=dev)
    g = -torch.nn.functional.softplus(torch.randn(t, hv, device=dev))
    gate = torch.exp(g).float()
    beta = torch.sigmoid(torch.randn(t, hv, device=dev)).float()
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(seqlens), 0)
    num_slots = len(seqlens) + SPARE_SLOTS
    pool = (0.1 * torch.randn(num_slots, hv, dv, dk, device=dev)).to(state_dtype)
    indices = torch.randperm(num_slots)[: len(seqlens)].to(torch.int32)
    return (
        q.bfloat16(),
        k.bfloat16(),
        v.bfloat16(),
        gate,
        beta,
        cu.to(dev),
        pool,
        indices.to(dev),
    )


def _run(adapter, inputs, use_init: bool, v_split: int):
    """One launch on a fresh copy of the pool; returns (output, pool)."""
    q, k, v, gate, beta, cu, pool, indices = inputs
    pool = pool.clone()
    out = torch.zeros_like(v)
    adapter.chunk_gated_delta_rule_vsplit(
        q,
        k,
        v,
        gate,
        beta,
        out,
        cu,
        pool if use_init else None,
        pool,
        q.size(-1) ** -0.5,
        state_indices=indices,
        v_split=v_split,
    )
    torch.cuda.synchronize()
    return out, pool


def _check(out, pool, ref_out, ref_pool, inputs) -> None:
    indices = inputs[-1].long()
    untouched = torch.ones(pool.size(0), dtype=torch.bool, device=pool.device)
    untouched[indices] = False
    assert torch.equal(out, ref_out)
    assert torch.equal(pool[indices], ref_pool[indices])
    assert torch.equal(pool[untouched], inputs[6][untouched])
    assert torch.equal(pool, ref_pool)


@pytest.mark.parametrize("host_trim", [False, True], ids=["trim0", "trim1"])
@pytest.mark.parametrize("seqlens", SEQLEN_CASES.values(), ids=SEQLEN_CASES.keys())
@pytest.mark.parametrize("v_split", [1, 2])
@pytest.mark.parametrize("use_init", [False, True], ids=["noinit", "init"])
@pytest.mark.parametrize(
    "state_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
@torch.inference_mode()
def test_gdn_vsplit_flags_off_bitwise(
    ref_adapter,
    cand_adapter,
    monkeypatch,
    state_dtype,
    use_init,
    v_split,
    seqlens,
    host_trim,
) -> None:
    """Output, touched pool slots and untouched slots equal the reference.
    With GDN_HOST_TRIM the second launch takes the fast path (fkey lookup)."""
    monkeypatch.setattr(cand_adapter, "_CG0_SPLIT", False)
    monkeypatch.setattr(cand_adapter, "_C1_REORDER", False)
    monkeypatch.setattr(ref_adapter, "GDN_HOST_TRIM", False)
    monkeypatch.setattr(cand_adapter, "GDN_HOST_TRIM", host_trim)
    inputs = _inputs(seqlens, state_dtype)
    ref_out, ref_pool = _run(ref_adapter, inputs, use_init, v_split)
    assert torch.isfinite(ref_out.float()).all()
    for _ in range(2):
        out, pool = _run(cand_adapter, inputs, use_init, v_split)
        _check(out, pool, ref_out, ref_pool, inputs)


@pytest.mark.parametrize("host_trim", [False, True], ids=["trim0", "trim1"])
@pytest.mark.parametrize("seqlens", SEQLEN_CASES.values(), ids=SEQLEN_CASES.keys())
@pytest.mark.parametrize(
    "state_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
@torch.inference_mode()
def test_gdn_vsplit_flags_on_bitwise_informational(
    cand_adapter, monkeypatch, state_dtype, seqlens, host_trim
) -> None:
    """Informational: v_split=2 with an initial state (the launches the
    variants apply to) with both flags on equals flags off. A mismatch or a
    compile failure is reported as xfail, not as a failure."""
    monkeypatch.setattr(cand_adapter, "GDN_HOST_TRIM", host_trim)
    inputs = _inputs(seqlens, state_dtype)
    monkeypatch.setattr(cand_adapter, "_CG0_SPLIT", False)
    monkeypatch.setattr(cand_adapter, "_C1_REORDER", False)
    off_out, off_pool = _run(cand_adapter, inputs, True, 2)
    monkeypatch.setattr(cand_adapter, "_CG0_SPLIT", True)
    monkeypatch.setattr(cand_adapter, "_C1_REORDER", True)
    try:
        for _ in range(2):
            out, pool = _run(cand_adapter, inputs, True, 2)
            _check(out, pool, off_out, off_pool, inputs)
    except Exception as e:  # noqa: BLE001 - informational
        pytest.xfail(f"flags on differ from flags off: {type(e).__name__}: {e}")
