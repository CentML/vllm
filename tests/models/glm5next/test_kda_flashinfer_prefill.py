# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA chunked prefill through FlashInfer's CuTe DSL kernel
(``kda_prefill_backend=flashinfer``).

It updates the recurrent state pool in place at the slot indices instead of
gathering and scattering, so it must zero fresh sequences' slots itself, leave
other slots alone and match the Triton chunk path and an fp32 reference.
"""

import pytest
import torch

from tests.models.glm5next.test_kda_flashinfer_decode import (
    NUM_SLOTS,
    _make_inputs,
    _make_layer,
    _non_spec_metadata,
    _run,
    _spec_metadata,
)
from vllm.models.glm5next.common import kda as glm_kda
from vllm.models.glm5next.common.kda import (
    _flashinfer_kda_prefill_error,
    _resolve_kda_prefill_backend,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.usefixtures("default_vllm_config")

D = 128
LOWER_BOUND = -5.0


def _require_flashinfer_prefill(num_heads: int, state_dtype: torch.dtype):
    if not current_platform.is_cuda():
        pytest.skip("FlashInfer KDA prefill requires CUDA")
    error = _flashinfer_kda_prefill_error(num_heads, state_dtype, LOWER_BOUND)
    if error is not None:
        pytest.skip(f"FlashInfer KDA prefill cannot run: {error}")


def _prefill_layer(num_heads, state_dtype, prefill_backend, decode_backend="triton"):
    layer = _make_layer(num_heads, 0, state_dtype, False, decode_backend)
    layer.kda_prefill_backend = prefill_backend
    return layer


def _reference(layer, inputs, query_lens, slots, has_state):
    """fp32 token-by-token KDA over the conv outputs of the Triton path."""
    H = layer.local_num_heads
    conv_state, state = (x.clone() for x in layer.kv_cache)
    conv_state = conv_state.transpose(-1, -2)  # SD -> [slot, 3P, width]
    weight = torch.cat(
        [m.weight.squeeze(1) for m in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)]
    )
    a_log = layer.A_log.view(-1).float()
    out = torch.zeros(sum(query_lens), H, D, device="cuda")
    start = 0
    for n, (length, slot) in enumerate(zip(query_lens, slots)):
        hist = (
            conv_state[slot].float()
            if has_state[n]
            else torch.zeros_like(conv_state[slot], dtype=torch.float32)
        )
        s = state[slot].float() if has_state[n] else torch.zeros_like(state[slot])
        s = s.float()
        for t in range(start, start + length):
            x = inputs["qkv_proj_states"][t].float()
            window = torch.cat([hist, x[:, None]], dim=1)
            hist = window[:, 1:].bfloat16().float()
            y = (window * weight).sum(-1)
            y = (y * torch.sigmoid(y)).bfloat16().float()
            q, k, v = y.view(3, H, D).unbind(0)
            q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * D**-0.5
            k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            raw_g = inputs["g1"][0, t].float() + layer.dt_bias.view(H, D)
            decay = (LOWER_BOUND * torch.sigmoid(a_log.exp()[:, None] * raw_g)).exp()
            beta = torch.sigmoid(inputs["beta"][0, t].float())
            s = s * decay[:, None, :]
            s = (
                s
                + (beta[:, None] * (v - torch.einsum("hvk,hk->hv", s, k)))[:, :, None]
                * k[:, None, :]
            )
            o = torch.einsum("hvk,hk->hv", s, q)
            o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + 1e-5)
            out[t] = (
                o * layer.o_norm.weight.float() * torch.sigmoid(inputs["g2"][t].float())
            )
        state[slot] = s.to(state.dtype)
        start += length
    return out, state


def test_resolve_flashinfer_prefill_backend(monkeypatch: pytest.MonkeyPatch):
    args = ("flashinfer", D, torch.bfloat16, LOWER_BOUND, 64, torch.float32)
    monkeypatch.setattr(glm_kda.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(glm_kda, "_flashinfer_kda_prefill_error", lambda *a: None)
    assert _resolve_kda_prefill_backend(*args) == "flashinfer"
    monkeypatch.setattr(
        glm_kda,
        "_flashinfer_kda_prefill_error",
        lambda *a: "ValueError: backend='cute-dsl' does not support this call",
    )
    with pytest.raises(RuntimeError, match="cute-dsl"):
        _resolve_kda_prefill_backend(*args)
    with pytest.raises(RuntimeError, match="bounded"):
        _resolve_kda_prefill_backend("flashinfer", D, torch.bfloat16, None)
    # Prefix-cache checkpoints are FlashKDA-only.
    assert glm_kda.kda_prefill_checkpoint_alignment("flashinfer") is None


@pytest.mark.parametrize(
    "state_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
@pytest.mark.parametrize("num_heads", [16, 32, 64])
@torch.inference_mode()
def test_prefill_step_matches_triton_and_reference(monkeypatch, num_heads, state_dtype):
    _require_flashinfer_prefill(num_heads, state_dtype)
    # Fresh and continued sequences, a one-token one, and lengths across the
    # 16-token chunk boundary.
    query_lens = [40, 1, 17, 3, 96]
    slots = [3, 5, 8, 13, 21]
    has_state = [False, True, True, False, True]
    meta = _non_spec_metadata(query_lens, 0, slots, has_state)
    out = {}
    for backend in ("triton", "flashinfer"):
        layer = _prefill_layer(num_heads, state_dtype, backend)
        inputs = _make_inputs(num_heads, sum(query_lens))
        out[backend] = (_run(layer, meta, inputs, monkeypatch), layer.kv_cache)
    out_f, (conv_f, state_f) = out["flashinfer"]
    out_t, (conv_t, state_t) = out["triton"]
    ref_layer = _prefill_layer(num_heads, state_dtype, "triton")
    ref, ref_state = _reference(
        ref_layer,
        _make_inputs(num_heads, sum(query_lens)),
        query_lens,
        slots,
        has_state,
    )

    torch.testing.assert_close(conv_f, conv_t, atol=0, rtol=0)
    tol = 3e-2 if state_dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(state_f[slots], state_t[slots], atol=tol, rtol=tol)
    # Different chunked kernels: compare in norm, then each against fp32.
    assert (out_f - out_t).float().norm() < 1e-2 * out_t.float().norm()
    err_f = (out_f[0].float() - ref).abs().max()
    err_t = (out_t[0].float() - ref).abs().max()
    assert err_f <= 1.5 * err_t + 1e-2, (err_f, err_t)
    torch.testing.assert_close(state_f[slots], ref_state[slots], atol=tol, rtol=tol)
    # Only the sequences' slots (and the null slot 0) may change.
    fresh = _prefill_layer(num_heads, state_dtype, "triton").kv_cache[1]
    untouched = [s for s in range(1, NUM_SLOTS) if s not in slots]
    torch.testing.assert_close(state_f[untouched], fresh[untouched], atol=0, rtol=0)


@pytest.mark.usefixtures("workspace_init")
@pytest.mark.parametrize("num_prefills", [2, 3], ids=["routed", "not-routed"])
@torch.inference_mode()
def test_few_prefills_route_to_flashkda(monkeypatch, num_prefills):
    """With FlashInfer prefill, a step with at most ``_flashkda_max_prefills``
    sequences runs FlashKDA (FlashInfer's few-sequence route is slower) and a
    larger one keeps FlashInfer; each matches that backend bit for bit."""
    num_heads = 64
    _require_flashinfer_prefill(num_heads, torch.float32)
    if _resolve_kda_prefill_backend("auto", D, torch.bfloat16, LOWER_BOUND) != (
        "flashkda"
    ):
        pytest.skip("FlashKDA is not available")
    import vllm._flashkda_C  # noqa: F401

    query_lens = [40, 70, 17][:num_prefills]
    slots = [3, 8, 21][:num_prefills]
    has_state = [False, True, True][:num_prefills]
    meta = _non_spec_metadata(query_lens, 0, slots, has_state)
    T, N = sum(query_lens), len(query_lens)
    workspace = torch.ops._flashkda_C.get_workspace_size(T, num_heads, N)

    def run(backend, max_prefills):
        layer = _prefill_layer(num_heads, torch.float32, backend)
        layer._flashkda_max_prefills = max_prefills
        layer._flashkda_buffer_specs = (
            ((N, num_heads, D, D), torch.float32),
            ((N, num_heads, D, D), torch.float32),
            ((workspace,), torch.uint8),
            ((1, T, num_heads, D), torch.bfloat16),
        )
        out = _run(layer, meta, _make_inputs(num_heads, T), monkeypatch)
        return out, layer.kv_cache[1]

    out, state = run("flashinfer", 2)
    expected_out, expected_state = run(
        "flashkda" if num_prefills <= 2 else "flashinfer", 0
    )
    torch.testing.assert_close(out, expected_out, atol=0, rtol=0)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)


@torch.inference_mode()
def test_one_token_prefill_step(monkeypatch):
    """A step whose only prefill row is one token takes the recurrent kernel
    (FlashInfer's prefill needs two tokens)."""
    num_heads = 32
    _require_flashinfer_prefill(num_heads, torch.float32)
    meta = _non_spec_metadata([1], 0, [6], [False])
    out = {}
    for backend in ("triton", "flashinfer"):
        layer = _prefill_layer(num_heads, torch.float32, backend)
        inputs = _make_inputs(num_heads, 1)
        out[backend] = (_run(layer, meta, inputs, monkeypatch), layer.kv_cache)
    torch.testing.assert_close(
        out["flashinfer"][0], out["triton"][0], atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        out["flashinfer"][1][1][6], out["triton"][1][1][6], atol=2e-3, rtol=2e-3
    )


@pytest.mark.parametrize("num_heads", [16, 32, 64])
@torch.inference_mode()
def test_mixed_step_with_flashinfer_decode(monkeypatch, num_heads):
    """Decode rows through the fused decode kernel, prefill rows through the
    FlashInfer prefill: the prefill rows equal a prefill-only step."""
    _require_flashinfer_prefill(num_heads, torch.float32)
    error = glm_kda._flashinfer_kda_decode_error(
        num_heads, 1, False, torch.float32, LOWER_BOUND, 1e-5
    )
    if error is not None:
        pytest.skip(f"FlashInfer fused KDA decode cannot run: {error}")
    decode_slots = [2, 4, 6]
    prefill_lens, prefill_slots = [33, 70], [9, 12]
    n = len(decode_slots)
    meta = _non_spec_metadata(
        [1] * n + prefill_lens,
        n,
        decode_slots + prefill_slots,
        [True] * n + [False, True],
    )
    layer = _prefill_layer(num_heads, torch.float32, "flashinfer", "flashinfer")
    inputs = _make_inputs(num_heads, n + sum(prefill_lens))
    mixed = _run(layer, meta, inputs, monkeypatch)

    prefill_only = _prefill_layer(num_heads, torch.float32, "flashinfer", "flashinfer")
    full = _make_inputs(num_heads, n + sum(prefill_lens))
    tail = {
        "qkv_proj_states": full["qkv_proj_states"][n:],
        "beta": full["beta"][:, n:],
        "g1": full["g1"][:, n:],
        "g2": full["g2"][n:],
    }
    tail_meta = _non_spec_metadata(prefill_lens, 0, prefill_slots, [False, True])
    expected = _run(prefill_only, tail_meta, tail, monkeypatch)
    torch.testing.assert_close(mixed[:, n:], expected, atol=0, rtol=0)
    torch.testing.assert_close(
        layer.kv_cache[1][prefill_slots],
        prefill_only.kv_cache[1][prefill_slots],
        atol=0,
        rtol=0,
    )


@torch.inference_mode()
def test_prefill_with_spec_verify_rows(monkeypatch):
    """Steps with draft-verify rows write the prefill rows to their own buffer
    and scatter them; no FlashKDA workspace is needed."""
    num_heads, num_spec = 32, 2
    _require_flashinfer_prefill(num_heads, torch.float32)
    slots = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [11, 11, 11], [14, 14, 14]]
    meta = _spec_metadata(num_spec, [1, 2, 3], [25, 9], 3, slots)
    out = {}
    for backend in ("triton", "flashinfer"):
        layer = _make_layer(num_heads, num_spec, torch.float32, False, "triton")
        layer.kda_prefill_backend = backend
        inputs = _make_inputs(num_heads, meta.num_actual_tokens)
        out[backend] = (_run(layer, meta, inputs, monkeypatch), layer.kv_cache)
    out_f, out_t = out["flashinfer"][0].float(), out["triton"][0].float()
    assert (out_f - out_t).norm() < 1e-2 * out_t.norm()
    torch.testing.assert_close(
        out["flashinfer"][1][1][1:], out["triton"][1][1][1:], atol=1e-2, rtol=1e-2
    )


@pytest.mark.usefixtures("workspace_init")
@torch.inference_mode()
def test_routed_flashkda_prefill_with_spec_verify_rows(monkeypatch):
    """A step with draft-verify rows routed from FlashInfer to FlashKDA writes
    the prefill rows to a new buffer (the reserved one holds one token)."""
    num_heads, num_spec = 32, 2
    _require_flashinfer_prefill(num_heads, torch.float32)
    if _resolve_kda_prefill_backend("auto", D, torch.bfloat16, LOWER_BOUND) != (
        "flashkda"
    ):
        pytest.skip("FlashKDA is not available")
    import vllm._flashkda_C  # noqa: F401

    slots = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [11, 11, 11], [14, 14, 14]]
    meta = _spec_metadata(num_spec, [1, 2, 3], [25, 9], 3, slots)
    out = {}
    for backend in ("triton", "flashinfer"):
        layer = _make_layer(num_heads, num_spec, torch.float32, False, "triton")
        layer.kda_prefill_backend = backend
        if backend == "flashinfer":
            workspace = torch.ops._flashkda_C.get_workspace_size(34, num_heads, 2)
            layer._flashkda_max_prefills = 2
            layer._flashkda_buffer_specs = (
                ((2, num_heads, D, D), torch.float32),
                ((2, num_heads, D, D), torch.float32),
                ((workspace,), torch.uint8),
                ((1, 1, num_heads, D), torch.bfloat16),
            )
        inputs = _make_inputs(num_heads, meta.num_actual_tokens)
        out[backend] = (_run(layer, meta, inputs, monkeypatch), layer.kv_cache)
    out_r, out_t = out["flashinfer"][0].float(), out["triton"][0].float()
    assert (out_r - out_t).norm() < 1e-2 * out_t.norm()
    torch.testing.assert_close(
        out["flashinfer"][1][1][1:], out["triton"][1][1][1:], atol=1e-2, rtol=1e-2
    )
