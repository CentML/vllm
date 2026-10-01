# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel choice for chunked prefill on the trtllm-gen generation kernel."""

import pytest

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.flashinfer import trtllm_gen_prefill_sm_count

HQ, HKV, D, SMS = 16, 2, 256, 212
WS = 394 * 1024 * 1024


def choose(q, kv, ws=WS):
    return trtllm_gen_prefill_sm_count(q, kv, HQ, HKV, D, SMS, ws)


@pytest.mark.parametrize("q,kv", [(4096, 4096), (2048, 10240), (4096, 30000)])
def test_short_prefix_keeps_context_kernel(q, kv):
    # Causal tiles of a chunk without a long prefix are unequal; the wave model
    # does not hold there.
    assert choose(q, kv) is None


@pytest.mark.parametrize(
    "q,kv",
    [(128, 131200), (512, 98816), (2096, 123664), (2144, 40736), (3336, 131976)],
)
def test_split_count_is_a_multiple_of_ctas_and_fits(q, kv):
    sm = choose(q, kv)
    assert sm is not None
    num_ctas = cdiv(q, 16) * HKV  # 128 rows = 16 tokens x 8 q heads per CTA
    assert sm % num_ctas == 0
    splits = sm // num_ctas
    assert 1 <= splits <= 8
    assert splits == 1 or kv // splits >= 8192
    # bf16 partial O + float2 stats for sm_count x 128 rows within half the workspace
    assert sm * 128 * (2 * D + 8) <= WS // 2


def test_workspace_caps_splits():
    q, kv = 2144, 190000
    full = choose(q, kv)
    num_ctas = cdiv(q, 16) * HKV
    tight = choose(q, kv, ws=2 * num_ctas * 128 * (2 * D + 8))
    assert full is not None and full > num_ctas
    assert tight == num_ctas or tight is None


def test_full_waves_keep_context_kernel():
    # 128-token context tiles fill whole waves (17 * 16 = 272 CTAs = 2 waves are
    # not full, but 13 * 16 = 208 CTAs are one full wave of 212 SMs).
    assert choose(13 * 128, 13 * 128 * 40) is None
