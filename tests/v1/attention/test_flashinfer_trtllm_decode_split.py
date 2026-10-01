# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV policy for FlashInfer trtllm-gen decode attention."""

import pytest

flashinfer_backend = pytest.importorskip("vllm.v1.attention.backends.flashinfer")
sm_count_for = flashinfer_backend._trtllm_gen_decode_sm_count

# Qwen3.6-35B-A3B full attention on a 212-SM GPU, CUDA-graph capture
# (max_seq_len = max_model_len), vLLM's default 394 MiB workspace.
HQ, HKV, D, SM = 16, 2, 256, 212
MAX_MODEL_LEN = 262144
WS = 394 * 1024 * 1024


def splits(num_reqs, max_q_len=4, max_seq_len=MAX_MODEL_LEN, max_kv=16384, ws=WS):
    sm = sm_count_for(num_reqs, max_q_len, max_seq_len, HQ, HKV, D, SM, ws, max_kv)
    # FlashInfer: numCtasPerSeqKv = max(1, sm // (num_kv_heads * num_reqs)).
    return max(1, sm // (HKV * num_reqs)), sm


@pytest.mark.parametrize("num_reqs", [8, 40, 52, 54, 128])
@pytest.mark.parametrize("max_q_len", [1, 4])
def test_split_reaches_kv_cap(num_reqs, max_q_len):
    # 262144 / 16384 = 16 splits, including where FlashInfer would pick
    # Persistent (54+ requests).
    assert splits(num_reqs, max_q_len)[0] == 16


@pytest.mark.parametrize("num_reqs", [1, 2, 4, 6])
def test_small_batches_keep_flashinfer_choice(num_reqs):
    # FlashInfer's one-wave split is already >= 16 here.
    assert splits(num_reqs) == (SM // (HKV * num_reqs), SM)


def test_eager_split_follows_max_seq_len():
    assert splits(54, max_seq_len=30000)[0] == 2
    assert splits(8, max_seq_len=30000) == (SM // 16, SM)


def test_disabled_and_ungrouped_queries_keep_flashinfer_choice():
    assert splits(54, max_kv=0) == (1, SM)
    # 20 tokens x 8 heads > 128 rows: more than one Q CTA per request.
    assert splits(54, max_q_len=20) == (1, SM)


def test_workspace_caps_split():
    # Gmem-reduction partials (32 rows x (8 + 4 * 256) B per CTA) must fit
    # in half the workspace.
    k, _ = splits(512, ws=WS)
    assert k == (WS // 2) // (32 * (8 + 4 * D) * HKV * 512)
    assert 1 < k < 16
    assert splits(4096, ws=WS) == (1, SM)
