# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU byte-level regressions without importing the CUDA worker runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _worker_functions():
    source = Path(__file__).resolve().parents[3] / "vllm/v1/worker/utils.py"
    names = {"widest_alias_views", "_copy_kv_cache_blocks_inplace_per_storage"}
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "np": np,
        "envs": SimpleNamespace(VLLM_FUSED_KV_BLOCK_COPY=False),
        "async_tensor_h2d": lambda data, device: torch.from_numpy(data).to(device),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("split", [1, 2])
def test_widest_alias_preserves_attention_tail_and_other_blocks(reverse, split):
    functions = _worker_functions()
    raw = torch.arange(4 * 32 + 16, dtype=torch.uint8)
    pages = raw[:128].view(4, 32)
    narrow = pages[:, :12]
    attention = pages.view(4 * split, 32 // split)
    views = [narrow, attention] if not reverse else [attention, narrow]
    before = raw.clone()
    functions["_copy_kv_cache_blocks_inplace_per_storage"](
        views, 4, np.array([[0, 2]], dtype=np.int64)
    )
    expected = before.clone()
    expected[64:96] = before[:32]
    assert torch.equal(raw, expected)


def test_equal_span_keeps_first_view_and_group_order():
    functions = _worker_functions()
    raw = torch.zeros(256, dtype=torch.uint8)
    first = raw[:128].view(4, 32)
    second = raw[128:].view(4, 32)
    alias = first.view(4, 8, 4)
    selected = functions["widest_alias_views"]([first, second, alias], 4)
    assert selected[0] is first
    assert selected[1] is second
    assert len(selected) == 2


def test_byte_extent_not_element_count_selects_view():
    functions = _worker_functions()
    raw = torch.zeros(256, dtype=torch.uint8)
    byte_view = raw.view(4, 64)[:, :20]
    word_view = raw.view(torch.int32).view(4, 16)[:, :8]
    selected = functions["widest_alias_views"]([byte_view, word_view], 4)
    assert len(selected) == 1
    assert selected[0] is word_view


def test_chained_copy_reads_original_sources():
    functions = _worker_functions()
    raw = torch.arange(4 * 32 + 16, dtype=torch.uint8)
    pages = raw[:128].view(4, 32)
    before = raw.clone()
    functions["_copy_kv_cache_blocks_inplace_per_storage"](
        [pages[:, :12], pages], 4, np.array([[0, 1], [1, 2]], dtype=np.int64)
    )
    expected = before.clone()
    expected[32:64] = before[:32]
    expected[64:96] = before[32:64]
    assert torch.equal(raw, expected)
