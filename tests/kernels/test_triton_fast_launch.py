# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FastLaunch must launch the same compiled kernel as ``kernel[grid](...)``,
including across argument specializations, and must find the kernel the JIT
path cached (else every launch silently takes the slow path).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if not current_platform.is_cuda():
    pytest.skip(reason="needs CUDA", allow_module_level=True)

from vllm.triton_utils.fast_launch import FastLaunch  # noqa: E402


@triton.jit
def _axpb_kernel(x_ptr, out_ptr, n, a, b, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * a + b, mask=mask)


@pytest.mark.parametrize("n", [1, 16, 1000])
@pytest.mark.parametrize("offset", [0, 1])
def test_fast_launch_matches_jit(n: int, offset: int) -> None:
    # n = 1 and n % 16 == 0 specialize the int, offset 1 the pointer alignment.
    x = torch.randn(n + offset, device="cuda")[offset:]
    out_jit = torch.empty_like(x)
    out_fast = torch.full_like(x, float("nan"))
    grid = (triton.cdiv(n, 256),)
    _axpb_kernel[grid](x, out_jit, n, 3, 0.5, BLOCK=256)
    fast = FastLaunch(_axpb_kernel)
    cached = set(_axpb_kernel.device_caches[x.device.index][0])
    fast[grid](x, out_fast, n, 3, 0.5, BLOCK=256)
    assert not fast._off
    # The JIT launch above cached this specialization: no new compile.
    assert set(_axpb_kernel.device_caches[x.device.index][0]) == cached
    torch.testing.assert_close(out_fast, out_jit, atol=0, rtol=0)
