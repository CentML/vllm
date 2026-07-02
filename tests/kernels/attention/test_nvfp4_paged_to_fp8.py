# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused NVFP4-paged -> fp8 pre-dequant kernel must be byte-identical to a
pure-torch reference built from ``nvfp4_kv_dequantize`` (K scales linear, V
scales de-swizzled per-page 16x8 -- see nvfp4_kv_cache_kernels.cu)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="NVFP4 KV cache requires compute capability of 10 or above.",
        allow_module_level=True,
    )
if not has_flashinfer():
    pytest.skip(reason="flashinfer required.", allow_module_level=True)

from flashinfer.fp4_quantization import nvfp4_kv_dequantize  # noqa: E402

from vllm.utils.torch_utils import (  # noqa: E402
    nvfp4_kv_cache_full_dim,
    nvfp4_kv_cache_split_views,
)
from vllm.v1.attention.ops.nvfp4_paged_to_fp8 import (  # noqa: E402
    predequant_nvfp4_paged_to_fp8,
)

NUM_KV_HEADS = 4
HEAD_DIM = 128
PAGE = 16
SCALE_DIM = HEAD_DIM // 16


def _alloc_nvfp4_cache(num_blocks: int) -> torch.Tensor:
    full_dim = nvfp4_kv_cache_full_dim(HEAD_DIM)
    shape = (num_blocks, 2, PAGE, NUM_KV_HEADS, full_dim)
    stride_order = (0, 1, 3, 2, 4)
    physical = [shape[i] for i in stride_order]
    inverse = [stride_order.index(i) for i in range(5)]
    return torch.zeros(*physical, dtype=torch.uint8, device="cuda").permute(*inverse)


@pytest.mark.parametrize("num_tokens", [512, 1000])
@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_predequant_matches_torch_reference(num_tokens: int, seed: int) -> None:
    set_random_seed(seed)
    fp8 = current_platform.fp8_dtype()
    num_pages = (num_tokens + PAGE - 1) // PAGE
    # Page 0 is reserved (the kernel treats page index <= 0 as padding), so
    # real data lives in pages 1..num_pages.
    cache = _alloc_nvfp4_cache(num_pages + 1)

    key = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, device="cuda").bfloat16()
    value = torch.randn_like(key)
    k_scale = (key.abs().amax().float() / 448.0).clamp(min=1e-8)
    v_scale = (value.abs().amax().float() / 448.0).clamp(min=1e-8)
    slot_mapping = torch.arange(
        PAGE, PAGE + num_tokens, device="cuda", dtype=torch.long
    )
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        cache[:, 0],
        cache[:, 1],
        slot_mapping,
        "nvfp4",
        k_scale.clone(),
        v_scale.clone(),
    )
    kv_data, kv_scales = nvfp4_kv_cache_split_views(cache.permute(0, 1, 3, 2, 4))

    block_tables = torch.arange(
        1, num_pages + 1, device="cuda", dtype=torch.int32
    ).reshape(1, -1)
    mock, mock_bt = predequant_nvfp4_paged_to_fp8(kv_data, kv_scales, block_tables)
    assert mock.shape == (num_pages + 1, 2, NUM_KV_HEADS, PAGE, HEAD_DIM)
    assert torch.equal(
        mock_bt, torch.arange(1, num_pages + 1, device="cuda").int().reshape(1, -1)
    )

    # Torch reference: dequant with block scales only (global scale 1.0), K
    # scales read linearly, V scales de-swizzled per-page 16x8.
    tok = torch.arange(PAGE, device="cuda")
    sbl = torch.arange(SCALE_DIM, device="cuda")
    tok_g, sbl_g = torch.meshgrid(tok, sbl, indexing="ij")
    group = SCALE_DIM // 4
    swz_t = (tok_g // 4) * 4 + (sbl_g // group)
    swz_s = (sbl_g % group) * 4 + (tok_g % 4)
    one = torch.tensor([1.0], device="cuda")
    pages = block_tables.reshape(-1).long()

    def reference_half(data: torch.Tensor, sf: torch.Tensor, swizzled: bool):
        d = data[pages]
        s = sf[pages]
        if swizzled:
            s = s[:, :, swz_t, swz_s]
        deq = nvfp4_kv_dequantize(
            d.reshape(-1, HEAD_DIM // 2).contiguous(),
            s.reshape(-1, SCALE_DIM).contiguous(),
            one,
            output_dtype=torch.bfloat16,
        )
        return deq.reshape(num_pages, NUM_KV_HEADS, PAGE, HEAD_DIM).to(fp8)

    ref_k = reference_half(kv_data[0], kv_scales[0], swizzled=False)
    ref_v = reference_half(kv_data[1], kv_scales[1], swizzled=True)
    assert torch.equal(mock[1:, 0].view(torch.uint8), ref_k.view(torch.uint8))
    assert torch.equal(mock[1:, 1].view(torch.uint8), ref_v.view(torch.uint8))
