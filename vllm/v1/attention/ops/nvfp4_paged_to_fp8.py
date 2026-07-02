# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-dequantize paged NVFP4 KV cache to an fp8 mock cache for prefill.

The trtllm-gen FMHA runs an fp8 MMA and unpacks e2m1 -> fp8 inside the
compute-bound prefill kernel; this module does the same unpack as a cheaper
separate memory-bound pass instead. One Triton launch reads the paged NVFP4
cache (K block scales linear, V block scales per-page 16x8 swizzled -- see
nvfp4_kv_cache_kernels.cu) and writes an fp8 mock cache holding
``e2m1 * block_scale``, so the caller's bmm1/bmm2 scales apply unchanged.
Enabled via ``attention_config.nvfp4_kv_prefill_predequant``; decode keeps
the native NVFP4 path.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _decode_e2m1(packed_byte, nibble):
    """Unpack one e2m1 (fp4) nibble of a packed uint8 to float32."""
    code = tl.where(nibble == 0, packed_byte & 0xF, (packed_byte >> 4) & 0xF)
    sign = tl.where((code & 0x8) != 0, -1.0, 1.0)
    mag = code & 0x7
    exp = mag >> 1
    mant = mag & 1
    pow2 = (1 << tl.maximum(exp - 1, 0)).to(tl.float32)
    return (
        tl.where(
            exp == 0,
            mant.to(tl.float32) * 0.5,
            (1.0 + mant.to(tl.float32) * 0.5) * pow2,
        )
        * sign
    )


@triton.jit
def _nvfp4_paged_to_fp8_kernel(
    k_data_ptr,
    v_data_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    mock_ptr,
    d0,
    d1,
    d2,
    d3,
    s0,
    s1,
    s2,
    s3,
    b0,
    b1,
    m0,
    m1,
    m2,
    m3,
    pages_per_seq,
    PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE_DIM: tl.constexpr,
):
    # One program handles K and V of one (mock page, kv head).
    page = tl.program_id(0)
    head = tl.program_id(1)
    orig = tl.load(
        block_tables_ptr + (page // pages_per_seq) * b0 + (page % pages_per_seq) * b1
    )
    if orig > 0:  # page index <= 0 is padding/unused
        tok = tl.arange(0, PAGE)[:, None]
        dim = tl.arange(0, HEAD_DIM)[None, :]
        byte = dim // 2
        nibble = dim % 2
        sblk = dim // 16
        mock_page = page + 1
        base = orig * d0 + head * d1 + tok * d2 + byte * d3
        sbase = orig * s0 + head * s1

        # K: linear scale layout.
        k_val = _decode_e2m1(tl.load(k_data_ptr + base).to(tl.int32), nibble)
        k_sc = (
            tl.load(k_scale_ptr + sbase + tok * s2 + sblk * s3)
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float32)
        )
        tl.store(
            mock_ptr + mock_page * m0 + head * m2 + tok * m3 + dim,
            (k_val * k_sc).to(mock_ptr.dtype.element_ty),
        )

        # V: per-page 16x8 swizzled scale layout (swizzle_scale_offset).
        swz_t = (tok // 4) * 4 + (sblk // (SCALE_DIM // 4))
        swz_s = (sblk % (SCALE_DIM // 4)) * 4 + (tok % 4)
        v_val = _decode_e2m1(tl.load(v_data_ptr + base).to(tl.int32), nibble)
        v_sc = (
            tl.load(v_scale_ptr + sbase + swz_t * s2 + swz_s * s3)
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float32)
        )
        tl.store(
            mock_ptr + mock_page * m0 + m1 + head * m2 + tok * m3 + dim,
            (v_val * v_sc).to(mock_ptr.dtype.element_ty),
        )


def predequant_nvfp4_paged_to_fp8(
    kv_data: tuple[torch.Tensor, torch.Tensor],
    kv_scales: tuple[torch.Tensor, torch.Tensor],
    block_tables: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an fp8 mock KV cache for the prefill pages of an NVFP4 cache.

    Args:
        kv_data: (k_data, v_data) e2m1 views from
            ``nvfp4_kv_cache_split_views`` of the HND-permuted cache, each
            [num_blocks, num_kv_heads, page, head_dim // 2] uint8.
        kv_scales: (k_scales, v_scales) e4m3 block-scale views, each
            [num_blocks, num_kv_heads, page, head_dim // 16] uint8.
        block_tables: [num_prefills, pages_per_seq] int32 prefill block table.

    Returns:
        (mock_cache [G + 1, 2, num_kv_heads, page, head_dim] fp8,
         mock_block_tables [num_prefills, pages_per_seq] int32), where
        G = block_tables.numel(). Pass ``kv_cache_sf=None`` with the mock.
    """
    k_data, v_data = kv_data
    k_scales, v_scales = kv_scales
    # The single-launch kernel uses one stride set for K and V each.
    assert k_data.stride() == v_data.stride()
    assert k_scales.stride() == v_scales.stride()
    num_kv_heads, page = k_data.shape[1], k_data.shape[2]
    head_dim = k_data.shape[3] * 2
    scale_dim = k_scales.shape[3]
    num_prefills, pages_per_seq = block_tables.shape
    num_pages = num_prefills * pages_per_seq
    # torch.empty: every valid page is fully written; page 0 and padding
    # pages are never read (same contract as trtllm_prefill_attn_kvfp8_dequant).
    mock = torch.empty(
        (num_pages + 1, 2, num_kv_heads, page, head_dim),
        dtype=current_platform.fp8_dtype(),
        device=k_data.device,
    )
    ms = mock.stride()
    _nvfp4_paged_to_fp8_kernel[(num_pages, num_kv_heads)](
        k_data,
        v_data,
        k_scales,
        v_scales,
        block_tables,
        mock,
        *k_data.stride(),
        *k_scales.stride(),
        *block_tables.stride(),
        ms[0],
        ms[1],
        ms[2],
        ms[3],
        pages_per_seq,
        PAGE=page,
        HEAD_DIM=head_dim,
        SCALE_DIM=scale_dim,
    )
    mock_block_tables = torch.arange(
        1, num_pages + 1, dtype=torch.int32, device=block_tables.device
    ).reshape(num_prefills, pages_per_seq)
    return mock, mock_block_tables
