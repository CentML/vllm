# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused QK-RMSNorm + (partial) RoPE (+ gate copy) Triton kernels.

Currently used by the Qwen3.5 attention path (``attn_output_gate`` with
NeoX-style partial RoPE). The unfused reference sequence is
``split -> GemmaRMSNorm -> RoPE -> gate chunk``; this collapses it into a
single Triton launch. See :func:`fused_qk_rmsnorm_rope_gate` and its
token-tile variant :func:`fused_qk_rmsnorm_rope`, which can leave the gate in
place and emit q as FP8.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _fused_qk_rmsnorm_rope_gate_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_gate_stride_t,
    k_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    gate_out_stride_t,
    cache_stride_p,
    positions_stride_m,
    positions_stride_t,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    norm_beta: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)

    if is_k:
        in_base = k_ptr + token * k_stride_t + local_head * head_dim
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * k_out_stride_t + local_head * head_dim
    else:
        in_base = q_gate_ptr + token * q_gate_stride_t + local_head * 2 * head_dim
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * q_out_stride_t + local_head * head_dim

    # --- RMSNorm: variance over the full head_dim ---
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < head_dim
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32) + norm_beta
    # Round-trip through INPUT_DTYPE so the RoPE input matches the bf16-storage
    # behavior of the unfused (qk_rmsnorm -> memory -> apply_rope) reference path.
    x_norm = (x * inv_rms * w).to(INPUT_DTYPE).to(tl.float32)

    # --- Pass-through tail [rotary_dim, head_dim): RMSNorm-only, no rotation ---
    # The rotary head [0, rotary_dim) will be overwritten by the RoPE store below.
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= rotary_dim)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)

    # --- Partial RoPE on the first rotary_dim elements ---
    # Triton lacks easy sub-vector slicing of x_norm, so we recompute the
    # normalized rotary halves on a smaller block (next_pow2(half_rotary)).
    # The extra ~rotary_dim element reload hits L1, so the cost is negligible.
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < half_rotary
    x_rot1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    x_rot2 = tl.load(in_base + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    w_rot1 = (
        tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32) + norm_beta
    )
    w_rot2 = (
        tl.load(w_ptr + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
        + norm_beta
    )
    x_rot1 = (x_rot1 * inv_rms * w_rot1).to(INPUT_DTYPE).to(tl.float32)
    x_rot2 = (x_rot2 * inv_rms * w_rot2).to(INPUT_DTYPE).to(tl.float32)

    # Always use int64 for positions to avoid overflow in address computation.
    pos_t = tl.load(positions_ptr + token * positions_stride_t).to(tl.int64)
    if HAS_MROPE:
        pos_h = tl.load(
            positions_ptr + positions_stride_m + token * positions_stride_t
        ).to(tl.int64)
        pos_w = tl.load(
            positions_ptr + 2 * positions_stride_m + token * positions_stride_t
        ).to(tl.int64)
        is_h = (rot_offs % 3 == 1) & (rot_offs < 3 * MROPE_SECTION_H)
        is_w = (rot_offs % 3 == 2) & (rot_offs < 3 * MROPE_SECTION_W)
        pos = tl.where(is_h, pos_h, tl.where(is_w, pos_w, pos_t))
    else:
        pos = pos_t
    cache_offset = pos * cache_stride_p
    cos = tl.load(
        cos_sin_cache_ptr + cache_offset + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_offset + half_rotary + rot_offs,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)

    o1 = x_rot1 * cos - x_rot2 * sin
    o2 = x_rot2 * cos + x_rot1 * sin
    tl.store(out_base + rot_offs, o1, mask=rot_mask)
    tl.store(out_base + half_rotary + rot_offs, o2, mask=rot_mask)

    # --- Gate copy (q heads only, verbatim) ---
    if not is_k:
        gate_in_base = in_base + head_dim
        gate_out_base = gate_out_ptr + token * gate_out_stride_t + local_head * head_dim
        g = tl.load(gate_in_base + head_offs, mask=head_mask, other=0.0)
        tl.store(gate_out_base + head_offs, g, mask=head_mask)


def _check_inputs(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | tuple[int, int, int] | None,
) -> tuple[bool, int, int, int, int]:
    """Validate the fused-op inputs.

    Returns ``(has_mrope, mrope_section_h, mrope_section_w, positions_stride_m,
    positions_stride_t)``.
    """
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim must be a positive even integer <= head_dim, "
            f"got rotary_dim={rotary_dim}, head_dim={head_dim}"
        )
    if q_gate.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q_gate.dtype:
        raise ValueError(
            "q_gate and k must have the same FP16 or BF16 dtype, "
            f"got {q_gate.dtype} and {k.dtype}"
        )
    for name, tensor in (
        ("q_gate", q_gate),
        ("k", k),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("cos_sin_cache", cos_sin_cache),
    ):
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous in its last dimension")

    if positions.ndim not in (1, 2):
        raise ValueError(f"positions must be 1D or 2D, got shape={positions.shape}")
    if positions.shape[-1] != q_gate.shape[0]:
        raise ValueError(
            "positions token dimension must match q_gate, "
            f"got {positions.shape[-1]} and {q_gate.shape[0]}"
        )

    has_mrope = positions.ndim == 2
    if has_mrope:
        if positions.shape[0] != 3:
            raise ValueError(
                f"MRoPE positions must have shape (3, n_tokens), got {positions.shape}"
            )
        if mrope_section is None or len(mrope_section) != 3:
            raise ValueError("mrope_section must contain the T/H/W frequency counts")
        if sum(mrope_section) != rotary_dim // 2:
            raise ValueError(
                "mrope_section must sum to rotary_dim // 2, "
                f"got {mrope_section} and rotary_dim={rotary_dim}"
            )
        positions_stride_m, positions_stride_t = positions.stride()
        return (
            True,
            mrope_section[1],
            mrope_section[2],
            positions_stride_m,
            positions_stride_t,
        )
    if mrope_section is not None:
        raise ValueError("mrope_section requires 2D MRoPE positions")
    return False, 0, 0, 0, positions.stride(0)


def fused_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | tuple[int, int, int] | None = None,
    norm_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused split + QK-RMSNorm + (partial) RoPE + gate copy for Qwen attn.

    Args:
        q_gate: (n_tokens, num_q_heads * 2 * head_dim) -- per head: [q|gate]
        k: (n_tokens, num_kv_heads * head_dim)
        q_weight: (head_dim,) RMSNorm weight
        k_weight: (head_dim,) RMSNorm weight
        cos_sin_cache: (max_pos, rotary_dim) packed [cos|sin]
        positions: (n_tokens,) or (3, n_tokens) int32 or int64
        eps: RMSNorm epsilon
        num_q_heads: number of Q heads (after TP split)
        num_kv_heads: number of KV heads (after TP split)
        head_dim: per-head dimension
        rotary_dim: rotary dimension; must be even and <= head_dim
        mrope_section: interleaved T/H/W frequency counts for 2D positions
        norm_beta: scalar added to the RMSNorm weight

    Returns:
        (q_out, k_out, gate_out) -- all contiguous (n_tokens, heads * head_dim).
        ``gate_out`` is the raw (pre-sigmoid) gate.

    """
    (
        has_mrope,
        mrope_section_h,
        mrope_section_w,
        positions_stride_m,
        positions_stride_t,
    ) = _check_inputs(
        q_gate,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        head_dim,
        rotary_dim,
        mrope_section,
    )

    n_tokens = q_gate.shape[0]
    q_out = torch.empty(
        (n_tokens, num_q_heads * head_dim), dtype=q_gate.dtype, device=q_gate.device
    )
    k_out = torch.empty(
        (n_tokens, num_kv_heads * head_dim), dtype=k.dtype, device=k.device
    )
    gate_out = torch.empty_like(q_out)
    if n_tokens == 0:
        return q_out, k_out, gate_out

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rot_half_block = triton.next_power_of_2(half_rotary)
    num_warps = max(1, head_block // 64)

    grid = (n_tokens, num_q_heads + num_kv_heads)
    _fused_qk_rmsnorm_rope_gate_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        cos_sin_cache.stride(0),
        positions_stride_m,
        positions_stride_t,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        half_rotary,
        eps,
        norm_beta=norm_beta,
        INPUT_DTYPE=tl.bfloat16 if q_gate.dtype == torch.bfloat16 else tl.float16,
        HEAD_BLOCK=head_block,
        ROT_HALF_BLOCK=rot_half_block,
        HAS_PASS=rotary_dim < head_dim,
        HAS_MROPE=has_mrope,
        MROPE_SECTION_H=mrope_section_h,
        MROPE_SECTION_W=mrope_section_w,
        num_warps=num_warps,
        num_stages=2,
    )
    return q_out, k_out, gate_out


@triton.jit
def _static_fp8_quant(x, inv_scale):
    """``QuantFP8(static, per-tensor).forward_native`` as Inductor compiles it.

    ``inv_scale`` is ``1.0 / scale`` in fp32; NaN propagates through the clamp
    like ``triton_helpers.maximum/minimum``.
    """
    y = x * inv_scale
    y = tl.where((y > -448.0) | (y != y), y, -448.0)
    y = tl.where((y < 448.0) | (y != y), y, 448.0)
    return y.to(tl.float8e4nv)


@triton.jit
def _norm_rope_head_rows(
    in_ptr,
    in_stride_t,
    out_ptr,
    out_stride_t,
    w_ptr,
    cos_sin_cache_ptr,
    cache_stride_p,
    positions_ptr,
    positions_stride_m,
    positions_stride_t,
    q_scale_ptr,
    gate_out_ptr,
    gate_out_stride_t,
    tok0,
    num_tokens,
    IN_HEAD_STRIDE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    ROWS_BLOCK: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    norm_beta: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
    OUT_FP8: tl.constexpr,
    STORE_GATE: tl.constexpr,
):
    """RMSNorm + partial RoPE of ``TOKENS x NUM_HEADS`` heads (one row each).

    The per-element math is ``_fused_qk_rmsnorm_rope_gate_kernel``'s. The
    ``[rows, HEAD_BLOCK // 2, 2]`` tile gets that kernel's layout along a head
    (2 elements per thread, the pairs spread over all lanes and
    ``HEAD_BLOCK // 64`` warps), so the sum of squares uses its exact reduction
    tree: in-thread pair, lane butterfly, then the cross-warp round.
    """
    ROWS: tl.constexpr = TOKENS * NUM_HEADS
    rows = tl.arange(0, ROWS_BLOCK)
    tok = tok0 + rows // NUM_HEADS
    head = rows % NUM_HEADS
    row_mask = (rows < ROWS) & (tok < num_tokens)
    tok = tok.to(tl.int64)
    in_row = in_ptr + tok * in_stride_t + head * IN_HEAD_STRIDE
    out_row = out_ptr + tok * out_stride_t + head * head_dim

    # Issue the position -> cos/sin chain and all head loads before the
    # reduction's barrier.
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < half_rotary
    rot_mask2 = row_mask[:, None] & rot_mask[None, :]
    pos_t = tl.load(
        positions_ptr + tok * positions_stride_t, mask=row_mask, other=0
    ).to(tl.int64)
    if HAS_MROPE:
        pos_h = tl.load(
            positions_ptr + positions_stride_m + tok * positions_stride_t,
            mask=row_mask,
            other=0,
        ).to(tl.int64)
        pos_w = tl.load(
            positions_ptr + 2 * positions_stride_m + tok * positions_stride_t,
            mask=row_mask,
            other=0,
        ).to(tl.int64)
        is_h = (rot_offs % 3 == 1) & (rot_offs < 3 * MROPE_SECTION_H)
        is_w = (rot_offs % 3 == 2) & (rot_offs < 3 * MROPE_SECTION_W)
        pos = tl.where(
            is_h[None, :],
            pos_h[:, None],
            tl.where(is_w[None, :], pos_w[:, None], pos_t[:, None]),
        )
    else:
        pos = pos_t[:, None]
    cache_row = cos_sin_cache_ptr + pos * cache_stride_p
    cos = tl.load(cache_row + rot_offs[None, :], mask=rot_mask2, other=0.0).to(
        tl.float32
    )
    sin = tl.load(
        cache_row + half_rotary + rot_offs[None, :], mask=rot_mask2, other=0.0
    ).to(tl.float32)

    head_offs = tl.arange(0, HEAD_BLOCK // 2)[:, None] * 2 + tl.arange(0, 2)[None, :]
    head_mask = head_offs < head_dim
    mask = row_mask[:, None, None] & head_mask[None, :, :]
    x = tl.load(in_row[:, None, None] + head_offs[None, :, :], mask=mask, other=0.0)
    x = x.to(tl.float32)
    in_rot = in_row[:, None] + rot_offs[None, :]
    x_rot1 = tl.load(in_rot, mask=rot_mask2, other=0.0).to(tl.float32)
    x_rot2 = tl.load(in_rot + half_rotary, mask=rot_mask2, other=0.0).to(tl.float32)
    if STORE_GATE:
        # Verbatim copy of the gate half of each q head (as the per-head kernel).
        g = tl.load(in_row[:, None, None] + head_dim + head_offs[None, :, :], mask=mask)
        gate_row = gate_out_ptr + tok * gate_out_stride_t + head * head_dim
        tl.store(gate_row[:, None, None] + head_offs[None, :, :], g, mask=mask)

    # --- RMSNorm: variance over the full head_dim ---
    var = tl.sum(tl.sum(x * x, axis=2), axis=1) / head_dim
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32) + norm_beta
    x_norm = (x * inv_rms[:, None, None] * w[None, :, :]).to(INPUT_DTYPE).to(tl.float32)
    if OUT_FP8:
        inv_scale = 1.0 / tl.load(q_scale_ptr)

    # --- Pass-through tail [rotary_dim, head_dim): RMSNorm-only ---
    if HAS_PASS:
        pass_mask = mask & (head_offs >= rotary_dim)[None, :, :]
        if OUT_FP8:
            x_pass = _static_fp8_quant(x_norm, inv_scale)
        else:
            x_pass = x_norm.to(INPUT_DTYPE)
        tl.store(out_row[:, None, None] + head_offs[None, :, :], x_pass, mask=pass_mask)

    # --- Partial RoPE on the first rotary_dim elements ---
    w_rot1 = (
        tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32) + norm_beta
    )
    w_rot2 = (
        tl.load(w_ptr + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
        + norm_beta
    )
    x_rot1 = (x_rot1 * inv_rms[:, None] * w_rot1[None, :]).to(INPUT_DTYPE)
    x_rot2 = (x_rot2 * inv_rms[:, None] * w_rot2[None, :]).to(INPUT_DTYPE)
    x_rot1 = x_rot1.to(tl.float32)
    x_rot2 = x_rot2.to(tl.float32)
    o1 = (x_rot1 * cos - x_rot2 * sin).to(INPUT_DTYPE)
    o2 = (x_rot2 * cos + x_rot1 * sin).to(INPUT_DTYPE)
    if OUT_FP8:
        o1 = _static_fp8_quant(o1.to(tl.float32), inv_scale)
        o2 = _static_fp8_quant(o2.to(tl.float32), inv_scale)
    out_rot = out_row[:, None] + rot_offs[None, :]
    tl.store(out_rot, o1, mask=rot_mask2)
    tl.store(out_rot + half_rotary, o2, mask=rot_mask2)


@triton.jit(do_not_specialize=["num_tokens"])
def _fused_qk_rmsnorm_rope_tokens_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_scale_ptr,
    gate_out_ptr,
    num_tokens,
    q_gate_stride_t,
    k_stride_t,
    q_out_stride_t,
    gate_out_stride_t,
    k_out_stride_t,
    cache_stride_p,
    positions_stride_m,
    positions_stride_t,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    norm_beta: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
    Q_TOKENS: tl.constexpr,
    K_TOKENS: tl.constexpr,
    Q_ROWS_BLOCK: tl.constexpr,
    K_ROWS_BLOCK: tl.constexpr,
    Q_FP8: tl.constexpr,
    STORE_GATE: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    # CTAs [0, cdiv(n, Q_TOKENS)) take all q heads of Q_TOKENS tokens; the rest
    # take all kv heads of K_TOKENS tokens.
    pid = tl.program_id(0)
    num_q_ctas = tl.cdiv(num_tokens, Q_TOKENS)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
    if pid < num_q_ctas:
        _norm_rope_head_rows(
            q_gate_ptr,
            q_gate_stride_t,
            q_out_ptr,
            q_out_stride_t,
            q_weight_ptr,
            cos_sin_cache_ptr,
            cache_stride_p,
            positions_ptr,
            positions_stride_m,
            positions_stride_t,
            q_scale_ptr,
            gate_out_ptr,
            gate_out_stride_t,
            pid * Q_TOKENS,
            num_tokens,
            2 * head_dim,
            num_q_heads,
            Q_TOKENS,
            Q_ROWS_BLOCK,
            head_dim,
            rotary_dim,
            half_rotary,
            eps,
            norm_beta,
            INPUT_DTYPE,
            HEAD_BLOCK,
            ROT_HALF_BLOCK,
            HAS_PASS,
            HAS_MROPE,
            MROPE_SECTION_H,
            MROPE_SECTION_W,
            Q_FP8,
            STORE_GATE,
        )
    else:
        _norm_rope_head_rows(
            k_ptr,
            k_stride_t,
            k_out_ptr,
            k_out_stride_t,
            k_weight_ptr,
            cos_sin_cache_ptr,
            cache_stride_p,
            positions_ptr,
            positions_stride_m,
            positions_stride_t,
            q_scale_ptr,
            gate_out_ptr,
            gate_out_stride_t,
            (pid - num_q_ctas) * K_TOKENS,
            num_tokens,
            head_dim,
            num_kv_heads,
            K_TOKENS,
            K_ROWS_BLOCK,
            head_dim,
            rotary_dim,
            half_rotary,
            eps,
            norm_beta,
            INPUT_DTYPE,
            HEAD_BLOCK,
            ROT_HALF_BLOCK,
            HAS_PASS,
            HAS_MROPE,
            MROPE_SECTION_H,
            MROPE_SECTION_W,
            False,
            False,
        )
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _tokens_launch_config(
    num_q_heads: int, num_kv_heads: int, head_block: int
) -> tuple[int, int, int]:
    """(Q_TOKENS, K_TOKENS, num_warps): 16 head rows on the warps of one head."""
    q_tokens = max(1, 16 // num_q_heads)
    k_tokens = max(1, 16 // num_kv_heads)
    return q_tokens, k_tokens, head_block // 64


def fused_qk_rmsnorm_rope(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_scale: torch.Tensor | None,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | None = None,
    norm_beta: float = 0.0,
    store_gate: bool = False,
    launch_config: tuple[int, int, int] | None = None,
    launch_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split + QK-RMSNorm + (partial) RoPE for Qwen attn, one CTA per token tile.

    Same inputs and q/k values as :func:`fused_qk_rmsnorm_rope_gate`. The gate
    is copied only with ``store_gate``; otherwise read it in place as
    ``q_gate.view(n, num_q_heads, 2, head_dim)[:, :, 1]`` (a consumer that
    reads it strided, e.g. ``vllm::attn_gate_mxfp8_quant``). With ``q_scale``
    (the attention layer's static per-tensor ``_q_scale``), q is returned as
    e4m3, bit-identical to ``QuantFP8.forward_native`` of the bf16 q.

    Returns:
        (q_out, k_out, gate_out) -- contiguous (n_tokens, heads * head_dim);
        ``gate_out`` has zero columns unless ``store_gate``.

    """
    (
        has_mrope,
        mrope_section_h,
        mrope_section_w,
        positions_stride_m,
        positions_stride_t,
    ) = _check_inputs(
        q_gate,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        head_dim,
        rotary_dim,
        mrope_section,
    )
    head_block = triton.next_power_of_2(head_dim)
    if head_block < 64:
        raise ValueError(f"head_dim must be >= 33, got {head_dim}")
    if q_scale is not None and (q_scale.numel() != 1 or q_scale.dtype != torch.float32):
        raise ValueError("q_scale must be a single fp32 per-tensor scale")

    n_tokens = q_gate.shape[0]
    q_out = torch.empty(
        (n_tokens, num_q_heads * head_dim),
        dtype=q_gate.dtype if q_scale is None else torch.float8_e4m3fn,
        device=q_gate.device,
    )
    k_out = torch.empty(
        (n_tokens, num_kv_heads * head_dim), dtype=k.dtype, device=k.device
    )
    gate_out = torch.empty(
        (n_tokens, num_q_heads * head_dim if store_gate else 0),
        dtype=q_gate.dtype,
        device=q_gate.device,
    )
    if n_tokens == 0:
        return q_out, k_out, gate_out

    q_tokens, k_tokens, num_warps = launch_config or _tokens_launch_config(
        num_q_heads, num_kv_heads, head_block
    )
    if launch_pdl is None:
        # As the fused RMSNorm -> MXFP8 producer: PDL below 4096 tokens.
        launch_pdl = n_tokens < 4096 and current_platform.is_arch_support_pdl()
    grid = (triton.cdiv(n_tokens, q_tokens) + triton.cdiv(n_tokens, k_tokens),)
    _fused_qk_rmsnorm_rope_tokens_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_out if q_scale is None else q_scale,
        gate_out,
        n_tokens,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        gate_out.stride(0),
        k_out.stride(0),
        cos_sin_cache.stride(0),
        positions_stride_m,
        positions_stride_t,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        rotary_dim // 2,
        eps,
        norm_beta=norm_beta,
        INPUT_DTYPE=tl.bfloat16 if q_gate.dtype == torch.bfloat16 else tl.float16,
        HEAD_BLOCK=head_block,
        ROT_HALF_BLOCK=triton.next_power_of_2(rotary_dim // 2),
        HAS_PASS=rotary_dim < head_dim,
        HAS_MROPE=has_mrope,
        MROPE_SECTION_H=mrope_section_h,
        MROPE_SECTION_W=mrope_section_w,
        Q_TOKENS=q_tokens,
        K_TOKENS=k_tokens,
        Q_ROWS_BLOCK=triton.next_power_of_2(q_tokens * num_q_heads),
        K_ROWS_BLOCK=triton.next_power_of_2(k_tokens * num_kv_heads),
        Q_FP8=q_scale is not None,
        STORE_GATE=store_gate,
        LAUNCH_PDL=launch_pdl,
        launch_pdl=launch_pdl,
        num_warps=num_warps,
    )
    return q_out, k_out, gate_out


def _fused_qk_rmsnorm_rope_fake(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_scale: torch.Tensor | None,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | None = None,
    norm_beta: float = 0.0,
    store_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_tokens = q_gate.shape[0]
    q_dtype = q_gate.dtype if q_scale is None else torch.float8_e4m3fn
    return (
        q_gate.new_empty((n_tokens, num_q_heads * head_dim), dtype=q_dtype),
        k.new_empty((n_tokens, num_kv_heads * head_dim)),
        q_gate.new_empty((n_tokens, num_q_heads * head_dim if store_gate else 0)),
    )


def _fused_qk_rmsnorm_rope_impl(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_scale: torch.Tensor | None,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | None = None,
    norm_beta: float = 0.0,
    store_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fused_qk_rmsnorm_rope(
        q_gate,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_scale,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        mrope_section,
        norm_beta,
        store_gate,
    )


# An opaque op, so the token-count launch heuristic runs at call time instead
# of being traced into (and guarded by) the compiled graph.
direct_register_custom_op(
    op_name="fused_qk_rmsnorm_rope",
    op_func=_fused_qk_rmsnorm_rope_impl,
    fake_impl=_fused_qk_rmsnorm_rope_fake,
)
