# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    get_fp8_min_max,
    group_broadcast,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_MIN, _FP8_MAX = get_fp8_min_max()
_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


@triton.jit
def _quantize_pad_fp8_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    stride_xs,  # input stride along token (seq) dim — may be non-contiguous
    stride_xh,  # input stride along head dim
    stride_xd,  # input stride along head_dim dim (usually 1)
    stride_ys,  # output stride along token dim (contiguous)
    stride_yh,  # output stride along head dim
    stride_yd,  # output stride along head_dim dim (usually 1)
    num_heads,
    n_rows,     # total rows = S * H
    n_cols,
    n_cols_padded,
    fp8_min,
    fp8_max,
    SKIP_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < n_rows
    mask_out = mask_m[:, None] & (offs_n[None, :] < n_cols_padded)
    mask_in = mask_m[:, None] & (offs_n[None, :] < n_cols)

    # Decompose flattened row into (token, head) for 3D stride indexing.
    # This lets the kernel read directly from non-contiguous QKV views.
    s = offs_m // num_heads
    h = offs_m % num_heads

    x_ptrs = (x_ptr
              + s[:, None] * stride_xs
              + h[:, None] * stride_xh
              + offs_n[None, :] * stride_xd)
    x = tl.load(x_ptrs, mask=mask_in, other=0.0).to(tl.float32)
    if SKIP_SCALE:
        x_q = x
    else:
        scale = tl.load(scale_ptr)
        x_q = x / scale
    x_q = tl.where(mask_in, x_q, 0.0)
    x_q = tl.clamp(x_q, fp8_min, fp8_max).to(y_ptr.dtype.element_ty)

    y_ptrs = (y_ptr
              + s[:, None] * stride_ys
              + h[:, None] * stride_yh
              + offs_n[None, :] * stride_yd)
    tl.store(y_ptrs, x_q, mask=mask_out)


def _get_fp8_pad_quant_config(padded_head_dim: int) -> tuple[int, int, int]:
    # Blackwell: use a single static config to avoid recompiles.
    if current_platform.is_device_capability_family(100):
        block_n, num_warps, block_m = 128, 4, 16
    else:
        block_n = triton.next_power_of_2(padded_head_dim)
        block_n = max(16, min(block_n, 256))
        num_warps = 4 if block_n >= 128 else 2
        block_m = 16

    env_block_n = os.getenv("VLLM_FP8_PAD_QUANT_BLOCK_N")
    env_num_warps = os.getenv("VLLM_FP8_PAD_QUANT_NUM_WARPS")
    env_block_m = os.getenv("VLLM_FP8_PAD_QUANT_BLOCK_M")
    if env_block_n is not None:
        block_n = max(16, min(int(env_block_n), 256))
    if env_num_warps is not None:
        num_warps = int(env_num_warps)
    if env_block_m is not None:
        block_m = max(1, int(env_block_m))

    return block_n, num_warps, block_m


def quantize_fp8_pad_head_dim_triton(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    skip_scale: bool = False,
    block_n: int | None = None,
    num_warps: int | None = None,
    block_m: int | None = None,
) -> torch.Tensor:
    """Quantize a 4D (B, S, H, D) or 3D (S, H, D) tensor to FP8 while padding D to a multiple of 16.

    Reads directly from the input using its 3D strides, so non-contiguous
    views (e.g. Q/K/V slices from an interleaved QKV buffer) are handled
    without an extra copy.  Output is always a fresh contiguous tensor
    with shape (S, H, padded_D).
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is required to quantize with head_dim padding."
        )

    original_shape = tensor.shape
    if tensor.dim() == 4:
        tensor = tensor.view(-1, tensor.shape[-2], tensor.shape[-1])
    assert tensor.dim() == 3, (
        f"Expected 3D input (S, H, D), got {tensor.dim()}D"
    )
    S, H, D = tensor.shape
    padded_head_dim = (D + 15) // 16 * 16
    out_dtype = current_platform.fp8_dtype()
    output = torch.empty(
        (S, H, padded_head_dim),
        device=tensor.device,
        dtype=out_dtype,
    )

    scale_1d = scale.reshape(-1)
    fp8_min, fp8_max = get_fp8_min_max()
    n_rows = S * H

    if block_n is None or num_warps is None or block_m is None:
        block_n, num_warps, block_m = _get_fp8_pad_quant_config(padded_head_dim)

    grid = (triton.cdiv(n_rows, block_m),
            triton.cdiv(padded_head_dim, block_n))

    _quantize_pad_fp8_kernel[grid](
        tensor,
        output,
        scale_1d,
        tensor.stride(0), tensor.stride(1), tensor.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        H,
        n_rows,
        D,
        padded_head_dim,
        fp8_min,
        fp8_max,
        SKIP_SCALE=skip_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )

    return output.view((*original_shape[:-1], padded_head_dim))


# --8<-- [start:quant_fp8]
@CustomOp.register("quant_fp8")
class QuantFP8(CustomOp):
    """
    Quantize input tensor to FP8 (per-tensor, per-token, per-channel, or per-group).
    This CustomOp supports both static and dynamic quantization.
    """

    # --8<-- [end:quant_fp8]

    def __init__(
        self,
        static: bool,
        group_shape: GroupShape,
        num_token_padding: int | None = None,
        column_major_scales: bool = False,
        use_ue8m0: bool | None = None,  # for Torch compile
    ):
        """
        :param static: static or dynamic quantization
        :param group_shape: quantization group shape (PER_TOKEN, PER_TENSOR,
            or arbitrary block size)
        :param num_token_padding: Pad the token dimension of output to this
            size
        :param column_major_scales: For group quantization, output scales in
            column major format
        """
        super().__init__()
        self.static = static
        self.group_shape = group_shape
        self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN
        self.num_token_padding = num_token_padding
        self.column_major_scales = column_major_scales
        self.use_ue8m0 = use_ue8m0

        self.use_aiter = rocm_aiter_ops.is_linear_fp8_enabled()

        self.is_group_quant = group_shape.is_per_group()
        if self.is_group_quant:
            self.group_size = group_shape.col
        else:
            self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN
            if not static:
                assert group_shape in (GroupShape.PER_TOKEN, GroupShape.PER_TENSOR), (
                    "Only per-token or per-tensor scales are supported for dynamic "
                    "non-group quantization."
                )

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_group_quant and not self.static:
            assert scale is None, "Dynamic group quantization does not use scale"
            from vllm.model_executor.layers.quantization.utils import fp8_utils

            return fp8_utils.per_token_group_quant_fp8(
                x,
                group_size=self.group_size,
                column_major_scales=self.column_major_scales,
                dtype=_FP8_DTYPE,
                use_ue8m0=self.use_ue8m0,
            )

        assert (scale is not None) == self.static
        assert scale_ub is None or (
            not self.static
            and self.group_shape == GroupShape.PER_TOKEN
            and scale_ub.numel() == 1
        )

        return ops.scaled_fp8_quant(
            x,
            scale,
            num_token_padding=self.num_token_padding,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            group_shape=(self.group_shape.row, self.group_shape.col)
            if self.static
            else None,
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_aiter_quant = (
            not self.is_group_quant
            and self.use_aiter
            and scale_ub is None
            and x.is_contiguous()
        )
        use_aiter_per_tensor_quant = (
            use_aiter_quant and self.group_shape == GroupShape.PER_TENSOR
        )
        use_aiter_per_token_quant = (
            use_aiter_quant and self.group_shape == GroupShape.PER_TOKEN
        )

        if use_aiter_per_tensor_quant:
            return rocm_aiter_ops.per_tensor_quant(x, _FP8_DTYPE, scale)
        if use_aiter_per_token_quant:
            return rocm_aiter_ops.per_token_quant(x, _FP8_DTYPE, scale)

        # Fallback to CUDA implementation
        return self.forward_cuda(x, scale, scale_ub)

    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
    ):
        if self.is_group_quant and not self.static:
            assert scale is None, "Dynamic group quantization does not use scale"
            return self._quantize_group_native(x)

        assert (scale is not None) == self.static
        assert scale_ub is None or (
            not self.static
            and self.group_shape == GroupShape.PER_TOKEN
            and scale_ub.numel() == 1
        )

        if scale is None:
            if self.group_shape == GroupShape.PER_TOKEN:
                x_max, _ = x.abs().max(dim=-1)
                x_max = x_max.unsqueeze(-1).to(torch.float32)
                if scale_ub is not None:
                    x_max = x_max.clamp(max=scale_ub)
            else:
                x_max = x.abs().max().unsqueeze(-1).to(torch.float32)

            scale = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)

        # Even for dynamic per-token scales,
        # reciprocal performs slightly better than division
        out = (
            x.to(torch.float32)
            * group_broadcast(scale.to(torch.float32), x.shape[-2:]).reciprocal()
        )
        out = out.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        # This currently generates an extra Triton kernel in compilation.
        # Fortunately, we don't use padding if compiling.
        # TODO(luka): benchmark torch._scaled_mm to hopefully remove padding
        #  in general.
        if self.num_token_padding is not None:
            padding = max(self.num_token_padding - out.size(0), 0)
            out = F.pad(out, (0, 0, 0, padding), "constant", 0.0)

        return out, scale

    def _quantize_group_native(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        num_groups = (hidden_dim + self.group_size - 1) // self.group_size
        padded_dim = num_groups * self.group_size

        if padded_dim != hidden_dim:
            padding = padded_dim - hidden_dim
            x = F.pad(x, (0, padding), mode="constant", value=0.0)

        x_grouped = x.view(-1, num_groups, self.group_size)
        absmax = x_grouped.abs().max(dim=-1, keepdim=True)[0].float()
        scales_raw = absmax / _FP8_MAX
        if self.use_ue8m0:
            scales_raw = torch.exp2(torch.ceil(torch.log2(scales_raw)))
        scales = (scales_raw).clamp(min=_FP8_MIN_SCALING_FACTOR)

        x_scaled = x_grouped / scales
        x_quant = x_scaled.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        x_quant = x_quant.view(-1, padded_dim)
        if padded_dim != hidden_dim:
            x_quant = x_quant[..., :hidden_dim]
        x_quant = x_quant.view(orig_shape)

        scales = scales.squeeze(-1)
        scales = scales.reshape(orig_shape[:-1] + (num_groups,))

        if self.column_major_scales:
            scales = scales.transpose(-2, -1).contiguous().transpose(-1, -2)

        return x_quant, scales
