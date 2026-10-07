# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen-style shared-expert gate ``sigmoid(g) * out`` in fusable form.

Eager PyTorch evaluates the gate as two kernels and rounds both the sigmoid and
the product to the activation dtype. Inside an Inductor-fused kernel, bf16
intermediates stay in fp32 and ``.to(torch.bfloat16)`` round trips are dropped,
so the gate is written with an explicit bit-level round-to-nearest-even. The
fused result then equals the eager one bitwise.
"""

import torch


def round_to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round fp32 ``x`` to ``dtype`` precision, keeping it fp32."""
    if dtype == torch.bfloat16:
        # c10::BFloat16's round-to-nearest-even, on the fp32 bit pattern. The
        # carry can turn a NaN payload into an infinity or -0.0, so NaNs pass
        # through unrounded.
        bits = x.view(torch.int32)
        bits = bits + (0x7FFF + ((bits >> 16) & 1))
        rounded = (bits & -65536).view(torch.float32)
        return torch.where(x.isnan(), x, rounded)
    return x.to(dtype).float()


def apply_shared_expert_gate(
    gate_logits: torch.Tensor, shared_output: torch.Tensor
) -> torch.Tensor:
    """``sigmoid(gate_logits) * shared_output`` with the eager roundings."""
    dtype = shared_output.dtype
    gate = round_to_dtype(torch.sigmoid(gate_logits.float()), dtype)
    return round_to_dtype(gate * shared_output.float(), dtype).to(dtype)
