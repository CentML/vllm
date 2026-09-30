# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels and launch configs for the tail of the Qwen GDN core op.

- ``gdn_norm_launch_config``: launch config of the FLA gated RMSNorm for the
  ``[T, HV * V]`` mixed-batch launch.
"""

import torch


def gdn_norm_launch_config(num_rows: int, device: torch.device) -> tuple[int, int]:
    """(rows_per_block, num_warps) of the FLA gated RMSNorm for the mixed-batch
    ``[T, HV * V]`` launch with one group per head (grid ``cdiv(T, rows) x HV``).

    Warps split rows (each warp keeps whole 128-wide rows), so the per-row
    reduction, and the output, match the default launch whenever that uses
    at least 2 rows per block (T > 2 * SMs). Below that the default config is
    kept, as its 1-row tile reduces in a different order.
    """
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        calc_rows_per_block,
    )

    rows_per_block = calc_rows_per_block(num_rows, device)
    if rows_per_block < 2 or num_rows < 512:
        return rows_per_block, 1
    if num_rows < 2048:
        return 8, 2
    # VR sweep, T = 2144..8192: (64, 8) is fastest or tied (-35..-44% vs default).
    return 64, 8
