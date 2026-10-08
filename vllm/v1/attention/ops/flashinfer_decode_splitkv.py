# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in split-KV policy; the backend owns launch and counter lifetime.

The virtual-SM budget is bounded both per device and per batch/head. It does
not select a fixed split count at every batch size. Changing splits is
FLOAT-ORDER versus the stock policy, not a precision-format change.
"""


def policy_sm_count(
    real_sms: int,
    batch: int,
    num_kv_heads: int,
    scale: float = 0.0,
    ctas_per_sm: float = 16.0,
    max_splits: int = 16,
) -> int:
    """Return the virtual SM count without device access or allocation."""
    if scale and scale > 0:
        return int(real_sms * scale)
    want = min(
        int(ctas_per_sm * real_sms),
        max_splits * max(1, batch) * max(1, num_kv_heads),
    )
    return max(real_sms, want)


def check_workspace(
    workspace_bytes: int, sm_count: int, q_rows: int, head_size: int
) -> None:
    """Refuse unsafe scratch sizes rather than silently changing the split.

    Reserve at least the minimum query tile and cover larger grouped-query
    tiles. This is a host-only warmup/capture check.
    """
    tile_q = max(32, 1 << (max(1, q_rows) - 1).bit_length())
    needed = sm_count * tile_q * (8 + head_size * 4) + (1 << 20)
    if workspace_bytes < needed:
        raise RuntimeError(
            f"FI_DECODE_SPLITKV workspace {workspace_bytes} B < {needed} B; "
            "increase VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE or reduce the "
            "split-KV policy budget"
        )
