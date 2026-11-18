# SPDX-License-Identifier: Apache-2.0
"""Kernel layouts of the NVFP4 MegaMoE weights, written shard by shard at load time.

The kernel takes, per local expert:

* FC1 (gate/up) data as row-major ``[2I, H/2]`` bytes (K-major packed E2M1) whose
  rows are 16-row stripes alternating gate (w1) and up (w3): row ``32p + 16h + r``
  holds row ``16p + r`` of half ``h`` (0 = w1, 1 = w3);
* FC2 (down) data as the checkpoint's row-major ``[H, I/2]``;
* block scales as the flat 32x4x4 atom swizzle of the plain ``[rows, cols]`` scale
  plane (rows in the same FC1 stripe order): element ``(n, c)`` sits at
  ``(((n // 128) * (cols // 4) + c // 4) * 32 + n % 32) * 16 + ((n % 128) // 32) * 4 + c % 4``.

Every checkpoint shard is copied straight into those layouts through strided views,
so loading needs no post-load transform and reloads keep the parameter addresses.
The shared expert uses the same layouts with MXFP8 data and UE8M0 1x32 scales.
"""

import torch

STRIPE = 16  # gate/up interleave of the FC1 N axis
SF_ROWS = 128  # scale atom rows (32 x 4)
SF_COLS = 4  # scale atom columns


def swizzled_scale_view(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """A ``[rows/128, 4, 32, cols/4, 4]`` view of a flat swizzled scale plane.

    Index ``[b, j, i, q, k]`` is plane element ``(128 b + 32 j + i, 4 q + k)``, so a
    plain ``[rows, cols]`` plane viewed as ``[rows/128, 4, 32, cols/4, 4]`` copies
    into it element for element.
    """
    if rows % SF_ROWS or cols % SF_COLS:
        raise ValueError(f"scale plane ({rows}, {cols}) is not a whole number of 128x4 atoms")
    if flat.numel() != rows * cols:
        raise ValueError(f"flat scale of {flat.numel()} elements is not ({rows}, {cols})")
    return flat.view(rows // SF_ROWS, cols // SF_COLS, 32, 4, 4).permute(0, 3, 2, 1, 4)


def write_plain(dst_flat: torch.Tensor, src: torch.Tensor) -> None:
    """Swizzle a whole plain ``[rows, cols]`` scale plane into ``dst_flat``."""
    rows, cols = src.shape
    view = swizzled_scale_view(dst_flat, rows, cols)
    view.copy_(src.reshape(rows // SF_ROWS, 4, 32, cols // SF_COLS, 4))


def write_gate_up_half(dst_flat: torch.Tensor, src: torch.Tensor, half: int) -> None:
    """Swizzle the plain ``[I, cols]`` scales of one FC1 half into the striped plane.

    Half ``h`` row ``g = 64 b + 16 j + r`` is striped row ``128 b + 32 j + 16 h + r``,
    i.e. atom index ``[b, j, 16 h + r]``.
    """
    rows, cols = src.shape
    if rows % (SF_ROWS // 2):
        raise ValueError(f"FC1 half has {rows} rows, not a multiple of 64")
    view = swizzled_scale_view(dst_flat, 2 * rows, cols)
    view[:, :, STRIPE * half : STRIPE * (half + 1)].copy_(
        src.reshape(rows // (SF_ROWS // 2), 4, STRIPE, cols // SF_COLS, 4)
    )


def write_gate_up_data(dst: torch.Tensor, src: torch.Tensor, half: int) -> None:
    """Copy one FC1 half ``[I, k]`` into its 16-row stripes of ``dst`` ``[2I, k]``."""
    rows, k = src.shape
    if rows % STRIPE or tuple(dst.shape) != (2 * rows, k):
        raise ValueError(f"FC1 half {tuple(src.shape)} does not fit {tuple(dst.shape)}")
    dst.view(rows // STRIPE, 2, STRIPE, k)[:, half].copy_(src.view(rows // STRIPE, STRIPE, k))


def expand_block_scales(scale: torch.Tensor, rows: int, cols: int, vec: int) -> torch.Tensor:
    """Expand ``[rows/br, cols/bc]`` block scales to one scale per ``1 x vec`` block.

    Exact for power-of-two (UE8M0) scales; ``bc`` must be a multiple of ``vec``.
    """
    sf = scale.view(torch.uint8)
    if sf.dim() != 2 or rows % sf.shape[0] or cols % sf.shape[1]:
        raise ValueError(f"scale {tuple(sf.shape)} does not tile ({rows}, {cols})")
    row_rep, col_block = rows // sf.shape[0], cols // sf.shape[1]
    if col_block % vec:
        raise ValueError(f"scale blocks of {col_block} columns are not whole {vec}-blocks")
    if row_rep > 1:
        sf = sf.repeat_interleave(row_rep, 0)
    if col_block // vec > 1:
        sf = sf.repeat_interleave(col_block // vec, 1)
    return sf.contiguous()


def gate_up_half(shard_id: str) -> int:
    if shard_id == "w1":
        return 0
    if shard_id == "w3":
        return 1
    raise ValueError(f"not an FC1 shard: {shard_id}")


# ----------------------------------------------------------------------------- reference (tests only)
def to_blocked_reference(plane: torch.Tensor) -> torch.Tensor:
    """Plain ``[rows, cols]`` -> flat 32x4x4 swizzle, written independently of the views above."""
    rows, cols = plane.shape
    blocks = plane.view(rows // 128, 128, cols // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)


def interleave_reference(w13: torch.Tensor, intermediate: int) -> torch.Tensor:
    """Canonical ``[w1; w3]`` rows -> 16-row stripes, written independently of the views above."""
    gate, up = w13[:intermediate], w13[intermediate:]
    pairs = intermediate // STRIPE
    return torch.stack(
        (gate.view(pairs, STRIPE, -1), up.view(pairs, STRIPE, -1)), dim=1
    ).reshape(2 * intermediate, -1)
