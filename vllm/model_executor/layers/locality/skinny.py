# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Domain-aware skinny GEMM over a weight placed in localized memory.

``DomainGemm`` owns ``out[M, N] = x[M, K] @ W[N, K]^T`` for a BF16 weight
(the MXFP8 draft head uses ``mxgemm.DomainMxGemm``). One persistent CTA per SM
reads ``%smid``, looks up its domain in the topology table and streams only the
16-row groups whose memory is on its own domain (work stealing from the other
domain's list once its own is empty). See ``_ext`` for the kernel. One launch,
no green contexts, fixed device pointers: CUDA-graph capturable (the work
counters are reset by the last warp of each launch). Each instance has its own
counters, so two instances may run concurrently on different streams; one
instance must not.
"""

from __future__ import annotations

import torch

from . import _ext
from .memory import Localized, row_group_tiles
from .topology import Topology

BF16_MAX_M = 32
# stage configs (see _ext.lm_bf16):
#   100 * (16-B loads per row per stage) + stages in flight;
#   + 1000: cooperative kernel (one 16-row group per CTA, K split over the warps);
#   3000 + PD: cp.async-staged kernel
BF16_CFGS = (402, 404, 204, 208, 108, 116, 1402, 1404, 1204, 1208, 3002, 3003, 3103)


class DomainGemm:
    def __init__(
        self,
        topo: Topology,
        weight: Localized | torch.Tensor,
        cfg: int | None = None,
    ) -> None:
        w = weight.tensor if isinstance(weight, Localized) else weight
        assert w.dim() == 2 and w.is_contiguous()
        self.n, self.k = w.shape
        self.w = w
        self.topo = topo
        ords = weight.ordinals if isinstance(weight, Localized) else None
        cb = weight.chunk_bytes if isinstance(weight, Localized) else 1 << 21
        self._row_bytes, self._ords, self._cb = self.k * w.element_size(), ords, cb
        self._tiles: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.tiles0, self.tiles1 = self.tiles(1)
        self.ctrs = torch.zeros(4, dtype=torch.int32, device=w.device)
        self.max_m = BF16_MAX_M
        self.cfg = cfg
        self._keep = weight  # keeps the localized mapping alive

    def tiles(self, g: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-domain work lists of 16 * g-row units."""
        if g not in self._tiles:
            self._tiles[g] = row_group_tiles(
                self.w, self.n, self._row_bytes, self._ords, self._cb, 16 * g
            )
        return self._tiles[g]

    def default_cfg(self, m: int) -> int:
        if self.cfg is not None:
            return self.cfg
        # measured best on VR200 (bench_lm_head.py): k_lm3 (cp.async-staged, 8 warps)
        return 3002 if m <= 16 else 1402

    def __call__(
        self, x: torch.Tensor, out: torch.Tensor | None = None, cfg: int | None = None
    ) -> torch.Tensor:
        m = x.shape[0]
        assert 1 <= m <= self.max_m
        if out is None:
            out = torch.empty((m, self.n), dtype=torch.bfloat16, device=x.device)
        cfg = cfg or self.default_cfg(m)
        ext = _ext.load()
        t0, t1 = self.tiles(units_of(cfg))
        ext.lm_bf16(x, self.w, out, self.topo.sm_domain, t0, t1, self.ctrs, 8, cfg)
        return out


def units_of(cfg: int) -> int:
    """16-row groups per work unit of a stage config (k_lm3: tens digit, 0 = 1)."""
    return max(1, (cfg // 10) % 10) if cfg >= 3000 else 1


def cfgs_for(m: int) -> list[int]:
    """Stage configs whose kernel supports M = m rows."""

    def ok(c: int) -> bool:
        return m <= (16 if c >= 3000 else BF16_MAX_M)

    return [c for c in BF16_CFGS if ok(c)]
