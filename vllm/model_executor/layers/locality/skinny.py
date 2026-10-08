# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Domain-aware skinny GEMM over a weight placed in localized memory.

``DomainGemm`` owns ``out[M, N] = x[M, K] @ W[N, K]^T`` for a BF16 weight, or
an MXFP8 weight (e4m3 values + one E8M0 scale per 32 K of every row, scales
in plain ``[N, K/32]`` layout). One persistent CTA per SM reads ``%smid``,
looks up its domain in the topology table and streams only the 16-row groups
whose memory is on its own domain (work stealing from the other domain's list
once its own is empty). See ``_ext`` for the kernel. One launch, no green
contexts, fixed device pointers: CUDA-graph capturable (the work counters are
reset by the last warp of each launch). Each instance has its own counters, so
two instances may run concurrently on different streams; one instance must not.
"""

from __future__ import annotations

import torch

from . import _ext
from .memory import Localized, localize, row_group_tiles
from .topology import Topology

BF16_MAX_M = 32
FP8_MAX_M = 64
# stage configs (see _ext.lm_bf16 / lm_fp8): 100 * (16-B loads | 32-K blocks per row per stage) + stages in flight
BF16_CFGS = (402, 404, 204, 208, 108, 116, 1402, 1404, 1204, 1208, 3002, 3003, 3103)
FP8_CFGS = (802, 804, 404, 408, 1802, 1804, 1404, 1408, 2003, 2004, 2005, 3004, 3006, 3104, 3106, 3023, 3123)
# + 1000: cooperative kernel (one 16-row group per CTA, K split over the warps); 2000 + PD: fp8 CUDA-core kernel (M <= 4)


class DomainGemm:
    def __init__(
        self,
        topo: Topology,
        weight: Localized | torch.Tensor,
        scale: torch.Tensor | None = None,
        cfg: int | None = None,
    ) -> None:
        w = weight.tensor if isinstance(weight, Localized) else weight
        self.fp8 = w.element_size() == 1
        assert w.dim() == 2 and w.is_contiguous()
        self.n, self.k = w.shape
        self.w = w.view(torch.uint8) if self.fp8 else w
        self.scales: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.fp8:
            # E8M0 scales [N, K/32] (3% of the bytes): one copy per domain, so every
            # CTA reads its own die's copy whatever the row placement.
            assert scale is not None
            s = scale.contiguous().view(torch.uint8).reshape(self.n, self.k // 32)
            if isinstance(weight, Localized):
                self.scales = (localize(s, "dom0").tensor, localize(s, "dom1").tensor)
            else:
                self.scales = (s, s)
        self.topo = topo
        ords = weight.ordinals if isinstance(weight, Localized) else None
        cb = weight.chunk_bytes if isinstance(weight, Localized) else 1 << 21
        self._row_bytes, self._ords, self._cb = self.k * w.element_size(), ords, cb
        self._tiles: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.tiles0, self.tiles1 = self.tiles(1)
        self.ctrs = torch.zeros(4, dtype=torch.int32, device=w.device)
        self.max_m = FP8_MAX_M if self.fp8 else BF16_MAX_M
        self.cfg = cfg
        self._keep = weight  # keeps the localized mapping alive

    def tiles(self, g: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-domain work lists of 16 * g-row units."""
        if g not in self._tiles:
            self._tiles[g] = row_group_tiles(self.w, self.n, self._row_bytes, self._ords, self._cb, 16 * g)
        return self._tiles[g]

    def default_cfg(self, m: int) -> int:
        if self.cfg is not None:
            return self.cfg
        # measured best on VR200 (bench_lm_head.py): k_lm3 (cp.async-staged, 8 warps)
        if self.fp8:
            return 3004 if m <= 2 else 804
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
        if self.fp8:
            assert self.scales is not None
            ext.lm_fp8(x, self.w, self.scales[0], self.scales[1], out, self.topo.sm_domain,
                       t0, t1, self.ctrs, 8, cfg)
        else:
            ext.lm_bf16(x, self.w, out, self.topo.sm_domain, t0, t1, self.ctrs, 8, cfg)
        return out


def units_of(cfg: int) -> int:
    """16-row groups per work unit of a stage config (k_lm3: tens digit, 0 = 1)."""
    return max(1, (cfg // 10) % 10) if cfg >= 3000 else 1


def cfgs_for(fp8: bool, m: int) -> list[int]:
    """Stage configs whose kernel supports M = m rows."""
    def ok(c: int) -> bool:
        if c >= 3000:
            return m <= (2 if fp8 else 16)
        if fp8 and c >= 2000:
            return m <= 4
        return m <= (FP8_MAX_M if fp8 else BF16_MAX_M)

    return [c for c in (FP8_CFGS if fp8 else BF16_CFGS) if ok(c)]
