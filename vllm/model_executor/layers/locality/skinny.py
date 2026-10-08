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
BF16_CFGS = (402, 404, 204, 208, 108, 116)
FP8_CFGS = (802, 804, 404, 408)


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
        self.tiles0, self.tiles1 = row_group_tiles(w, self.n, self.k * w.element_size(), ords, cb)
        self.ctrs = torch.zeros(4, dtype=torch.int32, device=w.device)
        self.max_m = FP8_MAX_M if self.fp8 else BF16_MAX_M
        self.cfg = cfg
        self._keep = weight  # keeps the localized mapping alive

    def default_cfg(self, m: int) -> int:
        if self.cfg is not None:
            return self.cfg
        return 804 if self.fp8 else 404

    def __call__(
        self, x: torch.Tensor, out: torch.Tensor | None = None, cfg: int | None = None
    ) -> torch.Tensor:
        m = x.shape[0]
        assert 1 <= m <= self.max_m
        if out is None:
            out = torch.empty((m, self.n), dtype=torch.bfloat16, device=x.device)
        cfg = cfg or self.default_cfg(m)
        ext = _ext.load()
        if self.fp8:
            assert self.scales is not None
            ext.lm_fp8(x, self.w, self.scales[0], self.scales[1], out, self.topo.sm_domain,
                       self.tiles0, self.tiles1, self.ctrs, 8, cfg)
        else:
            ext.lm_bf16(x, self.w, out, self.topo.sm_domain, self.tiles0, self.tiles1,
                        self.ctrs, 8, cfg)
        return out
