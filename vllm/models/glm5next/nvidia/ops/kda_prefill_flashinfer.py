# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer CuTe DSL KDA chunked prefill for GLM-5.3, in place on the pool.

``flashinfer.recurrent_kda(backend="cute-dsl")`` (FlashInfer >= 0.7.2, BT=16
"engine" kernel: chunk preparation and recurrence fused in one launch) reads
and writes the recurrent state pool ``[slot, H, V, K]`` (fp32 or bf16) through
``ssm_state_indices``, so the gather / scatter of the FlashKDA path and its
final-state buffer disappear. Its contract differs from vLLM's in two places
handled here:

- It has no per-sequence "has initial state" flag: slots of fresh sequences
  are zeroed first (one small launch).
- FlashInfer gates the backend to compute capability 10.0 / 10.3; the CuTe
  DSL kernel also builds and runs on SM107 (VR200), so the device capability
  is added to that Python allow-list.

With few sequences (``2 * num_seqs * H <= SM count``) FlashInfer picks a
two-kernel route that reads ``cu_seqlens`` on the host once per distinct
tensor object; callers pass the same int64 tensor to every layer of a step.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _zero_fresh_states_kernel(
    state_ptr,
    indices_ptr,
    has_initial_state_ptr,
    stride_state,
    stride_indices,
    stride_has_initial_state,
    row_size,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    if tl.load(has_initial_state_ptr + seq * stride_has_initial_state):
        return
    slot = tl.load(indices_ptr + seq * stride_indices).to(tl.int64)
    if slot <= 0:  # PAD_SLOT_ID (-1) or the null block (0)
        return
    offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros([BLOCK], dtype=state_ptr.dtype.element_ty)
    tl.store(state_ptr + slot * stride_state + offs, zeros, mask=offs < row_size)


def zero_fresh_states(
    state: torch.Tensor, indices: torch.Tensor, has_initial_state: torch.Tensor
) -> None:
    """``state[indices[i]] = 0`` for every ``i`` without an initial state."""
    if indices.numel() == 0:
        return
    row_size = state[0].numel()
    assert state[0].is_contiguous()
    block = 4096
    grid = (indices.numel(), triton.cdiv(row_size, block))
    _zero_fresh_states_kernel[grid](
        state,
        indices,
        has_initial_state,
        state.stride(0),
        indices.stride(0),
        has_initial_state.stride(0),
        row_size,
        BLOCK=block,
        num_warps=4,
    )


class FlashInferKdaPrefill:
    """Bounded-gate KDA prefill via FlashInfer's CuTe DSL BT16 kernel."""

    # Shared by all layers: one int64 cu_seqlens object per step, so
    # FlashInfer's host-side cu_seqlens memo hits after the first layer.
    _cu_source: torch.Tensor | None = None
    _cu_rows: int = -1
    _cu64: torch.Tensor | None = None

    def __init__(self, num_heads: int, head_dim: int, lower_bound: float) -> None:
        import flashinfer
        from flashinfer import kda_prefill_cute

        if head_dim != 128 or not (num_heads % 8 == 0 or num_heads <= 14):
            raise RuntimeError(
                "FlashInfer KDA prefill needs head_dim 128 and a local head "
                "count that is a multiple of 8 (or <= 14)."
            )
        capability = torch.cuda.get_device_capability()
        supported = kda_prefill_cute._SUPPORTED_COMPUTE_CAPABILITIES
        if capability not in supported:
            if capability[0] != 10:
                raise RuntimeError(
                    f"FlashInfer KDA prefill: unsupported compute capability "
                    f"{capability} (SM10x only)."
                )
            kda_prefill_cute._SUPPORTED_COMPUTE_CAPABILITIES = set(supported) | {
                capability
            }
            logger.info_once(
                "FlashInfer KDA prefill: allowing compute capability %s "
                "(CuTe DSL BT16 kernel).",
                capability,
            )
        self._recurrent_kda = flashinfer.recurrent_kda
        self._lower_bound = lower_bound
        self._scale = head_dim**-0.5

    @classmethod
    def _int64_cu_seqlens(cls, cu_seqlens: torch.Tensor, rows: int) -> torch.Tensor:
        # The builder's cu_seqlens object is shared by all layers of a step.
        if cls._cu_source is not cu_seqlens or cls._cu_rows != rows:
            cls._cu_source = cu_seqlens
            cls._cu_rows = rows
            cls._cu64 = cu_seqlens[: rows + 1].to(torch.int64)
        assert cls._cu64 is not None
        return cls._cu64

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_seqs: int,
        out: torch.Tensor,
    ) -> None:
        """q/k/v/g ``[1, T, H, 128]`` bf16 dense, raw gate ``g``, raw beta
        logits ``[1, T, H]``; ``state`` updated in place at ``state_indices``
        for the first ``num_seqs`` sequences of ``cu_seqlens``."""
        state_indices = state_indices[:num_seqs]
        zero_fresh_states(state, state_indices, has_initial_state[:num_seqs])
        self._recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta.contiguous(),
            A_log=A_log,
            dt_bias=dt_bias,
            scale=self._scale,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=self._lower_bound,
            cu_seqlens=self._int64_cu_seqlens(cu_seqlens, num_seqs),
            ssm_state_indices=state_indices.to(torch.int32).contiguous(),
            output=out,
            beta_is_logit=True,
            backend="cute-dsl",
        )
