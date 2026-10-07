# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side trims of the eager Qwen GDN path (``VLLM_GDN_HOST_TRIM=1``).

Mixed prefill/decode steps run the GDN core of every layer eagerly between
piecewise CUDA graphs; at low concurrency those steps are host-bound (each
layer issues ~8 small kernels whose launch-to-launch Python is 20-60 us on
the serving host). With the flag the wrappers do less per call; the kernels,
their arguments and the launch order are unchanged, except that the zeroing
of fresh SSM pool rows is skipped when the metadata builder knows on the host
that no prefill row is fresh (an exact no-op):

- Triton launches of the path go through ``FastLaunch`` (same compiled
  kernel, without ``JITFunction.run``'s per-call bookkeeping), and empty
  Triton launch hook chains are replaced by None process-wide (the C launcher
  then skips the hook calls and ``launch_metadata`` returns None);
- per-layer views (2-D conv weight, conv state layout) are built once, per-step
  metadata slices once per metadata object, the PDL capability once per
  process, and no-op ``.to()`` / ``.contiguous()`` calls are skipped.
"""

import os

import torch

from vllm.platforms import current_platform

GDN_HOST_TRIM = os.environ.get("VLLM_GDN_HOST_TRIM", "0") == "1"
# Second round of exact host trims for host-bound C32 mixed / prefill-only
# steps (same kernels and values; fewer launches and less per-step Python):
# - prefill-only batches hand the row-strided b/a views straight to the fused
#   conv-prep kernel (which reads them strided) instead of copying them
#   compact first (2 launches per GDN layer);
# - the GDN metadata builders compute the state-slot gather indices once per
#   step instead of once per KV-cache group (they depend on seq_lens only);
# - the causal_conv1d_fn metadata (H2D copies + fills) is built on first use,
#   i.e. never when every layer takes the fused conv-prep path;
# - the int64 copy of the prefill state indices (gather path only) is not
#   built when the chunk kernel updates the SSM pool in place.
GDN_HOST_TRIM2 = os.environ.get("VLLM_GDN_HOST_TRIM2", "0") == "1"
# Third round (exact; same kernels, arguments and values), for the GDN layers
# of steps with prefill rows:
# - BUFS: the fused conv prep writes q/k/v/exp(g)/beta into buffers allocated
#   once per (step, stream) and shared by all GDN layers of the step (every
#   KV-cache group), instead of 5 allocations per layer;
# - ZERO: the fresh SSM pool rows of all GDN layers sharing a metadata object
#   are zeroed by one launch at the first of them (before the mixed-batch
#   fork), instead of one launch per layer; the contiguous prefill state
#   indices are made once per metadata object;
# - HOIST_CONV: the spec rows' causal_conv1d_update launch of a mixed step is
#   bound once (compiled kernel and arguments) and relaunched for every layer
#   whose tensors have the same specialization, skipping the wrapper and the
#   Triton binder.
GDN_HOST_TRIM3 = os.environ.get("VLLM_GDN_HOST_TRIM3", "0") == "1"

_pdl: list[bool] = []


def step_conv_bufs(
    step,
    num_rows: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...] | None:
    """VLLM_GDN_HOST_TRIM3 (BUFS): the q, k [num_rows, H, K], v
    [num_rows, HV, V] (``dtype``) and exp(g), beta [num_rows, HV] (fp32)
    outputs of the prefill conv prep, allocated on the first call per (step,
    current stream) and returned again to every later GDN layer of the step
    on that stream. ``step`` is the step's forward context: the buffers live
    in it, so they are freed with it at the end of the step (one set per
    stream, the peak of the per-layer allocations, held across the step).
    Each layer consumes them (chunk kernel, checkpoint tails) on the stream
    that wrote them before the next layer overwrites them, so sharing equals
    the per-layer allocations. None while capturing a CUDA graph (the caller
    allocates).
    """
    index = device.index
    handle = torch._C._cuda_getCurrentRawStream(
        torch.accelerator.current_device_index() if index is None else index
    )
    cache = step.__dict__.get("_htrim3_conv_bufs")
    if cache is None:
        cache = step.__dict__["_htrim3_conv_bufs"] = {}
    key = (num_rows, num_k_heads, head_k_dim, num_v_heads, head_v_dim, dtype)
    entry = cache.get(handle)
    if entry is not None and entry[0] == key:
        return entry[1]
    if torch.cuda.is_current_stream_capturing():
        return None
    P, H, K, HV, V = key[:5]
    bufs = (
        torch.empty(P, H, K, dtype=dtype, device=device),
        torch.empty(P, H, K, dtype=dtype, device=device),
        torch.empty(P, HV, V, dtype=dtype, device=device),
        torch.empty(P, HV, dtype=torch.float32, device=device),
        torch.empty(P, HV, dtype=torch.float32, device=device),
    )
    cache[handle] = (key, bufs)
    return bufs


def arch_support_pdl() -> bool:
    """``current_platform.is_arch_support_pdl()``, read once per process (a
    worker drives one GPU).
    """
    if not _pdl:
        _pdl.append(current_platform.is_arch_support_pdl())
    return _pdl[0]


def launcher(fn):
    """``fn`` itself, or its ``FastLaunch`` with ``VLLM_GDN_HOST_TRIM=1``;
    either is launched as ``launcher(fn)[grid](*args, **kwargs)``.
    """
    if not GDN_HOST_TRIM:
        return fn
    from vllm.triton_utils.fast_launch import FastLaunch

    return FastLaunch(fn)


def drop_empty_triton_launch_hooks() -> None:
    """Replace empty Triton launch hook chains by None (see module doc)."""
    if not GDN_HOST_TRIM:
        return
    from vllm.triton_utils import triton

    runtime = triton.knobs.runtime
    for name in ("launch_enter_hook", "launch_exit_hook"):
        h = getattr(runtime, name)
        if isinstance(h, triton.knobs.HookChain) and not h.calls:
            setattr(runtime, name, None)
