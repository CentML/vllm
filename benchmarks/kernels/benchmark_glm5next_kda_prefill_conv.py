# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA prefill short conv: causal_conv1d_fn vs the planar kernel.

Inputs follow the model: q|k|v is a row-strided slice of the merged in_proj
output (row = 3P + H + 2 * 128), decode rows come first, every sequence owns a
scattered slot of a bf16 conv-state pool. ``stock`` is causal_conv1d_fn with
precomputed metadata (as in serving), ``stock+copies`` adds the q/k/v
``.contiguous()`` copies that a dense-input consumer makes, ``planar`` is
kda_causal_conv1d_planar. Outputs and conv states are checked bit-identical
before timing; times are medians over CUDA events with L2 flushed.
"""

import argparse
import statistics
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.models.glm5next.nvidia.ops.kda_prefill_conv import (
    kda_causal_conv1d_planar,
)
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

HEAD_DIM = 128
WIDTH = 4


def parse_batch(spec: str) -> list[int]:
    """``"8x8192+194x1"`` -> lengths, decode rows (length 1) first."""
    lens: list[int] = []
    for term in spec.split("+"):
        count, length = (int(v) for v in term.split("x"))
        lens += [length] * count
    return sorted(lens, key=lambda n: n != 1)


def make_inputs(num_heads: int, lens: list[int], dim_first: bool, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dim = 3 * num_heads * HEAD_DIM
    num_tokens = sum(lens)
    row = dim + num_heads + 2 * HEAD_DIM
    proj = torch.randn(
        num_tokens, row, generator=gen, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(dim, WIDTH, generator=gen, device="cuda") * 0.5
    num_slots = len(lens) + 1
    shape = (num_slots, dim, WIDTH - 1) if dim_first else (num_slots, WIDTH - 1, dim)
    pool = torch.randn(shape, generator=gen, device="cuda", dtype=torch.bfloat16)
    qsl_cpu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()]).int()
    slots = torch.randperm(len(lens), generator=torch.Generator().manual_seed(seed))
    return SimpleNamespace(
        x=proj[:, :dim],
        weight=weight,
        pool=pool,
        dim_first=dim_first,
        query_start_loc=qsl_cpu.cuda(),
        query_start_loc_cpu=qsl_cpu,
        cache_indices=(slots + 1).int().cuda(),
        has_initial_state=torch.tensor([n == 1 for n in lens], device="cuda"),
    )


def conv_states(c, pool: torch.Tensor) -> torch.Tensor:
    return pool if c.dim_first else pool.transpose(-1, -2)


def run_stock(c, pool: torch.Tensor, metadata, copies: bool):
    out = causal_conv1d_fn(
        c.x.transpose(0, 1),
        c.weight,
        None,
        conv_states=conv_states(c, pool),
        query_start_loc=c.query_start_loc,
        cache_indices=c.cache_indices,
        has_initial_state=c.has_initial_state,
        activation="silu",
        metadata=metadata,
    ).transpose(0, 1)
    q, k, v = out.split(c.x.shape[1] // 3, dim=-1)
    if copies:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    return out


def run_planar(c, pool: torch.Tensor, **kwargs):
    return kda_causal_conv1d_planar(
        c.x,
        c.weight,
        None,
        conv_states(c, pool),
        query_start_loc=c.query_start_loc,
        cache_indices=c.cache_indices,
        has_initial_state=c.has_initial_state,
        **kwargs,
    )


def time_ms(fn, iters: int, warmup: int, flush: torch.Tensor) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def bench_case(
    num_heads: int, spec: str, args: argparse.Namespace, flush: torch.Tensor
) -> None:
    c = make_inputs(num_heads, parse_batch(spec), args.layout == "DS")
    nums_dict, batch_ptr, offset_ptr = compute_causal_conv1d_metadata(
        c.query_start_loc_cpu, device=c.x.device
    )
    metadata = SimpleNamespace(
        nums_dict=nums_dict, batch_ptr=batch_ptr, token_chunk_offset_ptr=offset_ptr
    )

    ref_pool, pool = c.pool.clone(), c.pool.clone()
    ref = run_stock(c, ref_pool, metadata, copies=False)
    for impl in args.impl:
        pool.copy_(c.pool)
        out = run_planar(c, pool, impl=impl)
        assert torch.equal(torch.cat(out.unbind(0), dim=1), ref), impl
        assert torch.equal(pool, ref_pool), impl

    # x read once, y written once.
    nbytes = 2 * c.x.numel() * c.x.element_size()
    paths = {
        "stock": lambda: run_stock(c, pool, metadata, copies=False),
        "stock+copies": lambda: run_stock(c, pool, metadata, copies=True),
    }
    for impl in args.impl:
        paths[f"planar-{impl}"] = lambda impl=impl: run_planar(c, pool, impl=impl)
    base = None
    for name, fn in paths.items():
        ms = time_ms(fn, args.iters, args.warmup, flush)
        base = base or ms
        print(
            f"{num_heads:>3} {spec:>14} {name:>14} {ms:>8.3f} "
            f"{nbytes / ms / 1e9:>6.2f} {base / ms:>7.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, nargs="+", default=[64, 32, 16])
    parser.add_argument(
        "--batches", nargs="+", default=["8x8192+194x1", "8x8192", "1x8192"]
    )
    parser.add_argument("--layout", choices=["DS", "SD"], default="DS")
    parser.add_argument(
        "--impl", nargs="+", choices=["tma", "regs"], default=["tma", "regs"]
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    print(f"{torch.cuda.get_device_name()}, conv-state layout {args.layout}")
    header = ["H", "batch", "path", "ms", "TB/s", "vs stock"]
    print("{:>3} {:>14} {:>14} {:>8} {:>6} {:>8}".format(*header))
    with torch.inference_mode():
        for num_heads in args.heads:
            for spec in args.batches:
                bench_case(num_heads, spec, args, flush)


if __name__ == "__main__":
    main()
