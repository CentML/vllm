# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark for MLA fused RoPE + KV cache kernel vs separate operations.

Uses flashinfer.testing.bench_gpu_time_with_cupti for accurate GPU timing.

Usage:
    python benchmarks/kernels/bench_mla_fused_rope.py
    python benchmarks/kernels/bench_mla_fused_rope.py --seq-lens 1,4,16
    python benchmarks/kernels/bench_mla_fused_rope.py --kv-cache-dtype fp8
"""

import argparse
import torch
import numpy as np

from flashinfer.testing import bench_gpu_time_with_cupti

from vllm import _custom_ops as ops
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.config import set_current_vllm_config, VllmConfig, CompilationConfig


def benchmark_single_config(
    seq_len: int,
    kv_cache_dtype: str,
    num_q_heads: int = 128,
    qk_rope_head_dim: int = 64,
    kv_lora_rank: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda:0",
    dry_run_time_ms: float = 10,
    repeat_time_ms: float = 50,
):
    """Benchmark a single configuration and return timing results."""
    rope = RotaryEmbedding(
        qk_rope_head_dim,
        qk_rope_head_dim,
        8192,
        10000,
        True,  # is_neox_style
        torch.float32,
    )
    rope = rope.to(dtype=dtype, device=device)

    positions = torch.randint(0, 8192, (seq_len,), device=device)
    q_pe = torch.randn(seq_len, num_q_heads, qk_rope_head_dim, dtype=dtype, device=device)
    k_pe = torch.randn(seq_len, qk_rope_head_dim, dtype=dtype, device=device)
    kv_c = torch.randn(seq_len, kv_lora_rank, dtype=dtype, device=device)

    slot_mapping = torch.arange(seq_len, dtype=torch.long, device=device)
    entry_size = kv_lora_rank + qk_rope_head_dim
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    kv_cache = torch.zeros(64, 64, entry_size, dtype=cache_dtype, device=device)
    kv_cache_scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    # Benchmark rotary_embedding kernel
    def run_rope():
        ops.rotary_embedding(
            positions, q_pe, k_pe, qk_rope_head_dim, rope.cos_sin_cache, True
        )

    times = bench_gpu_time_with_cupti(
        fn=run_rope,
        input_args=(),
        dry_run_time_ms=dry_run_time_ms,
        repeat_time_ms=repeat_time_ms,
        cold_l2_cache=True,
    )
    rope_time = np.median(times) * 1000  # ms to us

    # Benchmark concat_and_cache_mla
    def run_concat():
        ops.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping,
            kv_cache_dtype=kv_cache_dtype, scale=kv_cache_scale
        )

    times = bench_gpu_time_with_cupti(
        fn=run_concat,
        input_args=(),
        dry_run_time_ms=dry_run_time_ms,
        repeat_time_ms=repeat_time_ms,
        cold_l2_cache=True,
    )
    concat_time = np.median(times) * 1000

    # Benchmark fused kernel
    def run_fused():
        ops.concat_and_cache_mla_rope_fused(
            positions, q_pe, k_pe, kv_c, rope.cos_sin_cache, True,
            slot_mapping, kv_cache, kv_cache_dtype, kv_cache_scale
        )

    times = bench_gpu_time_with_cupti(
        fn=run_fused,
        input_args=(),
        dry_run_time_ms=dry_run_time_ms,
        repeat_time_ms=repeat_time_ms,
        cold_l2_cache=True,
    )
    fused_time = np.median(times) * 1000

    return rope_time, concat_time, fused_time


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MLA fused RoPE + KV cache kernel"
    )
    parser.add_argument(
        "--seq-lens", type=str, default="1,4,16,64,128,256",
        help="Comma-separated sequence lengths to benchmark"
    )
    parser.add_argument(
        "--kv-cache-dtype", type=str, default="both",
        choices=["auto", "fp8", "both"],
        help="KV cache dtype to benchmark"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    kv_dtypes = ["auto", "fp8"] if args.kv_cache_dtype == "both" else [args.kv_cache_dtype]

    vllm_config = VllmConfig(compilation_config=CompilationConfig())

    print("\n" + "=" * 100)
    print("MLA Fused RoPE + KV Cache Benchmark")
    print("=" * 100)
    print(f"{'seq_len':>8} {'kv_dtype':>8} | {'rope':>10} {'concat':>10} {'fused':>10} | {'separate':>10} {'speedup':>8}")
    print("-" * 100)

    with set_current_vllm_config(vllm_config):
        for seq_len in seq_lens:
            for kv_dtype in kv_dtypes:
                rope_time, concat_time, fused_time = benchmark_single_config(
                    seq_len=seq_len,
                    kv_cache_dtype=kv_dtype,
                    device=args.device,
                )
                separate_total = rope_time + concat_time
                speedup = separate_total / fused_time
                print(
                    f"{seq_len:>8} {kv_dtype:>8} | "
                    f"{rope_time:>9.2f}us {concat_time:>9.2f}us {fused_time:>9.2f}us | "
                    f"{separate_total:>9.2f}us {speedup:>7.2f}x"
                )

    print("=" * 100)
    print("Note: Speedup is from eliminating one kernel launch and reducing memory traffic")
    print()


if __name__ == "__main__":
    main()
