#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark for FlashInfer rope_quantize_fp8 Q-only fusion vs existing path.

This benchmark compares:
1. Existing path: _DecodeConcatQuantFP8 (concat + quant, RoPE already applied)
2. FlashInfer fused: rope_quantize_fp8 (RoPE + quant in single kernel)

Uses flashinfer.testing.bench_gpu_time_with_cupti with cold L2 cache as required.
"""

import argparse
import statistics

import torch

from vllm.platforms import current_platform

# Check dependencies
try:
    from flashinfer.rope import rope_quantize_fp8
    from flashinfer.testing import bench_gpu_time_with_cupti

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
    print("WARNING: FlashInfer not available. Some benchmarks will be skipped.")


def compute_cos_sin_cache(
    rotary_dim: int,
    max_position_embeddings: int,
    base: float = 10000.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute cos/sin cache for RoPE embedding (float32 required by FlashInfer)."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache


def benchmark_existing_path(
    batch_size: int,
    num_q_heads: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    dtype: torch.dtype,
    device: str,
    repeat_time_ms: int = 100,
) -> dict:
    """Benchmark the existing concat + quant path."""
    fp8_dtype = current_platform.fp8_dtype()

    # Create inputs (after BMM, RoPE already applied)
    mqa_ql_nope = torch.randn(
        batch_size, num_q_heads, kv_lora_rank, dtype=dtype, device=device
    )
    mqa_q_pe = torch.randn(
        batch_size, num_q_heads, qk_rope_head_dim, dtype=dtype, device=device
    )
    q_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Output tensor
    output = torch.empty(
        batch_size,
        num_q_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=fp8_dtype,
        device=device,
    )

    def run():
        # Simulate the existing path: concat then quantize
        concat = torch.cat([mqa_ql_nope, mqa_q_pe], dim=-1)
        # Simple FP8 quantization
        scaled = concat * q_scale
        fp8_max = torch.finfo(fp8_dtype).max
        clamped = torch.clamp(scaled, -fp8_max, fp8_max)
        output.copy_(clamped.to(fp8_dtype))
        return output

    times = bench_gpu_time_with_cupti(
        fn=run,
        input_args=(),
        input_kwargs={},
        dry_run_time_ms=25,
        repeat_time_ms=repeat_time_ms,
        cold_l2_cache=True,
        use_cuda_graph=False,
        sleep_after_run=False,
    )

    return {
        "median_us": statistics.median(times) * 1e6,
        "std_error_us": statistics.stdev(times) * 1e6 / (len(times) ** 0.5)
        if len(times) > 1
        else 0,
        "n_samples": len(times),
    }


def benchmark_flashinfer_fused(
    batch_size: int,
    num_q_heads: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    dtype: torch.dtype,
    device: str,
    max_position: int = 8192,
    repeat_time_ms: int = 100,
) -> dict:
    """Benchmark the FlashInfer fused RoPE + quant path."""
    if not HAS_FLASHINFER:
        return {"error": "FlashInfer not available"}

    fp8_dtype = current_platform.fp8_dtype()

    # Create cos_sin_cache (float32 required)
    cos_sin_cache = compute_cos_sin_cache(qk_rope_head_dim, max_position, 10000.0, device)

    # Create inputs (before RoPE)
    q_rope_unrotated = torch.randn(
        batch_size, num_q_heads, qk_rope_head_dim, dtype=dtype, device=device
    )
    q_nope = torch.randn(
        batch_size, num_q_heads, kv_lora_rank, dtype=dtype, device=device
    )

    # Dummy K tensors (Q-only fusion strategy)
    dummy_k_rope = torch.zeros(batch_size, qk_rope_head_dim, dtype=dtype, device=device)
    dummy_k_nope = torch.zeros(batch_size, kv_lora_rank, dtype=dtype, device=device)

    # Random positions
    positions = torch.randint(0, max_position, (batch_size,), device=device)

    def run():
        q_rope_fp8, _, q_nope_fp8, _ = rope_quantize_fp8(
            q_rope=q_rope_unrotated,
            k_rope=dummy_k_rope,
            q_nope=q_nope,
            k_nope=dummy_k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            is_neox=True,
            quantize_dtype=fp8_dtype,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )
        return torch.cat([q_nope_fp8, q_rope_fp8], dim=-1)

    times = bench_gpu_time_with_cupti(
        fn=run,
        input_args=(),
        input_kwargs={},
        dry_run_time_ms=25,
        repeat_time_ms=repeat_time_ms,
        cold_l2_cache=True,
        use_cuda_graph=False,
        sleep_after_run=False,
    )

    return {
        "median_us": statistics.median(times) * 1e6,
        "std_error_us": statistics.stdev(times) * 1e6 / (len(times) ** 0.5)
        if len(times) > 1
        else 0,
        "n_samples": len(times),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark rope_quantize_fp8 Q-only fusion"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 32, 128, 512, 1024, 2048],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--num-q-heads", type=int, default=128, help="Number of Q heads (DeepSeek-R1)"
    )
    parser.add_argument(
        "--qk-rope-head-dim", type=int, default=64, help="RoPE head dimension"
    )
    parser.add_argument(
        "--kv-lora-rank", type=int, default=512, help="KV LoRA rank (DeepSeek-R1)"
    )
    parser.add_argument(
        "--repeat-time-ms", type=int, default=100, help="Repeat time for benchmarking"
    )
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    print("=" * 80)
    print("Benchmark: rope_quantize_fp8 Q-only fusion vs existing path")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"num_q_heads: {args.num_q_heads}")
    print(f"qk_rope_head_dim: {args.qk_rope_head_dim}")
    print(f"kv_lora_rank: {args.kv_lora_rank}")
    print(f"dtype: {dtype}")
    print(f"repeat_time_ms: {args.repeat_time_ms}")
    print("Using CUPTI timing with cold L2 cache")
    print("=" * 80)
    print()

    print(
        f"{'Batch':<10} {'Existing (us)':<20} {'FlashInfer (us)':<20} {'Speedup':<10}"
    )
    print("-" * 60)

    for batch_size in args.batch_sizes:
        existing_result = benchmark_existing_path(
            batch_size=batch_size,
            num_q_heads=args.num_q_heads,
            qk_rope_head_dim=args.qk_rope_head_dim,
            kv_lora_rank=args.kv_lora_rank,
            dtype=dtype,
            device=device,
            repeat_time_ms=args.repeat_time_ms,
        )

        flashinfer_result = benchmark_flashinfer_fused(
            batch_size=batch_size,
            num_q_heads=args.num_q_heads,
            qk_rope_head_dim=args.qk_rope_head_dim,
            kv_lora_rank=args.kv_lora_rank,
            dtype=dtype,
            device=device,
            repeat_time_ms=args.repeat_time_ms,
        )

        existing_time = existing_result["median_us"]
        existing_err = existing_result["std_error_us"]

        if "error" in flashinfer_result:
            print(
                f"{batch_size:<10} {existing_time:>8.2f} ± {existing_err:>5.2f}    "
                f"{'N/A':<20} {'N/A':<10}"
            )
        else:
            flashinfer_time = flashinfer_result["median_us"]
            flashinfer_err = flashinfer_result["std_error_us"]
            speedup = existing_time / flashinfer_time if flashinfer_time > 0 else 0

            print(
                f"{batch_size:<10} {existing_time:>8.2f} ± {existing_err:>5.2f}    "
                f"{flashinfer_time:>8.2f} ± {flashinfer_err:>5.2f}    "
                f"{speedup:>6.2f}x"
            )

    print()
    print("Note: The existing path simulates concat+quant (RoPE already applied).")
    print("      FlashInfer fused path includes RoPE+quant in single kernel.")
    print("      Speedup > 1.0 means FlashInfer is faster.")


if __name__ == "__main__":
    main()
