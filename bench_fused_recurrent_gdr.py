#!/usr/bin/env python3
"""
Benchmark script for fused_recurrent_gated_delta_rule_fwd (Triton kernel only)

This script benchmarks the fused recurrent gated delta rule forward pass
using actual tensor shapes collected from Qwen3-Next-80B model inference.

Tests both TP=2 and TP=4 configurations with multiple batch sizes.

Benchmarking methodology:
- Uses CUPTI for accurate GPU kernel timing
- Cold L2 cache measurements
- Multiple runs for statistical stability
"""

import sys
import os
sys.path.insert(0, '/home/scratch.vgimpelson_ent/flashinfer')

import torch
import numpy as np
from flashinfer.testing import bench_gpu_time_with_cupti

# Import the Triton function to benchmark
from vllm.model_executor.layers.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule_fwd

# Batch sizes to benchmark
BATCH_SIZES = [32, 128, 256, 512, 1024]


def create_inputs(tp_size=4, batch_size=1024, num_speculative_tokens=0):
    """Create synthetic input tensors matching collected shapes for given TP size.
    
    Args:
        tp_size: Tensor parallel size (2 or 4)
        batch_size: Number of sequences in the batch
        num_speculative_tokens: Number of tokens per sequence (1 for decode, >1 for MTP)
        
    Returns:
        Dictionary of input tensors and parameters
    """
    device = torch.device("cuda:0")
    
    # Model config (total)
    total_num_key_heads = 16
    total_num_value_heads = 32
    key_head_dim = 128
    value_head_dim = 128
    
    # Per-GPU dimensions based on TP size
    H = total_num_key_heads // tp_size  # num key heads per GPU
    HV = total_num_value_heads // tp_size  # num value heads per GPU
    K = key_head_dim
    V = value_head_dim
    
    # Total tokens = batch_size * (1 + num_speculative_tokens)
    # num_speculative_tokens=0 means 1 token per seq (pure decode)
    # num_speculative_tokens=N means 1 base + N speculative = N+1 tokens per seq
    tokens_per_seq = 1 + num_speculative_tokens
    total_tokens = batch_size * tokens_per_seq
    
    # Main tensors - total_tokens is the token dimension
    q = torch.randn(1, total_tokens, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, total_tokens, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, total_tokens, HV, V, dtype=torch.bfloat16, device=device)
    g = torch.randn(1, total_tokens, HV, dtype=torch.float32, device=device)
    beta = torch.randn(1, total_tokens, HV, dtype=torch.bfloat16, device=device)
    
    # Scale
    scale = 0.08838834764831845
    
    # Initial state - needs to be large enough for all state slots
    # For speculative decoding: batch_size * tokens_per_seq state slots
    # Use a fixed large size for benchmarking (simulating KV cache state pool)
    max_state_slots = max(41069, batch_size * tokens_per_seq)
    initial_state = torch.randn(max_state_slots, HV, K, V, dtype=torch.bfloat16, device=device)
    
    # cu_seqlens: cumulative sequence lengths
    # For N=batch_size sequences, each with tokens_per_seq tokens:
    # [0, tokens_per_seq, 2*tokens_per_seq, ..., batch_size*tokens_per_seq]
    cu_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * tokens_per_seq
    
    # ssm_state_indices: for speculative decoding, this is 2D (batch_size, tokens_per_seq)
    # Each row contains state indices for each token in the sequence
    if num_speculative_tokens > 0:
        # 2D tensor: shape (batch_size, tokens_per_seq)
        # Each sequence has tokens_per_seq state slots
        # State index = seq_idx * tokens_per_seq + token_idx
        ssm_state_indices = torch.arange(0, batch_size * tokens_per_seq, dtype=torch.int32, device=device)
        ssm_state_indices = ssm_state_indices.reshape(batch_size, tokens_per_seq)
        # num_accepted_tokens = tokens_per_seq for each sequence (all accepted)
        num_accepted_tokens = torch.full((batch_size,), tokens_per_seq, dtype=torch.int32, device=device)
    else:
        # 1D tensor for regular decode (num_speculative_tokens=0 means 1 token per seq)
        ssm_state_indices = torch.arange(0, batch_size, dtype=torch.int32, device=device)
        num_accepted_tokens = None
    
    return {
        'q': q,
        'k': k,
        'v': v,
        'g': g,
        'beta': beta,
        'scale': scale,
        'initial_state': initial_state,
        'inplace_final_state': True,
        'cu_seqlens': cu_seqlens,
        'ssm_state_indices': ssm_state_indices,
        'num_accepted_tokens': num_accepted_tokens,
        'use_qk_l2norm_in_kernel': True,
    }


def benchmark_kernel(tp_size=4, batch_size=1024, num_speculative_tokens=0, verbose=True):
    """Benchmark the fused_recurrent_gated_delta_rule_fwd Triton kernel.
    
    Args:
        tp_size: Tensor parallel size (2 or 4)
        batch_size: Number of sequences in the batch
        num_speculative_tokens: Number of tokens per sequence (1 for decode, >1 for MTP)
        verbose: Print detailed output
        
    Returns:
        Dictionary with benchmark statistics
    """
    if verbose:
        print(f"  Benchmarking [Triton] TP={tp_size}, batch_size={batch_size}, spec_tokens={num_speculative_tokens}...")
    
    # Create inputs
    inputs = create_inputs(tp_size=tp_size, batch_size=batch_size, num_speculative_tokens=num_speculative_tokens)
    
    # Warmup Triton kernel
    for _ in range(3):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(**inputs)
        torch.cuda.synchronize()
    
    # Wrapper function for CUPTI benchmarking
    def run_kernel(q, k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices):
        return fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=inputs['scale'],
            initial_state=initial_state,
            inplace_final_state=inputs['inplace_final_state'],
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=inputs['num_accepted_tokens'],
            use_qk_l2norm_in_kernel=inputs['use_qk_l2norm_in_kernel'],
        )
    
    # Run CUPTI benchmark with all arguments properly specified
    times_ms = bench_gpu_time_with_cupti(
        fn=run_kernel,
        input_args=(
            inputs['q'],
            inputs['k'],
            inputs['v'],
            inputs['g'],
            inputs['beta'],
            inputs['initial_state'],
            inputs['cu_seqlens'],
            inputs['ssm_state_indices'],
        ),
        dry_run_time_ms=25,
        repeat_time_ms=100,
        cold_l2_cache=True,  # MANDATORY: cold L2 measurements
        use_cuda_graph=False,  # Launch overhead not a bottleneck for this kernel
        sleep_after_run=False,
    )
    
    # Convert to microseconds
    times_us = np.array(times_ms) * 1000.0
    
    median_us = np.median(times_us)
    mean_us = np.mean(times_us)
    std_us = np.std(times_us)
    std_err_us = std_us / np.sqrt(len(times_us))
    p10_us = np.percentile(times_us, 10)
    p90_us = np.percentile(times_us, 90)
    
    return {
        'median_us': median_us,
        'mean_us': mean_us,
        'std_us': std_us,
        'std_err_us': std_err_us,
        'p10_us': p10_us,
        'p90_us': p90_us,
        'n_runs': len(times_us),
    }


if __name__ == "__main__":
    # Set device
    torch.cuda.set_device(0)
    
    # Configuration
    tp_sizes = [4, 2]
    batch_sizes = BATCH_SIZES  # [32, 128, 256, 512, 1024]
    spec_token_counts = [0, 1, 2, 3]  # Test with 0, 1, 2, 3 speculative tokens
    
    # Results storage: results[spec_tokens][tp_size][batch_size] = stats
    results = {
        st: {tp: {} for tp in tp_sizes}
        for st in spec_token_counts
    }
    
    print("\n" + "=" * 100)
    print("BENCHMARKING fused_recurrent_gated_delta_rule_fwd (Triton Kernel)")
    print("=" * 100)
    print(f"TP configurations: {tp_sizes}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Speculative token counts: {spec_token_counts}")
    print("  (spec_tokens=0 → 1 tok/seq, spec_tokens=1 → 2 tok/seq, etc.)")
    print("Using CUPTI with cold L2 cache")
    print("=" * 100)
    
    total_configs = len(spec_token_counts) * len(tp_sizes) * len(batch_sizes)
    current = 0
    
    for spec_tokens in spec_token_counts:
        tokens_per_seq = 1 + spec_tokens
        print(f"\n{'='*60}")
        print(f"SPEC_TOKENS={spec_tokens} (tokens_per_seq={tokens_per_seq})")
        print(f"{'='*60}")
        
        for tp in tp_sizes:
            print(f"\n  [TP={tp}]")
            for bs in batch_sizes:
                current += 1
                total_tokens = bs * tokens_per_seq
                print(f"    [{current}/{total_configs}] batch={bs}, tokens={total_tokens}...", end=" ", flush=True)
                stats = benchmark_kernel(tp_size=tp, batch_size=bs, num_speculative_tokens=spec_tokens, verbose=False)
                results[spec_tokens][tp][bs] = stats
                print(f"median={stats['median_us']:.2f} μs ± {stats['std_err_us']:.2f} μs")
    
    # Print results table
    print("\n" + "=" * 100)
    print("DETAILED STATISTICS (all times in μs)")
    print("=" * 100)
    print(f"{'spec_tok':>10} | {'TP':>4} | {'Batch':>6} | {'Median':>10} | {'Mean':>10} | {'Std Err':>10} | {'P10':>10} | {'P90':>10} | {'N':>4}")
    print("-" * 100)
    
    for st in spec_token_counts:
        for tp in tp_sizes:
            for bs in batch_sizes:
                stats = results[st][tp][bs]
                print(f"{st:>10} | {tp:>4} | {bs:>6} | {stats['median_us']:>10.2f} | {stats['mean_us']:>10.2f} | "
                      f"{stats['std_err_us']:>10.2f} | {stats['p10_us']:>10.2f} | {stats['p90_us']:>10.2f} | {stats['n_runs']:>4}")
    
    print("=" * 100)
    
    # Print scaling analysis
    print("\nSCALING ANALYSIS (ratio vs spec_tokens=0)")
    print("=" * 100)
    
    for tp in tp_sizes:
        print(f"\n--- TP={tp} ---")
        header = f"{'Batch':>8}"
        for st in spec_token_counts:
            tps = 1 + st
            header += f" | spec={st} ({tps}x tok)"
        print(header)
        print("-" * 90)
        
        for bs in batch_sizes:
            row = f"{bs:>8}"
            base_time = results[0][tp][bs]['median_us']
            for st in spec_token_counts:
                ratio = results[st][tp][bs]['median_us'] / base_time
                row += f" | {ratio:>13.2f}x"
            print(row)
    
    print("\n" + "=" * 100)
    print("All benchmarks completed successfully!")
    print("=" * 100)
