#!/usr/bin/env python3
"""
Benchmark: FlashInfer vs Triton (vllm) Gated Delta Rule Kernels

Compares three implementations:
1. Triton (vllm): fused_recurrent_gated_delta_rule_fwd
   - State: bfloat16, shape [N, HV, K, V]
   - Parameters: g (decay), beta (gate)
   - Supports spec=0,1,2,3
   
2. FlashInfer Decode: gated_delta_rule_decode (nontranspose version)
   - State: float32, shape [B, HV, K, V] (k-major)
   - Parameters: A_log, a, dt_bias, b
   - Only T=1 (decode mode)
   
3. FlashInfer MTP: gated_delta_rule_mtp
   - State: float32, shape [pool_size, HV, V, K] (k-last)
   - Parameters: A_log, a, dt_bias, b
   - Supports T=1,2,3,4 (spec=0,1,2,3)
   
Uses CUPTI timing (bench_gpu_time_with_cupti) with warm cache.
Benchmark settings: 10 warmup iterations + 100 measurement iterations.
Verified against FlashInfer benchmarks/bench_gdn_decode.py.
"""

import sys
import os
sys.path.insert(0, '/home/vgimpelson/1/flashinfer')
sys.path.insert(0, '/home/scratch.vgimpelson_ent/flashinfer')

import torch
import numpy as np
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.gdn_decode import gated_delta_rule_decode, gated_delta_rule_mtp
from vllm.model_executor.layers.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule_fwd

# Config
BATCH_SIZES = [32, 128, 256, 512, 1024]
TP_SIZES = [4, 2, 1]  # TP=1 matches FlashInfer default (H=16, HV=32)
SPEC_TOKENS = [0, 1, 2, 3]  # spec=0 means T=1, spec=1 means T=2, etc.


def create_decode_inputs(tp_size, batch_size):
    """Create input tensors for decode kernel benchmarking.
    
    Matches FlashInfer bench_gdn_decode.py setup for gated_delta_rule_decode (nontranspose).
    State layout: [B, HV, K, V] (k-major)
    """
    device = torch.device("cuda:0")
    H = 16 // tp_size  # num_q_heads = num_k_heads
    HV = 32 // tp_size  # num_v_heads (also num_o_heads)
    K, V = 128, 128  # head_size
    T = 1  # Decode mode: single token
    
    # Inputs for decode
    q = torch.randn(batch_size, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch_size, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch_size, T, HV, V, dtype=torch.bfloat16, device=device)
    
    # State: [B, HV, K, V] in float32 (k-major layout for nontranspose version)
    state = torch.randn(batch_size, HV, K, V, dtype=torch.float32, device=device)
    
    # Decay parameters (matching FlashInfer benchmark)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    a = torch.randn(batch_size, T, HV, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device=device)  # Can be bfloat16 or float32
    b = torch.randn(batch_size, T, HV, dtype=torch.bfloat16, device=device)
    
    # Output buffer
    output = torch.empty(batch_size, T, HV, V, dtype=torch.bfloat16, device=device)
    
    # Scale factor: 1 / sqrt(head_size)
    scale = 1.0 / (K ** 0.5)
    
    return {
        'q': q, 'k': k, 'v': v, 'state': state,
        'A_log': A_log, 'a': a, 'dt_bias': dt_bias, 'b': b,
        'scale': scale,
        'output': output,
        'use_qk_l2norm': True,
    }


def create_mtp_inputs(tp_size, batch_size, num_spec_tokens):
    """Create input tensors for MTP kernel benchmarking.
    
    Matches FlashInfer bench_gdn_decode.py setup for gated_delta_rule_mtp.
    State layout: [pool_size, HV, V, K] (K-last, interpreted from [pool, HV, head_size, head_size])
    """
    device = torch.device("cuda:0")
    H = 16 // tp_size  # num_q_heads = num_k_heads
    HV = 32 // tp_size  # num_v_heads (also num_o_heads)
    K, V = 128, 128  # head_size
    T = 1 + num_spec_tokens  # Total tokens to process
    
    # Inputs for MTP
    q = torch.randn(batch_size, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch_size, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch_size, T, HV, V, dtype=torch.bfloat16, device=device)
    
    # State pool: [pool_size, HV, head_size, head_size] in float32
    # Interpreted as [pool_size, HV, V, K] (K-last layout for MTP)
    pool_size = max(batch_size, 1024)
    initial_state = torch.randn(pool_size, HV, K, K, dtype=torch.float32, device=device)
    
    # Map each batch to a state in the pool
    initial_state_indices = torch.arange(0, batch_size, dtype=torch.int32, device=device)
    
    # Decay parameters (matching FlashInfer benchmark)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    a = torch.randn(batch_size, T, HV, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(batch_size, T, HV, dtype=torch.bfloat16, device=device)
    
    # Output buffer
    output = torch.empty(batch_size, T, HV, V, dtype=torch.bfloat16, device=device)
    
    # Intermediate states buffer: [pool_size, T, HV, head_size, head_size]
    # Interpreted as [pool_size, T, HV, V, K]
    intermediate_states_buffer = torch.empty(pool_size, T, HV, K, K, dtype=torch.float32, device=device)
    
    # Scale factor: 1 / sqrt(head_size)
    scale = 1.0 / (K ** 0.5)
    
    return {
        'q': q, 'k': k, 'v': v, 
        'initial_state': initial_state,
        'initial_state_indices': initial_state_indices,
        'A_log': A_log, 'a': a, 'dt_bias': dt_bias, 'b': b,
        'scale': scale,
        'output': output,
        'intermediate_states_buffer': intermediate_states_buffer,
        'disable_state_update': True,
        'use_qk_l2norm': True,
    }


def create_triton_inputs(tp_size, batch_size, num_spec_tokens):
    """Create input tensors for Triton (vllm) kernel benchmarking.
    
    vllm's fused_recurrent_gated_delta_rule_fwd uses:
    - Batch dimension flattened to 1 with cu_seqlens for sequence boundaries
    - State in bfloat16 (not float32)
    - Different parameterization: g (decay) and beta (gate) instead of A_log, a, dt_bias, b
    """
    device = torch.device("cuda:0")
    H = 16 // tp_size
    HV = 32 // tp_size
    K, V = 128, 128
    tokens_per_seq = 1 + num_spec_tokens
    total_tokens = batch_size * tokens_per_seq
    
    # vllm Triton uses batch dim = 1, with cu_seqlens for sequence boundaries
    q = torch.randn(1, total_tokens, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, total_tokens, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, total_tokens, HV, V, dtype=torch.bfloat16, device=device)
    g = torch.randn(1, total_tokens, HV, dtype=torch.float32, device=device)  # Decay parameter
    beta = torch.randn(1, total_tokens, HV, dtype=torch.bfloat16, device=device)  # Update gate
    
    # State in bfloat16 for Triton (different from FlashInfer's float32)
    max_state_slots = max(41069, batch_size * tokens_per_seq)
    initial_state = torch.randn(max_state_slots, HV, K, V, dtype=torch.bfloat16, device=device)
    
    # Sequence boundaries: [0, tokens_per_seq, 2*tokens_per_seq, ..., batch_size*tokens_per_seq]
    cu_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * tokens_per_seq
    
    # State indices mapping
    if num_spec_tokens > 0:
        # Speculative decoding: each token gets its own state slot
        ssm_state_indices = torch.arange(0, batch_size * tokens_per_seq, dtype=torch.int32, device=device).reshape(batch_size, tokens_per_seq)
        num_accepted_tokens = torch.full((batch_size,), tokens_per_seq, dtype=torch.int32, device=device)
    else:
        # Regular decode: one state per sequence
        ssm_state_indices = torch.arange(0, batch_size, dtype=torch.int32, device=device)
        num_accepted_tokens = None
    
    # Scale: 1 / sqrt(K) = 1 / sqrt(128) ≈ 0.0883883...
    scale = 1.0 / (K ** 0.5)
    
    return {
        'q': q, 'k': k, 'v': v, 'g': g, 'beta': beta,
        'scale': scale,
        'initial_state': initial_state,
        'inplace_final_state': True,
        'cu_seqlens': cu_seqlens,
        'ssm_state_indices': ssm_state_indices,
        'num_accepted_tokens': num_accepted_tokens,
        'use_qk_l2norm_in_kernel': True,
    }


def benchmark_decode_kernel(tp_size, batch_size):
    """Benchmark decode kernel and return median time in microseconds."""
    inputs = create_decode_inputs(tp_size, batch_size)
    
    # Warmup (matching FlashInfer benchmark)
    for _ in range(10):
        gated_delta_rule_decode(
            inputs['q'], inputs['k'], inputs['v'], inputs['state'],
            inputs['A_log'], inputs['a'], inputs['dt_bias'], inputs['b'],
            inputs['scale'], inputs['output'], inputs['use_qk_l2norm']
        )
        torch.cuda.synchronize()
    
    def run_kernel(q, k, v, state, A_log, a, dt_bias, b):
        return gated_delta_rule_decode(
            q, k, v, state, A_log, a, dt_bias, b,
            inputs['scale'], inputs['output'], inputs['use_qk_l2norm']
        )
    
    times_ms = bench_gpu_time_with_cupti(
        fn=run_kernel,
        input_args=(inputs['q'], inputs['k'], inputs['v'], inputs['state'],
                    inputs['A_log'], inputs['a'], inputs['dt_bias'], inputs['b']),
        dry_run_iters=0, repeat_iters=100, cold_l2_cache=True, use_cuda_graph=True, sleep_after_run=False,
    )
    return np.median(times_ms) * 1000.0


def benchmark_mtp_kernel(tp_size, batch_size, num_spec_tokens):
    """Benchmark MTP kernel and return median time in microseconds."""
    inputs = create_mtp_inputs(tp_size, batch_size, num_spec_tokens)
    
    # Warmup (matching FlashInfer benchmark)
    for _ in range(10):
        gated_delta_rule_mtp(
            inputs['q'], inputs['k'], inputs['v'], 
            inputs['initial_state'], inputs['initial_state_indices'],
            inputs['A_log'], inputs['a'], inputs['dt_bias'], inputs['b'],
            inputs['scale'], inputs['output'], inputs['intermediate_states_buffer'],
            inputs['disable_state_update'], inputs['use_qk_l2norm']
        )
        torch.cuda.synchronize()
    
    def run_kernel(q, k, v, initial_state, initial_state_indices, A_log, a, dt_bias, b):
        return gated_delta_rule_mtp(
            q, k, v, initial_state, initial_state_indices,
            A_log, a, dt_bias, b,
            inputs['scale'], inputs['output'], inputs['intermediate_states_buffer'],
            inputs['disable_state_update'], inputs['use_qk_l2norm']
        )
    
    times_ms = bench_gpu_time_with_cupti(
        fn=run_kernel,
        input_args=(inputs['q'], inputs['k'], inputs['v'], 
                    inputs['initial_state'], inputs['initial_state_indices'],
                    inputs['A_log'], inputs['a'], inputs['dt_bias'], inputs['b']),
        dry_run_iters=0, repeat_iters=100, cold_l2_cache=True, use_cuda_graph=True, sleep_after_run=False,
    )
    return np.median(times_ms) * 1000.0


def benchmark_triton_kernel(tp_size, batch_size, num_spec_tokens):
    """Benchmark Triton (vllm) kernel and return median time in microseconds."""
    inputs = create_triton_inputs(tp_size, batch_size, num_spec_tokens)
    
    # Warmup (matching FlashInfer benchmark)
    for _ in range(10):
        fused_recurrent_gated_delta_rule_fwd(**inputs)
        torch.cuda.synchronize()
    
    def run_kernel(q, k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices):
        return fused_recurrent_gated_delta_rule_fwd(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=inputs['scale'],
            initial_state=initial_state,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=inputs['num_accepted_tokens'],
            use_qk_l2norm_in_kernel=True,
        )
    
    times_ms = bench_gpu_time_with_cupti(
        fn=run_kernel,
        input_args=(inputs['q'], inputs['k'], inputs['v'], inputs['g'], inputs['beta'],
                    inputs['initial_state'], inputs['cu_seqlens'], inputs['ssm_state_indices']),
        dry_run_iters=0, repeat_iters=100, cold_l2_cache=True, use_cuda_graph=True, sleep_after_run=False,
    )
    return np.median(times_ms) * 1000.0


if __name__ == "__main__":
    torch.cuda.set_device(0)
    
    print(f"\n{'='*90}")
    print(f"Benchmark: FlashInfer vs Triton (vllm) Gated Delta Rule")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernels: Triton (vllm) + FlashInfer decode + FlashInfer MTP")
    print(f"Timing: CUPTI cold L2 + CUDA graphs | Warmup: 10 iters, Benchmark: 100 iters")
    print(f"{'='*90}")
    
    # Run benchmarks
    triton_results = {}
    decode_results = {}
    mtp_results = {}
    
    # Total: triton (TP * BS * SPEC) + decode (TP * BS) + mtp (TP * BS * SPEC)
    total = len(TP_SIZES) * len(BATCH_SIZES) * (len(SPEC_TOKENS) + 1 + len(SPEC_TOKENS))
    current = 0
    
    for tp in TP_SIZES:
        triton_results[tp] = {}
        decode_results[tp] = {}
        mtp_results[tp] = {}
        
        for bs in BATCH_SIZES:
            # Triton kernel (spec=0,1,2,3)
            triton_results[tp][bs] = {}
            for spec in SPEC_TOKENS:
                current += 1
                print(f"[{current}/{total}] TP={tp} BS={bs:4d} triton spec={spec} (T={1+spec})...", end=" ", flush=True)
                triton_results[tp][bs][spec] = benchmark_triton_kernel(tp, bs, spec)
                print(f"{triton_results[tp][bs][spec]:.1f} μs")
            
            # Decode kernel (T=1, k-major state)
            current += 1
            print(f"[{current}/{total}] TP={tp} BS={bs:4d} flashinfer decode (T=1)...", end=" ", flush=True)
            decode_results[tp][bs] = benchmark_decode_kernel(tp, bs)
            print(f"{decode_results[tp][bs]:.1f} μs")
            
            # MTP kernel (T=1,2,3,4, k-last state)
            mtp_results[tp][bs] = {}
            for spec in SPEC_TOKENS:
                current += 1
                print(f"[{current}/{total}] TP={tp} BS={bs:4d} flashinfer mtp spec={spec} (T={1+spec})...", end=" ", flush=True)
                mtp_results[tp][bs][spec] = benchmark_mtp_kernel(tp, bs, spec)
                print(f"{mtp_results[tp][bs][spec]:.1f} μs")
    
    # Print results tables
    print(f"\n{'='*100}")
    print("RESULTS - ALL KERNELS (Median Time in μs)")
    print(f"{'='*100}")
    
    for tp in TP_SIZES:
        print(f"\nTP={tp} (H={16//tp}, HV={32//tp})")
        print(f"{'Batch':>6} | {'Spec':>4} | {'Triton (vllm)':>15} | {'FI Decode':>12} | {'FI MTP':>12} | {'Triton/Decode':>14} | {'Triton/MTP':>12}")
        print("-" * 100)
        
        for bs in BATCH_SIZES:
            for spec in SPEC_TOKENS:
                triton_time = triton_results[tp][bs][spec]
                decode_time = decode_results[tp][bs] if spec == 0 else None
                mtp_time = mtp_results[tp][bs][spec]
                
                decode_str = f"{decode_time:>12.1f}" if decode_time else "      -     "
                ratio_decode = f"{triton_time/decode_time:>14.3f}" if decode_time else "      -       "
                ratio_mtp = f"{triton_time/mtp_time:>12.3f}"
                
                print(f"{bs:>6} | {spec:>4} | {triton_time:>15.1f} | {decode_str} | {mtp_time:>12.1f} | {ratio_decode} | {ratio_mtp}")
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY (Mean of Medians across batch sizes):")
    print(f"{'='*100}")
    
    for tp in TP_SIZES:
        print(f"\nTP={tp}:")
        for spec in SPEC_TOKENS:
            triton_times = np.array([triton_results[tp][bs][spec] for bs in BATCH_SIZES])
            mtp_times = np.array([mtp_results[tp][bs][spec] for bs in BATCH_SIZES])
            
            speedup = triton_times.mean() / mtp_times.mean()
            
            print(f"  Spec={spec} (T={1+spec}): Triton={triton_times.mean():.1f}μs  FI-MTP={mtp_times.mean():.1f}μs  "
                  f"Speedup={speedup:.2f}x {'(FI faster)' if speedup > 1 else '(Triton faster)'}")
        
        # Decode comparison only for spec=0
        decode_times = np.array([decode_results[tp][bs] for bs in BATCH_SIZES])
        triton_spec0_times = np.array([triton_results[tp][bs][0] for bs in BATCH_SIZES])
        speedup_decode = triton_spec0_times.mean() / decode_times.mean()
        print(f"  Spec=0 Decode: Triton={triton_spec0_times.mean():.1f}μs  FI-Decode={decode_times.mean():.1f}μs  "
              f"Speedup={speedup_decode:.2f}x {'(FI faster)' if speedup_decode > 1 else '(Triton faster)'}")
    
    # Best kernel per configuration
    print(f"\n{'='*100}")
    print("WINNER (fastest kernel per TP/Spec combination):")
    print(f"{'='*100}")
    
    for tp in TP_SIZES:
        print(f"\nTP={tp}:")
        for spec in SPEC_TOKENS:
            triton_mean = np.mean([triton_results[tp][bs][spec] for bs in BATCH_SIZES])
            mtp_mean = np.mean([mtp_results[tp][bs][spec] for bs in BATCH_SIZES])
            
            if spec == 0:
                decode_mean = np.mean([decode_results[tp][bs] for bs in BATCH_SIZES])
                times = [('Triton', triton_mean), ('FI-Decode', decode_mean), ('FI-MTP', mtp_mean)]
            else:
                times = [('Triton', triton_mean), ('FI-MTP', mtp_mean)]
            
            winner = min(times, key=lambda x: x[1])
            print(f"  Spec={spec} (T={1+spec}): {winner[0]} ({winner[1]:.1f} μs)")
    
    print(f"{'='*100}")
