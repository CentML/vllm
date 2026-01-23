#!/usr/bin/env python3
"""
Benchmark: fused_recurrent_gated_delta_rule_fwd
Compares Triton (all specs) vs CUDA v50 TMA (CUTLASS) for spec=0
Uses CUPTI timing with cold L2 cache.
"""

import sys
import os
sys.path.insert(0, '/home/scratch.vgimpelson_ent/flashinfer')

import torch
import numpy as np
import ctypes
from flashinfer.testing import bench_gpu_time_with_cupti
from vllm.model_executor.layers.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule_fwd

# TMA descriptor type
class CUtensorMap(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 128)]

# Load v50 TMA kernel (CUTLASS barriers)
_lib_v50 = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libgdr_v50_tma.so'))
_lib_v50.create_tma_descriptor_state_tensor.argtypes = [
    ctypes.POINTER(CUtensorMap), ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
]
_lib_v50.create_tma_descriptor_state_tensor.restype = ctypes.c_int
_lib_v50.launch_gdr_tma.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_float, ctypes.c_int64, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
    ctypes.c_void_p, ctypes.c_void_p,
]
_lib_v50.launch_gdr_tma.restype = None

# Config
BATCH_SIZES = [32, 128, 256, 512, 1024]
TP_SIZES = [4, 2, 1]


def create_inputs(tp_size, batch_size, num_spec_tokens=0):
    """Create input tensors for benchmarking."""
    device = torch.device("cuda:0")
    H = 16 // tp_size
    HV = 32 // tp_size
    K, V = 128, 128
    tokens_per_seq = 1 + num_spec_tokens
    total_tokens = batch_size * tokens_per_seq
    
    q = torch.randn(1, total_tokens, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, total_tokens, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, total_tokens, HV, V, dtype=torch.bfloat16, device=device)
    g = torch.randn(1, total_tokens, HV, dtype=torch.float32, device=device)
    beta = torch.randn(1, total_tokens, HV, dtype=torch.bfloat16, device=device)
    
    max_state_slots = max(41069, batch_size * tokens_per_seq)
    initial_state = torch.randn(max_state_slots, HV, K, V, dtype=torch.bfloat16, device=device)
    cu_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * tokens_per_seq
    
    if num_spec_tokens > 0:
        ssm_state_indices = torch.arange(0, batch_size * tokens_per_seq, dtype=torch.int32, device=device).reshape(batch_size, tokens_per_seq)
        num_accepted_tokens = torch.full((batch_size,), tokens_per_seq, dtype=torch.int32, device=device)
    else:
        ssm_state_indices = torch.arange(0, batch_size, dtype=torch.int32, device=device)
        num_accepted_tokens = None
    
    return {
        'q': q, 'k': k, 'v': v, 'g': g, 'beta': beta,
        'scale': 0.08838834764831845,
        'initial_state': initial_state,
        'inplace_final_state': True,
        'cu_seqlens': cu_seqlens,
        'ssm_state_indices': ssm_state_indices,
        'num_accepted_tokens': num_accepted_tokens,
        'use_qk_l2norm_in_kernel': True,
    }


def benchmark_kernel(tp_size, batch_size, num_spec_tokens=0, kernel='triton'):
    """Benchmark kernel and return median time in microseconds."""
    inputs = create_inputs(tp_size, batch_size, num_spec_tokens)
    
    if kernel == 'triton':
        for _ in range(3):
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
    
    elif kernel == 'cuda_v50':
        if num_spec_tokens != 0:
            raise RuntimeError("cuda_v50 only supports spec=0")
        
        state = inputs['initial_state']
        HV, K, V = state.shape[1], state.shape[2], state.shape[3]
        tma_desc = CUtensorMap()
        _lib_v50.create_tma_descriptor_state_tensor(
            ctypes.byref(tma_desc), state.data_ptr(),
            state.shape[0] * HV * K, V, V * state.element_size()
        )
        tma_desc_device = torch.frombuffer(bytearray(bytes(tma_desc.data)), dtype=torch.uint8).clone().cuda()
        stride = state.stride(0)
        
        for _ in range(3):
            stream = torch.cuda.current_stream().cuda_stream
            _lib_v50.launch_gdr_tma(
                inputs['q'].data_ptr(), inputs['k'].data_ptr(), inputs['v'].data_ptr(),
                inputs['g'].data_ptr(), inputs['beta'].data_ptr(),
                torch.empty_like(inputs['v']).data_ptr(),
                state.data_ptr(), state.data_ptr(), inputs['ssm_state_indices'].data_ptr(),
                ctypes.c_float(inputs['scale']), ctypes.c_int64(inputs['q'].shape[1]),
                ctypes.c_int(inputs['q'].shape[2]), ctypes.c_int(HV),
                ctypes.c_int(K), ctypes.c_int(V), ctypes.c_int64(stride),
                tma_desc_device.data_ptr(), stream
            )
            torch.cuda.synchronize()
        
        def run_kernel(q, k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices):
            out = torch.empty_like(v)
            stream = torch.cuda.current_stream().cuda_stream
            _lib_v50.launch_gdr_tma(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
                out.data_ptr(), initial_state.data_ptr(), initial_state.data_ptr(),
                ssm_state_indices.data_ptr(), ctypes.c_float(inputs['scale']),
                ctypes.c_int64(q.shape[1]), ctypes.c_int(q.shape[2]),
                ctypes.c_int(HV), ctypes.c_int(K), ctypes.c_int(V),
                ctypes.c_int64(stride), tma_desc_device.data_ptr(), stream
            )
            return out
    else:
        raise RuntimeError(f"Unknown kernel: {kernel}")
    
    times_ms = bench_gpu_time_with_cupti(
        fn=run_kernel,
        input_args=(inputs['q'], inputs['k'], inputs['v'], inputs['g'], inputs['beta'],
                    inputs['initial_state'], inputs['cu_seqlens'], inputs['ssm_state_indices']),
        cold_l2_cache=True, use_cuda_graph=False, sleep_after_run=False,
    )
    return np.median(times_ms) * 1000.0  # Return median in microseconds


if __name__ == "__main__":
    torch.cuda.set_device(0)
    
    print(f"\n{'='*90}")
    print(f"Benchmark: fused_recurrent_gated_delta_rule_fwd | GPU: {torch.cuda.get_device_name(0)}")
    print(f"Comparing: Triton (baseline) vs v50 (CUTLASS barriers)")
    print(f"{'='*90}")
    
    # Run benchmarks
    results = {}
    total = len(TP_SIZES) * len(BATCH_SIZES) * (2 + 3)  # 2 kernels for spec0, 3 specs for triton
    current = 0
    
    for tp in TP_SIZES:
        results[tp] = {}
        for bs in BATCH_SIZES:
            results[tp][bs] = {}
            # Spec=0: triton and cuda_v50
            for kernel in ['triton', 'cuda_v50']:
                current += 1
                print(f"[{current}/{total}] TP={tp} BS={bs:4d} spec=0 {kernel:>9s}...", end=" ", flush=True)
                results[tp][bs][f's0_{kernel}'] = benchmark_kernel(tp, bs, 0, kernel)
                print(f"{results[tp][bs][f's0_{kernel}']:.1f} μs")
            # Spec=1,2,3: triton only
            for spec in [1, 2, 3]:
                current += 1
                print(f"[{current}/{total}] TP={tp} BS={bs:4d} spec={spec} triton...", end=" ", flush=True)
                results[tp][bs][f's{spec}_triton'] = benchmark_kernel(tp, bs, spec, 'triton')
                print(f"{results[tp][bs][f's{spec}_triton']:.1f} μs")
    
    # Print results table
    print(f"\n{'='*90}")
    print("RESULTS (μs) | Speedup = Triton/CUDA (>1 means CUDA faster)")
    print(f"{'='*90}")
    
    for tp in TP_SIZES:
        print(f"\nTP={tp}")
        print(f"{'Batch':>6} | {'-----SPEC=0-----':^32} | {'SPEC=1':>8} | {'SPEC=2':>8} | {'SPEC=3':>8}")
        print(f"{'':>6} | {'Triton':>8} {'v50(CUT)':>10} {'v50 spd':>8} | {'Triton':>8} | {'Triton':>8} | {'Triton':>8}")
        print("-" * 90)
        
        for bs in BATCH_SIZES:
            r = results[tp][bs]
            speedup_v50 = r['s0_triton'] / r['s0_cuda_v50']
            print(f"{bs:>6} | {r['s0_triton']:>8.1f} {r['s0_cuda_v50']:>10.1f} {speedup_v50:>7.2f}x |"
                  f" {r['s1_triton']:>8.1f} | {r['s2_triton']:>8.1f} | {r['s3_triton']:>8.1f}")
    
    # Summary
    speedups_v50 = [results[tp][bs]['s0_triton'] / results[tp][bs]['s0_cuda_v50'] 
                    for tp in TP_SIZES for bs in BATCH_SIZES]
    speedups_v50 = np.array(speedups_v50)
    
    print(f"\n{'='*90}")
    print(f"SUMMARY (Spec=0):")
    print(f"  v50 (CUTLASS):       Min={speedups_v50.min():.2f}x  Max={speedups_v50.max():.2f}x  Mean={speedups_v50.mean():.2f}x")
    
    if speedups_v50.mean() > 1.0:
        print(f"\n✓ CUDA v50 (CUTLASS) is {(speedups_v50.mean()-1)*100:.0f}% faster than Triton on average")
    else:
        print(f"\n✗ Triton is {(1/speedups_v50.mean()-1)*100:.0f}% faster than CUDA v50 on average")
    print(f"{'='*90}")
