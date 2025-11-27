"""
Combined performance benchmark comparing:
- FP16 torch.nn.functional.linear
- FP8 gemm_fp8_nt_groupwise (cutlass backend)
- FP8 gemm_fp8_nt_groupwise (trtllm backend)

Tests various shapes and batch sizes with unified reporting.
"""

import os
import sys
import torch
import numpy as np

# Set environment variables
os.environ['CUDA_HOME'] = '/home/scratch.vgimpelson_ent/cuda/cuda130'
os.environ['LD_LIBRARY_PATH'] = '/home/scratch.vgimpelson_ent/cuda/cuda130/lib64:/home/scratch.vgimpelson_ent/cuda/cuda130/extras/CUPTI/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'

# Add flashinfer path
sys.path.insert(0, '/home/vgimpelson/1/flashinfer')

from flashinfer.testing.utils import bench_gpu_time_with_cupti, quantize_fp8

# Import FlashInfer after path setup
sys.path.insert(0, '/home/scratch.vgimpelson_ent/flashinfer')
from flashinfer.gemm import gemm_fp8_nt_groupwise

# Import vLLM for cutlass_scaled_mm
try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        cutlass_scaled_mm as vllm_cutlass_scaled_mm,
        per_token_group_quant_fp8,
    )
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, cutlass_scaled_mm will be skipped")


def create_fp8_tensors(m, n, k, scale_major_mode="MN"):
    """Create FP8 tensors with proper scaling for gemm_fp8_nt_groupwise."""
    block_size = 128
    
    # Create input tensors in bfloat16
    a_bf16 = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    b_bf16 = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    
    a_scale_shape = (k // block_size, m)
    a_tile_shape = (1, block_size)
    
    b_scale_shape = (k // block_size, n // block_size)
    b_tile_shape = (block_size, block_size)
    
    a_fp8, a_scale = quantize_fp8(a_bf16, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_bf16, b_scale_shape, b_tile_shape, scale_major_mode)
    
    # Create output tensor
    out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    
    return a_fp8, b_fp8, a_scale, b_scale, out


def benchmark_fp16_linear(batch_size, out_features, in_features):
    """Benchmark FP16 torch.nn.functional.linear."""
    device = 'cuda'
    dtype = torch.float16
    
    # Create tensors
    weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
    bias = torch.randn(out_features, device=device, dtype=dtype)
    input_tensor = torch.randn(batch_size, in_features, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(5):
        _ = torch.nn.functional.linear(input_tensor, weight, bias)
    torch.cuda.synchronize()
    
    # Benchmark function
    def benchmark_fn():
        return torch.nn.functional.linear(input_tensor, weight, bias)
    
    # Run benchmark with CUPTI
    times = bench_gpu_time_with_cupti(
        benchmark_fn,
        l2_flush=False,
        repeat_iters=100,
    )
    
    # Calculate statistics (convert ms to us)
    times_np = np.array(times) * 1000  # ms to us
    median_time = np.median(times_np)
    std_time = np.std(times_np)
    std_pct = (std_time / median_time) * 100 if median_time > 0 else 0
    
    # Calculate FLOPS
    flops = 2 * batch_size * in_features * out_features + batch_size * out_features
    tflops_per_sec = flops / (median_time * 1e-6) / 1e12
    
    return {
        'median_us': median_time,
        'std_pct': std_pct,
        'tflops_per_sec': tflops_per_sec,
    }


def benchmark_fp8_gemm(batch_size, out_features, in_features, backend='cutlass'):
    """Benchmark FP8 gemm_fp8_nt_groupwise."""
    m, n, k = batch_size, out_features, in_features
    scale_major_mode = 'MN'
    
    # Create tensors
    a_fp8, b_fp8, a_scale, b_scale, out = create_fp8_tensors(m, n, k, scale_major_mode)
    
    # Warmup
    for _ in range(5):
        gemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale, b_scale,
            scale_major_mode=scale_major_mode,
            mma_sm=1,
            out=out,
            backend=backend,
        )
    torch.cuda.synchronize()
    
    # Benchmark function
    def benchmark_fn():
        return gemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale, b_scale,
            scale_major_mode=scale_major_mode,
            mma_sm=1,
            out=out,
            backend=backend,
        )
    
    # Run benchmark with CUPTI
    times = bench_gpu_time_with_cupti(
        benchmark_fn,
        l2_flush=False,
        repeat_iters=100,
    )
    
    # Calculate statistics (convert ms to us)
    times_np = np.array(times) * 1000  # ms to us
    median_time = np.median(times_np)
    std_time = np.std(times_np)
    std_pct = (std_time / median_time) * 100 if median_time > 0 else 0
    
    # Calculate FLOPS
    flops = 2 * m * n * k
    tflops_per_sec = flops / (median_time * 1e-6) / 1e12
    
    return {
        'median_us': median_time,
        'std_pct': std_pct,
        'tflops_per_sec': tflops_per_sec,
    }


def benchmark_cutlass_scaled_mm(batch_size, out_features, in_features):
    """Benchmark vLLM's cutlass_scaled_mm with blockwise scaling (128x128 blocks)."""
    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM not available")
    
    m, n, k = batch_size, out_features, in_features
    device = 'cuda'
    block_size = [128, 128]  # Same as FlashInfer blockwise
    
    # Create input tensor in bfloat16
    input_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    
    # Quantize input with per-token blockwise quantization
    # This creates [m, k] FP8 tensor with [m, k//128] scales (column-major)
    input_fp8, input_scale = per_token_group_quant_fp8(
        input_bf16,
        group_size=128,
        column_major_scales=True,
        use_ue8m0=False,
    )
    
    # Create weight tensor with blockwise quantization
    # Weight is [n, k], scale is [n//128, k//128]
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    
    # Quantize weight blockwise (128x128 blocks)
    block_n, block_k = block_size
    num_blocks_n = (n + block_n - 1) // block_n
    num_blocks_k = (k + block_k - 1) // block_k
    
    weight_scale = torch.empty(num_blocks_n, num_blocks_k, device=device, dtype=torch.float32)
    weight_fp8 = torch.empty_like(weight_bf16, dtype=torch.float8_e4m3fn)
    
    # Simple blockwise quantization for weight
    for i in range(num_blocks_n):
        for j in range(num_blocks_k):
            block_start_n = i * block_n
            block_end_n = min((i + 1) * block_n, n)
            block_start_k = j * block_k
            block_end_k = min((j + 1) * block_k, k)
            
            block = weight_bf16[block_start_n:block_end_n, block_start_k:block_end_k]
            absmax = torch.max(torch.abs(block))
            scale = absmax / 448.0  # FP8 E4M3 max value
            weight_scale[i, j] = scale
            weight_fp8[block_start_n:block_end_n, block_start_k:block_end_k] = (block / scale).to(torch.float8_e4m3fn)
    
    # Use vLLM's cutlass_scaled_mm with blockwise scales
    result = vllm_cutlass_scaled_mm(input_fp8, weight_fp8, input_scale, weight_scale, block_size, torch.bfloat16)
    
    # Warmup
    for _ in range(5):
        result = vllm_cutlass_scaled_mm(input_fp8, weight_fp8, input_scale, weight_scale, block_size, torch.bfloat16)
    torch.cuda.synchronize()
    
    # Benchmark function
    def benchmark_fn():
        return vllm_cutlass_scaled_mm(input_fp8, weight_fp8, input_scale, weight_scale, block_size, torch.bfloat16)
    
    # Run benchmark with CUPTI
    times = bench_gpu_time_with_cupti(
        benchmark_fn,
        l2_flush=False,
        repeat_iters=100,
    )
    
    # Calculate statistics (convert ms to us)
    times_np = np.array(times) * 1000  # ms to us
    median_time = np.median(times_np)
    std_time = np.std(times_np)
    std_pct = (std_time / median_time) * 100 if median_time > 0 else 0
    
    # Calculate FLOPS
    flops = 2 * m * n * k
    tflops_per_sec = flops / (median_time * 1e-6) / 1e12
    
    return {
        'median_us': median_time,
        'std_pct': std_pct,
        'tflops_per_sec': tflops_per_sec,
    }


def run_combined_benchmark():
    """Run combined benchmark for all configurations."""
    
    print("=" * 170)
    print("Combined Performance Benchmark: FP16 Linear vs FP8 GEMM")
    print("=" * 170)
    print("Configurations:")
    print("  - FP16: torch.nn.functional.linear")
    print("  - FlashInfer Cutlass: gemm_fp8_nt_groupwise (cutlass backend, blockwise)")
    print("  - FlashInfer TRT-LLM: gemm_fp8_nt_groupwise (trtllm backend, blockwise)")
    print("  - vLLM Cutlass: vllm.fp8_utils.cutlass_scaled_mm (blockwise)")
    print("  - Benchmarking with CUPTI, L2 flush enabled, 128x128 blocks")
    print("=" * 170)
    print()
    
    # Test configurations
    shapes = [
        [2048, 1024],
        [2048, 128],
        [256, 2048],
        [2560, 2048],
        [3072, 2048],
    ]
    
    batch_sizes = [128, 256, 512, 1024]
    
    results = []
    
    for out_features, in_features in shapes:
        for batch_size in batch_sizes:
            print(f"\nTesting: Batch={batch_size}, Out={out_features}, In={in_features}")
            
            result = {
                'batch': batch_size,
                'out': out_features,
                'in': in_features,
            }
            
            # Benchmark FP16
            try:
                print("  Running FP16 linear...")
                fp16_result = benchmark_fp16_linear(batch_size, out_features, in_features)
                result['fp16_median'] = fp16_result['median_us']
                result['fp16_std_pct'] = fp16_result['std_pct']
                result['fp16_tflops'] = fp16_result['tflops_per_sec']
            except Exception as e:
                print(f"  FP16 ERROR: {e}")
                result['fp16_median'] = None
                result['fp16_std_pct'] = None
                result['fp16_tflops'] = None
            
            # Benchmark FP8 Cutlass
            try:
                print("  Running FP8 Cutlass...")
                cutlass_result = benchmark_fp8_gemm(batch_size, out_features, in_features, backend='cutlass')
                result['cutlass_median'] = cutlass_result['median_us']
                result['cutlass_std_pct'] = cutlass_result['std_pct']
                result['cutlass_tflops'] = cutlass_result['tflops_per_sec']
                
                # Calculate ratio
                if result['fp16_median'] is not None:
                    result['cutlass_ratio'] = result['fp16_median'] / result['cutlass_median']
                else:
                    result['cutlass_ratio'] = None
            except Exception as e:
                print(f"  Cutlass ERROR: {e}")
                result['cutlass_median'] = None
                result['cutlass_std_pct'] = None
                result['cutlass_tflops'] = None
                result['cutlass_ratio'] = None
            
            # Benchmark FP8 TRT-LLM
            try:
                print("  Running FP8 TRT-LLM...")
                trtllm_result = benchmark_fp8_gemm(batch_size, out_features, in_features, backend='trtllm')
                result['trtllm_median'] = trtllm_result['median_us']
                result['trtllm_std_pct'] = trtllm_result['std_pct']
                result['trtllm_tflops'] = trtllm_result['tflops_per_sec']
                
                # Calculate ratio
                if result['fp16_median'] is not None:
                    result['trtllm_ratio'] = result['fp16_median'] / result['trtllm_median']
                else:
                    result['trtllm_ratio'] = None
            except Exception as e:
                print(f"  TRT-LLM ERROR: {e}")
                result['trtllm_median'] = None
                result['trtllm_std_pct'] = None
                result['trtllm_tflops'] = None
                result['trtllm_ratio'] = None
            
            # Benchmark vLLM cutlass_scaled_mm (blockwise)
            try:
                print("  Running vLLM Cutlass...")
                vllm_cutlass_result = benchmark_cutlass_scaled_mm(batch_size, out_features, in_features)
                result['vllm_cutlass_median'] = vllm_cutlass_result['median_us']
                result['vllm_cutlass_std_pct'] = vllm_cutlass_result['std_pct']
                result['vllm_cutlass_tflops'] = vllm_cutlass_result['tflops_per_sec']
                
                # Calculate ratio
                if result['fp16_median'] is not None:
                    result['vllm_cutlass_ratio'] = result['fp16_median'] / result['vllm_cutlass_median']
                else:
                    result['vllm_cutlass_ratio'] = None
            except Exception as e:
                print(f"  vLLM Cutlass ERROR: {e}")
                result['vllm_cutlass_median'] = None
                result['vllm_cutlass_std_pct'] = None
                result['vllm_cutlass_tflops'] = None
                result['vllm_cutlass_ratio'] = None
            
            results.append(result)
    
    # Print summary table
    print("\n" + "=" * 170)
    print("SUMMARY TABLE")
    print("=" * 170)
    # First header line with column categories - must align "|" with data rows
    # Data format: {5} {5} {6} | {10} {7} | {10} {7} {14} | {10} {7} {14} | {10} {7} {14}
    # Section widths after "|": 18 | 33 | 33 | 33
    print(f"{'Out':>5} {'In':>5} {'Batch':>6} | "
          f"{'FP16':^18} | "
          f"{'FlashInfer Cutlass':^33} | "
          f"{'FlashInfer TRT-LLM':^33} | "
          f"{'vLLM Cutlass':^33}")
    # Second header line with specific metrics - must match exact spacing of data rows
    print(f"{'':>5} {'':>5} {'':>6} | "
          f"{'Median(us)':>10} {'Std(%)':>7} | "
          f"{'Median(us)':>10} {'Std(%)':>7} {'Ratio to FP16':>14} | "
          f"{'Median(us)':>10} {'Std(%)':>7} {'Ratio to FP16':>14} | "
          f"{'Median(us)':>10} {'Std(%)':>7} {'Ratio to FP16':>14}")
    print("-" * 170)
    
    for r in results:
        # FP16 values
        fp16_med = f"{r['fp16_median']:10.2f}" if r['fp16_median'] is not None else "         -"
        fp16_std = f"{r['fp16_std_pct']:7.2f}" if r['fp16_std_pct'] is not None else "      -"
        
        # Cutlass blockwise values
        cutlass_med = f"{r['cutlass_median']:10.2f}" if r['cutlass_median'] is not None else "         -"
        cutlass_std = f"{r['cutlass_std_pct']:7.2f}" if r['cutlass_std_pct'] is not None else "      -"
        cutlass_ratio = f"{r['cutlass_ratio']:14.2f}" if r['cutlass_ratio'] is not None else "             -"
        
        # TRT-LLM blockwise values
        trtllm_med = f"{r['trtllm_median']:10.2f}" if r['trtllm_median'] is not None else "         -"
        trtllm_std = f"{r['trtllm_std_pct']:7.2f}" if r['trtllm_std_pct'] is not None else "      -"
        trtllm_ratio = f"{r['trtllm_ratio']:14.2f}" if r['trtllm_ratio'] is not None else "             -"
        
        # vLLM Cutlass blockwise values
        vllm_cutlass_med = f"{r['vllm_cutlass_median']:10.2f}" if r['vllm_cutlass_median'] is not None else "         -"
        vllm_cutlass_std = f"{r['vllm_cutlass_std_pct']:7.2f}" if r['vllm_cutlass_std_pct'] is not None else "      -"
        vllm_cutlass_ratio = f"{r['vllm_cutlass_ratio']:14.2f}" if r['vllm_cutlass_ratio'] is not None else "             -"
        
        print(f"{r['out']:5d} {r['in']:5d} {r['batch']:6d} | "
              f"{fp16_med} {fp16_std} | "
              f"{cutlass_med} {cutlass_std} {cutlass_ratio} | "
              f"{trtllm_med} {trtllm_std} {trtllm_ratio} | "
              f"{vllm_cutlass_med} {vllm_cutlass_std} {vllm_cutlass_ratio}")
    
    print("=" * 170)
    print("Notes:")
    print("  - Median: Median execution time in microseconds")
    print("  - Std(%): Standard deviation as percentage of median")
    print("  - Ratio to FP16: FP16 time / FP8 time (higher is better for FP8)")
    print("  - All FP8 methods use 128x128 block quantization with per-token activation scaling")
    print("  - FlashInfer uses gemm_fp8_nt_groupwise, vLLM uses cutlass_scaled_mm")
    print("  - '-' indicates unsupported configuration")
    print("=" * 170)
    
    return results


if __name__ == "__main__":
    results = run_combined_benchmark()

