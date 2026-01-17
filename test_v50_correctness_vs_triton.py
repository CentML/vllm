#!/usr/bin/env python3
"""
Functional test for v50 TMA kernel - tests correctness vs Triton reference.

The v50 kernel (fused_recurrent_gated_delta_rule_v50_tma.cu) uses:
- TMA for state tensor loading
- CUTLASS barrier primitives (instead of raw PTX assembly)
- Standard thread block (no clusters)
- Works only for spec=0 (1 token per sequence)

This test verifies the v50 CUDA kernel produces the same outputs as the
Triton reference implementation.
"""

import sys
import os

sys.path.insert(0, '/home/scratch.vgimpelson_ent/flashinfer')

import torch
import ctypes

# Import Triton reference
from vllm.model_executor.layers.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule_fwd

# TMA descriptor type
class CUtensorMap(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 128)]

# Load v50 kernel from pre-compiled .so
_lib_v50 = None
v50_available = False

try:
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libgdr_v50_tma.so')
    _lib_v50 = ctypes.CDLL(lib_path)
    
    _lib_v50.create_tma_descriptor_state_tensor.argtypes = [
        ctypes.POINTER(CUtensorMap),
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    _lib_v50.create_tma_descriptor_state_tensor.restype = ctypes.c_int

    # TMA kernel (returns void)
    _lib_v50.launch_gdr_tma.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_float, ctypes.c_int64, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    _lib_v50.launch_gdr_tma.restype = None  # Returns void
    
    v50_available = True
    print("✓ V50 TMA kernel (CUTLASS barriers) loaded from libgdr_v50_tma.so")
except OSError as e:
    print(f"✗ V50 kernel not available: {e}")
    sys.exit(1)


device = torch.device("cuda:0")


def test_v50_vs_triton(batch_size: int, tp_size: int = 4, seed: int = 42, verbose: bool = True):
    """
    Test v50 kernel against Triton reference for spec=0 (1 token per sequence).
    
    Args:
        batch_size: Number of sequences (each with 1 token for spec=0)
        tp_size: Tensor parallel size (affects H and HV)
        seed: Random seed for reproducibility
        verbose: Print detailed output
        
    Returns:
        Dictionary with test results
    """
    torch.manual_seed(seed)
    
    # Model configuration
    total_num_key_heads = 16
    total_num_value_heads = 32
    key_head_dim = 128
    value_head_dim = 128
    
    # Per-GPU dimensions based on TP size
    H = total_num_key_heads // tp_size     # num key heads per GPU
    HV = total_num_value_heads // tp_size  # num value heads per GPU
    K = key_head_dim
    V = value_head_dim
    
    # For spec=0: 1 token per sequence
    total_tokens = batch_size
    
    # Create input tensors - Triton expects [1, total_tokens, H/HV, K/V] shape
    q = torch.randn(1, total_tokens, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, total_tokens, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, total_tokens, HV, V, dtype=torch.bfloat16, device=device)
    g = torch.randn(1, total_tokens, HV, dtype=torch.float32, device=device)
    beta = torch.randn(1, total_tokens, HV, dtype=torch.bfloat16, device=device)
    
    scale = 0.08838834764831845  # 1.0 / sqrt(128)
    
    # State tensors - only allocate what we need for this batch
    max_state_slots = batch_size
    initial_state = torch.randn(max_state_slots, HV, K, V, dtype=torch.bfloat16, device=device)
    
    # For spec=0: 1D ssm_state_indices
    ssm_state_indices = torch.arange(0, batch_size, dtype=torch.int32, device=device)
    
    # cu_seqlens for varlen: [0, 1, 2, ..., N]
    cu_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
    
    # =========================================================================
    # Run Triton reference
    # =========================================================================
    o_triton, final_state_triton = fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state.clone(),
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,  # None for spec=0
        use_qk_l2norm_in_kernel=True,  # v50 uses L2 norm
    )
    torch.cuda.synchronize()
    
    # =========================================================================
    # Run v50 CUDA kernel
    # =========================================================================
    
    # Create TMA descriptor
    state_tensor = initial_state.clone()  # Fresh copy for v50
    tma_desc = CUtensorMap()
    max_states = state_tensor.shape[0]
    
    total_rows = max_states * HV * K
    num_cols = V
    stride_row_bytes = V * state_tensor.element_size()
    
    result = _lib_v50.create_tma_descriptor_state_tensor(
        ctypes.byref(tma_desc),
        state_tensor.data_ptr(),
        total_rows,
        num_cols,
        stride_row_bytes
    )
    
    if result != 0:
        raise RuntimeError(f"TMA descriptor creation failed: {result}")
    
    tma_desc_device = torch.frombuffer(
        bytearray(bytes(tma_desc.data)), dtype=torch.uint8
    ).clone().to(device)
    
    # Output tensor - same shape as v
    o_v50 = torch.empty_like(v)
    
    stride = state_tensor.stride(0)
    stream = torch.cuda.current_stream().cuda_stream
    
    # Use TMA kernel (returns void, no error check available)
    _lib_v50.launch_gdr_tma(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        g.data_ptr(),
        beta.data_ptr(),
        o_v50.data_ptr(),
        state_tensor.data_ptr(),
        state_tensor.data_ptr(),
        ssm_state_indices.data_ptr(),
        ctypes.c_float(scale),
        ctypes.c_int64(total_tokens),  # N = total_tokens
        ctypes.c_int(H),
        ctypes.c_int(HV),
        ctypes.c_int(K),
        ctypes.c_int(V),
        ctypes.c_int64(stride),
        tma_desc_device.data_ptr(),
        stream
    )
    torch.cuda.synchronize()
    
    # For state comparison
    ht_v50 = state_tensor
    
    # =========================================================================
    # Compare outputs
    # =========================================================================
    
    # Both outputs should have same shape now - squeeze both for comparison
    o_triton_flat = o_triton.squeeze(0) if o_triton.dim() > 3 else o_triton
    o_v50_flat = o_v50.squeeze(0) if o_v50.dim() > 3 else o_v50
    
    # Compare output tensors
    o_diff = (o_v50_flat.float() - o_triton_flat.float()).abs()
    o_max_err = o_diff.max().item()
    o_mean_err = o_diff.mean().item()
    
    # Use sampling for quantile if tensor is too large (>16M elements)
    o_flat = o_diff.flatten()
    if o_flat.numel() > 16_000_000:
        # Sample 1M elements for quantile estimation
        indices = torch.randperm(o_flat.numel(), device=o_flat.device)[:1_000_000]
        o_sample = o_flat[indices]
        o_p99_err = torch.quantile(o_sample, 0.99).item()
        o_p999_err = torch.quantile(o_sample, 0.999).item()
    else:
        o_p99_err = torch.quantile(o_flat, 0.99).item()
        o_p999_err = torch.quantile(o_flat, 0.999).item()
    
    # Compare final states (only the first batch_size states that were updated)
    ht_triton_subset = final_state_triton[:batch_size]
    ht_v50_subset = ht_v50[:batch_size]
    
    ht_diff = (ht_v50_subset.float() - ht_triton_subset.float()).abs()
    ht_max_err = ht_diff.max().item()
    ht_mean_err = ht_diff.mean().item()
    
    # Use sampling for quantile if tensor is too large
    ht_flat = ht_diff.flatten()
    if ht_flat.numel() > 16_000_000:
        indices = torch.randperm(ht_flat.numel(), device=ht_flat.device)[:1_000_000]
        ht_sample = ht_flat[indices]
        ht_p99_err = torch.quantile(ht_sample, 0.99).item()
        ht_p999_err = torch.quantile(ht_sample, 0.999).item()
    else:
        ht_p99_err = torch.quantile(ht_flat, 0.99).item()
        ht_p999_err = torch.quantile(ht_flat, 0.999).item()
    
    # BF16 tolerance - use p99 for pass/fail (more robust to outliers)
    tol_p99 = 0.1   # 99th percentile tolerance
    
    passed = o_p99_err < tol_p99 and ht_p99_err < tol_p99
    
    if verbose:
        print(f"\n  Results for batch_size={batch_size}, TP={tp_size}:")
        print(f"    Output errors:  max={o_max_err:.4f}, p99.9={o_p999_err:.4f}, p99={o_p99_err:.4f}, mean={o_mean_err:.4f}")
        print(f"    State errors:   max={ht_max_err:.4f}, p99.9={ht_p999_err:.4f}, p99={ht_p99_err:.4f}, mean={ht_mean_err:.4f}")
        print(f"    Pass criteria: p99 < {tol_p99}")
        print(f"    Status: {'✓ PASS' if passed else '✗ FAIL'}")
        
        # Show sample comparison
        print(f"\n    Sample comparison (first elements):")
        print(f"      Triton o[0,0,:5]: {o_triton_flat[0, 0, :5].tolist()}")
        print(f"      V50 o[0,0,:5]:    {o_v50_flat[0, 0, :5].tolist()}")
    
    # Compute exact match rates
    exact_match_o = (o_triton_flat == o_v50_flat).float().mean().item() * 100
    exact_match_ht = (ht_triton_subset == ht_v50_subset).float().mean().item() * 100
    
    # Build result before cleanup
    result = {
        'batch_size': batch_size,
        'tp_size': tp_size,
        'o_max_err': o_max_err,
        'o_mean_err': o_mean_err,
        'o_p99_err': o_p99_err,
        'ht_max_err': ht_max_err,
        'ht_mean_err': ht_mean_err,
        'ht_p99_err': ht_p99_err,
        'exact_match_o': exact_match_o,
        'exact_match_ht': exact_match_ht,
        'passed': passed,
    }
    
    # Cleanup to free GPU memory - delete all large tensors
    del q, k, v, g, beta, initial_state, state_tensor, ssm_state_indices, cu_seqlens
    del o_triton, final_state_triton, o_v50, ht_v50
    del o_diff, o_flat, ht_diff, ht_flat, tma_desc_device
    del o_triton_flat, o_v50_flat, ht_triton_subset, ht_v50_subset
    torch.cuda.empty_cache()
    
    return result


def run_all_tests():
    """Run all correctness tests."""
    print("=" * 80)
    print("V50 TMA Kernel (CUTLASS Barriers) Correctness Test vs Triton Reference")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print()
    
    # Test configurations: (batch_size, tp_size)
    test_configs = [
        # =====================================================================
        # TP=4 (H=4, HV=8) - Standard configuration
        # =====================================================================
        (1, 4),      # Single sequence - edge case
        (2, 4),      # Minimal batch
        (3, 4),      # Odd number
        (4, 4),      # Small batch
        (7, 4),      # Prime number
        (8, 4),      # Power of 2
        (15, 4),     # 2^4 - 1
        (16, 4),     # Power of 2
        (17, 4),     # Prime number
        (31, 4),     # Prime number
        (32, 4),     # Power of 2
        (33, 4),     # 2^5 + 1
        (48, 4),     # Multiple of 16
        (63, 4),     # 2^6 - 1
        (64, 4),     # Power of 2
        (96, 4),     # Multiple of 32
        (100, 4),    # Round number
        (127, 4),    # Prime number
        (128, 4),    # Power of 2
        (200, 4),    # Round number
        (255, 4),    # 2^8 - 1
        (256, 4),    # Power of 2
        (384, 4),    # Multiple of 128
        (500, 4),    # Round number
        (512, 4),    # Power of 2
        (768, 4),    # Multiple of 256
        (1000, 4),   # Round number
        (1024, 4),   # Power of 2
        
        # =====================================================================
        # TP=2 (H=8, HV=16) - More heads per GPU
        # =====================================================================
        # Edge cases
        (1, 2),      # Single sequence
        (2, 2),      # Minimal batch
        (3, 2),      # Smallest odd
        # Prime numbers
        (5, 2),      # Prime
        (7, 2),      # Prime
        (11, 2),     # Prime
        (13, 2),     # Prime
        (17, 2),     # Prime
        (19, 2),     # Prime
        (23, 2),     # Prime
        (29, 2),     # Prime
        (31, 2),     # Prime
        (37, 2),     # Prime
        (41, 2),     # Prime
        (43, 2),     # Prime
        (47, 2),     # Prime
        (53, 2),     # Prime
        (59, 2),     # Prime
        (61, 2),     # Prime
        (67, 2),     # Prime
        (71, 2),     # Prime
        (73, 2),     # Prime
        (79, 2),     # Prime
        (83, 2),     # Prime
        (89, 2),     # Prime
        (97, 2),     # Prime
        (101, 2),    # Prime
        (103, 2),    # Prime
        (107, 2),    # Prime
        (109, 2),    # Prime
        (113, 2),    # Prime
        (127, 2),    # Prime (Mersenne)
        (131, 2),    # Prime
        (137, 2),    # Prime
        (139, 2),    # Prime
        (149, 2),    # Prime
        (151, 2),    # Prime
        (157, 2),    # Prime
        (163, 2),    # Prime
        (167, 2),    # Prime
        (173, 2),    # Prime
        (179, 2),    # Prime
        (181, 2),    # Prime
        (191, 2),    # Prime
        (193, 2),    # Prime
        (197, 2),    # Prime
        (199, 2),    # Prime
        (211, 2),    # Prime
        (223, 2),    # Prime
        (227, 2),    # Prime
        (229, 2),    # Prime
        (233, 2),    # Prime
        (239, 2),    # Prime
        (241, 2),    # Prime
        (251, 2),    # Prime
        (257, 2),    # Prime
        # Powers of 2
        (4, 2),      # 2^2
        (8, 2),      # 2^3
        (16, 2),     # 2^4
        (32, 2),     # 2^5
        (64, 2),     # 2^6
        (128, 2),    # 2^7
        (256, 2),    # 2^8
        (512, 2),    # 2^9
        # Edge cases (2^n - 1, 2^n + 1)
        (15, 2),     # 2^4 - 1
        (33, 2),     # 2^5 + 1
        (63, 2),     # 2^6 - 1
        (65, 2),     # 2^6 + 1
        (255, 2),    # 2^8 - 1
        (511, 2),    # 2^9 - 1
        # Round numbers
        (10, 2),     # Round
        (20, 2),     # Round
        (25, 2),     # Round
        (30, 2),     # Round
        (40, 2),     # Round
        (50, 2),     # Round
        (60, 2),     # Round
        (70, 2),     # Round
        (75, 2),     # Round
        (80, 2),     # Round
        (90, 2),     # Round
        (100, 2),    # Round
        (120, 2),    # Round
        (125, 2),    # Round
        (150, 2),    # Round
        (175, 2),    # Round
        (200, 2),    # Round
        (250, 2),    # Round
        (300, 2),    # Round
        (350, 2),    # Round
        (400, 2),    # Round
        (450, 2),    # Round
        (500, 2),    # Round
        # Large batch sizes up to 8K
        (512, 2),    # 2^9
        (600, 2),    # Round
        (700, 2),    # Round
        (750, 2),    # Round
        (768, 2),    # Multiple of 256
        (800, 2),    # Round
        (900, 2),    # Round
        (1000, 2),   # Round
        (1024, 2),   # 2^10
        (1200, 2),   # Round
        (1500, 2),   # Round
        (1536, 2),   # Multiple of 512
        (2000, 2),   # Round
        (2048, 2),   # 2^11
        (2500, 2),   # Round
        (3000, 2),   # Round
        (3072, 2),   # Multiple of 1024
        (3500, 2),   # Round
        (4000, 2),   # Round
        (4096, 2),   # 2^12
        (4500, 2),   # Round
        (5000, 2),   # Round
        (5500, 2),   # Round
        (6000, 2),   # Round
        (6144, 2),   # Multiple of 2048
        (6500, 2),   # Round
        (7000, 2),   # Round
        (7500, 2),   # Round
        (7680, 2),   # Multiple of 2560
        (8000, 2),   # Round
        (8192, 2),   # 2^13 (8K)
        # Large primes
        (509, 2),    # Prime
        (521, 2),    # Prime
        (1009, 2),   # Prime
        (1013, 2),   # Prime
        (2003, 2),   # Prime
        (2017, 2),   # Prime
        (4001, 2),   # Prime
        (4003, 2),   # Prime
        (4007, 2),   # Prime
        (7919, 2),   # Prime
        (7927, 2),   # Prime
        (8191, 2),   # Mersenne prime (2^13-1)
        # Edge cases for large batches
        (1023, 2),   # 2^10 - 1
        (1025, 2),   # 2^10 + 1
        (2047, 2),   # 2^11 - 1
        (2049, 2),   # 2^11 + 1
        (4095, 2),   # 2^12 - 1
        (4097, 2),   # 2^12 + 1
        
        # =====================================================================
        # TP=1 (H=16, HV=32) - Full heads, single GPU
        # =====================================================================
        # Edge cases
        (1, 1),      # Single sequence
        (2, 1),      # Minimal batch
        (3, 1),      # Smallest odd
        # Prime numbers
        (5, 1),      # Prime
        (7, 1),      # Prime
        (11, 1),     # Prime
        (13, 1),     # Prime
        (17, 1),     # Prime
        (19, 1),     # Prime
        (23, 1),     # Prime
        (29, 1),     # Prime
        (31, 1),     # Prime
        (37, 1),     # Prime
        (41, 1),     # Prime
        (43, 1),     # Prime
        (47, 1),     # Prime
        (53, 1),     # Prime
        (59, 1),     # Prime
        (61, 1),     # Prime
        (67, 1),     # Prime
        (71, 1),     # Prime
        (73, 1),     # Prime
        (79, 1),     # Prime
        (83, 1),     # Prime
        (89, 1),     # Prime
        (97, 1),     # Prime
        (101, 1),    # Prime
        (103, 1),    # Prime
        (107, 1),    # Prime
        (109, 1),    # Prime
        (113, 1),    # Prime
        (127, 1),    # Prime (Mersenne)
        (131, 1),    # Prime
        (137, 1),    # Prime
        (139, 1),    # Prime
        (149, 1),    # Prime
        (151, 1),    # Prime
        (157, 1),    # Prime
        (163, 1),    # Prime
        (167, 1),    # Prime
        (173, 1),    # Prime
        (179, 1),    # Prime
        (181, 1),    # Prime
        (191, 1),    # Prime
        (193, 1),    # Prime
        (197, 1),    # Prime
        (199, 1),    # Prime
        (211, 1),    # Prime
        (223, 1),    # Prime
        (227, 1),    # Prime
        (229, 1),    # Prime
        (233, 1),    # Prime
        (239, 1),    # Prime
        (241, 1),    # Prime
        (251, 1),    # Prime
        (257, 1),    # Prime
        # Powers of 2
        (4, 1),      # 2^2
        (8, 1),      # 2^3
        (16, 1),     # 2^4
        (32, 1),     # 2^5
        (64, 1),     # 2^6
        (128, 1),    # 2^7
        (256, 1),    # 2^8
        # Edge cases (2^n - 1, 2^n + 1)
        (15, 1),     # 2^4 - 1
        (33, 1),     # 2^5 + 1
        (63, 1),     # 2^6 - 1
        (65, 1),     # 2^6 + 1
        (127, 1),    # 2^7 - 1
        (129, 1),    # 2^7 + 1
        (255, 1),    # 2^8 - 1
        # Round numbers
        (10, 1),     # Round
        (20, 1),     # Round
        (25, 1),     # Round
        (30, 1),     # Round
        (40, 1),     # Round
        (50, 1),     # Round
        (60, 1),     # Round
        (70, 1),     # Round
        (75, 1),     # Round
        (80, 1),     # Round
        (90, 1),     # Round
        (100, 1),    # Round
        (120, 1),    # Round
        (125, 1),    # Round
        (150, 1),    # Round
        (175, 1),    # Round
        (200, 1),    # Round
        (250, 1),    # Round
        # Large batch sizes up to 8K
        (300, 1),    # Round
        (350, 1),    # Round
        (400, 1),    # Round
        (450, 1),    # Round
        (500, 1),    # Round
        (512, 1),    # 2^9
        (600, 1),    # Round
        (700, 1),    # Round
        (750, 1),    # Round
        (768, 1),    # Multiple of 256
        (800, 1),    # Round
        (900, 1),    # Round
        (1000, 1),   # Round
        (1024, 1),   # 2^10
        (1200, 1),   # Round
        (1500, 1),   # Round
        (1536, 1),   # Multiple of 512
        (2000, 1),   # Round
        (2048, 1),   # 2^11
        (2500, 1),   # Round
        (3000, 1),   # Round
        (3072, 1),   # Multiple of 1024
        (3500, 1),   # Round
        (4000, 1),   # Round
        (4096, 1),   # 2^12
        (4500, 1),   # Round
        (5000, 1),   # Round
        (5500, 1),   # Round
        (6000, 1),   # Round
        (6144, 1),   # Multiple of 2048
        (6500, 1),   # Round
        (7000, 1),   # Round
        (7500, 1),   # Round
        (7680, 1),   # Multiple of 2560
        (8000, 1),   # Round
        (8192, 1),   # 2^13 (8K)
        # Large primes
        (509, 1),    # Prime
        (521, 1),    # Prime
        (1009, 1),   # Prime
        (1013, 1),   # Prime
        (2003, 1),   # Prime
        (2017, 1),   # Prime
        (4001, 1),   # Prime
        (4003, 1),   # Prime
        (4007, 1),   # Prime
        (7919, 1),   # Prime
        (7927, 1),   # Prime
        (8191, 1),   # Mersenne prime (2^13-1)
        # Edge cases for large batches
        (511, 1),    # 2^9 - 1
        (513, 1),    # 2^9 + 1
        (1023, 1),   # 2^10 - 1
        (1025, 1),   # 2^10 + 1
        (2047, 1),   # 2^11 - 1
        (2049, 1),   # 2^11 + 1
        (4095, 1),   # 2^12 - 1
        (4097, 1),   # 2^12 + 1
        
        # =====================================================================
        # TP=8 (H=2, HV=4) - Minimal heads per GPU
        # =====================================================================
        (1, 8),      # Single sequence, minimal heads
        (4, 8),      # Small batch
        (8, 8),      # Small batch
        (16, 8),     # Medium-small batch
        (32, 8),     # Medium batch
        (64, 8),     # Medium batch
        (128, 8),    # Larger batch
        (256, 8),    # Large batch
        
        # =====================================================================
        # TP=16 (H=1, HV=2) - Single key head per GPU
        # =====================================================================
        (1, 16),     # Single sequence, single key head
        (4, 16),     # Small batch
        (16, 16),    # Medium batch
        (64, 16),    # Medium batch
        (128, 16),   # Larger batch
    ]
    
    results = []
    total_passed = 0
    total_skipped = 0
    
    # Triton kernel uses grid.z = N * HV, CUDA max grid.z = 65535
    # So max_batch = 65535 // HV
    CUDA_MAX_GRID_Z = 65535
    total_num_value_heads = 32
    
    # Filter out configs that exceed Triton's grid limit
    valid_configs = []
    for batch_size, tp_size in test_configs:
        HV = total_num_value_heads // tp_size
        max_batch_for_triton = CUDA_MAX_GRID_Z // HV
        if batch_size > max_batch_for_triton:
            print(f"  ⊘ bs={batch_size:>4}, tp={tp_size} | SKIP: exceeds Triton grid.z limit (max={max_batch_for_triton})")
            total_skipped += 1
        else:
            valid_configs.append((batch_size, tp_size))
    
    total_tests = len(valid_configs)
    
    print(f"Running {total_tests} test configurations ({total_skipped} skipped due to Triton grid limits)...")
    print("-" * 80)
    
    for batch_size, tp_size in valid_configs:
        try:
            # Use compact output for large test suites
            result = test_v50_vs_triton(batch_size=batch_size, tp_size=tp_size, verbose=False)
            results.append(result)
            status = "✓" if result['passed'] else "✗"
            match_rate = f"{(result.get('exact_match_o', 0)):.0f}%" if 'exact_match_o' in result else "N/A"
            print(f"  {status} bs={batch_size:>4}, tp={tp_size} | O_max={result['o_max_err']:.4f}, Ht_max={result['ht_max_err']:.4f}")
            if result['passed']:
                total_passed += 1
        except Exception as e:
            print(f"  ✗ bs={batch_size:>4}, tp={tp_size} | ERROR: {e}")
            results.append({
                'batch_size': batch_size,
                'tp_size': tp_size,
                'error': str(e),
                'passed': False,
            })
            # Clear cache after OOM or other errors
            torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Config':>20} | {'O p99':>10} | {'O max':>10} | {'Ht p99':>10} | {'Ht max':>10} | {'Status':>8}")
    print("-" * 80)
    
    for r in results:
        config = f"bs={r['batch_size']}, tp={r['tp_size']}"
        if 'error' in r:
            status = "ERROR"
            o_p99 = "N/A"
            o_max = "N/A"
            ht_p99 = "N/A"
            ht_max = "N/A"
        else:
            status = "✓ PASS" if r['passed'] else "✗ FAIL"
            o_p99 = f"{r['o_p99_err']:.4f}"
            o_max = f"{r['o_max_err']:.2f}"
            ht_p99 = f"{r['ht_p99_err']:.4f}"
            ht_max = f"{r['ht_max_err']:.2f}"
        print(f"{config:>20} | {o_p99:>10} | {o_max:>10} | {ht_p99:>10} | {ht_max:>10} | {status:>8}")
    
    print("-" * 60)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed ({total_skipped} skipped due to Triton limits)")
    
    if total_passed == total_tests:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print(f"\n✗ {total_tests - total_passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
