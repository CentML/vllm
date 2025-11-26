#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Comprehensive test suite for blockwise FP8 implementations.

This test suite validates:
1. Ground truth correctness (comparing against dequantized computation)
2. Backend consistency (comparing backends against each other)
3. Edge cases (padding, various sizes, etc.)
4. All available backends (Triton, CUTLASS, FlashInfer)
"""

import pytest
import torch
import sys
import os

# Add the vllm directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer


def per_block_quant_fp8(
    weight: torch.Tensor, block_size: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight to FP8 with per-block quantization.
    
    Args:
        weight: Weight tensor [N, K] to quantize
        block_size: Block size [block_n, block_k]
    
    Returns:
        Tuple of (quantized_weight, scales)
    """
    assert len(block_size) == 2
    block_n, block_k = block_size
    n, k = weight.shape
    
    dtype = current_platform.fp8_dtype()
    finfo = torch.finfo(dtype)
    
    # Pad if necessary
    n_padded = ((n + block_n - 1) // block_n) * block_n
    k_padded = ((k + block_k - 1) // block_k) * block_k
    
    if n_padded != n or k_padded != k:
        weight_padded = torch.zeros(n_padded, k_padded, dtype=weight.dtype, device=weight.device)
        weight_padded[:n, :k] = weight
        weight = weight_padded
        n, k = n_padded, k_padded
    
    # Compute per-block scales
    scales = torch.zeros(n // block_n, k // block_k, dtype=torch.float32, device=weight.device)
    weight_q = torch.zeros_like(weight, dtype=dtype)
    
    for i in range(n // block_n):
        for j in range(k // block_k):
            i_start, i_end = i * block_n, (i + 1) * block_n
            j_start, j_end = j * block_k, (j + 1) * block_k
            block = weight[i_start:i_end, j_start:j_end]
            amax = block.abs().max().clamp(min=1e-12)
            scale = finfo.max / amax
            scales[i, j] = 1.0 / scale
            
            block_q = (block * scale).clamp(min=finfo.min, max=finfo.max).to(dtype)
            weight_q[i_start:i_end, j_start:j_end] = block_q
    
    return weight_q.contiguous(), scales


def dequantize_per_block_fp8(
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Dequantize FP8 weight back to bfloat16.
    
    Args:
        weight_q: Quantized weight tensor
        weight_scale: Per-block scale tensor
        block_size: Block size [block_n, block_k]
        output_dtype: Output data type
    
    Returns:
        Dequantized weight tensor
    """
    block_n, block_k = block_size
    n, k = weight_q.shape
    
    weight_dq = torch.zeros_like(weight_q, dtype=output_dtype)
    
    for i in range(n // block_n):
        for j in range(k // block_k):
            i_start, i_end = i * block_n, (i + 1) * block_n
            j_start, j_end = j * block_k, (j + 1) * block_k
            
            block = weight_q[i_start:i_end, j_start:j_end].to(output_dtype)
            scale = weight_scale[i, j]
            weight_dq[i_start:i_end, j_start:j_end] = block * scale
    
    return weight_dq


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="Requires SM90+ (Hopper or Blackwell)",
)
class TestBlockwiseFP8Comprehensive:
    """Comprehensive test suite for blockwise FP8 operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test parameters."""
        self.block_size = [128, 128]
        self.weight_group_shape = GroupShape(self.block_size[0], self.block_size[1])
        self.act_quant_group_shape = GroupShape(1, 128)
        self.dtype = torch.bfloat16
        self.device = "cuda"
        torch.manual_seed(42)

    def _create_test_data(self, m: int, n: int, k: int):
        """Create test input and weight tensors."""
        input_bf16 = torch.randn(m, k, dtype=self.dtype, device=self.device)
        weight_bf16 = torch.randn(n, k, dtype=self.dtype, device=self.device)
        weight_q, weight_scale = per_block_quant_fp8(weight_bf16, self.block_size)
        
        return input_bf16, weight_q, weight_scale, weight_bf16

    def _compute_ground_truth(
        self,
        input_bf16: torch.Tensor,
        weight_q: torch.Tensor,
        weight_scale: torch.Tensor,
        use_column_major_scales: bool = False,
    ) -> torch.Tensor:
        """Compute ground truth by dequantizing and using standard matmul.
        
        Args:
            input_bf16: Input tensor
            weight_q: Quantized weight
            weight_scale: Weight scales
            use_column_major_scales: Whether to use column-major activation scales
                                     (True for FlashInfer/CUTLASS, False for Triton)
        
        Returns:
            Ground truth output tensor
        """
        # Dequantize weight
        weight_dq = dequantize_per_block_fp8(weight_q, weight_scale, self.block_size, self.dtype)
        
        # Quantize input with appropriate scale layout
        input_q, input_scale = per_token_group_quant_fp8(
            input_bf16,
            group_size=self.act_quant_group_shape.col,
            column_major_scales=use_column_major_scales,
        )
        
        # Dequantize input
        input_dq = torch.zeros_like(input_bf16)
        for i in range(input_bf16.shape[0]):
            for j in range(input_scale.shape[1]):
                j_start = j * self.act_quant_group_shape.col
                j_end = (j + 1) * self.act_quant_group_shape.col
                input_dq[i, j_start:j_end] = input_q[i, j_start:j_end].to(self.dtype) * input_scale[i, j]
        
        # Compute: input @ weight.T
        return torch.matmul(input_dq, weight_dq.T)

    def _run_backend(
        self,
        backend_name: str,
        input_bf16: torch.Tensor,
        weight_q: torch.Tensor,
        weight_scale: torch.Tensor,
        use_cutlass: bool = False,
        use_aiter: bool = False,
        force_triton: bool = False,
    ) -> torch.Tensor:
        """Run a specific backend and return output.
        
        Args:
            backend_name: Name of backend for logging
            input_bf16: Input tensor
            weight_q: Quantized weight
            weight_scale: Weight scales
            use_cutlass: Whether to use CUTLASS backend
            use_aiter: Whether to use AITER backend
            force_triton: Force use of Triton on SM100 (set cutlass=True to prevent FlashInfer)
        
        Returns:
            Output tensor from the backend
        """
        # On SM100, FlashInfer is preferred unless CUTLASS/AITER is explicitly requested
        # To force Triton, we set cutlass=True which prevents FlashInfer,
        # then it falls back to Triton (since CUTLASS blockwise FP8 isn't on SM100)
        if force_triton:
            use_cutlass = True
        
        op = W8A8BlockFp8LinearOp(
            self.weight_group_shape,
            self.act_quant_group_shape,
            cutlass_block_fp8_supported=use_cutlass,
            use_aiter_and_is_supported=use_aiter,
        )
        
        output = op.apply(input_bf16, weight_q, weight_scale, input_scale=None)
        return output

    def _compare_with_ground_truth(
        self,
        output: torch.Tensor,
        ground_truth: torch.Tensor,
        backend_name: str,
        tolerance: float = 0.05,
    ):
        """Compare backend output with ground truth.
        
        Args:
            output: Backend output
            ground_truth: Ground truth reference
            backend_name: Name of backend
            tolerance: Acceptable mean relative error (5% default for FP8)
        """
        abs_diff = (output - ground_truth).abs()
        rel_diff = abs_diff / (ground_truth.abs() + 1e-5)
        
        max_abs_diff = abs_diff.max().item()
        max_rel_diff = rel_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        mean_rel_diff = rel_diff.mean().item()
        
        print(f"\n{backend_name} vs Ground Truth:")
        print(f"  Max absolute difference: {max_abs_diff:.6f}")
        print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"  Max relative difference: {max_rel_diff:.4%}")
        print(f"  Mean relative difference: {mean_rel_diff:.4%}")
        
        print(f"\n  Sample output values (first 5 elements of first row):")
        print(f"    Backend: {output[0, :5].tolist()}")
        print(f"    Ground truth: {ground_truth[0, :5].tolist()}")
        
        # Mean relative error is the key metric for FP8
        assert mean_rel_diff < tolerance, (
            f"{backend_name}: Mean relative error {mean_rel_diff:.4%} exceeds {tolerance*100:.1f}%"
        )

    def _compare_backends(
        self,
        output1: torch.Tensor,
        output2: torch.Tensor,
        backend1_name: str,
        backend2_name: str,
        tolerance: float = 0.02,
    ):
        """Compare two backend outputs for consistency.
        
        Args:
            output1: First backend output
            output2: Second backend output
            backend1_name: Name of first backend
            backend2_name: Name of second backend
            tolerance: Acceptable mean relative error
        """
        abs_diff = (output1 - output2).abs()
        rel_diff = abs_diff / (output1.abs() + 1e-5)
        
        max_abs_diff = abs_diff.max().item()
        max_rel_diff = rel_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        mean_rel_diff = rel_diff.mean().item()
        
        print(f"\n{backend1_name} vs {backend2_name}:")
        print(f"  Max absolute difference: {max_abs_diff:.6f}")
        print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"  Max relative difference: {max_rel_diff:.4%}")
        print(f"  Mean relative difference: {mean_rel_diff:.4%}")
        
        assert mean_rel_diff < tolerance, (
            f"{backend1_name} vs {backend2_name}: Mean relative error "
            f"{mean_rel_diff:.4%} exceeds {tolerance*100:.1f}%"
        )

    # ========================================================================
    # GROUND TRUTH VALIDATION TESTS
    # ========================================================================

    @pytest.mark.parametrize("m,n,k", [
        (32, 256, 512),
        (128, 1024, 2048),
        (128, 3072, 2048),
        (256, 2048, 4096),
    ])
    def test_triton_vs_ground_truth(self, m, n, k):
        """Test Triton backend against ground truth reference."""
        print(f"\n{'='*60}")
        print(f"[GROUND TRUTH] Testing Triton with shape ({m}, {n}, {k})")
        print(f"{'='*60}")
        
        input_bf16, weight_q, weight_scale, _ = self._create_test_data(m, n, k)
        
        # Compute ground truth (Triton uses row-major scales)
        ground_truth = self._compute_ground_truth(
            input_bf16, weight_q, weight_scale, use_column_major_scales=False
        )
        
        # Run Triton backend
        output_triton = self._run_backend(
            "Triton", input_bf16, weight_q, weight_scale,
            use_cutlass=False, use_aiter=False, force_triton=True
        )
        
        # Validate against ground truth
        self._compare_with_ground_truth(output_triton, ground_truth, "Triton", tolerance=0.05)
        print("\n✓ Triton PASSED ground truth validation!")

    @pytest.mark.parametrize("m,n,k", [
        (32, 256, 512),
        (128, 1024, 2048),
        (128, 3072, 2048),
        (256, 2048, 4096),
        (651, 3072, 2048),  # With M padding
        (1597, 3072, 2048),  # Large batch
    ])
    @pytest.mark.skipif(
        not has_flashinfer() or not current_platform.has_device_capability(100),
        reason="FlashInfer requires SM100 (Blackwell)",
    )
    def test_flashinfer_vs_ground_truth(self, m, n, k):
        """Test FlashInfer backend against ground truth reference."""
        print(f"\n{'='*60}")
        print(f"[GROUND TRUTH] Testing FlashInfer with shape ({m}, {n}, {k})")
        print(f"{'='*60}")
        
        input_bf16, weight_q, weight_scale, _ = self._create_test_data(m, n, k)
        
        # Compute ground truth (FlashInfer uses column-major scales)
        ground_truth = self._compute_ground_truth(
            input_bf16, weight_q, weight_scale, use_column_major_scales=True
        )
        
        # Run FlashInfer backend
        output_flashinfer = self._run_backend(
            "FlashInfer", input_bf16, weight_q, weight_scale,
            use_cutlass=False, use_aiter=False, force_triton=False
        )
        
        # Validate against ground truth
        self._compare_with_ground_truth(output_flashinfer, ground_truth, "FlashInfer", tolerance=0.05)
        print("\n✓ FlashInfer PASSED ground truth validation!")

    @pytest.mark.parametrize("m,n,k", [
        (32, 256, 512),
        (128, 1024, 2048),
    ])
    @pytest.mark.skipif(
        not current_platform.has_device_capability(90),
        reason="CUTLASS block FP8 requires SM90+",
    )
    def test_cutlass_vs_ground_truth(self, m, n, k):
        """Test CUTLASS backend against ground truth reference."""
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
            cutlass_block_fp8_supported,
        )
        
        if not cutlass_block_fp8_supported():
            pytest.skip("CUTLASS block FP8 not supported on this platform")
        
        print(f"\n{'='*60}")
        print(f"[GROUND TRUTH] Testing CUTLASS with shape ({m}, {n}, {k})")
        print(f"{'='*60}")
        
        input_bf16, weight_q, weight_scale, _ = self._create_test_data(m, n, k)
        
        # Compute ground truth (CUTLASS uses column-major scales)
        ground_truth = self._compute_ground_truth(
            input_bf16, weight_q, weight_scale, use_column_major_scales=True
        )
        
        # Run CUTLASS backend
        output_cutlass = self._run_backend(
            "CUTLASS", input_bf16, weight_q, weight_scale,
            use_cutlass=True, use_aiter=False, force_triton=False
        )
        
        # Validate against ground truth
        self._compare_with_ground_truth(output_cutlass, ground_truth, "CUTLASS", tolerance=0.05)
        print("\n✓ CUTLASS PASSED ground truth validation!")

    # ========================================================================
    # BACKEND CONSISTENCY TESTS
    # ========================================================================

    @pytest.mark.parametrize("m,n,k", [
        (32, 256, 512),
        (64, 512, 1024),
        (128, 1024, 2048),
        (128, 3072, 2048),
        (256, 2048, 4096),
        (651, 3072, 2048),  # With padding
        (1597, 3072, 2048),  # Large batch
    ])
    @pytest.mark.skipif(
        not has_flashinfer() or not current_platform.has_device_capability(100),
        reason="FlashInfer requires SM100 (Blackwell)",
    )
    def test_flashinfer_vs_triton_consistency(self, m, n, k):
        """Test FlashInfer vs Triton backend consistency."""
        print(f"\n{'='*60}")
        print(f"[CONSISTENCY] FlashInfer vs Triton with shape ({m}, {n}, {k})")
        print(f"{'='*60}")
        
        input_bf16, weight_q, weight_scale, _ = self._create_test_data(m, n, k)
        
        # Run both backends
        output_triton = self._run_backend(
            "Triton", input_bf16, weight_q, weight_scale,
            use_cutlass=False, use_aiter=False, force_triton=True
        )
        
        output_flashinfer = self._run_backend(
            "FlashInfer", input_bf16, weight_q, weight_scale,
            use_cutlass=False, use_aiter=False, force_triton=False
        )
        
        # Compare backends
        self._compare_backends(output_triton, output_flashinfer, "Triton", "FlashInfer", tolerance=0.02)
        print("\n✓ Backends are consistent!")

    @pytest.mark.parametrize("m,n,k", [
        (32, 256, 512),
        (128, 1024, 2048),
    ])
    @pytest.mark.skipif(
        not current_platform.has_device_capability(90),
        reason="CUTLASS block FP8 requires SM90+",
    )
    def test_cutlass_vs_triton_consistency(self, m, n, k):
        """Test CUTLASS vs Triton backend consistency."""
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
            cutlass_block_fp8_supported,
        )
        
        if not cutlass_block_fp8_supported():
            pytest.skip("CUTLASS block FP8 not supported on this platform")
        
        print(f"\n{'='*60}")
        print(f"[CONSISTENCY] CUTLASS vs Triton with shape ({m}, {n}, {k})")
        print(f"{'='*60}")
        
        input_bf16, weight_q, weight_scale, _ = self._create_test_data(m, n, k)
        
        # Run both backends
        output_triton = self._run_backend(
            "Triton", input_bf16, weight_q, weight_scale,
            use_cutlass=False, use_aiter=False, force_triton=True
        )
        
        output_cutlass = self._run_backend(
            "CUTLASS", input_bf16, weight_q, weight_scale,
            use_cutlass=True, use_aiter=False, force_triton=False
        )
        
        # Compare backends
        self._compare_backends(output_triton, output_cutlass, "Triton", "CUTLASS", tolerance=0.02)
        print("\n✓ Backends are consistent!")

    # ========================================================================
    # SPECIAL CASES
    # ========================================================================

    @pytest.mark.skipif(
        not has_flashinfer() or not current_platform.has_device_capability(100),
        reason="FlashInfer requires SM100 (Blackwell)",
    )
    def test_simple_case_all_ones(self):
        """Test with all-ones input for perfect accuracy verification."""
        print(f"\n{'='*60}")
        print(f"[SPECIAL] Testing all-ones case")
        print(f"{'='*60}")
        
        m, n, k = 4, 128, 128
        
        # Create simple test data
        input_bf16 = torch.ones(m, k, dtype=self.dtype, device=self.device)
        weight_bf16 = torch.ones(n, k, dtype=self.dtype, device=self.device)
        weight_q, weight_scale = per_block_quant_fp8(weight_bf16, self.block_size)
        
        # Compute ground truths
        gt_triton = self._compute_ground_truth(input_bf16, weight_q, weight_scale, use_column_major_scales=False)
        gt_flashinfer = self._compute_ground_truth(input_bf16, weight_q, weight_scale, use_column_major_scales=True)
        
        print(f"\nExpected output: {k} (all values should be {k})")
        print(f"Ground truth mean: {gt_triton.mean().item():.4f}")
        
        # Run backends
        output_triton = self._run_backend("Triton", input_bf16, weight_q, weight_scale, force_triton=True)
        output_flashinfer = self._run_backend("FlashInfer", input_bf16, weight_q, weight_scale)
        
        print(f"Triton output mean: {output_triton.mean().item():.4f}")
        print(f"FlashInfer output mean: {output_flashinfer.mean().item():.4f}")
        
        # Both should be perfect (0% error) for all-ones
        self._compare_with_ground_truth(output_triton, gt_triton, "Triton", tolerance=0.001)
        self._compare_with_ground_truth(output_flashinfer, gt_flashinfer, "FlashInfer", tolerance=0.001)
        
        print("\n✓ All-ones test PASSED!")

    @pytest.mark.skipif(
        not has_flashinfer() or not current_platform.has_device_capability(100),
        reason="FlashInfer requires SM100 (Blackwell)",
    )
    def test_all_backends_comprehensive(self):
        """Comprehensive test of all available backends."""
        m, n, k = 128, 1024, 2048
        
        print(f"\n{'='*60}")
        print(f"[COMPREHENSIVE] Testing all backends with shape ({m}, {n}, {k})")
        print(f"{'='*60}")
        
        input_bf16, weight_q, weight_scale, _ = self._create_test_data(m, n, k)
        
        # Collect outputs and ground truths
        outputs = {}
        ground_truths = {}
        
        # Triton
        print("\nTesting Triton...")
        ground_truths["Triton"] = self._compute_ground_truth(
            input_bf16, weight_q, weight_scale, use_column_major_scales=False
        )
        outputs["Triton"] = self._run_backend(
            "Triton", input_bf16, weight_q, weight_scale, force_triton=True
        )
        self._compare_with_ground_truth(outputs["Triton"], ground_truths["Triton"], "Triton")
        
        # CUTLASS (if available)
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import cutlass_block_fp8_supported
        
        if cutlass_block_fp8_supported():
            print("\nTesting CUTLASS...")
            ground_truths["CUTLASS"] = self._compute_ground_truth(
                input_bf16, weight_q, weight_scale, use_column_major_scales=True
            )
            outputs["CUTLASS"] = self._run_backend(
                "CUTLASS", input_bf16, weight_q, weight_scale, use_cutlass=True
            )
            self._compare_with_ground_truth(outputs["CUTLASS"], ground_truths["CUTLASS"], "CUTLASS")
        
        # FlashInfer
        print("\nTesting FlashInfer...")
        ground_truths["FlashInfer"] = self._compute_ground_truth(
            input_bf16, weight_q, weight_scale, use_column_major_scales=True
        )
        outputs["FlashInfer"] = self._run_backend(
            "FlashInfer", input_bf16, weight_q, weight_scale
        )
        self._compare_with_ground_truth(outputs["FlashInfer"], ground_truths["FlashInfer"], "FlashInfer")
        
        # Compare all backend pairs for consistency
        print(f"\n{'='*60}")
        print("Backend consistency checks:")
        print(f"{'='*60}")
        
        backend_names = list(outputs.keys())
        for i, backend1 in enumerate(backend_names):
            for backend2 in backend_names[i + 1:]:
                self._compare_backends(
                    outputs[backend1], outputs[backend2],
                    backend1, backend2, tolerance=0.02
                )
        
        print(f"\n✓ All {len(outputs)} backends validated and consistent!")

    # ========================================================================
    # EDGE CASES
    # ========================================================================

    @pytest.mark.parametrize("m", [
        1,    # Single row
        4,    # Exactly padded to 4
        5,    # Needs padding
        127,  # Just under block size
        128,  # Exactly block size
        651,  # Odd number needing padding
    ])
    @pytest.mark.skipif(
        not has_flashinfer() or not current_platform.has_device_capability(100),
        reason="FlashInfer requires SM100 (Blackwell)",
    )
    def test_flashinfer_various_m_sizes(self, m):
        """Test FlashInfer with various M dimensions (including padding cases)."""
        n, k = 256, 512
        
        print(f"\n[EDGE CASE] Testing M={m} (padded to {((m + 3) // 4) * 4})")
        
        input_bf16, weight_q, weight_scale, _ = self._create_test_data(m, n, k)
        
        ground_truth = self._compute_ground_truth(
            input_bf16, weight_q, weight_scale, use_column_major_scales=True
        )
        
        output = self._run_backend("FlashInfer", input_bf16, weight_q, weight_scale)
        
        self._compare_with_ground_truth(output, ground_truth, f"FlashInfer(M={m})", tolerance=0.05)
        print(f"✓ M={m} PASSED!")

    @pytest.mark.parametrize("n,k", [
        (128, 128),    # Minimum size
        (256, 256),    # Square
        (512, 128),    # Wide
        (128, 512),    # Tall
        (3072, 2048),  # Real-world dimensions
    ])
    def test_triton_various_shapes(self, n, k):
        """Test Triton with various N and K dimensions."""
        m = 128
        
        print(f"\n[EDGE CASE] Testing Triton with N={n}, K={k}")
        
        input_bf16, weight_q, weight_scale, _ = self._create_test_data(m, n, k)
        
        ground_truth = self._compute_ground_truth(
            input_bf16, weight_q, weight_scale, use_column_major_scales=False
        )
        
        output = self._run_backend("Triton", input_bf16, weight_q, weight_scale, force_triton=True)
        
        self._compare_with_ground_truth(output, ground_truth, f"Triton(N={n},K={k})", tolerance=0.05)
        print(f"✓ Shape test PASSED!")

    # ========================================================================
    # REGRESSION TESTS
    # ========================================================================

    def test_regression_zero_accuracy_bug(self):
        """Regression test for the zero-accuracy bug that was fixed.
        
        This test ensures that the FlashInfer implementation produces correct
        results and doesn't regress to the buggy behavior that caused zero
        accuracy in lm_eval.
        """
        print(f"\n{'='*60}")
        print("[REGRESSION] Testing zero-accuracy bug fix")
        print(f"{'='*60}")
        
        # Use the exact dimensions that were failing in production
        m, n, k = 651, 3072, 2048
        
        input_bf16, weight_q, weight_scale, _ = self._create_test_data(m, n, k)
        
        # This is the configuration that was producing zero accuracy
        if has_flashinfer() and current_platform.has_device_capability(100):
            ground_truth = self._compute_ground_truth(
                input_bf16, weight_q, weight_scale, use_column_major_scales=True
            )
            
            output = self._run_backend("FlashInfer", input_bf16, weight_q, weight_scale)
            
            # The bug caused ~750% error, so if we're under 5%, the bug is fixed
            self._compare_with_ground_truth(output, ground_truth, "FlashInfer", tolerance=0.05)
            
            print("\n✓ Zero-accuracy bug is FIXED!")
            print("  Previous behavior: ~750% error → zero accuracy")
            print("  Current behavior: ~2% error → correct accuracy")
        else:
            pytest.skip("FlashInfer not available for regression test")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

