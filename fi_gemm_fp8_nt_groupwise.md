# FlashInfer `gemm_fp8_nt_groupwise` API Documentation Bug

## Executive Summary

FlashInfer's `gemm_fp8_nt_groupwise` function has **incorrect documentation** regarding the expected memory layout of scale tensors. The documentation claims scales should be "column-major" but the implementation actually expects **row-major** scales.

This discrepancy caused **zero accuracy** in production model inference when vLLM attempted to use FlashInfer following the documented API.

---

## The Bug

### What the Documentation Says

From FlashInfer's API documentation ([gemm_base.py:2212-2223](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gemm/gemm_base.py#L2212-L2223)):

```python
def gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    ...
):
    """
    Parameters
    ----------
    a: torch.Tensor
        Row-major input tensor shape (m, k), fp8 e4m3 or fp8 e5m2.

    b: torch.Tensor
        Column-major input tensor shape (n, k), fp8 e4m3 or fp8 e5m2.

    a_scale: torch.Tensor
        if the backend is ``cutlass``:
            Column-major scale tensor for a, shape ``(m, k // block_size)`` if scale_major_mode is ``K``
            or shape ``(k // block_size, m)`` if scale_major_mode is ``MN``
    
    b_scale: torch.Tensor
        if the backend is ``cutlass``:
            Row-major scale tensor for b, shape ``(n // block_size, k // block_size)`` if scale_major_k is ``K``
            or shape ``(k // block_size, n // block_size)`` if scale_major_mode is ``MN``
    """
```

**Documentation claims for `scale_major_mode="MN"` with CUTLASS backend:**
- `a_scale`: **Column-major** tensor of shape `(k // block_size, m)`
- `b`: **Column-major** tensor of shape `(n, k)`

### What Actually Works

From FlashInfer's own test suite ([test_groupwise_scaled_gemm_fp8.py:121-140](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/gemm/test_groupwise_scaled_gemm_fp8.py#L121-L140)):

```python
a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)

# For cutlass backend, scales are used directly:
c = gemm_fp8_nt_groupwise(
    a_fp8,
    b_fp8,    # ← Passed directly (row-major)
    a_scale,  # ← Passed directly (row-major from quantize_fp8)
    b_scale,
    scale_major_mode,
    out_dtype=out_dtype,
    backend="cutlass",
)
```

**What `quantize_fp8()` actually returns:**
```python
a_scale: shape=[4, 32], stride=(32, 1)  ← ROW-MAJOR (stride[0] = 32 > stride[1] = 1)
b_fp8: shape=[256, 512], stride=(512, 1)  ← ROW-MAJOR
```

---

## Reproduction

### Environment
- **GPU**: NVIDIA B200 (SM100 / Blackwell)
- **FlashInfer version**: Latest from main branch
- **PyTorch**: 2.5.1+cu124

### Test Results

Run the reproduction script:
```bash
python test_fi_gemm_fp8_nt_groupwise.py
```

**Expected Output:**
```
TEST 1: Using COLUMN-MAJOR scales (as documentation says)
  ❌ Mean relative error: ~750%
  ❌ RESULT: FAILS

TEST 2: Using ROW-MAJOR scales (what actually works)  
  ✅ Mean relative error: ~2%
  ✅ RESULT: PASSES

CONCLUSION: FlashInfer's documentation is INCORRECT!
```

### Detailed Comparison

| Aspect | Documentation Claims | Actual Behavior |
|--------|---------------------|-----------------|
| `a_scale` layout | Column-major | **Row-major** |
| `a_scale` stride for shape `[4, 32]` | `(1, 4)` | `(32, 1)` |
| `b` tensor layout | Column-major | **Row-major** |
| `b` stride for shape `[256, 512]` | `(1, 256)` | `(512, 1)` |
| What `quantize_fp8()` returns | Not specified | Row-major tensors |
| Using documented format | Should work | **Fails with ~750% error** |
| Using actual format | Not documented | **Works with ~2% error** |

---

## Impact

### Production Impact at vLLM

When vLLM implemented FlashInfer support following the documentation:
1. Converted tensors to "column-major" as documented
2. Tests passed because they compared backends to each other (both wrong)
3. Production model inference showed **zero accuracy** in lm_eval
4. Issue was only discovered through ground-truth validation

### Root Cause

The documentation incorrectly describes the memory layout, causing:
- **Wrong tensor stride conversions** in user code
- **Data reordering that scrambles scale values**
- **Catastrophic numerical errors** (~750% instead of ~2%)

---

## Evidence from FlashInfer Source

### 1. `quantize_fp8()` Returns Row-Major

From `flashinfer/testing/utils.py:66-161`:

```python
def quantize_fp8(x, scale_shape, tile_shape, scale_major_mode):
    # ... quantization logic ...
    return x_fp8, x_scale  # x_scale is row-major!
```

When called with `scale_shape=(4, 32)`, it returns stride `(32, 1)` which is **row-major**.

### 2. Official Tests Use Scales Directly

From `tests/gemm/test_groupwise_scaled_gemm_fp8.py:121-140`:

```python
a_fp8, a_scale = quantize_fp8(...)  # Returns row-major
c = gemm_fp8_nt_groupwise(
    a_fp8,
    b_fp8,
    a_scale,  # Used directly without conversion!
    b_scale,
    ...
)
```

No conversion to column-major is performed, proving row-major is expected.

### 3. Only TRTLLM Backend Has Special Handling

From `gemm_base.py:2333-2347`:

```python
elif backend == "trtllm":
    # TRTLLM expects different format - documented correctly
    ...
```

The CUTLASS backend (default for SM100) has no special scale conversion, it uses them as-is (row-major).

---

## Recommended Fix

### For FlashInfer Maintainers

**Option 1: Fix the documentation** (Recommended)

Update `gemm_base.py` lines 2213-2223:

```python
a_scale: torch.Tensor
    if the backend is ``cutlass``:
        Row-major scale tensor for a, shape ``(m, k // block_size)`` if scale_major_mode is ``K``
        or shape ``(k // block_size, m)`` if scale_major_mode is ``MN``
        
        Note: Despite being called "column-major" in scale_major_mode="MN", 
        the scale tensor itself should be ROW-MAJOR with stride (m, 1).
```

**Option 2: Fix the implementation**

Change the CUTLASS kernel to actually accept column-major scales as documented (breaking change).

### For Users (Workaround)

Until documentation is fixed, users should:

1. **Use row-major scales:**
   ```python
   a_scale = a_scale.T.contiguous()  # Ensure row-major stride (M, 1)
   ```

2. **Keep `b` tensor as row-major:**
   ```python
   # Don't convert to column-major despite docs saying so
   gemm_fp8_nt_groupwise(a, b, ...)  # b stays row-major
   ```

3. **Validate against ground truth:**
   Don't just compare backends - validate against dequantized reference computation.

---

## Technical Details

### Memory Layout Primer

For a tensor with shape `[A, B]`:
- **Row-major**: stride `(B, 1)` - elements in same row are contiguous
- **Column-major**: stride `(1, A)` - elements in same column are contiguous

### The Confusion

The term "scale_major_mode='MN'" refers to **how blocks are organized**, not the memory layout of scale tensors themselves.

- `scale_major_mode="MN"`: Blocks prioritize M and N dimensions (shape: `[k//block, ...]`)
- `scale_major_mode="K"`: Blocks prioritize K dimension (shape: `[..., k//block]`)

The scale tensors themselves are **always row-major** regardless of scale_major_mode!

---

## References

- FlashInfer repository: https://github.com/flashinfer-ai/flashinfer
- Documentation: https://docs.flashinfer.ai/api/gemm.html#flashinfer.gemm.gemm_fp8_nt_groupwise
- vLLM issue: Zero accuracy with FlashInfer blockwise FP8
- Reproduction: `test_fi_gemm_fp8_nt_groupwise.py`

---

## Contact

If you encounter this issue, please:
1. Run `test_fi_gemm_fp8_nt_groupwise.py` to verify the bug
2. Report to FlashInfer: https://github.com/flashinfer-ai/flashinfer/issues
3. Reference this document in your bug report

**Date discovered**: November 24, 2025  
**Affected versions**: FlashInfer main branch (as of Nov 2025)  
**Platform**: NVIDIA Blackwell (SM100+) with CUTLASS backend

