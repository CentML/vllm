# mnnvl `allreduce_fusion` with `kARResidualRMSNorm` deadlocks after ~3 calls in Inductor-compiled graphs

## Summary

The `mnnvl` backend for `allreduce_fusion` with `kARResidualRMSNorm` pattern deadlocks on the GPU after approximately 3 successful invocations when called from within torch.compile/Inductor-compiled piecewise subgraphs. The `trtllm` backend works correctly in the same scenario.

The CUDA kernel is launched (Python call returns immediately) but never completes — `torch.cuda.synchronize()` hangs indefinitely on all ranks. This was discovered while trying to make `mnnvl` the default allreduce backend in vLLM.

## Environment

- **GPU**: 4x B200 with NVSwitch (cc=10.0)
- **FlashInfer version**: [fill in]
- **PyTorch version**: [fill in]
- **CUDA version**: [fill in]
- **vLLM version**: main branch

## Reproduction

**Cannot reproduce with standalone flashinfer calls** — the bug only manifests when the allreduce is called from within Inductor-compiled code, specifically in vLLM's piecewise compilation pipeline. We tried many standalone variations (different `use_oneshot`, `norm_out` patterns, interleaved matmuls, tensor reuse) and none deadlock.

**Reproduces with vLLM**:
```bash
VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl \
  vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 -tp 4 --trust-remote-code
```

Works fine with `trtllm`:
```bash
VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 -tp 4 --trust-remote-code
```

## Detailed Findings

### What works
- `kAllReduce` pattern (plain allreduce, no RMSNorm fusion) with mnnvl — unlimited calls, no issues
- `kARResidualRMSNorm` with trtllm — unlimited calls, no issues
- `kARResidualRMSNorm` with mnnvl in standalone scripts (outside Inductor) — no issues
- tp=2 with mnnvl — works fine; issue is specific to tp=4 (world_size=4)
- First 3 calls to `kARResidualRMSNorm` with mnnvl in Inductor — complete successfully

### What deadlocks
- The 4th call to `kARResidualRMSNorm` with mnnvl backend, when called from Inductor-compiled piecewise subgraphs in vLLM

### Call parameters (identical for working and deadlocking calls)
```
shape=[512, 2688], backend=mnnvl, pattern=kARResidualRMSNorm (1)
dtype=bf16, world_size=4, fp32_acc=True, launch_with_pdl=True
```

### Execution flow (from instrumented debug logs)

The model is compiled into 30 piecewise subgraphs. Each subgraph contains the same structure:
```
Mamba gated_rmsnorm → linear (out_proj) → allreduce → FusedAddRMSNorm →
MoE (router_gemm + moe_forward) → allreduce → FusedAddRMSNorm → linear (in_proj)
```

The allreduce is called via `torch.ops.vllm.all_reduce` in the Inductor-generated code, which dispatches to `flashinfer.comm.allreduce_fusion(pattern=kARResidualRMSNorm)` at runtime.

**Timeline** (4 ranks, all show same behavior):
```
submod_0: allreduce #1 → sync OK ✓
submod_2: allreduce #2 → sync OK ✓, allreduce #3 → sync OK ✓
submod_4: allreduce #4 → Python returns, torch.cuda.synchronize() HANGS ✗
```

The pre-allreduce sync passes (all prior GPU ops completed). The allreduce kernel is launched on all 4 ranks. But the kernel never completes — `torch.cuda.synchronize()` blocks indefinitely.

### FX graph structure

Both working (submod_2) and deadlocking (submod_4) subgraphs have **identical structure** — only the layer indices differ. The compiled Inductor code is reused across subgraphs. This rules out graph-structural differences as the cause.

<details>
<summary>submod_2 FX graph (works)</summary>

```
Layers 0, 1, 2:
  Mamba Mixer2RMSNormGated (silu, group norm, weight mul)
  → linear out_proj [s72, 1024] → [s72, 2688]
  → torch.ops.vllm.all_reduce (tp:0)
  → FusedAddRMSNorm (to_f32, add_residual, pow, mean, rsqrt, mul, to_bf16, mul_weight)
  → MoE (router_gemm_bf16_fp32, moe_forward_shared, scale*2.5, add_shared)
  → torch.ops.vllm.all_reduce (tp:0)
  → FusedAddRMSNorm
  → linear in_proj [s72, 2688] → [s72, 2576]
  → allocate ssm_output, extract gate
```
</details>

<details>
<summary>submod_4 FX graph (deadlocks) — structurally identical</summary>

```
Layers 2, 3, 4:
  Mamba Mixer2RMSNormGated (silu, group norm, weight mul)
  → linear out_proj [s72, 1024] → [s72, 2688]
  → torch.ops.vllm.all_reduce (tp:0)
  → FusedAddRMSNorm (to_f32, add_residual, pow, mean, rsqrt, mul, to_bf16, mul_weight)
  → MoE (router_gemm_bf16_fp32, moe_forward_shared, scale*2.5, add_shared)
  → torch.ops.vllm.all_reduce (tp:0)
  → FusedAddRMSNorm
  → linear in_proj [s72, 2688] → [s72, 2576]
  → allocate ssm_output, extract gate
```
</details>

## Hypothesis

Since the graphs are identical, the compiled code is the same, and standalone repros don't trigger it, the most likely cause is **cumulative state corruption or resource exhaustion in the mnnvl backend** after 3 successful fused allreduce+RMSNorm operations. The internal workspace state (counters, buffers, synchronization primitives) may not be properly reset between calls, leading to a deadlock on the 4th invocation.

The fact that it only reproduces within Inductor-compiled code suggests the issue may be related to how Inductor schedules operations on the CUDA stream — possibly memory buffer reuse by Inductor's memory planner overlapping with mnnvl workspace buffers, or Triton kernel execution patterns affecting mnnvl's synchronization.
