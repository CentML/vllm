# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and time vLLM MXFP8 linear backends on explicit logical shapes."""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from functools import partial
from pathlib import Path


def measure(torch, call, samples, calls):
    for _ in range(5):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            call()
    graph.replay()
    torch.accelerator.synchronize()

    def sample(operation):
        times = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / calls)
        return {
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "samples_ms": times,
        }

    def eager():
        for _ in range(calls):
            call()

    return {"eager": sample(eager), "cuda_graph": sample(graph.replay)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", nargs="+", type=int, default=[14190, 15740])
    parser.add_argument("--n", type=int, default=12352)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["cutlass", "cute-dsl", "cute-dsl-sm107"],
        default=["cute-dsl", "cute-dsl-sm107"],
    )
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--require-native-sm107", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.m, args.n, args.k, args.samples, args.calls) <= 0:
        parser.error("dimensions, samples, and calls must be positive")

    import torch
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.quantization import mxfp8_dequantize_host

    from vllm.model_executor.kernels.linear.mxfp8 import flashinfer as module
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    torch.accelerator.set_device_index(0)
    if args.require_native_sm107:
        assert torch.cuda.get_device_capability(0) == (10, 7), "Requires SM107"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    classes = {
        "cutlass": module.FlashInferCutlassMxfp8LinearKernel,
        "cute-dsl": module.FlashInferCutedslMxfp8LinearKernel,
        "cute-dsl-sm107": module.FlashInferCutedslSm107Mxfp8LinearKernel,
    }
    source = Path(module.__file__)
    native_source = source.with_name("flashinfer_sm107.py")
    result = {
        "device": str(torch.cuda.get_device_properties(0)),
        "cuda_build": torch.version.cuda,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "vllm", "flashinfer-python", "nvidia-cutlass-dsl")
        },
        "kernel_source": str(source),
        "kernel_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "native_helper_sha256": hashlib.sha256(native_source.read_bytes()).hexdigest(),
        "samples": args.samples,
        "calls_per_sample": args.calls,
        "scope": "Synthetic linear-layer validation; no model-accuracy claim.",
        "baseline_policy": (
            "Fresh in-process FlashInfer autotuning per backend with the default "
            "tuning buckets, after clearing its cache; no historical cache loaded."
        ),
        "timing": (
            "Inclusive apply_weights operator timings, not bare GEMM timings. "
            "Repeated resident tensors; clocks not locked; no cache flush. "
            "Weight conversion and warmup excluded. BF16-input timing includes "
            "activation quantization. Prequantized-input timing includes any "
            "backend padding/cropping, so it is not isolated MMA execution."
        ),
        "cases": [],
    }

    def dequant(q, scale, swizzled):
        return mxfp8_dequantize_host(
            q.view(torch.uint8).cpu(),
            scale.reshape(-1).cpu(),
            is_sf_swizzled_layout=swizzled,
        ).to(device=q.device)

    def check(output, expected):
        assert output.shape == expected.shape and torch.isfinite(output).all()
        diff = output.float() - expected.float()
        rel = (
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ).item()
        assert rel < 5e-4, f"Kernel relative L2 {rel} exceeded 5e-4"
        torch.testing.assert_close(output[-1], expected[-1], rtol=0.01, atol=0.01)
        return rel

    with torch.inference_mode():
        for m in args.m:
            torch.manual_seed(1309)
            x = torch.randn(m, args.k, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
            wq, ws = mxfp8_e4m3_quantize(weight, False)
            xq, xs = mxfp8_e4m3_quantize(x, True)
            reference = (dequant(xq, xs, True) @ dequant(wq, ws, False).t()).to(x.dtype)
            for backend in args.backends:
                if backend == "cute-dsl-sm107":
                    supported, reason = classes[backend].is_supported()
                    if not supported:
                        raise RuntimeError(
                            f"Native SM107 backend unavailable: {reason}"
                        )
                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(wq.clone(), requires_grad=False)
                layer.weight_scale = torch.nn.Parameter(ws.clone(), requires_grad=False)
                kernel = classes[backend](module.Mxfp8LinearLayerConfig())
                kernel.process_weights_after_loading(layer)
                qa = QuantizedActivation(
                    xq, xs, x.dtype, x.shape, kernel.input_quant_key()
                )
                call = partial(kernel.apply_weights, layer, x)
                prequantized_call = partial(kernel.apply_weights, layer, qa)
                AutoTuner.get().clear_cache()
                with autotune(tune_mode=True):
                    call()
                    prequantized_call()
                call()
                torch.accelerator.synchronize()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as profile:
                    output = call()
                    torch.accelerator.synchronize()
                names = sorted({event.name for event in profile.events()})
                native = any("Mxfp8TensorAdapter" in name for name in names)
                if args.require_native_sm107 and backend == "cute-dsl-sm107":
                    assert native, f"No Mxfp8TensorAdapter native kernel: {names}"
                error = check(output, reference)
                check(prequantized_call(), reference)
                record = {
                    "backend": backend,
                    "logical_mnk": [m, args.n, args.k],
                    "output_shape": list(output.shape),
                    "kernel_names": names,
                    "native_sm107": native,
                    "kernel_relative_l2": error,
                    "prequantized_input": measure(
                        torch, prequantized_call, args.samples, args.calls
                    ),
                    "bf16_input": measure(torch, call, args.samples, args.calls),
                }
                result["cases"].append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(record), flush=True)
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
