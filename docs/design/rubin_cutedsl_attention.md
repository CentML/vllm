# Opt-in Rubin CuTeDSL attention

This integration calls the matching FlashInfer AOT interfaces. It does not
compile DKG source, substitute a different JIT kernel, or change default dispatch.
The paired `cubin_publishing` and FlashInfer changes must be installed first.

## ViT

Set `VLLM_RUBIN_CUTEDSL_VIT=1`, choose the `FLASHINFER` vision backend, enable
FP8 vision attention and provide `--mm-encoder-fp8-scale-path`. This path requires
SM107, logical head dimension 72, equal Q/KV head counts and BF16 output.
It reuses existing zero-padding/quantization to D80, preserves token indptr
instead of cuDNN's element offsets, and trims the BF16 output back to D72.
Static host scales avoid device-to-host synchronization in captured execution.
Dynamic-scale operation is deliberately rejected rather than silently using
stale capture-time scales. Other configurations retain the existing backend
when the opt-in is unset; incompatible explicit ViT opt-ins raise an error.

## GQA prefill

Set `VLLM_RUBIN_CUTEDSL_PREFILL=1` with the `FLASHINFER` attention backend,
FP8 E4M3 queries/KV, HND-compatible cache layout, and the TRTLLM metadata route.
The opt-in replaces only compatible causal prefill calls. Decoder attention,
cascade/DCP/native-FlashInfer routes and unsupported configurations retain their
existing implementations. Both model runners reach the same `FlashInferImpl`.

Supported shapes use D128 and page size 16/64/128, without sliding windows,
sinks or logit soft caps. BF16 output is supported; fused NVFP4 output requires
64 query heads and the existing attention/output-quantization fusion. The packed
output and whole Layout128x4 E4M3 scale buffer are forwarded without copying;
mixed batches retain the original decode-token row offset for the scale buffer.

The metadata builder prepares a persistent **KV token** indptr once per batch.
The existing TRTLLM `cum_seq_lens_kv` counts **pages** and cannot be reused as
DKG token lengths. K/V views retain their strides, including current vLLM's
interleaved `[pages, heads, page_size, 2*D]` storage; the exporter must support
these strides. No full-cache conversion is performed.

## Artifact and validation requirements

The MR199 artifact collection supplies ViT D72/D80, but does not supply the new
HND, corrected multi-sequence ragged, or fused-NVFP4 GQA exports. Build those with
the paired exporter changes and use `FLASHINFER_DSL_FMHA_LOCAL_DIR`, or publish
them and update the FlashInfer artifact manifest before enabling GQA. Missing
artifacts and kernel ABI errors propagate; they are not silently caught.

This branch stages source integration, not an E2E correctness/performance claim.
Before deployment, run FlashInfer kernel numerical and CUDA-graph replay tests,
then vLLM model accuracy and offline benchmarks with the intended artifacts.
In particular, test changed ragged lengths, empty padded graph entries, shuffled
pages and mixed decode/prefill FP4 scale offsets.

The host contract tests are:

```bash
.venv/bin/python -m pytest tests/v1/attention/test_rubin_cutedsl_dispatch.py -v
```

Look for `Rubin CuTeDSL ViT active` and `Rubin CuTeDSL paged prefill active` in
the worker logs. An opt-in alone is not evidence that every attention call used
the new kernel.
