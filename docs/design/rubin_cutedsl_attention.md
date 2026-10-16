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

### Enabling it

`VLLM_RUBIN_CUTEDSL_PREFILL=1` is the only switch; it is off by default and no
artifact-path variable turns it on. The exports come from the FlashInfer
artifact manifest or from `FLASHINFER_DSL_FMHA_LOCAL_DIR`:

```bash
export VLLM_RUBIN_CUTEDSL_PREFILL=1
export FLASHINFER_DSL_FMHA_LOCAL_DIR=/path/to/exports    # if not published
export VLLM_RUBIN_CUTEDSL_PREFILL_POLICY=tuned           # optional, see below
```

`VLLM_RUBIN_CUTEDSL_PREFILL_POLICY` selects the kernel policy:

| Policy | Query layout | Kernel options |
| --- | --- | --- |
| `legacy` (default) | packed Q with `qo_indptr`; ragged-TMA (NVFP4) or direct (BF16) store | export defaults; the call is unchanged |
| `tuned` | per batch, from `rubin_paged_prefill_query_layout()` on the host query lengths: uniform `[batch, q_len]` Q when two or more prefills have equal lengths, else packed ragged TMA (NVFP4), dense TMA or direct (BF16) | `RUBIN_GQA_TUNED_POLICY` (cluster launch, CLC Work IDs for TMA stores, no sequence barrier, MMA row sum, packed NVFP4 epilogue) |

`tuned` needs a FlashInfer that exports `rubin_paged_prefill_query_layout` and
`RUBIN_GQA_TUNED_POLICY`; with an older FlashInfer, vLLM warns and uses
`legacy`. It also needs the matching exports for every layout a batch can
select. They are not part of the published artifacts, so a missing one raises
at the first batch that needs it. The builder copies the prefill query lengths
from `query_start_loc_cpu` (no device sync). The options are cached per
(lengths, output dtype), so the selection runs once per batch, not per layer.

### CUDA graphs

The query layout, `max_q_len` and the total query count are host arguments that
set the kernel's views and launch grid, so this call must not be captured in a
CUDA graph and replayed for another batch. It never is:

- `FlashInferMetadataBuilder.get_cudagraph_support()` returns at most
  `UNIFORM_BATCH`, so `CompilationConfig.resolve_cudagraph_mode_and_sizes()`
  never keeps FULL graphs for mixed batches. It downgrades to
  `FULL_AND_PIECEWISE` when attention is a splitting op, else to
  `FULL_DECODE_ONLY`.
- FULL graphs hold only batches whose query lengths are within the decode
  threshold. `split_decodes_and_prefills()` counts them as decodes, so
  `num_prefills == 0` and this path does not run inside them.
- Batches with prefills run PIECEWISE or eagerly. `unified_attention_with_output`
  is a splitting op. Dynamo splitting cuts the FX graph there,
  `use_inductor_graph_partition=True` partitions Inductor output at the same op,
  and `VLLM_USE_BREAKABLE_CUDAGRAPH=1` makes it an eager break. With
  `fuse_attn_quant` and no Inductor partition, splitting ops are empty and mixed
  batches run without a graph (`FULL_DECODE_ONLY`). In every case
  `FlashInferImpl.forward` runs in Python for each step, with that step's
  metadata.
- PIECEWISE and eager batches use unpadded request counts in both model
  runners, so the host lengths match `qo_indptr` and the query rows exactly.

The layout can therefore change from step to step. FULL graphs for mixed
batches would need a kernel that reads the layout and lengths on the device.

Each export loads on first use and is then cached. The mixed prefill/decode
warmup runs a single prefill (packed layout), so the uniform export can first
load during serving. That is a one-time cost, visible in FlashInfer's
`Loading DSL FMHA kernel variant` log.

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
the worker logs. `Rubin CuTeDSL paged prefill policy: legacy` or `... tuned`
records the policy. With `FLASHINFER_DSL_FMHA_LOCAL_DIR`, FlashInfer logs
`Loading DSL FMHA kernel variant <name>` once for each export it loads. An
opt-in alone is not evidence that every attention call used the new kernel.
