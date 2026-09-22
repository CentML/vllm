# Production Metrics

vLLM exposes a number of metrics that can be used to monitor the health of the
system. These metrics are exposed via the `/metrics` endpoint on the vLLM
OpenAI compatible API server.

You can start the server using Python, or using [Docker](../deployment/docker.md):

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
```

Then query the endpoint to get the latest metrics from the server:

??? console "Output"

    ```console
    $ curl http://0.0.0.0:8000/metrics

    # HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
    # TYPE vllm:iteration_tokens_total histogram
    vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
    vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    ...
    ```

The following metrics are exposed:

## General Metrics

--8<-- "gen:metrics-general"

## Speculative Decoding Metrics

--8<-- "gen:metrics-spec-decode"

## Active and inactive prefix-cache occupancy

Enable `--kv-cache-usage-metrics` to expose physical cache occupancy in bytes.
Logging statistics must remain enabled. This is separate from the sampled
block-lifetime metrics enabled by `--kv-cache-metrics`.

| Gauge | Meaning |
| --- | --- |
| `vllm:kv_cache_active_bytes` | Blocks held by outstanding references, including in-flight work or transfer pins. |
| `vllm:kv_cache_inactive_cached_bytes` | Unreferenced, reclaimable blocks that still have valid prefix-cache entries. |
| `vllm:kv_cache_free_uncached_bytes` | Unreferenced blocks without valid cached prefixes. |
| `vllm:kv_cache_capacity_bytes` | Usable physical cache pool, excluding the reserved null block. |

The first three gauges sum to capacity. Shared references and multiple hash
aliases do not multiply physical occupancy. Bytes include allocation padding;
they describe the cache pool, not total GPU memory or just attention KV. A
hybrid model's attention and Mamba state blocks share the measured pool.
The allocator reserves this memory even when blocks are free. Divide by `1e9`
for GB, or `2**30` for GiB. Existing `vllm:kv_cache_usage_perc` measures
non-free blocks and therefore excludes retained, reclaimable prefixes.
Pool bytes follow the per-rank cache configuration: for TP greater than one,
they are not the sum of memory across all TP ranks.

**Valid does not mean needed again.** Inactive history may belong to a completed
conversation, a paused conversation, or a shared prefix. The engine cannot infer
future reuse or permanent conversation completion from block reference counts.
These gauges do not label inactive bytes as future-useful bytes.

### Eviction and observed extra prefill

| Counter | Meaning |
| --- | --- |
| `vllm:kv_cache_evicted_blocks_total` | Valid physical blocks discarded when allocation reuses capacity, including duplicate copies. |
| `vllm:kv_cache_evicted_bytes_total` | The same discarded capacity in bytes, including padding. |
| `vllm:kv_cache_evicted_prefixes_total` | Prefix/group keys losing their last cached copy in those evictions. |
| `vllm:kv_cache_invalidated_blocks_total` | Explicit invalidations or checkpoint re-keying, counted separately from allocation pressure. |
| `vllm:kv_cache_eviction_recompute_requests_total` | Requests whose completed prefill included tokens attributable to remembered evictions. |
| `vllm:kv_cache_eviction_recompute_tokens_total` | Those completed prompt tokens. |
| `vllm:kv_cache_prefill_computed_tokens_total` | All prompt tokens in completed model steps, including repeated prefill, excluding decode. |

An eviction by itself does **not** prove wasted work. Attribution remembers
evicted prefix/group keys, performs a read-only hypothetical prefix lookup with
those entries retained, and compares its hit length with the actual lookup.
It reuses the engine's hybrid alignment and speculative-decoding block-drop
rules. Only the extra interval that subsequently executes as prompt work is
counted; failed admissions, decode tokens, and explicit cache resets do not
create recomputation counts. Asynchronous in-flight chunks are accounted when
their model results return, including already-issued work after cancellation.
The hypothetical lookup never retains KV tensors or alters scheduling.

This is a bounded counterfactual, not a measured latency saving. Set
`--kv-cache-eviction-history-size` to change its default bound of 100,000 keys.
`vllm:kv_cache_eviction_history_entries` shows current metadata size;
`vllm:kv_cache_eviction_history_dropped_total` reports keys forgotten because
of that bound. Attribution can undercount when history is lost. Resetting the
prefix cache clears remembered entries but does not reset monotonic counters.
No request IDs, token text, or prefix hashes are Prometheus labels.

Check `vllm:kv_cache_eviction_attribution_supported == 1` before interpreting
recomputation counters. Attribution currently supports full attention and Mamba
groups with local prefix caching and no KV connector. Other ordinary GPU pool
types expose occupancy but disable attribution. Host-resident pools are rejected
explicitly. Connector restore misses require separate offload-aware attribution.

Metrics have the existing model and engine labels. For independent workers,
scrape each worker's metrics endpoint and preserve its instance identity. Sum
bytes across workers for aggregate capacity; divide summed active bytes by
summed capacity for an aggregate active fraction. Do not average unequal pools.

Inactive occupancy plus eviction/recomputation counters can identify cache
pressure worth investigating, but cannot establish that offloading will improve
latency. Compare the additional prefill with transfer cost in an unprofiled
replay under the same concurrency, cache conditions, and stopping policy.

## NIXL KV Connector Metrics

--8<-- "gen:metrics-nixl"

## Model Flops Utilization (MFU) Performance Metrics

These metrics are available via `--enable-mfu-metrics`:

--8<-- "gen:metrics-mfu"

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
