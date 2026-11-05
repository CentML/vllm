# Serving feature flags (Qwen3.6 hybrid GDN + MoE, MTP speculative decoding)

Every feature below is read from the environment of the serving process and
is off by default, except `VLLM_H53_DRAFT_PARITY` and `VLLM_SPEC_LOCAL_TAIL_NO_PAD`. Value-type settings (sizes,
token lists, tables) only take effect while their feature switch is on.

`VLLM_H53_DRAFT_PARITY` is on by default. It keeps the target distribution
exact (the verifier tests the drafts against the distribution they were drawn
from) and acts only when the speculative config sets
`draft_sample_method: probabilistic`; with greedy drafts (the default) the
stock path runs, bit for bit, with no log line.

`VLLM_SPEC_LOCAL_TAIL_NO_PAD` is on by default. With probabilistic drafts, a
new request with exactly one uncached token is scheduled without the
never-proposed placeholder draft slots the scheduler would pad it with
(verifying placeholders is exact only for one-hot drafts; same issue as
upstream fedbc3b564). Greedy drafts and KV loaded by a connector keep the
padding, so the stock path is unchanged there.

Back to stock: `VLLM_H53_DRAFT_PARITY=0`, `VLLM_SPEC_LOCAL_TAIL_NO_PAD=0` and
none of the other flags set (greedy drafts make both defaults inert).

Settings that differ between serving configurations (MTP k, prefill schedule
interval, prefix match unit, memory utilization, per-domain KV and MoE
settings, tactic tables) belong in each configuration's server config / env
block, not in the engine. Values that depend on the GPU or the configuration
(token-count windows and thresholds, split-k tiers, the attention fork's cost
model) have no defaults here: unset, a feature runs as noted below and logs
that once.

## MoE front end, finalize and norm

| Flag | What it does |
|---|---|
| `VLLM_H82_ROUTER_PDL=1` | Router GEMM launched with programmatic dependent launch (PDL) at the stock position |
| `VLLM_H82_ROUTE_EARLY=<mask>` | Early PDL trigger in the routing kernels (1: kernels feeding FC1, 2: score kernel) |
| `VLLM_H82_FINALIZE_PREFETCH=1` | MoE finalize + RMSNorm kernel reads its PDL-independent inputs before the wait |
| `VLLM_H105_ROUTER_FIRST=1` | Router GEMM captured before the shared-expert fork |
| `VLLM_H105_PDL=1` | PDL on the remaining draft-sampling, finalize, attention and LM-head edges |
| `VLLM_H155_RF_SKIP_TOKENS`, `VLLM_H155_RF_SHARED_HIPRI_TOKENS`, `VLLM_H155_RF_DRAFT` | Per token count: router-first off, or shared experts on a highest-priority stream |
| `VLLM_H158_LEAN_ROUTE=1` (`_MIN`, `_MAX`) | Lean single-CTA routing kernel from raw logits for small token counts (unset: the kernel's whole range) |
| `VLLM_H158_SHARED_PRIO=1` (`_MIN`, `_MAX`) | Shared-expert stream priority by token count (an unset bound is open) |
| `VLLM_MOE_FIN2_ONE_LOAD=1`, `VLLM_MOE_FIN2_UNFUSED_MIN_TOKENS` | Two-buffer finalize: one row load per route; unfused norm above a token count |
| `VLLM_MXFP8_NORM_LARGE_M=1` (`_TOKENS`) | Fewer warps per row in the deferred-MoE norm from `_TOKENS` tokens (unset: every token count; not bitwise) |
| `VLLM_MOE_DIE_SPLIT=1` (`_CHECK`, `_TACTICS`) | Routed experts as per-memory-domain halves with a two-buffer finalize; per-half tactic table |

## Speculative decoding (MTP)

| Flag | What it does |
|---|---|
| `VLLM_H53_DRAFT_PARITY` (default on; `=0` off; `_KP`) | Drafts sampled from the target-processed (penalties, temperature, top-k/top-p) draft distribution; the verifier tests against it. Acts only with `draft_sample_method=probabilistic` |
| `VLLM_SPEC_LOCAL_TAIL_NO_PAD` (default on; `=0` off) | Probabilistic drafts: no placeholder draft slots for new one-token tails |
| `VLLM_H97_FAST_SELECT=1` | Restructured draft select kernels for the above; same draws |
| `VLLM_H159_TAIL=1` | Sampler tail folded into the verify step |
| `VLLM_DRAFT_TAIL_FUSE=1`, `VLLM_DRAFT_PRELUDE_FUSE=1` (`_MAX_ROWS`) | Draft glue copies fused into single kernels |
| `VLLM_H53_DIAG*` | Diagnostic log only |

## GDN (linear attention) decode and verify

| Flag | What it does |
|---|---|
| `VLLM_GDN_MTP_RING=1` (`_LEAN`, `_CONV_OUT`, `_NO_SPEC_BLOCKS`) | Replay ring for the speculative GDN states instead of per-token snapshots |
| `VLLM_GDN_MTP_DEFER_WRITES=1` (`_STREAM`) | WY verify kernel writes a replay log; accepted states are committed after sampling |
| `VLLM_GDN_MTP_VERIFY_V2=1\|2` | WY verify kernel v2 (value heads per CTA) |
| `VLLM_GDN_GFD_MIN_REQS`, `VLLM_GDN_GFD_MAX_REQS` | One fused CUDA decode kernel for decode-only batches of `_MIN_REQS` (unset: 1) to `_MAX_REQS` requests |
| `VLLM_GDN_LAYER_GRAPHS_CKPT=1` | GDN layer graphs also cover prefill checkpoint steps |

Deferred writes and verify v2 refuse to start next to the ring or other replacements of the same verify contract.

Without `VLLM_GDN_MTP_DEFER_WRITES` and `VLLM_GDN_MTP_VERIFY_V2` the WY kernel
(`VLLM_GDN_MTP_CUDA_WY`) is built from its stock source (`gdn_mtp_wy_stock.py`).

## GDN prefill

| Flag | What it does |
|---|---|
| `VLLM_GDN_VSPLIT_SCHED=1` | Chunked prefill kernel: one CTA per tile with a sequence map and scheduling hooks |
| `VLLM_GDN_CKPT_SIDE=1` | Prefill checkpoint gather/store on a side stream |
| `VLLM_H58_TAIL_CKPT=1` | Short prefills inside one Mamba block export their tail checkpoint in the same step |
| `VLLM_H58_SPLIT_DIAG*` | Diagnostic log only |

## Attention

| Flag | What it does |
|---|---|
| `VLLM_ATTN_MIXED_FORK=1` (`_KF_SMS`, `_MODEL`, `_DEC_GBPS_PER_SM`) | In mixed steps the prefill attention kernel runs on a high-priority side stream on a reduced SM budget beside the decode; `_KF_SMS=-1` picks the budget with a cost model whose parameters (`_MODEL`, `_DEC_GBPS_PER_SM`) have no defaults (unset: every SM) |
| `VLLM_KF_PLAN_EARLY=1` | Prefill attention plan computed on a worker thread at the start of the step |
| `VLLM_ATTN_DROP_SUPERSEDED_TAIL=1` | Drops superseded attention tail blocks from the prefix cache |
| `VLLM_H211_DL_VERIFY=1` (`VLLM_H211_DL_CACHE_DIR`, `VLLM_H211_DL_NVCC`) | Builds (nvcc, sm_107a) the die-aware MTP-verify attention kernel for KV whose blocks alternate between the two memory domains (`vllm/v1/attention/ops/die_verify_attn.py`, sources in `vllm/third_party/q36k_attn_dl/`) before the KV memory profile ends; with `VLLM_DIE_LOCAL_KV_STRIPE=1` it replaces the trtllm-gen decode launches of at least `VLLM_H211_DL_VERIFY_MIN_ROWS` rows (unset: every launch) with q_len <= 8 on the layers of `VLLM_H211_DL_VERIFY_PARTS` (`t` target, `d` MTP draft; unset: `t`). Outputs change (not bitwise). `VLLM_H211_DL_VERIFY_CHECK=N` (debug): the first N eager launches per layer also run trtllm-gen and log the difference |

## KV cache placement (GPUs with two memory locality domains)

| Flag | What it does |
|---|---|
| `VLLM_DIE_LOCAL_KV=1` | KV blocks placed per memory domain; decode launches per domain |
| `VLLM_DIE_LOCAL_KV_LPT=1` | Longest-first decode order |
| `VLLM_DIE_LOCAL_KV_STRIPE=1` | With `VLLM_DIE_LOCAL_KV=1`: block i of each request's attention block table on domain (i + s) % 2, s alternating per new request (a prefix-cache hit keeps its first block's domain), instead of one domain per request; no decode launch is split per domain |
| `VLLM_DIE_LOCAL_KV_ADDR_ORDER=1` | Each allocation's blocks are handed out in ascending block id per domain (the same blocks the LRU queues pop, so evictions and prefix-cache hits are unchanged); `=0` keeps the stock order; either logs page-order counts |
| `VLLM_DIE_LOCAL_KV_HOST_TRIM=1` | Eager per-domain decode launches reuse preallocated fork/join events, and the target model's attention layers compute the per-domain row masks once per forward |
| `VLLM_DIE_LOCAL_KV_SPLIT_K`, `_SPLIT_K_MIN_SEQ`, `_SM_GROUPS`, `_PROBE_MIN_RATIO`, `_SHARED_OUT`, `_MIN_ROWS`, `_SLACK`, `_STICKY`, `_EVICT_STALE` | Settings of the above (unset: no per-die split-k tiers, every launch split, no probe margin) |
| `VLLM_DIE_LOCAL_KV_STATS`, `_LOG_EVERY`, `_LPT_LOG_EVERY` | Log only |

## Prefix cache, Mamba state and scheduler

| Flag | What it does |
|---|---|
| `VLLM_PREFIX_EXACT_JUNCTION=1` (`_MIN_TOKENS`, `_MIN_GAIN`, `_MIN_SHARERS`) | Prefill chunks end exactly at shared-prefix junctions so they become cache hits |
| `VLLM_MAMBA_DROP_SUPERSEDED_COW=1` | Drops superseded Mamba checkpoints left as copy-on-write sources |
| `VLLM_KV_POOL_STATS` | Log only |

## Frontend

| Flag | What it does |
|---|---|
| `VLLM_PACKED_PROMPT_IDS=1` (`_MIN`, `_AB`, `_VERIFY`, `_TIME`, `_LOG_EVERY`) | Prompt token ids passed packed from the frontend to the engine core |
| `VLLM_PROMPT_TOKEN_BUFFER=1` | With `VLLM_PROMPT_TOKEN_ARRAY=1`, a packed prompt stays an int32 buffer that reads like a list (no per-token Python ints) in the engine core's `Request` |
| `VLLM_FEH_UNPACK_FAST=1` | The engine core splits shipped block hashes in one call (same hash objects) |
| `VLLM_INGEST_GATE=1` (`_LOG_EVERY`, `_TIMEOUT_S`) | The engine core's input thread decodes and preprocesses new requests only while the main thread waits (GPU sync, idle, input drain); each request is admitted not later than the first drain after the input thread takes it |

Existing flags whose code paths these features extend (`VLLM_GDN_LAYER_GRAPHS*`,
`VLLM_GDN_FUSED_DECODE_MAX_REQS`, `VLLM_KF_DECODE_ATTN*`, `VLLM_LOCALITY_LM_HEAD`,
`VLLM_FRONTEND_HASH_BLOCK_SIZE`, `VLLM_GDN_MTP_TRITON_MAX_REQS`) keep their stock
defaults and meanings.
