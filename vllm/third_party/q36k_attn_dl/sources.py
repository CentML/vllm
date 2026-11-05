# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# mypy: ignore-errors
# fmt: off
"""Sources of the die-aware decode / MTP-verify attention kernel q36k_attn_decode_dl (H211), by Yibai Meng.

attn/attn_dl.cu is adapted by its author from Kernel Factory decode candidate 34c2c89d (CuTe DSL, Apache-2.0); see
its header.

vllm/v1/attention/ops/die_verify_attn.py builds these sources (nvcc, sm_107a) after applying H211's one change to
the kernel (STRIPE_PATCHES there: striped placement in runs of several pages). SHA256 per file of the text here,
checked before every build.
"""

SHA256 = {
    "include/q36k/common.h": "6a89e5d0651a26a19999c6be6091355eb423aab29a8e53725fcaa8f91fc9bbfe",
    "include/q36k/die.h": "2a972e070ec7951c3add86aaabbfe880f579b44517024892fff6256724775476",
    "include/q36k/attn.h": "18c27e3de2eb4012ca384d1e7b3bfcdb4853532735b246d604ed0de341c2b9ac",
    "common/ptx.cuh": "a21a274e64d719cbc091a7c645af94729899721d12b66d5113f12e68c41d0b97",
    "common/pdl.cuh": "ee2b74c28233cfc4b22860a477d085ca6fef2e92316e0cc40b5053a4f59a4381",
    "common/die.cuh": "1b5b6700ae6a2e395a9ef99fc602cea8b2584568b27e21a8885a484bb4ae6e06",
    "common/common.cu": "c5d8a1bc60358f5c151ab2596747bb85faa20aea01aae67237d9b20e663a1b0c",
    "common/die.cu": "0da5ca67e056d97b7ea4b68ade757e50d0ec66b5b2629b859c51d8812a658aae",
    "attn/attn_dl.h": "0fdd3d0e269a36c53fa3206f174cd5ffe9802391e6ab3433bc4dbf6c5898d57f",
    "attn/attn_dl.cu": "f1542f3d0321df50b1cb43cb1e340f06e6702c23809b014f76a3484761608514",
}

FILES = {}

FILES["include/q36k/common.h"] = r'''#pragma once
// Shared conventions and status/PDL ABI for every q36k entry point.
//
// Conventions:
//  - extern "C", returns a q36k_status; raw device pointers; explicit cudaStream_t.
//  - No allocation, no synchronization, CUDA-graph capturable.
//  - Shapes passed as int64_t; strides in elements unless named *_bytes.
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  Q36K_OK = 0,
  Q36K_ERR_INVALID_ARG = 1,
  Q36K_ERR_UNSUPPORTED = 2,
  Q36K_ERR_CUDA = 3,
  Q36K_ERR_WORKSPACE = 4,
} q36k_status;

const char* q36k_status_string(int status);

// Programmatic dependent launch (PDL): every q36k launch that supports it (own kernels, trtllm-gen GEMM / MoE /
// FMHA runners, CuTe DSL GEMMs) sets the programmatic-stream-serialization attribute while this is on. On by
// default; the environment variable Q36K_PDL=0 turns it off (read at the first call), q36k_set_pdl overrides
// it. Launches captured into a CUDA graph keep the setting of their capture. Kernels keep their
// griddepcontrol instructions, which are no-ops when launched without the attribute.
int q36k_pdl_enabled(void);
void q36k_set_pdl(int enabled);

#ifdef __cplusplus
}
#endif
'''

FILES["include/q36k/die.h"] = r'''#pragma once
// Dies (locality domains) of a dual-die GPU (VR200: 2 dies, each with half of the HBM).
//
// cudaMalloc memory interleaves both dies in 4 KiB chunks, so about half of every read stream crosses the
// die-to-die link, while reads of die-local memory by the same die's SMs stay on that die. Die-aware kernels
// therefore take data placed in die-local memory (cuMemCreate with CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN)
// and let each CTA process only the work whose data sits on its own SM's die. The SM -> die map is not a fixed
// function of %smid across boards, so it is probed once at startup (q36k_die_probe) into a small device table
// that kernels index with %smid (common/die.cuh).
#include <stddef.h>
#include <stdint.h>

#include "q36k/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Entries of an SM -> die table (indexed by %smid, which is < %nsmid <= this on every supported GPU).
#define Q36K_DIE_MAX_SMS 256
// Smallest pool per die for a reliable probe (random loads should miss L2).
#define Q36K_DIE_PROBE_MIN_BYTES (64u << 20)

// Probe the die of every SM by load latency: one block per SM chases dependent random loads through the
// die-0 pool [pool, pool + bytes_per_die) and the die-1 pool [pool + bytes_per_die, pool + 2 * bytes_per_die)
// (one LocalizedBuffer split at its half, bytes_per_die >= Q36K_DIE_PROBE_MIN_BYTES; contents may be uninitialized,
// the last 256 bytes are overwritten); the SM's die is the pool with the lower median latency. SMs probe in small
// groups one after another, all blocks resident at once: call it on an otherwise idle device.
//   sm_die  device uint8 [Q36K_DIE_MAX_SMS]: 0 / 1 per probed SM, 0xFF for SM ids that do not exist.
//   lat     device float [Q36K_DIE_MAX_SMS][2] or NULL: mean load latency (SM cycles) per pool.
// Stream ordered; not graph capturable (memsets + launch). Call once at startup.
int q36k_die_probe(const void* pool, size_t bytes_per_die, uint8_t* sm_die, float* lat, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
'''

FILES["include/q36k/attn.h"] = r'''#pragma once
// Full-attention layers of Qwen3.6-35B-A3B (10 target layers + the MTP layer).
//
// FP8 decode and context/generation prefill use trtllm-gen FMHA cubins and runner from FlashInfer 0.6.18.post1
// (Apache-2.0); long-KV prefill uses the Kernel Factory CuTe DSL kernel (candidate 511e91b9) and host planner from
// CentML vLLM d48687fdee (Apache-2.0); die-aware tcgen05 decode adapts Kernel Factory decode candidate 34c2c89d
// (Apache-2.0); pre-attention RMSNorm/RoPE/KV-write follows CentML vLLM d48687fdee fused_qk_norm_rope.py.
//
// Geometry (fixed): 16 q heads, 2 kv heads (GQA 8), head_dim 256, partial NeoX RoPE on the first 64 dims (theta 1e7),
// Gemma (1+w) q/k RMSNorm (eps 1e-6), softmax scale 1/16, sigmoid output gate.
//
// Per-step flow for one attention layer (T = tokens of the step, B = sequences):
//   qkv  = x @ Wqkv^T                                  (gemm area)  bf16 [T, 9216]
//   q36k_attn_prep(qkv, ...)                           q_fp8 [T,16,256]; K/V of the step -> KV cache
//   q36k_attn_decode(... seqs [0, Bd) ...)             decode/verify sequences (q_len 1..16)
//   q36k_attn_prefill(... seqs [Bd, B) ...)            prefill chunks (q_len >= 1)
//   q36k_attn_gate(out, qkv, ...)                      out *= sigmoid(gate), in place
//   y    = out @ Wo^T                                  (gemm area)
//
// Layouts:
//   qkv        bf16 [T, 9216] row-major: per q head h (0..15) columns [h*512, h*512+256) are q and [h*512+256,
//              h*512+512) the output gate (checkpoint q_proj row order); then k [8192, 8704) (kv head j at
//              8192 + j*256), v [8704, 9216).
//   q_fp8      e4m3 [T, 16, 256] (scale 1.0, saturating) = post-norm post-RoPE q.
//   out        bf16 [T, 16*256].
//   kv_cache   per layer e4m3 [num_pages, 2 (kv head), 128 (token in page), 512]: per token K (256, post-norm
//              post-RoPE) then V (256); scale 1.0 (RN, saturating). 128 KiB per page.
//   page_table int32 [B][page_table_stride]: page ids of each sequence (row b, entry p = page holding tokens
//              [128p, 128p+128) of sequence b).
//
// Batch metadata (int32 device buffers, rewritten every step; host args are per-graph-bucket sizes):
//   positions[T]       absolute position of each token (its index in its sequence).
//   cu_seqlens_q[B+1]  packed token offsets (absolute row indices into qkv/q/out); decode/verify sequences first,
//                      then prefill chunks. Rows >= cu_seqlens_q[B] are padding.
//   kv_lens[B]         KV length after this step's writes (cached prefix + this step's tokens), so the tokens of
//                      sequence b occupy positions [kv_lens[b] - q_len_b, kv_lens[b]).
//   Mixed batches: call decode on sequences [0, Bd) and prefill on [Bd, B) by offsetting the sequence-indexed
//   pointers (cu_seqlens_q + Bd, kv_lens + Bd, page_table + Bd * stride); q/out stay the batch base pointers because
//   cu_seqlens_q values are absolute row indices.
//
// Workspace: one buffer per stream, sized by q36k_attn_workspace_size, zeroed once by q36k_attn_workspace_init (the
// split-KV semaphores at its start reset themselves after every launch).
//
// All entry points except q36k_attn_init / q36k_attn_prefill_plan_make are CUDA-graph capturable and do not allocate
// or synchronize.
#include <stddef.h>
#include <stdint.h>

#include "q36k/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q36K_ATTN_NUM_Q_HEADS 16
#define Q36K_ATTN_NUM_KV_HEADS 2
#define Q36K_ATTN_HEAD_DIM 256
#define Q36K_ATTN_ROTARY_DIM 64
#define Q36K_ATTN_PAGE_SIZE 128
#define Q36K_ATTN_QKV_DIM 9216
#define Q36K_ATTN_OUT_DIM 4096
#define Q36K_ATTN_PAGE_BYTES 131072
// Upper bounds accepted by the entry points.
#define Q36K_ATTN_MAX_SEQS 1024
#define Q36K_ATTN_MAX_DECODE_Q_LEN 16

// ---- Initialization & workspace -------------------------------------------------------------------------------------

// Load the embedded attention cubins into the current device's context. Call once per device before the first
// attention launch (and before any CUDA-graph capture). Not graph capturable.
int q36k_attn_init(void);

// Workspace bytes for decode launches with up to max_seqs sequences, q_len <= max_q_len and kv_lens <= max_kv_len (the
// host max_kv_len you will pass, e.g. the model max for graph buckets). Prefill launches use whatever workspace is
// available (their KV-split choice is capped by it).
size_t q36k_attn_workspace_size(int64_t max_seqs, int64_t max_q_len, int64_t max_kv_len);

// Zero the workspace's split-KV semaphores. Once after allocation (stream-ordered memset).
int q36k_attn_workspace_init(void* workspace, size_t workspace_bytes, cudaStream_t stream);

// ---- Pre-attention prep & decode ------------------------------------------------------------------------------------

// Fused pre-attention op for every row t < num_tokens:
//   q, k = GemmaRMSNorm(q_h; q_norm_w), GemmaRMSNorm(k_j; k_norm_w)   (fp32, rounded to bf16)
//   q, k = NeoX RoPE on dims [0,64) with bf16 cos/sin of positions[t] * 1e7^(-i/32)  (bf16 result)
//   q_fp8[t] = e4m3(q); for rows t < cu_seqlens_q[num_seqs]: K = e4m3(k), V = e4m3(v) are written to
//   kv_cache at token positions[t] of the row's sequence (page page_table[b][pos/128], slot pos%128).
// q_norm_w, k_norm_w: the checkpoint's bf16 [256] q_norm / k_norm weights (used as 1 + w).
int q36k_attn_prep(const void* qkv, const void* q_norm_w, const void* k_norm_w, const int32_t* positions,
                   const int32_t* cu_seqlens_q, int64_t num_seqs, const int32_t* page_table, int64_t page_table_stride,
                   void* kv_cache, int64_t num_tokens, void* q_fp8, cudaStream_t stream);

// Decode / MTP verify: per sequence b, q_len_b = cu_seqlens_q[b+1] - cu_seqlens_q[b] in [1, max_q_len] query rows
// attend causally to the sequence's kv_lens[b] cached tokens (row i of the sequence sees KV positions <
// kv_lens[b] - q_len_b + 1 + i). out rows of the sequences are written.
//   num_tokens   rows of the q_fp8/out buffers (>= cu_seqlens_q[num_seqs]).
//   max_q_len    host upper bound of q_len_b (<= Q36K_ATTN_MAX_DECODE_Q_LEN).
//   max_kv_len   host upper bound of kv_lens (graph buckets: the model max); with num_seqs, kv_skew and the SM count
//                it sets the KV split (CTAs per sequence and kv head).
//   kv_skew      host upper bound of max(kv_lens) / mean(kv_lens) over the sequences, rounded up; 0 = unknown, assume
//                4. A larger value splits long KV finer so that one
//                sequence far above the mean does not become the critical path. Graph replays keep the split of their
//                capture: key graphs by it.
//   num_pages    pages of this layer's kv_cache pool.
// Performance: for batches of many long sequences the split assumes max/mean KV <= max(kv_skew, 4) and that sequences
// are ordered by kv_lens, longest first (CTAs launch in batch order); any order, any lengths and any kv_skew are
// correct.
int q36k_attn_decode(void* out, const void* q_fp8, const void* kv_cache, int64_t num_pages, const int32_t* page_table,
                     int64_t page_table_stride, const int32_t* kv_lens, const int32_t* cu_seqlens_q, int64_t num_seqs,
                     int64_t num_tokens, int64_t max_q_len, int64_t max_kv_len, int64_t kv_skew, void* workspace,
                     size_t workspace_bytes, cudaStream_t stream);

// KV split (CTAs per sequence and kv head; 1 = unsplit) q36k_attn_decode uses for these arguments on the current
// device, or a negative error. Host only.
int64_t q36k_attn_decode_splits(int64_t num_seqs, int64_t max_q_len, int64_t max_kv_len, int64_t kv_skew,
                                size_t workspace_bytes);
// Die-aware decode / MTP verify (own tcgen05 kernel, attn/attn_dl.cu): the computation of q36k_attn_decode
// for q_len <= Q36K_ATTN_DL_MAX_Q_LEN, with a persistent grid (one CTA per SM) that splits the KV of every
// (sequence, kv head) on the device from kv_lens (graph replays with any lengths stay balanced; no kv_skew or
// max_kv_len hint, any batch order).
//   sm_die          device SM -> die table from q36k_die_probe (q36k/die.h): die-local mode, where the KV pages are
//                   in die-localized memory (e.g. one pool with die 0's pages in its first half and die 1's in the
//                   second) placed as die_split_page says; each die's SMs read only their die's pages. A page on the
//                   other die is still read correctly (remotely). NULL: one work queue for all SMs (interleaved KV).
//   die_split_page  with sm_die: 0 = striped placement, page p of every sequence (its page_table[b][p]) expected on
//                   die p % 2 (every step's KV balanced between the dies whatever the batch); > 0 = per-sequence
//                   placement, the whole sequence on the die of its first page (pages >= die_split_page on die 1).
//                   Ignored without sm_die. Must be < num_pages.
//   chunks_per_sm   work chunks per SM of a die (1..Q36K_ATTN_DL_MAX_CHUNKS_PER_SM): each die's pages are cut into
//                   chunks of equal size taken from a per-die queue; more chunks balance better and write more
//                   partials.
//   steal           1: a CTA whose die queue is empty takes the other die's remaining chunks (remote reads).
// Workspace: an attention workspace (q36k_attn_workspace_size / _init) of at least
// q36k_attn_decode_dl_workspace_size(num_seqs) bytes. Output rows of sequences with q_len 0 or kv_len 0 are not
// written. All CTAs of the grid must become resident (split streams are merged at the end by CTAs that wait for
// pieces still in flight on other CTAs).
// Numerics: P keeps ~8 significant bits (hi/lo e4m3 split), V and Q are FP8 as stored; rows of split streams are
// merged in a fixed order (bitwise reproducible for the same lengths, die table, placement and chunks_per_sm).
#define Q36K_ATTN_DL_MAX_Q_LEN 8
#define Q36K_ATTN_DL_MAX_CHUNKS_PER_SM 4
// Self-resetting counters of q36k_attn_decode_dl in the attention workspace header (chunk queues, exit counter,
// per-stream piece counters, merge list): [offset, +bytes), all zero between launches (engine pool audits check it).
// Zeroed by q36k_attn_workspace_init.
#define Q36K_ATTN_DL_COUNTER_OFFSET (3584u * 1024u)
#define Q36K_ATTN_DL_COUNTER_BYTES ((32 + 4 * Q36K_ATTN_MAX_SEQS) * 4)
size_t q36k_attn_decode_dl_workspace_size(int64_t max_seqs);
int q36k_attn_decode_dl(void* out, const void* q_fp8, const void* kv_cache, int64_t num_pages,
                        const int32_t* page_table, int64_t page_table_stride, const int32_t* kv_lens,
                        const int32_t* cu_seqlens_q, int64_t num_seqs, int64_t num_tokens, int64_t max_q_len,
                        const uint8_t* sm_die, int64_t die_split_page, int64_t chunks_per_sm, int64_t steal,
                        void* workspace, size_t workspace_bytes, cudaStream_t stream);

// ---- Draft-only NVFP4 KV attention (MTP layer) ----------------------------------------------------------------------

// Draft-only NVFP4 KV for the MTP layer (target layers keep FP8 KV).
// Pool layout: page p at byte p * page_bytes (page_bytes >= Q36K_ATTN_KV4_PAGE_BYTES, multiple of 16), kv head j's
// block at + j * Q36K_ATTN_KV4_HEAD_BYTES:
//   K rows [128 tokens][128 B] | V pair rows [64][256 B] | K scales [128][16 B] | V scales [128][16 B]
// K row t: 256 e2m1 values of token t, element 2i in the low nibble of byte i; its 16-byte chunk c is stored at chunk
// c ^ (t & 7). V pair row p: byte i = element i of token 2p (low nibble) and of token 2p+1 (high nibble); chunk c
// stored at c ^ h(p & 3), h(x) = (x & 1) | (x & 2) << 1. (Both make the decode kernel's MMA fragment loads
// bank-conflict free; a V byte is one f16x2 A register.) Scales: e4m3, scale byte i of token t covers its elements
// [16i, 16i+16); value = e2m1 * scale (NVFP4 without a global scale: scale = e4m3(amax / 6) RN of the block, elements
// e2m1(x / scale) RN saturating).
#define Q36K_ATTN_KV4_HEAD_BYTES 36864
#define Q36K_ATTN_KV4_PAGE_BYTES 73728

// As q36k_attn_prep, with K and V (fp32 of the bf16 post-norm/RoPE k and the bf16 v) written to an NVFP4 kv_cache of
// the layout above.
int q36k_attn_prep_kv4(const void* qkv, const void* q_norm_w, const void* k_norm_w, const int32_t* positions,
                       const int32_t* cu_seqlens_q, int64_t num_seqs, const int32_t* page_table,
                       int64_t page_table_stride, void* kv_cache, int64_t page_bytes, int64_t num_tokens, void* q_fp8,
                       cudaStream_t stream);

// Workspace bytes q36k_attn_decode_kv4 needs for up to max_seqs sequences (uses the bytes past the first 4 MiB of an
// attention workspace, which the FP8 kernels' persistent state does not touch).
size_t q36k_attn_decode_kv4_workspace_size(int64_t max_seqs);

// Decode with single-token queries (cu_seqlens_q[b+1] - cu_seqlens_q[b] = 1, rows cu_seqlens_q[b]) over an NVFP4
// kv_cache: row b attends to kv_lens[b] tokens. out bf16 [*, 16*256] as q36k_attn_decode. The KV split is chosen on
// the device from kv_lens (graph replays with any lengths stay balanced).
int q36k_attn_decode_kv4(void* out, const void* q_fp8, const void* kv_cache, int64_t page_bytes,
                         const int32_t* page_table, int64_t page_table_stride, const int32_t* kv_lens,
                         const int32_t* cu_seqlens_q, int64_t num_seqs, void* workspace, size_t workspace_bytes,
                         cudaStream_t stream);

// ---- Prefill & output gate ------------------------------------------------------------------------------------------

#define Q36K_ATTN_PREFILL_CONTEXT 0     // trtllm-gen context kernel (128-token x 1-head tiles)
#define Q36K_ATTN_PREFILL_GENERATION 1  // trtllm-gen generation kernel (16-token x 8-head tiles, KV split across CTAs)
#define Q36K_ATTN_PREFILL_KF 2          // Kernel Factory CuTe DSL kernel (AOT, host work list)

// Kernel choice for one prefill launch. Zero-initialized = context kernel.
typedef struct {
  int32_t kernel;      // Q36K_ATTN_PREFILL_*
  int32_t sm_count;    // GENERATION: SM count its KV-split heuristic sees
  int64_t first_row;   // KF: row of the launch's first token (= cu_seqlens_q of its first seq)
  int64_t num_seqs;    // KF: sequences the work list was planned for
  int32_t kf_variant;  // KF: kernel form and work-list sizes (internal)
  int32_t kf_nwork, kf_nbins, kf_nmrg, kf_nmbin, kf_nslot;
} q36k_attn_prefill_plan;

// Bytes of the host staging buffer for a KF work list (allocate pinned, reuse across steps).
size_t q36k_attn_prefill_plan_host_bytes(void);

// Choose the prefill kernel from host-side lengths of the launch's sequences (q_lens = new tokens, kv_lens = KV length
// after the step). Host only (no GPU work, no allocation).
//   KF when host_staging != NULL, the library was built with it, page_table_stride equals its compiled stride (2057),
//   the device has enough SMs, every kv_len >= 4096 (the recipe's gate) and its partials fit the workspace. The work
//   list is written to host_staging and must be uploaded with q36k_attn_prefill_plan_upload before the launches; it is
//   only valid for exactly these lengths (the KF kernel reads lengths from the work list, not from kv_lens).
//   Otherwise GENERATION or CONTEXT by the vLLM recipe's cost model
//   (CentML vLLM d48687fdee, v1/attention/backends/flashinfer.py trtllm_gen_prefill_sm_count).
// num_seqs = 0 or NULL arrays: context kernel.
int q36k_attn_prefill_plan_make(int64_t num_seqs, const int32_t* q_lens, const int32_t* kv_lens, int64_t first_row,
                                int64_t page_table_stride, size_t workspace_bytes, void* host_staging,
                                q36k_attn_prefill_plan* plan);

// Copy a KF work list from host_staging into the workspace (async H2D on stream; keep the staging buffer unchanged
// until the copy has executed). No-op for other kernels. Once per step: all layers with the same lengths (target
// layers and the MTP layer) reuse it.
int q36k_attn_prefill_plan_upload(const q36k_attn_prefill_plan* plan, const void* host_staging, void* workspace,
                                  size_t workspace_bytes, cudaStream_t stream);

// Prefill / chunked prefill: sequences with q_len_b >= 1 new tokens on top of kv_lens[b] - q_len_b cached tokens,
// causal. Arguments as q36k_attn_decode; max_q_len / max_kv_len are host upper bounds of the launch's q_len_b /
// kv_lens. plan NULL = context kernel (graph capturable with any lengths); a KF plan needs its uploaded work list and
// is not meant for graph replay with new lengths.
int q36k_attn_prefill(void* out, const void* q_fp8, const void* kv_cache, int64_t num_pages, const int32_t* page_table,
                      int64_t page_table_stride, const int32_t* kv_lens, const int32_t* cu_seqlens_q, int64_t num_seqs,
                      int64_t num_tokens, int64_t max_q_len, int64_t max_kv_len, const q36k_attn_prefill_plan* plan,
                      void* workspace, size_t workspace_bytes, cudaStream_t stream);

// Output gate, in place: out[t, h*256+d] = bf16(out * sigmoid(gate)), gate = qkv[t, h*512+256+d] (fp32 math), for
// t < num_tokens.
int q36k_attn_gate(void* out, const void* qkv, int64_t num_tokens, cudaStream_t stream);

// ---- FP4 KV-cache simulation (debug) --------------------------------------------------------------------------------
// FP4 KV-cache simulation (debug, quality experiments only). The FP8 cache keeps its layout; values of the simulated
// parts are rounded to an FP4 grid, which e4m3 holds exactly for the
// MXFP4 formats (a deployable FP4 cache with these scales would hold the same numbers):
//   MXFP4_2D: e2m1 x 2^j, one power of two per 32 positions x 32 dims (positions [32i, 32i+32) of a sequence),
//             2^j = 2^(floor(log2 amax) - 2); amax over the block's positions written so far (an unfinished block is
//             re-quantized from the originals, kept in `staging`, whenever it grows, as a staging buffer would do).
//   MXFP4_1D: the same scale rule per token and 32 dims.
//   NVFP4:    e2m1 x e4m3(amax / 6) per token and 16 dims (no global scale), the product rounded to e4m3 (not exact).
// Elements e2m1(x / scale) RN (ties to even), saturating at 6.
#define Q36K_ATTN_KV_SIM_MXFP4_2D 1
#define Q36K_ATTN_KV_SIM_MXFP4_1D 2
#define Q36K_ATTN_KV_SIM_NVFP4 3
// Parts (bit mask).
#define Q36K_ATTN_KV_SIM_K 1
#define Q36K_ATTN_KV_SIM_V 2

// As q36k_attn_prep, with the KV parts in rto_parts (Q36K_ATTN_KV_SIM_K | _V) written as e4m3 rounded to odd
// (saturating), so that q36k_attn_kv_sim's FP4 rounding of them equals rounding the bf16 values directly.
int q36k_attn_prep_rto(const void* qkv, const void* q_norm_w, const void* k_norm_w, const int32_t* positions,
                       const int32_t* cu_seqlens_q, int64_t num_seqs, const int32_t* page_table,
                       int64_t page_table_stride, void* kv_cache, int64_t num_tokens, void* q_fp8, int rto_parts,
                       cudaStream_t stream);

// Bytes of one layer's MXFP4_2D staging buffer for num_slots sequence slots: [slot][kv head][64 positions][512 B]
// (originals of the last two 32-position blocks a slot wrote). Contents need no initialization.
size_t q36k_attn_kv_sim_staging_bytes(int64_t num_slots);

// FP4-round the parts `parts` of the KV the step wrote (rows < cu_seqlens_q[num_seqs], from q36k_attn_prep_rto with
// rto_parts = parts; batch arguments as q36k_attn_prep), in place in kv_cache. Call after the layer's attention
// launches (the step's own queries read its new KV in FP8, earlier KV FP4-rounded). Sequence b uses staging slot
// slots[b] (MXFP4_2D; staging NULL otherwise): a sequence whose first row starts inside a 32-position block must have
// written that block's earlier positions through the same slot (the engine's resumes start at page boundaries).
// The rows of a sequence have consecutive positions.
int q36k_attn_kv_sim(void* kv_cache, void* staging, const int32_t* slots, const int32_t* positions,
                     const int32_t* cu_seqlens_q, int64_t num_seqs, const int32_t* page_table,
                     int64_t page_table_stride, int64_t num_tokens, int format, int parts, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
'''

FILES["common/ptx.cuh"] = r'''#pragma once
// Shared device PTX primitives (mbarrier, async bulk copy, ldmatrix/stmatrix, mma.sync, warp reductions,
// BF16 conversions) and host/device integer & status helpers for q36k kernels.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "q36k/common.h"

// gemm/moe objects (and vendored FlashInfer code) compile with -fvisibility=hidden so their C++ symbols cannot
// interpose with another FlashInfer build in the same process; public C ABI functions are exported explicitly.
#ifndef Q36K_API
#define Q36K_API extern "C" __attribute__((visibility("default")))
#endif

#ifndef Q36K_CUDA_CHECK
#define Q36K_CUDA_CHECK(expr)                        \
  do {                                               \
    if ((expr) != cudaSuccess) return Q36K_ERR_CUDA; \
  } while (0)
#endif

namespace q36k {

// ---- Host / device integer & status helpers -------------------------------------------------------------------------

template <typename T, typename U>
__host__ __device__ constexpr T div_up(T a, U b) {
  return (a + static_cast<T>(b) - 1) / static_cast<T>(b);
}

template <typename T, typename U>
__host__ __device__ constexpr T round_up(T a, U b) {
  return div_up(a, b) * static_cast<T>(b);
}

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

inline int cuda_status(cudaError_t err) { return err == cudaSuccess ? Q36K_OK : Q36K_ERR_CUDA; }

#ifdef __CUDACC__

// ---- Shared-memory address & barrier helpers ------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_arrive_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}" ::"r"(bar),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void bar_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// ---- Async bulk copy & L2 cache policy helpers ----------------------------------------------------------------------

__device__ __forceinline__ uint64_t evict_first_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}

__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(bar), "l"(pol)
      : "memory");
}

// ---- Tensor-core matrix load/store & mma.sync helpers ---------------------------------------------------------------

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) { ldsm_x4(r, smem_u32(p)); }

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) { ldsm_x4_t(r, smem_u32(p)); }

__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r0), "r"(r1),
               "r"(r2), "r"(r3)
               : "memory");
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  mma_bf16(d, a[0], a[1], a[2], a[3], b0, b1);
}

__device__ __forceinline__ void mma_f16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  mma_f16(d, a[0], a[1], a[2], a[3], b0, b1);
}

// ---- Warp reductions & BF16 helpers ---------------------------------------------------------------------------------

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
  return v;
}

__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

#endif  // __CUDACC__

}  // namespace q36k
'''

FILES["common/pdl.cuh"] = r'''#pragma once
// Programmatic dependent launch (PDL) helpers for q36k kernels (see q36k_pdl_enabled in include/q36k/common.h).
//
// Contract of every q36k kernel launched through launch_pdl:
//  - Before pdl_wait() it touches only memory no kernel of the current step writes: weights, constants and
//    per-slot state written by an earlier step (each kernel documents what it reads early). It writes nothing
//    to global memory before pdl_wait().
//  - Every CTA calls pdl_wait() before its first access to anything a predecessor may write, and before it exits
//    (also CTAs with nothing to do): the predecessor's completion and memory flush are guaranteed only after it,
//    and this grid's completion must imply its predecessor's (a dependent waits only for this grid).
//  - pdl_trigger() comes after pdl_wait() in every CTA, so a dependent launched early still waits (in its own
//    pdl_wait) for this grid, which itself waited for its predecessor: completion stays transitive, and a
//    dependent's pre-wait part never overlaps a grid older than its immediate predecessor.
// Without the launch attribute (PDL off) both instructions are no-ops.
#include <cuda_runtime.h>

#include <utility>

#include "q36k/common.h"

namespace q36k {

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// The common case: nothing to prefetch, so wait and let the next kernel launch at once.
__device__ __forceinline__ void pdl_wait_trigger() {
  pdl_wait();
  pdl_trigger();
}

// Launch configuration with the PDL attribute set iff q36k_pdl_enabled(); more attributes may be appended
// through add() (cluster dims, ...).
struct PdlConfig {
  cudaLaunchConfig_t cfg{};
  cudaLaunchAttribute attr[4]{};
  PdlConfig(dim3 grid, dim3 block, size_t smem, cudaStream_t stream) {
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attr;
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = q36k_pdl_enabled() ? 1 : 0;
    cfg.numAttrs = 1;
  }
  cudaLaunchAttribute& add() { return attr[cfg.numAttrs++]; }
};

template <class... KArgs, class... Args>
cudaError_t launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                       Args&&... args) {
  PdlConfig c(grid, block, smem, stream);
  return cudaLaunchKernelEx(&c.cfg, kernel, std::forward<Args>(args)...);
}

}  // namespace q36k
'''

FILES["common/die.cuh"] = r'''#pragma once
// Device helpers for die-aware kernels (see include/q36k/die.h).
#include <cstdint>

#include "q36k/die.h"

namespace q36k {

__device__ __forceinline__ uint32_t sm_id() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}

// Die (0 / 1) of the calling SM from the table written by q36k_die_probe; 0 for a NULL table (single die).
// Read it once per CTA: a CTA never migrates between SMs.
__device__ __forceinline__ int die_of_sm(const uint8_t* sm_die) {
  if (!sm_die) return 0;
  const uint32_t s = sm_id();
  const uint32_t d = s < Q36K_DIE_MAX_SMS ? sm_die[s] : 0u;
  return d == 1u ? 1 : 0;
}

}  // namespace q36k
'''

FILES["common/common.cu"] = r'''// Status strings and programmatic dependent launch (PDL) runtime toggle (see include/q36k/common.h).
#include "q36k/common.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

#include "ptx.cuh"

namespace {

// -1: not read yet; else 0 / 1.
std::atomic<int> g_pdl{-1};

}  // namespace

Q36K_API const char* q36k_status_string(int status) {
  switch (status) {
    case Q36K_OK:
      return "ok";
    case Q36K_ERR_INVALID_ARG:
      return "invalid argument";
    case Q36K_ERR_UNSUPPORTED:
      return "unsupported shape or configuration";
    case Q36K_ERR_CUDA:
      return "CUDA error";
    case Q36K_ERR_WORKSPACE:
      return "workspace too small";
    default:
      return "unknown error";
  }
}

Q36K_API int q36k_pdl_enabled(void) {
  int v = g_pdl.load(std::memory_order_relaxed);
  if (v < 0) {
    const char* e = std::getenv("Q36K_PDL");
    v = (e && std::strcmp(e, "0") == 0) ? 0 : 1;
    int expected = -1;
    if (!g_pdl.compare_exchange_strong(expected, v, std::memory_order_relaxed)) v = expected;
  }
  return v;
}

Q36K_API void q36k_set_pdl(int enabled) { g_pdl.store(enabled ? 1 : 0, std::memory_order_relaxed); }
'''

FILES["common/die.cu"] = r'''// SM -> die latency probe (see include/q36k/die.h).
//
// On VR200 a load from the other die's HBM takes more cycles than a local one, which identifies every SM's
// die including SMs outside any green-context locality partition.
//
// One block per SM (the shared-memory request keeps it at one), all resident at once on an idle device. The blocks
// probe in groups of kGroup SMs (a block of ticket t starts when every block of the earlier groups has finished),
// so the chains do not queue behind each other's misses.
#include "q36k/die.h"

#include <cuda_runtime.h>

#include <cstdint>

#include "die.cuh"
#include "ptx.cuh"

namespace {

constexpr int kThreads = 32;
constexpr int kChain = 48;              // dependent loads per timed chain
constexpr int kReps = 5;                // chains per pool (odd: the median is one of them)
constexpr int kGroup = 8;               // SMs probing concurrently
constexpr int kLineBytes = 256;         // address granule of the random loads
constexpr int kBlockSmem = 160 * 1024;  // one block per SM
constexpr int kCounterBytes = 256;      // two counters at the end of the die-1 pool (zeroed by the launcher)

__device__ __forceinline__ uint32_t ld_volatile(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__global__ void die_probe_kernel(const uint8_t* __restrict__ pool, uint64_t bytes_per_die, uint64_t line_mask,
                                 uint32_t* counters, uint8_t* __restrict__ sm_die, float* __restrict__ lat) {
  if (threadIdx.x != 0) return;
  const uint32_t sm = q36k::sm_id();
  const uint32_t ticket = atomicAdd(&counters[0], 1u);
  const uint32_t start = ticket / kGroup * kGroup;
  while (ld_volatile(&counters[1]) < start) __nanosleep(1000);
  long long cyc[2][kReps];
  uint32_t v = 0;
  for (int rep = -1; rep < kReps; ++rep) {  // rep -1: untimed warm-up (TLB) of both pools
    for (int k = 0; k < 2; ++k) {
      const int dom = (rep & 1) ? 1 - k : k;
      const uint8_t* const base = pool + static_cast<uint64_t>(dom) * bytes_per_die;
      const uint32_t seed = sm * 0x9E3779B1u + static_cast<uint32_t>(2 * rep + 2 + dom) * 0x85EBCA6Bu;
      // Both candidate lines of every step are dropped from L2 first, so each timed load goes to DRAM (an L2 hit
      // would hide the difference between the dies). The counters at the end of the die-1
      // pool are outside the probed lines (line_mask covers the first half of each pool).
      uint32_t h = seed;
      for (int i = 0; i < kChain; ++i) {
        h = h * 1664525u + 1013904223u;
        const uint64_t line = static_cast<uint64_t>(h & ~1u) & line_mask;
        asm volatile("discard.global.L2 [%0], 128;" ::"l"(base + line * kLineBytes) : "memory");
        asm volatile("discard.global.L2 [%0], 128;" ::"l"(base + (line + 1) * kLineBytes) : "memory");
      }
      h = seed;
      long long t0, t1;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0)::"memory");
      for (int i = 0; i < kChain; ++i) {
        h = h * 1664525u + 1013904223u;
        // The loaded value feeds the next address: every load waits for the previous one. Volatile loads cannot be
        // removed by ptxas (the chain's values are otherwise dead) and bypass L1.
        const uint64_t line = static_cast<uint64_t>((h & ~1u) | (v & 1u)) & line_mask;
        v = ld_volatile(reinterpret_cast<const uint32_t*>(base + line * kLineBytes));
      }
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)::"memory");
      if (rep >= 0) cyc[dom][rep] = t1 - t0;
    }
  }
  __threadfence();
  atomicAdd(&counters[1], 1u);
  float med[2];
  for (int d = 0; d < 2; ++d) {  // median of kReps by selection
    for (int i = 0; i <= kReps / 2; ++i) {
      for (int j = i + 1; j < kReps; ++j) {
        if (cyc[d][j] < cyc[d][i]) {
          const long long t = cyc[d][i];
          cyc[d][i] = cyc[d][j];
          cyc[d][j] = t;
        }
      }
    }
    med[d] = static_cast<float>(cyc[d][kReps / 2]) / kChain;
  }
  if (sm >= Q36K_DIE_MAX_SMS) return;
  sm_die[sm] = med[1] < med[0] ? 1 : 0;
  if (lat) {
    lat[2 * sm] = med[0];
    lat[2 * sm + 1] = med[1];
  }
}

}  // namespace

Q36K_API int q36k_die_probe(const void* pool, size_t bytes_per_die, uint8_t* sm_die, float* lat, cudaStream_t stream) {
  if (!pool || !sm_die || bytes_per_die < Q36K_DIE_PROBE_MIN_BYTES) return Q36K_ERR_INVALID_ARG;
  int dev = 0, sms = 0;
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
    return Q36K_ERR_CUDA;
  if (sms > Q36K_DIE_MAX_SMS) return Q36K_ERR_UNSUPPORTED;
  uint64_t lines = 1;
  while (lines * 2 * kLineBytes <= bytes_per_die) lines *= 2;
  auto* counters = reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(static_cast<const uint8_t*>(pool)) +
                                               2 * bytes_per_die - kCounterBytes);
  if (cudaFuncSetAttribute(die_probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kBlockSmem) != cudaSuccess ||
      cudaMemsetAsync(sm_die, 0xFF, Q36K_DIE_MAX_SMS, stream) != cudaSuccess ||
      (lat && cudaMemsetAsync(lat, 0, Q36K_DIE_MAX_SMS * 2 * sizeof(float), stream) != cudaSuccess))
    return Q36K_ERR_CUDA;
  // Two passes, the second one counts: the first probe in a process runs with cold address translation and
  // clocks.
  for (int pass = 0; pass < 2; ++pass) {
    if (cudaMemsetAsync(counters, 0, kCounterBytes, stream) != cudaSuccess) return Q36K_ERR_CUDA;
    die_probe_kernel<<<sms, kThreads, kBlockSmem, stream>>>(static_cast<const uint8_t*>(pool), bytes_per_die, lines - 1,
                                                            counters, sm_die, lat);
    if (cudaPeekAtLastError() != cudaSuccess) return Q36K_ERR_CUDA;
  }
  return Q36K_OK;
}
'''

FILES["attn/attn_dl.h"] = r'''#pragma once
// Workspace layout shared by the trtllm-gen launcher (attn_fmha.cu) and the die-aware decode kernel (attn_dl.cu)
// (see include/q36k/attn.h).
//
// Adapted from Kernel Factory decode candidate 34c2c89d (CuTe DSL, Apache-2.0) for the 10-warp tcgen05 tile
// pipeline.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Workspace bytes q36k_attn_decode_dl needs on the current device (header + partial slots).
size_t attn_dl_workspace_bytes(int64_t max_seqs);
// Zero the counters (stream ordered).
cudaError_t attn_dl_workspace_init(void* workspace, cudaStream_t stream);
'''

FILES["attn/attn_dl.cu"] = r'''// Die-aware paged FP8 decode / MTP-verify attention (q36k_attn_decode_dl): persistent tcgen05 kernel (see
// include/q36k/attn.h).
//
// Adapted from Kernel Factory decode candidate 34c2c89d (CuTe DSL, Apache-2.0) for the 10-warp tcgen05 (umma + tmem +
// mbarrier: 1 producer + 1 MMA + 8 softmax warps) tile pipeline, with a new die-aware persistent work queue and
// deferred merge list.
//
// Design:
//   * One persistent CTA per SM. Work is the page space of every (sequence, kv head) stream. On a dual-die GPU each
//     die has its own page space, cut into chunks of equal page count, about chunks_per_sm per SM of the die. Die
//     placement (where the engine allocated the KV): striped (die_split_page 0: page p of every sequence on die
//     p % 2, so every step's KV is balanced between the dies whatever the batch) or per sequence (die_split_page > 0:
//     the whole sequence on the die of its first page, pages >= die_split_page on die 1). A CTA takes chunks only from
//     its own SM's die queue (die from the probed %smid table, q36k/die.h), then (optionally) steals the other die's
//     remaining chunks. Without a die table there is one queue and one page space (interleaved memory).
//   * A chunk is cut at stream boundaries into segments. A segment that covers a whole stream writes the output;
//     otherwise it writes an fp32 partial (O, row max, row sum) to a slot of the workspace. The CTA finishing a
//     stream's last piece appends the stream to a global merge list; CTAs whose chunk queues are exhausted take
//     streams from that list and merge their pieces in a fixed order (deterministic for given lengths and SM count).
//     Deferring the merges keeps every CTA's KV stream running and spreads the merges over all CTAs at the end
//     (merging inline would leave the merges of many streams to the CTA that happens to finish their last pieces).
//   * Warp roles: warp 8 = producer (chunk queue, segment ring, TMA of K and V halves of each (page, kv head) through
//     a 6-stage 32 KiB ring), warp 9 = MMA issuer, warps 0-7 = softmax / epilogue / merge.
//   * S = Q K^T: tcgen05 kind::f8f6f4 M=128 x N=128 tokens x K=256 (4 MMAs of K=64) into one of two TMEM S buffers.
//     The 128 MMA rows hold up to 8 query tokens x 8 heads twice: lane 32 qd + 16 gq + 8 lo + i is query i of head
//     2 qd + gq, lo = 0 the "hi" copy and lo = 1 the "lo" copy of the same row. Hi lanes store P_hi = e4m3(x), lo
//     lanes P_lo = e4m3((x - P_hi) * 16), so O = O_hi + O_lo / 16 carries ~8 significant bits of P with FP8 V.
//     x = exp2(s * scale * log2e - m_ref + log2(448) - tau), m_ref a lazily raised row max (O rescaled only when
//     the max grows by more than tau).
//   * O += P V: A = P from TMEM (in place of the S buffer), B = V (MN-major SW128), M=128 x N=256 x K=128 tokens.
//   * Softmax warp w: TMEM lane quadrant qd = w % 4, token half hf = w / 4 of every page; the hi and lo lanes of a
//     row split the half's 64 tokens (32 each) and swap the other part of P through a shuffle.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>

#include "../common/die.cuh"
#include "../common/pdl.cuh"
#include "../common/ptx.cuh"
#include "attn_dl.h"
#include "q36k/attn.h"
#include "q36k/die.h"

namespace {

// ---- Constants & shared-memory layout -------------------------------------------------------------------------------

constexpr int kNumQHeads = Q36K_ATTN_NUM_Q_HEADS;
constexpr int kNumKvHeads = Q36K_ATTN_NUM_KV_HEADS;
constexpr int kGroup = kNumQHeads / kNumKvHeads;
constexpr int kHeadDim = Q36K_ATTN_HEAD_DIM;
constexpr int kPageSize = Q36K_ATTN_PAGE_SIZE;
constexpr int kMaxQ = Q36K_ATTN_DL_MAX_Q_LEN;
constexpr int kRows = kMaxQ * kGroup;  // partial rows per stream: r = i * 8 + head in group
static_assert(2 * kRows == 128, "hi/lo copies of every row fill the 128 MMA rows");

constexpr int kSmWarps = 8, kProdWarp = 8, kMmaWarp = 9, kThreads = 320, kSmThreads = 256;
constexpr int kStageBytes = 32768;  // K or V of one (page, kv head): [2 x 128 B column chunk][128 tok][128 B]
constexpr int kStages = 6;
constexpr int kSegRing = 8;
constexpr int kMinChunk = 8;                                 // pages (of one kv head) per chunk, at least
constexpr float kTau = 4.f;                                  // lazy rescale threshold (log2 units)
constexpr float kScaleLog2 = 0.0625f * 1.4426950408889634f;  // 1/sqrt(256) * log2(e)
constexpr float kPOff = 8.807354922057604f - kTau;           // log2(448) - tau
// TMEM columns: S (one buffer, read out by the softmax warps right after QK lands), P (two 32-column e4m3 buffers),
// Q (64 columns = the 256 B of each MMA row, the A operand of QK), O (256 fp32 columns).
constexpr uint32_t kTmemCols = 512, kTmemS = 0, kTmemP = 128, kTmemQ = 192, kTmemO = 256;

// Partial slot: O fp32 [kRows][256], then m [kRows], l [kRows].
constexpr int kSlotFloats = kRows * kHeadDim + 2 * kRows;
// Workspace: counters in the attention workspace header, partials in its scratch.
constexpr size_t kCtrOffset = Q36K_ATTN_DL_COUNTER_OFFSET;
constexpr size_t kScratchOffset = 4u << 20;
constexpr size_t kTraceOffset = kCtrOffset + 32768;  // diagnostics: Q36K_DIE_MAX_SMS x 8 u64
static_assert(kTraceOffset + Q36K_DIE_MAX_SMS * 64 <= kScratchOffset, "trace in the header");
static_assert(Q36K_ATTN_DL_COUNTER_BYTES <= 32768, "counters below the trace");
// Counters (u32 at kCtrOffset): [0] [1] die chunk queues, [2] exits, [3] merge list length, [4] merge list reads;
// [32, +2 MAX_SEQS) pieces finished per stream; [kCtrList, +2 MAX_SEQS) merge list (stream 2 b + h, plus 1; 0 empty).
constexpr int kCtrList = 32 + 2 * Q36K_ATTN_MAX_SEQS;

// Shared memory (offsets from a 1024-byte aligned base).
constexpr int kSmStage = 0;                                    // kStages x 32 KiB
constexpr int kSmIncl = kSmStage + kStages * kStageBytes;      // int [2][MAX_SEQS]: per-die inclusive page sums
constexpr int kSmKv = kSmIncl + 2 * Q36K_ATTN_MAX_SEQS * 4;    // int [MAX_SEQS]: kv_len
constexpr int kSmQi = kSmKv + Q36K_ATTN_MAX_SEQS * 4;          // int [MAX_SEQS]: row0 | q_len << 28
constexpr int kSmCnt = kSmQi + Q36K_ATTN_MAX_SEQS * 4;         // uint16 [2][MAX_SEQS]: sequences before b in each die
constexpr int kSmMerge = kSmCnt + 2 * Q36K_ATTN_MAX_SEQS * 2;  // float [2][kMergeCap] + [kRows]: merge scratch
constexpr int kMergeCap = 512;                                 // (piece, row) pairs staged per merge batch
constexpr int kSmMx = kSmMerge + (2 * kMergeCap + kRows) * 4;  // float [2 buf][2 hf][128]: page max exchange
constexpr int kSmLp = kSmMx + 2048;                            // float [2 hf][128]: row sums
constexpr int kSmSeg = kSmLp + 1024;                           // Seg [kSegRing]
constexpr int kSmBar = kSmSeg + kSegRing * 64;                 // mbarriers
constexpr int kNumBars = 2 * kStages + 2 + 2 + 3 + 2 * kSegRing;
constexpr int kSmScal = kSmBar + kNumBars * 8;  // scalars
constexpr int kSmDie = kSmScal + 128;           // uint8 [256]: SM -> die table
constexpr int kSmEnd = kSmDie + Q36K_DIE_MAX_SMS;
constexpr int kSmBytes = kSmEnd + 1024;  // + alignment slack
static_assert(kSmBytes <= 227 * 1024, "smem budget");
// Barrier indices.
constexpr int kBarFull = 0, kBarEmpty = kStages, kBarSFull = 2 * kStages, kBarSFree = kBarSFull + 1,
              kBarPFull = kBarSFree + 1, kBarODone = kBarPFull + 2, kBarOFinal = kBarODone + 1,
              kBarQFull = kBarOFinal + 1, kBarSegFull = kBarQFull + 1, kBarSegEmpty = kBarSegFull + kSegRing;
// Scalars (int offsets in kSmScal): tmem base, total pages T[2], chunk size C[2], chunks R[2], slot base S[2],
// merge flag, split streams (merges of the launch), scan scratch [20] (2 values x 10 warps; the 4-value scan below
// reuses it twice).
constexpr int kScTmem = 0, kScT = 1, kScC = 3, kScR = 5, kScBase = 7, kScFlag = 9, kScNMerge = 10, kScRed = 12;

// Instruction descriptors (kind::f8f6f4, e4m3 x e4m3 -> f32, K = 64 per MMA on sm_107: bit 29).
constexpr uint32_t kIdescQK = (1u << 4) | ((128u >> 3) << 17) | ((128u >> 4) << 24) | (1u << 29);
constexpr uint32_t kIdescPV = (1u << 4) | (1u << 16) | ((256u >> 3) << 17) | ((128u >> 4) << 24) | (1u << 29);

struct Seg {
  int b, h, p0, p1;  // stream (sequence b, kv head h), its pages pstride * j + poff for j in [p0, p1)
  int kv_len, q_len, row0;
  int slot;           // partial slot, -1: the segment is the whole stream
  int npieces;        // pieces of the stream (both dies)
  int pstride, poff;  // 2, die (die-local placement) or 1, 0
  int pad[5];
};
static_assert(sizeof(Seg) == 64, "segment ring entry");

// ---- Device PTX & tcgen05 helpers -----------------------------------------------------------------------------------

__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

__device__ __forceinline__ void tma_load_5d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, int c3,
                                            int c4, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;" ::"r"(dst),
      "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(bar), "l"(pol)
      : "memory");
}

// UMMA shared-memory descriptor: SW128, version 1 (sm_100 family).
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return static_cast<uint64_t>((addr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(lbo >> 4) << 16) |
         (static_cast<uint64_t>(sbo >> 4) << 32) | (1ull << 46) | (2ull << 61);
}
__device__ __forceinline__ void umma_ts(uint32_t d, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;\n}" ::"r"(d),
      "r"(a_tmem), "l"(b), "r"(idesc), "r"(acc)
      : "memory");
}
__device__ __forceinline__ void umma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}

#define Q36K_R8(p, i)                                                                                         \
  "=r"(p[i]), "=r"(p[i + 1]), "=r"(p[i + 2]), "=r"(p[i + 3]), "=r"(p[i + 4]), "=r"(p[i + 5]), "=r"(p[i + 6]), \
      "=r"(p[i + 7])
#define Q36K_W8(p, i) \
  "r"(p[i]), "r"(p[i + 1]), "r"(p[i + 2]), "r"(p[i + 3]), "r"(p[i + 4]), "r"(p[i + 5]), "r"(p[i + 6]), "r"(p[i + 7])
#define Q36K_F8(p, i)                                                                                         \
  "=f"(p[i]), "=f"(p[i + 1]), "=f"(p[i + 2]), "=f"(p[i + 3]), "=f"(p[i + 4]), "=f"(p[i + 5]), "=f"(p[i + 6]), \
      "=f"(p[i + 7])

// 64 consecutive TMEM columns of this warp's 32 lanes (one lane per thread) and their maximum.
__device__ __forceinline__ void tmem_ld64_max(uint32_t taddr, float (&v)[64], float& mx) {
  asm volatile(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];\n"
      "tcgen05.wait::ld.sync.aligned;"
      : Q36K_F8(v, 0), Q36K_F8(v, 8), Q36K_F8(v, 16), Q36K_F8(v, 24), Q36K_F8(v, 32), Q36K_F8(v, 40), Q36K_F8(v, 48),
        Q36K_F8(v, 56), "=f"(mx)
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_ld64(uint32_t taddr, uint32_t (&v)[64]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];\n"
      "tcgen05.wait::ld.sync.aligned;"
      : Q36K_R8(v, 0), Q36K_R8(v, 8), Q36K_R8(v, 16), Q36K_R8(v, 24), Q36K_R8(v, 32), Q36K_R8(v, 40), Q36K_R8(v, 48),
        Q36K_R8(v, 56)
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_st64(uint32_t taddr, const uint32_t (&v)[64]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x64.b32 [%0], "
      "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
      "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, "
      "%33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, "
      "%49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};" ::"r"(taddr),
      Q36K_W8(v, 0), Q36K_W8(v, 8), Q36K_W8(v, 16), Q36K_W8(v, 24), Q36K_W8(v, 32), Q36K_W8(v, 40), Q36K_W8(v, 48),
      Q36K_W8(v, 56)
      : "memory");
}
__device__ __forceinline__ void tmem_st32(uint32_t taddr, const uint32_t (&v)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
      "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
      "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};" ::"r"(taddr),
      Q36K_W8(v, 0), Q36K_W8(v, 8), Q36K_W8(v, 16), Q36K_W8(v, 24)
      : "memory");
}
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t (&v)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::"r"(taddr),
      Q36K_W8(v, 0), Q36K_W8(v, 8)
      : "memory");
}
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
// 16x256b.x8: thread t gets lane base + t/4 (the hi copy) and lane base + 8 + t/4 (the lo copy) at columns
// 8k + 2(t%4) + {0, 1}, k < 8; returns hi + lo / 16 in column order.
__device__ __forceinline__ void tmem_ld_hilo(uint32_t taddr, float (&o)[16]) {
  uint32_t r[32];
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x8.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
      "tcgen05.wait::ld.sync.aligned;"
      : Q36K_R8(r, 0), Q36K_R8(r, 8), Q36K_R8(r, 16), Q36K_R8(r, 24)
      : "r"(taddr)
      : "memory");
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    o[2 * k] = fmaf(__uint_as_float(r[4 * k + 2]), 0.0625f, __uint_as_float(r[4 * k]));
    o[2 * k + 1] = fmaf(__uint_as_float(r[4 * k + 3]), 0.0625f, __uint_as_float(r[4 * k + 1]));
  }
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// hi = e4m3(f16(x)), lo = e4m3((f16(x) - hi) * 16) for 4 values (a in the low byte); own/send = hi/lo on hi lanes.
__device__ __forceinline__ void pack_own_send(float a, float b, float c, float d, uint32_t sel_hi, uint32_t& own,
                                              uint32_t& send) {
  asm("{\n.reg .b32 xab, xcd, hab, hcd, k16, hiw, low;\n.reg .b16 eab, ecd, lab, lcd;\n.reg .pred p;\n"
      "mov.b32 k16, 0x4C004C00;\n"
      "cvt.rn.f16x2.f32 xab, %3, %2;\n"
      "cvt.rn.f16x2.f32 xcd, %5, %4;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 eab, xab;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 ecd, xcd;\n"
      "mov.b32 hiw, {eab, ecd};\n"
      "cvt.rn.f16x2.e4m3x2 hab, eab;\n"
      "cvt.rn.f16x2.e4m3x2 hcd, ecd;\n"
      "sub.rn.f16x2 xab, xab, hab;\n"
      "sub.rn.f16x2 xcd, xcd, hcd;\n"
      "mul.rn.f16x2 xab, xab, k16;\n"
      "mul.rn.f16x2 xcd, xcd, k16;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 lab, xab;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 lcd, xcd;\n"
      "mov.b32 low, {lab, lcd};\n"
      "setp.ne.u32 p, %6, 0;\n"
      "selp.b32 %0, hiw, low, p;\n"
      "selp.b32 %1, low, hiw, p;\n}"
      : "=r"(own), "=r"(send)
      : "f"(a), "f"(b), "f"(c), "f"(d), "r"(sel_hi));
}
__device__ __forceinline__ float ldcg_f(const float* p) { return __ldcg(p); }

// ---- Die-aware decode kernel ----------------------------------------------------------------------------------------

struct Params {
  __nv_bfloat16* out;
  const uint8_t* q;
  const int32_t* page_table;
  int64_t pt_stride;
  const int32_t* kv_lens;
  const int32_t* cu_q;
  const uint8_t* sm_die;
  float* part;
  uint32_t* ctr;    // counters, see kCtrList
  uint64_t* trace;  // NULL, or per CTA [8]: smid | die << 32, start, plan done, prod done, end, pages, segs, merges
  int num_seqs;
  int chunks_per_sm;
  int steal;
  int die_split;  // 0: striped placement (page p on die p % 2); > 0: per sequence, pages >= die_split on die 1
};

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Smallest b in [0, n) with incl[b] > x (incl non-decreasing, incl[n-1] > x).
__device__ __forceinline__ int find_seq(const int* incl, int n, int x) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (incl[mid] > x)
      hi = mid;
    else
      lo = mid + 1;
  }
  return lo;
}

// Stream (b, h) in die d's page space: start s0 and n pages; its pieces = chunks of die d it falls into; the partial
// slot of chunk k's piece of the stream.
__device__ __forceinline__ void dl_stream_in_die(const uint8_t* sm, int b, int h, int d, int& s0, int& n) {
  const int* const incl = reinterpret_cast<const int*>(sm + kSmIncl) + d * Q36K_ATTN_MAX_SEQS;
  const int excl = b > 0 ? incl[b - 1] : 0;
  n = (incl[b] - excl) >> 1;
  s0 = excl + h * n;
}
__device__ __forceinline__ int dl_pieces_in_die(const uint8_t* sm, int s0, int n, int d) {
  const int c = reinterpret_cast<const int*>(sm + kSmScal)[kScC + d];
  return n > 0 ? (s0 + n - 1) / c - s0 / c + 1 : 0;
}
__device__ __forceinline__ int dl_slot_of(const uint8_t* sm, int d, int k, int b, int h) {
  const int* const sc = reinterpret_cast<const int*>(sm + kSmScal);
  const uint16_t* const cnt = reinterpret_cast<const uint16_t*>(sm + kSmCnt);
  return sc[kScBase + d] + k + 2 * cnt[d * Q36K_ATTN_MAX_SEQS + b] + h;
}

struct MergeSeg {
  int b, h, q_len, row0, npieces;
};

// The CTA finishing a stream's last piece merges all its pieces (256 softmax threads), two (row, 8-column) tasks
// per thread at a time (more accumulators spilled registers of the softmax loop).
__device__ __forceinline__ void merge_stream(const Params& prm, uint8_t* sm, MergeSeg cur, int tid) {
  const int* const s_sc = reinterpret_cast<const int*>(sm + kSmScal);
  // ---- merge the stream's pieces: die 0's in chunk order, then die 1's. The (max, sum) pairs of a batch of
  // pieces are staged in shared memory with one round of loads; each thread then folds its (row, 8 columns)
  // tasks piece by piece, all tasks' loads of a piece in flight together.
  float* const s_m = reinterpret_cast<float*>(sm + kSmMerge);
  float* const s_l = s_m + kMergeCap;
  float* const s_inv = s_l + kMergeCap;
  int ps0[2], pn[2], pkf[2], pcnt[2];
#pragma unroll
  for (int d = 0; d < 2; ++d) {
    dl_stream_in_die(sm, cur.b, cur.h, d, ps0[d], pn[d]);
    pcnt[d] = dl_pieces_in_die(sm, ps0[d], pn[d], d);
    pkf[d] = pn[d] > 0 ? ps0[d] / s_sc[kScC + d] : 0;
  }
  auto piece_slot = [&](int k) {
    const int d = k < pcnt[0] ? 0 : 1;
    return dl_slot_of(sm, d, pkf[d] + (d ? k - pcnt[0] : k), cur.b, cur.h);
  };
  const int nrow = kGroup * cur.q_len, total = cur.npieces, batch = kMergeCap / nrow;
  constexpr int kTasks = kRows * 32 / kSmThreads;  // (row, 8-column group) tasks per thread, at most
  constexpr int kGroupT = 2;                       // tasks folded together (registers: 2 x 8 accumulators)
  constexpr int kUnroll = 4;                       // pieces loaded together per task group
  float M = -INFINITY, L = 0.f;
  auto stage = [&](int k0, int kn) {  // (m, l) of pieces [k0, k0 + kn) -> s_m, s_l [piece][row]
    for (int e = tid; e < kn * nrow; e += kSmThreads) {
      const float* const ps =
          prm.part + static_cast<int64_t>(piece_slot(k0 + e / nrow)) * kSlotFloats + kRows * kHeadDim + e % nrow;
      s_m[e] = ldcg_f(ps);
      s_l[e] = ldcg_f(ps + kRows);
    }
    q36k::bar_sync(5, kSmThreads);
  };
  auto weights = [&](int kn, bool sum) {  // s_m := exp2(m - M) per (piece, row); L += w l
    if (tid < nrow) {
      for (int k = 0; k < kn; ++k) {
        const float m = s_m[k * nrow + tid], w = m == -INFINITY ? 0.f : ex2(m - M);
        if (sum) L = fmaf(w, s_l[k * nrow + tid], L);
        s_m[k * nrow + tid] = w;
      }
    }
    q36k::bar_sync(5, kSmThreads);
  };
  const bool multi = total > batch;            // more (piece, row) pairs than the stage holds: restage per pass
  for (int k0 = 0; k0 < total; k0 += batch) {  // row maxima
    const int kn = min(batch, total - k0);
    stage(k0, kn);
    if (tid < nrow) {
      for (int k = 0; k < kn; ++k) M = fmaxf(M, s_m[k * nrow + tid]);
    }
    q36k::bar_sync(5, kSmThreads);
  }
  for (int k0 = 0; k0 < total; k0 += batch) {  // row sums (and the weights of a single batch)
    const int kn = min(batch, total - k0);
    if (multi) stage(k0, kn);
    weights(kn, true);
  }
  if (tid < nrow) s_inv[tid] = L > 0.f ? 1.f / L : 0.f;
  q36k::bar_sync(5, kSmThreads);
  // A task group covers rows [8 t0, 8 t0 + 16): groups past the stream's rows are skipped (uniform over the CTA).
  for (int t0 = 0; t0 < kTasks && 8 * t0 < nrow; t0 += kGroupT) {
    float acc[kGroupT][8];
#pragma unroll
    for (int t = 0; t < kGroupT; ++t) {
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[t][e] = 0.f;
    }
    for (int k0 = 0; k0 < total; k0 += batch) {
      const int kn = min(batch, total - k0);
      if (multi) {
        stage(k0, kn);
        weights(kn, false);
      }
      // kUnroll pieces' loads in flight together (not one dependent L2 round trip per piece); each accumulator still
      // adds the pieces in order k = 0, 1, ...
      for (int k = 0; k < kn; k += kUnroll) {
        float4 va[kUnroll][kGroupT], vb[kUnroll][kGroupT];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          if (k + u < kn) {
            const float* const ps = prm.part + static_cast<int64_t>(piece_slot(k0 + k + u)) * kSlotFloats;
#pragma unroll
            for (int t = 0; t < kGroupT; ++t) {
              const int e = tid + kSmThreads * (t0 + t), r = e >> 5;
              if (r < nrow) {
                va[u][t] = __ldcg(reinterpret_cast<const float4*>(ps + r * kHeadDim + 8 * (e & 31)));
                vb[u][t] = __ldcg(reinterpret_cast<const float4*>(ps + r * kHeadDim + 8 * (e & 31) + 4));
              }
            }
          }
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          if (k + u < kn) {
#pragma unroll
            for (int t = 0; t < kGroupT; ++t) {
              const int e = tid + kSmThreads * (t0 + t), r = e >> 5;
              if (r < nrow) {
                const float w = s_m[(k + u) * nrow + r];
                acc[t][0] = fmaf(w, va[u][t].x, acc[t][0]);
                acc[t][1] = fmaf(w, va[u][t].y, acc[t][1]);
                acc[t][2] = fmaf(w, va[u][t].z, acc[t][2]);
                acc[t][3] = fmaf(w, va[u][t].w, acc[t][3]);
                acc[t][4] = fmaf(w, vb[u][t].x, acc[t][4]);
                acc[t][5] = fmaf(w, vb[u][t].y, acc[t][5]);
                acc[t][6] = fmaf(w, vb[u][t].z, acc[t][6]);
                acc[t][7] = fmaf(w, vb[u][t].w, acc[t][7]);
              }
            }
          }
        }
      }
      if (multi) q36k::bar_sync(5, kSmThreads);  // s_m / s_l free for the next stage
    }
#pragma unroll
    for (int t = 0; t < kGroupT; ++t) {
      const int e = tid + kSmThreads * (t0 + t), r = e >> 5;
      if (r < nrow) {
        const float inv = s_inv[r];
        __nv_bfloat162 v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = __floats2bfloat162_rn(acc[t][2 * q] * inv, acc[t][2 * q + 1] * inv);
        __nv_bfloat16* const dst =
            prm.out +
            (static_cast<int64_t>(cur.row0 + r / kGroup) * kNumQHeads + cur.h * kGroup + r % kGroup) * kHeadDim +
            8 * (e & 31);
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(v);
      }
    }
  }
  q36k::bar_sync(5, kSmThreads);  // merge scratch free
}

__global__ void __launch_bounds__(kThreads, 1)
    attn_dl_kernel(const __grid_constant__ CUtensorMap kv_map, const Params prm) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* const sm =
      reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~static_cast<uintptr_t>(1023));
  const uint32_t sb = q36k::smem_u32(sm);
  int* const s_incl = reinterpret_cast<int*>(sm + kSmIncl);
  int* const s_kv = reinterpret_cast<int*>(sm + kSmKv);
  int* const s_qi = reinterpret_cast<int*>(sm + kSmQi);
  int* const s_sc = reinterpret_cast<int*>(sm + kSmScal);
  uint8_t* const s_die = sm + kSmDie;
  Seg* const s_seg = reinterpret_cast<Seg*>(sm + kSmSeg);
  auto bar = [&](int i) { return sb + kSmBar + 8 * i; };
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint64_t t_start = prm.trace ? globaltimer() : 0;

  // ---- prologue before the PDL wait: barriers, TMEM, the die table (constant)
  if (warp == kProdWarp && lane == 0) {
    for (int s = 0; s < kStages; ++s) {
      q36k::mbar_init(bar(kBarFull + s), 1);
      q36k::mbar_init(bar(kBarEmpty + s), 1);
    }
    q36k::mbar_init(bar(kBarSFull), 1);
    q36k::mbar_init(bar(kBarSFree), kSmWarps);
    for (int b = 0; b < 2; ++b) q36k::mbar_init(bar(kBarPFull + b), kSmWarps);
    q36k::mbar_init(bar(kBarODone), 1);
    q36k::mbar_init(bar(kBarOFinal), 1);
    q36k::mbar_init(bar(kBarQFull), kSmWarps);
    for (int s = 0; s < kSegRing; ++s) {
      q36k::mbar_init(bar(kBarSegFull + s), 1);
      q36k::mbar_init(bar(kBarSegEmpty + s), 1 + kSmWarps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&kv_map) : "memory");
  }
  if (warp == kMmaWarp) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(sb + kSmScal + 4 * kScTmem),
                 "r"(kTmemCols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  if (tid < Q36K_DIE_MAX_SMS) s_die[tid] = prm.sm_die ? prm.sm_die[tid] : 0;
  const int my_die = q36k::die_of_sm(prm.sm_die);
  q36k::pdl_wait();
  q36k::pdl_trigger();

  // ---- plan: per sequence pages, die and the per-die inclusive page sums (all threads)
  const int B = prm.num_seqs;
  {
    constexpr int kPer = (Q36K_ATTN_MAX_SEQS + kThreads - 1) / kThreads;
    // Per sequence: pages in die 0 / die 1 (x 2 kv heads) and whether it has any there; block scans of both.
    int a[4][kPer];
    int t[4] = {0, 0, 0, 0};
#pragma unroll
    for (int e = 0; e < kPer; ++e) {
      const int b = tid * kPer + e;
      int c[4] = {0, 0, 0, 0};
      if (b < B) {
        const int r0 = __ldg(prm.cu_q + b), ql = __ldg(prm.cu_q + b + 1) - r0, kv = __ldg(prm.kv_lens + b);
        const int np = (ql > 0 && kv > 0) ? (kv + kPageSize - 1) / kPageSize : 0;
        // Pages of each die (two kv heads per page): striped, page p on die p % 2; per sequence, all pages on the
        // die of the first one; one die without a die table.
        const bool one = prm.die_split > 0 && np > 0;
        const int d1 = one && __ldg(prm.page_table + static_cast<int64_t>(b) * prm.pt_stride) >= prm.die_split;
        c[0] = !prm.sm_die ? 2 * np : one ? (d1 ? 0 : 2 * np) : 2 * ((np + 1) >> 1);
        c[1] = !prm.sm_die ? 0 : one ? (d1 ? 2 * np : 0) : 2 * (np >> 1);
        c[2] = c[0] > 0;
        c[3] = c[1] > 0;
        s_kv[b] = kv;
        s_qi[b] = r0 | (ql << 28);
      }
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        a[v][e] = t[v];  // exclusive within the thread
        t[v] += c[v];
      }
    }
    int* const red = s_sc + kScRed;  // [10 warps][2]
    int base[4];
#pragma unroll
    for (int v0 = 0; v0 < 4; v0 += 2) {  // block exclusive scan of the per-thread totals, two values at a time
      int x0 = t[v0], x1 = t[v0 + 1];
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y0 = __shfl_up_sync(0xffffffffu, x0, o), y1 = __shfl_up_sync(0xffffffffu, x1, o);
        if (lane >= o) {
          x0 += y0;
          x1 += y1;
        }
      }
      if (lane == 31) {
        red[2 * warp] = x0;
        red[2 * warp + 1] = x1;
      }
      __syncthreads();
      int b0 = x0 - t[v0], b1 = x1 - t[v0 + 1];
      for (int w = 0; w < warp; ++w) {
        b0 += red[2 * w];
        b1 += red[2 * w + 1];
      }
      base[v0] = b0;
      base[v0 + 1] = b1;
      __syncthreads();
    }
    uint16_t* const s_cnt = reinterpret_cast<uint16_t*>(sm + kSmCnt);
#pragma unroll
    for (int e = 0; e < kPer; ++e) {
      const int b = tid * kPer + e;
      if (b < B) {
        const int c0 = (e + 1 < kPer ? a[0][e + 1] : t[0]) - a[0][e],
                  c1 = (e + 1 < kPer ? a[1][e + 1] : t[1]) - a[1][e];
        s_incl[b] = base[0] + a[0][e] + c0;
        s_incl[Q36K_ATTN_MAX_SEQS + b] = base[1] + a[1][e] + c1;
        s_cnt[b] = static_cast<uint16_t>(base[2] + a[2][e]);
        s_cnt[Q36K_ATTN_MAX_SEQS + b] = static_cast<uint16_t>(base[3] + a[3][e]);
      }
    }
    if (tid == kThreads - 1) {  // streams per die = 2 x sequences with pages there
      red[0] = base[2] + t[2];
      red[1] = base[3] + t[3];
    }
    __syncthreads();
    // SMs per die (all threads: one table entry each)
    uint32_t nsm;
    asm("mov.u32 %0, %%nsmid;" : "=r"(nsm));
    const bool on = prm.sm_die && tid < static_cast<int>(min(nsm, static_cast<uint32_t>(Q36K_DIE_MAX_SMS)));
    const int n_d1 = __syncthreads_count(on && s_die[tid] == 1);
    const int n_d0 = __syncthreads_count(on && s_die[tid] == 0);
    if (tid == 0) {
      int n[2] = {n_d0, n_d1};
      if (!prm.sm_die) n[0] = gridDim.x;
      // Partial slots of die d: chunk k's piece of stream i (i-th stream of the die) is slot base_d + k + i, unique
      // because a chunk's pieces are consecutive streams; per die at most R_d + streams_d slots.
      int sbase = 0;
      for (int d = 0; d < 2; ++d) {
        const int T = B > 0 ? s_incl[d * Q36K_ATTN_MAX_SEQS + B - 1] : 0;
        const int per = max(n[d], 1) * prm.chunks_per_sm;
        const int c = max(kMinChunk, (T + per - 1) / per);
        const int R = (T + c - 1) / c;
        s_sc[kScT + d] = T;
        s_sc[kScC + d] = c;
        s_sc[kScR + d] = R;
        s_sc[kScBase + d] = sbase;
        sbase += R + 2 * red[d];
      }
      s_sc[kScNMerge] = 0;
    }
    // TMEM base: fence around the barrier that publishes it.
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    // Split streams (more than one piece): the merges of this launch.
    int nsplit = 0;
#pragma unroll
    for (int e = 0; e < kPer; ++e) {
      const int b = tid * kPer + e;
      if (b < B) {
        for (int h = 0; h < 2; ++h) {
          int np = 0;
          for (int d = 0; d < 2; ++d) {
            int s0, n;
            dl_stream_in_die(sm, b, h, d, s0, n);
            np += dl_pieces_in_die(sm, s0, n, d);
          }
          nsplit += np > 1;
        }
      }
    }
    if (nsplit) atomicAdd(s_sc + kScNMerge, nsplit);
    __syncthreads();
  }
  const uint32_t tmem = static_cast<uint32_t>(s_sc[kScTmem]);
  // Stream (b, h) in die d's page space: start s0 and n pages; pieces = chunks of die d it falls into.
  auto stream_in_die = [&](int b, int h, int d, int& s0, int& n) { dl_stream_in_die(sm, b, h, d, s0, n); };
  auto pieces_in_die = [&](int s0, int n, int d) { return dl_pieces_in_die(sm, s0, n, d); };
  auto slot_of = [&](int d, int k, int b, int h) { return dl_slot_of(sm, d, k, b, h); };
  uint64_t* const trace = prm.trace ? prm.trace + 8 * blockIdx.x : nullptr;
  if (trace && tid == 0) {
    trace[0] = q36k::sm_id() | (static_cast<uint64_t>(my_die) << 32);
    trace[1] = t_start;
    trace[2] = globaltimer();
    trace[7] = 0;
  }

  if (warp == kProdWarp) {
    // ---- Producer warp ----------------------------------------------------------------------------------------------
    const uint64_t pol = q36k::evict_first_policy();
    bool own_done = false;
    auto fetch = [&]() -> int {  // (die << 24) | chunk, or -1; lane 0 result broadcast
      int r = -1;
      if (lane == 0) {
        if (!own_done) {
          const int k = static_cast<int>(atomicAdd(prm.ctr + my_die, 1u));
          if (k < s_sc[kScR + my_die])
            r = (my_die << 24) | k;
          else
            own_done = true;
        }
        if (r < 0 && own_done && prm.steal) {
          const int o = 1 - my_die;
          const int k = static_cast<int>(atomicAdd(prm.ctr + o, 1u));
          if (k < s_sc[kScR + o]) r = (o << 24) | k;
        }
      }
      return __shfl_sync(0xffffffffu, r, 0);
    };
    int seg_n = 0;  // segments pushed
    auto push = [&](const Seg& s) {
      const int i = seg_n % kSegRing;
      q36k::mbar_wait(bar(kBarSegEmpty + i), ((seg_n / kSegRing) & 1) ^ 1);
      if (lane == 0) {
        s_seg[i] = s;
        q36k::mbar_arrive(bar(kBarSegFull + i));
      }
      __syncwarp();
      ++seg_n;
    };
    int slot = 0, phase = 0;  // ring position of the next K/V half
    auto issue = [&](int h, int page, int c2) {
      q36k::mbar_wait(bar(kBarEmpty + slot), phase ^ 1);
      if (lane == 0) {
        q36k::mbar_arrive_tx(bar(kBarFull + slot), kStageBytes);
        tma_load_5d(sb + kSmStage + slot * kStageBytes, &kv_map, 0, 0, c2, h, page, bar(kBarFull + slot), pol);
      }
      __syncwarp();
      if (++slot == kStages) {
        slot = 0;
        phase ^= 1;
      }
    };
    // The next chunk is claimed when kPrefetchPages pages of the current one remain to be issued (claiming it at the
    // start of the current chunk let the first CTAs hold two chunks each while late CTAs found the queue empty).
    constexpr int kPrefetchPages = 4;
    int npages = 0;
    int next = fetch();
    while (true) {
      const int chunk = next;
      if (chunk < 0) break;
      bool fetched = false;
      const int d = chunk >> 24, k = chunk & 0xFFFFFF;
      const int c = s_sc[kScC + d], T = s_sc[kScT + d];
      const int* const incl = s_incl + d * Q36K_ATTN_MAX_SEQS;
      const int x0 = k * c, x1 = min(x0 + c, T);
      if (x1 - x0 <= kPrefetchPages) {
        next = fetch();
        fetched = true;
      }
      for (int x = x0; x < x1;) {
        const int b = find_seq(incl, B, x);
        const int excl = b > 0 ? incl[b - 1] : 0;
        const int h = x - excl >= ((incl[b] - excl) >> 1) ? 1 : 0;
        int s0, n, so, no;
        stream_in_die(b, h, d, s0, n);
        stream_in_die(b, h, 1 - d, so, no);
        const int xe = min(x1, s0 + n);
        Seg sg;
        sg.b = b;
        sg.h = h;
        sg.p0 = x - s0;
        sg.p1 = xe - s0;
        sg.kv_len = s_kv[b];
        sg.q_len = static_cast<uint32_t>(s_qi[b]) >> 28;
        sg.row0 = s_qi[b] & 0x0FFFFFFF;
        sg.npieces = pieces_in_die(s0, n, d) + pieces_in_die(so, no, 1 - d);
        sg.slot = sg.npieces == 1 ? -1 : slot_of(d, k, b, h);
        const bool striped = prm.sm_die && prm.die_split == 0;
        sg.pstride = striped ? 2 : 1;
        sg.poff = striped ? d : 0;
        push(sg);
        const int32_t* const pt = prm.page_table + static_cast<int64_t>(b) * prm.pt_stride;
        for (int pb = sg.p0; pb < sg.p1; pb += 32) {
          const int n = min(32, sg.p1 - pb);
          const int id = lane < n ? __ldg(pt + sg.pstride * (pb + lane) + sg.poff) : 0;
          for (int t = 0; t < n; ++t) {
            const int page = __shfl_sync(0xffffffffu, id, t);
            issue(h, page, 0);  // K: column chunks 0, 1 of the token row
            issue(h, page, 2);  // V: column chunks 2, 3
            ++npages;
            if (!fetched && x1 - (s0 + pb + t) <= kPrefetchPages + 1) {  // (pb, t: die-local page index)
              next = fetch();
              fetched = true;
            }
          }
        }
        x = xe;
      }
    }
    Seg end{};
    end.b = -1;
    push(end);
    if (trace && lane == 0) {
      trace[3] = globaltimer();
      trace[5] = static_cast<uint64_t>(npages) | (static_cast<uint64_t>(seg_n - 1) << 32);
    }
  } else if (warp == kMmaWarp) {
    // ---- MMA issuer warp (lane 0) -----------------------------------------------------------------------------------
    if (lane == 0) {
      int seg_n = 0, g = 0;
      uint32_t qph = 0;
      auto pv = [&](int gp, bool first) {
        const int u = 2 * gp + 1, s = u % kStages;
        q36k::mbar_wait(bar(kBarFull + s), (u / kStages) & 1);
        q36k::mbar_wait(bar(kBarPFull + (gp & 1)), (gp >> 1) & 1);
        tc_fence_after();
        const uint32_t sV = sb + kSmStage + s * kStageBytes;
#pragma unroll
        for (int kb = 0; kb < 2; ++kb)
          umma_ts(tmem + kTmemO, tmem + kTmemP + 32 * (gp & 1) + 16 * kb, sdesc(sV + kb * 8192, 16384, 1024), kIdescPV,
                  (first && kb == 0) ? 0u : 1u);
        umma_commit(bar(kBarEmpty + s));
        umma_commit(bar(kBarODone));
      };
      while (true) {
        const int i = seg_n % kSegRing;
        q36k::mbar_wait(bar(kBarSegFull + i), (seg_n / kSegRing) & 1);
        const Seg sg = s_seg[i];
        q36k::mbar_arrive(bar(kBarSegEmpty + i));
        ++seg_n;
        if (sg.b < 0) break;
        q36k::mbar_wait(bar(kBarQFull), qph);
        qph ^= 1;
        const int np = sg.p1 - sg.p0;
        for (int j = 0; j < np; ++j, ++g) {
          const int u = 2 * g, s = u % kStages;
          q36k::mbar_wait(bar(kBarFull + s), (u / kStages) & 1);
          if (g > 0) q36k::mbar_wait(bar(kBarSFree), (g - 1) & 1);  // the softmax warps hold S(g - 1) in registers
          tc_fence_after();
          const uint32_t sK = sb + kSmStage + s * kStageBytes;
#pragma unroll
          for (int kb = 0; kb < 4; ++kb)
            umma_ts(tmem + kTmemS, tmem + kTmemQ + 16 * kb, sdesc(sK + (kb >> 1) * 16384 + (kb & 1) * 64, 16, 1024),
                    kIdescQK, kb > 0 ? 1u : 0u);
          umma_commit(bar(kBarEmpty + s));
          umma_commit(bar(kBarSFull));
          if (j > 0) pv(g - 1, j == 1);
        }
        pv(g - 1, np == 1);
        umma_commit(bar(kBarOFinal));
      }
    }
    __syncwarp();
  } else {
    // ---- Softmax / epilogue / merge warps ---------------------------------------------------------------------------
    const int qd = warp & 3, hf = warp >> 2;
    const int tl = 32 * qd + lane;  // TMEM lane of this thread's S row
    const int i_row = lane & 7, lo = (lane >> 3) & 1, gq = lane >> 4, jh = 2 * qd + gq;
    const uint32_t sel_hi = lo ? 0u : 1u;
    float* const s_mx = reinterpret_cast<float*>(sm + kSmMx);
    float* const s_lp = reinterpret_cast<float*>(sm + kSmLp);
    int seg_n = 0, g = 0, nseg = 0;
    auto read_seg = [&]() {
      const int i = seg_n % kSegRing;
      q36k::mbar_wait(bar(kBarSegFull + i), (seg_n / kSegRing) & 1);
      const Seg sg = s_seg[i];
      __syncwarp();
      if (lane == 0) q36k::mbar_arrive(bar(kBarSegEmpty + i));
      ++seg_n;
      return sg;
    };
    // Q of a segment into TMEM (the A operand of QK): this thread's MMA row (query i_row of head jh; the hi and the
    // lo copy alike), bytes [128 hf, +128) = columns kTmemQ + 32 hf + [0, 32). Padding rows (i_row >= q_len) are 0.
    auto q_src = [&](const Seg& sg) {
      const int qi = i_row < sg.q_len ? i_row : 0;
      return prm.q + (static_cast<int64_t>(sg.row0 + qi) * kNumQHeads + sg.h * kGroup + jh) * kHeadDim + 128 * hf;
    };
    auto load_q = [&](const Seg& sg) {
      const uint4* const src = reinterpret_cast<const uint4*>(q_src(sg));
      const bool ok = i_row < sg.q_len;
      uint32_t v[32];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const uint4 x = ok ? __ldg(src + c) : make_uint4(0, 0, 0, 0);
        v[4 * c] = x.x;
        v[4 * c + 1] = x.y;
        v[4 * c + 2] = x.z;
        v[4 * c + 3] = x.w;
      }
      tmem_st32(tmem + (static_cast<uint32_t>(32 * qd) << 16) + kTmemQ + 32 * hf, v);
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) q36k::mbar_arrive(bar(kBarQFull));
    };

    Seg cur = read_seg();
    if (cur.b >= 0) load_q(cur);
    while (cur.b >= 0) {
      const int np = cur.p1 - cur.p0;
      const int np_stream = (cur.kv_len + kPageSize - 1) / kPageSize;
      const bool live = i_row < cur.q_len;
      const int lim = min(cur.kv_len - cur.q_len + i_row, cur.kv_len - 1);  // last key this row sees
      float m_ref = -INFINITY, l_sum = 0.f;
      Seg nxt;
      nxt.b = -1;
      for (int jj = 0; jj < np; ++jj, ++g) {
        const int p = cur.pstride * (cur.p0 + jj) + cur.poff, buf = g & 1;
        const bool last = jj == np - 1;
        q36k::mbar_wait(bar(kBarSFull), g & 1);
        tc_fence_after();
        float s[64], mx;
        tmem_ld64_max(tmem + (static_cast<uint32_t>(32 * qd) << 16) + kTmemS + hf * 64, s, mx);
        tc_fence_before();
        __syncwarp();
        if (lane == 0) q36k::mbar_arrive(bar(kBarSFree));  // QK of the next page may overwrite S
        if (last) {
          // S-full of the segment's last page: its QK MMAs are done, Q may be replaced once the softmax below is
          // done; bring the next segment's Q rows towards L2 meanwhile.
          nxt = read_seg();
          if (nxt.b >= 0) asm volatile("prefetch.global.L2 [%0];" ::"l"(q_src(nxt)));
        }
        // own 32 tokens: hi lane [0, 32) of the half, lo lane [32, 64)
#pragma unroll
        for (int t = 0; t < 32; ++t) s[t] = lo ? s[32 + t] : s[t];
        const int tok0 = p * kPageSize + hf * 64;
        if (tok0 + 63 > cur.kv_len - cur.q_len) {  // tail page (warp uniform): mask, then the maximum in software
          const int tb = tok0 + 32 * lo;
          float m = -INFINITY;
#pragma unroll
          for (int t = 0; t < 32; ++t) {
            if (tb + t > lim) s[t] = -INFINITY;
            m = fmaxf(m, s[t]);
          }
          mx = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
        }
        s_mx[(buf * 2 + hf) * 128 + tl] = mx;
        q36k::bar_sync(1 + qd, 64);
        const float m_blk = fmaxf(mx, s_mx[(buf * 2 + (1 - hf)) * 128 + tl]) * kScaleLog2;
        const bool need = m_blk > m_ref + kTau;
        float alpha = 1.f;
        if (need) {
          alpha = ex2(m_ref - m_blk);
          m_ref = m_blk;
        }
        l_sum *= alpha;
        if (jj > 0 && __any_sync(0xffffffffu, need && live)) {
          // rescale this warp's O columns [hf * 128, +128) once P V of the previous page is done
          q36k::mbar_wait(bar(kBarODone), (g - 1) & 1);
          tc_fence_after();
#pragma unroll 1
          for (int cc = 0; cc < 2; ++cc) {
            const uint32_t ta = tmem + (static_cast<uint32_t>(32 * qd) << 16) + kTmemO + hf * 128 + cc * 64;
            uint32_t o[64];
            tmem_ld64(ta, o);
#pragma unroll
            for (int e = 0; e < 64; ++e) o[e] = __float_as_uint(__uint_as_float(o[e]) * alpha);
            tmem_st64(ta, o);
          }
          tmem_wait_st();
        }
        const float off = kPOff - (m_ref == -INFINITY ? 0.f : m_ref);
        uint32_t w[16];
        float l0 = 0.f, l1 = 0.f, l2 = 0.f, l3 = 0.f;
#pragma unroll
        for (int q8 = 0; q8 < 8; ++q8) {
          const float x0 = ex2(fmaf(s[4 * q8], kScaleLog2, off));
          const float x1 = ex2(fmaf(s[4 * q8 + 1], kScaleLog2, off));
          const float x2 = ex2(fmaf(s[4 * q8 + 2], kScaleLog2, off));
          const float x3 = ex2(fmaf(s[4 * q8 + 3], kScaleLog2, off));
          l0 += x0;
          l1 += x1;
          l2 += x2;
          l3 += x3;
          uint32_t own, send;
          pack_own_send(x0, x1, x2, x3, sel_hi, own, send);
          const uint32_t recv = __shfl_xor_sync(0xffffffffu, send, 8);
          // hi lane: [own(0-31) | recv(32-63)] (P_hi); lo lane: [recv(0-31) | own(32-63)] (P_lo)
          w[q8] = lo ? recv : own;
          w[8 + q8] = lo ? own : recv;
        }
        l_sum += (l0 + l1) + (l2 + l3);
        tmem_st16(tmem + (static_cast<uint32_t>(32 * qd) << 16) + kTmemP + 32 * buf + hf * 16, w);
        tmem_wait_st();
        if (p == np_stream - 1 && (cur.kv_len & (kPageSize - 1))) {
          // last page of the stream: zero the V rows past kv_len (whatever bytes they hold, P = 0 there)
          const int u = 2 * g + 1, vs = u % kStages;
          q36k::mbar_wait(bar(kBarFull + vs), (u / kStages) & 1);
          const int r0 = cur.kv_len & (kPageSize - 1), nz = 2 * (kPageSize - r0) * 8;
          uint8_t* const v = sm + kSmStage + vs * kStageBytes;
          for (int e = tid; e < nz; e += kSmThreads) {
            const int ch = e / ((kPageSize - r0) * 8), rem = e % ((kPageSize - r0) * 8);
            *reinterpret_cast<uint4*>(v + ch * 16384 + (r0 + rem / 8) * 128 + (rem & 7) * 16) = make_uint4(0, 0, 0, 0);
          }
          fence_proxy_async();
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) q36k::mbar_arrive(bar(kBarPFull + buf));
        if (last && nxt.b >= 0) load_q(nxt);
      }

      // ---- epilogue: O = O_hi + O_lo / 16, normalized (whole stream) or as a partial
      s_lp[hf * 128 + tl] = l_sum;
      q36k::bar_sync(1 + qd, 64);
      q36k::mbar_wait(bar(kBarOFinal), nseg & 1);
      ++nseg;
      tc_fence_after();
      const int ei = lane >> 2, ec = 2 * (lane & 3);
      const bool split = cur.slot >= 0;
      float* const pslot = prm.part + static_cast<int64_t>(split ? cur.slot : 0) * kSlotFloats;
#pragma unroll
      for (int gg = 0; gg < 2; ++gg) {
        const int hl = 32 * qd + 16 * gg + ei;  // hi lane of row (query ei, head 2 qd + gg)
        const float ltot = (s_lp[hl] + s_lp[hl + 8]) + (s_lp[128 + hl] + s_lp[128 + hl + 8]);
        const float inv = ltot > 0.f ? 1.f / ltot : 0.f;
        const bool ok = ei < cur.q_len;
        const int hq = cur.h * kGroup + 2 * qd + gg;
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          float o[16];
          tmem_ld_hilo(tmem + (static_cast<uint32_t>(32 * qd + 16 * gg) << 16) + kTmemO + hf * 128 + cc * 64, o);
          if (ok) {
            const int col0 = hf * 128 + cc * 64 + ec;
            if (split) {
              float* const dst = pslot + (ei * kGroup + 2 * qd + gg) * kHeadDim + col0;
#pragma unroll
              for (int k = 0; k < 8; ++k) *reinterpret_cast<float2*>(dst + 8 * k) = make_float2(o[2 * k], o[2 * k + 1]);
            } else {
              __nv_bfloat16* const dst =
                  prm.out + (static_cast<int64_t>(cur.row0 + ei) * kNumQHeads + hq) * kHeadDim + col0;
#pragma unroll
              for (int k = 0; k < 8; ++k) {
                *reinterpret_cast<__nv_bfloat162*>(dst + 8 * k) =
                    __floats2bfloat162_rn(o[2 * k] * inv, o[2 * k + 1] * inv);
              }
            }
          }
        }
      }
      if (split) {
        if (hf == 0 && lo == 0 && live) {
          const float lt = (s_lp[tl] + s_lp[tl + 8]) + (s_lp[128 + tl] + s_lp[128 + tl + 8]);
          pslot[kRows * kHeadDim + i_row * kGroup + jh] = m_ref;
          pslot[kRows * kHeadDim + kRows + i_row * kGroup + jh] = lt;
        }
        __threadfence();
        q36k::bar_sync(5, kSmThreads);
        if (tid == 0) {
          uint32_t* const cnt = prm.ctr + 32 + 2 * cur.b + cur.h;
          const uint32_t old = atomicAdd(cnt, 1u);
          if (old == static_cast<uint32_t>(cur.npieces - 1)) {  // last piece: append the stream to the merge list
            *cnt = 0;                                           // self-reset for the next launch
            const uint32_t i = atomicAdd(prm.ctr + 3, 1u);
            __threadfence();
            asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(prm.ctr + kCtrList + i),
                         "r"(static_cast<uint32_t>(2 * cur.b + cur.h + 1))
                         : "memory");
          }
        }
      }
      cur = nxt;
    }

    // ---- merges: take streams from the merge list until all of this launch's split streams are claimed (a
    // claimed entry not yet appended is waited for: its last piece is in flight on another CTA).
    const int nmerge = s_sc[kScNMerge];
    int* const s_flag = s_sc + kScFlag;
    while (true) {
      if (tid == 0) {
        int v = -1;
        const uint32_t i = atomicAdd(prm.ctr + 4, 1u);
        if (i < static_cast<uint32_t>(nmerge)) {
          uint32_t* const e = prm.ctr + kCtrList + i;
          uint32_t x;
          while (true) {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(x) : "l"(e) : "memory");
            if (x) break;
            __nanosleep(100);
          }
          *e = 0;  // self-reset for the next launch
          v = static_cast<int>(x) - 1;
          if (trace) trace[7] += 1;
        }
        *s_flag = v;
      }
      q36k::bar_sync(5, kSmThreads);
      const int v = *s_flag;
      if (v < 0) break;
      const int b = v >> 1, h = v & 1;
      int np = 0;
#pragma unroll
      for (int d = 0; d < 2; ++d) {
        int s0, n;
        stream_in_die(b, h, d, s0, n);
        np += pieces_in_die(s0, n, d);
      }
      __threadfence();
      merge_stream(prm, sm,
                   MergeSeg{b, h, static_cast<int>(static_cast<uint32_t>(s_qi[b]) >> 28), s_qi[b] & 0x0FFFFFFF, np},
                   tid);  // ends with a barrier: s_flag free
    }
  }

  // ---- teardown
  tc_fence_before();
  __syncthreads();
  if (warp == kMmaWarp) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(kTmemCols) : "memory");
  }
  if (trace && tid == 0) trace[4] = globaltimer();
  if (tid == 0) {
    // The last CTA out resets the chunk queues and the merge list counters for the next launch.
    __threadfence();
    if (atomicAdd(prm.ctr + 2, 1u) == gridDim.x - 1) {
      prm.ctr[0] = 0;
      prm.ctr[1] = 0;
      prm.ctr[2] = 0;
      prm.ctr[3] = 0;
      prm.ctr[4] = 0;
    }
  }
}

int sm_count() {
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  return sms;
}

// Partial slots: per die R_d + 2 x sequences (R_0 + R_1 <= chunks_per_sm x SMs).
size_t partial_bytes(int64_t num_seqs, int64_t chunks_per_sm, int sms) {
  return static_cast<size_t>(chunks_per_sm * sms + 4 * num_seqs) * kSlotFloats * sizeof(float);
}

}  // namespace

// ---- Workspace helpers & public C ABI -------------------------------------------------------------------------------

size_t attn_dl_workspace_bytes(int64_t max_seqs) {
  return kScratchOffset + partial_bytes(std::clamp<int64_t>(max_seqs, 1, Q36K_ATTN_MAX_SEQS),
                                        Q36K_ATTN_DL_MAX_CHUNKS_PER_SM, sm_count());
}

cudaError_t attn_dl_workspace_init(void* workspace, cudaStream_t stream) {
  return cudaMemsetAsync(static_cast<uint8_t*>(workspace) + kCtrOffset, 0, Q36K_ATTN_DL_COUNTER_BYTES, stream);
}

Q36K_API size_t q36k_attn_decode_dl_workspace_size(int64_t max_seqs) { return attn_dl_workspace_bytes(max_seqs); }

Q36K_API int q36k_attn_decode_dl(void* out, const void* q_fp8, const void* kv_cache, int64_t num_pages,
                                 const int32_t* page_table, int64_t page_table_stride, const int32_t* kv_lens,
                                 const int32_t* cu_seqlens_q, int64_t num_seqs, int64_t num_tokens, int64_t max_q_len,
                                 const uint8_t* sm_die, int64_t die_split_page, int64_t chunks_per_sm, int64_t steal,
                                 void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  if (num_seqs == 0) return Q36K_OK;
  if (!out || !q_fp8 || !kv_cache || !page_table || !kv_lens || !cu_seqlens_q || !workspace || num_seqs < 0 ||
      num_seqs > Q36K_ATTN_MAX_SEQS || num_pages < 1 || num_pages > INT_MAX || num_tokens < 1 ||
      num_tokens >= (1 << 28) || max_q_len < 1 || max_q_len > kMaxQ || page_table_stride < 1 ||
      page_table_stride > INT_MAX || chunks_per_sm < 1 || chunks_per_sm > Q36K_ATTN_DL_MAX_CHUNKS_PER_SM ||
      die_split_page < 0 || die_split_page >= num_pages || reinterpret_cast<uintptr_t>(kv_cache) % 128 ||
      reinterpret_cast<uintptr_t>(q_fp8) % 16 || reinterpret_cast<uintptr_t>(out) % 16 ||
      reinterpret_cast<uintptr_t>(workspace) % 256)
    return Q36K_ERR_INVALID_ARG;
  const int sms = sm_count();
  if (sms > Q36K_DIE_MAX_SMS) return Q36K_ERR_UNSUPPORTED;
  if (workspace_bytes < kScratchOffset + partial_bytes(num_seqs, chunks_per_sm, sms)) return Q36K_ERR_WORKSPACE;
  // KV pool as a 5-D tensor: [128 B column][token (512 B)][column chunk (128 B): K0 K1 V0 V1][kv head][page].
  CUtensorMap map;
  const cuuint64_t dims[5] = {128, static_cast<cuuint64_t>(kPageSize), 4, 2, static_cast<cuuint64_t>(num_pages)};
  const cuuint64_t strides[4] = {2 * kHeadDim, 128, static_cast<cuuint64_t>(kPageSize) * 2 * kHeadDim,
                                 static_cast<cuuint64_t>(Q36K_ATTN_PAGE_BYTES)};
  const cuuint32_t box[5] = {128, static_cast<cuuint32_t>(kPageSize), 2, 1, 1};
  const cuuint32_t es[5] = {1, 1, 1, 1, 1};
  if (cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, const_cast<void*>(kv_cache), dims, strides, box,
                             es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return Q36K_ERR_CUDA;
  static const bool attr_ok =
      cudaFuncSetAttribute(attn_dl_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmBytes) == cudaSuccess;
  if (!attr_ok) return Q36K_ERR_CUDA;
  Params prm{};
  prm.out = static_cast<__nv_bfloat16*>(out);
  prm.q = static_cast<const uint8_t*>(q_fp8);
  prm.page_table = page_table;
  prm.pt_stride = page_table_stride;
  prm.kv_lens = kv_lens;
  prm.cu_q = cu_seqlens_q;
  prm.sm_die = sm_die;
  prm.part = reinterpret_cast<float*>(static_cast<uint8_t*>(workspace) + kScratchOffset);
  prm.ctr = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(workspace) + kCtrOffset);
  prm.num_seqs = static_cast<int>(num_seqs);
  prm.chunks_per_sm = static_cast<int>(chunks_per_sm);
  prm.steal = steal ? 1 : 0;
  prm.die_split = static_cast<int>(die_split_page);
  // Q36K_DL_TRACE=1 (diagnostics): per-CTA timestamps into the workspace header at kTraceOffset.
  static const bool trace = getenv("Q36K_DL_TRACE") && atoi(getenv("Q36K_DL_TRACE")) > 0;
  prm.trace = trace ? reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(workspace) + kTraceOffset) : nullptr;
  q36k::PdlConfig c(dim3(sms), dim3(kThreads), kSmBytes, stream);
  const cudaError_t e = cudaLaunchKernelEx(&c.cfg, attn_dl_kernel, map, prm);
  return q36k::cuda_status(e);
}
'''
