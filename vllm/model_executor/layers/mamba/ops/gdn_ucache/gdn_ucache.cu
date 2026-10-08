// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// U-cache GDN MTP decode on qwen-v2: device helpers around the vendored FlashInfer #4081 u-cache
// kernel (vllm/third_party/flashinfer_gdn_ucache). Deferred-state prep, norm and materialization,
// with the page tag moved to a side table and a per-slot active bit.
//
// State model (per GDN layer). The block at a request's spec column 0 holds a CHECKPOINT S0; the history since S0
// lives in a per-request-slot ring (k: L2-normed key, u: delta-rule correction, g: cumulative log-decay since S0).
// The state after a window [start, start + cnt) of ring entries is
//   S = e^{G_last} S0 + sum_j e^{G_last - G_j} u_j k_j^T,   G_last = G_{start + cnt - 1}.
// Validity of the ring for block X and slot r (layer l):
//   active[r] != 0  &&  tags[l][X][vh] == uc_tag(cursor[l][r].count, r)
// active[r] is cleared when a new request takes slot r (MambaHybridModelState.add_request) and when a request leaves
// the u-cache band (prepass PRE_STOCK + CLEAR); ucache_prep sets it. The tag table is written by ucache_prep (the
// checkpoint block of the call) and cleared on every fold destination. A stale tag can only match the CURRENT count
// of the slot's cursor, which only the last call's checkpoint block carries.
//
// Kernels:
//   uc_prep_kernel   one CTA per decode row: ring cursor commit + tag + staging (T < 4 rows zero-padded when staged)
//   uc_norm_kernel   gated RMSNorm of the unfused outputs (rows with sidx < 0 get zeros; tokens t >= T untouched)
//   uc_fold_kernel   grid (rows, H, layers): materialize / copy for R1 (precopy), R2 (postprocess align) and the
//                    band-transition prepass (PRE_UC: stock -> u-cache, PRE_STOCK: u-cache -> stock)
//   uc_clear_kernel  active[idx_mapping[row]] = 0 for the prepass rows (after PRE_STOCK)
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace guc {

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kMaxT = 4;
constexpr int kUcMask = 31;
constexpr int kUcMaxWin = 20;  // P (<= 16) + accepted (<= 4)

__device__ __forceinline__ float sigmoid_fast(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu_fast(float x) { return x * sigmoid_fast(x); }
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

__device__ __forceinline__ int uc_tag(int count, int slot) { return ((count & 0x1FFFFF) << 10) | ((slot + 1) & 0x3FF); }

// [0] decode rows the u-cache cannot run (zero output), [1] decode rows with an active slot whose checkpoint tag did
// not match (history lost), [2] prepass PRE_STOCK rows with an active slot, no valid tag and acc > 1 (state lost),
// [3] decode rows started fresh, [4] bad ring slot, [5] strided row with cu_seqlens[row] != 4 row, [6] T out of range,
// [7] R1/R2 fold rows that fell back to the stock column copy (active slot, no valid tag: legal after an in-place R2)
__device__ unsigned long long guc_err[8];
__device__ int guc_first[8];  // first rejected decode row: {valid, row, T, bos, slot, block, n_rows, stage}

// ring window of block `blk` (value head vh) for `R` accepted tokens of the slot's last call
__device__ __forceinline__ bool uc_window(const int* tags, int64_t blk, int HV, int vh, int R, const int* cursor,
                                          const int* active, int slot, int slots, int& start, int& cnt) {
  if (slot < 0 || slot >= slots || active[slot] == 0) return false;
  const int tag = tags[blk * HV + vh];
  const int* cur = cursor + static_cast<int64_t>(slot) * 4;
  if (tag == 0 || tag != uc_tag(cur[3], slot)) return false;
  const int P = cur[0], base = cur[1], flushed = cur[2];
  start = flushed ? (base + P) & kUcMask : base;
  cnt = flushed ? R : P + R;
  cnt = cnt > kUcMaxWin ? kUcMaxWin : cnt;
  return true;
}

// ---------------------------------------------------------------------------------------------------------------
// ucache_prep
// ---------------------------------------------------------------------------------------------------------------
struct UcPrepArgs {
  const __nv_bfloat16* mixed_qkv;
  const __nv_bfloat16* a;
  const __nv_bfloat16* b;
  const int* state_indices;
  const int* cu_seqlens;
  const int* num_accepted;
  const int* slot_rows;
  int* cursor;
  int* tags;
  int* active;
  __nv_bfloat16 *qs, *ks, *vs, *as, *bs;
  int *sidx, *ridx, *hist, *base;
  int64_t mixed_row, a_row, b_row;
  int si_width, max_slots, H, HV, flush_min;
  int stage;  // 1: copy the rows' tokens into the compact staging (T in 1..4, zero-padded); 0: strided (T == 4,
              // cu_seqlens[row] == 4 row)
};

// Cursor commit (caller-owned, ReplaySSM commit_gdn_replayssm_spec): with a valid ring, a row whose last call flushed
// slides base past the folded window and restarts at P = accepted; otherwise P += accepted. A row without a valid
// ring starts a fresh ring on its (full) checkpoint block: P = 0, base = 0. Identical to gsc uc_prep_kernel except
// the tag source (side table + active bit) and the zero-padded staging of T < 4 rows.
__global__ __launch_bounds__(256) void uc_prep_kernel(UcPrepArgs p) {
  const int row = blockIdx.x, tid = threadIdx.x;
  __shared__ int s_ok, s_bos, s_tag, s_T;
  __shared__ int* s_flag;
  if (tid == 0) {
    const int bos = p.cu_seqlens[row], eos = p.cu_seqlens[row + 1];
    const int T = eos - bos;
    const int bslot = p.state_indices[static_cast<int64_t>(row) * p.si_width];
    const int r = p.slot_rows[row];
    int ok = 0, P = 0, base = 0;
    if (T > 0 && bslot > 0) {
      const bool t_ok = p.stage ? (T <= kMaxT) : (T == kMaxT);
      const bool r_ok = r >= 0 && r < p.max_slots && r < 1023;
      if (!t_ok || !r_ok || (!p.stage && bos != kMaxT * row)) {
        atomicAdd(&guc_err[0], 1ull);
        atomicAdd(&guc_err[!t_ok ? 6 : (!r_ok ? 4 : 5)], 1ull);
        if (atomicCAS(&guc_first[0], 0, 1) == 0) {
          guc_first[1] = row; guc_first[2] = T; guc_first[3] = bos; guc_first[4] = r;
          guc_first[5] = bslot; guc_first[6] = gridDim.x; guc_first[7] = p.stage;
        }
      } else {
        ok = 1;
        int* flag = p.tags + static_cast<int64_t>(bslot) * p.HV;
        const int tag = flag[0];
        int* cur = p.cursor + static_cast<int64_t>(r) * 4;
        const int cP = cur[0], cB = cur[1], cF = cur[2], cC = cur[3];
        int acc = p.num_accepted[row];
        acc = acc < 0 ? 0 : (acc > kMaxT ? kMaxT : acc);
        if (p.active[r] != 0 && tag != 0 && tag == uc_tag(cC, r)) {
          if (cF) { base = (cB + cP) & kUcMask; P = acc; }
          else { base = cB; P = cP + acc; }
        } else {
          atomicAdd(&guc_err[(p.active[r] != 0 && tag != 0) ? 1 : 3], 1ull);
        }
        const int nC = cC + 1;
        cur[0] = P; cur[1] = base; cur[2] = P >= p.flush_min ? 1 : 0; cur[3] = nC;
        p.active[r] = 1;
        s_tag = uc_tag(nC, r);
        s_flag = flag;
      }
    }
    p.sidx[row] = ok ? bslot : -1;
    p.ridx[row] = ok ? r : 0;
    p.hist[row] = P;
    p.base[row] = base;
    s_ok = ok;
    s_bos = bos;
    s_T = T;
  }
  __syncthreads();
  if (!s_ok) return;
  if (tid < p.HV) s_flag[tid] = s_tag;
  if (!p.stage) return;
  const int bos = s_bos, T = s_T;
  const int qk16 = p.H * kDimK / 8, v16 = p.HV * kDimV / 8, per = 2 * qk16 + v16;
  const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
  for (int i = tid; i < kMaxT * per; i += blockDim.x) {
    const int t = i / per, c = i % per;
    const uint4 x = t < T ? *reinterpret_cast<const uint4*>(p.mixed_qkv + static_cast<int64_t>(bos + t) * p.mixed_row + c * 8)
                          : z4;
    const int64_t rt = static_cast<int64_t>(row) * kMaxT + t;
    if (c < qk16) reinterpret_cast<uint4*>(p.qs + rt * p.H * kDimK)[c] = x;
    else if (c < 2 * qk16) reinterpret_cast<uint4*>(p.ks + rt * p.H * kDimK)[c - qk16] = x;
    else reinterpret_cast<uint4*>(p.vs + rt * p.HV * kDimV)[c - 2 * qk16] = x;
  }
  const __nv_bfloat16 zb = __float2bfloat16(0.0f);
  for (int i = tid; i < kMaxT * p.HV; i += blockDim.x) {
    const int t = i / p.HV, h = i % p.HV;
    const int64_t o = (static_cast<int64_t>(row) * kMaxT + t) * p.HV + h;
    p.as[o] = t < T ? p.a[static_cast<int64_t>(bos + t) * p.a_row + h] : zb;
    p.bs[o] = t < T ? p.b[static_cast<int64_t>(bos + t) * p.b_row + h] : zb;
  }
}

// Gated RMSNorm of the u-cache outputs (gsc uc_norm_kernel, verbatim): out[token][head] =
// bf16(o * rsqrt(mean(o^2) + eps) * w * gate(z)); rows the u-cache did not run (sidx < 0) get zeros.
template <bool SigmoidGate>
__global__ __launch_bounds__(256) void uc_norm_kernel(const __nv_bfloat16* __restrict__ o, const int* __restrict__ cu,
                                                      const int* __restrict__ sidx,
                                                      const __nv_bfloat16* __restrict__ gate, int64_t gate_row,
                                                      const void* __restrict__ w, bool w_bf16,
                                                      __nv_bfloat16* __restrict__ out, int HV, float eps) {
  const int row = blockIdx.x / kMaxT, t = blockIdx.x % kMaxT;
  const int lane = threadIdx.x & 31, head = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (head >= HV) return;
  const int bos = cu[row], T = cu[row + 1] - bos;
  if (t >= T) return;
  const int token = bos + t;
  __nv_bfloat16* dst = out + (static_cast<int64_t>(token) * HV + head) * kDimV;
  if (sidx[row] < 0 || t >= kMaxT) {
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[lane + i * 32] = __float2bfloat16(0.0f);
    return;
  }
  const __nv_bfloat16* src = o + ((static_cast<int64_t>(row) * kMaxT + t) * HV + head) * kDimV;
  float v[4];
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = __bfloat162float(src[lane + i * 32]);
    ss += v[i] * v[i];
  }
  ss = warp_reduce_sum(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(kDimV) + eps);
  const __nv_bfloat16* g = gate + static_cast<int64_t>(token) * gate_row + head * kDimV;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c = lane + i * 32;
    const float gi = __bfloat162float(g[c]);
    const float gv = SigmoidGate ? sigmoid_fast(gi) : silu_fast(gi);
    const float wt = w_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(w)[c]) : static_cast<const float*>(w)[c];
    dst[c] = __float2bfloat16(v[i] * rstd * wt * gv);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// fold / copy (R1, R2, prepass)
// ---------------------------------------------------------------------------------------------------------------
enum FoldMode : int { kR1 = 1, kR2 = 2, kPreUc = 3, kPreStock = 4 };

struct FoldArgs {
  int mode;
  int num_rows;
  // per batch row / request (V2 runner: batch row -> request-state slot)
  const int64_t* idx_mapping;  // [rows] (-1: skip); int64 as the V2 runner's InputBatch.idx_mapping
  const int* state_idx;        // [max_reqs] R1: post-advance dst column; R2: running (src) column
  const int* src_col;          // [max_reqs] R1 only: pre-advance src column (-1 fresh)
  const int* src_off;          // [max_reqs] R1 only: accepted-token bias (num_accepted - 1, pre-reset)
  const int* num_accepted;     // [max_reqs] R2 / prepass (by request slot)
  const int* num_computed;     // [max_reqs] R2: post-step new_num_computed (V2 PRECOMPUTED_NEW_COMPUTED)
  const int* seq_lens;         // [rows] prepass: batch-order seq_lens (spec column 0 = (seq_len - 1) / block_size)
  int block_size;
  // per KV-cache group block tables (int32 [max_reqs, max_blocks], batch order)
  const int64_t* bt_ptrs;
  int64_t bt_stride;
  // per layer
  const int64_t* layer_state;  // ssm pool base (bf16 [blocks, HV, V, K], block stride = slot_stride elements)
  const int* layer_group;
  const int64_t* layer_kr;
  const int64_t* layer_ur;
  const int64_t* layer_gr;
  const int64_t* layer_cur;
  const int64_t* layer_tags;
  int64_t slot_stride;
  int H, HV;
  const int* active;           // [slots]
  int slots;
};

__device__ __forceinline__ void copy_head(__nv_bfloat16* dh, const __nv_bfloat16* sh) {
  for (int i = threadIdx.x; i < kDimV * kDimK / 8; i += kThreads)
    reinterpret_cast<uint4*>(dh)[i] = reinterpret_cast<const uint4*>(sh)[i];
}

__device__ __forceinline__ float ring_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float ring_f(__half x) { return __half2float(x); }

// dst head = fold(src checkpoint head, ring window [start, start + cnt)): gsc mat_item_uc arithmetic, verbatim.
// RT = ring element type: bf16, or fp16 with GDN_UCACHE_RING_DTYPE=fp16 (the fold form is unchanged:
// with VLLM_GDN_UCACHE_KQFIX the k ring holds the raw key and the u ring u * inv|k|).
template <typename RT>
__device__ __forceinline__ void fold_head(__nv_bfloat16* dh, const __nv_bfloat16* sh, const RT* kr,
                                          const RT* ur, const float* gr, int start, int cnt,
                                          float (*s_k)[kDimK], float* s_g) {
  const int tid = threadIdx.x;
  for (int i = tid; i < cnt * kDimK; i += kThreads) {
    const int j = i / kDimK, c = i % kDimK;
    s_k[j][c] = ring_f(kr[((start + j) & kUcMask) * kDimK + c]);
  }
  if (tid < cnt) s_g[tid] = gr[(start + tid) & kUcMask];
  __syncthreads();
  const float g_last = s_g[cnt - 1];
  const float bdec = expf(g_last);
  const int row = tid >> 1, c0 = (tid & 1) * (kDimK / 2);
  float cj[kUcMaxWin];
#pragma unroll
  for (int j = 0; j < kUcMaxWin; ++j)
    cj[j] = j < cnt ? expf(g_last - s_g[j]) * ring_f(ur[((start + j) & kUcMask) * kDimV + row]) : 0.0f;
  for (int c = c0; c < c0 + kDimK / 2; c += 8) {
    const uint4 w = *reinterpret_cast<const uint4*>(sh + row * kDimK + c);
    const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
    float acc[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      acc[2 * e] = bdec * __uint_as_float(wv[e] << 16);
      acc[2 * e + 1] = bdec * __uint_as_float(wv[e] & 0xffff0000u);
    }
    for (int j = 0; j < cnt; ++j) {
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = fmaf(cj[j], s_k[j][c + e], acc[e]);
    }
    uint4 o;
    uint32_t* ov = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const __nv_bfloat162 pr = __floats2bfloat162_rn(acc[2 * e], acc[2 * e + 1]);
      ov[e] = *reinterpret_cast<const uint32_t*>(&pr);
    }
    *reinterpret_cast<uint4*>(dh + row * kDimK + c) = o;
  }
}

template <typename RT>
__global__ __launch_bounds__(kThreads, 2) void uc_fold_kernel(FoldArgs args) {
  const int brow = blockIdx.x, key_head = blockIdx.y, layer = blockIdx.z;
  const int tid = threadIdx.x;
  if (brow >= args.num_rows) return;
  const int req = static_cast<int>(args.idx_mapping[brow]);
  if (req < 0 || req >= args.slots) return;
  const int* bt = reinterpret_cast<const int*>(args.bt_ptrs[args.layer_group[layer]]) + brow * args.bt_stride;
  // ---- per-row decision (layer-independent; mirrors the stock Triton kernels)
  int src_col = -1, dst_col = -1, n = 0, fb_col = -1;  // fb_col: stock copy source column if the ring is not valid
  const bool act = args.active[req] != 0;
  if (args.mode == kR1) {
    if (!act) return;  // stock precopy handles inactive slots
    src_col = args.src_col[req];
    dst_col = args.state_idx[req];
    if (src_col < 0 || src_col == dst_col) return;
    n = args.src_off[req] + 1;
    fb_col = src_col + args.src_off[req];
  } else if (args.mode == kR2) {
    if (!act) return;  // stock postprocess handles inactive slots
    const int acc = args.num_accepted[req];
    const int new_computed = args.num_computed[req];
    const int running = new_computed - acc + 1;
    const int aligned = (new_computed / args.block_size) * args.block_size;
    if (aligned < running) return;
    const int bias = aligned - running;
    src_col = args.state_idx[req];
    dst_col = aligned / args.block_size - 1;
    n = bias + 1;
    // in place with bias 0: the stock path is a no-op, but the published block must hold the full state
    fb_col = (src_col == dst_col && bias == 0) ? -1 : src_col + bias;
  } else {
    const int seq_len = args.seq_lens[brow];
    const int start = seq_len > 0 ? (seq_len - 1) / args.block_size : 0;
    const int acc = args.num_accepted[req];
    if (args.mode == kPreUc) {  // stock layout -> u-cache: the state of the last accepted token into column 0
      if (act || acc <= 1) return;
      src_col = start + acc - 1;
      dst_col = start;
    } else {  // kPreStock: u-cache -> stock layout: materialize column acc - 1
      if (!act) return;
      src_col = start;
      dst_col = start + acc - 1;
      n = acc;
    }
  }
  const int src = bt[src_col], dst = bt[dst_col];
  if (src <= 0 || dst <= 0) return;
  const int VPK = args.HV / args.H;
  __nv_bfloat16* state = reinterpret_cast<__nv_bfloat16*>(args.layer_state[layer]);
  const int64_t hstride = static_cast<int64_t>(kDimV) * kDimK;
  int* tags = reinterpret_cast<int*>(args.layer_tags[layer]);
  const int* cursor = reinterpret_cast<const int*>(args.layer_cur[layer]);
  __shared__ __align__(16) float s_k[kUcMaxWin][kDimK];
  __shared__ float s_g[kUcMaxWin];
  for (int p = 0; p < VPK; ++p) {
    const int vh = key_head * VPK + p;
    __nv_bfloat16* dh = state + static_cast<int64_t>(dst) * args.slot_stride + vh * hstride;
    if (args.mode == kPreUc) {
      copy_head(dh, state + static_cast<int64_t>(src) * args.slot_stride + vh * hstride);
      continue;
    }
    int start = 0, cnt = 0;
    const bool valid = uc_window(tags, src, args.HV, vh, n, cursor, args.active, req, args.slots, start, cnt);
    if (valid && cnt > 0) {
      const RT* kr = reinterpret_cast<const RT*>(args.layer_kr[layer]) +
                     (static_cast<int64_t>(req) * args.H + key_head) * 32 * kDimK;
      const RT* ur = reinterpret_cast<const RT*>(args.layer_ur[layer]) + (static_cast<int64_t>(req) * args.HV + vh) * 32 * kDimV;
      const float* gr = reinterpret_cast<const float*>(args.layer_gr[layer]) + (static_cast<int64_t>(req) * args.HV + vh) * 32;
      fold_head(dh, state + static_cast<int64_t>(src) * args.slot_stride + vh * hstride, kr, ur, gr, start, cnt, s_k, s_g);
    } else if (valid) {  // empty window (n == 0 cannot happen; kept for safety): plain copy
      if (src != dst) copy_head(dh, state + static_cast<int64_t>(src) * args.slot_stride + vh * hstride);
    } else if (args.mode == kPreStock) {
      // no valid ring: the state is already stock layout at column 0 (fresh page, acc reset to 1)
      if (n > 1 && tid == 0 && key_head == 0 && p == 0 && layer == 0) atomicAdd(&guc_err[2], 1ull);
      __syncthreads();
      continue;
    } else {
      if (tid == 0 && key_head == 0 && p == 0 && layer == 0) atomicAdd(&guc_err[7], 1ull);
      if (fb_col >= 0) {
        const int fb = bt[fb_col];
        if (fb > 0 && fb != dst) copy_head(dh, state + static_cast<int64_t>(fb) * args.slot_stride + vh * hstride);
      }
    }
    __syncthreads();  // s_k / s_g reuse; every read of this head's source done before its tag changes
    if (tid == 0) tags[static_cast<int64_t>(dst) * args.HV + vh] = 0;
  }
}

__global__ void uc_clear_kernel(const int64_t* idx_mapping, int num_rows, int* active, int slots) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_rows) return;
  const int64_t r = idx_mapping[i];
  if (r >= 0 && r < slots) active[r] = 0;
}

// ---------------------------------------------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------------------------------------------
void ucache_prep(torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b, torch::Tensor state_indices,
                 torch::Tensor cu_seqlens, torch::Tensor num_accepted, torch::Tensor slot_rows, torch::Tensor cursor,
                 torch::Tensor tags, torch::Tensor active, int64_t HV, torch::Tensor qs, torch::Tensor ks,
                 torch::Tensor vs, torch::Tensor as, torch::Tensor bs, torch::Tensor sidx, torch::Tensor ridx,
                 torch::Tensor hist, torch::Tensor base, int64_t flush_min, bool stage) {
  const int n = state_indices.size(0);
  if (n == 0) return;
  const int H = static_cast<int>((mixed_qkv.size(1) - HV * kDimV) / (2 * kDimK));
  TORCH_CHECK(mixed_qkv.scalar_type() == at::kBFloat16 && mixed_qkv.stride(1) == 1 && mixed_qkv.stride(0) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(mixed_qkv.data_ptr()) % 16 == 0, "mixed_qkv rows must be 16-B aligned");
  TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1);
  TORCH_CHECK(state_indices.scalar_type() == at::kInt && state_indices.is_contiguous());
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && num_accepted.scalar_type() == at::kInt);
  TORCH_CHECK(slot_rows.scalar_type() == at::kInt && slot_rows.numel() >= n);
  TORCH_CHECK(cursor.scalar_type() == at::kInt && cursor.is_contiguous() && cursor.dim() == 2 && cursor.size(1) == 4);
  TORCH_CHECK(tags.scalar_type() == at::kInt && tags.is_contiguous() && tags.dim() == 2 && tags.size(1) == HV);
  TORCH_CHECK(active.scalar_type() == at::kInt && active.numel() >= cursor.size(0));
  TORCH_CHECK(qs.size(0) >= n && vs.size(0) >= n && sidx.numel() >= n, "u-cache staging buffers too small");
  TORCH_CHECK(flush_min >= 1 && flush_min <= 13);
  UcPrepArgs p;
  p.mixed_qkv = reinterpret_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr());
  p.a = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr());
  p.b = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr());
  p.state_indices = state_indices.data_ptr<int>();
  p.cu_seqlens = cu_seqlens.data_ptr<int>();
  p.num_accepted = num_accepted.data_ptr<int>();
  p.slot_rows = slot_rows.data_ptr<int>();
  p.cursor = cursor.data_ptr<int>();
  p.tags = tags.data_ptr<int>();
  p.active = active.data_ptr<int>();
  p.qs = reinterpret_cast<__nv_bfloat16*>(qs.data_ptr());
  p.ks = reinterpret_cast<__nv_bfloat16*>(ks.data_ptr());
  p.vs = reinterpret_cast<__nv_bfloat16*>(vs.data_ptr());
  p.as = reinterpret_cast<__nv_bfloat16*>(as.data_ptr());
  p.bs = reinterpret_cast<__nv_bfloat16*>(bs.data_ptr());
  p.sidx = sidx.data_ptr<int>();
  p.ridx = ridx.data_ptr<int>();
  p.hist = hist.data_ptr<int>();
  p.base = base.data_ptr<int>();
  p.mixed_row = mixed_qkv.stride(0);
  p.a_row = a.stride(0);
  p.b_row = b.stride(0);
  p.si_width = static_cast<int>(state_indices.size(1));
  p.max_slots = static_cast<int>(cursor.size(0));
  p.H = H;
  p.HV = static_cast<int>(HV);
  p.flush_min = static_cast<int>(flush_min);
  p.stage = stage ? 1 : 0;
  uc_prep_kernel<<<n, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ucache_norm(torch::Tensor o, torch::Tensor cu_seqlens, torch::Tensor sidx, torch::Tensor output_gate,
                 torch::Tensor norm_weight, torch::Tensor out, double eps, bool sigmoid_gate, int64_t n) {
  if (n == 0) return;
  const int HV = out.size(1);
  TORCH_CHECK(out.is_contiguous() && out.size(2) == kDimV && output_gate.stride(2) == 1 &&
              output_gate.stride(1) == kDimV);
  const dim3 grid(static_cast<unsigned>(n * kMaxT), (HV + 7) / 8);
  const bool wbf = norm_weight.scalar_type() == at::kBFloat16;
  auto st = c10::cuda::getCurrentCUDAStream();
  auto oq = reinterpret_cast<const __nv_bfloat16*>(o.data_ptr());
  auto g = reinterpret_cast<const __nv_bfloat16*>(output_gate.data_ptr());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if (sigmoid_gate)
    uc_norm_kernel<true><<<grid, 256, 0, st>>>(oq, cu_seqlens.data_ptr<int>(), sidx.data_ptr<int>(), g,
                                               output_gate.stride(0), norm_weight.data_ptr(), wbf, op, HV,
                                               static_cast<float>(eps));
  else
    uc_norm_kernel<false><<<grid, 256, 0, st>>>(oq, cu_seqlens.data_ptr<int>(), sidx.data_ptr<int>(), g,
                                                output_gate.stride(0), norm_weight.data_ptr(), wbf, op, HV,
                                                static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static const int* opt_i32(const c10::optional<torch::Tensor>& t) {
  if (!t.has_value()) return nullptr;
  TORCH_CHECK(t->scalar_type() == at::kInt && t->is_contiguous());
  return t->data_ptr<int>();
}

// mode: 1 R1, 2 R2, 3 PRE_UC, 4 PRE_STOCK. layer_ptrs int64 [6][L] = {state, kring, uring, gring, cursor, tags};
// layer_group int32 [L]; bt_ptrs int64 [G].
void ucache_fold(int64_t mode, int64_t num_rows, torch::Tensor idx_mapping, c10::optional<torch::Tensor> state_idx,
                 c10::optional<torch::Tensor> src_col, c10::optional<torch::Tensor> src_off,
                 c10::optional<torch::Tensor> num_accepted, c10::optional<torch::Tensor> num_computed,
                 c10::optional<torch::Tensor> seq_lens, int64_t block_size, torch::Tensor bt_ptrs, int64_t bt_stride,
                 torch::Tensor layer_ptrs, torch::Tensor layer_group, int64_t slot_stride, int64_t H, int64_t HV,
                 torch::Tensor active, bool ring_f16) {
  if (num_rows <= 0) return;
  const int L = static_cast<int>(layer_group.numel());
  TORCH_CHECK(layer_ptrs.scalar_type() == at::kLong && layer_ptrs.is_contiguous() && layer_ptrs.size(0) == 6 &&
              layer_ptrs.size(1) == L);
  TORCH_CHECK(bt_ptrs.scalar_type() == at::kLong && bt_ptrs.is_contiguous());
  TORCH_CHECK(idx_mapping.scalar_type() == at::kLong && idx_mapping.is_contiguous() && idx_mapping.numel() >= num_rows);
  TORCH_CHECK(HV % H == 0 && HV / H <= 8);
  FoldArgs f;
  f.mode = static_cast<int>(mode);
  f.num_rows = static_cast<int>(num_rows);
  f.idx_mapping = idx_mapping.data_ptr<int64_t>();
  f.state_idx = opt_i32(state_idx);
  f.src_col = opt_i32(src_col);
  f.src_off = opt_i32(src_off);
  f.num_accepted = opt_i32(num_accepted);
  f.num_computed = opt_i32(num_computed);
  f.seq_lens = opt_i32(seq_lens);
  TORCH_CHECK(mode != 1 || (f.state_idx && f.src_col && f.src_off), "R1 needs state_idx / src_col / src_off");
  TORCH_CHECK(mode != 2 || (f.state_idx && f.num_accepted && f.num_computed), "R2 needs state_idx / acc / computed");
  TORCH_CHECK(mode < 3 || (f.seq_lens && f.num_accepted), "prepass needs seq_lens / num_accepted");
  f.block_size = static_cast<int>(block_size);
  f.bt_ptrs = bt_ptrs.data_ptr<int64_t>();
  f.bt_stride = bt_stride;
  const int64_t* lp = layer_ptrs.data_ptr<int64_t>();
  f.layer_state = lp;
  f.layer_kr = lp + L;
  f.layer_ur = lp + 2 * L;
  f.layer_gr = lp + 3 * L;
  f.layer_cur = lp + 4 * L;
  f.layer_tags = lp + 5 * L;
  f.layer_group = layer_group.data_ptr<int>();
  f.slot_stride = slot_stride;
  f.H = static_cast<int>(H);
  f.HV = static_cast<int>(HV);
  f.active = active.data_ptr<int>();
  f.slots = static_cast<int>(active.numel());
  const dim3 grid(static_cast<unsigned>(num_rows), static_cast<unsigned>(H), static_cast<unsigned>(L));
  if (ring_f16) uc_fold_kernel<__half><<<grid, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(f);
  else uc_fold_kernel<__nv_bfloat16><<<grid, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ucache_clear(torch::Tensor idx_mapping, int64_t num_rows, torch::Tensor active) {
  if (num_rows <= 0) return;
  TORCH_CHECK(idx_mapping.scalar_type() == at::kLong && active.scalar_type() == at::kInt);
  uc_clear_kernel<<<(num_rows + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      idx_mapping.data_ptr<int64_t>(), static_cast<int>(num_rows), active.data_ptr<int>(),
      static_cast<int>(active.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor ucache_errors(bool reset) {
  auto r = torch::zeros({16}, torch::dtype(torch::kInt64));
  unsigned long long h[8];
  int f[8];
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(h, guc_err, sizeof(h)));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(f, guc_first, sizeof(f)));
  for (int i = 0; i < 8; ++i) r[i] = static_cast<int64_t>(h[i]);
  for (int i = 0; i < 8; ++i) r[8 + i] = f[i];
  if (reset) {
    const unsigned long long z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int zf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    C10_CUDA_CHECK(cudaMemcpyToSymbol(guc_err, z, sizeof(z)));
    C10_CUDA_CHECK(cudaMemcpyToSymbol(guc_first, zf, sizeof(zf)));
  }
  return r;
}

}  // namespace guc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ucache_prep", &guc::ucache_prep);
  m.def("ucache_norm", &guc::ucache_norm);
  m.def("ucache_fold", &guc::ucache_fold);
  m.def("ucache_clear", &guc::ucache_clear);
  m.def("ucache_errors", &guc::ucache_errors);
}
