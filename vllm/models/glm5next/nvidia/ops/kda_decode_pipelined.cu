// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3-Flash KDA single-token decode, one launch per layer:
//   depthwise causal conv (width 4) + SiLU on q|k|v, L2-normalised q/k,
//   bounded gate decay lb*sigmoid(exp(A_log)*(g + dt_bias)), sigmoid beta,
//   one gated delta-rule step on the recurrent state [slot, H, V=128, K=128]
//   (fp32 or bf16), and the gated RMSNorm  rms(o) * w * sigmoid(g2).
//
// The step is bound by streaming the recurrent state (read + write, 128 KB
// per (row, head) item in fp32). The kernel is persistent and warp
// specialised so the state stream never waits on per-item latency:
//   * producer warp: cp.async.bulk of each item's state, in 16 KB chunks, into
//     a ring of shared-memory slots (mbarrier transaction counts), running up
//     to `ring` chunks ahead;
//   * aux warp groups: per-item prologue (conv, norms, gates; writes the conv
//     state) and epilogue (gated RMSNorm, output), several items ahead;
//   * compute warps: per chunk, read state rows from shared memory, update
//     them, store them straight to global memory and emit o[v].
// Null rows (state index <= 0) produce a zero output and touch no cache.
// Every launch is self-contained (no host sync, no scratch), so it can be
// captured in CUDA graphs.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace glm5next_kda {

constexpr int kD = 128;
constexpr int kComputeWarps = 16;
#ifndef GLM_KDA_AUX_GROUPS
  #define GLM_KDA_AUX_GROUPS 2
#endif
constexpr int kAuxGroups = GLM_KDA_AUX_GROUPS;
constexpr int kAuxWarpsPerGroup = 4;
constexpr int kAuxThreads = kAuxWarpsPerGroup * 32;
constexpr int kNumBuf = 2 * kAuxGroups;
constexpr int kProducerWarp = kComputeWarps + kAuxGroups * kAuxWarpsPerGroup;
constexpr int kThreads = (kProducerWarp + 1) * 32;
constexpr int kChunkBytes = 16384;
constexpr int kMaxRing = 16;
// Default ring depth: VR200 (jobs 800580/802514) streams bf16 state fastest
// with 6 slots (96 KB in flight per SM; 13 slots was 1-8% slower), fp32 is
// flat between 6 and 13.
constexpr int kDefaultRing = 6;
#ifndef GLM_KDA_DEFAULT_FLAGS
  #define GLM_KDA_DEFAULT_FLAGS 0
#endif
constexpr int64_t kDefaultFlags = GLM_KDA_DEFAULT_FLAGS;

struct alignas(16) ItemBuf {
  float q[kD];       // conv+SiLU q, L2-normalised and scaled by D^-0.5
  float k[kD];       // conv+SiLU k, L2-normalised
  float decay[kD];   // exp(gate)
  float v[kD];       // conv+SiLU v
  float oscale[kD];  // norm_w * sigmoid(output gate)
  float o[kD];       // recurrence output (bf16-rounded), written by compute
  float beta;
  float qk;  // q_hat . k_hat
  int slot;
  int pad;
};

struct Params {
  const __nv_bfloat16* x;
  int64_t x_row;
  const float* conv_w;  // [3P, 4]
  __nv_bfloat16* conv_state;
  int64_t cs_slot, cs_ch, cs_t;
  const __nv_bfloat16* gate;
  int64_t gate_row;
  const __nv_bfloat16* beta;
  int64_t beta_row;
  const float* a_log;
  const float* dt_bias;
  const int32_t* idx;
  void* state;
  int64_t state_slot;
  const __nv_bfloat16* ogate;
  int64_t ogate_row;
  const float* norm_w;
  __nv_bfloat16* out;
  int64_t out_row;
  int heads;
  int P;  // heads * kD
  float lower_bound;
  float norm_eps;
  int items;
  int ring;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(count)
               : "memory");
}

__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar))
               : "memory");
}

// Releases a ring slot after the warp has consumed what it read from it:
// `dep` is a value computed from those shared-memory loads (via the warp
// shuffles), so the arrive cannot issue while a load is still outstanding.
// Without it the producer's next bulk copy could overwrite the slot under an
// in-flight ld.shared (seen as rare single-item corruption).
__device__ __forceinline__ void mbar_arrive_after(uint64_t* bar, float dep) {
  asm volatile(
      "{\n\t.reg .f32 d;\n\tmov.f32 d, %1;\n\t"
      "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t}" ::"r"(smem_u32(bar)),
      "f"(dep)
      : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar,
                                                      uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   smem_u32(bar)),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra DONE;\n\t"
      "bra LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void bulk_g2s(void* dst, const void* src,
                                         uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], %2, [%3];" ::"r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}

// Bulk shared -> global store (TMA engine), its group commit and waits.
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void bulk_s2g(void* dst, const void* src,
                                         uint32_t bytes) {
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst),
      "r"(smem_u32(src)), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

template <int N>
__device__ __forceinline__ void bulk_wait() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

constexpr int kComputeBarrier = 15;  // named barrier of the compute warps

__device__ __forceinline__ void named_bar_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ float sigmoidf(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float bf16_round(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

template <typename StateT>
struct StateIO;

template <>
struct StateIO<float> {
  __device__ __forceinline__ static void load(const float* p, float (&s)[4]) {
    float4 v = *reinterpret_cast<const float4*>(p);
    s[0] = v.x;
    s[1] = v.y;
    s[2] = v.z;
    s[3] = v.w;
  }
  __device__ __forceinline__ static void store(float* p, const float (&s)[4]) {
    *reinterpret_cast<float4*>(p) = make_float4(s[0], s[1], s[2], s[3]);
  }
};

template <>
struct StateIO<__nv_bfloat16> {
  __device__ __forceinline__ static void load(const __nv_bfloat16* p,
                                              float (&s)[4]) {
    uint2 raw = *reinterpret_cast<const uint2*>(p);
    __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&raw.x);
    __nv_bfloat162 b = *reinterpret_cast<__nv_bfloat162*>(&raw.y);
    float2 fa = __bfloat1622float2(a), fb = __bfloat1622float2(b);
    s[0] = fa.x;
    s[1] = fa.y;
    s[2] = fb.x;
    s[3] = fb.y;
  }
  __device__ __forceinline__ static void store(__nv_bfloat16* p,
                                               const float (&s)[4]) {
    __nv_bfloat162 a = __floats2bfloat162_rn(s[0], s[1]);
    __nv_bfloat162 b = __floats2bfloat162_rn(s[2], s[3]);
    uint2 raw;
    raw.x = *reinterpret_cast<uint32_t*>(&a);
    raw.y = *reinterpret_cast<uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p) = raw;
  }
};

struct SmemLayout {
  uint8_t* ring;
  ItemBuf* bufs;
  uint64_t* full_s;
  uint64_t* empty_s;
  uint64_t* pfull;
  uint64_t* pdone;
  float* red;  // [kAuxGroups][2 (pro/epi)][kAuxWarpsPerGroup][4]
};

__host__ __device__ inline size_t smem_bytes(int ring) {
  return static_cast<size_t>(ring) * kChunkBytes + kNumBuf * sizeof(ItemBuf) +
         (2 * ring + 2 * kNumBuf) * sizeof(uint64_t) +
         kAuxGroups * 2 * kAuxWarpsPerGroup * 4 * sizeof(float);
}

__device__ __forceinline__ SmemLayout carve(uint8_t* smem, int ring) {
  SmemLayout s;
  s.ring = smem;
  s.bufs = reinterpret_cast<ItemBuf*>(smem +
                                      static_cast<size_t>(ring) * kChunkBytes);
  s.full_s = reinterpret_cast<uint64_t*>(s.bufs + kNumBuf);
  s.empty_s = s.full_s + ring;
  s.pfull = s.empty_s + ring;
  s.pdone = s.pfull + kNumBuf;
  s.red = reinterpret_cast<float*>(s.pdone + kNumBuf);
  return s;
}

// ---------------------------------------------------------------- producer
template <typename StateT>
__device__ __forceinline__ void producer(const Params& p, const SmemLayout& sm,
                                         int first, int count) {
  constexpr int kChunks =
      kD * kD * static_cast<int>(sizeof(StateT)) / kChunkBytes;
  const StateT* st = static_cast<const StateT*>(p.state);
  int rs = 0;
  uint32_t rph = 0;
  for (int i = 0; i < count; ++i) {
    const int t = first + i;
    const int r = t / p.heads;
    const int h = t - r * p.heads;
    const int slot = p.idx[r];
    if (slot <= 0) continue;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(
        st + static_cast<int64_t>(slot) * p.state_slot +
        static_cast<int64_t>(h) * kD * kD);
#pragma unroll 1
    for (int c = 0; c < kChunks; ++c) {
      mbar_wait(&sm.empty_s[rs], rph ^ 1u);
      mbar_arrive_expect_tx(&sm.full_s[rs], kChunkBytes);
      bulk_g2s(sm.ring + static_cast<size_t>(rs) * kChunkBytes,
               src + c * kChunkBytes, kChunkBytes, &sm.full_s[rs]);
      if (++rs == p.ring) {
        rs = 0;
        rph ^= 1u;
      }
    }
  }
}

// ---------------------------------------------------------------- aux
// Each aux group (kAuxThreads threads) owns every kAuxGroups-th item of the
// CTA; thread `at` handles channels at, at + kAuxThreads, ... of q, k and v.
constexpr int kCpt = kD / kAuxThreads;  // channels per aux thread

__device__ __forceinline__ void aux_prologue(const Params& p,
                                             const SmemLayout& sm, int group,
                                             int at, int first, int i) {
  ItemBuf& B = sm.bufs[i % kNumBuf];
  const int t = first + i;
  const int r = t / p.heads;
  const int h = t - r * p.heads;
  const int slot = p.idx[r];
  // Issue every global load of the item up front (one memory round trip; the
  // conv-state address depends on the slot, null rows read the null slot 0).
  const int64_t rd_slot = slot > 0 ? slot : 0;
  float gv[kCpt], ov[kCpt], nw[kCpt], db[kCpt];
  float4 wv[kCpt][3];
  __nv_bfloat16 xraw[kCpt][3], hraw[kCpt][3][3];
  const float a_log = p.a_log[h];
  const float beta_raw = __bfloat162float(p.beta[r * p.beta_row + h]);
#pragma unroll
  for (int cc = 0; cc < kCpt; ++cc) {
    const int c = at + cc * kAuxThreads;
    gv[cc] = __bfloat162float(p.gate[r * p.gate_row + h * kD + c]);
    ov[cc] = __bfloat162float(p.ogate[r * p.ogate_row + h * kD + c]);
    nw[cc] = p.norm_w[c];
    db[cc] = p.dt_bias[h * kD + c];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int64_t ch =
          static_cast<int64_t>(j) * p.P + static_cast<int64_t>(h) * kD + c;
      xraw[cc][j] = p.x[r * p.x_row + ch];
      wv[cc][j] = *reinterpret_cast<const float4*>(p.conv_w + ch * 4);
      const __nv_bfloat16* cs =
          p.conv_state + rd_slot * p.cs_slot + ch * p.cs_ch;
#pragma unroll
      for (int tt = 0; tt < 3; ++tt) hraw[cc][j][tt] = cs[tt * p.cs_t];
    }
  }
  if (slot <= 0) {  // null row: zero output (epilogue), no cache write
    if (at == 0) B.slot = 0;
    mbar_arrive(&sm.pfull[i % kNumBuf]);
    return;
  }
  float y[kCpt][3];
  float sq = 0.f, sk = 0.f, sqk = 0.f;
#pragma unroll
  for (int cc = 0; cc < kCpt; ++cc) {
    const int c = at + cc * kAuxThreads;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int64_t ch =
          static_cast<int64_t>(j) * p.P + static_cast<int64_t>(h) * kD + c;
      const float4 w = wv[cc][j];
      const float acc = __bfloat162float(hraw[cc][j][0]) * w.x +
                        __bfloat162float(hraw[cc][j][1]) * w.y +
                        __bfloat162float(hraw[cc][j][2]) * w.z +
                        __bfloat162float(xraw[cc][j]) * w.w;
      y[cc][j] = bf16_round(acc * sigmoidf(acc));
      __nv_bfloat16* cs =
          p.conv_state + static_cast<int64_t>(slot) * p.cs_slot + ch * p.cs_ch;
      cs[0] = hraw[cc][j][1];
      cs[p.cs_t] = hraw[cc][j][2];
      cs[2 * p.cs_t] = xraw[cc][j];
    }
    sq += y[cc][0] * y[cc][0];
    sk += y[cc][1] * y[cc][1];
    sqk += y[cc][0] * y[cc][1];
  }
  sq = warp_sum(sq);
  sk = warp_sum(sk);
  sqk = warp_sum(sqk);
  float* red = sm.red + (group * 2 + 0) * kAuxWarpsPerGroup * 4;
  const int wig = at >> 5;
  if ((at & 31) == 0) {
    red[wig * 4 + 0] = sq;
    red[wig * 4 + 1] = sk;
    red[wig * 4 + 2] = sqk;
  }
  const float A = __expf(a_log);
  float decay[kCpt], oscale[kCpt];
#pragma unroll
  for (int cc = 0; cc < kCpt; ++cc) {
    decay[cc] = __expf(p.lower_bound * sigmoidf(A * (gv[cc] + db[cc])));
    oscale[cc] = nw[cc] * sigmoidf(ov[cc]);
  }
  named_bar_sync(1 + group, kAuxThreads);
  sq = 0.f;
  sk = 0.f;
  sqk = 0.f;
#pragma unroll
  for (int w = 0; w < kAuxWarpsPerGroup; ++w) {
    sq += red[w * 4 + 0];
    sk += red[w * 4 + 1];
    sqk += red[w * 4 + 2];
  }
  const float qs = rsqrtf(sq + 1e-6f) * 0.08838834764831845f;  // 128^-0.5
  const float ks = rsqrtf(sk + 1e-6f);
#pragma unroll
  for (int cc = 0; cc < kCpt; ++cc) {
    const int c = at + cc * kAuxThreads;
    B.q[c] = y[cc][0] * qs;
    B.k[c] = y[cc][1] * ks;
    B.v[c] = y[cc][2];
    B.decay[c] = decay[cc];
    B.oscale[c] = oscale[cc];
  }
  if (at == 0) {
    B.beta = sigmoidf(beta_raw);
    B.qk = sqk * qs * ks;
    B.slot = slot;
  }
  mbar_arrive(&sm.pfull[i % kNumBuf]);
}

__device__ __forceinline__ void aux_epilogue(const Params& p,
                                             const SmemLayout& sm, int group,
                                             int at, int first, int i) {
  mbar_wait(&sm.pdone[i % kNumBuf], (i / kNumBuf) & 1);
  const ItemBuf& B = sm.bufs[i % kNumBuf];
  const int t = first + i;
  const int r = t / p.heads;
  const int h = t - r * p.heads;
  __nv_bfloat16* o_out = p.out + r * p.out_row + h * kD;
  if (B.slot <= 0) {
#pragma unroll
    for (int cc = 0; cc < kCpt; ++cc)
      o_out[at + cc * kAuxThreads] = __float2bfloat16(0.f);
    return;
  }
  float o[kCpt];
  float ss = 0.f;
#pragma unroll
  for (int cc = 0; cc < kCpt; ++cc) {
    o[cc] = B.o[at + cc * kAuxThreads];
    ss += o[cc] * o[cc];
  }
  ss = warp_sum(ss);
  float* red = sm.red + (group * 2 + 1) * kAuxWarpsPerGroup * 4;
  const int wig = at >> 5;
  if ((at & 31) == 0) red[wig * 4] = ss;
  named_bar_sync(1 + group, kAuxThreads);
  ss = 0.f;
#pragma unroll
  for (int w = 0; w < kAuxWarpsPerGroup; ++w) ss += red[w * 4];
  const float inv = rsqrtf(ss * (1.0f / kD) + p.norm_eps);
#pragma unroll
  for (int cc = 0; cc < kCpt; ++cc) {
    const int c = at + cc * kAuxThreads;
    o_out[c] = __float2bfloat16(o[cc] * inv * B.oscale[c]);
  }
}

// ---------------------------------------------------------------- compute
// A half-warp owns one state row (V index) at a time; lane j = lane & 15 holds
// the K elements [4j, 4j+4) and [64+4j, 64+4j+4), so shared-memory reads and
// global stores are conflict-free / coalesced and each row needs a 4-level
// shuffle reduction shared by the two rows of the warp.
// kBulkStore: updated rows go back into the ring slot and one thread stores
// the whole chunk with a bulk async copy (TMA), releasing the slot once the
// copy has read it; otherwise each lane stores its rows with st.global.
template <typename StateT, bool kBulkStore>
__device__ __forceinline__ void compute(const Params& p, const SmemLayout& sm,
                                        int w, int lane, int first, int count) {
  constexpr int kChunks =
      kD * kD * static_cast<int>(sizeof(StateT)) / kChunkBytes;
  constexpr int kChunkRows =
      kChunkBytes / (kD * static_cast<int>(sizeof(StateT)));
  constexpr int kPairs =
      kChunkRows / (2 * kComputeWarps);  // row pairs per warp per chunk
  StateT* st = static_cast<StateT*>(p.state);
  const int half = lane >> 4;
  const int j = lane & 15;
  int rs = 0;
  uint32_t rph = 0;
  int prev_rs = -1;  // kBulkStore: slot of the last chunk stored (thread 0)
  for (int i = 0; i < count; ++i) {
    ItemBuf& B = sm.bufs[i % kNumBuf];
    mbar_wait(&sm.pfull[i % kNumBuf], (i / kNumBuf) & 1);
    const int slot = B.slot;
    if (slot > 0) {
      const int h = (first + i) % p.heads;
      StateT* gst = st + static_cast<int64_t>(slot) * p.state_slot +
                    static_cast<int64_t>(h) * kD * kD;
      float q[8], k[8], d[8];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const float4 q4 =
            *reinterpret_cast<const float4*>(&B.q[64 * u + 4 * j]);
        const float4 k4 =
            *reinterpret_cast<const float4*>(&B.k[64 * u + 4 * j]);
        const float4 d4 =
            *reinterpret_cast<const float4*>(&B.decay[64 * u + 4 * j]);
        q[4 * u] = q4.x, q[4 * u + 1] = q4.y, q[4 * u + 2] = q4.z,
              q[4 * u + 3] = q4.w;
        k[4 * u] = k4.x, k[4 * u + 1] = k4.y, k[4 * u + 2] = k4.z,
              k[4 * u + 3] = k4.w;
        d[4 * u] = d4.x, d[4 * u + 1] = d4.y, d[4 * u + 2] = d4.z,
              d[4 * u + 3] = d4.w;
      }
      const float beta = B.beta;
      const float qk = B.qk;
#pragma unroll 1
      for (int c = 0; c < kChunks; ++c) {
        mbar_wait(&sm.full_s[rs], rph);
        StateT* cs = reinterpret_cast<StateT*>(
            sm.ring + static_cast<size_t>(rs) * kChunkBytes);
        const int cur_rs = rs;
        float s[kPairs][8];
#pragma unroll
        for (int pp = 0; pp < kPairs; ++pp) {
          const int lr = (pp * kComputeWarps + w) * 2 + half;
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            float v4[4];
            StateIO<StateT>::load(cs + lr * kD + 64 * u + 4 * j, v4);
#pragma unroll
            for (int e = 0; e < 4; ++e) s[pp][4 * u + e] = v4[e];
          }
        }
        if (++rs == p.ring) {
          rs = 0;
          rph ^= 1u;
        }
        float a[kPairs], b[kPairs];
#pragma unroll
        for (int pp = 0; pp < kPairs; ++pp) {
          a[pp] = 0.f;
          b[pp] = 0.f;
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            s[pp][e] *= d[e];
            a[pp] += s[pp][e] * k[e];
            b[pp] += s[pp][e] * q[e];
          }
        }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
#pragma unroll
          for (int pp = 0; pp < kPairs; ++pp) {
            a[pp] += __shfl_xor_sync(0xffffffffu, a[pp], off);
            b[pp] += __shfl_xor_sync(0xffffffffu, b[pp], off);
          }
        }
        if constexpr (!kBulkStore) {
          float dep = 0.f;
#pragma unroll
          for (int pp = 0; pp < kPairs; ++pp) dep += a[pp] + b[pp];
          if (lane == 0) mbar_arrive_after(&sm.empty_s[cur_rs], dep);
        }
#pragma unroll
        for (int pp = 0; pp < kPairs; ++pp) {
          const int row = c * kChunkRows + (pp * kComputeWarps + w) * 2 + half;
          const float delta = (B.v[row] - a[pp]) * beta;
          const float o = b[pp] + delta * qk;
#pragma unroll
          for (int e = 0; e < 8; ++e) s[pp][e] += delta * k[e];
          const int lr = (pp * kComputeWarps + w) * 2 + half;
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            float v4[4] = {s[pp][4 * u], s[pp][4 * u + 1], s[pp][4 * u + 2],
                           s[pp][4 * u + 3]};
            if constexpr (kBulkStore) {
              StateIO<StateT>::store(cs + lr * kD + 64 * u + 4 * j, v4);
            } else {
              StateIO<StateT>::store(gst + row * kD + 64 * u + 4 * j, v4);
            }
          }
          if (j == 0) B.o[row] = bf16_round(o);
        }
        if constexpr (kBulkStore) {
          fence_proxy_async();
          named_bar_sync(kComputeBarrier, kComputeWarps * 32);
          if (w == 0 && lane == 0) {
            bulk_s2g(gst + static_cast<int64_t>(c) * kChunkRows * kD, cs,
                     kChunkBytes);
            bulk_commit();
            if (prev_rs >= 0) {
              bulk_wait_read<1>();
              mbar_arrive(&sm.empty_s[prev_rs]);
            }
            prev_rs = cur_rs;
          }
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&sm.pdone[i % kNumBuf]);
  }
  if constexpr (kBulkStore) {
    if (w == 0 && lane == 0) {
      bulk_wait<0>();
      if (prev_rs >= 0) mbar_arrive(&sm.empty_s[prev_rs]);
    }
  }
}

// Whole-item variant: every compute warp gathers its rows of all chunks of the
// item first (releasing each slot as soon as it is read, unless bulk stores
// need it), then runs the kChunks * kPairs row-pair updates interleaved. More
// ILP and one pass per item instead of one per chunk.
template <typename StateT, bool kBulkStore>
__device__ __forceinline__ void compute_wide(const Params& p,
                                             const SmemLayout& sm, int w,
                                             int lane, int first, int count) {
  constexpr int kChunks =
      kD * kD * static_cast<int>(sizeof(StateT)) / kChunkBytes;
  constexpr int kChunkRows =
      kChunkBytes / (kD * static_cast<int>(sizeof(StateT)));
  constexpr int kPairs = kChunkRows / (2 * kComputeWarps);
  constexpr int kItemPairs = kChunks * kPairs;
  StateT* st = static_cast<StateT*>(p.state);
  const int half = lane >> 4;
  const int j = lane & 15;
  int rs = 0;
  uint32_t rph = 0;
  int prev_slots[kChunks];
  bool have_prev = false;
  for (int i = 0; i < count; ++i) {
    ItemBuf& B = sm.bufs[i % kNumBuf];
    mbar_wait(&sm.pfull[i % kNumBuf], (i / kNumBuf) & 1);
    const int slot = B.slot;
    if (slot > 0) {
      const int h = (first + i) % p.heads;
      StateT* gst = st + static_cast<int64_t>(slot) * p.state_slot +
                    static_cast<int64_t>(h) * kD * kD;
      float s[kItemPairs][8];
      int slots[kChunks];
#pragma unroll
      for (int c = 0; c < kChunks; ++c) {
        mbar_wait(&sm.full_s[rs], rph);
        slots[c] = rs;
        const StateT* cs = reinterpret_cast<const StateT*>(
            sm.ring + static_cast<size_t>(rs) * kChunkBytes);
#pragma unroll
        for (int pp = 0; pp < kPairs; ++pp) {
          const int lr = (pp * kComputeWarps + w) * 2 + half;
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            float v4[4];
            StateIO<StateT>::load(cs + lr * kD + 64 * u + 4 * j, v4);
#pragma unroll
            for (int e = 0; e < 4; ++e) s[c * kPairs + pp][4 * u + e] = v4[e];
          }
        }
        if (++rs == p.ring) {
          rs = 0;
          rph ^= 1u;
        }
      }
      float q[8], k[8], d[8];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const float4 q4 =
            *reinterpret_cast<const float4*>(&B.q[64 * u + 4 * j]);
        const float4 k4 =
            *reinterpret_cast<const float4*>(&B.k[64 * u + 4 * j]);
        const float4 d4 =
            *reinterpret_cast<const float4*>(&B.decay[64 * u + 4 * j]);
        q[4 * u] = q4.x, q[4 * u + 1] = q4.y, q[4 * u + 2] = q4.z,
              q[4 * u + 3] = q4.w;
        k[4 * u] = k4.x, k[4 * u + 1] = k4.y, k[4 * u + 2] = k4.z,
              k[4 * u + 3] = k4.w;
        d[4 * u] = d4.x, d[4 * u + 1] = d4.y, d[4 * u + 2] = d4.z,
              d[4 * u + 3] = d4.w;
      }
      float a[kItemPairs], b[kItemPairs];
#pragma unroll
      for (int pp = 0; pp < kItemPairs; ++pp) {
        a[pp] = 0.f;
        b[pp] = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          s[pp][e] *= d[e];
          a[pp] += s[pp][e] * k[e];
          b[pp] += s[pp][e] * q[e];
        }
      }
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) {
#pragma unroll
        for (int pp = 0; pp < kItemPairs; ++pp) {
          a[pp] += __shfl_xor_sync(0xffffffffu, a[pp], off);
          b[pp] += __shfl_xor_sync(0xffffffffu, b[pp], off);
        }
      }
      if constexpr (!kBulkStore) {
        float dep = 0.f;
#pragma unroll
        for (int pp = 0; pp < kItemPairs; ++pp) dep += a[pp] + b[pp];
        if (lane == 0) {
#pragma unroll
          for (int c = 0; c < kChunks; ++c)
            mbar_arrive_after(&sm.empty_s[slots[c]], dep);
        }
      }
      const float beta = B.beta;
      const float qk = B.qk;
#pragma unroll
      for (int c = 0; c < kChunks; ++c) {
#pragma unroll
        for (int pp = 0; pp < kPairs; ++pp) {
          const int lr = (pp * kComputeWarps + w) * 2 + half;
          const int row = c * kChunkRows + lr;
          const int ip = c * kPairs + pp;
          const float delta = (B.v[row] - a[ip]) * beta;
          const float o = b[ip] + delta * qk;
#pragma unroll
          for (int e = 0; e < 8; ++e) s[ip][e] += delta * k[e];
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            float v4[4] = {s[ip][4 * u], s[ip][4 * u + 1], s[ip][4 * u + 2],
                           s[ip][4 * u + 3]};
            if constexpr (kBulkStore) {
              StateT* cs = reinterpret_cast<StateT*>(
                  sm.ring + static_cast<size_t>(slots[c]) * kChunkBytes);
              StateIO<StateT>::store(cs + lr * kD + 64 * u + 4 * j, v4);
            } else {
              StateIO<StateT>::store(gst + row * kD + 64 * u + 4 * j, v4);
            }
          }
          if (j == 0) B.o[row] = bf16_round(o);
        }
      }
      if constexpr (kBulkStore) {
        fence_proxy_async();
        named_bar_sync(kComputeBarrier, kComputeWarps * 32);
        if (w == 0 && lane == 0) {
#pragma unroll
          for (int c = 0; c < kChunks; ++c) {
            bulk_s2g(gst + static_cast<int64_t>(c) * kChunkRows * kD,
                     sm.ring + static_cast<size_t>(slots[c]) * kChunkBytes,
                     kChunkBytes);
            bulk_commit();
          }
          if (have_prev) {
            bulk_wait_read<kChunks>();
#pragma unroll
            for (int c = 0; c < kChunks; ++c)
              mbar_arrive(&sm.empty_s[prev_slots[c]]);
          }
#pragma unroll
          for (int c = 0; c < kChunks; ++c) prev_slots[c] = slots[c];
          have_prev = true;
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&sm.pdone[i % kNumBuf]);
  }
  if constexpr (kBulkStore) {
    if (w == 0 && lane == 0) {
      bulk_wait<0>();
      if (have_prev) {
#pragma unroll
        for (int c = 0; c < kChunks; ++c)
          mbar_arrive(&sm.empty_s[prev_slots[c]]);
      }
    }
  }
}

template <typename StateT, bool kBulkStore, bool kWide>
__global__ void __launch_bounds__(kThreads, 1)
    kda_decode_kernel(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const SmemLayout sm = carve(smem, p.ring);
  const int G = gridDim.x;
  const int b = blockIdx.x;
  const int base = p.items / G;
  const int rem = p.items - base * G;
  const int first = b * base + min(b, rem);
  const int count = base + (b < rem ? 1 : 0);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  if (threadIdx.x == 0) {
    for (int i = 0; i < p.ring; ++i) {
      mbar_init(&sm.full_s[i], 1);
      mbar_init(&sm.empty_s[i], kBulkStore ? 1 : kComputeWarps);
    }
    for (int i = 0; i < kNumBuf; ++i) {
      mbar_init(&sm.pfull[i], kAuxThreads);
      mbar_init(&sm.pdone[i], kComputeWarps);
    }
    fence_barrier_init();
  }
  __syncthreads();

  if (warp == kProducerWarp) {
    if (lane == 0) producer<StateT>(p, sm, first, count);
    return;
  }
  if (warp >= kComputeWarps) {
    const int group = (warp - kComputeWarps) / kAuxWarpsPerGroup;
    const int at =
        threadIdx.x - (kComputeWarps + group * kAuxWarpsPerGroup) * 32;
    if (group < count) aux_prologue(p, sm, group, at, first, group);
    for (int i = group; i < count; i += kAuxGroups) {
      if (i + kAuxGroups < count)
        aux_prologue(p, sm, group, at, first, i + kAuxGroups);
      aux_epilogue(p, sm, group, at, first, i);
    }
    return;
  }
  if constexpr (kWide) {
    compute_wide<StateT, kBulkStore>(p, sm, warp, lane, first, count);
  } else {
    compute<StateT, kBulkStore>(p, sm, warp, lane, first, count);
  }
}

// ---------------------------------------------------------------- roofline
// The kernel's streaming pattern without the math: producer bulk-loads 16 KB
// chunks of `src` into the ring, the 16 compute warps read them back and store
// them to `dst` (512 B per warp per row, as the fp32 state rows). Measures the
// bandwidth ceiling of this pipeline design on the device. kBulkStore: one
// thread stores each chunk straight from shared memory with a bulk async copy.
template <bool kBulkStore>
__global__ void __launch_bounds__(kThreads, 1)
    stream_copy_kernel(const uint8_t* src, uint8_t* dst, int64_t chunks,
                       int ring) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring_smem = smem;
  uint64_t* full_s = reinterpret_cast<uint64_t*>(
      smem + static_cast<size_t>(ring) * kChunkBytes);
  uint64_t* empty_s = full_s + ring;
  const int64_t G = gridDim.x, b = blockIdx.x;
  const int64_t base = chunks / G, rem = chunks - base * G;
  const int64_t first = b * base + min(b, rem);
  const int64_t count = base + (b < rem ? 1 : 0);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int i = 0; i < ring; ++i) {
      mbar_init(&full_s[i], 1);
      mbar_init(&empty_s[i], kBulkStore ? 1 : kComputeWarps);
    }
    fence_barrier_init();
  }
  __syncthreads();
  if (warp == kProducerWarp) {
    if (lane != 0) return;
    int rs = 0;
    uint32_t rph = 0;
    for (int64_t c = 0; c < count; ++c) {
      mbar_wait(&empty_s[rs], rph ^ 1u);
      mbar_arrive_expect_tx(&full_s[rs], kChunkBytes);
      bulk_g2s(ring_smem + static_cast<size_t>(rs) * kChunkBytes,
               src + (first + c) * kChunkBytes, kChunkBytes, &full_s[rs]);
      if (++rs == ring) {
        rs = 0;
        rph ^= 1u;
      }
    }
    return;
  }
  if (warp >= kComputeWarps) return;
  int rs = 0;
  uint32_t rph = 0;
  if constexpr (kBulkStore) {
    if (warp != 0 || lane != 0) return;
    int prev = -1;
    for (int64_t c = 0; c < count; ++c) {
      mbar_wait(&full_s[rs], rph);
      bulk_s2g(dst + (first + c) * kChunkBytes,
               ring_smem + static_cast<size_t>(rs) * kChunkBytes, kChunkBytes);
      bulk_commit();
      if (prev >= 0) {
        bulk_wait_read<1>();
        mbar_arrive(&empty_s[prev]);
      }
      prev = rs;
      if (++rs == ring) {
        rs = 0;
        rph ^= 1u;
      }
    }
    bulk_wait<0>();
    return;
  }
  for (int64_t c = 0; c < count; ++c) {
    mbar_wait(&full_s[rs], rph);
    const float* cs = reinterpret_cast<const float*>(
        ring_smem + static_cast<size_t>(rs) * kChunkBytes);
    float v[2][4];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      StateIO<float>::load(cs + (rr * kComputeWarps + warp) * kD + 4 * lane,
                           v[rr]);
    const int cur = rs;
    if (++rs == ring) {
      rs = 0;
      rph ^= 1u;
    }
    float* d = reinterpret_cast<float*>(dst + (first + c) * kChunkBytes);
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      StateIO<float>::store(d + (rr * kComputeWarps + warp) * kD + 4 * lane,
                            v[rr]);
    float dep = v[0][0] + v[1][0];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      dep += __shfl_xor_sync(0xffffffffu, dep, off);
    if (lane == 0) mbar_arrive_after(&empty_s[cur], dep);
  }
}

// Plain vectorised copy (no shared memory): grid-stride, 4 x 16 B in flight
// per thread. The "ordinary kernel" reference point next to torch's copy.
__global__ void __launch_bounds__(512)
    plain_copy_kernel(const float4* src, float4* dst, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    float4 a = src[i], b = src[i + stride], c = src[i + 2 * stride],
           d = src[i + 3 * stride];
    dst[i] = a;
    dst[i + stride] = b;
    dst[i + 2 * stride] = c;
    dst[i + 3 * stride] = d;
  }
  for (; i < n; i += stride) dst[i] = src[i];
}

// ---------------------------------------------------------------- host
struct DeviceInfo {
  int sms = 0;
  int smem_max_block = 0;
  bool attr_set[8] = {false, false, false, false, false, false, false, false};
  bool copy_attr_set[2] = {false, false};
};

inline DeviceInfo& device_info(int dev) {
  static DeviceInfo infos[64];
  DeviceInfo& d = infos[dev];
  if (d.sms == 0) {
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&d.smem_max_block,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return d;
}

inline int default_ring(const DeviceInfo& d) {
  int ring = kDefaultRing;
  while (ring > 2 && smem_bytes(ring) > static_cast<size_t>(d.smem_max_block))
    --ring;
  return ring;
}

#define GLM_KDA_CHECK_DTYPE(t, dt)                         \
  TVM_FFI_ICHECK(encode_dlpack_dtype((t).dtype()) == (dt)) \
      << #t " has the wrong dtype"

void kda_decode(TensorView x, TensorView conv_w, TensorView conv_state,
                TensorView raw_gate, TensorView raw_beta, TensorView a_log,
                TensorView dt_bias, TensorView state_indices, TensorView state,
                TensorView out_gate, TensorView norm_w, TensorView out,
                double lower_bound, double norm_eps, int64_t ring, int64_t ctas,
                int64_t flags) {
  GLM_KDA_CHECK_DTYPE(x, bfloat16_code);
  GLM_KDA_CHECK_DTYPE(conv_w, float32_code);
  GLM_KDA_CHECK_DTYPE(conv_state, bfloat16_code);
  GLM_KDA_CHECK_DTYPE(raw_gate, bfloat16_code);
  GLM_KDA_CHECK_DTYPE(raw_beta, bfloat16_code);
  GLM_KDA_CHECK_DTYPE(a_log, float32_code);
  GLM_KDA_CHECK_DTYPE(dt_bias, float32_code);
  GLM_KDA_CHECK_DTYPE(state_indices, int32_code);
  GLM_KDA_CHECK_DTYPE(out_gate, bfloat16_code);
  GLM_KDA_CHECK_DTYPE(norm_w, float32_code);
  GLM_KDA_CHECK_DTYPE(out, bfloat16_code);
  const int64_t state_code = encode_dlpack_dtype(state.dtype());
  TVM_FFI_ICHECK(state_code == float32_code || state_code == bfloat16_code)
      << "state must be float32 or bfloat16";

  TVM_FFI_ICHECK(x.ndim() == 2 && x.stride(1) == 1)
      << "x must be [rows, 3*H*128], channel-contiguous";
  const int64_t rows = x.size(0);
  TVM_FFI_ICHECK(x.size(1) % (3 * kD) == 0) << "x width must be 3*H*128";
  const int64_t P = x.size(1) / 3;
  const int64_t H = P / kD;
  TVM_FFI_ICHECK(rows > 0);
  TVM_FFI_ICHECK(conv_w.ndim() == 2 && conv_w.size(0) == 3 * P &&
                 conv_w.size(1) == 4 && conv_w.stride(1) == 1 &&
                 conv_w.stride(0) == 4)
      << "conv_w must be contiguous [3*H*128, 4]";
  TVM_FFI_ICHECK(conv_state.ndim() == 3 && conv_state.size(1) == 3 * P &&
                 conv_state.size(2) == 3)
      << "conv_state must be [slots, 3*H*128, 3] (no speculative columns)";
  TVM_FFI_ICHECK(raw_gate.size(raw_gate.ndim() - 1) == kD &&
                 raw_gate.size(raw_gate.ndim() - 2) == H &&
                 raw_gate.stride(raw_gate.ndim() - 1) == 1 &&
                 raw_gate.stride(raw_gate.ndim() - 2) == kD &&
                 raw_gate.size(raw_gate.ndim() - 3) == rows)
      << "raw_gate must be [(1,) rows, H, 128] with contiguous head rows";
  TVM_FFI_ICHECK(out_gate.size(out_gate.ndim() - 1) == kD &&
                 out_gate.size(out_gate.ndim() - 2) == H &&
                 out_gate.stride(out_gate.ndim() - 1) == 1 &&
                 out_gate.stride(out_gate.ndim() - 2) == kD &&
                 out_gate.size(out_gate.ndim() - 3) == rows)
      << "out_gate must be [(1,) rows, H, 128] with contiguous head rows";
  TVM_FFI_ICHECK(
      out.size(out.ndim() - 1) == kD && out.size(out.ndim() - 2) == H &&
      out.stride(out.ndim() - 1) == 1 && out.stride(out.ndim() - 2) == kD &&
      out.size(out.ndim() - 3) == rows)
      << "out must be [(1,) rows, H, 128] with contiguous head rows";
  TVM_FFI_ICHECK(raw_beta.size(raw_beta.ndim() - 1) == H &&
                 raw_beta.stride(raw_beta.ndim() - 1) == 1 &&
                 raw_beta.size(raw_beta.ndim() - 2) == rows)
      << "raw_beta must be [(1,) rows, H]";
  TVM_FFI_ICHECK(a_log.numel() == H && a_log.IsContiguous())
      << "A_log must be [H]";
  TVM_FFI_ICHECK(dt_bias.numel() == P && dt_bias.IsContiguous())
      << "dt_bias must be [H*128]";
  TVM_FFI_ICHECK(norm_w.numel() == kD && norm_w.IsContiguous())
      << "norm_w must be [128]";
  TVM_FFI_ICHECK(state_indices.ndim() == 1 && state_indices.size(0) == rows &&
                 state_indices.IsContiguous())
      << "state_indices must be contiguous [rows]";
  const int64_t esize = state_code == float32_code ? 4 : 2;
  TVM_FFI_ICHECK(state.ndim() == 4 && state.size(1) == H &&
                 state.size(2) == kD && state.size(3) == kD &&
                 state.stride(3) == 1 && state.stride(2) == kD &&
                 state.stride(1) == kD * kD && state.stride(0) >= H * kD * kD)
      << "state must be [slots, H, 128, 128] with contiguous slot contents";
  TVM_FFI_ICHECK((state.stride(0) * esize) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0)
      << "state slots must be 16-byte aligned";
  TVM_FFI_ICHECK(state.size(0) == conv_state.size(0))
      << "conv_state and state must have the same number of slots";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(conv_w.data_ptr()) % 16 == 0);
  TVM_FFI_ICHECK(lower_bound < 0) << "lower_bound must be negative";

  const int dev = x.device().device_id;
  ffi::CUDADeviceGuard device_guard(dev);
  DeviceInfo& di = device_info(dev);
  int ring_used = ring > 0 ? static_cast<int>(ring) : default_ring(di);
  TVM_FFI_ICHECK(ring_used >= 2 && ring_used <= kMaxRing &&
                 smem_bytes(ring_used) <=
                     static_cast<size_t>(di.smem_max_block))
      << "ring depth " << ring_used << " does not fit in shared memory";
  const int64_t items = rows * H;
  int64_t grid = ctas > 0 ? ctas : di.sms;
  if (grid > items) grid = items;

  Params p;
  p.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
  p.x_row = x.stride(0);
  p.conv_w = static_cast<const float*>(conv_w.data_ptr());
  p.conv_state = static_cast<__nv_bfloat16*>(conv_state.data_ptr());
  p.cs_slot = conv_state.stride(0);
  p.cs_ch = conv_state.stride(1);
  p.cs_t = conv_state.stride(2);
  p.gate = static_cast<const __nv_bfloat16*>(raw_gate.data_ptr());
  p.gate_row = raw_gate.stride(raw_gate.ndim() - 3);
  p.beta = static_cast<const __nv_bfloat16*>(raw_beta.data_ptr());
  p.beta_row = raw_beta.stride(raw_beta.ndim() - 2);
  p.a_log = static_cast<const float*>(a_log.data_ptr());
  p.dt_bias = static_cast<const float*>(dt_bias.data_ptr());
  p.idx = static_cast<const int32_t*>(state_indices.data_ptr());
  p.state = state.data_ptr();
  p.state_slot = state.stride(0);
  p.ogate = static_cast<const __nv_bfloat16*>(out_gate.data_ptr());
  p.ogate_row = out_gate.stride(out_gate.ndim() - 3);
  p.norm_w = static_cast<const float*>(norm_w.data_ptr());
  p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  p.out_row = out.stride(out.ndim() - 3);
  p.heads = static_cast<int>(H);
  p.P = static_cast<int>(P);
  p.lower_bound = static_cast<float>(lower_bound);
  p.norm_eps = static_cast<float>(norm_eps);
  p.items = static_cast<int>(items);
  p.ring = ring_used;

  const cudaStream_t stream = get_stream(x.device());
  const size_t smem = smem_bytes(ring_used);
  {
    const int chunks = static_cast<int>(kD * kD * esize / kChunkBytes);
    const int64_t fl = flags < 0 ? kDefaultFlags : flags;
    // Bulk stores hold a slot until its copy-out has read it: the whole-item
    // pass needs two items of slots, the per-chunk pass two chunks.
    const int need =
        (fl & 1) ? ((fl & 2) ? 2 * chunks : 2) : ((fl & 2) ? chunks : 1);
    TVM_FFI_ICHECK(ring_used >= need)
        << "ring depth " << ring_used << " too small for flags " << fl;
  }
  // flags bit 0: bulk (TMA) state stores, bit 1: whole-item compute pass;
  // flags < 0: the default variant.
  const int64_t f = flags < 0 ? kDefaultFlags : flags;
  const bool is_f32 = state_code == float32_code;
  const int which = (is_f32 ? 0 : 1) + static_cast<int>(f & 3) * 2;
  void (*kernel)(Params) = nullptr;
  switch (which) {
    case 0:
      kernel = kda_decode_kernel<float, false, false>;
      break;
    case 1:
      kernel = kda_decode_kernel<__nv_bfloat16, false, false>;
      break;
    case 2:
      kernel = kda_decode_kernel<float, true, false>;
      break;
    case 3:
      kernel = kda_decode_kernel<__nv_bfloat16, true, false>;
      break;
    case 4:
      kernel = kda_decode_kernel<float, false, true>;
      break;
    case 5:
      kernel = kda_decode_kernel<__nv_bfloat16, false, true>;
      break;
    case 6:
      kernel = kda_decode_kernel<float, true, true>;
      break;
    default:
      kernel = kda_decode_kernel<__nv_bfloat16, true, true>;
      break;
  }
  if (!di.attr_set[which]) {
    cudaError_t e = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, di.smem_max_block);
    TVM_FFI_ICHECK(e == cudaSuccess)
        << "cudaFuncSetAttribute: " << cudaGetErrorString(e);
    di.attr_set[which] = true;
  }
  kernel<<<static_cast<unsigned>(grid), kThreads, smem, stream>>>(p);
  cudaError_t e = cudaGetLastError();
  TVM_FFI_ICHECK(e == cudaSuccess)
      << "kda_decode launch failed: " << cudaGetErrorString(e);
}

// mode 0: ring + warp st.global, 1: ring + bulk store, 2: plain copy kernel.
void stream_copy(TensorView src, TensorView dst, int64_t ring, int64_t ctas,
                 int64_t mode) {
  TVM_FFI_ICHECK(src.IsContiguous() && dst.IsContiguous());
  const int64_t nbytes = src.numel() * ((src.dtype().bits + 7) / 8);
  TVM_FFI_ICHECK(nbytes % kChunkBytes == 0 && dst.numel() == src.numel());
  const int dev = src.device().device_id;
  ffi::CUDADeviceGuard device_guard(dev);
  DeviceInfo& di = device_info(dev);
  if (mode == 2) {
    const int64_t grid = ctas > 0 ? ctas : 4 * di.sms;
    plain_copy_kernel<<<static_cast<unsigned>(grid), 512, 0,
                        get_stream(src.device())>>>(
        static_cast<const float4*>(src.data_ptr()),
        static_cast<float4*>(dst.data_ptr()), nbytes / 16);
    cudaError_t e = cudaGetLastError();
    TVM_FFI_ICHECK(e == cudaSuccess) << cudaGetErrorString(e);
    return;
  }
  const int ring_used = ring > 0 ? static_cast<int>(ring) : default_ring(di);
  const size_t smem = static_cast<size_t>(ring_used) * kChunkBytes +
                      2 * ring_used * sizeof(uint64_t);
  TVM_FFI_ICHECK(smem <= static_cast<size_t>(di.smem_max_block));
  auto kernel =
      mode == 1 ? stream_copy_kernel<true> : stream_copy_kernel<false>;
  if (!di.copy_attr_set[mode == 1]) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         di.smem_max_block);
    di.copy_attr_set[mode == 1] = true;
  }
  const int64_t chunks = nbytes / kChunkBytes;
  int64_t grid = ctas > 0 ? ctas : di.sms;
  if (grid > chunks) grid = chunks;
  kernel<<<static_cast<unsigned>(grid), kThreads, smem,
           get_stream(src.device())>>>(
      static_cast<const uint8_t*>(src.data_ptr()),
      static_cast<uint8_t*>(dst.data_ptr()), chunks, ring_used);
  cudaError_t e = cudaGetLastError();
  TVM_FFI_ICHECK(e == cudaSuccess) << cudaGetErrorString(e);
}

// Launch configuration the defaults resolve to on a device (for the harness).
int64_t default_ring_for_device(int64_t dev) {
  return default_ring(device_info(dev));
}
int64_t sm_count_for_device(int64_t dev) { return device_info(dev).sms; }

}  // namespace glm5next_kda

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kda_decode, glm5next_kda::kda_decode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(default_ring_for_device,
                              glm5next_kda::default_ring_for_device);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm_count_for_device,
                              glm5next_kda::sm_count_for_device);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(stream_copy, glm5next_kda::stream_copy);
