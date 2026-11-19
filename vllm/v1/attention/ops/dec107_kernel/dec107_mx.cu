// Adapted from FlashInfer's decode_balanced_bf16_mtp_n32 kernel (Apache-2.0).
#define DEC107_MX 1
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeFmhaTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeFmhaTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeFmhaTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeFmhaTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeFmhaTensorMapPack { CakeFmhaTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeFmhaTensorMap) >= alignof(CUtensorMap), "CakeFmhaTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_FMHA_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O_OFFSET 256
#define NUM_Q_PIPE_STAGES 1
#define NUM_K_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 3
#define NUM_SM_PIPE_STAGES 2
#define NUM_STATS_PIPE_STAGES 2
#define NUM_PAGE_PIPE_STAGES 6
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_XMAX_OFF 1024
#define SMEM_SMEM_XMAX_STAGE_BYTES 1024
#define SMEM_SMEM_XMAX_STRIDE 1024
#define SMEM_SMEM_SUM_OFF 3072
#define SMEM_SMEM_SUM_STAGE_BYTES 512
#define SMEM_SMEM_SUM_STRIDE 512
#define SMEM_SMEM_CORR_FLAG_OFF 4608
#define SMEM_SMEM_CORR_FLAG_STAGE_BYTES 16
#define SMEM_SMEM_CORR_FLAG_STRIDE 16
#define SMEM_SMEM_MAX_OFF 3584
#define SMEM_SMEM_MAX_STAGE_BYTES 512
#define SMEM_SMEM_MAX_STRIDE 512
#define SMEM_SMEM_PAGE_OFFSETS_OFF 4096
#define SMEM_SMEM_PAGE_OFFSETS_STAGE_BYTES 192
#define SMEM_SMEM_PAGE_OFFSETS_STRIDE 192
#define SMEM_WORK_TOKEN_WORDS_OFF 4352
#define SMEM_WORK_TOKEN_WORDS_STAGE_BYTES 256
#define SMEM_WORK_TOKEN_WORDS_STRIDE 256
#define SMEM_SCHED_SEQ_LENS_OFF 5120
#define SMEM_SCHED_SEQ_LENS_STAGE_BYTES 4096
#define SMEM_SCHED_SEQ_LENS_STRIDE 4096
#define SMEM_SMEM_QT_OFF 9216
#define SMEM_SMEM_QT_STAGE_BYTES 16384
#define SMEM_SMEM_QT_STRIDE 16384
#define SMEM_SMEM_K_OFF 25600
#define SMEM_SMEM_K_STAGE_BYTES 32768
#define SMEM_SMEM_K_STRIDE 32768
#define SMEM_SMEM_V_OFF 123904
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_TOTAL (MX_V_OFF + 98304 + 320)
#define LOC_PG_OFF (MX_V_OFF + 98304)
#ifdef DEC107_SINGLE_S
#define DEC107_SBUF_STRIDE 0
#else
#define DEC107_SBUF_STRIDE 128
#endif
#define LOC_DOM_OFF (LOC_PG_OFF + 128)
#define LOC_PLAN_OFF (LOC_PG_OFF + 256)
#if defined(DEC107_LDG)
#error "DEC107_LDG is not supported in the MXFP4-K build"
#endif
#if defined(DEC107_LOWB) && (defined(DEC107_LOCAL) || defined(DEC107_LDG))
#error "DEC107_LOCAL / DEC107_LDG are not supported with DEC107_LOWB"
#endif
#define THREADS 512
#define DEC107_LB 512, 1
#define DEC107_KVF_CNT 1
#define DEC107_WE_CNT 480
#define DEC107_LSEL(a, b) (b)
#define DEC107_TX(bar, n) mbarrier_arrive_expect_tx(bar, n)
#define MX_K_OFF 41984
#define MX_K_STAGE 17408
#define MX_K_BYTES 17408
#define MX_SFA_COL 512
#define MX_SFB_COL 520
#define MX_IDESC ((0u << 7) | (5u << 10) | ((128u >> 3) << 17) | (1u << 23) | ((128u >> 7) << 27) | (1u << 31))
#define BLOCK_N 128
#define HEAD_DIM 256
#define PAGE_SIZE 32
#define NUM_K_STAGES 3
#ifndef DEC107_KV_STAGES
#define DEC107_KV_STAGES 3
#endif
#ifndef DEC107_K_STAGES
#define DEC107_K_STAGES 5
#endif
#if DEC107_K_STAGES > 5 || DEC107_K_STAGES < DEC107_KV_STAGES
#error "DEC107_K_STAGES must be in [KV_STAGES, 5] (page-offset ring 6)"
#endif
#define MX_V_OFF (MX_K_OFF + DEC107_K_STAGES * MX_K_STAGE)
#define NUM_V_STAGES 3

#include <math_constants.h>
#ifndef DEC107_LOG2_P_SCALE
#define DEC107_LOG2_P_SCALE 7.807354922057604f  // log2(224): P is stored as E4M3 x224
#endif
#ifndef DEC107_THR
#define DEC107_THR 1.0f  // lazy-rescale threshold (log2 units); P max = 2^THR * 2^P_SCALE <= 448
#endif
// Optional device-side diagnostics; disabled in the runtime build.
#ifdef DEC107_PROF
__device__ unsigned long long dec107_prof[2048 * 16];
__device__ __forceinline__ unsigned long long dec107_now() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define DEC107_STAMP(s) do { dec107_prof[blockIdx.x * 16 + (s)] = dec107_now(); } while (0)
#define DEC107_STAMP_ONCE(s) do { if (dec107_prof[blockIdx.x * 16 + (s)] == 0ull) DEC107_STAMP(s); } while (0)
#define DEC107_COUNT(s, n) atomicAdd(&dec107_prof[blockIdx.x * 16 + (s)], (unsigned long long)(n))
#else
#define DEC107_STAMP(s)
#define DEC107_STAMP_ONCE(s)
#define DEC107_COUNT(s, n)
#endif


__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count) : "memory");
}


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}


__device__ __forceinline__ void tcgen05_mma_f8(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void tmem_ld_x32(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
        : "r"(tmem_addr));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}


__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}


__device__ __forceinline__ float2 ex2_emulation_f32x2_value(float2 value) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(value.x, -127.0f), x1 = max_noftz(value.y, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    return make_float2(r0, r1);
}

__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    float2 result = ex2_emulation_f32x2_value(make_float2(*x0_ptr, *x1_ptr));
    *x0_ptr = result.x; *x1_ptr = result.y;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
    }
}



__device__ __forceinline__ void softmax_block_sum(const float* sv, float2* acc) {
    const float2* sv2 = reinterpret_cast<const float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        asm("add.f32x2 %0, %1, %2;"
            : "+l"(reinterpret_cast<uint64_t&>(*acc))
            : "l"(reinterpret_cast<uint64_t&>(*acc)),
              "l"(reinterpret_cast<const uint64_t&>(sv2[j])));
    }
}


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void fma_f32x2_noftz_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("add.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("sub.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 add_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void dec107_mma_mx(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t sfa, uint32_t sfb,
                                              uint32_t acc) {
    asm volatile("{.reg .pred p; setp.ne.u32 p, %6, 0; tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32 "
                 "[%0], %1, %2, %3, [%4], [%5], p;}"
                 :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(sfa), "r"(sfb), "r"(acc) : "memory");
}

__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}


#ifdef DEC107_PROF
__device__ unsigned long long dec107_prof2[2048 * 8];
#define DEC107_STAMP2(s) do { dec107_prof2[blockIdx.x * 8 + (s)] = dec107_now(); } while (0)
#else
#define DEC107_STAMP2(s)
#endif
#ifdef DEC107_LOWB
__device__ __forceinline__ unsigned dec107_cta_rank() { unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned dec107_ncta() { unsigned r; asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned dec107_mapa(unsigned a, unsigned rank) {
    unsigned r; asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r; }
__device__ __forceinline__ void dec107_remote_arrive(unsigned a) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void dec107_wait_cluster(unsigned a, unsigned parity) {
    asm volatile("{\n\t.reg .pred p;\n\tLAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n\t"
                 "@!p bra LAB_WAIT;\n\t}" :: "r"(a), "r"(parity) : "memory"); }
__device__ __forceinline__ float dec107_ldc(unsigned a) {
    float v; asm("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(a)); return v; }
__device__ __forceinline__ float4 dec107_ldc4(unsigned a) {
    float4 v; asm("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];"
                  : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a)); return v; }
#endif

extern "C" {

__global__ __launch_bounds__(DEC107_LB) void
kernel_dec107_decode_mxk_hd256_p32(const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK, const __grid_constant__ CUtensorMap tmV, const __grid_constant__ CUtensorMap tmKs, __nv_bfloat16* __restrict__ O_ptr, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, float* __restrict__ partial_o, float* __restrict__ partial_stats, unsigned int* __restrict__ tile_counters, unsigned int* __restrict__ queue_counters, int max_pages_per_seq, float softmax_scale_log2, int num_q_heads, int num_kv_heads, int batch_size, int q_len, unsigned int max_items, float out_scale, const signed char* __restrict__ sm_domain, int pages_per_domain, const uint8_t* __restrict__ kv_base, long long kv_page_stride, int kv_head_stride, int kv_tok_stride, int kv_v_off, const float* __restrict__ q_scale_dev)
{
    const CUtensorMap* Q = &tmQ;
    const CUtensorMap* K = &tmK;
    const CUtensorMap* V = &tmV;
    const CUtensorMap* KS = &tmKs;
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#else
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#endif

    const int mbar_base = smem;
    if (tid == 0) DEC107_STAMP(0);
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define k_full_addr (mbar_base + 656)
    #define k_empty_addr (mbar_base + 704)
    #define v_full_addr (mbar_base + 64)
    #define v_empty_addr (mbar_base + 88)
    #define s_full_addr (mbar_base + 112)
    #define p_full_addr (mbar_base + 128)
    #define o_ready_addr (mbar_base + 144)
    #define o_empty_addr (mbar_base + 160)
    #define stats_full_addr (mbar_base + 168)
    #define stats_empty_addr (mbar_base + 184)
    #define tmem_dealloc_addr (mbar_base + 200)
    #define page_offsets_full_addr (mbar_base + 208)
    #define page_offsets_empty_addr (mbar_base + 256)
    #define work_full_addr (mbar_base + 304)
    #define work_empty_addr (mbar_base + 336)
    #define claim_gate_addr (mbar_base + 368)
    #define mx_sfb_full_addr (mbar_base + 608)
    #define mx_sfb_empty_addr (mbar_base + 624)
    #define mx_sfa_ready_addr (mbar_base + 640)
    #define lk_full_addr (mbar_base + 512)
    #define lk_empty_addr (mbar_base + 536)
    #define lv_full_addr (mbar_base + 560)
    #define lv_empty_addr (mbar_base + 584)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;


    // Kernel setup ops
    float* smem_xmax = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_xmax_addr = smem + 1024;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 3072);
    const int smem_sum_addr = smem + 3072;
    unsigned int* smem_corr_flag = reinterpret_cast<unsigned int*>(smem_raw + 4608);
    const int smem_corr_flag_addr = smem + 4608;
    float* smem_max = reinterpret_cast<float*>(smem_raw + 3584);
    const int smem_max_addr = smem + 3584;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 4096);
    const int smem_page_offsets_addr = smem + 4096;
    unsigned int* work_token_words = reinterpret_cast<unsigned int*>(smem_raw + 4352);
    const int work_token_words_addr = smem + 4352;
    int* sched_seq_lens = reinterpret_cast<int*>(smem_raw + 5120);
    int* dec107_ldg_pg = reinterpret_cast<int*>(smem_raw + LOC_PG_OFF);
    unsigned int* sched_dom = reinterpret_cast<unsigned int*>(smem_raw + LOC_DOM_OFF);
    unsigned int* loc_plan = reinterpret_cast<unsigned int*>(smem_raw + LOC_PLAN_OFF);
    (void)dec107_ldg_pg; (void)sched_dom; (void)loc_plan;
    const int sched_seq_lens_addr = smem + 5120;
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_qt_addr = smem + 9216;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + MX_K_OFF);
    const int smem_k_addr = smem + MX_K_OFF;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + MX_V_OFF);
    const int smem_v_addr = smem + MX_V_OFF;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Q)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory");

    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 47 barriers)
    // Mbarriers at smem_raw[0..376)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, DEC107_KVF_CNT);
            mbarrier_init(smem + 24, DEC107_KVF_CNT);
            mbarrier_init(smem + 32, DEC107_KVF_CNT);
            // k_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 3 barriers, init_count=1
            mbarrier_init(smem + 64, DEC107_KVF_CNT);
            mbarrier_init(smem + 72, DEC107_KVF_CNT);
            mbarrier_init(smem + 80, DEC107_KVF_CNT);
            // v_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'sm_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 128, 256);
            mbarrier_init(smem + 136, 256);
            // o_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            // --- pipeline 'stats_pipe' ---
            // stats_full: 2 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            // stats_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 184, 4);
            mbarrier_init(smem + 192, 4);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            // --- pipeline 'page_pipe' ---
            // page_offsets_full: 6 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // page_offsets_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            // work_empty: 4 barriers, init_count=352
            mbarrier_init(smem + 336, DEC107_WE_CNT);
            mbarrier_init(smem + 344, DEC107_WE_CNT);
            mbarrier_init(smem + 352, DEC107_WE_CNT);
            mbarrier_init(smem + 360, DEC107_WE_CNT);
            // claim_gate: 1 barriers, init_count=1
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 608, 128);
            mbarrier_init(smem + 616, 128);
            mbarrier_init(smem + 624, 1);
            mbarrier_init(smem + 632, 1);
            mbarrier_init(smem + 640, 128);
            for (int _k = 0; _k < 6; _k++) { mbarrier_init(smem + 656 + 8 * _k, 1); mbarrier_init(smem + 704 + 8 * _k, 1); }
#ifdef DEC107_LDG
            for (int _l = 0; _l < 12; _l++) mbarrier_init(smem + 512 + 8 * _l, ((_l / 3) & 1) ? 128 : 1);
#endif
#ifdef DEC107_LOWB
            mbarrier_init(smem + 384, (int)dec107_ncta());
            mbarrier_init(smem + 392, (int)dec107_ncta());
#endif
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 376);
    if (warp == 0) {
        int _tmem_hold = smem + 376;
        asm volatile("tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(576) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#ifdef DEC107_LOWB
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif

    const int taddr = tmem_addr_storage[0];
    // PDL: wait for the producer grid before the first global read (seq_lens, page_table, Q, KV). No-op without PDL.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (q_scale_dev != nullptr) softmax_scale_log2 *= __ldg(q_scale_dev);
    if (tid == 0) DEC107_STAMP(1);

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o = taddr + 256;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }
    if (warp >= 12) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // softmax_main
            const int s_warp = warp;
            int sm_tid = s_warp * 32 + lane;
            int my_row = s_warp * 16 + lane % 16;
            int half = lane / 16;
            int tok_base = half * 64;
            int my_s_base = taddr + (unsigned int)(s_warp * 32 << 16);
            int rows_live = ((s_warp * 16 < 32) ? 1 : 0);
            int row_j = my_row / 8;
            int _min_1 = ((row_j) < (q_len - 1) ? (row_j) : (q_len - 1));
            int vis_j = _min_1;
            unsigned int sm_stage = 0;
            unsigned int sm_phase = 0;
            unsigned int xm_slot_s = 0;
            unsigned int st_stage_s = 0;
            unsigned int st_phase_s = 1;
            float _rcp_11 = approx_rcp(softmax_scale_log2);
            float thr_raw = DEC107_THR * _rcp_11;
            unsigned int work_stage_s = 0;
            unsigned int _phase_work_full = 0;
            mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
            unsigned int base = work_stage_s * 16;
            unsigned int valid = work_token_words[base];
            unsigned int kind = work_token_words[base + 1];
            unsigned int batch = work_token_words[base + 2];
            unsigned int kv_head = work_token_words[base + 3];
            unsigned int block_begin = work_token_words[base + 4];
            unsigned int block_end = work_token_words[base + 5];
            unsigned int seqlen = work_token_words[base + 6];
            unsigned int n_chunks = work_token_words[base + 7];
            unsigned int slot_tile_base = work_token_words[base + 8];
            unsigned int counter_idx = work_token_words[base + 9];
            unsigned int chunk = work_token_words[base + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
            work_stage_s += 1;
            if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
            unsigned int valid_s = valid;
            int kind_s = (int)kind;
            int block_begin_s = (int)block_begin;
            int block_end_s = (int)block_end;
            int seqlen_s = (int)seqlen;
            int batch_s = (int)batch;
            int kv_head_s = (int)kv_head;
            int n_chunks_s = (int)n_chunks;
            int slot_tile_base_s = (int)slot_tile_base;
            int live_rows_s = q_len * 8;
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < max_items; _tile_iter_s++) {
                if (valid_s == 0) {
                    break;
                }
                if (kind_s == 0) {
                    int cnt_s = block_end_s - block_begin_s;
                    int vis_min = seqlen_s - (q_len - 1);
                    int vis_col = vis_min + vis_j;
                    float row_max = -CAKE_FMHA_INF;
                    float psum = 0.0f;
                    #pragma unroll 1
                    for (int n = 0; n < cnt_s; n++) {
                        if (sm_tid == 0) {
                        }
                        mbarrier_wait(s_full_addr + (sm_stage) * 8, sm_phase);
                        if (sm_tid == 0) DEC107_STAMP_ONCE(6);
                        if (sm_tid == 0) {
                        }
                        int my_block = block_begin_s + cnt_s - 1 - n;
                        int blk_pos = my_block * BLOCK_N + tok_base;
                        float sv[32];
                        float lmax = -CAKE_FMHA_INF;
                        float new_max = row_max;
                        float acc_scale = 1.0f;
                        float lsum = 0.0f;
                        int xm_off_s = (int)xm_slot_s * 128;
                        if (rows_live != 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sv[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv[3])), "=r"(*reinterpret_cast<uint32_t*>(&sv[4])), "=r"(*reinterpret_cast<uint32_t*>(&sv[5])), "=r"(*reinterpret_cast<uint32_t*>(&sv[6])), "=r"(*reinterpret_cast<uint32_t*>(&sv[7])), "=r"(*reinterpret_cast<uint32_t*>(&sv[8])), "=r"(*reinterpret_cast<uint32_t*>(&sv[9])), "=r"(*reinterpret_cast<uint32_t*>(&sv[10])), "=r"(*reinterpret_cast<uint32_t*>(&sv[11])), "=r"(*reinterpret_cast<uint32_t*>(&sv[12])), "=r"(*reinterpret_cast<uint32_t*>(&sv[13])), "=r"(*reinterpret_cast<uint32_t*>(&sv[14])), "=r"(*reinterpret_cast<uint32_t*>(&sv[15])), "=r"(*reinterpret_cast<uint32_t*>(&sv[16])), "=r"(*reinterpret_cast<uint32_t*>(&sv[17])), "=r"(*reinterpret_cast<uint32_t*>(&sv[18])), "=r"(*reinterpret_cast<uint32_t*>(&sv[19])), "=r"(*reinterpret_cast<uint32_t*>(&sv[20])), "=r"(*reinterpret_cast<uint32_t*>(&sv[21])), "=r"(*reinterpret_cast<uint32_t*>(&sv[22])), "=r"(*reinterpret_cast<uint32_t*>(&sv[23])), "=r"(*reinterpret_cast<uint32_t*>(&sv[24])), "=r"(*reinterpret_cast<uint32_t*>(&sv[25])), "=r"(*reinterpret_cast<uint32_t*>(&sv[26])), "=r"(*reinterpret_cast<uint32_t*>(&sv[27])), "=r"(*reinterpret_cast<uint32_t*>(&sv[28])), "=r"(*reinterpret_cast<uint32_t*>(&sv[29])), "=r"(*reinterpret_cast<uint32_t*>(&sv[30])), "=r"(*reinterpret_cast<uint32_t*>(&sv[31]))
                                : "r"((unsigned int)my_s_base + sm_stage * (unsigned int)DEC107_SBUF_STRIDE));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            int n_vis = vis_col - blk_pos;
                            if (my_row >= 32) {
                                n_vis = 0;
                            }
                            if (n_vis < 32) {
                                int _max_5 = ((n_vis) > (0) ? (n_vis) : (0));
                                int n_lo = _max_5;
                                uint32_t _slice_lo_mask_0;
                                {
                                    int _lim_0 = n_lo;
                                    if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                                    else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                                    }
                                }
                                if (!(_slice_lo_mask_0 & (1u << 0))) sv[0] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 1))) sv[1] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 2))) sv[2] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 3))) sv[3] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 4))) sv[4] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 5))) sv[5] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 6))) sv[6] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 7))) sv[7] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 8))) sv[8] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 9))) sv[9] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 10))) sv[10] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 11))) sv[11] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 12))) sv[12] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 13))) sv[13] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 14))) sv[14] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 15))) sv[15] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 16))) sv[16] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 17))) sv[17] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 18))) sv[18] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 19))) sv[19] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 20))) sv[20] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 21))) sv[21] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 22))) sv[22] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 23))) sv[23] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 24))) sv[24] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 25))) sv[25] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 26))) sv[26] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 27))) sv[27] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 28))) sv[28] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 29))) sv[29] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 30))) sv[30] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 31))) sv[31] = -CAKE_FMHA_INF;
                            }
                            float2 _reg_reduce_max2_1 = {-CAKE_FMHA_INF, -CAKE_FMHA_INF};
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[0], sv[1]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[2], sv[3]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[4], sv[5]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[6], sv[7]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[8], sv[9]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[10], sv[11]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[12], sv[13]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[14], sv[15]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[16], sv[17]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[18], sv[19]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[20], sv[21]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[22], sv[23]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[24], sv[25]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[26], sv[27]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[28], sv[29]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[30], sv[31]));
                            float sv_max = row_max_reduce(_reg_reduce_max2_1);
                            lmax = sv_max;
                            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, lmax, 16);
                            float _max_6 = max_noftz(lmax, _shfl_xor_2);
                            lmax = _max_6;
                            if (half == 0) {
                                smem_xmax[xm_off_s + my_row] = lmax;
                            }
                        }
                        if (sm_tid == 0) {
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (rows_live != 0) {
                            float _max_7 = max_noftz(lmax, smem_xmax[xm_off_s + 64 + my_row]);
                            lmax = _max_7;
                            if (lmax > row_max + thr_raw) {
                                new_max = lmax;
                                if (row_max > -CAKE_FMHA_INF) {
                                    float _exp2_0 = approx_exp2(softmax_scale_log2 * (row_max - new_max));
                                    acc_scale = _exp2_0;
                                }
                            }
                        }
                        xm_slot_s = xm_slot_s ^ 1;
                        if (sm_tid == 0) {
                        }
                        if (rows_live != 0) {
                            float safe_max = ((new_max == -CAKE_FMHA_INF) ? 0.0f : new_max);
                            const float2 _fma_b2_2 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_3 = {(-safe_max) * softmax_scale_log2 + DEC107_LOG2_P_SCALE, (-safe_max) * softmax_scale_log2 + DEC107_LOG2_P_SCALE};
                            float2 _fma_pair_4 = fma_f32x2(make_float2(sv[0], sv[1]), _fma_b2_2, _fma_c2_3);
                            sv[0] = _fma_pair_4.x;
                            sv[1] = _fma_pair_4.y;
                            float2 _fma_pair_5 = fma_f32x2(make_float2(sv[2], sv[3]), _fma_b2_2, _fma_c2_3);
                            sv[2] = _fma_pair_5.x;
                            sv[3] = _fma_pair_5.y;
                            float2 _fma_pair_6 = fma_f32x2(make_float2(sv[4], sv[5]), _fma_b2_2, _fma_c2_3);
                            sv[4] = _fma_pair_6.x;
                            sv[5] = _fma_pair_6.y;
                            float2 _fma_pair_7 = fma_f32x2(make_float2(sv[6], sv[7]), _fma_b2_2, _fma_c2_3);
                            sv[6] = _fma_pair_7.x;
                            sv[7] = _fma_pair_7.y;
                            float2 _fma_pair_8 = fma_f32x2(make_float2(sv[8], sv[9]), _fma_b2_2, _fma_c2_3);
                            sv[8] = _fma_pair_8.x;
                            sv[9] = _fma_pair_8.y;
                            float2 _fma_pair_9 = fma_f32x2(make_float2(sv[10], sv[11]), _fma_b2_2, _fma_c2_3);
                            sv[10] = _fma_pair_9.x;
                            sv[11] = _fma_pair_9.y;
                            float2 _fma_pair_10 = fma_f32x2(make_float2(sv[12], sv[13]), _fma_b2_2, _fma_c2_3);
                            sv[12] = _fma_pair_10.x;
                            sv[13] = _fma_pair_10.y;
                            float2 _fma_pair_11 = fma_f32x2(make_float2(sv[14], sv[15]), _fma_b2_2, _fma_c2_3);
                            sv[14] = _fma_pair_11.x;
                            sv[15] = _fma_pair_11.y;
                            float2 _fma_pair_12 = fma_f32x2(make_float2(sv[16], sv[17]), _fma_b2_2, _fma_c2_3);
                            sv[16] = _fma_pair_12.x;
                            sv[17] = _fma_pair_12.y;
                            float2 _fma_pair_13 = fma_f32x2(make_float2(sv[18], sv[19]), _fma_b2_2, _fma_c2_3);
                            sv[18] = _fma_pair_13.x;
                            sv[19] = _fma_pair_13.y;
                            float2 _fma_pair_14 = fma_f32x2(make_float2(sv[20], sv[21]), _fma_b2_2, _fma_c2_3);
                            sv[20] = _fma_pair_14.x;
                            sv[21] = _fma_pair_14.y;
                            float2 _fma_pair_15 = fma_f32x2(make_float2(sv[22], sv[23]), _fma_b2_2, _fma_c2_3);
                            sv[22] = _fma_pair_15.x;
                            sv[23] = _fma_pair_15.y;
                            float2 _fma_pair_16 = fma_f32x2(make_float2(sv[24], sv[25]), _fma_b2_2, _fma_c2_3);
                            sv[24] = _fma_pair_16.x;
                            sv[25] = _fma_pair_16.y;
                            float2 _fma_pair_17 = fma_f32x2(make_float2(sv[26], sv[27]), _fma_b2_2, _fma_c2_3);
                            sv[26] = _fma_pair_17.x;
                            sv[27] = _fma_pair_17.y;
                            float2 _fma_pair_18 = fma_f32x2(make_float2(sv[28], sv[29]), _fma_b2_2, _fma_c2_3);
                            sv[28] = _fma_pair_18.x;
                            sv[29] = _fma_pair_18.y;
                            float2 _fma_pair_19 = fma_f32x2(make_float2(sv[30], sv[31]), _fma_b2_2, _fma_c2_3);
                            sv[30] = _fma_pair_19.x;
                            sv[31] = _fma_pair_19.y;
                            #pragma unroll
                            for (int _le = 0; _le < 32; _le++) {
                                sv[_le] = approx_exp2(sv[_le]);
                            }
                            float2 _reg_reduce_sum2_20 = make_float2(0.0f, 0.0f);
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[0], sv[1]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[2], sv[3]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[4], sv[5]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[6], sv[7]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[8], sv[9]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[10], sv[11]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[12], sv[13]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[14], sv[15]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[16], sv[17]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[18], sv[19]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[20], sv[21]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[22], sv[23]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[24], sv[25]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[26], sv[27]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[28], sv[29]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[30], sv[31]));
                            float sv_sum = _reg_reduce_sum2_20.x + _reg_reduce_sum2_20.y;
                            lsum = sv_sum;
                            psum = psum * acc_scale + lsum;
                            row_max = new_max;
                        }
                        if (sm_tid == 0) {
                        }
                        if (rows_live != 0) {
                            unsigned int regs_p[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                unsigned short _lo, _hi;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(sv[_lp*4 + 1]), "f"(sv[_lp*4 + 0]));
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(sv[_lp*4 + 3]), "f"(sv[_lp*4 + 2]));
                                regs_p[_lp] = (unsigned int)_lo | ((unsigned int)_hi << 16);
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                                " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "r"((unsigned int)my_s_base + sm_stage * (unsigned int)DEC107_SBUF_STRIDE), "r"(regs_p[0]), "r"(regs_p[1]), "r"(regs_p[2]), "r"(regs_p[3]), "r"(regs_p[4]), "r"(regs_p[5]), "r"(regs_p[6]), "r"(regs_p[7]));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(p_full_addr + (sm_stage) * 8), "r"((uint32_t)(32)) : "memory");
                        }
                        sm_stage += 1;
                        if (sm_stage == 2) { sm_stage = 0; sm_phase ^= 1; }
                        if (sm_tid == 0) {
                        }
                    }
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, psum, 16);
                    float total = psum + _shfl_xor_3;
                    if (sm_tid == 0) {
                    }
                    mbarrier_wait(stats_empty_addr + (st_stage_s) * 8, st_phase_s);
                    if (rows_live != 0) {
                        if (half == 0) {
                            smem_sum[st_stage_s * 64 + (unsigned int)my_row] = total;
                            smem_max[st_stage_s * 64 + (unsigned int)my_row] = row_max;
                        }
                    }
                    __syncwarp();
                    if (lane == 0) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                            :: "r"(stats_full_addr + (st_stage_s) * 8), "r"((uint32_t)(32)) : "memory");
                    }
                    st_stage_s += 1;
                    if (st_stage_s == 2) { st_stage_s = 0; st_phase_s ^= 1; }
                    if (sm_tid == 0) {
                    }
                } else {
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                    int r_row = (128 + sm_tid) / 4;
                    if (r_row < live_rows_s) {
                        int stats_stride_r = num_kv_heads * 128;
                        int o_stride_r = num_kv_heads * 16384;
                        int stats_row_r = slot_tile_base_s * 128 + r_row;
                        int f_r = 64 >> block_end_s;
                        int d0_r = block_begin_s * (4 * f_r) + (128 + sm_tid) % 4 * f_r;
                        int o_row_r = slot_tile_base_s * 16384 + r_row * HEAD_DIM + d0_r;
                        int j_r = r_row / 8;
                        int h_r = r_row % 8;
                        int q_head_r = kv_head_s * 8 + h_r;
                        int o_idx_r = ((batch_s * q_len + j_r) * num_q_heads + q_head_r) * HEAD_DIM + d0_r;
                        {
                            int n_groups_r = 16 >> block_end_s;
                            float acc_f[4];
                            float out4[4];
                            int n_pad_r = (n_chunks_s + 15) / 16 * 16;
                            #pragma unroll 1
                            for (int g_r = 0; g_r < n_groups_r; g_r++) {
                                acc_f[0] = 0.0f;
                                acc_f[1] = 0.0f;
                                acc_f[2] = 0.0f;
                                acc_f[3] = 0.0f;
                                float m_f = -1e+30f;
                                float l_f = 0.0f;
                                int o_col_r = o_row_r + g_r * 4;
                                #pragma unroll 16
                                for (int c_m = 0; c_m < n_pad_r; c_m++) {
                                    int c_c = c_m;
                                    if (n_chunks_s <= c_m) {
                                        c_c = n_chunks_s - 1;
                                    }
                                    float m_k = partial_stats[stats_row_r + c_c * stats_stride_r];
                                    float l_k = partial_stats[stats_row_r + 64 + c_c * stats_stride_r];
                                    if (n_chunks_s <= c_m) {
                                        m_k = -1e+30f;
                                        l_k = 0.0f;
                                    }
                                    float _max_10 = max_noftz(m_f, m_k);
                                    float m_new = _max_10;
                                    float _exp2_3 = approx_exp2((m_f - m_new) * softmax_scale_log2);
                                    float a_k = _exp2_3;
                                    float _exp2_4 = approx_exp2((m_k - m_new) * softmax_scale_log2);
                                    float b_k = _exp2_4;
                                    float _fma_2 = __fmaf_rn(l_k, b_k, l_f * a_k);
                                    l_f = _fma_2;
                                    float _vec_load_0[4];
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (o_col_r + c_c * o_stride_r) + 0);
                                        _vec_load_0[0 + 0] = _v4.x;
                                        _vec_load_0[0 + 1] = _v4.y;
                                        _vec_load_0[0 + 2] = _v4.z;
                                        _vec_load_0[0 + 3] = _v4.w;
                                    }
                                    #pragma unroll
                                    for (int k = 0; k < 4; k++) {
                                        float _fma_3 = __fmaf_rn(_vec_load_0[k], b_k, acc_f[k] * a_k);
                                        acc_f[k] = _fma_3;
                                    }
                                    m_f = m_new;
                                }
                                float _rcp_13 = approx_rcp(l_f);
                                float inv_f = ((l_f > 0.0f) ? _rcp_13 * out_scale : 0.0f);
                                #pragma unroll
                                for (int k4 = 0; k4 < 4; k4++) {
                                    out4[k4] = acc_f[k4] * inv_f;
                                }
                                {
                                    uint2 _pk2;
                                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                    _pk[0] = __floats2bfloat162_rn(out4[0 + 0], out4[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(out4[0 + 2], out4[0 + 3]);
                                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O_ptr + (o_idx_r + g_r * 4)))[0]) = _pk2;
                                }
                            }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
                unsigned int base_0 = work_stage_s * 16;
                unsigned int valid_1 = work_token_words[base_0];
                unsigned int kind_2 = work_token_words[base_0 + 1];
                unsigned int batch_3 = work_token_words[base_0 + 2];
                unsigned int kv_head_4 = work_token_words[base_0 + 3];
                unsigned int block_begin_5 = work_token_words[base_0 + 4];
                unsigned int block_end_6 = work_token_words[base_0 + 5];
                unsigned int seqlen_7 = work_token_words[base_0 + 6];
                unsigned int n_chunks_8 = work_token_words[base_0 + 7];
                unsigned int slot_tile_base_9 = work_token_words[base_0 + 8];
                unsigned int counter_idx_10 = work_token_words[base_0 + 9];
                unsigned int chunk_11 = work_token_words[base_0 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
                work_stage_s += 1;
                if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
                valid_s = valid_1;
                kind_s = (int)kind_2;
                block_begin_s = (int)block_begin_5;
                block_end_s = (int)block_end_6;
                seqlen_s = (int)seqlen_7;
                batch_s = (int)batch_3;
                kv_head_s = (int)kv_head_4;
                n_chunks_s = (int)n_chunks_8;
                slot_tile_base_s = (int)slot_tile_base_9;
            }
            if (sm_tid == 0) {
            }
        }
    // ---- Role: correction ----
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // correction_main
            const int warp_in_wg_c = warp % 4;
            const int corr_row = warp_in_wg_c * 32 << 16;
            int wg_tid_c = warp_in_wg_c * 32 + lane;
            int my_row_c = warp_in_wg_c * 16 + lane % 16;
            int half_c = lane / 16;
            int o_row_base = taddr + 256 + (unsigned int)corr_row;
            int my_s_base_c = taddr + (unsigned int)corr_row;
            int tok_base_c = half_c * 64 + 32;
            int rows_live_c = ((warp_in_wg_c * 16 < 32) ? 1 : 0);
            int row_j_c = my_row_c / 8;
            int _min_3 = ((row_j_c) < (q_len - 1) ? (row_j_c) : (q_len - 1));
            int vis_j_c = _min_3;
            float _rcp_14 = approx_rcp(softmax_scale_log2);
            float thr_raw_c = DEC107_THR * _rcp_14;
            int live_rows = q_len * 8;
            unsigned int sm_stage_c = 0;
            unsigned int sm_phase_c = 0;
            unsigned int xm_slot_c = 0;
            unsigned int st_stage_c = 0;
            unsigned int st_phase_c = 0;
            unsigned int pv_base = 0;
            int n_reduces_seen = 0;
            int lowb_have = 0;
            {
                int spec_tiles_c = batch_size * num_kv_heads;
                int spec_id_c = blockIdx.x;
                if (warp_in_wg_c == 0) {
                    if (spec_id_c < spec_tiles_c) {
                        int spec_batch_c = spec_id_c / num_kv_heads;
                        int spec_head_c = spec_id_c - spec_batch_c * num_kv_heads;
                        int spec_seqlen_c = seq_lens_kv[spec_batch_c];
                        int spec_blocks_c = (spec_seqlen_c + BLOCK_N - 1) / BLOCK_N;
                        int spec_max_pg_c = (spec_seqlen_c + PAGE_SIZE - 1) / PAGE_SIZE - 1;
                        int spec_nb_c = lane >> 3;
                        int spec_pg_c = lane >> 1 & 3;
                        int spec_hg_c = lane & 1;
                        int spec_block_c = spec_blocks_c - 1 - spec_nb_c;
                        if (spec_block_c >= 0) {
                            int spec_page_idx_c = spec_block_c * 4 + spec_pg_c;
                            if (spec_page_idx_c > spec_max_pg_c) {
                                spec_page_idx_c = spec_max_pg_c;
                            }
                            int spec_page_c = page_table[spec_batch_c * max_pages_per_seq + spec_page_idx_c];
                            if (spec_hg_c == 0) asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)(K)), "r"((int)(0)), "r"((int)(0)), "r"((int)(spec_head_c)), "r"((int)(spec_page_c)) : "memory");
                            if (spec_nb_c < 2) {
                                asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile [%0, {%1, %2, %3, %4, %5}];" :: "l"((uint64_t)(V)), "r"((int)(0)), "r"((int)(0)), "r"((int)(spec_hg_c)), "r"((int)(spec_head_c)), "r"((int)(spec_page_c)) : "memory");
                            }
                        }
                        if (lane == 0) {
                            asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)(Q)), "r"((int)(0)), "r"((int)(spec_head_c * 8)), "r"((int)(spec_batch_c * q_len)), "r"((int)(0)) : "memory");
                        }
                    }
                }
            }
            unsigned int work_stage_c = 0;
            unsigned int _phase_work_full_1 = 0;
            mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
            unsigned int base_1 = work_stage_c * 16;
            unsigned int valid_2 = work_token_words[base_1];
            unsigned int kind_1 = work_token_words[base_1 + 1];
            unsigned int batch_1 = work_token_words[base_1 + 2];
            unsigned int kv_head_1 = work_token_words[base_1 + 3];
            unsigned int block_begin_1 = work_token_words[base_1 + 4];
            unsigned int block_end_1 = work_token_words[base_1 + 5];
            unsigned int seqlen_1 = work_token_words[base_1 + 6];
            unsigned int n_chunks_1 = work_token_words[base_1 + 7];
            unsigned int slot_tile_base_1 = work_token_words[base_1 + 8];
            unsigned int counter_idx_1 = work_token_words[base_1 + 9];
            unsigned int chunk_1 = work_token_words[base_1 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
            work_stage_c += 1;
            if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
            unsigned int valid_c = valid_2;
            int kind_c = (int)kind_1;
            int batch_c = (int)batch_1;
            int kv_head_c = (int)kv_head_1;
            int block_begin_c = (int)block_begin_1;
            int block_end_c = (int)block_end_1;
            int seqlen_c = (int)seqlen_1;
            int n_chunks_c = (int)n_chunks_1;
            int slot_tile_base_c = (int)slot_tile_base_1;
            int counter_idx_c = (int)counter_idx_1;
            int chunk_c = (int)chunk_1;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < max_items; _tile_iter_c++) {
                if (valid_c == 0) {
                    break;
                }
                if (kind_c == 0) {
                    int cnt_c = block_end_c - block_begin_c;
                    int my_slot = slot_tile_base_c + chunk_c * num_kv_heads;
                    int vis_min_c = seqlen_c - (q_len - 1);
                    int vis_col_c = vis_min_c + vis_j_c;
                    float row_max_c = -CAKE_FMHA_INF;
                    float psum_c = 0.0f;
                    #pragma unroll 1
                    for (int n_1 = 0; n_1 < cnt_c; n_1++) {
                        if (wg_tid_c == 0) {
                        }
                        mbarrier_wait(s_full_addr + (sm_stage_c) * 8, sm_phase_c);
                        if (wg_tid_c == 0) {
                        }
                        int my_block_c = block_begin_c + cnt_c - 1 - n_1;
                        int blk_pos_c = my_block_c * BLOCK_N + tok_base_c;
                        float sv_c[32];
                        float lmax_c = -CAKE_FMHA_INF;
                        float new_max_c = row_max_c;
                        float acc_scale_c = 1.0f;
                        int xm_off_c = (int)xm_slot_c * 128;
                        if (rows_live_c != 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sv_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[3])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[5])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[6])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[7])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[8])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[9])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[10])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[11])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[12])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[13])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[14])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[15])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[16])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[17])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[18])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[19])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[20])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[21])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[22])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[23])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[24])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[25])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[26])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[27])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[28])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[29])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[30])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[31]))
                                : "r"((unsigned int)my_s_base_c + sm_stage_c * (unsigned int)DEC107_SBUF_STRIDE + 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            int n_vis_c = vis_col_c - blk_pos_c;
                            if (my_row_c >= 32) {
                                n_vis_c = 0;
                            }
                            if (n_vis_c < 32) {
                                int _max_11 = ((n_vis_c) > (0) ? (n_vis_c) : (0));
                                int n_lo_c = _max_11;
                                uint32_t _slice_lo_mask_1;
                                {
                                    int _lim_0 = n_lo_c;
                                    if (_lim_0 <= 0) { _slice_lo_mask_1 = 0u; }
                                    else if (_lim_0 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_0));
                                    }
                                }
                                if (!(_slice_lo_mask_1 & (1u << 0))) sv_c[0] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 1))) sv_c[1] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 2))) sv_c[2] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 3))) sv_c[3] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 4))) sv_c[4] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 5))) sv_c[5] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 6))) sv_c[6] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 7))) sv_c[7] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 8))) sv_c[8] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 9))) sv_c[9] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 10))) sv_c[10] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 11))) sv_c[11] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 12))) sv_c[12] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 13))) sv_c[13] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 14))) sv_c[14] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 15))) sv_c[15] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 16))) sv_c[16] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 17))) sv_c[17] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 18))) sv_c[18] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 19))) sv_c[19] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 20))) sv_c[20] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 21))) sv_c[21] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 22))) sv_c[22] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 23))) sv_c[23] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 24))) sv_c[24] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 25))) sv_c[25] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 26))) sv_c[26] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 27))) sv_c[27] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 28))) sv_c[28] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 29))) sv_c[29] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 30))) sv_c[30] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 31))) sv_c[31] = -CAKE_FMHA_INF;
                            }
                            float2 _reg_reduce_max2_1 = {-CAKE_FMHA_INF, -CAKE_FMHA_INF};
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[0], sv_c[1]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[2], sv_c[3]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[4], sv_c[5]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[6], sv_c[7]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[8], sv_c[9]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[10], sv_c[11]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[12], sv_c[13]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[14], sv_c[15]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[16], sv_c[17]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[18], sv_c[19]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[20], sv_c[21]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[22], sv_c[23]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[24], sv_c[25]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[26], sv_c[27]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[28], sv_c[29]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[30], sv_c[31]));
                            float sv_c_max = row_max_reduce(_reg_reduce_max2_1);
                            lmax_c = sv_c_max;
                            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, lmax_c, 16);
                            float _max_12 = max_noftz(lmax_c, _shfl_xor_4);
                            lmax_c = _max_12;
                            if (half_c == 0) {
                                smem_xmax[xm_off_c + 64 + my_row_c] = lmax_c;
                            }
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (rows_live_c != 0) {
                            float _max_13 = max_noftz(lmax_c, smem_xmax[xm_off_c + my_row_c]);
                            lmax_c = _max_13;
                            if (lmax_c > row_max_c + thr_raw_c) {
                                new_max_c = lmax_c;
                                if (row_max_c > -CAKE_FMHA_INF) {
                                    float _exp2_5 = approx_exp2(softmax_scale_log2 * (row_max_c - new_max_c));
                                    acc_scale_c = _exp2_5;
                                }
                            }
                        }
                        xm_slot_c = xm_slot_c ^ 1;
                        if (wg_tid_c == 0) {
                        }
                        if (rows_live_c != 0) {
                            float safe_max_c = ((new_max_c == -CAKE_FMHA_INF) ? 0.0f : new_max_c);
                            const float2 _fma_b2_2 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_3 = {(-safe_max_c) * softmax_scale_log2 + DEC107_LOG2_P_SCALE, (-safe_max_c) * softmax_scale_log2 + DEC107_LOG2_P_SCALE};
                            float2 _fma_pair_4 = fma_f32x2(make_float2(sv_c[0], sv_c[1]), _fma_b2_2, _fma_c2_3);
                            sv_c[0] = _fma_pair_4.x;
                            sv_c[1] = _fma_pair_4.y;
                            float2 _fma_pair_5 = fma_f32x2(make_float2(sv_c[2], sv_c[3]), _fma_b2_2, _fma_c2_3);
                            sv_c[2] = _fma_pair_5.x;
                            sv_c[3] = _fma_pair_5.y;
                            float2 _fma_pair_6 = fma_f32x2(make_float2(sv_c[4], sv_c[5]), _fma_b2_2, _fma_c2_3);
                            sv_c[4] = _fma_pair_6.x;
                            sv_c[5] = _fma_pair_6.y;
                            float2 _fma_pair_7 = fma_f32x2(make_float2(sv_c[6], sv_c[7]), _fma_b2_2, _fma_c2_3);
                            sv_c[6] = _fma_pair_7.x;
                            sv_c[7] = _fma_pair_7.y;
                            float2 _fma_pair_8 = fma_f32x2(make_float2(sv_c[8], sv_c[9]), _fma_b2_2, _fma_c2_3);
                            sv_c[8] = _fma_pair_8.x;
                            sv_c[9] = _fma_pair_8.y;
                            float2 _fma_pair_9 = fma_f32x2(make_float2(sv_c[10], sv_c[11]), _fma_b2_2, _fma_c2_3);
                            sv_c[10] = _fma_pair_9.x;
                            sv_c[11] = _fma_pair_9.y;
                            float2 _fma_pair_10 = fma_f32x2(make_float2(sv_c[12], sv_c[13]), _fma_b2_2, _fma_c2_3);
                            sv_c[12] = _fma_pair_10.x;
                            sv_c[13] = _fma_pair_10.y;
                            float2 _fma_pair_11 = fma_f32x2(make_float2(sv_c[14], sv_c[15]), _fma_b2_2, _fma_c2_3);
                            sv_c[14] = _fma_pair_11.x;
                            sv_c[15] = _fma_pair_11.y;
                            float2 _fma_pair_12 = fma_f32x2(make_float2(sv_c[16], sv_c[17]), _fma_b2_2, _fma_c2_3);
                            sv_c[16] = _fma_pair_12.x;
                            sv_c[17] = _fma_pair_12.y;
                            float2 _fma_pair_13 = fma_f32x2(make_float2(sv_c[18], sv_c[19]), _fma_b2_2, _fma_c2_3);
                            sv_c[18] = _fma_pair_13.x;
                            sv_c[19] = _fma_pair_13.y;
                            float2 _fma_pair_14 = fma_f32x2(make_float2(sv_c[20], sv_c[21]), _fma_b2_2, _fma_c2_3);
                            sv_c[20] = _fma_pair_14.x;
                            sv_c[21] = _fma_pair_14.y;
                            float2 _fma_pair_15 = fma_f32x2(make_float2(sv_c[22], sv_c[23]), _fma_b2_2, _fma_c2_3);
                            sv_c[22] = _fma_pair_15.x;
                            sv_c[23] = _fma_pair_15.y;
                            float2 _fma_pair_16 = fma_f32x2(make_float2(sv_c[24], sv_c[25]), _fma_b2_2, _fma_c2_3);
                            sv_c[24] = _fma_pair_16.x;
                            sv_c[25] = _fma_pair_16.y;
                            float2 _fma_pair_17 = fma_f32x2(make_float2(sv_c[26], sv_c[27]), _fma_b2_2, _fma_c2_3);
                            sv_c[26] = _fma_pair_17.x;
                            sv_c[27] = _fma_pair_17.y;
                            float2 _fma_pair_18 = fma_f32x2(make_float2(sv_c[28], sv_c[29]), _fma_b2_2, _fma_c2_3);
                            sv_c[28] = _fma_pair_18.x;
                            sv_c[29] = _fma_pair_18.y;
                            float2 _fma_pair_19 = fma_f32x2(make_float2(sv_c[30], sv_c[31]), _fma_b2_2, _fma_c2_3);
                            sv_c[30] = _fma_pair_19.x;
                            sv_c[31] = _fma_pair_19.y;
                            #pragma unroll
                            for (int _le = 0; _le < 32; _le++) {
                                sv_c[_le] = approx_exp2(sv_c[_le]);
                            }
                            float2 _reg_reduce_sum2_20 = make_float2(0.0f, 0.0f);
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[0], sv_c[1]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[2], sv_c[3]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[4], sv_c[5]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[6], sv_c[7]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[8], sv_c[9]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[10], sv_c[11]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[12], sv_c[13]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[14], sv_c[15]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[16], sv_c[17]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[18], sv_c[19]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[20], sv_c[21]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[22], sv_c[23]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[24], sv_c[25]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[26], sv_c[27]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[28], sv_c[29]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[30], sv_c[31]));
                            float sv_c_sum = _reg_reduce_sum2_20.x + _reg_reduce_sum2_20.y;
                            float lsum_c = sv_c_sum;
                            psum_c = psum_c * acc_scale_c + lsum_c;
                            row_max_c = new_max_c;
                            unsigned int regs_pc[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                unsigned short _lo, _hi;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(sv_c[_lp*4 + 1]), "f"(sv_c[_lp*4 + 0]));
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(sv_c[_lp*4 + 3]), "f"(sv_c[_lp*4 + 2]));
                                regs_pc[_lp] = (unsigned int)_lo | ((unsigned int)_hi << 16);
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                                " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "r"((unsigned int)my_s_base_c + sm_stage_c * (unsigned int)DEC107_SBUF_STRIDE + 8), "r"(regs_pc[0]), "r"(regs_pc[1]), "r"(regs_pc[2]), "r"(regs_pc[3]), "r"(regs_pc[4]), "r"(regs_pc[5]), "r"(regs_pc[6]), "r"(regs_pc[7]));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        int _vote_2 = __any_sync(0xFFFFFFFF, acc_scale_c != 1.0f);
                        if (_vote_2 != 0) {
                            if (n_1 > 0) {
                                unsigned int k_c = pv_base + (unsigned int)n_1 - 1;
                                int o_st_c = (int)(k_c & 1);
                                int o_ph_c = (int)(k_c >> 1 & 1);
                                mbarrier_wait(o_ready_addr + (o_st_c) * 8, o_ph_c);
                                if (wg_tid_c == 0) {
                                }
                                asm volatile("tcgen05.fence::after_thread_sync;");
#pragma unroll 1
for (int _dp = 0; _dp < 2; _dp++) {
float _tmem_load_0[64];
asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 128;"
    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
    : "r"(o_row_base + _dp * 64));
asm volatile("tcgen05.wait::ld.sync.aligned;");
asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 128;"
    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
    : "r"(o_row_base + _dp * 64 + 32));
asm volatile("tcgen05.wait::ld.sync.aligned;");
const float2 _scale2_21 = {acc_scale_c, acc_scale_c};
#pragma unroll
for (int _ls = 0; _ls < 32; _ls++)
    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_21);
asm volatile("tcgen05.st.sync.aligned.16x32bx2.x64.b32 [%0], 128, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
    :: "r"(o_row_base + _dp * 64), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[63])));
asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
                            }
                        }
                        if (n_1 >= 2) {
                            unsigned int k_o = pv_base + (unsigned int)n_1 - 2;
                            int o_st_o = (int)(k_o & 1);
                            int o_ph_o = (int)(k_o >> 1 & 1);
                            mbarrier_wait(o_ready_addr + (o_st_o) * 8, o_ph_o);
                        }
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(p_full_addr + (sm_stage_c) * 8), "r"((uint32_t)(32)) : "memory");
                        }
                        sm_stage_c += 1;
                        if (sm_stage_c == 2) { sm_stage_c = 0; sm_phase_c ^= 1; }
                        if (wg_tid_c == 0) {
                        }
                    }
                    if (cnt_c >= 2) {
                        unsigned int k_e2 = pv_base + (unsigned int)cnt_c - 2;
                        int o_st_e2 = (int)(k_e2 & 1);
                        int o_ph_e2 = (int)(k_e2 >> 1 & 1);
                        mbarrier_wait(o_ready_addr + (o_st_e2) * 8, o_ph_e2);
                    }
                    unsigned int k_e = pv_base + (unsigned int)cnt_c - 1;
                    int o_st_e = (int)(k_e & 1);
                    int o_ph_e = (int)(k_e >> 1 & 1);
                    mbarrier_wait(o_ready_addr + (o_st_e) * 8, o_ph_e);
                    pv_base = pv_base + (unsigned int)cnt_c;
                    if (wg_tid_c == 0) { DEC107_STAMP_ONCE(7); DEC107_COUNT(12, 1); }
                    asm volatile("tcgen05.fence::after_thread_sync;");
float _tmem_load_1[128];
asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 128;"
    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
    : "r"(o_row_base + 0));
asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 128;"
    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
    : "r"(o_row_base + 32));
asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 128;"
    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[64])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[65])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[66])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[67])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[68])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[69])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[70])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[71])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[72])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[73])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[74])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[75])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[76])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[77])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[78])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[79])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[80])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[81])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[82])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[83])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[84])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[85])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[86])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[87])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[88])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[89])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[90])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[91])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[92])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[93])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[94])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[95]))
    : "r"(o_row_base + 64));
asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 128;"
    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[96])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[97])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[98])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[99])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[100])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[101])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[102])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[103])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[104])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[105])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[106])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[107])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[108])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[109])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[110])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[111])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[112])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[113])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[114])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[115])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[116])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[117])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[118])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[119])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[120])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[121])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[122])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[123])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[124])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[125])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[126])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[127]))
    : "r"(o_row_base + 96));
asm volatile("tcgen05.wait::ld.sync.aligned;");
                    __syncwarp();
                    if (lane == 0) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                            :: "r"(o_empty_addr), "r"((uint32_t)(32)) : "memory");
                    }
                    if (wg_tid_c == 0) {
                    }
                    mbarrier_wait(stats_full_addr + (st_stage_c) * 8, st_phase_c);
                    int st_off = (int)st_stage_c * 64;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, psum_c, 16);
                    float total_c = psum_c + _shfl_xor_5;
                    if (rows_live_c != 0) {
                        if (half_c == 0) {
                            smem_sum[st_off + my_row_c] = smem_sum[st_off + my_row_c] + total_c;
                        }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (wg_tid_c == 0) {
                    }
                    int publish_split = ((n_chunks_c > 1) ? 1 : 0);
#ifdef DEC107_LOWB
                    if (true) {
                        float* sp = reinterpret_cast<float*>(smem_raw + 25600);
                        float* ss = sp + 32 * 256;
                        if (my_row_c < 32) {
                            #pragma unroll
                            for (int i = 0; i < 128; i += 4)
                                *reinterpret_cast<float4*>(sp + my_row_c * 256 + half_c * 128 + i) =
                                    make_float4(_tmem_load_1[i], _tmem_load_1[i + 1], _tmem_load_1[i + 2], _tmem_load_1[i + 3]);
                        }
                        if (wg_tid_c < 32) {
                            ss[wg_tid_c] = smem_max[st_off + wg_tid_c];
                            ss[32 + wg_tid_c] = smem_sum[st_off + wg_tid_c];
                        }
                        lowb_have = 1;
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(stats_empty_addr + (st_stage_c) * 8);
                        }
                    } else
#endif
                    if (publish_split == 0) {
                        float row_sum_c = 1.0f;
                        if (my_row_c < 32) {
                            row_sum_c = smem_sum[st_off + my_row_c];
                        }
                        float _rcp_15 = approx_rcp(row_sum_c);
                        float inv_c = ((row_sum_c > 0.0f) ? _rcp_15 * out_scale : 0.0f);
                        if (my_row_c < live_rows) {
                            int j_c = my_row_c / 8;
                            int h_c = my_row_c % 8;
                            int q_head_c = kv_head_c * 8 + h_c;
                            int o_idx = ((batch_c * q_len + j_c) * num_q_heads + q_head_c) * HEAD_DIM + half_c * 128;
                            #pragma unroll
                            for (int off = 0; off < 128; off += 8) {
                                {
                                    const float2 _prescale2_22 = {inv_c, inv_c};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 4; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_1[off])[_ps], _prescale2_22);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        _tmem_load_1[off + _ps] *= inv_c;
                                    #endif
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_1[off + 0], _tmem_load_1[off + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_1[off + 2], _tmem_load_1[off + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_tmem_load_1[off + 4], _tmem_load_1[off + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_tmem_load_1[off + 6], _tmem_load_1[off + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx + off)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(stats_empty_addr + (st_stage_c) * 8);
                        }
                    } else if (n_chunks_c == 2) {
                        if (my_row_c < 32) {
                            int p_base = my_slot * 16384 + my_row_c * HEAD_DIM + half_c * 128;
                            #pragma unroll
                            for (int off_1 = 0; off_1 < 128; off_1 += 4) {
                                {
                                    float4 _v4 = make_float4(_tmem_load_1[off_1 + 0], _tmem_load_1[off_1 + 1], _tmem_load_1[off_1 + 2], _tmem_load_1[off_1 + 3]);
                                    *reinterpret_cast<float4*>(partial_o + (p_base + off_1) + 0) = _v4;
                                }
                            }
                        }
                        if (wg_tid_c < 32) {
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + wg_tid_c)) + (0)) = smem_max[st_off + wg_tid_c];
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + 64 + wg_tid_c)) + (0)) = smem_sum[st_off + wg_tid_c];
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        if (wg_tid_c == 0) {
                            asm volatile("fence.release.gpu;" ::: "memory");
                            unsigned int _atomic_old_4;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                : "=r"(_atomic_old_4) : "l"(&tile_counters[counter_idx_c * 4]), "r"(static_cast<uint32_t>(1)) : "memory");
                            unsigned int arrived_old = _atomic_old_4;
                            smem_corr_flag[0] = arrived_old + 1;
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        unsigned int arrived_c = smem_corr_flag[0];
                        if ((int)arrived_c == n_chunks_c) {
                            if (wg_tid_c == 0) {
                            }
                            asm volatile("fence.acquire.gpu;" ::: "memory");
                            int other_slot = slot_tile_base_c + (1 - chunk_c) * num_kv_heads;
                            float w_s_c = 0.0f;
                            float w_o_c = 0.0f;
                            if (my_row_c < 32) {
                                float m_o = partial_stats[other_slot * 128 + my_row_c];
                                float l_o = partial_stats[other_slot * 128 + 64 + my_row_c];
                                float m_s = smem_max[st_off + my_row_c];
                                float l_s = smem_sum[st_off + my_row_c];
                                float _max_14 = max_noftz(m_s, m_o);
                                float m_row_i = _max_14;
                                float _exp2_6 = approx_exp2((m_s - m_row_i) * softmax_scale_log2);
                                float w_s = _exp2_6;
                                float _exp2_7 = approx_exp2((m_o - m_row_i) * softmax_scale_log2);
                                float w_o = _exp2_7;
                                float _fma_4 = __fmaf_rn(w_s, l_s, w_o * l_o);
                                float den_i = _fma_4;
                                if (chunk_c != 0) {
                                    float _fma_5 = __fmaf_rn(w_o, l_o, w_s * l_s);
                                    den_i = _fma_5;
                                }
                                float _rcp_16 = approx_rcp(den_i);
                                float inv_i = ((den_i > 0.0f) ? _rcp_16 * out_scale : 0.0f);
                                w_s_c = w_s * inv_i;
                                w_o_c = w_o * inv_i;
                            }
                            int oth_base = other_slot * 16384 + my_row_c * HEAD_DIM + half_c * 128;
                            if (my_row_c < live_rows) {
                                #pragma unroll
                                for (int c0 = 0; c0 < 128; c0 += 4) {
                                    float _vec_load_1[4];
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (oth_base + c0) + 0);
                                        _vec_load_1[0 + 0] = _v4.x;
                                        _vec_load_1[0 + 1] = _v4.y;
                                        _vec_load_1[0 + 2] = _v4.z;
                                        _vec_load_1[0 + 3] = _v4.w;
                                    }
                                    #pragma unroll
                                    for (int c = 0; c < 4; c++) {
                                        {
                                            float _fma_6 = __fmaf_rn(_tmem_load_1[c0 + c], w_s_c, _vec_load_1[c] * w_o_c);
                                            float _fma_7 = __fmaf_rn(_vec_load_1[c], w_o_c, _tmem_load_1[c0 + c] * w_s_c);
                                            float o_det_d1 = ((chunk_c == 0) ? _fma_6 : _fma_7);
                                            _tmem_load_1[c0 + c] = o_det_d1;
                                        }
                                    }
                                }
                                int j_c_1 = my_row_c / 8;
                                int h_c_1 = my_row_c % 8;
                                int q_head_c_1 = kv_head_c * 8 + h_c_1;
                                int o_idx_1 = ((batch_c * q_len + j_c_1) * num_q_heads + q_head_c_1) * HEAD_DIM + half_c * 128;
                                #pragma unroll
                                for (int off_2 = 0; off_2 < 128; off_2 += 8) {
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 0], _tmem_load_1[off_2 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 2], _tmem_load_1[off_2 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 4], _tmem_load_1[off_2 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 6], _tmem_load_1[off_2 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx_1 + off_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                }
                            }
                            asm volatile("barrier.sync 9, 128;" ::: "memory");
                            if (elect_sync()) {
                                mbarrier_arrive(stats_empty_addr + (st_stage_c) * 8);
                            }
                            n_reduces_seen = n_reduces_seen + 1;
                            if (wg_tid_c == 0) {
                                *(reinterpret_cast<unsigned int*>(tile_counters + (counter_idx_c * 4)) + (0)) = 0;
                            }
                        } else {
                            __syncwarp();
                            if (elect_sync()) {
                                mbarrier_arrive(stats_empty_addr + (st_stage_c) * 8);
                            }
                        }
                    } else {
                        if (my_row_c < 32) {
                            int p_base_1 = my_slot * 16384 + my_row_c * HEAD_DIM + half_c * 128;
                            #pragma unroll
                            for (int off_3 = 0; off_3 < 128; off_3 += 4) {
                                {
                                    float4 _v4 = make_float4(_tmem_load_1[off_3 + 0], _tmem_load_1[off_3 + 1], _tmem_load_1[off_3 + 2], _tmem_load_1[off_3 + 3]);
                                    *reinterpret_cast<float4*>(partial_o + (p_base_1 + off_3) + 0) = _v4;
                                }
                            }
                        }
                        if (wg_tid_c < 32) {
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + wg_tid_c)) + (0)) = smem_max[st_off + wg_tid_c];
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + 64 + wg_tid_c)) + (0)) = smem_sum[st_off + wg_tid_c];
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(stats_empty_addr + (st_stage_c) * 8);
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        if (wg_tid_c == 0) {
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;"
                                :: "l"(&tile_counters[counter_idx_c * 4]), "r"(static_cast<uint32_t>(1)) : "memory");
                        }
                    }
                    if (wg_tid_c == 0) DEC107_STAMP(8);
                    st_stage_c += 1;
                    if (st_stage_c == 2) { st_stage_c = 0; st_phase_c ^= 1; }
                    if (wg_tid_c == 0) {
                        if (_tile_iter_c == 0) {
                        }
                    }
                } else {
                    if (wg_tid_c == 0) {
                    }
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                    if (wg_tid_c == 0) { DEC107_STAMP_ONCE(9); DEC107_COUNT(14, 1); }
                    int r_row_1 = wg_tid_c / 4;
                    if (r_row_1 < live_rows) {
                        int stats_stride_r_1 = num_kv_heads * 128;
                        int o_stride_r_1 = num_kv_heads * 16384;
                        int stats_row_r_1 = slot_tile_base_c * 128 + r_row_1;
                        int f_r_1 = 64 >> block_end_c;
                        int d0_r_1 = block_begin_c * (4 * f_r_1) + wg_tid_c % 4 * f_r_1;
                        int o_row_r_1 = slot_tile_base_c * 16384 + r_row_1 * HEAD_DIM + d0_r_1;
                        int j_r_1 = r_row_1 / 8;
                        int h_r_1 = r_row_1 % 8;
                        int q_head_r_1 = kv_head_c * 8 + h_r_1;
                        int o_idx_r_1 = ((batch_c * q_len + j_r_1) * num_q_heads + q_head_r_1) * HEAD_DIM + d0_r_1;
                        {
                            int n_groups_r_1 = 16 >> block_end_c;
                            float acc_f_1[4];
                            float out4_1[4];
                            int n_pad_r_1 = (n_chunks_c + 15) / 16 * 16;
                            #pragma unroll 1
                            for (int g_r_1 = 0; g_r_1 < n_groups_r_1; g_r_1++) {
                                acc_f_1[0] = 0.0f;
                                acc_f_1[1] = 0.0f;
                                acc_f_1[2] = 0.0f;
                                acc_f_1[3] = 0.0f;
                                float m_f_1 = -1e+30f;
                                float l_f_1 = 0.0f;
                                int o_col_r_1 = o_row_r_1 + g_r_1 * 4;
                                #pragma unroll 16
                                for (int c_m_1 = 0; c_m_1 < n_pad_r_1; c_m_1++) {
                                    int c_c_1 = c_m_1;
                                    if (n_chunks_c <= c_m_1) {
                                        c_c_1 = n_chunks_c - 1;
                                    }
                                    float m_k_1 = partial_stats[stats_row_r_1 + c_c_1 * stats_stride_r_1];
                                    float l_k_1 = partial_stats[stats_row_r_1 + 64 + c_c_1 * stats_stride_r_1];
                                    if (n_chunks_c <= c_m_1) {
                                        m_k_1 = -1e+30f;
                                        l_k_1 = 0.0f;
                                    }
                                    float _max_17 = max_noftz(m_f_1, m_k_1);
                                    float m_new_1 = _max_17;
                                    float _exp2_10 = approx_exp2((m_f_1 - m_new_1) * softmax_scale_log2);
                                    float a_k_1 = _exp2_10;
                                    float _exp2_11 = approx_exp2((m_k_1 - m_new_1) * softmax_scale_log2);
                                    float b_k_1 = _exp2_11;
                                    float _fma_11 = __fmaf_rn(l_k_1, b_k_1, l_f_1 * a_k_1);
                                    l_f_1 = _fma_11;
                                    float _vec_load_2[4];
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (o_col_r_1 + c_c_1 * o_stride_r_1) + 0);
                                        _vec_load_2[0 + 0] = _v4.x;
                                        _vec_load_2[0 + 1] = _v4.y;
                                        _vec_load_2[0 + 2] = _v4.z;
                                        _vec_load_2[0 + 3] = _v4.w;
                                    }
                                    #pragma unroll
                                    for (int k_1 = 0; k_1 < 4; k_1++) {
                                        float _fma_12 = __fmaf_rn(_vec_load_2[k_1], b_k_1, acc_f_1[k_1] * a_k_1);
                                        acc_f_1[k_1] = _fma_12;
                                    }
                                    m_f_1 = m_new_1;
                                }
                                float _rcp_18 = approx_rcp(l_f_1);
                                float inv_f_1 = ((l_f_1 > 0.0f) ? _rcp_18 * out_scale : 0.0f);
                                #pragma unroll
                                for (int k4_1 = 0; k4_1 < 4; k4_1++) {
                                    out4_1[k4_1] = acc_f_1[k4_1] * inv_f_1;
                                }
                                {
                                    uint2 _pk2;
                                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                    _pk[0] = __floats2bfloat162_rn(out4_1[0 + 0], out4_1[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(out4_1[0 + 2], out4_1[0 + 3]);
                                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O_ptr + (o_idx_r_1 + g_r_1 * 4)))[0]) = _pk2;
                                }
                            }
                            if (n_groups_r_1 == 0) {
                                float a0 = 0.0f, a1 = 0.0f;
                                float m_f = -1e+30f;
                                float l_f = 0.0f;
                                #pragma unroll 16
                                for (int c_m = 0; c_m < n_pad_r_1; c_m++) {
                                    const int c_c = (c_m < n_chunks_c) ? c_m : (n_chunks_c - 1);
                                    float m_k = partial_stats[stats_row_r_1 + c_c * stats_stride_r_1];
                                    float l_k = partial_stats[stats_row_r_1 + 64 + c_c * stats_stride_r_1];
                                    if (n_chunks_c <= c_m) {
                                        m_k = -1e+30f;
                                        l_k = 0.0f;
                                    }
                                    const float m_new = max_noftz(m_f, m_k);
                                    const float a_k = approx_exp2((m_f - m_new) * softmax_scale_log2);
                                    const float b_k = approx_exp2((m_k - m_new) * softmax_scale_log2);
                                    l_f = __fmaf_rn(l_k, b_k, l_f * a_k);
                                    const float* src = partial_o + (o_row_r_1 + c_c * o_stride_r_1);
                                    float v0, v1 = 0.0f;
                                    if (f_r_1 == 2) {
                                        const float2 v2 = *reinterpret_cast<const float2*>(src);
                                        v0 = v2.x; v1 = v2.y;
                                    } else {
                                        v0 = src[0];
                                    }
                                    a0 = __fmaf_rn(v0, b_k, a0 * a_k);
                                    a1 = __fmaf_rn(v1, b_k, a1 * a_k);
                                    m_f = m_new;
                                }
                                const float inv_s = ((l_f > 0.0f) ? approx_rcp(l_f) * out_scale : 0.0f);
                                if (f_r_1 == 2) {
                                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O_ptr + o_idx_r_1))[0]) =
                                        __floats2bfloat162_rn(a0 * inv_s, a1 * inv_s);
                                } else {
                                    ((__nv_bfloat16*)O_ptr)[o_idx_r_1] = __float2bfloat16_rn(a0 * inv_s);
                                }
                            }
                        }
                    }
                    if (wg_tid_c == 0) DEC107_STAMP(10);
                    n_reduces_seen = n_reduces_seen + 1;
                    if (wg_tid_c == 0) {
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
                unsigned int base_0_1 = work_stage_c * 16;
                unsigned int valid_1_1 = work_token_words[base_0_1];
                unsigned int kind_2_1 = work_token_words[base_0_1 + 1];
                unsigned int batch_3_1 = work_token_words[base_0_1 + 2];
                unsigned int kv_head_4_1 = work_token_words[base_0_1 + 3];
                unsigned int block_begin_5_1 = work_token_words[base_0_1 + 4];
                unsigned int block_end_6_1 = work_token_words[base_0_1 + 5];
                unsigned int seqlen_7_1 = work_token_words[base_0_1 + 6];
                unsigned int n_chunks_8_1 = work_token_words[base_0_1 + 7];
                unsigned int slot_tile_base_9_1 = work_token_words[base_0_1 + 8];
                unsigned int counter_idx_10_1 = work_token_words[base_0_1 + 9];
                unsigned int chunk_11_1 = work_token_words[base_0_1 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                work_stage_c += 1;
                if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
                valid_c = valid_1_1;
                kind_c = (int)kind_2_1;
                batch_c = (int)batch_3_1;
                kv_head_c = (int)kv_head_4_1;
                block_begin_c = (int)block_begin_5_1;
                block_end_c = (int)block_end_6_1;
                seqlen_c = (int)seqlen_7_1;
                n_chunks_c = (int)n_chunks_8_1;
                slot_tile_base_c = (int)slot_tile_base_9_1;
                counter_idx_c = (int)counter_idx_10_1;
                chunk_c = (int)chunk_11_1;
            }
            if (wg_tid_c == 0) {
            }
#ifdef DEC107_LOWB
            {
                const int t = (int)work_token_words[48];
                if (t >= 0) {
                    const int kc = (int)work_token_words[49], rank = (int)work_token_words[50];
                    const int K = (int)work_token_words[51], G = (int)work_token_words[52];
                    const int C = (int)dec107_ncta();
                    const int bidx = t / num_kv_heads, hk = t % num_kv_heads;
                    const unsigned sp_a = (unsigned)(smem + 25600);
                    const unsigned ss_a = sp_a + 32 * 256 * 4;
                    float* sp = reinterpret_cast<float*>(smem_raw + 25600);
                    float* ss = sp + 32 * 256;
                    if (!lowb_have) {
                        for (int i = wg_tid_c; i < 32 * 256; i += 128) sp[i] = 0.0f;
                        if (wg_tid_c < 32) { ss[wg_tid_c] = -1e+30f; ss[32 + wg_tid_c] = 0.0f; }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (wg_tid_c == 0) {
                        DEC107_STAMP(9);
                        for (int p = 0; p < C; p++) dec107_remote_arrive(dec107_mapa((unsigned)(smem + 384), (unsigned)p));
                    }
                    dec107_wait_cluster((unsigned)(smem + 384), 0);
                    if (wg_tid_c == 0) DEC107_STAMP2(0);
                    // merge d-slice `rank` over the C peers (fixed rank order)
                    const int Ds = 256 / C, per = Ds / 4;            // per = 8 (C=8) or 4 (C=16)
                    const int row = wg_tid_c >> 2, sub = wg_tid_c & 3;
                    const int d = rank * Ds + sub * per;
                    const int live = q_len * 8;
                    float M = -1e+30f, L = 0.0f;
                    float wv[8];
                    if (row < live) {
                        float mv[8], lv[8];
                        #pragma unroll
                        for (int p = 0; p < 8; p++) {
                            if (p < C) {
                                const unsigned pb = dec107_mapa(ss_a, (unsigned)p);
                                mv[p] = dec107_ldc(pb + row * 4);
                                lv[p] = dec107_ldc(pb + (32 + row) * 4);
                            } else { mv[p] = -1e+30f; lv[p] = 0.0f; }
                        }
                        #pragma unroll
                        for (int p = 0; p < 8; p++) M = max_noftz(M, mv[p]);
                        #pragma unroll
                        for (int p = 0; p < 8; p++) {
                            wv[p] = (p < C) ? approx_exp2((mv[p] - M) * softmax_scale_log2) : 0.0f;
                            L = __fmaf_rn(wv[p], lv[p], L);
                        }
                    }
                    float accs[32];
                    #pragma unroll
                    for (int i = 0; i < 32; i++) accs[i] = 0.0f;
                    if (row < live) {
                        #pragma unroll
                        for (int ch = 0; ch < 32; ch += 8) {
                            if (ch < per) {
                                float4 va[8], vb[8];
                                #pragma unroll
                                for (int p = 0; p < 8; p++) {
                                    if (p < C) {
                                        const unsigned oa = dec107_mapa(sp_a, (unsigned)p) + (row * 256 + d + ch) * 4;
                                        va[p] = dec107_ldc4(oa); vb[p] = dec107_ldc4(oa + 16);
                                    }
                                }
                                #pragma unroll
                                for (int p = 0; p < 8; p++) {
                                    if (p < C) {
                                        const float w = wv[p];
                                        accs[ch + 0] = __fmaf_rn(w, va[p].x, accs[ch + 0]); accs[ch + 1] = __fmaf_rn(w, va[p].y, accs[ch + 1]);
                                        accs[ch + 2] = __fmaf_rn(w, va[p].z, accs[ch + 2]); accs[ch + 3] = __fmaf_rn(w, va[p].w, accs[ch + 3]);
                                        accs[ch + 4] = __fmaf_rn(w, vb[p].x, accs[ch + 4]); accs[ch + 5] = __fmaf_rn(w, vb[p].y, accs[ch + 5]);
                                        accs[ch + 6] = __fmaf_rn(w, vb[p].z, accs[ch + 6]); accs[ch + 7] = __fmaf_rn(w, vb[p].w, accs[ch + 7]);
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (wg_tid_c == 0) {
                        for (int p = 0; p < C; p++) dec107_remote_arrive(dec107_mapa((unsigned)(smem + 392), (unsigned)p));
                    }
                    if (wg_tid_c == 0) DEC107_STAMP2(1);
                    const int jrow = row >> 3, qh = hk * 8 + (row & 7);
                    __nv_bfloat16* op = (__nv_bfloat16*)O_ptr + ((size_t)(bidx * q_len + jrow) * num_q_heads + qh) * HEAD_DIM + d;
                    if (K == 1) {
                        if (row < live) {
                            const float inv = (L > 0.0f) ? approx_rcp(L) * out_scale : 0.0f;
                            _Pragma("unroll") for (int i = 0; i < 32; i += 2) if (i < per)
                                *reinterpret_cast<__nv_bfloat162*>(op + i) = __floats2bfloat162_rn(accs[i] * inv, accs[i + 1] * inv);
                        }
                    } else {
                        const size_t slot = ((size_t)(t * G + kc) * C + rank);
                        if (row < live) {
                            float* go = partial_o + (slot * 32 + row) * Ds + sub * per;
                            _Pragma("unroll") for (int i = 0; i < 32; i += 4) if (i < per)
                                *reinterpret_cast<float4*>(go + i) = make_float4(accs[i], accs[i + 1], accs[i + 2], accs[i + 3]);
                            if (sub == 0) { partial_stats[slot * 64 + row] = M; partial_stats[slot * 64 + 32 + row] = L; }
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        if (wg_tid_c == 0) {
                            asm volatile("fence.acq_rel.gpu;" ::: "memory");
                            unsigned old;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old)
                                         : "l"(&tile_counters[(t * C + rank) * 4]), "r"(1u) : "memory");
                            smem_corr_flag[0] = (old + 1 == (unsigned)K) ? 1u : 0u;
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        if (wg_tid_c == 0) DEC107_STAMP2(2);
                        if (smem_corr_flag[0] != 0u) {
                            asm volatile("fence.acquire.gpu;" ::: "memory");
                            if (row < live) {
                                float M2 = -1e+30f;
                                for (int k0 = 0; k0 < K; k0 += 8) {
                                    float mk[8];
                                    #pragma unroll
                                    for (int k = 0; k < 8; k++)
                                        mk[k] = (k0 + k < K) ? partial_stats[((size_t)(t * G + k0 + k) * C + rank) * 64 + row] : -1e+30f;
                                    #pragma unroll
                                    for (int k = 0; k < 8; k++) M2 = max_noftz(M2, mk[k]);
                                }
                                float L2 = 0.0f, a2[32];
                                #pragma unroll
                                for (int i = 0; i < 32; i++) a2[i] = 0.0f;
                                for (int k0 = 0; k0 < K; k0 += 8) {
                                    float wk[8], lk[8];
                                    #pragma unroll
                                    for (int k = 0; k < 8; k++) {
                                        const size_t sk = (size_t)(t * G + k0 + k) * C + rank;
                                        const bool ok = (k0 + k < K);
                                        const float m_k = ok ? partial_stats[sk * 64 + row] : -1e+30f;
                                        lk[k] = ok ? partial_stats[sk * 64 + 32 + row] : 0.0f;
                                        wk[k] = ok ? approx_exp2((m_k - M2) * softmax_scale_log2) : 0.0f;
                                    }
                                    #pragma unroll
                                    for (int k = 0; k < 8; k++) L2 = __fmaf_rn(wk[k], lk[k], L2);
                                    #pragma unroll
                                    for (int ch = 0; ch < 32; ch += 8) {
                                        if (ch < per) {
                                            float4 va[8], vb[8];
                                            #pragma unroll
                                            for (int k = 0; k < 8; k++) {
                                                if (k0 + k < K) {
                                                    const float* gk = partial_o + (((size_t)(t * G + k0 + k) * C + rank) * 32 + row) * Ds + sub * per + ch;
                                                    va[k] = *reinterpret_cast<const float4*>(gk);
                                                    vb[k] = *reinterpret_cast<const float4*>(gk + 4);
                                                } else { va[k] = make_float4(0.f, 0.f, 0.f, 0.f); vb[k] = va[k]; }
                                            }
                                            #pragma unroll
                                            for (int k = 0; k < 8; k++) {
                                                const float w = wk[k];
                                                a2[ch + 0] = __fmaf_rn(w, va[k].x, a2[ch + 0]); a2[ch + 1] = __fmaf_rn(w, va[k].y, a2[ch + 1]);
                                                a2[ch + 2] = __fmaf_rn(w, va[k].z, a2[ch + 2]); a2[ch + 3] = __fmaf_rn(w, va[k].w, a2[ch + 3]);
                                                a2[ch + 4] = __fmaf_rn(w, vb[k].x, a2[ch + 4]); a2[ch + 5] = __fmaf_rn(w, vb[k].y, a2[ch + 5]);
                                                a2[ch + 6] = __fmaf_rn(w, vb[k].z, a2[ch + 6]); a2[ch + 7] = __fmaf_rn(w, vb[k].w, a2[ch + 7]);
                                            }
                                        }
                                    }
                                }
                                const float inv = (L2 > 0.0f) ? approx_rcp(L2) * out_scale : 0.0f;
                                _Pragma("unroll") for (int i = 0; i < 32; i += 2) if (i < per)
                                    *reinterpret_cast<__nv_bfloat162*>(op + i) = __floats2bfloat162_rn(a2[i] * inv, a2[i + 1] * inv);
                            }
                            asm volatile("barrier.sync 9, 128;" ::: "memory");
                            if (wg_tid_c == 0) { tile_counters[(t * C + rank) * 4] = 0u; DEC107_STAMP2(3); }
                        }
                    }
                    if (wg_tid_c == 0) DEC107_STAMP(10);
                    dec107_wait_cluster((unsigned)(smem + 392), 0);   // peers finished reading my SMEM
                }
            }
#endif
            mbarrier_arrive(tmem_dealloc_addr);
        }
    // ---- Role: mma_warp ----
    } else if (warp == 8) {
        unsigned int mx_slot = 0, mx_sph = 0;
        mbarrier_wait(mx_sfa_ready_addr, 0);
        { // mma_warp_main
            if (lane == 0) {
            }
            unsigned int work_stage_m = 0;
            unsigned int q_cons_stage = 0;
            unsigned int q_cons_phase = 0;
            unsigned int k_cons_stage = 0;
            unsigned int k_cons_phase = 0;
            unsigned int v_cons_stage = 0;
            unsigned int v_cons_phase = 0;
            int s_buf = 0;
            unsigned int pf_stage = 0;
            unsigned int pf_phase = 0;
            unsigned int pv_idx_m = 0;
            unsigned int _phase_work_full_2 = 0;
            mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
            unsigned int base_2 = work_stage_m * 16;
            unsigned int valid_3 = work_token_words[base_2];
            unsigned int kind_3 = work_token_words[base_2 + 1];
            unsigned int batch_2 = work_token_words[base_2 + 2];
            unsigned int kv_head_2 = work_token_words[base_2 + 3];
            unsigned int block_begin_2 = work_token_words[base_2 + 4];
            unsigned int block_end_2 = work_token_words[base_2 + 5];
            unsigned int seqlen_2 = work_token_words[base_2 + 6];
            unsigned int n_chunks_2 = work_token_words[base_2 + 7];
            unsigned int slot_tile_base_2 = work_token_words[base_2 + 8];
            unsigned int counter_idx_2 = work_token_words[base_2 + 9];
            unsigned int chunk_2 = work_token_words[base_2 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
            work_stage_m += 1;
            if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
            unsigned int valid_m = valid_3;
            int kind_m = (int)kind_3;
            int block_begin_m = (int)block_begin_2;
            int block_end_m = (int)block_end_2;
            unsigned int _phase_o_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < max_items; _tile_iter_m++) {
                if (valid_m == 0) {
                    break;
                }
                if (kind_m == 0) {
                    int cnt_m = block_end_m - block_begin_m;
                    mbarrier_wait(q_full_addr + (q_cons_stage) * 8, q_cons_phase);
                    mbarrier_wait(k_full_addr + (k_cons_stage) * 8, k_cons_phase);
                    if (lane == 0) { DEC107_STAMP_ONCE(5); DEC107_COUNT(13, cnt_m); }
                    if (_tile_iter_m == 0) {
                        if (lane == 0) {
                        }
                    }
                    {
                        mbarrier_wait(mx_sfb_full_addr + mx_slot * 8, mx_sph);
                        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                        const uint32_t mx_qa = (((uint32_t)smem_qt_addr >> 4) & 0x3FFF) + (q_cons_stage) * 2048;
                        const uint32_t mx_kb = (((uint32_t)smem_k_addr >> 4) & 0x3FFF) + (k_cons_stage) * (MX_K_STAGE >> 4);
                        if (elect_sync()) {
                            #pragma unroll
                            for (int t = 0; t < 4; t++) {
                                const uint64_t ad = (static_cast<uint64_t>(0x40004040U) << 32) | (mx_qa + (uint32_t)(t >> 1) * 1024u + (uint32_t)(t & 1) * 4u);
                                const uint64_t bd = (static_cast<uint64_t>(0x40004040U) << 32) | (mx_kb + (uint32_t)t * 2u);
                                const uint32_t sid = (uint32_t)((2 * t) & 3);
                                const uint32_t cc = (uint32_t)(t >> 1) * 4u;
                                dec107_mma_mx((uint32_t)(tmem_tmem_s + (s_buf * DEC107_SBUF_STRIDE)), ad, bd, MX_IDESC | (sid << 4) | (sid << 29),
                                              (uint32_t)taddr + MX_SFA_COL + cc, (uint32_t)taddr + MX_SFB_COL + mx_slot * 8u + cc, t > 0 ? 1u : 0u);
                            }
                        }
                        __syncwarp();
                    }
                    elect_commit(s_full_addr + (s_buf) * 8);
                    elect_commit(k_empty_addr + (k_cons_stage) * 8);
                    elect_commit(mx_sfb_empty_addr + mx_slot * 8);
                    mx_slot ^= 1u; if (mx_slot == 0) mx_sph ^= 1u;
                    k_cons_stage += 1;
                    if (k_cons_stage == DEC107_K_STAGES) { k_cons_stage = 0; k_cons_phase ^= 1; }
                    s_buf = s_buf ^ 1;
                    if (cnt_m == 1) {
                        elect_commit(q_empty_addr + (q_cons_stage) * 8);
                    }
                    if (lane == 0) {
                    }
                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                    _phase_o_empty_0 ^= 1;
                    if (lane == 0) {
                    }
                    int first_pv = 1;
                    #pragma unroll 1
                    for (int n_2 = 0; n_2 < cnt_m; n_2++) {
                        int next_n = n_2 + 1;
                        if (lane == 0) {
                        }
#ifndef DEC107_SINGLE_S
                        if (next_n < cnt_m) {
                            if (lane == 0) {
                            }
                            mbarrier_wait(k_full_addr + (k_cons_stage) * 8, k_cons_phase);
                            if (lane == 0) {
                            }
                            {
                                mbarrier_wait(mx_sfb_full_addr + mx_slot * 8, mx_sph);
                                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                                const uint32_t mx_qa = (((uint32_t)smem_qt_addr >> 4) & 0x3FFF) + (q_cons_stage) * 2048;
                                const uint32_t mx_kb = (((uint32_t)smem_k_addr >> 4) & 0x3FFF) + (k_cons_stage) * (MX_K_STAGE >> 4);
                                if (elect_sync()) {
                                    #pragma unroll
                                    for (int t = 0; t < 4; t++) {
                                        const uint64_t ad = (static_cast<uint64_t>(0x40004040U) << 32) | (mx_qa + (uint32_t)(t >> 1) * 1024u + (uint32_t)(t & 1) * 4u);
                                        const uint64_t bd = (static_cast<uint64_t>(0x40004040U) << 32) | (mx_kb + (uint32_t)t * 2u);
                                        const uint32_t sid = (uint32_t)((2 * t) & 3);
                                        const uint32_t cc = (uint32_t)(t >> 1) * 4u;
                                        dec107_mma_mx((uint32_t)(tmem_tmem_s + (s_buf * DEC107_SBUF_STRIDE)), ad, bd, MX_IDESC | (sid << 4) | (sid << 29),
                                                      (uint32_t)taddr + MX_SFA_COL + cc, (uint32_t)taddr + MX_SFB_COL + mx_slot * 8u + cc, t > 0 ? 1u : 0u);
                                    }
                                }
                                __syncwarp();
                            }
                            elect_commit(s_full_addr + (s_buf) * 8);
                            elect_commit(k_empty_addr + (k_cons_stage) * 8);
                    elect_commit(mx_sfb_empty_addr + mx_slot * 8);
                    mx_slot ^= 1u; if (mx_slot == 0) mx_sph ^= 1u;
                            k_cons_stage += 1;
                            if (k_cons_stage == DEC107_K_STAGES) { k_cons_stage = 0; k_cons_phase ^= 1; }
                            s_buf = s_buf ^ 1;
                            if (next_n + 1 == cnt_m) {
                                elect_commit(q_empty_addr + (q_cons_stage) * 8);
                            }
                        }
#endif
                        if (lane == 0) {
                        }
                        mbarrier_wait(v_full_addr + (v_cons_stage) * 8, v_cons_phase);
                        if (lane == 0) {
                        }
                        mbarrier_wait(p_full_addr + (pf_stage) * 8, pf_phase);
                        if (lane == 0) {
                        }
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int first_pv_flag = first_pv;
                        int _mma_b_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_cons_stage) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 0x04410010;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_o), "r"(_mma_b_lo_2), "r"(tmem_tmem_s + (int)pf_stage * DEC107_SBUF_STRIDE), "r"(((first_pv_flag) ? 0 : 1)));
                        int o_st_m = (int)(pv_idx_m & 1);
                        elect_commit(v_empty_addr + (v_cons_stage) * 8);
                        elect_commit(o_ready_addr + (o_st_m) * 8);
#ifdef DEC107_SINGLE_S
                        if (next_n < cnt_m) {
                            if (lane == 0) {
                            }
                            mbarrier_wait(k_full_addr + (k_cons_stage) * 8, k_cons_phase);
                            if (lane == 0) {
                            }
                            {
                                mbarrier_wait(mx_sfb_full_addr + mx_slot * 8, mx_sph);
                                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                                const uint32_t mx_qa = (((uint32_t)smem_qt_addr >> 4) & 0x3FFF) + (q_cons_stage) * 2048;
                                const uint32_t mx_kb = (((uint32_t)smem_k_addr >> 4) & 0x3FFF) + (k_cons_stage) * (MX_K_STAGE >> 4);
                                if (elect_sync()) {
                                    #pragma unroll
                                    for (int t = 0; t < 4; t++) {
                                        const uint64_t ad = (static_cast<uint64_t>(0x40004040U) << 32) | (mx_qa + (uint32_t)(t >> 1) * 1024u + (uint32_t)(t & 1) * 4u);
                                        const uint64_t bd = (static_cast<uint64_t>(0x40004040U) << 32) | (mx_kb + (uint32_t)t * 2u);
                                        const uint32_t sid = (uint32_t)((2 * t) & 3);
                                        const uint32_t cc = (uint32_t)(t >> 1) * 4u;
                                        dec107_mma_mx((uint32_t)(tmem_tmem_s + (s_buf * DEC107_SBUF_STRIDE)), ad, bd, MX_IDESC | (sid << 4) | (sid << 29),
                                                      (uint32_t)taddr + MX_SFA_COL + cc, (uint32_t)taddr + MX_SFB_COL + mx_slot * 8u + cc, t > 0 ? 1u : 0u);
                                    }
                                }
                                __syncwarp();
                            }
                            elect_commit(s_full_addr + (s_buf) * 8);
                            elect_commit(k_empty_addr + (k_cons_stage) * 8);
                    elect_commit(mx_sfb_empty_addr + mx_slot * 8);
                    mx_slot ^= 1u; if (mx_slot == 0) mx_sph ^= 1u;
                            k_cons_stage += 1;
                            if (k_cons_stage == DEC107_K_STAGES) { k_cons_stage = 0; k_cons_phase ^= 1; }
                            s_buf = s_buf ^ 1;
                            if (next_n + 1 == cnt_m) {
                                elect_commit(q_empty_addr + (q_cons_stage) * 8);
                            }
                        }
#endif
                        pv_idx_m = pv_idx_m + 1;
                        v_cons_stage += 1;
                        if (v_cons_stage == DEC107_KV_STAGES) { v_cons_stage = 0; v_cons_phase ^= 1; }
                        pf_stage += 1;
                        if (pf_stage == 2) { pf_stage = 0; pf_phase ^= 1; }
                        first_pv = 0;
                        if (lane == 0) {
                        }
                    }
                    q_cons_phase ^= 1;
                    if (lane == 0) {
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
                unsigned int base_0_2 = work_stage_m * 16;
                unsigned int valid_1_2 = work_token_words[base_0_2];
                unsigned int kind_2_2 = work_token_words[base_0_2 + 1];
                unsigned int batch_3_2 = work_token_words[base_0_2 + 2];
                unsigned int kv_head_4_2 = work_token_words[base_0_2 + 3];
                unsigned int block_begin_5_2 = work_token_words[base_0_2 + 4];
                unsigned int block_end_6_2 = work_token_words[base_0_2 + 5];
                unsigned int seqlen_7_2 = work_token_words[base_0_2 + 6];
                unsigned int n_chunks_8_2 = work_token_words[base_0_2 + 7];
                unsigned int slot_tile_base_9_2 = work_token_words[base_0_2 + 8];
                unsigned int counter_idx_10_2 = work_token_words[base_0_2 + 9];
                unsigned int chunk_11_2 = work_token_words[base_0_2 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                work_stage_m += 1;
                if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
                valid_m = valid_1_2;
                kind_m = (int)kind_2_2;
                block_begin_m = (int)block_begin_5_2;
                block_end_m = (int)block_end_6_2;
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            if (lane == 0) {
            }
        }
    // ---- Role: load_pgoff ----
    } else if (warp == 9) {
        { // load_pgoff_main
            int pg_blk_p = lane >> 3;
            int pg_lane_p = lane & 7;
            unsigned int page_prod_stage = 0;
            unsigned int page_prod_phase = 1;
            unsigned int work_stage_p = 0;
            unsigned int _phase_work_full_3 = 0;
            mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
            unsigned int base_3 = work_stage_p * 16;
            unsigned int valid_4 = work_token_words[base_3];
            unsigned int kind_4 = work_token_words[base_3 + 1];
            unsigned int batch_4 = work_token_words[base_3 + 2];
            unsigned int kv_head_3 = work_token_words[base_3 + 3];
            unsigned int block_begin_3 = work_token_words[base_3 + 4];
            unsigned int block_end_3 = work_token_words[base_3 + 5];
            unsigned int seqlen_3 = work_token_words[base_3 + 6];
            unsigned int n_chunks_3 = work_token_words[base_3 + 7];
            unsigned int slot_tile_base_3 = work_token_words[base_3 + 8];
            unsigned int counter_idx_3 = work_token_words[base_3 + 9];
            unsigned int chunk_3 = work_token_words[base_3 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
            work_stage_p += 1;
            if (work_stage_p == 4) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
            unsigned int valid_p = valid_4;
            int kind_p = (int)kind_4;
            int batch_idx_p = (int)batch_4;
            int block_begin_p = (int)block_begin_3;
            int block_end_p = (int)block_end_3;
            int seqlen_kv_p = (int)seqlen_3;
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < max_items; _tile_iter_p++) {
                if (valid_p == 0) {
                    break;
                }
                if (kind_p == 0) {
                    int cta_n_blocks_p = block_end_p - block_begin_p;
                    int max_pg_p = (seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1;
                    int pt_base_p = batch_idx_p * max_pages_per_seq;
                    #pragma unroll 1
                    for (int ni0_p = 0; ni0_p < cta_n_blocks_p; ni0_p += 4) {
                        int g_cnt_p = cta_n_blocks_p - ni0_p;
                        if (g_cnt_p > 4) {
                            g_cnt_p = 4;
                        }
                        int n_block_p = block_begin_p + cta_n_blocks_p - 1 - (ni0_p + pg_blk_p);
                        int page_idx_p = n_block_p * 4 + pg_lane_p;
                        if (page_idx_p > max_pg_p) {
                            page_idx_p = max_pg_p;
                        }
                        int page_id_p = 0;
                        if (pg_blk_p < g_cnt_p) {
                            page_id_p = page_table[pt_base_p + page_idx_p];
                        }
                        #pragma unroll
                        for (int gc_p = 0; gc_p < 4; gc_p++) {
                            if (g_cnt_p > gc_p) {
                                int st_u_p = page_prod_stage + (unsigned int)gc_p;
                                int st_p = ((st_u_p >= 6) ? st_u_p - 6 : st_u_p);
                                int ph_p = ((st_u_p >= 6) ? page_prod_phase ^ 1 : page_prod_phase);
                                mbarrier_wait(page_offsets_empty_addr + (st_p) * 8, ph_p);
                                if (pg_blk_p == gc_p && pg_lane_p < 4) {
                                    smem_page_offsets[st_p * 8 + pg_lane_p] = page_id_p;
                                }
                                __syncwarp();
                                if (elect_sync()) {
                                    mbarrier_arrive(page_offsets_full_addr + (st_p) * 8);
                                }
                            }
                        }
                        #pragma unroll
                        for (int gv_p = 0; gv_p < 4; gv_p++) {
                            if (g_cnt_p > gv_p) {
                                page_prod_stage += 1;
                                if (page_prod_stage == 6) { page_prod_stage = 0; page_prod_phase ^= 1; }
                            }
                        }
                        if (lane == 0) {
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
                unsigned int base_0_3 = work_stage_p * 16;
                unsigned int valid_1_3 = work_token_words[base_0_3];
                unsigned int kind_2_3 = work_token_words[base_0_3 + 1];
                unsigned int batch_3_3 = work_token_words[base_0_3 + 2];
                unsigned int kv_head_4_3 = work_token_words[base_0_3 + 3];
                unsigned int block_begin_5_3 = work_token_words[base_0_3 + 4];
                unsigned int block_end_6_3 = work_token_words[base_0_3 + 5];
                unsigned int seqlen_7_3 = work_token_words[base_0_3 + 6];
                unsigned int n_chunks_8_3 = work_token_words[base_0_3 + 7];
                unsigned int slot_tile_base_9_3 = work_token_words[base_0_3 + 8];
                unsigned int counter_idx_10_3 = work_token_words[base_0_3 + 9];
                unsigned int chunk_11_3 = work_token_words[base_0_3 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
                work_stage_p += 1;
                if (work_stage_p == 4) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
                valid_p = valid_1_3;
                kind_p = (int)kind_2_3;
                batch_idx_p = (int)batch_3_3;
                block_begin_p = (int)block_begin_5_3;
                block_end_p = (int)block_end_6_3;
                seqlen_kv_p = (int)seqlen_7_3;
            }
            if (lane == 0) {
            }
        }
    // ---- Role: scheduler ----
    } else if (warp == 10) {
        { // scheduler_main
            int lane_0 = lane;
            int num_ctas = gridDim.x;
            if (lane_0 == 0) {
            }
            int items_per_chunk = num_kv_heads;
            int num_groups = (batch_size + 32 - 1) / 32;
            #pragma unroll 1
            for (int gs = 0; gs < num_groups; gs++) {
                int bs = gs * 32 + lane_0;
                if (bs < batch_size) {
                    sched_seq_lens[bs] = seq_lens_kv[bs];
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane_0 == 0) DEC107_STAMP(2);
#ifdef DEC107_LOCAL
            #pragma unroll 1
            for (int gs = 0; gs < num_groups; gs++) {
                int bs = gs * 32 + lane_0;
                unsigned int d1 = 0;
                if (bs < batch_size && pages_per_domain > 0) {
                    d1 = (page_table[(long long)bs * max_pages_per_seq] >= pages_per_domain) ? 1u : 0u;
                }
                unsigned int m1 = __ballot_sync(0xFFFFFFFFu, d1 != 0);
                if (lane_0 == 0) sched_dom[gs] = m1;
            }
            __syncwarp();
            int my_dom = 0;
            if (sm_domain != nullptr) {
                unsigned int smid_l;
                asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_l));
                my_dom = (int)sm_domain[smid_l & 255];
            }
            unsigned int cur_q = (my_dom < 0) ? (blockIdx.x & 1u) : (unsigned int)my_dom;
            unsigned int switched_q = 0;
#define DEC107_DOMQ(b) ((((sched_dom[(b) >> 5] >> ((b) & 31)) & 1u)) == cur_q)
#endif

#ifdef DEC107_LOWB
            {
                // Static low-B schedule. Clusters (C CTAs) -> tiles (b, h_k) in proportion to KV blocks; each CTA of a
                // cluster streams a contiguous block range of its tile. Info for the merge goes to token stage 3.
                const int C = (int)dec107_ncta();
                const int G = (int)gridDim.x / C;
                const int cid = (int)blockIdx.x / C;
                const int rank = (int)dec107_cta_rank();
                const int T = batch_size * num_kv_heads;
                int nb = 0;
                if (lane_0 < T) nb = (sched_seq_lens[lane_0 / num_kv_heads] + BLOCK_N - 1) / BLOCK_N;
                const int nz = (int)__reduce_add_sync(0xFFFFFFFFu, nb > 0 ? 1u : 0u);
                const int N = (int)__reduce_add_sync(0xFFFFFFFFu, (unsigned)nb);
                const int Gr = (G > nz) ? G - nz : 0;
                int Kt = 0;
                if (N > 0) {
                    const unsigned prod = (unsigned)Gr * (unsigned)nb;
                    const int base = (int)(prod / (unsigned)N);
                    const unsigned frac = prod - (unsigned)base * (unsigned)N;
                    const int left = Gr - (int)__reduce_add_sync(0xFFFFFFFFu, (unsigned)base);
                    int rk = 0;   // rank of this lane's remainder (desc, ties by lane)
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {   // all lanes: ranking over lanes 0-7 only over-assigned CTAs at T > 8 (lowb-dec)
                        const unsigned fj = __shfl_sync(0xFFFFFFFFu, frac, j);
                        const int nbj = __shfl_sync(0xFFFFFFFFu, nb, j);
                        if (nbj > 0 && j != lane_0 && (fj > frac || (fj == frac && j < lane_0))) rk++;
                    }
                    if (nb > 0) Kt = 1 + base + ((rk < left) ? 1 : 0);
                    int cap = nb / C; if (cap < 1) cap = 1;
                    if (Kt > cap) Kt = cap;
                }
                int incl = Kt;
                #pragma unroll
                for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xFFFFFFFFu, incl, o); if (lane_0 >= o) incl += v; }
                int excl = incl - Kt;
                // fail closed: the static schedule must fit the grid and the tile count must fit one warp
                if (T > 32 || __shfl_sync(0xFFFFFFFFu, incl, 31) > G) asm volatile("trap;");
                unsigned hit = __ballot_sync(0xFFFFFFFFu, Kt > 0 && cid >= excl && cid < incl);
                int t = hit ? __ffs(hit) - 1 : -1;
                int tK = __shfl_sync(0xFFFFFFFFu, Kt, t < 0 ? 0 : t);
                int tbase = __shfl_sync(0xFFFFFFFFu, excl, t < 0 ? 0 : t);
                int tnb = __shfl_sync(0xFFFFFFFFu, nb, t < 0 ? 0 : t);
                if (lane_0 == 0) {
                    int kc = cid - tbase, nsplit = tK * C, u = kc * C + rank;
                    int bb = (t < 0) ? 0 : (int)(((long long)tnb * u) / nsplit);
                    int be = (t < 0) ? 0 : (int)(((long long)tnb * (u + 1)) / nsplit);
                    int bidx = (t < 0) ? 0 : t / num_kv_heads;
                    work_token_words[48] = (unsigned)t; work_token_words[49] = (unsigned)kc;
                    work_token_words[50] = (unsigned)rank; work_token_words[51] = (unsigned)tK;
                    work_token_words[52] = (unsigned)G;
                    mbarrier_wait(work_empty_addr + 0 * 8, 1);
                    work_token_words[0] = (t >= 0 && be > bb) ? 1u : 0u;
                    work_token_words[1] = 0; work_token_words[2] = (unsigned)bidx;
                    work_token_words[3] = (unsigned)((t < 0) ? 0 : t % num_kv_heads);
                    work_token_words[4] = (unsigned)bb; work_token_words[5] = (unsigned)be;
                    work_token_words[6] = (unsigned)((t < 0) ? 0 : sched_seq_lens[bidx]);
                    work_token_words[7] = (unsigned)nsplit; work_token_words[8] = 0; work_token_words[9] = 0;
                    work_token_words[10] = (unsigned)u;
                    DEC107_STAMP_ONCE(3);
                    mbarrier_arrive(work_full_addr + 0 * 8);
                    mbarrier_wait(work_empty_addr + 1 * 8, 1);
                    work_token_words[16] = 0;
                    mbarrier_arrive(work_full_addr + 1 * 8);
                }
                __syncwarp();
            }
#else
            unsigned int total_pairs = 0;
            unsigned int p_max = 0;
            unsigned int p_min = 4294967295;
            #pragma unroll 1
            for (int g1 = 0; g1 < num_groups; g1++) {
                int b1 = g1 * 32 + lane_0;
                unsigned int pairs1 = 0;
                unsigned int pairs1_min = 4294967295;
                if (b1 < batch_size) {
                    int s1 = sched_seq_lens[b1];
                    pairs1 = (unsigned int)((s1 + 255) / 256);
                    pairs1_min = pairs1;
                }
                unsigned int _warp_redux_u32_0;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(pairs1));
                total_pairs += _warp_redux_u32_0;
                unsigned int _warp_redux_u32_1;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(pairs1));
                unsigned int _max_0 = ((p_max) > (_warp_redux_u32_1) ? (p_max) : (_warp_redux_u32_1));
                p_max = _max_0;
                unsigned int _warp_redux_u32_2;
                asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"(pairs1_min));
                unsigned int _min_0 = ((p_min) < (_warp_redux_u32_2) ? (p_min) : (_warp_redux_u32_2));
                p_min = _min_0;
            }
            unsigned int total_work = total_pairs * (unsigned int)items_per_chunk;
            float _rcp_0 = approx_rcp((float)num_ctas);
            float ctas_rcp = _rcp_0;
            float _rcp_1 = approx_rcp((float)items_per_chunk);
            float items_rcp = _rcp_1;
            unsigned int q = (unsigned int)((float)total_work * ctas_rcp);
            if (total_work < q * (unsigned int)num_ctas) {
                q = q - 1;
            }
            if (total_work >= (q + 1) * (unsigned int)num_ctas) {
                q = q + 1;
            }
            if (total_work > q * (unsigned int)num_ctas) {
                q = q + 1;
            }
            unsigned int ideal_pairs = q;
            unsigned int balance_k = (ideal_pairs + 64 - 1) / 64;
            if (balance_k < 1) {
                balance_k = 1;
            }
            if (balance_k > 8) {
                balance_k = 8;
            }
            unsigned int chunk_divisor = balance_k * (unsigned int)num_ctas;
            float _rcp_2 = approx_rcp((float)chunk_divisor);
            unsigned int q_1 = (unsigned int)((float)total_work * _rcp_2);
            if (total_work < q_1 * chunk_divisor) {
                q_1 = q_1 - 1;
            }
            if (total_work >= (q_1 + 1) * chunk_divisor) {
                q_1 = q_1 + 1;
            }
            if (total_work > q_1 * chunk_divisor) {
                q_1 = q_1 + 1;
            }
            unsigned int chunk_pairs_u = q_1;
            if (chunk_pairs_u < 2) {
                chunk_pairs_u = 2;
            }
            if (chunk_pairs_u < p_max) {
                unsigned int split_ok = 0;
                if (p_max > ideal_pairs + chunk_pairs_u) {
                    split_ok = 1;
                }
                if (chunk_pairs_u < 2 * (p_max - p_min)) {
                    split_ok = 1;
                }
                if (split_ok == 0) {
                    chunk_pairs_u = p_max;
                }
            }
            int whole_items = batch_size * items_per_chunk;
            if (whole_items <= num_ctas) {
                float _rcp_3 = approx_rcp((float)whole_items);
                unsigned int q_0 = (unsigned int)((float)(unsigned int)num_ctas * _rcp_3);
                if (q_0 * (unsigned int)whole_items > (unsigned int)num_ctas) {
                    q_0 = q_0 - 1;
                }
                if ((q_0 + 1) * (unsigned int)whole_items <= (unsigned int)num_ctas) {
                    q_0 = q_0 + 1;
                }
                int n_even = (int)q_0;
                if (n_even > 1) {
                    if (p_max >= 8 * (p_max - p_min)) {
                        float _rcp_4 = approx_rcp((float)n_even);
                        unsigned int q_2 = (unsigned int)((float)p_max * _rcp_4);
                        if (p_max < q_2 * (unsigned int)n_even) {
                            q_2 = q_2 - 1;
                        }
                        if (p_max >= (q_2 + 1) * (unsigned int)n_even) {
                            q_2 = q_2 + 1;
                        }
                        if (p_max > q_2 * (unsigned int)n_even) {
                            q_2 = q_2 + 1;
                        }
                        unsigned int l_even = q_2;
                        if (l_even < 2) {
                            l_even = 2;
                        }
                        if (l_even < p_max) {
                            chunk_pairs_u = l_even;
                        }
                    }
                }
            }
            unsigned int q_2_1 = (unsigned int)((float)total_work * ctas_rcp);
            if (total_work < q_2_1 * (unsigned int)num_ctas) {
                q_2_1 = q_2_1 - 1;
            }
            if (total_work >= (q_2_1 + 1) * (unsigned int)num_ctas) {
                q_2_1 = q_2_1 + 1;
            }
            if (total_work > q_2_1 * (unsigned int)num_ctas) {
                q_2_1 = q_2_1 + 1;
            }
            unsigned int l_one = q_2_1;
            unsigned int sm_floor = ((unsigned int)num_ctas * 55 + 99) / 100;
            float _rcp_5 = approx_rcp((float)(100 * num_ctas));
            float ctas100_rcp = _rcp_5;
            if (l_one < 2) {
                l_one = 2;
            }
            unsigned int fit_valid = 0;
            unsigned int l_fit = p_max;
            unsigned int whole_items_u = (unsigned int)(batch_size * items_per_chunk);
            unsigned int fit_search = 0;
            if (whole_items_u <= (unsigned int)num_ctas) {
                if (p_max < 8 * (p_max - p_min)) {
                    fit_search = 1;
                }
            }
            if (fit_search != 0) {
                unsigned int l_hi = p_max;
                if (whole_items_u < (unsigned int)num_ctas) {
                    unsigned int denom_f = (unsigned int)num_ctas - whole_items_u;
                    float _rcp_6 = approx_rcp((float)denom_f);
                    unsigned int q_0_1 = (unsigned int)((float)total_work * _rcp_6);
                    if (total_work < q_0_1 * denom_f) {
                        q_0_1 = q_0_1 - 1;
                    }
                    if (total_work >= (q_0_1 + 1) * denom_f) {
                        q_0_1 = q_0_1 + 1;
                    }
                    if (total_work > q_0_1 * denom_f) {
                        q_0_1 = q_0_1 + 1;
                    }
                    unsigned int l_hi_f = q_0_1;
                    if (l_hi_f < p_max) {
                        l_hi = l_hi_f;
                    }
                }
                if (l_hi < l_one) {
                    l_hi = l_one;
                }
                unsigned int step_f = (l_hi - l_one + 31 - 1) / 31;
                if (step_f < 1) {
                    step_f = 1;
                }
                unsigned int l_lane = l_one + (unsigned int)lane_0 * step_f;
                if (l_lane > l_hi) {
                    l_lane = l_hi;
                }
                float _rcp_7 = approx_rcp((float)l_lane);
                float lane_rcp = _rcp_7;
                unsigned int t_lane = 0;
                #pragma unroll 4
                for (int bf = 0; bf < batch_size; bf++) {
                    int sf = sched_seq_lens[bf];
                    unsigned int pairs_f = (unsigned int)((sf + 255) / 256);
                    unsigned int q_0_2 = (unsigned int)((float)pairs_f * lane_rcp);
                    if (pairs_f < q_0_2 * l_lane) {
                        q_0_2 = q_0_2 - 1;
                    }
                    if (pairs_f >= (q_0_2 + 1) * l_lane) {
                        q_0_2 = q_0_2 + 1;
                    }
                    if (pairs_f > q_0_2 * l_lane) {
                        q_0_2 = q_0_2 + 1;
                    }
                    t_lane += q_0_2;
                }
                t_lane = t_lane * (unsigned int)items_per_chunk;
                unsigned int ok_key = 32;
                if (t_lane <= (unsigned int)num_ctas) {
                    ok_key = (unsigned int)lane_0;
                }
                unsigned int _warp_redux_u32_3;
                asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_3) : "r"(ok_key));
                unsigned int first_ok = _warp_redux_u32_3;
                if (first_ok < 32) {
                    unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, l_lane, (int)first_ok);
                    l_fit = _shfl_0;
                    fit_valid = 1;
                }
            }
            int cand_idx = lane_0 & 15;
            int req_parity = lane_0 >> 4;
            unsigned int cand = chunk_pairs_u;
            unsigned int cand_valid = 0;
            if (cand_idx < 13) {
                cand_valid = 1;
            }
            if (cand_idx == 9) {
                cand = p_max;
            }
            if (cand_idx > 9) {
                if (cand_idx < 13) {
                    cand = l_one * (unsigned int)(cand_idx - 8);
                    if (cand >= p_max) {
                        cand_valid = 0;
                    }
                }
            }
            if (cand_idx < 8) {
                unsigned int div_c = (unsigned int)(cand_idx + 1) * (unsigned int)num_ctas;
                float _rcp_8 = approx_rcp((float)div_c);
                unsigned int q_0_3 = (unsigned int)((float)total_work * _rcp_8);
                if (total_work < q_0_3 * div_c) {
                    q_0_3 = q_0_3 - 1;
                }
                if (total_work >= (q_0_3 + 1) * div_c) {
                    q_0_3 = q_0_3 + 1;
                }
                if (total_work > q_0_3 * div_c) {
                    q_0_3 = q_0_3 + 1;
                }
                cand = q_0_3;
                if (cand < 2) {
                    cand = 2;
                }
            }
            if (cand_idx == 13) {
                cand = l_fit;
                cand_valid = fit_valid;
            }
            float _rcp_9 = approx_rcp((float)cand);
            float cand_rcp = _rcp_9;
            unsigned int tickets_c = 0;
            unsigned int nmax_c = 0;
            int half_batch = (batch_size + 1) / 2;
            #pragma unroll 4
            for (int hc = 0; hc < half_batch; hc++) {
                int bc = 2 * hc + req_parity;
                unsigned int pairs_c = 0;
                if (bc < batch_size) {
                    int sc = sched_seq_lens[bc];
                    pairs_c = (unsigned int)((sc + 255) / 256);
                }
                unsigned int q_0_4 = (unsigned int)((float)pairs_c * cand_rcp);
                if (pairs_c < q_0_4 * cand) {
                    q_0_4 = q_0_4 - 1;
                }
                if (pairs_c >= (q_0_4 + 1) * cand) {
                    q_0_4 = q_0_4 + 1;
                }
                if (pairs_c > q_0_4 * cand) {
                    q_0_4 = q_0_4 + 1;
                }
                unsigned int nb_c = q_0_4;
                tickets_c += nb_c;
                unsigned int _max_1 = ((nmax_c) > (nb_c) ? (nmax_c) : (nb_c));
                nmax_c = _max_1;
            }
            unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, tickets_c, 16);
            tickets_c += _shfl_xor_0;
            unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, nmax_c, 16);
            unsigned int _max_2 = ((nmax_c) > (_shfl_xor_1) ? (nmax_c) : (_shfl_xor_1));
            nmax_c = _max_2;
            tickets_c = tickets_c * (unsigned int)items_per_chunk;
            unsigned int q_3 = (unsigned int)((float)tickets_c * ctas_rcp);
            if (tickets_c < q_3 * (unsigned int)num_ctas) {
                q_3 = q_3 - 1;
            }
            if (tickets_c >= (q_3 + 1) * (unsigned int)num_ctas) {
                q_3 = q_3 + 1;
            }
            if (tickets_c > q_3 * (unsigned int)num_ctas) {
                q_3 = q_3 + 1;
            }
            unsigned int waves_c = q_3;
            unsigned int tail_c = 0;
            if (nmax_c == 2) {
                tail_c = 8;
            }
            if (nmax_c > 2) {
                unsigned int sh_c = 0;
                if (nmax_c > 4) {
                    sh_c = 1;
                }
                if (nmax_c > 8) {
                    sh_c = 2;
                }
                if (nmax_c > 16) {
                    sh_c = 3;
                }
                if (nmax_c > 32) {
                    sh_c = 4;
                }
                if (nmax_c > 48) {
                    sh_c = 5;
                }
                if (nmax_c > 96) {
                    sh_c = 6;
                }
                unsigned int one_c = 1;
                tail_c = 11 + (nmax_c + (one_c << sh_c) - 1 >> sh_c);
            }
            unsigned int a_last_c = tickets_c - (waves_c - 1) * (unsigned int)num_ctas;
            unsigned int _max_3 = ((a_last_c) > (sm_floor) ? (a_last_c) : (sm_floor));
            unsigned int eff_c = _max_3;
            unsigned int cost_c = 4 * (waves_c - 1) * (cand + 20) + tail_c;
            unsigned int q_4 = (unsigned int)((float)(4 * cand * eff_c) * ctas_rcp);
            if (q_4 * (unsigned int)num_ctas > 4 * cand * eff_c) {
                q_4 = q_4 - 1;
            }
            if ((q_4 + 1) * (unsigned int)num_ctas <= 4 * cand * eff_c) {
                q_4 = q_4 + 1;
            }
            cost_c += q_4 + 80;
            unsigned int q_5 = (unsigned int)((float)(4 * cand * 5 * a_last_c) * ctas100_rcp);
            if (q_5 * (100 * (unsigned int)num_ctas) > 4 * cand * 5 * a_last_c) {
                q_5 = q_5 - 1;
            }
            if ((q_5 + 1) * (100 * (unsigned int)num_ctas) <= 4 * cand * 5 * a_last_c) {
                q_5 = q_5 + 1;
            }
            cost_c += q_5;
            unsigned int cost_key = 4294967295;
            if (cand_valid == 1) {
                if (req_parity == 0) {
                    cost_key = cost_c * 32 + (unsigned int)lane_0;
                }
            }
            unsigned int _warp_redux_u32_4;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_4) : "r"(cost_key));
            unsigned int best_key = _warp_redux_u32_4;
            int best_lane = (int)(best_key & 31);
            unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, cand, best_lane);
            chunk_pairs_u = _shfl_1;
            int chunk_pairs = (int)chunk_pairs_u;
            float _rcp_10 = approx_rcp((float)chunk_pairs_u);
            float chunk_rcp = _rcp_10;
            unsigned int n_full_chunks = 0;
            unsigned int n_chunks_total = 0;
            unsigned int n_split_requests = 0;
            unsigned int nmax_split = 0;
            unsigned int rem_bucket_total[4];
            #pragma unroll
            for (int bi = 0; bi < 4; bi++) {
                rem_bucket_total[bi] = 0;
            }
#ifdef DEC107_LOCAL
            unsigned int loc_full1 = 0, loc_chunks1 = 0, loc_split1 = 0, loc_rem1[4] = {0u, 0u, 0u, 0u};
#endif
            #pragma unroll 1
            for (int g2 = 0; g2 < num_groups; g2++) {
                int b2 = g2 * 32 + lane_0;
                unsigned int full2 = 0;
                unsigned int n2 = 0;
                unsigned int pack2 = 0;
                unsigned int split2 = 0;
                unsigned int nsplit2 = 0;
                if (b2 < batch_size) {
                    int s2 = sched_seq_lens[b2];
                    unsigned int pairs2 = (unsigned int)((s2 + 255) / 256);
                    unsigned int q_0_5 = (unsigned int)((float)pairs2 * chunk_rcp);
                    if (pairs2 < q_0_5 * chunk_pairs_u) {
                        q_0_5 = q_0_5 - 1;
                    }
                    if (pairs2 >= (q_0_5 + 1) * chunk_pairs_u) {
                        q_0_5 = q_0_5 + 1;
                    }
                    if (pairs2 > q_0_5 * chunk_pairs_u) {
                        q_0_5 = q_0_5 + 1;
                    }
                    n2 = q_0_5;
                    full2 = n2;
                    if (pairs2 < n2 * chunk_pairs_u) {
                        full2 = n2 - 1;
                        unsigned int rem2 = pairs2 - full2 * chunk_pairs_u;
                        int rb2 = 1;
                        #pragma unroll
                        for (int _rb = 0; _rb < 2; _rb++) {
                            if (chunk_pairs_u > rem2 << (unsigned int)rb2) {
                                rb2 = rb2 + 1;
                            }
                        }
                        pack2 = (unsigned int)(1 << 8 * (rb2 - 1));
                    }
                    if (n2 > 2) {
                        split2 = 1;
                        nsplit2 = n2;
                    }
                }
                unsigned int _warp_redux_u32_5;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_5) : "r"(full2));
                n_full_chunks += _warp_redux_u32_5;
                unsigned int _warp_redux_u32_6;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_6) : "r"(n2));
                n_chunks_total += _warp_redux_u32_6;
                unsigned int _warp_redux_u32_7;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_7) : "r"(split2));
                n_split_requests += _warp_redux_u32_7;
                unsigned int _warp_redux_u32_8;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_8) : "r"(nsplit2));
                unsigned int _max_4 = ((nmax_split) > (_warp_redux_u32_8) ? (nmax_split) : (_warp_redux_u32_8));
                nmax_split = _max_4;
                unsigned int _warp_redux_u32_9;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_9) : "r"(pack2));
                unsigned int pack_group = _warp_redux_u32_9;
#ifdef DEC107_LOCAL
                {
                    const unsigned int dm = (b2 < batch_size) ? ((sched_dom[b2 >> 5] >> (b2 & 31)) & 1u) : 0u;
                    const unsigned int f1 = __reduce_add_sync(0xFFFFFFFFu, dm ? full2 : 0u);
                    const unsigned int n1 = __reduce_add_sync(0xFFFFFFFFu, dm ? n2 : 0u);
                    const unsigned int s1 = __reduce_add_sync(0xFFFFFFFFu, dm ? split2 : 0u);
                    const unsigned int p1 = __reduce_add_sync(0xFFFFFFFFu, dm ? pack2 : 0u);
                    loc_full1 += f1; loc_chunks1 += n1; loc_split1 += s1;
                    #pragma unroll
                    for (int bi_l = 1; bi_l < 4; bi_l++) loc_rem1[bi_l] += (p1 >> (unsigned int)(8 * (bi_l - 1))) & 255u;
                }
#endif
                #pragma unroll
                for (int bi_1 = 1; bi_1 < 4; bi_1++) {
                    rem_bucket_total[bi_1] = rem_bucket_total[bi_1] + (pack_group >> (unsigned int)(8 * (bi_1 - 1)) & 255);
                }
            }
            unsigned int bucket_end[4];
            bucket_end[0] = n_full_chunks * (unsigned int)items_per_chunk;
            #pragma unroll
            for (int bi_2 = 1; bi_2 < 4; bi_2++) {
                bucket_end[bi_2] = bucket_end[bi_2 - 1] + rem_bucket_total[bi_2] * (unsigned int)items_per_chunk;
            }
            unsigned int chunk_items = n_chunks_total * (unsigned int)items_per_chunk;
            unsigned int reduce_shift = 0;
            if (nmax_split > 4) {
                reduce_shift = 1;
            }
            if (nmax_split > 8) {
                reduce_shift = 2;
            }
            if (nmax_split > 16) {
                reduce_shift = 3;
            }
            if (nmax_split > 32) {
                reduce_shift = 4;
            }
            if (nmax_split > 48) {
                reduce_shift = 5;
            }
            if (nmax_split > 96) {
                reduce_shift = 6;
            }
            unsigned int reduce_items = n_split_requests * (unsigned int)items_per_chunk << reduce_shift;
            unsigned int total_items = chunk_items + reduce_items;
#ifdef DEC107_LOCAL
            if (lane_0 == 0) {
                unsigned int be1[4];
                be1[0] = loc_full1 * (unsigned int)items_per_chunk;
                #pragma unroll
                for (int bi_l = 1; bi_l < 4; bi_l++) be1[bi_l] = be1[bi_l - 1] + loc_rem1[bi_l] * (unsigned int)items_per_chunk;
                const unsigned int ch1 = loc_chunks1 * (unsigned int)items_per_chunk;
                const unsigned int red1 = loc_split1 * (unsigned int)items_per_chunk << reduce_shift;
                loc_plan[8 + 0] = ch1 + red1; loc_plan[8 + 1] = ch1;
                loc_plan[0] = total_items - (ch1 + red1); loc_plan[1] = chunk_items - ch1;
                #pragma unroll
                for (int bi_l = 0; bi_l < 4; bi_l++) { loc_plan[8 + 2 + bi_l] = be1[bi_l]; loc_plan[2 + bi_l] = bucket_end[bi_l] - be1[bi_l]; }
            }
            __syncwarp();
#endif
            if (blockIdx.x == 0) {
                if (lane_0 == 0) {
                    int plan_off = num_ctas * 2048;
                    *(reinterpret_cast<float*>(partial_stats + plan_off) + (0)) = (float)chunk_pairs_u;
                    *(reinterpret_cast<float*>(partial_stats + (plan_off + 1)) + (0)) = (float)total_items;
                }
            }
            if (lane_0 == 0) {
            }
            unsigned int work_stage_sched = 0;
            int cur_group_b[4];
            unsigned int before_b[4];
            unsigned int si_before_b[4];
            unsigned int st_before_b[4];
            #pragma unroll
            for (int bi_3 = 0; bi_3 < 4; bi_3++) {
                cur_group_b[bi_3] = 0;
                before_b[bi_3] = 0;
                si_before_b[bi_3] = 0;
                st_before_b[bi_3] = 0;
            }
            unsigned int first_claim = 1;
            unsigned int gate_phase = 0;
            unsigned int _phase_work_empty = 1;
            #pragma unroll 1
            for (unsigned int _claim = 0; _claim < max_items + 1; _claim++) {
                mbarrier_wait(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty);
#ifdef DEC107_LOCAL
                unsigned int ticket_lane0 = 0;
                if (first_claim == 0) {
                    mbarrier_wait_hint(claim_gate_addr, gate_phase, 1000);
                    gate_phase = gate_phase ^ 1;
                }
                {
                    unsigned int q_l = cur_q, sw_l = switched_q;
                    if (lane_0 == 0) {
                        unsigned int t_l = atomicAdd(&queue_counters[2 + q_l], 1);
                        if (t_l >= loc_plan[q_l * 8] && sw_l == 0) {
                            q_l ^= 1u; sw_l = 1;
                            t_l = atomicAdd(&queue_counters[2 + q_l], 1);
                        }
                        ticket_lane0 = t_l;
                    }
                    q_l = __shfl_sync(0xFFFFFFFFu, q_l, 0);
                    sw_l = __shfl_sync(0xFFFFFFFFu, sw_l, 0);
                    if (sw_l != switched_q) {   // queue switch: the per-bucket cursors belong to the old queue
                        #pragma unroll
                        for (int bi_l = 0; bi_l < 4; bi_l++) { cur_group_b[bi_l] = 0; before_b[bi_l] = 0; si_before_b[bi_l] = 0; st_before_b[bi_l] = 0; }
                    }
                    cur_q = q_l; switched_q = sw_l;
                    total_items = loc_plan[cur_q * 8 + 0];
                    chunk_items = loc_plan[cur_q * 8 + 1];
                    #pragma unroll
                    for (int bi_l = 0; bi_l < 4; bi_l++) bucket_end[bi_l] = loc_plan[cur_q * 8 + 2 + bi_l];
                }
#else
                unsigned int ticket_lane0 = blockIdx.x;
                if (first_claim == 0) {
                    if (total_items <= (unsigned int)num_ctas) {
                        ticket_lane0 = (unsigned int)num_ctas;
                    } else {
                        mbarrier_wait_hint(claim_gate_addr, gate_phase, 1000);
                        gate_phase = gate_phase ^ 1;
                        if (lane_0 == 0) {
                            unsigned int _atomic_old_0 = atomicAdd(&queue_counters[0], 1);
                            ticket_lane0 = _atomic_old_0 + (unsigned int)num_ctas;
                        }
                    }
                }
#endif
                first_claim = 0;
                unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, ticket_lane0, 0);
                unsigned int ticket = _shfl_2;
                unsigned int token_base = work_stage_sched * 16;
                unsigned int valid_tok = ((ticket < total_items) ? 1 : 0);
                int counter_idx_r = 0;
                if (valid_tok != 0) {
                    if (ticket >= chunk_items) {
                        unsigned int r_red = ticket - chunk_items;
                        unsigned int rt_idx = r_red >> reduce_shift;
                        unsigned int rq_slice = r_red - (rt_idx << reduce_shift);
                        unsigned int rt_before = 0;
                        unsigned int si_before_r = 0;
                        unsigned int st_before_r = 0;
                        int b_r = 0;
                        int n_r = 0;
                        unsigned int mine_r = 0;
                        unsigned int si_r = 0;
                        unsigned int st_r = 0;
                        unsigned int incl_r = 0;
                        unsigned int incl_si_r = 0;
                        unsigned int incl_st_r = 0;
                        #pragma unroll 1
                        for (int _adv_r = 0; _adv_r < 32; _adv_r++) {
                            b_r = _adv_r * 32 + lane_0;
                            n_r = 0;
                            mine_r = 0;
                            si_r = 0;
                            st_r = 0;
                            if (b_r < batch_size) {
                                int s_r = sched_seq_lens[b_r];
                                int pairs_r = (s_r + 255) / 256;
                                unsigned int q_0_6 = (unsigned int)((float)(unsigned int)pairs_r * chunk_rcp);
                                if (q_0_6 * chunk_pairs_u > (unsigned int)pairs_r) {
                                    q_0_6 = q_0_6 - 1;
                                }
                                if ((q_0_6 + 1) * chunk_pairs_u <= (unsigned int)pairs_r) {
                                    q_0_6 = q_0_6 + 1;
                                }
                                if (q_0_6 * chunk_pairs_u < (unsigned int)pairs_r) {
                                    q_0_6 = q_0_6 + 1;
                                }
                                n_r = (int)q_0_6;
                                if (n_r > 1) {
                                    si_r = (unsigned int)n_r;
                                    st_r = 1;
                                }
                                if (n_r > 2) {
                                    mine_r = (unsigned int)items_per_chunk;
                                }
                            }
#ifdef DEC107_LOCAL
                            if (b_r < batch_size && !DEC107_DOMQ(b_r)) mine_r = 0;
#endif
                            uint32_t _warp_scan_sum_u32_0 = mine_r;
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(1));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(2));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(4));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(8));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(16));
                            incl_r = _warp_scan_sum_u32_0;
                            uint32_t _warp_scan_sum_u32_1 = si_r;
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(1));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(2));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(4));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(8));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(16));
                            incl_si_r = _warp_scan_sum_u32_1;
                            uint32_t _warp_scan_sum_u32_2 = st_r;
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(1));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(2));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(4));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(8));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(16));
                            incl_st_r = _warp_scan_sum_u32_2;
                            unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, incl_r, 31);
                            unsigned int group_total_r = _shfl_3;
                            if (rt_idx < rt_before + group_total_r) {
                                break;
                            }
                            rt_before = rt_before + group_total_r;
                            unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, incl_si_r, 31);
                            si_before_r = si_before_r + _shfl_4;
                            unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, incl_st_r, 31);
                            st_before_r = st_before_r + _shfl_5;
                        }
                        unsigned int in_group_r = rt_idx - rt_before;
                        unsigned int excl_r = incl_r - mine_r;
                        unsigned int excl_si_r = incl_si_r - si_r;
                        unsigned int excl_st_r = incl_st_r - st_r;
                        int hit_r = 0;
                        if (mine_r > 0) {
                            if (excl_r <= in_group_r) {
                                if (in_group_r < incl_r) {
                                    hit_r = 1;
                                }
                            }
                        }
                        unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, hit_r != 0);
                        unsigned int hit_mask_r = _vote_0;
                        int _ffs_0 = __ffs(hit_mask_r);
                        int hit_lane_r = _ffs_0 - 1;
                        int _shfl_6 = __shfl_sync(0xFFFFFFFF, b_r, hit_lane_r);
                        int sel_batch_r = _shfl_6;
                        int _shfl_7 = __shfl_sync(0xFFFFFFFF, n_r, hit_lane_r);
                        int sel_n_r = _shfl_7;
                        unsigned int _shfl_8 = __shfl_sync(0xFFFFFFFF, excl_r, hit_lane_r);
                        int sel_excl_r = (int)_shfl_8;
                        unsigned int _shfl_9 = __shfl_sync(0xFFFFFFFF, excl_si_r, hit_lane_r);
                        int sel_excl_si_r = (int)_shfl_9;
                        unsigned int _shfl_10 = __shfl_sync(0xFFFFFFFF, excl_st_r, hit_lane_r);
                        int sel_excl_st_r = (int)_shfl_10;
                        int kv_head_r = (int)in_group_r - sel_excl_r;
                        int slot_tile_base_r = ((int)si_before_r + sel_excl_si_r) * items_per_chunk + kv_head_r;
                        counter_idx_r = ((int)st_before_r + sel_excl_st_r) * items_per_chunk + kv_head_r;
                        if (lane_0 == 0) {
                            unsigned int arrived_r = 0;
                            #pragma unroll 1
                            for (int _poll_r = 0; _poll_r < 1073741824; _poll_r++) {
                                unsigned int _atomic_old_2;
                                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                    : "=r"(_atomic_old_2) : "l"(&tile_counters[counter_idx_r * 4]), "r"(static_cast<uint32_t>(0)) : "memory");
                                arrived_r = _atomic_old_2;
                                if ((int)arrived_r == sel_n_r) {
                                    break;
                                }
                            }
                            work_token_words[token_base + 1] = 1;
                            work_token_words[token_base + 2] = (unsigned int)sel_batch_r;
                            work_token_words[token_base + 3] = (unsigned int)kv_head_r;
                            work_token_words[token_base + 7] = (unsigned int)sel_n_r;
                            work_token_words[token_base + 8] = (unsigned int)slot_tile_base_r;
                            work_token_words[token_base + 4] = rq_slice;
                            work_token_words[token_base + 5] = reduce_shift;
                            work_token_words[token_base + 6] = 0;
                            work_token_words[token_base + 9] = (unsigned int)counter_idx_r;
                            work_token_words[token_base + 10] = 0;
                        }
                    } else {
                        int bucket = 0;
                        unsigned int bucket_start = 0;
                        #pragma unroll
                        for (int bi_4 = 0; bi_4 < 4; bi_4++) {
                            if (ticket >= bucket_end[bi_4]) {
                                bucket = bi_4 + 1;
                                bucket_start = bucket_end[bi_4];
                            }
                        }
                        unsigned int local_items = ticket - bucket_start;
                        unsigned int q_0_7 = (unsigned int)((float)local_items * items_rcp);
                        if (local_items < q_0_7 * (unsigned int)items_per_chunk) {
                            q_0_7 = q_0_7 - 1;
                        }
                        if (local_items >= (q_0_7 + 1) * (unsigned int)items_per_chunk) {
                            q_0_7 = q_0_7 + 1;
                        }
                        unsigned int local_chunk = q_0_7;
                        int in_chunk = (int)(local_items - local_chunk * (unsigned int)items_per_chunk);
                        int cursor_group = 0;
                        unsigned int before = 0;
                        unsigned int si_before = 0;
                        unsigned int st_before = 0;
                        #pragma unroll
                        for (int bi_5 = 0; bi_5 < 4; bi_5++) {
                            if (bucket == bi_5) {
                                cursor_group = cur_group_b[bi_5];
                                before = before_b[bi_5];
                                si_before = si_before_b[bi_5];
                                st_before = st_before_b[bi_5];
                            }
                        }
                        int s3 = 0;
                        int pairs3 = 0;
                        int n3 = 0;
                        int fullc3 = 0;
                        unsigned int mine3 = 0;
                        unsigned int split_items3 = 0;
                        unsigned int split_tiles3 = 0;
                        unsigned int incl3 = 0;
                        unsigned int incl_si3 = 0;
                        unsigned int incl_st3 = 0;
                        int b3 = 0;
                        #pragma unroll 1
                        for (int _adv = 0; _adv < 32; _adv++) {
                            b3 = cursor_group * 32 + lane_0;
                            s3 = 0;
                            pairs3 = 0;
                            n3 = 0;
                            fullc3 = 0;
                            mine3 = 0;
                            split_items3 = 0;
                            split_tiles3 = 0;
                            if (b3 < batch_size) {
                                s3 = sched_seq_lens[b3];
                                pairs3 = (s3 + 255) / 256;
                                unsigned int q_6 = (unsigned int)((float)(unsigned int)pairs3 * chunk_rcp);
                                if (q_6 * chunk_pairs_u > (unsigned int)pairs3) {
                                    q_6 = q_6 - 1;
                                }
                                if ((q_6 + 1) * chunk_pairs_u <= (unsigned int)pairs3) {
                                    q_6 = q_6 + 1;
                                }
                                if (q_6 * chunk_pairs_u < (unsigned int)pairs3) {
                                    q_6 = q_6 + 1;
                                }
                                n3 = (int)q_6;
                                fullc3 = n3;
                                int rem3 = 0;
                                if (pairs3 < n3 * chunk_pairs) {
                                    fullc3 = n3 - 1;
                                    rem3 = pairs3 - fullc3 * chunk_pairs;
                                }
                                if (n3 > 1) {
                                    split_items3 = (unsigned int)n3;
                                    split_tiles3 = 1;
                                }
                                if (bucket == 0) {
                                    mine3 = (unsigned int)fullc3;
                                } else if (rem3 > 0) {
                                    int rb3 = 1;
                                    #pragma unroll
                                    for (int _rb3 = 0; _rb3 < 2; _rb3++) {
                                        if (chunk_pairs > rem3 << rb3) {
                                            rb3 = rb3 + 1;
                                        }
                                    }
                                    if (rb3 == bucket) {
                                        mine3 = 1;
                                    }
                                }
                            }
#ifdef DEC107_LOCAL
                            if (b3 < batch_size && !DEC107_DOMQ(b3)) mine3 = 0;
#endif
                            uint32_t _warp_scan_sum_u32_3 = mine3;
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(1));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(2));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(4));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(8));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(16));
                            incl3 = _warp_scan_sum_u32_3;
                            uint32_t _warp_scan_sum_u32_4 = split_items3;
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(1));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(2));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(4));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(8));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(16));
                            incl_si3 = _warp_scan_sum_u32_4;
                            uint32_t _warp_scan_sum_u32_5 = split_tiles3;
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(1));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(2));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(4));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(8));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(16));
                            incl_st3 = _warp_scan_sum_u32_5;
                            unsigned int _shfl_11 = __shfl_sync(0xFFFFFFFF, incl3, 31);
                            unsigned int group_total = _shfl_11;
                            if (local_chunk < before + group_total) {
                                break;
                            }
                            before = before + group_total;
                            unsigned int _shfl_12 = __shfl_sync(0xFFFFFFFF, incl_si3, 31);
                            si_before = si_before + _shfl_12;
                            unsigned int _shfl_13 = __shfl_sync(0xFFFFFFFF, incl_st3, 31);
                            st_before = st_before + _shfl_13;
                            cursor_group = cursor_group + 1;
                        }
                        #pragma unroll
                        for (int bi_6 = 0; bi_6 < 4; bi_6++) {
                            if (bucket == bi_6) {
                                cur_group_b[bi_6] = cursor_group;
                                before_b[bi_6] = before;
                                si_before_b[bi_6] = si_before;
                                st_before_b[bi_6] = st_before;
                            }
                        }
                        unsigned int in_group = local_chunk - before;
                        unsigned int excl3 = incl3 - mine3;
                        unsigned int excl_si3 = incl_si3 - split_items3;
                        unsigned int excl_st3 = incl_st3 - split_tiles3;
                        int hit3 = 0;
                        if (mine3 > 0) {
                            if (excl3 <= in_group) {
                                if (in_group < incl3) {
                                    hit3 = 1;
                                }
                            }
                        }
                        unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, hit3 != 0);
                        unsigned int hit_mask = _vote_1;
                        int _ffs_1 = __ffs(hit_mask);
                        int hit_lane = _ffs_1 - 1;
                        int _shfl_14 = __shfl_sync(0xFFFFFFFF, b3, hit_lane);
                        int sel_batch = _shfl_14;
                        int _shfl_15 = __shfl_sync(0xFFFFFFFF, s3, hit_lane);
                        int sel_seqlen = _shfl_15;
                        int _shfl_16 = __shfl_sync(0xFFFFFFFF, n3, hit_lane);
                        int sel_n = _shfl_16;
                        int _shfl_17 = __shfl_sync(0xFFFFFFFF, fullc3, hit_lane);
                        int sel_full = _shfl_17;
                        unsigned int _shfl_18 = __shfl_sync(0xFFFFFFFF, excl3, hit_lane);
                        int sel_excl = (int)_shfl_18;
                        unsigned int _shfl_19 = __shfl_sync(0xFFFFFFFF, excl_si3, hit_lane);
                        int sel_excl_si = (int)_shfl_19;
                        unsigned int _shfl_20 = __shfl_sync(0xFFFFFFFF, excl_st3, hit_lane);
                        int sel_excl_st = (int)_shfl_20;
                        int sel_chunk_off = (int)in_group - sel_excl;
                        int chunk_idx = sel_full;
                        if (bucket == 0) {
                            chunk_idx = sel_chunk_off;
                        }
                        int kv_head_sel = in_chunk;
                        int n_blocks_tile = (sel_seqlen + BLOCK_N - 1) / BLOCK_N;
                        int block_begin_4 = 2 * chunk_idx * chunk_pairs;
                        int block_end_4 = 2 * (chunk_idx + 1) * chunk_pairs;
                        if (chunk_idx + 1 == sel_n) {
                            block_end_4 = n_blocks_tile;
                        }
                        int slot_tile_base_4 = 0;
                        int counter_idx_4 = 0;
                        if (sel_n > 1) {
                            slot_tile_base_4 = ((int)si_before + sel_excl_si) * items_per_chunk + kv_head_sel;
                            counter_idx_4 = ((int)st_before + sel_excl_st) * items_per_chunk + kv_head_sel;
                        }
                        if (lane_0 == 0) {
                            work_token_words[token_base + 1] = 0;
                            work_token_words[token_base + 2] = (unsigned int)sel_batch;
                            work_token_words[token_base + 3] = (unsigned int)kv_head_sel;
                            work_token_words[token_base + 4] = (unsigned int)block_begin_4;
                            work_token_words[token_base + 5] = (unsigned int)block_end_4;
                            work_token_words[token_base + 6] = (unsigned int)sel_seqlen;
                            work_token_words[token_base + 7] = (unsigned int)sel_n;
                            work_token_words[token_base + 8] = (unsigned int)slot_tile_base_4;
                            work_token_words[token_base + 9] = (unsigned int)counter_idx_4;
                            work_token_words[token_base + 10] = (unsigned int)chunk_idx;
                        }
                    }
                }
                if (lane_0 == 0) {
                    work_token_words[token_base] = valid_tok;
                    DEC107_STAMP_ONCE(3);
                    mbarrier_arrive(work_full_addr + (work_stage_sched) * 8);
                    if (valid_tok != 0) {
                        if (ticket >= chunk_items) {
                            unsigned int one_r = 1;
                            unsigned int slices_r = one_r << reduce_shift;
                            unsigned int _atomic_old_3;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                : "=r"(_atomic_old_3) : "l"(&tile_counters[counter_idx_r * 4 + 1]), "r"(static_cast<uint32_t>(1)) : "memory");
                            unsigned int pops_old = _atomic_old_3;
                            if (pops_old + 1 == slices_r) {
                                *(reinterpret_cast<unsigned int*>(tile_counters + (counter_idx_r * 4)) + (0)) = 0;
                                *(reinterpret_cast<unsigned int*>(tile_counters + (counter_idx_r * 4 + 1)) + (0)) = 0;
                            }
                        }
                    }
                }
                work_stage_sched += 1;
                if (work_stage_sched == 4) { work_stage_sched = 0; _phase_work_empty ^= 1; }
                if (valid_tok == 0) {
                    break;
                }
            }
            unsigned int done_old = 0;
            if (lane_0 == 0) {
                uint32_t _atomic_inc_old_0;
                asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                    : "=r"(_atomic_inc_old_0) : "l"(&queue_counters[1]), "r"(static_cast<uint32_t>(num_ctas - 1)) : "memory");
                done_old = _atomic_inc_old_0;
            }
            unsigned int _shfl_21 = __shfl_sync(0xFFFFFFFF, done_old, 0);
            done_old = _shfl_21;
            if ((int)done_old == num_ctas - 1) {
                if (lane_0 == 0) {
                    *(reinterpret_cast<unsigned int*>(queue_counters) + (0)) = 0;
#ifdef DEC107_LOCAL
                    queue_counters[2] = 0; queue_counters[3] = 0;
#endif
                }
            }
#endif
        }
    // ---- Role: load_warp ----
    } else if (warp == 11) {
        { // load_warp_main
            if (lane == 0) {
            }
            unsigned int q_prod_stage = 0;
            unsigned int q_prod_phase = 1;
            unsigned int page_cons_stage = 0;
            unsigned int page_cons_phase = 0;
            unsigned int k_prod_stage = 0;
            unsigned int k_prod_phase = 1;
            unsigned int v_prod_stage = 0;
            unsigned int v_prod_phase = 1;
            unsigned int work_stage_l = 0;
            unsigned int _phase_work_full_4 = 0;
            mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_4);
            unsigned int base_4 = work_stage_l * 16;
            unsigned int valid_5 = work_token_words[base_4];
            unsigned int kind_5 = work_token_words[base_4 + 1];
            unsigned int batch_5 = work_token_words[base_4 + 2];
            unsigned int kv_head_5 = work_token_words[base_4 + 3];
            unsigned int block_begin_6 = work_token_words[base_4 + 4];
            unsigned int block_end_5 = work_token_words[base_4 + 5];
            unsigned int seqlen_4 = work_token_words[base_4 + 6];
            unsigned int n_chunks_4 = work_token_words[base_4 + 7];
            unsigned int slot_tile_base_5 = work_token_words[base_4 + 8];
            unsigned int counter_idx_5 = work_token_words[base_4 + 9];
            unsigned int chunk_4 = work_token_words[base_4 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
            work_stage_l += 1;
            if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
            unsigned int valid_l = valid_5;
            int kind_l = (int)kind_5;
            int batch_idx_l = (int)batch_5;
            int kv_head_idx = (int)kv_head_5;
            int block_begin_l = (int)block_begin_6;
            int block_end_l = (int)block_end_5;
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < max_items; _tile_iter_l++) {
                if (valid_l == 0) {
                    break;
                }
                if (kind_l != 0) {
                    if (lane == 0) {
                        mbarrier_arrive(claim_gate_addr);
                    }
                }
                if (kind_l == 0) {
                    int cta_n_blocks = block_end_l - block_begin_l;
                    int n_pre = ((cta_n_blocks < DEC107_K_STAGES) ? cta_n_blocks : DEC107_K_STAGES);
                    int gate_block = cta_n_blocks - DEC107_KV_STAGES;
                    if (gate_block < 0) {
                        gate_block = 0;
                    }
                    if (elect_sync()) {
                        #pragma unroll 1
                        for (int ni = 0; ni < n_pre; ni++) {
                            int pre_stage_u = page_cons_stage + (unsigned int)ni;
                            int pre_stage = ((pre_stage_u >= 6) ? pre_stage_u - 6 : pre_stage_u);
                            int pre_phase = ((pre_stage_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                            int pre_pg_base = pre_stage * 8;
                            mbarrier_wait(page_offsets_full_addr + (pre_stage) * 8, pre_phase);
                            int pg_pre[8];
                            #pragma unroll
                            for (int pg_i = 0; pg_i < 4; pg_i++) {
                                pg_pre[pg_i] = smem_page_offsets[pre_pg_base + pg_i];
                            }
                            mbarrier_wait(DEC107_LSEL(lk_empty_addr, k_empty_addr) + (k_prod_stage) * 8, k_prod_phase);
                            DEC107_STAMP_ONCE(4);
                            DEC107_TX(k_full_addr + (k_prod_stage) * 8, MX_K_BYTES);
                            int kdst0 = smem_k_addr + k_prod_stage * MX_K_STAGE;
                            #pragma unroll
                            for (int pg_i_1 = 0; pg_i_1 < 4; pg_i_1++) {
                                int kpg0 = pg_pre[pg_i_1];
                                #pragma unroll
                                for (int hg = 0; hg < 2; hg++) {
                                    int ktoff0 = hg * 16384 + pg_i_1 * 4096;
                                    
#ifdef DEC107_LDG
if (hg == 0) dec107_ldg_pg[(0 * 3 + k_prod_stage) * 4 + pg_i_1] = kpg0;
if (hg == 1 && pg_i_1 == 3) mbarrier_arrive(lk_full_addr + (k_prod_stage) * 8);
#else
if (hg == 0) { tma_4d_gmem2smem(kdst0 + pg_i_1 * 4096, K, 0, 0, kv_head_idx, kpg0, k_full_addr + (k_prod_stage) * 8); tma_3d_gmem2smem(kdst0 + 16384 + pg_i_1 * 256, KS, 0, kv_head_idx, kpg0, k_full_addr + (k_prod_stage) * 8); }
#endif

                                }
                            }
                            k_prod_stage += 1;
                            if (k_prod_stage == DEC107_K_STAGES) { k_prod_stage = 0; k_prod_phase ^= 1; }
                            if (ni == 0) {
                                mbarrier_wait(q_empty_addr + (q_prod_stage) * 8, q_prod_phase);
                                if (_tile_iter_l == 0) {
                                }
                                mbarrier_arrive_expect_tx(q_full_addr + (q_prod_stage) * 8, 2 * 64 * HEAD_DIM);
                                #pragma unroll
                                for (int qw = 0; qw < 8; qw++) {   // 16 logical rows (2 q positions x 8 heads), twice, per K half
                                    #pragma unroll
                                    for (int qh = 0; qh < 2; qh++)
                                        tma_4d_gmem2smem(smem_qt_addr + q_prod_stage * 32768 + qh * 16384 + qw * 2048, Q, 0, kv_head_idx * 8,
                                                         batch_idx_l * q_len + 2 * (qw >> 1), qh, q_full_addr + (q_prod_stage) * 8);
                                }
                            }
                            if (ni < DEC107_KV_STAGES) {
                                mbarrier_wait(DEC107_LSEL(lv_empty_addr, v_empty_addr) + (v_prod_stage) * 8, v_prod_phase);
                                DEC107_TX(v_full_addr + (v_prod_stage) * 8, 32768);
                                int vdst0 = smem_v_addr + v_prod_stage * 32768;
                                #pragma unroll
                                for (int pg_i_2 = 0; pg_i_2 < 4; pg_i_2++) {
                                    int vpg0 = pg_pre[pg_i_2];
                                    #pragma unroll
                                    for (int hg_1 = 0; hg_1 < 2; hg_1++) {
                                        int vtoff0 = hg_1 * 16384 + pg_i_2 * 4096;
                                        
#ifdef DEC107_LDG
if (hg_1 == 0) dec107_ldg_pg[(1 * 3 + v_prod_stage) * 4 + pg_i_2] = vpg0;
if (hg_1 == 1 && pg_i_2 == 3) mbarrier_arrive(lv_full_addr + (v_prod_stage) * 8);
#else
tma_5d_gmem2smem(vdst0 + vtoff0, V, 0, 0, hg_1, kv_head_idx, vpg0, v_full_addr + (v_prod_stage) * 8);
#endif

                                    }
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == DEC107_KV_STAGES) { v_prod_stage = 0; v_prod_phase ^= 1; }
                            }
                        }
                        #pragma unroll 1
                        for (int ni_1 = 0; ni_1 < cta_n_blocks; ni_1++) {
                            int nk = ni_1 + DEC107_K_STAGES;
                            if (nk < cta_n_blocks) {
                                int k_page_u = page_cons_stage + DEC107_K_STAGES;
                                int k_page = ((k_page_u >= 6) ? k_page_u - 6 : k_page_u);
                                int k_page_phase = ((k_page_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                                int kpg_base = k_page * 8;
                                mbarrier_wait(page_offsets_full_addr + (k_page) * 8, k_page_phase);
                                int pg_nk[8];
                                #pragma unroll
                                for (int pg_i_3 = 0; pg_i_3 < 4; pg_i_3++) {
                                    pg_nk[pg_i_3] = smem_page_offsets[kpg_base + pg_i_3];
                                }
                                mbarrier_wait(DEC107_LSEL(lk_empty_addr, k_empty_addr) + (k_prod_stage) * 8, k_prod_phase);
                                DEC107_TX(k_full_addr + (k_prod_stage) * 8, MX_K_BYTES);
                                int kdst = smem_k_addr + k_prod_stage * MX_K_STAGE;
                                #pragma unroll
                                for (int pg_i_4 = 0; pg_i_4 < 4; pg_i_4++) {
                                    int npg0 = pg_nk[pg_i_4];
                                    #pragma unroll
                                    for (int hg_2 = 0; hg_2 < 2; hg_2++) {
                                        int ntoff = hg_2 * 16384 + pg_i_4 * 4096;
                                        
#ifdef DEC107_LDG
if (hg_2 == 0) dec107_ldg_pg[(0 * 3 + k_prod_stage) * 4 + pg_i_4] = npg0;
if (hg_2 == 1 && pg_i_4 == 3) mbarrier_arrive(lk_full_addr + (k_prod_stage) * 8);
#else
if (hg_2 == 0) { tma_4d_gmem2smem(kdst + pg_i_4 * 4096, K, 0, 0, kv_head_idx, npg0, k_full_addr + (k_prod_stage) * 8); tma_3d_gmem2smem(kdst + 16384 + pg_i_4 * 256, KS, 0, kv_head_idx, npg0, k_full_addr + (k_prod_stage) * 8); }
#endif

                                    }
                                }
                                k_prod_stage += 1;
                                if (k_prod_stage == DEC107_K_STAGES) { k_prod_stage = 0; k_prod_phase ^= 1; }
                            }
                            int nv = ni_1 + DEC107_KV_STAGES;
                            if (nv < cta_n_blocks) {
                                int v_page_u = page_cons_stage + DEC107_KV_STAGES;
                                int v_page = ((v_page_u >= 6) ? v_page_u - 6 : v_page_u);
                                int v_page_phase = ((v_page_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                                int vpg_base = v_page * 8;
                                mbarrier_wait(page_offsets_full_addr + (v_page) * 8, v_page_phase);
                                int pg_nv[8];
                                #pragma unroll
                                for (int pg_i_5 = 0; pg_i_5 < 4; pg_i_5++) {
                                    pg_nv[pg_i_5] = smem_page_offsets[vpg_base + pg_i_5];
                                }
                                mbarrier_wait(DEC107_LSEL(lv_empty_addr, v_empty_addr) + (v_prod_stage) * 8, v_prod_phase);
                                DEC107_TX(v_full_addr + (v_prod_stage) * 8, 32768);
                                int vdst = smem_v_addr + v_prod_stage * 32768;
                                #pragma unroll
                                for (int pg_i_6 = 0; pg_i_6 < 4; pg_i_6++) {
                                    int vpg1 = pg_nv[pg_i_6];
                                    #pragma unroll
                                    for (int hg_3 = 0; hg_3 < 2; hg_3++) {
                                        int vtoff = hg_3 * 16384 + pg_i_6 * 4096;
                                        
#ifdef DEC107_LDG
if (hg_3 == 0) dec107_ldg_pg[(1 * 3 + v_prod_stage) * 4 + pg_i_6] = vpg1;
if (hg_3 == 1 && pg_i_6 == 3) mbarrier_arrive(lv_full_addr + (v_prod_stage) * 8);
#else
tma_5d_gmem2smem(vdst + vtoff, V, 0, 0, hg_3, kv_head_idx, vpg1, v_full_addr + (v_prod_stage) * 8);
#endif

                                    }
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == DEC107_KV_STAGES) { v_prod_stage = 0; v_prod_phase ^= 1; }
                            }
                            mbarrier_arrive(page_offsets_empty_addr + (page_cons_stage) * 8);
                            page_cons_stage += 1;
                            if (page_cons_stage == 6) { page_cons_stage = 0; page_cons_phase ^= 1; }
                            if (ni_1 == gate_block) {
                                mbarrier_arrive(claim_gate_addr);
                            }
                        }
                    }
                    q_prod_phase ^= 1;
                    if (lane == 0) {
                        if (_tile_iter_l == 0) {
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_4);
                unsigned int base_0_4 = work_stage_l * 16;
                unsigned int valid_1_4 = work_token_words[base_0_4];
                unsigned int kind_2_4 = work_token_words[base_0_4 + 1];
                unsigned int batch_3_4 = work_token_words[base_0_4 + 2];
                unsigned int kv_head_4_4 = work_token_words[base_0_4 + 3];
                unsigned int block_begin_5_4 = work_token_words[base_0_4 + 4];
                unsigned int block_end_6_4 = work_token_words[base_0_4 + 5];
                unsigned int seqlen_7_4 = work_token_words[base_0_4 + 6];
                unsigned int n_chunks_8_4 = work_token_words[base_0_4 + 7];
                unsigned int slot_tile_base_9_4 = work_token_words[base_0_4 + 8];
                unsigned int counter_idx_10_4 = work_token_words[base_0_4 + 9];
                unsigned int chunk_11_4 = work_token_words[base_0_4 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                work_stage_l += 1;
                if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
                valid_l = valid_1_4;
                kind_l = (int)kind_2_4;
                batch_idx_l = (int)batch_3_4;
                kv_head_idx = (int)kv_head_4_4;
                block_begin_l = (int)block_begin_5_4;
                block_end_l = (int)block_end_6_4;
            }
            if (lane == 0) {
            }
        }
    // ---- Role: ldg (DEC107_LDG) / MX scale-factor warpgroup ----
    } else if (warp >= 12) {
#ifdef DEC107_MX
        {
            const uint32_t lq = (uint32_t)((warp - 12) * 32) << 16;
            {
                const uint32_t one = 0x7F7F7F7Fu;    // UE8M0 1.0 for every Q row / K-block
                asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %1, %1, %1, %1, %1, %1, %1};"
                             :: "r"((uint32_t)taddr + MX_SFA_COL + lq), "r"(one) : "memory");
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                mbarrier_arrive(mx_sfa_ready_addr);
            }
            unsigned int ks = 0, kph = 0, slot = 0, sph = 1, wst = 0, wph = 0;
            #pragma unroll 1
            for (unsigned int _it = 0; _it < max_items; _it++) {
                mbarrier_wait(work_full_addr + wst * 8, wph);
                const unsigned int wb = wst * 16;
                const unsigned int valid = work_token_words[wb], kind = work_token_words[wb + 1];
                const int bb = (int)work_token_words[wb + 4], be = (int)work_token_words[wb + 5];
                mbarrier_arrive(work_empty_addr + wst * 8);
                wst += 1; if (wst == 4) { wst = 0; wph ^= 1; }
                if (valid == 0) break;
                if (kind != 0) continue;
                #pragma unroll 1
                for (int n = 0; n < be - bb; n++) {
                    mbarrier_wait(k_full_addr + ks * 8, kph);
                    mbarrier_wait(mx_sfb_empty_addr + slot * 8, sph);
                    const unsigned char* sc = reinterpret_cast<const unsigned char*>(smem_raw) + MX_K_OFF + ks * MX_K_STAGE + 16384 + lane * 8;
                    uint32_t w[8];
                    #pragma unroll
                    for (int g = 0; g < 2; g++) {
                        #pragma unroll
                        for (int pg = 0; pg < 4; pg++) w[g * 4 + pg] = *reinterpret_cast<const uint32_t*>(sc + pg * 256 + g * 4);
                    }
                    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                 :: "r"((uint32_t)taddr + MX_SFB_COL + slot * 8u + lq), "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]),
                                    "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7]) : "memory");
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                    mbarrier_arrive(mx_sfb_full_addr + slot * 8);
                    ks += 1; if (ks == DEC107_K_STAGES) { ks = 0; kph ^= 1; }
                    slot ^= 1u; if (slot == 0) sph ^= 1u;
                }
            }
        }
#endif
#ifdef DEC107_LDG
#ifndef DEC107_LDG_DEPTH
#define DEC107_LDG_DEPTH (2 * DEC107_KV_STAGES - 2)   // K refills its stage 5 groups later in K0,K1,V0,K2,V1,K3 order
#endif
        // Issue order = the MMA warp's consumption order: K0, then per block n: K(n+1), V(n). Completed groups are
        // signalled (fence.proxy.async + arrive) in issue order, at most DEPTH groups behind; with consumer order ==
        // issue order the stage being refilled (>= 2*S-1 groups back) is already signalled when DEPTH <= 2*S-2.
        const int tt = tid - 384;                       // token row of the 128-token tile
        const int pg_i = tt >> 5, tok = tt & 31;
        const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
        unsigned int lks = 0, lkp = 0, lvs = 0, lvp = 0;   // page-id rings (consumer)
        unsigned int ks = 0, kp = 1, vs = 0, vp = 1;       // FP8 K / V rings (producer)
        unsigned int aks = 0, avs = 0;                     // next K / V stage to signal
        unsigned int npend = 0, pend = 0;                  // pending groups (FIFO of sides, bit 0 = oldest)
        unsigned int wst = 0, wph = 0;
        #pragma unroll 1
        for (unsigned int _it = 0; _it < max_items; _it++) {
            mbarrier_wait(work_full_addr + wst * 8, wph);
            const unsigned int wb = wst * 16;
            const unsigned int valid = work_token_words[wb], kind = work_token_words[wb + 1];
            const int kvh = (int)work_token_words[wb + 3];
            const int bb = (int)work_token_words[wb + 4], be = (int)work_token_words[wb + 5];
            mbarrier_arrive(work_empty_addr + wst * 8);
            wst += 1; if (wst == 4) { wst = 0; wph ^= 1; }
            if (valid == 0) break;
            if (kind != 0) continue;
            const uint8_t* hbase = kv_base + (long long)kvh * kv_head_stride + (long long)tok * kv_tok_stride;
            const int cnt = be - bb;
            auto issue = [&](int side) {
                const int lfull = side ? lv_full_addr : lk_full_addr, lempty = side ? lv_empty_addr : lk_empty_addr;
                const unsigned int ls = side ? lvs : lks, lph = side ? lvp : lkp;
                const unsigned int st8 = side ? vs : ks, ph8 = side ? vp : kp;
                mbarrier_wait(lfull + ls * 8, lph);
                const int page = dec107_ldg_pg[(side * 3 + ls) * 4 + pg_i];
                mbarrier_arrive(lempty + ls * 8);
                mbarrier_wait((side ? v_empty_addr : k_empty_addr) + st8 * 8, ph8);
                const uint8_t* src = hbase + (long long)page * kv_page_stride + (side ? kv_v_off : 0);
                const uint32_t dst = sbase + (side ? 123904u : 25600u) + st8 * 32768u + (uint32_t)tt * 128u;
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    const uint32_t d = dst + (uint32_t)(c >> 3) * 16384u + ((uint32_t)((c & 7) ^ (tt & 7)) << 4);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src + c * 16) : "memory");
                }
                asm volatile("cp.async.commit_group;" ::: "memory");
                if (side) { lvs += 1; if (lvs == DEC107_KV_STAGES) { lvs = 0; lvp ^= 1; } vs += 1; if (vs == DEC107_KV_STAGES) { vs = 0; vp ^= 1; } }
                else      { lks += 1; if (lks == DEC107_KV_STAGES) { lks = 0; lkp ^= 1; } ks += 1; if (ks == DEC107_KV_STAGES) { ks = 0; kp ^= 1; } }
                pend |= (unsigned int)side << npend; npend += 1;
                if (npend > DEC107_LDG_DEPTH) {
                    asm volatile("cp.async.wait_group %0;" :: "n"(DEC107_LDG_DEPTH) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    const unsigned int sd = pend & 1u; pend >>= 1; npend -= 1;
                    if (sd) { mbarrier_arrive(v_full_addr + avs * 8); avs += 1; if (avs == DEC107_KV_STAGES) avs = 0; }
                    else    { mbarrier_arrive(k_full_addr + aks * 8); aks += 1; if (aks == DEC107_KV_STAGES) aks = 0; }
                }
            };
            if (cnt > 0) issue(0);
            #pragma unroll 1
            for (int m = 0; m < cnt; m++) {
                if (m + 1 < cnt) issue(0);
                issue(1);
            }
            // drain: the consumers of this item must see every block before the next (possibly reduce) item
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            #pragma unroll 1
            while (npend > 0) {
                const unsigned int sd = pend & 1u; pend >>= 1; npend -= 1;
                if (sd) { mbarrier_arrive(v_full_addr + avs * 8); avs += 1; if (avs == DEC107_KV_STAGES) avs = 0; }
                else    { mbarrier_arrive(k_full_addr + aks * 8); aks += 1; if (aks == DEC107_KV_STAGES) aks = 0; }
            }
        }
#endif
    }

    // Cleanup
    if (tid == 0) DEC107_STAMP(11);
    asm volatile("griddepcontrol.launch_dependents;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(576));
    }
}

} // extern "C"
