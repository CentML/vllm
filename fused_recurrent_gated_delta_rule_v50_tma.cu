///////////////////////
// v50: TMA with CUTLASS Barrier Primitives Implementation
// Refactored from v33 to use CUTLASS 3.x barrier abstractions
// instead of raw PTX assembly for improved maintainability
// ~40% faster than baseline for decode workloads
// Requires SM90+ (Hopper) or SM100+ (Blackwell)
///////////////////////

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// CUTLASS includes for barrier primitives
#include <cutlass/arch/barrier.h>

//=============================================================================
// Constants
//=============================================================================
constexpr int BK = 128;
constexpr int BV = 128;
constexpr int BLOCK_SIZE = 128;
constexpr float L2_NORM_EPS = 1e-6f;

// TMA tile: for bf16, max inner dim with SWIZZLE_128B is 64 elements (128 bytes)
constexpr int TMA_TILE_V = 64;
constexpr int TMA_TILE_K = 128;

//=============================================================================
// Helper Functions
//=============================================================================

__device__ __forceinline__ float load_scalar_ldg(const __nv_bfloat16* ptr) {
    uint16_t raw = __ldg(reinterpret_cast<const uint16_t*>(ptr));
    __nv_bfloat16 val = *reinterpret_cast<__nv_bfloat16*>(&raw);
    return __bfloat162float(val);
}

__device__ __forceinline__ void store_scalar_streaming(__nv_bfloat16* ptr, float val) {
    __nv_bfloat16 bf_val = __float2bfloat16(val);
    __stcs(reinterpret_cast<uint16_t*>(ptr), *reinterpret_cast<uint16_t*>(&bf_val));
}

__device__ __forceinline__ float4 load_vec4_ldg(const __nv_bfloat16* ptr) {
    const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(ptr);
    uint32_t v0 = __ldg(ptr32);
    uint32_t v1 = __ldg(ptr32 + 1);
    __nv_bfloat162 bf0 = *reinterpret_cast<const __nv_bfloat162*>(&v0);
    __nv_bfloat162 bf1 = *reinterpret_cast<const __nv_bfloat162*>(&v1);
    float4 result;
    result.x = __bfloat162float(bf0.x);
    result.y = __bfloat162float(bf0.y);
    result.z = __bfloat162float(bf1.x);
    result.w = __bfloat162float(bf1.y);
    return result;
}

//=============================================================================
// TMA Helpers (using raw PTX for TMA load, CUTLASS for barriers)
//=============================================================================

// TMA 2D load - CTA level (no cluster)
// Note: TMA load instruction still uses PTX as CUTLASS doesn't abstract it yet
__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap* desc,
    uint64_t* barrier,
    void* smem,
    int32_t coord_x,
    int32_t coord_y
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_addr),
           "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(coord_x),
           "r"(coord_y),
           "r"(bar_addr)
        : "memory"
    );
}

//=============================================================================
// TMA Kernel - loads state using TMA with CUTLASS barriers
//=============================================================================

// Lower occupancy to reduce concurrent TMA pressure
// Using 1 block/SM to minimize TMA contention at scale
extern "C" __global__ void 
__launch_bounds__(BLOCK_SIZE, 1)
gdr_tma_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ o,
    const __nv_bfloat16* __restrict__ h0,
    __nv_bfloat16* __restrict__ ht,
    const int32_t* __restrict__ ssm_state_indices,
    const float scale,
    const int64_t N,
    const int H,
    const int HV,
    const int K,
    const int V,
    const int64_t stride_state_token,
    const CUtensorMap* __restrict__ tma_desc_device
) {
    const CUtensorMap* desc = tma_desc_device;
    const int i_v = blockIdx.y;
    const int i_nh = blockIdx.z;
    const int i_n = i_nh / HV;
    const int i_hv = i_nh % HV;
    const int i_h = i_hv / (HV / H);
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // CUTLASS barriers use raw uint64_t storage in shared memory
    // The ClusterTransactionBarrier type is used via static methods
    __shared__ __align__(8) uint64_t tma_barrier[2];
    __shared__ float smem_k[BK];
    __shared__ float smem_q[BK];
    __shared__ __align__(256) __nv_bfloat16 smem_state[BK * BV];
    
    int64_t state_idx = static_cast<int64_t>(__ldg(ssm_state_indices + i_n));
    
    // Initialize both barriers using CUTLASS ClusterTransactionBarrier static methods
    if (tid == 0) {
        cutlass::arch::ClusterTransactionBarrier::init(&tma_barrier[0], 1);
        cutlass::arch::ClusterTransactionBarrier::init(&tma_barrier[1], 1);
    }
    __syncthreads();
    
    int32_t row_offset = static_cast<int32_t>((state_idx * HV + i_hv) * K);
    int32_t col_base = i_v * BV;
    
    // Issue both TMA loads before waiting (pipelined)
    // CUTLASS ClusterTransactionBarrier::arrive_and_expect_tx handles transaction tracking
    if (tid == 0) {
        uint32_t bytes = TMA_TILE_K * TMA_TILE_V * sizeof(__nv_bfloat16);
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&tma_barrier[0], bytes);
        tma_load_2d(desc, &tma_barrier[0], smem_state, col_base, row_offset);
        
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&tma_barrier[1], bytes);
        tma_load_2d(desc, &tma_barrier[1], &smem_state[BK * TMA_TILE_V], col_base + TMA_TILE_V, row_offset);
    }
    __syncthreads();
    
    // Wait for both loads using CUTLASS barrier wait() - phase 0
    // Note: CUTLASS barrier wait() may have different nanosleep behavior
    // than the custom v33 implementation (which yields every 256 iterations)
    cutlass::arch::ClusterTransactionBarrier::wait(&tma_barrier[0], 0);
    cutlass::arch::ClusterTransactionBarrier::wait(&tma_barrier[1], 0);
    
    if (warp_id == 0) {
        float4 k_vec = load_vec4_ldg(k + (i_n * H + i_h) * K + lane_id * 4);
        float4 q_vec = load_vec4_ldg(q + (i_n * H + i_h) * K + lane_id * 4);
        
        float k_sq = k_vec.x*k_vec.x + k_vec.y*k_vec.y + k_vec.z*k_vec.z + k_vec.w*k_vec.w;
        float q_sq = q_vec.x*q_vec.x + q_vec.y*q_vec.y + q_vec.z*q_vec.z + q_vec.w*q_vec.w;
        
        #pragma unroll
        for (int off = 16; off > 0; off /= 2) {
            k_sq += __shfl_xor_sync(0xffffffff, k_sq, off);
            q_sq += __shfl_xor_sync(0xffffffff, q_sq, off);
        }
        
        float k_inv = rsqrtf(k_sq + L2_NORM_EPS);
        float q_inv = rsqrtf(q_sq + L2_NORM_EPS) * scale;
        
        k_vec.x *= k_inv; k_vec.y *= k_inv; k_vec.z *= k_inv; k_vec.w *= k_inv;
        q_vec.x *= q_inv; q_vec.y *= q_inv; q_vec.z *= q_inv; q_vec.w *= q_inv;
        
        smem_k[lane_id*4+0] = k_vec.x; smem_k[lane_id*4+1] = k_vec.y;
        smem_k[lane_id*4+2] = k_vec.z; smem_k[lane_id*4+3] = k_vec.w;
        smem_q[lane_id*4+0] = q_vec.x; smem_q[lane_id*4+1] = q_vec.y;
        smem_q[lane_id*4+2] = q_vec.z; smem_q[lane_id*4+3] = q_vec.w;
    }
    __syncthreads();
    
    const int o_v = tid;
    float b_v = load_scalar_ldg(v + (i_n * HV + i_hv) * V + o_v);
    float b_beta = load_scalar_ldg(beta + i_n * HV + i_hv);
    float decay = __expf(__ldg(g + i_n * HV + i_hv));
    float delta = b_v;
    
    const int tile_idx = tid / TMA_TILE_V;
    const int col_in_tile = tid % TMA_TILE_V;
    const int tile_base = tile_idx * BK * TMA_TILE_V;
    
    #pragma unroll 16
    for (int k_idx = 0; k_idx < BK; ++k_idx) {
        int smem_idx = tile_base + k_idx * TMA_TILE_V + col_in_tile;
        float h_val = __bfloat162float(smem_state[smem_idx]);
        delta = fmaf(-h_val * decay, smem_k[k_idx], delta);
    }
    delta *= b_beta;
    
    float b_o = 0.0f;
    __nv_bfloat16* p_ht = ht + state_idx * stride_state_token + i_hv * K * V + i_v * BV;
    
    #pragma unroll 8
    for (int k_idx = 0; k_idx < BK; k_idx += 2) {
        int smem_idx0 = tile_base + k_idx * TMA_TILE_V + col_in_tile;
        int smem_idx1 = tile_base + (k_idx+1) * TMA_TILE_V + col_in_tile;
        
        float h0_val = __bfloat162float(smem_state[smem_idx0]);
        float h1_val = __bfloat162float(smem_state[smem_idx1]);
        
        float h_new_0 = fmaf(smem_k[k_idx], delta, h0_val * decay);
        float h_new_1 = fmaf(smem_k[k_idx+1], delta, h1_val * decay);
        
        b_o = fmaf(h_new_0, smem_q[k_idx], b_o);
        b_o = fmaf(h_new_1, smem_q[k_idx+1], b_o);
        
        store_scalar_streaming(p_ht + k_idx * V + o_v, h_new_0);
        store_scalar_streaming(p_ht + (k_idx+1) * V + o_v, h_new_1);
    }
    
    o[(i_n * HV + i_hv) * V + o_v] = __float2bfloat16(b_o);
}

//=============================================================================
// Host API: Create TMA descriptor
//=============================================================================

extern "C" CUresult create_tma_descriptor_state_tensor(
    CUtensorMap* tensor_map,
    void* global_address,
    uint64_t total_rows,
    uint64_t num_cols,
    uint64_t stride_row_bytes
) {
    uint64_t dims[2] = {num_cols, total_rows};
    uint64_t strides[1] = {stride_row_bytes};
    uint32_t box_dims[2] = {TMA_TILE_V, TMA_TILE_K};
    uint32_t elem_strides[2] = {1, 1};
    
    if ((uintptr_t)global_address % 16 != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    if (stride_row_bytes % 16 != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    return cuTensorMapEncodeTiled(
        tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        global_address,
        dims,
        strides,
        box_dims,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

//=============================================================================
// Host API: Launch kernels
//=============================================================================

extern "C" void launch_gdr_tma(
    const void* q, const void* k, const void* v, const float* g, const void* beta,
    void* o, const void* h0, void* ht,
    const int32_t* ssm_state_indices,
    float scale,
    int64_t N, int H, int HV, int K, int V,
    int64_t stride_state_token,
    const CUtensorMap* tma_desc_device,
    cudaStream_t stream
) {
    dim3 grid(1, (V + BV - 1) / BV, N * HV);
    dim3 block(BLOCK_SIZE);
    
    gdr_tma_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        g,
        reinterpret_cast<const __nv_bfloat16*>(beta),
        reinterpret_cast<__nv_bfloat16*>(o),
        reinterpret_cast<const __nv_bfloat16*>(h0),
        reinterpret_cast<__nv_bfloat16*>(ht),
        ssm_state_indices,
        scale, N, H, HV, K, V, stride_state_token,
        tma_desc_device
    );
}
