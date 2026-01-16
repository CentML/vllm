#!/bin/bash
# Build script for v33 TMA kernel

set -e

CUDA_HOME=${CUDA_HOME:-/home/scratch.vgimpelson_ent/cuda/cuda129}
NVCC=${CUDA_HOME}/bin/nvcc

# Force SM100 only
GPU_ARCH="sm_100"

echo "Building v33 for ${GPU_ARCH}..."
echo "CUDA_HOME: ${CUDA_HOME}"

${NVCC} \
    -arch=${GPU_ARCH} \
    -O3 \
    --use_fast_math \
    -Xptxas=-O3 \
    -std=c++17 \
    --expt-relaxed-constexpr \
    -I${CUDA_HOME}/include \
    -L${CUDA_HOME}/lib64 \
    -lcuda \
    -shared \
    -Xcompiler -fPIC \
    -o libgdr_v33_tma.so \
    fused_recurrent_gated_delta_rule_v33_tma.cu

echo "Built libgdr_v33_tma.so successfully!"
