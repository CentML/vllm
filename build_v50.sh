#!/bin/bash
# Build script for v50 TMA kernel with CUTLASS barriers

set -e

CUDA_HOME=${CUDA_HOME:-/home/scratch.vgimpelson_ent/cuda/cuda129}
NVCC=${CUDA_HOME}/bin/nvcc

# Try to find CUTLASS directory
# Priority order: venv deep_gemm > standalone cutlass > env vars > CMake build dirs
# Note: Auto-detection takes precedence over env vars to avoid stale values
if [ -d "/home/scratch.vgimpelson_ent/venv_b200/lib/python3.12/site-packages/deep_gemm/include/cutlass" ]; then
    # Use DeepGEMM's CUTLASS headers from venv_b200 (tested and working)
    CUTLASS_DIR="/home/scratch.vgimpelson_ent/venv_b200/lib/python3.12/site-packages/deep_gemm"
elif [ -d "/home/scratch.vgimpelson_ent/cutlass/include/cutlass" ]; then
    # Use standalone CUTLASS installation
    CUTLASS_DIR="/home/scratch.vgimpelson_ent/cutlass"
elif [ -n "$VLLM_CUTLASS_SRC_DIR" ]; then
    CUTLASS_DIR="$VLLM_CUTLASS_SRC_DIR"
elif [ -n "$CUTLASS_DIR" ]; then
    # Already set, use it (but auto-detection is preferred)
    :
elif [ -d "_deps/cutlass-src/include" ]; then
    CUTLASS_DIR="_deps/cutlass-src"
elif [ -d "build/_deps/cutlass-src/include" ]; then
    CUTLASS_DIR="build/_deps/cutlass-src"
else
    echo "ERROR: CUTLASS directory not found!"
    echo "Please set CUTLASS_DIR or VLLM_CUTLASS_SRC_DIR environment variable"
    echo "Example: export CUTLASS_DIR=/path/to/cutlass"
    echo "Or run from CMake build directory where CUTLASS was downloaded"
    exit 1
fi

# Verify CUTLASS include path exists
if [ ! -f "${CUTLASS_DIR}/include/cutlass/cutlass.h" ]; then
    echo "ERROR: CUTLASS headers not found at ${CUTLASS_DIR}/include/cutlass/cutlass.h"
    echo "Please check your CUTLASS_DIR: ${CUTLASS_DIR}"
    exit 1
fi

# Force SM100 only
GPU_ARCH="sm_100"

echo "Building v50 for ${GPU_ARCH}..."
echo "CUDA_HOME: ${CUDA_HOME}"
echo "CUTLASS_DIR: ${CUTLASS_DIR}"

${NVCC} \
    -arch=${GPU_ARCH} \
    -O3 \
    --use_fast_math \
    -Xptxas=-O3 \
    -std=c++17 \
    --expt-relaxed-constexpr \
    -I${CUDA_HOME}/include \
    -I${CUTLASS_DIR}/include \
    -L${CUDA_HOME}/lib64 \
    -lcuda \
    -shared \
    -Xcompiler -fPIC \
    -o libgdr_v50_tma.so \
    fused_recurrent_gated_delta_rule_v50_tma.cu

echo "Built libgdr_v50_tma.so successfully!"
