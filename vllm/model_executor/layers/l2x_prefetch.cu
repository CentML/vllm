// l2-x: L2 prefetch of static weight ranges (exact by construction: a cache hint only).
// qwen-v2 build: plain CUDA runtime + extern "C", loaded with ctypes. It does NOT include torch/ATen headers, because the
// CUDAContext headers may require additional development headers unavailable in runtime images.
// One kernel issues cp.async.bulk.prefetch.L2.global over up to L2X_MAX_RANGES (ptr, bytes) ranges passed BY VALUE
// (baked into the CUDA-graph node: no dependent load). Optional evict_last policy; optional PDL launch (inline mode).
#include <cuda_runtime.h>
#include <stdint.h>

#define L2X_MAX_RANGES 48

struct L2xRanges {
  unsigned long long ptr[L2X_MAX_RANGES];
  unsigned long long bytes[L2X_MAX_RANGES];
  int n;
  int chunk;       // bytes per prefetch instruction (multiple of 16)
  int evict_last;  // 1: .L2::cache_hint with createpolicy evict_last
  int pdl;         // 1: griddepcontrol.launch_dependents at entry
};

__global__ void __launch_bounds__(256) l2x_prefetch_kernel(const __grid_constant__ L2xRanges r) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (r.pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  uint64_t pol = 0;
  if (r.evict_last) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t nth = (uint64_t)gridDim.x * blockDim.x;
  const uint64_t chunk = (uint64_t)r.chunk;
  for (int i = 0; i < r.n; ++i) {
    const uint64_t base = r.ptr[i], len = r.bytes[i];
    const uint64_t nch = (len + chunk - 1) / chunk;
    for (uint64_t c = tid; c < nch; c += nth) {
      const uint64_t off = c * chunk;
      uint32_t sz = (uint32_t)((len - off) < chunk ? (len - off) : chunk);
      sz &= ~15u;
      if (sz == 0) continue;
      if (r.evict_last) {
        asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
                     :: "l"(base + off), "r"(sz), "l"(pol) : "memory");
      } else {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                     :: "l"(base + off), "r"(sz) : "memory");
      }
    }
  }
#endif
}

// Returns a cudaError_t (0 = success). ptrs/bytes are host arrays of length n <= L2X_MAX_RANGES; stream is a cudaStream_t.
extern "C" int l2x_prefetch(const unsigned long long* ptrs, const unsigned long long* bytes, int n, int chunk,
                            int ctas, int threads, int evict_last, int pdl, void* stream) {
  if (n <= 0) return 0;
  if (n > L2X_MAX_RANGES) return (int)cudaErrorInvalidValue;
  L2xRanges r;
  for (int i = 0; i < n; ++i) {
    if (ptrs[i] & 15ull) return (int)cudaErrorInvalidValue;
    r.ptr[i] = ptrs[i];
    r.bytes[i] = bytes[i];
  }
  r.n = n;
  r.chunk = chunk & ~15;
  r.evict_last = evict_last ? 1 : 0;
  r.pdl = pdl ? 1 : 0;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)ctas);
  cfg.blockDim = dim3((unsigned)threads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute attr[1];
  if (pdl) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  return (int)cudaLaunchKernelEx(&cfg, l2x_prefetch_kernel, r);
}

// Defensive: drop any persisting-L2 lines this context may hold (l2-x itself never sets a carve-out).
extern "C" int l2x_reset_persisting() {
  cudaError_t e = cudaCtxResetPersistingL2Cache();
  (void)cudaGetLastError();
  return (int)e;
}
