// Build an SM-to-locality-domain map for domain-aware SM107 decode.
// Green-context probes identify domain membership. Unattributed SMs are mapped
// by dependent-load probes against domain-local buffers, with a tie-break.
// Exported for ctypes; returns zero on success and nonzero if unavailable.
// nsm_out[2] packs attributed and remaining SM counts for the caller.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <vector>

__global__ void dec107_sm_probe(int* tab, int dom) {
  if (threadIdx.x == 0) { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); tab[s & 511] = dom; }
}

__global__ void dec107_chase_fill(unsigned* b, unsigned n, unsigned seed) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    unsigned x = i * 2654435761u ^ seed; x ^= x >> 15; x *= 0x2c1b3c6du; x ^= x >> 12; b[i] = (x % (n / 64)) * 64;  // 256-B hops
  }
}
// one thread per CTA; the first CTA on each orphan SM measures its mean dependent-load latency to both buffers
__global__ void dec107_orphan_lat(const unsigned* b0, const unsigned* b1, const int* tab, int* claimed, long long* lat) {
  if (threadIdx.x) return;
  unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); s &= 511;
  if (tab[s] >= 0 || atomicCAS(&claimed[s], 0, 1) != 0) return;
  for (int d = 0; d < 2; d++) {
    const unsigned* b = d ? b1 : b0;
    unsigned idx = (s * 977u) * 64u % (1u << 20);
    for (int i = 0; i < 16; i++) idx = __ldcg(b + idx);                 // warm the TLB
    long long t0 = clock64();
    for (int i = 0; i < 64; i++) idx = __ldcg(b + idx);
    long long t1 = clock64();
    lat[2 * s + d] = (t1 - t0) / 64 + (idx == 0xFFFFFFFFu);
  }
}

static void* dec107_alloc_dom(size_t bytes, int dev, int d) {
  CUmemAllocationProp prop; memset(&prop, 0, sizeof prop); prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN; prop.location.localized.deviceId = (unsigned char)dev;
  prop.location.localized.localityDomainId = (unsigned char)d;
  size_t g = 0; if (cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) return nullptr;
  bytes = (bytes + g - 1) / g * g;
  CUmemGenericAllocationHandle h; if (cuMemCreate(&h, bytes, &prop, 0) != CUDA_SUCCESS) return nullptr;
  CUdeviceptr va; if (cuMemAddressReserve(&va, bytes, g, 0, 0) != CUDA_SUCCESS) { cuMemRelease(h); return nullptr; }
  cuMemMap(va, bytes, 0, h, 0); cuMemRelease(h);
  CUmemAccessDesc a; memset(&a, 0, sizeof a); a.location.type = CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = dev; a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuMemSetAccess(va, bytes, &a, 1);
  return (void*)va;
}

// latency attribution of the orphan SMs (h[smid] == -1) to a die; returns the number attributed
static int dec107_attribute_orphans(int dev, int* h, int nsm_total) {
  const size_t NB = (size_t)256 << 20;                 // Probe working set intended to miss L2.
  unsigned* b[2] = {(unsigned*)dec107_alloc_dom(NB, dev, 0), (unsigned*)dec107_alloc_dom(NB, dev, 1)};
  if (!b[0] || !b[1]) return -1;
  for (int d = 0; d < 2; d++) dec107_chase_fill<<<1024, 256>>>(b[d], (unsigned)(NB / 4), 0x9E3779B9u * (d + 1));
  int *dt, *cl; long long* lat;
  cudaMalloc(&dt, 512 * sizeof(int)); cudaMalloc(&cl, 512 * sizeof(int)); cudaMalloc(&lat, 1024 * sizeof(long long));
  cudaMemcpy(dt, h, 512 * sizeof(int), cudaMemcpyHostToDevice); cudaMemset(cl, 0, 512 * sizeof(int)); cudaMemset(lat, 0, 1024 * sizeof(long long));
  for (int rep = 0; rep < 4; rep++) dec107_orphan_lat<<<nsm_total * 8, 32>>>(b[0], b[1], dt, cl, lat);   // oversubscribe so every SM runs one
  std::vector<long long> l(1024); cudaMemcpy(l.data(), lat, 1024 * sizeof(long long), cudaMemcpyDeviceToHost);
  int n = 0;
  std::vector<int> orphan(512, 0);
  for (int sm = 0; sm < 512; sm++) orphan[sm] = (h[sm] == -1);
  for (int sm = 0; sm < 512; sm++) {
    if (!orphan[sm] || l[2 * sm] <= 0 || l[2 * sm + 1] <= 0) continue;
    const long long a = l[2 * sm], c = l[2 * sm + 1];
    if (a * 10 < c * 9) { h[sm] = 0; n++; } else if (c * 10 < a * 9) { h[sm] = 1; n++; }   // Require a clear separation.
  }
  // Tie-break for undecided SMs: prefer the paired SM's decision, then parity.
  for (int sm = 0; sm < 512; sm++) {
    if (!orphan[sm] || h[sm] != -1 || sm >= nsm_total) continue;
    const int p = sm ^ 1;
    h[sm] = (p < 512 && orphan[p] && h[p] >= 0) ? h[p] : (sm & 1);
  }
  cudaFree(dt); cudaFree(cl); cudaFree(lat);
  for (int d = 0; d < 2; d++) { size_t g = 0; CUmemAllocationProp p; memset(&p, 0, sizeof p); p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN; p.location.localized.deviceId = (unsigned char)dev; p.location.localized.localityDomainId = (unsigned char)d;
    cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED); size_t nb = (NB + g - 1) / g * g;
    cuMemUnmap((CUdeviceptr)b[d], nb); cuMemAddressFree((CUdeviceptr)b[d], nb); }
  return n;
}

extern "C" int dec107_sm_domain_table(int dev, signed char* out256, int* nsm_out) {
  for (int i = 0; i < 256; i++) out256[i] = -1;
  int nd = 1;
  if (cudaDeviceGetAttribute(&nd, cudaDevAttrLocalityDomainCount, dev) != cudaSuccess || nd < 2) { cudaGetLastError(); return 1; }
  CUcontext prim = nullptr;
  if (cuCtxGetCurrent(&prim) != CUDA_SUCCESS || prim == nullptr) return 2;
  int* dt = nullptr;
  if (cudaMalloc(&dt, 512 * sizeof(int)) != cudaSuccess) return 3;
  int h[512]; for (int i = 0; i < 512; i++) h[i] = -1;
  cudaMemcpy(dt, h, sizeof h, cudaMemcpyHostToDevice);
  CUdevice cd; cuDeviceGet(&cd, dev);
  int rc = 0;
  for (int d = 0; d < nd && d < 2 && rc == 0; d++) {
    CUdevResource in, res, rem;
    if (cuDeviceGetDevResource(cd, &in, CU_DEV_RESOURCE_TYPE_SM) != CUDA_SUCCESS) { rc = 4; break; }
    CU_DEV_SM_RESOURCE_GROUP_PARAMS gp; memset(&gp, 0, sizeof gp);
    gp.flags = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID; gp.localityDomainId = (unsigned)d;
    if (cuDevSmResourceSplit(&res, 1, &in, &rem, 0, &gp) != CUDA_SUCCESS) { rc = 5; break; }
    CUdevResourceDesc desc; CUgreenCtx g; CUcontext c; CUstream s;
    if (cuDevResourceGenerateDesc(&desc, &res, 1) != CUDA_SUCCESS || cuGreenCtxCreate(&g, desc, cd, CU_GREEN_CTX_DEFAULT_STREAM) != CUDA_SUCCESS) { rc = 6; break; }
    cuCtxFromGreenCtx(&c, g); cuGreenCtxStreamCreate(&s, g, CU_STREAM_NON_BLOCKING, 0);
    cuCtxSetCurrent(c);
    dec107_sm_probe<<<(int)res.sm.smCount * 8, 32, 0, (cudaStream_t)s>>>(dt, d);
    if (cudaGetLastError() != cudaSuccess || cuStreamSynchronize(s) != CUDA_SUCCESS) rc = 7;
    cuCtxSetCurrent(prim); cuStreamDestroy(s); cuGreenCtxDestroy(g);
  }
  cuCtxSetCurrent(prim);
  cudaMemcpy(h, dt, sizeof h, cudaMemcpyDeviceToHost); cudaFree(dt);
  if (rc) return rc;
  int nsm_total = 0; cudaDeviceGetAttribute(&nsm_total, cudaDevAttrMultiProcessorCount, dev);
  std::vector<int> was_orphan(512, 0);
  int orphans = 0;
  for (int i = 0; i < nsm_total && i < 512; i++) { was_orphan[i] = h[i] < 0; orphans += was_orphan[i]; }
  // DEC107_SM_DOMAIN_FORCE=parity: orphans -> smid parity without probing (forced map: tests / reproducibility);
  // DEC107_NO_ORPHAN_ATTRIB present: leave orphans at -1 (kernel starts on blockIdx & 1, then steals).
  // default: DRAM-latency probe + deterministic tie-break (TPC partner, then parity).
  const char* force = getenv("DEC107_SM_DOMAIN_FORCE");
  if (orphans > 0 && force && !strcmp(force, "parity")) {
    for (int i = 0; i < nsm_total && i < 512; i++) if (h[i] < 0) h[i] = i & 1;
  } else if (orphans > 0 && !getenv("DEC107_NO_ORPHAN_ATTRIB")) {
    dec107_attribute_orphans(dev, h, nsm_total);
  }
  int attributed = 0;
  for (int i = 0; i < nsm_total && i < 512; i++) attributed += was_orphan[i] && h[i] >= 0;
  nsm_out[2] = (attributed << 16) | (orphans - attributed);
  nsm_out[0] = nsm_out[1] = 0;
  for (int i = 0; i < 256; i++) { out256[i] = (signed char)h[i]; if (h[i] == 0 || h[i] == 1) nsm_out[h[i]]++; }
  return (nsm_out[0] > 0 && nsm_out[1] > 0) ? 0 : 8;
}
