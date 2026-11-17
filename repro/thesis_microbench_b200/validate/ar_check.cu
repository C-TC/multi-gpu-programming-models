// Correctness check for exactly the NVSHMEM allreduce paths plotted in the paper's figure:
//   nvshmemx_float_sum_reduce_on_stream  (host on-stream: multi-CTA / NVLS intra-node)
//   nvshmemx_float_sum_reduce_block      (device, single CTA, as in reduction_focus)
// over NVSHMEM_TEAM_WORLD, float/sum, sizes 128 B .. MAX bytes (x4 steps).
// Inputs are small integers so every partial sum is exact in fp32 -> bitwise comparison.
// Prints one line per (path,size) and a final "AR_CHECK <PASS|FAIL> errors=N" on PE 0; exit code = #errors>0.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(2);} } while (0)

__global__ void init(float *src, float *dst, size_t n, int pe) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        src[i] = (float)((i * 7 + pe * 13) % 31);  // 0..30
        dst[i] = -1.0f;
    }
}
__global__ void check(const float *dst, size_t n, int npes, unsigned long long *bad) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float exp = 0.f;
        for (int p = 0; p < npes; p++) exp += (float)((i * 7 + p * 13) % 31);
        if (dst[i] != exp) atomicAdd(bad, 1ULL);
    }
}
__global__ void dev_block_reduce(nvshmem_team_t team, float *dst, const float *src, size_t n) {
    nvshmemx_float_sum_reduce_block(team, dst, src, n);
}

int main(int argc, char **argv) {
    size_t maxb = argc > 1 ? strtoull(argv[1], 0, 0) : (512ull << 20);
    size_t maxb_dev = argc > 2 ? strtoull(argv[2], 0, 0) : maxb;
    nvshmem_init();
    int pe = nvshmem_my_pe(), npes = nvshmem_n_pes();
    CK(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    size_t nmax = maxb / sizeof(float);
    float *src = (float *)nvshmem_malloc(maxb), *dst = (float *)nvshmem_malloc(maxb);
    unsigned long long *bad; CK(cudaMalloc(&bad, sizeof(*bad)));
    unsigned long long total = 0;
    for (int path = 0; path < 2; path++) {
        size_t lim = path ? maxb_dev : maxb;
        for (size_t bytes = 128; bytes <= lim; bytes *= 4) {
            size_t n = bytes / sizeof(float);
            init<<<1024, 256, 0, s>>>(src, dst, n, pe);
            CK(cudaMemsetAsync(bad, 0, sizeof(*bad), s));
            CK(cudaStreamSynchronize(s));
            nvshmem_barrier_all();
            if (path == 0) {
                nvshmemx_float_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, dst, src, n, s);
            } else {
                nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
                void *args[] = {&team, &dst, &src, &n};
                if (nvshmemx_collective_launch((const void *)dev_block_reduce, 1, 256, args, 0, s)) { fprintf(stderr, "launch failed\n"); exit(2); }
            }
            CK(cudaStreamSynchronize(s));
            check<<<1024, 256, 0, s>>>(dst, n, npes, bad);
            unsigned long long h = 0;
            CK(cudaMemcpyAsync(&h, bad, sizeof(h), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            if (h) printf("AR_CHECK_LOCAL pe=%d path=%d bytes=%zu bad_elems=%llu\n", pe, path, bytes, h);
            // gather error counts to PE 0 via a symmetric reduction of a counter
            unsigned long long *cnt = (unsigned long long *)nvshmem_malloc(2 * sizeof(unsigned long long));
            CK(cudaMemcpy(cnt, &h, sizeof(h), cudaMemcpyHostToDevice));
            nvshmem_barrier_all();
            nvshmemx_uint64_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, (uint64_t *)cnt + 1, (uint64_t *)cnt, 1, s);
            CK(cudaStreamSynchronize(s));
            unsigned long long all = 0;
            CK(cudaMemcpy(&all, cnt + 1, sizeof(all), cudaMemcpyDeviceToHost));
            nvshmem_free(cnt);
            total += all;
            if (pe == 0) printf("AR_CHECK path=%s npes=%d bytes=%zu bad_elems=%llu\n", path ? "device_block" : "on_stream", npes, bytes, all);
        }
    }
    if (pe == 0) printf("AR_CHECK %s errors=%llu\n", total ? "FAIL" : "PASS", total);
    nvshmem_free(src); nvshmem_free(dst);
    nvshmem_finalize();
    return total ? 1 : 0;
}
