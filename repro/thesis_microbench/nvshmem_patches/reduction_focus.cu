/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

/* Stripped-down reduction perftest: only float-sum-block scope, no inner
 * dtype/redop/scope iteration. Used to fairly compare against NCCL's
 * all_reduce(float, sum) which engages NVLink-SHARP (LDMC/STMC) on H200.
 *
 * Original `reduction_latency.cu` always iterates 2 dtypes (int32, int64) ×
 * 7 redops × 3 scopes (thread, warp, block) per size, which (a) takes too
 * long for cross-node sweeps and (b) is restricted to integer types so
 * NVSHMEM never engages the multicast path. This file only does float-sum
 * with block scope so the binary completes in the slurmstep timeout AND
 * the path NVSHMEM picks for sum-on-float can engage NVLS where available.
 */

#define CUMODULE_NAME "reduction_focus.cubin"

#include "utils.h"
#include "coll_test.h"
#define LARGEST_DT float

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define CALL_RDXN(TG_PRE, TG, TYPENAME, TYPE, OP, THREAD_COMP, ELEM_COMP)                         \
                                                                                                  \
    void call_test_##TYPENAME##_##OP##_reduce_kern##TG##_cubin(                                   \
        int num_blocks, int num_tpb, cudaStream_t stream, void **arglist) {                       \
        CUfunction test_##TYPENAME##_##OP##_reduce_kern##TG_cubin;                                \
                                                                                                  \
        init_test_case_kernel(&test_##TYPENAME##_##OP##_reduce_kern##TG_cubin,                    \
                              NVSHMEMI_TEST_STRINGIFY(test_##TYPENAME##_##OP##_reduce_kern##TG)); \
        CU_CHECK(cuLaunchCooperativeKernel(test_##TYPENAME##_##OP##_reduce_kern##TG_cubin,        \
                                           num_blocks, 1, 1, num_tpb, 1, 1, 0, stream, arglist)); \
    }                                                                                             \
                                                                                                  \
    __global__ void test_##TYPENAME##_##OP##_reduce_kern##TG(                                     \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, int nelems, int iter) {              \
        int i;                                                                                    \
                                                                                                  \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP) && (nelems < ELEM_COMP)) {                 \
            for (i = 0; i < iter; i++) {                                                          \
                nvshmem##TG_PRE##_##TYPENAME##_##OP##_reduce##TG(team, dest, source, nelems);     \
            }                                                                                     \
        }                                                                                         \
    }

#define CALL_RDXN_KERNEL(TYPENAME, OP, TG, BLOCKS, THREADS, ARG_LIST, STREAM)                     \
    if (use_cubin) {                                                                              \
        call_test_##TYPENAME##_##OP##_reduce_kern##TG##_cubin(BLOCKS, THREADS, STREAM, ARG_LIST); \
    } else {                                                                                      \
        status =                                                                                  \
            nvshmemx_collective_launch((const void *)test_##TYPENAME##_##OP##_reduce_kern##TG,    \
                                       BLOCKS, THREADS, ARG_LIST, 0, STREAM);                     \
        if (status != NVSHMEMX_SUCCESS) {                                                         \
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);                     \
            exit(-1);                                                                             \
        }                                                                                         \
    }

/* Only float-sum-block. Float supports sum/prod/min/max in NVSHMEM
 * (and/or/xor are bitwise — integer-only). We only need sum.
 */
CALL_RDXN(x, _block, float, float, sum, INT_MAX, INT_MAX)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define SET_SIZE_ARR(TYPE, ELEM_COMP)                                                      \
    do {                                                                                   \
        j = 0;                                                                             \
        for (num_elems = min_elems; num_elems <= max_elems; num_elems *= step_factor) {    \
            if (num_elems < ELEM_COMP) {                                                   \
                size_arr[j] =                                                              \
                    calculate_collective_size("reduction", num_elems, sizeof(TYPE), npes); \
            } else {                                                                       \
                size_arr[j] = 0;                                                           \
            }                                                                              \
            j++;                                                                           \
        }                                                                                  \
    } while (0)

#define RUN_ITERS_OP(TYPENAME, TYPE, GROUP, OP, ELEM_COMP)                                       \
    do {                                                                                         \
        void *skip_arg_list[] = {&team, &dest, &source, &num_elems, &skip};                      \
        void *time_arg_list[] = {&team, &dest, &source, &num_elems, &iter};                      \
        float milliseconds;                                                                      \
        cudaEvent_t start, stop;                                                                 \
        cudaEventCreate(&start);                                                                 \
        cudaEventCreate(&stop);                                                                  \
        SET_SIZE_ARR(TYPE, ELEM_COMP);                                                           \
                                                                                                 \
        nvshmem_barrier_all();                                                                   \
        j = 0;                                                                                   \
        for (num_elems = min_elems; num_elems < ELEM_COMP; num_elems *= 2) {                     \
            CALL_RDXN_KERNEL(TYPENAME, OP, GROUP, num_blocks, nvshm_test_num_tpb, skip_arg_list, \
                             stream);                                                            \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                           \
            nvshmem_barrier_all();                                                               \
                                                                                                 \
            cudaEventRecord(start, stream);                                                      \
            CALL_RDXN_KERNEL(TYPENAME, OP, GROUP, num_blocks, nvshm_test_num_tpb, time_arg_list, \
                             stream);                                                            \
            cudaEventRecord(stop, stream);                                                       \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                           \
                                                                                                 \
            if (!mype) {                                                                         \
                cudaEventElapsedTime(&milliseconds, start, stop);                                \
                h_##OP##_lat[j] = (milliseconds * 1000.0) / (float)iter;                         \
            }                                                                                    \
            nvshmem_barrier_all();                                                               \
            j++;                                                                                 \
        }                                                                                        \
    } while (0)

int rdxn_calling_kernel(nvshmem_team_t team, void *dest, const void *source, int mype,
                        cudaStream_t stream, run_opt_t run_options, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = threads_per_block;
    int num_blocks = 1;
    size_t num_elems = 1, min_elems, max_elems;
    int iter = iters;
    int skip = warmup_iters;
    int j;
    int npes = nvshmem_n_pes();
    uint64_t *size_arr = (uint64_t *)h_tables[0];
    double *h_sum_lat = (double *)h_tables[1];

    /* Only float-sum-block. */
    min_elems = max(static_cast<size_t>(1), min_size / sizeof(float));
    max_elems = max(static_cast<size_t>(1), max_size / sizeof(float));
    RUN_ITERS_OP(float, float, _block, sum, max_elems);
    if (!mype) {
        print_table_v1("device_reduction", "float-sum-b", "size (Bytes)", "latency", "us", '-',
                       size_arr, h_sum_lat, j);
    }

    return status;
}

int main(int argc, char **argv) {
    int status = 0;
    int mype, array_size;
    size_t size = 0;

    read_args(argc, argv);
    float *h_buffer = NULL;
    float *d_source, *d_dest;
    float *h_source, *h_dest;
    char size_string[100];
    cudaStream_t cstrm;
    run_opt_t run_options;
    void **h_tables;

    /* Only run_block matters; the others are unused but kept for API parity. */
    run_options.run_thread = run_options.run_warp = 0;
    run_options.run_block = 1;

    size = page_size_roundoff(max_size);   // send buf
    size += page_size_roundoff(max_size);  // recv buf

    DEBUG_PRINT("symmetric size requested %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    array_size = max_size_log;

    init_wrapper(&argc, &argv);
    /* Only need 2 tables: size_arr + h_sum_lat. Allocate the original 8 to
     * stay compatible with utils.cu's table layout. */
    alloc_tables(&h_tables, 8, array_size);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    mype = nvshmem_my_pe();

    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    CUDA_CHECK(cudaHostAlloc(&h_buffer, max_size * 2, cudaHostAllocDefault));
    h_source = h_buffer;
    h_dest = &h_source[max_size / sizeof(float)];

    d_source = (float *)nvshmem_align(getpagesize(), max_size);
    d_dest = (float *)nvshmem_align(getpagesize(), max_size);

    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, max_size, cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, max_size, cudaMemcpyHostToDevice, cstrm));

    rdxn_calling_kernel(NVSHMEM_TEAM_WORLD, d_dest, d_source, mype, cstrm, run_options, h_tables);

    DEBUG_PRINT("last error = %s\n", cudaGetErrorString(cudaGetLastError()));

    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, max_size, cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, max_size, cudaMemcpyDeviceToHost, cstrm));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_source);
    nvshmem_free(d_dest);

    CUDA_CHECK(cudaStreamDestroy(cstrm));

    finalize_wrapper();

out:
    return 0;
}
