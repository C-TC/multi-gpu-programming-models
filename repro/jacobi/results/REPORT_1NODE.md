# 1-node Jacobi reproduction — H200 cluster (2026-05-04)

Runs from `results-1node-20260504-161658.csv`. Times are mean ± stdev (s) over reps.

**Cluster**: 1 × H200 SXM 144 GB, CUDA 13.0.88, NCCL 2.28.8 (system) /
2.30.4 (DeepEP V2 path), NVSHMEM 3.6.5 (pip wheel), Open MPI 4.1.9a1,
container `gpu_882f6e72.sqsh`. Source code unchanged from the original H100
run except for the `-lnvshmem → -lnvshmem_host -lnvshmem_device` Makefile
patch needed for NVSHMEM 3 (split host/device libs). Re-run via
`bash repro/jacobi/build_all.sh && bash repro/jacobi/run_sweep.sh`.

## Sweep A — scale-up at nx=ny=16384, niter=1000

| Variant | 1 GPU | 2 GPU | 4 GPU | 8 GPU |
|---|---|---|---|---|
| single_gpu | 1.465 ± 0.000 | — | — | — |
| mpi | — | 1.546 ± 0.300 | 1.519 ± 0.014 | 2.192 ± 0.049 |
| mpi_overlap | — | 0.864 ± 0.159 | 1.074 ± 0.264 | 2.031 ± 0.062 |
| nccl | — | 0.797 ± 0.002 | 0.416 ± 0.003 | 0.234 ± 0.002 |
| nccl_overlap | — | 0.776 ± 0.004 | 0.406 ± 0.002 | 0.222 ± 0.001 |
| nccl_graphs | — | 0.774 ± 0.001 | 0.417 ± 0.015 | 0.220 ± 0.001 |
| nvshmem | — | 0.904 ± 0.003 | 0.469 ± 0.001 | 0.254 ± 0.002 |
| nvshmem_block | — | 0.923 ± 0.002 | 0.479 ± 0.002 | 0.260 ± 0.000 |
| nvshmem_block_nbsync | — | 0.928 ± 0.007 | 0.477 ± 0.001 | 0.257 ± 0.000 |

## Sweep B — nx axis at 8 GPU, ny=16384, niter=1000

| Variant | nx=2048 | nx=4096 | nx=8192 | nx=16384 |
|---|---|---|---|---|
| nccl | 0.056 ± 0.000 | 0.083 ± 0.002 | 0.137 ± 0.003 | 0.236 ± 0.003 |
| nccl_graphs | 0.110 ± 0.014 | 0.095 ± 0.029 | 0.135 ± 0.014 | 0.230 ± 0.021 |
| nvshmem | 0.052 ± 0.000 | 0.089 ± 0.001 | 0.144 ± 0.002 | 0.255 ± 0.003 |
| nvshmem_block_nbsync | 0.054 ± 0.002 | 0.088 ± 0.003 | 0.148 ± 0.003 | 0.259 ± 0.002 |

## Headline numbers (8 GPU, 16384², 1000 iter)

| Variant | H200 (this run) | H100 (original) |
|---|---:|---:|
| nccl                | **0.234 s** | ≈0.243 s |
| nccl_graphs         | **0.220 s** | ≈0.220 s |
| nvshmem             | **0.254 s** | ≈0.252 s |
| nvshmem_block_nbsync | **0.257 s** | (not in original) |

* Numbers are 3-rep means; H200 stdev < 0.003 s for all four variants.
* `nccl_graphs` is again the fastest variant on a single node — same shape
  as the original H100 result. `nccl` (~6%) and `nvshmem` (~15%) trail.
* The H200's extra HBM and compute don't change the picture much at this
  size — these workloads are NVLink-bound, and the NVLink topology /
  per-link bandwidth is unchanged from H100.

## Same conclusions as the original H100 run

1. **At 8 GPU 16384², `nccl_graphs` ≈ 6% faster than `nccl`, ≈ 15% faster
   than `nvshmem`** — the extra latency saved by replacing per-iteration
   NCCL launch + halo with a captured CUDA graph is worth more than what
   NVSHMEM's lower-overhead RDMA-style halo gives back, on a single-node
   NVLink-only setup.
2. **`mpi`/`mpi_overlap` regress at higher GPU counts** (2.19 s and 2.03 s
   at 8 GPU) — the container's Open MPI is not CUDA-aware here either.
   Same as on H100. Read these two rows as "MPI baseline without GPUDirect"
   not as the achievable MPI number.
3. **The `-use_block_comm` and `-use_block_comm -neighborhood_sync`
   tunings make no measurable difference on a single node** — they only
   matter for the multi-node case (see [REPORT_2NODE.md](REPORT_2NODE.md)).

## Figures

Six PNGs in [figures/](figures/):

* `fig_a_runtime_vs_gpu_1node.png` — runtime vs GPU count, all variants
* `fig_a_speedup_vs_gpu_1node.png` — speedup vs `single_gpu` baseline
* `fig_b_runtime_vs_nx_1node.png`  — runtime vs nx, 8 GPU
* `fig_a_runtime_vs_gpu_2node.png` — same as A but for 2 nodes
* `fig_b_runtime_vs_nx_2node.png`  — same as B but for 2 nodes
* `fig_nvshmem_block_ablation.png` — nvshmem flag ablation, side-by-side
