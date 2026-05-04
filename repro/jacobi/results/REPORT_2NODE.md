# 2-node Jacobi reproduction — H200 cluster (2026-05-04)

Runs from `results-2node-20260504-205315.csv`. Times are mean ± stdev (s)
over **3 reps**. Full Sweep A2 + Sweep B2 captured this time after
optimizing `run_sweep_2node.sh` to reuse a pyxis container via
`--container-name=` (cuts per-srun cost from 50 s → 11 s).

**Cluster**: 2 × H200 SXM nodes, 8 GPU each, ConnectX-7 IB. Same toolchain
as `REPORT_1NODE.md`. Run with NPN=4 (8 ranks total) via:
```
JOBID=<id> NPN=4 REPS=3 bash repro/jacobi/run_sweep_2node.sh
```
Total wall time ≈ 30 min for 51 runs.

## Sweep A2 — scale-up at nx=ny=16384, niter=1000, 2 nodes × 4 GPU = 8 ranks

| Variant | runtime (s) |
|---|---:|
| nccl                  | 0.254 ± 0.007 |
| nccl_graphs           | 0.233 ± 0.019 |
| nvshmem               | **5.740 ± 0.197** |
| nvshmem_block         | 0.302 ± 0.003 |
| nvshmem_block_nbsync  | 0.290 ± 0.001 |

## Sweep B2 — nx axis at full 2-node, ny=16384, niter=1000

| Variant | nx=2048 | nx=4096 | nx=8192 | nx=16384 |
|---|---:|---:|---:|---:|
| nccl                  | 0.070 ± 0.002 | 0.101 ± 0.002 | 0.147 ± 0.003 | 0.251 ± 0.003 |
| nccl_graphs           | 0.098 ± 0.021 | 0.114 ± 0.031 | 0.146 ± 0.004 | 0.237 ± 0.021 |
| nvshmem_block_nbsync  | 0.083 ± 0.002 | 0.117 ± 0.003 | 0.174 ± 0.001 | 0.287 ± 0.001 |

## Headline (16384², niter=1000, 2 nodes × 4 GPU)

| Variant | H200 (this run) | H100 (original) |
|---|---:|---:|
| nccl                            | **0.254 s** | 0.266 s |
| nccl_graphs                     | **0.233 s** | (not in original) |
| nvshmem (no -use_block_comm)    | **5.740 s** | 4.222 s — *20× slowdown without `-use_block_comm`!* |
| nvshmem -use_block_comm -nbsync | **0.290 s** | 0.285 s |

## Conclusions (same shape as H100 run)

1. **nccl_graphs is fastest** at 0.233 s, marginally beating nccl (0.254 s).
2. **nvshmem without `-use_block_comm` is catastrophic** at 5.740 s
   (≈20× slower than nccl) — per-element `nvshmem_float_p` puts swamp the
   IB fabric. Adding `-use_block_comm` recovers performance to 0.302 s
   (using block-collective puts), and `-neighborhood_sync` on top trims
   another ~4% to 0.290 s — within ~25% of nccl_graphs.
3. **Sweep B2 nx scan**: at small nx (2048), nvshmem with block-comm
   (0.083 s) is actually slightly slower than nccl (0.070 s) — comm
   overhead dominates. As nx grows, the gap is consistent at ~10–15%.
4. **`nccl_graphs` shows higher stdev** (0.019 s on the headline, 0.031 at
   nx=4096) — captured-graph latency on first invocation is variable.

See [figures/](figures/) for 6 PNGs (paper Fig 4.19 mirrors).
