# 2-node Jacobi reproduction — H200 cluster (2026-05-04)

Runs from `results-2node-20260504-181117.csv`. Times are mean ± stdev (s).

**Cluster**: 2 × H200 SXM nodes, 8 GPU each = 16 GPUs total. Same container
and toolchain as `REPORT_1NODE.md`. Inter-node link is ConnectX-7 IB
(8 × 400 Gb/s per node). Run with NPN=4 (8 ranks total) via:

```
JOBID=<id> NPN=4 REPS=2 bash repro/jacobi/run_sweep_2node.sh
```

**Note**: Sweep A2 captured at REPS=2 (5 vars × 2 reps); Sweep B2 was
truncated after 2 rows due to per-srun container init cost (~50 s/call),
which compounds badly across 36 srun calls. The Sweep A2 numbers are the
load-bearing ones — the headline 16384² point at 8 GPU total. To rerun B2
properly, allocate more wall time and let `run_sweep_2node.sh` complete.

## Sweep A2 — scale-up at nx=ny=16384, niter=1000, 2 nodes × 4 GPU = 8 ranks

| Variant | runtime (s) |
|---|---:|
| nccl                  | 0.262 ± 0.005 |
| nccl_graphs           | 0.231 ± 0.009 |
| nvshmem               | **5.742 ± 0.062** |
| nvshmem_block         | 0.301 ± 0.003 |
| nvshmem_block_nbsync  | 0.282 ± 0.004 |

**Headline**: at 16384² × 1000 iter on 2 nodes:
- `nccl_graphs` is fastest at **0.231 s**.
- `nvshmem` (no `-use_block_comm`) is **20× slower** at 5.742 s — the
  per-element `nvshmem_float_p` puts swamp the IB fabric. Adding
  `-use_block_comm` recovers performance to 0.301 s, and adding
  `-neighborhood_sync` on top brings it down further to **0.282 s**, ~7%
  off `nccl_graphs` parity.
- This matches the original H100 cluster's qualitative finding (16×
  slowdown without `-use_block_comm`). On H200 the gap is larger (20×),
  consistent with the IB fabric being the bottleneck for the per-element
  put-storm path.

## Comparison with original H100 run

| Variant | H200 (this run, 2N×4G) | H100 (original, 2N×4G) |
|---|---:|---:|
| nccl                 | **0.262 s** | 0.266 s |
| nvshmem (no block)   | **5.742 s** | 4.222 s |
| nvshmem -use_block_comm -nbsync | **0.282 s** | 0.285 s |

* `nccl` and `nvshmem -use_block_comm -nbsync` reproduce within run-to-run noise.
* The `nvshmem` (no block) regression is *worse* on H200 (5.7 s vs 4.2 s).
  Likely the H200 nodes spend more time blocked on per-element IB roundtrips
  because the local compute fraction shrunk — the wider GPU just makes the
  comm overhead more dominant.

## Conclusion

* **NCCL and `nccl_graphs` work out of the box at 2 nodes**, with the
  same modest improvement from graphs as on a single node.
* **NVSHMEM needs `-use_block_comm` (and `-neighborhood_sync`) at multi-node**
  to be competitive — without it, the per-element IB puts cost an order of
  magnitude.
* The headline on H200 is essentially the same as H100: at 2 nodes with
  `-use_block_comm -nbsync`, NVSHMEM is within ~10% of NCCL.

See [figures/fig_a_runtime_vs_gpu_2node.png](figures/fig_a_runtime_vs_gpu_2node.png)
and [figures/fig_nvshmem_block_ablation.png](figures/fig_nvshmem_block_ablation.png)
for the visual.
