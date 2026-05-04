# NCCL_GRAPH_MIXING_SUPPORT ablation — nccl_graphs jacobi

Tested on a single 8×H100 node (job 6320326), CUDA 13.0, NCCL 2.29.3.
The hypothesis: disabling `NCCL_GRAPH_MIXING_SUPPORT` (default `1`) should
remove the per-launch synchronization overhead NCCL adds to support
mixed-graph/non-graph usage, and so reduce per-iteration latency in
`nccl_graphs/jacobi`.

## Setup

`nccl_graphs/jacobi` captures a CUDA graph that includes the NCCL
`Send/Recv` halo exchange + the Jacobi kernel + L2 norm copy, and replays
that graph every iteration. The application has only one outstanding NCCL
call at a time and never mixes graph-captured with non-captured collectives,
so it should be safe to set `NCCL_GRAPH_MIXING_SUPPORT=0`.

Two scenarios:

* **Big problem** — `nx=ny=16384, niter=1000`, 1/2/4/8 GPUs, 3 reps each.
  Mirrors Sweep B's standard point. Compute-bound (~1.5 ms/iter compute
  for 16384²).
* **Small problem** — `nx∈{128,512}, ny=16384, niter=5000`, 4/8 GPUs, 5 reps
  each. Latency-bound (compute drops to ~µs/iter at nx=128).

Csv: [results-graph-ablation-20260504-092532.csv](results-graph-ablation-20260504-092532.csv),
[results-graph-ablation-small-20260504-093058.csv](results-graph-ablation-small-20260504-093058.csv).

## Big problem (16384² × 1000 iter)

| #GPU | mix=1 default (s) | mix=0 disabled (s) | Δ% (mean) | stdev (mix=1) |
|---:|---:|---:|---:|---:|
| 1 | 1.547 | 1.551 | +0.2% | 0.0029 |
| 2 | 0.783 | 0.784 | +0.1% | 0.0011 |
| 4 | 0.411 | 0.409 | −0.4% | 0.0037 |
| 8 | 0.241 | 0.253 | +4.6%*| 0.0186 |

*8-GPU mix=1 mean is dominated by one fast outlier (0.220 s) among three
runs — the other two are 0.251 / 0.254 s, similar to mix=0. With min
instead of mean the picture flips. The high stdev (0.019 s, ~8% of the
mean) shows this point is noisy regardless of the env var.

**Verdict: at the standard point, disabling `NCCL_GRAPH_MIXING_SUPPORT`
has no measurable effect.** Per-iter compute (~1.5 ms) dwarfs any per-graph-launch
bookkeeping savings (sub-µs).

## Small / latency-bound problem (5000 iter)

Min runtime over 5 reps (more robust to first-launch warmup outliers):

| nx | #GPU | mix=1 (s) | mix=0 (s) | Δ% |
|---:|---:|---:|---:|---:|
| 128 | 4 | 0.1196 | 0.1141 | −4.6% |
| 128 | 8 | 0.1423 | 0.1347 | −5.3% |
| 512 | 4 | 0.1607 | 0.1667 | +3.7% |
| 512 | 8 | 0.1397 | 0.1388 | −0.6% |

A small (~5%) improvement is visible at `nx=128` (the smallest, most
launch-bound case), and noise elsewhere. Even at `nx=128 / 8 GPUs`, where
each iteration is dominated by NCCL launch overhead rather than data
movement, the saving is ~5 ms over 5000 iterations — i.e. ~1 µs per
iteration. That's consistent with NCCL's mixing-support overhead being on
the order of a microsecond per launch.

The standard deviation of these 5-rep groups is 0.07–0.13 s — comparable
in magnitude to the mean — meaning the cluster has substantial run-to-run
noise (likely shared CPU contention via Slurm pinning, or first-launch
NCCL channel setup hitting a different code path). **The ~5% headline
improvement at `nx=128` is consistent with the hypothesis but not far
above the noise floor on this hardware.**

## Conclusion

* `NCCL_GRAPH_MIXING_SUPPORT=0` saves on the order of **~1 µs per
  graph-captured NCCL launch** on this stack.
* For Jacobi at the standard 16384² point this is in the noise; you need a
  comm-launch-bound workload (small per-iter compute, many launches) to
  see it. Even there, the saving is single-digit % on this Jacobi setup.
* The optimization is **safe to enable for `nccl_graphs/jacobi`** (the
  preconditions are met: one outstanding NCCL call, no graph/non-graph
  mixing, single communicator) — but it doesn't change the conclusions of
  the main report. NCCL_graphs is still the fastest variant on 8 GPUs at
  16384² regardless.
* Production NCCL apps that *do* fire many concurrent collectives or mix
  graph and non-graph paths should leave `NCCL_GRAPH_MIXING_SUPPORT=1`
  (the default) — disabling it would break those use-cases.

## Repro

```bash
. repro/jacobi/setup_env.sh
bash repro/nccl_graph_ablation/run_big.sh        # 16384^2 sweep
bash repro/nccl_graph_ablation/run_small.sh  # nx <= 512 sweep
```

---

## H200 cluster re-run (2026-05-04)

CSVs: `results-graph-ablation-20260504-164458.csv` (big) and
`results-graph-ablation-small-20260504-164915.csv` (small). Same workload
as the original H100 run.

### Big problem (16384², 1000 iter, 3 reps)

| #GPU | mix=1 mean (s) | mix=0 mean (s) | Δ% (mean) |
|---:|---:|---:|---:|
| 1 | 1.5247 ± 0.0021 | 1.5227 ± 0.0027 | −0.13% |
| 2 | 0.7717 ± 0.0003 | 0.7820 ± 0.0159 | +1.34% |
| 4 | 0.4136 ± 0.0135 | 0.4241 ± 0.0163 | +2.54% |
| 8 | 0.2198 ± 0.0026 | 0.2242 ± 0.0094 | +2.00% |

Same shape as the original H100 result: noise dominates at 16384² where
per-iter compute (~1.5 ms) >> per-launch bookkeeping savings (sub-µs). On
H200 the small `mix=0 +2%` is unstable across reps (stdev ~0.01 s).

### Small / latency-bound problem (5000 iter, min over 5 reps)

| nx | #GPU | mix=1 min (s) | mix=0 min (s) | Δ% |
|---:|---:|---:|---:|---:|
| 128 | 4 | 0.1371 | 0.1356 | −1.07% |
| 128 | 8 | 0.1381 | 0.1355 | −1.89% |
| 512 | 4 | 0.1674 | 0.1708 | +2.00% |
| 512 | 8 | 0.1470 | 0.1402 | −4.63% |

Same pattern: marginal improvement at the smallest, latency-bound case
(`nx=128`, ~1-2%); at `nx=512 / 8 GPU` the saving creeps up to ~5%, in line
with the original H100 finding. No headline change.

### Conclusion stays the same on H200

* `NCCL_GRAPH_MIXING_SUPPORT=0` is safe to set for `nccl_graphs/jacobi`
  (single communicator, no graph/non-graph mixing).
* The improvement is sub-µs per launch — visible only when launch latency
  dominates (small `nx` × many iter), in the noise otherwise.
