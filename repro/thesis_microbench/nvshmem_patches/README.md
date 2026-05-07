# NVSHMEM perftest patch — reduction_focus

`reduction_focus.cu` is a stripped-down variant of the upstream NVSHMEM
[`perftest/device/coll/reduction_latency.cu`](https://github.com/NVIDIA/nvshmem/blob/main/perftest/device/coll/reduction_latency.cu)
that does ONLY `float-sum-block` (no inner dtype × redop × scope iteration).

## Why

Stock `reduction_latency` always loops 2 dtypes (int32, int64) × 7 redops
× 3 scopes (thread, warp, block) per size. Two consequences for our
`all_reduce` comparison against NCCL:

1. **Cross-node sweeps time out.** The thread + warp pre-pass eats > 5 min
   of slurmstep timeout before block scope ever starts.
2. **Integer-only inhibits NVLS.** NCCL's `all_reduce(float, sum)` is
   eligible for the NVLink-SHARP / `LDMC/STMC` multicast path on H200,
   but stock `reduction_latency` only emits int32 / int64 rows, so the
   comparison was apples-to-oranges. The real fair comparison is
   `nvshmemx_float_sum_reduce_block` against `ncclAllReduce(... float, ncclSum)`.

`reduction_focus.cu` does the float-sum-block call directly. Bench finishes
in ~30 s intra (1 GiB max) / ~70 s inter (256 MiB max).

## How to build

Drop `reduction_focus.cu` + `reduction_focus.args` into
`nvshmem-<ver>/perftest/device/coll/`, add one line to that directory's
`CMakeLists.txt`:

```cmake
nvshmem_add_perftest(reduction_focus.cu)
```

then rebuild:

```bash
cmake --build $BUILD --target reduction_focus -j 8
cp $SRC/build/perftest/device/coll/reduction_focus $INSTALL/bin/perftest/device/coll/
```

## Diff vs upstream

See `reduction_focus.diff` for the exact changes against
`reduction_latency.cu`.
