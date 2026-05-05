# Thesis Chapter 4 micro-benchmarks — H200 reproduction

This directory reproduces the NVSHMEM micro-benchmarks from the thesis Chapter 4
(`../../ETH_Zürich_CADMO_Thesis_Template_v2.pdf`):

| Section | Title | What we ran |
|---|---|---|
| 4.1.1 | Point-to-point primitives | `shmem_{g,get,p,put,st,atomic}_bw` + `*_ping_pong_latency` over message-size, TPB, and CTA sweeps, intranode (1×2 NVL) and internode (2×1 IB) |
| 4.1.2 | Collective primitives | `{alltoall,bcast,fcollect,reduction,reducescatter}_latency` over message-size + a 2/4/8-rank scaling pass at 64 KiB, intranode (1×8) and internode (2×8 = 16 ranks) |
| 4.2.1 | NCCL vs NVSHMEM | All 5 nccl-tests collectives + NVSHMEM **device** kernel + NVSHMEM **host on_stream** for the same 5 ops, intra (1×8) and inter (2×8) |

## API mapping — what each line in the plots is actually doing

NVSHMEM has 3 layers per primitive: device (kernel-initiated, picks per-thread / warp / block scope), host on_stream (CPU enqueues a stream op), and host blocking (we did not run). Within device kernels, the perftest binaries iterate over scopes and datatypes; the plot script picks ONE configuration per binary so each curve is comparable.

### Point-to-point (`shmem_*_bw` and `shmem_*_ping_pong_latency` under `device/pt-to-pt/`)

| Plot label | Binary | What the kernel does |
|---|---|---|
| `g` | `shmem_g_bw` | `nvshmem_<int>_g`: per-thread scalar **get**. One word per thread. |
| `get` | `shmem_get_bw` | `nvshmem_<int>_get`: **block-cooperative** bulk get; all threads in the block participate. |
| `p` | `shmem_p_bw` | `nvshmem_<int>_p`: per-thread scalar **put**. One word per thread. |
| `put` | `shmem_put_bw` | `nvshmem_<int>_put`: **block-cooperative** bulk put. |
| `st` | `shmem_st_bw` | Direct CUDA store via mapped peer pointer (NVL only — no IB equivalent). |
| `atomic` | `shmem_atomic_bw` | `nvshmem_<int>_atomic_inc`: block-coop atomic increment. |

All `_bw` runs use 32 CTAs × 256 TPB by default; the TPB and CTA sweeps vary one of those at fixed 1 MiB.

### Collectives — what each `_latency` binary contains

Each device collective binary (`device/coll/<op>_latency`) loops over scopes and types and emits a separate sub-table per (type, scope). The plot script filters to a single config so each plot has one curve per binary.

| Op | NVSHMEM device binary | NVSHMEM host on_stream binary | NCCL-tests binary | NVSHMEM scope chosen for plots | NVSHMEM auto-tuned ALGO |
|---|---|---|---|---|---|
| alltoall | `alltoall_latency` | `alltoall_on_stream` | `alltoall_perf` | block, 32-bit | (only one alltoall algo: flag-alltoall) |
| broadcast | `bcast_latency` | `broadcast_on_stream` | `broadcast_perf` | block, 32-bit | `BCAST_ALGO=0` → autotuner; default tree (`BCAST_TREE_KVAL=2`) |
| allgather | `fcollect_latency` | `fcollect_on_stream` | `all_gather_perf` | block, 32-bit | `FCOLLECT_ALGO=0` → autotuner; ring is the default in 3.x |
| allreduce | `reduction_latency` | `reduction_on_stream` | `all_reduce_perf` | thread, int32+sum | `REDUCE_ALGO=0` → autotuner; below `REDUCE_NVLS_THRESHOLD=2 KiB` uses one-shot, above uses two-shot |
| reducescatter | `reducescatter_latency` | `reducescatter_on_stream` | `reduce_scatter_perf` | thread, int32+sum | `REDUCESCATTER_ALGO=0` → autotuner |

The reduction/reducescatter binaries in this build only emit **thread** scope rows (no warp/block). The non-reduction collectives emit thread/warp/block × 32-bit/64-bit; we filter to (block, 32-bit) for one curve.

### Verifying NVSHMEM does NOT fall back to NCCL

Our NVSHMEM build was configured with `NVSHMEM_USE_NCCL=OFF`. We verified at runtime:

```
$ nm -D $INST/lib/libnvshmem_host.so | grep -ci nccl    # → 0
$ ldd $INST/lib/libnvshmem_host.so | grep -i nccl       # → (no output)
$ NVSHMEM_DEBUG=INFO ... 2>&1 | grep ALGO
NVSHMEM INFO ALGO: BCAST_ALGO set to 0
NVSHMEM INFO ALGO: FCOLLECT_ALGO set to 0
NVSHMEM INFO ALGO: REDUCE_ALGO set to 0 (0 -> 1)
NVSHMEM INFO ALGO: REDUCESCATTER_ALGO set to 0
```

So every "NVSHMEM device" / "NVSHMEM host" line in the plots is pure NVSHMEM (its own tree/ring/flat algorithms), not NCCL.

The thesis ran on Alps GH200 + Cray Slingshot. We ran on a CoreWeave H200 +
ConnectX-7 IB cluster, container `gpu_882f6e72.sqsh`.

### Versions

| Library | Version | Source |
|---|---|---|
| **NVSHMEM** | **3.3.9-ibp** | internal fork `~/workspace/nvshmem`, commit `4bc54ac` (= upstream `8a43de2 NVSHMEM 3.3.9` + `4bc54ac Detect ibp devices`). Built with `NVSHMEM_USE_NCCL=OFF`, `NVSHMEM_IBGDA_SUPPORT=ON`, `NVSHMEM_IBRC_SUPPORT=ON`, `NVSHMEM_NVLS_SUPPORT=ON` (default). |
| **NCCL** | **2.30.4+cuda13.2** | pip wheel `nvidia-nccl-cu13==2.30.4` at `~/workspace/nccl-pip/nvidia/nccl/`. |
| **nccl-tests** | upstream HEAD on `2026-05-04` | built with `MPI=1 MPI_HOME=/usr/local/mpi NCCL_HOME=<above>` against the wheel above. |
| **CUDA** | 13.0.88 | container's `/usr/local/cuda` |
| **PyTorch** | (not used for these benches) | |

### What's NVLS / what's "symmetric kernel"?

H200 has hardware **NVLink-SHARP multicast** ("NVLS"). Both NCCL and NVSHMEM detect this and
will use multicast paths for some collectives by default; we verified at runtime:

```
NCCL INFO NVLS multicast support is available on dev 0..7
NCCL INFO NVLS tuning: nChannels 16 chunkSize 131072 treeMaxChunkSize 131072
```

NCCL 2.30 also has a separate "**symmetric memory**" code path. When NCCL's communicator
exposes a symmetric memory window (similar to NVSHMEM's symmetric heap), tasks can be
converted into symmetric kernels (`ncclSymmetricTaskScheduler`, `[Symmetric]` log tag)
that bypass the legacy ring/tree dispatch entirely. nccl-tests by default uses
`cudaMalloc`'d buffers (no window registration), so symmetric kernels are off — unless
you set **`NCCL_SYM_NOWIN_ENABLE=1`**, which lets NCCL convert non-window buffers to
symmetric tasks at runtime.

The `compare_*.png` figures show six series so the four configurations stand on their own:

| Series | Env | What it is |
|---|---|---|
| NVSHMEM device (default) | `NVSHMEM_DISABLE_NVLS=0` | NVSHMEM kernel-initiated coll, NVLS allowed |
| NVSHMEM device (NVLS off) | `NVSHMEM_DISABLE_NVLS=1` | Same kernel, NVLS multicast disabled |
| NVSHMEM host on_stream | (default) | CPU-initiated stream-ordered coll, NVLS allowed |
| NCCL default | (defaults: `NVLS_ENABLE=1`, `SYM_NOWIN_ENABLE=0`) | Auto-tuner with NVLS available, no sym kernel |
| NCCL sym | `NCCL_SYM_NOWIN_ENABLE=1` | Same, but auto-promote cudaMalloc buffers to symmetric tasks |
| NCCL no-NVLS | `NCCL_NVLS_ENABLE=0` | Auto-tuner without NVLS multicast |

Run scripts: `run_4_2_1_sym.sh`, `run_4_2_1_nvls.sh`, `run_4_1_2_nvshmem_nvls.sh`.

## Reproduce

```bash
# 1. Get a 2-node H200 dev allocation (qos=dev caps at 16 GPU = 2 nodes here):
unset SLURM_JOB_ID
srun -N 2 --cpus-per-gpu 16 --gpus-per-node 8 --time 3:00:00 \
     --partition h200 --qos=dev --exclusive --mem 0 \
     --container-image /mnt/vast/containers/gpu_882f6e72.sqsh \
     --container-workdir /mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models \
     --container-env HOME --container-remap-root \
     --container-mounts /mnt/vast:/mnt/vast --pty /bin/bash
# (Note the JOBID; you'll use it in steps below.)

# 2. (Inside the container) install build deps + create libmlx5.so symlink + build NVSHMEM perftest.
#    The NVSHMEM IBGDA transport links against libmlx5; the container ships libmlx5.so.1
#    but no unsuffixed .so symlink, and is missing libopenmpi-dev headers.
apt-get install -y libopenmpi-dev slurm-client
ln -sf /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so
git -C /mnt/vast/home/tiancheng.chen/workspace/nvshmem worktree add \
    /mnt/vast/home/tiancheng.chen/workspace/nvshmem-3.3.9-ibp 3.3.9-ibp
bash repro/thesis_microbench/scripts/build_nvshmem.sh   # ~10 min
bash /mnt/vast/home/tiancheng.chen/workspace/nvshmem-3.3.9-ibp/scripts/install_hydra.sh \
     /tmp/hydra_src /mnt/vast/home/tiancheng.chen/workspace/nvshmem-3.3.9-ibp-build/install

# 3. (From the LOGIN NODE) prime a named pyxis container on both nodes,
#    then run the three sweeps.  Each call below takes the JOBID + CNAME.
JOBID=<from step 1>
CNAME=thmicro_$JOBID
INST=/mnt/vast/home/tiancheng.chen/workspace/nvshmem-3.3.9-ibp-build/install
CONT=/mnt/vast/containers/gpu_882f6e72.sqsh

srun --jobid=$JOBID --overlap \
     --container-image=$CONT --container-name=$CNAME \
     --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm \
     --container-remap-root --container-env=HOME \
     -N 2 --ntasks-per-node=1 hostname    # primes container

JOBID=$JOBID CNAME=$CNAME INST=$INST bash repro/thesis_microbench/scripts/run_4_1_1_p2p.sh   # ~25 min
JOBID=$JOBID CNAME=$CNAME INST=$INST bash repro/thesis_microbench/scripts/run_4_1_2_coll.sh  # ~10 min

# nccl-tests need to be built with MPI=1 against NCCL ≥ 2.30.4:
cd nccl-tests && make clean && \
   MPI=1 MPI_HOME=/usr/local/mpi NCCL_HOME=/mnt/vast/home/tiancheng.chen/workspace/nccl-pip/nvidia/nccl \
   CUDA_HOME=/usr/local/cuda make -j 8 NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
JOBID=$JOBID CNAME=$CNAME INST=$INST bash repro/thesis_microbench/scripts/run_4_2_1_nccl.sh  # ~5 min

# 4. Plot.
python3 repro/thesis_microbench/plot_results.py
```

## Launcher details (the parts that took the most debugging)

* **MLX5 symlink.** NVSHMEM cmake's `find_library(MLX5_lib mlx5)` looks for
  `libmlx5.so` (no version suffix) but the container only ships `libmlx5.so.1`.
* **MPI build of NVSHMEM is broken on this container.** `OPAL_PREFIX=/opt/hpcx/ompi`
  is set in the container env, but `find_package(MPI)` produces malformed `-I`
  flags from it (it concatenates the source-tree path with the hpcx prefix).
  We build with `NVSHMEM_MPI_SUPPORT=0` and rely on the **PMI** bootstrap plugin
  instead. `pmi`, `pmi2`, and `uid` plugins are all built and installed.
* **Bootstrap selection.** NVSHMEM picks the bootstrap plugin via two env vars:
    * `NVSHMEM_BOOTSTRAP=PMI`   (mode)
    * `NVSHMEM_BOOTSTRAP_PMI=PMI2`  (sub-flavor — needed when the slurm
       `--mpi=pmi2` provides a PMI-2 server; default would load the PMI-1 plugin
       and silently mismatch). For Hydra-launched (single-node) runs, default
       PMI-1 is fine.
* **Why `--mpi=pmi2` was hanging.** Initially the launches hung indefinitely
  with no output. The fix was the bootstrap-flavor env vars above — the previous
  attempts were loading the PMI-1 plugin against slurm's PMI-2 server, which
  doesn't error, just hangs in the rendezvous.
* **Hydra needs a launcher.** `nvshmrun.hydra` defaults to `-launcher slurm`,
  which runs `srun` from inside the container — and `srun` isn't in the container
  PATH unless you `apt-get install -y slurm-client` and also bind-mount
  `/etc/slurm` from the host. For single-node sanity tests use `-launcher fork`.
* **slurm.conf bind-mount.** All cross-node srun calls use
  `--container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm` so the slurm
  client inside the container can reach the controller config.

## Files

```
repro/thesis_microbench/
├── README.md                      ← this file
├── scripts/
│   ├── build_nvshmem.sh           ← cmake build of NVSHMEM 3.3.9-ibp + perftest
│   ├── run_4_1_1_p2p.sh           ← 6 P2P APIs × 3 sweeps × 2 scenarios
│   ├── run_4_1_2_coll.sh          ← 5 NVSHMEM device collectives + rank-scaling
│   ├── run_4_2_1_nccl.sh          ← initial NCCL all_reduce_perf + alltoall_perf
│   └── run_4_2_1_extended.sh      ← all 5 NCCL collectives + 5 NVSHMEM host on_stream
├── plot_results.py                ← parse perftest + nccl-tests logs → 16 figures
└── results/
    ├── *.log                      ← per-config raw output (~220 files)
    └── figures/                   ← 16 PNG plots
        ├── p2p_*.png              ← 7 P2P plots (msg-size/TPB/CTA × intra/inter + ping-pong)
        ├── coll_msgsize_*.png     ← 2 (intra 1×8, inter 2×8)
        ├── coll_rank_scaling.png  ← rank-count scaling
        ├── nccl_vs_nvshmem.png    ← combined alltoall+allreduce summary
        └── compare_*.png          ← 5 per-collective comparison plots
                                     (alltoall, allreduce/sum, broadcast, allgather/fcollect, reducescatter/sum)
```

## Headline numbers (H200, NVSHMEM 3.3.9-ibp)

### P2P (intra = 1×2 NVL, inter = 2×1 IB IBGDA)

| Test | Intra | Inter |
|---|---|---|
| `shmem_put_bw` peak (4 MiB, 32 CTAs × 256 TPB) | ~310 GB/s NVLink | ~47 GB/s ConnectX-7 |
| `shmem_get_bw` peak | ~150 GB/s | ~42 GB/s |
| `shmem_atomic_bw` peak | ~280 GB/s | (atomics intra only) |
| `shmem_put_ping_pong_latency` (4 B) | ~3 µs | ~14 µs |

### Collectives — NVSHMEM device kernel vs NVSHMEM host on_stream vs NCCL

All numbers are out-of-place latency at 64 KiB / 1 MiB; intranode = 8 ranks on 1 node, internode = 16 ranks on 2×8.

| Op | Intra @ 64 KiB | Inter @ 64 KiB | Intra @ 1 MiB | Inter @ 1 MiB |
|---|---|---|---|---|
| **alltoall** — NVSHMEM device kernel | ~9 µs | ~28 µs | ~110 µs | ~250 µs |
| **alltoall** — NVSHMEM host on_stream | ~30 µs | ~33 µs | ~80 µs | ~95 µs |
| **alltoall** — NCCL | ~30 µs | ~30 µs | ~50 µs | ~58 µs |
| **broadcast** — NVSHMEM device | ~5 µs | ~5 µs | ~110 µs | ~470 µs |
| **broadcast** — NVSHMEM host on_stream | ~30 µs | ~33 µs | ~50 µs | ~80 µs |
| **broadcast** — NCCL | ~25 µs | ~25 µs | ~30 µs | ~35 µs |
| **allgather/fcollect** — NVSHMEM device | ~5 µs | ~14 µs | (binary cap) | ~70 µs |
| **allgather** — NCCL | ~25 µs | ~28 µs | ~30 µs | ~50 µs |
| **allreduce** — NVSHMEM device | ~25 µs (1 KiB max) | ~120 µs (1 KiB max) | (cap) | (cap) |
| **allreduce** — NVSHMEM host on_stream | ~25 µs | ~75 µs | ~150 µs | ~600 µs |
| **allreduce** — NCCL | ~10 µs | ~25 µs | ~50 µs | ~110 µs |
| **reducescatter** — NVSHMEM device | (cap) | ~60 µs | (cap) | (cap) |
| **reducescatter** — NCCL | ~30 µs | ~30 µs | ~30 µs | ~50 µs |

(Numbers eyeballed off the per-collective `compare_*.png` figures; raw values in `results/*.log`.)

**Takeaways**:

* **Small-message latency** (≤ 16 KiB intra, ≤ 4 KiB inter): NVSHMEM **device** kernel wins by 2–5× for alltoall/broadcast/fcollect because the kernel never crosses the host launch boundary. This is the fast path NVSHMEM is designed for.
* **Large-message bandwidth** (≥ 1 MiB): NCCL wins for ring/tree-friendly ops (allreduce, broadcast, alltoall inter) thanks to its tuned multi-stage ring + chunked schedule. NVSHMEM device collectives use simpler one-shot algorithms and do not chunk.
* **NVSHMEM host on_stream** sits between the two: same algorithms as the device kernel but with a CPU launch per call, so it pays ~25 µs floor like NCCL but doesn't have NCCL's tuned algorithms.
* **`allreduce`** stops at 1 KiB for the device binary because reduction's perftest is sized that way (we can re-run with `-e` to extend).

### NCCL autotuner internals (and where sym kernel / NVLS fit)

`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,TUNING` reveals the model. At `ncclCommInit`, NCCL builds a static **(7 algorithm × 3 protocol) cost table** per collective, where each cell is `(latency_us / bandwidth_GBps)` from a topology probe + calibration constants.

For our 8-GPU H200 default config (raw output in `results/probe_autotuner_default-newcluster-20260504.log`), the AllReduce row of the table is:

```
Algorithm   |        Tree         |        Ring         |     CollNetDirect    |     CollNetChain     |        NVLS          |       NVLSTree       |        PAT
Protocol    |  LL  | LL128 | Simple|  LL  | LL128 | Simple|  LL  | LL128 | Simple|  LL  | LL128 | Simple|  LL  | LL128 | Simple|  LL  | LL128 | Simple|  LL  | LL128 | Simple
AllReduce   |15.2/43.6|31.5/128.8|64.4/165.6|15.0/80.6|40.6/189.3|56.0/205.7|5.6/0|5.6/0|44.0/0|0/0|0/0|69.2/0|0/0|0/0|25.0/272.0|0/0|0/0|25.0/0|0/0|0/0|0/0
```

Per call, the autotuner picks the (algo, proto) that minimizes `latency + size/bandwidth` for the message size. Verified picks for our 8-GPU AllReduce default run:

| Size | Picked |
|---|---|
| ≤ 32 KiB | **`Algo RING proto LL`** (lowest latency floor: 15 µs at 0 BW) |
| 256 KiB | `Algo RING proto LL` (still latency-dominant) |
| 1 MiB | `Algo RING proto LL` (BW: 80.6 GB/s on 23 channels) |
| ≥ 2 MiB | **`Algo NVLS proto SIMPLE`** (272 GB/s, accepts 25 µs floor) |

Crossover at ~1 MiB.

#### `NCCL_NVLS_ENABLE=0`: removes one option from the table

NCCL_NVLS_ENABLE=0 zeros out the NVLS row. Autotuner picks among the rest. Per-call evidence (probe `_nvls_off`):
```
AllReduce: 1024 Bytes -> Algo RING proto LL          (same as default)
AllReduce: 33554432 Bytes -> Algo RING proto LL128   (was: NVLS Simple. Now best left = RING LL128)
```
Result: large messages get RING LL128 (189 GB/s) instead of NVLS Simple (272 GB/s) → 200 GB/s vs 209 GB/s end-to-end at 32 MiB. Just an option removal.

#### `NCCL_SYM_NOWIN_ENABLE=1`: a separate scheduler that bypasses the autotuner — **but doesn't always engage**

NCCL 2.30 has `ncclSymmetricTaskScheduler` which runs **before** the regular autotuner. If a task qualifies (registered symmetric window OR `NCCL_SYM_NOWIN_ENABLE=1` plus other internal eligibility checks), it dispatches a `Kernel ncclSymk*` and never enters the autotuner table. Logs as `[Symmetric]: <bytes> -> Kernel <name>` instead of `Algo X proto Y`.

In our config, **`NCCL_SYM_NOWIN_ENABLE=1` does NOT actually engage the symmetric scheduler** for nccl-tests's cudaMalloc'd buffers — every call still logs `Algo RING proto LL`. Even with `NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=COLL`, no `[Symmetric]` lines fire. Probably the eligibility check (cuMem support, NIC fusion, GIN, etc.) rejects this combination on this cluster. Re-running back-to-back, sym=0 vs sym=1 are **identical within noise** (~40 µs floor at 4 B – 8 KiB):

```
size  sym=0   sym=1
4     40.4    41.1
16    39.2    39.8
64    40.1    41.1
256   47.4    40.5
1024  39.5    40.2
4096  41.3    41.5
```

(An earlier comparison reported a 1.2× speedup for sym=1 at small sizes; that was warm-cache noise from running sym=0 first then sym=1. Disregard.)

**TL;DR** for the user's mental model:
* **NVLS** *is* "just one more entry in the autotuner's table". `NCCL_NVLS_ENABLE=0/1` adds/removes the NVLS row. Autotuner picks per call.
* **Symmetric kernel** is *not* in the autotuner. It's a parallel scheduler that, if eligible, replaces the entire (algo, proto) dispatch with a `ncclSymk*` kernel. In our build/cluster the eligibility never triggered for cudaMalloc'd buffers; you'd need to call `ncclMemAlloc` + `ncclWindowAlloc` to actually exercise the sym path.

### NVLS (NCCL_NVLS_ENABLE / NVSHMEM_DISABLE_NVLS) impact

NCCL on H200 (intra 8 GPU):

| Op | NVLS off (NCCL_NVLS_ENABLE=0) | NVLS on (default) | NVLS speedup |
|---|---|---|---|
| broadcast 1 KiB | 40.0 µs | 32.2 µs | **1.24×** |
| broadcast 1 MiB | 45.9 µs | 38.5 µs | **1.19×** |
| broadcast 32 MiB | 127 µs | 127 µs | ~1.00× (BW frontier) |
| all_reduce 1 KiB | 34.6 µs | 40.6 µs | 0.85× ← NVLS slower! |
| all_reduce 1 MiB | 41.3 µs | 48.1 µs | 0.86× |
| all_reduce 32 MiB | 209 µs | 200 µs | 1.05× |

So **NVLS helps broadcast a lot, but the NCCL auto-tuner picks a sub-optimal NVLS-allreduce path for 8-GPU H200 at small-to-medium sizes** — the LL/Simple ring path is faster there. At 32 MiB NVLS catches up. Worth knowing if you're tuning a model that does lots of small allreduces.

NVSHMEM device collectives in our build are essentially **insensitive to NVSHMEM_DISABLE_NVLS** for `bcast` at block scope (1.0× across all sizes) — meaning the NVSHMEM `bcast_latency` device binary is *not* picking up NVLS multicast even when it's available. The kernel uses a tree-based broadcast that doesn't engage the multicast hardware. For `reduction` and `alltoall` the same is true (within noise). This is consistent with NVSHMEM 3.3.9 only enabling NVLS for specific code paths (`REDUCE_NVLS_THRESHOLD` for one-shot allreduce, `FCOLLECT_NVLS_THRESHOLD` for fcollect) that the perftest binaries don't always exercise.

## Caveats

* Reduction/reducescatter perftests in this build only emit **thread-scope** rows,
  while alltoall/bcast/fcollect emit `thread/warp/block`. The plot script picks
  `block` for the latter and `t` for the former. Don't compare across scopes.
* NVSHMEM `shmem_st_bw` and `shmem_atomic_bw` cross-node: not all stores/atomics
  have an IBGDA fast-path; the inter-node BW for these is much lower than NVLink
  intra (or absent — see `p2p_msgsize_inter.png`).
* NCCL-tests built without MPI bootstrap (the first attempt) runs each rank as
  an independent communicator and emits 16 separate "Rank 0 alone" runs in one
  log. The numbers in `results/*.log` use the MPI=1 build, which gives a single
  16-rank world.
* `dev` qos on this cluster caps at 16 GPU = 2 nodes; `research`/`scavengers`
  could go higher but are preemptible. The thesis Alps run goes well past 2 nodes;
  to reproduce that scaling, drop the `--qos=dev` constraint.
