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
ConnectX-7 IB cluster, container `gpu_882f6e72.sqsh`, NVSHMEM **3.3.9-ibp**
(internal fork at `~/workspace/nvshmem` on branch `3.3.9-ibp`, which adds
`ibp*` device-name detection on top of upstream 3.3.9).

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
