# Thesis Chapter 4 micro-benchmarks — H200 reproduction

This directory reproduces the NVSHMEM micro-benchmarks from the thesis Chapter 4
(`../../ETH_Zürich_CADMO_Thesis_Template_v2.pdf`):

| Section | Title | What we ran |
|---|---|---|
| 4.1.1 | Point-to-point primitives | `shmem_{g,get,p,put,st,atomic}_bw` + `*_ping_pong_latency` over message-size, TPB, and CTA sweeps, intranode (1×2 NVL) and internode (2×1 IB) |
| 4.1.2 | Collective primitives | `{alltoall,bcast,fcollect,reduction,reducescatter}_latency` over message-size + a 2/4/8-rank scaling pass at 64 KiB, intranode (1×8) and internode (2×8 = 16 ranks) |
| 4.2.1 | NCCL vs NVSHMEM | NCCL `all_reduce_perf` and `alltoall_perf` (nccl-tests built with MPI=1) at the same scenarios, alongside the NVSHMEM 4.1.2 numbers |

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
│   ├── run_4_1_2_coll.sh          ← 5 collectives + rank-scaling
│   └── run_4_2_1_nccl.sh          ← NCCL all_reduce_perf + alltoall_perf
├── plot_results.py                ← parse perftest + nccl-tests logs → 11 figures
└── results/
    ├── *.log                      ← per-config raw output (~200 files)
    └── figures/                   ← 11 PNG plots
```

## Headline numbers (H200, NVSHMEM 3.3.9-ibp)

| Test | Intranode (1×2 NVL) | Internode (2×1 IB) |
|---|---|---|
| `shmem_put_bw` peak (4 MiB, 32 CTAs × 256 TPB) | ~310 GB/s NVLink | ~47 GB/s ConnectX-7 |
| `shmem_get_bw` peak | ~150 GB/s | ~42 GB/s |
| `shmem_atomic_bw` peak | ~280 GB/s | (atomics intra only) |
| `shmem_put_ping_pong_latency` (4 B) | ~3 µs | ~14 µs |

| Collective (16-rank 2×8) | NVSHMEM block-scope | NCCL (-g 1, MPI=1) |
|---|---|---|
| `alltoall` @ 64 KiB | ~28 µs | ~38 µs |
| `alltoall` @ 4 MiB | ~140 µs | ~110 µs |
| `allreduce/reduction` @ 1 KiB | ~120 µs | ~25 µs |

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
