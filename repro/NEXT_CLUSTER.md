# Moving to a different cluster — what to verify and what to bring

This work was done on a CoreWeave H100 cluster, in the `gpu_a4d0481d.sqsh` container shipped via the mistral repo's [`scripts/utils/cluster.py ggpus`](../../mistral/scripts/utils/cluster.py). A few things that block important measurements were specific to that environment. If you're switching clusters, validate the items below **before** sinking time into running the whole reproduction.

## What to validate up front (≈30 min)

### 1. NVSHMEM IBGDA actually works cross-node

This is the **single biggest blocker** in the current cluster — V1 (legacy) DeepEP low-latency and V1 internode HT both crash on NVSHMEM IBGDA bring-up or in the kernel-side RDMA write. Full diagnostic in [deepep/IBGDA_DEBUG.md](deepep/IBGDA_DEBUG.md). The minimum sanity check:

```bash
# On 2 nodes inside the container of choice. Replace with whatever launcher works.
srun --jobid=$JOBID --mpi=pmi2 -N 2 --ntasks-per-node=8 \
     /opt/nvshmem/bin/perftest/device/pt-to-pt/shmem_put_bw -d gpu
```

If this exits cleanly with bandwidth numbers, IBGDA is healthy and V1 DeepEP should run. If it fails the same way (`init failed for transport: IBGDA` followed by `Peer GPU N is not accessible` or `cudaErrorIllegalAddress` in-kernel), V1 LL/internode are still blocked and this cluster has the same fabric/firmware issue as the current one.

Other things to check on a fresh node (same as [deepep/probe_ibgda.sh](deepep/probe_ibgda.sh)):

* `ibv_devices` lists `mlx5_0` … `mlx5_N` (not just `ibpX` aliases)
* `/dev/gdrdrv` exists and is world-readable
* `lsmod | grep -E '^mlx5|^nvidia_peermem|^gdrdrv'` shows all three drivers loaded
* `mlxfwmanager --query` (if accessible) shows ConnectX-7 firmware ≥ 22.36.x
* Container has `CAP_NET_RAW` and `CAP_IPC_LOCK`, no IOMMU passthrough mode

### 2. Allocator can give you ≥ 2 nodes × 8 GPU at the same time

V1 `tests/legacy/test_internode.py` hardcodes `assert num_local_ranks == 8` and the C++ buffer hardcodes `LEGACY_NUM_MAX_NVL_PEERS == 8`. So **V1 internode HT** requires every node to expose all 8 GPUs to the test. On the current cluster this worked (`dev` qos passed `2 nodes × 8 GPU = 16`); on a tighter cluster confirm the qos and partition limits allow it.

### 3. Toolchain expectations

| Tool | Required minimum | What we hit on the current cluster |
|---|---|---|
| CUDA | 11.0 (Jacobi) / 12.3 (DeepEP V2) | 13.0 ✓ |
| NCCL (V2 DeepEP) | 2.30.4 | container had 2.29.3 → had to `pip install --no-deps "nvidia-nccl-cu13>=2.30.4"` |
| PyTorch (DeepEP V2) | 2.10 | 2.10.0a0 ✓ |
| OpenMPI | any CUDA-aware OR plain OpenMPI for bootstrap-only | 4.1.9a1 (NOT CUDA-aware, fine for the NCCL/NVSHMEM measurements but `mpi/jacobi` baseline is unfairly slow) |
| NVSHMEM | 0.4.1+ for `nvshmem/jacobi`, 3.x for DeepEP V2's NCCL/NVSHMEM coexistence | 3.4.5 ✓, but bootstrap MPI plugin was in `/usr/lib/x86_64-linux-gnu/nvshmem/13`, **not** `/opt/nvshmem/lib` — if your container puts them in the same place you can simplify `LD_LIBRARY_PATH` |
| C++ standard | C++17 (CUDA 13 CCCL requirement) | repo Makefiles default to C++14 → had to `sed -i 's/-std=c++14/-std=c++17/g'` |

### 4. Cluster overhead is reasonable

This cluster's `srun --container-image=...` cost ~52 s **per call**, regardless of payload (even a bare `srun hostname` was 44 s). This made the 2-node Jacobi sweep take ~40 min and forced the DeepEP multi-node tests to be 1-srun-per-test (no batching, since multiple sequential MPI sessions in one srun fail with SIGPIPE — first `MPI_Finalize` closes srun's PMI server).

If your new cluster's srun is fast (a few seconds), you can:

* Restore the larger Jacobi 2-node sweep parameters (more nx points, more reps, more GPN configs)
* Re-batch the NCCL graph ablation rather than running 8 separate srun calls

The current scripts' parameters are sized for ~50 s/srun overhead; bumping them up is a knob in each script.

## Items currently blocked / partial

These three are the ones to retry on the new cluster:

| Test | Blocker | Will it work on a fresh cluster? |
|---|---|---|
| **V1 DeepEP internode HT** (multi-node) | DeepEP V2 release `b306af0` regressed the V1 Python wrapper — `legacy.py:internode_dispatch` passes the wrong-shape args to the C++ binding (`TypeError`). | **No** — this is upstream code, not a cluster issue. To measure V1 internode HT honestly, `git checkout` an older DeepEP commit (pre-V2 release) and rebuild. |
| **V1 DeepEP low-latency multi-node** | NVSHMEM IBGDA either fails to init (default config) or hits `cudaErrorIllegalAddress` in the LL kernel (with `--disable-nvlink`). See [deepep/IBGDA_DEBUG.md](deepep/IBGDA_DEBUG.md). | **Yes if IBGDA works** on the new fabric. Validate with the IBGDA sanity check above first. |
| **V2 DeepEP low-latency 2 nodes × 4 GPU** | Just wasn't run; we already had 2-node × 8 GPU data and didn't need 2×4 too. | **Yes** — script in [deepep/run_2x4.sh](deepep/run_2x4.sh) can be extended. |

## What's portable as-is

Everything in the `repro/` tree assumes:

* The container working directory is the multi-gpu repo root (or you `cd` there first).
* Paths to NVSHMEM/CUDA/NCCL are settable via env vars in [`jacobi/setup_env.sh`](jacobi/setup_env.sh) — change the four `*_HOME` values for the new cluster.
* For multi-node, `srun --mpi=pmi2 -N <nodes> --ntasks-per-node=<N> ...` is the launcher. If your cluster has PMIx (most modern Slurm builds do), `--mpi=pmix` is fine too — just change the `--mpi=` argument in the multi-node scripts.

The DeepEP scripts have an extra setup step (`pip install nvidia-nccl-cu13`, `pip uninstall deep_ep`, symlink `_C.so`) baked into a `SETUP=` heredoc inside each `srun bash -c "..."` — see top of [deepep/run_2x8.sh](deepep/run_2x8.sh). If your new cluster's container ships a current `deep_ep` or PyTorch already linked to NCCL ≥ 2.30.4, you can drop those steps.

## Smallest "is the new cluster ready?" loop

```bash
# On a fresh cluster, in the container:
. repro/jacobi/setup_env.sh
bash repro/jacobi/recon.sh > /tmp/recon.log
cat /tmp/recon.log     # validate CUDA/NCCL/NVSHMEM versions match the table above

# 5-min smoke test:
bash repro/jacobi/build_all.sh
mpirun --allow-run-as-root --oversubscribe -np 2 \
   ./nccl/jacobi -csv -nx 1024 -ny 1024 -niter 100
NVSHMEM_SYMMETRIC_SIZE=4G mpirun --allow-run-as-root --oversubscribe -np 2 \
   -x LD_LIBRARY_PATH -x NVSHMEM_SYMMETRIC_SIZE \
   ./nvshmem/jacobi -csv -nx 1024 -ny 1024 -niter 100
```

Both should print a `nccl, ...` / `nvshmem, ...` CSV-style line in under a second. If either fails, the cluster setup needs more work before launching the full sweep.
