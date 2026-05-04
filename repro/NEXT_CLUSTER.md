# Moving to a different cluster — what to verify and what to bring

This work was originally done on a CoreWeave H100 cluster (container
`gpu_a4d0481d.sqsh`), and **re-run on a CoreWeave H200 cluster on
2026-05-04** (container `gpu_882f6e72.sqsh`). The two clusters share the
same NIC stack and exhibit the same NVSHMEM IBGDA failure mode (V1 LL
multi-node and V1 internode HT remain blocked on both). This document
captures the items to validate before running on a third cluster.

**TL;DR for the H200 run** (notes captured 2026-05-04):
* Container path / partition changed: `gpu_882f6e72.sqsh`, partition `h200`.
* `/opt/nvshmem` on the H200 container has a **broken `nvshmem.h` symlink**
  → install `nvidia-nvshmem-cu13` to a workspace prefix and point
  `NVSHMEM_HOME` at it (the new `repro/jacobi/setup_env.sh` does this by
  default; the new `repro/deepep/setup.sh` falls back to a pip install if
  the symlink is broken).
* `nvshmem/Makefile` needed `-lnvshmem → -lnvshmem_host -lnvshmem_device`
  (NVSHMEM 3 split; same fix needed on H100 or H200, already committed).
* MPI lives at `/usr/local/mpi/bin/mpirun` here, not `/usr/bin/mpirun` —
  setup_env.sh updated.
* `srun --container-image=...` cost ~45 s per call regardless of payload,
  so the 2-node Jacobi sweep is again the most painful piece. The script
  now supports `REPS=2` to halve the wall time at the cost of one rep.
* IBGDA is **still broken** on this fabric — same `cudaErrorIllegalAddress`
  in the LL kernel after the topology check is bypassed with
  `NVSHMEM_DISABLE_P2P=1`. See [deepep/IBGDA_DEBUG.md](deepep/IBGDA_DEBUG.md)
  for the H200-specific findings.

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

| Tool | Required minimum | H100 cluster (`gpu_a4d0481d.sqsh`) | H200 cluster (`gpu_882f6e72.sqsh`) |
|---|---|---|---|
| CUDA | 11.0 (Jacobi) / 12.3 (DeepEP V2) | 13.0 ✓ | 13.0.88 ✓ |
| NCCL (V2 DeepEP) | 2.30.4 | container had 2.29.3 → `pip install --no-deps "nvidia-nccl-cu13>=2.30.4"` | container has 2.28.8 → same workaround |
| PyTorch (DeepEP V2) | 2.10 | 2.10.0a0 ✓ | 2.10.0a0+b558c986e8.nv25.11 ✓ |
| OpenMPI | any CUDA-aware OR plain OpenMPI for bootstrap-only | 4.1.9a1 at `/usr/bin/mpirun` (not CUDA-aware) | 4.1.9a1 at `/usr/local/mpi/bin/mpirun` (not CUDA-aware) — note the path change |
| NVSHMEM | 0.4.1+ for `nvshmem/jacobi`, 3.x for DeepEP V2 | 3.4.5 ✓; bootstrap MPI plugin in `/usr/lib/x86_64-linux-gnu/nvshmem/13` | 3.4.5 in `/opt/nvshmem` BUT `/opt/nvshmem/include/nvshmem.h` is a **broken symlink** to `/usr/include/nvshmem_13/` (which doesn't exist) → install `nvidia-nvshmem-cu13` (3.6.5) to a workspace prefix and point `NVSHMEM_HOME` at it |
| C++ standard | C++17 (CUDA 13 CCCL requirement) | repo Makefiles default to C++14 → already patched on this branch | same; same patch |
| nvshmem `-l` flag | NVSHMEM 3 split into _host.so + _device.a | `nvshmem/Makefile` already patched: `-lnvshmem` → `-lnvshmem_host -lnvshmem_device` | same fix is already in the branch |

### 4. Cluster overhead is reasonable

This cluster's `srun --container-image=...` cost ~52 s **per call**, regardless of payload (even a bare `srun hostname` was 44 s). This made the 2-node Jacobi sweep take ~40 min and forced the DeepEP multi-node tests to be 1-srun-per-test (no batching, since multiple sequential MPI sessions in one srun fail with SIGPIPE — first `MPI_Finalize` closes srun's PMI server).

If your new cluster's srun is fast (a few seconds), you can:

* Restore the larger Jacobi 2-node sweep parameters (more nx points, more reps, more GPN configs)
* Re-batch the NCCL graph ablation rather than running 8 separate srun calls

The current scripts' parameters are sized for ~50 s/srun overhead; bumping them up is a knob in each script.

## Items currently blocked / partial

Status update from H200 re-run (2026-05-04):

| Test | H100 result | H200 result | Notes |
|---|---|---|---|
| **V1 DeepEP internode HT** (V2 release `b306af0`) | ✗ TypeError in Python wrapper | ✗ same TypeError | Upstream bug; not cluster-dependent. |
| **V1 DeepEP internode HT** (pre-V2 commit `92fe2de`) | not retried originally | ✗ `RuntimeError: DeepEP error: timeout (dispatch CPU)` — wrapper now matches but C++ NVSHMEM RDMA dispatch hangs | Same fundamental IBGDA-doesn't-work issue. Build needs `CPATH=/usr/local/cuda/include/cccl` for CCCL. |
| **V1 DeepEP low-latency multi-node** | ✗ IBGDA init / `cudaErrorIllegalAddress` | ✗ same fingerprint on H200 (NVSHMEM 3.6.5) — see [deepep/IBGDA_DEBUG.md](deepep/IBGDA_DEBUG.md) | Both clusters share NIC/firmware path that NVSHMEM IBGDA can't use. |
| **V2 DeepEP low-latency 2 nodes × 4 GPU** | (skipped) | ✓ collected (~110 µs combine, 33 GB/s SO) | Now in the H200 results. |

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
