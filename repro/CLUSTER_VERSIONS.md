# Cluster, hardware and software stack — for paper methodology section

All measurements in `repro/` were taken on a single CoreWeave H200 cluster
between 2026-05-04 and 2026-05-05.

## Hardware

| Component | Spec |
|---|---|
| GPU | **NVIDIA H200 SXM5, 144 GB HBM3e** (compute capability 9.0, Hopper) |
| GPUs / node | **8** (full SXM mesh; `nvidia-smi -L` reports devices `0000:19:00 / 0000:2d:00 / 0000:3f:00 / 0000:66:00 / 0000:9b:00 / 0000:ae:00 / 0000:bf:00 / 0000:e4:00`) |
| Intra-node interconnect | **NVLink-4** (900 GB/s/GPU bidirectional, 18 links/GPU); **NVLink-SHARP (NVLS) multicast supported and active** (verified at runtime: `NCCL INFO NVLS multicast support is available on dev 0..7`, `NVLS tuning: nChannels 16 chunkSize 131072 treeMaxChunkSize 131072`) |
| Inter-node interconnect | **Mellanox ConnectX-7 InfiniBand**, 8 NICs/node (PCIe gen5 x16 each); IB devices reported as `ibp0..ibp7` via `ibv_devinfo` (not `mlx5_*`); user-space libmlx5 1.25.56.0 |
| CPU | Intel/AMD x86_64 (varied per node; not the bottleneck) |
| Inter-node topology | 2 H200 nodes used per multi-node measurement (dev qos cap = 16 GPU per single job) |

## Cluster software

| Component | Version |
|---|---|
| Slurm | 23.11.4 (slurm-bar) |
| Job launcher | `srun --mpi=pmi2` from login node + pyxis container reuse via `--container-name` (cuts per-srun cost from ~50 s → ~11 s) |
| Container runtime | pyxis (NVIDIA enroot/pyxis) |
| Container image | `gpu_882f6e72.sqsh` (CoreWeave standard H200 image) |
| Storage | `/mnt/vast` NFS-shared between login + compute nodes |
| Slurm partition | `h200`, qos `dev` (16 GPU max per single job) |

## Compiler + base toolchain (inside container)

| Component | Version | Source |
|---|---|---|
| OS | Ubuntu (Python 3.12 base) | container |
| CUDA | **13.0.88** | `/usr/local/cuda` |
| CCCL | bundled with CUDA 13 | `/usr/local/cuda/include/cccl` |
| GCC / G++ | 13.3.0 | system |
| Python | 3.12.3 | system |
| OpenMPI | 4.1.9a1 (system at `/usr/local/mpi`); `libopenmpi-dev` from apt for `mpi.h` | container + apt |
| MPI bootstrap | PMI2 via `srun --mpi=pmi2`; `mpi.h` from `/opt/hpcx/ompi/include/mpi.h` (apt-installed) |

## NVIDIA stack — comm libraries

| Component | Version | Source / build flags |
|---|---|---|
| **NCCL** | **2.30.4 + cuda13.2** (`NCCL git version HEAD 747384637`) | pip wheel `nvidia-nccl-cu13==2.30.4` at `~/workspace/nccl-pip/nvidia/nccl/lib/`. Container shipped 2.28.8; pip-overrode for all measurements. |
| **NVSHMEM (thesis micro-benches)** | **3.3.9** + 4-line patch | Public NVSHMEM 3.3.9 source release from <https://developer.nvidia.com/nvshmem-downloads> (NVIDIA-distributed tarball, internal commit hash `e91f4bd5238a147f18d30791f43e0aeff0097a51` per `version.txt`). One 4-line patch applied to `src/modules/transport/ibgda/ibgda.cpp` to accept `ibp*` IB device names (see "Patch" below). Built with: `NVSHMEM_USE_NCCL=OFF`, `NVSHMEM_IBGDA_SUPPORT=ON`, `NVSHMEM_IBRC_SUPPORT=ON`, `NVSHMEM_NVLS_SUPPORT=ON` (default), `NVSHMEM_MPI_SUPPORT=OFF`, `CMAKE_CUDA_ARCHITECTURES=90`. Verified at runtime: `nm -D libnvshmem_host.so | grep -ci nccl` → 0; no NCCL fallback. |
| **NVSHMEM (jacobi)** | 3.6.5 | pip wheel `nvidia-nvshmem-cu13` (used `NVSHMEM_HOME=...` to point env at it) |
| **NVSHMEM (DeepEP V1 path)** | 3.4.5 | pip wheel `nvidia-nvshmem-cu13==3.4.5` (specific version required to match DeepEP V1's bundled `ibgda_device.cuh` v1 device_state struct) |

### Why three NVSHMEM versions?

The three NVSHMEM versions in the table are *not* mix-and-match — each
investigation had a hard constraint that pinned its choice. None of the
versions can substitute for each other in their respective roles.

| Investigation | NVSHMEM | Why this version specifically |
|---|---|---|
| **Jacobi** | **3.6.5** (pip wheel) | `nvshmem/jacobi` is a self-contained C++ program — it just needs *any* working NVSHMEM 3.x runtime. The container's bundled `/opt/nvshmem` was unusable (broken `nvshmem.h` symlink to `/usr/include/nvshmem_13/` which didn't exist), so we pip-installed `nvidia-nvshmem-cu13` and got 3.6.5 (whatever was current). 3.4.5 or 3.3.9 would also work. |
| **DeepEP V1 path** | **3.4.5** (pip wheel, pinned exactly) | DeepEP V1's bundled `csrc/kernels/internode_ll.cu` and `csrc/kernels/ibgda_device.cuh` call into NVSHMEM's IBGDA device kernels using a fixed `nvshmemi_ibgda_device_state_t` struct layout. **NVSHMEM 3.5 changed that struct (v1 → v2 layout)**; building DeepEP V1 against any NVSHMEM ≥ 3.5 silently corrupts memory in the kernel-side RDMA path. NVSHMEM 3.4.5 is the latest release that retains the v1 struct layout DeepEP V1 was written against. (Reproducible from public sources: clone `deepseek-ai/DeepEP` at commit `73b6ea4`, inspect the bundled `ibgda_device.cuh`, then compare against NVSHMEM 3.4.x vs 3.5.x release tarball headers.) |
| **Thesis micro-benches** | **3.3.9** + 4-line patch (built from source) | The pip wheels ship only the runtime libs, not the perftest binaries we needed (`shmem_*_bw`, `shmem_*_latency`, etc.). So we built from the public NVSHMEM 3.3.9 source release (downloaded tarball) and applied a 4-line patch to `src/modules/transport/ibgda/ibgda.cpp` so the IBGDA device-name filter accepts this cluster's `ibp0..ibp7` IB devices alongside `mlx5*`. The patch is reproduced in full in the "IBP device-name patch" section below — small enough to inline in a paper appendix. |

So in short:
* **3.6.5**: convenience (pip), no functional constraint, jacobi doesn't care.
* **3.4.5**: hard ABI requirement from DeepEP V1's IBGDA device-side code (public DeepEP V1 commits ≤ `73b6ea4` predate the v2 device_state).
* **3.3.9 + patch**: built from source so we get perftest binaries, plus a 4-line cluster-specific device-name patch (reproduced below).

If you were starting fresh today and only doing the thesis micro-benches +
jacobi (no DeepEP V1), you could collapse to a single NVSHMEM build —
`3.6.5 source + ibp patch + perftest enabled` would cover both. The DeepEP
V1 investigation is the only one that *requires* a separately pinned
NVSHMEM version.

### IBP device-name patch (reproduce)

The single patch we applied on top of NVSHMEM 3.3.9 source. Lines 3712-3715
of `src/modules/transport/ibgda/ibgda.cpp`:

```diff
@@ -3709,10 +3709,10 @@ int nvshmemt_init(nvshmem_transport_t *t, struct nvshmemi_cuda_fn_table *table,
         const char *name = ftable.get_device_name(device->dev);
         NVSHMEMI_NULL_ERROR_JMP(name, status, NVSHMEMX_ERROR_INTERNAL, out,
                                 "ibv_get_device_name failed \n");
-        if (!strstr(name, "mlx5")) {
+        if (!strstr(name, "mlx5") && !strstr(name, "ibp")) {
             ftable.close_device(device->context);
             device->context = NULL;
-            NVSHMEMI_WARN_PRINT("device %s is not enumerated as an mlx5 device. Skipping...\n",
+            NVSHMEMI_WARN_PRINT("device %s is not enumerated as an mlx5 or ibp device. Skipping...\n",
                                 name);
             continue;
         }
```

Equivalent runtime workaround (no rebuild) is **`NVSHMEM_HCA_PREFIX=`** (empty
string) which bypasses the default `mlx5` prefix filter — but the source
patch is more explicit about which devices we expect.
| **Hydra launcher** | mpich-4.0.2 (`nvshmrun.hydra`) | built from upstream via NVSHMEM's `scripts/install_hydra.sh`; needed for single-node sanity tests inside container |

## Test harnesses

| Component | Version | Notes |
|---|---|---|
| **nccl-tests** | upstream HEAD as of 2026-05-04 (`f727aa2 NCCL_TESTS_VERSION 2.18.3`) | Built with `MPI=1 MPI_HOME=/usr/local/mpi NCCL_HOME=$NCCL_PIP CUDA_HOME=/usr/local/cuda make -j 8 NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"`. The version-2.18.3 number on the binary is the `nccl-tests` repo's own version, not NCCL's; the linked NCCL is 2.30.4 (verified by `nccl-headers=23004 nccl-library=23004` in test output). |
| **NVSHMEM perftest** | bundled with the 3.3.9-ibp build | Both device (`device/pt-to-pt/`, `device/coll/`) and host on_stream (`host/coll/`) variants used. |
| **PyTorch (DeepEP V2 only)** | 2.10.0a0+b558c986e8.nv25.11 | container's pre-installed nvidia PyTorch build |
| **DeepEP** | commit `73b6ea4` (V1, pre-V2) and `b306af0` (V2 release `[Public release 26/04] EPv2`) | Both from public `deepseek-ai/DeepEP` GitHub repo. `73b6ea4` (PR #458 "support hidden-dim 3072") is the last commit before the V2 layout transition; we use it because it has the V1 IBGDA kernels intact. `b306af0` (PR #605) is the public V2 release. |

## RDMA / networking userland

| Component | Version |
|---|---|
| `libmlx5` | 1.25.56.0 (container ships only `libmlx5.so.1`; `ln -sf libmlx5.so.1 libmlx5.so` was needed for NVSHMEM cmake's `find_library(mlx5)`) |
| `libibverbs1` | 56.0-1 |
| `librdmacm1` | 56.0-1 |
| `ibverbs-providers` | 56.0-1 |
| `doca-sdk-rdma` | 3.1.0105-1 |
| `gdrcopy` | not available in container (`NVSHMEM_USE_GDRCOPY=0`) |
| `slurm-client` | 23.11.4 (apt-installed inside container so `srun` can be invoked from inside; also bind-mounted host `/etc/slurm` into container) |

## Run methodology — what was actually measured

All comparisons reported in `repro/REPORT.md` use **8 trials per (config, scenario, op)**, each trial issuing a fresh `srun --mpi=pmi2 --jobid=$JOBID --overlap --container-name=$CNAME ...` so each trial pays its own cold-cache cost. Two methodologies coexist:

| Methodology | NCCL flags | NVSHMEM flags | NCCL env | When to use |
|---|---|---|---|---|
| **Rigorous** (`bench_rigorous.sh`) | `-b 4 -e 33554432 -f 2 -w 20 -n 50` (no graphs) | `-b 4 -e 33554432 -n 50 -w 20` | (default) | Stress-test of per-call **launch overhead** + transport |
| **Fair recipe** (`bench_fair.sh`) | `-b 128 -e 1G -f 2 -w 50 -n 100 -c 0 -R 2 -G 10` | `-b 128 -e 268435456 -n 100 -w 50 --cudagraph` | `NCCL_GRAPH_MIXING_SUPPORT=0` | Real perf comparison: **CUDA graphs (10× replay/iter) amortise launch overhead**, sym kernel (`-R 2`) engages `ncclSymmetricTaskScheduler` (verified via `[Symmetric]` debug tag) |

For both methodologies:
* **NCCL bootstrap**: `srun --mpi=pmi2`, optionally `NCCL_NVLS_ENABLE=0` to ablate NVLS
* **NVSHMEM bootstrap**: `srun --mpi=pmi2 + NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2` (the `BOOTSTRAP_PMI=PMI2` sub-flavor selector is required — default loads PMI-1 plugin against srun's PMI-2 server and silently hangs in rendezvous)

## Recommended one-paragraph methodology blurb (paper-ready)

> All measurements were performed on a CoreWeave H200 cluster: 2 nodes ×
> 8 NVIDIA H200 SXM (144 GB HBM3e, compute capability 9.0) connected by
> NVLink-4 intra-node (NVLink-SHARP / NVLS multicast hardware-supported
> and confirmed active) and Mellanox ConnectX-7 InfiniBand cross-node
> (8 NICs/node, 400 Gb/s each). Software: CUDA 13.0.88, NCCL 2.30.4
> (`nvidia-nccl-cu13` pip wheel; container's stock 2.28.8 was overridden),
> NVSHMEM 3.3.9 (publicly-released NVIDIA source tarball,
> [developer.nvidia.com/nvshmem-downloads](https://developer.nvidia.com/nvshmem-downloads),
> with a 4-line patch to `src/modules/transport/ibgda/ibgda.cpp` extending
> the IBGDA device-name filter from `mlx5*` to also accept `ibp*`; the
> patch is reproduced in full in our methodology appendix; built with
> `NVSHMEM_USE_NCCL=OFF` so there is no NCCL fallback in the NVSHMEM hot
> path; verified at runtime via `nm` on the resulting `libnvshmem_host.so`).
> NCCL collectives were exercised through nccl-tests (upstream HEAD, built
> with `MPI=1` against the 2.30.4 wheel); NVSHMEM through its bundled
> perftest. All numbers are mean ± standard deviation over **8 trials**
> per configuration; each trial uses the recommended NCCL recipe
> (`NCCL_GRAPH_MIXING_SUPPORT=0`, 50 warmup + 100 timed iterations,
> per-iter CUDA-graph replay = 10, symmetric memory registration via
> `ncclMemAlloc + ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC)`); the
> NVSHMEM equivalent uses the perftest's `--cudagraph` mode with matched
> warmup and iteration counts. Multi-node launches use `srun --mpi=pmi2`
> with the NVSHMEM bootstrap configured as `NVSHMEM_BOOTSTRAP=PMI
> NVSHMEM_BOOTSTRAP_PMI=PMI2`.
