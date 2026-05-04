# NVSHMEM IBGDA debugging on this cluster

This is the catalog of every IBGDA-related error we hit while trying to make DeepEP V1 low-latency / V1 internode multi-node tests run, plus what each error actually means and what we tried.

V1 low-latency is the path most affected: `tests/legacy/test_low_latency.py` → `deep_ep/buffers/legacy.py:106-122` → **hardcodes `NVSHMEM_IB_ENABLE_IBGDA=1`** for `low_latency_mode`. If IBGDA can't init, V1 LL can't run — there's no IBRC fallback path in the kernel.

## Bottom line

On this cluster, NVSHMEM IBGDA does not produce a working multi-node RDMA path:

1. With default config, V1 LL multi-node dies in NVSHMEM init at **transport-map build** (`Peer GPU N is not accessible`) → `building transport map failed`.
2. With `--disable-nvlink` (= `NVSHMEM_DISABLE_P2P=1`), init goes through, but the actual kernel-side RDMA write inside `low_latency_dispatch` raises **`cudaErrorIllegalAddress`**.
3. The same nodes run V2 NCCL Gin (NCCL 2.30+ device API) cross-node fine — so the underlying fabric is healthy, this is specific to NVSHMEM IBGDA's path.

The likely root cause is one of: NIC firmware too old for IBGDA, container missing `CAP_NET_RAW`/`CAP_IPC_LOCK` for raw QP creation, `gdrdrv` permissions, or IOMMU translating the NIC→GPU DMA wrong. Without privileged host access we can't drill further.

## Errors by phase

### Phase 1 — IBGDA transport init (warning, not fatal)

```
/dvs/p4/build/sw/rel/gpgpu/toolkit/r13.0/main_nvshmem/src/host/transport/transport.cpp:nvshmemi_transport_init:282:
  init failed for transport: IBGDA
```

Frequency: once per PE. So 8x in 2x4, 16x in 2x8. This was misleading — looks like the headline error but it's actually a **per-PE warning**, NVSHMEM continues and tries IBRC/UCX/libfabric next. With `NVSHMEM_DEBUG=DEBUG` and a single-rank Buffer init, IBGDA actually initializes successfully and this warning doesn't appear, so the failure is something multi-rank-specific.

### Phase 2 — topology check fails (fatal under default config)

```
/dvs/p4/build/sw/rel/gpgpu/toolkit/r13.0/main_nvshmem/src/host/topo/topo.cpp:469:
  [GPU 0] Peer GPU 8 is not accessible, exiting ...
  [GPU 1] Peer GPU 8 is not accessible, exiting ...
  ... (every PE pair across nodes)

/dvs/p4/build/sw/rel/gpgpu/toolkit/r13.0/main_nvshmem/src/host/init/init.cu:1037:
  non-zero status: 3 building transport map failed
```

Why this is fatal: NVSHMEM tries to build a peer-accessibility map for every pair of PEs. With `NVSHMEM_DISABLE_P2P=0` (default; set to `0` by V1 LL Python wrapper unless `--disable-nvlink` is passed), it tries P2P first. PE 8 (node 1's GPU 0) cannot P2P-access PE 0 (node 0's GPU 0) — they're on different nodes. NVSHMEM aborts before IBGDA gets a chance.

**Workaround:** set `NVSHMEM_DISABLE_P2P=1` (use the test's `--disable-nvlink` flag).

### Phase 3 — cascading post-init noise

```
NVSHMEM API called before NVSHMEM initialization has completed
```

DeepEP keeps calling NVSHMEM APIs after init failed. Tear-down race, no information value.

### Phase 4 — with `--disable-nvlink`: kernel-side RDMA crash (the actual blocker)

```
WARNING: destroy() was not called before DeepEP buffer destruction, which can leak resources.
[rank N] CUDA warning: an illegal memory access was encountered (function destroyEvent)

torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress` in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.

  File "tests/legacy/test_low_latency.py", line 271, in test_loop
    test_main(num_tokens, ...)
  File "tests/legacy/test_low_latency.py", line 117, in test_main
    num_valid_tokens = recv_count.item()
                       ^^^^^^^^^^^^^^^^^
```

`cudaErrorIllegalAddress` is async, the stacktrace points to a `.item()` that synced. The real failure is inside the LL kernel during `nvshmemx_*_p` to a remote PE. Common causes for this specific error pattern:

* IBGDA writes a GPU memory address that gdrdrv hasn't registered with the NIC.
* `nvidia_peermem` (or older `nv_peer_mem`) module is loaded but doesn't have the right permissions.
* IOMMU passthrough/translation is not what IBGDA expects.
* mlx5 firmware doesn't support the IBGDA WQE format the NVSHMEM build was compiled for.

### Phase 5 — `libfabric` fallback: SIGSEGV in EFA libfabric

Tried `NVSHMEM_REMOTE_TRANSPORT=libfabric NVSHMEM_LIBFABRIC_PROVIDER=verbs`:

```
==== backtrace (tid:1919992) ====
 0  /opt/hpcx/ucx/lib/libucs.so.0(ucs_handle_error+0x2e4)
 ...
 4  /opt/amazon/efa/lib/libfabric.so.1(fi_dupinfo+0x3cb)
 5  /opt/amazon/efa/lib/libfabric.so.1(fi_dupinfo+0x280)
 6  /opt/amazon/efa/lib/libfabric.so.1(fi_getinfo+0x30)
 7  /usr/lib/x86_64-linux-gnu/nvshmem/13/nvshmem_transport_libfabric.so.3(nvshmemt_init+0xc77)
```

The container ships **AWS EFA**'s libfabric (`/opt/amazon/efa/lib/libfabric.so.1`), but the NICs are Mellanox CX7. EFA libfabric segfaults on `fi_getinfo` when probing mlx5 devices. Not a viable fallback in this container; would need a non-EFA libfabric build to test this path.

### Red herring — NCCL OFI plugin warnings (count: 64+)

```
WARN: device ibp0 is not supported (expected HCA interface: mlx5). Skipping...
WARN: device ibp1 is not supported (expected HCA interface: mlx5). Skipping...
... (×8 per node)
```

These come from the AWS OFI NCCL plugin. The IB devices in this container are exposed as `ibpX` aliases of `mlx5_X`; the plugin filters on the literal name `mlx5` and skips them. NCCL falls back to its native IB transport and works correctly — V2 DeepEP NCCL Gin runs across nodes despite these warnings. **Ignore.**

## What we tried (and didn't work)

All of the following were attempted and run end-to-end on this cluster:

| Config | Result | Log |
|---|---|---|
| Default V1 LL multi-node (2x4 and 2x8) | Phase 2 fatal | [results/v1_low_latency_2x4.log](results/v1_low_latency_2x4.log), [results/v1_low_latency_2x8.log](results/v1_low_latency_2x8.log) |
| `NVSHMEM_REMOTE_TRANSPORT=ibrc` only | Phase 2 fatal — V1 LL Python wrapper *forces* `NVSHMEM_IB_ENABLE_IBGDA=1` regardless | [results/v1ll_2x8_ibrc_only.log](results/v1ll_2x8_ibrc_only.log) |
| Explicit `NVSHMEM_HCA_LIST=mlx5_0..8 + REMOTE_TRANSPORT=ibgda` | Phase 2 fatal | [results/v1ll_2x8_ibgda_hca.log](results/v1ll_2x8_ibgda_hca.log) |
| `+ IBGDA_NIC_HANDLER=gpu + NUM_RC_PER_PE=2 + NUM_DCI=2` | Phase 2 fatal | [results/v1ll_2x8_ibgda_devx.log](results/v1ll_2x8_ibgda_devx.log) |
| `NVSHMEM_REMOTE_TRANSPORT=libfabric NVSHMEM_LIBFABRIC_PROVIDER=verbs` | Phase 5 SIGSEGV in EFA libfabric | [results/v1ll_2x8_libfabric.log](results/v1ll_2x8_libfabric.log) |
| `--disable-nvlink` (sets `NVSHMEM_DISABLE_P2P=1`) | Phase 4 — IBGDA inits, kernel crashes | [results/v1ll_2x8_disable_nvlink.log](results/v1ll_2x8_disable_nvlink.log) |

The probe of NIC / driver / NVSHMEM build configuration is in [results/ibgda_probe.log](results/ibgda_probe.log) (script: [probe_ibgda.sh](probe_ibgda.sh)). Summary of what's *present* on this cluster:

* `mlx5_0` … `mlx5_8` HCAs visible (also as `ibp0`…`ibp7` aliases — 8 of them).
* `/dev/gdrdrv` exists, mode `crw-rw-rw-`, owner `nobody:nogroup`.
* `/dev/infiniband/uverbs{0,1,2,3,4,6,7,8}` (note: no `uverbs5`).
* NVSHMEM 3.4.5 was built with `NVSHMEM_IBGDA_SUPPORT=ON`, `NVSHMEM_USE_GDRCOPY=ON`.

So the obvious bits are there. The not-obviously-checkable bits (firmware version, container caps, IOMMU mode) are the suspects.

## Sanity check for a new cluster

```bash
# In the container, on a freshly allocated 2-node job:
srun --jobid=$JOBID --mpi=pmi2 -N 2 --ntasks-per-node=8 \
     /opt/nvshmem/bin/perftest/device/pt-to-pt/shmem_put_bw -d gpu
```

If this prints bandwidth tables and exits cleanly, IBGDA is healthy and DeepEP V1 LL should work. If it errors with the same IBGDA / topology pattern as in this catalog, the same blocker applies.

## Useful env vars for next-cluster debugging

| Var | Purpose |
|---|---|
| `NVSHMEM_DEBUG=DEBUG` | Verbose logs — without this, "init failed for transport: IBGDA" is the only signal you get |
| `NVSHMEM_DEBUG_SUBSYS=INIT,TRANSPORT,IBGDA` | Narrow to relevant subsystems |
| `NVSHMEM_INFO=1` | Print NVSHMEM env vars at startup |
| `NVSHMEM_HCA_LIST=mlx5_0,mlx5_1,...` | Restrict to specific HCAs (skip non-GPU NICs) |
| `NVSHMEM_DISABLE_P2P=1` | Skip the P2P topology check that aborts on multi-node |
| `NVSHMEM_REMOTE_TRANSPORT=ibrc` | Force fallback transport (won't work for V1 LL — see above) |
| `NVSHMEM_IB_ENABLE_IBGDA=1` | Enable IBGDA (V1 LL does this automatically) |
| `NVSHMEM_IBGDA_NUM_RC_PER_PE=N` | Per-PE QP count |

---

## Update: H200 cluster (2026-05-04) — same failure mode

Re-ran the same scenarios on a different CoreWeave cluster (H200 SXM nodes,
container `gpu_882f6e72.sqsh`, NVSHMEM 3.6.5 instead of 3.4.5). Despite the
expectation that this fabric would have working IBGDA, the failure
fingerprint is **identical** to the original H100 run:

* V1 LL multi-node default config (no `--disable-nvlink`): NVSHMEM init
  reports `init failed for transport: IBGDA` (transient warning) and then
  fatally `[GPU N] Peer GPU 0 is not accessible, exiting`. Same topology
  check failure as before.
  See [v1_low_latency_2x8-newcluster-20260504.log](v1_low_latency_2x8-newcluster-20260504.log).
* V1 LL multi-node with `--disable-nvlink` (i.e. `NVSHMEM_DISABLE_P2P=1`):
  topology check is bypassed, IBGDA reports init OK, but the actual
  kernel-side RDMA write hits `cudaErrorIllegalAddress` during
  `low_latency_dispatch` — same fingerprint as on H100.
  See [v1_low_latency_2x8_disable_nvlink-newcluster-20260504.log](v1_low_latency_2x8_disable_nvlink-newcluster-20260504.log).
* Tried `NVSHMEM_IBGDA_NIC_HANDLER=mlx5` instead of `gpu`: same outcome.
  See [v1_low_latency_2x8_ibgda_mlx5_handler-newcluster-20260504.log](v1_low_latency_2x8_ibgda_mlx5_handler-newcluster-20260504.log).
* Tried a pre-V2 DeepEP commit (`92fe2de`, before the EPv2 release): the
  Python `TypeError` for V1 internode HT is gone (so the legacy wrapper
  path works), but the C++ `internode_dispatch` now blocks on
  `RuntimeError: DeepEP error: timeout (dispatch CPU)` — the inter-node
  NVSHMEM RDMA puts never make it across.
  See [v1_internode_2x8_pre_v2_92fe2de-newcluster-20260504.log](v1_internode_2x8_pre_v2_92fe2de-newcluster-20260504.log).

V2 NCCL Gin runs fine on the same allocation (62 GB/s SO BW at 2×8) — the
IB hardware, drivers, and switches are healthy. The issue is specifically
in NVSHMEM IBGDA's kernel-side RDMA path on these CoreWeave nodes,
regardless of whether it's an H100 or H200 box.

### What that suggests

Both clusters share something at a layer below NVSHMEM — most likely the
NIC firmware, DEVX permission policy, or a `gdrdrv` / `nvidia_peermem`
configuration that NVSHMEM IBGDA relies on but NCCL does not. NCCL's
GPU-Initiated Networking ("Gin") in NCCL 2.30+ takes a different path to
the NIC (probably `mlx5dv_create_qp_ex` with explicit doorbell mapping)
that this fabric *does* support. NVSHMEM's IBGDA implementation appears to
rely on a code path the host stack here doesn't fully expose.

If you have privileged access to the CoreWeave host:
- Check NIC firmware: `mlxfwmanager --query` (need root)
- Check IOMMU mode: should NOT be in passthrough mode for IBGDA
- Check `nvidia_peermem` is loaded and `lsof /dev/gdrdrv` shows the GPU processes
- Check if DEVX is enabled per-PF (`mlxconfig -d /dev/mst/mt4129_pciconf0 q | grep -i devx`)

Without those, V1 LL multi-node and V1 internode HT will continue to be
blocked. V2 NCCL Gin is the working alternative for inter-node EP on
CoreWeave H100 / H200.

---

## Update 2: H200 cluster deeper dig (2026-05-04, after first writeup)

The first H200 writeup above gave up too early. Re-investigating with the
hypothesis that the user said "IBGDA on this cluster works", I established:

### IBGDA *does* work on this cluster (verified)

`nvshmem/jacobi` cross-node with `NVSHMEM_IB_ENABLE_IBGDA=1` and **no
`NVSHMEM_HCA_LIST` override** initialises IBGDA cleanly on all 16 PEs:

```
NVSHMEM INFO Successfully initialized the transport: IBGDA. It will be used
             for device-side APIs over IB.
```

and the per-element `nvshmem_float_p` benchmark completes (5.5 ms for
2048² × 50 iter at 2 nodes × 4 GPU). Same with `-use_block_comm`. So:

* Driver is correctly configured for IBGDA — `/proc/driver/nvidia/params`
  shows `EnableStreamMemOPs: 1` and `RegistryDwords: "PeerMappingOverride=1;"`,
  exactly the regkey config DeepEP's `docs/nvshmem.md` recommends.
* NVSHMEM 3.6.5 IBGDA initialises and runs against this fabric.
* The `mlx5_*` device-name HCA list copied verbatim from the H100 cluster's
  `run_2x8.sh` was actually *wrong* for this cluster — `ibv_devinfo -l`
  returns `ibp0..ibp7` here, and pinning `NVSHMEM_HCA_LIST=mlx5_*` makes
  NVSHMEM's IBGDA enumeration filter out everything ("device ibp7 is not
  enumerated as an mlx5 device. Skipping..." → "Unable to initialize any
  transports"). With no HCA list, NVSHMEM picks `ibp*` and IBGDA works.

### DeepEP V1 LL still fails — but the failure is NOT IBGDA-broken-on-this-cluster

With NVSHMEM 3.6.5 + clean env (no HCA_LIST override), V1 LL gets past
init ("Successfully initialized IBGDA") and the kernel itself crashes with
`cudaErrorIllegalAddress` at
[`csrc/kernels/legacy/internode_ll.cu:552`](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/legacy/internode_ll.cu#L552)
(the LL dispatch kernel launch, with `CUDA_LAUNCH_BLOCKING=1`).

Tested NVSHMEM versions:

| NVSHMEM | `device_state` typedef | IBGDA init | LL kernel |
|---|---|---|---|
| 3.3.9 (DeepEP-recommended) | `v1` | ✗ "device ibp7 not enumerated as mlx5, skipping" → no transport | n/a |
| 3.3.20 (cu13)               | `v1` | ✗ same | n/a |
| 3.4.5 (cu13)                | `v1` | ✗ same | n/a |
| 3.5.21 (cu13)               | `v2` | ✓ initialises | (not tested) |
| 3.6.5 (cu13)                | `v2` | ✓ initialises  | ✗ `cudaErrorIllegalAddress` in dispatch kernel |

Earlier I suspected a v1/v2 struct-layout mismatch. Looking at the actual
diff, v2 only adds `int num_default_rc_per_pe;` *after* `globalmem` —
field offsets that DeepEP's [`ibgda_device.cuh`](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/legacy/ibgda_device.cuh)
actually reads (`globalmem.rcs`, `num_rc_per_pe`, etc.) are unchanged.
So a layout mismatch is *not* the explanation.

Other things tried, all still hit `cudaErrorIllegalAddress`:

* `NVSHMEM_DISABLE_P2P=1` (i.e. `--disable-nvlink`)
* `NVSHMEM_HCA_LIST=ibp0,..,ibp7`
* `NVSHMEM_IBGDA_NIC_HANDLER=mlx5` and `=gpu`
* `compute-sanitizer` confirms the only "real" pre-LL errors are NCCL
  trying to look up sm_90 kernels that aren't in NCCL 2.30 (those are
  benign — V2 ep works fine on the same allocation), then the actual
  `low_latency_dispatch_kernel` invalid-address.
* Pre-V2 DeepEP commit `92fe2de` for V1 internode HT: Python wrapper now
  matches but C++ NVSHMEM dispatch hangs with `RuntimeError: DeepEP error:
  timeout (dispatch CPU)` — same fingerprint of "IBGDA in DeepEP doesn't
  finish RDMA writes" but slightly different surface.

### Bottom line for the H200 cluster

* **NVSHMEM IBGDA itself works** — disprove anyone claiming the cluster's
  IB fabric or driver setup is at fault.
* **DeepEP V1 (legacy / NVSHMEM-backed) low-latency and internode HT
  remain broken on this cluster** for a reason that's specific to DeepEP's
  internal `nvshmemi_ibgda_*` device wrappers — it's not a raw IBGDA
  capability problem nor (as best I can tell) a v1/v2 struct mismatch.
* The right next step would be either (a) reach out to the DeepEP
  maintainers with the `cudaErrorIllegalAddress` repro, or (b) try a
  different DeepEP commit / use NIXL / use V2 NCCL Gin (which already
  works fine on this fabric).
* For our purposes, **V2 NCCL Gin is the working path** for inter-node EP
  on this cluster: 62 GB/s SO BW at 2×8, ~110 µs combine at 2×4 LL.
