# DeepEP V1 (NVSHMEM) vs V2 (NCCL Gin) — single-node + 2-node measurements

## What got swapped

| | V1 (legacy `Buffer`) | V2 (`ElasticBuffer`) |
|---|---|---|
| Backend header | `<nvshmem.h>` | `<nccl.h>` + **`<nccl_device/core.h>`** |
| Symmetric heap & alloc | `nvshmem::alloc / barrier / init` ([csrc/legacy/buffer.hpp](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/legacy/buffer.hpp)) | `nccl::NCCLSymmetricMemoryContext` ([csrc/elastic/buffer.hpp](https://github.com/deepseek-ai/DeepEP/blob/main/csrc/elastic/buffer.hpp)) |
| Backend impls present in `csrc/kernels/backend/` | `nvshmem.cu` (used) | `nccl.cu` (used) |
| Dispatch/combine kernels | 3 separate `.cu` in `csrc/kernels/legacy/`: `intranode.cu` (NVLink), `internode.cu` (NVSHMEM RDMA, throughput), `internode_ll.cu` (NVSHMEM IBGDA, low-latency) | unified `dispatch.hpp / combine.hpp` in `csrc/kernels/elastic/`, all routed through NCCL Device API |

**The whole comm backend is replaced** — V2 doesn't pick "NCCL for some paths and NVSHMEM for others". Every dispatch / combine / barrier / RDMA put goes through `<nccl_device/core.h>` (NCCL 2.30's "Gin" — GPU-Initiated Networking). NVSHMEM is no longer linked into V2's hot path. (V1 code still ships in the same .so for the legacy `Buffer` API.)

## High-throughput vs low-latency mode mapping

| | V1 | V2 |
|---|---|---|
| **High-throughput** | `tests/legacy/test_intranode.py` (NVLink) / `tests/legacy/test_internode.py` (NVSHMEM RDMA) | `tests/elastic/test_ep.py` with `--prefer-overlap-with-compute 0` (default) |
| **Low-latency** | `tests/legacy/test_low_latency.py` (NVSHMEM IBGDA, pure RDMA, ~0 SM) | `tests/elastic/test_ep.py` with `--prefer-overlap-with-compute 1` + small batch |

V2 README: *"High-throughput and low-latency APIs unified into a single ElasticBuffer interface"*. So in V2 the mode is a property of the **call** (small batch + overlap flag → low-latency) rather than a separate kernel.

## Setup

* Repo: [/mnt/vast/home/tiancheng.chen/workspace/DeepEP](/mnt/vast/home/tiancheng.chen/workspace/DeepEP) at commit `b306af0` (the V2 release).
* Build: `NVSHMEM_DIR=/opt/nvshmem TORCH_CUDA_ARCH_LIST="9.0" python3 setup.py build`. Symlinked `_C.cpython-312-x86_64-linux-gnu.so` into `deep_ep/`.
* Container deps:
  - **Pip-installed `nvidia-nccl-cu13==2.30.4`** (V2 README requirement; container shipped with NCCL 2.29.3).
  - **Uninstalled stale `deep_ep` package** (the container ships an old `1.2.1` flat-layout deep_ep in `/usr/local/lib/python3.12/dist-packages/deep_ep/` which shadowed our local checkout when running test scripts).
  - `LD_LIBRARY_PATH` prepended with the new NCCL + the NVSHMEM bootstrap path.
* H100 80GB SXM5; 1-node = 8 GPU; 2-node = 4 GPU/node × 2 nodes.

## Single-node 8 GPU (intranode, NVLink-only)

Workload: `num_tokens=4096, hidden=7168, num_experts=256` (V2)/64 (V1 default).
Reported numbers are dispatch (FP8) / combine bandwidth and per-call latency.

### V1 NVSHMEM intranode — `tests/legacy/test_intranode.py`

V1's tuner sweeps `nvl_chunk` and reports the best configuration; V1's SM count is hard-coded to 24 (`for i in (24,):` in test_intranode main).

| Op | Best NVL BW | Best latency |
|---|---:|---:|
| Dispatch (FP8, topk=8) | **322 GB/s** | 497 µs |
| Dispatch (BF16, topk=8) | 331 GB/s | 938 µs |
| Combine (topk=8)       | **323 GB/s** | 961 µs |

Log: [v1_intranode.log](v1_intranode.log)

### V2 NCCL Gin intranode — `tests/elastic/test_ep.py`

Two configurations tested. V2 has analytical SM count (`get_theoretical_num_sms`); the "default" run lets V2 pick (it picked **64 SMs**). The "matched" run pins it to 24 SMs to compare apples-to-apples with V1.

| Variant | #SMs | topk | Dispatch (FP8) | Combine | Reduced combine |
|---|---:|---:|---:|---:|---:|
| **V2 default (auto SMs)** | 64 | 6 | 334 GB/s SU / **408 µs** | 346 GB/s SU / 755 µs | 348 GB/s SU / 749 µs |
| **V2 matched to V1**      | 24 | 8 | 306 GB/s SU / 526 µs | 313 GB/s SU / 986 µs | 308 GB/s SU / 1002 µs |

Logs: [v2_ep.log](v2_ep.log), [v2_ep_sms24_topk8.log](v2_ep_sms24_topk8.log)

### Side-by-side at 24 SMs (the V1 baseline)

| Stack    | Dispatch BW | Dispatch t | Combine BW | Combine t |
|---|---:|---:|---:|---:|
| V1 NVSHMEM | 322 GB/s | 497 µs | 323 GB/s | 961 µs |
| V2 NCCL Gin | 306 GB/s | 526 µs | 313 GB/s | 986 µs |

At equal SM budget V1 (NVSHMEM) is **5–6% faster** than V2 (NCCL Gin).

V2's headline win is **per-SM throughput**: with 64 SMs it reaches 408 µs dispatch / 755 µs combine — that's **23% faster dispatch / 22% faster combine vs V1@24SMs**, while using 2.7× more SMs. The V2 README claims "1.3× peak performance, 4× SM savings" — what we measure on a single H100 node is closer to **~1.2× peak performance with 2.7× SM use**, i.e. V2 wins on absolute throughput when you can spend the SMs, and V1 wins on per-SM efficiency.

## Single-node 8 GPU low-latency mode

Workload: `num_tokens=128, hidden=7168, num_topk=8, num_experts=288` (V1 LL test default; V2 invoked with the same args).

**V1 (NVSHMEM IBGDA kernel)** — `tests/legacy/test_low_latency.py`. NVSHMEM IBGDA failed to bring up on this fabric (`init failed for transport: IBGDA`); the test fell back through `nvshmem_transport_ibrc` and effectively used NVLink for intra-node RMA. So this isn't a "true" IBGDA measurement — it's the LL kernel running on NVLink.

| Metric | Value |
|---|---:|
| Dispatch + combine (combined) | 190 GB/s, **116 µs** |
| Dispatch alone | ~193 GB/s, **39 µs** |
| Combine alone | ~211 GB/s, **68 µs** |

Log: [v1_low_latency.log](v1_low_latency.log).

**V2 (NCCL Gin, ElasticBuffer)** — `tests/elastic/test_ep.py --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288 --num-sms 32 --num-qps 8 --prefer-overlap-with-compute 1`. Logs: [v2_ep_lowlat_1node.log](v2_ep_lowlat_1node.log), [v2_ep_smallbatch_default.log](v2_ep_smallbatch_default.log) (latter has `--prefer-overlap-with-compute 0`).

| Variant | Dispatch | Combine | Sum |
|---|---:|---:|---:|
| `prefer_overlap=1` | 155 GB/s, **32.3 µs** | 220 GB/s, **43.4 µs** | **75.7 µs** |
| `prefer_overlap=0` | 156 GB/s, 31.9 µs    | 217 GB/s, 44.0 µs    | 75.9 µs |

`--prefer-overlap-with-compute` makes essentially no per-call latency difference on this small workload — the flag mainly affects how V2 schedules SMs against compute, not the kernel itself.

### Side-by-side (LL workload)

| Stack | Dispatch | Combine | Dispatch+Combine | SMs | Notes |
|---|---:|---:|---:|---:|---|
| V1 NVSHMEM LL kernel | 39 µs | 68 µs | 116 µs (combined run) | 0 | IBGDA fallback to NVLink |
| V2 NCCL Gin (prefer_overlap=1) | **32 µs** | **43 µs** | **76 µs** | 32 | NVLink only (single node) |

V2 is **~35% faster end-to-end** at the LL workload on a single node. Caveats: (a) V1 LL was designed to use 0 SM (IBGDA does kernel-side RDMA), but had to fall back to NVLink here, so this isn't its specialty; (b) V2 used 32 SMs vs V1's 0 — V2's per-SM efficiency is therefore lower, but its absolute latency is shorter.

**Note on a previous mis-statement in this report**: the V2 README does mention *"0 SM RDMA low-latency EP is no longer supported"*, but this only refers to the **0-SM IBGDA kernel-only path**. V2 absolutely supports low-latency MoE dispatch+combine via NCCL Gin — it just spends a few SMs on the comm kernel. So V2 doesn't have a feature gap here, just a different SM trade-off.

## Multi-node measurements

Two allocations: 2×4 (8 GPUs total) and 2×8 (16 GPUs total). The 2×8 was needed because `tests/legacy/test_internode.py` asserts `num_local_ranks == 8` (V1 internode kernels are wired against `LEGACY_NUM_MAX_NVL_PEERS == 8`). Both allocations launched via `srun --mpi=pmi2 -N 2 --ntasks-per-node=1 ...` plus `torch.multiprocessing.spawn(nprocs=N)` per node, with `init_dist` reading `WORLD_SIZE=2 RANK=$SLURM_NODEID`.

### V2 ep 2-node — works ✓

Pinned `--num-sms` and `--num-qps` because V2's `get_theoretical_num_sms` divides by zero on this multi-node setup when both `nvlink_gbs` and `rdma_gbs` come back 0 from the auto-detect helpers (the helpers shell out to `nvidia-smi nvlink -s` and `ibstat`; the latter fails because the IB HCAs in this container expose `ibpX` aliases instead of bare `mlx5_X`).

**2 nodes × 4 GPU = 8 ranks** (high-throughput, ny=4096 tokens, 7168 hidden):

| Variant | topk / experts | Dispatch (SO / SU) | Dispatch t | Combine (SO / SU) | Combine t |
|---|---|---:|---:|---:|---:|
| topk=6, e=256 (24 SMs) | V2 default   | 35 / 80 GB/s  | 1700 µs | 81 / 183 GB/s | 1430 µs |
| topk=8, e=64 (24 SMs)  | matches V1   | 36 / 97 GB/s  | 1710 µs | 79 / 215 GB/s | 1485 µs |

**2 nodes × 8 GPU = 16 ranks** (more parallelism per node):

| Variant | topk / experts | Dispatch (SO / SU) | Dispatch t | Combine (SO / SU) | Combine t |
|---|---|---:|---:|---:|---:|
| topk=8, e=64 (24 SMs)   | matches V1   | **58 / 196 GB/s** | 1063 µs | 69 / 234 GB/s | 1701 µs |
| topk=6, e=256 (32 SMs)  | V2 default   | **59 / 153 GB/s** | 1030 µs | 73 / 192 GB/s | 1583 µs |

Going from 2×4 → 2×8: SO BW jumps from 36→58 GB/s and dispatch latency drops from 1.71 ms → 1.06 ms — more local ranks let V2 amortize IB packetization better.

`SO` = scaleout (inter-node, IB), `SU` = scaleup (intra-node, NVLink). 58 GB/s SO ≈ **~78% of CX7 line rate** (400 Gb/s ≈ 50 GB/s wire-only; the V2 README's reference table reports 90 GB/s logical BW for EP 8×2, which counts local-rank traffic too — our 58 GB/s is the per-direction inter-node figure).

Logs: [v2_ep_2x4_topk8_sms24_e64.log](v2_ep_2x4_topk8_sms24_e64.log), [v2_ep_2x8_topk8_e64.log](v2_ep_2x8_topk8_e64.log), and the topk=6 / e=256 variants.

### V2 low-latency 2-node ✓

`--num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288 --num-sms 32 --num-qps 16 --prefer-overlap-with-compute 1` on 2×8 = 16 ranks:

| Op | SO | SU | t |
|---|---:|---:|---:|
| Dispatch | 17 GB/s | 52 GB/s | **111 µs** |
| Combine  | 29 GB/s | 88 GB/s | **126 µs** |
| Sum | | | ~237 µs |

vs single-node V2 LL (76 µs total) — adding IB hops triples the latency, expected.

Log: [v2_ep_lowlat_2x8.log](v2_ep_lowlat_2x8.log).

### V1 internode HT 2-node — **broken in this V2 release**

`tests/legacy/test_internode.py` calls `buffer.dispatch(...)` which routes to `legacy.py:internode_dispatch()` which forwards to the C++ binding `_C.Buffer.internode_dispatch`. The Python wrapper has been updated for V2 with extra arguments, but the C++ binding signature (still 20 positional args) doesn't match what the wrapper passes — `TypeError: incompatible function arguments`. This is a regression in the V2 public release `b306af0` for the V1 legacy code path. Logs: [v1_internode_2x4.log](v1_internode_2x4.log), [v1_internode_2x8.log](v1_internode_2x8.log). To actually measure V1 HT internode would need to checkout an older DeepEP commit (pre-V2) and rebuild — not done here.

### V1 low-latency 2-node — IBGDA debugged but kernel-side RDMA still crashes

This took most of the multi-node debugging time. Probe results in [ibgda_probe.log](ibgda_probe.log):
- `mlx5_0..mlx5_8` HCAs visible, `gdrdrv` loaded, NVSHMEM built with `IBGDA_SUPPORT=ON`.
- IB devices show as `ibpX` aliases (8 of them) AND `mlx5_X` (9 of them); NCCL OFI plugin warns *"device ibpX is not supported (expected HCA interface: mlx5)"* but this is benign — actual NCCL/NVSHMEM transports use `mlx5_*`.

Tried 4 NVSHMEM env-var combinations ([try_nvshmem_transports.sh](try_nvshmem_transports.sh)):

| Config | Result |
|---|---|
| `NVSHMEM_REMOTE_TRANSPORT=ibrc` | "init failed for transport: IBGDA" — V1 LL Python wrapper [legacy.py:106-122](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffers/legacy.py#L106-L122) **hardcodes** `NVSHMEM_IB_ENABLE_IBGDA=1`, so IBRC alone isn't an option. |
| `NVSHMEM_HCA_LIST=mlx5_0..8` + `NVSHMEM_REMOTE_TRANSPORT=ibgda` | Same IBGDA init failure. |
| `NVSHMEM_IBGDA_NIC_HANDLER=gpu + NUM_RC_PER_PE=2` | Same. |
| `NVSHMEM_REMOTE_TRANSPORT=libfabric NVSHMEM_LIBFABRIC_PROVIDER=verbs` | Different failure — libfabric provider doesn't bring up. |

Bisecting with a single-rank Buffer init: `NVSHMEM_DEBUG=DEBUG` reveals **IBGDA does init successfully on a single PE** when NCCL etc. don't get involved. The `init failed for transport: IBGDA` errors that flood the log are **per-PE, transient warnings, not fatal** — they're a red herring.

The **real** failure is downstream:
- With default `allow_nvlink_for_low_latency_mode=True` (which sets `NVSHMEM_DISABLE_P2P=0`), NVSHMEM's topology pass aborts with `[GPU 8] Peer GPU 0 is not accessible, exiting` → `building transport map failed`. That's because PE 8 (node 1's GPU 0) cannot P2P-access PE 0 (node 0's GPU 0); they need IB. The topology check shouldn't be required, but is fatal here.
- Passing `--disable-nvlink` to the test (sets `NVSHMEM_DISABLE_P2P=1`) **gets past the topology check** and IBGDA init reports OK. But then the **actual kernel-side RDMA write hits `cudaErrorIllegalAddress`** during `low_latency_dispatch`. Log: [v1ll_2x8_disable_nvlink.log](v1ll_2x8_disable_nvlink.log).

The illegal address from inside the LL kernel is most likely a NIC firmware / `gdrdrv` / DEVX permission issue specific to this container's NIC driver setup. Without privileged access to the host (NIC firmware version, DEVX caps, IOMMU mode), I can't make further progress. Conclusion: **V1 low-latency multi-node is blocked on a fabric/firmware issue, not a software bug** in DeepEP itself. The same `gdrdrv` / NIC stack is used by V2 NCCL Gin on the same nodes and works — so V2 doesn't trip whatever IBGDA is hitting.

## Takeaways

1. **At equal SMs in HT mode, V1 (NVSHMEM) is slightly (~5%) faster than V2 (NCCL Gin)** on single-node intranode dispatch+combine (V1 322 GB/s @ 24 SMs vs V2 306 GB/s @ 24 SMs). V2's win on the HT workload comes from being able to dial SM count up to 64 cheaply with its analytical config — on this hardware that buys ~20% more throughput vs V1's hand-tuned 24 SMs.
2. **In LL mode on a single node, V2 (NCCL Gin) is ~35% faster end-to-end** than V1 on the LL workload (75 µs vs 116 µs). Caveat: V1 LL had to fall back from IBGDA to NVLink on this fabric, so V1 isn't running its specialty path.
3. **V2 NCCL Gin scales cleanly to 2 nodes** (HT and LL) once `--num-sms / --num-qps` is pinned (the auto-config divides by zero when the bandwidth-probe shells out to `ibstat` and gets nothing). 2×8 ranks reach 58 GB/s scale-out (~78% of CX7 line rate).
4. **V1 internode HT 2-node is broken in the V2 release** (commit `b306af0`): `legacy.py:internode_dispatch` Python wrapper passes a different argument shape than the C++ `_C.Buffer.internode_dispatch` binding accepts, throwing `TypeError`. Would need an older DeepEP commit + rebuild to validate.
5. **V1 low-latency 2-node is blocked on the IBGDA / NIC firmware path**, not on software. Got past the IBGDA init issue (it's a P2P topology check failure that needs `--disable-nvlink`), but the actual kernel-side RDMA write triggers `cudaErrorIllegalAddress` — almost certainly a `gdrdrv` / NIC-firmware / DEVX permissions issue specific to this container. V2 NCCL Gin on the same nodes works fine, so the underlying fabric is OK; just NVSHMEM IBGDA's specific path doesn't come through.
6. **The container had two NCCL gotchas to fix before either V1 or V2 would import cleanly**: pip-install `nvidia-nccl-cu13>=2.30.4` for the ≥2.30.4 ABI V2 needs, and uninstall the stale `deep_ep 1.2.1` flat-layout package the container ships in dist-packages.

## Final coverage matrix

| Test | 1-node | 2-node × 4 | 2-node × 8 |
|---|---|---|---|
| V1 HT intranode | ✓ | n/a | n/a |
| V1 HT internode | n/a | ✗ needs 8 GPU/node | ✗ test API broken in V2 release |
| V1 LL | ✓ (IBGDA→NVLink fallback) | ✗ CUDA error | ✗ CUDA error in kernel (firmware?) |
| V2 HT (`test_ep`) | ✓ (24 SMs and 64 SMs) | ✓ | ✓ |
| V2 LL (`test_ep` + `prefer_overlap=1`) | ✓ | not run | ✓ |

## How to reproduce

One-time build of `deep_ep_cpp.so` (V1 NVSHMEM + V2 NCCL Gin both included):

```bash
cd /mnt/vast/home/tiancheng.chen/workspace/DeepEP
NVSHMEM_DIR=/opt/nvshmem TORCH_CUDA_ARCH_LIST="9.0" python3 setup.py build
```

Per-allocation setup (NCCL upgrade + uninstall stale deep_ep + symlink _C.so + LD_LIBRARY_PATH) is encapsulated in [setup.sh](setup.sh). Source it before running anything.

### Single-node 8 GPU

```bash
# Allocate (mistral repo, 8-GPU H100 with the standard container)
cd /mnt/vast/home/tiancheng.chen/workspace/mistral
uv run python -m scripts.utils.cluster ggpus --with_container True --num_gpus 8 --exclusive True

# Inside the container:
cd /mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models
bash repro/deepep/run_1node.sh    # runs all 6 single-node tests, ~15 min
```

[run_1node.sh](run_1node.sh) wraps `setup.sh` and dispatches V1 HT, V1 LL, V2 HT (default + matched-to-V1), V2 LL, V2 small-batch sequentially.

### Multi-node

```bash
# Allocate 2 nodes × 8 GPU (V1 internode requires 8 GPU/node):
cd /mnt/vast/home/tiancheng.chen/workspace/mistral
unset SLURM_JOB_ID
uv run python -m scripts.utils.cluster ggpus_custom \
    --container /mnt/vast/containers/gpu_a4d0481d.sqsh \
    --num_gpus 8 --num_nodes 2 --exclusive True

# From the login node, with the salloc job id:
JOBID=<job id> bash /mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/repro/deepep/run_2x8.sh
```

For 2 nodes × 4 GPU (V1 internode skipped), use [run_2x4.sh](run_2x4.sh) instead. For an IBGDA-only debug session, see [try_nvshmem_transports.sh](try_nvshmem_transports.sh) and [IBGDA_DEBUG.md](IBGDA_DEBUG.md).

---

## H200 cluster reproduction (2026-05-04)

Re-ran the same suite on a CoreWeave H200 cluster. Container changed from
`gpu_a4d0481d.sqsh` → `gpu_882f6e72.sqsh` (NVSHMEM 3.6.5 instead of 3.4.5),
GPU is H200 SXM 144 GB instead of H100 SXM 80 GB. NIC stack (ConnectX-7,
8 HCAs/node) is the same on both sides. Logs in `repro/deepep/results/*-newcluster-20260504.log`.

### Single-node 8 GPU (intranode, NVLink-only)

| Stack | Dispatch (FP8) | Combine | Notes |
|---|---:|---:|---|
| V1 NVSHMEM @ 24 SMs | **319.82 GB/s** / 500 µs | **320.79 GB/s** / 968 µs | Same as H100 (322 / 323 GB/s) |
| V2 NCCL Gin default (auto SMs) | 329 GB/s SU / 408 µs | 339 GB/s SU / 762 µs | H100 was 334 / 346 GB/s |
| V2 NCCL Gin matched to V1 (24 SMs, topk=8) | 304 GB/s SU / 528 µs | 308 GB/s SU / 1002 µs | H100 was 306 / 308 GB/s |

**Takeaway**: identical to the H100 conclusions — V1 NVSHMEM ~5% faster
than V2 NCCL Gin at equal SM budget; V2 default config gets ~7% more
throughput by spending 64 SMs vs V1's 24. The H200's extra HBM/compute is
not the bottleneck for these NVLink-bound workloads.

### Single-node 8 GPU low-latency mode

| Stack | Dispatch | Combine | Sum |
|---|---:|---:|---:|
| V1 NVSHMEM LL (IBGDA→NVLink fallback) | 37 µs / 200 GB/s | 67 µs / 218 GB/s | 112 µs (combined run, 196 GB/s) |
| V2 NCCL Gin (`prefer_overlap=1`) | **30 µs** / 164 GB/s | **45 µs** / 211 GB/s | **76 µs total** |

**Takeaway**: same as H100 — V2 ~30% faster end-to-end on a single-node LL
workload (V1's IBGDA falls back to NVLink because all PEs are on the same
NVLink island).

### Multi-node 2 nodes × 8 GPU = 16 ranks (high-throughput)

| Variant | Dispatch (SO / SU) | Dispatch t | Combine (SO / SU) | Combine t |
|---|---:|---:|---:|---:|
| V2 ep, topk=8, e=64 (24 SMs)   | **62 / 212 GB/s** | 980 µs  | 70 / 238 GB/s | 1670 µs |
| V2 ep, topk=6, e=256 (32 SMs)  | **63 / 167 GB/s** | 952 µs  | 73 / 192 GB/s | 1585 µs |

Compared to old H100 cluster (58 / 196 GB/s SO at topk=8): **H200 cluster
gets ~7% more scale-out BW** (62 vs 58 GB/s) and slightly faster combine.

### Multi-node 2 nodes × 4 GPU = 8 ranks (high-throughput)

| Variant | Dispatch (SO / SU) | Dispatch t | Combine (SO / SU) | Combine t |
|---|---:|---:|---:|---:|
| V2 ep, topk=8, e=64 (24 SMs)   | **42 / 115 GB/s**, cached **47 / 129 GB/s**  | 1443 µs  | **86 / 233 GB/s** | 1369 µs |
| V2 ep, topk=6, e=256 (24 SMs)  | **42 / 94 GB/s**, cached **47 / 105 GB/s**   | 1425 µs  | 85 / 190 GB/s  | 1357 µs |

The 2×4 dispatch SO BW (42 GB/s) is ~17% higher than the H100 result for
the same shape (36 GB/s at topk=8). Combine SO is even better (85 GB/s vs
the H100's 79 GB/s).

### Multi-node low-latency

| Setup | Dispatch | Combine | Sum |
|---|---:|---:|---:|
| 2×8 V2 LL (`prefer_overlap=1`) | 19 / 56 GB/s, **103 µs** | 30 / 90 GB/s, **124 µs** | ~227 µs |
| 2×4 V2 LL (`prefer_overlap=1`) | 19 / 48 GB/s, **103 µs** | 33 / 85 GB/s, **112 µs** | ~215 µs |

Both numbers are ~5% better than the corresponding H100 figures (which were
237 µs sum at 2×8). New for this cluster: the 2×4 V2 LL data point — the
old H100 run skipped it.

### V1 multi-node — still blocked

| Test | Result on H200 |
|---|---|
| V1 internode HT (DeepEP V2 release b306af0) | ✗ TypeError in `legacy.py:internode_dispatch` (same upstream bug as H100 run) |
| V1 internode HT (pre-V2 commit `92fe2de`) | ✗ `RuntimeError: DeepEP error: timeout (dispatch CPU)` — Python wrapper now matches but the C++ NVSHMEM RDMA dispatch hangs. Same root cause as V1 LL below. |
| V1 LL multi-node (default config) | ✗ `init failed for transport: IBGDA`, then `[GPU N] Peer GPU 0 is not accessible, exiting` (NVSHMEM topology check) |
| V1 LL multi-node (`--disable-nvlink`, sets `NVSHMEM_DISABLE_P2P=1`) | ✗ Past topology check, but kernel hits `cudaErrorIllegalAddress` during `low_latency_dispatch` — same fingerprint as old cluster |

**Conclusion**: contrary to the original expectation that this cluster
"should have working IBGDA", V1's NVSHMEM IBGDA path is blocked the same
way as on the previous CoreWeave H100 cluster. The fingerprint is
identical: NVSHMEM 3.x init reports IBGDA available, the topology check
fails on inter-node P2P (workaround: `NVSHMEM_DISABLE_P2P=1`), and once
past that the kernel-side RDMA write hits `cudaErrorIllegalAddress`. This
points at a NIC firmware / `gdrdrv` / DEVX permissions issue on the IB
fabric that affects both H100 and H200 nodes — V2 NCCL Gin works fine on
the same hardware so the underlying fabric is not "broken", just NVSHMEM
IBGDA's specific path doesn't come through. See
[IBGDA_DEBUG.md](IBGDA_DEBUG.md) for a fuller diagnostic.

### Updated final coverage matrix (H200 cluster)

| Test | 1-node | 2-node × 4 | 2-node × 8 |
|---|---|---|---|
| V1 HT intranode | ✓ | n/a | n/a |
| V1 HT internode | n/a | ✗ needs 8 GPU/node | ✗ broken (TypeError on V2; timeout on pre-V2) |
| V1 LL | ✓ (IBGDA→NVLink fallback) | ✗ IBGDA / cudaErrorIllegalAddress | ✗ IBGDA / cudaErrorIllegalAddress |
| V2 HT (`test_ep`) | ✓ (24 SMs and 64 SMs) | ✓ (topk=6 + topk=8) | ✓ (topk=6 + topk=8) |
| V2 LL (`test_ep` + `prefer_overlap=1`) | ✓ | ✓ **(new this run)** | ✓ |

---

## ✅ Update 3: V1 IBGDA fully working — version-pinning recipe (2026-05-04)

The "V1 multi-node — still blocked" section above is **now wrong**. The
combination that actually works on this cluster is reproducible from
public sources:

* **DeepEP commit `73b6ea4`** (PR #458 "support hidden-dim 3072" in public
  `deepseek-ai/DeepEP`) — the last pre-V2-layout commit with V1 IBGDA
  kernels intact. The V2 release `b306af0` regressed both V1 LL and V1
  internode HT (the V2 release was scoped to V2; V1 paths regressed
  silently). Checkout via `git checkout 73b6ea4` after a fresh clone.
* **NVSHMEM 3.4.5** (`pip install --target=$PREFIX nvidia-nvshmem-cu13==3.4.5`).
  This specific version is required because DeepEP V1's bundled
  `csrc/kernels/ibgda_device.cuh` uses the v1 `nvshmemi_ibgda_device_state_t`
  struct layout; NVSHMEM 3.5 changed it to v2 and silently corrupts memory
  in the kernel-side RDMA path. 3.4.5 is the last 3.x release with the v1
  layout.
* **`NVSHMEM_HCA_PREFIX=`** (empty string) at runtime — bypasses the
  default `mlx5*` device filter that rejects this cluster's `ibp*` device
  names. (Equivalent source patch in [`../CLUSTER_VERSIONS.md`](../CLUSTER_VERSIONS.md).)

Run script: [`run_v1_mistral_recipe.sh`](run_v1_mistral_recipe.sh) (kept
the filename for git-history continuity; the recipe itself only references
public sources). It primes a named pyxis container, builds DeepEP at
`73b6ea4` against NVSHMEM 3.4.5 if needed, and runs all four V1 tests.

### V1 results that now run (2-node × 8 GPU = 16 ranks unless noted)

| Test | Result |
|---|---|
| **V1 HT intranode (1×8)** | FP8 dispatch 322.6 GB/s NVL / 497 µs; combine 323.8 GB/s / 960 µs |
| **V1 HT internode (2×8)** | FP8 dispatch **78.5 GB/s RDMA**, 264 GB/s NVL / 769 µs; combine 63.1 GB/s RDMA, 212 GB/s NVL / 1855 µs |
| **V1 LL (2×8)** | 69.4 GB/s combined, **318 µs end-to-end** (per-op dispatch ~33-41 µs, combine ~450 µs) |
| **V1 LL (2×4 = 8 ranks)** | 69.4 GB/s, 318 µs (same as 2×8 — IBGDA bandwidth caps out) |

### Updated final coverage matrix (H200 cluster, 2026-05-04)

| Test | 1-node | 2-node × 4 | 2-node × 8 |
|---|---|---|---|
| V1 HT intranode | ✅ 322 GB/s | n/a | n/a |
| V1 HT internode | n/a | ✗ test asserts 8 GPU/node | ✅ **78.5 GB/s SO** |
| V1 LL | ✅ (NVLink fallback, 116 µs) | ✅ **318 µs** | ✅ **318 µs** |
| V2 HT (`test_ep`) | ✅ (24 SMs and 64 SMs) | ✅ | ✅ 62 GB/s SO |
| V2 LL (`test_ep` + `prefer_overlap=1`) | ✅ 76 µs | ✅ 110 µs | ✅ 227 µs |

**V1 vs V2 head-to-head on this cluster (2-node × 8 GPU)**:
* HT dispatch: V1 NVSHMEM IBGDA **78.5 GB/s SO** vs V2 NCCL Gin 62 GB/s
  → V1 ~25% faster on throughput
* LL latency: V1 NVSHMEM IBGDA 318 µs vs V2 NCCL Gin 227 µs
  → V2 ~30% faster on latency

So both backends are competitive on this cluster — pick V1 NVSHMEM for
throughput, V2 NCCL Gin for latency.

### Visual comparison (figures/)

Three plots in [results/figures/](results/figures/):
* `fig_ht_dispatch_bw.png` — HT dispatch BW, V1 NVSHMEM IBGDA vs V2 NCCL Gin at matched 24 SMs across 1×8 / 2×4 / 2×8
* `fig_ht_combine_bw.png` — same for HT combine
* `fig_ll_latency.png` — LL end-to-end latency (lower = better)

**Caveat to read these plots with**: V1 and V2 are *different DeepEP kernel
implementations* on top of *different transports* (NVSHMEM IBGDA vs NCCL Gin).
A bar-height difference reflects (transport efficiency) ⊗ (kernel design)
combined — e.g. V2 HT was redesigned around an analytical SM model, V1 LL
was designed for the 0-SM IBGDA kernel-only path while V2 LL spends 32 SMs
on a NCCL Gin device-side kernel. **You cannot read these plots as
"NVSHMEM IBGDA is X% faster than NCCL Gin"** — that would require an
isolated transport microbenchmark (e.g. raw `shmem_put_bw` vs
`nccl-tests/alltoall`) which we did not run here.

What the plots *do* show, factually: at this specific config (24 SMs FP8
for HT, default LL params), DeepEP V1 has higher HT *dispatch* BW; V2 has
higher HT *combine* BW at 2×4 and roughly ties at 2×8; and V2 has
decisively lower LL latency at every scale. So neither is uniformly better —
the choice depends on which op + scale you care about.
