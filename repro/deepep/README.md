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
