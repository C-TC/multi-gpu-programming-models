# B200 rerun of the NVSHMEM paper micro-benchmarks (arXiv 2606.05951, Sec. "Microbenchmarking")

Re-measurement of the paper's two figures — **one-sided RMA (put/get/p/g, intra 1×2 + inter 2×1)** and
**AllReduce (8 GPUs intra, 16 GPUs across two nodes)** — on DGX B200, with the newest public NCCL/NVSHMEM,
using the *same benchmark binaries, arguments, environment variables and 8-trial protocol* as the H200 data in
`C-TC/multi-gpu-programming-models@reproduction-h100-nvshmem-vs-nccl:repro/thesis_microbench`
(`bench_rigorous.sh` → P2P figure, `bench_allreduce_focused.sh` → AllReduce figure).

**Drop-in figures** (same series, layout and insets as the paper; B200 references: NVLink 900 GB/s, IB 50 GB/s, 8×IB 400 GB/s):
`results/figures/nvshmem-p2p-bandwidth-onecol.pdf` and `results/figures/nvshmem-allreduce-bandwidth-onecol.pdf` (paper configs; NCCL on the refactored enqueue path).
Variants: `*-tuned.pdf` (tuned NVSHMEM) and `*-nccl-legacy.pdf` (NCCL legacy enqueue path).
`results/tables/paper_text_numbers.md` recomputes every number the paper's text quotes, using the figure's definitions.

| Paper config, 8 trials | H200 (paper data) | **B200** |
|---|---|---|
| intra put / get / p / g peak (GB/s) | 313 / 141 / 172 / 8.8 | **672 / 73 / 181 / 18** |
| inter put / get / p / g peak (GB/s), IBGDA tuned | 48.0 / 48.2 / 15.6 / 1.28 | **48.9 / 49.1 / 15.7 / 5.6** |
| AllReduce intra peak: NVSHMEM on-stream / device / NCCL ring / NCCL NVLS (GB/s) | 262.5 / 29.7 / 164.9 / 274.8 | **303.5 / 15.4 / 398.7 / 479.2** |
| AllReduce inter peak: NVSHMEM on-stream / device / NCCL ring / NCCL NVLS-Tree (GB/s) | 0.14 / 0.16 / 179.6 / 246.4 | **0.14 / 0.13 / 195.7 / 366.1** |
| AllReduce inter latency @64 KiB: NVSHMEM on-stream / device / NCCL ring / NVLS (µs) | 2324 / 2248 / 53 / 31 | **2266 / 2399 / 62 / 37** |

**Latency:** the P2P insets show *measured* one-sided latency from `shmem_*_latency` (8 trials), not the paper's size/BW: intra put/get/p/g
2.5 / 4.2 / 2.2 / 1.9 µs; inter (IBGDA) 20.4 / 17.9 / 14.8 / 14.3 µs at 8 B. See *Latency*; the upstream intra `g` latency test needed a fix.

Low B200 numbers at the paper's settings are configuration-bound, not hardware limits: intra `get` (73 GB/s), scalar p/g, and intra on-stream (16 CTAs).
With a few NVSHMEM knobs they reach 532 / 699 / 584 / 474 GB/s (see *Tuned*). B200 intra small-message latencies for NVSHMEM P2P and device
AllReduce are 25–60% higher than the H200 data (e.g. put 2.3 vs 1.8 µs; device AllReduce 5.9 vs 3.8 µs), while NCCL's symmetric kernels
are as fast as on H200 (≈4 µs).

## Setup

| | H200 (paper) | B200 (this rerun) |
|---|---|---|
| GPUs / node | 8× H200 SXM, NVLink4 (450 GB/s/dir) | 8× B200, NVLink5 18 links × 53.1 GB/s (900 GB/s/dir nominal) |
| NICs | 8× CX-7 IB (50 GB/s) | 8× CX-7 NDR 400G (50 GB/s), one PXB-local NIC per GPU |
| NVSHMEM | 3.3.9 (+ibp patch) | **3.8.0-0** (GitHub release, built from source, `NVSHMEM_USE_NCCL=OFF`, IBGDA+IBRC on, 0 NCCL symbols) |
| NCCL | 2.30.4 | **2.32.6-1** (built from source, sm_100) |
| nccl-tests | HEAD 2026-05-04 | **v2.21.1** (`MPI=1`, HPC-X OpenMPI from the container) |
| CUDA | 13.0.88 | 13.3 (NGC `pytorch:26.06-py3` container; driver 580.178.04 + CUDA forward-compat) |
| Nodes | CoreWeave | prenyx (MARS); P2P/AllReduce node names are recorded in every log header |

Hardware references drawn in the figures: NVLink 900 GB/s per GPU per direction; IB 50 GB/s per NIC, 8×50 = 400 GB/s per node.

## What was run (identical to the paper unless listed under *Deviations*)

P2P (`device/pt-to-pt/shmem_{put,get,p,g}_bw`, 8 trials):
* intra, 1 node × 2 PEs: `-b 4 -e 33554432 -n 50 -w 20 -c 32 -t 256`, env: PMI2 bootstrap only.
* inter, 2 nodes × 1 PE, "IBGDA tuned" (this is what the paper's inter panel actually plots, for **all four** ops):
  `NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_DISABLE_NVLS=1 NVSHMEM_IBGDA_NUM_RC_PER_PE=64`, `-b 4 -e 33554432 -n 20 -w 5 -c 64 -t 1024`.

P2P latency, **measured** (not ping-pong RTT, and not size/BW). These are the dedicated `device/pt-to-pt/shmem_{put,get,p,g}_latency` perftests
with NVSHMEM's shipped `.args`: `-b 4 -e 65536 -n 200 -w 20 --repetitions 3`, `-t 1024` (put/get) and `-t 512` (p/g; p uses `-n 50 -w 5`), 8 trials.
One PE launches a single 1-CTA kernel and times `iters` back-to-back operations:
* put/get: `{put,get}_nbi` + `nvshmem_quiet` per iteration (Block-scope table = same bulk op as `*_bw`).
* p: scalar `p` per element + `quiet`.
* g: blocking scalar `g` per element.

So this is one-sided completion latency: remote-completion ack for put/p, data return for get/g. For p and g above 512 elements (2 KiB), each thread
serializes several operations, so the number becomes "time to p/g the whole buffer". The paper's insets were size/BW from the `*_bw` runs; the new insets plot these measured values.

AllReduce (float, sum, 8 trials):
* NCCL [NVLS]: `all_reduce_perf -b 128 -e 1G -f 2 -w 50 -n 100 -c 0 -R 2 -G 10`, `NCCL_GRAPH_MIXING_SUPPORT=0`.
* NCCL [Ring]: same + `NCCL_NVLS_ENABLE=0 NCCL_ALGO=Ring`.
* NVSHMEM [Device Block]: `reduction_focus` (paper's float-sum-block perftest, ported to 3.8; file in `nvshmem_patches/`),
  intra `-b 128 -e 2147483648 -n 100 -w 50 --cudagraph`, inter `... -n 5 -w 2 --cudagraph`.
* NVSHMEM [On-stream]: `reduction_on_stream -d float -o sum`, intra `-b 128 -e 1073741824 -n 50 -w 20 --cudagraph`,
  inter `-b 128 -n 5 -w 2` (no graph, as in the paper).

### Deviations from the paper setup (and why)
1. **Library versions** are the newest public releases (NCCL 2.32.6-1, NVSHMEM 3.8.0-0, nccl-tests 2.21.1) instead of 2.30.4 / 3.3.9.
2. **NCCL enqueue path.** The main figures use NCCL's *refactored enqueue pipeline* (`NCCL_ENQUEUE_REARCH_ENABLE=1`). It is the default only on
   NCCL's `dev` branch; in the 2.32.6 release, `master` and `staging` it is still opt-in (`src/enqueue/enqueue.cc`: default 0 → `groupLaunchLegacy`).
   Every NCCL series was measured on **both** paths (`ar_rearch_*` vs `ar_paper_*`, 8 trials each); see *Verification* for the difference.
3. **NVSHMEM 3.8 perftest defaults.** `--use_smem 0` is passed to the P2P perftests: 3.8 registers shared memory for TMA by default
   (a no-op because `NVSHMEM_TMA_POLICY` defaults to `DISABLE`, but turned off to match the 3.3.9 kernels).
   `NVSHMEM_IBGDA_RC_MAP_BY=cta` is pinned for the paper's "IBGDA tuned" inter-node P2P config, because 3.8 changed the default from `cta` to `none`.
   The per-iteration inter-CTA barrier and final quiet in the bw kernels are unchanged from 3.3.9.
4. **`reduction_focus`** was re-derived from 3.8's `reduction_latency.cu` (float-sum-block only, same timing loop and size sweep) instead of
   copying the 3.3.9 file, so that it uses 3.8's perftest harness (launch macros with the dynamic-smem argument, repetition stats).
5. **Inter-node NVSHMEM on-stream max size.** The paper used `-e 2147483648`, which sweeps on-stream up to **2 GiB**. Over the IBRC proxy that single unplotted point costs about 8.5 of roughly 17 min per trial. The sweep was capped at `-e 536870912` (= 2^29, the largest *plotted* size), so every plotted point is measured exactly as in the paper.
   The inter-node device-block runs keep the paper's exact arguments.
6. **Node pairs.** The 8 trials of a series normally ran on one node pair. The exception is the paper-config inter-node device-block series:
   trials 1–7 ran on prenyx0047/0238 and trial 8 on prenyx0234/0235 (hold expiry). Every log header records its nodes.
7. **NVLink / IB references** in the figures are B200's: 900 GB/s per GPU per direction, 8×50 GB/s.

## Verification
All logs referenced below are in `results/raw/` (each starts with a `#BENCH` header recording nodes, env and command, and ends with the exit code).

**Software actually loaded** (`vfy_ldd`): `libnccl.so.2 → install/nccl` (2.32.6, banner `NCCL version 2.32.6+cuda13.3`),
`libnvshmem_host.so.3 → install/nvshmem` (3.8.0; `nm -D … | grep -ci nccl` = 0, so there is no hidden NCCL fallback), and HPC-X OpenMPI from the image.
The image's own NCCL 2.30.5 / NVSHMEM 3.6.5 are not used, except deliberately in `inv_ring230_*`. `NCCL_NET_PLUGIN=none` keeps NCCL on its
internal IB transport, as in the paper (the image presets the Spectrum-X plugin).

**Data correctness**
* NCCL: `all_reduce_perf -c 1` for both configs, intra (8 GPUs) and inter (16 GPUs), 128 B–1 GiB, on **both** enqueue paths
  (`vfy_correct_*`, `vfy_rearch_correct_*`): `#wrong = 0` at every size, "Out of bounds values : 0 OK".
* NVSHMEM AllReduce: `validate/ar_check.cu` checks float-sum results **bitwise** (integer-valued inputs, so fp32 sums are exact) for exactly the
  plotted paths, `nvshmemx_float_sum_reduce_on_stream` and single-CTA `nvshmemx_float_sum_reduce_block` on TEAM_WORLD, 128 B–512 MiB:
  **PASS, 0 bad elements** intra (default and `NVSHMEM_MAX_CTAS=64`) and inter (IBRC, IBGDA, and tuned 256 MiB scratch).
* NVSHMEM functional tests `reduce_block_sum` and host `reduce` (all types/ops/teams): exit 0 with no error lines, intra and inter.
  Intra needs `NVSHMEM_MAX_TEAMS=1024`, because the test's many team splits plus NVLS duplicate teams exceed the default 256 — a harness limit, not a data error.

**Measurement validity**
* 8 trials per series (zero-byte check: 3; tuning sweeps: 1). Each `results/tables/*.csv` has mean, stddev and n per size.
  Max relative stddev per size across trials is ≤ 4% for every plotted series except paper-config inter `g` (5.5% at 512 KiB). Every plotted size has n = 8.
* Zero-byte put/get sizes are detected from each run's own `-c` and dropped (see Problem 1).
* `reduction_focus` always steps sizes ×2 in its timing loop, but labels them using `-f`. So it is only run with `-f 2`; the upstream perftest has the same issue.
* NCCL algorithm and kernel per size, from `NCCL_DEBUG_SUBSYS=TUNING`:
  * Intra NVLS (legacy path): `AllReduce_AGxLLMC_R` ≤ 8 KiB, `AllReduce_RSxLDMC_AGxSTMC` ≥ 64 KiB.
  * Intra NVLS (refactored path): same, **except RING LL for 16 KiB–512 KiB**.
  * Intra Ring: RING LL → LL128 → SIMPLE.
  * Inter NVLS: TREE LL/LL128 < 4 MiB, **NVLS_TREE** ≥ 4 MiB. Inter Ring: RING.
* NVSHMEM transport per config, from `NVSHMEM_DEBUG=INFO`: IBRC by default, and IBGDA (64 RC QPs, CTA mapping, mlx5_4, GPU-side NIC handler) with the IBGDA env.

**Refactored vs legacy NCCL enqueue path** (same binary and flags, 8 trials each):
* Intra-node NVLS config (`-R 2`, symmetric windows): the refactored path picks **RING LL instead of the symmetric `RSxLDMC_AGxSTMC` kernel
  for 16 KiB–512 KiB** and is **2.4–5.4× slower** there. For example, 16 KiB takes 23.3 vs 4.3 µs and 64 KiB takes 24.4 vs 8.8 µs.
  ≤ 8 KiB is 1–7% slower; ≥ 1 MiB is identical. This was confirmed on the same node (`ab_*_intra_*`, 3 trials) and in the 8-trial series on different nodes.
  It is visible in the main figure as a dip in the NCCL [NVLS] curve; the `*-nccl-legacy` figures show the legacy path.
* Intra Ring, inter Ring and inter NVLS-Tree: identical within 1–2% at every size on the same node pair (`ab_legacy_inter_*` vs `ar_rearch_inter_*`, both on prenyx0234/0235).
* Correctness: `#wrong = 0` on both paths.
* Caveat on node placement: the legacy 8-trial inter series ran on prenyx0047/0238 and the refactored one on prenyx0234/0235. The pairs differ by 8–17% in
  small-message NCCL latency (e.g. Ring 128 B: 59.7 vs 49.3 µs), which is placement, not the enqueue path.
  The paper-config inter NVSHMEM series ran on prenyx0047/0238, apart from device-block trial 8 (prenyx0234/0235, within 3% of trials 1–7).

**Parsing robustness:** in one trial, another rank's banner line (`mype: 15 …`) was interleaved into PE 0's result table. The parser now skips
any line that is not a complete numeric row instead of ending the table, and every plotted size of every series was checked to have n = 8.

## Problems found in the published H200 data
Each item was checked against the H200 logs in the reference repo and, where possible, reproduced on B200.

1. **Inter-node put/get at 256 B time an empty transfer.** `shmem_{put,get}_bw` give each CTA `floor(len/nCTAs)` doubles,
   so any size < 8 B × nCTAs moves **zero bytes** while BW is still computed from the nominal size. The inter panel uses 64 CTAs,
   so its 2^8 put/get point (and the inset's ~7 µs latency) is an empty iteration.
   * H200 logs: tuned (64 CTAs) per-iteration time is flat at 6.9 µs (put) / 6.7 µs (get) from 4 B to 256 B, then jumps to 15.0 / 14.6 µs at 512 B.
     Untuned (32 CTAs) is flat at 2.2 µs up to 128 B, then 9.4–9.5 µs at 256 B.
   * B200 (`vfy_zerobyte_*`, 3 trials): 64 CTAs flat at 12.0 / 8.7 µs to 256 B, then 21.8 / 18.4 µs at 512 B. 32 CTAs flat at 3.3 µs to 128 B.
     Control with 1 CTA: only 4 B is flat, and 8 B is already the full ~17.9 / 14.8 µs. The threshold is exactly 8 B × nCTAs.
   * In the new figures, such sizes are dropped (`analyze.py: p2p_series`, using each run's logged `-c`).
2. **Text and figure come from different runs.** The text's inter-node "put/get 9.4–9.5 µs at 256 B, ~9.7 µs at 64 KiB" are the *untuned*
   IBGDA run (32 CTAs: 9.39/9.52 µs at 256 B, 9.68/9.75 µs at 64 KiB). The figure plots the *tuned* run (64 CTAs: 256 B is empty, 15.0/14.6 µs at 64 KiB).
   The tuned config was also used for put/get, not only for p/g as the setup text says: the 48.0 / 48.2 GB/s peaks are the tuned run.
   Similarly, the text's p latency of 7.5 µs at 256 B is not in the plotted data (tuned p: 10.8 µs).
3. **Two peak numbers in the text use the unplotted 1 GiB point.** On-stream 264 GB/s and NVLS-Tree 252 GB/s are the 2^30 values
   (per-trial 262.8–264.6 and 251.9–252.2). The figure stops at 2^29 and labels 262.5 / 246.4 (8-trial means), which is what the data gives.
   Minor: the NCCL NVLS intra latency in the data is 4.9–6.0 µs (text: 5.6–5.9). Scalar g intra rises to 8.8 µs at 64 KiB (text: p/g 1.3–2.2 µs).
4. **The inter-node NVSHMEM AllReduce ran over the IBRC CPU proxy with the default 512 KiB reduce scratch.** `bench_allreduce_focused.sh`
   sets only the PMI bootstrap variables. In 3.3.9 (and 3.8), `NVSHMEM_IB_ENABLE_IBGDA` defaults to 0 and `NVSHMEM_REMOTE_TRANSPORT` defaults to `ibrc`.
   Confirmed on B200 with `NVSHMEM_DEBUG=INFO` (`vfy_transport_*`): default → "IBRC: selected slot 0 uses mlx5_4", and the IBGDA env → "Successfully initialized the transport: IBGDA".
   Across nodes, the reduction falls back to a segmented linear algorithm: `REDUCE_SCRATCH_SIZE/2/8` = 32 KiB segments, with puts to all 15 peers and a barrier per segment.
   That is why the paper's curve is flat at ~0.03 GB/s from 16 KiB, with ms latencies by 64 KiB. B200 reproduces this exactly (0.027 GB/s).
   With a 256 MiB scratch it is 15–25× faster (see *Tuned*), so "NVSHMEM < 0.20 GB/s across nodes" describes this configuration rather than NVSHMEM itself.
5. **The paper's intra-node "NCCL [Ring]" is not a ring (new finding).** With NCCL 2.30.x, `NCCL_ALGO=Ring NCCL_NVLS_ENABLE=0` plus nccl-tests
   `-R 2` (symmetric window registration) still dispatches **symmetric kernels** (`AllReduce_AGxLL_R` ≤ 8 KiB, `AllReduce_RSxLD_AGxST` above)
   at every size intra-node, because NCCL_ALGO did not gate the symmetric scheduler. Reproduced on B200 with the container's NCCL 2.30.5 (`inv_ring230_intra_*`):
   paper flags give 4.3 µs at 128 B; a real ring (`-R 0`) gives 22.0 µs. The paper's 4.7–8.9 µs "ring" latency matches the symmetric kernel.
   Inter-node, 2.30.5 does run a real RING (`inv_ring230_inter_dbg`), so only the intra "Ring" line is affected.
   NCCL 2.32.6 masks symmetric kernels when `NCCL_ALGO` is set ("No algorithm/protocol nor symKernelId available … NCCL_ALGO was set to Ring"),
   so the B200 "NCCL [Ring]" line is a genuine ring: 22–24 µs latency floor (14 dependent NVLink hops), 398 GB/s peak.

## Latency (measured with `shmem_*_latency`)
Insets of `nvshmem-p2p-bandwidth-onecol*.pdf` and `results/tables/lat_*.csv` (all scopes); `results/tables/paper_text_numbers.md` ("MEASURED" rows).
The figure configurations are the intra default env and inter IBGDA with default RC-QP count; `g`/`get` use the `_rot` perftests (see 1. below).

| one-sided latency, µs (8 trials) | 8 B | 256 B | 4 KiB | 64 KiB |
|---|---|---|---|---|
| intra put (Block) | 2.57 | 2.49 | 2.57 | 3.86 |
| intra get (Block) | 4.20 | 4.17 | 4.29 | 9.95 |
| intra p | 2.22 | 2.26 | 2.34 | 8.31 |
| intra g | 1.90 | 1.97 | 3.89 | 58.6 |
| inter put (Block), IBGDA | 20.4 | 20.4 | 20.6 | 22.1 |
| inter get (Block), IBGDA | 17.9 | 17.9 | 18.1 | 19.7 |
| inter p, IBGDA | 14.8 | 15.9 | 29.1 | 205 |
| inter g, IBGDA | 14.3 | 15.0 | 39.8 | 586 |
| inter put / get / p / g, IBRC (CPU proxy) | 13.9 / 9.9 / 8.3 / 10.1 | 13.9 / 9.9 / 21.6 / 11.0 | 14.0 / 9.9 / 270 / 53 | 15.9 / 11.5 / 4250 / 884 |

Two pitfalls, both measured:
1. **Upstream intra `shmem_g_latency` times cache hits.** On the P2P path, `nvshmem_g` is a plain load
   (`nvshmemi_g`: `return *source_actual;`), and the perftest re-reads the same remote address every iteration. It reports 0.18–0.28 µs, an L1 hit, not an
   NVLink round trip. `nvshmem_patches/shmem_{g,get}_latency_rot.cu` are the upstream files with one change: iteration *i* reads region
   `i % 256` of a 256×`max_size` buffer. With that change intra `g` is **1.9–2.0 µs** (and 58.6 vs 2.75 µs at 64 KiB). Bulk `get` changes by only 2–5% (4.2 vs 4.1 µs).
   Over IB the patched and upstream tests agree within 1%, so the artifact is specific to NVLink loads.
2. **The paper's inter-node P2P env (64 RC QPs/PE) is a bandwidth config.** Every `nvshmem_quiet` drains all RC QPs, adding ~0.9 µs per QP:
   put latency is 18.7 / 20.3 / 25.4 / 75.3 µs with 1 / 2 / 8 / 64 QPs (`tune_lat_inter_*_rc*`). In that env put/get measure 77 / 73 µs (`lat_inter_ibgda_*`).
   The figure therefore uses IBGDA's default QP count for latency.

For single operations, IBRC is lower than IBGDA on this system (put 13.9 vs 20.4 µs, g 10.1 vs 14.3 µs), but scalar `p` over IBRC
collapses with size (4.2 ms at 64 KiB vs 205 µs) because every element goes through the CPU proxy.

**Collectives** were never derived from bandwidth. The AllReduce insets plot the measured time per collective: the nccl-tests out-of-place `time`
column, and NVSHMEM's `latency(us)` from `reduction_focus` / `reduction_on_stream` (elapsed / iterations). Only algbw is derived (size / time),
as nccl-tests itself does.

## Tuned NVSHMEM configurations
Figures `*-tuned.{pdf,png}` replace the NVSHMEM series with the configurations below; NCCL series are unchanged.
Every choice comes from a single-trial sweep (`tune_*` logs) and was then re-measured with the paper's protocol (8 trials).
Tuned AllReduce envs were re-checked with `ar_check` (bitwise PASS).

| Series | Paper config (B200) | Tuned config | Peak / key number: paper → tuned |
|---|---|---|---|
| intra put | c32×t256 | c64×t1024 | 672 → **697 GB/s** |
| intra get | c32×t256 | c64×t1024 | 73 → **532 GB/s** (latency-bound with too few CTAs on B200; TMA path `--use_smem 1 NVSHMEM_TMA_POLICY=ENABLE` reaches 579 GB/s but has ~8 µs small-msg latency and bimodal noise, kept as `p2p_tunedtma_intra_get`) |
| intra p | c32×t256 | c296×t1024 | 181 → **699 GB/s** |
| intra g | c32×t256 | c296×t1024 | 18 → **584 GB/s** |
| inter put/get | IBGDA, 64 RC/PE, map=cta, c64×t1024 | 64 RC/PE, **map=warp** | 48.9/49.1 → 49.1/49.3 GB/s (line rate either way) |
| inter p | same | same | 15.7 → **17.4 GB/s** |
| inter g | same | same | 5.55 → 5.81 GB/s; 256 B latency 19.9 → 16.0 µs |
| intra AllReduce on-stream | default (16 CTAs on the NVLS path) | `NVSHMEM_MAX_CTAS=64` (+`NVSHMEM_MAX_TEAMS=1024`) | 303 → **474 GB/s** (NCCL NVLS: 479) |
| intra AllReduce device block | 1 CTA × 256 thr | — (512/1024 threads fail cooperative launch: out of resources) | 15.5 GB/s |
| inter AllReduce device block | IBRC, scratch 512 KiB | IBRC, `NVSHMEM_REDUCE_SCRATCH_SIZE=256 MiB` | 0.13 → **0.48 GB/s**; 64 KiB 2.4 ms → 154 µs |
| inter AllReduce on-stream | IBRC, scratch 512 KiB | same | ~0.14 → **0.74 GB/s**; 64 KiB 2.3 ms → 112 µs |

Things that did *not* help, as measured:
* IBGDA for the inter-node AllReduce: slower than IBRC at small sizes (8 KiB 188 µs vs 58 µs) and no better at large sizes.
* More RC QPs (16/PE) for collectives: worse small-message latency.
* A 1 GiB reduce scratch: fails to initialize (heap exhausted across teams).
* 128/256 RC QPs for P2P: lower or failed.

Inter-node NVSHMEM AllReduce stays ~3 orders of magnitude below NCCL even when tuned. The limit is the algorithm: one CTA and a segmented linear algorithm (O(M·N) traffic per PE plus a barrier per segment), not the transport.

## Files
```
repro/
├── README.md                 this file
├── analyze.py                parses results/raw → results/figures + results/tables (paper-config figures)
├── analyze_tuned.py          tuned-NVSHMEM figures + paper-vs-tuned peak table
├── nvshmem_patches/          reduction_focus.cu (paper's float-sum-block perftest, ported to 3.8);
│                             shmem_{g,get}_latency_rot.cu (upstream latency tests minus the same-address cache artifact)
├── validate/ar_check.cu      bitwise float-sum AllReduce correctness check for the plotted NVSHMEM paths
├── scripts/                  env.sh, build_all.sh, stage_node.sh, prep_hold.sh, bench.sh (+ groups_*.sh)
└── results/
    ├── raw/                  one log per (config, trial): '#BENCH' header (nodes, env, cmd) + raw output + exit code
    ├── tables/               per-series CSV (mean, stddev, n per size) + peaks_paper_vs_tuned.csv
    └── figures/              nvshmem-{p2p,allreduce}-bandwidth-onecol{,-tuned}.{pdf,png}
```
Storage: sources/scripts/results on NFS home; image + builds + binaries on node-local `/raid/scratch`; nothing
executed from Lustre (the container image is copied off Lustre once per node; `srun --export=NONE` keeps the
login shell's Lustre-hosted HPC-X environment out of the container).
