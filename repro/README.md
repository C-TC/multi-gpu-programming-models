# Reproduction work — NCCL vs NVSHMEM on H100

Three independent investigations, each in its own subdirectory:

| Subdir | Investigation | Status |
|---|---|---|
| [jacobi/](jacobi/) | Reproducing the Jacobi NCCL-vs-NVSHMEM benchmark from §4.2.2 of [`../ETH_Zürich_CADMO_Thesis_Template_v2.pdf`](../ETH_Zürich_CADMO_Thesis_Template_v2.pdf), 1-node and 2-node | ✅ done |
| [nccl_graph_ablation/](nccl_graph_ablation/) | Effect of `NCCL_GRAPH_MIXING_SUPPORT=0` on `nccl_graphs/jacobi` latency | ✅ done |
| [deepep/](deepep/) | DeepEP V1 (NVSHMEM) vs V2 (NCCL Gin) high-throughput + low-latency, 1-node and 2-node | ⚠️ partial — see [deepep/README.md](deepep/README.md) and [IBGDA_DEBUG.md](deepep/IBGDA_DEBUG.md) for what's blocked on this cluster |

If you're moving to a different cluster, **start with [NEXT_CLUSTER.md](NEXT_CLUSTER.md)** — it lists the cluster prerequisites that must hold for everything here to run, and the items that were blocked on the current cluster so you know what to validate first.

## Repository layout

```
repro/
├── README.md                  ← this file
├── NEXT_CLUSTER.md            ← what to set up / verify on a new cluster
│
├── jacobi/
│   ├── README.md              ← Jacobi reproduction guide (full methodology, vs-paper notes)
│   ├── setup_env.sh           ← env vars (NVSHMEM/NCCL/MPI paths)
│   ├── build_all.sh           ← build the 7 jacobi variants for sm_90
│   ├── run_sweep.sh           ← 1-node sweep (Sweep A + Sweep B)
│   ├── run_sweep_2node.sh     ← 2-node sweep (Sweep A2 + Sweep B2)
│   ├── analyze.py             ← csv → markdown, 1-node
│   ├── analyze_2node.py       ← csv → markdown, 2-node
│   ├── plot_results.py        ← csv → 6 figures
│   ├── recon.sh               ← one-shot toolchain inventory
│   └── results/
│       ├── REPORT_1NODE.md    ← rendered tables, 1-node
│       ├── REPORT_2NODE.md    ← rendered tables, 2-node
│       ├── results-*.csv      ← raw runtimes
│       ├── figures/           ← 6 PNGs (paper Fig 4.19a/b mirrors + ablation views)
│       └── logs/              ← build & run stderr
│
├── nccl_graph_ablation/
│   ├── REPORT.md              ← findings (~5% improvement only at small/latency-bound nx)
│   ├── run_big.sh             ← 16384², 1000 iter, 1/2/4/8 GPU, mix=1 vs mix=0
│   ├── run_small.sh           ← nx∈{128,512}, 5000 iter, latency-bound case
│   └── results/
│
└── deepep/
    ├── README.md              ← V1 vs V2 measurements + setup notes (high-throughput + low-latency)
    ├── IBGDA_DEBUG.md         ← catalog of NVSHMEM IBGDA errors found on this cluster
    ├── probe_ibgda.sh         ← inventory NIC / gdrdrv / NVSHMEM IBGDA prereqs
    ├── try_nvshmem_transports.sh ← sweep of NVSHMEM_REMOTE_TRANSPORT configs
    ├── run_2x4.sh             ← V2 ep on 2 nodes × 4 GPU (V1 needs ≥8 GPU/node)
    ├── run_2x8.sh             ← V1 internode + V2 ep + V1/V2 LL on 2 nodes × 8 GPU
    └── results/               ← all V1 / V2 stdout logs, IBGDA probe log
```

## What's in each report

| Report | Headline numbers |
|---|---|
| [jacobi/results/REPORT_1NODE.md](jacobi/results/REPORT_1NODE.md) | 1-node 8 GPU 16384² 1000 iter: nccl_graphs **0.220 s**, nvshmem **0.252 s** (NCCL ~14% faster on NVLink-only) |
| [jacobi/results/REPORT_2NODE.md](jacobi/results/REPORT_2NODE.md) | 2-node 4×4 GPU 16384² 1000 iter: NCCL **0.266 s**, NVSHMEM baseline **4.222 s** (16× slowdown without `-use_block_comm`!), NVSHMEM (-use_block_comm -nbsync) **0.285 s** (parity with NCCL) |
| [nccl_graph_ablation/REPORT.md](nccl_graph_ablation/REPORT.md) | `NCCL_GRAPH_MIXING_SUPPORT=0` saves ~1 µs per graph launch — ≤ 5% effect at small nx, lost in noise at 16384² |
| [deepep/README.md](deepep/README.md) | Single-node 8 GPU HT @ 24 SMs: V1 NVSHMEM 322 GB/s vs V2 NCCL Gin 306 GB/s. Single-node LL: V2 75 µs vs V1 116 µs (V2 faster, but V1 IBGDA fell back to NVLink) |

## Quick reproduce on this cluster

```bash
# (mistral repo) get an 8-GPU H100 node with the standard container:
cd /mnt/vast/home/tiancheng.chen/workspace/mistral
uv run python -m scripts.utils.cluster ggpus --with_container True --num_gpus 8 --exclusive True

# Inside the container:
cd /mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models
. repro/jacobi/setup_env.sh
bash repro/jacobi/build_all.sh        # ~5 min
bash repro/jacobi/run_sweep.sh        # ~25 min (1-node Sweep A+B)
python3 repro/jacobi/analyze.py > repro/jacobi/results/REPORT_1NODE.md
python3 repro/jacobi/plot_results.py
```

For 2-node and DeepEP runs, see the per-investigation READMEs.

## Branch

This work lives on the `reproduction-h100-nvshmem-vs-nccl` branch. The only changes outside `repro/` are:

* **C++14 → C++17** in every `Makefile` (CUDA 13's CCCL refuses to compile against C++14). One-liner:
  ```bash
  for d in mpi mpi_overlap multi_node_p2p multi_threaded_* nccl* nvshmem single_*; do
      sed -i 's/-std=c++14/-std=c++17/g' "$d/Makefile"
  done
  ```
