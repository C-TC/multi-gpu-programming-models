#!/usr/bin/env python3
"""Plot 1-node + 2-node jacobi sweep CSVs → PNGs in repro/jacobi/results/figures/.

Generates 6 figures (paper Fig 4.19a/b mirrors + ablation views):
  1. fig_a_runtime_vs_gpu_1node.png  — Sweep A: runtime vs GPU count, 16384^2
  2. fig_b_runtime_vs_nx_1node.png   — Sweep B: runtime vs nx, 8 GPU
  3. fig_a_speedup_vs_gpu_1node.png  — Sweep A: speedup vs single_gpu baseline
  4. fig_a_runtime_vs_gpu_2node.png  — Sweep A2: runtime vs GPU count, 2 nodes (if 2-node CSV present)
  5. fig_b_runtime_vs_nx_2node.png   — Sweep B2: runtime vs nx, 2 nodes (if present)
  6. fig_nvshmem_block_ablation.png  — nvshmem vs nvshmem+block vs nvshmem+block+nbsync
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "repro" / "jacobi" / "results"
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

VARIANT_STYLE = {
    "single_gpu":             ("k", "o"),
    "mpi":                    ("tab:blue", "s"),
    "mpi_overlap":            ("tab:cyan", "D"),
    "nccl":                   ("tab:green", "o"),
    "nccl_overlap":           ("tab:olive", "D"),
    "nccl_graphs":            ("tab:red", "^"),
    "nvshmem":                ("tab:purple", "v"),
    "nvshmem_block":          ("tab:pink", "v"),
    "nvshmem_block_nbsync":   ("tab:brown", "X"),
}


def load(pattern: str) -> list[dict]:
    cands = sorted(RESULTS.glob(pattern))
    if not cands:
        return []
    return list(csv.DictReader(cands[-1].open()))


def mean(seq: list[float]) -> float:
    return statistics.mean(seq) if seq else float("nan")


def by_key(rows: list[dict], sweep: str) -> dict:
    out: defaultdict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["sweep"] != sweep:
            continue
        out[(r["variant"], int(r["nx"]), int(r["num_gpus"]))].append(float(r["runtime_s"]))
    return out


def plot_runtime_vs_gpu(rows: list[dict], variants: list[str], out_path: Path, title: str) -> None:
    g = by_key(rows, "A" if "1node" in str(out_path) else "A2")
    fig, ax = plt.subplots(figsize=(7, 5))
    gpus = sorted({k[2] for k in g})
    for v in variants:
        ys = [mean(g.get((v, 16384, x), [])) for x in gpus]
        if all(y != y for y in ys):
            continue
        c, m = VARIANT_STYLE.get(v, ("gray", "o"))
        ax.plot(gpus, ys, label=v, color=c, marker=m)
    ax.set_xlabel("# GPUs")
    ax.set_ylabel("runtime (s, 16384²×1000 iter)")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_runtime_vs_nx(rows: list[dict], variants: list[str], out_path: Path, sweep: str, title: str) -> None:
    g = by_key(rows, sweep)
    gpus = sorted({k[2] for k in g})
    if not gpus:
        return
    gp = gpus[-1]  # use the largest GPU count present
    fig, ax = plt.subplots(figsize=(7, 5))
    nxs = sorted({k[1] for k in g})
    for v in variants:
        ys = [mean(g.get((v, nx, gp), [])) for nx in nxs]
        if all(y != y for y in ys):
            continue
        c, m = VARIANT_STYLE.get(v, ("gray", "o"))
        ax.plot(nxs, ys, label=v, color=c, marker=m)
    ax.set_xlabel("nx")
    ax.set_ylabel(f"runtime (s, ny=16384, niter=1000, {gp} GPU)")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_speedup(rows: list[dict], out_path: Path) -> None:
    g = by_key(rows, "A")
    gpus = sorted({k[2] for k in g if k[0] != "single_gpu"})
    base_seq = g.get(("single_gpu", 16384, 1), [])
    base = mean(base_seq) if base_seq else float("nan")
    if base != base:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for v in ["mpi", "nccl", "nccl_graphs", "nvshmem", "nvshmem_block_nbsync"]:
        ys = [base / mean(g.get((v, 16384, x), [])) for x in gpus]
        c, m = VARIANT_STYLE.get(v, ("gray", "o"))
        ax.plot(gpus, ys, label=v, color=c, marker=m)
    ax.plot(gpus, gpus, "k--", label="ideal", alpha=0.5)
    ax.set_xlabel("# GPUs")
    ax.set_ylabel("speedup vs single_gpu")
    ax.set_title("Speedup at 16384² (1 node)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_block_ablation(rows1: list[dict], rows2: list[dict], out_path: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, rows, sweep, label in [
        (axs[0], rows1, "A", "1 node"),
        (axs[1], rows2, "A2", "2 nodes"),
    ]:
        if not rows:
            ax.set_title(f"{label}: no data")
            continue
        g = by_key(rows, sweep)
        gpus = sorted({k[2] for k in g if k[0].startswith("nvshmem")})
        for v in ["nvshmem", "nvshmem_block", "nvshmem_block_nbsync"]:
            ys = [mean(g.get((v, 16384, x), [])) for x in gpus]
            c, m = VARIANT_STYLE.get(v, ("gray", "o"))
            ax.plot(gpus, ys, label=v, color=c, marker=m)
        ax.set_xlabel("# GPUs")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{label}: nvshmem block-comm ablation")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("runtime (s)")
    fig.suptitle("nvshmem flag ablation at 16384²×1000 iter")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(_argv: list[str]) -> None:
    rows1 = load("results-1node-*.csv")
    rows2 = load("results-2node-*.csv")
    if not rows1 and not rows2:
        sys.exit("No results CSVs found in repro/jacobi/results/")

    if rows1:
        plot_runtime_vs_gpu(rows1, ["mpi", "nccl", "nccl_graphs", "nvshmem", "nvshmem_block_nbsync"],
                            FIG / "fig_a_runtime_vs_gpu_1node.png",
                            "Runtime vs GPU count — 1 node, 16384²×1000 iter")
        plot_runtime_vs_nx(rows1, ["nccl", "nccl_graphs", "nvshmem", "nvshmem_block_nbsync"],
                           FIG / "fig_b_runtime_vs_nx_1node.png", "B",
                           "Runtime vs nx — 1 node, 8 GPU")
        plot_speedup(rows1, FIG / "fig_a_speedup_vs_gpu_1node.png")

    if rows2:
        plot_runtime_vs_gpu(rows2, ["nccl", "nccl_graphs", "nvshmem", "nvshmem_block_nbsync"],
                            FIG / "fig_a_runtime_vs_gpu_2node.png",
                            "Runtime vs GPU count — 2 nodes, 16384²×1000 iter")
        plot_runtime_vs_nx(rows2, ["nccl", "nccl_graphs", "nvshmem_block_nbsync"],
                           FIG / "fig_b_runtime_vs_nx_2node.png", "B2",
                           "Runtime vs nx — 2 nodes")

    plot_block_ablation(rows1, rows2, FIG / "fig_nvshmem_block_ablation.png")
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main(sys.argv)
