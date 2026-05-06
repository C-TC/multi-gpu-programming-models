#!/usr/bin/env python3
"""Analyse the fair NCCL-recipe benchmark.

Reads bench_fair_<config>_<scenario>_<op>_t<trial>.log and emits a single
3×2 grid figure plus a text comparison table.

Configs:
  fair_nccl              NCCL with -G 10 -R 2 -c 0 -w 50 -n 100, GRAPH_MIXING_SUPPORT=0
  fair_nccl_nvlsoff      same but NCCL_NVLS_ENABLE=0
  fair_nvshmem           NVSHMEM device coll with --cudagraph -w 50 -n 100
"""

import re
import statistics
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "repro" / "thesis_microbench" / "results"
FIG = RES / "figures"
TAG = "-newcluster-20260504"
N_TRIAL = 8


def parse_nccl(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        f = s.split()
        if len(f) < 8 or not f[0].isdigit():
            continue
        size = int(f[0])
        if size == 0:
            continue
        try:
            rows[size] = float(f[5])  # out-of-place time (us)
        except (ValueError, IndexError):
            pass
    return rows


def parse_nvshmem_coll(path: Path) -> dict[int, float]:
    """Handle all 4 NVSHMEM coll perftest layouts:
       A) device alltoall/bcast/fcollect: size count type scope latency algbw busbw  (filter: 32-bit, block)
       B) device reduction/reducescatter: size count type redop scope latency algbw busbw  (filter: int32 sum b)
       C) host on_stream alltoall/bcast/fcollect: size count type latency min_lat max_lat algbw busbw  (filter: type=int)
       D) host on_stream reduction/reducescatter: size count type redop latency min_lat max_lat algbw busbw (int sum)"""
    if not path.exists():
        return {}
    rows = {}
    in_table = False
    layout = None
    for line in path.read_text(errors="ignore").splitlines():
        if "size(B)" in line:
            in_table = True
            hdr = line.split()
            has_redop = "redop" in hdr
            has_min_lat = "min_lat(us)" in line or any("min_lat" in h for h in hdr)
            if has_redop and has_min_lat:
                layout = "D"
            elif has_redop:
                layout = "B"
            elif has_min_lat:
                layout = "C"
            else:
                layout = "A"
            continue
        if in_table:
            f = line.split()
            if not f or not f[0].isdigit():
                if line.strip() == "" or line.startswith("Runtime") or line.startswith("Note"):
                    in_table = False
                continue
            try:
                size = int(f[0])
            except ValueError:
                continue
            if layout == "A" and len(f) >= 5 and f[2] == "32-bit" and f[3] == "block":
                rows[size] = float(f[4])
            elif layout == "B" and len(f) >= 6 and f[2] == "int32" and f[3] == "sum" and f[4] == "b":
                rows[size] = float(f[5])
            elif layout == "C" and len(f) >= 4 and f[2] == "int":
                rows[size] = float(f[3])
            elif layout == "D" and len(f) >= 5 and f[2] == "int" and f[3] == "sum":
                rows[size] = float(f[4])
    return rows


def collect(prefix: str, parser) -> dict[int, list[float]]:
    out = defaultdict(list)
    for trial in range(1, N_TRIAL + 1):
        log = RES / f"{prefix}_t{trial}{TAG}.log"
        for sz, v in parser(log).items():
            out[sz].append(v)
    return out


def stats(vals):
    if not vals: return (float("nan"), float("nan"))
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


OPS = ["all_reduce", "alltoall", "broadcast"]


def plot_fair():
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey="row")
    for col, op in enumerate(OPS):
        for row, (scen, scen_label) in enumerate([("intra", "intranode 1×8"), ("inter", "internode 2×8")]):
            ax = axes[row, col]
            for prefix, label, color, ls in [
                (f"bench_fair_nccl_{scen}_{op}",              "NCCL recipe (-R 2 -G 10 -c 0)",          "tab:blue",   "-"),
                (f"bench_fair_nccl_nvlsoff_{scen}_{op}",      "NCCL recipe + NVLS off",                 "tab:red",    "--"),
                (f"bench_fair_nvshmem_{scen}_{op}",           "NVSHMEM device + --cudagraph",           "tab:purple", "-"),
                (f"bench_fair_nvshmem_host_{scen}_{op}",      "NVSHMEM host on_stream + --cudagraph",   "tab:orange", "-."),
            ]:
                parser = parse_nvshmem_coll if "nvshmem" in prefix else parse_nccl
                d = collect(prefix, parser)
                if not d:
                    continue
                sizes = sorted(d.keys())
                means = [stats(d[s])[0] for s in sizes]
                sds = [stats(d[s])[1] for s in sizes]
                ax.errorbar(sizes, means, yerr=sds, marker="o", ms=3, capsize=2,
                            label=label, color=color, linestyle=ls)
            ax.set_xscale("log"); ax.set_yscale("log")
            if row == 0: ax.set_title(op, fontsize=11)
            if col == 0: ax.set_ylabel(f"{scen_label}\nlatency (µs)")
            if row == 1: ax.set_xlabel("size (B)")
            ax.grid(True, which="both", alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Fair recipe — NCCL (-R 2 -G 10 -c 0 -w 50 -n 100, GRAPH_MIXING_SUPPORT=0) vs\n"
        "NVSHMEM device kernel (--cudagraph) and NVSHMEM host on_stream (--cudagraph) — 8 trials, mean ± stddev\n"
        "NVSHMEM built with NVSHMEM_USE_NCCL=OFF (no NCCL fallback in NVSHMEM hot path; verified via nm + ldd)",
        y=0.995, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fair_recipe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG / 'fair_recipe.png'}")


def text_table():
    print(f"\n=== Fair recipe (N={N_TRIAL} trials × (50 warmup + 100 timed iters), -G 10 / --cudagraph) ===")
    for op in OPS:
        for scen in ["intra", "inter"]:
            print(f"\n{op} {scen}:")
            cols = [
                (f"bench_fair_nccl_{scen}_{op}", parse_nccl, "NCCL"),
                (f"bench_fair_nccl_nvlsoff_{scen}_{op}", parse_nccl, "NCCL_nvlsoff"),
                (f"bench_fair_nvshmem_{scen}_{op}", parse_nvshmem_coll, "NVSdev"),
                (f"bench_fair_nvshmem_host_{scen}_{op}", parse_nvshmem_coll, "NVShost"),
            ]
            data = [(label, collect(prefix, parser)) for prefix, parser, label in cols]
            sizes = sorted({sz for _, d in data for sz in d})
            print(f"  {'size':>10} | " + " | ".join(f"{l:>15s}" for l, _ in data))
            for sz in sizes:
                cells = []
                for _, d in data:
                    vals = d.get(sz, [])
                    if not vals:
                        cells.append(f"{'—':>15}")
                    else:
                        m, s = stats(vals)
                        cells.append(f"{m:7.2f}±{s:5.2f}")
                print(f"  {sz:>10} | " + " | ".join(cells))


if __name__ == "__main__":
    plot_fair()
    text_table()
