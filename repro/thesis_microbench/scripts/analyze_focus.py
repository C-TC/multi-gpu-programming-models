#!/usr/bin/env python3
"""Analyse the focused all_reduce comparison.

Reads bench_focus_<config>_<scenario>_all_reduce_t<trial>.log and emits
a single 1×2 grid (intra | inter) plus a text comparison table.

Configs:
  focus_nccl              NCCL recipe (-G 10 -R 2 -c 0 -w 50 -n 100, GRAPH_MIXING_SUPPORT=0)
  focus_nccl_ring         same, plus NCCL_NVLS_ENABLE=0 NCCL_ALGO=Ring
  focus_nvsdev            NVSHMEM device reduction_latency (parser extracts int32-sum-block)
  focus_nvshost           NVSHMEM host reduction_on_stream (parser extracts int-sum)
"""

import statistics
from collections import defaultdict
from pathlib import Path

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
            rows[size] = float(f[5])
        except (ValueError, IndexError):
            pass
    return rows


def parse_nvshmem_coll(path: Path) -> dict[int, float]:
    """Extract the row we care about for the all_reduce comparison.

    Layout B (device reduction): filter <dtype>-sum-block where dtype is one of
        {float, int32}. The new `reduction_focus` binary emits only float-sum-b;
        the older `reduction_latency` rigorous bench logs (used as fallback for
        nvsdev_inter when reduction_focus wasn't available) emit int32-sum-b.
    Layout D (host reduction_on_stream): filter int-sum.
    """
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
            has_min_lat = any("min_lat" in h for h in hdr)
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
            if (
                layout == "B"
                and len(f) >= 6
                and f[2] in ("float", "int32")
                and f[3] == "sum"
                and f[4] == "b"
            ):
                rows[size] = float(f[5])
            elif layout == "D" and len(f) >= 5 and f[2] in ("int", "float") and f[3] == "sum":
                rows[size] = float(f[4])
    return rows


def collect(prefix: str, parser) -> dict[int, list[float]]:
    out: dict[int, list[float]] = defaultdict(list)
    for trial in range(1, N_TRIAL + 1):
        log = RES / f"bench_{prefix}_t{trial}{TAG}.log"
        for sz, v in parser(log).items():
            out[sz].append(v)
    return out


# nvsdev_inter is now collected directly with the patched `reduction_focus` binary
# (float-sum-block only, no thread/warp pre-pass). The previous alias to the
# rigorous-bench int32-sum-block logs is no longer needed.
NVSDEV_INTER_ALIAS = None


def stats(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


CONFIGS = [
    ("focus_nccl",      "NCCL recipe (R 2 / G 10)",      "tab:blue",   "-",  parse_nccl),
    ("focus_nccl_ring", "NCCL legacy ring (NVLS off)",   "tab:cyan",   "--", parse_nccl),
    ("focus_nvsdev",    "NVSHMEM device block",          "tab:purple", "-",  parse_nvshmem_coll),
    ("focus_nvshost",   "NVSHMEM host on_stream",        "tab:orange", "-.", parse_nvshmem_coll),
]


def plot_focus() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for col, (scen, scen_label) in enumerate([("intra", "intranode 1×8"), ("inter", "internode 2×8")]):
        ax = axes[col]
        for prefix_root, label, color, ls, parser in CONFIGS:
            if prefix_root == "focus_nvsdev" and scen == "inter" and NVSDEV_INTER_ALIAS:
                prefix = NVSDEV_INTER_ALIAS
            else:
                prefix = f"{prefix_root}_{scen}_all_reduce"
            d = collect(prefix, parser)
            if not d:
                continue
            sizes = sorted(d.keys())
            means = [stats(d[s])[0] for s in sizes]
            sds = [stats(d[s])[1] for s in sizes]
            ax.errorbar(sizes, means, yerr=sds, marker="o", ms=3, capsize=2,
                        label=label, color=color, linestyle=ls)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"all_reduce — {scen_label}", fontsize=11)
        ax.set_xlabel("size (B)")
        if col == 0:
            ax.set_ylabel("latency (µs)")
            ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(
        "Focused all_reduce comparison (8 trials, mean ± stddev) — H200 cluster\n"
        "NCCL: -b 128 -e 1G -w 50 -n 100 -c 0 -R 2 -G 10, NCCL_GRAPH_MIXING_SUPPORT=0  |  "
        "NVSHMEM device reduction_focus (float-sum-block, --cudagraph): intra -n 100 -w 50 to 1 GiB; "
        "inter -n 5 -w 2 to 128 MiB\n"
        "NVSHMEM host reduction_on_stream (int-sum, --cudagraph -n 10 -w 3): intra to 16 MiB; inter to 64 KiB",
        y=1.02, fontsize=9)
    fig.tight_layout()
    out = FIG / "focus_all_reduce.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def text_table() -> None:
    print(f"\n=== Focused all_reduce (N={N_TRIAL} trials) ===")
    for scen in ["intra", "inter"]:
        print(f"\n{scen}:")
        data = []
        for prefix_root, label, _color, _ls, parser in CONFIGS:
            if prefix_root == "focus_nvsdev" and scen == "inter" and NVSDEV_INTER_ALIAS:
                prefix = NVSDEV_INTER_ALIAS
            else:
                prefix = f"{prefix_root}_{scen}_all_reduce"
            data.append((label, collect(prefix, parser)))
        sizes = sorted({sz for _, d in data for sz in d})
        print(f"  {'size':>10} | " + " | ".join(f"{l:>22s}" for l, _ in data))
        for sz in sizes:
            cells = []
            for _, d in data:
                vals = d.get(sz, [])
                if not vals:
                    cells.append(f"{'—':>22}")
                else:
                    m, s = stats(vals)
                    cells.append(f"{m:11.2f}±{s:8.2f}")
            print(f"  {sz:>10} | " + " | ".join(cells))


if __name__ == "__main__":
    plot_focus()
    text_table()
