#!/usr/bin/env python3
"""Parse the unified rigorous bench logs and produce mean ± stddev plots.

File naming: bench_<config>_<scenario>_<op>_t<trial>{TAG}.log
configs:
  nccl_default      nccl_sym       nccl_nvlsoff
  nvsdev_nvlson     nvsdev_nvlsoff
  nvshost_nvlson    nvshost_nvlsoff
  p2p
ops:
  NCCL: all_reduce alltoall broadcast all_gather reduce_scatter
  NVSHMEM coll: alltoall_latency bcast_latency fcollect_latency reduction_latency reducescatter_latency
  NVSHMEM coll host: alltoall_on_stream broadcast_on_stream fcollect_on_stream reduction_on_stream reducescatter_on_stream
  NVSHMEM P2P: shmem_g_bw shmem_get_bw shmem_p_bw shmem_put_bw shmem_st_bw shmem_atomic_bw
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
FIG.mkdir(parents=True, exist_ok=True)
TAG = "-newcluster-20260504"
N_TRIAL = 8


def parse_nccl(path: Path) -> dict[int, float]:
    """nccl-tests row: size count type op root time algbw busbw #wrong (in-place repeats); take out-of-place time."""
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
    """NVSHMEM coll perftest: pick (block scope, 32-bit) for non-reduction; (thread, int32+sum) for reduction.
    Coll layouts:
      (A) device alltoall/bcast/fcollect: size count type scope latency algbw busbw  (7 cols)
      (B) device reduction/reducescatter: size count type redop scope latency algbw busbw  (8 cols)
      (C) host on_stream alltoall/bcast/fcollect: size count type latency min_lat max_lat algbw busbw  (8 cols)
      (D) host on_stream reduction/reducescatter: size count type redop latency min_lat max_lat algbw busbw  (9 cols)
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
            if "redop" in hdr:
                layout = "D" if "min_lat(us)" in line else "B"
            elif "min_lat(us)" in line:
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
            elif layout == "B" and len(f) >= 6 and f[2] == "int32" and f[3] == "sum" and f[4] == "t":
                rows[size] = float(f[5])
            elif layout == "C" and len(f) >= 4 and f[2] == "int":
                rows[size] = float(f[3])
            elif layout == "D" and len(f) >= 5 and f[2] == "int" and f[3] == "sum":
                rows[size] = float(f[4])
    return rows


def parse_nvshmem_p2p(path: Path) -> dict[int, float]:
    """P2P device perftest: size scope BW(GB/sec) — return BW."""
    if not path.exists():
        return {}
    rows = {}
    in_table = False
    for line in path.read_text(errors="ignore").splitlines():
        if "size(B)" in line:
            in_table = True
            continue
        if in_table:
            f = line.split()
            if not f or not f[0].isdigit():
                if line.strip() == "" or line.startswith("Runtime") or line.startswith("Note"):
                    in_table = False
                continue
            try:
                size = int(f[0]); rows[size] = float(f[-1])
            except ValueError:
                pass
    return rows


def collect_trials(prefix: str, parser) -> dict[int, list[float]]:
    """Aggregate N_TRIAL trial files matching f'{prefix}_t{trial}'."""
    out = defaultdict(list)
    for trial in range(1, N_TRIAL + 1):
        log = RES / f"{prefix}_t{trial}{TAG}.log"
        for sz, v in parser(log).items():
            out[sz].append(v)
    return out


def stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


# ---------- per-collective comparison plot ----------
NCCL_OPS = ["all_reduce", "alltoall", "broadcast", "all_gather", "reduce_scatter"]
NVS_DEV_OPS = {"all_reduce": "reduction_latency", "alltoall": "alltoall_latency",
               "broadcast": "bcast_latency", "all_gather": "fcollect_latency",
               "reduce_scatter": "reducescatter_latency"}
NVS_HOST_OPS = {"all_reduce": "reduction_on_stream", "alltoall": "alltoall_on_stream",
                "broadcast": "broadcast_on_stream", "all_gather": "fcollect_on_stream",
                "reduce_scatter": "reducescatter_on_stream"}


def plot_collective_comparison(op: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plotted_any = False
    for ax, scen, scen_label in [(axes[0], "intra", "intranode 1×8"),
                                 (axes[1], "inter", "internode 2×8 (16 ranks)")]:
        series = []
        # NCCL three configs
        for cfg, label, color, ls in [
            ("nccl_default", "NCCL default (-R 0, NVLS+autotuner)", "tab:green", "--"),
            ("nccl_sym",     "NCCL sym kernel (-R 2, ncclSymmetricTaskScheduler)", "tab:blue", "-"),
            ("nccl_nvlsoff", "NCCL no-NVLS (-R 0, NCCL_NVLS_ENABLE=0)", "tab:red", "--"),
        ]:
            d = collect_trials(f"bench_{cfg}_{scen}_{op}", parse_nccl)
            if not d:
                continue
            sizes = sorted(d.keys())
            means = [stats(d[s])[0] for s in sizes]
            sds = [stats(d[s])[1] for s in sizes]
            ax.errorbar(sizes, means, yerr=sds, marker="s", ms=3, capsize=2,
                        label=label, color=color, linestyle=ls)
            plotted_any = True

        # NVSHMEM device + host (NVLS on)
        nvs_op = NVS_DEV_OPS.get(op)
        if nvs_op:
            d = collect_trials(f"bench_nvsdev_nvlson_{scen}_{nvs_op}", parse_nvshmem_coll)
            if d:
                sizes = sorted(d.keys())
                means = [stats(d[s])[0] for s in sizes]
                sds = [stats(d[s])[1] for s in sizes]
                ax.errorbar(sizes, means, yerr=sds, marker="o", ms=3, capsize=2,
                            label="NVSHMEM device kernel (NVLS on)", color="tab:purple", linestyle="-")
                plotted_any = True
            d_off = collect_trials(f"bench_nvsdev_nvlsoff_{scen}_{nvs_op}", parse_nvshmem_coll)
            if d_off:
                sizes = sorted(d_off.keys())
                means = [stats(d_off[s])[0] for s in sizes]
                sds = [stats(d_off[s])[1] for s in sizes]
                ax.errorbar(sizes, means, yerr=sds, marker="o", ms=3, mfc="none", capsize=2,
                            label="NVSHMEM device (NVLS off)", color="tab:pink", linestyle="-")
                plotted_any = True
        nvs_h = NVS_HOST_OPS.get(op)
        if nvs_h:
            d = collect_trials(f"bench_nvshost_nvlson_{scen}_{nvs_h}", parse_nvshmem_coll)
            if d:
                sizes = sorted(d.keys())
                means = [stats(d[s])[0] for s in sizes]
                sds = [stats(d[s])[1] for s in sizes]
                ax.errorbar(sizes, means, yerr=sds, marker="^", ms=3, capsize=2,
                            label="NVSHMEM host on_stream (NVLS on)", color="tab:orange", linestyle="-.")
                plotted_any = True

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("message size (B)"); ax.set_ylabel("latency (µs, mean ± stddev)")
        ax.set_title(f"{op} — {scen_label}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    if not plotted_any:
        plt.close(fig); return
    fig.suptitle(
        f"NCCL 2.30.4 vs NVSHMEM 3.3.9-ibp on H200 — {op}\n"
        f"Each point: mean ± stddev over {N_TRIAL} trials of (20 warmup + 50 timed iters); "
        f"NVSHMEM_USE_NCCL=OFF in this build (no NCCL fallback in NVSHMEM)",
        y=1.04, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"compare_{op}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_p2p_summary():
    """6 P2P APIs intra+inter (BW vs message size)."""
    P2P_APIS = ["shmem_g_bw", "shmem_get_bw", "shmem_p_bw", "shmem_put_bw", "shmem_st_bw", "shmem_atomic_bw"]
    P2P_LABEL = {
        "shmem_g_bw": "g (single-elem get)",
        "shmem_get_bw": "get (block-coop bulk)",
        "shmem_p_bw": "p (single-elem put)",
        "shmem_put_bw": "put (block-coop bulk)",
        "shmem_st_bw": "st (mapped-store, NVL only)",
        "shmem_atomic_bw": "atomic_inc (block-coop)",
    }
    COLORS = plt.cm.tab10.colors
    for scen in ["intra", "inter"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False
        for i, api in enumerate(P2P_APIS):
            d = collect_trials(f"bench_p2p_{scen}_{api}", parse_nvshmem_p2p)
            if not d: continue
            sizes = sorted(d.keys())
            means = [stats(d[s])[0] for s in sizes]
            sds = [stats(d[s])[1] for s in sizes]
            ax.errorbar(sizes, means, yerr=sds, marker="o", ms=3, capsize=2,
                        label=P2P_LABEL[api], color=COLORS[i])
            plotted = True
        if not plotted:
            plt.close(fig); continue
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("message size (B)"); ax.set_ylabel("BW (GB/s, mean ± stddev)")
        scen_label = "intranode 1×2 (NVLink)" if scen == "intra" else "internode 2×1 (IB IBGDA)"
        ax.set_title(f"NVSHMEM 3.3.9-ibp P2P device APIs — {scen_label}\n"
                     f"32 CTAs × 256 TPB; mean ± stddev over {N_TRIAL} trials")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / f"p2p_{scen}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def text_table_for_op(op: str, scen: str = "intra"):
    """Print a compact text table for one op + scenario."""
    print(f"\n=== {op} {scen} (N={N_TRIAL} trials × 20 warmup + 50 timed iters) ===")
    nvs_op = NVS_DEV_OPS.get(op)
    nvs_h = NVS_HOST_OPS.get(op)
    cols = [
        (f"bench_nccl_default_{scen}_{op}", parse_nccl, "NCCL_def"),
        (f"bench_nccl_sym_{scen}_{op}", parse_nccl, "NCCL_sym"),
        (f"bench_nccl_nvlsoff_{scen}_{op}", parse_nccl, "NCCL_nvls_off"),
    ]
    if nvs_op:
        cols.append((f"bench_nvsdev_nvlson_{scen}_{nvs_op}", parse_nvshmem_coll, "NVSdev"))
    if nvs_h:
        cols.append((f"bench_nvshost_nvlson_{scen}_{nvs_h}", parse_nvshmem_coll, "NVShost"))

    data = [(label, collect_trials(prefix, parser)) for prefix, parser, label in cols]
    sizes = sorted({sz for _, d in data for sz in d})
    print(f"{'size(B)':>10} | " + " | ".join(f"{l:>14s}" for l, _ in data))
    print("-" * (12 + 16 * len(data)))
    for sz in sizes:
        cells = []
        for label, d in data:
            vals = d.get(sz, [])
            if not vals:
                cells.append(f"{'—':>14}"); continue
            mean, sd = stats(vals)
            cells.append(f"{mean:6.2f}±{sd:5.2f}")
        print(f"{sz:>10} | " + " | ".join(c if len(c) >= 14 else f"{c:>14}" for c in cells))


if __name__ == "__main__":
    print(f"Analyzing rigorous bench data in {RES}")
    for op in NCCL_OPS:
        plot_collective_comparison(op)
        text_table_for_op(op, "intra")
    plot_p2p_summary()
    print(f"\nPlots: {FIG}")
