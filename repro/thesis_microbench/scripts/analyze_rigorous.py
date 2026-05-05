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


def plot_overview_per_scenario(scen: str, n_ranks: int):
    """All 5 collectives × 3 NCCL configs in a 5-panel grid for one scenario.
    Lets you eyeball which collective benefits most from sym kernel / NVLS."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
    plotted_any = False
    for idx, op in enumerate(NCCL_OPS):
        ax = axes[idx]
        for cfg, label, color, ls in [
            ("nccl_default", "NCCL default", "tab:green", "-"),
            ("nccl_sym",     "NCCL sym (-R 2)", "tab:blue", "-"),
            ("nccl_nvlsoff", "NCCL NVLS off", "tab:red", "--"),
        ]:
            d = collect_trials(f"bench_{cfg}_{scen}_{op}", parse_nccl)
            if not d: continue
            sizes = sorted(d.keys())
            means = [stats(d[s])[0] for s in sizes]
            sds = [stats(d[s])[1] for s in sizes]
            ax.errorbar(sizes, means, yerr=sds, marker="o", ms=3, capsize=2,
                        label=label, color=color, linestyle=ls)
            plotted_any = True
        # Add NVSHMEM device on top
        nvs_op = NVS_DEV_OPS.get(op)
        if nvs_op:
            d = collect_trials(f"bench_nvsdev_nvlson_{scen}_{nvs_op}", parse_nvshmem_coll)
            if d:
                sizes = sorted(d.keys())
                means = [stats(d[s])[0] for s in sizes]
                sds = [stats(d[s])[1] for s in sizes]
                ax.errorbar(sizes, means, yerr=sds, marker="s", ms=3, capsize=2,
                            label="NVSHMEM device", color="tab:purple", linestyle=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(op)
        ax.set_xlabel("size (B)");
        if idx == 0: ax.set_ylabel("latency (µs)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    if not plotted_any:
        plt.close(fig); return
    scen_label = f"intranode 1×{n_ranks}" if scen == "intra" else f"internode 2×{n_ranks // 2}"
    fig.suptitle(f"All 5 NCCL collectives — {scen_label} (mean ± stddev, 8 trials each)", y=1.04, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"overview_{scen}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sym_speedup_summary():
    """For each NCCL op + scenario, show sym/default speedup ratio across sizes.
    Heatmap-like grid: rows=ops, cols=sizes, color=speedup."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax_idx, scen in enumerate(["intra", "inter"]):
        ax = axes[ax_idx]
        for op in NCCL_OPS:
            d_def = collect_trials(f"bench_nccl_default_{scen}_{op}", parse_nccl)
            d_sym = collect_trials(f"bench_nccl_sym_{scen}_{op}", parse_nccl)
            sizes = sorted(set(d_def.keys()) & set(d_sym.keys()))
            ratios = []
            for sz in sizes:
                m_def, _ = stats(d_def.get(sz, []))
                m_sym, _ = stats(d_sym.get(sz, []))
                if m_def and m_sym:
                    ratios.append(m_sym / m_def)
                else:
                    ratios.append(float("nan"))
            ax.plot(sizes, ratios, marker="o", ms=4, label=op)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_ylabel(f"sym/default ratio\n(< 1 = sym faster)")
        ax.set_title(f"{scen}node — NCCL sym kernel (-R 2) speedup vs default (-R 0)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, ncol=5)
    axes[1].set_xlabel("message size (B)")
    fig.suptitle("Where does the symmetric kernel help? (intra and inter, 5 collectives)", y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "sym_speedup_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nvls_impact():
    """For each NCCL op + scenario, show nvls_off/default ratio across sizes.
    Quantifies how much NVLS multicast contributes to NCCL on H200."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax_idx, scen in enumerate(["intra", "inter"]):
        ax = axes[ax_idx]
        for op in NCCL_OPS:
            d_def = collect_trials(f"bench_nccl_default_{scen}_{op}", parse_nccl)
            d_off = collect_trials(f"bench_nccl_nvlsoff_{scen}_{op}", parse_nccl)
            sizes = sorted(set(d_def.keys()) & set(d_off.keys()))
            ratios = []
            for sz in sizes:
                m_def, _ = stats(d_def.get(sz, []))
                m_off, _ = stats(d_off.get(sz, []))
                if m_def and m_off:
                    ratios.append(m_off / m_def)
                else:
                    ratios.append(float("nan"))
            ax.plot(sizes, ratios, marker="o", ms=4, label=op)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_ylabel(f"NVLS_off/default ratio\n(> 1 = NVLS helps)")
        ax.set_title(f"{scen}node — NCCL NVLS multicast contribution (NCCL_NVLS_ENABLE=0 vs default)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, ncol=5)
    axes[1].set_xlabel("message size (B)")
    fig.suptitle("Where does NVLink-SHARP help? (intra: yes for some ops; inter: not applicable, ~1×)", y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "nvls_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nvshmem_nvls_impact():
    """NVSHMEM device collectives: NVLS off / NVLS on ratio."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    nvs_ops = list(NVS_DEV_OPS.values())
    for ax_idx, scen in enumerate(["intra", "inter"]):
        ax = axes[ax_idx]
        for op in nvs_ops:
            d_on = collect_trials(f"bench_nvsdev_nvlson_{scen}_{op}", parse_nvshmem_coll)
            d_off = collect_trials(f"bench_nvsdev_nvlsoff_{scen}_{op}", parse_nvshmem_coll)
            sizes = sorted(set(d_on.keys()) & set(d_off.keys()))
            ratios = []
            for sz in sizes:
                m_on, _ = stats(d_on.get(sz, []))
                m_off, _ = stats(d_off.get(sz, []))
                if m_on and m_off:
                    ratios.append(m_off / m_on)
                else:
                    ratios.append(float("nan"))
            ax.plot(sizes, ratios, marker="o", ms=4, label=op.replace("_latency", ""))
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_ylabel(f"NVLS_off/NVLS_on ratio\n(> 1 = NVLS helps NVSHMEM)")
        ax.set_title(f"{scen}node — NVSHMEM device collective: NVSHMEM_DISABLE_NVLS=1 vs default")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, ncol=5)
    axes[1].set_xlabel("message size (B)")
    fig.suptitle("NVSHMEM device collectives: do they actually use NVLS? (most: no — perftest's block-scope kernels don't engage multicast)",
                 y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "nvshmem_nvls_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grand_summary():
    """One figure summarising NCCL default vs NCCL sym vs NVSHMEM device for all 5 collectives,
    intra and inter — 10 panels total. Lets you scan everything at once."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey="row")
    for col, op in enumerate(NCCL_OPS):
        for row, (scen, scen_label) in enumerate([("intra", "intra 1×8"), ("inter", "inter 2×8")]):
            ax = axes[row, col]
            for cfg, label, color, ls in [
                ("nccl_default", "NCCL default (NVLS+autotuner)", "tab:green", "-"),
                ("nccl_sym",     "NCCL sym (-R 2)", "tab:blue", "-"),
                ("nccl_nvlsoff", "NCCL NVLS off", "tab:red", "--"),
            ]:
                d = collect_trials(f"bench_{cfg}_{scen}_{op}", parse_nccl)
                if not d: continue
                sizes = sorted(d.keys())
                means = [stats(d[s])[0] for s in sizes]
                sds = [stats(d[s])[1] for s in sizes]
                ax.errorbar(sizes, means, yerr=sds, marker="o", ms=2, capsize=1, lw=0.8,
                            label=label, color=color, linestyle=ls)
            nvs_op = NVS_DEV_OPS.get(op)
            if nvs_op:
                d = collect_trials(f"bench_nvsdev_nvlson_{scen}_{nvs_op}", parse_nvshmem_coll)
                if d:
                    sizes = sorted(d.keys())
                    means = [stats(d[s])[0] for s in sizes]
                    sds = [stats(d[s])[1] for s in sizes]
                    ax.errorbar(sizes, means, yerr=sds, marker="s", ms=2, capsize=1, lw=0.8,
                                label="NVSHMEM device", color="tab:purple", linestyle=":")
            ax.set_xscale("log"); ax.set_yscale("log")
            if row == 0: ax.set_title(op, fontsize=10)
            if col == 0: ax.set_ylabel(f"{scen_label}\nlatency (µs)")
            if row == 1: ax.set_xlabel("size (B)")
            ax.grid(True, which="both", alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Grand summary: NCCL 2.30.4 vs NVSHMEM 3.3.9-ibp on H200 — all 5 collectives × {intra, inter}\n"
                 "8 trials × (20 warmup + 50 timed iters) per data point; errorbars = stddev",
                 y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "grand_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Analyzing rigorous bench data in {RES}")
    for op in NCCL_OPS:
        plot_collective_comparison(op)
        text_table_for_op(op, "intra")
    plot_p2p_summary()
    plot_overview_per_scenario("intra", 8)
    plot_overview_per_scenario("inter", 16)
    plot_sym_speedup_summary()
    plot_nvls_impact()
    plot_nvshmem_nvls_impact()
    plot_grand_summary()
    print(f"\nPlots: {FIG}")
