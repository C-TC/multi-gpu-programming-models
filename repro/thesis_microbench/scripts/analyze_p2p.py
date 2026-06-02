#!/usr/bin/env python3
"""Compare NVSHMEM P2P transport: default (ibrc proxy) vs IBGDA (kernel-init RDMA).

Reads bench_p2p_<transport>_<scenario>_<api>_t<trial>.log and renders BW vs size
for each of {g, get, p, put} × intra+inter. The original bench used only the
default `ibrc` transport (the env vars to engage IBGDA were missing) so cross-
node small-message rate was limited by the host-CPU proxy poll loop. The
ibgda variant adds NVSHMEM_IB_ENABLE_IBGDA=1 + NVSHMEM_HCA_PREFIX= +
NVSHMEM_DISABLE_NVLS=1 and the kernel posts WQEs directly into NIC-mapped
GPU memory.
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

APIS = [
    ("shmem_g_bw", "g (scalar get)"),
    ("shmem_get_bw", "get (bulk)"),
    ("shmem_p_bw", "p (scalar put)"),
    ("shmem_put_bw", "put (bulk)"),
    ("shmem_st_bw", "st (mapped store)"),
    ("shmem_atomic_bw", "atomic_inc"),
]


def parse(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    rows = {}
    in_table = False
    for line in path.read_text(errors="ignore").splitlines():
        if "size(B)" in line and "BW" in line:
            in_table = True
            continue
        if not in_table:
            continue
        f = line.split()
        if not f or not f[0].isdigit():
            if line.strip() == "" or line.startswith("Runtime") or line.startswith("Note"):
                in_table = False
            continue
        try:
            size = int(f[0])
            bw = float(f[2])
        except (ValueError, IndexError):
            continue
        rows[size] = bw
    return rows


def collect(prefix: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = defaultdict(list)
    for trial in range(1, N_TRIAL + 1):
        log = RES / f"bench_{prefix}_t{trial}{TAG}.log"
        for sz, v in parse(log).items():
            out[sz].append(v)
    return out


def stats(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def plot_p2p() -> None:
    fig, axes = plt.subplots(2, 6, figsize=(28, 9), sharex=True)
    for col, (api, api_label) in enumerate(APIS):
        for row, (scen, scen_label) in enumerate([("intra", "intranode 1×2"), ("inter", "internode 2×1")]):
            ax = axes[row, col]
            any_data = False
            series = [
                ("p2p", "tab:orange", "o", "ibrc (default = CPU proxy)"),
                ("p2p_ibgda", "tab:green", "s", "IBGDA, 1 RC/PE, c32×t256 (untuned)"),
            ]
            # tuned IBGDA only exists for inter (NVLink intra doesn't use RC QPs)
            if scen == "inter":
                series.append(("p2p_ibgdatuned", "tab:blue", "^",
                               "IBGDA tuned: 64 RC/PE, c64×t1024"))
            for transport, color, marker, label in series:
                d = collect(f"{transport}_{scen}_{api}")
                if not d:
                    continue
                any_data = True
                sizes = sorted(d.keys())
                means = [stats(d[s])[0] for s in sizes]
                sds = [stats(d[s])[1] for s in sizes]
                ax.errorbar(sizes, means, yerr=sds, marker=marker, ms=4, capsize=2,
                            label=label, color=color)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            if row == 0: ax.set_title(api_label, fontsize=11)
            if col == 0: ax.set_ylabel(f"{scen_label}\nBW (GB/s)")
            if row == 1: ax.set_xlabel("size (B)")
            if col == 0: ax.legend(fontsize=8, loc="lower right")
            # Annotate the empty st-inter cells with a short note instead of leaving blank
            if not any_data and api == "shmem_st_bw" and scen == "inter":
                ax.text(0.5, 0.5, "n/a — peer LD/ST is\nintra-NVLink only", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
    fig.suptitle(
        "NVSHMEM P2P: ibrc (CPU-proxy) vs IBGDA untuned (1 RC/PE) vs IBGDA tuned (64 RC/PE, c64×t1024) — H200, 8 trials, mean ± stddev\n"
        "Intra: NVLink P2P (transport/RC-QP irrelevant — lines overlap).  Inter: scalar p/g are concurrency-bound — "
        "throughput ∝ (#RC QPs × #issuing threads). Bulk put/get already saturate the NIC (~48 GB/s) so tuning doesn't move them.",
        y=1.02, fontsize=10)
    fig.tight_layout()
    out = FIG / "p2p_ibrc_vs_ibgda.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def text_table() -> None:
    print(f"\n=== P2P inter: ibrc vs IBGDA-untuned vs IBGDA-tuned (mean GB/s) ===")
    for api, _ in APIS:
        data_a = collect(f"p2p_inter_{api}")
        data_b = collect(f"p2p_ibgda_inter_{api}")
        data_c = collect(f"p2p_ibgdatuned_inter_{api}")
        sizes = sorted(set(data_a) | set(data_b) | set(data_c))
        if not sizes:
            print(f"\n{api}: (no data — st inter is n/a)")
            continue
        print(f"\n{api}:")
        print(f"  {'size':>10} | {'ibrc':>10s} | {'IBGDA':>10s} | {'IBGDAtuned':>10s} | tuned/untuned")
        for sz in sizes:
            a, b, c = (stats(d.get(sz, [])) for d in (data_a, data_b, data_c))
            def f(x): return f"{x[0]:10.4f}" if x[0] == x[0] else "         —"
            if b[0] == b[0] and c[0] == c[0] and b[0]:
                ratio = f"{c[0]/b[0]:.1f}x"
            else:
                ratio = "—"
            print(f"  {sz:>10} | {f(a)} | {f(b)} | {f(c)} | {ratio}")


if __name__ == "__main__":
    plot_p2p()
    text_table()
