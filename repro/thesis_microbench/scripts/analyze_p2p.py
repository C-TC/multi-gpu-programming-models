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
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
    for col, (api, api_label) in enumerate(APIS):
        for row, (scen, scen_label) in enumerate([("intra", "intranode 1×2"), ("inter", "internode 2×1")]):
            ax = axes[row, col]
            for transport, color, marker in [
                ("p2p", "tab:orange", "o"),
                ("p2p_ibgda", "tab:green", "s"),
            ]:
                d = collect(f"{transport}_{scen}_{api}")
                if not d:
                    continue
                sizes = sorted(d.keys())
                means = [stats(d[s])[0] for s in sizes]
                sds = [stats(d[s])[1] for s in sizes]
                label = "ibrc (default = CPU proxy)" if transport == "p2p" else "IBGDA (GPU-init RDMA)"
                ax.errorbar(sizes, means, yerr=sds, marker=marker, ms=4, capsize=2,
                            label=label, color=color)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            if row == 0: ax.set_title(api_label, fontsize=11)
            if col == 0: ax.set_ylabel(f"{scen_label}\nBW (GB/s)")
            if row == 1: ax.set_xlabel("size (B)")
            if row == 0 and col == 0: ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "NVSHMEM P2P: default (ibrc CPU-proxy) vs IBGDA (GPU-init RDMA) — H200, 8 trials, mean ± stddev\n"
        "Intra: NVLink P2P, transport selection irrelevant (lines should overlap).  "
        "Inter: IBRC = host CPU proxy posts WRs; IBGDA = GPU posts WRs directly via DCI/RC QPs in GPU-mapped NIC memory.",
        y=1.02, fontsize=9)
    fig.tight_layout()
    out = FIG / "p2p_ibrc_vs_ibgda.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def text_table() -> None:
    print(f"\n=== P2P ibrc vs IBGDA (mean GB/s, IBGDA / IBRC speedup) ===")
    for scen in ["inter"]:  # focus on inter where transport matters
        for api, _ in APIS:
            print(f"\n{scen} {api}:")
            data_a = collect(f"p2p_{scen}_{api}")
            data_b = collect(f"p2p_ibgda_{scen}_{api}")
            sizes = sorted(set(data_a) | set(data_b))
            print(f"  {'size':>10} | {'ibrc':>10s} | {'IBGDA':>10s} | speedup")
            for sz in sizes:
                a = stats(data_a.get(sz, []))
                b = stats(data_b.get(sz, []))
                a_str = f"{a[0]:8.4f}" if a[0] == a[0] else "       —"
                b_str = f"{b[0]:8.4f}" if b[0] == b[0] else "       —"
                if a[0] and a[0] == a[0] and b[0] == b[0]:
                    ratio_str = f"{b[0]/a[0]:.2f}x"
                else:
                    ratio_str = "—"
                print(f"  {sz:>10} | {a_str} | {b_str} | {ratio_str}")


if __name__ == "__main__":
    plot_p2p()
    text_table()
