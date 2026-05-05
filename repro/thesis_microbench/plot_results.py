#!/usr/bin/env python3
"""Parse NVSHMEM perftest logs from repro/thesis_microbench/results/ and produce plots
mirroring the thesis Chapter 4 micro-benches.

Plots produced:
  4.1.1 P2P:
    - p2p_msgsize_intra.png  : 6 APIs, intranode, BW vs message size
    - p2p_msgsize_inter.png  : 6 APIs, internode, BW vs message size
    - p2p_tpb_intra.png      : BW vs threads-per-block at 1 MiB (intra)
    - p2p_tpb_inter.png      : BW vs threads-per-block at 1 MiB (inter)
    - p2p_cta_intra.png      : BW vs CTAs at 1 MiB (intra)
    - p2p_cta_inter.png      : BW vs CTAs at 1 MiB (inter)
    - p2p_pp_latency.png     : ping-pong latency, intra vs inter
  4.1.2 Coll:
    - coll_msgsize_intra.png : 5 collectives, intranode 8 ranks
    - coll_msgsize_inter.png : 5 collectives, internode 16 ranks
    - coll_rank_scaling.png  : alltoall/fcollect/reduction at 64KB vs num_ranks
"""

import re
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "repro" / "thesis_microbench" / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TAG = "-newcluster-20260504"

# perftest device output format:
#   #shmem_<api>_bw...
#   size(B)     scope     BW (GB/sec)
#   4           None      0.001820
#   ...
# Latency tests:
#   #<test>...
#   size(B)     ...    latency (us)
#   4           ...    1.234
ROW_RE = re.compile(r"^\s*(\d+)\s+\S+\s+([\d.eE+-]+)\s*$")
LAT_HDR_RE = re.compile(r"latency.*us|us.*latency", re.IGNORECASE)


def parse_size_bw(path: Path) -> list[tuple[int, float]]:
    """Return list of (size_bytes, value) rows from a perftest log.

    Handles two formats:
      P2P device:  "size(B)     scope     BW (GB/sec)"   -> 3 columns, last = BW
      Coll device: "size(B)  count  type  scope  latency(us)  algbw  busbw" -> 7 cols, latency col 4
    """
    if not path.exists():
        return []
    rows = []
    in_table = False
    is_coll = False
    # Two coll header layouts:
    #   (A) alltoall/bcast/fcollect: size count type scope latency algbw busbw  (cols=7)
    #       scope values: thread/warp/block
    #   (B) reduction/reducescatter: size count type redop scope latency algbw busbw  (cols=8)
    #       scope values: t/w/b
    coll_layout = None  # "A" or "B"
    for line in path.read_text(errors="ignore").splitlines():
        if "size(B)" in line:
            in_table = True
            is_coll = "latency" in line
            if is_coll:
                hdr = line.split()
                coll_layout = "B" if "redop" in hdr else "A"
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
            if is_coll:
                # Filter to a single (type, scope) so we get one curve per file.
                if coll_layout == "A" and len(f) >= 5:
                    if f[2] != "32-bit" or f[3] != "block":
                        continue
                    rows.append((size, float(f[4])))
                elif coll_layout == "B" and len(f) >= 6:
                    # reduction/reducescatter: type=int32, redop=sum, scope='t' (thread-only is the
                    # default for reductions in this perftest build).
                    if f[2] != "int32" or f[3] != "sum" or f[4] != "t":
                        continue
                    rows.append((size, float(f[5])))
            else:
                if len(f) >= 3:
                    try:
                        rows.append((size, float(f[-1])))
                    except ValueError:
                        pass
    return rows


# ---- Discover what runs we have ----
def find_logs(prefix: str) -> dict[str, Path]:
    out = {}
    for p in RES.glob(f"{prefix}*{TAG}.log"):
        # strip prefix, suffix
        key = p.name[len(prefix):-len(TAG) - len(".log")]
        out[key] = p
    return out


P2P_APIS = ["shmem_g_bw", "shmem_get_bw", "shmem_p_bw", "shmem_put_bw", "shmem_st_bw", "shmem_atomic_bw"]
COLLS = ["alltoall_latency", "bcast_latency", "fcollect_latency", "reduction_latency", "reducescatter_latency"]

COLORS = plt.cm.tab10.colors


def plot_p2p_msgsize(scenario: str):
    """scenario is 'intra' or 'inter'."""
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False
    for i, api in enumerate(P2P_APIS):
        log = RES / f"p2p_{scenario}_{api}_msgsize{TAG}.log"
        rows = parse_size_bw(log)
        if not rows:
            continue
        any_data = True
        sizes, bws = zip(*rows)
        ax.loglog(sizes, bws, marker="o", ms=3, label=api.replace("shmem_", "").replace("_bw", ""), color=COLORS[i])
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("message size (B)")
    ax.set_ylabel("BW (GB/s)")
    title = "intranode (1 node, 2 ranks, NVLink)" if scenario == "intra" else "internode (2 nodes, 1 rank each, IB)"
    ax.set_title(f"4.1.1 P2P device BW vs message size — {title}\n(NVSHMEM 3.3.9-ibp on H200, 32 CTAs × 256 TPB)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"p2p_msgsize_{scenario}.png", dpi=150)
    plt.close(fig)


def _parse_swept_param(prefix: str, param_name: str) -> dict[str, list[tuple[int, float]]]:
    """Look at logs of the form prefix_{api}_{param}{N}.log; return per-API list of (N, peak_bw)."""
    per_api: dict[str, list[tuple[int, float]]] = defaultdict(list)
    pat = re.compile(rf"{prefix}_(.+?)_{param_name}(\d+){TAG}\.log$")
    for log in RES.glob(f"{prefix}_*_{param_name}*{TAG}.log"):
        m = pat.match(log.name)
        if not m:
            continue
        api = m.group(1)
        n = int(m.group(2))
        rows = parse_size_bw(log)
        if not rows:
            continue
        # for TPB/CTA sweeps we pinned size; take the last (largest) row's BW
        peak_bw = rows[-1][1]
        per_api[api].append((n, peak_bw))
    for api in per_api:
        per_api[api].sort()
    return per_api


def plot_p2p_param(param: str, scenario: str):
    per_api = _parse_swept_param(f"p2p_{scenario}", param)
    if not per_api:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (api, points) in enumerate(sorted(per_api.items())):
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", ms=4, label=api.replace("shmem_", "").replace("_bw", ""), color=COLORS[i % len(COLORS)])
    ax.set_xlabel(f"{param} (threads/CTA)" if param == "tpb" else "number of CTAs")
    ax.set_ylabel("BW at 1 MiB (GB/s)")
    title = "intranode" if scenario == "intra" else "internode"
    ax.set_title(f"4.1.1 P2P BW vs {param.upper()} — {title} (1 MiB payload)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"p2p_{param}_{scenario}.png", dpi=150)
    plt.close(fig)


def plot_p2p_pp_latency():
    """Ping-pong latency (us) vs message size, for both intra and inter."""
    fig, ax = plt.subplots(figsize=(8, 5))
    pp_bins = [
        "shmem_p_ping_pong_latency",
        "shmem_put_ping_pong_latency",
        "shmem_signal_ping_pong_latency",
        "shmem_atomic_ping_pong_latency",
    ]
    plotted = False
    for i, b in enumerate(pp_bins):
        for j, scen in enumerate(["intra", "inter"]):
            log = RES / f"p2p_{scen}_{b}{TAG}.log"
            rows = parse_size_bw(log)
            if not rows:
                continue
            xs, ys = zip(*rows)
            ax.loglog(
                xs, ys, marker="o" if scen == "intra" else "s", ms=3,
                linestyle="-" if scen == "intra" else "--",
                label=f"{b.replace('shmem_', '').replace('_ping_pong_latency', '')}/{scen}",
                color=COLORS[i],
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("message size (B)")
    ax.set_ylabel("round-trip latency (µs)")
    ax.set_title("4.1.1 P2P ping-pong latency (intranode solid, internode dashed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "p2p_pp_latency.png", dpi=150)
    plt.close(fig)


def plot_coll_msgsize(scenario: str, n_ranks: int):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for i, c in enumerate(COLLS):
        log = RES / f"coll_{scenario}_{c}_{n_ranks}r_msgsize{TAG}.log"
        rows = parse_size_bw(log)
        if not rows:
            continue
        plotted = True
        xs, ys = zip(*rows)
        ax.loglog(xs, ys, marker="o", ms=3, label=c.replace("_latency", ""), color=COLORS[i])
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("message size (B)")
    ax.set_ylabel("latency (µs)")
    title = f"intranode 1×{n_ranks}" if scenario == "intra" else f"internode 2×{n_ranks // 2}"
    ax.set_title(f"4.1.2 Collective latency vs message size — {title}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"coll_msgsize_{scenario}_{n_ranks}r.png", dpi=150)
    plt.close(fig)


def plot_coll_rank_scaling():
    """At fixed 64 KiB, scaling intra ranks (2/4/8) and inter ranks (4/8/16)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for i, c in enumerate(["alltoall_latency", "fcollect_latency", "reduction_latency"]):
        intra_pts = []
        for r in [2, 4, 8]:
            rows = parse_size_bw(RES / f"coll_intra_{c}_{r}r{TAG}.log")
            if rows:
                intra_pts.append((r, rows[-1][1]))
        inter_pts = []
        for r in [2, 4, 8]:
            rows = parse_size_bw(RES / f"coll_inter_{c}_{r * 2}r_2x{r}{TAG}.log")
            if rows:
                inter_pts.append((r * 2, rows[-1][1]))
        if intra_pts:
            xs, ys = zip(*intra_pts)
            ax.plot(xs, ys, marker="o", ms=5, linestyle="-", color=COLORS[i],
                    label=f"{c.replace('_latency', '')} (intra NVL)")
            plotted = True
        if inter_pts:
            xs, ys = zip(*inter_pts)
            ax.plot(xs, ys, marker="s", ms=5, linestyle="--", color=COLORS[i],
                    label=f"{c.replace('_latency', '')} (inter IB)")
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("number of ranks")
    ax.set_ylabel("latency (µs)")
    ax.set_title("4.1.2 Collective latency vs rank count (64 KiB payload)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "coll_rank_scaling.png", dpi=150)
    plt.close(fig)


def parse_nccl_log(path: Path) -> list[tuple[int, float, float]]:
    """Return (size, latency_us, busbw_GBs) rows from an nccl-tests output.

    Format:
      #       size         count      type   redop    root     time   algbw   busbw  #wrong  ... (in-place repeats)
      #        (B)    (elements)                               (us)  (GB/s)  (GB/s)
                4             1     float     sum      -1    34.05    0.00    0.00       0    ...
    Drop size==0 (warmup probe rows nccl emits) and use the first 'time' column (out-of-place).
    """
    if not path.exists():
        return []
    rows = []
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
            rows.append((size, float(f[5]), float(f[7])))
        except (ValueError, IndexError):
            pass
    return rows


def plot_nccl_vs_nvshmem():
    """4.2.1 NCCL vs NVSHMEM allreduce + alltoall (intra 8 GPU and inter 16 GPU)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pairings = [
        ("intra", 8, "1×8 NVL", "alltoall"),
        ("inter", 16, "2×8 NVL+IB", "alltoall"),
    ]
    # alltoall plot
    ax = axes[0]
    plotted = False
    for scen, n, label, op in pairings:
        # NVSHMEM -- coll latency (us)
        nvs_log = RES / f"coll_{scen}_alltoall_latency_{n}r_msgsize{TAG}.log"
        nvs = parse_size_bw(nvs_log)
        nccl_label = "alltoall_8g" if scen == "intra" else "alltoall_2x8"
        nccl_log = RES / f"nccl_{scen}_{nccl_label}{TAG}.log"
        nccl = parse_nccl_log(nccl_log)
        if nvs:
            xs, ys = zip(*nvs)
            ax.loglog(xs, ys, marker="o", ms=3, label=f"NVSHMEM {label}", linestyle="-")
            plotted = True
        if nccl:
            xs = [r[0] for r in nccl]; ys = [r[1] for r in nccl]
            ax.loglog(xs, ys, marker="s", ms=3, label=f"NCCL {label}", linestyle="--")
            plotted = True
    ax.set_xlabel("message size (B)")
    ax.set_ylabel("latency (µs)")
    ax.set_title("alltoall — NCCL vs NVSHMEM")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # allreduce plot
    ax = axes[1]
    for scen, n, label, _ in pairings:
        nvs = parse_size_bw(RES / f"coll_{scen}_reduction_latency_{n}r_msgsize{TAG}.log")
        nccl_label = "allreduce_8g" if scen == "intra" else "allreduce_2x8"
        nccl = parse_nccl_log(RES / f"nccl_{scen}_{nccl_label}{TAG}.log")
        if nvs:
            xs, ys = zip(*nvs)
            ax.loglog(xs, ys, marker="o", ms=3, label=f"NVSHMEM {label}", linestyle="-")
        if nccl:
            xs = [r[0] for r in nccl]; ys = [r[1] for r in nccl]
            ax.loglog(xs, ys, marker="s", ms=3, label=f"NCCL {label}", linestyle="--")
    ax.set_xlabel("message size (B)")
    ax.set_ylabel("latency (µs)")
    ax.set_title("allreduce / reduction — NCCL vs NVSHMEM")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    if not plotted:
        plt.close(fig)
        return
    fig.suptitle("4.2.1 NCCL vs NVSHMEM device collectives (H200, sum, float)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "nccl_vs_nvshmem.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- main ----
print("Plots generated to:", FIG)
plot_p2p_msgsize("intra")
plot_p2p_msgsize("inter")
plot_p2p_param("tpb", "intra")
plot_p2p_param("tpb", "inter")
plot_p2p_param("cta", "intra")
plot_p2p_param("cta", "inter")
plot_p2p_pp_latency()
plot_coll_msgsize("intra", 8)
plot_coll_msgsize("inter", 16)
plot_coll_rank_scaling()
plot_nccl_vs_nvshmem()
print("Done.")
