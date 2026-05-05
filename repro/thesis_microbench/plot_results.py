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
    # Three coll perftest header layouts:
    #   (A) device alltoall/bcast/fcollect:
    #       size count type scope latency algbw busbw  (7 cols)
    #       type ∈ {32-bit, 64-bit}; scope ∈ {thread, warp, block}
    #   (B) device reduction/reducescatter:
    #       size count type redop scope latency algbw busbw  (8 cols)
    #       type=int32, redop=sum, scope ∈ {t, w, b}
    #   (C) host on_stream:
    #       size count type latency min_lat max_lat algbw busbw  (8 cols, 'latency' instead of 'scope')
    #       type=int (single)
    coll_layout = None
    for line in path.read_text(errors="ignore").splitlines():
        if "size(B)" in line:
            in_table = True
            is_coll = "latency" in line
            if is_coll:
                hdr = line.split()
                has_redop = "redop" in hdr
                has_min_lat = "min_lat(us)" in line or any("min_lat" in h for h in hdr)
                if has_redop and has_min_lat:
                    coll_layout = "D"  # host on_stream reduction/reducescatter
                elif has_redop:
                    coll_layout = "B"  # device reduction/reducescatter
                elif has_min_lat:
                    coll_layout = "C"  # host on_stream alltoall/bcast/fcollect
                else:
                    coll_layout = "A"  # device alltoall/bcast/fcollect
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
                if coll_layout == "A" and len(f) >= 5:
                    if f[2] != "32-bit" or f[3] != "block":
                        continue
                    rows.append((size, float(f[4])))
                elif coll_layout == "B" and len(f) >= 6:
                    if f[2] != "int32" or f[3] != "sum" or f[4] != "t":
                        continue
                    rows.append((size, float(f[5])))
                elif coll_layout == "C" and len(f) >= 4:
                    # size count type latency min_lat max_lat algbw busbw
                    rows.append((size, float(f[3])))
                elif coll_layout == "D" and len(f) >= 5:
                    # size count type redop latency min_lat max_lat algbw busbw
                    if f[3] != "sum":
                        continue
                    rows.append((size, float(f[4])))
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


# Friendly labels for each P2P binary -- includes API kind + scope hint.
# The perftest binaries name their kernel:
#   shmem_g_bw          -- nvshmem_g_<type>  (single-element get, scalar load)
#   shmem_get_bw        -- nvshmem_<type>_get  (block-cooperative get, all threads in block participate)
#   shmem_p_bw          -- nvshmem_p_<type>  (single-element put, scalar store)
#   shmem_put_bw        -- nvshmem_<type>_put  (block-cooperative put)
#   shmem_st_bw         -- direct CUDA store via mapped peer pointer (NVL only)
#   shmem_atomic_bw     -- nvshmem_<type>_atomic_inc (block-cooperative atomic)
P2P_LABEL = {
    "shmem_g_bw": "g (single-elem get, scalar)",
    "shmem_get_bw": "get (block-coop bulk)",
    "shmem_p_bw": "p (single-elem put, scalar)",
    "shmem_put_bw": "put (block-coop bulk)",
    "shmem_st_bw": "st (direct store, NVL only)",
    "shmem_atomic_bw": "atomic_inc (block-coop)",
}


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
        ax.loglog(sizes, bws, marker="o", ms=3, label=P2P_LABEL.get(api, api), color=COLORS[i])
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("message size (B)")
    ax.set_ylabel("BW (GB/s)")
    title = "intranode (1 node, 2 ranks, NVLink)" if scenario == "intra" else "internode (2 nodes, 1 rank each, IB IBGDA)"
    ax.set_title(
        f"4.1.1 P2P device BW vs message size — {title}\n"
        f"NVSHMEM 3.3.9-ibp on H200; all *_bw kernels use 32 CTAs × 256 TPB; default datatype=int32"
    )
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


COLL_LABEL = {
    "alltoall_latency": "alltoall (32-bit, block scope)",
    "bcast_latency": "bcast (32-bit, block scope)",
    "fcollect_latency": "fcollect/allgather (32-bit, block scope)",
    "reduction_latency": "reduction/allreduce (int32+sum, thread scope, ALGO autotuner)",
    "reducescatter_latency": "reducescatter (int32+sum, thread scope)",
}


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
        ax.loglog(xs, ys, marker="o", ms=3, label=COLL_LABEL.get(c, c), color=COLORS[i])
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("message size (B)")
    ax.set_ylabel("latency (µs)")
    title = f"intranode 1×{n_ranks}" if scenario == "intra" else f"internode 2×{n_ranks // 2}"
    ax.set_title(
        f"4.1.2 NVSHMEM device collective latency vs message size — {title}\n"
        f"alltoall/bcast/fcollect: block-scope kernel; reduction/reducescatter: thread-scope (only scope this build emits)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
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


def _series(label, ls, marker, color):
    return {"label": label, "ls": ls, "marker": marker, "color": color, "ms": 4}


# Map "thesis collective name" -> (NVSHMEM device coll log basename, NVSHMEM host on_stream basename, NCCL binary basename)
COLL_MAP = [
    ("alltoall",       "alltoall_latency",       "alltoall_on_stream",       "alltoall"),
    ("allreduce/sum",  "reduction_latency",      "reduction_on_stream",      "all_reduce"),
    ("broadcast",      "bcast_latency",          "broadcast_on_stream",      "broadcast"),
    ("allgather/fcollect", "fcollect_latency",   "fcollect_on_stream",       "all_gather"),
    ("reducescatter/sum", "reducescatter_latency","reducescatter_on_stream", "reduce_scatter"),
]


def plot_per_collective_comparison(name: str, dev_base: str, host_base: str, nccl_base: str):
    """Per-collective: 2 subplots (intra 8r, inter 16r). Each shows up to 3 series:
       NVSHMEM device, NVSHMEM host on_stream, NCCL."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plotted_any = False
    for ax, scen, n in [(axes[0], "intra", 8), (axes[1], "inter", 16)]:
        # NVSHMEM device kernel
        dev_log = RES / f"coll_{scen}_{dev_base}_{n}r_msgsize{TAG}.log"
        dev_rows = parse_size_bw(dev_log)
        # NVSHMEM host on_stream
        host_log = RES / f"coll_{scen}_{host_base}_{n}r_msgsize{TAG}.log"
        host_rows = parse_size_bw(host_log)
        # NCCL
        nccl_label = f"{nccl_base}_8g" if scen == "intra" else f"{nccl_base}_2x8"
        nccl_log = RES / f"nccl_{scen}_{nccl_label}{TAG}.log"
        nccl_rows = parse_nccl_log(nccl_log)

        # Determine NVSHMEM dev sub-config label dynamically.
        if dev_base in ("alltoall_latency", "bcast_latency", "fcollect_latency"):
            nvs_dev_label = "NVSHMEM device kernel (block scope, 32-bit)"
        else:
            nvs_dev_label = "NVSHMEM device kernel (thread scope, int32+sum)"

        if dev_rows:
            xs, ys = zip(*dev_rows)
            ax.loglog(xs, ys, marker="o", ms=4, color="tab:blue", linestyle="-",
                      label=nvs_dev_label)
            plotted_any = True
        if host_rows:
            xs, ys = zip(*host_rows)
            ax.loglog(xs, ys, marker="^", ms=4, color="tab:orange", linestyle="-.",
                      label="NVSHMEM host on_stream (CPU-initiated)")
            plotted_any = True
        if nccl_rows:
            xs = [r[0] for r in nccl_rows]; ys = [r[1] for r in nccl_rows]
            ax.loglog(xs, ys, marker="s", ms=4, color="tab:green", linestyle="--",
                      label="NCCL (auto-tuner)")
            plotted_any = True

        scen_label = f"intranode 1×{n}" if scen == "intra" else f"internode 2×{n // 2}"
        ax.set_title(f"{name} — {scen_label}")
        ax.set_xlabel("message size (B)")
        ax.set_ylabel("latency (µs)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    if not plotted_any:
        plt.close(fig)
        return
    fig.suptitle(f"4.2.1 NCCL vs NVSHMEM — {name} (H200, NVSHMEM 3.3.9-ibp; no NCCL fallback in this build)",
                 y=1.02)
    fig.tight_layout()
    safe = name.replace("/", "_").replace(" ", "_")
    fig.savefig(FIG / f"compare_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nccl_vs_nvshmem():
    """Backward-compat: keep the old combined alltoall+allreduce 2-panel figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plotted = False
    for ax_idx, (name, dev_base, _, nccl_base) in enumerate(
        [("alltoall", "alltoall_latency", "", "alltoall"),
         ("allreduce", "reduction_latency", "", "all_reduce")]
    ):
        ax = axes[ax_idx]
        for scen, n, label_short in [("intra", 8, "1×8 NVL"), ("inter", 16, "2×8 NVL+IB")]:
            nvs = parse_size_bw(RES / f"coll_{scen}_{dev_base}_{n}r_msgsize{TAG}.log")
            nccl_label = f"{nccl_base}_8g" if scen == "intra" else f"{nccl_base}_2x8"
            nccl = parse_nccl_log(RES / f"nccl_{scen}_{nccl_label}{TAG}.log")
            if nvs:
                xs, ys = zip(*nvs)
                ax.loglog(xs, ys, marker="o", ms=3, label=f"NVSHMEM device {label_short}", linestyle="-")
                plotted = True
            if nccl:
                xs = [r[0] for r in nccl]; ys = [r[1] for r in nccl]
                ax.loglog(xs, ys, marker="s", ms=3, label=f"NCCL {label_short}", linestyle="--")
                plotted = True
        ax.set_xlabel("message size (B)")
        ax.set_ylabel("latency (µs)")
        ax.set_title(f"{name} — NCCL vs NVSHMEM device kernel")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    if not plotted:
        plt.close(fig)
        return
    fig.suptitle("4.2.1 NCCL vs NVSHMEM device collectives (H200, sum, float; 32-bit block scope)", y=1.02)
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
plot_nccl_vs_nvshmem()  # combined 2-panel summary
for entry in COLL_MAP:
    plot_per_collective_comparison(*entry)
print("Done.")
