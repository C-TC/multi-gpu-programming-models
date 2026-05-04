#!/usr/bin/env python3
"""Generate V1 (NVSHMEM IBGDA) vs V2 (NCCL Gin) comparison plots from
the DeepEP newcluster-20260504 logs.

Produces three figures in repro/deepep/results/figures/:
  1. fig_ht_dispatch_bw.png  — HT dispatch BW comparison (1-node + 2-node)
  2. fig_ht_combine_bw.png   — HT combine BW comparison
  3. fig_ll_latency.png      — LL end-to-end latency comparison

Run from the repo root: python3 repro/deepep/plot_results.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "repro" / "deepep" / "results"
FIG = LOGS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TAG = "newcluster-20260504"


def grep_first(path: Path, pattern: str, group: int = 1) -> float | None:
    """Return the first regex match value (as float) from `path`, or None."""
    if not path.exists():
        return None
    rx = re.compile(pattern)
    for line in path.read_text(errors="ignore").splitlines():
        m = rx.search(line)
        if m:
            return float(m.group(group))
    return None


def best_metric(path: Path, op: str, dtype: str = "FP8") -> dict:
    """Parse `[tuning] Best dispatch (FP8): ... GB/s (RDMA), GB/s (NVL), t: ... us` lines."""
    if not path.exists():
        return {}
    # Capture both single-node ("X GB/s (NVL), t: Y us") and 2-node ("A + B us, X GB/s (RDMA), Y GB/s (NVL)")
    rx2node = re.compile(
        rf"\[tuning\]\s+Best\s+{op}\s*\({dtype}\)?:.*?(\d+(?:\.\d+)?)\s+(?:\+ )?(\d+(?:\.\d+)?)\s*us,\s+"
        rf"(\d+(?:\.\d+)?)\s+GB/s\s+\(RDMA\),\s+(\d+(?:\.\d+)?)\s+GB/s\s+\(NVL\)"
    )
    rx1node = re.compile(
        rf"\[tuning\]\s+Best\s+{op}\s*\({dtype}\)?:.*?(\d+(?:\.\d+)?)\s+GB/s\s+\(NVL\),\s+t:\s*(\d+(?:\.\d+)?)\s*us"
    )
    rx_combine_1node = re.compile(
        rf"\[tuning\]\s+Best\s+{op}:.*?(\d+(?:\.\d+)?)\s+GB/s\s+\(NVL\),\s+t:\s*(\d+(?:\.\d+)?)\s*us"
    )
    rx_combine_2node = re.compile(
        rf"\[tuning\]\s+Best\s+{op}:.*?(\d+(?:\.\d+)?)\s+(?:\+ )?(\d+(?:\.\d+)?)\s*us,\s+"
        rf"(\d+(?:\.\d+)?)\s+GB/s\s+\(RDMA\),\s+(\d+(?:\.\d+)?)\s+GB/s\s+\(NVL\)"
    )

    last = None
    for line in path.read_text(errors="ignore").splitlines():
        m2 = rx2node.search(line) or rx_combine_2node.search(line)
        if m2:
            last = {"t_us": float(m2.group(2)), "rdma_gbs": float(m2.group(3)), "nvl_gbs": float(m2.group(4))}
            continue
        m1 = rx1node.search(line) or rx_combine_1node.search(line)
        if m1:
            last = {"t_us": float(m1.group(2)), "nvl_gbs": float(m1.group(1)), "rdma_gbs": 0.0}
    return last or {}


def parse_v1_ll_combined(path: Path) -> float | None:
    """V1 LL `[rank N] Dispatch + combine bandwidth: X GB/s, avg_t=Y us` — return mean avg_t."""
    if not path.exists():
        return None
    rx = re.compile(r"Dispatch \+ combine bandwidth:\s+([\d.]+)\s+GB/s,\s+avg_t=([\d.]+)\s+us")
    times = []
    for line in path.read_text(errors="ignore").splitlines():
        m = rx.search(line)
        if m:
            times.append(float(m.group(2)))
    return sum(times) / len(times) if times else None


def parse_v2_ep(path: Path) -> dict:
    """V2 ep test_ep.py — `+ EP: 0/N | reduced combine: A GB/s (SO), B GB/s (SU), C us, D bytes` etc.
    Returns dict with dispatch + combine SO/SU/t parsed for the * (dispatch) and + (reduced combine) lines.
    """
    if not path.exists():
        return {}
    rx_dispatch = re.compile(
        r"\*\s+EP:\s+0/\d+\s+\|\s+dispatch:\s+(\d+)\s+GB/s\s+\(SO\),\s+(\d+)\s+GB/s\s+\(SU\),\s+([\d.]+)\s+us"
    )
    rx_combine = re.compile(
        r"\+\s+EP:\s+0/\d+\s+\|\s+reduced combine:\s+(\d+)\s+GB/s\s+\(SO\),\s+(\d+)\s+GB/s\s+\(SU\),\s+([\d.]+)\s+us"
    )
    out = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = rx_dispatch.search(line)
        if m:
            out["dispatch_so"] = int(m.group(1))
            out["dispatch_su"] = int(m.group(2))
            out["dispatch_t_us"] = float(m.group(3))
        m = rx_combine.search(line)
        if m:
            out["combine_so"] = int(m.group(1))
            out["combine_su"] = int(m.group(2))
            out["combine_t_us"] = float(m.group(3))
    return out


# ---- Gather data ----
data = {
    "V1 intranode 1×8": best_metric(LOGS / f"v1_intranode_mistral_recipe-{TAG}.log", "dispatch", "FP8"),
    "V1 internode HT 2×8": best_metric(LOGS / f"v1_internode_2x8_mistral_recipe-{TAG}.log", "dispatch", "FP8"),
    "V1 intranode 1×8 (combine)": best_metric(LOGS / f"v1_intranode_mistral_recipe-{TAG}.log", "combine", ""),
    "V1 internode HT 2×8 (combine)": best_metric(LOGS / f"v1_internode_2x8_mistral_recipe-{TAG}.log", "combine", ""),
    "V2 ep 1×8 default": parse_v2_ep(LOGS / f"v2_ep-{TAG}.log"),
    "V2 ep 1×8 sms24+topk8": parse_v2_ep(LOGS / f"v2_ep_sms24_topk8-{TAG}.log"),
    "V2 ep 2×8 sms24+topk8": parse_v2_ep(LOGS / f"v2_ep_2x8_topk8_e64-{TAG}.log"),
    "V2 ep 2×4 sms24+topk8": parse_v2_ep(LOGS / f"v2_ep_2x4_topk8_sms24_e64-{TAG}.log"),
    "V1 LL 2×8 (avg_t)": parse_v1_ll_combined(LOGS / f"v1_low_latency_2x8_mistral_recipe-{TAG}.log"),
    "V1 LL 2×4 (avg_t)": parse_v1_ll_combined(LOGS / f"v1_low_latency_2x4_mistral_recipe-{TAG}.log"),
    "V2 LL 1×8": parse_v2_ep(LOGS / f"v2_ep_lowlat_1node-{TAG}.log"),
    "V2 LL 2×8": parse_v2_ep(LOGS / f"v2_ep_lowlat_2x8-{TAG}.log"),
    "V2 LL 2×4": parse_v2_ep(LOGS / f"v2_ep_lowlat_2x4-{TAG}.log"),
}

print("Parsed data:")
for k, v in data.items():
    print(f"  {k}: {v}")


# ---- Plot 1: HT dispatch BW (V1 vs V2) ----
fig, ax = plt.subplots(figsize=(8, 5))
configs = ["1×8 NVL only", "2×4 = 8 ranks", "2×8 = 16 ranks"]
v1_so = [
    data["V1 intranode 1×8"].get("rdma_gbs", 0) + data["V1 intranode 1×8"].get("nvl_gbs", 0) * 0,  # NVL only
    None,  # V1 2×4 internode HT not run (test asserts 8 GPU/node)
    data["V1 internode HT 2×8"].get("rdma_gbs", 0),  # SO BW (RDMA)
]
v1_su = [
    data["V1 intranode 1×8"].get("nvl_gbs", 0),
    None,
    data["V1 internode HT 2×8"].get("nvl_gbs", 0),
]
v2_so = [
    data["V2 ep 1×8 sms24+topk8"].get("dispatch_so", 0),
    data["V2 ep 2×4 sms24+topk8"].get("dispatch_so", 0),
    data["V2 ep 2×8 sms24+topk8"].get("dispatch_so", 0),
]
v2_su = [
    data["V2 ep 1×8 sms24+topk8"].get("dispatch_su", 0),
    data["V2 ep 2×4 sms24+topk8"].get("dispatch_su", 0),
    data["V2 ep 2×8 sms24+topk8"].get("dispatch_su", 0),
]

x = range(len(configs))
w = 0.2
def safe(seq):
    return [v if v is not None else 0 for v in seq]

ax.bar([i - 1.5 * w for i in x], safe(v1_su), w, label="V1 NVSHMEM (NVL/SU)", color="tab:purple")
ax.bar([i - 0.5 * w for i in x], safe(v1_so), w, label="V1 NVSHMEM (RDMA/SO)", color="mediumpurple")
ax.bar([i + 0.5 * w for i in x], safe(v2_su), w, label="V2 NCCL Gin (SU)", color="tab:green")
ax.bar([i + 1.5 * w for i in x], safe(v2_so), w, label="V2 NCCL Gin (SO)", color="lightgreen")
for i, v in enumerate(v1_su):
    if v is None:
        ax.text(i - 0.5 * w, 5, "V1 N/A\n(needs 8 GPU/node)", ha="center", color="gray", fontsize=7)
ax.set_xticks(list(x))
ax.set_xticklabels(configs)
ax.set_ylabel("Dispatch BW (GB/s, FP8 token=4096 hidden=7168 topk=8)")
ax.set_title("DeepEP HT dispatch: V1 NVSHMEM IBGDA vs V2 NCCL Gin (H200, 24 SMs)\n[different kernels + transports — not isolated transport comparison]")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig_ht_dispatch_bw.png", dpi=150)
plt.close(fig)


# ---- Plot 2: HT combine BW (V1 vs V2) ----
fig, ax = plt.subplots(figsize=(8, 5))
v1c_so = [None, None, data["V1 internode HT 2×8 (combine)"].get("rdma_gbs", 0)]
v1c_su = [
    data["V1 intranode 1×8 (combine)"].get("nvl_gbs", 0),
    None,
    data["V1 internode HT 2×8 (combine)"].get("nvl_gbs", 0),
]
v2c_so = [
    data["V2 ep 1×8 sms24+topk8"].get("combine_so", 0),
    data["V2 ep 2×4 sms24+topk8"].get("combine_so", 0),
    data["V2 ep 2×8 sms24+topk8"].get("combine_so", 0),
]
v2c_su = [
    data["V2 ep 1×8 sms24+topk8"].get("combine_su", 0),
    data["V2 ep 2×4 sms24+topk8"].get("combine_su", 0),
    data["V2 ep 2×8 sms24+topk8"].get("combine_su", 0),
]
ax.bar([i - 1.5 * w for i in x], safe(v1c_su), w, label="V1 NVSHMEM (NVL/SU)", color="tab:purple")
ax.bar([i - 0.5 * w for i in x], safe(v1c_so), w, label="V1 NVSHMEM (RDMA/SO)", color="mediumpurple")
ax.bar([i + 0.5 * w for i in x], safe(v2c_su), w, label="V2 NCCL Gin (SU)", color="tab:green")
ax.bar([i + 1.5 * w for i in x], safe(v2c_so), w, label="V2 NCCL Gin (SO)", color="lightgreen")
for i, v in enumerate(v1c_su):
    if v is None:
        ax.text(i - 0.5 * w, 5, "V1 N/A\n(needs 8 GPU/node)", ha="center", color="gray", fontsize=7)
ax.set_xticks(list(x))
ax.set_xticklabels(configs)
ax.set_ylabel("Combine BW (GB/s, reduced combine, 24 SMs)")
ax.set_title("DeepEP HT combine: V1 NVSHMEM IBGDA vs V2 NCCL Gin\n[different kernels + transports — not isolated transport comparison]")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig_ht_combine_bw.png", dpi=150)
plt.close(fig)


# ---- Plot 3: LL latency (V1 vs V2) ----
fig, ax = plt.subplots(figsize=(8, 5))
ll_configs = ["1×8 (NVL fallback)", "2×4 = 8 ranks", "2×8 = 16 ranks"]
# V1 LL single-node uses different log (NVLink fallback) — older log
v1_ll_1node = grep_first(LOGS / f"v1_low_latency-{TAG}.log",
                         r"avg_t=([\d.]+)\s+us, min_t=[\d.]+\s+us")
v1_ll = [v1_ll_1node, data["V1 LL 2×4 (avg_t)"], data["V1 LL 2×8 (avg_t)"]]
# V2 LL: dispatch + combine us summed
def v2_sum(d):
    return d.get("dispatch_t_us", 0) + d.get("combine_t_us", 0)
v2_ll = [
    v2_sum(data["V2 LL 1×8"]),
    v2_sum(data["V2 LL 2×4"]),
    v2_sum(data["V2 LL 2×8"]),
]

x2 = range(len(ll_configs))
ww = 0.35
ax.bar([i - ww / 2 for i in x2], [v or 0 for v in v1_ll], ww,
       label="V1 NVSHMEM IBGDA (avg dispatch+combine)", color="tab:purple")
ax.bar([i + ww / 2 for i in x2], v2_ll, ww,
       label="V2 NCCL Gin (dispatch+combine)", color="tab:green")
for i, (a, b) in enumerate(zip(v1_ll, v2_ll)):
    if a:
        ax.text(i - ww / 2, a + 5, f"{a:.0f}", ha="center", fontsize=8)
    if b:
        ax.text(i + ww / 2, b + 5, f"{b:.0f}", ha="center", fontsize=8)
ax.set_xticks(list(x2))
ax.set_xticklabels(ll_configs)
ax.set_ylabel("Dispatch + combine latency (µs, num_tokens=128)")
ax.set_title("DeepEP LL: V1 NVSHMEM IBGDA vs V2 NCCL Gin (H200, lower = better)\n[V2 LL uses 32 SMs; V1 LL uses 0 SM IBGDA kernel — different cost models]")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig_ll_latency.png", dpi=150)
plt.close(fig)

print(f"Wrote 3 figures to {FIG}")
