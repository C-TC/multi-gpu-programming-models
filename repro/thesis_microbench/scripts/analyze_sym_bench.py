#!/usr/bin/env python3
"""Analyze the 8-trial sym-kernel benchmark.

For each (config, message size) compute mean and stddev of the latency across the
8 trials and print a comparison table + a per-collective comparison plot."""

import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "repro" / "thesis_microbench" / "results"
TAG = "-newcluster-20260504"
N_TRIAL = 8
CONFIGS = ["default", "sym", "nvls_off"]

# perftest table row (cleaned of NCCL_DEBUG noise that we know is absent in this run):
ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+float\s+sum\s+-1\s+([\d.]+)")


def parse_perftest(path: Path) -> dict[int, float]:
    """Return {size: out_of_place_time_us}."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = ROW_RE.match(line)
        if m:
            size = int(m.group(1))
            t_us = float(m.group(3))  # out-of-place time
            out[size] = t_us
    return out


def stats(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, stddev, median)."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    med = statistics.median(values)
    return mean, sd, med


def main():
    op = "all_reduce"
    by_config: dict[str, dict[int, list[float]]] = {c: {} for c in CONFIGS}
    for cfg in CONFIGS:
        for trial in range(1, N_TRIAL + 1):
            log = RES / f"bench_{cfg}_{op}_t{trial}{TAG}.log"
            data = parse_perftest(log)
            for sz, t in data.items():
                by_config[cfg].setdefault(sz, []).append(t)

    sizes = sorted({sz for cfg in CONFIGS for sz in by_config[cfg]})

    # Print summary table
    print(f"\n=== {op}, intra 8 GPU H200, N={N_TRIAL} trials each, 100 warmup + 200 timed iters ===\n")
    print(f"{'size':>10}  {'default':>20}  {'sym (-R 2)':>20}  {'nvls_off':>20}  {'sym/default':>10}")
    print(f"{'(B)':>10}  {'mean ± sd  (med) µs':>20}  {'mean ± sd  (med) µs':>20}  {'mean ± sd  (med) µs':>20}")
    print("-" * 95)
    for sz in sizes:
        cells = []
        for cfg in CONFIGS:
            mean, sd, med = stats(by_config[cfg].get(sz, []))
            cells.append(f"{mean:6.2f} ± {sd:5.2f} ({med:6.2f})")
        d = stats(by_config['default'].get(sz, []))[0]
        s = stats(by_config['sym'].get(sz, []))[0]
        ratio = f"{s/d:.3f}" if d and s else "—"
        print(f"{sz:>10}  {cells[0]:>20}  {cells[1]:>20}  {cells[2]:>20}  {ratio:>10}")
    print()

    # Significance: |sym - default| / sd_default, is it > 2?
    print("--- Are sym vs default differences significant? (>2σ) ---")
    for sz in sizes:
        d_vals = by_config['default'].get(sz, [])
        s_vals = by_config['sym'].get(sz, [])
        if len(d_vals) < 2 or len(s_vals) < 2:
            continue
        d_mean, d_sd, _ = stats(d_vals)
        s_mean, s_sd, _ = stats(s_vals)
        diff = s_mean - d_mean
        # Use pooled stddev for a Welch's-t-style number
        pooled_sd = ((d_sd**2 + s_sd**2) / 2) ** 0.5
        z = diff / pooled_sd if pooled_sd > 0 else float("nan")
        marker = "***" if abs(z) > 2 else ("*" if abs(z) > 1 else " ")
        print(f"  size={sz:>9} default={d_mean:6.2f}±{d_sd:.2f}  sym={s_mean:6.2f}±{s_sd:.2f}  Δ={diff:+5.2f}µs  z={z:+.2f} {marker}")
    print()


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    op = "all_reduce"
    by_config: dict[str, dict[int, list[float]]] = {c: {} for c in CONFIGS}
    for cfg in CONFIGS:
        for trial in range(1, N_TRIAL + 1):
            log = RES / f"bench_{cfg}_{op}_t{trial}{TAG}.log"
            data = parse_perftest(log)
            for sz, t in data.items():
                by_config[cfg].setdefault(sz, []).append(t)

    sizes = sorted({sz for cfg in CONFIGS for sz in by_config[cfg]})

    fig, (ax_lat, ax_ratio) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"default": "tab:green", "sym": "tab:blue", "nvls_off": "tab:red"}
    labels = {
        "default": "default (-R 0, NVLS on, autotuner)",
        "sym": "sym kernel (-R 2, ncclSymmetricTaskScheduler)",
        "nvls_off": "no NVLS (-R 0, NCCL_NVLS_ENABLE=0)",
    }
    for cfg in CONFIGS:
        xs, means, sds = [], [], []
        for sz in sizes:
            vals = by_config[cfg].get(sz, [])
            if not vals:
                continue
            mean, sd, _ = stats(vals)
            xs.append(sz); means.append(mean); sds.append(sd)
        ax_lat.errorbar(xs, means, yerr=sds, marker="o", ms=4, capsize=3,
                        label=labels[cfg], color=colors[cfg])

    # ratio plot: sym/default and nvls_off/default
    d_means = {sz: stats(by_config["default"][sz])[0] for sz in sizes if by_config["default"].get(sz)}
    for cfg in ["sym", "nvls_off"]:
        xs, ratios = [], []
        for sz in sizes:
            v = by_config[cfg].get(sz, [])
            if v and sz in d_means:
                xs.append(sz)
                ratios.append(stats(v)[0] / d_means[sz])
        ax_ratio.plot(xs, ratios, marker="o", ms=4, label=f"{cfg}/default", color=colors[cfg])
    ax_ratio.axhline(1.0, color="gray", linestyle=":", linewidth=0.7)

    ax_lat.set_xscale("log"); ax_lat.set_yscale("log")
    ax_lat.set_xlabel("message size (B)"); ax_lat.set_ylabel("latency (µs, mean ± stddev over 8 trials)")
    ax_lat.set_title("NCCL all_reduce on 8×H200 — 100 warmup + 200 timed iters per data point")
    ax_lat.grid(True, which="both", alpha=0.3)
    ax_lat.legend(fontsize=9)

    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("message size (B)"); ax_ratio.set_ylabel("latency ratio vs default")
    ax_ratio.set_title("Speedup of sym kernel and NVLS-off vs default (lower = faster)")
    ax_ratio.grid(True, which="both", alpha=0.3)
    ax_ratio.legend(fontsize=9)

    fig.suptitle(
        "NCCL 2.30.4 sym kernel vs default vs NVLS-off  (intra 8 GPU, all_reduce float32+sum)\n"
        "sym kernel = nccl-tests -R 2 (ncclCommWindowRegister + NCCL_WIN_COLL_SYMMETRIC) → ncclSymmetricTaskScheduler engages",
        y=1.04, fontsize=10)
    fig.tight_layout()
    out = REPO / "repro" / "thesis_microbench" / "results" / "figures" / "sym_kernel_8trials.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
    plot()
