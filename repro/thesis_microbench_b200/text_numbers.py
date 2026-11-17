#!/usr/bin/env python3
"""Every number the paper's Sec. 'Microbenchmarking' text quotes, recomputed from the B200 data with the SAME
definition as the figure (peak = max of the 8-trial mean within the plotted size range; latency = mean time per op).
Writes results/tables/paper_text_numbers.md. Run after analyze.py (imports it)."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("an", HERE / "analyze.py")
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)


def p2p(prefix):
    d, n, dropped = an.p2p_series(prefix)
    sz = [s for s in sorted(d) if 2 ** 8 <= s <= 2 ** 25]
    m = {s: an.ms(d[s])[0] for s in sz}
    lat = {s: s / (m[s] * 1e3) for s in sz}  # us
    return m, lat, n, dropped


def ar(prefix):
    lat, bw, n = an.ar_series(prefix)
    sz = [s for s in sorted(bw) if 2 ** 7 <= s <= 2 ** 29 and bw[s]]
    return {s: an.ms(bw[s])[0] for s in sz}, {s: an.ms(lat[s])[0] for s in sz}, n


def rng(vals):
    return f"{min(vals):.1f}–{max(vals):.1f}"


rows = []
def row(claim, h200, b200, note=""):
    rows.append(f"| {claim} | {h200} | {b200} | {note} |")


I = {a: p2p(f"p2p_paper_intra_{a}") for a in ("put", "get", "p", "g")}
X = {a: p2p(f"p2p_paper_inter_ibgdatuned_{a}") for a in ("put", "get", "p", "g")}
for a, h in (("put", "313"), ("get", "141"), ("p", "172"), ("g", "< 9")):
    row(f"intra {a} peak (GB/s)", h, f"{max(I[a][0].values()):.1f}")
for a, h in (("put", "48.0"), ("get", "48.2"), ("p", "15.6"), ("g", "1.28")):
    row(f"inter {a} peak (GB/s), IBGDA tuned", h, f"{max(X[a][0].values()):.2f}")
small = lambda lat: [v for s, v in lat.items() if s <= 2 ** 16]
row("intra latency, bulk put/get, ≤64 KiB (µs)", "1.8–2.5", rng(small(I["put"][1]) + small(I["get"][1])))
row("intra latency, scalar p/g, ≤64 KiB (µs)", "1.3–2.2", rng(small(I["p"][1]) + small(I["g"][1])),
    "H200 text range omits g's rise at ≥16 KiB (8.8 µs @64 KiB in the H200 data)")
first = min(s for s in X["put"][1])
row(f"inter put/get latency at smallest valid size ({first} B) (µs)", "9.4–9.5 @256 B*",
    f"{X['put'][1][first]:.1f} / {X['get'][1][first]:.1f}",
    "*H200 text value came from the untuned c32 run; the tuned c64 run plotted moves 0 B at 256 B")
row("inter put/get latency at 64 KiB (µs)", "~9.7*", f"{X['put'][1][65536]:.1f} / {X['get'][1][65536]:.1f}",
    "*untuned-run value; plotted tuned run: 15.0 / 14.6")
row("inter p / g latency at 256 B (µs)", "7.5 / 25.3", f"{X['p'][1][256]:.1f} / {X['g'][1][256]:.1f}",
    "H200 data actually gives 10.8 / 25.3")

NP = {"nccl_ring": "ar_rearch", "nccl_nvls": "ar_rearch"}  # NCCL: refactored enqueue path (main figures)
A = {k: ar(f"{NP.get(k, 'ar_paper')}_intra_{k}") for k in ("nvshost", "nvsdev", "nccl_ring", "nccl_nvls")}
B = {k: ar(f"{NP.get(k, 'ar_paper')}_inter_{k}") for k in ("nvshost", "nvsdev", "nccl_ring", "nccl_nvls")}
for k, lab, h in (("nvshost", "NVSHMEM on-stream", "264*"), ("nccl_nvls", "NCCL NVLS", "276*"), ("nvsdev", "NVSHMEM device block", "30"),
                  ("nccl_ring", "NCCL ring", "(below on-stream)")):
    if A[k][0]:
        row(f"AllReduce intra peak algbw, {lab} (GB/s)", h, f"{max(A[k][0].values()):.1f}",
            "*H200 text used the unplotted 1 GiB point; plotted peaks were 262.5 / 274.8" if "*" in h else "")
for k, lab, h in (("nvsdev", "NVSHMEM device", "3.8–7.1"), ("nccl_ring", "NCCL ring", "4.7–8.9**"), ("nccl_nvls", "NCCL NVLS", "5.6–5.9")):
    if A[k][1]:
        row(f"AllReduce intra latency ≤64 KiB, {lab} (µs)", h, rng([v for s, v in A[k][1].items() if s <= 2 ** 16]),
            "**H200 'ring' was actually NCCL symmetric kernels (NCCL 2.30 ignores NCCL_ALGO=Ring with -R 2)" if k == "nccl_ring" else "")
for k, lab, h in (("nvshost", "NVSHMEM on-stream", "< 0.20"), ("nvsdev", "NVSHMEM device", "< 0.20"), ("nccl_ring", "NCCL ring", "180"),
                  ("nccl_nvls", "NCCL NVLS-Tree", "252*")):
    if B[k][0]:
        row(f"AllReduce inter peak algbw, {lab} (GB/s)", h, f"{max(B[k][0].values()):.2f}",
            "*H200 text used the unplotted 1 GiB point; plotted peak 246.4" if "*" in h else "")
for k in ("nvshost", "nvsdev", "nccl_ring", "nccl_nvls"):
    if B[k][1] and 65536 in B[k][1]:
        row(f"AllReduce inter latency at 64 KiB, {k} (µs)", "ms (NVSHMEM) / tens of µs (NCCL)", f"{B[k][1][65536]:.1f}")

out = HERE / "results" / "tables" / "paper_text_numbers.md"
out.write_text("| Quantity (paper text) | H200 (paper) | B200 (this rerun, paper config) | Note |\n|---|---|---|---|\n" + "\n".join(rows) + "\n")
print(out.read_text())
