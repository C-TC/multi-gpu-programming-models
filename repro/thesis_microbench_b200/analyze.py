#!/usr/bin/env python3
"""Parse B200 rerun logs (results/raw/*.log) and produce the paper's two figures + tables.

Figures (drop-in replacements for the paper, same series / layout / insets):
  figures/nvshmem-p2p-bandwidth-onecol.pdf        put/get/p/g, intra (1x2 NVLink) and inter (2x1 IB, IBGDA tuned)
  figures/nvshmem-allreduce-bandwidth-onecol.pdf  NVSHMEM [On-stream]/[Device Block], NCCL [Ring]/[NVLS], 8 and 16 GPUs
Extra:
  figures/tuned-*.pdf                             paper config vs tuned NVSHMEM configs
  tables/*.csv                                    mean/stddev/n per size for every series

Every data point is the mean over trials (default 8) of the per-trial value; error bars = sample stddev.
Effective latency in the P2P insets = size / BW (time per iteration), as in the paper.
Sizes where the benchmark moves no data (put/get_bw: size < 8 B * nCTAs) are dropped and reported.
"""
import argparse, csv, math, re, statistics, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw"
FIG = HERE / "results" / "figures"
TAB = HERE / "results" / "tables"


# ----------------------------------------------------------------------------- parsing
def _bench_ok(text):
    return re.search(r"^#BENCH_EXIT 0\s*$", text, re.M) is not None


def parse_nvshmem_table(text, value_col_pred):
    """Return {size: value} from the first NVSHMEM result table (header line starts with 'size(B)')."""
    rows, cols = {}, None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("size(B)"):
            if rows:
                break  # only first table
            # header tokens like 'BW (GB/sec)' contain spaces -> split on 2+ spaces
            cols = [c.strip() for c in re.split(r"\s{2,}", s) if c.strip()]
            continue
        if cols is None:
            continue
        f = s.split()
        if rows and s.startswith("#"):
            break  # next table title -> first table finished
        # Data rows: integer size + exactly one token per header column. Anything else (other ranks' stdout such as
        # 'mype: 15 mype_node: 7 ...' interleaved into PE 0's table, srun messages, blank lines) is skipped, not a stop.
        if not f or not f[0].isdigit() or len(f) != len(cols):
            continue
        idx = next(i for i, c in enumerate(cols) if value_col_pred(c))
        try:
            rows[int(f[0])] = float(f[idx])
        except (ValueError, IndexError):
            pass
    return rows


def parse_p2p(text):
    return parse_nvshmem_table(text, lambda c: c.startswith("BW"))


def parse_nvs_coll(text):
    return parse_nvshmem_table(text, lambda c: c.startswith("latency"))


def parse_nccl(text):
    """nccl-tests: out-of-place time (us) column; also returns #wrong values seen."""
    rows = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        f = s.split()
        if len(f) < 8 or not f[0].isdigit() or not f[1].isdigit():
            continue
        try:
            rows[int(f[0])] = float(f[5])
        except ValueError:
            pass
    return rows


def collect(prefix, parser, raw=RAW):
    """{size: [v_trial...]} over all good trial logs named <prefix>_t<k>.log; also count of trials."""
    out, ntr = defaultdict(list), 0
    for log in sorted(raw.glob(f"{prefix}_t*.log"), key=lambda p: int(re.search(r"_t(\d+)\.log$", p.name).group(1))):
        if not re.fullmatch(re.escape(prefix) + r"_t\d+\.log", log.name):
            continue
        text = log.read_text(errors="ignore")
        if not _bench_ok(text):
            continue
        d = parser(text)
        if not d:
            continue
        ntr += 1
        for k, v in d.items():
            out[k].append(v)
    return out, ntr


def ms(vals):
    if not vals:
        return float("nan"), float("nan")
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


# ----------------------------------------------------------------------------- series
def ctas_of(prefix, default=32):
    """CTA count actually used by a series, read from the logged command line (-c N)."""
    for log in sorted(RAW.glob(f"{prefix}_t*.log")):
        m = re.search(r"^#BENCH cmd: .* -c (\d+)", log.read_text(errors="ignore"), re.M)
        if m:
            return int(m.group(1))
    return default


def p2p_series(prefix, ctas=None):
    d, n = collect(prefix, parse_p2p)
    ctas = ctas_of(prefix) if ctas is None else ctas
    # put/get_bw split the message as floor(len/nCTAs) doubles per CTA -> sizes < 8 B * nCTAs move no data
    dropped = [s for s in d if prefix.split("_")[-1] in ("put", "get") and s < 8 * ctas]
    for s in dropped:
        d.pop(s)
    return d, n, sorted(dropped)


def ar_series(prefix):
    parser = parse_nccl if "_nccl_" in prefix else parse_nvs_coll
    d, n = collect(prefix, parser)
    # convert latency(us) -> algbw GB/s, keep latency too
    bw = {s: [s / (v * 1e3) for v in vals if v > 0] for s, vals in d.items()}
    return d, bw, n


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------------- plotting helpers
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def _pow2_ticks(ax, lo, hi):
    ticks = [2 ** k for k in range(lo, hi + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$2^{{{k}}}$" for k in range(lo, hi + 1)], rotation=45)
    ax.minorticks_off()
    ax.set_xlim(2 ** lo / 1.3, 2 ** hi * 1.3)


def _plain_log_ticks(ax):
    from matplotlib.ticker import FormatStrFormatter, NullFormatter, LogLocator
    if ax.get_yscale() != "log":
        return
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _peak_label(ax, sizes, means, color, xy_text):
    i = max(range(len(means)), key=lambda k: means[k])
    ax.annotate(f"{means[i]:.1f}", xy=(sizes[i], means[i]), xytext=xy_text, textcoords="axes fraction",
                color=color, fontsize=9, arrowprops=dict(arrowstyle="->", color=color, lw=1),
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))


# ----------------------------------------------------------------------------- figures
GREENS = {"put": "#8fbc3f", "get": "#2e8b57", "p": "#8fbc3f", "g": "#2e8b57"}
STY = {"put": ("-", "o", "white"), "get": ("-", "s", None), "p": ("--", "o", None), "g": ("--", "s", None)}


def fig_p2p(cfg, out_name, report, legend_title=None):
    """cfg: {'intra': (prefix_fmt, ctas, title), 'inter': (...)}; prefix_fmt has {api}."""
    plt = _setup_mpl()
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(4.96, 3.2)
    lo, hi = 8, 25
    for ax, scen in zip(axes, ("intra", "inter")):
        pfx_fmt, ctas, title, ref, ref_label = cfg[scen]
        ins = ax.inset_axes([0.56, 0.08, 0.42, 0.38] if scen == "intra" else [0.56, 0.08, 0.42, 0.38])
        best = (None, -1, None, None)
        for api in ("put", "get", "p", "g"):
            prefix = pfx_fmt.format(api=api)
            d, n, dropped = p2p_series(prefix)
            if not d:
                report.append(f"MISSING {prefix}")
                continue
            sizes = [s for s in sorted(d) if 2 ** lo <= s <= 2 ** hi]
            m = [ms(d[s])[0] for s in sizes]
            sd = [ms(d[s])[1] for s in sizes]
            ls, mk, mfc = STY[api]
            c = GREENS[api]
            ax.errorbar(sizes, m, yerr=sd, color=c, ls=ls, marker=mk, ms=3.5, lw=1.4, capsize=1.5,
                        mfc=mfc or c, label=api)
            lat = [s / (v * 1e3) for s, v in zip(sizes, m)]  # us
            sl = [(s, l) for s, l in zip(sizes, lat) if s <= 2 ** 16]
            ins.plot([x for x, _ in sl], [y for _, y in sl], color=c, ls=ls, marker=mk, ms=2.5, lw=1,
                     mfc=mfc or c)
            pk = max(m)
            if pk > best[1]:
                best = (api, pk, sizes, m)
            report.append(f"{prefix}: trials={n} peak={pk:.2f} GB/s @ {sizes[m.index(pk)]} B; "
                          f"lat@256B={(256/(ms(d[256])[0]*1e3)) if 256 in d else float('nan'):.2f}us "
                          f"lat@512B={(512/(ms(d[512])[0]*1e3)) if 512 in d else float('nan'):.2f}us "
                          f"lat@64KiB={(65536/(ms(d[65536])[0]*1e3)) if 65536 in d else float('nan'):.2f}us "
                          f"max_rel_sd={max((ms(d[s])[1]/ms(d[s])[0]) for s in sizes):.3f}"
                          + (f"; dropped zero-byte sizes {dropped}" if dropped else ""))
        ax.axhline(ref, color="#d62728", ls="--", lw=1.3)
        ax.text(0.99, 0.97, f"{ref_label}", transform=ax.transAxes, ha="right", va="top", color="#d62728", fontsize=8.5)
        ax.text(0.42, 0.97, title, transform=ax.transAxes, ha="center", va="top", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylabel("Bandwidth (GB/s)")
        ax.grid(True, which="major", alpha=0.35)
        if best[0]:
            _peak_label(ax, best[2], best[3], GREENS[best[0]], (0.80, 0.55))
        ins.set_xscale("log", base=2)
        ins.set_yscale("log" if scen == "inter" else "linear")
        ins.set_xticks([2 ** k for k in (8, 10, 12, 14, 16)])
        ins.set_xticklabels([f"$2^{{{k}}}$" for k in (8, 10, 12, 14, 16)], fontsize=6.5)
        ins.tick_params(labelsize=6.5)
        ins.set_ylabel("Latency [µs]", fontsize=6.5)
        ins.grid(True, which="both", alpha=0.3)
        _plain_log_ticks(ins)
        ax.set_ylim(top=ref * 8)
    _pow2_ticks(axes[1], lo, hi)
    axes[0].legend(ncol=2, loc="upper left", bbox_to_anchor=(0.0, 0.86), frameon=False,
                   title=legend_title, title_fontsize=7)
    axes[1].set_xlabel("Message size [Bytes]")
    fig.tight_layout(pad=0.3, h_pad=0.6)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{out_name}.pdf")
    fig.savefig(FIG / f"{out_name}.png", dpi=200)
    plt.close(fig)


NCCLP = "ar_rearch"
AR_COL = {"nvshost": ("#8fbc3f", "o"), "nvsdev": ("#2e8b57", "s"), "nccl_ring": ("#2f6f7e", "D"), "nccl_nvls": ("#6fb3e0", "^")}
AR_LAB = {"nvshost": "NVSHMEM [On-stream]", "nvsdev": "NVSHMEM [Device Block]", "nccl_ring": "NCCL [Ring]", "nccl_nvls": "NCCL [NVLS]"}


def fig_ar(cfg, out_name, report, lo=7, hi=29):
    """cfg: {'intra': ({key: prefix}, title, ref, ref_label), 'inter': ...}"""
    plt = _setup_mpl()
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(4.4, 3.0)
    for ax, scen in zip(axes, ("intra", "inter")):
        series, title, ref, ref_label = cfg[scen]
        ins = ax.inset_axes([0.57, 0.10, 0.40, 0.42])
        best = (None, -1, None, None)
        for key, prefix in series.items():
            lat, bw, n = ar_series(prefix)
            if not bw:
                report.append(f"MISSING {prefix}")
                continue
            sizes = [s for s in sorted(bw) if 2 ** lo <= s <= 2 ** hi and bw[s]]
            m = [ms(bw[s])[0] for s in sizes]
            sd = [ms(bw[s])[1] for s in sizes]
            base = key.split("+")[0]
            c, mk = AR_COL.get(base, ("gray", "x"))
            lab = AR_LAB.get(base, key) + (" tuned" if key.endswith("+tuned") else "")
            ax.errorbar(sizes, m, yerr=sd, color=c, marker=mk, ms=3, lw=1.4, capsize=1.5, label=lab)
            ls = [s for s in sizes if s <= 2 ** 16]
            ins.plot(ls, [ms(lat[s])[0] for s in ls], color=c, marker=mk, ms=2.5, lw=1)
            pk = max(m)
            if pk > best[1]:
                best = (base, pk, sizes, m)
            small = [ms(lat[s])[0] for s in ls]
            report.append(f"{prefix}: trials={n} sizes={len(sizes)} [{sizes[0]}..{sizes[-1]}] peak={pk:.2f} GB/s @ {sizes[m.index(pk)]} B; "
                          f"lat(<=64KiB) {min(small):.2f}-{max(small):.2f} us; "
                          f"max_rel_sd={max(ms(lat[s])[1]/ms(lat[s])[0] for s in sizes):.3f}")
        ax.axhline(ref, color="#d62728", ls="--", lw=1.3)
        ax.text(0.99, 0.985, ref_label, transform=ax.transAxes, ha="right", va="top", color="#d62728", fontsize=8.5)
        ax.text(0.40, 0.985, title, transform=ax.transAxes, ha="center", va="top", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylabel("Algorithm BW [GB/s]")
        ax.grid(True, which="major", alpha=0.35)
        ax.set_ylim(top=ref * 12)
        if best[0]:
            _peak_label(ax, best[2], best[3], AR_COL[best[0]][0], (0.72, 0.58))
        ins.set_xscale("log", base=2)
        ins.set_yscale("log" if scen == "inter" else "linear")
        ins.set_xticks([2 ** k for k in range(lo, 17)])
        ins.set_xticklabels([f"$2^{{{k}}}$" if (k - lo) % 3 == 0 else "" for k in range(lo, 17)], fontsize=6)
        ins.tick_params(labelsize=6)
        ins.set_ylabel("Latency [µs]", fontsize=6.5)
        ins.grid(True, which="both", alpha=0.3)
        _plain_log_ticks(ins)
    _pow2_ticks(axes[1], lo, hi)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=True, fontsize=6.5, framealpha=0.85)
    axes[1].set_xlabel("Message size [Bytes]")
    fig.tight_layout(pad=0.3, h_pad=0.6)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{out_name}.pdf")
    fig.savefig(FIG / f"{out_name}.png", dpi=200)
    plt.close(fig)


def dump_tables():
    """One CSV per series prefix with mean/stddev/n (P2P: GB/s; AR: latency us and algbw GB/s)."""
    prefixes = sorted({re.sub(r"_t\d+\.log$", "", p.name) for p in RAW.glob("*_t*.log")})
    for pfx in prefixes:
        if pfx.startswith("p2p"):
            d, n = collect(pfx, parse_p2p)
            rows = [(s, *ms(d[s]), len(d[s])) for s in sorted(d)]
            write_csv(TAB / f"{pfx}.csv", ["size_B", "bw_GBps_mean", "bw_GBps_std", "n"], rows)
        elif pfx.startswith("ar") or pfx.startswith("tune"):
            parser = parse_nccl if "_nccl_" in pfx else (parse_p2p if "p2p" in pfx else parse_nvs_coll)
            d, n = collect(pfx, parser)
            rows = []
            for s in sorted(d):
                lm, lsd = ms(d[s])
                rows.append((s, lm, lsd, s / (lm * 1e3) if lm == lm and lm > 0 else "", len(d[s])))
            write_csv(TAB / f"{pfx}.csv", ["size_B", "lat_us_mean", "lat_us_std", "algbw_GBps_of_mean", "n"], rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvlink", type=float, default=900.0, help="per-GPU per-direction NVLink BW (GB/s)")
    ap.add_argument("--ib", type=float, default=50.0, help="per-NIC IB BW (GB/s)")
    ap.add_argument("--nics", type=int, default=8)
    ap.add_argument("--nccl-path", choices=["rearch", "legacy"], default="rearch",
                    help="NCCL enqueue path for the NCCL series: refactored (NCCL_ENQUEUE_REARCH_ENABLE=1) or legacy")
    ap.add_argument("--suffix", default="", help="suffix for figure file names")
    a = ap.parse_args()
    NCCLP = "ar_rearch" if a.nccl_path == "rearch" else "ar_paper"
    rep = [f"# NCCL series from {NCCLP}_* ({a.nccl_path} enqueue path)"]
    nv = f"NVLink: {a.nvlink:.0f} GB/s"
    fig_p2p({"intra": ("p2p_paper_intra_{api}", 32, "Intra-node, B200", a.nvlink, nv),
             "inter": ("p2p_paper_inter_ibgdatuned_{api}", 64, "Inter-node, B200", a.ib, f"IB: {a.ib:.0f} GB/s")},
            "nvshmem-p2p-bandwidth-onecol" + a.suffix, rep)
    fig_ar({"intra": ({"nvshost": "ar_paper_intra_nvshost", "nvsdev": "ar_paper_intra_nvsdev",
                       "nccl_ring": f"{NCCLP}_intra_nccl_ring", "nccl_nvls": f"{NCCLP}_intra_nccl_nvls"},
                      "Intra-node, 8 B200 GPUs", a.nvlink, nv),
            "inter": ({"nvshost": "ar_paper_inter_nvshost", "nvsdev": "ar_paper_inter_nvsdev",
                       "nccl_ring": f"{NCCLP}_inter_nccl_ring", "nccl_nvls": f"{NCCLP}_inter_nccl_nvls"},
                      "Inter-node, 16 B200 GPUs", a.ib * a.nics, f"{a.nics}xIB: {a.ib*a.nics:.0f} GB/s")},
           "nvshmem-allreduce-bandwidth-onecol" + a.suffix, rep)
    tuned = HERE / "analyze_tuned.py"
    if tuned.exists():
        ns = {"__name__": "analyze_tuned"}
        exec(compile(tuned.read_text(), str(tuned), "exec"), ns)
        ns["main"](sys.modules[__name__], a, rep)
    dump_tables()
    print("\n".join(rep))
