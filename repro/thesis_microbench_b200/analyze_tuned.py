"""Tuned-NVSHMEM variants of the two paper figures + a paper-vs-tuned peak table.
Loaded by analyze.py (main(mod, args, report)). Series with no tuned data fall back to the paper config.
"""


def main(mod, a, rep):
    def pick(tuned, paper):
        return tuned if list(mod.RAW.glob(f"{tuned}_t*.log")) else paper

    nv = f"NVLink: {a.nvlink:.0f} GB/s"
    rep.append("\n# ---- tuned NVSHMEM figures ----")
    p2p = {}
    for scen, paper_fmt in (("intra", "p2p_paper_intra_{api}"), ("inter", "p2p_paper_inter_ibgdatuned_{api}")):
        p2p[scen] = {api: pick(f"p2p_tuned_{scen}_{api}", paper_fmt.format(api=api)) for api in ("put", "get", "p", "g")}

    # fig_p2p takes a format string; use a tiny shim mapping {api} -> chosen prefix
    class Fmt(str):
        def __new__(cls, m):
            o = str.__new__(cls, "x"); o.m = m; return o
        def format(self, api):
            return self.m[api]

    mod.fig_p2p({"intra": (Fmt(p2p["intra"]), None, "Intra-node, B200", a.nvlink, nv),
                 "inter": (Fmt(p2p["inter"]), None, "Inter-node, B200", a.ib, f"IB: {a.ib:.0f} GB/s")},
                "nvshmem-p2p-bandwidth-onecol-tuned" + a.suffix, rep, legend_title="tuned NVSHMEM configs")
    ar = {}
    for scen in ("intra", "inter"):
        ar[scen] = {}
        for key in ("nvshost", "nvsdev"):
            pfx = pick(f"ar_tuned_{scen}_{key}", f"ar_paper_{scen}_{key}")
            ar[scen][key + ("+tuned" if "_tuned_" in pfx else "")] = pfx
        ar[scen]["nccl_ring"] = f"{mod.NCCLP}_{scen}_nccl_ring"
        ar[scen]["nccl_nvls"] = f"{mod.NCCLP}_{scen}_nccl_nvls"
    mod.fig_ar({"intra": (ar["intra"], "Intra-node, 8 B200 GPUs", a.nvlink, nv),
                "inter": (ar["inter"], "Inter-node, 16 B200 GPUs", a.ib * a.nics, f"{a.nics}xIB: {a.ib*a.nics:.0f} GB/s")},
               "nvshmem-allreduce-bandwidth-onecol-tuned" + a.suffix, rep)

    # paper-config vs tuned peak table (plotted range only)
    rows = []
    for scen in ("intra", "inter"):
        for api in ("put", "get", "p", "g"):
            paper = f"p2p_paper_intra_{api}" if scen == "intra" else f"p2p_paper_inter_ibgdatuned_{api}"
            for label, pfx in (("paper", paper), ("tuned", p2p[scen][api])):
                d, n, _ = mod.p2p_series(pfx)
                if d:
                    sz = [s for s in sorted(d) if 2 ** 8 <= s <= 2 ** 25]
                    m = [mod.ms(d[s])[0] for s in sz]
                    rows.append((f"p2p {scen} {api}", label, pfx, n, max(m), sz[m.index(max(m))]))
        for key in ("nvshost", "nvsdev"):
            tuned_pfx = ar[scen].get(key + "+tuned", ar[scen].get(key))
            for label, pfx in (("paper", f"ar_paper_{scen}_{key}"), ("tuned", tuned_pfx)):
                lat, bw, n = mod.ar_series(pfx)
                if bw:
                    sz = [s for s in sorted(bw) if 2 ** 7 <= s <= 2 ** 29 and bw[s]]
                    m = [mod.ms(bw[s])[0] for s in sz]
                    rows.append((f"allreduce {scen} {key}", label, pfx, n, max(m), sz[m.index(max(m))]))
    mod.write_csv(mod.TAB / "peaks_paper_vs_tuned.csv", ["series", "config", "prefix", "trials", "peak_GBps", "at_size_B"], rows)
    rep.append("series | config | prefix | trials | peak GB/s | @size")
    rep.extend(f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.2f} | {r[5]}" for r in rows)
