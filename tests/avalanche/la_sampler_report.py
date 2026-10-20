#!/usr/bin/env python3
"""Figures and tables for the large-angle sampler report.

    la_sampler_report.py --bench la_bench_a100.csv [--ref DIR --fast DIR] --out DIR

--bench  CSV from la_sampler_bench (time per secondary of both samplers)
--ref, --fast  fig11a run directories (fig11a_<i>) with the reference and the
         fast sampler at the same settings; their growth rates are plotted
         against E_0 with the reference table, and their wall times compared.
Writes OUT/la_speedup.png, OUT/la_fig11a_samplers.png and OUT/la_report.tex
(table fragments).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def bench(path):
    rows = list(csv.DictReader(open(path)))
    key = lambda r: (float(r["gamma_min"]), float(r["p"]), float(r["xi"]))
    ref = {key(r): float(r["ns_per_secondary"]) for r in rows if r["sampler"] == "reference"}
    fast = {key(r): float(r["ns_per_secondary"]) for r in rows if r["sampler"] == "fast"}
    return ref, fast


def plot_speedup(ref, fast, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gs = sorted({k[0] for k in ref})
    ps = sorted({k[1] for k in ref})
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    x = np.arange(len(gs))
    geo = []
    for i, g in enumerate(gs):
        s = np.array([ref[k] / fast[k] for k in ref if k[0] == g])
        gm = np.exp(np.mean(np.log(s)))
        geo.append(gm)
        ax.bar(i, gm, color="#2a78d6", width=0.6)
        ax.plot([i, i], [s.min(), s.max()], color="black", lw=1.2)
        ax.text(i, gm * 1.25, f"×{gm:,.0f}", ha="center", fontsize=9)
    cm = plt.get_cmap("viridis")
    for j, p in enumerate(ps):
        ys = [np.exp(np.mean(np.log([ref[k] / fast[k] for k in ref if k[0] == g and k[1] == p]))) for g in gs]
        ax.plot(x + (j - (len(ps) - 1) / 2) * 0.09, ys, "o", color=cm(j / max(1, len(ps) - 1)), ms=4,
                label=f"p = {p:g}")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{g:g}" for g in gs])
    ax.set_xlabel(r"secondary threshold $\gamma_{\min}$")
    ax.set_ylabel("speed-up (reference / fast time per secondary)")
    ax.set_title("A100: speed-up of the fast sampler\nbar: geometric mean over p and ξ; line: min–max", loc="left", fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=7, title="primary momentum", title_fontsize=7, loc="upper right")
    em = np.array(gs) - 1.0
    for j, p in enumerate(ps):
        c = cm(j / max(1, len(ps) - 1))
        axt.plot(em, [np.mean([ref[k] for k in ref if k[0] == g and k[1] == p]) for g in gs], "o-", color=c,
                 label=f"reference, p = {p:g}")
        axt.plot(em, [np.mean([fast[k] for k in fast if k[0] == g and k[1] == p]) for g in gs], "s--", color=c, ms=4)
    axt.plot([], [], "o-", color="0.4", label="reference sampler")
    axt.plot([], [], "s--", color="0.4", label="fast sampler")
    axt.set_xscale("log")
    axt.set_yscale("log")
    axt.set_xticks(em, [f"{e:g}" for e in em])
    axt.minorticks_off()
    axt.set_xlabel(r"$\gamma_{\min} - 1$")
    axt.set_ylabel("ns per secondary (A100)")
    axt.set_title("Time per secondary", loc="left", fontsize=9)
    axt.grid(True, which="both", alpha=0.3)
    axt.legend(fontsize=7, ncol=2)
    fig.savefig(out / "la_speedup.png", dpi=150)
    plt.close(fig)
    return gs, geo


def growth(d):
    from fig11a import growth_rate
    out = {}
    for run in sorted(Path(d).glob("fig11a_*")):
        g = growth_rate(run)
        t = run / "time.txt"
        if g is not None:
            out[int(run.name.split("_")[1])] = (g[0], float(t.read_text()) if t.exists() else float("nan"))
    return out


def plot_fig11a(ref_d, fast_d, out):
    from fig11a import reference, psi_n_of_r0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ES, R0s, _, G = reference()
    ref, fast = growth(ref_d), growth(fast_d)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2, 1]), constrained_layout=True)
    for r0, c in zip(sorted(set(np.round(R0s, 6))), cols):
        m = np.isclose(R0s, r0)
        ax.plot(ES[m], G[m], "o", mfc="none", mec=c, ms=8, mew=1.2)
        for d, mk, ms, lab in [(ref, "D", 6, "reference sampler"), (fast, "s", 6, "fast sampler")]:
            pts = sorted((ES[i], d[i][0], G[i]) for i in np.flatnonzero(m) if i in d)
            if pts:
                e, g, gr = map(np.array, zip(*pts))
                ax.plot(e, g, mk, color=c, mec="black", mew=0.5, ms=ms)
                axr.plot(e, g / gr, mk, color=c, mec="black", mew=0.5, ms=ms)
        ax.plot([], [], "o", color=c, label=rf"$\psi_n$ = {psi_n_of_r0(r0):.3g}")
    ax.plot([], [], "o", mfc="none", mec="0.3", ms=8, label="reference (fig11a.jld2)")
    ax.plot([], [], "D", color="0.6", mec="black", mew=0.5, ms=6, label="reference LA sampler, A100")
    ax.plot([], [], "s", color="0.6", mec="black", mew=0.5, ms=6, label="fast LA sampler, A100")
    ax.set_ylabel(r"$\gamma$ [$1/\tau_c$]")
    ax.set_title(r"fig11a with both large-angle samplers ($\gamma_{\min}=1.02$, 1% growth per LA step, rk4 $2\times10^{-6}$)",
                 loc="left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    axr.axhspan(0.9, 1.1, color="0.92", zorder=0)
    axr.axhline(1, color="0.4", lw=0.8)
    axr.set_ylabel(r"$\gamma/\gamma_{\rm ref}$")
    axr.set_xlabel(r"$E_0$")
    axr.grid(True, alpha=0.3)
    fig.savefig(out / "la_fig11a_samplers.png", dpi=150)
    plt.close(fig)
    return ref, fast, ES, R0s, G


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--fast")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ref, fast = bench(args.bench)
    gs, geo = plot_speedup(ref, fast, args.out)
    L = [r"\begin{tabular}{rrrr}\toprule $\gamma_{\min}$ & geo.\ mean & min & max\\\midrule"]
    for g, gm in zip(gs, geo):
        s = [ref[k] / fast[k] for k in ref if k[0] == g]
        L.append(rf"{g:g} & {gm:,.0f} & {min(s):,.0f} & {max(s):,.0f}\\")
    L.append(r"\bottomrule\end{tabular}")
    tex = ["% speed-up table", "\n".join(L)]
    if args.ref and args.fast:
        r, f, ES, R0s, G = plot_fig11a(args.ref, args.fast, args.out)
        L = [r"\begin{tabular}{rrrrrrrrr}\toprule row & $r_0$ & $E_0$ & $\gamma_{\rm ref}$ & reference & fast & fast/ref$-1$ & $t_{\rm ref}$ [s] & $t_{\rm fast}$ [s]\\\midrule"]
        for i in sorted(set(r) & set(f)):
            L.append(rf"{i} & {R0s[i]:.1f} & {ES[i]:.2f} & {G[i]:.4f} & {r[i][0]:.4f} & {f[i][0]:.4f} & "
                     rf"{f[i][0] / r[i][0] - 1:+.3f} & {r[i][1]:.0f} & {f[i][1]:.0f}\\")
        L.append(r"\bottomrule\end{tabular}")
        tex += ["% fig11a both samplers", "\n".join(L)]
        for i in sorted(set(r) & set(f)):
            print(f"row {i:2d}: ref {r[i][0]:.4f} ({r[i][1]:.0f} s)  fast {f[i][0]:.4f} ({f[i][1]:.0f} s)  "
                  f"diff {f[i][0] / r[i][0] - 1:+.3f}  vs table ref {r[i][0]/G[i]-1:+.3f} / {f[i][0]/G[i]-1:+.3f}")
    (args.out / "la_report.tex").write_text("\n".join(tex) + "\n")
    for g, gm in zip(gs, geo):
        print(f"gamma_min {g:g}: geometric-mean speed-up {gm:,.0f}")


if __name__ == "__main__":
    sys.exit(main())
