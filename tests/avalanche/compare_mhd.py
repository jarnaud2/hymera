#!/usr/bin/env python3
"""fig11a on the MHD field against the analytic baseline and the reference.

    compare_mhd.py --mhd LABEL=DIR [--mhd LABEL=DIR ...] [--out PNG]

Each DIR holds fig11a_<i> runs of fig11a_mhd.input, seeded on the flux surface
psi_n(r_0) of table row i. The growth rate is fitted as in fig11a.py. The figure
shows gamma against E_0, one colour per seed surface (legend: psi_n, 3 digits):
the reference (fig11a.jld2, open circles), the analytic baseline
(baseline_analytic.csv, small filled circles) and each MHD set (its own marker).
The lower panel shows gamma / gamma_ref.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fig11a import reference, growth_rate, psi_n_of_r0  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
MARKERS = ["s", "D", "^", "v"]


def baseline():
    rows = [r for r in csv.DictReader(l for l in open(HERE / "baseline_analytic.csv")
                                      if not l.startswith("#"))]
    return {int(r["row"]): float(r["gamma"]) for r in rows}


def mhd_runs(d):
    out = {}
    for run in sorted(Path(d).glob("fig11a_*")):
        g = growth_rate(run)
        if g is not None:
            out[int(run.name.split("_")[1])] = g[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mhd", action="append", required=True, help="LABEL=DIR")
    ap.add_argument("--out", type=Path, default=HERE / "fig11a_mhd_vs_baseline.png")
    args = ap.parse_args()

    ES, R0s, _, gref = reference()
    base = baseline()
    sets = []
    for spec in args.mhd:
        label, d = spec.rsplit("=", 1)
        sets.append((label, mhd_runs(d)))
    if not any(m for _, m in sets):
        print("no MHD runs found")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 7.4), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2, 1]),
                                  constrained_layout=True)
    r0_vals = sorted(set(np.round(R0s, 6)))
    for r0, c in zip(r0_vals, COLORS):
        m = np.isclose(R0s, r0)
        idx = np.flatnonzero(m)
        ax.plot(ES[m], gref[m], "o", mfc="none", mec=c, ms=8, mew=1.2)
        ax.plot(ES[m], [base[i] for i in idx], "o", color=c, ms=3.5,
                label=rf"$\psi_n$ = {psi_n_of_r0(r0):.3g}")
        axr.plot(ES[m], np.array([base[i] for i in idx]) / gref[m], "o", color=c, ms=3.5)
        for (label, runs), mk in zip(sets, MARKERS):
            pts = sorted((ES[i], runs[i], gref[i]) for i in idx if i in runs)
            if pts:
                e, g, gr = map(np.array, zip(*pts))
                ax.plot(e, g, mk, color=c, ms=7, mec="black", mew=0.6)
                axr.plot(e, g / gr, mk, color=c, ms=7, mec="black", mew=0.6)
    ax.plot([], [], "o", mfc="none", mec="0.3", ms=8, label="reference (fig11a.jld2)")
    ax.plot([], [], "o", color="0.3", ms=3.5, label="analytic field (baseline)")
    for (label, _), mk in zip(sets, MARKERS):
        ax.plot([], [], mk, color="0.6", mec="black", mew=0.6, ms=7, label=f"MHD field, {label}")
    ax.set_ylabel(r"growth rate $\gamma$ [$1/\tau_c$]")
    ax.set_title("fig11a: MHD field and analytic baseline against the reference", loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    axr.axhspan(0.9, 1.1, color="0.92", zorder=0)
    axr.axhline(1.0, color="0.4", lw=0.8)
    axr.set_ylabel(r"$\gamma / \gamma_{\rm ref}$")
    axr.set_xlabel(r"$E_0$")
    axr.grid(True, alpha=0.3)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    print(f"{'row':>4} {'psi_n':>6} {'E0':>6} {'ref':>7} {'analytic':>17} "
          + " ".join(f"{l[:17]:>17}" for l, _ in sets))
    for i in sorted({i for _, m in sets for i in m}):
        vals = " ".join(f"{m[i]:8.4f} ({m[i] / gref[i] - 1:+.3f})" if i in m else " " * 17
                        for _, m in sets)
        print(f"{i:4d} {psi_n_of_r0(R0s[i]):6.3g} {ES[i]:6.2f} {gref[i]:7.4f} "
              f"{base[i]:8.4f} ({base[i] / gref[i] - 1:+.3f}) {vals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
