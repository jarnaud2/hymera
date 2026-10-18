#!/usr/bin/env python3
"""fig11a avalanche benchmark: growth rates against the reference (tests/avalanche).

    fig11a.py DIR [DIR ...] [--out OUTDIR] [--label NAME] [--tol 0.10]

Each DIR holds runs fig11a_<i> (one per row i of the reference table in
fig11a.jld2: ES, R0s, tlims, gamma_av). From each run the growth rate gamma is
fitted as avalanche.jl does it:

  * take the last 50 of the 100 outputs (t, weighted particle count N),
  * normalise y = N / N(first of these 50), t' = (t - t(first)) / tau_c,
  * fit y = exp(gamma t') by least squares (LsqFit in the notebook; Gauss-Newton
    here, the same objective).

N is read from <problem_id>.diag.h5 /timeseries/avalanche (W_alive, the sum of
the weights of alive particles) or, if that is missing, from the .phdf outputs
as the notebook does (sum over blocks of counts times the weight of the first
particle of the block).

Writes OUTDIR/gamma.csv and OUTDIR/fig11a_gamma_vs_E0.png (gamma against E_0,
one colour per r_0; reference open markers, this code filled markers) and
OUTDIR/fig11a_gamma_vs_psin.png (the same against E_0, labelled by the
normalised flux psi_n of the seed surface).
Exit status 1 if any run is off the reference by more than --tol (relative).
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
REF = HERE / "fig11a.jld2"
N_FIT = 50
Q0, Q2 = 2.1, 2.0     # AnalyticField/q0, q2 of the reference runs


def reference():
    with h5py.File(REF, "r") as f:
        ES = f["ES"][()].ravel()
        R0s = f["R0s"][()].ravel()
        tlims = f["tlims"][()].ravel()
        gam = f["γ_av"][()].ravel()
    return ES, R0s, tlims, gam


def psi_n_of_r0(r0):
    """Normalised flux of the analytic field at minor radius r0 (edge r = 1)."""
    return np.log1p(Q2 * np.asarray(r0) ** 2 / Q0) / np.log1p(Q2 / Q0)


def fit_exponential(t, y, g0=None):
    """Least-squares fit of y = exp(g t): Gauss-Newton on sum (exp(g t) - y)^2.
    Starts from the log-linear slope, so large growth does not overflow."""
    g = g0 if g0 is not None else np.polyfit(t, np.log(np.maximum(y, 1e-300)), 1)[0]
    for _ in range(200):
        e = np.exp(g * t)
        r = e - y
        J = t * e
        step = (J @ r) / (J @ J)
        g -= step
        if abs(step) < 1e-14 * max(1.0, abs(g)):
            break
    return g


def series_from_diag(run):
    files = glob.glob(str(run / "*.diag.h5"))
    if not files:
        return None
    with h5py.File(files[0], "r") as f:
        if "timeseries/avalanche" not in f:
            return None
        g = f["timeseries/avalanche"]
        return g["t"][()], g["W_alive"][()]


def series_from_phdf(run):
    files = sorted(glob.glob(str(run / "*.out2.0*.phdf")))
    if not files:
        return None
    t, n = [], []
    for fn in files:
        with h5py.File(fn, "r") as f:
            t.append(f["Info"].attrs["Time"])
            counts = f["particles/counts"][()]
            offsets = f["particles/offsets"][()]
            w = f["particles/SwarmVars/particle.weight"][()]
            n.append(sum(c * (w[o] if c > 0 else 0.0) for c, o in zip(counts, offsets)))
    return np.array(t), np.array(n)


def tau_c(run):
    """tau_c [s] from an output (Params/Deck/tau_c); the reference value otherwise."""
    for fn in sorted(glob.glob(str(run / "*.phdf")))[:1]:
        with h5py.File(fn, "r") as f:
            return float(f["Params"].attrs["Deck/tau_c"])
    return 0.016713825101654162


def growth_rate(run):
    s = series_from_diag(run) or series_from_phdf(run)
    if s is None:
        return None
    t, n = s
    # The final output (use_final_label) can repeat the last time; keep unique times.
    t, idx = np.unique(t, return_index=True)
    n = n[idx]
    t, n = t[-N_FIT:], n[-N_FIT:]
    tc = tau_c(run)
    return fit_exponential((t - t[0]) / tc, n / n[0]), len(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label", default="hymera")
    ap.add_argument("--tol", type=float, default=0.10)
    args = ap.parse_args()
    out = args.out or args.dirs[0]
    out.mkdir(parents=True, exist_ok=True)

    ES, R0s, tlims, gam_ref = reference()
    rows = []
    for d in args.dirs:
        for i in range(len(ES)):
            run = d / f"fig11a_{i}"
            if not run.exists():
                continue
            g = growth_rate(run)
            if g is None:
                continue
            rows.append(dict(i=i, E0=ES[i], r0=R0s[i], psi_n=float(psi_n_of_r0(R0s[i])),
                             tlim=tlims[i], gamma_ref=gam_ref[i], gamma=g[0], n_fit=g[1],
                             rel=(g[0] - gam_ref[i]) / gam_ref[i]))
    if not rows:
        print("no runs found")
        return 1

    with open(out / "gamma.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if not abs(r["rel"]) <= args.tol]   # NaN counts as a failure
    for r in rows:
        print(f"{'ok  ' if abs(r['rel']) <= args.tol else 'FAIL'} i={r['i']:2d} r0={r['r0']:.1f} "
              f"E0={r['E0']:6.2f} gamma={r['gamma']:.4f} ref={r['gamma_ref']:.4f} "
              f"rel={r['rel']:+.3f} (n_fit {r['n_fit']})")
    print(f"{len(rows) - len(bad)}/{len(rows)} within {args.tol:.0%}")

    plot(rows, ES, R0s, gam_ref, out, args.label)
    return 1 if bad else 0


def plot(rows, ES, R0s, gam_ref, out, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r0_vals = sorted(set(np.round(R0s, 6)))
    colors = dict(zip(r0_vals, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for r0 in r0_vals:
        m = np.isclose(R0s, r0)
        ax.plot(ES[m], gam_ref[m], "o", mfc="none", mec=colors[r0], ms=7, mew=1.3)
        mine = [r for r in rows if np.isclose(r["r0"], r0)]
        if mine:
            ax.plot([r["E0"] for r in mine], [r["gamma"] for r in mine], "o",
                    color=colors[r0], ms=4.5, label=f"$r_0$ = {r0:g}")
        else:
            ax.plot([], [], "o", color=colors[r0], ms=4.5, label=f"$r_0$ = {r0:g}")
    ax.plot([], [], "o", mfc="none", mec="0.3", label="reference (fig11a.jld2)")
    ax.plot([], [], "o", color="0.3", ms=4.5, label=label)
    ax.set_xlabel(r"$E_0$")
    ax.set_ylabel(r"avalanche growth rate $\gamma$ [$1/\tau_c$]")
    ax.set_title("fig11a: avalanche growth rate, analytic field", loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(out / "fig11a_gamma_vs_E0.png", dpi=150)
    plt.close(fig)

    # The same data with the seed surfaces labelled by their normalised flux, so
    # that the curves carry over to the MHD field (seeded on the same psi_n).
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for r0 in r0_vals:
        m = np.isclose(R0s, r0)
        lab = rf"$\psi_n$ = {psi_n_of_r0(r0):.3g}"
        ax.plot(ES[m], gam_ref[m], "o", mfc="none", mec=colors[r0], ms=7, mew=1.3)
        mine = [r for r in rows if np.isclose(r["r0"], r0)]
        ax.plot([r["E0"] for r in mine], [r["gamma"] for r in mine], "o",
                color=colors[r0], ms=4.5, label=lab)
    ax.plot([], [], "o", mfc="none", mec="0.3", label="reference (fig11a.jld2)")
    ax.plot([], [], "o", color="0.3", ms=4.5, label=label)
    ax.set_xlabel(r"$E_0$")
    ax.set_ylabel(r"avalanche growth rate $\gamma$ [$1/\tau_c$]")
    ax.set_title(r"fig11a: growth rate by seed flux surface $\psi_n$", loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(out / "fig11a_gamma_vs_psin.png", dpi=150)
    plt.close(fig)
    print(f"figures in {out}")


if __name__ == "__main__":
    sys.exit(main())
