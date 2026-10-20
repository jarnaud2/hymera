#!/usr/bin/env python3
"""Compare the secondary (gamma, xi) samples of LargeAngleCollision (reference) and
LargeAngleCollisionFast written by la_sampler_test.

    la_sampler_check.py SAMPLES.bin [--out PNG]

For each primary (p, xi, gamma_min): two-sample Kolmogorov-Smirnov distance of the
gamma and of the xi marginals against its 1% critical value, and a chi^2 test
of the joint distribution on 10 x 10 quantile bins of the reference sample.
Exit status 1 if any test fails at the 1% level.
"""
import argparse
import sys
from math import erf, sqrt

import numpy as np


def read(path):
    with open(path, "rb") as f:
        nc, n = np.fromfile(f, np.int32, 2)
        cases = []
        for _ in range(nc):
            hdr = np.fromfile(f, np.float64, 3)
            runs = []
            for _ in range(2):
                dt = np.fromfile(f, np.float64, 1)[0]
                s = np.fromfile(f, np.float64, 2 * n).reshape(n, 2)
                runs.append((dt, s))
            cases.append((hdr, runs))
    return cases


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return np.max(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b)))


def chi2_sf(x, k):
    # Wilson-Hilferty approximation of the chi^2 survival function
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / sqrt(2 / (9 * k))
    return 0.5 * (1 - erf(z / sqrt(2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("samples")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cases = read(args.samples)
    ok = True
    rows = []
    for (p, xi, gmin), ((dt_r, r), (dt_f, f)) in cases:
        r = r[r[:, 0] > 0]
        f = f[f[:, 0] > 0]
        crit = 1.63 * np.sqrt(1 / len(r) + 1 / len(f))     # 1% two-sample KS
        dg, dx = ks(r[:, 0], f[:, 0]), ks(r[:, 1], f[:, 1])
        qg = np.quantile(r[:, 0], np.linspace(0, 1, 11)); qg[0], qg[-1] = -np.inf, np.inf
        qx = np.quantile(r[:, 1], np.linspace(0, 1, 11)); qx[0], qx[-1] = -np.inf, np.inf
        hr, _, _ = np.histogram2d(r[:, 0], r[:, 1], [qg, qx])
        hf, _, _ = np.histogram2d(f[:, 0], f[:, 1], [qg, qx])
        m = (hr + hf) > 0
        # two-sample chi^2 with equal sizes scaled
        kr, kf = np.sqrt(len(f) / len(r)), np.sqrt(len(r) / len(f))
        chi2 = np.sum((kr * hr[m] - kf * hf[m]) ** 2 / (hr[m] + hf[m]))
        dof = m.sum() - 1
        pval = chi2_sf(chi2, dof)
        good = dg < crit and dx < crit and pval > 0.01
        ok &= good
        rows.append((p, xi, gmin, r, f))
        print(f"{'ok  ' if good else 'FAIL'} p={p:5.1f} xi={xi:+.2f} gmin={gmin:.3f}: "
              f"KS gamma {dg:.4f} xi {dx:.4f} (crit {crit:.4f}), joint chi2 {chi2:.1f}/{dof} "
              f"p={pval:.3f}; time ref {dt_r:.3f} s fast {dt_f:.3f} s")
    if args.out:
        plot(rows, args.out)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.2), constrained_layout=True)
    for k, (p, xi, gmin, r, f) in enumerate(rows):
        ax = axes[0, k]
        e_r, e_f = r[:, 0] - 1, f[:, 0] - 1
        bins = np.geomspace(min(e_r.min(), e_f.min()), max(e_r.max(), e_f.max()), 40)
        ax.hist(e_r, bins, histtype="step", color="#2a78d6", density=True, label="reference")
        ax.hist(e_f, bins, histtype="step", color="#eb6834", ls="--", density=True, label="fast")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"p={p:g}, ξ={xi:+.2f}\nγmin={gmin:g}", fontsize=8)
        ax.set_xlabel(r"$\gamma_s - 1$", fontsize=8)
        ax = axes[1, k]
        bins = np.linspace(-1, 1, 41)
        ax.hist(r[:, 1], bins, histtype="step", color="#2a78d6", density=True)
        ax.hist(f[:, 1], bins, histtype="step", color="#eb6834", ls="--", density=True)
        ax.set_xlabel(r"$\xi_s$", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(out, dpi=140)


if __name__ == "__main__":
    sys.exit(main())
