#!/usr/bin/env python3
"""Figures for the GCE conservation scan (see scan.py).

    plot.py DIR

Reads DIR/summary.csv and DIR/runs/*/*/single_particle.dat and writes:
  DIR/figures/error_time_rk4.png     |p_phi(t)-p_phi(0)| (solid) and |mu(t)-mu(0)|
  DIR/figures/error_time_dopri5.png  (dashed), relative, averaged over particles;
                                     one colour per hRK (rk4) or rtol (dopri5)
  DIR/figures/error_vs_cost.png      end-of-run error against wall time, both integrators
  DIR/figures/orbits.png             R-Z orbit of each test particle (finest dopri5 run)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Sequential single-hue ramp for an ordered parameter (step size / tolerance):
# light = coarse, dark = fine.
def ramp(n):
    return [plt.get_cmap("Blues")(x) for x in np.linspace(0.35, 1.0, n)]


def load_runs(out):
    rows = list(csv.DictReader(open(out / "summary.csv")))
    for r in rows:
        r["failed"] = int(r["failed"])
        r["wall_s"] = float(r["wall_s"])
        r["dat"] = out / "runs" / f"{r['integrator']}_{r['setting']}" / r["particle"] / "r1" / "r2" / "single_particle.dat"
    return rows


def rel_error_series(dat):
    a = np.loadtxt(dat, ndmin=2)
    t, pp, mu = a[:, 0], a[:, 6], a[:, 7]
    return t, np.abs(pp - pp[0]) / abs(pp[0]), np.abs(mu - mu[0]) / abs(mu[0])


def mean_over_particles(runs):
    """Mean of the relative errors over the particles that completed, on the
    common output times (every run writes at the same dt_out)."""
    series = [rel_error_series(r["dat"]) for r in runs if not r["failed"]]
    if not series:
        return None
    n = min(len(s[0]) for s in series)
    t = series[0][0][:n]
    dpp = np.mean([s[1][:n] for s in series], axis=0)
    dmu = np.mean([s[2][:n] for s in series], axis=0)
    return t, dpp, dmu, len(series)


def error_time_plot(rows, integ, key, label_fmt, title, path, t_orbit):
    settings = sorted({(float(r[key]), r["setting"]) for r in rows if r["integrator"] == integ},
                      reverse=True)
    colors = ramp(len(settings))
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    n_parts = len({r["particle"] for r in rows})
    for (val, setting), c in zip(settings, colors):
        runs = [r for r in rows if r["integrator"] == integ and r["setting"] == setting]
        m = mean_over_particles(runs)
        if m is None:
            ax.plot([], [], color=c, label=f"{label_fmt(val)}: all particles failed")
            continue
        t, dpp, dmu, n = m
        x = t / t_orbit
        tag = label_fmt(val) + ("" if n == n_parts else f" ({n}/{n_parts} particles)")
        ax.plot(x[1:], dpp[1:], color=c, lw=1.8, label=tag)
        ax.plot(x[1:], dmu[1:], color=c, lw=1.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("time [poloidal transit times of the reference orbit]")
    ax.set_ylabel("relative error, mean over particles")
    ax.set_title(title, loc="left")
    # Line-style key, separate from the colour legend.
    style = [plt.Line2D([], [], color="0.3", lw=1.8, label=r"$|p_\phi(t)-p_\phi(0)|/|p_\phi(0)|$"),
             plt.Line2D([], [], color="0.3", lw=1.8, ls="--", label=r"$|\mu(t)-\mu(0)|/\mu(0)$")]
    leg1 = ax.legend(handles=style, loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(loc="lower right", fontsize=8, title=key if key != "hRK" else r"$h_{RK}$ [$\tau_c$]")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_vs_cost(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True, sharey=True)
    for ax, (col, name) in zip(axes, [("end_dpphi", r"$p_\phi$"), ("end_dmu", r"$\mu$")]):
        for integ, color, marker in [("rk4", "#2a78d6", "o"), ("dopri5", "#eb6834", "s")]:
            pts = {}
            for r in rows:
                if r["integrator"] != integ or r["failed"]:
                    continue
                pts.setdefault(r["setting"], []).append((r["wall_s"], float(r[col])))
            xs = [np.mean([p[0] for p in v]) for v in pts.values()]
            ys = [np.mean([p[1] for p in v]) for v in pts.values()]
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], marker=marker, color=color, label=integ)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("wall time per run [s], mean over particles")
        ax.set_title(f"End-of-run relative error in {name}", loc="left")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("relative error, mean over particles")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def orbits(rows, path):
    dp = sorted({(float(r["rtol"]), r["setting"]) for r in rows if r["integrator"] == "dopri5"})
    if not dp:
        return
    finest = dp[0][1]
    runs = [r for r in rows if r["integrator"] == "dopri5" and r["setting"] == finest and not r["failed"]]
    fig, ax = plt.subplots(figsize=(5, 6), constrained_layout=True)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    for r, c in zip(runs, palette):
        a = np.loadtxt(r["dat"], ndmin=2)
        ax.plot(a[:, 3], a[:, 5], color=c, lw=1.0, label=r["particle"])
        ax.plot(a[0, 3], a[0, 5], "o", color=c, ms=6)
    ax.set_aspect("equal")
    ax.set_xlabel("R [a]"); ax.set_ylabel("Z [a]")
    ax.set_title(f"Test orbits (dopri5 {finest}); dots: start", loc="left", fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    out = Path(sys.argv[1])
    figs = out / "figures"; figs.mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scan import T_ORBIT  # noqa: E402
    rows = load_runs(out)
    error_time_plot(rows, "rk4", "hRK", lambda v: f"{v:.1e}", "rk4: fixed step $h_{RK}$",
                    figs / "error_time_rk4.png", T_ORBIT)
    error_time_plot(rows, "dopri5", "rtol", lambda v: f"rtol {v:.0e}",
                    "dopri5: tolerance rtol (atol = 0.01 rtol)", figs / "error_time_dopri5.png", T_ORBIT)
    error_vs_cost(rows, figs / "error_vs_cost.png")
    orbits(rows, figs / "orbits.png")
    print(f"figures in {figs}")


if __name__ == "__main__":
    main()
