#!/usr/bin/env python3
"""Radial drift of the runaway particles over one MHD step, for runs that differ in hRK.

Usage: drift_hrk.py SWEEP_DIR [--start 16] [--end 17] [--id vde_pilot3] [--out DIR]

SWEEP_DIR holds h<hRK>/r2 run directories, each restarted from the same dump
(cycle --start) with collisions off and run to cycle --end. Particles are not
tracked individually (swarm.id is always 0), so the population is compared:
with collisions off no particle is created or removed (CleanupParticles runs only
with large-angle collisions), so the alive set changes only by losses.

For each run: alive count, mean minor radius r = |(R, Z) - axis| and mean psi
(normalised to 0 on the axis) at the start and the end, and the change. The axis
is the psi maximum on the Output plot grid. Writes drift_hrk.txt and
drift_hrk.png (histograms of r at the end, and the mean drift against hRK).
"""
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402

ALIVE = 2


def bilinear(gR, gZ, F, R, Z):
    i = np.clip(np.searchsorted(gR, R) - 1, 0, len(gR) - 2)
    j = np.clip(np.searchsorted(gZ, Z) - 1, 0, len(gZ) - 2)
    tR = np.clip((R - gR[i]) / (gR[i + 1] - gR[i]), 0, 1)
    tZ = np.clip((Z - gZ[j]) / (gZ[j + 1] - gZ[j]), 0, 1)
    return ((1 - tR) * (1 - tZ) * F[i, j] + tR * (1 - tZ) * F[i + 1, j]
            + (1 - tR) * tZ * F[i, j + 1] + tR * tZ * F[i + 1, j + 1])


def population(fn):
    with h5py.File(fn, "r") as f:
        s = f["particles/SwarmVars"]
        al = (s["particle.status"][()] & ALIVE) > 0
        R, Z, p, w = (s[f"particle.{k}"][()][al] for k in ("R", "Z", "p", "weight"))
        par = f["Params"].attrs
        F = np.asarray(par["Deck/Hermite_Field_Plot_data_F"])
        F = F.reshape(F.shape[-5:])[:, :, 0]
        gR = np.asarray(par["Deck/Hermite_Field_Plot_data_R"])
        gZ = np.asarray(par["Deck/Hermite_Field_Plot_data_Z"])
        psi = F[:, :, 0, 6]
        hrk = float(par["Deck/hRK"]) if "Deck/hRK" in par else float("nan")
        t = float(f["Info"].attrs["Time"])
    box = (gR[:, None] > 2.2) & (gR[:, None] < 4.0) & (np.abs(gZ[None, :]) < 2.0)
    i, j = np.unravel_index(np.argmax(np.where(box, psi, -np.inf)), psi.shape)
    Rc, Zc = gR[i], gZ[j]
    return dict(R=R, Z=Z, p=p, w=w, r=np.hypot(R - Rc, Z - Zc), psi=bilinear(gR, gZ, psi, R, Z),
                n=int(al.sum()), Rc=Rc, Zc=Zc, t=t, hrk=hrk)


def snapshot(run, pid, c):
    fn = glob.glob(str(run / f"{pid}.out2.{c:06d}.phdf"))
    if not fn:
        fn = [f for f in glob.glob(str(run / f"{pid}.out2.final.phdf"))
              if int(h5py.File(f, "r")["Info"].attrs["NCycle"]) == c]
    return population(fn[0]) if fn else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--start", type=int, default=16)
    ap.add_argument("--end", type=int, default=17)
    ap.add_argument("--id", default="vde_pilot3")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.sweep / "drift"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(args.sweep.glob("h*/r2")):
        m = re.fullmatch(r"h(.+)", d.parent.name)
        a, b = snapshot(d, args.id, args.start), snapshot(d, args.id, args.end)
        if a is None or b is None:
            continue
        rows.append(dict(hrk=float(m.group(1)), a=a, b=b))
    rows.sort(key=lambda r: r["hrk"])
    lines = ["# hRK alive_start alive_end mean_r_start mean_r_end d_mean_r mean_psi_start mean_psi_end "
             "d_mean_psi wmean_r_end axis_end"]
    for r in rows:
        a, b = r["a"], r["b"]
        wr = np.sum(b["w"] * b["r"]) / np.sum(b["w"])
        lines.append(f"{r['hrk']:.1e} {a['n']} {b['n']} {a['r'].mean():.5f} {b['r'].mean():.5f} "
                     f"{b['r'].mean() - a['r'].mean():+.5f} {a['psi'].mean():.5e} {b['psi'].mean():.5e} "
                     f"{b['psi'].mean() - a['psi'].mean():+.4e} {wr:.5f} ({b['Rc']:.3f},{b['Zc']:.3f})")
    (out / "drift_hrk.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    bins = np.linspace(0, 1.8, 73)
    if rows:
        a1.hist(rows[0]["a"]["r"], bins=bins, histtype="step", color="k", lw=1.5,
                label=f"start (cycle {args.start})")
    for r in rows:
        a1.hist(r["b"]["r"], bins=bins, histtype="step", lw=1.0, label=f"end, hRK = {r['hrk']:.0e}")
    a1.set_xlabel("minor radius r from the axis [a]")
    a1.set_ylabel("alive particles")
    a1.legend(fontsize=8)
    a1.set_title("distribution of r, one MHD step, collisions off", fontsize=9)
    h = np.array([r["hrk"] for r in rows])
    dr = np.array([r["b"]["r"].mean() - r["a"]["r"].mean() for r in rows])
    a2.semilogx(h, dr, "o-")
    a2.axhline(0, color="0.6", lw=0.6)
    a2.set_xlabel("hRK [tau_c]")
    a2.set_ylabel("change of mean r over the step [a]")
    a2.grid(True, which="both", alpha=0.3)
    a2.set_title("inward drift (negative) against the rk4 step", fontsize=9)
    fig.savefig(out / "drift_hrk.png", dpi=110)
    print(f"wrote {out / 'drift_hrk.png'}")


if __name__ == "__main__":
    main()
