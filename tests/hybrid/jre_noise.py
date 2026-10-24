#!/usr/bin/env python3
"""Noise of the runaway current passed to MHD (Hybrid/jre_map = 1).

Usage: jre_noise.py RUN_DIR [RUN_DIR ...] [--cycles c1,c2,...] [--label L ...]
                    [--out DIR] [--geometry inputs/AxisSymmetricGeometry.dat]

For each map /profiles/jre_c<cycle> in RUN_DIR/<id>.diag.h5, with psi and the alive
particles from RUN_DIR/<id>.out2.<cycle>.phdf, it measures in the core (the plasma
region where J_re > 10% of its maximum):

  fs_noise    rms(J - <J>_psi) / mean(J): the deviation from the flux-surface
              average, with <J>_psi the mean over 40 bins of psi. A smooth current
              that is a function of psi gives 0.
  cell_noise  rms(J - S J) / mean(J), with S the 3 x 3 mean of the neighbours: the
              cell-to-cell scatter, independent of psi.

Writes jre_noise.txt and jre_noise.png (noise against the alive particle count and
the maps with their profile through the axis) into --out (default: the first run).
"""
import argparse
import glob
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402

RMIN, RMAX, ZMIN, ZMAX = 1.525, 4.975, -2.975, 2.975


def maps_of(run):
    diag = sorted(glob.glob(str(Path(run) / "*.diag.h5")))
    out = {}
    if not diag:
        return out
    with tempfile.TemporaryDirectory() as tmp:
        cp = Path(tmp) / "d.h5"
        shutil.copy(diag[0], cp)
        with h5py.File(cp, "r") as f:
            for k in f.get("profiles", {}):
                m = re.fullmatch(r"jre_c(\d+)", k)
                if not m:
                    continue
                g = f["profiles"][k]
                R, Z, J = g["R"][()], g["Z"][()], g["J_phi"][()]
                nR = int(np.sum(Z == Z[0]))
                out[int(m.group(1))] = (R[:nR], Z[::nR], J.reshape(-1, nR).T)
    return out


def snapshot(run, cycle):
    fn = glob.glob(str(Path(run) / f"*.out2.{cycle:06d}.phdf"))
    if not fn:  # the last output of a run may only be the final one
        fn = [f for f in glob.glob(str(Path(run) / "*.out2.final.phdf"))
              if int(h5py.File(f, "r")["Info"].attrs["NCycle"]) == cycle]
    if not fn:
        return None
    with h5py.File(fn[0], "r") as f:
        par = f["Params"].attrs
        F = np.asarray(par["Deck/Hermite_Field_Plot_data_F"])
        F = F.reshape(F.shape[-5:])[:, :, 0]
        s = f["particles/SwarmVars"]
        n = int(((s["particle.status"][()] & 2) > 0).sum())
        return dict(gR=np.asarray(par["Deck/Hermite_Field_Plot_data_R"]),
                    gZ=np.asarray(par["Deck/Hermite_Field_Plot_data_Z"]),
                    psi=F[:, :, 0, 6], n=n, t=float(f["Info"].attrs["Time"]),
                    nCD=int(par["Deck/nCD"]) if "Deck/nCD" in par else None)


def bilinear(gR, gZ, F, R, Z):
    """F(R, Z) on the grid (gR, gZ) at the points R x Z (outer product)."""
    i = np.clip(np.searchsorted(gR, R) - 1, 0, len(gR) - 2)
    j = np.clip(np.searchsorted(gZ, Z) - 1, 0, len(gZ) - 2)
    tR = np.clip((R - gR[i]) / (gR[i + 1] - gR[i]), 0, 1)[:, None]
    tZ = np.clip((Z - gZ[j]) / (gZ[j + 1] - gZ[j]), 0, 1)[None, :]
    I, J = i[:, None], j[None, :]
    return ((1 - tR) * (1 - tZ) * F[I, J] + tR * (1 - tZ) * F[I + 1, J]
            + (1 - tR) * tZ * F[I, J + 1] + tR * tZ * F[I + 1, J + 1])


def noise(R, Z, J, snap, nbins=40):
    psi = bilinear(snap["gR"], snap["gZ"], snap["psi"], R, Z)
    core = J > 0.1 * J.max()
    mJ = J[core].mean()
    edges = np.linspace(psi[core].min(), psi[core].max(), nbins + 1)
    k = np.clip(np.digitize(psi, edges) - 1, 0, nbins - 1)
    fs = np.array([J[core & (k == b)].mean() if np.any(core & (k == b)) else 0.0 for b in range(nbins)])
    fs_res = J - fs[k]
    P = np.pad(J, 1, mode="edge")
    S = sum(P[1 + a:1 + a + J.shape[0], 1 + b:1 + b + J.shape[1]] for a in (-1, 0, 1) for b in (-1, 0, 1)) / 9.0
    return (float(np.sqrt(np.mean(fs_res[core] ** 2)) / mJ),
            float(np.sqrt(np.mean((J - S)[core] ** 2)) / mJ), int(core.sum()), fs, edges, psi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--cycles", default="")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--geometry", type=Path)
    args = ap.parse_args()
    out = args.out or args.runs[0]
    out.mkdir(parents=True, exist_ok=True)
    want = {int(c) for c in args.cycles.split(",") if c}
    rows = []
    for k, run in enumerate(args.runs):
        label = args.label[k] if k < len(args.label) else run.name
        for c, (R, Z, J) in sorted(maps_of(run).items()):
            if (want and c not in want) or not np.any(J > 0):
                continue
            snap = snapshot(run, c)
            if snap is None:
                continue
            fsn, cn, ncore, *_ = noise(R, Z, J, snap)
            rows.append(dict(label=label, run=run, cycle=c, t=snap["t"], n=snap["n"], nCD=snap["nCD"],
                             fs=fsn, cell=cn, ncore=ncore, R=R, Z=Z, J=J, snap=snap))
    with open(out / "jre_noise.txt", "w") as fh:
        fh.write("# label cycle t_ms alive nCD core_cells fs_noise cell_noise\n")
        for r in rows:
            fh.write(f"{r['label']} {r['cycle']} {r['t'] * 1e3:.3f} {r['n']} {r['nCD']} {r['ncore']} "
                     f"{r['fs']:.4f} {r['cell']:.4f}\n")
    print((out / "jre_noise.txt").read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(dict.fromkeys(r["label"] for r in rows))
    show = [max((r for r in rows if r["label"] == l), key=lambda r: r["cycle"]) for l in labels][:6]
    fig = plt.figure(figsize=(3.0 * len(show) + 4.5, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, len(show) + 1, height_ratios=[1.6, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    for l in labels:
        rr = [r for r in rows if r["label"] == l]
        ax0.loglog([r["n"] for r in rr], [r["fs"] for r in rr], "o-", ms=4, label=f"{l}: flux surface")
        ax0.loglog([r["n"] for r in rr], [r["cell"] for r in rr], "s--", ms=3, label=f"{l}: cell")
    nn = np.array(sorted({r["n"] for r in rows}))
    if len(nn) > 1:
        ref = [r for r in rows if r["n"] == nn[0]][0]["fs"]
        ax0.loglog(nn, ref * np.sqrt(nn[0] / nn), ":", color="0.4", label=r"$\propto N^{-1/2}$")
    ax0.set_xlabel("alive particles")
    ax0.set_ylabel("relative noise of J_re (core)")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(fontsize=7)
    for k, r in enumerate(show):
        ax = fig.add_subplot(gs[0, k + 1])
        pc = ax.pcolormesh(r["R"], r["Z"], r["J"].T, cmap="magma_r", shading="nearest",
                           vmin=0, vmax=np.percentile(r["J"], 99.8))
        ax.set_aspect("equal")
        ax.set_xlim(2.0, 4.3)
        ax.set_ylim(-1.8, 2.3)
        ax.set_title(f"{r['label']}, cycle {r['cycle']}\nN = {r['n']}, fs {r['fs']:.3f}, cell {r['cell']:.3f}",
                     fontsize=8)
        fig.colorbar(pc, ax=ax, shrink=0.6)
        axp = fig.add_subplot(gs[1, k + 1])
        jz = int(np.argmax(r["J"].max(axis=0)))
        axp.plot(r["R"], r["J"][:, jz] / r["J"].max(), "-", lw=1)
        axp.set_xlim(2.0, 4.3)
        axp.set_title(f"row Z = {r['Z'][jz]:.2f} a", fontsize=8)
        axp.set_xlabel("R [a]")
        axp.grid(True, alpha=0.3)
    fig.suptitle("Runaway current passed to MHD (eta J, normalised): noise and maps", fontsize=10)
    fig.savefig(out / "jre_noise.png", dpi=110)
    print(f"wrote {out / 'jre_noise.png'}")


if __name__ == "__main__":
    main()
