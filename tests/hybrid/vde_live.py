#!/usr/bin/env python3
"""Live plots and movies for a running hybrid VDE simulation.

Usage: vde_live.py RUN_DIR --geometry inputs/AxisSymmetricGeometry.dat [--out DIR]
                   [--check-end] [--zwall-margin 0.03] [--ohmic-frac 0.05]

Reads, in RUN_DIR (the r1/r2 directory of the run):
  <id>.diag.h5           /timeseries/currents (copied first: the run holds it open)
  <id>.out2.NNNNNN.phdf  particles and the Output plot grid (psi, B, J, E)
  <id>.out8.NNNNN.rdat   the MHD state; its vertex DOFs are V_r, V_phi, V_z
  <id>.out8.NNNNN.rhdf   only for the time of the matching .rdat

Writes into --out (default RUN_DIR/live):
  currents.png   I_ohmic and I_re against t
  axis.png       magnetic axis Z_c and R_c against t (psi maximum on the plot grid),
                 with the wall height above it
  frames/<movie>_<cycle>.png, one per output, made only once
  <movie>.webm   for psi, profiles, particles, jre, B, E, V (ffmpeg, VP9: Perlmutter has no libx264)
                 profiles: J_phi (Ohmic, MHD), J_re,phi (particles), E_phi and E_par against R
                 on the plot-grid row nearest the magnetic axis (psi maximum)
  status.txt     the last time, axis, currents and the end test

--check-end exits with status 3 when the end of the VDE is reached: the axis is within
--zwall-margin (in a) of the first wall above it, and I_ohmic is below --ohmic-frac of
its first value.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ALIVE = 2
MOVIES = ("psi", "profiles", "particles", "jre", "B", "E", "V")
MU0 = 4e-7 * np.pi
B0 = 5.3  # Reference/B0 of the deck [T]; J on the plot grid is in units of B0 / (mu0 a)
# Grid of the MHD solver and the geometry file, in units of a (KineticParams.cpp).
RMIN, RMAX, ZMIN, ZMAX = 1.525, 4.975, -2.975, 2.975


def read_geometry(path):
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    g = np.array(rows, dtype=float)
    nR, nZ = g.shape
    R = RMIN + (np.arange(nR) + 0.5) * (RMAX - RMIN) / nR
    Z = ZMIN + (np.arange(nZ) + 0.5) * (ZMAX - ZMIN) / nZ
    return R, Z, g


def wall_top(geom, Rc, Zc):
    """Upper edge of the plasma region (tags 1 and 2) above the axis, at R = Rc."""
    R, Z, g = geom
    i = int(np.argmin(abs(R - Rc)))
    dZ = Z[1] - Z[0]
    j = int(np.argmin(abs(Z - Zc)))
    col = g[i] >= 0.5
    if not col[j]:
        return Zc  # the axis is outside the plasma region
    while j + 1 < len(Z) and col[j + 1]:
        j += 1
    return Z[j] + 0.5 * dZ


def axis_from_psi(o, geom):
    """Magnetic axis as the psi maximum (psi is shifted to 0 on the axis and is negative
    elsewhere) inside the plasma region, refined by a parabola through the neighbours.
    Deck/Rc and Deck/Zc are not used: RefreshFluxAndAxis finds the axis into local
    copies and never writes them back, so they keep their initial values."""
    gR, gZ, psi = o["gR"], o["gZ"], o["psi"]
    R, Z, g = geom
    iR = np.clip(np.searchsorted(R, gR), 0, len(R) - 1)
    iZ = np.clip(np.searchsorted(Z, gZ), 0, len(Z) - 1)
    inside = g[np.ix_(iR, iZ)] >= 0.5
    i, j = np.unravel_index(np.argmax(np.where(inside, psi, -np.inf)), psi.shape)

    def refine(x, f0, fm, fp, k):
        d = fm - 2 * f0 + fp
        if 0 < k < len(x) - 1 and d < 0:
            return x[k] + 0.5 * (fm - fp) / d * (x[1] - x[0])
        return x[k]
    Rc = refine(gR, psi[i, j], psi[max(i - 1, 0), j], psi[min(i + 1, len(gR) - 1), j], i)
    Zc = refine(gZ, psi[i, j], psi[i, max(j - 1, 0)], psi[i, min(j + 1, len(gZ) - 1)], j)
    return float(Rc), float(Zc)


def draw_walls(ax, geom):
    R, Z, g = geom
    ax.contour(R, Z, g.T, levels=[-0.5, 0.5, 1.5], colors=["0.15", "0.15", "0.55"],
               linewidths=[0.8, 1.2, 0.6], linestyles=["-", "-", ":"])


def read_jre_maps(run):
    """/profiles/jre_c<cycle> (Hybrid/jre_map = 1): the runaway current passed to MHD,
    eta J in units of B_0 V_A on the MHD cell grid. Returns {cycle: (R, Z, J_phi[R, Z])}."""
    diag = sorted(glob.glob(str(run / "*.diag.h5")))
    out = {}
    if not diag:
        return out
    with tempfile.TemporaryDirectory() as tmp:
        cp = Path(tmp) / "diag.h5"
        shutil.copy(diag[0], cp)
        try:
            with h5py.File(cp, "r") as f:
                for k in f.get("profiles", {}):
                    if not k.startswith("jre_c"):
                        continue
                    g = f["profiles"][k]
                    R, Z, J = g["R"][()], g["Z"][()], g["J_phi"][()]
                    nR = int(np.sum(Z == Z[0]))
                    out[int(k[5:])] = (R[:nR], Z[::nR], J.reshape(-1, nR).T)
        except OSError:
            pass
    return out


def deposited_jre(o, maps):
    """The deposited runaway current of this output in A/m^2, or None. E_ohm = En eta_norm
    J_phi with J_phi in J_0 = B_0 / (mu0 a), and E_re = En J_re,map, so J_re = J_re,map /
    eta_norm in J_0."""
    m = maps.get(o["cycle"])
    if m is None or not o.get("eta_norm"):
        return None
    R, Z, J = m
    return R, Z, J / o["eta_norm"] * B0 / (MU0 * o["a"])


def read_currents(run):
    diag = sorted(glob.glob(str(run / "*.diag.h5")))
    if not diag:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        cp = Path(tmp) / "diag.h5"
        shutil.copy(diag[0], cp)
        try:
            with h5py.File(cp, "r") as f:
                if "timeseries/currents" not in f:
                    return None
                g = f["timeseries/currents"]
                return {k: g[k][()] for k in g}
        except OSError:
            return None


def snapshot_files(run):
    """The numbered out2 files that open, by cycle; 'final' repeats a time and is skipped."""
    out = []
    for fn in sorted(glob.glob(str(run / "*.out2.[0-9]*.phdf"))):
        try:
            with h5py.File(fn, "r") as f:
                out.append((int(f["Info"].attrs["NCycle"]), float(f["Info"].attrs["Time"]), fn))
        except (OSError, KeyError):
            pass  # being written
    return out


def read_snapshot(fn):
    with h5py.File(fn, "r") as f:
        s = f["particles/SwarmVars"]
        a = (s["particle.status"][()] & ALIVE) > 0
        d = {k: s[f"particle.{k}"][()][a] for k in ("R", "Z", "p", "xi", "weight")}
        par = f["Params"].attrs
        F = np.asarray(par["Deck/Hermite_Field_Plot_data_F"])
        F = F.reshape(F.shape[-5:])[:, :, 0]          # (NR, NZ, 3, 7) at the output time
        return dict(t=float(f["Info"].attrs["Time"]), cycle=int(f["Info"].attrs["NCycle"]),
                    parts=d, n=int(a.sum()),
                    gR=np.asarray(par["Deck/Hermite_Field_Plot_data_R"]),
                    gZ=np.asarray(par["Deck/Hermite_Field_Plot_data_Z"]),
                    B=F[:, :, :, 0], J=F[:, :, :, 1], E=F[:, :, :, 4], psi=F[:, :, 0, 6],
                    Rc=float(par["Deck/Rc"]), Zc=float(par["Deck/Zc"]),
                    eta_norm=float(par["Deck/eta_norm"]) if "Deck/eta_norm" in par else None,
                    a=float(np.asarray(par["Deck/a"])) if "Deck/a" in par else 2.0)


def read_velocity(rdat, nr=100, nz=200, nphi=2):
    """V_r, V_phi, V_z at the vertices (phi index 0) from a stag_vec_io file."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regression" / "viz"))
    from mhd_data import read_petsc_vecs
    recs = read_petsc_vecs(rdat)
    shape = (nz + 1, nphi + 1, nr + 1)
    return [recs[c].reshape(shape)[:, 0, :].T for c in range(3)]  # (R, Z) each


def rdat_by_time(run):
    """Time of each out8 .rdat, from the .rhdf of the same number."""
    out = {}
    for fn in glob.glob(str(run / "*.out8.[0-9]*.rdat")):
        rh = fn[:-len(".rdat")] + ".rhdf"
        try:
            with h5py.File(rh, "r") as f:
                out[round(float(f["Info"].attrs["Time"]), 9)] = fn
        except (OSError, KeyError):
            pass
    return out


def jre_map(o, I_re, nR=100, nZ=200):
    d = o["parts"]
    v = d["p"] / np.sqrt(1 + d["p"] ** 2)
    q = d["weight"] * d["xi"] * v
    eR = np.linspace(RMIN, RMAX, nR + 1)
    eZ = np.linspace(ZMIN, ZMAX, nZ + 1)
    H, _, _ = np.histogram2d(d["R"], d["Z"], [eR, eZ], weights=q / d["R"])
    dA = np.diff(eR)[0] * np.diff(eZ)[0] * o["a"] ** 2
    J = H / dA
    tot = np.sum(J) * dA
    if tot != 0 and I_re is not None:
        J *= I_re / tot
    return 0.5 * (eR[1:] + eR[:-1]), 0.5 * (eZ[1:] + eZ[:-1]), J


def frame_axes(n, title):
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.6, 6.4), constrained_layout=True,
                             sharex=True, sharey=True, squeeze=False)
    fig.suptitle(title, fontsize=10)
    return fig, axes[0]


def finish_axes(ax, o, geom):
    draw_walls(ax, geom)
    ax.contour(o["gR"], o["gZ"], o["psi"].T, levels=14, colors="0.4", linewidths=0.5)
    ax.plot(o["Rc"], o["Zc"], "+", color="black", ms=10, mew=1.5)
    ax.set_aspect("equal")
    ax.set_xlim(RMIN, RMAX)
    ax.set_ylim(ZMIN, ZMAX)
    ax.set_xlabel("R [a]")


def component_frame(o, geom, fields, names, title, path, R=None, Z=None):
    fig, axes = frame_axes(3, title)
    R = o["gR"] if R is None else R
    Z = o["gZ"] if Z is None else Z
    for k, ax in enumerate(axes):
        f = fields[k]
        lim = np.nanpercentile(abs(f), 99.5) or 1.0
        pc = ax.pcolormesh(R, Z, f.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
        fig.colorbar(pc, ax=ax, shrink=0.55)
        ax.set_title(names[k], fontsize=10)
        finish_axes(ax, o, geom)
    axes[0].set_ylabel("Z [a]")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_frames(o, I_re, V, geom, fdir):
    tag = f"{o['cycle']:06d}"
    head = f"t = {o['t'] * 1e3:.2f} ms, axis (R, Z) = ({o['Rc']:.3f}, {o['Zc']:.3f}) a"
    # poloidal flux: psi = 0 on the axis and negative elsewhere (RefreshFluxAndAxis)
    p = fdir / f"psi_{tag}.png"
    if not p.exists():
        fig, axes = frame_axes(1, f"{head}\npoloidal flux $\\psi$, 0 on the axis")
        ax = axes[0]
        # One colour scale for the whole movie: the lower limit from the first frame
        # (psi_low.txt), so that the frames do not flicker.
        low = fdir / "psi_low.txt"
        if not low.exists():
            low.write_text(repr(float(np.percentile(o["psi"], 1))))
        lev = np.linspace(float(low.read_text()), 0.0, 25)
        cf = ax.contourf(o["gR"], o["gZ"], o["psi"].T, levels=lev, cmap="viridis", extend="both")
        ax.contour(o["gR"], o["gZ"], o["psi"].T, levels=lev[::2], colors="white", linewidths=0.4)
        fig.colorbar(cf, ax=ax, shrink=0.55, label=r"$\psi$")
        draw_walls(ax, geom)
        ax.plot(o["Rc"], o["Zc"], "+", color="red", ms=12, mew=2)
        ax.set_aspect("equal")
        ax.set_xlim(RMIN, RMAX)
        ax.set_ylim(ZMIN, ZMAX)
        ax.set_xlabel("R [a]")
        ax.set_ylabel("Z [a]")
        fig.savefig(p, dpi=110)
        plt.close(fig)
    # particles on psi
    p = fdir / f"profiles_{tag}.png"
    if not p.exists():
        profile_frame(o, I_re, geom, p, fdir)
    p = fdir / f"particles_{tag}.png"
    if not p.exists():
        fig, axes = frame_axes(1, f"{head}\nalive particles: {o['n']} (at most 30000 shown)")
        d = o["parts"]
        m = np.random.default_rng(0).permutation(len(d["R"]))[:30000]
        sc = axes[0].scatter(d["R"][m], d["Z"][m], s=0.8, c=np.log10(d["p"][m]), cmap="viridis",
                             vmin=0, vmax=2.6, linewidths=0)
        fig.colorbar(sc, ax=axes[0], shrink=0.55, label=r"$\log_{10} p$")
        finish_axes(axes[0], o, geom)
        axes[0].set_ylabel("Z [a]")
        fig.savefig(p, dpi=110)
        plt.close(fig)
    p = fdir / f"jre_{tag}.png"
    if not p.exists():
        src = "passed to MHD (step average)" if deposited_jre(o, o.get("maps", {})) is not None \
            else "particles (snapshot)"
        fig, axes = frame_axes(1, f"{head}\n$J_{{re}}$ {src}"
                               + (f", I_re = {I_re / 1e3:.1f} kA" if I_re is not None else ""))
        dep = deposited_jre(o, o.get("maps", {}))
        R, Z, J = dep if dep is not None else jre_map(o, I_re)
        vmax = np.percentile(J[J > 0], 99.5) / 1e3 if np.any(J > 0) else 1.0
        pc = axes[0].pcolormesh(R, Z, J.T / 1e3, cmap="magma_r", vmin=0, vmax=vmax, shading="nearest")
        fig.colorbar(pc, ax=axes[0], shrink=0.55, label=r"$J_{re}$ [kA/m$^2$]")
        finish_axes(axes[0], o, geom)
        axes[0].set_ylabel("Z [a]")
        fig.savefig(p, dpi=110)
        plt.close(fig)
    p = fdir / f"B_{tag}.png"
    if not p.exists():
        component_frame(o, geom, [o["B"][:, :, k] for k in range(3)], [r"$B_R$", r"$B_\phi$", r"$B_Z$"],
                        f"{head}\nmagnetic field (normalised, Output plot grid)", p)
    p = fdir / f"E_{tag}.png"
    if not p.exists():
        component_frame(o, geom, [o["E"][:, :, k] for k in range(3)], [r"$E_R$", r"$E_\phi$", r"$E_Z$"],
                        f"{head}\nelectric field (normalised, Output plot grid)", p)
    p = fdir / f"V_{tag}.png"
    if V is not None and not p.exists():
        Rv = np.linspace(RMIN, RMAX, V[0].shape[0])
        Zv = np.linspace(ZMIN, ZMAX, V[0].shape[1])
        component_frame(o, geom, V, [r"$V_R$", r"$V_\phi$", r"$V_Z$"],
                        f"{head}\nplasma velocity (MHD vertices, code units)", p, Rv, Zv)


def profile_frame(o, I_re, geom, path, fdir):
    """J_phi (Ohmic, MHD), J_re,phi (particles), E_phi and E_par against R at the
    plot-grid row nearest the magnetic axis. The axis heights follow the axis (psi
    maximum); Deck/Zc, used by the code's /profiles/midplane, stays at its initial value."""
    gR, gZ = o["gR"], o["gZ"]
    j = int(np.argmin(abs(gZ - o["Zc"])))
    Zrow = gZ[j]
    J0 = B0 / (MU0 * o["a"])                                   # A/m^2
    Jphi = o["J"][:, j, 1] * J0 / 1e6                          # MA/m^2
    E = o["E"][:, j, :]
    Bv = o["B"][:, j, :]
    Bm = np.sqrt(np.sum(Bv ** 2, axis=1))
    Epar = np.sum(E * Bv, axis=1) / np.where(Bm > 0, Bm, 1.0)
    # J_re,phi from the particles on the plot-grid row, in the cells around Zrow
    dep = deposited_jre(o, o.get("maps", {}))
    Rr, Zr, Jmap = dep if dep is not None else jre_map(o, I_re)
    jr = int(np.argmin(abs(Zr - Zrow)))
    Jre = np.interp(gR, Rr, Jmap[:, jr]) / 1e6
    # plasma region along the row, for the shading
    R, Z, g = geom
    jg = int(np.argmin(abs(Z - Zrow)))
    inside = np.interp(gR, R, (g[:, jg] >= 0.5).astype(float)) > 0.5
    # fixed axis limits for the movie, from the first frame and widened when exceeded
    lim = fdir / "profiles_lim.txt"
    cur = [float(np.nanmax(abs(Jphi))), float(np.nanmax(abs(Jre))) if np.any(Jre) else 0.0,
           float(np.nanmax(abs(E[:, 1]))), float(np.nanmax(abs(Epar)))]
    old = [float(x) for x in lim.read_text().split()] if lim.exists() else [0.0] * 4
    new = [max(a, b) for a, b in zip(old, cur)]
    lim.write_text(" ".join(repr(x) for x in new))
    jmax = 1.1 * max(new[0], new[1])
    emax = 1.1 * max(new[2], new[3])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True, constrained_layout=True)
    fig.suptitle(f"t = {o['t'] * 1e3:.2f} ms, profiles at Z = {Zrow:.3f} a "
                 f"(magnetic axis Z = {o['Zc']:.3f} a, R = {o['Rc']:.3f} a)", fontsize=10)
    for ax in (a1, a2):
        ax.fill_between(gR, -1e9, 1e9, where=~inside, color="0.92", step="mid", lw=0)
        ax.axvline(o["Rc"], color="0.4", lw=0.8, ls=":")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.grid(True, alpha=0.3)
    a1.plot(gR, Jphi, "-", color="#eb6834", lw=1.6, label=r"$J_\phi$ (Ohmic, MHD)")
    a1.plot(gR, Jre, "-", color="#1baf7a", lw=1.6,
            label=r"$J_{re,\phi}$ (" + ("passed to MHD" if dep is not None else "particles")
            + (f", I_re = {I_re / 1e6:.3f} MA)" if I_re is not None else ")"))
    a1.set_ylim(-0.3 * jmax, jmax)
    a1.set_ylabel(r"current density [MA/m$^2$]")
    a1.legend(fontsize=8, loc="upper right")
    a2.plot(gR, E[:, 1], "-", color="#2a78d6", lw=1.6, label=r"$E_\phi$")
    a2.plot(gR, Epar, "--", color="#7a3fb0", lw=1.4, label=r"$E_\parallel$")
    a2.set_ylim(-emax, emax)
    a2.set_ylabel("electric field (normalised, plot grid)")
    a2.set_xlabel(r"$R$ [$a$] (dotted: magnetic axis; grey: outside the plasma region)")
    a2.legend(fontsize=8, loc="upper right")
    a2.set_xlim(RMIN, RMAX)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_movies(fdir, out, suffix=""):
    if shutil.which("ffmpeg") is None:
        return
    for m in MOVIES:
        if not glob.glob(str(fdir / f"{m}_*.png")):
            continue
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4", "-pattern_type", "glob",
                        "-i", str(fdir / f"{m}_*.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-pix_fmt", "yuv420p", "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32",
                        str(out / f"{m}{suffix}.webm")],
                       check=False)


def plot_series(cur, track, geom, out):
    if cur is not None and len(cur.get("t", [])):
        t = cur["t"] * 1e3
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        ax.plot(t, cur["I_ohmic"] / 1e6, "s-", ms=3, color="#eb6834", label=r"$I_{\rm ohmic}$")
        ax.plot(t, cur["I_re_particles"] / 1e6, "o-", ms=3, color="#1baf7a", label=r"$I_{re}$ (particles)")
        ax.plot(t, (cur["I_ohmic"] + cur["I_re_particles"]) / 1e6, "-", color="0.4", lw=1,
                label="total")
        ax.set_xlabel("t [ms]")
        ax.set_ylabel("current [MA]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(f"VDE run: currents, last t = {t[-1]:.2f} ms", loc="left", fontsize=9)
        fig.savefig(out / "currents.png", dpi=150)
        plt.close(fig)
    if track:
        t = np.array([x[0] for x in track]) * 1e3
        Rc = np.array([x[1] for x in track])
        Zc = np.array([x[2] for x in track])
        Zw = np.array([wall_top(geom, r, z) for _, r, z in track])
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        ax.plot(t, Zc, "o-", ms=3, color="#2a78d6", label=r"$Z_c$")
        ax.plot(t, Zw, "--", color="0.2", label="first wall above the axis")
        ax.plot(t, Rc, "s-", ms=3, color="#eb6834", label=r"$R_c$")
        ax.set_xlabel("t [ms]")
        ax.set_ylabel("[a]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Magnetic axis", loc="left", fontsize=9)
        fig.savefig(out / "axis.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--geometry", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check-end", action="store_true")
    ap.add_argument("--zwall-margin", type=float, default=0.03)
    ap.add_argument("--ohmic-frac", type=float, default=0.05)
    ap.add_argument("--part", default="", help="suffix of the movie files, for a sequel")
    args = ap.parse_args()
    out = args.out or args.run / "live"
    fdir = out / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    geom = read_geometry(args.geometry)

    cur = read_currents(args.run)
    maps = read_jre_maps(args.run)
    rdat = rdat_by_time(args.run)
    track_file = out / "axis_track.txt"
    track = {}
    if track_file.exists():
        for ln in track_file.read_text().splitlines():
            c, t, r, z = ln.split()
            track[int(c)] = (float(t), float(r), float(z))
    for cycle, t, fn in snapshot_files(args.run):
        have = all((fdir / f"{m}_{cycle:06d}.png").exists() for m in MOVIES) and cycle in track
        if have:
            continue
        o = read_snapshot(fn)
        o["Rc"], o["Zc"] = axis_from_psi(o, geom)
        o["maps"] = maps
        track[cycle] = (o["t"], o["Rc"], o["Zc"])
        I_re = None
        if cur is not None and len(cur.get("t", [])):
            j = int(np.argmin(abs(cur["t"] - o["t"])))
            I_re = float(cur["I_re_particles"][j])
        V = None
        r = rdat.get(round(o["t"], 9))
        if r is not None:
            try:
                V = read_velocity(r)
            except (ValueError, IndexError):
                V = None
        make_frames(o, I_re, V, geom, fdir)
    track_file.write_text("".join(f"{c} {t!r} {r!r} {z!r}\n" for c, (t, r, z) in sorted(track.items())))
    plot_series(cur, [track[c] for c in sorted(track)], geom, out)
    make_movies(fdir, out, args.part)

    end = False
    lines = []
    if track:
        t, Rc, Zc = track[max(track)]
        Zw = wall_top(geom, Rc, Zc)
        hit = Zc >= Zw - args.zwall_margin
        lines.append(f"t = {t * 1e3:.3f} ms, axis (R, Z) = ({Rc:.4f}, {Zc:.4f}) a, wall above at Z = {Zw:.4f} a,"
                     f" axis at wall: {hit}")
        if cur is not None and len(cur.get("t", [])):
            io = cur["I_ohmic"]
            dep = io[-1] < args.ohmic_frac * io[0]
            lines.append(f"I_ohmic {io[-1] / 1e6:.4f} MA ({io[-1] / io[0]:.4f} of the first),"
                         f" I_re {cur['I_re_particles'][-1] / 1e3:.2f} kA, Ohmic depleted: {dep}")
            end = hit and dep
    lines.append(f"END: {end}")
    (out / "status.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.check_end and end:
        sys.exit(3)


if __name__ == "__main__":
    main()
