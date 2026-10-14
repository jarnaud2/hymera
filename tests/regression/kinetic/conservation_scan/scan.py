#!/usr/bin/env python3
"""GCE conservation scan: rk4 vs dopri5 on p_phi and mu.

Runs the `single_particle` executable for every particle in particles.txt and
every integrator setting below, with collisions off and alpha0 = 0 (no
synchrotron radiation). B, J and psi come from the field fixture (an MHD
solution); E is the analytic toroidal field E_phi = E_0 R_a / R that
HijackEField writes. Each run writes (t, p, xi, R, phi, Z, p_phi, mu) every
dt_out; p_phi includes the toroidal-impulse term t c_aw0 E_phi R, so with a
static B both p_phi and mu are invariants of the exact guiding-centre motion.

    scan.py --exe build/bin/single_particle --out DIR [--orbits 100] [--quick]
            [--field grid|analytic|analytic_grid] [--E0 77]

--field grid (default) uses the MHD fixture; analytic the exact
AnalyticFieldEvaluator (no interpolation); analytic_grid the analytic field
sampled onto the grid and put through the production interpolation.

Writes DIR/runs/<integrator>_<setting>/<particle>/single_particle.dat and
DIR/summary.csv, then calls plot.py to make the figures in DIR/figures.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# HYMERA_REPO overrides the repository location when this script is run from a copy.
REPO = Path(os.environ.get("HYMERA_REPO", HERE.parents[3]))
DECK = REPO / "tests/regression/kinetic/decks/single_particle.input"
FIELDS = REPO / "tests/regression/kinetic/fixtures/fields.h5"

# Step sizes and tolerances (time in units of tau_c). rk4 is unstable above
# about 1e-5 for these orbits; dopri5's hRK is only the first trial step.
RK4_H = [8e-6, 4e-6, 2e-6, 1e-6, 5e-7, 2.5e-7]
DOPRI5_RTOL = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
ATOL_OVER_RTOL = 1e-2
QUICK_RK4_H = [5e-7, 2.5e-7]
QUICK_DOPRI5_RTOL = [1e-11, 1e-12]

# Poloidal transit time of the reference orbit (passing_hi, p = 15, xi = 0.9),
# measured with dopri5 rtol=1e-12 at 5e-6 tau_c sampling as the time at which
# the poloidal angle about the axis (3.10, 0.33) has turned by 2 pi: 1.05e-4 tau_c.
# (passing_lo: 1.30e-4; the trapped bounce period is ~3.2e-4.)
T_ORBIT = 1.05e-4


def read_particles(path):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, p, xi, R, Z = line.split()
            out.append(dict(name=name, p=float(p), xi=float(xi), R=float(R), Z=float(Z)))
    return out


def settings(quick):
    rk4 = QUICK_RK4_H if quick else RK4_H
    dp = QUICK_DOPRI5_RTOL if quick else DOPRI5_RTOL
    s = [("rk4", f"h{h:.1e}", {"Simulation/integrator": "rk4", "Simulation/hRK": h}) for h in rk4]
    s += [("dopri5", f"rtol{r:.0e}",
           {"Simulation/integrator": "dopri5", "Simulation/rtol": r,
            "Simulation/atol": r * ATOL_OVER_RTOL, "Simulation/hRK": 1e-7}) for r in dp]
    return s


def run_one(exe, launch, work, part, over, tlim, dt_out, threads, field, E0):
    # Parthenon step: dt_out rounded down to a whole multiple of the largest rk4
    # step, at least that step. Outputs then fall on step boundaries.
    # Below that (fine orbit sampling) the step is dt_out itself.
    h_max = max(RK4_H)
    dt_step = max(h_max, h_max * int(dt_out / h_max)) if dt_out >= h_max else dt_out
    run = work / "r1" / "r2"
    if work.exists():
        shutil.rmtree(work)
    run.mkdir(parents=True)
    (work / "inputs").symlink_to(REPO / "inputs")
    shutil.copy(FIELDS, run / "fields.h5")
    args = {
        "GuidingCenterEquations/alpha0": 0,
        "Simulation/EnableSmallAngleCollisions": 0,
        "Simulation/EnableLargeAngleCollisions": 0,
        "Simulation/verbosity": "quiet",
        # single_particle pushes one Parthenon step of dt_force per step, so
        # the integrator step is capped at dt_force: keep it a whole multiple of
        # the largest hRK so that every rk4 step size in RK4_H is the one used.
        "parthenon/time/dt_force": dt_step,
        "parthenon/time/tlim": tlim,
        # single_particle appends one row per Parthenon output, so the output
        # cadence is the sampling of the conservation series.
        "parthenon/output2/dt": dt_out,
        # Keep the hdf5 output (it fires the hook that writes the rows) but
        # without the swarm; the .phdf files are deleted after each run.
        "parthenon/output2/swarms": "none",
        "SingleParticle/p": part["p"], "SingleParticle/xi": part["xi"],
        "SingleParticle/R": part["R"],
        # particles.txt is relative to the MHD axis (Z = 0.33); the analytic axis is at Z = 0
        "SingleParticle/Z": part["Z"] - (0.33 if field != "grid" else 0.0),
        "SingleParticle/output_file": "single_particle.dat",
        "SingleParticle/field_source": field,
        "AnalyticField/E_0": E0,
    }
    args.update(over)
    cmd = launch + [str(exe), "-i", str(DECK)] + [f"{k}={v}" for k, v in args.items()]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    t0 = time.perf_counter()
    with open(run / "log.txt", "w") as log:
        rc = subprocess.run(cmd, cwd=run, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    wall = time.perf_counter() - t0
    for f in run.glob("*.phdf"):
        f.unlink()
    (run / "fields.h5").unlink(missing_ok=True)
    (run / "command.txt").write_text(" ".join(cmd) + "\n")
    logtxt = (run / "log.txt").read_text(errors="replace")
    failed = rc != 0 or "push verify failed" in logtxt
    return run / "single_particle.dat", wall, failed, rc


def metrics(dat):
    a = np.loadtxt(dat, ndmin=2)
    if a.shape[0] < 2:
        return None
    pp, mu = a[:, 6], a[:, 7]
    dpp = np.abs(pp - pp[0]) / abs(pp[0])
    dmu = np.abs(mu - mu[0]) / abs(mu[0])
    return dict(rows=a.shape[0], t_end=a[-1, 0],
                max_dpphi=float(dpp.max()), end_dpphi=float(dpp[-1]),
                max_dmu=float(dmu.max()), end_dmu=float(dmu[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--particles", type=Path, default=HERE / "particles.txt")
    ap.add_argument("--orbits", type=float, default=100.0,
                    help="run length in poloidal transits of the reference orbit (T_ORBIT)")
    ap.add_argument("--points-per-orbit", type=int, default=12,
                    help="output samples per transit (also the orbit-plot resolution); "
                         "dt_out = T_ORBIT / this must be >= the largest rk4 step (8e-6)")
    ap.add_argument("--launch", default="mpirun -n 1" if shutil.which("mpirun") else "",
                    help='launcher prefix, e.g. "srun -n 1" or ""')
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="2 rk4 and 2 dopri5 settings only")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--field", default="grid", choices=["grid", "analytic", "analytic_grid"])
    ap.add_argument("--E0", type=float, default=77.0, help="AnalyticField/E_0 (toroidal E)")
    args = ap.parse_args()

    exe = args.exe.resolve()
    tlim = args.orbits * T_ORBIT
    dt_out = T_ORBIT / args.points_per_orbit
    parts = read_particles(args.particles)
    launch = args.launch.split()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for integ, label, over in settings(args.quick):
        for part in parts:
            work = args.out / "runs" / f"{integ}_{label}" / part["name"]
            dat, wall, failed, rc = run_one(exe, launch, work, part, over, tlim, dt_out,
                                            args.threads, args.field, args.E0)
            m = metrics(dat) if dat.exists() else None
            row = dict(integrator=integ, setting=label, particle=part["name"],
                       hRK=over.get("Simulation/hRK"), rtol=over.get("Simulation/rtol", ""),
                       atol=over.get("Simulation/atol", ""), wall_s=f"{wall:.3f}",
                       failed=int(failed or m is None or m["t_end"] < tlim * (1 - 1e-9)), rc=rc)
            row.update(m or {})
            rows.append(row)
            status = "FAILED" if row["failed"] else (
                f"dp_phi {m['max_dpphi']:.2e}  dmu {m['max_dmu']:.2e}")
            print(f"{integ:6s} {label:12s} {part['name']:12s} {wall:7.2f}s  {status}", flush=True)
    # Finely sampled first transits for the orbit figure: dopri5 at the finest
    # tolerance, 2 transits at 200 samples per transit.
    fine = settings(args.quick)[-1]
    for part in parts:
        work = args.out / "orbits" / part["name"]
        run_one(exe, launch, work, part, fine[2], 2 * T_ORBIT, T_ORBIT / 200, args.threads,
                args.field, args.E0)

    keys = ["integrator", "setting", "particle", "hRK", "rtol", "atol", "wall_s", "failed", "rc",
            "rows", "t_end", "max_dpphi", "end_dpphi", "max_dmu", "end_dmu"]
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"wrote {args.out / 'summary.csv'}")
    if not args.no_plot:
        subprocess.run([sys.executable, str(HERE / "plot.py"), str(args.out)], check=True)


if __name__ == "__main__":
    main()
