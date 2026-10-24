#!/usr/bin/env python3
"""Compare an uninterrupted hybrid run (A) with a run restarted from one of its dumps (B).

Usage: compare_restart.py DIR_A DIR_B PROBLEM_ID RESTART_CYCLE [--exact]

B is a copy of A's directory in which `hybrid -r <id>.out8.<RESTART_CYCLE>.rhdf` was
run to the same nlim, so B's files after the restart cycle are overwritten by the
restarted run and A's are the uninterrupted ones.

Checks:
  1. At the restart: B's first output after the restart reads the same state, i.e.
     the restart dump B loaded is byte-identical in both directories (B never rewrites
     it before reading).
  2. After the restart, for every later out2 / out8 output:
     - MHD state (.rdat): max |A - B| over all records;
     - particle count, total weight, and per-particle arrays when the counts agree;
     - time and cycle.
  3. diag.h5 currents: B must keep A's rows up to the restart time, drop the rest and
     append its own; t strictly increasing; rows before the restart identical.
With collisions on, the random-number pool is re-seeded at restart (not restored), so
step 2 can only agree statistically; with collisions off it must be bit-identical.
--exact (for runs with collisions off) also requires the MHD state, particles and
currents after the restart to be bit-identical.
Exit status 0 when every check that applies passes.
"""
import glob
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regression" / "viz"))
from mhd_data import read_petsc_vecs  # noqa: E402

ALIVE = 2


def info(fn):
    with h5py.File(fn, "r") as f:
        return float(f["Info"].attrs["Time"]), int(f["Info"].attrs["NCycle"])


def particles(fn):
    with h5py.File(fn, "r") as f:
        s = f["particles/SwarmVars"]
        a = (s["particle.status"][()] & ALIVE) > 0
        d = {k: s[f"particle.{k}"][()][a] for k in ("R", "Z", "p", "xi", "weight")}
    order = np.lexsort((d["xi"], d["p"], d["Z"], d["R"]))
    return {k: v[order] for k, v in d.items()}


def currents(fn):
    with h5py.File(fn, "r") as f:
        g = f["timeseries/currents"]
        return {k: g[k][()] for k in g}


def main():
    A, B, pid, rc = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    exact = "--exact" in sys.argv[5:]
    ok = True
    log = (B / "logB.txt").read_text() if (B / "logB.txt").exists() else ""
    tr = info(A / f"{pid}.out8.{rc:05d}.rhdf")[0]
    print(f"restart from cycle {rc}, t = {tr * 1e3:.4f} ms")

    # 1. the dump B loaded
    for ext in (".rhdf", ".rdat"):
        a = (A / f"{pid}.out8.{rc:05d}{ext}").read_bytes()
        b = (B / f"{pid}.out8.{rc:05d}{ext}").read_bytes()
        same = a == b
        ok &= same
        print(f"  restart dump {ext}: byte-identical in A and B: {same}")
    if "Restart: reading MHD state from" in log:
        print("  B log: " + [ln for ln in log.splitlines() if "Restart: reading MHD state" in ln][0].strip())
    else:
        print("  B log: no 'Restart: reading MHD state' line: FAIL")
        ok = False

    # 2. later outputs
    print("  after the restart (A uninterrupted vs B restarted):")
    for fa in sorted(glob.glob(str(A / f"{pid}.out8.[0-9]*.rdat"))):
        n = int(Path(fa).name.split(".")[2])
        if n <= rc:
            continue
        fb = B / Path(fa).name
        va, vb = read_petsc_vecs(fa), read_petsc_vecs(fb)
        d = max(float(np.max(np.abs(x - y))) for x, y in zip(va, vb))
        sc = max(float(np.max(np.abs(x))) for x in va)
        print(f"    MHD state out8.{n:05d}.rdat: max|A-B| = {d:.3e} (max|A| = {sc:.3e}), "
              f"identical: {d == 0.0}")
        if exact:
            ok &= d == 0.0
    for fa in sorted(glob.glob(str(A / f"{pid}.out2.[0-9]*.phdf"))):
        ta, ca = info(fa)
        if ca <= rc:
            continue
        fb = B / Path(fa).name
        tb, cb = info(fb)
        pa, pb = particles(fa), particles(fb)
        line = (f"    out2 cycle {ca}: t A {ta:.6e} B {tb:.6e}, cycle B {cb}; alive A {len(pa['p'])} "
                f"B {len(pb['p'])}; weight A {pa['weight'].sum():.6e} B {pb['weight'].sum():.6e}")
        if len(pa["p"]) == len(pb["p"]):
            dm = max(float(np.max(np.abs(pa[k] - pb[k]))) if len(pa[k]) else 0.0 for k in pa)
            line += f"; max|particle A-B| = {dm:.3e}"
            if exact:
                ok &= dm == 0.0
        elif exact:
            ok = False
        print(line)
        ok &= (ta == tb and ca == cb)

    # 3. diagnostics
    ca, cb = currents(A / f"{pid}.diag.h5"), currents(B / f"{pid}.diag.h5")
    keep = ca["t"] <= tr
    pre = all(np.array_equal(ca[k][keep], cb[k][: keep.sum()]) for k in ca)
    mono = bool(np.all(np.diff(cb["t"]) > 0))
    same_t = np.array_equal(ca["t"], cb["t"])
    print(f"  diag currents: rows A {len(ca['t'])} B {len(cb['t'])}; rows up to the restart identical: {pre}; "
          f"t strictly increasing in B: {mono}; same times: {same_t}")
    for k in ("I_ohmic", "I_re_particles"):
        r = np.max(np.abs(ca[k] - cb[k]) / np.maximum(np.abs(ca[k]), 1e-300)) if same_t else np.nan
        print(f"    {k}: A {ca[k][-1]:.6e} B {cb[k][-1]:.6e} at the end; max relative |A-B| {r:.3e}")
        if exact:
            ok &= bool(r == 0.0)
    ok &= pre and mono and same_t
    print(("PASS" if ok else "FAIL") + (" (--exact: bit-identical after the restart)" if exact else " (restart mechanics)"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
