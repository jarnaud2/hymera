#!/usr/bin/env python3
"""Pass/fail check of guiding-centre invariants (ctest kin_gce_conservation).

    check.py --exe build/bin/single_particle

Runs scan.py --quick (2 rk4 and 2 dopri5 settings) for each field source over
3 poloidal transits, collisions off, alpha0 = 0, E_0 = 77, and checks the
largest end-of-run relative error over all particles at the finer setting of
each integrator:

  field          |dp_phi|/|p_phi0|   |dmu|/mu0
  analytic       < 1e-6              < 1e-8      exact field
  analytic_grid  < 1e-6              < 1e-8      production interpolation
  grid (MHD)     < 2e-5              < 5e-5      MHD fixture

(rk4 h = 2.5e-7 tau_c, dopri5 rtol = 1e-12) and that every run reached tlim.
The bounds are about 10x above the values measured when the test was written;
it also checks that the coarser setting is worse by at least 4x in p_phi, so
the errors measured are integration errors that converge.
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOUNDS = {"analytic": (1e-6, 1e-8), "analytic_grid": (1e-6, 1e-8), "grid": (2e-5, 5e-5)}
FINE = {"rk4": "h2.5e-07", "dopri5": "rtol1e-12"}
COARSE = {"rk4": "h5.0e-07", "dopri5": "rtol1e-11"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exe", required=True)
    ap.add_argument("--keep", type=Path, help="keep run directories here")
    args = ap.parse_args()
    base = args.keep or Path(tempfile.mkdtemp())
    ok = True
    for field, (b_pphi, b_mu) in BOUNDS.items():
        out = base / field
        subprocess.run([sys.executable, str(HERE / "scan.py"), "--exe", args.exe, "--out", str(out),
                        "--orbits", "3", "--quick", "--no-plot", "--field", field],
                       check=True, stdout=subprocess.DEVNULL)
        rows = list(csv.DictReader(open(out / "summary.csv")))
        failed = [r for r in rows if r["failed"] != "0"]
        if failed:
            print(f"FAIL {field}: {len(failed)} runs did not finish, e.g. {failed[0]['setting']} {failed[0]['particle']}")
            ok = False
            continue
        for integ, setting in FINE.items():
            rs = [r for r in rows if r["integrator"] == integ and r["setting"] == setting]
            e_pphi = max(float(r["end_dpphi"]) for r in rs)
            e_mu = max(float(r["end_dmu"]) for r in rs)
            rc = [r for r in rows if r["integrator"] == integ and r["setting"] == COARSE[integ]]
            ratio = max(float(r["end_dpphi"]) for r in rc) / max(e_pphi, 1e-300)
            good = e_pphi < b_pphi and e_mu < b_mu and ratio > 4
            ok &= good
            print(f"{'PASS' if good else 'FAIL'} {field:13s} {integ:6s} {setting:10s} "
                  f"max dp_phi {e_pphi:.2e} (< {b_pphi:.0e})  max dmu {e_mu:.2e} (< {b_mu:.0e})  "
                  f"coarse/fine p_phi {ratio:.1f} (> 4)")
    print("PASS: GCE invariants" if ok else "FAIL: GCE invariants")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
