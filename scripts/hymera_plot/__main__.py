"""Command line: python -m hymera_plot {timeseries,profiles,phase-space,fields,all} RUN_DIR"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import io, plots, style


def _timeseries(run, out):
    written = []
    d = io.find_diagnostics(run)
    cons = d.timeseries.get("conservation") if d else None
    if cons is None and (run / "conservation.dat").exists():
        cons = io.read_legacy_conservation(run / "conservation.dat")
    if cons is not None and len(cons) > 0:
        written.append(plots.conservation(cons, out / "conservation.png"))
    cur = d.timeseries.get("currents") if d else None
    if cur is not None and len(cur) > 0:
        written.append(plots.currents(cur, out / "currents.png"))
    return written


def _profiles(run, out):
    d = io.find_diagnostics(run)
    prof = d.profiles.get("collision") if d else None
    if prof is None and (run / "collision_profiles.dat").exists():
        prof = io.read_legacy_collision_profiles(run / "collision_profiles.dat")
    return [plots.collision_profiles(prof, out / "collision_profiles.png")] if prof else []


def _phase_space(run, out):
    paths = io.snapshot_paths(run, "phdf")
    if not paths:
        return []
    snaps = [io.read_snapshot(p) for p in (paths[0], paths[-1])]
    r = plots.phase_space(snaps, out / "phase_space.png")
    return [r] if r else []


def _fields(run, out):
    for p in reversed(io.snapshot_paths(run, "phdf")):
        s = io.read_snapshot(p)
        if s.field_grid is not None:
            return [plots.field_maps(s, out / "fields.png")]
    return []


COMMANDS = {"timeseries": _timeseries, "profiles": _profiles,
            "phase-space": _phase_space, "fields": _fields}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hymera_plot", description=__doc__)
    ap.add_argument("command", choices=[*COMMANDS, "all"])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="output directory (default RUN_DIR/plots)")
    args = ap.parse_args(argv)

    style.apply()
    out = args.out or args.run_dir / "plots"
    todo = COMMANDS.values() if args.command == "all" else [COMMANDS[args.command]]
    written = [p for f in todo for p in f(args.run_dir, out) if p]
    for p in written:
        print(p)
    if not written:
        print(f"nothing to plot in {args.run_dir} for '{args.command}'", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
