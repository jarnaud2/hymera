"""Readers for the files a kinetic run writes.

* ``<problem_id>.diag.h5``: the run's diagnostics file (time series and profiles),
  see src/kinetic/DiagnosticsFile.hpp.
* ``<problem_id>.outN.*.phdf`` / ``.rhdf``: Parthenon outputs. The particle swarm
  is under ``particles/SwarmVars``; the field-plot grid, when written
  (Output/plot_fields=1), is the ``Hermite_Field_Plot_*`` attributes of ``Params``.
* Legacy text files (``conservation.dat``, ``collision_profiles.dat``), for runs
  made before the diagnostics file existed.
"""
from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


@dataclass
class Table:
    """Named columns of equal length, with their units."""
    name: str
    columns: dict[str, np.ndarray]
    units: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, key):
        return self.columns[key]

    def __len__(self):
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def label(self, key):
        u = self.units.get(key, "")
        return f"{key} [{u}]" if u else key


def _read_group(g, name):
    cols, units = {}, {}
    for key in g:
        ds = g[key]
        cols[key] = ds[()]
        u = ds.attrs.get("units", "")
        units[key] = u.decode() if isinstance(u, bytes) else str(u)
    return Table(name, cols, units)


class Diagnostics:
    """A ``<problem_id>.diag.h5`` file."""

    def __init__(self, path):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.problem_id = f["meta"].attrs.get("problem_id", "")
            self.timeseries = {k: _read_group(f["timeseries"][k], k) for k in f.get("timeseries", {})}
            self.profiles = {k: _read_group(f["profiles"][k], k) for k in f.get("profiles", {})}

    def __repr__(self):
        return (f"Diagnostics({self.path.name}: timeseries {sorted(self.timeseries)}, "
                f"profiles {sorted(self.profiles)})")


def find_diagnostics(run_dir, problem_id=None):
    """The run's ``<problem_id>.diag.h5`` in run_dir, or None.

    Without problem_id it is taken from the Parthenon outputs in run_dir; failing
    that, run_dir must hold exactly one ``*.diag.h5``."""
    run_dir = Path(run_dir)
    if problem_id is None:
        outs = sorted(run_dir.glob("*.out*.phdf")) + sorted(run_dir.glob("*.out*.rhdf"))
        if outs:
            problem_id = outs[0].name.split(".out")[0]
    if problem_id is not None:
        path = run_dir / f"{problem_id}.diag.h5"
        return Diagnostics(path) if path.exists() else None
    hits = sorted(run_dir.glob("*.diag.h5"))
    if len(hits) > 1:
        raise ValueError(f"several diagnostics files in {run_dir}: {[h.name for h in hits]}; "
                         "pass problem_id")
    return Diagnostics(hits[0]) if hits else None


def read_legacy_conservation(path):
    """``conservation.dat``: t, <p_phi>, <mu>, total weight."""
    a = np.loadtxt(path, ndmin=2)
    names = ["t", "p_phi_mean", "mu_mean", "weight_total"]
    units = ["tau_c", "", "", ""]
    return Table("conservation", dict(zip(names, a.T)), dict(zip(names, units)))


def read_legacy_collision_profiles(path):
    """``collision_profiles.dat``: columns named by its header line."""
    a = np.loadtxt(path, ndmin=2)
    names = ["p", "gamma", "dtSA", "psi", "CB", "CF", "CouLogee_ratio", "probability"]
    return Table("collision", dict(zip(names, a.T)))


@dataclass
class Snapshot:
    """One Parthenon output: time, cycle, particle arrays, optional field grid."""
    path: Path
    time: float
    cycle: int
    particles: dict[str, np.ndarray]
    field_grid: dict[str, np.ndarray] | None


FIELD_NAMES = ["B", "J", "dBdR", "dBdZ", "E", "dbdt"]
SCALAR_NAMES = ["Psi", "Bsq", "Bmag"]  # stored in quantity slot 6, components 0..2


def read_snapshot(path):
    with h5py.File(path, "r") as f:
        info = f["Info"].attrs
        parts = {}
        if "particles/SwarmVars" in f:
            for key, ds in f["particles/SwarmVars"].items():
                parts[key.removeprefix("particle.").removeprefix("swarm.")] = ds[()]
        grid = None
        params = f["Params"].attrs if "Params" in f else {}
        if "Deck/Hermite_Field_Plot_data_F" in params:
            grid = {
                "R": np.asarray(params["Deck/Hermite_Field_Plot_data_R"]),
                "Z": np.asarray(params["Deck/Hermite_Field_Plot_data_Z"]),
                "t": np.asarray(params["Deck/Hermite_Field_Plot_data_t"]),
                # (NR, NZ, Nt, 3 components, 7 quantities). Parthenon stores the
                # ParArrayND padded to rank 7 with leading unit axes.
                "F": _squeeze_leading(np.asarray(params["Deck/Hermite_Field_Plot_data_F"]), 5),
            }
        return Snapshot(Path(path), float(info["Time"]), int(info["NCycle"]), parts, grid)


def _squeeze_leading(a, rank):
    """Drop leading unit axes until a has the given rank."""
    while a.ndim > rank and a.shape[0] == 1:
        a = a[0]
    if a.ndim != rank:
        raise ValueError(f"field grid has shape {a.shape}, expected rank {rank}")
    return a


def snapshot_paths(run_dir, kind="phdf"):
    """Parthenon outputs in run_dir, in cycle order (``final`` last)."""
    paths = glob.glob(str(Path(run_dir) / f"*.{kind}"))

    def key(p):
        m = re.search(r"\.(\d+)\." + kind + "$", p)
        return (0, int(m.group(1))) if m else (1, 0)

    return [Path(p) for p in sorted(paths, key=key)]


def field_component(grid, quantity, component, time_index=0):
    """2-D array (NR, NZ) of one field quantity. quantity: one of FIELD_NAMES
    (component 0..2 = R, phi, Z) or SCALAR_NAMES (component ignored)."""
    F = grid["F"]
    if quantity in FIELD_NAMES:
        return F[:, :, time_index, component, FIELD_NAMES.index(quantity)]
    return F[:, :, time_index, SCALAR_NAMES.index(quantity), 6]
