#!/usr/bin/env python3
"""Canonical text fingerprint of a kinetic run directory.

Writes the numbers a run produced in exact form, so that a SHA-256 of the
output is equal between two runs if and only if every number they wrote is
bit-identical. Each HDF5 array becomes one line holding the SHA-256 of its
canonical bytes; arrays of up to FULL_LIST_MAX values are also listed.
The fingerprint reads values, not file layout:

  * Parthenon HDF5 outputs (*.phdf, *.rhdf): every dataset, every attribute
    except the volatile ones listed in VOLATILE_ATTRS (wall time, build
    stamps), which change between identical runs.
  * PETSc binary MHD state dumps (*.rdat): SHA-256 of the raw bytes.
  * Whitespace-separated numeric text files (*.dat, *.out, *.txt): every
    numeric token. Comment and header lines are skipped.

The run log is deliberately not fingerprinted: its format is being cleaned
up, and every number it prints (dtLA, nLA, seed point, cycle, time, dt) is
also stored in the Params and Info attributes of the HDF5 outputs.

Files are visited in sorted order and each section starts with the file
name, so a diff of two fingerprints names the file, the dataset and the
index of the first differing value.

  fingerprint_kinetic.py RUN_DIR > fingerprint.txt
"""
import argparse
import hashlib
import re
import struct
import sys
from pathlib import Path

import h5py
import numpy as np

# Attributes that differ between two runs of the same binary on the same input.
VOLATILE_ATTRS = {
    "WallTime",
    "ParthenonBuildTimestamp",
    "ParthenonGitHash",
    "ParthenonGitBranch",
    "KokkosConfig",
}

# Inputs staged into the run directory, not produced by the run.
STAGED = {"fields.h5", "X_4.petscbin", "X_4.petscbin.info", "log.txt"}

# Arrays up to this size are also listed value by value.
FULL_LIST_MAX = 64

# Text outputs whose numbers are fingerprinted.
TEXT_SUFFIXES = {".dat", ".out", ".txt"}

# PETSc binary outputs (MHD restart dumps written by WorkBeforeRestartOutput).
PETSC_SUFFIXES = {".rdat"}

NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$|^[+-]?(nan|inf)$", re.I)


def hexd(x):
    """Exact representation of one value."""
    if isinstance(x, (float, np.floating)):
        return struct.pack(">d", float(x)).hex()
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode(errors="replace")
    return str(x)


def canonical_bytes(arr):
    """Byte image of an array that is independent of storage dtype width and
    byte order: floats as big-endian float64, integers as big-endian int64,
    strings as their UTF-8 text separated by NUL."""
    if arr.dtype.kind in "SUO":
        return b"\0".join(hexd(v).encode() for v in arr.ravel())
    if arr.dtype.kind == "f":
        return arr.astype(">f8").tobytes()
    return arr.astype(">i8").tobytes()


def emit_array(out, prefix, arr):
    """One line per array: shape, kind and the SHA-256 of its canonical bytes.

    Large arrays (the Hermite field grid stored in every output is 67M
    doubles) cannot be listed value by value. Small arrays are listed in full
    as well, so a diff of two fingerprints shows the differing values."""
    arr = np.asarray(arr)
    digest = hashlib.sha256(canonical_bytes(arr)).hexdigest()
    out.write(f"{prefix} shape={arr.shape} kind={arr.dtype.kind} sha256={digest}\n")
    if arr.size <= FULL_LIST_MAX:
        for i, v in enumerate(arr.ravel()):
            out.write(f"{prefix}[{i}] {hexd(v)}\n")


def emit_attrs(out, path, obj):
    for k in sorted(obj.attrs):
        if k in VOLATILE_ATTRS:
            continue
        emit_array(out, f"{path}@{k}", obj.attrs[k])


def emit_hdf5(out, fname):
    with h5py.File(fname, "r") as f:
        emit_attrs(out, "/", f)
        names = []
        f.visit(names.append)
        for name in sorted(names):
            obj = f[name]
            emit_attrs(out, name, obj)
            if isinstance(obj, h5py.Dataset):
                emit_array(out, name, obj[()])


def emit_text(out, fname):
    for lineno, line in enumerate(fname.read_text(errors="replace").splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        toks = [t for t in line.split() if NUMBER.match(t)]
        if toks:
            vals = " ".join(struct.pack(">d", float(t)).hex() for t in toks)
            out.write(f"L{lineno} {vals}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    out = sys.stdout
    for fname in sorted(args.run_dir.iterdir()):
        if fname.name in STAGED or not fname.is_file():
            continue
        if fname.suffix in (".phdf", ".rhdf", ".h5"):
            out.write(f"== {fname.name}\n")
            emit_hdf5(out, fname)
        elif fname.suffix in PETSC_SUFFIXES:
            # PETSc binary Vec dumps (MHD restart state): fixed big-endian
            # format, so the raw bytes are already canonical.
            digest = hashlib.sha256(fname.read_bytes()).hexdigest()
            out.write(f"== {fname.name}\nbytes sha256={digest}\n")
        elif fname.suffix in TEXT_SUFFIXES:
            out.write(f"== {fname.name}\n")
            emit_text(out, fname)


if __name__ == "__main__":
    main()
