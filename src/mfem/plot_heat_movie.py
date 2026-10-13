#!/usr/bin/env python3
"""Render a movie of the mfem_heat solution from its ParaView collection.

Reads <output_dir>/heat/heat.pvd as written by mfem_heat (MFEM ParaViewDataCollection,
zlib-compressed binary VTU, one piece per MPI rank) and writes an mp4 or gif. Left panel:
u(x, y). Right panel: the y = 0.5 centerline against the exact decaying eigenmode
u = exp(-2 pi^2 kappa t) sin(pi x) sin(pi y).

meshio rejects MFEM's VTU version 2.2 header, so the files are decoded here directly.

Usage: plot_heat_movie.py mfem_heat/heat/heat.pvd -o heat.mp4 [--kappa 1.0] [--fps 10]
"""
import argparse
import base64
import os
import struct
import xml.etree.ElementTree as ET
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter

DTYPES = {"Float64": np.float64, "Float32": np.float32, "Int32": np.int32,
          "Int64": np.int64, "UInt8": np.uint8, "UInt32": np.uint32, "UInt64": np.uint64}


def decode_array(elem, compressed, header_fmt):
    """Decode one inline binary DataArray: base64 header + base64 zlib blocks."""
    text = elem.text.strip()
    hsize = struct.calcsize(header_fmt)
    dtype = DTYPES[elem.get("type")]
    if not compressed:
        raw = base64.b64decode(text)
        return np.frombuffer(raw[hsize:], dtype=dtype)
    # Header is [nblocks, blocksize, lastblocksize, csize_0..csize_{n-1}], base64-encoded
    # on its own, so its encoded length must be computed before the data can be split.
    nblocks = struct.unpack(header_fmt, base64.b64decode(text[:8])[:hsize])[0]
    hlen = 4 * -(-((3 + nblocks) * hsize) // 3)
    header = struct.unpack("<" + header_fmt[1] * (3 + nblocks),
                           base64.b64decode(text[:hlen]))
    data = base64.b64decode(text[hlen:])
    out, pos = [], 0
    for csize in header[3:]:
        out.append(zlib.decompress(data[pos:pos + csize]))
        pos += csize
    return np.frombuffer(b"".join(out), dtype=dtype)


def read_vtu(path):
    root = ET.parse(path).getroot()
    compressed = root.get("compressor") is not None
    header_fmt = "<Q" if root.get("header_type") == "UInt64" else "<I"
    piece = root.find("UnstructuredGrid/Piece")
    pts = decode_array(piece.find("Points/DataArray"), compressed, header_fmt).reshape(-1, 3)
    u = None
    for da in piece.find("PointData"):
        if da.get("Name") == "u":
            u = decode_array(da, compressed, header_fmt)
    return pts[:, :2], u


def read_frame(pvtu):
    """Concatenate all rank pieces listed in a .pvtu."""
    base = os.path.dirname(pvtu)
    root = ET.parse(pvtu).getroot()
    xy, u = [], []
    for piece in root.iter("Piece"):
        p, v = read_vtu(os.path.join(base, piece.get("Source")))
        xy.append(p)
        u.append(v)
    xy, u = np.concatenate(xy), np.concatenate(u)
    # MFEM writes each element's nodes separately, so shared nodes are duplicated.
    _, keep = np.unique(np.round(xy, 12), axis=0, return_index=True)
    return xy[keep], u[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pvd")
    ap.add_argument("-o", "--output", default="heat.mp4")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    base = os.path.dirname(args.pvd)
    frames = [(float(ds.get("timestep")), os.path.join(base, ds.get("file")))
              for ds in ET.parse(args.pvd).getroot().iter("DataSet")]

    xy0, _ = read_frame(frames[0][1])
    tri = mtri.Triangulation(xy0[:, 0], xy0[:, 1])
    line = np.abs(xy0[:, 1] - 0.5) < 1e-9
    order = np.argsort(xy0[line, 0])
    xs = xy0[line, 0][order]
    xfine = np.linspace(0, 1, 200)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    levels = np.linspace(0, 1, 41)
    writer = (FFMpegWriter(fps=args.fps, bitrate=2400) if args.output.endswith(".mp4")
              else PillowWriter(fps=args.fps))
    with writer.saving(fig, args.output, dpi=110):
        cbar = None
        for i, (t, pvtu) in enumerate(frames):
            xy, u = read_frame(pvtu)
            assert np.allclose(xy, xy0), "mesh changed between frames"
            ax0.clear()
            cf = ax0.tricontourf(tri, u, levels=levels, cmap="inferno", extend="both")
            if cbar is None:
                cbar = fig.colorbar(cf, ax=ax0, label="u")
            ax0.set_aspect("equal")
            ax0.set_xlabel("x")
            ax0.set_ylabel("y")
            ax0.set_title(f"mfem_heat  t = {t:.4f}")

            ax1.clear()
            amp = np.exp(-2 * np.pi**2 * args.kappa * t)
            ax1.plot(xfine, amp * np.sin(np.pi * xfine), "k-", lw=1, label="exact")
            ax1.plot(xs, u[line][order], "o", ms=3, color="tab:orange", label="MFEM")
            ax1.set_ylim(0, 1.05)
            ax1.set_xlim(0, 1)
            ax1.set_xlabel("x")
            ax1.set_ylabel("u(x, 0.5)")
            err = np.max(np.abs(u[line][order] - amp * np.sin(np.pi * xs)))
            ax1.set_title(f"centerline y = 0.5, max error {err:.2e}")
            ax1.legend(loc="upper right")
            writer.grab_frame()
            if i == len(frames) - 1:
                fig.savefig(os.path.splitext(args.output)[0] + "_final.png", dpi=110)
    print(f"wrote {args.output} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
