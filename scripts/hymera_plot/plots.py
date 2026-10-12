"""The plots. Each function takes the data it needs and an output path, and
returns the path written (or None when the run has no such data)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import io, style


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def conservation(table: io.Table, out):
    """Relative drift of the weight-averaged invariants. One panel per quantity,
    on its own scale: they differ by orders of magnitude, so a shared or dual
    axis would hide one of them."""
    t = table["t"]
    names = [c for c in ("p_phi_mean", "mu_mean") if c in table.columns]
    titles = {"p_phi_mean": "Toroidal canonical momentum ⟨p_φ⟩",
              "mu_mean": "Magnetic moment ⟨μ⟩"}
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(7, 2.4 * (len(names) + 1)),
                             sharex=True, constrained_layout=True)
    for ax, name, color in zip(axes, names, style.CATEGORICAL):
        y = table[name]
        ref = y[0] if y[0] != 0 else 1.0
        ax.plot(t, (y - y[0]) / abs(ref), color=color)
        ax.set_title(titles[name], loc="left")
        ax.set_ylabel("relative change")
        ax.axhline(0.0, color=style.TEXT_SECONDARY, linewidth=0.8)
    ax = axes[-1]
    ax.plot(t, table["weight_total"], color=style.CATEGORICAL[len(names)])
    ax.set_title("Total particle weight", loc="left")
    ax.set_ylabel("weight")
    ax.set_xlabel(table.label("t"))
    return _save(fig, out)


def currents(table: io.Table, out):
    """Runaway current: particle sum vs deposited-grid integral, plus the Ohmic
    current. The two runaway estimates share units and are drawn on one axis,
    each labelled at its line end; the Ohmic current is a different magnitude
    and gets its own panel."""
    t = table["t"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, constrained_layout=True)
    for i, (key, label) in enumerate([("I_re_particles", "particle sum"),
                                      ("I_re_grid", "deposited on grid")]):
        a1.plot(t, table[key], color=style.series_color(i), marker="o" if len(t) < 30 else None,
                label=label)
    a1.set_title("Runaway current", loc="left")
    a1.set_ylabel(table.label("I_re_particles"))
    a1.legend(loc="best")
    a2.plot(t, table["I_ohmic"], color=style.series_color(2), marker="o" if len(t) < 30 else None)
    a2.set_title("Ohmic current (Hermite field)", loc="left")
    a2.set_ylabel(table.label("I_ohmic"))
    a2.set_xlabel(table.label("t"))
    return _save(fig, out)


def collision_profiles(table: io.Table, out):
    """Collision coefficients against momentum: small multiples, one quantity per
    panel, shared x."""
    p = table["p"]
    keys = [k for k in ("CB", "CF", "dtSA", "probability") if k in table.columns]
    titles = {"CB": "Pitch-angle scattering C_B", "CF": "Friction C_F",
              "dtSA": "Small-angle step Δt_SA", "probability": "Large-angle probability"}
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True, constrained_layout=True)
    for ax, key in zip(axes.flat, keys):
        y = table[key]
        ax.plot(p, y, color=style.CATEGORICAL[0])
        if np.all(y > 0):
            ax.set_yscale("log")
        ax.set_title(titles[key], loc="left")
    for ax in axes[-1]:
        ax.set_xlabel(table.label("p") if table.units.get("p") else "p [m_e c]")
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    return _save(fig, out)


def phase_space(snapshots: list[io.Snapshot], out):
    """(p, xi) histogram of alive particles, first vs last output, on one shared
    colour scale."""
    picks = [snapshots[0], snapshots[-1]] if len(snapshots) > 1 else snapshots[:1]
    data = []
    for s in picks:
        p, xi = s.particles.get("p"), s.particles.get("xi")
        if p is None or xi is None or len(p) == 0:
            return None
        alive = (s.particles["status"] & 2) != 0 if "status" in s.particles else np.ones(len(p), bool)
        w = s.particles.get("weight", np.ones_like(p))
        data.append((p[alive], xi[alive], w[alive], s))
    p_all = np.concatenate([d[0] for d in data])
    p_edges = np.linspace(p_all.min(), p_all.max(), 41) if np.ptp(p_all) > 0 else 41
    xi_edges = np.linspace(-1.0, 1.0, 41)
    hists = [np.histogram2d(d[0], d[1], bins=[p_edges, xi_edges], weights=d[2]) for d in data]
    vmax = max(h[0].max() for h in hists) or 1.0
    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 4), sharey=True,
                             constrained_layout=True, squeeze=False)
    for ax, (h, pe, xe), (_, _, _, s) in zip(axes[0], hists, data):
        m = ax.pcolormesh(pe, xe, h.T, cmap=style.SEQUENTIAL_CMAP, vmin=0, vmax=vmax)
        ax.set_title(f"cycle {s.cycle}, t = {s.time:.4g} s  ({int((h > 0).sum())} cells)", loc="left")
        ax.set_xlabel("p [m_e c]")
        ax.grid(False)
    axes[0][0].set_ylabel("pitch ξ")
    fig.colorbar(m, ax=axes[0].tolist(), label="particle weight per cell")
    return _save(fig, out)


def field_maps(snapshot: io.Snapshot, out, time_index=0):
    """Toroidal B, J, E and flux psi on the field-plot grid: small multiples."""
    g = snapshot.field_grid
    if g is None:
        return None
    R, Z = g["R"], g["Z"]
    panels = [("B", 1, "B_φ", style.SEQUENTIAL_CMAP), ("J", 1, "J_φ", style.DIVERGING_CMAP),
              ("E", 1, "E_φ", style.DIVERGING_CMAP), ("Psi", 0, "ψ", style.SEQUENTIAL_CMAP)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True, constrained_layout=True)
    for ax, (q, comp, title, cmap) in zip(axes, panels):
        v = io.field_component(g, q, comp, time_index)
        if cmap == style.DIVERGING_CMAP:
            lim = float(np.max(np.abs(v))) or 1.0
            m = ax.pcolormesh(R, Z, v.T, cmap=cmap, vmin=-lim, vmax=lim, shading="nearest")
        else:
            m = ax.pcolormesh(R, Z, v.T, cmap=cmap, shading="nearest")
        ax.set_aspect("equal")
        ax.set_title(title, loc="left")
        ax.set_xlabel("R [a]")
        ax.grid(False)
        fig.colorbar(m, ax=ax, shrink=0.8)
    axes[0].set_ylabel("Z [a]")
    fig.suptitle(f"Interpolated fields, cycle {snapshot.cycle}, t = {snapshot.time:.4g} s "
                 f"(grid {len(R)} x {len(Z)})", x=0.01, ha="left")
    return _save(fig, out)
