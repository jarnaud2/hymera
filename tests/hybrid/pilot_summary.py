#!/usr/bin/env python3
"""LaTeX summary of a hybrid run: every parameter, its source, the derived values,
the run log facts and the midplane E / current profiles.

    pilot_summary.py RUN_DIR [--out DIR] [--title TEXT] [--timing FILE]

Reads from RUN_DIR:
  *.out2.000000.phdf    Input@File (Parthenon's dump of every parameter that was
                        read: deck values, command-line values and defaults, each
                        marked) and Params (the Deck package values, including
                        the derived ones)
  <problem_id>.diag.h5  /profiles/midplane_c<cycle> (Hybrid/midplane_profile = 1),
                        /timeseries
  log.txt               if present, the [info] lines
Writes DIR/summary.tex (stand-alone), DIR/midplane.png and DIR/midplane_last.png.
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import h5py

# Parameters computed from others in KineticParams.cpp (not read from the deck),
# with their definitions, for the "derived" table.
DERIVED_DOC = {
    "tau_c": r"collision time $\tau_c$ [s]",
    "En": r"$E_0/E_c$: MHD electric field unit over the critical field",
    "eta_norm": r"$\eta_{\rm plasma}/\eta_0$: resistivity in MHD units",
    "eta_a3VaB0": r"converts the deposited runaway current to $\eta J$ in MHD units",
    "c_aw0": r"$c/(a\,\omega_{ce})$ (guiding-centre drift coefficient)",
    "ct_a": r"$c\,\tau_c/a$ (parallel streaming coefficient)",
    "alpha0": r"$\tau_c/\tau_{\rm rad}$ (synchrotron reaction)",
    "dtLA_over_tauC": r"large-angle step $\Delta t_{LA}/\tau_c$ as used (divides the CD step)",
    "nLA": r"large-angle steps per current-deposition step",
    "nCD": r"current-deposition steps per MHD step",
    "p_BC": r"momentum boundary $p_{BC}$ ($m_ec$)",
    "p_RE": r"runaway threshold $p_{RE}$ ($m_ec$)",
    "gamma_min": r"secondary threshold $\gamma_{\min}$ (deck)",
    "gamma_min_dyn": r"secondary threshold $\gamma_{\min}$ from max $|E_\parallel|$",
    "Rc": r"magnetic axis $R$ ($a$)",
    "Zc": r"magnetic axis $Z$ ($a$)",
    "pmin": r"seed $p_{\min}$ ($m_ec$)", "pmax": r"seed $p_{\max}$ ($m_ec$)",
    "seed_current": r"seed runaway current [A]",
}


def tex(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#").replace("$", r"\$").replace("{", r"\{")
            .replace("}", r"\}").replace("^", r"\^{}").replace("~", r"\~{}"))


def parse_input_dump(text):
    """[(block, name, value, source)] from Parthenon's PAR_DUMP."""
    rows, block = [], None
    for line in text.splitlines():
        line = line.rstrip()
        m = re.match(r"^<(.+)>$", line.strip())
        if m:
            block = m.group(1)
            continue
        if block is None or block == "par_end" or "=" not in line or line.lstrip().startswith("#"):
            continue
        body, _, comment = line.partition("#")
        name, _, value = body.partition("=")
        c = comment.strip().lower()
        if "default value added" in c:
            src = "default"
        elif "added from command line" in c:
            src = "command line"
        elif "updated during run time" in c:
            src = "deck (updated at run time)"
        else:
            src = "deck"
        rows.append((block, name.strip(), value.strip(), src))
    return rows


def fmt(v):
    if isinstance(v, (bytes, np.bytes_)):
        v = v.decode()
    a = np.asarray(v)
    if a.dtype.kind in "fc" and a.size == 1:
        return f"{float(a):.6g}"
    if a.dtype.kind in "iu" and a.size == 1:
        return str(int(a))
    if a.size > 4:
        return f"array {a.shape}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default="VDE pilot: hybrid run summary")
    ap.add_argument("--timing", type=Path, default=None, help="text file with wall-time lines")
    ap.add_argument("--note", action="append", default=[], help="extra line for the set-up section")
    ap.add_argument("--cmdline", default="", help="the run's Block/key=value overrides, to mark them")
    args = ap.parse_args()
    out = args.out or args.run
    out.mkdir(parents=True, exist_ok=True)

    first = sorted(glob.glob(str(args.run / "*.out2.000000.phdf")))
    if not first:
        sys.exit("no *.out2.000000.phdf in run directory")
    with h5py.File(first[0], "r") as f:
        dump = f["Input"].attrs["File"]
        dump = dump.decode() if isinstance(dump, bytes) else str(dump)
        params = {k: f["Params"].attrs[k] for k in f["Params"].attrs}
        info = {k: f["Info"].attrs[k] for k in f["Info"].attrs}
    inputs = parse_input_dump(dump)
    over = {}
    for tok in args.cmdline.split():
        if "=" in tok and "/" in tok.split("=")[0]:
            k, v = tok.split("=", 1)
            b, n = k.rsplit("/", 1)
            over[(b, n)] = v
    inputs = [(b, n, v, "command line" if (b, n) in over else src) for b, n, v, src in inputs]
    last = sorted(glob.glob(str(args.run / "*.out2.0*.phdf")))[-1]
    with h5py.File(last, "r") as f:
        params_end = {k: f["Params"].attrs[k] for k in f["Params"].attrs}
        t_end = float(f["Info"].attrs["Time"])

    diag = sorted(glob.glob(str(args.run / "*.diag.h5")))
    profiles = {}
    if diag:
        with h5py.File(diag[0], "r") as f:
            for k in f.get("profiles", {}):
                if k.startswith("midplane_c"):
                    g = f["profiles"][k]
                    profiles[int(k.split("_c")[1])] = {c: g[c][()] for c in g}
    log = (args.run / "log.txt").read_text(errors="replace") if (args.run / "log.txt").exists() else ""
    info_lines = [l.strip() for l in log.splitlines() if l.strip().startswith("[info]")]

    # Times of the profiles from the outputs (cycle -> time).
    times = {}
    for fn in glob.glob(str(args.run / "*.out2.*.phdf")):
        with h5py.File(fn, "r") as f:
            times[int(f["Info"].attrs["NCycle"])] = float(f["Info"].attrs["Time"])

    plot_profiles(profiles, times, params, out)
    series = {}
    if diag:
        with h5py.File(diag[0], "r") as f:
            if "timeseries/currents" in f:
                g = f["timeseries/currents"]
                series = {k: g[k][()] for k in g}
    steps = [(int(m.group(1)), float(m.group(2))) for m in
             re.finditer(r"^cycle=(\d+) .*wsec_step=([0-9.e+-]+)", log, re.M)]
    plot_currents(series, out)
    write_tex(args, out, inputs, params, info, info_lines, profiles, times, series, steps, params_end, t_end)
    print(f"wrote {out / 'summary.tex'}")


def plot_profiles(profiles, times, params, out):
    """E_phi (solid) and the Ohmic E_n eta J_phi (dashed) on the same axes, one
    colour per time; the runaway part E_n J_re,phi (dotted) if it is not zero."""
    if not profiles:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cyc = sorted(profiles)
    sel = cyc if len(cyc) <= 6 else [cyc[int(round(k))] for k in np.linspace(0, len(cyc) - 1, 6)]
    cm = plt.get_cmap("viridis")
    has_re = any(np.any(profiles[c]["E_re"] != 0) for c in cyc)
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for k, cy in enumerate(sel):
        p = profiles[cy]
        c = cm(k / max(1, len(sel) - 1))
        t = times.get(cy, float("nan"))
        ax.plot(p["R"], p["E_phi"], "-", color=c, lw=1.5, label=f"t = {t * 1e3:.2f} ms")
        ax.plot(p["R"], p["E_ohm"], "--", color=c, lw=1.2)
        if has_re:
            ax.plot(p["R"], p["E_re"], ":", color=c, lw=1.2)
    ax.plot([], [], "-", color="0.3", label=r"$E_\phi$")
    ax.plot([], [], "--", color="0.3", label=r"$E_n\,\eta\,J_\phi$ (Ohmic, MHD $J$)")
    if has_re:
        ax.plot([], [], ":", color="0.3", label=r"$E_n\,J_{re,\phi}$ (runaway)")
    ax.axvline(float(params.get("Deck/Rc", np.nan)), color="0.5", lw=0.8, ls=":")
    ax.set_xlabel(r"$R$ [$a$] at $Z = Z_{\rm axis}$ (dotted vertical: magnetic axis)")
    ax.set_ylabel(r"electric field [$E_c$]")
    ax.set_title(r"Midplane: $E_\phi$ and $\eta J_\phi$ in the same units", loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out / "midplane.png", dpi=150)
    plt.close(fig)


def plot_currents(series, out):
    if not series:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = series["t"] * 1e3
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(t, series["I_re_particles"] / 1e3, "o-", color="#1baf7a", label=r"$I_{re}$ (particles) [kA]")
    ax.set_xlabel("t [ms]")
    ax.set_ylabel(r"$I_{re}$ [kA]")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, series["I_ohmic"] / 1e6, "s--", color="#eb6834", label=r"$I_{\rm ohmic}$ [MA]")
    ax2.set_ylabel(r"$I_{\rm ohmic}$ [MA]")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
    fig.savefig(out / "currents.png", dpi=150)
    plt.close(fig)


def write_tex(args, out, inputs, params, info, info_lines, profiles, times, series, steps, params_end, t_end):
    L = []
    w = L.append
    w(r"\documentclass[10pt,a4paper]{article}")
    w(r"\usepackage[margin=1.8cm]{geometry}\usepackage{booktabs,longtable,graphicx}")
    w(r"\usepackage[hidelinks]{hyperref}\usepackage{xurl}")
    w(r"\begin{document}")
    w(rf"\section*{{{tex(args.title)}}}")
    w(r"\subsection*{Run}")
    w(r"\begin{itemize}")
    w(rf"\item Run directory: \url{{{args.run.resolve()}}}")
    w(rf"\item Cycles: {int(info.get('NCycle', 0))} at the first output; outputs with midplane profiles: {len(profiles)}")
    if times:
        w(rf"\item Simulated time covered: {min(times.values()) * 1e3:.3f}--{max(times.values()) * 1e3:.3f} ms")
    w(rf"\item Mesh blocks: {int(info.get('NumMeshBlocks', 0))}, ranks: {int(np.size(info.get('BlocksPerPE', [0])))}")
    for n in args.note:
        w(rf"\item {n}")
    if args.timing and args.timing.exists():
        for line in args.timing.read_text().splitlines():
            if line.strip():
                w(rf"\item {tex(line.strip())}")
    w(r"\end{itemize}")

    if profiles:
        w(r"\subsection*{Midplane profiles}")
        w(r"Profiles along $R$ at the magnetic-axis height $Z=Z_c$, in units of the")
        w(r"critical field $E_c$: the toroidal electric field $E_\phi$, the Ohmic part")
        w(r"$E_n\eta J_\phi$ with $J$ the MHD current, and the runaway part $E_nJ_{re,\phi}$")
        w(r"(the current deposited by the particles and passed to the MHD solve). In the")
        w(r"MHD Ohm's law, $E = E_n(-V\times B + \eta J) - E_nJ_{re}$.")
        w(r"\begin{center}\includegraphics[width=0.9\linewidth]{midplane.png}\end{center}")

    if series:
        w(r"\subsection*{Currents}")
        w(r"Total toroidal currents from the diagnostics file: the Ohmic (MHD) current and")
        w(r"the runaway current of the particles.")
        w(r"\begin{center}\includegraphics[width=0.7\linewidth]{currents.png}\end{center}")
        w(r"\begin{center}\small\begin{tabular}{rrrr}\toprule $t$ [ms] & $I_{\rm ohmic}$ [MA] & $I_{re}$ [kA] & growth per step\\\midrule")
        t, io, ir = series["t"], series["I_ohmic"], series["I_re_particles"]
        for k in range(len(t)):
            gr = f"{ir[k] / ir[k - 1]:.3f}" if k > 0 and ir[k - 1] > 0 else "--"
            w(rf"{t[k] * 1e3:.2f} & {io[k] / 1e6:.4f} & {ir[k] / 1e3:.2f} & {gr}\\")
        w(r"\bottomrule\end{tabular}\end{center}")
    if steps:
        w(r"\subsection*{Cost per MHD step}")
        w(r"Wall time per MHD step from the run log (\texttt{wsec\_step}); cycle 0 is the")
        w(r"set-up (MHD initialisation, first MHD step, seeding).")
        w(r"\begin{center}\small\begin{tabular}{rr}\toprule cycle & wall [s]\\\midrule")
        for c, ws in steps:
            w(rf"{c} & {ws:.0f}\\")
        w(r"\bottomrule\end{tabular}\end{center}")

    if info_lines:
        w(r"\subsection*{Set-up facts from the log}")
        w(r"\begin{itemize}\small")
        for l in info_lines[:40]:
            w(rf"\item \texttt{{{tex(l.replace('[info]', '').strip())}}}")
        w(r"\end{itemize}")

    w(r"\subsection*{Derived values (Deck package)}")
    w(rf"Values at $t=0$ (first output) and at the last output ($t={t_end * 1e3:.2f}$\,ms); they")
    w(r"differ only for parameters the run updates (momentum boundary, magnetic axis).")
    w(r"\begin{longtable}{lllp{7.5cm}}\toprule parameter & $t=0$ & end & meaning\\\midrule\endhead")
    for k in sorted(params):
        if not k.startswith("Deck/"):
            continue
        name = k[5:]
        v = fmt(params[k])
        if v.startswith("array"):
            continue
        ve = fmt(params_end.get(k, params[k]))
        ve = ve if ve != v else ""
        w(rf"\texttt{{{tex(name)}}} & {tex(v)} & {tex(ve)} & {DERIVED_DOC.get(name, '')}\\")
    w(r"\bottomrule\end{longtable}")

    w(r"\subsection*{All input parameters}")
    w(r"Every parameter the run read, from Parthenon's parameter dump in the first")
    w(r"output. Source: \emph{deck} (the input file), \emph{command line}, or")
    w(r"\emph{default} (not in the deck; the code's default was used).")
    w(r"\begin{longtable}{lllp{3cm}}\toprule block & parameter & value & source\\\midrule\endhead")
    last = None
    for block, name, value, src in inputs:
        b = tex(block) if block != last else ""
        last = block
        val = value if len(value) < 60 else value[:57] + "..."
        w(rf"{b} & \texttt{{{tex(name)}}} & \texttt{{{tex(val)}}} & {src}\\")
    w(r"\bottomrule\end{longtable}")
    w(r"\end{document}")
    (out / "summary.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
