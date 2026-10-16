# fig11a avalanche benchmark

Growth rate γ of the runaway avalanche against the reference of the avalanche paper
(fig. 11a), on the exact analytic field. The benchmark covers the guiding-centre push,
small-angle collisions and large-angle (Møller) collisions.

- `fig11a.jld2` (HDF5): the reference table, 45 rows.
  - `ES`: E_0.
  - `R0s`: r_0 ∈ {0, 0.2, 0.4, 0.6, 0.8}.
  - `tlims`: run length [s].
  - `γ_av`: growth rate [1/τ_c].
- `avalanche.jl`: the owner's Pluto notebook that compares runs with the reference. The
  fit in `fig11a.py` follows it:
  - take the last 50 of the 100 outputs;
  - divide N by N at the first of them, and t by τ_c;
  - fit `exp(γ t)` by least squares.
  - On the same data this fit agrees with Julia's LsqFit to 3e-10.
- `fig11a.input`: the deck of the reference run `fig11a_20k_33`, read from its output
  file (kept outside git, `*.phdf` is ignored). Changes from that run:
  - `Avalanche/field_source = analytic` and `Avalanche/seed = ring`. These replace the
    old `InitializeAnalytic` (now in `src/kinetic/attic`) and `GenerateParticleRings`.
  - `Avalanche/E_0`: E_0 as a real number. `AnalyticField/E_0` is read as an integer
    elsewhere, and Parthenon does not allow one parameter to be read as two types.
  - `BoundaryConditions/gamma_min = gamma_BC = 1.02`, the reference values. The code
    default `1 + 1e-4` makes the large-angle sampler about 100× slower, because it
    keeps far fewer of its draws.
  - `integrator = rk4`, `hRK = 2e-6`. The reference used dopri5 with rtol 1e-9.
    rk4 at `hRK = 1e-5` is unstable for off-axis orbits. A particle at r = 0.4 loses
    85% of p_phi in 0.02 τ_c and falls onto the magnetic axis, which inflated γ at
    r_0 ≥ 0.2 by 20% and more. At 2e-6, p_phi is kept to 4e-4 over that time. rk4 at
    2e-6 costs 2.7× less than dopri5 at rtol 1e-9.
- `run_fig11a.sh EXE OUTDIR [--rows "..."] [--set "..."]`: runs table rows. Each row
  sets E_0, r_0 and tlim, with `dt_force = output dt = tlim / 100`. Output goes to
  `OUTDIR/fig11a_<i>/`.
- `fig11a.py OUTDIR`: fits γ, writes `gamma.csv` and plots γ against E_0 and against
  psi_n. It exits with status 1 if any row is off by more than 10%.

The particle count N is the sum of weights of alive particles, read from
`<problem_id>.diag.h5:/timeseries/avalanche`, written at every output.

## Fields and seeds of `avalanche`

| `Avalanche/field_source` | field | MHD solver |
|---|---|---|
| `mhd` (default) | MHD initial state, E replaced by `HijackEField` | yes |
| `analytic` | `AnalyticFieldEvaluator` (q0, q2, R_a = 3, B0 = +1, Avalanche/E_0) | no |
| `file` | `Simulation/load_fields` (fields.h5), E replaced by `HijackEField` | no |

| `Avalanche/seed` | placement |
|---|---|
| `point` (default) | `ParticleSeed/Rseed, Zseed` |
| `ring` | circle of radius `ParticleSeed/r_0` around the magnetic axis |
| `surface` | the ring, then moved along each ray to the flux surface `psi_n = Avalanche/psi_n`. If `psi_n` is not given, the analytic map of r_0 is used, so one table drives both field sources. |

psi_n = psi / psi_edge, with psi = 0 on the axis.
- `analytic`: psi_edge is psi at minor radius 1, so
  psi_n(r) = ln(1 + q2 r²/q0) / ln(1 + q2/q0).
- `file`: psi_edge is psi on the separatrix.

**Pitch sign.** Runaways move against E, so the sign of ξ (along b) must be
−sign(B_phi E_phi).
- The analytic field has B_phi > 0, so ξ ∈ [−0.95, −0.85].
- The MHD fixture has B_phi < 0, so the MHD deck needs ξ ∈ [0.85, 0.95].
- The run stops at set-up if the sign is wrong.

## ctest

`ctest -L avalanche` runs `kin_avalanche_fig11a`, which is `check_fig11a.sh` on rows
0 (r_0 = 0, E_0 = 70.5) and 9 (r_0 = 0.2, E_0 = 71.4).
- Duration: 26 + 57 min on 6 cores.
- Measured at rk4 hRK = 2e-6: row 0 gives γ = 2.508 (reference 2.502, +0.2%); row 9
  gives 2.155 (reference 2.135, +0.9%).
- Seed scatter on row 0 is σ = 1.1% (3 seeds at hRK = 1e-5).
- Negative test: `--set Simulation/EnableSmallAngleCollisions=0` on row 0 gives
  γ ≈ −0.6 and fails. Without pitch-angle scattering the beam makes almost no
  secondaries above gamma_min.
