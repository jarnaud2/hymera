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

**E on the MHD field.** `HijackEField` sets E_phi = E_0 R_a / R with R_a = 3.
The MHD axis is at R = 3.10, so E_phi on the axis is 0.968 E_0. With
`Avalanche/E_axis = axis`, R_a is the magnetic axis of the loaded field, and
E_phi = E_0 on the axis. The default, `fixed`, keeps R_a = 3.

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

## Analytic baseline

`baseline_analytic.csv` holds the γ of all 45 rows, with the integrator and the wall
time of each row. The figures are `fig11a_baseline_gamma_vs_E0.png` and
`fig11a_baseline_gamma_vs_psin.png`.
- Integrators: rk4 hRK = 2e-6 for 35 rows; dopri5 rtol 1e-9 for the 10 lowest-E_0
  rows (two per r_0). Those runs last up to 70 τ_c, and rk4 at 2e-6 drifts p_phi
  4% per 2 τ_c there.
- 44/45 rows are within 10%. Row 8 (r_0 = 0, E_0 = 22.5) is −10.6%. It was also −8%
  with rk4 at 1e-5 and −9% with rk4 at 2e-6. On the axis there is no orbit to drift,
  so this difference is not from the integrator.

## Large-angle step

`run_fig11a.sh --la-growth G` sets the large-angle step to dtLA = G / γ_ref, the step
over which the expected growth is G, capped at the force step. The reference table
has γ_ref·dt_force = 0.092 in every row. Measured on the analytic field (rk4 hRK =
2e-6; local, 3 threads per run):

| G | nLA | row 0 γ (ref 2.502) | row 9 γ (ref 2.135) | wall time, rows 0 + 9 |
|---|---|---|---|---|
| baseline, dtLA = 1e-5 τ_c | 3682, 4313 | 2.507 | 2.160 | (6 threads: 1536 s + 3424 s) |
| 0.01 | 10 | 2.497 (−0.2%) | 2.141 (+0.3%) | 3888 s + 2975 s |
| 0.1 | 2 | 2.488 (−0.5%) | 2.075 (−2.9%) | 2900 s + 2578 s |

- G = 0.01 (1% growth per step) agrees with the baseline within the seed scatter.
- G = 0.1 is 4% low at r_0 = 0.2. A particle makes at most one secondary per
  large-angle step, which caps the growth when that probability is not small.
- On the CPU the run time hardly depends on dtLA. The cost is the rk4 push itself,
  about 1.8e6 steps per particle per row.
