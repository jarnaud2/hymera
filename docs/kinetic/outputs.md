# Kinetic outputs

What a kinetic run writes, and the input parameters that control it.

## Parthenon outputs (`<problem_id>.outN.*.phdf`, `.rhdf`)

These are set up by the `<parthenon/outputN>` blocks of the deck. Each file holds
the particle swarm variables named in `particles_variables`, and every parameter
of the `Deck` package.

### Field-plot grid (`Hermite_Field_Plot_*`)

The `Deck` package holds the interpolated fields sampled on a uniform (R, Z, t)
grid: B, J, dB/dR, dB/dZ, E, db/dt, plus psi, B² and |B|. The t samples span one
MHD step. It is a `Deck` parameter, so it is copied into every output file.
`PlotFieldsTime` refills it before each output in `hybrid`, `avalanche` and `mhd`.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `Output/plot_fields` | 1 | 0: the grid is not created, filled or written. |
| `Output/NR_plot` | 10 | R samples |
| `Output/NZ_plot` | 20 | Z samples |
| `Output/Nt_plot` | 3 | t samples across one MHD step |

`Hermite_Field_Plot_data_F` has shape `NR_plot x NZ_plot x Nt_plot x 3 x 7`. At
the defaults it is 12,600 doubles, about 100 KB. Before 2026-10 the defaults were
400 x 800 x 10, which added 537 MB to every output file. In the reduced `hybrid`
test deck, a particle output is now 172 KB with the grid and 68 KB without it,
down from 537 MB. To get the old resolution back, set
`Output/NR_plot=400 Output/NZ_plot=800 Output/Nt_plot=10`.

The grid is diagnostics only. Two things used to depend on it and no longer do:

- The momentum boundary (`UpdateMomentumBoundary`, the max |E_par| search that
  sets `p_BC` and `p_RE`) samples its own points, `Numerical/NR_maxE` x
  `Numerical/NZ_maxE` (default 400 x 800).
- The psi and magnetic-axis refresh before each output (`RefreshFluxAndAxis`)
  runs whether or not the grid is written. The particle push reads that psi.

Results are identical with `plot_fields` 0 or 1, and for any plot size.

## Other files

| File | Written by | Control |
|------|------------|---------|
| `collision_profiles.dat` | `Initialize`, rank 0 | always |
| `conservation.dat` | `ComputeConservedQuantities` | `Simulation/EnableComputeConservedQuantities=1`; name `Simulation/conservation_log` |
| `current.out` | truncated by `Initialize` | `Simulation/file_path` |
| `single_particle.dat` | `single_particle` | `SingleParticle/output_file` |
| `<basename>.<n>.rdat` | `WorkBeforeRestartOutput`: MHD state next to each Parthenon restart | `MHD_Config/file_extention` |
| `fields.h5` | `hybrid` with `Simulation/dump_raw_fields=1`, then exits | `Simulation/raw_fields_file` |
| `TotalCurrent.txt` | MHD monitor (`src/mhd`) | `MHD_Config/monitor` |
