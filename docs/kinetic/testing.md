# Kinetic regression harness

The kinetic side (`src/kinetic`, `src/tasks`, `src/util`, `src/pgen.cpp`) is being
refactored under the same rule as the MHD solver: **no change may alter results**.
This harness is the gate. Each case runs one kinetic executable on a reduced deck,
fingerprints every number the run writes, and compares the fingerprint's SHA-256
with a recorded baseline.

```sh
cd build && ctest -L kinetic          # light cases, about 35 seconds in total
ctest -L slow                         # kin_hybrid: coupled MHD solve, ~20 minutes, run alone
tests/regression/kinetic/run_kinetic.sh build/bin/conservation \
    tests/regression/kinetic/decks/conservation.input \
    --baseline kin_conservation --keep /tmp/kc parthenon/time/nlim=4
                                      # keep the run directory and fingerprint
```

`kin_hybrid` peaks at about 11 GB during MHD set-up. In a container with a
12 GB memory limit, anything running beside it (a parallel build, other tests)
gets it killed with signal 9; run it on its own.

The harness needs `h5py` in `~/.venv-hymera`; `tests/regression/setup_env.sh`
installs it.

## Cases

| Case | Executable | Covers | Threads |
|------|------------|--------|---------|
| `kin_conservation` | `conservation` | `GenerateParticlePoint`, rk4 push, small-angle collisions, `ComputeConservedQuantities` and `conservation.dat` | 1 |
| `kin_conservation_dopri5_nocoll` | `conservation` | dopri5 adaptive push, collisions off | 4 |
| `kin_profile` | `profile` | `ProfileDriver` push-only task list | 4 |
| `kin_single_particle` | `single_particle` | `SingleParticleDriver`, host-side single orbit, `single_particle.dat` | 4 |
| `kin_hybrid` | `hybrid` | MHD coupling, `GenerateParticleCurrentDensity`, field interpolation, `HybridDriver` | 4 |

All decks set `Simulation/parallel_push=0` (serial host push), which keeps
collision runs bit-identical at any thread count. Production runs default to the
parallel push; see `docs/kinetic/outputs.md`.

Each case is run at 1 and 4 OpenMP threads before its baseline is accepted. All
particle data is bit-identical between the two. The exception is
`conservation.dat` in `kin_conservation`: its totals are a `Kokkos::parallel_reduce`
whose summation order depends on the thread split, so that case gates at 1 thread.

## What is fingerprinted

`fingerprint_kinetic.py` walks the run directory in sorted order:

- **HDF5 outputs** (`*.phdf`, `*.rhdf`, `*.h5`): every dataset and attribute, as
  the SHA-256 of its canonical bytes (floats as big-endian float64, integers as
  big-endian int64). Arrays of up to 64 values are also listed value by value, so
  a diff of two fingerprints shows the changed numbers. Volatile attributes
  (`WallTime`, build timestamp, git hash, Kokkos config) are dropped.
- **PETSc MHD state dumps** (`*.rdat`, written at each restart output): SHA-256
  of the raw bytes, which PETSc writes in a fixed big-endian format.
- **Text outputs** (`*.dat`, `*.out`, `*.txt`): every numeric token, as exact hex.
- **Not the log.** Its format is being cleaned up, and every number it prints is
  also stored in the HDF5 `Params`/`Info` attributes.

The fingerprint reads values, not file layout. Moving diagnostics from `.dat`
files into one HDF5 file changes the fingerprint's section names, so that step
re-records baselines once, in a commit of its own that changes nothing else.

## Fixtures

- `fixtures/fields.h5`: the pre-interpolated field grid that the light cases
  load. It was written by `hybrid -i decks/hybrid.input Simulation/dump_raw_fields=1`
  on the CPU (OpenMP) build.
- `fixtures/X_4.petscbin`: relaxed MHD state, loaded with `MHD_Config/ic_binary_load=1`
  so that `kin_hybrid` skips the relaxation solve.

`fixtures/fields.h5` is marked `field_io_version` 2: the indicator is stored
row-major. Files written before that hold the producing build's memory order,
which is column-major for a CUDA build (`inputs/fields.h5` is one).
`LoadRawFieldData` detects the order of such files by comparing them with
`Geometry/input_file`. Before that fix a CPU build read GPU files transposed, and
every particle in `conservation.input` died with `WallImpact` at t=0, before
any push.

## Negative tests

A harness that cannot fail proves nothing. Each perturbation below was applied
to one live line, the executables rebuilt, and the cases run; then the file was
restored.

| Perturbation | conservation | dopri5_nocoll | profile | single_particle |
|--------------|:-:|:-:|:-:|:-:|
| `pgen.cpp:411` initial weight `1.0` to `1.0000000001` | FAIL | FAIL | FAIL | pass (no swarm) |
| `GuidingCenterEquations.hpp:163` radiation drag scaled by 1+1e-7 | FAIL | FAIL | FAIL | FAIL |
| `SmallAngleCollision.hpp:153` drag scaled by 1+1e-7 | FAIL | pass (collisions off) | FAIL | FAIL |
| `rk4.hpp` first `0.5 * h` stage to `0.5000001 * h` | FAIL | pass (dopri5) | FAIL | FAIL |

Each "pass" follows from what the case runs, so none of them is a blind spot.

## Not covered

- `avalanche`: needs the MHD solve, like `hybrid`, and adds no task that
  `kin_hybrid` does not run (`CheckScatter`, `AddSecondaries`, `CleanupParticles`,
  `RandomRemove`). It has no case yet; add one before refactoring `AvalancheDriver.cpp`.
- MPI decomposition: every case runs on one rank.
- Code in `src/kinetic/attic/` (not compiled; see its README).
- `MakeOutputs`: kept as a hook for in-step diagnostics but not called by any driver.

## Recording a baseline

Re-record (`--update`) only when the source tree equals the revision whose
behaviour you mean to freeze, and only in a commit that says so.
