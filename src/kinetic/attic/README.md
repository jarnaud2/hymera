# attic/

This directory holds kinetic-side source that is **not built** and **not part of
any CMake target**. Each function was moved verbatim (byte-for-byte identical
lines) out of its source file because nothing in `src/` or `tests/` called it. A
coverage build of the kinetic regression cases (`docs/kinetic/testing.md`) never
ran any of them either. They were moved rather than deleted so the code stays
in-tree and greppable.

Re-enabling any of them (adding a caller or a CMake target) changes what the
harness checks; add a regression case for it in the same commit.

## kinetic_attic.cpp (from `src/kinetic/kinetic.cpp`)

- **InitializeAnalytic** -- an older copy of `Initialize` for the analytic
  (`AnalyticField`) slab model, without the MHD context. About 280 lines that
  duplicate most of `Initialize`. Its only caller went away when every driver moved
  to `Initialize(pin, mhd_context)`, with `mhd_context = nullptr` for runs that
  load fields from a file.
- **GetCoords** (three overloads) -- one-line accessors for `MeshBlock::coords`.
  Never referenced.
- **SaveState**, **RestoreState**, **RestoreJre** -- the particle and current
  snapshot used by the old predictor-corrector reset loop. Before each corrector
  pass, `HybridDriver` reset particles and MHD back to the start of the step;
  the reset tasks were removed in `e04ea11` (2026-06-25). The `saved_p`,
  `saved_xi`, `saved_R`, `saved_phi`, `saved_Z` and `saved_w` swarm variables
  that held the snapshot were removed from `kinetic.hpp` and from the swarm
  registration in `Initialize` at the same time as this move. Restart files
  written after this commit therefore no longer carry those variables.
  `Time/nPR` is still set in some input decks but no code reads it.

## pgen_attic.cpp (from `src/pgen.cpp`)

- **GenerateParticleSquare** -- seeds particles uniformly in (R, Z) over the
  whole domain, with random p and xi. Its only caller was the disabled
  `tests/kinetic/conservation_test.cpp`.

## Whole files never compiled

These three files were in `src/kinetic/` but in no CMake target. They are kept
whole, under an `_attic` name:

- **RunawayDriver_attic.cpp / .h** -- an earlier single driver (`RunawayDriver`)
  with its own `PushParticles(Mesh *, SimTime)`, `CheckScatter`,
  `CleanupParticles` and `AddSecondaries`. The live versions of those tasks are
  in `src/tasks/Particles.cpp`, used by `HybridDriver` and `AvalancheDriver`. The
  header was still included by four compiled files, none of which used it; they
  now include `<parthenon/driver.hpp>` directly.
- **ConfigurationDomainGeometry_attic.cpp** -- an unfinished `InitializeGeometry`
  that does not compile (`std::stared_ptr`, undeclared `NR`, `NZ` and file name).
  The geometry is read in `SetupFieldGrid` (`kinetic.cpp`).

`MakeOutputs` was unreachable too, but stays live in `Diagnostics.cpp` on
purpose: it is the in-step snapshot hook (`Output/substep_snapshots`).
