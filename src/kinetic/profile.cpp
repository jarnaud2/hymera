//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC // for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
//
// `profile` executable: profiles the particle-push kernel (PushParticles) in
// isolation. No MHD solve -- the interpolated Hermite field grid is LOADED from
// an HDF5 file (produced by a `hybrid` run with Simulation/dump_raw_fields=1)
// instead of being interpolated from the MHD state. Particles are seeded fresh
// via the standard pgen generators. Use with the Kokkos-tools NVTX connector
// (KOKKOS_TOOLS_LIBS=.../libkp_nvtx_connector.so) under nsys for profiling.

#include <iostream>
#include <format>
#include <fstream>
#include <sstream>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <parthenon_manager.hpp>

#include <Kokkos_Core.hpp>
#include "ProfileDriver.h"
#include "pgen.hpp"
#include "kinetic.hpp"
#include "AppSetup.hpp"
#include "mhd.h"

using namespace parthenon;
using namespace parthenon::driver::prelude;


int main(int argc, char *argv[]) {
  User * p_mhd_config;

  ParthenonManager pman;
  // Petsc/MPI env init only -- we do NOT build or step the MHD solver.
  if (auto rc = Kinetic::InitApp(pman, argc, argv, &p_mhd_config)) return *rc;

  // Pass nullptr as the MHD context: Kinetic::Initialize then skips the
  // MHD-driven field-fill pipeline and leaves the Hermite grid to be loaded.
  Kinetic::RegisterKineticPackage(pman, nullptr);
  // GenerateParticlePoint uses the "Deck" package and does not evaluate fields
  // at seed time (fields are loaded later, in UserWorkBeforeLoop).
  pman.app_input->ProblemGenerator = GenerateParticlePoint;

  // Load the pre-interpolated field grid before the push loop.
  pman.app_input->UserWorkBeforeLoop = [=](Mesh * pm, ParameterInput * pin, SimTime const & tm) {
    Kinetic::LoadFieldsFromInput(pm, pin);
  };

  pman.ParthenonInitPackagesAndMesh();

  Kinetic::ProfileDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());
  driver.Execute();

  pman.ParthenonFinalize();
  return 0;
}
