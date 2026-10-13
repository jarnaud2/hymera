//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#include <iostream>
#include <memory>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <parthenon_manager.hpp>

#include "MFEMDriver.h"
#include "HeatSolver.hpp"

using namespace parthenon;
using namespace parthenon::driver::prelude;


int main(int argc, char *argv[]) {
  ParthenonManager pman;
  auto manager_status = pman.ParthenonInitEnv(argc, argv);

  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  // Created after the mesh exists so it can read its params from the package; shared
  // with the output hook below, which is installed before that.
  auto solver = std::make_shared<std::shared_ptr<HeatSolver>>();
  int output_cycle = 0;

  pman.app_input->ProcessPackages = [=](std::unique_ptr<ParameterInput> &pin) {
    Packages_t packages;
    packages.Add(MFEMHeat::Initialize(pin.get()));
    return packages;
  };

  pman.app_input->ProblemGenerator = [=](MeshBlock *pmb, ParameterInput *pin) {

  };

  pman.app_input->UserMeshWorkBeforeOutput = [=, &output_cycle](Mesh * pm, ParameterInput * pin, SimTime const & tm) {
    if (*solver) (*solver)->WriteOutput(output_cycle++);
  };

  pman.ParthenonInitPackagesAndMesh();

  {
    auto pkg = pman.pmesh->packages.Get("MFEM_Heat");
    *solver = std::make_shared<HeatSolver>(MPI_COMM_WORLD, MFEMHeat::GetParams(pkg.get()));

    MFEMDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get(),
        *solver);
    driver.Execute();
  }

  // Free hypre/MFEM objects before Parthenon finalizes MPI.
  solver->reset();
  pman.ParthenonFinalize();
  return 0;
}
