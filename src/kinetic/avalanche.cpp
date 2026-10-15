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
#include <format>
#include <fstream>
#include <sstream>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <parthenon_manager.hpp>

#include <Kokkos_Core.hpp>
#include "AvalancheDriver.h"
#include "pgen.hpp"
#include "kinetic.hpp"
#include "AppSetup.hpp"

using namespace parthenon;
using namespace parthenon::driver::prelude;

// Field source, Avalanche/field_source:
//   mhd (default)  the MHD solver's initial state, E replaced by HijackEField
//   analytic       the exact AnalyticFieldEvaluator, no grid and no MHD solver
//                  (the fig11a benchmark, tests/avalanche)
//   file           a static field grid from Simulation/load_fields (fields.h5),
//                  E replaced by HijackEField, no MHD solver
// Seeding, Avalanche/seed:
//   point (default)  GenerateParticlePoint
//   ring             circle of radius ParticleSeed/r_0 around the magnetic axis
//   surface          the ring, then moved along each ray to the flux surface
//                    psi_n = Avalanche/psi_n (see AvalancheSetup.cpp)
int main(int argc, char *argv[]) {
  User * p_mhd_config;

  ParthenonManager pman;
  if (auto rc = Kinetic::InitApp(pman, argc, argv, &p_mhd_config)) return *rc;

  const Kinetic::AvalancheOptions opt = Kinetic::AvalancheOptions::FromInput(pman.pinput.get());
  User *mhd = opt.field_source == "mhd" ? p_mhd_config : nullptr;

  if (opt.field_source == "mhd") {
    Kinetic::RegisterKineticPackage(pman, mhd);
  } else {
    pman.app_input->ProcessPackages = [opt](std::unique_ptr<ParameterInput> &pin) {
      Packages_t packages;
      auto pkg = Kinetic::Initialize(pin.get(), nullptr);
      Kinetic::AddAvalancheParams(pin.get(), opt, pkg);
      packages.Add(pkg);
      return packages;
    };
  }
  pman.app_input->ProblemGenerator =
      opt.seed == "point" ? GenerateParticlePoint : GenerateParticleRing;

  pman.app_input->UserMeshWorkBeforeOutput = [=](Mesh * pm, ParameterInput * pin, SimTime const & tm) {
    if (opt.field_source == "mhd") Kinetic::PlotFieldsTime(pm, pin, tm, mhd);
    if (opt.field_source != "mhd") Kinetic::AppendAvalancheRow(pm, tm);
  };

  pman.app_input->UserWorkBeforeLoop = [=](Mesh * pm, ParameterInput * pin, SimTime const & tm) {
    if (opt.field_source == "mhd") {
      Kinetic::WorkBeforeLoop(pm, mhd);
      Kinetic::HijackEField(pm);
    } else if (opt.field_source == "file") {
      Kinetic::LoadFieldsFromInput(pm, pin);
      Kinetic::HijackEField(pm);
      Kinetic::RefreshFluxAndAxis(pm);
    }
    if (opt.field_source != "mhd") Kinetic::PrepareAvalancheSeeds(pm, opt);
  };

  Kinetic::AttachDiagnostics(pman);
  pman.ParthenonInitPackagesAndMesh();

  Kinetic::AvalancheDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());
  driver.Execute();
  Kinetic::FinishDiagnostics();
  pman.ParthenonFinalize();
  return 0;
}
