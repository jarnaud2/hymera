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
//
// `single_particle` executable: advance ONE guiding-center particle with one
// collisionless push + one small-angle collision kick per parthenon timestep,
// tracking the conserved quantities (p_phi, mu). No swarm, no MHD -- the field
// grid is loaded from an HDF5 dump (fields.h5) exactly as the conservation and
// profile drivers do. The particle is a plain host variable seeded from the
// <SingleParticle> input block. A text file (one row per parthenon output
// cadence) records t, p, xi, R, phi, Z, p_phi, mu.

#include <iostream>
#include <fstream>
#include <format>
#include <memory>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <parthenon_manager.hpp>
#include <globals.hpp>

#include <Kokkos_Core.hpp>
#include "SingleParticleDriver.h"
#include "kinetic.hpp"
#include "AppSetup.hpp"
#include "AnalyticFieldEvaluator.hpp"

using namespace parthenon;
using namespace parthenon::driver::prelude;

int main(int argc, char *argv[]) {

  ParthenonManager pman;
  if (auto rc = Kinetic::InitApp(pman, argc, argv)) return *rc;

  // Shared particle state, owned here so the driver and output hook see the
  // same object. Seeded from the <SingleParticle> input block.
  auto state = std::make_shared<Kinetic::SingleParticleState>();

  // The Deck package, plus the single-particle field source:
  //   SingleParticle/field_source = grid (default) | analytic
  // "analytic" pushes in the exact AnalyticFieldEvaluator (no interpolation);
  // its parameters are AnalyticField/q0, q2, R_a, B0 and E_0.
  pman.app_input->ProcessPackages = [](std::unique_ptr<ParameterInput> &pin) {
    Packages_t packages;
    auto pkg = Kinetic::Initialize(pin.get(), nullptr);
    const std::string src = pin->GetOrAddString("SingleParticle", "field_source", "grid");
    // "analytic_grid": sample the analytic field onto the FD grid and run the
    // production interpolation (Hermite fit, divergence cleaning, psi
    // reconstruction); SingleParticle/grid_J = data (default, as in production)
    // or curlB (J recomputed as the curl of the interpolated B).
    pkg->AddParam("sp_analytic", src == "analytic" ? 1 : 0);
    pkg->AddParam("sp_analytic_grid", src == "analytic_grid" ? 1 : 0);
    // SingleParticle/grid_J: data (default) | curlB (the stored J coefficients
    // are refit from the curl of the interpolated B; affects only the MHD-J
    // users, since the GCE always uses the pointwise curl of B).
    const std::string gridJ = pin->GetOrAddString("SingleParticle", "grid_J", "data");
    pkg->AddParam("sp_grid_J_curlB", gridJ == "curlB" ? 1 : 0);
    pkg->AddParam("sp_analytic_field",
                  Kinetic::ReadAnalyticField(pin.get(), pkg->Param<Real>("E_0"), 3.1, -1.0));
    packages.Add(pkg);
    return packages;
  };

  // Particle generation does nothing: the single particle is a host variable,
  // not a swarm particle.
  pman.app_input->ProblemGenerator = [](MeshBlock *pmb, ParameterInput *pin) {};

  const std::string output_file =
      pman.pinput->GetOrAddString("SingleParticle", "output_file", "single_particle.dat");

  pman.app_input->UserWorkBeforeLoop =
      [=](Mesh *pm, ParameterInput *pin, SimTime const &tm) {
        // Seed the particle state from input.
        state->X[0] = pin->GetReal("SingleParticle", "p");
        state->X[1] = pin->GetReal("SingleParticle", "xi");
        state->X[2] = pin->GetReal("SingleParticle", "R");
        state->X[3] = pin->GetOrAddReal("SingleParticle", "phi", 0.0);
        state->X[4] = pin->GetReal("SingleParticle", "Z");
        state->t = 0.0;

        // Load the pre-interpolated field grid (no MHD); not used with the
        // analytic field source.
        auto deck = pm->packages.Get("Deck");
        if (deck->Param<int>("sp_analytic_grid") == 1) {
          Kinetic::FillGridFromAnalytic(pm);
        } else if (deck->Param<int>("sp_analytic") == 0) {
          Kinetic::LoadFieldsFromInput(pm, pin);
          Kinetic::HijackEField(pm);
          if (deck->Param<int>("sp_grid_J_curlB") == 1)
            Kinetic::ReplaceJByCurlOfHermiteB(deck->Param<Kinetic::FieldData_t>("FieldData"));
        }

        // Populate the initial conserved quantities and start the output file.
        Kinetic::ComputeConservedForState(pm, state);
        if (Globals::my_rank == 0) {
          std::ofstream ofs(output_file, std::ios::trunc);
          ofs << "# t p xi R phi Z p_phi mu" << std::endl;
        }
      };

  // Append one row per parthenon output trigger (cycle 0, each output dt, final).
  pman.app_input->UserMeshWorkBeforeOutput =
      [=](Mesh *pm, ParameterInput *pin, SimTime const &tm) {
        if (Globals::my_rank != 0) return;
        std::ofstream ofs(output_file, std::ios::app);
        ofs << std::format(
                   "{:20.14e} {:20.14e} {:20.14e} {:20.14e} {:20.14e} {:20.14e} "
                   "{:20.14e} {:20.14e}",
                   state->t, state->X[0], state->X[1], state->X[2], state->X[3],
                   state->X[4], state->p_phi, state->mu)
            << std::endl;
      };

  Kinetic::AttachDiagnostics(pman);
  pman.ParthenonInitPackagesAndMesh();

  Kinetic::SingleParticleDriver driver(pman.pinput.get(), pman.app_input.get(),
                                       pman.pmesh.get(), state);
  driver.Execute();
  Kinetic::FinishDiagnostics();
  pman.ParthenonFinalize();
  return 0;
}
