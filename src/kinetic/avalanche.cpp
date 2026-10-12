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


int main(int argc, char *argv[]) {
  User * p_mhd_config;

  ParthenonManager pman;
  if (auto rc = Kinetic::InitApp(pman, argc, argv, &p_mhd_config)) return *rc;

  Kinetic::RegisterKineticPackage(pman, p_mhd_config);
  pman.app_input->ProblemGenerator = GenerateParticlePoint;

  pman.app_input->UserMeshWorkBeforeOutput = [=](Mesh * pm, ParameterInput * pin, SimTime const & tm) {
    Kinetic::PlotFieldsTime(pm, pin, tm, p_mhd_config);
  };

  pman.app_input->UserWorkBeforeLoop = [=](Mesh * pm, ParameterInput * pin, SimTime const & tm) {
    Kinetic::WorkBeforeLoop(pm, p_mhd_config);
    Kinetic::HijackEField(pm);
  };

  Kinetic::AttachDiagnostics(pman);
  pman.ParthenonInitPackagesAndMesh();

  Kinetic::AvalancheDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());
  driver.Execute();
  Kinetic::FinishDiagnostics();
  pman.ParthenonFinalize();
  return 0;
}
