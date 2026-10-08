#include <iostream>
#include <format>
#include <fstream>
#include <sstream>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <parthenon_manager.hpp>

#include <Kokkos_Core.hpp>
#include "ConservationDriver.h"
#include "pgen.hpp"
#include "kinetic.hpp"
#include "AppSetup.hpp"

using namespace parthenon;
using namespace parthenon::driver::prelude;


int main(int argc, char *argv[]) {

  ParthenonManager pman;
  if (auto rc = Kinetic::InitApp(pman, argc, argv)) return *rc;

  Kinetic::RegisterKineticPackage(pman, nullptr);
  pman.app_input->ProblemGenerator = GenerateParticlePoint;

  pman.app_input->UserWorkBeforeLoop = [=](Mesh * pm, ParameterInput * pin, SimTime const & tm) {
    Kinetic::LoadFieldsFromInput(pm, pin);
    Kinetic::HijackEField(pm);
  };

  pman.ParthenonInitPackagesAndMesh();

  Kinetic::ConservationDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());
  driver.Execute();
  pman.ParthenonFinalize();
  return 0;
}
