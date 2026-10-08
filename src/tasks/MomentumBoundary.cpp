#include "Tasks.h"

#include "kinetic/ConfigurationDomainGeometry.hpp"
#include "kinetic/FieldEvaluator.hpp"

TaskStatus UpdateMomentumBoundary(
    Mesh* pm,
    const Real time_0, const Real time_1) {
  auto pkg = pm->packages.Get("Deck");

  auto data = pkg->Param<FieldData_t>("FieldData");
  auto cdg = pkg->Param<ConfigurationDomainGeometry>("CDG");
  FieldEvaluator fev{cdg.hermite_locator, data.hermite_data.view_device()};

  // Uniform NR_maxE x NZ_maxE sample points just inside the Hermite grid. The
  // expressions match the ones that built the field-plot grid this search used
  // to read, so the sampled points are bit-identical at the default sizes.
  const int NR_maxE = pkg->Param<int>("NR_maxE");
  const int NZ_maxE = pkg->Param<int>("NZ_maxE");
  const auto &hl = data.hermite_locator;
  const Real R0_s = hl.R0 + 1e-10;
  const Real Z0_s = hl.Z0 + 1e-10;
  const Real dR_s = (hl.nR * hl.dR - 2e-10) / static_cast<Real>(NR_maxE);
  const Real dZ_s = (hl.nZ * hl.dZ - 2e-10) / static_cast<Real>(NZ_maxE);

  Real maxE = 0.01;
  Kokkos::parallel_reduce(
      "max E",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NR_maxE, NZ_maxE}),
      KOKKOS_LAMBDA(const int i, const int j, Real& Epar) {
        const Real R = R0_s + i * dR_s;
        const Real Z = Z0_s + j * dZ_s;

        EvalBE be;
        fev.eval(be, R, Z, time_0);

        Real value = Kokkos::abs(dot_product(be.B, be.E) / be.Bmag);
        if (value > Epar) {
          Epar = value;
        }

        fev.eval(be, R, Z, time_1);

        value = Kokkos::abs(dot_product(be.B, be.E) / be.Bmag);
        if (value > Epar) {
          Epar = value;
        }
      },
      Kokkos::Max<Real>(maxE));

  const Real p_BC_update = momentum_(1.0 + 0.1 / maxE);

  pkg->UpdateParam("p_BC", p_BC_update);
  pkg->UpdateParam("p_RE", p_BC_update);

  return TaskStatus::complete;
}
