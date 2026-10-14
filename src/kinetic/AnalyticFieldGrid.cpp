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

// Analytic field put through the production field pipeline (test path only).

#include <parthenon/package.hpp>
#include <hFlux/Interpolator.hpp>

#include "kinetic/kinetic.hpp"
#include "kinetic/AnalyticFieldEvaluator.hpp"
#include "kinetic/ConfigurationDomainGeometry.hpp"
#include "kinetic/Log.hpp"
#include "tasks/Tasks.h"

namespace Kinetic {

// Replaces the Hermite J coefficients by a fit of the curl of the Hermite B,
// sampled at the FD cell centres, so that J and B come from the same field.
// Cells whose centre lies outside the Hermite grid keep their J data.
void ReplaceJByCurlOfHermiteB(FieldData_t data) {
  const auto loc = data.fd_locator;
  const int NR = data.data.view_device().extent_int(0);
  const int NZ = data.data.view_device().extent_int(1);
  auto d = data.data.view_device();
  data.hermite_data.sync_device();
  data.data.sync_device();
  FieldEvaluator f{data.hermite_locator, data.hermite_data.view_device()};
  const auto hl = data.hermite_locator;
  Kokkos::parallel_for("CurlOfHermiteB",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NR, NZ}),
      KOKKOS_LAMBDA(const int i, const int j) {
        const Real R = loc.R0 + i * loc.dR;
        const Real Z = loc.Z0 + j * loc.dZ;
        if (R < hl.R0 || R >= hl.R1 || Z < hl.Z0 || Z >= hl.Z1) return;
        EvalGCE ev;
        f.eval(ev, R, Z, 0.0);
        d(i, j, FieldComponents::J + 0) = -ev.dBdZ[1];
        d(i, j, FieldComponents::J + 1) = ev.dBdZ[0] - ev.dBdR[2];
        d(i, j, FieldComponents::J + 2) = ev.dBdR[1] + ev.B[1] / R;
      });
  data.data.modify_device();
  Interpolator<FIELD_SMOOTHNESS, FIELD_FD_STENSIL> itrp;
  data.data.sync_device();
  itrp.interpolateRange<3>(data.fd_locator, data.hermite_locator, data.data.view_device(),
                           data.hermite_data.view_device(), FieldComponents::J);
  data.hermite_data.modify_device();
}

void FillGridFromAnalytic(parthenon::Mesh *pm) {
  auto pkg = pm->packages.Get("Deck");
  auto data = pkg->Param<FieldData_t>("FieldData");
  const auto af = pkg->Param<AnalyticFieldEvaluator>("sp_analytic_field");
  const bool J_from_curlB = pkg->Param<int>("sp_grid_J_curlB") == 1;

  // Sample at the FD cell centres, in the storage convention of the MHD data:
  // R*B (the evaluator divides by R), plain J and E, zero time derivatives.
  const auto loc = data.fd_locator;
  const int NR = data.data.view_device().extent_int(0);
  const int NZ = data.data.view_device().extent_int(1);
  auto d = data.data.view_device();
  Kokkos::parallel_for("FillGridFromAnalytic",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NR, NZ}),
      KOKKOS_LAMBDA(const int i, const int j) {
        const Real R = loc.R0 + i * loc.dR;
        const Real Z = loc.Z0 + j * loc.dZ;
        EvalGCE ev;
        af.eval(ev, R, Z, 0.0);
        for (int c = 0; c < 3; ++c) {
          d(i, j, FieldComponents::B + c) = R * ev.B[c];
          d(i, j, FieldComponents::J + c) = ev.J[c];
          d(i, j, FieldComponents::E + c) = ev.E[c];
          d(i, j, FieldComponents::Bt + c) = 0.0;
          d(i, j, FieldComponents::Jt + c) = 0.0;
          d(i, j, FieldComponents::Et + c) = 0.0;
        }
      });
  data.data.modify_device();

  // Production pipeline: Hermite fit of B, J, E with divergence cleaning of B,
  // for the field and its time derivative.
  InterpolateHermiteBJE(data, FieldComponents::B);
  InterpolateHermiteBJE(data, FieldComponents::Bt);

  if (J_from_curlB) ReplaceJByCurlOfHermiteB(data);

  // psi reconstruction and the shift to zero on the magnetic axis, as before
  // every output in production. The axis search starts from the analytic axis.
  pkg->UpdateParam("Rc", af.R_a);
  pkg->UpdateParam("Zc", Real(0.0));
  RefreshFluxAndAxis(pm);

  Log::Info("Field from AnalyticFieldEvaluator sampled onto the grid and interpolated "
            "(J from {})", J_from_curlB ? "curl of interpolated B" : "sampled data");
}

} // namespace Kinetic
