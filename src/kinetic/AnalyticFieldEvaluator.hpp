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

// Exact analytic field with the same eval() interface as FieldEvaluator, for
// testing the guiding-centre push without interpolation.
//
// Large-aspect-ratio equilibrium with safety-factor-like q(R,Z) =
// q0 + q2 ((R - R_a)^2 + Z^2) (as in AnalyticField.hpp), scaled by B0:
//
//   B_R   = -B0 Z / (q R),   B_phi = B0 R_a / R,   B_Z = B0 (R - R_a) / (q R)
//   E_phi = E_0 R_a / R      (static)
//
// B_pol = grad(psi) x grad(phi) with  psi = B0 ln(q / q0) / (2 q2),  so
//   R B_R = -dpsi/dZ,  R B_Z = dpsi/dR,  and psi = 0 on the axis (R_a, 0), the
// gauge the code uses. J = curl B is computed exactly. The magnetic axis is
// (R_a, 0); sign(B_phi) = sign(B0).
#pragma once

#include <Kokkos_Core.hpp>
#include <hFlux/hFlux.hpp>

#include "kinetic/FieldEvaluator.hpp"

namespace Kinetic {

struct AnalyticFieldEvaluator {
  Real q0, q2, R_a, B0, E_0;

  KOKKOS_INLINE_FUNCTION Real q(Real R, Real Z) const {
    return q0 + q2 * ((R - R_a) * (R - R_a) + Z * Z);
  }

  /// Poloidal flux, zero on the magnetic axis.
  KOKKOS_INLINE_FUNCTION Real psi(Real R, Real Z) const {
    return B0 * Kokkos::log(q(R, Z) / q0) / (2.0 * q2);
  }

  KOKKOS_INLINE_FUNCTION void eval(EvalGCE &out, const Real R, const Real Z,
                                   const Real /*t*/) const {
    const Real qq = q(R, Z);
    const Real qR = 2.0 * q2 * (R - R_a);   // dq/dR
    const Real qZ = 2.0 * q2 * Z;           // dq/dZ

    out.B[0] = -B0 * Z / (qq * R);
    out.B[1] = B0 * R_a / R;
    out.B[2] = B0 * (R - R_a) / (qq * R);

    // d/dR and d/dZ of each component.
    out.dBdR[0] = B0 * Z * (qq + R * qR) / (qq * qq * R * R);
    out.dBdR[1] = -B0 * R_a / (R * R);
    out.dBdR[2] = B0 * (R_a * qq - R * (R - R_a) * qR) / (qq * qq * R * R);

    out.dBdZ[0] = -B0 * (qq - Z * qZ) / (qq * qq * R);
    out.dBdZ[1] = 0.0;
    out.dBdZ[2] = -B0 * (R - R_a) * qZ / (qq * qq * R);

    // J = curl B in (R, phi, Z), axisymmetric.
    out.J[0] = -out.dBdZ[1];
    out.J[1] = out.dBdZ[0] - out.dBdR[2];
    out.J[2] = out.dBdR[1] + out.B[1] / R;

    out.E[0] = 0.0;
    out.E[1] = E_0 * R_a / R;
    out.E[2] = 0.0;

    out.dbdt[0] = out.dbdt[1] = out.dbdt[2] = 0.0;

    out.Bsq = out.B[0] * out.B[0] + out.B[1] * out.B[1] + out.B[2] * out.B[2];
    out.Bmag = Kokkos::sqrt(out.Bsq);
  }

  KOKKOS_INLINE_FUNCTION void eval(EvalBE &out, const Real R, const Real Z,
                                   const Real t) const {
    EvalGCE g;
    eval(g, R, Z, t);
    out.B = g.B; out.E = g.E; out.Bsq = g.Bsq; out.Bmag = g.Bmag;
  }
};

} // namespace Kinetic

namespace Kinetic {

/// Test path for `single_particle`, SingleParticle/field_source=analytic_grid:
/// sample AnalyticFieldEvaluator onto the field grid with the storage
/// conventions of the MHD data (R*B, J, E per cell centre; time derivatives 0),
/// then run the production interpolation: Hermite fit with divergence cleaning
/// (InterpolateHermiteBJE), psi reconstruction (computeFlux) and the magnetic-
/// axis shift (RefreshFluxAndAxis). With SingleParticle/grid_J=curlB the
/// Hermite J coefficients are then replaced by the curl of the Hermite B.
void FillGridFromAnalytic(parthenon::Mesh *pm);

/// Test helper: refit the Hermite J from the curl of the Hermite B (see
/// AnalyticFieldGrid.cpp). Used with SingleParticle/grid_J=curlB.
void ReplaceJByCurlOfHermiteB(FieldData_t data);

} // namespace Kinetic
