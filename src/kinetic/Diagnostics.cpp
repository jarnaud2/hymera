//========================================================================================(
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

// Output-time diagnostics: field, guiding-centre and current plots, and the in-step MakeOutputs hook.

#include <memory>
#include <numeric>
#include <iostream>
#include <format>
#include <random>
#include <typeinfo>  //for 'typeid' to work
#include <parthenon/package.hpp>
#include <hFlux/dopri.hpp>
#include <hFlux/FieldData.hpp>
#include <hFlux/Evaluator.hpp>

using namespace parthenon;

#include "kinetic/RunawayDriver.h"
#include "kinetic/kinetic.hpp"
#include "kinetic/KineticParams.hpp"
#include "kinetic/GuidingCenterEquations.hpp"
#include "kinetic/LargeAngleCollision.hpp"
#include "kinetic/SmallAngleCollision.hpp"
#include "kinetic/kinetic.hpp"
#include "kinetic/ConfigurationDomainGeometry.hpp"
#include "kinetic/CurrentDensity.hpp"
#include "kinetic/AnalyticField.hpp"
#include "kinetic/FieldEvaluator.hpp"
#include "tasks/Tasks.h"
#include "mhd/mhd.h"
#include "util/H5io.hpp"

using parthenon::constants::SI;
using parthenon::constants::PhysicalConstants;
using pc = PhysicalConstants<SI>;

namespace Kinetic {

// Recomputes the flux psi from the current Hermite field grid, finds the
// magnetic axis again and shifts psi so that it is zero there. The particle push
// reads psi_data (conserved quantities), so this is physics, not diagnostics:
// it runs before every output whether or not the field plot is written.
void RefreshFluxAndAxis(Mesh *pm) {
  auto pkg = pm->packages.Get("Deck");
  auto data = pkg->Param<FieldData_t>("FieldData");

  Interpolator<FIELD_SMOOTHNESS, FIELD_FD_STENSIL> itrp;
  itrp.computeFlux(data.hermite_locator,
      data.hermite_data.view_device(),
      data.psi_data.view_device());
  data.psi_data.modify_device();  // Mark psi data as modified

  int accent = 1;
  Real Psi_min;
  data.hermite_data.sync_host();
  data.psi_data.sync_host();

  auto Rc = pkg->Param<Real>("Rc");
  auto Zc = pkg->Param<Real>("Zc");
  Evaluator psi_ev{data.hermite_locator};
  findMagneticAxis(Rc, Zc,
              data.hermite_data.view_host(),
              data.psi_data.view_host(),
              psi_ev,
              accent,
              Psi_min);

  std::cout << std::format("New Mag. Axis center (R,Z) = {:g}, {:g}\n Psi_min = {:g}", Rc, Zc, Psi_min) << std::endl;

  data.psi_data.sync_device();
  Kokkos::parallel_for("Normalize psi",
  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {data.hermite_locator.nR,data.hermite_locator.nZ}),
  KOKKOS_LAMBDA(int i, int j){
    data.psi_data.view_device()(0,0,i,j) -= Psi_min;
  });
  data.psi_data.modify_device();
}

// Samples the interpolated fields on the Hermite_Field_Plot_* grid, for output.
// With Output/plot_fields=0 the grid does not exist and only the psi refresh runs.
void PlotFieldsTime(Mesh *pm, ParameterInput * pin, SimTime const & tm, User* mhd_context) {
  RefreshFluxAndAxis(pm);

  auto pkg = pm->packages.Get("Deck");
  if (pkg->Param<int>("plot_fields") == 0) return;

  auto hpd_R = pkg->Param<ParArray1D<Real>>("Hermite_Field_Plot_data_R");
  auto hpd_Z = pkg->Param<ParArray1D<Real>>("Hermite_Field_Plot_data_Z");
  auto hpd_t = pkg->Param<ParArray1D<Real>>("Hermite_Field_Plot_data_t");
  auto hpd_F = pkg->Param<ParArrayND<Real>>("Hermite_Field_Plot_data_F");
  auto data = pkg->Param<FieldData_t>("FieldData");
  auto cdg  = pkg->Param<ConfigurationDomainGeometry>("CDG");

  const int NR_plot = hpd_R.size();
  const int NZ_plot = hpd_Z.size();
  const int Nt_plot = hpd_t.size();

  Evaluator psi_ev{data.hermite_locator};
  FieldEvaluator f{cdg.hermite_locator, data.hermite_data.view_device()};


  Kokkos::parallel_for("FillInterpolatedData_plot",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {NR_plot,NZ_plot,Nt_plot}),
      KOKKOS_LAMBDA(const int i, const int j, const int s) {
        Real R = hpd_R(i);
        Real Z = hpd_Z(j);
        Real t = hpd_t(s);
        EvalGCE ev;

        f.eval(ev, R, Z, t);

        for (int k = 0; k < 3; ++k) {
          hpd_F(i,j,s,k,0) = ev.B[k];
          hpd_F(i,j,s,k,1) = ev.J[k];
          hpd_F(i,j,s,k,2) = ev.dBdR[k];
          hpd_F(i,j,s,k,3) = ev.dBdZ[k];
          hpd_F(i,j,s,k,4) = ev.E[k];
          hpd_F(i,j,s,k,5) = ev.dbdt[k];
        }

        Real Psi = 0.0;
        psi_ev.evalPsi(Psi, R, Z, data.psi_data.view_device());
        hpd_F(i,j,s,0,6) = Psi;
        hpd_F(i,j,s,1,6) = ev.Bsq;
        hpd_F(i,j,s,2,6) = ev.Bmag;
      });
}

void PlotGCETime(Mesh *pm, ParameterInput * pin, SimTime const & tm, User* mhd_context) {
  auto pkg = pm->packages.Get("Deck");
  if (pkg->Param<int>("plot_fields") == 0) return;

  auto hpd_R = pkg->Param<ParArray1D<Real>>("Hermite_Field_Plot_data_R");
  auto hpd_Z = pkg->Param<ParArray1D<Real>>("Hermite_Field_Plot_data_Z");
  auto hpd_t = pkg->Param<ParArray1D<Real>>("Hermite_Field_Plot_data_t");

  const auto pmin  = pkg->Param<Real>("pmin");
  const auto pmax  = pkg->Param<Real>("pmax");
  const auto ximin = pkg->Param<Real>("ximin");
  const auto ximax = pkg->Param<Real>("ximax");

  const Real p0 = .5 * (pmax + pmin);
  const Real xi0 = .5 * (ximax + ximin);

//  auto gce_data = pkg->Param<ParArrayND<Real>>("GCE_data");

/*  const auto c_aw0  = pkg->Param<Real>("c_aw0");
  const auto ct_a   = pkg->Param<Real>("ct_a");
  const auto alpha0 = pkg->Param<Real>("alpha0");
  auto data = pkg->Param<FieldData_t>("FieldData");
  auto cdg  = pkg->Param<ConfigurationDomainGeometry>("CDG");
  FieldEvaluator f{cdg.hermite_locator, data.hermite_data.view_device()};
  GuidingCenterEquations<decltype(f), true, false> gce(f, c_aw0, ct_a, alpha0);

  const int NR_plot = hpd_R.size();
  const int NZ_plot = hpd_Z.size();
  const int Nt_plot = hpd_t.size();

  Kokkos::parallel_for("FillInterpolatedData_plot",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {NR_plot,NZ_plot,Nt_plot}),
      KOKKOS_LAMBDA(const int i, const int j, const int s) {
        Real R = hpd_R(i);
        Real Z = hpd_Z(j);
        Real t = hpd_t(s);
        Dim5 X = {p0, xi0, R, 0.0, Z};
        EvalGCE ev;

        Dim5 dX = {};
        gce(0.0, X, dX);
        for (int k = 0; k < 5; ++k) {
          gce_data(i,j,s,k) = dX[k];
        }
      });
      */
}



void PlotCurrents(Mesh * pm, ParameterInput * pin, SimTime const & tm, User* mhd_context) {
  auto pkg = pm->packages.Get("Deck");

  auto data = pkg->Param<FieldData_t>("FieldData");
  auto cdg  = pkg->Param<ConfigurationDomainGeometry>("CDG");
  FieldEvaluator f{cdg.hermite_locator, data.hermite_data.view_device()};

  auto desc_swarm_r = parthenon::MakeSwarmPackDescriptor<
      swarm_position::x, swarm_position::y, swarm_position::z, Kinetic::p,
      Kinetic::xi, Kinetic::R, Kinetic::phi, Kinetic::Z, Kinetic::weight>(
      "particles");
  auto desc_swarm_i =
      parthenon::MakeSwarmPackDescriptor<Kinetic::will_scatter,
                                         Kinetic::secondary_index,
                                         Kinetic::status>("particles");
  auto md = pm->mesh_data.Get();
  auto pack_swarm_r = desc_swarm_r.GetPack(md.get());
  auto pack_swarm_i = desc_swarm_i.GetPack(md.get());

  Real I_re = 0.0;

  const Real p_RE = pkg->Param<Real>("p_RE");
  Kokkos::parallel_reduce(
      PARTHENON_AUTO_LABEL, pack_swarm_r.GetMaxFlatIndex() + 1,
      // loop over all particles
      KOKKOS_LAMBDA(const int idx, Real &weight) {
        // block and particle indices
        auto [b, n] = pack_swarm_r.GetBlockParticleIndices(idx);
        const auto swarm_d = pack_swarm_r.GetContext(b);
        if (swarm_d.IsActive(n) && !swarm_d.IsMarkedForRemoval(n) && (pack_swarm_i(b, Kinetic::status(), n) & Kinetic::ALIVE)) {
          Dim5 X;
          Real t = 0.0;
          X[0] = pack_swarm_r(b, Kinetic::p(), n);
          X[1] = pack_swarm_r(b, Kinetic::xi(), n);
          X[2] = pack_swarm_r(b, Kinetic::R(), n);
          X[3] = pack_swarm_r(b, Kinetic::phi(), n);
          X[4] = pack_swarm_r(b, Kinetic::Z(), n);
          Real w = pack_swarm_r(b, Kinetic::weight(), n);
          if (X[0] > p_RE) {
            weight += getParticleCurrent(X, t, w, f);
          }
        }
      },
      I_re);


  auto Jre_mhd = pkg->Param<DualView3>("Jre_mhd");

  Jre_mhd.sync_device();
  auto Jre_d = Jre_mhd.view_device();

  Real R0 = cdg.indicator_locator.R0;
  Real Z0 = cdg.indicator_locator.Z0;
  Real dR = cdg.indicator_locator.dR;
  Real dZ = cdg.indicator_locator.dZ;
  int nR = cdg.indicator_locator.nR;
  int nZ = cdg.indicator_locator.nZ;

  data.data.sync_device();
  auto data_d = data.data.view_device();

  Real I_ohmic_fd = 0.0;
  Real I_ohmic_hermite = 0.0;
  Real I_re_integral = 0.0;
  Kokkos::parallel_reduce(
      PARTHENON_AUTO_LABEL,
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nR, nZ}),
      // loop over all particles
      KOKKOS_LAMBDA(int i, int j, Real& integral, Real& integral_ohmic_hermite, Real& integral_ohmic_fd) {
        Real R = R0 + i * dR;
        Real Z = Z0 + j * dZ;

        int ii,jj;
        Region region;
        cdg.locate_region(R, Z, ii,jj, region);

        if (region > 0) {

          EvalGCE ev;
          Real t = 0.0;

          f.eval(ev, R, Z, t);

          Real area = dR * dZ;
          integral_ohmic_fd += area * data_d(i, j, FieldComponents::J + 1);
          integral_ohmic_hermite += area * ev.J[1];
          integral += area * Jre_d(i,j,1);
        }

      },
      I_re_integral, I_ohmic_hermite, I_ohmic_fd);

  MPI_Allreduce(MPI_IN_PLACE,&I_re,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  if (Globals::my_rank == 0) {
    std::cout << std::format("{:20.14e} {:20.14e} {:20.14e}",
        I_re * .5, I_re_integral * 5.3 * 2.0,
        I_ohmic_hermite * 5.3  * 2.0 / pc::mu0) << std::endl;
  }
}

TaskStatus MakeOutputs(Outputs* pouts, Mesh* pmesh, ParameterInput* pinput, Real time, int iPR) {
  auto md = pmesh->mesh_data.Get();
  auto pkg = pmesh->packages.Get("Deck");
//  pkg->UpdateParam("MyTime", time);
//  pkg->UpdateParam("PredictorIterationNumber", iPR);
//
//  pouts->MakeOutputs(pmesh, pinput);
  return TaskStatus::complete;
}

} // namespace Kinetic
