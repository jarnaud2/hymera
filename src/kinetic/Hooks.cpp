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

// Parthenon application hooks (before loop / output / restart) and one-off setup tasks.

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

#include <parthenon/driver.hpp>
#include "kinetic/kinetic.hpp"
#include "kinetic/KineticParams.hpp"
#include "kinetic/Log.hpp"
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

void WorkBeforeOutput(Mesh * pm, ParameterInput * pin, SimTime const & tm, User* mhd_context) {
  PlotFieldsTime(pm, pin, tm, mhd_context);
  PlotGCETime(pm, pin, tm, mhd_context);
  PlotCurrents(pm, pin, tm, mhd_context);
}


void WorkBeforeRestartOutput(Mesh * pm, ParameterInput * pin, OutputParameters * op, User* mhd_context) {
  auto signal = SignalHandler::CheckSignalFlags();
  std::string ext = pin->GetOrAddString("MHD_Config", "file_extention", "dat");

  std::string filename = std::format("{}.{}.",
      op -> file_basename,
      op -> file_id);

  if (signal == SignalHandler::OutputSignal::now) {
    filename.append("now");
  } else if (signal == SignalHandler::OutputSignal::final && op -> file_label_final) {
    filename.append("final");
    // default time based data dump
  } else {
    filename.append(std::format("{:0>{}}", op->file_number, op -> file_number_width));
  }

  filename.append(".r");
  filename.append(ext);

  MHD_CHECK(mhd_savesolution(mhd_context, filename.c_str()));

  auto pkg = pm->packages.Get("Deck");
  pkg->UpdateParam("mhd_restart_filename", filename);
}

void WorkBeforeLoop(Mesh * pm, User* mhd_context) {
  if (Globals::is_restart) {
    Log::Info("Restart: reading MHD state from {}", pm->packages.Get("Deck")->Param<std::string>("mhd_restart_filename"));
    auto pkg = pm->packages.Get("Deck");
    auto filename = pkg -> Param<std::string>("mhd_restart_filename");

    MHD_CHECK(mhd_loadsolution(mhd_context, filename.c_str()));
  }
}

TaskStatus HijackEField(Mesh * pm) {
  auto pkg = pm->packages.Get("Deck");

  auto data = pkg->Param<FieldData_t>("FieldData");
  auto cdg  = pkg->Param<ConfigurationDomainGeometry>("CDG");

  // E_0_exact: the real-valued Avalanche/E_0 of the avalanche field sources;
  // otherwise the Deck E_0 (AnalyticField/E_0, read as an integer).
  const Real E_0 = pkg->AllParams().hasKey("E_0_exact") ? pkg->Param<Real>("E_0_exact")
                                                        : pkg->Param<Real>("E_0");
  // E_phi = E_0 R_a / R with R_a = 3, or, if the Deck param E_R_axis is set
  // (avalanche, Avalanche/E_axis = axis), R_a = the magnetic axis Rc, so that
  // E_phi = E_0 on the axis of the loaded field.
  const Real R_a = pkg->AllParams().hasKey("E_R_axis") ? pkg->Param<Real>("Rc") : 3.0;

  const int NR = cdg.indicator_locator.nR;
  const int NZ = cdg.indicator_locator.nZ;

  auto data_d = data.data.view_device();
  data.data.sync_device();
  Kokkos::parallel_for("Set intial runaway current",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {NR, NZ}),
      KOKKOS_LAMBDA(const int i, const int j) {

        Real R = cdg.indicator_locator.R0 + i * cdg.indicator_locator.dR;

        data_d(i,j,FieldComponents::E + 0) = 0.0;
        data_d(i,j,FieldComponents::E + 1) = E_0 * R_a  / R;
        data_d(i,j,FieldComponents::E + 2) = 0.0;
        data_d(i,j,FieldComponents::Et + 0) = 0.0;
        data_d(i,j,FieldComponents::Et + 1) = 0.0;
        data_d(i,j,FieldComponents::Et + 2) = 0.0;
      });
  data.data.modify_device();

  InterpolateHermiteE(data, FieldComponents::E);
  InterpolateHermiteE(data, FieldComponents::Et);

  return TaskStatus::complete;
}

void ComputeParticleWeights(Mesh* pm) {

  auto md = pm->mesh_data.Get();
  auto pkg = pm->packages.Get("Deck");
  if (pkg->Param<int>("ComputeInitialWeights") == 0) return;
  auto data = pkg->Param<FieldData_t>("FieldData");
  auto cdg  = pkg->Param<ConfigurationDomainGeometry>("CDG");
  FieldEvaluator f{cdg.hermite_locator, data.hermite_data.view_device()};

  const Real p_RE = pkg->Param<Real>("p_RE");
  const Real seed_current = pkg->Param<Real>("seed_current");

  auto desc_swarm_r = parthenon::MakeSwarmPackDescriptor<
      Kinetic::p, Kinetic::xi, Kinetic::R, Kinetic::phi, Kinetic::Z, Kinetic::weight>(
      "particles");
  auto desc_swarm_i = parthenon::MakeSwarmPackDescriptor<
      Kinetic::status>(
      "particles");

  Real I_re = 0.0;

  auto pack_swarm_r = desc_swarm_r.GetPack(md.get());
  auto pack_swarm_i = desc_swarm_i.GetPack(md.get());


  Kokkos::parallel_reduce(
      PARTHENON_AUTO_LABEL, pack_swarm_r.GetMaxFlatIndex() + 1,
      // loop over all particles
      KOKKOS_LAMBDA(const int idx, Real &weight) {
        // block and particle indices
        auto [b, n] = pack_swarm_r.GetBlockParticleIndices(idx);
        const auto swarm_d = pack_swarm_r.GetContext(b);
        if (swarm_d.IsActive(n) && (pack_swarm_i(b, Kinetic::status(),n) & Kinetic::ALIVE)) {
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

  MPI_Allreduce(MPI_IN_PLACE,&I_re,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  Real w = seed_current / I_re;

  Log::Info("Initial particle weights: runaway current of unit-weight particles {:.8E}, weight {:.8E}", I_re, w);
  parthenon::par_for(DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL,
                     DevExecSpace(), 0, pack_swarm_r.GetMaxFlatIndex(),
                     // new_n ranges from 0 to N_new_particles
                     KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = pack_swarm_r.GetBlockParticleIndices(idx);
        // block and particle indices
        pack_swarm_r(b, Kinetic::weight(), n) = w;
      });

  pkg->UpdateParam("ComputeInitialWeights", 0);
}

} // namespace Kinetic
