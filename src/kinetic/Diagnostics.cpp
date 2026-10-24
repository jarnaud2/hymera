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

#include <map>
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
#include "kinetic/DiagnosticsFile.hpp"
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

  Log::Debug("Magnetic axis (R, Z) = ({:g}, {:g}), Psi_min = {:g}", Rc, Zc, Psi_min);

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

  // The scale factors are the ones this diagnostic has always printed with.
  const Real I_re_particles = I_re * .5;
  const Real I_re_grid = I_re_integral * 5.3 * 2.0;
  const Real I_ohmic = I_ohmic_hermite * 5.3  * 2.0 / pc::mu0;
  Log::Debug("Currents: particles {:20.14e}, deposited {:20.14e}, ohmic {:20.14e}",
             I_re_particles, I_re_grid, I_ohmic);
  if (auto *diag = DiagnosticsFile::Get()) {
    diag->DeclareSeries("currents", {"t", "I_re_particles", "I_re_grid", "I_ohmic"},
                        {"s", "", "", ""});
    diag->Append("currents", {tm.time, I_re_particles, I_re_grid, I_ohmic});
  }
}

// In-step snapshot hook. A driver adds it to its task list where the state in
// the middle of a time step is wanted, e.g.
//
//   dep = tl->AddTask(dep, MakeOutputs, pmesh, tm.ncycle, iCD, iLA, t0 + dtLA);
//
// Every Output/substep_snapshots-th call (0: never, the default) writes the
// alive particles of all ranks to /snapshots/c<cycle>_cd<iCD>_la<iLA>/ in the
// diagnostics file, with the time within the step [tau_c] and the cycle as
// attributes. It reads the particles and changes nothing, and Parthenon's own
// outputs and their numbering are untouched.
TaskStatus MakeOutputs(Mesh* pmesh, int cycle, int iCD, int iLA, Real t_in_step) {
  auto pkg = pmesh->packages.Get("Deck");
  const int every = pkg->Param<int>("substep_snapshots");
  if (every <= 0) return TaskStatus::complete;
  static long long calls = 0;
  if (calls++ % every != 0) return TaskStatus::complete;

  auto md = pmesh->mesh_data.Get();
  auto desc_r = parthenon::MakeSwarmPackDescriptor<
      Kinetic::p, Kinetic::xi, Kinetic::R, Kinetic::phi, Kinetic::Z, Kinetic::weight>(
      "particles");
  auto desc_i = parthenon::MakeSwarmPackDescriptor<Kinetic::status>("particles");
  auto pack_r = desc_r.GetPack(md.get());
  auto pack_i = desc_i.GetPack(md.get());

  // Copy the alive particles of this rank into a flat device array, then to host.
  constexpr int NCOL = 7;  // p, xi, R, phi, Z, weight, status
  const int nmax = pack_r.GetMaxFlatIndex() + 1;
  Kokkos::View<Real **> buf("snapshot_buf", nmax, NCOL);
  Kokkos::View<int> count("snapshot_count");
  Kokkos::parallel_for(PARTHENON_AUTO_LABEL, nmax, KOKKOS_LAMBDA(const int idx) {
    auto [b, n] = pack_r.GetBlockParticleIndices(idx);
    const auto swarm_d = pack_r.GetContext(b);
    if (!swarm_d.IsActive(n) || swarm_d.IsMarkedForRemoval(n) ||
        !(pack_i(b, Kinetic::status(), n) & Kinetic::ALIVE))
      return;
    const int k = Kokkos::atomic_fetch_add(&count(), 1);
    buf(k, 0) = pack_r(b, Kinetic::p(), n);
    buf(k, 1) = pack_r(b, Kinetic::xi(), n);
    buf(k, 2) = pack_r(b, Kinetic::R(), n);
    buf(k, 3) = pack_r(b, Kinetic::phi(), n);
    buf(k, 4) = pack_r(b, Kinetic::Z(), n);
    buf(k, 5) = pack_r(b, Kinetic::weight(), n);
    buf(k, 6) = pack_i(b, Kinetic::status(), n);
  });
  auto buf_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), buf);
  int nlocal = 0;
  Kokkos::deep_copy(nlocal, count);

  // Column-major local block, so each column is contiguous for the gather.
  std::vector<double> local(static_cast<size_t>(nlocal) * NCOL);
  for (int c = 0; c < NCOL; ++c)
    for (int k = 0; k < nlocal; ++k) local[c * nlocal + k] = buf_h(k, c);

  int nranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  std::vector<int> counts(nranks), displs(nranks);
  MPI_Gather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  int ntotal = 0;
  for (int r = 0; r < nranks; ++r) { displs[r] = ntotal; ntotal += counts[r]; }

  static const char *names[NCOL] = {"p", "xi", "R", "phi", "Z", "weight", "status"};
  std::map<std::string, std::vector<double>> columns;
  for (int c = 0; c < NCOL; ++c) {
    std::vector<double> all(Globals::my_rank == 0 ? ntotal : 0);
    MPI_Gatherv(local.data() + static_cast<size_t>(c) * nlocal, nlocal, MPI_DOUBLE,
                all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (Globals::my_rank == 0) columns[names[c]] = std::move(all);
  }

  if (auto *diag = DiagnosticsFile::Get()) {
    const std::string label = std::format("c{:06d}_cd{:03d}_la{:05d}", cycle, iCD, iLA);
    diag->WriteSnapshot(label, columns,
                        {{"cycle", double(cycle)}, {"iCD", double(iCD)}, {"iLA", double(iLA)},
                         {"t_in_step_tau_c", t_in_step}, {"n_particles", double(ntotal)}});
    Log::Debug("Snapshot {}: {} particles", label, ntotal);
  }
  return TaskStatus::complete;
}

} // namespace Kinetic

namespace Kinetic {

// Radial profile through the magnetic axis, Z = Zc, written at each output to
// /profiles/midplane_c<cycle> of the diagnostics file (hybrid, with
// Hybrid/midplane_profile = 1). In the units of the kinetic E (E_c):
//   E_phi, E_par    from the interpolated field
//   E_ohm = En eta_norm J_phi, with J the MHD current (the resistive part of
//                   Ohm's law as ComputeBaseElectricField builds it)
//   E_re  = En J_re,phi, with J_re the runaway current of the last MHD step
//                   as handed to the MHD solve (Jre_last, nearest grid row),
//                   the part subtracted in ComputeAdjustedElectricField
// and J_phi, J_re,phi in their own normalisations.
void WriteMidplaneProfile(Mesh *pm, SimTime const &tm) {
  auto *diag = DiagnosticsFile::Get();
  auto pkg = pm->packages.Get("Deck");
  auto data = pkg->Param<FieldData_t>("FieldData");
  auto cdg = pkg->Param<ConfigurationDomainGeometry>("CDG");
  // This MHD step's runaway current (Jre_last, kept before the corrector zeroes
  // Jre_mhd); hybrid registers it with Hybrid/midplane_profile = 1.
  auto jre = pkg->Param<View3>("Jre_last");
  const Real Zc = pkg->Param<Real>("Zc");
  const Real En = pkg->Param<Real>("En");
  const Real eta_norm = pkg->Param<Real>("eta_norm");
  const auto hl = data.hermite_locator;
  const auto il = cdg.indicator_locator;
  constexpr int N = 400;
  constexpr int NC = 7;
  const Real R0 = hl.R0 + 1e-10;
  const Real dR = (hl.nR * hl.dR - 2e-10) / (N - 1);

  data.hermite_data.sync_device();
  FieldEvaluator f{cdg.hermite_locator, data.hermite_data.view_device()};
  int jz = static_cast<int>(Kokkos::round((Zc - il.Z0) / il.dZ));
  jz = jz < 0 ? 0 : (jz >= il.nZ ? il.nZ - 1 : jz);

  Kokkos::View<Real **> buf("midplane", N, NC);
  Kokkos::parallel_for("MidplaneProfile", N, KOKKOS_LAMBDA(const int k) {
    const Real R = R0 + k * dR;
    EvalGCE ev;
    f.eval(ev, R, Zc, 0.0);
    int ir = static_cast<int>(Kokkos::round((R - il.R0) / il.dR));
    ir = ir < 0 ? 0 : (ir >= il.nR ? il.nR - 1 : ir);
    const Real Epar = (ev.E[0] * ev.B[0] + ev.E[1] * ev.B[1] + ev.E[2] * ev.B[2]) / ev.Bmag;
    buf(k, 0) = R;
    buf(k, 1) = ev.E[1];
    buf(k, 2) = Epar;
    buf(k, 3) = En * eta_norm * ev.J[1];
    buf(k, 4) = En * jre(ir, jz, 1);
    buf(k, 5) = ev.J[1];
    buf(k, 6) = jre(ir, jz, 1);
  });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), buf);
  if (diag == nullptr) return;
  std::vector<std::vector<double>> cols(NC, std::vector<double>(N));
  for (int k = 0; k < N; ++k)
    for (int c = 0; c < NC; ++c) cols[c][k] = h(k, c);
  diag->WriteProfile(std::format("midplane_c{:06d}", tm.ncycle),
                     {"R", "E_phi", "E_par", "E_ohm", "E_re", "J_phi", "J_re_phi"},
                     {"a", "E_c", "E_c", "E_c", "E_c", "J_0", "eta J"}, cols);
  diag->Flush();
}

// The runaway current of the last MHD step as handed to the MHD solve
// (Jre_last: the time average over the step of the current-deposition averages,
// eta J in units of B_0 V_A), on the MHD cell grid, written at each output to
// /profiles/jre_c<cycle> (hybrid, with Hybrid/jre_map = 1). Flattened with R
// fastest: columns R, Z (cell centres, a) and J_R, J_phi, J_Z.
void WriteJreMap(Mesh *pm, SimTime const &tm) {
  auto *diag = DiagnosticsFile::Get();
  auto pkg = pm->packages.Get("Deck");
  auto cdg = pkg->Param<ConfigurationDomainGeometry>("CDG");
  auto jre = pkg->Param<View3>("Jre_last");
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), jre);
  if (diag == nullptr) return;
  const auto il = cdg.indicator_locator;
  const int nR = h.extent_int(0), nZ = h.extent_int(1);
  std::vector<std::vector<double>> cols(5, std::vector<double>(static_cast<size_t>(nR) * nZ));
  for (int j = 0; j < nZ; ++j)
    for (int i = 0; i < nR; ++i) {
      const size_t k = static_cast<size_t>(j) * nR + i;
      cols[0][k] = il.R0 + i * il.dR;
      cols[1][k] = il.Z0 + j * il.dZ;
      for (int c = 0; c < 3; ++c) cols[2 + c][k] = h(i, j, c);
    }
  diag->WriteProfile(std::format("jre_c{:06d}", tm.ncycle),
                     {"R", "Z", "J_R", "J_phi", "J_Z"},
                     {"a", "a", "eta J (B_0 V_A)", "eta J (B_0 V_A)", "eta J (B_0 V_A)"}, cols);
}

} // namespace Kinetic
