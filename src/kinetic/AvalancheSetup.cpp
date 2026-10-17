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

// Set-up of the non-MHD field sources of `avalanche` (analytic, file): seeding
// on a flux surface, the pitch-sign check and the avalanche time series.
//
// Normalised flux, psi_n = psi / psi_edge, with psi = 0 on the magnetic axis:
//   analytic  psi_edge = psi at minor radius 1 (q = q0 + q2), so for the
//             reference ring of radius r_0, psi_n = ln(1 + q2 r_0^2 / q0) /
//             ln(1 + q2 / q0)
//   file      psi_edge = psi on the separatrix: psi of the plasma cell
//             (region 1) next to a scrape-off-layer cell (region 2) that is
//             closest to zero in |psi|

#include <cmath>
#include <utility>
#include <limits>
#include <stdexcept>

#include <parthenon/package.hpp>
#include <globals.hpp>
#include <hFlux/Interpolator.hpp>

#include "kinetic/AvalancheDriver.h"
#include "kinetic/AnalyticFieldEvaluator.hpp"
#include "kinetic/ConfigurationDomainGeometry.hpp"
#include "kinetic/DiagnosticsFile.hpp"
#include "kinetic/FieldEvaluator.hpp"
#include "kinetic/Log.hpp"
#include "kinetic/kinetic.hpp"

namespace Kinetic {

AvalancheOptions AvalancheOptions::FromInput(ParameterInput *pin) {
  AvalancheOptions o;
  o.field_source = pin->GetOrAddString("Avalanche", "field_source", "mhd");
  o.seed = pin->GetOrAddString("Avalanche", "seed", "point");
  if (o.field_source != "mhd" && o.field_source != "analytic" && o.field_source != "file")
    throw std::invalid_argument("Avalanche/field_source must be mhd, analytic or file");
  if (o.seed != "point" && o.seed != "ring" && o.seed != "surface")
    throw std::invalid_argument("Avalanche/seed must be point, ring or surface");
  if (o.seed == "surface" && o.field_source == "mhd")
    throw std::invalid_argument("Avalanche/seed=surface needs field_source analytic or file");
  if (o.seed == "ring" && o.field_source == "file")
    throw std::invalid_argument("Avalanche/seed=ring with field_source=file: the axis is "
                                "known only after the fields are loaded; use seed=surface");
  return o;
}

void AddAvalancheParams(ParameterInput *pin, const AvalancheOptions &opt,
                        std::shared_ptr<StateDescriptor> &pkg) {
  // The Deck E_0 is read from AnalyticField/E_0 as an integer (KineticParams,
  // kept for bit-exactness). These modes take the real value Avalanche/E_0, as
  // the fig11a reference runs used it; Parthenon does not allow reading one
  // parameter as two types.
  const Real E_0 = pin->GetReal("Avalanche", "E_0");
  pkg->AddParam("E_0_exact", E_0);
  pkg->AddParam("seed_r0", pin->GetOrAddReal("ParticleSeed", "r_0", 0.0));
  pkg->AddParam("seed_psi_n", pin->GetOrAddReal("Avalanche", "psi_n", -1.0));
  // Avalanche/E_axis: fixed (default) puts E_phi = E_0 at R = 3 (HijackEField);
  // axis puts it at the magnetic axis of the loaded field (file source).
  const std::string e_axis = pin->GetOrAddString("Avalanche", "E_axis", "fixed");
  if (e_axis != "fixed" && e_axis != "axis")
    throw std::invalid_argument("Avalanche/E_axis must be fixed or axis");
  if (e_axis == "axis") pkg->AddParam("E_R_axis", 1);
  // The analytic q profile also maps the ring radius r_0 to psi_n for the
  // file field source, so that one deck table drives both.
  const auto af = ReadAnalyticField(pin, E_0, 3.0, 1.0);
  pkg->AddParam("q0_ref", af.q0);
  pkg->AddParam("q2_ref", af.q2);
  if (opt.field_source == "analytic") {
    // fig11a: B_phi = +R_a / R, axis (3, 0).
    pkg->AddParam("analytic_field", af);
    pkg->UpdateParam("Rc", af.R_a);
    pkg->UpdateParam("Zc", Real(0.0));
  }
  Log::Info("Avalanche: field_source = {}, seed = {}, E_0 = {:g}", opt.field_source,
            opt.seed, E_0);
}

namespace {

// psi on the separatrix of a field grid: over plasma cells (region 1) with a
// scrape-off-layer neighbour (region 2), the psi closest to zero in magnitude.
Real SeparatrixPsi(const std::shared_ptr<StateDescriptor> &pkg) {
  auto data = pkg->Param<FieldData_t>("FieldData");
  auto cdg = pkg->Param<ConfigurationDomainGeometry>("CDG");
  data.psi_data.sync_host();
  auto ind = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cdg.indicator_view);
  const auto loc = cdg.indicator_locator;
  Evaluator ev{data.hermite_locator};
  auto psi_h = data.psi_data.view_host();
  Real best = std::numeric_limits<Real>::max(), psi_edge = 0.0;
  for (int i = 1; i + 1 < loc.nR; ++i)
    for (int j = 1; j + 1 < loc.nZ; ++j) {
      if (ind(i, j) != 1) continue;
      if (ind(i - 1, j) != 2 && ind(i + 1, j) != 2 && ind(i, j - 1) != 2 && ind(i, j + 1) != 2)
        continue;
      Real psi = 0.0;
      ev.evalPsi(psi, loc.R0 + i * loc.dR, loc.Z0 + j * loc.dZ, psi_h);
      if (std::abs(psi) < best) { best = std::abs(psi); psi_edge = psi; }
    }
  return psi_edge;
}

// Moves every alive particle along the ray from (Rc, Zc) through its seed
// position to the point where psi_n = target. On the ray, psi_n rises from 0 on
// the axis; the bracket ends where the ray leaves the plasma (or at s_max).
template <class PsiFn, class InPlasma>
void MoveToSurface(Mesh *pm, const PsiFn &psi_n, const InPlasma &in_plasma, Real Rc,
                   Real Zc, Real target, Real s_max) {
  auto md = pm->mesh_data.Get();
  auto desc = parthenon::MakeSwarmPackDescriptor<Kinetic::R, Kinetic::Z>("particles");
  auto pack = desc.GetPack(md.get());
  Kokkos::parallel_for(PARTHENON_AUTO_LABEL, pack.GetMaxFlatIndex() + 1,
      KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = pack.GetBlockParticleIndices(idx);
        if (!pack.GetContext(b).IsActive(n)) return;
        const Real dR = pack(b, Kinetic::R(), n) - Rc;
        const Real dZ = pack(b, Kinetic::Z(), n) - Zc;
        Real th = (dR == 0.0 && dZ == 0.0) ? 0.0 : Kokkos::atan2(dZ, dR);
        const Real c = Kokkos::cos(th), s = Kokkos::sin(th);
        // Bracket: the last in-plasma point of a coarse march along the ray.
        Real hi = 0.0;
        const int nmarch = 400;
        for (int k = 1; k <= nmarch; ++k) {
          const Real r = s_max * k / nmarch;
          if (!in_plasma(Rc + r * c, Zc + r * s) || psi_n(Rc + r * c, Zc + r * s) > target) {
            hi = r;
            break;
          }
          hi = r;
        }
        Real lo = 0.0;
        for (int it = 0; it < 80; ++it) {
          const Real mid = 0.5 * (lo + hi);
          if (in_plasma(Rc + mid * c, Zc + mid * s) && psi_n(Rc + mid * c, Zc + mid * s) < target)
            lo = mid;
          else
            hi = mid;
        }
        const Real r = 0.5 * (lo + hi);
        pack(b, Kinetic::R(), n) = Rc + r * c;
        pack(b, Kinetic::Z(), n) = Zc + r * s;
      });
}

// Smallest and largest psi_n over the seeded particles (check of MoveToSurface).
template <class PsiFn>
std::pair<Real, Real> SeedPsiRange(Mesh *pm, const PsiFn &psi_n) {
  auto md = pm->mesh_data.Get();
  auto desc = parthenon::MakeSwarmPackDescriptor<Kinetic::R, Kinetic::Z>("particles");
  auto pack = desc.GetPack(md.get());
  Real lo = 0.0, hi = 0.0;
  Kokkos::parallel_reduce(PARTHENON_AUTO_LABEL, pack.GetMaxFlatIndex() + 1,
      KOKKOS_LAMBDA(const int idx, Real &mn, Real &mx) {
        auto [b, n] = pack.GetBlockParticleIndices(idx);
        if (!pack.GetContext(b).IsActive(n)) return;
        const Real v = psi_n(pack(b, Kinetic::R(), n), pack(b, Kinetic::Z(), n));
        mn = Kokkos::min(mn, v);
        mx = Kokkos::max(mx, v);
      },
      Kokkos::Min<Real>(lo), Kokkos::Max<Real>(hi));
  return {lo, hi};
}

// Sign of B_phi * E_phi at a point, and the pitch range check: particles run
// away against E, so their pitch xi (along b) must have the sign of
// -B_phi * E_phi.
void CheckPitchSign(const std::shared_ptr<StateDescriptor> &pkg, Real Bphi, Real Ephi) {
  const Real ximin = pkg->Param<Real>("ximin"), ximax = pkg->Param<Real>("ximax");
  const Real want = -Bphi * Ephi;
  Log::Info("Avalanche: at the axis B_phi = {:g}, E_phi = {:g}; seed xi in [{:g}, {:g}]", Bphi,
            Ephi, ximin, ximax);
  if (want == 0.0) return;
  if ((want > 0.0 && ximin < 0.0) || (want < 0.0 && ximax > 0.0))
    throw std::invalid_argument(
        "Avalanche: the seed pitch range has the wrong sign for this field: runaways "
        "need sign(xi) = -sign(B_phi E_phi). Set ParticleSeed/ximin, ximax accordingly.");
}

} // namespace

void PrepareAvalancheSeeds(Mesh *pm, const AvalancheOptions &opt) {
  auto pkg = pm->packages.Get("Deck");
  const Real Rc = pkg->Param<Real>("Rc"), Zc = pkg->Param<Real>("Zc");
  const Real r0 = pkg->Param<Real>("seed_r0");
  const Real psi_n_in = pkg->Param<Real>("seed_psi_n");

  if (opt.field_source == "analytic") {
    const auto af = pkg->Param<AnalyticFieldEvaluator>("analytic_field");
    const Real psi_edge = af.psi(af.R_a + 1.0, 0.0);
    pkg->AddParam("psi_edge", psi_edge);
    EvalGCE ev;
    af.eval(ev, Rc, Zc, 0.0);
    CheckPitchSign(pkg, ev.B[1], ev.E[1]);
    const Real psi_n_ring = af.psi(af.R_a + r0, 0.0) / psi_edge;
    Log::Info("Avalanche: analytic psi_edge = {:g}; ring r_0 = {:g} is psi_n = {:g}", psi_edge,
              r0, psi_n_ring);
    if (opt.seed == "surface" && psi_n_in > 0.0) {
      const Real target = psi_n_in;
      auto psi_n = KOKKOS_LAMBDA(Real R, Real Z) { return af.psi(R, Z) / psi_edge; };
      auto inside = KOKKOS_LAMBDA(Real, Real) { return true; };
      MoveToSurface(pm, psi_n, inside, Rc, Zc, target, 1.0);
    }
  } else {  // file
    const Real psi_edge = SeparatrixPsi(pkg);
    pkg->AddParam("psi_edge", psi_edge);
    auto data = pkg->Param<FieldData_t>("FieldData");
    auto cdg = pkg->Param<ConfigurationDomainGeometry>("CDG");
    data.hermite_data.sync_host();
    FieldEvaluator f_h{cdg.hermite_locator, data.hermite_data.view_host()};
    EvalGCE ev;
    f_h.eval(ev, Rc, Zc, 0.0);
    CheckPitchSign(pkg, ev.B[1], ev.E[1]);
    // Target: Avalanche/psi_n, or the psi_n of the analytic ring of radius r_0
    // (AnalyticField/q0, q2) so that one deck table drives both field sources.
    const Real q0 = pkg->Param<Real>("q0_ref"), q2 = pkg->Param<Real>("q2_ref");
    const Real target = psi_n_in >= 0.0
                            ? psi_n_in
                            : std::log1p(q2 * r0 * r0 / q0) / std::log1p(q2 / q0);
    Log::Info("Avalanche: separatrix psi_edge = {:g}; seeding on psi_n = {:g}", psi_edge, target);
    if (target > 0.0) {
      Evaluator pev{data.hermite_locator};
      auto psi_v = data.psi_data.view_device();
      auto psi_n = KOKKOS_LAMBDA(Real R, Real Z) {
        Real psi = 0.0;
        pev.evalPsi(psi, R, Z, psi_v);
        return psi / psi_edge;
      };
      auto inside = KOKKOS_LAMBDA(Real R, Real Z) {
        const auto &l = cdg.indicator_locator;
        if (R < l.R0 || R >= l.R0 + l.nR * l.dR || Z < l.Z0 || Z >= l.Z0 + l.nZ * l.dZ)
          return false;
        int i, j;
        Region region;
        cdg.locate_region(R, Z, i, j, region);
        return region == 1;
      };
      MoveToSurface(pm, psi_n, inside, Rc, Zc, target, 2.0);
      const auto [lo, hi] = SeedPsiRange(pm, psi_n);
      Log::Info("Avalanche: seeds on psi_n in [{:.6f}, {:.6f}] (target {:.6f})", lo, hi, target);
    } else {
      // psi_n = 0: all particles on the magnetic axis.
      auto md = pm->mesh_data.Get();
      auto desc = parthenon::MakeSwarmPackDescriptor<Kinetic::R, Kinetic::Z>("particles");
      auto pack = desc.GetPack(md.get());
      Kokkos::parallel_for(PARTHENON_AUTO_LABEL, pack.GetMaxFlatIndex() + 1,
          KOKKOS_LAMBDA(const int idx) {
            auto [b, n] = pack.GetBlockParticleIndices(idx);
            pack(b, Kinetic::R(), n) = Rc;
            pack(b, Kinetic::Z(), n) = Zc;
          });
    }
  }

  if (auto *diag = DiagnosticsFile::Get())
    diag->DeclareSeries("avalanche", {"t", "N_alive", "W_alive"}, {"s", "1", "1"});
}

void AppendAvalancheRow(Mesh *pm, SimTime const &tm) {
  auto md = pm->mesh_data.Get();
  auto desc_r = parthenon::MakeSwarmPackDescriptor<Kinetic::weight>("particles");
  auto desc_i = parthenon::MakeSwarmPackDescriptor<Kinetic::status>("particles");
  auto pack_r = desc_r.GetPack(md.get());
  auto pack_i = desc_i.GetPack(md.get());
  Real n_alive = 0.0, w_alive = 0.0;
  Kokkos::parallel_reduce(PARTHENON_AUTO_LABEL, pack_r.GetMaxFlatIndex() + 1,
      KOKKOS_LAMBDA(const int idx, Real &n, Real &w) {
        auto [b, k] = pack_r.GetBlockParticleIndices(idx);
        const auto swarm_d = pack_r.GetContext(b);
        if (swarm_d.IsActive(k) && !swarm_d.IsMarkedForRemoval(k) &&
            (pack_i(b, Kinetic::status(), k) & Kinetic::ALIVE)) {
          n += 1.0;
          w += pack_r(b, Kinetic::weight(), k);
        }
      },
      n_alive, w_alive);
  Real v[2] = {n_alive, w_alive};
  MPI_Allreduce(MPI_IN_PLACE, v, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (auto *diag = DiagnosticsFile::Get()) diag->Append("avalanche", {tm.time, v[0], v[1]});
}

} // namespace Kinetic
