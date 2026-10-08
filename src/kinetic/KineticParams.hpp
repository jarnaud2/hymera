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

#pragma once

#include <string>
#include <parthenon/package.hpp>

namespace Kinetic {

using parthenon::ParameterInput;
using parthenon::Real;

/// Run parameters of the kinetic model, read from the input deck once, plus the
/// normalisation constants derived from them.
///
/// FromInput reads the deck in the order Initialize always has. That order
/// matters: GetOrAdd* records each default it adds into the parameter dump that
/// every output carries, so reordering the reads changes the output files.
/// Parameters read later, after MHD set-up and the field fill (seed point,
/// guiding-centre coefficients, seeding, switches), are read by
/// ParseLate. The MHD solver's own switches (MHD_Config/*) are read where the
/// MHD context is filled.
struct KineticParams {
  // Time discretisation.
  Real dt_mhd;                  ///< parthenon/time/dt_force, MHD step [s]
  Real final_time;              ///< parthenon/time/tlim [s]
  int nCorrectorSteps;
  int nCD;                      ///< current-deposition sub-steps per MHD step
  Real dtLA_over_tauC;          ///< requested large-angle step, in tau_c

  // Particle push.
  Real timeStep;                ///< Simulation/hRK, Runge-Kutta step in tau_c
  Real atol, rtol;
  int integrator;               ///< IntegratorType
  std::string filePath;

  // Reference scales.
  Real B0, a, R0, nD0, Te0;

  // Derived MHD scales (overridable from the Derived block).
  Real VA, tauA, E0, J0, eta0, eta, Re;

  // Plasma composition.
  Real vTe, Z0, ZI, fI, nI, n_e0, Zeff, NeI, Coulog0;
  Real Rc, Zc;                  ///< initial guess for the magnetic axis
  Real L11, sigmapar, etaplasma;

  // Wall and region resistivities, domain.
  Real etawall, etawallperp, etawallphi, etawallphi_isol_cell, etasepwal, etaVV, etaout;
  Real Rmin, Rmax, Zmin, Zmax;

  // Momentum boundary.
  Real gamma_BC, gamma_min, p_BC, p_RE;

  // Collisions.
  Real c_vTe;
  int NSA;
  Real k, aI, FineStructure, II, PSCoefDnRA;

  // Numerical grid of the MHD solver (independent of the Parthenon mesh).
  Real dampV, itime;
  int NR, Nphi, NZ;
  Real dR, dZ;
  Real RminCellCenter, RmaxCellCenter, ZminCellCenter, ZmaxCellCenter;

  // Normalisation.
  Real tau_a, tau_c, Ec, En, eta_norm, eta_a3VaB0;

  // Field-plot grid.
  int NR_plot, NZ_plot, Nt_plot;

  Real E_0;                     ///< AnalyticField/E_0

  /// Reads every parameter needed before the MHD context is filled.
  static KineticParams FromInput(ParameterInput *pin);
};

/// Parameters Initialize reads after the field fill, in that order.
struct KineticLateParams {
  Real Rseed, Zseed;            ///< single-point seed location (default: magnetic axis)
  Real c_aw0, ct_a, alpha0;     ///< guiding-centre coefficients
  int npart;                    ///< ParticleSeed/num_particles_per_block
  int num_particles_max;
  int rng_seed;                 ///< already offset by the MPI rank
  Real seed_current;            ///< [A], not yet normalised
  Real gammamin, gammamax, ximin, ximax;

  /// Reads the seed point. Rc, Zc are the magnetic axis found by the field fill.
  void ParseSeedPoint(ParameterInput *pin, Real Rc, Real Zc);
  /// Reads the guiding-centre coefficients and the particle seeding.
  void ParseParticles(ParameterInput *pin, const KineticParams &p);
};

/// On/off switches read at the end of Initialize.
struct KineticSwitches {
  int EnableLargeAngleCollisions;
  int EnableSmallAngleCollisions;
  int EnableComputeConservedQuantities;
  std::string conservation_log;

  static KineticSwitches FromInput(ParameterInput *pin);
};

} // namespace Kinetic
