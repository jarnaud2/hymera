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

#include <cmath>
#include <globals.hpp>

#include "kinetic/KineticParams.hpp"
#include "kinetic/kinetic.hpp"
#include "tasks/Tasks.h"
#include "util/common.hpp"

using parthenon::constants::SI;
using parthenon::constants::PhysicalConstants;
using pc = PhysicalConstants<SI>;

namespace Kinetic {

KineticParams KineticParams::FromInput(ParameterInput *pin) {
  /// Physical constants
  static constexpr Real eps0 = pc::eps0; ///< Vacuum permittivity [F / M]
  static constexpr Real c = pc::c;       ///< Speed of light [m/s]
  static constexpr Real mi = pc::amu;    ///< Ion mass = atomic mass unit [kg]
  static constexpr Real mu0 = pc::mu0;   ///< Vacuum permeabulity [N / A^2]
  static constexpr Real me = pc::me;     ///< electron mass [kg]
  static constexpr Real e = pc::qe;      ///< electron charge [C]

  KineticParams p;

  /// Time discretization parameters
  p.dt_mhd     = pin->GetReal("parthenon/time","dt_force");
  p.final_time = pin->GetReal("parthenon/time","tlim");

  p.nCorrectorSteps = pin->GetOrAddInteger("Time","nCorrectorSteps", 0);
  p.nCD = pin->GetOrAddInteger("Time","nCD", 100);  ///< current deposit sub-steps per MHD step
  p.dtLA_over_tauC = pin->GetOrAddReal("Time","dtLA_over_tauC", 1e-5);

  p.timeStep = pin->GetOrAddReal("Simulation", "hRK", 1.e-6);    /// Runge kutta time in tau_c [-]
  p.atol = pin->GetOrAddReal("Simulation", "atol", 1.e-6);      /// Absoulte tolerance for RK [-]
  p.rtol = pin->GetOrAddReal("Simulation", "rtol", 1.e-5);       /// Realative toleratnce for RK[ [-]
  p.integrator = parseIntegrator(pin->GetOrAddString("Simulation", "integrator", "dopri5")); /// ODE integrator: dopri5 (adaptive) or rk4 (fixed hRK)
  p.filePath = pin->GetOrAddString("Simulation", "file_path", "current.out");

  /// Reference parameters
  p.B0 = pin->GetOrAddReal("Reference", "B0", 5.3);  ///< On-axis magnetic field [T]
  p.a  = pin->GetOrAddReal("Reference", "a", 2.0);    ///< Minor radius [m] and reference length
  p.R0 = pin->GetOrAddReal("Reference", "R0", 6.0);    ///< Major radius [m]
  p.nD0 = pin->GetOrAddReal("Reference", "nD0", 1e20);    ///< Deutirium density [m^-3]
  p.Te0 = pin->GetOrAddReal("Reference", "Te0", 2.0198);  ///< Electron temperature [eV], and plasma temperature single-temperature model

  const Real B0 = p.B0, a = p.a, nD0 = p.nD0, Te0 = p.Te0;

  /// Derived parameters
  p.VA   = pin->GetOrAddReal("Derived", "VA",  B0 / sqrt(mi * nD0 * mu0)); ///< Alfven velocity [m/s]
  p.tauA = pin->GetOrAddReal("Derived", "tauA", a / p.VA);                  ///< Alfven time     [s]
  p.E0   = pin->GetOrAddReal("Derived", "E0", B0 * p.VA);                   ///< Reference electric field in MHD [V/m]
  p.J0   = pin->GetOrAddReal("Derived", "J0", B0 / (mu0 * a));            ///< Reference current density [A/m^2]
  p.eta0 = pin->GetOrAddReal("Derived", "eta0", a * p.VA * mu0);            ///< Reference resitivity [Ohm*m]
  p.eta  = pin->GetOrAddReal("Derived", "eta",  1.0);                     ///< Resitivity scale     [-]
  p.Re   = pin->GetOrAddReal("Derived", "Re",  200.0);                    ///< Reinolds Number

  ///< Plasma composition parameters
  p.vTe = sqrt(2.0*Te0*e / me); ///< Thermal velocity
  p.Z0 = pin->GetOrAddReal("Plasma", "Z0", 10.0); ///< Atomic number of impurity (Z)
  p.ZI = pin->GetOrAddReal("Plasma", "ZI", 1.0);  ///< Charge of impurity
  p.fI = pin->GetOrAddReal("Plasma", "fI", 1.0);  ///<Fraction of impurity density, normalized to deuterium denstiy (nD0)
  const Real ZI = p.ZI, fI = p.fI;
  p.nI = fI*nD0; ///< Impurity density [m^-3]
  const Real nI = p.nI;
  p.n_e0 = nD0 + ZI*nI; ///< Free electron density [m^-3]
  const Real n_e0 = p.n_e0;
  p.Zeff = pin->GetOrAddReal("Plasma", "Zeff", (ZI*ZI*nI + nD0)/n_e0); // TODO: why ZI is a square?
  p.NeI = p.Z0 - ZI; ///< Number of bound electrons
  p.Coulog0 = pin->GetOrAddReal("Plasma", "Coulog0", 14.9 - 0.5*log(n_e0/1.0e20) + log(Te0/1.e3));
  p.Rc = pin->GetOrAddReal("Plasma", "Rc", 3.1158966549999998e+00); ///< Initial guess for magnetic axis, R [-], length normalized
  p.Zc = pin->GetOrAddReal("Plasma", "Zc", 3.7114360000000002e-01); ///< Initial guess for magnetic axis, Z [-], length normalized
  const Real Zeff = p.Zeff, Coulog0 = p.Coulog0;

  p.L11 = 0.58 * 32.0 / (3.0 * M_PI);
  p.sigmapar = 12.0 * pow(M_PI, 1.5) / sqrt(2.0) * pow(Te0 * e, 1.5) * pow(eps0, 2) / (Zeff * pow(e, 2) * sqrt(me) * Coulog0) * p.L11;
  p.etaplasma = pin->GetOrAddReal("Plasma", "etaplasma", 1.0 / p.sigmapar);

  ///< Geometry parameters
  p.etawall               = pin->GetOrAddReal("Geometry", "etawall", 4.4e-2);                     ///< Wall resistivity [Ohm*m]
  p.etawallperp            = pin->GetOrAddReal("Geometry", "etawallperp", p.etawall);
  p.etawallphi             = pin->GetOrAddReal("Geometry", "etawallphi", p.etawall);
  p.etawallphi_isol_cell   = pin->GetOrAddReal("Geometry", "etawallphi_isol_cell", p.etawall);
  p.etasepwal              = pin->GetOrAddReal("Geometry", "etasepwal",  p.etaplasma);
  p.etaVV                  = pin->GetOrAddReal("Geometry", "etaVV",  1.30288e-6);
  p.etaout                 = pin->GetOrAddReal("Geometry", "etaout",  1.30288e-3);

  p.Rmin                   = pin->GetOrAddReal("Geometry", "rmin",  1.525); ///< Minimum R [-]
  p.Rmax                   = pin->GetOrAddReal("Geometry", "rmax",  4.975); ///< Maximum R [-]
  p.Zmin                   = pin->GetOrAddReal("Geometry", "zmin",  -2.975);///<  Minimum Z [-]
  p.Zmax                   = pin->GetOrAddReal("Geometry", "zmax",   2.975); ///< Maximum Z [-]

  p.gamma_BC  = pin->GetOrAddReal("BoundaryConditions", "gamma_BC", 1. + 1e-4);   // Assuming E max = 1000
  p.gamma_min = pin->GetOrAddReal("BoundaryConditions", "gamma_min", 2. * (p.gamma_BC - 1.) + 1.0);
  p.p_BC      = momentum_(p.gamma_BC);
  p.p_RE      = p.p_BC;  // Updated from maximum electric field during the run.

  ///< Runaway parameters
  p.c_vTe = pin->GetOrAddReal("Collisions", "c_vTe", c / p.vTe); ///< Guiding center equations coefficient [-]
  p.NSA = pin->GetOrAddInteger("Collisions", "NSA", 150);       ///< Number of small angle collisions     [-]
  p.k = pin->GetOrAddReal("Collisions", "k", 5.0);
  p.aI            = pin->GetOrAddReal("Collisions", "aI", 0.3285296762792767);  ///<
  p.FineStructure = 1. / 137.035999;  // Fine Structure constant
  p.II            = pin->GetOrAddReal("Collisions", "II", 235.2 / pc::eV / pc::me / pc::c / pc::c); // Mean exitation energy
  p.PSCoefDnRA = 1.0;
  if (PartialScreening)
    p.PSCoefDnRA    = 1.0 + p.NeI * fI / (1.0 + ZI * fI);

  ///< Numerical paremters
  p.dampV                  = pin->GetOrAddReal("Numerical", "dampV", 0.01); ///< Stabilization coefficeint for velocity gradient
  p.itime                  = pin->GetOrAddReal("Numerical", "itime", 0.0); ///< Initial time for mhd counters [sec]
  p.NR                      = pin->GetOrAddInteger("Numerical", "NR", 100);
  p.Nphi                    = pin->GetOrAddInteger("Numerical", "Nphi", 2);
  p.NZ                      = pin->GetOrAddInteger("Numerical", "NZ", 200);

  p.dR = (p.Rmax - p.Rmin) / (Real) p.NR;
  p.dZ = (p.Zmax - p.Zmin) / (Real) p.NZ;

  p.RminCellCenter = p.Rmin + .5 * p.dR;
  p.RmaxCellCenter = p.Rmax - .5 * p.dR;
  p.ZminCellCenter = p.Zmin + .5 * p.dZ;
  p.ZmaxCellCenter = p.Zmax - .5 * p.dZ;

  p.tau_a = 6*M_PI*eps0*pow(me * c, 3) / pow(e,4) / pow(B0,2);     ///< Syncrotron radiation damping time
  p.tau_c = 4*M_PI*pow(eps0,2)*me*me*c*c*c/(e*e*e*e*n_e0*Coulog0); ///< Relativistic collision time
  p.Ec = me * c / e / p.tau_c;                                       ///< Connor-Hastie Electric field
  p.En = p.E0 / p.Ec;
  p.eta_norm = p.etaplasma / p.eta0; // converts eta * \curl B to V_A B_0
  p.eta_a3VaB0 = p.etaplasma / pow(a,3) / p.E0; // converts eta J to V_A B_0

  // Create plotting mesh for interpolated fields
  p.NR_plot = pin->GetOrAddInteger("Output", "NR_plot", 400);
  p.NZ_plot = pin->GetOrAddInteger("Output", "NZ_plot", 800);
  p.Nt_plot = pin->GetOrAddInteger("Output", "Nt_plot", 10);

  // The max |E_par| search used to sample the field-plot grid; its defaults
  // keep that resolution, so the momentum boundary does not depend on Output/*.
  p.NR_maxE = pin->GetOrAddInteger("Numerical", "NR_maxE", 400);
  p.NZ_maxE = pin->GetOrAddInteger("Numerical", "NZ_maxE", 800);

  // GetOrAddInteger: an E_0 given with a fractional part is truncated. Kept as is.
  p.E_0 = pin->GetOrAddInteger("AnalyticField", "E_0", 40.0);

  return p;
}

void KineticLateParams::ParseSeedPoint(ParameterInput *pin, Real Rc, Real Zc) {
  Rseed = pin->GetOrAddReal("ParticleSeed", "Rseed", Rc);  // Used in pgen if pparticles are generated at single point
  Zseed = pin->GetOrAddReal("ParticleSeed", "Zseed", Zc);
}

void KineticLateParams::ParseParticles(ParameterInput *pin, const KineticParams &p) {
  const Real wce0 = pc::qe * p.B0 / pc::me; // Electron gyrofrequency
  c_aw0 =  pin->GetOrAddReal("GuidingCenterEquations", "c_aw0", pc::c/p.a/wce0);
  ct_a =   pin->GetOrAddReal("GuidingCenterEquations", "ct_a", pc::c * p.tau_c / p.a);
  alpha0 = pin->GetOrAddReal("GuidingCenterEquations", "alpha0", p.tau_c/p.tau_a);

  npart =  pin->GetOrAddInteger("ParticleSeed", "num_particles_per_block", 16);
  num_particles_max = pin->GetOrAddInteger("ParticleSeed", "MaxParticles", 500000);
  rng_seed = pin->GetOrAddInteger("ParticleSeed", "rng_seed", 1234) + parthenon::Globals::my_rank;
  seed_current = pin->GetOrAddReal("ParticleSeed", "current", 15e3); // 15 kAmps

  gammamin = pin->GetOrAddReal("ParticleSeed", "gammamin", 10.0);
  gammamax = pin->GetOrAddReal("ParticleSeed", "gammamax", 20.0);
  ximin = pin->GetOrAddReal("ParticleSeed", "ximin", 0.8);
  ximax = pin->GetOrAddReal("ParticleSeed", "ximax", 1.0);
}

KineticSwitches KineticSwitches::FromInput(ParameterInput *pin) {
  KineticSwitches s;
  s.EnableLargeAngleCollisions = pin->GetOrAddInteger("Simulation", "EnableLargeAngleCollisions", 1);
  s.EnableSmallAngleCollisions = pin->GetOrAddInteger("Simulation", "EnableSmallAngleCollisions", 1);
  s.EnableComputeConservedQuantities = pin->GetOrAddInteger("Simulation", "EnableComputeConservedQuantities", 0);
  // Log file for the weight-averaged conservation diagnostic (p_phi, mu).
  s.conservation_log =
      pin->GetOrAddString("Simulation", "conservation_log", "conservation.dat");
  return s;
}

} // namespace Kinetic
