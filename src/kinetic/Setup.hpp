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

// The stages of Kinetic::Initialize, in the order it calls them. Each stage reads
// its own input-deck parameters; the order of those reads is the order of the
// parameter dump every output carries, so the stages must keep this order.
//
//   SetupMHD          fill the MHD context, initialise the solver, take one step
//   SetupFieldGrid    field grid, geometry, collision models; with MHD, fill the
//                     interpolated field grid and find the magnetic axis
//   SetupParticles    seeding, guiding-centre coefficients, RNG, swarm
//   SetupDiagnostics  switches, diagnostics file, collision profile
#pragma once

#include <memory>
#include <parthenon/package.hpp>

#include "kinetic/kinetic.hpp"
#include "kinetic/KineticParams.hpp"
#include "util/common.hpp"

namespace Kinetic {

using parthenon::ParameterInput;
using parthenon::StateDescriptor;

/// Fills mhd_context from the run parameters and the MHD_Config block, points
/// its runaway current at Jre_mhd, initialises the solver and takes one step so
/// the fields exist before particles are seeded.
void SetupMHD(ParameterInput *pin, const KineticParams &prm, User *mhd_context,
              DualView3 &Jre_mhd);

/// What SetupFieldGrid hands on to the later stages.
struct FieldGridResult {
  Real Rc, Zc;      ///< magnetic axis (the input guess without MHD)
  Real Psi_min;     ///< flux on the axis; 0 without MHD
};

/// Registers the time-step, tolerance and domain parameters, the collision
/// models, the field grid, the geometry and the current views. With MHD, fills
/// the interpolated field grid from the solver and finds the magnetic axis.
FieldGridResult SetupFieldGrid(ParameterInput *pin, const KineticParams &prm,
                               User *mhd_context, DualView3 &Jre_mhd,
                               std::shared_ptr<StateDescriptor> &pkg);

/// Seed point, psi normalisation, field-plot grid, guiding-centre
/// coefficients, particle seeding, RNG and the particle swarm.
void SetupParticles(ParameterInput *pin, const KineticParams &prm, User *mhd_context,
                    const FieldGridResult &axis, std::shared_ptr<StateDescriptor> &pkg);

/// On/off switches, legacy text files, the diagnostics file, the collision
/// profile and the restart-bookkeeping parameters.
void SetupDiagnostics(ParameterInput *pin, const KineticParams &prm,
                      std::shared_ptr<StateDescriptor> &pkg);

} // namespace Kinetic
