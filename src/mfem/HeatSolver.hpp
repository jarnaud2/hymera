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
#include <memory>
#include <string>
#include <mpi.h>

// Heat equation du/dt = kappa * Laplace(u) on the unit square, u = 0 on the boundary,
// discretized with MFEM H1 elements and backward Euler. The initial condition
// u0 = sin(pi x) sin(pi y) is an eigenmode, so the exact solution is
// u0 * exp(-2 pi^2 kappa t), which L2Error() compares against.
//
// MFEM headers stay in HeatSolver.cpp: mfem::Mesh and parthenon::Mesh would otherwise
// collide in translation units that use both namespaces.

struct HeatParams {
  int nx = 16;
  int ny = 16;
  int serial_refinements = 0;
  int parallel_refinements = 0;
  int order = 2;
  double kappa = 1.0;
  double solver_rtol = 1e-12;
  std::string output_dir = "mfem_heat";
};

class HeatSolver {
public:
  HeatSolver(MPI_Comm comm, const HeatParams &params);
  ~HeatSolver();

  void Step(double dt);
  double Time() const;
  double L2Error() const;
  long long GlobalDofs() const;
  void WriteOutput(int cycle);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
