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
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <globals.hpp>
#include <iostream>
#include <format>

#include "MFEMDriver.h"

namespace MFEMHeat {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("MFEM_Heat");
  const std::string block = "MFEM_Heat";
  pkg->AddParam<int>("nx", pin->GetOrAddInteger(block, "nx", 16));
  pkg->AddParam<int>("ny", pin->GetOrAddInteger(block, "ny", 16));
  pkg->AddParam<int>("serial_refinements",
                     pin->GetOrAddInteger(block, "serial_refinements", 0));
  pkg->AddParam<int>("parallel_refinements",
                     pin->GetOrAddInteger(block, "parallel_refinements", 0));
  pkg->AddParam<int>("order", pin->GetOrAddInteger(block, "order", 2));
  pkg->AddParam<Real>("kappa", pin->GetOrAddReal(block, "kappa", 1.0));
  pkg->AddParam<Real>("solver_rtol", pin->GetOrAddReal(block, "solver_rtol", 1e-12));
  pkg->AddParam<std::string>("output_dir",
                             pin->GetOrAddString(block, "output_dir", "mfem_heat"));
  return pkg;
}

HeatParams GetParams(StateDescriptor *pkg) {
  HeatParams p;
  p.nx = pkg->Param<int>("nx");
  p.ny = pkg->Param<int>("ny");
  p.serial_refinements = pkg->Param<int>("serial_refinements");
  p.parallel_refinements = pkg->Param<int>("parallel_refinements");
  p.order = pkg->Param<int>("order");
  p.kappa = pkg->Param<Real>("kappa");
  p.solver_rtol = pkg->Param<Real>("solver_rtol");
  p.output_dir = pkg->Param<std::string>("output_dir");
  return p;
}

} // namespace MFEMHeat

static TaskStatus HeatStep(HeatSolver *solver, Real dt) {
  solver->Step(dt);
  return TaskStatus::complete;
}

MFEMDriver::MFEMDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pmesh,
    std::shared_ptr<HeatSolver> solver)
      : EvolutionDriver(pin, app_in, pmesh), solver(std::move(solver)) {
}

TaskCollection MFEMDriver::MakeTaskCollection(BlockList_t &blocks, SimTime tm) {
  TaskCollection tc;
  TaskID none(0);

  auto *tl = &tc.AddRegion(1)[0];
  tl->AddTask(none, HeatStep, solver.get(), tm.dt);

  return tc;
}

void MFEMDriver::PreExecute() {
  EvolutionDriver::PreExecute();
  if (Globals::my_rank == 0)
    std::cout << std::format("MFEM heat: {} global dofs\n", solver->GlobalDofs());
}

void MFEMDriver::PostExecute(parthenon::DriverStatus st) {
  const double err = solver->L2Error();
  if (Globals::my_rank == 0)
    std::cout << std::format("MFEM heat: t = {:.6e}, L2 error vs exact = {:.6e}\n",
                             solver->Time(), err);
  EvolutionDriver::PostExecute(st);
}

TaskListStatus MFEMDriver::Step() {
  PARTHENON_INSTRUMENT
  using DriverUtils::ConstructAndExecuteTaskLists;
  TaskListStatus status = ConstructAndExecuteTaskLists<>(this, tm);
  return status;
}
