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
#include <numbers>

#include <mfem.hpp>

#include "HeatSolver.hpp"

using std::numbers::pi;

struct HeatSolver::Impl {
  HeatParams params;
  double t = 0.0;

  std::unique_ptr<mfem::ParMesh> pmesh;
  std::unique_ptr<mfem::H1_FECollection> fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> fes;
  mfem::Array<int> ess_tdof_list;

  // Mass matrix for the right-hand side, and A = M + dt K for the implicit solve.
  // A is rebuilt only when dt changes. FormSystemMatrix makes M and A references into
  // the forms' assembled matrices, so the forms must outlive them.
  std::unique_ptr<mfem::ParBilinearForm> m_form;
  std::unique_ptr<mfem::ParBilinearForm> a_form;
  std::unique_ptr<mfem::ConstantCoefficient> dt_kappa;
  mfem::HypreParMatrix M;
  mfem::HypreParMatrix A;
  double A_dt = -1.0;
  std::unique_ptr<mfem::HypreBoomerAMG> amg;
  std::unique_ptr<mfem::CGSolver> cg;

  mfem::ParGridFunction u;
  mfem::Vector U;  // true-dof vector, the evolved state
  std::unique_ptr<mfem::ParaViewDataCollection> dc;

  double Exact(const mfem::Vector &x) const {
    return std::exp(-2.0 * pi * pi * params.kappa * t) *
           std::sin(pi * x(0)) * std::sin(pi * x(1));
  }

  void BuildOperator(double dt) {
    dt_kappa = std::make_unique<mfem::ConstantCoefficient>(dt * params.kappa);
    a_form = std::make_unique<mfem::ParBilinearForm>(fes.get());
    a_form->AddDomainIntegrator(new mfem::MassIntegrator);
    a_form->AddDomainIntegrator(new mfem::DiffusionIntegrator(*dt_kappa));
    a_form->Assemble();
    a_form->FormSystemMatrix(ess_tdof_list, A);

    // Point cg at the new preconditioner before it touches the old (freed) one.
    auto new_amg = std::make_unique<mfem::HypreBoomerAMG>(A);
    new_amg->SetPrintLevel(0);
    cg->SetPreconditioner(*new_amg);
    cg->SetOperator(A);
    amg = std::move(new_amg);
    A_dt = dt;
  }
};

HeatSolver::HeatSolver(MPI_Comm comm, const HeatParams &params)
    : impl_(std::make_unique<Impl>()) {
  Impl &s = *impl_;
  s.params = params;

  // MPI itself is owned by Parthenon; hypre still needs its own init.
  mfem::Hypre::Init();

  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      params.nx, params.ny, mfem::Element::QUADRILATERAL, false, 1.0, 1.0);
  for (int i = 0; i < params.serial_refinements; i++) serial.UniformRefinement();
  s.pmesh = std::make_unique<mfem::ParMesh>(comm, serial);
  for (int i = 0; i < params.parallel_refinements; i++) s.pmesh->UniformRefinement();

  s.fec = std::make_unique<mfem::H1_FECollection>(params.order, s.pmesh->Dimension());
  s.fes = std::make_unique<mfem::ParFiniteElementSpace>(s.pmesh.get(), s.fec.get());

  mfem::Array<int> ess_bdr(s.pmesh->bdr_attributes.Max());
  ess_bdr = 1;
  s.fes->GetEssentialTrueDofs(ess_bdr, s.ess_tdof_list);

  s.m_form = std::make_unique<mfem::ParBilinearForm>(s.fes.get());
  s.m_form->AddDomainIntegrator(new mfem::MassIntegrator);
  s.m_form->Assemble();
  s.m_form->FormSystemMatrix(s.ess_tdof_list, s.M);

  s.cg = std::make_unique<mfem::CGSolver>(comm);
  s.cg->SetRelTol(params.solver_rtol);
  s.cg->SetMaxIter(500);
  s.cg->SetPrintLevel(0);
  s.cg->iterative_mode = true;  // previous state is the initial guess

  s.u.SetSpace(s.fes.get());
  mfem::FunctionCoefficient u0([&s](const mfem::Vector &x) { return s.Exact(x); });
  s.u.ProjectCoefficient(u0);
  s.U.SetSize(s.fes->GetTrueVSize());
  s.u.GetTrueDofs(s.U);

  s.dc = std::make_unique<mfem::ParaViewDataCollection>("heat", s.pmesh.get());
  s.dc->SetPrefixPath(params.output_dir);
  s.dc->SetLevelsOfDetail(params.order);
  s.dc->SetHighOrderOutput(true);
  s.dc->SetDataFormat(mfem::VTKFormat::BINARY);
  s.dc->RegisterField("u", &s.u);
}

HeatSolver::~HeatSolver() {
  impl_.reset();
  mfem::Hypre::Finalize();
}

void HeatSolver::Step(double dt) {
  Impl &s = *impl_;
  if (dt != s.A_dt) s.BuildOperator(dt);

  // (M + dt K) U^{n+1} = M U^n, with homogeneous Dirichlet rows zeroed.
  mfem::Vector B(s.U.Size());
  s.M.Mult(s.U, B);
  B.SetSubVector(s.ess_tdof_list, 0.0);
  s.cg->Mult(B, s.U);
  MFEM_VERIFY(s.cg->GetConverged(), "MFEM heat solve did not converge");

  s.t += dt;
}

double HeatSolver::Time() const { return impl_->t; }

double HeatSolver::L2Error() const {
  Impl &s = *impl_;
  s.u.SetFromTrueDofs(s.U);
  mfem::FunctionCoefficient exact([&s](const mfem::Vector &x) { return s.Exact(x); });
  return s.u.ComputeL2Error(exact);
}

long long HeatSolver::GlobalDofs() const {
  return static_cast<long long>(impl_->fes->GlobalTrueVSize());
}

void HeatSolver::WriteOutput(int cycle) {
  Impl &s = *impl_;
  s.u.SetFromTrueDofs(s.U);
  s.dc->SetCycle(cycle);
  s.dc->SetTime(s.t);
  s.dc->Save();
}
