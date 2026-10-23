#include "Tasks.h"
#include "ViewManipulation.h"

void AttachFieldPredictor(TaskList* tl, TaskID& dep,
    Mesh* pm,
    User* p_mhd_config, const Real dtField) {

  auto pkg = pm->packages.Get("Deck");

  FieldData_t data = pkg->Param<FieldData_t>("FieldData");
  View3 E_base = pkg->Param<View3>("E_base");
  View3 Jre =  pkg->Param<View3>("Jre");
  const Real eta_norm = pkg->Param<Real>("eta_norm");
  const Real En       = pkg->Param<Real>("En");
  auto Jre_mhd = pkg->Param<DualView3>("Jre_mhd");

  dep = tl->AddTask(dep, CommunicateBJV, data, p_mhd_config, FieldComponents::B);
  dep = tl->AddTask(dep, ComputeBaseElectricField, E_base, data, En, eta_norm, FieldComponents::B);

  dep = tl->AddTask(dep, AdvanceBackgroundFields, p_mhd_config, Jre_mhd);

  dep = tl->AddTask(dep, CommunicateBJV, data, p_mhd_config, FieldComponents::Bt);
  dep = tl->AddTask(dep, ComputeBaseElectricField_in_place, data, En, eta_norm, FieldComponents::Bt);

  dep = tl->AddTask(dep, InterpolateTime, data, dtField);

  dep = tl->AddTask(dep, ComputeAdjustedElectricField, data, E_base, Jre, En);
  dep = tl->AddTask(dep, InterpolateHermiteBJE, data, FieldComponents::B);

  dep = tl->AddTask(dep, InterpolateHermiteBJE, data, FieldComponents::Bt);

  dep = tl->AddTask(dep, ResetBackgroundFields, p_mhd_config);
}

void AttachFieldCorrector(TaskList* tl, TaskID& dep,
    Mesh* pm, User* p_mhd_config) {

  auto pkg = pm->packages.Get("Deck");
  auto Jre_mhd = pkg->Param<DualView3>("Jre_mhd");
  // Jre_mhd holds the sum of the nCD current-deposition averages of this MHD step
  // (AttachCurrentCollector); the MHD solver expects their time average over the
  // step, in the same units (eta J in B_0 V_A).
  const int nCD = pkg->Param<int>("nCD");
  dep = tl->AddTask(dep, [Jre_mhd, nCD]() mutable {
    Jre_mhd.sync_device();
    auto j = Jre_mhd.view_device();
    const Real scl = 1.0 / nCD;
    Kokkos::parallel_for("Average Jre_mhd",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {j.extent_int(0), j.extent_int(1)}),
        KOKKOS_LAMBDA(const int i, const int k) {
          j(i, k, 0) *= scl;
          j(i, k, 1) *= scl;
          j(i, k, 2) *= scl;
        });
    Jre_mhd.modify_device();
    return TaskStatus::complete;
  });
  // Keep this step's runaway current, as passed to MHD, for the output diagnostics
  // (Jre_last, set by hybrid with Hybrid/midplane_profile = 1): the corrector zeroes Jre_mhd.
  if (pkg->AllParams().hasKey("Jre_last")) {
    auto last = pkg->Param<View3>("Jre_last");
    dep = tl->AddTask(dep, [Jre_mhd, last]() mutable {
      Jre_mhd.sync_device();
      Kokkos::deep_copy(last, Jre_mhd.view_device());
      return TaskStatus::complete;
    });
  }
  dep = tl->AddTask(dep, AdvanceBackgroundFields, p_mhd_config, Jre_mhd);
}

void AttachCurrentCollector(TaskList* tl, TaskID& dep,
    Mesh* pm, const Real dtCD) {

  auto pkg = pm->packages.Get("Deck");

  FieldData_t data = pkg->Param<FieldData_t>("FieldData");
  View3 E_base = pkg->Param<View3>("E_base");
  View3 Jre =  pkg->Param<View3>("Jre");
  auto Jre_mhd = pkg->Param<DualView3>("Jre_mhd");
  const Real eta_a3VaB0 = pkg->Param<Real>("eta_a3VaB0");
  const Real En       = pkg->Param<Real>("En");

  dep = tl->AddTask(dep, ScaleView,  Jre, eta_a3VaB0 / dtCD);
  dep = tl->AddTask(dep, ReduceView, Jre);
  dep = tl->AddTask(dep, AccumulateView, Jre_mhd, Jre);
  dep = tl->AddTask(dep, ComputeAdjustedElectricField, data, E_base, Jre, En);
  dep = tl->AddTask(dep, InterpolateHermiteE, data, FieldComponents::E);
  dep = tl->AddTask(dep, ZeroView, Jre);
}
