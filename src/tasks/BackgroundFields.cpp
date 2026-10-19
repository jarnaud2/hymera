#include "mhd/mhd.h"
#include "Tasks.h"

TaskStatus AdvanceBackgroundFields(User* p_mhd_config, DualView3 Jre) {

  Jre.sync_host();
  MHD_CHECK(mhd_step(p_mhd_config));
  // Zero both copies: AccumulateView adds this step's current on the device,
  // so a device copy left at the old sum would carry it into the next step (on
  // a GPU; on the CPU host and device are the same memory).
  Kokkos::deep_copy(Jre.view_host(), 0.0);
  Kokkos::deep_copy(Jre.view_device(), 0.0);
  return TaskStatus::complete;
}

TaskStatus ResetBackgroundFields(User *p_mhd_config) {
  MHD_CHECK(mhd_resetState(p_mhd_config));
  return TaskStatus::complete;
}
