//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
//========================================================================================
#include "kinetic/PushTiming.hpp"

#include <mpi.h>

#include "kinetic/DiagnosticsFile.hpp"

namespace Kinetic {

namespace {
double g_seconds = 0.0;
std::int64_t g_launches = 0, g_slots = 0;
bool g_declared = false;
} // namespace

bool PushTiming::enabled = false;

void PushTiming::Add(double seconds, std::int64_t slots) {
  g_seconds += seconds;
  g_launches += 1;
  g_slots += slots;
}

void PushTiming::Flush(double time) {
  if (!enabled) return;
  // Slowest rank's push time; slots summed over ranks.
  double sec = g_seconds;
  double slots = static_cast<double>(g_slots);
  MPI_Allreduce(MPI_IN_PLACE, &sec, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &slots, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (auto *diag = DiagnosticsFile::Get()) {
    if (!g_declared) {
      diag->DeclareSeries("push", {"t", "push_s", "launches", "slots", "ns_per_slot"},
                          {"s", "s", "1", "1", "ns"});
      g_declared = true;
    }
    diag->Append("push", {time, sec, static_cast<double>(g_launches), slots,
                          slots > 0 ? sec / slots * 1e9 : 0.0});
  }
  g_seconds = 0.0;
  g_launches = 0;
  g_slots = 0;
}

} // namespace Kinetic
