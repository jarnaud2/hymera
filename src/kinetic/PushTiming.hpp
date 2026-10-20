//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
//========================================================================================
// Push-kernel timing (Simulation/push_timing = 1): seconds spent in the particle
// push kernel between outputs, the number of push launches and particle slots
// pushed, written to /timeseries/push of the diagnostics file at each output.
#pragma once

#include <cstdint>

namespace Kinetic {

struct PushTiming {
  /// Accumulates one push launch of `slots` particle slots taking `seconds`.
  static void Add(double seconds, std::int64_t slots);
  /// Appends (t, push_s, launches, slots, ns_per_slot) to /timeseries/push and
  /// resets the counters. No-op unless push timing is on.
  static void Flush(double time);
  static bool enabled;
};

} // namespace Kinetic
