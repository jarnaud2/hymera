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

// Host-side run log for the kinetic code. One line per call, written by MPI rank 0
// only, prefixed with its level, and filtered by Simulation/verbosity:
//
//   quiet   warnings only
//   info    set-up milestones and once-per-run values (default)
//   debug   also per-step messages
//
//   Kinetic::Log::Info("Placing particles at ({},{})", R, Z);
//
// Device code cannot use this; it keeps Kokkos::printf.
#pragma once

#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <globals.hpp>
#include <parthenon/package.hpp>

namespace Kinetic {
namespace Log {

enum class Level : int { Quiet = 0, Info = 1, Debug = 2 };

inline Level &CurrentLevel() {
  static Level level = Level::Info;
  return level;
}

/// Reads Simulation/verbosity (quiet, info or debug; default info).
inline void Configure(parthenon::ParameterInput *pin) {
  const std::string v = pin->GetOrAddString("Simulation", "verbosity", "info");
  if (v == "quiet") CurrentLevel() = Level::Quiet;
  else if (v == "debug") CurrentLevel() = Level::Debug;
  else if (v == "info") CurrentLevel() = Level::Info;
  else {
    CurrentLevel() = Level::Info;
    if (parthenon::Globals::my_rank == 0)
      std::cout << "[warn] Simulation/verbosity = " << v
                << " is not quiet, info or debug; using info\n";
  }
}

inline bool Enabled(Level level) {
  return parthenon::Globals::my_rank == 0 &&
         static_cast<int>(level) <= static_cast<int>(CurrentLevel());
}

template <typename... Args>
void Write(Level level, std::string_view tag, std::format_string<Args...> fmt,
           Args &&...args) {
  if (!Enabled(level)) return;
  std::cout << tag << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

template <typename... Args>
void Info(std::format_string<Args...> fmt, Args &&...args) {
  Write(Level::Info, "[info]  ", fmt, std::forward<Args>(args)...);
}

template <typename... Args>
void Debug(std::format_string<Args...> fmt, Args &&...args) {
  Write(Level::Debug, "[debug] ", fmt, std::forward<Args>(args)...);
}

/// Shown at every level, including quiet.
template <typename... Args>
void Warn(std::format_string<Args...> fmt, Args &&...args) {
  Write(Level::Quiet, "[warn]  ", fmt, std::forward<Args>(args)...);
}

} // namespace Log
} // namespace Kinetic
