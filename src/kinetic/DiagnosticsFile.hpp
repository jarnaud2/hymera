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

// One HDF5 file per run, <problem_id>.diag.h5, for the kinetic diagnostics that
// are not Parthenon outputs:
//
//   /timeseries/<name>/<column>   1-D float64, one entry per row appended
//   /profiles/<name>/<column>     1-D float64, written once
//   /meta                         attributes: problem_id, units of each column
//
// Rank 0 owns the file; other ranks' calls do nothing. Rows are buffered and
// written at Flush (every Parthenon output, and at the end of the run), so a job
// that is killed loses at most one output interval and leaves a readable file.
//
// On restart the existing file is reopened and every time series is cut back to
// its rows with t <= restart time (TruncateAfter), so a crashed and restarted run
// family has one continuous series with no duplicated rows.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Kinetic {

class DiagnosticsFile {
 public:
  /// The run's file; nullptr on ranks other than 0 or before Open.
  static DiagnosticsFile *Get();

  /// Creates <problem_id>.diag.h5, or on restart reopens it. Called once, by
  /// Initialize.
  static void Open(const std::string &problem_id, bool is_restart);

  /// Drops the rows of every time series with t > time. Called on restart, with
  /// the restart time, before the first step.
  void TruncateAfter(double time);

  /// Writes buffered rows and closes the file.
  static void Close();

  /// Declares a time series. The first column must be the time "t". Calling it
  /// again with the same name is allowed and does nothing.
  void DeclareSeries(const std::string &name, const std::vector<std::string> &columns,
                     const std::vector<std::string> &units);

  /// Buffers one row; values in the declared column order.
  void Append(const std::string &name, const std::vector<double> &row);

  /// Writes a profile (columns of equal length) and overwrites any earlier one.
  void WriteProfile(const std::string &name, const std::vector<std::string> &columns,
                    const std::vector<std::string> &units,
                    const std::vector<std::vector<double>> &data);

  /// Writes all buffered rows to disk and flushes the file.
  void Flush();

  ~DiagnosticsFile();

 private:
  struct Impl;
  explicit DiagnosticsFile(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// Flushes the run's diagnostics file, if any. Safe to call on every rank.
inline void FlushDiagnostics() {
  if (auto *d = DiagnosticsFile::Get()) d->Flush();
}

} // namespace Kinetic
