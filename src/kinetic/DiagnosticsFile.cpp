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

#include "kinetic/DiagnosticsFile.hpp"

#include <H5Cpp.h>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <globals.hpp>

#include "kinetic/Log.hpp"

namespace Kinetic {

namespace {

std::unique_ptr<DiagnosticsFile> &Instance() {
  static std::unique_ptr<DiagnosticsFile> instance;
  return instance;
}

H5::Group OpenOrCreateGroup(H5::H5Location &parent, const std::string &name) {
  if (parent.nameExists(name)) return parent.openGroup(name);
  return parent.createGroup(name);
}

void WriteStringAttr(H5::H5Object &obj, const std::string &name, const std::string &value) {
  if (obj.attrExists(name)) obj.removeAttr(name);
  H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::Attribute a = obj.createAttribute(name, type, H5::DataSpace(H5S_SCALAR));
  a.write(type, value);
}

// Appends n values to a 1-D extendible float64 dataset, creating it if needed.
void AppendColumn(H5::Group &g, const std::string &name, const std::vector<double> &v) {
  if (v.empty() && g.nameExists(name)) return;
  H5::DataSet ds;
  hsize_t old_size = 0;
  if (g.nameExists(name)) {
    ds = g.openDataSet(name);
    ds.getSpace().getSimpleExtentDims(&old_size);
  } else {
    hsize_t zero = 0, unlimited = H5S_UNLIMITED, chunk = 1024;
    H5::DataSpace space(1, &zero, &unlimited);
    H5::DSetCreatPropList prop;
    prop.setChunk(1, &chunk);
    ds = g.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space, prop);
  }
  if (v.empty()) return;
  hsize_t new_size = old_size + v.size(), count = v.size();
  ds.extend(&new_size);
  H5::DataSpace file_space = ds.getSpace();
  file_space.selectHyperslab(H5S_SELECT_SET, &count, &old_size);
  H5::DataSpace mem_space(1, &count);
  ds.write(v.data(), H5::PredType::NATIVE_DOUBLE, mem_space, file_space);
}

} // namespace

struct DiagnosticsFile::Impl {
  H5::H5File file;
  struct Series {
    std::vector<std::string> columns;
    std::vector<std::vector<double>> pending;  // one vector per column
  };
  std::map<std::string, Series> series;
};

DiagnosticsFile::DiagnosticsFile(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DiagnosticsFile::~DiagnosticsFile() {
  try {
    Flush();
  } catch (...) {
    // Never throw from a destructor; the data written at earlier flushes stays.
  }
}

DiagnosticsFile *DiagnosticsFile::Get() { return Instance().get(); }

void DiagnosticsFile::Open(const std::string &problem_id, bool is_restart) {
  if (parthenon::Globals::my_rank != 0 || Instance()) return;
  const std::string name = problem_id + ".diag.h5";

  auto impl = std::make_unique<Impl>();
  // Check existence first: H5File::isHdf5 on a missing file reports an HDF5
  // error, which Parthenon's HDF5 error handler turns into an abort.
  const bool reopened = is_restart && std::filesystem::exists(name) && H5::H5File::isHdf5(name);
  impl->file = H5::H5File(name, reopened ? H5F_ACC_RDWR : H5F_ACC_TRUNC);
  H5::Group meta = OpenOrCreateGroup(impl->file, "meta");
  WriteStringAttr(meta, "problem_id", problem_id);
  WriteStringAttr(meta, "format", "hymera kinetic diagnostics, version 1");
  OpenOrCreateGroup(impl->file, "timeseries");
  OpenOrCreateGroup(impl->file, "profiles");
  Log::Info("Diagnostics: {} {}", reopened ? "reopened" : "writing", name);
  Instance().reset(new DiagnosticsFile(std::move(impl)));
}

void DiagnosticsFile::TruncateAfter(double time) {
  Flush();
  H5::Group ts = impl_->file.openGroup("timeseries");
  for (hsize_t i = 0; i < ts.getNumObjs(); ++i) {
    H5::Group g = ts.openGroup(ts.getObjnameByIdx(i));
    if (!g.nameExists("t")) continue;
    H5::DataSet tds = g.openDataSet("t");
    hsize_t n = 0;
    tds.getSpace().getSimpleExtentDims(&n);
    std::vector<double> t(n);
    if (n > 0) tds.read(t.data(), H5::PredType::NATIVE_DOUBLE);
    hsize_t keep = 0;
    while (keep < n && t[keep] <= time) ++keep;
    for (hsize_t c = 0; c < g.getNumObjs(); ++c) {
      H5::DataSet ds = g.openDataSet(g.getObjnameByIdx(c));
      ds.extend(&keep);
    }
    if (keep < n)
      Log::Info("Diagnostics: dropped {} rows of {} after the restart time {:g} s",
                n - keep, ts.getObjnameByIdx(i), time);
  }
  impl_->file.flush(H5F_SCOPE_GLOBAL);
}

void DiagnosticsFile::Close() { Instance().reset(); }

void DiagnosticsFile::DeclareSeries(const std::string &name,
                                    const std::vector<std::string> &columns,
                                    const std::vector<std::string> &units) {
  if (impl_->series.count(name)) return;
  if (columns.empty() || columns[0] != "t" || units.size() != columns.size())
    throw std::invalid_argument(std::format(
        "DiagnosticsFile: series {} needs a first column \"t\" and one unit per column",
        name));
  H5::Group ts = impl_->file.openGroup("timeseries");
  H5::Group g = OpenOrCreateGroup(ts, name);
  for (size_t c = 0; c < columns.size(); ++c) {
    AppendColumn(g, columns[c], {});
    H5::DataSet ds = g.openDataSet(columns[c]);
    WriteStringAttr(ds, "units", units[c]);
  }
  impl_->series[name] = {columns, std::vector<std::vector<double>>(columns.size())};
}

void DiagnosticsFile::Append(const std::string &name, const std::vector<double> &row) {
  auto it = impl_->series.find(name);
  if (it == impl_->series.end())
    throw std::invalid_argument("DiagnosticsFile: series " + name + " was not declared");
  auto &s = it->second;
  if (row.size() != s.columns.size())
    throw std::invalid_argument(std::format("DiagnosticsFile: series {} has {} columns, row has {}",
                                            name, s.columns.size(), row.size()));
  for (size_t c = 0; c < row.size(); ++c) s.pending[c].push_back(row[c]);
}

void DiagnosticsFile::WriteProfile(const std::string &name,
                                   const std::vector<std::string> &columns,
                                   const std::vector<std::string> &units,
                                   const std::vector<std::vector<double>> &data) {
  H5::Group profiles = impl_->file.openGroup("profiles");
  if (profiles.nameExists(name)) profiles.unlink(name);
  H5::Group g = profiles.createGroup(name);
  for (size_t c = 0; c < columns.size(); ++c) {
    hsize_t n = data[c].size();
    H5::DataSet ds = g.createDataSet(columns[c], H5::PredType::NATIVE_DOUBLE,
                                     H5::DataSpace(1, &n));
    if (n > 0) ds.write(data[c].data(), H5::PredType::NATIVE_DOUBLE);
    WriteStringAttr(ds, "units", units[c]);
  }
  impl_->file.flush(H5F_SCOPE_GLOBAL);
}

void DiagnosticsFile::WriteSnapshot(const std::string &label,
                                    const std::map<std::string, std::vector<double>> &columns,
                                    const std::map<std::string, double> &attributes) {
  H5::Group snaps = OpenOrCreateGroup(impl_->file, "snapshots");
  if (snaps.nameExists(label)) snaps.unlink(label);
  H5::Group g = snaps.createGroup(label);
  for (const auto &[name, v] : columns) {
    hsize_t n = v.size();
    H5::DataSet ds = g.createDataSet(name, H5::PredType::NATIVE_DOUBLE, H5::DataSpace(1, &n));
    if (n > 0) ds.write(v.data(), H5::PredType::NATIVE_DOUBLE);
  }
  for (const auto &[name, value] : attributes) {
    H5::Attribute a = g.createAttribute(name, H5::PredType::NATIVE_DOUBLE,
                                        H5::DataSpace(H5S_SCALAR));
    a.write(H5::PredType::NATIVE_DOUBLE, &value);
  }
  impl_->file.flush(H5F_SCOPE_GLOBAL);
}

void DiagnosticsFile::Flush() {
  H5::Group ts = impl_->file.openGroup("timeseries");
  for (auto &[name, s] : impl_->series) {
    if (s.pending[0].empty()) continue;
    H5::Group g = ts.openGroup(name);
    for (size_t c = 0; c < s.columns.size(); ++c) {
      AppendColumn(g, s.columns[c], s.pending[c]);
      s.pending[c].clear();
    }
  }
  impl_->file.flush(H5F_SCOPE_GLOBAL);
}

} // namespace Kinetic
