//========================================================================================(
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC // for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

// Raw field-grid files: dump and load the interpolated field data (profile, conservation, single_particle).

#include <memory>
#include <numeric>
#include <iostream>
#include <format>
#include <random>
#include <typeinfo>  //for 'typeid' to work
#include <parthenon/package.hpp>
#include <hFlux/dopri.hpp>
#include <hFlux/FieldData.hpp>
#include <hFlux/Evaluator.hpp>

using namespace parthenon;

#include "kinetic/RunawayDriver.h"
#include "kinetic/kinetic.hpp"
#include "kinetic/KineticParams.hpp"
#include "kinetic/Log.hpp"
#include "kinetic/GuidingCenterEquations.hpp"
#include "kinetic/LargeAngleCollision.hpp"
#include "kinetic/SmallAngleCollision.hpp"
#include "kinetic/kinetic.hpp"
#include "kinetic/ConfigurationDomainGeometry.hpp"
#include "kinetic/CurrentDensity.hpp"
#include "kinetic/AnalyticField.hpp"
#include "kinetic/FieldEvaluator.hpp"
#include "tasks/Tasks.h"
#include "mhd/mhd.h"
#include "util/H5io.hpp"

using parthenon::constants::SI;
using parthenon::constants::PhysicalConstants;
using pc = PhysicalConstants<SI>;

namespace Kinetic {

// Dump the raw, pre-interpolated field grid (the Hermite/Taylor coefficient
// arrays the particle push consumes) plus geometry to a single HDF5 file, so a
// separate `profile` run can load it and exercise PushParticles without MHD.
// The FieldData views are stored flat in their memory order. hFlux fixes their
// layouts explicitly, so that order is the same on every backend. The CDG
// indicator has the default layout of the execution space (LayoutLeft on CUDA,
// LayoutRight on host), so it is written in a fixed row-major order (R index
// slowest) and the file is marked field_io_version 2. Rank 0 writes; the field
// grid is a full-domain replicated array.
void SaveRawFieldData(Mesh * pm, const char* filename) {
  if (Globals::my_rank != 0) return;

  auto pkg = pm->packages.Get("Deck");
  auto data = pkg->Param<FieldData_t>("FieldData");
  auto cdg  = pkg->Param<ConfigurationDomainGeometry>("CDG");
  const Real Rc = pkg->Param<Real>("Rc");
  const Real Zc = pkg->Param<Real>("Zc");

  data.data.sync_host();
  data.hermite_data.sync_host();
  data.psi_data.sync_host();

  auto data_h = data.data.view_host();
  auto herm_h = data.hermite_data.view_host();
  auto psi_h  = data.psi_data.view_host();

  auto indicator_h = Kokkos::create_mirror_view(cdg.indicator_view);
  Kokkos::deep_copy(indicator_h, cdg.indicator_view);

  H5::H5File file(filename, H5F_ACC_TRUNC);
  H5::Group vgroup = file.createGroup(H5::viewsGroupName);
  H5::Group sgroup = file.createGroup(H5::scalarsGroupName);

  H5::writeVector(vgroup, "data",         data_h.data(), static_cast<int>(data_h.size()));
  H5::writeVector(vgroup, "hermite_data", herm_h.data(), static_cast<int>(herm_h.size()));
  H5::writeVector(vgroup, "psi_data",     psi_h.data(),  static_cast<int>(psi_h.size()));
  const int ind_nR = indicator_h.extent(0);
  const int ind_nZ = indicator_h.extent(1);
  std::vector<int> indicator_row_major(ind_nR * ind_nZ);
  for (int i = 0; i < ind_nR; ++i)
    for (int j = 0; j < ind_nZ; ++j)
      indicator_row_major[i * ind_nZ + j] = indicator_h(i, j);
  H5::writeVector(vgroup, "indicator", indicator_row_major);

  H5::writeScalar(sgroup, "field_io_version", 2);
  int hermite_nR = data.hermite_locator.nR;
  int hermite_nZ = data.hermite_locator.nZ;
  int fd_nR = data.fd_locator.nR;
  int fd_nZ = data.fd_locator.nZ;
  H5::writeScalar(sgroup, "hermite_nR", hermite_nR);
  H5::writeScalar(sgroup, "hermite_nZ", hermite_nZ);
  H5::writeScalar(sgroup, "fd_nR", fd_nR);
  H5::writeScalar(sgroup, "fd_nZ", fd_nZ);
  H5::writeScalar(sgroup, "Rc", Rc);
  H5::writeScalar(sgroup, "Zc", Zc);

  Log::Info("Saved raw field data to {}", filename);
}

// Element order of the indicator stored in a raw field file. Version-2 files
// are row-major. Older files hold the producing build's memory order: row-major
// from a host build, column-major from a CUDA build. That order is recovered by
// comparing the file with the indicator read from Geometry/input_file, which
// the producing run also used.
template <typename HostIndicator>
static bool IndicatorFileIsRowMajor(H5::Group &sgroup, const std::vector<int> &file,
                                    const HostIndicator &geometry, const char *filename) {
  if (sgroup.attrExists("field_io_version")) {
    int version = 0;
    H5::readScalar(sgroup, "field_io_version", version);
    if (version >= 2) return true;
  }
  const int nR = geometry.extent(0);
  const int nZ = geometry.extent(1);
  bool row_major = true, col_major = true;
  for (int i = 0; i < nR; ++i)
    for (int j = 0; j < nZ; ++j) {
      row_major = row_major && file[i * nZ + j] == geometry(i, j);
      col_major = col_major && file[j * nR + i] == geometry(i, j);
    }
  if (row_major) return true;
  if (col_major) return false;
  throw std::runtime_error(std::format(
      "LoadRawFieldData: the indicator in {} matches Geometry/input_file in neither "
      "row- nor column-major order. The file predates field_io_version 2, so its "
      "element order cannot be determined; use the geometry file of the producing run.",
      filename));
}

// Inverse of SaveRawFieldData: fill the (already-constructed) FieldData grid and
// CDG indicator from an HDF5 dump, then update the magnetic-axis params. Used by
// the `profile` executable in place of the MHD field-fill pipeline. All ranks
// read the same replicated grid.
void LoadRawFieldData(Mesh * pm, const char* filename) {
  auto pkg = pm->packages.Get("Deck");
  auto data = pkg->Param<FieldData_t>("FieldData");
  auto cdg  = pkg->Param<ConfigurationDomainGeometry>("CDG");

  auto data_h = data.data.view_host();
  auto herm_h = data.hermite_data.view_host();
  auto psi_h  = data.psi_data.view_host();

  H5::H5File file(filename, H5F_ACC_RDONLY);
  H5::Group vgroup = file.openGroup(H5::viewsGroupName);

  // Dimension sanity check against the file.
  H5::Group sgroup = file.openGroup(H5::scalarsGroupName);
  int hermite_nR = 0, hermite_nZ = 0;
  H5::readScalar(sgroup, "hermite_nR", hermite_nR);
  H5::readScalar(sgroup, "hermite_nZ", hermite_nZ);
  if (hermite_nR != data.hermite_locator.nR || hermite_nZ != data.hermite_locator.nZ) {
    throw std::runtime_error(std::format(
        "LoadRawFieldData: grid mismatch. File hermite ({},{}) vs configured ({},{}). "
        "Match Numerical/NR,NZ and Geometry/* to the producing run.",
        hermite_nR, hermite_nZ, data.hermite_locator.nR, data.hermite_locator.nZ));
  }

  H5::readVector(vgroup, "data",         data_h.data(), static_cast<int>(data_h.size()));
  H5::readVector(vgroup, "hermite_data", herm_h.data(), static_cast<int>(herm_h.size()));
  H5::readVector(vgroup, "psi_data",     psi_h.data(),  static_cast<int>(psi_h.size()));

  data.data.modify_host();
  data.hermite_data.modify_host();
  data.psi_data.modify_host();
  data.data.sync_device();
  data.hermite_data.sync_device();
  data.psi_data.sync_device();

  // The indicator already holds the geometry read from Geometry/input_file in
  // Initialize. The file's copy replaces it.
  auto indicator_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cdg.indicator_view);
  const int ind_nR = indicator_h.extent(0);
  const int ind_nZ = indicator_h.extent(1);
  std::vector<int> indicator_file(ind_nR * ind_nZ);
  H5::readVector(vgroup, "indicator", indicator_file.data(), static_cast<int>(indicator_file.size()));
  const bool row_major = IndicatorFileIsRowMajor(sgroup, indicator_file, indicator_h, filename);
  for (int i = 0; i < ind_nR; ++i)
    for (int j = 0; j < ind_nZ; ++j)
      indicator_h(i, j) = row_major ? indicator_file[i * ind_nZ + j]
                                    : indicator_file[j * ind_nR + i];
  Kokkos::deep_copy(cdg.indicator_view, indicator_h);

  Real Rc = 0.0, Zc = 0.0;
  H5::readScalar(sgroup, "Rc", Rc);
  H5::readScalar(sgroup, "Zc", Zc);
  pkg->UpdateParam("Rc", Rc);
  pkg->UpdateParam("Zc", Zc);

  Log::Info("Loaded raw field data from {}: magnetic axis (R, Z) = ({:g}, {:g})", filename, Rc, Zc);
}

} // namespace Kinetic
