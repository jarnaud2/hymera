#pragma once
#include <memory>
#include <string>

#include <parthenon/driver.hpp>
#include "mhd.h"

namespace Kinetic {

using namespace parthenon;
using namespace parthenon::driver::prelude;

class AvalancheDriver : public EvolutionDriver {
public:
  AvalancheDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pmesh);
  TaskCollection MakeTaskCollection(BlockList_t &blocks, SimTime tm);
  TaskListStatus Step();
};

/// Options of the `avalanche` executable, read from <Avalanche> (see avalanche.cpp).
struct AvalancheOptions {
  std::string field_source;  ///< mhd | analytic | file
  std::string seed;          ///< point | ring | surface
  static AvalancheOptions FromInput(ParameterInput *pin);
};

/// Adds the Deck params of the non-MHD field sources: the seed ring radius, the
/// exact (real-valued) E_0, the analytic field and the surface target.
void AddAvalancheParams(ParameterInput *pin, const AvalancheOptions &opt,
                        std::shared_ptr<StateDescriptor> &pkg);

/// After the fields are loaded: moves surface seeds to their flux surface,
/// checks the seed pitch sign against E and B, and declares the
/// /timeseries/avalanche series.
void PrepareAvalancheSeeds(Mesh *pm, const AvalancheOptions &opt);

/// Appends (t, N_alive, W_alive) to /timeseries/avalanche.
void AppendAvalancheRow(Mesh *pm, SimTime const &tm);

}
