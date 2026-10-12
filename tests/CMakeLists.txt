# ========================================================================================
#  (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)
include("Utils")

# add_parthenon_test(avalanche_test
#   SRCS
#     ../src/util/common.hpp
#     ../src/kinetic/AnalyticField.hpp
#     ../src/kinetic/GuidingCenterEquations.hpp
#     ../src/kinetic/SmallAngleCollision.hpp
#     ../src/kinetic/LargeAngleCollision.hpp
#     ../src/pgen.cpp
#     ../src/pgen.hpp
#     ../src/kinetic/EM_Field.hpp
#     ../src/kinetic/CurrentDensity.hpp
#     kinetic/avalanche_test.cpp
#   INPUT
#     avalanche_test.input
# )

# add_parthenon_test(current_replacement
#   SRCS
#     ../src/util/common.hpp
#     ../src/kinetic/AnalyticField.hpp
#     ../src/kinetic/GuidingCenterEquations.hpp
#     ../src/kinetic/SmallAngleCollision.hpp
#     ../src/kinetic/LargeAngleCollision.hpp
#     ../src/kinetic/RunawayDriver.cpp
#     ../src/kinetic/kinetic.cpp
#     ../src/kinetic/EM_Field.cpp
#     ../src/pgen.cpp
#     ../src/pgen.hpp
#     ../src/kinetic/EM_Field.hpp
#     ../src/kinetic/CurrentDensity.hpp
#     kinetic/current_replacement.cpp
#     ../src/kinetic/EM_Field.cpp
#     ../src/kinetic/c_wrapper.cpp
#   INPUT
#     avalanche_test.input
# )

find_package(HDF5 COMPONENTS CXX REQUIRED)

# --------------------------------------------------------------------------------------
# MHD regression harness, tier 0: the discrete mass-matrix coefficients.
#
# Evaluates every live alpha-/beta-/rese/condu function in mass_matrix_coefficients.c
# over the full index/location/material space (~9.8M evaluations, a few seconds) and
# compares a SHA-256 of the exact hexadecimal output against a recorded value. Links
# mhd_core directly; needs no input files, no solver and no time step.
#
# This is the gate for deduplicating that file: any refactor of it must leave the
# hash unchanged. On failure, regenerate the full output on both revisions and diff
# them -- tests/regression/run_t0.sh does this.
# --------------------------------------------------------------------------------------
enable_language(C)
add_executable(t0_coefficients ${CMAKE_CURRENT_SOURCE_DIR}/regression/t0_coefficients.c)
target_link_libraries(t0_coefficients PRIVATE mhd_core MPI::MPI_C m)
add_test(NAME mhd_t0_coefficients
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/regression/run_t0.sh
                 $<TARGET_FILE:t0_coefficients>)
set_tests_properties(mhd_t0_coefficients PROPERTIES LABELS "tier0" TIMEOUT 120)

# Tier 0, second test: the mimetic operators that consume the coefficients above.
# Applies each live Vec-to-Vec operator to a fixed input on the production DOF
# layout and hashes the exact output. Gate for operator-level refactors.
add_executable(t0_operators ${CMAKE_CURRENT_SOURCE_DIR}/regression/t0_operators.c)
target_link_libraries(t0_operators PRIVATE mhd_core MPI::MPI_C m)
add_test(NAME mhd_t0_operators
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/regression/run_t0.sh
                 $<TARGET_FILE:t0_operators> --baseline t0_operators)
set_tests_properties(mhd_t0_operators PROPERTIES LABELS "tier0" TIMEOUT 300)

# Tier 1: the live implicit residuals, each evaluated once on a fixed state.
# Gate for refactoring ts_functions.c; see regression/t1_residuals.c.
add_executable(t1_residuals ${CMAKE_CURRENT_SOURCE_DIR}/regression/t1_residuals.c)
target_link_libraries(t1_residuals PRIVATE mhd_core MPI::MPI_C m)
add_test(NAME mhd_t1_residuals
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/regression/run_t0.sh
                 $<TARGET_FILE:t1_residuals> --baseline t1_residuals)
set_tests_properties(mhd_t1_residuals PROPERTIES LABELS "tier1" TIMEOUT 300)

# --------------------------------------------------------------------------------------
# Kinetic regression harness: each case runs one kinetic executable on a reduced deck
# from regression/kinetic/decks/ and hashes every number it writes (particle swarm,
# field grid, diagnostics files). Gate for the kinetic/pgen refactor, which must leave
# every hash unchanged. Fixtures (CPU-layout fields.h5, relaxed MHD state X_4.petscbin)
# are in regression/kinetic/fixtures/. See docs/kinetic/testing.md.
#
# The light cases load the pre-interpolated field grid and need no MHD solve (seconds).
# kin_hybrid runs the coupled MHD solve and takes several minutes, so it carries its
# label `slow` (not `kinetic`, which ctest -L would match as a regex) and needs
# about 4 GB of memory, so run it alone.
# --------------------------------------------------------------------------------------
set(KINETIC_HARNESS ${CMAKE_CURRENT_SOURCE_DIR}/regression/kinetic)
function(add_kinetic_test name exe deck label threads)
  add_test(NAME kin_${name}
           COMMAND ${KINETIC_HARNESS}/run_kinetic.sh $<TARGET_FILE:${exe}>
                   ${KINETIC_HARNESS}/decks/${deck}.input --baseline kin_${name}
                   --threads ${threads} ${ARGN})
  set_tests_properties(kin_${name} PROPERTIES LABELS "${label}" TIMEOUT 3600)
endfunction()

# conservation gates at 1 thread: its conservation.dat totals come from a
# parallel_reduce whose summation order follows the OpenMP thread split.
add_kinetic_test(conservation             conservation    conservation             kinetic 1 parthenon/time/nlim=4)
add_kinetic_test(conservation_dopri5_nocoll conservation  conservation_dopri5_nocoll kinetic 4 parthenon/time/nlim=4)
add_kinetic_test(profile                  profile         profile                  kinetic 4 parthenon/time/nlim=4)
add_kinetic_test(single_particle          single_particle single_particle          kinetic 4 parthenon/time/nlim=4)
add_kinetic_test(hybrid                   hybrid          hybrid                   slow    4 parthenon/time/nlim=1)

# Diagnostics file across a restart: the restarted run must drop the rows after
# the restart time and reproduce the uninterrupted run's series exactly.
add_test(NAME kin_restart_diag
         COMMAND ${KINETIC_HARNESS}/run_restart_diag.sh $<TARGET_FILE:conservation>)
set_tests_properties(kin_restart_diag PROPERTIES LABELS "kinetic" TIMEOUT 600)
