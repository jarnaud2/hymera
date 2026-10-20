// Samples of the secondary (gamma, xi) from LargeAngleCollision (reference) and
// LargeAngleCollisionFast for a set of primaries, for la_sampler_check.py.
//
//   la_sampler_test OUT.bin N
//
// Writes, per case, N reference and N fast samples (gamma, xi, tries) as doubles
// with a small header; the reference sampler is run with the same N, so its
// cost per sample is also recorded (wall time per case).
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kinetic/LargeAngleCollision.hpp"

struct Case { double p, xi, gmin; };

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    if (argc < 3) { std::fprintf(stderr, "usage: %s OUT N\n", argv[0]); return 2; }
    const int N = std::atoi(argv[2]);
    const std::vector<Case> cases = {
        {15.0, -0.90, 1.02}, {15.0, 0.30, 1.02}, {2.0, -0.95, 1.02}, {50.0, -0.99, 1.02},
        {15.0, -0.90, 1.002}, {5.0, 0.0, 1.1}, {0.6, -0.5, 1.02}};
    std::FILE *f = std::fopen(argv[1], "wb");
    const int nc = cases.size();
    std::fwrite(&nc, sizeof(int), 1, f);
    std::fwrite(&N, sizeof(int), 1, f);
    RNGPool pool(argc > 3 ? std::atoi(argv[3]) : 12345);
    for (const auto &c : cases) {
      double hdr[3] = {c.p, c.xi, c.gmin};
      std::fwrite(hdr, sizeof(double), 3, f);
      for (int which = 0; which < 2; ++which) {
        Kokkos::View<double *[2]> s("s", N);
        auto t0 = std::chrono::steady_clock::now();
        Kokkos::parallel_for("sample", N, KOKKOS_LAMBDA(const int i) {
          Real p = c.p, xi = c.xi;
          if (which == 0)
            LargeAngleCollision(p, xi, 1.0, 1e-5, c.gmin, pool);
          else
            LargeAngleCollisionFast(p, xi, 1.0, 1e-5, c.gmin, pool);
          s(i, 0) = p > 0 ? gamma_(p) : -1.0;
          s(i, 1) = xi;
        });
        Kokkos::fence();
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s);
        std::fwrite(&dt, sizeof(double), 1, f);
        std::fwrite(h.data(), sizeof(double), 2 * N, f);
        std::printf("p=%5.1f xi=%+.2f gmin=%.3f %-9s %8.3f s for %d samples\n", c.p, c.xi, c.gmin,
                    which == 0 ? "reference" : "fast", dt, N);
      }
    }
    std::fclose(f);
  }
  Kokkos::finalize();
  return 0;
}
