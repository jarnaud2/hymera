// Time per secondary of LargeAngleCollision (reference) and LargeAngleCollisionFast
// on the default execution space (GPU in a CUDA build).
//
//   la_sampler_bench OUT.csv [N] [REPEATS]
//
// For every gamma_min x primary momentum p x pitch, N secondaries are drawn in one
// parallel_for (one primary per thread, as AddSecondaries does), REPEATS times;
// the CSV holds the best time of the repeats and ns per secondary.
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Random.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kinetic/LargeAngleCollision.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    if (argc < 2) { std::fprintf(stderr, "usage: %s OUT.csv [N] [REPEATS]\n", argv[0]); return 2; }
    const int N = argc > 2 ? std::atoi(argv[2]) : 1 << 20;
    const int REP = argc > 3 ? std::atoi(argv[3]) : 3;
    const std::vector<double> gmins = {1.0002, 1.002, 1.02, 1.1};
    const std::vector<double> ps = {2.0, 5.0, 15.0, 50.0, 150.0};
    const std::vector<double> xis = {-0.9, 0.3};
    std::FILE *f = std::fopen(argv[1], "w");
    std::fprintf(f, "sampler,gamma_min,p,xi,N,seconds,ns_per_secondary,failed\n");
    RNGPool pool(2024);
    Kokkos::View<int> nfail("nfail");
    for (double gmin : gmins)
      for (double p0 : ps)
        for (double xi0 : xis)
          for (int which = 0; which < 2; ++which) {
            double best = 1e30;
            int fails = 0;
            for (int r = 0; r < REP; ++r) {
              Kokkos::deep_copy(nfail, 0);
              Kokkos::fence();
              auto t0 = std::chrono::steady_clock::now();
              Kokkos::parallel_for("la_bench", N, KOKKOS_LAMBDA(const int) {
                Real p = p0, xi = xi0;
                if (which == 0)
                  LargeAngleCollision(p, xi, 1.0, 1e-5, gmin, pool);
                else
                  LargeAngleCollisionFast(p, xi, 1.0, 1e-5, gmin, pool);
                if (!(p > 0.0)) Kokkos::atomic_add(&nfail(), 1);
              });
              Kokkos::fence();
              best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
              auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nfail);
              fails = h();
            }
            std::fprintf(f, "%s,%g,%g,%g,%d,%.6e,%.4f,%d\n", which == 0 ? "reference" : "fast", gmin, p0,
                         xi0, N, best, best / N * 1e9, fails);
            std::fflush(f);
            std::printf("%-9s gmin=%-7g p=%-5g xi=%+.1f  %.4e s  %.2f ns/secondary\n",
                        which == 0 ? "reference" : "fast", gmin, p0, xi0, best, best / N * 1e9);
          }
    std::fclose(f);
  }
  Kokkos::finalize();
  return 0;
}
