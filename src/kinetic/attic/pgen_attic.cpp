// Quarantined from src/pgen.cpp: no caller in any compiled target.
// NOT COMPILED. Kept verbatim for reference; see README.md in this directory.

void GenerateParticleSquare(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin) {
  auto &data = pmb->meshblock_data.Get();

  // pull out information/global params from package
  auto pkg = pmb->packages.Get("Kinetic");
  auto rng_pool = pkg->Param<Kinetic::RNGPool>("rng_pool");
  const int N = pkg->Param<int>("num_particles_per_block");
  const Real pmin  = pkg->Param<Real>("pmin");
  const Real pmax  = pkg->Param<Real>("pmax");
  const Real ximin = pkg->Param<Real>("ximin");
  const Real ximax = pkg->Param<Real>("ximax");
  const Real Rmin  = pkg->Param<Real>("Rmin");
  const Real Rmax  = pkg->Param<Real>("Rmax");
  const Real Zmin  = pkg->Param<Real>("Zmin");
  const Real Zmax  = pkg->Param<Real>("Zmax");

  // Pull out swarm object
  auto swarm = data->GetSwarmData()->Get("particles");

  // Meshblock geometry
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int &nx_i = pmb->cellbounds.ncellsi(IndexDomain::interior);
  const int &nx_j = pmb->cellbounds.ncellsj(IndexDomain::interior);
  const int &nx_k = pmb->cellbounds.ncellsk(IndexDomain::interior);
  const Real &dx_i = pmb->coords.Dxf<1>(ib.s);
  const Real &dx_j = pmb->coords.Dxf<2>(jb.s);
  const Real &dx_k = pmb->coords.Dxf<3>(kb.s);
  const Real &minx_i = pmb->coords.Xf<1>(ib.s);
  const Real &minx_j = pmb->coords.Xf<2>(jb.s);
  const Real &minx_k = pmb->coords.Xf<3>(kb.s);


  // Create an accessor to particles, allocate particles
  auto newParticlesContext = swarm->AddEmptyParticles(N);

  // Make a SwarmPack via types to get positions
  static auto desc_swarm =
    parthenon::MakeSwarmPackDescriptor<
    swarm_position::x,
    swarm_position::y,
    swarm_position::z,
    Kinetic::p,
    Kinetic::xi,
    Kinetic::R,
    Kinetic::phi,
    Kinetic::Z,
    Kinetic::weight>("particles");
  auto pack_swarm = desc_swarm.GetPack(data.get());
  auto swarm_d = swarm->GetDeviceContext();

  // loop over new particles created
  parthenon::par_for(DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL,
      DevExecSpace(), 0,
      newParticlesContext.GetNewParticlesMaxIndex(),
      // new_n ranges from 0 to N_new_particles
      KOKKOS_LAMBDA(const int new_n) {
      // this is the particle index inside the swarm
      const int n = newParticlesContext.GetNewParticleIndex(new_n);
      auto rng_gen = rng_pool.get_state();

      // Normally b would be free-floating and set by pack.GetBlockparticleIndices
      // but since we're on a single meshblock for this loop, it's just 0
      // because block index = 0
      const int b = 0;
      //auto [b, n] = pack_swarm.GetBlockparticleIndices(idx);

      // Find the cell index in 1D
      int ind = n % (nx_i * nx_j * nx_k);

      int x_i = ind % nx_i;
      ind /= nx_i;
      int x_j = ind % nx_j;
      ind /= nx_j;
      int x_k = ind;

      pack_swarm(b, swarm_position::x(), n) = minx_i + (x_i+0.5) * dx_i;
      pack_swarm(b, swarm_position::y(), n) = minx_j + (x_j+0.5) * dx_j;
      pack_swarm(b, swarm_position::z(), n) = minx_k + (x_k+0.5) * dx_k;
      // randomly sample particle positions

      // set canonical momentum
      pack_swarm(b, Kinetic::p(), n) = rng_gen.drand(pmin, pmax);
      pack_swarm(b, Kinetic::xi(), n) = rng_gen.drand(ximin, ximax);
      // randomly sample particle positions (TODO: r, phi, z)
      pack_swarm(b, Kinetic::R(), n) = rng_gen.drand(Rmin, Rmax);
      pack_swarm(b, Kinetic::phi(), n) = rng_gen.drand(0, 2.0*M_PI);
      pack_swarm(b, Kinetic::Z(), n) = rng_gen.drand(Zmin, Zmax);

      // set weights to 1
      pack_swarm(b, Kinetic::weight(), n) = 1.0;

      // release random number generator
      rng_pool.free_state(rng_gen);
      });
}
