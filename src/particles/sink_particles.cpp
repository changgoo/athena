//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file sink_particles.cpp
//! \brief implements functions in the SinkParticles class

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "particles.hpp"

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::SinkParticles(MeshBlock *pmb, ParameterInput *pin)
//! \brief constructs a SinkParticles instance.

SinkParticles::SinkParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp)
  : StarParticles(pmb, pin, pp) {}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::~SinkParticles()
//! \brief destroys a SinkParticles instance.

SinkParticles::~SinkParticles() {
  // nothing to do
  return;
}

void SinkParticles::AccreteMass() {
  // loop over all particles
  for (int idx=0; idx<npar_; ++idx) {
    // Determine the dM_sink accreted by the sink particle.
    // Because we reset the density in the control volume by extrapolating from the
    // adjacent active cells, the total mass in the control volume can change.
    // The mass dM_flux that has flown into the control volume during dt must be equal
    // to the dM_sink plus the mass change dM_ctrl inside the control volume. That is,
    // we can calculate dM_sink by
    //   dM_sink = dM_flux - dM_ctrl       -- (1)
    // Meanwhile, hydro integrator will update M^{n}_ctrl at time t^n (which is already
    // reset to the extrapolated value) to M^{n+1}, which will be subsequently reset to
    // the extrapolated value M^{n+1}_ctrl. This is done by
    //   M^{n+1} = M^{n}_ctrl + dM_flux    -- (2)
    // Therefore, instead using Riemann fluxes directly to calculate dM_flux in eq. (1),
    // we can use eq. (2) to substitute dM_flux in eq. (1) with M^{n+1} - M^{n}_ctrl,
    // yielding
    //   dM_sink = M^{n+1} - M^{n+1}_ctrl  -- (3)

    // Step 1. Calculate total mass in the control volume M^{n+1} updated by hydro
    // integrator, before applying extrapolation.

    // Step 2. Reset the density inside the control volume by extrapolation

    // Step 3. Calculate M^{n+1}_ctrl

    // Step 4. Calculate dM_sink by subtracting M^{n+1}_ctrl from M^{n+1}

    // Step 5. Check whether the particle has crossed the grid boundaries

    // Step 6... Do corrections for grid crossing.

// Update mass and velocity of the particle
//    mass(idx) += dm;
//    vpx(idx) += dvx;
//    vpy(idx) += dvy;
//    vpz(idx) += dvz;
  }
}
