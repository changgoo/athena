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

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::AccreteMass()
//! \brief accrete gas from neighboring cells

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

    // Step 0. Prepare

    // find the indices of the particle-containing cell.
    int ip = GetCellIndex1(idx);
    int jp = GetCellIndex2(idx);
    int kp = GetCellIndex3(idx);

    AthenaArray<Real> &cons = pmy_block->phydro->u;

    if (COORDINATE_SYSTEM != "cartesian") {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [SinkParticles::AccreteMass]" << std::endl
          << "Only Cartesian coordinate system is supported " << std::endl;
      ATHENA_ERROR(msg);
    }

    // Step 1. Calculate total mass in the control volume M^{n+1} updated by hydro
    // integrator, before applying extrapolation.

    Real m{0.}, M1{0.}, M2{0.}, M3{0.};
    for (int k=kp-rctrl_; k<=kp+rctrl_; ++k) {
      for (int j=jp-rctrl_; j<=jp+rctrl_; ++j) {
        for (int i=ip-rctrl_; i<=ip+rctrl_; ++i) {
          Real dV = pmy_block->pcoord->GetCellVolume(k,j,i);
          m += cons(IDN,k,j,i)*dV;
          M1 += cons(IM1,k,j,i)*dV;
          M2 += cons(IM2,k,j,i)*dV;
          M3 += cons(IM3,k,j,i)*dV;
        }
      }
    }

    // Step 2. Reset the density inside the control volume by extrapolation
    SetGhostRegion(cons, ip, jp, kp);

    // Step 3. Calculate M^{n+1}_ctrl

    Real mext{0.}, M1ext{0.}, M2ext{0.}, M3ext{0.};
    for (int k=kp-rctrl_; k<=kp+rctrl_; ++k) {
      for (int j=jp-rctrl_; j<=jp+rctrl_; ++j) {
        for (int i=ip-rctrl_; i<=ip+rctrl_; ++i) {
          Real dV = pmy_block->pcoord->GetCellVolume(k,j,i);
          mext += cons(IDN,k,j,i)*dV;
          M1ext += cons(IM1,k,j,i)*dV;
          M2ext += cons(IM2,k,j,i)*dV;
          M3ext += cons(IM3,k,j,i)*dV;
        }
      }
    }

    // Step 4. Calculate dM_sink by subtracting M^{n+1}_ctrl from M^{n+1}

    Real dm = m - mext;
    Real dM1 = M1 - M1ext;
    Real dM2 = M2 - M2ext;
    Real dM3 = M3 - M3ext;

    // Step 5. Check whether the particle has crossed the grid boundaries

    // Step 6... Do corrections for grid crossing.

    // Update mass and velocity of the particle
    Real minv = 1.0 / (mass(idx) + dm);
    vpx(idx) = (mass(idx)*vpx(idx) + dM1)*minv;
    vpy(idx) = (mass(idx)*vpy(idx) + dM2)*minv;
    vpz(idx) = (mass(idx)*vpz(idx) + dM3)*minv;
    mass(idx) += dm;
  } // end of the loop over particles
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::SetGhostRegion(AthenaArray<Real> &cons, int ip, int jp, int kp)
//! \brief set control volume quantities by extrapolating from neighboring active cells.

void SinkParticles::SetGhostRegion(AthenaArray<Real> &cons, int ip, int jp, int kp) {
  for (int k=kp-rctrl_; k<=kp+rctrl_; ++k) {
    for (int j=jp-rctrl_; j<=jp+rctrl_; ++j) {
      for (int i=ip-rctrl_; i<=ip+rctrl_; ++i) {
        // temporary implementation
        cons(IDN,k,j,i) = 1.0;
        cons(IM1,k,j,i) = 0.0;
        cons(IM2,k,j,i) = 0.0;
        cons(IM3,k,j,i) = 0.0;
      }
    }
  }
}
