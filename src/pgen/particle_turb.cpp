//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_turb.cpp
//! \brief Problem generator for turbulence + gravity + particles
//

// C headers

// C++ headers
#include <cmath>
#include <ctime>
#include <random>     // mt19937, normal_distribution, uniform_real_distribution
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fft/athena_fft.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../particles/particles.hpp"
#include "../utils/utils.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (SELF_GRAVITY_ENABLED) {
    Real d0 = pin->GetOrAddReal("problem", "d0", 1.0);
    Real four_pi_G = pin->GetReal("problem","four_pi_G");
    Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
    SetFourPiG(four_pi_G);
    SetGravityThreshold(eps);

    if (PARTICLES) {
      Real dpar0 = d0 * pin->GetOrAddReal("problem", "dtog", 1.0);
    }
  }

  // turb_flag is initialzed in the Mesh constructor to 0 by default;
  // turb_flag = 1 for decaying turbulence
  // turb_flag = 2 for impulsively driven turbulence
  // turb_flag = 3 for continuously driven turbulence
  turb_flag = pin->GetInteger("problem","turb_flag");
  if (turb_flag != 0) {
#ifndef FFT
    std::stringstream msg;
    msg << "### FATAL ERROR in TurbulenceDriver::TurbulenceDriver" << std::endl
        << "non zero Turbulence flag is set without FFT!" << std::endl;
    ATHENA_ERROR(msg);
    return;
#endif
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real d0 = pin->GetOrAddReal("problem", "d0", 1.0);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IDN,k,j,i) = d0;

        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = 1.0;
        }
      }
    }
  }

  if (PARTICLES) {
    // Get the dust-to-gas ratio and the velocity of the particles.
    Real dtog, vamp;
    dtog = pin->GetOrAddReal("problem", "dtog", 1.0);

    // Find the total number of particles in each direction.
    RegionSize& mesh_size = pmy_mesh->mesh_size;
    Real np_per_cell = pin->GetOrAddReal("problem", "np_per_cell",1);
    int npx1 = (block_size.nx1 > 1) ? static_cast<int>(mesh_size.nx1*np_per_cell) : 1;
    int npx2 = (block_size.nx2 > 1) ? static_cast<int>(mesh_size.nx2*np_per_cell) : 1;
    int npx3 = (block_size.nx3 > 1) ? static_cast<int>(mesh_size.nx3*np_per_cell) : 1;

    // Uniformly separted particles, uniform mass = total mass / N particles
    // Find the mass of each particle and the distance between adjacent particles.
    Real vol = mesh_size.x1len * mesh_size.x2len * mesh_size.x3len;
    Real dx1 = mesh_size.x1len / npx1,
         dx2 = mesh_size.x2len / npx2,
         dx3 = mesh_size.x3len / npx3;
    DustParticles::SetOneParticleMass(dtog * vol / (npx1 * npx2 * npx3));

    // Determine number of particles in the block.
    int npx1_loc = static_cast<int>(std::round(block_size.x1len / dx1)),
        npx2_loc = static_cast<int>(std::round(block_size.x2len / dx2)),
        npx3_loc = static_cast<int>(std::round(block_size.x3len / dx3));
    int npar = ppar->npar = npx1_loc * npx2_loc * npx3_loc;
    if (npar > ppar->nparmax)
      ppar->UpdateCapacity(npar);

    // Assign the particles.
    // Ramdomizing position. Or velocity perturbation
    std::random_device device;
    std::mt19937_64 rng_generator;
    // std::int64_t rseed = static_cast<std::int64_t>(device());
    std::int64_t rseed = gid;
    std::uniform_real_distribution<Real> udist(0.0,1.0); // uniform in [0,1)
    rng_generator.seed(rseed);

    // Real ph = udist(rng_generator)*TWO_PI;
    int ipar = 0;
    for (int k = 0; k < npx3_loc; ++k) {
      Real zp1 = block_size.x3min + (k + 0.5) * dx3;
      for (int j = 0; j < npx2_loc; ++j) {
        Real yp1 = block_size.x2min + (j + 0.5) * dx2;
        for (int i = 0; i < npx1_loc; ++i) {
          Real xp1 = block_size.x1min + (i + 0.5) * dx1;
          ppar->xp(ipar) = xp1 + dx1 * (udist(rng_generator) - 0.5);
          ppar->yp(ipar) = yp1 + dx2 * (udist(rng_generator) - 0.5);
          ppar->zp(ipar) = zp1;
          if (mesh_size.nx3 > 1)
            ppar->zp(ipar) += dx3 * (udist(rng_generator) - 0.5);

          ppar->vpx(ipar) = 0.0;
          ppar->vpy(ipar) = 0.0;
          ppar->vpz(ipar) = 0.0;
          ++ipar;
        }
      }
    }

    // Initialize the stopping time.
    if (DustParticles::GetVariableTaus()) {
      Real taus0 = DustParticles::GetStoppingTime();
      for (int k = 0; k < npar; ++k)
        ppar->taus(k) = taus0;
    }
  }
}


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
}
