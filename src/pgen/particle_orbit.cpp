//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_orbit.cpp
//! \brief Problem generator to test particle orbits
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

Real m0;
//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = pin->GetReal("self_gravity","four_pi_G");
    Real eps = pin->GetOrAddReal("self_gravity","grav_eps", 0.0);
    SetFourPiG(four_pi_G);
    SetGravityThreshold(eps);
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

  // set central mass
  m0 = pin->GetReal("problem","m0");
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // uniform background gas initialization
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

  // initialize particle
  if (PARTICLES) {
    if (!(ppar[0]->partype.compare("star") == 0)){
      std::stringstream msg;
      msg << "### FATAL ERROR in function [MeshBlock::ProblemGenerator]" << std::endl
          << "Only star particle is allowed. " << std::endl;
      ATHENA_ERROR(msg);
    }

    StarParticles *pp = dynamic_cast<StarParticles*>(ppar[0]);

    Real x1 = pin->GetOrAddReal(pp->input_block_name, "x1", 1.0);
    Real m1 = pin->GetOrAddReal(pp->input_block_name, "m1", 1.0);
    Real v1 = pin->GetOrAddReal(pp->input_block_name, "v1", 1.0);

    // Find the total number of particles in each direction.
    RegionSize& mesh_size = pmy_mesh->mesh_size;

    pp->AddOneParticle(m1,x1,0.0,0.0,0.0,v1,0.0);
    pp->AddOneParticle(m1,x1,0.0,0.0,0.0,v1*1.2,0.0);
    pp->AddOneParticle(m1,x1,0.0,0.0,0.0,v1*0.8,0.0);
    pp->AddOneParticle(m1,x1,0.0,0.0,0.0,v1*0.6,0.0);

    std::cout << " nparmax: " << pp->nparmax << " npar: " << pp->npar << std::endl;
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  return;
}

void Mesh::UserWorkInLoop() {
  int np = Particles::num_particles;
  for (int b = 0; b < nblocal; ++b) {
    MeshBlock *pmb(my_blocks(b));
    for (int i = 0; i < np; ++i) {
      Particles *ppar = pmb->ppar[i];
      ppar->OutputParticles();
    }
  }

  return;
}

void StarParticles::UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  const Coordinates *pc = pmy_block->pcoord;
  for (int k = 0; k < npar; ++k) {
    Real x1, x2, x3;
    pc->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);

    Real r = std::sqrt(x1*x1 + x2*x2 + x3*x3); // m0 is at (0,0,0)
    Real acc = -m0/(r*r); // G=1
    Real ax = acc*x1/r, ay = acc*x2/r, az = acc*x3/r;

    vpx(k) = vpx(k) + dt*ax;
    vpy(k) = vpy(k) + dt*ay;
    vpz(k) = vpz(k) + dt*az;
  }
}
