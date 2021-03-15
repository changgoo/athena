//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_grav.cpp
//! \brief Problem generator to test gravity on particles
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

Real DeltaRho(MeshBlock *pmb, int iout);
//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = pin->GetReal("problem","four_pi_G");
    Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
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
    if (!(ppar[0]->partype.compare("star") == 0)){
      std::stringstream msg;
      msg << "### FATAL ERROR in function [MeshBlock::ProblemGenerator]" << std::endl
          << "Only star particle is allowed. " << std::endl;
      ATHENA_ERROR(msg);
    }

    StarParticles *pp = dynamic_cast<StarParticles*>(ppar[0]);

    // Find the total number of particles in each direction.
    RegionSize& mesh_size = pmy_mesh->mesh_size;
    Real npartot = pin->GetOrAddReal(pp->input_block_name, "npartot",10000);
    Real Mtot = pin->GetOrAddReal("problem", "Mtot", 1);
    Real a = pin->GetOrAddReal("problem", "a", 1);
    Real m0 = Mtot/npartot;

    // Determine number of particles in the block.
    pp->UpdateCapacity(npartot);

    // Assign the particles.
    // Ramdomizing position. Or velocity perturbation
    std::random_device device;
    std::mt19937_64 rng_generator;
    // std::int64_t rseed = static_cast<std::int64_t>(device());
    std::int64_t rseed = gid;
    std::uniform_real_distribution<Real> udist(0.0,1.0); // uniform in [0,1)
    rng_generator.seed(rseed);

    // Real ph = udist(rng_generator)*TWO_PI;
    int ipid = 0;
    // generate global positions

    for (int k = 0; k < npartot; ++k ) {
      Real mr = std::pow(udist(rng_generator),2/3.);
      Real r = a*std::sqrt(mr/(1-mr));
      Real costh = (udist(rng_generator)-0.5)*2; // costh
      Real sinth = std::sqrt(1-SQR(costh));
      Real phi = (udist(rng_generator)*TWO_PI); // phi
      Real x1 = r*std::cos(phi)*sinth;
      Real x2 = r*std::sin(phi)*sinth;
      Real x3 = r*costh;
      // if inside the mesh block
      if ((x1>=block_size.x1min) && (x1<block_size.x1max) &&
          (x2>=block_size.x2min) && (x2<block_size.x2max) &&
          (x3>=block_size.x3min) && (x3<block_size.x3max)) {
        pp->mp(ipid) = m0;
        pp->mzp(ipid) = m0;
        pp->tage(ipid) = 0.0;

        pp->xp(ipid) = x1;
        pp->yp(ipid) = x2;
        pp->zp(ipid) = x3;

        pp->vpx(ipid) = 0.0;
        pp->vpy(ipid) = 0.0;
        pp->vpz(ipid) = 0.0;

        ipid++;
      }
    }
    pp->npar = ipid;
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  return;
}
