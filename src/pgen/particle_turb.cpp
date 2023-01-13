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
#include "../fft/perturbation.hpp"
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

TurbulenceDriver *ptrbd;
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

  if (pmy_mesh->particle) {
    for (Particles *ppar : ppars) {
      // Only assign particles either side of x
      // ipar == 0 for x<0
      // ipar == 1 for x>0
      Real xp1min = pin->GetReal(ppar->input_block_name,"x1min");
      Real xp1max = pin->GetReal(ppar->input_block_name,"x1max");
      // Find the total number of particles in each direction.
      RegionSize& mesh_size = pmy_mesh->mesh_size;
      Real np_per_cell = pin->GetOrAddReal(ppar->input_block_name, "np_per_cell",1);
      int npx1 = (block_size.nx1 > 1) ? static_cast<int>(mesh_size.nx1*np_per_cell) : 1;
      int npx2 = (block_size.nx2 > 1) ? static_cast<int>(mesh_size.nx2*np_per_cell) : 1;
      int npx3 = (block_size.nx3 > 1) ? static_cast<int>(mesh_size.nx3*np_per_cell) : 1;

      // Uniformly separted particles, uniform mass = total mass / N particles
      // Find the mass of each particle and the distance between adjacent particles.
      Real vol = mesh_size.x1len * mesh_size.x2len * mesh_size.x3len;
      Real dx1 = mesh_size.x1len / npx1,
           dx2 = mesh_size.x2len / npx2,
           dx3 = mesh_size.x3len / npx3;
      Real mpar;
      if (DustParticles *pp = dynamic_cast<DustParticles*>(ppar)) {
        Real dtog = pin->GetOrAddReal(pp->input_block_name,"dtog",1);
        mpar = dtog * d0 * vol / (npx1 * npx2 * npx3);
      } else if (TracerParticles *pp = dynamic_cast<TracerParticles*>(ppar)) {
        mpar = d0 * vol / (npx1 * npx2 * npx3);
      } else {
        std::stringstream msg;
        msg << "### FATAL ERROR in ProblemGenerator " << std::endl
            << " partype: " << ppar->partype
            << " is not supported" << std::endl;
        ATHENA_ERROR(msg);
        return;
      }

      // Determine number of particles in the block.
      int npx1_loc = static_cast<int>(std::round(block_size.x1len / dx1)),
          npx2_loc = static_cast<int>(std::round(block_size.x2len / dx2)),
          npx3_loc = static_cast<int>(std::round(block_size.x3len / dx3));
      int npar = npx1_loc * npx2_loc * npx3_loc;

      // Assign the particles.
      // Ramdomizing position. Or velocity perturbation
      std::random_device device;
      std::mt19937_64 rng_generator;
      // std::int64_t rseed = static_cast<std::int64_t>(device());
      std::int64_t rseed = gid;
      std::uniform_real_distribution<Real> udist(0.0,1.0); // uniform in [0,1)
      rng_generator.seed(rseed);

      // Real ph = udist(rng_generator)*TWO_PI;
      for (int k = 0; k < npx3_loc; ++k) {
        Real zp1 = block_size.x3min + (k + 0.5) * dx3;
        for (int j = 0; j < npx2_loc; ++j) {
          Real yp1 = block_size.x2min + (j + 0.5) * dx2;
          for (int i = 0; i < npx1_loc; ++i) {
            Real xp1 = block_size.x1min + (i + 0.5) * dx1;
            if ((xp1>xp1min) && (xp1<xp1max)) {
              Real xp = xp1 + dx1 * (udist(rng_generator) - 0.5);
              Real yp = yp1 + dx2 * (udist(rng_generator) - 0.5);
              Real zp = zp1;
              if (mesh_size.nx3 > 1)
                zp += dx3 * (udist(rng_generator) - 0.5);
              int pid = ppar->AddOneParticle(mpar, xp, yp, zp, 0.0, 0.0, 0.0);
              if (DustParticles *pp = dynamic_cast<DustParticles*>(ppar)) {
                if (pp->IsVariableTaus()) {
                  Real taus0 = pp->GetStoppingTime();
                  if (pid != -1) pp->taus(pid) = taus0;
                }
              }
            }
          }
        }
      }
    }
  }
}

//========================================================================================
//! \fn void Mesh::PostInitialize(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::PostInitialize(int res_flag, ParameterInput *pin) {
  ptrbd = new TurbulenceDriver(this, pin);

  if (((ptrbd->turb_flag == 1) || (ptrbd->turb_flag == 2)) && (res_flag == 0)) {
    ptrbd->Driving();
    if (ptrbd->turb_flag == 1) delete ptrbd;
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief
//========================================================================================
void Mesh::UserWorkInLoop() {
  if (ptrbd->turb_flag > 1) ptrbd->Driving(); // driven turbulence
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop()
//  \brief
//========================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  if (ptrbd->turb_flag > 1) delete ptrbd; // driven turbulence
}
