//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_uniform_motion.cpp
//! \brief Problem generator for uniformly moving particles (multicontainer tracers)
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
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../particles/particles.hpp"
#include "../utils/utils.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

Real d0;
Real DeltaRho(MeshBlock *pmb, int iout);
bool InBoundary(Real x, Real y, Real z, Real x1min, Real x1max,
  Real x2min, Real x2max, Real x3min, Real x3max);
//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  // std::stringstream hstr;
  // initial density
  d0 = pin->GetReal("problem","d0");

  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, DeltaRho, "drhog");
  EnrollUserHistoryOutput(1, DeltaRho, "drhop");

  return;
}

void Mesh::UserWorkInLoop() {
  Particles::FindDensityOnMesh(this, false);
  // output history of selected particle(s)
  // for (int b = 0; b < nblocal; ++b) {
  //   MeshBlock *pmb(my_blocks(b));
  //   for (int ipar=0; ipar<Particles::num_particles; ++ipar) {
  //     pmb->ppar[ipar]->OutputParticles((ncycle == 0),1234);
  //   }
  // }
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // initial translational velocity
  Real vx0 = pin->GetReal("problem","vx0");
  Real vy0 = pin->GetReal("problem","vy0");
  // shearing box parameters
  Real qshear = pin->GetReal("orbital_advection","qshear");
  Real Omega0 = pin->GetReal("orbital_advection","Omega0");
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x1 = pcoord->x1v(i);

        phydro->u(IDN,k,j,i) = d0;

        phydro->u(IM1,k,j,i) = d0*vx0;
        phydro->u(IM2,k,j,i) = d0*vy0;
        phydro->u(IM3,k,j,i) = 0.0;
        if(!porb->orbital_advection_defined)
          phydro->u(IM2,k,j,i) -= d0*qshear*Omega0*x1;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = 1.0*d0;
          phydro->u(IEN,k,j,i) += (SQR(phydro->u(IM1,k,j,i))
                                  +SQR(phydro->u(IM2,k,j,i))
                                  +SQR(phydro->u(IM3,k,j,i)))/(2.0*d0);
        }
      }
    }
  }

  if (pmy_mesh->particle) {
    for (int ipar = 0; ipar < Particles::num_particles; ++ipar) {
      // Assign particles in each container to different regions
      Real xp1min = pin->GetReal(ppar[ipar]->input_block_name,"x1min");
      Real xp1max = pin->GetReal(ppar[ipar]->input_block_name,"x1max");
      Real xp2min = pin->GetReal(ppar[ipar]->input_block_name,"x2min");
      Real xp2max = pin->GetReal(ppar[ipar]->input_block_name,"x2max");

      // Find the total number of particles in each direction.
      RegionSize& mesh_size = pmy_mesh->mesh_size;
      int npartot = pin->GetInteger("problem","npartot");
      // pin->GetOrAddReal(ppar[ipar]->input_block_name, "npartot",100);

      // Update capacity of particle container
      ppar[ipar]->UpdateCapacity(npartot);

      // Assign the particles.
      // Ramdomizing position.
      std::random_device device;
      std::mt19937_64 rng_generator;
      // std::int64_t rseed = static_cast<std::int64_t>(device());
      std::int64_t rseed = gid; // deterministic for tests
      std::uniform_real_distribution<Real> udist(0.0,1.0); // uniform in [0,1)
      rng_generator.seed(rseed);

      for (int k = 0; k < npartot; ++k ) {
        // uniformly distributed within the
        Real x = udist(rng_generator)*mesh_size.x1len + mesh_size.x1min;
        Real y = udist(rng_generator)*mesh_size.x2len + mesh_size.x2min;
        Real z = udist(rng_generator)*mesh_size.x3len + mesh_size.x3min;
        while (!(InBoundary(x,y,z,xp1min,xp1max,xp2min,xp2max,
                           mesh_size.x3min,mesh_size.x3max))) {
          x = udist(rng_generator)*mesh_size.x1len + mesh_size.x1min;
          y = udist(rng_generator)*mesh_size.x2len + mesh_size.x2min;
          z = udist(rng_generator)*mesh_size.x3len + mesh_size.x3min;
        }
        ppar[ipar]->AddOneParticle(x,y,z,vx0,vy0,0.0);
      }

      Real vol = (xp1max-xp1min)*(xp2max-xp2min)*mesh_size.x3len;
      if (TracerParticles *pp = dynamic_cast<TracerParticles*>(ppar[ipar])) {
        pp->SetOneParticleMass(d0 * vol / static_cast<Real>(npartot));
      } else {
        std::stringstream msg;
        msg << "### FATAL ERROR in ProblemGenerator " << std::endl
            << " partype: " << ppar[ipar]->partype << "is not tracer" << std::endl;
        ATHENA_ERROR(msg);
        return;
      }

      std::cout << " ipar: " << ipar << " type: " << ppar[ipar]->partype
                << " nparmax: " << ppar[ipar]->nparmax
                << " npar: " << ppar[ipar]->npar << std::endl;

      // calculate PM density every substeps (for history dumps)
      // ppar[ipar]->pm_stages[0] = true;
      // ppar[ipar]->pm_stages[1] = true;
    }
  }
}

//========================================================================================
//! \fn Real DeltaRho(MeshBlock *pmb, int iout)
//! \brief Difference in gas and trace densities for history variable
//========================================================================================
Real DeltaRho(MeshBlock *pmb, int iout) {
  Real l1_err{0};
  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> vol(pmb->ncells1);
  AthenaArray<Real> rho;
  if (iout == 0) {
    rho.InitWithShallowSlice(pmb->phydro->u,4,IDN,1);
  } else {
    rho.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
    for (int ipar=0; ipar<Particles::num_particles; ++ipar) {
      AthenaArray<Real> rhop(pmb->ppar[ipar]->GetMassDensity());
      for (int k=ks; k<=ke; ++k)
        for (int j=js; j<=je; ++j)
          for (int i=is; i<=ie; ++i)
            rho(k,j,i) += rhop(k,j,i);
    }
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
      for (int i=is; i<=ie; ++i) {
        Real drho;
        drho = rho(k,j,i) - d0;
        l1_err += std::abs(drho)*vol(i);
      }
    }
  }

  return l1_err;
}

//--------------------------------------------------------------------------------------
//! \fn void InBoundary()
//! \brief check whether given position is within given boundary

bool InBoundary(Real x, Real y, Real z, Real x1min, Real x1max,
  Real x2min, Real x2max, Real x3min, Real x3max) {
  if ((x>=x1min) && (x<x1max) &&
      (y>=x2min) && (y<x2max) &&
      (z>=x3min) && (z<x3max)) {
    return true;
  } else {
    return false;
  }
}
