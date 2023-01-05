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

Real d0, mass;
Real DeltaRho(MeshBlock *pmb, int iout);
Real TotalMass(MeshBlock *pmb, int iout);
bool InBoundary(Real x, Real y, Real z, Real x1min, Real x1max,
  Real x2min, Real x2max, Real x3min, Real x3max);
//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = pin->GetOrAddReal("gravity","four_pi_G",1.0);
    Real eps = pin->GetOrAddReal("gravity","grav_eps", 0.0);
    SetFourPiG(four_pi_G);
    SetGravityThreshold(eps);
  }
  // std::stringstream hstr;
  // initial density
  d0 = pin->GetReal("problem","d0");

  AllocateUserHistoryOutput(2+Particles::num_particles);
  EnrollUserHistoryOutput(0, DeltaRho, "drhog");
  EnrollUserHistoryOutput(1, DeltaRho, "drhop");
  for (int ipar = 0; ipar < Particles::num_particles; ++ipar) {
    std::string head = "pm";
    head.append(std::to_string(ipar));
    EnrollUserHistoryOutput(ipar+2, TotalMass, (head+"-m").data());
  }
  return;
}

void Mesh::UserWorkInLoop() {
  Particles::FindDensityOnMesh(this, false);
//   // output history of selected particle(s)
//   for (int b = 0; b < nblocal; ++b) {
//     MeshBlock *pmb(my_blocks(b));
//     for (int ipar=0; ipar<Particles::num_particles; ++ipar) {
//       pmb->ppar[ipar]->OutputParticles((ncycle == 0),1234);
//     }
//   }
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
    // Ramdomizing position.
    std::random_device device;
    std::mt19937_64 rng_generator;
    // std::int64_t rseed = static_cast<std::int64_t>(device());
    std::int64_t rseed = gid; // deterministic for tests
    std::uniform_real_distribution<Real> udist(0.0,1.0); // uniform in [0,1)
    rng_generator.seed(rseed);

    int npartot = pin->GetInteger("problem","npartot");
    for (Particles *ppar : ppars) {

      // Update capacity of particle container
      // ppar->UpdateCapacity(static_cast<int>(npartot/Globals::nranks));

      // Assign the particles.
      RegionSize& mesh_size = pmy_mesh->mesh_size;
      Real radius = pin->GetOrAddReal(ppar->input_block_name,"radius",-1);
      if (radius == -1) {
        // Assign particles in each container to different regions
        Real xp1min = pin->GetReal(ppar->input_block_name,"x1min");
        Real xp1max = pin->GetReal(ppar->input_block_name,"x1max");
        Real xp2min = pin->GetReal(ppar->input_block_name,"x2min");
        Real xp2max = pin->GetReal(ppar->input_block_name,"x2max");

        // Find the total number of particles in each direction.
        Real vol = (xp1max-xp1min)*(xp2max-xp2min)*mesh_size.x3len;
        mass = d0 * vol / static_cast<Real>(npartot);
        for (int k = 0; k < npartot; ++k ) {
          // uniformly distributed within the mesh
          Real x = udist(rng_generator)*mesh_size.x1len + mesh_size.x1min;
          Real y = udist(rng_generator)*mesh_size.x2len + mesh_size.x2min;
          Real z = udist(rng_generator)*mesh_size.x3len + mesh_size.x3min;
          while (!(InBoundary(x,y,z,xp1min,xp1max,xp2min,xp2max,
                            mesh_size.x3min,mesh_size.x3max))) {
            x = udist(rng_generator)*mesh_size.x1len + mesh_size.x1min;
            y = udist(rng_generator)*mesh_size.x2len + mesh_size.x2min;
            z = udist(rng_generator)*mesh_size.x3len + mesh_size.x3min;
          }
          Real vy = vy0;
          if (!porb->orbital_advection_defined) vy -= qshear*Omega0*x;
          if (StarParticles *pp = dynamic_cast<StarParticles*>(ppar)) {
            pp->AddOneParticle(mass,x,y,z,vx0,vy,0.0);
          } else if (TracerParticles *pp = dynamic_cast<TracerParticles*>(ppar)) {
            pp->AddOneParticle(x,y,z,vx0,vy,0.0);
          } else {
            std::stringstream msg;
            msg << "### FATAL ERROR in ProblemGenerator " << std::endl
                << " partype: " << ppar->partype
                << " is not supported" << std::endl;
            ATHENA_ERROR(msg);
            return;
          }
        }
      } else {
        // Assign particles within a cylinder
        Real x0 = pin->GetOrAddReal(ppar->input_block_name,"x0",0.0);
        Real y0 = pin->GetOrAddReal(ppar->input_block_name,"y0",0.0);
        for (int k = 0; k < npartot; ++k ) {
          Real x = (udist(rng_generator)-0.5)*2*radius + x0;
          Real y = (udist(rng_generator)-0.5)*2*radius + y0;
          Real z = udist(rng_generator)*mesh_size.x3len + mesh_size.x3min;
          Real r = std::sqrt(x*x+y*y+z*z);
          Real vol = PI*radius*radius*mesh_size.x3len;
          mass = d0 * vol / static_cast<Real>(npartot);
          while (r>radius) {
            // reject particle outside the sphere
            x = (udist(rng_generator)-0.5)*2*radius + x0;
            y = (udist(rng_generator)-0.5)*2*radius + y0;
            z = udist(rng_generator)*mesh_size.x3len + mesh_size.x3min;
            r = std::sqrt(x*x+y*y+z*z);
          }
          Real vy = vy0;
          if(!porb->orbital_advection_defined) vy -= qshear*Omega0*x;
          if (StarParticles *pp = dynamic_cast<StarParticles*>(ppar)) {
            pp->AddOneParticle(mass,x,y,z,vx0,vy,0.0);
          } else if (TracerParticles *pp = dynamic_cast<TracerParticles*>(ppar)) {
            pp->AddOneParticle(x,y,z,vx0,vy,0.0);
          } else {
            std::stringstream msg;
            msg << "### FATAL ERROR in ProblemGenerator " << std::endl
                << " partype: " << ppar->partype
                << " is not supported" << std::endl;
            ATHENA_ERROR(msg);
            return;
          }
        }
      }

      if (ppar->npar > 0)
        std::cout << " ipar: " << ppar->ipar << " type: " << ppar->partype
                  << " nparmax: " << ppar->nparmax
                  << " npar: " << ppar->npar << " mass: " << mass << std::endl;

      // calculate PM density every substeps (for history dumps)
      // ppar->pm_stages[0] = true;
      // ppar->pm_stages[1] = true;
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
    for (Particles *ppar : pmb->ppars) {
      AthenaArray<Real> rhop(ppar->ppm->GetMassDensity());
      if (TracerParticles *pp = dynamic_cast<TracerParticles*>(ppar)) {
        for (int k=ks; k<=ke; ++k)
          for (int j=js; j<=je; ++j)
            for (int i=is; i<=ie; ++i)
              rhop(k,j,i) *= mass;
      }
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

//========================================================================================
//! \fn Real TotalMass(MeshBlock *pmb, int iout)
//! \brief Total particle mass from particles and partcle mesh
//========================================================================================
Real TotalMass(MeshBlock *pmb, int iout) {
  Real mtot{0};
  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> vol(pmb->ncells1);

  int ipar = iout-2;
  Particles *ppar = pmb->ppars[ipar];

  AthenaArray<Real> rhop(ppar->ppm->GetMassDensity());
  if (TracerParticles *pp = dynamic_cast<TracerParticles*>(ppar)) {
    for (int k=ks; k<=ke; ++k)
      for (int j=js; j<=je; ++j)
        for (int i=is; i<=ie; ++i)
          rhop(k,j,i) *= mass;
  }
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
      for (int i=is; i<=ie; ++i) {
        mtot += rhop(k,j,i)*vol(i);
      }
    }
  }

  return mtot;
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
