//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_isosheet.cpp
//! \brief Problem generator for Spitzer Isothermal sheet for both star and gas
//========================================================================================

// C headers

// C++ headers
#include <cmath>     // sqrt()
#include <ctime>
#include <iomanip>   // setprecision, scientific
#include <iostream>  // cout, endl
#include <random>
#include <sstream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace {
Real z0, rho0_gas, rho0_star;
}

Real DeltaRho(MeshBlock *pmb, int iout);
// Declarations
void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties
//======================================================================================
//
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = pin->GetReal("self_gravity","four_pi_G");
    Real eps = pin->GetOrAddReal("self_gravity","grav_eps", 0.0);
    SetFourPiG(four_pi_G);
    SetGravityThreshold(eps);
  }

  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, FixedBoundary);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, FixedBoundary);
  }

  Real cs = pin->GetReal("hydro","iso_sound_speed");
  Real fgas = pin->GetReal("problem","fgas");
  Real d0 = pin->GetOrAddReal("problem","rho0",1.0);
  z0 = cs/std::sqrt(2.0*four_pi_G_*d0);
  rho0_gas = d0*fgas;
  rho0_star = d0*(1-fgas);

  // Enroll user-defined functions
  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, DeltaRho, "drhog");
  EnrollUserHistoryOutput(1, DeltaRho, "drhop");
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real sech2 = 1/SQR(std::cosh(0.5*x3/z0));
        phydro->u(IDN,k,j,i) = rho0_gas*sech2;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
      }
    }
  }

  if (pmy_mesh->particle && (rho0_star > 0.0)) {
    Real sigz = pin->GetReal("problem","sigma_star");

    Particles *ppar = ppars[0];
    if (!((ppar->partype).compare("star") == 0)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [MeshBlock::ProblemGenerator]" << std::endl
          << "Only star particle is allowed. " << std::endl;
      ATHENA_ERROR(msg);
    }

    // Find the total number of particles in each direction.
    RegionSize& mesh_size = pmy_mesh->mesh_size;
    int npartot = pin->GetInteger(ppar->input_block_name, "npartot");

    // Assign the particles.
    // Ramdomizing position. Or velocity perturbation
    std::random_device device;
    std::mt19937_64 rng_generator;
    // std::int64_t rseed = static_cast<std::int64_t>(device());
    std::int64_t rseed = gid;
    std::uniform_real_distribution<Real> udist(0.0,1.0); // uniform in [0,1)
    std::normal_distribution<Real> ndist(0.0,sigz); // normal distribution

    rng_generator.seed(rseed);

    // Real dz = mesh_size.x3len/static_cast<Real>(mesh_size.nx3);
    Real Surf = 4.0*z0*rho0_star;
    Real mpar = Surf*mesh_size.x1len*mesh_size.x2len/static_cast<Real>(npartot);
    for (int k = 0; k < npartot; ++k ) {
      int sgn = udist(rng_generator) > 0.5 ? 1 : -1;
      Real vx = ndist(rng_generator);
      Real vy = ndist(rng_generator);
      Real vz = ndist(rng_generator);
      Real x = udist(rng_generator)*mesh_size.x1len + mesh_size.x1min;
      Real y = udist(rng_generator)*mesh_size.x2len + mesh_size.x2min;
      Real z = z0*std::atanh(udist(rng_generator))*2.0*sgn;
      ppar->AddOneParticle(mpar,x,y,z,vx,vy,vz);
    }

//    std::cout << "[Problem IsoSheet] nparmax: " << ppar->nparmax_ << " npar: " << ppar->npar_
//              << " mpar: " << mpar << std::endl;
  }
  return;
}

//========================================================================================
//! \fn Real DeltaRho(MeshBlock *pmb, int iout)
//! \brief Difference in gas and trace densities for history variable
//========================================================================================
Real DeltaRho(MeshBlock *pmb, int iout) {
  Real l1_err{0};

  if ((!pmb->pmy_mesh->particle || (rho0_star==0)) && (iout == 1)) return l1_err;

  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> vol(pmb->ncells1);
  AthenaArray<Real> rho;
  rho.InitWithShallowSlice(pmb->phydro->u,4,IDN,1);
  AthenaArray<Real> rhop(pmb->ppars[0]->ppm->GetMassDensity());
  for (int k=ks; k<=ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
      for (int i=is; i<=ie; ++i) {
        Real sech2 = 1/SQR(std::cosh(0.5*x3/z0));
        Real drho;
        if (iout == 0) { // drho_gas
          drho = rho(k,j,i) - rho0_gas*sech2;
        } else if (iout == 1) { //}
          drho = rhop(k,j,i) - rho0_star*sech2;
        } else {
          std::stringstream msg;
          msg << "### FATAL ERROR in function [DeltaRho]"
              << std::endl << "iout: " << iout << "not allowed" <<std::endl;
          ATHENA_ERROR(msg);
        }
        l1_err += std::abs(drho)*vol(i);
      }
    }
  }
  return l1_err;
}

//----------------------------------------------------------------------------------------
// Fixed boundary condition
// Inputs:
//   pmb: pointer to MeshBlock
//   pcoord: pointer to Coordinates
//   time,dt: current time and timestep of simulation
//   is,ie,js,je,ks,ke: indices demarkating active region
// Outputs:
//   prim: primitives set in ghost zones
//   bb: face-centered magnetic field set in ghost zones
// Notes:
//   does nothing

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  return;
}
