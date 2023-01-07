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
#include "../fft/perturbation.hpp"
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

Real m0;
Real ParticleEnergy(MeshBlock *pmb, int iout);
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

  // set central mass
  m0 = pin->GetReal("problem","m0");

  // Enroll user-defined functions
  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, ParticleEnergy, "Ep1");
  EnrollUserHistoryOutput(1, ParticleEnergy, "Ep2");
  return;
}

//========================================================================================
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in MeshBlock class.  Can also be
//! used to initialize variables which are global to other functions in this file.
//! Called in MeshBlock constructor before ProblemGenerator.
//========================================================================================
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // Allocate storage for keeping track of particle energy
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(4);
  ruser_meshblock_data[0](0) = 0.0; // p1
  ruser_meshblock_data[0](1) = 0.0; // p2
  ruser_meshblock_data[0](2) = 0.0; // p3
  ruser_meshblock_data[0](3) = 0.0; // p4

  return;
}
//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // uniform background gas initialization
  Real d0 = pin->GetOrAddReal("problem", "d0", 1.0);
  // shearing box parameters
  Real qshear = pin->GetReal("orbital_advection","qshear");
  Real Omega0 = pin->GetReal("orbital_advection","Omega0");
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x1 = pcoord->x1v(i);
        phydro->u(IDN,k,j,i) = d0;

        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        if(!porb->orbital_advection_defined)
          phydro->u(IM2,k,j,i) -= d0*qshear*Omega0*x1;
        phydro->u(IM3,k,j,i) = 0.0;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = 1.0 + 0.5*(SQR(phydro->u(IM1,k,j,i)) +
                                            SQR(phydro->u(IM2,k,j,i)) +
                                            SQR(phydro->u(IM3,k,j,i))
                                          ) / phydro->u(IDN,k,j,i);
        }
      }
    }
  }

  // initialize particle
  if (pmy_mesh->particle) {
    if (pstarpar == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [MeshBlock::ProblemGenerator]" << std::endl
          << "This problem requires star particles." << std::endl;
      ATHENA_ERROR(msg);
    }

    Real x1 = pin->GetOrAddReal(pstarpar->input_block_name, "x1", 1.0);
    Real m1 = pin->GetOrAddReal(pstarpar->input_block_name, "m1", 1.0);
    Real v1 = pin->GetOrAddReal(pstarpar->input_block_name, "v1", 1.0);

    if (pmy_mesh->particle_gravity) {
      // self-gravitating two-body problem
      // have to turn off gravity from gas (modify gravity/block_fft_gravity)
      // and to gas (modify hydro/srcterms/hydro_srcterms)
      Real mratio = pin->GetOrAddReal(pstarpar->input_block_name, "mratio", 1.0);
      Real m2 = mratio*m1;
      Real mu = m1*m2/(m1+m2);
      Real x2 = -x1/mratio;
      Real vcirc = std::sqrt((m1+m2)/(x1-x2));
      Real v1 = vcirc*m2/(m1+m2);
      Real v2 = -v1/mratio;
      pstarpar->AddOneParticle(m1,x1,0.0,0.0,0.0,v1,0.0);
      pstarpar->AddOneParticle(m2,x2,0.0,0.0,0.0,v2,0.0);
    } else {
      // simple particle orbit tests
      if (pmy_mesh->shear_periodic) {
        // epicyclic motions
        Real x0 = 0.5;
        pstarpar->AddOneParticle(m1,x1,0.0,0.0,0.0,v1,0.0);
        pstarpar->AddOneParticle(m1,x0+x1,0.0,0.0,0.0,v1-x0,0.0);
      } else {
        // kepler orbits
        pstarpar->AddOneParticle(m1,x1,0.0,0.0,0.0,v1,0.0);
        pstarpar->AddOneParticle(m1,x1,0.0,0.0,0.0,v1*1.2,0.0);
        pstarpar->AddOneParticle(m1,x1,0.0,0.0,0.0,v1*0.8,0.0);
        pstarpar->AddOneParticle(m1,x1,0.0,0.0,0.0,v1*0.6,0.0);
      }
    }

//    std::cout << " nparmax: " << ppar->nparmax_ << " npar: " << ppar->npar_ << std::endl;
    // if (ppar->npar_>1) {
    //   ppar->OutputOneParticle(std::cout, 0, true);
    //   for (int ip=1; ip<ppar->npar_; ++ip)
    //     ppar->OutputOneParticle(std::cout, ip, false);
    // }
    pstarpar->ToggleParHstOutFlag();
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief
//========================================================================================
void Mesh::UserWorkInLoop() {
  for (int b = 0; b < nblocal; ++b) { // output particle history
    MeshBlock *pmb(my_blocks(b));
    for (Particles *ppar : pmb->ppars) ppar->OutputParticles(false);
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop()
//  \brief
//========================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop()
//! \brief calculate and store particle's total energy
//========================================================================================

void MeshBlock::UserWorkInLoop() {
  const Coordinates *pc = pcoord;
  Particles *ppar = pstarpar;
  const AthenaArray<int>& pid = ppar->pid();
  const AthenaArray<Real>& mass = ppar->mass();
  const AthenaArray<Real>& xp0 = ppar->xp0();
  const AthenaArray<Real>& yp0 = ppar->yp0();
  const AthenaArray<Real>& zp0 = ppar->zp0();
  const AthenaArray<Real>& vpx0 = ppar->vpx0();
  const AthenaArray<Real>& vpy0 = ppar->vpy0();
  const AthenaArray<Real>& vpz0 = ppar->vpz0();

  for (int k=0; k<ppar->GetNumPar(); ++k) {
    Real x1, x2, x3;
    pc->CartesianToMeshCoords(xp0(k), yp0(k), zp0(k), x1, x2, x3);
    Real Ek = 0.5*mass(k)*(SQR(vpx0(k)) + SQR(vpy0(k)) + SQR(vpz0(k)));
    Real phi;
    if (pmy_mesh->shear_periodic) {
      // simple particle orbit tests
      phi = -porb->qshear*SQR(porb->Omega0*x1);
    } else {
      // kepler orbits
      Real r = std::sqrt(x1*x1 + x2*x2 + x3*x3); // m0 is at (0,0,0)
      phi = -m0/r; // G=1
    }
    Real Etot = Ek + phi;
    ruser_meshblock_data[0](pid(k)-1) = Etot;
  }
  return;
}

//========================================================================================
//! \fn Real ParticleEnergy(MeshBlock *pmb, int iout)
//! \brief Particle's total energy
//========================================================================================
Real ParticleEnergy(MeshBlock *pmb, int iout) {
  Real Etot = pmb->ruser_meshblock_data[0](iout);
  pmb->ruser_meshblock_data[0](iout) = 0;
  return Etot;
}

//========================================================================================
//! \fn void StarParticles::UserSourceTerms(ParameterInput *pin)
//! \brief point mass acceleration acting only on particles for orbit tests
//========================================================================================

void StarParticles::UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  const Coordinates *pc = pmy_block->pcoord;
  for (int k = 0; k < npar_; ++k) {
    if (age(k) > 0) { // first kick (from n-1/2 to n) is skipped for the new particles
      Real x1, x2, x3;
      pc->CartesianToMeshCoords(xp_(k), yp_(k), zp_(k), x1, x2, x3);

      Real r = std::sqrt(x1*x1 + x2*x2 + x3*x3); // m0 is at (0,0,0)
      Real acc = -m0/(r*r); // G=1
      Real ax = acc*x1/r, ay = acc*x2/r, az = acc*x3/r;

      vpx_(k) = vpx_(k) + dt*ax;
      vpy_(k) = vpy_(k) + dt*ay;
      vpz_(k) = vpz_(k) + dt*az;
    }
  }
}
