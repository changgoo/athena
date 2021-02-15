//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file star_particles.cpp
//! \brief implements functions in the starParticles class

// C++ headers
#include <algorithm>  // min()

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "particle_gravity.hpp"
#include "particles.hpp"

//--------------------------------------------------------------------------------------
//! \fn StarParticles::StarParticles(MeshBlock *pmb, ParameterInput *pin)
//! \brief constructs a StarParticles instance.

StarParticles::StarParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParmeters *pp)
  : Particles(pmb, pin, pp) {
  // Add working array at particles for gas velocity/particle momentum change.
  iwx = AddWorkingArray();
  iwy = AddWorkingArray();
  iwz = AddWorkingArray();

  // Re-Allocate working arrays.
  work.DeleteAthenaArray();
  work.NewAthenaArray(nwork,nparmax);

  // Define mass.
  mass = pin->GetOrAddReal(input block_name, "mass", 1.0);

  // Assign shorthands (need to do this for every constructor of a derived class)
  AssignShorthands();
}

//--------------------------------------------------------------------------------------
//! \fn StarParticles::~StarParticles()
//! \brief destroys a StarParticles instance.

StarParticles::~StarParticles() {
  // nothing to do
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::SetOneParticleMass(Real new_mass)
//! \brief sets the mass of each particle.

void StarParticles::SetOneParticleMass(Real new_mass) {
  pinput->SetReal("particles", "mass", mass = new_mass);
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::AssignShorthands()
//! \brief assigns shorthands by shallow coping slices of the data.

void StarParticles::AssignShorthands() {
  Particles::AssignShorthands();
  wx.InitWithShallowSlice(work, 2, iwx, 1);
  wy.InitWithShallowSlice(work, 2, iwy, 1);
  wz.InitWithShallowSlice(work, 2, iwz, 1);
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::SourceTerms()
//! \brief adds acceleration to particles.

void StarParticles::SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  ppm->InterpolateMeshToParticles(meshsrc, IVX, work, iwx, 3);

  // Transform the gas velocity into Cartesian.
  const Coordinates *pc = pmy_block->pcoord;
  for (int k = 0; k < npar; ++k) {
    Real x1, x2, x3;
    //! \todo (ccyang):
    //! - using (xp0, yp0, zp0) is a temporary hack.
    pc->CartesianToMeshCoords(xp0(k), yp0(k), zp0(k), x1, x2, x3);
    pc->MeshCoordsToCartesianVector(x1, x2, x3, wx(k), wy(k), wz(k),
                                                wx(k), wy(k), wz(k));
  }

  // Star particles
  for (int k = 0; k < npar; ++k) {
    vpx(k) = wx(k);
    vpy(k) = wy(k);
    vpz(k) = wz(k);
  }

  return;
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::UserSourceTerms(Real t, Real dt,
//!                                         const AthenaArray<Real>& meshsrc)
//! \brief adds additional source terms to particles, overloaded by the user.

void __attribute__((weak)) StarParticles::UserSourceTerms(
    Real t, Real dt, const AthenaArray<Real>& meshsrc) {
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::ReactToMeshAux(
//!              Real t, Real dt, const AthenaArray<Real>& meshsrc)
//! \brief Reacts to meshaux before boundary communications.

void StarParticles::ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  // Nothing to do for stars
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::DepositToMesh(Real t, Real dt,
//!              const AthenaArray<Real>& meshsrc, AthenaArray<Real>& meshdst);
//! \brief Deposits meshaux to Mesh.

void StarParticles::DepositToMesh(
         Real t, Real dt, const AthenaArray<Real>& meshsrc, AthenaArray<Real>& meshdst) {
  // Nothing to do for stars
  return;
}
