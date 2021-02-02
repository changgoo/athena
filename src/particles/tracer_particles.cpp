//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file tracer_particles.cpp
//! \brief implements functions in the TracerParticles class

// C++ headers
#include <algorithm>  // min()

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "particle_gravity.hpp"
#include "particles.hpp"

// Class variable initialization
bool TracerParticles::initialized(false);

//--------------------------------------------------------------------------------------
//! \fn void TracerParticles::Initialize(Mesh *pm, ParameterInput *pin)
//! \brief initializes the class.

void TracerParticles::Initialize(Mesh *pm, ParameterInput *pin) {
  // Initialize first the parent class.
  Particles::Initialize(pm, pin);

  if (!initialized) {
    initialized = true;
  }
}

//--------------------------------------------------------------------------------------
//! \fn TracerParticles::TracerParticles(MeshBlock *pmb, ParameterInput *pin)
//! \brief constructs a TracerParticles instance.

TracerParticles::TracerParticles(MeshBlock *pmb, ParameterInput *pin)
  : Particles(pmb, pin) {
  // Add working array at particles for gas velocity/particle momentum change.
  iwx = AddWorkingArray();
  iwy = AddWorkingArray();
  iwz = AddWorkingArray();

  // Re-Allocate working arrays.
  work.DeleteAthenaArray();
  work.NewAthenaArray(nwork,nparmax);

  // Assign shorthands (need to do this for every constructor of a derived class)
  AssignShorthands();
  Particles::PrintVariables();
}

//--------------------------------------------------------------------------------------
//! \fn TracerParticles::~TracerParticles()
//! \brief destroys a TracerParticles instance.

TracerParticles::~TracerParticles() {
  wx.DeleteAthenaArray();
  wy.DeleteAthenaArray();
  wz.DeleteAthenaArray();
}
//--------------------------------------------------------------------------------------
//! \fn void TracerParticles::AssignShorthands()
//! \brief assigns shorthands by shallow coping slices of the data.

void TracerParticles::AssignShorthands() {
  // std::cout << "assign shorthands called " << std::endl;
  Particles::AssignShorthands();
  wx.InitWithShallowSlice(work, 2, iwx, 1);
  wy.InitWithShallowSlice(work, 2, iwy, 1);
  wz.InitWithShallowSlice(work, 2, iwz, 1);
}

//--------------------------------------------------------------------------------------
//! \fn void TracerParticles::SourceTerms()
//! \brief adds acceleration to particles.

void TracerParticles::SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
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

  // Tracer particles
  for (int k = 0; k < npar; ++k) {
    vpx(k) = wx(k);
    vpy(k) = wy(k);
    vpz(k) = wz(k);
  }

  return;
}

//--------------------------------------------------------------------------------------
//! \fn void TracerParticles::UserSourceTerms(Real t, Real dt,
//!                                         const AthenaArray<Real>& meshsrc)
//! \brief adds additional source terms to particles, overloaded by the user.

void __attribute__((weak)) TracerParticles::UserSourceTerms(
    Real t, Real dt, const AthenaArray<Real>& meshsrc) {
}

//--------------------------------------------------------------------------------------
//! \fn void TracerParticles::ReactToMeshAux(
//!              Real t, Real dt, const AthenaArray<Real>& meshsrc)
//! \brief Reacts to meshaux before boundary communications.

void TracerParticles::ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  // Nothing to do for tracers
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void TracerParticles::DepositToMesh(Real t, Real dt,
//!              const AthenaArray<Real>& meshsrc, AthenaArray<Real>& meshdst);
//! \brief Deposits meshaux to Mesh.

void TracerParticles::DepositToMesh(
         Real t, Real dt, const AthenaArray<Real>& meshsrc, AthenaArray<Real>& meshdst) {
  // Nothing to do for tracers
  return;
}
