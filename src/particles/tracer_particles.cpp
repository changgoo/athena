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
#include "particles.hpp"

//--------------------------------------------------------------------------------------
//! \fn TracerParticles::TracerParticles(MeshBlock *pmb, ParameterInput *pin)
//! \brief constructs a TracerParticles instance.

TracerParticles::TracerParticles(MeshBlock *pmb, ParameterInput *pin,
  ParticleParameters *pp)
  : Particles(pmb, pin, pp) {
  // Add working array at particles for gas velocity/particle momentum change.
  iwx = AddWorkingArray();
  iwy = AddWorkingArray();
  iwz = AddWorkingArray();

  // allocate memory
  Particles::AllocateMemory();

  // Assign shorthands (need to do this for every constructor of a derived class)
  AssignShorthands();
}

//--------------------------------------------------------------------------------------
//! \fn TracerParticles::~TracerParticles()
//! \brief destroys a TracerParticles instance.

TracerParticles::~TracerParticles() {
  // nothing to do
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void TracerParticles::AddOneParticle()
//! \brief add one particle if position is within the mesh block

void TracerParticles::AddOneParticle(Real mp, Real x1, Real x2, Real x3,
  Real v1, Real v2, Real v3) {
  if (CheckInMeshBlock(x1,x2,x3)) {
    if (npar_ == nparmax_) UpdateCapacity(npar_*2);
    pid_(npar_) = -1;
    mass_(npar_) = mp;
    xp_(npar_) = x1;
    yp_(npar_) = x2;
    zp_(npar_) = x3;
    vpx_(npar_) = v1;
    vpy_(npar_) = v2;
    vpz_(npar_) = v3;
    npar_++;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void TracerParticles::AssignShorthands()
//! \brief assigns shorthands by shallow coping slices of the data.

void TracerParticles::AssignShorthands() {
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
  for (int k = 0; k < npar_; ++k) {
    Real x1, x2, x3;
    //! \todo (ccyang):
    //! - using (xp0, yp0, zp0) is a temporary hack.
    pc->CartesianToMeshCoords(xp0_(k), yp0_(k), zp0_(k), x1, x2, x3);
    pc->MeshCoordsToCartesianVector(x1, x2, x3, wx(k), wy(k), wz(k),
                                                wx(k), wy(k), wz(k));
  }

  // Tracer particles
  for (int k = 0; k < npar_; ++k) {
    Real tmpx = vpx_(k), tmpy = vpy_(k), tmpz = vpz_(k);
    vpx_(k) = wx(k);
    vpy_(k) = wy(k);
    vpz_(k) = wz(k);
    vpx0_(k) = tmpx; vpy0_(k) = tmpy; vpz0_(k) = tmpz;
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
