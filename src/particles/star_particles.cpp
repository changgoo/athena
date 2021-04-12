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
#include "particles.hpp"

//--------------------------------------------------------------------------------------
//! \fn StarParticles::StarParticles(MeshBlock *pmb, ParameterInput *pin)
//! \brief constructs a StarParticles instance.

StarParticles::StarParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp)
  : Particles(pmb, pin, pp), imetal(-1), iage(-1), igas(-1) {
  // Add particle mass, metal mass
  imass = AddRealProperty();
  imetal = AddRealProperty();
  realfieldname.push_back("mass");
  realfieldname.push_back("metal");

  // Add particle age
  iage = AddRealProperty();
  realfieldname.push_back("age");

  // Add gas fraction as aux peroperty
  igas = AddAuxProperty();
  realfieldname.push_back("fgas");

  // allocate memory
  Particles::AllocateMemory();

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
//! \fn void StarParticles::AssignShorthands()
//! \brief assigns shorthands by shallow coping slices of the data.

void StarParticles::AssignShorthands() {
  Particles::AssignShorthands();
  mp.InitWithShallowSlice(realprop, 2, imass, 1);
  mzp.InitWithShallowSlice(realprop, 2, imetal, 1);
  tage.InitWithShallowSlice(realprop, 2, iage, 1);

  fgas.InitWithShallowSlice(auxprop, 2, igas, 1);
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::AddOneParticle()
//! \brief add one particle if position is within the mesh block

void StarParticles::AddOneParticle(Real mass, Real x1, Real x2, Real x3,
  Real v1, Real v2, Real v3) {
  if (Particles::CheckInMeshBlock(x1,x2,x3)) {
    if (npar == nparmax) Particles::UpdateCapacity(npar*2);
    mp(npar) = mass;
    xp(npar) = x1;
    yp(npar) = x2;
    zp(npar) = x3;
    vpx(npar) = v1;
    vpy(npar) = v2;
    vpz(npar) = v3;

    // initialize other properties
    mzp(npar) = mass;
    tage(npar) = 0.0;
    fgas(npar) = 0.0;

    npar++;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Integrate(int step)
//! \brief updates all particle positions and velocities from t to t + dt.
//!
//! KDK Leapflog algorithm; assuming integrator=vl2
//! - kick from n-1/2->n+1/2 is done in stage 1
//! - drift from n->n+1 is done in stage 2
//! - temporary half time kick has to be done from n+1/2->n+1
//!   for output (use ApplyUserWorkBeforeOutput or write an explicit function for it)
void StarParticles::Integrate(int stage) {
  Real t = 0, dt = 0;

  // Determine the integration cofficients.
  switch (stage) {
  case 1:
    t = pmy_mesh->time;
    dt = 0.5*pmy_mesh->dt;

    // Calculate force on particles
    if (SELF_GRAVITY_ENABLED) {
      ppgrav->FindGravitationalForce(pmy_block->pgrav->phi);
      ppgrav->InterpolateGravitationalForce();
    }
    // kick from t^n-1/2 to t^n
    // if(t>0) Kick(t,0.5*dt,pmy_block->phydro->w);
    SaveStatus();
    // kick from t^n to t^n+1/2
    Kick(t,dt,pmy_block->phydro->w);
    break;

  case 2:
    t = pmy_mesh->time + 0.5 * pmy_mesh->dt;
    dt = pmy_mesh->dt;
    // drift from t^n to t^n+1
    Drift(t,dt);
    // kick from t^n+1/2 to t^n+1
    Kick(t,0.5*dt,pmy_block->phydro->w);
    Age(t,dt);
    break;
  }

  // this can be used for feedback
  ReactToMeshAux(t, dt, pmy_block->phydro->w);

  // Update the position index.
  SetPositionIndices();
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Age(Real t, Real dt)
//! \brief aging particles

void StarParticles::Age(Real t, Real dt) {
  // aging particles
  for (int k = 0; k < npar; ++k) tage(k) += dt;
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Drift(Real t, Real dt)
//! \brief drift particles

void StarParticles::Drift(Real t, Real dt) {
  // drift position
  for (int k = 0; k < npar; ++k) {
    xp(k) = xp0(k) + dt * vpx(k);
    yp(k) = yp0(k) + dt * vpy(k);
    zp(k) = zp0(k) + dt * vpz(k);
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Kick(Real t, Real dt)
//! \brief kick particles

void StarParticles::Kick(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  // Integrate the source terms (e.g., acceleration).
  SourceTerms(t, dt, meshsrc);
  UserSourceTerms(t, dt, meshsrc);
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::SourceTerms()
//! \brief adds acceleration to particles.

void StarParticles::SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  if (SELF_GRAVITY_ENABLED) ppgrav->ExertGravitationalForce(dt);
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::FindLocalDensityOnMesh(Mesh *pm, bool include_momentum)
//! \brief finds the number/mass density of particles on the mesh.
//!
//!   If include_momentum is true, the momentum density field is also computed,
//!   assuming mass of each particle is unity.
//! \note
//!   Postcondition: ppm->weight becomes the density in each cell, and
//!   if include_momentum is true, ppm->meshaux(imom1:imom3,:,:,:)
//!   becomes the momentum density.

void StarParticles::FindLocalDensityOnMesh(bool include_momentum) {
  Coordinates *pc(pmy_block->pcoord);

  if (include_momentum) {
    AthenaArray<Real> vp, vp1, vp2, vp3, mpar;
    vp.NewAthenaArray(4, npar);
    vp1.InitWithShallowSlice(vp, 2, 0, 1);
    vp2.InitWithShallowSlice(vp, 2, 1, 1);
    vp3.InitWithShallowSlice(vp, 2, 2, 1);
    mpar.InitWithShallowSlice(vp, 2, 3, 1);
    for (int k = 0; k < npar; ++k) {
      pc->CartesianToMeshCoordsVector(xp(k), yp(k), zp(k),
        mp(k)*vpx(k), mp(k)*vpy(k), mp(k)*vpz(k), vp1(k), vp2(k), vp3(k));
      mpar(k) = mp(k);
    }
    ppm->AssignParticlesToMeshAux(vp, 0, ppm->imom1, 4);
  } else {
    ppm->AssignParticlesToMeshAux(mp, 0, ppm->imass, 1);
  }
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
