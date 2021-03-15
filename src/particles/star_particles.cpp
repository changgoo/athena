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

StarParticles::StarParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp)
  : Particles(pmb, pin, pp) {
  // Add particle mass, metallicity
  imass = AddRealProperty();
  imetal = AddRealProperty();

  // Add particle age
  iage = AddRealProperty();

  // Add gas fraction as aux peroperty
  igas = AddAuxProperty();

  if (SELF_GRAVITY_ENABLED) {
    isgravity_ = pp->gravity;
    pmy_mesh->particle_gravity = true;
    // Add working arrays for gravity forces
    igx = AddWorkingArray();
    igy = AddWorkingArray();
    igz = AddWorkingArray();
    // Activate particle gravity.
    ppgrav = new ParticleGravity(this);
  }

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
//! \fn void StarParticles::SetOneParticleMass(Real new_mass)
//! \brief sets the mass of each particle.

void StarParticles::SetOneParticleMass(Real new_mass) {
  pinput->SetReal(input_block_name, "mass", mass = new_mass);
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
    if (t>0) Kick(t,dt,pmy_block->phydro->w);
    SaveStatus();
    // kick from t^n to t^n+1/2
    Kick(t,dt,pmy_block->phydro->w);
    break;

  case 2:
    t = pmy_mesh->time + 0.5 * pmy_mesh->dt;
    dt = pmy_mesh->dt;
    // drift from t^n to t^n+1
    Drift(t,dt);
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
