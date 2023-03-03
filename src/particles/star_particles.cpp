//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file star_particles.cpp
//! \brief implements functions in the StarParticles class

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
  : Particles(pmb, pin, pp), dt_old(0), imetal(-1), iage(-1), ifgas(-1) {
  // Add metal mass
  imetal = AddRealProperty();
  realpropname.push_back("metal");

  // Add particle age
  iage = AddRealProperty();
  realpropname.push_back("age");

  // Add gas fraction as aux peroperty
  ifgas = AddAuxProperty();
  auxpropname.push_back("fgas");

  // Allocate memory and assign shorthands (shallow slices).
  // Every derived Particles need to call these two functions.
  AllocateMemory();
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
//! \fn void StarParticles::AssignShorthandsForDerived()
//! \brief assigns shorthands by shallow coping slices of the data.

void StarParticles::AssignShorthandsForDerived() {
  metal.InitWithShallowSlice(realprop, 2, imetal, 1);
  age.InitWithShallowSlice(realprop, 2, iage, 1);

  fgas.InitWithShallowSlice(auxprop, 2, ifgas, 1);
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Integrate(int step)
//! \brief updates all particle positions and velocities from t to t + dt.
//!
//! KDK Leapflog algorithm with Boris push; assuming integrator=vl2
//! - kick from n-1/2->n+1/2 is done in stage 1
//! - drift from n->n+1 is done in stage 2
//! - temporary half time kick has to be done from n+1/2->n+1
//!   for output (use ApplyUserWorkBeforeOutput or write an explicit function for it)
void StarParticles::Integrate(int stage) {
  Real t = 0, dt = 0, dth = 0;

  // Determine the integration cofficients.
  switch (stage) {
  case 1:
    t = pmy_mesh_->time;
    dt = pmy_mesh_->dt; // t^(n+1)-t^n;
    dth = 0.5*(dt + dt_old); // t^(n+1/2)-t^(n-1/2)

    // Calculate force on particles at t = t^n
    if (SELF_GRAVITY_ENABLED) {
      ppgrav->FindGravitationalForce(pmy_block->pgrav->phi);
      ppgrav->InterpolateGravitationalForce();
    }

    // closing kick by 0.5*dt_old
    // this kick has to be skipped for new particles
    Kick(t,0.5*dt_old,pmy_block->phydro->w);
    // x -> x0, v -> v0
    SaveStatus();
    // a temporary heck; later we will have a flag for new particles
    // aging first to distinguish new particle
    Age(t,dt);
    // Boris push for velocity dependent terms: Coriolis force
    if (pmy_mesh_->shear_periodic) BorisKick(t,dth);
    // opening kick by 0.5*dt
    Kick(t,0.5*dt,pmy_block->phydro->w);
    // drift from t^n to t^n+1
    Drift(t,dt);

    dt_old = dt; // save dt for the future use
    // Update the position index.
    UpdatePositionIndices();
    break;
  case 2:
    // particle --> mesh
    ReactToMeshAux(t, dt, pmy_block->phydro->w);
    break;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Age(Real t, Real dt)
//! \brief aging particles

void StarParticles::Age(Real t, Real dt) {
  // aging particles
  for (int k = 0; k < npar_; ++k) age(k) += dt;
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Drift(Real t, Real dt)
//! \brief drift particles

void StarParticles::Drift(Real t, Real dt) {
  // drift position
  for (int k = 0; k < npar_; ++k) {
    xp(k) = xp0(k) + dt * vpx(k);
    yp(k) = yp0(k) + dt * vpy(k);
    zp(k) = zp0(k) + dt * vpz(k);
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Kick(Real t, Real dt)
//! \brief kick particles
//!
//! forces from self gravity, external gravity, and tidal potential
//! Coriolis force is treated by BorisKick
//! dt must be the half dt

void StarParticles::Kick(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  // Integrate the source terms (e.g., acceleration).
  SourceTerms(t, dt, meshsrc);
  UserSourceTerms(t, dt, meshsrc);
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::BorisKick(Real t, Real dt)
//! \brief Coriolis force with Boris algorithm
//!
//! symmetric force application using the Boris algorithm
//! velocity must be updated to the midpoint before this call
//! dt must be the full dt (t^n+1/2 - t^n-1/2)

void StarParticles::BorisKick(Real t, Real dt) {
  Real Omdt = Omega_0_*dt, hOmdt = 0.5*Omdt;
  Real Omdt2 = SQR(Omdt), hOmdt2 = 0.25*Omdt2;
  Real f1 = (1-Omdt2)/(1+Omdt2), f2 = 2*Omdt/(1+Omdt2);
  Real hf1 = (1-hOmdt2)/(1+hOmdt2), hf2 = 2*hOmdt/(1+hOmdt2);
  for (int k = 0; k < npar_; ++k) {
    Real vpxm = vpx(k), vpym = vpy(k);
    if (age(k) == 0) { // for the new particles
      vpx(k) = hf1*vpxm + hf2*vpym;
      vpy(k) = -hf2*vpxm + hf1*vpym;
    } else {
      vpx(k) = f1*vpxm + f2*vpym;
      vpy(k) = -f2*vpxm + f1*vpym;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::ExertTidalForce(Real t, Real dt)
//! \brief force from tidal potential (qshear != 0)
//!
//! Phi_tidal = - q Omega^2 x^2
//! acc = 2 q Omega^2 x xhat
//! \note first kick (from n-1/2 to n) is skipped for the new particles

void StarParticles::ExertTidalForce(Real t, Real dt) {
  Real acc0 = 2*qshear_*SQR(Omega_0_);
  for (int k = 0; k < npar_; ++k) {
    Real acc = age(k) > 0 ? acc0*dt*xp(k) : 0.;
    vpx(k) += acc;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::PointMass(Real t, Real dt)
//! \brief force from a point mass at origin
//! \note first kick (from n-1/2 to n) is skipped for the new particles

void StarParticles::PointMass(Real t, Real dt, Real gm) {
  const Coordinates *pc = pmy_block->pcoord;
  for (int k = 0; k < npar_; ++k) {
    if (age(k) > 0) {
      Real x1, x2, x3;
      pc->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);

      Real r = std::sqrt(x1*x1 + x2*x2 + x3*x3); // m0 is at (0,0,0)
      Real acc = -gm/(r*r); // G=1
      Real ax = acc*x1/r, ay = acc*x2/r, az = acc*x3/r;

      vpx(k) += dt*ax;
      vpy(k) += dt*ay;
      vpz(k) += dt*az;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::ConstantAcceleration(Real t, Real dt)
//! \brief constant acceleration

void StarParticles::ConstantAcceleration(Real t, Real dt, Real g1, Real g2, Real g3) {
  for (int k = 0; k < npar_; ++k) {
    if (age(k) > 0) { // first kick (from n-1/2 to n) is skipped for the new particles
      vpx(k) += dt*g1;
      vpy(k) += dt*g2;
      vpz(k) += dt*g3;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::SourceTerms()
//! \brief adds acceleration to particles.
//!
//! star particles will feel all forces that gas feels
//! \note first kick (from n-1/2 to n) is skipped for the new particles

void StarParticles::SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  Hydro *ph = pmy_block->phydro;
  // accleration due to point mass (MUST BE AT ORIGIN)
  Real gm = ph->hsrc.GetGM();
  if (gm != 0) PointMass(t,dt,gm);

  // constant acceleration (e.g. for RT instability)
  Real g1 = ph->hsrc.GetG1(), g2 = ph->hsrc.GetG2(), g3 = ph->hsrc.GetG3();
  if (g1 != 0.0 || g2 != 0.0 || g3 != 0.0)
    ConstantAcceleration(t,dt,g1,g2,g3);

  if (pmy_mesh_->shear_periodic) ExertTidalForce(t,dt);
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
