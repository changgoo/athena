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
  // Add metal mass
  imetal = AddRealProperty();
  realfieldname.push_back("metal");

  // Add particle age
  iage = AddRealProperty();
  realfieldname.push_back("age");

  // Add gas fraction as aux peroperty
  igas = AddAuxProperty();
  auxfieldname.push_back("fgas");

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
  mzp.InitWithShallowSlice(realprop, 2, imetal, 1);
  tage.InitWithShallowSlice(realprop, 2, iage, 1);

  fgas.InitWithShallowSlice(auxprop, 2, igas, 1);
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::AddOneParticle()
//! \brief add one particle if position is within the mesh block

void StarParticles::AddOneParticle(Real mp, Real x1, Real x2, Real x3,
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

    // initialize other properties
    mzp(npar_) = mp;
    tage(npar_) = 0.0;
    fgas(npar_) = 0.0;

    npar_++;
  }
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

    // kick by 0.5*dth
    // this kick has to be skipped for new particles
    Kick(t,0.5*dth,pmy_block->phydro->w);
    // x -> x0, v -> v0
    SaveStatus();
    // a temporary heck; later we will have a flag for new particles
    // aging first to distinguish new particle
    Age(t,dt);
    // Boris push for velocity dependent terms: Coriolis force
    if (pmy_mesh_->shear_periodic) BorisKick(t,dth);
    // kick by another 0.5*dth
    Kick(t,0.5*dth,pmy_block->phydro->w);
    // drift from t^n to t^n+1
    Drift(t,dt);

    dt_old = dt; // save dt for the future use
    // Update the position index.
    SetPositionIndices();
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
  for (int k = 0; k < npar_; ++k) tage(k) += dt;
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Drift(Real t, Real dt)
//! \brief drift particles

void StarParticles::Drift(Real t, Real dt) {
  // drift position
  for (int k = 0; k < npar_; ++k) {
    xp_(k) = xp0_(k) + dt * vpx_(k);
    yp_(k) = yp0_(k) + dt * vpy_(k);
    zp_(k) = zp0_(k) + dt * vpz_(k);
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
    Real vpxm = vpx_(k), vpym = vpy_(k);
    if (tage(k) == 0) { // for the new particles
      vpx_(k) = hf1*vpxm + hf2*vpym;
      vpy_(k) = -hf2*vpxm + hf1*vpym;
    } else {
      vpx_(k) = f1*vpxm + f2*vpym;
      vpy_(k) = -f2*vpxm + f1*vpym;
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
    Real acc = tage(k) > 0 ? acc0*dt*xp_(k) : 0.;
    vpx_(k) += acc;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::PointMass(Real t, Real dt)
//! \brief force from a point mass at origin
//! \note first kick (from n-1/2 to n) is skipped for the new particles

void StarParticles::PointMass(Real t, Real dt, Real gm) {
  const Coordinates *pc = pmy_block->pcoord;
  for (int k = 0; k < npar_; ++k) {
    if (tage(k) > 0) {
      Real x1, x2, x3;
      pc->CartesianToMeshCoords(xp_(k), yp_(k), zp_(k), x1, x2, x3);

      Real r = std::sqrt(x1*x1 + x2*x2 + x3*x3); // m0 is at (0,0,0)
      Real acc = -gm/(r*r); // G=1
      Real ax = acc*x1/r, ay = acc*x2/r, az = acc*x3/r;

      vpx_(k) += dt*ax;
      vpy_(k) += dt*ay;
      vpz_(k) += dt*az;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::ConstantAcceleration(Real t, Real dt)
//! \brief constant acceleration

void StarParticles::ConstantAcceleration(Real t, Real dt, Real g1, Real g2, Real g3) {
  for (int k = 0; k < npar_; ++k) {
    if (tage(k) > 0) { // first kick (from n-1/2 to n) is skipped for the new particles
      vpx_(k) += dt*g1;
      vpy_(k) += dt*g2;
      vpz_(k) += dt*g3;
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
