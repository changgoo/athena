//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file dust_particles.cpp
//! \brief implements functions in the DustParticles class

// C++ headers
#include <algorithm>  // min()

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "particles.hpp"

//--------------------------------------------------------------------------------------
//! \fn DustParticles::DustParticles(MeshBlock *pmb, ParameterInput *pin)
//! \brief constructs a DustParticles instance.

DustParticles::DustParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp)
  : Particles(pmb, pin, pp),
  backreaction{pin->GetOrAddBoolean(input_block_name, "backreaction", false)},
  variable_taus{pin->GetOrAddBoolean(input_block_name, "variable_taus", false)},
  iwx(-1), iwy(-1), iwz(-1), itaus(-1),
  taus0{pin->GetOrAddReal(input_block_name, "taus0", 0.0)} {
  // Add working array at particles for gas velocity/particle momentum change.
  iwx = AddWorkingArray();
  iwy = AddWorkingArray();
  iwz = AddWorkingArray();

  // Define stopping time.
  if (variable_taus) itaus = AddAuxProperty();

  dragforce = (taus0 >= 0.0);
  if ((taus0 == 0.0) && backreaction) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [DustParticles::DustParticles]" << std::endl
        << "backreaction must be turned off when stopping time is zero" << std::endl;
    ATHENA_ERROR(msg);
  }

  // TODO(SMOON): It is user's responsibility to set isgravity_ through input file.
  // Temporarily commenting out the below line; this may be replaced by exception
  // throwing when (!backreaction && isgravity_)
//  if (!backreaction) isgravity_ = false;

  // Allocate memory and assign shorthands (shallow slices).
  // Every derived Particles need to call these two functions.
  AllocateMemory();
  AssignShorthands();
}

//--------------------------------------------------------------------------------------
//! \fn DustParticles::~DustParticles()
//! \brief destroys a DustParticles instance.

DustParticles::~DustParticles() {
  // nothing to do
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void DustParticles::AddOneParticle()
//! \brief add one particle if position is within the mesh block

void DustParticles::AddOneParticle(Real mp, Real x1, Real x2, Real x3,
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
//! \fn void DustParticles::AddOneParticle()
//! \brief add one particle if position is within the mesh block

void DustParticles::AddOneParticle(Real mp, Real x1, Real x2, Real x3,
  Real v1, Real v2, Real v3, Real taus) {
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
    taus_(npar_) = taus;

    npar_++;
  }
}

//--------------------------------------------------------------------------------------
//! \fn Real DustParticles::OtherCharacteristicTime1();
//! \brief returns the drag timescale.

Real DustParticles::NewDtForDerived() {
  // No further constraints for infinitely tight coupling (zero stopping time), which is
  // equivalent to a tracer particle
  if (taus0 <= 0.0) return std::numeric_limits<Real>::max();

  Real epsmax = 0;
  if (backreaction) {
    // Find the maximum local solid-to-gas density ratio.
    Coordinates *pc = pmy_block->pcoord;
    Hydro *phydro = pmy_block->phydro;
    const AthenaArray<Real> &rhop = ppm->GetMassDensity();
    const int is = ppm->is, js = ppm->js, ks = ppm->ks;
    const int ie = ppm->ie, je = ppm->je, ke = ppm->ke;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          // TODO(SMOON) Is FindLocalDensity called before NewBlockTimeStep?
          // can we enforce this precondition using the variable "updated"?
          Real epsilon = rhop(k,j,i) / phydro->u(IDN,k,j,i);
          epsmax = std::max(epsmax, epsilon);
        }
      }
    }
  }
  Real dt = taus0 / (1.0 + epsmax);
  return dt; // cfl_par is multiplied in the caller (NewBlockTimeStep)
}

//--------------------------------------------------------------------------------------
//! \fn void DustParticles::AssignShorthandsForDerived()
//! \brief assigns shorthands by shallow coping slices of the data.

void DustParticles::AssignShorthandsForDerived() {
  wx.InitWithShallowSlice(work, 2, iwx, 1);
  wy.InitWithShallowSlice(work, 2, iwy, 1);
  wz.InitWithShallowSlice(work, 2, iwz, 1);
  if (variable_taus) taus_.InitWithShallowSlice(auxprop, 2, itaus, 1);
}

//--------------------------------------------------------------------------------------
//! \fn void DustParticles::SourceTerms()
//! \brief adds acceleration to particles.

void DustParticles::SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  if (dragforce) {
    // Interpolate gas velocity onto particles.
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

    // Add drag force to particles.
    if (variable_taus) {
      // Variable stopping time
      UserStoppingTime(t, dt, meshsrc);
      for (int k = 0; k < npar_; ++k) {
        //! \todo (ccyang):
        //! - This is a temporary hack; to be fixed.
        Real tmpx = vpx_(k), tmpy = vpy_(k), tmpz = vpz_(k);
        //
        Real c = dt / taus_(k);
        wx(k) = c * (vpx_(k) - wx(k));
        wy(k) = c * (vpy_(k) - wy(k));
        wz(k) = c * (vpz_(k) - wz(k));
        vpx_(k) = vpx0_(k) - wx(k);
        vpy_(k) = vpy0_(k) - wy(k);
        vpz_(k) = vpz0_(k) - wz(k);
        //
        vpx0_(k) = tmpx; vpy0_(k) = tmpy; vpz0_(k) = tmpz;
        //
      }
    } else if (taus0 > 0.0) {
      // Constant stopping time
      Real c = dt / taus0;
      for (int k = 0; k < npar_; ++k) {
        //! \todo (ccyang):
        //! - This is a temporary hack; to be fixed.
        Real tmpx = vpx_(k), tmpy = vpy_(k), tmpz = vpz_(k);
        //
        wx(k) = c * (vpx_(k) - wx(k));
        wy(k) = c * (vpy_(k) - wy(k));
        wz(k) = c * (vpz_(k) - wz(k));
        vpx_(k) = vpx0_(k) - wx(k);
        vpy_(k) = vpy0_(k) - wy(k);
        vpz_(k) = vpz0_(k) - wz(k);
        //
        vpx0_(k) = tmpx; vpy0_(k) = tmpy; vpz0_(k) = tmpz;
        //
      }
    } else if (taus0 == 0.0) {
      // Tracer particles
      for (int k = 0; k < npar_; ++k) {
        vpx_(k) = wx(k);
        vpy_(k) = wy(k);
        vpz_(k) = wz(k);
      }
    }
  } else {
    for (int k = 0; k < npar_; ++k) {
      //! \todo (ccyang):
      //! - This is a temporary hack; to be fixed.
      Real tmpx = vpx_(k), tmpy = vpy_(k), tmpz = vpz_(k);
      vpx_(k) = vpx0_(k); vpy_(k) = vpy0_(k); vpz_(k) = vpz0_(k);
      vpx0_(k) = tmpx; vpy0_(k) = tmpy; vpz0_(k) = tmpz;
    }
  }

  if (SELF_GRAVITY_ENABLED && backreaction) {
    // SMOON: Why backreaction matters here?
    // Add gravitational force from the Poisson solution.
    ppgrav->FindGravitationalForce(pmy_block->pgrav->phi);
    ppgrav->InterpolateGravitationalForce();
    ppgrav->ExertGravitationalForce(dt);
  }
}

//--------------------------------------------------------------------------------------
//! \fn void DustParticles::UserSourceTerms(Real t, Real dt,
//!                                         const AthenaArray<Real>& meshsrc)
//! \brief adds additional source terms to particles, overloaded by the user.

void __attribute__((weak)) DustParticles::UserSourceTerms(
    Real t, Real dt, const AthenaArray<Real>& meshsrc) {
}

//--------------------------------------------------------------------------------------
//! \fn void DustParticles::UserStoppingTime(Real t, Real dt,
//!                                          const AthenaArray<Real>& meshsrc)
//! \brief assigns time-dependent stopping time to each particle, overloaded by the user.

void __attribute__((weak)) DustParticles::UserStoppingTime(
    Real t, Real dt, const AthenaArray<Real>& meshsrc) {
}

//--------------------------------------------------------------------------------------
//! \fn void DustParticles::ReactToMeshAux(
//!              Real t, Real dt, const AthenaArray<Real>& meshsrc)
//! \brief Reacts to meshaux before boundary communications.

void DustParticles::ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  // Nothing to do if no back reaction.
  if (!dragforce || !backreaction) return;

  // Transform the momentum change in mesh coordinates.
  const Coordinates *pc = pmy_block->pcoord;
  for (int k = 0; k < npar_; ++k)
    //! \todo (ccyang):
    //! - using (xp0, yp0, zp0) is a temporary hack.
    pc->CartesianToMeshCoordsVector(xp0_(k), yp0_(k), zp0_(k),
        mass_(k)*wx(k), mass_(k)*wy(k), mass_(k)*wz(k), wx(k), wy(k), wz(k));

  // Assign the momentum change onto mesh.
  ppm->DepositParticlesToMeshAux(work, iwx, ppm->imom1, 3);
}

//--------------------------------------------------------------------------------------
//! \fn void DustParticles::DepositToMesh(Real t, Real dt,
//!              const AthenaArray<Real>& meshsrc, AthenaArray<Real>& meshdst);
//! \brief Deposits meshaux to Mesh.

void DustParticles::DepositToMesh(
         Real t, Real dt, const AthenaArray<Real>& meshsrc, AthenaArray<Real>& meshdst) {
  if (dragforce && backreaction)
    // Deposit particle momentum changes to the gas.
    ppm->DepositMeshAux(meshdst, ppm->imom1, IM1, 3);
}
