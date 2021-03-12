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
#include "particle_gravity.hpp"
#include "particles.hpp"

//--------------------------------------------------------------------------------------
//! \fn DustParticles::DustParticles(MeshBlock *pmb, ParameterInput *pin)
//! \brief constructs a DustParticles instance.

DustParticles::DustParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp)
  : Particles(pmb, pin, pp), backreaction(false), dragforce(true), variable_taus(false),
  iwx(-1), iwy(-1), iwz(-1), idpx1(-1), idpx2(-1), idpx3(-1),
  itaus(-1), taus0(0.0) {
  // Add working array at particles for gas velocity/particle momentum change.
  iwx = AddWorkingArray();
  iwy = AddWorkingArray();
  iwz = AddWorkingArray();

  // Define mass.
  mass = pin->GetOrAddReal(input_block_name, "mass", 1.0);

  // Define stopping time.
  variable_taus = pin->GetOrAddBoolean(input_block_name, "variable_taus", variable_taus);
  taus0 = pin->GetOrAddReal(input_block_name, "taus0", taus0);
  if (variable_taus) itaus = AddAuxProperty();

  // Turn on/off back reaction.
  dragforce = taus0 >= 0.0;
  backreaction = pin->GetOrAddBoolean(input_block_name, "backreaction", false);
  if (taus0 == 0.0) backreaction = false;

  if (backreaction) {
    idpx1 = imom1;
    idpx2 = imom2;
    idpx3 = imom3;
  }

  if (backreaction) {
    dpx1.InitWithShallowSlice(ppm->meshaux, 4, idpx1, 1);
    dpx2.InitWithShallowSlice(ppm->meshaux, 4, idpx2, 1);
    dpx3.InitWithShallowSlice(ppm->meshaux, 4, idpx3, 1);
  }

  if (SELF_GRAVITY_ENABLED && backreaction) {
    isgravity_ = pp->gravity;
    pmy_mesh->particle_gravity = true;
    // Add working arrays for gravity forces
    igx = AddWorkingArray();
    igy = AddWorkingArray();
    igz = AddWorkingArray();
    // Activate particle gravity.
    ppgrav = new ParticleGravity(this);
  }

  // Re-Allocate working arrays.
  work.DeleteAthenaArray();
  work.NewAthenaArray(nwork,nparmax);

  // Assign shorthands (need to do this for every constructor of a derived class)
  AssignShorthands();
}

//--------------------------------------------------------------------------------------
//! \fn DustParticles::~DustParticles()
//! \brief destroys a DustParticles instance.

DustParticles::~DustParticles() {
  if (SELF_GRAVITY_ENABLED && backreaction)
    delete ppgrav;
}

//--------------------------------------------------------------------------------------
//! \fn void DustParticles::SetOneParticleMass(Real new_mass)
//! \brief sets the mass of each particle.

void DustParticles::SetOneParticleMass(Real new_mass) {
  pinput->SetReal(input_block_name, "mass", mass = new_mass);
}

//--------------------------------------------------------------------------------------
//! \fn Real DustParticles::NewBlockTimeStep();
//! \brief returns the time step required by particles in the block.

Real DustParticles::NewBlockTimeStep() {
  // Run first the parent class.
  Real dt = Particles::NewBlockTimeStep();

  // Nothing to do for tracer particles.
  if (taus0 <= 0.0) return dt;

  Real epsmax = 0;
  if (backreaction) {
    // Find the maximum local solid-to-gas density ratio.
    Coordinates *pc = pmy_block->pcoord;
    Hydro *phydro = pmy_block->phydro;
    const int is = ppm->is, js = ppm->js, ks = ppm->ks;
    const int ie = ppm->ie, je = ppm->je, ke = ppm->ke;
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i) {
          Real epsilon = ppm->weight(k,j,i) / (
                         pc->GetCellVolume(k,j,i) * phydro->u(IDN,k,j,i));
          epsmax = std::max(epsmax, epsilon);
        }
    epsmax *= mass;
  }

  // Return the drag timescale.
  return std::min(dt, static_cast<Real>(cfl_par * taus0 / (1.0 + epsmax)));
}

//--------------------------------------------------------------------------------------
//! \fn void DustParticles::AssignShorthands()
//! \brief assigns shorthands by shallow coping slices of the data.

void DustParticles::AssignShorthands() {
  Particles::AssignShorthands();
  wx.InitWithShallowSlice(work, 2, iwx, 1);
  wy.InitWithShallowSlice(work, 2, iwy, 1);
  wz.InitWithShallowSlice(work, 2, iwz, 1);
  if (variable_taus) taus.InitWithShallowSlice(auxprop, 2, itaus, 1);
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
    for (int k = 0; k < npar; ++k) {
      Real x1, x2, x3;
      //! \todo (ccyang):
      //! - using (xp0, yp0, zp0) is a temporary hack.
      pc->CartesianToMeshCoords(xp0(k), yp0(k), zp0(k), x1, x2, x3);
      pc->MeshCoordsToCartesianVector(x1, x2, x3, wx(k), wy(k), wz(k),
                                                  wx(k), wy(k), wz(k));
    }

    // Add drag force to particles.
    if (variable_taus) {
      // Variable stopping time
      UserStoppingTime(t, dt, meshsrc);
      for (int k = 0; k < npar; ++k) {
        //! \todo (ccyang):
        //! - This is a temporary hack; to be fixed.
        Real tmpx = vpx(k), tmpy = vpy(k), tmpz = vpz(k);
        //
        Real c = dt / taus(k);
        wx(k) = c * (vpx(k) - wx(k));
        wy(k) = c * (vpy(k) - wy(k));
        wz(k) = c * (vpz(k) - wz(k));
        vpx(k) = vpx0(k) - wx(k);
        vpy(k) = vpy0(k) - wy(k);
        vpz(k) = vpz0(k) - wz(k);
        //
        vpx0(k) = tmpx; vpy0(k) = tmpy; vpz0(k) = tmpz;
        //
      }
    } else if (taus0 > 0.0) {
      // Constant stopping time
      Real c = dt / taus0;
      for (int k = 0; k < npar; ++k) {
        //! \todo (ccyang):
        //! - This is a temporary hack; to be fixed.
        Real tmpx = vpx(k), tmpy = vpy(k), tmpz = vpz(k);
        //
        wx(k) = c * (vpx(k) - wx(k));
        wy(k) = c * (vpy(k) - wy(k));
        wz(k) = c * (vpz(k) - wz(k));
        vpx(k) = vpx0(k) - wx(k);
        vpy(k) = vpy0(k) - wy(k);
        vpz(k) = vpz0(k) - wz(k);
        //
        vpx0(k) = tmpx; vpy0(k) = tmpy; vpz0(k) = tmpz;
        //
      }
    } else if (taus0 == 0.0) {
      // Tracer particles
      for (int k = 0; k < npar; ++k) {
        vpx(k) = wx(k);
        vpy(k) = wy(k);
        vpz(k) = wz(k);
      }
    }
  } else {
    for (int k = 0; k < npar; ++k) {
      //! \todo (ccyang):
      //! - This is a temporary hack; to be fixed.
      Real tmpx = vpx(k), tmpy = vpy(k), tmpz = vpz(k);
      vpx(k) = vpx0(k); vpy(k) = vpy0(k); vpz(k) = vpz0(k);
      vpx0(k) = tmpx; vpy0(k) = tmpy; vpz0(k) = tmpz;
    }
  }

  if (SELF_GRAVITY_ENABLED && backreaction) {
    // Add gravitational force from the Poisson solution.
    ppgrav->FindGravitationalForce(pmy_block->pgrav->phi);
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
  for (int k = 0; k < npar; ++k)
    //! \todo (ccyang):
    //! - using (xp0, yp0, zp0) is a temporary hack.
    pc->CartesianToMeshCoordsVector(xp0(k), yp0(k), zp0(k),
        mass * wx(k), mass * wy(k), mass * wz(k), wx(k), wy(k), wz(k));

  // Assign the momentum change onto mesh.
  ppm->AssignParticlesToMeshAux(work, iwx, idpx1, 3);
}

//--------------------------------------------------------------------------------------
//! \fn void DustParticles::DepositToMesh(Real t, Real dt,
//!              const AthenaArray<Real>& meshsrc, AthenaArray<Real>& meshdst);
//! \brief Deposits meshaux to Mesh.

void DustParticles::DepositToMesh(
         Real t, Real dt, const AthenaArray<Real>& meshsrc, AthenaArray<Real>& meshdst) {
  if (dragforce && backreaction)
    // Deposit particle momentum changes to the gas.
    ppm->DepositMeshAux(meshdst, idpx1, IM1, 3);
}
