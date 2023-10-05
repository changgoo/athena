//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file isothermal_hydro.cpp
//! \brief implements functions in class EquationOfState for isothermal hydrodynamics`

// C headers

// C++ headers
#include <cmath>   // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "eos.hpp"

// EquationOfState constructor

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block_(pmb),
    iso_sound_speed_{pin->GetReal("hydro", "iso_sound_speed")},  // error if missing!
    density_floor_{pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*float_min) )},
    scalar_floor_{pin->GetOrAddReal("hydro", "sfloor", std::sqrt(1024*float_min))} {
  // initialization of FOFC tagging array
  if (pmb->phydro->fofc_enabled)
    fofc_.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
//!           const AthenaArray<Real> &prim_old, const FaceField &b,
//!           AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//!           int il, int iu, int jl, int ju, int kl, int ku)
void EquationOfState::ConservedToPrimitive(
    AthenaArray<Real> &cons, const AthenaArray<Real> &prim_old, const FaceField &b,
    AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
    Coordinates *pco, int il, int iu, int jl, int ju, int kl, int ku) {
  int nbad_d = 0, nbad_p = 0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);

        Real w_d, w_vx, w_vy, w_vz;
        bool dfloor_used = false;
        SingleConservedToPrimitive(u_d, u_m1, u_m2, u_m3,
                                   w_d, w_vx, w_vy, w_vz,
                                   dfloor_used);
        if (dfloor_used) nbad_d++;

        prim(IDN,k,j,i) = w_d;
        prim(IVX,k,j,i) = w_vx;
        prim(IVY,k,j,i) = w_vy;
        prim(IVZ,k,j,i) = w_vz;
      }
    }
  }

  if (nbad_d>0)
    std::cerr << "ncycle = " << pmy_block_->pmy_mesh->ncycle
              << " [Cons2Prim] floored " << nbad_d << " bad density " << std::endl;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
//!           const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
//!           int il, int iu, int jl, int ju, int kl, int ku);
//! \brief Converts primitive variables into conservative variables

void EquationOfState::PrimitiveToConserved(
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bc,
    AthenaArray<Real> &cons, Coordinates *pco,
    int il, int iu, int jl, int ju, int kl, int ku) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);

        const Real& w_d  = prim(IDN,k,j,i);
        const Real& w_vx = prim(IVX,k,j,i);
        const Real& w_vy = prim(IVY,k,j,i);
        const Real& w_vz = prim(IVZ,k,j,i);

        u_d = w_d;
        u_m1 = w_vx*w_d;
        u_m2 = w_vy*w_d;
        u_m3 = w_vz*w_d;
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::SoundSpeed(Real dummy_arg[NHYDRO])
//! \brief returns isothermal sound speed

Real EquationOfState::SoundSpeed(const Real dummy_arg[NHYDRO]) {
  return iso_sound_speed_;
}

//---------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j,
//!                                                 int i)
//! \brief Apply density floor to reconstructed L/R cell interface states

void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i) {
  Real& w_d  = prim(IDN,i);

  // apply density floor
  if (w_d < density_floor_) std::cerr << "[Recon1] Floor applied" << std::endl;
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyPrimitiveConservedFloors(AthenaArray<Real> &prim,
//!           AthenaArray<Real> &cons, FaceField &b, int k, int j, int i) {
//! \brief Apply pressure (prim) floor and correct energy (cons) (typically after W(U))

void EquationOfState::ApplyPrimitiveConservedFloors(
    AthenaArray<Real> &prim, AthenaArray<Real> &cons, AthenaArray<Real> &bcc,
    int k, int j, int i) {
  Real& w_d  = prim(IDN,k,j,i);
  Real& u_d  = cons(IDN,k,j,i);

  // apply (prim) density floor, without changing momentum or energy
  if (w_d < density_floor_) std::cerr << "[Recon2] Floor applied" << std::endl;
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // ensure cons density matches
  u_d = w_d;

  return;
}
