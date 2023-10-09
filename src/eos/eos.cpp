//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eos.cpp
//! \brief implements common functions in class EquationOfState

// C headers

// C++ headers
#include <cmath>   // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ConservedToPrimitiveTest(AthenaArray<Real> &cons,
//!   const AthenaArray<Real> &prim_old, const FaceField &b,
//!   AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//!   int il, int iu, int jl, int ju, int kl, int ku);
//! \brief Just for test. cons(IEN) only contains e_int + e_k even if it is MHD

void EquationOfState::ConservedToPrimitiveTest(
    const AthenaArray<Real> &cons, const AthenaArray<Real> &bcc,
    int il, int iu, int jl, int ju, int kl, int ku) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real u_d  = cons(IDN,k,j,i);
        Real u_m1 = cons(IM1,k,j,i);
        Real u_m2 = cons(IM2,k,j,i);
        Real u_m3 = cons(IM3,k,j,i);

        Real w_d, w_vx, w_vy, w_vz;

        bool dfloor_used = false, pfloor_used = false;
        if (NON_BAROTROPIC_EOS) {
          Real u_e  = cons(IEN,k,j,i);
          Real e_mag = 0.0;
          if (MAGNETIC_FIELDS_ENABLED) {
            Real bcc1 = bcc(IB1,k,j,i);
            Real bcc2 = bcc(IB2,k,j,i);
            Real bcc3 = bcc(IB3,k,j,i);
            e_mag = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
          }

          Real w_p, dp;
          SingleConservedToPrimitive(u_d, u_m1, u_m2, u_m3, u_e,
                                     w_d, w_vx, w_vy, w_vz, w_p,
                                     dp, dfloor_used, pfloor_used,
                                     e_mag);
        } else {
          SingleConservedToPrimitive(u_d, u_m1, u_m2, u_m3,
                                     w_d, w_vx, w_vy, w_vz,
                                     dfloor_used);
        }
        fofc_(k,j,i) = dfloor_used || pfloor_used;
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::SingleConservedToPrimitive(
//!  Real &u_d, Real &u_m1, Real &u_m2, Real &u_m3,
//!  Real &w_d, Real &w_vx, Real &w_vy, Real &w_vz,
//!  bool &dfloor_used)
//! \brief Converts single conserved variable into primitive variable in isothermal.
//!        Checks floor needs
void EquationOfState::SingleConservedToPrimitive(
  Real &u_d, Real &u_m1, Real &u_m2, Real &u_m3,
  Real &w_d, Real &w_vx, Real &w_vy, Real &w_vz,
  bool &dfloor_used)  {
  // apply density floor, without changing momentum or energy
  if (u_d < density_floor_) {
    u_d = density_floor_;
    dfloor_used = true;
  }

  w_d = u_d;

  Real di = 1.0/u_d;
  w_vx = u_m1*di;
  w_vy = u_m2*di;
  w_vz = u_m3*di;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::SingleConservedToPrimitive(
//!  Real &u_d, Real &u_m1, Real &u_m2, Real &u_m3, Real &u_e,
//!  Real &w_d, Real &w_vx, Real &w_vy, Real &w_vz, Real &w_p,
//!  Real &dp, bool &dfloor_used, bool &pfloor_used)
//! \brief Converts single conserved variable into primitive variable in hydro.
//!        Checks floor needs
void EquationOfState::SingleConservedToPrimitive(
  Real &u_d, Real &u_m1, Real &u_m2, Real &u_m3, Real &u_e,
  Real &w_d, Real &w_vx, Real &w_vy, Real &w_vz, Real &w_p,
  Real &dp, bool &dfloor_used, bool &pfloor_used)  {
  // apply density floor, without changing momentum or energy
  if (u_d < density_floor_) {
    u_d = density_floor_;
    dfloor_used = true;
  }

  w_d = u_d;

  Real di = 1.0/u_d;
  w_vx = u_m1*di;
  w_vy = u_m2*di;
  w_vz = u_m3*di;

  Real gm1 = gamma_ - 1.0;
  Real e_k = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
  w_p = gm1*(u_e - e_k);

  // apply pressure floor, correct total energy
  if (w_p < pressure_floor_) {
    dp = pressure_floor_ - w_p;
    w_p = pressure_floor_;
    u_e = w_p/gm1 + e_k;
    pfloor_used = true;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::SingleConservedToPrimitive(
//!  Real &u_d, Real &u_m1, Real &u_m2, Real &u_m3, Real &u_e, Real emag,
//!  Real &w_d, Real &w_vx, Real &w_vy, Real &w_vz, Real &w_p,
//!  Real &dp, bool &dfloor_used, bool &pfloor_used, const Real e_mag)
//! \brief Converts single conserved variable into primitive variable in MHD.
//!        Checks floor needs
void EquationOfState::SingleConservedToPrimitive(
  Real &u_d, Real &u_m1, Real &u_m2, Real &u_m3, Real &u_e,
  Real &w_d, Real &w_vx, Real &w_vy, Real &w_vz, Real &w_p,
  Real &dp, bool &dfloor_used, bool &pfloor_used, const Real e_mag)  {
  // apply density floor, without changing momentum or energy
  if (u_d < density_floor_) {
    u_d = density_floor_;
    dfloor_used = true;
  }

  w_d = u_d;

  Real di = 1.0/u_d;
  w_vx = u_m1*di;
  w_vy = u_m2*di;
  w_vz = u_m3*di;

  Real gm1 = gamma_ - 1.0;
  Real e_k = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
  w_p = gm1*(u_e - e_k - e_mag);

  // apply pressure floor, correct total energy
  if (w_p < pressure_floor_) {
    dp = pressure_floor_ - w_p;
    w_p = pressure_floor_;
    u_e = w_p/gm1 + e_k + e_mag;
    pfloor_used = true;
  }
}

