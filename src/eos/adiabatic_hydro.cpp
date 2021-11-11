//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adiabatic_hydro.cpp
//! \brief implements functions in class EquationOfState for adiabatic hydrodynamics`

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
    gamma_{pin->GetReal("hydro", "gamma")},
    density_floor_{pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*float_min))},
    pressure_floor_{pin->GetOrAddReal("hydro", "pfloor", std::sqrt(1024*float_min))},
    scalar_floor_{pin->GetOrAddReal("hydro", "sfloor", std::sqrt(1024*float_min))},
    neighbor_flooring_{pin->GetOrAddBoolean("hydro", "neighbor_flooring", false)} {}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
//!          const AthenaArray<Real> &prim_old, const FaceField &b,
//!          AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//!          int il, int iu, int jl, int ju, int kl, int ku)
//! \brief Converts conserved into primitive variables in adiabatic hydro.

void EquationOfState::ConservedToPrimitive(
    AthenaArray<Real> &cons, const AthenaArray<Real> &prim_old, const FaceField &b,
    AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
    Coordinates *pco, int il, int iu, int jl, int ju, int kl, int ku) {
  Real gm1 = GetGamma() - 1.0;
  int nbad_d = 0, nbad_p = 0;
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
// #pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);
        Real& u_e  = cons(IEN,k,j,i);

        Real& w_d  = prim(IDN,k,j,i);
        Real& w_vx = prim(IVX,k,j,i);
        Real& w_vy = prim(IVY,k,j,i);
        Real& w_vz = prim(IVZ,k,j,i);
        Real& w_p  = prim(IPR,k,j,i);

        // apply density floor, without changing momentum or energy
        if (neighbor_flooring_) {
          w_d = u_d; // store old value
          if (ApplyNeighborFloorsDensity(cons,bcc,k,j,i,il,iu,jl,ju,kl,ku)) {
            nbad_d++;
            std::cout << "[Neighbor Flooring] rank=" << Globals::my_rank
                      << " density floor applied: old=" << w_d
                      << " new=" << u_d << std::endl;
          }
        }

        u_d = (u_d > density_floor_) ?  u_d : density_floor_;
        w_d = u_d;

        Real di = 1.0/u_d;
        w_vx = u_m1*di;
        w_vy = u_m2*di;
        w_vz = u_m3*di;

        Real e_k = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));

        // apply pressure floor, correct total energy
        u_e -= e_k; // make u_e internal energy temporarily
        if (neighbor_flooring_) {
          if (ApplyNeighborFloorsPressure(cons,bcc,k,j,i,il,iu,jl,ju,kl,ku)) {
            // pressure corrected
            nbad_p++;
          }
        }
        w_p = gm1*u_e; // calculate pressure
        // update total energy with floor
        u_e = (w_p > pressure_floor_) ?  (u_e+e_k) : ((pressure_floor_/gm1) + e_k);
        w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;
      }
    }
  }

  // if (nbad_p > 0) std::cout << nbad_p << " cells had negative pressure" << std::endl;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
//!          const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
//!          int il, int iu, int jl, int ju, int kl, int ku);
//! \brief Converts primitive variables into conservative variables

void EquationOfState::PrimitiveToConserved(
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bc,
    AthenaArray<Real> &cons, Coordinates *pco,
    int il, int iu, int jl, int ju, int kl, int ku) {
  Real igm1 = 1.0/(GetGamma() - 1.0);

  // Force outer-loop vectorization
#pragma omp simd
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      //#pragma omp simd
#pragma novector
      for (int i=il; i<=iu; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);
        Real& u_e  = cons(IEN,k,j,i);

        const Real& w_d  = prim(IDN,k,j,i);
        const Real& w_vx = prim(IVX,k,j,i);
        const Real& w_vy = prim(IVY,k,j,i);
        const Real& w_vz = prim(IVZ,k,j,i);
        const Real& w_p  = prim(IPR,k,j,i);

        u_d = w_d;
        u_m1 = w_vx*w_d;
        u_m2 = w_vy*w_d;
        u_m3 = w_vz*w_d;
        u_e = w_p*igm1 + 0.5*w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz));
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::SoundSpeed(Real prim[NHYDRO])
//! \brief returns adiabatic sound speed given vector of primitive variables
Real EquationOfState::SoundSpeed(const Real prim[NHYDRO]) {
  return std::sqrt(gamma_*prim[IPR]/prim[IDN]);
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j,
//!                                                 =int i)
//! \brief Apply density and pressure floors to reconstructed L/R cell interface states
void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i) {
  Real& w_d  = prim(IDN,i);
  Real& w_p  = prim(IPR,i);

  // apply (prim) density floor
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // apply pressure floor
  w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyPrimitiveConservedFloors(AthenaArray<Real> &prim,
//!           AthenaArray<Real> &cons, FaceField &b, int k, int j, int i) {
//! \brief Apply pressure (prim) floor and correct energy (cons) (typically after W(U))
void EquationOfState::ApplyPrimitiveConservedFloors(
    AthenaArray<Real> &prim, AthenaArray<Real> &cons, AthenaArray<Real> &bcc,
    int k, int j, int i) {
  Real gm1 = GetGamma() - 1.0;
  Real& w_d  = prim(IDN,k,j,i);
  Real& w_p  = prim(IPR,k,j,i);

  Real& u_d  = cons(IDN,k,j,i);
  Real& u_e  = cons(IEN,k,j,i);
  // apply (prim) density floor, without changing momentum or energy
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // ensure cons density matches
  u_d = w_d;

  Real e_k = 0.5*w_d*(SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i))
                      + SQR(prim(IVZ,k,j,i)));
  // apply pressure floor, correct total energy
  u_e = (w_p > pressure_floor_) ?
        u_e : ((pressure_floor_/gm1) + e_k);
  w_p = (w_p > pressure_floor_) ?
        w_p : pressure_floor_;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyNeighborFloorsDensity(AthenaArray<Real> &cons,
//!        int n, int k, int j, int i)
//! \brief Apply neighbor cell average for flooring
bool EquationOfState::ApplyNeighborFloorsDensity(AthenaArray<Real> &cons,
  AthenaArray<Real> &bcc, int k, int j, int i,
  int il, int iu, int jl, int ju, int kl, int ku) {
  // apply density floor, without changing momentum or energy
  Real u_d = cons(IDN,k,j,i);
  if (u_d < density_floor_) {
    // std::cout<< "[Neighbor Flooring] n=" << n << " q_old=" << q;
    Real n_neighbors = 0.0;
    // container for density, momentum, and internal energy
    AthenaArray<Real> q_neighbors(NHYDRO);
    q_neighbors.ZeroClear();

    int koff[] = {1,-1,0,0,0,0};
    int joff[] = {0,0,1,-1,0,0};
    int ioff[] = {0,0,0,0,1,-1};

    for (int idx=0; idx<6; ++idx) {
      int k0=k+koff[idx];
      int j0=j+joff[idx];
      int i0=i+ioff[idx];
      // skip idices outside mesh block
      if ((i0<il) || (i0>iu) || (j0<jl) || (j0>ju) || (k0<kl) || (k0 >ku)) continue;

      if ((cons(IDN,k0,j0,i0)>density_floor_)) {
        // this must be volume if non-uniform
        n_neighbors += 1.0;
        // sum density and momentum
        for (int n=0; n<(NHYDRO-1); ++n)
          q_neighbors(n) += cons(n,k0,j0,i0);
        // sum internal energy
        Real u_m1 = cons(IM1,k0,j0,i0);
        Real u_m2 = cons(IM2,k0,j0,i0);
        Real u_m3 = cons(IM3,k0,j0,i0);
        Real u_e  = cons(IEN,k0,j0,i0);
        Real di = 1.0/cons(IDN,k0,j0,i0);
        Real e_k = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
        q_neighbors(IEN) += (u_e - e_k);
      }
    }
    // assign averaged density, momentum, internal energy
    for (int n=0; n<(NHYDRO); ++n)
      cons(n,k,j,i) = q_neighbors(n)/n_neighbors;
    Real u_m1 = cons(IM1,k,j,i);
    Real u_m2 = cons(IM2,k,j,i);
    Real u_m3 = cons(IM3,k,j,i);
    Real di = 1.0/cons(IDN,k,j,i);
    Real e_k = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
    cons(IEN,k,j,i) += e_k;
    // std::cout<< " q_new=" << q << " n_neighbors=" << n_neighbors << std::endl;
    return true;
  }

  return false;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyNeighborFloorsPressure(AthenaArray<Real> &prim,
//!        int n, int k, int j, int i)
//! \brief Apply neighbor cell average for flooring
bool EquationOfState::ApplyNeighborFloorsPressure(AthenaArray<Real> &cons,
  AthenaArray<Real> &bcc, int k, int j, int i,
  int il, int iu, int jl, int ju, int kl, int ku) {
  Real gm1 = GetGamma() - 1.0;
  // calculate internal energy
  // this cell density must be good
  Real eint = cons(IEN,k,j,i);
  if (eint < pressure_floor_*gm1) {
    Real n_neighbors = 0.0;
    Real q_neighbors = 0.0;
    int koff[] = {1,-1,0,0,0,0};
    int joff[] = {0,0,1,-1,0,0};
    int ioff[] = {0,0,0,0,1,-1};

    for (int idx=0; idx<6; ++idx) {
      int k0=k+koff[idx];
      int j0=j+joff[idx];
      int i0=i+ioff[idx];
      // skip idices outside mesh block
      if ((i0<il) || (i0>iu) || (j0<jl) || (j0>ju) || (k0<kl) || (k0 >ku)) continue;

      if ((cons(IDN,k0,j0,i0)>density_floor_)) {
        // calculate internal energy only if the neighboring cell is good
        Real nu_d  = cons(IDN,k0,j0,i0);
        Real nu_m1 = cons(IM1,k0,j0,i0);
        Real nu_m2 = cons(IM2,k0,j0,i0);
        Real nu_m3 = cons(IM3,k0,j0,i0);
        Real nu_e  = cons(IEN,k0,j0,i0);
        Real ndi = 1.0/nu_d;
        Real ne_k = 0.5*ndi*(SQR(nu_m1) + SQR(nu_m2) + SQR(nu_m3));
        Real neint = nu_e - ne_k;
        if (neint > pressure_floor_*gm1) {
          // take internal energy only if the neighboring cell is good
          n_neighbors += 1.0;
          q_neighbors += neint;
        }
      }
    }
    cons(IEN,k,j,i) = q_neighbors/n_neighbors;
    return true;
  }

  return false;
}
