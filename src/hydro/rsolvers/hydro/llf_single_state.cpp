//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file llf_single_state.cpp
//! \brief Local Lax Friedrichs (LLF) Riemann solver for hydrodynamics
//!
//!  Computes 1D fluxes using the LLF Riemann solver, also known as Rusanov's
//!  method. This flux is very diffusive, even more diffusive than HLLE, and so
//!  it is not recommended for use in applications.  However, it is useful for
//!  testing, or for problems where other Riemann solvers fail.
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd
//! ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.

// C headers

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../eos/eos.hpp"
#include "../../hydro.hpp"

//----------------------------------------------------------------------------------------
//! \fn void Hydro::SingleStateLLF_Hyd
//! \brief The LLF Riemann solver for hydrodynamics (both adiabatic and
//! isothermal)

void Hydro::SingleStateLLF_Hydro(Real wli[], Real wri[], Real flx[]) {
  Real fl[(NHYDRO)], fr[(NHYDRO)], du[(NHYDRO)];
  Real gm1 = pmy_block->peos->GetGamma() - 1.0;
  Real iso_cs = pmy_block->peos->GetIsoSoundSpeed();

  //--- Step 2.  Compute wave speeds in L,R states (see Toro eq. 10.43)

  Real cl = pmy_block->peos->SoundSpeed(wli);
  Real cr = pmy_block->peos->SoundSpeed(wri);
  Real a = 0.5 * std::max((std::abs(wli[IVX]) + cl), (std::abs(wri[IVX]) + cr));

  //--- Step 3.  Compute L/R fluxes

  Real mxl = wli[IDN] * wli[IVX];
  Real mxr = wri[IDN] * wri[IVX];

  fl[IDN] = mxl;
  fr[IDN] = mxr;

  fl[IVX] = mxl * wli[IVX];
  fr[IVX] = mxr * wri[IVX];

  fl[IVY] = mxl * wli[IVY];
  fr[IVY] = mxr * wri[IVY];

  fl[IVZ] = mxl * wli[IVZ];
  fr[IVZ] = mxr * wri[IVZ];

  Real el, er;
  if (NON_BAROTROPIC_EOS) {
    el = wli[IPR] / gm1 +
         0.5 * wli[IDN] * (SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ]));
    er = wri[IPR] / gm1 +
         0.5 * wri[IDN] * (SQR(wri[IVX]) + SQR(wri[IVY]) + SQR(wri[IVZ]));
    fl[IVX] += wli[IPR];
    fr[IVX] += wri[IPR];
    fl[IEN] = (el + wli[IPR]) * wli[IVX];
    fr[IEN] = (er + wri[IPR]) * wri[IVX];
  } else {
    fl[IVX] += (iso_cs * iso_cs) * wli[IDN];
    fr[IVX] += (iso_cs * iso_cs) * wri[IDN];
  }

  //--- Step 4.  Compute difference in L/R states dU

  du[IDN] = wri[IDN] - wli[IDN];
  du[IVX] = wri[IDN] * wri[IVX] - wli[IDN] * wli[IVX];
  du[IVY] = wri[IDN] * wri[IVY] - wli[IDN] * wli[IVY];
  du[IVZ] = wri[IDN] * wri[IVZ] - wli[IDN] * wli[IVZ];
  if (NON_BAROTROPIC_EOS)
    du[IEN] = er - el;

  //--- Step 5. Compute the LLF flux at interface (see Toro eq. 10.42).

  flx[IDN] = 0.5 * (fl[IDN] + fr[IDN]) - a * du[IDN];
  flx[IVX] = 0.5 * (fl[IVX] + fr[IVX]) - a * du[IVX];
  flx[IVY] = 0.5 * (fl[IVY] + fr[IVY]) - a * du[IVY];
  flx[IVZ] = 0.5 * (fl[IVZ] + fr[IVZ]) - a * du[IVZ];
  if (NON_BAROTROPIC_EOS) {
    flx[IEN] = 0.5 * (fl[IEN] + fr[IEN]) - a * du[IEN];
  }

  return;
}
