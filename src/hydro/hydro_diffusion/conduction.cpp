//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file conduction.cpp
//! \brief Add explicit thermal conduction term to the energy equation
//!
//! dE/dt = -Div(Q)
//!
//!   where
//!    - Q = -kappa_iso Grad(T) - kappa_aniso([b Dot Grad(T)]b) = heat flux
//!    - T = (P/d)*(mbar/k_B) = temperature
//!    - b = magnetic field unit vector
//!
//!   Here
//!    - kappa_iso   is the   isotropic coefficient of thermal diffusion
//!    - kappa_aniso is the anisotropic coefficient of thermal diffusion
//!
//! Note the kappa's are DIFFUSIVITIES, not CONDUCTIVITIES.  Also note this
//! version uses "dimensionless units" in that the factor (mbar/k_B) is not
//! included in calculating the temperature (instead, T=P/d is used).  For cgs
//! units, kappa must be entered in units of [cm^2/s], and the heat fluxes would
//! need to be multiplied by (k_B/mbar).
//!
//! Anisotropic conduction is adapted from Athena-Cversion
//! - Originally developed by Ian Parrish and Jim Stone
//! - Ported by Jono Squire and Chang-Goo Kim
//!
//! REFERENCE:
//! - Parrish, I.~J. \& Stone, J.~M.\ 2005, \apj, 633, 334. doi:10.1086/444589
//! - Sharma, P. \& Hammett, G.~W.\ 2007, Journal of Computational Physics,
//!   227, 123. doi:10.1016/j.jcp.2007.07.026

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../eos/eos.hpp"
#include "../../mesh/mesh.hpp"
#include "../hydro.hpp"
#include "hydro_diffusion.hpp"

//---------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::ThermalFluxIso(const AthenaArray<Real> &p,
//!        AthenaArray<Real> *flx)
//! \brief Calculate isotropic thermal conduction
//!
//! Q = -kappa_iso Grad(T)

void HydroDiffusion::ThermalFluxIso(const AthenaArray<Real> &p,
                                    AthenaArray<Real> *flx) {
  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;
  AthenaArray<Real> &x1flux = flx[X1DIR];
  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is;
  int js = pmb_->js;
  int ks = pmb_->ks;
  int ie = pmb_->ie;
  int je = pmb_->je;
  int ke = pmb_->ke;
  Real kappaf, denf, dTdx, dTdy, dTdz;

  // i-direction
  jl = js, ju = je, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) {
      if (!f3) // 2D
        jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      else // 3D
        jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    }
  }
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
#pragma omp simd private(kappaf, denf, dTdx)
      for (int i = is; i <= ie + 1; ++i) {
        kappaf = 0.5 * (kappa(DiffProcess::iso, k, j, i) +
                        kappa(DiffProcess::iso, k, j, i - 1));
        denf = 0.5 * (p(IDN, k, j, i) + p(IDN, k, j, i - 1));
        dTdx = (p(IPR, k, j, i) / p(IDN, k, j, i) -
                p(IPR, k, j, i - 1) / p(IDN, k, j, i - 1)) /
               pco_->dx1v(i - 1);
        x1flux(k, j, i) -= kappaf * denf * dTdx;
      }
    }
  }

  // j-direction
  il = is, iu = ie, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (!f3) // 2D
      il = is - 1, iu = ie + 1, kl = ks, ku = ke;
    else // 3D
      il = is - 1, iu = ie + 1, kl = ks - 1, ku = ke + 1;
  }
  if (f2) { // 2D or 3D
    AthenaArray<Real> &x2flux = flx[X2DIR];
    for (int k = kl; k <= ku; ++k) {
      for (int j = js; j <= je + 1; ++j) {
#pragma omp simd private(kappaf, denf, dTdy)
        for (int i = il; i <= iu; ++i) {
          kappaf = 0.5 * (kappa(DiffProcess::iso, k, j, i) +
                          kappa(DiffProcess::iso, k, j - 1, i));
          denf = 0.5 * (p(IDN, k, j, i) + p(IDN, k, j - 1, i));
          dTdy = (p(IPR, k, j, i) / p(IDN, k, j, i) -
                  p(IPR, k, j - 1, i) / p(IDN, k, j - 1, i)) /
                 pco_->h2v(i) / pco_->dx2v(j - 1);
          x2flux(k, j, i) -= kappaf * denf * dTdy;
        }
      }
    }
  } // zero flux for 1D

  // k-direction
  il = is, iu = ie, jl = js, ju = je;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) // 2D or 3D
      il = is - 1, iu = ie + 1, jl = js - 1, ju = je + 1;
    else // 1D
      il = is - 1, iu = ie + 1;
  }
  if (f3) { // 3D
    AthenaArray<Real> &x3flux = flx[X3DIR];
    for (int k = ks; k <= ke + 1; ++k) {
      for (int j = jl; j <= ju; ++j) {
#pragma omp simd private(kappaf, denf, dTdz)
        for (int i = il; i <= iu; ++i) {
          kappaf = 0.5 * (kappa(DiffProcess::iso, k, j, i) +
                          kappa(DiffProcess::iso, k - 1, j, i));
          denf = 0.5 * (p(IDN, k, j, i) + p(IDN, k - 1, j, i));
          dTdz = (p(IPR, k, j, i) / p(IDN, k, j, i) -
                  p(IPR, k - 1, j, i) / p(IDN, k - 1, j, i)) /
                 pco_->dx3v(k - 1) / pco_->h31v(i) / pco_->h32v(j);
          x3flux(k, j, i) -= kappaf * denf * dTdz;
        }
      }
    }
  } // zero flux for 1D/2D
  return;
}

//---------------------------------------------------------------------------------------
//! \fn void HydroDiffusion::ThermalFluxAniso(
//!     const AthenaArray<Real> &p, const FaceField &b,
//!     const AthenaArray<Real> &bcc, AthenaArray<Real> *flx)
//! \brief Calculate anisotropic thermal conduction
//!
//! Q = -kappa_aniso([b Dot Grad(T)]b)
//! b = magnetic field unit vector

void HydroDiffusion::ThermalFluxAniso(const AthenaArray<Real> &p,
                                      const FaceField &b,
                                      const AthenaArray<Real> &bcc,
                                      AthenaArray<Real> *flx) {
  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;
  AthenaArray<Real> &x1flux = flx[X1DIR];
  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is;
  int js = pmb_->js;
  int ks = pmb_->ks;
  int ie = pmb_->ie;
  int je = pmb_->je;
  int ke = pmb_->ke;
  Real kappaf, denf, dTdx, dTdy, dTdz;

  // i-direction
  jl = js, ju = je, kl = ks, ku = ke;
  // magnetic fields must be enabled
  // if (MAGNETIC_FIELDS_ENABLED)
  if (f2) {
    if (!f3) // 2D
      jl = js - 1, ju = je + 1, kl = ks, ku = ke;
    else // 3D
      jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
  }

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      if (f3) { // 3D
#pragma omp simd private(kappaf, denf, dTdy, dTdz)
        for (int i = is; i <= ie + 1; ++i) {
          Real bx = b.x1f(k, j, i);
          Real by = 0.5 * (bcc(IB2, k, j, i) + bcc(IB2, k, j, i - 1));
          Real bz = 0.5 * (bcc(IB3, k, j, i) + bcc(IB3, k, j, i - 1));
          Real bsq = bx * bx + by * by + bz * bz;
          bsq = (bsq > TINY_NUMBER) ? bsq : TINY_NUMBER;

          Real dx = pco_->dx1v(i - 1);
          Real dy = pco_->dx2v(j) * pco_->h2v(i);
          Real dz = pco_->dx3v(k) * pco_->h31v(i) * pco_->h32v(j);

          Real Tkji = p(IPR, k, j, i) / p(IDN, k, j, i);
          Real Tkjim1 = p(IPR, k, j, i - 1) / p(IDN, k, j, i - 1);

          // monotoized temperature difference dTdy
          Real Tjp1i = p(IPR, k, j + 1, i) / p(IDN, k, j + 1, i);
          Real Tjm1i = p(IPR, k, j - 1, i) / p(IDN, k, j - 1, i);
          Real Tjp1im1 = p(IPR, k, j + 1, i - 1) / p(IDN, k, j + 1, i - 1);
          Real Tjm1im1 = p(IPR, k, j - 1, i - 1) / p(IDN, k, j - 1, i - 1);
          dTdy = FourLimiter(Tjp1i - Tkji, Tkji - Tjm1i, Tjp1im1 - Tkjim1,
                             Tkjim1 - Tjm1im1);
          dTdy /= dy;

          // monotoized temperature difference dTdz, 3D only
          Real Tkp1i = p(IPR, k + 1, j, i) / p(IDN, k + 1, j, i);
          Real Tkm1i = p(IPR, k - 1, j, i) / p(IDN, k - 1, j, i);
          Real Tkp1im1 = p(IPR, k + 1, j, i - 1) / p(IDN, k + 1, j, i - 1);
          Real Tkm1im1 = p(IPR, k - 1, j, i - 1) / p(IDN, k - 1, j, i - 1);
          dTdz = FourLimiter(Tkp1i - Tkji, Tkji - Tkm1i, Tkp1im1 - Tkjim1,
                             Tkjim1 - Tkm1im1);
          dTdz /= dz;

          Real bdotdT = bx * (Tkji - Tkjim1) / dx + by * dTdy + bz * dTdz;
          kappaf = 0.5 * (kappa(DiffProcess::aniso, k, j, i) +
                          kappa(DiffProcess::aniso, k, j, i - 1));
          denf = 0.5 * (p(IDN, k, j, i) + p(IDN, k, j, i - 1));
          x1flux(k, j, i) -= kappaf * denf * bdotdT * bx / bsq;
        }
      } else if (f2) { // 2D
#pragma omp simd private(kappaf, denf, dTdy)
        for (int i = is; i <= ie + 1; ++i) {
          Real bx = b.x1f(k, j, i);
          Real by = 0.5 * (bcc(IB2, k, j, i) + bcc(IB2, k, j, i - 1));
          Real bz = 0.5 * (bcc(IB3, k, j, i) + bcc(IB3, k, j, i - 1));
          Real bsq = bx * bx + by * by + bz * bz;
          bsq = (bsq > TINY_NUMBER) ? bsq : TINY_NUMBER;

          Real dx = pco_->dx1v(i - 1);
          Real dy = pco_->dx2v(j) * pco_->h2v(i);

          Real Tkji = p(IPR, k, j, i) / p(IDN, k, j, i);
          Real Tkjim1 = p(IPR, k, j, i - 1) / p(IDN, k, j, i - 1);

          // monotoized temperature difference dTdy
          Real Tjp1i = p(IPR, k, j + 1, i) / p(IDN, k, j + 1, i);
          Real Tjm1i = p(IPR, k, j - 1, i) / p(IDN, k, j - 1, i);
          Real Tjp1im1 = p(IPR, k, j + 1, i - 1) / p(IDN, k, j + 1, i - 1);
          Real Tjm1im1 = p(IPR, k, j - 1, i - 1) / p(IDN, k, j - 1, i - 1);
          dTdy = FourLimiter(Tjp1i - Tkji, Tkji - Tjm1i, Tjp1im1 - Tkjim1,
                             Tkjim1 - Tjm1im1);
          dTdy /= dy;

          Real bdotdT = bx * (Tkji - Tkjim1) / dx + by * dTdy;
          kappaf = 0.5 * (kappa(DiffProcess::aniso, k, j, i) +
                          kappa(DiffProcess::aniso, k, j, i - 1));
          denf = 0.5 * (p(IDN, k, j, i) + p(IDN, k, j, i - 1));
          x1flux(k, j, i) -= kappaf * denf * bdotdT * bx / bsq;
        }
      }
    }
  } // 1D no flux?

  // j-direction
  il = is, iu = ie, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (!f3) // 2D
      il = is - 1, iu = ie + 1, kl = ks, ku = ke;
    else // 3D
      il = is - 1, iu = ie + 1, kl = ks - 1, ku = ke + 1;
  }
  if (f2) { // 2D or 3D
    AthenaArray<Real> &x2flux = flx[X2DIR];
    for (int k = kl; k <= ku; ++k) {
      for (int j = js; j <= je + 1; ++j) {
        if (f3) {
#pragma omp simd private(kappaf, denf, dTdx, dTdz)
          for (int i = il; i <= iu; ++i) {
            Real bx = 0.5 * (bcc(IB1, k, j, i) + bcc(IB1, k, j - 1, i));
            Real by = b.x2f(k, j, i);
            Real bz = 0.5 * (bcc(IB3, k, j, i) + bcc(IB3, k, j - 1, i));
            Real bsq = bx * bx + by * by + bz * bz;
            bsq = (bsq > TINY_NUMBER) ? bsq : TINY_NUMBER;

            Real dx = pco_->dx1v(i);
            Real dy = pco_->dx2v(j - 1) * pco_->h2v(i);
            Real dz = pco_->dx3v(k) * pco_->h31v(i) * pco_->h32v(j);
            Real Tkji = p(IPR, k, j, i) / p(IDN, k, j, i);
            Real Tkjm1i = p(IPR, k, j - 1, i) / p(IDN, k, j - 1, i);

            // monotoized temperature difference dTdx
            Real Tjip1 = p(IPR, k, j, i + 1) / p(IDN, k, j, i + 1);
            Real Tjim1 = p(IPR, k, j, i - 1) / p(IDN, k, j, i - 1);
            Real Tjm1ip1 = p(IPR, k, j - 1, i + 1) / p(IDN, k, j - 1, i + 1);
            Real Tjm1im1 = p(IPR, k, j - 1, i - 1) / p(IDN, k, j - 1, i - 1);
            dTdx = FourLimiter(Tjip1 - Tkji, Tkji - Tjim1, Tjm1ip1 - Tkjm1i,
                               Tkjm1i - Tjm1im1);
            dTdx /= dx;

            // monotoized temperature difference dTdz, 3D only
            Real Tkp1j = p(IPR, k + 1, j, i) / p(IDN, k + 1, j, i);
            Real Tkm1j = p(IPR, k - 1, j, i) / p(IDN, k - 1, j, i);
            Real Tkp1jm1 = p(IPR, k + 1, j - 1, i) / p(IDN, k + 1, j - 1, i);
            Real Tkm1jm1 = p(IPR, k - 1, j - 1, i) / p(IDN, k - 1, j - 1, i);
            dTdz = FourLimiter(Tkp1j - Tkji, Tkji - Tkm1j, Tkp1jm1 - Tkjm1i,
                               Tkjm1i - Tkm1jm1);
            dTdz /= dz;

            Real bdotdT = bx * dTdx + by * (Tkji - Tkjm1i) / dy + bz * dTdz;
            kappaf = 0.5 * (kappa(DiffProcess::aniso, k, j, i) +
                            kappa(DiffProcess::aniso, k, j - 1, i));
            denf = 0.5 * (p(IDN, k, j, i) + p(IDN, k, j - 1, i));
            x2flux(k, j, i) -= kappaf * denf * bdotdT * by / bsq;
          }
        } else { // 2D
#pragma omp simd private(kappaf, denf, dTdx)
          for (int i = il; i <= iu; ++i) {
            Real bx = 0.5 * (bcc(IB1, k, j, i) + bcc(IB1, k, j - 1, i));
            Real by = b.x2f(k, j, i);
            Real bz = 0.5 * (bcc(IB3, k, j, i) + bcc(IB3, k, j - 1, i));
            Real bsq = bx * bx + by * by + bz * bz;
            bsq = (bsq > TINY_NUMBER) ? bsq : TINY_NUMBER;

            Real dx = pco_->dx1v(i);
            Real dy = pco_->dx2v(j - 1) * pco_->h2v(i);

            Real Tkji = p(IPR, k, j, i) / p(IDN, k, j, i);
            Real Tkjm1i = p(IPR, k, j - 1, i) / p(IDN, k, j - 1, i);

            // monotoized temperature difference dTdx
            Real Tjip1 = p(IPR, k, j, i + 1) / p(IDN, k, j, i + 1);
            Real Tjim1 = p(IPR, k, j, i - 1) / p(IDN, k, j, i - 1);
            Real Tjm1ip1 = p(IPR, k, j - 1, i + 1) / p(IDN, k, j - 1, i + 1);
            Real Tjm1im1 = p(IPR, k, j - 1, i - 1) / p(IDN, k, j - 1, i - 1);
            dTdx = FourLimiter(Tjip1 - Tkji, Tkji - Tjim1, Tjm1ip1 - Tkjm1i,
                               Tkjm1i - Tjm1im1);
            dTdx /= dx;

            Real bdotdT = bx * dTdx + by * (Tkji - Tkjm1i) / dy;
            kappaf = 0.5 * (kappa(DiffProcess::aniso, k, j, i) +
                            kappa(DiffProcess::aniso, k, j - 1, i));
            denf = 0.5 * (p(IDN, k, j, i) + p(IDN, k, j - 1, i));
            x2flux(k, j, i) -= kappaf * denf * bdotdT * by / bsq;
          }
        }
      }
    }
  } // zero flux for 1D

  // k-direction
  il = is, iu = ie, jl = js, ju = je;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) // 2D or 3D
      il = is - 1, iu = ie + 1, jl = js - 1, ju = je + 1;
    else // 1D
      il = is - 1, iu = ie + 1;
  }
  if (f3) { // 3D
    AthenaArray<Real> &x3flux = flx[X3DIR];
    for (int k = ks; k <= ke + 1; ++k) {
      for (int j = jl; j <= ju; ++j) {
#pragma omp simd private(kappaf, denf, dTdy, dTdx)
        for (int i = il; i <= iu; ++i) {
          Real bx = 0.5 * (bcc(IB1, k, j, i) + bcc(IB1, k - 1, j, i));
          Real by = 0.5 * (bcc(IB2, k, j, i) + bcc(IB2, k - 1, j, i));
          Real bz = b.x3f(k, j, i);
          Real bsq = bx * bx + by * by + bz * bz;
          bsq = (bsq > TINY_NUMBER) ? bsq : TINY_NUMBER;

          Real dx = pco_->dx1v(i);
          Real dy = pco_->dx2v(j) * pco_->h2v(i);
          Real dz = pco_->dx3v(k - 1) * pco_->h31v(i) * pco_->h32v(j);
          Real Tkji = p(IPR, k, j, i) / p(IDN, k, j, i);
          Real Tkm1ji = p(IPR, k - 1, j, i) / p(IDN, k - 1, j, i);

          // monotoized temperature difference dTdx
          Real Tkip1 = p(IPR, k, j, i + 1) / p(IDN, k, j, i + 1);
          Real Tkim1 = p(IPR, k, j, i - 1) / p(IDN, k, j, i - 1);
          Real Tkm1ip1 = p(IPR, k - 1, j, i + 1) / p(IDN, k - 1, j, i + 1);
          Real Tkm1im1 = p(IPR, k - 1, j, i - 1) / p(IDN, k - 1, j, i - 1);
          dTdx = FourLimiter(Tkip1 - Tkji, Tkji - Tkim1, Tkm1ip1 - Tkm1ji,
                             Tkm1ji - Tkm1im1);
          dTdx /= dx;

          // monotoized temperature difference dTdz, 3D only
          Real Tkjp1 = p(IPR, k, j + 1, i) / p(IDN, k, j + 1, i);
          Real Tkjm1 = p(IPR, k, j - 1, i) / p(IDN, k, j - 1, i);
          Real Tkm1jp1 = p(IPR, k - 1, j + 1, i) / p(IDN, k - 1, j + 1, i);
          Real Tkm1jm1 = p(IPR, k - 1, j - 1, i) / p(IDN, k - 1, j - 1, i);
          dTdy = FourLimiter(Tkjp1 - Tkji, Tkji - Tkjm1, Tkm1jp1 - Tkm1ji,
                             Tkm1ji - Tkm1jm1);
          dTdy /= dy;

          Real bdotdT = bx * dTdx + by * dTdy + bz * (Tkji - Tkm1ji) / dz;
          kappaf = 0.5 * (kappa(DiffProcess::aniso, k, j, i) +
                          kappa(DiffProcess::aniso, k - 1, j, i));
          denf = 0.5 * (p(IDN, k, j, i) + p(IDN, k - 1, j, i));
          x3flux(k, j, i) -= kappaf * denf * bdotdT * bz / bsq;
        }
      }
    }
  } // zero flux for 1D/2D

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ConstConduction(HydroDiffusion *phdif, MeshBlock *pmb,
//!                      const AthenaArray<Real> &prim,
//!                      const AthenaArray<Real> &bcc,
//!                      int is, int ie, int js, int je, int ks, int ke)
//! \brief constant thermal diffusivity

void ConstConduction(HydroDiffusion *phdif, MeshBlock *pmb,
                     const AthenaArray<Real> &prim,
                     const AthenaArray<Real> &bcc, int is, int ie, int js,
                     int je, int ks, int ke) {
  if (phdif->kappa_iso > 0.0) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
#pragma omp simd
        for (int i = is; i <= ie; ++i)
          phdif->kappa(HydroDiffusion::DiffProcess::iso, k, j, i) =
              phdif->kappa_iso;
      }
    }
  }
  if (phdif->kappa_aniso > 0.0) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
#pragma omp simd
        for (int i = is; i <= ie; ++i)
          phdif->kappa(HydroDiffusion::DiffProcess::aniso, k, j, i) =
              phdif->kappa_aniso;
      }
    }
  }
  return;
}
