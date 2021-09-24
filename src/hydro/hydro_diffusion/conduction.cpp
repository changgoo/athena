//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file conduction.cpp
//! \brief

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
//! Calculate isotropic thermal conduction
//! Including saturation effect

void HydroDiffusion::ThermalFluxIso(
     const AthenaArray<Real> &p, AthenaArray<Real> *flx) {
  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;
  AthenaArray<Real> &x1flux = flx[X1DIR];
  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;
  Real kappaf, denf, dTdx, dTdy, dTdz, cs2f, qsat, kappa_eff;

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
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd private(kappaf, denf, dTdx, cs2f, qsat, kappa_eff)
      for (int i=is; i<=ie+1; ++i) {
        kappaf = 0.5*(kappa(DiffProcess::iso,k,j,i) + kappa(DiffProcess::iso,k,j,i-1));
        denf = 0.5*(p(IDN,k,j,i) + p(IDN,k,j,i-1));
        dTdx = (p(IPR,k,j,i)/p(IDN,k,j,i) - p(IPR,k,j,i-1)/
                p(IDN,k,j,i-1))/pco_->dx1v(i-1);
        if (heatflux_saturation) {
          // saturation of heat flux
          cs2f = 0.5*(p(IPR,k,j,i)/p(IDN,k,j,i) + p(IPR,k,j,i-1)/p(IDN,k,j,i-1));
          qsat = 1.5*denf*cs2f*std::sqrt(cs2f);
          kappa_eff = 1/(1/(kappaf*denf)+std::abs(dTdx)/qsat);
          kappa_eff1(k,j,i) = kappa_eff;
          x1flux(k,j,i) -= kappa_eff*dTdx;
        } else {
          x1flux(k,j,i) -= kappaf*denf*dTdx;
        }
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
    for (int k=kl; k<=ku; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd private(kappaf, denf, dTdy, cs2f, qsat, kappa_eff)
        for (int i=il; i<=iu; ++i) {
          kappaf = 0.5*(kappa(DiffProcess::iso,k,j,i) + kappa(DiffProcess::iso,k,j-1,i));
          denf = 0.5*(p(IDN,k,j,i) + p(IDN,k,j-1,i));
          dTdy = (p(IPR,k,j,i)/p(IDN,k,j,i) - p(IPR,k,j-1,i)/
                  p(IDN,k,j-1,i))/pco_->h2v(i)/pco_->dx2v(j-1);
          if (heatflux_saturation) {
            // saturation of heat flux
            cs2f = 0.5*(p(IPR,k,j,i)/p(IDN,k,j,i) + p(IPR,k,j-1,i)/p(IDN,k,j-1,i));
            qsat = 1.5*denf*cs2f*std::sqrt(cs2f);
            kappa_eff = 1/(1/(kappaf*denf)+std::abs(dTdy)/qsat);
            kappa_eff2(k,j,i) = kappa_eff;
            x2flux(k,j,i) -= kappa_eff*dTdy;
          } else {
            x2flux(k,j,i) -= kappaf*denf*dTdy;
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
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd private(kappaf, denf, dTdz, cs2f, qsat, kappa_eff)
        for (int i=il; i<=iu; ++i) {
          kappaf = 0.5*(kappa(DiffProcess::iso,k,j,i) + kappa(DiffProcess::iso,k-1,j,i));
          denf = 0.5*(p(IDN,k,j,i) + p(IDN,k-1,j,i));
          dTdz = (p(IPR,k,j,i)/p(IDN,k,j,i) - p(IPR,k-1,j,i)/
                  p(IDN,k-1,j,i))/pco_->dx3v(k-1)/pco_->h31v(i)/pco_->h32v(j);
          if (heatflux_saturation) {
            // saturation of heat flux
            cs2f = 0.5*(p(IPR,k,j,i)/p(IDN,k,j,i) + p(IPR,k-1,j,i)/p(IDN,k-1,j,i));
            qsat = 1.5*denf*cs2f*std::sqrt(cs2f);
            kappa_eff = 1/(1/(kappaf*denf)+std::abs(dTdz)/qsat);
            kappa_eff3(k,j,i) = kappa_eff;
            x3flux(k,j,i) -= kappa_eff*dTdz;
          } else {
            x3flux(k,j,i) -= kappaf*denf*dTdz;
          }
        }
      }
    }

    if (heatflux_saturation) {
      // update kappa to the value actually used for the heat flux calculation
      // for correct dt_parabolic calcuation
      // (TODO) changgoo: can be moved outside the loop when it properly treats 1D/2D
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            Real& kcc = kappa(DiffProcess::iso,k,j,i);
            const Real& kappa1_i   = kappa_eff1(k,j,i  );
            const Real& kappa1_ip1 = kappa_eff1(k,j,i+1);
            const Real& kappa2_j   = kappa_eff2(k,j  ,i);
            const Real& kappa2_jp1 = kappa_eff2(k,j+1,i);
            const Real& kappa3_k   = kappa_eff3(k  ,j,i);
            const Real& kappa3_kp1 = kappa_eff3(k+1,j,i);
            Real kcc1, kcc2, kcc3;
            kcc1 = 0.5*(kappa1_i + kappa1_ip1);
            kcc2 = 0.5*(kappa2_j + kappa2_jp1);
            kcc3 = 0.5*(kappa3_k + kappa3_kp1);
            kcc = std::max({kcc1, kcc2, kcc3});
          }
        }
      }
    }
  } // zero flux for 1D/2D
  return;
}


//---------------------------------------------------------------------------------------
//! Calculate anisotropic thermal conduction

void HydroDiffusion::ThermalFluxAniso(
     const AthenaArray<Real> &p, AthenaArray<Real> *flx) {
  return;
}


//----------------------------------------------------------------------------------------
//! constant viscosity

void ConstConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                     const AthenaArray<Real> &bcc,
                     int is, int ie, int js, int je, int ks, int ke) {
  if (phdif->kappa_iso > 0.0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i)
          phdif->kappa(HydroDiffusion::DiffProcess::iso,k,j,i) = phdif->kappa_iso;
      }
    }
  }
  if (phdif->kappa_aniso > 0.0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i)
          phdif->kappa(HydroDiffusion::DiffProcess::aniso,k,j,i) = phdif->kappa_aniso;
      }
    }
  }
  return;
}
