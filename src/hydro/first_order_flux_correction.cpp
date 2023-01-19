//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file first_order_flux_correction.cpp
//! \brief Computes divergence of the Hydro fluxes and
//! adds that to a temporary conserved variable register
//! then replace flux to the first order fluxes if it is bad

// C headers

// C++ headers
#include <algorithm>  // std::binary_search
#include <vector>     // std::vector

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../mesh/mesh.hpp"
#include "hydro.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AddFluxDivergence
//! \brief Adds flux divergence to weighted average of conservative variables from
//! previous step(s) of time integrator algorithm

// TODO(felker): consider combining with PassiveScalars implementation + (see 57cfe28b)
// (may rename to AddPhysicalFluxDivergence or AddQuantityFluxDivergence to explicitly
// distinguish from CoordTerms)
// (may rename to AddHydroFluxDivergence and AddScalarsFluxDivergence, if
// the implementations remain completely independent / no inheritance is
// used)
void Hydro::FirstOrderFluxCorrection(Real gam0, Real gam1, Real beta) {
  MeshBlock *pmb = pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int il, iu, jl, ju, kl, ku;
  // estimate updated conserved quantites and flag bad cells
  // assume second order integrator
  Real beta_dt = beta*pmb->pmy_mesh->dt;

  // estimate next step conserved quantities
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          utest_(n,k,j,i) = gam0*u1(n,k,j,i) + gam1*u(n,k,j,i);
        }
      }
    }
  }

  AddFluxDivergence(beta_dt, utest_);

  // test only active zones
  // this call does not change w, w1
  pmb->peos->test_flag = true;
  pmb->peos->ConservedToPrimitive(utest_, w, pmb->pfield->b,
                                  w1, pmb->pfield->bcc, pmb->pcoord,
                                  is, ie, js, je, ks, ke);

  Real wim1[(NHYDRO)],wi[(NHYDRO)],wip1[(NHYDRO)],flx[(NHYDRO)];
  AthenaArray<Real> &x1flux = flux[X1DIR];
  AthenaArray<Real> &x2flux = flux[X2DIR];
  AthenaArray<Real> &x3flux = flux[X3DIR];

  // now replace fluxes with first-order fluxes
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        if (pmb->peos->fofc_(k,j,i)) {
          // x1-flux at i
          // first order left state
          wim1[IDN] = w(IDN,k,j,i-1);
          wim1[IVX] = w(IVX,k,j,i-1);
          wim1[IVY] = w(IVY,k,j,i-1);
          wim1[IVZ] = w(IVZ,k,j,i-1);
          if (NON_BAROTROPIC_EOS) wim1[IPR] = w(IPR,k,j,i-1);

          // first order right state
          wi[IDN] = w(IDN,k,j,i);
          wi[IVX] = w(IVX,k,j,i);
          wi[IVY] = w(IVY,k,j,i);
          wi[IVZ] = w(IVZ,k,j,i);
          if (NON_BAROTROPIC_EOS) wi[IPR] = w(IPR,k,j,i);

          // compute LLF flux
          SingleStateLLF_Hyd(wim1, wi, flx);

          // replace fluxes
          x1flux(IDN,k,j,i) = flx[IDN];
          x1flux(IM1,k,j,i) = flx[IVX];
          x1flux(IM2,k,j,i) = flx[IVY];
          x1flux(IM3,k,j,i) = flx[IVZ];
          if (NON_BAROTROPIC_EOS) x1flux(IEN,k,j,i) = flx[IEN];

          // x1-flux at i+1
          // first order right state
          wip1[IDN] = w(IDN,k,j,i+1);
          wip1[IVX] = w(IVX,k,j,i+1);
          wip1[IVY] = w(IVY,k,j,i+1);
          wip1[IVZ] = w(IVZ,k,j,i+1);
          if (NON_BAROTROPIC_EOS) wip1[IPR] = w(IPR,k,j,i+1);

          // compute LLF flux
          SingleStateLLF_Hyd(wi, wip1, flx);

          // replace fluxes
          x1flux(IDN,k,j,i+1) = flx[IDN];
          x1flux(IM1,k,j,i+1) = flx[IVX];
          x1flux(IM2,k,j,i+1) = flx[IVY];
          x1flux(IM3,k,j,i+1) = flx[IVZ];
          if (NON_BAROTROPIC_EOS) x1flux(IEN,k,j,i+1) = flx[IEN];

          if (pmb->pmy_mesh->f2) {
            // 2D
            // x2-flux at j
            // first order left state
            wim1[IDN] = w(IDN,k,j-1,i);
            wim1[IVX] = w(IVY,k,j-1,i);
            wim1[IVY] = w(IVZ,k,j-1,i);
            wim1[IVZ] = w(IVX,k,j-1,i);
            if (NON_BAROTROPIC_EOS) wim1[IPR] = w(IPR,k,j-1,i);

            // first order right state
            wi[IDN] = w(IDN,k,j,i);
            wi[IVX] = w(IVY,k,j,i);
            wi[IVY] = w(IVZ,k,j,i);
            wi[IVZ] = w(IVX,k,j,i);
            if (NON_BAROTROPIC_EOS) wi[IPR] = w(IPR,k,j,i);

            // compute LLF flux
            SingleStateLLF_Hyd(wim1, wi, flx);

            // replace fluxes
            x2flux(IDN,k,j,i) = flx[IDN];
            x2flux(IM2,k,j,i) = flx[IVX];
            x2flux(IM3,k,j,i) = flx[IVY];
            x2flux(IM1,k,j,i) = flx[IVZ];
            if (NON_BAROTROPIC_EOS) x2flux(IEN,k,j,i) = flx[IEN];

            // x2-flux at j+1
            // first order right state
            wip1[IDN] = w(IDN,k,j+1,i);
            wip1[IVX] = w(IVY,k,j+1,i);
            wip1[IVY] = w(IVZ,k,j+1,i);
            wip1[IVZ] = w(IVX,k,j+1,i);
            if (NON_BAROTROPIC_EOS) wip1[IPR] = w(IPR,k,j+1,i);

            // compute LLF flux
            SingleStateLLF_Hyd(wi, wip1, flx);

            // replace fluxes
            x2flux(IDN,k,j+1,i) = flx[IDN];
            x2flux(IM2,k,j+1,i) = flx[IVX];
            x2flux(IM3,k,j+1,i) = flx[IVY];
            x2flux(IM1,k,j+1,i) = flx[IVZ];
            if (NON_BAROTROPIC_EOS) x2flux(IEN,k,j+1,i) = flx[IEN];
          }

          if (pmb->pmy_mesh->f3) {
            // 3D
            // x3-flux at k
            // first order left state
            wim1[IDN] = w(IDN,k-1,j,i);
            wim1[IVX] = w(IVZ,k-1,j,i);
            wim1[IVY] = w(IVX,k-1,j,i);
            wim1[IVZ] = w(IVY,k-1,j,i);
            if (NON_BAROTROPIC_EOS) wim1[IPR] = w(IPR,k-1,j,i);

            // first order right state
            wi[IDN] = w(IDN,k,j,i);
            wi[IVX] = w(IVZ,k,j,i);
            wi[IVY] = w(IVX,k,j,i);
            wi[IVZ] = w(IVY,k,j,i);
            if (NON_BAROTROPIC_EOS) wi[IPR] = w(IPR,k,j,i);

            // compute LLF flux
            SingleStateLLF_Hyd(wim1, wi, flx);

            // replace fluxes
            x3flux(IDN,k,j,i) = flx[IDN];
            x3flux(IM3,k,j,i) = flx[IVX];
            x3flux(IM1,k,j,i) = flx[IVY];
            x3flux(IM2,k,j,i) = flx[IVZ];
            if (NON_BAROTROPIC_EOS) x3flux(IEN,k,j,i) = flx[IEN];

            // x1-flux at i+1
            // first order right state
            wip1[IDN] = w(IDN,k+1,j,i);
            wip1[IVX] = w(IVZ,k+1,j,i);
            wip1[IVY] = w(IVX,k+1,j,i);
            wip1[IVZ] = w(IVY,k+1,j,i);
            if (NON_BAROTROPIC_EOS) wip1[IPR] = w(IPR,k+1,j,i);

            // compute LLF flux
            SingleStateLLF_Hyd(wi, wip1, flx);

            // replace fluxes
            x3flux(IDN,k+1,j,i) = flx[IDN];
            x3flux(IM3,k+1,j,i) = flx[IVX];
            x3flux(IM1,k+1,j,i) = flx[IVY];
            x3flux(IM2,k+1,j,i) = flx[IVZ];
            if (NON_BAROTROPIC_EOS) x3flux(IEN,k+1,j,i) = flx[IEN];

            // diffusion fluxes needs to be added
            if (!STS_ENABLED)
              AddDiffusionFluxesSingleCell(i,j,k);
          }
        }
      }
    }
  }

  return;
}
