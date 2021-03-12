//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file cr.cpp
//! \brief implementation of functions in class CosmicRay
//======================================================================================


#include <stdio.h>  // fopen and fwrite
#include <iostream>  // cout
#include <sstream>  // msg
#include <stdexcept> // runtime erro

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/units.hpp"
#include "cr.hpp"
#include "integrators/cr_integrators.hpp"

// constructor, initializes data structures and parameters

// The default opacity function.

// This function also needs to set the streaming velocity
// This is needed to calculate the work term
inline void DefaultOpacity(MeshBlock *pmb, AthenaArray<Real> &u_cr,
              AthenaArray<Real> &prim, AthenaArray<Real> &bcc) {
  // set the default opacity to be a large value in the default hydro case
  CosmicRay *pcr=pmb->pcr;
  int kl=pmb->ks, ku=pmb->ke;
  int jl=pmb->js, ju=pmb->je;
  int il=pmb->is-1, iu=pmb->ie+1;
  if(pmb->block_size.nx2 > 1) {
    jl -= 1;
    ju += 1;
  }
  if(pmb->block_size.nx3 > 1) {
    kl -= 1;
    ku += 1;
  }

  Real invlim = 1.0/pcr->vmax;

  if(MAGNETIC_FIELDS_ENABLED) {
    for(int k=kl; k<=ku; ++k) {
      for(int j=jl; j<=ju; ++j) {
        // The diffusion coefficient is calculated with respect to B direction
        // Estimate of Grad Pc along B

        // x component
        pmb->pcoord->CenterWidth1(k,j,il-1,iu+1,pcr->cwidth);
        for(int i=il; i<=iu; ++i) {
          Real distance = 0.5*(pcr->cwidth(i-1) + pcr->cwidth(i+1))
                         + pcr->cwidth(i);
          Real dprdx=(u_cr(CRE,k,j,i+1) - u_cr(CRE,k,j,i-1))/3.0;
          dprdx /= distance;
          pcr->b_grad_pc(k,j,i) = bcc(IB1,k,j,i) * dprdx;
        }
        // y component
        if (pmb->block_size.nx2 > 1) {
          pmb->pcoord->CenterWidth2(k,j-1,il,iu,pcr->cwidth1);
          pmb->pcoord->CenterWidth2(k,j,il,iu,pcr->cwidth);
          pmb->pcoord->CenterWidth2(k,j+1,il,iu,pcr->cwidth2);

          for(int i=il; i<=iu; ++i) {
            Real distance = 0.5*(pcr->cwidth1(i) + pcr->cwidth2(i))
                           + pcr->cwidth(i);
            Real dprdy=(u_cr(CRE,k,j+1,i) - u_cr(CRE,k,j-1,i))/3.0;
            dprdy /= distance;
            pcr->b_grad_pc(k,j,i) += bcc(IB2,k,j,i) * dprdy;
          }
        }
        // z component
        if (pmb->block_size.nx3 > 1) {
          pmb->pcoord->CenterWidth3(k-1,j,il,iu,pcr->cwidth1);
          pmb->pcoord->CenterWidth3(k,j,il,iu,pcr->cwidth);
          pmb->pcoord->CenterWidth3(k+1,j,il,iu,pcr->cwidth2);

          for(int i=il; i<=iu; ++i) {
            Real distance = 0.5*(pcr->cwidth1(i) + pcr->cwidth2(i))
                            + pcr->cwidth(i);
            Real dprdz=(u_cr(CRE,k+1,j,i) -  u_cr(CRE,k-1,j,i))/3.0;
            dprdz /= distance;
            pcr->b_grad_pc(k,j,i) += bcc(IB3,k,j,i) * dprdz;
          }
        }

        for(int i=il; i<=iu; ++i) {
          Real btot = std::sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                           bcc(IB2,k,j,i)*bcc(IB2,k,j,i) +
                           bcc(IB3,k,j,i)*bcc(IB3,k,j,i));

          Real b_grad_pc = pcr->b_grad_pc(k,j,i);

          //diffusion coefficient
          pcr->sigma_diff(0,k,j,i) = pcr->Get_SigmaParallel(prim(IDN,k,j,i),
                                     prim(IPR,k,j,i),u_cr(CRE,k,j,i),
                                     fabs(b_grad_pc)/btot);
          if (pcr->perp_diff_flag == 0) {
            pcr->sigma_diff(1,k,j,i) = pcr->max_opacity;
            pcr->sigma_diff(2,k,j,i) = pcr->max_opacity;
          } else {
            pcr->sigma_diff(1,k,j,i) = pcr->sigma_diff(0,k,j,i)*pcr->perp_to_par_diff;
            pcr->sigma_diff(2,k,j,i) = pcr->sigma_diff(0,k,j,i)*pcr->perp_to_par_diff;
          }

          Real inv_sqrt_rho;
          if (pcr->self_consistent_flag == 0) {
            inv_sqrt_rho = 1.0/std::sqrt(prim(IDN,k,j,i));
          } else {
            Real rhoi = pcr->Get_IonDensity(prim(IDN,k,j,i),
                        prim(IPR,k,j,i),u_cr(CRE,k,j,i));
            inv_sqrt_rho = 1.0/std::sqrt(rhoi);
          }
          Real va1 = bcc(IB1,k,j,i) * inv_sqrt_rho;
          Real va2 = bcc(IB2,k,j,i) * inv_sqrt_rho;
          Real va3 = bcc(IB3,k,j,i) * inv_sqrt_rho;
          Real va = btot * inv_sqrt_rho;
          Real dpc_sign = 0.0;
          if(b_grad_pc > TINY_NUMBER) dpc_sign = 1.0;
          else if(-b_grad_pc > TINY_NUMBER) dpc_sign = -1.0;
          //streaming velocity
          pcr->v_adv(0,k,j,i) = -va1 * dpc_sign;
          pcr->v_adv(1,k,j,i) = -va2 * dpc_sign;
          pcr->v_adv(2,k,j,i) = -va3 * dpc_sign;

          //streaming coefficient
          if(va < TINY_NUMBER) {
            pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
          } else {
            pcr->sigma_adv(0,k,j,i) = fabs(b_grad_pc)/(btot * va * (1.0 + 1.0/3.0)
                                               * invlim * u_cr(CRE,k,j,i));
          }
          pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;

          // Here we calculate the angles of B needed to compute the rotation matrix
          // The information stored in the array
          // b_angle is
          // b_angle[0]=sin_theta_b
          // b_angle[1]=cos_theta_b
          // b_angle[2]=sin_phi_b
          // b_angle[3]=cos_phi_b

          Real bxby = std::sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                           bcc(IB2,k,j,i)*bcc(IB2,k,j,i));

          if(btot > TINY_NUMBER) {
            pcr->b_angle(0,k,j,i) = bxby/btot;
            pcr->b_angle(1,k,j,i) = bcc(IB3,k,j,i)/btot;
          } else {
            pcr->b_angle(0,k,j,i) = 1.0;
            pcr->b_angle(1,k,j,i) = 0.0;
          }
          if(bxby > TINY_NUMBER) {
            pcr->b_angle(2,k,j,i) = bcc(IB2,k,j,i)/bxby;
            pcr->b_angle(3,k,j,i) = bcc(IB1,k,j,i)/bxby;
          } else {
            pcr->b_angle(2,k,j,i) = 0.0;
            pcr->b_angle(3,k,j,i) = 1.0;
          }
        }//end i
      }// end j
    }// end k
  } else { // End MHD
    for(int k=kl; k<=ku; ++k) {
      for(int j=jl; j<=ju; ++j) {
#pragma omp simd
        for(int i=il; i<=iu; ++i) {
          pcr->sigma_diff(0,k,j,i) = pcr->sigma;
          pcr->sigma_diff(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_diff(2,k,j,i) = pcr->max_opacity;

          pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;

          pcr->v_adv(0,k,j,i) = 0.0;
          pcr->v_adv(1,k,j,i) = 0.0;
          pcr->v_adv(2,k,j,i) = 0.0;
        }
      }
    }
  }// end MHD and stream flag
}

CosmicRay::CosmicRay(MeshBlock *pmb, ParameterInput *pin):
    u_cr(NCR,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    u_cr1(NCR,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    //constructor overload resolution of non-aggregate class type AthenaArray<Real>
    coarse_cr_(NCR,pmb->ncc3, pmb->ncc2, pmb->ncc1,
              (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
              AthenaArray<Real>::DataStatus::empty)),
    sigma_diff(3,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    sigma_adv(3,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    v_adv(3,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    v_diff(3,pmb->ncells3,pmb->ncells2,pmb->ncells1),
    CRInjectionRate(pmb->ncells3,pmb->ncells2,pmb->ncells1),
    flux{{NCR, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
        {NCR,pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
        (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
        AthenaArray<Real>::DataStatus::empty)},
        {NCR,pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
        (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
        AthenaArray<Real>::DataStatus::empty)}},
    pmy_block(pmb),
    cr_bvar(pmb, &u_cr, &coarse_cr_, flux),
    UserSourceTerm_{} {
  Mesh *pm = pmy_block->pmy_mesh;
  // "Enroll" in S/AMR by adding to vector of tuples of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinement(&u_cr, &coarse_cr_);
  }

  cr_source_defined = false;

  cr_bvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&cr_bvar);
  pmb->pbval->bvars_main_int.push_back(&cr_bvar);

  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;

  b_grad_pc.NewAthenaArray(nc3,nc2,nc1);
  b_angle.NewAthenaArray(4,nc3,nc2,nc1);

  cwidth.NewAthenaArray(nc1);
  cwidth1.NewAthenaArray(nc1);
  cwidth2.NewAthenaArray(nc1);

  // set a default opacity function
  UpdateOpacity = DefaultOpacity;
  // set a default temperature function
  UpdateTemperature = DefaultTemperature;

  pcrintegrator = new CRIntegrator(this, pin);

  //Flags
  stream_flag = pin->GetOrAddInteger("cr","vs_flag",1);
  src_flag = pin->GetOrAddInteger("cr","src_flag",1);
  losses_flag = pin->GetOrAddInteger("cr","losses_flag",0);
  perp_diff_flag = pin->GetOrAddInteger("cr","perp_diff_flag",0);
  self_consistent_flag = pin->GetOrAddInteger("cr","self_consistent_flag",0);
  if (self_consistent_flag) losses_flag = 1;

  //Code units
  DensityUnit = pin->GetOrAddReal("problem", "DensityUnit",1.);
  LengthUnit = pin->GetOrAddReal("problem", "LengthUnit",1.);
  VelocityUnit = pin->GetOrAddReal("problem", "VelocityUnit",1.);
  punit = new Units(DensityUnit,LengthUnit,VelocityUnit);

  //Input parameters
  vmax = pin->GetOrAddReal("cr","vmax",1.0); //this should be in code units already
  sigma = pin->GetOrAddReal("cr","sigma",1.0);
  max_opacity = pin->GetOrAddReal("cr","max_opacity",1.e10);
  lambdac = pin->GetOrAddReal("cr","lambdac",0.0); //dec/dt = -lambdac nH ec
  perp_to_par_diff = pin->GetOrAddReal("cr","diff_ratio",10.0);
  ion_rate_norm = pin->GetOrAddReal("cr","ion_rate_norm",
                  1e-4); //in cgs unit -- the dafault value assumes delta = -0.35

  sigma *= vmax;
  sigma *= punit->second/(punit->cm*punit->cm);
  lambdac /= punit->second;
}

CosmicRay::~CosmicRay() {
  delete pcrintegrator;
  delete punit;
}

//Enrol the function to update opacity

void CosmicRay::EnrollOpacityFunction(CROpacityFunc MyOpacityFunction) {
  UpdateOpacity = MyOpacityFunction;
}

void CosmicRay::EnrollTemperatureFunction(CRTemperatureFunc MyTemperatureFunction) {
  UpdateTemperature = MyTemperatureFunction;
}

void CosmicRay::EnrollUserCRSource(CRSrcTermFunc my_func) {
  UserSourceTerm_ = my_func;
  cr_source_defined = true;
}

