//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cr_cloud.cpp
//! \brief Problem generator to reproduce the bottleneck effect for cosmic rays.
//========================================================================================

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <fstream>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()


// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../cr/cr.hpp"
#include "../cr/integrators/cr_integrators.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"


static Real vx = 0.0;
static Real vy = 0.0;
static Real vz = 0.0;
static int direction =0;

void Diffusion(MeshBlock *pmb, AthenaArray<Real> &u_cr,
        AthenaArray<Real> &prim, AthenaArray<Real> &bcc);

void FixMHDLeft(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je,
     int ks, int ke, int ngh);
void FixCRsourceLeft(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie,
    int js, int je, int ks, int ke, int ngh);

void Mesh::InitUserMeshData(ParameterInput *pin) {
  EnrollUserBoundaryFunction(inner_x1, FixMHDLeft);
  if(CR_ENABLED)
    EnrollUserCRBoundaryFunction(inner_x1, FixCRsourceLeft);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if(CR_ENABLED) {
    pcr->EnrollOpacityFunction(Diffusion);
  }
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Bottleneck test
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gamma = peos->GetGamma();
  Real vx=0.0;
  Real rho_c = 1.0;
  Real rho_h = 0.1;
  Real delta_z = 25.0;
  Real z_back = 200.0;
  Real z_front = 200.0;
  Real pgas=1.0;

  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);
        Real density = rho_h + (rho_c - rho_h) *
                       (1.0 + 1.0*tanh((x1-z_front)/delta_z))
                       *(1.0 + 1.0*tanh((z_back-x1)/delta_z));

        phydro->u(IDN,k,j,i) = density;

        phydro->u(IM1,k,j,i) = vx;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = 0.5*vx*vx+pgas/(gamma-1.0);
        }

        if(CR_ENABLED) {
            pcr->u_cr(CRE,k,j,i) = 1.e-6;
            pcr->u_cr(CRF1,k,j,i) = 0.0;
            pcr->u_cr(CRF2,k,j,i) = 0.0;
            pcr->u_cr(CRF3,k,j,i) = 0.0;
        }
      }// end i
    }
  }
  //Need to set opactiy sigma in the ghost zones
  if(CR_ENABLED) {
  // Default values are 1/3
    int nz1 = block_size.nx1 + 2*(NGHOST);
    int nz2 = block_size.nx2;
    if(nz2 > 1) nz2 += 2*(NGHOST);
    int nz3 = block_size.nx3;
    if(nz3 > 1) nz3 += 2*(NGHOST);
    for(int k=0; k<nz3; ++k) {
      for(int j=0; j<nz2; ++j) {
        for(int i=0; i<nz1; ++i) {
          pcr->sigma_diff(0,k,j,i) = pcr->sigma;
          pcr->sigma_diff(1,k,j,i) = pcr->sigma;
          pcr->sigma_diff(2,k,j,i) = pcr->sigma;
        }
      }
    }// end k,j,i
  }// End CR

  // Add horizontal magnetic field lines, to show streaming and diffusion
  // along magnetic field lines
  if(MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          pfield->b.x1f(k,j,i) = 1.0;
        }
      }
    }

    if(block_size.nx2 > 1) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x2f(k,j,i) = 0.0;
          }
        }
      }
    }

    if(block_size.nx3 > 1) {
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x3f(k,j,i) = 0.0;
          }
        }
      }
    }// end nx3

    // set cell centerd magnetic field
    // Add magnetic energy density to the total energy
    pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);

    for(int k=ks; k<=ke; ++k) {
      for(int j=js; j<=je; ++j) {
        for(int i=is; i<=ie; ++i) {
          phydro->u(IEN,k,j,i) +=
            0.5*(SQR((pfield->bcc(IB1,k,j,i)))
               + SQR((pfield->bcc(IB2,k,j,i)))
               + SQR((pfield->bcc(IB3,k,j,i))));
        }
      }
    }
  }// end MHD

  return;
}


void Diffusion(MeshBlock *pmb, AthenaArray<Real> &u_cr,
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

  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
#pragma omp simd
      for(int i=il; i<=iu; ++i) {
        pcr->sigma_diff(0,k,j,i) = pcr->sigma;
        pcr->sigma_diff(1,k,j,i) = pcr->sigma;
        pcr->sigma_diff(2,k,j,i) = pcr->sigma;
      }
    }
  }

  Real invlim=1.0/pcr->vmax;

  // The information stored in the array
  // b_angle is
  // b_angle[0]=sin_theta_b
  // b_angle[1]=cos_theta_b
  // b_angle[2]=sin_phi_b
  // b_angle[3]=cos_phi_b

  if(MAGNETIC_FIELDS_ENABLED) {
    //First, calculate B_dot_grad_Pc
    for(int k=kl; k<=ku; ++k) {
      for(int j=jl; j<=ju; ++j) {
        // x component
        pmb->pcoord->CenterWidth1(k,j,il-1,iu+1,pcr->cwidth);
        for(int i=il; i<=iu; ++i) {
          Real distance = 0.5*(pcr->cwidth(i-1) + pcr->cwidth(i+1))
                         + pcr->cwidth(i);
          Real dprdx=(u_cr(CRE,k,j,i+1) - u_cr(CRE,k,j,i-1))/3.0;
          dprdx /= distance;
          pcr->b_grad_pc(k,j,i) = bcc(IB1,k,j,i) * dprdx;
        }
        //y component
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
            Real dprdz=(u_cr(CRE,k+1,j,i) - u_cr(CRE,k-1,j,i))/3.0;
            dprdz /= distance;
            pcr->b_grad_pc(k,j,i) += bcc(IB3,k,j,i) * dprdz;
          }
        }

      // now calculate the streaming velocity
      // streaming velocity is calculated with respect to the current coordinate
      //  system
      // diffusion coefficient is calculated with respect to B direction
        for(int i=il; i<=iu; ++i) {
          Real pb= bcc(IB1,k,j,i)*bcc(IB1,k,j,i)
                  +bcc(IB2,k,j,i)*bcc(IB2,k,j,i)
                  +bcc(IB3,k,j,i)*bcc(IB3,k,j,i);
          Real inv_sqrt_rho = 1.0/std::sqrt(prim(IDN,k,j,i));
          Real va1 = bcc(IB1,k,j,i)*inv_sqrt_rho;
          Real va2 = bcc(IB2,k,j,i)*inv_sqrt_rho;
          Real va3 = bcc(IB3,k,j,i)*inv_sqrt_rho;

          Real va = std::sqrt(pb/prim(IDN,k,j,i));

          if(pcr->stream_flag) {
            Real dpc_sign = 0.0;
            if(pcr->b_grad_pc(k,j,i) > TINY_NUMBER) dpc_sign = 1.0;
            else if(-pcr->b_grad_pc(k,j,i) > TINY_NUMBER) dpc_sign = -1.0;
            pcr->v_adv(0,k,j,i) = -va1 * dpc_sign;
            pcr->v_adv(1,k,j,i) = -va2 * dpc_sign;
            pcr->v_adv(2,k,j,i) = -va3 * dpc_sign;

            if(va < TINY_NUMBER) {
              pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
            } else {
              pcr->sigma_adv(0,k,j,i) = std::fabs(pcr->b_grad_pc(k,j,i))
                          /(std::sqrt(pb)* va * (1.0 + 1.0/3.0)
                          * invlim * u_cr(CRE,k,j,i));
            }
            pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
            pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;
          } else {
            pcr->v_adv(0,k,j,i) = 0.;
            pcr->v_adv(1,k,j,i) = 0.;
            pcr->v_adv(2,k,j,i) = 0.;
            pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
            pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
            pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;
          }

          // Now calculate the angles of B
          Real bxby = std::sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                           bcc(IB2,k,j,i)*bcc(IB2,k,j,i));
          Real btot = std::sqrt(pb);
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
        // x component
        pmb->pcoord->CenterWidth1(k,j,il-1,iu+1,pcr->cwidth);
        for(int i=il; i<=iu; ++i) {
          Real distance = 0.5*(pcr->cwidth(i-1) + pcr->cwidth(i+1))
                        + pcr->cwidth(i);
          Real grad_pr=(u_cr(CRE,k,j,i+1) - u_cr(CRE,k,j,i-1))/3.0;
          grad_pr /= distance;

          Real va = 1.0/std::sqrt(prim(IDN,k,j,i));

          if(va < TINY_NUMBER) {
            pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
            pcr->v_adv(0,k,j,i) = 0.0;
          } else {
            Real sigma2 = std::fabs(grad_pr)/(va * (1.0 + 1.0/3.0)
                            * invlim * u_cr(CRE,k,j,i));
            if(std::fabs(grad_pr) < TINY_NUMBER) {
              pcr->sigma_adv(0,k,j,i) = 0.0;
              pcr->v_adv(0,k,j,i) = 0.0;
            } else {
              pcr->sigma_adv(0,k,j,i) = sigma2;
              pcr->v_adv(0,k,j,i) = -va * grad_pr/std::fabs(grad_pr);
            }
          }
          pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;
          pcr->v_adv(1,k,j,i) = 0.0;
          pcr->v_adv(2,k,j,i) = 0.0;
        }
      }
    }
  }
}


void FixCRsourceLeft(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie,
    int js, int je, int ks, int ke, int ngh) {
  Real fix_u = 3.0;
  if(CR_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=ngh; ++i) {
          u_cr(CRE,k,j,is-i) = fix_u;
          u_cr(CRF1,k,j,is-i) = u_cr(CRF1,k,j,is);
          u_cr(CRF2,k,j,is-i) = u_cr(CRF2,k,j,is);
          u_cr(CRF3,k,j,is-i) = u_cr(CRF3,k,j,is);
        }
      }
    }
  }
}


void FixMHDLeft(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je,
     int ks, int ke, int ngh) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,is-i) = prim(IDN,k,j,is);
        prim(IVX,k,j,is-i) = -prim(IVX,k,j,is); // reflect 1-velocity
        prim(IVY,k,j,is-i) = prim(IVY,k,j,is);
        prim(IVZ,k,j,is-i) = prim(IVZ,k,j,is);
        if(NON_BAROTROPIC_EOS)
          prim(IEN,k,j,is-i) = prim(IEN,k,j,is);
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b1
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=(NGHOST); ++i) {
          //b.x1f(k,j,(is-i)) = std::sqrt(2.0*const_pb);  // reflect 1-field
          b.x1f(k,j,(is-i)) =  b.x1f(k,j,is);
        }
      }
    }
    if(je > js) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je+1; ++j) {
#pragma omp simd
          for (int i=1; i<=(NGHOST); ++i) {
            b.x2f(k,j,(is-i)) =  b.x2f(k,j,is);
          }
        }
      }
    }
    if(ke > ks) {
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
         for (int i=1; i<=(NGHOST); ++i) {
           b.x3f(k,j,(is-i)) = b.x3f(k,j,is);
         }
       }
     }
    }
  }
}