//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cr_shear.cpp
//! \brief Problem generator to reproduce the propagation of cosmic rays in the presence
//! of shear.
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


void Diffusion(MeshBlock *pmb, AthenaArray<Real> &u_cr,
        AthenaArray<Real> &prim, AthenaArray<Real> &bcc);

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  int Nx_mesh = mesh_size.nx1;
  Real xmin = mesh_size.x1min;
  Real xmax = mesh_size.x1max;
  Real deltax = (xmax-xmin)/Nx_mesh;
  int Ny_mesh = mesh_size.nx2;
  Real ymin = mesh_size.x2min;
  Real ymax = mesh_size.x2max;
  Real deltay = (ymax-ymin)/Ny_mesh;

  Real yt, yt_grid, yt0 = pin->GetOrAddReal("problem","offset2",0.);
  int gridindex;
  AthenaArray<Real> Ec, dataEc, dataxEc;
  Ec.NewAthenaArray(Nx_mesh);
  dataEc.NewAthenaArray(Nx_mesh);
  dataxEc.NewAthenaArray(Nx_mesh);

  Real Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);
  Real qshear  = pin->GetOrAddReal("orbital_advection","qshear",0.0);

  // extract solution along the shear direction
  for (int n=0; n<Nx_mesh; ++n) {
    Ec(n) = 0.;
    dataxEc(n) = xmin + deltax * (0.5 + n);
    for (int b=0; b<nblocal; ++b) {
      MeshBlock *pmb = my_blocks(b);
      int is=pmb->is; int ie=pmb->ie; int js=pmb->js; int ks=pmb->ks;
      int Nx = pmb->block_size.nx1;
      int Ny = pmb->block_size.nx2;
      int gis = static_cast<int>(pmb->loc.lx1) * Nx;
      int gjs = static_cast<int>(pmb->loc.lx2) * Ny;
      int xindex = n;
      if (xindex >= gis && xindex < gis + Nx) {
        Real yt = yt0 - Omega0*qshear*dataxEc(n)*time;
        int yindex = static_cast<int>((yt - ymin)/deltay);
        if (yindex >= gjs && yindex < gjs + Ny) {
          Ec(n) = pmb->pcr->u_cr(CRE,ks,yindex-gjs+js,xindex-gis+is);
          dataEc(n) = Ec(n);
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  MPI_Reduce(Ec.data(), dataEc.data(), Nx_mesh, MPI_ATHENA_REAL,
             MPI_SUM, 0, MPI_COMM_WORLD);
#endif

  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign("cr_energy_profile.dat");
    std::stringstream msg;
    FILE *pfile;

    if((pfile = fopen(fname.c_str(),"w")) == NULL) {
      msg << "### FATAL ERROR in function [Mesh::UserWorkAfterLoop]"
          << std::endl << "Error output file could not be opened" <<std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
    fprintf(pfile,"#  x   Ec");
    fprintf(pfile,"\n");

    for(int i=0; i<Nx_mesh; ++i)
      fprintf(pfile,"  %lf  %lf  \n", dataxEc(i), dataEc(i));

    fclose(pfile);
  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if(CR_ENABLED) {
    pcr->EnrollOpacityFunction(Diffusion);
  }
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Testing propagation of cosmic rays in the presence of shear
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real xsize;
  Real Bx, By, Bz;
  Real direction;
  std::stringstream msg;

  // read in the mean velocity, diffusion coefficient
  direction = pin->GetOrAddReal("problem","direction",0);
  if(direction == 0) {
    Bx = pin->GetOrAddReal("problem","B0",0);
    xsize = (pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min)
            /pmy_mesh->mesh_size.nx1;
  } else if(direction == 1) {
    By = pin->GetOrAddReal("problem","B0",0);
    xsize = (pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min)
            /pmy_mesh->mesh_size.nx2;
  } else if(direction == 2) {
    Bz = pin->GetOrAddReal("problem","B0",0);
    xsize = (pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min)
            /pmy_mesh->mesh_size.nx3;
  } else {
    msg << "### FATAL ERROR in Problem Generator" << std::endl
      << "Invalid direction: " << direction << "!"
      << "It must be either 0 or 1 or 2" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  Real offset1 = pin->GetOrAddReal("problem","offset1",0.);
  Real offset2 = pin->GetOrAddReal("problem","offset2",0.);
  Real offset3 = pin->GetOrAddReal("problem","offset3",0.);
  Real Rinj = xsize*pin->GetOrAddReal("problem","cells",1.);

  Real Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);
  Real qshear  = pin->GetOrAddReal("orbital_advection","qshear",0.0);

  Real gamma = peos->GetGamma();

  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i) - offset1;
        Real x2 = pcoord->x2v(j) - offset2;
        Real x3 = pcoord->x3v(k) - offset3;

        Real vy = qshear*Omega0*pcoord->x1v(i);

        phydro->u(IDN,k,j,i) = 1.0;
        phydro->u(IM1,k,j,i) = 0.0;
        //background shearing flow
        phydro->u(IM2,k,j,i) -= phydro->u(IDN,k,j,i)*vy;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = 0.5*(vy*vy)+1.0/(gamma-1.0);
        }

        Real dist_sq=x1*x1;
        Real side1 = x2;
        Real side2 = x3;
        if(direction ==1) {
          dist_sq=x2*x2;
          side1 = x1;
          side2 = x3;
        } else if(direction == 2) {
          dist_sq=x3*x3;
          side1 = x2;
          side2 = x3;
        }

        if(CR_ENABLED) {
          if (std::abs(side1) <= Rinj)
            pcr->u_cr(CRE,k,j,i) = 1e-6 + exp(-40.0*dist_sq);
          else
            pcr->u_cr(CRE,k,j,i) = 1e-6;
          pcr->u_cr(CRF1,k,j,i) = 0.0;
          pcr->u_cr(CRF2,k,j,i) = 0.0;
          pcr->u_cr(CRF3,k,j,i) = 0.0;
        }
      }// end i
    }
  }
  //Need to set opacity sigma in the ghost zones
  if(CR_ENABLED) {
    int nz1 = block_size.nx1 + 2*(NGHOST);
    int nz2 = block_size.nx2;
    if(nz2 > 1) nz2 += 2*(NGHOST);
    int nz3 = block_size.nx3;
    if(nz3 > 1) nz3 += 2*(NGHOST);
    for(int k=0; k<nz3; ++k) {
      for(int j=0; j<nz2; ++j) {
        for(int i=0; i<nz1; ++i) {
          pcr->sigma_diff(0,k,j,i) = pcr->sigma;
          pcr->sigma_diff(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_diff(2,k,j,i) = pcr->max_opacity;
        }
      }
    }// end k,j,i
  }// End CR

  // Add horizontal magnetic field lines, to show streaming and diffusion
  // along magnetic field ines
  if(MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          pfield->b.x1f(k,j,i) = Bx;
        }
      }
    }

    if(block_size.nx2 > 1) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x2f(k,j,i) = By;
          }
        }
      }
    }

    if(block_size.nx3 > 1) {
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x3f(k,j,i) = Bz;
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
        pcr->sigma_diff(1,k,j,i) = pcr->max_opacity;
        pcr->sigma_diff(2,k,j,i) = pcr->max_opacity;
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
            if(pcr->b_grad_pc(k,j,i) > TINY_NUMBER)
              dpc_sign = 1.0;
            else if(-pcr->b_grad_pc(k,j,i) > TINY_NUMBER)
              dpc_sign = -1.0;

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
        }//
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

         Real va = 0.0;

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