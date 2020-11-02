//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fft_gravity.cpp
//  \brief implementation of functions in class FFTGravity

// C headers

// C++ headers
#include <cmath>
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "block_fft_gravity.hpp"
#include "../coordinates/coordinates.hpp"

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::BlockFFTGravity(MeshBlock *pmb, ParameterInput *pin)
//  \brief BlockFFTGravity constructor
BlockFFTGravity::BlockFFTGravity(MeshBlock *pmb, ParameterInput *pin)
    : BlockFFT(pmb), I_(1.0,0.0),
      dx1sq_(SQR(pmb->pcoord->dx1v(NGHOST))),
      dx2sq_(SQR(pmb->pcoord->dx2v(NGHOST))),
      dx3sq_(SQR(pmb->pcoord->dx3v(NGHOST))),
      Lx1_(pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min),
      Lx2_(pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min),
      Lx3_(pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min) {
  gtlist_ = new FFTGravitySolverTaskList(pin, pmb->pmy_mesh);
  Omega_0_ = pin->GetOrAddReal("problem","Omega0",0.0);
  qshear_  = pin->GetOrAddReal("problem","qshear",0.0);
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::~BlockFFTGravity()
//  \brief BlockFFTGravity destructor
BlockFFTGravity::~BlockFFTGravity() {
  delete gtlist_;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFTGravity::ExecuteForward()
//  \brief Forward transform for shearing sheet Poisson solvers
#ifdef GRAV_DISK
void BlockFFTGravity::ExecuteForward() {
#ifdef FFT
#ifdef MPI_PARALLEL
  Real time = pmy_block_->pmy_mesh->time;

  // cast std::complex* to FFT_SCALAR*
  FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR*>(in_);

  pf3d->remap(data,data,pf3d->remap_premid);                         // block2mid
  // phase shift
  for (int i=0; i<mid_nx1; i++) {
    for (int k=0; k<mid_nx3; k++) {
      for (int j=0; j<mid_nx2; j++) {
        int idx = j + mid_nx2*(k + mid_nx3*i);
        in_[idx] *=
          std::exp(TWO_PI*I_*qshear_*Omega_0_*time*(Lx1_/Lx2_)*(Real)j*(Real)i/(Real)Nx1);
      }
    }
  }
  pf3d->perform_ffts((FFT_DATA *) data,FFTW_FORWARD,pf3d->fft_mid);  // mid_forward
  pf3d->remap(data,data,pf3d->remap_midfast);                        // mid2fast
  pf3d->perform_ffts((FFT_DATA *) data,FFTW_FORWARD,pf3d->fft_fast); // fast_forward
  pf3d->remap(data,data,pf3d->remap_fastslow);                       // fast2slow
  pf3d->perform_ffts((FFT_DATA *) data,FFTW_FORWARD,pf3d->fft_slow); // slow_forward
#endif // MPI_PARALLEL
#endif // FFT

  return;
}
#endif // GRAV_DISK

//----------------------------------------------------------------------------------------
//! \fn void BlockFFTGravity::ApplyKernel()
//  \brief Apply kernel for fully periodic BC
// TODO disk BC kernel
#if defined(GRAV_PERIODIC)||defined(GRAV_DISK)
void BlockFFTGravity::ApplyKernel() {
  //      ACTION                     (slow, mid, fast)
  // Initial block decomposition          (k,j,i)
  // Remap to x-pencil, perform FFT in i  (k,j,i)
  // Remap to y-pencil, perform FFT in j  (i,k,j)
  // Remap to z-pencil, perform FFT in k  (j,i,k)
  // Apply Kernel                         (j,i,k)

  int idx;
  Real kx, ky, kz;
  Real kernel;

  for (int j=0; j<slow_nx2; j++) {
    for (int i=0; i<slow_nx1; i++) {
      for (int k=0; k<slow_nx3; k++) {
        idx = k + slow_nx3*(i + slow_nx1*j);
        kx = TWO_PI*(slow_ilo + i)/Nx1;
        ky = TWO_PI*(slow_jlo + j)/Nx2;
        kz = TWO_PI*(slow_klo + k)/Nx3;
        if (((slow_ilo+i) + (slow_jlo+j) + (slow_klo+k)) == 0) {
          kernel = 0.0;
        }
        else {
          kernel = -pmy_block_->pgrav->four_pi_G / ((2. - 2.*std::cos(kx))/dx1sq_ +
                                                    (2. - 2.*std::cos(ky))/dx2sq_ +
                                                    (2. - 2.*std::cos(kz))/dx3sq_);
        }
        in_[idx] *= kernel;
      }
    }
  }

  return;
}
#endif // GRAV_PERIODIC

void BlockFFTGravity::Solve(int stage) {
  AthenaArray<Real> rho;
  rho.InitWithShallowSlice(pmy_block_->phydro->u,4,IDN,1);
  LoadSource(rho);
  ExecuteForward();
  ApplyKernel();
  ExecuteBackward();
  RetrieveResult(pmy_block_->pgrav->phi);

  gtlist_->DoTaskListOneStage(pmy_block_->pmy_mesh, stage);

  return;
}
