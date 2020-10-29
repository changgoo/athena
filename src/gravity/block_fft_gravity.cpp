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
//! \fn BlockFFTGravity::BlockFFTGravity(MeshBlock *pmb)
//  \brief BlockFFTGravity constructor
BlockFFTGravity::BlockFFTGravity(MeshBlock *pmb)
    : BlockFFT(pmb) {
  gtlist_ = new FFTGravitySolverTaskList(NULL, pmb->pmy_mesh);
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::~BlockFFTGravity()
//  \brief BlockFFTGravity destructor
BlockFFTGravity::~BlockFFTGravity() {
  delete gtlist_;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFTGravity::ApplyKernel()
//  \brief Apply kernel
void BlockFFTGravity::ApplyKernel() {
  // local array size in z-pencil decomposition
  int slow_nx1 = pf3d->slow_ihi - pf3d->slow_ilo + 1;
  int slow_nx2 = pf3d->slow_jhi - pf3d->slow_jlo + 1;
  int slow_nx3 = pf3d->slow_khi - pf3d->slow_klo + 1;
  int idx;
  Real kx, ky, kz;
  Real kernel;
  Real dx1sq = SQR(pmy_block_->pcoord->dx1v(NGHOST));
  Real dx2sq = SQR(pmy_block_->pcoord->dx2v(NGHOST));
  Real dx3sq = SQR(pmy_block_->pcoord->dx3v(NGHOST));

  //      ACTION                     (slow, mid, fast)
  // Initial block decomposition          (k,j,i)
  // Remap to x-pencil, perform FFT in i  (k,j,i)
  // Remap to y-pencil, perform FFT in j  (i,k,j)
  // Remap to z-pencil, perform FFT in k  (j,i,k)
  // Apply Kernel                         (j,i,k)
  for (int j=0; j<slow_nx2; j++) {
    for (int i=0; i<slow_nx1; i++) {
      for (int k=0; k<slow_nx3; k++) {
        idx = k + slow_nx3*(i + slow_nx1*j);
        kx = TWO_PI*(pf3d->slow_ilo + i)/Nx1;
        ky = TWO_PI*(pf3d->slow_jlo + j)/Nx2;
        kz = TWO_PI*(pf3d->slow_klo + k)/Nx3;
        if (((pf3d->slow_ilo+i) + (pf3d->slow_jlo+j) + (pf3d->slow_klo+k)) == 0) {
          kernel = 0.0;
        }
        else {
          kernel = -pmy_block_->pgrav->four_pi_G / ((2. - 2.*std::cos(kx))/dx1sq +
                                                    (2. - 2.*std::cos(ky))/dx2sq +
                                                    (2. - 2.*std::cos(kz))/dx3sq);
        }
        in_[idx] *= kernel;
      }
    }
  }
  return;
}

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
