//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file athena_fft.cpp
//  \brief

// C headers

// C++ headers
#include <complex>
#include <iostream>
#include <sstream>
#include <stdexcept>  // runtime_error

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/meshblock_tree.hpp"
#include "block_fft.hpp"

// constructor, initializes data structures and parameters

BlockFFT::BlockFFT(MeshBlock *pmb) :
    pmy_block_(pmb),
    is(pmb->is), ie(pmb->ie), js(pmb->js), je(pmb->je), ks(pmb->ks), ke(pmb->ke),
    nx1(pmb->block_size.nx1), nx2(pmb->block_size.nx2), nx3(pmb->block_size.nx3),
    block_ilo((pmb->loc.lx1)*pmb->block_size.nx1),
    block_ihi(((pmb->loc.lx1+1)*pmb->block_size.nx1)-1),
    block_jlo((pmb->loc.lx2)*pmb->block_size.nx2),
    block_jhi(((pmb->loc.lx2+1)*pmb->block_size.nx2)-1),
    block_klo((pmb->loc.lx3)*pmb->block_size.nx3),
    block_khi(((pmb->loc.lx3+1)*pmb->block_size.nx3)-1) {
  int cnt = pmb->GetNumberOfMeshBlockCells();
  in_ = new std::complex<Real>[cnt];
  out_ = new std::complex<Real>[cnt];
}

// destructor

BlockFFT::~BlockFFT() {
  delete[] in_;
  delete[] out_;
#ifdef FFT
  fftw_cleanup();
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFT::LoadSource(const AthenaArray<Real> &src)
//  \brief Fill the source in the active zone

void BlockFFT::LoadSource(const AthenaArray<Real> &src) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        int idx = (i-is) + nx1*((j-js) + nx2*(k-ks));
        in_[idx] = {src(0,k,j,i), 0.0};
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFT::RetrieveResult(const AthenaArray<Real> &dst)
//  \brief Fill the result in the active zone

void BlockFFT::RetrieveResult(AthenaArray<Real> &dst) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        int idx = (i-is) + nx1*((j-js) + nx2*(k-ks));
        dst(k,j,i) = std::real(out_[idx]);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFT::ApplyKernel(int mode)
//  \brief Apply kernel

void BlockFFT::ApplyKernel(int mode) {
  return;
}
