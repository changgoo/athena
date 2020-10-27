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

using namespace FFTMPI_NS;

// constructor, initializes data structures and parameters

BlockFFT::BlockFFT(MeshBlock *pmb) :
    pmy_block_(pmb), ndim(pmb->pmy_mesh->ndim),
    is(pmb->is), ie(pmb->ie), js(pmb->js), je(pmb->je), ks(pmb->ks), ke(pmb->ke),
    Nx1(pmb->pmy_mesh->mesh_size.nx1),
    Nx2(pmb->pmy_mesh->mesh_size.nx2),
    Nx3(pmb->pmy_mesh->mesh_size.nx3),
    nx1(pmb->block_size.nx1),
    nx2(pmb->block_size.nx2),
    nx3(pmb->block_size.nx3),
    in_ilo((pmb->loc.lx1)*pmb->block_size.nx1),
    in_ihi(((pmb->loc.lx1+1)*pmb->block_size.nx1)-1),
    in_jlo((pmb->loc.lx2)*pmb->block_size.nx2),
    in_jhi(((pmb->loc.lx2+1)*pmb->block_size.nx2)-1),
    in_klo((pmb->loc.lx3)*pmb->block_size.nx3),
    in_khi(((pmb->loc.lx3+1)*pmb->block_size.nx3)-1) {
  int cnt = nx1*nx2*nx3;
  in_ = new std::complex<Real>[cnt];
  out_ = new std::complex<Real>[cnt];

  if (ndim==3) {
    // use Plimpton's fftMPI
    pf3d = new FFT3d(MPI_COMM_WORLD,2);
    // set output data layout equal to slow pencil decomposition
    // in order to prevent unnecessary data remap
    out_ilo = pf3d->slow_ilo;
    out_ihi = pf3d->slow_ihi;
    out_jlo = pf3d->slow_jlo;
    out_jhi = pf3d->slow_jhi;
    out_klo = pf3d->slow_klo;
    out_khi = pf3d->slow_khi;
    int permute=2; // will make output array (slow,mid,fast) = (y,x,z) = (j,i,k)
    int fftsize, sendsize, recvsize; // to be returned from setup
    // setup 3D FFT
    pf3d->setup(Nx1, Nx2, Nx3,
                in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi,
                out_ilo, out_ihi, out_jlo, out_jhi, out_klo, out_khi,
                permute, fftsize, sendsize, recvsize);
    if (Globals::my_rank==0) {
      std::cout << "-----------FFT3d setup------------" << std::endl;
      std::cout << "fftsize = " << fftsize << std::endl;
      std::cout << "sendsize = " << sendsize << std::endl;
      std::cout << "recvsize = " << recvsize << std::endl;
    }
  }

//  else if (ndim==2)
//  else if (ndim==1)
//  else
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
        in_[idx] = {src(k,j,i), 0.0};
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
//! \fn void BlockFFT::ExecuteForward()
//  \brief Forward transform

void BlockFFT::ExecuteForward() {
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFT::ApplyKernel()
//  \brief Apply kernel

void BlockFFT::ApplyKernel() {
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFT::ExecuteBackward()
//  \brief Backward transform

void BlockFFT::ExecuteBackward() {
  return;
}
