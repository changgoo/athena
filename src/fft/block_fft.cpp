//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file block_fft.cpp
//! \brief defines interface class to Plimpton's fftMPI

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

#ifdef FFT
  if (ndim==3) {
#ifdef MPI_PARALLEL
    // use Plimpton's fftMPI
    pf3d = new FFTMPI_NS::FFT3d(MPI_COMM_WORLD,2); // 2 for double precision
    // set output data layout equal to slow pencil decomposition
    // in order to prevent unnecessary data remap
    int permute=2; // will make output array (slow,mid,fast) = (y,x,z) = (j,i,k)
    int fftsize, sendsize, recvsize; // to be returned from setup
    // preliminary setup to get domain decomposition
    pf3d->setup(Nx1, Nx2, Nx3,
                in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi,
                in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi,
                permute, fftsize, sendsize, recvsize);

    // set global index in pencil decompositions
    fast_ilo = pf3d->fast_ilo;
    fast_ihi = pf3d->fast_ihi;
    fast_jlo = pf3d->fast_jlo;
    fast_jhi = pf3d->fast_jhi;
    fast_klo = pf3d->fast_klo;
    fast_khi = pf3d->fast_khi;
    fast_nx1 = fast_ihi - fast_ilo + 1;
    fast_nx2 = fast_jhi - fast_jlo + 1;
    fast_nx3 = fast_khi - fast_klo + 1;

    mid_ilo = pf3d->mid_ilo;
    mid_ihi = pf3d->mid_ihi;
    mid_jlo = pf3d->mid_jlo;
    mid_jhi = pf3d->mid_jhi;
    mid_klo = pf3d->mid_klo;
    mid_khi = pf3d->mid_khi;
    mid_nx1 = mid_ihi - mid_ilo + 1;
    mid_nx2 = mid_jhi - mid_jlo + 1;
    mid_nx3 = mid_khi - mid_klo + 1;

    slow_ilo = pf3d->slow_ilo;
    slow_ihi = pf3d->slow_ihi;
    slow_jlo = pf3d->slow_jlo;
    slow_jhi = pf3d->slow_jhi;
    slow_klo = pf3d->slow_klo;
    slow_khi = pf3d->slow_khi;
    slow_nx1 = slow_ihi - slow_ilo + 1;
    slow_nx2 = slow_jhi - slow_jlo + 1;
    slow_nx3 = slow_khi - slow_klo + 1;

    // reallocate and setup FFT
    delete pf3d;
    pf3d = new FFTMPI_NS::FFT3d(MPI_COMM_WORLD,2);
    pf3d->setup(Nx1, Nx2, Nx3,
                in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi,
                slow_ilo, slow_ihi, slow_jlo, slow_jhi, slow_klo, slow_khi,
                permute, fftsize, sendsize, recvsize);
#else // serial
    std::stringstream msg;
    msg << "### FATAL ERROR in BlockFFT::BlockFFT" << std::endl
        << "3D FFT only works with MPI " << std::endl;
    ATHENA_ERROR(msg);
    return;
#endif // MPI_PARALLEL
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in BlockFFT::BlockFFT" << std::endl
        << "BlockFFT only works in 3D" << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
#endif // FFT
}

// destructor

BlockFFT::~BlockFFT() {
  delete[] in_;
#ifdef FFT
  fftw_cleanup();
#ifdef MPI_PARALLEL
  delete pf3d;
#endif
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFT::LoadSource(const AthenaArray<Real> &src)
//! \brief Fill the source in the active zone

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
//! \brief Fill the result in the active zone

void BlockFFT::RetrieveResult(AthenaArray<Real> &dst) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        int idx = (i-is) + nx1*((j-js) + nx2*(k-ks));
        dst(k,j,i) = std::real(in_[idx]);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFT::ExecuteForward()
//! \brief Forward transform

void BlockFFT::ExecuteForward() {
#ifdef FFT
#ifdef MPI_PARALLEL
  FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR*>(in_);
  // block2fast
  if (pf3d->remap_prefast) pf3d->remap(data,data,pf3d->remap_prefast);
  // fast_forward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_FORWARD,pf3d->fft_fast);
  // fast2mid
  pf3d->remap(data,data,pf3d->remap_fastmid);
  // mid_forward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_FORWARD,pf3d->fft_mid);
  // mid2slow
  pf3d->remap(data,data,pf3d->remap_midslow);
  // slow_forward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_FORWARD,pf3d->fft_slow);
#endif
#endif

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFT::ApplyKernel()
//! \brief Apply kernel

void BlockFFT::ApplyKernel() {
  // do nothing
  // to be overriden in the derived classes
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFT::ExecuteBackward()
//! \brief Backward transform

void BlockFFT::ExecuteBackward() {
#ifdef FFT
#ifdef MPI_PARALLEL
  FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR*>(in_);
  // slow_backward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_BACKWARD,pf3d->fft_slow);
  // slow2mid
  pf3d->remap(data,data,pf3d->remap_slowmid);
  // mid_backward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_BACKWARD,pf3d->fft_mid);
  // mid2fast
  pf3d->remap(data,data,pf3d->remap_midfast);
  // fast_backward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_BACKWARD,pf3d->fft_fast);
  // fast2block
  if (pf3d->remap_postfast) pf3d->remap(data,data,pf3d->remap_postfast);
  // multiply norm factor
  for (int i=0; i<2*nx1*nx2*nx3; ++i) data[i] /= (Nx1*Nx2*Nx3);
#endif
#endif

  return;
}
