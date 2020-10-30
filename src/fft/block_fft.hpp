#ifndef FFT_BLOCK_FFT_HPP_
#define FFT_BLOCK_FFT_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file athena_fft.hpp
//  \brief defines minimalist FFT class

// C headers

// C++ headers
#include <complex>
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/meshblock_tree.hpp"

#ifdef FFT
#include <fftw3.h>
#ifdef MPI_PARALLEL
#include <mpi.h>
#include "fftmpi/src/fft2d.h"
#include "fftmpi/src/fft3d.h"
using namespace FFTMPI_NS;
#endif // MPI_PARALLEL
#endif


//! \class BlockFFT
//  \brief

class BlockFFT {
 public:
  BlockFFT(MeshBlock *pmb);
  virtual ~BlockFFT();

  void LoadSource(const AthenaArray<Real> &src);
  void RetrieveResult(AthenaArray<Real> &dst);
  virtual void ExecuteForward();
  virtual void ApplyKernel();
  virtual void ExecuteBackward();

  // data
  const int is, ie, js, je, ks, ke; // meshblock indices
  const int Nx1, Nx2, Nx3;          // mesh size (active zones)
  const int nx1, nx2, nx3;          // meshblock size (active zones)
  const int ndim;                   // number of dimensions
  // global index for input data layout
  const int in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi;
  int out_ilo, out_ihi, out_jlo, out_jhi, out_klo, out_khi;
  int out_nx1, out_nx2, out_nx3;
#ifdef MPI_PARALLEL
  FFT3d *pf3d;
#endif // MPI_PARALLEL

 protected:
  MeshBlock *pmy_block_;
  std::complex<Real> *in_;
};

#endif // FFT_BLOCK_FFT_HPP_
