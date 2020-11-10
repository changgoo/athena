#ifndef GRAVITY_BLOCK_FFT_GRAVITY_HPP_
#define GRAVITY_BLOCK_FFT_GRAVITY_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fft_gravity.hpp
//  \brief defines FFTGravity class

// C headers

// C++ headers
#include <complex>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../fft/block_fft.hpp"
#include "../hydro/hydro.hpp"
#include "gravity.hpp"
#include "../task_list/fft_grav_task_list.hpp"

//! \class BlockFFTGravity
//  \brief minimalist FFT gravity solver for each block

class BlockFFTGravity : public BlockFFT {
 public:
  BlockFFTGravity(MeshBlock *pmb, ParameterInput *pin);
  ~BlockFFTGravity();
  void ExecuteForward() final;
  void ApplyKernel() final;
  void ExecuteBackward() final;
  void Solve(int stage);

 private:
  FFTGravitySolverTaskList *gtlist_;
  Real Omega_0_,qshear_,rshear_;
  Real dx1sq_,dx2sq_,dx3sq_;
  Real Lx1_,Lx2_,Lx3_;
  const std::complex<Real> I_;
  std::complex<Real> *in_e_,*in_o_;
};

#endif // GRAVITY_BLOCK_FFT_GRAVITY_HPP_
