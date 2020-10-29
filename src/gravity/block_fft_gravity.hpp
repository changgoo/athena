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
  BlockFFTGravity(MeshBlock *pmb);
  ~BlockFFTGravity();
  void ApplyKernel() final;
  void Solve(int stage);

 private:
  FFTGravitySolverTaskList *gtlist_;
};

#endif // GRAVITY_BLOCK_FFT_GRAVITY_HPP_
