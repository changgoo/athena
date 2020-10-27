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

//----------------------------------------------------------------------------------------
//! \fn void BlockFFTGravity::ApplyKernel()
//  \brief Apply kernel
void BlockFFTGravity::ApplyKernel() {

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

  return;
}
