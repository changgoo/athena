#ifndef GRAVITY_BLOCK_FFT_GRAVITY_HPP_
#define GRAVITY_BLOCK_FFT_GRAVITY_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file block_fft_gravity.hpp
//! \brief defines BlockFFTGravity class

// C headers

// C++ headers
#include <complex>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../fft/block_fft.hpp"
#include "../hydro/hydro.hpp"
#include "../task_list/fft_grav_task_list.hpp"
#include "gravity.hpp"

//! identifiers for gravity boundary conditions
enum class GravityBoundaryFlag {periodic, disk, open};
//! free functions to return boundary flag given input string
GravityBoundaryFlag GetGravityBoundaryFlag(const std::string& input_string);
//! indefinite integral for the integrated Green's function
Real _GetIGF(Real x, Real y, Real z);

//! \class BlockFFTGravity
//! \brief minimalist FFT gravity solver for each block

class BlockFFTGravity : public BlockFFT {
 public:
  BlockFFTGravity(MeshBlock *pmb, ParameterInput *pin);
  ~BlockFFTGravity();

  // data
  bool SHEAR_PERIODIC; // flag for shear periodic boundary (true w/ , false w/o)
  GravityBoundaryFlag gbflag; // flag for the gravity boundary condition

  // functions
  void ExecuteForward() final;
  void ApplyKernel() final;
  void ExecuteBackward() final;
  void Solve(int stage);
  void InitGreen();
  void LoadOBCSource(const AthenaArray<Real> &src, int px, int py, int pz);
  void RetrieveOBCResult(AthenaArray<Real> &dst, int px, int py, int pz);
  void MultiplyGreen(int px, int py, int pz);
  void SetPhysicalBoundaries();

 private:
  FFTGravitySolverTaskList *gtlist_;
  Real Omega_0_,qshear_,rshear_;
  Real dx1_,dx2_,dx3_;
  Real dx1sq_,dx2sq_,dx3sq_;
  Real Lx1_,Lx2_,Lx3_;
  const std::complex<Real> I_; // sqrt(-1)
  std::complex<Real> *in2_,*in_e_,*in_o_;
  std::complex<Real> *grf_; // Green's function for open BC
#ifdef FFT
#ifdef MPI_PARALLEL
  FFTMPI_NS::FFT3d *pf3dgrf_;
#endif
#endif
};

#endif // GRAVITY_BLOCK_FFT_GRAVITY_HPP_
