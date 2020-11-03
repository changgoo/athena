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
  in_e_ = new std::complex<Real>[nx1*nx2*nx3];
  in_o_ = new std::complex<Real>[nx1*nx2*nx3];
  if (Nx3 != slow_nx3) {
    std::stringstream msg;
    msg << "### FATAL ERROR in BlockFFTGravity::BlockFFTGravity" << std::endl
        << "Something wrong with z-pencil decomposition" << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::~BlockFFTGravity()
//  \brief BlockFFTGravity destructor
BlockFFTGravity::~BlockFFTGravity() {
  delete gtlist_;
  delete[] in_e_;
  delete[] in_o_;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFTGravity::ExecuteForward()
//  \brief Forward transform for shearing sheet Poisson solvers
void BlockFFTGravity::ExecuteForward() {
#ifdef GRAV_DISK
#ifdef FFT
#ifdef MPI_PARALLEL
  Real time = pmy_block_->pmy_mesh->time;
  Real qomt = qshear_*Omega_0_*time;

  // cast std::complex* to FFT_SCALAR*
  FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR*>(in_);

  pf3d->remap(data,data,pf3d->remap_premid);                         // block2mid
  // phase shift
  for (int i=0; i<mid_nx1; i++) {
    for (int k=0; k<mid_nx3; k++) {
      for (int j=0; j<mid_nx2; j++) {
        int idx = j + mid_nx2*(k + mid_nx3*i);
        in_[idx] *= std::exp(TWO_PI*I_*qomt*(Lx1_/Lx2_)*(Real)j*(Real)i/(Real)Nx1);
      }
    }
  }
  pf3d->perform_ffts((FFT_DATA *) data,FFTW_FORWARD,pf3d->fft_mid);  // mid_forward
  pf3d->remap(data,data,pf3d->remap_midfast);                        // mid2fast
  pf3d->perform_ffts((FFT_DATA *) data,FFTW_FORWARD,pf3d->fft_fast); // fast_forward
  pf3d->remap(data,data,pf3d->remap_fastslow);                       // fast2slow
  std::memcpy(in_e_, in_, sizeof(std::complex<Real>)*nx1*nx2*nx3);
  std::memcpy(in_o_, in_, sizeof(std::complex<Real>)*nx1*nx2*nx3);
  for (int j=0; j<slow_nx2; j++) {
    for (int i=0; i<slow_nx1; i++) {
      for (int k=0; k<slow_nx3; k++) {
        int idx = k + slow_nx3*(i + slow_nx1*j);
        in_o_[idx] *= std::exp(-PI*I_*(Real)k/(Real)Nx3);
      }
    }
  }
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA*>(in_e_),FFTW_FORWARD,pf3d->fft_slow); // slow_forward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA*>(in_o_),FFTW_FORWARD,pf3d->fft_slow); // slow_forward
#endif // MPI_PARALLEL
#endif // FFT
#else
  BlockFFT::ExecuteForward();
#endif // GRAV_BC_OPTIONS

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFTGravity::ExecuteBackward()
//  \brief Backward transform for shearing sheet Poisson solvers
void BlockFFTGravity::ExecuteBackward() {
#ifdef GRAV_DISK
#ifdef FFT
#ifdef MPI_PARALLEL
  // cast std::complex* to FFT_SCALAR*
  FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR*>(in_);

  pf3d->perform_ffts(reinterpret_cast<FFT_DATA*>(in_e_),FFTW_BACKWARD,pf3d->fft_slow); // slow_backward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA*>(in_o_),FFTW_BACKWARD,pf3d->fft_slow); // slow_backward
  for (int j=0; j<slow_nx2; j++) {
    for (int i=0; i<slow_nx1; i++) {
      for (int k=0; k<slow_nx3; k++) {
        int idx = k + slow_nx3*(i + slow_nx1*j);
        in_[idx] = in_e_[idx] + std::exp(PI*I_*(Real)k/(Real)Nx3)*in_o_[idx];
      }
    }
  }
  pf3d->remap(data,data,pf3d->remap_slowmid);                         // slow2mid
  pf3d->perform_ffts((FFT_DATA *) data,FFTW_BACKWARD,pf3d->fft_mid);  // mid_backward
  pf3d->remap(data,data,pf3d->remap_midfast);                         // mid2fast
  pf3d->perform_ffts((FFT_DATA *) data,FFTW_BACKWARD,pf3d->fft_fast); // fast_backward
  if (pf3d->remap_postfast)
  pf3d->remap(data,data,pf3d->remap_postfast);                        // fast2block

  // multiply norm factor
  for (int i=0;i<2*nx1*nx2*nx3;++i) {
    data[i] *= Lx3_/(2.0*Nx1*Nx2*SQR(Nx3));
  }
#endif // MPI_PARALLEL
#endif // FFT
#else
  BlockFFT::ExecuteBackward();
#endif // GRAV_BC_OPTIONS

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFTGravity::ApplyKernel()
//  \brief Apply kernel for fully periodic BC
// TODO disk BC kernel
void BlockFFTGravity::ApplyKernel() {
  //      ACTION                     (slow, mid, fast)
  // Initial block decomposition          (k,j,i)
  // Remap to x-pencil, perform FFT in i  (k,j,i)
  // Remap to y-pencil, perform FFT in j  (i,k,j)
  // Remap to z-pencil, perform FFT in k  (j,i,k)
  // Apply Kernel                         (j,i,k)
  Real four_pi_G = pmy_block_->pgrav->four_pi_G;
#if defined(GRAV_PERIODIC)
  Real kx,ky,kz;
  Real kernel;
  for (int j=0; j<slow_nx2; j++) {
    for (int i=0; i<slow_nx1; i++) {
      for (int k=0; k<slow_nx3; k++) {
        int idx = k + slow_nx3*(i + slow_nx1*j);
        kx = TWO_PI*(slow_ilo + i)/Nx1;
        ky = TWO_PI*(slow_jlo + j)/Nx2;
        kz = TWO_PI*(slow_klo + k)/Nx3;
        if (((slow_ilo+i) + (slow_jlo+j) + (slow_klo+k)) == 0) {
          kernel = 0.0;
        }
        else {
          kernel = -four_pi_G / ((2. - 2.*std::cos(kx))/dx1sq_ +
                                 (2. - 2.*std::cos(ky))/dx2sq_ +
                                 (2. - 2.*std::cos(kz))/dx3sq_);
        }
        in_[idx] *= kernel;
      }
    }
  }
#elif defined(GRAV_DISK)
  Real kx,ky,kz,kxy;
  Real kernel_e,kernel_o;
  Real time = pmy_block_->pmy_mesh->time;
  Real qomt = qshear_*Omega_0_*time;
  for (int j=0; j<slow_nx2; j++) {
    for (int i=0; i<slow_nx1; i++) {
      for (int k=0; k<slow_nx3; k++) {
        int idx = k + slow_nx3*(i + slow_nx1*j);
        kx = TWO_PI*(slow_ilo + i)/Nx1;
        ky = TWO_PI*(slow_jlo + j)/Nx2;
        kz = TWO_PI*(slow_klo + k)/Nx3;
        kxy = std::sqrt((2. - 2.*std::cos(kx + qomt*(Lx1_/Lx2_)*(Nx2/Nx1)*ky))/dx1sq_ +
                        (2. - 2.*std::cos(ky))/dx2sq_);
        if (kxy==0) {
          kernel_e = 0;
          kernel_o = -four_pi_G*(Lx3_/Nx3) / (1. - std::cos(kz+PI/Nx3));
        }
        else {
          kernel_e = -0.5*four_pi_G/kxy*(1. - std::exp(-kxy*Lx3_))*
            std::sinh(kxy*Lx3_/Nx3) / (std::cosh(kxy*Lx3_/Nx3) - std::cos(kz));
          kernel_o = -0.5*four_pi_G/kxy*(1. + std::exp(-kxy*Lx3_))*
            std::sinh(kxy*Lx3_/Nx3) / (std::cosh(kxy*Lx3_/Nx3) - std::cos(kz+PI/Nx3));
        }
        in_e_[idx] *= kernel_e;
        in_o_[idx] *= kernel_o;
      }
    }
  }
#else
  std::stringstream msg;
  msg << "### FATAL ERROR in BlockFFTGravity::ApplyKernel" << std::endl
      << "Gravity boundary condition other than periodic or disk is not yet implemented"
      << std::endl;
  ATHENA_ERROR(msg);
  return;
#endif

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
