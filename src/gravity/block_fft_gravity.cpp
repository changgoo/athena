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
#include "../coordinates/coordinates.hpp"
#include "block_fft_gravity.hpp"

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::BlockFFTGravity(MeshBlock *pmb, ParameterInput *pin)
//  \brief BlockFFTGravity constructor
BlockFFTGravity::BlockFFTGravity(MeshBlock *pmb, ParameterInput *pin)
    : BlockFFT(pmb), I_(0.0,1.0),
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

  // cast std::complex* to FFT_SCALAR*
  FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR*>(in_);
  // block2mid
  pf3d->remap(data,data,pf3d->remap_premid);
  // mid_forward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_FORWARD,pf3d->fft_mid);
  // apply phase shift for shearing BC
  for (int i=0; i<mid_nx1; i++) {
    for (int k=0; k<mid_nx3; k++) {
      for (int j=0; j<mid_nx2; j++) {
        int idx = j + mid_nx2*(k + mid_nx3*i);
        in_[idx] *= std::exp(-TWO_PI*I_*rshear_*
            (Real)(mid_jlo+j)*(Real)(mid_ilo+i)/(Real)Nx1);
      }
    }
  }
  // mid2fast
  pf3d->remap(data,data,pf3d->remap_midfast);
  // fast_forward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_FORWARD,pf3d->fft_fast);
  // fast2slow
  pf3d->remap(data,data,pf3d->remap_fastslow);
  // TODO(SMOON) comment out below to disable z transform for 2D solver
  std::memcpy(in_e_, in_, sizeof(std::complex<Real>)*nx1*nx2*nx3); // even term (l=2p)
  std::memcpy(in_o_, in_, sizeof(std::complex<Real>)*nx1*nx2*nx3); // odd term (l=2p+1)
  // apply odd term phase shift for vertical open BC
  for (int j=0; j<slow_nx2; j++) {
    for (int i=0; i<slow_nx1; i++) {
      for (int k=0; k<slow_nx3; k++) {
        int idx = k + slow_nx3*(i + slow_nx1*j);
        in_o_[idx] *= std::exp(-PI*I_*(Real)k/(Real)Nx3);
      }
    }
  }
  // slow_forward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA*>(in_e_),FFTW_FORWARD,pf3d->fft_slow);
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA*>(in_o_),FFTW_FORWARD,pf3d->fft_slow);
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
#if defined(FFT) && defined(MPI_PARALLEL)
#ifdef GRAV_DISK

  // cast std::complex* to FFT_SCALAR*
  FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR*>(in_);

  // TODO(SMOON) comment out below to disable z transform for 2D solver
  // slow_backward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA*>(in_e_),FFTW_BACKWARD,pf3d->fft_slow);
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA*>(in_o_),FFTW_BACKWARD,pf3d->fft_slow);
  // combine even and odd terms for vertical open BC
  for (int j=0; j<slow_nx2; j++) {
    for (int i=0; i<slow_nx1; i++) {
      for (int k=0; k<slow_nx3; k++) {
        int idx = k + slow_nx3*(i + slow_nx1*j);
        in_[idx] = in_e_[idx] + std::exp(PI*I_*(Real)k/(Real)Nx3)*in_o_[idx];
      }
    }
  }
  // slow2fast
  pf3d->remap(data,data,pf3d->remap_slowfast);
  // fast_backward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA*>(data),FFTW_BACKWARD,pf3d->fft_fast);
  // fast2mid
  pf3d->remap(data,data,pf3d->remap_fastmid);
  // apply phase shift for shearing BC
  for (int i=0; i<mid_nx1; i++) {
    for (int k=0; k<mid_nx3; k++) {
      for (int j=0; j<mid_nx2; j++) {
        int idx = j + mid_nx2*(k + mid_nx3*i);
        in_[idx] *= std::exp(TWO_PI*I_*rshear_*
            (Real)(mid_jlo+j)*(Real)(mid_ilo+i)/(Real)Nx1);
      }
    }
  }
  // mid_backward
  pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_BACKWARD,pf3d->fft_mid);
  if (pf3d->remap_postmid)
    // mid2block
    pf3d->remap(data,data,pf3d->remap_postmid);

  // multiply norm factor
  // TODO(SMOON) comment out below to disable z transform for 2D solver
  for (int i=0; i<2*nx1*nx2*nx3; ++i) data[i] *= Lx3_/(2.0*Nx1*Nx2*SQR(Nx3));
  //TODO(SMOON) norm factor for 2D solver
//  for (int i=0;i<2*nx1*nx2*nx3;++i) data[i] *= 1.0/(Nx1*Nx2);

#else
  BlockFFT::ExecuteBackward();
#endif // GRAV_BC_OPTIONS

#endif // FFT & MPI_PARALLEL

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFTGravity::ApplyKernel()
//  \brief Apply kernel for fully periodic BC
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
        } else {
          kernel = -four_pi_G / ((2. - 2.*std::cos(kx))/dx1sq_ +
                                 (2. - 2.*std::cos(ky))/dx2sq_ +
                                 (2. - 2.*std::cos(kz))/dx3sq_);
        }
        in_[idx] *= kernel;
      }
    }
  }
#elif defined(GRAV_DISK)
  Real kx,kxt,ky,kz,kxy;
  Real kernel_e,kernel_o;

  for (int j=0; j<slow_nx2; j++) {
    for (int i=0; i<slow_nx1; i++) {
      for (int k=0; k<slow_nx3; k++) {
        int idx = k + slow_nx3*(i + slow_nx1*j);
        kx = TWO_PI*(Real)(slow_ilo + i)/((Real)Nx1);
        kxt = kx+rshear_*((Real)(Nx2)/(Real)(Nx1))*ky;
        ky = TWO_PI*(Real)(slow_jlo + j)/((Real)Nx2);
        kz = TWO_PI*(Real)(slow_klo + k)/((Real)Nx3);

        kxy = std::sqrt((2.-2.*std::cos(kxt))/dx1sq_ + (2.-2.*std::cos(ky))/dx2sq_);

        // continuous kernel
//        kxy = std::sqrt( SQR(TWO_PI*(slow_ilo+i)/Lx1_ + qomt*TWO_PI*(slow_jlo+j)/Lx2_)
//                       + SQR(TWO_PI*(slow_jlo+j)/Lx2_) );


        // TODO(SMOON) comment out below to disable z transform for 2D solver
        if ((slow_ilo+i==0)&&(slow_jlo+j==0)) {
          kernel_e = k==0 ? 0.5*four_pi_G*Lx3_*(Real)(Nx3) : 0;
          kernel_o = -four_pi_G*(Lx3_/(Real)(Nx3)) / (1. - std::cos(kz+PI/(Real)(Nx3)));
        } else {
          kernel_e = -0.5*four_pi_G/kxy*(1. - std::exp(-kxy*Lx3_))*
            std::sinh(kxy*Lx3_/(Real)(Nx3)) /
            (std::cosh(kxy*Lx3_/(Real)(Nx3)) - std::cos(kz));
          kernel_o = -0.5*four_pi_G/kxy*(1. + std::exp(-kxy*Lx3_))*
            std::sinh(kxy*Lx3_/(Real)(Nx3)) /
            (std::cosh(kxy*Lx3_/(Real)(Nx3)) - std::cos(kz+PI/(Real)(Nx3)));
        }
        in_e_[idx] *= kernel_e;
        in_o_[idx] *= kernel_o;

        // TODO(SMOON) this is temporary 2D solver
//        if ((slow_ilo+i==0)&&(slow_jlo+j==0)) {
//          kernel_e = 0.0;
//        }
//        else {
//          kernel_e = -four_pi_G/SQR(kxy);
//        }
//        in_[idx] *= kernel_e;
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
  Real time = pmy_block_->pmy_mesh->time;
  Real qomt = qshear_*Omega_0_*time;
  int p = std::nearbyint(qomt*Lx1_/Lx2_*(Real)Nx2);
  // force sheared distance == (integer) * dy
  rshear_ = (Real)p/(Real)Nx2;
  // continuous sheared distance (this introduces unwanted harmonic solutions)
//  rshear_ = qomt*Lx1_/Lx2_;

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
