//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file block_fft_gravity.cpp
//! \brief implementation of functions in class BlockFFTGravity

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
//! \brief BlockFFTGravity constructor

BlockFFTGravity::BlockFFTGravity(MeshBlock *pmb, ParameterInput *pin)
    : BlockFFT(pmb), I_(0.0,1.0),
      dx1sq_(SQR(pmb->pcoord->dx1v(NGHOST))),
      dx2sq_(SQR(pmb->pcoord->dx2v(NGHOST))),
      dx3sq_(SQR(pmb->pcoord->dx3v(NGHOST))),
      Lx1_(pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min),
      Lx2_(pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min),
      Lx3_(pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min),
      SHEAR_PERIODIC(pmb->pmy_mesh->shear_periodic) {
  gtlist_ = new FFTGravitySolverTaskList(pin, pmb->pmy_mesh);
  gbflag = GetGravityBoundaryFlag(pin->GetString("self_gravity", "grav_bc"));
  Omega_0_ = pin->GetReal("orbital_advection","Omega0");
  qshear_  = pin->GetReal("orbital_advection","qshear");
  in2_ = new std::complex<Real>[nx1*nx2*nx3];
  in_e_ = new std::complex<Real>[nx1*nx2*nx3];
  in_o_ = new std::complex<Real>[nx1*nx2*nx3];
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::~BlockFFTGravity()
//! \brief BlockFFTGravity destructor

BlockFFTGravity::~BlockFFTGravity() {
  delete gtlist_;
  delete[] in2_;
  delete[] in_e_;
  delete[] in_o_;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFTGravity::ExecuteForward()
//! \brief Executes forward transform

void BlockFFTGravity::ExecuteForward() {
#ifdef FFT
#ifdef MPI_PARALLEL
  if (gbflag==GravityBoundaryFlag::periodic) {
    if (SHEAR_PERIODIC) {
      FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR *>(in_);
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
      // slow_forward
      pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_FORWARD,pf3d->fft_slow);
    } else {
      BlockFFT::ExecuteForward();
    }
  } else if (gbflag==GravityBoundaryFlag::disk) {
    FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR *>(in_);
    // block2mid
    pf3d->remap(data,data,pf3d->remap_premid);
    // mid_forward
    pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_FORWARD,pf3d->fft_mid);
    if (SHEAR_PERIODIC) {
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
    }
    // mid2fast
    pf3d->remap(data,data,pf3d->remap_midfast);
    // fast_forward
    pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_FORWARD,pf3d->fft_fast);
    // fast2slow
    pf3d->remap(data,data,pf3d->remap_fastslow);
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
    pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(in_e_),FFTW_FORWARD,pf3d->fft_slow);
    pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(in_o_),FFTW_FORWARD,pf3d->fft_slow);
  } else if (gbflag==GravityBoundaryFlag::open) {
    std::stringstream msg;
    msg << "### FATAL ERROR in BlockFFTGravity::ExecuteForward" << std::endl
        << "open boundary condition is not yet implemented" << std::endl;
    ATHENA_ERROR(msg);
    return;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in BlockFFTGravity::ExecuteForward" << std::endl
        << "invalid gravity boundary condition" << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
#endif // MPI_PARALLEL
#endif // FFT
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFTGravity::ExecuteBackward()
//! \brief Executes backward transform

void BlockFFTGravity::ExecuteBackward() {
#ifdef FFT
#ifdef MPI_PARALLEL
  if (gbflag==GravityBoundaryFlag::periodic) {
    if (SHEAR_PERIODIC) {
      FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR *>(in_);
      // slow_backward
      pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_BACKWARD,pf3d->fft_slow);
      // slow2fast
      pf3d->remap(data,data,pf3d->remap_slowfast);
      // fast_backward
      pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_BACKWARD,pf3d->fft_fast);
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
      // mid2block
      if (pf3d->remap_postmid) pf3d->remap(data,data,pf3d->remap_postmid);
      // multiply norm factor
      for (int i=0; i<2*nx1*nx2*nx3; ++i) data[i] /= (Nx1*Nx2*Nx3);
    } else {
      BlockFFT::ExecuteBackward();
    }
  } else if (gbflag==GravityBoundaryFlag::disk) {
    FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR *>(in_);
    // slow_backward
    pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(in_e_),FFTW_BACKWARD,pf3d->fft_slow);
    pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(in_o_),FFTW_BACKWARD,pf3d->fft_slow);
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
    pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_BACKWARD,pf3d->fft_fast);
    // fast2mid
    pf3d->remap(data,data,pf3d->remap_fastmid);
    if (SHEAR_PERIODIC) {
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
    }
    // mid_backward
    pf3d->perform_ffts(reinterpret_cast<FFT_DATA *>(data),FFTW_BACKWARD,pf3d->fft_mid);
    // mid2block
    if (pf3d->remap_postmid) pf3d->remap(data,data,pf3d->remap_postmid);
    // multiply norm factor
    for (int i=0; i<2*nx1*nx2*nx3; ++i) data[i] *= Lx3_/(2.0*Nx1*Nx2*SQR(Nx3));
  } else if (gbflag==GravityBoundaryFlag::open) {
    std::stringstream msg;
    msg << "### FATAL ERROR in BlockFFTGravity::ExecuteBackward" << std::endl
        << "open boundary condition is not yet implemented" << std::endl;
    ATHENA_ERROR(msg);
    return;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in BlockFFTGravity::ExecuteForward" << std::endl
        << "invalid gravity boundary condition" << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
#endif // MPI_PARALLEL
#endif // FFT
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFTGravity::ApplyKernel()
//! \brief Applies kernel appropriate for each gravity boundary condition

void BlockFFTGravity::ApplyKernel() {
  //      ACTION                     (slow, mid, fast)
  // Initial block decomposition          (k,j,i)
  // Remap to x-pencil, perform FFT in i  (k,j,i)
  // Remap to y-pencil, perform FFT in j  (i,k,j)
  // Remap to z-pencil, perform FFT in k  (j,i,k)
  // Apply Kernel                         (j,i,k)
  Real four_pi_G = pmy_block_->pgrav->four_pi_G;
  if (gbflag==GravityBoundaryFlag::periodic) {
    Real kx,kxt,ky,kz;
    Real kernel;

    for (int j=0; j<slow_nx2; j++) {
      for (int i=0; i<slow_nx1; i++) {
        for (int k=0; k<slow_nx3; k++) {
          int idx = k + slow_nx3*(i + slow_nx1*j);
          kx = TWO_PI*(Real)(slow_ilo + i)/(Real)Nx1;
          ky = TWO_PI*(Real)(slow_jlo + j)/(Real)Nx2;
          kz = TWO_PI*(Real)(slow_klo + k)/(Real)Nx3;
          if (SHEAR_PERIODIC) {
            kxt = kx + rshear_*(Real)(Nx2)/(Real)(Nx1)*ky;
          } else {
            kxt = kx;
          }
          if (((slow_ilo+i) + (slow_jlo+j) + (slow_klo+k)) == 0) {
            kernel = 0.0;
          } else {
            kernel = -four_pi_G / ((2. - 2.*std::cos(kxt))/dx1sq_ +
                                   (2. - 2.*std::cos(ky ))/dx2sq_ +
                                   (2. - 2.*std::cos(kz ))/dx3sq_);
          }
          in_[idx] *= kernel;
        }
      }
    }
  } else if (gbflag==GravityBoundaryFlag::disk) {
    Real kx,kxt,ky,kz,kxy;
    Real kernel_e,kernel_o;

    for (int j=0; j<slow_nx2; j++) {
      for (int i=0; i<slow_nx1; i++) {
        for (int k=0; k<slow_nx3; k++) {
          int idx = k + slow_nx3*(i + slow_nx1*j);
          kx = TWO_PI*(Real)(slow_ilo + i)/(Real)Nx1;
          ky = TWO_PI*(Real)(slow_jlo + j)/(Real)Nx2;
          kz = TWO_PI*(Real)(slow_klo + k)/(Real)Nx3;
          if (SHEAR_PERIODIC) {
            kxt = kx + rshear_*(Real)(Nx2)/(Real)(Nx1)*ky;
          } else {
            kxt = kx;
          }
          kxy = std::sqrt((2.-2.*std::cos(kxt))/dx1sq_ +
                          (2.-2.*std::cos(ky ))/dx2sq_);
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
        }
      }
    }
  } else if (gbflag==GravityBoundaryFlag::open) {
    std::stringstream msg;
    msg << "### FATAL ERROR in BlockFFTGravity::ApplyKernel" << std::endl
        << "open boundary condition is not yet implemented" << std::endl;
    ATHENA_ERROR(msg);
    return;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in BlockFFTGravity::ExecuteForward" << std::endl
        << "invalid gravity boundary condition" << std::endl;
    ATHENA_ERROR(msg);
    return;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BlockFFTGravity::Solve(int stage)
//! \brief Solves Poisson equation and calls FFTGravityTaskList

void BlockFFTGravity::Solve(int stage) {
#ifdef FFT
#ifdef MPI_PARALLEL
  if (SHEAR_PERIODIC) {
    Real time = pmy_block_->pmy_mesh->time;
    Real qomt = qshear_*Omega_0_*time;
    AthenaArray<Real> rho;
    Real p,eps;

    // left integer point
    p = std::floor(qomt*Lx1_/Lx2_*(Real)Nx2);
    eps = qomt*Lx1_/Lx2_*(Real)Nx2 - p;
    rshear_ = p/(Real)Nx2;
    rho.InitWithShallowSlice(pmy_block_->phydro->u,4,IDN,1);
    LoadSource(rho);
    ExecuteForward();
    ApplyKernel();
    ExecuteBackward();
    std::memcpy(in2_, in_, sizeof(std::complex<Real>)*nx1*nx2*nx3);

    // right integer point
    p = std::floor(qomt*Lx1_/Lx2_*(Real)Nx2) + 1.;
    rshear_ = p/(Real)Nx2;
    rho.InitWithShallowSlice(pmy_block_->phydro->u,4,IDN,1);
    LoadSource(rho);
    ExecuteForward();
    ApplyKernel();
    ExecuteBackward();

    // linear interpolation in time
    FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR *>(in_);
    FFT_SCALAR *data2 = reinterpret_cast<FFT_SCALAR *>(in2_);
    for (int i=0; i<2*nx1*nx2*nx3; ++i) {
      data[i] = (1.-eps)*data2[i] + eps*data[i];
    }
  } else {
    AthenaArray<Real> rho;
    rho.InitWithShallowSlice(pmy_block_->phydro->u,4,IDN,1);
    LoadSource(rho);
    ExecuteForward();
    ApplyKernel();
    ExecuteBackward();
  }
  RetrieveResult(pmy_block_->pgrav->phi);
#else
  std::stringstream msg;
  msg << "### FATAL ERROR in BlockFFTGravity::Solve" << std::endl
      << "BlockFFTGravity only works with MPI" << std::endl;
  ATHENA_ERROR(msg);
  return;
#endif // MPI_PARALLEL
#else
  std::stringstream msg;
  msg << "### FATAL ERROR in BlockFFTGravity::Solve" << std::endl
      << "BlockFFTGravity only works with FFT" << std::endl;
  ATHENA_ERROR(msg);
  return;
#endif // FFT

  gtlist_->DoTaskListOneStage(pmy_block_->pmy_mesh, stage);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn GetGravityBoundaryFlag(std::string input_string)
//! \brief Parses input string to return scoped enumerator flag specifying gravity
//! boundary condition. Typically called in BlockFFTGravity() ctor.

GravityBoundaryFlag GetGravityBoundaryFlag(const std::string& input_string) {
  if (input_string == "periodic") {
    return GravityBoundaryFlag::periodic;
  } else if (input_string == "disk") {
    return GravityBoundaryFlag::disk;
  } else if (input_string == "open") {
    return GravityBoundaryFlag::open;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in GetGravityBoundaryFlag" << std::endl
        << "Input string=" << input_string << "\n"
        << "is an invalid boundary type" << std::endl;
    ATHENA_ERROR(msg);
  }
}
