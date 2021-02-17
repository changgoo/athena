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
    : BlockFFT(pmb),
      SHEAR_PERIODIC(pmb->pmy_mesh->shear_periodic),
      dx1sq_(SQR(pmb->pcoord->dx1v(NGHOST))),
      dx2sq_(SQR(pmb->pcoord->dx2v(NGHOST))),
      dx3sq_(SQR(pmb->pcoord->dx3v(NGHOST))),
      Lx1_(pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min),
      Lx2_(pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min),
      Lx3_(pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min),
      I_(0.0,1.0) {
  gtlist_ = new FFTGravitySolverTaskList(pin, pmb->pmy_mesh);
  gbflag = GetGravityBoundaryFlag(pin->GetString("self_gravity", "grav_bc"));
  Omega_0_ = pin->GetReal("orbital_advection","Omega0");
  qshear_  = pin->GetReal("orbital_advection","qshear");
  in2_ = new std::complex<Real>[nx1*nx2*nx3];
  in_e_ = new std::complex<Real>[nx1*nx2*nx3];
  in_o_ = new std::complex<Real>[nx1*nx2*nx3];
  grf_ = new std::complex<Real>[8*fast_nx3*fast_nx2*fast_nx1];
#ifdef FFT
#ifdef MPI_PARALLEL
  // setup fft in the 8x extended domain for the Green's function
  int permute=2; // will make output array (slow,mid,fast) = (y,x,z) = (j,i,k)
  int fftsize, sendsize, recvsize; // to be returned from setup
  pf3dgrf_ = new FFTMPI_NS::FFT3d(MPI_COMM_WORLD,2);
  pf3dgrf_->setup(2*Nx1, 2*Nx2, 2*Nx3,
                  2*fast_ilo, 2*fast_ihi+1, 2*fast_jlo, 2*fast_jhi+1,
                  2*fast_klo, 2*fast_khi+1, 2*slow_ilo, 2*slow_ihi+1,
                  2*slow_jlo, 2*slow_jhi+1, 2*slow_klo, 2*slow_khi+1,
                  permute, fftsize, sendsize, recvsize);
  if (gbflag==GravityBoundaryFlag::open)
    InitGreen();
#endif
#endif
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::~BlockFFTGravity()
//! \brief BlockFFTGravity destructor

BlockFFTGravity::~BlockFFTGravity() {
  delete gtlist_;
  delete[] in2_;
  delete[] in_e_;
  delete[] in_o_;
  delete[] grf_;
#ifdef FFT
#ifdef MPI_PARALLEL
  delete pf3dgrf_;
#endif
#endif
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
    // output Green's function as pgrav->phi
    for (int k=0; k<fast_nx3; k++) {
      for (int j=0; j<fast_nx2; j++) {
        for (int i=0; i<fast_nx1; i++) {
          int idx = i + fast_nx1*(j + fast_nx2*k);
          int idx2 = (fast_nx1+i) + (2*fast_nx1)*(j + (2*fast_nx2)*k);
          in_[idx] = grf_[idx2];
        }
      }
    }
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
    // For shearing-periodic BC, we use a 'phase shift' method, instead of
    // a roll-unroll method in old Athena. It was found that the phase shift
    // method introduces spurious error when the sheared distance at the boundary
    // is not an integer multiple of the cell width. To cure this, we solve the
    // Poisson equation at the nearest two 'integer times' when the sheared distance
    // is an integer multiple of the cell width, and then linearly interpolate the
    // solution in time.
    Real time = pmy_block_->pmy_mesh->time;
    Real qomt = qshear_*Omega_0_*time;
    AthenaArray<Real> rho;
    Real p,eps;

    // left integer time
    p = std::floor(qomt*Lx1_/Lx2_*(Real)Nx2);
    eps = qomt*Lx1_/Lx2_*(Real)Nx2 - p;
    rshear_ = p/(Real)Nx2;
    rho.InitWithShallowSlice(pmy_block_->phydro->u,4,IDN,1);
    LoadSource(rho);
    ExecuteForward();
    ApplyKernel();
    ExecuteBackward();
    std::memcpy(in2_, in_, sizeof(std::complex<Real>)*nx1*nx2*nx3);

    // right integer time
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
    RetrieveResult(pmy_block_->pgrav->phi);
  } else if (gbflag==GravityBoundaryFlag::open) {
    // For open boundary condition, we use a convolution method in which the
    // 8x-extended domain is assumed. Instead of zero-padding the density, we
    // multiply the appropriate phase shift to each parity and then combine them
    // to compute the full 8x-extended convolution.
    AthenaArray<Real> rho;
    rho.InitWithShallowSlice(pmy_block_->phydro->u,4,IDN,1);
    for (int pz=0; pz<=1; ++pz) {
      for (int py=0; py<=1; ++py) {
        for (int px=0; px<=1; ++px) {
          LoadOBCSource(rho,px,py,pz);
          ExecuteForward();
          MultiplyGreen(px,py,pz);
          ExecuteBackward();
          RetrieveOBCResult(rho,px,py,pz);
        }
      }
    }
  } else {
    // Periodic or disk BC without shearing box
    AthenaArray<Real> rho;
    rho.InitWithShallowSlice(pmy_block_->phydro->u,4,IDN,1);
    LoadSource(rho);
    ExecuteForward();
    ApplyKernel();
    ExecuteBackward();
    RetrieveResult(pmy_block_->pgrav->phi);
  }
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

  // apply physical BC
  if (gbflag==GravityBoundaryFlag::disk) {
    if (pmy_block_->loc.lx3==0) {
      for (int k=1; k<=NGHOST; k++) {
        for (int j=js-NGHOST; j<=je+NGHOST; j++) {
          for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
            // constant extrapolation
            pmy_block_->pgrav->phi(ks-k,j,i) = pmy_block_->pgrav->phi(ks,j,i);
            // linear extrapolation
//            pmy_block_->pgrav->phi(ks-k,j,i) = pmy_block_->pgrav->phi(ks,j,i)
//                + k*(pmy_block_->pgrav->phi(ks,j,i) - pmy_block_->pgrav->phi(ks+1,j,i));
          }
        }
      }
    }
    if (pmy_block_->loc.lx3==Nx3/nx3-1) {
      for (int k=1; k<=NGHOST; k++) {
        for (int j=js-NGHOST; j<=je+NGHOST; j++) {
          for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
            // constant extrapolation
            pmy_block_->pgrav->phi(ke+k,j,i) = pmy_block_->pgrav->phi(ke,j,i);
            // linear extrapolation
//            pmy_block_->pgrav->phi(ke+k,j,i) = pmy_block_->pgrav->phi(ke,j,i)
//                + k*(pmy_block_->pgrav->phi(ke,j,i) - pmy_block_->pgrav->phi(ke-1,j,i));
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::InitGreen()
//! \brief Initialize Green's function and its Fourier transform for open BC.

void BlockFFTGravity::InitGreen() {
  Real gconst = pmy_block_->pgrav->four_pi_G/(4.0*PI);
  for (int k=0; k<2*fast_nx3; k++) {
    for (int j=0; j<2*fast_nx2; j++) {
      for (int i=0; i<2*fast_nx1; i++) {
        // get global index in 8x mesh
        int gi = 2*fast_ilo + i;
        int gj = 2*fast_jlo + j;
        int gk = 2*fast_klo + k;
        // restrict index range to [-Nx, Nx-1]
        gi = (gi+Nx1)%(2*Nx1) - Nx1;
        gj = (gj+Nx2)%(2*Nx2) - Nx2;
        gk = (gk+Nx3)%(2*Nx3) - Nx3;
        // point-mass Green's function
        int idx = i + (2*fast_nx1)*(j + (2*fast_nx2)*k);
        if ((gi==0)&&(gj==0)&&(gk==0)) {
          grf_[idx] = {0.0, 0.0};
        } else {
          grf_[idx] = {-gconst/std::sqrt(SQR(gi)*dx1sq_ +
                                         SQR(gj)*dx2sq_ +
                                         SQR(gk)*dx3sq_), 0.0};
        }
        // TODO(SMOON) add integrated Green's function
      }
    }
  }
  // Apply DFT to the Green's function
#ifdef FFT
#ifdef MPI_PARALLEL
  FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR*>(grf_);
  // block2fast
  if (pf3d->remap_prefast) {
    std::cout << "### Warning in BlockFFTGravity::InitGreen()" << std::endl
              << "Input data layout for the Green's function is not "
              << "the x-pencil decomposition as desired." << std::endl;
    pf3d->remap(data,data,pf3d->remap_prefast);
  }
  // fast_forward
  pf3dgrf_->perform_ffts(reinterpret_cast<FFT_DATA *>(data), FFTW_FORWARD,
                         pf3dgrf_->fft_fast);
  // fast2mid
  pf3dgrf_->remap(data,data,pf3dgrf_->remap_fastmid);
  // mid_forward
  pf3dgrf_->perform_ffts(reinterpret_cast<FFT_DATA *>(data), FFTW_FORWARD,
                         pf3dgrf_->fft_mid);
  // mid2slow
  pf3dgrf_->remap(data,data,pf3dgrf_->remap_midslow);
  // slow_forward
  pf3dgrf_->perform_ffts(reinterpret_cast<FFT_DATA *>(data), FFTW_FORWARD,
                         pf3dgrf_->fft_slow);
#endif
#endif
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::LoadOBCSource(const AthenaArray<Real> &src, int px, int py, int pz)
//! \brief Load source and multiply phase shift term for the open boundary condition.
void BlockFFTGravity::LoadOBCSource(const AthenaArray<Real> &src, int px, int py, int pz) {
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::RetrieveOBCResult(AthenaArray<Real> &dst, int px, int py, int pz)
//! \brief Retrieve result and multiply phase shift term for the open boundary condition.
void BlockFFTGravity::RetrieveOBCResult(AthenaArray<Real> &dst, int px, int py, int pz) {
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::MultiplyGreen(int px, int py, int pz)
//! \brief Multiply Green's function
void BlockFFTGravity::MultiplyGreen(int px, int py, int pz) {
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
