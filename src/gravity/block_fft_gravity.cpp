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
      dx1_(pmb->pcoord->dx1v(NGHOST)),
      dx2_(pmb->pcoord->dx2v(NGHOST)),
      dx3_(pmb->pcoord->dx3v(NGHOST)),
      dx1sq_(SQR(pmb->pcoord->dx1v(NGHOST))),
      dx2sq_(SQR(pmb->pcoord->dx2v(NGHOST))),
      dx3sq_(SQR(pmb->pcoord->dx3v(NGHOST))),
      Lx1_(pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min),
      Lx2_(pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min),
      Lx3_(pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min),
      I_(0.0,1.0) {
  gtlist_ = new FFTGravitySolverTaskList(pin, pmb->pmy_mesh);
  gbflag = GetGravityBoundaryFlag(pin->GetString("self_gravity", "grav_bc"));
  grfflag = GetGreenFuncFlag(pin->GetOrAddString("self_gravity", "green_function",
                                                 "cell_averaged"));
  Omega_0_ = pin->GetReal("orbital_advection","Omega0");
  qshear_  = pin->GetReal("orbital_advection","qshear");
  in2_ = new std::complex<Real>[nx1*nx2*nx3];
  in_e_ = new std::complex<Real>[nx1*nx2*nx3];
  in_o_ = new std::complex<Real>[nx1*nx2*nx3];
  grf_ = new std::complex<Real>[8*fast_nx3*fast_nx2*fast_nx1];
#ifdef MPI_PARALLEL
#ifdef FFT
  // setup fft in the 8x extended domain for the Green's function
  int permute=2; // will make output array (slow,mid,fast) = (y,x,z) = (j,i,k)
  int fftsize, sendsize, recvsize;
  pf3dgrf_ = new FFTMPI_NS::FFT3d(MPI_COMM_WORLD,2);
  pf3dgrf_->setup(2*Nx1, 2*Nx2, 2*Nx3,
                  2*in_ilo, 2*in_ihi+1, 2*in_jlo, 2*in_jhi+1,
                  2*in_klo, 2*in_khi+1, 2*slow_ilo, 2*slow_ihi+1,
                  2*slow_jlo, 2*slow_jhi+1, 2*slow_klo, 2*slow_khi+1,
                  permute, fftsize, sendsize, recvsize);

  // initialize Green's function for open BC
  if (gbflag==GravityBoundaryFlag::open)
    InitGreen();
#endif
  send_buf.NewAthenaArray(nx3, nx2);
  recv_buf.NewAthenaArray(nx3, nx2);
  roll_var.NewAthenaArray(nx3+2*NGHOST, nx2+2*NGHOST, nx1+2*NGHOST);
  roll_buf.NewAthenaArray(nx3+2*NGHOST, nx2+2*NGHOST, nx1+2*NGHOST);
#endif

  // Compatibility checks
  if ((SHEAR_PERIODIC)&&(gbflag==GravityBoundaryFlag::open)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in BlockFFTGravity constructor" << std::endl
        << "open BC gravity is not compatible with shearing box" << std::endl;
    ATHENA_ERROR(msg);
  }

  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in BlockFFTGravity constructor" << std::endl
        << "BlockFFTGravity only compatible with cartesian coord" << std::endl;
    ATHENA_ERROR(msg);
  }
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
    BlockFFT::ExecuteForward();
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
    BlockFFT::ExecuteBackward();
    // divide by 8 to account for the normalization in the enlarged domain
    for (int i=0; i<nx1*nx2*nx3; ++i) in_[i] *= 0.125;
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

          if (grfflag==GreenFuncFlag::cell_averaged) {
            Real cos_e = std::cos(kz);
            Real cos_o = std::cos(kz + PI/Nx3);
            Real exp_kxydz = std::exp(-0.5*kxy*dx3_);
            Real exp_kxydz1 = SQR(exp_kxydz); // for std::exp(-kxy*dx3_)
            Real exp_kxydz2 = SQR(exp_kxydz1); // for std::exp(-2*kxy*dx3_)
            Real exp_kxydz3 = exp_kxydz1*exp_kxydz2; // for std::exp(-3*kxy*dx3_)
            Real exp_kxyLz = std::exp(-kxy*Lx3_);
            Real exp_kxyLz1 = std::exp(-kxy*(Lx3_-0.5*dx3_));

            if ((slow_ilo+i==0)&&(slow_jlo+j==0)) {
              kernel_e = k==0 ? 0.5*four_pi_G*dx3_*(SQR(Nx3) + 0.25)
                              : 0.125*four_pi_G*dx3_;
              kernel_o = -four_pi_G*dx3_ / (1. - cos_o) + 0.125*four_pi_G*dx3_;
            } else {
              kernel_e = -0.5*four_pi_G/SQR(kxy)/dx3_*(2*(1. - exp_kxydz)
                  + (2*exp_kxydz*(1. - exp_kxydz1)*(cos_e - exp_kxydz1))
                      / (1. + exp_kxydz2 - 2.*exp_kxydz1*cos_e)
                  - exp_kxyLz1*(1. - exp_kxydz1 - exp_kxydz2 + exp_kxydz3)
                      / (1. + exp_kxydz2 - 2.*exp_kxydz1*cos_e));
              kernel_o = -0.5*four_pi_G/SQR(kxy)/dx3_*(2*(1. - exp_kxydz)
                  + (2*exp_kxydz*(1. - exp_kxydz1)*(cos_o - exp_kxydz1))
                      / (1. + exp_kxydz2 - 2.*exp_kxydz1*cos_o)
                  + exp_kxyLz1*(1. - exp_kxydz1 - exp_kxydz2 + exp_kxydz3)
                      / (1. + exp_kxydz2 - 2.*exp_kxydz1*cos_o));
            }
          } else if (grfflag==GreenFuncFlag::point_mass) {
            Real cos_e = std::cos(kz);
            Real cos_o = std::cos(kz + PI/Nx3);
            Real exp_kxyLz = std::exp(-kxy*Lx3_);
            Real exp_kxydz = std::exp(-0.5*kxy*dx3_);
            Real exp_kxydz1 = SQR(exp_kxydz); // for std::exp(-kxy*dx3_)
            Real exp_kxydz2 = SQR(exp_kxydz1); // for std::exp(-2*kxy*dx3_)
            if ((slow_ilo+i==0)&&(slow_jlo+j==0)) {
              kernel_e = k==0 ? 0.5*four_pi_G*dx3_*SQR(Nx3) : 0;
              kernel_o = -four_pi_G*dx3_ / (1. - cos_o);
            } else {
              kernel_e = -0.5*four_pi_G/kxy*(1. - exp_kxyLz)*(1. - exp_kxydz2)
                  / (1. + exp_kxydz2 - 2.*exp_kxydz1*cos_e);
              kernel_o = -0.5*four_pi_G/kxy*(1. + exp_kxyLz)*(1. - exp_kxydz2)
                  / (1. + exp_kxydz2 - 2.*exp_kxydz1*cos_o);
            }
          } else {
            std::stringstream msg;
            msg << "### FATAL ERROR in BlockFFTGravity::ApplyKernel" << std::endl
                << "invalid Green's function" << std::endl;
            ATHENA_ERROR(msg);
            return;
          }
          in_e_[idx] *= kernel_e;
          in_o_[idx] *= kernel_o;
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
  bool PHASE_SHIFT = false;
#ifdef FFT
#ifdef MPI_PARALLEL
  if (SHEAR_PERIODIC & PHASE_SHIFT) {
    // Use "phase shift" method for the shearing-periodic boundary condition.
    // It was found that the phase shift method introduces spurious error when the
    // sheared distance at the boundary is not an integer multiple of the cell width.
    // To cure this, we solve the Poisson equation at the nearest two 'integer times'
    // when the sheared distance is an integer multiple of the cell width, and then
    // linearly interpolate the solution in time.
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
  } else if (SHEAR_PERIODIC) {
    // Use roll-unroll method for the shearing-periodic boundary condition.
    Real time = pmy_block_->pmy_mesh->time;
    AthenaArray<Real> rho;
    rho.InitWithShallowSlice(pmy_block_->phydro->u,4,IDN,1);
    RollUnroll(rho, roll_var, time); // Transform density to the shearing coordinates (Roll)
    LoadSource(roll_var);
    // TODO Solve Poisson equation in the shearing coordinates
    RetrieveResult(pmy_block_->pgrav->phi);
    // TODO Transform potential to the original coordinates (Unroll)
  } else if (gbflag==GravityBoundaryFlag::open) {
    // For open boundary condition, we use a convolution method in which the
    // 8x-extended domain is assumed. Instead of zero-padding the density, we
    // multiply the appropriate phase shift to each parity and then combine them
    // to compute the full 8x-extended convolution.
    AthenaArray<Real> rho;
    rho.InitWithShallowSlice(pmy_block_->phydro->u,4,IDN,1);
    pmy_block_->pgrav->phi.ZeroClear();
    for (int pz=0; pz<=1; ++pz) {
      for (int py=0; py<=1; ++py) {
        for (int px=0; px<=1; ++px) {
          LoadOBCSource(rho,px,py,pz);
          ExecuteForward();
          MultiplyGreen(px,py,pz);
          ExecuteBackward();
          RetrieveOBCResult(pmy_block_->pgrav->phi,px,py,pz);
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
  SetPhysicalBoundaries();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::InitGreen()
//! \brief Initialize Green's function and its Fourier transform for open BC.

void BlockFFTGravity::InitGreen() {
  Real gconst = pmy_block_->pgrav->four_pi_G/(4.0*PI);
  Real dvol = dx1_*dx2_*dx3_;

  for (int k=0; k<2*nx3; k++) {
    for (int j=0; j<2*nx2; j++) {
      for (int i=0; i<2*nx1; i++) {
        // get global index in 8x enlarged mesh
        int gi = 2*in_ilo + i;
        int gj = 2*in_jlo + j;
        int gk = 2*in_klo + k;
        // roll the index to [-Nx, Nx-1] range
        gi = (gi+Nx1)%(2*Nx1) - Nx1;
        gj = (gj+Nx2)%(2*Nx2) - Nx2;
        gk = (gk+Nx3)%(2*Nx3) - Nx3;
        int idx = i + (2*nx1)*(j + (2*nx2)*k);
        if (grfflag==GreenFuncFlag::cell_averaged) {
          // cell-averaged Green's function
          grf_[idx]  = _GetIGF((-gi+0.5)*dx1_, (-gj+0.5)*dx2_, (-gk+0.5)*dx3_);
          grf_[idx] -= _GetIGF((-gi+0.5)*dx1_, (-gj+0.5)*dx2_, (-gk-0.5)*dx3_);
          grf_[idx] -= _GetIGF((-gi+0.5)*dx1_, (-gj-0.5)*dx2_, (-gk+0.5)*dx3_);
          grf_[idx] += _GetIGF((-gi+0.5)*dx1_, (-gj-0.5)*dx2_, (-gk-0.5)*dx3_);
          grf_[idx] -= _GetIGF((-gi-0.5)*dx1_, (-gj+0.5)*dx2_, (-gk+0.5)*dx3_);
          grf_[idx] += _GetIGF((-gi-0.5)*dx1_, (-gj+0.5)*dx2_, (-gk-0.5)*dx3_);
          grf_[idx] += _GetIGF((-gi-0.5)*dx1_, (-gj-0.5)*dx2_, (-gk+0.5)*dx3_);
          grf_[idx] -= _GetIGF((-gi-0.5)*dx1_, (-gj-0.5)*dx2_, (-gk-0.5)*dx3_);
        } else if (grfflag==GreenFuncFlag::point_mass) {
          // point-mass Green's function
          if ((gi==0)&&(gj==0)&&(gk==0)) {
            // avoid singularity at r=0
            grf_[idx] = 0.0;
          } else {
            grf_[idx] = 1./std::sqrt(SQR(gi*dx1_) + SQR(gj*dx2_) + SQR(gk*dx3_))*dvol;
          }
        } else {
          std::stringstream msg;
          msg << "### FATAL ERROR in BlockFFTGravity::InitGreen" << std::endl
              << "invalid Green's function" << std::endl;
          ATHENA_ERROR(msg);
          return;
        }
        grf_[idx] *= -gconst;
      }
    }
  }
// Apply DFT to the Green's function
#ifdef FFT
#ifdef MPI_PARALLEL
  FFT_SCALAR *data = reinterpret_cast<FFT_SCALAR*>(grf_);
  // block2fast
  if (pf3dgrf_->remap_prefast)
    pf3dgrf_->remap(data,data,pf3dgrf_->remap_prefast);
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
//! \fn BlockFFTGravity::LoadOBCSource(const AthenaArray<Real> &src, int px, int py,
//                                     int pz)
//! \brief Load source and multiply phase shift term for the open boundary condition.
void BlockFFTGravity::LoadOBCSource(const AthenaArray<Real> &src, int px, int py,
                                    int pz) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        int idx = (i-is) + nx1*((j-js) + nx2*(k-ks));
        int gi = in_ilo + i-is;
        int gj = in_jlo + j-js;
        int gk = in_klo + k-ks;
        Real phase = PI*(gi*px/(Real)Nx1+gj*py/(Real)Nx2+gk*pz/(Real)Nx3);
        in_[idx] = {src(k,j,i)*std::cos(phase),
                    -src(k,j,i)*std::sin(phase)};
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::RetrieveOBCResult(AthenaArray<Real> &dst, int px, int py,
//                                         int pz)
//! \brief Retrieve result and multiply phase shift term for the open boundary condition.
void BlockFFTGravity::RetrieveOBCResult(AthenaArray<Real> &dst, int px, int py, int pz) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        int idx = (i-is) + nx1*((j-js) + nx2*(k-ks));
        int gi = in_ilo + i-is;
        int gj = in_jlo + j-js;
        int gk = in_klo + k-ks;
        Real phase = PI*(gi*px/(Real)Nx1+gj*py/(Real)Nx2+gk*pz/(Real)Nx3);
        dst(k,j,i) += in_[idx].real()*std::cos(phase)
                   - in_[idx].imag()*std::sin(phase);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::MultiplyGreen(int px, int py, int pz)
//! \brief Multiply Green's function
void BlockFFTGravity::MultiplyGreen(int px, int py, int pz) {
  for (int j=0; j<slow_nx2; j++) {
    for (int i=0; i<slow_nx1; i++) {
      for (int k=0; k<slow_nx3; k++) {
        int idx = k + slow_nx3*(i + slow_nx1*j);
        int grfidx = (2*k+pz) + (2*slow_nx3)*((2*i+px) + (2*slow_nx1)*(2*j+py));
        in_[idx] *= grf_[grfidx];
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::RollUnroll(const AthenaArray<Real> &in, AthenaArray<Real> &out,
//                                  Real dt)
//! \brief Transform to the shearing coordinates
void BlockFFTGravity::RollUnroll(const AthenaArray<Real> &in, AthenaArray<Real> &out,
                                 Real dt) {
#ifdef MPI_PARALLEL
  int joffset,jremap;
  int sendto_id,getfrom_id,cnt,ierr;
  int remapvar_tag=1992; // TODO How should I set the tag?
  Real yshear,eps;
  Real joverlap,Ngrids;
  LogicalLocation loc, &my_loc=pmy_block_->loc;
  Coordinates *pc = pmy_block_->pcoord;
  MeshBlockTree *proot = &(pmy_block_->pmy_mesh->tree), *pleaf=nullptr;
  MPI_Request rq;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        roll_buf(k,j,i) = in(k,j,i); // + flux(j+1) - flux(j)
      }
    }
  }

  for (int i=is; i<=ie; i++) {
    yshear = -qshear_*Omega_0_*pc->x1v(i)*dt;
    joffset = yshear/dx2_;
    // Case A: shear in positive-y direction.
    if (joffset >= 0) {
      Ngrids = static_cast<int>(joffset/nx2);
      joverlap = joffset - Ngrids*nx2;
      // Step A.1: Send density at [je-(joverlap-1):je] to [js:js+(joverlap-1)] of
      // the target MeshBlock. Skip this step if joverlap = 0.
      if (joverlap != 0) {
        // Find ids of procs that data in [je-(joverlap-1):je] is sent to, and data in
        // [js:js+(joverlap-1)] is received from. Only execute if joverlap>0
        loc = my_loc;
        loc.lx2 += (Ngrids + 1);
        pleaf = proot->FindMeshBlock(loc);
        sendto_id = pleaf->GetGid();

        loc = my_loc;
        loc.lx2 -= (Ngrids + 1);
        pleaf = proot->FindMeshBlock(loc);
        getfrom_id = pleaf->GetGid();

        // Post a non-blocking receive for the data to be sent
        cnt = joverlap*nx3;
        ierr = MPI_Irecv(recv_buf.data(), cnt, MPI_DOUBLE, getfrom_id,
                         remapvar_tag, MPI_COMM_WORLD, &rq);

        // Pack send buffer with the density in [je-(joverlap-1):je]
        for (int k=ks; k<=ke; k++) {
          for (int j=je-(joverlap-1); j<=je; j++) {
            send_buf(k-ks,j-(je-(joverlap-1))) = roll_buf(k,j,i);
          }
        }

        // Send data
        ierr = MPI_Send(send_buf.data(), cnt, MPI_DOUBLE, sendto_id,
                      remapvar_tag, MPI_COMM_WORLD);

        // Wait for the data to be received
        ierr = MPI_Wait(&rq, MPI_STATUS_IGNORE);

        // Unpack the data sent from remote [je-(joverlap-1):je] to the local
        // [js:js+(joverlap-1)]
        for (int k=ks; k<=ke; k++) {
          for (int j=js; j<=js+(joverlap-1); j++) {
            out(k,j,i) = recv_buf(k-ks,j-js);
          }
        }
      }
      // Step A.2: Send density at [js:je-joverlap] to [js+joverlap:je] of
      // the target MeshBlock.
      if (Ngrids == 0) {
        // If the target MeshBlock is of my own, do local copy instead of MPI calls.
        for (int k=ks; k<=ke; k++) {
          for (int j=js+joverlap; j<=je; j++) {
            out(k,j,i) = roll_buf(k,j-joverlap,i);
          }
        }
      } else {
        // Find the id of the MeshBlock that data in [js:je-joverlap] is sent to, and data
        // in [js+joverlap:je] is received from.
        loc = my_loc;
        loc.lx2 += Ngrids;
        pleaf = proot->FindMeshBlock(loc);
        sendto_id = pleaf->GetGid();

        loc = my_loc;
        loc.lx2 -= Ngrids;
        pleaf = proot->FindMeshBlock(loc);
        getfrom_id = pleaf->GetGid();

        // Post a non-blocking receive for the data to be sent
        cnt = (nx2-joverlap)*nx3;
        ierr = MPI_Irecv(recv_buf.data(), cnt, MPI_DOUBLE, getfrom_id,
                         remapvar_tag, MPI_COMM_WORLD, &rq);

        // Pack send buffer with the density in [js:je-joverlap]
        for (int k=ks; k<=ke; k++) {
          for (int j=js; j<=je-joverlap; j++) {
            send_buf(k-ks,j-js) = roll_buf(k,j,i);
          }
        }

        // Send data
        ierr = MPI_Send(send_buf.data(), cnt, MPI_DOUBLE, sendto_id,
                      remapvar_tag, MPI_COMM_WORLD);

        // Wait for the data to be received
        ierr = MPI_Wait(&rq, MPI_STATUS_IGNORE);

        // Unpack the data sent from remote [js:je-joverlap] to the local
        // [js+joverlap:je]
        for (int k=ks; k<=ke; k++) {
          for (int j=js+joverlap; j<=je; j++) {
            out(k,j,i) = recv_buf(k-ks,j-(js+joverlap));
          }
        }
      }
    } else { // end of joffset >= 0
      // Case B: shear in negative-y direction.
      // Here, the steps are identical to the case A, but with different array indices.
      joffset = -joffset;
      Ngrids = static_cast<int>(joffset/nx2);
      joverlap = joffset - Ngrids*nx2;
      // Step B.1: Send density at [js:js+(joverlap-1)] to [je-(joverlap-1):je] of
      // the target MeshBlock. Skip this step if joverlap = 0.
      if (joverlap != 0) {
        // Find ids of procs that data in [js:js+(joverlap-1)] is sent to, and data in
        // [je-(joverlap-1):je] is received from. Only execute if joverlap>0
        loc = my_loc;
        loc.lx2 -= (Ngrids + 1);
        pleaf = proot->FindMeshBlock(loc);
        sendto_id = pleaf->GetGid();

        loc = my_loc;
        loc.lx2 += (Ngrids + 1);
        pleaf = proot->FindMeshBlock(loc);
        getfrom_id = pleaf->GetGid();

        // Post a non-blocking receive for the data to be sent
        cnt = joverlap*nx3;
        ierr = MPI_Irecv(recv_buf.data(), cnt, MPI_DOUBLE, getfrom_id,
                         remapvar_tag, MPI_COMM_WORLD, &rq);

        // Pack send buffer with the density in [je-(joverlap-1):je]
        for (int k=ks; k<=ke; k++) {
          for (int j=js; j<=js+(joverlap-1); j++) {
            send_buf(k-ks,j-js) = roll_buf(k,j,i);
          }
        }

        // Send data
        ierr = MPI_Send(send_buf.data(), cnt, MPI_DOUBLE, sendto_id,
                      remapvar_tag, MPI_COMM_WORLD);

        // Wait for the data to be received
        ierr = MPI_Wait(&rq, MPI_STATUS_IGNORE);

        // Unpack the data sent from remote [js:js+(joverlap-1)] to the local
        // [je-(joverlap-1):je]
        for (int k=ks; k<=ke; k++) {
          for (int j=je-(joverlap-1); j<=je; j++) {
            out(k,j,i) = recv_buf(k-ks,j-(je-(joverlap-1)));
          }
        }
      }
      // Step B.2: Send density at [js+joverlap:je] to [js:je-joverlap] of
      // the target MeshBlock.
      if (Ngrids == 0) {
        // If the target MeshBlock is of my own, do local copy instead of MPI calls.
        for (int k=ks; k<=ke; k++) {
          for (int j=js; j<=je-joverlap; j++) {
            out(k,j,i) = roll_buf(k,j+joverlap,i);
          }
        }
      } else {
        // Find the id of the MeshBlock that data in [js+joverlap:je] is sent to, and data
        // in [js:je-joverlap] is received from.
        loc = my_loc;
        loc.lx2 -= Ngrids;
        pleaf = proot->FindMeshBlock(loc);
        sendto_id = pleaf->GetGid();

        loc = my_loc;
        loc.lx2 += Ngrids;
        pleaf = proot->FindMeshBlock(loc);
        getfrom_id = pleaf->GetGid();

        // Post a non-blocking receive for the data to be sent
        cnt = (nx2-joverlap)*nx3;
        ierr = MPI_Irecv(recv_buf.data(), cnt, MPI_DOUBLE, getfrom_id,
                         remapvar_tag, MPI_COMM_WORLD, &rq);

        // Pack send buffer with the density in [js+joverlap:je]
        for (int k=ks; k<=ke; k++) {
          for (int j=js+joverlap; j<=je; j++) {
            send_buf(k-ks,j-(js+joverlap)) = roll_buf(k,j,i);
          }
        }

        // Send data
        ierr = MPI_Send(send_buf.data(), cnt, MPI_DOUBLE, sendto_id,
                      remapvar_tag, MPI_COMM_WORLD);

        // Wait for the data to be received
        ierr = MPI_Wait(&rq, MPI_STATUS_IGNORE);

        // Unpack the data sent from remote [js+joverlap:je] to the local
        // [js:je-joverlap]
        for (int k=ks; k<=ke; k++) {
          for (int j=js; j<=je-joverlap; j++) {
            out(k,j,i) = recv_buf(k-ks,j-js);
          }
        }
      }
    } // end of joffset < 0
  } // end of i loop
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn BlockFFTGravity::SetPhysicalBoundaries()
//! \brief Multiply Green's function
void BlockFFTGravity::SetPhysicalBoundaries() {
  //TODO(SMOON) use BoundaryFace rather than loc.lx3, etc, and write Boundaryfunctions
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
  } else if (gbflag==GravityBoundaryFlag::open) {
    if (pmy_block_->loc.lx3==0) {
      for (int k=1; k<=NGHOST; k++) {
        for (int j=js-NGHOST; j<=je+NGHOST; j++) {
          for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
            pmy_block_->pgrav->phi(ks-k,j,i) = pmy_block_->pgrav->phi(ks,j,i);
          }
        }
      }
    }
    if (pmy_block_->loc.lx3==Nx3/nx3-1) {
      for (int k=1; k<=NGHOST; k++) {
        for (int j=js-NGHOST; j<=je+NGHOST; j++) {
          for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
            pmy_block_->pgrav->phi(ke+k,j,i) = pmy_block_->pgrav->phi(ke,j,i);
          }
        }
      }
    }

    if (pmy_block_->loc.lx2==0) {
      for (int k=ks-NGHOST; k<=ke+NGHOST; k++) {
        for (int j=1; j<=NGHOST; j++) {
          for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
            pmy_block_->pgrav->phi(k,js-j,i) = pmy_block_->pgrav->phi(k,js,i);
          }
        }
      }
    }
    if (pmy_block_->loc.lx2==Nx2/nx2-1) {
      for (int k=ks-NGHOST; k<=ke+NGHOST; k++) {
        for (int j=1; j<=NGHOST; j++) {
          for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
            pmy_block_->pgrav->phi(k,je+j,i) = pmy_block_->pgrav->phi(k,je,i);
          }
        }
      }
    }
    if (pmy_block_->loc.lx1==0) {
      for (int k=ks-NGHOST; k<=ke+NGHOST; k++) {
        for (int j=js-NGHOST; j<=je+NGHOST; j++) {
          for (int i=1; i<=NGHOST; i++) {
            pmy_block_->pgrav->phi(k,j,is-i) = pmy_block_->pgrav->phi(k,j,is);
          }
        }
      }
    }
    if (pmy_block_->loc.lx1==Nx1/nx1-1) {
      for (int k=ks-NGHOST; k<=ke+NGHOST; k++) {
        for (int j=js-NGHOST; j<=je+NGHOST; j++) {
          for (int i=1; i<=NGHOST; i++) {
            pmy_block_->pgrav->phi(k,j,ie+i) = pmy_block_->pgrav->phi(k,j,ie);
          }
        }
      }
    }
  }
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

//----------------------------------------------------------------------------------------
//! \fn GetGreenFuncFlag(std::string input_string)
//! \brief Parses input string to return scoped enumerator flag specifying Green's
//! function. Typically called in BlockFFTGravity() ctor.

GreenFuncFlag GetGreenFuncFlag(const std::string& input_string) {
  if (input_string == "point_mass") {
    return GreenFuncFlag::point_mass;
  } else if (input_string == "cell_averaged") {
    return GreenFuncFlag::cell_averaged;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in GetGreenFuncFlag" << std::endl
        << "Input string=" << input_string << "\n"
        << "is an invalid Green's function type" << std::endl;
    ATHENA_ERROR(msg);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real _GetIGF(Real x, Real y, Real z)
//! \brief indefinite integral for the cell-averaged Green's function

Real _GetIGF(Real x, Real y, Real z) {
  Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
  return y*z*std::log(x+r) + z*x*std::log(y+r) + x*y*std::log(z+r)
         - 0.5*SQR(x)*std::atan(y*z/x/r)
         - 0.5*SQR(y)*std::atan(z*x/y/r)
         - 0.5*SQR(z)*std::atan(x*y/z/r);
}
