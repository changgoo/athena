//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
//======================================================================================
//! \file particle_buffer.cpp
//! \brief implements ParticleBuffer class for communication of particles.
//======================================================================================

// C++ standard libraries
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "particle_buffer.hpp"

//--------------------------------------------------------------------------------------
//! \fn ParticleBuffer::ParticleBuffer()
//! \brief initiates a default instance of ParticleBuffer.

ParticleBuffer::ParticleBuffer() {
  ibuf = NULL;
  rbuf = NULL;
  nparmax_ = npar_ = npar_gh_ = nint_ = nreal_ = 0;
#ifdef MPI_PARALLEL
  reqn = reqi = reqr = MPI_REQUEST_NULL;
  flagn = flagi = flagr = 0;
  tag = -1;
#endif
}

//--------------------------------------------------------------------------------------
//! \fn ParticleBuffer::ParticleBuffer(int nparmax0, int nint, int nreal)
//! \brief initiates a new instance of ParticleBuffer with nparmax = nparmax0.

ParticleBuffer::ParticleBuffer(int nparmax0, int nint, int nreal) {
  // Sanity check
  if (nparmax0 <= 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [ParticleBuffer::ParticleBuffer]" << std::endl
        << "Invalid nparmax0 = " << nparmax0 << std::endl;
    ATHENA_ERROR(msg);

    ibuf = NULL;
    rbuf = NULL;
    nparmax_ = npar_ = npar_gh_ = 0;
    return;
  }

  // Initialize the instance variables.
  nparmax_ = nparmax0;
  nint_ = nint;
  nreal_ = nreal;
  ibuf = new int[nint * nparmax_];
  rbuf = new Real[nreal * nparmax_];
  npar_ = npar_gh_ = 0;
#ifdef MPI_PARALLEL
  reqn = reqi = reqr = MPI_REQUEST_NULL;
  flagn = flagi = flagr = 0;
  tag = -1;
#endif
}

//--------------------------------------------------------------------------------------
//! \fn ParticleBuffer::ParticleBuffer()
//! \brief destroys an instance of ParticleBuffer.

ParticleBuffer::~ParticleBuffer() {
  if (ibuf != NULL) delete [] ibuf;
  if (rbuf != NULL) delete [] rbuf;
#ifdef MPI_PARALLEL
  if (reqn != MPI_REQUEST_NULL) MPI_Request_free(&reqn);
  if (reqi != MPI_REQUEST_NULL) MPI_Request_free(&reqi);
  if (reqr != MPI_REQUEST_NULL) MPI_Request_free(&reqr);
#endif
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleBuffer::Reallocate(int new_nparmax, int nint, int nreal)
//! \brief reallocates the buffers; the old content is preserved.

void ParticleBuffer::Reallocate(int new_nparmax, int nint, int nreal) {
  int npartot = npar_ + npar_gh_;
  // Sanity check
  if (new_nparmax <= 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [ParticleBuffer::Reallocate]" << std::endl
        << "Invalid new_nparmax = " << new_nparmax << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
  if (new_nparmax < npartot) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [ParticleBuffer::Reallocate]" << std::endl
        << "new_nparmax = " << new_nparmax << " < npar + nghost = " << npartot << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
#ifdef MPI_PARALLEL
  if (reqn != MPI_REQUEST_NULL || reqi != MPI_REQUEST_NULL || reqr != MPI_REQUEST_NULL) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [ParticleBuffer::Reallocate]" << std::endl
        << "MPI requests are active. " << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
#endif

  // Allocate new space.
  nint_ = nint;
  nreal_ = nreal;
  int *ibuf_new = new int[nint * new_nparmax];
  Real *rbuf_new = new Real[nreal * new_nparmax];

  // Move existing data.
  if ((npartot > 0)&&(nparmax_ > 0)) {
    std::memcpy(ibuf_new, ibuf, nint * npartot * sizeof(int));
    std::memcpy(rbuf_new, rbuf, nreal * npartot * sizeof(Real));
  }

  // Delete old space.
  if (ibuf != NULL) delete [] ibuf;
  if (rbuf != NULL) delete [] rbuf;
  ibuf = ibuf_new;
  rbuf = rbuf_new;
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleBuffer::Append(const ParticleBuffer& pbin)
//! \brief Append another ParticleBuffer to this

void ParticleBuffer::Append(const ParticleBuffer& pb) {
  if (pb.npar_ + pb.npar_gh_ == 0)
    // nothing to append
    return;
  if ((npar_gh_ > 0)&&(pb.npar_ > 0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [ParticleBuffer::Append]" << std::endl
        << "You are trying to append active particles on top of ghost particles; "
        << "This is prohibited." << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
  if (nparmax_ == 0) {
    // if this buffer is not allocated, allocate it using incoming buffer info
    Reallocate(1, pb.nint_, pb.nreal_);
  } else if ((nint_ != pb.nint_)||(nreal_ != pb.nreal_)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [ParticleBuffer::Append]" << std::endl
        << "Cannot append a ParticleBuffer with different nint or nreal." << std::endl
        << "This buffer: nparmax = " << nparmax_ << ", npar = " << npar_
            << ", nghost = " << npar_gh_ << ", nint = " << nint_
            << ", nreal = "<< nreal_ << std::endl
        << "buffer to be appended: nparmax = " << pb.nparmax_ << ", npar = " << pb.npar_
            << ", nghost = " << pb.npar_gh_ << ", nint = " << pb.nint_
            << ", nreal = " << pb.nreal_ << std::endl;
    ATHENA_ERROR(msg);
    return;
  }

  // check size
  int new_npartot = npar_ + npar_gh_ + pb.npar_ + pb.npar_gh_;
  if (new_npartot > nparmax_)
    Reallocate(new_npartot, nint_, nreal_);

  int offset, cnt;
  // append int buffer
  offset = nint_*(npar_ + npar_gh_);
  cnt = pb.nint_*(pb.npar_ + pb.npar_gh_)*sizeof(int);
  std::memcpy(ibuf + offset, pb.ibuf, cnt);
  // append real buffer
  offset = nreal_*(npar_ + npar_gh_);
  cnt = pb.nreal_*(pb.npar_ + pb.npar_gh_)*sizeof(Real);
  std::memcpy(rbuf + offset, pb.rbuf, cnt);
  // update number of particles
  npar_ += pb.npar_;
  npar_gh_ += pb.npar_gh_;
}
