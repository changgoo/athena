//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_pm.cpp
//! \brief implements boundary functions for Particle Mesh variables
//!   and utilities to add particle densities rather than replacce

// C headers

// C++ headers

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../../orbital_advection/orbital_advection.hpp"
#include "../../../utils/buffer_utils.hpp"
#include "bvals_pm.hpp"

//----------------------------------------------------------------------------------------
//! \fn ParticleMeshBoundaryVariable::ParticleMeshBoundaryVariable
//! \brief

ParticleMeshBoundaryVariable::ParticleMeshBoundaryVariable(
    MeshBlock *pmb, AthenaArray<Real> *var, AthenaArray<Real> *coarse_var) :
    empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
    CellCenteredBoundaryVariable(pmb, var, coarse_var, empty_flux) {
  var_buf.NewAthenaArray(var->GetDim4(),pmb->ncells3,pmb->ncells2,pmb->ncells1);
}

//----------------------------------------------------------------------------------------
//! \fn int ParticleMeshBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
//!                                                             const NeighborBlock& nb)
//! \brief Set particle_mesh boundary buffers for sending to a block on the same level

int ParticleMeshBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
                                                              const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;

  // load ghost zones
  if (nb.ni.ox1 == 0)     si = pmb->is,        ei = pmb->ie;
  else if (nb.ni.ox1 > 0) si = pmb->ie + 1,      ei = pmb->ie + NGHOST;
  else              si = pmb->is - NGHOST, ei = pmb->is - 1;
  if (nb.ni.ox2 == 0)     sj = pmb->js,        ej = pmb->je;
  else if (nb.ni.ox2 > 0) sj = pmb->je + 1,      ej = pmb->je + NGHOST;
  else              sj = pmb->js - NGHOST, ej = pmb->js - 1;
  if (nb.ni.ox3 == 0)     sk = pmb->ks,        ek = pmb->ke;
  else if (nb.ni.ox3 > 0) sk = pmb->ke + 1,      ek = pmb->ke + NGHOST;
  else              sk = pmb->ks - NGHOST, ek = pmb->ks - 1;

  int p = 0;
  AthenaArray<Real> &var = *var_cc;
  BufferUtility::PackData(var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  return p;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleMeshBoundaryVariable::SetBoundarySameLevel(Real *buf,
//!                                                      const NeighborBlock& nb)
//! \brief Unpack PM boundary received from a block on the same level
//!
//! Unpack received to var_buf
//! Add it to var_cc if non-shear-periodic
//! or pass it to SendShearingBoxBoundaryBuffers for shift

void ParticleMeshBoundaryVariable::SetBoundarySameLevel(Real *buf,
                                                 const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;

  if (pbval_->shearing_box == 0) {
    // if non-shear-periodic,
    // unpack and add directly onto active zones
    si = (nb.ni.ox1 > 0) ? (pmb->ie - NGHOST + 1) : pmb->is;
    ei = (nb.ni.ox1 < 0) ? (pmb->is + NGHOST - 1) : pmb->ie;
    sj = (nb.ni.ox2 > 0) ? (pmb->je - NGHOST + 1) : pmb->js;
    ej = (nb.ni.ox2 < 0) ? (pmb->js + NGHOST - 1) : pmb->je;
    sk = (nb.ni.ox3 > 0) ? (pmb->ke - NGHOST + 1) : pmb->ks;
    ek = (nb.ni.ox3 < 0) ? (pmb->ks + NGHOST - 1) : pmb->ke;

    int p = 0;

    AthenaArray<Real> &var = *var_cc;
    for (int n=nl_; n<=nu_; ++n) {
      for (int k=sk; k<=ek; ++k) {
        for (int j=sj; j<=ej; ++j) {
#pragma omp simd
          for (int i=si; i<=ei; ++i) {
            var(n,k,j,i) += buf[p++];
          }
        }
      }
    }
  } else {
    // if shear-periodic,
    // unpack onto ghost zones of buffer
    if (nb.ni.ox1 == 0)     si = pmb->is,        ei = pmb->ie;
    else if (nb.ni.ox1 > 0) si = pmb->ie + 1,      ei = pmb->ie + NGHOST;
    else              si = pmb->is - NGHOST, ei = pmb->is - 1;
    if (nb.ni.ox2 == 0)     sj = pmb->js,        ej = pmb->je;
    else if (nb.ni.ox2 > 0) sj = pmb->je + 1,      ej = pmb->je + NGHOST;
    else              sj = pmb->js - NGHOST, ej = pmb->js - 1;
    if (nb.ni.ox3 == 0)     sk = pmb->ks,        ek = pmb->ke;
    else if (nb.ni.ox3 > 0) sk = pmb->ke + 1,      ek = pmb->ke + NGHOST;
    else              sk = pmb->ks - NGHOST, ek = pmb->ks - 1;

    int p = 0;
    // now particle mesh only assumes Cartesian
    BufferUtility::UnpackData(buf, var_buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleMeshBoundaryVariable::SendShearingBoxBoundaryBuffers()
//! \brief Send PM shearing box boundary buffers received

void ParticleMeshBoundaryVariable::SendShearingBoxBoundaryBuffers() {
  MeshBlock *pmb = pmy_block_;
  Mesh *pmesh = pmb->pmy_mesh;
  int ssize = nu_ + 1;
  int offset[2]{0, 4};
  for (int upper=0; upper<2; upper++) {
    if (pbval_->is_shear[upper]) {
      for (int n=0; n<4; n++) {
        SimpleNeighborBlock& snb = pbval_->sb_data_[upper].send_neighbor[n];
        if (snb.rank != -1) {
          LoadShearingBoxBoundarySameLevel(var_buf, shear_bd_var_[upper].send[n],
                                       n+offset[upper]);
          if (snb.rank == Globals::my_rank) {// on the same process
            CopyShearBufferSameProcess(snb, shear_send_count_cc_[upper][n]*ssize, n,
                                       upper);
          } else { // MPI
#ifdef MPI_PARALLEL
            int tag = pbval_->CreateBvalsMPITag(snb.lid, n+offset[upper],
                                                shear_cc_phys_id_);
            MPI_Isend(shear_bd_var_[upper].send[n], shear_send_count_cc_[upper][n]*ssize,
                      MPI_ATHENA_REAL, snb.rank, tag, MPI_COMM_WORLD,
                      &shear_bd_var_[upper].req_send[n]);
#endif
          }
        }
      }  // loop over recv[0] to recv[3]
    }  // if boundary is shearing
  }  // loop over inner/outer boundaries
  return;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParticleMeshBoundaryVariable::SetShearingBoxBoundaryBuffers()
//! \brief Add PM density received

void ParticleMeshBoundaryVariable::SetShearingBoxBoundaryBuffers() {
  MeshBlock *pmb = pmy_block_;
  Mesh *pmesh = pmb->pmy_mesh;
  OrbitalAdvection *porb = pmb->porb;
  AthenaArray<Real> &var = *var_cc;
  AthenaArray<Real> &pflux = pbval_->pflux_;
  int &xgh = pbval_->xgh_;
  int &xorder = pbval_->xorder_;
  int nb_offset[2]{0, 4};
  int ib[2]{pmb->is - NGHOST, pmb->ie + 1}; // for buffer
  int ia[2]{pmb->is, pmb->ie - NGHOST + 1}; // for active
  int js = pmb->js, je = pmb->je;
  int kl = pmb->ks, ku = pmb->ke;
  if (pmesh->mesh_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }
  int jl = js-NGHOST;
  int ju = je+NGHOST;

  for (int upper=0; upper<2; upper++) {
    if (pbval_->is_shear[upper]) { // check inner boundaries
      // step 1 -- (optionally) apply shear to shear_cc_ (does nothing by default)
      if (!porb->orbital_advection_defined)
        ShearQuantities(shear_cc_[upper], upper);  // Hydro overrides this
      // step 2. -- calculating remapping flux and update var
      Real eps = (1.0-2*upper)*pbval_->eps_;
      for (int n=nl_; n<=nu_; n++) {
        pbuf.InitWithShallowSlice(shear_cc_[upper], 4, n, 1);
        for (int k=kl; k<=ku; k++) {
          for (int i=0; i<NGHOST; i++) {
            int ii = ib[upper]+i;
            if (xorder<=2) {
              porb->RemapFluxPlm(pflux, pbuf, eps, 1-upper, k, i, jl, ju+1, xgh);
            } else {
              porb->RemapFluxPpm(pflux, pbuf, eps, 1-upper, k, i, jl, ju+1, xgh);
            }
            const int shift = xgh+1-upper;
            // set corresponding active zone indices where the shifted buffer to be added
            ii = ia[upper]+1;
            for (int j=jl; j<=ju; j++) {
              var(n,k,j,ii) += pbuf(k,i,j+shift) - (pflux(j+1) - pflux(j));
            }
          }
        }
      }
    }
  }
  return;
}