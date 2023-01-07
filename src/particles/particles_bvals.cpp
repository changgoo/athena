//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file particles_bvals.cpp
//! \brief implements functions for particle communications

// C++ Standard Libraries
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "particles.hpp"

// Local function prototypes
static int CheckSide(int xi, int xi1, int xi2);

//--------------------------------------------------------------------------------------
//! \fn void Particles::AMRCoarseToFine(MeshBlock* pmbc, MeshBlock* pmbf)
//! \brief load particles from a coarse meshblock to a fine meshblock.

void Particles::AMRCoarseToFine(Particles *pparc, Particles *pparf, MeshBlock* pmbf) {
  // Initialization
  const Real x1min = pmbf->block_size.x1min, x1max = pmbf->block_size.x1max;
  const Real x2min = pmbf->block_size.x2min, x2max = pmbf->block_size.x2max;
  const Real x3min = pmbf->block_size.x3min, x3max = pmbf->block_size.x3max;
  const bool active1 = pparc->active1_,
             active2 = pparc->active2_,
             active3 = pparc->active3_;
  const AthenaArray<Real> &xp = pparc->xp_, &yp = pparc->yp_, &zp = pparc->zp_;
  const Coordinates *pcoord = pmbf->pcoord;

  // Loop over particles in the coarse meshblock.
  for (int k = 0; k < pparc->npar_; ++k) {
    Real x1, x2, x3;
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if ((!active1 || (active1 && x1min <= x1 && x1 < x1max)) &&
        (!active2 || (active2 && x2min <= x2 && x2 < x2max)) &&
        (!active3 || (active3 && x3min <= x3 && x3 < x3max))) {
      // Load a particle to the fine meshblock.
      int npar = pparf->npar_;
      if (npar >= pparf->nparmax_) pparf->UpdateCapacity(2 * pparf->nparmax_);
      for (int j = 0; j < pparf->nint; ++j)
        pparf->intprop(j,npar) = pparc->intprop(j,k);
      for (int j = 0; j < pparf->nreal; ++j)
        pparf->realprop(j,npar) = pparc->realprop(j,k);
      for (int j = 0; j < pparf->naux; ++j)
        pparf->auxprop(j,npar) = pparc->auxprop(j,k);
      ++pparf->npar_;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::AMRFineToCoarse(MeshBlock* pmbf, MeshBlock* pmbc)
//! \brief load particles from a fine meshblock to a coarse meshblock.

void Particles::AMRFineToCoarse(Particles *pparc, Particles *pparf) {
  // Check the capacity.
  int nparf = pparf->npar_, nparc = pparc->npar_;
  int npar_new = nparf + nparc;
  if (npar_new > pparc->nparmax_) pparc->UpdateCapacity(npar_new);

  // Load the particles.
  for (int j = 0; j < pparf->nint; ++j)
    for (int k = 0; k < nparf; ++k)
      pparc->intprop(j,nparc+k) = pparf->intprop(j,k);
  for (int j = 0; j < pparf->nreal; ++j)
    for (int k = 0; k < nparf; ++k)
      pparc->realprop(j,nparc+k) = pparf->realprop(j,k);
  for (int j = 0; j < pparf->naux; ++j)
    for (int k = 0; k < nparf; ++k)
      pparc->auxprop(j,nparc+k) = pparf->auxprop(j,k);
  pparc->npar_ = npar_new;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ClearBoundary()
//! \brief resets boundary for particle transportation.

void Particles::ClearBoundary() {
  for (int i = 0; i < pbval_->nneighbor; ++i) {
    NeighborBlock& nb = pbval_->neighbor[i];
    bstatus_[nb.bufid] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
    if (nb.snb.rank != Globals::my_rank) {
      ParticleBuffer& recv = recv_[nb.bufid];
      recv.flagn = recv.flagi = recv.flagr = 0;
      send_[nb.bufid].npar_ = 0;
    }
#endif
  }

  // clear boundary information for shear
  ClearBoundaryShear();
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ClearNeighbors()
//! \brief clears links to neighbors.

void Particles::ClearNeighbors() {
  delete neighbor_[1][1][1].pnb;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        Neighbor *pn = &neighbor_[i][j][k];
        if (pn == NULL) continue;
        while (pn->next != NULL)
          pn = pn->next;
        while (pn->prev != NULL) {
          pn = pn->prev;
          delete pn->next;
          pn->next = NULL;
        }
        pn->pnb = NULL;
        pn->pmb = NULL;
      }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::LinkNeighbors(MeshBlockTree &tree,
//!         int64_t nrbx1, int64_t nrbx2, int64_t nrbx3, int root_level)
//! \brief fetches neighbor information for later communication.

void Particles::LinkNeighbors(MeshBlockTree &tree,
    int64_t nrbx1, int64_t nrbx2, int64_t nrbx3, int root_level) {
  // Set myself as one of the neighbors.
  Neighbor *pn = &neighbor_[1][1][1];
  pn->pmb = pmy_block;
  pn->pnb = new NeighborBlock;
  pn->pnb->SetNeighbor(Globals::my_rank, pmy_block->loc.level,
      pmy_block->gid, pmy_block->lid, 0, 0, 0, NeighborConnect::none,
      -1, -1, false, false, 0, 0);

  // Save pointer to each neighbor.
  for (int i = 0; i < pbval_->nneighbor; ++i) {
    NeighborBlock& nb = pbval_->neighbor[i];
    SimpleNeighborBlock& snb = nb.snb;
    NeighborIndexes& ni = nb.ni;
    Neighbor *pn = &neighbor_[ni.ox1+1][ni.ox2+1][ni.ox3+1];
    while (pn->next != NULL)
      pn = pn->next;
    if (pn->pnb != NULL) {
      pn->next = new Neighbor;
      pn->next->prev = pn;
      pn = pn->next;
    }
    pn->pnb = &nb;
    if (snb.rank == Globals::my_rank) {
      pn->pmb = pmy_mesh_->FindMeshBlock(snb.gid);
    } else {
#ifdef MPI_PARALLEL
      // assign unique tag
      // tag = local id of destination (remaining bits) + bufid (6 bits)
      // + particle container id (3 bits) + npar,intprop,realprop(2 bits)
      send_[nb.bufid].tag = (snb.lid<<11) | (nb.targetid<<5) | (ipar << 2);
      recv_[nb.bufid].tag = (pmy_block->lid<<11) | (nb.bufid<<5) | (ipar << 2);
#endif
    }
  }

  // Collect missing directions from fine to coarse level.
  if (pmy_mesh_->multilevel) {
    int my_level = pbval_->loc.level;
    for (int l = 0; l < 3; l++) {
      if (!active1_ && l != 1) continue;
      for (int m = 0; m < 3; m++) {
        if (!active2_ && m != 1) continue;
        for (int n = 0; n < 3; n++) {
          if (!active3_ && n != 1) continue;
          Neighbor *pn = &neighbor_[l][m][n];
          if (pn->pnb == NULL) {
            int nblevel = pbval_->nblevel[n][m][l];
            if (0 <= nblevel && nblevel < my_level) {
              int ngid = tree.FindNeighbor(pbval_->loc, l-1, m-1, n-1)->GetGid();
              for (int i = 0; i < pbval_->nneighbor; ++i) {
                NeighborBlock& nb = pbval_->neighbor[i];
                if (nb.snb.gid == ngid) {
                  pn->pnb = &nb;
                  if (nb.snb.rank == Globals::my_rank)
                    pn->pmb = pmy_mesh_->FindMeshBlock(ngid);
                  break;
                }
              }
            }
          }
        }
      }
    }
  }

  // Initiate boundary values.
  ClearBoundary();
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SendToNeighbors()
//! \brief sends particles outside boundary to the buffers of neighboring meshblocks.

void Particles::SendToNeighbors() {
  const int IS = pmy_block->is;
  const int IE = pmy_block->ie;
  const int JS = pmy_block->js;
  const int JE = pmy_block->je;
  const int KS = pmy_block->ks;
  const int KE = pmy_block->ke;

  for (int k = 0; k < npar_; ) {
    // Check if a particle is outside the boundary.
    // (changgoo) not sure why indices have used instead of position
    // but must be equivalent
    // since UpdatePositionIndices are called just before this function call
    int xi1i = static_cast<int>(xi1_(k)),
        xi2i = static_cast<int>(xi2_(k)),
        xi3i = static_cast<int>(xi3_(k));
    int ox1 = CheckSide(xi1i, IS, IE),
        ox2 = CheckSide(xi2i, JS, JE),
        ox3 = CheckSide(xi3i, KS, KE);

    // initialize a flag for shear boundary crossing
    if (pmy_mesh_->shear_periodic) auxprop(ish,k) = 0;

    if (ox1 == 0 && ox2 == 0 && ox3 == 0) {
      ++k;
      continue;
    }

    // Apply boundary conditions and find the mesh coordinates.
    // After this function call, position/velocity vectors are updated
    // but the indices remain the same to use in FindTargetNeighbor
    Real x1, x2, x3;
    ApplyBoundaryConditions(k, x1, x2, x3);

    // Find the neighbor block to send it to.
    if (!active1_) ox1 = 0;
    if (!active2_) ox2 = 0;
    if (!active3_) ox3 = 0;
    Neighbor *pn = FindTargetNeighbor(ox1, ox2, ox3, xi1i, xi2i, xi3i);
    NeighborBlock *pnb = pn->pnb;
    if (pnb == NULL) {
      RemoveOneParticle(k);
      continue;
    }

    // Determine which particle buffer to use.
    ParticleBuffer *ppb = NULL;
    if (pnb->snb.rank == Globals::my_rank) {
      // No need to send if back to the same block.
      if (pnb->snb.gid == pmy_block->gid) {
        // Update particle indices for particles that are in the same block
        pmy_block->pcoord->MeshCoordsToIndices(x1, x2, x3, xi1_(k), xi2_(k), xi3_(k));
        ++k;
        continue;
      }
      // Use the target receive buffer.
      ppb = &pn->pmb->ppars[ipar]->recv_[pnb->targetid];
    } else {
#ifdef MPI_PARALLEL
      // Use the send buffer.
      ppb = &send_[pnb->bufid];
#endif
    }

    // Load the particle to particle buffer
    LoadParticleBuffer(ppb, k);

    // Pop the particle from the current MeshBlock.
    RemoveOneParticle(k);
  }

  // Send to neighbor processes and update boundary status.
  for (int i = 0; i < pbval_->nneighbor; ++i) {
    NeighborBlock& nb = pbval_->neighbor[i];
    int dst = nb.snb.rank;
    if (dst == Globals::my_rank) {
      Particles *ppar = pmy_mesh_->FindMeshBlock(nb.snb.gid)->ppars[ipar];
      ppar->bstatus_[nb.targetid] =
          (ppar->recv_[nb.targetid].npar_ > 0) ? BoundaryStatus::arrived
                                              : BoundaryStatus::completed;
    } else {
#ifdef MPI_PARALLEL
      ParticleBuffer& send = send_[nb.bufid];
      SendParticleBuffer(send, nb.snb.rank);
#endif
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SendParticleBuffer()
//! \brief send particle buffer

#ifdef MPI_PARALLEL
void Particles::SendParticleBuffer(ParticleBuffer& send, int dst) {
  int npsend = send.npar_;
  int sendtag = send.tag;
  MPI_Send(&npsend, 1, MPI_INT, dst, sendtag, my_comm);
  if (npsend > 0) {
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Isend(send.ibuf, npsend * nint_buf, MPI_INT,
              dst, sendtag + 1, my_comm, &req);
    MPI_Request_free(&req);
    MPI_Isend(send.rbuf, npsend * nreal_buf, MPI_ATHENA_REAL,
              dst, sendtag + 2, my_comm, &req);
    MPI_Request_free(&req);
  }
}
#endif


//--------------------------------------------------------------------------------------
//! \fn void Particles::ReceiveParticleBuffer()
//! \brief recv particle buffer

#ifdef MPI_PARALLEL
void Particles::ReceiveParticleBuffer(int nb_rank, ParticleBuffer& recv,
                                      enum BoundaryStatus& bstatus) {
  // Communicate with neighbor processes.
  if (nb_rank != Globals::my_rank && bstatus == BoundaryStatus::waiting) {
    if (!recv.flagn) {
      // Get the number of incoming particles.
      if (recv.reqn == MPI_REQUEST_NULL)
        MPI_Irecv(&recv.npar_, 1, MPI_INT, nb_rank, recv.tag, my_comm, &recv.reqn);
      else
        MPI_Test(&recv.reqn, &recv.flagn, MPI_STATUS_IGNORE);
      if (recv.flagn) {
        if (recv.npar_ > 0) {
          // Check the buffer size.
          int nprecv = recv.npar_;
          if (nprecv > recv.nparmax_) {
            recv.npar_ = 0;
            recv.Reallocate(2 * nprecv - recv.nparmax_, nint_buf, nreal_buf);
            recv.npar_ = nprecv;
          }
        } else {
          // No incoming particles.
          bstatus = BoundaryStatus::completed;
        }
      }
    }
    if (recv.flagn && recv.npar_ > 0) {
      // Receive data from the neighbor.
      if (!recv.flagi) {
        if (recv.reqi == MPI_REQUEST_NULL)
          MPI_Irecv(recv.ibuf, recv.npar_ * nint_buf, MPI_INT,
                    nb_rank, recv.tag + 1, my_comm, &recv.reqi);
        else
          MPI_Test(&recv.reqi, &recv.flagi, MPI_STATUS_IGNORE);
      }
      if (!recv.flagr) {
        if (recv.reqr == MPI_REQUEST_NULL)
          MPI_Irecv(recv.rbuf, recv.npar_ * nreal_buf, MPI_ATHENA_REAL,
                    nb_rank, recv.tag + 2, my_comm, &recv.reqr);
        else
          MPI_Test(&recv.reqr, &recv.flagr, MPI_STATUS_IGNORE);
      }
      if (recv.flagi && recv.flagr)
        bstatus = BoundaryStatus::arrived;
    }
  }
}
#endif

//--------------------------------------------------------------------------------------
//! \fn bool Particles::ReceiveFromNeighbors()
//! \brief receives particles from neighboring meshblocks and returns a flag indicating
//!        if all receives are completed.

bool Particles::ReceiveFromNeighbors() {
  bool flag = true;

  for (int i = 0; i < pbval_->nneighbor; ++i) {
    NeighborBlock& nb = pbval_->neighbor[i];
    enum BoundaryStatus& bstatus = bstatus_[nb.bufid];
    ParticleBuffer& recv = recv_[nb.bufid];
#ifdef MPI_PARALLEL
    ReceiveParticleBuffer(nb.snb.rank, recv, bstatus);
#endif
    switch (bstatus) {
      case BoundaryStatus::completed:
        break;

      case BoundaryStatus::waiting:
        flag = false;
        break;

      case BoundaryStatus::arrived:
        FlushReceiveBuffer(recv);
        bstatus = BoundaryStatus::completed;
        break;
    }
  }

  return flag;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ApplyBoundaryConditions(int k, Real &x1, Real &x2, Real &x3)
//! \brief applies boundary conditions to particle k and returns its updated mesh
//!        coordinates (x1,x2,x3).
//! \todo (ccyang):
//! - implement nonperiodic boundary conditions.

void Particles::ApplyBoundaryConditions(int k, Real &x1, Real &x2, Real &x3) {
  bool flag = false;
  RegionSize& mesh_size = pmy_mesh_->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;

  // Find the mesh coordinates.
  Real x10, x20, x30;
  // these two must be equilvalent
  // pcoord->IndicesToMeshCoords(xi1_(k), xi2_(k), xi3_(k), x1, x2, x3);
  pcoord->CartesianToMeshCoords(xp_(k), yp_(k), zp_(k), x1, x2, x3);
  pcoord->CartesianToMeshCoords(xp0_(k), yp0_(k), zp0_(k), x10, x20, x30);

  // Convert velocity vectors in mesh coordinates.
  Real vp1, vp2, vp3, vp10, vp20, vp30;
  pcoord->CartesianToMeshCoordsVector(xp_(k), yp_(k), zp_(k),
                                      vpx_(k), vpy_(k), vpz_(k), vp1, vp2, vp3);
  pcoord->CartesianToMeshCoordsVector(xp0_(k), yp0_(k), zp0_(k),
                                      vpx0_(k), vpy0_(k), vpz0_(k), vp10, vp20, vp30);

  // Apply periodic boundary conditions in X1.
  if (x1 < mesh_size.x1min) {
    // Inner x1
    x1 += mesh_size.x1len;
    x10 += mesh_size.x1len;
    // the particle has crossed shear boundary to the left
    if (pmy_mesh_->shear_periodic) auxprop(ish,k) = -1;
    flag = true;
  } else if (x1 >= mesh_size.x1max) {
    // Outer x1
    x1 -= mesh_size.x1len;
    x10 -= mesh_size.x1len;
    // the particle has crossed shear boundary to the right
    if (pmy_mesh_->shear_periodic) auxprop(ish,k) = 1;
    flag = true;
  }

  // Apply periodic boundary conditions in X2.
  if (x2 < mesh_size.x2min) {
    // Inner x2
    x2 += mesh_size.x2len;
    x20 += mesh_size.x2len;
    flag = true;
  } else if (x2 >= mesh_size.x2max) {
    // Outer x2
    x2 -= mesh_size.x2len;
    x20 -= mesh_size.x2len;
    flag = true;
  }

  // Apply periodic boundary conditions in X3.
  if (x3 < mesh_size.x3min) {
    // Inner x3
    x3 += mesh_size.x3len;
    x30 += mesh_size.x3len;
    flag = true;
  } else if (x3 >= mesh_size.x3max) {
    // Outer x3
    x3 -= mesh_size.x3len;
    x30 -= mesh_size.x3len;
    flag = true;
  }

  if (flag) {
    // Convert positions and velocities back in Cartesian coordinates.
    pcoord->MeshCoordsToCartesian(x1, x2, x3, xp_(k), yp_(k), zp_(k));
    pcoord->MeshCoordsToCartesian(x10, x20, x30, xp0_(k), yp0_(k), zp0_(k));
    pcoord->MeshCoordsToCartesianVector(x1, x2, x3,
                                        vp1, vp2, vp3, vpx_(k), vpy_(k), vpz_(k));
    pcoord->MeshCoordsToCartesianVector(x10, x20, x30,
                                        vp10, vp20, vp30, vpx0_(k), vpy0_(k), vpz0_(k));
  }
}

//--------------------------------------------------------------------------------------
//! \fn MeshBlock* Particles::FindTargetNeighbor(
//!         int ox1, int ox2, int ox3, int xi1, int xi2, int xi3)
//! \brief finds the neighbor to send a particle to.

struct Neighbor* Particles::FindTargetNeighbor(
    int ox1, int ox2, int ox3, int xi1, int xi2, int xi3) {
  // Find the head of the linked list.
  Neighbor *pn = &neighbor_[ox1+1][ox2+1][ox3+1];

  // Search down the list if the neighbor is at a finer level.
  if (pmy_mesh_->multilevel && pn->pnb != NULL &&
      pn->pnb->snb.level > pmy_block->loc.level) {
    RegionSize& bs = pmy_block->block_size;
    int fi[2] = {0, 0}, i = 0;
    if (active1_ && ox1 == 0) fi[i++] = 2 * (xi1 - pmy_block->is) / bs.nx1;
    if (active2_ && ox2 == 0) fi[i++] = 2 * (xi2 - pmy_block->js) / bs.nx2;
    if (active3_ && ox3 == 0) fi[i++] = 2 * (xi3 - pmy_block->ks) / bs.nx3;
    while (pn != NULL) {
      NeighborIndexes& ni = pn->pnb->ni;
      if (ni.fi1 == fi[0] && ni.fi2 == fi[1]) break;
      pn = pn->next;
    }
  }

  // Return the target neighbor.
  return pn;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::FlushReceiveBuffer(ParticleBuffer& recv)
//! \brief adds particles from the receive buffer.

void Particles::FlushReceiveBuffer(ParticleBuffer& recv) {
  // Check the memory size.
  int nprecv = recv.npar_;
  if (npar_ + nprecv > nparmax_)
    UpdateCapacity(nparmax_ + 2 * (npar_ + nprecv - nparmax_));

  // Flush the receive buffers.
  int *pi = recv.ibuf;
  Real *pr = recv.rbuf;
  for (int k = npar_; k < npar_ + nprecv; ++k) {
    for (int j = 0; j < nint; ++j)
      intprop(j,k) = *pi++;
    for (int j = 0; j < nreal; ++j)
      realprop(j,k) = *pr++;
    for (int j = 0; j < naux; ++j)
      auxprop(j,k) = *pr++;
  }

  // Find their position indices.
  AthenaArray<Real> xps, yps, zps, xi1s, xi2s, xi3s;
  xps.InitWithShallowSlice(xp_, 1, npar_, nprecv);
  yps.InitWithShallowSlice(yp_, 1, npar_, nprecv);
  zps.InitWithShallowSlice(zp_, 1, npar_, nprecv);
  xi1s.InitWithShallowSlice(xi1_, 1, npar_, nprecv);
  xi2s.InitWithShallowSlice(xi2_, 1, npar_, nprecv);
  xi3s.InitWithShallowSlice(xi3_, 1, npar_, nprecv);
  UpdatePositionIndices(nprecv, xps, yps, zps, xi1s, xi2s, xi3s);

  // Clear the receive buffers.
  npar_ += nprecv;
  recv.npar_ = 0;
}


//--------------------------------------------------------------------------------------
//! \fn void Particles::LoadParticleBuffer(ParticleBuffer& recv)
//! \brief load the k-th particle to the particle buffer.

void Particles::LoadParticleBuffer(ParticleBuffer *ppb, int k) {
  // Check the buffer size.
  if (ppb->npar_ >= ppb->nparmax_)
    ppb->Reallocate((ppb->nparmax_ > 0) ? 2 * ppb->nparmax_ : 1, nint_buf, nreal_buf);

  // Copy the properties of the particle to the buffer.
  int *pi = ppb->ibuf + nint_buf * ppb->npar_;
  for (int j = 0; j < nint; ++j)
    *pi++ = intprop(j,k);
  Real *pr = ppb->rbuf + nreal_buf * ppb->npar_;
  for (int j = 0; j < nreal; ++j)
    *pr++ = realprop(j,k);
  for (int j = 0; j < naux; ++j)
    *pr++ = auxprop(j,k);
  ++ppb->npar_;
}

//--------------------------------------------------------------------------------------
//! \fn LogicalLocation Particles::FindTargetGidAlongX2(Real x2)
//! \brief searching meshblock along the x2 direction due to shear
//!        (uniform mesh is assumed)

int Particles::FindTargetGidAlongX2(Real x2) {
  LogicalLocation target_loc(pmy_block->loc);
  RegionSize msize(pmy_mesh_->mesh_size), bsize(pmy_block->block_size);
  Real x2min = msize.x2min;
  MeshBlockTree *proot = &(pmy_mesh_->tree), *pleaf=nullptr;

  for (int lx2=0; lx2<pmy_mesh_->nrbx2; ++lx2) {
    // calculate target block min/max assuming uniform mesh
    Real bmin = x2min+lx2*bsize.x2len, bmax = bmin+bsize.x2len;
    if ((bmin < x2) && (x2 <= bmax)) {
      target_loc.lx2 = lx2;
      pleaf = proot->FindMeshBlock(target_loc);
      return pleaf->GetGid();
    }
  }

  // failed to find corresponding meshblock
  return -1;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SendParticlesShear()
//! \brief send particles outside meshblock due to shear

void Particles::SendParticlesShear() {
  for (int k = 0; k < npar_; ) {
    // Apply shear offset and find the mesh coordinates.
    Real x1, x2, x3;
    if (auxprop(ish,k) == 0) {
      // the particle hasn't crossed shear boundary
      ++k;
      continue;
    } else {
      ApplyBoundaryConditionsShear(k, x1, x2, x3);
    }

    // we know which side to check
    int upper = (auxprop(ish,k) == -1) ? 1 : 0;
    if (!pbval_->is_shear[upper]) {
      // shouldn't reach here
      std::stringstream msg;
      if (upper == 0) {
        msg << "### FATAL ERROR in function [Particles::SendParticleShear]"
            << " Particle " << k << " has crossed outer shear-periodic boundary"
            << " but the current meshbock's inner boundary is not shearing-periodic."
            << " This doesn't make sens." << std::endl;
      } else {
        msg << "### FATAL ERROR in function [Particles::SendParticleShear]"
            << " Particle " << k << " has crossed inner shear-periodic boundary"
            << " but the current meshbock's outer boundary is not shearing-periodic."
            << " This doesn't make sens." << std::endl;
      }
      ATHENA_ERROR(msg);
    }

    // given this particle's new position, what is the correct meshblock id?
    int target_id = FindTargetGidAlongX2(x2);
    // find meshblock to send
    // only look-up shearing-box's send_neighbor list
    SimpleNeighborBlock snb;
    int shid = -1;
    for (int n=0; n<4; n++) {
      snb = pbval_->sb_data_[upper].send_neighbor[n];
      if (snb.gid == target_id) {
        shid = n+upper*4;
        break;
      }
    }

    // Determine which particle buffer to use.
    ParticleBuffer *ppb = NULL;
    if (snb.rank == Globals::my_rank) {
      // the target block is in the same processor
      if (snb.gid == pmy_block->gid) {
        // if the target block is me
        // Update particle indices and done.
        Coordinates *pc = pmy_block->pcoord;
        pc->MeshCoordsToIndices(x1, x2, x3, xi1_(k), xi2_(k), xi3_(k));
        ++k;
        continue;
      }
      MeshBlock *ptarget_block = pmy_mesh_->FindMeshBlock(snb.gid);
      // Use the target receive buffer.
      ppb = &ptarget_block->ppars[ipar]->recv_sh_[shid];
    } else {
#ifdef MPI_PARALLEL
      // Use the send buffer.
      ppb = &send_sh_[shid];
#endif
    }
    // Load the particle to particle buffer
    LoadParticleBuffer(ppb, k);

    // Pop the particle from the current MeshBlock.
    RemoveOneParticle(k);
  }  // loop over all particles

  for (int upper=0; upper<2; upper++) {
    if (pbval_->is_shear[upper]) {
      for (int n=0; n<4; n++) {
        SimpleNeighborBlock& snb = pbval_->sb_data_[upper].send_neighbor[n];
        if (snb.rank == Globals::my_rank) {
          // neighbor is on the same processor
          // set the BoundaryStatus flag on the destination buffer
          Particles *ppar = pmy_mesh_->FindMeshBlock(snb.gid)->ppars[ipar];
          ppar->bstatus_recv_sh_[n+upper*4] =
              (ppar->recv_sh_[n+upper*4].npar_ > 0) ? BoundaryStatus::arrived
                                                   : BoundaryStatus::completed;
        } else {
#ifdef MPI_PARALLEL
          // neighbor is on the different processor
          // send particle using MPI
          ParticleBuffer& send = send_sh_[n+upper*4];
          if (bstatus_send_sh_[n+upper*4] == BoundaryStatus::waiting) {
            SendParticleBuffer(send, snb.rank);
          }
#endif
        }
      }
    }
  }
}


//--------------------------------------------------------------------------------------
//! \fn bool Particles::ReceiveFromNeighborsShear()
//! \brief receives particles shifted out due to shear and returns a flag indicating
//!        if all receives are completed.

bool Particles::ReceiveFromNeighborsShear() {
  bool flag = true;
  for (int upper=0; upper<2; upper++) {
    if (pbval_->is_shear[upper]) {
      for (int n=0; n<4; n++) {
        SimpleNeighborBlock& snb = pbval_->sb_data_[upper].recv_neighbor[n];
        enum BoundaryStatus& bstatus = bstatus_recv_sh_[n+upper*4];
        ParticleBuffer& recv = recv_sh_[n+upper*4];
#ifdef MPI_PARALLEL
        ReceiveParticleBuffer(snb.rank, recv, bstatus);
#endif
        switch (bstatus) {
          case BoundaryStatus::completed:
            break;

          case BoundaryStatus::waiting:
            flag = false;
            break;

          case BoundaryStatus::arrived:
            FlushReceiveBuffer(recv);
            bstatus = BoundaryStatus::completed;
            break;
        }
      }
    }
  }

  return flag;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ApplyBoundaryConditionsShear(int k, Real &x1, Real &x2, Real &x3)
//! \brief apply shift due to shear boudnary crossings

void Particles::ApplyBoundaryConditionsShear(int k, Real &x1, Real &x2, Real &x3) {
  bool flag = false;
  RegionSize& mesh_size = pmy_mesh_->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;

  // Find the mesh coordinates.
  Real x10, x20, x30;
  pcoord->CartesianToMeshCoords(xp_(k), yp_(k), zp_(k), x1, x2, x3);
  pcoord->CartesianToMeshCoords(xp0_(k), yp0_(k), zp0_(k), x10, x20, x30);

  // Convert velocity vectors in mesh coordinates.
  Real vp1, vp2, vp3, vp10, vp20, vp30;
  pcoord->CartesianToMeshCoordsVector(xp_(k), yp_(k), zp_(k),
                                      vpx_(k), vpy_(k), vpz_(k), vp1, vp2, vp3);
  pcoord->CartesianToMeshCoordsVector(xp0_(k), yp0_(k), zp0_(k),
                                      vpx0_(k), vpy0_(k), vpz0_(k), vp10, vp20, vp30);

  // Apply periodic boundary conditions in X1.
  Real yshear = qomL_*pmy_mesh_->time;
  Real deltay = std::fmod(yshear, mesh_size.x2len);

  if (auxprop(ish,k) == 1) {
    // this particle crossed shear boundary to the right
    x2 += deltay;
    x20 += deltay;
    vp2 += qomL_;
    vp20 += qomL_;
    flag = true;
    if (x2 >= mesh_size.x2max) {
      // this particle can cross the y-boundary to the top again
      x2 -= mesh_size.x2len;
      x20 -= mesh_size.x2len;
    }
  } else if (auxprop(ish,k) == -1) {
    // this particle crossed shear boundary to the left
    x2 -= deltay;
    x20 -= deltay;
    vp2 -= qomL_;
    vp20 -= qomL_;
    flag = true;
    if (x2 < mesh_size.x2min) {
      // this particle can cross the y-boundary to the bottom again
      x2 += mesh_size.x2len;
      x20 += mesh_size.x2len;
    }
  }

  if (flag) {
    // Convert positions and velocities back in Cartesian coordinates.
    pcoord->MeshCoordsToCartesian(x1, x2, x3, xp_(k), yp_(k), zp_(k));
    pcoord->MeshCoordsToCartesian(x10, x20, x30, xp0_(k), yp0_(k), zp0_(k));
    pcoord->MeshCoordsToCartesianVector(x1, x2, x3,
                                        vp1, vp2, vp3, vpx_(k), vpy_(k), vpz_(k));
    pcoord->MeshCoordsToCartesianVector(x10, x20, x30,
                                        vp10, vp20, vp30, vpx0_(k), vpy0_(k), vpz0_(k));
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::StartReceivingParticlesShear()
//! \brief initialize boundary status and MPI tags
//!
//! Since the meshblocks to be communicated due to shear change in time,
//! communication tags, boundary status should be set after the ComputeShear call,
//! which is done in StartupTaskList. This is necessary function call at every substeps.

void Particles::StartReceivingParticlesShear() {
  for (int upper=0; upper<2; upper++) {
    if (pbval_->is_shear[upper]) {
      for (int n=0; n<4; n++) {
        if (pbval_->sb_data_[upper].recv_count[n]>0) {
          bstatus_recv_sh_[n+upper*4] = BoundaryStatus::waiting;
        } else {
          bstatus_recv_sh_[n+upper*4] = BoundaryStatus::completed;
        }
#ifdef MPI_PARALLEL
        if (pbval_->sb_data_[upper].send_count[n]>0) {
          bstatus_send_sh_[n+upper*4] = BoundaryStatus::waiting;
        } else {
          bstatus_send_sh_[n+upper*4] = BoundaryStatus::completed;
        }
        // assign unique tag for communications due to shear
        // tag = local id of destination (remaining bits) + bufid (6 bits)
        // + particle container id (3 bits) + npar,intprop,realprop(2 bits)
        SimpleNeighborBlock& snb = pbval_->sb_data_[upper].send_neighbor[n];
        ParticleBuffer& recv = recv_sh_[n+upper*4];
        ParticleBuffer& send = send_sh_[n+upper*4];
        send.tag = (snb.lid<<11) | ((n+upper*4) << 5) | (ipar << 2);
        recv.tag = (pmy_block->lid<<11) | ((n+upper*4) << 5) | (ipar << 2);
#endif
      }
    }
  }
}
//--------------------------------------------------------------------------------------
//! \fn void Particles::ClearBoundaryShear()
//! \brief resets boundary for particle transportation due to shear.

void Particles::ClearBoundaryShear() {
  for (int upper=0; upper<2; upper++) {
    if (pbval_->is_shear[upper]) {
      for (int n=0; n<4; n++) {
        bstatus_recv_sh_[n+upper*4] = BoundaryStatus::completed;
#ifdef MPI_PARALLEL
        ParticleBuffer& recv = recv_sh_[n+upper*4];
        ParticleBuffer& send = send_sh_[n+upper*4];
        recv.flagn = recv.flagi = recv.flagr = 0;
        send.npar_ = 0;
        bstatus_send_sh_[n+upper*4] = BoundaryStatus::completed;
#endif
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn int CheckSide(int xi, nx, int xi1, int xi2)
//! \brief returns -1 if xi < xi1, +1 if xi > xi2, or 0 otherwise.

inline int CheckSide(int xi, int xi1, int xi2) {
  if (xi < xi1) return -1;
  if (xi > xi2) return +1;
  return 0;
}
