//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file sink_particles.cpp
//! \brief implements functions in the SinkParticles class

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "particles.hpp"

int sgn(int val) {
  return (0 < val) - (val < 0);
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::SinkParticles(MeshBlock *pmb, ParameterInput *pin)
//! \brief constructs a SinkParticles instance.

SinkParticles::SinkParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp)
  : StarParticles(pmb, pin, pp) {
  int xorder = pmb->precon->xorder;
  if (xorder + rctrl != noverlap_) {
    std::stringstream msg;
    msg << "### FATAL ERROR in SinkParticles constructor" << std::endl
      << "Control volume radius = " << rctrl << " plus the required number of ghost"
      << " cells for hydro/MHD = " << xorder << " does not match with the number of"
      << " overlapping cells for ghost particle exchange = " << noverlap_ << std::endl;
    ATHENA_ERROR(msg);
  }
  if (xorder + 1 > NGHOST) {
    std::stringstream msg;
    msg << "### FATAL ERROR in SinkParticles constructor" << std::endl
      << "At least " << xorder + 1 << " ghost cells are required to extrapolate the"
      << " control volume." << std::endl;
    ATHENA_ERROR(msg);
  }
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::~SinkParticles()
//! \brief destroys a SinkParticles instance.

SinkParticles::~SinkParticles() {
  // nothing to do
  return;
}


//--------------------------------------------------------------------------------------
//! \fn SinkParticles::InteractWithMesh()
//! \brief Interact with Mesh variables (e.g., feedback, accretion)

void SinkParticles::InteractWithMesh() {
  AccreteMass();
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::SetControlVolume()
//! \brief Interface to set current control volume of all particles

void SinkParticles::SetControlVolume() {
  AthenaArray<Real> &cons = pmy_block->phydro->u;
  // loop over all active plus ghost particles
  for (int idx=0; idx<npar_+npar_gh_; ++idx) {
    // find the indices of the particle-containing cell.
    int ip, jp, kp, ip0, jp0, kp0;
    GridIndex(xp(idx), yp(idx), zp(idx), ip, jp, kp);
    SetControlVolume(cons, ip, jp, kp);
  }
}


//--------------------------------------------------------------------------------------
//! \fn SinkParticles::AccreteMass()
//! \brief accrete gas from neighboring cells

void SinkParticles::AccreteMass() {
  if (COORDINATE_SYSTEM != "cartesian") {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [SinkParticles::AccreteMass]" << std::endl
        << "Only Cartesian coordinate system is supported " << std::endl;
    ATHENA_ERROR(msg);
  }
  AthenaArray<Real> &cons = pmy_block->phydro->u;

  // loop over all active particles
  for (int idx=0; idx<npar_; ++idx) {
    // Determine the dM_sink accreted by the sink particle.
    // Because we reset the density in the control volume by extrapolating from the
    // adjacent active cells, the total mass in the control volume can change.
    // The mass dM_flux that has flown into the control volume during dt must be equal
    // to the dM_sink plus the mass change dM_ctrl inside the control volume. That is,
    // we can calculate dM_sink by
    //   dM_sink = dM_flux - dM_ctrl       -- (1)
    // Meanwhile, hydro integrator will update M^{n}_ctrl at time t^n (which is already
    // reset to the extrapolated value) to M^{n+1}, which will be subsequently reset to
    // the extrapolated value M^{n+1}_ctrl. This is done by
    //   M^{n+1} = M^{n}_ctrl + dM_flux    -- (2)
    // Therefore, instead using Riemann fluxes directly to calculate dM_flux in eq. (1),
    // we can use eq. (2) to substitute dM_flux in eq. (1) with M^{n+1} - M^{n}_ctrl,
    // yielding
    //   dM_sink = M^{n+1} - M^{n+1}_ctrl  -- (3)
    // TODO AMR compatibility?

    // Step 0. Prepare

    // find the indices of the particle-containing cell.
    int ip, jp, kp, ip0, jp0, kp0;
    GridIndex(xp(idx), yp(idx), zp(idx), ip, jp, kp);
    GridIndex(xp0(idx), yp0(idx), zp0(idx), ip0, jp0, kp0);

    // Step 1(a). Calculate total mass in the control volume M^{n+1} updated by hydro
    // integrator, before applying extrapolation.
    Real m{0.}, M1{0.}, M2{0.}, M3{0.};
    for (int k=kp-rctrl; k<=kp+rctrl; ++k) {
      for (int j=jp-rctrl; j<=jp+rctrl; ++j) {
        for (int i=ip-rctrl; i<=ip+rctrl; ++i) {
          Real dV = pmy_block->pcoord->GetCellVolume(k,j,i);
          m += cons(IDN,k,j,i)*dV;
          M1 += cons(IM1,k,j,i)*dV;
          M2 += cons(IM2,k,j,i)*dV;
          M3 += cons(IM3,k,j,i)*dV;
        }
      }
    }
    // Step 1(b). Reset the density inside the control volume by extrapolation
    SetControlVolume(cons, ip, jp, kp);
    // Step 1(c). Calculate M^{n+1}_ctrl
    Real mext{0.}, M1ext{0.}, M2ext{0.}, M3ext{0.};
    for (int k=kp-rctrl; k<=kp+rctrl; ++k) {
      for (int j=jp-rctrl; j<=jp+rctrl; ++j) {
        for (int i=ip-rctrl; i<=ip+rctrl; ++i) {
          Real dV = pmy_block->pcoord->GetCellVolume(k,j,i);
          mext += cons(IDN,k,j,i)*dV;
          M1ext += cons(IM1,k,j,i)*dV;
          M2ext += cons(IM2,k,j,i)*dV;
          M3ext += cons(IM3,k,j,i)*dV;
        }
      }
    }
    // Step 1(d). Calculate dM_sink by subtracting M^{n+1}_ctrl from M^{n+1}
    Real dm = m - mext;
    Real dM1 = M1 - M1ext;
    Real dM2 = M2 - M2ext;
    Real dM3 = M3 - M3ext;

    // Check whether the particle has crossed the grid boundaries.
    // If so, reset the old control volume and add that change to the sink particle
    // SMOON: No need to avoid double counting if not for performance reasons; because
    // we reset overlap region again, the additional change in the overlap region should
    // be accounted in the sink accretion.
    if ((ip != ip0) || (jp != jp0) || (kp != kp0)) {
      // Step 2(a). Calculate total mass in the control volume M^{n+1} updated by hydro
      // integrator, before applying extrapolation.
      Real m{0.}, M1{0.}, M2{0.}, M3{0.};
      for (int k=kp0-rctrl; k<=kp0+rctrl; ++k) {
        for (int j=jp0-rctrl; j<=jp0+rctrl; ++j) {
          for (int i=ip0-rctrl; i<=ip0+rctrl; ++i) {
            Real dV = pmy_block->pcoord->GetCellVolume(k,j,i);
            m += cons(IDN,k,j,i)*dV;
            M1 += cons(IM1,k,j,i)*dV;
            M2 += cons(IM2,k,j,i)*dV;
            M3 += cons(IM3,k,j,i)*dV;
          }
        }
      }
      // Step 2(b). Reset the density inside the control volume by extrapolation
      SetControlVolume(cons, ip0, jp0, kp0);
      // Step 2(c). Calculate M^{n+1}_ctrl
      Real mext{0.}, M1ext{0.}, M2ext{0.}, M3ext{0.};
      for (int k=kp0-rctrl; k<=kp0+rctrl; ++k) {
        for (int j=jp0-rctrl; j<=jp0+rctrl; ++j) {
          for (int i=ip0-rctrl; i<=ip0+rctrl; ++i) {
            Real dV = pmy_block->pcoord->GetCellVolume(k,j,i);
            mext += cons(IDN,k,j,i)*dV;
            M1ext += cons(IM1,k,j,i)*dV;
            M2ext += cons(IM2,k,j,i)*dV;
            M3ext += cons(IM3,k,j,i)*dV;
          }
        }
      }
      // Step 2(d). Calculate dM_sink by subtracting M^{n+1}_ctrl from M^{n+1}
      dm += m - mext;
      dM1 += M1 - M1ext;
      dM2 += M2 - M2ext;
      dM3 += M3 - M3ext;
    }

    // Step 3. Update mass and velocity of the particle
    Real minv = 1.0 / (mass(idx) + dm);
    vpx(idx) = (mass(idx)*vpx(idx) + dM1)*minv;
    vpy(idx) = (mass(idx)*vpy(idx) + dM2)*minv;
    vpz(idx) = (mass(idx)*vpz(idx) + dM3)*minv;
    mass(idx) += dm;
  } // end of the loop over particles

  // loop over all ghost particles and reset the control volume
  for (int idx=npar_; idx<npar_+npar_gh_; ++idx) {
    // find the indices of the particle-containing cell.
    int ip, jp, kp, ip0, jp0, kp0;
    GridIndex(xp(idx), yp(idx), zp(idx), ip, jp, kp);
    GridIndex(xp0(idx), yp0(idx), zp0(idx), ip0, jp0, kp0);
    SetControlVolume(cons, ip, jp, kp);
    if ((ip != ip0) || (jp != jp0) || (kp != kp0))
      SetControlVolume(cons, ip0, jp0, kp0);
  }
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::SetControlVolume(AthenaArray<Real> &cons, int ip, int jp, int kp)
//! \brief set control volume quantities by extrapolating from neighboring active cells.

void SinkParticles::SetControlVolume(AthenaArray<Real> &cons, int ip, int jp, int kp) {
  // Do extrapolation using "face-neighbors", that is, neighboring cells
  // that are outside the control volume and share a cell face with the
  // cell being extrapolated.
  // In Athena-TIGRESS, larger stencil that include edge- and corner-neighbors
  // were not compatible with shearing box. Is that true also in athena++?
  // TODO AMR compatibility?

  int xorder = pmy_block->precon->xorder;
  const int is(pmy_block->is), ie(pmy_block->ie);
  const int js(pmy_block->js), je(pmy_block->je);
  const int ks(pmy_block->ks), ke(pmy_block->ke);
  const int il(is - xorder), iu(ie + xorder);
  const int jl(js - xorder), ju(je + xorder);
  const int kl(ks - xorder), ku(ke + xorder);
  int cil, ciu, cjl, cju, ckl, cku;

  // Start extrapolation from the outermost shell, marching inward
  for (int s=rctrl; s>=1; --s) {
    // 6 front faces
    // Each has one neighbor. Do simple copy.

    // x1-face
    // SMOON: these loop limits restricts the cells to be modified to be within
    // [is - nghost, ie + nghost], etc., where nghost is the number of ghost cells
    // that are required for the hydro/MHD integrator, which could be different
    // from NGHOST variable. This is because 1) only these cells need to be updated
    // and 2) otherwise loop indices go beyond the proper limits of "cons" array,
    // particularly for ghost particles.
    ckl = std::max(kp-s+1, kl);
    cjl = std::max(jp-s+1, jl);
    cil = std::max(ip-s,   il);
    cku = std::min(kp+s-1, ku);
    cju = std::min(jp+s-1, ju);
    ciu = std::min(ip+s  , iu);
    for (int k=ckl; k<=cku; ++k) {
      for (int j=cjl; j<=cju; ++j) {
        for (int i=cil; i<=ciu; i+=2*s) {
          int ioff = sgn(i-ip);
          cons(IDN,k,j,i) = cons(IDN,k,j,i+ioff);
          cons(IM1,k,j,i) = cons(IM1,k,j,i+ioff);
          cons(IM2,k,j,i) = cons(IM2,k,j,i+ioff);
          cons(IM3,k,j,i) = cons(IM3,k,j,i+ioff);
        }
      }
    }
    // x2-face
    ckl = std::max(kp-s+1, kl);
    cjl = std::max(jp-s  , jl);
    cil = std::max(ip-s+1, il);
    cku = std::min(kp+s-1, ku);
    cju = std::min(jp+s  , ju);
    ciu = std::min(ip+s-1, iu);
    for (int k=ckl; k<=cku; ++k) {
      for (int j=cjl; j<=cju; j+=2*s) {
        for (int i=cil; i<=ciu; ++i) {
          int joff = sgn(j-jp);
          cons(IDN,k,j,i) = cons(IDN,k,j+joff,i);
          cons(IM1,k,j,i) = cons(IM1,k,j+joff,i);
          cons(IM2,k,j,i) = cons(IM2,k,j+joff,i);
          cons(IM3,k,j,i) = cons(IM3,k,j+joff,i);
        }
      }
    }
    // x3-face
    ckl = std::max(kp-s  , kl);
    cjl = std::max(jp-s+1, jl);
    cil = std::max(ip-s+1, il);
    cku = std::min(kp+s  , ku);
    cju = std::min(jp+s-1, ju);
    ciu = std::min(ip+s-1, iu);
    for (int k=ckl; k<=cku; k+=2*s) {
      for (int j=cjl; j<=cju; ++j) {
        for (int i=cil; i<=ciu; ++i) {
          int koff = sgn(k-kp);
          cons(IDN,k,j,i) = cons(IDN,k+koff,j,i);
          cons(IM1,k,j,i) = cons(IM1,k+koff,j,i);
          cons(IM2,k,j,i) = cons(IM2,k+koff,j,i);
          cons(IM3,k,j,i) = cons(IM3,k+koff,j,i);
        }
      }
    }

    // 8 corners
    // Each has three neighbors. Average them.
    ckl = std::max(kp-s, kl);
    cjl = std::max(jp-s, jl);
    cil = std::max(ip-s, il);
    cku = std::min(kp+s, ku);
    cju = std::min(jp+s, ju);
    ciu = std::min(ip+s, iu);
    for (int k=ckl; k<=cku; k+=2*s) {
      for (int j=cjl; j<=cju; j+=2*s) {
        for (int i=cil; i<=ciu; i+=2*s) {
          int koff = sgn(k-kp);
          int joff = sgn(j-jp);
          int ioff = sgn(i-ip);
          Real davg = ONE_3RD*(cons(IDN,k     ,j     ,i+ioff) +
                               cons(IDN,k     ,j+joff,i     ) +
                               cons(IDN,k+koff,j     ,i     ));
          Real M1avg = ONE_3RD*(cons(IM1,k     ,j     ,i+ioff) +
                                cons(IM1,k     ,j+joff,i     ) +
                                cons(IM1,k+koff,j     ,i     ));
          Real M2avg = ONE_3RD*(cons(IM2,k     ,j     ,i+ioff) +
                                cons(IM2,k     ,j+joff,i     ) +
                                cons(IM2,k+koff,j     ,i     ));
          Real M3avg = ONE_3RD*(cons(IM3,k     ,j     ,i+ioff) +
                                cons(IM3,k     ,j+joff,i     ) +
                                cons(IM3,k+koff,j     ,i     ));
          cons(IDN,k,j,i) = davg;
          cons(IM1,k,j,i) = M1avg;
          cons(IM2,k,j,i) = M2avg;
          cons(IM3,k,j,i) = M3avg;
        }
      }
    }

    // 4 sides for 3 middle-slices
    // Each has two neighbors. Average them.

    // x1-slice
    ckl = std::max(kp-s  , kl);
    cjl = std::max(jp-s  , jl);
    cil = std::max(ip-s+1, il);
    cku = std::min(kp+s  , ku);
    cju = std::min(jp+s  , ju);
    ciu = std::min(ip+s-1, iu);
    for (int k=ckl; k<=cku; k+=2*s) {
      for (int j=cjl; j<=cju; j+=2*s) {
        for (int i=cil; i<=ciu; ++i) {
          int koff = sgn(k-kp);
          int joff = sgn(j-jp);
          Real davg = 0.5*(cons(IDN,k+koff,j,i) + cons(IDN,k,j+joff,i));
          Real M1avg = 0.5*(cons(IM1,k+koff,j,i) + cons(IM1,k,j+joff,i));
          Real M2avg = 0.5*(cons(IM2,k+koff,j,i) + cons(IM2,k,j+joff,i));
          Real M3avg = 0.5*(cons(IM3,k+koff,j,i) + cons(IM3,k,j+joff,i));
          cons(IDN,k,j,i) = davg;
          cons(IM1,k,j,i) = M1avg;
          cons(IM2,k,j,i) = M2avg;
          cons(IM3,k,j,i) = M3avg;
        }
      }
    }
    // x2-slice
    ckl = std::max(kp-s  , kl);
    cjl = std::max(jp-s+1, jl);
    cil = std::max(ip-s  , il);
    cku = std::min(kp+s  , ku);
    cju = std::min(jp+s-1, ju);
    ciu = std::min(ip+s  , iu);
    for (int k=ckl; k<=cku; k+=2*s) {
      for (int j=cjl; j<=cju; ++j) {
        for (int i=cil; i<=ciu; i+=2*s) {
          int koff = sgn(k-kp);
          int ioff = sgn(i-ip);
          Real davg = 0.5*(cons(IDN,k+koff,j,i) + cons(IDN,k,j,i+ioff));
          Real M1avg = 0.5*(cons(IM1,k+koff,j,i) + cons(IM1,k,j,i+ioff));
          Real M2avg = 0.5*(cons(IM2,k+koff,j,i) + cons(IM2,k,j,i+ioff));
          Real M3avg = 0.5*(cons(IM3,k+koff,j,i) + cons(IM3,k,j,i+ioff));
          cons(IDN,k,j,i) = davg;
          cons(IM1,k,j,i) = M1avg;
          cons(IM2,k,j,i) = M2avg;
          cons(IM3,k,j,i) = M3avg;
        }
      }
    }
    // x3-slice
    ckl = std::max(kp-s+1, kl);
    cjl = std::max(jp-s  , jl);
    cil = std::max(ip-s  , il);
    cku = std::min(kp+s-1, ku);
    cju = std::min(jp+s  , ju);
    ciu = std::min(ip+s  , iu);
    for (int k=ckl; k<=cku; ++k) {
      for (int j=cjl; j<=cju; j+=2*s) {
        for (int i=cil; i<=ciu; i+=2*s) {
          int joff = sgn(j-jp);
          int ioff = sgn(i-ip);
          Real davg = 0.5*(cons(IDN,k,j,i+ioff) + cons(IDN,k,j+joff,i));
          Real M1avg = 0.5*(cons(IM1,k,j,i+ioff) + cons(IM1,k,j+joff,i));
          Real M2avg = 0.5*(cons(IM2,k,j,i+ioff) + cons(IM2,k,j+joff,i));
          Real M3avg = 0.5*(cons(IM3,k,j,i+ioff) + cons(IM3,k,j+joff,i));
          cons(IDN,k,j,i) = davg;
          cons(IM1,k,j,i) = M1avg;
          cons(IM2,k,j,i) = M2avg;
          cons(IM3,k,j,i) = M3avg;
        }
      }
    }
  }
  if ((kp>=kl)&&(kp<=ku)&&(jp>=jl)&&(jp<=ju)&&(ip>=il)&&(ip<=iu)) {
    // finally, fill the central cell containing the particle
    Real davg = (cons(IDN,kp,jp,ip-1) + cons(IDN,kp,jp,ip+1) +
                 cons(IDN,kp,jp-1,ip) + cons(IDN,kp,jp+1,ip) +
                 cons(IDN,kp-1,jp,ip) + cons(IDN,kp+1,jp,ip))/6.;
    Real M1avg = (cons(IM1,kp,jp,ip-1) + cons(IM1,kp,jp,ip+1) +
                  cons(IM1,kp,jp-1,ip) + cons(IM1,kp,jp+1,ip) +
                  cons(IM1,kp-1,jp,ip) + cons(IM1,kp+1,jp,ip))/6.;
    Real M2avg = (cons(IM2,kp,jp,ip-1) + cons(IM2,kp,jp,ip+1) +
                  cons(IM2,kp,jp-1,ip) + cons(IM2,kp,jp+1,ip) +
                  cons(IM2,kp-1,jp,ip) + cons(IM2,kp+1,jp,ip))/6.;
    Real M3avg = (cons(IM3,kp,jp,ip-1) + cons(IM3,kp,jp,ip+1) +
                  cons(IM3,kp,jp-1,ip) + cons(IM3,kp,jp+1,ip) +
                  cons(IM3,kp-1,jp,ip) + cons(IM3,kp+1,jp,ip))/6.;
    cons(IDN,kp,jp,ip) = davg;
    cons(IM1,kp,jp,ip) = M1avg;
    cons(IM2,kp,jp,ip) = M2avg;
    cons(IM3,kp,jp,ip) = M3avg;
  }
}
