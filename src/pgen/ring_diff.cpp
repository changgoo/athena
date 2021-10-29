//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ring_diff.cpp
//! \brief ring diffusion test
//!
//! Thermal diffusion under in-plane, circular magnetic fields
//! To use this, main integrator except diffusive process has to be turned off
//!   - For STS, simply comment out
//!     ptlist->DoTaskListOneStage(pmesh, stage);
//!   - For explicit diffusion, need to modify time integrator task list
//! REFERENCE:
//! - Parrish, I.~J. \& Stone, J.~M.\ 2005, \apj, 633, 334. doi:10.1086/444589
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt(), erfc
#include <cstring>    // strcmp()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

Real HistoryEleak(MeshBlock *pmb, int iout);
Real HistoryL1Error(MeshBlock *pmb, int iout);
Real AnalyticSolution(Real phi, Real r, Real t);

static const Real r1 = 0.5, r2 = 0.7;
static const Real phi1 = PI*5/12., phi2 = PI*7/12.;

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (!MAGNETIC_FIELDS_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ring_diff.cpp ProblemGenerator" << std::endl
        << "Magnetic field must be turned on" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (!NON_BAROTROPIC_EOS) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ring_diff.cpp ProblemGenerator" << std::endl
        << "EOS must be non barotropic" << std::endl;
    ATHENA_ERROR(msg);
  }
  int iprob = pin->GetOrAddInteger("problem", "iprob", 1);

  if (iprob == 1) {
    AllocateUserHistoryOutput(2);
    EnrollUserHistoryOutput(0, HistoryEleak, "Eleak");
    EnrollUserHistoryOutput(1, HistoryL1Error, "L1");
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real x1, x2, x3;
  Real r, theta, phi, z;
  Real v1 = 0., v2 = 0., v3 = 0.;

  Real d0 = 1.;
  int iprob = pin->GetOrAddInteger("problem", "iprob", 1);
  Real amp = pin->GetOrAddReal("problem", "amp", 1.e-6);
  Real t0 = pin->GetOrAddReal("problem", "t0", 0.5);
  Real kappa = pin->GetOrAddReal("problem", "kappa_iso", 0.);
  kappa += pin->GetOrAddReal("problem", "kappa_aniso", 1);
  Real x10 = pin->GetOrAddReal("problem", "x10", 0.);
  Real x20 = pin->GetOrAddReal("problem", "x20", 0.6);
  Real x30 = pin->GetOrAddReal("problem", "x30", 0.);

  // set a uniform, static medium
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phydro->w(IDN,k,j,i) = d0;
        phydro->w(IVX,k,j,i) = v1;
        phydro->w(IVY,k,j,i) = v2;
        phydro->w(IVZ,k,j,i) = v3;
      }
    }
  }

  // set thermal energy to diffuse
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x1 = pcoord->x1v(i);
          x2 = pcoord->x2v(j);
          x3 = pcoord->x3v(j);
          r = std::sqrt(SQR(x1)+SQR(x2));
          phi = std::atan2(x2,x1);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          r = pcoord->x1v(i);
          phi = pcoord->x2v(j);
          x1 = r*std::cos(phi);
          x2 = r*std::sin(phi);
          x3 = pcoord->x3v(j);
        } else {
          std::stringstream msg;
          msg << "### FATAL ERROR in ring_diff.cpp ProblemGenerator" << std::endl
              << "2-d diffusion test only compatible with cartesian or"
              << std::endl << "cylindrical coord" << std::endl;
          ATHENA_ERROR(msg);
        }
        if (iprob == 1) { // in-plane, ring diffusion
          if ((r>r1) & (r<r2) & (phi>phi1) & (phi<phi2))
            phydro->w(IPR,k,j,i) = 12;
          else
            phydro->w(IPR,k,j,i) = 10;
        } else if (iprob == 2) { //gaussian diffusion
          Real rsq = SQR(x1-x10);
          if (je > js) rsq += SQR(x2-x20);
          if (ke > ks) rsq += SQR(x3-x30);
          phydro->w(IPR,k,j,i) = amp/std::pow(std::sqrt(4.*PI*kappa*t0),pmy_mesh->ndim)
                                  * std::exp(-rsq/(4.*kappa*t0));
        } else {
          std::stringstream msg;
          msg << "### FATAL ERROR in ring_diff.cpp ProblemGenerator" << std::endl
              << "unsupported iprob = " << iprob << std::endl;
          ATHENA_ERROR(msg);
        }
      }
    }
  }

  // magnetic fields
  // circular in XY plane

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie+1; ++i) {
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x1 = pcoord->x1v(i);
          x2 = pcoord->x2v(j);
          r = std::sqrt(SQR(x1)+SQR(x2));
          phi = std::atan2(x2,x1);
          pfield->b.x1f(k,j,i) = x2/r;
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          r = pcoord->x1v(i);
          phi = pcoord->x2v(j);
          x1 = r*std::cos(phi);
          x2 = r*std::sin(phi);
          pfield->b.x1f(k,j,i) = 0.0;
        }
      }
    }
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
      for (int i=is; i<=ie; ++i) {
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x1 = pcoord->x1v(i);
          x2 = pcoord->x2v(j);
          r = std::sqrt(SQR(x1)+SQR(x2));
          phi = std::atan2(x2,x1);
          pfield->b.x2f(k,j,i) = -x1/r;
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          r = pcoord->x1v(i);
          phi = pcoord->x2v(j);
          x1 = r*std::cos(phi);
          x2 = r*std::sin(phi);
          pfield->b.x2f(k,j,i) = 1.0;
        }
      }
    }
  }

  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        pfield->b.x3f(k,j,i) = 0.0;
      }
    }
  }
  // Calculate CellCentered B
  pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord,
                                      is, ie, js, je, ks, ke);

  // Initialize conserved values
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                              is, ie, js, je, ks, ke);

  return;
}

Real HistoryEleak(MeshBlock *pmb, int iout) {
  Real x1, x2;
  Real r, phi;
  Real Eleak = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, is, ie, volume);
      for (int i=is; i<=ie; i++) {
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x1 = pmb->pcoord->x1v(i);
          x2 = pmb->pcoord->x2v(j);
          r = std::sqrt(SQR(x1)+SQR(x2));
          phi = std::atan2(x2,x1);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          r = pmb->pcoord->x1v(i);
          phi = pmb->pcoord->x2v(j);
          x1 = r*std::cos(phi);
          x2 = r*std::sin(phi);
        }
        if ((r<r1) | (r>r2)) Eleak += (w(IPR,k,j,i)-10)*volume(i);
      }
    }
  }
  return Eleak;
}

Real HistoryL1Error(MeshBlock *pmb, int iout) {
  Real x1, x2;
  Real r, phi, pa;
  Real l1_error=0.0;

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, is, ie, volume);
      for (int i=is; i<=ie; i++) {
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x1 = pmb->pcoord->x1v(i);
          x2 = pmb->pcoord->x2v(j);
          r = std::sqrt(SQR(x1)+SQR(x2));
          phi = std::atan2(x2,x1);
        } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          r = pmb->pcoord->x1v(i);
          phi = pmb->pcoord->x2v(j);
          x1 = r*std::cos(phi);
          x2 = r*std::sin(phi);
        }
        if ((r>r1) & (r<r2)) {
          pa = AnalyticSolution(phi,r,pmb->pmy_mesh->time);
        } else {
          pa = 10.;
        }
        l1_error += std::abs(w(IPR,k,j,i)-pa)*volume(i);
      }
    }
  }
  return l1_error;
}

Real AnalyticSolution(Real phi, Real r, Real t) {
  Real dphi = 0.5*(phi2-phi1);
  Real phi0 = 0.5*(phi2+phi1);
  Real D = std::sqrt(4*t); // sqrt(4*t*(gamma-1)/kappa_aniso)
  return 10+std::erfc((phi-phi0-dphi)*r/D) - std::erfc((phi-phi0+dphi)*r/D);
}