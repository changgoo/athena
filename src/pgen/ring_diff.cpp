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
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
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

Real threshold;

int RefinementCondition(MeshBlock *pmb);

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (adaptive) {
    EnrollUserRefinementCondition(RefinementCondition);
    threshold = pin->GetReal("problem","thr");
  }
  if (!MAGNETIC_FIELDS_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ring_diff.cpp ProblemGenerator" << std::endl
        << "Magnetic field must be turned on" << std::endl;
    ATHENA_ERROR(msg);
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
  Real r1 = 0.5, r2 = 0.7;
  Real phi1 = PI*5/12., phi2 = PI*7/12.;
  Real d0 = 1.;
  int iprob = pin->GetOrAddInteger("problem", "iprob", 0);

  if (iprob == 1) { // 2-D diffusion
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            x1 = pcoord->x1v(i);
            x2 = pcoord->x2v(j);
            r = std::sqrt(SQR(x1)+SQR(x2));
            phi = std::atan2(x2,x1);
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            r = pcoord->x1v(i);
            phi = pcoord->x2v(j);
            x1 = r*std::cos(phi);
            x2 = r*std::sin(phi);
          } else {
            std::stringstream msg;
            msg << "### FATAL ERROR in ring_diff.cpp ProblemGenerator" << std::endl
                << "2-d diffusion test only compatible with cartesian or"
                << std::endl << "cylindrical coord" << std::endl;
            ATHENA_ERROR(msg);
          }
          phydro->w(IDN,k,j,i) = d0;

          phydro->w(IVX,k,j,i) = v1;
          phydro->w(IVY,k,j,i) = v2;
          phydro->w(IVZ,k,j,i) = v3;
          if (NON_BAROTROPIC_EOS) {
            if ((r>r1) & (r<r2) & (phi>phi1) & (phi<phi2))
              phydro->w(IPR,k,j,i) = 12;
            else
              phydro->w(IPR,k,j,i) = 10;
          }
        }
      }
    }
  } else if (iprob == 2) {

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

// refinement condition: check the maximum scalar gradient
int RefinementCondition(MeshBlock *pmb) {
  int f2 = pmb->pmy_mesh->f2, f3 = pmb->pmy_mesh->f3;
  AthenaArray<Real> &r = pmb->pscalars->r;
  Real maxeps = 0.;
  if (NSCALARS > 0) {
    if (f3) {
      for (int k=pmb->ks-1; k<=pmb->ke+1; k++) {
        for (int j=pmb->js-1; j<=pmb->je+1; j++) {
          for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
            Real eps = std::sqrt(SQR(0.5*(r(0,k,j,i+1) - r(0,k,j,i-1)))
                                 + SQR(0.5*(r(0,k,j+1,i) - r(0,k,j-1,i)))
                                 + SQR(0.5*(r(0,k+1,j,i) - r(0,k-1,j,i))));
            maxeps = std::max(maxeps, eps);
          }
        }
      }
    } else if (f2) {
      int k = pmb->ks;
      for (int j=pmb->js-1; j<=pmb->je+1; j++) {
        for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
          Real eps = std::sqrt(SQR(0.5*(r(0,k,j,i+1) - r(0,k,j,i-1)))
                               + SQR(0.5*(r(0,k,j+1,i) - r(0,k,j-1,i))));
          maxeps = std::max(maxeps, eps);
        }
      }
    } else {
      int k = pmb->ks;
      int j = pmb->js;
      for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
        Real eps = std::sqrt(SQR(0.5*(r(0,k,j,i+1) - r(0,k,j,i-1))));
        maxeps = std::max(maxeps, eps);
      }
    }
  } else {
    return 0;
  }

  if (maxeps > threshold) return 1;
  if (maxeps < 0.25*threshold) return -1;
  return 0;
}
