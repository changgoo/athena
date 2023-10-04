//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file turb.cpp
//! \brief Problem generator for turbulence driver

// C headers

// C++ headers
#include <cmath>
#include <ctime>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fft/athena_fft.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = pin->GetReal("problem","four_pi_G");
    SetFourPiG(four_pi_G);
  }

  // turb_flag is initialzed in the Mesh constructor to 0 by default;
  // turb_flag = 1 for decaying turbulence
  // turb_flag = 2 for impulsively driven turbulence
  // turb_flag = 3 for continuously driven turbulence
  turb_flag = pin->GetInteger("problem","turb_flag");
  if (turb_flag != 0) {
#ifndef FFT
    std::stringstream msg;
    msg << "### FATAL ERROR in TurbulenceDriver::TurbulenceDriver" << std::endl
        << "non zero Turbulence flag is set without FFT!" << std::endl;
    ATHENA_ERROR(msg);
    return;
#endif
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real b0, angle;
  if (MAGNETIC_FIELDS_ENABLED) {
    b0 = pin->GetReal("problem", "b0");
    angle = (PI/180.0)*pin->GetReal("problem", "angle");
  }

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IDN,k,j,i) = 1.0;

        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = 1.0;
        }
      }
    }
  }

  // initialize interface B and total energy
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            pfield->b.x1f(k,j,i) = b0 * std::cos(angle);
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            Real phi = pcoord->x2v(j);
            pfield->b.x1f(k,j,i) =
                b0 * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
          } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            Real theta = pcoord->x2v(j);
            Real phi = pcoord->x3v(k);
            pfield->b.x1f(k,j,i) = b0 * std::abs(std::sin(theta))
                                   * (std::cos(angle) * std::cos(phi)
                                      + std::sin(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is; i<=ie; ++i) {
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            pfield->b.x2f(k,j,i) = b0 * std::sin(angle);
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            Real phi = pcoord->x2v(j);
            pfield->b.x2f(k,j,i) =
                b0 * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
          } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            Real theta = pcoord->x2v(j);
            Real phi = pcoord->x3v(k);
            pfield->b.x2f(k,j,i) = b0 * std::cos(theta)
                                   * (std::cos(angle) * std::cos(phi)
                                      + std::sin(angle) * std::sin(phi));
            if (std::sin(theta) < 0.0)
              pfield->b.x2f(k,j,i) *= -1.0;
          }
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0
              || std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            pfield->b.x3f(k,j,i) = 0.0;
          } else { //if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            Real phi = pcoord->x3v(k);
            pfield->b.x3f(k,j,i) =
                b0 * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
          }
        }
      }
    }
    if (NON_BAROTROPIC_EOS) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            phydro->u(IEN,k,j,i) += 0.5*b0*b0;
          }
        }
      }
    }
  }
}


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
}
