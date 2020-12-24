//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file msa.cpp
//! \brief Modified swing amplification problem generator.
//! REFERENCE: Kim & Ostriker, ApJ, 2001
//======================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <fstream>    // ofstream
#include <iomanip>    // setprecision
#include <iostream>   // cout, endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../outputs/outputs.hpp"
#include "../parameter_input.hpp"

#if MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires does not work with MHD."
#endif

namespace {
Real cs, gm1, d0, p0, gconst;
Real Q, nJ, beta, amp;
int nwx, nwy; // wavenumbers
Real x1size,x2size,x3size;
Real qshear, Omega0; // shear parameters
} // namespace

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties
//======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (!shear_periodic) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ssheet.cpp ProblemGenerator" << std::endl
        << "This problem generator requires shearing box."   << std::endl;
    ATHENA_ERROR(msg);
  }

  if (mesh_size.nx2 == 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ssheet.cpp ProblemGenerator" << std::endl
        << "This problem does NOT work on a 1D grid." << std::endl;
    ATHENA_ERROR(msg);
  }

  x1size = mesh_size.x1max - mesh_size.x1min;
  x2size = mesh_size.x2max - mesh_size.x2min;
  x3size = mesh_size.x3max - mesh_size.x3min;

  // shearing box parameters
  qshear = pin->GetReal("orbital_advection","qshear");
  Omega0 = pin->GetReal("orbital_advection","Omega0");

  // hydro parameters
  if (NON_BAROTROPIC_EOS) {
    gm1 = (pin->GetReal("hydro","gamma") - 1.0);
  }

  // MSA parameters
  Q = pin->GetReal("problem","Q");
  nJ = pin->GetReal("problem","nJ");
  beta = pin->GetReal("problem","beta");
  amp = pin->GetReal("problem","amp");
  nwx = pin->GetInteger("problem","nwx");
  nwy = pin->GetInteger("problem","nwy");
  cs = std::sqrt(4.0-2.0*qshear)/PI/nJ/Q;
  d0 = 1.0;
  if (NON_BAROTROPIC_EOS) {
    p0 = SQR(cs)/(gm1+1.0);
  }

  if (SELF_GRAVITY_ENABLED) {
    gconst = nJ*SQR(cs);
    SetGravitationalConstant(gconst);
    Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
    SetGravityThreshold(eps);
    SetMeanDensity(d0);
  }
  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  if (gid == 0) {
    std::cout << "cs = " << cs << std::endl;
    std::cout << "G = " << gconst << std::endl;
    std::cout << "[ssheet.cpp]: [Lx,Ly,Lz] = [" <<x1size <<","<<x2size
              <<","<<x3size<<"]"<<std::endl;
  }

  // set wavenumbers
  Real kx = (TWO_PI/x1size)*(static_cast<Real>(nwx));
  Real ky = (TWO_PI/x2size)*(static_cast<Real>(nwy));

  Real x1, x2, rd, rp, rvx, rvy;
  // update the physical variables as initial conditions
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        x1 = pcoord->x1v(i);
        x2 = pcoord->x2v(j);
        rd = amp*std::cos(kx*x1 + ky*x2);
        rvx = amp*kx/ky*std::sin(kx*x1 + ky*x2);
        rvy = amp*std::sin(kx*x1 + ky*x2);
        phydro->u(IDN,k,j,i) = d0+rd;
        phydro->u(IM1,k,j,i) = (d0+rd)*rvx;
        phydro->u(IM2,k,j,i) = (d0+rd)*(rvy - qshear*Omega0*x1);
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          rp = SQR(cs)*rd;
          phydro->u(IEN,k,j,i) = (p0+rp)/gm1 + 0.5*(SQR(phydro->u(IM1,k,j,i)) +
                                                    SQR(phydro->u(IM2,k,j,i)) +
                                                    SQR(phydro->u(IM3,k,j,i))
                                                    ) / phydro->u(IDN,k,j,i);
        }
      }
    }
  }
  return;
}
