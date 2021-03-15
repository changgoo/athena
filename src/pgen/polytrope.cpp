//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file polytrope.cpp
//  \brief Problem generator for polytropic equilibrium.
//
//========================================================================================

// C headers

// C++ headers
#include <cmath>     // sqrt()
#include <ctime>
#include <iomanip>   // setprecision, scientific
#include <iostream>  // cout, endl
#include <sstream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties
//======================================================================================
//
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = pin->GetReal("self_gravity","four_pi_G");
    Real eps = pin->GetOrAddReal("self_gravity","grav_eps", 0.0);
    SetFourPiG(four_pi_G);
    SetGravityThreshold(eps);
  }
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gamma = peos->GetGamma();
  Real gm1 = gamma - 1.0;
  if (gamma != 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in polytrope.cpp ProblemGenerator" << std::endl
        << "invalid gamma " << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  Real four_pi_G = pmy_mesh->four_pi_G_;
  // center position of the polytrope
  Real x1c   = pin->GetReal("problem","x1c");
  Real x2c   = pin->GetReal("problem","x2c");
  Real x3c   = pin->GetReal("problem","x3c");
  // background velociy
  Real v1    = pin->GetReal("problem","v1");
  Real v2    = pin->GetReal("problem","v2");
  Real v3    = pin->GetReal("problem","v3");
  Real rhoc  = pin->GetReal("problem","rhoc");
  Real damb  = pin->GetReal("problem","damb");  // ambient density
  Real pamb  = pin->GetReal("problem","pamb");  // ambient pressure
  Real rsurf = pin->GetReal("problem","rsurf"); // radius of the polytrope
  Real Pc = four_pi_G*SQR(rhoc*rsurf)/SQR(PI)/2; // central pressure
  Real xi,x1,x2,x3,den,prs;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        x1 = pcoord->x1v(i);
        x2 = pcoord->x2v(j);
        x3 = pcoord->x3v(k);
        if (COORDINATE_SYSTEM=="cartesian")
          xi = PI*std::sqrt(SQR(x1-x1c)+SQR(x2-x2c)+SQR(x3-x3c))/rsurf;
        else if (COORDINATE_SYSTEM=="cylindrical")
          xi = PI*std::sqrt(SQR(x1)+SQR(x1c)-2*x1*x1c*cos(x2-x2c)+SQR(x3-x3c))/rsurf;
        if (xi < PI) {
          den = rhoc*sin(xi)/xi;
          prs = Pc*SQR(sin(xi)/xi);
        } else {
          den = damb;
          prs = pamb;
        }
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = den*v1;
        phydro->u(IM2,k,j,i) = den*v2;
        phydro->u(IM3,k,j,i) = den*v3;
        phydro->u(IEN,k,j,i) = prs/gm1 + 0.5*den*(SQR(v1)+SQR(v2)+SQR(v3));
      }
    }
  }
  return;
}
