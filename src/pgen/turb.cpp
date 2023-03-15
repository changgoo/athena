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
#include "../fft/perturbation.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// user function
void CollectCounters(Mesh *pm);

int turb_flag;
Real rho0=1.0;
TurbulenceDriver *ptrbd;
//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = pin->GetReal("gravity","four_pi_G");
    SetFourPiG(four_pi_G);
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IDN,k,j,i) = rho0;

        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = 1.0;
        }
      }
    }
  }
}

//========================================================================================
//! \fn void Mesh::PostInitialize(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::PostInitialize(int res_flag, ParameterInput *pin) {
  turb_flag = pin->GetOrAddInteger("problem","turb_flag",0);
  if (turb_flag != 0) {
    ptrbd = new TurbulenceDriver(this, pin);

    if (res_flag == 0) {
      if (turb_flag == 3) {
        MeshBlock *pmb = my_blocks(0);
        Real dx = pmb->pcoord->dx1v(0);
        // get rough estimate of dt
        dt = std::cbrt(1.5*rho0/ptrbd->dedt*SQR(dx));
      }
      ptrbd->Driving();
      if (turb_flag == 1) delete ptrbd;
    }
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief
//========================================================================================
void Mesh::UserWorkInLoop() {
  if (turb_flag > 1) ptrbd->Driving(); // driven turbulence
  // check number of bad cells
  CollectCounters(this);
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop()
//  \brief
//========================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  if (turb_flag > 1) delete ptrbd; // driven turbulence
}

//========================================================================================
//! \fn void CollectCounters(Mesh *pm)
//! \brief collect counters
//========================================================================================
void CollectCounters(Mesh *pm) {
  int nbad_d=0, nbad_p=0;

  // summing up over meshblocks within the rank
  for (int b=0; b<pm->nblocal; ++b) {
    MeshBlock *pmb = pm->my_blocks(b);
    nbad_d += pmb->nbad_d;
    nbad_p += pmb->nbad_p;
  }

  // calculate total feedback region volume
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &nbad_d, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &nbad_p, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (Globals::my_rank == 0) {
    if (nbad_p > 0) std::cerr << nbad_p << " cells had negative pressure" << std::endl;
    if (nbad_d > 0) std::cerr << nbad_d << " cells had negative density" << std::endl;
  }
}
