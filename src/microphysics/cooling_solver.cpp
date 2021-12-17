//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cooling_function.cpp
//! \brief prototypes of various cooling functions

// C headers

// C++ headers
#include <iostream>   // cout, endl
#include <sstream>    // stringstream

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"          // ParameterInput
#include "cooling.hpp"
#include "units.hpp"

CoolingFunctionBase* CoolingSolver::pcf = nullptr;
Units* CoolingSolver::punit = nullptr;
AthenaArray<Real> CoolingSolver::edot, CoolingSolver::edot_floor;
Real CoolingSolver::cfl_cool, CoolingSolver::cfl_op_cool;

//========================================================================================
//! \fn CoolingFunctionBase::CoolingFunctionBase(ParameterInput *pin)
//! \brief ctor of the base class for cooling function
//! \note Read parameters from "cooling" block in the input file
//========================================================================================
CoolingSolver::CoolingSolver(ParameterInput *pin) :
  nsub_max(pin->GetOrAddInteger("cooling","nsub_max",-1)) {
  CoolingSolver::cfl_cool=pin->GetReal("cooling", "cfl_cool"); // min dt_hydro/dt_cool
  CoolingSolver::cfl_op_cool=pin->GetOrAddReal("cooling","cfl_op_cool",-1);

  std::string coolftn = pin->GetOrAddString("cooling", "coolftn", "tigress");

  if (coolftn.compare("tigress") == 0) {
    CoolingSolver::pcf = new TigressClassic(pin);
    std::cout << "Cooling function is set to TigressClassic" << std::endl;
  } else if (coolftn.compare("plf") ==0) {
    CoolingSolver::pcf = new PiecewiseLinearFits(pin);
    std::cout << "Cooling function is set to PiecewiseLinearFits" << std::endl;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in CoolingSolver" << std::endl
        << "coolftn = " << coolftn.c_str() << " is not supported" << std::endl;
    ATHENA_ERROR(msg);
  }

  CoolingSolver::punit = CoolingSolver::pcf->punit;

  op_flag = CoolingSolver::cfl_op_cool > 0 ? true: false;
  if (op_flag && (nsub_max == -1)) {
    nsub_max = static_cast<int>(CoolingSolver::cfl_cool/CoolingSolver::cfl_op_cool*10);
    std::cout << "nsub_max is set to " << nsub_max << std::endl;
  }

  // error if cooling solver is enrolled but cfl_cool > 1
  if ((CoolingSolver::cfl_cool > 1) && (!op_flag)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in CoolingSolver" << std::endl
        << "Cooling will be enrolled but cfl_cool = "
        << CoolingSolver::cfl_cool << " > 1" << std::endl;
    ATHENA_ERROR(msg);
  }

  // error if op cooling solver is used but cfl_op_cool > 1
  if ((CoolingSolver::cfl_op_cool > 1) && (op_flag)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in CoolingSolver" << std::endl
        << "Cooling will be solved by operator split method but cfl_op_cool = "
        << CoolingSolver::cfl_op_cool << " > 1" << std::endl;
    ATHENA_ERROR(msg);
  }
}


void CoolingSolver::InitEdotArray(MeshBlock *pmb) {
  CoolingSolver::edot.NewAthenaArray(pmb->ncells3,pmb->ncells2,pmb->ncells1);
}

void CoolingSolver::InitEdotArray(AthenaArray<Real> uov, int index) {
  CoolingSolver::edot.InitWithShallowSlice(uov,4,index,1);
}

void CoolingSolver::InitEdotFloorArray(MeshBlock *pmb) {
  CoolingSolver::edot_floor.NewAthenaArray(pmb->ncells3,pmb->ncells2,pmb->ncells1);
}

void CoolingSolver::InitEdotFloorArray(AthenaArray<Real> uov, int index) {
  CoolingSolver::edot_floor.InitWithShallowSlice(uov,4,index,1);
}

//========================================================================================
//! \fn Real CoolingTimeStep(MeshBlock *pmb)
//! \brief Function to calculate the timestep required to resolve cooling
//!        tcool = 3/2 P/Edot_cool
//========================================================================================
Real CoolingSolver::CoolingTimeStep(MeshBlock *pmb) {
  Real min_dt=HUGE_NUMBER;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real press = pmb->phydro->w(IPR,k,j,i);
        Real rho = pmb->phydro->w(IDN,k,j,i);
        Real dtcool = CoolingSolver::cfl_cool*
          std::abs(CoolingSolver::pcf->CoolingTime(rho,press));
        min_dt = std::min(min_dt, dtcool);
      }
    }
  }
  return min_dt;
}

void CoolingSolver::OperatorSplitSolver(MeshBlock *pmb) {
  // boundary comm. will not be called.
  // need to solve cooling in the ghost zones
  // Prepare index bounds
  int il = pmb->is - NGHOST;
  int iu = pmb->ie + NGHOST;
  int jl = pmb->js;
  int ju = pmb->je;
  if (pmb->pmy_mesh->f2) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = pmb->ks;
  int ku = pmb->ke;
  if (pmb->pmy_mesh->f3) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }

  Real dt_mhd = pmb->pmy_mesh->dt;
  Real temp_floor = CoolingSolver::pcf->Get_Tfloor(); // temperature floor
  Real gm1 = CoolingSolver::pcf->gamma_adi-1; // gamma-1

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        // both u and w are updated by integrator
        Real& u_d  = pmb->phydro->u(IDN,k,j,i);
        Real& u_e  = pmb->phydro->u(IEN,k,j,i);

        Real& w_d  = pmb->phydro->w(IDN,k,j,i);
        Real& w_p  = pmb->phydro->w(IPR,k,j,i);
        // find non-thermal part of energy to keep it the same
        Real e_non_thermal = u_e - w_p/gm1;

        Real press_floor = w_d*temp_floor/CoolingSolver::punit->Temperature;
        Real press_next = CoolingExplicitSubcycling(dt_mhd,w_p,w_d);

        Real delta_press, delta_press_floor;
        if (press_next < press_floor) {
          // cooled too much; apply floor
          // store both cooling loss and artificial heating
          delta_press = press_next-w_p;
          delta_press_floor = press_floor-press_next;
        } else {
          // normal cooling without floor;
          // store cooling loss only
          delta_press = press_next-w_p;
          delta_press_floor = 0.;
        }
        CoolingSolver::edot(k,j,i) = delta_press/gm1/dt_mhd;
        CoolingSolver::edot_floor(k,j,i) = delta_press_floor/gm1/dt_mhd;

        // apply floor if cooled too much
        Real press_after = std::max(press_next,press_floor);
        Real u_after = press_after/gm1;

        // change internal energy
        u_e = u_after + e_non_thermal;
        w_p = press_after;
      }
    }
  }
}

//========================================================================================
//! \fn Real CoolingExplicitSubcycling(Real tend, Real P, const Real rho)
//! \brief explicit cooling solver from 0 to tend with subcycling
//========================================================================================
Real CoolingSolver::CoolingExplicitSubcycling(Real tend, Real press, const Real rho) {
  Real tnow = 0., tleft = tend;
  int icount = 0;
  while ((icount<nsub_max) && (tnow<tend)) {
    Real dt_net = CoolingSolver::pcf->NetCoolingTime(rho, press);
    dt_net *= CoolingSolver::cfl_op_cool;
    Real dt_sub = std::min(std::min(tend,dt_net),tleft);
    press = press*(1-dt_sub/CoolingSolver::pcf->CoolingTime(rho, press));

    tnow += dt_sub;
    tleft = tend-tnow;
    icount++;
  }

  if (tnow < tend)
    std::cout << "Too many substeps required: tnow = " << tnow
              << " tend = " << tend << std::endl;
  return press;
}

//========================================================================================
//! \fn void CoolingEuler(MeshBlock *pmb, const Real t, const Real dt,
//!       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
//!       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
//!       AthenaArray<Real> &cons_scalar)
//! \brief enrollable function for cooling source term
//!        must use prim to set cons
//========================================================================================
void CoolingSolver::CoolingEuler(MeshBlock *pmb, const Real t, const Real dt,
       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
       AthenaArray<Real> &cons_scalar) {
  // Extract indices
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  Real dt_mhd = pmb->pmy_mesh->dt;
  Real temp_floor = CoolingSolver::pcf->Get_Tfloor(); // temperature floor
  Real gm1 = CoolingSolver::pcf->gamma_adi-1; // gamma-1

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        // Extract rho and P from previous steps
        const Real press = prim(IPR,k,j,i);
        const Real rho = prim(IDN,k,j,i);

        // pressure and interanl energy update done in cod units
        Real press_floor = press*temp_floor/CoolingSolver::punit->Temperature;
        Real press_next = press*(1 - dt/CoolingSolver::pcf->CoolingTime(rho, press));

        Real delta_press, delta_press_floor;
        if (press_next < press_floor) {
          // cooled too much; apply floor
          // store both cooling loss and artificial heating
          delta_press = press_next-press;
          delta_press_floor = press_floor-press_next;
        } else {
          // normal cooling without floor;
          // store cooling loss only
          delta_press = press_next-press;
          delta_press_floor = 0.;
        }
        CoolingSolver::edot(k,j,i) = delta_press/gm1/dt;
        CoolingSolver::edot_floor(k,j,i) = delta_press_floor/gm1/dt;

        Real delta_e = (delta_press+delta_press_floor)/gm1;
        // change internal energy
        cons(IEN,k,j,i) += delta_e;
      }
    }
  }

  return;
}

void CoolingSolver::CalculateTotalCoolingRate(MeshBlock *pmb, Real dt) {
  // Extract indices
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;
  // sum up cooling only done in the active cells
  AthenaArray<Real> vol(pmb->ncells1);
  Real delta_e_block = 0.0, delta_ef_block = 0.0;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
#pragma omp simd reduction(+:delta_e_block,delta_ef_block)
      for (int i = is; i <= ie; ++i) {
        delta_e_block += CoolingSolver::edot(k,j,i)*dt*vol(i);
        delta_ef_block += CoolingSolver::edot_floor(k,j,i)*dt*vol(i);
      }
    }
  }
  // add cooling and ceiling to hist outputs
  pmb->ruser_meshblock_data[0](0) += delta_e_block;
  pmb->ruser_meshblock_data[0](1) += delta_ef_block;
}
