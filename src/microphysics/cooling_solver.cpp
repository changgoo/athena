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
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"          // ParameterInput
#include "cooling.hpp"
#include "units.hpp"

// static free functions to be enrolled
//========================================================================================
//! \fn Real CoolingTimeStep(MeshBlock *pmb)
//! \brief Function to calculate the timestep required to resolve cooling
//!        tcool = 3/2 P/Edot_cool
//========================================================================================
Real CoolingSolver::CoolingTimeStep(MeshBlock *pmb) {
  Real min_dt=HUGE_NUMBER;
  CoolingFunctionBase *pcf = pmb->pcool->pcf;
  Units *punit = pmb->pcool->punit;
  Real cfl_cool = pmb->pcool->cfl_cool;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real press = pmb->phydro->w(IPR,k,j,i);
        Real rho = pmb->phydro->w(IDN,k,j,i);
        Real press_floor = rho*pcf->Get_Tfloor()/punit->Temperature;
        press = std::max(press,press_floor);
        Real dtcool = cfl_cool*std::abs(pcf->NetCoolingTime(rho,press));
        min_dt = std::min(min_dt, dtcool);
      }
    }
  }
  return min_dt;
}

//========================================================================================
//! \fn void CoolingSourceTerm(MeshBlock *pmb, const Real t, const Real dt,
//!       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
//!       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
//!       AthenaArray<Real> &cons_scalar)
//! \brief enrollable function for cooling source term
//!        must use prim to set cons
//========================================================================================
void CoolingSolver::CoolingSourceTerm(MeshBlock *pmb, const Real t, const Real dt,
       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
       AthenaArray<Real> &cons_scalar) {
  // Extract indices
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  CoolingSolver *pcool = pmb->pcool;
  CoolingFunctionBase *pcf = pmb->pcool->pcf;
  Units *punit = pmb->pcool->punit;

  Real dt_mhd = pmb->pmy_mesh->dt;
  Real temp_floor = pcf->Get_Tfloor(); // temperature floor
  Real gm1 = pcf->gamma_adi-1; // gamma-1

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        // Extract rho and P from previous steps
        const Real press = prim(IPR,k,j,i);
        const Real rho = prim(IDN,k,j,i);

        Real delta_press=0.0, delta_press_floor=0.0;
        // apply floor before solving the cooling
        Real press_floor = rho*temp_floor/punit->Temperature;
        if (press < press_floor) delta_press_floor = press_floor-press;
        Real press_before = std::max(press,press_floor);

        // solve cooling
        Real press_after = pcool->Solver(press_before,rho,dt);
        delta_press = press_after-press_before; // save difference due to cooling

        // apply floor after solving the cooling
        if (press_after < press_floor) delta_press_floor += press_floor-press_after;
        Real press_next = std::max(press_after,press_floor);

        if (pcool->bookkeeping) {
          if (pmb->pmy_mesh->time_integrator == "vl2") {
            pcool->edot(k,j,i) = delta_press/gm1/dt;
            pcool->edot_floor(k,j,i) = delta_press_floor/gm1/dt;
          } else if (pmb->pmy_mesh->time_integrator == "rk2") {
            pcool->edot(k,j,i) += 0.5*delta_press/gm1/dt;
            pcool->edot_floor(k,j,i) += 0.5*delta_press_floor/gm1/dt;
          }
        }

        Real delta_e = (press_next-press)/gm1;
        // change internal energy
        cons(IEN,k,j,i) += delta_e;
      }
    }
  }

  return;
}

//========================================================================================
//! \fn CoolingSolver::CoolingSolver(MeshBlock *pmb, ParameterInput *pin)
//! \brief ctor of the base class for cooling solver
//! \note Read parameters from "cooling" block in the input file
//========================================================================================
CoolingSolver::CoolingSolver(MeshBlock *pmb, ParameterInput *pin) :
  cfl_cool(pin->GetReal("cooling", "cfl_cool")),
  cfl_op_cool(pin->GetOrAddReal("cooling","cfl_op_cool",-1)),
  op_flag(false), bookkeeping(false),
  coolftn(pin->GetOrAddString("cooling", "coolftn", "tigress")),
  cooling(pin->GetOrAddString("cooling", "cooling", "none")),
  solver(pin->GetOrAddString("cooling","solver","forward_euler")),
  uov_idx_(-1), nsub_max_(pin->GetOrAddInteger("cooling","nsub_max",-1)) {
  if (cooling.compare("op_split") == 0) {
    op_flag = true;
    if (nsub_max_ == -1) {
      nsub_max_ = static_cast<int>(CoolingSolver::cfl_cool/CoolingSolver::cfl_op_cool)*10;
      std::cout << "[CoolingSolver] nsub_max_ is set to " << nsub_max_ << std::endl;
    }
    // error if op cooling solver is used but cfl_op_cool > 1
    if (cfl_op_cool > 1) {
      std::stringstream msg;
      msg << "### FATAL ERROR in CoolingSolver" << std::endl
          << "Cooling will be solved by operator split method but cfl_op_cool = "
          << CoolingSolver::cfl_op_cool << " > 1" << std::endl;
      ATHENA_ERROR(msg);
    }
  } else if (cooling.compare("enroll") == 0) {
    if (cfl_cool > 1) {
      std::stringstream msg;
      msg << "### FATAL ERROR in CoolingSolver" << std::endl
          << "Cooling will be enrolled but cfl_cool = "
          << CoolingSolver::cfl_cool << " > 1" << std::endl;
      ATHENA_ERROR(msg);
    }
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in CoolingSolver" << std::endl
        << "cooling/cooling must be one of [none, op_split, enroll], but "
        << cooling << " is given" << std::endl;
    ATHENA_ERROR(msg);
  }

  if (coolftn.compare("tigress") == 0) {
    pcf = new TigressClassic(pin);
    if (Globals::my_rank == 0)
      std::cout << "[CoolingSoler] Cooling function is set to TigressClassic"
                << std::endl;
  } else if (coolftn.compare("plf") == 0) {
    pcf = new PiecewiseLinearFits(pin);
    if (Globals::my_rank == 0)
      std::cout << "[CoolingSoler] Cooling function is set to PiecewiseLinearFits"
                << std::endl;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in CoolingSolver" << std::endl
        << "coolftn = " << coolftn.c_str() << " is not supported" << std::endl;
    ATHENA_ERROR(msg);
  }

  punit = pcf->punit;

  // set function pointer
  if (solver.compare("forward_euler") == 0) {
    if (Globals::my_rank == 0)
      std::cout << "[CoolingSoler] Solver is set to ForwardEuler" << std::endl;
  } else {
    std::cout << "cooling/solver must be one of [forward_euler], but "
              << solver << " is given" << std::endl;
  }
}

//========================================================================================
//! \fn CoolingSolver::Solver(Real press, Real rho, Real dt)
//! \brief solve one cell. potentially extended with different methods
//========================================================================================
Real CoolingSolver::Solver(Real press, Real rho, Real dt) {
  // maybe added some conditionals for different solvers
  // if (solver.compare("forward_euler") == 0)
  return press*(1-dt/pcf->CoolingTime(rho, press));
}

//========================================================================================
//! \fn CoolingSolver::OperatorSplitSolver(MeshBlock *pmb, ParameterInput *pin)
//! \brief Update cooling source term in a operatoer split manner using subcycling
//========================================================================================
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
  Real temp_floor = pcf->Get_Tfloor(); // temperature floor
  Real gm1 = pcf->gamma_adi-1; // gamma-1

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

        Real delta_press=0.0, delta_press_floor=0.0;
        // apply floor before solving the cooling
        Real press_floor = w_d*temp_floor/punit->Temperature;
        if (w_p < press_floor) delta_press_floor = press_floor-w_p;
        Real press_before = std::max(w_p,press_floor);

        // solve cooling
        Real press_after = CoolingExplicitSubcycling(dt_mhd,press_before,w_d);
        delta_press = press_after-press_before; // save diference due to cooling

        // apply floor after solving the cooling
        if (press_after < press_floor) delta_press_floor += press_floor-press_after;
        Real press_next = std::max(press_after,press_floor);

        if (bookkeeping) {
          edot(k,j,i) = delta_press/gm1/dt_mhd;
          edot_floor(k,j,i) = delta_press_floor/gm1/dt_mhd;
        }

        // apply floor if cooled too much
        Real u_after = press_next/gm1;

        // change internal energy
        u_e = u_after + e_non_thermal;
        w_p = press_next;
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
  Real dt_net, dt_sub;
  int icount = 0;
  while ((icount<nsub_max_) && (tnow<tend)) {
    dt_net = std::abs(pcf->CoolingTime(rho, press))*cfl_op_cool;
    dt_sub = std::min(std::min(tend,dt_net),tleft);
    press = Solver(press,rho,dt_sub);

    tnow += dt_sub;
    tleft = tend-tnow;
    icount++;
  }

  if (tnow < tend)
    std::cout << "Too many substeps required: "
              << " nsub_max_ = " << nsub_max_
              << " nsub_max_needed_ = " << tend/dt_net
              << " tnow = " << tnow
              << " tend = " << tend
              << std::endl;
  return press;
}
