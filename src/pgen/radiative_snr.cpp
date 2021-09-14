//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file radiative_snr.cpp
//! \brief Problem generator for radiative snr
//========================================================================================

// C++ headers
#include <algorithm>  // min()
#include <cmath>      // abs(), pow(), sqrt()
#include <fstream>    // ofstream
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string

// Athena++ headers
#include "../athena.hpp"                   // macros, enums, declarations
#include "../athena_arrays.hpp"            // AthenaArray
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../field/field.hpp"              // Field
#include "../globals.hpp"                  // Globals
#include "../hydro/hydro.hpp"              // Hydro
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"          // ParameterInput
#include "../utils/cooling_function.hpp"   // Cooling function namespace
#include "../utils/units.hpp"              // Units, Constants

// Global variables ---
// Pointer to unit class. This is now attached to the Cooling class
Units *punit;
// Pointer to Cooling function class,
// will be set to specific function depending on the input parameter (cooling/coolftn).
CoolingFunctionBase *pcool;

// explicit cooling solver using RK4 method for integration
// slightly modified to update T*(n/n_H) rather than T itself
Real CoolingLosses(MeshBlock *pmb, int iout);
static Real cooling_timestep(MeshBlock *pmb);

// calculate tcool = e/L(rho, P)
static Real tcool(CoolingFunctionBase *pcool, const Real rho, const Real Press);
static Real dtnet(CoolingFunctionBase *pcool, const Real rho, const Real Press);

// Utility functions for debugging
void PrintCoolingFunction(CoolingFunctionBase *pcool, std::string coolftn);
void PrintParameters(CoolingFunctionBase *pcool, const Real rho, const Real Press);

Real cfl_op_cool=-1;
//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  // initialize cooling function
  // currently, two cooling functions supported (tigress, plf)
  std::string coolftn = pin->GetOrAddString("cooling", "coolftn", "tigress");

  if (coolftn.compare("tigress") == 0) {
    pcool = new TigressClassic(pin);
    std::cout << "Cooling function is set to TigressClassic" << std::endl;
  } else if (coolftn.compare("plf") ==0) {
    pcool = new PiecewiseLinearFits(pin);
    std::cout << "Cooling function is set to PiecewiseLinearFits" << std::endl;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in ProblemGenerator" << std::endl
        << "coolftn = " << coolftn.c_str() << " is not supported" << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
  }

  // shorthand for unit class
  // not unit class is initialized within cooling function constructor
  // to use appropreate mu and muH
  punit = pcool->punit;


  // show some values for sanity check.
  if (Globals::my_rank == 0) {
    // dump cooling function used in ascii format to e.g., tigress_coolftn.txt
    PrintCoolingFunction(pcool,coolftn);

    // print out units and constants in code units
    punit->PrintCodeUnits();
    punit->PrintConstantsInCodeUnits();
  }

  // use operator split cooling solver
  cfl_op_cool=pin->GetOrAddReal("cooling","cfl_op_cool",0.1);

  // Enroll timestep so that dt <= min t_cool
  EnrollUserTimeStepFunction(cooling_timestep);

  // Enroll user-defined functions
  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, CoolingLosses, "e_cool");
}

//========================================================================================
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in MeshBlock class.  Can also be
//! used to initialize variables which are global to other functions in this file.
//! Called in MeshBlock constructor before ProblemGenerator.
//========================================================================================
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // Allocate storage for keeping track of cooling
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(1);
  ruser_meshblock_data[0](0) = 0.0; // e_cool

  // Set output variables
  AllocateUserOutputVariables(1);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Should be used to set initial conditions.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }

  Real rho_0   = pin->GetReal("problem", "rho_0"); // measured in m_p muH cm^-3
  Real pgas_0  = pin->GetReal("problem", "pgas_0"); // measured in kB K cm^-3

  // below is for sanity check. Uncomment if needed
  Real mu = pcool->Get_mu(rho_0, pgas_0/pcool->to_pok);
  Real muH = pcool->Get_muH();
  Real T = pgas_0/rho_0*(mu/muH);
  //
  std::cout << "============== Check Initialization ===============" << std::endl
            << " Input (nH, P/k, T) in cgs = " << rho_0 << " " << pgas_0
            << " " << T << std::endl
            << "  mu = " << mu << " mu(punit) = " << punit->mu
            << " muH = " << muH << std::endl;
            // << " tcool = " << tcool(pcool, T, rho_0) << std::endl;

  rho_0 /= pcool->to_nH; // to code units
  pgas_0 /= pcool->to_pok; // to code units

  PrintParameters(pcool,rho_0,pgas_0);
  T = pcool->GetTemperature(rho_0,pgas_0);
  Real nH = rho_0*pcool->to_nH;
  std::cout << "  Tempearture = " << T << std::endl;
  std::cout << " tcool = " << tcool(pcool, rho_0, pgas_0) << std::endl;
  std::cout << " sound speed = " << std::sqrt(pgas_0/rho_0) << std::endl;

  // Initialize primitive values
  Real my_vol = 0;
  Real r_sn = pin->GetReal("problem","r_sn");
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->w(IDN,k,j,i) = rho_0;
        phydro->w(IPR,k,j,i) = pgas_0;
        phydro->w(IVX,k,j,i) = 0.0;
        phydro->w(IVY,k,j,i) = 0.0;
        phydro->w(IVZ,k,j,i) = 0.0;
        Real r = std::sqrt(SQR(pcoord->x1v(i))
                          +SQR(pcoord->x2v(j))
                          +SQR(pcoord->x3v(k)));
        if (r<r_sn) my_vol += pcoord->GetCellVolume(k,j,i);
      }
    }
  }

  // calculate total feedback region volume
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &my_vol, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // get pressure fron SNe in the code unit
  Real Esn = pin->GetOrAddReal("problem","Esn",1.0)*pcool->punit->Bethe_in_code;
  Real Psn = Esn/my_vol*(peos->GetGamma()-1);

  // add the SN energy
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        Real r = std::sqrt(SQR(pcoord->x1v(i))
                          +SQR(pcoord->x2v(j))
                          +SQR(pcoord->x3v(k)));
        if (r<r_sn)
          phydro->w(IPR,k,j,i) += Psn;
      }
    }
  }

  // Initialize conserved values
  AthenaArray<Real> b;
  peos->PrimitiveToConserved(phydro->w, b, phydro->u, pcoord,
       il, iu, jl, ju, kl, ku);
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================

void MeshBlock::UserWorkInLoop() {
  if (cfl_op_cool < 0) return; // no operator split cooling

  Real dt_mhd = pmy_mesh->dt*punit->Time;

  AthenaArray<Real> edot;
  edot.InitWithShallowSlice(user_out_var, 4, 0, 1);
  Real delta_e_block = 0.0;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        // both u and w are updated by integrator
        Real& u_d  = phydro->u(IDN,k,j,i);
        Real& u_e  = phydro->u(IEN,k,j,i);

        Real& w_d  = phydro->w(IDN,k,j,i);
        Real& w_p  = phydro->w(IPR,k,j,i);
        // find non-thermal part of energy to keep it the same
        Real e_non_thermal = u_e - w_p/(pcool->gamma_adi-1.0);
        // check bad cell
        if (w_d < 0)
          std::cout << " density is bad: d("
                    << k << "," << j << "," << i << ") = "
                    << w_d << std::endl;
        if (w_p < 0)
          std::cout << " pressure is bad: d("
                    << k << "," << j << "," << i << ") = "
                    << w_p << std::endl;

        // set the initial conditions
        Real P_before = w_p; // store original P
        Real rho_before = w_d; // store original d
        Real nH_before = w_d*pcool->to_nH; // store original nH
        Real T_before = P_before*pcool->to_pok/nH_before;
        Real tnow = 0., tleft = dt_mhd;
        Real P_next, T_next; // declare vars outside while-scope
        int nsub=0;
        while (tnow < dt_mhd) {
          Real dt_net = cfl_op_cool*dtnet(pcool, rho_before, P_before);
          Real dt_sub = std::min(std::min(dt_mhd,dt_net),tleft);

          T_next = T_before*(1-dt_sub/tcool(pcool, rho_before, P_before));
          P_next = nH_before*T_next/pcool->to_pok;

          tnow += dt_sub;
          tleft = dt_mhd-tnow;

          P_before = P_next;
          T_before = P_before*pcool->to_pok/nH_before;
        }

        // dont cool below cooling floor and find new internal thermal energy
        Real T_floor = pcool->Get_Tfloor();
        Real P_floor = T_floor*nH_before/pcool->to_pok;

        Real P_after = std::max(P_next,P_floor);
        Real u_after = P_after/(pcool->gamma_adi-1.0);

        // change internal energy
        u_e = u_after + e_non_thermal;
        w_p = P_after;

        // store edot (code units)
        Real delta_e = (P_after-P_before)/(pcool->gamma_adi-1.0);
        edot(k,j,i) = delta_e/pmy_mesh->dt;

        delta_e_block += delta_e;
      }
    }
  }
  // add cooling and ceiling to hist outputs
  ruser_meshblock_data[0](0) += delta_e_block;

  return;
}

//========================================================================================
//! \fn Real cooling_timestep(MeshBlock *pmb)
//! \brief Function to calculate the timestep required to resolve cooling
//!        tcool = 3/2 P/Edot_cool
//========================================================================================
static Real cooling_timestep(MeshBlock *pmb) {
  Real min_dt=1.0e10;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real Press = pmb->phydro->w(IPR,k,j,i);
        Real rho = pmb->phydro->w(IDN,k,j,i);
        Real T_before = pcool->GetTemperature(rho, Press);
        // Real nH = rho*pcool->to_nH;
        Real T_floor = pcool->Get_Tfloor();
        if (T_before > 1.01 * T_floor) {
          Real dtcool = pcool->cfl_cool*std::abs(dtnet(pcool,rho,Press))
                       /pcool->punit->Time;
          min_dt = std::min(min_dt, dtcool);
        }
        // min_dt = std::max(dt_cutoff,min_dt);
      }
    }
  }
  return min_dt;
}

//========================================================================================
//! \fn static Real tcool(CoolingFunctionBase *pcool, const Real rho, const Real Press)
//! \brief tcool = e / (n^2*Cool - n*heat)
//! \note
//! - input rho and P are in code Units
//! - output tcool is in second
//========================================================================================
static Real tcool(CoolingFunctionBase *pc, const Real rho, const Real Press) {
  Real nH = rho*pc->to_nH;
  Real cool = nH*nH*pc->Lambda_T(rho, Press);
  Real heat = nH*pc->Gamma_T(rho, Press);
  Real eint = Press*pc->punit->Pressure/(pc->gamma_adi-1);
  Real tcool = eint/(cool - heat);
  return tcool;
}

//========================================================================================
//! \fn static Real dtnet(CoolingFunctionBase *pcool, const Real rho, const Real Press)
//! \brief dtnet = e / (n^2*Cool + n*heat)
//! \note
//! - input rho and P are in code Units
//! - output dtnet is in second
//========================================================================================
static Real dtnet(CoolingFunctionBase *pc, const Real rho, const Real Press) {
  Real nH = rho*pc->to_nH;
  Real cool = nH*nH*pc->Lambda_T(rho, Press);
  Real heat = nH*pc->Gamma_T(rho, Press);
  Real eint = Press*pc->punit->Pressure/(pc->gamma_adi-1);
  Real tcool = eint/(cool + heat);
  return tcool;
}

//========================================================================================
//! \fn Real CoolingLosses(MeshBlock *pmb, int iout)
//! \brief Cooling losses for history variable
//!        return sum of all energy losses due to different cooling mechanisms and
//!        resets time-integrated values to 0 for the next step
//========================================================================================
Real CoolingLosses(MeshBlock *pmb, int iout) {
  Real delta_e = pmb->ruser_meshblock_data[0](iout);
  pmb->ruser_meshblock_data[0](iout) = 0.0;
  return delta_e;
}

//========================================================================================
//! \fn void PrintCoolingFunction(CoolingFunctionBase *pcool,std::string coolftn)
//! \brief private function to check cooling and heating functions
//========================================================================================
void PrintCoolingFunction(CoolingFunctionBase *pcool,std::string coolftn) {
  Real Pok = 3.e3;
  std::string coolfilename(coolftn);
  coolfilename.append("_coolftn.txt");
  std::ofstream coolfile (coolfilename.c_str());
  coolfile << "#rho,Press,Temp,cool,heat,tcool" << "\n";

  for (int i=0; i<1000; ++i) {
    Real logn = 5.0*((static_cast<Real>(i)/500.)-1.0)-2; // logn = -7 ~ 3
    Real rho = std::pow(10,logn);
    Real Press = Pok/pcool->to_pok;
    Real Temp = pcool->GetTemperature(rho, Press);
    Real cool = pcool->Lambda_T(rho,Press);
    Real heat = pcool->Gamma_T(rho,Press);
    Real t_cool = tcool(pcool,rho,Press);
    coolfile << rho << "," << Press << "," << Temp << ","
             << cool << "," << heat << "," << t_cool << "\n";
  }
}

//========================================================================================
//! \fn void PrintParameters(CoolingFunctionBase *pcool, const Real rho,
//!       const Real Press)
//! \brief print function for sanity check
//========================================================================================
void PrintParameters(CoolingFunctionBase *pcool, const Real rho, const Real Press) {
  Real Temp_K = pcool->GetTemperature(rho, Press);
  Real nH = rho*pcool->to_nH;
  Real pok = Press*pcool->to_pok;
  Real cool = pcool->Lambda_T(rho,Press);
  Real heat = pcool->Gamma_T(rho,Press);
  Real netcool = nH*(nH*cool-heat);
  Real mu = pcool->Get_mu(rho,Press);
  Real muH = pcool->Get_muH();

  std::cout << "============== Cooling Parameters =============" << std::endl
            << " Input (rho, P) in code = " << rho << " " << Press << std::endl
            << " Converted (nH, P/k, T) = " << nH << " " << pok
            << " " << Temp_K << std::endl
            << "  mu = " << mu << "  muH = " << muH << std::endl
            << "  cool = " << cool << "  heat = " << heat
            << "  netcool = " << netcool << std::endl;
}
