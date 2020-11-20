//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cooling_rk4.c
//! \brief Problem generator for cooling test using RK4 integration
//========================================================================================

// C++ headers
#include <algorithm>  // min()
#include <cmath>      // abs(), pow(), sqrt()
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string
#include <fstream>

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
#include "../utils/units.hpp"              // Units, Constants
#include "../utils/cooling_function.hpp"   // Cooling function namespace


// Global variables ---
// Pointer to unit class. This is now attached to the Cooling class
Units *punit;
// Pointer to Cooling function class,
// will be set to specific function depending on the input parameter (cooling/coolftn).
CoolingFunctionBase *pcool;

// explicit cooling solver using RK4 method for integration
// slightly modified to update T*(n/n_H) rather than T itself
void Cooling(MeshBlock *pmb, const Real t, const Real dt,
       const AthenaArray<Real> &prim,const AthenaArray<Real> &prim_scalar,
       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
       AthenaArray<Real> &cons_scalar);
Real CoolingLosses(MeshBlock *pmb, int iout);
static Real cooling_timestep(MeshBlock *pmb);

// calculate tcool = e/L(rho, P)
static Real tcool(CoolingFunctionBase *pcool, const Real rho, const Real Press);

// Utility functions for debugging
void PrintCoolingFunction(CoolingFunctionBase *pcool, std::string coolftn);
void PrintParameters(CoolingFunctionBase *pcool, const Real rho, const Real Press);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  // turb_flag is initialzed in the Mesh constructor to 0 by default;
  // turb_flag = 1 for decaying turbulence
  // turb_flag = 2 for driven turbulence
  // turb_flag = 3 for density perturbations
  turb_flag = pin->GetInteger("problem","turb_flag");
  if (turb_flag != 0) {
#ifndef FFT
    std::stringstream msg;
    msg << "### FATAL ERROR in ProblemGenerator " << std::endl
        << "non zero Turbulence flag is set without FFT!" << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
#endif
  }

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

  // shorhand for unit class
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

  // Enroll source function
  EnrollUserExplicitSourceFunction(Cooling);

  // Enroll timestep so that dt <= min t_cool
  EnrollUserTimeStepFunction(cooling_timestep);

  // Enroll user-defined functions
  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, CoolingLosses, "e_cool");
  EnrollUserHistoryOutput(1, CoolingLosses, "e_ceil");
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
  ruser_meshblock_data[0].NewAthenaArray(2);
  ruser_meshblock_data[0](0) = 0.0; // e_cool
  ruser_meshblock_data[0](1) = 0.0; // e_ceil

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
  // Real mu = pcool->Get_mu(rho_0, pgas_0/pcool->to_pok);
  // Real muH = pcool->Get_muH();
  // Real T = pgas_0/rho_0*(mu/muH);
  //
  // std::cout << "============== Check Initialization ===============" << std::endl
  //           << " Input (nH, P/k, T) in cgs = " << rho_0 << " " << pgas_0
  //           << " " << T << std::endl
  //           << "  mu = " << mu << " mu(punit) = " << punit->mu
  //           << " muH = " << muH << std::endl;
  //           // << " tcool = " << tcool(pcool, T, rho_0) << std::endl;

  rho_0 /= pcool->to_nH; // to code units
  pgas_0 /= pcool->to_pok; // to code units

  // PrintParameters(pcool,rho_0,pgas_0);
  // T = pcool->GetTemperature(rho_0,pgas_0);
  // Real nH = rho_0*pcool->to_nH;
  // std::cout << "  Tempearture = " << T << std::endl;
  // std::cout << " tcool = " << tcool(pcool, rho_0, pgas_0) << std::endl;
  // std::cout << " sound speed = " << std::sqrt(pgas_0/rho_0) << std::endl;
  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        phydro->w(IDN,k,j,i) = rho_0;
        phydro->w(IPR,k,j,i) = pgas_0;
        phydro->w(IVX,k,j,i) = 0.0;
        phydro->w(IVY,k,j,i) = 0.0;
        phydro->w(IVZ,k,j,i) = 0.0;
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
//! \fn void Cooling(MeshBlock *pmb, const Real t, const Real dt,
//!       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
//!       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
//!       AthenaArray<Real> &cons_scalar)
//! \brief function for cooling source term
//!        must use prim to set cons
//========================================================================================
void Cooling(MeshBlock *pmb, const Real t, const Real dt,
       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
       AthenaArray<Real> &cons_scalar) {
  // Determine which part of step this is
  bool predict_step = prim.data() == pmb->phydro->w.data();

  AthenaArray<Real> edot;
  edot.InitWithShallowSlice(pmb->user_out_var, 4, 0, 1);

  Real delta_e_block = 0.0;
  Real delta_e_ceil_block  = 0.0;

  // Extract indices
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  // get dt in physical units
  Real dt_ = dt*punit->Time;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        // Extract rho and P from previous steps
        const Real P_before = prim(IPR,k,j,i);
        const Real rho_before = prim(IDN,k,j,i);

        Real gamma_adi = pcool->gamma_adi;
        Real u = P_before/(gamma_adi - 1.0); // internal energy in code units

        // calculate nH in physical units before cooling
        // not necessary for our typical choice but added here for completeness
        Real nH_before = rho_before*pcool->to_nH;
        // T here is not T = P/(n*k_B) but T*(n/nH)=P/(n_H*k_B)
        Real T_before = P_before*pcool->to_pok/nH_before;

        Real T_update = 0.;
        T_update += T_before;
        // dT/dt = - T/tcool(P(T,nH),nH) ---- RK4
        // T and k are in physical units,
        // but rho and P passed into tcool function are in code units
        Real k1 = -1.0 * (T_update/tcool(pcool, rho_before, P_before));
        Real T2 = T_update + 0.5*dt_*k1, P2 = T2*nH_before/pcool->to_pok;
        Real k2 = -1.0 * T2 / tcool(pcool, rho_before, P2);
        Real T3 = T_update + 0.5*dt_*k2, P3 = T3*nH_before/pcool->to_pok;
        Real k3 = -1.0 * T3 / tcool(pcool, rho_before, P3);
        Real T4 = T_update + dt_*k3, P4 = T4*nH_before/pcool->to_pok;
        Real k4 = -1.0 * T4 / tcool(pcool, rho_before, P4);
        T_update += (k1 + 2.*k2 + 2.*k3 + k4)/6.0 * dt_;

        // dont cool below cooling floor and find new internal thermal energy
        Real T_floor = pcool->Get_Tfloor();
        Real T_max = pcool->Get_Tmax();

        // Both P and u are in code units
        Real P_after = std::max(T_update,T_floor)*nH_before/pcool->to_pok;
        Real u_after = P_after/(gamma_adi-1.0);

        // temperature ceiling
        Real delta_e_ceil = 0.0;
        if (T_update > T_max) {
          delta_e_ceil -= u_after;
          // Both P and u are in code units
          P_after = T_max*nH_before/pcool->to_pok;
          u_after = P_after/(gamma_adi-1.0);
          delta_e_ceil += u_after;
          T_update = T_max;
        }

        // double check that you aren't cooling away all internal thermal energy
        Real delta_e = u_after - u;

        // change internal energy
        cons(IEN,k,j,i) += delta_e;

        // store edot
        edot(k,j,i) = delta_e / dt;

        // gather total cooling and total ceiling
        if (!predict_step) {
          delta_e_block += delta_e;
          delta_e_ceil_block += delta_e_ceil;
        }
      }
    }
  }
  // add cooling and ceiling to hist outputs
  pmb->ruser_meshblock_data[0](0) += delta_e_block;
  pmb->ruser_meshblock_data[0](1) += delta_e_ceil_block;
  // Free arrays
  edot.DeleteAthenaArray();
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
          Real dtcool = pcool->cfl_cool*std::abs(tcool(pcool,rho,Press))
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
static Real tcool(CoolingFunctionBase *pcool, const Real rho, const Real Press) {
  Real nH = rho*pcool->to_nH;
  Real cool = nH*nH*pcool->Lambda_T(rho, Press);
  Real heat = nH*pcool->Gamma_T(rho, Press);
  Real eint = Press*pcool->punit->Pressure/(pcool->gamma_adi-1);
  Real tcool = eint/(cool - heat);
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
  coolfile << "rho,Press,Temp,cool,heat,tcool" << "\n";

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
