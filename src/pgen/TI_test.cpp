//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file TI_test.cpp
//! \brief Problem generator for thermal instability test using TIGRESS classic cooling
//! function and RK4 integration. A small perturbation is given to a uniform medium in
//! equilibrium but which is thermally unstable. We then track thr growth rate of the
//! maximum density deviation.
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
// mean density input to the simulation, for comparing to the maximum density
Real rhobar_init;
// Length of the box in code units, initialized in InitUserMeshData()
Real Lbox;
// Length of the box in code units, initialized in InitUserMeshData()
Real gamma_adi;

// explicit cooling solver using RK4 method for integration
// slightly modified to update T*(n/n_H) rather than T itself
void CoolingRK4(MeshBlock *pmb, const Real t, const Real dt,
       const AthenaArray<Real> &prim,const AthenaArray<Real> &prim_scalar,
       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
       AthenaArray<Real> &cons_scalar);
void CoolingEuler(MeshBlock *pmb, const Real t, const Real dt,
      const AthenaArray<Real> &prim,const AthenaArray<Real> &prim_scalar,
      const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
      AthenaArray<Real> &cons_scalar);
Real CoolingLosses(MeshBlock *pmb, int iout);
Real MaxOverDens(MeshBlock *pmb, int iout);
static Real CoolingTimestep(MeshBlock *pmb);

// calculate tcool = e/L(rho, P)
static Real tcool(const Real rho, const Real Press);
static Real dtnet(const Real rho, const Real Press);

// calculate growth rate of perturbation
static Real SolveCubic(const Real b, const Real c, const Real d);
static Real OmegaG(const Real rho, const Real Press, const Real k);

// Utility functions for debugging
void PrintCoolingFunction(std::string coolftn);
void PrintParameters(const Real rho, const Real Press);

bool op_cooling=false;
//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  // determine length of the box in code units
  Lbox = mesh_size.x1max - mesh_size.x1min;

  // initialize cooling function
  // currently, two cooling functions supported (tigress, plf)
  std::string coolftn = pin->GetOrAddString("cooling", "coolftn", "tigress");

  if (coolftn.compare("tigress") == 0) {
    pcool = new TigressClassic(pin);
    if (Globals::my_rank == 0)
      std::cout << "Cooling function is set to TigressClassic" << std::endl;
  } else if (coolftn.compare("plf") ==0) {
    pcool = new PiecewiseLinearFits(pin);
    if (Globals::my_rank == 0)
      std::cout << "Cooling function is set to PiecewiseLinearFits" << std::endl;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in ProblemGenerator" << std::endl
        << "coolftn = " << coolftn.c_str() << " is not supported" << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
  }

  // shorthand for unit class
  // unit class is initialized within cooling function constructor
  // to use appropreate mu and muH
  punit = pcool->punit;
  // set gamma as a global variable in the problem
  gamma_adi = pcool->gamma_adi;

  // show some values for sanity check.
  if (Globals::my_rank == 0) {
    // dump cooling function used in ascii format to e.g., tigress_coolftn.txt
    // PrintCoolingFunction(coolftn);

    // print out units and constants in code units
    punit->PrintCodeUnits();
    punit->PrintConstantsInCodeUnits();
  }

  // Enroll source function
  // currently, two cooling solvers are supported (euler, rk4)
  std::string coolsolver = pin->GetOrAddString("cooling", "solver", "euler");
  if (coolsolver.compare("euler") == 0) {
    EnrollUserExplicitSourceFunction(CoolingEuler);
    if (Globals::my_rank == 0)
      std::cout << "Cooling solver is set to Euler" << std::endl;
  } else if (coolsolver.compare("rk4") ==0) {
    EnrollUserExplicitSourceFunction(CoolingRK4);
    if (Globals::my_rank == 0)
      std::cout << "Cooling solver is set to RK4" << std::endl;
  } else if (coolsolver.compare("op") ==0) {
    op_cooling=true;
    std::cout << "Cooling solver is set to operator split" << std::endl;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in ProblemGenerator" << std::endl
        << "coolsolver = " << coolsolver.c_str() << " is not supported" << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
  }


  // Enroll timestep so that dt <= min t_cool
  EnrollUserTimeStepFunction(CoolingTimestep);

  // Enroll user-defined functions
  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, CoolingLosses, "e_cool");
  EnrollUserHistoryOutput(1, CoolingLosses, "e_ceil");
  EnrollUserHistoryOutput(1, MaxOverDens, "rho_max");
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

  // the relative amplitude of the initial density perturbation
  Real alpha = pin->GetReal("problem", "alpha");
  // the wavenumber of the perturbation
  int kn = pin->GetInteger("problem","kn");
  // background mean density
  Real rho_0   = pin->GetReal("problem", "rho_0"); // measured in m_p muH cm^-3
  // the gas pressure is then set by requiring that the gas be in thermal
  // equilibrium at the given gas density. We do this via root finding the
  // correct temperature.
  Real pgas_low, pgas_high, pgas_mid;
  Real tcool_low, tcool_high, tcool_mid;
  // tolerance to stop root finding convergence search
  Real tol = 1e-12;
  // set the initial pgas at the approximate extremes
  pgas_low = pcool->Get_Tfloor()*rho_0/pcool->to_pok;
  pgas_high = pcool->Get_Tmax()*rho_0/pcool->to_pok;
  pgas_mid = (pgas_high + pgas_low)/2.;
  rho_0 /= pcool->to_nH;
  // initialize the cooling times
  tcool_low = tcool(rho_0, pgas_low);
  tcool_high = tcool(rho_0, pgas_high);
  tcool_mid = tcool(rho_0, pgas_mid);
  // if the endpoints do not have opposite signs on their cooling times we
  // are not guaranteed a zero exists, so we throw an error
  if (tcool_low*tcool_high>0) {
    std::stringstream msg;
    msg << "### ERROR in ProblemGenerator " << std::endl
        << "pressure guesses must have tcool with opposite signs!" << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
  }
  // find the equilibrium point of the cooling curve using
  // bisection root finding
  while ((pgas_high-pgas_low)/pgas_mid > tol) {
    if (tcool_low*tcool_mid < 0) {
      pgas_high = pgas_mid;
      tcool_high = tcool_mid;
      pgas_mid = (pgas_high + pgas_low)/2.;
      tcool_mid = tcool(rho_0, pgas_mid);
    } else {
      pgas_low = pgas_mid;
      tcool_low = tcool_mid;
      pgas_mid = (pgas_high + pgas_low)/2.;
      tcool_mid = tcool(rho_0, pgas_mid);
    }
  }
  // set the pressure to the middle value found
  Real pgas_0 = pgas_mid*pcool->to_pok;
  rho_0 = rho_0*pcool->to_nH;
  // below is for sanity check. Uncomment if needed
  Real muH = pcool->Get_muH();
  Real mu = pcool->Get_mu(rho_0, pgas_0/pcool->to_pok);
  Real T = pgas_0/rho_0*(mu/muH);
  //
  if (Globals::my_rank == 0) {
    std::cout << "============== Check Initialization ===============" << std::endl
              << " Input (nH, P/k, T) in cgs = " << rho_0 << " " << pgas_0
              << " " << T << std::endl
              << "  mu = " << mu << " mu(punit) = " << punit->mu
              << " muH = " << muH << std::endl;
              // << " tcool = " << tcool(T, rho_0) << std::endl;
  }
  rho_0 /= pcool->to_nH; // to code units
  pgas_0 /= pcool->to_pok; // to code units
  // store the initial mean density as a global variable
  rhobar_init = rho_0;

  PrintParameters(rho_0,pgas_0);
  T = pcool->GetTemperature(rho_0,pgas_0);
  Real nH = rho_0*pcool->to_nH;
  if (Globals::my_rank == 0) {
    std::cout << "  Tempearture = " << T << std::endl;
    std::cout << "  tcool = " << tcool(rho_0, pgas_0) << std::endl;
    std::cout << "  sound speed = " << std::sqrt(pgas_0/rho_0) << std::endl;
  }
  // determine wavenumber in inverse code length
  Real kx = 2*PI*kn/Lbox;
  // determine growth rate via dispersion relation
  Real om = OmegaG(rho_0,pgas_0,kx);
  if (Globals::my_rank == 0) {
    std::cout << "  Lbox = " << Lbox << std::endl;
    std::cout << "  k = " << kx << std::endl;
    std::cout << "  omega = " << om << std::endl;
  }
  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        Real x = pcoord->x1v(i);
        Real pdev = -1*alpha*rho_0*(om/kx)*(om/kx)*std::cos(kx*x);
        Real vdev = -1*alpha*(om/kx)*std::sin(kx*x);
        phydro->w(IDN,k,j,i) = rho_0*(1 + alpha*std::cos(kx*x));
        phydro->w(IPR,k,j,i) = pgas_0 + pdev;
        phydro->w(IVX,k,j,i) = vdev;
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
//! \fn void MeshBlock::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================

void MeshBlock::UserWorkInLoop() {
  if (!op_cooling) return;
  Real dt_mhd = pmy_mesh->dt*punit->Time;
  // std::cout << " userwork" << std::endl;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        // both u and w are updated by integrator
        Real& u_d  = phydro->u(IDN,k,j,i);
        Real& u_e  = phydro->u(IEN,k,j,i);

        Real& w_d  = phydro->w(IDN,k,j,i);
        Real& w_p  = phydro->w(IPR,k,j,i);
        // std::cout << " denstiy: " << w_d << " pressure: " << w_p << std::endl;
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

        // first cooling
        Real P_before = w_p; // store original P
        Real rho_before = w_d; // store original d
        Real nH_before = w_d*pcool->to_nH; // store original nH
        Real T_before = P_before*pcool->to_pok/nH_before;

        Real dt_net = dtnet(rho_before, P_before);
        Real dt_sub = std::min(dt_mhd,dt_net);

        Real T_next = T_before*(1-dt_sub/tcool(rho_before, P_before));
        Real P_next = nH_before*T_next/pcool->to_pok;

        Real tnow = dt_sub, tleft = dt_mhd-dt_sub;
        // std::cout << " [Initial] P: " << P_before
        //           << " T: " << T_before << std::endl
        //           << " [Next] P: " << P_next
        //           << " T: " << T_next << std::endl
        //           << " dt: " << dt_mhd << " " << dt_net << " " << dt_sub << std::endl
        //           << " t: " << tnow << " " << tleft << std::endl;

        while (tnow < dt_mhd) {
          P_before = P_next;
          T_before = P_before*pcool->to_pok/nH_before;
          dt_net = dtnet(rho_before, P_before);
          dt_sub = std::min(std::min(dt_mhd,dt_net),tleft);

          T_next = T_before*(1-dt_sub/tcool(rho_before, P_before));
          P_next = nH_before*T_next/pcool->to_pok;

          tnow += dt_sub;
          tleft = dt_mhd-tnow;
        }
        // dont cool below cooling floor and find new internal thermal energy
        Real T_floor = pcool->Get_Tfloor();
        Real P_floor = T_floor*nH_before/pcool->to_pok;

        Real P_after = std::max(P_next,P_floor);
        Real u_after = P_after/(pcool->gamma_adi-1.0);

        // change internal energy
        u_e = u_after + e_non_thermal;
        w_p = P_after;
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void CoolingEuler(MeshBlock *pmb, const Real t, const Real dt,
//!       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
//!       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
//!       AthenaArray<Real> &cons_scalar)
//! \brief function for cooling source term
//!        must use prim to set cons
//========================================================================================
void CoolingEuler(MeshBlock *pmb, const Real t, const Real dt,
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

        Real u = P_before/(gamma_adi - 1.0); // internal energy in code units

        // calculate nH in physical units before cooling
        // not necessary for our typical choice but added here for completeness
        Real nH_before = rho_before*pcool->to_nH;
        // T here is not T = P/(n*k_B) but T*(n/nH)=P/(n_H*k_B)
        Real T_before = P_before*pcool->to_pok/nH_before;

        Real T_update = T_before - T_before/tcool(rho_before, P_before)*dt_;

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
//! \fn void CoolingRK4(MeshBlock *pmb, const Real t, const Real dt,
//!       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
//!       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
//!       AthenaArray<Real> &cons_scalar)
//! \brief function for cooling source term
//!        must use prim to set cons
//========================================================================================
void CoolingRK4(MeshBlock *pmb, const Real t, const Real dt,
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

        Real u = P_before/(gamma_adi - 1.0); // internal energy in code units

        // calculate nH in physical units before cooling
        // not necessary for our typical choice but added here for completeness
        Real nH_before = rho_before*pcool->to_nH;
        // T here is not T = P/(n*k_B) but T*(n/nH)=P/(n_H*k_B)
        Real T_before = P_before*pcool->to_pok/nH_before;

        Real T_update = 0.;
        T_update += T_before;
        // dT/dt = - T/tcool(P(T,nH),nH) ---- RK4
        // T, k, and tcool are in physical units,
        // but rho and P passed into tcool function are in code units
        Real k1 = -1.0 * (T_update/tcool(rho_before, P_before));
        Real T2 = T_update + 0.5*dt_*k1, P2 = T2*nH_before/pcool->to_pok;
        Real k2 = -1.0 * T2 / tcool(rho_before, P2);
        Real T3 = T_update + 0.5*dt_*k2, P3 = T3*nH_before/pcool->to_pok;
        Real k3 = -1.0 * T3 / tcool(rho_before, P3);
        Real T4 = T_update + dt_*k3, P4 = T4*nH_before/pcool->to_pok;
        Real k4 = -1.0 * T4 / tcool(rho_before, P4);
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
//! \fn Real CoolingTimestep(MeshBlock *pmb)
//! \brief Function to calculate the timestep required to resolve cooling
//!        tcool = 3/2 P/Edot_cool
//========================================================================================
static Real CoolingTimestep(MeshBlock *pmb) {
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
          Real dtcool = pcool->cfl_cool*std::abs(dtnet(rho,Press))
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
//! \fn static Real tcool(const Real rho, const Real Press)
//! \brief tcool = e / (n_H^2*Cool - n_H*heat)
//! \note
//! - input rho and P are in code Units
//! - output tcool is in second
//========================================================================================
static Real tcool(const Real rho, const Real Press) {
  Real nH = rho*pcool->to_nH;
  Real cool = nH*nH*pcool->Lambda_T(rho, Press);
  Real heat = nH*pcool->Gamma_T(rho, Press);
  Real eint = Press*pcool->punit->Pressure/(pcool->gamma_adi-1);
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
static Real dtnet(const Real rho, const Real Press) {
  Real nH = rho*pcool->to_nH;
  Real cool = nH*nH*pcool->Lambda_T(rho, Press);
  Real heat = nH*pcool->Gamma_T(rho, Press);
  Real eint = Press*pcool->punit->Pressure/(pcool->gamma_adi-1);
  Real tcool = eint/(cool + heat);
  return tcool;
}

//========================================================================================
//! \fn static Real SolveCubic(const Real a, const Real b, const Real c, const Real d)
//! \brief solve a cubic equation
//! \note
//! - input coeffs of eq. x^3 + b*x^2 + c*x + d = 0
//! - output greatest real solution to the cubic equation (due to Cardano 1545)
//! - reference https://mathworld.wolfram.com/CubicFormula.html
//========================================================================================
static Real SolveCubic(const Real b, const Real c, const Real d) {
  Real Q,R,D,S,T,res;
  Real theta,z1,z2,z3;
  // std::cout << "  B = " << b << "  C = " << c << "  D = " << d << std::endl;
  // variables of use in solution Eqs. 22, 23 of reference
  Q = (3*c - b*b)/9;
  R = (9*b*c - 27*d - 2*b*b*b)/54;
  // calculate the polynomial discriminant
  D = Q*Q*Q + R*R;
  // if the dsicriminant is positive there is one real root
  if (D>0) {
    S = std::cbrt(R + std::sqrt(D));
    T = std::cbrt(R - std::sqrt(D));
    res =  S + T - b/3;
  } else {
  // if the discriminant is zero all roots are real and at least two are equal
  // if the discriminant is negative there are three, distinct, real roots
    theta = std::acos(R/std::sqrt( -1*Q*Q*Q ));
    // calculate the three real roots
    z1 = 2*std::sqrt(-1*Q)*std::cos(theta/3) - b/3;
    z2 = 2*std::sqrt(-1*Q)*std::cos((theta + 2*PI)/3) - b/3;
    z3 = 2*std::sqrt(-1*Q)*std::cos((theta + 4*PI)/3) - b/3;
    // std::cout << "  z1 = " << z1 << "  z2 = " << z2 << "  z3 = " << z3 << std::endl;
    if ((z1>z2)&&(z1>z3))
      res = z1;
    else if(z2>z3)
      res = z2;
    else
      res = z3;
  }
  return res;
}

//========================================================================================
//! \fn static Real OmegaG(const Real rho, const Real Press, const Real k)
//! \brief growth rate of instability
//! \note
//! - input rho and P are in code Units
//! - output growth rate of instability in code Units
//========================================================================================
static Real OmegaG(const Real rho, const Real Press, const Real k) {
  Real gm1 = gamma_adi-1;
  // get Temperature in Kelvin
  Real T = pcool->GetTemperature(rho,Press);
  // density in nH
  Real nH = rho*pcool->to_nH;
  // Pressure in c.g.s
  Real P = Press*punit->Pressure;
  // sounds spped in c.g.s
  Real cs = std::sqrt(gamma_adi*(Press/rho))*punit->Velocity;
  // krho in c.g.s
  Real krho = gm1*nH*nH*pcool->Lambda_T(rho,Press)/(P*cs);
  krho *= punit->Length; // krho in code units
  cs /= punit->Velocity; // cs in code units

  // std::cout << "  cs = " << cs << std::endl;
  // std::cout << "  krho = " << krho << std::endl;

  Real dlnL_dlnT = pcool->dlnL_dlnT(rho,Press);
  // std::cout << "  dlnL_dlnT = " << dlnL_dlnT << std::endl;
  // kT in code units based on derivative
  Real kT = krho*dlnL_dlnT;
  // get coefficients in cubic dispersion relation
  Real B = cs*kT;
  Real C = cs*cs*k*k;
  Real D = (3./5)*cs*cs*cs*k*k*(kT - krho);
  // solve  dispersion relation for the growth rate
  Real om = SolveCubic(B,C,D);
  return om;
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
//! \fn Real MaxOverDens(MeshBlock *pmb, int iout);
//! \brief Maximum over density as a history variable, used to track the growth rate of
//!        the overdensity that is initialized
//========================================================================================
Real MaxOverDens(MeshBlock *pmb, int iout) {
  // Prepare index bounds
  int il = pmb->is - NGHOST;
  int iu = pmb->ie + NGHOST;
  int jl = pmb->js;
  int ju = pmb->je;
  if (pmb->block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = pmb->ks;
  int ku = pmb->ke;
  if (pmb->block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }
  Real dmax = 0.0;
  // loop over grid to get highest density
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        Real den = pmb->phydro->w(IDN,k,j,i) - rhobar_init;
        if (den>dmax) dmax = den;
      }
    }
  }
  return dmax/rhobar_init;
}

//========================================================================================
//! \fn void PrintCoolingFunction(std::string coolftn)
//! \brief private function to check cooling and heating functions
//========================================================================================
void PrintCoolingFunction(std::string coolftn) {
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
    Real t_cool = tcool(rho,Press);
    coolfile << rho << "," << Press << "," << Temp << ","
             << cool << "," << heat << "," << t_cool << "\n";
  }
}

//========================================================================================
//! \fn void PrintParameters(const Real rho, const Real Press)
//! \brief print function for sanity check
//========================================================================================
void PrintParameters(const Real rho, const Real Press) {
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
