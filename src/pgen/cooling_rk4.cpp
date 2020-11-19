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

// Global variables --- hydro & units
// Pointer to unit class. Need to decide if we want to attach this to Mesh or MeshBlock
// Removed redundant global variables for units and constants
Units *punit;

static Real gamma_adi;
static const Real mu = 0.62; // mean molecular weight
static const Real muH = 1.4; // mean molecular weight per H
static const Real mean_mass_per_H = muH*Constants::mH; // mean mass per H
static Real Gamma, T_PE, T_floor, T_max;
static Real cfl_cool, dt_cutoff;

// Global variables --- cooling
static int nfit_cool = 12;
static Real T_cooling_curve[12] =
  {0.99999999e1,
   1.0e+02, 6.0e+03, 1.75e+04,
   4.0e+04, 8.7e+04, 2.30e+05,
   3.6e+05, 1.5e+06, 3.50e+06,
   2.6e+07, 1.0e+12};

static Real lambda_cooling_curve[12] =
  { 1e-30,
    1.00e-27,   2.00e-26,   1.50e-22,
    1.20e-22,   5.25e-22,   5.20e-22,
    2.25e-22,   1.25e-22,   3.50e-23,
    2.10e-23,   4.12e-21};

static Real exponent_cooling_curve[12] =
  {3.,
   0.73167566,  8.33549431, -0.26992783,
   1.89942352, -0.00984338, -1.8698263 ,
   -0.41187018, -1.50238273, -0.25473349,
   0.5000359, 0.5 };

static Real Lambda_T(const Real T);
static Real tcool(const Real T, const Real nH);

void Cooling(MeshBlock *pmb, const Real t, const Real dt,
       const AthenaArray<Real> &prim,const AthenaArray<Real> &prim_scalar,
       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
       AthenaArray<Real> &cons_scalar);

static Real cooling_timestep(MeshBlock *pmb);
Real CoolingLosses(MeshBlock *pmb, int iout);

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
    msg << "### FATAL ERROR in TurbulenceDriver::TurbulenceDriver" << std::endl
        << "non zero Turbulence flag is set without FFT!" << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
#endif
  }

  // set density, length, and velocity units
  Real dunit = mean_mass_per_H; // denisty in code units is number density of hydrogen
  Real lunit = Constants::pc; // length in code units is parsec
  Real vunit = Constants::kms; // velocity in code units is km/s

  // initialize the unit class
  punit = new Units(dunit, lunit, vunit, mu);

  // print out units and constants in code units
  if (Globals::my_rank == 0) {
    punit->PrintCodeUnits();
    punit->PrintConstantsInCodeUnits();
  }
  // Read general parameters from input file
  gamma_adi     = pin->GetReal("hydro", "gamma");

  // temperature below which PE heating is applied
  T_PE          = pin->GetReal("problem", "T_PE");
  Gamma         = pin->GetReal("problem", "Gamma"); //heating rate in ergs / sec
  cfl_cool      = pin->GetReal("problem", "cfl_cool"); // min dt_hydro/dt_cool
  T_floor       = pin->GetReal("problem", "T_floor");
  T_max         = pin->GetReal("problem", "T_max");

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

  std::cout << "in c.g.s. " << "nH = " << rho_0
            << " P/k = " << pgas_0
            << " T[K] = " << pgas_0/rho_0*(mu/muH) << std::endl;
  rho_0 *= mean_mass_per_H/punit->Density;
  pgas_0 *= Constants::kB/punit->Pressure;
  std::cout << "in code units" << "den = " << rho_0
            << " P = " << pgas_0
            << " T[K] = " << pgas_0/rho_0*punit->Temperature << std::endl;



  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    Real z = pcoord->x3v(k);
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
//! \brief Source function for cooling
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

  // set conversion factors
  // this should be one for our typical unit choice but put it here to make it clear
  Real to_nH = punit->Density/mean_mass_per_H;
  // this is obvious for a fixed mu
  Real to_Kelvin = punit->Temperature;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        // Extract rho and P from previous steps
        const Real P_before = prim(IPR,k,j,i);
        const Real rho_before = prim(IDN,k,j,i);

        Real u = P_before/(gamma_adi - 1.0); // internal energy in code units

        // calculate nH, temperature, dt in physical units before cooling
        Real nH_before = rho_before*to_nH;
        Real T_before = P_before/rho_before*punit->Temperature;
        Real dt_ = dt*punit->Time;

        Real T_update = 0.;
        T_update += T_before;
        // dT/dt = - T/tcool(T,nH) ---- RK4
        // everythin here must be in physical units (c.g.s.)
        Real k1 = -1.0 * (T_update/tcool(T_update, nH_before));
        Real k2 = -1.0 * (T_update + 0.5*dt_*k1) /
                  tcool(T_update + 0.5*dt_*k1, nH_before);
        Real k3 = -1.0 * (T_update + 0.5*dt_*k2) /
                  tcool(T_update + 0.5*dt_*k2, nH_before);
        Real k4 = -1.0 * (T_update + dt_*k3) /
                  tcool(T_update + dt_*k3, nH_before);
        T_update += (k1 + 2.*k2 + 2.*k3 + k4)/6.0 * dt_;

        // dont cool below cooling floor and find new internal thermal energy
        // Note (muH/mu) is absorbed into the Temperature unit for a fixed mu
        // Both P and u are in code units
        Real P_after = (std::max(T_update,T_floor)/punit->Temperature * rho_before);
        Real u_after = P_after/(gamma_adi-1.0);

        // temperature ceiling
        Real delta_e_ceil = 0.0;
        if (T_update > T_max) {
          delta_e_ceil -= u_after;
          P_after = (T_max * rho_before * punit->Temperature);
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
Real cooling_timestep(MeshBlock *pmb) {
  Real min_dt=1.0e10;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real Press = pmb->phydro->w(IPR,k,j,i);
        Real rho = pmb->phydro->w(IDN,k,j,i);
        Real T_before = Press/rho*punit->Temperature;
        Real nH = rho*(punit->Density/mean_mass_per_H);
        if (T_before > 1.01 * T_floor) {
          min_dt = std::min(min_dt, cfl_cool * std::abs(tcool(T_before,nH))/punit->Time);
        }
        // min_dt = std::max(dt_cutoff,min_dt);
      }
    }
  }
  return min_dt;
}

//========================================================================================
//! \fn static Real Lambda_T(const Real T)
//! \brief piecewise power-law fit to the cooling curve with
//!        temperature in K and L in erg cm^3 / s
//========================================================================================
static Real Lambda_T(const Real T) {
  int k, n=nfit_cool-1;
  // first find the temperature bin
  for (k=n; k>=0; k--) {
    if (T >= T_cooling_curve[k]) break;
  }
  if (T > T_cooling_curve[0]) {
    return (lambda_cooling_curve[k] *
      pow(T/T_cooling_curve[k], exponent_cooling_curve[k]));
  } else {
    return 1.0e-50;
  }
}

//========================================================================================
//! \fn static Real tcool(const Real T, const Real nH)
//! \brief tcool = e / (n^2*Cool - heat)
//========================================================================================
static Real tcool(const Real T, const Real nH) {
  Real netcool = nH*Lambda_T(T);
  if (T < T_PE) netcool -= Gamma;
  return (Constants::kB*T)/((gamma_adi-1.0)*(mu/muH)*netcool);
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
