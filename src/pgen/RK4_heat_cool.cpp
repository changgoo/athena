// Problem generator for testing exact cooling method

// C++ headers
#include <algorithm>  // min()
#include <cmath>      // abs(), pow(), sqrt()
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string

// Athena++ headers
#include "../mesh/mesh.hpp"
#include "../athena.hpp"                   // macros, enums, declarations
#include "../athena_arrays.hpp"            // AthenaArray
#include "../globals.hpp"                  // Globals
#include "../parameter_input.hpp"          // ParameterInput
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro

// External library headers
#include <hdf5.h>  // H5*, hid_t, hsize_t, H5*()

// Global variables --- hydro & units
static Real gamma_adi;
static Real rho_0, pgas_0; // in code units
static Real time_scale;//, vel_scale, length_scale; 
static const Real mu=0.62; // mean molecular weight
static const Real muH=1.4; // mean molecular weight per H
static const Real Pconv=169.739; // kB K cm^-3 -> code energy density
static const Real mp = 1.67373522381e-24; // proton mass in grams
static const Real kb = 1.3806488e-16; // Boltzmann's in ergs/K
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



//-----------------------------------------------------------------------------
// Function for preparing Mesh
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  // turb_flag is initialzed in the Mesh constructor to 0 by default;
  // turb_flag = 1 for decaying turbulence
  // turb_flag = 2 for driven turbulence
  // turb_flag = 3 for density perturbations
  turb_flag = pin->GetInteger("problem","turb_flag");
  if(turb_flag != 0){
#ifndef FFT
    std::stringstream msg;
    msg << "### FATAL ERROR in TurbulenceDriver::TurbulenceDriver" << std::endl
        << "non zero Turbulence flag is set without FFT!" << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
#endif
  }

  // Read general parameters from input file
  gamma_adi     = pin->GetReal("hydro", "gamma");
  time_scale    = pin->GetReal("problem", "time_scale"); // (1 erg/K)/kB //length_scale/vel_scale;
  rho_0         = pin->GetReal("problem", "rho_0"); // measured in m_p muH cm^-3
  pgas_0        = pin->GetReal("problem", "pgas_0"); // measured in kB K cm^-3
  T_PE          = pin->GetReal("problem", "T_PE"); // temperature below which PE heating is applied
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

//-----------------------------------------------------------------------------
// Function for preparing MeshBlock
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  // Allocate storage for keeping track of cooling
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(2);
  ruser_meshblock_data[0](0) = 0.0; // e_cool
  ruser_meshblock_data[0](1) = 0.0; // e_ceil

  // Set output variables
  AllocateUserOutputVariables(1);

  return;
}

//-----------------------------------------------------------------------------
// Function for setting initial conditions
// Inputs:
//   pin: parameters
// Outputs: (none)

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
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

  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    Real z = pcoord->x3v(k);
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        phydro->w(IDN,k,j,i) = rho_0;
        phydro->w(IPR,k,j,i) = pgas_0*Pconv;
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




//----------------------------------------------------------------------------------------
// Source function for cooling 
// Inputs:
//   pmb: pointer to MeshBlock
//   t,dt: time (not used) and timestep
//   prim: primitives
//   bcc: cell-centered magnetic fields (not used)
// Outputs:
//   cons: conserved variables updated

void Cooling(MeshBlock *pmb, const Real t, const Real dt,
	     const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
	     const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, 
	     AthenaArray<Real> &cons_scalar)
{


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


  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        // Extract primitive and conserved quantities
        const Real &rho_half = prim(IDN,k,j,i);
        const Real &pgas_half = prim(IPR,k,j,i);
        Real &rho = cons(IDN,k,j,i);
        Real &e = cons(IEN,k,j,i);
        Real &m1 = cons(IM1,k,j,i);
        Real &m2 = cons(IM2,k,j,i);
        Real &m3 = cons(IM3,k,j,i);

        Real kinetic = (SQR(m1) + SQR(m2) + SQR(m3)) / (2.0 * rho);
        //Real u = e - kinetic;
	//Real P = u * (gamma_adi-1.0)/Pconv;
	Real P = pgas_half/Pconv;
	Real u = pgas_half/(gamma_adi - 1.0);
	//std::cout << "P_i=" << P << std::endl;

        // calculate temperature in physical units before cooling
        Real T_before = (mu/muH) * P/rho_half;
        Real nH = rho_half;

        Real T_update = 0.;
        T_update += T_before;
        // dT/dt = - T/tcool(T,nH) ---- RK4
        Real k1 = -1.0 * (T_update/tcool(T_update, nH));
        Real k2 = -1.0 * (T_update + 0.5*dt*time_scale * k1)/tcool(T_update + 0.5*dt*time_scale * k1, nH);
        Real k3 = -1.0 * (T_update + 0.5*dt*time_scale * k2)/tcool(T_update + 0.5*dt*time_scale * k2, nH);
        Real k4 = -1.0 * (T_update + dt*time_scale * k3)/tcool(T_update + dt*time_scale * k3, nH);
        T_update += (k1 + 2.*k2 + 2.*k3 + k4)/6.0 * dt*time_scale; 

        // dont cool below cooling floor and find new internal thermal energy 
	Real P_after = (std::max(T_update,T_floor) * rho_half *(muH/mu));
        Real u_after = Pconv*P_after/(gamma_adi-1.0);

        // temperature ceiling 
        Real delta_e_ceil = 0.0;
        if (T_update > T_max){
          delta_e_ceil -= u_after;
          u_after = Pconv*(std::min(T_update,T_max) * rho_half* (muH/mu))/(gamma_adi-1.0);
          delta_e_ceil += u_after;
          T_update = T_max;
        }

        // double check that you aren't cooling away all internal thermal energy
        Real delta_e = u_after - u;

        // change internal energy
        e += delta_e;  
        
        // store edot
        edot(k,j,i) = delta_e / dt;

        // gather total cooling and total ceiling
        if (not predict_step) {
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



//----------------------------------------------------------------------------------------
// Function to calculate the timestep required to resolve cooling 
//          tcool = 3/2 P/Edot_cool
// Inputs:
//   pmb: pointer to MeshBlock
Real cooling_timestep(MeshBlock *pmb)
{
  Real min_dt=1.0e10;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real &P = pmb->phydro->w(IPR,k,j,i);
        Real &rho = pmb->phydro->w(IDN,k,j,i);
        Real T_before = (mu/muH) * P/(rho*Pconv);
        Real nH = rho;
        if (T_before > 1.01 * T_floor){
          min_dt = std::min(min_dt, cfl_cool * std::abs(tcool(T_before,nH))/time_scale);
        }
        min_dt = std::max(dt_cutoff,min_dt);
      }
    }
  }
  return min_dt;
}


// ============================================================================
// piecewise power-law fit to the cooling curve with
// temperature in K and L in erg cm^3 / s 
static Real Lambda_T(const Real T)
{
  int k, n=nfit_cool-1;
  // first find the temperature bin 
  for(k=n; k>=0; k--){
    if (T >= T_cooling_curve[k])
      break;
  }
  if (T > T_cooling_curve[0]){
    return (lambda_cooling_curve[k] * 
	    pow(T/T_cooling_curve[k], exponent_cooling_curve[k]));
  } else {
    return 1.0e-50;
  }
}
// ============================================================================



// ============================================================================
static Real tcool(const Real T, const Real nH)
{
  if (T < T_PE){
    return (kb * T) / ( (gamma_adi-1.0) * (mu/muH) * (nH * Lambda_T(T) - Gamma) );
  } else {
    return (kb * T) / ( (gamma_adi-1.0) * (mu/muH) * nH * Lambda_T(T) );
  }
}
// ============================================================================



//----------------------------------------------------------------------------------------
// Cooling losses
// Inputs:
//   pmb: pointer to MeshBlock
//   iout: index of history output
// Outputs:
//   returned value: sum of all energy losses due to different cooling mechanisms
// Notes:
//   resets time-integrated values to 0
//   cooling mechanisms are:
//     0: physical radiative losses
//     0: ceiling

Real CoolingLosses(MeshBlock *pmb, int iout)
{
  Real delta_e = pmb->ruser_meshblock_data[0](iout);
  pmb->ruser_meshblock_data[0](iout) = 0.0;
  return delta_e;
}
