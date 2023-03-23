//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linear_ti_test.cpp
//! \brief Problem generator for thermal instability test using TIGRESS classic cooling
//! function. A small perturbation is given to a uniform medium in
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
#include "../microphysics/cooling.hpp"     // CoolingSolver, CoolingFunctionBase
#include "../microphysics/units.hpp"     // Units
#include "../parameter_input.hpp"          // ParameterInput

// mean density input to the simulation, for comparing to the maximum density
Real rhobar_init;
// Length of the box in code units, initialized in InitUserMeshData()
Real Lbox;

Real MaxOverDens(MeshBlock *pmb, int iout);

// calculate growth rate of perturbation
static Real SolveCubic(const Real b, const Real c, const Real d);
static Real OmegaG(MeshBlock *pmb, const Real rho, const Real Press, const Real k);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  // determine length of the box in code units
  Lbox = mesh_size.x1max - mesh_size.x1min;

  if (cooling) {
    std::string cooling_type = pin->GetString("cooling", "cooling");
    if (cooling_type.compare("enroll") == 0) {
      EnrollUserExplicitSourceFunction(&CoolingSolver::CoolingSourceTerm);
      std::cout << "[Problem] Cooling solver is enrolled" << std::endl;
    } else if (cooling_type.compare("op_split") == 0) {
      std::cout << "[Problem] Cooling solver is set to operator split" << std::endl;
    }
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in ProblemGenerator" << std::endl
        << "Cooling must be turned on" << std::endl;
    ATHENA_ERROR(msg);
  }

  // Enroll timestep so that dt <= min t_cool
  EnrollUserTimeStepFunction(&CoolingSolver::CoolingTimeStep);

  // Enroll user-defined functions
  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, MaxOverDens, "rho_max");
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
  int kn = pin->GetInteger("problem", "kn");
  // background mean density
  Real nH_0   = pin->GetReal("problem", "nH_0"); // measured in m_p muH cm^-3
  // the gas pressure is then set by requiring that the gas be in thermal
  // equilibrium at the given gas density. We do this via root finding the
  // correct temperature.
  // tolerance to stop root finding convergence search
  Real tol = 1e-12;
  // set the initial pgas at the approximate extremes
  CoolingFunctionBase *pcf = pcool->pcf;
  Real pgas_low = 2*pcf->Get_Tfloor()*nH_0*pcf->pok_to_code_press;
  Real pgas_high = pcf->Get_Tmax()*nH_0*pcf->pok_to_code_press;
  Real pgas_mid = (pgas_high + pgas_low)/2.;
  Real rho_0 = nH_0*pcf->nH_to_code_den;
  // initialize the cooling times
  Real tcool_low = pcf->CoolingTime(rho_0, pgas_low);
  Real tcool_high = pcf->CoolingTime(rho_0, pgas_high);
  Real tcool_mid = pcf->CoolingTime(rho_0, pgas_mid);
  // if the endpoints do not have opposite signs on their cooling times we
  // are not guaranteed a zero exists, so we throw an error
  if (tcool_low*tcool_high>=0) {
    std::stringstream msg;
    msg << "### ERROR in ProblemGenerator " << std::endl
        << "pressure guesses must have tcool with opposite signs!" << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
  }
  // find the equilibrium point of the cooling curve using
  // bisection root finding
  while ((pgas_high-pgas_low)/pgas_mid > tol) {
    // std::cout << " (P/k, tcool) in cgs = " << pgas_mid*pcf->code_press_to_pok << " "
    //           << tcool_low << " " << tcool_mid << " " << tcool_high << std::endl;
    if (tcool_low*tcool_mid < 0) {
      pgas_high = pgas_mid;
      tcool_high = tcool_mid;
      pgas_mid = (pgas_high + pgas_low)/2.;
      tcool_mid = pcf->CoolingTime(rho_0, pgas_mid);
    } else {
      pgas_low = pgas_mid;
      tcool_low = tcool_mid;
      pgas_mid = (pgas_high + pgas_low)/2.;
      tcool_mid = pcf->CoolingTime(rho_0, pgas_mid);
    }
  }
  // set the pressure to the middle value found
  Real pgas_0 = pgas_mid*pcf->code_press_to_pok;
  // below is for sanity check. Uncomment if needed
  Real muH = pcf->Get_muH();
  Real mu = pcf->Get_mu(rho_0, pgas_mid);
  Real T = pcf->GetTemperature(rho_0, pgas_mid);
  //
  if (Globals::my_rank == 0) {
    std::cout << "============== Check Initialization ===============" << std::endl
              << " Input (nH, P/k, T) in cgs = " << rho_0 << " " << pgas_0
              << " " << T << std::endl
              << " nH to code den = " << pcf->nH_to_code_den
              << "  mu = " << mu << " muH = " << muH << std::endl;
  }
  // store the initial mean density as a global variable
  rhobar_init = rho_0;

  // determine wavenumber in inverse code length
  Real kx = 2*PI*kn/Lbox;
  // determine growth rate via dispersion relation
  Real om = OmegaG(this, rho_0, pgas_mid, kx);
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
        phydro->w(IPR,k,j,i) = pgas_mid + pdev;
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
  if (pcool->op_flag) pcool->OperatorSplitSolver(this);
  // pcool->CalculateTotalCoolingRate(this,pmy_mesh->dt);

  return;
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
static Real OmegaG(MeshBlock *pmb, const Real rho, const Real Press, const Real k) {
  CoolingSolver *pcool=pmb->pcool;
  CoolingFunctionBase *pcf = pcool->pcf;
  Units *punit = pmb->punit;
  Real gm1 = pcf->gamma_adi-1;
  // get Temperature in Kelvin
  Real T = pcf->GetTemperature(rho,Press);
  // density in nH
  Real nH = rho*pcf->code_den_to_nH;
  // Pressure in c.g.s
  Real P = Press*punit->code_pressure_cgs;
  // sounds spped in c.g.s
  Real cs = std::sqrt(pcf->gamma_adi*(Press/rho))*punit->code_velocity_cgs;
  // krho in c.g.s
  Real krho = gm1*nH*nH*pcf->Lambda_T(rho,Press)/(P*cs);
  krho *= punit->code_length_cgs; // krho in code units
  cs /= punit->code_velocity_cgs; // cs in code units

  // std::cout << "  cs = " << cs << std::endl;
  // std::cout << "  krho = " << krho << std::endl;

  Real dlnL_dlnT = pcf->dlnL_dlnT(rho,Press);
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
