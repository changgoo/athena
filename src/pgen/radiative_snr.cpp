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
#include "../fft/perturbation.hpp"         // PerturbationGenerator
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

// user function
void AddSupernova(Mesh *pm);
// user history
Real CoolingLosses(MeshBlock *pmb, int iout);
Real HistoryMass(MeshBlock *pmb, int iout);
Real HistoryEnergy(MeshBlock *pmb, int iout);
Real HistoryRadialMomentum(MeshBlock *pmb, int iout);
Real HistoryShell(MeshBlock *pmb, int iout);

// user timestep
static Real cooling_timestep(MeshBlock *pmb);

// cooling solver related private function
// calculate tcool = e/L(rho, P)
static Real CoolingExplicitSubcycling(Real tend, Real P, const Real rho);
static Real tcool(CoolingFunctionBase *pcool, const Real rho, const Real Press);
static Real dtnet(CoolingFunctionBase *pcool, const Real rho, const Real Press);

// Utility functions for debugging
void PrintCoolingFunction(CoolingFunctionBase *pcool, std::string coolftn);
void PrintParameters(CoolingFunctionBase *pcool, const Real rho, const Real Press);

Real cfl_op_cool=-1;
Real Thot0 = 2.e4, vr0=0.1;
int i_M_hot, i_e_hot, i_M_sh, i_pr_sh, i_RM_sh; // indicies for history

// SN related parameters
Real r_SN, E_SN, t_SN, dt_SN;
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
  // SN related parameters
  t_SN = pin->GetOrAddReal("problem","t_SN",0.0); // when to explode SN
  if (t_SN >= 0.0) {
    r_SN = pin->GetReal("problem","r_SN");
    E_SN = pin->GetOrAddReal("problem","E_SN",1.0);
    dt_SN = pin->GetOrAddReal("problem","dt_SN",-1); // interval of SNe
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
  int n_user_hst=10, i_user_hst=0;
  AllocateUserHistoryOutput(n_user_hst);

  // these should come first
  EnrollUserHistoryOutput(i_user_hst++, CoolingLosses, "e_cool");
  EnrollUserHistoryOutput(i_user_hst++, CoolingLosses, "e_floor");
  //
  EnrollUserHistoryOutput(i_user_hst++, HistoryMass, "M");
  i_M_hot = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryMass, "M_hot");
  EnrollUserHistoryOutput(i_user_hst++, HistoryRadialMomentum, "pr");
  i_M_sh = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryShell, "M_sh");
  i_pr_sh = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryShell, "pr_sh");
  i_RM_sh = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryShell, "RM_sh");
  EnrollUserHistoryOutput(i_user_hst++, HistoryEnergy, "e_th");
  i_e_hot = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryEnergy, "e_hot");

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
  ruser_meshblock_data[0](0) = 0.0; // total e_cool between history dumps
  ruser_meshblock_data[0](1) = 0.0; // total e_cool between history dumps

  // Set output variables
  AllocateUserOutputVariables(2); // instantanoues e_dot, e_dot_floor

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

  rho_0 /= pcool->to_nH; // to code units
  pgas_0 /= pcool->to_pok; // to code units

  // Initialize primitive values
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->u(IDN,k,j,i) = rho_0;
        phydro->u(IEN,k,j,i) = pgas_0/(peos->GetGamma()-1);
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void Mesh::PostInitialize(int res_flag, ParameterInput *pin)
//! \brief add SN energy
//========================================================================================
void Mesh::PostInitialize(int res_flag, ParameterInput *pin) {
  if (t_SN == 0.0) {
    AddSupernova(this);
    t_SN += dt_SN;
  }

  // Add density perturbation
  Real rho_0   = pin->GetReal("problem", "rho_0"); // measured in m_p muH cm^-3
  Real amp_den = pin->GetOrAddReal("problem","amp_den",0.0);
  if (amp_den>0) {
    PerturbationGenerator *ppert;
    ppert = new PerturbationGenerator(this, pin);

    ppert->GenerateScalar(); // generate a scalar in Fourier space
    ppert->AssignScalar(); // do backward FFT and assign the scalar array
    // calculate total
    Real dtot = 0.0, d2tot = 0.0;
    for (int nb=0; nb<nblocal; ++nb) {
      MeshBlock *pmb = my_blocks(nb);
      AthenaArray<Real> dden(ppert->GetScalar(nb));
      for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
          for (int i=pmb->is; i<=pmb->ie; i++) {
            dtot += dden(k,j,i);
            d2tot += SQR(dden(k,j,i));
          }
        }
      }
    }
#ifdef MPI_PARALLEL
    int mpierr;
    // Sum the perturbations over all processors
    mpierr = MPI_Allreduce(MPI_IN_PLACE, &dtot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mpierr = MPI_Allreduce(MPI_IN_PLACE, &d2tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    d2tot /= GetTotalCells();
    dtot /= GetTotalCells();
    Real d_std= std::sqrt(d2tot - dtot*dtot);

    std::cout << " unscaled density std: " << d_std << std::endl;
    // assign normalized density perturbation with assigned amplitude
    for (int nb=0; nb<nblocal; ++nb) {
      MeshBlock *pmb = my_blocks(nb);
      Hydro *phydro = pmb->phydro;
      AthenaArray<Real> dden(ppert->GetScalar(nb));
      for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
          for (int i=pmb->is; i<=pmb->ie; i++) {
            phydro->u(IDN,k,j,i) += rho_0*(amp_den/d_std)*dden(k,j,i);
          }
        }
      }
    }

    delete ppert;
  }

  // velocity perturbation; decaying turbulence
  Real turb_flag(pin->GetOrAddInteger("problem","turb_flag",0));
  if (turb_flag>0) {
    TurbulenceDriver *ptrbd;
    ptrbd = new TurbulenceDriver(this, pin);
    if (res_flag == 0) ptrbd->Driving();
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//! \brief Function called once every time step for Mesh-level user-defined work.
//========================================================================================
void Mesh::UserWorkInLoop() {
  if ((t_SN>0) && (t_SN < time)) {
    AddSupernova(this);
    t_SN += dt_SN;
  }
}
//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================
void MeshBlock::UserWorkInLoop() {
  if (cfl_op_cool < 0) return; // no operator split cooling

  // boundary comm. will not be called.
  // need to solve cooling in the ghost zones
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


  Real dt_mhd = pmy_mesh->dt*punit->Time;

  AthenaArray<Real> edot, edot_floor;
  edot.InitWithShallowSlice(user_out_var, 4, 0, 1);
  edot_floor.InitWithShallowSlice(user_out_var, 4, 1, 1);

  Real T_floor = pcool->Get_Tfloor(); // temperature floor
  Real gm1 = pcool->gamma_adi-1; // gamma-1

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
#pragma omp simd
      for (int i = il; i <= iu; ++i) {
        // both u and w are updated by integrator
        Real& u_d  = phydro->u(IDN,k,j,i);
        Real& u_e  = phydro->u(IEN,k,j,i);

        Real& w_d  = phydro->w(IDN,k,j,i);
        Real& w_p  = phydro->w(IPR,k,j,i);
        // find non-thermal part of energy to keep it the same
        Real e_non_thermal = u_e - w_p/gm1;
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

        // calculate pressure floor
        Real P_floor = rho_before*T_floor/pcool->punit->Temperature;
        Real P_next;
        if (P_before < P_floor)
          P_next = P_floor;
        else
          P_next = CoolingExplicitSubcycling(dt_mhd,P_before,rho_before);

        Real delta_P, delta_P_floor;
        if (P_next == P_floor) {
          // original P is too low; cooling solver is skipped
          // store artificial heating by flooring
          delta_P = 0.;
          delta_P_floor = P_floor-P_before;
        } else if (P_next < P_floor) {
          // cooled too much; apply floor
          // store both cooling loss and artificial heating
          delta_P = P_next-P_before;
          delta_P_floor = P_floor-P_next;
        } else {
          // normal cooling without floor;
          // store cooling loss only
          delta_P = P_next-P_before;
          delta_P_floor = 0.;
        }
        edot(k,j,i) = delta_P/gm1/pmy_mesh->dt;
        edot_floor(k,j,i) = delta_P_floor/gm1/pmy_mesh->dt;

        // apply floor if cooled too much
        Real P_after = std::max(P_next,P_floor);
        Real u_after = P_after/gm1;

        // change internal energy
        u_e = u_after + e_non_thermal;
        w_p = P_after;
      }
    }
  }

  // sum up cooling only done in the active cells
  AthenaArray<Real> vol(ncells1);
  Real delta_e_block = 0.0, delta_ef_block = 0.0;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      pcoord->CellVolume(k, j, is, ie, vol);
#pragma omp simd reduction(+:delta_e_block,delta_ef_block)
      for (int i = is; i <= ie; ++i) {
        delta_e_block += edot(k,j,i)*pmy_mesh->dt*vol(i);
        delta_ef_block += edot_floor(k,j,i)*pmy_mesh->dt*vol(i);
      }
    }
  }
  // add cooling and ceiling to hist outputs
  ruser_meshblock_data[0](0) += delta_e_block;
  ruser_meshblock_data[0](1) += delta_ef_block;

  return;
}

//========================================================================================
//! \fn Real Real CoolingExplicitSubcycling(Real tend, Real P, const Real rho)
//! \brief explicit cooling solver from 0 to tend with subcycling
//========================================================================================
static Real CoolingExplicitSubcycling(Real tend, Real P, const Real rho) {
  Real tnow = 0., tleft = tend;
  Real T1 = P/rho*pcool->punit->Temperature;
  int nsub_max = pcool->cfl_cool/cfl_op_cool*10;
  for (int i=0; i<nsub_max; ++i) {
    Real dt_net = cfl_op_cool*dtnet(pcool, rho, P);
    Real dt_sub = std::min(std::min(tend,dt_net),tleft);

    T1 *= (1-dt_sub/tcool(pcool, rho, P));
    P = rho*T1/pcool->punit->Temperature;

    tnow += dt_sub;
    tleft = tend-tnow;

    if (tnow >= tend) break;
  }

  if (tnow < tend)
    std::cout << "Too many substeps required: tnow = " << tnow
              << " tend = " << tend << std::endl;
  return P;
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
//! \fn void AddSupernova(Mesh *pm)
//! \brief add a SN at the center
//========================================================================================
void AddSupernova(Mesh *pm) {
  Real my_vol = 0;

  // Add SN
  for (int b=0; b<pm->nblocal; ++b){
    MeshBlock *pmb = pm->my_blocks(b);
    // Initialize primitive values
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real r = std::sqrt(SQR(pmb->pcoord->x1v(i))
                            +SQR(pmb->pcoord->x2v(j))
                            +SQR(pmb->pcoord->x3v(k)));
          if (r<r_SN) my_vol += pmb->pcoord->GetCellVolume(k,j,i);
        }
      }
    }
  }

  // calculate total feedback region volume
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &my_vol, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // get pressure fron SNe in the code unit
  Real usn = E_SN*pcool->punit->Bethe_in_code/my_vol;

  // add the SN energy
  for (int b=0; b<pm->nblocal; ++b){
    MeshBlock *pmb = pm->my_blocks(b);
    Hydro *phydro = pmb->phydro;
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real r = std::sqrt(SQR(pmb->pcoord->x1v(i))
                            +SQR(pmb->pcoord->x2v(j))
                            +SQR(pmb->pcoord->x3v(k)));
          if (r<r_SN) phydro->u(IEN,k,j,i) += usn;
        }
      }
    }
  }
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
//! \fn Real HistoryMass(MeshBlock *pmb, int iout)
//! \brief Total gas mass (total/hot)
//========================================================================================
Real HistoryMass(MeshBlock *pmb, int iout) {
  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> vol(pmb->ncells1);
  AthenaArray<Real> rho, press;
  rho.InitWithShallowSlice(pmb->phydro->u,4,IDN,1);
  press.InitWithShallowSlice(pmb->phydro->w,4,IPR,1);
  Real mass=0.0;
  Real Temp;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i=is; i<=ie; ++i) {
        if (iout == i_M_hot) {
          Temp = pcool->GetTemperature(rho(k,j,i), press(k,j,i));
          if (Temp > Thot0) {
            mass += rho(k,j,i)*vol(i);
          }
        } else {
          mass += rho(k,j,i)*vol(i);
        }
      }
    }
  }
  return mass;
}

//========================================================================================
//! \fn Real HistoryEnergy(MeshBlock *pmb, int iout)
//! \brief Total gas mass (total/hot)
//========================================================================================
Real HistoryEnergy(MeshBlock *pmb, int iout) {
  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> vol(pmb->ncells1);
  AthenaArray<Real> rho,press;
  rho.InitWithShallowSlice(pmb->phydro->w,4,IDN,1);
  press.InitWithShallowSlice(pmb->phydro->w,4,IPR,1);
  Real energy=0.0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i=is; i<=ie; ++i) {
        Real eint = press(k,j,i)*vol(i)/(pmb->peos->GetGamma()-1);
        if (iout == i_e_hot) {
          Real Temp = pcool->GetTemperature(rho(k,j,i), press(k,j,i));
          if (Temp > Thot0) energy += eint;
        } else {
          energy += eint;
        }
      }
    }
  }
  return energy;
}

//========================================================================================
//! \fn Real HistoryMass(MeshBlock *pmb, int iout)
//! \brief Total gas mass
//========================================================================================
Real HistoryRadialMomentum(MeshBlock *pmb, int iout) {
  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> vol(pmb->ncells1);

  Real pr = 0;
  Real x0 = 0.0, y0 = 0.0, z0 = 0.0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i=is; i<=ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        Real y = pmb->pcoord->x2v(j);
        Real z = pmb->pcoord->x3v(k);
        Real Mx = pmb->phydro->u(IM1,k,j,i);
        Real My = pmb->phydro->u(IM2,k,j,i);
        Real Mz = pmb->phydro->u(IM3,k,j,i);
        Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        pr += vol(i)*(Mx*(x-x0) + My*(y-y0) + Mz*(z-z0))/rad;
      }
    }
  }

  return pr;

}

//========================================================================================
//! \fn Real HistoryMass(MeshBlock *pmb, int iout)
//! \brief Total gas mass (test)
//========================================================================================
Real HistoryShell(MeshBlock *pmb, int iout) {
  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> vol(pmb->ncells1);
  AthenaArray<Real> rho, press;
  rho.InitWithShallowSlice(pmb->phydro->u,4,IDN,1);
  press.InitWithShallowSlice(pmb->phydro->w,4,IPR,1);

  Real sum = 0.0;
  Real x0 = 0.0, y0 = 0.0, z0 = 0.0;
  Real Temp;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
      for (int i=is; i<=ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        Real y = pmb->pcoord->x2v(j);
        Real z = pmb->pcoord->x3v(k);

        Real vx = pmb->phydro->w(IVX,k,j,i);
        Real vy = pmb->phydro->w(IVY,k,j,i);
        Real vz = pmb->phydro->w(IVZ,k,j,i);
        Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        Real vr = (vx*(x-x0) + vy*(y-y0) + vz*(z-z0))/rad;
        Temp = pcool->GetTemperature(rho(k,j,i), press(k,j,i));
        if ((Temp < Thot0) && (vr > vr0)) {
          if (iout == i_M_sh) { // Mass
            sum += rho(k,j,i)*vol(i);
          } else if (iout == i_pr_sh) { // Momentum
            sum += rho(k,j,i)*vr*vol(i);
          } else if (iout == i_RM_sh) {
            // mass-weighted radius (need to be divided by shell mass)
            sum += rho(k,j,i)*vol(i)*rad;
          }
        }
      }
    }
  }
  return sum;
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
