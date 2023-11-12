//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file radiative_snr.cpp
//! \brief Problem generator for radiative snr without conduction
//========================================================================================

// C++ headers
#include <algorithm> // min()
#include <cmath>     // abs(), pow(), sqrt()
#include <fstream>   // ofstream
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // string

// Athena++ headers
#include "../athena.hpp"                  // macros, enums, declarations
#include "../athena_arrays.hpp"           // AthenaArray
#include "../coordinates/coordinates.hpp" // Coordinates
#include "../eos/eos.hpp"                 // EquationOfState
#include "../fft/perturbation.hpp"        // PerturbationGenerator
#include "../field/field.hpp"             // Field
#include "../globals.hpp"                 // Globals
#include "../hydro/hydro.hpp"             // Hydro
#include "../mesh/mesh.hpp"
#include "../microphysics/cooling.hpp" // CoolingSolver
#include "../parameter_input.hpp"      // ParameterInput

// Global variables ---
// CoolingSolver *pcool;

// user function
void AddSupernova(Mesh *pm);
void CollectCounters(Mesh *pm);

// user history
Real CoolingLosses(MeshBlock *pmb, int iout);
Real HistoryMass(MeshBlock *pmb, int iout);
Real HistoryEnergy(MeshBlock *pmb, int iout);
Real HistoryRadialMomentum(MeshBlock *pmb, int iout);
Real HistoryShell(MeshBlock *pmb, int iout);

// SN history related parameters
Real Thot0 = 2.e4, vr0 = 0.1;
int i_M_hot, i_e_hot, i_M_sh, i_pr_sh, i_RM_sh; // indicies for history

// SN related parameters
Real r_SN, M_ej, E_SN, t_SN, dt_SN;
//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also
//! be used to initialize variables which are global to (and therefore can be
//! passed to) other functions in this file.  Called in Mesh constructor.
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Initialize Cooling Solver
  // pcool = new CoolingSolver(pin);
  // Enroll source function
  if (cooling) {
    std::string cooling_type = pin->GetString("cooling", "cooling");
    if (cooling_type.compare("enroll") == 0) {
      EnrollUserExplicitSourceFunction(&CoolingSolver::CoolingSourceTerm);
      // Enroll timestep so that dt <= cfl_cool * min(t_cool)
      EnrollUserTimeStepFunction(&CoolingSolver::CoolingTimeStep);
      if (Globals::my_rank == 0) {
        std::cout << "Cooling solver is enrolled" << std::endl;
      }
    } else if (cooling_type.compare("op_split") == 0) {
      if (Globals::my_rank == 0)
        std::cout << "Cooling solver is set to operator split" << std::endl;
    }
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in ProblemGenerator" << std::endl
        << "Cooling must be turned on" << std::endl;
    ATHENA_ERROR(msg);
  }

  // SN related parameters
  t_SN = pin->GetOrAddReal("problem", "t_SN", 0.0); // when to explode SN
  if (t_SN >= 0.0) {
    r_SN = pin->GetReal("problem", "r_SN");
    E_SN = pin->GetOrAddReal("problem", "E_SN", 1.0);
    M_ej = pin->GetOrAddReal("problem", "M_ej", 10.0);
    dt_SN = pin->GetOrAddReal("problem", "dt_SN", -1); // interval of SNe
  }

  // Enroll user-defined functions
  int n_user_hst = 11, i_user_hst = 0;
  AllocateUserHistoryOutput(n_user_hst);

  // these should come first
  EnrollUserHistoryOutput(i_user_hst++, CoolingLosses, "e_cool");
  EnrollUserHistoryOutput(i_user_hst++, CoolingLosses, "e_floor");
  EnrollUserHistoryOutput(i_user_hst++, CoolingLosses, "e_floor2");
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
//! \brief Function to initialize problem-specific data in MeshBlock class.  Can
//! also be used to initialize variables which are global to other functions in
//! this file. Called in MeshBlock constructor before ProblemGenerator.
//========================================================================================
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // Allocate storage for keeping track of cooling
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(3);
  ruser_meshblock_data[0](0) = 0.0; // total e_cool between history dumps
  ruser_meshblock_data[0](1) =
      0.0; // total e_floor in coolsolver between history dumps
  ruser_meshblock_data[0](2) =
      0.0; // total e_floor in cons2prim between history dumps

  // Set output variables
  int num_user_variables = 3; // for edot bookkeeping
  // instantanoues e_dot, e_dot_floor, efloor (in eos)
  AllocateUserOutputVariables(num_user_variables);

  // initialize arrays for bookkeeping
  // shallow copy existing arrays
  pcool->edot.InitWithShallowSlice(user_out_var, 4, 0, 1);
  pcool->edot_floor.InitWithShallowSlice(user_out_var, 4, 1, 1);
  pcool->bookkeeping = true;

  peos->efloor.InitWithShallowSlice(user_out_var, 4, 2, 1);
  peos->bookkeeping = true;

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

  Real nH_0 = pin->GetReal("problem", "nH_0");     // measured in m_p muH cm^-3
  Real pgas_0 = pin->GetReal("problem", "pgas_0"); // measured in kB K cm^-3

  Real rho_0 = nH_0 * pcool->pcf->nH_to_code_den; // to code units
  pgas_0 *= pcool->pcf->pok_to_code_press;        // to code units

  Real gamma = peos->GetGamma();
  Real gm1 = gamma - 1.0;

  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        phydro->u(IDN, k, j, i) = rho_0;
        phydro->u(IEN, k, j, i) = pgas_0 / gm1;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
      }
    }
  }

  // initialize interface B and total energy
  if (MAGNETIC_FIELDS_ENABLED) {
    Real b0 = pin->GetReal("problem", "B_0"); // in uG
    b0 *= 1.e-6 / punit->MagneticField;
    std::cout << "B0 (code) = " << b0 << std::endl;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie + 1; ++i) {
          pfield->b.x1f(k, j, i) = 0.0;
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je + 1; ++j) {
        for (int i = is; i <= ie; ++i) {
          pfield->b.x2f(k, j, i) = b0;
        }
      }
    }
    for (int k = ks; k <= ke + 1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          pfield->b.x3f(k, j, i) = 0.0;
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          phydro->u(IEN, k, j, i) += 0.5 * b0 * b0;
        }
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
    t_SN = time + dt_SN;
    if (Globals::my_rank == 0)
      std::cout << "SN exploded at " << time << " --> next SN will be at "
                << t_SN << std::endl;
  }

  // Add density perturbation
  Real nH_0 = pin->GetReal("problem", "nH_0"); // measured in m_p muH cm^-3
  Real rho_0 = nH_0 * pcool->pcf->nH_to_code_den;
  Real amp_den = pin->GetOrAddReal("problem", "amp_den", 0.0);
  if (amp_den > 0) {
    PerturbationGenerator *ppert;
    ppert = new PerturbationGenerator(this, pin);

    ppert->GenerateScalar(); // generate a scalar in Fourier space
    ppert->AssignScalar();   // do backward FFT and assign the scalar array
    // calculate total
    Real dtot = 0.0, d2tot = 0.0;
    for (int nb = 0; nb < nblocal; ++nb) {
      MeshBlock *pmb = my_blocks(nb);
      AthenaArray<Real> dden(ppert->GetScalar(nb));
      for (int k = pmb->ks; k <= pmb->ke; k++) {
        for (int j = pmb->js; j <= pmb->je; j++) {
          for (int i = pmb->is; i <= pmb->ie; i++) {
            dtot += dden(k, j, i);
            d2tot += SQR(dden(k, j, i));
          }
        }
      }
    }
#ifdef MPI_PARALLEL
    int mpierr;
    // Sum the perturbations over all processors
    mpierr = MPI_Allreduce(MPI_IN_PLACE, &dtot, 1, MPI_DOUBLE, MPI_SUM,
                           MPI_COMM_WORLD);
    mpierr = MPI_Allreduce(MPI_IN_PLACE, &d2tot, 1, MPI_DOUBLE, MPI_SUM,
                           MPI_COMM_WORLD);
#endif
    d2tot /= GetTotalCells();
    dtot /= GetTotalCells();
    Real d_std = std::sqrt(d2tot - dtot * dtot);

    // assign normalized density perturbation with assigned amplitude
    for (int nb = 0; nb < nblocal; ++nb) {
      MeshBlock *pmb = my_blocks(nb);
      Hydro *phydro = pmb->phydro;
      AthenaArray<Real> dden(ppert->GetScalar(nb));
      for (int k = pmb->ks; k <= pmb->ke; k++) {
        for (int j = pmb->js; j <= pmb->je; j++) {
          for (int i = pmb->is; i <= pmb->ie; i++) {
            phydro->u(IDN, k, j, i) +=
                rho_0 * (amp_den / d_std) * dden(k, j, i);
          }
        }
      }
    }

    delete ppert;
  }

  // velocity perturbation; decaying turbulence
  Real turb_flag(pin->GetOrAddInteger("problem", "turb_flag", 0));
  if (turb_flag > 0) {
    TurbulenceDriver *ptrbd;
    ptrbd = new TurbulenceDriver(this, pin);
    Real ek = 0.5 * rho_0 * SQR(pin->GetReal("problem", "v3d")); // alpha=Ek/Eth
    Real vol = mesh_size.x1len * mesh_size.x2len * mesh_size.x3len;
    ptrbd->dedt = ek * vol;
    pin->SetReal("problem", "dedt", ek * vol);
    if (res_flag == 0)
      ptrbd->Driving();
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//! \brief Function called once every time step for Mesh-level user-defined
//! work.
//========================================================================================
void Mesh::UserWorkInLoop() {
  if ((t_SN > 0) && (t_SN < time)) {
    AddSupernova(this);
    t_SN += dt_SN;
    if (Globals::my_rank == 0)
      std::cout << "SN exploded at " << time << " --> next SN will be at "
                << t_SN << std::endl;
  }
  // check number of bad cells
  CollectCounters(this);
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================
void MeshBlock::UserWorkInLoop() {
  if (pcool->op_flag)
    pcool->OperatorSplitSolver(this);
  // sum up energy lost/gain through cooling and flooring in cooling solver
  if (pcool->bookkeeping) {
    AthenaArray<Real> vol(ncells1);
    Real delta_e_cool = 0.0, delta_e_floor = 0.0, delta_e_floor2 = 0.0;
    Real dt = pmy_mesh->dt;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        pcoord->CellVolume(k, j, is, ie, vol);
#pragma omp simd reduction(+ : delta_e_cool, delta_e_floor, delta_e_floor2)
        for (int i = is; i <= ie; ++i) {
          delta_e_cool += pcool->edot(k, j, i) * dt * vol(i);
          delta_e_floor += pcool->edot_floor(k, j, i) * dt * vol(i);
          delta_e_floor2 += peos->efloor(k, j, i) * vol(i);
        }
      }
    }
    ruser_meshblock_data[0](0) += delta_e_cool;
    ruser_meshblock_data[0](1) += delta_e_floor;
    ruser_meshblock_data[0](2) += delta_e_floor2;
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
//! \brief Function called before generating output files
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  pin->SetReal("problem", "t_SN", t_SN); // update t_SN in restart files
}

//========================================================================================
//! \fn void AddSupernova(Mesh *pm)
//! \brief add a SN at the center
//========================================================================================
void AddSupernova(Mesh *pm) {
  Real my_vol = 0;

  // Add SN
  for (int b = 0; b < pm->nblocal; ++b) {
    MeshBlock *pmb = pm->my_blocks(b);
    // Initialize primitive values
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real r =
              std::sqrt(SQR(pmb->pcoord->x1v(i)) + SQR(pmb->pcoord->x2v(j)) +
                        SQR(pmb->pcoord->x3v(k)));
          if (r < r_SN)
            my_vol += pmb->pcoord->GetCellVolume(k, j, i);
        }
      }
    }
  }

  // calculate total feedback region volume
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &my_vol, 1, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  Units *punit = pm->punit;
  // get pressure fron SNe in the code unit
  Real rhosn = M_ej * punit->Msun_in_code / my_vol;
  Real usn = E_SN * punit->Bethe_in_code / my_vol;

  // add the SN energy
  for (int b = 0; b < pm->nblocal; ++b) {
    MeshBlock *pmb = pm->my_blocks(b);
    Hydro *phydro = pmb->phydro;
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real r =
              std::sqrt(SQR(pmb->pcoord->x1v(i)) + SQR(pmb->pcoord->x2v(j)) +
                        SQR(pmb->pcoord->x3v(k)));
          if (r < r_SN) {
            phydro->u(IDN, k, j, i) += rhosn;
            phydro->u(IEN, k, j, i) += usn;
          }
        }
      }
    }
  }
}

//========================================================================================
//! \fn void CollectCounters(Mesh *pm)
//! \brief collect counters
//========================================================================================
void CollectCounters(Mesh *pm) {
  int nbad_d = 0, nbad_p = 0;

  // summing up over meshblocks within the rank
  for (int b = 0; b < pm->nblocal; ++b) {
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
    if (nbad_p > 0)
      std::cerr << nbad_p << " cells had negative pressure" << std::endl;
    if (nbad_d > 0)
      std::cerr << nbad_d << " cells had negative density" << std::endl;
  }
}

//========================================================================================
//! \fn Real CoolingLosses(MeshBlock *pmb, int iout)
//! \brief Cooling losses for history variable
//!        return sum of all energy losses due to different cooling mechanisms
//!        and resets time-integrated values to 0 for the next step
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
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> vol(pmb->ncells1);
  AthenaArray<Real> rho, press;
  rho.InitWithShallowSlice(pmb->phydro->u, 4, IDN, 1);
  press.InitWithShallowSlice(pmb->phydro->w, 4, IPR, 1);
  Real mass = 0.0;
  Real Temp;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i = is; i <= ie; ++i) {
        if (iout == i_M_hot) {
          Temp = pmb->pcool->pcf->GetTemperature(rho(k, j, i), press(k, j, i));
          if (Temp > Thot0) {
            mass += rho(k, j, i) * vol(i);
          }
        } else {
          mass += rho(k, j, i) * vol(i);
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
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> vol(pmb->ncells1);
  AthenaArray<Real> rho, press;
  rho.InitWithShallowSlice(pmb->phydro->w, 4, IDN, 1);
  press.InitWithShallowSlice(pmb->phydro->w, 4, IPR, 1);
  Real energy = 0.0;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i = is; i <= ie; ++i) {
        Real eint = press(k, j, i) * vol(i) / (pmb->peos->GetGamma() - 1);
        if (iout == i_e_hot) {
          Real Temp =
              pmb->pcool->pcf->GetTemperature(rho(k, j, i), press(k, j, i));
          if (Temp > Thot0)
            energy += eint;
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
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> vol(pmb->ncells1);

  Real pr = 0;
  Real x0 = 0.0, y0 = 0.0, z0 = 0.0;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i = is; i <= ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        Real y = pmb->pcoord->x2v(j);
        Real z = pmb->pcoord->x3v(k);
        Real Mx = pmb->phydro->u(IM1, k, j, i);
        Real My = pmb->phydro->u(IM2, k, j, i);
        Real Mz = pmb->phydro->u(IM3, k, j, i);
        Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        pr += vol(i) * (Mx * (x - x0) + My * (y - y0) + Mz * (z - z0)) / rad;
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
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> vol(pmb->ncells1);
  AthenaArray<Real> rho, press;
  rho.InitWithShallowSlice(pmb->phydro->u, 4, IDN, 1);
  press.InitWithShallowSlice(pmb->phydro->w, 4, IPR, 1);

  Real sum = 0.0;
  Real x0 = 0.0, y0 = 0.0, z0 = 0.0;
  Real Temp;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
      for (int i = is; i <= ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        Real y = pmb->pcoord->x2v(j);
        Real z = pmb->pcoord->x3v(k);
        Real vx = pmb->phydro->w(IVX, k, j, i);
        Real vy = pmb->phydro->w(IVY, k, j, i);
        Real vz = pmb->phydro->w(IVZ, k, j, i);
        Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        Real vr = (vx * (x - x0) + vy * (y - y0) + vz * (z - z0)) / rad;
        Temp = pmb->pcool->pcf->GetTemperature(rho(k, j, i), press(k, j, i));
        if ((Temp < Thot0) && (vr > vr0)) {
          if (iout == i_M_sh) { // Mass
            sum += rho(k, j, i) * vol(i);
          } else if (iout == i_pr_sh) { // Momentum
            sum += rho(k, j, i) * vr * vol(i);
          } else if (iout == i_RM_sh) {
            // mass-weighted radius (need to be divided by shell mass)
            sum += rho(k, j, i) * vol(i) * rad;
          }
        }
      }
    }
  }
  return sum;
}
