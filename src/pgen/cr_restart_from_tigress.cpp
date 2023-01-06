//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cr_restart_from_tigress.cpp
//! \brief Problem generator to study the propagation of cosmic rays in the galactic
//! environment reproduced in TIGRESS. MHD quantities are initialised by reading in
//! athena vtk files.
//======================================================================================

// C++ headers
#include <algorithm>  // min, max
#include <cmath>
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../cr/cr.hpp"
#include "../cr/integrators/cr_integrators.hpp"
#include "../defs.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../microphysics/cooling.hpp"
#include "../microphysics/units.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

#define MAXLEN 256

// Global variables ---
CoolingSolver *pcool;

// total rate of CR injection at a given time step
static Real tot_InjectionRate;

//star particle structure
struct StarParS {
  int id;
  int merge_history;
  int isnew;
  int active;
  Real m;
  Real x1;
  Real x2;
  Real x3;
  Real v1;
  Real v2;
  Real v3;
  Real age;
  Real mage;
  Real mdot;
};
static Real age_th; //maximum age of stars exploding as supernovae
static int inj_cells; //parameter to calculate the distribution of injected CR energy

//function to split a string into a vector
static std::vector<std::string> split(std::string str, char delimiter);
//function to get rid of white space leading/trailing a string
static void trim(std::string &s);
static void ath_bswap(void *vdat, int len, int cnt);

//function to calculate the gas temperature
void TempCalculation(Units *punit, Real rho, Real Press, Real &Temp, Real &mu, Real &muH);

//functions for the injection of CR energy density
void CalculateInjectionRate(ParameterInput *pin, MeshBlock *pmb,
                            AthenaArray<Real> &CRInjectionRate);
void Source_CR(MeshBlock *pmb, Real time, Real dt,
               const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
               AthenaArray<Real> &u_cr, AthenaArray<Real> &CRInjectionRate);
Real set_CR_Luminosity(const Real tage);
static inline Real lin_interpol(const Real x, const Real xi,
                    const Real xi1,const Real yi, const Real yi1);

//functions to read data field from vtk files
static void read_starpar_vtk(MeshBlock *mb, std::string filename,
                             std::vector<StarParS> &pList);
static void read_vtk(MeshBlock *mb, std::string vtkdir, std::string vtkfile0,
  std::string field, int component, AthenaArray<Real> &data);

void Outflow_down(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je,
     int ks, int ke, int ngh);
void CROutflow_down(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie,
    int js, int je, int ks, int ke, int ngh);
void Outflow_up(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je,
     int ks, int ke, int ngh);
void CROutflow_up(MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie,
    int js, int je, int ks, int ke, int ngh);

// function to calculate the total rate of injected CR
//energy density -- for the hystory file
static Real hst_Injected_CRenergy(MeshBlock *pmb, int iout) {
  Real Inj_en = 0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        Inj_en += pmb->pcr->CRInjectionRate(k,j,i);
      }
    }
  }
  return Inj_en;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  pcool = new CoolingSolver(pin);
  // Enroll source function
  if (!pcool->op_flag) {
    EnrollUserExplicitSourceFunction(&CoolingSolver::CoolingEuler);
    std::cout << "Cooling solver is enrolled" << std::endl;
  } else {
    std::cout << "Cooling solver is set to operator split" << std::endl;
  }

  // Enroll timestep so that dt <= min t_cool
  EnrollUserTimeStepFunction(&CoolingSolver::CoolingTimeStep);

  //CR energy density is injected as a consequence of supernova explosion
  //maximum age of stars exploding as supernovae
  age_th = pin->GetOrAddReal("problem", "young_stars_age", 0);
  //parameter to calculate the distribution of injected CR energy
  inj_cells = pin->GetOrAddInteger("problem", "inj_cells", 0);

  // User boundary conditions
  EnrollUserBoundaryFunction(inner_x3, Outflow_down);
  EnrollUserBoundaryFunction(outer_x3, Outflow_up);
  if(CR_ENABLED) {
    EnrollUserCRBoundaryFunction(inner_x3, CROutflow_down);
    EnrollUserCRBoundaryFunction(outer_x3, CROutflow_up);
    AllocateUserHistoryOutput(1);
    EnrollUserHistoryOutput(0, hst_Injected_CRenergy,
              "CRInjEn", UserHistoryOperation::sum);
  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if(CR_ENABLED) {
    Units *punit = pcool->punit;
    pcr->punit = pcool->punit;
    CalculateInjectionRate(pin,this,pcr->CRInjectionRate);
    pcr->EnrollUserCRSource(Source_CR);
    pcr->EnrollTemperatureFunction(TempCalculation);
    pcr->sigma = pin->GetOrAddReal("cr","sigma",1.0);
    pcr->sigma *= pcr->vmax;
    pcr->sigma *= punit->second/(punit->cm*punit->cm);
    pcr->lambdac = pin->GetOrAddReal("cr","lambdac",5.3e-16);
    pcr->lambdac /= punit->second;
  }
}

void CalculateInjectionRate(ParameterInput *pin, MeshBlock *pmb,
         AthenaArray<Real> &CRInjectionRate) {
  int ks=pmb->ks, ke=pmb->ke;
  int js=pmb->js, je=pmb->je;
  int is=pmb->is, ie=pmb->ie;
  int ierr;
  size_t Nstars(0);

  std::vector<StarParS> pList;

  //In each cell the injection rate of CR energy density is inizialised to zero
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        CRInjectionRate(k,j,i) = 0.0;
      }
    }
  }

  //vtk athena file containing information about star particles
  std::string starparfile = pin->GetString("problem", "star_vtkfile");
  //Reading the star-particle information from the vtk file
  if (Globals::my_rank == 0) {
    if (pmb->loc.lx1 == 0 && pmb->loc.lx2 == 0 && pmb->loc.lx3 == 0)
      std::cout<<"Reading star particle vtk file..."<<std::endl;
    read_starpar_vtk(pmb,starparfile,pList);
    Nstars = pList.size();
    if (pmb->loc.lx1 == 0 && pmb->loc.lx2 == 0 && pmb->loc.lx3 == 0)
      std::cout<<"Number of stars = "<<Nstars<<std::endl;
  }

  #ifdef MPI_PARALLEL
  ierr = MPI_Bcast(&Nstars, 1, MPI_INT, 0, MPI_COMM_WORLD);
  pList.resize(Nstars);
  ierr = MPI_Bcast(&pList[0], Nstars*sizeof(struct StarParS),
                MPI_BYTE, 0, MPI_COMM_WORLD);
  #endif


  //The amount of CR luminosity injected from each star cluster follows
  //a Gaussian distribution -- here we calculate the standard devation
  //depending on the value of inj_cells
  Real cell_x = pmb->pcoord->x1f(1) - pmb->pcoord->x1f(0);
  Real cell_y = pmb->pcoord->x2f(1) - pmb->pcoord->x2f(0);
  Real cell_z = pmb->pcoord->x3f(1) - pmb->pcoord->x3f(0);
  Real inj_R = inj_cells * cell_x;
  Real cell_volume = cell_x * cell_y * cell_z;
  Real inj_volume = 0.;
  //Calculate the volume within a sphere with radius inj_R
  for(int k=-inj_cells; k<=inj_cells; ++k) {
    for(int j=-inj_cells; j<=inj_cells; ++j) {
      for(int i=-inj_cells; i<=inj_cells; ++i) {
        Real xx = i * cell_x;
        Real yy = j * cell_y;
        Real zz = k * cell_z;
        Real R2 = std::pow(xx,2) + std::pow(yy,2) + std::pow(zz,2);
        Real R = std::sqrt(R2);
        if (R <= inj_R) inj_volume += cell_volume;
      }
    }
  }
  // sigma is calculated in order for the Gaussian distribution
  //to occupy the same volume and to have the same amplitude as
  //a uniform spherical distribution with injection radius inj_R
  Real sigma = std::pow(inj_volume/(2.*PI),1./3.)*std::pow(2*PI,-1./6.);

  int tot_young_stars = 0;
  tot_InjectionRate = 0.;
  Real InjectionRate;
  Real InjectionRate_in_code;
  Units *punit = pcool->punit;
  for(size_t s=0; s<Nstars; ++s) {
    // convert the star age from code units in Myr
    Real tstar = pList[s].mage / punit->Myr_in_code;
    if (tstar<age_th && pList[s].m>0.) {
      tot_young_stars++;
      //Calculate the CR luminosity produced by each star cluster
      InjectionRate =  set_CR_Luminosity(tstar) * pList[s].m
                       / (punit->Msun_in_code * punit->erg); //in cgs units
      InjectionRate_in_code = InjectionRate / punit->Myr_in_code; //in code units
      tot_InjectionRate += InjectionRate_in_code;

      //Calculate the cell-centered grid position associated to each star cluster
      Real sign_xs = pList[s].x1 / std::abs(pList[s].x1);
      Real sign_ys = pList[s].x2 / std::abs(pList[s].x2);
      Real sign_zs = pList[s].x3 / std::abs(pList[s].x3);
      Real grid_xs = static_cast<int>(pList[s].x1/cell_x) *
                     cell_x + sign_xs * 0.5 * cell_x;
      Real grid_ys = static_cast<int>(pList[s].x2/cell_y) *
                     cell_y + sign_ys * 0.5 * cell_y;
      Real grid_zs = static_cast<int>(pList[s].x3/cell_z) *
                     cell_z + sign_zs * 0.5 * cell_z;

      for(int k=ks; k<=ke; ++k) {
        for(int j=js; j<=je; ++j) {
          for(int i=is; i<=ie; ++i) {
            Real R2 = std::pow(pmb->pcoord->x1v(i)-grid_xs,2)
                    + std::pow(pmb->pcoord->x2v(j)-grid_ys,2)
                    + std::pow(pmb->pcoord->x3v(k)-grid_zs,2);
            Real R = std::sqrt(R2);
            CRInjectionRate(k,j,i) += InjectionRate_in_code/inj_volume
              * exp(-R2/(2.*sigma*sigma)); //gaussian distribution

            // This is done for periodic boundaries only
            Real Rper1, Rper2;
            if (pmb->pmy_mesh->mesh_bcs[BoundaryFace::inner_x1]
               == BoundaryFlag::periodic
            && pmb->pmy_mesh->mesh_bcs[BoundaryFace::outer_x1]
               == BoundaryFlag::periodic) {
              Real Bound_x1 = pmb->pmy_mesh->mesh_size.x1max;
              if (grid_xs<0)
                Rper1 = std::pow(-2*Bound_x1+pmb->pcoord->x1v(i)-grid_xs,2)
                      + std::pow(pmb->pcoord->x2v(j)-grid_ys,2)
                      + std::pow(pmb->pcoord->x3v(k)-grid_zs,2);
              if (grid_xs>=0)
                Rper1 = std::pow(2*Bound_x1+pmb->pcoord->x1v(i)-grid_xs,2)
                      + std::pow(pmb->pcoord->x2v(j)-grid_ys,2)
                      + std::pow(pmb->pcoord->x3v(k)-grid_zs,2);
              CRInjectionRate(k,j,i) += InjectionRate_in_code/inj_volume
                                     * exp(-Rper1/(2.*sigma*sigma));
            }
            if (pmb->pmy_mesh->mesh_bcs[BoundaryFace::inner_x2]
               == BoundaryFlag::periodic
            && pmb->pmy_mesh->mesh_bcs[BoundaryFace::outer_x2]
               == BoundaryFlag::periodic) {
              Real Bound_x2 = pmb->pmy_mesh->mesh_size.x2max;
              if (grid_ys<0)
                Rper2 = std::pow(pmb->pcoord->x1v(i)-grid_xs,2)
                      + std::pow(-2*Bound_x2+pmb->pcoord->x2v(j)-grid_ys,2)
                      + std::pow(pmb->pcoord->x3v(k)-grid_zs,2);
              if (grid_ys>=0)
                Rper2 = std::pow(pmb->pcoord->x1v(i)-grid_xs,2)
                      + std::pow(2*Bound_x2+pmb->pcoord->x2v(j)-grid_ys,2)
                      + std::pow(pmb->pcoord->x3v(k)-grid_zs,2);
              CRInjectionRate(k,j,i) += InjectionRate_in_code/inj_volume
                                     * exp(-(Rper2)/(2.*sigma*sigma));
            }
          }
        }
      }// end k,j,i
    }
  }//end star-particle cycle
  if (pmb->loc.lx1 == 0 && pmb->loc.lx2 == 0 && pmb->loc.lx3 == 0)
    std::cout<<"Number of young stars = "<<tot_young_stars<<std::endl;
}

void Source_CR(MeshBlock *pmb, Real time, Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
              AthenaArray<Real> &u_cr, AthenaArray<Real> &CRInjectionRate) {
  int ks=pmb->ks, ke=pmb->ke;
  int js=pmb->js, je=pmb->je;
  int is=pmb->is, ie=pmb->ie;
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        u_cr(CRE,k,j,i) += CRInjectionRate(k,j,i) * dt;
      }
    }
  }
}

void TempCalculation(Units *punit, Real rho, Real Press,
                     Real &Temp, Real &mu, Real &muH) {
  Temp = pcool->pcf->GetTemperature(rho, Press);
  muH = pcool->pcf->Get_muH();
  mu = pcool->pcf->Get_mu(rho, Press);
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator to initialize mesh by reading in athena vtk files
//!
//! \note This Problem Generator can be used only to read vtk files corresponding to
//! times t that are integer multiples of 1/Omega, where Omega is the Galactic Rotation
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  int ierr;

  //! Path for the directory where all the id folders are stored
  int a;
  std::string vtkdir = pin->GetString("problem", "vtk_directory");
  //! Name of the file in the id0 folder
  std::string vtkfile0 = pin->GetString("problem", "vtk_file");

  //dimensions of meshblock
  const int Nx = ie - is + 1;
  const int Ny = je - js + 1;
  const int Nz = ke - ks + 1;

  //dimensions of mesh
  const int Nx_mesh = pmy_mesh->mesh_size.nx1;
  const int Ny_mesh = pmy_mesh->mesh_size.nx2;
  const int Nz_mesh = pmy_mesh->mesh_size.nx3;

  int nsize = static_cast<int>(pmy_mesh->GetTotalCells());

  AthenaArray<Real> data; //temporary array to store data of the entire mesh
  data.NewAthenaArray(Nz_mesh, Ny_mesh, Nx_mesh);
  AthenaArray<Real> b; //needed for PrimitiveToConserved()
  b.NewAthenaArray(Nz,Ny,Nz);

  int gis = static_cast<int>(loc.lx1) * Nx;
  int gjs = static_cast<int>(loc.lx2) * Ny;
  int gks = static_cast<int>(loc.lx3) * Nz;

  if (Globals::my_rank == 0) {
    if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0)
      printf("Reading density ... \n");
    read_vtk(this, vtkdir, vtkfile0, "density", 0, data);
  }

#ifdef MPI_PARALLEL
  ierr = MPI_Bcast(data.data(), nsize, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

  for (int k=ks; k<=ke; ++k)
    for (int j=js; j<=je; ++j)
      for (int i=is; i<=ie; ++i)
        phydro->w(IDN, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);


  if (Globals::my_rank == 0) {
    if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading pressure ... \n");
    read_vtk(this, vtkdir, vtkfile0, "pressure", 0, data);
  }

#ifdef MPI_PARALLEL
  ierr = MPI_Bcast(data.data(), nsize, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

  for (int k=ks; k<=ke; ++k)
    for (int j=js; j<=je; ++j)
      for (int i=is; i<=ie; ++i)
        phydro->w(IPR, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);


  if (Globals::my_rank == 0) {
    if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0)
      printf("Reading x1-velocity ... \n");
    read_vtk(this, vtkdir, vtkfile0, "velocity", 0, data);
  }

#ifdef MPI_PARALLEL
  ierr = MPI_Bcast(data.data(), nsize, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

  for (int k=ks; k<=ke; ++k)
    for (int j=js; j<=je; ++j)
      for (int i=is; i<=ie; ++i)
        phydro->w(IVX, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);


  if (Globals::my_rank == 0) {
    if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0)
      printf("Reading x2-velocity ... \n");
    read_vtk(this, vtkdir, vtkfile0, "velocity", 1, data);
  }

#ifdef MPI_PARALLEL
  ierr = MPI_Bcast(data.data(), nsize, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

  for (int k=ks; k<=ke; ++k)
    for (int j=js; j<=je; ++j)
      for (int i=is; i<=ie; ++i)
        phydro->w(IVY, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);


  if (Globals::my_rank == 0) {
    if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0)
      printf("Reading x3-velocity ... \n");
    read_vtk(this, vtkdir, vtkfile0, "velocity", 2, data);
  }

#ifdef MPI_PARALLEL
  ierr = MPI_Bcast(data.data(), nsize, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

  for (int k=ks; k<=ke; ++k)
    for (int j=js; j<=je; ++j)
      for (int i=is; i<=ie; ++i)
        phydro->w(IVZ, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);


  if (MAGNETIC_FIELDS_ENABLED) {
    //temporary array to store data of the entire face-centered magnetic field
    AthenaArray<Real> data_B;

    data_B.NewAthenaArray(Nz_mesh, Ny_mesh, Nx_mesh+2);

    if (Globals::my_rank == 0) {
      if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0)
        printf("Reading x1 B-field ... \n");
      read_vtk(this, vtkdir, vtkfile0, "cell_centered_B", 0, data);
    }

#ifdef MPI_PARALLEL
    ierr = MPI_Bcast(data.data(), nsize, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

    for (int k=0; k<=Nz_mesh-1; ++k) {
      for (int j=0; j<=Ny_mesh-1; ++j) {
        for (int i=0; i<=Nx_mesh-1; ++i) {
          data_B(k,j,i+1) = data(k,j,i);
        }
        //periodic boundary conditions along the x-direction
        data_B(k,j,0) = data_B(k,j,Nx_mesh);
        data_B(k,j,Nx_mesh+1) = data_B(k,j,1);
      }
    }

    for (int k=ks; k<=ke; ++k)
      for (int j=js; j<=je; ++j)
        for (int i=is; i<=ie+1; ++i)
          pfield->b.x1f(k,j,i) = 0.5*(data_B(k-ks+gks, j-js+gjs, i-is+gis)
                                     +data_B(k-ks+gks, j-js+gjs, i+1-is+gis));

    data_B.DeleteAthenaArray();


    if (Globals::my_rank == 0) {
      if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0)
        printf("Reading x2 B-field ... \n");
      read_vtk(this, vtkdir, vtkfile0, "cell_centered_B", 1, data);
    }

#ifdef MPI_PARALLEL
    ierr = MPI_Bcast(data.data(), nsize, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

    data_B.NewAthenaArray(Nz_mesh, Ny_mesh+2, Nx_mesh);

    for (int k=0; k<=Nz_mesh-1; ++k) {
      for (int i=0; i<=Nx_mesh-1; ++i) {
        for (int j=0; j<=Ny_mesh-1; ++j) {
          data_B(k,j+1,i) = data(k,j,i);
        }
        //periodic boundary conditions along the y-direction
        data_B(k,0,i) = data_B(k,Ny_mesh,i);
        data_B(k,Ny_mesh+1,i) = data_B(k,1,i);
      }
    }

    for (int k=ks; k<=ke; ++k)
      for (int j=js; j<=je+1; ++j)
        for (int i=is; i<=ie; ++i)
          pfield->b.x2f(k,j,i) = 0.5*(data_B(k-ks+gks, j-js+gjs, i-is+gis)
                                     +data_B(k-ks+gks, j+1-js+gjs, i-is+gis));

    data_B.DeleteAthenaArray();


    if (Globals::my_rank == 0) {
      if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0)
        printf("Reading x3 B-field ... \n");
      read_vtk(this, vtkdir, vtkfile0, "cell_centered_B", 2, data);
    }

#ifdef MPI_PARALLEL
    ierr = MPI_Bcast(data.data(), nsize, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

    data_B.NewAthenaArray(Nz_mesh+2, Ny_mesh, Nx_mesh);

    for (int i=0; i<=Nx_mesh-1; ++i) {
      for (int j=0; j<=Ny_mesh-1; ++j) {
        for (int k=0; k<=Nz_mesh-1; ++k) {
          data_B(k+1,j,i) = data(k,j,i);
        }
        //outflow boundary conditions along the z-direction
        data_B(0,j,i) = data_B(1,j,i);
        data_B(Nz_mesh+1,j,i) = data_B(Nz_mesh,j,i);
      }
    }

    for (int k=ks; k<=ke+1; ++k)
      for (int j=js; j<=je; ++j)
        for (int i=is; i<=ie; ++i)
          pfield->b.x3f(k,j,i) = 0.5*(data_B(k-ks+gks, j-js+gjs, i-is+gis)
                                     +data_B(k+1-ks+gks, j-js+gjs, i-is+gis));

    data_B.DeleteAthenaArray();

    pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);
  } //End MHD */

  data.DeleteAthenaArray();

  if (MAGNETIC_FIELDS_ENABLED) {
    peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                                         is, ie, js, je, ks, ke);
  } else {
    peos->PrimitiveToConserved(phydro->w, b, phydro->u, pcoord,
                                     is, ie, js, je, ks, ke);
  }
  b.DeleteAthenaArray();

  if (CR_ENABLED) {
    Real x1size = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
    Real x2size = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
    for(int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real z1 = pcoord->x3v(k);
          Real v2 = phydro->w(IVX,k,j,i)*phydro->w(IVX,k,j,i)+phydro->w(IVY,k,j,i)*
            phydro->w(IVY,k,j,i)+phydro->w(IVZ,k,j,i)*phydro->w(IVZ,k,j,i);
          Real Flux_Bs = 0.5 * tot_InjectionRate / (x1size * x2size);
          pcr->u_cr(CRE,k,j,i) = 3./4.*Flux_Bs/std::sqrt(v2);
          pcr->u_cr(CRF1,k,j,i) = 0.0;
          pcr->u_cr(CRF2,k,j,i) = 0.0;
          pcr->u_cr(CRF3,k,j,i) = z1/std::abs(z1)*Flux_Bs/pcr->vmax;
        }
      }
    }
  }
  return;
}

//! \fn void read_vtk(MeshBlock *mb, std::string vtkdir, std::string vtkfile0,
//!                   std::string field, int component, AthenaArray<Real> &data)
//! \brief Read the field values in the athena rst file
static void read_vtk(MeshBlock *mb, std::string vtkdir, std::string vtkfile0,
  std::string field, int component, AthenaArray<Real> &data) {
  std::stringstream msg;
  FILE *fp = NULL;
  char cline[MAXLEN], type[MAXLEN], variable[MAXLEN];
  char format[MAXLEN], t_type[MAXLEN], t_format[MAXLEN];
  std::string line;

  const std::string athena_header = "# vtk DataFile Version 2.0"; //athena4.2 header
  const std::string athena_header3 = "# vtk DataFile Version 3.0"; //athena4.2 header
  bool SHOW_OUTPUT = false;
  int Nx_vtk, Ny_vtk, Nz_vtk; //dimensions of vtk files

  //dimensions of meshblock
  int Nx_mb = mb->ie - mb->is + 1;
  int Ny_mb = mb->je - mb->js + 1;
  int Nz_mb = mb->ke - mb->ks + 1;

  //dimensions of mesh
  int Nx_mesh = mb->pmy_mesh->mesh_size.nx1;
  int Ny_mesh = mb->pmy_mesh->mesh_size.nx2;
  int Nz_mesh = mb->pmy_mesh->mesh_size.nx3;

  double ox_vtk, oy_vtk, oz_vtk; //origins of vtk file
  double dx_vtk, dy_vtk, dz_vtk; //spacings of vtk file
  int cell_dat_vtk; //total number of cells in vtk file
  //total number of cells in MeshBlock
  const int cell_dat_mb = Nx_mb * Ny_mb * Nz_mb;
  int retval; //file handler return value
  float fdat, fvec[3], ften[9];//store float format scaler, vector, and tensor

  std::string vtkfile;
  vtkfile = vtkdir + "id0/" + vtkfile0;

  if ( (fp = fopen(vtkfile.c_str(),"r")) == NULL ) {
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Unable to open file: " << vtkfile << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  //get header
  fgets(cline,MAXLEN,fp);
  line.assign(cline);
  trim(line);
  if (SHOW_OUTPUT) {
    std::cout << line << std::endl;
  }
  if (line != athena_header && line != athena_header3) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Assuming Athena4.2 header " << athena_header << ", get header "
      << line << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  //get comment field
  fgets(cline,MAXLEN,fp);
  line.assign(cline);
  trim(line);
  if (SHOW_OUTPUT) {
    std::cout << line << std::endl;
  }

  //get BINARY or ASCII
  fgets(cline,MAXLEN,fp);
  line.assign(cline);
  trim(line);
  if (SHOW_OUTPUT) {
    std::cout << line << std::endl;
  }
  if (line != "BINARY") {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
    << "Unsupported file format: " << line << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  //get DATASET STRUCTURED_POINTS or DATASET UNSTRUCTURED_GRID
  fgets(cline,MAXLEN,fp);
  line.assign(cline);
  trim(line);
  if (SHOW_OUTPUT) {
    std::cout << line << std::endl;
  }
  if (line != "DATASET STRUCTURED_POINTS") {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
    << "Unsupported file data set structure: " << line << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  //I'm assuming from this point on that the header is in good shape

  //read dimensions
  fgets(cline,MAXLEN,fp);
  if (SHOW_OUTPUT) {
    std::cout << cline;
  }
  sscanf(cline,"DIMENSIONS %d %d %d\n",&Nx_vtk,&Ny_vtk,&Nz_vtk);
  //We want to store the number of grid cells, not the number of grid
  //cell corners.
  Nx_vtk--;
  Ny_vtk--;
  Nz_vtk--;

  // Origin
  fgets(cline,MAXLEN,fp);
  if (SHOW_OUTPUT) {
    std::cout << cline;
  }
  sscanf(cline,"ORIGIN %le %le %le\n",&ox_vtk,&oy_vtk,&oz_vtk);

  // spacing
  fgets(cline,MAXLEN,fp);
  if (SHOW_OUTPUT) {
    std::cout << cline;
  }
  sscanf(cline,"ORIGIN %le %le %le\n",&dx_vtk,&dy_vtk,&dz_vtk);

  // Cell Data = Nx*Ny*Nz
  fgets(cline,MAXLEN,fp);
  if (SHOW_OUTPUT) {
    std::cout << cline;
  }
  sscanf(cline,"CELL_DATA %d\n",&cell_dat_vtk);
  fclose(fp); //close file

  int tigress_zmeshblocks, tigress_ymeshblocks, tigress_xmeshblocks;
  int tigress_Nx_mesh, tigress_Ny_mesh, tigress_Nz_mesh;
  int tigress_Nx, tigress_Ny, tigress_Nz;

  tigress_Nx = Nx_vtk;
  tigress_Ny = Ny_vtk;
  tigress_Nz = Nz_vtk;

  tigress_Nx_mesh = Nx_mesh;
  tigress_Ny_mesh = Ny_mesh;
  tigress_Nz_mesh = Nz_mesh;

  tigress_xmeshblocks = tigress_Nx_mesh/tigress_Nx;
  tigress_ymeshblocks = tigress_Ny_mesh/tigress_Ny;
  tigress_zmeshblocks = tigress_Nz_mesh/tigress_Nz;

  if (SHOW_OUTPUT) {
    printf ("Number of meshbloks %d %d %d \n",
      tigress_xmeshblocks,tigress_ymeshblocks,tigress_zmeshblocks);
  }

  for(int k=0; k<tigress_zmeshblocks; ++k) {
    for (int j=0; j<tigress_ymeshblocks; ++j) {
      for (int i=0; i<tigress_xmeshblocks; ++i) {
        //find the corresponding athena4.2 global id
        std::int64_t id_old = i + j * tigress_xmeshblocks
                            + k * tigress_xmeshblocks * tigress_ymeshblocks;
        std::stringstream id_str_stream;
        id_str_stream << "id" << id_old; // id#
        std::string id_str = id_str_stream.str();
        std::string vtk_name0 = vtkfile0;
        std::size_t pos1 = vtk_name0.find_first_of('.');
        std::string vtk_name;
        if (k == 0)
          vtk_name = vtk_name0.substr(0, pos1) + vtk_name0.substr(pos1);
        else
          vtk_name = vtk_name0.substr(0, pos1) + "-" + id_str + vtk_name0.substr(pos1);
        vtkfile = vtkdir + id_str + "/" + vtk_name;

        //origin of the meshblock
        int xstart = i*tigress_Nx;
        int ystart = j*tigress_Ny;
        int zstart = k*tigress_Nz;

        //Reading file
        if ( (fp = fopen(vtkfile.c_str(),"r")) == NULL ) {
          msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
            << "Unable to open file: " << vtkfile << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }

        do {
          fgets(cline,MAXLEN,fp);
        } while(strncmp(cline,"CELL_DATA",9) != 0);

        // Now read the rest of the data in
        while (true) {
          retval = fscanf(fp,"%s %s %s\n", type, variable, format);
          if (retval == EOF) { // Assuming no errors, we are done.
            fclose(fp); //close file
            break;
          }
          if (SHOW_OUTPUT) {
            printf("%s %s %s\n", type, variable ,format);
          }
          //check format
          if (strcmp(format, "float") != 0) {
            fclose(fp);
            msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
            << "expected  \"float\" format, found " << type << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
          //check lookup table
          if (strcmp(type, "SCALARS") == 0) {
            // Read in the LOOKUP_TABLE (only default supported for now)
            fscanf(fp,"%s %s\n", t_type, t_format);
            if (strcmp(t_type, "LOOKUP_TABLE") != 0 ||
                strcmp(t_format, "default") != 0 ) {
              fclose(fp);
              msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
              << "Expected \"LOOKUP_TABLE default, found "
              << t_type << " " << t_format << std::endl;
              throw std::runtime_error(msg.str().c_str());
            }
            if (SHOW_OUTPUT) {
              printf("%s %s\n", t_type, t_format);
            }
          }

          //determine variable type and read data
          //read scalars
          if (strcmp(type, "SCALARS") == 0) {
            if (strcmp(variable, field.c_str()) == 0) {
              //printf("  Reading %s...\n", variable);
              for (int k=0; k<Nz_vtk; k++) {
                for (int j=0; j<Ny_vtk; j++) {
                  for (int i=0; i<Nx_vtk; i++) {
                    if (fread(&fdat, sizeof(float), 1, fp) != 1) {
                      fclose(fp);
                      msg << "### FATAL ERROR in Problem Generator [read_vtk]"
                          << std::endl
                          << "Error reading SCALARS... " << std::endl;
                      throw std::runtime_error(msg.str().c_str());
                    }
                    ath_bswap(&fdat, sizeof(float), 1);
                    data(k+zstart, j+ystart, i+xstart) = fdat;
                  }
                }
              }
              fclose(fp);
              break;
            } else {
              if (SHOW_OUTPUT) printf("  Skipping %s...\n",variable);
              fseek(fp, cell_dat_vtk*sizeof(float), SEEK_CUR);
            }
          //read vectors
          } else if (strcmp(type, "VECTORS") == 0) {
            if (strcmp(variable, field.c_str()) == 0) {
              for (int k=0; k<Nz_vtk; k++) {
                for (int j=0; j<Ny_vtk; j++) {
                  for (int i=0; i<Nx_vtk; i++) {
                    if (fread(&fvec, sizeof(float), 3, fp) != 3) {
                      fclose(fp);
                      msg << "### FATAL ERROR in Problem Generator [read_vtk]"
                          << std::endl
                          << "Error reading VECTORS... " << std::endl;
                      throw std::runtime_error(msg.str().c_str());
                    }
                    ath_bswap(&fvec, sizeof(float), 3);
                    data(k+zstart, j+ystart, i+xstart) = fvec[component];
                  }
                }
              }
              fclose(fp);
              break;
            } else {
              if (SHOW_OUTPUT) printf("  Skipping %s...\n", variable);
              fseek(fp, 3*cell_dat_vtk*sizeof(float), SEEK_CUR);
              }
            //read tensors, not supported yet
          } else if (strcmp(type, "TENSORS") == 0) {
            if (strcmp(variable, field.c_str()) == 0) {
              fclose(fp);
              msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
                  << "TENSORS reading not supported." << std::endl;
              throw std::runtime_error(msg.str().c_str());
            } else {
              if (SHOW_OUTPUT) printf("  Skipping %s...\n", variable);
              fseek(fp, 9*cell_dat_vtk*sizeof(float), SEEK_CUR);
            }
          } else {
            fclose(fp);
            msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
                << "Input type not supported: " << type << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
        }
      }
    }
  }
}

//! \fn void read_vtk(MeshBlock *mb, std::string filename,
//!                   std::vector<StarParS> &pList)
//! \brief Read the star particle information in the athena vtk file
static void read_starpar_vtk(MeshBlock *mb, std::string filename,
                             std::vector<StarParS> &pList) {
  std::stringstream msg;
  FILE *fp = NULL;
  char cline[256], type[256], variable[256], format[256], t_type[256], t_format[256];
  int retval,i,itmp1,itmp2,nstars,idat,ivec[3];
  size_t nread;
  float fdat,fvec[3];
  std::string line;
  bool SHOW_OUTPUT = false;
  const std::string athena_header = "# vtk DataFile Version 2.0"; //athena4.2 header
  const std::string athena_header3 = "# vtk DataFile Version 3.0"; //athena4.2 header

  if ( (fp = fopen(filename.c_str(),"r")) == NULL ) {
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Unable to open file: " << filename << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  //get header
  fgets(cline,256,fp);
  line.assign(cline);
  trim(line);
  if (SHOW_OUTPUT) {
    std::cout << line << std::endl;
  }
  if (line != athena_header && line != athena_header3) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Assuming Athena4.2 header " << athena_header << ", get header "
      << line << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  //get comment field
  fgets(cline,256,fp);
  line.assign(cline);
  trim(line);
  if (SHOW_OUTPUT) {
    std::cout << line << std::endl;
  }

  //get BINARY or ASCII
  fgets(cline,256,fp);
  line.assign(cline);
  trim(line);
  if (SHOW_OUTPUT) {
    std::cout << line << std::endl;
  }
  if (line != "BINARY") {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Unsupported file format: " << line << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  //get DATASET STRUCTURED_POINTS or DATASET UNSTRUCTURED_GRID
  fgets(cline,256,fp);
  line.assign(cline);
  trim(line);
  if (SHOW_OUTPUT) {
    std::cout << line << std::endl;
  }
  if (line != "DATASET UNSTRUCTURED_GRID") {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Unsupported file data set structure: " << line << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  // I'm assuming from this point on that the header is in good shape

  fgets(cline,256,fp);
  if (SHOW_OUTPUT) {
    std::cout << cline;
  }
  sscanf(cline,"POINTS %d \n",&nstars);
  if (SHOW_OUTPUT) std::cout<<"POINTS"<<nstars<<std::endl;

  /* Read in star particle locations */
  for (i=0; i<nstars; i++) {
    if ((nread = fread(&fvec, sizeof(float), 3, fp)) != 3) {
      fclose(fp);
      msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
        << "Error reading star particle locations... " << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
    ath_bswap(&fvec,sizeof(float),3);

    // Allocate new star particle list element
    StarParS pStar;
    pStar.x1 = (Real)fvec[0];
    pStar.x2 = (Real)fvec[1];
    pStar.x3 = (Real)fvec[2];
    pStar.active = -99;
    pList.push_back(pStar);
  }
  fscanf(fp,"\n");

  // Read in CELL data
  fscanf(fp,"CELLS %d %d\n",&itmp1,&itmp2);
  if (SHOW_OUTPUT) std::cout<<"CELLS"<<itmp1<<" "<<itmp2<<std::endl;
  if (itmp1 != nstars && itmp2 != 2*nstars) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Error reading CELLS " << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  for (i=0; i<nstars; i++) {
    if ((nread = fread(&ivec, sizeof(int), 2, fp)) != 2) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Error reading CELLS List" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
    ath_bswap(&ivec,sizeof(int),2);
  }
  fscanf(fp,"\n");

  // Real in CELL_TYPES
  fscanf(fp,"CELLS_TYPES %d \n",&itmp1);
  if (SHOW_OUTPUT) std::cout<<"CELL_TYPES"<<itmp1<<std::endl;
  if (itmp1 != nstars) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Error reading CELLS_TYPES " << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  for (i=0; i<nstars; i++) {
    if ((nread = fread(&idat, sizeof(int), 1, fp)) != 1) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Error reading CELLS_TYPES List" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
    ath_bswap(&ivec,sizeof(int),2);
  }
  fscanf(fp,"\n");

  fgets(cline,256,fp);
  line.assign(cline);
  trim(line);
  if (SHOW_OUTPUT) {
    std::cout << cline << std::endl;
  }
  // Time & Cycle
  fscanf(fp,"FIELD FieldData %d\n",&itmp1);
  if (SHOW_OUTPUT) std::cout<<"FIELD FieldData "<<itmp1<<std::endl;
  if (itmp1 != 2) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Expected 2 fields for time and cycle..." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  fgets(cline,256,fp);
  line.assign(cline);
  trim(line);
  if (SHOW_OUTPUT) {
    std::cout << line << std::endl;
  }
  if ((nread = fread(&fdat, sizeof(float), 1, fp)) != 1) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Error reading TIME..." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  ath_bswap(&fdat,sizeof(float),1);
  if (SHOW_OUTPUT) std::cout << fdat << std::endl;
  fscanf(fp,"\n");

  fgets(cline,256,fp);
  line.assign(cline);
  trim(line);
  if (SHOW_OUTPUT) {
    std::cout << line << std::endl;
  }
  if ((nread = fread(&itmp1, sizeof(int), 1, fp)) != 1) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Error reading CYCLE..." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  ath_bswap(&fdat,sizeof(float),1);
  if (SHOW_OUTPUT) std::cout << idat << std::endl;
  fscanf(fp,"\n");

  // Read star particle CELL/POINT data
  fscanf(fp,"CELL_DATA %d\n",&itmp1);
  if (SHOW_OUTPUT) std::cout<<"CELL_DATA "<<itmp1<<std::endl;
  if (itmp1 != nstars) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Error reading CELL_DATA..." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  fscanf(fp,"POINT_DATA %d\n",&itmp1);
  if (SHOW_OUTPUT) std::cout<<"POINT_DATA "<<itmp1<<std::endl;
  if (itmp1 != nstars) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Error reading POINT_DATA..." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  while (true) {
    retval = fscanf(fp,"%s %s %s\n",type,variable,format);

    if (retval == EOF) { //Assuming no errors, we are done...
      fclose(fp);
      return;
    }

    if (strcmp(type, "SCALARS") == 0) {
      // Read in the LOOKUP_TABLE (only default supported for now)
      fscanf(fp,"%s %s\n",t_type,t_format);
      if (strcmp(t_type, "LOOKUP_TABLE") != 0 || strcmp(t_format, "default") != 0 ) {
        fclose(fp);
        msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
          << "Expected \"LOOKUP_TABLE default, found "
          << t_type << " " << t_format << std::endl;
        throw std::runtime_error(msg.str().c_str());
      }
      if (SHOW_OUTPUT) {
        printf("%s %s\n", t_type, t_format);
      }
    }

    if (strcmp(type, "SCALARS") == 0) {
      if (strcmp(variable, "star_particle_id") == 0) {
        // Read star particle IDs
        if (SHOW_OUTPUT) std::cout<<"Reading..."<<variable<<std::endl;
        for (i=0; i<nstars; i++) {
          if ((nread = fread(&idat, sizeof(int), 1, fp)) != 1) {
            fclose(fp);
            msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
            << "Error reading star_particle_id list..." << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
           ath_bswap(&idat,sizeof(int),1);
          // Assign star particle IDs
          pList[i].id = idat;
        }
        fscanf(fp,"\n");
      } else if (strcmp(variable, "star_particle_flag") == 0) {
        // Read star particle IDs
        if (SHOW_OUTPUT) std::cout<<"Reading..."<<variable<<std::endl;
        for (i=0; i<nstars; i++) {
          if ((nread = fread(&idat, sizeof(int), 1, fp)) != 1) {
            fclose(fp);
            msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
            << "Error reading star_particle_id list..." << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
          ath_bswap(&idat,sizeof(int),1);
          // Assign star particle IDs
          pList[i].active = idat;
        }
        fscanf(fp,"\n");
      } else if (strcmp(variable, "star_particle_mass") == 0) {
        // Read star particle mass
        if (SHOW_OUTPUT) std::cout<<"Reading..."<<variable<<std::endl;
        for (i=0; i<nstars; i++) {
          if ((nread = fread(&fdat, sizeof(float), 1, fp)) != 1) {
            fclose(fp);
            msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
            << "Error reading star_particle_mass list..." << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
          ath_bswap(&fdat,sizeof(float),1);
          // Assign star particle masses
          pList[i].m = (Real)fdat;
        }
        fscanf(fp,"\n");
      } else if (strcmp(variable, "star_particle_age") == 0) {
        // Read star particle age
        if (SHOW_OUTPUT) std::cout<<"Reading..."<<variable<<std::endl;
        for (i=0; i<nstars; i++) {
          if ((nread = fread(&fdat, sizeof(float), 1, fp)) != 1) {
            fclose(fp);
            msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
            << "Error reading star_particle_age list..." << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
          ath_bswap(&fdat,sizeof(float),1);
          // Assign star particle ages
          pList[i].age = (Real)fdat;
        }
        fscanf(fp,"\n");
      } else if (strcmp(variable, "star_particle_mage") == 0) {
        // Read star particle mage
        if (SHOW_OUTPUT) std::cout<<"Reading..."<<variable<<std::endl;
        for (i=0; i<nstars; i++) {
          if ((nread = fread(&fdat, sizeof(float), 1, fp)) != 1) {
            fclose(fp);
            msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
            << "Error reading star_particle_age list..." << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
          ath_bswap(&fdat,sizeof(float),1);
          // Assign star particle mages
          pList[i].mage = (Real)fdat;
        }
        fscanf(fp,"\n");
      } else {
        if (SHOW_OUTPUT) printf("  Skipping %s...\n",variable);
        fseek(fp, nstars*sizeof(float), SEEK_CUR);
      }
    } else if (strcmp(type, "VECTORS") == 0) {
      if (strcmp(variable, "star_particle_position") == 0) {
        if (SHOW_OUTPUT) std::cout<<"Reading..."<<variable<<std::endl;
        for (i=0; i<nstars; i++) {
          if ((nread = fread(&fvec, sizeof(float), 3, fp)) != 3) {
            fclose(fp);
            msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
            << "Error reading star_particle_position... " << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
          ath_bswap(&fvec,sizeof(float),3);
          // Assign star particle positions
          pList[i].x1 = (Real)fvec[0];
          pList[i].x2 = (Real)fvec[1];
          pList[i].x3 = (Real)fvec[2];
        }
        fscanf(fp,"\n");
      } else if (strcmp(variable, "star_particle_velocity") == 0) {
        if (SHOW_OUTPUT) printf("Reading %s...\n",variable);
        for (i=0; i<nstars; i++) {
          if ((nread = fread(&fvec, sizeof(float), 3, fp)) != 3) {
            fclose(fp);
            msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
            << "Error reading star_particle_velocity... " << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
          ath_bswap(&fvec,sizeof(float),3);
          // Assign star particle positions
          pList[i].v1 = (Real)fvec[0];
          pList[i].v2 = (Real)fvec[1];
          pList[i].v3 = (Real)fvec[2];
        }
        fscanf(fp,"\n");
      }
    } else {
      fclose(fp);
      msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Input type not supported: " << type << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
  }

  fclose(fp);

  return;
}

//======================================================================================
//! \fn static void ath_bswap(void *vdat, int len, int cnt)
//! \brief Swap bytes, code stolen from Athena4.2, NEMO
//======================================================================================
static void ath_bswap(void *vdat, int len, int cnt) {
  char tmp, *dat = reinterpret_cast<char *>(vdat);
  int k;

  if (len==1) {
    return;
  } else if (len==2) {
    while (cnt--) {
      tmp = dat[0];  dat[0] = dat[1];  dat[1] = tmp;
      dat += 2;
    }
  } else if (len==4) {
    while (cnt--) {
      tmp = dat[0];  dat[0] = dat[3];  dat[3] = tmp;
      tmp = dat[1];  dat[1] = dat[2];  dat[2] = tmp;
      dat += 4;
    }
  } else if (len==8) {
    while (cnt--) {
      tmp = dat[0];  dat[0] = dat[7];  dat[7] = tmp;
      tmp = dat[1];  dat[1] = dat[6];  dat[6] = tmp;
      tmp = dat[2];  dat[2] = dat[5];  dat[5] = tmp;
      tmp = dat[3];  dat[3] = dat[4];  dat[4] = tmp;
      dat += 8;
    }
  } else {  /* the general SLOOOOOOOOOW case */
    for(k=0; k<len/2; k++) {
      tmp = dat[k];
      dat[k] = dat[len-1-k];
      dat[len-1-k] = tmp;
    }
  }
}

//======================================================================================
//! \fn static void trim(std::string &s)
//! \brief get rid of white spaces leading and trailing a string
//======================================================================================
static void trim(std::string &s) {
  size_t p = s.find_first_not_of(" \t\n");
  s.erase(0, p);

  p = s.find_last_not_of(" \t\n");
  if (p != std::string::npos) {
    s.erase(p+1);
  }
}

//======================================================================================
//! \fn static std::vector<std::string> split(std::string str, char delimiter)
//! \brief split a string, and store sub strings in a vector
//======================================================================================
static std::vector<std::string> split(std::string str, char delimiter) {
  std::vector<std::string> internal;
  std::stringstream ss(str); // Turn the string into a stream.
  std::string tok;

  while(getline(ss, tok, delimiter)) {
    trim(tok);
    internal.push_back(tok);
  }

  return internal;
}

//! \fn Real set_CR_Luminosity(Real tage)
//! \brief This function calculate the cosmic-rate luminosity
//! produced pruduced by each star cluster
Real set_CR_Luminosity(Real tage) {
  int i, iage;
  Real dage = 0.2; //Myr
  int const Narray = 201;
  Real N_SNe, CR_Lum;
  Real SN_energy = pcool->punit->Bethe_in_code;
  Real CR_eff = 0.1;

  // SNRate for Msun cluster per Myr (Starburst99)
  Real SNR[Narray] = {
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   4.75335226e-05,
         5.52077439e-04,   5.22396189e-04,   5.48276965e-04,
         5.33334895e-04,   5.44502653e-04,   5.79428696e-04,
         5.58470195e-04,   5.94292159e-04,   6.10942025e-04,
         5.98411595e-04,   6.39734835e-04,   6.50129690e-04,
         6.28058359e-04,   5.79428696e-04,   5.22396189e-04,
         5.06990708e-04,   4.72063041e-04,   4.77529274e-04,
         4.83058802e-04,   4.75335226e-04,   4.50816705e-04,
         4.42588372e-04,   4.34510224e-04,   4.27562886e-04,
         4.19758984e-04,   4.13047502e-04,   4.07380278e-04,
         4.00866718e-04,   3.99024902e-04,   3.99024902e-04,
         3.94457302e-04,   3.89941987e-04,   3.85478358e-04,
         3.81944271e-04,   3.77572191e-04,   3.74110588e-04,
         3.69828180e-04,   3.66437575e-04,   3.63078055e-04,
         3.59749335e-04,   3.55631319e-04,   3.52370871e-04,
         3.49945167e-04,   3.46736850e-04,   3.43557948e-04,
         3.40408190e-04,   3.37287309e-04,   3.34195040e-04,
         3.31894458e-04,   3.26587832e-04,   3.22849412e-04,
         3.19889511e-04,   3.16956746e-04,   3.14774831e-04,
         3.11888958e-04,   3.09741930e-04,   3.06902199e-04,
         3.04789499e-04,   3.01995172e-04,   2.99916252e-04,
         2.97851643e-04,   2.95801247e-04,   2.93089325e-04,
         2.91071712e-04,   2.89067988e-04,   2.87078058e-04,
         2.85101827e-04,   2.83139200e-04,   2.81190083e-04,
         2.79254384e-04,   2.77332010e-04,   2.75422870e-04,
         2.73526873e-04,   2.71643927e-04,   2.69773943e-04,
         2.68534445e-04,   2.66685866e-04,   2.66072506e-04,
         2.64240876e-04,   2.63026799e-04,   2.61216135e-04,
         2.60015956e-04,   2.58226019e-04,   2.57039578e-04,
         2.55270130e-04,   2.54097271e-04,   2.52348077e-04,
         2.51188643e-04,   2.50034536e-04,   2.48885732e-04,
         2.47172415e-04,   2.46036760e-04,   2.44906324e-04,
         2.43220401e-04,   2.42102905e-04,   2.40990543e-04,
         2.39883292e-04,   2.38781128e-04,   2.37137371e-04,
         2.36047823e-04,   2.34963282e-04,   2.33883724e-04,
         2.32809126e-04,   2.31739465e-04,   2.30674719e-04,
         2.29614865e-04,   2.28034207e-04,   2.27509743e-04,
         2.26464431e-04,   2.25423921e-04,   2.24388192e-04,
         2.23357222e-04,   2.22330989e-04,   2.21309471e-04,
         2.20292646e-04,   2.19280494e-04,   2.18272991e-04,
         2.17270118e-04,   2.16271852e-04,   2.15774441e-04,
         2.14783047e-04,   2.13796209e-04,   2.12813905e-04,
         2.12324446e-04,   2.10862815e-04,   2.10377844e-04,
         2.09411246e-04,   2.08449088e-04,   2.07491352e-04,
         2.07014135e-04,   2.06062991e-04,   2.05116218e-04,
         2.04644464e-04,   2.03704208e-04,   2.02768272e-04,
         2.02301918e-04,   2.01372425e-04,   2.00447203e-04,
         1.99986187e-04,   1.99067334e-04,   1.98609492e-04,
         1.97696964e-04,   1.90107828e-04,   1.89670592e-04,
         1.88799135e-04,   1.87931682e-04,   1.87068214e-04,
         1.86208714e-04,   1.85353162e-04,   1.84926862e-04,
         1.83653834e-04,   1.83231442e-04,   1.82389570e-04,
         1.81551566e-04,   1.80717413e-04,   1.80301774e-04,
         1.79473363e-04,   1.78648757e-04,   1.78237877e-04,
         1.77418948e-04,   1.76603782e-04,   1.75792361e-04,
         1.75388050e-04,   1.74582215e-04,   1.73780083e-04,
         1.72981636e-04,   1.72583789e-04,   1.71790839e-04,
         1.71395731e-04,   1.70608239e-04,   1.69824365e-04,
         1.69044093e-04,   1.68655303e-04,   1.68267406e-04,
         1.67494288e-04,   1.66724721e-04,   1.65958691e-04,
         1.65576996e-04,   1.64816239e-04,   1.64437172e-04,
         1.63681652e-04,   1.63305195e-04,   1.62554876e-04,
  };

  iage = static_cast<int>(tage / dage);
  iage = std::max(iage, 0);
  iage = std::min(iage,Narray-1);
  if (tage >= 40) iage--;

  if (SNR[iage] == SNR[iage+1]) return 0;

  //instanteneous rate of supernove per cluster mass
  N_SNe = lin_interpol(tage,iage*dage,(iage+1)*dage,SNR[iage],SNR[iage+1]);
  //time-averaged CR energy injection rate per cluster mass =
  //0.1 * SN luminosity per cluster mass
  CR_Lum = CR_eff * (N_SNe * SN_energy);

  return CR_Lum;
}

//! \fn static inline Real lin_interpol(const Real x, const Real xi,
//! const Real xi1, const Real yi, const Real yi1)
//! \brief Function for linear interpolation
static inline Real lin_interpol(const Real x, const Real xi, const Real xi1,
                                const Real yi, const Real yi1) {
  return (yi*(xi1-x)+yi1*(x-xi))/(xi1-xi);
}

//! \fn Outflow_down (MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//! FaceField &b, Real time, Real dt, int is, int ie, int js, int je,
//! int ks, int ke, int ngh)
//! \brief MHD boundary conditions on the left side of the z-axis
void Outflow_down (MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je,
     int ks, int ke, int ngh) {
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        prim(IDN,ks-k,j,i) = prim(IDN,ks,j,i);
        prim(IVX,ks-k,j,i) = prim(IVX,ks,j,i);
        prim(IVY,ks-k,j,i) = prim(IVY,ks,j,i);
        if (prim(IVZ,ks,j,i) <= 0.)
          prim(IVZ,ks-k,j,i) = prim(IVZ,ks,j,i);
        else
          prim(IVZ,ks-k,j,i) = 0.;
        if(NON_BAROTROPIC_EOS)
          prim(IEN,ks-k,j,i) = prim(IEN,ks,j,i);
      }
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          b.x3f(ks-k,j,i) =  b.x3f(ks-k+2,j,i);
        }
      }
      if(je > js) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            b.x2f(ks-k,j,i) =  b.x2f(ks,j,i);
          }
        }
      }
      if(ie > is) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(ks-k,j,i) = b.x1f(ks,j,i);
          }
        }
      }
    }
  }
}

//! \fn Outflow_up (MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//! FaceField &b, Real time, Real dt, int is, int ie, int js, int je,
//! int ks, int ke, int ngh)
//! \brief MHD boundary conditions on the right side of the z-axis
void Outflow_up (MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je,
     int ks, int ke, int ngh) {
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        prim(IDN,ke+k,j,i) = prim(IDN,ke,j,i);
        prim(IVX,ke+k,j,i) = prim(IVX,ke,j,i);
        prim(IVY,ke+k,j,i) = prim(IVY,ke,j,i);
        if (prim(IVZ,ke,j,i) >= 0.)
          prim(IVZ,ke+k,j,i) = prim(IVZ,ke,j,i);
        else
           prim(IVZ,ke+k,j,i) = 0.;
        if(NON_BAROTROPIC_EOS)
          prim(IEN,ke+k,j,i) = prim(IEN,ke,j,i);
      }
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          b.x3f(ke+k+1,j,i) =  b.x3f(ke+k-1,j,i);
        }
      }
      if(je > js) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            b.x2f(ke+k,j,i) =  b.x2f(ke,j,i);
          }
        }
      }
      if(ie > is) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(ke+k,j,i) = b.x1f(ke,j,i);
          }
        }
      }
    }
  }
}


//! \fn CROutflow_down (MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr,
//!    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
//!    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie,
//!    int js, int je, int ks, int ke, int ngh)
//! \brief CR boundary conditions on the left side of the z-axis
void CROutflow_down (MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie,
    int js, int je, int ks, int ke, int ngh) {
  if(CR_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          u_cr(CRE,ks-k,j,i) = u_cr(CRE,ks,j,i) - k*(u_cr(CRE,ks+1,j,i)-u_cr(CRE,ks,j,i));
          u_cr(CRF1,ks-k,j,i) = u_cr(CRF1,ks,j,i);
          u_cr(CRF2,ks-k,j,i) = u_cr(CRF2,ks,j,i);
          if (u_cr(CRF3,ks,j,i)<=0.)
            u_cr(CRF3,ks-k,j,i) = u_cr(CRF3,ks,j,i);
          else
            u_cr(CRF3,ks-k,j,i) = 0;
        }
      }
    }
  }
}

//! \fn CROutflow_up (MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr,
//!    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
//!    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie,
//!    int js, int je, int ks, int ke, int ngh)
//! \brief CR boundary conditions on the right side of the z-axis
void CROutflow_up (MeshBlock *pmb, Coordinates *pco, CosmicRay *pcr,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &u_cr, Real time, Real dt, int is, int ie,
    int js, int je, int ks, int ke, int ngh) {
  if(CR_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          u_cr(CRE,ke+k,j,i) = u_cr(CRE,ke,j,i) + k*(u_cr(CRE,ke,j,i)-u_cr(CRE,ke-1,j,i));
          u_cr(CRF1,ke+k,j,i) = u_cr(CRF1,ke,j,i);
          u_cr(CRF2,ke+k,j,i) = u_cr(CRF2,ke,j,i);
          if (u_cr(CRF3,ke,j,i)>=0.)
            u_cr(CRF3,ke+k,j,i) = u_cr(CRF3,ke,j,i);
          else
            u_cr(CRF3,ke+k,j,i) = 0;
        }
      }
    }
  }
}
