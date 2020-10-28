//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file from_athena_rst.cpp
//! \brief problem generator, initialize mesh by reading in athena rst files.
//======================================================================================

// C++ headers
#include <algorithm>  // min, max
#include <cmath>
#include <cstring>    // strcmp()
#include <string>     // c_str()
#include <iostream>   // endl
#include <fstream>
#include <sstream>    // stringstream

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../defs.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator to initialize mesh by reading in athena rst files

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  int ierr;

  //dimensions of meshblock
  const int Nx = ie - is + 1;
  const int Ny = je - js + 1;
  const int Nz = ke - ks + 1;
  //dimensions of mesh
  const int Nx_mesh = pmy_mesh->mesh_size.nx1;
  const int Ny_mesh = pmy_mesh->mesh_size.nx2;
  const int Nz_mesh = pmy_mesh->mesh_size.nx3;

  AthenaArray<Real> data; //temporary array to store data of the entire mesh
  data.NewAthenaArray(Nz_mesh, Ny_mesh, Nx_mesh);
  AthenaArray<Real> b; //needed for PrimitiveToConserved()
  b.NewAthenaArray(Nz,Ny,Nz);

  std::stringstream msg; //error message
  std::string vtkfile; //corresponding vtk file for this meshblock
  std::string vtkdir = pin->GetString("problem", "vtkdirectory");
  std::string vtkfile0 = pin->GetString("problem", "vtkfile");
  int gis = loc.lx1 * Nx;
  int gjs = loc.lx2 * Ny;
  int gks = loc.lx3 * Nz;

  //TIGRESS parameters//
  int first_id = pin->GetInteger("problem", "first_id");
  int last_id = pin->GetInteger("problem", "last_id");
  int tigress_xmeshblocks = pin->GetInteger("problem", "tigress_xmeshblocks");
  int tigress_ymeshblocks = pin->GetInteger("problem", "tigress_ymeshblocks");
  int tigress_zmeshblocks = pin->GetInteger("problem", "tigress_zmeshblocks");
  int Nx_mesh_tigress = pin->GetInteger("problem", "tigress_Nxmesh");
  int Ny_mesh_tigress = pin->GetInteger("problem", "tigress_Nymesh");
  int Nz_mesh_tigress = pin->GetInteger("problem", "tigress_Nzmesh");
  int Nx_tigress = Nx_mesh_tigress/tigress_xmeshblocks;
  int Ny_tigress = Ny_mesh_tigress/tigress_ymeshblocks;
  int Nz_tigress = Nz_mesh_tigress/tigress_zmeshblocks;

  Real x1size = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  Real x2size = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
  Real x3size = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;

if ((last_id-first_id)*Nz_tigress != Nz_mesh) printf ("Grid parameter check %d \n", (last_id-first_id)*Nz_tigress);

  if (Globals::my_rank == 0) {
    for(int k=first_id; k<last_id; ++k) {
      for (int j=0; j<tigress_ymeshblocks; ++j) {
        for (int i=0; i<tigress_xmeshblocks; ++i) {
          //find the corespoinding athena4.2 global id
          long int id_old = i + j * tigress_xmeshblocks 
            + k * tigress_xmeshblocks * tigress_ymeshblocks;
          //get vtk file name .../id#/problem-id#.????.vtk
          std::stringstream id_str_stream;
          id_str_stream << "id" << id_old;// id#
          std::string id_str = id_str_stream.str();
          std::size_t pos1 = vtkfile0.find_last_of('/');//last /
          std::string vtk_name0 = vtkfile0.substr(pos1);// "/bala.????.vtk"
          std::size_t pos3 = vtk_name0.find_first_of('.');
          std::string vtk_name;
          if (k == 0) vtk_name = vtk_name0.substr(0, pos3) + vtk_name0.substr(pos3);
          else vtk_name = vtk_name0.substr(0, pos3) + "-" + id_str + vtk_name0.substr(pos3);
          vtkfile = vtkdir + vtk_name;
          if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading Density in %s \n", vtkfile.c_str());
          read_vtk(this, vtkfile, "density", 0, data, i*Nx_tigress, j*Ny_tigress, (k-first_id)*Nz_tigress);
    }}}
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phydro->w(IDN, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);
  }}}

  if (Globals::my_rank == 0) {
    for(int k=first_id; k<last_id; ++k) {
      for (int j=0; j<tigress_ymeshblocks; ++j) {
        for (int i=0; i<tigress_xmeshblocks; ++i) {
          //find the corespoinding athena4.2 global id
          long int id_old = i + j * tigress_xmeshblocks 
            + k * tigress_xmeshblocks * tigress_ymeshblocks;
          //get vtk file name .../id#/problem-id#.????.vtk
          std::stringstream id_str_stream;
          id_str_stream << "id" << id_old;// id#
          std::string id_str = id_str_stream.str();
          std::size_t pos1 = vtkfile0.find_last_of('/');//last /
          std::string vtk_name0 = vtkfile0.substr(pos1);// "/bala.????.vtk"
          std::size_t pos3 = vtk_name0.find_first_of('.');
          std::string vtk_name;
          if (k == 0) vtk_name = vtk_name0.substr(0, pos3) + vtk_name0.substr(pos3);
          else vtk_name = vtk_name0.substr(0, pos3) + "-" + id_str + vtk_name0.substr(pos3);
          vtkfile = vtkdir + vtk_name;
          if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading Pressure in %s \n", vtkfile.c_str());
          read_vtk(this, vtkfile, "pressure", 0, data, i*Nx_tigress, j*Ny_tigress, (k-first_id)*Nz_tigress);
    }}}
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phydro->w(IPR, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);
  }}}

  if (Globals::my_rank == 0) {
    for(int k=first_id; k<last_id; ++k) {
      for (int j=0; j<tigress_ymeshblocks; ++j) {
        for (int i=0; i<tigress_xmeshblocks; ++i) {
          //find the corespoinding athena4.2 global id
          long int id_old = i + j * tigress_xmeshblocks 
            + k * tigress_xmeshblocks * tigress_ymeshblocks;
          //get vtk file name .../id#/problem-id#.????.vtk
          std::stringstream id_str_stream;
          id_str_stream << "id" << id_old;// id#
          std::string id_str = id_str_stream.str();
          std::size_t pos1 = vtkfile0.find_last_of('/');//last /
          std::string vtk_name0 = vtkfile0.substr(pos1);// "/bala.????.vtk"
          std::size_t pos3 = vtk_name0.find_first_of('.');
          std::string vtk_name;
          if (k == 0) vtk_name = vtk_name0.substr(0, pos3) + vtk_name0.substr(pos3);
          else vtk_name = vtk_name0.substr(0, pos3) + "-" + id_str + vtk_name0.substr(pos3);
          vtkfile = vtkdir + vtk_name;
          if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading velocity x in %s \n", vtkfile.c_str());
          read_vtk(this, vtkfile, "velocity", 0, data, i*Nx_tigress, j*Ny_tigress, (k-first_id)*Nz_tigress);
    }}}
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
    if (pcr->v_ini > 0) phydro->w(IVX, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);
    else phydro->w(IVX, k, j, i) = 0;
  }}}

  if (Globals::my_rank == 0) {
    for(int k=first_id; k<last_id; ++k) {
      for (int j=0; j<tigress_ymeshblocks; ++j) {
        for (int i=0; i<tigress_xmeshblocks; ++i) {
          //find the corespoinding athena4.2 global id
          long int id_old = i + j * tigress_xmeshblocks 
            + k * tigress_xmeshblocks * tigress_ymeshblocks;
          //get vtk file name .../id#/problem-id#.????.vtk
          std::stringstream id_str_stream;
          id_str_stream << "id" << id_old;// id#
          std::string id_str = id_str_stream.str();
          std::size_t pos1 = vtkfile0.find_last_of('/');//last /
          std::string vtk_name0 = vtkfile0.substr(pos1);// "/bala.????.vtk"
          std::size_t pos3 = vtk_name0.find_first_of('.');
          std::string vtk_name;
          if (k == 0) vtk_name = vtk_name0.substr(0, pos3) + vtk_name0.substr(pos3);
          else vtk_name = vtk_name0.substr(0, pos3) + "-" + id_str + vtk_name0.substr(pos3);
          vtkfile = vtkdir + vtk_name;
          if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading velocity y in %s \n", vtkfile.c_str());
          read_vtk(this, vtkfile, "velocity", 1, data, i*Nx_tigress, j*Ny_tigress, (k-first_id)*Nz_tigress);
    }}}
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
    if (pcr->v_ini> 0) phydro->w(IVY, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);
    else phydro->w(IVY, k, j, i) = 0;
  }}}

  if (Globals::my_rank == 0) {
    for(int k=first_id; k<last_id; ++k) {
      for (int j=0; j<tigress_ymeshblocks; ++j) {
        for (int i=0; i<tigress_xmeshblocks; ++i) {
          //find the corespoinding athena4.2 global id
          long int id_old = i + j * tigress_xmeshblocks 
            + k * tigress_xmeshblocks * tigress_ymeshblocks;
          //get vtk file name .../id#/problem-id#.????.vtk
          std::stringstream id_str_stream;
          id_str_stream << "id" << id_old;// id#
          std::string id_str = id_str_stream.str();
          std::size_t pos1 = vtkfile0.find_last_of('/');//last /
          std::string vtk_name0 = vtkfile0.substr(pos1);// "/bala.????.vtk"
          std::size_t pos3 = vtk_name0.find_first_of('.');
          std::string vtk_name;
          if (k == 0) vtk_name = vtk_name0.substr(0, pos3) + vtk_name0.substr(pos3);
          else vtk_name = vtk_name0.substr(0, pos3) + "-" + id_str + vtk_name0.substr(pos3);
          vtkfile = vtkdir + vtk_name;
          if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading velocity z in %s \n", vtkfile.c_str());
          read_vtk(this, vtkfile, "velocity", 2, data, i*Nx_tigress, j*Ny_tigress, (k-first_id)*Nz_tigress);
    }}}
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
    if (pcr->v_ini > 0) phydro->w(IVZ, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);
    else phydro->w(IVZ, k, j, i) = 0;
  }}}  

  if(MAGNETIC_FIELDS_ENABLED){
  
  if (Globals::my_rank == 0) {
    for(int k=first_id; k<last_id; ++k) {
      for (int j=0; j<tigress_ymeshblocks; ++j) {
      for (int i=0; i<tigress_xmeshblocks; ++i) {
        //find the corespoinding athena4.2 global id
        long int id_old = i + j * tigress_xmeshblocks 
          + k * tigress_xmeshblocks * tigress_ymeshblocks;
        //get vtk file name .../id#/problem-id#.????.vtk
        std::stringstream id_str_stream;
        id_str_stream << "id" << id_old;// id#
        std::string id_str = id_str_stream.str();
        std::size_t pos1 = vtkfile0.find_last_of('/');//last /
        std::string vtk_name0 = vtkfile0.substr(pos1);// "/bala.????.vtk"
        std::size_t pos3 = vtk_name0.find_first_of('.');
        std::string vtk_name;
        if (k == 0) vtk_name = vtk_name0.substr(0, pos3) + vtk_name0.substr(pos3);
        else vtk_name = vtk_name0.substr(0, pos3) + "-" + id_str + vtk_name0.substr(pos3);
        vtkfile = vtkdir + vtk_name;
        if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading Bx in %s \n", vtkfile.c_str());
        read_vtk(this, vtkfile, "cell_centered_B", 0, data, i*Nx_tigress, j*Ny_tigress, (k-first_id)*Nz_tigress);
      }}}
    }

  MPI_Barrier(MPI_COMM_WORLD);
  ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  
  AthenaArray<Real> data_B; //temporary array to store data of the entire mesh
  data_B.NewAthenaArray(Nz_mesh, Ny_mesh, Nx_mesh+2);

  for (int k=0; k<=Nz_mesh-1; ++k) {
    for (int j=0; j<=Ny_mesh-1; ++j) {
      for (int i=0; i<=Nx_mesh-1; ++i) {
        data_B(k,j,i+1) = data(k,j,i);
      }  
    data_B(k,j,0) = data(k,j,Nx_mesh-1); //periodic boundary conditions
    data_B(k,j,Nx_mesh+1) = data(k,j,0);
  }}

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie+1; ++i) {
        pfield->b.x1f(k,j,i) = 0.5*(data_B(k-ks+gks, j-js+gjs, i-is+gis)+data_B(k-ks+gks, j-js+gjs, i+1-is+gis));  
  }}}  

  data_B.DeleteAthenaArray();

  if (Globals::my_rank == 0) {
    for(int k=first_id; k<last_id; ++k) {
      for (int j=0; j<tigress_ymeshblocks; ++j) {
        for (int i=0; i<tigress_xmeshblocks; ++i) {
          //find the corespoinding athena4.2 global id
          long int id_old = i + j * tigress_xmeshblocks 
            + k * tigress_xmeshblocks * tigress_ymeshblocks;
          //get vtk file name .../id#/problem-id#.????.vtk
          std::stringstream id_str_stream;
          id_str_stream << "id" << id_old;// id#
          std::string id_str = id_str_stream.str();
          std::size_t pos1 = vtkfile0.find_last_of('/');//last /
          std::string vtk_name0 = vtkfile0.substr(pos1);// "/bala.????.vtk"
          std::size_t pos3 = vtk_name0.find_first_of('.');
          std::string vtk_name;
          if (k == 0) vtk_name = vtk_name0.substr(0, pos3) + vtk_name0.substr(pos3);
          else vtk_name = vtk_name0.substr(0, pos3) + "-" + id_str + vtk_name0.substr(pos3);
          vtkfile = vtkdir + vtk_name;
          if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading By in %s \n", vtkfile.c_str());
          read_vtk(this, vtkfile, "cell_centered_B", 1, data, i*Nx_tigress, j*Ny_tigress, (k-first_id)*Nz_tigress);
    }}}
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  
  data_B.NewAthenaArray(Nz_mesh, Ny_mesh+2, Nx_mesh);

  for (int k=0; k<=Nz_mesh-1; ++k) {
    for (int i=0; i<=Nx_mesh-1; ++i) {
      for (int j=0; j<=Ny_mesh-1; ++j) {
        data_B(k,j+1,i) = data(k,j,i);
    }  
    data_B(k,0,i) = data(k,Ny_mesh-1,i); //periodic boundary conditions
    data_B(k,Ny_mesh+1,i) = data(k,0,i);
  }}

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
      for (int i=is; i<=ie; ++i) {
        pfield->b.x2f(k,j,i) = 0.5*(data_B(k-ks+gks, j-js+gjs, i-is+gis)+data_B(k-ks+gks, j+1-js+gjs, i-is+gis));  
    }}}  

  data_B.DeleteAthenaArray();


  if (Globals::my_rank == 0) {
    for(int k=first_id; k<last_id; ++k) {
      for (int j=0; j<tigress_ymeshblocks; ++j) {
        for (int i=0; i<tigress_xmeshblocks; ++i) {
          //find the corespoinding athena4.2 global id
          long int id_old = i + j * tigress_xmeshblocks 
            + k * tigress_xmeshblocks * tigress_ymeshblocks;
          //get vtk file name .../id#/problem-id#.????.vtk
          std::stringstream id_str_stream;
          id_str_stream << "id" << id_old;// id#
          std::string id_str = id_str_stream.str();
          std::size_t pos1 = vtkfile0.find_last_of('/');//last /
          std::string vtk_name0 = vtkfile0.substr(pos1);// "/bala.????.vtk"
          std::size_t pos3 = vtk_name0.find_first_of('.');
          std::string vtk_name;
          if (k == 0) vtk_name = vtk_name0.substr(0, pos3) + vtk_name0.substr(pos3);
          else vtk_name = vtk_name0.substr(0, pos3) + "-" + id_str + vtk_name0.substr(pos3);
          vtkfile = vtkdir + vtk_name;
          if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading Bz in %s \n", vtkfile.c_str());
          read_vtk(this, vtkfile, "cell_centered_B", 2, data, i*Nx_tigress, j*Ny_tigress, (k-first_id)*Nz_tigress);
    }}}
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  
  data_B.NewAthenaArray(Nz_mesh+2, Ny_mesh, Nx_mesh);

  for (int i=0; i<=Nx_mesh-1; ++i) {
    for (int j=0; j<=Ny_mesh-1; ++j) {
      for (int k=0; k<=Nz_mesh-1; ++k) {
        data_B(k+1,j,i) = data(k,j,i);
      }  
    data_B(0,j,i) = data(0,j,i); //outflow boundary conditions
    data_B(Nz_mesh+1,j,i) = data(Nz_mesh-1,j,i);
  }}

  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        pfield->b.x3f(k,j,i) = 0.5*(data_B(k-ks+gks, j-js+gjs, i-is+gis)+data_B(k+1-ks+gks, j-js+gjs, i-is+gis));  
  }}}  

  data_B.DeleteAthenaArray();  

  pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);

  }//End MHD

     
  data.DeleteAthenaArray();
  
  if(MAGNETIC_FIELDS_ENABLED){
    peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                                         is, ie, js, je, ks, ke);
  } else {                   
    peos->PrimitiveToConserved(phydro->w, b, phydro->u, pcoord,
                                     is, ie, js, je, ks, ke);
  }
  b.DeleteAthenaArray();          
  return;
}

static void read_vtk(MeshBlock *mb, std::string filename, std::string field,
                    int component, AthenaArray<Real> &data, int xstart, int ystart, int zstart) 
{
  std::stringstream msg;
  FILE *fp = NULL;
  char cline[256], type[256], variable[256], format[256], t_type[256], t_format[256];
  std::string line;
  const std::string athena_header = "# vtk DataFile Version 2.0"; //athena4.2 header
  const std::string athena_header3 = "# vtk DataFile Version 3.0"; //athena4.2 header
  bool SHOW_OUTPUT = false;
  int Nx_vtk, Ny_vtk, Nz_vtk; //dimensions of vtk files
  //dimensions of meshblock
  int Nx_mb, Ny_mb, Nz_mb;
  
  Nx_mb = mb->ie - mb->is + 1;
  Ny_mb = mb->je - mb->js + 1;
  Nz_mb = mb->ke - mb->ks + 1;

  double ox_vtk, oy_vtk, oz_vtk; //origins of vtk file
  double dx_vtk, dy_vtk, dz_vtk; //spacings of vtk file
  int cell_dat_vtk; //total number of cells in vtk file
  //total number of cells in MeshBlock
  const int cell_dat_mb = Nx_mb * Ny_mb * Nz_mb; 
  int retval, nread; //file handler return value
  float fdat, fvec[3], ften[9];//store float format scaler, vector, and tensor

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
  if (line != "DATASET STRUCTURED_POINTS") {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Unsupported file data set structure: " << line << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  //I'm assuming from this point on that the header is in good shape 
  
  //read dimensions
  fgets(cline,256,fp);
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
  fgets(cline,256,fp);
  if (SHOW_OUTPUT) {
    std::cout << cline;
  }
  sscanf(cline,"ORIGIN %le %le %le\n",&ox_vtk,&oy_vtk,&oz_vtk);

  // spacing
  fgets(cline,256,fp);
  if (SHOW_OUTPUT) {
    std::cout << cline;
  }
  sscanf(cline,"ORIGIN %le %le %le\n",&dx_vtk,&dy_vtk,&dz_vtk);

  // Cell Data = Nx*Ny*Nz
  fgets(cline,256,fp);
  if (SHOW_OUTPUT) {
    std::cout << cline;
  }
  sscanf(cline,"CELL_DATA %d\n",&cell_dat_vtk);

  // Now read the rest of the data in 
  while (true) {
 
      retval = fscanf(fp,"%s %s %s\n", type, variable, format);
      if (retval == EOF) { // Assuming no errors, we are done.
        fclose(fp); //close file
        return;
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

      //determine variable type and read data
      //read scalars
      if (strcmp(type, "SCALARS") == 0) {
        if (strcmp(variable, field.c_str()) == 0) {      
          //printf("  Reading %s...\n", variable);
          for (int k=0; k<Nz_vtk; k++) {
            for (int j=0; j<Ny_vtk; j++) {
              for (int i=0; i<Nx_vtk; i++) {
                if ((nread = fread(&fdat, sizeof(float), 1, fp)) != 1) {
                  fclose(fp);
                  msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
                    << "Error reading SCALARS... " << std::endl;
                  throw std::runtime_error(msg.str().c_str());
                }
                ath_bswap(&fdat, sizeof(float), 1);
                data(k+zstart, j+ystart, i+xstart) = fdat;
              }
            }
          }
          fclose(fp);
          return;
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
                if ((nread = fread(&fvec, sizeof(float), 3, fp)) != 3) {
                  fclose(fp);
                  msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
                    << "Error reading VECTORS... " << std::endl;
                  throw std::runtime_error(msg.str().c_str());
                }
                ath_bswap(&fvec, sizeof(float), 3);
                data(k+zstart, j+ystart, i+xstart) = fvec[component];
              }
            }
          }
          fclose(fp);
          return;
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

//======================================================================================
//! \fn static void ath_bswap(void *vdat, int len, int cnt)

//!  \brief Swap bytes, code stolen from Athena4.2, NEMO
//======================================================================================
static void ath_bswap(void *vdat, int len, int cnt)
{
  char tmp, *dat = (char *) vdat;
  int k;
 
  if (len==1)
    return;
  else if (len==2)
    while (cnt--) {
      tmp = dat[0];  dat[0] = dat[1];  dat[1] = tmp;
      dat += 2;
    }
  else if (len==4)
    while (cnt--) {
      tmp = dat[0];  dat[0] = dat[3];  dat[3] = tmp;
      tmp = dat[1];  dat[1] = dat[2];  dat[2] = tmp;
      dat += 4;
    }
  else if (len==8)
    while (cnt--) {
      tmp = dat[0];  dat[0] = dat[7];  dat[7] = tmp;
      tmp = dat[1];  dat[1] = dat[6];  dat[6] = tmp;
      tmp = dat[2];  dat[2] = dat[5];  dat[5] = tmp;
      tmp = dat[3];  dat[3] = dat[4];  dat[4] = tmp;
      dat += 8;
    }
  else {  /* the general SLOOOOOOOOOW case */
    for(k=0; k<len/2; k++) {
      tmp = dat[k];
      dat[k] = dat[len-1-k];
      dat[len-1-k] = tmp;
    }
  }
}

//======================================================================================
//! \fn static void trim(std::string &s)
//!  \brief get rid of white spaces leading and trailing a string
//======================================================================================
static void trim(std::string &s)
{
  size_t p = s.find_first_not_of(" \t\n");
  s.erase(0, p);

  p = s.find_last_not_of(" \t\n");
  if (p != std::string::npos) {
    s.erase(p+1);
  }
}

//======================================================================================
//! \fn static std::vector<std::string> split(std::string str, char delimiter)
//!  \brief split a string, and store sub strings in a vector
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
