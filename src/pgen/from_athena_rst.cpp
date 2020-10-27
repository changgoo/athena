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

static void read_rst(std::string filename, std::string field, AthenaArray<Real> &data,  
                     int iu, int ju, int ku,
                     int xstart = 0, int ystart = 0, int zstart = 0, int flagB = 0) 
{
  std::stringstream msg;
  int MAXLEN = 256;
  FILE *fp;
  char line[MAXLEN];
  
  Real fdat; 
  
  //Open the restart file    
  if((fp = fopen(filename.c_str(),"r")) == NULL){
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
    << "Error opening the restart file" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  
  //Search for the field to read in
  do{
    fgets(line,MAXLEN,fp);
  }while(strncmp(line,field.c_str(),field.size()) != 0);

  //Read the field 
  if(strncmp(line,field.c_str(),field.size()) != 0) {
    msg << "Expected " << field.c_str() << ", found " << line << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  for (int k=0; k<ku; k++) {
    for (int j=0; j<ju; j++) {
      for (int i=0; i<iu; i++) {
        fread(&fdat,sizeof(Real),1,fp);
        if ((flagB == 1 && i==iu-1) || (flagB == 2 && j==ju-1) || (flagB == 3 && k==ku-1)) continue;
        else data(k+zstart, j+ystart, i+xstart) = fdat;
      }
    }
  }
  
  fclose(fp);
  
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
// \brief Problem Generator to initialize mesh by reading in athena rst files

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  std::stringstream msg; //error message
  std::string rstfile; //corresponding rst file for this meshblock
  std::string rstdir = pin->GetString("problem", "rst_directory");
  std::string rstfile0 = pin->GetString("problem", "rst_file");

  int flag = pin->GetOrAddInteger("problem", "flag_rst", 0);
  
  //dimensions of meshblock
  const int Nx = ie - is + 1;
  const int Ny = je - js + 1;
  const int Nz = ke - ks + 1;
  //dimensions of mesh
  const int Nx_mesh = pmy_mesh->mesh_size.nx1;
  const int Ny_mesh = pmy_mesh->mesh_size.nx2;
  const int Nz_mesh = pmy_mesh->mesh_size.nx3;
  
  int gis = loc.lx1 * Nx;
  int gjs = loc.lx2 * Ny;
  int gks = loc.lx3 * Nz;
  
  if (flag == 1){
    //find the corespoinding athena4.2 global id
    long int id_old = loc.lx1 + loc.lx2 * pmy_mesh->nrbx1 
      + loc.lx3 * pmy_mesh->nrbx1 * pmy_mesh->nrbx2;
    std::stringstream id_str_stream;
    id_str_stream << "id" << id_old;// id#
    std::string id_str = id_str_stream.str();
    std::size_t pos1 = rstfile0.find_last_of('/');
    std::size_t pos2 = rstfile0.find_last_of('/', pos1-1);
    std::string rst_name0 = rstfile0.substr(pos1);
    std::size_t pos3 = rst_name0.find_first_of('.');
    std::string rst_name;
    if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0){
      rst_name = rst_name0.substr(0, pos3) + rst_name0.substr(pos3);
    }else{
      rst_name = rst_name0.substr(0, pos3) + "-" + id_str + rst_name0.substr(pos3);
    }
    rstfile = rstdir + rst_name;
    std::cout<<rstdir + rst_name<<std::endl;
    
    AthenaArray<Real> data; //temporary array to store data of the entire mesh
    data.NewAthenaArray(Nz, Ny, Nx);
            
    //Read the density
    read_rst(rstfile, "DENSITY", data, Nx, Ny, Nz);
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IDN, k, j, i) = data(k-ks, j-js, i-is);
    }}}
    
    //Read the x1-momentum
    read_rst(rstfile, "1-MOMENTUM", data, Nx, Ny, Nz);
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IM1, k, j, i) = data(k-ks, j-js, i-is);
    }}}
    
    //Read the x2-momentum
    read_rst(rstfile, "2-MOMENTUM", data, Nx, Ny, Nz);
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IM2, k, j, i) = data(k-ks, j-js, i-is);
    }}}
    
    //Read the x3-momentum
    read_rst(rstfile, "3-MOMENTUM", data, Nx, Ny, Nz);
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IM3, k, j, i) = data(k-ks, j-js, i-is);
    }}}
    
    //Read the energy density
    read_rst(rstfile, "ENERGY", data, Nx, Ny, Nz);
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IEN, k, j, i) = data(k-ks, j-js, i-is);
    }}}

    data.DeleteAthenaArray();
    
    if (MAGNETIC_FIELDS_ENABLED) {
      
      AthenaArray<Real> data_b; 
      
      //Read the face-centered x1 B-field 
      data_b.NewAthenaArray(Nz,Ny,Nx+1);
      read_rst(rstfile, "1-FIELD", data_b, Nx+1, Ny, Nz);
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie+1; ++i) {
            pfield->b.x1f(k,j,i) = data_b(k-ks, j-js, i-is);
      }}}
      data_b.DeleteAthenaArray();	
    
      //Read the face-centered x2 B-field 
      data_b.NewAthenaArray(Nz,Ny+1,Nx);
      read_rst(rstfile, "2-FIELD", data_b, Nx, Ny+1, Nz);
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x2f(k,j,i) = data_b(k-ks, j-js, i-is);
      }}}
      data_b.DeleteAthenaArray();	
    
      //Read the face-centered x3 B-field 
      data_b.NewAthenaArray(Nz+1,Ny,Nx);
      read_rst(rstfile, "3-FIELD", data_b, Nx, Ny, Nz+1);
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x3f(k,j,i) = data_b(k-ks, j-js, i-is);
      }}}
      data_b.DeleteAthenaArray();	
      
      pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);
      
   }
  } else {
    
    int tigress_zmeshblocks, tigress_ymeshblocks, tigress_xmeshblocks;
    int tigress_Nx_mesh, tigress_Ny_mesh, tigress_Nz_mesh;
    int tigress_Nx, tigress_Ny, tigress_Nz;

    std::stringstream msg;
    int MAXLEN = 256;
    FILE *fp;
    char line[MAXLEN];
  
    //Open the restart file    
    rstfile = rstdir + rstfile0;
    if((fp = fopen(rstfile.c_str(),"r")) == NULL){
      msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Error opening the restart file" << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }

    while(fgets(line,MAXLEN,fp))
    {
      std::string s(line);
      if(s.find("Nx1")!=std::string::npos)
        tigress_Nx_mesh = atoi(s.substr(s.find("=")+2,4).c_str());
      if(s.find("NGrid_x1")!=std::string::npos)
        tigress_xmeshblocks = atoi(s.substr(s.find("=")+2,3).c_str());
      if(s.find("Nx2")!=std::string::npos)
        tigress_Ny_mesh = atoi(s.substr(s.find("=")+2,4).c_str());
      if(s.find("NGrid_x2")!=std::string::npos)
        tigress_ymeshblocks = atoi(s.substr(s.find("=")+2,3).c_str());
      if(s.find("Nx3")!=std::string::npos)
        tigress_Nz_mesh = atoi(s.substr(s.find("=")+2,4).c_str());
      if(s.find("NGrid_x3")!=std::string::npos){
        tigress_zmeshblocks = atoi(s.substr(s.find("=")+2,3).c_str());
        break;
      }  
    } 
    fclose(fp);
    
    tigress_Nx = tigress_Nx_mesh/tigress_xmeshblocks;
    tigress_Ny = tigress_Ny_mesh/tigress_ymeshblocks;
    tigress_Nz = tigress_Nz_mesh/tigress_zmeshblocks;
      
    std::size_t pos1 = rstfile0.find_last_of('/');
    std::size_t pos2 = rstfile0.find_last_of('/', pos1-1);
    std::string rst_name0 = rstfile0.substr(pos1);
    std::size_t pos3 = rst_name0.find_first_of('.');
    std::string rst_name;
    
    AthenaArray<Real> data; //temporary array to store data of the entire mesh
    data.NewAthenaArray(Nz_mesh, Ny_mesh, Nx_mesh);
    
    if (Globals::my_rank == 0) {
      if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading density ... \n");
      for(int k=0; k<tigress_zmeshblocks; ++k) {
        for (int j=0; j<tigress_ymeshblocks; ++j) {
          for (int i=0; i<tigress_xmeshblocks; ++i) {
            //find the corespoinding athena4.2 global id
            long int id_old = i + j * tigress_xmeshblocks 
            + k * tigress_xmeshblocks * tigress_ymeshblocks;
            std::stringstream id_str_stream;
            id_str_stream << "id" << id_old;// id#
            std::string id_str = id_str_stream.str();
            if (i == 0 && j == 0 && k == 0){
              rst_name = rst_name0.substr(0, pos3) + rst_name0.substr(pos3);
            }else{
              rst_name = rst_name0.substr(0, pos3) + "-" + id_str + rst_name0.substr(pos3);
            }
            rstfile = rstdir + rst_name;
            
            int xs = i*tigress_Nx;
            int ys = j*tigress_Ny;
            int zs = k*tigress_Nz;
            
            read_rst(rstfile, "DENSITY", data, tigress_Nx, tigress_Ny, tigress_Nz, xs, ys, zs);
    }}}}

    int ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
      
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IDN, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);
    }}}
    
    if (Globals::my_rank == 0) {
      if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading x1-momentum ... \n");
      for(int k=0; k<tigress_zmeshblocks; ++k) {
        for (int j=0; j<tigress_ymeshblocks; ++j) {
          for (int i=0; i<tigress_xmeshblocks; ++i) {
            //find the corespoinding athena4.2 global id
            long int id_old = i + j * tigress_xmeshblocks 
            + k * tigress_xmeshblocks * tigress_ymeshblocks;
            std::stringstream id_str_stream;
            id_str_stream << "id" << id_old;// id#
            std::string id_str = id_str_stream.str();
            if (i == 0 && j == 0 && k == 0){
              rst_name = rst_name0.substr(0, pos3) + rst_name0.substr(pos3);
            }else{
              rst_name = rst_name0.substr(0, pos3) + "-" + id_str + rst_name0.substr(pos3);
            }
            rstfile = rstdir + rst_name;
            
            int xs = i*tigress_Nx;
            int ys = j*tigress_Ny;
            int zs = k*tigress_Nz;
            
            read_rst(rstfile, "1-MOMENTUM", data, tigress_Nx, tigress_Ny, tigress_Nz, xs, ys, zs);
    }}}}

    ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
      
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IM1, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);
    }}}
    
    if (Globals::my_rank == 0) {
      if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading x2-momentum ... \n");
      for(int k=0; k<tigress_zmeshblocks; ++k) {
        for (int j=0; j<tigress_ymeshblocks; ++j) {
          for (int i=0; i<tigress_xmeshblocks; ++i) {
            //find the corespoinding athena4.2 global id
            long int id_old = i + j * tigress_xmeshblocks 
            + k * tigress_xmeshblocks * tigress_ymeshblocks;
            std::stringstream id_str_stream;
            id_str_stream << "id" << id_old;// id#
            std::string id_str = id_str_stream.str();
            if (i == 0 && j == 0 && k == 0){
              rst_name = rst_name0.substr(0, pos3) + rst_name0.substr(pos3);
            }else{
              rst_name = rst_name0.substr(0, pos3) + "-" + id_str + rst_name0.substr(pos3);
            }
            rstfile = rstdir + rst_name;
            
            int xs = i*tigress_Nx;
            int ys = j*tigress_Ny;
            int zs = k*tigress_Nz;
            
            read_rst(rstfile, "2-MOMENTUM", data, tigress_Nx, tigress_Ny, tigress_Nz, xs, ys, zs);
    }}}}

    ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
      
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IM2, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);
    }}}
    
    if (Globals::my_rank == 0) {
      if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading x3-momentum ... \n");
      for(int k=0; k<tigress_zmeshblocks; ++k) {
        for (int j=0; j<tigress_ymeshblocks; ++j) {
          for (int i=0; i<tigress_xmeshblocks; ++i) {
            //find the corespoinding athena4.2 global id
            long int id_old = i + j * tigress_xmeshblocks 
            + k * tigress_xmeshblocks * tigress_ymeshblocks;
            std::stringstream id_str_stream;
            id_str_stream << "id" << id_old;// id#
            std::string id_str = id_str_stream.str();
            if (i == 0 && j == 0 && k == 0){
              rst_name = rst_name0.substr(0, pos3) + rst_name0.substr(pos3);
            }else{
              rst_name = rst_name0.substr(0, pos3) + "-" + id_str + rst_name0.substr(pos3);
            }
            rstfile = rstdir + rst_name;
            
            int xs = i*tigress_Nx;
            int ys = j*tigress_Ny;
            int zs = k*tigress_Nz;
            
            read_rst(rstfile, "3-MOMENTUM", data, tigress_Nx, tigress_Ny, tigress_Nz, xs, ys, zs);
    }}}}

    ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
      
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IM3, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);
    }}}
    
    if (Globals::my_rank == 0) {
      if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading energy density ... \n");
      for(int k=0; k<tigress_zmeshblocks; ++k) {
        for (int j=0; j<tigress_ymeshblocks; ++j) {
          for (int i=0; i<tigress_xmeshblocks; ++i) {
            //find the corespoinding athena4.2 global id
            long int id_old = i + j * tigress_xmeshblocks 
            + k * tigress_xmeshblocks * tigress_ymeshblocks;
            std::stringstream id_str_stream;
            id_str_stream << "id" << id_old;// id#
            std::string id_str = id_str_stream.str();
            if (i == 0 && j == 0 && k == 0){
              rst_name = rst_name0.substr(0, pos3) + rst_name0.substr(pos3);
            }else{
              rst_name = rst_name0.substr(0, pos3) + "-" + id_str + rst_name0.substr(pos3);
            }
            rstfile = rstdir + rst_name;
            
            int xs = i*tigress_Nx;
            int ys = j*tigress_Ny;
            int zs = k*tigress_Nz;
            
            read_rst(rstfile, "ENERGY", data, tigress_Nx, tigress_Ny, tigress_Nz, xs, ys, zs);
    }}}}

    ierr = MPI_Bcast(data.data(), Nx_mesh*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
      
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IEN, k, j, i) = data(k-ks+gks, j-js+gjs, i-is+gis);
    }}}
    
    data.DeleteAthenaArray();
    
    if (MAGNETIC_FIELDS_ENABLED) {
      
      AthenaArray<Real> data_b; 
      
      data_b.NewAthenaArray(Nz_mesh, Ny_mesh, Nx_mesh+1);
      if (Globals::my_rank == 0) {
        if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading x1 B-field ... \n");
        for(int k=0; k<tigress_zmeshblocks; ++k) {
          for (int j=0; j<tigress_ymeshblocks; ++j) {
            for (int i=0; i<tigress_xmeshblocks; ++i) {
              //find the corespoinding athena4.2 global id
              long int id_old = i + j * tigress_xmeshblocks 
              + k * tigress_xmeshblocks * tigress_ymeshblocks;
              std::stringstream id_str_stream;
              id_str_stream << "id" << id_old;// id#
              std::string id_str = id_str_stream.str();
              if (i == 0 && j == 0 && k == 0){
                rst_name = rst_name0.substr(0, pos3) + rst_name0.substr(pos3);
              }else{
                rst_name = rst_name0.substr(0, pos3) + "-" + id_str + rst_name0.substr(pos3);
              }
              rstfile = rstdir + rst_name;
            
              int xs = i*tigress_Nx;
              int ys = j*tigress_Ny;
              int zs = k*tigress_Nz;
            
              if (i == tigress_xmeshblocks-1) read_rst(rstfile, "1-FIELD", data_b, tigress_Nx+1, tigress_Ny, tigress_Nz, xs, ys, zs);
              else read_rst(rstfile, "1-FIELD", data_b, tigress_Nx+1, tigress_Ny, tigress_Nz, xs, ys, zs, 1);
      }}}}

      ierr = MPI_Bcast(data_b.data(), (Nx_mesh+1)*Ny_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie+1; ++i) {
            pfield->b.x1f(k,j,i) = data_b(k-ks+gks, j-js+gjs, i-is+gis);
      }}}
      
      data_b.DeleteAthenaArray();	
      
      data_b.NewAthenaArray(Nz_mesh, Ny_mesh+1, Nx_mesh);
      if (Globals::my_rank == 0) {
        if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading x2 B-field ... \n");
        for(int k=0; k<tigress_zmeshblocks; ++k) {
          for (int j=0; j<tigress_ymeshblocks; ++j) {
            for (int i=0; i<tigress_xmeshblocks; ++i) {
              //find the corespoinding athena4.2 global id
              long int id_old = i + j * tigress_xmeshblocks 
              + k * tigress_xmeshblocks * tigress_ymeshblocks;
              std::stringstream id_str_stream;
              id_str_stream << "id" << id_old;// id#
              std::string id_str = id_str_stream.str();
              if (i == 0 && j == 0 && k == 0){
                rst_name = rst_name0.substr(0, pos3) + rst_name0.substr(pos3);
              }else{
                rst_name = rst_name0.substr(0, pos3) + "-" + id_str + rst_name0.substr(pos3);
              }
              rstfile = rstdir + rst_name;
            
              int xs = i*tigress_Nx;
              int ys = j*tigress_Ny;
              int zs = k*tigress_Nz;
            
              if (j == tigress_ymeshblocks-1) read_rst(rstfile, "2-FIELD", data_b, tigress_Nx, tigress_Ny+1, tigress_Nz, xs, ys, zs);
              else read_rst(rstfile, "2-FIELD", data_b, tigress_Nx, tigress_Ny+1, tigress_Nz, xs, ys, zs, 2);
      }}}}

      ierr = MPI_Bcast(data_b.data(), (Ny_mesh+1)*Nx_mesh*Nz_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x2f(k,j,i) = data_b(k-ks+gks, j-js+gjs, i-is+gis);
      }}}
      
      data_b.DeleteAthenaArray();	
      
      data_b.NewAthenaArray(Nz_mesh+1, Ny_mesh, Nx_mesh);
      if (Globals::my_rank == 0) {
        if (loc.lx1 == 0 && loc.lx2 == 0 && loc.lx3 == 0) printf("Reading x3 B-field ... \n");
        for(int k=0; k<tigress_zmeshblocks; ++k) {
          for (int j=0; j<tigress_ymeshblocks; ++j) {
            for (int i=0; i<tigress_xmeshblocks; ++i) {
              //find the corespoinding athena4.2 global id
              long int id_old = i + j * tigress_xmeshblocks 
              + k * tigress_xmeshblocks * tigress_ymeshblocks;
              std::stringstream id_str_stream;
              id_str_stream << "id" << id_old;// id#
              std::string id_str = id_str_stream.str();
              if (i == 0 && j == 0 && k == 0){
                rst_name = rst_name0.substr(0, pos3) + rst_name0.substr(pos3);
              }else{
                rst_name = rst_name0.substr(0, pos3) + "-" + id_str + rst_name0.substr(pos3);
              }
              rstfile = rstdir + rst_name;
            
              int xs = i*tigress_Nx;
              int ys = j*tigress_Ny;
              int zs = k*tigress_Nz;
            
              if (k == tigress_zmeshblocks-1) read_rst(rstfile, "3-FIELD", data_b, tigress_Nx, tigress_Ny, tigress_Nz+1, xs, ys, zs);
              else read_rst(rstfile, "3-FIELD", data_b, tigress_Nx, tigress_Ny, tigress_Nz+1, xs, ys, zs, 3);
      }}}}

      ierr = MPI_Bcast(data_b.data(), (Nz_mesh+1)*Ny_mesh*Nx_mesh, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x3f(k,j,i) = data_b(k-ks+gks, j-js+gjs, i-is+gis);
      }}}
      
      data_b.DeleteAthenaArray();	
      
      pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);
    }
    
  }  
  return;
}
