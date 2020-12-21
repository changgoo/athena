//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file from_athena_vtk.cpp
//! \brief problem generator, initialize mesh by reading in athena rst files.
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
#include "../defs.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

#define MAXLEN 256

//function to split a string into a vector
static std::vector<std::string> split(std::string str, char delimiter);
//function to get rid of white space leading/trailing a string
static void trim(std::string &s);
static void ath_bswap(void *vdat, int len, int cnt);
//functions to read data field from vtk files
static void read_vtk(MeshBlock *mb, std::string vtkdir, std::string vtkfile0,
  std::string field, int component, AthenaArray<Real> &data);

//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator to initialize mesh by reading in athena rst files
//!
//! \note This Problem Generator can be used only to read vtk files corresponding to
//! times t that are integer multiples of 1/Omega, where Omega is the Galactic Rotation

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

  std::int64_t nsize = pmy_mesh->GetTotalCells();

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
