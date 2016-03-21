//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file read_vtk.c
//  \brief problem generator, initalize mesh by read in vtk files.
//======================================================================================

// C++ headers
#include <string>     // c_str()
#include <iostream>   // endl
#include <vector>     // vector container
#include <sstream>    // stringstream
#include <string>     // c_str()
#include <stdio.h>    // c style file
#include <string.h>   // strcmp()
#include <algorithm>  // std::find()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../field/field.hpp"
#include "../hydro/eos/eos.hpp"
#include "../coordinates/coordinates.hpp"


//function to split a string into a vector
static std::vector<std::string> split(std::string str, char delimiter);
//function to get rid of white space leading/trailing a string
static void trim(std::string &s);
//function to read data field from vtk file
static void readvtk(MeshBlock *mb, std::string filename, std::string field,
                    int component, AthenaArray<float> &data);
//swap bytes
static void ath_bswap(void *vdat, int len, int cnt);

void Mesh::InitUserMeshProperties(ParameterInput *pin)
{
  return;
}

//======================================================================================
//! \fn void Mesh::TerminateUserMeshProperties(void)
//  \brief Clean up the Mesh properties
//======================================================================================
void Mesh::TerminateUserMeshProperties(void)
{
  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief initialize problem by reading in vtk file.
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  //dimensions of meshblock
  const int Nx = ie - is + 1;
  const int Ny = je - js + 1;
  const int Nz = ke - ks + 1;
  AthenaArray<float> data; //temporary array to store data;
  data.NewAthenaArray(Nz, Ny, Nx);
  AthenaArray<Real> b; //needed for PrimitiveToConserved()
  b.NewAthenaArray(Nz, Ny, Nx);
  std::stringstream msg; //error message

  //parse input parameters
  std::string vtkfile = pin->GetString("problem", "vtkfile");
  std::string str_scalers = pin->GetString("problem", "scalers");
  std::string str_vectors = pin->GetString("problem", "vectors");
  std::vector<std::string> scaler_fields = split(str_scalers, ',');
  std::vector<std::string> vector_fields = split(str_vectors, ',');
  
  //---diagnostic output---
  std::cout << "vtkfile = (" << vtkfile << ")" << std::endl;
  std::cout << "scalers = " << str_scalers << std::endl;
  std::cout << "vectors = " << str_vectors << std::endl;


  for(int i = 0; i < vector_fields.size(); ++i) {
    std::cout << "(" << vector_fields[i] << ")" << std::endl;
  }
  //----------------------

  //TODO: need corresponding name for multiple processors
  
  //read scalers
  for(int i = 0; i < scaler_fields.size(); ++i) {
    if (scaler_fields[i] == "density") {
      readvtk(this, vtkfile, "density", 0, data);
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            phydro->w(IDN, k, j, i) = data(k-ks, j-js, i-is);
          }
        }
      }
    } else if (scaler_fields[i] == "pressure") {
      readvtk(this, vtkfile, "pressure", 0, data);
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            phydro->w(IEN, k, j, i) = data(k-ks, j-js, i-is);
          }
        }
      }
    } else {
      msg << "### FATAL ERROR in Problem Generator [ProblemGenerator]" << std::endl
        << "Scaler field not recognized: " << scaler_fields[i] << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
  }
  //read vectors
  for(int i = 0; i < vector_fields.size(); ++i) {
    if (vector_fields[i] == "velocity") {
      //vx
      readvtk(this, vtkfile, "velocity", 0, data);
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            phydro->w(IVX, k, j, i) = data(k-ks, j-js, i-is);
          }
        }
      }
      //vy
      readvtk(this, vtkfile, "velocity", 1, data);
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            phydro->w(IVY, k, j, i) = data(k-ks, j-js, i-is);
          }
        }
      }
      //vz
      readvtk(this, vtkfile, "velocity", 2, data);
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            phydro->w(IVZ, k, j, i) = data(k-ks, j-js, i-is);
          }
        }
      }
    } else {
      msg << "### FATAL ERROR in Problem Generator [ProblemGenerator]" << std::endl
        << "Scaler field not recognized: " << scaler_fields[i] << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
  }

  //change primative variables to conservative variables.
  phydro->peos->PrimitiveToConserved(phydro->w, b, phydro->u, pcoord,
                                     is, ie, js, je, ks, ke);

  data.DeleteAthenaArray();
  b.DeleteAthenaArray();
  return;
}

//======================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief User-defined work function for every time step
//======================================================================================
void MeshBlock::UserWorkInLoop(void)
{
  return;
}

//======================================================================================
//! \fn static std::vector<std::string> split(std::string str, char delimiter)
//  \brief split a string, and store sub strings in a vector
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

//======================================================================================
//! \fn static void trim(std::string &s)
//  \brief get rid of white spaces leading and trailing a string
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

//TODO: put comments here and declaration at top.
static void readvtk(MeshBlock *mb, std::string filename, std::string field,
                    int component, AthenaArray<float> &data) {
  std::stringstream msg;
  FILE *fp = NULL;
  char cline[256], type[256], variable[256], format[256], t_type[256], t_format[256];
  std::string line;
  const std::string athena_header = "# vtk DataFile Version 2.0"; //athena4.2 header
  bool SHOW_OUTPUT = true;
  int Nx_vtk, Ny_vtk, Nz_vtk; //dimensions of vtk files
  //dimensions of meshblock
  const int Nx_mb = mb->ie - mb->is + 1;
  const int Ny_mb = mb->je - mb->js + 1;
  const int Nz_mb = mb->ke - mb->ks + 1;
  double ox_vtk, oy_vtk, oz_vtk; //origins of vtk file
  double dx_vtk, dy_vtk, dz_vtk; //spacings of vtk file
  int cell_dat_vtk; //total number of cells in vtk file
  //total number of cells in MeshBlock
  const int cell_dat_mb = Nx_mb * Ny_mb * Nz_mb; 
  int retval, nread; //file handler return value
  float fdat, fvec[3], ften[9];//store float format scaler, vector, and tensor

  if ( (fp = fopen(filename.c_str(),"r")) == NULL ) {
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Unable to open file" << filename << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  //get header
  fgets(cline,256,fp);
  line.assign(cline);
  trim(line);
  if (SHOW_OUTPUT) {
    std::cout << line << std::endl;
  }
  if (line != athena_header) {
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
  if (Nx_vtk != Nx_mb || Ny_vtk != Ny_mb || Nz_vtk != Nz_mb) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Dimensions of VTK file" << filename 
      << " is (" << Nx_vtk << ", " << Ny_vtk << ", " << Nz_vtk 
      << "), does not match the dimensions of meshblock ("
      << Nx_mb << ", " << Ny_mb << ", " << Nz_mb << ")." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

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
  if (cell_dat_vtk != cell_dat_mb) {
    fclose(fp);
    msg << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Cell data in vtk file " << filename 
      << " is " << cell_dat_vtk << ", does not match the cell data of meshblock "
      << cell_dat_mb << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  //-------------read data--------------
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
    //read scalers
    if (strcmp(type, "SCALARS") == 0) {
      if (strcmp(variable, field.c_str()) == 0) {      
        printf("  Reading %s...\n", variable);
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
              data(k, j, i) = fdat;
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
        printf("  Reading %s%d...\n", variable, component);
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
              data(k, j, i) = fvec[component];
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

//  \brief Swap bytes, code stolen from Athena4.2, NEMO
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
