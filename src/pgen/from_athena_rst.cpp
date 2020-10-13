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
// \brief Problem Generator to initialize mesh by reading in athena rst files

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  int MAXLEN = 256;
  FILE *fp;
  char line[MAXLEN];
    
  std::stringstream msg; //error message
  std::string rstfile; //corresponding rst file for this meshblock
  std::string rstdir = pin->GetString("problem", "rst_directory");
  std::string rstfile0 = pin->GetString("problem", "rst_file");

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
    rst_name = rst_name0.substr(0, pos3) + "-" + id_str + rst_name0.substr(pos3);}
  rstfile = rstdir + rst_name;
  std::cout<<rstdir + rst_name<<std::endl;
    
  //Open the restart file    
  if((fp = fopen(rstfile.c_str(),"r")) == NULL)
    std::cout << "### FATAL ERROR in Problem Generator [read_vtk]" << std::endl
      << "Error opening the restart file" << std::endl;
  
  //Skip over the parameter file at the start of the restart file
  do{
    fgets(line,MAXLEN,fp);
  }while(strncmp(line,"TIME_STEP",9) != 0);

  //Read the density
  fgets(line,MAXLEN,fp);
  fgets(line,MAXLEN,fp);
  if(strncmp(line,"DENSITY",7) != 0) std::cout << "Expected DENSITY, found" << line << std::endl;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        fread(&(phydro->u(IDN,k,j,i)),sizeof(Real),1,fp);
      }
    }
  }

  //Read the x2-momentum 
  fgets(line,MAXLEN,fp); 
  fgets(line,MAXLEN,fp);
  if(strncmp(line,"1-MOMENTUM",10) != 0) std::cout << "Expected 1-MOMENTUM, found" << line << std::endl;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        fread(&(phydro->u(IM1,k,j,i)),sizeof(Real),1,fp);
      }
    }
  }
  
  //Read the x2-momentum 
  fgets(line,MAXLEN,fp); 
  fgets(line,MAXLEN,fp);
  if(strncmp(line,"2-MOMENTUM",10) != 0) std::cout << "Expected 2-MOMENTUM, found" << line << std::endl;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        fread(&(phydro->u(IM2,k,j,i)),sizeof(Real),1,fp);
      }
    }
  }

  //Read the x3-momentum 
  fgets(line,MAXLEN,fp); 
  fgets(line,MAXLEN,fp);
  if(strncmp(line,"3-MOMENTUM",10) != 0) std::cout << "Expected 3-MOMENTUM, found" << line << std::endl;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        fread(&(phydro->u(IM3,k,j,i)),sizeof(Real),1,fp);
      }
    }
  }
  
  //Read the energy density 
  fgets(line,MAXLEN,fp);
  fgets(line,MAXLEN,fp);
  if(strncmp(line,"ENERGY",6) != 0) std::cout << "Expected ENERGY, found" << line << std::endl;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        fread(&(phydro->u(IEN,k,j,i)),sizeof(Real),1,fp);
      }
    }
  }
  
  
  if (MAGNETIC_FIELDS_ENABLED) {
    
    //Skip Potential
    do{
      fgets(line,MAXLEN,fp);
    }while(strncmp(line,"1-FIELD",7) != 0);
    
    //Read the face-centered x1 B-field 
    if(strncmp(line,"1-FIELD",7) != 0) std::cout << "Expected 1-FIELD, found" << line << std::endl;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          fread(&(pfield->b.x1f(k,j,i)),sizeof(Real),1,fp);
        }
      }
    }
    
    //Read the face-centered x2 B-field 
    fgets(line,MAXLEN,fp); 
    fgets(line,MAXLEN,fp);
    if(strncmp(line,"2-FIELD",7) != 0) std::cout << "Expected 2-FIELD, found" << line << std::endl;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {
          fread(&(pfield->b.x2f(k,j,i)),sizeof(Real),1,fp);
        }
      }
    }
    
    //Read the face-centered x3 B-field 
    fgets(line,MAXLEN,fp); 
    fgets(line,MAXLEN,fp);
    if(strncmp(line,"3-FIELD",7) != 0) std::cout << "Expected 3-FIELD, found" << line << std::endl;
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          fread(&(pfield->b.x3f(k,j,i)),sizeof(Real),1,fp);
        }
      }
    }
    
    pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);
  }
  
  fclose(fp);

  return;
}
