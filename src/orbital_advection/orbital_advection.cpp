//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//  \brief define OrbitalAdvection class

// C/C++ headers
#include <algorithm>  // max(), min()
#include <cfloat>     // FLT_MAX, FLT_MIN
#include <iostream>   // cout, endl
#include <sstream>    //
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"

#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../scalars/scalars.hpp"

// this class header
#include "orbital_advection.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif


// constructor of OrbitalAdvection class
OrbitalAdvection::OrbitalAdvection(MeshBlock *pmb, ParameterInput *pin)
    : pmb_(pmb), pm_(pmb->pmy_mesh), ph_(pmb->phydro),
      pf_(pmb->pfield), pco_(pmb->pcoord), pbval_(pmb->pbval), ps_(pmb->pscalars) {
  // read parameters from input file
  orbital_advection_defined = pm_->orbital_advection;

  // check xorder for reconstruction
  xorder = pmb_->precon->xorder;
  xgh = (xorder<=2)? 1 : 2;

  // Read parameters
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    Omega0  = pin->GetOrAddReal("problem","Omega0",0.0);
    qshear  = pin->GetOrAddReal("problem","qshear",0.0);
    onx = pmb_->block_size.nx2;
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    gm  = pin->GetOrAddReal("problem","GM",0.0);
    Omega0  = pin->GetOrAddReal("problem","Omega0",0.0);
    onx = pmb_->block_size.nx2;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    gm  = pin->GetOrAddReal("problem","GM",0.0);
    Omega0  = pin->GetOrAddReal("problem","Omega0",0.0);
    onx = pmb_->block_size.nx3;
  }

  // allocate pre-defined orbital velocity functions
  if (pmb->pmy_mesh->OrbitalVelocity_ == nullptr) {
    // not user-defined functions
    if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
      OrbitalVelocity = CartOrbitalVelocity;
      OrbitalVelocityDerivative[0] = CartOrbitalVelocity_x;
      OrbitalVelocityDerivative[1] = ZeroOrbitalVelocity;
    } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
      if (pmb->block_size.nx3>1) { // 3D
        OrbitalVelocity = CylOrbitalVelocity3D;
        OrbitalVelocityDerivative[0] = CylOrbitalVelocity3D_r;
        OrbitalVelocityDerivative[1] = CylOrbitalVelocity3D_z;
      } else { // 2D
        OrbitalVelocity = CylOrbitalVelocity2D;
        OrbitalVelocityDerivative[0] = CylOrbitalVelocity2D_r;
        OrbitalVelocityDerivative[1] = ZeroOrbitalVelocity;
      }
    } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
      OrbitalVelocity = SphOrbitalVelocity;
      OrbitalVelocityDerivative[0] = SphOrbitalVelocity_r;
      OrbitalVelocityDerivative[1] = SphOrbitalVelocity_t;
    }
  } else { // user-defined orbital velocity functions
    OrbitalVelocity = pmb->pmy_mesh->OrbitalVelocity_;
    // if derivatives are not given, caluclate them automatically
    if (pmb->pmy_mesh->OrbitalVelocityDerivative_[0] != nullptr)
      OrbitalVelocityDerivative[0] = pmb->pmy_mesh->OrbitalVelocityDerivative_[0];
    if (pmb->pmy_mesh->OrbitalVelocityDerivative_[1] != nullptr)
      OrbitalVelocityDerivative[1] = pmb->pmy_mesh->OrbitalVelocityDerivative_[1];
  }

  // set meshblock size & orbital_direction
  // For orbital_direction ==1, x2 is the orbital direction (2D/3D)
  // For orbital_direction ==2, x3 is the orbital direction (3D)
  orbital_direction = 0;
  if(orbital_advection_defined) {
    nc1 = pmb_->block_size.nx1 + 2*NGHOST;
    nc2 = 1, nc3 = 1;
    if ((std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0)
         || (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0)) {
      if (pmb_->block_size.nx2 > 1) { // 2D or 3D
        nc2 = pmb_->block_size.nx2 + 2*(NGHOST);
        orbital_direction = 1;
      } else { // 1D
        orbital_advection_defined = false;
        std::stringstream msg;
        msg << "### FATAL ERROR in OrbitalAdvection Class."<<std::endl
            << "Orbital advection needs 2D or 3D in cartesian and cylindrical."<<std::endl
            << "Check <problem> orbital_advection parameter in the input file"<<std::endl;
        ATHENA_ERROR(msg);
      }
      if (pmb_->block_size.nx3 > 1) // 3D
        nc3 = pmb_->block_size.nx3 + 2*(NGHOST);
    } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
      if (pmb_->block_size.nx3 > 1) { // 3D
        nc3 = pmb_->block_size.nx3 + 2*(NGHOST);
        orbital_direction = 2;
      } else { // 1D or 2D
        orbital_advection_defined = false;
        std::stringstream msg;
        msg << "### FATAL ERROR in OrbitalAdvection Class."<<std::endl
            << "Orbital advection needs 3D in spherical_polar coordinates."<<std::endl
            << "Check <problem> orbital_advection parameter in the input file"<<std::endl;
        ATHENA_ERROR(msg);
      }
      nc2 = pmb_->block_size.nx2 + 2*(NGHOST);
    } else {
      orbital_advection_defined = false;
      std::stringstream msg;
      msg << "### FATAL ERROR in OrbitalAdvection Class."<<std::endl
          << "Orbital advection works only in cartesian, cylindrical, "
          << "or spherical_polar coordinates."<<std::endl
          << "Check <problem> orbital_advection parameter in the input file"<<std::endl;
      ATHENA_ERROR(msg);
    }

    // check parameters about the orbital motion for using pre-defined orbital velocity
    if (pmb->pmy_mesh->OrbitalVelocity_ == nullptr) {
      if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
        if ((Omega0 == 0.0) || (qshear == 0.0)) {
          orbital_advection_defined = false;
          std::stringstream msg;
          msg << "### FATAL ERROR in OrbitalAdvection Class."<<std::endl
              << "Pre-defined orbital velocity requires that "
              << "both Omega0 and qshear are non-zero."<<std::endl
              << "Check <problem> Omega0 and qshear in the input file"<<std::endl;
          ATHENA_ERROR(msg);
        }
      } else if ((std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0)
                || (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)) {
        if (gm == 0.0) {
          orbital_advection_defined = false;
          std::stringstream msg;
          msg << "### FATAL ERROR in OrbitalAdvection Class."<<std::endl
              << "Pre-defined orbital velocity requires that non-zero GM."<<std::endl
              << "Check <problem> GM and <problem> qshear in the input file"<<std::endl;
          ATHENA_ERROR(msg);
        }
      }
    }

    // check orbital_refinement
    orbital_refinement = false;
    if (pm_->adaptive==true) { //AMR
      orbital_refinement = true;
    } else if (pm_->multilevel==true) { //SMR
      if (orbital_direction == 1) { // cartesian or cylindrical
        int64_t nbx = pm_->nrbx2 * (1L << (pmb_->loc.level - pm_->root_level));
        LogicalLocation loc;
        loc.level = pmb_->loc.level;
        loc.lx1   = pmb_->loc.lx1;
        loc.lx3   = pmb_->loc.lx3;
        for (int64_t dlx=1; dlx < nbx; dlx++) {
          loc.lx2 = pmb_->loc.lx2+dlx;
          if (loc.lx2>=nbx) loc.lx2-=nbx;
          //check level of meshblocks at same i, k
          MeshBlockTree *mbt = pm_->tree.FindMeshBlock(loc);
          if(mbt == nullptr || mbt->GetGid() == -1) {
            orbital_refinement = true;
            break;
          }
        }
      } else if (orbital_direction == 2) { // spherical_polar
        int64_t nbx = pm_->nrbx3 * (1L << (pmb_->loc.level - pm_->root_level));
        LogicalLocation loc;
        loc.level = pmb_->loc.level;
        loc.lx1   = pmb_->loc.lx1;
        loc.lx2   = pmb_->loc.lx2;
        for (int64_t dlx=0; dlx < nbx; dlx++) {
          loc.lx3   = pmb_->loc.lx3+dlx;
          if (loc.lx3>=nbx) loc.lx3-=nbx;
          //check level of meshblocks at same i, j
          MeshBlockTree *mbt = pm_->tree.FindMeshBlock(loc);
          if(mbt == nullptr || mbt->GetGid() == -1) {
            orbital_refinement = true;
            break;
          }
        }
      }
    }

    // check boundary conditions in the orbital direction
    if (orbital_direction == 1) { // cartesian or cylindrical
      if(pin->GetOrAddString("mesh", "ix2_bc", "none")!="periodic"
        ||pin->GetOrAddString("mesh", "ox2_bc", "none")!="periodic") {
        orbital_advection_defined = false;
        std::stringstream msg;
        msg << "### FATAL ERROR in OrbitalAdvection Class."<<std::endl
            << "Orbital advection requires boundary conditions of x2 periodic"
            << "in cartesian or cylindrical coordinates."<<std::endl
            << "Check <mesh> ix2_bc and ox2_bc in the input file"<<std::endl;
        ATHENA_ERROR(msg);
      }
    } else if (orbital_direction==2) { // spherical_polar
      if(pin->GetOrAddString("mesh", "ix3_bc", "none")!="periodic"
          ||pin->GetOrAddString("mesh", "ox3_bc", "none")!="periodic") {
        orbital_advection_defined = false;
        std::stringstream msg;
        msg << "### FATAL ERROR in OrbitalAdvection Class."<<std::endl
            << "Orbital advection requires boundary conditions of x3 periodic"
            << "in spherical_polar coordinates."<<std::endl
            << "Check <mesh> ix3_bc and ox3_bc in the input file"<<std::endl;
        ATHENA_ERROR(msg);
      }
    }

    // initialize orbital_system_output_done flag
    orbital_system_output_done = false;

    // check orbital_uniform_mesh
    // TODO(tomo-ono): non-uniform mesh grids are not allowed now.
    orbital_uniform_mesh = pm_->use_uniform_meshgen_fn_[orbital_direction];
    if (!orbital_uniform_mesh) {
      std::stringstream msg;
      msg << "### FATAL ERROR in OrbitalAdvection Class."<<std::endl
          << "Orbital advection currently does not support non-uniform mesh"
          << "in the orbital direction." << std::endl;
      ATHENA_ERROR(msg);
    }

    // memory allocation
    if (orbital_direction == 1) { // cartesian or cylindrical
      vKc.NewAthenaArray(nc3, nc1);
      dvKc1.NewAthenaArray(nc3, nc1);
      dvKc2.NewAthenaArray(nc3, nc1);
      vKf[0].NewAthenaArray(nc3,nc1+1);
      vKf[1].NewAthenaArray(nc3+1,nc1);
      orbital_cons.NewAthenaArray(NHYDRO, nc3, nc1, nc2+onx+1);

      orc.NewAthenaArray(nc3, nc1);
      ofc.NewAthenaArray(nc3, nc1);

      if (orbital_refinement) {
        vKc_coarse.NewAthenaArray((nc3+2*NGHOST)/2, (nc1+2*NGHOST)/2);
        ofc_coarse.NewAthenaArray((nc3+2*NGHOST)/2, (nc1+2*NGHOST)/2);
      }

      if (MAGNETIC_FIELDS_ENABLED) {
        orbital_b1.NewAthenaArray(nc3, nc1+1, nc2+onx+1);
        orbital_b2.NewAthenaArray(nc3+1, nc1, nc2+onx+1);

        orf[0].NewAthenaArray(nc3, nc1+1);
        orf[1].NewAthenaArray(nc3+1, nc1);
        off[0].NewAthenaArray(nc3, nc1+1);
        off[1].NewAthenaArray(nc3+1, nc1);

        if (orbital_refinement) {
          vKf_coarse[0].NewAthenaArray((nc3+2*NGHOST)/2, (nc1+2*NGHOST)/2+1);
          off_coarse[0].NewAthenaArray((nc3+2*NGHOST)/2, (nc1+2*NGHOST)/2+1);
          vKf_coarse[1].NewAthenaArray((nc3+2*NGHOST)/2+1, (nc1+2*NGHOST)/2);
          off_coarse[1].NewAthenaArray((nc3+2*NGHOST)/2+1, (nc1+2*NGHOST)/2);
        }
      }

      if (NSCALARS>0)
        orbital_scalar.NewAthenaArray(NSCALARS, nc3, nc1, nc2+onx+1);
    } else if (orbital_direction==2) { // spherical_polar
      vKc.NewAthenaArray(nc2, nc1);
      dvKc1.NewAthenaArray(nc2, nc1);
      dvKc2.NewAthenaArray(nc2, nc1);
      vKf[0].NewAthenaArray(nc2,nc1+1);
      vKf[1].NewAthenaArray(nc2+1,nc1);
      orbital_cons.NewAthenaArray(NHYDRO, nc2, nc1, nc3+onx+1);

      orc.NewAthenaArray(nc2, nc1);
      ofc.NewAthenaArray(nc2, nc1);

      if (orbital_refinement) {
        vKc_coarse.NewAthenaArray((nc2+2*NGHOST)/2, (nc1+2*NGHOST)/2);
        ofc_coarse.NewAthenaArray((nc2+2*NGHOST)/2, (nc1+2*NGHOST)/2);
      }

      if (MAGNETIC_FIELDS_ENABLED) {
        orbital_b1.NewAthenaArray(nc2, nc1+1, nc3+onx+1);
        orbital_b2.NewAthenaArray(nc2+1, nc1, nc3+onx+1);

        orf[0].NewAthenaArray(nc2, nc1+1);
        orf[1].NewAthenaArray(nc2+1, nc1);
        off[0].NewAthenaArray(nc2, nc1+1);
        off[1].NewAthenaArray(nc2+1, nc1);

        if (orbital_refinement) {
          vKf_coarse[0].NewAthenaArray((nc2+2*NGHOST)/2, (nc1+2*NGHOST)/2+1);
          off_coarse[0].NewAthenaArray((nc2+2*NGHOST)/2, (nc1+2*NGHOST)/2+1);
          vKf_coarse[1].NewAthenaArray((nc2+2*NGHOST)/2+1, (nc1+2*NGHOST)/2);
          off_coarse[1].NewAthenaArray((nc2+2*NGHOST)/2+1, (nc1+2*NGHOST)/2);
        }
      }

      if (NSCALARS>0)
        orbital_scalar.NewAthenaArray(NSCALARS, nc2, nc1, nc3+onx+1);
    }
    pflux.NewAthenaArray(onx+2*NGHOST+1);
    w_orb.NewAthenaArray(NHYDRO,nc3,nc2,nc1);
    u_orb.NewAthenaArray(NHYDRO,nc3,nc2,nc1);

    if (orbital_refinement) {
      u_coarse_send.NewAthenaArray(NHYDRO, (nc3+2*NGHOST)/2,
                                   (nc2+2*NGHOST)/2, (nc1+2*NGHOST)/2);
      u_coarse_recv.NewAthenaArray(NHYDRO, (nc3+2*NGHOST)/2,
                                   (nc2+2*NGHOST)/2, (nc1+2*NGHOST)/2);
      u_temp.NewAthenaArray(NHYDRO, nc3, nc2, nc1);
      if (NSCALARS>0) {
        s_coarse_send.NewAthenaArray(NSCALARS, (nc3+2*NGHOST)/2,
                                     (nc2+2*NGHOST)/2, (nc1+2*NGHOST)/2);
        s_coarse_recv.NewAthenaArray(NSCALARS, (nc3+2*NGHOST)/2,
                                     (nc2+2*NGHOST)/2, (nc1+2*NGHOST)/2);
        s_temp.NewAthenaArray(NSCALARS, nc3, nc2, nc1);
      }
      if (MAGNETIC_FIELDS_ENABLED) {
        b1_coarse_send.NewAthenaArray((nc3+2*NGHOST)/2,
                                      (nc2+2*NGHOST)/2, (nc1+2*NGHOST)/2+1);
        b_coarse_recv.x1f.NewAthenaArray((nc3+2*NGHOST)/2,
                                        (nc2+2*NGHOST)/2, (nc1+2*NGHOST)/2+1);
        b_coarse_recv.x2f.NewAthenaArray((nc3+2*NGHOST)/2,
                                        (nc2+2*NGHOST)/2+1, (nc1+2*NGHOST)/2);
        b_coarse_recv.x3f.NewAthenaArray((nc3+2*NGHOST)/2+1,
                                        (nc2+2*NGHOST)/2, (nc1+2*NGHOST)/2);
        b_temp.x1f.NewAthenaArray(nc3, nc2, nc1+1);
        b_temp.x2f.NewAthenaArray(nc3, nc2+1, nc1);
        b_temp.x3f.NewAthenaArray(nc3+1, nc2, nc1);
        if (orbital_direction == 1) {
          b2_coarse_send.NewAthenaArray((nc3+2*NGHOST)/2+1,
                                        (nc2+2*NGHOST)/2, (nc1+2*NGHOST)/2);
        } else if (orbital_direction == 2) {
          b2_coarse_send.NewAthenaArray((nc3+2*NGHOST)/2,
                                        (nc2+2*NGHOST)/2+1, (nc1+2*NGHOST)/2);
        }
      }
    }
    //if(!orbital_uniform_mesh) {
    //}

    // call OrbitalAdvectionBoundaryVariable
    orb_bc = new OrbitalBoundaryCommunication(this);
  }

  // preparation for shear_periodic boundary
  if ((orbital_advection_defined || pm_->shear_periodic)) {
    int pnum = onx+2*NGHOST+1;
    if (pm_->shear_periodic && MAGNETIC_FIELDS_ENABLED) pnum++;
    if (xorder>2) {
      for (int n=0; n<13; n++) {
        d_src[n].NewAthenaArray(pnum);
      }
    }
  }
}

// destructor
OrbitalAdvection::~OrbitalAdvection() {
  if (orbital_advection_defined) {
    // destroy OrbitalBoundaryCommunication
    delete orb_bc;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvection::InitializeOrbitalAdvection()
//  \brief Setup for OrbitalAdvection in void Mesh::Initialize
void OrbitalAdvection::InitializeOrbitalAdvection() {
  //set grids edge in the orbital direction
  int xs, xe, xl, xu;
  if (orbital_direction == 1) { // cartesian or cylindrical
    xs = pmb_->js; xe = pmb_->je;
  } else if (orbital_direction ==2) { // spherical_polar
    xs = pmb_->ks; xe = pmb_->ke;
  }
  xl = xs - NGHOST; xu = xe + NGHOST;

  // set orbital velocity
  SetVKc();
  SetVKf();
  SetDvKc();
  if (orbital_refinement) {
    SetVKcCoarse();
    if (MAGNETIC_FIELDS_ENABLED) {
      SetVKfCoarse();
    }
  }

  if(orbital_uniform_mesh) { // uniform mesh
    // set dx
    dx = (orbital_direction == 1) ?
         pco_->dx2f(pmb_->js): pco_->dx3f(pmb_->ks);
  }
//  else { // non-uniform mesh
//  }

  //set vK_max, vK_min
  vK_max = -(FLT_MAX);
  vK_min = (FLT_MAX);
  if (orbital_direction == 1) { // cartesian or cylindrical
    // cell center
    for (int k=pmb_->ks; k<= pmb_->ke; k++) {
      for (int i=pmb_->is; i<= pmb_->ie; i++) {
        Real pvk = 1.0/pco_->h2v(i);
        vK_max   = std::max(vK_max, vKc(k,i)*pvk);
        vK_min   = std::min(vK_min, vKc(k,i)*pvk);
      }
    }
    if (MAGNETIC_FIELDS_ENABLED) {
      // x1 surface
      for (int k=pmb_->ks; k<= pmb_->ke; k++) {
        for (int i=pmb_->is; i<= pmb_->ie+1; i++) {
          Real pvk = 1.0/pco_->h2f(i);
          vK_max   = std::max(vK_max, vKf[0](k,i)*pvk);
          vK_min   = std::min(vK_min, vKf[0](k,i)*pvk);
        }
      }
      // x3 surface
      for (int k=pmb_->ks; k<= pmb_->ke+1; k++) {
        for (int i=pmb_->is; i<= pmb_->ie; i++) {
          Real pvk = 1.0/pco_->h2v(i);
          vK_max   = std::max(vK_max, vKf[1](k,i)*pvk);
          vK_min   = std::min(vK_min, vKf[1](k,i)*pvk);
        }
      }
    }
  } else if (orbital_direction == 2) {// spherical_polar
    // cell center
    for (int j=pmb_->js; j<= pmb_->je; ++j) {
      for (int i=pmb_->is; i<= pmb_->ie; ++i) {
        Real pvk = 1.0/(pco_->h2v(i)*pco_->h32v(j));
        vK_max   = std::max(vK_max, vKc(j,i)*pvk);
        vK_min   = std::min(vK_min, vKc(j,i)*pvk);
      }
    }
    if (MAGNETIC_FIELDS_ENABLED) {
      // x1 surface
      for (int j=pmb_->js; j<= pmb_->je; ++j) {
        for (int i=pmb_->is; i<= pmb_->ie+1; ++i) {
          Real pvk = 1.0/(pco_->h2f(i)*pco_->h32v(j));
          vK_max   = std::max(vK_max, vKf[0](j,i)*pvk);
          vK_min   = std::min(vK_min, vKf[0](j,i)*pvk);
        }
      }
      // x2 surface
      for (int j=pmb_->js; j<= pmb_->je+1; ++j) {
        for (int i=pmb_->is; i<= pmb_->ie; ++i) {
          Real pvk = 1.0/(pco_->h2v(i)*pco_->h32f(j));
          vK_max   = std::max(vK_max, vKf[1](j,i)*pvk);
          vK_min   = std::min(vK_min, vKf[1](j,i)*pvk);
        }
      }
    }
  }

  // set min_dt
  int mylevel = pmb_->loc.level;
  int lblevel, rblevel;
  if (orbital_direction == 1) {
    for(int n=0; n<pbval_->nneighbor; n++) {
      NeighborBlock& nb = pbval_->neighbor[n];
      if(nb.ni.ox1==0 && nb.ni.ox3==0) {
        if(nb.ni.ox2==-1) lblevel = nb.snb.level;
        else if(nb.ni.ox2== 1) rblevel = nb.snb.level;
      }
    }
  } else if (orbital_direction == 2) {
    for(int n=0; n<pbval_->nneighbor; n++) {
      NeighborBlock& nb = pbval_->neighbor[n];
      if(nb.ni.ox1==0 && nb.ni.ox2==0) {
        if(nb.ni.ox3==-1) lblevel = nb.snb.level;
        else if(nb.ni.ox3== 1) rblevel = nb.snb.level;
      }
    }
  }
  min_dt = (FLT_MAX);
  // restrictions from meshblock size
  if(orbital_uniform_mesh) { // uniform mesh
    if(vK_max>0.0) {
      if(lblevel > mylevel)
        min_dt = std::min(min_dt, dx*(onx/2-xgh)/vK_max);
      else
        min_dt = std::min(min_dt, dx*(onx-xgh)/vK_max);
    }
    if(vK_min<0.0) {
      if(rblevel > mylevel)
        min_dt = std::min(min_dt, -dx*(onx/2-xgh)/vK_min);
      else
        min_dt = std::min(min_dt, -dx*(onx-xgh)/vK_min);
    }
  }
//  else {
//  }
  // restrictions from derivatives of orbital velocity
  if(orbital_direction == 1) {
    for(int k=pmb_->ks; k<=pmb_->ke; k++) {
      for(int i=pmb_->is; i<=pmb_->ie; i++) {
        Real dvk_ghost = fabs(vKc(k,i+NGHOST)/pco_->h2v(i+NGHOST)
                           - vKc(k,i-NGHOST)/pco_->h2v(i-NGHOST));
        if(nc3>1) {
          Real temp = fabs(vKc(k+NGHOST,i) - vKc(k-NGHOST,i))/pco_->h2v(i);
          dvk_ghost = std::max(dvk_ghost, temp);
        }
        if(dvk_ghost == 0.0) {
          continue;
        } else if(orbital_uniform_mesh) { // uniform mesh
          Real orb_dt = 0.5*dx/dvk_ghost;
          if (min_dt > orb_dt) min_dt = orb_dt;
          // tomo-ono: if using std::min, sometimes an error occurs
          // min_dt = std::min(min_dt, 0.5*dx/dvk_ghost);
        }
//        else { // non-uniform mesh
//        }
      }
    }
  } else if (orbital_direction == 2) {
    for(int j=pmb_->js; j<=pmb_->je; j++) {
      for(int i=pmb_->is; i<=pmb_->ie; i++) {
        Real dvk_ghost = fabs(vKc(j,i+NGHOST)/pco_->h2v(i+NGHOST)
                           - vKc(j,i-NGHOST)/pco_->h2v(i-NGHOST))/pco_->h32v(j);
        Real temp = fabs(vKc(j+NGHOST,i)/pco_->h32v(j+NGHOST)
                     - vKc(j-NGHOST,i)/pco_->h32v(j-NGHOST))/pco_->h2v(i);
        dvk_ghost = std::max(dvk_ghost, temp);
        if(dvk_ghost == 0.0) {
          continue;
        } else if(orbital_uniform_mesh) { // uniform mesh
          Real orb_dt = 0.5*dx/dvk_ghost;
          if (min_dt > orb_dt) min_dt = orb_dt;
          // tomo-ono: if using std::min, sometimes an error occurs
          //min_dt = std::min(min_dt, 0.5*dx/dvk_ghost);
        }
//        else { // non-uniform mesh
//        }
      }
    }
  }
  return;
}

//---------------------------------------------------------------------------------------
//! \fn Real OrbitalAdvection::NewOrbitalAdvectionDt()
// Calculate time step for OrbitalAdvection

Real OrbitalAdvection::NewOrbitalAdvectionDt() {
  return min_dt;
}

//---------------------------------------------------------------------------------------
//! \fn void OrbitalAdvection::SetVKc()
// Calculate Orbital Velocity at cell-centered

void OrbitalAdvection::SetVKc() {
  int il = pmb_->is-(NGHOST); int jl = pmb_->js-(NGHOST); int kl = pmb_->ks;
  int iu = pmb_->ie+(NGHOST); int ju = pmb_->je+(NGHOST); int ku = pmb_->ke;
  if (nc3>1) {
    kl -= NGHOST;
    ku += NGHOST;
  }
  if (orbital_direction == 1) {
    for(int k=kl; k<=ku; k++) {
      Real z_ = pco_->x3v(k);
#pragma omp simd
      for(int i=il; i<=iu; i++) {
        Real x_ = pco_->x1v(i);
        vKc(k,i)   = OrbitalVelocity(this, x_, 0.0, z_);
      }
    }
  } else if (orbital_direction == 2) {
    for(int j=jl; j<=ju; j++) {
      Real y_ = pco_->x2v(j);
#pragma omp simd
      for(int i=il; i<=iu; i++) {
        Real x_ = pco_->x1v(i);
        vKc(j,i)   = OrbitalVelocity(this, x_, y_, 0.0);
      }
    }
  }
}

//---------------------------------------------------------------------------------------
//! \fn void OrbitalAdvection::SetVKf()
// Calculate Orbital Velocity at cell surface
void OrbitalAdvection::SetVKf() {
  int il = pmb_->is-(NGHOST); int jl = pmb_->js-(NGHOST); int kl = pmb_->ks;
  int iu = pmb_->ie+(NGHOST); int ju = pmb_->je+(NGHOST); int ku = pmb_->ke;
  if (nc3>1) {
    kl -= NGHOST;
    ku += NGHOST;
  }
  if (orbital_direction == 1) {
    for(int k=kl; k<=ku; k++) {
      Real z_ = pco_->x3v(k);
#pragma omp simd
      for(int i=il; i<=iu+1; i++) {
        Real x_ = pco_->x1f(i);
        vKf[0](k,i)   = OrbitalVelocity(this, x_, 0.0, z_);
      }
    }
    for(int k=kl; k<=ku+1; k++) {
      Real z_ = pco_->x3f(k);
#pragma omp simd
      for(int i=il; i<=iu; i++) {
        Real x_ = pco_->x1v(i);
        vKf[1](k,i)   = OrbitalVelocity(this, x_, 0.0, z_);
      }
    }
  } else if (orbital_direction == 2) {
    for(int j=jl; j<=ju; j++) {
      Real y_ = pco_->x2v(j);
#pragma omp simd
      for(int i=il; i<=iu+1; i++) {
        Real x_ = pco_->x1f(i);
        vKf[0](j,i)   = OrbitalVelocity(this, x_, y_, 0.0);
      }
    }
    for(int j=jl; j<=ju+1; j++) {
      Real y_ = pco_->x2f(j);
#pragma omp simd
      for(int i=il; i<=iu; i++) {
        Real x_ = pco_->x1v(i);
        vKf[1](j,i)   = OrbitalVelocity(this, x_, y_, 0.0);
      }
    }
  }
}

//---------------------------------------------------------------------------------------
//! \fn void OrbitalAdvection::SetDvKc()
// Calculate Orbital Velocity at cell center

void OrbitalAdvection::SetDvKc() {
  int il = pmb_->is-(NGHOST); int jl = pmb_->js-(NGHOST); int kl = pmb_->ks;
  int iu = pmb_->ie+(NGHOST); int ju = pmb_->je+(NGHOST); int ku = pmb_->ke;
  if (nc3>1) {
    kl -= NGHOST;
    ku += NGHOST;
  }
  if (orbital_direction == 1) { // cartesian or cylindrical
    if (OrbitalVelocityDerivative[0] == nullptr) {
      // calculate dvK using user-defined vKf
      for(int k=kl; k<=ku; k++) {
#pragma omp simd
        for(int i=il; i<=iu; i++) {
          dvKc1(k,i) = (vKf[0](k,i+1)-vKf[0](k,i))/pco_->dx1f(i);
        }
      }
    } else {
      // set dvK from user-defined dvK
      for(int k=kl; k<=ku; k++) {
        Real z_ = pco_->x3v(k);
#pragma omp simd
        for(int i=il; i<=iu; i++) {
          Real x_ = pco_->x1v(i);
          dvKc1(k,i)   = OrbitalVelocityDerivative[0](this, x_, 0.0, z_);
        }
      }
    }
    if (nc3>1) { // 3D
      if (OrbitalVelocityDerivative[1] == nullptr) {
        // calculate dvK using user-defined vKf
        for(int k=kl; k<=ku; k++) {
#pragma omp simd
          for(int i=il; i<=iu; i++) {
            dvKc2(k,i) = (vKf[1](k+1,i)-vKf[1](k,i))/pco_->dx3f(k);
          }
        }
      } else {
        // set dvK from user-defined dvK
        for(int k=kl; k<=ku; k++) {
          Real z_ = pco_->x3v(k);
#pragma omp simd
          for(int i=il; i<=iu; i++) {
            Real x_ = pco_->x1v(i);
            dvKc2(k,i)   = OrbitalVelocityDerivative[1](this, x_, 0.0, z_);
          }
        }
      }
    } else { // 2D
      for(int k=kl; k<=ku; k++) {
#pragma omp simd
        for(int i=il; i<=iu; i++) {
          dvKc2(k,i)   = 0.0;
        }
      }
    }
  } else if (orbital_direction == 2) { // spherical_polar
    if (OrbitalVelocityDerivative[0] == nullptr) {
      // calculate dvK using user-defined vKf
      for(int j=jl; j<=ju; j++) {
#pragma omp simd
        for(int i=il; i<=iu; i++) {
          dvKc1(j,i) = (vKf[0](j,i+1)-vKf[0](j,i))/pco_->dx1f(i);
        }
      }
    } else {
      // set dvK from user-defined dvK
      for(int j=jl; j<=ju; j++) {
        Real y_ = pco_->x2v(j);
#pragma omp simd
        for(int i=il; i<=iu; i++) {
          Real x_ = pco_->x1v(i);
          dvKc1(j,i)   = OrbitalVelocityDerivative[0](this, x_, y_, 0.0);
        }
      }
    }
    if (OrbitalVelocityDerivative[1] == nullptr) {
      // calculate dvK using user-defined vKf
      for(int j=jl; j<=ju; j++) {
#pragma omp simd
        for(int i=il; i<=iu; i++) {
          dvKc2(j,i) = (vKf[1](j+1,i)-vKf[1](j,i))/pco_->dx2f(j);
        }
      }
    } else {
      // set dvK from user-defined dvK
      for(int j=jl; j<=ju; j++) {
        Real y_ = pco_->x2v(j);
#pragma omp simd
        for(int i=il; i<=iu; i++) {
          Real x_ = pco_->x1v(i);
          dvKc2(j,i)   = OrbitalVelocityDerivative[1](this, x_, y_, 0.0);
        }
      }
    }
  }
}

//---------------------------------------------------------------------------------------
//! \fn void OrbitalAdvection::SetVKcCoarse()
// Calculate Orbital Velocity at cell center for coarse cells

void OrbitalAdvection::SetVKcCoarse() {
  int il = pmb_->cis-(NGHOST); int jl = pmb_->cjs-(NGHOST); int kl = pmb_->cks;
  int iu = pmb_->cie+(NGHOST); int ju = pmb_->cje+(NGHOST); int ku = pmb_->cke;
  if (nc3>1) {
    kl -= NGHOST;
    ku += NGHOST;
  }
  Coordinates *cpco = pmb_->pmr->pcoarsec;
  if (orbital_direction == 1) {
    for(int k=kl; k<=ku; k++) {
      Real z_ = cpco->x3v(k);
#pragma omp simd
      for(int i=il; i<=iu; i++) {
        Real x_ = cpco->x1v(i);
        vKc_coarse(k,i)   = OrbitalVelocity(this, x_, 0.0, z_);
      }
    }
  } else if (orbital_direction == 2) {
    for(int j=jl; j<=ju; j++) {
      Real y_ = cpco->x2v(j);
#pragma omp simd
      for(int i=il; i<=iu; i++) {
        Real x_ = cpco->x1v(i);
        vKc_coarse(j,i)   = OrbitalVelocity(this, x_, y_, 0.0);
      }
    }
  }
}

//---------------------------------------------------------------------------------------
//! \fn void OrbitalAdvection::SetVKfCoarse()
// Calculate Orbital Velocity at cell surface for coarse cells
void OrbitalAdvection::SetVKfCoarse() {
  int il = pmb_->cis-(NGHOST); int jl = pmb_->cjs-(NGHOST); int kl = pmb_->cks;
  int iu = pmb_->cie+(NGHOST); int ju = pmb_->cje+(NGHOST); int ku = pmb_->cke;
  if (nc3>1) {
    kl -= NGHOST;
    ku += NGHOST;
  }
  Coordinates *cpco = pmb_->pmr->pcoarsec;
  if (orbital_direction == 1) {
    for(int k=kl; k<=ku; k++) {
      Real z_ = cpco->x3v(k);
#pragma omp simd
      for(int i=il; i<=iu+1; i++) {
        Real x_ = cpco->x1f(i);
        vKf_coarse[0](k,i)   = OrbitalVelocity(this, x_, 0.0, z_);
      }
    }
    for(int k=kl; k<=ku+1; k++) {
      Real z_ = cpco->x3f(k);
#pragma omp simd
      for(int i=il; i<=iu; i++) {
        Real x_ = cpco->x1v(i);
        vKf_coarse[1](k,i)   = OrbitalVelocity(this, x_, 0.0, z_);
      }
    }
  } else if (orbital_direction == 2) {
    for(int j=jl; j<=ju; j++) {
      Real y_ = cpco->x2v(j);
#pragma omp simd
      for(int i=il; i<=iu+1; i++) {
        Real x_ = cpco->x1f(i);
        vKf_coarse[0](j,i)   = OrbitalVelocity(this, x_, y_, 0.0);
      }
    }
    for(int j=jl; j<=ju+1; j++) {
      Real y_ = cpco->x2f(j);
#pragma omp simd
      for(int i=il; i<=iu; i++) {
        Real x_ = cpco->x1v(i);
        vKf_coarse[1](j,i)   = OrbitalVelocity(this, x_, y_, 0.0);
      }
    }
  }
}
