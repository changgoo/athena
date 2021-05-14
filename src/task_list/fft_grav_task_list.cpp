//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fft_grav_task_list.cpp
//! \brief function implementation for FFTGravitySolverTaskList

// C headers

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "fft_grav_task_list.hpp"
#include "task_list.hpp"

//----------------------------------------------------------------------------------------
//! FFTGravitySolverTaskList constructor

FFTGravitySolverTaskList::FFTGravitySolverTaskList(ParameterInput *pin, Mesh *pm) {
  integrator = pin->GetOrAddString("time", "integrator", "vl2");

  // Read a flag for orbital advection
  ORBITAL_ADVECTION = (pm->orbital_advection != 0)? true : false;

  // Read a flag for shear periodic
  SHEAR_PERIODIC = pm->shear_periodic;

  if (integrator == "vl2") {
    //! \note `integrator == "vl2"`
    //! - VL: second-order van Leer integrator (Stone & Gardiner, NewA 14, 139 2009)
    //! - Simple predictor-corrector scheme similar to MUSCL-Hancock
    //! - Expressed in 2S or 3S* algorithm form

    // set number of stages and time coeff.
    if (ORBITAL_ADVECTION) {
      std::stringstream msg;
      msg << "### FATAL ERROR in FFTGravitySolverTaskList constructor" << std::endl
          << "FFT gravity is not tested with orbital advection" << std::endl;
      ATHENA_ERROR(msg);
    } else { // w/o orbital advection
      nstages = 2;
      // To be fully consistent with the TimeIntegratorTaskList, sbeta and ebeta
      // must be set identical to the corresponding values in
      // TimeIntegratorTaskList::stage_wghts.
      sbeta[0] = 0.0;
      ebeta[0] = 0.5;
      sbeta[1] = 0.5;
      ebeta[1] = 1.0;
    }
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in FFTGravitySolverTaskList constructor" << std::endl
        << "integrator=" << integrator << " not tested with FFT gravity" << std::endl;
    ATHENA_ERROR(msg);
  }

  // Now assemble list of tasks for each stage of time integrator
  {using namespace FFTGravitySolverTaskNames; // NOLINT (build/namespace)
    AddTask(SEND_GRAV_BND,NONE);
    AddTask(RECV_GRAV_BND,NONE);
    AddTask(SETB_GRAV_BND,(RECV_GRAV_BND|SEND_GRAV_BND));
    if (SHEAR_PERIODIC) { // Shearingbox BC for Gravity
      AddTask(SEND_GRAV_SH,SETB_GRAV_BND);
      AddTask(RECV_GRAV_SH,SETB_GRAV_BND);
      AddTask(GRAV_PHYS_BND,(SEND_GRAV_SH|RECV_GRAV_SH));
    } else {
      AddTask(GRAV_PHYS_BND,SETB_GRAV_BND);
    }
    AddTask(CLEAR_GRAV, GRAV_PHYS_BND);
  } // end of using namespace block
}

//----------------------------------------------------------------------------------------
//! \fn void FFTGravitySolverTaskList::AddTask(const TaskID& id, const TaskID& dep)
//! \brief Sets id and dependency for "ntask" member of task_list_ array, then iterates
//! value of ntask.

void FFTGravitySolverTaskList::AddTask(const TaskID& id, const TaskID& dep) {
  task_list_[ntasks].task_id=id;
  task_list_[ntasks].dependency=dep;
  task_list_[ntasks].task_name.assign("");

  using namespace FFTGravitySolverTaskNames; // NOLINT (build/namespace)
  if (id == CLEAR_GRAV) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FFTGravitySolverTaskList::ClearFFTGravityBoundary);
    task_list_[ntasks].lb_time = false;
    task_list_[ntasks].task_name.append("ClearFFTGravityBoundary");
  } else if (id == SEND_GRAV_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FFTGravitySolverTaskList::SendFFTGravityBoundary);
    task_list_[ntasks].lb_time = true;
    task_list_[ntasks].task_name.append("SendFFTGravityBoundary");
  } else if (id == RECV_GRAV_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FFTGravitySolverTaskList::ReceiveFFTGravityBoundary);
    task_list_[ntasks].lb_time = false;
    task_list_[ntasks].task_name.append("ReceiveFFTGravityBoundary");
  } else if (id == SETB_GRAV_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FFTGravitySolverTaskList::SetFFTGravityBoundary);
  } else if (id == SEND_GRAV_SH) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FFTGravitySolverTaskList::SendFFTGravityShear);
  } else if (id == RECV_GRAV_SH) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FFTGravitySolverTaskList::ReceiveFFTGravityShear);
  } else if (id == GRAV_PHYS_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&FFTGravitySolverTaskList::PhysicalBoundary);
    task_list_[ntasks].lb_time = true;
    task_list_[ntasks].task_name.append("PhysicalBoundary");
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in FFTGravitySolverTaskList::AddTask" << std::endl
        << "Invalid Task is specified" << std::endl;
    ATHENA_ERROR(msg);
  }
  ntasks++;
  return;
}

void FFTGravitySolverTaskList::StartupTaskList(MeshBlock *pmb, int stage) {
  // Mimics TimeIntegratorTaskList::StartupTaskList
  if (SHEAR_PERIODIC) {
    Real dt_fc   = pmb->pmy_mesh->dt*sbeta[stage-1];
    Real dt_int  = pmb->pmy_mesh->dt*ebeta[stage-1];
    Real time = pmb->pmy_mesh->time;
    if (pmb->pmy_mesh->ncycle == 0) {
      pmb->pbval->ComputeShear(time, time);
    } else {
      pmb->pbval->ComputeShear(time+dt_fc, time+dt_int);
    }
  }
  pmb->pbval->StartReceivingSubset(BoundaryCommSubset::poisson,
                                   pmb->pbval->bvars_fft_grav);
  return;
}

TaskStatus FFTGravitySolverTaskList::ClearFFTGravityBoundary(MeshBlock *pmb, int stage) {
  // Mimics TimeIntegratorTaskList::ClearAllBoundary
  pmb->pbval->ClearBoundarySubset(BoundaryCommSubset::poisson,
                                  pmb->pbval->bvars_fft_grav);
  return TaskStatus::success;
}

TaskStatus FFTGravitySolverTaskList::SendFFTGravityBoundary(MeshBlock *pmb, int stage) {
  pmb->pgrav->gbvar.SendBoundaryBuffers();
  return TaskStatus::success;
}

TaskStatus FFTGravitySolverTaskList::ReceiveFFTGravityBoundary(MeshBlock *pmb,
                                                               int stage) {
  bool ret = pmb->pgrav->gbvar.ReceiveBoundaryBuffers();
  if (!ret)
    return TaskStatus::fail;
  return TaskStatus::success;
}

TaskStatus FFTGravitySolverTaskList::SendFFTGravityShear(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pgrav->gbvar.SendShearingBoxBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}

TaskStatus FFTGravitySolverTaskList::ReceiveFFTGravityShear(MeshBlock *pmb, int stage) {
  bool ret;
  ret = false;
  if (stage <= nstages) {
    ret = pmb->pgrav->gbvar.ReceiveShearingBoxBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  if (ret) {
    pmb->pgrav->gbvar.SetShearingBoxBoundaryBuffers();
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}

TaskStatus FFTGravitySolverTaskList::SetFFTGravityBoundary(MeshBlock *pmb, int stage) {
  pmb->pgrav->gbvar.SetBoundaries();
  return TaskStatus::success;
}

TaskStatus FFTGravitySolverTaskList::PhysicalBoundary(MeshBlock *pmb, int stage) {
  return TaskStatus::next;
}
