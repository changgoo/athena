//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file task_list.cpp
//! \brief functions for TaskList base class


// C headers

// C++ headers
#include <string> // string
#include <fstream>
//#include <vector> // formerly needed for vector of MeshBlock ptrs in DoTaskListOneStage

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "task_list.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus TaskList::DoAllAvailableTasks
//! \brief do all tasks that can be done (are not waiting for a dependency to be
//! cleared) in this TaskList, return status.

TaskListStatus TaskList::DoAllAvailableTasks(MeshBlock *pmb, int stage, TaskStates &ts) {
  int skip = 0;
  TaskStatus ret;
  if (ts.num_tasks_left == 0) return TaskListStatus::nothing_to_do;

  for (int i=ts.indx_first_task; i<ntasks; i++) {
    Task &taski = task_list_[i];
    if (ts.finished_tasks.IsUnfinished(taski.task_id)) { // task not done
      // check if dependency clear
      if (ts.finished_tasks.CheckDependencies(taski.dependency)) {
        if (taski.lb_time) pmb->StartTimeMeasurement();
        ret = (this->*task_list_[i].TaskFunc)(pmb, stage);
        if (taski.lb_time) {
          pmb->StopTimeMeasurement();
          taski.task_time += pmb->lb_time_;
        }
        if (ret != TaskStatus::fail) { // success
          ts.num_tasks_left--;
          ts.finished_tasks.SetFinished(taski.task_id);
          if (skip == 0) ts.indx_first_task++;
          if (ts.num_tasks_left == 0) return TaskListStatus::complete;
          if (ret == TaskStatus::next) continue;
          return TaskListStatus::running;
        }
      }
      skip++; // increment number of tasks processed

    } else if (skip == 0) { // this task is already done AND it is at the top of the list
      ts.indx_first_task++;
    }
  }
  // there are still tasks to do but nothing can be done now
  return TaskListStatus::stuck;
}

//----------------------------------------------------------------------------------------
//! \fn void TaskList::DoTaskListOneStage(Mesh *pmesh, int stage)
//! \brief completes all tasks in this list, will not return until all are tasks done

void TaskList::DoTaskListOneStage(Mesh *pmesh, int stage) {
  int nthreads = pmesh->GetNumMeshThreads();
  int nmb = pmesh->nblocal;

  // clear the task states, startup the integrator and initialize mpi calls
#pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
  for (int i=0; i<nmb; ++i) {
    pmesh->my_blocks(i)->tasks.Reset(ntasks);
    StartupTaskList(pmesh->my_blocks(i), stage);
  }

  int nmb_left = nmb;
  // cycle through all MeshBlocks and perform all tasks possible
  while (nmb_left > 0) {
    //! \note
    //! KNOWN ISSUE: Workaround for unknown OpenMP race condition. See #183 on GitHub.
#pragma omp parallel for reduction(- : nmb_left) num_threads(nthreads) schedule(dynamic,1)
    for (int i=0; i<nmb; ++i) {
      if (DoAllAvailableTasks(pmesh->my_blocks(i), stage, pmesh->my_blocks(i)->tasks)
          == TaskListStatus::complete) {
        nmb_left--;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TaskList::OutputAllTaskTime()
//! \brief print all task_time

void TaskList::OutputAllTaskTime(const int ncycle, std::string basename) {
  double time_per_step = 0.;
  double all_task_time[128];
  int ntask_time = 0;

  for (int i=0; i<ntasks; i++) {
    Task &taski = task_list_[i];
    if (taski.lb_time) {
      time_per_step += taski.task_time;
      all_task_time[ntask_time] = taski.task_time;
      taski.task_time = 0.; // reset task_time
      ntask_time++;
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, all_task_time, ntask_time, MPI_DOUBLE, MPI_SUM,
    MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &time_per_step, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (Globals::my_rank == 0) {
    std::ofstream os;
    std::string fname;
    fname.assign(basename);
    fname.append(".task_time.txt");
    // open 'loop_time.txt' file
    if (newfile_) {
      os.open(fname.c_str(), std::ofstream::out);
      newfile_ = false;
    } else {
      os.open(fname.c_str(), std::ofstream::app);
    }

    if (!os.is_open()) {
      std::cout << "### ERROR in function OutputAllTaskTime" << std::endl
                << "Cannot open " << fname << std::endl;
      return;
    }

    int j = 0;
    os << "# ncycle=" << ncycle
       << ", TaskList=" << task_list_name
       << ", time=" << time_per_step << std::endl;
    for (int i=0; i<ntasks; i++) {
      Task &taski = task_list_[i];
      if (taski.lb_time) {
        os << "  " << taski.task_name
           << ", time=" << all_task_time[j]
           << ", fraction=" << all_task_time[j]/time_per_step << std::endl;
        j++;
      }
    }
    os.close();
  }
}
