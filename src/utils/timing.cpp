//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file timing.cpp
//! \brief utility function for timing

// C headers

// C++ headers
#include <iostream>
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

static bool newfile_ = true;
//----------------------------------------------------------------------------------------
//! \fn Real MarkTime()
//! \brief mark time using proper scheme

double MarkTime() {
#ifdef OPENMP_PARALLEL
  return omp_get_wtime();
#else
#ifdef MPI_PARLLEL
  return MPI_Wtime();
#else
  return static_cast<double>(clock());
#endif
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void OutputLoopTime(Real dt_array[])
//! \brief output loop time breakdown
void OutputLoopTime(int ncycle, double dt_array[]) {
  #ifdef MPI_PARALLEL
    // pack array, MPI allreduce over array, then unpack into Mesh variables
    MPI_Allreduce(MPI_IN_PLACE, dt_array, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  #endif
  double time_per_step = dt_array[0] + dt_array[1] + dt_array[2]
                     + dt_array[3] + dt_array[4];
  if (Globals::my_rank == 0) {
    FILE *fp = nullptr;
    char fop{ 'a' };
    // open 'loop_time.txt' file
    if (newfile_) {
      fop = 'w';
      newfile_ = false;
    }

    if ((fp = std::fopen("loop_time.txt",&fop)) == nullptr) {
      std::cout << "### ERROR in function OutputLoopTime" << std::endl
                << "Cannot open loop_time.txt" << std::endl;
      return;
    }

    std::fprintf(fp,"ncycle=%d, time=%e,",ncycle, time_per_step);
    std::fprintf(fp,"Before=%e,",dt_array[0]);
    std::fprintf(fp,"TurbulenceDriver=%e,",dt_array[1]);
    std::fprintf(fp,"TimeIntegratorTaskList=%e,",dt_array[2]);
    std::fprintf(fp,"SelfGravity=%e,",dt_array[3]);
    std::fprintf(fp,"After=%e\n",dt_array[4]);
    std::fclose(fp);
  }
}
