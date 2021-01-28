#ifndef TASK_LIST_FFT_GRAV_TASK_LIST_HPP_
#define TASK_LIST_FFT_GRAV_TASK_LIST_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fft_grav_task_list.hpp
//! \brief define FFTGravitySolverTaskList

// C headers

// C++ headers
#include <cstdint>      // std::uint64_t
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "task_list.hpp"

// forward declarations
class Mesh;
class MeshBlock;

//----------------------------------------------------------------------------------------
//! \class FFTGravitySolverTaskList
//! \brief data and function definitions for FFTGravitySolverTaskList derived class

class FFTGravitySolverTaskList : public TaskList {
 public:
  FFTGravitySolverTaskList(ParameterInput *pin, Mesh *pm);

  // data
  std::string integrator;

  // functions
  TaskStatus ClearFFTGravityBoundary(MeshBlock *pmb, int stage);
  TaskStatus SendFFTGravityBoundary(MeshBlock *pmb, int stage);
  TaskStatus ReceiveFFTGravityBoundary(MeshBlock *pmb, int stage);
  TaskStatus SendFFTGravityShear(MeshBlock *pmb, int stage);
  TaskStatus ReceiveFFTGravityShear(MeshBlock *pmb, int stage);
  TaskStatus SetFFTGravityBoundary(MeshBlock *pmb, int stage);
  TaskStatus PhysicalBoundary(MeshBlock *pmb, int stage);

 private:
  bool ORBITAL_ADVECTION; // flag for orbital advection (true w/ , false w/o)
  bool SHEAR_PERIODIC; // flag for shear periodic boundary (true w/ , false w/o)
  Real sbeta[2], ebeta[2];

  void AddTask(const TaskID& id, const TaskID& dep) override;
  void StartupTaskList(MeshBlock *pmb, int stage) override;
};


//----------------------------------------------------------------------------------------
//! 64-bit integers with "1" in different bit positions used to ID  each hydro task.
namespace FFTGravitySolverTaskNames {
const TaskID NONE(0);
const TaskID CLEAR_GRAV(1);

const TaskID SEND_GRAV_BND(2);
const TaskID RECV_GRAV_BND(3);
const TaskID SETB_GRAV_BND(4);
const TaskID GRAV_PHYS_BND(5);

const TaskID SEND_GRAV_SH(6);
const TaskID RECV_GRAV_SH(7);
} // namespace FFTGravitySolverTaskNames
#endif // TASK_LIST_FFT_GRAV_TASK_LIST_HPP_
