#ifndef BVALS_PM_HPP_
#define BVALS_PM_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_pm.hpp
//! \brief

// C headers

// C++ headers

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../bvals_cc.hpp"

//----------------------------------------------------------------------------------------
//! \class CellCenteredBoundaryVariable
//! \brief

class ParticleMeshBoundaryVariable : public CellCenteredBoundaryVariable {
 public:
  ParticleMeshBoundaryVariable(MeshBlock *pmb,
                        AthenaArray<Real> *var, AthenaArray<Real> *coarse_var);

  virtual ~ParticleMeshBoundaryVariable() = default;

  AthenaArray<Real> empty_flux[3];
  AthenaArray<Real> var_buf;

  void SendShearingBoxBoundaryBuffers() override;
  void SetShearingBoxBoundaryBuffers() override;

 private:
  void SetBoundarySameLevel(Real *buf, const NeighborBlock& nb) override;
  int LoadBoundaryBufferSameLevel(Real *buf, const NeighborBlock& nb) override;
};

#endif // BVALS_PM_HPP_
