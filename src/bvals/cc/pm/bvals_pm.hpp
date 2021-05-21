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
#include "../../../particles/particle_mesh.hpp"

//----------------------------------------------------------------------------------------
//! \class CellCenteredBoundaryVariable
//! \brief

class ParticleMeshBoundaryVariable : public CellCenteredBoundaryVariable {
 public:
  ParticleMeshBoundaryVariable(MeshBlock *pmb, AthenaArray<Real> *var,
                               AthenaArray<Real> *coarse_var,
                               ParticleMesh *ppm);

  virtual ~ParticleMeshBoundaryVariable() = default;

  AthenaArray<Real> empty_flux[3];
  AthenaArray<Real> var_buf;

  void SendShearingBoxBoundaryBuffers() override;
  void SetShearingBoxBoundaryBuffers() override;



 private:
  void SetBoundarySameLevel(Real *buf, const NeighborBlock& nb) override;
  int LoadBoundaryBufferSameLevel(Real *buf, const NeighborBlock& nb) override;

  void ShearQuantities(AthenaArray<Real> &shear_cc_, bool upper) override;
  void LoadShearingBoxBoundarySameLevel(AthenaArray<Real> &src,
                                      Real *buf, int nb) override;

  ParticleMesh *ppm_;
};

#endif // BVALS_PM_HPP_
