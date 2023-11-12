//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file pertB.cpp
//! \brief Problem generator creating perturbed initial B-field

// C headers

// C++ headers
#include <cmath>
#include <ctime>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fft/perturbation.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
#ifndef FFT
  std::stringstream msg;
  msg << "### FATAL ERROR this problem requires FFT" << std::endl;
  ATHENA_ERROR(msg);
  return;
#endif
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        phydro->u(IDN, k, j, i) = 1.0;

        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i) = 1.0;
        }
      }
    }
  }
}

//========================================================================================
//! \fn void Mesh::PostInitialize(int res_flag, ParameterInput *pin)
//! \brief create perturbed vector potential and calculate face field B
//========================================================================================
void Mesh::PostInitialize(int res_flag, ParameterInput *pin) {
  PerturbationGenerator *ppert;
  ppert = new PerturbationGenerator(this, pin);

  ppert->GenerateVector(); // generate vector potental in Fourier space
  ppert->AssignVector();   // do backward FFT and assign the vec array
  ppert->SetBoundary();    // fill in the ghost zones

  // In the problem generator, I only set E with internal energy assuming a
  // static medium Therefore, I expect to have Emag = Eint(gamma-1)/beta and the
  // normalization factor B0 = sqrt((gamma-1)/beta/sum(b))
  Real beta = pin->GetOrAddReal("problem", "beta", 1);
  Real Emag_tot = 0;

  for (int nb = 0; nb < nblocal; ++nb) {
    MeshBlock *pmb = my_blocks(nb);
    Hydro *phydro = pmb->phydro;
    Field *pfield = pmb->pfield;
    AthenaArray<Real> vecpot(ppert->GetVector(nb)), Ax, Ay, Az;
    Ax.InitWithShallowSlice(vecpot, 4, 0, 1);
    Ay.InitWithShallowSlice(vecpot, 4, 1, 1);
    Az.InitWithShallowSlice(vecpot, 4, 2, 1);
    // Get B from curl A
    // Bx
    for (int k = pmb->ks; k <= pmb->ke; k++) {
      for (int j = pmb->js; j <= pmb->je; j++) {
        for (int i = pmb->is; i <= pmb->ie + 1; i++) {
          Real dy(pmb->pcoord->dx2f(j)), dz(pmb->pcoord->dx3f(k));
          Real &Bx = pfield->b.x1f(k, j, i);
          Bx = 0.25 / dy *
                   ((Az(k, j + 1, i) - Az(k, j - 1, i)) +
                    (Az(k, j + 1, i - 1) - Az(k, j - 1, i - 1))) -
               0.25 / dz *
                   ((Ay(k + 1, j, i) - Ay(k - 1, j, i)) +
                    (Ay(k + 1, j, i - 1) - Ay(k - 1, j, i - 1)));
        }
      }
    }

    // By
    for (int k = pmb->ks; k <= pmb->ke; k++) {
      for (int j = pmb->js; j <= pmb->je + 1; j++) {
        for (int i = pmb->is; i <= pmb->ie; i++) {
          Real dx(pmb->pcoord->dx1f(i)), dz(pmb->pcoord->dx3f(k));
          Real &By = pfield->b.x2f(k, j, i);
          By = 0.25 / dz *
                   ((Ax(k + 1, j, i) - Ax(k - 1, j, i)) +
                    (Ax(k + 1, j - 1, i) - Ax(k - 1, j - 1, i))) -
               0.25 / dx *
                   ((Az(k, j, i + 1) - Az(k, j, i - 1)) +
                    (Az(k, j - 1, i + 1) - Az(k, j - 1, i - 1)));
        }
      }
    }

    // Bz
    for (int k = pmb->ks; k <= pmb->ke + 1; k++) {
      for (int j = pmb->js; j <= pmb->je; j++) {
        for (int i = pmb->is; i <= pmb->ie; i++) {
          Real dx(pmb->pcoord->dx1f(i)), dy(pmb->pcoord->dx2f(j));
          Real &Bz = pfield->b.x3f(k, j, i);
          Bz = 0.25 / dx *
                   ((Ay(k, j, i + 1) - Ay(k, j, i - 1)) +
                    (Ay(k - 1, j, i + 1) - Ay(k - 1, j, i - 1))) -
               0.25 / dy *
                   ((Ax(k, j + 1, i) - Ax(k, j - 1, i)) +
                    (Ax(k - 1, j + 1, i) - Ax(k - 1, j - 1, i)));
        }
      }
    }

    // calculate un-normalized magnetic energy
    for (int k = pmb->ks; k <= pmb->ke; k++) {
      for (int j = pmb->js; j <= pmb->je; j++) {
        for (int i = pmb->is; i <= pmb->ie; i++) {
          Emag_tot += 0.5 * (SQR(0.5 * (pfield->b.x1f(k, j, i) +
                                        pfield->b.x1f(k, j, i + 1))) +
                             SQR(0.5 * (pfield->b.x2f(k, j, i) +
                                        pfield->b.x2f(k, j + 1, i))) +
                             SQR(0.5 * (pfield->b.x3f(k, j, i) +
                                        pfield->b.x3f(k + 1, j, i))));
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  int mpierr;
  // Sum the perturbations over all processors
  mpierr = MPI_Allreduce(MPI_IN_PLACE, &Emag_tot, 1, MPI_DOUBLE, MPI_SUM,
                         MPI_COMM_WORLD);
#endif
  Emag_tot /= GetTotalCells();
  for (int i = 0; i < nblocal; ++i) {
    MeshBlock *pmb = my_blocks(i);
    Hydro *phydro = pmb->phydro;
    Field *pfield = pmb->pfield;

    Real Pth, B0; // nomalization factor
    // find thermal pressure
    if (NON_BAROTROPIC_EOS) {
      Pth = (pmb->peos->GetGamma() - 1); // I've set E = 1 in ProblemGenerator
    } else {
      Pth = SQR(
          pmb->peos->GetIsoSoundSpeed()); // I've set d = 1 in ProblemGenerator
    }
    B0 = std::sqrt(Pth / beta / Emag_tot);

    // normalize face fields
    for (int k = pmb->ks; k <= pmb->ke; k++) {
      for (int j = pmb->js; j <= pmb->je; j++) {
        for (int i = pmb->is; i <= pmb->ie; i++) {
          pfield->b.x1f(k, j, i) *= B0;
          pfield->b.x2f(k, j, i) *= B0;
          pfield->b.x3f(k, j, i) *= B0;
          if (i == pmb->ie)
            pfield->b.x1f(k, j, i + 1) *= B0;
          if (j == pmb->je)
            pfield->b.x2f(k, j + 1, i) *= B0;
          if (k == pmb->ke)
            pfield->b.x3f(k + 1, j, i) *= B0;
        }
      }
    }

    // add magnetic energy to total energy
    if (NON_BAROTROPIC_EOS) {
      for (int k = pmb->ks; k <= pmb->ke; k++) {
        for (int j = pmb->js; j <= pmb->je; j++) {
          for (int i = pmb->is; i <= pmb->ie; i++) {
            Real Emag = 0.5 * (SQR(0.5 * (pfield->b.x1f(k, j, i) +
                                          pfield->b.x1f(k, j, i + 1))) +
                               SQR(0.5 * (pfield->b.x2f(k, j, i) +
                                          pfield->b.x2f(k, j + 1, i))) +
                               SQR(0.5 * (pfield->b.x3f(k, j, i) +
                                          pfield->b.x3f(k + 1, j, i))));
            phydro->u(IEN, k, j, i) += Emag;
          }
        }
      }
    }

    // check divB
    Real max_divB = TINY_NUMBER;
    for (int k = pmb->ks; k <= pmb->ke; k++) {
      for (int j = pmb->js; j <= pmb->je; j++) {
        for (int i = pmb->is; i <= pmb->ie; i++) {
          Real dx(pmb->pcoord->dx1f(i)), dy(pmb->pcoord->dx2f(j)),
              dz(pmb->pcoord->dx3f(k));
          Real divB =
              (pfield->b.x1f(k, j, i + 1) - pfield->b.x1f(k, j, i)) / dx;
          divB += (pfield->b.x2f(k, j + 1, i) - pfield->b.x2f(k, j, i)) / dy;
          divB += (pfield->b.x3f(k + 1, j, i) - pfield->b.x3f(k, j, i)) / dz;
          max_divB = std::max(max_divB, divB);
        }
      }
    }
    std::cout << "magnetic fields are initialized with Emag "
              << Emag_tot * SQR(B0) << " and max divB=" << max_divB
              << std::endl;
  }

  delete ppert;

  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {}
