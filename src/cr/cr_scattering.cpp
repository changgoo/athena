//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cr_scattering.cpp
//! \brief Calculate scattering coefficient
//======================================================================================


// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/units.hpp"
#include "cr.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

Real Get_IonFraction(Real Temp, Real rho, Real ecr, Real ZetaNorm);
Real Get_mui(Real Temp, Real mu, Real xi);
Real Get_NeutralDensity(Real Temp, Real rho, Real rhoi, Real mui, Real muH);

void DefaultTemperature(Units *punit, Real rho, Real Press,
                        Real &Temp, Real &mu, Real &muH) {
  mu = 1.27; //assuming neutral gas
  muH = 1.4272;
  Temp = Press/rho*mu*punit->Temperature;
}

Real CosmicRay::Get_SigmaParallel(Real rho, Real Press, Real ecr, Real grad_pc_par) {
  Real sigma_par;

  if (self_consistent_flag==0) {
    sigma_par = sigma;
  } else {
    //Self-conistent calculation of sigma_par
    Real nu = 3.e-9; //frequency of collisions between ions and neutrals
    Real nu_in_code = nu*(1./punit->second);
    Real cr_kinenergy = 1e9 * 1.6021773e-12; //1 GeV in erg
    Real kinenergy_in_code = cr_kinenergy * punit->erg;

    Real mu, muH, Temp;
    UpdateTemperature(punit, rho, Press, Temp, mu, muH);

    // ion fraction; the value of the CR energy density in cgs
    // is required to calculate the CR ionization rate
    Real ecr_in_cgs = ecr*punit->EnergyDensity;
    if (ecr < TINY_NUMBER) ecr_in_cgs = TINY_NUMBER*punit->EnergyDensity;
    Real xion = Get_IonFraction(Temp, rho, ecr_in_cgs, ion_rate_norm);

    // ion mass-density
    Real ni = xion * rho;
    Real mui = Get_mui(Temp, mu, xion);
    Real rho_ion = ni * mui/muH;

    //neutral number-density
    Real nn = Get_NeutralDensity(Temp, rho, rho_ion, mui, muH);

    // ion thermal velocity, assumed to be equal to the sound speed
    Real vi = std::sqrt(1.67*Press/rho);

    Real sigma_in = 0.1 * 3. * PI / 16. * grad_pc_par * punit->e_in_code
                    / punit->c_in_code / kinenergy_in_code /
                    std::sqrt(rho_ion) * 2. / (nu_in_code * nn);
    Real sigma_nll = std::sqrt (0.1 * 3. * PI / 16. * grad_pc_par *
                     punit->e_in_code/punit->c_in_code / kinenergy_in_code
                     / std::sqrt(rho_ion) / (0.3 * vi * punit->c_in_code));
    sigma_par = std::min(sigma_nll,sigma_in) * vmax;
  }

  return sigma_par;
}

Real CosmicRay::Get_IonDensity(Real rho, Real Press, Real ecr) {
  Real rho_ion;
  Real mu, muH, Temp;
  UpdateTemperature(punit, rho, Press, Temp, mu, muH);

  // ion fraction; the value of the CR energy density in cgs is
  // required to calculate the CR ionization rate
  Real ecr_in_cgs = ecr*punit->EnergyDensity;
  if (ecr < TINY_NUMBER) ecr_in_cgs = TINY_NUMBER*punit->EnergyDensity;
  Real xion = Get_IonFraction(Temp, rho, ecr, ion_rate_norm);

  Real ni = xion * rho;
  Real mui = Get_mui(Temp, mu, xion);
  rho_ion = ni * mui/muH;
  if (rho_ion < rho)
    rho_ion = 0.999;

  return rho_ion;
}

Real Get_IonFraction(Real Temp, Real rho, Real ecr, Real ion_rate_norm) {
  Real xi;
  int i,j;

  Real alpharr = 1.42e-12;
  Real alphagr = 2.83e-14;
  Real xM = 1.68e-4;
  Real IonRate; //primary ionization rate for an H atom

  IonRate = ecr * ion_rate_norm;
  IonRate *= 1.5; //total ionization rate for an H atom

  Real beta = IonRate/rho/alpharr;
  Real gammaCR = alphagr/alpharr;

  if (Temp < 2e4)
    xi = 0.5*(std::sqrt(std::pow(beta+gammaCR+xM,2)
                  + 4.*beta)-(beta+gammaCR+xM)) + xM;
  else
    xi = 1.15;

  return xi;
}

Real Get_mui(Real Temp, Real mu, Real xi) {
  Real mui, mu_cold;
  Real mui_max = 1.236;
  Real xM = 1.68e-4;
  Real xH = xi - xM;

  if (Temp<=2e4) {
    mu_cold = (xH + 12.*xM)/xi;
    mui = std::max(mui_max,mu_cold);
  } else {
    mui = 2*mu;
  }
  return mui;
}

Real Get_NeutralDensity(Real Temp, Real rho, Real rhoi, Real mui, Real muH) {
  Real mun;
  if (Temp > 2e4) {
    mun = 0;
  } else {
    if (Temp <= 100)
      mun = 2.;
    else
      mun = 1.;
  }
  Real nn = muH*(rho - rhoi)/(mun + mui);
  return nn;
}
