#ifndef CR_CR_HPP_
#define CR_CR_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cr.hpp
//! \brief definitions for CosmicRay class
//========================================================================================

#include <string>

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/cc/bvals_cc.hpp"
#include "../utils/cooling_function.hpp"
#include "../utils/units.hpp"

class MeshBlock;
class ParameterInput;
class CRIntegrator;

//! \class CosmicRay
//! \brief CosmicRay data and functions

// Array indices for  moments
enum {CRE=0, CRF1=1, CRF2=2, CRF3=3};

class CosmicRay {
  friend class CRIntegrator;
  friend class BoundaryValues;
 public:
  CosmicRay(MeshBlock *pmb, ParameterInput *pin);
  ~CosmicRay();

  AthenaArray<Real> u_cr, u_cr1, u_cr2; //cosmic ray energy density and flux

  AthenaArray<Real> coarse_cr_;

  //diffusion coefficients for both normal diffusion term, and advection term
  AthenaArray<Real> sigma_diff, sigma_adv;

  AthenaArray<Real> v_adv; // streaming velocity
  AthenaArray<Real> v_diff; // the diffuion velocity, need to calculate the flux

  AthenaArray<Real> CRInjectionRate; //rate of injection CR energy density

  int refinement_idx{-1};

  AthenaArray<Real> flux[3]; // store transport flux, also need for refinement

  //Flags
  int stream_flag; // to include streaming or not
  int src_flag; // to update the gas energy and momentum equations
  int losses_flag; //to include losses of CR energy and momentum due
                   //to collisional interaction with the ambient gas
  int perp_diff_flag; //to include the presence of diffusion in
                      //the direction perpendicular to the magnetic field
  int self_consistent_flag; //to enable the self-consistent calculation of sigma

  //Input parameters
  Real vmax; // the maximum velocity (effective speed of light)
  Real sigma; //scattering coefficient
  Real max_opacity;
  Real lambdac; //rate of hadronic or ionizing losses per unit volume
  Real perp_to_par_diff; //ratio between the scattering coefficient in the direction
                         //perpendicular to the magntic field and the scattering
                         //coefficient in the direction parallel to the magntic field
  Real ion_rate_norm; //ionization rate normalization

  Real DensityUnit, LengthUnit, VelocityUnit;

  MeshBlock* pmy_block; // ptr to MeshBlock containing this Fluid
  CellCenteredBoundaryVariable cr_bvar;

  CRIntegrator *pcrintegrator;
  Units *punit;

  //Function in problem generators to update opacity
  void EnrollOpacityFunction(CROpacityFunc MyOpacityFunction);

  void EnrollTemperatureFunction(CRTemperatureFunc MyTemperatureFunction);

  void EnrollUserCRSource(CRSrcTermFunc my_func);

  bool cr_source_defined;

  CROpacityFunc UpdateOpacity;

  CRTemperatureFunc UpdateTemperature;

  //Function to calculate the scattering coefficient in the
  //direction parallel to the magnetic field
  Real Get_SigmaParallel(Real rho, Real Press, Real ecr, Real grad_pc_par);
  //Function to calculate the ion Alfvèn speed
  Real Get_IonDensity(Real rho, Real Press, Real ecr);

  AthenaArray<Real> cwidth;
  AthenaArray<Real> cwidth1;
  AthenaArray<Real> cwidth2;
  AthenaArray<Real> b_grad_pc; // array to store B\dot Grad Pc
  AthenaArray<Real> b_angle; //sin\theta,cos\theta,sin\phi,cos\phi of B direction

 private:
  CRSrcTermFunc UserSourceTerm_;
};

//default function to calculate the gas temperature
void DefaultTemperature(Units *punit, Real rho, Real Press,
     Real &Temp, Real &mu, Real &muH);

#endif // CR_CR_HPP_
