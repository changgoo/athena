//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file particle_jeans.cpp
//! \brief Jeans instability tests on particles.

// C++ standard libraries
#include <cmath>    // round()
#include <sstream>  // stringstream
#include <random>     // mt19937, normal_distribution, uniform_real_distribution

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../particles/particles.hpp"

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (SELF_GRAVITY_ENABLED) {
    Real njeans = pin->GetReal("problem","njeans");
    Real lambda = mesh_size.x1len;
    if (mesh_size.nx2 > 1) lambda = std::min(lambda,mesh_size.x2len);
    if (mesh_size.nx3 > 1) lambda = std::min(lambda,mesh_size.x3len);

    Real d0 = 1.0;
    Real dpar0 = d0 * pin->GetOrAddReal("problem", "dtog", 1.0);
    Real cs2;

    if (NON_BAROTROPIC_EOS) {
      Real gam = pin->GetReal("hydro","gamma");
      Real p0 = 1.0;
      cs2 = gam * p0 / d0;
    } else {
      Real iso_cs = pin->GetReal("hydro","iso_sound_speed");
      cs2 = SQR(iso_cs);
    }

    Real gconst = cs2*PI*njeans*njeans/(d0*lambda*lambda);

    SetGravitationalConstant(gconst);
    Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
    SetGravityThreshold(eps);
    SetMeanDensity(d0 + dpar0);

    if (Globals::my_rank==0) {
      // moved print statements here from MeshBlock::ProblemGenerator
      std::cout << "four_pi_G " << gconst*4.0*PI << std::endl;
      std::cout << "lambda " << lambda << std::endl;
    }
  }
  return;
}
//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Sets the initial conditions.
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Sanity check.
  if (!PARTICLES) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [MeshBlock::ProblemGenerator]" << std::endl
        << "Particles were not enabled at configuration. " << std::endl;
    ATHENA_ERROR(msg);
  }

  // Medium is static
  Real ux0, uy0, uz0;
  ux0 = pin->GetOrAddReal("problem", "ux0", 0.0);
  uy0 = pin->GetOrAddReal("problem", "uy0", 0.0);
  uz0 = pin->GetOrAddReal("problem", "uz0", 0.0);

  // Set a uniform, static medium.
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->u(IDN,k,j,i) = 1.0;
        phydro->u(IM1,k,j,i) = ux0;
        phydro->u(IM2,k,j,i) = uy0;
        phydro->u(IM3,k,j,i) = uz0;
      }
    }
  }
  if (NON_BAROTROPIC_EOS) {
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i)
          phydro->u(IEN,k,j,i) = 1.0 / (peos->GetGamma() - 1.0);
  }

  // Get the dust-to-gas ratio and the velocity of the particles.
  Real dtog, vpx0, vpy0, vpz0, vamp;
  dtog = pin->GetOrAddReal("problem", "dtog", 1.0);
  vpx0 = pin->GetOrAddReal("problem", "vpx0", 0.0);
  vpy0 = pin->GetOrAddReal("problem", "vpy0", 0.0);
  vpz0 = pin->GetOrAddReal("problem", "vpz0", 0.0);
  vamp = pin->GetOrAddReal("problem", "vamp", 0.0);

  // Find the total number of particles in each direction.
  RegionSize& mesh_size = pmy_mesh->mesh_size;
  Real np_per_cell = pin->GetOrAddReal("problem", "np_per_cell",1);
  int npx1 = (block_size.nx1 > 1) ? static_cast<int>(mesh_size.nx1*np_per_cell) : 1;
  int npx2 = (block_size.nx2 > 1) ? static_cast<int>(mesh_size.nx2*np_per_cell) : 1;
  int npx3 = (block_size.nx3 > 1) ? static_cast<int>(mesh_size.nx3*np_per_cell) : 1;

  // Uniformly separted particles, uniform mass = total mass / N particles
  // Find the mass of each particle and the distance between adjacent particles.
  Real vol = mesh_size.x1len * mesh_size.x2len * mesh_size.x3len;
  Real dx1 = mesh_size.x1len / npx1,
       dx2 = mesh_size.x2len / npx2,
       dx3 = mesh_size.x3len / npx3;
  DustParticles::SetOneParticleMass(dtog * vol / (npx1 * npx2 * npx3));

  // Determine number of particles in the block.
  int npx1_loc = static_cast<int>(std::round(block_size.x1len / dx1)),
      npx2_loc = static_cast<int>(std::round(block_size.x2len / dx2)),
      npx3_loc = static_cast<int>(std::round(block_size.x3len / dx3));
  int npar = ppar->npar = npx1_loc * npx2_loc * npx3_loc;
  if (npar > ppar->nparmax)
    ppar->UpdateCapacity(npar);

  // Assign the particles.
  // Ramdomizing position. Or velocity perturbation
  std::random_device device;
  std::mt19937_64 rng_generator;
  // std::int64_t rseed = static_cast<std::int64_t>(device());
  std::int64_t rseed = 1;
  std::uniform_real_distribution<Real> udist(0.0,1.0); // uniform in [0,1)
  rng_generator.seed(rseed);

  // Real ph = udist(rng_generator)*TWO_PI;
  int ipar = 0;
  for (int k = 0; k < npx3_loc; ++k) {
    Real zp1 = block_size.x3min + (k + 0.5) * dx3;
    for (int j = 0; j < npx2_loc; ++j) {
      Real yp1 = block_size.x2min + (j + 0.5) * dx2;
      for (int i = 0; i < npx1_loc; ++i) {
        Real xp1 = block_size.x1min + (i + 0.5) * dx1;
        ppar->xp(ipar) = xp1 + dx1 * (udist(rng_generator) - 0.5);
        ppar->yp(ipar) = yp1 + dx2 * (udist(rng_generator) - 0.5);
        ppar->zp(ipar) = zp1;
        if (mesh_size.nx3 > 1)
          ppar->zp(ipar) += dx3 * (udist(rng_generator) - 0.5);

        ppar->vpx(ipar) = vpx0 + vamp * (udist(rng_generator) - 0.5);
        ppar->vpy(ipar) = vpy0 + vamp * (udist(rng_generator) - 0.5);
        ppar->vpz(ipar) = vpz0;
        if (mesh_size.nx3 > 1)
          ppar->vpz(ipar) += vamp * (udist(rng_generator) - 0.5);
        ++ipar;
      }
    }
  }

  // Initialize the stopping time.
  if (DustParticles::GetVariableTaus()) {
    Real taus0 = DustParticles::GetStoppingTime();
    for (int k = 0; k < npar; ++k)
      ppar->taus(k) = taus0;
  }
}
