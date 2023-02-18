//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_accretion.cpp
//! \brief Problem generator to sink particle accretion

// C headers

// C++ headers
#include <sstream>
#include <boost/numeric/odeint.hpp>

// Athena++ headers
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../particles/particles.hpp"

typedef std::vector<Real> state_type;

// Self-similar ODE; Equations (11)-(12) in Shu (1977)
// y[0] = alpha (dimensionless density), y[1] = v (dimensionless velocity)
void shu77(const state_type &y, state_type &dydx, const Real x) {
  Real denom = SQR(x-y[1]) - 1.;
  dydx[0] = y[0]*(y[0] - (2./x)*(x-y[1]))*(x-y[1])/denom;
  dydx[1] = (y[0]*(x-y[1]) - 2./x)*(x-y[1])/denom;
}

// Asymptotic series for large x; Equation (19) in Shu (1977)
void set_shu77_ic(state_type &y, const Real xi0, const Real A) {
  y[0] = A/SQR(xi0) - A*(A-2.)/(2*pow(xi0, 4));
  y[1] = -(A-2.)/xi0 - (1.-A/6.)*(A-2.)/pow(xi0, 3);
}

Real SimilarityVar(Real r, Real t, Real cs) {
  Real xi = r / (cs*t);
  return xi;
}

Real rctrl; // radius of the control volume
Real dctrl; // density in the control volume

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (SELF_GRAVITY_ENABLED) {
    Real gconst = pin->GetReal("self_gravity","gconst");
    SetGravitationalConstant(gconst);
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  const Real rmax = 0.75*pmy_mesh->mesh_size.x1max;
  const Real t0 = pin->GetOrAddReal("problem","t0",0.43);
  const Real A = pin->GetOrAddReal("problem","A",2.0004);
  //TODO retire this input parameter; implement sink creation
  rctrl = pin->GetOrAddReal("problem","rctrl",1.5)*pcoord->dx1f(0);
  const Real cs = peos->GetIsoSoundSpeed();
  Real vadv = pin->GetOrAddReal("problem","vadv",0)*cs;
  const Real density_scale = 1.0/pgrav->four_pi_G/SQR(t0);
  const Real velocity_scale = cs;
  const Real xi0 = 10; // initial location to integrate shu77 ODE
  const Real step = -0.0001; // initial step size for adaptive ODE integrator
  Real xi;
  state_type res(2);

  // TODO retire this; instead, initialize density all the way down to the center
  // and reset the control volume using sink creation method
  // Set dctrl = self-similar solution at r=rctrl
  set_shu77_ic(res, xi0, A);
  xi = SimilarityVar(rctrl, t0, cs);
  boost::numeric::odeint::integrate(shu77, res, xi0, xi, step);
  dctrl = density_scale*res[0];

  for (int k=ks; k<=ke; k++) {
    Real z = pcoord->x3v(k);
    for (int j=js; j<=je; j++) {
      Real y = pcoord->x2v(j);
        for (int i=is; i<=ie; i++) {
        Real x = pcoord->x1v(i);
        Real R = std::sqrt(SQR(x) + SQR(y));
        Real r = std::sqrt(SQR(R) + SQR(z));
        Real sinth, costh, sinph, cosph;
        // calculate trigonometric functions and avoid singularity
        if (r==0) {
          sinth = 0;
          costh = 0;
        } else {
          sinth = R/r;
          costh = z/r;
        }
        if (R==0) {
          sinph = 0;
          cosph = 0;
        } else {
          sinph = y/R;
          cosph = x/R;
        }
        r = std::min(r, rmax); // pressure confinement - constant beyond the cloud radius
        if (r <= rctrl) {
          phydro->u(IDN,k,j,i) = dctrl;
          phydro->u(IM1,k,j,i) = dctrl*vadv;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
        } else {
          set_shu77_ic(res, xi0, A);
          xi = SimilarityVar(r, t0, cs);
          boost::numeric::odeint::integrate(shu77, res, xi0, xi, step);
          Real rho = density_scale*res[0];
          Real vr = velocity_scale*res[1];
          phydro->u(IDN,k,j,i) = rho;
          phydro->u(IM1,k,j,i) = rho*(vr*sinth*cosph + vadv);
          phydro->u(IM2,k,j,i) = rho*(vr*sinth*sinph);
          phydro->u(IM3,k,j,i) = rho*(vr*costh);
        }
        if (NON_BAROTROPIC_EOS) {
          std::stringstream msg;
          msg << "### FATAL ERROR in function [MeshBlock::ProblemGenerator]" << std::endl
              << "Only isothermal EOS is allowed. " << std::endl;
          ATHENA_ERROR(msg);
        }
      }
    }
  }

  // initialize particle
  // Particle mass is set to the mass contained inside r=r_ctrl of the Shu77
  // similarity solution (see Gong & Ostriker 2013)
  if (pmy_mesh->particle) {
    if (ppars[0]->partype != "sink") {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [MeshBlock::ProblemGenerator]" << std::endl
          << "Only sink particle is allowed. " << std::endl;
      ATHENA_ERROR(msg);
    }
    SinkParticles *ppar = dynamic_cast<SinkParticles*>(ppars[0]);

    // Create a sink particle
    set_shu77_ic(res, xi0, A);
    xi = SimilarityVar(rctrl, t0, cs);
    boost::numeric::odeint::integrate(shu77, res, xi0, xi, step);
    Real mstar = SQR(xi)*res[0]*(xi - res[1]); // Eq. (10) in Shu (1977)
    mstar *= std::pow(cs,3)*t0/pgrav->gconst; // Eq. (8) in Shu (1977)
    ppar->AddOneParticle(mstar,0,0,0,vadv,0,0);
    ppar->ToggleParHstOutFlag();
  }
}


//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Output particle history
//========================================================================================
void Mesh::UserWorkInLoop() {
  for (int b = 0; b < nblocal; ++b) {
    MeshBlock *pmb(my_blocks(b));
    for (Particles *ppar : pmb->ppars) ppar->OutputParticles(false);
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop()
//  \brief
//========================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop()
//! \brief
//========================================================================================

void MeshBlock::UserWorkInLoop() {
  if (pmy_mesh->particle) {
    // TODO move this to Integrate() stage 2
    SinkParticles *ppar = dynamic_cast<SinkParticles*>(ppars[0]);
    ppar->AccreteMass();

    // Temporary sink accretion.
    // To be replace with ppar->AccreteMass()
    // Think about where to call AccreteMass().
    // Currently, Integrate() is inside the time integrator task list.
    // We may put AccreteMass there. In any case, mass accretion must be done
    // after hydro update (see Kim & Ostriker 2017)
//    SinkParticles *ppar = dynamic_cast<SinkParticles*>(ppars[0]);
//    if (ppar->GetNumPar() == 0) return;
//    for (int k=ks; k<=ke; k++) {
//      Real z = pcoord->x3v(k);
//      for (int j=js; j<=je; j++) {
//        Real y = pcoord->x2v(j);
//        for (int i=is; i<=ie; i++) {
//          Real x = pcoord->x1v(i);
//          Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
//          if (r <= rctrl) {
//            ppar->mass(0) += (phydro->u(IDN,k,j,i) - dctrl)*pcoord->GetCellVolume(k,j,i);
//            phydro->u(IDN,k,j,i) = dctrl;
//          }
//        }
//      }
//    }
  }
}
