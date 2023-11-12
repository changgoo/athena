//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file cooling_function.cpp
//! \brief prototypes of various cooling functions

// C headers

// C++ headers
#include <fstream>  // ofstream
#include <iostream> // cout, endl
#include <sstream>  // stringstream

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../globals.hpp"
#include "../parameter_input.hpp" // ParameterInput
#include "cooling.hpp"
#include "units.hpp"

//========================================================================================
//! \fn CoolingFunctionBase::CoolingFunctionBase(ParameterInput *pin)
//! \brief ctor of the base class for cooling function
//! \note Read parameters from "cooling" block in the input file
//========================================================================================
CoolingFunctionBase::CoolingFunctionBase(ParameterInput *pin, Units *punit)
    : punit(punit), T_max(pin->GetOrAddReal("cooling", "T_max", 1.e9)),
      T_floor(pin->GetOrAddReal("cooling", "T_floor", 10)),
      gamma_adi(pin->GetReal("hydro", "gamma")), coolftn_name("base"), mu(1.27),
      muH(1.4) {}

//========================================================================================
//! \fn void CoolingFunctionBase::Initialize(Real muH)
//! \brief initialize units and some conveinent conversion factors
//========================================================================================
void CoolingFunctionBase::Initialize(Real muH) {
  mean_mass_per_H = muH * Constants::mH;
  code_den_to_nH = punit->Density / mean_mass_per_H;
  code_press_to_pok = punit->Pressure / Constants::kB;
  nH_to_code_den = 1 / code_den_to_nH;
  pok_to_code_press = 1 / code_press_to_pok;
}

//========================================================================================
//! \fn static Real CoolingTime(const Real rho, const Real press)
//! \brief tcool = e / (n^2*Cool - n*heat)
//! \note
//! - input rho and P are in code units
//! - output tcool is in code units
//========================================================================================
Real CoolingFunctionBase::CoolingTime(const Real rho, const Real press) {
  Real nH = rho * code_den_to_nH;
  Real cool = nH * nH * Lambda_T(rho, press);
  Real heat = nH * Gamma_T(rho, press);
  Real eint = press * punit->Pressure / (gamma_adi - 1);
  Real tcool = eint / (cool - heat);
  return tcool / punit->Time;
}

//========================================================================================
//! \fn static Real NetCoolingTime(const Real rho, const Real press)
//! \brief tcool = e / (|n^2*Cool| + |n*heat|)
//! \note
//! - input rho and P are in code units
//! - output tcool is in code units
//========================================================================================
Real CoolingFunctionBase::NetCoolingTime(const Real rho, const Real press) {
  Real nH = rho * code_den_to_nH;
  Real cool = nH * nH * Lambda_T(rho, press);
  Real heat = nH * Gamma_T(rho, press);
  Real eint = press * punit->Pressure / (gamma_adi - 1);
  Real tcool = eint / (std::abs(cool) + std::abs(heat));
  return tcool / punit->Time;
}

//========================================================================================
//! \fn void PrintCoolingFunction()
//! \brief private function to check cooling and heating functions
//========================================================================================
void CoolingFunctionBase::PrintCoolingFunction() {
  Real pok = 3.e3;
  std::string coolfilename(coolftn_name);
  coolfilename.append("_coolftn.txt");
  std::ofstream coolfile(coolfilename.c_str());
  coolfile << "#rho,Press,Temp,cool,heat,tcool"
           << "\n";

  for (int i = 0; i < 1000; ++i) {
    Real logn =
        5.0 * ((static_cast<Real>(i) / 500.) - 1.0) - 2; // logn = -7 ~ 3
    Real rho = std::pow(10, logn);
    Real press = pok * pok_to_code_press;
    Real temp = GetTemperature(rho, press);
    Real cool = Lambda_T(rho, press);
    Real heat = Gamma_T(rho, press);
    Real t_cool = CoolingTime(rho, press);
    coolfile << rho << "," << press << "," << temp << "," << cool << "," << heat
             << "," << t_cool << "\n";
  }
}

//========================================================================================
//! \fn PiecewiseLinearFits::PiecewiseLinearFits(ParameterInput *pin)
//! \brief Cooling function with Piecewise Linear Fits used in El-Badry et al.
//! 2019
//!        provided by Drummond Fielding
//! \note
//! - constant PE heating is applied at T<T_PE (Gamma and T_PE must set in
//! input)
//! - mu = 0.62 (fixed), muH = 1.4
//! - not very good for low-T cooling (T<T_PE)
//========================================================================================
PiecewiseLinearFits::PiecewiseLinearFits(ParameterInput *pin, Units *punit)
    : CoolingFunctionBase(pin, punit), coolftn_name("plf"),
      T_PE(pin->GetReal(
          "cooling", "T_PE")), // temperature below which PE heating is applied
      Gamma0(pin->GetReal("cooling", "Gamma")), // heating rate in ergs / sec
      mu(0.62), muH(1.4) {
  Initialize(muH);
}

//========================================================================================
//! \fn Real PiecewiseLinearFits::Lambda_T(const Real rho, chost Real Press)
//! \brief piecewise linear fit for cooling
//!
//! - input rho, Press in code units
//! - return Lambda in erg cm^3 / s
//========================================================================================
Real PiecewiseLinearFits::Lambda_T(const Real rho, const Real Press) {
  Real T = GetTemperature(rho, Press);
  int k, n = nfit_cool - 1;
  // first find the temperature bin
  for (k = n; k >= 0; k--) {
    if (T >= T_cooling_curve[k])
      break;
  }
  Real cool;
  if (T > T_cooling_curve[0]) {
    cool = (lambda_cooling_curve[k] *
            std::pow(T / T_cooling_curve[k], exponent_cooling_curve[k]));
  } else {
    cool = 1.0e-30;
  }
  return cool;
}

//========================================================================================
//! \fn Real PiecewiseLinearFits::dlnL_dlnT(const Real rho, chost Real Press)
//! \brief logarithmic derivative of cooling functions Lambda_T(T)
//!
//! - input rho, Press in code units
//! - return d ln(Lambda)/ d ln(T)
//! - In the PLF cooling function this simply returns the tabulated exponents
//========================================================================================
Real PiecewiseLinearFits::dlnL_dlnT(const Real rho, const Real Press) {
  Real T = GetTemperature(rho, Press);
  int k, n = nfit_cool - 1;
  // first find the temperature bin
  for (k = n; k >= 0; k--) {
    if (T >= T_cooling_curve[k])
      break;
  }
  Real dcool;
  if (T > T_cooling_curve[0]) {
    dcool = exponent_cooling_curve[k];
  } else {
    dcool = 0.0;
  }
  return dcool;
}

//========================================================================================
//! \fn Real PiecewiseLinearFits::Gamma_T(const Real rho, const Real Press)
//! \brief constant heating for T<T_PE
//!
//! - input rho, Press in code units
//! - return Gamma in erg / s
//========================================================================================
Real PiecewiseLinearFits::Gamma_T(const Real rho, const Real Press) {
  Real T = GetTemperature(rho, Press);
  Real heat = T < T_PE ? Gamma0 : 0;
  return heat;
}

//========================================================================================
//! \fn Real PiecewiseLinearFits::GetTemperature(const Real rho, const Real
//! Press) \brief conversion between (rho, P) in code --> T in K
//!
//! - input rho, Press in code units
//! - return T in K
//! - a constant conversion factor is applied since mu is fixed
//========================================================================================
Real PiecewiseLinearFits::GetTemperature(const Real rho, const Real Press) {
  return Press / rho * punit->Temperature_mu * mu;
}

//========================================================================================
//! \fn TigressClassic::TigressClassic(ParameterInput *pin)
//! \brief Cooling function with tables for
//!        Koyama & Inutsuka (2002) + Sutherland and Dopita (1993) used in
//!        TIGRESS classic
//! \note
//! - mu is a function of Temperature, but for unit definition mu = 1 is used
//! - conversion from (rho, P) in code --> T in K is non-trivial; using
//! tabulated
//!   relation between T_1 --> T, where T_1 =
//!   (P*Punit)/(rho*rhounit)*(1.0*m_H/k_B) or T_1 = P/rho*punit->Temperature as
//!   mu = 1 is used to set Units classs
//! - muH = 1.4271, mu = T/T_1
//! - allow time-dependent, spatially varying heating (SetHeatRatio to change
//! heat_ratio)
//========================================================================================
TigressClassic::TigressClassic(ParameterInput *pin, Units *punit)
    : CoolingFunctionBase(pin, punit), coolftn_name("tigress"),
      heat_ratio(pin->GetReal("cooling", "heat_ratio")), muH(1.4271) {
  Initialize(muH);
}

//========================================================================================
//! \fn Real TigressClassic::Lambda_T(const Real rho, const Real Press)
//! \brief interpolate cooling table for KI02+SD93
//!
//! - input rho, Press in code units
//! - return Lambda in erg cm^3 / s
//========================================================================================
Real TigressClassic::Lambda_T(const Real rho, const Real Press) {
  Real T1 = Press / rho * punit->Temperature_mu;
  int T1idx = get_Tidx(T1);
  Real dTemp = (T1 - T1_tbl[T1idx]) / (T1_tbl[T1idx + 1] - T1_tbl[T1idx]);
  Real cool =
      cool_table[T1idx] + (cool_table[T1idx + 1] - cool_table[T1idx]) * dTemp;
  cool = T1 < Tmin_tbl ? 0.0 : cool;
  return cool;
}

//========================================================================================
//! \fn Real TigressClassic::dlnL_dlnT(const Real rho, const Real Press)
//! \brief give logarthmic derivative of cooling
//!
//! - input rho, Press in code units
//! - return d ln(Lambda)/ d ln(T)
//! - In the TigressClassic cooling function this returns the instantaneous
//!   derivative in according to the tabulated values of the cooling function
//========================================================================================
Real TigressClassic::dlnL_dlnT(const Real rho, const Real Press) {
  Real T1 = Press / rho * punit->Temperature_mu;
  int T1idx = get_Tidx(T1);
  Real dLdT = (cool_table[T1idx + 1] - cool_table[T1idx]) /
              (T1_tbl[T1idx + 1] - T1_tbl[T1idx]);
  Real dlnLdlnT = dLdT * T1 / (cool_table[T1idx] + dLdT * (T1 - T1_tbl[T1idx]));
  return dlnLdlnT;
}

//========================================================================================
//! \fn Real TigressClassic::Gamma_T(const Real rho, const Real Press)
//! \brief interpolate heating table (constant and smooth drop at high T)
//!
//! - input rho, Press in code units
//! - return Gamma in erg / s
//! - scaled by heat_ratio
//========================================================================================
Real TigressClassic::Gamma_T(const Real rho, const Real Press) {
  Real T1 = Press / rho * punit->Temperature_mu;
  int T1idx = get_Tidx(T1);
  Real dTemp = (T1 - T1_tbl[T1idx]) / (T1_tbl[T1idx + 1] - T1_tbl[T1idx]);
  Real heat =
      heat_table[T1idx] + (heat_table[T1idx + 1] - heat_table[T1idx]) * dTemp;
  return heat * heat_ratio;
}

//========================================================================================
//! \fn Real TigressClassic::GetTemperature(const Real rho, const Real Press)
//! \brief conversion between (rho, P) in code --> T in K
//!
//! - input rho, Press in code units
//! - return T in K
//! - use pre-tabulated relation for T1 --> T, where T1=P/rho*mH/k_B
//========================================================================================
Real TigressClassic::GetTemperature(const Real rho, const Real Press) {
  Real T1 = Press / rho * punit->Temperature_mu;
  int T1idx = get_Tidx(T1);
  Real T1i = T1_tbl[T1idx];
  Real T1ip1 = T1_tbl[T1idx + 1];
  Real Ti = temp_tbl[T1idx];
  Real Tip1 = temp_tbl[T1idx + 1];
  Real Tnew = Ti + (Tip1 - Ti) * (T1 - T1i) / (T1ip1 - T1i);

  return Tnew;
}

//========================================================================================
//! \fn Real TigressClassic::Get_mu(const Real rho, const Real Press)
//! \brief return mu for given (rho, P)
//!
//! - input rho, Press in code units
//! - return mu = T/T1, T from GetTemperature function
//========================================================================================
Real TigressClassic::Get_mu(const Real rho, const Real Press) {
  Real T1 = Press / rho * punit->Temperature_mu;
  Real Temp = GetTemperature(rho, Press);

  return Temp / T1;
}

//========================================================================================
//! \fn int TigressClassic::get_Tidx(const Real T1)
//! \brief return table index for given T1
//! \note
//! - here, Tmin and Tmax are not T_floor and T_ceil but min/max of tables
//! - cooling/heating will be extrapolated beyond this range
//========================================================================================
int TigressClassic::get_Tidx(const Real T1) {
  int idx, ireturn;

  idx = static_cast<int>(std::log10(T1 / Tmin_tbl) / dlnT_tbl);
  ireturn = T1 < Tmin_tbl ? 0 : idx;
  ireturn = T1 > Tmax_tbl ? NTBL - 2 : ireturn;

  return ireturn;
}
