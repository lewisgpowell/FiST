// Functions to calculate DOS and tunneling spectra from several theoretical models
// Copyright (C) 2024 Lewis Powell
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along with this program.
// If not, see https://www.gnu.org/licenses/. 
#pragma once
#include <complex>
#include <array>
#include <functional>

#ifdef FITTING_FUNCS_EXPORTS
#define FITTING_FUNCS_API __declspec(dllexport)
#else
#define FITTING_FUNCS_API __declspec(dllimport)
#endif

// Dyne's fitting model for superconductor density of states with lifetime broadening
// E = quasiparticle energy; D = superconductor gap; L = lifetime width
static double dyne_func_eval(const double E, const std::array<double, 3>& params);
extern "C" FITTING_FUNCS_API double dyne_func(const double E, const double N, const double D, const double L);

static double dyne_func_skew(const double E, const std::array<double, 4>& params);

// Derivative of Fermi-Dirac distribution
// T = temperature
extern "C" FITTING_FUNCS_API double fermi_dirac_deriv(const double E, const double T);

// General convolution of density of states with the derivative of the Fermi-Dirac distribution
//static double convolution(const std::function<double(double, const std::vector<double>&)>& density_of_states, const double V, const double T, const std::vector<double>& density_of_states_params);

// Dyne's fit with thermal and lifetime broadening
// N = normalisation
extern "C" FITTING_FUNCS_API void dyne_fit(double*voltages, const size_t num_points, const double N, const double D, const double L, const double T);

// Version of Dyne fit with asymmetry
// Na = constant factor, Nb = asymmetric factor
extern "C" FITTING_FUNCS_API void dyne_skew_fit(double* voltages, const size_t num_points, const double Na, const double Nb, const double D, const double L, const double T);

// Fit for gap dependence on temperature
struct gap_func_params { double D0; double T; double T0; };
static double gap_temp_func(const double D, void* p);
extern "C" FITTING_FUNCS_API double gap_at_temp(const double T, const double D0);

// Density of states and fit for gap with point nodes
static double dos_point_node_eval(const double E, const std::array<double, 3>& params);
extern "C" FITTING_FUNCS_API double dos_point_node(const double E, const double N, const double D, const double L);
extern "C" FITTING_FUNCS_API void point_node_fit(double* voltages, const size_t num_points, const double N, const double D, const double L, const double T);

// Density of states and fit for gap with line nodes
static double dos_line_node_eval(const double E, const std::array<double, 3>& params);
extern "C" FITTING_FUNCS_API double dos_line_node(const double E, const double N, const double D, const double L);
extern "C" FITTING_FUNCS_API void line_node_fit(double* voltages, const size_t num_points, const double N, const double D, const double L, const double T);

// Model for two gaps (nodeless and line nodes)
static double dos_combined_eval(const double E, const std::array<double, 4>& params);
extern "C" FITTING_FUNCS_API double dos_combined(const double E, const double N, const double Ds, const double Dp, const double L);
extern "C" FITTING_FUNCS_API void combined_fit(double* voltages, const size_t num_points, const double N, const double Ds, const double Dp, const double L, const double T);

// BTK model
static double btk_eval(const double E, const std::array<double, 4>& params);
extern "C" FITTING_FUNCS_API void btk_fit(double* voltages, const size_t num_points, const double N, const double D, const double L, const double Z, const double T);

// Maki equation for tunneling in a field
// B = field, xi = orbital depairing parameter
static std::complex <double> maki_equation(const std::complex<double>& u, std::array<double, 2>& params);
static std::complex <double> maki_equation_deriv(const std::complex<double>& u, std::array<double, 2>& params);
static std::complex <double> maki_equation_root(const double E, const double D, const double B, const double xi, const bool is_spin_up);
extern "C" FITTING_FUNCS_API void maki_equation_ctype(double* results , const double E, const double D, const double B, const double xi, const bool is_spin_up); // Allow calling from python
static double maki_dos_eval(const double E, const std::array<double, 4>& params);
extern "C" FITTING_FUNCS_API double maki_dos(const double E, const double N, const double D, const double xi, const double B);
extern "C" FITTING_FUNCS_API void maki_fit(double* voltages, const size_t num_points, const double N, const double D, const double xi, const double B, const double T);