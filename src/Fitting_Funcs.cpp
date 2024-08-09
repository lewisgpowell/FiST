// Functions to calculate DOS and tunneling spectra from several theoretical models
// Copyright (C) 2024 Lewis Powell
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along with this program.
// If not, see https://www.gnu.org/licenses/. 

#include "pch.h"
#include "Fitting_Funcs.h"
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include "gsl_integral.h"
#include "convolution.h"
#include "newton_raphson.h"

static double dyne_func_eval(const double E, const std::array<double, 3>& params)
{
	const double N{ params[0] };
	const double D{ params[1] };
	const double L{ params[2] };
	std::complex<double> comp_E{ E, -L };
	return abs((comp_E / sqrt(comp_E * comp_E - D * D)).real()) * N;
}

static double dyne_func_skew(const double E, const std::array<double, 4>& params)
{
	const double D{ params[0] };
	const double L{ params[1] };
	const double Na{ params[2] };
	const double Nb{ params[3] };
	std::complex<double> comp_E{ E, -L };
	return abs((comp_E / sqrt(comp_E * comp_E - D * D)).real()) * (Na + Nb*E);
}

double dyne_func(const double E, const double N, const double D, const double L)
{
	std::array<double, 3> params{ N, D, L };
	return dyne_func_eval(E, params);
}

static double dos_point_node_eval(const double E, const std::array<double, 3>& params)
{
	const double N{ params[0] };
	const double D{ params[1] };
	const double L{ params[2] };
	std::complex<double> comp_E{ E, -L };
	return (E / D) * std::log(std::abs((1.0 + comp_E / D) / (1.0 - comp_E / D))) * N/2;
}

double dos_point_node(const double E, const double N, const double D, const double L)
{
	std::array<double, 3> params{ N, D, L };
	return dos_point_node_eval(E, params);
}

static double dos_line_node_eval(const double E, const std::array<double, 3>& params)
{
	const double N{ params[0] };
	const double D{ params[1] };
	const double L{ params[2] };
	std::complex<double> comp_E{ E, -L };
	std::complex<double> ratio{ comp_E / D };
	std::complex<double> prefactor{ N * ratio };
	std::complex<double> result{ prefactor * std::asin(1.0 / ratio) };
	return abs(result.real());
}

static double dos_combined_eval(const double E, const std::array<double, 4>& params)
{
	const double N{ params[0] };
	const double Ds{ params[1] };
	const double Dp{ params[2] };
	const double L{ params[3] };
	std::complex<double> comp_E{ E, -L };
	std::complex<double> prefactor{ N * comp_E / (Dp*2) };
	std::complex<double> result{ prefactor * (std::asin((Dp + Ds)/comp_E) - std::asin((Ds - Dp) / comp_E)) };
	return abs(result.real());
}

double dos_line_node(const double E, const double N, const double D, const double L) {
	std::array<double, 3> params{ N, D, L };
	return dos_line_node_eval(E, params);
}

double dos_combined(const double E, const double N, const double Ds, const double Dp, const double L) {
	std::array<double, 4> params{ N, Ds, Dp, L };
	return dos_combined_eval(E, params);
}

double fermi_dirac_deriv(const double E, const double T)
{
	const double k_b{ 8.617333262145E-2 };
	const double exponent{ exp(E / (k_b * T)) };
	if (exponent > 600) // Avoid overflow for large exponent
		return 0;
	const double result{ -exponent / (k_b * T * (exponent + 1) * (exponent + 1)) };
	return result;
}

void dyne_fit(double* voltages, const std::size_t num_points, const double N, const double D, const double L, const double T)
{
	const std::array<double, 3> dos_params{ N, D, L };
	convolution(dyne_func_eval, voltages, num_points, T, dos_params);
}

void dyne_skew_fit(double* voltages, const std::size_t num_points, const double Na, const double Nb, const double D, const double L, const double T)
{
	const std::array<double, 4> dos_params{ D, L , Na, Nb};
	convolution(dyne_func_skew, voltages, num_points, T, dos_params);
}

void point_node_fit(double* voltages, const std::size_t num_points, const double N, const double D, const double L, const double T)
{
	const std::array<double, 3> dos_params{ N, D, L };
	convolution(dos_point_node_eval, voltages, num_points, T, dos_params);
}

void line_node_fit(double* voltages, const std::size_t num_points, const double N, const double D, const double L, const double T)
{
	const std::array<double, 3> dos_params{ N, D, L };
	convolution(dos_line_node_eval, voltages, num_points, T, dos_params);
}

void combined_fit(double* voltages, const std::size_t num_points, const double N, const double Ds, const double Dp, const double L, const double T)
{
	const std::array<double, 4> dos_params{ N, Ds, Dp, L };
	convolution(dos_combined_eval, voltages, num_points, T, dos_params);
}

static double btk_eval(const double E, const std::array<double, 4>& params)
{
	// Unpack parameters
	const double N{ params[0] };
	const double D{ params[1] };
	const double L{ params[2] };
	const double Z{ params[3] };
	const std::complex<double> comp_E{ E, -L }; // Use Dyne's gamma parameter to add width to the peak
	std::complex<double> A{}; std::complex<double> B{}; // A = probability of Andreev reflextion, B of ordinary reflection
	if (std::abs(E) < std::abs(D)) {
		const double z{ 1 + 2 * Z * Z };
		A = D * D / (comp_E * comp_E + (D * D - comp_E * comp_E) * z * z);
		B = 1.0 - A;
	}
	else {
		// Compute intermediate values
		const std::complex<double> u2{ (1.0 + std::sqrt((comp_E * comp_E - D * D) / (comp_E * comp_E))) / 2.0 };
		const std::complex<double> v2{ 1.0 - u2 };
		const std::complex<double> diff{ u2 - v2 };
		const std::complex<double> g{ u2 + Z * Z * diff }; // lower case gamma in paper
		const std::complex<double> g2{ g * g };
		
		A = u2 * v2 / g2;
		B = diff* diff * Z * Z * (1 + Z * Z) / g2;
	}
	return (1.0 + Z*Z)*(1.0 + A - B).real() * N;
}

void btk_fit(double* voltages, const std::size_t num_points, const double N, const double D, const double L, const double Z, const double T)
{
	const std::array<double, 4> dos_params{ N, D, L, Z };
	convolution(btk_eval, voltages, num_points, T, dos_params);
}

static double gap_temp_func(const double D, void* p) {
	struct gap_func_params* params = (struct gap_func_params*)p;
	const double D0{ params->D0 };
	const double T{ params->T };
	const double T0{ params->T0 };
	const double t{ T / T0 };
	const double d{ D / D0 };
	return tanh(d/t) - d;
}

double gap_at_temp(const double T, const double D0) {
	const double T0{ D0 / (1.764 * 8.617333262145e-2) };
	if (T >= T0 - 1e-3)
		return 1e-3;

	// Define function to find roots of
	gsl_function gap_func{};
	struct gap_func_params params{D0, T, T0};

	gap_func.function = &gap_temp_func;
	gap_func.params = &params;

	// Set up root solver
	int status{};
	int iter{}; int max_iter{ 1000 };
	const int workspace_size{ 10000 };
	gsl_root_fsolver* solver{ gsl_root_fsolver_alloc(gsl_root_fsolver_brent) };
	gsl_root_fsolver_set(solver, &gap_func, 1e-4, 1);

	// Get root by iterating
	double D{}; double D_prev{};
	do {
		iter++;
		status = gsl_root_fsolver_iterate(solver);
		D_prev = D;
		D = gsl_root_fsolver_root(solver);
		status = gsl_root_test_delta(D, D_prev, 0, 1e-3);
	} while (status == GSL_CONTINUE && iter < max_iter);

	gsl_root_fsolver_free(solver);

	return D;
}

static std::complex <double> maki_equation(const std::complex<double>& u, std::array<double, 2>& params)
{
	const double A{ params[0] };
	const double xi{ params[1] };
	return A + xi * u / std::sqrt(1.0 - u * u) - u;
}

static std::complex <double> maki_equation_deriv(const std::complex<double>& u, std::array<double, 2>& params)
{
	const double xi{ params[1] };
	std::complex<double> F{ 1.0 - u * u };
	return xi * u * u / (F * std::sqrt(F)) + xi / F - 1.0;
}

static std::complex <double> maki_equation_root(const double E, const double D, const double B, const double xi, const bool is_spin_up) {
	const double mu = 5.7883818012e-2; // Bohr magneton in meV/T
	// Set the sign of the magnetic field depending on the spin
	int spin_sign{1};
	if (is_spin_up)
		spin_sign = -1;
	const double u0{ (E + spin_sign * B * mu) / D }; // value without pair breaking
	std::complex<double> u{u0, xi};

	if (xi == 0) return u;

	std::array<double, 2> params{};
	params[0] = u0;
	params[1] = xi;

	return solve_complex_newton_raphson(
		[&params](const std::complex<double> z) {return maki_equation(z, params); },
		[&params](const std::complex<double> z) {return maki_equation_deriv(z, params); },
		u);

}

void maki_equation_ctype(double* result, const double E, const double D, const double B, const double xi, const bool is_spin_up) {
	const std::complex<double> complex_result{ maki_equation_root(E, D, B, xi, is_spin_up) };
	result[0] = complex_result.real();
	result[1] = complex_result.imag();
}

static double maki_dos_eval(const double E, const std::array<double, 4>& params) {
	//const double D, const double B, const double Z) {
	const double N{ params[0] };
	const double D{ params[1] };
	const double B{ params[2] };
	const double xi{ params[3] };
	const std::complex<double> u_plus{ maki_equation_root(E, D, B, xi, true) };
	const std::complex<double> u_minus{ maki_equation_root(E, D, B, xi, false) };
	double E_sign{ 1.0 };
	if (E < 0) E_sign = -1.0;
	// Calculate density of states for each spin species
	std::complex<double> sqrt_argument1{ u_plus * u_plus - 1.0 };
	std::complex<double> sqrt_argument2{ u_minus * u_minus - 1.0 };
	// Check phases of sqrt arguments to avoid problem if values are on either side of branch cut [-inf, 0]
	const double phase_difference{ std::arg(sqrt_argument2) - std::arg(sqrt_argument1) };
	const std::complex<double> sqrt1{ u_plus * u_plus - 1.0 };
	const std::complex<double> sqrt2{ u_minus * u_minus - 1.0 };
	int sign_flip{ 1 };
	if (std::abs(phase_difference) > M_PI)
		sign_flip = -1;
	const double rho_up{ E_sign * (u_plus / std::sqrt(sqrt_argument1)).real() };
	const double rho_down{ E_sign * (u_minus / std::sqrt(sqrt_argument2)).real() };
	const double result{ N * std::abs(rho_up + rho_down * sign_flip) / 2 };
	if (result < 1e-3) return 1e-3; // Prevent integration error from value being too small
	return result;
}

double maki_dos(const double E, const double N, const double D, const double xi, const double B) {
	const std::array<double, 4> dos_params{ N, D, B, xi };
	return maki_dos_eval(E, dos_params);
}

void maki_fit(double* voltages, const size_t num_points, const double N, const double D, const double xi, const double B, const double T) {
	const std::array<double, 4> dos_params{ N, D, B, xi };
	convolution(maki_dos_eval, voltages, num_points, T, dos_params);
}